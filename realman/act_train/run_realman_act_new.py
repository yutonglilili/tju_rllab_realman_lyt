#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    DEFAULT_DEPLOY_CONFIG_NAME,
    load_json,
    render_status_preview,
    resize_rgb_frame,
    wait_for_robot_state,
)

MODEL_TRANSLATION_SCALE_M = 0.001


def parse_args() -> argparse.Namespace:
    """解析命令行参数，确定模型 bundle、设备、控制频率和运行模式等配置。"""
    parser = argparse.ArgumentParser(
        description="Run a trained ACT policy online on the Realman arm from a deployment bundle."
    )
    parser.add_argument("--bundle", required=True, help="Deployment bundle directory created by prepare_realman_deploy.py.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu or mps.")
    parser.add_argument("--robot-ip", default=None, help="Override the bundled robot IP.")
    parser.add_argument("--camera-serial", default=None, help="Override the bundled camera serial.")
    parser.add_argument("--task", default=None, help="Override the bundled task string.")
    parser.add_argument("--control-fps", type=float, default=None, help="Override the bundled control FPS.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum policy steps per running episode.")
    parser.add_argument("--warmup-seconds", type=float, default=1.0, help="Camera warmup time before starting.")
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the model translation output before safety clamping.",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the model rotation output before safety clamping.",
    )
    parser.add_argument(
        "--timing-log-freq",
        type=int,
        default=0,
        help="Print timing stats every N executed steps. 0 disables timing logs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run perception and policy inference without moving the robot.")
    parser.add_argument("--auto-start", action="store_true", help="Start policy execution immediately.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV preview window.")
    return parser.parse_args()


def resolve_control_fps(
    args_control_fps: float | None,
    deploy_config: dict,
    model_dir: Path,
) -> float:
    """
    决定推理控制频率。

    优先级如下：
    1. 用户命令行显式传入的 --control-fps
    2. 若模型目录里有 train_run.json，则优先采用训练数据的 fps
    3. 否则退回 deploy_config.json 里打包时写入的 control_fps

    这样做是为了尽量让部署时的时序节奏和训练时保持一致。
    """
    if args_control_fps is not None:
        return float(args_control_fps)

    deploy_fps = float(deploy_config["control_fps"])
    train_run_path = model_dir / "train_run.json"
    if not train_run_path.exists():
        return deploy_fps

    try:
        train_run = load_json(train_run_path)
        train_fps = float(train_run["fps"])
    except Exception:
        return deploy_fps

    if abs(train_fps - deploy_fps) > 1e-6:
        print(
            "Bundled control_fps differs from the training dataset fps; "
            f"using training fps {train_fps:.3f} instead of bundled {deploy_fps:.3f}."
        )
    return train_fps


def clamp_action(
    action: np.ndarray,
    max_translation_mm: float,
    max_rotation_rad: float,
) -> np.ndarray:
    """
    对模型输出动作做安全裁剪。

    - 前 3 维是平移增量，按“毫米量级动作”裁剪
    - 中间 3 维是旋转增量，按弧度裁剪
    - 最后 1 维是夹爪开合，限制在 0~1
    """
    clamped = np.asarray(action, dtype=np.float32).copy()
    clamped[:3] = np.clip(clamped[:3], -max_translation_mm, max_translation_mm)
    clamped[3:6] = np.clip(clamped[3:6], -max_rotation_rad, max_rotation_rad)
    clamped[6] = np.clip(clamped[6], 0.0, 1.0)
    return clamped


def scale_action(
    action: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    """
    按用户给定倍率放大模型动作。

    这个放大发生在安全裁剪之前，因此最终真正执行的动作仍然会被
    max_delta_translation_mm / max_delta_rotation_rad 保护住。
    """
    scaled = np.asarray(action, dtype=np.float32).copy()
    scaled[:3] *= float(translation_scale)
    scaled[3:6] *= float(rotation_scale)
    return scaled


def build_runtime_state(robot_state, state_source: str, max_gripper_width: float) -> np.ndarray:
    """
    把 RealmanEnv 返回的机器人状态整理成模型可直接使用的 state 向量。

    参数:
    - robot_state: 环境返回的 RobotState
    - state_source: 选择用 pose 还是 joint 作为状态主体
    - max_gripper_width: 用于把真实夹爪宽度归一化到 0~1

    返回:
    - float32 的一维向量
      - state_source='pose'  -> [x, y, z, roll, pitch, yaw, gripper]
      - state_source='joint' -> [joint1...joint7, gripper]
    """
    if state_source == "pose":
        base = np.asarray(robot_state.pose, dtype=np.float32)
    elif state_source == "joint":
        base = np.asarray(robot_state.joint, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported state_source: {state_source}")
    gripper = np.clip(float(robot_state.gripper) / max(max_gripper_width, 1e-6), 0.0, 1.0)
    return np.concatenate([base, np.array([gripper], dtype=np.float32)], axis=0).astype(np.float32)


def model_action_to_env_step(
    action: np.ndarray,
    max_gripper_width: float,
) -> tuple[np.ndarray, float]:
    """
    把模型输出动作转换成 RealmanEnv.step(...) 需要的执行格式。

    背景:
    - 采集脚本里，遥操作平移量是先以“毫米量级动作”表达
    - 真正发给机器人前，会乘 0.001 变成“米”
    - 因此模型学到的平移输出也沿用了这套语义

    这里做两件事:
    1. 把模型输出的前三维平移增量从“毫米量级动作”换成 env.step 所需的米制增量
    2. 把最后一维 0~1 的夹爪值换成真实夹爪宽度（单位: 米）

    返回:
    - delta_pose: 6 维增量位姿，可直接传给 env.step({"delta_pose": ...})
    - gripper_width: 真实夹爪目标宽度（米）
    """
    action = np.asarray(action, dtype=np.float32)
    delta_pose = action[:6].copy()
    delta_pose[:3] *= MODEL_TRANSLATION_SCALE_M
    gripper_width = float(np.clip(action[6], 0.0, 1.0) * max_gripper_width)
    return delta_pose, gripper_width


def should_enable_preview(no_preview: bool) -> tuple[bool, str | None]:
    """
    判断是否启用 OpenCV 预览窗口。

    返回:
    - bool: 是否启用预览
    - str|None: 若禁用，则返回原因，方便在控制台打印说明
    """
    if no_preview:
        return False, "Preview disabled by --no-preview."

    if os.name != "nt":
        display = os.environ.get("DISPLAY")
        wayland_display = os.environ.get("WAYLAND_DISPLAY")
        if not display and not wayland_display:
            return False, "No DISPLAY/WAYLAND_DISPLAY detected; disabling OpenCV preview."

    return True, None


def main() -> None:
    """
    推理主入口。

    整体流程是一个同步闭环：
    1. 读取 bundle 配置和训练好的 ACT 模型
    2. 连接 Realman 机械臂环境与 RealSense 相机
    3. 每一轮循环:
       - 读取当前机器人状态
       - 读取一帧 RGB 图像
       - 组织成模型输入
       - 模型输出一组“末端增量 + 夹爪”
       - 调用 env.step(...) 同步执行
       - 再进入下一轮
    """
    args = parse_args()

    # 读取部署 bundle 中的运行配置。
    bundle_dir = Path(args.bundle).resolve()
    deploy_config = load_json(bundle_dir / DEFAULT_DEPLOY_CONFIG_NAME)

    # 命令行参数优先于 bundle 默认值。
    robot_ip = args.robot_ip or deploy_config["robot_ip"]
    camera_serial = args.camera_serial or deploy_config["camera_serial"]
    task = args.task or deploy_config["task"]
    control_fps = resolve_control_fps(args.control_fps, deploy_config, model_dir=(bundle_dir / deploy_config["model_dir"]).resolve())
    state_source = deploy_config["state_source"]
    max_gripper_width = float(deploy_config["max_gripper_width"])
    max_delta_translation_mm = float(deploy_config["max_delta_translation_mm"])
    max_delta_rotation_rad = float(deploy_config["max_delta_rotation_rad"])
    model_dir = (bundle_dir / deploy_config["model_dir"]).resolve()

    # 这里按需导入，避免仅查看脚本时就要求所有运行依赖都已安装。
    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from open3d_realsense_env import Open3dRealsenseEnv
    from realman_env import RealmanEnv

    device = torch.device(args.device)

    # 加载训练好的 ACT 模型，以及与训练统计量一致的预处理/后处理器。
    model = ACTPolicy.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=str(model_dir))

    # 自动识别模型配置中的视觉输入和状态输入键名。
    # 这样脚本不强耦合于固定字段名，只要模型配置里是 1 个图像输入 + 1 个状态输入即可。
    visual_keys = [
        key
        for key, feature in model.config.input_features.items()
        if getattr(feature, "type", None) == FeatureType.VISUAL
    ]
    state_keys = [
        key
        for key, feature in model.config.input_features.items()
        if getattr(feature, "type", None) == FeatureType.STATE
    ]
    if len(visual_keys) != 1:
        raise ValueError(
            f"Expected exactly one visual input feature in the trained policy, found {visual_keys}."
        )
    if len(state_keys) != 1:
        raise ValueError(
            f"Expected exactly one state input feature in the trained policy, found {state_keys}."
        )

    image_key = visual_keys[0]
    state_key = state_keys[0]
    image_shape = tuple(model.config.input_features[image_key].shape)
    if len(image_shape) != 3:
        raise ValueError(f"Unexpected image feature shape: {image_shape}")
    image_shape_hwc = (image_shape[1], image_shape[2], image_shape[0])

    # 打印当前运行时配置，方便部署时确认 bundle 和模型都加载正确。
    print("=" * 72)
    print("Realman ACT runtime")
    print("=" * 72)
    print(f"bundle          : {bundle_dir}")
    print(f"model_dir       : {model_dir}")
    print(f"device          : {device}")
    print(f"robot_ip        : {robot_ip}")
    print(f"camera_serial   : {camera_serial}")
    print(f"task            : {task}")
    print(f"state_source    : {state_source}")
    print(f"control_fps     : {control_fps:.3f}")
    print(f"translation_scale : {args.translation_scale:.3f}")
    print(f"rotation_scale    : {args.rotation_scale:.3f}")
    print(f"image_key       : {image_key}")
    print(f"state_key       : {state_key}")
    print(f"image_shape_hwc : {image_shape_hwc}")
    print(f"dry_run         : {args.dry_run}")
    print("=" * 72)

    # 连接机器人环境和相机。
    # 这里强制使用 sync 模式，因为当前脚本设计就是：
    # “每一步感知 -> 模型推理 -> env.step() 执行 -> 再感知下一步”。
    env = RealmanEnv(robot_ip, mode="sync")
    camera = Open3dRealsenseEnv(camera_serial)
    print("STEP 1: after camera init")

    # running 表示当前是否真正开始执行模型输出。
    # episode_step / episode_index 主要用于交互显示和手动 reset。
    running = bool(args.auto_start)
    episode_step = 0
    episode_index = 0
    preview_enabled, preview_reason = should_enable_preview(args.no_preview)
    if preview_reason:
        print(preview_reason)
    if not preview_enabled and not running:
        running = True
        print("Preview is disabled, so the runtime will auto-start.")
    period = 1.0 / max(control_fps, 1e-6)

    if preview_enabled:
        import cv2

        print("Initializing OpenCV preview window...")
        cv2.namedWindow("Realman ACT", cv2.WINDOW_NORMAL)
    else:
        cv2 = None

    try:
        print(f"Warming up camera for {args.warmup_seconds:.1f}s...")
        time.sleep(max(args.warmup_seconds, 0.0))
        print("STEP 2: before first robot state read")

        # 等待第一次有效机器人状态，避免刚启动时 state 还未准备好。
        state = wait_for_robot_state(env)

        # 每次新开一段 rollout 前，都重置一次模型内部历史状态。
        model.reset()

        while True:
            loop_start = time.perf_counter()

            # 先读取当前机器人状态。
            # 这一步是为了把“上一时刻执行后的真实状态”作为本轮模型输入的一部分。
            state = env.get_state()
            if state is None:
                print("Robot state unavailable, waiting for the next valid state...")
                time.sleep(0.05)
                continue

            # 再读取当前相机 RGB 图像，并 resize 到训练时的输入尺寸。
            camera_obs = camera.step()
            rgb = np.asarray(camera_obs["rgb"], dtype=np.uint8)
            rgb_model = resize_rgb_frame(rgb, image_shape_hwc)

            # 处理预览窗口里的键盘交互。
            if preview_enabled:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            # Q: 退出程序
            if key in (ord("q"), ord("Q")):
                print("Quit requested.")
                break

            # Enter: 结束当前 rollout，并把模型状态清空，准备下一段 episode
            if key in (13, 10):
                running = False
                episode_step = 0
                episode_index += 1
                model.reset()
                print(f"Episode reset -> {episode_index}")

            # Space: 在开始/暂停之间切换，同时重置模型内部动作历史
            if key == ord(" "):
                running = not running
                episode_step = 0
                model.reset()
                status = "running" if running else "paused"
                print(f"Policy toggled -> {status}")

            # H: 仅重置策略历史，不改变环境状态
            if key in (ord("h"), ord("H")):
                model.reset()
                episode_step = 0
                print("Policy history reset.")

            # 用“当前真实状态”构造模型输入里的 observation.state。
            state_vec = build_runtime_state(
                robot_state=state,
                state_source=state_source,
                max_gripper_width=max_gripper_width,
            )

            action_np = None
            if running and episode_step < args.max_steps:
                inference_start = time.perf_counter()

                # 组织模型推理输入。
                # 一次输入包含：
                # - 当前 RGB 图像
                # - 当前机器人状态向量
                obs = prepare_observation_for_inference(
                    {
                        image_key: rgb_model.copy(),
                        state_key: state_vec.copy(),
                    },
                    device=device,
                    task=task,
                    robot_type=deploy_config["robot_type"],
                )

                # 先做与训练一致的预处理，再执行模型推理，最后做后处理还原动作量纲。
                obs = preprocess(obs)
                with torch.inference_mode():
                    action = model.select_action(obs)
                    action = postprocess(action)
                action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
                inference_elapsed = time.perf_counter() - inference_start

                # 先按用户给定倍率放大动作，再交给后续安全裁剪。
                action_np = scale_action(
                    action=action_np,
                    translation_scale=args.translation_scale,
                    rotation_scale=args.rotation_scale,
                )

                # 对模型输出动作做安全限制，避免异常值直接打到机械臂。
                action_np = clamp_action(
                    action=action_np,
                    max_translation_mm=max_delta_translation_mm,
                    max_rotation_rad=max_delta_rotation_rad,
                )

                # 把模型动作换成 env.step(...) 可直接执行的格式。
                pose_delta, gripper_width = model_action_to_env_step(
                    action=action_np,
                    max_gripper_width=max_gripper_width,
                )

                if not args.dry_run:
                    motion_start = time.perf_counter()

                    # 同步执行一步：
                    # 输入增量位姿 + 夹爪宽度，
                    # 机器人执行完成后再返回，进入下一轮感知。
                    state = env.step(
                        {
                            "delta_pose": pose_delta,
                            "motion": "pose",
                            "gripper": gripper_width,
                            "wait_gripper": False,
                        }
                    )
                    motion_elapsed = time.perf_counter() - motion_start
                else:
                    motion_elapsed = 0.0

                # 记录 rollout 内的步数，用于上限控制和界面显示。
                episode_step += 1
                if args.timing_log_freq > 0 and episode_step % args.timing_log_freq == 0:
                    loop_elapsed = time.perf_counter() - loop_start
                    print(
                        f"[timing] step={episode_step} "
                        f"infer={inference_elapsed*1000:.1f}ms "
                        f"motion={motion_elapsed*1000:.1f}ms "
                        f"loop={loop_elapsed*1000:.1f}ms "
                        f"hz={1.0 / max(loop_elapsed, 1e-6):.2f}"
                    )
                if episode_step >= args.max_steps:
                    running = False
                    model.reset()
                    print(f"Reached max steps for episode {episode_index}.")

            if preview_enabled:
                # 实时预览窗口里显示当前 rollout 状态和最近一次动作输出。
                lines = [
                    f"episode={episode_index} step={episode_step}/{args.max_steps}",
                    "space=start/pause enter=reset h=reset-policy q=quit",
                    f"task={task}",
                    f"state_source={state_source} dry_run={args.dry_run}",
                ]
                if action_np is not None:
                    lines.append(
                        "action="
                        + " ".join(f"{value:+.3f}" for value in action_np[:6])
                        + f" grip={action_np[6]:.3f}"
                    )
                else:
                    lines.append("action=paused")
                render_status_preview(rgb_model, lines, window_name="Realman ACT")

            # 按控制频率睡眠，形成比较稳定的主循环节奏。
            elapsed = time.perf_counter() - loop_start
            if period > elapsed:
                time.sleep(period - elapsed)
    finally:
        # 无论正常退出还是中途报错，都要释放硬件资源。
        env.close()
        camera.close()
        if preview_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
