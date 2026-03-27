import os
import sys
import time
import json
import argparse
import threading

import numpy as np
import cv2
import pyrealsense2 as rs
from typing import Tuple
from tqdm import tqdm
from termcolor import cprint

from scipy.spatial.transform import Rotation as R

# -----------------------------
# 路径兼容：导入 cuRobo 规划器与 RealmanEnv
# -----------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)  # realman_env.py 在仓库根目录

DYNAMIC_ALIGNMENT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "dynamic_alignment"))
sys.path.insert(0, DYNAMIC_ALIGNMENT_DIR)  # constrained_solver.py 在 dynamic_alignment/

from constrained_solver import solve_motion, init_curobo  # noqa: E402
from realman_env import RealmanEnv, T_from_realman_xyzrpy, T_TCP2REALMANEEF  # noqa: E402


# Franka 版本里用于写入 init.json 的 gains（当前脚本对 Realman 控制不直接使用，但保持数据格式一致）
K_GAINS = [400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
D_GAINS = [320.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]


def init_realsense(fps: int = 30):
    """启动 RealSense 管线，并返回 pipeline/profile。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
    profile = pipeline.start(config)
    return pipeline, profile


def _rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    SciPy 的 as_quat() 输出是 [x, y, z, w]。
    cuRobo Pose 的 quaternion 顺序是 [w, x, y, z]。
    """
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def pose_xyzrpy_to_ee_pose(
    pose_xyzrpy: np.ndarray,
    apply_tcp2eef: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 RealmanEnv 的 xyzrpy 位姿转换为（ee_translation, ee_quat_wxyz, ee_rotation_matrix）。

    apply_tcp2eef=True：把 TCP 坐标系下的位姿通过 T_TCP2REALMANEEF 映射到 Realman EEF 坐标系，
    再把 EEF 作为 cuRobo 的 ee_link 输入。
    """
    T_base_tcp = T_from_realman_xyzrpy(pose_xyzrpy)
    if apply_tcp2eef:
        T_base_ee = T_base_tcp @ T_TCP2REALMANEEF
    else:
        T_base_ee = T_base_tcp

    ee_translation = T_base_ee[:3, 3].astype(np.float32)
    ee_rotation_matrix = T_base_ee[:3, :3].astype(np.float32)
    ee_quat_wxyz = _rotation_matrix_to_quaternion_wxyz(ee_rotation_matrix)
    return ee_translation, ee_quat_wxyz, ee_rotation_matrix


def get_index(args) -> int:
    """沿用 Franka 版本的断点续采逻辑：从最后一个 traj_xxxxx + 1 开始。"""
    os.makedirs(args.save_dir, exist_ok=True)
    all_entries = os.listdir(args.save_dir)
    all_entries.sort()
    if len(all_entries) >= 1:
        cnt = int(all_entries[-1].split("_")[-1]) + 1
    else:
        cnt = 0
    return cnt


def generate_cmd(
    args,
    motion_gen,
    kin_model,
    joint_state_rad: np.ndarray,
    ee_translation: np.ndarray,
    z_proj: np.ndarray,
    ee_quaternion_wxyz: np.ndarray,
):
    """
    复刻 dynamic_alignment/pushing.py 的生成方式：
    在若干段目标末端平移上进行规划拼接，并抽取稀疏点形成离散关节下发序列。
    """
    trans_goals = []
    rot_goals = []

    X = 0.05
    for i in range(5):
        trans_goals.append(ee_translation + z_proj * X * i)
        rot_goals.append(ee_quaternion_wxyz)

    new_joint_state = joint_state_rad
    joints_traj = []

    for i in range(0, len(trans_goals) - 1):
        j_traj, _ee_traj, new_joint_state = solve_motion(
            args,
            new_joint_state,
            ee_translation_goal=trans_goals[i + 1],
            ee_orientation_goal=rot_goals[i + 1],
            motion_gen=motion_gen,
            kin_model=kin_model,
        )
        joints_traj += j_traj

    # 与 Franka 版本一致：稀疏化下发点
    joints_traj = joints_traj[::40]
    cprint(f"len of cmd: {len(joints_traj)}", "green")
    return joints_traj


def control_thread_realman(
    env: RealmanEnv,
    joints_traj_rad: list,
    init_time: float,
    dir_name: str,
    cmd_rate_hz: float,
    repeat_last_n: int = 3,
):
    """
    对 Realman 做“逐点关节下发 + 记录时间戳”的采集控制线程。
    """
    cmd_rate_hz = float(cmd_rate_hz)
    period = 1.0 / max(cmd_rate_hz, 1e-6)

    cmd_timestamps = []
    joints_cmd = list(joints_traj_rad)
    joints_cmd.extend([joints_traj_rad[-1]] * repeat_last_n)

    for i, cmd_rad in enumerate(joints_cmd):
        robot_timestamp = time.time() - init_time
        cmd_timestamps.append(
            {
                "id": i,
                "ros_timestamp": robot_timestamp,
                "cmd": cmd_rad,  # 保持与 Franka 版一致：记录关节角（这里是 rad）
            }
        )

        # Realman SDK 的 rm_movej_follow 注释/实现里一般以“度”为单位；realman_env 的 send_joint 也不自动转单位
        cmd_deg = np.degrees(np.array(cmd_rad, dtype=np.float32)).tolist()
        env.send_joint(np.array(cmd_deg, dtype=np.float32))

        time.sleep(period)

    # 给相机采集线程一点缓冲
    time.sleep(0.5)

    with open(os.path.join(dir_name, "control.json"), "w") as f:
        json.dump(cmd_timestamps, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Alignment Data Collection (Realman)")
    parser.add_argument(
        "--robot",
        type=str,
        default="dynamic_alignment_realman/realman.yml",
        help="cuRobo robot config yaml（需要你为 Realman 准备对应的 URDF/kinematics 配置）",
    )
    parser.add_argument("--debug", type=int, default=0, help="debug level")
    parser.add_argument("--len", type=int, default=200, help="total trajectories to record")
    parser.add_argument("--save_dir", type=str, default="records/realman-dynamic", help="output root dir")

    parser.add_argument("--fps", type=int, default=30, help="RealSense fps")
    parser.add_argument("--record_frames", type=int, default=90, help="number of frames to record per trajectory")
    parser.add_argument("--cmd_rate", type=float, default=20.0, help="关节下发频率（Hz），用于时序近似对齐")

    parser.add_argument("--robot_ip", type=str, default="192.168.101.19", help="Realman robot ip")
    parser.add_argument("--safety_mode", type=int, default=1, help="1=enable safety mode, 0=disable")
    parser.add_argument("--gripper_open_after_reset", type=float, default=0.0, help="reset 后夹爪开度(米宽)")

    parser.add_argument(
        "--apply_tcp2eef",
        type=int,
        default=1,
        help="1=把 Realman TCP 位姿映射到 EEF（使用 realman_env.T_TCP2REALMANEEF），更匹配 cuRobo ee_link",
    )

    args = parser.parse_args()

    if not os.path.exists(args.robot):
        raise FileNotFoundError(
            f"cuRobo robot config not found: {args.robot}\n"
            "你需要为 Realman 准备对应的 cuRobo yaml（包含 URDF/ee_link/base_link/joint_names 等）。"
        )

    # cuRobo 规划初始化
    motion_gen, kin_model = init_curobo(args)

    # RealSense 初始化
    pipeline, profile = init_realsense(args.fps)

    # Realman 初始化（使用官方 SDK + realman_env 封装）
    env = RealmanEnv(
        robot_ip=args.robot_ip,
        safety_mode=bool(args.safety_mode),
        async_mode=True,
        min_cmd_interval=1.0 / float(max(args.cmd_rate, 1.0)),
        control_mode="absolute",
    )

    apply_tcp2eef = bool(args.apply_tcp2eef)
    cnt = get_index(args)

    align = rs.align(rs.stream.color)

    try:
        for idx in range(cnt, args.len + 1):
            dir_name = os.path.join(args.save_dir, f"traj_{idx:05d}")
            depth_dir = os.path.join(dir_name, "depth")
            color_dir = os.path.join(dir_name, "rgb")
            vis_dir = os.path.join(dir_name, "vis")
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(color_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)

            cprint("=" * 60, "cyan")
            cprint(f"Recording traj {idx} to {dir_name}", "cyan")

            # 每条轨迹开始：复位 + 夹爪到固定状态
            cprint("reset realman", "green")
            env.reset(target_gripper=args.gripper_open_after_reset)

            # 获取起始关节与末端位姿，用于规划起点与对齐方向
            joint_state_rad = env.get_joint()
            if joint_state_rad is None:
                raise RuntimeError("Realman state not ready: env.get_joint() returned None")
            joint_state_rad = np.array(joint_state_rad, dtype=np.float32)

            pose_xyzrpy = env.get_pose()
            if pose_xyzrpy is None:
                raise RuntimeError("Realman state not ready: env.get_pose() returned None")
            pose_xyzrpy = np.array(pose_xyzrpy, dtype=np.float32)

            ee_translation, ee_quat_wxyz, rotation_matrix = pose_xyzrpy_to_ee_pose(
                pose_xyzrpy, apply_tcp2eef=apply_tcp2eef
            )

            # 计算“末端 z 轴在 xy 平面上的投影方向”
            z_axis = rotation_matrix[:, 2]
            z_proj = -z_axis
            z_proj = z_proj.copy()
            z_proj[2] = 0.0
            z_proj = z_proj / np.linalg.norm(z_proj)
            print(f"projection of z axis of ee on XoY plain {z_proj}")

            # 与 Franka 版一致：写入 init.json
            init_info = {
                "init_pose": joint_state_rad.tolist(),
                "k_gains": K_GAINS,
                "d_gains": D_GAINS,
            }
            with open(os.path.join(dir_name, "init.json"), "w") as f:
                json.dump(init_info, f, indent=4)

            # 用 cuRobo 规划出离散关节轨迹序列
            joints_traj = generate_cmd(
                args,
                motion_gen,
                kin_model,
                joint_state_rad=joint_state_rad,
                ee_translation=ee_translation,
                z_proj=z_proj,
                ee_quaternion_wxyz=ee_quat_wxyz,
            )

            input("Press enter to start moving")

            # 保存相机内参（与 Franka 版一致：仅保存彩色流 cam_K）
            color_profile = profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            cam_K = np.array(
                [
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            with open(os.path.join(dir_name, "cam_K.txt"), "w") as f:
                for row in cam_K:
                    f.write(" ".join(f"{x:.10f}" for x in row) + "\n")

            # 同时启动：控制线程 + 相机采集
            timestamps = []
            depth_images = []
            color_images = []

            init_time = time.time()
            control_thread_obj = threading.Thread(
                target=control_thread_realman,
                args=(env, joints_traj, init_time, dir_name, args.cmd_rate),
            )
            control_thread_obj.start()

            for i in range(args.record_frames):
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    cprint("No camera!", "red")
                    continue

                camera_timestamp = frames.get_timestamp() / 1000.0

                env_state = env.get_state()
                if env_state is None:
                    # 状态缓存尚未就绪
                    time.sleep(0.001)
                    continue

                robot_timestamp = env_state["timestamp"] - init_time
                joint_state = env_state["joint"]

                pose_xyzrpy = env_state["pose"]
                pose_xyzrpy = np.array(pose_xyzrpy, dtype=np.float32)
                ee_trans, ee_quat_wxyz, _ = pose_xyzrpy_to_ee_pose(
                    pose_xyzrpy, apply_tcp2eef=apply_tcp2eef
                )

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_images.append((depth_image.astype(np.float32) / 1000.0).copy())
                color_images.append(color_image.copy())

                timestamps.append(
                    {
                        "id": i,
                        "ros_timestamp": robot_timestamp,
                        "camera_timestamp": camera_timestamp,
                        "joint_state": joint_state,
                        "ee_trans": ee_trans.tolist(),
                        "ee_quat_wxyz": ee_quat_wxyz.tolist(),
                    }
                )

            control_thread_obj.join()

            # 写入图像
            for save_idx, (depth_image, color_image) in enumerate(
                tqdm(zip(depth_images, color_images), desc="saving...")
            ):
                np.savez_compressed(os.path.join(depth_dir, f"{save_idx:05d}.npz"), depth=depth_image)
                cv2.imwrite(os.path.join(color_dir, f"{save_idx:05d}.png"), color_image)

            with open(os.path.join(dir_name, "frame.json"), "w") as f:
                json.dump(timestamps, f, indent=4)

    finally:
        env.close()

