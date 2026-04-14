"""
Realman 单相机遥操作数据采集脚本

==================================================
一、功能说明
==================================================
本脚本用于通过 SpaceMouse 遥操作 Realman 机械臂，同步采集 RGB-D + 机器人状态 + 控制动作数据，并保存为 Zarr 数据集（支持断点续采）。

采集内容包括：
- RGB 图像
- 深度图(Depth)
- 关节角(Joint)
- 末端位姿(Pose)
- 控制动作(Action)
- 夹爪状态(Gripper)

==================================================
二、基本用法
==================================================

1. 默认运行:
    python collect_new.py

2. 指定参数:
    python collect_new.py \
        --dataset datasets/vlm/test.zarr \
        --episodes 10 \
        --fps 30 \
        --image-size 320 240

3. 服务器(无GUI)推荐:
    python collect_new.py --no-preview --save-video

==================================================
三、参数说明
==================================================

【数据】
--dataset           数据集路径(Zarr目录)
--episodes          目标 episode 数(支持断点续采)

【机器人】
--robot-ip          机械臂 IP 地址
--home-joint        初始关节角(单位：度，7维)

【相机】
--camera-serial     RealSense 相机序列号
--image-size        保存分辨率(W H)

【采集】
--fps               采集频率(Hz)
--warmup-time       相机预热时间(秒)

【可视化】
--no-preview        关闭 OpenCV 预览窗口(服务器必须开)
--save-video        保存每个 episode 的回放视频
--video-fps         视频帧率(Hz)

==================================================
四、操作方式
==================================================

【SpaceMouse 控制】
- 平移：控制末端 XYZ
- 旋转：控制末端姿态
- 按钮：控制夹爪开合

【键盘控制】
Space   : 开始录制当前 episode
Enter   : 结束并保存 episode
Q       : 退出程序
O       : 打开夹爪
C       : 关闭夹爪
R       : 回到初始位姿

==================================================
五、数据格式(Zarr)
==================================================

数据保存在:
    xxx.zarr/

结构如下:

data/
    rgb            (N, 3, H, W)
    depth          (N, 1, H, W)
    joint          (N, 7)
    pose           (N, 6)
    action         (N, 6)
    gripper_state  (N, 1)
    gripper_width  (N, 1)
    gripper_action (N, 1)
    timestamp      (N,)
    episode        (N,)

meta/
    episode_ends

==================================================
六、断点续采机制
==================================================

如果 dataset 已存在:

    python collect_new.py --dataset xxx.zarr --episodes 10

例如:
- 当前已有 3 个 episode
- 会继续采集到 10 个(再采 7 个)
"""


from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from pytransform3d.rotations import active_matrix_from_angle


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.collect_data_by_tele_op.spacemouse_agent import SpacemouseAgent
from realman.open3d_realsense_env import Open3dRealsenseEnv
from realman.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T


DEFAULT_ROBOT_IP = "192.168.101.19"
DEFAULT_CAMERA_SERIAL = "f1471338"
DEFAULT_DATASET = "datasets/vlm/demo.zarr"
PREVIEW_WINDOW = "Realman RGBD Collector"
GRIPPER_RATE = 0.8
GRIPPER_MIN_DELTA = 0.005
MAX_GRIPPER_WIDTH = 0.09


@dataclass
class EpisodeStats:
    episode_id: int
    steps: int
    duration: float
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect single-camera RGBD teleop data into Zarr.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to the Zarr dataset.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Target total episode count. Resume automatically if the dataset already exists.",
    )
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help="Realman robot IP.")
    parser.add_argument(
        "--camera-serial",
        type=str,
        default=DEFAULT_CAMERA_SERIAL,
        help="RealSense camera serial number.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[320, 240],
        metavar=("W", "H"),
        help="Saved RGBD size in pixels.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Collection loop target FPS.")
    parser.add_argument("--warmup-time", type=float, default=1.0, help="Warmup seconds before collection.")
    parser.add_argument(
        "--home-joint",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Optional home joint in degrees. Defaults to the current joint at startup.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save a replay MP4 for each episode.")
    parser.add_argument("--video-fps", type=float, default=15.0, help="Replay video FPS.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window. Console keyboard control still works.",
    )
    return parser.parse_args()


def delta_to_transform(delta: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = delta[:3] * 0.001

    rx = active_matrix_from_angle(0, delta[3])
    ry = active_matrix_from_angle(1, delta[4])
    rz = active_matrix_from_angle(2, delta[5])
    transform[:3, :3] = rz @ ry @ rx
    return transform


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    if depth.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo = float(np.percentile(valid, 5))
    hi = float(np.percentile(valid, 95))
    if hi <= lo:
        hi = lo + 1.0

    normalized = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth == 0] = 0
    return colored


def make_preview_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    status_lines: list[str],
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    rgb_resized = cv2.resize(rgb, (width, height))
    depth_resized = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

    rgb_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
    depth_bgr = colorize_depth(depth_resized)
    preview = np.hstack([rgb_bgr, depth_bgr])

    overlay = preview.copy()
    line_height = 24
    box_height = 14 + line_height * len(status_lines)
    cv2.rectangle(overlay, (0, 0), (preview.shape[1], box_height), (0, 0, 0), -1)
    preview = cv2.addWeighted(overlay, 0.45, preview, 0.55, 0)

    for idx, text in enumerate(status_lines):
        y = 24 + idx * line_height
        cv2.putText(
            preview,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


class KeyboardInput:
    """Cross-platform non-blocking keyboard helper."""

    def __init__(self):
        self._is_windows = os.name == "nt"
        self._raw_enabled = False
        self._old_settings = None
        self._msvcrt = None
        self._termios = None

        if self._is_windows:
            import msvcrt

            self._msvcrt = msvcrt
        else:
            import termios

            self._termios = termios

    def start(self):
        if self._is_windows:
            return

        import tty

        self._old_settings = self._termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._raw_enabled = True

    def stop(self):
        if not self._is_windows and self._raw_enabled and self._old_settings is not None:
            self._termios.tcsetattr(sys.stdin.fileno(), self._termios.TCSADRAIN, self._old_settings)
            self._raw_enabled = False

    def poll(self, preview_enabled: bool) -> set[str]:
        events: set[str] = set()

        if preview_enabled:
            key = cv2.waitKey(1) & 0xFF
            events |= self._map_key_code(key)

        if self._is_windows:
            while self._msvcrt.kbhit():
                ch = self._msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    if self._msvcrt.kbhit():
                        self._msvcrt.getwch()
                    continue
                events |= self._map_char(ch)
        else:
            import select

            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                events |= self._map_char(ch)

        return events

    @staticmethod
    def _map_key_code(key: int) -> set[str]:
        mapping = {
            13: {"enter"},
            10: {"enter"},
            32: {"space"},
            ord("q"): {"quit"},
            ord("Q"): {"quit"},
            ord("o"): {"open"},
            ord("O"): {"open"},
            ord("c"): {"close"},
            ord("C"): {"close"},
            ord("r"): {"home"},
            ord("R"): {"home"},
        }
        return mapping.get(key, set())

    @staticmethod
    def _map_char(ch: str) -> set[str]:
        if ch in ("\r", "\n"):
            return {"enter"}
        if ch == " ":
            return {"space"}
        if ch in ("\x03", "\x1b"):
            return {"quit"}
        if ch in ("q", "Q"):
            return {"quit"}
        if ch in ("o", "O"):
            return {"open"}
        if ch in ("c", "C"):
            return {"close"}
        if ch in ("r", "R"):
            return {"home"}
        return set()


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _append_to_dataset(arr: zarr.Array, values: np.ndarray):
    count = values.shape[0]
    old_size = arr.shape[0]
    new_shape = list(arr.shape)
    new_shape[0] = old_size + count
    arr.resize(tuple(new_shape))
    arr[old_size : old_size + count] = values


def _wait_for_state(env: RealmanEnv, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = env.get_state()
        if state is not None:
            return state
        time.sleep(0.05)
    raise RuntimeError("Timed out waiting for robot state.")


def _compute_episode_ends(data_group: zarr.Group, meta_group: zarr.Group):
    episodes = np.asarray(data_group["episode"][:], dtype=np.int64)
    if episodes.size == 0:
        ends = np.array([], dtype=np.uint32)
    else:
        _, counts = np.unique(episodes, return_counts=True)
        ends = np.cumsum(counts).astype(np.uint32)

    if "episode_ends" in meta_group:
        del meta_group["episode_ends"]
    meta_group.create_dataset("episode_ends", data=ends, dtype=np.uint32)


def _open_or_create_dataset(
    dataset_path: str,
    image_size: tuple[int, int],
    camera_serial: str,
    camera_meta: dict,
    fps: float,
) -> tuple[zarr.Group, zarr.Group, int]:
    _ensure_parent_dir(dataset_path)

    width, height = image_size
    rgb_shape = (3, height, width)
    depth_shape = (1, height, width)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    if os.path.exists(dataset_path):
        root = zarr.open(dataset_path, mode="a")
        if "data" not in root or "meta" not in root:
            raise RuntimeError(
                f"Existing dataset at {dataset_path} is not in the new episode-based format. "
                "Please use a new path or migrate the old dataset first."
            )

        data_group = root["data"]
        meta_group = root["meta"]
        if "episode" not in data_group:
            raise RuntimeError(
                f"Dataset {dataset_path} has no data/episode field, so resume is not possible."
            )

        existing_episodes = len(np.unique(np.asarray(data_group["episode"][:], dtype=np.int64)))
        if "episode_ends" not in meta_group:
            _compute_episode_ends(data_group, meta_group)
        root.attrs["last_opened_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return data_group, meta_group, existing_episodes

    root = zarr.open(dataset_path, mode="w")
    root.attrs["dataset_type"] = "realman_single_camera_rgbd"
    root.attrs["camera_serial"] = camera_serial
    root.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    root.attrs["fps"] = float(fps)
    root.attrs["image_size"] = [width, height]

    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    meta_group.attrs["camera"] = camera_meta

    data_group.create_dataset(
        "rgb",
        shape=(0, *rgb_shape),
        chunks=(1, *rgb_shape),
        dtype=np.uint8,
        compressor=compressor,
    )
    data_group.create_dataset(
        "depth",
        shape=(0, *depth_shape),
        chunks=(1, *depth_shape),
        dtype=np.uint16,
        compressor=compressor,
    )
    data_group.create_dataset("joint", shape=(0, 7), chunks=(256, 7), dtype=np.float32)
    data_group.create_dataset("pose", shape=(0, 6), chunks=(256, 6), dtype=np.float32)
    data_group.create_dataset("action", shape=(0, 6), chunks=(256, 6), dtype=np.float32)
    data_group.create_dataset("gripper_state", shape=(0, 1), chunks=(256, 1), dtype=np.float32)
    data_group.create_dataset("gripper_width", shape=(0, 1), chunks=(256, 1), dtype=np.float32)
    data_group.create_dataset("gripper_action", shape=(0, 1), chunks=(256, 1), dtype=np.float32)
    data_group.create_dataset("timestamp", shape=(0,), chunks=(256,), dtype=np.float64)
    data_group.create_dataset("episode", shape=(0,), chunks=(256,), dtype=np.uint32)

    meta_group.create_dataset("episode_ends", data=np.array([], dtype=np.uint32), dtype=np.uint32)
    return data_group, meta_group, 0


class Collector:
    def __init__(
        self,
        env: RealmanEnv,
        camera: Open3dRealsenseEnv,
        agent: SpacemouseAgent,
        dataset_path: str,
        target_total_episodes: int,
        image_size: tuple[int, int],
        fps: float,
        save_video: bool,
        video_fps: float,
        preview_enabled: bool,
        warmup_time: float,
        home_joint_rad: np.ndarray,
    ):
        self.env = env
        self.camera = camera
        self.agent = agent
        self.dataset_path = dataset_path
        self.target_total_episodes = target_total_episodes
        self.image_size = image_size
        self.period = 1.0 / fps
        self.fps = fps
        self.save_video = save_video
        self.video_fps = video_fps
        self.preview_enabled = preview_enabled
        self.warmup_time = warmup_time
        self.home_joint_rad = np.asarray(home_joint_rad, dtype=np.float64)

    def _print_banner(self, existing_episodes: int, remaining: int, total_steps: int):
        width, height = self.image_size
        lines = [
            "",
            "==============================================",
            " Realman Single-Camera Zarr Collector",
            "==============================================",
            f"dataset           : {self.dataset_path}",
            f"episodes target   : {self.target_total_episodes}",
            f"episodes existing : {existing_episodes}",
            f"episodes remaining: {remaining}",
            f"total steps       : {total_steps}",
            f"image size        : {width}x{height}",
            f"collector fps     : {self.fps:.1f}",
            f"save replay video : {'ON' if self.save_video else 'OFF'}",
            f"preview window    : {'ON' if self.preview_enabled else 'OFF'}",
            "controls          : Space=start, Enter=end, Q=quit, O=open, C=close, R=go-home",
            "",
        ]
        print("\n".join(lines))

    def _print_episode_summary(self, stats: EpisodeStats, saved_in_session: int, total_steps: int, remaining: int):
        print(
            "\n"
            f"Episode {stats.episode_id} saved\n"
            f"  steps    : {stats.steps}\n"
            f"  duration : {stats.duration:.1f}s\n"
            f"  avg fps  : {stats.fps:.1f}\n"
            f"  session  : {saved_in_session} episodes saved this run\n"
            f"  dataset  : {total_steps} total steps\n"
            f"  remain   : {remaining} episodes\n"
        )

    def _build_preview(self, rgb: np.ndarray, depth: np.ndarray, lines: list[str]) -> np.ndarray | None:
        if not self.preview_enabled and not self.save_video:
            return None

        preview = make_preview_frame(rgb, depth, lines, self.image_size)
        if self.preview_enabled:
            cv2.imshow(PREVIEW_WINDOW, preview)
        return preview

    def _capture_frame(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.camera.step()
        rgb = np.asarray(obs["rgb"])
        depth = np.asarray(obs["depth"]).astype(np.uint16)
        return rgb, depth

    def _resize_frame(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.image_size
        rgb_small = cv2.resize(rgb, (width, height))
        depth_small = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        return rgb_small, depth_small

    def _video_path(self, episode_id: int) -> str:
        base = os.path.splitext(self.dataset_path)[0]
        video_dir = f"{base}_videos"
        os.makedirs(video_dir, exist_ok=True)
        return os.path.join(video_dir, f"ep{episode_id:04d}_preview.mp4")

    def _wait_for_robot_state(self) -> np.ndarray:
        for _ in range(100):
            state = self.env.get_state()
            if state is not None:
                return T_from_realman_xyzrpy(state.pose)
            time.sleep(0.05)
        raise RuntimeError("Failed to read initial robot state from RealmanEnv.")

    def _move_to_home_joint(self) -> np.ndarray:
        print("Returning to home joint...")
        self.env.slow_stop()
        time.sleep(0.1)

        move_ret = self.env.driver.movej(np.degrees(self.home_joint_rad))
        if move_ret != 0:
            raise RuntimeError(f"movej(home_joint) failed with ret={move_ret}")

        deadline = time.time() + 8.0
        while time.time() < deadline:
            state = self.env.get_state()
            if state is not None:
                joint_err = np.linalg.norm(state.joint - self.home_joint_rad)
                if joint_err < 0.08:
                    return T_from_realman_xyzrpy(state.pose)
            time.sleep(0.05)

        state = self.env.get_state()
        if state is None:
            raise RuntimeError("Robot state unavailable after moving to home joint.")
        return T_from_realman_xyzrpy(state.pose)

    def _wait_for_episode_start(self, episode_id: int, keyboard: KeyboardInput, goal_gripper: float) -> tuple[bool, float]:
        print(f"\nEpisode {episode_id}: waiting for Space to start. Press Q to quit.")
        last_sent_gripper = goal_gripper

        while True:
            loop_start = time.perf_counter()
            rgb, depth = self._capture_frame()
            events = keyboard.poll(self.preview_enabled)
            _, buttons = self.agent.act()

            goal_gripper = self._update_goal_gripper(goal_gripper, buttons, events, self.period)
            if abs(goal_gripper - last_sent_gripper) >= GRIPPER_MIN_DELTA:
                self.env.send_gripper(goal_gripper * MAX_GRIPPER_WIDTH)
                last_sent_gripper = goal_gripper

            status = [
                f"Episode {episode_id} | waiting to start",
                "Space=start  Enter=ignored  Q=quit  O/C=gripper  R=go-home",
                f"gripper_norm={goal_gripper:.2f}",
            ]
            self._build_preview(rgb, depth, status)

            if "home" in events:
                self._move_to_home_joint()
                continue

            if "quit" in events:
                return False, goal_gripper
            if "space" in events:
                return True, goal_gripper

            elapsed = time.perf_counter() - loop_start
            if self.period > elapsed:
                time.sleep(self.period - elapsed)

    @staticmethod
    def _update_goal_gripper(
        goal_gripper: float,
        buttons: np.ndarray,
        events: set[str],
        dt: float,
    ) -> float:
        close_pressed = bool(buttons[0]) or "close" in events
        open_pressed = bool(buttons[1]) or "open" in events

        if close_pressed ^ open_pressed:
            direction = -1.0 if close_pressed else 1.0
            goal_gripper += direction * GRIPPER_RATE * dt

        return float(np.clip(goal_gripper, 0.0, 1.0))

    def _save_episode_buffer(self, data_group: zarr.Group, episode_id: int, buffer: dict[str, list[np.ndarray]]):
        stacked = {
            "rgb": np.stack(buffer["rgb"], axis=0).astype(np.uint8),
            "depth": np.stack(buffer["depth"], axis=0).astype(np.uint16),
            "joint": np.stack(buffer["joint"], axis=0).astype(np.float32),
            "pose": np.stack(buffer["pose"], axis=0).astype(np.float32),
            "action": np.stack(buffer["action"], axis=0).astype(np.float32),
            "gripper_state": np.asarray(buffer["gripper_state"], dtype=np.float32)[:, None],
            "gripper_width": np.asarray(buffer["gripper_width"], dtype=np.float32)[:, None],
            "gripper_action": np.asarray(buffer["gripper_action"], dtype=np.float32)[:, None],
            "timestamp": np.asarray(buffer["timestamp"], dtype=np.float64),
            "episode": np.full((len(buffer["rgb"]),), episode_id, dtype=np.uint32),
        }

        stacked["rgb"] = np.transpose(stacked["rgb"], (0, 3, 1, 2))
        stacked["depth"] = stacked["depth"][:, None, :, :]

        for key, values in stacked.items():
            _append_to_dataset(data_group[key], values)

    def _run_episode(
        self,
        data_group: zarr.Group,
        episode_id: int,
        keyboard: KeyboardInput,
        goal_gripper: float,
    ) -> tuple[EpisodeStats | None, bool, float]:
        start_ok, goal_gripper = self._wait_for_episode_start(episode_id, keyboard, goal_gripper)
        if not start_ok:
            return None, True, goal_gripper

        target_transform = self._wait_for_robot_state()
        print(f"Episode {episode_id}: recording... Press Enter to save, Q to quit.")

        buffer: dict[str, list[np.ndarray]] = {
            "rgb": [],
            "depth": [],
            "joint": [],
            "pose": [],
            "action": [],
            "gripper_state": [],
            "gripper_width": [],
            "gripper_action": [],
            "timestamp": [],
        }

        last_sent_gripper = goal_gripper
        replay_writer = None
        episode_start_time = time.time()
        steps = 0

        try:
            while True:
                loop_start = time.perf_counter()
                events = keyboard.poll(self.preview_enabled)
                if "quit" in events:
                    raise KeyboardInterrupt
                if "enter" in events:
                    break

                action, buttons = self.agent.act()
                goal_gripper = self._update_goal_gripper(goal_gripper, buttons, events, self.period)

                if "home" in events:
                    target_transform = self._move_to_home_joint()
                    continue

                rgb, depth = self._capture_frame()
                rgb_small, depth_small = self._resize_frame(rgb, depth)
                state = self.env.get_state()
                if state is None:
                    raise RuntimeError("Lost robot state during recording.")

                target_transform = target_transform @ delta_to_transform(action)
                self.env.send_pose(realman_xyzrpy_from_T(target_transform))

                if abs(goal_gripper - last_sent_gripper) >= GRIPPER_MIN_DELTA:
                    self.env.send_gripper(goal_gripper * MAX_GRIPPER_WIDTH)
                    last_sent_gripper = goal_gripper

                preview_lines = [
                    f"Episode {episode_id} | recording | step={steps}",
                    "Enter=finish  Q=quit  O/C=gripper  R=go-home",
                    f"gripper_norm={goal_gripper:.2f}  width={state.gripper:.4f}m",
                ]
                preview = self._build_preview(rgb_small, depth_small, preview_lines)

                if self.save_video and preview is not None:
                    if replay_writer is None:
                        video_path = self._video_path(episode_id)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        size = (preview.shape[1], preview.shape[0])
                        replay_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, size)
                    replay_writer.write(preview)

                buffer["rgb"].append(rgb_small)
                buffer["depth"].append(depth_small)
                buffer["joint"].append(state.joint.copy())
                buffer["pose"].append(state.pose.copy())
                buffer["action"].append(np.asarray(action, dtype=np.float32))
                buffer["gripper_state"].append(goal_gripper)
                buffer["gripper_width"].append(float(state.gripper))
                buffer["gripper_action"].append(goal_gripper * MAX_GRIPPER_WIDTH)
                buffer["timestamp"].append(time.time() - episode_start_time)

                steps += 1
                if steps % 50 == 0:
                    elapsed = time.time() - episode_start_time
                    fps = steps / max(elapsed, 1e-6)
                    print(f"  step={steps} fps={fps:.1f} gripper={goal_gripper:.2f}")

                elapsed = time.perf_counter() - loop_start
                if self.period > elapsed:
                    time.sleep(self.period - elapsed)
        finally:
            if replay_writer is not None:
                replay_writer.release()

        if steps == 0:
            print(f"Episode {episode_id}: skipped because no frames were recorded.")
            return None, False, goal_gripper

        self._save_episode_buffer(data_group, episode_id, buffer)
        duration = time.time() - episode_start_time
        stats = EpisodeStats(
            episode_id=episode_id,
            steps=steps,
            duration=duration,
            fps=steps / max(duration, 1e-6),
        )
        return stats, False, goal_gripper

    def run(self):
        data_group, meta_group, existing_episodes = _open_or_create_dataset(
            dataset_path=self.dataset_path,
            image_size=self.image_size,
            camera_serial=self.camera.meta_obs.get("serial", "unknown"),
            camera_meta=self.camera.meta_obs,
            fps=self.fps,
        )

        remaining = self.target_total_episodes - existing_episodes
        total_steps = len(data_group["episode"])
        self._print_banner(existing_episodes, max(remaining, 0), total_steps)

        if remaining <= 0:
            print(
                f"Dataset already has {existing_episodes} episodes, "
                f"which meets the target {self.target_total_episodes}. Nothing to do."
            )
            return

        print(f"Warming up for {self.camera.meta_obs.get('size', ['?', '?'])} camera stream...")
        time.sleep(self.period)
        time.sleep(max(0.0, self.warmup_time))

        keyboard = KeyboardInput()
        keyboard.start()
        goal_gripper = 1.0
        saved_in_session = 0

        if self.preview_enabled:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

        try:
            while saved_in_session < remaining:
                episode_id = existing_episodes + saved_in_session
                try:
                    stats, should_quit, goal_gripper = self._run_episode(
                        data_group=data_group,
                        episode_id=episode_id,
                        keyboard=keyboard,
                        goal_gripper=goal_gripper,
                    )
                except KeyboardInterrupt:
                    print("\nQuit requested. Leaving collector gracefully.")
                    break
                except Exception:
                    print(
                        f"\nEpisode {episode_id} failed. No partial data was written.\n"
                        f"{traceback.format_exc()}"
                    )
                    continue

                if should_quit:
                    print("\nQuit requested before recording started.")
                    break

                if stats is None:
                    continue

                saved_in_session += 1
                _compute_episode_ends(data_group, meta_group)
                total_steps = len(data_group["episode"])
                remaining_after = remaining - saved_in_session
                self._print_episode_summary(stats, saved_in_session, total_steps, remaining_after)
        finally:
            keyboard.stop()
            if self.preview_enabled:
                cv2.destroyAllWindows()


def main():
    args = parse_args()

    # ===== 自动生成带时间戳的dataset路径 =====
    if args.dataset == DEFAULT_DATASET:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(DEFAULT_DATASET)
        args.dataset = os.path.join(
            base_dir,
            f"{timestamp}_ep{args.episodes}.zarr"
        )

    print("Dataset will be saved to:", os.path.abspath(args.dataset))

    env = RealmanEnv(args.robot_ip, mode="async")
    camera = Open3dRealsenseEnv(args.camera_serial)
    camera.meta_obs["serial"] = args.camera_serial
    agent = SpacemouseAgent()

    initial_state = _wait_for_state(env)

    home_joint_rad= np.radians([90.0, 0, 0, -90, 0, -90, 60])

    collector = Collector(
        env=env,
        camera=camera,
        agent=agent,
        dataset_path=args.dataset,
        target_total_episodes=args.episodes,
        image_size=tuple(args.image_size),
        fps=args.fps,
        save_video=args.save_video,
        video_fps=args.video_fps,
        preview_enabled=not args.no_preview,
        warmup_time=args.warmup_time,
        home_joint_rad=home_joint_rad,
    )

    try:
        collector.run()
    finally:
        env.close()
        camera.close()
        agent.close()


if __name__ == "__main__":
    main()
