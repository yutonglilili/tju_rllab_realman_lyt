"""
Realman 双相机遥操作数据采集脚本（ForceFlow 适配版）

==================================================
一、功能说明
==================================================
本脚本通过 SpaceMouse 遥操作 Realman 机械臂，同步采集双视角 RGB + 机器人状态 + 控制动作，
保存为 Zarr 数据集（支持断点续采）。相对原 ForceFlow xArm 采集脚本：
    - 移除了 force/ft 传感器相关字段（Realman 无 6 轴力传感器）
    - 双相机方案：rgb_arm (D435 腕部) + rgb_fix (L515 固定)
    - 深度暂时禁用（搜索 "暂时禁用深度" 可一键恢复 depth_fix）
    - 字段命名兼顾两侧：pos / action / gripper_state / gripper_action 沿用 ForceFlow，
      额外保留 realman 风格的 joint / gripper_width / timestamp

==================================================
二、基本用法
==================================================

所有可改参数集中在文件顶部「用户可修改参数」一节，直接编辑即可运行：

    python -m scripts.collect

需要改采集条数、保存路径、相机/机械臂参数等都在那里改。

==================================================
三、操作方式
==================================================

【SpaceMouse 控制】
- 平移：控制末端 XYZ
- 旋转：控制末端姿态
- 按钮：左键 → 夹爪闭合(0)，右键 → 夹爪打开(1)，二值切换

【键盘控制】
Space   : 开始录制当前 episode
Enter   : 结束并保存 episode
Q       : 退出程序
O       : 打开夹爪
C       : 关闭夹爪
R       : 回到初始位姿

==================================================
四、数据格式(Zarr)
==================================================

data/
    rgb_arm        (N, 3, H, W)  uint8    -- D435 腕部相机
    rgb_fix        (N, 3, H, W)  uint8    -- L515 固定相机
    # depth_fix    (N, 1, H, W)  uint16   -- 暂时禁用深度
    joint          (N, 7)        float32
    pos            (N, 6)        float32  -- EEF xyzrpy (SDK 原生坐标)
    action         (N, 6)        float32  -- spacemouse delta
    gripper_state  (N, 1)        float32  -- 二值 {0,1} (obs)
    gripper_action (N, 1)        float32  -- 二值 {0,1} (action)
    gripper_width  (N, 1)        float32  -- 实际开度 [0,1]
    timestamp      (N,)          float64
    episode        (N,)          uint32

meta/
    episode_ends
"""


from __future__ import annotations

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


PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FORCEFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (PROJECT_ROOT, FORCEFLOW_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T
from env.realsense_env import RealsenseEnv
from env.spacemouse import SpacemouseAgent
from CleanDiffuser.image_codecs import jpeg as jpeg_codec  # rgb 用 jpeg 压缩


# ═══════════════════════════════════════════════════════════════════
# 用户可修改参数（直接在此处编辑即可，无需命令行）
# ═══════════════════════════════════════════════════════════════════

# ----- 数据集 -----
DATASET_PATH   = "data/demo.zarr"     # Zarr 数据集保存路径
NUM_EPISODES   = 3                    # 目标采集 episode 数(支持断点续采)

# ----- 机器人 -----
ROBOT_IP       = "192.168.101.19"     # 机械臂 IP
# home 关节角写死在 env.realman_env.RealmanEnv.reset() 里, 想改去那里改

# ----- 相机 -----
CAMERA_SERIAL_FIX  = "f1471338"        # 固定视角 (L515)
CAMERA_SERIAL_ARM  = "342522073663"    # 腕部视角 (D435)
L515_VISUAL_PRESET = "RS2_L500_VISUAL_PRESET_MAX_RANGE"  # 只用于 L515

# 相机原始分辨率(主要影响相机端工作模式, 之后会 resize 到 IMAGE_SIZE)
# - D435: 可设 640x480
# - L515: color 仅支持 1280x720 / 1920x1080 / 960x540, 留空就用 SDK 默认
ARM_COLOR_RES = (640, 480)            # D435 color
ARM_DEPTH_RES = (640, 480)            # D435 depth
FIX_COLOR_RES = None                  # L515 color, None=SDK 默认
FIX_DEPTH_RES = None                  # L515 depth, None=SDK 默认

IMAGE_SIZE   = (320, 240)             # 入库分辨率 (W, H), 三个图像字段都按此 resize
JPEG_QUALITY = 90                     # rgb_arm/rgb_fix 的 jpeg 编码质量 (1-100, 90 是常用平衡点)

# ----- 采集时序 -----
FPS            = 30.0                 # 采集频率(Hz)
WARMUP_TIME    = 1.0                  # 启动前预热时长(秒)

# ----- 可视化 -----
PREVIEW_ENABLED = True                # OpenCV 预览窗口(服务器无 GUI 设 False)
SAVE_VIDEO      = False               # 是否保存每个 episode 的回放 MP4
VIDEO_FPS       = 15.0                # 回放视频帧率

# ----- 夹爪(一般无需修改) -----
# 新 env 用 [0,1] 接口，1=全开 0=全闭，不需要再乘物理宽度
GRIPPER_MIN_DELTA = 0.005             # 夹爪命令最小变化阈值(防颤抖；二值切换会一次性跳 1.0)

# ----- 内部常量(无需修改) -----
PREVIEW_WINDOW = "Realman RGBD Collector (ForceFlow)"

# ═══════════════════════════════════════════════════════════════════


@dataclass
class EpisodeStats:
    episode_id: int
    steps: int
    duration: float
    fps: float


def delta_to_transform(delta: np.ndarray) -> np.ndarray:
    """Convert a 6-dof spacemouse delta (mm + rad) into a 4x4 SE(3) transform."""
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
    root.attrs["dataset_type"] = "realman_dual_camera_rgbd_forceflow"
    root.attrs["camera_serial_arm"] = camera_serial.get("arm", "")
    root.attrs["camera_serial_fix"] = camera_serial.get("fix", "")
    root.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    root.attrs["fps"] = float(fps)
    root.attrs["image_size"] = [width, height]

    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    meta_group.attrs["camera_arm"] = camera_meta.get("arm", {})
    meta_group.attrs["camera_fix"] = camera_meta.get("fix", {})

    # rgb 用 jpeg 编码 (lossy 但压缩比高, 与 ForceFlow xArm 原版一致)
    rgb_compressor = jpeg_codec((1, *rgb_shape), quality=JPEG_QUALITY)
    data_group.create_dataset(
        "rgb_arm",
        shape=(0, *rgb_shape),
        chunks=(1, *rgb_shape),
        dtype=np.uint8,
        compressor=rgb_compressor,
    )
    data_group.create_dataset(
        "rgb_fix",
        shape=(0, *rgb_shape),
        chunks=(1, *rgb_shape),
        dtype=np.uint8,
        compressor=rgb_compressor,
    )
    # 暂时禁用深度
    # data_group.create_dataset(
    #     "depth_fix",
    #     shape=(0, *depth_shape),
    #     chunks=(1, *depth_shape),
    #     dtype=np.uint16,
    #     compressor=compressor,
    # )
    data_group.create_dataset("joint", shape=(0, 7), chunks=(256, 7), dtype=np.float32)
    data_group.create_dataset("pos", shape=(0, 6), chunks=(256, 6), dtype=np.float32)
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
        camera_arm: RealsenseEnv,
        camera_fix: RealsenseEnv,
        agent: SpacemouseAgent,
        dataset_path: str,
        target_total_episodes: int,
        image_size: tuple[int, int],
        fps: float,
        save_video: bool,
        video_fps: float,
        preview_enabled: bool,
        warmup_time: float,
    ):
        self.env = env
        self.camera_arm = camera_arm
        self.camera_fix = camera_fix
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

    def _print_banner(self, existing_episodes: int, remaining: int, total_steps: int):
        width, height = self.image_size
        lines = [
            "",
            "==============================================",
            " Realman Single-Camera Zarr Collector (ForceFlow)",
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

    def _build_preview(
        self,
        rgb_arm: np.ndarray,
        rgb_fix: np.ndarray,
        # depth_fix: np.ndarray,    # 暂时禁用深度
        lines: list[str],
    ) -> np.ndarray | None:
        """双视图横向拼接: rgb_arm | rgb_fix"""
        if not self.preview_enabled and not self.save_video:
            return None

        width, height = self.image_size
        arm_bgr = cv2.cvtColor(cv2.resize(rgb_arm, (width, height)), cv2.COLOR_RGB2BGR)
        fix_bgr = cv2.cvtColor(cv2.resize(rgb_fix, (width, height)), cv2.COLOR_RGB2BGR)
        # 暂时禁用深度
        # depth_bgr = colorize_depth(cv2.resize(depth_fix, (width, height), interpolation=cv2.INTER_NEAREST))

        preview = np.hstack([arm_bgr, fix_bgr])

        overlay = preview.copy()
        line_height = 24
        box_height = 14 + line_height * len(lines)
        cv2.rectangle(overlay, (0, 0), (preview.shape[1], box_height), (0, 0, 0), -1)
        preview = cv2.addWeighted(overlay, 0.45, preview, 0.55, 0)
        for idx, text in enumerate(lines):
            y = 24 + idx * line_height
            cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        if self.preview_enabled:
            cv2.imshow(PREVIEW_WINDOW, preview)
        return preview

    def _capture_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """同时抓两台相机, 返回 (rgb_arm, rgb_fix), 都是原始分辨率"""
        obs_arm = self.camera_arm.step()
        obs_fix = self.camera_fix.step()
        rgb_arm = np.asarray(obs_arm["rgb"])
        rgb_fix = np.asarray(obs_fix["rgb"])
        # 暂时禁用深度
        # depth_fix = np.asarray(obs_fix["depth"]).astype(np.uint16)
        return rgb_arm, rgb_fix

    def _resize_frame(
        self,
        rgb_arm: np.ndarray,
        rgb_fix: np.ndarray,
        # depth_fix: np.ndarray,   # 暂时禁用深度
    ) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.image_size
        return (
            cv2.resize(rgb_arm, (width, height)),
            cv2.resize(rgb_fix, (width, height)),
            # cv2.resize(depth_fix, (width, height), interpolation=cv2.INTER_NEAREST),  # 暂时禁用深度
        )

    def _video_path(self, episode_id: int) -> str:
        base = os.path.splitext(self.dataset_path)[0]
        video_dir = f"{base}_videos"
        os.makedirs(video_dir, exist_ok=True)
        return os.path.join(video_dir, f"ep{episode_id:04d}_preview.mp4")

    def _wait_for_robot_state(self) -> np.ndarray:
        """阻塞直到 env 返回 state, 把 6 维 pose 转 4x4 给 delta 累加用"""
        for _ in range(100):
            state = self.env.get_state()
            if state is not None:
                return T_from_realman_xyzrpy(state["pose"])
            time.sleep(0.05)
        raise RuntimeError("Failed to read initial robot state from RealmanEnv.")

    def _goto_home(self, goal_gripper: float) -> np.ndarray:
        """复位到 env 内置 home 关节角并对齐夹爪, 返回新的 4x4 target 用于 delta 累加"""
        print("[Collector] Returning to home...")
        obs = self.env.reset(target_gripper=goal_gripper, speed_ratio=30)
        return T_from_realman_xyzrpy(obs["pose"])

    def _wait_for_episode_start(self, episode_id: int, keyboard: KeyboardInput, goal_gripper: float) -> tuple[bool, float]:
        print(f"\nEpisode {episode_id}: waiting for Space to start. Press Q to quit.")
        last_sent_gripper = goal_gripper

        while True:
            loop_start = time.perf_counter()
            rgb_arm, rgb_fix = self._capture_frame()
            events = keyboard.poll(self.preview_enabled)
            _, buttons = self.agent.act()

            goal_gripper = self._update_goal_gripper(goal_gripper, buttons, events)
            if abs(goal_gripper - last_sent_gripper) >= GRIPPER_MIN_DELTA:
                self.env.send_gripper(goal_gripper)
                last_sent_gripper = goal_gripper

            status = [
                f"Episode {episode_id} | waiting to start",
                "Space=start  Enter=ignored  Q=quit  O/C=gripper  R=go-home",
                f"gripper_norm={goal_gripper:.2f}",
            ]
            self._build_preview(rgb_arm, rgb_fix, status)

            if "home" in events:
                self._goto_home(goal_gripper)
                last_sent_gripper = goal_gripper  # reset 内部已对齐夹爪
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
    ) -> float:
        """二值切换：左键(buttons[0]/C 键) → 0(闭)，右键(buttons[1]/O 键) → 1(开)。
        没按按钮时保持当前值，同时按下两键忽略。"""
        close_pressed = bool(buttons[0]) or "close" in events
        open_pressed = bool(buttons[1]) or "open" in events
        if open_pressed and not close_pressed:
            return 1.0
        if close_pressed and not open_pressed:
            return 0.0
        return goal_gripper

    def _save_episode_buffer(self, data_group: zarr.Group, episode_id: int, buffer: dict[str, list[np.ndarray]]):
        n = len(buffer["rgb_arm"])
        stacked = {
            "rgb_arm":       np.stack(buffer["rgb_arm"], axis=0).astype(np.uint8),
            "rgb_fix":       np.stack(buffer["rgb_fix"], axis=0).astype(np.uint8),
            # "depth_fix":     np.stack(buffer["depth_fix"], axis=0).astype(np.uint16),  # 暂时禁用深度
            "joint":         np.stack(buffer["joint"], axis=0).astype(np.float32),
            "pos":           np.stack(buffer["pos"], axis=0).astype(np.float32),
            "action":        np.stack(buffer["action"], axis=0).astype(np.float32),
            "gripper_state": np.asarray(buffer["gripper_state"], dtype=np.float32)[:, None],
            "gripper_width": np.asarray(buffer["gripper_width"], dtype=np.float32)[:, None],
            "gripper_action":np.asarray(buffer["gripper_action"], dtype=np.float32)[:, None],
            "timestamp":     np.asarray(buffer["timestamp"], dtype=np.float64),
            "episode":       np.full((n,), episode_id, dtype=np.uint32),
        }

        # HWC → CHW
        stacked["rgb_arm"]   = np.transpose(stacked["rgb_arm"], (0, 3, 1, 2))
        stacked["rgb_fix"]   = np.transpose(stacked["rgb_fix"], (0, 3, 1, 2))
        # 暂时禁用深度
        # stacked["depth_fix"] = stacked["depth_fix"][:, None, :, :]

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
            "rgb_arm": [],
            "rgb_fix": [],
            # "depth_fix": [],   # 暂时禁用深度
            "joint": [],
            "pos": [],
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
                goal_gripper = self._update_goal_gripper(goal_gripper, buttons, events)

                if "home" in events:
                    target_transform = self._goto_home(goal_gripper)
                    last_sent_gripper = goal_gripper
                    continue

                rgb_arm, rgb_fix = self._capture_frame()
                rgb_arm_s, rgb_fix_s = self._resize_frame(rgb_arm, rgb_fix)
                state = self.env.get_state()
                if state is None:
                    raise RuntimeError("Lost robot state during recording.")

                # 4x4 只在这一行用：spacemouse delta SE(3) 复合到当前目标位姿
                target_transform = target_transform @ delta_to_transform(action)
                # 转回 6 维 EEF xyzrpy 发给 SDK
                self.env.send_pose(realman_xyzrpy_from_T(target_transform))

                if abs(goal_gripper - last_sent_gripper) >= GRIPPER_MIN_DELTA:
                    self.env.send_gripper(goal_gripper)
                    last_sent_gripper = goal_gripper

                preview_lines = [
                    f"Episode {episode_id} | recording | step={steps}",
                    "Enter=finish  Q=quit  O/C=gripper  R=go-home",
                    f"gripper_norm={goal_gripper:.2f}  open={state['gripper_open']:.2f}",
                ]
                preview = self._build_preview(rgb_arm_s, rgb_fix_s, preview_lines)

                if self.save_video and preview is not None:
                    if replay_writer is None:
                        video_path = self._video_path(episode_id)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        size = (preview.shape[1], preview.shape[0])
                        replay_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, size)
                    replay_writer.write(preview)

                buffer["rgb_arm"].append(rgb_arm_s)
                buffer["rgb_fix"].append(rgb_fix_s)
                # buffer["depth_fix"].append(depth_fix_s)  # 暂时禁用深度
                buffer["joint"].append(state["joint"].copy())
                buffer["pos"].append(state["pose"].astype(np.float32))
                buffer["action"].append(np.asarray(action, dtype=np.float32))
                buffer["gripper_state"].append(goal_gripper)
                buffer["gripper_width"].append(float(state["gripper_open"]))
                buffer["gripper_action"].append(goal_gripper)
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
            camera_serial={
                "arm": self.camera_arm.meta_obs.get("serial", "unknown"),
                "fix": self.camera_fix.meta_obs.get("serial", "unknown"),
            },
            camera_meta={
                "arm": self.camera_arm.meta_obs,
                "fix": self.camera_fix.meta_obs,
            },
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

        print(f"Warming up cameras (arm + fix)...")
        time.sleep(self.period)
        time.sleep(max(0.0, self.warmup_time))

        keyboard = KeyboardInput()
        keyboard.start()
        goal_gripper = 1.0
        saved_in_session = 0

        if self.preview_enabled:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

        # 录制开始前先复位一次
        print("[Collector] 录制开始, 复位机械臂到 home...")
        self.env.reset(target_gripper=goal_gripper, speed_ratio=30)

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

                # Episode 结束后复位 (除非已采够)
                if saved_in_session < remaining:
                    print(f"[Collector] Episode {episode_id} 结束, 复位机械臂...")
                    self.env.reset(target_gripper=1.0, speed_ratio=30)
                    goal_gripper = 1.0
        finally:
            keyboard.stop()
            if self.preview_enabled:
                cv2.destroyAllWindows()


def main():
    print("Dataset will be saved to:", os.path.abspath(DATASET_PATH))

    env = RealmanEnv(robot_ip=ROBOT_IP, async_mode=True)
    camera_arm = RealsenseEnv(
        serial=CAMERA_SERIAL_ARM,
        color_resolution=ARM_COLOR_RES,
        depth_resolution=ARM_DEPTH_RES,
    )
    camera_fix = RealsenseEnv(
        serial=CAMERA_SERIAL_FIX,
        color_resolution=FIX_COLOR_RES,
        depth_resolution=FIX_DEPTH_RES,
        visual_preset=L515_VISUAL_PRESET,
    )
    agent = SpacemouseAgent()

    _wait_for_state(env)

    collector = Collector(
        env=env,
        camera_arm=camera_arm,
        camera_fix=camera_fix,
        agent=agent,
        dataset_path=DATASET_PATH,
        target_total_episodes=NUM_EPISODES,
        image_size=tuple(IMAGE_SIZE),
        fps=FPS,
        save_video=SAVE_VIDEO,
        video_fps=VIDEO_FPS,
        preview_enabled=PREVIEW_ENABLED,
        warmup_time=WARMUP_TIME,
    )

    try:
        collector.run()
    finally:
        env.close()
        camera_arm.close()
        camera_fix.close()
        agent.close()


if __name__ == "__main__":
    main()
