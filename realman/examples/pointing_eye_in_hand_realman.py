"""
当前脚本可以用于验证腕部相机标定结果。
启动 hand_eye 环境，直接运行即可。
Eye-in-hand verification without importing realman.realman_env or
realman.open3d_realsense_env.

Usage:
1. Run this script.
2. Left click a point in the wrist camera image.
3. Press "g".
4. The robot TCP moves to the estimated 3D point.

This file is intentionally self-contained so it does not depend on:
- pytransform3d via realman.realman_env
- open3d via realman.open3d_realsense_env
"""

import os
import sys

HAND_EYE_PYTHON = "/home/lyt/miniconda3/envs/hand_eye/bin/python"
SDK_PYTHON_ROOT = "/home/lyt/tju_rllab_realman_lyt/realman/RM_API2/Python"


def ensure_hand_eye_python() -> None:
    # Re-exec before importing compiled packages.
    if os.path.realpath(sys.executable) == os.path.realpath(HAND_EYE_PYTHON):
        return

    if os.environ.get("POINTING_EYE_IN_HAND_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["POINTING_EYE_IN_HAND_REEXEC"] = "1"
    os.execve(
        HAND_EYE_PYTHON,
        [HAND_EYE_PYTHON, os.path.abspath(__file__), *sys.argv[1:]],
        env,
    )


ensure_hand_eye_python()

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SDK_PYTHON_ROOT not in sys.path:
    sys.path.insert(0, SDK_PYTHON_ROOT)

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


WINDOW_NAME = "Eye-in-Hand Verify"
DEFAULT_ROBOT_IP = "192.168.101.19"
DEFAULT_CAMERA_SERIAL = "342522073663"
DEFAULT_HAND_EYE_FRAME = "eef"
DEFAULT_ROTATION_MATRIX = np.array(
    [
        [0.87829927, -0.47804545, 0.00793399],
        [0.47797864, 0.87832556, 0.00897952],
        [-0.01126125, -0.00409443, 0.99992821],
    ],
    dtype=np.float64,
)
DEFAULT_TRANSLATION_VECTOR = np.array(
    [0.00440791, -0.10251168, 0.02981116],
    dtype=np.float64,
)

JOINT_MAX_SPEED_DEG_S = 80.0
SYNC_MOVEP_SPEED_PERCENT = 90
SYNC_MOVEL_SPEED_PERCENT = 60
TCP_TO_EEF_TRANSLATION = np.array([0.0, 0.0, 0.22], dtype=np.float64)


@dataclass
class RobotState:
    pose: np.ndarray
    joint: np.ndarray
    timestamp: float


def rotation_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def T_from_realman_xyzrpy(xyzrpy: np.ndarray) -> np.ndarray:
    x, y, z, rx, ry, rz = xyzrpy

    T = np.eye(4, dtype=np.float64)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ],
        dtype=np.float64,
    )
    Rz = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def realman_xyzrpy_from_T(T: np.ndarray) -> np.ndarray:
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    ry = np.arcsin(np.clip(-T[2, 0], -1.0, 1.0))
    if np.cos(ry) != 0:
        rx = np.arctan2(T[2, 1] / np.cos(ry), T[2, 2] / np.cos(ry))
        rz = np.arctan2(T[1, 0] / np.cos(ry), T[0, 0] / np.cos(ry))
    else:
        rx = 0.0
        rz = np.arctan2(-T[0, 1], T[1, 1])
    return np.array([x, y, z, rx, ry, rz], dtype=np.float64)


def build_tcp_to_eef_transform() -> np.ndarray:
    base_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z(-np.pi / 3.0) @ base_rotation
    transform[:3, 3] = TCP_TO_EEF_TRANSLATION
    return transform


T_TCP2REALMANEEF = build_tcp_to_eef_transform()
T_TCP2REALMANEEF_INV = np.linalg.inv(T_TCP2REALMANEEF)


def pose_eef2tcp(pose_eef: np.ndarray) -> np.ndarray:
    T_eef2base = T_from_realman_xyzrpy(pose_eef)
    T_tcp2base = T_eef2base @ T_TCP2REALMANEEF
    return realman_xyzrpy_from_T(T_tcp2base)


def pose_tcp2eef(pose_tcp: np.ndarray) -> np.ndarray:
    T_tcp2base = T_from_realman_xyzrpy(pose_tcp)
    T_eef2base = T_tcp2base @ T_TCP2REALMANEEF_INV
    return realman_xyzrpy_from_T(T_eef2base)


class PyRealSenseCamera:
    def __init__(self, serial: str):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        # Let exposure settle a little.
        for _ in range(10):
            self.pipeline.wait_for_frames()

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        intrinsic = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale_m = float(depth_sensor.get_depth_scale())
        depth_units_per_meter = 1.0 / depth_scale_m

        self.meta_obs = {
            "size": [intr.height, intr.width],
            "intrinsic": intrinsic.tolist(),
            "depth_scale": depth_units_per_meter,
        }

    def step(self) -> dict:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to read aligned color/depth frames from RealSense.")

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        return {
            "rgb": rgb,
            "depth": depth,
        } | self.meta_obs

    def close(self) -> None:
        self.pipeline.stop()


class SimpleRealmanRobot:
    def __init__(self, robot_ip: str):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(robot_ip, 8080)
        if handle.id <= 0:
            raise RuntimeError(f"Failed to connect robot: {robot_ip}")

        self.arm.rm_set_arm_max_line_speed(0.4)
        self.arm.rm_set_arm_max_line_acc(1.0)
        self.arm.rm_set_arm_max_angular_speed(0.4)
        self.arm.rm_set_arm_max_angular_acc(1.0)
        for joint_idx in range(1, 8):
            self.arm.rm_set_joint_max_speed(joint_idx, JOINT_MAX_SPEED_DEG_S)

    def _get_state_raw(self) -> dict:
        for attempt in range(200):
            ret, state = self.arm.rm_get_current_arm_state()
            if ret == 0 and state is not None:
                return {
                    "pose_eef": np.array(state["pose"], dtype=np.float64),
                    "joint": np.radians(state["joint"]),
                }
            time.sleep(0.02)
        raise RuntimeError("Failed to read current robot state from controller.")

    def get_state(self) -> RobotState:
        raw = self._get_state_raw()
        pose_tcp = pose_eef2tcp(raw["pose_eef"])
        return RobotState(
            pose=pose_tcp,
            joint=raw["joint"],
            timestamp=time.time(),
        )

    def _move_pose_eef(self, pose_eef: np.ndarray, motion: str) -> int:
        move_fn = self.arm.rm_movel if motion == "linear" else self.arm.rm_movej_p
        speed_percent = SYNC_MOVEL_SPEED_PERCENT if motion == "linear" else SYNC_MOVEP_SPEED_PERCENT
        blend_radius = 0 if motion == "linear" else 1

        ret = move_fn(pose_eef, speed_percent, r=blend_radius, connect=0, block=1)
        if ret == 0:
            return ret

        for _ in range(100):
            ret = move_fn(pose_eef, speed_percent, r=blend_radius, connect=0, block=1)
            if ret == 0:
                return ret
            time.sleep(0.02)
        return ret

    def move_pose_tcp(self, pose_tcp: np.ndarray, motion: str = "pose") -> None:
        pose_eef = pose_tcp2eef(pose_tcp)
        ret = self._move_pose_eef(pose_eef, motion)
        if ret != 0:
            raise RuntimeError(f"Robot move failed, ret={ret}, pose_eef={pose_eef}")

    def slow_stop(self) -> int:
        return self.arm.rm_set_arm_slow_stop()

    def emergency_stop(self) -> int:
        return self.arm.rm_set_arm_stop()

    def close(self) -> None:
        self.arm.rm_delete_robot_arm()


def parse_float_list(text: str, expected_len: int) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {len(values)} from: {text}")
    return np.asarray(values, dtype=np.float64)


def build_transform(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    return transform


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_transform_from_json(data: dict) -> tuple[np.ndarray, str]:
    matrix_keys = [
        "T_cam2tool",
        "Tcam2tool",
        "T_cam2eef",
        "Tcam2eef",
        "T_cam2end",
        "Tcam2end",
        "T_camera_to_end_effector",
        "T_cam2tcp",
        "Tcam2tcp",
    ]
    for key in matrix_keys:
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{key} must be a 4x4 matrix, got shape {matrix.shape}")
            return matrix, key

    if "rotation_matrix" in data and "translation_vector" in data:
        rotation_matrix = np.asarray(data["rotation_matrix"], dtype=np.float64)
        translation_vector = np.asarray(data["translation_vector"], dtype=np.float64)
        if rotation_matrix.shape != (3, 3):
            raise ValueError(
                f"rotation_matrix must be 3x3, got shape {rotation_matrix.shape}"
            )
        if translation_vector.shape not in ((3,), (3, 1), (1, 3)):
            raise ValueError(
                "translation_vector must contain 3 values, "
                f"got shape {translation_vector.shape}"
            )
        translation_vector = translation_vector.reshape(3)
        return build_transform(rotation_matrix, translation_vector), "rotation_matrix+translation_vector"

    raise ValueError(
        "Unsupported calibration json. Provide either a 4x4 matrix key such as "
        "T_cam2eef/T_cam2end/T_cam2tool, or both rotation_matrix and translation_vector."
    )


def load_handeye_transform(args: argparse.Namespace) -> tuple[np.ndarray, str, str]:
    if args.calib_json:
        data = load_json(args.calib_json)
        transform, source_key = extract_transform_from_json(data)
        frame = data.get("frame", args.handeye_frame)
        return transform, frame, f"{args.calib_json}:{source_key}"

    if not args.rotation and not args.translation:
        transform = build_transform(DEFAULT_ROTATION_MATRIX, DEFAULT_TRANSLATION_VECTOR)
        return transform, args.handeye_frame, "built-in defaults (2026-05-28 eye-in-hand)"

    if not args.rotation or not args.translation:
        raise ValueError("Provide both --rotation and --translation together.")

    rotation_matrix = parse_float_list(args.rotation, 9).reshape(3, 3)
    translation_vector = parse_float_list(args.translation, 3)
    transform = build_transform(rotation_matrix, translation_vector)
    return transform, args.handeye_frame, "command-line rotation/translation"


def sample_depth_meters(
    depth_image: np.ndarray,
    u: int,
    v: int,
    depth_scale: float,
    patch_radius: int,
) -> tuple[float | None, int, str]:
    height, width = depth_image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return None, 0, "out_of_bounds"

    center_depth_raw = int(depth_image[v, u])
    if center_depth_raw > 0:
        return center_depth_raw / depth_scale, center_depth_raw, "center"

    x0 = max(0, u - patch_radius)
    x1 = min(width, u + patch_radius + 1)
    y0 = max(0, v - patch_radius)
    y1 = min(height, v + patch_radius + 1)
    patch = depth_image[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None, 0, "invalid"

    depth_raw = int(np.median(valid))
    return depth_raw / depth_scale, depth_raw, "patch_median"


def deproject_pixel_to_camera(
    intrinsic_inv: np.ndarray,
    u: int,
    v: int,
    depth_m: float,
) -> np.ndarray:
    pixel = np.array([u, v, 1.0], dtype=np.float64)
    return intrinsic_inv @ (pixel * depth_m)


def get_tool_transform_in_base(robot_pose_tcp: np.ndarray, handeye_frame: str) -> np.ndarray:
    if handeye_frame == "tcp":
        return T_from_realman_xyzrpy(robot_pose_tcp)

    if handeye_frame == "eef":
        pose_eef = pose_tcp2eef(robot_pose_tcp)
        return T_from_realman_xyzrpy(pose_eef)

    raise ValueError(f"Unsupported handeye frame: {handeye_frame}")


def compute_target(
    robot_pose_tcp: np.ndarray,
    depth_image: np.ndarray,
    intrinsic_inv: np.ndarray,
    depth_scale: float,
    handeye_transform: np.ndarray,
    handeye_frame: str,
    pixel: tuple[int, int],
    patch_radius: int,
) -> dict | None:
    u, v = pixel
    depth_m, depth_raw, depth_mode = sample_depth_meters(
        depth_image, u, v, depth_scale, patch_radius
    )
    if depth_m is None:
        return None

    point_cam = deproject_pixel_to_camera(intrinsic_inv, u, v, depth_m)
    point_cam_h = np.append(point_cam, 1.0)

    T_tool2base = get_tool_transform_in_base(robot_pose_tcp, handeye_frame)
    point_tool_h = handeye_transform @ point_cam_h
    point_base_h = T_tool2base @ point_tool_h

    return {
        "pixel": [int(u), int(v)],
        "depth_raw": int(depth_raw),
        "depth_m": float(depth_m),
        "depth_mode": depth_mode,
        "point_cam": point_cam.tolist(),
        "point_tool": point_tool_h[:3].tolist(),
        "point_base": point_base_h[:3].tolist(),
        "robot_pose_tcp_at_click": robot_pose_tcp.tolist(),
    }


def draw_text_block(image: np.ndarray, lines: list[str], origin=(10, 24)) -> None:
    x0, y0 = origin
    line_height = 22
    for idx, line in enumerate(lines):
        y = y0 + idx * line_height
        cv2.putText(image, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(image, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def save_snapshot(
    save_dir: Path,
    rgb_bgr: np.ndarray,
    target: dict | None,
    robot_pose_tcp: np.ndarray,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = save_dir / f"{stamp}_rgb.jpg"
    meta_path = save_dir / f"{stamp}_meta.json"

    cv2.imwrite(str(image_path), rgb_bgr)
    meta = {
        "robot_pose_tcp": robot_pose_tcp.tolist(),
        "target": target,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] rgb: {image_path}")
    print(f"[save] meta: {meta_path}")


def print_target(target: dict | None) -> None:
    if not target:
        print("[target] none")
        return

    point_cam = np.asarray(target["point_cam"], dtype=np.float64)
    point_base = np.asarray(target["point_base"], dtype=np.float64)
    print(
        "[target] pixel=%s depth=%.4fm mode=%s cam_xyz=%s base_xyz=%s"
        % (
            tuple(target["pixel"]),
            target["depth_m"],
            target["depth_mode"],
            np.array2string(point_cam, precision=4, suppress_small=True),
            np.array2string(point_base, precision=4, suppress_small=True),
        )
    )


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Click a wrist-camera pixel and move the robot TCP to the estimated 3D point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="Robot controller IP")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL, help="RealSense serial number")
    parser.add_argument(
        "--calib-json",
        default=None,
        help="Path to a JSON file containing T_cam2eef/T_cam2end/T_cam2tool or rotation_matrix+translation_vector",
    )
    parser.add_argument(
        "--rotation",
        default=None,
        help="Comma-separated 3x3 rotation matrix in row-major order",
    )
    parser.add_argument(
        "--translation",
        default=None,
        help="Comma-separated translation vector tx,ty,tz in meters",
    )
    parser.add_argument(
        "--handeye-frame",
        choices=["eef", "tcp"],
        default=DEFAULT_HAND_EYE_FRAME,
        help="Frame used by the hand-eye result. compute_in_hand.py should normally use eef.",
    )
    parser.add_argument(
        "--motion",
        choices=["linear", "pose"],
        default="pose",
        help="Robot motion mode when executing the move",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Optional extra base-frame Z offset added during the move command",
    )
    parser.add_argument(
        "--depth-patch-radius",
        type=int,
        default=3,
        help="If the clicked depth is invalid, use the median valid depth in this radius",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=1.5,
        help="Reject targets farther than this depth in camera frame",
    )
    parser.add_argument(
        "--min-depth-m",
        type=float,
        default=0.05,
        help="Reject targets closer than this depth in camera frame",
    )
    parser.add_argument(
        "--save-dir",
        default="realman/examples/eye_in_hand_verify_logs",
        help="Directory used by the snapshot hotkey",
    )
    return parser


def main() -> None:
    args = make_argparser().parse_args()
    handeye_transform, handeye_frame, handeye_source = load_handeye_transform(args)

    print("=" * 70)
    print("Eye-in-hand verification")
    print("=" * 70)
    print(f"robot_ip       : {args.robot_ip}")
    print(f"camera_serial  : {args.camera_serial}")
    print(f"handeye_frame  : {handeye_frame}")
    print(f"handeye_source : {handeye_source}")
    print(f"motion         : {args.motion}")
    print(f"z_offset       : {args.z_offset:.4f} m")
    print("-" * 70)
    print("Controls")
    print("  left click : select a pixel and estimate its 3D point")
    print("  g          : move TCP to target")
    print("  c          : clear current target")
    print("  p          : print current target")
    print("  s          : save current RGB frame and target metadata")
    print("  x          : slow stop")
    print("  e          : emergency stop")
    print("  q          : quit")
    print("=" * 70)

    robot = None
    camera = None

    clicked_pixel = None
    clicked_flag = False
    current_target = None
    last_rgb_bgr = None
    last_robot_pose_tcp = None

    def on_mouse(event, x, y, flags, param):
        del flags, param
        nonlocal clicked_pixel, clicked_flag
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixel = (int(x), int(y))
            clicked_flag = True

    try:
        robot = SimpleRealmanRobot(args.robot_ip)
        camera = PyRealSenseCamera(args.camera_serial)

        intrinsic = np.asarray(camera.meta_obs["intrinsic"], dtype=np.float64)
        intrinsic_inv = np.linalg.inv(intrinsic)
        depth_scale = float(camera.meta_obs["depth_scale"])

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        while True:
            robot_state = robot.get_state()
            cam_obs = camera.step()

            rgb = cam_obs["rgb"]
            depth = cam_obs["depth"]
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            last_rgb_bgr = rgb_bgr
            last_robot_pose_tcp = robot_state.pose.copy()

            if clicked_flag and clicked_pixel is not None:
                clicked_flag = False
                target = compute_target(
                    robot_pose_tcp=robot_state.pose.copy(),
                    depth_image=depth,
                    intrinsic_inv=intrinsic_inv,
                    depth_scale=depth_scale,
                    handeye_transform=handeye_transform,
                    handeye_frame=handeye_frame,
                    pixel=clicked_pixel,
                    patch_radius=args.depth_patch_radius,
                )
                if target is None:
                    print(f"[click] invalid depth around pixel {clicked_pixel}")
                    current_target = None
                else:
                    if not (args.min_depth_m <= target["depth_m"] <= args.max_depth_m):
                        print(
                            "[click] depth %.4fm at pixel %s is outside allowed range [%.3f, %.3f]"
                            % (
                                target["depth_m"],
                                tuple(target["pixel"]),
                                args.min_depth_m,
                                args.max_depth_m,
                            )
                        )
                        current_target = None
                    else:
                        current_target = target
                        print_target(current_target)

            overlay_lines = [
                f"frame={handeye_frame} motion={args.motion} z_offset={args.z_offset:.3f}m",
                "click: select target | g: move | c: clear | q: quit",
            ]

            if current_target is None:
                overlay_lines.append("target: none")
            else:
                point_cam = np.asarray(current_target["point_cam"], dtype=np.float64)
                point_base = np.asarray(current_target["point_base"], dtype=np.float64)
                overlay_lines.append(
                    "target pixel=%s depth=%.3fm mode=%s"
                    % (
                        tuple(current_target["pixel"]),
                        current_target["depth_m"],
                        current_target["depth_mode"],
                    )
                )
                overlay_lines.append(
                    "cam xyz=(%.3f, %.3f, %.3f)  base xyz=(%.3f, %.3f, %.3f)"
                    % (
                        point_cam[0],
                        point_cam[1],
                        point_cam[2],
                        point_base[0],
                        point_base[1],
                        point_base[2],
                    )
                )
                u, v = current_target["pixel"]
                cv2.circle(rgb_bgr, (u, v), 5, (0, 0, 255), -1)
                cv2.drawMarker(
                    rgb_bgr,
                    (u, v),
                    color=(0, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=18,
                    thickness=2,
                )

            pose = robot_state.pose
            overlay_lines.append(
                "tcp xyz=(%.3f, %.3f, %.3f) rpy=(%.3f, %.3f, %.3f)"
                % (pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
            )
            draw_text_block(rgb_bgr, overlay_lines)

            cv2.imshow(WINDOW_NAME, rgb_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                current_target = None
                print("[target] cleared")
                continue

            if key == ord("p"):
                print_target(current_target)
                continue

            if key == ord("s"):
                if last_rgb_bgr is not None and last_robot_pose_tcp is not None:
                    save_snapshot(
                        save_dir=Path(args.save_dir),
                        rgb_bgr=last_rgb_bgr,
                        target=current_target,
                        robot_pose_tcp=last_robot_pose_tcp,
                    )
                continue

            if key == ord("x"):
                ret = robot.slow_stop()
                print(f"[robot] slow stop ret={ret}")
                continue

            if key == ord("e"):
                ret = robot.emergency_stop()
                print(f"[robot] emergency stop ret={ret}")
                continue

            if key != ord("g"):
                continue

            if current_target is None:
                print("[move] no target selected")
                continue

            current_state = robot.get_state()
            target_xyz = np.asarray(current_target["point_base"], dtype=np.float64)
            move_xyz = target_xyz.copy()
            move_xyz[2] += args.z_offset

            target_pose = current_state.pose.copy()
            target_pose[:3] = move_xyz

            print(
                "[move] command tcp xyz=%s (raw target=%s)"
                % (
                    np.array2string(move_xyz, precision=4, suppress_small=True),
                    np.array2string(target_xyz, precision=4, suppress_small=True),
                )
            )
            robot.move_pose_tcp(target_pose, motion=args.motion)

    finally:
        cv2.destroyAllWindows()
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
