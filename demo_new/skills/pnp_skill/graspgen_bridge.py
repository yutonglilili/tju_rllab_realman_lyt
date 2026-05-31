"""
Lightweight wrist-camera to GraspGen bridge for the main control environment.

This module intentionally stays independent from the GraspGen Python package so
that the robot control process can live in its own conda environment. The main
process performs all robot/camera-specific preprocessing locally and only sends
the cleaned object point cloud to the GraspGen ZMQ server.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SDK_PYTHON_ROOT = os.path.join(WORKSPACE_ROOT, "realman", "RM_API2", "Python")

for path in (WORKSPACE_ROOT, SDK_PYTHON_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from realman.realman_env import T_from_realman_xyzrpy, realman_xyzrpy_from_T, pose_tcp2eef

try:
    import msgpack
    import msgpack_numpy
    import zmq

    msgpack_numpy.patch()
except ImportError as exc:  # pragma: no cover - import is environment-specific
    msgpack = None
    zmq = None
    _ZMQ_IMPORT_ERROR = exc
else:
    _ZMQ_IMPORT_ERROR = None


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


@dataclass
class WristHandEyeConfig:
    transform_camera_to_tool: np.ndarray
    handeye_frame: str = DEFAULT_HAND_EYE_FRAME
    source: str = "built-in defaults"


@dataclass
class WristProcessingConfig:
    min_depth_m: float = 0.10
    max_depth_m: float = 1.20
    depth_patch_radius: int = 3
    click_seed_radius_px: int = 5
    seed_search_radius_px: int = 15
    region_grow_neighbor_radius_px: int = 2
    region_grow_3d_threshold_m: float = 0.018
    region_grow_depth_threshold_m: float = 0.030
    region_grow_color_threshold: float = 140.0
    region_grow_max_seed_distance_m: float = 0.24
    min_mask_pixels: int = 120
    mask_kernel_size: int = 3
    table_height_percentile: float = 8.0
    table_remove_margin_m: float = 0.008
    max_object_points: int = 4096
    object_voxel_size: float = 0.003
    max_scene_points: int = 8192
    scene_voxel_size: float = 0.004


@dataclass
class GraspFilterConfig:
    grasp_threshold: float = -1.0
    num_grasps: int = 200
    topk_num_grasps: int = 50
    max_candidates: int = 5
    candidate_pregrasp_offset_m: float = 0.10
    direction_rule_target_dir_camera: tuple[float, float, float] = (0.0, 0.64, 0.77)
    direction_rule_max_angle_deg: float = 35.0
    direction_rule_min_forward_component: float = 0.30
    direction_rule_min_down_component: float = 0.20
    direction_rule_max_lateral_component: float = 0.45
    # GraspGen base_link -> Realman TCP correction (see build_grasp_to_tcp_transform).
    # depth = gripper depth (robotiq_2f_140 = 0.195 m); roll about the approach axis.
    grasp_to_tcp_depth_m: float = 0.195
    grasp_to_tcp_roll_deg: float = 0.0


@dataclass
class ClickSelection:
    mask: np.ndarray
    click_pixel: tuple[int, int]
    seed_pixel: tuple[int, int]
    seed_depth_m: float
    point_cam: np.ndarray
    point_base: np.ndarray
    seed_radius_px: int = 0
    num_seed_pixels: int = 1
    depth_mode: str = "unknown"


@dataclass
class WristGraspResult:
    success: bool
    object_pc_base: np.ndarray
    scene_pc_base: np.ndarray
    object_colors: np.ndarray | None
    scene_colors: np.ndarray | None
    grasp_pose_pool_base: np.ndarray
    grasp_pregrasp_pool_base: np.ndarray
    grasp_pose_pool_scores: np.ndarray
    grasp_eef_xyzrpy_pool: np.ndarray  # (N, 6) eef2base xyzrpy — 定版最终输出
    all_grasps_base: np.ndarray
    all_scores: np.ndarray
    direction_rule_keep_mask: np.ndarray
    table_height_m: float
    click_pixel: tuple[int, int]
    seed_pixel: tuple[int, int]
    seed_depth_m: float
    seed_radius_px: int
    num_seed_pixels: int
    depth_mode: str
    mask: np.ndarray
    debug_info: dict[str, Any]
    error: str | None = None


class GraspGenClientBridge:
    """Small ZMQ client used by the main control process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout_ms: int = 60_000,
        wait_for_server: bool = True,
        retry_interval_s: float = 2.0,
    ) -> None:
        if _ZMQ_IMPORT_ERROR is not None:  # pragma: no cover - runtime dependency
            raise ImportError(
                "pyzmq, msgpack and msgpack_numpy are required in the main control "
                "environment to use the GraspGen ZMQ bridge."
            ) from _ZMQ_IMPORT_ERROR

        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = int(timeout_ms)
        self._ctx = zmq.Context()
        self._socket = None
        self._server_metadata: dict[str, Any] | None = None

        if wait_for_server:
            self._wait_for_server(retry_interval_s=retry_interval_s)

    def _create_socket(self):
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._addr)
        return sock

    def _ensure_connected(self) -> None:
        if self._socket is None:
            self._socket = self._create_socket()

    def _wait_for_server(self, retry_interval_s: float) -> None:
        while True:
            try:
                self._socket = self._create_socket()
                self._server_metadata = self._request({"action": "metadata"})
                return
            except Exception:
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
                time.sleep(float(retry_interval_s))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        raw = self._socket.recv()
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    @property
    def server_metadata(self) -> dict[str, Any] | None:
        return self._server_metadata

    def health_check(self) -> bool:
        try:
            response = self._request({"action": "health"})
        except Exception:
            return False
        return response.get("status") == "ok"

    def infer(
        self,
        object_pc: np.ndarray,
        *,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = 50,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        object_pc = np.asarray(object_pc, dtype=np.float32)
        if object_pc.ndim != 2 or object_pc.shape[1] != 3:
            raise ValueError(f"object_pc must be (N, 3), got {object_pc.shape}")

        response = self._request(
            {
                "action": "infer",
                "point_cloud": object_pc,
                "grasp_threshold": float(grasp_threshold),
                "num_grasps": int(num_grasps),
                "topk_num_grasps": int(topk_num_grasps),
                "min_grasps": int(min_grasps),
                "max_tries": int(max_tries),
                "remove_outliers": bool(remove_outliers),
            }
        )
        # msgpack_numpy may hand back read-only ndarray views; return writable copies
        # so downstream filtering code can safely normalize homogeneous coordinates.
        grasps = np.array(response["grasps"], dtype=np.float32, copy=True)
        confidences = np.array(response["confidences"], dtype=np.float32, copy=True)
        return grasps, confidences

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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


def extract_transform_from_json(data: dict[str, Any]) -> tuple[np.ndarray, str]:
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
        translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(-1)
        if translation_vector.shape != (3,):
            raise ValueError("translation_vector must contain 3 values.")
        return (
            build_transform(rotation_matrix, translation_vector),
            "rotation_matrix+translation_vector",
        )

    raise ValueError(
        "Unsupported hand-eye json. Provide a 4x4 matrix key such as T_cam2eef "
        "or both rotation_matrix and translation_vector."
    )


def load_wrist_handeye_config(
    *,
    calib_json: str | None = None,
    rotation: str | None = None,
    translation: str | None = None,
    handeye_frame: str = DEFAULT_HAND_EYE_FRAME,
) -> WristHandEyeConfig:
    if calib_json:
        with open(calib_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        transform, source_key = extract_transform_from_json(data)
        frame = data.get("frame", handeye_frame)
        return WristHandEyeConfig(
            transform_camera_to_tool=transform,
            handeye_frame=frame,
            source=f"{calib_json}:{source_key}",
        )

    if rotation and translation:
        rotation_matrix = parse_float_list(rotation, 9).reshape(3, 3)
        translation_vector = parse_float_list(translation, 3)
        return WristHandEyeConfig(
            transform_camera_to_tool=build_transform(rotation_matrix, translation_vector),
            handeye_frame=handeye_frame,
            source="command-line rotation/translation",
        )

    return WristHandEyeConfig(
        transform_camera_to_tool=build_transform(
            DEFAULT_ROTATION_MATRIX,
            DEFAULT_TRANSLATION_VECTOR,
        ),
        handeye_frame=handeye_frame,
        source="built-in defaults",
    )


def project_depth_to_xyz(
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
) -> np.ndarray:
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    height, width = depth_m.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_m
    x = (x_coords.astype(np.float32) - cx) * z / fx
    y = (y_coords.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def transform_points_camera_to_base(
    points_camera: np.ndarray,
    handeye_transform: np.ndarray,
    tool_to_base: np.ndarray,
) -> np.ndarray:
    points_flat = points_camera.reshape(-1, 3).astype(np.float64)
    r_cam_to_tool = handeye_transform[:3, :3]
    t_cam_to_tool = handeye_transform[:3, 3]
    r_tool_to_base = tool_to_base[:3, :3]
    t_tool_to_base = tool_to_base[:3, 3]

    points_tool = points_flat @ r_cam_to_tool.T + t_cam_to_tool[None]
    points_base = points_tool @ r_tool_to_base.T + t_tool_to_base[None]
    return points_base.reshape(points_camera.shape).astype(np.float32)


def get_tool_transform_in_base(
    robot_pose_tcp: np.ndarray,
    handeye_frame: str,
) -> np.ndarray:
    if handeye_frame == "tcp":
        return T_from_realman_xyzrpy(robot_pose_tcp)
    if handeye_frame == "eef":
        pose_eef = pose_tcp2eef(robot_pose_tcp)
        return T_from_realman_xyzrpy(pose_eef)
    raise ValueError(f"Unsupported hand-eye frame: {handeye_frame}")


def estimate_table_height(
    points_base: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float,
) -> float:
    valid_z = points_base[..., 2][valid_mask]
    if valid_z.size == 0:
        raise ValueError("No valid base-frame points are available for tabletop estimation.")

    percentile = float(np.clip(percentile, 0.0, 100.0))
    rough_table_z = float(np.percentile(valid_z, percentile))
    band_half_width = 0.015
    band = valid_z[np.abs(valid_z - rough_table_z) <= band_half_width]
    if band.size >= 50:
        return float(np.median(band))
    return rough_table_z


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(component_areas)) + 1
    return labels == largest_idx


def clean_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(int(kernel_size), 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return largest_connected_component(mask_u8.astype(bool))


def sample_depth_meters(
    depth_m: np.ndarray,
    u: int,
    v: int,
    patch_radius: int,
) -> tuple[float | None, str]:
    height, width = depth_m.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return None, "out_of_bounds"

    center_depth_m = float(depth_m[v, u])
    if np.isfinite(center_depth_m) and center_depth_m > 0.0:
        return center_depth_m, "center"

    x0 = max(0, u - patch_radius)
    x1 = min(width, u + patch_radius + 1)
    y0 = max(0, v - patch_radius)
    y1 = min(height, v + patch_radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None, "invalid"
    return float(np.median(valid)), "patch_median"


def collect_valid_pixels_in_disk(
    valid_mask: np.ndarray,
    u: int,
    v: int,
    radius: int,
) -> list[tuple[int, int]]:
    height, width = valid_mask.shape
    if not (0 <= u < width and 0 <= v < height):
        return []

    radius = max(int(radius), 0)
    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    ys, xs = np.nonzero(valid_mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return []

    xs = xs + x0
    ys = ys + y0
    dist_sq = (xs - u) ** 2 + (ys - v) ** 2
    keep = dist_sq <= radius * radius
    if not np.any(keep):
        return []

    xs = xs[keep]
    ys = ys[keep]
    dist_sq = dist_sq[keep]
    order = np.lexsort((ys, xs, dist_sq))
    return [(int(xs[idx]), int(ys[idx])) for idx in order]


def find_seed_pixels_near_click(
    valid_mask: np.ndarray,
    u: int,
    v: int,
    preferred_radius: int,
    max_radius: int,
) -> tuple[list[tuple[int, int]], int, str]:
    preferred_radius = max(int(preferred_radius), 0)
    max_radius = max(int(max_radius), preferred_radius)

    for radius in range(preferred_radius, max_radius + 1):
        pixels = collect_valid_pixels_in_disk(valid_mask, u, v, radius)
        if len(pixels) == 0:
            continue
        if radius == preferred_radius:
            return pixels, radius, "seed_disk"
        return pixels, radius, f"seed_disk_expand_r{radius}"
    return [], -1, "invalid"


def grow_mask_from_seed(
    points_camera: np.ndarray,
    rgb: np.ndarray,
    candidate_mask: np.ndarray,
    seed_pixels: list[tuple[int, int]],
    cfg: WristProcessingConfig,
) -> np.ndarray:
    height, width = candidate_mask.shape
    if len(seed_pixels) == 0:
        raise ValueError("At least one valid seed pixel is required for region growing.")

    region_mask = np.zeros((height, width), dtype=bool)
    queued = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    seed_points = np.asarray(
        [points_camera[seed_v, seed_u] for seed_u, seed_v in seed_pixels],
        dtype=np.float32,
    )
    seed_points = seed_points[np.all(np.isfinite(seed_points), axis=1)]
    if len(seed_points) == 0:
        raise ValueError("The selected seed region does not contain valid 3D points.")

    seed_color = np.asarray(
        [rgb[seed_v, seed_u] for seed_u, seed_v in seed_pixels],
        dtype=np.float32,
    ).mean(axis=0)

    for seed_u, seed_v in seed_pixels:
        if not candidate_mask[seed_v, seed_u]:
            continue
        queue.append((seed_u, seed_v))
        queued[seed_v, seed_u] = True
        region_mask[seed_v, seed_u] = True

    neighbor_radius_px = max(int(cfg.region_grow_neighbor_radius_px), 1)

    while queue:
        u, v = queue.popleft()
        current_point = np.asarray(points_camera[v, u], dtype=np.float32)
        current_color = rgb[v, u].astype(np.float32)

        for dv in range(-neighbor_radius_px, neighbor_radius_px + 1):
            for du in range(-neighbor_radius_px, neighbor_radius_px + 1):
                if du == 0 and dv == 0:
                    continue

                nu = u + du
                nv = v + dv
                if not (0 <= nu < width and 0 <= nv < height):
                    continue
                if queued[nv, nu]:
                    continue
                if not candidate_mask[nv, nu]:
                    continue

                neighbor_point = np.asarray(points_camera[nv, nu], dtype=np.float32)
                if not np.all(np.isfinite(neighbor_point)):
                    continue
                if (
                    np.linalg.norm(neighbor_point - current_point)
                    > float(cfg.region_grow_3d_threshold_m)
                ):
                    continue
                if (
                    abs(float(neighbor_point[2] - current_point[2]))
                    > float(cfg.region_grow_depth_threshold_m)
                ):
                    continue
                if (
                    np.min(np.linalg.norm(seed_points - neighbor_point[None], axis=1))
                    > float(cfg.region_grow_max_seed_distance_m)
                ):
                    continue

                neighbor_color = rgb[nv, nu].astype(np.float32)
                local_color_distance = float(np.linalg.norm(neighbor_color - current_color))
                seed_color_distance = float(np.linalg.norm(neighbor_color - seed_color))
                if (
                    local_color_distance > float(cfg.region_grow_color_threshold)
                    and seed_color_distance > float(cfg.region_grow_color_threshold)
                ):
                    continue

                region_mask[nv, nu] = True
                queued[nv, nu] = True
                queue.append((nu, nv))

    return region_mask


def build_click_selection(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    points_camera: np.ndarray,
    points_base: np.ndarray,
    candidate_mask: np.ndarray,
    click_pixel: tuple[int, int],
    cfg: WristProcessingConfig,
) -> ClickSelection:
    click_u, click_v = click_pixel
    sampled_depth_m, depth_mode = sample_depth_meters(
        depth_m,
        click_u,
        click_v,
        patch_radius=cfg.depth_patch_radius,
    )
    seed_pixels, seed_radius_px, seed_mode = find_seed_pixels_near_click(
        candidate_mask,
        click_u,
        click_v,
        preferred_radius=cfg.click_seed_radius_px,
        max_radius=cfg.seed_search_radius_px,
    )
    if len(seed_pixels) == 0:
        raise ValueError(
            "Could not find a valid non-table seed point near the click."
        )
    if sampled_depth_m is None:
        depth_mode = seed_mode
    elif depth_mode != "center":
        depth_mode = f"{depth_mode}+{seed_mode}"

    seed_pixel = seed_pixels[0]
    seed_u, seed_v = seed_pixel
    mask = grow_mask_from_seed(
        points_camera=points_camera,
        rgb=rgb,
        candidate_mask=candidate_mask,
        seed_pixels=seed_pixels,
        cfg=cfg,
    )
    mask = clean_binary_mask(mask, kernel_size=cfg.mask_kernel_size)
    if int(mask.sum()) < int(cfg.min_mask_pixels):
        raise ValueError(
            f"Only {int(mask.sum())} pixels remained after click-based mask extraction; "
            f"expected at least {int(cfg.min_mask_pixels)}."
        )

    point_cam = np.asarray(points_camera[seed_v, seed_u], dtype=np.float32)
    point_base = np.asarray(points_base[seed_v, seed_u], dtype=np.float32)
    return ClickSelection(
        mask=mask,
        click_pixel=(int(click_u), int(click_v)),
        seed_pixel=(int(seed_u), int(seed_v)),
        seed_depth_m=float(point_cam[2]),
        point_cam=point_cam,
        point_base=point_base,
        seed_radius_px=int(seed_radius_px),
        num_seed_pixels=int(len(seed_pixels)),
        depth_mode=depth_mode,
    )


def downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) == 0:
        return points.astype(np.float32), colors

    points_out = points.astype(np.float32)
    colors_out = None if colors is None else colors.copy()

    if voxel_size > 0:
        voxel_ids = np.floor(points_out / float(voxel_size)).astype(np.int64)
        unique_voxels, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
        counts = np.bincount(inverse)

        point_sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
        np.add.at(point_sums, inverse, points_out.astype(np.float64))
        points_out = (point_sums / counts[:, None]).astype(np.float32)

        if colors_out is not None:
            color_sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
            np.add.at(color_sums, inverse, colors_out.astype(np.float64))
            colors_out = np.clip(color_sums / counts[:, None], 0, 255).astype(np.uint8)

    if max_points is not None and len(points_out) > max_points:
        indices = np.random.choice(len(points_out), size=max_points, replace=False)
        points_out = points_out[indices]
        if colors_out is not None:
            colors_out = colors_out[indices]

    return points_out, colors_out


def sort_by_score_desc(
    grasps: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    return grasps[order], scores[order]


def normalize_vec3(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(3)
    norm = max(float(np.linalg.norm(v)), 1e-8)
    return (v / norm).astype(np.float32)


def compute_camera_direction_rule_metrics(
    grasp_poses_camera: np.ndarray,
    target_dir_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(grasp_poses_camera) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, empty

    approach_dirs = np.asarray(grasp_poses_camera[:, :3, 2], dtype=np.float32)
    approach_norms = np.linalg.norm(approach_dirs, axis=1, keepdims=True)
    approach_dirs = approach_dirs / np.clip(approach_norms, 1e-8, None)

    target_dir = normalize_vec3(target_dir_camera)
    cosines = np.clip(approach_dirs @ target_dir, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosines)).astype(np.float32)
    lateral_components = np.abs(approach_dirs[:, 0]).astype(np.float32)
    down_components = approach_dirs[:, 1].astype(np.float32)
    forward_components = approach_dirs[:, 2].astype(np.float32)
    return angles_deg, lateral_components, down_components, forward_components


def filter_grasps_by_camera_direction_rule(
    grasp_poses_camera: np.ndarray,
    cfg: GraspFilterConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    (
        angles_deg,
        lateral_components,
        down_components,
        forward_components,
    ) = compute_camera_direction_rule_metrics(
        grasp_poses_camera,
        target_dir_camera=np.asarray(cfg.direction_rule_target_dir_camera, dtype=np.float32),
    )
    keep_mask = (
        (angles_deg <= float(cfg.direction_rule_max_angle_deg))
        & (forward_components >= float(cfg.direction_rule_min_forward_component))
        & (down_components >= float(cfg.direction_rule_min_down_component))
        & (lateral_components <= float(cfg.direction_rule_max_lateral_component))
    )
    metrics = {
        "angles_deg": angles_deg,
        "lateral_components": lateral_components,
        "down_components": down_components,
        "forward_components": forward_components,
    }
    return keep_mask.astype(bool), metrics


def transform_grasp_poses(T_out_from_in: np.ndarray, grasp_poses: np.ndarray) -> np.ndarray:
    if len(grasp_poses) == 0:
        return grasp_poses.astype(np.float32)
    return np.einsum("ij,njk->nik", T_out_from_in, grasp_poses).astype(np.float32)


def build_grasp_to_tcp_transform(depth_m: float, roll_deg: float = 0.0) -> np.ndarray:
    """Constant correction from a GraspGen grasp pose to the Realman TCP frame.

    GraspGen outputs poses in the gripper base_link frame (approach = local +z,
    finger closing = local +x) with the origin at the gripper mount. The Realman
    TCP / gripper-center frame instead uses approach = local +x with the origin at
    the contact center. Apply as:

        T_tcp2base = grasp2base @ build_grasp_to_tcp_transform(...)

    This mirrors `build_grasp_to_tcp_transform` in
    GraspGen/scripts/demo_wrist_camera_graspgen.py (verified correct on the real
    robot). `roll_deg` rotates the TCP frame about the (shared) approach axis to
    match the physical finger-closing orientation.
    """
    C = np.eye(4, dtype=np.float64)
    # Origin -> gripper center: tool tcp lies along the GraspGen approach (+z) by
    # the gripper depth. Expressed in base_link frame, so it is not rotated below.
    C[:3, 3] = [0.0, 0.0, float(depth_m)]
    # Columns are the Realman TCP basis vectors expressed in the GraspGen frame:
    #   TCP +x (approach) = GraspGen +z
    #   TCP +y            = GraspGen +x   (default finger-closing assignment)
    #   TCP +z            = GraspGen +y
    R_align = np.array(
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    roll = np.deg2rad(float(roll_deg))
    # Roll about TCP +x (= approach); preserves the approach mapping.
    Rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(roll), -np.sin(roll)],
         [0.0, np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    C[:3, :3] = R_align @ Rx
    return C.astype(np.float32)


def level_gripper_x_axis(grasp_pose_base: np.ndarray, grasp_depth_m: float = 0.195) -> np.ndarray:
    """对 GraspGen base_link 姿态做夹爪 x 轴水平矫正。

    GraspGen 夹爪坐标系:
      z = approach（接近方向）
      x = 闭合方向（两指连线）
      y = 夹爪平面法线

    矫正方式: 绕 y 轴旋转使 x 轴水平（x[2]=0）。
    以 TCP（手指中心）为旋转中心，保证矫正后夹爪中心不移动。
    """
    R = np.array(grasp_pose_base[:3, :3], dtype=np.float64)
    origin = np.array(grasp_pose_base[:3, 3], dtype=np.float64)
    x_axis = R[:, 0].copy()
    z_axis = R[:, 2].copy()

    # 绕 y 轴旋转角 θ 使 x_new[2] = 0
    # x_new = cos(θ)*x - sin(θ)*z => x_new[2] = cos(θ)*x[2] - sin(θ)*z[2] = 0
    theta = np.arctan2(x_axis[2], z_axis[2])

    c, s = np.cos(theta), np.sin(theta)
    x_new = c * x_axis - s * z_axis
    z_new = s * x_axis + c * z_axis

    # 选择 approach 朝下的解（z_new[2] < 0），否则取另一个解（加 π）
    if z_new[2] > 0:
        theta = theta + np.pi
        c, s = np.cos(theta), np.sin(theta)
        x_new = c * x_axis - s * z_axis
        z_new = s * x_axis + c * z_axis
    # y 不变

    R_new = R.copy()
    R_new[:, 0] = x_new
    R_new[:, 2] = z_new

    # 以 TCP 为旋转中心：TCP = origin + depth * z_old，保持不动
    tcp_pos = origin + grasp_depth_m * z_axis
    new_origin = tcp_pos - grasp_depth_m * z_new

    result = grasp_pose_base.copy().astype(np.float64)
    result[:3, :3] = R_new
    result[:3, 3] = new_origin
    return result.astype(np.float32)


def build_pregrasp_pose_from_grasp(
    grasp_pose: np.ndarray,
    retreat_m: float,
) -> np.ndarray:
    """Retreat behind the grasp along the approach axis to get a pre-grasp pose.

    Expects `grasp_pose` in the Realman TCP convention (approach = local +x), so
    the retreat is along local -x.
    """
    retreat_transform = np.eye(4, dtype=np.float32)
    retreat_transform[0, 3] = -float(retreat_m)
    return (grasp_pose @ retreat_transform).astype(np.float32)


def infer_pick_grasp_candidates_from_wrist(
    wrist_obs: dict[str, Any],
    click_point_2d: tuple[int, int],
    robot_pose_tcp: np.ndarray,
    handeye_config: WristHandEyeConfig,
    graspgen_client: GraspGenClientBridge,
    *,
    processing_cfg: WristProcessingConfig,
    grasp_filter_cfg: GraspFilterConfig,
) -> WristGraspResult:
    rgb = np.asarray(wrist_obs["rgb"], dtype=np.uint8)
    depth_raw = np.asarray(wrist_obs["depth"])
    intrinsic_matrix = np.asarray(wrist_obs["intrinsic"], dtype=np.float64)
    depth_scale = float(wrist_obs["depth_scale"])
    depth_m = depth_raw.astype(np.float32) / depth_scale

    valid_depth_mask = (
        np.isfinite(depth_m)
        & (depth_m > float(processing_cfg.min_depth_m))
        & (depth_m < float(processing_cfg.max_depth_m))
    )

    points_camera = project_depth_to_xyz(depth_m, intrinsic_matrix)
    tool_to_base = get_tool_transform_in_base(
        np.asarray(robot_pose_tcp, dtype=np.float64),
        handeye_config.handeye_frame,
    )
    points_base = transform_points_camera_to_base(
        points_camera,
        handeye_config.transform_camera_to_tool,
        tool_to_base,
    )
    table_height_m = estimate_table_height(
        points_base,
        valid_mask=valid_depth_mask,
        percentile=processing_cfg.table_height_percentile,
    )
    non_table_mask = valid_depth_mask & (
        points_base[..., 2] > float(table_height_m + processing_cfg.table_remove_margin_m)
    )

    selection = build_click_selection(
        rgb=rgb,
        depth_m=depth_m,
        points_camera=points_camera,
        points_base=points_base,
        candidate_mask=non_table_mask,
        click_pixel=tuple(int(v) for v in click_point_2d),
        cfg=processing_cfg,
    )

    mask = selection.mask & non_table_mask
    if int(mask.sum()) < int(processing_cfg.min_mask_pixels):
        raise ValueError(
            "The final click-based object mask became too small after tabletop removal."
        )

    object_mask = mask & valid_depth_mask
    scene_mask = valid_depth_mask & ~object_mask

    object_pc_base = np.asarray(points_base[object_mask], dtype=np.float32)
    scene_pc_base = np.asarray(points_base[scene_mask], dtype=np.float32)
    object_colors = np.asarray(rgb[object_mask], dtype=np.uint8)
    scene_colors = np.asarray(rgb[scene_mask], dtype=np.uint8)

    object_pc_base, object_colors = downsample_point_cloud(
        object_pc_base,
        object_colors,
        voxel_size=processing_cfg.object_voxel_size,
        max_points=processing_cfg.max_object_points,
    )
    scene_pc_base, scene_colors = downsample_point_cloud(
        scene_pc_base,
        scene_colors,
        voxel_size=processing_cfg.scene_voxel_size,
        max_points=processing_cfg.max_scene_points,
    )

    if len(object_pc_base) < 32:
        raise ValueError(
            f"Only {len(object_pc_base)} object points remain after downsampling."
        )

    grasps_base, scores = graspgen_client.infer(
        object_pc_base,
        grasp_threshold=grasp_filter_cfg.grasp_threshold,
        num_grasps=grasp_filter_cfg.num_grasps,
        topk_num_grasps=grasp_filter_cfg.topk_num_grasps,
    )

    if len(grasps_base) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return WristGraspResult(
            success=False,
            object_pc_base=object_pc_base,
            scene_pc_base=scene_pc_base,
            object_colors=object_colors,
            scene_colors=scene_colors,
            grasp_pose_pool_base=np.zeros((0, 4, 4), dtype=np.float32),
            grasp_pregrasp_pool_base=np.zeros((0, 4, 4), dtype=np.float32),
            grasp_pose_pool_scores=empty,
            grasp_eef_xyzrpy_pool=np.zeros((0, 6), dtype=np.float64),
            all_grasps_base=np.zeros((0, 4, 4), dtype=np.float32),
            all_scores=empty,
            direction_rule_keep_mask=np.zeros((0,), dtype=bool),
            table_height_m=float(table_height_m),
            click_pixel=selection.click_pixel,
            seed_pixel=selection.seed_pixel,
            seed_depth_m=selection.seed_depth_m,
            seed_radius_px=selection.seed_radius_px,
            num_seed_pixels=selection.num_seed_pixels,
            depth_mode=selection.depth_mode,
            mask=mask,
            debug_info={
                "message": "GraspGen returned no grasps",
                "table_height_m": float(table_height_m),
            },
            error="GraspGen returned no grasps",
        )

    # Ensure the arrays are writable before in-place cleanup like `[:, 3, 3] = 1.0`.
    grasps_base = np.array(grasps_base, dtype=np.float32, copy=True)
    scores = np.array(scores, dtype=np.float32, copy=True)
    grasps_base[:, 3, 3] = 1.0
    grasps_base, scores = sort_by_score_desc(grasps_base, scores)

    T_base_from_camera = tool_to_base @ handeye_config.transform_camera_to_tool
    T_camera_from_base = np.linalg.inv(T_base_from_camera).astype(np.float32)
    grasps_camera = transform_grasp_poses(T_camera_from_base, grasps_base)
    direction_keep_mask, direction_metrics = filter_grasps_by_camera_direction_rule(
        grasps_camera,
        cfg=grasp_filter_cfg,
    )

    # === 定版流程 ===
    # 步骤 2 已完成: direction rule 在矫正前过滤（上面的 filter_grasps_by_camera_direction_rule）
    # 步骤 3: 置信度排序 + 取 top-5（在通过朝向筛选的姿态里）
    kept_indices = np.flatnonzero(direction_keep_mask)
    if len(kept_indices) > 0:
        # grasps_base 已按 score 降序排列，kept_indices 保持该顺序
        kept_indices = kept_indices[: int(grasp_filter_cfg.max_candidates)]
        top_grasps_base = grasps_base[kept_indices].copy()
        top_scores = scores[kept_indices].copy()
    else:
        top_grasps_base = np.zeros((0, 4, 4), dtype=np.float32)
        top_scores = np.zeros((0,), dtype=np.float32)

    # 步骤 4: 对 top-5 各做"夹爪 x 轴水平矫正"（在 GraspGen base_link 坐标系）
    grasp_depth_m = float(grasp_filter_cfg.grasp_to_tcp_depth_m)
    leveled_grasps_base = np.asarray(
        [level_gripper_x_axis(g, grasp_depth_m=grasp_depth_m) for g in top_grasps_base],
        dtype=np.float32,
    ) if len(top_grasps_base) > 0 else np.zeros((0, 4, 4), dtype=np.float32)

    # 步骤 6: 格式转换 — 矫正后走 grasp_to_tcp → realman_xyzrpy_from_T → pose_tcp2eef
    grasp_to_tcp = build_grasp_to_tcp_transform(
        depth_m=grasp_filter_cfg.grasp_to_tcp_depth_m,
        roll_deg=grasp_filter_cfg.grasp_to_tcp_roll_deg,
    )

    eef_xyzrpy_pool = []
    grasp_pose_pool_base = []
    for leveled_g in leveled_grasps_base:
        T_tcp2base = (leveled_g @ grasp_to_tcp).astype(np.float64)
        tcp_xyzrpy = realman_xyzrpy_from_T(T_tcp2base)
        eef_xyzrpy = pose_tcp2eef(tcp_xyzrpy)
        eef_xyzrpy_pool.append(eef_xyzrpy)
        grasp_pose_pool_base.append(T_tcp2base.astype(np.float32))

    if len(eef_xyzrpy_pool) > 0:
        grasp_eef_xyzrpy_pool = np.asarray(eef_xyzrpy_pool, dtype=np.float64)
        grasp_pose_pool_base = np.asarray(grasp_pose_pool_base, dtype=np.float32)
    else:
        grasp_eef_xyzrpy_pool = np.zeros((0, 6), dtype=np.float64)
        grasp_pose_pool_base = np.zeros((0, 4, 4), dtype=np.float32)

    # 步骤 5: 仍按原始置信度顺序（已保持）
    grasp_pose_pool_scores = top_scores

    grasp_pregrasp_pool_base = np.asarray(
        [
            build_pregrasp_pose_from_grasp(
                grasp_pose,
                retreat_m=grasp_filter_cfg.candidate_pregrasp_offset_m,
            )
            for grasp_pose in grasp_pose_pool_base
        ],
        dtype=np.float32,
    )
    if len(grasp_pregrasp_pool_base) == 0:
        grasp_pregrasp_pool_base = np.zeros((0, 4, 4), dtype=np.float32)

    debug_info = {
        "table_height_m": float(table_height_m),
        "click_pixel": [int(selection.click_pixel[0]), int(selection.click_pixel[1])],
        "seed_pixel": [int(selection.seed_pixel[0]), int(selection.seed_pixel[1])],
        "seed_depth_m": float(selection.seed_depth_m),
        "seed_radius_px": int(selection.seed_radius_px),
        "num_seed_pixels": int(selection.num_seed_pixels),
        "depth_mode": selection.depth_mode,
        "num_object_points": int(len(object_pc_base)),
        "num_scene_points": int(len(scene_pc_base)),
        "num_all_grasps": int(len(grasps_base)),
        "num_direction_rule_kept": int(direction_keep_mask.sum()),
        "graspgen_pose_frame": "eef2base_xyzrpy (leveled → grasp_to_tcp → tcp2eef)",
        "grasp_to_tcp_depth_m": float(grasp_filter_cfg.grasp_to_tcp_depth_m),
        "grasp_to_tcp_roll_deg": float(grasp_filter_cfg.grasp_to_tcp_roll_deg),
        "direction_rule_metrics": {
            key: value.astype(float).tolist()
            for key, value in direction_metrics.items()
        },
    }

    return WristGraspResult(
        success=len(grasp_pose_pool_base) > 0,
        object_pc_base=object_pc_base,
        scene_pc_base=scene_pc_base,
        object_colors=object_colors,
        scene_colors=scene_colors,
        grasp_pose_pool_base=grasp_pose_pool_base,
        grasp_pregrasp_pool_base=grasp_pregrasp_pool_base,
        grasp_pose_pool_scores=grasp_pose_pool_scores,
        grasp_eef_xyzrpy_pool=grasp_eef_xyzrpy_pool,
        all_grasps_base=grasps_base,
        all_scores=scores,
        direction_rule_keep_mask=direction_keep_mask,
        table_height_m=float(table_height_m),
        click_pixel=selection.click_pixel,
        seed_pixel=selection.seed_pixel,
        seed_depth_m=selection.seed_depth_m,
        seed_radius_px=selection.seed_radius_px,
        num_seed_pixels=selection.num_seed_pixels,
        depth_mode=selection.depth_mode,
        mask=mask,
        debug_info=debug_info,
        error=None if len(grasp_pose_pool_base) > 0 else "No grasps survived direction filtering",
    )
