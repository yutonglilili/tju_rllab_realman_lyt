from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ACT_TRAIN_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ACT_TRAIN_ROOT.parent
TOOLKIT_ROOT = WORKSPACE_ROOT / "iffyuan-XArm-Toolkit-main" / "iffyuan-XArm-Toolkit-main"
LEROBOT_SRC = TOOLKIT_ROOT / "lerobot" / "src"

DEFAULT_MAX_GRIPPER_WIDTH = 0.09
DEFAULT_ROBOT_TYPE = "realman"
DEFAULT_DEPLOY_CONFIG_NAME = "deploy_config.json"

POSE_STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
JOINT_STATE_NAMES = [f"joint_{idx}" for idx in range(1, 8)] + ["gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def bootstrap_python_path() -> None:
    for path in (WORKSPACE_ROOT, TOOLKIT_ROOT, LEROBOT_SRC):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


bootstrap_python_path()


@dataclass(frozen=True)
class RealmanDatasetFields:
    rgb: str = "rgb"
    depth: str = "depth"
    pose: str = "pose"
    joint: str = "joint"
    action: str = "action"
    gripper_width: str = "gripper_width"
    gripper_state: str = "gripper_state"
    gripper_action: str = "gripper_action"
    episode: str = "episode"


DATASET_FIELDS = RealmanDatasetFields()


def ensure_exists(path: str | Path, description: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def ensure_clean_dir(path: str | Path, force: bool = False) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {resolved}. Pass --force to overwrite it."
            )
        shutil.rmtree(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def require_packages(packages: Iterable[str]) -> None:
    missing: list[str] = []
    for package in packages:
        import_name = package
        pretty_name = package
        if package == "Pillow":
            import_name = "PIL"
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pretty_name)
    if missing:
        raise ImportError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install the toolkit dependencies and lerobot first."
        )


def save_json(path: str | Path, payload: dict) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def resolve_episode_ranges(
    data_group,
    meta_group,
    max_episodes: int | None = None,
) -> list[tuple[int, int]]:
    if "episode_ends" in meta_group:
        episode_ends = np.asarray(meta_group["episode_ends"][:], dtype=np.int64)
    elif DATASET_FIELDS.episode in data_group:
        episode_ids = np.asarray(data_group[DATASET_FIELDS.episode][:], dtype=np.int64)
        if episode_ids.size == 0:
            episode_ends = np.array([], dtype=np.int64)
        else:
            _, counts = np.unique(episode_ids, return_counts=True)
            episode_ends = np.cumsum(counts).astype(np.int64)
    else:
        raise KeyError("Could not determine episode boundaries from zarr dataset.")

    if max_episodes is not None:
        episode_ends = episode_ends[:max_episodes]

    ranges: list[tuple[int, int]] = []
    start = 0
    for end in episode_ends.tolist():
        ranges.append((int(start), int(end)))
        start = int(end)
    return ranges


def infer_rgb_shape(data_group, rgb_key: str = DATASET_FIELDS.rgb) -> tuple[int, int, int]:
    if rgb_key not in data_group:
        raise KeyError(f"Expected RGB key '{rgb_key}' in zarr data group.")
    shape = tuple(data_group[rgb_key].shape)
    if len(shape) != 4:
        raise ValueError(f"Expected {rgb_key} to have shape (N, C, H, W), got {shape}.")
    _, channels, height, width = shape
    if channels != 3:
        raise ValueError(f"Expected {rgb_key} to have 3 channels, got {channels}.")
    return height, width, channels


def chw_to_hwc_uint8(image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D RGB array, got shape {image.shape}.")
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def maybe_squeeze_last_dim(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


def normalize_gripper(values: np.ndarray, max_gripper_width: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    scale = max(float(max_gripper_width), 1e-6)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def build_state_array(
    data_group,
    start: int,
    end: int,
    state_source: str,
    max_gripper_width: float,
) -> tuple[np.ndarray, list[str]]:
    if state_source == "pose":
        base_key = DATASET_FIELDS.pose
        names = POSE_STATE_NAMES[:-1]
    elif state_source == "joint":
        base_key = DATASET_FIELDS.joint
        names = JOINT_STATE_NAMES[:-1]
    else:
        raise ValueError(f"Unsupported state_source: {state_source}")

    if base_key not in data_group:
        raise KeyError(f"Zarr dataset does not contain required state key '{base_key}'.")

    base = np.asarray(data_group[base_key][start:end], dtype=np.float32)

    if DATASET_FIELDS.gripper_width in data_group:
        gripper = normalize_gripper(
            maybe_squeeze_last_dim(data_group[DATASET_FIELDS.gripper_width][start:end]),
            max_gripper_width=max_gripper_width,
        )
    elif DATASET_FIELDS.gripper_state in data_group:
        gripper = np.clip(
            np.asarray(
                maybe_squeeze_last_dim(data_group[DATASET_FIELDS.gripper_state][start:end]),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
    else:
        gripper = np.zeros((end - start,), dtype=np.float32)

    state = np.concatenate([base, gripper[:, None]], axis=1).astype(np.float32)
    return state, names + ["gripper"]


def build_action_array(
    data_group,
    start: int,
    end: int,
    max_gripper_width: float,
) -> tuple[np.ndarray, list[str]]:
    if DATASET_FIELDS.action not in data_group:
        raise KeyError(f"Zarr dataset does not contain required action key '{DATASET_FIELDS.action}'.")

    delta = np.asarray(data_group[DATASET_FIELDS.action][start:end], dtype=np.float32)

    if DATASET_FIELDS.gripper_action in data_group:
        gripper = normalize_gripper(
            maybe_squeeze_last_dim(data_group[DATASET_FIELDS.gripper_action][start:end]),
            max_gripper_width=max_gripper_width,
        )
    elif DATASET_FIELDS.gripper_state in data_group:
        gripper = np.clip(
            np.asarray(
                maybe_squeeze_last_dim(data_group[DATASET_FIELDS.gripper_state][start:end]),
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
    else:
        gripper = np.zeros((end - start,), dtype=np.float32)

    action = np.concatenate([delta, gripper[:, None]], axis=1).astype(np.float32)
    action_names = ACTION_NAMES[: action.shape[1]]
    if len(action_names) != action.shape[1]:
        action_names = [f"action_{idx}" for idx in range(action.shape[1])]
    return action, action_names


def wait_for_robot_state(env, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = env.get_state()
        if state is not None:
            return state
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for robot state.")


def resize_rgb_frame(rgb_hwc: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    import cv2

    height, width, _ = image_shape
    resized = cv2.resize(np.asarray(rgb_hwc), (width, height), interpolation=cv2.INTER_LINEAR)
    if resized.dtype != np.uint8:
        resized = np.clip(resized, 0, 255).astype(np.uint8)
    return resized


def render_status_preview(
    rgb_hwc: np.ndarray,
    lines: list[str],
    window_name: str = "Realman ACT",
) -> np.ndarray:
    import cv2

    preview = cv2.cvtColor(np.asarray(rgb_hwc), cv2.COLOR_RGB2BGR)
    overlay = preview.copy()
    line_height = 24
    box_height = 10 + line_height * max(len(lines), 1)
    cv2.rectangle(overlay, (0, 0), (preview.shape[1], box_height), (0, 0, 0), -1)
    preview = cv2.addWeighted(overlay, 0.45, preview, 0.55, 0.0)
    for idx, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (10, 24 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow(window_name, preview)
    return preview
