"""
Validate the Realman-collected Zarr dataset and emit a normalizer JSON
compatible with the (Realman-adapted) ForceFlow pipeline.

Expected schema (produced by scripts/collect.py):

    data/
        rgb_arm        (N, 3, H, W)  uint8    -- D435 腕部相机
        rgb_fix        (N, 3, H, W)  uint8    -- L515 固定相机
        # depth_fix    (N, 1, H, W)  uint16   -- 暂时禁用深度
        joint          (N, 7)        float32
        pos            (N, 6)        float32   -- EEF xyzrpy (SDK native)
        action         (N, 6)        float32   -- spacemouse delta
        gripper_state  (N, 1)        float32   -- continuous [0,1] (obs)
        gripper_action (N, 1)        float32   -- continuous [0,1] (action)
        gripper_width  (N, 1)        float32   -- actual open ratio [0,1]
        timestamp      (N,)          float64
        episode        (N,)          uint32

    meta/
        episode_ends

No force / ft fields are present on Realman.
"""

from __future__ import annotations

import json
import os

import numpy as np
import zarr
from termcolor import cprint

# 触发 jpeg codec 注册, 才能读 collect 写入的 rgb_arm / rgb_fix
try:
    from CleanDiffuser.image_codecs import jpeg as _jpeg_codec  # noqa: F401
except ImportError:
    cprint("warning: jpeg codec not found, rgb arrays may fail to read", "yellow")


DATASET_PATH = "data/demo.zarr"  # set to your dataset path


# Fields whose statistics we save for downstream normalization
NORMALIZER_FIELDS = ("pos", "action")


# Expected dataset schema. Use None for the leading episode dimension and for
# image H/W so that the validator only checks fixed-rank dims.
REQUIRED_DATASETS = {
    "rgb_arm":        {"shape": (None, 3, None, None), "dtype": np.uint8},
    "rgb_fix":        {"shape": (None, 3, None, None), "dtype": np.uint8},
    # 暂时禁用深度
    # "depth_fix":      {"shape": (None, 1, None, None), "dtype": np.uint16},
    "joint":          {"shape": (None, 7),             "dtype": np.float32},
    "pos":            {"shape": (None, 6),             "dtype": np.float32},
    "action":         {"shape": (None, 6),             "dtype": np.float32},
    "gripper_state":  {"shape": (None, 1),             "dtype": np.float32},
    "gripper_width":  {"shape": (None, 1),             "dtype": np.float32},
    "gripper_action": {"shape": (None, 1),             "dtype": np.float32},
    "timestamp":      {"shape": (None,),               "dtype": np.float64},
    "episode":        {"shape": (None,),               "dtype": np.uint32},
}


def calculate_normalizer_params(dataset_path):
    """Compute per-channel min/max for fields listed in NORMALIZER_FIELDS."""
    try:
        dataset = zarr.open(dataset_path, "r")
        data_group = dataset["data"]

        cprint("Computing normalizer params...", "cyan")
        normalizer_info = {}

        for field in NORMALIZER_FIELDS:
            if field not in data_group:
                cprint(f"Skipping {field}: not found in dataset", "yellow")
                continue

            arr = data_group[field][:]
            arr_max = np.max(arr, axis=0)
            arr_min = np.min(arr, axis=0)
            normalizer_info[field] = {
                "max": arr_max.tolist(),
                "min": arr_min.tolist(),
                "shape": arr.shape,
                "dtype": str(arr.dtype),
            }
            cprint(f"{field} statistics:", "green")
            cprint(f"  shape: {arr.shape}", "cyan")
            cprint(f"  max: {arr_max}", "cyan")
            cprint(f"  min: {arr_min}", "cyan")

        return normalizer_info

    except Exception as e:
        cprint(f"Failed to compute normalizer params: {e}", "red")
        return None


def fix_episode_ends(dataset_path):
    """Recompute episode_ends from the episode field and write to dataset."""
    try:
        dataset = zarr.open(dataset_path, "a")
        data_group = dataset["data"]
        meta_group = dataset["meta"]

        if "episode" not in data_group:
            cprint("Missing 'episode' field, cannot recompute episode_ends", "red")
            return False

        episode_ids = data_group["episode"][:]
        total_steps = len(episode_ids)

        cprint("Recomputing episode_ends...", "cyan")
        cprint(f"  total steps: {total_steps}", "cyan")

        if total_steps == 0:
            correct_episode_ends = np.array([], dtype=np.uint32)
        else:
            transitions = []
            current_id = episode_ids[0]
            for i in range(1, total_steps):
                if episode_ids[i] != current_id:
                    transitions.append(i)
                    current_id = episode_ids[i]
            transitions.append(total_steps)
            correct_episode_ends = np.array(transitions, dtype=np.uint32)

        current_episode_ends = None
        if "episode_ends" in meta_group:
            current_episode_ends = meta_group["episode_ends"][:]
            cprint(f"  current: {len(current_episode_ends)} episodes", "cyan")
        else:
            cprint("  no episode_ends found, will create", "cyan")

        cprint(f"  recomputed: {len(correct_episode_ends)} episodes", "cyan")

        needs_fix = False
        if current_episode_ends is None:
            needs_fix = True
            cprint("  reason: episode_ends missing", "yellow")
        elif len(current_episode_ends) != len(correct_episode_ends):
            needs_fix = True
            cprint(
                f"  reason: episode count mismatch ({len(current_episode_ends)} vs {len(correct_episode_ends)})",
                "yellow",
            )
        elif not np.array_equal(current_episode_ends, correct_episode_ends):
            needs_fix = True
            cprint("  reason: episode boundaries differ", "yellow")

        if needs_fix:
            if "episode_ends" in meta_group:
                del meta_group["episode_ends"]
            meta_group.create_dataset("episode_ends", data=correct_episode_ends, dtype=np.uint32)
            cprint("episode_ends updated", "green")
            cprint(f"  new episode_ends: {correct_episode_ends}", "cyan")
            if len(correct_episode_ends) > 1:
                lengths = np.diff(np.concatenate([[0], correct_episode_ends]))
                cprint(
                    f"  episode lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}",
                    "cyan",
                )
        else:
            cprint("episode_ends already correct", "green")

        return True

    except Exception as e:
        cprint(f"Failed to update episode_ends: {e}", "red")
        import traceback

        cprint(f"Traceback:\n{traceback.format_exc()}", "red")
        return False


def build_normalizer_json(normalizer_info):
    """Build the dict to save as normalizer JSON, keeping only min/max statistics."""
    result = {}
    for field in NORMALIZER_FIELDS:
        if field in normalizer_info:
            result[field] = {
                "max": normalizer_info[field]["max"],
                "min": normalizer_info[field]["min"],
            }
    return result


def save_normalizer_info(dataset_path, normalizer_info):
    """Save normalizer statistics to a JSON file next to the dataset directory."""
    try:
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        json_file = f"{dataset_path}/{base_name}_normalizer.json"
        json_payload = build_normalizer_json(normalizer_info)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)
        cprint(f"Saved normalizer JSON: {json_file}", "green")
        return json_file
    except Exception as e:
        cprint(f"Failed to save normalizer JSON: {e}", "red")
        return None


def validate_dataset_structure(dataset_path):
    """Validate dataset group structure and field shapes/dtypes."""
    try:
        dataset = zarr.open(dataset_path, "r")
        cprint(f"Opened dataset: {dataset_path}", "green")

        if "data" not in dataset:
            cprint("Missing 'data' group", "red")
            return False
        if "meta" not in dataset:
            cprint("Missing 'meta' group", "red")
            return False

        data_group = dataset["data"]

        for dataset_name, expected in REQUIRED_DATASETS.items():
            if dataset_name not in data_group:
                cprint(f"Missing dataset: {dataset_name}", "red")
                return False

            ds = data_group[dataset_name]
            actual_shape = ds.shape
            actual_dtype = ds.dtype
            expected_shape = expected["shape"]

            if len(actual_shape) != len(expected_shape):
                cprint(
                    f"{dataset_name}: rank mismatch (expected {len(expected_shape)}, got {len(actual_shape)})",
                    "red",
                )
                return False

            for i in range(1, len(expected_shape)):
                if expected_shape[i] is None:
                    continue
                if actual_shape[i] != expected_shape[i]:
                    cprint(
                        f"{dataset_name}: shape mismatch (expected {expected_shape}, got {actual_shape})",
                        "red",
                    )
                    return False

            if actual_dtype != expected["dtype"]:
                cprint(
                    f"{dataset_name}: dtype mismatch (expected {expected['dtype']}, got {actual_dtype})",
                    "red",
                )
                return False

            cprint(f"{dataset_name}: shape={actual_shape} dtype={actual_dtype}", "green")

        return True

    except Exception as e:
        cprint(f"Failed to open dataset: {e}", "red")
        return False


def validate_data_consistency(dataset_path):
    """Check that all data fields have the same number of steps."""
    try:
        dataset = zarr.open(dataset_path, "r")
        data_group = dataset["data"]

        lengths = {key: data_group[key].shape[0] for key in data_group.keys()}

        cprint("Step counts:", "cyan")
        for key, length in lengths.items():
            cprint(f"  {key}: {length}", "cyan")

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            cprint(f"Inconsistent lengths: {lengths}", "red")
            return False

        main_length = next(iter(unique_lengths))
        cprint(f"All fields consistent: {main_length} steps", "green")
        return True

    except Exception as e:
        cprint(f"Failed to check consistency: {e}", "red")
        return False


def validate_data_ranges(dataset_path):
    """Validate value ranges for rgb, gripper_state, gripper_action, episode."""
    try:
        dataset = zarr.open(dataset_path, "r")
        data_group = dataset["data"]

        cprint("Data range check:", "cyan")

        for rgb_key in ("rgb_arm", "rgb_fix"):
            if rgb_key in data_group:
                try:
                    rgb_data = data_group[rgb_key][:]
                    if rgb_data.min() < 0 or rgb_data.max() > 255:
                        cprint(f"{rgb_key} out of range: [{rgb_data.min()}, {rgb_data.max()}]", "red")
                        return False
                    cprint(f"{rgb_key} range: [{rgb_data.min()}, {rgb_data.max()}]", "green")
                except Exception as e:
                    cprint(f"{rgb_key} read failed: {e}", "yellow")

        # 暂时禁用深度
        # if "depth_fix" in data_group:
        #     try:
        #         depth_data = data_group["depth_fix"][:]
        #         cprint(
        #             f"depth_fix range: [{depth_data.min()}, {depth_data.max()}] (uint16 raw units)",
        #             "green",
        #         )
        #     except Exception as e:
        #         cprint(f"depth_fix read failed: {e}", "yellow")

        if "gripper_state" in data_group:
            try:
                gs = data_group["gripper_state"][:]
                if gs.min() < -1e-3 or gs.max() > 1.0 + 1e-3:
                    cprint(
                        f"gripper_state out of [0,1] range: [{gs.min()}, {gs.max()}]",
                        "yellow",
                    )
                else:
                    cprint(
                        f"gripper_state range: [{gs.min():.3f}, {gs.max():.3f}] (continuous)",
                        "green",
                    )
                cprint(f"  mean={gs.mean():.3f}  std={gs.std():.3f}", "cyan")
            except Exception as e:
                cprint(f"gripper_state read failed: {e}", "yellow")

        if "gripper_action" in data_group:
            try:
                ga = data_group["gripper_action"][:]
                if ga.min() < -1e-3 or ga.max() > 1.0 + 1e-3:
                    cprint(
                        f"gripper_action out of [0,1] range: [{ga.min()}, {ga.max()}]",
                        "yellow",
                    )
                else:
                    cprint(
                        f"gripper_action range: [{ga.min():.3f}, {ga.max():.3f}] (continuous, same scale as state)",
                        "green",
                    )
            except Exception as e:
                cprint(f"gripper_action read failed: {e}", "yellow")

        if "gripper_width" in data_group:
            try:
                gw = data_group["gripper_width"][:]
                cprint(
                    f"gripper_width range: [{gw.min():.4f}, {gw.max():.4f}] m (physical, debug)",
                    "green",
                )
            except Exception as e:
                cprint(f"gripper_width read failed: {e}", "yellow")

        if "episode" in data_group:
            try:
                episode_data = data_group["episode"][:]
                unique_episodes = np.unique(episode_data)
                cprint(f"Episode ids: {unique_episodes}", "green")
            except Exception as e:
                cprint(f"episode field read failed: {e}", "yellow")

        return True

    except Exception as e:
        cprint(f"Data range check failed: {e}", "red")
        return False


def validate_episode_structure(dataset_path):
    """Print per-episode step counts."""
    try:
        dataset = zarr.open(dataset_path, "r")
        data_group = dataset["data"]

        if "episode" not in data_group:
            cprint("No episode field, skipping structure check", "yellow")
            return True

        episode_data = data_group["episode"][:]
        unique_episodes = np.unique(episode_data)
        cprint("Episode structure:", "cyan")
        cprint(f"  total episodes: {len(unique_episodes)}", "cyan")
        for episode_id in unique_episodes:
            episode_length = int(np.sum(episode_data == episode_id))
            cprint(f"  episode {episode_id}: {episode_length} steps", "cyan")
        return True

    except Exception as e:
        cprint(f"Episode structure validation failed: {e}", "red")
        return False


def main():
    dataset_path = DATASET_PATH

    if dataset_path is None:
        raise ValueError("dataset_path is not set")

    cprint("Validating dataset...", "yellow")
    cprint("=" * 50, "cyan")
    cprint(f"Dataset: {dataset_path}", "cyan")

    cprint("\n1. Checking dataset structure...", "yellow")
    if not validate_dataset_structure(dataset_path):
        return

    cprint("\n2. Checking data consistency...", "yellow")
    if not validate_data_consistency(dataset_path):
        return

    cprint("\n3. Checking data ranges...", "yellow")
    if not validate_data_ranges(dataset_path):
        cprint("Data range check failed", "red")
        return

    cprint("\n4. Checking episode structure...", "yellow")
    if not validate_episode_structure(dataset_path):
        return

    cprint("\n5. Fixing episode_ends...", "yellow")
    if not fix_episode_ends(dataset_path):
        cprint("episode_ends fix failed", "red")
        return

    cprint("\n6. Computing normalizer params...", "yellow")
    normalizer_info = calculate_normalizer_params(dataset_path)
    if normalizer_info:
        json_file = save_normalizer_info(dataset_path, normalizer_info)
        if json_file:
            cprint(f"Generated file: {json_file}", "green")

    cprint("\n" + "=" * 50, "green")
    cprint("All checks passed!", "green")
    cprint("=" * 50, "green")


if __name__ == "__main__":
    main()
