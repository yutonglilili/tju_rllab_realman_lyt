#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from common import ensure_exists, require_packages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight structural and statistical validation on a local LeRobot dataset."
    )
    parser.add_argument("--path", "-p", required=True, help="Local LeRobot dataset directory.")
    parser.add_argument("--repo-id", default=None, help="Repo id. Defaults to the directory name.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only scan the first N frames when you want a quicker pass.",
    )
    return parser.parse_args()


def print_vector_stats(name: str, array: np.ndarray, labels: list[str] | None) -> None:
    labels = labels or [f"dim_{idx}" for idx in range(array.shape[1])]
    print(f"\n{name} stats")
    print(f"{'name':<16} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}")
    print("-" * 68)
    for idx in range(array.shape[1]):
        label = labels[idx] if idx < len(labels) else f"dim_{idx}"
        column = array[:, idx]
        print(
            f"{label:<16} {column.min():>12.4f} {column.max():>12.4f} "
            f"{column.mean():>12.4f} {column.std():>12.4f}"
        )


def print_anomaly_report(name: str, array: np.ndarray) -> list[str]:
    warnings: list[str] = []
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    zero_rows = int(np.all(array == 0, axis=1).sum())

    if nan_count:
        warnings.append(f"{name}: found {nan_count} NaN values")
    if inf_count:
        warnings.append(f"{name}: found {inf_count} Inf values")
    if zero_rows:
        warnings.append(f"{name}: found {zero_rows} all-zero frames")
    return warnings


def main() -> None:
    args = parse_args()
    require_packages(["lerobot"])

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_path = ensure_exists(args.path, "LeRobot dataset")
    repo_id = args.repo_id or dataset_path.name
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)

    frame_count = len(dataset)
    scan_count = min(frame_count, args.max_frames) if args.max_frames else frame_count
    meta = dataset.meta
    feature_keys = list(meta.features.keys())
    vector_names = {
        key: meta.features[key].get("names")
        for key in meta.features
        if meta.features[key]["dtype"] not in ("image", "video", "string")
    }
    image_keys = [key for key in meta.features if meta.features[key]["dtype"] in ("image", "video")]

    print("=" * 72)
    print("LeRobot dataset verification")
    print("=" * 72)
    print(f"path            : {dataset_path}")
    print(f"repo_id         : {repo_id}")
    print(f"robot_type      : {meta.robot_type}")
    print(f"fps             : {meta.fps}")
    print(f"episodes        : {dataset.num_episodes}")
    print(f"frames          : {frame_count}")
    print(f"scan_frames     : {scan_count}")
    print(f"features        : {feature_keys}")
    print("=" * 72)

    sample = dataset[0]
    print("\nSample frame")
    for key in feature_keys:
        if key not in sample:
            print(f"[missing] {key}")
            continue
        value = sample[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value).__name__)
        print(f"{key:<24} shape={tuple(shape) if shape is not None else '-'} dtype={dtype}")

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_ids: list[int] = []

    for index in range(scan_count):
        frame = dataset[index]
        if "observation.state" in frame:
            all_states.append(frame["observation.state"].detach().cpu().numpy())
        if "action" in frame:
            all_actions.append(frame["action"].detach().cpu().numpy())
        if "episode_index" in frame:
            episode_ids.append(int(frame["episode_index"]))

    warnings: list[str] = []
    if episode_ids:
        episode_ids_np = np.asarray(episode_ids, dtype=np.int64)
        unique_episodes, counts = np.unique(episode_ids_np, return_counts=True)
        print("\nEpisode length summary")
        print(f"min             : {counts.min()} frames")
        print(f"max             : {counts.max()} frames")
        print(f"mean            : {counts.mean():.1f} frames")
        print(f"episodes_seen   : {len(unique_episodes)}")

    if all_states:
        states = np.stack(all_states).astype(np.float32)
        print_vector_stats("observation.state", states, vector_names.get("observation.state"))
        warnings.extend(print_anomaly_report("observation.state", states))

    if all_actions:
        actions = np.stack(all_actions).astype(np.float32)
        print_vector_stats("action", actions, vector_names.get("action"))
        warnings.extend(print_anomaly_report("action", actions))

    if image_keys:
        print("\nImage features")
        for key in image_keys:
            feature = meta.features[key]
            print(f"{key:<24} shape={tuple(feature['shape'])} storage={feature['dtype']}")

    print("\nVerification result")
    if warnings:
        for warning in warnings:
            print(f"[warn] {warning}")
    else:
        print("[pass] no obvious NaN/Inf/all-zero issues were found in the scanned frames")


if __name__ == "__main__":
    main()
