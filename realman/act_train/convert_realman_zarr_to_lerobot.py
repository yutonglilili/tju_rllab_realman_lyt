#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from common import (
    DATASET_FIELDS,
    DEFAULT_MAX_GRIPPER_WIDTH,
    DEFAULT_ROBOT_TYPE,
    build_action_array,
    build_state_array,
    chw_to_hwc_uint8,
    ensure_clean_dir,
    ensure_exists,
    infer_rgb_shape,
    require_packages,
    resolve_episode_ranges,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Realman teleoperation zarr dataset into a LeRobot v3 dataset."
    )
    parser.add_argument("--input", "-i", required=True, help="Input zarr dataset path.")
    parser.add_argument("--output", "-o", required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--repo-id", required=True, help="Local repo id used by LeRobot.")
    parser.add_argument("--task", default=None, help="Task description stored in the dataset.")
    parser.add_argument("--fps", type=int, default=15, help="Dataset FPS for training and inference.")
    parser.add_argument(
        "--robot-type",
        default=DEFAULT_ROBOT_TYPE,
        help=f"Robot type written into info.json. Default: {DEFAULT_ROBOT_TYPE}.",
    )
    parser.add_argument(
        "--state-source",
        choices=["pose", "joint"],
        default="pose",
        help="State feature source. 'pose' gives 6D tcp pose + gripper, 'joint' gives 7D joints + gripper.",
    )
    parser.add_argument(
        "--max-gripper-width",
        type=float,
        default=DEFAULT_MAX_GRIPPER_WIDTH,
        help="Used to normalize gripper width/action into the 0..1 range.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Only convert the first N episodes. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_packages(["zarr", "Pillow", "lerobot"])

    import zarr
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    zarr_path = ensure_exists(args.input, "Input zarr dataset")
    output_dir = ensure_clean_dir(args.output, force=args.force)
    task_name = args.task or Path(args.input).stem.replace("_", " ")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    store = zarr.open(str(zarr_path), mode="r")
    data_group = store["data"]
    meta_group = store["meta"]

    image_shape = infer_rgb_shape(data_group, rgb_key=DATASET_FIELDS.rgb)
    episode_ranges = resolve_episode_ranges(
        data_group=data_group,
        meta_group=meta_group,
        max_episodes=args.episodes,
    )
    total_frames = sum(end - start for start, end in episode_ranges)

    if episode_ranges:
        state_probe, state_names = build_state_array(
            data_group=data_group,
            start=episode_ranges[0][0],
            end=episode_ranges[0][1],
            state_source=args.state_source,
            max_gripper_width=args.max_gripper_width,
        )
        action_probe, action_names = build_action_array(
            data_group=data_group,
            start=episode_ranges[0][0],
            end=episode_ranges[0][1],
            max_gripper_width=args.max_gripper_width,
        )
    else:
        state_probe = np.zeros((0, 7), dtype=np.float32)
        action_probe = np.zeros((0, 7), dtype=np.float32)
        state_names = []
        action_names = []

    features = {
        "observation.image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_probe.shape[1],),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_probe.shape[1],),
            "names": action_names,
        },
    }

    print("=" * 72)
    print("Realman zarr -> LeRobot conversion")
    print("=" * 72)
    print(f"input           : {zarr_path}")
    print(f"output          : {output_dir}")
    print(f"repo_id         : {args.repo_id}")
    print(f"task            : {task_name}")
    print(f"episodes        : {len(episode_ranges)}")
    print(f"frames          : {total_frames}")
    print(f"image_shape     : {image_shape}")
    print(f"state_source    : {args.state_source}")
    print(f"state_names     : {state_names}")
    print(f"action_names    : {action_names}")
    print("=" * 72)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        root=output_dir,
        use_videos=False,
        image_writer_threads=4,
    )

    start_time = time.time()
    for episode_index, (start, end) in enumerate(episode_ranges):
        rgb_batch = np.asarray(data_group[DATASET_FIELDS.rgb][start:end])
        state_batch, _ = build_state_array(
            data_group=data_group,
            start=start,
            end=end,
            state_source=args.state_source,
            max_gripper_width=args.max_gripper_width,
        )
        action_batch, _ = build_action_array(
            data_group=data_group,
            start=start,
            end=end,
            max_gripper_width=args.max_gripper_width,
        )

        print(f"episode {episode_index:04d}: frames [{start}, {end}) -> {end - start} steps")
        for frame_index in range(end - start):
            dataset.add_frame(
                {
                    "observation.image": Image.fromarray(chw_to_hwc_uint8(rgb_batch[frame_index])),
                    "observation.state": state_batch[frame_index].astype(np.float32),
                    "action": action_batch[frame_index].astype(np.float32),
                    "task": task_name,
                }
            )
        dataset.save_episode()

    elapsed = time.time() - start_time
    save_json(
        output_dir / "realman_conversion.json",
        {
            "source_zarr": str(zarr_path),
            "repo_id": args.repo_id,
            "task": task_name,
            "fps": args.fps,
            "robot_type": args.robot_type,
            "state_source": args.state_source,
            "state_names": state_names,
            "action_names": action_names,
            "episodes": len(episode_ranges),
            "frames": total_frames,
            "max_gripper_width": args.max_gripper_width,
        },
    )

    print("\nConversion finished.")
    print(f"saved_to        : {output_dir}")
    print(f"elapsed_sec     : {elapsed:.1f}")
    print(f"frames_per_sec  : {total_frames / max(elapsed, 1e-6):.1f}")


if __name__ == "__main__":
    main()
