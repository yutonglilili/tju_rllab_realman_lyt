#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil

from common import (
    DEFAULT_DEPLOY_CONFIG_NAME,
    DEFAULT_MAX_GRIPPER_WIDTH,
    DEFAULT_ROBOT_TYPE,
    ensure_clean_dir,
    ensure_exists,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle a trained ACT checkpoint with the runtime settings needed for Realman deployment."
    )
    parser.add_argument("--checkpoint", required=True, help="Trained ACT checkpoint directory.")
    parser.add_argument("--output", "-o", required=True, help="Deployment bundle directory.")
    parser.add_argument("--task", default="teleop task", help="Task string passed to the policy runtime.")
    parser.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE, help="Robot type label.")
    parser.add_argument("--robot-ip", default="192.168.101.19", help="Default robot IP for the bundle.")
    parser.add_argument("--camera-serial", default="f1471338", help="Default RealSense serial for the bundle.")
    parser.add_argument(
        "--state-source",
        choices=["pose", "joint"],
        default="pose",
        help="Must match the state source used during dataset conversion.",
    )
    parser.add_argument(
        "--control-fps",
        type=float,
        default=15.0,
        help="Nominal runtime control loop frequency.",
    )
    parser.add_argument(
        "--max-gripper-width",
        type=float,
        default=DEFAULT_MAX_GRIPPER_WIDTH,
        help="Robot gripper max opening used to denormalize gripper commands.",
    )
    parser.add_argument(
        "--max-delta-translation-mm",
        type=float,
        default=20.0,
        help="Runtime safety clamp for per-step xyz deltas.",
    )
    parser.add_argument(
        "--max-delta-rotation-rad",
        type=float,
        default=0.35,
        help="Runtime safety clamp for per-step rpy deltas.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output bundle directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = ensure_exists(args.checkpoint, "Checkpoint directory")
    bundle_dir = ensure_clean_dir(args.output, force=args.force)
    model_dir = bundle_dir / "model"
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(checkpoint_dir, model_dir)

    deploy_config = {
        "task": args.task,
        "robot_type": args.robot_type,
        "robot_ip": args.robot_ip,
        "camera_serial": args.camera_serial,
        "state_source": args.state_source,
        "control_fps": args.control_fps,
        "max_gripper_width": args.max_gripper_width,
        "max_delta_translation_mm": args.max_delta_translation_mm,
        "max_delta_rotation_rad": args.max_delta_rotation_rad,
        "model_dir": "model",
    }
    save_json(bundle_dir / DEFAULT_DEPLOY_CONFIG_NAME, deploy_config)

    print("=" * 72)
    print("Deployment bundle prepared")
    print("=" * 72)
    print(f"checkpoint      : {checkpoint_dir}")
    print(f"bundle          : {bundle_dir}")
    print(f"model_dir       : {model_dir}")
    print(f"config_file     : {bundle_dir / DEFAULT_DEPLOY_CONFIG_NAME}")
    print("=" * 72)


if __name__ == "__main__":
    main()
