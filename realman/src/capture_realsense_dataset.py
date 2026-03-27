#!/usr/bin/env python3
"""
采集 RealSense 图像并保存为配套数据集。

目录结构示例：
    /home/zhangzhao/lyt/camera/20260324_153000/
        ├── depth/
        │   ├── 00001.npz
        │   ├── ...
        │   └── 00009.npz
        ├── rgb/
        │   ├── 00001.png
        │   ├── ...
        │   └── 00009.png
        └── vis/
            ├── 00001.png
            ├── ...
            └── 00009.png
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from time import sleep

import cv2
import numpy as np

# 兼容直接以脚本方式运行：把 /home/zhangzhao/lyt/realman 加入导入路径
CURRENT_DIR = Path(__file__).resolve().parent
REALMAN_DIR = CURRENT_DIR.parent
if str(REALMAN_DIR) not in sys.path:
    sys.path.insert(0, str(REALMAN_DIR))

from open3d_realsense_env import Open3dRealsenseEnv


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """把深度图可视化为伪彩色 BGR 图。"""
    depth_normalized = depth.astype(np.float32)
    depth_normalized = np.clip(depth_normalized, 0, 10000)
    depth_normalized = (depth_normalized / 10000 * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    depth_colored[depth == 0] = [0, 0, 0]
    return depth_colored


def make_output_dirs(root_dir: Path) -> tuple[Path, Path, Path]:
    """创建输出目录和三个子目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root_dir / timestamp
    depth_dir = out_dir / "depth"
    rgb_dir = out_dir / "rgb"
    vis_dir = out_dir / "vis"

    depth_dir.mkdir(parents=True, exist_ok=False)
    rgb_dir.mkdir(parents=True, exist_ok=False)
    vis_dir.mkdir(parents=True, exist_ok=False)
    return depth_dir, rgb_dir, vis_dir


def save_sample(
    idx: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
    depth_dir: Path,
    rgb_dir: Path,
    vis_dir: Path,
) -> None:
    """保存一组对应的 depth/rgb/vis。"""
    stem = f"{idx:05d}"
    depth_path = depth_dir / f"{stem}.npz"
    rgb_path = rgb_dir / f"{stem}.png"
    vis_path = vis_dir / f"{stem}.png"

    np.savez_compressed(depth_path, depth=depth.astype(np.uint16), depth_scale=depth_scale)
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(vis_path), colorize_depth(depth))


def main() -> None:
    parser = argparse.ArgumentParser(description="采集 RealSense 数据并保存 depth/rgb/vis 三类文件")
    parser.add_argument("--serial", type=str, default="f1471338", help="相机序列号")
    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/zhangzhao/lyt/camera",
        help="数据保存根目录",
    )
    parser.add_argument("--num_frames", type=int, default=10, help="采集帧数")
    parser.add_argument("--warmup_frames", type=int, default=30, help="预热帧数")
    parser.add_argument("--interval_sec", type=float, default=0.0, help="相邻两帧保存间隔秒数")
    args = parser.parse_args()

    if args.num_frames <= 0:
        raise ValueError("--num_frames 必须 > 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup_frames 必须 >= 0")
    if args.interval_sec < 0:
        raise ValueError("--interval_sec 必须 >= 0")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    depth_dir, rgb_dir, vis_dir = make_output_dirs(save_root)
    out_dir = depth_dir.parent

    print("=" * 60)
    print("开始采集 RealSense 数据")
    print(f"相机序列号: {args.serial}")
    print(f"目标帧数: {args.num_frames}")
    print(f"输出目录: {out_dir}")
    print("=" * 60)

    rs_env = Open3dRealsenseEnv(args.serial)
    depth_scale = rs_env.meta_obs["depth_scale"]

    try:
        for _ in range(args.warmup_frames):
            rs_env.step()

        for i in range(args.num_frames):
            obs = rs_env.step()
            rgb = obs["rgb"]
            depth = obs["depth"]
            file_idx = i + 1
            save_sample(file_idx, rgb, depth, depth_scale, depth_dir, rgb_dir, vis_dir)
            print(f"[{i + 1}/{args.num_frames}] 已保存: {file_idx:05d}.npz/.png/.png")

            if args.interval_sec > 0 and i < args.num_frames - 1:
                sleep(args.interval_sec)
    finally:
        rs_env.close()

    print("=" * 60)
    print("采集完成。三个目录中的文件名一一对应。")
    print(f"depth: {depth_dir}")
    print(f"rgb:   {rgb_dir}")
    print(f"vis:   {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

