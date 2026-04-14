"""
VLM pointing test: run 50 times and visualize all points with overlap counts.
"""

import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from datetime import datetime

from multi_pointing_vllm_get_point_utils import get_point_vllm

# =====================================================
# Modify here
# =====================================================

IMAGE_PATH = "/home/zhangzhao/lyt/camera/20260414_134926/rgb/00001.png"
INSTRUCTION = "Point at the top of the Rubik's Cube."
NUM_SAMPLES = 20             
                     
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _clip_point(x, y, width, height):
    x = max(0, min(int(round(x)), width - 1))
    y = max(0, min(int(round(y)), height - 1))
    return x, y


def collect_points(image_rgb, num_samples):
    """Call VLM multiple times to collect points."""
    all_points = []

    for i in range(num_samples):
        print(f"[{i+1}/{num_samples}] Running VLM...")

        pt = np.asarray(
            get_point_vllm(
                image_rgb=image_rgb,
                text_prompt=INSTRUCTION,
                save_path=None,
            ),
            dtype=float,
        )

        if pt.size != 2:
            print("Warning: bad output:", pt)
            continue

        all_points.append(pt.tolist())

    return np.array(all_points, dtype=float)


def count_points(points, width, height):
    """
    Convert points to integer pixel coords and count overlaps.
    """
    counter = defaultdict(int)

    for pt in points:
        x, y = _clip_point(pt[0], pt[1], width, height)
        counter[(x, y)] += 1

    return counter


def visualize_points(image_rgb, counter):
    """Draw unique points and annotate counts."""
    vis_img = image_rgb.copy()
    height, width = vis_img.shape[:2]

    for idx, ((x, y), count) in enumerate(counter.items(), start=1):
        label = f"P{idx} ({x},{y}) x{count}"

        # draw point
        cv2.circle(vis_img, (x, y), 8, (0, 0, 255), -1)

        # text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        text_x = min(x + 10, max(0, width - text_width - 10))
        text_y = max(y - 10, text_height + 10)

        box_top_left = (text_x - 4, text_y - text_height - 4)
        box_bottom_right = (text_x + text_width + 4, text_y + baseline + 4)

        # background box
        # cv2.rectangle(vis_img, box_top_left, box_bottom_right, (0, 0, 0), -1)

        # text
        """
        cv2.putText(
            vis_img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        """

    return vis_img


def main():
    print("\n========== VLM MULTI-POINT TEST ==========\n")

    print("Image:", IMAGE_PATH)
    print("Instruction:", INSTRUCTION)
    print("Num samples:", NUM_SAMPLES)

    os.makedirs(SAVE_DIR, exist_ok=True)

    image_rgb = load_image(IMAGE_PATH)
    height, width = image_rgb.shape[:2]

    # 1. collect points
    points = collect_points(image_rgb, NUM_SAMPLES)

    print("\nRaw points shape:", points.shape)

    # 2. count overlaps
    counter = count_points(points, width, height)

    print("\nUnique points:", len(counter))
    for k, v in counter.items():
        print(k, "->", v)

    # 3. visualize
    vis_img = visualize_points(image_rgb, counter)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(SAVE_DIR, f"{timestamp}_{NUM_SAMPLES}points_vis.png")

    cv2.imwrite(save_path, vis_img[:, :, ::-1])

    print("\nSaved to:", save_path)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()
