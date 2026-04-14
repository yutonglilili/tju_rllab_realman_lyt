"""
Simple test script for VLM pointing.
Modify IMAGE_PATH and INSTRUCTION directly in the script.
The annotated image is saved to the same directory as this script.
"""

import os

import cv2
import numpy as np
from PIL import Image

from multi_pointing_vllm_get_point_utils import get_point_vllm

# =====================================================
# Modify here
# =====================================================

IMAGE_PATH = "/home/zhangzhao/lyt/camera/20260414_134926/rgb/00001.png"
INSTRUCTION = "Point at the empty area to the near right of the Rubik's Cube."
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================


def load_image(image_path):
    """Load image as an RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _clip_point(x, y, width, height):
    """Keep a point inside the image bounds."""
    x = max(0, min(int(round(x)), width - 1))
    y = max(0, min(int(round(y)), height - 1))
    return x, y


def visualize_points(image_rgb, points):
    """Draw points and coordinate labels on the image."""
    vis_img = image_rgb.copy()
    height, width = vis_img.shape[:2]

    points = np.asarray(points, dtype=float).reshape(-1, 2)

    for idx, pt in enumerate(points, start=1):
        x, y = _clip_point(pt[0], pt[1], width, height)
        label = f"P{idx} ({x}, {y})"

        cv2.circle(vis_img, (x, y), 8, (0, 255, 0), -1)

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

        cv2.rectangle(vis_img, box_top_left, box_bottom_right, (0, 0, 0), -1)
        cv2.putText(
            vis_img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return vis_img


def main():
    print("\n========== VLM POINTING TEST ==========\n")
    print("Image path:", IMAGE_PATH)
    print("Instruction:", INSTRUCTION)

    os.makedirs(SAVE_DIR, exist_ok=True)

    image_rgb = load_image(IMAGE_PATH)
    print("Image shape:", image_rgb.shape)

    point = np.asarray(
        get_point_vllm(
            image_rgb=image_rgb,
            text_prompt=INSTRUCTION,
            save_path=None,
        ),
        dtype=float,
    )

    if point.size != 2:
        raise RuntimeError(f"Unexpected point format from get_point_vllm: {point!r}")

    points = point.reshape(1, 2)
    print("Predicted point:", points[0].tolist())

    vis_img = visualize_points(image_rgb, points)

    image_name = os.path.basename(IMAGE_PATH)
    name_no_ext, _ = os.path.splitext(image_name)
    save_path = os.path.join(SAVE_DIR, f"{name_no_ext}_point_vis.png")

    cv2.imwrite(save_path, vis_img[:, :, ::-1])

    print("Saved visualization to:", save_path)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    main()
