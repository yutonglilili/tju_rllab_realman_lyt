"""
Simple test script for VLM task planner.
Modify IMAGE_PATH and INSTRUCTION directly in the script.
"""

import os
import cv2
import json
import numpy as np
from PIL import Image

from multi_pointing_vllm_get_point_utils import *

# =====================================================
# Modify here
# =====================================================

IMAGE_PATH = "/home/zhangzhao/lyt/camera/20260403_225419/rgb/00001.png"

INSTRUCTION = "Clear the table.Pick all toys and place them on the pink plate."

SAVE_DIR = "/home/zhangzhao/lyt/demo/pick_and_place"

# =====================================================


def load_image(image_path):
    """Load image as RGB numpy array"""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def visualize_points(image_rgb, points):
    """
    Draw points on image
    """

    vis_img = image_rgb.copy()

    for pt in points:

        x = int(pt[0])
        y = int(pt[1])

        cv2.circle(
            vis_img,
            (x, y),
            8,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            vis_img,
            f"({x},{y})",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return vis_img

def main():

    print("\n========== VLM TASK PLANNER TEST ==========\n")

    print("Image path:", IMAGE_PATH)
    print("Instruction:", INSTRUCTION)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # =====================================
    # Load image
    # =====================================

    image_rgb = load_image(IMAGE_PATH)

    print("Image shape:", image_rgb.shape)

    # =====================================
    # Generate task table
    # =====================================

    print("\n---- Generating task table ----\n")

    task_table = generate_task_table_from_scene(
        image_rgb=image_rgb,
        instruction=INSTRUCTION
    )

    # =====================================
    # Print results
    # =====================================

    print("\n========== TASK TABLE ==========\n")

    print("Number of tasks:", task_table["num_tasks"])
    print()

    for i, task in enumerate(task_table["tasks"]):

        pick = task["pick"]
        place = task["place"]

        print(f"Task {i+1}")
        print("  pick :", pick)
        print("  place:", place)
        print()

    print("\n========== DONE ==========\n")


"""
# 测试打点
def main():

    print("\n========== VLM PIPELINE TEST ==========\n")

    print("Image path:", IMAGE_PATH)
    print("Instruction:", INSTRUCTION)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # =====================================
    # Load image
    # =====================================

    image_rgb = load_image(IMAGE_PATH)

    print("Image shape:", image_rgb.shape)

    # =====================================
    # Step 1: Generate task table
    # =====================================

    print("\n---- Generating task table ----")

    points = get_point_vllm(
        image_rgb,
        INSTRUCTION
    )

    points = np.array(points)

    # 如果只返回一个点
    if points.ndim == 1:
        points = points.reshape(1, 2)

    print("\nPlanner output:")

    print(points)

    # =====================================
    # Visualization
    # =====================================

    vis_img = visualize_points(image_rgb, points)

    # =====================================
    # Save results
    # =====================================

    image_name = os.path.basename(IMAGE_PATH)
    name_no_ext = os.path.splitext(image_name)[0]

    save_path = os.path.join(
        SAVE_DIR,
        f"{name_no_ext}_points.png"
    )

    cv2.imwrite(
        save_path,
        vis_img[:, :, ::-1]
    )

    print("\nSaved visualization to:")

    print(save_path)

    print("\n========== DONE ==========\n")
"""

if __name__ == "__main__":
    main()
