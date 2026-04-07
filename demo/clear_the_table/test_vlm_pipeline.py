"""
Simple test script for VLM task planner.
Modify IMAGE_PATH and INSTRUCTION directly in the script.
"""

import os
import cv2
import json
import ast
import re
import numpy as np
from PIL import Image

from multi_pointing_vllm_get_point_utils import *
import multi_pointing_vllm_get_point_utils as vlm_utils

# =====================================================
# Modify here
# =====================================================

IMAGE_PATH = "/home/zhangzhao/lyt/tmp_vlm_image.png"

INSTRUCTION = "Clear the table.Pick all toys and place them on the white plate."

SAVE_DIR = "/home/zhangzhao/lyt/demo/clear_the_table"

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


def _extract_first_json_debug(text: str):
    """
    Debug parser used only in this test script.
    It accepts both JSON object and JSON array, and prints what was extracted.
    """
    candidates = [
        ("array", r"\[[\s\S]*\]"),
        ("object", r"\{[\s\S]*\}"),
    ]

    for kind, pattern in candidates:
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            continue

        json_str = match.group()
        print(f"[debug] extracted {kind}, length={len(json_str)}")
        preview = json_str[:200].replace("\n", "\\n")
        print(f"[debug] json preview: {preview}...")

        try:
            data = json.loads(json_str)
        except Exception:
            data = ast.literal_eval(json_str)

        print(f"[debug] parsed type: {type(data).__name__}")
        return data

    print("[debug] no json fragment found in model output")
    print("[debug] raw output preview:", text[:400].replace("\n", "\\n"))
    raise RuntimeError("No JSON found in model output")

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

    task_table = generate_tasks_from_scene(
        image_rgb=image_rgb,
        instruction=INSTRUCTION
    )

    # If task table is empty, run a second pass with a local debug parser
    # to verify whether the model returned a JSON array that the default parser misses.
    if len(task_table) == 0:
        print("\n---- Debug pass (local parser in test script) ----\n")
        original_parser = vlm_utils.extract_first_json
        try:
            vlm_utils.extract_first_json = _extract_first_json_debug
            debug_task_table = generate_tasks_from_scene(
                image_rgb=image_rgb,
                instruction=INSTRUCTION
            )
        finally:
            vlm_utils.extract_first_json = original_parser

        print("\n[debug] task count with local parser:", len(debug_task_table))
        if len(debug_task_table) > 0:
            task_table = debug_task_table
            print("[debug] using debug parser result as final output for this run")

    # =====================================
    # Print results
    # =====================================

    print("\n========== TASK TABLE ==========\n")

    print("Number of tasks:", len(task_table))
    print()

    for i, task in enumerate(task_table):

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
