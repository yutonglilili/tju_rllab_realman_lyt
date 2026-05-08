"""
此脚本用于直接测试 multi_pointing_vllm_get_point_utils.py 中的函数。
"""

import os
import re
import json
from datetime import datetime
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

from multi_pointing_vllm_get_point_utils import *


# 配置
# IMAGE_PATH = "/home/zhangzhao/lyt/水果和饮料.png"
# IMAGE_PATH = "/home/zhangzhao/lyt/水果和玩具plus.png"
IMAGE_PATH = "/home/zhangzhao/lyt/方位.png"
INSTRUCTION = "魔方和橘子的中间"
NUM_SAMPLES = 10             
                     
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_vllm_test")

# 加载图像
def load_image(image_path):
    """Load image as an RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

# 裁剪点
def _clip_point(x, y, width, height):
    """Keep a point inside the image bounds."""
    x = max(0, min(int(round(x)), width - 1))
    y = max(0, min(int(round(y)), height - 1))
    return x, y

# 调用模型多次打点，收集点
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

# 统计点重叠数量
def count_points(points, width, height):
    """
    Convert points to integer pixel coords and count overlaps.
    """
    counter = defaultdict(int)

    for pt in points:
        x, y = _clip_point(pt[0], pt[1], width, height)
        counter[(x, y)] += 1

    return counter

# 可视化点
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

# 清理文件名文本
def _sanitize_filename_text(text, max_length=80):
    """Convert free-form text into a safe filename fragment."""
    text = str(text).strip().replace("\n", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = text.strip("._")

    if not text:
        return "test"

    return text[:max_length]

# 构建测试保存路径
def _build_test_save_path(save_dir, instruction, suffix):
    """Build a timestamped save path for test outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instruction_tag = _sanitize_filename_text(instruction)
    filename = f"{timestamp}_{instruction_tag}_{suffix}"
    return os.path.join(save_dir, filename)


# =========================================================
# 测试函数
# =========================================================

# 给定指令和图像路径，测试模型打点。
def test_get_point_vllm(instruction, image_path, save_dir, num_samples):
    print("\n========== VLM MULTI-POINT TEST ==========\n")

    print("Image:", image_path)
    print("Instruction:", instruction)
    print("Num samples:", num_samples)

    os.makedirs(save_dir, exist_ok=True)

    image_rgb = load_image(image_path)
    height, width = image_rgb.shape[:2]

    # 1. collect points
    points = collect_points(image_rgb, num_samples)

    print("\nRaw points shape:", points.shape)

    # 2. count overlaps
    counter = count_points(points, width, height)

    print("\nUnique points:", len(counter))
    for k, v in counter.items():
        print(k, "->", v)

    # 3. visualize
    vis_img = visualize_points(image_rgb, counter)

    save_path = _build_test_save_path(
        save_dir=save_dir,
        instruction=instruction,
        suffix=f"{num_samples}points_vis.png",
    )

    cv2.imwrite(save_path, vis_img[:, :, ::-1])

    print("\nSaved to:", save_path)
    print("\n========== DONE ==========\n")

# 给定指令和图像路径，测试模型生成pnp任务列表。
def test_generate_tasks_with_descriptions(instruction, image_path, save_dir, num_samples):
    print("\n========== VLM TASK GENERATION TEST ==========\n")

    print("Image:", image_path)
    print("Instruction:", instruction)
    print("Num samples:", num_samples)

    os.makedirs(save_dir, exist_ok=True)

    image_rgb = load_image(image_path)
    all_task_lists = []

    for i in range(num_samples):
        print(f"[{i+1}/{num_samples}] Generating tasks...")

        tasks = generate_tasks_with_descriptions(
            image_rgb=image_rgb,
            instruction=instruction,
        )

        if not tasks:
            tasks = []

        all_task_lists.append(tasks)
        print(json.dumps(tasks, ensure_ascii=False))

    save_path = _build_test_save_path(
        save_dir=save_dir,
        instruction=instruction,
        suffix=f"{num_samples}task_lists.json",
    )

    # Keep the file as valid JSON while ensuring each sampled task list stays on one line.
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for idx, task_list in enumerate(all_task_lists):
            line = json.dumps(task_list, ensure_ascii=False)
            suffix = "," if idx < len(all_task_lists) - 1 else ""
            f.write(f"  {line}{suffix}\n")
        f.write("]\n")

    print("\nSaved to:", save_path)
    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    # test_generate_tasks_with_descriptions(INSTRUCTION, IMAGE_PATH, SAVE_DIR, NUM_SAMPLES)
    test_get_point_vllm(INSTRUCTION, IMAGE_PATH, SAVE_DIR, NUM_SAMPLES)