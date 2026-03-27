import os
import cv2

import copy
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum, auto

import cv2
import numpy as np
import requests
from pytransform3d.rotations import active_matrix_from_angle
from pytransform3d.transformations import transform_from
from tvla_realenv.open3d_realsense_env import Open3dRealsenseEnv
from multi_pointing_vllm_get_point_utils import *

# 从工具函数脚本导入功能函数
from pick_and_place_utils import *

# ==============================
# 需要导入你的VLLM判断函数
# ==============================
from multi_pointing_vllm_get_point_utils import *

def load_rgb(image_path):
    """
    读取RGB图像
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def save_check_image(image_rgb, prefix, object_name, container_name=None, save_dir=None):

    os.makedirs(save_dir, exist_ok=True)

    object_name = object_name.replace(" ", "_")

    if container_name is not None:
        container_name = container_name.replace(" ", "_")

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if container_name:
        filename = f"{prefix}_{object_name}_to_{container_name}_{timestamp}.png"
    else:
        filename = f"{prefix}_{object_name}_{timestamp}.png"

    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    print(f"📸 Image saved to: {save_path}")

def test_pick(image, object_name):
    """
    测试 pick success
    """
    print("\n==============================")
    print("Testing PICK success detection")
    print("==============================")

    rgb = image

    result = check_grasp_success_vllm(rgb, object_name)

    print(f"Object: {object_name}")
    print(f"Result: {result}")

    if result:
        print("✅ Model thinks grasp SUCCESS")
    else:
        print("❌ Model thinks grasp FAILED")


def test_place(image, object_name, container_name):
    """
    测试 place success
    """
    print("\n==============================")
    print("Testing PLACE success detection")
    print("==============================")

    rgb = image

    result = check_place_success_vllm(rgb, object_name, container_name)

    print(f"Object   : {object_name}")
    print(f"Container: {container_name}")
    print(f"Result   : {result}")

    if result:
        print("✅ Model thinks place SUCCESS")
    else:
        print("❌ Model thinks place FAILED")


if __name__ == "__main__":

    # ============================
    # 这里换成你自己的测试图片
    # ============================

    rs_env_left = Open3dRealsenseEnv("f1471338")
    rs_env_right = Open3dRealsenseEnv("f1471193")

    with open("data/20260202_170600/camera_results.json", "r") as f:
        left_cam_results = json.load(f)

    with open("data/20260131_204802/camera_results.json", "r") as f:
        right_cam_results = json.load(f)

    obs_rs_left = rs_env_left.reset()
    obs_rs_right = rs_env_right.reset()

    image_rgb = obs_rs_left["rgb"]

    object_name = "white ball"
    container_name = "basket"

    SAVE_DIR = "/home/zhangzhao/tvla-realenv/lyt/pick_and_place_2.0/test/"

    # 保存图像
    save_check_image(image_rgb, prefix="pick_check", object_name=object_name, save_dir=SAVE_DIR)

    # ============================
    # 运行测试
    # ============================

    # test_pick(pick_image, object_name)

    test_place(image_rgb, object_name, container_name)
