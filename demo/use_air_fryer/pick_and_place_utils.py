import ast
import re
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import copy
import json
import os
import time
from datetime import datetime
import cv2
import requests
from pytransform3d.rotations import active_matrix_from_angle
from pytransform3d.transformations import transform_from




# ===============================
# 日志记录功能（简单版本：只记录发送给web的数据）
# ===============================
_log_file = None  # 全局变量，保存当前会话的日志文件路径
def log_web_data(payload, log_dir="lyt/logs", url=None, web_control_url=None):
    """
    记录发送给web的数据
    
    Args:
        payload: 发送给web的payload数据
        log_dir: 日志文件保存目录
        url: 请求的URL，如果为None则使用默认的WEB_CONTROL_URL
        web_control_url: 默认的WEB_CONTROL_URL字典，如果url为None时使用
    """
    global _log_file
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果是第一次调用，创建新的日志文件
    if _log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = os.path.join(log_dir, f"web_send_{timestamp}.json")
        print(f"📝 日志文件: {_log_file}")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "url": url if url is not None else (web_control_url if web_control_url is not None else "unknown"),
        "payload": payload
    }
    
    # 追加到文件（作为数组）
    if os.path.exists(_log_file):
        with open(_log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(log_entry)
        else:
            data = [data, log_entry]
        with open(_log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        # 新文件，创建数组
        with open(_log_file, 'w', encoding='utf-8') as f:
            json.dump([log_entry], f, ensure_ascii=False, indent=2)

# 保存检查图像
def save_check_image(image_rgb, prefix, object_name, container_name=None, save_dir="lyt/logs"):

    os.makedirs(save_dir, exist_ok=True)

    object_name = object_name.replace(" ", "_")

    if container_name is not None:
        container_name = container_name.replace(" ", "_")

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if container_name:
        filename = f"{timestamp}_check_{prefix}_{object_name}_to_{container_name}.png"
    else:
        filename = f"{timestamp}_check_{prefix}_{object_name}.png"

    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    print(f"📸 Image saved to: {save_path}")

def save_pointed_image(image_rgb, point_2d, save_dir="logs", prefix="track"):
    import cv2
    import numpy as np
    import os
    import time

    os.makedirs(save_dir, exist_ok=True)

    img = np.array(image_rgb).copy()

    # 处理类型
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 画点
    if point_2d is not None:
        x, y = int(point_2d[0]), int(point_2d[1])
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    path = os.path.join(save_dir, filename)

    cv2.imwrite(path, img)
    print(f"📸 Saved pointed image: {path}")


# ===============================
# 可视化工具
# ===============================
# 可视化RGB图像并打点
def visualize_rgb_with_point(rgb, point=None, window_name="Image"):

    import cv2
    import numpy as np

    if rgb is None:
        print("❌ rgb is None")
        return

    img = np.array(rgb)

    # 保证内存连续（非常关键）
    img = np.ascontiguousarray(img)

    # =========================
    # 处理float
    # =========================
    if img.dtype != np.uint8:

        img_min = img.min()
        img_max = img.max()

        if img_max <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # =========================
    # RGB -> BGR
    # =========================
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # =========================
    # 打点
    # =========================
    if point is not None:

        x = int(point[0])
        y = int(point[1])

        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

        cv2.putText(
            img,
            f"({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    print("Image shape:", img.shape, "dtype:", img.dtype)

    cv2.imshow(window_name, img)

    print("Press q to continue")

    while True:
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyWindow(window_name)

# 以2D打点为中心裁剪图像
def crop_image_around_point(image_rgb, point_2d, crop_size=480):
    """
    以 2D 打点为中心裁剪图像，用于 check 阶段放大目标区域。

    Args:
        image_rgb: RGB 图像，shape=(H, W, 3)
        point_2d: 中心点 [x, y]
        crop_size: 正方形裁剪边长（像素）

    Returns:
        裁剪后的 RGB 图像；如果 point_2d 无效则返回原图
    """
    if image_rgb is None or point_2d is None:
        return image_rgb

    h, w = image_rgb.shape[:2]
    crop_size = int(max(32, min(crop_size, h, w)))
    half = crop_size // 2

    x = int(round(point_2d[0]))
    y = int(round(point_2d[1]))

    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    x1 = x - half
    y1 = y - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image_rgb[y1:y2, x1:x2].copy()


# ===============================
# 坐标计算工具
# ===============================
def make_target_T(obs, u, v, rs_env, cam_results, ref_T, z_offset=0.0):
    """
    ref_T: 参考姿态（通常是 home_T），确保旋转矩阵永远是标准向下的。
    """
    T = copy.deepcopy(ref_T) 
    d = obs["depth"][v, u] / rs_env.meta_obs["depth_scale"]
    
    """
    # 深度有效性过滤
    if d <= 0 or d > 1.2:
        print("⚠ Warning: Invalid depth, using default 0.6m")
        d = 0.6
    """
    
    # 投影到基座坐标系
    intrinsic_inv = np.linalg.inv(np.array(rs_env.meta_obs["intrinsic"]))
    xyz_cam = intrinsic_inv @ (np.array([u, v, 1.0]) * d)
    xyz_base = np.array(cam_results["Tcam2base"]) @ np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0])
    
    # 应用高度偏移
    xyz_base[2] += z_offset
    # 强制安全限位：绝不允许 Z 轴低于桌面以下 1cm
    xyz_base[2] = max(xyz_base[2], -0.01) 
    
    T[:3, 3] = xyz_base[:3]
    return T

# 对矩阵的x,y,z进行校正
def make_lift_T(T, lift_x= 0.0, lift_y=0.0, lift_z=0.0):
    T_lift = copy.deepcopy(T)
    T_lift[0, 3] += lift_x
    T_lift[2, 3] += lift_z
    T_lift[1, 3] += lift_y
    return T_lift

