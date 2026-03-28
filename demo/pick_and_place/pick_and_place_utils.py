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


def omni_decode_points(output: str) -> List[List[float]]:
    """
    通用解码器：从 VLM 模型的字符串输出中解析 2D 点坐标。
    
    这是一个多策略解析函数，能够处理各种 VLM 模型（Qwen、GPT-4V、XML 风格等）的输出格式。
    函数会依次尝试多种解析策略，直到成功提取到点坐标为止。
    
    支持的格式：
    1. Qwen/JSON 字典格式: '[{"point_2d": [[x, y]], "label": "target"}]'
    2. 列表/元组格式: '[[x1, y1], [x2, y2]]' 或 '[(x1, y1)]'
    3. XML 标签格式: '<point>[[x, y]]</point>' 或 '<points>[x1,y1],[x2,y2]</points>'
    4. XML 属性格式: '<point x="63.5" y="44.5" alt="label">text</point>'
    5. 自然语言格式: 'The point is at 100, 200'
    6. Markdown 代码块格式: '```json\n[x, y]\n```'
    
    解析策略（按顺序尝试）：
    - 策略1: 从 XML 属性中提取坐标（如 x="10" y="20"）
    - 策略2: 预处理文本，去除 Markdown 和 XML 标签
    - 策略3: 使用 Python 字面量解析（ast.literal_eval）处理结构化数据
    - 策略4: 使用正则表达式作为兜底方案，从自然语言中提取坐标
    
    参数:
        output: VLM 模型输出的原始字符串
    
    返回:
        List[List[float]]: 点坐标列表，每个元素为 [x, y]。如果未找到任何点，返回空列表。
    
    使用示例:
        >>> omni_decode_points('[{"point_2d": [[100, 200]]}]')
        [[100.0, 200.0]]
        >>> omni_decode_points('The point is at (50, 75)')
        [[50.0, 75.0]]
    """
    if not isinstance(output, str) or not output.strip():
        return []

    points = []

    # --- Strategy 1: XML Attribute Extraction ---
    # Matches: <point x="10" y="20"> or <point y="20" x="10">
    if '<point' in output.lower():
        points = _extract_from_xml_attributes(output)
        if points:
            return points

    # --- Strategy 2: Clean Markdown & XML Tags ---
    text = _preprocess_text(output)

    # --- Strategy 3: Python Literal / JSON Parsing ---
    # We try ast.literal_eval first because models often use single quotes or
    # Python-like structures that valid JSON doesn't support.
    try:
        # Help literal_eval by removing common prefix labels like "Output: "
        clean_text = re.sub(r'^[a-zA-Z0-9_\s]+:\s*', '', text)
        data = ast.literal_eval(clean_text)
        points = _parse_structured_data(data)
        if points:
            return points
    except (ValueError, SyntaxError, MemoryError):
        pass

    # --- Strategy 4: Regex Fallback (The "Catch-All") ---
    # Matches: [10, 20], (10, 20), or even raw 10.5, 20.1
    # This captures points embedded in natural language.
    points = _extract_points_by_regex(text)

    return points

def _preprocess_text(text: str) -> str:
    """
    文本预处理函数：清理和提取文本中的有效内容。
    
    该函数用于在解析点坐标之前清理文本，主要做两件事：
    1. 去除 Markdown 代码块包裹（如 ```json ... ```）
    2. 从 XML 标签中提取内容（如 <point>...</point>）
    
    这样可以让后续的解析函数更容易处理纯坐标数据，而不受格式标记的干扰。
    
    参数:
        text: 需要预处理的原始文本字符串
    
    返回:
        str: 清理后的文本，去除 Markdown 和 XML 标签包裹，只保留核心内容
    
    示例:
        >>> _preprocess_text('```json\n[100, 200]\n```')
        '[100, 200]'
        >>> _preprocess_text('<point>[50, 75]</point>')
        '[50, 75]'
    """
    # 去除 Markdown 代码块（支持 json、python、html 等语言标记）
    text = re.sub(r'```(?:json|python|html)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)

    # 从 <point> 或 <points> 标签中提取内容（如果存在）
    tag_match = re.search(r'<(?:point|points)>(.*?)</(?:point|points)>', text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        text = tag_match.group(1)

    return text.strip()

def _parse_structured_data(data: Any) -> List[List[float]]:
    """
    递归解析结构化数据，从 Python 对象（字典、列表等）中提取点坐标。
    
    该函数采用递归策略，能够处理各种嵌套的数据结构：
    - 字典：查找常见的键名（如 "point_2d", "points", "point", "coordinates"）
    - 列表/元组：识别单个点 [x, y] 或多个点的列表 [[x1, y1], [x2, y2]]
    - 嵌套结构：递归处理多层嵌套的数据
    
    参数:
        data: 待解析的 Python 对象，可以是字典、列表、元组等
    
    返回:
        List[List[float]]: 提取到的点坐标列表，每个元素为 [x, y]
    
    示例:
        >>> _parse_structured_data({"point_2d": [[100, 200]]})
        [[100.0, 200.0]]
        >>> _parse_structured_data([[10, 20], [30, 40]])
        [[10.0, 20.0], [30.0, 40.0]]
        >>> _parse_structured_data([50, 75])
        [[50.0, 75.0]]
    """
    points = []

    if isinstance(data, dict):
        # 处理 Qwen/VLM 常用的键名
        for key in ["point_2d", "points", "point", "coordinates"]:
            if key in data:
                return _parse_structured_data(data[key])

    elif isinstance(data, (list, tuple)):
        if not data:
            return []

        # 检查是否为单个点格式: [x, y]
        if len(data) == 2 and all(isinstance(x, (int, float)) for x in data):
            return [[float(data[0]), float(data[1])]]

        # 检查是否为嵌套结构: [[x, y], ...] 或 [{"point_2d": [x, y]}, ...]
        for item in data:
            extracted = _parse_structured_data(item)
            if extracted:
                points.extend(extracted)

    return points


def _extract_from_xml_attributes(text: str) -> List[List[float]]:
    """
    从 XML 属性或特殊格式的文本中提取点坐标。
    
    该函数使用多种正则表达式模式来匹配不同格式的坐标表示：
    1. Click(x, y) 格式：如 "Click(100.5, 200.3)"
    2. 括号坐标格式：如 "(100.5, 200.3)" 或 "(100.5,200.3)"
    3. XML 属性格式：如 'x="100" y="200"' 或 'x1="100.5" y2="200.3"'
    4. 特殊格式：如 "p = 100, 200" 或 "1 = 100, 200"（坐标需要除以10）
    
    参数:
        text: 包含坐标信息的文本字符串
    
    返回:
        List[List[float]]: 提取到的所有点坐标列表
    
    示例:
        >>> _extract_from_xml_attributes('<point x="100" y="200">')
        [[100.0, 200.0]]
        >>> _extract_from_xml_attributes('Click(50.5, 75.3)')
        [[50.5, 75.3]]
    """
    all_points = []
    
    # 模式1: Click(x, y) 格式
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)

    # 模式2: 括号坐标格式 (x, y) 或 (x,y)
    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)
    
    # 模式3: XML 属性格式 x="100" y="200" 或 x1="100.5" y2="200.3"
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)
    
    # 模式4: 特殊格式 p = 100, 200 或 1 = 100, 200（坐标需要除以10转换为实际值）
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            all_points.append(point)

    return all_points


def _extract_points_by_regex(text: str) -> List[List[float]]:
    """
    使用正则表达式从文本中提取点坐标（兜底策略）。
    
    这是解析函数的最后一道防线，当其他解析策略都失败时使用。
    该函数能够从自然语言或松散格式的文本中提取坐标对。
    
    提取策略（按优先级）：
    1. 优先匹配带括号的坐标：如 [x, y] 或 (x, y)
    2. 如果未找到括号格式，则匹配纯数字对：如 "100, 200"
    
    注意：为了避免重复提取，只有在没有找到括号格式时才会尝试提取纯数字对。
    
    参数:
        text: 包含坐标信息的文本字符串
    
    返回:
        List[List[float]]: 提取到的点坐标列表
    
    示例:
        >>> _extract_points_by_regex('The point is at [100, 200]')
        [[100.0, 200.0]]
        >>> _extract_points_by_regex('Coordinates: 50, 75')
        [[50.0, 75.0]]
        >>> _extract_points_by_regex('Points: (10, 20) and 30, 40')
        [[10.0, 20.0], [30.0, 40.0]]
    """
    points = []
    
    # 模式1: 匹配带括号的坐标格式 [x, y] 或 (x, y)
    bracket_pattern = r'[\[\(]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\]\)]'
    matches = re.findall(bracket_pattern, text)

    if matches:
        for m in matches:
            points.append([float(m[0]), float(m[1])])
    else:
        # 兜底策略：在文本中查找 "数字, 数字" 模式
        # 只有在没有找到括号格式时才使用，避免重复提取
        raw_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        matches = re.findall(raw_pattern, text)
        for m in matches:
            points.append([float(m[0]), float(m[1])])

    return points


# ===============================
# 机械臂模拟环境转换
# ===============================
class RealmanEnvWebSim:
    T_TCP2REALMANEEF = transform_from(
        active_matrix_from_angle(2, -np.pi / 3) @ np.array([[0, 0, 1],
                                                            [0, -1, 0],
                                                            [1, 0, 0]]),
        np.array([0, 0, 0.22])
    )

    def __init__(self, gripper_open=0.09):
        self.gripper_open = gripper_open
        self.reset()

    @staticmethod
    def T_from_realman_xyzrpy(xyzrpy):
        x, y, z, rx, ry, rz = xyzrpy
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        T = np.eye(4)
        T[:3,:3] = Rz @ Ry @ Rx
        T[:3,3] = [x,y,z]
        return T
    
    @staticmethod
    def realman_xyzrpy_from_T(T):
        """将 4x4 变换矩阵转换为 RealMan 的 xyzrpy"""
        x = T[0, 3]
        y = T[1, 3]
        z = T[2, 3]
        ry = np.arcsin(np.clip(-T[2, 0], -1, 1))
        if np.cos(ry) != 0:
            rx = np.arctan2(T[2, 1]/np.cos(ry), T[2, 2]/np.cos(ry))
            rz = np.arctan2(T[1, 0]/np.cos(ry), T[0, 0]/np.cos(ry))
        else:
            rx = 0
            rz = np.arctan2(-T[0, 1], T[1, 1])
        return np.array([x, y, z, rx, ry, rz])

    def reset(self):
        home_eef_xyzrpy = [-0.036,-0.220,0.352,3.141,0,-2.618]
        home_T_eef2base = RealmanEnvWebSim.T_from_realman_xyzrpy(np.array(home_eef_xyzrpy))
        self.home_T_tcp2base = home_T_eef2base @ RealmanEnvWebSim.T_TCP2REALMANEEF
        self.home_T_tcp2base_xyzrpy = RealmanEnvWebSim.realman_xyzrpy_from_T(self.home_T_tcp2base)
        self.cached_gripper = self.gripper_open
        
        return {"Ttcp2base": self.home_T_tcp2base, "gripper_open": self.cached_gripper}


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
# 可视化工具
# ===============================
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


# ===============================
# 通讯逻辑 (批量发送动作序列)
# ===============================
def action_to_payload(action):
    """将单个动作转换为web API的payload格式"""
    # gripper 转换
    gripper = 1 if action["gripper_open"] > 0.05 else 0 

    # 默认字段
    payload = {
        "control_mode": action.get("control_mode", "pose"),
        "xyzrpy": None,
        "joint": None,
        "gripper": gripper,
        "use_moveit": action.get("use_moveit", False)
    }

    # Pose 控制
    if payload["control_mode"] == "pose":

        T_tcp2base = action["Ttcp2base"]
        T_eef2base = T_tcp2base @ np.linalg.inv(RealmanEnvWebSim.T_TCP2REALMANEEF)
        # 从变换矩阵中提取完整的 xyzrpy（包括从 graspnet 获取的旋转信息）
        eef2base_xyzrpy = RealmanEnvWebSim.realman_xyzrpy_from_T(T_eef2base)
    
        payload["xyzrpy"] = eef2base_xyzrpy.tolist()

    # Joint 控制
    elif payload["control_mode"] == "joint":

        payload["joint"] = action["joint"]
    
    return payload

def send_action_to_web(action, arm_name=None, web_batch_url=None, batch_use_moveit=True, 
                       batch_speed_pct=25, batch_blend_r_pct=5, gripper_open=0.09, gripper_close=0.03):
    if arm_name is None:
        raise ValueError("arm_name is required to send action to web")
    if web_batch_url is None:
        raise ValueError("web_batch_url is required to send action to web")
    
    """发送单个动作到web（保留用于兼容性）"""
    payload = action_to_payload(action)
    xyzrpy = np.array(payload["xyzrpy"])  # 使用实际发送给web的xyzrpy
    gripper = payload["gripper"]
    # print(f"📡 Sending Action -> EEF2Base XYZRPY: {np.round(xyzrpy, 3)}, Gripper: {gripper}")
    
    # 使用新的批量接口格式（单个动作包装为只有一个waypoint的批量请求）
    waypoint = {
        "xyzrpy": payload["xyzrpy"],
        "gripper": payload["gripper"]
    }
    batch_payload = {
        "waypoints": [waypoint],
        "use_moveit": batch_use_moveit,
        "rm_speed_pct": batch_speed_pct,
        "rm_blend_r_pct": batch_blend_r_pct
    }
    
    # 记录发送给web的数据
    arm_control_url_to_send = web_batch_url[f"{arm_name}_batch_control_url"]
    log_web_data(batch_payload, url=arm_control_url_to_send)

    while True:
        if cv2.waitKey(1) == ord("q"):
            raise KeyboardInterrupt("💥 User requested exit")

        try:
            res = requests.post(arm_control_url_to_send, json=batch_payload, timeout=(5, 60))
            data = res.json() if res.status_code == 200 else None
            
            if res.status_code != 200:
                time.sleep(1)
                continue

            if data.get("status", False) or data.get("done", False): # 兼容两种 API 字段
                T_curr_eef = RealmanEnvWebSim.T_from_realman_xyzrpy(np.array(data["xyzrpy"]))
                T_curr_tcp = T_curr_eef @ RealmanEnvWebSim.T_TCP2REALMANEEF
                g_val = data.get("gripper", 1)
                g_curr = gripper_open if g_val == 1 else gripper_close
                return T_curr_tcp, g_curr
            else:
                time.sleep(0.2)
        except Exception as e:
            print(f"⚠ Web error: {e}")
            time.sleep(1)

def send_action_sequence_to_web(action_sequence, action_names=None, arm_name=None, web_batch_url=None,
                                 batch_use_moveit=True, batch_speed_pct=25, batch_blend_r_pct=5,
                                 gripper_open=0.09, gripper_close=0.03):
    """
    一次性发送整个任务的动作序列到web，等待所有动作执行完成
    
    Args:
        action_sequence: 动作列表，每个动作是 {"Ttcp2base": T, "gripper_open": g}
        action_names: 可选的动作名称列表，用于日志输出
        arm_name: 机械臂名称（'left_arm' 或 'right_arm'）
        web_batch_url: WEB_BATCH_URL字典
        batch_use_moveit: moveit 总开关
        batch_speed_pct: RealMan 机械臂速度百分比
        batch_blend_r_pct: RealMan 机械臂混合半径百分比
        gripper_open: 夹爪打开值
        gripper_close: 夹爪关闭值
    
    Returns:
        (T_curr_tcp, g_curr): 执行完成后的最终位姿和夹爪状态
    """

    if arm_name is None:
        raise ValueError("arm_name is required to send action sequence to web")
    if web_batch_url is None:
        raise ValueError("web_batch_url is required to send action sequence to web")
    
    # 将动作序列转换为waypoints格式
    waypoints = []
    for i, action in enumerate(action_sequence):
        payload = action_to_payload(action)
        waypoint = {
            "control_mode": payload["control_mode"],
            "xyzrpy": payload["xyzrpy"],
            "joint": payload["joint"],
            "gripper": payload["gripper"],
            "use_moveit": payload["use_moveit"]
        }
        waypoints.append(waypoint)

    # 构造批量请求payload
    batch_payload = {
        "waypoints": waypoints,
        "use_moveit": batch_use_moveit,
        "rm_speed_pct": batch_speed_pct,
        "rm_blend_r_pct": batch_blend_r_pct
    }
    
    #print(f"\n📡 发送任务序列到web ({len(action_sequence)}个动作)...")
    
    # 打印发送给web的waypoints数据
    print("🔍 发送给web的waypoints数据:")
    for i, wp in enumerate(waypoints):
        action_name = action_names[i] if action_names and i < len(action_names) else f"Action_{i+1}"
        print(f"📋 [{i+1}/{len(waypoints)}] {action_name} -> control_mode: {wp['control_mode']}, xyzrpy: {wp['xyzrpy']}, joint: {wp['joint']}, gripper: {wp['gripper']}, use_moveit: {wp['use_moveit']}")
    
    # 记录发送给web的数据
    arm_control_url_to_send = web_batch_url[f"{arm_name}_batch_control_url"]
    log_web_data(batch_payload, url=arm_control_url_to_send)
    
    try:
        res = requests.post(
            arm_control_url_to_send,
            json=batch_payload,
            timeout=(5, 300)
        )

        data = res.json() if res.status_code == 200 else None

        if res.status_code != 200 or data is None:
            return False, False, None, None, None

        status = data.get("status", False)
        plan_success = data.get("plan_success", True)
        failed_wp = data.get("failed_waypoint", -1)

        # 获取当前pose
        if "cur_eef_xyzrpy" in data and data["cur_eef_xyzrpy"] is not None:
            T_curr_eef = RealmanEnvWebSim.T_from_realman_xyzrpy(
                np.array(data["cur_eef_xyzrpy"])
            )
            T_curr_tcp = T_curr_eef @ RealmanEnvWebSim.T_TCP2REALMANEEF
        else:
            T_curr_tcp = None

        g_val = data.get("cur_gripper", 1)
        g_curr = gripper_open if g_val == 1 else gripper_close

        return status, plan_success, failed_wp, T_curr_tcp, g_curr

    except Exception as e:
        print(f"⚠ web exception: {e}")
        return False, False, None, None, None
        
    """
    # 尝试批量发送
    try:
        res = requests.post(
            arm_control_url_to_send, 
            json=batch_payload, 
            timeout=(5, 300)        # 增加超时时间，因为要执行多个动作
        )  

        data = res.json() if res.status_code == 200 else None
        
        if res.status_code == 200 and data:
            # 检查是否所有动作都执行完成
            if data.get("status", False):
                if data["status"] is not True:
                    print(f"❌ 任务序列执行失败!")
                    raise ValueError(
                        f"[TASK STATUS NOT TRUE] data['status'] = {data['status']}, "
                        f"full response = {data}"
                    )
                else:
                    print(f"✅ 任务序列执行完成!")
                
                    # 获取最终状态
                    if "cur_eef_xyzrpy" in data and data["cur_eef_xyzrpy"] is not None:
                        T_curr_eef = RealmanEnvWebSim.T_from_realman_xyzrpy(np.array(data["cur_eef_xyzrpy"]))
                        T_curr_tcp = T_curr_eef @ RealmanEnvWebSim.T_TCP2REALMANEEF

                    else:

                        print("⚠ cur_eef_xyzrpy is None, using last commanded pose")

                        # fallback：使用最后一个pose动作
                        for action in reversed(action_sequence):

                            if action.get("control_mode") == "pose":

                                T_curr_tcp = action["Ttcp2base"]
                                break  
                        
                    g_val = data["cur_gripper"]
                    g_curr = gripper_open if g_val == 1 else gripper_close
                    return T_curr_tcp, g_curr
            else:
                print(f"❌ data.get('status', False) failed!")
                raise ValueError(
                    f"[TASK FAILED OR STATUS MISSING] "
                    f"data.get('status') = {data.get('status')}, "
                    f"full response = {data}"
                )
        else:
            print(f"❌ res.status_code == 200 and data failed!")
            raise ValueError(
                f"[HTTP OR EMPTY DATA ERROR] "
                f"status_code = {res.status_code}, "
                f"data = {data}, "
                f"response_text = {res.text}"
            )
    except Exception as e:
        print(f"⚠ 批量请求异常: {e}")
        raise ValueError(
            f"[send_action_sequence_to_web FAILED] "
            f"Exception type = {type(e).__name__}, "
            f"message = {e}"
        )
    """

# ===============================
# 坐标计算工具
# ===============================
def make_target_T(obs, u, v, rs_env, cam_results, ref_T, z_offset=0.0):
    """
    ref_T: 参考姿态（通常是 home_T），确保旋转矩阵永远是标准向下的。
    """
    T = copy.deepcopy(ref_T) 
    d = obs["depth"][v, u] / rs_env.meta_obs["depth_scale"]
    
    # 深度有效性过滤
    if d <= 0 or d > 1.2:
        print("⚠ Warning: Invalid depth, using default 0.6m")
        d = 0.6

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


if __name__ == "__main__":
    test_cases = [
        '[{"point_2d": [[100, 200]], "label": "eye"}]',        # Qwen-style
        '<point x="63.5" y="44.5">Mountain</point><point x="63.8" y="44.5">Mountain</point>',           # Tag attributes
        '```json\n[[10, 20], [30, 40]]\n```',                  # Markdown
        'The center is at (500, 500) and (789, 1000).',                        # Natural language
        '<points>[123, 456]</points>',
        '<points>[[122, 333], [222, 333]]</points>',           # Custom tags
        'point: 12.5, 13.5',                                   # Lazy labeling
    ]

    for case in test_cases:
        print(f"Input: {case}")
        print(f"Parsed: {omni_decode_points(case)}\n")
