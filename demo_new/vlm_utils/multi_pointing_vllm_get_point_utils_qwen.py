"""
此函数库调用Qwen来实现高层推理功能。
"""
import json
import re
import os
import sys
import ast
from typing import Any, List
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_new.vlm_utils.pointing_vllm_client import VLLMOnlineClient

# =========================================================
# Global VLM Configuration
# =========================================================

"""
# ER1.5
BASE_URL = "http://172.28.102.11:22215/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Embodied-R1.5"
"""


# Qwen3.6-35B-A3B
BASE_URL = "http://172.28.102.11:26353/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3.6-35B-A3B"


"""
# Qwen3-VL
BASE_URL = "http://172.28.102.11:22003/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen3.6-27B"
"""

TMP_IMAGE_PATH = "tmp_vlm_image.png"

_global_client = None

# 获取VLM客户端
def get_vlm_client():
    """Singleton VLM client"""
    global _global_client

    if _global_client is None:
        _global_client = VLLMOnlineClient(
            base_url=BASE_URL,
            api_key=API_KEY,
            model_name=MODEL_NAME
        )

    return _global_client


# =========================================================
# Utility Functions
# =========================================================

# 保存图像到临时文件
def save_image_tmp(image_rgb):
    """Save image to temporary file"""
    img = image_rgb.copy()

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(TMP_IMAGE_PATH)

    return TMP_IMAGE_PATH

# JSON-response helpers
def _parse_json_candidate(text: str):
    """Try parsing text as JSON or as a Python literal."""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None

def _flatten_text_content(payload):
    """Flatten OpenAI-compatible content payloads into plain text."""
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    if isinstance(payload, list):
        parts = []
        for item in payload:
            text = _flatten_text_content(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)

    if isinstance(payload, dict):
        for key in ("text", "content", "value"):
            if key in payload:
                text = _flatten_text_content(payload[key]).strip()
                if text:
                    return text
        return ""

    return str(payload)

def _extract_message_text(message):
    """Read the most useful text field from a model message."""
    for attr in ("content", "text", "reasoning_content"):
        text = _flatten_text_content(getattr(message, attr, None)).strip()
        if text:
            return text
    return ""


def _collect_message_text(message):
    """Collect all non-empty text fields from a model message."""
    parts = []
    for attr in ("content", "text", "reasoning_content"):
        text = _flatten_text_content(getattr(message, attr, None)).strip()
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts)

def _merge_extra_body(extra_body=None, disable_thinking=False):
    """Merge request extras while optionally disabling thinking output."""
    if extra_body is None:
        merged = {}
    elif isinstance(extra_body, dict):
        merged = dict(extra_body)
    else:
        return extra_body

    if disable_thinking:
        chat_template_kwargs = dict(merged.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        merged["chat_template_kwargs"] = chat_template_kwargs

    return merged or None

def _iter_json_candidates(text: str):
    """Yield balanced JSON-like blocks from free-form model output."""
    stack = []
    start = None
    quote_char = None
    escape = False

    for index, ch in enumerate(text):
        if quote_char is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                quote_char = None
            continue

        if stack and ch in ('"', "'"):
            quote_char = ch
            continue

        if ch in "{[":
            if start is None:
                start = index
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                continue

            left = stack[-1]
            expected = "}" if left == "{" else "]"
            if ch != expected:
                continue

            stack.pop()
            if not stack and start is not None:
                yield text[start:index + 1]
                start = None

def extract_first_json(text):
    """Extract the first valid JSON-like payload from model output."""
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("No JSON found in model output")

    cleaned = re.sub(
        r"```(?:json|python|text)?\s*(.*?)\s*```",
        r"\1",
        text.strip(),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    parsed = _parse_json_candidate(cleaned)
    if parsed is not None:
        return parsed

    for candidate in _iter_json_candidates(cleaned):
        parsed = _parse_json_candidate(candidate.strip().rstrip(",;"))
        if parsed is not None:
            return parsed

    print("\n========== RAW MODEL OUTPUT ==========\n")
    print(cleaned)

    raise RuntimeError("Failed to extract valid JSON")


def extract_first_json_or_none(text):
    """Extract the first valid JSON-like payload, or return None."""
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = re.sub(
        r"```(?:json|python|text)?\s*(.*?)\s*```",
        r"\1",
        text.strip(),
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    parsed = _parse_json_candidate(cleaned)
    if parsed is not None:
        return parsed

    for candidate in _iter_json_candidates(cleaned):
        parsed = _parse_json_candidate(candidate.strip().rstrip(",;"))
        if parsed is not None:
            return parsed

    return None


def _read_success_flag(data, key):
    data = _unwrap_single_item(data)

    if not isinstance(data, dict) or key not in data:
        return False

    value = data[key]

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return False

def _request_json_response(
    client,
    messages,
    *,
    model=None,
    max_tokens=512,
    temperature=0.2,
    extra_body=None,
    retry_max_tokens=None,
    **kwargs,
):
    """Call the model and robustly recover a JSON payload from its reply."""
    model = model or MODEL_NAME
    merged_extra_body = _merge_extra_body(extra_body, disable_thinking=True)

    token_budgets = [max_tokens]
    if retry_max_tokens is None and max_tokens < 2048:
        token_budgets.append(max(max_tokens * 2, 2048))
    elif retry_max_tokens is not None and retry_max_tokens > max_tokens:
        token_budgets.append(retry_max_tokens)

    last_error = None
    last_content = ""

    for attempt_index, current_max_tokens in enumerate(dict.fromkeys(token_budgets)):
        response = client.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=current_max_tokens,
            temperature=temperature,
            extra_body=merged_extra_body,
            **kwargs,
        )

        last_content = _extract_message_text(response.choices[0].message)

        try:
            return extract_first_json(last_content)
        except RuntimeError as exc:
            last_error = exc
            if attempt_index < len(token_budgets) - 1:
                next_budget = token_budgets[attempt_index + 1]
                print(
                    f"Warning: failed to extract JSON with max_tokens={current_max_tokens}; "
                    f"retrying with max_tokens={next_budget}."
                )

    if last_content:
        print("\n========== RAW MODEL OUTPUT ==========\n")
        print(last_content)

    raise last_error or RuntimeError("Failed to extract valid JSON from model output")

def _unwrap_single_item(data):
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], (dict, list)):
        data = data[0]
    return data

# 规范化任务条目
def _normalize_task_entry(item):
    if not isinstance(item, dict):
        return None

    pick = item.get("pick")
    place = item.get("place")

    if pick is None or place is None:
        return None

    pick = str(pick).strip()
    place = str(place).strip()

    if not pick or not place:
        return None

    return {
        "pick": pick,
        "place": place,
    }

# 规范化任务列表
def _normalize_task_list(data):
    data = _unwrap_single_item(data)

    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]

    if isinstance(data, dict):
        task = _normalize_task_entry(data)
        return [task] if task is not None else []

    if not isinstance(data, list):
        return []

    tasks = []
    for item in data:
        item = _unwrap_single_item(item)

        if isinstance(item, dict) and "tasks" in item:
            tasks.extend(_normalize_task_list(item["tasks"]))
            continue

        task = _normalize_task_entry(item)
        if task is not None:
            tasks.append(task)

    return tasks

# 规范化完成结果
def _normalize_completion_result(data):
    data = _unwrap_single_item(data)

    if isinstance(data, dict):
        return bool(data.get("completed", False)), str(data.get("reason", "")).strip()

    if isinstance(data, list):
        for item in data:
            item = _unwrap_single_item(item)
            if isinstance(item, dict) and ("completed" in item or "reason" in item):
                return bool(item.get("completed", False)), str(item.get("reason", "")).strip()

    raise RuntimeError(f"Unexpected completion result format: {data!r}")

def _normalize_roast_result(data):
    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        items = []
        minutes = 0

        for d in data:
            if not isinstance(d, dict):
                continue
            items.extend(d.get("items", []))
            minutes = max(minutes, int(d.get("minutes", 0)))

        return {
            "items": list(set(items)),  # 去重
            "minutes": minutes if minutes > 0 else 20
        }

    return {"items": [], "minutes": 20}

# 验证模型身份
def test_model_identity():
    client = get_vlm_client()

    response = client.client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "What model are you?"
        }],
        max_tokens=50,
        temperature=0
    )

    print(response.choices[0].message.content)


# =========================================================
# Point Decoding
# =========================================================

# 通用解码器：从VLM模型的字符串输出中解析2D点坐标。
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

# 文本预处理函数：清理和提取文本中的有效内容。
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

# 递归解析结构化数据，从Python对象（字典、列表等）中提取点坐标。
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

# 从XML属性中提取坐标
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

# 从自然语言中提取坐标
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


# =========================================================
# 调用模型解析多步抓取和放置任务（只适用于明确指令）
# =========================================================
def parse_multi_pick_place_tasks(text_prompt):

    # ====== VLM call ======
    client = get_vlm_client()

    prompt = f"""
        You are a robot task planner.

        Given a human instruction, decompose it into multiple sequential pick-and-place tasks.

        You MUST output ONLY JSON.
        The output MUST be a JSON object (NOT a list).

        Required format:
        {{
            "num_tasks": N,
            "tasks": [
                {{"pick": "...", "place": "..."}}
            ]
        }}

        Instruction:
        {text_prompt}
    """

    response = client.client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )

    content = _collect_message_text(response.choices[0].message)
    data = extract_first_json_or_none(content)
    return _read_success_flag(data, "grasp_success")

    # =========================================================
    # Step 1: 去 markdown 包裹
    # =========================================================
    content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(r"```$", "", content.strip())

    # =========================================================
    # Step 2: 修复 JSON
    # =========================================================
    def _fix_broken_json(s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s+", " ", s)

        if s.count('[') > s.count(']'):
            s += ']' * (s.count('[') - s.count(']'))

        if s.count('{') > s.count('}'):
            s += '}' * (s.count('{') - s.count('}'))

        s = re.sub(r"[^\}\]]+$", "", s)
        return s

    content = _fix_broken_json(content)

    # =========================================================
    # Step 3: JSON 解析
    # =========================================================
    try:
        data = json.loads(content)
    except Exception:
        try:
            data = ast.literal_eval(content)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to parse:\n{content}") from e

    # =========================================================
    # Step 4: 直接获取 tasks，不 unwrap 最外层
    # =========================================================
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    elif isinstance(data, list):
        tasks = data
        data = {"tasks": tasks}
    else:
        raise ValueError(f"❌ Invalid structure:\n{data}")

    # 如果 tasks 是 dict，wrap 成 list
    if isinstance(tasks, dict):
        tasks = [tasks]

    # =========================================================
    # Step 5: 清洗任务
    # =========================================================
    cleaned_tasks = []

    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            print(f"⚠ Skip invalid task {i}: {t}")
            continue

        pick = t.get("pick", None)
        place = t.get("place", None)

        # 自动修复字段（兼容奇葩输出）
        if pick is None:
            for k in t.keys():
                if "pick" in k.lower():
                    pick = t[k]

        if place is None:
            for k in t.keys():
                if "place" in k.lower():
                    place = t[k]

        if pick is None or place is None:
            print(f"⚠ Skip broken task {i}: {t}")
            continue

        cleaned_tasks.append({
            "pick": str(pick).strip(),
            "place": str(place).strip()
        })

    if len(cleaned_tasks) == 0:
        raise RuntimeError("❌ No valid tasks parsed")

    # =========================================================
    # Step 6: 输出标准格式
    # =========================================================
    result = {
        "num_tasks": len(cleaned_tasks),
        "tasks": cleaned_tasks
    }

    print("✅ Parsed task plan:", result)

    return result


# =========================================================
# 针对空气炸锅任务，调用模型从指令中拆解出pnp列表和旋转角度
# =========================================================
def parse_roast_with_timer(instruction: str):
    """
    Parse roasting instruction into:
    - pick & place tasks
    - rotate angle (derived from time)

    Returns:
        {
            "tasks": [...],
            "rotate_angle": int
        }
    """

    client = get_vlm_client()

    prompt = f"""
        You are a robot task planner.

        Extract structured information from the instruction.

        Instruction:
        {instruction}

        Requirements:

        1. Identify all food items to roast.

        2. Extract cooking time in minutes.

        3. Output JSON ONLY.

        Format:
        {{
            "items": ["sweet potato", "corn"],
            "minutes": 10
        }}

        Rules:
        - If multiple items, include all.
        - If no time is specified, set minutes = 20 by default.
        - Only use the allowed item names.
    """

    data = _request_json_response(
        client,
        [{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        max_tokens=512,
        temperature=0.2,
    )

    # =========================
    # 解析结果
    # =========================
    data = _normalize_roast_result(data)

    items = data.get("items", [])
    minutes = int(data.get("minutes", 20))

    # =========================
    # 构建任务列表
    # =========================
    tasks = [
        {"pick": item, "place": "open air fryer drawer"}
        for item in items if item
    ]

    # =========================
    # 时间 → 角度映射(10 min -> 45°)
    # =========================

    minutes = max(0, min(minutes, 40))

    rotate_angle = int(minutes * 4.5)

    return tasks, rotate_angle

# 能根据指令和观测拆出所有的子任务。
def generate_air_fryer_subtasks(image_rgb, instruction):

    client = get_vlm_client()

    img_path = save_image_tmp(image_rgb)

    # ---------------------------------
    # Prompt
    # ---------------------------------

    prompt = f"""
    You are a robot task planner for a kitchen robot.

    You are given:
    1. A kitchen RGB image
    2. A human instruction

    Your job:
    - Understand the instruction
    - Analyze the current scene
    - Infer environment states
    - Generate the minimal required robot subtasks

    ==================================================
    AVAILABLE SKILLS
    ==================================================

    Skill 0:
    open_container

    Args:
    {{
        "target": str
    }}

    --------------------------------------------------

    Skill 1:
    close_container

    Args:
    {{
        "target": str
    }}

    --------------------------------------------------

    Skill 2:
    pick_and_place

    Args:
    {{
        "pick": str,
        "place": str
    }}

    Motion Context:
    {{
        "pick_context": str,
        "place_context": str
    }}

    Allowed motion contexts:
    - normal
    - inside_air_fryer

    Use inside_air_fryer when manipulating objects inside the air fryer basket or drawer.

    --------------------------------------------------

    Skill 3:
    set_timer

    Args:
    {{
        "minutes": int
    }}

    ==================================================
    PLANNING RULES
    ==================================================

    1. First infer whether the air fryer drawer is:
        - open
        - closed

    2. Only generate necessary actions.

    3. If the drawer is already open:
        - do NOT generate Skill 0

    4. If the drawer is closed and the task requires interacting with food inside the air fryer:
        - first generate Skill 0

    5. Before setting timer:
        - drawer must be closed

    6. After all pick_and_place actions involving the air fryer are completed (either putting food in or taking food out), generate Skill 1 to close the drawer. The drawer should not be left open at the end of the task.

    7. Multiple food items require multiple pick_and_place actions.

    8. Visible objects in the image should be preferred for manipulation.

    9. If a food item mentioned in the instruction is not visible, you may assume it is already inside the air fryer basket unless the image clearly indicates otherwise.

    10. Use this assumption to continue planning normally according to the user instruction.

    11. Do NOT assume the task is already completed unless the image clearly shows the instruction has already been satisfied.

    12. Output subtasks in execution order.

    13. Output JSON ONLY.

    14. Do NOT output explanations.

    ==================================================
    USER INSTRUCTION
    ==================================================

    {instruction}

    ==================================================
    OUTPUT FORMAT
    ==================================================

    {{
        "subtasks": [
            {{
                "skill": 0,
                "args": {{
                    "target": "air fryer drawer"
                }}
            }},
            {{
                "skill": 2,
                "args": {{
                    "pick": "sweet potato",
                    "place": "air fryer basket"
                }},
                "motion": {{
                    "pick_context": "normal",
                    "place_context": "inside_air_fryer"
                }}
            }},
            {{
                "skill": 1,
                "args": {{
                    "target": "air fryer drawer"
                }}
            }},
            {{
                "skill": 3,
                "args": {{
                    "minutes": 30
                }}
            }}
        ]
    }}

    If the instruction is already satisfied:

    {{
        "subtasks": []
    }}
    """

    # ---------------------------------
    # 构建 VLM 输入
    # ---------------------------------

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    # ---------------------------------
    # 调用模型
    # ---------------------------------

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=512,
        temperature=0.2,
    )

    # ---------------------------------
    # JSON 提取
    # ---------------------------------


    # ---------------------------------
    # 结构清洗
    # ---------------------------------

    if not isinstance(data, dict):
        return []

    subtasks = data.get("subtasks", [])

    if not isinstance(subtasks, list):
        return []

    parsed_subtasks = []

    for task in subtasks:

        if not isinstance(task, dict):
            continue

        skill = task.get("skill", None)
        args = task.get("args", {})
        motion = task.get("motion", {})

        if skill is None:
            continue

        try:
            parsed_subtasks.append({
                "skill": int(skill),
                "args": args if isinstance(args, dict) else {},
                "motion": motion if isinstance(motion, dict) else {}
            })
        except:
            continue

    return parsed_subtasks

'''
# =========================================================
# 调用模型打点
# =========================================================
def get_point_vllm(image_rgb, text_prompt="you need to grasp the mug", save_path="debug_pointing_vllm.png", color=(0, 0, 255)):
    # Sampling parameters
    greedy = False
    seed = 3407
    top_p = 0.8
    top_k = 20
    temperature = 0.7
    repetition_penalty = 1.0
    presence_penalty = 1.5
    max_tokens = 4096  # out_seq_length

    # Video processing parameters (for mm_processor_kwargs)
    video_fps = 2  # Frames per second for video sampling
    video_do_sample_frames = True  # Enable frame sampling

    # Initialize client
    client = get_vlm_client()

    height, width = image_rgb.shape[:2]

    tmp_image_path = "vllm_image.png"

    Image.fromarray(image_rgb).save(tmp_image_path)
    
    test_case = {
    'idx': 2,
    'answer': '',
    'prompt': f"""
        Provide ONE 2D point for: {text_prompt}

        Rules:
            Output JSON only: [{{"point_2d":[x,y]}}]
            x,y must be in [0,1000]

        If the target is a container:
            The point MUST be inside the container
            The point MUST NOT be on any object inside the container
            The point MUST NOT be on the eage of the container
            Prefer a position near the center of the container
        If the target is a normal object:
            Choose a point near the top-center of the object
            Avoid edges of the object
        You may think step by step, but the final line must be exactly one JSON object.
    """,
    'image': tmp_image_path,
    'video': '',
    'type': 'single_image'
    }

    # Build messages for this test case
    messages = client.prepare_messages_from_test_case(test_case)

    # Generate output
    # Use mm_processor_kwargs for video processing (safe to use)
    # Note: Do NOT add top_k, repetition_penalty, presence_penalty - they cause crashes
    response = client.client.chat.completions.create(
        model=client.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        extra_body={
            "mm_processor_kwargs": {
                "fps": video_fps,
                "do_sample_frames": video_do_sample_frames
            }
        }
    )
    generated_text = response.choices[0].message.content

    import numpy as np

    pointing = (np.array(omni_decode_points(generated_text)) / 1000 * np.array([width, height]))[0]

    if save_path:
        img = cv2.imread(test_case["image"])
        img = cv2.circle(img, (int(pointing[0]), int(pointing[1])), 5, color, -1)
        cv2.imwrite(save_path, img)

    return pointing # x, y following opencv camera coord
'''

# =========================================================
# 调用模型检测抓取和放置是否成功
# =========================================================
def check_grasp_success_vllm(image_rgb, object_name):

    client = get_vlm_client()

    # =========================
    # 图像处理（关键）
    # =========================

    tmp_image_path = save_image_tmp(image_rgb)

    # print("Saved grasp check image:", tmp_image_path)

    # =========================
    # Prompt
    # =========================

    prompt = f"""
        You are a robot perception system.

        The robot attempted to grasp an object.

        Target object: {object_name}

        Look carefully at the image and determine whether the robot gripper is currently holding the object.

        SUCCESS conditions:
        - The object is clearly inside the robot gripper

        FAILURE conditions:
        - The gripper is empty
        - The object is not inside the gripper

        You may think step by step, but the final line must be exactly one JSON object.

        Example:
        {{
        "grasp_success": true
        }}
    """

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": tmp_image_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    response = client.client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )

    content = _collect_message_text(response.choices[0].message)
    data = extract_first_json_or_none(content)
    return _read_success_flag(data, "grasp_success")

    # print("[VLM grasp check raw output]")
    # print(content)

    # =========================
    # 提取JSON
    # =========================

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if match is None:
        print("❌ No JSON detected in model output")
        return False

    json_str = match.group()

    try:
        data = json.loads(json_str)
    except Exception:
        print("❌ JSON parse failed")
        return False

    if "grasp_success" not in data:
        print("❌ Invalid output format")
        return False

    result = bool(data["grasp_success"])

    return result

def check_place_success_vllm(image_rgb, object_name, container_name):

    client = get_vlm_client()

    # =========================
    # 图像处理
    # =========================

    tmp_image_path = save_image_tmp(image_rgb)

    # print("Saved place check image:", tmp_image_path)

    # =========================
    # Prompt
    # =========================

    prompt = f"""
        You are a robot perception system.

        The robot attempted to place an object into a container.

        Object: {object_name}
        Target container: {container_name}

        Look at the image and determine whether the object is already inside the container.

        SUCCESS conditions:
        - The object is clearly inside the container.

        FAILURE conditions:
        - The object is outside the container.

        You may think step by step, but the final line must be exactly one JSON object.

        Example:
        {{
        "place_success": true
        }}
    """

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": tmp_image_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    response = client.client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )

    content = _collect_message_text(response.choices[0].message)
    data = extract_first_json_or_none(content)
    return _read_success_flag(data, "place_success")

    # =========================
    # 提取JSON
    # =========================

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if match is None:
        print("❌ No JSON detected in model output")
        return False

    json_str = match.group()

    try:
        data = json.loads(json_str)
    except Exception:
        print("❌ JSON parse failed")
        return False

    if "place_success" not in data:
        print("❌ Invalid output format")
        return False

    result = bool(data["place_success"])

    return result


# =========================================================
# 根据指令生产pnp任务表
# =========================================================

# 生成一组 pnp 目标的名字
def generate_task_from_scene(image_rgb, instruction, pick_candidates=None, place_candidates=None):

    client = get_vlm_client()

    img_path = save_image_tmp(image_rgb)

    # 默认候选列表
    if pick_candidates is None:
        pick_candidates = [
            "baseball",
            "tennis ball",
            "cup",
            "carrot",
            "tiddy bear",
            "toy horse",
            "brush",
            "rubic's cube",
            "red pen",
            "glue stick",
        ]

    if place_candidates is None:
        place_candidates = [
            "pink plate",
            "white plate",
            "blue plate",
            "basket",
            "rubic's cube",
            "brown shelf"
        ]

    prompt = f"""
        You are a robot task planner.

        You are given:
        1. A tabletop RGB image
        2. A human instruction

        Instruction:
        {instruction}

        Pick candidates:
        {pick_candidates}

        Place candidates:
        {place_candidates}

        Goal:
        According to the requirements described in the directive, find a object that should be picked and a container that should be placed into, and return the name of the object and the name of the container.
        
        Rules:

        1. Select an object that matches the type required by the instruction and is visible in the image.
        2. Ignore any objects that are already inside the target container.
        3. Never choose objects that are already in the container, even if they match the instruction.
        4. If the current scene already satisfies the instruction (for example, all required objects are already inside the container or there are no valid objects to move), output empty values for both "pick" and "place".
        5. Return the name of the object and the name of the container.

        Return JSON ONLY.

        Format:

        {{
            "pick":"object_name",
            "place":"target_name"
        }}

        If the instruction is already satisfied, output:

        {{
            "pick": "",
            "place": ""
        }}
    """

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=512,
        temperature=0.2,
    )

    # -----------------------------
    # 结构清洗
    # -----------------------------

    tasks = _normalize_task_list(data)

    if not tasks:
        print("⚠️ No task detected")
        return None

    return tasks[0]


# 生成多组 pnp 目标的名字
def generate_tasks_from_scene(image_rgb, instruction, pick_candidates=None, place_candidates=None):

    client = get_vlm_client()

    img_path = save_image_tmp(image_rgb)

    # 默认候选
    if pick_candidates is None:
        pick_candidates = [
            "baseball",
            "tennis ball",
            "cup",
            "carrot",
            "tiddy bear",
            "toy horse",
            "brush",
            "rubic's cube",
            "red pen",
            "glue stick",
        ]

    if place_candidates is None:
        place_candidates = [
            "pink plate",
            "white plate",
            "blue plate",
            "basket",
            "rubic's cube",
            "shelf"
        ]

    # -----------------------------
    # 语义类别定义（关键）
    # -----------------------------

    prompt = f"""
        You are a robot task planner.

        You are given:
        1. A tabletop RGB image
        2. A human instruction

        Instruction:
        {instruction}

        Object semantic categories:

        Goal:
        Generate a sequence of pick-and-place tasks needed to satisfy the instruction.

        Rules:

        1. Only select objects that are visible in the image.
        2. Ignore objects that are already inside the correct container.
        3. If the scene already satisfies the instruction, return an empty list [].
        4. Each task must contain exactly one pick and one place.

        Return JSON ONLY.

        Format:

        [
            {{
                "pick": "object_name",
                "place": "target_name"
            }},
            ...
        ]

        Example:

        [
            {{
                "pick": "baseball",
                "place": "basket"
            }},
            {{
                "pick": "tennis ball",
                "place": "basket"
            }}
        ]

        If the instruction is already satisfied, output:

        [
            {{
                "pick": "",
                "place": ""
            }}
        ]
    """

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=512,
        temperature=0.2,
    )

    # -----------------------------
    # 结构清洗
    # -----------------------------

    tasks = _normalize_task_list(data)

    if not tasks:
        print("⚠️ No task detected")

    return tasks


# 带错误原因的生成多组 pnp 目标名字
def generate_tasks_from_scene_with_failure_reason(image_rgb, instruction, failure_reason=None, pick_candidates=None, place_candidates=None):

    client = get_vlm_client()

    img_path = save_image_tmp(image_rgb)

    # 默认候选
    if pick_candidates is None:
        pick_candidates = [
            "baseball",
            "tennis ball",
            "cup",
            "carrot",
            "tiddy bear",
            "toy horse",
            "brush",
            "rubic's cube",
            "red pen",
            "glue stick",
        ]

    if place_candidates is None:
        place_candidates = [
            "pink plate",
            "white plate",
            "blue plate",
            "basket",
            "rubic's cube",
            "shelf"
        ]

    # -----------------------------
    # 语义类别定义（关键）
    # -----------------------------

    prompt = f"""
        You are a robot task planner.

        You are given:
        1. A tabletop RGB image
        2. A human instruction

        Instruction:
        {instruction}

        Reason the instruction is not satisfied:
        {failure_reason}

        Goal:
        Generate pick-and-place tasks to fix the problem described in the reason.
        
        Planning procedure:

        Step1:
        Understand the instruction.

        Step2:
        Analyze the scene.

        Step3:
        Use the failure reason to identify which objects still need to be moved.

        Step4:
        Generate pick-and-place tasks that will satisfy the instruction.

        Rules:

        1. Only select objects visible in the image.
        2. Ignore objects already placed correctly.
        3. Each task must contain exactly one pick and one place.
        4. Do not generate unnecessary tasks.
        5. If the instruction is already satisfied, return [].

        Return JSON ONLY.

        Format:

        [
            {{
                "pick": "object_name",
                "place": "target_name"
            }},
            ...
        ]

        Example:

        [
            {{
                "pick": "baseball",
                "place": "basket"
            }},
            {{
                "pick": "tennis ball",
                "place": "basket"
            }}
        ]

        If the instruction is already satisfied, output:

        [
            {{
                "pick": "",
                "place": ""
            }}
        ]
    """

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=512,
        temperature=0.2,
    )

    # -----------------------------
    # 结构清洗
    # -----------------------------

    tasks = _normalize_task_list(data)

    if not tasks:
        print("⚠️ No task detected")

    return tasks


# 生成带方位描述的 pnp 目标列表
def generate_tasks_with_descriptions_old_prompt(image_rgb, instruction):

    client = get_vlm_client()
    img_path = save_image_tmp(image_rgb)

    prompt = f"""
        You are a robot.

        Convert instruction into a list of pick and place tasks.

        Instruction:
        {instruction}

        Rules:
        - pick = object
        - place = location
        - short natural phrases only
        - If multiple objects need to be moved, output multiple tasks
        - Each task = one pick + one place

        Output JSON list:

        [
        {{"pick": "...", "place": "..."}},
        {{"pick": "...", "place": "..."}}
        ]

        Examples:

        Instruction: put the ball on the right of the cube
        [
        {{"pick": "ball", "place": "right of cube"}}
        ]

        Instruction: put all balls into the basket
        [
        {{"pick": "red ball", "place": "inside basket"}},
        {{"pick": "blue ball", "place": "inside basket"}}
        ]

        Instruction: put toys into the white plate
        [
        {{"pick": "toy horse", "place": "inside white plate"}},
        {{"pick": "teddy bear", "place": "inside white plate"}}
        ]

        Instruction: put the ball between the cup and the plate
        [
        {{"pick": "ball", "place": "between cup and plate"}}
        ]
    """


    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.2,
    )

    tasks = _normalize_task_list(data)

    # -----------------------------
    # 后处理（关键增强）
    # -----------------------------
    cleaned_tasks = []
    for t in tasks:
        pick = t.get("pick", "").strip()
        place = t.get("place", "").strip()

        if pick and place:
            cleaned_tasks.append({
                "pick": pick,
                "place": place
            })

    if not cleaned_tasks:
        print("⚠️ No task detected")

    return cleaned_tasks

OBJECT_CATEGORIES = """
    Category definitions:

    toy:
    - teddy bear
    - toy horse
    - baseball
    - toy car
    - toy duck

    fruit:
    - apple
    - banana
    - orange
    - pear

    stationery:
    - marker
    - pen
    - pencil
    - glue stick
    - ruler
    - eraser

    container:
    - basket
    - plate
    - bowl
    - tray
    - bag
"""

def generate_tasks_with_descriptions(image_rgb, instruction):

    client = get_vlm_client()
    img_path = save_image_tmp(image_rgb)

    prompt = f"""
        You are a robot.

        Convert instruction into a list of pick and place tasks.

        Instruction:
        {instruction}

        Known object categories:
        {OBJECT_CATEGORIES}

        Rules:
        - pick = object
        - place = location
        - short natural phrases only
        - If multiple objects need to be moved, output multiple tasks
        - Each task = one pick + one place

        Output JSON list:

        [
        {{"pick": "...", "place": "..."}},
        {{"pick": "...", "place": "..."}}
        ]

        Examples:

        Instruction: put the ball on the right of the cube
        [
        {{"pick": "ball", "place": "right of cube"}}
        ]

        Instruction: put all balls into the basket
        [
        {{"pick": "red ball", "place": "inside basket"}},
        {{"pick": "blue ball", "place": "inside basket"}}
        ]

        Instruction: put toys into the white plate
        [
        {{"pick": "toy horse", "place": "inside white plate"}},
        {{"pick": "teddy bear", "place": "inside white plate"}}
        ]

        Instruction: put the ball between the cup and the plate
        [
        {{"pick": "ball", "place": "between cup and plate"}}
        ]
    """

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.2,
    )

    tasks = _normalize_task_list(data)

    # -----------------------------
    # 后处理（关键增强）
    # -----------------------------
    cleaned_tasks = []
    for t in tasks:
        pick = t.get("pick", "").strip()
        place = t.get("place", "").strip()

        if pick and place:
            cleaned_tasks.append({
                "pick": pick,
                "place": place
            })

    if not cleaned_tasks:
        print("⚠️ No task detected")

    return cleaned_tasks

# =========================================================
# Check Instruction Completion
# =========================================================
def check_instruction_complete(image_rgb, instruction):
    """
    Check whether the high-level instruction has been completed.

    Returns:
        (completed: bool, reason: str)
    """

    client = get_vlm_client()

    img_path = save_image_tmp(image_rgb)

    prompt = f"""
        You are a robot task checker.

        Given ONE tabletop RGB image,
        determine whether the following instruction is completed.

        Instruction:
        {instruction}

        Rules:

        1. The reason must be ONE short sentence.
        2. Mention all the objects that are not placed correctly.
        3. Mention the target container.
        4. If the instruction is satisfied, say:"all required objects are already in the correct place".

        You may think step by step, but the final line must be exactly one JSON object.

        Format:
        {{
        "completed": true or false,
        "reason": "one  sentence explaining why the instruction is not satisfied"
        }}
    """

    test_case = {
        "idx": 1,
        "answer": "",
        "prompt": prompt,
        "image": img_path,
        "video": "",
        "type": "single_image"
    }

    messages = client.prepare_messages_from_test_case(test_case)

    data = _request_json_response(
        client,
        messages,
        model=MODEL_NAME,
        max_tokens=200,
        temperature=0.1,
    )

    return _normalize_completion_result(data)
