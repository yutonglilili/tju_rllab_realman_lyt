"""
本脚本包含 pick and place 任务对指令拆解出目标物体，并逐个获取2D点的功能函数。
"""
from pointing_vllm_client import VLLMOnlineClient
from pick_and_place_utils import omni_decode_points
from PIL import Image
import cv2
import numpy as np

import json
import re
import ast

def _fix_broken_json(s: str) -> str:
    s = s.strip()

    # 去掉奇怪空白
    s = re.sub(r"\s+", " ", s)

    # ===== 补括号 =====
    if s.count('[') > s.count(']'):
        s += ']' * (s.count('[') - s.count(']'))

    if s.count('{') > s.count('}'):
        s += '}' * (s.count('{') - s.count('}'))

    # ===== 修复常见错误 =====
    # 去掉末尾多余字符
    s = re.sub(r"[^\}\]]+$", "", s)

    return s

def parse_multi_pick_place_tasks(text_prompt):
    """Robust multi-step pick & place task parser (production-ready)"""

    import json
    import re
    import ast

    # ====== VLM call ======
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-0128"

    from pointing_vllm_client import VLLMOnlineClient
    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

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
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

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

def parse_multi_pick_place_tasks_old(text_prompt):
    """Robust multi-step pick & place task parser (production-ready)"""

    import json
    import re
    import ast

    # ====== VLM call ======
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-0128"

    from pointing_vllm_client import VLLMOnlineClient
    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

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
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    print("[VLM multi-task raw output]\n", content)

    # =========================================================
    # 🧠 Step 1: 去 markdown 包裹
    # =========================================================
    content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(r"```$", "", content.strip())

    # =========================================================
    # 🧠 Step 2: 修复 JSON
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
    # 🧠 Step 3: JSON 解析
    # =========================================================
    try:
        data = json.loads(content)
    except Exception:
        try:
            data = ast.literal_eval(content)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to parse:\n{content}") from e

    # =========================================================
    # 🧠 Step 4: 自动 unwrap（核心）
    # =========================================================
    def unwrap(data):
        """不断剥离 list / 嵌套"""
        while isinstance(data, list) and len(data) == 1:
            data = data[0]
        return data

    data = unwrap(data)

    # =========================================================
    # 🧠 Step 5: 标准化结构
    # =========================================================
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
    elif isinstance(data, list):
        tasks = data
        data = {"tasks": tasks}
    else:
        raise ValueError(f"❌ Invalid structure:\n{data}")

    # 再 unwrap tasks（防止 tasks 被包一层）
    tasks = unwrap(tasks)

    if not isinstance(tasks, list):
        raise TypeError("❌ 'tasks' must be list")

    # =========================================================
    # 🧠 Step 6: 清洗任务（极重要）
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
    # 🧠 Step 7: 输出标准格式
    # =========================================================
    result = {
        "num_tasks": len(cleaned_tasks),
        "tasks": cleaned_tasks
    }

    print("✅ Parsed task plan:", result)

    return result


def get_point_vllm(image_rgb, text_prompt="you need to grasp the mug", save_path="debug_pointing_vllm.png", color=(0, 0, 255)):
    # Model and server configuration
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-0128"

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
    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

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

        If the target is a basket:
            The point MUST be inside the basket
            The point MUST NOT be on any object inside the basket
            Prefer a position near the center of the basket
        If the target is a normal object:
            Choose a point near the top-center of the object
            Avoid edges of the object
        Return JSON only.
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


def check_grasp_success_vllm(image_rgb, object_name):

    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-0128"

    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

    # =========================
    # 图像处理（关键）
    # =========================

    img = image_rgb.copy()

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.ascontiguousarray(img)

    tmp_image_path = "check_grasp_image.png"
    Image.fromarray(img).save(tmp_image_path)

    print("Saved grasp check image:", tmp_image_path)

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
        - The object is lifted from the table

        FAILURE conditions:
        - The object is still on the table
        - The gripper is empty
        - The object is not inside the gripper

        Return JSON only.

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
        model=model_name,
        messages=messages,
        max_tokens=200,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    print("[VLM grasp check raw output]")
    print(content)

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

    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-0128"

    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

    # =========================
    # 图像处理
    # =========================

    img = image_rgb.copy()

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    img = np.ascontiguousarray(img)

    tmp_image_path = "check_place_image.png"
    Image.fromarray(img).save(tmp_image_path)

    print("Saved place check image:", tmp_image_path)

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
        - The object is still in the robot gripper.

        Return JSON only.

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
        model=model_name,
        messages=messages,
        max_tokens=200,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    print("[VLM place check raw output]")
    print(content)

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



if __name__ == "__main__":
    print(get_point_vllm(np.array(Image.open("aff.png")), "you need to grasp the mug"))


