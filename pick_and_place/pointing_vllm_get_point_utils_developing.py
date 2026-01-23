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

def parse_pick_place_objects(text_prompt):
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-v1"

    client = VLLMOnlineClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name
    )

    prompt = f"""
        You are a robot perception module.
        Given a human instruction, extract:
        - the object to pick
        - the object/place to put onto

        You MUST output ONLY a valid JSON object.
        Do NOT output explanations or extra text.

        Format:
        {{
        "pick": "<object_name>",
        "place": "<object_name>"
        }}

        Example:
        Instruction: "Pick up the white can and put it on the plate."
        Output:
        {{
        "pick": "white can",
        "place": "plate"
        }}

        Instruction: "{text_prompt}"
        Output:
    """

    response = client.client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    print("[VLM raw parse output]\n", content)

    # ---------- Strategy 1: JSON / python dict ----------
    try:
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            return {
                "pick": data.get("pick"),
                "place": data.get("place"),
            }
    except Exception:
        pass

    # ---------- Strategy 2: regex fallback ----------
    pick_match = re.search(r'pick\s*[:=]\s*"?([\w\s]+)"?', content, re.I)
    place_match = re.search(r'place\s*[:=]\s*"?([\w\s]+)"?', content, re.I)

    if pick_match or place_match:
        return {
            "pick": pick_match.group(1).strip() if pick_match else None,
            "place": place_match.group(1).strip() if place_match else None,
        }

    raise RuntimeError(f"Failed to parse pick/place from:\n{content}")

import json
import re
import ast

def parse_multi_pick_place_tasks(text_prompt):
    """
    Robust multi-step pick & place task parser.

    Expected final format (dict):
    {
        "num_tasks": int,
        "tasks": [
            {"pick": str, "place": str},
            ...
        ]
    }
    """
    # ====== VLM call ======
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-v1"

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
        DO NOT include markdown fences or explanations.

        Required format:
        {{
        "num_tasks": N,
        "tasks": [
            {{"pick": "...", "place": "..."}},
            ...
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

    # ---------- Step 1: 去掉 ```json ``` 包裹 ----------
    content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(r"```$", "", content.strip())

    # ---------- Step 2: 尝试 json.loads ----------
    try:
        data = json.loads(content)
    except Exception:
        # ---------- Step 3: fallback ast.literal_eval ----------
        try:
            data = ast.literal_eval(content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse multi-task pick/place:\n{content}") from e

    # ---------- Step 4: 处理多任务 ----------
    if isinstance(data, list):
        # 如果是 [{pick:..., place:...}, ...] 直接封装成 dict
        data = {"tasks": data}

    if "tasks" not in data:
        raise KeyError("Missing 'tasks' field in parsed output")

    tasks = data["tasks"]

    if not isinstance(tasks, list):
        raise TypeError("'tasks' must be a list")

    # 自动补 num_tasks
    data["num_tasks"] = len(tasks)

    # 校验每个 task
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            raise TypeError(f"Task {i} is not dict")
        if "pick" not in t or "place" not in t:
            raise KeyError(f"Task {i} missing pick/place")

    return data


def get_point_vllm(image_rgb, text_prompt="you need to grasp the mug", save_path="debug_pointing_vllm.png", color=(0, 0, 255)):
    # Model and server configuration
    base_url = "http://172.28.102.11:22002/v1"
    api_key = "EMPTY"
    model_name = "Embodied-R1.5-SFT-v1"

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
        'prompt': 'Provide one or more points coordinate of objects region this sentence describes: ' + text_prompt + '. When you give the grasp point on the items, you should give the position bewteen the top and the center of the item but more closer to the top. The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].',
        'image': tmp_image_path,
        'video': '',
        'type': 'single_image' 
    }

    print("-"*20 + "prompt")
    print(test_case["prompt"])
    print("-"*20 + "image_path")
    print(test_case["image"]) # image path

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

    print("-"*20 + "generated")
    print(generated_text)

    import numpy as np

    pointing = (np.array(omni_decode_points(generated_text)) / 1000 * np.array([width, height]))[0]

    if save_path:
        img = cv2.imread(test_case["image"])
        img = cv2.circle(img, (int(pointing[0]), int(pointing[1])), 5, color, -1)
        cv2.imwrite(save_path, img)

    return pointing # x, y following opencv camera coord

if __name__ == "__main__":
    print(get_point_vllm(np.array(Image.open("aff.png")), "you need to grasp the mug"))