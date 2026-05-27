import argparse
import json
from pathlib import Path

from multi_pointing_vllm_get_point_utils_qwen import MODEL_NAME, get_vlm_client


def build_grasp_prompt(object_name):
    return f"""
        You are a robot perception system.

        The robot attempted to grasp an object.

        Target object: {object_name}

        Look carefully at the image and determine whether the robot gripper is currently holding the object.

        SUCCESS conditions:
        - The object is clearly inside the robot gripper

        FAILURE conditions:
        - The gripper is empty
        - The object is not inside the gripper

        Return JSON only.

        Example:
        {{
        "grasp_success": true
        }}
    """


def build_place_prompt(object_name, container_name):
    return f"""
        You are a robot perception system.

        The robot attempted to place an object into a container.

        Object: {object_name}
        Target container: {container_name}

        Look at the image and determine whether the object is already inside the container.

        SUCCESS conditions:
        - The object is clearly inside the container.

        FAILURE conditions:
        - The object is outside the container.

        Return JSON only.

        Example:
        {{
        "place_success": true
        }}
    """


def flatten_content(payload):
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts = [flatten_content(item).strip() for item in payload]
        return "\n".join(part for part in parts if part)
    if isinstance(payload, dict):
        for key in ("text", "content", "value"):
            if key in payload:
                return flatten_content(payload[key])
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def main():
    parser = argparse.ArgumentParser(
        description="Debug raw model output for grasp/place success checks.",
    )
    parser.add_argument(
        "--mode",
        choices=("grasp", "place"),
        required=True,
        help="Which check prompt to use.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        required=True,
        help="Target object name.",
    )
    parser.add_argument(
        "--container",
        dest="container_name",
        default=None,
        help="Target container name for place mode.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens for the model response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the raw output as a JSON file.",
    )
    args = parser.parse_args()

    if args.mode == "place" and not args.container_name:
        parser.error("--container is required when --mode place")

    image_path = str(Path(args.image).expanduser().resolve())
    client = get_vlm_client()

    if args.mode == "grasp":
        prompt = build_grasp_prompt(args.object_name)
    else:
        prompt = build_place_prompt(args.object_name, args.container_name)

    test_case = {
        "idx": 0,
        "answer": "",
        "prompt": prompt,
        "image": image_path,
        "video": "",
        "type": "single_image",
    }

    messages = client.prepare_messages_from_test_case(test_case)
    response = client.client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    message = response.choices[0].message
    raw_content = getattr(message, "content", None)
    raw_text = getattr(message, "text", None)
    raw_reasoning = getattr(message, "reasoning_content", None)
    flattened = flatten_content(raw_content or raw_text or raw_reasoning)

    print("\n========== REQUEST INFO ==========\n")
    print("mode:", args.mode)
    print("image:", image_path)
    print("object:", args.object_name)
    print("container:", args.container_name)
    print("model:", MODEL_NAME)

    print("\n========== PROMPT ==========\n")
    print(prompt)

    print("\n========== RAW message.content ==========\n")
    print(raw_content)

    print("\n========== RAW message.text ==========\n")
    print(raw_text)

    print("\n========== RAW message.reasoning_content ==========\n")
    print(raw_reasoning)

    print("\n========== FLATTENED TEXT ==========\n")
    print(flattened)

    if args.save:
        save_path = str(Path(args.save).expanduser().resolve())
        payload = {
            "mode": args.mode,
            "image": image_path,
            "object_name": args.object_name,
            "container_name": args.container_name,
            "model": MODEL_NAME,
            "prompt": prompt,
            "raw_content": raw_content,
            "raw_text": raw_text,
            "raw_reasoning_content": raw_reasoning,
            "flattened_text": flattened,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("\nSaved raw output to:", save_path)


if __name__ == "__main__":
    main()
