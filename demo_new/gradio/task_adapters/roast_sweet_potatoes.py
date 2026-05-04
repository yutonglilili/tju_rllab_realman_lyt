from __future__ import annotations

import re
from typing import Any

from task_adapters.common import (
    DISPLAY_AIR_FRYER_DRAWER,
    OPEN_AIR_FRYER_DRAWER,
    ROAST_CONFIG_PATH,
    TaskDefinition,
    TaskExecutionContext,
    clean_instruction,
)


FOOD_ALIASES: tuple[tuple[str, str, str], ...] = (
    ("红薯", "红薯", "sweet potato"),
    ("sweet potato", "红薯", "sweet potato"),
    ("sweet potatoes", "红薯", "sweet potato"),
    ("玉米", "玉米", "corn"),
    ("corn", "玉米", "corn"),
    ("corns", "玉米", "corn"),
    ("香蕉", "香蕉", "banana"),
    ("banana", "香蕉", "banana"),
    ("bananas", "香蕉", "banana"),
    ("橙子", "橙子", "orange"),
    ("orange", "橙子", "orange"),
    ("oranges", "橙子", "orange"),
    ("土豆", "土豆", "potato"),
    ("potato", "土豆", "potato"),
    ("potatoes", "土豆", "potato"),
    ("鸡翅", "鸡翅", "chicken wing"),
    ("chicken wing", "鸡翅", "chicken wing"),
    ("chicken wings", "鸡翅", "chicken wing"),
)


def _load_pnp_runtime_symbols() -> dict[str, Any]:
    from demo_new.skills.pnp_skill.pick_and_place import (
        init_state,
        run_all_tasks,
        shutdown_pnp_system,
        start_pnp_system,
    )

    return {
        "init_state": init_state,
        "run_all_tasks": run_all_tasks,
        "shutdown_pnp_system": shutdown_pnp_system,
        "start_pnp_system": start_pnp_system,
    }


def _normalize_text_token(token: str) -> str:
    token = token.strip().strip("。．.，,！!？?；;:：")
    token = re.sub(r"^(请|帮我|麻烦|我想|我需要|需要|想要|帮忙)\s*", "", token)
    token = re.sub(r"^(please\s+)?(help me\s+)?", "", token, flags=re.IGNORECASE)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _extract_food_tokens(instruction: str) -> list[tuple[str, str]]:
    original = clean_instruction(instruction)
    lowered = original.lower()
    found: list[tuple[str, str]] = []
    seen_exec: set[str] = set()
    matched_spans: list[tuple[int, int]] = []

    sorted_aliases = sorted(FOOD_ALIASES, key=lambda item: len(item[0]), reverse=True)

    for alias, display_name, exec_name in sorted_aliases:
        if exec_name in seen_exec:
            continue

        matched = False
        matched_span: tuple[int, int] | None = None

        if alias.isascii():
            pattern = r"(?<![a-z])" + re.escape(alias.lower()) + r"(?![a-z])"
            match = re.search(pattern, lowered)
            if match is not None:
                candidate_span = (match.start(), match.end())
                overlaps = any(
                    not (candidate_span[1] <= span[0] or candidate_span[0] >= span[1])
                    for span in matched_spans
                )
                if not overlaps:
                    matched = True
                    matched_span = candidate_span
        else:
            matched = alias in original

        if matched:
            found.append((display_name, exec_name))
            seen_exec.add(exec_name)
            if matched_span is not None:
                matched_spans.append(matched_span)

    if found:
        return found

    candidate_match = re.search(
        r"(?:烤|烘烤|空气炸锅|air fry|roast|cook)(?P<items>.+)",
        original,
        flags=re.IGNORECASE,
    )
    candidate_text = candidate_match.group("items") if candidate_match else original
    raw_items = re.split(r"(?:、|,|，|和|及|与|还有|/| and )", candidate_text, flags=re.IGNORECASE)

    extracted: list[tuple[str, str]] = []
    for item in raw_items:
        normalized = _normalize_text_token(item)
        normalized = re.sub(r"^(一下|一些|几个|一个|一份|the|some)\s*", "", normalized, flags=re.IGNORECASE)
        if not normalized:
            continue
        extracted.append((normalized, normalized))

    return extracted


def build_roast_plan(instruction: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    food_items = _extract_food_tokens(instruction)
    if not food_items:
        raise ValueError("没有从指令里识别出要放入空气炸锅的食材。")

    human_plan = [
        {"pick": display_name, "place": DISPLAY_AIR_FRYER_DRAWER}
        for display_name, _ in food_items
    ]
    execution_plan = [
        {"pick": exec_name, "place": OPEN_AIR_FRYER_DRAWER}
        for _, exec_name in food_items
    ]

    return human_plan, execution_plan


def _ensure_not_stopped(runtime: Any, phase_name: str) -> None:
    if runtime.stop_requested.is_set():
        raise InterruptedError(f"{phase_name} 前收到停止请求。")


def _find_air_fryer_target_pose(
    *,
    rs_env: Any,
    cam_results: Any,
    home_T_tcp2base: Any,
    prompt: str,
    lift_offsets: dict[str, float],
    fixed_rpy: tuple[float, float, float],
) -> Any:
    import numpy as np
    from demo_new.skills.tools.utils import make_lift_T, make_target_T
    from demo_new.vlm_utils.multi_pointing_vllm_get_point_utils import get_point_vllm
    from realman.realman_env import realman_xyzrpy_from_T

    obs = rs_env.step()
    image_rgb = obs["rgb"]
    point_2d = get_point_vllm(image_rgb, prompt, save_path=None)
    target_T = make_target_T(
        obs,
        int(point_2d[0]),
        int(point_2d[1]),
        rs_env,
        cam_results,
        home_T_tcp2base,
    )
    target_T = make_lift_T(target_T, **lift_offsets)
    tcp_pose = realman_xyzrpy_from_T(target_T)
    tcp_pose[3:] = np.array(fixed_rpy)
    return tcp_pose


def execute_roast_task(context: TaskExecutionContext) -> dict[str, Any]:
    import numpy as np

    from demo_new.skills.air_fryer_skill.air_fryer import (
        close_air_fryer,
        open_air_fryer,
        rotate_air_fryer_timer_button,
    )

    instruction = clean_instruction(context.instruction)
    runtime = context.runtime
    resources = runtime.require_resources()
    _, execution_plan = build_roast_plan(instruction)
    rotate_angle = int(context.params.get("rotate_angle", 90))
    symbols = _load_pnp_runtime_symbols()

    runtime.set_current_task(task_title=context.task_def.title, instruction=instruction)
    runtime.log("开始执行空气炸锅任务。")

    _ensure_not_stopped(runtime, "打开空气炸锅")
    runtime.log("正在打开空气炸锅。")
    tcp_pose_open = _find_air_fryer_target_pose(
        rs_env=resources.rs_env,
        cam_results=resources.cam_results,
        home_T_tcp2base=resources.home_T_tcp2base,
        prompt="Point at the handle of the air fryer.",
        lift_offsets={"lift_x": 0.02, "lift_y": -0.01},
        fixed_rpy=(0.0623, 0.4881, 3.1218),
    )
    open_air_fryer(resources.env, tcp_pose_open, np.array([1, 0, 0]))
    runtime.log("空气炸锅已打开，开始放食材。")

    state = symbols["init_state"](task_config_path=str(ROAST_CONFIG_PATH))
    runtime.attach_task_state(state)
    symbols["start_pnp_system"](
        state,
        resources.env,
        resources.rs_env,
        resources.cam_results,
        resources.home_T_tcp2base,
    )

    try:
        _ensure_not_stopped(runtime, "放食材")
        symbols["run_all_tasks"](
            state,
            resources.env,
            resources.rs_env,
            resources.cam_results,
            execution_plan,
            resources.home_T_tcp2base,
        )
    finally:
        symbols["shutdown_pnp_system"](state)
        runtime.detach_task_state(state)

    _ensure_not_stopped(runtime, "关闭空气炸锅")
    runtime.log("食材已放入，正在关闭空气炸锅。")
    tcp_pose_close = _find_air_fryer_target_pose(
        rs_env=resources.rs_env,
        cam_results=resources.cam_results,
        home_T_tcp2base=resources.home_T_tcp2base,
        prompt="Point at the handle of the air fryer.",
        lift_offsets={"lift_x": 0.02, "lift_y": -0.01},
        fixed_rpy=(0.0623, 0.4881, 3.1218),
    )
    close_air_fryer(resources.env, tcp_pose_close, np.array([1, 0, 0]))

    _ensure_not_stopped(runtime, "旋转定时旋钮")
    runtime.log("正在设置空气炸锅时间。")
    tcp_pose_rotate = _find_air_fryer_target_pose(
        rs_env=resources.rs_env,
        cam_results=resources.cam_results,
        home_T_tcp2base=resources.home_T_tcp2base,
        prompt="Point at the round knob of the air fryer.",
        lift_offsets={"lift_x": 0.03, "lift_y": -0.02, "lift_z": -0.02},
        fixed_rpy=(0.0, 0.0, 3.1412),
    )
    rotate_air_fryer_timer_button(
        resources.env,
        tcp_pose_rotate,
        np.array([1, 0, 0]),
        rotate_angle=rotate_angle,
    )

    runtime.log("空气炸锅任务执行完成。")
    return {
        "status": "completed",
        "task_id": context.task_def.task_id,
        "instruction": instruction,
        "rotate_angle": rotate_angle,
        "execution_tasks": execution_plan,
    }


def build_definition() -> TaskDefinition:
    return TaskDefinition(
        task_id="roast_sweet_potatoes",
        title="Air Fryer Roast",
        input_label="输入空气炸锅任务指令",
        default_instruction="我需要烤红薯和玉米",
        default_params={"rotate_angle": 90},
        execute=execute_roast_task,
        show_mode=False,
        show_rotate_angle=True,
    )
