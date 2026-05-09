from __future__ import annotations

from typing import Any

from interactive_interface.task_adapters.common import (
    OPEN_AIR_FRYER_DRAWER,
    ROAST_CONFIG_PATH,
    TaskDefinition,
    TaskExecutionContext,
    clean_instruction,
)


def build_roast_plan(instruction: str) -> tuple[list[dict[str, str]], int]:
    from demo_new.vlm_utils.multi_pointing_vllm_get_point_utils import parse_roast_with_timer

    execution_plan, rotate_angle = parse_roast_with_timer(clean_instruction(instruction))
    cleaned_plan: list[dict[str, str]] = []

    for task in execution_plan:
        if not isinstance(task, dict):
            continue

        pick = str(task.get("pick", "")).strip()
        place = str(task.get("place", "")).strip() or OPEN_AIR_FRYER_DRAWER
        if not pick:
            continue

        cleaned_plan.append({"pick": pick, "place": place})

    if not cleaned_plan:
        raise ValueError("没有从指令里解析出有效的空气炸锅放食材计划。")

    return cleaned_plan, int(rotate_angle)


def _ensure_not_stopped(runtime: Any, phase_name: str) -> None:
    if runtime.stop_requested.is_set():
        raise InterruptedError(f"{phase_name} 前收到了停止请求。")


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
        close_action,
        open_action,
        rotate_action,
    )
    from demo_new.skills.pnp_skill.pick_and_place import (
        init_state,
        run_all_tasks,
        shutdown_pnp_system,
        start_pnp_system,
    )

    instruction = clean_instruction(context.instruction)
    runtime = context.runtime
    resources = runtime.require_resources()
    execution_plan, rotate_angle = build_roast_plan(instruction)

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
    open_action(resources.env, tcp_pose_open, np.array([1, 0, 0]))
    runtime.log("空气炸锅已打开，开始放食材。")

    state = init_state(task_config_path=str(ROAST_CONFIG_PATH))
    runtime.attach_task_state(state)
    start_pnp_system(
        state,
        resources.env,
        resources.rs_env,
        resources.cam_results,
        resources.home_T_tcp2base,
    )

    try:
        _ensure_not_stopped(runtime, "放食材")
        run_all_tasks(
            state,
            resources.env,
            resources.rs_env,
            resources.cam_results,
            execution_plan,
            resources.home_T_tcp2base,
        )
    finally:
        shutdown_pnp_system(state)
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
    close_action(resources.env, tcp_pose_close, np.array([1, 0, 0]))

    _ensure_not_stopped(runtime, "旋转定时旋钮")
    runtime.log("正在根据指令设置空气炸锅时间。")
    tcp_pose_rotate = _find_air_fryer_target_pose(
        rs_env=resources.rs_env,
        cam_results=resources.cam_results,
        home_T_tcp2base=resources.home_T_tcp2base,
        prompt="Point at the round knob of the air fryer.",
        lift_offsets={"lift_x": 0.03, "lift_y": -0.02, "lift_z": -0.02},
        fixed_rpy=(0.0, 0.0, 3.1412),
    )
    rotate_action(
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
        default_instruction="我需要烤红薯和玉米，定时 20 分钟。",
        default_params={},
        execute=execute_roast_task,
    )
