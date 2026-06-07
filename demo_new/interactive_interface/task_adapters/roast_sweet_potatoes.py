from __future__ import annotations

from typing import Any

from interactive_interface.task_adapters.common import (
    ROAST_CONFIG_PATH,
    TaskDefinition,
    TaskExecutionContext,
    clean_instruction,
)


def _ensure_not_stopped(runtime: Any, phase_name: str) -> None:
    if runtime.stop_requested.is_set():
        raise InterruptedError(f"{phase_name} was interrupted by a stop request.")


def _log_subtask(runtime: Any, task: dict[str, Any]) -> None:
    skill = task.get("skill")
    args = task.get("args", {})

    if skill == 0:
        runtime.log("Opening the air fryer drawer.")
        return

    if skill == 1:
        runtime.log("Closing the air fryer drawer.")
        return

    if skill == 2:
        pick = args.get("pick", "-")
        place = args.get("place", "-")
        runtime.log(f"Running pick and place: {pick} -> {place}.")
        return

    if skill == 3:
        minutes = args.get("minutes", "-")
        runtime.log(f"Setting the air fryer timer to {minutes} minutes.")
        return

    runtime.log(f"Executing subtask: {task}")


def execute_roast_task(context: TaskExecutionContext) -> dict[str, Any]:
    from demo_new.skills.pnp_skill.pick_and_place import (
        init_state,
        start_pnp_system,
        shutdown_pnp_system,
    )
    from demo_new.task.roast_sweet_potatoes.run import execute_subtasks
    from demo_new.vlm_utils.vllm_from_api_key import generate_air_fryer_subtasks

    instruction = clean_instruction(context.instruction)
    runtime = context.runtime
    resources = runtime.require_resources()

    runtime.set_current_task(task_title=context.task_def.title, instruction=instruction)
    runtime.log("Starting the air fryer task.")

    state = init_state(task_config_path=str(ROAST_CONFIG_PATH))
    runtime.attach_task_state(state)

    wrist_runtime = getattr(resources, "wrist_runtime", None)
    enable_graspgen = getattr(state.config, "ENABLE_GRASPGEN", True)
    if wrist_runtime is not None and enable_graspgen:
        start_pnp_system(
            state,
            resources.env,
            resources.rs_env,
            resources.cam_results,
            resources.home_T_tcp2base,
            wrist_rs_env=wrist_runtime.wrist_rs_env,
            graspgen_client=wrist_runtime.graspgen_client,
            wrist_handeye_config=wrist_runtime.wrist_handeye_config,
        )
    else:
        runtime.log("GraspGen not enabled; using heuristic grasping.")
        start_pnp_system(
            state,
            resources.env,
            resources.rs_env,
            resources.cam_results,
            resources.home_T_tcp2base,
        )

    try:
        _ensure_not_stopped(runtime, "Air fryer planning")
        obs = resources.rs_env.step()
        image_rgb = obs["rgb"]
        subtasks = generate_air_fryer_subtasks(
            image_rgb=image_rgb,
            instruction=instruction,
        )

        if not isinstance(subtasks, list):
            raise TypeError("generate_air_fryer_subtasks() must return a list.")

        runtime.log(f"Generated {len(subtasks)} subtasks.")

        for task in subtasks:
            _ensure_not_stopped(runtime, "Air fryer execution")
            if isinstance(task, dict):
                _log_subtask(runtime, task)

        execute_subtasks(
            subtasks=subtasks,
            state=state,
            env=resources.env,
            rs_env=resources.rs_env,
            cam_results=resources.cam_results,
            home_T_tcp2base=resources.home_T_tcp2base,
            config=state.config,
        )

        if runtime.stop_requested.is_set() or state.stop_all.is_set():
            runtime.log("The air fryer task was stopped.")
            return {
                "status": "stopped",
                "task_id": context.task_def.task_id,
                "instruction": instruction,
                "subtasks": subtasks,
            }

        runtime.log("The air fryer task finished.")
        return {
            "status": "completed",
            "task_id": context.task_def.task_id,
            "instruction": instruction,
            "subtasks": subtasks,
        }
    finally:
        shutdown_pnp_system(state)
        runtime.detach_task_state(state)


def build_definition() -> TaskDefinition:
    return TaskDefinition(
        task_id="roast_sweet_potatoes",
        title="Air Fryer Roast",
        input_label="输入指令",
        default_instruction="帮我把烤苹果和香蕉，定时20分钟。",
        candidate_instructions=(
            "帮我把烤苹果和香蕉，定时20分钟。",
            "帮我把烤苹果，定时15分钟。",
            "帮我把空气炸锅里的苹果和香蕉取出来放到盘子上。",
        ),
        default_params={},
        execute=execute_roast_task,
    )
