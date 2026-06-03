from __future__ import annotations

from interactive_interface.task_adapters.common import (
    PICK_AND_PLACE_CONFIG_PATH,
    TaskDefinition,
    TaskExecutionContext,
    clean_instruction,
)


def execute_pick_and_place(context: TaskExecutionContext) -> dict[str, str]:
    from demo_new.skills.pnp_skill.pick_and_place import (
        init_state,
        run_all_tasks_by_instruction_with_position_description,
        shutdown_pnp_system,
        start_pnp_system,
    )

    instruction = clean_instruction(context.instruction)
    runtime = context.runtime
    resources = runtime.require_resources()

    runtime.set_current_task(task_title=context.task_def.title, instruction=instruction)
    runtime.log("开始执行桌面抓取任务。")

    state = init_state(task_config_path=str(PICK_AND_PLACE_CONFIG_PATH))
    runtime.attach_task_state(state)
    runtime.log("已启动抓取执行链路。")

    wrist_runtime = getattr(resources, "wrist_runtime", None)
    if wrist_runtime is not None:
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
        runtime.log("未启用 GraspGen，使用启发式抓取。")
        start_pnp_system(
            state,
            resources.env,
            resources.rs_env,
            resources.cam_results,
            resources.home_T_tcp2base,
        )

    try:
        run_all_tasks_by_instruction_with_position_description(
            state,
            resources.env,
            resources.rs_env,
            resources.cam_results,
            instruction,
            resources.home_T_tcp2base,
        )

        if runtime.stop_requested.is_set() or state.stop_all.is_set():
            runtime.log("桌面抓取任务已停止。")
            return {
                "status": "stopped",
                "task_id": context.task_def.task_id,
                "instruction": instruction,
            }

        runtime.log("桌面抓取任务执行完成。")
        return {
            "status": "completed",
            "task_id": context.task_def.task_id,
            "instruction": instruction,
        }
    finally:
        shutdown_pnp_system(state)
        runtime.detach_task_state(state)


def build_definition() -> TaskDefinition:
    return TaskDefinition(
        task_id="pick_and_place",
        title="Pick and Place",
        input_label="输入指令",
        default_instruction="把香蕉放到盘子里。",
        candidate_instructions=(
            "把苹果和橘子放到盘子里。",
            "把所有水果放到盘子里。",
            "把魔方放到泰迪熊和玩具马的中间。",
        ),
        default_params={},
        execute=execute_pick_and_place,
    )
