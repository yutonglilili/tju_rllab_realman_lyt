from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


GRADIO_DIR = Path(__file__).resolve().parent
if str(GRADIO_DIR) not in sys.path:
    sys.path.insert(0, str(GRADIO_DIR))

from task_adapters.common import TaskDefinition, TaskExecutionContext
from task_adapters.pick_and_place import build_definition as build_pick_and_place_definition
from task_adapters.roast_sweet_potatoes import build_definition as build_roast_definition


TASKS: dict[str, TaskDefinition] = {
    "pick_and_place": build_pick_and_place_definition(),
    "roast_sweet_potatoes": build_roast_definition(),
}


def get_task_definitions() -> dict[str, TaskDefinition]:
    return TASKS


def get_task_definition(task_id: str) -> TaskDefinition:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"未知任务类型: {task_id}") from exc


def get_task_ui_payload(task_id: str) -> dict[str, Any]:
    task = get_task_definition(task_id)
    return {
        "task_id": task.task_id,
        "title": task.title,
        "input_label": task.input_label,
        "default_instruction": task.default_instruction,
        "show_mode": task.show_mode,
        "mode_choices": list(task.mode_choices),
        "default_mode": task.default_params.get("mode"),
        "show_rotate_angle": task.show_rotate_angle,
        "default_rotate_angle": task.default_params.get("rotate_angle", 90),
    }


def collect_params(task_id: str, mode: str | None, rotate_angle: float | int | None) -> dict[str, Any]:
    task = get_task_definition(task_id)
    params = dict(task.default_params)

    if task.show_mode and mode:
        params["mode"] = mode

    if task.show_rotate_angle and rotate_angle is not None:
        params["rotate_angle"] = int(rotate_angle)

    return params


def execute_task(task_id: str, runtime: Any, instruction: str, params: dict[str, Any]) -> dict[str, Any]:
    task = get_task_definition(task_id)
    context = TaskExecutionContext(
        runtime=runtime,
        task_def=task,
        instruction=instruction,
        params=params,
    )
    return task.execute(context)
