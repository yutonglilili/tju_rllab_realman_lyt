from __future__ import annotations

from typing import Any

from interactive_interface.task_adapters.common import TaskDefinition, TaskExecutionContext
from interactive_interface.task_adapters.pick_and_place import (
    build_definition as build_pick_and_place_definition,
)
from interactive_interface.task_adapters.roast_sweet_potatoes import (
    build_definition as build_roast_definition,
)


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
    }


def collect_params(task_id: str) -> dict[str, Any]:
    task = get_task_definition(task_id)
    return dict(task.default_params)


def execute_task(task_id: str, runtime: Any, instruction: str, params: dict[str, Any]) -> dict[str, Any]:
    task = get_task_definition(task_id)
    context = TaskExecutionContext(
        runtime=runtime,
        task_def=task,
        instruction=instruction,
        params=params,
    )
    return task.execute(context)
