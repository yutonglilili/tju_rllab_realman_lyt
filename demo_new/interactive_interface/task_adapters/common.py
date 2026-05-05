from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = PROJECT_ROOT / "task"

PICK_AND_PLACE_CONFIG_PATH = TASK_ROOT / "pick_and_place" / "config.yaml"
ROAST_CONFIG_PATH = TASK_ROOT / "roast_sweet_potatoes" / "config.yaml"
OPEN_AIR_FRYER_DRAWER = "open air fryer drawer"


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    input_label: str
    default_instruction: str
    default_params: dict[str, Any]
    execute: Callable[["TaskExecutionContext"], dict[str, Any]]


@dataclass
class TaskExecutionContext:
    runtime: Any
    task_def: TaskDefinition
    instruction: str
    params: dict[str, Any]


def clean_instruction(instruction: str) -> str:
    cleaned = (instruction or "").strip()
    if not cleaned:
        raise ValueError("请输入任务指令。")
    return cleaned
