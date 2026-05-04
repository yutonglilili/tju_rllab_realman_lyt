from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


GRADIO_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = GRADIO_DIR.parent
PROJECT_PARENT = PROJECT_ROOT.parent
TASK_ROOT = PROJECT_ROOT / "task"

if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


PICK_AND_PLACE_CONFIG_PATH = TASK_ROOT / "pick_and_place" / "config.yaml"
ROAST_CONFIG_PATH = TASK_ROOT / "roast_sweet_potatoes" / "config.yaml"
OPEN_AIR_FRYER_DRAWER = "open air fryer drawer"
DISPLAY_AIR_FRYER_DRAWER = "空气炸锅抽屉"


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    input_label: str
    default_instruction: str
    default_params: dict[str, Any]
    execute: Callable[["TaskExecutionContext"], dict[str, Any]]
    mode_choices: tuple[str, ...] = ()
    show_mode: bool = False
    show_rotate_angle: bool = False


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
