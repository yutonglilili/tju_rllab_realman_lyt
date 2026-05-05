from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


INTERFACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTERFACE_DIR.parent
PROJECT_PARENT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from interactive_interface.runtime import AppRuntime
from interactive_interface.task_interface import (
    collect_params,
    get_task_definitions,
    get_task_ui_payload,
)
from interactive_interface.ui import build_ui


def _patch_gradio_schema_bug() -> None:
    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    original_get_type = getattr(client_utils, "get_type", None)
    if original_get_type is None:
        return

    if getattr(original_get_type, "_demo_new_bool_schema_patch", False):
        return

    def patched_get_type(schema: Any) -> Any:
        if isinstance(schema, bool):
            return "boolean"
        return original_get_type(schema)

    patched_get_type._demo_new_bool_schema_patch = True  # type: ignore[attr-defined]
    client_utils.get_type = patched_get_type


def _load_gradio() -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Third-party gradio is not installed. Run `python -m pip install gradio` first, then rerun `python interactive_interface/main.py`."
        ) from exc

    return gr


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None

    try:
        return int(raw.strip())
    except ValueError:
        return None


APP = AppRuntime()
UI_RENDER_CACHE = {
    "status": None,
    "logs": None,
}


def _diff_text_output(gr: Any, cache_key: str, value: str) -> Any:
    if UI_RENDER_CACHE.get(cache_key) == value:
        return gr.skip()
    UI_RENDER_CACHE[cache_key] = value
    return value


def _refresh_status_outputs(gr: Any) -> tuple[Any, ...]:
    status_markdown, logs_text = APP.snapshot_status()
    return (
        _diff_text_output(gr, "status", status_markdown),
        _diff_text_output(gr, "logs", logs_text),
    )


def _refresh_all_outputs(gr: Any) -> tuple[Any, ...]:
    status_markdown, logs_text, camera_frame = APP.snapshot_full()
    return (
        _diff_text_output(gr, "status", status_markdown),
        _diff_text_output(gr, "logs", logs_text),
        camera_frame,
    )


def _on_task_change(gr: Any, task_id: str) -> tuple[Any, ...]:
    payload = get_task_ui_payload(task_id)
    APP.log(f"切换到任务：{payload['title']}")

    return (
        gr.update(
            label=payload["input_label"],
            value=payload["default_instruction"],
        ),
        *_refresh_status_outputs(gr),
    )


def _on_init_runtime(
    gr: Any,
    robot_ip: str,
    camera_serial: str,
    cam_results_path: str,
) -> tuple[Any, ...]:
    APP.ensure_runtime(robot_ip, camera_serial, cam_results_path)
    return _refresh_all_outputs(gr)


def _on_start(
    gr: Any,
    task_id: str,
    instruction: str,
    robot_ip: str,
    camera_serial: str,
    cam_results_path: str,
) -> tuple[Any, ...]:
    if not APP.ensure_runtime(robot_ip, camera_serial, cam_results_path):
        return _refresh_all_outputs(gr)

    params = collect_params(task_id)
    APP.launch_task(task_id=task_id, instruction=instruction, params=params)
    return _refresh_all_outputs(gr)


def _on_stop(gr: Any) -> tuple[Any, ...]:
    APP.request_stop()
    return _refresh_all_outputs(gr)


def build_demo() -> Any:
    gr = _load_gradio()
    _patch_gradio_schema_bug()

    task_definitions = get_task_definitions()
    demo, components = build_ui(gr, task_definitions)

    with demo:
        components["task_dropdown"].change(
            fn=lambda task_id: _on_task_change(gr, task_id),
            inputs=[components["task_dropdown"]],
            outputs=[
                components["instruction_input"],
                components["current_status_output"],
                components["recent_actions_output"],
            ],
            api_name=False,
            show_api=False,
        )

        components["init_runtime_button"].click(
            fn=lambda robot_ip, camera_serial, cam_results_path: _on_init_runtime(
                gr,
                robot_ip,
                camera_serial,
                cam_results_path,
            ),
            inputs=[
                components["robot_ip_input"],
                components["camera_serial_input"],
                components["cam_results_path_input"],
            ],
            outputs=[
                components["current_status_output"],
                components["recent_actions_output"],
                components["camera_image_output"],
            ],
            api_name=False,
            show_api=False,
        )

        components["start_button"].click(
            fn=lambda task_id, instruction, robot_ip, camera_serial, cam_results_path: _on_start(
                gr,
                task_id,
                instruction,
                robot_ip,
                camera_serial,
                cam_results_path,
            ),
            inputs=[
                components["task_dropdown"],
                components["instruction_input"],
                components["robot_ip_input"],
                components["camera_serial_input"],
                components["cam_results_path_input"],
            ],
            outputs=[
                components["current_status_output"],
                components["recent_actions_output"],
                components["camera_image_output"],
            ],
            api_name=False,
            show_api=False,
        )

        components["stop_button"].click(
            fn=lambda: _on_stop(gr),
            outputs=[
                components["current_status_output"],
                components["recent_actions_output"],
                components["camera_image_output"],
            ],
            api_name=False,
            show_api=False,
        )

        components["instruction_input"].submit(
            fn=lambda task_id, instruction, robot_ip, camera_serial, cam_results_path: _on_start(
                gr,
                task_id,
                instruction,
                robot_ip,
                camera_serial,
                cam_results_path,
            ),
            inputs=[
                components["task_dropdown"],
                components["instruction_input"],
                components["robot_ip_input"],
                components["camera_serial_input"],
                components["cam_results_path_input"],
            ],
            outputs=[
                components["current_status_output"],
                components["recent_actions_output"],
                components["camera_image_output"],
            ],
            api_name=False,
            show_api=False,
        )

        demo.load(
            fn=lambda: _refresh_all_outputs(gr),
            outputs=[
                components["current_status_output"],
                components["recent_actions_output"],
                components["camera_image_output"],
            ],
            api_name=False,
            show_api=False,
        )

        demo.load(
            fn=lambda: _refresh_all_outputs(gr),
            outputs=[
                components["current_status_output"],
                components["recent_actions_output"],
                components["camera_image_output"],
            ],
            every=1.5,
            api_name=False,
            show_api=False,
        )

    return demo


def main() -> None:
    demo = build_demo()
    queued_demo = demo.queue()

    launch_kwargs: dict[str, Any] = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "share": _read_bool_env("GRADIO_SHARE", True),
    }

    server_port = _read_int_env("GRADIO_SERVER_PORT")
    if server_port is not None:
        launch_kwargs["server_port"] = server_port

    queued_demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
