from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402
from nav_dual_sweep import nav_dual_sweep  # noqa: E402
from smart_shelf_search import smart_shelf_search_from_request  # noqa: E402
from sweep_request_rules import resolve_nav_dual_sweep_request  # noqa: E402
from visualize_utils import VisualizeContext  # noqa: E402

# Median depth sample count for nav target estimation in nav dual sweep.
DEFAULT_NAV_DEPTH_MEDIAN_N = 5
# Median depth sample count for swap target estimation in nav dual sweep.
DEFAULT_SWAP_DEPTH_MEDIAN_N = 5
# Vote count for goal-presence checking before nav accepts a target.
DEFAULT_NAV_VOTE_TIMES = 3
# Whether nav dual sweep keeps the original local navigation debug flow enabled.
DEFAULT_NAV_LOCAL_DEBUG = True
# Whether nav dual sweep keeps the original local swap debug flow enabled.
DEFAULT_SWAP_LOCAL_DEBUG = True

# Whether smart shelf search allows navigation rotate-recover behavior.
DEFAULT_SHELF_ROTATE_RECOVER = True
# Median depth sample count for smart shelf search nav/pick/place perception.
DEFAULT_SHELF_DEPTH_MEDIAN_N = 10
# Whether smart shelf search keeps the original local navigation debug flow enabled.
DEFAULT_SHELF_NAV_DEBUG = True
# Whether smart shelf search keeps the original local pick/place debug flow enabled.
DEFAULT_SHELF_PICK_DEBUG = True

# Camera polling interval for refreshing the three live camera feeds.
CAMERA_POLL_INTERVAL_S = 0.08
# Sleep interval while waiting for the ARX environment before camera polling starts.
CAMERA_IDLE_SLEEP_S = 0.10
# Gradio UI timer interval for pulling the latest controller state.
UI_REFRESH_INTERVAL_S = 0.12


def _serialize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serialize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _serialize_json(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _dump_json(value: Any) -> str:
    return json.dumps(
        _serialize_json(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _to_display_image(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image[..., ::-1])
    return image


@dataclass
class RuntimeSnapshot:
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    active_task: str = "idle"
    task_state: str = "idle"
    current_stage: str = "Waiting for task"
    parsed_request: Optional[dict[str, Any]] = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    last_error: str = ""


class DemoController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RuntimeSnapshot()
        self._env: Optional[ARXRobotEnv] = None
        self._camera_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._stop_task = threading.Event()
        self._camera_error_logged = False
        self._task_running = False
        self._task_queue: queue.Queue[Optional[Callable[[], None]]] = queue.Queue(
        )
        self._task_worker_thread = threading.Thread(
            target=self._task_worker_loop,
            name="gradio-task-worker",
            daemon=True,
        )
        self._task_worker_thread.start()

    def _task_worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                task = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                self._task_queue.task_done()
                break
            try:
                task()
            finally:
                with self._lock:
                    self._task_running = False
                self._task_queue.task_done()

    def ensure_env(self) -> ARXRobotEnv:
        if self._env is None:
            self._append_log("Create ARX environment")
            self._env = ARXRobotEnv(
                duration_per_step=1.0 / 20.0,
                min_steps=20,
                max_v_xyz=0.15,
                max_a_xyz=0.1,
                max_v_rpy=0.5,
                max_a_rpy=0.7,
                camera_type="all",
                camera_view=("camera_l", "camera_h", "camera_r"),
                img_size=(640, 480),
            )
        self._start_camera_thread()
        return self._env

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="gradio-camera-loop",
            daemon=True,
        )
        self._camera_thread.start()

    def _camera_loop(self) -> None:
        while not self._shutdown.is_set():
            env = self._env
            if env is None:
                time.sleep(CAMERA_IDLE_SLEEP_S)
                continue
            try:
                frames, status = env.get_camera(
                    target_size=(640, 480),
                    return_status=True,
                )
                with self._lock:
                    self._state.frames = dict(frames)
                    self._state.status = dict(status)
                    self._state.last_error = ""
                self._camera_error_logged = False
            except Exception as exc:
                with self._lock:
                    self._state.last_error = str(exc)
                if not self._camera_error_logged:
                    self._append_log(f"Camera poll failed: {exc}")
                    self._camera_error_logged = True
            time.sleep(CAMERA_POLL_INTERVAL_S)

    def close(self) -> None:
        self._shutdown.set()
        self._stop_task.set()
        self._task_queue.put(None)
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

    def _append_log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        with self._lock:
            self._state.logs.append(line)
            self._state.logs = self._state.logs[-200:]

    def _set_task_header(self, task_name: str, task_state: str, stage: str) -> None:
        with self._lock:
            self._state.active_task = task_name
            self._state.task_state = task_state
            self._state.current_stage = stage

    def handle_event(self, event: str, payload: dict[str, Any]) -> None:
        source = str(payload.get("source", "demo"))
        stage = payload.get("stage")
        message = payload.get("message")

        if event == "log":
            prefix = f"{source}"
            if stage:
                prefix += f"/{stage}"
            self._append_log(f"{prefix}: {message}")
            return

        if event == "stage":
            stage_text = str(message or stage or source)
            with self._lock:
                self._state.current_stage = stage_text
                self._state.telemetry["latest_stage"] = {
                    "source": source,
                    "stage": stage,
                    "message": message,
                }
            return

        with self._lock:
            if event == "parsed_request":
                self._state.parsed_request = dict(payload)
                self._state.telemetry["parsed_request"] = dict(payload)
            elif event == "result":
                self._state.result = dict(payload)
                status = str(payload.get("status", "completed"))
                if status in {"success", "completed"}:
                    self._state.task_state = "completed"
                elif status in {"failed", "error"}:
                    self._state.task_state = "failed"
                elif status in {"stopped", "canceled"}:
                    self._state.task_state = status
            else:
                self._state.telemetry[event] = dict(payload)

    def _begin_task(self, task_name: str, debug_enabled: bool) -> Optional[str]:
        with self._lock:
            if self._task_running:
                return "Another task is still running."
            self._task_running = True
        self.ensure_env()
        self._stop_task = threading.Event()
        with self._lock:
            self._state.active_task = task_name
            self._state.task_state = "starting"
            self._state.current_stage = "Preparing task"
            self._state.parsed_request = None
            self._state.telemetry = {"local_debug_enabled": debug_enabled}
            self._state.result = None
            self._state.logs = []
            self._state.last_error = ""
        return None

    def _reset_after_task(self, task_name: str, env: Optional[ARXRobotEnv]) -> None:
        if env is None or self._shutdown.is_set():
            return
        self._append_log(f"Reset robot after {task_name}")
        self._set_task_header(task_name, "resetting", "Reset robot")
        try:
            env.reset()
        except Exception as exc:
            self._append_log(f"Reset after {task_name} failed: {exc}")
            with self._lock:
                self._state.task_state = "failed"
                self._state.current_stage = "Reset failed"
                self._state.last_error = str(exc)
            return
        with self._lock:
            self._state.active_task = "idle"
            self._state.task_state = "idle"
            self._state.current_stage = "Waiting for task"

    def start_nav_dual_sweep(
        self,
        prompt: str,
        distance: float,
        nav_lift_height: float,
    ) -> str:
        if not prompt.strip():
            return "Prompt is required."
        resolution = resolve_nav_dual_sweep_request(prompt.strip())
        error = self._begin_task(
            "nav dual sweep",
            DEFAULT_NAV_LOCAL_DEBUG or DEFAULT_SWAP_LOCAL_DEBUG,
        )
        if error:
            return error

        def _runner() -> None:
            env: Optional[ARXRobotEnv] = None
            try:
                env = self.ensure_env()
                self._append_log("Reset robot before nav dual sweep")
                self._set_task_header(
                    "nav dual sweep", "running", "Reset robot")
                env.reset()
                self.handle_event(
                    "nav_dual_sweep_request",
                    {
                        "source": "nav_dual_sweep",
                        "raw_request": resolution.raw_request,
                        "goal_prompt": resolution.goal_prompt,
                        "intent": resolution.intent,
                        "reason": resolution.reason,
                        "matched_alias": resolution.matched_alias,
                    },
                )
                self._append_log(
                    "nav_dual_sweep/request: "
                    f"{resolution.raw_request} -> {resolution.goal_prompt}"
                )
                visualize = VisualizeContext(
                    on_event=self.handle_event,
                    stop_checker=self._stop_task.is_set,
                    page_debug=False,
                )
                nav_dual_sweep(
                    env,
                    goal=resolution.goal_prompt,
                    distance=float(distance),
                    nav_lift_height=float(nav_lift_height),
                    nav_debug_raw=DEFAULT_NAV_LOCAL_DEBUG,
                    swap_debug_raw=DEFAULT_SWAP_LOCAL_DEBUG,
                    nav_depth_median_n=DEFAULT_NAV_DEPTH_MEDIAN_N,
                    swap_depth_median_n=DEFAULT_SWAP_DEPTH_MEDIAN_N,
                    vote_times=DEFAULT_NAV_VOTE_TIMES,
                    visualize=visualize,
                )
                if self._stop_task.is_set():
                    self.handle_event(
                        "result",
                        {
                            "source": "nav_dual_sweep",
                            "status": "stopped",
                            "message": "nav dual sweep stopped",
                        },
                    )
                else:
                    self.handle_event(
                        "result",
                        {
                            "source": "nav_dual_sweep",
                            "status": "completed",
                            "message": "nav dual sweep finished",
                        },
                    )
            except Exception as exc:
                self._append_log(f"Task failed: {exc}")
                self._append_log(traceback.format_exc().strip())
                with self._lock:
                    self._state.task_state = "failed"
                    self._state.current_stage = "Task failed"
                    self._state.last_error = str(exc)
            finally:
                self._reset_after_task("nav dual sweep", env)
                with self._lock:
                    if self._state.task_state == "running":
                        self._state.task_state = "completed"

        self._task_queue.put(_runner)
        return "Started nav dual sweep."

    def start_smart_shelf_search(
        self,
        request: str,
        first_nav_height: float,
    ) -> str:
        if not request.strip():
            return "Request is required."
        error = self._begin_task(
            "smart shelf search",
            DEFAULT_SHELF_NAV_DEBUG or DEFAULT_SHELF_PICK_DEBUG,
        )
        if error:
            return error

        def _runner() -> None:
            env: Optional[ARXRobotEnv] = None
            try:
                env = self.ensure_env()
                self._append_log("Reset robot before smart shelf search")
                self._set_task_header(
                    "smart shelf search", "running", "Reset robot")
                env.reset()
                visualize = VisualizeContext(
                    on_event=self.handle_event,
                    stop_checker=self._stop_task.is_set,
                    page_debug=False,
                )
                result = smart_shelf_search_from_request(
                    arx=env,
                    request=request.strip(),
                    first_nav_height=float(first_nav_height),
                    rotate_recover=DEFAULT_SHELF_ROTATE_RECOVER,
                    nav_debug=DEFAULT_SHELF_NAV_DEBUG,
                    debug_pick_place=DEFAULT_SHELF_PICK_DEBUG,
                    depth_median_n=DEFAULT_SHELF_DEPTH_MEDIAN_N,
                    visualize=visualize,
                )
                message = str(result.get("message", ""))
                status = "success" if result.get("success") else "failed"
                if self._stop_task.is_set() or message.lower().startswith("stopped"):
                    status = "stopped"
                self.handle_event(
                    "result",
                    {
                        "source": "smart_shelf_search",
                        "status": status,
                        "message": message,
                        "result": result,
                    },
                )
            except Exception as exc:
                self._append_log(f"Task failed: {exc}")
                self._append_log(traceback.format_exc().strip())
                with self._lock:
                    self._state.task_state = "failed"
                    self._state.current_stage = "Task failed"
                    self._state.last_error = str(exc)
            finally:
                self._reset_after_task("smart shelf search", env)

        self._task_queue.put(_runner)
        return "Started smart shelf search."

    def stop_current_task(self) -> str:
        with self._lock:
            is_running = self._task_running
            task_state = self._state.task_state
        if not is_running and task_state not in {"running", "starting", "stopping"}:
            return "No task is running."

        self._stop_task.set()
        self._append_log("Stop requested from page")
        with self._lock:
            if self._state.task_state in {"running", "starting"}:
                self._state.task_state = "stopping"
                self._state.current_stage = "Stopping task"
        return "Stop signal sent."

    def _summarize_status(self) -> str:
        with self._lock:
            state = self._state
            status = dict(state.status)
            active_task = state.active_task
            task_state = state.task_state
            current_stage = state.current_stage
            last_error = state.last_error
        base = status.get("base")
        base_height = None
        if base is not None and hasattr(base, "height"):
            try:
                base_height = float(base.height)
            except Exception:
                base_height = None
        lines = [
            f"**Task**: {active_task}",
            f"**State**: {task_state}",
            f"**Stage**: {current_stage}",
        ]
        if base_height is not None:
            lines.append(f"**Base Height**: {base_height:.2f}")
        if last_error:
            lines.append(f"**Last Error**: {last_error}")
        return "  \n".join(lines)

    def read_ui_state(self):
        try:
            self.ensure_env()
        except Exception as exc:
            self._append_log(f"Environment init failed: {exc}")

        with self._lock:
            state = self._state
            frames = dict(state.frames)
            telemetry = dict(state.telemetry)
            result = state.result
            logs = list(state.logs)

        if result is not None:
            telemetry = dict(telemetry)
            telemetry["result"] = result

        return (
            self._summarize_status(),
            _to_display_image(frames.get("camera_l_color")),
            _to_display_image(frames.get("camera_h_color")),
            _to_display_image(frames.get("camera_r_color")),
            "\n".join(logs[-80:]),
            _dump_json(telemetry or {}),
        )


def _require_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "gradio is not installed. Run `pip install gradio` first."
        ) from exc
    return gr


def build_app(controller: DemoController):
    gr = _require_gradio()

    with gr.Blocks(title="ARX Demo Console") as demo:
        gr.Markdown(
            "# ARX Demo Console\n"
            "Three camera feeds stay live. Task debug stays in the original "
            "terminal and local OpenCV windows."
        )

        action_feedback = gr.Textbox(
            label="Action Feedback",
            interactive=False,
            lines=1,
        )
        status_md = gr.Markdown()

        with gr.Row():
            cam_l = gr.Image(label="camera_l", interactive=False)
            cam_h = gr.Image(label="camera_h", interactive=False)
            cam_r = gr.Image(label="camera_r", interactive=False)

        with gr.Tabs():
            with gr.Tab("Nav Dual Sweep"):
                nav_prompt = gr.Textbox(
                    label="Prompt / Request",
                    value="paper cup or paper ball or bottle on the floor",
                )
                with gr.Accordion("Advanced", open=False):
                    nav_distance = gr.Number(label="Stop Distance", value=0.55)
                    nav_lift_height = gr.Number(
                        label="Nav Lift Height", value=0.0)
                start_nav_btn = gr.Button(
                    "Start Nav Dual Sweep", variant="primary")

            with gr.Tab("Smart Shelf Search"):
                shelf_request = gr.Textbox(
                    label="Request",
                    value="我要一个蓝色盒子",
                )
                with gr.Accordion("Advanced", open=False):
                    first_nav_height = gr.Number(
                        label="First Nav Height",
                        value=14.5,
                    )
                start_shelf_btn = gr.Button(
                    "Start Smart Shelf Search",
                    variant="primary",
                )

        with gr.Row():
            with gr.Accordion("Runtime Log", open=False):
                log_box = gr.Textbox(
                    label="Runtime Log",
                    lines=12,
                    interactive=False,
                )
            with gr.Accordion("Runtime Telemetry", open=False):
                telemetry_box = gr.Textbox(
                    label="Runtime Telemetry",
                    lines=12,
                    interactive=False,
                )

        stop_btn = gr.Button("Stop Current Task", variant="stop")

        start_nav_btn.click(
            fn=controller.start_nav_dual_sweep,
            inputs=[
                nav_prompt,
                nav_distance,
                nav_lift_height,
            ],
            outputs=[action_feedback],
            queue=False,
        )
        start_shelf_btn.click(
            fn=controller.start_smart_shelf_search,
            inputs=[
                shelf_request,
                first_nav_height,
            ],
            outputs=[action_feedback],
            queue=False,
        )
        stop_btn.click(
            fn=controller.stop_current_task,
            outputs=[action_feedback],
            queue=False,
        )

        timer = gr.Timer(UI_REFRESH_INTERVAL_S)
        timer.tick(
            fn=controller.read_ui_state,
            outputs=[
                status_md,
                cam_l,
                cam_h,
                cam_r,
                log_box,
                telemetry_box,
            ],
            queue=False,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX demo Gradio console")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    controller = DemoController()
    demo = build_app(controller)
    try:
        demo.queue().launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
        )
    finally:
        controller.close()


if __name__ == "__main__":
    main()
