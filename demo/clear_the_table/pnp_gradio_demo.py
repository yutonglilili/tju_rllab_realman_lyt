"""
使用 gradio 构建的 PnP 任务演示界面
调用 clear_the_table 中的任务执行函数，实现 PnP 任务演示
"""

from __future__ import annotations

import threading
import time
import json
import sys
import os
import numpy as np
import gradio as gr

def _patch_gradio_schema_bool_bug():
    """
    兼容 gradio/gradio_client 在解析 JSON Schema 时遇到 bool schema 的已知问题。
    某些版本会在 `if "const" in schema` 处直接对 bool 做成员判断导致 TypeError。
    """
    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    original_get_type = getattr(client_utils, "get_type", None)
    if original_get_type is None:
        return

    def safe_get_type(schema):
        # JSON Schema 允许布尔 schema（True/False），这里兜底转成字符串类型描述，避免崩溃
        if isinstance(schema, bool):
            return "boolean"
        return original_get_type(schema)

    client_utils.get_type = safe_get_type

# ================================
# 导入你的系统
# ================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.realman_env import RealmanEnv, T_from_realman_xyzrpy
from realman.open3d_realsense_env import Open3dRealsenseEnv

# 你的主系统
from clear_the_table import (
    SharedState,
    perception_thread,
    planning_thread,
    execution_thread,
    run_all_tasks_by_instruction,
    run_all_tasks_by_instruction_with_list,
)


class FakeRobotEnv:
    def reset(self):
        print("[FakeRobotEnv] reset called")
    def get_state(self):
        # 返回假的 TCP pose
        return type("State", (), {"pose": [0,0,0,0,0,0]})()
    def move_to(self, *args, **kwargs):
        print("[FakeRobotEnv] move_to called", args, kwargs)
    def open_gripper(self):
        print("[FakeRobotEnv] open_gripper called")
    def close_gripper(self):
        print("[FakeRobotEnv] close_gripper called")


# ================================
# Controller
# ================================


class PnPController:

    def __init__(self):

        self.lock = threading.Lock()

        # env
        self.env = None
        self.rs_env = None
        self.cam_results = None
        self.home_T_tcp2base = None

        # state
        self.state = None

        # camera
        self.latest_frame = None

        # task
        self.current_instruction = ""
        self.running = False
        self.stop_flag = False
        self.shutdown_event = threading.Event()

        # threads
        self.camera_thread = None
        self.worker_thread = None
        self.worker_threads = []

        self._init_env()

    # ================================
    # 初始化机器人环境
    # ================================

    def _init_env(self):

        print("[Controller] 初始化环境")

        
        self.env = RealmanEnv(
            robot_ip="192.168.101.19",
            mode="sync"
        )
        
        # self.env = FakeRobotEnv()

        self.rs_env = Open3dRealsenseEnv("f1471338")

        cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"

        with open(cam_results_path, "r") as f:
            self.cam_results = json.load(f)

        self.env.reset()

        robot_state = self.env.get_state()
        self.home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)

        # shared state
        self.state = SharedState()

        # 启动三线程
        self._start_worker_threads()

        # 启动相机线程
        self._start_camera_thread()

    # ================================
    # 启动三线程系统
    # ================================

    def _start_worker_threads(self):

        print("[Controller] 启动三线程")

        threads = [

            threading.Thread(
                target=perception_thread,
                args=(
                    self.state,
                    self.env,
                    self.rs_env,
                    self.cam_results,
                    self.home_T_tcp2base
                ),
                daemon=True,
                name="PerceptionThread",
            ),

            threading.Thread(
                target=planning_thread,
                args=(
                    self.state,
                    self.env,
                    None,
                    self.home_T_tcp2base
                ),
                daemon=True,
                name="PlanningThread",
            ),

            threading.Thread(
                target=execution_thread,
                args=(self.state, self.env),
                daemon=True,
                name="ExecutionThread",
            )

        ]

        self.worker_threads = threads

        for t in threads:
            t.start()

    def _workers_alive(self):
        return any(t.is_alive() for t in self.worker_threads)

    def _ensure_workers(self):
        if self.state is None or self.state.stop_all.is_set() or not self._workers_alive():
            self.state = SharedState()
            self._start_worker_threads()

    def _signal_worker_stop(self):
        if self.state is None:
            return

        with self.state.lock:
            self.state.tracking_mode = False
            self.state.verify_mode = False
            self.state.task_success = False

        self.state.abort_execution.set()
        self.state.task_done.set()
        self.state.need_replan.set()
        self.state.plan_ready.set()
        self.state.stop_all.set()

    @staticmethod
    def _join_thread(thread, timeout=1.0):
        if thread is None:
            return
        if thread is threading.current_thread():
            return
        if thread.is_alive():
            thread.join(timeout=timeout)

    @staticmethod
    def _safe_close(obj, *method_names):
        if obj is None:
            return

        for method_name in method_names:
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as e:
                    print(f"[Cleanup] {method_name} failed:", e)
                return

    # ================================
    # 相机线程
    # ================================

    def _start_camera_thread(self):

        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True
        )

        self.camera_thread.start()

    def _camera_loop(self):

        while not self.shutdown_event.is_set():

            try:

                obs = self.rs_env.step()

                frame = obs["rgb"]

                with self.lock:
                    self.latest_frame = frame

            except Exception as e:

                if self.shutdown_event.is_set():
                    break
                print("[Camera Error]", e)

            time.sleep(0.05)

    # ================================
    # 启动任务
    # ================================

    def start_task(self, instruction):

        if instruction.strip() == "":
            return "Instruction empty"

        if self.running:
            return "Task already running"

        if self.worker_thread is not None and self.worker_thread.is_alive():
            return "Previous task is still stopping"

        self._ensure_workers()

        self.running = True
        self.current_instruction = instruction
        self.stop_flag = False

        self.worker_thread = threading.Thread(
            target=self._task_runner,
            daemon=True,
            name="TaskRunner",
        )

        self.worker_thread.start()

        return "Task started"

    # ================================
    # 任务执行
    # ================================

    def _task_runner(self):

        try:

            run_all_tasks_by_instruction_with_list(
                self.state,
                self.env,
                self.rs_env,
                self.cam_results,
                self.current_instruction,
                self.home_T_tcp2base
            )

        except Exception as e:

            print("[Task Error]", e)

        finally:
            self.running = False

    # ================================
    # 停止任务
    # ================================

    def stop_task(self):

        if not self.running:
            return "No task running"

        self._signal_worker_stop()
        self._join_thread(self.worker_thread, timeout=5.0)
        for thread in list(self.worker_threads):
            self._join_thread(thread, timeout=1.0)

        self.running = False
        self.worker_thread = None
        self.worker_threads = []

        return "Stop signal sent"

    def shutdown(self):
        if self.shutdown_event.is_set():
            return

        self.shutdown_event.set()
        self.running = False
        self._signal_worker_stop()

        self._join_thread(self.worker_thread, timeout=5.0)
        for thread in list(self.worker_threads):
            self._join_thread(thread, timeout=1.0)
        self._join_thread(self.camera_thread, timeout=1.0)

        self.worker_thread = None
        self.worker_threads = []

        self._safe_close(self.rs_env, "close", "stop", "release")
        self._safe_close(self.env, "close", "stop", "disconnect")

    # ================================
    # UI读取状态
    # ================================

    def read_state(self):

        with self.lock:
            frame = self.latest_frame

        task_text = "None"
        phase_text = "IDLE"
        xyz_text = "-"

        if self.state is None:
            return frame, task_text, phase_text, xyz_text

        with self.state.lock:

            if self.state.current_task:

                task_text = f"{self.state.current_task}"

            phase_text = str(self.state.task_phase)

            if self.state.latest_point_3d is not None:

                xyz = self.state.latest_point_3d

                xyz_text = f"x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f}"

        return frame, task_text, phase_text, xyz_text


# ================================
# Gradio UI
# ================================


def build_ui(controller):

    with gr.Blocks(title="Robot Pick and Place Demo") as demo:

        gr.Markdown(
            "# 🤖 机械臂抓取放置任务演示"
        )

        with gr.Row():

            instruction = gr.Textbox(
                label="Instruction",
                value="Pick the baseball and place on white plate."
            )

        with gr.Row():

            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")

        with gr.Row():

            camera = gr.Image(label="Realsense Camera")

        with gr.Row():

            task_text = gr.Textbox(label="当前任务")

            phase_text = gr.Textbox(label="任务阶段")

            xyz_text = gr.Textbox(label="目标位置")

        action_feedback = gr.Textbox(label="动作反馈")

        # 按钮逻辑
        start_btn.click(
            fn=controller.start_task,
            inputs=[instruction],
            outputs=[action_feedback],
            queue=False
        )

        stop_btn.click(
            fn=controller.stop_task,
            outputs=[action_feedback],
            queue=False
        )

        # 定时刷新 UI
        timer = gr.Timer(0.1)

        timer.tick(
            fn=controller.read_state,
            outputs=[
                camera,
                task_text,
                phase_text,
                xyz_text
            ],
            queue=False
        )

    return demo


# ================================
# main
# ================================


def main():
    _patch_gradio_schema_bool_bug()

    controller = PnPController()

    demo = build_ui(controller)

    try:
        demo.queue(api_open=False).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_api=False
        )
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
