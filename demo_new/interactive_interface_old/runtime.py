from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from types import SimpleNamespace
from typing import Any

from interactive_interface.task_interface import execute_task, get_task_definition


def _humanize_task_item(task: Any) -> str:
    if not isinstance(task, dict):
        return "-"

    pick = task.get("pick")
    place = task.get("place")
    if pick and place:
        return f"抓取 {pick} -> 放到 {place}"
    return str(task)


class _CameraEnvProxy:
    def __init__(self, runtime: "AppRuntime", wrapped_env: Any) -> None:
        self._runtime = runtime
        self._wrapped_env = wrapped_env
        self._step_lock = threading.Lock()

    def step(self, *args: Any, **kwargs: Any) -> Any:
        with self._step_lock:
            obs = self._wrapped_env.step(*args, **kwargs)
        self._runtime._cache_camera_observation(obs)
        return obs

    def close(self) -> None:
        self._wrapped_env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_env, name)


class AppRuntime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.logs: deque[str] = deque(maxlen=80)
        self.initialized = False
        self.initialization_error = ""
        self.status_message = "等待初始化。"
        self.runtime_signature: tuple[str, str, str] | None = None
        self.env = None
        self.rs_env = None
        self.cam_results = None
        self.home_T_tcp2base = None
        self.last_camera_frame = None
        self.last_camera_error = ""
        self.task_state = None
        self.stop_requested = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.active_task_id = ""
        self.active_task_title = ""
        self.active_instruction = ""

        self.log("界面已启动，等待任务。")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.logs.append(f"[{timestamp}] {message}")
            self.status_message = message

    def set_current_task(self, *, task_title: str, instruction: str) -> None:
        with self.lock:
            self.active_task_title = task_title
            self.active_instruction = instruction.strip()

    def is_busy(self) -> bool:
        with self.lock:
            return self.worker_thread is not None and self.worker_thread.is_alive()

    def attach_task_state(self, task_state: Any) -> None:
        with self.lock:
            self.task_state = task_state

    def detach_task_state(self, task_state: Any) -> None:
        with self.lock:
            if self.task_state is task_state:
                self.task_state = None

    def _cache_camera_observation(self, obs: Any) -> Any:
        frame = None
        if isinstance(obs, dict):
            frame = obs.get("rgb")

        if frame is None:
            return None

        with self.lock:
            self.last_camera_frame = frame
            self.last_camera_error = ""

        return frame

    def _read_camera_frame(self, rs_env: Any, cached_frame: Any) -> Any:
        try:
            obs = rs_env.step()
        except Exception as exc:
            with self.lock:
                self.last_camera_error = f"{type(exc).__name__}: {exc}"
            return cached_frame

        frame = self._cache_camera_observation(obs)
        if frame is None:
            return cached_frame
        return frame

    def _sync_home_pose_from_robot(self) -> None:
        with self.lock:
            env = self.env

        if env is None:
            return

        try:
            from realman.realman_env import T_from_realman_xyzrpy

            robot_state = env.get_state()
            home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)
        except Exception:
            return

        with self.lock:
            self.home_T_tcp2base = home_T_tcp2base

    def _reset_robot_to_initial_state(self) -> None:
        with self.lock:
            initialized = self.initialized
            env = self.env
            rs_env = self.rs_env
            cached_frame = self.last_camera_frame

        if not initialized or env is None:
            return

        try:
            env.reset()
        except Exception as exc:
            error_message = f"Stop reset failed: {type(exc).__name__}: {exc}"
            with self.lock:
                self.status_message = error_message
            self.log(error_message)
            return

        self._sync_home_pose_from_robot()

        if rs_env is not None:
            self._read_camera_frame(rs_env, cached_frame)

        self.log("Current task exited and robot reset to the initial state.")

    def _close_resources(self) -> None:
        env = self.env
        rs_env = self.rs_env

        self.env = None
        self.rs_env = None
        self.cam_results = None
        self.home_T_tcp2base = None
        self.initialized = False
        self.runtime_signature = None

        if rs_env is not None:
            try:
                rs_env.close()
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    def ensure_runtime(self, robot_ip: str, camera_serial: str, cam_results_path: str) -> bool:
        signature = (robot_ip.strip(), camera_serial.strip(), cam_results_path.strip())
        with self.lock:
            if self.is_busy():
                self.log("任务运行中，继续使用当前连接。")
                return self.initialized

            if self.initialized and self.runtime_signature == signature:
                self.log("运行时已就绪。")
                return True

        self.shutdown_runtime("准备连接机器人和相机。", allow_busy=False, log_message=False)

        try:
            from demo_new.skills.pnp_skill.pick_and_place import init_camera_env, init_robot_env
        except Exception as exc:
            with self.lock:
                self.initialization_error = f"{type(exc).__name__}: {exc}"
                self.status_message = "缺少机器人运行依赖，初始化失败。"
            self.log("初始化失败。")
            return False

        env = None
        rs_env = None
        cam_results = None
        home_T_tcp2base = None

        try:
            env, home_T_tcp2base = init_robot_env(signature[0])
            rs_env, cam_results = init_camera_env(signature[1], signature[2])
        except Exception as exc:
            if rs_env is not None:
                try:
                    rs_env.close()
                except Exception:
                    pass
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

            with self.lock:
                self.initialized = False
                self.initialization_error = f"{type(exc).__name__}: {exc}"
                self.status_message = "运行时初始化失败。"
            self.log("运行时初始化失败。")
            return False

        rs_env = _CameraEnvProxy(self, rs_env)

        with self.lock:
            self.env = env
            self.rs_env = rs_env
            self.cam_results = cam_results
            self.home_T_tcp2base = home_T_tcp2base
            self.runtime_signature = signature
            self.initialized = True
            self.initialization_error = ""
            self.status_message = "运行时初始化成功。"

        self.log("运行时初始化成功。")
        self.get_camera_frame(force_refresh=True)
        return True

    def shutdown_runtime(self, reason: str, *, allow_busy: bool, log_message: bool = True) -> bool:
        with self.lock:
            busy = self.is_busy()
            if busy and not allow_busy:
                self.status_message = "任务运行中，暂不关闭运行时。"
                if log_message:
                    self.log("任务运行中，暂不关闭运行时。")
                return False

            self._close_resources()
            self.last_camera_frame = None
            self.last_camera_error = ""
            self.initialization_error = ""
            self.status_message = reason

        if log_message:
            self.log(reason)
        return True

    def require_resources(self) -> SimpleNamespace:
        with self.lock:
            if not self.initialized or self.env is None or self.rs_env is None:
                raise RuntimeError("运行时未初始化，请先初始化再开始执行。")

            return SimpleNamespace(
                env=self.env,
                rs_env=self.rs_env,
                cam_results=self.cam_results,
                home_T_tcp2base=self.home_T_tcp2base,
            )

    def request_stop(self) -> None:
        self.stop_requested.set()
        busy = self.is_busy()

        with self.lock:
            task_state = self.task_state
            self.status_message = "已发送停止请求。"

        self.status_message = "Stopping the current task and resetting the robot."

        if task_state is not None:
            try:
                task_state.stop_all.set()
            except Exception:
                pass
            try:
                task_state.abort_execution.set()
            except Exception:
                pass
            try:
                task_state.need_replan.set()
            except Exception:
                pass

        self.log("已发送停止请求。")

        if not busy:
            self._reset_robot_to_initial_state()
            with self.lock:
                self.active_task_id = ""
                self.active_task_title = ""
                self.active_instruction = ""
                self.task_state = None
                self.stop_requested.clear()

    def get_camera_frame(self, *, force_refresh: bool = False) -> Any:
        with self.lock:
            initialized = self.initialized
            busy = self.is_busy()
            rs_env = self.rs_env
            cached_frame = self.last_camera_frame

        if not initialized or rs_env is None:
            return cached_frame

        if busy:
            return cached_frame

        return self._read_camera_frame(rs_env, cached_frame)

    def _snapshot_task_state(self) -> dict[str, Any]:
        with self.lock:
            task_state = self.task_state

        if task_state is None:
            return {
                "current_task": None,
                "task_phase": None,
                "target_description": None,
            }

        try:
            with task_state.lock:
                current_task = task_state.current_task
                task_phase = getattr(task_state.task_phase, "name", str(task_state.task_phase))
                target_description = task_state.target_description
        except Exception:
            current_task = None
            task_phase = None
            target_description = None

        return {
            "current_task": current_task,
            "task_phase": task_phase,
            "target_description": target_description,
        }

    def snapshot_status(self) -> tuple[str, str]:
        with self.lock:
            initialized = self.initialized
            status_message = self.status_message
            active_task_title = self.active_task_title or "-"
            active_instruction = self.active_instruction or "-"
            logs_text = "\n".join(list(self.logs)[-12:])
            init_error = self.initialization_error

        task_snapshot = self._snapshot_task_state()
        current_task_text = _humanize_task_item(task_snapshot["current_task"])
        status_lines = [
            f"运行状态: {'已连接' if initialized else '未连接'}",
            f"当前任务: {active_task_title}",
            f"任务指令: {active_instruction}",
            f"正在执行: {status_message}",
        ]

        if current_task_text != "-":
            status_lines.append(f"当前步骤: {current_task_text}")

        if init_error and not initialized:
            status_lines.append(f"错误: {init_error}")

        return "\n".join(status_lines), logs_text

    def snapshot_full(self) -> tuple[str, str, Any]:
        status_markdown, logs_text = self.snapshot_status()
        camera_frame = self.get_camera_frame(force_refresh=False)
        return status_markdown, logs_text, camera_frame

    def launch_task(
        self,
        *,
        task_id: str,
        instruction: str,
        params: dict[str, Any],
    ) -> bool:
        task_def = get_task_definition(task_id)

        with self.lock:
            if self.is_busy():
                self.log("已有任务在运行。")
                return False

            self.worker_thread = threading.Thread(
                target=self._run_task_worker,
                args=(task_id, instruction, params),
                daemon=True,
            )
            self.active_task_id = task_id
            self.active_task_title = task_def.title
            self.active_instruction = instruction.strip()
            self.status_message = f"开始执行 {task_def.title}。"
            self.stop_requested.clear()
            self.worker_thread.start()

        self.log(f"已启动 {task_def.title}。")
        return True

    def _run_task_worker(
        self,
        task_id: str,
        instruction: str,
        params: dict[str, Any],
    ) -> None:
        task_def = get_task_definition(task_id)

        try:
            result = execute_task(
                task_id=task_id,
                runtime=self,
                instruction=instruction,
                params=params,
            )
        except InterruptedError as exc:
            with self.lock:
                self.status_message = str(exc)
            self.log(str(exc))
        except Exception as exc:
            with self.lock:
                self.status_message = f"{task_def.title} 执行失败。"
            self.log(f"{task_def.title} 执行失败。")
            self.log(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            result_status = result.get("status", "completed")
            if result_status == "stopped":
                self.log(f"{task_def.title} 已停止。")
            else:
                self.log(f"{task_def.title} 已完成。")
        finally:
            stop_requested = self.stop_requested.is_set()
            if stop_requested:
                self._reset_robot_to_initial_state()

            with self.lock:
                self.worker_thread = None
                self.active_task_id = ""
                self.active_task_title = ""
                self.active_instruction = ""
                self.task_state = None
                self.stop_requested.clear()
