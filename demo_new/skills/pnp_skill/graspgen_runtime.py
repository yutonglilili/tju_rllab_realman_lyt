"""Shared GraspGen + wrist-camera runtime helpers for the pnp skill.

These were originally written inline in
``demo_new/task/pick_and_place/run.py``. They are extracted here so that every
caller that starts the pnp system (the standalone pnp task, the air-fryer /
roast task, and the interactive interface) can reuse exactly the same GraspGen
server lifecycle management and wrist-camera runtime construction instead of
re-implementing it.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from demo_new.skills.pnp_skill.graspgen_bridge import GraspGenClientBridge
from demo_new.skills.pnp_skill.pick_and_place import init_wrist_grasp_env


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

DEFAULT_WRIST_CAMERA_SERIAL = "342522073663"
DEFAULT_GRASPGEN_ENV = "GraspGen"
DEFAULT_GRASPGEN_GRIPPER_CONFIG = os.path.join(
    PROJECT_ROOT,
    "GraspGen",
    "GraspGenModels",
    "checkpoints",
    "graspgen_robotiq_2f_140.yml",
)
DEFAULT_GRASPGEN_SERVER_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "GraspGen",
    "client-server",
    "graspgen_server.py",
)
DEFAULT_GRASPGEN_STARTUP_TIMEOUT_S = 120.0

def resolve_conda_executable():
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        "/home/lyt/miniconda3/bin/conda",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate the `conda` executable. Set CONDA_EXE or add conda to PATH."
    )


@dataclass
class ManagedGraspGenServer:
    host: str
    port: int
    timeout_ms: int
    env_name: str
    server_script: str
    gripper_config: str
    startup_timeout_s: float
    auto_start: bool = True

    def __post_init__(self):
        self.process = None
        self.started_here = False
        self.port = int(self.port)
        self._log_path = None

    def _health_check(self, timeout_ms=None, port=None):
        timeout_ms = int(self.timeout_ms if timeout_ms is None else timeout_ms)
        port = int(self.port if port is None else port)
        client = None
        try:
            client = GraspGenClientBridge(
                host=self.host,
                port=port,
                timeout_ms=timeout_ms,
                wait_for_server=False,
            )
            return bool(client.health_check())
        except Exception:
            return False
        finally:
            if client is not None:
                client.close()

    def _read_log_tail(self, max_chars=4000):
        if self._log_path is None or not os.path.exists(self._log_path):
            return ""
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return ""
        return text[-max_chars:]

    def _read_recent_log_lines(self, max_lines=6):
        tail = self._read_log_tail()
        if not tail:
            return []
        lines = [line.rstrip() for line in tail.splitlines() if line.strip()]
        return lines[-max_lines:]

    def _find_listening_pids(self):
        pid_set = set()

        lsof_path = shutil.which("lsof")
        if lsof_path:
            result = subprocess.run(
                [lsof_path, "-nP", "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
            )
            if result.returncode in (0, 1):
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        pid_set.add(int(line))

        if not pid_set:
            fuser_path = shutil.which("fuser")
            if fuser_path:
                result = subprocess.run(
                    [fuser_path, "-n", "tcp", str(self.port)],
                    capture_output=True,
                    text=True,
                )
                output = f"{result.stdout}\n{result.stderr}"
                for token in output.replace("/", " ").replace(":", " ").split():
                    if token.isdigit():
                        pid_set.add(int(token))

        return sorted(pid_set)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _terminate_pid(self, pid: int):
        for sig, wait_s in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
            if not self._pid_exists(pid):
                return
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return

            deadline = time.time() + wait_s
            while time.time() < deadline:
                if not self._pid_exists(pid):
                    return
                time.sleep(0.2)

        if self._pid_exists(pid):
            raise RuntimeError(
                f"Failed to terminate the process occupying port {self.port}: pid={pid}"
            )

    def _force_release_port(self):
        pids = self._find_listening_pids()
        if not pids:
            return

        print(f"[graspgen_runtime] ⚠️ 端口 {self.port} 已被占用，正在强制关闭占用进程: {pids}")
        for pid in pids:
            self._terminate_pid(pid)
        time.sleep(0.5)

    def ensure_running(self):
        if not self.auto_start:
            if self._health_check(timeout_ms=min(self.timeout_ms, 1500), port=self.port):
                print(f"[graspgen_runtime] ✅ 复用现有 GraspGen server: tcp://{self.host}:{self.port}")
                return
            raise RuntimeError(
                f"GraspGen server is not reachable at tcp://{self.host}:{self.port}. "
                "Please start it manually or enable auto-start."
            )

        conda_exe = resolve_conda_executable()
        if not os.path.exists(self.server_script):
            raise FileNotFoundError(f"GraspGen server script not found: {self.server_script}")
        if not os.path.exists(self.gripper_config):
            raise FileNotFoundError(f"GraspGen gripper config not found: {self.gripper_config}")
        self._force_release_port()

        cmd = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            self.env_name,
            "python",
            self.server_script,
            "--gripper_config",
            self.gripper_config,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        print(f"[graspgen_runtime] 🚀 启动本地 GraspGen server: tcp://{self.host}:{self.port}")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".log",
            prefix="graspgen_server_",
            delete=False,
        ) as log_file:
            self._log_path = log_file.name
            self.process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                start_new_session=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        self.started_here = True

        deadline = time.time() + float(self.startup_timeout_s)
        last_progress_bucket = -1
        while time.time() < deadline:
            if self.process.poll() is not None:
                log_tail = self._read_log_tail()
                raise RuntimeError(
                    f"GraspGen server exited early with code {self.process.returncode}.\n"
                    f"---- server log tail ----\n{log_tail}"
                )

            if self._health_check(timeout_ms=min(self.timeout_ms, 1500), port=self.port):
                print(f"[graspgen_runtime] ✅ GraspGen server ready: tcp://{self.host}:{self.port}")
                return

            elapsed_s = float(self.startup_timeout_s) - max(0.0, deadline - time.time())
            progress_bucket = int(elapsed_s // 5)
            if progress_bucket != last_progress_bucket:
                last_progress_bucket = progress_bucket
                recent_lines = self._read_recent_log_lines()
                print(
                    f"[graspgen_runtime] ⏳ 等待 GraspGen server 启动中... "
                    f"{elapsed_s:.0f}s / {self.startup_timeout_s:.0f}s"
                )
                if recent_lines:
                    print("[graspgen_runtime] recent server log:")
                    for line in recent_lines:
                        print(f"[graspgen_runtime]   {line}")
            time.sleep(1.0)

        log_tail = self._read_log_tail()
        raise TimeoutError(
            f"Timed out after {self.startup_timeout_s:.1f}s waiting for GraspGen server "
            f"on port {self.port}.\n---- server log tail ----\n{log_tail}"
        )

    def stop(self):
        if not self.started_here or self.process is None:
            return
        if self.process.poll() is not None:
            return

        print("[graspgen_runtime] 🛑 关闭当前进程拉起的 GraspGen server...")
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5.0)

@dataclass
class WristGraspRuntime:
    """Bundle of everything start_pnp_system() needs for the GraspGen branch."""

    wrist_rs_env: object
    graspgen_client: object
    wrist_handeye_config: object
    server_manager: Optional[ManagedGraspGenServer]

    def shutdown(self):
        if self.server_manager is not None:
            try:
                self.server_manager.stop()
            except Exception as exc:
                print(f"[graspgen_runtime] ⚠️ 关闭 GraspGen server 失败: {exc}")


def build_wrist_runtime(
    config,
    *,
    wrist_camera_serial=DEFAULT_WRIST_CAMERA_SERIAL,
    graspgen_host=None,
    graspgen_port=None,
    graspgen_timeout_ms=None,
    handeye_calib_json=None,
    handeye_frame="eef",
    env_name=DEFAULT_GRASPGEN_ENV,
    server_script=DEFAULT_GRASPGEN_SERVER_SCRIPT,
    gripper_config=DEFAULT_GRASPGEN_GRIPPER_CONFIG,
    startup_timeout_s=DEFAULT_GRASPGEN_STARTUP_TIMEOUT_S,
    auto_start_server=True,
):
    """Start (or reuse) a GraspGen server and build the wrist-camera runtime.

    Resolves connection params from ``config`` (the pnp SharedState config) when
    explicit overrides are not given, mirroring task/pick_and_place/run.py.
    Returns a fully-wired WristGraspRuntime. Raises on failure; callers that want
    a graceful fallback (e.g. the interactive interface) should catch and degrade
    to the heuristic pick branch.
    """
    host = graspgen_host or str(getattr(config, "GRASPGEN_SERVER_HOST", "127.0.0.1"))
    port = int(
        graspgen_port
        if graspgen_port is not None
        else getattr(config, "GRASPGEN_SERVER_PORT", 5556)
    )
    timeout_ms = int(
        graspgen_timeout_ms
        if graspgen_timeout_ms is not None
        else getattr(config, "GRASPGEN_TIMEOUT_MS", 60_000)
    )

    # Fail fast if the main control environment cannot create the local ZMQ client.
    dep_probe_client = None
    try:
        dep_probe_client = GraspGenClientBridge(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            wait_for_server=False,
        )
    except ImportError as exc:
        raise ImportError(
            "The main control environment is missing GraspGen client dependencies. "
            "Install `pyzmq`, `msgpack`, and `msgpack-numpy` into `realman_env_lyt`."
        ) from exc
    finally:
        if dep_probe_client is not None:
            dep_probe_client.close()

    server_manager = ManagedGraspGenServer(
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        env_name=env_name,
        server_script=os.path.abspath(server_script),
        gripper_config=os.path.abspath(gripper_config),
        startup_timeout_s=float(startup_timeout_s),
        auto_start=auto_start_server,
    )
    server_manager.ensure_running()

    wrist_rs_env, graspgen_client, wrist_handeye_config = init_wrist_grasp_env(
        wrist_camera_serial,
        graspgen_host=host,
        graspgen_port=server_manager.port,
        graspgen_timeout_ms=timeout_ms,
        handeye_calib_json=handeye_calib_json,
        handeye_frame=handeye_frame,
        wait_for_server=False,
    )
    print(f"[graspgen_runtime] ✅ Wrist camera ready: {wrist_camera_serial}")
    print(f"[graspgen_runtime] ✅ Wrist hand-eye source: {wrist_handeye_config.source}")
    print(
        f"[graspgen_runtime] ✅ Wrist GraspGen client connected to "
        f"tcp://{host}:{server_manager.port}"
    )
    return WristGraspRuntime(
        wrist_rs_env=wrist_rs_env,
        graspgen_client=graspgen_client,
        wrist_handeye_config=wrist_handeye_config,
        server_manager=server_manager,
    )


