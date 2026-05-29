import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_new.skills.pnp_skill.graspgen_bridge import GraspGenClientBridge
from demo_new.skills.pnp_skill.pick_and_place import (
    init_camera_env,
    init_robot_env,
    init_state,
    init_wrist_grasp_env,
    run_all_tasks_by_instruction_with_position_description,
    shutdown_pnp_system,
    start_pnp_system,
)
from demo_new.skills.tools.config_utils import resolve_config_path


DEFAULT_ROBOT_IP = "192.168.101.19"
DEFAULT_FIXED_CAMERA_SERIAL = "f1471338"
DEFAULT_WRIST_CAMERA_SERIAL = "342522073663"
DEFAULT_CAM_RESULTS_PATH = (
    "/home/lyt/tju_rllab_realman_lyt/camera/20260325_031804/camera_results.json"
)
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
DEFAULT_INSTRUCTION = "把马克笔放到粉色盘子里。"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pick-and-place pipeline with third-view tracking, wrist-camera GraspGen grasping, and execution fallback.",
    )
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--camera-serial", type=str, default=DEFAULT_FIXED_CAMERA_SERIAL)
    parser.add_argument(
        "--cam-results-path",
        type=str,
        default=DEFAULT_CAM_RESULTS_PATH,
        help="Third-view camera calibration json used by make_target_T.",
    )
    parser.add_argument(
        "--wrist-camera-serial",
        type=str,
        default=DEFAULT_WRIST_CAMERA_SERIAL,
        help="Wrist RealSense serial for local close-range grasp planning.",
    )
    parser.add_argument(
        "--disable-wrist-graspgen",
        action="store_true",
        help="Disable the wrist-camera + GraspGen branch and use heuristic pick only.",
    )
    parser.add_argument(
        "--wrist-handeye-json",
        type=str,
        default=None,
        help="Optional hand-eye calibration json. If omitted, built-in wrist calibration defaults are used.",
    )
    parser.add_argument(
        "--wrist-handeye-frame",
        type=str,
        default="eef",
        choices=("eef", "tcp"),
    )
    parser.add_argument(
        "--graspgen-host",
        type=str,
        default=None,
        help="Override the GraspGen ZMQ host. Defaults to the task config / skill config value.",
    )
    parser.add_argument(
        "--graspgen-port",
        type=int,
        default=None,
        help="Override the GraspGen ZMQ port. Defaults to the task config / skill config value.",
    )
    parser.add_argument(
        "--graspgen-timeout-ms",
        type=int,
        default=None,
        help="Override the GraspGen client timeout in milliseconds.",
    )
    parser.add_argument(
        "--graspgen-env-name",
        type=str,
        default=DEFAULT_GRASPGEN_ENV,
        help="Conda environment name used to run the GraspGen server.",
    )
    parser.add_argument(
        "--graspgen-gripper-config",
        type=str,
        default=DEFAULT_GRASPGEN_GRIPPER_CONFIG,
        help="Gripper config yaml passed to the GraspGen server.",
    )
    parser.add_argument(
        "--graspgen-server-script",
        type=str,
        default=DEFAULT_GRASPGEN_SERVER_SCRIPT,
        help="Server entry script that runs inside the GraspGen environment.",
    )
    parser.add_argument(
        "--no-auto-start-graspgen-server",
        action="store_true",
        help="Require an existing GraspGen server instead of auto-starting one from this script.",
    )
    parser.add_argument(
        "--graspgen-startup-timeout-s",
        type=float,
        default=120.0,
        help="Maximum wait time for a freshly started GraspGen server.",
    )
    return parser.parse_args()


def _resolve_conda_executable():
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
                [
                    lsof_path,
                    "-nP",
                    "-t",
                    f"-iTCP:{self.port}",
                    "-sTCP:LISTEN",
                ],
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
            raise RuntimeError(f"Failed to terminate the process occupying port {self.port}: pid={pid}")

    def _force_release_port(self):
        pids = self._find_listening_pids()
        if not pids:
            return

        print(f"[run.py] ⚠️ 端口 {self.port} 已被占用，正在强制关闭占用进程: {pids}")
        for pid in pids:
            self._terminate_pid(pid)
        time.sleep(0.5)

    def ensure_running(self):
        if not self.auto_start:
            if self._health_check(timeout_ms=min(self.timeout_ms, 1500), port=self.port):
                print(f"[run.py] ✅ 复用现有 GraspGen server: tcp://{self.host}:{self.port}")
                return
            raise RuntimeError(
                f"GraspGen server is not reachable at tcp://{self.host}:{self.port}. "
                "Please start it manually or remove --no-auto-start-graspgen-server."
            )

        conda_exe = _resolve_conda_executable()
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

        print(f"[run.py] 🚀 启动本地 GraspGen server: tcp://{self.host}:{self.port}")
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
                print(f"[run.py] ✅ GraspGen server ready: tcp://{self.host}:{self.port}")
                return

            elapsed_s = float(self.startup_timeout_s) - max(0.0, deadline - time.time())
            progress_bucket = int(elapsed_s // 5)
            if progress_bucket != last_progress_bucket:
                last_progress_bucket = progress_bucket
                recent_lines = self._read_recent_log_lines()
                print(
                    f"[run.py] ⏳ 等待 GraspGen server 启动中... "
                    f"{elapsed_s:.0f}s / {self.startup_timeout_s:.0f}s"
                )
                if recent_lines:
                    print("[run.py] recent server log:")
                    for line in recent_lines:
                        print(f"[run.py]   {line}")
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

        print("[run.py] 🛑 关闭当前脚本拉起的 GraspGen server...")
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5.0)


def _build_wrist_runtime(state, args):
    if args.disable_wrist_graspgen:
        print("[run.py] ℹ️ 已显式关闭 Wrist GraspGen 分支，脚本将退回启发式抓取。")
        return None, None, None, None

    host = args.graspgen_host or str(getattr(state.config, "GRASPGEN_SERVER_HOST", "127.0.0.1"))
    port = int(
        args.graspgen_port
        if args.graspgen_port is not None
        else getattr(state.config, "GRASPGEN_SERVER_PORT", 5556)
    )
    timeout_ms = int(
        args.graspgen_timeout_ms
        if args.graspgen_timeout_ms is not None
        else getattr(state.config, "GRASPGEN_TIMEOUT_MS", 60_000)
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
        env_name=args.graspgen_env_name,
        server_script=os.path.abspath(args.graspgen_server_script),
        gripper_config=os.path.abspath(args.graspgen_gripper_config),
        startup_timeout_s=float(args.graspgen_startup_timeout_s),
        auto_start=not args.no_auto_start_graspgen_server,
    )
    server_manager.ensure_running()

    wrist_rs_env, graspgen_client, wrist_handeye_config = init_wrist_grasp_env(
        args.wrist_camera_serial,
        graspgen_host=host,
        graspgen_port=server_manager.port,
        graspgen_timeout_ms=timeout_ms,
        handeye_calib_json=args.wrist_handeye_json,
        handeye_frame=args.wrist_handeye_frame,
        wait_for_server=False,
    )
    print(f"[run.py] ✅ Wrist camera ready: {args.wrist_camera_serial}")
    print(f"[run.py] ✅ Wrist hand-eye source: {wrist_handeye_config.source}")
    print(f"[run.py] ✅ Wrist GraspGen client connected to tcp://{host}:{server_manager.port}")
    return wrist_rs_env, graspgen_client, wrist_handeye_config, server_manager


def main():
    args = parse_args()
    task_config_path = resolve_config_path(__file__)

    state = init_state(task_config_path=task_config_path)

    env = None
    rs_env = None
    wrist_rs_env = None
    graspgen_client = None
    wrist_handeye_config = None
    server_manager = None

    try:
        env, home_T_tcp2base = init_robot_env(args.robot_ip)
        rs_env, cam_results = init_camera_env(args.camera_serial, args.cam_results_path)
        (
            wrist_rs_env,
            graspgen_client,
            wrist_handeye_config,
            server_manager,
        ) = _build_wrist_runtime(state, args)

        start_pnp_system(
            state,
            env,
            rs_env,
            cam_results,
            home_T_tcp2base,
            wrist_rs_env=wrist_rs_env,
            graspgen_client=graspgen_client,
            wrist_handeye_config=wrist_handeye_config,
        )

        run_all_tasks_by_instruction_with_position_description(
            state,
            env,
            rs_env,
            cam_results,
            args.instruction,
            home_T_tcp2base,
        )

    except KeyboardInterrupt:
        print("\n[停止] 收到键盘中断，正在停止...")
    except Exception as exc:
        print(f"\n[错误] 未捕获异常: {exc}")
        traceback.print_exc()
    finally:
        shutdown_pnp_system(
            state,
            env=env,
            rs_env=rs_env,
            wrist_rs_env=wrist_rs_env,
            graspgen_client=graspgen_client,
        )
        if server_manager is not None:
            server_manager.stop()


if __name__ == "__main__":
    main()
