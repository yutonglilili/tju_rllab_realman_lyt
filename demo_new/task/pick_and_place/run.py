import argparse
import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_new.skills.pnp_skill.graspgen_runtime import (
    DEFAULT_GRASPGEN_ENV,
    DEFAULT_GRASPGEN_GRIPPER_CONFIG,
    DEFAULT_GRASPGEN_SERVER_SCRIPT,
    DEFAULT_GRASPGEN_STARTUP_TIMEOUT_S,
    DEFAULT_WRIST_CAMERA_SERIAL,
    build_wrist_runtime,
)
from demo_new.skills.pnp_skill.pick_and_place import (
    init_camera_env,
    init_robot_env,
    init_state,
    run_all_tasks_by_instruction_with_position_description,
    shutdown_pnp_system,
    start_pnp_system,
)
from demo_new.skills.tools.config_utils import resolve_config_path


DEFAULT_ROBOT_IP = "192.168.101.19"
DEFAULT_FIXED_CAMERA_SERIAL = "f1471338"
DEFAULT_CAM_RESULTS_PATH = (
    "/home/lyt/tju_rllab_realman_lyt/camera/20260325_031804/camera_results.json"
)
DEFAULT_INSTRUCTION = "把所有水果放到蓝色盘子里，把所有玩具放到粉色盘子里。"

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
        default=DEFAULT_GRASPGEN_STARTUP_TIMEOUT_S,
        help="Maximum wait time for a freshly started GraspGen server.",
    )
    return parser.parse_args()

def _build_wrist_runtime(state, args):
    """Thin wrapper over the shared build_wrist_runtime().

    Returns (wrist_rs_env, graspgen_client, wrist_handeye_config, server_manager),
    or all-None when the GraspGen branch is explicitly disabled.
    """
    if args.disable_wrist_graspgen:
        print("[run.py] ℹ️ 已显式关闭 Wrist GraspGen 分支，脚本将退回启发式抓取。")
        return None, None, None, None

    runtime = build_wrist_runtime(
        state.config,
        wrist_camera_serial=args.wrist_camera_serial,
        graspgen_host=args.graspgen_host,
        graspgen_port=args.graspgen_port,
        graspgen_timeout_ms=args.graspgen_timeout_ms,
        handeye_calib_json=args.wrist_handeye_json,
        handeye_frame=args.wrist_handeye_frame,
        env_name=args.graspgen_env_name,
        server_script=args.graspgen_server_script,
        gripper_config=args.graspgen_gripper_config,
        startup_timeout_s=args.graspgen_startup_timeout_s,
        auto_start_server=not args.no_auto_start_graspgen_server,
    )
    return (
        runtime.wrist_rs_env,
        runtime.graspgen_client,
        runtime.wrist_handeye_config,
        runtime.server_manager,
    )


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



