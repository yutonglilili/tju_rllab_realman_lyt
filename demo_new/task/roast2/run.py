import json
import os
import sys
import traceback
import numpy as np

# =========================================================
# project path
# =========================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# imports
# =========================================================

from realman.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T
from realman.open3d_realsense_env import Open3dRealsenseEnv

from demo_new.vlm_utils.multi_pointing_vllm_get_point_utils import get_point_vllm
from demo_new.vlm_utils.vllm_from_api_key import generate_air_fryer_subtasks

from demo_new.skills.tools.config_utils import resolve_config_path
from demo_new.skills.tools.utils import make_target_T, make_lift_T

from demo_new.skills.air_fryer_skill.air_fryer import open_action, close_action, rotate_action
from demo_new.skills.pnp_skill.pick_and_place import init_state, start_pnp_system, run_single_task

# =========================================================
# helper
# =========================================================

def get_motion_profile(config, profile_name):

    profiles = {}

    if hasattr(config, "get"):
        profiles = config.get("motion_profiles", {}) or {}
    elif isinstance(config, dict):
        profiles = config.get("motion_profiles", {}) or {}

    if not profiles:

        if profile_name != "normal":
            print(
                f"[WARN] no motion_profiles configured, "
                f"profile '{profile_name}' will use base pnp config"
            )

        return {}

    if profile_name not in profiles:

        print(
            f"[WARN] profile '{profile_name}' "
            f"not found -> fallback to normal"
        )

        profile_name = "normal"

    return profiles.get(profile_name, {})


def get_motion_profile_name(motion, *, profile_key, context_key):

    if isinstance(motion, str):
        return motion or "normal"

    if not isinstance(motion, dict):
        return "normal"

    profile_name = motion.get(profile_key)

    if profile_name:
        return profile_name

    context_name = motion.get(context_key)

    if context_name:
        return context_name

    return "normal"

# =========================================================
# air fryer skill
# =========================================================

def open_air_fryer(env, rs_env, cam_results, home_T_tcp2base):

    obs = rs_env.step()

    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(
        image_rgb,
        "Point at the handle of the air fryer.",
        save_path=None
    )

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    target_T = make_lift_T(target_T, lift_x=0.02, lift_y=-0.01)

    tcp_pose = realman_xyzrpy_from_T(target_T)

    tcp_pose[3:] = np.array([0.0623, 0.4881, 3.1218])

    direction_xyz = np.array([1, 0, 0])

    open_action(env, tcp_pose, direction_xyz)


def close_air_fryer(env, rs_env, cam_results, home_T_tcp2base):

    obs = rs_env.step()

    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(
        image_rgb,
        "Point at the handle of the air fryer.",
        save_path=None
    )

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    target_T = make_lift_T(target_T, lift_x=0.02, lift_y=-0.01, lift_z=-0.01)

    tcp_pose = realman_xyzrpy_from_T(target_T)

    tcp_pose[3:] = np.array([0.0623, 0.4881, 3.1218])

    direction_xyz = np.array([1, 0, 0])

    close_action(env, tcp_pose, direction_xyz)


def set_time(env,rs_env,cam_results,home_T_tcp2base,rotate_angle):

    obs = rs_env.step()

    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(
        image_rgb,
        "Point at the round knob of the air fryer.",
        save_path=None
    )

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    target_T = make_lift_T(target_T, lift_x=0.035, lift_y=-0.02, lift_z=-0.02)

    tcp_pose = realman_xyzrpy_from_T(target_T)

    tcp_pose[3:] = np.array([0, 0, 3.1412])

    direction_xyz = np.array([1, 0, 0])

    rotate_action(env, tcp_pose, direction_xyz, rotate_angle)

# =========================================================
# pnp
# =========================================================

def execute_pnp_task(
    task,
    state,
    env,
    rs_env,
    cam_results,
    home_T_tcp2base,
    config,
):

    args = task["args"]

    motion = task.get("motion", {})

    pick_profile_name = get_motion_profile_name(
        motion,
        profile_key="pick_profile",
        context_key="pick_context",
    )

    place_profile_name = get_motion_profile_name(
        motion,
        profile_key="place_profile",
        context_key="place_context",
    )

    pick_profile = get_motion_profile(
        config,
        pick_profile_name
    )

    place_profile = get_motion_profile(
        config,
        place_profile_name
    )

    print(
        "[PnP] profiles -> "
        f"pick: {pick_profile_name}, "
        f"place: {place_profile_name}"
    )

    run_single_task(
        state=state,

        env=env,

        rs_env=rs_env,

        cam_results=cam_results,

        task={
            "pick": args["pick"],
            "place": args["place"]
        },

        home_T_tcp2base=home_T_tcp2base,

        pick_profile=pick_profile,

        place_profile=place_profile,
    )

# =========================================================
# execute subtasks
# =========================================================

def execute_subtasks(
    subtasks,
    state,
    env,
    rs_env,
    cam_results,
    home_T_tcp2base,
    config,
):

    for task in subtasks:

        print(
            json.dumps(
                task,
                ensure_ascii=False,
                indent=2
            )
        )

        skill = task["skill"]

        # =================================================
        # open
        # =================================================

        if skill == 0:

            open_air_fryer(
                env,
                rs_env,
                cam_results,
                home_T_tcp2base
            )

        # =================================================
        # close
        # =================================================

        elif skill == 1:

            close_air_fryer(
                env,
                rs_env,
                cam_results,
                home_T_tcp2base
            )

        # =================================================
        # pnp
        # =================================================

        elif skill == 2:

            execute_pnp_task(
                task,
                state,
                env,
                rs_env,
                cam_results,
                home_T_tcp2base,
                config
            )

        # =================================================
        # timer
        # =================================================

        elif skill == 3:

            minutes = task["args"]["minutes"]

            set_time(
                env,
                rs_env,
                cam_results,
                home_T_tcp2base,
                rotate_angle=minutes
            )

# =========================================================
# main
# =========================================================

def main():

    instruction = "帮我烤苹果，定时20分钟"

    # =====================================================
    # config
    # =====================================================

    task_config_path = resolve_config_path(
        __file__
    )

    # =====================================================
    # env
    # =====================================================

    env = RealmanEnv(
        robot_ip="192.168.101.19",
        mode="sync"
    )

    rs_env = Open3dRealsenseEnv(
        "f1471338"
    )

    cam_results_path = (
        "/home/zhangzhao/lyt/camera/"
        "20260325_031804/camera_results.json"
    )

    with open(cam_results_path, "r") as f:

        cam_results = json.load(f)

    env.reset()

    robot_state = env.get_state()

    home_T_tcp2base = T_from_realman_xyzrpy(
        robot_state.pose
    )

    # =====================================================
    # init pnp
    # =====================================================

    state = init_state(
        task_config_path=task_config_path
    )

    start_pnp_system(
        state,
        env,
        rs_env,
        cam_results,
        home_T_tcp2base
    )

    # =====================================================
    # vlm planning
    # =====================================================

    obs = rs_env.step()

    image_rgb = obs["rgb"]

    subtasks = generate_air_fryer_subtasks(
        image_rgb=image_rgb,
        instruction=instruction
    )

    print(
        json.dumps(
            subtasks,
            ensure_ascii=False,
            indent=2
        )
    )

    # =====================================================
    # execute
    # =====================================================

    try:

        execute_subtasks(
            subtasks=subtasks,

            state=state,

            env=env,

            rs_env=rs_env,

            cam_results=cam_results,

            home_T_tcp2base=home_T_tcp2base,

            config=state.config
        )

    except KeyboardInterrupt:

        print("\n[STOP] KeyboardInterrupt")

    except Exception as e:

        print("\n[ERROR]", e)

        traceback.print_exc()

    finally:

        print("\n[CLEANUP] stop threads")

        state.stop_all.set()

# =========================================================

if __name__ == "__main__":

    main()
