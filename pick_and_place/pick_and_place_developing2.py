"""
Clean multi-task pick & place FSM version
"""

from tvla_realenv.open3d_realsense_env import Open3dRealsenseEnv
from tvla_realenv.realman_env import RealmanEnv
import cv2
import copy
import json
import numpy as np
from enum import Enum, auto
import time

from pointing_vllm_get_point_utils_developing import (
    get_point_vllm,
    parse_multi_pick_place_tasks,
)

# ===============================
# Parameters
# ===============================
safe_height = 0.06

h_Parameters_table = {
    "yellow ball": {"pick_down": -0.06, "place_down": 0.06},
    "white can": {"pick_down": -0.03, "place_down": 0.06},
}

def get_h_Parameters_for_object(obj_name):
    return h_Parameters_table.get(obj_name, {"pick_down": 0.03, "place_down": 0.02})

GRIPPER_OPEN = 0.09
GRIPPER_CLOSE = 0.00

# ===============================
# FSM State
# ===============================
class PickPlaceState(Enum):
    IDLE = auto()
    MOVE_ABOVE_PICK = auto()
    MOVE_DOWN_PICK = auto()
    CLOSE_GRIPPER = auto()
    LIFT_AFTER_PICK = auto()
    MOVE_ABOVE_PLACE = auto()
    MOVE_DOWN_PLACE = auto()
    OPEN_GRIPPER = auto()
    LIFT_AFTER_PLACE = auto()
    RETURN_HOME = auto()
    DONE = auto()

# ===============================
# Geometry helpers
# ===============================
def calculate_3d_position(u, v, obs, rs_env, cam_results, z_offset=0.02):
    d = obs["depth"][v, u] / rs_env.meta_obs["depth_scale"]
    xyz_cam = np.linalg.inv(
        np.array(rs_env.meta_obs["intrinsic"])
    ) @ (np.array([u, v, 1.0]) * d)

    xyz_base = np.array(cam_results["Tcam2base"]) @ np.array(
        [xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0]
    )
    xyz_base[2] += z_offset
    return xyz_base[:3]

def make_target_T(obs, u, v, rs_env, cam_results, z_offset):
    T = copy.deepcopy(obs["Ttcp2base"])
    T[:3, 3] = calculate_3d_position(
        u, v, obs, rs_env, cam_results, z_offset
    )
    return T

def make_lift_T(T, lift_height=0.02):
    T_lift = copy.deepcopy(T)
    T_lift[2, 3] += lift_height
    return T_lift

# ===============================
# FSM step
# ===============================
def step_pick_place_fsm(state, state_t0, now, action, targets):
    if state == PickPlaceState.MOVE_ABOVE_PICK:
        action["Ttcp2base"] = targets["pick_T_above"]
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.MOVE_DOWN_PICK, now, action

    elif state == PickPlaceState.MOVE_DOWN_PICK:
        action["Ttcp2base"] = targets["pick_T_down"]
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.CLOSE_GRIPPER, now, action

    elif state == PickPlaceState.CLOSE_GRIPPER:
        action["Ttcp2base"] = targets["pick_T_down"]
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.0:
            return PickPlaceState.LIFT_AFTER_PICK, now, action

    elif state == PickPlaceState.LIFT_AFTER_PICK:
        action["Ttcp2base"] = make_lift_T(targets["pick_T_above"])
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.0:
            return PickPlaceState.MOVE_ABOVE_PLACE, now, action

    elif state == PickPlaceState.MOVE_ABOVE_PLACE:
        action["Ttcp2base"] = targets["place_T_above"]
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.5:
            return PickPlaceState.MOVE_DOWN_PLACE, now, action

    elif state == PickPlaceState.MOVE_DOWN_PLACE:
        action["Ttcp2base"] = targets["place_T_down"]
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.0:
            return PickPlaceState.OPEN_GRIPPER, now, action

    elif state == PickPlaceState.OPEN_GRIPPER:
        action["Ttcp2base"] = targets["place_T_down"]
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.LIFT_AFTER_PLACE, now, action

    elif state == PickPlaceState.LIFT_AFTER_PLACE:
        action["Ttcp2base"] = make_lift_T(targets["place_T_above"])
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.RETURN_HOME, now, action

    elif state == PickPlaceState.RETURN_HOME:
        action["Ttcp2base"] = targets["home_T"]
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.5:
            return PickPlaceState.DONE, now, action

    return state, state_t0, action

# ===============================
# Task init
# ===============================
def start_task(task_idx, obs, tasks, rs_env, cam_results, home_T):
    task = tasks[task_idx]

    pick_pt = get_point_vllm(
        obs["rgb"], f"Pick the {task['pick']}", f"pick_{task_idx}.png"
    )
    place_pt = get_point_vllm(
        obs["rgb"], f"Place the {task['place']}", f"place_{task_idx}.png"
    )

    h = get_h_Parameters_for_object(task["pick"])

    targets = {
        "pick_T_above": make_target_T(
            obs, int(pick_pt[0]), int(pick_pt[1]),
            rs_env, cam_results, safe_height
        ),
        "pick_T_down": make_target_T(
            obs, int(pick_pt[0]), int(pick_pt[1]),
            rs_env, cam_results, h["pick_down"]
        ),
        "place_T_above": make_target_T(
            obs, int(place_pt[0]), int(place_pt[1]),
            rs_env, cam_results, safe_height
        ),
        "place_T_down": make_target_T(
            obs, int(place_pt[0]), int(place_pt[1]),
            rs_env, cam_results, h["place_down"]
        ),
        "home_T": home_T
    }

    return pick_pt, place_pt, targets

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    with open("data/20260115_190746/camera_results.json", "r") as f:
        cam_results = json.load(f)

    env = RealmanEnv("192.168.101.19")
    rs_env = Open3dRealsenseEnv("f1471193")

    obs = env.reset()
    obs |= rs_env.reset()

    instruction = "Pick the white can and place it in the pink plate, then pick the bottle and place it in the blue plate"

    tasks = parse_multi_pick_place_tasks(instruction)["tasks"]

    current_task_idx = 0
    state = PickPlaceState.IDLE
    state_t0 = time.time()
    auto_run = False

    home_T = copy.deepcopy(obs["Ttcp2base"])
    action = {"Ttcp2base": obs["Ttcp2base"], "gripper_open": obs["gripper_open"]}

    pick_pt = place_pt = targets = None

    print("\nPress 'a' ONCE to start all tasks\n")

    while True:
        now = time.time()

        if state != PickPlaceState.IDLE and targets is not None:
            state, state_t0, action = step_pick_place_fsm(
                state, state_t0, now, action, targets
            )

        if state == PickPlaceState.DONE:
            print(f"✅ Task {current_task_idx} finished")
            current_task_idx += 1

            if current_task_idx >= len(tasks):
                print("🎉 All tasks finished")
                break

            pick_pt, place_pt, targets = start_task(
                current_task_idx, obs, tasks, rs_env, cam_results, home_T
            )
            state = PickPlaceState.MOVE_ABOVE_PICK
            state_t0 = now

        img = obs["rgb"][:, :, ::-1].copy()
        if pick_pt is not None:
            cv2.circle(img, tuple(map(int, pick_pt)), 7, (0, 0, 255), -1)
            cv2.circle(img, tuple(map(int, place_pt)), 7, (255, 0, 0), -1)
        cv2.imshow("rgb", img)

        k = cv2.waitKey(1)
        if k == ord("a") and not auto_run:
            auto_run = True
            pick_pt, place_pt, targets = start_task(
                current_task_idx, obs, tasks, rs_env, cam_results, home_T
            )
            state = PickPlaceState.MOVE_ABOVE_PICK
            state_t0 = now
            print(f"▶ Start ALL Tasks")

        elif k == ord("q"):
            break

        obs = env.step(action)
        obs |= rs_env.step(action)

    env.close()
    rs_env.close()
