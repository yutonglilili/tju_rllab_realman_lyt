"""
开发中...
目标：
1. 打通连续的 pick & place 的流程：已完成
2. 取消一次任务后回到初始位置的问题：已完成
2. eef 轨迹点可视化，直接将过程中设定好的点显示出来即可
3. 邻域内深度检测，选取最优 pick 点，当前认为 depth 最小的就是最优 pick 点
4. 接入 grasp_net,moveit
5. 验证action是阻塞式还是非阻塞式
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
# 定义每个物体的 pick & place 高度
safe_height = 0.06
h_Parameters_list = {
    "yellow ball": {"pick_down": -0.06, "place_down": 0.06},
    #"white can": {"pick_down": -0.02, "place_down": 0.06},
    "white can": {"pick_down": 0.00, "place_down": 0.00},
    "rubik's cube": {"pick_down": 0.01, "place_down": 0.07},
    "glue stick": {"pick_down": -0.02, "place_down": 0.07},
    # 可以继续添加其他物体
}

def get_h_Parameters_for_object(obj_name):
    return h_Parameters_list.get(obj_name, {"pick_down": 0.00, "place_down": 0.04})  # 新物体取默认值

GRIPPER_OPEN = 0.09
GRIPPER_CLOSE = 0.03

# ===============================
# State definition
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
    xyz_cam = np.linalg.inv(np.array(rs_env.meta_obs["intrinsic"])) @ (
        np.array([u, v, 1.0]) * d
    )
    xyz_base = np.array(cam_results["Tcam2base"]) @ np.array(
        [xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0]
    )
    xyz_base[2] += z_offset
    return xyz_base[:3]

def make_target_T(obs, u, v, rs_env, cam_results, z_offset=0.02):
    T = copy.deepcopy(obs["Ttcp2base"])
    T[:3, 3] = calculate_3d_position(u, v, obs, rs_env, cam_results, z_offset)
    return T

def make_lift_T(T, lift_height=0.02):
    T_lift = copy.deepcopy(T)
    T_lift[2, 3] += lift_height
    return T_lift

# ===============================
# FSM step function
# ===============================
def step_pick_place_fsm(state, state_t0, now, action, targets, is_last_task=False):
    if state == PickPlaceState.IDLE:
        return state, state_t0, action
    elif state == PickPlaceState.MOVE_ABOVE_PICK:
        action["Ttcp2base"] = targets["pick_T_above"]
        print(f"MOVE_ABOVE_PICK: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.MOVE_DOWN_PICK, now, action
    elif state == PickPlaceState.MOVE_DOWN_PICK:
        action["Ttcp2base"] = targets["pick_T_down"]
        print(f"MOVE_DOWN_PICK: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.CLOSE_GRIPPER, now, action
    elif state == PickPlaceState.CLOSE_GRIPPER:
        action["Ttcp2base"] = targets["pick_T_down"]
        print(f"CLOSE_GRIPPER: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.0:
            return PickPlaceState.LIFT_AFTER_PICK, now, action
    elif state == PickPlaceState.LIFT_AFTER_PICK:
        action["Ttcp2base"] = make_lift_T(targets["pick_T_down"])
        print(f"LIFT_AFTER_PICK: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.0:
            return PickPlaceState.MOVE_ABOVE_PLACE, now, action
    elif state == PickPlaceState.MOVE_ABOVE_PLACE:
        action["Ttcp2base"] = targets["place_T_above"]
        print(f"MOVE_ABOVE_PLACE: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 2.0:
            return PickPlaceState.MOVE_DOWN_PLACE, now, action
    elif state == PickPlaceState.MOVE_DOWN_PLACE:
        action["Ttcp2base"] = targets["place_T_down"]
        print(f"MOVE_DOWN_PLACE: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_CLOSE
        if now - state_t0 > 1.0:
            return PickPlaceState.OPEN_GRIPPER, now, action
    elif state == PickPlaceState.OPEN_GRIPPER:
        action["Ttcp2base"] = targets["place_T_down"]
        print(f"OPEN_GRIPPER: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            return PickPlaceState.LIFT_AFTER_PLACE, now, action
    elif state == PickPlaceState.LIFT_AFTER_PLACE:
        action["Ttcp2base"] = make_lift_T(targets["place_T_down"])
        print(f"LIFT_AFTER_PLACE: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 1.0:
            # 只有最后一个任务才返回 home，否则直接完成当前任务
            if is_last_task:
                return PickPlaceState.RETURN_HOME, now, action
            else:
                return PickPlaceState.DONE, now, action
    elif state == PickPlaceState.RETURN_HOME:
        action["Ttcp2base"] = targets["home_T"]
        print(f"RETURN_HOME: {action['Ttcp2base']}")
        action["gripper_open"] = GRIPPER_OPEN
        if now - state_t0 > 2.0:
            return PickPlaceState.DONE, now, action
    return state, state_t0, action

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    # --- Load camera calibration ---
    with open("data/20260124_002604/camera_results.json", "r") as f:
        cam_results = json.load(f)

    # left arm
    env = RealmanEnv("192.168.101.19")
    # right arm
    # env_right = RealmanEnv("192.168.101.20")
    rs_env = Open3dRealsenseEnv("f1471338")

    obs = env.reset()
    obs |= rs_env.reset()

    # --- Tasks ---
    instruction = "Pick the white can and place it in the pink plate, then pick the yellow ball and place it in the blue plate"
    task_plan = parse_multi_pick_place_tasks(instruction)
    tasks = task_plan["tasks"]
    print(f"Total tasks: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"[Task {i}] pick={t['pick']} place={t['place']}")

    # --- FSM initialization ---
    current_task_idx = 0
    state = PickPlaceState.IDLE
    state_t0 = time.time()
    home_T = copy.deepcopy(obs["Ttcp2base"])
    action = {"Ttcp2base": obs["Ttcp2base"], "gripper_open": obs["gripper_open"]}

    pick_pt = None
    place_pt = None
    targets = None

    # --- Precompute first task points ---
    if current_task_idx < len(tasks):
        task = tasks[current_task_idx]
        pick_pt = get_point_vllm(obs["rgb"], f"Pick the {task['pick']}", f"pick_{current_task_idx}.png")
        place_pt = get_point_vllm(obs["rgb"], f"Place onto the {task['place']}", f"place_{current_task_idx}.png")
        print(f"obj:{task['pick']},{make_target_T(obs, int(pick_pt[0]), int(pick_pt[1]), rs_env, cam_results, 0)}")

    print("\nPress 'a' to start each task\n")

    while True:
        now = time.time()

        # FSM step
        if state != PickPlaceState.IDLE and targets is not None:
            is_last_task = (current_task_idx == len(tasks) - 1)
            state, state_t0, action = step_pick_place_fsm(state, state_t0, now, action, targets, is_last_task)

        # Task done → move to next
        if state == PickPlaceState.DONE:
            print(f"✅ Task {current_task_idx} finished")
            current_task_idx += 1
            state = PickPlaceState.IDLE
            targets = None

            if current_task_idx >= len(tasks):
                print("🎉 All tasks finished")
                break

            # 更新观测，以便下一个任务能基于当前位置继续
            obs = env.step(action)
            obs |= rs_env.step(action)

            # Precompute next task points
            task = tasks[current_task_idx]
            h_params = get_h_Parameters_for_object(task["pick"])
            
            targets = {
                "pick_T_down": make_target_T(obs, int(pick_pt[0]), int(pick_pt[1]), rs_env, cam_results, h_params["pick_down"]),
                "place_T_down": make_target_T(obs, int(place_pt[0]), int(place_pt[1]), rs_env, cam_results, h_params["place_down"]),
                "home_T": home_T
            }

            pick_pt = get_point_vllm(obs["rgb"], f"Pick the {task['pick']}", f"pick_{current_task_idx}.png")
            place_pt = get_point_vllm(obs["rgb"], f"Place the {task['place']}", f"place_{current_task_idx}.png")

        # Visualization
        img = obs["rgb"][:, :, ::-1].copy()
        if pick_pt is not None and place_pt is not None:
            cv2.circle(img, (int(pick_pt[0]), int(pick_pt[1])), 7, (0, 0, 255), -1)
            cv2.circle(img, (int(place_pt[0]), int(place_pt[1])), 7, (255, 0, 0), -1)
        cv2.imshow("rgb", img)

        # Key handling
        k = cv2.waitKey(1)
        if k == ord("a") and state == PickPlaceState.IDLE:
            # Build targets for FSM
            task = tasks[current_task_idx]
            h_Parameters = get_h_Parameters_for_object(task["pick"])
            targets = {
                "pick_T_above": make_target_T(obs, int(pick_pt[0]), int(pick_pt[1]), rs_env, cam_results, safe_height),
                "pick_T_down": make_target_T(obs, int(pick_pt[0]), int(pick_pt[1]), rs_env, cam_results, h_Parameters["pick_down"]),
                "place_T_above": make_target_T(obs, int(place_pt[0]), int(place_pt[1]), rs_env, cam_results, safe_height),
                "place_T_down": make_target_T(obs, int(place_pt[0]), int(place_pt[1]), rs_env, cam_results, h_Parameters["place_down"]),
                "home_T": home_T
            }
            state = PickPlaceState.MOVE_ABOVE_PICK
            state_t0 = now
            print(f"▶ Start Task {current_task_idx}")

        elif k == ord("q"):
            break

        # Step env
        obs = env.step(action)
        obs |= rs_env.step(action)

    env.close()
    rs_env.close()
 
 
