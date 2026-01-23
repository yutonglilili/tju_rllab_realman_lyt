"""
本脚本实现自动 pick & place 任务。通过修改基本参数，可以支持不同的物体和盘子。
主要流程：
1. VLLM 根据 prompt, 输出 pick_obj 和 place_obj;
2. 根据 pick_obj 和 place_obj, 再调用两次 VLLM, 分别获取 pick_point 和 place_point;
3. 由于考虑安全高度，需要分别计算 pick_T_above、pick_T_down、place_T_above、place_T_down;
4. 根据 pick_T_above、pick_T_down、place_T_above、place_T_down, 计算 action;
5. 执行 action, 完成 pick and place 任务。
每次执行前需要 check 的基本参数包括：
- PICK_ABOVE_HEIGHT
- PICK_DOWN_HEIGHT
- PLACE_ABOVE_HEIGHT
- PLACE_DOWN_HEIGHT
- prompt
操作步骤：
1. 运行脚本；
2. 按 'a' 键启动自动序列；
3. 按 'q' 键退出。
"""

from tvla_realenv.open3d_realsense_env import Open3dRealsenseEnv
from tvla_realenv.realman_env import RealmanEnv
import cv2
import copy
import json
import numpy as np
from enum import Enum, auto
import time
from pointing_vllm_get_point_utils import (
    parse_pick_place_objects,
    get_point_vllm,
)

# ========== 基本参数 ==========
"""
# white can:
PICK_ABOVE_HEIGHT = 0.04
PICK_DOWN_HEIGHT = -0.02
PLACE_ABOVE_HEIGHT = 0.06
PLACE_DOWN_HEIGHT = 0.04

# blue stir stick:
PICK_ABOVE_HEIGHT = 0.03
PICK_DOWN_HEIGHT = 0.02
PLACE_ABOVE_HEIGHT = 0.06
PLACE_DOWN_HEIGHT = 0.04

# yellow ball:
PICK_ABOVE_HEIGHT = 0.04
PICK_DOWN_HEIGHT = -0.06
PLACE_ABOVE_HEIGHT = 0.04
PLACE_DOWN_HEIGHT = 0.03
"""
# pink plate:
PICK_ABOVE_HEIGHT = 0.04
PICK_DOWN_HEIGHT = -0.01
PLACE_ABOVE_HEIGHT = 0.10
PLACE_DOWN_HEIGHT = 0.04
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

def calculate_3d_position(u, v, obs, rs_env, cam_results, z_offset=0.002):
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

# 抓取后抬升 5cm
def make_lift_T(T, lift_height=0.02):
    T_lift = copy.deepcopy(T)
    T_lift[2,3] += lift_height
    return T_lift

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

    import sys; sys.path.append("examples")

    instruction = "You need to pick the edge of the pink plate and place it in the whtie bucket"

    objects = parse_pick_place_objects(instruction)
    pick_obj = objects["pick"]
    place_obj = objects["place"]

    print("Pick object:", pick_obj, "Place object:", place_obj)

    pick_point = get_point_vllm(obs["rgb"],f"Pick the {pick_obj}",save_path="debug_pick.png",color=(0, 0, 255),)
    place_point = get_point_vllm(obs["rgb"],f"Place the {place_obj}",save_path="debug_place.png",color=(255, 0, 0),)

    # -------- Targets --------
    pick_T_above = make_target_T(obs, int(pick_point[0]), int(pick_point[1]), rs_env, cam_results, z_offset=0.04)  # 上方0.04
    pick_T_down  = make_target_T(obs, int(pick_point[0]), int(pick_point[1]), rs_env, cam_results, z_offset=-0.02)   # 下降到物体
    place_T_above = make_target_T(obs, int(place_point[0]), int(place_point[1]), rs_env, cam_results, z_offset=0.06)  # 上方0.04
    place_T_down  = make_target_T(obs, int(place_point[0]), int(place_point[1]), rs_env, cam_results, z_offset=0.04)   # 下降到盘子
    home_T = copy.deepcopy(obs["Ttcp2base"])

    # -------- State machine --------
    state = PickPlaceState.IDLE
    state_t0 = time.time()

    GRIPPER_OPEN = 0.09
    GRIPPER_CLOSE = 0.00

    action = {
        "Ttcp2base": obs["Ttcp2base"],
        "gripper_open": obs["gripper_open"],
    }

    print("\nPress 'a' to start pick & place\n")

    while True:
        now = time.time()

        if state == PickPlaceState.IDLE:
            pass

        elif state == PickPlaceState.MOVE_ABOVE_PICK:
            action["Ttcp2base"] = pick_T_above
            action["gripper_open"] = GRIPPER_OPEN
            if now - state_t0 > 1.0:
                state = PickPlaceState.MOVE_DOWN_PICK
                state_t0 = now
                print("→ MOVE_DOWN_PICK")


        elif state == PickPlaceState.MOVE_DOWN_PICK:
            action["Ttcp2base"] = pick_T_down
            action["gripper_open"] = GRIPPER_OPEN
            if now - state_t0 > 1.0:
                state = PickPlaceState.CLOSE_GRIPPER
                state_t0 = now
                print("→ CLOSE_GRIPPER")

        elif state == PickPlaceState.CLOSE_GRIPPER:
            action["Ttcp2base"] = pick_T_down
            action["gripper_open"] = GRIPPER_CLOSE
            if now - state_t0 > 1.0:
                state = PickPlaceState.LIFT_AFTER_PICK
                state_t0 = now
                print("→ LIFT_AFTER_PICK")

        elif state == PickPlaceState.LIFT_AFTER_PICK:
            action["Ttcp2base"] = make_lift_T(pick_T_down, lift_height=0.1)
            action["gripper_open"] = GRIPPER_CLOSE
            if now - state_t0 > 1.0:
                state = PickPlaceState.MOVE_ABOVE_PLACE
                state_t0 = now
                print("→ MOVE_ABOVE_PLACE")

        elif state == PickPlaceState.MOVE_ABOVE_PLACE:
            action["Ttcp2base"] = place_T_above
            action["gripper_open"] = GRIPPER_CLOSE
            if now - state_t0 > 2.0:
                state = PickPlaceState.MOVE_DOWN_PLACE
                state_t0 = now
                print("→ MOVE_DOWN_PLACE")

        elif state == PickPlaceState.MOVE_DOWN_PLACE:
            action["Ttcp2base"] = place_T_down
            action["gripper_open"] = GRIPPER_CLOSE
            if now - state_t0 > 1.0:
                state = PickPlaceState.OPEN_GRIPPER
                state_t0 = now
                print("→ OPEN_GRIPPER")

        elif state == PickPlaceState.OPEN_GRIPPER:
            action["Ttcp2base"] = place_T_down
            action["gripper_open"] = GRIPPER_OPEN
            if now - state_t0 > 1.0:
                state = PickPlaceState.LIFT_AFTER_PLACE
                state_t0 = now
                print("→ LIFT_AFTER_PLACE")

        elif state == PickPlaceState.LIFT_AFTER_PLACE:
            action["Ttcp2base"] = make_lift_T(place_T_down, lift_height=0.1)
            action["gripper_open"] = GRIPPER_OPEN
            if now - state_t0 > 1.0:
                state = PickPlaceState.RETURN_HOME
                state_t0 = now
                print("→ RETURN_HOME")

        elif state == PickPlaceState.RETURN_HOME:
            action["Ttcp2base"] = home_T
            action["gripper_open"] = GRIPPER_OPEN
            if now - state_t0 > 2.0:
                state = PickPlaceState.DONE
                print("→ DONE")

        elif state == PickPlaceState.DONE:
            print("Pick & Place finished")
            state = PickPlaceState.IDLE

        # ===============================
        # Step env (ONLY ONE PLACE)
        # ===============================
        obs = env.step(action)
        obs |= rs_env.step(action)

        # ===============================
        # Visualization
        # ===============================
        img = obs["rgb"][:, :, ::-1].copy()
        cv2.circle(img, (int(pick_point[0]), int(pick_point[1])), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(place_point[0]), int(place_point[1])), 5, (255, 0, 0), -1)
        cv2.imshow("color", img)

        k = cv2.waitKey(1)
        if k == ord("a") and state == PickPlaceState.IDLE:
            print("→ MOVE_TO_PICK")
            state = PickPlaceState.MOVE_ABOVE_PICK
            state_t0 = time.time()
        elif k == ord("q"):
            break

    env.close()
    rs_env.close()
