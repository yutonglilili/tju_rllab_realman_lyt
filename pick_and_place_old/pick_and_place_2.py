"""
20260309 23:50更新 2.0版本
目前已实现：
1. 接入 graspnet 和 moveit，通过开关控制是否调用。
2. 实现自动化检测抓取和放置结果，通过开关控制是否跳过检测。(也可替换为人工检测模式，通过键入 y / n 的方式，实现对抓取和放置结果的确认)。
3. 实现 pick 和 place 失败后重推理并执行的机制。
4. 添加 vlm 打点不理想和 graspnet 提供位姿不理想后重试的机制，此判断需人工审核。
5. 使用 joint 控制机械臂 go home 。当前版本已支持传递 pose 和 joint 两种动作信息，通过 action["control_mode"] 控制。
主要计划：
1. 在不使用 graspnet 的时候，根据 xyz 的值，选择几种预设定的不同的基础位姿，以提高执行成功率；
2. 优化 moveit server 脚本，改为一次性规划好全部动作，全部规划成功再执行，而不是执行到一半退掉；
3. 优化 main 函数，当前在 moveit server 给出规划失败后会直接退出程序，改为重试机制，如果规划失败，则重新 prepare。如果使用预设姿态，可以在尝试 n 次仍失败后依次选择其他预设姿态，如果所有预设姿态都失败，才会退出。（优先级靠后）
"""
import copy
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum, auto

import cv2
import numpy as np
import requests
from pytransform3d.rotations import active_matrix_from_angle
from pytransform3d.transformations import transform_from
from tvla_realenv.open3d_realsense_env import Open3dRealsenseEnv
from multi_pointing_vllm_get_point_utils import * 
import multi_pointing_vllm_get_point_utils as mp_utils


# 路径设置（FIXME：导入的逻辑问题）
sys.path.append("/home/zhangzhao/tvla-realenv/examples")
from pointing_vllm_get_point_utils import * 
import pointing_vllm_get_point_utils as p_utils

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(_project_root, 'ruihao'))
sys.path.insert(0, _project_root)  # 加上项目根目录，这样 examples.xxx 才能被解析
from graspnet_moveit import *

# 从工具函数脚本导入功能函数
from pick_and_place_utils import *


# ===============================
# 参数和开关
# ===============================

# 全自动模式开关(开启后，不进行人工干预，直接执行)
AUTO_MODE = True

# 安全高度
SAFE_HEIGHT = 0.06

# 夹爪开合程度
GRIPPER_OPEN = 0.09
GRIPPER_CLOSE = 0.03

# 检测开关（模式 1：自动化检测，模式 2：跳过检测，模式 3：人工检测）
CHECK_PICK_SUCCESS = 2
CHECK_PLACE_SUCCESS = 2

# 使用 graspnet 开关
USE_GRASPNET = False

# 控制参数
BATCH_USE_MOVEIT = True     # moveit 总开关，打开后还可单独控制每次动作是否是使用 moveit
BATCH_SPEED_PCT = 25        # RealMan 机械臂速度百分比
BATCH_BLEND_R_PCT = 20       # RealMan 机械臂混合半径百分比

# 保存图像路径
SAVE_DIR = "/home/zhangzhao/tvla-realenv/lyt/pick_and_place_2.0/save_images/"

# 左右臂的控制URL
WEB_CONTROL_URL = {
    'left_arm_control_url': "http://192.168.101.68:8888/control/left",
    'right_arm_control_url': "http://192.168.101.68:8888/control/right"
    }
WEB_BATCH_URL = {
    'left_arm_batch_control_url': "http://192.168.101.68:8888/control/left/batch",
    'right_arm_batch_control_url': "http://192.168.101.68:8888/control/right/batch"
}


# ===============================
# Pick Place FSM Main Controller
# ===============================
class PickPlaceFSMState(Enum):

    PLAN_PICK = auto()
    EXEC_PICK = auto()
    CHECK_PICK = auto()

    PLAN_PLACE = auto()
    EXEC_PLACE = auto()
    CHECK_PLACE = auto()

    TASK_DONE = auto()


# ===============================
# CHECK FUNCTIONS
# ===============================

def check_pick_success(obs, obs_rs_dual, rs_env, cam_results, pick_name, arm_name):
    
    if CHECK_PICK_SUCCESS == 1:
        
        print("Checking if the capture was successful...")
        
        # 更新观测
        obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
        image_rgb = obs[f"{arm_name}_rs_obs"]["rgb"]

        # 保存图像
        save_check_image(image_rgb, prefix="pick_check", object_name=pick_name, save_dir=SAVE_DIR)

        is_success = check_grasp_success_vllm(image_rgb, pick_name)
        
        if is_success:
            print("✅ Grasp success detected.")
            return True
        else:
            print("❌ Grasp failed.")
            return False
    
    elif CHECK_PICK_SUCCESS == 2:
        print("Skip to check if capture was successful.")
        return True
    
    else:
        print("Manual inspection to determine if capture was successful.")
        
        while True:

            key = input("Enter y / n : ").strip()

            if key == 'y':
                print("Pick success")
                return True

            elif key == 'n':
                print("Pick failed")
                return False

            else:
                print("Invalid input. Please enter y / n.")
                continue

def check_place_success(obs, obs_rs_dual, rs_env, cam_results, pick_name, place_name, arm_name):
    
    if CHECK_PLACE_SUCCESS == 1:
        
        print("Checking if the place was successful...")
        
        # 更新观测
        obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
        image_rgb = obs[f"{arm_name}_rs_obs"]["rgb"]

        # 保存图像
        save_check_image(image_rgb, prefix="place_check", object_name=pick_name, container_name=place_name, save_dir=SAVE_DIR)

        is_success = check_place_success_vllm(image_rgb, pick_name, place_name)
        
        if is_success:
            print("✅ Place success detected.")
            return True
        
        else:
            print("❌ Place failed.")
            return False
    
    elif CHECK_PLACE_SUCCESS == 2:
        
        print("Skip to check if place was successful.")
        return True
    
    else:
        print("Manual inspection to determine if place was successful.")
        while True:

            key = input("Enter y / n : ").strip()

            if key == 'y':
                print("Place success")
                return True

            elif key == 'n':
                print("Place failed")
                return False

            else:
                print("Invalid input. Please enter y / n.")
                continue



# ===============================
# PICK PREPARE（后续在此处做可视化）
# ===============================

def prepare_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None):

    print("🧠 PLAN PICK")
    
    pick_pt = None
    obs_rs = None

    while True:
        # ===============================
        # Step1: VLM 打点
        # ===============================
        if pick_pt is None:

            # 更新相机观测
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)

            print("🔍 Calling VLM for pick points...")

            pick_pt_raw = mp_utils.get_point_vllm(
                obs[f"{arm_name}_rs_obs"]["rgb"],
                f"Point the {task['pick']}"
            )

            pick_pt = pick_pt_raw[0]["point_2d"] if isinstance(pick_pt_raw, list) else pick_pt_raw

            print("pick_pt:", pick_pt)
            
            if AUTO_MODE==False:
                visualize_rgb_with_point(
                    obs[f"{arm_name}_rs_obs"]["rgb"],
                    pick_pt,
                    "Pick Point"
                )

        # ===============================
        # Step2: 计算抓取位姿
        # ===============================
        if USE_GRASPNET:

            print("🧠 Running GraspNet...")

            grasp_out_dir = "/home/zhangzhao/tvla-realenv/lyt/graspnet_debug"

            pick_Ttcp2base, pre_pick_Ttcpbase = do_pipeline_final(
                object_name=task['pick'],
                grasp_out_dir=grasp_out_dir,
                obs=obs[f"{arm_name}_rs_obs"],
                pick_pt=pick_pt,
                Ttcp2base=home_T_tcp2base,
                Tcam2base=np.array(cam_results["Tcam2base"]),
            )

            # 修正相机的标定问题，将 x 轴向左偏移 1cm
            pick_Ttcp2base = make_lift_T(pick_Ttcp2base, lift_x=0.01)

            pre_pick_Ttcpbase = make_lift_T(pick_Ttcp2base, lift_z=SAFE_HEIGHT, lift_y=0.03)      # 覆盖掉graspnet的预抓取位姿
            post_pick_Ttcp2base = make_lift_T(pick_Ttcp2base, lift_z=SAFE_HEIGHT, lift_y=0.03)

            targets = {
                "pick_Ttcp2base": pick_Ttcp2base,
                "pre_pick_Ttcp2base": pre_pick_Ttcpbase,
                "post_pick_Ttcp2base": post_pick_Ttcp2base,
            }
        
        else:
            print("⚙ Using heuristic grasp pose")
            rx_degree_close = 10
            rx_degree_far_high = 45
            rx_degree_far_low = 30

            # 先获取物体的xyz坐标
            T_base_rotated = copy.deepcopy(home_T_tcp2base)

            pick_Ttcp2base = make_target_T(obs[f"{arm_name}_rs_obs"], int(pick_pt[0]), int(pick_pt[1]), rs_env, cam_results, T_base_rotated)

            x, y, z = pick_Ttcp2base[:3, 3]
            print("x, y, z:", x, y, z)

            # 根据 y 的值选择适当的 rx_degree
            if y > -0.35:
                rx_degree = rx_degree_close
                print("Using rx_degree_close:", rx_degree)
            elif z > 0.12:
                rx_degree = rx_degree_far_high
                print("Using rx_degree_far_high:", rx_degree)
            else:
                rx_degree = rx_degree_far_low
                print("Using rx_degree_far_low:", rx_degree)

            rx = -1 * (rx_degree / 180) * np.pi

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])

            T_base_rotated = copy.deepcopy(home_T_tcp2base)
            T_base_rotated[:3, :3] = Rx @ home_T_tcp2base[:3, :3]

            pick_Ttcp2base = make_target_T(
                obs[f"{arm_name}_rs_obs"],
                int(pick_pt[0]),
                int(pick_pt[1]),
                rs_env,
                cam_results,
                T_base_rotated
            )

            # 修正相机的标定问题，将 x 轴向左偏移 1cm
            pick_Ttcp2base = make_lift_T(pick_Ttcp2base, lift_x=0.01)

            pre_pick_Ttcpbase = make_lift_T(pick_Ttcp2base, lift_z=SAFE_HEIGHT, lift_y=0.03)
            post_pick_Ttcp2base = make_lift_T(pick_Ttcp2base, lift_z=SAFE_HEIGHT, lift_y=0.03)

            targets = {
                "pre_pick_Ttcp2base": pre_pick_Ttcpbase,
                "pick_Ttcp2base": pick_Ttcp2base,
                "post_pick_Ttcp2base": post_pick_Ttcp2base,
            }
        
        # 全自动模式下，跳过打点和 graspnet 检测
        # if AUTO_MODE == True:
        return targets

        # ===============================
        # Step3: 等待用户按键
        # ===============================
        print("\n============================")
        print("Press key:")
        print("1 -> Accept result and continue")
        print("2 -> Re-run VLM + GraspNet")
        print("3 -> Re-run GraspNet only")
        print("4 -> Exit")
        print("============================")

        while True:

            key = input("Enter 1 / 2 / 3 / 4 : ").strip()

            if key == '1':
                print("✅ Accept current result")
                return targets

            elif key == '2':
                print("🔄 Re-running VLM + GraspNet")
                pick_pt = None
                break

            elif key == '3':
                print("🔄 Re-running GraspNet only")
                break

            elif key == '4':
                print("exiting...")
                exit()

            else:
                print("Invalid input. Please enter 1 / 2 / 3 / 4.")


# ===============================
# PLACE PREPARE（后续在此处做可视化）
# ===============================

def prepare_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, pick_targets, arm_name):

    print("🧠 PLAN PLACE")

    place_pt = None
    obs_rs = None

    while True:

        # ===============================
        # Step1: VLM 打点
        # ===============================
        if place_pt is None:

            # 更新相机观测
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)

            print("🔍 Calling VLM for place points...")

            place_pt_raw = mp_utils.get_point_vllm(
                obs[f"{arm_name}_rs_obs"]["rgb"],
                f"Point the {task['place']}"
            )

            place_pt = place_pt_raw[0]["point_2d"] if isinstance(place_pt_raw, list) else place_pt_raw

            print("place_pt:", place_pt)

            if AUTO_MODE==False:
                visualize_rgb_with_point(
                    obs[f"{arm_name}_rs_obs"]["rgb"],
                    place_pt,
                    "Place Point"
                )


        # ===============================
        # Step2: 计算放置位姿
        # ===============================
        place_point_world = make_target_T(
                obs[f"{arm_name}_rs_obs"],
                int(place_pt[0]),
                int(place_pt[1]),
                rs_env,
                cam_results,
                home_T_tcp2base,
                z_offset=0.0
            )
        
        # 从之前的 pick 的 target 中获取 pick_Ttcp2base 和 post_pick_Ttcp2base
        pick_Ttcp2base = copy.deepcopy(pick_targets["pick_Ttcp2base"])
        post_pick_Ttcp2base = copy.deepcopy(pick_targets["post_pick_Ttcp2base"])
        
        place_Ttcp2base = copy.deepcopy(pick_Ttcp2base)
        place_Ttcp2base[:3, 3] = place_point_world[:3, 3]


        # 修正相机的标定问题，将 x 轴向左偏移 1cm
        place_Ttcp2base = make_lift_T(place_Ttcp2base, lift_x=0.01)

        place_Ttcp2base = make_lift_T(place_Ttcp2base, lift_z=0.13)
        pre_place_Ttcp2base = make_lift_T(place_Ttcp2base, lift_z=SAFE_HEIGHT)
        post_place_Ttcp2base = make_lift_T(place_Ttcp2base, lift_z=SAFE_HEIGHT)

        xyzrpy_post_pick = RealmanEnvWebSim.realman_xyzrpy_from_T(post_pick_Ttcp2base)
        xyzrpy_pre_place = RealmanEnvWebSim.realman_xyzrpy_from_T(pre_place_Ttcp2base)

        xyzrpy_intermediate = np.zeros(6)
        xyzrpy_intermediate[:3] = 0.5 * (xyzrpy_post_pick[:3] + xyzrpy_pre_place[:3])
        xyzrpy_intermediate[3:] = xyzrpy_post_pick[3:]

        intermediate_Ttcp2base = RealmanEnvWebSim.T_from_realman_xyzrpy(xyzrpy_intermediate)

        # 将过渡点抬高 8cm
        intermediate_Ttcp2base = make_lift_T(intermediate_Ttcp2base, lift_z=0.03)

        reset_Ttcp2base = copy.deepcopy(home_T_tcp2base)

        targets = {
            "intermediate_Ttcp2base": intermediate_Ttcp2base,
            "pre_place_Ttcp2base": pre_place_Ttcp2base,
            "place_Ttcp2base": place_Ttcp2base,
            "post_place_Ttcp2base": post_place_Ttcp2base,
            "reset_Ttcp2base": reset_Ttcp2base,
        }

        # 全自动模式下，跳过打点和 graspnet 检测
        # if AUTO_MODE == True:
        return targets

        # ===============================
        # Step3: 等待用户按键
        # ===============================
        print("\n============================")
        print("Press key:")
        print("1 -> Accept result and continue")
        print("2 -> Re-run VLM")
        print("4 -> Exit")
        print("============================")

        while True:

            key = input("Enter 1 / 2 / 4 : ").strip()

            if key == '1':
                print("✅ Accept current result")
                return targets

            elif key == '2':
                print("🔄 Re-running VLM")
                place_pt = None
                break

            elif key == '4':
                print("exiting...")
                exit()

            else:
                print("Invalid input. Please enter 1 / 2 / 4.")
    

# ===============================
# BUILD SEQUENCE
# ===============================

def build_pick_sequence(targets):

    seq = []
    names = []

    seq.append({
        "Ttcp2base": targets["pre_pick_Ttcp2base"],
        "gripper_open": GRIPPER_OPEN,
        "control_mode": "pose",
        "use_moveit": True
    })
    names.append("MOVE_PRE_PICK")

    seq.append({
        "Ttcp2base": targets["pick_Ttcp2base"],
        "gripper_open": GRIPPER_CLOSE,
        "control_mode": "pose",
        "use_moveit": False
    })
    names.append("MOVE_PICK")

    seq.append({
        "Ttcp2base": targets["post_pick_Ttcp2base"],
        "gripper_open": GRIPPER_CLOSE,
        "control_mode": "pose",
        "use_moveit": False
    })
    names.append("MOVE_POST_PICK")

    return seq, names

def build_place_sequence(targets):

    seq = []
    names = []

    seq.append({
        "Ttcp2base": targets["intermediate_Ttcp2base"],
        "gripper_open": GRIPPER_CLOSE,
        "control_mode": "pose",
        "use_moveit": False
    })
    names.append("MOVE_INTERMEDIATE")

    seq.append({
        "Ttcp2base": targets["pre_place_Ttcp2base"],
        "gripper_open": GRIPPER_CLOSE,
        "control_mode": "pose",
        "use_moveit": False
    })
    names.append("MOVE_PRE_PLACE")

    seq.append({
        "Ttcp2base": targets["place_Ttcp2base"],
        "gripper_open": GRIPPER_OPEN,
        "control_mode": "pose",
        "use_moveit": False
    })
    names.append("MOVE_PLACE")

    seq.append({
        "Ttcp2base": targets["post_place_Ttcp2base"],
        "gripper_open": GRIPPER_OPEN,
        "control_mode": "pose",
        "use_moveit": False
    })
    names.append("MOVE_POST_PLACE")

    seq.append({
        # "Ttcp2base": targets["reset_Ttcp2base"],
        "joint": [90, 0, 0, -90, 0, -90, 60],
        "gripper_open": GRIPPER_OPEN,
        "control_mode": "joint",    # 使用关节模式，因为 go home 需要保证关节角度不变
        "use_moveit": False
    })
    names.append("MOVE_HOME")

    return seq, names


# 更新观测
def update_obs(obs, obs_rs_dual, rs_env, arm_name):

    obs_rs = rs_env.step(obs)
    if arm_name == "left_arm":
        obs_rs_dual['left_arm_rs_obs'].update(obs_rs)
    else:
        obs_rs_dual['right_arm_rs_obs'].update(obs_rs)
    if obs_rs:
        obs.update(obs_rs_dual)
    
    return obs_rs


def generate_fallback_place_pose(place_targets):

    # 备用放置姿态（fallback）
    FALLBACK_PLACE_RX_DEG = 10

    print("⚠ Using fallback place pose")

    place_T = copy.deepcopy(place_targets["place_Ttcp2base"])

    rx = np.deg2rad(FALLBACK_PLACE_RX_DEG)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    place_T[:3, :3] = Rx @ place_T[:3, :3]

    pre_place = make_lift_T(place_T, lift_z=SAFE_HEIGHT)
    post_place = make_lift_T(place_T, lift_z=SAFE_HEIGHT)

    place_targets["place_Ttcp2base"] = place_T
    place_targets["pre_place_Ttcp2base"] = pre_place
    place_targets["post_place_Ttcp2base"] = post_place

    return place_targets


# ===============================
# MAIN FSM (完整执行一个pick-place任务)
# ===============================

def run_fsm(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None):

    state = PickPlaceFSMState.PLAN_PICK

    pick_targets = {}
    place_targets = {}

    while True:

        # -----------------

        if state == PickPlaceFSMState.PLAN_PICK:

            pick_targets = prepare_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name)

            state = PickPlaceFSMState.EXEC_PICK

        elif state == PickPlaceFSMState.EXEC_PICK:

            # 构建抓取动作序列
            action_seq, action_names = build_pick_sequence(pick_targets)

            # 发送抓取动作序列到web
            status, plan_success, failed_wp, T_res, G_res = send_action_sequence_to_web(action_seq, action_names, arm_name, WEB_BATCH_URL, BATCH_USE_MOVEIT, BATCH_SPEED_PCT, BATCH_BLEND_R_PCT, GRIPPER_OPEN, GRIPPER_CLOSE)

            # 更新反馈
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
            obs["Ttcp2base"] = T_res
            obs["gripper_open"] = G_res

            # 更新状态
            state = PickPlaceFSMState.CHECK_PICK

        # -----------------

        elif state == PickPlaceFSMState.CHECK_PICK:

            success = check_pick_success(obs, obs_rs_dual, rs_env, cam_results, task['pick'], arm_name)

            if success:

                state = PickPlaceFSMState.PLAN_PLACE

            else:

                print("Pick failed -> retry")

                state = PickPlaceFSMState.PLAN_PICK

        # -----------------

        elif state == PickPlaceFSMState.PLAN_PLACE:

            place_targets = prepare_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, pick_targets, arm_name)

            state = PickPlaceFSMState.EXEC_PLACE

        # -----------------

        elif state == PickPlaceFSMState.EXEC_PLACE:

            # 构建放置动作序列
            action_seq, action_names = build_place_sequence(place_targets)

            # 发送放置动作序列到web
            status, plan_success, failed_wp, T_res, G_res = send_action_sequence_to_web(action_seq, action_names, arm_name, WEB_BATCH_URL, BATCH_USE_MOVEIT, BATCH_SPEED_PCT, BATCH_BLEND_R_PCT, GRIPPER_OPEN, GRIPPER_CLOSE)
            
            # 更新反馈
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
            obs["Ttcp2base"] = T_res
            obs["gripper_open"] = G_res

            state = PickPlaceFSMState.CHECK_PLACE

        # -----------------

        elif state == PickPlaceFSMState.CHECK_PLACE:

            success = check_place_success(obs, obs_rs_dual, rs_env, cam_results, task['pick'], task['place'], arm_name)

            if success:

                state = PickPlaceFSMState.TASK_DONE

            else:

                print("Place failed -> replan pick")

                state = PickPlaceFSMState.PLAN_PICK

        # -----------------

        elif state == PickPlaceFSMState.TASK_DONE:

            return True


# ===============================
# Main Loop
# ===============================

if __name__ == "__main__":

    

    env = RealmanEnvWebSim(gripper_open=GRIPPER_OPEN)

    rs_env_left = Open3dRealsenseEnv("f1471338")
    rs_env_right = Open3dRealsenseEnv("f1471193")

    with open("data/20260202_170600/camera_results.json", "r") as f:
        left_cam_results = json.load(f)

    with open("data/20260131_204802/camera_results.json", "r") as f:
        right_cam_results = json.load(f)

    obs = env.reset()

    obs_rs_left = rs_env_left.reset()
    obs_rs_right = rs_env_right.reset()

    obs_rs_dual = {
        'left_arm_rs_obs': obs_rs_left,
        'right_arm_rs_obs': obs_rs_right,
    }

    obs.update(obs_rs_dual)

    home_T_tcp2base = copy.deepcopy(obs["Ttcp2base"])

    # ==========================================================
    # 任务解析
    # ==========================================================

    # instruction = "Pick the white ball and place it on the basket, then pick the yellow ball and place it on the basket, then pick the red horse and place it on the basket, then pick the rubic's cube and place it on the basket, then pick the glue stick and place it on the basket, then pick the carrot and place it on the basket."
    instruction = "Pick the white ball and place it on the plate, then pick the carrot and place it on the plate."
    task_plan = parse_multi_pick_place_tasks(instruction)
    tasks = task_plan["tasks"]

    print(f"Total tasks: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"[Task {i}] pick={t['pick']} place={t['place']}")

    # 设置每个任务使用哪个机械臂
    arm_asign_list = ['left_arm'] * len(tasks)

    # 初始化
    current_task_idx = 0
    action = {"Ttcp2base": home_T_tcp2base, "gripper_open": GRIPPER_OPEN}
    pick_pt = place_pt = pick_targets = place_targets = None

    print("🚀 FSM Started")

    while current_task_idx < len(tasks):

        task = tasks[current_task_idx]
        arm_name = arm_asign_list[current_task_idx]
        if arm_name == "left_arm":
            rs_env = rs_env_left
            cam_results = left_cam_results
        else:
            rs_env = rs_env_right
            cam_results = right_cam_results

        is_success = run_fsm(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name)

        if is_success:
            print(f"✅ Task[{current_task_idx}] finished")
        else:
            print(f"❌ Task[{current_task_idx}] failed")
            break

        current_task_idx += 1

    if(current_task_idx == len(tasks)):
        print("🎉 All tasks finished")
    exit()
