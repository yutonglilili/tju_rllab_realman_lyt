"""

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


# 将项目根目录加入 import 路径，便于导入 /home/zhangzhao/lyt 下的模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from open3d_realsense_env import Open3dRealsenseEnv
from realman_env import RealmanEnv


# 从工具函数脚本导入功能函数
from pick_and_place_utils import *
from multi_pointing_vllm_get_point_utils import * 


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

# 保存图像路径
SAVE_DIR = "/home/zhangzhao/lyt/pick_and_place_2.1/save_images/"


# ===============================
# Pick Place FSM Main Controller
# ===============================
class PickPlaceFSMState(Enum):

    PLAN_PICK = auto()
    EXEC_PRE_PICK = auto()
    VERIFY_PRE_PICK = auto()    # 检测在执行 pre pick 后和执行 pick 前，物体位置是否改变，若改变则重新 plan pick；若不变，则执行 pick；
    EXEC_PICK = auto()
    CHECK_PICK = auto()

    PLAN_PLACE = auto()
    EXEC_PRE_PLACE = auto()
    VERIFY_PRE_PLACE = auto()    # 检测在执行 pre place 后和执行 place 前，物体位置是否改变，若改变则重新 plan place；若不变，则执行 place；
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


def verify_pre_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None, old_pick_pt=None):

    print("🔍 VERIFY PRE PICK")

    # 1. 更新观测
    obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
    image_rgb = obs[f"{arm_name}_rs_obs"]["rgb"]

    # 2. 重新打点
    new_pick_pt_raw = mp_utils.get_point_vllm(
        image_rgb,
        f"Point the {task['pick']}"
    )
    new_pick_pt = new_pick_pt_raw[0]["point_2d"] if isinstance(new_pick_pt_raw, list) else new_pick_pt_raw

    print("old_pick_pt:", old_pick_pt)
    print("new_pick_pt:", new_pick_pt)

    # 3. 转换为3D
    old_T = make_target_T(
        obs[f"{arm_name}_rs_obs"],
        int(old_pick_pt[0]),
        int(old_pick_pt[1]),
        rs_env,
        cam_results,
        home_T_tcp2base
    )

    new_T = make_target_T(
        obs[f"{arm_name}_rs_obs"],
        int(new_pick_pt[0]),
        int(new_pick_pt[1]),
        rs_env,
        cam_results,
        home_T_tcp2base
    )

    old_xyz = old_T[:3, 3]
    new_xyz = new_T[:3, 3]

    dist = np.linalg.norm(old_xyz - new_xyz)
    print(f"📏 distance = {dist:.4f} m")

    # 4. 判断
    if dist <= 0.05:
        print("✅ Object stable")
        return True, None
    else:
        print("❌ Object moved")
        return False, new_pick_pt

def verify_pre_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None, old_place_pt=None):

    print("🔍 VERIFY PRE PLACE")

    obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
    image_rgb = obs[f"{arm_name}_rs_obs"]["rgb"]

    new_place_pt_raw = mp_utils.get_point_vllm(
        image_rgb,
        f"Point the {task['place']}"
    )
    new_place_pt = new_place_pt_raw[0]["point_2d"] if isinstance(new_place_pt_raw, list) else new_place_pt_raw

    old_T = make_target_T(
        obs[f"{arm_name}_rs_obs"],
        int(old_place_pt[0]),
        int(old_place_pt[1]),
        rs_env,
        cam_results,
        home_T_tcp2base
    )

    new_T = make_target_T(
        obs[f"{arm_name}_rs_obs"],
        int(new_place_pt[0]),
        int(new_place_pt[1]),
        rs_env,
        cam_results,
        home_T_tcp2base
    )

    dist = np.linalg.norm(old_T[:3,3] - new_T[:3,3])
    print(f"📏 distance = {dist:.4f} m")

    if dist <= 0.05:
        print("✅ Place target stable")
        return True, None
    else:
        print("❌ Place moved")
        return False, new_place_pt

"""
def verify_pre_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None):
    return True

def verify_pre_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None):
    return True
"""

# ===============================
# PICK PREPARE（后续在此处做可视化）
# ===============================

def prepare_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name=None, input_pick_pt=None):

    print("🧠 PLAN PICK")
    
    pick_pt = input_pick_pt
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

        
        # 全自动模式下，跳过打点和 graspnet 检测
        # if AUTO_MODE == True:
        return pre_pick_Ttcpbase, pick_Ttcp2base, post_pick_Ttcp2base, pick_pt

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

def prepare_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, pick_Ttcp2base,post_pick_Ttcp2base, arm_name, input_place_pt=None):

    print("🧠 PLAN PLACE")

    place_pt = input_place_pt
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


        # 全自动模式下，跳过打点和 graspnet 检测
        # if AUTO_MODE == True:
        return intermediate_Ttcp2base, pre_place_Ttcp2base, place_Ttcp2base, post_place_Ttcp2base, place_pt

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

def build_pre_pick_sequence(targets):
    seq = []
    names = []

    seq.append({
        "Ttcp2base": targets["pre_pick_Ttcp2base"],
        "gripper_open": GRIPPER_OPEN,
        "control_mode": "pose",
        "use_moveit": True
    })
    names.append("MOVE_PRE_PICK")

    return seq, names


def build_pick_sequence(targets):

    seq = []
    names = []

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

def build_pre_place_sequence(targets):
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

    return seq, names

def build_place_sequence(targets):

    seq = []
    names = []

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
        "joint": [90, 0, 0, -90, 0, -90, 60],
        "gripper_open": GRIPPER_OPEN,
        "control_mode": "joint",
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

    pre_pick_target = None
    pick_target = None
    post_pick_target = None
    intermediate_target = None
    pre_place_target = None
    place_target = None
    post_place_target = None

    cached_pick_pt = None
    cached_place_pt = None

    while True:

        if state == PickPlaceFSMState.PLAN_PICK:

            pre_pick_target, pick_target, post_pick_target, pick_pt = prepare_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name, input_pick_pt=cached_pick_pt)

            cached_pick_pt = None

            state = PickPlaceFSMState.EXEC_PRE_PICK

        elif state == PickPlaceFSMState.EXEC_PRE_PICK:

            # 构建 pre pick 动作序列
            targets={
                "pre_pick_Ttcp2base": pre_pick_target,
            }
            action_seq, action_names = build_pre_pick_sequence(targets)

            # 发送 pre pick 动作序列到web
            status, plan_success, failed_wp, T_res, G_res = send_action_sequence_to_web(action_seq, action_names, arm_name, WEB_BATCH_URL, BATCH_USE_MOVEIT, BATCH_SPEED_PCT, BATCH_BLEND_R_PCT, GRIPPER_OPEN, GRIPPER_CLOSE)

            # 更新反馈
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
            obs["Ttcp2base"] = T_res
            obs["gripper_open"] = G_res

            state = PickPlaceFSMState.VERIFY_PRE_PICK

        elif state == PickPlaceFSMState.VERIFY_PRE_PICK:

            success, new_pick_pt = verify_pre_pick(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name, old_pick_pt=pick_pt)
            
            if success:
                state = PickPlaceFSMState.EXEC_PICK
            else:
                print("Object moved -> re-plan pick")
                cached_pick_pt = new_pick_pt
                state = PickPlaceFSMState.PLAN_PICK

        elif state == PickPlaceFSMState.EXEC_PICK:

            # 构建抓取动作序列
            targets={
                "pick_Ttcp2base": pick_target,
                "post_pick_Ttcp2base": post_pick_target,
            }
            action_seq, action_names = build_pick_sequence(targets)

            # 发送抓取动作序列到web
            status, plan_success, failed_wp, T_res, G_res = send_action_sequence_to_web(action_seq, action_names, arm_name, WEB_BATCH_URL, BATCH_USE_MOVEIT, BATCH_SPEED_PCT, BATCH_BLEND_R_PCT, GRIPPER_OPEN, GRIPPER_CLOSE)

            # 更新反馈
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
            obs["Ttcp2base"] = T_res
            obs["gripper_open"] = G_res

            state = PickPlaceFSMState.CHECK_PICK

        elif state == PickPlaceFSMState.CHECK_PICK:

            success = check_pick_success(obs, obs_rs_dual, rs_env, cam_results, task['pick'], arm_name)

            if success:
                state = PickPlaceFSMState.PLAN_PLACE
            else:
                print("Pick failed -> retry")
                state = PickPlaceFSMState.PLAN_PICK

        elif state == PickPlaceFSMState.PLAN_PLACE:

            intermediate_target, pre_place_target, place_target, post_place_target, place_pt = prepare_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, pick_target, post_pick_target, arm_name)

            cached_place_pt = None

            state = PickPlaceFSMState.EXEC_PRE_PLACE

        elif state == PickPlaceFSMState.EXEC_PRE_PLACE:

            # 构建 pre place 动作序列
            targets={
                "intermediate_Ttcp2base": intermediate_target,
                "pre_place_Ttcp2base": pre_place_target,
            }
            action_seq, action_names = build_pre_place_sequence(targets)

            # 发送 pre place 动作序列到web
            status, plan_success, failed_wp, T_res, G_res = send_action_sequence_to_web(action_seq, action_names, arm_name, WEB_BATCH_URL, BATCH_USE_MOVEIT, BATCH_SPEED_PCT, BATCH_BLEND_R_PCT, GRIPPER_OPEN, GRIPPER_CLOSE)

            # 更新反馈
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
            obs["Ttcp2base"] = T_res
            obs["gripper_open"] = G_res

            state = PickPlaceFSMState.VERIFY_PRE_PLACE

        elif state == PickPlaceFSMState.VERIFY_PRE_PLACE:

            success, new_place_pt = verify_pre_place(obs, obs_rs_dual, rs_env, cam_results, task, home_T_tcp2base, arm_name, old_place_pt=place_pt)
            
            if success:
                state = PickPlaceFSMState.EXEC_PLACE
            else:
                print("Container moved -> re-plan place")
                cached_place_pt = new_place_pt
                state = PickPlaceFSMState.PLAN_PLACE

        elif state == PickPlaceFSMState.EXEC_PLACE:
            # 构建放置动作序列
            targets={
                "place_Ttcp2base": place_target,
                "post_place_Ttcp2base": post_place_target,
            }
            action_seq, action_names = build_place_sequence(targets)

            # 发送放置动作序列到web
            status, plan_success, failed_wp, T_res, G_res = send_action_sequence_to_web(action_seq, action_names, arm_name, WEB_BATCH_URL, BATCH_USE_MOVEIT, BATCH_SPEED_PCT, BATCH_BLEND_R_PCT, GRIPPER_OPEN, GRIPPER_CLOSE)

            # 更新反馈
            obs_rs = update_obs(obs, obs_rs_dual, rs_env, arm_name)
            obs["Ttcp2base"] = T_res
            obs["gripper_open"] = G_res

            state = PickPlaceFSMState.CHECK_PLACE

        elif state == PickPlaceFSMState.CHECK_PLACE:

            success = check_place_success(obs, obs_rs_dual, rs_env, cam_results, task['pick'], task['place'], arm_name)

            if success:
                state = PickPlaceFSMState.TASK_DONE
            else:
                print("Place failed -> retry")
                state = PickPlaceFSMState.PLAN_PICK

        elif state == PickPlaceFSMState.TASK_DONE:
            return True

        else:
            print("Invalid state -> exit")
            return False


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
    instruction = "Pick the white ball and place it on the blue plate, then pick the carrot and place it on the green plate."
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
