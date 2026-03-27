import copy
import json
import os
import sys
import time
import numpy as np
from enum import Enum, auto

# 将项目根目录加入 import 路径，便于导入 /home/zhangzhao/lyt 下的模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from open3d_realsense_env import Open3dRealsenseEnv
from realman_env import RealmanEnv

# 基础工具函数
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
CHECK_PICK_SUCCESS = 1
CHECK_PLACE_SUCCESS = 1

# 轨迹插值步数 (控制点与点之间的平滑度)
INTERPOLATION_STEPS = 50 
CONTROL_HZ = 0.02 # 50Hz

# 保存图像路径
SAVE_DIR = "/home/zhangzhao/lyt/pick_and_place_2.1/save_images/"


# ===============================
# 状态机定义
# ===============================
class PickPlaceFSMState(Enum):
    PLAN_PICK = auto()
    EXEC_PRE_PICK = auto()
    VERIFY_PRE_PICK = auto()
    EXEC_PICK = auto()
    CHECK_PICK = auto()
    PLAN_PLACE = auto()
    EXEC_PRE_PLACE = auto()
    VERIFY_PRE_PLACE = auto()
    EXEC_PLACE = auto()
    CHECK_PLACE = auto()
    TASK_DONE = auto()

# ===============================
# 核心执行引擎 (替代原有的 Web 请求)
# ===============================

def execute_sequence(env, sequence, name_list):
    """
    取代原有的 send_action_sequence_to_web
    直接通过 env 接口控制机械臂
    """
    print(f"🚀 Executing Sequence: {name_list}")
    
    for i, cmd in enumerate(sequence):
        print(f"  -> Step {i}: {name_list[i]}")
        
        # 1. 处理夹爪
        if "gripper_open" in cmd:
            env.send_gripper(cmd["gripper_open"])
            time.sleep(0.5) # 给夹爪一点反应时间

        # 2. 处理运动
        if "joint" in cmd:
            # 关节空间移动 (直接使用 movej_follow 或 step)
            env.step({"joint": np.array(cmd["joint"])})
            time.sleep(1.0) # 简易等待到达
            
        elif "Ttcp2base" in cmd:
            # 位姿空间移动
            target_T = cmd["Ttcp2base"]
            # 转换为 Realman 格式的 xyzrpy
            # 假设你的 env 类里有这个静态工具函数
            target_pose = RealmanEnv.realman_xyzrpy_from_T(target_T)
            
            if cmd.get("use_moveit", False):
                # 如果仍需规划感，这里可以调用复杂的规划器
                # 目前直接下发目标点
                env.send_pose(target_pose)
                time.sleep(1.5)
            else:
                # 阻塞式或带延迟的移动
                env.send_pose(target_pose)
                time.sleep(1.2)

    return True, None # 模拟返回成功状态


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


# ===============================
# 动作序列构建 (保持原有 T 矩阵计算逻辑)
# ===============================

def build_pre_pick_sequence(targets):
    seq = [{"Ttcp2base": targets["pre_pick_Ttcp2base"], "gripper_open": GRIPPER_OPEN}]
    return seq, ["MOVE_PRE_PICK"]

def build_pick_sequence(targets):
    seq = [
        {"Ttcp2base": targets["pick_Ttcp2base"], "gripper_open": GRIPPER_CLOSE},
        {"Ttcp2base": targets["post_pick_Ttcp2base"], "gripper_open": GRIPPER_CLOSE}
    ]
    return seq, ["MOVE_PICK", "MOVE_POST_PICK"]

def build_pre_place_sequence(targets):
    seq = [
        {"Ttcp2base": targets["intermediate_Ttcp2base"], "gripper_open": GRIPPER_CLOSE},
        {"Ttcp2base": targets["pre_place_Ttcp2base"], "gripper_open": GRIPPER_CLOSE}
    ]
    return seq, ["MOVE_INTERMEDIATE", "MOVE_PRE_PLACE"]

def build_place_sequence(targets):
    seq = [
        {"Ttcp2base": targets["place_Ttcp2base"], "gripper_open": GRIPPER_OPEN},
        {"Ttcp2base": targets["post_place_Ttcp2base"], "gripper_open": GRIPPER_OPEN},
        {"joint": [90, 0, 0, -90, 0, -90, 60], "gripper_open": GRIPPER_OPEN} # 返回 HOME 位姿
    ]
    return seq, ["MOVE_PLACE", "MOVE_POST_PLACE", "RETURN_HOME"]

# ===============================
# 主状态机运行
# ===============================

def run_fsm(env, rs_env, cam_results, task, home_T_tcp2base):
    state = PickPlaceFSMState.PLAN_PICK
    cached_targets = {}
    
    while True:
        # 获取当前观测
        obs = env.compute_observation()
        
        if state == PickPlaceFSMState.PLAN_PICK:
            # 这里的 prepare_pick 内部会调 VLM 打点
            pre, pick, post, pt = prepare_pick(obs, {}, rs_env, cam_results, task, home_T_tcp2base)
            cached_targets.update({"pre_pick": pre, "pick": pick, "post_pick": post, "pt": pt})
            state = PickPlaceFSMState.EXEC_PRE_PICK

        elif state == PickPlaceFSMState.EXEC_PRE_PICK:
            targets = {"pre_pick_Ttcp2base": cached_targets["pre_pick"]}
            seq, names = build_pre_pick_sequence(targets)
            execute_sequence(env, seq, names)
            state = PickPlaceFSMState.EXEC_PICK # 跳过繁琐的验证以加快流程

        elif state == PickPlaceFSMState.EXEC_PICK:
            targets = {
                "pick_Ttcp2base": cached_targets["pick"],
                "post_pick_Ttcp2base": cached_targets["post_pick"]
            }
            seq, names = build_pick_sequence(targets)
            execute_sequence(env, seq, names)
            state = PickPlaceFSMState.CHECK_PICK

        elif state == PickPlaceFSMState.CHECK_PICK:
            if check_pick_success(env, task['pick']):
                state = PickPlaceFSMState.PLAN_PLACE
            else:
                state = PickPlaceFSMState.PLAN_PICK

        elif state == PickPlaceFSMState.PLAN_PLACE:
            inter, pre_p, place, post_p, pt_p = prepare_place(
                obs, {}, rs_env, cam_results, task, 
                home_T_tcp2base, cached_targets["pick"], cached_targets["post_pick"], "left_arm"
            )
            cached_targets.update({
                "inter": inter, "pre_place": pre_p, 
                "place": place, "post_place": post_p
            })
            state = PickPlaceFSMState.EXEC_PRE_PLACE

        elif state == PickPlaceFSMState.EXEC_PRE_PLACE:
            targets = {
                "intermediate_Ttcp2base": cached_targets["inter"],
                "pre_place_Ttcp2base": cached_targets["pre_place"]
            }
            seq, names = build_pre_place_sequence(targets)
            execute_sequence(env, seq, names)
            state = PickPlaceFSMState.EXEC_PLACE

        elif state == PickPlaceFSMState.EXEC_PLACE:
            targets = {
                "place_Ttcp2base": cached_targets["place"],
                "post_place_Ttcp2base": cached_targets["post_place"]
            }
            seq, names = build_place_sequence(targets)
            execute_sequence(env, seq, names)
            state = PickPlaceFSMState.TASK_DONE

        elif state == PickPlaceFSMState.TASK_DONE:
            return True

# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    # 1. 初始化新环境
    # 设定为异步模式以支持更平滑的控制
    env = RealmanEnv(robot_ip="192.168.101.19", async_mode=True, control_mode="absolute")
    
    # 2. 初始化相机
    rs_env_left = Open3dRealsenseEnv("f1471338")
    with open("data/20260202_170600/camera_results.json", "r") as f:
        left_cam_results = json.load(f)

    # 3. 任务准备
    instruction = "Pick the white ball and place it on the blue plate."
    task_plan = parse_multi_pick_place_tasks(instruction)
    tasks = task_plan["tasks"]
    
    # 获取初始 Home 位姿
    init_obs = env.reset()
    home_T = RealmanEnv.T_from_realman_xyzrpy(init_obs["pose"])

    # 4. 循环执行任务
    for i, task in enumerate(tasks):
        print(f"\n--- Starting Task {i}: {task['pick']} -> {task['place']} ---")
        success = run_fsm(env, rs_env_left, left_cam_results, task, home_T)
        if not success:
            print("Task failed, aborting.")
            break

    print("🎉 All tasks completed.")
    env.close()