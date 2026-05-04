"""
原子动作库
"""
import copy
import os
import sys
import numpy as np

# 项目路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.realman_env import pose_tcp2eef

# 移动到目标位姿并夹紧夹爪
def move_and_close_gripper(env, target_tcp_xyzrpy, gripper_close_value=0.01):
    action = {
        "pose": np.asarray(target_tcp_xyzrpy, dtype=float),
        "motion": "linear",
        "gripper": gripper_close_value,
        "wait_gripper": True,
    }
    env.step(action)
    return True

# 移动到目标位姿并松开夹爪
def move_and_open_gripper(env, target_tcp_xyzrpy, gripper_open_value=0.09):
    action = {
        "pose": np.asarray(target_tcp_xyzrpy, dtype=float),
        "motion": "pose",
        "gripper": gripper_open_value,
        "wait_gripper": False,
    }
    env.step(action)
    return True

# 沿给定方向做直线运动
def move_by_direction(env, direction_xyz, move_distance=0.05):
    """
    沿着 direction_xyz 指定的方向进行直线运动，运动距离为 move_distance。
    其中:
        - direction_xyz 为运动方向，格式为 [x, y, z]，为单位向量。
        - move_distance 为运动距离，单位为米。
    """
    # 获取当前机械臂状态
    state = env.get_state()
    current_pose = state.pose
    current_gripper = state.gripper

    direction_xyz = np.asarray(direction_xyz, dtype=float)

    # 计算目标位姿
    target_xyz = current_pose[:3] + direction_xyz * float(move_distance)
    target_pose = np.concatenate([target_xyz, current_pose[3:]])

    # 移动到目标位姿
    action = {
        "pose": np.asarray(target_pose, dtype=float),
        "motion": "linear",
        "gripper": current_gripper,
    }

    env.step(action)

    return True

# 沿给定方向为轴心做旋转运动
def rotate_by_direction(env, direction_xyz, rotate_angle=30):
    """
    沿着 direction_xyz 指定的方向轴做旋转运动，旋转角度为 rotate_angle。
    其中:
        - direction_xyz 为旋转方向，格式为 [x, y, z]，为单位向量。
        - rotate_angle 为旋转角度，单位为度。
    """
    # 获取当前机械臂状态
    state = env.get_state()
    current_pose = state.pose
    current_gripper = state.gripper
    
    direction_xyz = np.asarray(direction_xyz, dtype=float)

    # 角度 -> 弧度
    angle_rad = np.deg2rad(rotate_angle)

    # 简单hack：把旋转分配到rpy
    delta_rpy = direction_xyz * angle_rad

    target_rpy = current_pose[3:] + delta_rpy
    target_pose = np.concatenate([current_pose[:3], target_rpy])

    action = {
        "pose": target_pose,
        "motion": "linear",
        "gripper": current_gripper,
    }

    print("current_tcp_pose: ", current_pose)
    print("current_eef_pose: ", pose_tcp2eef(current_pose))
    print("target_tcp_pose: ", target_pose)
    print("target_eef_pose: ", pose_tcp2eef(target_pose))

    env.step(action)

    return True