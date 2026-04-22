from argparse import Action
import copy
import json
import os
import sys
import time
import threading
import numpy as np
from enum import Enum, auto   

# 项目路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.realman_env import RealmanEnv, pose_tcp2eef, pose_eef2tcp, T_from_realman_xyzrpy, realman_xyzrpy_from_T
from pick_and_place_utils import make_lift_T


def adjust_target_T(target_T, home_T_tcp2base):

    rz_degree = 90
    ry_degree = 30

    # 转换为弧度
    # rx = -1 * (rx_degree / 180) * np.pi
    rz = -1 * (rz_degree / 180) * np.pi
    ry = -1 * (ry_degree / 180) * np.pi
    # rz = np.deg2rad(rz_degree)
    # ry = np.deg2rad(ry_degree)

    # 绕 Z 轴旋转矩阵
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])

    # 绕 Y 轴旋转矩阵
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1, 0         ],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # 组合旋转：先 Z 后 Y (外在坐标轴旋转使用左乘)
    # R_total = R_y * R_z
    R_combined = Ry @ Rz

    # 应用旋转
    grasp_T = copy.deepcopy(home_T_tcp2base)
    # 将组合后的旋转应用到基准姿态上
    grasp_T[:3, :3] = R_combined @ home_T_tcp2base[:3, :3]
    
    # 保持位置与目标一致
    grasp_T[:3, 3] = target_T[:3, 3]

    return grasp_T


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


# 拉开空气炸锅
def open_air_fryer(env, target_tcp_xyzrpy, direction_xyz):
    
    target_tcp_T = T_from_realman_xyzrpy(target_tcp_xyzrpy)
    pre_T = make_lift_T(target_tcp_T, lift_x=0.05, lift_z=0.05)
    pre_pose = realman_xyzrpy_from_T(pre_T)

    env.step({
        "pose": pre_pose,
        "motion": "pose",
        "gripper": 0.09,
    })

    # 移动到空气炸锅把手处并关闭夹爪
    move_and_close_gripper(env, target_tcp_xyzrpy)
    # 拉开空气炸锅
    move_by_direction(env, direction_xyz, move_distance=0.14)
    # 放开夹爪
    env.step({
        "pose": env.get_state().pose,
        "motion": "linear",
        "gripper": 0.09,
    })

    state = env.get_state()
    current_pose = state.pose
    current_T = T_from_realman_xyzrpy(current_pose)
    post_T = make_lift_T(current_T, lift_x=0.05, lift_z=0.05)
    post_pose = realman_xyzrpy_from_T(post_T)
    env.step({
        "pose": post_pose,
        "motion": "linear",
        "gripper": 0.09,
    })
    # reset
    env.reset()
    
    return True

# 关闭空气炸锅
def close_air_fryer(env, target_tcp_xyzrpy, direction_xyz):

    target_tcp_T = T_from_realman_xyzrpy(target_tcp_xyzrpy)
    pre_T = make_lift_T(target_tcp_T, lift_x=0.05, lift_z=0.05)
    pre_pose = realman_xyzrpy_from_T(pre_T)

    env.step({
        "pose": pre_pose,
        "motion": "pose",
        "gripper": 0.09,
    })

    # 移动到空气炸锅把手处并关闭夹爪
    move_and_close_gripper(env, target_tcp_xyzrpy)

    # 关闭前的微调
    fixed_T = make_lift_T(target_tcp_T, lift_y=-0.005,lift_z=0.02)
    fixed_pose = realman_xyzrpy_from_T(fixed_T)
    env.step({
        "pose": fixed_pose,
        "motion": "linear",
    })

    # 关闭空气炸锅
    move_by_direction(env, direction_xyz, move_distance=-0.16)
    env.step({
        "pose": env.get_state().pose,
        "motion": "linear",
        "gripper": 0.09,
    })

    state = env.get_state()
    current_pose = state.pose
    current_T = T_from_realman_xyzrpy(current_pose)
    post_T = make_lift_T(current_T, lift_x=0.05, lift_z=0.05)
    post_pose = realman_xyzrpy_from_T(post_T)
    env.step({
        "pose": post_pose,
        "motion": "linear",
        "gripper": 0.09,
    })

    return True

# 旋转空气炸锅定时按钮
def rotate_air_fryer_timer_button(env, target_tcp_xyzrpy, direction_xyz, rotate_angle=90):
    # 移动到空气炸锅把手处并关闭夹爪
    target_T = T_from_realman_xyzrpy(target_tcp_xyzrpy)
    pre_T = make_lift_T(target_T, lift_x=0.05, lift_z=0.05)
    pre_pose = realman_xyzrpy_from_T(pre_T)
    env.step({
        "pose": pre_pose,
        "motion": "pose",
        "gripper": 0.09,
    })
    move_and_close_gripper(env, target_tcp_xyzrpy)
    print("tcp_pose_1: ", target_tcp_xyzrpy)
    
    # 旋转空气炸锅
    rotate_by_direction(env, direction_xyz, rotate_angle=rotate_angle)
    
    # 放开夹爪
    env.step({
        "pose": env.get_state().pose,
        "motion": "linear",
        "gripper": 0.09,
    })

    current_pose = env.get_state().pose
    current_T = T_from_realman_xyzrpy(current_pose)
    post_T = make_lift_T(current_T, lift_x=0.05, lift_z=0.05)
    post_pose = realman_xyzrpy_from_T(post_T)
    env.step({
        "pose": post_pose,
        "motion": "linear",
        "gripper": 0.09,
    })
    
    # reset
    env.reset()

    return True


# 测试
def main():
    
    # 左臂
    env = RealmanEnv(robot_ip="192.168.101.19", mode="sync")

    env.reset()

    robot_state = env.get_state()
    home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)
    
    # 打开空气炸锅
    eef_pose_open = np.array([-0.12,-0.377,0.181,-2.106,-0.404,1.78])
    tcp_pose_open = pose_eef2tcp(eef_pose_open)
    direction_xyz_open = np.array([1,0,0])
    #open_air_fryer(env, tcp_pose_open, direction_xyz_open)

    
    # 关闭空气炸锅
    eef_pose_close = np.array([-0.00,-0.377,0.179,-2.106,-0.404,1.78])
    tcp_pose_close = pose_eef2tcp(eef_pose_close)
    direction_xyz_close = np.array([1,0,0])
    #close_air_fryer(env, tcp_pose_close, direction_xyz_close)
    
    
    # 旋转空气炸锅定时按钮
    #eef_pose_rotate = np.array([-0.144,-0.403,0.263,-1.57,-0.523,1.57])
    eef_pose_rotate = np.array([-0.11,-0.334,0.153,-1.57,-0.523,1.57])
    tcp_pose_rotate=pose_eef2tcp(eef_pose_rotate)
    direction_xyz_rotate=np.array([1,0,0])
    rotate_angle=40

    #rotate_air_fryer_timer_button(env, tcp_pose_rotate, direction_xyz_rotate, rotate_angle=rotate_angle)

    print("tcp_pose_open: ", tcp_pose_open)
    print("eef_pose_open: ", eef_pose_open)
    print("tcp_pose_close: ", tcp_pose_close)
    print("eef_pose_close: ", eef_pose_close)
    print("tcp_pose_rotate: ", tcp_pose_rotate)
    print("eef_pose_rotate: ", eef_pose_rotate)
    

if __name__ == "__main__":
    main()




