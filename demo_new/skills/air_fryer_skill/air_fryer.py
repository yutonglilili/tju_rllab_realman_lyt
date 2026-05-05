import os
import sys
import numpy as np

# 项目路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.realman_env import RealmanEnv, pose_eef2tcp, T_from_realman_xyzrpy, realman_xyzrpy_from_T
from demo_new.skills.tools.atomic_actions import move_and_close_gripper, move_by_direction, rotate_by_direction
from demo_new.skills.tools.utils import make_lift_T


# ═══════════════════════════════════════════════════
# air fryer skills 函数
# ═══════════════════════════════════════════════════

# 给点执行打开动作
def open_action(env, target_tcp_xyzrpy, direction_xyz):
    
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

# 给点执行关闭动作
def close_action(env, target_tcp_xyzrpy, direction_xyz):

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

# 给点执行旋转动作
def rotate_action(env, target_tcp_xyzrpy, direction_xyz, rotate_angle=90):
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

# ═══════════════════════════════════════════════════
# air fryer skills 使用示例
# ═══════════════════════════════════════════════════
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
    open_action(env, tcp_pose_open, direction_xyz_open)

    
    # 关闭空气炸锅
    eef_pose_close = np.array([-0.00,-0.377,0.179,-2.106,-0.404,1.78])
    tcp_pose_close = pose_eef2tcp(eef_pose_close)
    direction_xyz_close = np.array([1,0,0])
    close_action(env, tcp_pose_close, direction_xyz_close)
    
    
    # 旋转空气炸锅定时按钮
    #eef_pose_rotate = np.array([-0.144,-0.403,0.263,-1.57,-0.523,1.57])
    eef_pose_rotate = np.array([-0.11,-0.334,0.153,-1.57,-0.523,1.57])
    tcp_pose_rotate=pose_eef2tcp(eef_pose_rotate)
    direction_xyz_rotate=np.array([1,0,0])
    rotate_angle=40

    rotate_action(env, tcp_pose_rotate, direction_xyz_rotate, rotate_angle=rotate_angle)

    print("tcp_pose_open: ", tcp_pose_open)
    print("eef_pose_open: ", eef_pose_open)
    print("tcp_pose_close: ", tcp_pose_close)
    print("eef_pose_close: ", eef_pose_close)
    print("tcp_pose_rotate: ", tcp_pose_rotate)
    print("eef_pose_rotate: ", eef_pose_rotate)
    

if __name__ == "__main__":
    main()