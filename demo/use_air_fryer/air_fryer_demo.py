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

from realman.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T
from realman.open3d_realsense_env import Open3dRealsenseEnv

# 基础工具函数
from pick_and_place_utils import (
    make_target_T,
    make_lift_T,
    save_pointed_image,
    crop_image_around_point,
)
from multi_pointing_vllm_get_point_utils import (
    generate_tasks_from_scene_with_failure_reason,
    get_point_vllm,
    check_grasp_success_vllm,
    check_place_success_vllm,
    generate_task_from_scene,
    check_instruction_complete,
    generate_tasks_from_scene,
    generate_tasks_from_scene_with_failure_reason,
)
from atomic_skill_library import *
from pick_and_place import *

def main():

    # ============================
    # 1. 初始化环境
    # ============================
    env = RealmanEnv(robot_ip="192.168.101.19", mode="sync")

    rs_env = Open3dRealsenseEnv("f1471338")

    cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"
    with open(cam_results_path, "r") as f:
        cam_results = json.load(f)

    env.reset()

    robot_state = env.get_state()
    home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)

    
    # ============================
    # 2. 拉开空气炸锅
    # ============================
    obs = rs_env.step()
    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(image_rgb, "Point at the handle of the air fryer.", save_path=None)

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    # 偏置
    target_T = make_lift_T(target_T, lift_x=0.02, lift_y=-0.01)

    tcp_pose_open = realman_xyzrpy_from_T(target_T)

    # 修正 rpy
    tcp_pose_open[3:] = np.array([0.0623, 0.4881, 3.1218])

    print("tcp_pose_open: ", tcp_pose_open)
    print("eef_pose_open: ", pose_tcp2eef(tcp_pose_open))

    direction_xyz_open = np.array([1,0,0])

    open_air_fryer(env, tcp_pose_open, direction_xyz_open)
    
    
    # ============================
    # 3. 将红薯放到空气炸锅中
    # ============================
    state = SharedState()
    curobo_planner = None

    # 启动三线程
    print("\n[启动] 启动工作线程...")
    threads = [
        threading.Thread(
            target=perception_thread,
            args=(state, env, rs_env, cam_results, home_T_tcp2base),
            daemon=True,
            name="PerceptionThread",
        ),
        threading.Thread(
            target=planning_thread,
            args=(state, env, curobo_planner, home_T_tcp2base),
            daemon=True,
            name="PlanningThread",
        ),
        threading.Thread(
            target=execution_thread,
            args=(state, env),
            daemon=True,
            name="ExecutionThread",
        ),
    ]
    for t in threads:
        t.start()
        print(f"  ✅ {t.name} 已启动")
    
    # 任务列表
    task_list = [
        {
        "pick": "sweet potato", "place": "open air fryer drawer",
        }, 
        {
        "pick": "corn", "place": "open air fryer drawer",
        }
    ]

    # 执行任务
    try:
        run_all_tasks(state, env, rs_env, cam_results, task_list, home_T_tcp2base)
    except KeyboardInterrupt:
        print("\n[停止] 收到键盘中断，正在停止...")
    except Exception as e:
        print(f"\n[错误] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[清理] 停止所有线程...")
        state.stop_all.set()
    
    # ============================
    # 4. 关闭空气炸锅
    # ============================
    obs = rs_env.step()
    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(image_rgb, "Point at the handle of the air fryer.", save_path=None)

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    # 偏置
    target_T = make_lift_T(target_T, lift_x=0.02, lift_y=-0.01)

    tcp_pose_close = realman_xyzrpy_from_T(target_T)

    # 修正 rpy
    tcp_pose_close[3:] = np.array([0.0623, 0.4881, 3.1218])

    direction_xyz_close = np.array([1,0,0])

    close_air_fryer(env, tcp_pose_close, direction_xyz_close)
    
    # ============================
    # 5. 设置时间
    # ============================
    obs = rs_env.step()
    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(image_rgb, "Point at the round knob of the air fryer.", save_path=None)

    save_pointed_image(image_rgb, point_2d, save_dir="logs", prefix="time_button")

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    # 偏置
    target_T = make_lift_T(target_T, lift_x=0.03, lift_y=-0.02, lift_z=-0.02)

    tcp_pose_rotate = realman_xyzrpy_from_T(target_T)
    
    # 修正 rpy
    tcp_pose_rotate[3:] = np.array([0,0,3.1412])

    print("tcp_pose_rotate: ", tcp_pose_rotate)
    print("eef_pose_rotate: ", pose_tcp2eef(tcp_pose_rotate))

    direction_xyz_rotate=np.array([1,0,0])
    rotate_angle = 90

    rotate_air_fryer_timer_button(env, tcp_pose_rotate, direction_xyz_rotate, rotate_angle)

if __name__ == "__main__":
    main()