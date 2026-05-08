import json
import os
import sys
import traceback
import numpy as np

# 项目路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T, pose_tcp2eef
from realman.open3d_realsense_env import Open3dRealsenseEnv

from demo_new.vlm_utils.multi_pointing_vllm_get_point_utils import get_point_vllm, parse_roast_with_timer

from demo_new.skills.tools.config_utils import resolve_config_path
from demo_new.skills.tools.utils import make_target_T, make_lift_T, save_pointed_image

from demo_new.skills.air_fryer_skill.air_fryer import open_action, close_action, rotate_action
from demo_new.skills.pnp_skill.pick_and_place import init_state, start_pnp_system, run_all_tasks

# 打开空气炸锅
def open_air_fryer(env, rs_env, cam_results, home_T_tcp2base):
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

    open_action(env, tcp_pose_open, direction_xyz_open)

# 关闭空气炸锅
def close_air_fryer(env, rs_env, cam_results, home_T_tcp2base):
    obs = rs_env.step()
    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(image_rgb, "Point at the handle of the air fryer.", save_path=None)

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    # 偏置
    target_T = make_lift_T(target_T, lift_x=0.02, lift_y=-0.01, lift_z=-0.01)

    tcp_pose_close = realman_xyzrpy_from_T(target_T)

    # 修正 rpy
    tcp_pose_close[3:] = np.array([0.0623, 0.4881, 3.1218])

    direction_xyz_close = np.array([1,0,0])

    close_action(env, tcp_pose_close, direction_xyz_close)


# 设置时间
def set_time(env, rs_env, cam_results, home_T_tcp2base, rotate_angle=90):
    obs = rs_env.step()
    image_rgb = obs["rgb"]

    point_2d = get_point_vllm(image_rgb, "Point at the round knob of the air fryer.", save_path=None)

    # save_pointed_image(image_rgb, point_2d, save_dir="logs", prefix="time_button")

    target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

    # 偏置
    target_T = make_lift_T(target_T, lift_x=0.035, lift_y=-0.02, lift_z=-0.02)

    tcp_pose_rotate = realman_xyzrpy_from_T(target_T)
    
    # 修正 rpy
    tcp_pose_rotate[3:] = np.array([0,0,3.1412])

    direction_xyz_rotate=np.array([1,0,0])

    rotate_action(env, tcp_pose_rotate, direction_xyz_rotate, rotate_angle)


def main():

    task_config_path = resolve_config_path(__file__)

    # 指令
    instruction = "帮我烤橘子和苹果，定时 20 分钟。"

    # 1. 初始化环境
    env = RealmanEnv(robot_ip="192.168.101.19", mode="sync")

    rs_env = Open3dRealsenseEnv("f1471338")

    cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"
    with open(cam_results_path, "r") as f:
        cam_results = json.load(f)

    env.reset()

    robot_state = env.get_state()
    home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)

    
    # 2. 拉开空气炸锅
    open_air_fryer(env, rs_env, cam_results, home_T_tcp2base)
    

    # 3. 将食材放到空气炸锅中

    # 启动系统
    state = init_state(task_config_path=task_config_path)
    start_pnp_system(state, env, rs_env, cam_results, home_T_tcp2base)

    # 拆解指令
    pnp_list, rotate_angle = parse_roast_with_timer(instruction)
    print(pnp_list)
    print(rotate_angle)

    
    # 执行任务
    try:
        run_all_tasks(state, env, rs_env, cam_results, pnp_list, home_T_tcp2base)
    except KeyboardInterrupt:
        print("\n[停止] 收到键盘中断，正在停止...")
    except Exception as e:
        print(f"\n[错误] 未捕获异常: {e}")
        traceback.print_exc()
    finally:
        print("[清理] 停止所有线程...")
        state.stop_all.set()
    

    # 4. 关闭空气炸锅
    close_air_fryer(env, rs_env, cam_results, home_T_tcp2base)
    
    # 5. 设置时间
    set_time(env, rs_env, cam_results, home_T_tcp2base, rotate_angle)
    

if __name__ == "__main__":
    main()