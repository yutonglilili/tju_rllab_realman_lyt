import traceback
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from demo_new.skills.tools.config_utils import resolve_config_path

from demo_new.skills.pnp_skill.pick_and_place import(
    init_robot_env, init_camera_env, init_state, start_pnp_system,
    run_all_tasks_by_instruction_with_position_description, shutdown_pnp_system
)
    

def main():
    task_config_path = resolve_config_path(__file__)

    # 左臂
    robot_ip = "192.168.101.19"
    camera_serial = "f1471338"
    cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"

    # 指令
    instruction = "把魔方放到泰迪熊和玩具马的中间。"

    # 初始化资源
    env, home_T_tcp2base = init_robot_env(robot_ip)
    rs_env, cam_results = init_camera_env(camera_serial, cam_results_path)

    # 状态
    state = init_state(task_config_path=task_config_path)

    # 启动系统
    start_pnp_system(state, env, rs_env, cam_results, home_T_tcp2base)

    # 执行
    try:
        run_all_tasks_by_instruction_with_position_description(state, env, rs_env, cam_results, instruction, home_T_tcp2base)
    except KeyboardInterrupt:
        print("\n[停止] 收到键盘中断，正在停止...")
    except Exception as e:
        print(f"\n[错误] 未捕获异常: {e}")
        traceback.print_exc()
    finally:
        shutdown_pnp_system(state, env)


if __name__ == "__main__":
    main()
