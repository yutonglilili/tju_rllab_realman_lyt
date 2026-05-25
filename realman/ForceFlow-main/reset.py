"""
重置机械臂脚本

功能：
1. 连接机械臂
2. 重置到初始位置（慢速）
3. 打开夹爪
"""

import sys
sys.path.append('/home/zhangzhao/tvla-realenv/zsh')

from realman_env_zsh import RealmanEnv  # 使用带速度控制的版本
import time
import numpy as np

def main():
    # 右臂配置
    robot_ip = "192.168.101.19"
    reset_speed = 50  # 重置速度比例
    
    print("="*60)
    print(f"机械臂重置脚本 (速度比例={reset_speed}%)")
    print("="*60)
    
    try:
        # 连接机械臂（会自动设置速度上限）
        print(f"\n[连接] 连接机械臂 {robot_ip}...")
        env = RealmanEnv(robot_ip)
        print("[连接] 连接成功")
        
        # 获取当前状态
        print("\n[状态] 获取当前位姿...")
        obs = env.compute_observation()
        current_pos = obs["Ttcp2base"][:3, 3]
        print(f"  当前位置: X={current_pos[0]:.3f}m, Y={current_pos[1]:.3f}m, Z={current_pos[2]:.3f}m")
        print(f"  当前夹爪: {obs['gripper_open']:.3f}m")
        
        # 重置机械臂
        print(f"\n[重置] 开始重置（速度比例={reset_speed}%）...")
        
        # 使用 step 方法手动重置，带速度控制
        target_joints = np.array([90, 0, 0, -90, 0, -90, 60])
        from realman_env_zsh import realman_xyzrpy_from_T, T_TCP2REALMANEEF
        
        # 直接调用 rm_movej 进行关节运动
        ret = env.arm.rm_movej(target_joints, v=reset_speed, r=0, connect=0, block=True)
        if ret != 0:
            print(f"[警告] 关节运动返回错误码: {ret}")
        
        # 打开夹爪
        env._set_gripper(1)
        
        print("[重置] 等待机械臂到达初始位置...")
        time.sleep(2)
        
        # 获取重置后状态
        obs = env.compute_observation()
        reset_pos = obs["Ttcp2base"][:3, 3]
        print(f"\n[完成] 重置完成")
        print(f"  重置位置: X={reset_pos[0]:.3f}m, Y={reset_pos[1]:.3f}m, Z={reset_pos[2]:.3f}m")
        print(f"  重置夹爪: {obs['gripper_open']:.3f}m")
        
        print("\n[成功] 机械臂已重置到初始位置")
        
    except KeyboardInterrupt:
        print("\n\n[中断] 用户终止")
        
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'env' in locals():
            print("\n[关闭] 断开机械臂连接...")
            env.close()
            print("[关闭] 完成")

if __name__ == "__main__":
    main()

