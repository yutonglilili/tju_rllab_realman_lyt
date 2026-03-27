import threading
import copy
import json
import os
import sys
import time
import numpy as np
from enum import Enum, auto

# 项目路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from open3d_realsense_env import *
from realman_env import *
from pick_and_place_utils import *
from multi_pointing_vllm_get_point_utils import *

# ===============================
# 全局共享状态
# ===============================
class RobotState:
    def __init__(self):

        # 1. 基础同步锁
        self._lock = threading.Lock()

        # 2. 感知输出
        self.target_name = None                 # 目标物体名称
        self.object_pose = None                 # 目标物体位姿 (Base 坐标系)
        self.container_name = None              # 容器物体名称
        self.container_pose = None              # 容器中放置物体位姿 (Base 坐标系)

        # 3. 控制输出
        self.current_phase = "IDLE"            # 当前任务阶段 (IDLE, APPROACH, GRASP, PLACE)
        self.is_moving = False                 # 控制线程是否正在运动
        self.abort_event = threading.Event()   # 用于瞬间终止控制线程
        

# ===============================
# 感知线程
# ===============================
def perception_thread(state, rs_env, cam_results, home_T_tcp2base, arm_name="left_arm"):
    """
    
    """
    print("核心感知线程已启动...")
    last_pose = None

    while True:
        
        # --- 获取最新 RGB 图像 ---
        obs = rs_env.step()   # Open3dRealsenseEnv 返回字典
        image_rgb = obs["rgb"]

        # --- 调用模型打点 ---
        res = get_point_vllm(image_rgb, f"Point the {state.target_name}")
        new_pt = res[0]["point_2d"] if isinstance(res, list) else res

        # --- 转换为 3D 坐标 (Base 坐标系) ---
        new_T = make_target_T(obs, int(new_pt[0]), int(new_pt[1]), rs_env, cam_results, home_T_tcp2base)
        new_xyz = new_T[:3, 3]

        with state.lock:
            
        
        if current_target:
            try:

                # --- 更新全局状态 & 简单异常检测 ---
                with state.lock:
                    # 简单异常检测：如果新旧坐标突变 > 5cm，标记异常
                    if state.target_xyz is not None:
                        if np.linalg.norm(new_xyz - state.target_xyz) > 0.05:
                            print(f"⚠️ 检测到物体 {current_target} 发生大幅位移！")
                            state.anomaly_detected = True
                    state.target_xyz = new_xyz
            
            except Exception as e:
                print(f"感知解析异常: {e}")
        
        time.sleep(0.1) # 感知频率，取决于 VLM 推理速度

# ===============================
# 核心控制器 (视觉伺服类)
# ===============================
class DynamicController:
    def __init__(self, env, state):
        self.env = env
        self.state = state
        self.safe_height = 0.08

    def move_to_target_dynamic(self, object_name, approach_offset=0.05):
        """
        动态接近物体：在移动过程中，不断根据 state.target_xyz 修正路径
        """
        print(f"🎯 开始动态追踪任务: {object_name}")
        with self.state.lock:
            self.state.target_name = object_name
            self.state.target_xyz = None
            self.state.anomaly_detected = False

        # 等待第一次定位成功
        while self.state.target_xyz is None:
            time.sleep(0.1)

        while True:
            with self.state.lock:
                target_pos = copy.deepcopy(self.state.target_xyz)
                if self.state.anomaly_detected:
                    print("🛑 任务因物体异常移动而中止")
                    return False

            # 获取当前 TCP 位置
            current_obs = self.env.compute_observation()
            current_tcp = np.array(current_obs["pose"][:3]) # [x, y, z]

            # 计算残差 (Residual)
            target_with_offset = target_pos + np.array([0, 0, approach_offset])
            error = target_with_offset - current_tcp
            dist = np.linalg.norm(error)

            if dist < 0.005: # 达到 5mm 精度视为到达
                print(f"✅ 已动态到达 {object_name} 上方")
                break

            # 视觉伺服步进：发一个中间点，防止机械臂突变
            step_size = 0.02 
            direction = error / dist
            next_step = current_tcp + direction * min(dist, step_size)
            
            # 构建实时下发位姿 (保持原有的旋转角)
            cmd_pose = list(next_step) + current_obs["pose"][3:]
            self.env.send_pose(cmd_pose)
            
            time.sleep(0.04) # 25Hz 运动控制频率

        return True

# ===============================
# 主逻辑
# ===============================
def main():
    # --- 初始化硬件 ---
    env = RealmanEnv(robot_ip="192.168.101.19", async_mode=True)
    rs_env = Open3dRealsenseEnv("f1471338")
    with open("/home/zhangzhao/lyt/camera/20260202_170600/camera_results.json", "r") as f:
        cam_results = json.load(f)

    home_eef_xyzrpy = [-0.036,-0.220,0.352,3.141,0,-2.618]
    home_T_eef2base = T_from_realman_xyzrpy(np.array(home_eef_xyzrpy))
    home_T_tcp2base = home_T_eef2base @ T_TCP2REALMANEEF

    # --- 启动共享状态 & 感知线程 ---
    state = SharedState()
    percept_thread = threading.Thread(
        target=perception_worker, 
        args=(state, rs_env, cam_results, home_T_tcp2base), 
        daemon=True
    )
    percept_thread.start()

    controller = DynamicController(env, state)

    # --- 任务循环 ---
    instruction = "Pick the white ball and place it on the blue plate."
    task_plan = parse_multi_pick_place_tasks(instruction)
    
    try:
        for task in task_plan["tasks"]:
            # --- 动态抓取阶段 ---
            # 1. 移动到物体上方（动态修正）
            success = controller.move_to_target_dynamic(task['pick'], approach_offset=0.06)
            if not success: continue

            # 2. 执行物理抓取 (此阶段通常较快，可以使用短序列)
            print("执行抓取动作...")
            env.send_gripper(0.09) # Open
            time.sleep(0.5)
            # 此处可以根据最新的 state.target_xyz 下压
            final_pick_pose = list(state.target_xyz) + [3.14, 0, -2.617] 
            env.send_pose(final_pick_pose)
            time.sleep(1.0)
            env.send_gripper(0.03) # Close
            time.sleep(0.5)

            # --- 动态放置阶段 ---
            # 切换追踪目标为放置容器
            success = controller.move_to_target_dynamic(task['place'], approach_offset=0.1)
            if success:
                print("执行放置动作...")
                # 同样的下压和松开逻辑...
                env.send_gripper(0.09)

    except KeyboardInterrupt:
        print("手动停止")
    finally:
        state.is_running = False
        env.close()

if __name__ == "__main__":
    main()