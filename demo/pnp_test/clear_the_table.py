"""
此版本支持连续 pnp 任务，并在每个任务中使用感知，规划，执行三线程协同工作，实现物体跟踪，错误检测，重规划等功能。
"""
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
    save_check_image,
    crop_image_around_point,
)
from multi_pointing_vllm_get_point_utils import (
    get_point_vllm,
    check_grasp_success_vllm,
    check_place_success_vllm,
    generate_task_from_scene,
    check_instruction_complete,
    generate_tasks_from_scene,
)


# ═══════════════════════════════════════════════════
# 参数配置
# ═══════════════════════════════════════════════════

# 感知参数
PERCEPTION_INTERVAL = 0.3       # 打点频率（秒），取决于 VLM 推理速度
TASK_DISCOVERY_INTERVAL = 2.0  # 没任务时监视频率
PLACE_Z_OFFSET = 0.08           # place 阶段 z 轴高度偏移（米）
MOVE_OBJECT_THRESHOLD = 0.05    # 物体移动检测阈值（米，5cm）
MOVE_CONTAINER_THRESHOLD = 0.20 # 容器移动检测阈值（米，10cm）

# 规划参数
SAFE_HEIGHT = 0.06              # 安全高度（米）
TRAJECTORY_DOWNSAMPLE = 2       # 轨迹下采样率

# 执行参数
CONTROL_INTERVAL = 0.0          # 执行线程循环间隔（秒），sync 模式下 movep 本身阻塞，此值仅为防空转
GRIPPER_OPEN = 0.09             # 夹爪全开
GRIPPER_CLOSE = 0.03            # 夹爪全闭
# 连续运动失败（SyncController 抛 RuntimeError）达到此次数则放弃本段轨迹，触发 need_replan 重新规划
MAX_CONSECUTIVE_MOTION_FAILURES = 5

# 任务参数
MAX_PICK_RETRIES = 5            # pick 最大重试次数
MAX_PLACE_RETRIES = 5           # place 最大重试次数

# 检测开关（1: 自动化检测, 2: 跳过检测, 3: 人工检测）
CHECK_PICK_SUCCESS_MODE = 1
CHECK_PLACE_SUCCESS_MODE = 1

CHECK_PICK_CROP_SIZE = 560
CHECK_PLACE_CROP_SIZE = 640

# 抓取/放置成功距离阈值
PICK_SUCCESS_DIST_THRESHOLD = 0.10
PLACE_SUCCESS_DIST_THRESHOLD = 0.20

# 保存图像路径
SAVE_DIR = "/home/zhangzhao/lyt/demo/pick_and_place/save_images/"

# 抓取角度参数
RX_DEGREE_CLOSE = 10
RX_DEGREE_FAR_HIGH = 45
RX_DEGREE_FAR_LOW = 30

# 预抓取位姿偏移
APPROACH_Y_OFFSET = 0.04
APPROACH_Z_OFFSET = 0.08
APPROACH_LINEAR_STEP = 0.03      # APPROACH 直线插值的单段位移（米）
APPROACH_ANGULAR_STEP = 0.15     # APPROACH 直线插值的单段姿态变化（弧度）


# ═══════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════

class TaskPhase(Enum):
    """当前任务所处阶段"""
    IDLE = auto()
    PICK = auto()
    PLACE = auto()
    COMPLETE = auto()


# ═══════════════════════════════════════════════════
# 共享状态
# ═══════════════════════════════════════════════════

class SharedState:
    """线程间共享状态，所有读写必须在 self.lock 内"""

    def __init__(self):
        self.lock = threading.Lock()

        # ===== 任务信息 =====
        self.current_task = None                    # {'pick': ..., 'place': ...}
        self.task_phase = TaskPhase.IDLE

        # ===== 感知输出 =====
        self.target_name = None                     # 当前追踪的目标名称
        self.latest_point_2d = None                 # 最新 2D 打点结果 (x, y)
        self.latest_point_3d = None                 # 最新 3D 坐标 (base 坐标系)
        self.latest_target_T = None                 # 最新目标物体的 4x4 位姿矩阵（此处为 TCP2BASE）
        self.last_stable_point_3d = None            # 上次稳定 3D 坐标
        self.point_changed = False
        self.is_first_point = True
        self.tracking_mode = False                  # 追踪模式
        self.verify_mode = False                    # 验证模式

        # ===== 规划输出 =====
        self.action_list = []                       # [{"joints": ..., "gripper": ..., "tag": ...}, ...]
        self.action_index = 0
        self.plan_ready = threading.Event()         # 规划完成信号
        self.need_replan = threading.Event()        # 需要重规划信号

        # ===== 执行控制 =====
        self.attemp_count = 0
        self.abort_execution = threading.Event()    # 停止当前执行信号

        # ===== 任务结果 =====
        self.task_done = threading.Event()
        self.task_success = False

        # ===== 全局控制 =====
        self.stop_all = threading.Event()           # 全局停止
    
    def reset_state(self):
        with self.lock:

            self.current_task = None
            self.task_phase = TaskPhase.IDLE

            self.target_name = None
            self.latest_point_2d = None
            self.latest_point_3d = None
            self.latest_target_T = None
            self.last_stable_point_3d = None
            self.point_changed = False
            self.is_first_point = True
            self.tracking_mode = False
            self.verify_mode = False

            self.action_list = []
            self.action_index = 0
            self.plan_ready.clear()
            self.need_replan.clear()

            self.attemp_count = 0
            self.abort_execution.clear()

            self.task_done.clear()
            self.task_success = False


# ═══════════════════════════════════════════════════
# 感知线程
# ═══════════════════════════════════════════════════

def perception_thread(state, env, rs_env, cam_results, home_T_tcp2base):
    """
    感知线程, 分为两种模式
    1. 追踪模式: 持续以固定频率调用 VLM 打点，检测物体位置变化。
    2. 验证模式: 调用 VLM 验证抓取/放置是否成功。

    Args:
        state: SharedState 实例
        env: RealmanEnv 实例
        rs_env: Open3dRealsenseEnv 实例
        cam_results: 相机标定结果
        home_T_tcp2base: home 位姿矩阵

    Returns:
        None
    """
    print("[感知线程] 已启动")

    while not state.stop_all.is_set():

        # 追踪模式
        if state.tracking_mode:

            # 获取当前追踪目标
            with state.lock:
                target_name = state.target_name
                task_phase = state.task_phase
                if target_name is None:
                    continue

            try:
                # 获取 RGB 图像
                obs = rs_env.step()
                image_rgb = obs["rgb"]

                # double check tracking 状态(防止 post 阶段误打点)
                with state.lock:
                    if not state.tracking_mode:
                        time.sleep(PERCEPTION_INTERVAL)
                        continue

                # 调用 VLM 打点
                point_2d = get_point_vllm(image_rgb, f"Point the {target_name}", save_path=None)

                # 保存打点图片
                # save_check_image(image_rgb, point_2d, SAVE_DIR)
                # get_point_vllm 返回 np.array([x, y])

                # 2D → 3D 转换
                target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

                # 对 place 阶段修正 z 轴高度
                if task_phase == TaskPhase.PLACE:
                    target_T = make_lift_T(target_T, lift_z=PLACE_Z_OFFSET)

                # 修正通用的相机标定偏移
                target_T = make_lift_T(target_T, lift_x=-0.01, lift_y=-0.01)

                target_xyz = target_T[:3, 3]

                # 4. 更新共享状态 & 变化检测
                with state.lock:
                    state.latest_point_2d = point_2d.copy()
                    state.latest_point_3d = target_xyz.copy()
                    state.latest_target_T = target_T.copy()

                    if state.is_first_point:
                        # 第一个有效点
                        state.last_stable_point_3d = target_xyz.copy()
                        state.is_first_point = False
                        state.point_changed = True
                        state.need_replan.set()
                        print(f"[感知] 📍 首次定位 {target_name}: xyz={np.round(target_xyz, 4)}")

                    else:
                        dist = np.linalg.norm(target_xyz - state.last_stable_point_3d)

                        # 根据当前阶段选择移动阈值
                        if state.task_phase == TaskPhase.PICK:
                            threshold = MOVE_OBJECT_THRESHOLD
                        elif state.task_phase == TaskPhase.PLACE:
                            threshold = MOVE_CONTAINER_THRESHOLD
                        else:
                            continue

                        if dist > threshold:
                            # 目标移动了！
                            state.last_stable_point_3d = target_xyz.copy()
                            state.point_changed = True
                            state.abort_execution.set() # 中止当前执行
                            state.need_replan.set()     # 触发重规划

                            label = "物体" if state.task_phase == TaskPhase.PICK else "容器"
                            print(f"[感知] ⚠️ {label} {target_name} 移动！距离: {dist:.4f}m (阈值={threshold}m) → 重规划")
                        else:
                            state.point_changed = False # 目标没有移动

            except Exception as e:
                print(f"[感知] 异常: {e}")

        # 验证模式
        elif state.verify_mode:

            with state.lock:
                current_task = state.current_task
                task_phase = state.task_phase
                target_name = state.target_name
                check_point_2d = None if state.latest_point_2d is None else state.latest_point_2d.copy()
                attemp_count = state.attemp_count

            if task_phase == TaskPhase.PICK:
                
                pick_success = do_check_pick_success(env,rs_env, current_task['pick'], point_2d=check_point_2d,cam_results=cam_results, home_T_tcp2base=home_T_tcp2base)
                
                if pick_success:
                    # 重置状态
                    current_task = state.current_task
                    state.reset_state()
                    # 改为 place 阶段
                    with state.lock:
                        state.current_task = current_task
                        state.task_phase = TaskPhase.PLACE
                        state.target_name = current_task['place']
                        state.tracking_mode = True
                        state.verify_mode = False
                
                else:
                    if attemp_count + 1 >= MAX_PICK_RETRIES:
                        state.task_success = False
                        state.task_done.set()
                        continue
                    
                    else:
                        # 回到感知模式，重新打点
                        state.abort_execution.set()
                        state.plan_ready.clear()

                        with state.lock:
                            state.latest_point_2d = None
                            state.latest_point_3d = None
                            state.latest_target_T = None
                            state.last_stable_point_3d = None
                            state.point_changed = False
                            state.is_first_point = True
                            state.tracking_mode = True
                            state.verify_mode = False
                            state.attemp_count = attemp_count + 1
                            

            elif task_phase == TaskPhase.PLACE:
                
                place_success = do_check_place_success(rs_env, current_task['pick'], current_task['place'], point_2d=check_point_2d,cam_results=cam_results, home_T_tcp2base=home_T_tcp2base)
            
                if place_success:
                    state.reset_state()
                    with state.lock:
                        state.task_success = True
                        state.task_done.set()
                        continue
                else:
                    if attemp_count + 1 >= MAX_PLACE_RETRIES:
                        state.task_success = False
                        state.task_done.set()
                        continue
                    
                    else:
                        # 回到感知模式，重新打点
                        with state.lock:
                            state.target_name = current_task['pick']
                            state.task_phase = TaskPhase.PICK
                            state.latest_point_2d = None
                            state.latest_point_3d = None
                            state.latest_target_T = None
                            state.last_stable_point_3d = None
                            state.point_changed = False
                            state.is_first_point = True
                            state.tracking_mode = True
                            state.verify_mode = False
                            state.attemp_count = attemp_count + 1
            

    print("[感知线程] 已停止")


# 结果检测
def do_check_pick_success(env, rs_env, pick_name, point_2d=None,cam_results=None, home_T_tcp2base=None):
    """
    通过两种方式进行自动化检测：
        1. 通过 VLM 检测抓取是否成功
        2. 计算抓取点与物体中心点的距离，如果距离小于阈值，则认为抓取成功

    Args:
        env: RealmanEnv 实例
        rs_env: Open3dRealsenseEnv 实例
        pick_name: 物体名称
        point_2d: 抓取点 2D 坐标
        cam_results: 相机结果
        home_T_tcp2base: home 位姿矩阵
    Returns:
        bool: 抓取是否成功
    """

    if CHECK_PICK_SUCCESS_MODE == 1:
        print("[检测] 检查抓取是否成功...")

        # 1. 通过 VLM 检测抓取是否成功
        obs = rs_env.step()
        image_rgb = obs["rgb"]
        image_for_check = crop_image_around_point(
            image_rgb,
            point_2d,
            crop_size=CHECK_PICK_CROP_SIZE,
        )
        # save_check_image(image_for_check, prefix="pick", object_name=pick_name, save_dir=SAVE_DIR)

        is_success_1 = check_grasp_success_vllm(image_for_check, pick_name)

        # 2. 计算抓取点与物体中心点的距离
        # 物体 xyz 坐标
        object_2d = get_point_vllm(image_rgb,f"Point the {pick_name}",save_path=None)

        object_current_T = make_target_T(obs,int(object_2d[0]),int(object_2d[1]),rs_env,cam_results,home_T_tcp2base)
        object_xyzrpy = realman_xyzrpy_from_T(object_current_T)
        
        # 夹爪 xyz 坐标
        tcp_xyzrpy = env.get_state().pose

        dist = np.linalg.norm(object_xyzrpy[:3] - tcp_xyzrpy[:3])

        if dist < PICK_SUCCESS_DIST_THRESHOLD:
            is_success_2 = True
        else:
            is_success_2 = False
        
        if is_success_1 and is_success_2:
            print(f"VLM 检测抓取成功，距离检测抓取成功，抓取成功!")
            return True
        elif is_success_2:
            print(f"VLM 检测抓取失败，距离检测抓取成功，抓取成功!")
            return True
        else:
            print(f"VLM 检测抓取失败，距离检测抓取失败，抓取失败!")
            return False

    elif CHECK_PICK_SUCCESS_MODE == 2:
        print("[检测] 跳过 pick 检测")
        return True
    
    else:
        print("[检测] 人工检测 pick 是否成功")
        while True:
            key = input("Pick 成功? (y/n): ")
            if key == 'y':
                return True
            elif key == 'n':
                return False


def do_check_place_success(rs_env, pick_name, place_name, point_2d=None,cam_results=None, home_T_tcp2base=None):
    """调用 VLM 检测 place 是否成功"""

    if CHECK_PLACE_SUCCESS_MODE == 1:
        
        # 1. 通过 VLM 检测放置是否成功
        print("[检测] 检查放置是否成功...")
        obs = rs_env.step()
        image_rgb = obs["rgb"]
        image_for_check = crop_image_around_point(
            image_rgb,
            point_2d,
            crop_size=CHECK_PLACE_CROP_SIZE,
        )
        # save_check_image(image_for_check, prefix="place", object_name=pick_name, container_name=place_name, save_dir=SAVE_DIR)
        
        is_success_1 = check_place_success_vllm(image_for_check, pick_name, place_name)
        
        # 2. 计算物体与容器的距离
        object_2d = get_point_vllm(image_rgb,f"Point the {pick_name}",save_path=None)
        object_current_T = make_target_T(obs,int(object_2d[0]),int(object_2d[1]),rs_env,cam_results,home_T_tcp2base)
        object_xyzrpy = realman_xyzrpy_from_T(object_current_T)
            
        container_2d = get_point_vllm(image_rgb,f"Point the {place_name}",save_path=None)
        container_current_T = make_target_T(obs,int(container_2d[0]),int(container_2d[1]),rs_env,cam_results,home_T_tcp2base)
        container_xyzrpy = realman_xyzrpy_from_T(container_current_T) 

        dist = np.linalg.norm(object_xyzrpy[:3] - container_xyzrpy[:3])

        is_success_2 = True if dist < PLACE_SUCCESS_DIST_THRESHOLD else False
        
        if is_success_1 and is_success_2:
            print(f"VLM 检测放置成功，距离检测放置成功，放置成功!")
            return True
        elif is_success_2:
            print(f"VLM 检测放置失败，距离检测放置成功，放置成功!")
            return True
        else:
            print(f"VLM 检测放置失败，距离检测放置失败，放置失败!")
            return False

    elif CHECK_PLACE_SUCCESS_MODE == 2:
        print("[检测] 跳过 place 检测")
        return True

    else:
        print("[检测] 人工检测 place 是否成功")
        while True:
            key = input("Place 成功? (y/n): ").strip().lower()
            if key == 'y':
                return True
            elif key == 'n':
                return False


# ═══════════════════════════════════════════════════
# 规划线程
# ═══════════════════════════════════════════════════

def planning_thread(state, env, curobo_planner, home_T_tcp2base):
    """
    规划线程：接收 need_replan 信号，调用 curobo 生成 pick/place 单段轨迹。
    """
    print("[规划线程] 已启动")

    while not state.stop_all.is_set():

        triggered = state.need_replan.wait(timeout=0.05)
        
        if not triggered:
            continue

        state.need_replan.clear()

        with state.lock:
            if state.latest_target_T is None:
                continue
            task_phase = state.task_phase
            target_T = state.latest_target_T.copy()

        try:
            # 获取当前关节状态
            robot_state = env.get_state()           # TODO：这里很不稳定，需要优化
            current_joint = robot_state.joint       # 弧度制

            # 构建动作序列
            action_list = build_action_list(env, target_T, home_T_tcp2base, curobo_planner, task_phase)

            if action_list is None or len(action_list) == 0:
                print("[规划] ⚠️ 规划失败，重新规划")
                state.need_replan.set()             # 触发重规划
                continue

            # 更新共享状态
            state.abort_execution.set()         # 中止旧执行

            # 等 execution thread 确认停止
            while state.plan_ready.is_set():
                time.sleep(0.01)
            
            with state.lock:
                state.action_list = action_list
                state.action_index = 0
                state.abort_execution.clear()       # 允许新执行
                state.plan_ready.set()              # 通知执行线程
                print(f"[规划] 📐 规划完成，动作序列长度: {len(action_list)}")

        except Exception as e:
            print(f"[规划] 异常: {e}")


def build_action_list(env, target_T, home_T_tcp2base, curobo_planner, task_phase):
    """
    构建动作序列(完整的 pre_pick-pick-post_pick 或 pre_place-place-post_place)。

    Args:
        env: RealmanEnv 实例
        target_T: 目标物体的 4x4 位姿矩阵
        home_T_tcp2base: home 位姿矩阵（用于旋转参考）
        curobo_planner: curobo 规划器实例
        task_phase: TaskPhase.PICK 或 TaskPhase.PLACE
        
    Returns:
        action_list: [{"pose": np.array, "gripper": float, "tag": int}, ...]
        其中 tag=0 为 approach 动作, tag=1 为 target 动作, tag=2 为 post 动作
        None 表示规划失败
    """

    # 根据目标 xyz 重新设置 pose
    target_T_new = adjust_target_T(target_T, home_T_tcp2base)

    # 计算 pre 位姿（目标上方安全位置）
    if task_phase == TaskPhase.PICK:
        pre_target_T = make_lift_T(target_T_new, lift_y=APPROACH_Y_OFFSET, lift_z=APPROACH_Z_OFFSET)
        pre_target_pose = realman_xyzrpy_from_T(pre_target_T)
        pre_gripper_state = GRIPPER_OPEN
    else:
        pre_target_T = make_lift_T(target_T_new, lift_z=APPROACH_Z_OFFSET)
        pre_target_pose = realman_xyzrpy_from_T(pre_target_T)
        pre_gripper_state = GRIPPER_CLOSE

    
    # TODO: 使用 pre_target_T 作为目标，调用 curobo 规划器生成 pre 段轨迹
    # trajectory = curobo_planner.plan(current_joint, pre_target_T)   

    if task_phase == TaskPhase.PICK:

        
        target_pose = realman_xyzrpy_from_T(target_T_new)
        target_gripper_state = GRIPPER_CLOSE

        post_target_T = make_lift_T(target_T_new, lift_y=APPROACH_Y_OFFSET, lift_z=APPROACH_Z_OFFSET+0.03)
        post_target_pose = realman_xyzrpy_from_T(post_target_T)

    else:

        target_pose = realman_xyzrpy_from_T(target_T_new)
        target_gripper_state = GRIPPER_OPEN

        post_target_T = make_lift_T(target_T_new, lift_z=APPROACH_Z_OFFSET)
        post_target_pose = realman_xyzrpy_from_T(post_target_T)

    # post 不传 gripper 状态
    action_list = [
        {"pose": pre_target_pose, "gripper": pre_gripper_state, "tag": 0, "motion": "pose", "wait_gripper": True},
        {"pose": target_pose, "gripper": target_gripper_state, "tag": 1, "motion": "linear", "wait_gripper": True},
        {"pose": post_target_pose, "tag": 2, "motion": "linear"},
    ]

    return action_list
            

def adjust_target_T(target_T, home_T_tcp2base):
    """
    根据物体位置计算合适的抓取姿态旋转矩阵。

    Args:
        target_T: 目标物体 4x4 位姿矩阵
        home_T_tcp2base: home 位姿矩阵

    Returns:
        带有正确旋转的抓取位姿 4x4 矩阵
    """
    x, y, z = target_T[:3, 3]

    # 根据 y 值（距离）选择适当的俯仰角
    if y > -0.35:
        rx_degree = RX_DEGREE_CLOSE
    elif z > 0.12:
        rx_degree = RX_DEGREE_FAR_HIGH
    else:
        rx_degree = RX_DEGREE_FAR_LOW

    rx = -1 * (rx_degree / 180) * np.pi
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    grasp_T = copy.deepcopy(home_T_tcp2base)
    grasp_T[:3, :3] = Rx @ home_T_tcp2base[:3, :3]
    grasp_T[:3, 3] = target_T[:3, 3]

    return grasp_T


# ═══════════════════════════════════════════════════
# 执行线程
# ═══════════════════════════════════════════════════

def execution_thread(state, env):
    """
    执行线程：依次执行动作列表中的动作点。

    - 支持中断 abort_execution
    - 支持重新开始 plan_ready
    - 动作列表执行完毕
    - 连续运动失败达到 MAX_CONSECUTIVE_MOTION_FAILURES 时放弃本段轨迹并 need_replan
    """
    print("[执行线程] 已启动")

    while not state.stop_all.is_set():

        # 等待规划完成信号
        triggered = state.plan_ready.wait(timeout=0.05)
        
        if not triggered:
            continue

        print("[执行] ▶️ 开始执行动作序列")
        motion_fail_streak = 0

        while not state.stop_all.is_set():

            # 检查中止信号
            if state.abort_execution.is_set():
                print("[执行] ⏹️ 执行被中止，等待重新规划")
                
                with state.lock:
                    state.action_list = []
                    state.action_index = 0
                
                state.plan_ready.clear()
                break

            with state.lock:
                
                # 检查是否执行完毕
                if state.action_index >= len(state.action_list):
                    state.plan_ready.clear()
                    state.tracking_mode = False
                    state.verify_mode = True   # 感知线程切换到验证模式
                    print("[执行] ✅ 动作序列执行完成")
                    break

                # 取出当前动作（成功执行后再推进 action_index，失败则重试本步）
                action = state.action_list[state.action_index]

            # === 执行动作 ===

            step_action = {}

            if "joints" in action:
                joint_deg = np.degrees(action["joints"]) if np.max(np.abs(action["joints"])) < 2 * np.pi else action["joints"]
                step_action["joint"] = joint_deg
            
            elif "pose" in action:
                step_action["pose"] = action["pose"]

            if "motion" in action:
                step_action["motion"] = action["motion"]

            if "gripper" in action:
                step_action["gripper"] = action["gripper"]

            if "wait_gripper" in action:
                step_action["wait_gripper"] = action["wait_gripper"]

            if state.abort_execution.is_set():
                print("[执行] ⏹️ 下发动作前检测到中止信号，停止旧动作序列")
                with state.lock:
                    state.action_list = []
                    state.action_index = 0
                state.plan_ready.clear()
                break

            # post阶段感知线程空转，避免影响执行
            if action["tag"] == 2:
                with state.lock:
                    state.tracking_mode = False
                    state.verify_mode = False

            try:
                env.step(step_action)
            except RuntimeError as e:
                motion_fail_streak += 1
                print(
                    f"[执行] ⚠️ 运动失败 ({motion_fail_streak}/{MAX_CONSECUTIVE_MOTION_FAILURES}): {e}"
                )
                if motion_fail_streak >= MAX_CONSECUTIVE_MOTION_FAILURES:
                    print(
                        "[执行] ⛔ 连续运动失败达到上限，终止本段轨迹并请求重新规划"
                    )
                    with state.lock:
                        state.action_list = []
                        state.action_index = 0
                    state.abort_execution.set()
                    state.plan_ready.clear()
                    state.need_replan.set()
                    break
                continue

            if state.abort_execution.is_set():
                print("[执行] ⏹️ 动作执行完成后检测到重规划请求，停止后续动作")
                with state.lock:
                    state.action_list = []
                    state.action_index = 0
                state.plan_ready.clear()
                break

            motion_fail_streak = 0
            with state.lock:
                state.action_index += 1

    print("[执行线程] 已停止")

# ═══════════════════════════════════════════════════
# 主线程调度逻辑
# ═══════════════════════════════════════════════════

# 执行单个任务
def run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base):
    if state.stop_all.is_set():
        return False

    state.reset_state()

    with state.lock:
        state.current_task = task
        state.task_phase = TaskPhase.PICK
        state.tracking_mode = True
        state.target_name = task['pick']

    while not state.stop_all.is_set():
        if state.task_done.wait(timeout=0.1):
            break

    if not state.task_done.is_set():
        return False

    env.reset()

    if state.task_success:
        return True
    else:
        return False

# 按照动作列表执行所有任务
def run_all_tasks(state, env, rs_env, cam_results, task_list, home_T_tcp2base):
    if not task_list:
        print("[主线程] 未生成有效任务列表，等待下一轮检测...")
        return

    for i, task in enumerate(task_list):
        if state.stop_all.is_set():
            break

        print(f"\n{'='*60}")
        print(f"🚀 Task [{i+1}/{len(task_list)}]: pick={task['pick']} → place={task['place']}")
        print(f"{'='*60}")

        success = run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base)

        if state.stop_all.is_set():
            break

        if not success:
            print(f"⛔ Task [{i}] 失败，继续下一个任务。")
            continue

        # === 当前任务执行成功后的处理 ===
        if i + 1 < len(task_list):
            next_task = task_list[i + 1]

            # 在 reset 之前，先设置下一个任务的感知目标
            # 这样在 reset 的阻塞时间内，感知线程已经在为下一个任务打点
            with state.lock:
                state.task_phase = TaskPhase.PICK
                state.tracking_mode = True
                state.verify_mode = False
                state.target_name = next_task['pick']
                state.is_first_point = True
                state.last_stable_point_3d = None

            print("[主线程] 🔄 机械臂 Reset 中（感知线程已提前启动下一任务）...")
            env.reset()
        else:
            print("[主线程] 🔄 最后一个任务完成，Reset...")
            env.reset()

    if state.stop_all.is_set():
        print("\n[主线程] 收到停止信号，结束当前任务循环。")
    else:
        print("\n🎉 所有任务完成!")

# 按照模糊指令执行所有任务（一次只输出一组pnp目标）
def run_all_tasks_by_instruction(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    根据自然语言指令持续执行任务：
    1. 调用 VLM 从图像解析 pick/place 任务
    2. 执行单个任务
    3. 循环直到检测到任务完成
    """

    print(f"[主线程] 🧠 指令: {instruction}")

    while not state.stop_all.is_set():

        # 获取当前图像
        obs = rs_env.step()
        image_rgb = obs["rgb"]

        try:
            # 获取一组 pnp 任务目标
            task = generate_task_from_scene(image_rgb, instruction)
            print(f"task: {task}")

            # 如果发现任务，则执行
            if task:
                run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base)
                if state.stop_all.is_set():
                    break
            
            else:
                print("[主线程] 未发现可执行任务，等待...")
                time.sleep(TASK_DISCOVERY_INTERVAL)
                continue

        except Exception as e:
            print(f"[主线程] 异常: {e}")
            if state.stop_all.is_set():
                break

# 按照模糊指令持续完成任务（一次生成多组pnp目标）
def run_all_tasks_by_instruction_with_list(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    根据自然语言指令持续执行任务：
    1. 调用 VLM 判断当前场景是否满足指令的要求，如果满足则定频检测，不满足则生成 pnp list。
    2. 按照list依次执行pnp任务，并在完成一组pnp任务后更新list（将已完成的pnp任务从list中移除，调整新放的和拿走的物体）
    """

    print(f"[主线程] 🧠 指令: {instruction}")

    tasks_list = None

    while not state.stop_all.is_set():

        # 获取当前图像
        obs = rs_env.step()
        image_rgb = obs["rgb"]

        try:
            # 判断当前场景是否满足顶层指令的要求
            check_start = time.perf_counter()
            is_complete, reason = check_instruction_complete(image_rgb, instruction)
            check_elapsed = time.perf_counter() - check_start
            print(f"[主线程] 完成检测耗时: {check_elapsed:.2f}s")
            print(f"is_complete: {is_complete}, reason: {reason}")

            if is_complete:
                print("[主线程] 当前场景满足指令的要求，开始定频检测")
                time.sleep(TASK_DISCOVERY_INTERVAL)
                continue
            else:
                tasks_list = generate_tasks_from_scene(image_rgb, instruction)
                print(f"tasks_list: {tasks_list}")

                if not tasks_list:
                    print("[主线程] 未生成有效任务，等待下一轮检测...")
                    time.sleep(TASK_DISCOVERY_INTERVAL)
                    continue

                run_all_tasks(state, env, rs_env, cam_results, tasks_list, home_T_tcp2base)
                if state.stop_all.is_set():
                    break
        
        except Exception as e:
            print(f"[主线程] 异常: {e}")
            time.sleep(TASK_DISCOVERY_INTERVAL)
            if state.stop_all.is_set():
                break
            continue

            
# ═══════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════

def main():

    # ============================
    # 1. 初始化环境
    # ============================
    
    # 左臂
    env = RealmanEnv(robot_ip="192.168.101.19", mode="sync")

    rs_env = Open3dRealsenseEnv("f1471338")

    cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"
    with open(cam_results_path, "r") as f:
        cam_results = json.load(f)

    # ============================
    # 2. 初始化 curobo
    # ============================

    # curobo_planner = init_curobo(robot_config_path)
    # curobo_planner.warmup()
    curobo_planner = None  # 替换为实际的 curobo 规划器

    # ============================
    # 3. 获取初始位姿
    # ============================

    env.reset()

    robot_state = env.get_state()
    home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)

    # ============================
    # 4. 指令输入
    # ============================

    # instruction = "Clear the table. Pick all toys on the table and place them on the white plate."
    instruction = "Pick the baseball and place it on the rubic's cube."

    # ============================
    # 5. 初始化共享状态
    # ============================

    state = SharedState()

    # ============================
    # 6. 启动三个工作线程
    # ============================
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


    # ============================
    # 7. 主线程：调度任务
    # ============================
    print("\n[运行] 开始执行任务...\n")
    try:
        run_all_tasks_by_instruction(state, env, rs_env, cam_results, instruction, home_T_tcp2base)
    except KeyboardInterrupt:
        print("\n[停止] 收到键盘中断，正在停止...")
    except Exception as e:
        print(f"\n[错误] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[清理] 停止所有线程...")
        state.stop_all.set()
        time.sleep(0.5)
        print("[清理] 关闭环境...")
        env.close()
        print("[完成] 程序退出")


if __name__ == "__main__":
    main()
