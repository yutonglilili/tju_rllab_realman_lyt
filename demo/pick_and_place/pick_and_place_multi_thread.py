"""
多线程 Pick-and-Place 系统

架构：三线程（感知 + 规划 + 执行）+ 主线程调度
- 感知线程：持续调用 VLM 打点，检测物体位置变化
- 规划线程：收到重规划信号后调用 curobo 生成 APPROACH 段轨迹
- 执行线程：依次执行动作列表中的关节点
- 主线程：两段式调度（APPROACH → CRITICAL → CHECKING），任务流程控制

关键设计：
- APPROACH 段：三线程协作，允许物体移动时重规划
- CRITICAL 段：感知线程暂停，主线程阻塞式执行下压/抓取/提起
- CHECKING 段：调用 VLM 检测 pick/place 是否成功
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
)
from multi_pointing_vllm_get_point_utils import (
    parse_multi_pick_place_tasks,
    get_point_vllm,
    check_grasp_success_vllm,
    check_place_success_vllm,
)


# ═══════════════════════════════════════════════════
# 参数配置
# ═══════════════════════════════════════════════════

# 感知参数
PERCEPTION_INTERVAL = 0.3       # 打点频率（秒），取决于 VLM 推理速度
MOVE_OBJECT_THRESHOLD = 0.03    # 物体移动检测阈值（米，3cm）
MOVE_CONTAINER_THRESHOLD = 0.10 # 容器移动检测阈值（米，10cm）

# 规划参数
SAFE_HEIGHT = 0.06              # 安全高度（米）
TRAJECTORY_DOWNSAMPLE = 2       # 轨迹下采样率

# 执行参数
CONTROL_INTERVAL = 0.05         # 执行线程循环间隔（秒），sync 模式下 movep 本身阻塞，此值仅为防空转
GRIPPER_OPEN = 0.09             # 夹爪全开
GRIPPER_CLOSE = 0.03            # 夹爪全闭

# 任务参数
MAX_PICK_RETRIES = 3            # pick 最大重试次数
MAX_PLACE_RETRIES = 3           # place 最大重试次数

# 检测开关（1: 自动化检测, 2: 跳过检测, 3: 人工检测）
CHECK_PICK_SUCCESS_MODE = 1
CHECK_PLACE_SUCCESS_MODE = 1

# 保存图像路径
SAVE_DIR = "/home/zhangzhao/lyt/demo/pick_and_place/save_images/"

# 抓取角度参数
RX_DEGREE_CLOSE = 10
RX_DEGREE_FAR_HIGH = 45
RX_DEGREE_FAR_LOW = 30


# ═══════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════

class TaskPhase(Enum):
    """当前任务所处阶段"""
    PICK = auto()
    PLACE = auto()


class SystemMode(Enum):
    """系统工作模式"""
    TRACKING = auto()       # 持续打点追踪模式（感知线程活跃）
    EXECUTING = auto()      # 动作执行模式（CRITICAL 段，仅执行）
    CHECKING = auto()       # 结果检测模式
    RESETTING = auto()      # 机械臂复位模式
    IDLE = auto()           # 空闲


# ═══════════════════════════════════════════════════
# 共享状态
# ═══════════════════════════════════════════════════

class SharedState:
    """线程间共享状态，所有读写必须在 self.lock 内"""

    def __init__(self):
        self.lock = threading.Lock()

        # ===== 任务信息 =====
        self.current_task = None                    # {'pick': ..., 'place': ...}
        self.task_phase = TaskPhase.PICK
        self.system_mode = SystemMode.IDLE

        # ===== 感知输出 =====
        self.target_name = None                     # 当前追踪的目标名称
        self.latest_point_2d = None                 # 最新 2D 打点结果 (x, y)
        self.latest_point_3d = None                 # 最新 3D 坐标 (base 坐标系)
        self.latest_target_T = None                 # 最新目标物体的 4x4 位姿矩阵（需要检查tcp还是eef）
        self.last_stable_point_3d = None            # 上次稳定 3D 坐标
        self.point_changed = False
        self.is_first_point = True

        # ===== 规划输出 =====
        self.plan_stage = "APPROACH"
        self.action_list = []                       # [{"joints": ..., "gripper": ...}, ...]
        self.action_index = 0
        self.plan_ready = threading.Event()         # 规划完成信号
        self.need_replan = threading.Event()        # 需要重规划信号

        # ===== 执行控制 =====
        self.execution_done = threading.Event()     # 动作列表执行完毕信号
        self.abort_execution = threading.Event()    # 中止当前执行信号

        # ===== 全局控制 =====
        self.stop_all = threading.Event()           # 全局停止
        self.pause_perception = threading.Event()   # 暂停感知线程


# ═══════════════════════════════════════════════════
# 感知线程
# ═══════════════════════════════════════════════════

def perception_thread(state, rs_env, cam_results, home_T_tcp2base):
    """
    感知线程：持续以固定频率调用 VLM 打点，检测物体位置变化。

    - 只在 APPROACH 段活跃（CRITICAL / CHECKING 段被暂停）
    - 首次获取到点位 → 触发 need_replan
    - 物体移动超过阈值 → 触发 abort_execution + need_replan
    """
    print("[感知线程] 已启动")

    while not state.stop_all.is_set():

        # --- 暂停检测 ---
        if state.pause_perception.is_set():
            time.sleep(0.05)
            continue

        # --- 获取当前追踪目标 ---
        with state.lock:
            target_name = state.target_name
            if target_name is None:
                pass  # fall through to sleep
            else:
                target_name_copy = target_name

        if target_name is None:
            time.sleep(0.1)
            continue

        try:
            # 1. 获取 RGB 图像
            obs = rs_env.step()
            image_rgb = obs["rgb"]

            # 2. 调用 VLM 打点
            point_2d = get_point_vllm(
                image_rgb,
                f"Point the {target_name_copy}",
                save_path=None,  # 不保存调试图片，提升速度
            )
            # get_point_vllm 返回 np.array([x, y])

            # 3. 2D → 3D 转换
            target_T = make_target_T(
                obs,
                int(point_2d[0]),
                int(point_2d[1]),
                rs_env,
                cam_results,
                home_T_tcp2base,
            )
            new_xyz = target_T[:3, 3]

            # 4. 更新共享状态 & 变化检测
            with state.lock:
                state.latest_point_2d = point_2d.copy()
                state.latest_point_3d = new_xyz.copy()
                state.latest_target_T = target_T.copy()

                if state.is_first_point:
                    # 第一个有效点
                    state.last_stable_point_3d = new_xyz.copy()
                    state.is_first_point = False
                    state.point_changed = True
                    state.need_replan.set()
                    print(f"[感知] 📍 首次定位 {target_name_copy}: xyz={np.round(new_xyz, 4)}")

                else:
                    dist = np.linalg.norm(new_xyz - state.last_stable_point_3d)

                    # 根据当前阶段选择移动阈值：
                    # PICK 阶段追踪物体 → 小阈值（物体小，容易被推动）
                    # PLACE 阶段追踪容器 → 大阈值（容器大，打点波动更大）
                    threshold = MOVE_OBJECT_THRESHOLD if state.task_phase == TaskPhase.PICK else MOVE_CONTAINER_THRESHOLD

                    if dist > threshold:
                        # 目标移动了！
                        # 感知线程只在 APPROACH 段活跃，所以一定可以触发重规划
                        state.last_stable_point_3d = new_xyz.copy()
                        state.point_changed = True
                        state.abort_execution.set()
                        state.need_replan.set()
                        label = "物体" if state.task_phase == TaskPhase.PICK else "容器"
                        print(f"[感知] ⚠️ {label} {target_name_copy} 移动! dist={dist:.4f}m (阈值={threshold}m) → 重规划")
                    else:
                        state.point_changed = False

        except Exception as e:
            print(f"[感知] 异常: {e}")

        time.sleep(PERCEPTION_INTERVAL)

    print("[感知线程] 已停止")


# ═══════════════════════════════════════════════════
# 规划线程
# ═══════════════════════════════════════════════════

def planning_thread(state, env, curobo_planner, home_T_tcp2base):
    """
    规划线程：接收 need_replan 信号，调用 curobo 生成 APPROACH 段轨迹。

    - 只规划到 pre_pick / pre_place（物体上方安全位置）
    - CRITICAL 段（下压/抓取/提起）由主线程直接下发，不经过此线程
    """
    print("[规划线程] 已启动")

    while not state.stop_all.is_set():

        # 等待重规划信号
        triggered = state.need_replan.wait(timeout=0.5)
        if not triggered:
            continue
        state.need_replan.clear()

        # 只在 TRACKING 模式下规划
        with state.lock:
            if state.system_mode != SystemMode.TRACKING:
                continue
            target_T = state.latest_target_T
            task_phase = state.task_phase
            if target_T is None:
                continue
            target_T = target_T.copy()

        try:
            # 1. 获取当前关节状态（sync 模式下实时读取 SDK）
            robot_state = env.get_state()
            current_joint = robot_state.joint  # 弧度制

            # 2. 构建 APPROACH 段动作列表（只规划到 pre_pick/pre_place）
            action_list = build_approach_action_list(
                target_T, home_T_tcp2base, current_joint,
                curobo_planner, task_phase, env,
            )

            if action_list is None or len(action_list) == 0:
                print("[规划] ⚠️ 规划失败，等待下次重规划")
                continue

            # 3. 更新共享状态（原子操作）
            with state.lock:
                state.abort_execution.set()         # 先中止旧执行
            time.sleep(0.03)                        # 给执行线程响应时间
            with state.lock:
                state.action_list = action_list
                state.action_index = 0
                state.execution_done.clear()
                state.abort_execution.clear()       # 允许新执行
                state.plan_ready.set()              # 通知执行线程
                print(f"[规划] 📐 规划完成，动作序列长度: {len(action_list)}")

        except Exception as e:
            print(f"[规划] 异常: {e}")

    print("[规划线程] 已停止")


def build_approach_action_list(target_T, home_T_tcp2base, current_joint,
                                curobo_planner, task_phase, env):
    """
    构建 APPROACH 段动作序列（只规划到 pre_pick / pre_place）。

    CRITICAL 段（下压/抓取/提起）由主线程单独处理，不在此函数中。

    Args:
        target_T: 目标物体的 4x4 位姿矩阵
        home_T_tcp2base: home 位姿矩阵（用于旋转参考）
        current_joint: 当前关节角度（弧度制）
        curobo_planner: curobo 规划器实例
        task_phase: TaskPhase.PICK 或 TaskPhase.PLACE
        env: RealmanEnv 实例

    Returns:
        action_list: [{"joints": np.array, "gripper": float}, ...]
        None 表示规划失败
    """

    # 1. 根据阶段计算抓取姿态的旋转
    pick_T = compute_grasp_T(target_T, home_T_tcp2base)

    # 2. 计算 pre 位姿（目标上方安全位置）
    pre_target_T = make_lift_T(pick_T, lift_z=SAFE_HEIGHT)

    # 3. 调用 curobo 规划从当前关节状态到 pre_target 的无碰撞轨迹
    # TODO: 替换为实际的 curobo plan 调用
    # trajectory = curobo_planner.plan(current_joint, pre_target_T)
    trajectory = [realman_xyzrpy_from_T(pre_target_T)]

    if trajectory is None:
        return None

    # 4. 下采样
    if len(trajectory) > 2:
        trajectory = trajectory[::TRAJECTORY_DOWNSAMPLE]

    # 5. 构建动作列表
    gripper_state = GRIPPER_OPEN if task_phase == TaskPhase.PICK else GRIPPER_CLOSE
    action_list = []
    for waypoint in trajectory:
        action_list.append({
            "pose": waypoint,       # xyzrpy
            "gripper": gripper_state,
        })

    return action_list


def compute_grasp_T(target_T, home_T_tcp2base):
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

    # 修正相机标定偏移（x 轴偏移 1cm）
    grasp_T = make_lift_T(grasp_T, lift_x=0.01)

    return grasp_T



# ═══════════════════════════════════════════════════
# 执行线程
# ═══════════════════════════════════════════════════

def execution_thread(state, env):
    """
    执行线程：依次执行动作列表中的动作点。

    - 支持中断（abort_execution）
    - 支持重新开始（plan_ready）
    - 动作列表执行完毕 → 设置 execution_done
    """
    print("[执行线程] 已启动")

    while not state.stop_all.is_set():

        # 等待规划完成信号
        triggered = state.plan_ready.wait(timeout=0.5)
        if not triggered:
            continue

        print("[执行] ▶️ 开始执行动作序列")

        while not state.stop_all.is_set():

            # 检查中止信号
            if state.abort_execution.is_set():
                print("[执行] ⏹️ 执行被中止，等待新规划")
                state.plan_ready.clear()
                break

            with state.lock:
                # 检查是否执行完毕
                if state.action_index >= len(state.action_list):
                    state.plan_ready.clear()
                    state.execution_done.set()
                    print("[执行] ✅ APPROACH 动作序列执行完毕")
                    break

                # 取出当前动作
                action = state.action_list[state.action_index]
                state.action_index += 1

            # === 执行动作（锁外，避免长时间持锁）===
            # 通过 env.step() 统一下发（阻塞式）

            step_action = {}

            if "joints" in action:
                joint_deg = np.degrees(action["joints"]) if np.max(np.abs(action["joints"])) < 2 * np.pi else action["joints"]
                step_action["joint"] = joint_deg
            elif "pose" in action:
                step_action["pose"] = action["pose"]

            if "gripper" in action:
                step_action["gripper"] = action["gripper"]

            env.step(step_action)

            time.sleep(CONTROL_INTERVAL)

    print("[执行线程] 已停止")


# ═══════════════════════════════════════════════════
# CRITICAL 段执行器（主线程阻塞式）
# ═══════════════════════════════════════════════════

def execute_critical_sequence(env, critical_actions):
    """
    阻塞式执行 CRITICAL 动作序列（短距离确定性动作）。
    直接在调用线程中顺序执行，不经过多线程。

    Args:
        env: RealmanEnv 实例
        critical_actions: [{"pose": xyzrpy, "gripper": float, "wait": float}, ...]
    """
    for i, action in enumerate(critical_actions):
        print(f"  [CRITICAL] Step {i+1}/{len(critical_actions)}: "
              f"pose={np.round(action['pose'], 3)}, gripper={action['gripper']:.2f}")

        # 先运动到位，再操作夹爪（通过 env.step 统一下发）
        env.step({"pose": action["pose"]})
        time.sleep(1.0)  # 等待到位（短距离运动，留余量确保稳定）

        env.step({"gripper": action["gripper"]})
        wait_time = action.get("wait", 0)
        if wait_time > 0:
            time.sleep(wait_time)


# ═══════════════════════════════════════════════════
# 结果检测
# ═══════════════════════════════════════════════════

def do_check_pick_success(rs_env, pick_name):
    """调用 VLM 检测 pick 是否成功"""

    if CHECK_PICK_SUCCESS_MODE == 1:
        print("[检测] 检查抓取是否成功...")
        obs = rs_env.step()
        image_rgb = obs["rgb"]
        save_check_image(image_rgb, prefix="pick_check", object_name=pick_name, save_dir=SAVE_DIR)
        return check_grasp_success_vllm(image_rgb, pick_name)

    elif CHECK_PICK_SUCCESS_MODE == 2:
        print("[检测] 跳过 pick 检测")
        return True

    else:
        print("[检测] 人工检测 pick 是否成功")
        while True:
            key = input("Pick 成功? (y/n): ").strip().lower()
            if key == 'y':
                return True
            elif key == 'n':
                return False


def do_check_place_success(rs_env, pick_name, place_name):
    """调用 VLM 检测 place 是否成功"""

    if CHECK_PLACE_SUCCESS_MODE == 1:
        print("[检测] 检查放置是否成功...")
        obs = rs_env.step()
        image_rgb = obs["rgb"]
        save_check_image(image_rgb, prefix="place_check", object_name=pick_name,
                         container_name=place_name, save_dir=SAVE_DIR)
        return check_place_success_vllm(image_rgb, pick_name, place_name)

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
# 主线程调度逻辑（两段式）
# ═══════════════════════════════════════════════════

def run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base):
    """
    执行单个 pick-and-place 任务（两段式调度）。

    每个 PICK / PLACE 阶段拆分为：
    - 子阶段 1 APPROACH: 三线程协作（感知✅ 规划✅ 执行✅）
    - 子阶段 2 CRITICAL: 主线程阻塞执行（感知⛔ 规划⛔）
    - 结果检测 CHECKING
    """

    pick_success = False
    place_success = False

    # ════════════════════════════════════════════
    # PICK 阶段
    # ════════════════════════════════════════════
    for attempt in range(MAX_PICK_RETRIES):
        print(f"\n🎯 PICK attempt {attempt + 1}/{MAX_PICK_RETRIES}: {task['pick']}")

        # ─────────────────────────────────────
        # 子阶段 1: APPROACH（到达 pre_pick）
        #   感知✅ 规划✅ 执行✅
        # ─────────────────────────────────────
        with state.lock:
            state.current_task = task
            state.task_phase = TaskPhase.PICK
            state.system_mode = SystemMode.TRACKING
            state.target_name = task['pick']
            state.is_first_point = True
            state.last_stable_point_3d = None
            state.plan_stage = "APPROACH"
            state.action_list = []
            state.action_index = 0
            state.execution_done.clear()
            state.abort_execution.clear()
            state.plan_ready.clear()
            state.need_replan.clear()
            state.pause_perception.clear()      # 感知线程活跃

        # 等待 APPROACH 段执行完毕（到达 pre_pick）
        print("[主线程] 等待 APPROACH 完成...")
        state.execution_done.wait()
        print("[主线程] 📍 已到达 pre_pick 位置")

        # ─────────────────────────────────────
        # 子阶段 2: CRITICAL（下压 → 闭合 → 提起）
        #   感知⛔ 规划⛔ 仅主线程执行
        # ─────────────────────────────────────
        with state.lock:
            state.system_mode = SystemMode.EXECUTING
            state.pause_perception.set()        # ⛔ 暂停感知线程

            # 基于感知线程最后一次打点结果计算抓取位姿
            target_T = state.latest_target_T
            if target_T is None:
                print("[主线程] ⚠️ 没有有效的目标位姿，跳过")
                continue
            target_T = target_T.copy()

        # 计算抓取位姿
        grasp_T = compute_grasp_T(target_T, home_T_tcp2base)
        pick_xyzrpy = realman_xyzrpy_from_T(grasp_T)
        post_pick_T = make_lift_T(grasp_T, lift_z=SAFE_HEIGHT, lift_y=0.03)
        post_pick_xyzrpy = realman_xyzrpy_from_T(post_pick_T)

        critical_actions = [
            {"pose": pick_xyzrpy,       "gripper": GRIPPER_OPEN,  "wait": 0.0},   # 下压到物体位置
            {"pose": pick_xyzrpy,       "gripper": GRIPPER_CLOSE, "wait": 0.5},   # 闭合夹爪
            {"pose": post_pick_xyzrpy,  "gripper": GRIPPER_CLOSE, "wait": 0.0},   # 提起
        ]

        print("[主线程] 🤏 执行 CRITICAL 序列（下压→闭合→提起）...")
        execute_critical_sequence(env, critical_actions)
        print("[主线程] 🤏 CRITICAL 执行完毕")

        # ─────────────────────────────────────
        # 结果检测: CHECKING
        # ─────────────────────────────────────
        with state.lock:
            state.system_mode = SystemMode.CHECKING
            # 感知线程仍然暂停

        pick_success = do_check_pick_success(rs_env, task['pick'])

        if pick_success:
            print("✅ Pick 成功!")
            break
        else:
            print("❌ Pick 失败, 重试...")
            env.step({"gripper": GRIPPER_OPEN})
            time.sleep(0.5)
            continue

    if not pick_success:
        print("⛔ Pick 多次重试失败，跳过此任务")
        return False

    # ════════════════════════════════════════════
    # PLACE 阶段
    # ════════════════════════════════════════════
    for attempt in range(MAX_PLACE_RETRIES):
        print(f"\n📦 PLACE attempt {attempt + 1}/{MAX_PLACE_RETRIES}: {task['place']}")

        # ─────────────────────────────────────
        # 子阶段 1: APPROACH（到达 pre_place）
        #   感知✅ 规划✅ 执行✅
        # ─────────────────────────────────────
        with state.lock:
            state.task_phase = TaskPhase.PLACE
            state.system_mode = SystemMode.TRACKING
            state.target_name = task['place']
            state.is_first_point = True
            state.last_stable_point_3d = None
            state.plan_stage = "APPROACH"
            state.action_list = []
            state.action_index = 0
            state.execution_done.clear()
            state.abort_execution.clear()
            state.plan_ready.clear()
            state.need_replan.clear()
            state.pause_perception.clear()      # 感知线程恢复

        print("[主线程] 等待 APPROACH 完成...")
        state.execution_done.wait()
        print("[主线程] 📍 已到达 pre_place 位置")

        # ─────────────────────────────────────
        # 子阶段 2: CRITICAL（下放 → 松开 → 抬起）
        #   感知⛔ 规划⛔ 仅主线程执行
        # ─────────────────────────────────────
        with state.lock:
            state.system_mode = SystemMode.EXECUTING
            state.pause_perception.set()
            target_T = state.latest_target_T
            if target_T is None:
                print("[主线程] ⚠️ 没有有效的目标位姿，跳过")
                continue
            target_T = target_T.copy()

        # 计算放置位姿（使用 pick 时的抓取旋转，保持物体姿态）
        place_T = compute_grasp_T(target_T, home_T_tcp2base)
        place_T = make_lift_T(place_T, lift_z=0.13)     # 放置高度偏移
        place_xyzrpy = realman_xyzrpy_from_T(place_T)

        post_place_T = make_lift_T(place_T, lift_z=SAFE_HEIGHT)
        post_place_xyzrpy = realman_xyzrpy_from_T(post_place_T)

        critical_actions = [
            {"pose": place_xyzrpy,      "gripper": GRIPPER_CLOSE, "wait": 0.0},   # 下放
            {"pose": place_xyzrpy,      "gripper": GRIPPER_OPEN,  "wait": 0.5},   # 松开夹爪
            {"pose": post_place_xyzrpy, "gripper": GRIPPER_OPEN,  "wait": 0.0},   # 抬起
        ]

        print("[主线程] 📤 执行 CRITICAL 序列（下放→松开→抬起）...")
        execute_critical_sequence(env, critical_actions)
        print("[主线程] 📤 CRITICAL 执行完毕")

        # ─────────────────────────────────────
        # 结果检测
        # ─────────────────────────────────────
        with state.lock:
            state.system_mode = SystemMode.CHECKING

        place_success = do_check_place_success(rs_env, task['pick'], task['place'])

        if place_success:
            print("✅ Place 成功!")
            break
        else:
            print("❌ Place 失败, 从 pick 重新开始...")
            # Place 失败 → reset → 重新从 pick 开始
            env.step({"gripper": GRIPPER_OPEN})
            time.sleep(0.3)

            with state.lock:
                state.system_mode = SystemMode.RESETTING
                state.pause_perception.set()

            print("[主线程] 🔄 机械臂 Reset...")
            env.reset()

            # 递归重试整个任务
            return run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base)

    if not place_success:
        print("⛔ Place 多次重试失败")
        return False

    return True


def run_all_tasks(state, env, rs_env, cam_results, task_list, home_T_tcp2base):
    """执行所有任务"""

    for i, task in enumerate(task_list):
        print(f"\n{'='*60}")
        print(f"🚀 Task [{i+1}/{len(task_list)}]: pick={task['pick']} → place={task['place']}")
        print(f"{'='*60}")

        success = run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base)

        if not success:
            print(f"⛔ Task [{i}] 失败，终止")
            break

        # === Place 成功后的处理 ===
        if i + 1 < len(task_list):
            next_task = task_list[i + 1]

            # 在 reset 之前，先设置下一个任务的感知目标
            # 这样在 reset 的阻塞时间内，感知线程已经在为下一个任务打点
            with state.lock:
                state.task_phase = TaskPhase.PICK
                state.system_mode = SystemMode.TRACKING
                state.target_name = next_task['pick']
                state.is_first_point = True
                state.last_stable_point_3d = None
                state.pause_perception.clear()      # 恢复感知，开始打点

            print("[主线程] 🔄 机械臂 Reset 中（感知线程已提前启动下一任务）...")
            env.reset()
        else:
            print("[主线程] 🔄 最后一个任务完成，Reset...")
            env.reset()

    print("\n🎉 所有任务完成!")


# ═══════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════

def main():

    # ============================
    # 1. 初始化环境
    # ============================
    env = RealmanEnv(robot_ip="192.168.101.19", mode="sync")

    rs_env = Open3dRealsenseEnv("f1471338")

    cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"
    with open(cam_results_path, "r") as f:
        cam_results = json.load(f)

    # ============================
    # 2. 初始化 curobo（TODO）
    # ============================
    # curobo_planner = init_curobo(robot_config_path)
    # curobo_planner.warmup()
    curobo_planner = None  # TODO: 替换为实际的 curobo 规划器

    # ============================
    # 3. 获取初始位姿
    # ============================
    env.reset()

    robot_state = env.get_state()
    home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)
    print(f"[初始化] Home 位姿: {np.round(robot_state.pose, 4)}")

    exit(0)
    # ============================
    # 4. 指令拆解
    # ============================
    instruction = "Pick the white ball and place it on the blue plate."
    print(f"\n[任务] 指令: {instruction}")
    print("[任务] 调用 VLM 拆解指令...")

    task_plan = parse_multi_pick_place_tasks(instruction)
    task_list = task_plan["tasks"]

    print(f"[任务] 📋 共 {len(task_list)} 个任务:")
    for i, t in enumerate(task_list):
        print(f"  [{i+1}] pick={t['pick']} → place={t['place']}")

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
            args=(state, rs_env, cam_results, home_T_tcp2base),
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
        time.sleep(0.5)
        print("[清理] 关闭环境...")
        env.close()
        print("[完成] 程序退出")


if __name__ == "__main__":
    main()
