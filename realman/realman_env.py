"""
重新封装的realman环境，目标是充分利用realman api已有的函数，并提供一个统一的接口，方便使用。
希望具备以下功能：
1.支持pose和joint两种动作模式: 已实现
2.支持相对位置和绝对位置两种控制模式；
3.支持夹爪闭合程度控制；
4.支持避障，自碰撞检测和路径规划功能，通过curobo实现；
5.支持急停功能；
6.later。
"""

import time
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import active_matrix_from_angle

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_thread_mode_e,
    rm_peripheral_read_write_params_t,
)

JOINT_MAX_SPEED_DEG_S = 75.0
SYNC_MOVEJ_SPEED_PERCENT = 80
SYNC_MOVEP_SPEED_PERCENT = 80
SYNC_MOVEL_SPEED_PERCENT = 60
GRIPPER_SPEED = 30
GRIPPER_TOLERANCE = 0.001
GRIPPER_TIMEOUT_S = 2.0

# =========================
# Utils
# =========================

# 将 RealMan 的 xyzrpy (位置+欧拉角) 转换为 4x4 变换矩阵
def T_from_realman_xyzrpy(xyzrpy):
    x, y, z, rx, ry, rz = xyzrpy

    T = np.eye(4)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx  # 先绕 x轴旋转 再绕y轴旋转 最后绕z轴旋转
    T[:3, 3] = [x, y, z]
    return T

# 将 4x4 变换矩阵转换为 RealMan 的 xyzrpy
def realman_xyzrpy_from_T(T):
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    ry = np.arcsin(np.clip(-T[2, 0], -1, 1))
    if np.cos(ry) != 0:
        rx = np.arctan2(T[2, 1]/np.cos(ry), T[2, 2]/np.cos(ry))
        rz = np.arctan2(T[1, 0]/np.cos(ry), T[0, 0]/np.cos(ry))
    else:
        rx = 0
        rz = np.arctan2(-T[0, 1], T[1, 1])
    return np.array([x, y, z, rx, ry, rz])

# TCP 到 RealMan 末端执行器 EEF 的变换矩阵
T_TCP2REALMANEEF = transform_from(
    active_matrix_from_angle(2, -np.pi / 3) @ np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ]),
    np.array([0, 0, 0.22])  
)

T_TCP2REALMANEEF_INV = np.linalg.inv(T_TCP2REALMANEEF)  # 预缓存逆矩阵

# 将末端执行器 EEF xyzrpy 转换为夹爪中心 TCP xyzrpy
def pose_eef2tcp(pose_eef: np.ndarray) -> np.ndarray:
    T_eef2base = T_from_realman_xyzrpy(pose_eef)
    T_tcp2base = T_eef2base @ T_TCP2REALMANEEF
    pose_tcp = realman_xyzrpy_from_T(T_tcp2base)
    return pose_tcp

# 将夹爪中心 TCP xyzrpy 转换为末端执行器 EEF xyzrpy
def pose_tcp2eef(pose_tcp: np.ndarray) -> np.ndarray:
    T_tcp2base = T_from_realman_xyzrpy(pose_tcp)
    T_eef2base = T_tcp2base @ np.linalg.inv(T_TCP2REALMANEEF)
    pose_eef = realman_xyzrpy_from_T(T_eef2base)
    return pose_eef

# 将夹爪宽度(m)转换为 RealMan 夹爪值
def realman_gripper_value_from_width(width: float) -> int:
    return int(9000 - int(width * 1e5))

# 将 RealMan 夹爪值转换为夹爪宽度(m)
def width_from_realman_gripper_value(gripper_value: int) -> float:
    return (9000 - gripper_value) * 1e-5


# =========================
# State
# =========================

@dataclass
class RobotState:
    """机器人状态快照"""
    pose: np.ndarray        # xyzrpy 夹爪中心 TCP 位姿
    joint: np.ndarray       # 关节角度
    gripper: float          # 夹爪开度(单位: 米)
    timestamp: float        # 时间戳


# =========================
# Driver(只做 SDK 封装)
# =========================
class RealmanDriver:
    """
    RealMan 机械臂底层驱动封装(Driver Layer)
    
    注意：该层的 pose 均为末端执行器 EEF 位姿
    """
    def __init__(self, robot_ip: str):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        # 创建机械臂连接
        handle = self.arm.rm_create_robot_arm(robot_ip, 8080)
        assert handle.id > 0, f"连接失败: {robot_ip}"

        # 设置 Modbus
        self.arm.rm_set_modbus_mode(1, 115200, 2)
        time.sleep(0.5)     # 等待设备就绪

        # 设置夹爪速度（寄存器 260，1=最慢，100=最快）
        param = rm_peripheral_read_write_params_t(1, 260, 1)
        self.arm.rm_write_single_register(param, GRIPPER_SPEED)

        # 限制速度，避免危险动作
        self.arm.rm_set_arm_max_line_speed(0.4)
        self.arm.rm_set_arm_max_line_acc(1.0)
        self.arm.rm_set_arm_max_angular_speed(0.4)
        self.arm.rm_set_arm_max_angular_acc(1.0)

        # 限制关节速度，避免危险动作
        for joint_idx in range(1, 8):
            self.arm.rm_set_joint_max_speed(joint_idx, JOINT_MAX_SPEED_DEG_S)

    # =========================
    # 机械臂运动控制
    # =========================

    def movej(self, joint):
        """
        关节空间运动(Joint Control, 阻塞)

        Args:
            joint: 目标关节角(单位: 度, 7维)

        Returns:
            ret: SDK 返回码(0 表示成功)
        """
        ret = self.arm.rm_movej(joint, SYNC_MOVEJ_SPEED_PERCENT, 0, 0, 1)
        return ret

    def _move_pose_with_retry(self, pose, *, linear: bool):
        move_fn = self.arm.rm_movel if linear else self.arm.rm_movej_p
        speed_percent = SYNC_MOVEL_SPEED_PERCENT if linear else SYNC_MOVEP_SPEED_PERCENT
        blend_radius = 0 if linear else 1

        ret = move_fn(pose, speed_percent, r=blend_radius, connect=0, block=1)
        if ret == 0:
            return ret

        for _ in range(100):
            ret = move_fn(pose, speed_percent, r=blend_radius, connect=0, block=1)
            if ret == 0:
                return ret
            time.sleep(0.02)

        return ret

    def movep(self, pose):
        """
        笛卡尔空间运动(Pose Control, 阻塞)

        Args:
            pose: xyzrpy (6维) 末端执行器 EEF 位姿

        Returns:
            ret: SDK 返回码
        """
        ret = self._move_pose_with_retry(pose, linear=False)
        if ret == 0:
            return ret
        if ret != 0:
            for i in range(100):
                ret = self.arm.rm_movej_p(pose, SYNC_MOVEP_SPEED_PERCENT, r=1, connect=0, block=1)
                if ret == 0:
                    print(f"movep 第 {i+1} 次才解出来。")
                    return ret
                time.sleep(0.02)
            print("================================================")
            print(f"pose: {pose}")
            print(f"movep挂了，解了{i}次解不出来。")
            print("================================================")
            return ret

    def movel(self, pose):
        """Cartesian linear motion (blocking)."""
        ret = self._move_pose_with_retry(pose, linear=True)
        if ret != 0:
            print("================================================")
            print(f"pose: {pose}")
            print("movel failed after retries.")
            print("================================================")
        return ret

    def movej_follow(self, joint):
        """关节空间跟随运动(用于异步流式控制)"""
        return self.arm.rm_movej_follow(joint)

    def movep_follow(self, pose):
        """笛卡尔空间跟随运动(用于异步流式控制)"""
        return self.arm.rm_movep_follow(pose)

    def slow_stop(self) -> int:
        """
        轨迹缓停：沿当前规划轨迹减速停止（SDK: rm_set_arm_slow_stop）。

        Returns:
            SDK 返回码，0 表示成功。
        """
        return self.arm.rm_set_arm_slow_stop()

    def emergency_stop(self) -> int:
        """
        轨迹急停：关节最快速度停止，当前轨迹不可恢复（SDK: rm_set_arm_stop）。

        与 rm_set_arm_emergency_stop（四代控制器急停状态）不同，此为运动急停。

        Returns:
            SDK 返回码，0 表示成功。
        """
        return self.arm.rm_set_arm_stop()

    # =========================
    # 状态获取
    # =========================

    def get_state(self):
        """
        获取当前机械臂状态(一次性读取)

        Returns:
            dict:
                - pose: xyzrpy (6维) 末端执行器 EEF 位姿
                - joint: 弧度制关节角

            或 None(通信失败)

        注意:
        - 不做缓存,每次都访问 SDK
        - 可能较慢(~10-50ms)
        - 上层应避免高频直接调用(用 async cache)

        推荐:
        - SyncController:直接调用
        - AsyncController:放到 state_loop
        """
        ret, state = self.arm.rm_get_current_arm_state()
        if ret == 0:
            return {
                "pose": np.array(state["pose"]),
                "joint": np.radians(state["joint"]),
            }
        elif ret ==1: 
            return {
                "pose": None,
                "joint": None,
            }
        elif ret == -1 or ret == -2:
            # 通信问题，重试20次
            for i in range(500):
                ret, state = self.arm.rm_get_current_arm_state()
                if ret == 0:
                    print(f"get_state 第 {i+1} 次才读出来。")
                    return {
                        "pose": np.array(state["pose"]),
                        "joint": np.radians(state["joint"]),
                    }
                time.sleep(0.05)
            print("================================================")
            print(f"通信挂了，读了{i+1}次读不出来。")
            print("================================================")
            return None
        """
        if ret != 0 or state is None:
            return None
        return {
            "pose": np.array(state["pose"]),
            "joint": np.radians(state["joint"]),
        }
        """

    # =========================
    # 夹爪控制（Modbus）
    # =========================

    def set_gripper(self, width, wait=True, timeout=GRIPPER_TIMEOUT_S):
        """
        设置夹爪开度

        Args:
            width: 夹爪宽度(单位: 米)
            wait: 是否等待夹爪到位
            timeout: 等待超时时间(秒)

        内部流程：
        1. 宽度 → 寄存器值
        2. 写入目标位置寄存器(258)
        3. 触发执行(264)

        注意:
        - 通过读取当前位置轮询夹爪是否到位
        """
        current_width = self.get_gripper()
        if abs(current_width - width) < GRIPPER_TOLERANCE:
            return

        value = realman_gripper_value_from_width(width)

        # 写入目标位置寄存器(258)
        param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
        self.arm.rm_write_registers(param, [0, value, 0, 0])

        # 触发执行(264)
        param = rm_peripheral_read_write_params_t(1, 264, 1)
        self.arm.rm_write_single_register(param, 1)

        if not wait:
            return

        start_time = time.time()
        while time.time() - start_time < timeout:
            current_width = self.get_gripper()
            if abs(current_width - width) < GRIPPER_TOLERANCE:
                return
            time.sleep(0.02)

    def get_gripper(self):
        """
        获取夹爪当前开度

        Returns:
            width(单位: 米)

        注意:
        - 读取寄存器 259
        - 可能存在延迟或读取失败
        - 上层可做缓存(用 async cache)
        """
        param = rm_peripheral_read_write_params_t(1, 259, 1)
        ret, val = self.arm.rm_read_holding_registers(param)

        if ret == 0:
            return width_from_realman_gripper_value(val)
        
        # fallback value
        return 0.09

    # =========================
    # 资源释放
    # =========================

    def close(self):
        """
        关闭机械臂连接

        必须调用:
        - 释放 SDK 资源
        - 防止连接泄露
        """
        self.arm.rm_delete_robot_arm()


# =========================
# Sync Controller（阻塞）
# =========================

class SyncController:
    """
    同步控制器(Blocking Controller)

    设计目标:
    - 提供“调用即执行”的控制接口(类似 gym env)
    - 每个 step 是一个完整的闭环:
        action → 执行 → 读取状态 → 返回

    特点:
    - 阻塞式(blocking)
    - 控制频率低(~5-20Hz)
    - 行为确定(适合 RL / debug)

    不适合:
    - 高频控制
    - 轨迹跟踪
    - 遥操作

    类比:
    Gym Environment 的 step()
    """

    def __init__(self, driver: RealmanDriver):
        """
        Args:
            driver: 底层机器人驱动(只负责执行命令)
        """
        self.driver = driver

    def step(self, action: dict) -> RobotState:
        """
        执行动作(阻塞)

        Args:
            action:
                - "joint": 关节角
                - "pose": 夹爪中心 TCP 位姿(xyzrpy 6维)
                - "delta_pose": TCP 位姿增量(xyzrpy 6维)
                - "gripper": 夹爪开度

        Returns:
            RobotState(执行后的状态, pose 为夹爪中心 TCP 位姿, xyzrpy 6维)

        注意:
        - joint / pose 动作会阻塞到控制器返回完成
        - 每次调用都会访问真实机器人(较慢)
        - 不做频率控制

        使用场景:
        - RL training(低频)
        - 单步调试(debug)
        """
        move_ret = None

        if "joint" in action:
            move_ret = self.driver.movej(action["joint"])
            
        elif "pose" in action:
            pose_eef = pose_tcp2eef(action["pose"])     # 将上层的夹爪中心 TCP xyzrpy 转换为末端执行器 EEF xyzrpy, 传入 realman driver
            motion_type = action.get("motion", "pose")
            if motion_type == "linear":
                move_ret = self.driver.movel(pose_eef)
            else:
                move_ret = self.driver.movep(pose_eef)

        elif "delta_pose" in action:
            # 获取当前 TCP 位姿
            current_state = self.get_state()
            current_pose = current_state.pose   # TCP xyzrpy

            delta_pose = action["delta_pose"]

            # 转矩阵
            T_current = T_from_realman_xyzrpy(current_pose)
            T_delta = T_from_realman_xyzrpy(delta_pose)

            # 计算目标位姿矩阵
            T_target = T_current @ T_delta

            # 转 xyzrpy
            target_pose = realman_xyzrpy_from_T(T_target)

            # 转 EEF xyzrpy
            pose_eef = pose_tcp2eef(target_pose)

            # 执行运动
            motion_type = action.get("motion", "pose")
            if motion_type == "linear":
                move_ret = self.driver.movel(pose_eef)  # EEF xyzrpy
            else:
                move_ret = self.driver.movep(pose_eef)  # EEF xyzrpy
            
        state = self.get_state()

        if move_ret not in (None, 0):
            raise RuntimeError(f"机器人运动失败，ret={move_ret}")

        if "gripper" in action:
            wait_gripper = action.get("wait_gripper", True)
            self.driver.set_gripper(action["gripper"], wait=wait_gripper)
            state = self.get_state()

        return state

    def get_state(self) -> RobotState:
        """
        获取当前状态(同步读取)

        Returns:
            RobotState(执行后的状态, pose 为夹爪中心 TCP 位姿, xyzrpy 6维)

        注意:
        - 每次都会访问 SDK(慢)
        - 无缓存
        """
        s = self.driver.get_state()
        pose_tcp = pose_eef2tcp(s["pose"])

        return RobotState(
            pose=pose_tcp,
            joint=s["joint"],
            gripper=self.driver.get_gripper(),
            timestamp=time.time(),
        )

    def reset(self):
        """
        复位机械臂到默认姿态

        流程:
        1. 持续发送目标关节角
        2. 检查误差是否收敛
        3. 设置夹爪

        注意:
        - 简单 polling 实现(非最优)
        - 阻塞时间较长(最多 ~5s)

        适合:
        - 重置环境
        """
        target_joint = np.array([90, 0, 0, -90, 0, -90, 60])
        target_joint_rad = np.radians(target_joint)

        self.driver.movej(target_joint)

        start_time = time.time()

        # 等待关节角度收敛
        while True:
            state = self.driver.get_state()

            if state is None:
                time.sleep(0.02)
                continue

            err = np.linalg.norm(state["joint"] - target_joint_rad)

            # 收敛
            if err < 0.1:
                break

            # 超时保护
            if time.time() - start_time > 5:
                print(f"[SyncController] reset 超时，当前误差: {err:.4f}")
                break

            time.sleep(0.02)

        self.driver.set_gripper(0.09, wait=True)

        time.sleep(0.2)

        return self.get_state()

    def slow_stop(self) -> int:
        """轨迹缓停（与 RealmanDriver.slow_stop 一致）。"""
        return self.driver.slow_stop()

    def emergency_stop(self) -> int:
        """轨迹急停（与 RealmanDriver.emergency_stop 一致）。"""
        return self.driver.emergency_stop()


# =========================
# Async Controller（流式控制）
# =========================
class AsyncController:

    def __init__(self, driver: RealmanDriver, min_interval=0.02):
        """
        Args:
            driver: 底层驱动
            min_interval: 最小指令发送间隔(防止SDK堵塞)

        初始化:
        - 命令缓存
        - 状态缓存
        - 启动后台线程(控制线程和状态线程)
        """
        self.driver = driver
        self.min_interval = min_interval

        # 线程控制
        self._stop_event = threading.Event()

        # 状态缓存
        self._state_lock = threading.Lock()
        self._latest_state: Optional[RobotState] = None
        self._cached_gripper_for_state = 1.0  # 夹爪缓存 (0=全闭, 1=全开) TODO:这里要检查

        # 命令缓存
        self._cmd_lock = threading.Lock()
        self._pending_pose: Optional[np.ndarray] = None
        self._pending_joint: Optional[np.ndarray] = None
        self._pending_gripper: Optional[float] = None
        self._last_cmd_time: float = 0

        # 统计
        self._stats_lock = threading.Lock()
        self._cmd_count = 0
        self._cmd_latency_sum = 0.0
        self._state_count = 0
        self._last_stats_time = time.time()
        self._state_rate = 0.0
        self._avg_latency = 0.0
        
        # 启动后台线程(控制线程和状态线程)
        self._cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._cmd_thread.start()
        self._state_thread.start()

        # 等待首次状态
        for _ in range(50):
            if self._latest_state is not None:
                break
            time.sleep(0.02)

    def _update_stats(self):
        """更新统计信息"""
        now = time.time()
        with self._stats_lock:
            dt = now - self._last_stats_time
            if dt >= 1.0:
                self._state_rate = self._state_count / dt
                if self._cmd_count > 0:
                    self._avg_latency = (self._cmd_latency_sum / self._cmd_count) * 1000
                self._state_count = 0
                self._cmd_count = 0
                self._cmd_latency_sum = 0
                self._last_stats_time = now

    # -------- 状态线程 --------
    def _state_loop(self):
        """状态读取线程"""
        # 缓存的夹爪值 (从命令中更新)
        self._cached_gripper_for_state = 1.0

        while not self._stop_event.is_set():
            try:
                state = self.driver.get_state()

                if state is not None:
                    robot_state = RobotState(
                        pose=pose_eef2tcp(state["pose"]),
                        joint=state["joint"],
                        gripper=self._cached_gripper_for_state,
                        timestamp=time.time(),
                    )
                    with self._state_lock:
                        self._latest_state = robot_state
                    with self._stats_lock:
                        self._state_count += 1
                else:
                    # 通信失败时增加延迟
                    time.sleep(0.05)
                    continue
                
                # 状态读取频率 (低于命令频率, 避免与 _cmd_loop 抢 SDK 带宽)
                time.sleep(0.04)  # 25 Hz
                
            except:
                # 通信失败时增加延迟
                time.sleep(0.1)

    # -------- 控制线程 --------
    def _cmd_loop(self):
        """命令发送线程"""
        while not self._stop_event.is_set():
            try:
                with self._cmd_lock:
                    pose_tcp = self._pending_pose
                    joint = self._pending_joint
                    gripper = self._pending_gripper
                    self._pending_pose = None
                    self._pending_joint = None
                    self._pending_gripper = None
                
                if pose_tcp is None and joint is None and gripper is None:
                    time.sleep(0.005)
                    continue

                # 最小发送间隔
                now = time.time()
                elapsed = now - self._last_cmd_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                    
                start = time.time()

                if joint is not None:
                    ret = self.driver.movej_follow(joint)
                    if ret != 0 and self._cmd_count % 50 == 0:
                        print(f"[AsyncController] movej_follow 失败: ret={ret}")
                
                elif pose_tcp is not None:
                    pose_eef = pose_tcp2eef(pose_tcp)
                    ret = self.driver.movep_follow(pose_eef)
                    if ret != 0 and self._cmd_count % 50 == 0:
                        print(f"[AsyncController] movep_follow 失败: ret={ret}")
                
                if gripper is not None:
                    ret = self.driver.set_gripper(gripper, wait=False)
                    if ret != 0 and self._cmd_count % 50 == 0:
                        print(f"[AsyncController] set_gripper 失败: ret={ret}")
                    self._cached_gripper_for_state = gripper

                latency = time.time() - start
                self._last_cmd_time = time.time()
                
                with self._stats_lock:
                    self._cmd_count += 1
                    self._cmd_latency_sum += latency
                    
            except Exception as e:
                time.sleep(0.01)

     # -------- 用户接口 --------
    
    # -------- 用户接口 --------
    def send_joint(self, joint):
        """
        发送关节目标(非阻塞)

        注意:
        - 不立即执行
        - 会覆盖旧命令

        用于:
        - 轨迹跟踪
        """
        with self._cmd_lock:
            self._pending_joint = joint.copy()

    def send_pose(self, pose_tcp):
        """
        发送位姿目标（非阻塞）
        接收的是夹爪中心 TCP 位姿的 xyzrpy 6维, 转为末端执行器 EEF 位姿的 xyzrpy 6维
        """
        with self._cmd_lock:
            self._pending_pose = pose_tcp.copy()

    def send_gripper(self, gripper):
        """发送夹爪命令（非阻塞）"""
        with self._cmd_lock:
            self._pending_gripper = gripper

    def slow_stop(self) -> int:
        """轨迹缓停；清空待发送指令，避免停止后控制线程继续下发。"""
        with self._cmd_lock:
            self._pending_joint = None
            self._pending_pose = None
            self._pending_gripper = None
        return self.driver.slow_stop()

    def emergency_stop(self) -> int:
        """轨迹急停；清空待发送指令。"""
        with self._cmd_lock:
            self._pending_joint = None
            self._pending_pose = None
            self._pending_gripper = None
        return self.driver.emergency_stop()

    def get_state(self):
        """
        获取缓存状态(非阻塞)

        Returns:
            RobotState or None

        注意:
        - 可能是旧数据(延迟 ~20ms)
        """
        self._update_stats()
        return self._latest_state

    def stop(self):
        """
        停止后台线程

        注意:
        - 必须在程序退出时调用
        """
        self._stop_event.set()


# =========================
# Env（最终接口）
# =========================

class RealmanEnv:
    """
    机器人环境(统一接口层)

    作用:
    - 对外提供统一 API
    - 封装 sync / async 控制模式
    
    类比:
    - Gym Env + Robot Runtime Wrapper
    """

    def __init__(self, robot_ip, mode="sync"):
        """
        Args:
            robot_ip: 机械臂 IP
            mode: "sync" 或 "async"

        初始化：
        - 创建 driver
        - 选择 controller
        """
        self.driver = RealmanDriver(robot_ip)

        if mode == "async":
            self.ctrl = AsyncController(self.driver)
            self.mode = "async"
        else:
            self.ctrl = SyncController(self.driver)
            self.mode = "sync"

    # -------- 同步接口 --------

    def step(self, action):
        if self.mode != "sync":
            raise RuntimeError("async 模式下不能用 step")
        return self.ctrl.step(action)

    def reset(self):
        if self.mode != "sync":
            raise RuntimeError("async 模式下不能用 reset")
        return self.ctrl.reset()

    # -------- 异步接口 --------
    
    def send_joint(self, joint):
        assert self.mode == "async"
        self.ctrl.send_joint(joint)

    def send_pose(self, pose):
        assert self.mode == "async"
        self.ctrl.send_pose(pose)

    def send_gripper(self, g):
        assert self.mode == "async"
        self.ctrl.send_gripper(g)

    def slow_stop(self) -> int:
        """轨迹缓停（sync / async 均可用）。"""
        return self.ctrl.slow_stop()

    def emergency_stop(self) -> int:
        """轨迹急停（sync / async 均可用）。"""
        return self.ctrl.emergency_stop()

    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        with self.ctrl._stats_lock:
            return {
                "async_mode": self.mode == "async",
                "state_update_rate": self.ctrl._state_rate,
                "avg_latency_ms": self.ctrl._avg_latency,
            }

    # -------- 状态 --------
    def get_state(self):
        """
        获取当前状态

        Returns:
            RobotState or None

        注意:
        - sync: 实时
        - async: 缓存
        """
        return self.ctrl.get_state()

    def close(self):
        """
        关闭环境

        顺序:
        1. 停止线程(async)
        2. 释放硬件连接
        """
        if self.mode == "async":
            self.ctrl.stop()
        self.driver.close()
