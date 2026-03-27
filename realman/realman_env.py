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

        # 设置夹爪速度（寄存器 260）
        param = rm_peripheral_read_write_params_t(1, 260, 1)
        self.arm.rm_write_single_register(param, 1)

        # 限制速度，避免危险动作
        self.arm.rm_set_arm_max_line_speed(0.1)
        self.arm.rm_set_arm_max_line_acc(0.5)
        self.arm.rm_set_arm_max_angular_speed(0.5)
        self.arm.rm_set_arm_max_angular_acc(1.0)

    # =========================
    # 机械臂运动控制
    # =========================

    def movej(self, joint):
        """
        关节空间运动(Joint Control)

        Args:
            joint: 目标关节角(单位: 度, 7维)

        Returns:
            ret: SDK 返回码(0 表示成功)
        """
        return self.arm.rm_movej_follow(joint)

    def movep(self, pose):
        """
        笛卡尔空间运动(Pose Control)

        Args:
            pose: xyzrpy (6维) 末端执行器 EEF 位姿

        Returns:
            ret: SDK 返回码
        """
        return self.arm.rm_movep_follow(pose)

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
        if ret != 0 or state is None:
            return None
        return {
            "pose": np.array(state["pose"]),
            "joint": np.radians(state["joint"]),
        }

    # =========================
    # 夹爪控制（Modbus）
    # =========================

    def set_gripper(self, width):
        """
        设置夹爪开度

        Args:
            width: 夹爪宽度(单位: 米)

        内部流程：
        1. 宽度 → 寄存器值
        2. 写入目标位置寄存器(258)
        3. 触发执行(264)

        注意:
        - 无返回值(SDK未提供明确执行反馈)
        - 不保证执行成功(可由上层做重试)
        """
        value = realman_gripper_value_from_width(width)

        # 写入目标位置寄存器(258)
        param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
        self.arm.rm_write_registers(param, [0, value, 0, 0])

        # 触发执行(264)
        param = rm_peripheral_read_write_params_t(1, 264, 1)
        self.arm.rm_write_single_register(param, 1)

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
                - "gripper": 夹爪开度

        Returns:
            RobotState(执行后的状态, pose 为夹爪中心 TCP 位姿, xyzrpy 6维)

        注意:
        - 不保证运动完成(取决于 SDK)
        - 每次调用都会访问真实机器人(较慢)
        - 不做频率控制

        使用场景:
        - RL training(低频)
        - 单步调试(debug)
        """
        if "joint" in action:
            self.driver.movej(action["joint"])
        elif "pose" in action:
            pose_eef = pose_tcp2eef(action["pose"])     # 将上层的夹爪中心 TCP xyzrpy 转换为末端执行器 EEF xyzrpy, 传入 realman driver
            self.driver.movep(pose_eef)

        if "gripper" in action:
            self.driver.set_gripper(action["gripper"])

        return self.get_state()

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

        self.driver.movej(target_joint)

        start_time = time.time()

        # 等待关节角度收敛
        while True:
            state = self.driver.get_state()

            if state is None:
                time.sleep(0.02)
                continue

            err = np.linalg.norm(state["joint"] - target_joint)

            # 收敛
            if err < 0.1:
                break

            # 超时保护
            if time.time() - start_time > 5:
                print(f"[SyncController] reset 超时，当前误差: {err:.4f}")
                break

            time.sleep(0.02)

        self.driver.set_gripper(0.09)

        time.sleep(0.2)

        return self.get_state()


# =========================
# Async Controller（流式控制）
# =========================

class AsyncController:
    """
    异步控制器(Streaming Controller)

    设计目标:
    - 支持高频控制(50Hz+)
    - 支持轨迹跟踪(curobo / policy rollout)
    - 控制与状态解耦

    核心思想:
    - 用户线程:只“发送目标”
    - 控制线程:持续执行命令
    - 状态线程:持续更新状态

    本质是一个:
    Producer-Consumer + State Cache 系统(生产者-消费者模式 + 状态缓存系统)

    特点:
    - 非阻塞(non-blocking)
    - 高吞吐(high frequency)
    - 实时性好(但不是严格同步)

    风险:
    - 命令会被覆盖(只执行最新)
    - 状态有延迟(cache)
    """

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

        # 命令缓存(只保留最新)
        self._lock = threading.Lock()
        self._pending_joint = None
        self._pending_pose = None
        self._pending_gripper = None

        # 状态缓存
        self._state = None

        # 线程控制
        self._stop = False

        # 启动后台线程(控制线程和状态线程)
        self._cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)

        self._cmd_thread.start()
        self._state_thread.start()

    # -------- 状态线程 --------
    def _state_loop(self):
        """
        持续读取机器人状态(约50Hz)

        作用:
        - 更新缓存状态
        - 避免主线程频繁访问 SDK

        注意:
        - 状态是"近实时",不是严格同步
        """
        while not self._stop:
            s = self.driver.get_state()
            if s is not None:
                self._state = RobotState(
                    pose=s["pose"],
                    joint=s["joint"],
                    gripper=self.driver.get_gripper(),
                    timestamp=time.time(),
                )
            time.sleep(0.02)

    # -------- 控制线程 --------
    def _cmd_loop(self):
        """
        持续发送控制命令

        流程:
        1. 读取 pending 命令
        2. 清空缓存(避免重复执行)
        3. 控制发送频率
        4. 执行命令

        关键设计:
        - "只执行最新命令"(覆盖机制)
        - 防止命令堆积(低延迟)
        """
        last = 0
        
        while not self._stop:
            with self._lock:
                j = self._pending_joint
                p = self._pending_pose
                g = self._pending_gripper

                self._pending_joint = None
                self._pending_pose = None
                self._pending_gripper = None

            if j is None and p is None and g is None:
                time.sleep(0.005)
                continue

            # 频率控制
            now = time.time()
            if now - last < self.min_interval:
                time.sleep(self.min_interval - (now - last))

            if j is not None:
                self.driver.movej(j)
            elif p is not None:
                self.driver.movep(p)

            # 夹爪并行
            if g is not None:
                self.driver.set_gripper(g)

            last = time.time()  

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
        with self._lock:
            self._pending_joint = joint.copy()

    def send_pose(self, pose):
        """发送位姿目标（非阻塞）"""
        with self._lock:
            self._pending_pose = pose.copy()

    def send_gripper(self, g):
        """发送夹爪命令（非阻塞）"""
        with self._lock:
            self._pending_gripper = g

    def get_state(self):
        """
        获取缓存状态(非阻塞)

        Returns:
            RobotState or None

        注意:
        - 可能是旧数据(延迟 ~20ms)
        """
        return self._state

    def stop(self):
        """
        停止后台线程

        注意:
        - 必须在程序退出时调用
        """
        self._stop = True


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
