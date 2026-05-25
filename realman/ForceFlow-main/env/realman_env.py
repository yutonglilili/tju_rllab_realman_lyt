"""
RealMan Robot Environment (ForceFlow 适配版)

支持两种模式：
1. 同步模式 (默认) - 一次性 movel/movej + 阻塞等待
2. 异步模式 (async_mode=True) - 后台双线程，state 25Hz / cmd 50Hz, 用于高频遥操作

接口约定（全部对齐 SDK 原生格式）：
- 位姿 pose: ndarray shape=(6,) EEF xyzrpy (米 + 弧度), 即 rm_get_current_arm_state() 返回的 state["pose"]
- 夹爪 gripper_open: float ∈ [0, 1], 1=全开 0=全闭
- 关节 joint: ndarray shape=(7,) 弧度 (SDK 用度, env 内部转弧度)

不引入 TCP 偏移：策略学的是 SDK 直接控制的量, 多一道偏移会在推理时引入对齐误差。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_peripheral_read_write_params_t,
    rm_thread_mode_e,
)


# =========================
# 坐标 / 夹爪 工具函数
# =========================

def T_from_realman_xyzrpy(xyzrpy):
    """RealMan 的 xyzrpy → 4x4 齐次矩阵"""
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
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def realman_xyzrpy_from_T(T):
    """4x4 齐次矩阵 → RealMan 的 xyzrpy"""
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    ry = np.arcsin(np.clip(-T[2, 0], -1, 1))
    if np.cos(ry) != 0:
        rx = np.arctan2(T[2, 1] / np.cos(ry), T[2, 2] / np.cos(ry))
        rz = np.arctan2(T[1, 0] / np.cos(ry), T[0, 0] / np.cos(ry))
    else:
        rx = 0
        rz = np.arctan2(-T[0, 1], T[1, 1])
    return np.array([x, y, z, rx, ry, rz])


GRIPPER_REG_MAX = 9000  # 寄存器最大值 (0=全开, 9000=全闭)


def gripper_open_to_reg(gripper_open: float) -> int:
    """夹爪开度 [0,1] → 寄存器值 (0=全开, 9000=全闭)"""
    gripper_open = max(0.0, min(1.0, float(gripper_open)))
    return int(GRIPPER_REG_MAX * (1.0 - gripper_open))


def reg_to_gripper_open(reg_value: int) -> float:
    """寄存器值 (0=全开, 9000=全闭) → 夹爪开度 [0,1]"""
    reg_value = max(0, min(GRIPPER_REG_MAX, int(reg_value)))
    return 1.0 - reg_value / GRIPPER_REG_MAX


def _gripper_reg_to_bytes(value: int) -> list:
    """寄存器值 → 4 字节大端 [MSB, ..., LSB], 给 rm_write_registers 用"""
    value = max(0, min(GRIPPER_REG_MAX, int(value)))
    return [
        (value >> 24) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 8)  & 0xFF,
         value        & 0xFF,
    ]


# =========================
# State
# =========================

@dataclass
class RobotState:
    """机器人状态快照"""
    pose: np.ndarray        # EEF xyzrpy shape=(6,), 米 + 弧度, SDK 原生坐标
    gripper_open: float     # 夹爪开度 [0,1], 1=全开 0=全闭
    joint: np.ndarray       # 关节角度 shape=(7,), 弧度
    timestamp: float


# =========================
# RealmanEnv
# =========================

class RealmanEnv:
    """
    RealMan 机器人环境

    Args:
        robot_ip: 机械臂 IP
        safety_mode: 是否限速 (慢速调试用)
        async_mode: 是否启用异步双线程模式 (遥操作高频)
        min_cmd_interval: 异步模式命令最小间隔(秒)
    """

    def __init__(
        self,
        robot_ip: str = "192.168.101.19",
        safety_mode: bool = False,
        async_mode: bool = False,
        min_cmd_interval: float = 0.02,
    ):
        self.robot_ip = robot_ip
        self.safety_mode = safety_mode
        self.async_mode = async_mode
        self.min_cmd_interval = min_cmd_interval

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.arm.rm_create_robot_arm(robot_ip, 8080)
        assert handle.id > 0, f"机械臂连接失败，检查 IP: {robot_ip}"

        ret = self.arm.rm_set_modbus_mode(1, 115200, 2)
        assert ret == 0, "modbus 设置失败"
        time.sleep(0.5)

        # 夹爪速度
        gripper_speed = 50
        param = rm_peripheral_read_write_params_t(1, 260, 1)
        for attempt in range(3):
            ret = self.arm.rm_write_single_register(param, gripper_speed)
            if ret == 0:
                break
            print(f"[RealmanEnv] 写夹爪速度寄存器失败 (尝试 {attempt + 1}/3)")
            time.sleep(0.3)
        if ret != 0:
            print(f"[RealmanEnv] 警告: 夹爪速度设置失败 ret={ret}")
        else:
            print(f"[RealmanEnv] 夹爪速度: {gripper_speed}/100")

        # 速度上限
        if safety_mode:
            print(f"[RealmanEnv] 安全模式 ({robot_ip})")
            self.arm.rm_set_arm_max_line_speed(0.1)
            self.arm.rm_set_arm_max_line_acc(0.5)
            self.arm.rm_set_arm_max_angular_speed(0.5)
            self.arm.rm_set_arm_max_angular_acc(1.0)
        else:
            print(f"[RealmanEnv] 标准速度上限 ({robot_ip})")
            self.arm.rm_set_arm_max_line_speed(0.5)
            self.arm.rm_set_arm_max_line_acc(2.0)
            self.arm.rm_set_arm_max_angular_speed(2.0)
            self.arm.rm_set_arm_max_angular_acc(3.0)

        self.connected = True
        self._arm_lock = threading.RLock()

        if async_mode:
            self._init_async_mode()

        ret, state = self.arm.rm_get_current_arm_state()
        if ret == 0 and state:
            sys_err = state.get("sys_err", 0)
            if sys_err != 0:
                print(f"[RealmanEnv] 警告: sys_err={sys_err}, 请在示教器上解除")

        print(f"[RealmanEnv] 连接成功: {robot_ip}")

    # ====================================================================
    # 异步模式
    # ====================================================================

    def _init_async_mode(self):
        self._stop_event = threading.Event()

        self._state_lock = threading.Lock()
        self._latest_state: Optional[RobotState] = None
        self._cached_gripper_for_state = 1.0

        self._cmd_lock = threading.Lock()
        self._pending_pose: Optional[np.ndarray] = None
        self._pending_gripper: Optional[float] = None
        self._last_cmd_time: float = 0.0

        self._stats_lock = threading.Lock()
        self._cmd_count = 0
        self._cmd_latency_sum = 0.0
        self._state_count = 0
        self._last_stats_time = time.time()
        self._state_rate = 0.0
        self._avg_latency = 0.0

        self._cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._cmd_thread.start()
        self._state_thread.start()

        for _ in range(50):
            if self._latest_state is not None:
                break
            time.sleep(0.02)

    def _state_loop(self):
        """状态读取线程, ~25Hz, 失败立刻重试一次, 仍失败短睡后下轮再来"""
        self._cached_gripper_for_state = 1.0

        while not self._stop_event.is_set():
            try:
                ret, state = self.arm.rm_get_current_arm_state()
                # 单次瞬时失败先不睡, 立刻再试一次, 能抓回大部分 recv error
                if ret != 0 or state is None:
                    ret, state = self.arm.rm_get_current_arm_state()

                if ret == 0 and state is not None:
                    robot_state = RobotState(
                        pose=np.asarray(state["pose"], dtype=np.float64),
                        gripper_open=self._cached_gripper_for_state,
                        joint=np.radians(state["joint"]),
                        timestamp=time.time(),
                    )
                    with self._state_lock:
                        self._latest_state = robot_state
                    with self._stats_lock:
                        self._state_count += 1
                else:
                    # 两次都失败, 短睡避免狂刷 SDK (原 0.05 → 0.02 更快恢复)
                    time.sleep(0.02)
                    continue

                time.sleep(0.04)  # ~25Hz, 比 cmd 慢避免抢 SDK 带宽
            except Exception:
                time.sleep(0.1)

    def _cmd_loop(self):
        """命令发送线程"""
        while not self._stop_event.is_set():
            try:
                with self._cmd_lock:
                    pose = self._pending_pose
                    gripper = self._pending_gripper
                    self._pending_pose = None
                    self._pending_gripper = None

                if pose is None and gripper is None:
                    time.sleep(0.005)
                    continue

                # 最小发送间隔
                now = time.time()
                elapsed = now - self._last_cmd_time
                if elapsed < self.min_cmd_interval:
                    time.sleep(self.min_cmd_interval - elapsed)

                start = time.time()

                if pose is not None:
                    # pose 已经是 SDK 原生 6 维 EEF xyzrpy
                    ret = self.arm.rm_movep_follow(pose.tolist() if isinstance(pose, np.ndarray) else pose)
                    if ret != 0 and self._cmd_count % 50 == 0:
                        print(f"[RealmanEnv] movep_follow 失败: ret={ret}")

                if gripper is not None:
                    gripper_value = gripper_open_to_reg(gripper)
                    regs = _gripper_reg_to_bytes(gripper_value)
                    param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
                    self.arm.rm_write_registers(param, regs)
                    param = rm_peripheral_read_write_params_t(1, 264, 1)
                    self.arm.rm_write_single_register(param, 1)
                    self._cached_gripper_for_state = gripper

                latency = time.time() - start
                self._last_cmd_time = time.time()

                with self._stats_lock:
                    self._cmd_count += 1
                    self._cmd_latency_sum += latency
            except Exception:
                time.sleep(0.01)

    def _update_stats(self):
        if not self.async_mode:
            return
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

    # ====================================================================
    # 夹爪 (同步)
    # ====================================================================

    def _get_gripper(self, retries: int = 3) -> float:
        """读取夹爪当前开度 [0,1]"""
        for attempt in range(retries):
            try:
                param = rm_peripheral_read_write_params_t(1, 258, 1)
                ret, _ = self.arm.rm_read_holding_registers(param)
                param = rm_peripheral_read_write_params_t(1, 259, 1)
                ret, gripper_value_state = self.arm.rm_read_holding_registers(param)
                if ret == 0:
                    return reg_to_gripper_open(gripper_value_state)
            except Exception:
                if attempt < retries - 1:
                    time.sleep(0.1)
                    continue
        return 1.0  # fallback: 全开

    def _set_gripper(self, gripper_open: float, retries: int = 3) -> bool:
        """设置夹爪位置 [0,1] (带重试)

        寄存器映射:
            258-259: 夹爪目标位置, 32 位大端写入 (0=全开, 9000=全闭)
            260:     夹爪速度 (在 __init__ 中设置)
            264:     触发执行 (写 1 触发)
        """
        gripper_value_cmd = gripper_open_to_reg(gripper_open)
        regs = _gripper_reg_to_bytes(gripper_value_cmd)

        for attempt in range(retries):
            try:
                param = rm_peripheral_read_write_params_t(1, 258, 1, 2)
                ret = self.arm.rm_write_registers(param, regs)
                if ret != 0:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    print(f"[RealmanEnv] 写夹爪目标失败: ret={ret}")
                    return False

                param = rm_peripheral_read_write_params_t(1, 264, 1)
                ret = self.arm.rm_write_single_register(param, 1)
                if ret != 0:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                        continue
                    print(f"[RealmanEnv] 触发夹爪失败: ret={ret}")
                    return False

                return True
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.1)
                    continue
                print(f"[RealmanEnv] 夹爪异常: {e}")
                return False
        return False

    # ====================================================================
    # 同步接口
    # ====================================================================

    def compute_observation(self, retries: int = 3) -> dict:
        """读取当前状态 (同步, 带重试)"""
        for attempt in range(retries):
            try:
                ret, state = self.arm.rm_get_current_arm_state()
                if ret == 0 and state is not None:
                    return {
                        "pose": np.asarray(state["pose"], dtype=np.float64),
                        "gripper_open": self._get_gripper(),
                        "joint": np.radians(state["joint"]),
                    }
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(0.05)

        print("[RealmanEnv] 警告: 获取状态失败, 返回默认值")
        return {
            "pose": np.zeros(6),
            "gripper_open": 1.0,
            "joint": np.zeros(7),
        }

    def reset(
        self,
        target_gripper: float = 1.0,
        speed_ratio: int = 20,
        max_attempts: int = 100,
    ) -> dict:
        """复位到默认 home (joint)，再开夹爪"""
        target_joints = np.array([90.0, 0.0, 0.0, -90.0, 0.0, -90.0, 60.0])
        print(f"[RealmanEnv] 复位中... (速度比例={speed_ratio}%)")

        ret = self.arm.rm_movej(target_joints, v=speed_ratio, r=0, connect=0, block=True)
        if ret != 0:
            print(f"[RealmanEnv] movej 失败 ret={ret}, 兜底 movej_follow...")
            for _ in range(max_attempts):
                try:
                    ret = self.arm.rm_movej_follow(target_joints)
                    if ret != 0:
                        time.sleep(0.1)
                        continue
                    ret, state = self.arm.rm_get_current_arm_state()
                    if ret == 0 and state is not None:
                        err = np.linalg.norm(state["joint"] - target_joints)
                        if err < 0.1:
                            break
                    time.sleep(0.1)
                except Exception as e:
                    print(f"[RealmanEnv] 复位异常: {e}")
                    time.sleep(0.1)

        self._set_gripper(target_gripper)
        time.sleep(0.5)

        print("[RealmanEnv] 复位完成")
        return self.compute_observation()

    def step(self, action: dict, speed_ratio: int = 50) -> dict:
        """
        同步执行一个动作

        action 字段(均可选):
            - "pose":         6 维 EEF xyzrpy (米 + 弧度), SDK 原生坐标
            - "gripper_open": 夹爪开度 [0,1]
        """
        if "pose" in action:
            pose_target = action["pose"]
            if isinstance(pose_target, np.ndarray):
                pose_target = pose_target.tolist()

            ret = self.arm.rm_movel(pose_target, v=speed_ratio, r=0, connect=0, block=1)
            if ret != 0:
                error_msgs = {
                    1:  "控制器返回 false (参数错误或机械臂异常: 急停/碰撞保护等)",
                    -1: "数据发送失败",
                    -2: "数据接收失败或超时",
                    -3: "返回值解析失败",
                    -4: "当前到位设备校验失败",
                    -5: "单线程模式超时",
                    -6: "机械臂停止运动规划",
                }
                msg = error_msgs.get(ret, f"未知错误码: {ret}")
                print(f"[RealmanEnv] rm_movel 失败 ret={ret} - {msg}")
                print(f"[RealmanEnv] 目标位姿: {pose_target}")

        if "gripper_open" in action:
            self._set_gripper(action["gripper_open"])

        return self.compute_observation()

    def move_joint(self, joint_index: int, joint_angle: float, speed_ratio: int = 50) -> dict:
        """单关节运动 (joint_index 0..6, joint_angle 单位: 度)"""
        ret, state = self.arm.rm_get_current_arm_state()
        if ret != 0 or state is None:
            print(f"[RealmanEnv] 获取当前关节状态失败 ret={ret}")
            return self.compute_observation()

        current_joints = np.array(state["joint"])
        target_joints = current_joints.copy()
        target_joints[joint_index] = joint_angle

        print(f"[RealmanEnv] 关节{joint_index + 1}: {current_joints[joint_index]:.1f}° → {joint_angle:.1f}°")

        ret = self.arm.rm_movej(target_joints.tolist(), v=speed_ratio, r=0, connect=0, block=1)
        if ret != 0:
            print(f"[RealmanEnv] move_joint 失败 ret={ret}")
        return self.compute_observation()

    # ====================================================================
    # 异步接口 (遥操作高频)
    # ====================================================================

    def get_state(self) -> Optional[Dict[str, Any]]:
        """读取最新状态 (异步: 缓存; 同步: 走 compute_observation)"""
        if not self.async_mode:
            return self.compute_observation()

        self._update_stats()
        with self._state_lock:
            if self._latest_state is None:
                return None
            return {
                "pose":         self._latest_state.pose.copy(),
                "gripper_open": self._latest_state.gripper_open,
                "joint":        self._latest_state.joint.copy(),
                "timestamp":    self._latest_state.timestamp,
            }

    def get_pose(self) -> Optional[np.ndarray]:
        """读取最新 6 维 EEF xyzrpy (异步缓存)"""
        if not self.async_mode:
            return self.compute_observation()["pose"]

        with self._state_lock:
            if self._latest_state is None:
                return None
            return self._latest_state.pose.copy()

    def send_pose(self, pose: np.ndarray):
        """发位姿命令, pose = 6 维 EEF xyzrpy (异步: 非阻塞; 同步: 走 step)"""
        if not self.async_mode:
            self.step({"pose": pose})
            return

        with self._cmd_lock:
            self._pending_pose = np.asarray(pose, dtype=np.float64).copy()

    def send_gripper(self, gripper_open: float):
        """发夹爪命令 (异步: 非阻塞; 同步: 直接写)"""
        if not self.async_mode:
            self._set_gripper(gripper_open)
            return

        with self._cmd_lock:
            self._pending_gripper = gripper_open

    def get_communication_stats(self) -> Dict[str, Any]:
        if not self.async_mode:
            return {"connected": self.connected, "async_mode": False}

        with self._stats_lock:
            return {
                "connected": self.connected,
                "async_mode": True,
                "state_update_rate": self._state_rate,
                "avg_latency_ms": self._avg_latency,
            }

    def close(self):
        if self.async_mode:
            self._stop_event.set()
            if hasattr(self, "_cmd_thread"):
                self._cmd_thread.join(timeout=1.0)
            if hasattr(self, "_state_thread"):
                self._state_thread.join(timeout=1.0)

        self.arm.rm_delete_robot_arm()
        print(f"[RealmanEnv] 已关闭: {self.robot_ip}")
