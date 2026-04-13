"""
SpaceMouse 遥操作数据采集脚本（适配新版 RealmanEnv）

特点：
- 完全走 env API（不会再绕过 async controller）
- 支持 TCP pose 控制
- 夹爪单位统一（0~1 → 米）
- Recorder 使用 env.get_state（线程安全）
"""

import numpy as np
import time
import threading
import h5py
import json
import os


from realman.realman_env import (
    RealmanEnv,
    T_TCP2REALMANEEF,
    T_TCP2REALMANEEF_INV,
    realman_xyzrpy_from_T,
    T_from_realman_xyzrpy,
)
from realman.collect_data.spacemouse_agent import SpacemouseAgent
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import active_matrix_from_angle

# ============================================================
# 配置
# ============================================================
ARM_SIDE = "left"
FPS = 50
DATASET_ROOT = "datasets/vlm"

GRIPPER_RATE = 0.8
GRIPPER_MIN_DELTA = 0.005
MAX_GRIPPER_WIDTH = 0.09   # ⚠️ 根据你的夹爪实际调整

# ============================================================
# 工具函数
# ============================================================

def delta_to_transform(delta):
    T = np.eye(4)
    T[:3, 3] = delta[:3] * 0.001

    Rx = active_matrix_from_angle(0, delta[3])
    Ry = active_matrix_from_angle(1, delta[4])
    Rz = active_matrix_from_angle(2, delta[5])

    T[:3, :3] = Rz @ Ry @ Rx
    return T


# ============================================================
# Recorder（基于 env.get_state）
# ============================================================

class TrajectoryRecorder:

    def __init__(self, env, arm_side="left", fps=50):
        self.env = env
        self.arm_side = arm_side
        self.dt = 1.0 / fps

        self._joints = []
        self._gripper = []
        self._ts = []

        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

        self._gripper_val = 1.0

    def start(self):
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Recorder] 启动")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        print(f"[Recorder] 停止, 帧数={len(self._joints)}")

    def set_gripper(self, g):
        self._gripper_val = g

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()

            state = self.env.get_state()
            if state is not None:
                self._joints.append(state.joint.copy())
                self._gripper.append(self._gripper_val)
                self._ts.append(time.time() - self._t0)

            dt = time.time() - t0
            if self.dt - dt > 0:
                time.sleep(self.dt - dt)

    def save(self, path):
        N = len(self._joints)
        if N == 0:
            print("⚠️ 没有数据")
            return

        with h5py.File(path, "w") as f:
            f.attrs["fps"] = int(1 / self.dt)
            f.attrs["num_frames"] = N

            f.create_dataset("timestamp", data=np.array(self._ts))

            obs = f.create_group("observations")
            obs.create_dataset(f"{self.arm_side}_joints", data=np.array(self._joints))
            obs.create_dataset(f"{self.arm_side}_gripper", data=np.array(self._gripper))

        print(f"✅ 数据保存: {path}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    agent = SpacemouseAgent()
    env = RealmanEnv("192.168.101.19", mode="async")

    # ===== 初始状态 =====
    time.sleep(1.0)
    state = env.get_state()
    assert state is not None, "获取初始状态失败"

    T_target_tcp2base = T_from_realman_xyzrpy(state.pose)

    goal_gripper = 1.0
    last_gripper = 1.0

    # ===== 数据路径 =====
    ts = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(DATASET_ROOT, ts)
    os.makedirs(save_dir, exist_ok=True)

    recorder = TrajectoryRecorder(env, ARM_SIDE, FPS)
    recorder.start()

    print("===== 开始遥操作 =====")

    try:
        while True:

            delta, buttons = agent.act()

            # ===== 夹爪 =====
            close_pressed = bool(buttons[0])
            open_pressed = bool(buttons[1])

            if close_pressed ^ open_pressed:
                direction = -1 if close_pressed else 1
                goal_gripper = np.clip(
                    goal_gripper + direction * GRIPPER_RATE * 0.02,
                    0, 1
                )

            if abs(goal_gripper - last_gripper) >= GRIPPER_MIN_DELTA:
                env.send_gripper(goal_gripper * MAX_GRIPPER_WIDTH)
                recorder.set_gripper(goal_gripper)
                last_gripper = goal_gripper
            elif (
                not close_pressed
                and not open_pressed
                and abs(goal_gripper - last_gripper) > 1e-6
            ):
                env.send_gripper(goal_gripper * MAX_GRIPPER_WIDTH)
                recorder.set_gripper(goal_gripper)
                last_gripper = goal_gripper

            # ===== 位姿 =====
            T_delta = delta_to_transform(delta)
            T_target_tcp2base = T_target_tcp2base @ T_delta

            xyzrpy = realman_xyzrpy_from_T(T_target_tcp2base)

            env.send_pose(xyzrpy)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n停止采集")

    finally:
        recorder.stop()

        path = os.path.join(save_dir, "data.hdf5")
        recorder.save(path)

        env.close()
        agent.close()

        print("✅ 完成")
