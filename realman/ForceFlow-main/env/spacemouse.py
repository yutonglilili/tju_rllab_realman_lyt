"""
SpaceMouse → EEF-frame delta agent (Realman 适配版)

act() 返回 (delta_6d, buttons), delta_6d 为 EEF 局部坐标系下的 6-dof 增量
(xyz 单位约 mm 级 → 在 collect 里的 delta_to_transform *0.001 转米;
 rpy 单位为弧度)。

历史背景：
- 原 realman/collect_data_by_tele_op/spacemouse_agent.py 的两段经验式轴映射是
  为 TCP frame 调出来的 (旧 env 内部把 TCP 偏移叠在 SDK 之上)
- 新 env.realman_env 直接用 SDK 原生 EEF 坐标, 不再有 TCP 偏移; 同样物理动作
  在 EEF frame 下方向会偏 (TCP/EEF 之间有 z 轴 -60° 加一次基变换的固定旋转)
- 这里在第二段映射后再补一次 R_TCP2EEF, 把"TCP 等效 delta"转到"EEF 等效 delta",
  用户手感与旧版完全一致
"""

import threading
import time

import numpy as np
import pyspacemouse
from pytransform3d.rotations import active_matrix_from_angle


# TCP → EEF 的旋转矩阵 (与原 realman_env.T_TCP2REALMANEEF 的 R 部分一致)
R_TCP2EEF = active_matrix_from_angle(2, -np.pi / 3) @ np.array(
    [
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ],
    dtype=np.float64,
)


class SpacemouseAgent:
    """SpaceMouse 控制器, act() 返回 (delta_6d_EEF, buttons)"""

    def __init__(self):
        self._device = pyspacemouse.open()
        self.state_lock = threading.Lock()
        self.latest_data = {
            "action": np.zeros(6, dtype=np.float64),
            "buttons": np.zeros(2, dtype=np.bool_),
        }

        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = self._device.read()
            if state is not None:
                with self.state_lock:
                    # 阶段 1: 鼠标 6 轴重排 (与旧 realman 项目一致, 保留方向手感)
                    self.latest_data["action"] = np.array(
                        [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw],
                        dtype=np.float64,
                    )
                    self.latest_data["buttons"] = np.array(state.buttons, dtype=np.bool_)
            time.sleep(1 / 150)

    def act(self, observation=None):
        with self.state_lock:
            raw = self.latest_data["action"].copy()
            buttons = self.latest_data["buttons"].copy()

        raw[:3] *= 2.0        # 平移幅值
        raw[3:] *= 0.005     # 旋转幅值 (rad)

        # 阶段 2: 经验式互换, 等价于"在 TCP frame 下的 delta"
        tcp = np.empty(6, dtype=np.float64)
        tcp[0] = -raw[2]
        tcp[1] = -raw[1]
        tcp[2] = -raw[0]
        tcp[3] = -raw[5]
        tcp[4] = -raw[4]
        tcp[5] = -raw[3]

        # 阶段 3: TCP → EEF 旋转, 因为新 env 直接在 EEF 上累 delta
        # 同时作用在平移和旋转分量 (小角度近似下旋转向量也按帧间旋转矩阵转换)
        eef = np.empty(6, dtype=np.float64)
        eef[:3] = R_TCP2EEF @ tcp[:3]
        eef[3:] = R_TCP2EEF @ tcp[3:]

        return eef, buttons

    def close(self):
        if self._device is not None:
            self._device.close()


if __name__ == "__main__":
    agent = SpacemouseAgent()
    try:
        while True:
            action, buttons = agent.act()
            print(f"Action: {action}, Buttons: {buttons}")
            # time.sleep(0.03)
    except KeyboardInterrupt:
        agent.close()
