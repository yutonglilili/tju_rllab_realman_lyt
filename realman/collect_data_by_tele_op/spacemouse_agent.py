# Source: https://github.com/rail-berkeley/serl/blob/main/serl_robot_infra/franka_env/spacemouse/spacemouse_expert.py

import threading
import pyspacemouse
import numpy as np
import time

class SpacemouseAgent():
    # 初始化硬件连接
    def __init__(self):
        self._device = pyspacemouse.open()
        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        
        # 开启守护线程持续读取数据
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = self._device.read()
            if state is not None:
                with self.state_lock:
                    # 原始映射：处理鼠标内部的轴向偏移
                    self.latest_data["action"] = np.array(
                        [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw], 
                        dtype=np.float64
                    )  # spacemouse axis matched with robot base frame
                    self.latest_data["buttons"] = np.array(state.buttons, dtype=np.bool_)
            time.sleep(1 / 150)

    def act(self, observation=None):
        with self.state_lock:
            action = self.latest_data["action"].copy()
            buttons = self.latest_data["buttons"].copy()
        action[:3] *= 1.0
        action[3:] *= 0.0025

        pre_action = action.copy()
        action[0] = -pre_action[2]
        action[1] = -pre_action[1]
        action[2] = -pre_action[0]

        action[3] = -pre_action[5]
        action[4] = -pre_action[4]
        action[5] = -pre_action[3]
        
        return action, buttons
    
    def close(self):
        if self._device is not None:
            self._device.close()

if __name__ == "__main__":
    agent = SpacemouseAgent()
    try:
        while True:
            action, buttons = agent.act()
            print(f"Action: {action}, Buttons: {buttons}")
            time.sleep(0.03)
    except KeyboardInterrupt:
        agent.close()
