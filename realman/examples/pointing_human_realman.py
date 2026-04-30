"""
支持：
1. 点击图像控制机械臂
2. 记录动作
3. 回放动作
"""

import os
import sys
import cv2
import json
import copy
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realman.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T
from realman.open3d_realsense_env import Open3dRealsenseEnv

# =========================
# Utils
# =========================

def convert_robot_state(robot_state):
    """RobotState -> dict(obs)"""
    return {
        "Ttcp2base": T_from_realman_xyzrpy(robot_state.pose),
        "gripper_open": robot_state.gripper,
    }

def record_actions(obs):
    if is_recording:
        recorded_actions.append({
            "Ttcp2base": obs["Ttcp2base"].tolist(),
            "gripper_open": obs["gripper_open"]
        })

def replay_actions(path):
    actions = []
    with open(path, 'r') as f:
        for line in f:
            actions.append(json.loads(line))
    return actions

# =========================
# 全局变量
# =========================
recorded_actions = []
is_recording = True

clicked_point = None
clicked_flag = False

# =========================
# 主程序
# =========================
if __name__ == "__main__":

    with open("/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json", "r") as f:
        cam_results = json.load(f)

    env = RealmanEnv("192.168.101.19")
    rs_env = Open3dRealsenseEnv("f1471338")

    cv2.namedWindow("color")

    clicked_point = None
    clicked_flag = False

    def on_mouse(event, x, y, flags, param):
        global clicked_point, clicked_flag
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            clicked_flag = True

    cv2.setMouseCallback("color", on_mouse)

    try:
        robot_state = env.reset()
        obs = convert_robot_state(robot_state)
        obs |= rs_env.reset()

        action = {
            "pose": robot_state.pose,
            "gripper": robot_state.gripper,
        }

        disable_robot = False

        while True:
            try:
                if not disable_robot:
                    robot_state = env.step(action)
                else:
                    robot_state = env.get_state()

                obs = convert_robot_state(robot_state)
                obs |= rs_env.step()

            except Exception as e:
                print("ERROR:", e)
                disable_robot = True
                continue

            record_actions(obs)

            img = obs["rgb"][:, :, ::-1]

            # =========================
            # 点击处理
            # =========================
            if clicked_flag:
                clicked_flag = False
                u, v = clicked_point

                d = obs["depth"][v, u] / rs_env.meta_obs["depth_scale"]

                xyz = np.linalg.inv(np.array(rs_env.meta_obs["intrinsic"])) @ (np.array([u, v, 1]) * d)
                xyz = np.array(cam_results["Tcam2base"]) @ np.append(xyz, 1)

                x, y, z = xyz[:3]
                z += 0.02

                new_T = copy.deepcopy(obs["Ttcp2base"])
                new_T[:3, 3] = [x, y, z]

                pose_tcp = realman_xyzrpy_from_T(new_T)

                action = {
                    "pose": pose_tcp,
                    "gripper": obs["gripper_open"],
                }

                print("Move to:", x, y, z)

            # =========================
            # 显示
            # =========================
            if clicked_point:
                cv2.circle(img, clicked_point, 5, (0, 0, 255), -1)

            cv2.imshow("color", img)
            k = cv2.waitKey(1)

            # =========================
            # 控制
            # =========================
            if k == ord('c'):
                action["gripper"] = 0.0
                disable_robot = False
                print("Close")

            elif k == ord('o'):
                action["gripper"] = 0.09
                disable_robot = False
                print("Open")

            elif k == ord('d'):
                disable_robot = True
                print("Disable")

            elif k == ord('e'):
                disable_robot = False
                print("Enable")

            elif k == ord('s'):
                filename = f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w') as f:
                    for a in recorded_actions:
                        f.write(json.dumps(a) + '\n')
                print("Saved:", filename)

            elif k == ord('r'):
                files = glob.glob("recorded_*.txt")
                if not files:
                    print("No file")
                    continue

                latest = max(files, key=lambda x: Path(x).stat().st_mtime)
                print("Replay:", latest)

                data = replay_actions(latest)

                for i, a in enumerate(data):
                    pose_tcp = realman_xyzrpy_from_T(np.array(a["Ttcp2base"]))

                    action = {
                        "pose": pose_tcp,
                        "gripper": a["gripper_open"]
                    }

                    robot_state = env.step(action)
                    obs = convert_robot_state(robot_state)
                    obs |= rs_env.step()

                    img = obs["rgb"][:, :, ::-1]
                    cv2.putText(img, f"{i+1}/{len(data)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    cv2.imshow("color", img)
                    if cv2.waitKey(50) == ord('q'):
                        break

                print("Replay done")

            elif k == ord('q'):
                break

    finally:
        env.close()
        rs_env.close()
