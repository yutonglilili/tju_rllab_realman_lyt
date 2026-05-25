"""
Replay 脚本 - 把 dataset 里录制的 action delta 序列重新发给机械臂

说明：
- 只 replay ACTION (spacemouse 6-dof delta) 和 gripper_action (二值 0/1)
- 不 replay OBS pose -- 主循环里只调 env.send_pose, target 由 delta 累加得到
- pos[0] 只用来在 replay 前把机械臂 movel 到起始位姿(可关)，与 dataset 一致再开始
- 若 replay 完成后 actual_final 与 recorded pos[-1] 偏差大, 说明 SDK 控制有累积误差

操作：
- Ctrl+C 中断
"""

# ═══════════════════════════════════════════════════════════════════
# 用户可修改参数（直接在此处编辑即可）
# ═══════════════════════════════════════════════════════════════════

DATASET_PATH      = "data/demo.zarr"    # 要 replay 的 Zarr 数据集
EPISODE_ID        = 0                   # episode 索引
ROBOT_IP          = "192.168.101.19"

FPS               = 30.0                # 与采集一致, 决定主循环节拍
GO_TO_START_POS   = True                # True: replay 前先 movel 到 pos[0]; False: 从当前位置开始累 delta
START_SPEED       = 20                  # 走到 pos[0] 的 movel 速度比例 (1-100)

GRIPPER_MIN_DELTA = 0.005               # 夹爪命令最小变化阈值
PROGRESS_EVERY    = 30                  # 每多少步打印一次进度

# ═══════════════════════════════════════════════════════════════════


import os
import sys
import time

import numpy as np
import zarr
from pytransform3d.rotations import active_matrix_from_angle


PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FORCEFLOW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (PROJECT_ROOT, FORCEFLOW_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.realman_env import RealmanEnv, T_from_realman_xyzrpy, realman_xyzrpy_from_T


def delta_to_transform(delta: np.ndarray) -> np.ndarray:
    """与 collect.py 一致: 6-dof delta -> 4x4 SE(3). xyz mm -> m, rpy 弧度直用"""
    T = np.eye(4)
    T[:3, 3] = delta[:3] * 0.001
    Rx = active_matrix_from_angle(0, delta[3])
    Ry = active_matrix_from_angle(1, delta[4])
    Rz = active_matrix_from_angle(2, delta[5])
    T[:3, :3] = Rz @ Ry @ Rx
    return T


def load_episode(dataset_path: str, episode_id: int):
    root = zarr.open(dataset_path, "r")
    ee = root["meta/episode_ends"][:]
    if episode_id >= len(ee):
        raise IndexError(f"episode_id={episode_id} 超出范围 (共 {len(ee)} 个 episode)")
    start = 0 if episode_id == 0 else int(ee[episode_id - 1])
    end = int(ee[episode_id])

    return {
        "action":         root["data/action"][start:end],
        "gripper_action": root["data/gripper_action"][start:end].flatten(),
        "pos":            root["data/pos"][start:end],
        "timestamp":      root["data/timestamp"][start:end],
        "n_steps":        end - start,
    }


def main():
    print(f"=== Replay {DATASET_PATH} episode {EPISODE_ID} ===")
    ep = load_episode(DATASET_PATH, EPISODE_ID)
    action = ep["action"]
    gripper_action = ep["gripper_action"]
    pos = ep["pos"]
    n = ep["n_steps"]
    duration = float(ep["timestamp"][-1] - ep["timestamp"][0]) if n > 1 else 0.0

    print(f"  steps          : {n}")
    print(f"  duration       : {duration:.2f}s (采集时)")
    print(f"  fps target     : {FPS}")
    print(f"  pos[0]         : {pos[0]}")
    print(f"  pos[-1]        : {pos[-1]}")
    print(f"  gripper 0/1    : {(gripper_action == 0).sum()}/{(gripper_action == 1).sum()}")

    env = RealmanEnv(robot_ip=ROBOT_IP, async_mode=True)

    try:
        # 1. 回 home (避免从奇怪的姿态去 pos[0] 直线运动撞东西)
        print("[Replay] env.reset() -> home...")
        env.reset(target_gripper=float(gripper_action[0]), speed_ratio=30)

        # 2. 可选: movel 到采集时的起始位姿
        if GO_TO_START_POS:
            print(f"[Replay] movel -> pos[0] (speed_ratio={START_SPEED})...")
            env.step(
                {"pose": pos[0].tolist(), "gripper_open": float(gripper_action[0])},
                speed_ratio=START_SPEED,
            )
            target_xyzrpy = pos[0].astype(np.float64).copy()
        else:
            print("[Replay] GO_TO_START_POS=False, 从当前位置开始")
            state = env.get_state()
            if state is None:
                raise RuntimeError("无法读取机械臂状态")
            target_xyzrpy = state["pose"].astype(np.float64).copy()

        print(f"[Replay] 起始 target: {target_xyzrpy}")
        time.sleep(0.5)

        # 3. 主循环: 纯发 delta
        last_gripper = float(gripper_action[0])
        env.send_gripper(last_gripper)

        period = 1.0 / FPS
        t_start = time.time()
        n_pose_sent = 0
        n_gripper_sent = 0

        for i in range(n):
            loop_start = time.perf_counter()

            # delta 4x4 累加, 然后转 6 维 EEF xyzrpy
            T_target = T_from_realman_xyzrpy(target_xyzrpy)
            T_target = T_target @ delta_to_transform(action[i])
            target_xyzrpy = realman_xyzrpy_from_T(T_target)
            env.send_pose(target_xyzrpy)
            n_pose_sent += 1

            # 夹爪只在变化时发
            g = float(gripper_action[i])
            if abs(g - last_gripper) >= GRIPPER_MIN_DELTA:
                env.send_gripper(g)
                last_gripper = g
                n_gripper_sent += 1

            if i % PROGRESS_EVERY == 0 or i == n - 1:
                elapsed = time.time() - t_start
                tgt_pos = target_xyzrpy[:3]
                print(
                    f"  step={i:4d}/{n}  t={elapsed:5.2f}s  "
                    f"target xyz=[{tgt_pos[0]:+.3f}, {tgt_pos[1]:+.3f}, {tgt_pos[2]:+.3f}]  g={last_gripper:.0f}"
                )

            # 保持节拍
            elapsed = time.perf_counter() - loop_start
            if period > elapsed:
                time.sleep(period - elapsed)

        # 4. 收尾对比
        time.sleep(0.5)  # 等最后一帧 SDK 真正到位
        final_state = env.get_state()
        actual_final = final_state["pose"] if final_state else None

        print()
        print("=== Replay 完成 ===")
        print(f"  poses sent       : {n_pose_sent}")
        print(f"  gripper sent     : {n_gripper_sent}")
        print(f"  target final     : {target_xyzrpy}")
        print(f"  recorded pos[-1] : {pos[-1]}")
        if actual_final is not None:
            print(f"  actual final     : {actual_final}")
            xyz_err = np.linalg.norm(actual_final[:3] - pos[-1, :3])
            target_err = np.linalg.norm(target_xyzrpy[:3] - pos[-1, :3])
            print(f"  ‖actual - recorded‖ xyz : {xyz_err*1000:.2f} mm")
            print(f"  ‖target - recorded‖ xyz : {target_err*1000:.2f} mm  (action 累积误差)")

    except KeyboardInterrupt:
        print("\n[Replay] 中断")
    finally:
        env.close()


if __name__ == "__main__":
    main()
