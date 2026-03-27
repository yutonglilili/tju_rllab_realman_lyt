"""
用 ManiSkill2 / SAPIEN 可视化并控制你的 RM75 机械臂 URDF 关节。

特点：
  - 可视化窗口由 SAPIEN Viewer 提供
  - 控制方式：通过终端输入命令实时更新关节目标（避免依赖额外 GUI 滑条库）

依赖（需要你在已安装 ManiSkill2 / SAPIEN 的 Python 环境中运行）：
  - sapien.core
  - sapien.utils.Viewer

默认 URDF：
  /home/zhangzhao/lyt/rllab_urdfs/RM75+gripper/RM75-B/urdf/RM75-B.urdf

控制命令（在运行窗口时另起终端输入）：
  - `i v`：把第 i 个 active joint（从 0 开始）的目标角设置为 v（单位：弧度）
  - `name v`：把名为 name 的 active joint 设置为 v（弧度）
  - `reset`：把所有关节设置到各自 limit 的中间值
  - `quit`：退出
"""

import argparse
import os
import sys
import time
import threading

import numpy as np


def _set_robot_qpos(robot, qpos):
    """尽量兼容不同 SAPIEN 版本的设置方式（运行时判断方法是否存在）。"""
    if hasattr(robot, "set_qpos"):
        robot.set_qpos(qpos)
        return
    if hasattr(robot, "set_joint_positions"):
        robot.set_joint_positions(qpos)
        return
    # 如果没有提供统一接口，逐个 joint 尝试（不同版本 API 名字可能不同）
    if hasattr(robot, "get_active_joints"):
        active_joints = robot.get_active_joints()
        for idx, joint in enumerate(active_joints):
            if hasattr(joint, "set_position"):
                joint.set_position(float(qpos[idx]))
            elif hasattr(joint, "set_qpos"):
                joint.set_qpos(float(qpos[idx]))


def _get_joint_limits(joint):
    """
    兼容不同 SAPIEN 版本的 joint 限位 API。

    - 某些版本提供 get_limit()
    - 你当前的报错提示 joint 没有 get_limit，而是 get_limits()
    """
    if hasattr(joint, "get_limit"):
        return joint.get_limit()
    if hasattr(joint, "get_limits"):
        return joint.get_limits()
    # 连 limit 都没有的话，用一个保守范围兜底（弧度）
    return -np.pi, np.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/zhangzhao/lyt/rllab_urdfs/RM75+gripper/RM75-B/urdf/RM75-B.urdf",
        help="URDF path to visualize",
    )
    parser.add_argument("--step_hz", type=float, default=100.0, help="simulation step rate (Hz)")
    args = parser.parse_args()

    if not os.path.exists(args.urdf):
        raise FileNotFoundError(f"URDF not found: {args.urdf}")

    try:
        import sapien.core as sapien
        from sapien.utils import Viewer
    except Exception as e:
        raise RuntimeError(
            "ManiSkill2/SAPIEN environment is required for this script. "
            "Please run it inside a ManiSkill2 python env. "
            f"Import error: {e}"
        )

    engine = sapien.Engine()
    # 用默认渲染器；在 ManiSkill2 环境里通常可用
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1.0 / max(float(args.step_hz), 1e-6))

    # 基本光照与地面（避免全黑）
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    scene.add_point_light([1, 2, 2], [1, 1, 1])
    scene.add_point_light([1, -2, 2], [1, 1, 1])
    scene.add_ground(altitude=0)

    # 加载机器人
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(args.urdf)
    robot.set_root_pose(sapien.Pose([0, 0, 0]))

    active_joints = robot.get_active_joints()
    n = len(active_joints)
    if n == 0:
        raise RuntimeError("No active joints found in URDF.")

    # 记录关节 limit，用于 reset
    qpos_mid = []
    for joint in active_joints:
        lo, hi = _get_joint_limits(joint)
        if lo > hi:
            lo, hi = -np.pi, np.pi
        qpos_mid.append(0.5 * (lo + hi))
    qpos_mid = np.asarray(qpos_mid, dtype=np.float32)

    # 初始目标关节位置
    try:
        qpos = np.asarray(robot.get_qpos(), dtype=np.float32)
    except Exception:
        qpos = qpos_mid.copy()

    qpos_target = qpos.copy()
    qpos_lock = threading.Lock()
    stop_event = threading.Event()

    def input_thread():
        nonlocal qpos_target
        print("\n[控制说明]")
        print("  输入 `i v`：设置第 i 个 active joint 到 v（弧度）")
        print("  输入 `name v`：设置名为 name 的 active joint 到 v（弧度）")
        print("  输入 `reset`：设置所有关节到各自 limit 中点")
        print("  输入 `quit`：退出\n")
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                s = line.strip()
                if not s:
                    continue
                if s.lower() == "quit":
                    stop_event.set()
                    break
                if s.lower() == "reset":
                    with qpos_lock:
                        qpos_target = qpos_mid.copy()
                    continue

                parts = s.split()
                if len(parts) != 2:
                    print("格式错误：请用 `i v` 或 `name v`")
                    continue
                a, b = parts
                val = float(b)

                # i v
                if a.isdigit():
                    idx = int(a)
                    if 0 <= idx < n:
                        with qpos_lock:
                            qpos_target[idx] = val
                    else:
                        print(f"joint index out of range: {idx}")
                    continue

                # name v
                found = False
                for idx, joint in enumerate(active_joints):
                    if getattr(joint, "name", None) == a:
                        with qpos_lock:
                            qpos_target[idx] = val
                        found = True
                        break
                if not found:
                    print(f"Unknown joint name: {a}")
            except Exception:
                time.sleep(0.01)

    # 打印关节信息（便于你输入 i/name）
    print("Active joints:")
    for i, joint in enumerate(active_joints):
        lo, hi = _get_joint_limits(joint)
        name = getattr(joint, "name", f"joint_{i}")
        print(f"  [{i}] {name} limit=[{lo:.3f}, {hi:.3f}]")

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=1.5, y=0.0, z=1.0)
    viewer.set_camera_rpy(r=0.0, p=-0.5, y=0.0)

    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()

    # 主循环：每帧读取最新目标关节并设置到机器人，然后渲染
    print("\n开始仿真窗口渲染；关闭窗口或输入 `quit` 退出。")
    while (not viewer.closed) and (not stop_event.is_set()):
        with qpos_lock:
            qpos = qpos_target.copy()

        _set_robot_qpos(robot, qpos)

        scene.step()
        scene.update_render()
        viewer.render()

    try:
        viewer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

