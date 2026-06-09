"""TCP <-> EEF 坐标转换工具

用法:
    python tcp_eef_convert.py --tcp -2.863 -0.126 2.108       # TCP RPY -> EEF RPY
    python tcp_eef_convert.py --eef 1.110 0.397 -2.741        # EEF RPY -> TCP RPY
    python tcp_eef_convert.py --tcp-full 0.1 -0.3 0.05 -2.863 -0.126 2.108   # 完整 TCP xyzrpy -> EEF
    python tcp_eef_convert.py --eef-full -0.06 -0.51 0.15 1.11 0.40 -2.74    # 完整 EEF xyzrpy -> TCP
"""

import argparse
import numpy as np
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import active_matrix_from_angle


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
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def realman_xyzrpy_from_T(T):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    ry = np.arcsin(np.clip(-T[2, 0], -1, 1))
    if np.cos(ry) != 0:
        rx = np.arctan2(T[2, 1] / np.cos(ry), T[2, 2] / np.cos(ry))
        rz = np.arctan2(T[1, 0] / np.cos(ry), T[0, 0] / np.cos(ry))
    else:
        rx = 0
        rz = np.arctan2(-T[0, 1], T[1, 1])
    return np.array([x, y, z, rx, ry, rz])


T_TCP2REALMANEEF = transform_from(
    active_matrix_from_angle(2, -np.pi / 3) @ np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0],
    ]),
    np.array([0, 0, 0.22])
)


def tcp2eef(pose_tcp: np.ndarray) -> np.ndarray:
    T_tcp2base = T_from_realman_xyzrpy(pose_tcp)
    T_eef2base = T_tcp2base @ np.linalg.inv(T_TCP2REALMANEEF)
    return realman_xyzrpy_from_T(T_eef2base)


def eef2tcp(pose_eef: np.ndarray) -> np.ndarray:
    T_eef2base = T_from_realman_xyzrpy(pose_eef)
    T_tcp2base = T_eef2base @ T_TCP2REALMANEEF
    return realman_xyzrpy_from_T(T_tcp2base)


def main():
    parser = argparse.ArgumentParser(description="TCP <-> EEF 坐标转换")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tcp", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
                       help="TCP RPY -> EEF RPY (仅旋转，XYZ 默认 0)")
    group.add_argument("--eef", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
                       help="EEF RPY -> TCP RPY (仅旋转，XYZ 默认 0)")
    group.add_argument("--tcp-full", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                       help="完整 TCP xyzrpy -> EEF xyzrpy")
    group.add_argument("--eef-full", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                       help="完整 EEF xyzrpy -> TCP xyzrpy")
    args = parser.parse_args()

    if args.tcp is not None:
        pose_tcp = np.array([0, 0, 0] + args.tcp)
        pose_eef = tcp2eef(pose_tcp)
        print(f"TCP RPY:  {np.round(pose_tcp[3:], 4)}")
        print(f"EEF RPY:  {np.round(pose_eef[3:], 4)}")
        print(f"EEF full: {np.round(pose_eef, 4)}")

    elif args.eef is not None:
        pose_eef = np.array([0, 0, 0] + args.eef)
        pose_tcp = eef2tcp(pose_eef)
        print(f"EEF RPY:  {np.round(pose_eef[3:], 4)}")
        print(f"TCP RPY:  {np.round(pose_tcp[3:], 4)}")
        print(f"TCP full: {np.round(pose_tcp, 4)}")

    elif args.tcp_full is not None:
        pose_tcp = np.array(args.tcp_full)
        pose_eef = tcp2eef(pose_tcp)
        print(f"TCP: {np.round(pose_tcp, 4)}")
        print(f"EEF: {np.round(pose_eef, 4)}")

    elif args.eef_full is not None:
        pose_eef = np.array(args.eef_full)
        pose_tcp = eef2tcp(pose_eef)
        print(f"EEF: {np.round(pose_eef, 4)}")
        print(f"TCP: {np.round(pose_tcp, 4)}")


if __name__ == "__main__":
    main()
