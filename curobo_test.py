import torch
import math

from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


# ===============================
# 路径配置（写死）
# ===============================

ROBOT_CFG_PATH = "/home/zhangzhao/lyt/realman/robot_cfg/rm75_ag2f90c/rm75_ag2f90c_curobo.yml"


# ===============================
# rpy -> quaternion
# ===============================

def rpy_to_quat(roll, pitch, yaw):

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]


def quat_to_rpy(qw, qx, qy, qz):
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [roll, pitch, yaw]


def joint_deg_to_rad(joint_deg):
    """
    将关节角从角度制转换为弧度制
    输入:
        joint_deg: [deg1, deg2, ...]
    输出:
        [rad1, rad2, ...]
    """
    return [math.radians(v) for v in joint_deg]


# ===============================
# 初始化 cuRobo planner
# ===============================

_motion_gen = None
_robot_world = None


def init_curobo():

    global _motion_gen

    if _motion_gen is not None:
        return

    world_config = {
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0, 0, -0.1, 1, 0, 0, 0],
            }
        }
    }

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        ROBOT_CFG_PATH,
        world_config,
        interpolation_dt=0.02,
    )

    _motion_gen = MotionGen(motion_gen_config)

    print("Warming up cuRobo...")
    _motion_gen.warmup()
    print("cuRobo ready")


def init_collision_checker():
    global _robot_world

    if _robot_world is not None:
        return

    world_config = {
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0, 0, -0.1, 1, 0, 0, 0],
            }
        }
    }

    config = RobotWorldConfig.load_from_config(
        ROBOT_CFG_PATH,
        world_config,
        collision_activation_distance=0.0,
    )
    _robot_world = RobotWorld(config)


# ===============================
# 主规划函数
# ===============================

def plan_with_curobo(current_joint, target_pose):
    """
    输入:
        current_joint : [j1..jn]
        target_pose   : [x,y,z,roll,pitch,yaw]

    输出:
        trajectory : [[j1..jn], [j1..jn], ...]
    """

    init_curobo()

    device = "cuda"

    # --------------------------------
    # start state
    # --------------------------------

    current_joint = torch.tensor(
        current_joint, device=device, dtype=torch.float32
    ).unsqueeze(0)

    start_state = JointState.from_position(
        current_joint,
        joint_names=_motion_gen.joint_names,
    )

    # --------------------------------
    # goal pose
    # --------------------------------

    x, y, z, roll, pitch, yaw = target_pose

    qw, qx, qy, qz = rpy_to_quat(roll, pitch, yaw)

    goal_pose = Pose.from_list(
        [x, y, z, qw, qx, qy, qz]
    )

    # --------------------------------
    # 规划
    # --------------------------------

    result = _motion_gen.plan_single(
        start_state,
        goal_pose,
        MotionGenPlanConfig(max_attempts=3)
    )

    if not result.success:
        print("curobo planning failed")
        return None

    traj = result.get_interpolated_plan()

    # --------------------------------
    # 转成 python list
    # --------------------------------

    joint_traj = traj.position.cpu().numpy()

    trajectory = [q.tolist() for q in joint_traj]

    return trajectory


def check_joint_collision_distance(current_joint):
    """
    输入:
        current_joint: [j1..jn] (rad)
    输出:
        打印世界碰撞距离和自碰撞距离
    """
    init_collision_checker()
    device = "cuda"

    q = torch.tensor([current_joint], device=device, dtype=torch.float32)
    d_world, d_self = _robot_world.get_world_self_collision_distance_from_joints(q)

    world_v = float(d_world.squeeze().detach().cpu().item())
    self_v = float(d_self.squeeze().detach().cpu().item())

    print("\n===== Collision Distance =====")
    print(f"world collision distance: {world_v:.6f}")
    print(f"self  collision distance: {self_v:.6f}")
    if self_v < 0.0:
        print("self collision status : IN COLLISION")
    else:
        print("self collision status : collision free")


def print_trajectory_with_xyzrpy(trajectory):
    """
    输入:
        trajectory: [[j1..jn], [j1..jn], ...]
    输出:
        直接打印每个轨迹点的 joint 和末端 xyzrpy
    """
    if trajectory is None or len(trajectory) == 0:
        print("empty trajectory")
        return

    init_curobo()
    device = "cuda"

    print("\n===== cuRobo Trajectory =====")
    for idx, joint in enumerate(trajectory):
        q = torch.tensor(joint, device=device, dtype=torch.float32).unsqueeze(0)
        js = JointState.from_position(q, joint_names=_motion_gen.joint_names)
        state = _motion_gen.rollout_fn.compute_kinematics(js)

        ee_pos = state.ee_pos_seq.squeeze().detach().cpu().numpy().tolist()
        ee_quat = state.ee_quat_seq.squeeze().detach().cpu().numpy().tolist()  # [w, x, y, z]
        roll, pitch, yaw = quat_to_rpy(*ee_quat)

        print(f"[{idx:03d}]")
        print(f"  joint : {[round(v, 6) for v in joint]}")
        print(
            "  xyzrpy: "
            f"[{ee_pos[0]:.6f}, {ee_pos[1]:.6f}, {ee_pos[2]:.6f}, "
            f"{roll:.6f}, {pitch:.6f}, {yaw:.6f}]"
        )


if __name__ == "__main__":
    # 1) 在这里设置当前关节角（角度）
    current_joint_deg = [90.0, 0.0, 0.0, -90.0, 0.0, -90.0, 60]
    current_joint = joint_deg_to_rad(current_joint_deg)

    # 2) 在这里设置目标位姿 [x, y, z, roll, pitch, yaw]（rpy 为弧度）
    target_pose = [-0.3, -0.2, 0.3, 0, 0, 0]

    print("Current joint (deg):", current_joint_deg)
    print("Current joint (rad):", [round(v, 6) for v in current_joint])
    print("Target xyzrpy:", target_pose)
    print("Robot cfg:", ROBOT_CFG_PATH)

    check_joint_collision_distance(current_joint)

    traj = plan_with_curobo(current_joint, target_pose)
    if traj is None:
        print("No trajectory generated")
    else:
        print(f"Trajectory points: {len(traj)}")
        print_trajectory_with_xyzrpy(traj)
