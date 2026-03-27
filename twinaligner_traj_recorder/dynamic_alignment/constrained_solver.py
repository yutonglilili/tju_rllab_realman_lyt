import torch
import json
a = torch.zeros(4, device="cuda:0")

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.util_file import get_robot_path

def visualize(ee_translation):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    # Extract x, y, z coordinates
    x = ee_translation[:, 0]
    y = ee_translation[:, 1]
    z = ee_translation[:, 2]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x, y, z, c='r', marker='o', label='Points')

    # Add axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.set_title('3D Point Visualization')
    ax.legend()
     # Get minimum and maximum values of the axes
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Calculate the maximum range for each axis
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Calculate the center of the axes
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    # Set the range of the axes so that each axis has the same scale
    ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
    ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
    ax.set_zlim([z_center - max_range / 2, z_center + max_range / 2])
    plt.show()
def init_curobo(args):
    tensor_args = TensorDeviceType()

    config_file = load_yaml(args.robot)
    urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    # import ipdb
    # ipdb.set_trace()
    #robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    robot_cfg = RobotConfig.from_dict(config_file)
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        None,
        tensor_args,
        interpolation_dt=0.02,
        ee_link_name="ee_link",
    )
    motion_gen = MotionGen(motion_gen_config)
    print("warming up..")
    motion_gen.warmup(warmup_js_trajopt=False)
    # motion_gen.update_locked_joints({"panda_joint6": 1.95144463}, config_file)
    # import ipdb
    # ipdb.set_trace()
    return motion_gen, kin_model

def solve_motion(args, joint_state, ee_translation_goal, ee_orientation_goal, motion_gen, kin_model):
    js_names = ["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
      "panda_joint6","panda_joint7"]

    # compute forward kinematics:
    q = torch.tensor(joint_state).to(device="cuda:0")
    out = kin_model.get_state(q)
    print("fk_ee_position: ", out.ee_position)
    print("fk_ee_quaternion: ", out.ee_quaternion)


    tensor_args = TensorDeviceType()

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=2,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

    #print("Constrained: Holding ")
    pose_cost_metric = PoseCostMetric(
        #hold_partial_pose=True,
        hold_vec_weight=motion_gen.tensor_args.to_device([1, 1, 1, 0, 1, 0]),
    )

    plan_config.pose_cost_metric = pose_cost_metric

    # motion generation:
    cu_js = JointState(
        position=tensor_args.to_device(joint_state),
        velocity=tensor_args.to_device(joint_state) * 0.0,
        acceleration=tensor_args.to_device(joint_state) * 0.0,
        jerk=tensor_args.to_device(joint_state) * 0.0,
        joint_names=js_names,
    )
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

   

    # compute curobo solution:
    ik_goal = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_goal),
    )
    result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

    succ = result.success.item()  
    if succ:
        print("success")
        motion_plan = result.get_interpolated_plan()
        motion_plan = motion_gen.get_full_js(motion_plan)
        motion_plan = motion_plan.get_ordered_joint_state(js_names)

        fk_out=kin_model.get_state(motion_plan.position)

        motion_plan_dict = {
            "position": motion_plan.position.cpu().numpy().tolist(),
            "ee_translation": fk_out.ee_position.cpu().numpy().tolist(),
            "joint_names": motion_plan.joint_names,
        }
        if args.debug >= 2:
            with open("motion_plan.json", "w") as json_file:
                json.dump(motion_plan_dict, json_file, indent=4)
            print("Saved motion_plan to motion_plan.json")
        
        if args.debug >= 1:
            print("visualizing")
            visualize(fk_out.ee_position.cpu().numpy())
        return motion_plan.position.cpu().numpy().tolist(), fk_out.ee_position.cpu().numpy().tolist(), motion_plan.position[-1].cpu().numpy().tolist()

    else:
        print("failed")
        return None
        
    