import yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

robot_cfg = "/home/zhangzhao/lyt/realman/robot_cfg/rm75_ag2f90c/rm75_ag2f90c_curobo.yml"

with open(robot_cfg, "r") as f:
    robot_dict = yaml.safe_load(f)

tensor_args = TensorDeviceType()

robot = RobotConfig.from_dict(
    robot_dict,
    tensor_args
)

print("Robot loaded successfully")

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

model = CudaRobotModel(robot.kinematics)

print("Joint names:", model.joint_names)
print("Link names:", model.link_names)
print("EE link:", model.ee_link)

import torch
from curobo.types.state import JointState

q = torch.zeros((1,7), device="cuda")

js = JointState.from_position(q, joint_names=model.joint_names)

state = model.compute_kinematics(js)

print("EE position:", state.ee_pose)

#———————————————————————————————————————
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.state import JointState
import torch
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

world_config = RobotWorldConfig.load_from_config(robot_cfg, use_collision=True)  # 添加 use_collision=True
_robot_world = RobotWorld(world_config)
# 随便定义一个关节位置
q = torch.tensor([[0.0, -0.5, 0.8, 0.0, 0.5, 0.0, 0.0]], device="cuda")
d_world, d_self = _robot_world.get_world_self_collision_distance_from_joints(q)

print("World collision distance:", d_world)
print("Self collision distance:", d_self)
