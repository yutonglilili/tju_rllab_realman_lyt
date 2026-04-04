from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

model = CudaRobotModel(robot.kinematics)

print("Joint names:", model.joint_names)
print("Link names:", model.link_names)
print("EE link:", model.ee_link)
