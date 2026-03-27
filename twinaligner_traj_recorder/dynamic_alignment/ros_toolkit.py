#!/usr/bin/env python

import rospy
import open3d as o3d
import numpy as np
from std_msgs.msg import Float64MultiArray
from frankapy import FrankaArm  # Frankapy provides an interface with Franka Panda
from geometry_msgs.msg import Transform, Vector3, Quaternion
from scipy.spatial.transform import Rotation as R
class Ros_listener:
    def __init__(self):
        self.joint_state = None
        self.ee_pose = None
        self.ee_velocity = None
        self.joint_state_subscriber = rospy.Subscriber('/franka/joint_states', Float64MultiArray, self.state_callback_joint_state)
        self.ee_pose_subscriber = rospy.Subscriber('/franka/end_effector_pose', Transform, self.state_callback_ee_pose)
        #self.ee_velocity_subscriber = rospy.Subscriber('/franka/end_effector_velocity', Float64MultiArray, self.state_callback_ee_velocity)

    def state_callback_joint_state(self, msg):
        self.joint_state = msg.data

    def state_callback_ee_pose(self, msg):
        self.ee_pose = msg
    def state_callback_ee_velocity(self, msg):
        self.ee_velocity = msg.data

class Ros_publisher:
    def __init__(self, arm=None, vis_pose=False):
        # Initialize Franka Panda robot interface
        if arm is None:
            self.arm = FrankaArm()
        else:
            self.arm = arm
        self.vis_pose = vis_pose

        # Initialize Open3D if visualization is needed
        if self.vis_pose:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window()
            self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.first_frame = True

        rospy.loginfo("initing...")
        # Move the robot to the neutral position (ensure neutral position is valid in your workspace)
        #self.arm.move_to_neutral()

        # Get initial state
        self.current_joint_state = self.arm.get_joints()  # e.g., returns 7 joint angles (unit: radians)
        self.joint_state = self.current_joint_state[:]

        # Get end-effector pose, assuming it returns [x, y, z, roll, pitch, yaw]
        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose



        # Initialize ROS node
        #rospy.init_node('franka_arm_controller', anonymous=True)

        # Publish current joint state
        self.joint_state_publisher = rospy.Publisher('/franka/joint_states', Float64MultiArray, queue_size=1)

        # Publish end-effector pose
        self.end_effector_publisher = rospy.Publisher('/franka/end_effector_pose', Transform, queue_size=1)
        #self.end_effector_velocity_publisher = rospy.Publisher('/franka/end_effector_velocity', Float64MultiArray, queue_size=1)

    def read_joint_state(self):
        # Update current joint state and end-effector pose
        #rospy.loginfo("reading...")
        self.current_joint_state = self.arm.get_joints()
        self.joint_state = self.current_joint_state[:]

        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose

        # Construct and publish joint state message
        joint_state_msg = Float64MultiArray()
        joint_state_msg.data = self.joint_state
        self.joint_state_publisher.publish(joint_state_msg)
        #rospy.loginfo(self.joint_state)
        # Construct and publish end-effector pose message
        ee_pose_msg = Transform()

        # Extract translation (assuming self.ee_pose.translation is a numpy array)
        translation = self.ee_pose.translation
        # Convert numpy array to Vector3
        ee_pose_msg.translation = Vector3(translation[0], translation[1], translation[2])

        # Extract rotation matrix (assuming self.ee_pose.rotation is a 3x3 numpy matrix)
        rotation_matrix = self.ee_pose.rotation

        # Convert the rotation matrix to a quaternion
        rotation = R.from_matrix(rotation_matrix).as_quat()  # Returns [x, y, z, w]

        # Assign the quaternion to the Transform message
        ee_pose_msg.rotation = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        # rospy.loginfo("Published end effector pose: %s", ee_pose_msg)
        self.end_effector_publisher.publish(ee_pose_msg)
        # Construct and publish end-effector velocity message
        # ee_velocity = self.arm.get_ee_velocity()
        # ee_velocity_msg = Float64MultiArray()
        # ee_velocity_msg.data = ee_velocity
        # self.end_effector_velocity_publisher.publish(ee_velocity_msg)

    def _ee_pose_to_matrix(self, ee_pose):
        # Convert end-effector pose ([x, y, z, roll, pitch, yaw]) to a homogeneous transformation matrix
        pos = np.array(ee_pose[:3])
        rpy = ee_pose[3:]
        R = o3d.geometry.get_rotation_matrix_from_xyz(rpy)
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = pos
        return mat

    def _vis_pose(self, ee_pose):
        # Update the end-effector coordinate system shown in Open3D
        if self.first_frame:
            self.ee_frame.transform(self._ee_pose_to_matrix(ee_pose))
            self.o3d_vis.add_geometry(self.base_frame)
            self.o3d_vis.add_geometry(self.ee_frame)
            self.first_frame = False
        else:
            self.o3d_vis.remove_geometry(self.ee_frame)
            self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.ee_frame.transform(self._ee_pose_to_matrix(ee_pose))
            self.o3d_vis.add_geometry(self.ee_frame)
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()

    def run(self):
        rate = rospy.Rate(50)  # Set loop frequency
        while not rospy.is_shutdown():
            self.read_joint_state()
            # if self.vis_pose:
            #     self._vis_pose(self.ee_pose)
            rate.sleep()

    def shutdown(self):
        
        #self.arm.shutdown()
        rospy.loginfo("Franka arm controller shutdown.")



def run_publisher(arm):
    try:
        ros_publisher = Ros_publisher(arm, vis_pose=False)
        ros_publisher.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        ros_publisher.shutdown()
if __name__ == '__main__':
    run_publisher()
