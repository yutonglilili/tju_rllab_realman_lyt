unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
cd dependencies/frankapy
source catkin_ws/devel/setup.bash
cd ../..
python reset_franka.py