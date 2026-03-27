unset ROS_DISTRO
source /opt/ros/noetic/local_setup.bash
cd dependencies/frankapy/
source catkin_ws/devel/setup.bash
# Get all process PIDs matching "franka" but excluding the script name
PIDS=$(pgrep -f "franka" | grep -v "$$")

if [ -n "$PIDS" ]; then
    echo "Killing processes: $PIDS"
    sudo kill -9 $PIDS
else
    echo "No matching processes found."
fi
bash bash_scripts/start_control_pc.sh -i localhost