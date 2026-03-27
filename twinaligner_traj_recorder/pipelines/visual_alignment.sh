# This script is used to test the get_rgbd.py script

export DIR_NAME="records/franka-track"
export PREHEAT_TIME=5
export RECORD_FRAMES=10
export FPS=15
export PROJECT_ROOT=$(pwd)

python3 visual_alignment/get_rgbd.py \
--dir_name $DIR_NAME \
--preheat_time $PREHEAT_TIME \
--fps $FPS \
--record_frames $RECORD_FRAMES 