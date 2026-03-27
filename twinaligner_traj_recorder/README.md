# TwinAligner Trajectory Recorder

## Dependencies

- Ubuntu 20.04
- CUDA 11.8
- ROS noetic

## Installation

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8

git clone --recurse-submodules ssh://git@github.com:TwinAligner/twinaligner_traj_recorder.git
conda create -n twinaligner_recorder python=3.8
conda activate twinaligner_recorder
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e dependencies/curobo --no-build-isolation --verbose
ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 $CONDA_PREFIX/lib/libffi.so.7
```

### Franka Control

```bash
unset ROS_DISTRO && source /opt/ros/noetic/local_setup.bash
pip install -e dependencies/frankapy
cd dependencies/frankapy && ./bash_scripts/make_catkin.sh && cd ../..
```

## Run

### Start Franka-interface Daemon Process

```bash
bash pipelines/start_franka.sh
bash pipelines/reset_franka.sh
```

### View Alignment Collection

```bash
conda activate twinaligner_recorder
bash pipelines/visual_alignment.sh
```

### Dynamic Alignment Collection

```bash
conda activate twinaligner_recorder
bash pipelines/dynamic_alignment.sh
```