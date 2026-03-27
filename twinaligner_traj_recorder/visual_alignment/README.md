# GET_RGBD:a Python script designed to capture RGB and depth images 

## Overview
`get_rgbd.py` is a Python script designed to capture RGB and depth images using an Intel RealSense camera. The script saves the captured data in specified directories, including RGB images, depth images, and visualized depth images. It also saves the camera's intrinsic parameters for further use in computer vision tasks.

---
## Requirements
Before running the script, ensure you have the following dependencies installed:

- **Python 3.x**
- **Intel RealSense SDK** (`pyrealsense2`)
- **OpenCV** (`cv2`)
- **NumPy** (`numpy`)

You can also use ```pip3 install -r requirements.txt``` to install.

---

## Usage


You can run the script with the following command:

```python get_rgbd.py--dir_name <output_directory> --preheat_time <preheat_seconds> --fps <frames_per_second> --record_frames <number_of_frames> ```

Or you can use .sh files as below:  

```
# This script is used to test the get_rgbd.py script

export DIR_NAME="data"
export PREHEAT_TIME=1
export RECORD_FRAMES=120
export FPS=15


export PROJECT_ROOT=$(pwd)


python3 $PROJECT_ROOT/get_rgbd.py \
--dir_name $PROJECT_ROOT/$DIR_NAME \
--preheat_time $PREHEAT_TIME \
--fps $FPS \
--record_frames $RECORD_FRAMES 
```

## Arguments 

```--dir_name ``` 
 Directory to save the captured data.

```--preheat_time``` Preheat time in seconds before starting to save frames

```--fps```  
Frames per second for the camera.

```--record_frames```  
Number of frames to capture before stopping.


## Output

Output files are like below:
```
dir_name/
├── cam_K.txt
├── depth/
│   ├── 00001.npz
│   ├── 00002.npz
│   └── 00003.npz
├── rgb/
│   ├── 00001.png
│   ├── 00002.png
│   └── 00003.png
└── vis/
    ├── 00001.png
    ├── 00002.png
    └── 00003.png
```

