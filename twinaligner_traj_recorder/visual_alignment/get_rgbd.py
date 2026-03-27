# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Capture RGB and Depth frames using Intel RealSense.")
parser.add_argument("--dir_name", type=str, default="data", help="Directory to save captured frames.")
parser.add_argument("--preheat_time", type=int, default=5, help="Preheat time(s) in seconds before starting to save frames.")
parser.add_argument("--fps", type=int, default=15, help="Frames per second for the camera.")
parser.add_argument("--record_frames", type=int, default=150, help="Number of frames to capture before stopping.")
args = parser.parse_args()

dir_name = args.dir_name #
preheat_time = args.preheat_time
record_frames = args.record_frames
fps = args.fps

print(f"Preheat time: {preheat_time} seconds")
print(f"Record frames: {record_frames}")
print(f"FPS: {fps}")

start_frame = preheat_time * fps


# Create directories to save the images
os.makedirs(dir_name, exist_ok=True)
depth_dir = os.path.join(dir_name, "depth")
color_dir = os.path.join(dir_name, "rgb")
vis_dir = os.path.join(dir_name, "vis")
        
os.makedirs(depth_dir,exist_ok=True)
os.makedirs(color_dir,exist_ok=True)
os.makedirs(vis_dir,exist_ok=True)
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)#
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)#

profile = pipeline.start(config)

# Get the stream profiles for depth and color
depth_profile = profile.get_stream(rs.stream.depth)
color_profile = profile.get_stream(rs.stream.color)

# Get the intrinsics for depth and color streams
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
cam_K = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])
cam_depth_K = np.array([
    [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
    [0, depth_intrinsics.fy, depth_intrinsics.ppy],
    [0, 0, 1]
])
# Save the intrinsic matrix to a file
# Save the intrinsic matrix to a file without brackets
with open(os.path.join(dir_name, "cam_K.txt"), "w") as f:
    for row in cam_K:
        f.write(" ".join(f"{x:.10f}" for x in row) + "\n")
print("Camera intrinsics saved")

profile = pipeline.get_active_profile()
sensor = profile.get_device().query_sensors()[1]
sensor.set_option(rs.option.auto_exposure_priority, True)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    frame_counter = 0
    print("recording...")
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        
        save_index = frame_counter - start_frame
        if save_index>0:#

            #cv2.imwrite(os.path.join(depth_dir, f"frame_{frame_counter}.png"), depth_image)
            np.savez_compressed(os.path.join(depth_dir, f"{save_index:05d}.npz"), depth=depth_image.astype(np.float32)/1000.0)#
            cv2.imwrite(os.path.join(color_dir, f"{save_index:05d}.png"), color_image)
            cv2.imwrite(os.path.join(vis_dir, f"{save_index:05d}.png"), depth_colormap)

        # Increment the frame counter
        frame_counter += 1

        # Display the frames (optional)
        # cv2.imshow('Color Frame', color_image)
        # cv2.imshow('Depth Frame', depth_image)
        # Break the loop if 'q' is pressed
        if save_index == record_frames:#
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("finished recording")