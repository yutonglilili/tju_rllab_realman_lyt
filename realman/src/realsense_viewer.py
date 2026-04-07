#!/usr/bin/env python3
"""
RealSense 相机可视化工具

功能：
    - 实时显示 RGB 图像和深度图像
    - 支持鼠标点击查看像素坐标和深度值
    - 按键控制：
        - 'q': 退出
        - 's': 保存当前帧
        - 'p': 打印相机内参
        - 'c': 切换显示模式（RGB/Depth/Both）

使用方法：
    python src/tvla_realenv/realsense_viewer.py --serial f1471338
"""

import argparse
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# 兼容直接以脚本方式运行：把 /home/zhangzhao/lyt/realman 加入导入路径
CURRENT_DIR = Path(__file__).resolve().parent
REALMAN_DIR = CURRENT_DIR.parent
if str(REALMAN_DIR) not in sys.path:
    sys.path.insert(0, str(REALMAN_DIR))

from open3d_realsense_env import Open3dRealsenseEnv


class RealsenseViewer:
    """RealSense 相机可视化类"""
    
    def __init__(self, serial: str, save_dir: str = "captured_frames"):
        """
        初始化
        
        Args:
            serial: 相机序列号
            save_dir: 保存帧的目录
        """
        self.rs_env = Open3dRealsenseEnv(serial)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 显示模式: 0=Both, 1=RGB only, 2=Depth only
        self.display_mode = 0
        
        # 鼠标位置
        self.mouse_pos = None
        self.depth_value = None
        
        # 窗口名称
        self.window_name = "RealSense Viewer"
        
        print("=" * 50)
        print("RealSense Viewer 已启动")
        print("=" * 50)
        print(f"相机序列号: {serial}")
        print(f"图像尺寸: {self.rs_env.meta_obs['size']}")
        print(f"深度缩放: {self.rs_env.meta_obs['depth_scale']}")
        print()
        print("快捷键:")
        print("  q - 退出")
        print("  s - 保存当前帧")
        print("  p - 打印相机内参")
        print("  c - 切换显示模式 (RGB/Depth/Both)")
        print("  鼠标点击 - 显示像素坐标和深度值")
        print("=" * 50)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)
    
    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        将深度图转换为彩色可视化
        
        Args:
            depth: 原始深度图
            
        Returns:
            彩色深度图 (BGR)
        """
        # 归一化到 0-255
        depth_normalized = depth.astype(np.float32)
        depth_normalized = np.clip(depth_normalized, 0, 10000)  # 限制最大深度 10m
        depth_normalized = (depth_normalized / 10000 * 255).astype(np.uint8)
        
        # 应用 colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # 将无效深度（0）设为黑色
        depth_colored[depth == 0] = [0, 0, 0]
        
        return depth_colored
    
    def create_display(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        创建显示图像
        
        Args:
            rgb: RGB 图像
            depth: 深度图像
            
        Returns:
            用于显示的 BGR 图像
        """
        # RGB 转 BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 深度可视化
        depth_colored = self.colorize_depth(depth)
        
        if self.display_mode == 0:
            # 并排显示
            display = np.hstack([bgr, depth_colored])
        elif self.display_mode == 1:
            # 只显示 RGB
            display = bgr
        else:
            # 只显示深度
            display = depth_colored
        
        return display
    
    def add_overlay(self, display: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        添加叠加信息
        
        Args:
            display: 显示图像
            depth: 原始深度图
            
        Returns:
            带叠加信息的图像
        """
        h, w = display.shape[:2]
        
        # 添加模式标签
        mode_labels = ["RGB + Depth", "RGB Only", "Depth Only"]
        cv2.putText(display, f"Mode: {mode_labels[self.display_mode]} (press 'c' to change)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Mode: {mode_labels[self.display_mode]} (press 'c' to change)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 如果有鼠标位置，显示坐标和深度
        if self.mouse_pos is not None:
            mx, my = self.mouse_pos
            
            # 根据显示模式计算实际图像坐标
            if self.display_mode == 0:
                # 并排模式，判断在哪一半
                img_w = w // 2
                if mx < img_w:
                    img_x, img_y = mx, my
                else:
                    img_x, img_y = mx - img_w, my
            else:
                img_x, img_y = mx, my
            
            # 获取深度值
            dh, dw = depth.shape[:2]
            if 0 <= img_x < dw and 0 <= img_y < dh:
                depth_raw = depth[img_y, img_x]
                depth_m = depth_raw / self.rs_env.meta_obs['depth_scale']
                
                # 绘制十字线
                cv2.line(display, (mx - 10, my), (mx + 10, my), (0, 255, 0), 1)
                cv2.line(display, (mx, my - 10), (mx, my + 10), (0, 255, 0), 1)
                
                # 显示坐标和深度
                info_text = f"Pixel: ({img_x}, {img_y}) Depth: {depth_m:.3f}m"
                cv2.putText(display, info_text, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, info_text, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return display
    
    def save_frame(self, rgb: np.ndarray, depth: np.ndarray):
        """保存当前帧"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 RGB
        rgb_path = self.save_dir / f"{timestamp}_rgb.jpg"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # 保存深度（16位 PNG）
        depth_path = self.save_dir / f"{timestamp}_depth.png"
        cv2.imwrite(str(depth_path), depth.astype(np.uint16))
        
        # 保存深度可视化
        depth_vis_path = self.save_dir / f"{timestamp}_depth_vis.jpg"
        cv2.imwrite(str(depth_vis_path), self.colorize_depth(depth))
        
        print(f"帧已保存: {timestamp}")
        print(f"  - RGB: {rgb_path}")
        print(f"  - Depth: {depth_path}")
        print(f"  - Depth Vis: {depth_vis_path}")
    
    def print_intrinsics(self):
        """打印相机内参"""
        meta = self.rs_env.meta_obs
        K = np.array(meta['intrinsic'])
        
        print()
        print("=" * 50)
        print("相机内参")
        print("=" * 50)
        print(f"图像尺寸: {meta['size']} (height, width)")
        print(f"焦距: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        print(f"主点: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
        print(f"深度缩放: {meta['depth_scale']}")
        print(f"畸变系数: {meta['distortion']}")
        print()
        print("内参矩阵 K:")
        print(K)
        print("=" * 50)
    
    def run(self):
        """运行主循环"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        try:
            while True:
                # 获取帧
                obs = self.rs_env.step()
                rgb = obs['rgb']
                depth = obs['depth']
                
                # 创建显示
                display = self.create_display(rgb, depth)
                display = self.add_overlay(display, depth)
                
                # 显示
                cv2.imshow(self.window_name, display)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("退出...")
                    break
                elif key == ord('s'):
                    self.save_frame(rgb, depth)
                elif key == ord('p'):
                    self.print_intrinsics()
                elif key == ord('c'):
                    self.display_mode = (self.display_mode + 1) % 3
                    mode_labels = ["RGB + Depth", "RGB Only", "Depth Only"]
                    print(f"切换到: {mode_labels[self.display_mode]}")
        
        finally:
            self.rs_env.close()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="RealSense 相机可视化工具")
    parser.add_argument("--serial", type=str, default="f1471338",
                        help="相机序列号")
    parser.add_argument("--save_dir", type=str, default="captured_frames",
                        help="保存帧的目录")
    
    args = parser.parse_args()
    
    viewer = RealsenseViewer(args.serial, args.save_dir)
    viewer.run()


if __name__ == "__main__":
    main()

# f1471338
# f1471193