# 机器人PnP视觉任务系统

一个基于Realman机器人与视觉语言模型（VLM）的拾取放置（Pick-and-Place, PnP）系统，支持实时场景理解、智能运动规划和自适应任务执行。

## 🎯 项目概述

该项目实现了一个完整的机器人操作系统，集感知、规划、执行于一体，支持连续的物体拾取和放置任务。系统采用多线程架构，通过VLM进行场景理解，利用机器人SDK进行精准控制。

## 📁 项目结构

### 核心模块

#### `demo/pnp_final/` - 拾取放置系统核心演示

主要实现了基于VLM的连续PnP任务系统，支持物体跟踪、错误检测和重规划。

**主要脚本：**
- `pick_and_place.py` - 核心系统脚本，实现三线程（感知、规划、执行）协同工作
- `multi_pointing_vllm_get_point_utils.py` - VLM推理工具，包括物体定位、成功检测等
- `pick_and_place_utils.py` - 位姿计算、可视化等工具函数
- `pointing_vllm_client.py` - VLLM在线服务客户端
- `pnp_gradio_demo.py` - Gradio交互式演示界面

**功能特性：**
- 📷 实时视觉感知与物体检测（通过VLM）
- 🤖 多物体连续拾取放置操作
- 🔄 执行失败检测与自动重规划
- 🧵 感知/规划/执行三线程并行处理
- 📊 完整的任务日志与可视化

---

#### `realman/` - 机器人环境与控制接口

对Realman机械臂SDK的二次封装，提供统一的机器人控制接口。

**主要文件：**
- `realman_env.py` - **当前推荐版本**，同步/异步双模式控制
  - 同步模式：类Gym接口 `reset()/step()`
  - 异步模式：流式非阻塞控制，支持高频轨迹跟随
  - 统一TCP与EEF位姿转换
  - 夹爪控制与宽度管理

- `realman_env_old.py` - 旧版本实现（保留用于对照）

- `open3d_realsense_env.py` - RealSense相机集成
  - 实时RGB-D图像采集
  - 点云处理与可视化

**架构设计：**
```
工具函数层 (坐标变换、夹爪转换)
    ↓
RobotState (状态快照)
    ↓
RealmanDriver (底层SDK调用)
    ↓
SyncController / AsyncController
    ↓
RealmanEnv (统一对外接口)
```

**依赖：**
- `Robotic_Arm` - Realman官方SDK
- `pytransform3d` - 3D位姿变换
- `numpy` - 数值计算

---

#### `curobo/` - 轨迹规划库（可选）

NVIDIA CuRobo轨迹规划库的本地副本，用于运动规划和碰撞检测（可选）。

---

### 其他相关目录

- `camera/` - 相机标定数据和调试信息
- `captured_frames/` - 捕获的图像帧存储

## 🚀 快速开始

### 环境要求

**硬件：**
- Realman机器人手臂
- Intel RealSense深度相机
- VLLM推理服务（可本地或远程）

**Python依赖：**
```bash
pip install numpy opencv-python pillow gradio pytransform3d openai requests
```

### 基本使用

#### 1. 初始化机器人环境

```python
from realman.realman_env import RealmanEnv

# 同步模式（阻塞式）
env = RealmanEnv(mode="sync")

# 异步模式（流式非阻塞）
env = RealmanEnv(mode="async")
```

#### 2. 运行PnP系统

```bash
cd demo/pnp_final
python pick_and_place.py
```

#### 3. 启动Gradio演示界面

```bash
cd demo/pnp_final
python pnp_gradio_demo.py
```

## 🔧 配置说明

### VLM服务配置

编辑 `demo/pnp_final/multi_pointing_vllm_get_point_utils.py` 中的：
- VLM服务地址（本地/远程）
- 模型名称
- 推理参数

### 机器人参数

`realman/realman_env.py` 中包含：
- TCP/EEF位姿偏移
- 夹爪开口宽度映射
- 安全工作范围

### 任务参数

`demo/pnp_final/pick_and_place.py` 中的配置：
- 感知频率 `PERCEPTION_INTERVAL`
- 高度偏移 `PICK_Z_OFFSET`, `PLACE_Z_OFFSET`
- 安全高度 `SAFE_HEIGHT`

## 📊 系统架构

### 三线程协同工作流程

```
Perception Thread          Planning Thread           Execution Thread
     │                           │                          │
     ├─ 采集RGB-D图像            │                          │
     ├─ VLM场景理解              ├─ 生成任务队列            │
     ├─ 物体定位                 ├─ 轨迹规划                ├─ 执行移动命令
     ├─ 状态监测                 ├─ 碰撞检测                ├─ 实时控制反馈
     │                           │ ← 检测执行失败 ←        │
     └─ 更新场景状态             └─ 自适应重规划           └─ 更新实时状态
```

## 📚 文件使用指南

| 文件 | 用途 | 运行方式 |
|------|------|---------|
| `pick_and_place.py` | 连续PnP任务 | `python pick_and_place.py` |
| `pnp_gradio_demo.py` | 交互式演示 | `python pnp_gradio_demo.py` |
| `test_vlm_pipeline.py` | VLM测试 | 编辑参数后运行 |
| `realman_env.py` | 机器人控制 | 作为库导入使用 |
| `open3d_realsense_env.py` | 相机采集 | 作为库导入使用 |

## 🔍 调试与故障排除

### 查看相机标定数据
```
camera/20260325_031804/cam_intrinsic.json
camera/20260325_031804/camera_results.json
```

### 检查机器人连接
```python
from realman.realman_env import RealmanEnv
env = RealmanEnv(mode="sync")
print(env.get_state())  # 获取当前状态
```

### VLM推理测试
```bash
cd demo
python test_vlm_pipeline.py  # 修改IMAGE_PATH和INSTRUCTION后运行
```

## 📝 开发说明

### 二次开发建议

1. **扩展PnP功能** - 在 `demo/pnp_final/pick_and_place.py` 中修改任务生成逻辑
2. **自定义控制策略** - 继承 `RealmanEnv` 或实现自己的 `Controller`
3. **集成其他传感器** - 参考 `open3d_realsense_env.py` 的集成方式
4. **优化运动规划** - 利用curobo库进行轨迹规划优化

### 代码结构最佳实践

- 使用 `RealmanEnv` 作为机器人控制统一入口
- 将感知、规划、执行逻辑分离到独立线程
- 使用配置文件管理参数，避免硬编码
- 记录详细的执行日志便于调试

## 📞 相关资源

- **Realman SDK文档** - `realman/RM_API2/` 目录
- **机器人配置** - `realman/robot_cfg/` 目录
- **相机数据** - `camera/` 目录中的标定文件

## 📄 许可证

详见各模块的LICENSE文件。

---

**最后更新：** 2026年4月  
**主要开发者贡献：** 多线程PnP系统、VLM集成、机器人环境封装
