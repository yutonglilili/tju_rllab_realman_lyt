# PnP Final - 机器人拾取放置系统

## 项目概述

这是一个基于视觉语言模型 (VLM) 的机器人拾取放置 (Pick-and-Place, PnP) 系统。该系统实现了连续的 PnP 任务执行，支持物体跟踪、错误检测和重规划功能。通过集成感知、规划和执行三个线程，实现高效的机器人操作。

## 主要功能

- **连续 PnP 任务执行**: 支持多个物体的连续拾取和放置操作
- **实时感知**: 使用 VLM 进行场景理解和物体定位
- **智能规划**: 自动生成任务序列和运动规划
- **错误检测与重规划**: 实时检测执行失败并进行重新规划
- **可视化界面**: 提供 Gradio 演示界面进行任务演示
- **多线程协同**: 感知、规划、执行三线程并行工作，提高效率

## 系统架构

系统采用三线程架构：

1. **感知线程 (Perception Thread)**: 负责实时场景感知和物体检测
2. **规划线程 (Planning Thread)**: 基于感知结果生成执行计划
3. **执行线程 (Execution Thread)**: 控制机器人执行拾取和放置动作

## 文件说明

### 核心脚本

- `pick_and_place.py`: 主系统脚本，实现连续 PnP 任务执行的核心逻辑
- `multi_pointing_vllm_get_point_utils.py`: VLM 推理工具函数，包括物体定位、成功检测等
- `pick_and_place_utils.py`: 拾取放置工具函数，包括位姿计算、可视化等
- `pointing_vllm_client.py`: VLLM 在线客户端，用于与 VLM 服务通信

### 演示和测试

- `pnp_gradio_demo.py`: Gradio 演示界面，提供用户友好的任务演示
- `test_vlm_pipeline.py`: VLM 管道测试脚本，用于验证 VLM 功能

## 安装要求

### 依赖包

```bash
pip install numpy opencv-python pillow gradio pytransform3d openai requests
```

### 硬件要求

- Realman 机器人手臂
- RealSense 深度相机
- VLLM 服务 (配置在 `multi_pointing_vllm_get_point_utils.py` 中)

### 环境配置

1. 确保 VLLM 服务运行在指定地址 (默认: `http://172.28.102.11:22002/v1`)
2. 配置机器人环境路径
3. 设置图像保存目录

## 使用方法

### 1. 运行演示界面

```bash
python pnp_gradio_demo.py
```

这将启动 Gradio 界面，您可以通过浏览器访问进行任务演示。

### 2. 直接运行系统

```python
from pick_and_place import run_all_tasks_by_instruction

# 执行指令
instruction = "Clear the table. Pick all toys and place them on the white plate."
run_all_tasks_by_instruction(instruction)
```

### 3. 测试 VLM 管道

修改 `test_vlm_pipeline.py` 中的 `IMAGE_PATH` 和 `INSTRUCTION`，然后运行：

```bash
python test_vlm_pipeline.py
```

## 配置参数

### 感知参数

- `PERCEPTION_INTERVAL`: 感知间隔时间 (默认: 0.5秒)
- `TASK_DISCOVERY_INTERVAL`: 任务发现间隔 (默认: 2.0秒)
- `MOVE_OBJECT_THRESHOLD`: 物体移动检测阈值 (默认: 0.05米)

### 执行参数

- `MAX_PICK_RETRIES`: 拾取最大重试次数 (默认: 5)
- `MAX_PLACE_RETRIES`: 放置最大重试次数 (默认: 5)
- `GRIPPER_OPEN/CLOSE`: 夹爪开合角度

### VLM 配置

- `BASE_URL`: VLLM 服务地址
- `MODEL_NAME`: 使用的模型名称 (默认: "Embodied-R1.5-SFT-0128")

## API 参考

### 主要函数

#### pick_and_place.py

- `run_all_tasks_by_instruction(instruction)`: 根据指令执行所有任务
- `run_all_tasks_by_instruction_with_list(instruction, task_list)`: 根据指令和任务列表执行

#### multi_pointing_vllm_get_point_utils.py

- `get_point_vllm(image, prompt)`: 使用 VLM 获取物体位置
- `check_grasp_success_vllm(image, point, object_name)`: 检查抓取成功
- `generate_task_from_scene(image, instruction)`: 从场景生成任务

#### pick_and_place_utils.py

- `make_target_T(point_3d, rx_degree)`: 创建目标变换矩阵
- `save_check_image(image, prefix, object_name)`: 保存检查图像

## 日志和调试

- 系统会自动保存执行过程中的图像到 `SAVE_DIR` 目录
- 日志文件保存在 `lyt/logs` 目录
- 支持多种检测模式：自动化检测、跳过检测、人工检测

## 注意事项

1. 确保机器人安全区域内无障碍物
2. VLM 服务需要稳定的网络连接
3. 相机标定要准确，以确保定位精度
4. 执行前请检查机器人零点位置

## 贡献

欢迎提交 Issue 和 Pull Request 来改进系统。

## 许可证

[请添加许可证信息]