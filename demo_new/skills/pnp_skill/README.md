# PnP Skill 技术文档

## 概述

PnP Skill（Pick and Place Skill）是一个多线程机器人抓取放置系统，集成了 VLM 视觉感知、GraspGen 抓取姿态生成、运动规划与执行验证。系统支持两种抓取模式：

- **启发式抓取**：基于第三视角相机 VLM 打点 + 预设 RPY 的粗抓取
- **GraspGen 精抓取**：基于腕部相机点云 + GraspGen 模型推理 + 夹爪水平矫正的精细抓取

---

## 文件结构

```
pnp_skill/
├── pick_and_place.py      # PnP 主控（三线程架构 + 任务调度）
├── graspgen_bridge.py     # GraspGen ZMQ 客户端 + 点云处理 + 抓取姿态后处理
├── config.yaml            # 默认配置参数
└── README.md              # 本文档
```

---

## 系统架构

### 三线程模型

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   感知线程       │     │   规划线程       │     │   执行线程       │
│ perception_thread│     │ planning_thread  │     │ execution_thread │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ - VLM 打点追踪   │────▶│ - 动作序列生成   │────▶│ - 逐步执行动作   │
│ - 移动检测       │     │ - 预抓取规划     │     │ - GraspGen 候选  │
│ - 腕部 GraspGen  │     │ - GraspGen 就绪  │     │   逐个尝试       │
│ - 抓取/放置验证  │     │   信号转发       │     │ - 兜底抓取       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                          SharedState（线程锁）
```

### 状态机

```
TaskPhase:  IDLE → PICK → PLACE → COMPLETE

PickStage（GraspGen 精抓取分支）:
  IDLE → GLOBAL_TRACKING → PREGRASP_EXECUTING → WRIST_SENSING
       → GRASP_PLAN_READY → GRASP_EXECUTING → VERIFYING
```

---

## Pick 阶段完整流程（GraspGen 精抓取分支）

当 `wrist_grasp_enabled=True` 时，Pick 阶段走以下 6 步流程：

### 步骤 1：全局追踪定位（GLOBAL_TRACKING）

- 感知线程以固定频率调用 VLM 对第三视角 RGB 图像打点
- 获取目标物体的 2D 像素坐标，通过深度图 + 相机外参转换为 base 系 3D 坐标
- 检测目标是否发生移动（与上次打点的 3D 距离超过阈值）
- 首次定位成功后触发 `need_replan` 信号

### 步骤 2：粗预抓取规划与执行（PREGRASP_EXECUTING）

- 规划线程收到 `need_replan` 信号后，基于全局定位结果生成粗预抓取位姿
- 粗预抓取位姿 = 目标位置 + PICK 偏移 + PRE_PICK 偏移（抬高）
- 生成单步动作序列：移动到预抓取位姿 + 张开夹爪
- 执行线程执行该动作，到达预抓取位姿后切换到腕部感知模式

### 步骤 3：腕部感知 + GraspGen 推理（WRIST_SENSING）

感知线程切换到 `wrist_mode`，执行以下流程：

1. **腕部相机采集**：获取腕部 RGBD 帧
2. **VLM 打点**：在腕部 RGB 上对目标物体打点，获取 click_2d
3. **调用 `infer_pick_grasp_candidates_from_wrist`**（详见下方 GraspGen Bridge 章节）

### 步骤 4：抓取计划就绪（GRASP_PLAN_READY）

- 规划线程检测到 `wrist_result_ready` 信号
- 将候选池信息设置到 SharedState
- 触发 `grasp_plan_ready` 信号通知执行线程

### 步骤 5：逐候选执行抓取（GRASP_EXECUTING）

执行线程按置信度顺序逐个尝试候选：

```
for each candidate in grasp_pose_pool_base:
    1. IK 可达性检查（pregrasp → grasp → postgrasp 三点序列）
    2. 如果可达：执行 pregrasp(张开) → grasp(linear闭合) → postgrasp(抬起)
    3. 如果不可达：跳过，尝试下一个
```

如果所有 GraspGen 候选都不可达，回退到启发式兜底抓取（`execute_fallback_pick`）。

### 步骤 6：抓取验证（VERIFYING）

- 感知线程切换到 `verify_mode`
- 通过 VLM 检测 + 距离检测双重验证抓取是否成功
- 成功：进入 PLACE 阶段
- 失败：重试（回到步骤 1），直到达到 `MAX_PICK_RETRIES`

---

## Place 阶段流程

Place 阶段不使用 GraspGen，走启发式路径：

1. 全局追踪定位放置目标（容器/位置）
2. 生成 pre_place → place → post_place 三段动作序列
3. 执行动作（夹爪在 place 点松开）
4. VLM + 距离验证放置是否成功

---

## GraspGen Bridge 详解（graspgen_bridge.py）

### 整体职责

`graspgen_bridge.py` 负责将腕部相机的 RGBD 观测转换为可执行的抓取姿态候选池。它包含：

1. ZMQ 客户端（与 GraspGen 推理服务通信）
2. 点云处理管线（深度图 → 相机系点云 → base 系点云 → 物体分割）
3. 抓取姿态后处理管线（朝向过滤 → 置信度排序 → 水平矫正 → 坐标转换）

### 点云处理管线

```
腕部 RGBD 帧
    │
    ▼
深度图 → 相机系有组织点云 (project_depth_to_xyz)
    │
    ▼
手眼标定 + 当前机器人位姿 → base 系点云 (transform_points_camera_to_base)
    │
    ▼
估计桌面高度 (estimate_table_height) → 去除桌面像素
    │
    ▼
VLM click_2d → 种子点 → 区域生长 (grow_mask_from_seed) → 物体 mask
    │
    ▼
物体点云 + 场景点云（体素降采样）
    │
    ▼
发送给 GraspGen 推理服务
```

### GraspGen 抓取姿态处理管线（定版流程）

```
GraspGen 推理输出（相机系 4x4 姿态）
        │
        ▼
  转到 base 系（base_from_camera @ grasp_camera）
        │
        ▼
  ① 按置信度降序排列
        │
        ▼
  ② Direction Rule 朝向过滤（在相机系，矫正前）
     - 将 base 系姿态转回相机系
     - 检查 approach 方向与目标方向的夹角 ≤ max_angle_deg
     - 检查 forward/down/lateral 分量约束
        │
        ▼
  ③ 取 top-5（在通过筛选的姿态中按原始置信度取前 max_candidates 个）
        │
        ▼
  ④ 夹爪 x 轴水平矫正（level_gripper_x_axis）
     - 绕 y 轴（夹爪平面法线）旋转
     - 使 x 轴（闭合方向）水平（base-z 分量 = 0）
     - 约束 approach 朝下（z_new[2] < 0）
     - 以 TCP（手指中心）为旋转中心，保证夹爪中心不移动
        │
        ▼
  ⑤ 格式转换
     leveled_grasp_base @ grasp_to_tcp → T_tcp2base
     realman_xyzrpy_from_T(T_tcp2base) → tcp_xyzrpy
     pose_tcp2eef(tcp_xyzrpy) → eef_xyzrpy
        │
        ▼
  ⑥ 输出
     - grasp_pose_pool_base: (N, 4, 4) tcp2base 矩阵（供 IK + 执行）
     - grasp_eef_xyzrpy_pool: (N, 6) eef2base xyzrpy（供调试/示教器验证）
     - grasp_pregrasp_pool_base: (N, 4, 4) 预抓取位姿（沿 TCP +x 后退）
     - grasp_pose_pool_scores: (N,) 原始置信度
```

### 水平矫正算法（level_gripper_x_axis）

```
GraspGen 夹爪坐标系定义:
  z = approach（接近方向，夹爪往物体扎的方向）
  x = 闭合方向（两根手指张开/夹紧的连线方向）
  y = 夹爪平面法线（两指 + 接近方向构成的平面的法线）

矫正目标: 让 x 轴（闭合方向）水平，即 x[2] = 0

数学推导:
  绕 y 轴旋转角 θ:
    x_new = cos(θ)*x - sin(θ)*z
    z_new = sin(θ)*x + cos(θ)*z
    y_new = y（不变）

  约束 x_new[2] = 0:
    cos(θ)*x[2] - sin(θ)*z[2] = 0
    θ = atan2(x[2], z[2])

  约束 approach 朝下（z_new[2] < 0）:
    如果 z_new[2] > 0，取 θ + π

  以 TCP 为旋转中心（保证夹爪中心不移动）:
    tcp_pos = origin + depth * z_old    （固定点）
    new_origin = tcp_pos - depth * z_new
```

### grasp_to_tcp 转换

GraspGen 输出的是 gripper base_link 坐标系（原点在夹爪安装面），需要转换到 Realman TCP 坐标系（原点在手指中心）：

```
GraspGen base_link:  approach = +z, closing = +x
Realman TCP:         approach = +x, closing = +y

转换矩阵 C (build_grasp_to_tcp_transform):
  平移: [0, 0, depth]  沿 approach 偏移到手指中心
  旋转: TCP_x = GraspGen_z, TCP_y = GraspGen_x, TCP_z = GraspGen_y

应用: T_tcp2base = grasp2base @ C
```

### GraspGenClientBridge（ZMQ 客户端）

通过 ZMQ REQ/REP 模式与独立的 GraspGen 推理服务通信：

```python
# 发送: 物体点云 (N, 3) float32 + 推理参数
# 接收: grasps (M, 4, 4) float32 + scores (M,) float32
```

推理服务运行在 GPU 机器上，bridge 只负责点云预处理和结果后处理。

---

## 坐标系与位姿约定

| 名称 | 含义 | 原点 | 轴定义 |
|------|------|------|--------|
| base | 机器人基座坐标系 | 基座中心 | 标准右手系 |
| camera | 相机光学坐标系 | 相机光心 | x右 y下 z前 |
| GraspGen base_link | 夹爪安装面坐标系 | 夹爪安装面中心 | z=approach, x=closing, y=法线 |
| Realman TCP | 夹爪中心坐标系 | 手指中心 | x=approach, y=closing, z=法线 |
| Realman EEF | 末端执行器坐标系 | 法兰面 | 通过 T_TCP2REALMANEEF 转换 |

位姿表示：
- 4x4 矩阵：齐次变换矩阵，`T_A2B` 表示 A 在 B 中的位姿
- xyzrpy：6 维向量 `[x, y, z, rx, ry, rz]`，单位 米/弧度，旋转顺序 Rz @ Ry @ Rx

---

## 配置参数说明

### 感知参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| PERCEPTION_INTERVAL | 0.3s | VLM 打点频率 |
| MOVE_OBJECT_THRESHOLD | 0.05m | pick 目标移动检测阈值 |
| MOVE_CONTAINER_THRESHOLD | 0.2m | place 目标移动检测阈值 |
| CAMERA_X/Y/Z_OFFSET | 0.05/-0.01/0.0 | 相机标定偏移修正 |

### GraspGen 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| GRASPGEN_MAX_CANDIDATES | 5 | 最终输出的候选数量 |
| GRASPGEN_NUM_GRASPS | 200 | 采样候选数 |
| GRASPGEN_TOPK_NUM_GRASPS | 50 | 推理后保留的 top-k |
| GRASPGEN_GRASP_THRESHOLD | -1.0 | 置信度阈值（-1 表示不过滤） |
| GRASPGEN_CANDIDATE_PREGRASP_OFFSET_M | 0.10m | 预抓取后退距离 |
| DIRECTION_RULE_MAX_ANGLE_DEG | 35° | 朝向过滤最大角度偏差 |

### 腕部点云处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| WRIST_MIN/MAX_DEPTH_M | 0.10/1.20m | 有效深度范围 |
| WRIST_CLICK_SEED_RADIUS_PX | 5px | 种子区域半径 |
| WRIST_REGION_GROW_3D_THRESHOLD_M | 0.018m | 区域生长 3D 距离阈值 |
| WRIST_TABLE_HEIGHT_PERCENTILE | 8.0 | 桌面高度估计百分位 |
| WRIST_MAX_OBJECT_POINTS | 4096 | 物体点云最大点数 |
| WRIST_OBJECT_VOXEL_SIZE | 0.003m | 物体点云体素大小 |

---

## 使用方式

### 基本用法

```python
from demo_new.skills.pnp_skill.pick_and_place import (
    init_robot_env, init_camera_env, init_wrist_grasp_env,
    init_state, start_pnp_system, shutdown_pnp_system,
    run_single_task, run_all_tasks_by_instruction_with_position_description,
)

# 初始化
env, home_T = init_robot_env("192.168.101.19")
rs_env, cam_results = init_camera_env(serial, cam_results_path)
wrist_rs_env, graspgen_client, wrist_handeye = init_wrist_grasp_env(wrist_serial)

# 创建状态 & 启动系统
state = init_state()
start_pnp_system(state, env, rs_env, cam_results, home_T,
                 wrist_rs_env=wrist_rs_env,
                 graspgen_client=graspgen_client,
                 wrist_handeye_config=wrist_handeye)

# 执行任务
run_single_task(state, env, rs_env, cam_results,
                {"pick": "red cup", "place": "blue plate"}, home_T)

# 关闭
shutdown_pnp_system(state, env=env, rs_env=rs_env,
                    wrist_rs_env=wrist_rs_env, graspgen_client=graspgen_client)
```

### 任务模式

| 函数 | 说明 |
|------|------|
| `run_single_task` | 执行单个 pick-place 任务 |
| `run_all_tasks` | 按明确列表执行多个任务 |
| `run_all_tasks_by_instruction_with_list` | VLM 拆解指令为任务列表并执行 |
| `run_all_tasks_by_instruction_with_position_description` | 带方位描述的指令拆解 |

---

## 调试

- 设置 `DEBUG_EXIT_AFTER_GRASP_CANDIDATES: true` 可在生成候选后立即退出，打印所有候选的 tcp/eef xyzrpy
- `grasp_eef_xyzrpy_pool` 可直接在示教器上验证位姿是否合理
- GraspGen demo 脚本 (`GraspGen/scripts/demo_wrist_camera_graspgen.py`) 提供 viser 网页可视化，颜色含义：
  - 灰色：被朝向过滤掉的
  - 蓝色：通过朝向过滤但不在 top-5
  - 绿色：置信度前 5（矫正前原始姿态）
  - 黄色：矫正后的 top-5
  - 红色：置信度最高的矫正后姿态