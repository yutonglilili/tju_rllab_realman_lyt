# cuRobo 使用指南

本文基于当前仓库里的 `curobo` 源码、`examples/` 和 `tests/` 整理，目标不是复述官方文档，而是帮你快速回答几个实际问题：

- `curobo` 到底能做什么？
- 我应该优先用哪个类？
- 初始化时要传哪些对象？
- 最常用的函数分别是什么？
- 返回结果里哪些字段最重要？

## 1. cuRobo 是什么

`cuRobo` 是 NVIDIA 的一个 CUDA 加速机器人库，核心能力集中在下面几类：

- 正向/逆向运动学
- 机器人与环境碰撞检测
- 数值优化
- 图搜索规划
- 轨迹优化
- 运动生成（把 IK + 图搜索 + 轨迹优化串起来）
- MPC（模型预测控制）
- 机器人分割、点云/深度图相关工具

如果你只关心“机械臂从当前状态规划到目标位姿”，实际最常用的是高层封装：

- `curobo.wrap.reacher.motion_gen.MotionGen`
- `curobo.wrap.reacher.ik_solver.IKSolver`
- `curobo.wrap.reacher.trajopt.TrajOptSolver`
- `curobo.wrap.reacher.mpc.MpcSolver`

## 2. 目录结构怎么读

建议按这个顺序理解：

- `src/curobo/wrap/reacher/`
  - 高层 API，最适合业务代码直接调用
- `src/curobo/types/`
  - 常用数据结构，如 `Pose`、`JointState`、`RobotConfig`
- `src/curobo/geom/`
  - 环境建模、障碍物、碰撞世界
- `src/curobo/cuda_robot_model/`
  - 机器人模型、正运动学、自碰撞球等底层能力
- `src/curobo/wrap/model/`
  - 机器人-世界碰撞、机器人分割等中层封装
- `examples/`
  - 最值得抄的调用样例
- `src/curobo/content/configs/`
  - 机器人、世界、任务参数配置文件

你如果是“要把它接到自己的机械臂任务里”，最值得优先看的文件是：

- `examples/motion_gen_api_example.py`
- `examples/ik_example.py`
- `examples/trajopt_example.py`
- `examples/mpc_example.py`
- `examples/world_representation_example.py`

## 3. 先记住 5 个核心数据类型

### 3.1 `TensorDeviceType`

作用：统一张量设备和精度。

最常见写法：

```python
tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
```

如果你用 cuRobo，默认就应该假设主要跑在 CUDA 上。

### 3.2 `Pose`

文件：`src/curobo/types/math.py`

作用：表示末端或某个 link 的笛卡尔位姿。

重点：

- `position` 形状通常是 `[B, 3]`
- `quaternion` 顺序是 `[w, x, y, z]`
- 如果给的是 goalset，可用 `[B, N, 3]` 和 `[B, N, 4]`

常见写法：

```python
goal_pose = Pose(
    position=tensor_args.to_device([[0.5, 0.0, 0.3]]),
    quaternion=tensor_args.to_device([[1.0, 0.0, 0.0, 0.0]]),
)
```

常用辅助函数：

- `Pose.from_list(...)`
- `Pose.from_matrix(...)`
- `Pose.repeat(...)`
- `Pose.get_index(...)`
- `Pose.tolist()`

### 3.3 `JointState`

文件：`src/curobo/types/state.py`

作用：表示关节状态，除了位置，还能带速度、加速度、jerk。

最常用构造：

```python
q_start = JointState.from_position(
    tensor_args.to_device([[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]]),
    joint_names=[
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
)
```

常用函数：

- `JointState.from_position(...)`
- `JointState.from_numpy(...)`
- `JointState.get_state_tensor()`
- `JointState.repeat_seeds(...)`
- `JointState.clone()`

### 3.4 `RobotConfig`

文件：`src/curobo/types/robot.py`

作用：机器人模型配置。

两种常用来源：

1. 从 cuRobo 自带 yaml 读入
2. 直接从 URDF + base_link + ee_link 构造

常用函数：

- `RobotConfig.from_dict(...)`
- `RobotConfig.from_basic(...)`

### 3.5 `WorldConfig`

文件：`src/curobo/geom/types.py`

作用：表示环境障碍物。

支持的障碍物类型：

- `Cuboid`
- `Mesh`
- `Sphere`
- `Cylinder`
- `Capsule`
- `BloxMap`
- `VoxelGrid`

最常用是 `Cuboid` 和 `Mesh`。

示例：

```python
world = WorldConfig(
    cuboid=[
        Cuboid(
            name="obs_1",
            pose=[0.9, 0.0, 0.5, 1, 0, 0, 0],
            dims=[0.1, 0.5, 0.5],
        )
    ]
)
```

常用函数：

- `WorldConfig.from_dict(...)`
- `WorldConfig.add_obstacle(...)`
- `WorldConfig.get_obstacle(...)`
- `WorldConfig.get_cache_dict()`
- `WorldConfig.get_mesh_world(...)`
- `WorldConfig.get_collision_check_world(...)`
- `WorldConfig.save_world_as_mesh(...)`

## 4. 我到底该用哪个类

### 4.1 如果你要“从当前状态规划到目标位姿”

优先用 `MotionGen`。

它是最高层封装，会自动串起来：

- IK 求解
- 图搜索兜底
- 轨迹优化
- 轨迹插值

这是最适合 pick-and-place、抓取、笛卡尔到达任务的入口。

### 4.2 如果你只要“求目标位姿对应的关节解”

用 `IKSolver`。

适合：

- 抓取姿态筛选
- reachability 检查
- 在线 IK
- 给其他规划器提供 joint seed

### 4.3 如果你已经知道目标关节或 seed，只想做轨迹优化

用 `TrajOptSolver`。

适合：

- 从已有路径做平滑
- 已有 joint goal，只想最小 jerk 优化
- 自己控制 seed 轨迹

### 4.4 如果你做闭环控制，每个控制周期都要滚动优化

用 `MpcSolver`。

适合：

- 实时跟踪
- 连续 servo
- 控制回路中的局部避障

但要注意：`MpcSolver` 是局部优化器，不擅长全局绕障。全局任务还是优先 `MotionGen`。

### 4.5 如果你只想做 FK / 自碰撞 / 机器人-环境碰撞

用更底层的模块：

- `CudaRobotModel`
- `RobotWorld`

### 4.6 如果你要从深度图里去除机器人本体

用 `RobotSegmenter`。

## 5. 最常用的一条主线：`MotionGen`

## 5.1 初始化

最常见初始化方式是：

```python
import torch

from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

motion_gen_cfg = MotionGenConfig.load_from_robot_config(
    "franka.yml",
    "collision_table.yml",
    tensor_args,
    num_ik_seeds=50,
    num_trajopt_seeds=6,
    trajopt_tsteps=34,
    interpolation_steps=5000,
    interpolation_dt=0.02,
)

motion_gen = MotionGen(motion_gen_cfg)
motion_gen.warmup()
```

### 5.2 初始化最重要的函数

`MotionGenConfig.load_from_robot_config(...)`

这是你最应该先学会的函数。它会自动把下面几部分组装起来：

- `RobotConfig`
- `IKSolver`
- 图规划器
- `TrajOptSolver`
- 世界碰撞检测器

最重要的参数：

- `robot_cfg`
  - 机器人配置，可以是 yaml 路径、dict、`RobotConfig`
- `world_model`
  - 环境配置，可以是 yaml 路径、dict、`WorldConfig`
- `num_ik_seeds`
  - IK 并行 seed 数
- `num_graph_seeds`
  - 图规划 seed 数
- `num_trajopt_seeds`
  - 轨迹优化 seed 数
- `trajopt_tsteps`
  - 轨迹优化离散步数
- `interpolation_steps`
  - 输出插值轨迹 buffer 大小
- `interpolation_dt`
  - 插值轨迹时间分辨率
- `use_cuda_graph`
  - 是否启用 CUDA graph，通常建议开
- `collision_checker_type`
  - 环境碰撞检测类型
- `collision_cache`
  - 动态更新环境时很重要，决定缓存容量

### 5.3 运行前建议先调用

`motion_gen.warmup(...)`

作用：

- 预热 CUDA graph
- 预先建立常用 buffer
- 降低第一次规划延迟

如果你后续要用 goalset、batch、batch-env，最好按照最大规模提前 warmup。

### 5.4 最核心的规划函数

#### `plan_single(start_state, goal_pose, plan_config=...)`

单个起点到单个笛卡尔目标。

这是最常用函数。

#### `plan_goalset(start_state, goal_pose, plan_config=...)`

单个起点，到一组候选抓取位姿中的任意一个。

典型用途：

- 抓取姿态集合
- 多个可接受末端姿态

#### `plan_single_js(start_state, goal_state, plan_config=...)`

单个起点到目标关节角。

适合：

- 你已经有 joint goal
- 不需要先做 IK

#### `plan_batch(...)`

同一个世界里，批量规划多个 query。

#### `plan_batch_env(...)`

不同世界环境下批量规划。

这个模式要求初始化时就给足 `n_collision_envs` 或对应世界缓存。

#### `plan_grasp(...)`

抓取专用高层接口，做 approach / grasp / retract 的组合规划。

如果你现在只是做普通 reach，不需要先用它。

### 5.5 `MotionGenPlanConfig` 是什么

这个类是“单次规划请求配置”，和初始化配置不同。

最常用字段：

- `enable_graph`
  - 是否启用图规划兜底
- `enable_opt`
  - 是否启用轨迹优化
- `max_attempts`
  - 最大尝试次数
- `timeout`
  - 规划超时时间
- `enable_graph_attempt`
  - 第几次失败后开始启用图规划
- `success_ratio`
  - batch 模式成功率阈值
- `check_start_validity`
  - 是否检查起点合法性
- `enable_finetune_trajopt`
  - 是否进行二次精修
- `parallel_finetune`
  - 是否并行精修
- `time_dilation_factor`
  - 规划完成后整体放慢轨迹

经验上，常用起步配置是：

```python
plan_cfg = MotionGenPlanConfig(
    enable_graph=True,
    max_attempts=10,
    timeout=5.0,
    enable_finetune_trajopt=True,
)
```

### 5.6 结果怎么看

`plan_*` 返回 `MotionGenResult`。

最重要字段：

- `success`
  - 是否成功
- `valid_query`
  - query 是否有效，起点可能本身就碰撞或越界
- `status`
  - 失败原因或成功状态
- `optimized_plan`
  - 优化后的原始轨迹
- `optimized_dt`
  - 原始轨迹步长
- `interpolated_plan`
  - 插值后的轨迹
- `interpolation_dt`
  - 插值步长
- `goalset_index`
  - goalset 模式下最终命中的目标索引
- `used_graph`
  - 是否用了图规划 seed
- `solve_time`
  - 求解时间
- `motion_time`
  - 轨迹执行时间

最常用结果函数：

- `result.get_interpolated_plan()`
- `result.get_paths()`
- `result.get_successful_paths()`
- `result.retime_trajectory(...)`

### 5.7 最小可用模板

```python
import torch

from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

motion_gen = MotionGen(
    MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        interpolation_dt=0.02,
    )
)

motion_gen.warmup()

world = WorldConfig(
    cuboid=[
        Cuboid(
            name="obs_1",
            pose=[0.9, 0.0, 0.5, 1, 0, 0, 0],
            dims=[0.1, 0.5, 0.5],
        )
    ]
)
motion_gen.update_world(world)

start_state = JointState.from_position(
    tensor_args.to_device([[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]])
)
goal_pose = Pose(
    position=tensor_args.to_device([[0.5, 0.0, 0.3]]),
    quaternion=tensor_args.to_device([[1.0, 0.0, 0.0, 0.0]]),
)

result = motion_gen.plan_single(
    start_state,
    goal_pose,
    MotionGenPlanConfig(enable_graph=True, max_attempts=10),
)

if result.success.item():
    traj = result.get_interpolated_plan()
else:
    print(result.status)
```

## 6. 只做逆解：`IKSolver`

### 6.1 什么时候用

适合：

- 检查目标位姿能不能到
- 一次返回多个 IK 解
- 给抓取任务做候选筛选
- 给轨迹优化提供 seed

### 6.2 初始化

```python
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

ik_solver = IKSolver(
    IKSolverConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args=tensor_args,
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
        self_collision_check=True,
        use_cuda_graph=True,
    )
)
```

### 6.3 最常用函数

- `solve_single(goal_pose, ...)`
  - 单个位姿求 IK
- `solve_goalset(goal_pose, ...)`
  - 多个位姿里任选一个成功
- `solve_batch(goal_pose, ...)`
  - 批量位姿求 IK
- `solve_batch_goalset(goal_pose, ...)`
  - 批量 + goalset
- `solve_batch_env(goal_pose, ...)`
  - 不同环境下批量求 IK
- `update_world(world)`
  - 更新碰撞环境
- `sample_configs(n)`
  - 采样关节配置
- `fk(q)`
  - 计算正运动学

### 6.4 返回结果 `IKResult`

关键字段：

- `success`
- `solution`
  - 纯 tensor 形式的关节解
- `js_solution`
  - `JointState` 形式的解
- `position_error`
- `rotation_error`
- `solve_time`
- `goalset_index`

有用函数：

- `get_unique_solution(...)`
- `get_batch_unique_solution(...)`

如果你想一次求多个 IK seed，再去重，这两个函数很好用。

## 7. 只做轨迹优化：`TrajOptSolver`

### 7.1 什么时候用

适合：

- 已有 start / goal joint state
- 已有 seed trajectory
- 想只做最小 jerk 和避障优化
- 不想走完整个 MotionGen 流程

### 7.2 初始化

```python
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig

trajopt_solver = TrajOptSolver(
    TrajOptSolverConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args=tensor_args,
        num_seeds=2,
        traj_tsteps=32,
        interpolation_dt=0.02,
        use_cuda_graph=False,
    )
)
```

### 7.3 最常用函数

- `solve_single(goal, seed_traj=None, ...)`
  - 单个 query 优化
- `solve_goalset(goal, ...)`
- `solve_batch(goal, ...)`
- `solve_batch_env(goal, ...)`
- `solve(...)`
  - 通用入口
- `get_seed_set(...)`
  - 获取内部构造的 seed
- `get_interpolated_trajectory(...)`
  - 对优化后轨迹插值
- `fk(q)`
  - 正运动学
- `attach_spheres_to_robot(...)`
  - 给机器人附加物体球近似，便于抓取后碰撞检测
- `detach_spheres_from_robot(...)`

### 7.4 `Goal` 怎么传

`TrajOptSolver` 通常不是直接传 `Pose`，而是传 `Goal`：

```python
from curobo.rollout.rollout_base import Goal

goal = Goal(
    current_state=start_state,
    goal_pose=goal_pose,
)
```

如果目标是关节空间：

```python
goal = Goal(
    current_state=start_state,
    goal_state=goal_state,
)
```

### 7.5 返回结果 `TrajOptResult`

关键字段：

- `success`
- `solution`
  - 优化后的 `JointState` 轨迹
- `interpolated_solution`
- `optimized_dt`
- `position_error`
- `rotation_error`
- `cspace_error`
- `smooth_error`
- `goalset_index`
- `solve_time`

## 8. 实时闭环：`MpcSolver`

### 8.1 什么时候用

适合：

- 每个控制周期滚动输出下一个动作
- 连续追踪某个目标位姿
- 局部避障

不适合：

- 有复杂遮挡、需要明显绕障的全局规划

### 8.2 初始化

```python
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

mpc = MpcSolver(
    MpcSolverConfig.load_from_robot_config(
        "franka.yml",
        "collision_test.yml",
        tensor_args=tensor_args,
        step_dt=0.03,
        store_rollouts=True,
    )
)
```

### 8.3 调用流程

`MpcSolver` 的用法和前面几个不太一样，它分成两步：

1. 用 `Goal` 创建求解 buffer
2. 循环调用 `step(current_state)`

示例：

```python
from curobo.rollout.rollout_base import Goal

goal = Goal(
    current_state=start_state,
    goal_pose=goal_pose,
)

goal_buffer = mpc.setup_solve_single(goal, 1)
mpc.update_goal(goal_buffer)

result = mpc.step(start_state, 1)
next_action = result.action
```

### 8.4 最常用函数

- `setup_solve_single(goal, num_seeds=None)`
- `setup_solve_goalset(...)`
- `setup_solve_batch(...)`
- `setup_solve_batch_env(...)`
- `update_goal(goal_buffer)`
- `step(current_state, shift_steps=1, seed_traj=None, max_attempts=1)`
- `update_world(world)`
- `enable_cspace_cost(enable=True)`
- `enable_pose_cost(enable=True)`
- `get_visual_rollouts()`

### 8.5 结果怎么看

`step()` 返回的是 `WrapResult` 类型结果，最关键通常是：

- `result.action`
  - 下一步关节命令
- `result.metrics`
  - 当前优化指标
- `result.solve_time`

你的控制回路里最常见就是：

- 把 `result.action` 发送给机器人
- 把机器人新状态作为下一次 `current_state`

## 9. 低层功能模块

## 9.1 `CudaRobotModel`

适合只做底层机器人模型计算。

常用方式：

```python
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

robot_cfg = RobotConfig.from_dict(...)
kin_model = CudaRobotModel(robot_cfg.kinematics)
state = kin_model.get_state(q)
```

用途：

- FK
- link pose
- link collision spheres
- 自碰撞相关几何信息

如果你只想拿末端位姿，不需要上来就用 `MotionGen`。

## 9.2 `RobotWorld`

文件：`src/curobo/wrap/model/robot_world.py`

适合：

- 单独计算机器人与环境的碰撞距离
- 采样可行关节状态
- 把 cuRobo 当作 differentiable collision layer

常用函数：

- `RobotWorldConfig.load_from_config(...)`
- `get_kinematics(q)`
- `get_collision_distance(x_sph, ...)`
- `get_self_collision_distance(x_sph)`
- `get_world_self_collision_distance_from_joints(q, ...)`
- `update_world(world)`
- `clear_world_cache()`

## 9.3 `RobotSegmenter`

文件：`src/curobo/wrap/model/robot_segmenter.py`

适合：

- 从深度图中分割机器人本体
- 做视觉前处理

常用函数：

- `RobotSegmenter.from_robot_file(...)`
- `update_camera_projection(camera_obs)`
- `get_pointcloud_from_depth(camera_obs)`
- `get_robot_mask(camera_obs, joint_state)`

## 10. 环境建模怎么做

### 10.1 最推荐的两种障碍物

开发早期最推荐：

- `Cuboid`
- `Mesh`

原因：

- `Cuboid` 简单稳定，适合快速调试
- `Mesh` 适合真实场景几何

### 10.2 `WorldConfig` 更新世界

大多数高层类都支持：

- `motion_gen.update_world(world)`
- `ik_solver.update_world(world)`
- `mpc.update_world(world)`
- `robot_world.update_world(world)`

但有个关键限制：

新世界里的障碍物数量不能超过初始化时预分配的 collision cache 容量。

所以如果你要频繁动态换障碍物，初始化时要把 `collision_cache` 设够大。

### 10.3 常见世界配置来源

1. 直接代码构造 `WorldConfig`
2. 用 yaml 文件
3. 从 mesh/scene 转换

路径辅助函数在 `curobo.util_file`：

- `get_robot_configs_path()`
- `get_world_configs_path()`
- `get_task_configs_path()`
- `load_yaml(...)`
- `join_path(...)`

## 11. 典型使用建议

### 11.1 你在做 pick-and-place

建议路线：

1. `MotionGen` 做 approach / transfer / retreat 规划
2. 抓取候选多时，用 `plan_goalset(...)`
3. 抓取后如果要把工件也纳入碰撞，可在 `TrajOptSolver` 层附加球模型

### 11.2 你已经有目标关节角

建议直接：

- `MotionGen.plan_single_js(...)`

或者更低层：

- `TrajOptSolver.solve_single(...)`

### 11.3 你需要实时控制

建议：

- 全局先 `MotionGen`
- 局部跟踪用 `MpcSolver`

### 11.4 你只做可达性筛选

建议：

- `IKSolver.solve_single(...)`
- 批量场景用 `solve_batch(...)`
- 多抓取候选用 `solve_goalset(...)`

## 12. 最重要的坑

### 12.1 四元数顺序是 `wxyz`

不是很多库常见的 `xyzw`。

### 12.2 `use_cuda_graph=True` 时，问题形状不要随便变

源码里多个 solver 都明确依赖固定的：

- batch 大小
- seed 数量
- 求解类型

如果你频繁切换问题形状，可能需要：

- 提前按最大规模 `warmup`
- 或者关闭 `use_cuda_graph`

代价是性能会下降。

### 12.3 第一次调用通常慢，先 `warmup()`

特别是 `MotionGen`。

### 12.4 关节顺序一定要匹配内部 joint order

如果你的外部 joint name 顺序和内部不一致，要先重排。

高层接口里常见辅助：

- `get_active_js(...)`
- `get_full_js(...)`

### 12.5 `update_world()` 不是无限扩容

障碍物缓存是预分配的，超过容量会有问题。

### 12.6 `MpcSolver` 不等于全局规划器

它是局部滚动优化器，容易被障碍或关节极限卡住。

## 13. 一个实用的选型结论

如果你现在要把 `curobo` 接到你自己的机械臂项目里，推荐优先顺序是：

1. 先用 `MotionGen` 跑通单目标笛卡尔规划
2. 再接 `plan_goalset(...)` 做多抓取姿态
3. 需要动态避障或闭环跟踪时再上 `MpcSolver`
4. 需要做候选筛选时单独调用 `IKSolver`
5. 需要更底层碰撞距离或视觉分割时再看 `RobotWorld` / `RobotSegmenter`

也就是说，绝大多数任务里：

- 主规划器：`MotionGen`
- 逆解工具：`IKSolver`
- 实时局部控制：`MpcSolver`
- 底层几何和碰撞：`CudaRobotModel`、`RobotWorld`

## 14. 建议你下一步怎么读源码

如果你想继续深入，建议按这个顺序读：

1. `examples/motion_gen_api_example.py`
2. `src/curobo/wrap/reacher/motion_gen.py`
3. `src/curobo/wrap/reacher/ik_solver.py`
4. `src/curobo/wrap/reacher/trajopt.py`
5. `src/curobo/wrap/reacher/mpc.py`
6. `src/curobo/types/math.py`
7. `src/curobo/types/state.py`
8. `src/curobo/geom/types.py`

如果你的目标是“尽快接项目”，其实前 4 个已经够用了。
