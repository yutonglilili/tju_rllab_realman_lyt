# Franka 动力学对齐数采迁移到 Realman

本说明总结了如何把 `twinaligner_traj_recorder/dynamic_alignment/pushing.py`（Franka + ROS 控制）迁移到适配你的 Realman 机械臂（通过 Realman 官方 SDK / `realman_env.py` 控制），并给出一份可直接使用的 Realman 数采脚本与所需配置文件清单。

## 1. 原始 Franka 数据采集流程做了什么

Franka 版本的核心脚本是：

- `twinaligner_traj_recorder/dynamic_alignment/pushing.py`

整体流程可以拆成 3 段：

1. **cuRobo 规划（动力学对齐/约束规划）**
   - `constrained_solver.py:init_curobo(args)`：读取 `dynamic_alignment/franka.yml`，构建 cuRobo 的 `MotionGen` 与 `CudaRobotModel`。
   - `pushing.py:generate_cmd(...)`：根据起始末端姿态计算推送方向 `z_proj`，拼接多个末端目标（位置+固定朝向），并调用 `solve_motion(...)` 得到离散关节轨迹。

2. **机器人控制（逐点关节命令 + 记录命令时间戳）**
   - `pushing.py:control_thread(...)` 使用 `frankapy` 的动态关节控制接口；
   - 通过 ROS topic 发布 `JointPositionSensorMessage` 给机器人侧的控制/trajectory generator。

3. **RealSense 采集 + ROS 状态读取 + 数据打包**
   - 相机线程循环读取 depth/color 帧（`rs.align` 将 depth 对齐到 color）。
   - 通过 ROS 订阅的 `Ros_listener` 读取最新的 `joint_state` 与 `ee_pose`，把它们与每一帧的时间戳一起写入 `frame.json`。
   - depth 以 `*.npz` 保存（单位换算到米），color 以 `*.png` 保存。
   - 每条轨迹目录保存 `cam_K.txt`、`init.json`、`control.json`（如控制线程启用）。

输出目录结构（Franka）：

- `records/.../traj_xxxxx/`
  - `depth/00000.npz ...`
  - `rgb/00000.png ...`
  - `frame.json`
  - `cam_K.txt`
  - `init.json`
  - （可选）`control.json`

## 2. 迁移到 Realman 时替换/保持了什么

迁移后的脚本在这里：

- `twinaligner_traj_recorder/dynamic_alignment_realman/pushing_realman.py`

我尽量做到“输出数据格式一致”，同时把控制/状态读取从 ROS/Franka 替换为 Realman 官方接口：

### 2.1 保持不变的部分

- **cuRobo 规划逻辑**：仍然复用 `dynamic_alignment/constrained_solver.py`
- **推送/对齐方向计算方式**：仍然使用“末端 z 轴在 xy 平面投影”的方式得到 `z_proj`
- **数据集落盘格式**：仍然保存
  - `depth/*.npz`
  - `rgb/*.png`
  - `frame.json`（字段命名尽量复用）
  - `cam_K.txt`
  - `init.json`

### 2.2 替换的部分

1. **控制方式**
   - 原来：`frankapy` + ROS topic 发布逐点关节命令
   - 现在：使用 `realman_env.RealmanEnv(async_mode=True)`，在一个控制线程里逐点调用 `env.send_joint(...)`

2. **状态读取**
   - 原来：ROS subscriber 读取 `/franka/joint_states` 与 `/franka/end_effector_pose`
   - 现在：直接调用 `env.get_state()` 获取缓存的
     - `joint`（弧度）
     - `pose`（Realman SDK 的 xyzrpy）

3. **坐标/朝向转换**
   - cuRobo 的 `Pose.quaternion` 顺序是 **`[w, x, y, z]`**。
   - Realman 提供的是 `xyzrpy`，脚本里用 `realman_env.T_from_realman_xyzrpy` + 旋转矩阵转四元数生成 wxyz 四元数。

## 3. 单位与四元数顺序是最容易踩坑的点

### 3.1 关节单位：cuRobo vs Realman SDK

- cuRobo 规划使用的关节角通常是 **弧度（rad）**（脚本会把 `env.get_joint()` 作为规划输入）。
- Realman SDK 的 `rm_movej_follow` 注释/实现里使用 **度（deg）**。

因此在控制线程里做了单位转换：

- `cmd_deg = np.degrees(cmd_rad)`

如果你将来修改了规划/控制端单位，请同时检查这里。

### 3.2 四元数顺序：cuRobo 需要 wxyz

- cuRobo：`Pose` 里的 `quaternion` 是 `[w, x, y, z]`。
- SciPy 转四元数输出是 `[x, y, z, w]`。

所以脚本里重排为 wxyz 再传给 `solve_motion(...)`。

## 4. Realman 迁移还缺哪些“Franka 才有的文件/配置”

Franka 版本在规划器初始化时使用了：

- `twinaligner_traj_recorder/dynamic_alignment/franka.yml`

该 yaml 内包含 URDF、ee_link/base_link、joint_names、碰撞配置等。迁移到 Realman 时，你必须提供对应的 cuRobo robot yaml（脚本默认路径见下）。

### 4.1 必须提供：`dynamic_alignment_realman/realman.yml`

脚本默认需要：

- `twinaligner_traj_recorder/dynamic_alignment_realman/realman.yml`

内容至少需要包含（字段名/层级遵循 cuRobo 的 `RobotConfig.from_dict(...)` 规则）：

1. `robot_cfg.kinematics.urdf_path`（Realman 的 URDF 文件路径）
2. `robot_cfg.kinematics.base_link`（基座 link 名）
3. `robot_cfg.kinematics.ee_link`（末端 link 名）
4. `robot_cfg.kinematics.cspace.joint_names`（cuRobo 的关节顺序）
5. `robot_cfg.kinematics.cspace.retract_config`（规划的零姿/回退姿态）

并建议补齐（否则会影响碰撞/自碰撞/约束质量）：

- `collision_link_names`
- `collision_spheres`（对应的碰撞球模型 yaml）
- self_collision ignore/buffer、lock_joints

### 4.2 URDF/碰撞模型如何准备（建议）

如果你目前没有 Realman 的 URDF：

- 需要从 Realman 官方资料/导出工具获得 URDF（或至少获得可用于 FK/IK 的 kinematics 结构）。
- 并为 cuRobo 准备 collision spheres（可以先用粗略 collision 以保证“能规划”，再逐步提升精度）。

因为我无法从仓库现有代码中自动推导出 Realman 的 URDF 与 link 命名映射，所以只能在这里告诉你“必须缺这些文件”以及“脚本会如何用到它们”。

## 5. 使用方法（运行采集）

1. 激活你的 conda/环境（必须包含 cuRobo 依赖、torch、pyrealsense2、opencv、scipy 等）
2. 确保存在 cuRobo 的 Realman yaml：
   - `twinaligner_traj_recorder/dynamic_alignment_realman/realman.yml`
3. 启动采集：

```bash
bash twinaligner_traj_recorder/pipelines/dynamic_alignment_realman.sh
```

如果需要自定义参数（比如 `robot_ip`、`record_frames`、`cmd_rate`），建议直接运行：

```bash
python3 twinaligner_traj_recorder/dynamic_alignment_realman/pushing_realman.py \
  --robot twinaligner_traj_recorder/dynamic_alignment_realman/realman.yml \
  --robot_ip 192.168.101.19 \
  --len 50 \
  --save_dir records/realman-dynamic
```

## 6. 迁移对齐策略的差异总结（你后续调参时最该看什么）

如果你发现 Realman 的规划轨迹与实际期望的推送方向不一致，通常优先检查：

1. `apply_tcp2eef` 是否需要开启  
   - 采集脚本支持 `--apply_tcp2eef`（默认开启）
2. Realman SDK 的 `pose` 所代表坐标系与 cuRobo 的 `ee_link` 是否一致  
3. `realman.yml` 里的 `ee_link/base_link/joint_names` 与真实机械臂关节顺序是否对应
4. 关节单位转换是否仍正确（rad -> deg）

