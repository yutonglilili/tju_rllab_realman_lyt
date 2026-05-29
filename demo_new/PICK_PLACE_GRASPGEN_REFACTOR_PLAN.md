# 基于第三视角粗规划与腕部相机 GraspGen 精抓取的 PnP 改造方案

## 1. 目标

本次改造的目标是把当前 `demo_new/skills/pnp_skill/pick_and_place.py` 的抓取阶段拆成两段：

1. 第三视角相机负责持续打点、移动检测、粗定位。
2. 规划线程先根据第三视角给出的世界坐标规划一个安全可达的预抓取位姿。
3. 执行线程先把机械臂移动到这个预抓取位姿。
4. 到达预抓取位姿后，切换到腕部相机单次观测分支。
5. 腕部相机分支执行一次模型打点，只打一次点，不做持续追踪。
6. 根据腕部相机的 RGBD 和该点，做目标物体局部分割、桌面移除、点云清洗。
7. 将处理后的目标物体点云送入 GraspGen。
8. 对 GraspGen 输出的抓取姿态做过滤，保留前 5 个高置信度候选，不足 5 个就保留现有数量。
9. 将这些候选抓取姿态写入共享状态中的抓取姿态池。
10. 规划线程同时给出一个 `post pick` 姿态。
11. 执行线程依次尝试抓取姿态池中的候选：
    - 优先使用排名最靠前的候选。
    - 若 IK 解不出或执行失败，则切换到下一个候选。
    - 如果所有候选都失败，则退回当前的基于 `xyz` 位置选择抓取朝向的原始方案作为兜底。
12. 放置阶段不接入 GraspGen，继续沿用当前基于 `xyz` 大小和偏置的放置策略。

这套设计的核心思想是：

- 第三视角负责“找得到目标、跟得住目标、给出安全预抓取位置”。
- 腕部相机负责“靠近目标后做局部精抓取姿态推理”。
- GraspGen 只参与抓取阶段，不参与放置阶段。


## 2. 现有脚本架构

当前核心脚本是：

- `demo_new/skills/pnp_skill/pick_and_place.py`

它目前是一个典型的三线程结构：

### 2.1 SharedState

当前 `SharedState` 主要维护以下几类信息：

- 当前任务：`current_task`, `task_phase`
- 感知输出：`latest_point_2d`, `latest_point_3d`, `latest_target_T`
- 移动检测：`previous_point_3d`, `point_changed`, `tracking_mode`, `verify_mode`
- 规划输出：`action_list`, `action_index`, `plan_ready`, `need_replan`
- 执行控制：`abort_execution`
- 任务结束信号：`task_done`, `task_success`

当前状态机比较粗，抓取和放置都只区分为：

- `TaskPhase.PICK`
- `TaskPhase.PLACE`

但没有进一步区分“第三视角粗规划阶段”和“腕部相机精抓取阶段”。

### 2.2 感知线程

当前 `perception_thread(...)` 有两种模式：

- `tracking_mode`
  - 持续从第三视角相机取图。
  - 调用 `get_point_vllm(...)` 打点。
  - 用 `make_target_T(...)` 把 2D 点转换成 base/world 坐标下的目标位姿。
  - 做目标移动检测，触发 `need_replan`。
- `verify_mode`
  - 在 pick 结束后或 place 结束后做成功检测。

当前感知线程只绑定了一个相机环境 `rs_env`，没有区分：

- 第三视角相机
- 腕部相机

也没有单次腕部观测的分支。

### 2.3 规划线程

当前 `planning_thread(...)` 的逻辑是：

1. 等待 `need_replan`
2. 读取 `latest_target_T`
3. 调用 `build_action_list(...)`
4. 直接生成完整的动作序列：
   - `pre_pick`
   - `pick`
   - `post_pick`
   - 或 `pre_place/place/post_place`
5. 将结果写入 `action_list`

这意味着当前规划是“一次性把整个抓取段都规划完”，中间无法插入：

- 先移动到预抓取位姿
- 再用腕部相机做局部建图和 GraspGen 推理
- 再重新决定最终抓取姿态

### 2.4 执行线程

当前 `execution_thread(...)` 做的是顺序执行：

1. 等待 `plan_ready`
2. 依次执行 `action_list`
3. 如果某步失败则计数，超过阈值后重新规划
4. 执行结束后切到 `verify_mode`

当前执行线程没有“候选抓取姿态池”的概念，也没有：

- 依次尝试多个 grasp pose
- IK 失败时自动切换到下一个候选
- 失败后回退到兜底姿态

### 2.5 主线程

当前主线程负责：

- 初始化 `RealmanEnv`
- 初始化第三视角相机 `Open3dRealsenseEnv`
- 启动三线程
- 根据任务列表或自然语言指令驱动整套系统

当前初始化默认只有一个相机，且抓取流程默认完全依赖第三视角打点。


## 3. 现有抓取策略的局限

当前脚本的抓取阶段主要问题是：

1. 抓取姿态完全由第三视角目标点和手工规则 `adjust_target_T(...)` 决定。
2. 机械臂靠近目标后，没有使用腕部相机做局部精细感知。
3. 没有在抓取前构建“目标物体局部点云”。
4. 没有使用 GraspGen 的 6D 抓取候选能力。
5. 没有抓取候选池，也没有候选失败后的自动切换机制。

因此当前流程更像是：

- “用第三视角点一个位置，再用固定姿态去抓”

而不是：

- “用第三视角靠近目标，再用腕部相机在近距离做精抓取姿态决策”


## 4. 目标中的新抓取流程

改造后的 pick 阶段应该拆成如下步骤：

### 4.1 第三视角感知阶段

- 第三视角相机仍然负责持续打点与移动判断。
- 输出：
  - `latest_point_2d`
  - `latest_point_3d`
  - `latest_target_T`
- 这些信息继续作为粗规划输入。

### 4.2 粗规划阶段

- 规划线程基于第三视角的 `latest_target_T` 或 `latest_point_3d`：
  - 生成一个安全可达的 `pregrasp_pose_base`
  - 这个位姿的目标是“让机械臂靠近目标并让腕部相机能看清局部”
- 此阶段不直接生成最终 `pick_pose`
- 也不立即生成完整的 `pre_pick/pick/post_pick` 三段动作

### 4.3 预抓取执行阶段

- 执行线程先执行 `pregrasp_pose_base`
- 机械臂到位后，切换到腕部相机单次观测分支

### 4.4 腕部相机单次感知阶段

- 腕部相机只做一次观测，不做持续追踪
- 对腕部相机图像调用一次模型打点
- 这次打点不用于全局跟踪，只用于局部分割种子

### 4.5 腕部 RGBD 到目标物体点云

这里建议复用两部分现有代码思想：

- `GraspGen/scripts/demo_wrist_camera_graspgen.py`
  - 已经具备：
    - 腕部 RGBD 采集
    - 打点种子扩展
    - 点云转 base 系
    - 去桌面
    - GraspGen 推理与过滤
- `GraspGen/scripts/demo_collision_free_grasps.py`
  - 可复用其“物体点云 / 场景点云 / 碰撞过滤”的处理思路

腕部相机分支的处理链建议是：

1. 获取腕部相机 RGBD
2. 获取一次点击点
3. 以点击点为中心，在局部范围内扩展多个有效 seed 点
4. 基于深度、颜色、邻域连通性生成目标 mask
5. 把 mask 内像素转成目标物体点云
6. 把非 mask 区域转成 scene 点云
7. 使用手眼标定和当前 TCP 位姿把点云变换到 base 系
8. 根据 base 系桌面高度删掉桌面点
9. 对目标点云做 downsample / outlier removal
10. 把清洗后的 `object_pc` 送入 GraspGen

### 4.6 GraspGen 推理与候选池生成

GraspGen 输出的是若干个 4x4 抓取姿态矩阵和置信度。

在改造后的系统里：

- 规划线程或腕部相机分支需要对这些候选做规则过滤
- 再取保留结果中置信度前 5 的姿态
- 将结果写入共享状态中的抓取姿态池

建议池中元素使用 base 系姿态：

- `grasp_pose_pool_base = [T1, T2, T3, ...]`
- `grasp_pose_scores = [s1, s2, s3, ...]`

如果过滤后不足 5 个，则按实际数量保留。

### 4.7 后抓取姿态

规划线程还需要生成一个 `post_pick_pose_base`。

这里可以先保持简单：

- 使用一个固定的上提姿态
- 或者沿成功抓取姿态的局部安全方向上提

第一版可以不复杂化，先使用统一的普通 `post pick` 姿态即可。

### 4.8 候选执行与失败切换

执行线程在抓取候选阶段的逻辑应改为：

1. 从 `grasp_pose_pool_base` 取第 1 个候选
2. 尝试做 IK / 可达性检查
3. 若 IK 可解，则执行抓取
4. 若 IK 不可解或执行失败，则切换到下一个候选
5. 按顺序尝试前 5 个候选
6. 如果全部失败，则退回到当前原始的 `adjust_target_T(...)` 规则抓取策略

这一步是此次改造中最关键的执行层改动。


## 5. 放置阶段保持不变

本次改造只针对 pick 阶段。

place 阶段继续保持现有逻辑：

- 第三视角打点
- 基于 `xyz` 和偏置规则选择放置姿态
- 不调用 GraspGen
- 不使用腕部相机精抓取分支

这样可以把改造范围控制在抓取阶段，降低系统复杂度。


## 6. 推荐的新状态机设计

当前只有 `TaskPhase.PICK` 和 `TaskPhase.PLACE` 两级状态，不够表达新的抓取流程。

建议新增一层 pick 子状态，例如：

- `PICK_GLOBAL_TRACKING`
- `PICK_PREGRASP_PLANNING`
- `PICK_PREGRASP_EXECUTING`
- `PICK_WRIST_SENSING`
- `PICK_GRASPGEN_READY`
- `PICK_GRASP_EXECUTING`
- `PICK_POST_EXECUTING`

place 阶段可以继续简单保留：

- `PLACE_TRACKING`
- `PLACE_EXECUTING`
- `PLACE_VERIFY`

如果不想大改 `TaskPhase`，也可以新增一个字段，例如：

- `state.pick_stage`

把更细粒度的抓取阶段写进去。


## 7. SharedState 需要新增的内容

建议在 `SharedState` 中新增以下字段：

### 7.1 相机与阶段控制

- `perception_mode`
  - `third_view_tracking`
  - `wrist_single_shot`
  - `verify`
- `pick_stage`
- `wrist_request_id`
- `wrist_observation_ready`

### 7.2 第三视角粗规划结果

- `pregrasp_pose_base`
- `pregrasp_plan_ready`

### 7.3 腕部相机单次感知结果

- `wrist_rgb`
- `wrist_depth`
- `wrist_click_2d`
- `wrist_mask`
- `wrist_object_pc_base`
- `wrist_scene_pc_base`

### 7.4 GraspGen 输出

- `grasp_pose_pool_base`
- `grasp_pose_pool_scores`
- `grasp_pose_pool_ready`
- `grasp_pose_pool_source`
- `grasp_pose_pool_filtered_count`

### 7.5 抓取执行与兜底

- `selected_grasp_pose_base`
- `selected_grasp_score`
- `post_pick_pose_base`
- `fallback_pick_pose_base`


## 8. 需要修改的模块与函数

### 8.1 `pick_and_place.py`

这是主改造文件，需要改动的核心函数有：

- `SharedState`
- `perception_thread(...)`
- `planning_thread(...)`
- `build_action_list(...)`
- `execution_thread(...)`
- `init_camera_env(...)`
- `start_pnp_system(...)`
- `run_single_task(...)`

### 8.2 感知线程需要拆成双相机逻辑

当前 `perception_thread(...)` 默认只有一个 `rs_env`。

改造后建议：

- 第三视角相机：
  - 持续打点
  - 持续移动检测
  - 为粗规划提供目标世界坐标
- 腕部相机：
  - 只在到达预抓取位姿后触发一次
  - 执行一次单点打点和局部点云构建

这意味着初始化接口也要变化，至少要支持：

- `third_view_rs_env`
- `wrist_rs_env`

### 8.3 规划线程需要拆成两段

当前规划线程是“拿到目标就完整生成 pre/pick/post”。

改造后建议拆成两段：

#### 第一段：粗规划

- 输入：第三视角目标 `latest_target_T`
- 输出：`pregrasp_pose_base`
- 下发给执行线程

#### 第二段：精抓取规划

- 触发条件：执行线程已到达 `pregrasp_pose_base`
- 输入：腕部相机生成的 `grasp_pose_pool_base`
- 输出：
  - 候选抓取姿态池
  - `post_pick_pose_base`
  - 兜底抓取姿态

### 8.4 执行线程需要支持候选姿态池

当前执行线程只会顺序执行 `action_list`。

改造后建议支持：

1. 执行预抓取位姿
2. 等待腕部相机和 GraspGen 结果
3. 从抓取姿态池按顺序取候选
4. 对每个候选先做 IK / reachability check
5. 可行则执行
6. 不可行则换下一个
7. 全失败则执行兜底抓取姿态

### 8.5 当前 `adjust_target_T(...)` 保留为兜底策略

这个函数目前是按目标位置手工选择抓取姿态。

改造后它仍然有价值：

- 用作所有 GraspGen 候选失败时的兜底抓取姿态生成器
- 也继续作为 place 阶段的默认姿态选择器


## 9. 建议新增的桥接模块

不建议直接在 `pick_and_place.py` 里硬塞大量 GraspGen demo 逻辑。

更推荐新增一个桥接模块，例如：

- `demo_new/skills/pnp_skill/graspgen_bridge.py`

职责如下：

### 9.1 腕部相机相关

- 获取腕部 RGBD
- 获取一次点击点
- 点击点扩种子
- 构造局部目标 mask

### 9.2 点云相关

- RGBD 转 organized point cloud
- 相机系 -> base 系
- 去桌面
- 目标点云和场景点云分离
- 点云去噪 / 下采样

### 9.3 GraspGen 调用

- 在主控环境中通过轻量 client 调用 GraspGen 服务
- 获取候选抓取姿态
- 候选过滤
- 输出前 5 个抓取姿态

### 9.4 对外接口建议

建议桥接模块提供一个高层接口，例如：

```python
infer_pick_grasp_candidates_from_wrist(
    wrist_obs,
    click_point_2d,
    robot_pose_tcp,
    handeye_calib,
) -> dict
```

返回值可包含：

- `object_pc_base`
- `scene_pc_base`
- `grasp_pose_pool_base`
- `grasp_pose_pool_scores`
- `fallback_pick_pose_base`
- `debug_info`

这里推荐的实现方式不是在主控里直接 import `GraspGenSampler`，而是：

1. `realman_env_lyt` 环境中的桥接模块先完成：
   - 腕部 RGBD 获取
   - 打点
   - mask
   - 去桌面
   - `object_pc_base` / `scene_pc_base` 构建
2. 然后桥接模块通过本机 client 把 `object_pc_base` 发送给 `GraspGen` 环境中的推理服务
3. 收到 `grasps + scores` 后，再在主控侧继续做：
   - top 5 候选选取
   - IK 可达性筛选
   - 候选失败切换
   - 兜底姿态

这样主控环境不需要直接加载 `torch`、GraspGen checkpoint 或 CUDA 模型。


## 10. GraspGen 侧建议的复用方式

当前不建议把整个 `demo_wrist_camera_graspgen.py` 直接作为库来 import。

建议复用其中已经验证过的处理思想和函数：

- 点击点周围扩多个 seed 点
- 用 RGBD 做目标 mask
- base 系桌面过滤
- 目标点云清洗
- GraspGen 推理
- 抓取姿态规则过滤

建议参考两个脚本：

- `GraspGen/scripts/demo_wrist_camera_graspgen.py`
- `GraspGen/scripts/demo_collision_free_grasps.py`

其中：

- 前者更贴近你的真实腕部相机场景
- 后者更适合借用“物体点云 / 场景点云 / 碰撞过滤”的组织方式


## 11. 环境与进程建议

考虑到当前已经存在：

- `GraspGen` 在 `GraspGen` 环境中运行
- 主控在 `realman_env_lyt` 环境中运行

建议本次改造继续保持分环境，不强行合并依赖。

### 11.1 推荐的职责边界

推荐把两边的职责明确拆开：

#### `realman_env_lyt` 主控环境负责

- 第三视角相机感知
- 腕部相机感知
- 2D 打点
- RGBD 转点云
- 手眼标定变换
- 桌面移除
- 构建 `object_pc_base`
- 构建 `scene_pc_base`
- 调用 GraspGen client
- 接收抓取候选
- IK / reachability 检查
- 执行候选切换
- 兜底抓取策略

#### `GraspGen` 推理环境负责

- 常驻加载 GraspGen 模型
- 接收点云请求
- 运行 `GraspGenSampler.run_inference(...)`
- 返回 `grasps + confidences`

这样分工后，主控环境只负责机器人和感知逻辑，GraspGen 环境只负责模型推理。

### 11.2 推荐的连接方式

推荐使用本机 `ZMQ client/server`，不要每次抓取时临时拉起一个新的 `conda run` 进程。

推荐拓扑如下：

1. 在 `GraspGen` 环境中启动一个常驻 server
2. 在 `realman_env_lyt` 环境中启动主控脚本
3. 主控脚本内部持有一个常驻 client
4. 每次需要 GraspGen 时，只把处理好的点云发给 server
5. server 返回抓取候选

推荐原因：

- 避免每次抓取时重复加载 checkpoint
- 避免反复启动 Python 解释器
- 避免把 `torch/open3d/numpy/scipy` 全部揉进主控环境
- 通信开销很小，因为传的是目标物体点云，不是整张 RGBD

### 11.3 已有可复用的 server/client

仓库中已经有现成实现：

- 服务端：
  - `GraspGen/client-server/graspgen_server.py`
  - `GraspGen/grasp_gen/serving/zmq_server.py`
- 客户端：
  - `GraspGen/grasp_gen/serving/zmq_client.py`

其中服务端负责加载 `GraspGenSampler` 并常驻，客户端负责发送点云并接收抓取结果。

### 11.4 推荐的启动方式

推荐的运行方式是两个终端、两个环境：

#### 终端 A：启动 GraspGen 推理服务

```bash
conda activate GraspGen
cd /home/lyt/tju_rllab_realman_lyt/GraspGen
python client-server/graspgen_server.py \
  --gripper_config /home/lyt/tju_rllab_realman_lyt/GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml \
  --host 127.0.0.1 \
  --port 5556
```

说明：

- `host` 建议先绑定到 `127.0.0.1`，只允许本机访问。
- `port` 建议固定为一个明确端口，例如 `5556`。
- server 启动后模型会常驻显存，不要在每次抓取时重新启动。

#### 终端 B：启动主控脚本

```bash
conda activate realman_env_lyt
cd /home/lyt/tju_rllab_realman_lyt
python demo_new/skills/pnp_skill/pick_and_place.py
```

### 11.5 主控环境中的 client 放在哪里

虽然仓库里已经有 `GraspGen/grasp_gen/serving/zmq_client.py`，但主控环境不一定直接依赖 `grasp_gen` 包。

因此推荐两种做法：

#### 做法 A：在主控侧新增一个轻量 wrapper

新增例如：

- `demo_new/skills/pnp_skill/graspgen_client_bridge.py`

这个文件只依赖：

- `numpy`
- `pyzmq`
- `msgpack`
- `msgpack_numpy`

它内部可以直接复用或轻量拷贝 `GraspGenClient` 的实现方式。

#### 做法 B：让主控通过 `sys.path` 访问仓库中的 client

如果 `realman_env_lyt` 已经安装了：

- `pyzmq`
- `msgpack`
- `msgpack_numpy`

那么也可以直接 import：

```python
from grasp_gen.serving.zmq_client import GraspGenClient
```

但从工程隔离角度看，还是更推荐做法 A，把 client wrapper 固定在 `demo_new` 下。

### 11.6 两个环境之间到底传什么

推荐只传“已经处理好的目标物体点云”，不要直接传 RGBD。

也就是说：

#### 主控侧输入给 server 的内容

- `object_pc_base`
  - 类型：`np.ndarray`
  - 形状：`(N, 3)`
  - `dtype=float32`

第一版建议主控只发送：

- `object_pc_base`
- `num_grasps`
- `grasp_threshold`
- `topk_num_grasps`

示例 payload：

```python
{
    "action": "infer",
    "point_cloud": object_pc_base.astype(np.float32),
    "grasp_threshold": -1.0,
    "num_grasps": 200,
    "topk_num_grasps": 50,
    "min_grasps": 40,
    "max_tries": 6,
    "remove_outliers": True,
}
```

#### server 返回给主控的内容

- `grasps`
  - 形状：`(M, 4, 4)`
- `confidences`
  - 形状：`(M,)`

注意：

- 返回姿态的坐标系与输入点云坐标系一致。
- 如果发送的是 `object_pc_base`，那返回的就是 `base` 系抓取姿态。

### 11.7 第一版建议的职责分配

为了尽量少改现有 GraspGen server，第一版建议按下面方式分配职责：

#### `GraspGen` 环境只负责

- `object_pc_base -> raw grasps + scores`

#### 主控环境继续负责

- 腕部 RGBD 到 `object_pc_base`
- base 系桌面去除
- 抓取方向规则过滤
- top 5 候选选取
- IK / 可达性筛选
- 候选切换
- 兜底姿态

这样做的优点是：

- 可以直接复用现有 server/client
- 服务端协议改动最小
- 主控更容易结合机器人状态做可达性判断

### 11.8 如果后续要扩展 server

如果后面希望把更多过滤也移到 `GraspGen` 环境，可以扩展协议，在 request 中增加：

- `scene_point_cloud`
- `filter_options`
- `request_id`

例如：

```python
{
    "action": "infer",
    "point_cloud": object_pc_base,
    "scene_point_cloud": scene_pc_base,
    "filter_options": {
        "direction_rule": True,
        "collision_check": True,
        "topk_keep": 5,
    },
}
```

但这属于第二阶段优化，不建议作为第一版落地点。

### 11.9 主控里什么时候调用 GraspGen client

推荐调用时机如下：

1. 执行线程先把机械臂移动到 `pregrasp_pose_base`
2. 感知线程切到腕部相机单次观测模式
3. 主控侧完成：
   - 打点
   - mask
   - 去桌面
   - `object_pc_base` 构建
4. 规划线程或桥接模块调用 GraspGen client
5. 收到 `grasps + scores`
6. 主控侧更新：
   - `grasp_pose_pool_base`
   - `grasp_pose_pool_scores`
7. 执行线程开始按候选顺序尝试执行

### 11.10 失败与兜底策略

两个环境连接后，必须明确失败时的回退策略。

推荐如下：

#### server 不可用

- 主控在系统启动时先做一次 `health_check`
- 若 server 未启动：
  - 可以阻塞等待一小段时间并重试
  - 或直接降级到原始启发式抓取方案

#### 推理超时

- client 应设置超时
- 超时后本轮放弃 GraspGen，直接用兜底姿态

#### 推理返回空结果

- 若 `len(grasps) == 0`
- 则直接回退到当前 `adjust_target_T(...)` 方案

#### 候选姿态都不可达

- 若前 5 个候选 IK 全失败
- 则执行当前原始规则姿态

### 11.11 明确不推荐的方式

不推荐下面这种接法：

1. 主控每次需要抓取时
2. 临时执行一次：

```bash
conda run -n GraspGen python some_graspgen_script.py
```

3. 等脚本跑完后再把结果读回来

原因：

- 每次都会重复启动 Python
- 每次都会重复加载 torch 和 checkpoint
- 延迟大且不稳定
- 很难和主控线程状态机协同

这样可以避免：

- `torch/open3d/numpy/scipy` 依赖冲突
- 每次抓取都重新拉起 GraspGen 解释器和模型


## 12. 推荐的实施顺序

建议按以下顺序逐步实现：

### 第一步：状态机与双相机改造

- 在 `SharedState` 中加入新的抓取阶段字段
- 初始化第三视角相机和腕部相机
- 让感知线程支持两种相机模式

### 第二步：预抓取分段执行

- 将当前一次性 `pre/pick/post` 的规划拆开
- 先打通：
  - 第三视角打点
  - 规划预抓取
  - 执行预抓取

### 第三步：腕部单次感知与物体点云构建

- 从腕部相机获取单次图像
- 打一次点
- 构建目标 mask
- 去桌面
- 生成 `object_pc_base`

### 第四步：接入 GraspGen

- 将 `object_pc_base` 送入 GraspGen
- 获取抓取候选
- 输出前 5 个候选到共享状态

### 第五步：执行线程候选切换

- 依次尝试姿态池
- IK 失败时换下一个
- 全失败后用兜底抓取姿态

### 第六步：补充 debug 与可视化

- 保存腕部观测
- 保存 mask
- 保存候选抓取姿态池
- 保存最终选择的抓取姿态


## 13. 本次改造完成后的系统形态

改造完成后，整个抓取阶段将变成：

1. 第三视角持续感知目标
2. 规划线程生成安全预抓取位姿
3. 执行线程移动到预抓取位姿
4. 腕部相机单次观测
5. 局部物体点云构建
6. GraspGen 输出候选抓取姿态
7. 执行线程逐个尝试候选抓取姿态
8. 所有候选失败时回退到原始启发式抓取姿态
9. 成功抓取后执行统一的 `post pick`

而放置阶段仍然保持当前轻量策略，不引入 GraspGen。


## 14. 结论

本次改造的本质不是简单“把 GraspGen 塞进现有 pick_and_place.py”，而是要把当前的单段式抓取流程重构成：

- 第三视角粗定位
- 预抓取到位
- 腕部相机单次精感知
- GraspGen 候选抓取
- 候选失败切换
- 原始策略兜底

因此这次改造的关键点有三个：

1. 状态机要更细，必须显式区分预抓取前后两个感知阶段。
2. 规划线程不能再一次性生成完整抓取动作序列，必须拆段。
3. 执行线程必须从“顺序执行固定 action_list”升级为“从抓取候选池里按可达性逐个尝试”。

这份文档对应的是抓取阶段的目标实现方案，放置阶段默认保持现状。
