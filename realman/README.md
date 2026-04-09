# Realman Env 说明文档

这份文档面向当前项目里的机械臂环境封装，重点说明以下两个文件：

- `realman/realman_env.py`：当前建议使用的版本
- `realman/realman_env_old.py`：旧版本，保留用于对照和回溯

如果你现在主要在看 `demo/pnp_final/pick_and_place.py`，那它实际接的是 `realman_env.py`，不是 old 版本。

## 1. 这个 env 在项目里的定位

`RealmanEnv` 本质上是对睿尔曼 SDK `Robotic_Arm.rm_robot_interface` 的二次封装。它做了几件事：

- 把底层 SDK 的机械臂控制整理成统一接口
- 把 `joint` 控制、`pose` 控制、夹爪控制放到同一个环境对象里
- 在同步模式下提供类似 Gym 的 `reset()` / `step()` 用法
- 在异步模式下提供流式发送目标的接口，方便高频控制或轨迹跟随
- 统一处理 TCP 与机械臂末端执行器 EEF 的位姿转换
- 统一处理夹爪开口宽度与 RealMan 寄存器值之间的转换

从代码分层上看，两个版本的整体结构是一致的：

1. 工具函数层：坐标变换、夹爪宽度换算
2. `RobotState`：状态快照数据结构
3. `RealmanDriver`：只负责 SDK 调用
4. `SyncController`：阻塞式控制
5. `AsyncController`：流式非阻塞控制
6. `RealmanEnv`：最终对外统一接口

## 2. 相关文件关系

- `realman/realman_env.py`
  当前主实现。`pick_and_place.py` 使用的是它。
- `realman/realman_env_old.py`
  旧实现。接口骨架相同，但同步控制能力比新版本弱一些。
- `realman/open3d_realsense_env.py`
  相机环境，不控制机械臂，只负责 RealSense 图像和深度采集。
- `demo/pnp_final/pick_and_place.py`
  当前项目里最有代表性的接入示例。它使用 `RealmanEnv(mode="sync")` 执行 pick-and-place 动作序列。

## 3. 依赖与运行前提

机械臂 env 本身依赖这些库：

- `numpy`
- `pytransform3d`
- `Robotic_Arm`

相机环境另外依赖：

- `open3d`

一个常见安装方式是：

```bash
pip install numpy pytransform3d Robotic_Arm open3d
```

如果你不想从 pip 安装 `Robotic_Arm`，项目里也带了 SDK 目录：

- `realman/RM_API2/Python/Robotic_Arm`

这时要确保 Python 能找到它。常见做法有两种：

```powershell
$env:PYTHONPATH="C:\Users\admi\Desktop\siu\realman\RM_API2\Python"
```

或者从项目根目录启动，并手动把对应目录加入 `sys.path`。

另外需要注意：

- 机械臂控制器 IP 必须可连通
- SDK 动态库和 Python 包要匹配当前系统环境
- 项目里的 `realman` 目录没有显式 `__init__.py`，通常依赖项目根目录在 `sys.path` 中来完成导入

## 4. 坐标、单位和语义

这是使用时最容易踩坑的一部分。

### 4.1 pose 的格式

代码里的 pose 统一采用：

```python
[x, y, z, rx, ry, rz]
```

含义如下：

- `x, y, z`：单位是米
- `rx, ry, rz`：单位是弧度

### 4.2 joint 的单位

这里新手最容易混：

- 发给 `movej()` / `env.step({"joint": ...})` / `env.send_joint(...)` 的关节目标是角度制
- `get_state()` 返回的 `RobotState.joint` 是弧度制

也就是说：

- 输入 joint：`deg`
- 输出 joint state：`rad`

`pick_and_place.py` 里就专门做了这个兼容处理：如果动作里的关节值看起来像弧度，就先转成角度再发给 env。

### 4.3 TCP 和 EEF 的区别

代码里同时存在两套位姿语义：

- TCP：夹爪中心点
- EEF：RealMan SDK 直接控制的末端执行器坐标

`realman_env.py` 里定义了一个固定变换：

- `T_TCP2REALMANEEF`

它用来在“上层任务更关心的 TCP 位姿”和“底层 SDK 真正接收的 EEF 位姿”之间转换。

相关辅助函数：

- `T_from_realman_xyzrpy(xyzrpy)`：`xyzrpy -> 4x4` 变换矩阵
- `realman_xyzrpy_from_T(T)`：`4x4 -> xyzrpy`
- `pose_eef2tcp(pose_eef)`：EEF 位姿转 TCP 位姿
- `pose_tcp2eef(pose_tcp)`：TCP 位姿转 EEF 位姿

### 4.4 夹爪的单位

夹爪在上层接口里使用“开口宽度，单位米”。

相关函数：

- `realman_gripper_value_from_width(width)`
- `width_from_realman_gripper_value(gripper_value)`

例如：

- `0.09` 通常表示接近全开
- `0.03` 通常表示闭合抓取

## 5. `RobotState` 的含义

两个版本都定义了相同的数据结构：

```python
@dataclass
class RobotState:
    pose: np.ndarray
    joint: np.ndarray
    gripper: float
    timestamp: float
```

字段解释：

- `pose`：当前位姿
- `joint`：当前关节角
- `gripper`：夹爪宽度，单位米
- `timestamp`：采样时间戳

但这里有一个非常重要的实现细节：

- 在 `sync` 模式下，`get_state()` 返回的 `pose` 是 TCP 位姿
- 在当前 `async` 实现里，缓存状态直接写入的是 driver 层原始 pose，也就是 EEF 位姿

这意味着当前代码里“同步模式的 pose 语义”和“异步模式的 pose 语义”并不完全统一。这个现象在 `realman_env.py` 和 `realman_env_old.py` 里都存在，不是新版本单独引入的变化。

## 6. Driver 层做了什么

`RealmanDriver` 是真正跟 SDK 打交道的一层。它主要负责：

- 建立机械臂连接
- 配置 Modbus
- 配置夹爪速度
- 设置关节/直线/角速度上限
- 执行 `movej`、`movep`、`movel`、follow 类动作
- 读取当前机械臂状态
- 控制夹爪寄存器
- 关闭硬件连接

Driver 层的设计原则是：

- 尽量薄
- 尽量只做 SDK 封装
- 不在这一层混入任务逻辑

要注意，Driver 层对 pose 的理解是偏底层的：

- 它更接近 SDK 直接使用的 EEF 位姿

而上层同步 env 封装做了一次 TCP/EEF 转换，所以调用者平时不必直接跟 EEF 打交道。

## 7. 同步模式 `mode="sync"`

### 7.1 适用场景

同步模式更适合：

- 单步调试
- RL 风格的 `reset -> step -> observe`
- 任务型执行
- 希望“发一个动作，等它做完，再读状态”的流程

### 7.2 初始化

```python
from realman.realman_env import RealmanEnv

env = RealmanEnv(robot_ip="192.168.101.19", mode="sync")
```

### 7.3 reset

```python
state = env.reset()
```

当前实现里的 `reset()` 会做这些事：

- 把机械臂移动到固定 home 关节位姿 `[90, 0, 0, -90, 0, -90, 60]`
- 等待关节误差收敛，或者 5 秒超时
- 把夹爪打开到 `0.09`
- 返回 reset 后的 `RobotState`

### 7.4 step(action)

同步模式的核心接口是：

```python
state = env.step(action)
```

`action` 在当前新版本里支持这些字段：

- `joint`：目标关节角，角度制
- `pose`：目标 TCP 位姿，`[x, y, z, rx, ry, rz]`
- `gripper`：目标夹爪开口宽度，单位米
- `motion`：仅对 pose 动作有效，可选 `"pose"` 或 `"linear"`
- `wait_gripper`：夹爪动作是否等待到位，默认 `True`

常见示例 1，关节控制：

```python
import numpy as np

state = env.step({
    "joint": np.array([90, 0, 0, -90, 0, -90, 60])
})
```

常见示例 2，位姿控制：

```python
import numpy as np

target_tcp_pose = np.array([0.45, -0.15, 0.20, 3.14, 0.0, 1.57])

state = env.step({
    "pose": target_tcp_pose,
    "motion": "pose",
    "gripper": 0.09,
    "wait_gripper": False,
})
```

常见示例 3，直线接近：

```python
state = env.step({
    "pose": target_tcp_pose,
    "motion": "linear",
    "gripper": 0.03,
    "wait_gripper": True,
})
```

同步模式行为特征：

- 阻塞式
- 每个 `step()` 都是“动作执行完成后再返回”
- `get_state()` 是实时读 SDK，不是缓存
- 比较适合低频、确定性的任务执行

## 8. 异步模式 `mode="async"`

### 8.1 适用场景

异步模式更适合：

- 高频发送目标
- 轨迹跟随
- policy rollout
- 流式控制

### 8.2 初始化

```python
env = RealmanEnv(robot_ip="192.168.101.19", mode="async")
```

### 8.3 可用接口

异步模式不能使用：

- `step()`
- `reset()`

异步模式使用的是这些接口：

```python
env.send_joint(joint_deg)
env.send_pose(pose)
env.send_gripper(width)
state = env.get_state()
env.close()
```

### 8.4 内部机制

`AsyncController` 背后启动了两个线程：

- 状态线程：周期性读取机械臂状态并写入缓存
- 指令线程：从 pending 命令里取“最新目标”发给 SDK

它的特点是：

- 非阻塞
- 只保留最新命令，旧命令可能被覆盖
- `get_state()` 返回的是缓存状态，不是严格实时值
- 默认最小发送间隔是 20ms 左右

### 8.5 当前实现中的 pose 语义注意事项

这是最需要额外说明的一点。

按当前代码实现：

- `send_pose()` 会把 pose 直接交给 `driver.movep_follow()`
- 这一层没有做 `pose_tcp2eef()` 转换
- `AsyncController._state_loop()` 里缓存的 pose 也是直接来自 driver 原始状态

所以在当前实现中：

- `sync` 模式的 pose 以 TCP 为主
- `async` 模式的 pose 更接近 EEF 语义

如果你希望异步模式也统一用 TCP 位姿，需要自己在上层做转换，或者后续把 async 分支补齐和 sync 一样的转换逻辑。

## 9. `pick_and_place.py` 是怎么接这个 env 的

当前 `demo/pnp_final/pick_and_place.py` 使用方式非常有代表性：

- 用 `RealmanEnv(robot_ip=..., mode="sync")` 初始化机械臂环境
- 用 `Open3dRealsenseEnv(...)` 初始化相机环境
- 感知线程负责找目标点位
- 规划线程生成动作序列
- 执行线程逐条把动作喂给 `env.step()`

该 demo 里的动作列表大致长这样：

```python
[
    {
        "pose": pre_target_pose,
        "gripper": pre_gripper_state,
        "tag": 0,
        "motion": "pose",
        "wait_gripper": False,
        "speed_percent": 80,
    },
    {
        "pose": target_pose,
        "gripper": target_gripper_state,
        "tag": 1,
        "motion": "linear",
        "wait_gripper": True,
        "speed_percent": 100,
    },
    {
        "pose": post_target_pose,
        "tag": 2,
        "motion": "linear",
        "speed_percent": 100,
    },
]
```

这里有两个实践层面的结论：

- `realman_env.py` 的新接口已经支持 `motion` 和 `wait_gripper`，所以能直接承接这类动作序列
- `tag` 和 `speed_percent` 这些字段是 demo 自己的流程字段，当前 `realman_env.py` 并不会直接消费它们

也就是说，`pick_and_place.py` 里：

- `tag` 用来区分动作阶段
- `speed_percent` 目前只是被上层动作结构带着走，但 env 自身没有按 action 动态调速

## 10. `realman_env.py` 和 `realman_env_old.py` 的主要区别

这是这次最核心的部分。

### 10.1 总结版

一句话总结：

- `realman_env.py` 是在 old 版本基础上的增强版，重点强化了同步 pose 控制的表达能力和使用灵活性
- 两者的整体架构没有变，主要变化集中在 Driver 参数、同步 `step()` 的动作字段、pose 运动方式和夹爪处理细节上

### 10.2 详细对照

| 维度 | `realman_env_old.py` | `realman_env.py` |
| --- | --- | --- |
| 推荐程度 | 历史版本，主要用于对照 | 当前建议使用 |
| 整体架构 | Driver + SyncController + AsyncController + RealmanEnv | 相同 |
| 同步 pose 控制 | 只有 `movep`，本质上走 `rm_movej_p` | 新增 `movel`，可区分 `"pose"` 和 `"linear"` |
| `step(action)` 可选字段 | `joint` / `pose` / `gripper` | 在 old 基础上新增 `motion`、`wait_gripper` |
| 夹爪控制 | 每次都写寄存器 | 新增“目标宽度已接近则直接返回”的短路判断 |
| pose 重试逻辑 | `movep` 自己重复写一遍重试代码 | 抽出 `_move_pose_with_retry()`，`movep` 和 `movel` 共用 |
| 同步速度参数 | `JOINT_MAX_SPEED_DEG_S=90`，同步速度百分比约 80 | 调整为 `JOINT_MAX_SPEED_DEG_S=75`，同步速度百分比约 75，并单独区分 `MOVEP` 与 `MOVEL` |
| Driver 极限参数 | 线速度更保守：`0.1`，线加速度 `0.5`，角速度 `0.5` | 调整为：线速度 `0.4`，线加速度 `1.0`，角速度 `0.4` |
| `pick_and_place.py` 兼容性 | 不能完整表达 `motion="linear"`、`wait_gripper=False` 这种新动作 | 能直接承接当前 demo 的动作定义 |

### 10.3 新版本具体增强点

新版本相对 old，最实用的增强有四个：

- 支持 `motion="linear"`，也就是同步模式下可以显式做直线段接近/离开
- 支持 `wait_gripper=False`，可以让夹爪动作不阻塞整个 step
- `set_gripper()` 在目标宽度已接近时会直接返回，减少不必要的 Modbus 写入
- pose 运动的重试逻辑被整理得更清晰，后续继续扩展也更方便

### 10.4 为什么这些变化很重要

这些改动对 `pick-and-place` 任务尤其重要。

old 版本的问题不是“完全不能用”，而是：

- 很难在接口层明确表达“这一步我要走直线接近目标”
- 很难表达“这一步机械臂移动和夹爪动作是否要同步等待”
- 对接当前 demo 的动作结构时，语义会丢失

而新版本基本是在往“任务执行层更好用”的方向演进：

- 上层规划器可以明确指定直线 approach
- 执行线程可以更灵活地安排夹爪等待
- 和 `pick_and_place.py` 这类任务型代码更贴合

### 10.5 哪些地方其实没变

虽然新版本增强了同步控制，但下面这些核心设计是没变的：

- 仍然是同样的六层结构
- `reset()` 逻辑基本相同
- `AsyncController` 的总体机制基本相同
- `RealmanEnv` 对外仍然提供 sync / async 两种模式
- `RobotState` 数据结构不变

所以如果你已经基于 old 版本写过代码，迁移成本并不高。绝大多数情况下：

- 原来只用 `joint`、`pose`、`gripper` 的调用仍然能迁过来
- 如果你想利用新功能，再额外补 `motion` 和 `wait_gripper` 即可

## 11. 从 old 迁移到新 env 的建议

如果你手头还有基于 `realman_env_old.py` 的调用代码，建议按下面方式迁移：

1. 先把导入切到 `realman.realman_env`
2. 保持原有 `joint` / `pose` / `gripper` 调用不变，先确认基本行为一致
3. 对需要“直线接近”的步骤补上 `motion="linear"`
4. 对需要机械臂和夹爪并行一点的步骤补上 `wait_gripper=False`
5. 检查 joint 输入单位是不是角度制，避免把弧度直接送进 `step({"joint": ...})`
6. 如果你使用 async 模式，先明确自己到底是按 TCP 还是 EEF 在传 pose

## 12. 当前实现里的几个注意事项

这几条不是“新旧版本差异”，而是阅读代码后很值得提前知道的实现现状。

### 12.1 `speed_percent` 目前没有真正接入 action 接口

虽然 `pick_and_place.py` 的动作里带了：

- `speed_percent`

但当前 `realman_env.py` 的 `step()` 并不会根据 action 动态修改速度。也就是说：

- 这个字段现在更多是上层动作结构的一部分
- env 内部仍然使用文件顶部定义的固定速度参数

### 12.2 sync 和 async 的 pose 语义不完全统一

前面已经提过，再强调一次：

- sync 模式：上层以 TCP pose 为主
- async 模式：当前实现更接近直接传 EEF pose

如果后续你要做统一的控制接口，这里是优先建议整理的点。

### 12.3 `slow_stop()` / `emergency_stop()` 的同步分支需要再核对

代码里 `SyncController.slow_stop()` 和 `SyncController.emergency_stop()` 使用了：

- `self._op_lock`

但当前文件里没有看到这个锁的初始化语句。也就是说，这两个接口虽然已经写进类里，但在真正调用前建议先再核对一遍实现，避免运行时直接因为属性不存在而报错。

这个现象在 `realman_env.py` 和 `realman_env_old.py` 里都存在。

## 13. 推荐结论

如果你现在要继续开发当前项目，建议默认使用：

- `realman/realman_env.py`

理由很直接：

- 它和当前 `pick_and_place.py` 的动作设计更匹配
- 它已经支持同步直线运动语义
- 它对夹爪等待策略更灵活
- 它比 old 版本更接近“任务执行环境”而不只是“SDK 包装”

`realman_env_old.py` 更适合做这些事：

- 回看历史实现
- 对比行为变化
- 作为迁移参考

如果你后面准备继续完善这个 env，最值得优先补的三个方向是：

1. 把 async 分支的 pose 语义统一成和 sync 一样的 TCP 语义
2. 把 `speed_percent` 真正接入 action 接口
3. 把 stop 相关锁补完整并做一次实际联调
