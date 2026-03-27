# RealmanEnv 使用说明

本文档说明如何使用 `realman_env.py` 中的 `RealmanEnv` 环境类进行机械臂控制，包含同步模式与异步模式两种方式。

## 1. 文件与能力概览

核心文件：`/home/zhangzhao/lyt/realman_env.py`

`RealmanEnv` 提供了以下能力：

- 连接 RealMan 机械臂
- 读取机械臂状态（位姿、关节、夹爪）
- 发送末端位姿命令（`xyzrpy`）
- 控制夹爪开合
- 同步控制接口（阻塞）
- 异步遥操作接口（非阻塞，后台线程）

---

## 2. 基础概念

### 2.1 位姿格式

脚本中位姿统一使用 `xyzrpy`（长度 6 的数组）：

- `x, y, z`：位置
- `rx, ry, rz`：欧拉角

### 2.2 控制模式

构造时通过 `control_mode` 指定：

- `absolute`：绝对位姿控制（传入的 `pose` 是目标位姿）
- `relative`：相对增量控制（传入的 `pose` 是相对当前位姿的增量）

### 2.3 同步/异步模式

- `async_mode=False`（默认）：同步调用，接口通常会阻塞直到完成
- `async_mode=True`：异步调用，状态读取与命令发送通过后台线程完成，适合高频遥操作

---

## 3. 快速开始

## 3.1 基本初始化

```python
from realman_env import RealmanEnv

env = RealmanEnv(
    robot_ip="192.168.101.19",
    safety_mode=False,
    async_mode=False,        # 先用同步模式调通
    min_cmd_interval=0.02,   # 异步模式下生效
    control_mode="absolute", # 或 "relative"
)
```

## 3.2 资源释放

务必在结束时调用 `close()`：

```python
try:
    # 你的控制逻辑
    pass
finally:
    env.close()
```

---

## 4. 同步模式用法（`async_mode=False`）

同步模式适合：

- 低频控制
- 脚本式动作执行
- 容易调试的线性流程

### 4.1 重置机器人

```python
obs = env.reset(target_gripper=0.09)
print(obs["pose"], obs["gripper_open"], obs["joint"])
```

### 4.2 获取当前观测

```python
obs = env.compute_observation()
pose = obs["pose"]            # np.ndarray(6,)
gripper = obs["gripper_open"] # float
joint = obs["joint"]          # np.ndarray(7,)
```

### 4.3 执行动作

`step(action)` 支持只传位姿、只传夹爪，或二者同时传：

```python
import numpy as np

# 只控制位姿
obs = env.step({
    "pose": np.array([0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
})

# 只控制夹爪
obs = env.step({
    "gripper_open": 0.05
})

# 同时控制位姿和夹爪
obs = env.step({
    "pose": np.array([0.32, 0.02, 0.38, 0.0, 0.0, 0.0]),
    "gripper_open": 0.03
})
```

---

## 5. 异步模式用法（`async_mode=True`）

异步模式适合：

- 高频遥操作
- 控制线程不希望被 SDK 调用阻塞
- “持续发命令 + 随时取最新状态”的场景

### 5.1 初始化

```python
env = RealmanEnv(
    robot_ip="192.168.101.19",
    async_mode=True,
    min_cmd_interval=0.02,
    control_mode="absolute",
)
```

### 5.2 非阻塞发命令

```python
import numpy as np

env.send_pose(np.array([0.30, 0.00, 0.40, 0.00, 0.00, 0.00]))
env.send_gripper(0.04)
```

说明：

- `send_pose()` / `send_gripper()` 只写入待发送缓存，立即返回
- 后台命令线程按 `min_cmd_interval` 频率发送
- 若高频连续调用，采用“最新命令覆盖旧命令”策略，避免队列积压

### 5.3 非阻塞取状态

```python
state = env.get_state()
if state is not None:
    print("pose:", state["pose"])
    print("gripper_open:", state["gripper_open"])
    print("joint:", state["joint"])
    print("timestamp:", state["timestamp"])
```

只取位姿：

```python
pose = env.get_pose()
if pose is not None:
    print(pose)
```

### 5.4 通信统计

```python
stats = env.get_communication_stats()
print(stats)
# 可能包含:
# {
#   "connected": True,
#   "async_mode": True,
#   "state_update_rate": ...,
#   "avg_latency_ms": ...
# }
```

---

## 6. 常用参数建议

### 6.1 `robot_ip`

- 设置为机械臂控制器 IP（默认 `192.168.101.19`）
- 连接失败时优先检查网络与 IP 是否正确

### 6.2 `safety_mode`

- `True`：限制线速度/角速度与加速度，适合调试阶段
- `False`：不额外限制（按控制器默认参数）

### 6.3 `min_cmd_interval`

- 仅异步模式生效
- 值越小，命令发送频率越高（但通信压力更大）
- 推荐从 `0.02`（约 50Hz）开始

### 6.4 `control_mode`

- `absolute`：适合轨迹点控制
- `relative`：适合手柄/遥操作增量控制

---

## 7. 推荐使用模板

```python
import numpy as np
from realman_env import RealmanEnv

def main():
    env = RealmanEnv(
        robot_ip="192.168.101.19",
        safety_mode=True,
        async_mode=True,
        min_cmd_interval=0.02,
        control_mode="relative",
    )
    try:
        # 异步循环示例
        for _ in range(200):
            env.send_pose(np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]))  # x 正方向微小增量
            env.send_gripper(0.06)
            state = env.get_state()
            if state is not None:
                print(state["pose"])
    finally:
        env.close()

if __name__ == "__main__":
    main()
```

---

## 8. 常见问题排查

### 8.1 连接失败

- 检查机械臂 IP 与本机网段
- 确认控制器端口可达（脚本中使用 `8080`）

### 8.2 状态偶发读取失败

- 属于通信场景常见情况，脚本已包含重试与默认值回退
- 异步模式下可观察 `get_communication_stats()` 的更新率和延迟

### 8.3 程序退出后连接未释放

- 检查是否遗漏 `env.close()`
- 建议统一使用 `try/finally`

---

## 9. 最小可运行示例（同步）

```python
import numpy as np
from realman_env import RealmanEnv

env = RealmanEnv(async_mode=False, control_mode="absolute")
try:
    obs = env.reset(target_gripper=0.09)
    print("reset pose:", obs["pose"])
    obs = env.step({"pose": np.array([0.30, 0.00, 0.35, 0.00, 0.00, 0.00])})
    print("new pose:", obs["pose"])
finally:
    env.close()
```

