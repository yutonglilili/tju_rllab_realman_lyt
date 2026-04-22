# Realman 机械臂 ACT 训练与部署全流程

这个目录是一套专门针对你当前 `Realman + RealSense + zarr 数采脚本` 整理出来的 ACT 工作流。

它的目标不是泛化成一个大而全的框架，而是把你现在手上的数据链路真正接起来：

1. 用你已经能跑通的数采脚本采集 `zarr`
2. 把 `zarr` 转成 LeRobot v3 数据集
3. 用 LeRobot 的 ACT 实现训练策略模型
4. 把训练好的模型整理成部署 bundle
5. 在 Realman 真机上做在线推理

这套脚本已经按你当前项目里的数据格式做了适配，不是直接照搬 `xarm` 那套双相机版本。

---

## 目录说明

当前 `act_train` 目录下每个文件的作用如下：

- `common.py`
  公共工具模块。负责：
  - 自动把 `realman` 工程根目录、`iffyuan-XArm-Toolkit-main`、`lerobot/src` 加进 Python 路径
  - 解析 zarr 数据集里的 episode 边界
  - 做 `rgb` 图像格式转换
  - 做夹爪宽度和夹爪动作的归一化
  - 生成训练和推理都会用到的 state / action 向量
  - 运行时预览窗口渲染

- `convert_realman_zarr_to_lerobot.py`
  数据格式转换脚本。
  把你当前 Realman 数采脚本产生的 `zarr` 数据集转换成 LeRobot v3 格式，供 ACT 训练使用。

- `verify_realman_lerobot.py`
  数据校验脚本。
  用来检查转换后的 LeRobot 数据集是否结构正确、维度正确、是否存在明显异常值。

- `train_realman_act.py`
  ACT 训练脚本。
  负责读取本地 LeRobot 数据集，构建 ACT 模型，训练并保存 checkpoint。

- `prepare_realman_deploy.py`
  部署打包脚本。
  把训练好的模型目录复制成一个独立部署 bundle，同时写入推理阶段要用的运行参数。

- `run_realman_act.py`
  在线推理脚本。
  真机运行入口。连接 Realman 机械臂和 RealSense 相机，读取部署 bundle，做在线策略推理。

---

## 这套流程适配的原始数据格式

这套脚本默认适配你当前 `collect_zarr.py` / `collect_realman.py` 这一路采集出来的 zarr 数据格式。

当前预期的 zarr 数据字段如下：

- `data/rgb`
  形状：`(N, 3, H, W)`
  含义：RGB 图像，单相机

- `data/depth`
  形状：`(N, 1, H, W)`
  含义：深度图
  说明：当前 ACT 训练脚本默认不使用它

- `data/pose`
  形状：`(N, 6)`
  含义：末端位姿，通常是 `[x, y, z, roll, pitch, yaw]`

- `data/joint`
  形状：`(N, 7)`
  含义：7 维关节角

- `data/action`
  形状：`(N, 6)`
  含义：末端增量动作，通常是 `[dx, dy, dz, droll, dpitch, dyaw]`

- `data/gripper_width`
  形状：`(N, 1)`
  含义：真实夹爪开口宽度，单位通常是米

- `data/gripper_action`
  形状：`(N, 1)`
  含义：夹爪目标控制命令，单位通常也是米

- `data/gripper_state`
  形状：`(N, 1)`
  含义：归一化夹爪状态，通常在 `[0, 1]` 区间

- `meta/episode_ends`
  形状：`(E,)`
  含义：每个 episode 的结束帧索引

如果 `meta/episode_ends` 不存在，脚本也可以退化使用 `data/episode` 自动重建 episode 边界。

---

## 训练时的数据映射关系

为了让数据语义更合理，这里没有把所有字段直接照抄进训练集，而是做了明确映射。

### 默认图像映射

- `observation.image <- data/rgb`

当前版本按单相机训练设计，所以只使用一个 RGB 观察输入。

### 默认状态映射

默认使用：

- `observation.state <- data/pose + normalized(data/gripper_width)`

也就是说状态向量默认是：

`[x, y, z, roll, pitch, yaw, gripper]`

这里的 `gripper` 不是目标控制量，而是基于 `gripper_width` 归一化之后得到的观测量。

这样做更合理，因为：

- `gripper_width` 更接近真实状态
- `gripper_action` 更接近控制命令

如果你想用关节角而不是末端位姿做状态输入，也可以切换为：

- `observation.state <- data/joint + normalized(data/gripper_width)`

对应参数：

```powershell
--state-source joint
```

### 默认动作映射

- `action <- data/action + normalized(data/gripper_action)`

动作向量默认是：

`[dx, dy, dz, droll, dpitch, dyaw, gripper]`

这样训练时：

- 前 6 维学末端增量控制
- 最后 1 维学夹爪控制

---

## 环境依赖

这套脚本依赖以下环境已经准备好：

### 1. 你的主工程可导入

也就是下面这些模块能被 Python 看到：

- `realman_env.py`
- `open3d_realsense_env.py`
- 你的工程根目录

### 2. LeRobot 已安装

建议按你参考文档里的方式安装本地源码版本：

```powershell
cd C:\Users\admi\Desktop\realman\iffyuan-XArm-Toolkit-main\iffyuan-XArm-Toolkit-main\lerobot
pip install -e .
```

### 3. 其它常见依赖

至少需要这些包：

- `torch`
- `numpy`
- `zarr`
- `Pillow`
- `opencv-python`
- `tqdm`
- `pytransform3d`

如果你用 WandB 记录训练，还需要：

- `wandb`

---

## 一、zarr 转 LeRobot 数据集

### 脚本

- `convert_realman_zarr_to_lerobot.py`

### 作用

把你当前采集出来的 Realman zarr 数据集转换为 LeRobot v3 数据集。

这是 ACT 训练前必须做的一步，因为训练脚本直接读的是 LeRobotDataset。

### 常用命令

```powershell
python .\act_train\convert_realman_zarr_to_lerobot.py `
  --input .\collect_data_by_tele_op\datasets\demo.zarr `
  --output .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo `
  --task "pick and place" `
  --fps 15 `
  --state-source pose `
  --force
```

### 参数说明

- `--input`
  输入 zarr 数据集路径

- `--output`
  输出 LeRobot 数据集目录

- `--repo-id`
  LeRobot 本地数据集的 repo id
  这个名字后续训练和校验时都要保持一致

- `--task`
  当前任务描述
  会写入数据集元信息中

- `--fps`
  数据集帧率
  这个值会影响 ACT 的时间窗口解释，建议与你采集/部署的控制频率一致或接近

- `--state-source pose`
  状态输入用 `pose + gripper`

- `--state-source joint`
  状态输入用 `joint + gripper`

- `--episodes`
  只转换前 N 个 episode，适合先做小规模测试

- `--force`
  如果输出目录已存在，先删除再重建

### 转换结果

转换完成后，输出目录里会有：

- LeRobot 标准数据结构
- `realman_conversion.json`

这个 `realman_conversion.json` 记录了这次转换使用的关键参数，后面排查问题时很有用。

---

## 二、校验转换后的 LeRobot 数据集

### 脚本

- `verify_realman_lerobot.py`

### 作用

在正式训练前检查数据集：

- 基本结构是否正确
- 特征名是否齐全
- `observation.state` 和 `action` 的维度是否正常
- 是否存在 NaN / Inf / 全零帧
- 各 episode 长度分布是否正常

### 常用命令

```powershell
python .\act_train\verify_realman_lerobot.py `
  --path .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo
```

### 快速抽样校验

如果数据量很大，想先快速看前一部分数据：

```powershell
python .\act_train\verify_realman_lerobot.py `
  --path .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo `
  --max-frames 2000
```

### 建议

正式训练前，至少先跑一遍完整校验。

如果这里已经发现：

- 大量 NaN
- action 几乎全零
- episode 长度非常不均匀
- state 数值范围明显异常

优先回到数采环节修，不要直接开训。

---

## 三、训练 ACT 模型

### 脚本

- `train_realman_act.py`

### 作用

使用 LeRobot 的 ACT 实现训练你的 Realman 模型。

训练脚本会：

1. 读取 LeRobot 数据集
2. 自动根据数据集元信息构建 input/output features
3. 创建 ACTConfig
4. 构建 preprocessor / postprocessor
5. 训练模型
6. 定期保存 checkpoint
7. 最终保存完整模型目录

### 常用命令

```powershell
python .\act_train\train_realman_act.py `
  --dataset .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo `
  --output .\act_train\outputs\act_realman_demo `
  --batch-size 64 `
  --steps 20000 `
  --chunk-size 64 `
  --n-action-steps 64
```

### 重要参数说明

- `--dataset`
  LeRobot 数据集目录

- `--repo-id`
  必须和转换阶段写入的 repo id 对应

- `--output`
  模型输出目录

- `--device`
  训练设备，例如：
  - `cuda`
  - `cpu`
  - `mps`

- `--batch-size`
  batch 大小
  如果显存不够，先降到 `16` 或 `8`

- `--steps`
  总训练步数

- `--chunk-size`
  ACT 一次预测多少步动作

- `--n-action-steps`
  每次推理实际执行多少步动作
  一般不要超过 `chunk-size`

- `--vision-backbone`
  图像 backbone，默认 `resnet18`

- `--dim-model`
  Transformer 主维度

- `--kl-weight`
  VAE KL 损失权重

- `--temporal-ensemble-coeff`
  如果你后续部署时想使用 temporal ensemble，可以在这里写入配置

### WandB 训练记录

如果你想开 WandB：

```powershell
python .\act_train\train_realman_act.py `
  --dataset .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo `
  --output .\act_train\outputs\act_realman_demo `
  --batch-size 64 `
  --steps 20000 `
  --chunk-size 64 `
  --n-action-steps 64 `
  --wandb `
  --wandb-project realman-act
```

### 训练输出内容

训练完成后，输出目录里通常会有：

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `postprocessor_config.json`
- `train_run.json`
- `checkpoint_xxx/`

其中：

- `checkpoint_xxx/` 是中途保存点
- 输出目录根部是最终模型

---

## 四、打包部署 bundle

### 脚本

- `prepare_realman_deploy.py`

### 作用

把训练结果整理成一个可部署目录。

这个脚本会：

1. 复制训练好的模型目录到 bundle 内部
2. 写一个 `deploy_config.json`
3. 把运行推理时需要的默认参数一起保存进去

### 常用命令

```powershell
python .\act_train\prepare_realman_deploy.py `
  --checkpoint .\act_train\outputs\act_realman_demo `
  --output .\act_train\deploy\act_realman_demo `
  --task "pick and place" `
  --robot-ip 192.168.101.19 `
  --camera-serial f1471338 `
  --state-source pose `
  --force
```

### 参数说明

- `--checkpoint`
  训练完成后的模型目录

- `--output`
  部署 bundle 输出目录

- `--task`
  推理时给策略的任务字符串

- `--robot-ip`
  默认机械臂 IP

- `--camera-serial`
  默认 RealSense 序列号

- `--state-source`
  必须和训练数据构造时一致
  如果训练时用了 `pose`，部署也必须是 `pose`
  如果训练时用了 `joint`，部署也必须是 `joint`

- `--control-fps`
  推理时控制频率

- `--max-delta-translation-mm`
  每步最大平移增量安全限制

- `--max-delta-rotation-rad`
  每步最大旋转增量安全限制

### 输出结果

部署目录里会有：

- `model/`
- `deploy_config.json`

你后续在线推理只需要指定这个 bundle 目录即可。

---

## 五、Realman 真机在线推理

### 脚本

- `run_realman_act.py`

### 作用

连接：

- Realman 机械臂
- RealSense 相机
- 训练好的 ACT 模型

然后在真机上实时执行策略。

### 常用命令

```powershell
python .\act_train\run_realman_act.py `
  --bundle .\act_train\deploy\act_realman_demo `
  --device cuda `
  --max-steps 500
```

### 干跑模式

如果你想先只看视觉和推理，不实际发机械臂控制命令：

```powershell
python .\act_train\run_realman_act.py `
  --bundle .\act_train\deploy\act_realman_demo `
  --device cuda `
  --dry-run `
  --auto-start
```

### 运行时交互按键

预览窗口里支持以下按键：

- `Space`
  开始 / 暂停策略控制

- `Enter`
  重置当前 rollout
  同时清空 ACT 内部动作队列

- `H`
  把内部目标位姿重新同步到当前机器人位姿
  当你怀疑目标位姿累计漂移时，这个键很有用

- `Q`
  退出程序

### 推理脚本内部做了什么

每一轮循环大致逻辑是：

1. 读取 Realman 当前状态
2. 读取 RealSense 当前 RGB 图像
3. 把图像 resize 成训练时的输入尺寸
4. 构造 `observation.image` 和 `observation.state`
5. 走 preprocessor
6. 调用 `model.select_action`
7. 走 postprocessor
8. 对动作做安全裁剪
9. 把增量动作叠加到内部目标位姿
10. 把目标位姿和夹爪命令发给机器人

### 安全机制

推理阶段做了几层相对保守的安全限制：

- 每步平移增量限制
- 每步旋转增量限制
- 夹爪命令限制在 `[0, 1]`
- 默认不是自动开始，而是进入暂停状态

这意味着你可以先观察画面，确认没问题后再按空格启动。

---

## 推荐的完整使用顺序

建议每次按下面顺序走：

### 第一次联调

1. 先采少量 zarr 数据
2. 先只转换前几个 episode
3. 跑 `verify_realman_lerobot.py`
4. 先用很小的训练步数试跑
5. 用 `--dry-run` 检查在线推理链路
6. 最后再上真机动作

### 正式训练

1. 收集更多稳定的 demonstrations
2. 完整转换全部 episode
3. 完整校验数据
4. 正式训练 ACT
5. 打包部署
6. 真机逐步验证

---

## 常见问题与建议

### 1. 为什么现在只用单相机？

因为你当前 Realman 数采脚本写入的是单个 `data/rgb`，而不是 xarm 文档里那种固定相机 + 腕部相机双视角结构。

这套脚本是按你当前真实数据结构做的最直接适配。

如果后续你增加第二个相机，我可以再帮你把这套转换和推理脚本扩展成双相机版本。

### 2. 为什么状态默认用 `gripper_width` 而不是 `gripper_action`？

因为：

- `gripper_width` 更像观测
- `gripper_action` 更像控制命令

训练 imitation learning 时，把命令混进状态里通常不如把真实观测放进状态里合理。

### 3. `pose` 状态和 `joint` 状态该选哪个？

一般建议：

- 如果你的动作本身是末端增量控制，优先用 `pose`
- 如果你更关心关节空间一致性，或者后续控制链更贴近 joint space，可以试 `joint`

你当前采集动作是末端增量 `action(6)`，所以我默认选了 `pose`

### 4. `fps` 应该怎么设？

它不是一个随便写的数字。

它会影响：

- 数据时间尺度
- ACT chunk 对应的时间窗口长度
- 推理执行节奏

简单理解：

`时间窗口 = chunk_size / fps`

例如：

- `chunk_size=64`
- `fps=15`

那么模型大致在预测未来 `4.27 秒` 左右的动作片段。

### 5. 现在有没有导出 ONNX / TensorRT？

没有。

当前 `prepare_realman_deploy.py` 做的是 Python 运行时 bundle，而不是跨框架导出。

这是故意的，因为你当前先要把训练和真机闭环打通，Python 原生运行更稳，也更容易调试。

如果后续你要做更轻量部署，我可以下一步再帮你加：

- ONNX 导出
- TensorRT 推理
- 服务器/客户端式部署

### 6. 为什么推理脚本里要维护一个 `target_transform`？

因为你的动作定义是：

- 每步预测一个末端增量动作

所以推理时不能把预测结果当成绝对位姿，而应该：

1. 从当前目标位姿出发
2. 累计叠加增量动作
3. 再把累计后的目标发给机器人

这和你采集阶段的控制方式是一致的。

---

## 最小可用命令链

如果你现在只想先快速跑通最小流程，可以直接照这个顺序：

### 1. 转换

```powershell
python .\act_train\convert_realman_zarr_to_lerobot.py `
  --input .\collect_data_by_tele_op\datasets\demo.zarr `
  --output .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo `
  --task "pick and place" `
  --fps 15 `
  --state-source pose `
  --force
```

### 2. 校验

```powershell
python .\act_train\verify_realman_lerobot.py `
  --path .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo
```

### 3. 训练

```powershell
python .\act_train\train_realman_act.py `
  --dataset .\act_train\datasets\demo_lerobot `
  --repo-id realman_demo `
  --output .\act_train\outputs\act_realman_demo `
  --batch-size 16 `
  --steps 2000 `
  --chunk-size 32 `
  --n-action-steps 32
```

### 4. 打包

```powershell
python .\act_train\prepare_realman_deploy.py `
  --checkpoint .\act_train\outputs\act_realman_demo `
  --output .\act_train\deploy\act_realman_demo `
  --task "pick and place" `
  --robot-ip 192.168.101.19 `
  --camera-serial f1471338 `
  --state-source pose `
  --force
```

### 5. 干跑推理

```powershell
python .\act_train\run_realman_act.py `
  --bundle .\act_train\deploy\act_realman_demo `
  --device cuda `
  --dry-run `
  --auto-start
```

---

## 后续你可能会继续做的扩展

这套脚本已经够你当前单相机 ACT 工作流使用。

后续如果你要继续往前推进，我建议可以按这个方向扩展：

- 增加双相机输入
- 把 depth 也接进训练数据
- 加数据集可视化脚本
- 增加评估脚本
- 支持模型推理服务化
- 支持更稳的启动/复位逻辑

如果你希望，我下一步可以继续帮你做两件很实用的事：

1. 把 README 里的示例命令全部换成你当前真实的数据路径和任务名
2. 再给你补一套中文的 `.bat` 一键启动脚本，直接双击就能跑转换、训练和推理
