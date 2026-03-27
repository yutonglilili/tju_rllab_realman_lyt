# Vision-Guided Pick & Place System Logic

## 1. 系统目标

本系统实现一个 **基于视觉的 Pick & Place 自动循环任务**。  
机器人通过视觉识别目标进行抓取（Pick）与放置（Place），并在每个动作后进行成功检测。

系统必须具备 **闭环视觉恢复能力**，确保在任何动作失败时能够通过重新获取图像与重新识别来恢复任务。

---

# 2. 核心设计原则

系统必须遵循以下三个核心原则：

## 2.1 所有动作必须基于最新图像

无论执行 **Pick** 还是 **Place**，都必须：

1. 重新获取图像
2. 重新识别目标
3. 再执行动作

即：
Capture Image
→ Detect Target
→ Execute Motion


禁止使用旧的视觉识别结果执行动作。

---

## 2.2 每个动作后必须进行成功检测

系统在以下两个阶段必须进行检测：

- **Post-Pick Detection**
- **Post-Place Detection**

只有检测成功时，流程才允许进入下一阶段。

如果检测失败：
重新获取图像
→ 重新识别
→ 重新执行动作

并持续重试直到成功。

---

## 2.3 Place失败必须重新执行Pick

如果 **Place检测失败**：

不能直接重新Place，也不能沿用之前的识别结果。

必须重新执行：
获取图像
→ 识别Pick
→ Pick
→ Pick检测
→ 获取图像
→ 识别Place
→ Place
→ Place检测


---

# 3. 标准任务流程

在没有错误的情况下，系统流程如下：
Capture Image
↓
Detect Pick Target
↓
Execute Pick
↓
Post-Pick Detection
↓
Capture Image
↓
Detect Place Target
↓
Execute Place
↓
Post-Place Detection
↓
Task Completed


---

# 4. Pick阶段逻辑

## 4.1 Pick执行流程

执行Pick前必须重新进行视觉识别：
Capture Image
↓
Detect Pick Target
↓
Execute Pick
↓
Post-Pick Detection


---

## 4.2 Pick失败恢复逻辑

如果 **Post-Pick Detection 失败**：

系统必须重新执行Pick阶段：
Pick
↓
Pick Detection Failed
↓
Capture Image
↓
Detect Pick Target
↓
Execute Pick
↓
Post-Pick Detection


循环执行直到：
Pick Detection Success


---

# 5. Place阶段逻辑

## 5.1 Place执行流程

Place执行前必须重新获取图像并识别：
Capture Image
↓
Detect Place Target
↓
Execute Place
↓
Post-Place Detection


---

## 5.2 Place失败恢复逻辑

如果 **Post-Place Detection 失败**：

系统必须重新开始Pick阶段，而不是直接重复Place。

恢复流程如下：
Execute Place
↓
Post-Place Detection Failed
↓
Capture Image
↓
Detect Pick Target
↓
Execute Pick
↓
Post-Pick Detection


如果 **Pick失败**：
Capture Image
→ Detect Pick Target
→ Execute Pick
→ Post-Pick Detection


直到Pick成功。

---

## 5.3 Pick成功后的继续流程

当Pick检测成功后：
Capture Image
↓
Detect Place Target
↓
Execute Place
↓
Post-Place Detection


---

# 6. 完整恢复流程

当Place失败时完整流程如下：
Place
↓
Post-Place Detection Failed
↓
Capture Image
↓
Detect Pick Target
↓
Execute Pick
↓
Post-Pick Detection

├── Pick Failed
│       ↓
│   Capture Image
│   Detect Pick Target
│   Execute Pick
│   Post-Pick Detection
│
└── Pick Success
        ↓
    Capture Image
        ↓
    Detect Place Target
        ↓
    Execute Place
        ↓
    Post-Place Detection


---

# 7. 动作执行统一规则

为避免错误执行逻辑，系统必须遵守以下规则：

## 7.1 Pick动作规则

执行Pick之前必须执行：
Capture Image
→ Detect Pick Target
→ Execute Pick


---

## 7.2 Place动作规则

执行Place之前必须执行：
Capture Image
→ Detect Place Target
→ Execute Place


---

# 8. 系统循环逻辑（伪代码）

系统逻辑可以抽象为如下伪代码：

```python
while True:

    # Pick Stage
    while True:
        capture_image()
        detect_pick_target()
        execute_pick()

        if check_pick_success():
            break

    # Place Stage
    while True:
        capture_image()
        detect_place_target()
        execute_place()

        if check_place_success():
            break
        else:
            break   # return to Pick stage


---

# 9. 系统特性

该系统具有以下特点：

视觉闭环控制

每个动作前重新定位

每个动作后进行成功验证

自动恢复失败任务

避免使用过期视觉数据

这种策略常用于工业自动化中的 Robust Vision-Guided Picking Systems。

---

# 10. 总结

系统核心逻辑可以概括为：

所有动作必须基于最新视觉数据

Pick成功后才能执行Place

Place失败必须重新执行Pick

每个阶段必须检测成功

任何失败都通过重新视觉识别恢复
