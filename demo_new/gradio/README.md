# Gradio 重构方案

这版方案按你的意思收紧了，不再拆成很多层。

我的建议是 `gradio/` 里先只保留 3 个核心脚本：

```text
gradio/
├─ main.py
├─ task_interface.py
└─ ui.py
```

如果以后真的变复杂，再拆；现在先别为了“规范”把文件切太碎。


## 1. 设计原则

先把边界定死：

- `task/*/run.py` 只负责任务执行
- `gradio/*` 只负责界面、调度、状态展示
- Gradio 可以调用 `task`
- `task` 不反向持有 Gradio 逻辑

也就是说：

- 不在 `task/run.py` 里写页面元数据
- 不在 `task/run.py` 里写 Gradio controller
- 不在 `task/config.yaml` 里塞 UI 字段


## 2. 三个脚本分别干什么

### `gradio/main.py`

这是主控启动脚本。

它负责：

- 初始化机器人和相机运行时
- 创建 Gradio 页面
- 处理 Start / Stop / Preview / Timer 刷新
- 维护当前运行中的任务线程
- 从 `task_interface.py` 取任务定义
- 从 `ui.py` 取页面布局和样式

你可以把它理解成：

- “总入口”
- “总调度器”
- “总状态机”

这个文件会稍大一点，但我觉得这是合理的，因为它本来就是主控脚本。


### `gradio/task_interface.py`

这是 `gradio` 和 `task` 之间唯一的接口层。

它负责两类事情：

1. 定义“有哪些任务要出现在界面里”
2. 把界面输入转成 `task/run.py` 能执行的格式

它不负责页面样式，不负责按钮排版。

它里面建议放：

- 任务列表定义
- 每个任务的标题、说明、示例
- 每个任务的参数默认值
- 每个任务的 `preview()` 逻辑
- 每个任务的 `execute()` 逻辑

你可以把它理解成：

- “任务注册表”
- “任务适配层”

但这些都放在一个文件里，不再拆成 `registry.py + adapters/*.py`。


### `gradio/ui.py`

这是纯界面层。

它负责：

- 页面布局
- 组件定义
- CSS 样式
- 页面里静态的说明文案

它不负责：

- 机器人初始化
- 任务执行
- VLM 拆解
- 后台线程控制

你可以把它理解成：

- “长什么样”
- “有哪些控件”


## 3. 推荐目录结构

推荐最终长这样：

```text
demo_new/
├─ task/
│  ├─ pick_and_place/
│  │  ├─ run.py
│  │  └─ config.yaml
│  └─ roast_sweet_potatoes/
│     ├─ run.py
│     └─ config.yaml
│
├─ gradio/
│  ├─ README.md
│  ├─ main.py
│  ├─ task_interface.py
│  └─ ui.py
│
├─ skills/
├─ vlm_utils/
└─ pnp_gradio_demo.py
```

后面如果确定这套结构没问题，再让 `pnp_gradio_demo.py` 变成一句话入口：

```python
from gradio.main import main

main()
```


## 4. `task` 层应该长什么样

这部分很关键。

### `task/pick_and_place/run.py`

它只关心：

- PnP 任务怎么执行
- 需要什么 env
- 需要什么配置

以后最好能提供一个“给外部调用的执行函数”，例如：

```python
def run_from_instruction(
    *,
    env,
    rs_env,
    cam_results,
    home_T_tcp2base,
    instruction,
    mode="with_position_description",
    config_path=None,
    stop_event=None,
):
    ...
```

这里的重点不是一定叫这个名字，而是：

- 它是任务执行函数
- 它不是 Gradio 函数


### `task/roast_sweet_potatoes/run.py`

它也只关心执行。

我建议它最终提供的是这种接口：

```python
def run_from_task_list(
    *,
    env,
    rs_env,
    cam_results,
    home_T_tcp2base,
    task_list,
    rotate_angle=90,
    config_path=None,
    stop_event=None,
):
    ...
```

注意这里我故意建议它接 `task_list`，而不是直接接用户自然语言。

原因是：

- “一句话怎么拆解成红薯/玉米/空气炸锅抽屉”这件事，更像界面适配逻辑
- “空气炸锅怎么开、怎么放、怎么关、怎么拧旋钮”这件事，才是 task 执行逻辑

这样边界最干净。


## 5. `task_interface.py` 应该怎么设计

这是这次最重要的文件。

我的建议是：一个文件里放一个简单的任务表，每个任务就是一个字典，或者一个很轻的类。

例如：

```python
TASKS = {
    "pick_and_place": {
        "title": "Pick and Place",
        "description": "桌面抓取放置任务",
        "input_label": "PnP 指令",
        "examples": [
            "把棒球放到粉色盘子里",
            "把球放到魔方右边",
        ],
        "default_params": {
            "mode": "with_position_description",
        },
        "preview": preview_pick_and_place,
        "execute": execute_pick_and_place,
    },
    "roast_sweet_potatoes": {
        "title": "Air Fryer Roast",
        "description": "烤红薯/玉米等空气炸锅任务",
        "input_label": "烘烤指令",
        "examples": [
            "我需要烤红薯和玉米",
        ],
        "default_params": {
            "rotate_angle": 90,
        },
        "preview": preview_roast_task,
        "execute": execute_roast_task,
    },
}
```

这样足够简单，也足够清楚。


## 6. 烤红薯任务怎么接进去

你现在最关心的是这个，所以我单独讲清楚。

### 用户输入

用户在界面输入：

```text
我需要烤红薯和玉米
```


### `task_interface.py` 里做的事

`preview_roast_task()` 负责：

1. 调用 `vlm_utils` 的拆解函数
2. 得到用户可读的计划

例如：

```json
[
  {"pick": "红薯", "place": "空气炸锅抽屉"},
  {"pick": "玉米", "place": "空气炸锅抽屉"}
]
```

`execute_roast_task()` 负责：

1. 再把这个计划标准化成执行用格式

```json
[
  {"pick": "sweet potato", "place": "open air fryer drawer"},
  {"pick": "corn", "place": "open air fryer drawer"}
]
```

2. 调用：

```python
task.roast_sweet_potatoes.run.run_from_task_list(...)
```

也就是说：

- 拆解逻辑放在 `gradio/task_interface.py`
- 执行逻辑放在 `task/roast_sweet_potatoes/run.py`

这正符合你想要的边界。


## 7. `ui.py` 里放什么

`ui.py` 不要太聪明，就做两件事：

1. 定义页面布局
2. 定义样式

例如它可以提供：

```python
def build_ui(task_defs):
    ...
```

它返回：

- 下拉框
- 指令输入框
- 示例按钮
- 参数框
- Preview / Start / Stop
- Camera
- Current Task / Phase / Position / Logs

样式也放这里，比如：

- 更专业一点的卡片式布局
- 顶部任务简介区
- 状态面板
- 日志区

但不要在这里写任何 task 执行逻辑。


## 8. `main.py` 里怎么串起来

`main.py` 推荐做下面这条链路：

1. 初始化共享 runtime
2. 从 `task_interface.py` 取任务定义
3. 调用 `ui.py` 构建页面
4. 绑定按钮事件：
   - 切换任务
   - 填入示例
   - Preview
   - Start
   - Stop
5. 用 Timer 定时刷新：
   - 相机画面
   - 当前任务
   - 阶段
   - 目标位置
   - 日志

也就是说：

- `main.py` 负责“怎么跑起来”
- `ui.py` 负责“长什么样”
- `task_interface.py` 负责“任务怎么接上去”


## 9. 新增任务时怎么改

如果未来加一个新任务，比如 `wipe_table`，我建议只改两处：

### 第一步

新增：

```text
task/wipe_table/run.py
```

这个文件只写擦桌子的执行逻辑。


### 第二步

在 `gradio/task_interface.py` 里补一个任务定义：

```python
"wipe_table": {
    ...
}
```

同时补两个函数：

- `preview_wipe_table()`
- `execute_wipe_table()`

这样就够了。

`main.py` 和 `ui.py` 都不用动。


## 10. 为什么我现在推荐“三文件方案”

因为你现在的需求其实还没有复杂到要拆很多层：

- 任务种类不算特别多
- 页面交互模式也比较统一
- 你更看重“方便改”和“别藏太深”

所以现在最适合的不是“很标准的工程化切层”，而是：

- 少文件
- 边界清楚
- 一眼能找到改哪里

三文件正好够用。


## 11. 最终结论

我现在推荐的结构就是：

```text
gradio/
├─ main.py
├─ task_interface.py
└─ ui.py
```

职责分别是：

- `main.py`：主控启动、线程调度、状态刷新
- `task_interface.py`：和 `task` 的接口、任务注册、preview/execute 适配
- `ui.py`：页面布局和样式

这样既满足：

- `task` 纯执行
- `gradio` 纯界面和调度
- 快速增删任务
- 页面改动集中

又不会把工程拆得太碎。


如果你认可这版，我下一步就按这个三文件结构去实现，不再继续往下拆。  
