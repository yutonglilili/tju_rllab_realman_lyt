from __future__ import annotations

from typing import Any


APP_CSS = """
:root {
  --app-bg: linear-gradient(160deg, #eef4ff 0%, #f8fbff 55%, #fff7f0 100%);
  --panel-bg: rgba(255, 255, 255, 0.92);
  --panel-border: rgba(19, 48, 77, 0.10);
  --ink-strong: #16324f;
}

.gradio-container {
  background: var(--app-bg);
}

.app-panel {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  box-shadow: 0 18px 40px rgba(29, 53, 87, 0.08);
  padding: 10px;
}
"""


def build_ui(gr: Any, task_definitions: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    first_task = next(iter(task_definitions.values()))

    with gr.Blocks(title="Robot Console", css=APP_CSS) as demo:
        gr.Markdown("# Robot Console")

        with gr.Column(elem_classes=["app-panel"]):
            task_dropdown = gr.Dropdown(
                choices=[(task.title, task.task_id) for task in task_definitions.values()],
                value=first_task.task_id,
                label="任务类型",
            )

            instruction_input = gr.Textbox(
                label=first_task.input_label,
                value=first_task.default_instruction,
                lines=4,
                placeholder="输入任务指令后直接开始执行。",
            )

            with gr.Accordion("运行时连接", open=False):
                robot_ip_input = gr.Textbox(
                    label="Robot IP",
                    value="192.168.101.19",
                )
                camera_serial_input = gr.Textbox(
                    label="Camera Serial",
                    value="f1471338",
                )
                cam_results_path_input = gr.Textbox(
                    label="Camera Results Path",
                    value="/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json",
                )

            with gr.Row():
                init_runtime_button = gr.Button("初始化", variant="secondary")
                start_button = gr.Button("开始执行", variant="primary")
                stop_button = gr.Button("停止", variant="stop")

            camera_image_output = gr.Image(
                label="当前画面",
                type="numpy",
                interactive=False,
                height=440,
            )

            current_status_output = gr.Textbox(
                label="当前状态",
                value="运行状态: 未连接\n当前任务: -\n任务指令: -\n正在执行: 等待初始化",
                lines=5,
                interactive=False,
            )

            recent_actions_output = gr.Textbox(
                label="Log",
                lines=10,
                max_lines=14,
                placeholder="这里会显示最近在做什么。",
            )

        components = {
            "task_dropdown": task_dropdown,
            "instruction_input": instruction_input,
            "robot_ip_input": robot_ip_input,
            "camera_serial_input": camera_serial_input,
            "cam_results_path_input": cam_results_path_input,
            "init_runtime_button": init_runtime_button,
            "start_button": start_button,
            "stop_button": stop_button,
            "current_status_output": current_status_output,
            "camera_image_output": camera_image_output,
            "recent_actions_output": recent_actions_output,
        }

    return demo, components
