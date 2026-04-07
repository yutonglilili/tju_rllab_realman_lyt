"""
清桌任务 Gradio 控制台：在网页中选择/输入指令，点击启动后调用 clear_the_table.run_clear_table_session。

运行（请在包含本文件的目录下执行，以便正确加载 pick_and_place_utils 等本地模块）：

    cd demo/clear_the_table
    pip install -r requirements_gradio.txt
    python clear_the_table_gradio.py

默认监听 0.0.0.0:7860，可在文件末尾修改 server_port / share。
"""

from __future__ import annotations

import os
import sys
import threading

# 项目根目录与脚本目录（与 clear_the_table.py 一致）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr

from clear_the_table_test_gradio import (
    DEFAULT_CAM_RESULTS_PATH,
    DEFAULT_INSTRUCTION,
    DEFAULT_REALSENSE_SERIAL,
    DEFAULT_ROBOT_IP,
    run_clear_table_session,
    stop_clear_table_session,
)

# ── 运行状态与日志 ─────────────────────────────────────────

_log_lock = threading.Lock()
_log_text = ""
MAX_LOG_CHARS = 150_000

_run_lock = threading.Lock()
_is_running = False


def _append_log(chunk: str) -> None:
    global _log_text
    if not chunk:
        return
    with _log_lock:
        _log_text += chunk
        if len(_log_text) > MAX_LOG_CHARS:
            _log_text = _log_text[-MAX_LOG_CHARS:]


def _read_log() -> str:
    with _log_lock:
        return _log_text


def clear_log() -> str:
    global _log_text
    with _log_lock:
        _log_text = ""
    return ""


class TeeStdout:
    """同时写入原始 stdout 与网页日志缓冲区。"""

    def __init__(self, orig):
        self._orig = orig

    def write(self, s):
        if s is None:
            return 0
        self._orig.write(s)
        self._orig.flush()
        _append_log(s)
        return len(s)

    def flush(self):
        self._orig.flush()

    def isatty(self):
        return getattr(self._orig, "isatty", lambda: False)()


def apply_preset(preset_key: str) -> str:
    return PRESET_INSTRUCTIONS.get(preset_key, DEFAULT_INSTRUCTION)


def start_robot(
    instruction: str,
    robot_ip: str,
    realsense_serial: str,
    cam_results_path: str,
):
    global _is_running
    instruction = (instruction or "").strip()
    if not instruction:
        return "请先填写任务指令。", _read_log()

    with _run_lock:
        if _is_running:
            return "已有会话在运行，请等待结束或点击「停止」后再启动。", _read_log()
        _is_running = True

    clear_log()
    _append_log(f"[UI] 启动会话，指令: {instruction}\n")

    robot_ip = (robot_ip or "").strip() or DEFAULT_ROBOT_IP
    realsense_serial = (realsense_serial or "").strip() or DEFAULT_REALSENSE_SERIAL
    cam_results_path = (cam_results_path or "").strip() or DEFAULT_CAM_RESULTS_PATH

    def _worker():
        global _is_running
        tee = TeeStdout(sys.__stdout__)
        try:
            run_clear_table_session(
                instruction,
                robot_ip=robot_ip,
                realsense_serial=realsense_serial,
                cam_results_path=cam_results_path,
                stdout_tee=tee,
            )
        finally:
            with _run_lock:
                _is_running = False
            _append_log("\n[UI] 会话已结束。\n")

    threading.Thread(target=_worker, daemon=True, name="ClearTableSession").start()
    return "已在后台启动。日志下方自动刷新。", _read_log()


def stop_robot():
    ok = stop_clear_table_session()
    msg = "已发送停止信号（stop_all）。" if ok else "当前没有活跃会话。"
    return msg, _read_log()


def refresh_log():
    return _read_log()


PRESET_LABELS = [
    "默认（英文清桌）",
    "中文：收拾桌面",
]

PRESET_INSTRUCTIONS = {
    "默认（英文清桌）": DEFAULT_INSTRUCTION,
    "中文：收拾桌面": "收拾桌子。把桌面上散落的物品全部拿起，放到指定的收纳区域或盘子里。",
}


def build_ui():
    with gr.Blocks(title="清桌控制台") as demo:
        gr.Markdown("## 清桌任务控制台（Gradio）\n选择模板或输入自然语言指令，配置连接参数后点击 **启动**。")

        preset = gr.Dropdown(
            label="指令模板",
            choices=PRESET_LABELS,
            value="默认（英文清桌）",
        )
        instruction = gr.Textbox(
            label="任务指令（会传给 VLM 解析场景）",
            lines=3,
            value=DEFAULT_INSTRUCTION,
        )

        with gr.Accordion("机械臂 / 相机连接参数", open=False):
            robot_ip = gr.Textbox(label="机械臂 IP", value=DEFAULT_ROBOT_IP)
            realsense_serial = gr.Textbox(label="RealSense 序列号", value=DEFAULT_REALSENSE_SERIAL)
            cam_results_path = gr.Textbox(
                label="camera_results.json 路径",
                value=DEFAULT_CAM_RESULTS_PATH,
            )

        with gr.Row():
            btn_start = gr.Button("启动", variant="primary")
            btn_stop = gr.Button("停止")
            btn_refresh = gr.Button("刷新日志")

        status = gr.Textbox(label="状态", interactive=False, lines=1)
        log_box = gr.Textbox(label="运行日志", lines=24, max_lines=40, interactive=False)

        preset.change(fn=apply_preset, inputs=preset, outputs=instruction)

        btn_start.click(
            fn=start_robot,
            inputs=[instruction, robot_ip, realsense_serial, cam_results_path],
            outputs=[status, log_box],
        )
        btn_stop.click(fn=stop_robot, outputs=[status, log_box])
        btn_refresh.click(fn=refresh_log, outputs=log_box)

        # 周期性刷新日志（启动后无需手动点「刷新日志」）
        demo.load(refresh_log, None, log_box, every=0.5)

    return demo


if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    app = build_ui()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
