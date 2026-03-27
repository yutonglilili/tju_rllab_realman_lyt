#!/usr/bin/env bash
set -euo pipefail

# Realman 版本动态对齐数采入口。
# 注意：当前脚本默认需要一个 cuRobo 的 Realman robot yaml（包含 URDF/ee_link/base_link/joint_names 等）。

export PROJECT_ROOT="$(pwd)"
python3 dynamic_alignment_realman/pushing_realman.py \
  --robot dynamic_alignment_realman/realman.yml

