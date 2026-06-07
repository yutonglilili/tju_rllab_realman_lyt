"""
添加使用带方位描述的 pick 和 place 目标指令调用模型打点，并在感知线程中改为对目标的移动的感知，而不再是对物体的感知。
认为目标发生移动的判定：
1. 以当前打点和上一次打点的 3D 距离作为核心判断依据；
2. 当距离超过阈值时，认为目标发生移动并触发重规划；
3. 在 pick 阶段和 place 阶段，分别设置不同的移动阈值；
4. 阶段限制：只在 approach 阶段做移动感知，其他阶段不进行移动感知；
"""
import copy
import json
import os
import sys
import time
import threading
import numpy as np
import cv2
import traceback
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass

# 项目路径配置
DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SDK_PYTHON_ROOT = os.path.join(WORKSPACE_ROOT, "realman", "RM_API2", "Python")
for path in (DEMO_ROOT, WORKSPACE_ROOT, SDK_PYTHON_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from realman.realman_env import (
    RealmanEnv,
    T_from_realman_xyzrpy,
    pose_tcp2eef,
    realman_xyzrpy_from_T,
)
from realman.open3d_realsense_env import Open3dRealsenseEnv
from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t

from demo_new.skills.tools.config_utils import ConfigNamespace, load_config_with_defaults, resolve_config_path
from demo_new.skills.tools.utils import make_lift_T, make_target_T, save_obs_image, crop_image_around_point
from demo_new.skills.pnp_skill.graspgen_bridge import (
    GraspFilterConfig,
    GraspGenClientBridge,
    WristProcessingConfig,
    build_pregrasp_pose_from_grasp,
    infer_pick_grasp_candidates_from_wrist,
    load_wrist_handeye_config,
)

from demo_new.vlm_utils.multi_pointing_vllm_get_point_utils import get_point_vllm

from demo_new.vlm_utils.multi_pointing_vllm_get_point_utils_qwen import check_grasp_success_vllm, check_place_success_vllm, generate_tasks_with_descriptions

# from demo_new.vlm_utils.vllm_from_api_key import generate_tasks_with_descriptions

# ═══════════════════════════════════════════════════
# 配置参数
# ═══════════════════════════════════════════════════
DEFAULT_CONFIG_PATH = resolve_config_path(__file__)
SKILL_CONFIG_SECTION_KEYS = ("pnp_skill", "pick_and_place")
# BUFFER_SIZE/TRIGGER_COUNT_THRESHOLD are kept here for backward-compatible config loading.
SKILL_CONFIG_KEYS = (
    "PERCEPTION_INTERVAL",
    "TASK_DISCOVERY_INTERVAL",
    "BUFFER_SIZE",
    "TRIGGER_COUNT_THRESHOLD",
    "MOVE_OBJECT_THRESHOLD",
    "MOVE_CONTAINER_THRESHOLD",
    "CAMERA_X_OFFSET",
    "CAMERA_Y_OFFSET",
    "CAMERA_Z_OFFSET",
    "SAFE_HEIGHT",
    "TRAJECTORY_DOWNSAMPLE",
    "PICK_RPY",
    "PLACE_RPY",
    "RX_DEGREE_CLOSE",
    "RX_DEGREE_FAR_HIGH",
    "RX_DEGREE_FAR_LOW",
    "PRE_PICK_X_OFFSET",
    "PRE_PICK_Y_OFFSET",
    "PRE_PICK_Z_OFFSET",
    "PICK_X_OFFSET",
    "PICK_Y_OFFSET",
    "PICK_Z_OFFSET",
    "POST_PICK_X_OFFSET",
    "POST_PICK_Y_OFFSET",
    "POST_PICK_Z_OFFSET",
    "PRE_PLACE_X_OFFSET",
    "PRE_PLACE_Y_OFFSET",
    "PRE_PLACE_Z_OFFSET",
    "PLACE_X_OFFSET",
    "PLACE_Y_OFFSET",
    "PLACE_Z_OFFSET",
    "POST_PLACE_X_OFFSET",
    "POST_PLACE_Y_OFFSET",
    "POST_PLACE_Z_OFFSET",
    "CONTROL_INTERVAL",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSE",
    "MAX_CONSECUTIVE_MOTION_FAILURES",
    "MAX_PICK_RETRIES",
    "MAX_PLACE_RETRIES",
    "CHECK_PICK_SUCCESS_MODE",
    "CHECK_PLACE_SUCCESS_MODE",
    "CHECK_PICK_CROP_SIZE",
    "CHECK_PLACE_CROP_SIZE",
    "PICK_SUCCESS_DIST_THRESHOLD",
    "PLACE_SUCCESS_DIST_THRESHOLD",
    "SAVE_DIR",
)
OPTIONAL_SKILL_CONFIG_KEYS = (
    "motion_profiles",
    "SAVE_CHECK_FAIL_IMAGE",
    "GRASPGEN_SERVER_HOST",
    "GRASPGEN_SERVER_PORT",
    "GRASPGEN_TIMEOUT_MS",
    "GRASPGEN_NUM_GRASPS",
    "GRASPGEN_TOPK_NUM_GRASPS",
    "GRASPGEN_GRASP_THRESHOLD",
    "GRASPGEN_MAX_CANDIDATES",
    "DEBUG_EXIT_AFTER_GRASP_CANDIDATES",
    "GRASPGEN_CANDIDATE_PREGRASP_OFFSET_M",
    "WRIST_MIN_DEPTH_M",
    "WRIST_MAX_DEPTH_M",
    "WRIST_DEPTH_PATCH_RADIUS",
    "WRIST_CLICK_SEED_RADIUS_PX",
    "WRIST_SEED_SEARCH_RADIUS_PX",
    "WRIST_REGION_GROW_NEIGHBOR_RADIUS_PX",
    "WRIST_REGION_GROW_3D_THRESHOLD_M",
    "WRIST_REGION_GROW_DEPTH_THRESHOLD_M",
    "WRIST_REGION_GROW_COLOR_THRESHOLD",
    "WRIST_REGION_GROW_MAX_SEED_DISTANCE_M",
    "WRIST_MIN_MASK_PIXELS",
    "WRIST_MASK_KERNEL_SIZE",
    "WRIST_TABLE_HEIGHT_PERCENTILE",
    "WRIST_TABLE_REMOVE_MARGIN_M",
    "WRIST_MAX_OBJECT_POINTS",
    "WRIST_OBJECT_VOXEL_SIZE",
    "WRIST_MAX_SCENE_POINTS",
    "WRIST_SCENE_VOXEL_SIZE",
    "GRASPGEN_DIRECTION_RULE_TARGET_DIR_CAMERA",
    "GRASPGEN_DIRECTION_RULE_MAX_ANGLE_DEG",
    "GRASPGEN_DIRECTION_RULE_MIN_FORWARD_COMPONENT",
    "GRASPGEN_DIRECTION_RULE_MIN_DOWN_COMPONENT",
    "GRASPGEN_DIRECTION_RULE_MAX_LATERAL_COMPONENT",
)

Config = ConfigNamespace

def load_pnp_config(task_config_path=None, task_config=None):
    return load_config_with_defaults(
        default_config_path=DEFAULT_CONFIG_PATH,
        override_config_path=task_config_path,
        override_config=task_config,
        section_keys=SKILL_CONFIG_SECTION_KEYS,
        allowed_keys=SKILL_CONFIG_KEYS + OPTIONAL_SKILL_CONFIG_KEYS,
        required_keys=SKILL_CONFIG_KEYS,
        config_cls=Config,
    )


def _config_like_to_dict(config_like):
    if config_like is None:
        return {}

    if hasattr(config_like, "to_dict") and callable(config_like.to_dict):
        config_dict = config_like.to_dict()
        if not isinstance(config_dict, dict):
            raise TypeError("ConfigNamespace.to_dict() must return a dict.")
        return dict(config_dict)

    if isinstance(config_like, dict):
        return dict(config_like)

    raise TypeError(
        f"Motion profile must be a dict-like object, got {type(config_like).__name__}."
    )


def _normalize_motion_profile(profile):
    profile_dict = _config_like_to_dict(profile)

    for rpy_key in ("PICK_RPY", "PLACE_RPY"):
        rpy_value = profile_dict.get(rpy_key)

        if isinstance(rpy_value, np.ndarray):
            profile_dict[rpy_key] = rpy_value.tolist()
        elif isinstance(rpy_value, tuple):
            profile_dict[rpy_key] = list(rpy_value)

    return profile_dict


def _build_phase_config(state, task_phase):
    config_values = _config_like_to_dict(state.config)

    with state.lock:
        current_task = state.current_task or {}
        profile_key = "_pick_profile" if task_phase == TaskPhase.PICK else "_place_profile"
        profile = current_task.get(profile_key)

    if profile:
        config_values.update(_normalize_motion_profile(profile))

    return Config(config_values)


def _build_wrist_processing_config(config) -> WristProcessingConfig:
    return WristProcessingConfig(
        min_depth_m=float(getattr(config, "WRIST_MIN_DEPTH_M", 0.10)),
        max_depth_m=float(getattr(config, "WRIST_MAX_DEPTH_M", 1.20)),
        depth_patch_radius=int(getattr(config, "WRIST_DEPTH_PATCH_RADIUS", 3)),
        click_seed_radius_px=int(getattr(config, "WRIST_CLICK_SEED_RADIUS_PX", 5)),
        seed_search_radius_px=int(getattr(config, "WRIST_SEED_SEARCH_RADIUS_PX", 15)),
        region_grow_neighbor_radius_px=int(
            getattr(config, "WRIST_REGION_GROW_NEIGHBOR_RADIUS_PX", 2)
        ),
        region_grow_3d_threshold_m=float(
            getattr(config, "WRIST_REGION_GROW_3D_THRESHOLD_M", 0.018)
        ),
        region_grow_depth_threshold_m=float(
            getattr(config, "WRIST_REGION_GROW_DEPTH_THRESHOLD_M", 0.030)
        ),
        region_grow_color_threshold=float(
            getattr(config, "WRIST_REGION_GROW_COLOR_THRESHOLD", 140.0)
        ),
        region_grow_max_seed_distance_m=float(
            getattr(config, "WRIST_REGION_GROW_MAX_SEED_DISTANCE_M", 0.24)
        ),
        min_mask_pixels=int(getattr(config, "WRIST_MIN_MASK_PIXELS", 120)),
        mask_kernel_size=int(getattr(config, "WRIST_MASK_KERNEL_SIZE", 3)),
        table_height_percentile=float(
            getattr(config, "WRIST_TABLE_HEIGHT_PERCENTILE", 8.0)
        ),
        table_remove_margin_m=float(getattr(config, "WRIST_TABLE_REMOVE_MARGIN_M", 0.008)),
        max_object_points=int(getattr(config, "WRIST_MAX_OBJECT_POINTS", 4096)),
        object_voxel_size=float(getattr(config, "WRIST_OBJECT_VOXEL_SIZE", 0.003)),
        max_scene_points=int(getattr(config, "WRIST_MAX_SCENE_POINTS", 8192)),
        scene_voxel_size=float(getattr(config, "WRIST_SCENE_VOXEL_SIZE", 0.004)),
    )


def _build_grasp_filter_config(config) -> GraspFilterConfig:
    target_dir = getattr(config, "GRASPGEN_DIRECTION_RULE_TARGET_DIR_CAMERA", [0.0, 0.64, 0.77])
    return GraspFilterConfig(
        grasp_threshold=float(getattr(config, "GRASPGEN_GRASP_THRESHOLD", -1.0)),
        num_grasps=int(getattr(config, "GRASPGEN_NUM_GRASPS", 200)),
        topk_num_grasps=int(getattr(config, "GRASPGEN_TOPK_NUM_GRASPS", 50)),
        max_candidates=int(getattr(config, "GRASPGEN_MAX_CANDIDATES", 5)),
        candidate_pregrasp_offset_m=float(
            getattr(config, "GRASPGEN_CANDIDATE_PREGRASP_OFFSET_M", 0.10)
        ),
        direction_rule_target_dir_camera=tuple(float(v) for v in target_dir),
        direction_rule_max_angle_deg=float(
            getattr(config, "GRASPGEN_DIRECTION_RULE_MAX_ANGLE_DEG", 35.0)
        ),
        direction_rule_min_forward_component=float(
            getattr(config, "GRASPGEN_DIRECTION_RULE_MIN_FORWARD_COMPONENT", 0.30)
        ),
        direction_rule_min_down_component=float(
            getattr(config, "GRASPGEN_DIRECTION_RULE_MIN_DOWN_COMPONENT", 0.20)
        ),
        direction_rule_max_lateral_component=float(
            getattr(config, "GRASPGEN_DIRECTION_RULE_MAX_LATERAL_COMPONENT", 0.45)
        ),
        grasp_to_tcp_depth_m=float(
            getattr(config, "GRASPGEN_GRASP_TO_TCP_DEPTH_M", 0.195)
        ),
        grasp_to_tcp_roll_deg=float(
            getattr(config, "GRASPGEN_GRASP_TO_TCP_ROLL_DEG", 0.0)
        ),
    )


def _print_grasp_candidate_summary(state) -> None:
    grasp_pose_pool = state.grasp_pose_pool_base.copy()
    grasp_scores = state.grasp_pose_pool_scores.copy()

    print("[调试] GraspGen 候选抓取姿态如下（脚本将在执行前退出）:")
    if len(grasp_pose_pool) == 0:
        print("[调试] 当前没有可用候选姿态。")
        return

    for idx, grasp_T_base in enumerate(grasp_pose_pool):
        score = float(grasp_scores[idx]) if idx < len(grasp_scores) else None
        pose_tcp = _matrix_to_tcp_pose(grasp_T_base)
        pose_eef = pose_tcp2eef(pose_tcp)
        score_text = "None" if score is None else f"{score:.4f}"
        print(
            f"[调试] candidate {idx + 1}/{len(grasp_pose_pool)} "
            f"score={score_text} "
            f"tcp_xyzrpy={np.round(pose_tcp, 4).tolist()}"
            f"eef_xyzrpy={np.round(pose_eef, 4).tolist()}"
        )
        print(np.array2string(grasp_T_base, precision=4, suppress_small=True))


# ═══════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════

class TaskPhase(Enum):
    """当前任务所处阶段"""
    IDLE = auto()
    PICK = auto()
    PLACE = auto()
    COMPLETE = auto()


class PickStage(Enum):
    """抓取阶段的细粒度状态机。"""
    IDLE = auto()
    GLOBAL_TRACKING = auto()
    PREGRASP_EXECUTING = auto()
    WRIST_SENSING = auto()
    GRASP_PLAN_READY = auto()
    GRASP_EXECUTING = auto()
    VERIFYING = auto()


@dataclass
class PickPoseBundle:
    """Heuristic pick triplet used for coarse pregrasp and fallback grasping."""

    pick_T: np.ndarray
    pre_pick_T: np.ndarray
    post_pick_T: np.ndarray


# ═══════════════════════════════════════════════════
# 共享状态
# ═══════════════════════════════════════════════════

class SharedState:
    """线程间共享状态，所有读写必须在 self.lock 内"""

    def __init__(self, config):
        self.lock = threading.Lock()

        self.config = config

        # ===== 任务信息 =====
        self.current_task = None                    # {'pick': ..., 'place': ...}
        self.task_phase = TaskPhase.IDLE
        self.pick_stage = PickStage.IDLE

        # ===== 感知输出 =====
        # ===== 移动检测 =====

        self.target_description = None              # 当前追踪的目标描述
        self.latest_point_2d = None                 # 最新 2D 打点结果 (x, y)
        self.latest_point_3d = None                 # 最新 3D 坐标 (base 坐标系)
        self.latest_target_T = None                 # 最新目标物体的 4x4 位姿矩阵（此处为 TCP2BASE）
        self.previous_point_3d = None               # 上一次打点对应的 3D 坐标
        self.point_changed = False
        self.is_first_point = True
        self.tracking_mode = False                  # 追踪模式
        self.tracking_session_id = 0               # Invalidate in-flight tracking work across mode switches
        self.wrist_mode = False                     # 腕部单次感知模式
        self.wrist_session_id = 0
        self.verify_mode = False                    # 验证模式

        # ===== 规划输出 =====
        self.action_list = []                       # [{"joints": ..., "gripper": ..., "tag": ...}, ...]
        self.action_index = 0
        self.plan_ready = threading.Event()         # 规划完成信号
        self.need_replan = threading.Event()        # 需要重规划信号
        self.wrist_result_ready = threading.Event()
        self.grasp_plan_ready = threading.Event()

        self.pregrasp_pose_base = None
        self.fallback_pregrasp_pose_base = None
        self.fallback_pick_pose_base = None
        self.post_pick_pose_base = None

        self.grasp_pose_pool_base = np.zeros((0, 4, 4), dtype=np.float32)
        self.grasp_pregrasp_pool_base = np.zeros((0, 4, 4), dtype=np.float32)
        self.grasp_pose_pool_scores = np.zeros((0,), dtype=np.float32)
        self.grasp_eef_xyzrpy_pool = np.zeros((0, 6), dtype=np.float64)
        self.selected_grasp_pose_base = None
        self.selected_grasp_score = None
        self.grasp_plan_source = None

        # ===== 执行控制 =====
        self.attemp_count = 0
        self.abort_execution = threading.Event()    # 停止当前执行信号

        # ===== 任务结果 =====
        self.task_done = threading.Event()
        self.task_success = False

        # ===== 全局控制 =====
        self.stop_all = threading.Event()           # 全局停止

        # ===== 腕部相机与 GraspGen =====
        self.wrist_grasp_enabled = False
        self.wrist_click_2d = None
        self.wrist_mask = None
        self.wrist_object_pc_base = None
        self.wrist_scene_pc_base = None
        self.wrist_grasp_debug = None
    
    def reset_state(self):
        with self.lock:

            self.current_task = None
            self.task_phase = TaskPhase.IDLE
            self.pick_stage = PickStage.IDLE

            self.target_description = None
            self.latest_point_2d = None
            self.latest_point_3d = None
            self.latest_target_T = None
            self.previous_point_3d = None
            self.point_changed = False
            self.is_first_point = True
            self.tracking_session_id += 1
            self.tracking_mode = False
            self.wrist_session_id += 1
            self.wrist_mode = False
            self.verify_mode = False
            self.wrist_click_2d = None
            self.wrist_mask = None
            self.wrist_object_pc_base = None
            self.wrist_scene_pc_base = None
            self.wrist_grasp_debug = None

            self.action_list = []
            self.action_index = 0
            self.plan_ready.clear()
            self.need_replan.clear()
            self.wrist_result_ready.clear()
            self.grasp_plan_ready.clear()
            self.pregrasp_pose_base = None
            self.fallback_pregrasp_pose_base = None
            self.fallback_pick_pose_base = None
            self.post_pick_pose_base = None
            self.grasp_pose_pool_base = np.zeros((0, 4, 4), dtype=np.float32)
            self.grasp_pregrasp_pool_base = np.zeros((0, 4, 4), dtype=np.float32)
            self.grasp_pose_pool_scores = np.zeros((0,), dtype=np.float32)
            self.grasp_eef_xyzrpy_pool = np.zeros((0, 6), dtype=np.float64)
            self.selected_grasp_pose_base = None
            self.selected_grasp_score = None
            self.grasp_plan_source = None

            self.attemp_count = 0
            self.abort_execution.clear()

            self.task_done.clear()
            self.task_success = False


def _set_tracking_mode_locked(state, enabled):
    if state.tracking_mode != enabled:
        state.tracking_session_id += 1
    state.tracking_mode = enabled


def _set_wrist_mode_locked(state, enabled):
    if state.wrist_mode != enabled:
        state.wrist_session_id += 1
    state.wrist_mode = enabled


def _tracking_request_is_stale(state, tracking_session_id, task_phase, target_description):
    with state.lock:
        return (
            (not state.tracking_mode)
            or state.tracking_session_id != tracking_session_id
            or state.task_phase != task_phase
            or state.target_description != target_description
        )


def _wrist_request_is_stale(state, wrist_session_id, target_description):
    with state.lock:
        return (
            (not state.wrist_mode)
            or state.wrist_session_id != wrist_session_id
            or state.task_phase != TaskPhase.PICK
            or state.target_description != target_description
        )


def _empty_grasp_pose_pool():
    return np.zeros((0, 4, 4), dtype=np.float32)


def _empty_grasp_score_pool():
    return np.zeros((0,), dtype=np.float32)


def _clear_grasp_outputs_locked(state):
    state.wrist_click_2d = None
    state.wrist_mask = None
    state.wrist_object_pc_base = None
    state.wrist_scene_pc_base = None
    state.wrist_grasp_debug = None
    state.grasp_pose_pool_base = _empty_grasp_pose_pool()
    state.grasp_pregrasp_pool_base = _empty_grasp_pose_pool()
    state.grasp_pose_pool_scores = _empty_grasp_score_pool()
    state.grasp_eef_xyzrpy_pool = np.zeros((0, 6), dtype=np.float64)
    state.selected_grasp_pose_base = None
    state.selected_grasp_score = None
    state.grasp_plan_source = None


def _reset_pick_tracking_locked(state, target_description, *, attempt_count=None):
    state.task_phase = TaskPhase.PICK
    state.pick_stage = PickStage.GLOBAL_TRACKING
    state.target_description = target_description
    state.latest_point_2d = None
    state.latest_point_3d = None
    state.latest_target_T = None
    state.previous_point_3d = None
    state.point_changed = False
    state.is_first_point = True
    state.action_list = []
    state.action_index = 0
    state.plan_ready.clear()
    state.need_replan.clear()
    state.wrist_result_ready.clear()
    state.grasp_plan_ready.clear()
    state.abort_execution.clear()
    state.pregrasp_pose_base = None
    state.fallback_pregrasp_pose_base = None
    state.fallback_pick_pose_base = None
    state.post_pick_pose_base = None
    _clear_grasp_outputs_locked(state)
    _set_wrist_mode_locked(state, False)
    _set_tracking_mode_locked(state, True)
    state.verify_mode = False
    if attempt_count is not None:
        state.attemp_count = attempt_count


def _matrix_to_tcp_pose(T_base: np.ndarray, min_z: float = 0.01) -> np.ndarray:
    pose = np.asarray(realman_xyzrpy_from_T(T_base), dtype=np.float64).copy()
    pose[2] = max(float(pose[2]), float(min_z))
    return pose


def _build_pick_pose_bundle(state, target_T, home_T_tcp2base) -> PickPoseBundle:
    config = _build_phase_config(state, TaskPhase.PICK)

    target_T = make_lift_T(
        target_T,
        lift_x=config.PICK_X_OFFSET,
        lift_y=config.PICK_Y_OFFSET,
        lift_z=config.PICK_Z_OFFSET,
    )

    if config.PICK_RPY is not None:
        print(
            "[PnP] use PICK_RPY override:",
            np.round(np.asarray(config.PICK_RPY, dtype=float), 4).tolist(),
        )
        target_pose = realman_xyzrpy_from_T(target_T)
        target_pose[3:] = np.asarray(config.PICK_RPY, dtype=float)
        pick_T = T_from_realman_xyzrpy(target_pose)
    else:
        pick_T = adjust_target_T(state, target_T, home_T_tcp2base)

    pick_pose = _matrix_to_tcp_pose(pick_T)
    pick_T = T_from_realman_xyzrpy(pick_pose)

    pre_pick_T = make_lift_T(
        pick_T,
        lift_x=config.PRE_PICK_X_OFFSET,
        lift_y=config.PRE_PICK_Y_OFFSET,
        lift_z=config.PRE_PICK_Z_OFFSET,
    )

    # 预抓取位姿额外增加 20 度俯仰（绕 X 轴），让腕部相机更朝下看
    pre_pick_extra_rx = np.radians(20.0)
    Rx_pre = np.array([
        [1, 0, 0],
        [0, np.cos(pre_pick_extra_rx), -np.sin(pre_pick_extra_rx)],
        [0, np.sin(pre_pick_extra_rx), np.cos(pre_pick_extra_rx)]
    ])
    pre_pick_T[:3, :3] = Rx_pre @ pre_pick_T[:3, :3]
    post_pick_T = make_lift_T(
        pick_T,
        lift_x=config.POST_PICK_X_OFFSET,
        lift_y=config.POST_PICK_Y_OFFSET,
        lift_z=config.POST_PICK_Z_OFFSET,
    )

    return PickPoseBundle(
        pick_T=pick_T,
        pre_pick_T=pre_pick_T,
        post_pick_T=post_pick_T,
    )


def _build_pick_pregrasp_action_list(state, target_T, home_T_tcp2base):
    config = _build_phase_config(state, TaskPhase.PICK)
    pose_bundle = _build_pick_pose_bundle(state, target_T, home_T_tcp2base)
    pre_pick_pose = _matrix_to_tcp_pose(pose_bundle.pre_pick_T)
    action_list = [
        {
            "pose": pre_pick_pose,
            "gripper": config.GRIPPER_OPEN,
            "tag": 0,
            "motion": "pose",
            "wait_gripper": False,
        }
    ]
    return action_list, pose_bundle


def _build_post_pick_from_grasp(state, grasp_T_base):
    config = _build_phase_config(state, TaskPhase.PICK)
    return make_lift_T(
        grasp_T_base,
        lift_x=config.POST_PICK_X_OFFSET,
        lift_y=config.POST_PICK_Y_OFFSET,
        lift_z=config.POST_PICK_Z_OFFSET,
    )


def _build_candidate_pregrasp_from_grasp(state, grasp_T_base):
    grasp_filter_cfg = _build_grasp_filter_config(state.config)
    return build_pregrasp_pose_from_grasp(
        np.asarray(grasp_T_base, dtype=np.float32),
        retreat_m=grasp_filter_cfg.candidate_pregrasp_offset_m,
    )


def _build_fallback_pick_action_list(state, pick_T, pre_pick_T, post_pick_T):
    config = _build_phase_config(state, TaskPhase.PICK)
    return [
        {
            "pose": _matrix_to_tcp_pose(pre_pick_T),
            "gripper": config.GRIPPER_OPEN,
            "tag": 0,
            "motion": "pose",
            "wait_gripper": False,
        },
        {
            "pose": _matrix_to_tcp_pose(pick_T),
            "gripper": config.GRIPPER_CLOSE,
            "tag": 1,
            "motion": "pose",
            "wait_gripper": True,
        },
        {
            "pose": _matrix_to_tcp_pose(post_pick_T),
            "tag": 2,
            "motion": "pose",
        },
    ]


def _pose_to_step_action(action):
    step_action = {}

    if "joints" in action:
        joint_deg = (
            np.degrees(action["joints"])
            if np.max(np.abs(action["joints"])) < 2 * np.pi
            else action["joints"]
        )
        step_action["joint"] = joint_deg
    elif "pose" in action:
        step_action["pose"] = action["pose"]

    if "motion" in action:
        step_action["motion"] = action["motion"]

    if "gripper" in action:
        step_action["gripper"] = action["gripper"]

    if "wait_gripper" in action:
        step_action["wait_gripper"] = action["wait_gripper"]

    return step_action


def _get_current_joint_seed(env) -> np.ndarray:
    robot_state = env.get_state()
    if robot_state is None or robot_state.joint is None:
        raise RuntimeError("Failed to read the current robot joint state.")
    return np.asarray(robot_state.joint, dtype=np.float64)


def solve_pose_ik(env, pose_tcp, seed_joint_rad=None):
    pose_tcp = np.asarray(pose_tcp, dtype=np.float64).reshape(6)
    seed_joint_rad = (
        _get_current_joint_seed(env)
        if seed_joint_rad is None
        else np.asarray(seed_joint_rad, dtype=np.float64).reshape(-1)
    )

    arm = env.driver.arm
    arm_dof = int(getattr(arm, "arm_dof", 7) or 7)
    q_in_deg = np.degrees(seed_joint_rad[:arm_dof]).astype(np.float32)
    if arm_dof < 7:
        q_in_deg = np.pad(q_in_deg, (0, 7 - arm_dof))

    pose_eef = pose_tcp2eef(pose_tcp)
    params = rm_inverse_kinematics_params_t(
        q_in=q_in_deg.tolist(),
        q_pose=np.asarray(pose_eef, dtype=np.float32).tolist(),
        flag=1,
    )

    ik_method = "generic"
    ret, joint_deg = arm.rm_algo_inverse_kinematics(params)

    if ret != 0:
        try:
            arm_angle_ret, arm_angle = arm.rm_algo_calculate_arm_angle_from_config_rm75(
                q_in_deg.tolist()
            )
            if arm_angle_ret == 0:
                rm75_ret, rm75_joint_deg = arm.rm_algo_inverse_kinematics_rm75_for_arm_angle(
                    params,
                    arm_angle,
                )
                if rm75_ret == 0:
                    ret = rm75_ret
                    joint_deg = rm75_joint_deg
                    ik_method = "rm75_arm_angle"
        except Exception:
            pass

    if ret != 0:
        return False, None, {"ret": int(ret), "method": ik_method}

    joint_deg = np.asarray(joint_deg[:arm_dof], dtype=np.float64)
    if not np.all(np.isfinite(joint_deg)):
        return False, None, {"ret": int(ret), "method": ik_method, "reason": "nan_joint"}

    collision_ret = arm.rm_algo_safety_robot_self_collision_detection(joint_deg.tolist())
    if collision_ret != 0:
        return (
            False,
            np.radians(joint_deg),
            {
                "ret": int(ret),
                "collision_ret": int(collision_ret),
                "method": ik_method,
            },
        )

    return (
        True,
        np.radians(joint_deg),
        {
            "ret": int(ret),
            "collision_ret": int(collision_ret),
            "method": ik_method,
        },
    )


def check_pose_sequence_reachable(env, tcp_pose_sequence, seed_joint_rad=None):
    reachability_debug = []
    joint_seed = seed_joint_rad

    for pose_tcp in tcp_pose_sequence:
        reachable, joint_seed, ik_info = solve_pose_ik(env, pose_tcp, seed_joint_rad=joint_seed)
        reachability_debug.append(
            {
                "pose_tcp": np.round(np.asarray(pose_tcp, dtype=np.float64), 4).tolist(),
                "ik": ik_info,
            }
        )
        if not reachable:
            return False, joint_seed, reachability_debug

    return True, joint_seed, reachability_debug


def _execute_action_sequence_direct(env, action_list):
    for action in action_list:
        env.step(_pose_to_step_action(action))


def try_execute_grasp_candidate(state, env, grasp_T_base, pregrasp_T_base, score):
    grasp_pose_tcp = _matrix_to_tcp_pose(grasp_T_base)
    pregrasp_pose_tcp = _matrix_to_tcp_pose(pregrasp_T_base)
    postgrasp_T_base = _build_post_pick_from_grasp(state, grasp_T_base)
    postgrasp_pose_tcp = _matrix_to_tcp_pose(postgrasp_T_base)

    reachable, _, reachability_debug = check_pose_sequence_reachable(
        env,
        [pregrasp_pose_tcp, grasp_pose_tcp, postgrasp_pose_tcp],
    )
    if not reachable:
        return False, {"stage": "reachability", "reachability_debug": reachability_debug}

    config = _build_phase_config(state, TaskPhase.PICK)
    action_list = [
        {
            "pose": pregrasp_pose_tcp,
            "gripper": config.GRIPPER_OPEN,
            "tag": 0,
            "motion": "pose",
            "wait_gripper": False,
        },
        {
            "pose": grasp_pose_tcp,
            "gripper": config.GRIPPER_CLOSE,
            "tag": 1,
            "motion": "pose",
            "wait_gripper": True,
        },
        {
            "pose": postgrasp_pose_tcp,
            "tag": 2,
            "motion": "pose",
        },
    ]

    _execute_action_sequence_direct(env, action_list)

    with state.lock:
        state.selected_grasp_pose_base = np.asarray(grasp_T_base, dtype=np.float32).copy()
        state.selected_grasp_score = None if score is None else float(score)
        state.post_pick_pose_base = np.asarray(postgrasp_T_base, dtype=np.float32).copy()

    return (
        True,
        {
            "stage": "executed",
            "reachability_debug": reachability_debug,
        },
    )


def execute_fallback_pick(state, env):
    with state.lock:
        pick_T = None if state.fallback_pick_pose_base is None else state.fallback_pick_pose_base.copy()
        pre_pick_T = (
            None
            if state.fallback_pregrasp_pose_base is None
            else state.fallback_pregrasp_pose_base.copy()
        )
        post_pick_T = None if state.post_pick_pose_base is None else state.post_pick_pose_base.copy()

    if pick_T is None or pre_pick_T is None or post_pick_T is None:
        return False, {"stage": "fallback", "reason": "missing_fallback_pose"}

    pick_pose_tcp = _matrix_to_tcp_pose(pick_T)
    pre_pick_pose_tcp = _matrix_to_tcp_pose(pre_pick_T)
    post_pick_pose_tcp = _matrix_to_tcp_pose(post_pick_T)
    reachable, _, reachability_debug = check_pose_sequence_reachable(
        env,
        [pre_pick_pose_tcp, pick_pose_tcp, post_pick_pose_tcp],
    )
    if not reachable:
        return False, {"stage": "fallback_reachability", "reachability_debug": reachability_debug}

    action_list = _build_fallback_pick_action_list(state, pick_T, pre_pick_T, post_pick_T)
    _execute_action_sequence_direct(env, action_list)

    with state.lock:
        state.selected_grasp_pose_base = np.asarray(pick_T, dtype=np.float32).copy()
        state.selected_grasp_score = None
        state.grasp_plan_source = "heuristic_fallback"

    return True, {"stage": "fallback_executed", "reachability_debug": reachability_debug}


def _consume_action_list_execution(state, env):
    config = state.config
    motion_fail_streak = 0

    while not state.stop_all.is_set():
        if state.abort_execution.is_set():
            print("[执行] ⏹️ 执行被中止，等待重新规划")
            with state.lock:
                state.action_list = []
                state.action_index = 0
            state.plan_ready.clear()
            return "aborted"

        with state.lock:
            if state.action_index >= len(state.action_list):
                state.plan_ready.clear()
                return "completed"
            action = copy.deepcopy(state.action_list[state.action_index])

        if action.get("tag") != 0:
            with state.lock:
                _set_tracking_mode_locked(state, False)
                state.verify_mode = False

        try:
            env.step(_pose_to_step_action(action))
        except RuntimeError as exc:
            motion_fail_streak += 1
            print(
                f"[执行] ⚠️ 运动失败 ({motion_fail_streak}/{config.MAX_CONSECUTIVE_MOTION_FAILURES}): {exc}"
            )
            if motion_fail_streak >= config.MAX_CONSECUTIVE_MOTION_FAILURES:
                print("[执行] ⛔ 连续运动失败达到上限，终止本段轨迹并请求重新规划")
                with state.lock:
                    state.action_list = []
                    state.action_index = 0
                state.abort_execution.set()
                state.plan_ready.clear()
                state.need_replan.set()
                return "replan"
            continue

        if state.abort_execution.is_set():
            print("[执行] ⏹️ 动作完成后检测到中止信号，停止后续动作")
            with state.lock:
                state.action_list = []
                state.action_index = 0
            state.plan_ready.clear()
            return "aborted"

        motion_fail_streak = 0
        with state.lock:
            state.action_index += 1

    return "stopped"


# ═══════════════════════════════════════════════════
# 感知线程
# ═══════════════════════════════════════════════════

def perception_thread(
    state,
    env,
    rs_env,
    cam_results,
    home_T_tcp2base,
    wrist_rs_env=None,
    graspgen_client=None,
    wrist_handeye_config=None,
):
    """
    感知线程, 分为两种模式:
    1. 追踪模式: 持续以固定频率调用 VLM 打点，检测目标变化。
    2. 验证模式: 调用 VLM 验证抓取/放置是否成功。
    """
    print("[感知线程] 已启动")

    config = state.config
    wrist_processing_cfg = _build_wrist_processing_config(config)
    grasp_filter_cfg = _build_grasp_filter_config(config)

    while not state.stop_all.is_set():

        if state.tracking_mode:
            with state.lock:
                target_description = state.target_description
                task_phase = state.task_phase
                tracking_session_id = state.tracking_session_id
                if target_description is None:
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

            try:
                obs = rs_env.step()
                image_rgb = obs["rgb"]

                if _tracking_request_is_stale(
                    state,
                    tracking_session_id,
                    task_phase,
                    target_description,
                ):
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

                point_2d = get_point_vllm(image_rgb, f"Point the {target_description}", save_path=None)

                if _tracking_request_is_stale(
                    state,
                    tracking_session_id,
                    task_phase,
                    target_description,
                ):
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

                target_T = make_target_T(obs, int(point_2d[0]), int(point_2d[1]), rs_env, cam_results, home_T_tcp2base)

                target_T = make_lift_T(
                    target_T,
                    lift_x=config.CAMERA_X_OFFSET,
                    lift_y=config.CAMERA_Y_OFFSET,
                    lift_z=config.CAMERA_Z_OFFSET,
                )

                target_xyz = target_T[:3, 3]

                tracking_stale = False
                with state.lock:
                    tracking_stale = (
                        (not state.tracking_mode)
                        or state.tracking_session_id != tracking_session_id
                        or state.task_phase != task_phase
                        or state.target_description != target_description
                    )
                    if not tracking_stale:
                        state.latest_point_2d = point_2d.copy()
                        state.latest_point_3d = target_xyz.copy()
                        state.latest_target_T = target_T.copy()

                        if state.is_first_point:
                            state.previous_point_3d = target_xyz.copy()
                            state.is_first_point = False
                            state.point_changed = True
                            state.need_replan.set()
                            print(f"[感知] 📍 首次定位 {target_description}: xyz={np.round(target_xyz, 4)}")

                        else:
                            moved, dist = detect_target_movement(state, target_xyz, task_phase)

                            if moved:
                                state.point_changed = True
                                state.abort_execution.set()
                                state.need_replan.set()

                                label = "物体" if state.task_phase == TaskPhase.PICK else "容器"
                                print(f"[感知] ⚠️ {label} {target_description} 移动！距离: {dist:.4f}m → 重规划")
                            else:
                                state.point_changed = False

                if tracking_stale:
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

            except Exception as e:
                print(f"[感知] 异常: {e}")

        elif state.wrist_mode:
            with state.lock:
                target_description = state.target_description
                wrist_session_id = state.wrist_session_id
                wrist_enabled = state.wrist_grasp_enabled

            if target_description is None:
                time.sleep(config.PERCEPTION_INTERVAL)
                continue

            if not wrist_enabled or wrist_rs_env is None or graspgen_client is None or wrist_handeye_config is None:
                error_msg = "Wrist GraspGen resources are not available in the main control process."
                with state.lock:
                    wrist_stale = (
                        (not state.wrist_mode)
                        or state.wrist_session_id != wrist_session_id
                        or state.task_phase != TaskPhase.PICK
                        or state.target_description != target_description
                    )
                    if not wrist_stale:
                        _set_wrist_mode_locked(state, False)
                        _clear_grasp_outputs_locked(state)
                        state.wrist_grasp_debug = {"error": error_msg}
                        state.wrist_result_ready.set()
                print(f"[腕部感知] ⚠️ {error_msg}")
                time.sleep(config.PERCEPTION_INTERVAL)
                continue

            try:
                wrist_obs = wrist_rs_env.step()
                if _wrist_request_is_stale(state, wrist_session_id, target_description):
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

                wrist_rgb = wrist_obs["rgb"]
                click_2d = get_point_vllm(wrist_rgb, f"Point the {target_description}", save_path=None)
                click_2d = np.asarray(click_2d, dtype=np.int32).reshape(2)

                if _wrist_request_is_stale(state, wrist_session_id, target_description):
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

                robot_state = env.get_state()
                if robot_state is None or robot_state.pose is None:
                    raise RuntimeError("Failed to read robot TCP pose before wrist grasp inference.")

                wrist_result = infer_pick_grasp_candidates_from_wrist(
                    wrist_obs=wrist_obs,
                    click_point_2d=(int(click_2d[0]), int(click_2d[1])),
                    robot_pose_tcp=np.asarray(robot_state.pose, dtype=np.float64),
                    handeye_config=wrist_handeye_config,
                    graspgen_client=graspgen_client,
                    processing_cfg=wrist_processing_cfg,
                    grasp_filter_cfg=grasp_filter_cfg,
                )

                if _wrist_request_is_stale(state, wrist_session_id, target_description):
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

                wrist_stale = False
                with state.lock:
                    wrist_stale = (
                        (not state.wrist_mode)
                        or state.wrist_session_id != wrist_session_id
                        or state.task_phase != TaskPhase.PICK
                        or state.target_description != target_description
                    )
                    if not wrist_stale:
                        state.wrist_click_2d = np.asarray(click_2d, dtype=np.int32)
                        state.wrist_mask = wrist_result.mask.copy()
                        state.wrist_object_pc_base = wrist_result.object_pc_base.copy()
                        state.wrist_scene_pc_base = wrist_result.scene_pc_base.copy()
                        state.wrist_grasp_debug = {
                            **wrist_result.debug_info,
                            "success": bool(wrist_result.success),
                            "error": wrist_result.error,
                        }
                        state.grasp_pose_pool_base = wrist_result.grasp_pose_pool_base.copy()
                        state.grasp_pregrasp_pool_base = wrist_result.grasp_pregrasp_pool_base.copy()
                        state.grasp_pose_pool_scores = wrist_result.grasp_pose_pool_scores.copy()
                        state.grasp_eef_xyzrpy_pool = wrist_result.grasp_eef_xyzrpy_pool.copy()
                        _set_wrist_mode_locked(state, False)
                        state.wrist_result_ready.set()

                if wrist_stale:
                    time.sleep(config.PERCEPTION_INTERVAL)
                    continue

                print(
                    "[腕部感知] ✅ 完成局部抓取推理: "
                    f"候选 {len(wrist_result.grasp_pose_pool_base)} / 全部 {len(wrist_result.all_grasps_base)}"
                )

            except Exception as e:
                error_msg = str(e)
                debug_info = {
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                }
                with state.lock:
                    wrist_stale = (
                        (not state.wrist_mode)
                        or state.wrist_session_id != wrist_session_id
                        or state.task_phase != TaskPhase.PICK
                        or state.target_description != target_description
                    )
                    if not wrist_stale:
                        _set_wrist_mode_locked(state, False)
                        _clear_grasp_outputs_locked(state)
                        state.wrist_grasp_debug = debug_info
                        state.wrist_result_ready.set()
                print(f"[腕部感知] 异常: {error_msg}")

        elif state.verify_mode:

            with state.lock:
                current_task = state.current_task
                task_phase = state.task_phase
                target_description = state.target_description
                check_point_2d = None if state.latest_point_2d is None else state.latest_point_2d.copy()
                attemp_count = state.attemp_count

            if current_task is None:
                time.sleep(0.01)
                continue

            if task_phase == TaskPhase.PICK:
                
                pick_success = do_check_pick_success(state, env, rs_env, current_task['pick'], point_2d=check_point_2d, cam_results=cam_results, home_T_tcp2base=home_T_tcp2base)
                
                if pick_success:
                    current_task = state.current_task
                    state.reset_state()
                    with state.lock:
                        state.current_task = current_task
                        state.task_phase = TaskPhase.PLACE
                        state.pick_stage = PickStage.IDLE
                        state.target_description = current_task['place']
                        _set_tracking_mode_locked(state, True)
                        _set_wrist_mode_locked(state, False)
                        state.verify_mode = False
                
                else:
                    if attemp_count + 1 >= config.MAX_PICK_RETRIES:
                        state.task_success = False
                        state.task_done.set()
                        continue
                    
                    else:
                        state.abort_execution.set()
                        state.plan_ready.clear()

                        with state.lock:
                            _reset_pick_tracking_locked(
                                state,
                                current_task["pick"],
                                attempt_count=attemp_count + 1,
                            )

            elif task_phase == TaskPhase.PLACE:
                
                place_success = do_check_place_success(state, rs_env, current_task['pick'], current_task['place'], point_2d=check_point_2d,cam_results=cam_results, home_T_tcp2base=home_T_tcp2base)
            
                if place_success:
                    state.reset_state()
                    with state.lock:
                        state.task_success = True
                        state.task_done.set()
                        continue
                else:
                    if attemp_count + 1 >= config.MAX_PLACE_RETRIES:
                        state.task_success = False
                        state.task_done.set()
                        continue
                    
                    else:
                        with state.lock:
                            _reset_pick_tracking_locked(
                                state,
                                current_task["pick"],
                                attempt_count=attemp_count + 1,
                            )

        else:
            time.sleep(0.01)

    print("[感知线程] 已停止")

# 移动检测
def detect_target_movement(state, current_xyz, task_phase):
    """基于当前打点和上一次打点的距离判断目标是否移动。"""
    config = state.config

    if task_phase == TaskPhase.PICK:
        dist_threshold = config.MOVE_OBJECT_THRESHOLD
    else:
        dist_threshold = config.MOVE_CONTAINER_THRESHOLD

    current_xyz = np.asarray(current_xyz, dtype=float)
    if not np.all(np.isfinite(current_xyz)):
        return False, 0.0

    previous_xyz = state.previous_point_3d
    if previous_xyz is None:
        state.previous_point_3d = current_xyz.copy()
        return False, 0.0

    previous_xyz = np.asarray(previous_xyz, dtype=float)
    if not np.all(np.isfinite(previous_xyz)):
        state.previous_point_3d = current_xyz.copy()
        return False, 0.0

    dist = float(np.linalg.norm(current_xyz - previous_xyz))
    state.previous_point_3d = current_xyz.copy()
    return dist > dist_threshold, dist


def _save_check_fail_image(image, check_type, object_name, save_dir):
    """检测失败时保存传入模型的图片。"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = object_name.replace(" ", "_").replace("/", "_")
    filename = f"{timestamp}_{check_type}_{safe_name}.png"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return save_path


# 结果检测
def do_check_pick_success(state, env, rs_env, pick_target, point_2d=None, cam_results=None, home_T_tcp2base=None):
    """
    通过两种方式进行自动化检测：
        1. 通过 VLM 检测抓取是否成功
        2. 计算抓取点与物体中心点的距离，如果距离小于阈值，则认为抓取成功
    """
    config = state.config

    if config.CHECK_PICK_SUCCESS_MODE == 1:
        print("[检测] 检查抓取是否成功...")

        # 1. 通过 VLM 检测抓取是否成功
        obs = rs_env.step()
        image_rgb = obs["rgb"]
        image_for_check = crop_image_around_point(
            image_rgb,
            point_2d,
            crop_size=config.CHECK_PICK_CROP_SIZE,
        )
        # save_check_image(image_for_check, prefix="pick", object_name=pick_name, save_dir=SAVE_DIR)

        is_success_1 = check_grasp_success_vllm(image_for_check, pick_target)

        # 2. 计算抓取点与物体中心点的距离
        # 物体 xyz 坐标
        object_2d = get_point_vllm(image_rgb,f"Point the {pick_target}",save_path=None)

        object_current_T = make_target_T(obs,int(object_2d[0]),int(object_2d[1]),rs_env,cam_results,home_T_tcp2base)
        object_xyzrpy = realman_xyzrpy_from_T(object_current_T)
        
        # 夹爪 xyz 坐标
        tcp_xyzrpy = env.get_state().pose

        dist = np.linalg.norm(object_xyzrpy[:3] - tcp_xyzrpy[:3])

        if dist < config.PICK_SUCCESS_DIST_THRESHOLD:
            is_success_2 = True
        else:
            is_success_2 = False
        
        if is_success_1 and is_success_2:
            print(f"VLM 检测抓取成功，距离检测抓取成功，抓取成功!")
            return True
        elif is_success_2:
            print(f"VLM 检测抓取失败，距离检测抓取成功，抓取成功!")
            return True
        else:
            print(f"VLM 检测抓取失败，距离检测抓取失败，抓取失败!")
            if getattr(config, 'SAVE_CHECK_FAIL_IMAGE', False):
                save_path = _save_check_fail_image(image_for_check, "pick_check", pick_target, config.SAVE_DIR)
                print(f"[检测] 失败图片已保存: {save_path}")
            return False

    elif config.CHECK_PICK_SUCCESS_MODE == 2:
        print("[检测] 跳过 pick 检测")
        return True
    
    else:
        print("[检测] 人工检测 pick 是否成功")
        while True:
            key = input("Pick 成功? (y/n): ")
            if key == 'y':
                return True
            elif key == 'n':
                return False

def do_check_place_success(state, rs_env, pick_target, place_target, point_2d=None,cam_results=None, home_T_tcp2base=None):
    """调用 VLM 检测 place 是否成功"""
    config = state.config

    if config.CHECK_PLACE_SUCCESS_MODE == 1:
        
        # 1. 通过 VLM 检测放置是否成功
        print("[检测] 检查放置是否成功...")
        obs = rs_env.step()
        image_rgb = obs["rgb"]
        image_for_check = crop_image_around_point(
            image_rgb,
            point_2d,
            crop_size=config.CHECK_PLACE_CROP_SIZE,
        )
        # save_check_image(image_for_check, prefix="place", object_name=pick_name, container_name=place_name, save_dir=SAVE_DIR)
        
        is_success_1 = check_place_success_vllm(image_for_check, pick_target, place_target)
        
        # 2. 计算物体与容器的距离
        object_2d = get_point_vllm(image_rgb,f"Point the {pick_target}",save_path=None)
        object_current_T = make_target_T(obs,int(object_2d[0]),int(object_2d[1]),rs_env,cam_results,home_T_tcp2base)
        object_xyzrpy = realman_xyzrpy_from_T(object_current_T)
            
        container_2d = get_point_vllm(image_rgb,f"Point the {place_target}",save_path=None)
        container_current_T = make_target_T(obs,int(container_2d[0]),int(container_2d[1]),rs_env,cam_results,home_T_tcp2base)
        container_xyzrpy = realman_xyzrpy_from_T(container_current_T) 

        dist = np.linalg.norm(object_xyzrpy[:3] - container_xyzrpy[:3])

        is_success_2 = True if dist < config.PLACE_SUCCESS_DIST_THRESHOLD else False
        
        if is_success_1 and is_success_2:
            print(f"VLM 检测放置成功，距离检测放置成功，放置成功!")
            return True
        elif is_success_2:
            print(f"VLM 检测放置失败，距离检测放置成功，放置成功!")
            return True
        else:
            print(f"VLM 检测放置失败，距离检测放置失败，放置失败!")
            if getattr(config, 'SAVE_CHECK_FAIL_IMAGE', False):
                save_path = _save_check_fail_image(image_for_check, "place_check", f"{pick_target}_to_{place_target}", config.SAVE_DIR)
                print(f"[检测] 失败图片已保存: {save_path}")
            return False

    elif config.CHECK_PLACE_SUCCESS_MODE == 2:
        print("[检测] 跳过 place 检测")
        return True

    else:
        print("[检测] 人工检测 place 是否成功")
        while True:
            key = input("Place 成功? (y/n): ").strip().lower()
            if key == 'y':
                return True
            elif key == 'n':
                return False


# ═══════════════════════════════════════════════════
# 规划线程
# ═══════════════════════════════════════════════════

def planning_thread(state, env, curobo_planner, home_T_tcp2base):
    """
    规划线程：
    1. 第三视角阶段生成粗预抓取或普通 place 动作序列；
    2. 腕部单次观测完成后，把 GraspGen 候选池提交给执行线程。
    """
    print("[规划线程] 已启动")

    while not state.stop_all.is_set():
        if state.wrist_result_ready.is_set():
            state.wrist_result_ready.clear()

            with state.lock:
                if state.task_phase != TaskPhase.PICK:
                    continue

                num_candidates = int(len(state.grasp_pose_pool_base))
                state.pick_stage = PickStage.GRASP_PLAN_READY
                state.grasp_plan_source = "graspgen" if num_candidates > 0 else "heuristic_fallback"

            print(f"[规划] 🤖 腕部抓取计划已就绪，候选数: {num_candidates}")

            # DEBUG_EXIT_AFTER_GRASP_CANDIDATES START
            # 临时调试钩子：输出候选抓取姿态后，优雅停止整个脚本，
            # 避免执行线程继续尝试抓取动作导致真实机器人碰撞。
            if bool(getattr(state.config, "DEBUG_EXIT_AFTER_GRASP_CANDIDATES", False)):
                with state.lock:
                    _print_grasp_candidate_summary(state)
                    state.task_success = False
                state.task_done.set()
                state.stop_all.set()
                print(
                    "[调试] DEBUG_EXIT_AFTER_GRASP_CANDIDATES=True，"
                    "已在抓取执行前停止脚本。恢复执行时把该配置改回 False。"
                )
                return
            # DEBUG_EXIT_AFTER_GRASP_CANDIDATES END

            with state.lock:
                state.grasp_plan_ready.set()
            continue

        triggered = state.need_replan.wait(timeout=0.05)
        if not triggered:
            continue

        state.need_replan.clear()

        with state.lock:
            if state.latest_target_T is None:
                continue
            task_phase = state.task_phase
            target_T = state.latest_target_T.copy()
            wrist_grasp_enabled = bool(state.wrist_grasp_enabled)

        try:
            if task_phase == TaskPhase.PICK and wrist_grasp_enabled:
                action_list, pick_pose_bundle = _build_pick_pregrasp_action_list(
                    state,
                    target_T,
                    home_T_tcp2base,
                )
            else:
                pick_pose_bundle = None
                action_list = build_action_list(
                    state,
                    env,
                    target_T,
                    home_T_tcp2base,
                    curobo_planner,
                    task_phase,
                )

            if action_list is None or len(action_list) == 0:
                print("[规划] ⚠️ 规划失败，重新规划")
                state.need_replan.set()
                continue

            state.abort_execution.set()
            while state.plan_ready.is_set() or state.grasp_plan_ready.is_set():
                time.sleep(0.01)

            with state.lock:
                state.action_list = action_list
                state.action_index = 0
                state.abort_execution.clear()
                state.plan_ready.set()

                if pick_pose_bundle is not None:
                    state.pregrasp_pose_base = pick_pose_bundle.pre_pick_T.copy()
                    state.fallback_pregrasp_pose_base = pick_pose_bundle.pre_pick_T.copy()
                    state.fallback_pick_pose_base = pick_pose_bundle.pick_T.copy()
                    state.post_pick_pose_base = pick_pose_bundle.post_pick_T.copy()
                    state.pick_stage = PickStage.PREGRASP_EXECUTING
                    _clear_grasp_outputs_locked(state)
                    print("[规划] 📐 预抓取规划完成，等待腕部单次观测")
                else:
                    print(f"[规划] 📐 规划完成，动作序列长度: {len(action_list)}")

        except Exception as e:
            print(f"[规划] 异常: {e}")


def build_action_list(state, env, target_T, home_T_tcp2base, curobo_planner, task_phase):
    """
    构建动作序列(完整的 pre_pick-pick-post_pick 或 pre_place-place-post_place)。

    Args:
        env: RealmanEnv 实例
        target_T: 目标物体的 4x4 位姿矩阵
        home_T_tcp2base: home 位姿矩阵（用于旋转参考）
        curobo_planner: curobo 规划器实例
        task_phase: TaskPhase.PICK 或 TaskPhase.PLACE
        
    Returns:
        action_list: [{"pose": np.array, "gripper": float, "tag": int}, ...]
        其中 tag=0 为 approach 动作, tag=1 为 target 动作, tag=2 为 post 动作
        None 表示规划失败
    """

    # TODO: 使用 pre_target_T 作为目标，调用 curobo 规划器生成 pre 段轨迹
    # trajectory = curobo_planner.plan(current_joint, pre_target_T)  

    config = _build_phase_config(state, task_phase)

    if task_phase == TaskPhase.PICK:
        pose_bundle = _build_pick_pose_bundle(state, target_T, home_T_tcp2base)
        target_pose = _matrix_to_tcp_pose(pose_bundle.pick_T)
        pre_target_pose = _matrix_to_tcp_pose(pose_bundle.pre_pick_T)
        post_target_pose = _matrix_to_tcp_pose(pose_bundle.post_pick_T)

        action_list = [
            {"pose": pre_target_pose, "gripper": config.GRIPPER_OPEN, "tag": 0, "motion": "pose", "wait_gripper": False},
            {"pose": target_pose, "gripper": config.GRIPPER_CLOSE, "tag": 1, "motion": "pose", "wait_gripper": True},
            {"pose": post_target_pose, "tag": 2, "motion": "pose"},
        ]
    
    else:
        # 对 place 位姿进行偏置
        target_T = make_lift_T(target_T, lift_x=config.PLACE_X_OFFSET, lift_y=config.PLACE_Y_OFFSET, lift_z=config.PLACE_Z_OFFSET)

        # 修改 rpy
        if config.PLACE_RPY is not None:
            print(
                "[PnP] use PLACE_RPY override:",
                np.round(np.asarray(config.PLACE_RPY, dtype=float), 4).tolist()
            )
            target_pose = realman_xyzrpy_from_T(target_T)
            target_pose[3:] = np.array(config.PLACE_RPY)
            target_T_new = T_from_realman_xyzrpy(target_pose)
        
        else:
            target_T_new = adjust_target_T(state, target_T, home_T_tcp2base)

        pre_target_T = make_lift_T(target_T_new, lift_x=config.PRE_PLACE_X_OFFSET,lift_y=config.PRE_PLACE_Y_OFFSET, lift_z=config.PRE_PLACE_Z_OFFSET)
        pre_target_pose = realman_xyzrpy_from_T(pre_target_T)

        target_pose = realman_xyzrpy_from_T(target_T_new)

        post_target_T = make_lift_T(target_T_new, lift_x=config.POST_PLACE_X_OFFSET,lift_y=config.POST_PLACE_Y_OFFSET, lift_z=config.POST_PLACE_Z_OFFSET)
        post_target_pose = realman_xyzrpy_from_T(post_target_T)

        action_list = [
            {"pose": pre_target_pose, "gripper": config.GRIPPER_CLOSE, "tag": 0, "motion": "pose", "wait_gripper": False},
            {"pose": target_pose, "gripper": config.GRIPPER_OPEN, "tag": 1, "motion": "pose", "wait_gripper": True},
            {"pose": post_target_pose, "tag": 2, "motion": "pose"},
        ]
    
    return action_list


def adjust_target_T(state, target_T, home_T_tcp2base):
    """
    根据物体位置计算合适的抓取姿态旋转矩阵。

    Args:
        target_T: 目标物体 4x4 位姿矩阵
        home_T_tcp2base: home 位姿矩阵

    Returns:
        带有正确旋转的抓取位姿 4x4 矩阵
    """

    config = state.config

    x, y, z = target_T[:3, 3]

    # 根据 y 值（距离）选择适当的俯仰角
    if y > -0.35:
        rx_degree = config.RX_DEGREE_CLOSE
    elif z > 0.12:
        rx_degree = config.RX_DEGREE_FAR_HIGH
    else:
        rx_degree = config.RX_DEGREE_FAR_LOW

    rx = -1 * (rx_degree / 180) * np.pi
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    grasp_T = copy.deepcopy(home_T_tcp2base)
    grasp_T[:3, :3] = Rx @ home_T_tcp2base[:3, :3]
    grasp_T[:3, 3] = target_T[:3, 3]

    return grasp_T


# ═══════════════════════════════════════════════════
# 执行线程
# ═══════════════════════════════════════════════════

def execution_thread(state, env):
    """
    执行线程：依次执行动作列表中的动作点。

    - 支持中断 abort_execution
    - 支持重新开始 plan_ready
    - 动作列表执行完毕
    - 连续运动失败达到 MAX_CONSECUTIVE_MOTION_FAILURES 时放弃本段轨迹并 need_replan
    """
    print("[执行线程] 已启动")

    while not state.stop_all.is_set():
        if state.grasp_plan_ready.is_set():
            state.grasp_plan_ready.clear()

            with state.lock:
                if state.task_phase != TaskPhase.PICK:
                    continue

                grasp_pose_pool = state.grasp_pose_pool_base.copy()
                grasp_pregrasp_pool = state.grasp_pregrasp_pool_base.copy()
                grasp_scores = state.grasp_pose_pool_scores.copy()
                state.pick_stage = PickStage.GRASP_EXECUTING

            executed = False
            used_fallback = False

            if len(grasp_pose_pool) > 0:
                for idx, grasp_T_base in enumerate(grasp_pose_pool):
                    if state.stop_all.is_set():
                        break

                    pregrasp_T_base = (
                        grasp_pregrasp_pool[idx]
                        if idx < len(grasp_pregrasp_pool)
                        else _build_candidate_pregrasp_from_grasp(state, grasp_T_base)
                    )
                    score = float(grasp_scores[idx]) if idx < len(grasp_scores) else None
                    print(
                        f"[执行] 🤖 尝试抓取候选 {idx + 1}/{len(grasp_pose_pool)}"
                        + (f"，score={score:.4f}" if score is not None else "")
                    )

                    try:
                        success, debug_info = try_execute_grasp_candidate(
                            state,
                            env,
                            grasp_T_base,
                            pregrasp_T_base,
                            score,
                        )
                    except RuntimeError as exc:
                        success = False
                        debug_info = {"stage": "motion", "error": str(exc)}

                    if success:
                        executed = True
                        with state.lock:
                            state.grasp_plan_source = "graspgen"
                            state.wrist_grasp_debug = {
                                **(state.wrist_grasp_debug or {}),
                                "selected_candidate_index": idx,
                                "selected_candidate_score": score,
                                "selected_execution_debug": debug_info,
                            }
                        print(f"[执行] ✅ 抓取候选 {idx + 1} 执行成功")
                        break

                    print(f"[执行] ⚠️ 抓取候选 {idx + 1} 执行失败: {debug_info}")

            if not executed and not state.stop_all.is_set():
                print("[执行] ↩️ GraspGen 候选不可用，切换到启发式兜底抓取")
                try:
                    executed, fallback_debug = execute_fallback_pick(state, env)
                except RuntimeError as exc:
                    executed = False
                    fallback_debug = {"stage": "fallback_motion", "error": str(exc)}
                used_fallback = executed
                with state.lock:
                    state.wrist_grasp_debug = {
                        **(state.wrist_grasp_debug or {}),
                        "fallback_debug": fallback_debug,
                    }

            if executed:
                with state.lock:
                    _set_tracking_mode_locked(state, False)
                    _set_wrist_mode_locked(state, False)
                    state.verify_mode = True
                    state.pick_stage = PickStage.VERIFYING
                    if used_fallback:
                        state.grasp_plan_source = "heuristic_fallback"
                print("[执行] ✅ 抓取段执行完成，进入抓取验证")
            else:
                with state.lock:
                    current_task = state.current_task or {}
                    pick_target = current_task.get("pick")
                    if pick_target is not None:
                        _reset_pick_tracking_locked(state, pick_target)
                print("[执行] ❌ 所有抓取候选与兜底方案均失败，回到全局重感知")

            continue

        triggered = state.plan_ready.wait(timeout=0.05)
        if not triggered:
            continue

        print("[执行] ▶️ 开始执行动作序列")
        outcome = _consume_action_list_execution(state, env)

        if outcome != "completed":
            continue

        with state.lock:
            task_phase = state.task_phase
            pick_stage = state.pick_stage
            wrist_grasp_enabled = bool(state.wrist_grasp_enabled)

        if task_phase == TaskPhase.PICK and wrist_grasp_enabled and pick_stage == PickStage.PREGRASP_EXECUTING:
            with state.lock:
                _set_tracking_mode_locked(state, False)
                _set_wrist_mode_locked(state, True)
                state.verify_mode = False
                state.pick_stage = PickStage.WRIST_SENSING
            print("[执行] ✅ 已到达粗预抓取位姿，切换到腕部单次感知")
            continue

        with state.lock:
            _set_tracking_mode_locked(state, False)
            _set_wrist_mode_locked(state, False)
            state.verify_mode = True
            if task_phase == TaskPhase.PICK:
                state.pick_stage = PickStage.VERIFYING
        print("[执行] ✅ 动作序列执行完成，进入结果验证")

    print("[执行线程] 已停止")

# ═══════════════════════════════════════════════════
# 主线程调度逻辑
# ═══════════════════════════════════════════════════

# 执行单个任务
def run_single_task(
    state,
    env,
    rs_env,
    cam_results,
    task,
    home_T_tcp2base,
    pick_profile=None,
    place_profile=None,
    special_pick_rpy=None,
    special_place_rpy=None,
):
    if state.stop_all.is_set():
        return False

    resolved_pick_profile = _normalize_motion_profile(pick_profile)
    resolved_place_profile = _normalize_motion_profile(place_profile)

    if special_pick_rpy is not None:
        resolved_pick_profile["PICK_RPY"] = np.asarray(
            special_pick_rpy,
            dtype=float,
        ).tolist()

    if special_place_rpy is not None:
        resolved_place_profile["PLACE_RPY"] = np.asarray(
            special_place_rpy,
            dtype=float,
        ).tolist()

    current_task = dict(task)

    if resolved_pick_profile:
        current_task["_pick_profile"] = resolved_pick_profile

    if resolved_place_profile:
        current_task["_place_profile"] = resolved_place_profile

    state.reset_state()

    with state.lock:
        state.current_task = current_task
        state.task_phase = TaskPhase.PICK
        state.pick_stage = PickStage.GLOBAL_TRACKING
        state.verify_mode = False
        _set_wrist_mode_locked(state, False)
        _set_tracking_mode_locked(state, True)
        state.target_description = task['pick']

    while not state.stop_all.is_set():
        if state.task_done.wait(timeout=0.1):
            break

    if not state.task_done.is_set():
        with state.lock:
            if state.current_task is not None:
                state.current_task.pop("_pick_profile", None)
                state.current_task.pop("_place_profile", None)
        return False

    with state.lock:
        if state.current_task is not None:
            state.current_task.pop("_pick_profile", None)
            state.current_task.pop("_place_profile", None)

    if state.task_success:
        return True
    else:
        return False

# 按照明确的动作列表执行所有任务
def run_all_tasks(state, env, rs_env, cam_results, task_list, home_T_tcp2base):
    if not task_list:
        print("[主线程] 未生成有效任务列表，等待下一轮检测...")
        return

    for i, task in enumerate(task_list):
        if state.stop_all.is_set():
            break

        print(f"\n{'='*60}")
        print(f"🚀 Task [{i+1}/{len(task_list)}]: pick={task['pick']} → place={task['place']}")
        print(f"{'='*60}")

        success = run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base)

        if state.stop_all.is_set():
            break

        if not success:
            print(f"⛔ Task [{i}] 失败，继续下一个任务。")
            continue

        # === 当前任务执行成功后的处理 ===
        if i + 1 < len(task_list):
            next_task = task_list[i + 1]

            # 设置下一个任务的感知目标，直接从 post-place 位置开始下一个任务
            with state.lock:
                state.task_phase = TaskPhase.PICK
                _set_tracking_mode_locked(state, True)
                state.verify_mode = False
                state.target_description = next_task['pick']
                state.is_first_point = True
                state.previous_point_3d = None

            print("[主线程] ➡️ 直接进入下一个任务（跳过 Reset）...")

    if state.stop_all.is_set():
        print("\n[主线程] 收到停止信号，结束当前任务循环。")
    else:
        print("\n🎉 所有任务完成!")

    print("[主线程] 🔄 机械臂 Reset...")
    env.reset()

# 按照模糊指令执行所有任务（一次只输出一组pnp目标）
def run_all_tasks_by_instruction(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    根据自然语言指令持续执行任务：
    1. 调用 VLM 从图像解析 pick/place 任务
    2. 执行单个任务
    3. 循环直到检测到任务完成
    """

    print(f"[主线程] 🧠 指令: {instruction}")

    config = state.config

    while not state.stop_all.is_set():

        # 获取当前图像
        obs = rs_env.step()
        image_rgb = obs["rgb"]

        try:
            # 获取一组 pnp 任务目标
            task = generate_task_from_scene(image_rgb, instruction)
            print(f"task: {task}")

            # 如果发现任务，则执行
            if task:
                run_single_task(state, env, rs_env, cam_results, task, home_T_tcp2base)
                env.reset()
                if state.stop_all.is_set():
                    break
            
            else:
                print("[主线程] 未发现可执行任务，等待...")
                time.sleep(config.TASK_DISCOVERY_INTERVAL)
                continue

        except Exception as e:
            print(f"[主线程] 异常: {e}")
            if state.stop_all.is_set():
                break

# 按照模糊指令持续完成任务（一次生成多组pnp目标），完成列表后持续监控
def run_all_tasks_by_instruction_with_list_and_monitor(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    根据自然语言指令持续执行任务：
    1. 调用 VLM 判断当前场景是否满足指令的要求，如果满足则定频检测，不满足则生成 pnp list。
    2. 按照list依次执行pnp任务，并在完成一组pnp任务后更新list（将已完成的pnp任务从list中移除，调整新放的和拿走的物体）
    """

    print(f"[主线程] 🧠 指令: {instruction}")

    config = state.config

    tasks_list = None

    while not state.stop_all.is_set():

        # 获取当前图像
        obs = rs_env.step()
        image_rgb = obs["rgb"]

        try:
            # 判断当前场景是否满足顶层指令的要求
            check_start = time.perf_counter()
            is_complete, reason = check_instruction_complete(image_rgb, instruction)
            check_elapsed = time.perf_counter() - check_start
            print(f"[主线程] 完成检测耗时: {check_elapsed:.2f}s")
            print(f"is_complete: {is_complete}, reason: {reason}")

            if is_complete:
                print("[主线程] 当前场景满足指令的要求，开始定频检测")
                time.sleep(config.TASK_DISCOVERY_INTERVAL)
                continue
            else:
                tasks_list = generate_tasks_from_scene(image_rgb, instruction)
                print(f"tasks_list: {tasks_list}")

                if not tasks_list:
                    print("[主线程] 未生成有效任务，等待下一轮检测...")
                    time.sleep(config.TASK_DISCOVERY_INTERVAL)
                    continue

                run_all_tasks(state, env, rs_env, cam_results, tasks_list, home_T_tcp2base)
                if state.stop_all.is_set():
                    break
        
        except Exception as e:
            print(f"[主线程] 异常: {e}")
            time.sleep(config.TASK_DISCOVERY_INTERVAL)
            if state.stop_all.is_set():
                break
            continue

# 按照模糊指令持续完成任务（一次生成多组pnp目标），完成列表后如果认为满足指令则停止
def run_all_tasks_by_instruction_with_list(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    根据自然语言指令持续执行任务：
    1. 调用 VLM 判断当前场景是否满足指令的要求，如果满足则定频检测，不满足则生成 pnp list。
    2. 按照list依次执行pnp任务，并在完成一组pnp任务后更新list（将已完成的pnp任务从list中移除，调整新放的和拿走的物体）
    """

    print(f"[主线程] 🧠 指令: {instruction}")

    config = state.config

    tasks_list = None

    while not state.stop_all.is_set():

        # 获取当前图像
        obs = rs_env.step()
        image_rgb = obs["rgb"]

        try:
            # 判断当前场景是否满足顶层指令的要求
            is_complete, reason = check_instruction_complete(image_rgb, instruction)
            print(f"is_complete: {is_complete}, reason: {reason}")

            if is_complete:
                print("[主线程] 当前场景满足指令的要求，停止执行")
                break
            else:
                tasks_list = generate_tasks_from_scene(image_rgb, instruction)
                print(f"tasks_list: {tasks_list}")

                if not tasks_list:
                    print("[主线程] 未生成有效任务，等待下一轮检测...")
                    time.sleep(config.TASK_DISCOVERY_INTERVAL)
                    continue

                run_all_tasks(state, env, rs_env, cam_results, tasks_list, home_T_tcp2base)
                if state.stop_all.is_set():
                    break
        
        except Exception as e:
            print(f"[主线程] 异常: {e}")
            time.sleep(config.TASK_DISCOVERY_INTERVAL)
            if state.stop_all.is_set():
                break
            continue

# 按照带方位描述的指令持续完成任务
def run_all_tasks_by_instruction_with_position_description_with_monitor(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    1. 调用 vlm 根据长指令拆解出多个 pick 和 place 目标, 目标不只是 object name,还包含方位描述。
    2. 按照目标依次执行pnp任务, 并在完成一组pnp任务后更新目标(将已完成的pnp任务从list中移除, 调整新放的和拿走的物体)
    """
    print(f"[主线程] 🧠 指令: {instruction}")

    config = state.config

    tasks_list = None

    while not state.stop_all.is_set():

        # 获取当前图像
        obs = rs_env.step()
        image_rgb = obs["rgb"]

        try:
            # 判断当前场景是否满足顶层指令的要求
            is_complete, reason = check_instruction_complete(image_rgb, instruction)
            print(f"is_complete: {is_complete}, reason: {reason}")

            if is_complete:
                print("[主线程] 当前场景满足指令的要求，停止执行")
                break
            else:
                tasks_list = generate_tasks_with_descriptions(image_rgb, instruction)
                print(f"tasks_list: {tasks_list}")

                if not tasks_list:
                    print("[主线程] 未生成有效任务，等待下一轮检测...")
                    time.sleep(config.TASK_DISCOVERY_INTERVAL)
                    continue

                run_all_tasks(state, env, rs_env, cam_results, tasks_list, home_T_tcp2base)
                if state.stop_all.is_set():
                    break
        
        except Exception as e:
            print(f"[主线程] 异常: {e}")
            time.sleep(config.TASK_DISCOVERY_INTERVAL)
            if state.stop_all.is_set():
                break
            continue


def run_all_tasks_by_instruction_with_position_description(state, env, rs_env, cam_results, instruction, home_T_tcp2base):
    """
    调用 vlm 根据指令生成 pnp 子任务并一次性执行，不循环检测是否完成。
    """
    print(f"[主线程] 🧠 指令: {instruction}")

    config = state.config

    if state.stop_all.is_set():
        return

    obs = rs_env.step()
    image_rgb = obs["rgb"]

    try:
        tasks_list = generate_tasks_with_descriptions(image_rgb, instruction)
        print(f"tasks_list: {tasks_list}")

        if not tasks_list:
            print("[主线程] 未生成有效任务。")
            return

        run_all_tasks(state, env, rs_env, cam_results, tasks_list, home_T_tcp2base)

    except Exception as e:
        print(f"[主线程] 异常: {e}")


# ═══════════════════════════════════════════════════
# 系统初始化
# ═══════════════════════════════════════════════════
# 机械臂环境初始化
def init_robot_env(robot_ip):
    env = RealmanEnv(robot_ip=robot_ip, mode="sync")
    env.reset()

    robot_state = env.get_state()
    home_T_tcp2base = T_from_realman_xyzrpy(robot_state.pose)

    return env, home_T_tcp2base

# 相机环境初始化
def init_camera_env(camera_serial, cam_results_path):
    rs_env = Open3dRealsenseEnv(camera_serial)

    with open(cam_results_path, "r") as f:
        cam_results = json.load(f)

    return rs_env, cam_results


def init_wrist_grasp_env(
    wrist_camera_serial,
    *,
    graspgen_host="127.0.0.1",
    graspgen_port=5556,
    graspgen_timeout_ms=60_000,
    handeye_calib_json=None,
    handeye_rotation=None,
    handeye_translation=None,
    handeye_frame="eef",
    wait_for_server=True,
):
    wrist_rs_env = Open3dRealsenseEnv(wrist_camera_serial)
    wrist_handeye_config = load_wrist_handeye_config(
        calib_json=handeye_calib_json,
        rotation=handeye_rotation,
        translation=handeye_translation,
        handeye_frame=handeye_frame,
    )
    graspgen_client = GraspGenClientBridge(
        host=graspgen_host,
        port=int(graspgen_port),
        timeout_ms=int(graspgen_timeout_ms),
        wait_for_server=wait_for_server,
    )
    return wrist_rs_env, graspgen_client, wrist_handeye_config

# 状态初始化
def init_state(config=None, task_config_path=None):
    resolved_config = load_pnp_config(task_config_path=task_config_path, task_config=config)
    return SharedState(resolved_config)

# 启动系统
def start_pnp_system(
    state,
    env,
    rs_env,
    cam_results,
    home_T_tcp2base,
    wrist_rs_env=None,
    graspgen_client=None,
    wrist_handeye_config=None,
):
    curobo_planner = None

    with state.lock:
        state.wrist_grasp_enabled = bool(
            wrist_rs_env is not None
            and graspgen_client is not None
            and wrist_handeye_config is not None
        )

    threads = [
        threading.Thread(
            target=perception_thread,
            args=(
                state,
                env,
                rs_env,
                cam_results,
                home_T_tcp2base,
                wrist_rs_env,
                graspgen_client,
                wrist_handeye_config,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=planning_thread,
            args=(state, env, curobo_planner, home_T_tcp2base),
            daemon=True,
        ),
        threading.Thread(
            target=execution_thread,
            args=(state, env),
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()

    print("✅ PnP 系统已启动")
    if state.wrist_grasp_enabled:
        print("✅ Wrist GraspGen 精抓取分支已启用")
    else:
        print("ℹ️ Wrist GraspGen 精抓取分支未启用，系统将只使用原始启发式抓取")

    return threads

# 关闭系统
def shutdown_pnp_system(state, env=None, rs_env=None, wrist_rs_env=None, graspgen_client=None):
    print("🛑 正在关闭系统...")
    state.stop_all.set()
    time.sleep(0.5)
    
    if env is not None:
        env.close()

    if rs_env is not None:
        rs_env.close()

    if wrist_rs_env is not None:
        wrist_rs_env.close()

    if graspgen_client is not None:
        graspgen_client.close()


# ═══════════════════════════════════════════════════
# pnp_skill 使用示例
# ═══════════════════════════════════════════════════

def main():
    
    # 左臂
    robot_ip = "192.168.101.19"
    camera_serial = "f1471338"
    cam_results_path = "/home/zhangzhao/lyt/camera/20260325_031804/camera_results.json"

    # 指令
    instruction = "Pick the baseball and place it on the right side of the rubic's cube."

    # 初始化资源
    env, home_T_tcp2base = init_robot_env(robot_ip)
    rs_env, cam_results = init_camera_env(camera_serial, cam_results_path)

    # 状态
    state = init_state()

    # 启动系统
    start_pnp_system(state, env, rs_env, cam_results, home_T_tcp2base)

    # 执行
    try:
        run_all_tasks_by_instruction_with_position_description(state, env, rs_env, cam_results, instruction, home_T_tcp2base)
    except KeyboardInterrupt:
        print("\n[停止] 收到键盘中断，正在停止...")
    except Exception as e:
        print(f"\n[错误] 未捕获异常: {e}")
        traceback.print_exc()
    finally:
        shutdown_pnp_system(state, env=env, rs_env=rs_env)


if __name__ == "__main__":
    main()
