import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo_new.skills.pnp_skill.pick_and_place import (
    init_state,
    start_pnp_system,
    run_single_task,
)

CONFIG2_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "task",
    "roast_sweet_potatoes",
    "config2.yaml",
)
CONFIG2_PATH = os.path.abspath(CONFIG2_PATH)


def place_chopsticks(env, rs_env, cam_results, home_T_tcp2base, wrist_runtime, pick_target, place_target):
    """
    使用 GraspGen 精抓取执行放筷子/餐具任务。
    创建独立的 pnp state（启用 GraspGen），复用已有的 wrist_runtime。
    """
    state = init_state(task_config_path=CONFIG2_PATH)

    start_pnp_system(
        state,
        env,
        rs_env,
        cam_results,
        home_T_tcp2base,
        wrist_rs_env=wrist_runtime.wrist_rs_env,
        graspgen_client=wrist_runtime.graspgen_client,
        wrist_handeye_config=wrist_runtime.wrist_handeye_config,
    )

    task = {
        "pick": pick_target,
        "place": place_target,
    }

    try:
        run_single_task(
            state=state,
            env=env,
            rs_env=rs_env,
            cam_results=cam_results,
            task=task,
            home_T_tcp2base=home_T_tcp2base,
        )
    finally:
        state.stop_all.set()
        time.sleep(0.3)

    env.reset()
