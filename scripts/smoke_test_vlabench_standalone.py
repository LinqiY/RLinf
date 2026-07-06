#!/usr/bin/env python3
"""Smoke test for the RLinf VLABench MVP wrapper.

Expected environment:
  export VLABENCH_ROOT=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench/VLABench
  export PYTHONPATH=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench:$PYTHONPATH
  export MUJOCO_GL=egl
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
VLABENCH_REPO = WORKSPACE_ROOT / "VLABench"
VLABENCH_ROOT = VLABENCH_REPO / "VLABench"

os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
for path in (str(REPO_ROOT), str(VLABENCH_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rlinf.envs.vlabench import VLABenchEnv  # noqa: E402
from rlinf.envs.vlabench.utils import ee_action_to_ctrl, ee_state_to_policy_state  # noqa: E402


def make_cfg():
    return SimpleNamespace(
        env_type="vlabench",
        task_name="select_fruit",
        robot="franka",
        num_envs=1,
        control_mode="ee",
        action_mode="absolute_ee",
        reward_mode="success",
        ee_frame_offset=[0.0, -0.4, 0.78],
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        max_episode_steps=80,
        require_pcd=False,
        camera_id=2,
        use_extra_views=True,
        render_height=256,
        render_width=256,
        reset_wait_step=10,
        random_init=True,
        seed=0,
    )


def main():
    print("[1] Importing native VLABench")
    importlib.import_module("VLABench.robots")
    importlib.import_module("VLABench.tasks")
    from VLABench.envs import load_env

    print("[2] Creating native VLABench env")
    native_env = load_env("select_fruit", robot="franka", reset_wait_step=10, random_init=True)
    try:
        print("[3] Native reset")
        native_env.reset()
        print("[4] Native get_observation(require_pcd=False)")
        raw_obs = native_env.get_observation(require_pcd=False)
        assert "rgb" in raw_obs and "ee_state" in raw_obs
        instruction = native_env.task.get_instruction()
        print("rgb shape:", raw_obs["rgb"].shape)
        print("ee_state shape:", np.asarray(raw_obs["ee_state"]).shape)
        print("instruction:", instruction)
        print("nu:", native_env.physics.model.nu)

        print("[5] Convert 7D EE action to native ctrl")
        cfg = make_cfg()
        action = ee_state_to_policy_state(raw_obs, cfg.ee_frame_offset)
        ctrl_action, ik_success = ee_action_to_ctrl(
            native_env,
            action,
            ee_frame_offset=cfg.ee_frame_offset,
            gripper_open_threshold=cfg.gripper_open_threshold,
            gripper_open_value=cfg.gripper_open_value,
        )
        print("ctrl shape:", ctrl_action.shape, "ik_success:", ik_success)
        print("[6] Native env.step(ctrl_action)")
        native_env.step(ctrl_action)
    finally:
        native_env.close()

    print("[7] Creating RLinf VLABenchEnv")
    env = VLABenchEnv(make_cfg(), num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        print("[8] VLABenchEnv.reset()")
        obs, info = env.reset()
        assert "main_images" in obs and "states" in obs and "task_descriptions" in obs
        print("main_images:", obs["main_images"].shape, obs["main_images"].dtype)
        if obs["extra_view_images"] is not None:
            print("extra_view_images:", obs["extra_view_images"].shape, obs["extra_view_images"].dtype)
        print("states:", obs["states"].shape, obs["states"].dtype)
        print("info:", info)

        print("[9] VLABenchEnv.step(np.zeros((1, 7), dtype=np.float32))")
        result = env.step(np.zeros((1, 7), dtype=np.float32))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert "main_images" in obs and "states" in obs and "task_descriptions" in obs
        print("reward:", reward, "terminated:", terminated, "truncated:", truncated)
        print("info:", info)
        assert "success" in info and "ik_success" in info and "elapsed_steps" in info

        print("[10] VLABenchEnv.render()")
        image = env.render()
        assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8
        print("render:", image.shape, image.dtype)
    finally:
        env.close()

    print("Smoke test passed", flush=True)
    # VLABench.robots currently triggers a native-library abort during Python
    # interpreter teardown in this environment. The smoke path has completed by
    # this point, so exit without running those third-party destructors.
    os._exit(0)


if __name__ == "__main__":
    main()
