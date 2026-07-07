#!/usr/bin/env python3
"""Smoke tests for VLABench delta_ee action mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
VLABENCH_REPO = WORKSPACE_ROOT / "VLABench"
VLABENCH_ROOT = VLABENCH_REPO / "VLABench"

os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_ROOT))
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
for path in (str(REPO_ROOT), str(VLABENCH_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rlinf.envs import get_env_cls  # noqa: E402


def make_cfg(**overrides):
    cfg = dict(
        env_type="vlabench",
        task_name="select_fruit",
        robot="franka",
        num_envs=1,
        total_num_envs=1,
        group_size=1,
        seed=23,
        vector_mode="sync",
        control_mode="ee",
        action_mode="delta_ee",
        reward_mode="success",
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
        delta_position_scale=1.0,
        delta_rotation_scale=1.0,
        delta_position_clip=0.05,
        delta_rotation_clip=0.25,
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        auto_reset=False,
        max_episode_steps=5,
        max_steps_per_rollout_epoch=5,
        require_pcd=False,
        return_tensors=True,
        camera_id=2,
        use_extra_views=True,
        render_height=256,
        render_width=256,
        reset_wait_step=10,
        random_init=True,
        eval_track=None,
        episode_config_path=None,
        intention_score_threshold=0.1,
        video_cfg=SimpleNamespace(save_video=False),
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def make_env(cfg):
    env_cls = get_env_cls(cfg.env_type, cfg)
    return env_cls(cfg=cfg, num_envs=cfg.num_envs, seed_offset=0, total_num_processes=1)


def assert_obs(obs, batch: int):
    assert isinstance(obs, dict)
    assert isinstance(obs["main_images"], torch.Tensor)
    assert obs["main_images"].shape == (batch, 256, 256, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["states"].shape == (batch, 7)
    assert obs["states"].dtype == torch.float32
    assert isinstance(obs["task_descriptions"], list)
    assert len(obs["task_descriptions"]) == batch
    assert all(isinstance(text, str) for text in obs["task_descriptions"])


def assert_info(info, batch: int):
    assert isinstance(info, dict)
    for key in [
        "task_name",
        "instruction",
        "success",
        "success_once",
        "episode_return",
        "elapsed_steps",
        "terminated",
        "truncated",
        "ik_success",
    ]:
        assert key in info, key
    if batch == 1 and isinstance(info["task_name"], str):
        assert isinstance(info["instruction"], str)
        assert isinstance(info["success"], bool)
        assert isinstance(info["elapsed_steps"], int)
    else:
        assert len(info["task_name"]) == batch
        assert len(info["instruction"]) == batch
        assert info["success"].shape == (batch,)
        assert info["elapsed_steps"].shape == (batch,)
        assert info["ik_success"].shape == (batch,)


def run_single_env():
    print("[A] single-env delta_ee")
    cfg = make_cfg(num_envs=1, total_num_envs=1)
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 1)
        assert_info(info, 1)

        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert_obs(obs, 1)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert_info(info, 1)

        chunk = np.zeros((3, 7), dtype=np.float32)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(chunk)
        assert len(obs_list) == 3
        assert rewards.shape == (1, 3)
        assert terminations.shape == (1, 3)
        assert truncations.shape == (1, 3)
        assert_obs(obs_list[-1], 1)
        assert_info(infos[-1], 1)
        print("single elapsed:", infos[-1]["elapsed_steps"])
    finally:
        env.close()


def run_sync_vector():
    print("[B] sync vector delta_ee")
    cfg = make_cfg(
        num_envs=2,
        total_num_envs=2,
        task_names=["select_fruit", "select_drink"],
        task_sample_mode="sequential",
    )
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 2)
        assert_info(info, 2)
        assert info["task_name"] == ["select_fruit", "select_drink"]

        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 7), dtype=np.float32))
        assert_obs(obs, 2)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert_info(info, 2)

        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            torch.zeros((2, 2, 7), dtype=torch.float32)
        )
        assert len(obs_list) == 2
        assert rewards.shape == (2, 2)
        assert terminations.shape == (2, 2)
        assert truncations.shape == (2, 2)
        assert_obs(obs_list[-1], 2)
        assert_info(infos[-1], 2)
        print("sync tasks:", infos[-1]["task_name"])
    finally:
        env.close()


def run_subprocess_vector():
    print("[C] subprocess vector delta_ee")
    cfg = make_cfg(
        num_envs=2,
        total_num_envs=2,
        vector_mode="subprocess",
        task_names=["select_fruit", "select_drink"],
        task_sample_mode="sequential",
    )
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 2)
        assert_info(info, 2)

        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 7), dtype=np.float32)
        )
        assert len(obs_list) == 1
        assert rewards.shape == (2, 1)
        assert terminations.shape == (2, 1)
        assert truncations.shape == (2, 1)
        assert_info(infos[-1], 2)

        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            torch.zeros((2, 2, 7), dtype=torch.float32)
        )
        assert len(obs_list) == 2
        assert rewards.shape == (2, 2)
        assert terminations.shape == (2, 2)
        assert truncations.shape == (2, 2)
        assert_obs(obs_list[-1], 2)
        assert_info(infos[-1], 2)
        print("subprocess tasks:", infos[-1]["task_name"])
    finally:
        env.close()
        env.close()
        for process in env._subproc_processes:
            assert not process.is_alive(), f"worker still alive pid={process.pid}"


def main():
    run_single_env()
    run_sync_vector()
    run_subprocess_vector()
    print("VLABench delta_ee smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
