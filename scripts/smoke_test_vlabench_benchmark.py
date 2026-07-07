#!/usr/bin/env python3
"""Benchmark-style smoke tests for VLABench RLinf integration."""

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
TRACK_PATH = VLABENCH_ROOT / "configs" / "evaluation" / "tracks" / "track_1_in_distribution.json"

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
        seed=11,
        control_mode="ee",
        action_mode="absolute_ee",
        reward_mode="success",
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
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
        intention_score_threshold=0.1,
        video_cfg=SimpleNamespace(save_video=False),
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def assert_obs(obs, batch):
    assert isinstance(obs, dict)
    assert obs["main_images"].shape[0] == batch
    assert obs["main_images"].dtype == torch.uint8
    assert obs["states"].shape == (batch, 7)
    assert obs["states"].dtype == torch.float32
    assert isinstance(obs["task_descriptions"], list)
    assert len(obs["task_descriptions"]) == batch
    assert all(isinstance(text, str) for text in obs["task_descriptions"])


def assert_scalar_info(info):
    for key in [
        "task_name",
        "instruction",
        "episode_config_id",
        "success",
        "success_once",
        "episode_return",
        "elapsed_steps",
        "terminated",
        "truncated",
        "ik_success",
    ]:
        assert key in info, key
    assert isinstance(info["task_name"], str)
    assert isinstance(info["instruction"], str)


def assert_batched_info(info, batch):
    for key in [
        "task_name",
        "instruction",
        "episode_config_id",
        "success",
        "success_once",
        "episode_return",
        "elapsed_steps",
        "terminated",
        "truncated",
        "ik_success",
    ]:
        assert key in info, key
    assert len(info["task_name"]) == batch
    assert len(info["instruction"]) == batch
    assert len(info["episode_config_id"]) == batch
    assert info["success"].shape == (batch,)
    assert info["elapsed_steps"].shape == (batch,)


def make_env(cfg):
    env_cls = get_env_cls(cfg.env_type, cfg)
    return env_cls(cfg=cfg, num_envs=cfg.num_envs, seed_offset=0, total_num_processes=1)


def run_eval_track_smoke():
    print("[A] eval_track config smoke")
    cfg = make_cfg(eval_track="track_1_in_distribution", task_name=None, num_envs=1, total_num_envs=1)
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 1)
        assert_scalar_info(info)
        assert info["episode_config_id"] is not None
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((1, 2, 7), dtype=np.float32)
        )
        assert len(obs_list) == 2
        assert rewards.shape == (1, 2)
        assert_scalar_info(infos[-1])
        print("eval_track task:", infos[-1]["task_name"], infos[-1]["episode_config_id"])
    finally:
        env.close()


def run_episode_config_path_smoke():
    print("[B] episode_config_path smoke")
    cfg = make_cfg(episode_config_path=str(TRACK_PATH), task_name=None, num_envs=1, total_num_envs=1)
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 1)
        assert_scalar_info(info)
        assert str(TRACK_PATH) in info["episode_config_id"]
        _, rewards, _, _, infos = env.chunk_step(torch.zeros((1, 2, 7), dtype=torch.float32))
        assert rewards.shape == (1, 2)
        assert_scalar_info(infos[-1])
        print("episode_config_path task:", infos[-1]["task_name"], infos[-1]["episode_config_id"])
    finally:
        env.close()


def run_vector_metrics_and_tile_smoke():
    print("[C/D] vector metrics + tile render smoke")
    cfg = make_cfg(
        task_names=["select_fruit", "select_drink"],
        num_envs=2,
        total_num_envs=2,
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
    )
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 2)
        assert_batched_info(info, 2)
        assert info["task_name"] == ["select_fruit", "select_drink"]
        _, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 2, 7), dtype=np.float32)
        )
        assert rewards.shape == (2, 2)
        assert terminations.shape == (2, 2)
        assert truncations.shape == (2, 2)
        assert_batched_info(infos[-1], 2)
        assert "progress_score" not in infos[-1] or len(infos[-1]["progress_score"]) == 2
        assert "intention_score" not in infos[-1] or len(infos[-1]["intention_score"]) == 2
        image = env.render(mode="rgb_array", tile=True)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.shape[0] >= 256 and image.shape[1] >= 256
        print("vector tasks:", infos[-1]["task_name"])
        print("tile shape:", image.shape)
    finally:
        env.close()


def main():
    assert TRACK_PATH.exists(), TRACK_PATH
    run_eval_track_smoke()
    run_episode_config_path_smoke()
    run_vector_metrics_and_tile_smoke()
    print("VLABench benchmark smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
