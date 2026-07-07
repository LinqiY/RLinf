#!/usr/bin/env python3
"""Smoke tests for VLABench subprocess vector mode."""

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
        num_envs=2,
        total_num_envs=2,
        group_size=1,
        seed=19,
        vector_mode="subprocess",
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


def make_env(cfg):
    env_cls = get_env_cls(cfg.env_type, cfg)
    return env_cls(cfg=cfg, num_envs=cfg.num_envs, seed_offset=0, total_num_processes=1)


def assert_obs(obs, batch):
    assert isinstance(obs, dict)
    assert isinstance(obs["main_images"], torch.Tensor)
    assert obs["main_images"].shape == (batch, 256, 256, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["states"].shape == (batch, 7)
    assert obs["states"].dtype == torch.float32
    assert isinstance(obs["task_descriptions"], list)
    assert len(obs["task_descriptions"]) == batch
    assert all(isinstance(text, str) for text in obs["task_descriptions"])


def assert_info(info, batch):
    assert isinstance(info, dict)
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


def assert_closed(env):
    env.close()
    env.close()
    for process in env._subproc_processes:
        assert not process.is_alive(), f"worker still alive pid={process.pid}"


def run_basic_and_sampling():
    print("[A/B] subprocess basic + task sampling")
    cfg = make_cfg(task_names=["select_fruit", "select_drink"])
    env = make_env(cfg)
    try:
        assert len(env._subproc_processes) == 2
        obs, info = env.reset()
        assert_obs(obs, 2)
        assert_info(info, 2)
        assert info["task_name"] == ["select_fruit", "select_drink"]
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 7), dtype=np.float32)
        )
        assert len(obs_list) == 1
        assert rewards.shape == (2, 1)
        assert terminations.shape == (2, 1)
        assert truncations.shape == (2, 1)
        assert_info(infos[-1], 2)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            torch.zeros((2, 3, 7), dtype=torch.float32)
        )
        assert len(obs_list) == 3
        assert rewards.shape == (2, 3)
        assert terminations.shape == (2, 3)
        assert truncations.shape == (2, 3)
        assert_info(infos[-1], 2)
        image = env.render(mode="rgb_array", tile=True)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3
        print("tasks:", infos[-1]["task_name"], "tile:", image.shape)
    finally:
        assert_closed(env)


def run_eval_track():
    print("[C] subprocess eval_track")
    cfg = make_cfg(task_name=None, task_names=None, eval_track="track_1_in_distribution")
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 2)
        assert_info(info, 2)
        assert all(item is not None for item in info["episode_config_id"])
        _, rewards, _, _, infos = env.chunk_step(np.zeros((2, 2, 7), dtype=np.float32))
        assert rewards.shape == (2, 2)
        assert_info(infos[-1], 2)
        print("eval tasks:", infos[-1]["task_name"])
        print("episode ids:", infos[-1]["episode_config_id"])
    finally:
        assert_closed(env)


def run_done_latch_close():
    print("[D] subprocess done latch + close")
    cfg = make_cfg(task_names=["select_fruit", "select_drink"], max_episode_steps=2)
    env = make_env(cfg)
    try:
        env.reset()
        _, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 4, 7), dtype=np.float32)
        )
        assert rewards.shape == (2, 4)
        assert truncations[:, 1:].all().item()
        elapsed = infos[-1]["elapsed_steps"].clone()
        _, rewards2, _, truncations2, infos2 = env.chunk_step(np.zeros((2, 2, 7), dtype=np.float32))
        assert rewards2.sum().item() == 0.0
        assert truncations2.all().item()
        assert torch.equal(infos2[-1]["elapsed_steps"], elapsed)
    finally:
        assert_closed(env)


def main():
    run_basic_and_sampling()
    run_eval_track()
    run_done_latch_close()
    print("VLABench subprocess smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
