#!/usr/bin/env python3
"""Smoke test for VLABench sync vector and multi-task support."""

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
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
for path in (str(REPO_ROOT), str(VLABENCH_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rlinf.envs import get_env_cls  # noqa: E402


def make_cfg(num_envs: int = 2, max_episode_steps: int = 5):
    return SimpleNamespace(
        env_type="vlabench",
        task_name="select_fruit",
        task_names=["select_fruit", "select_drink"],
        robot="franka",
        num_envs=num_envs,
        total_num_envs=num_envs,
        group_size=1,
        seed=7,
        control_mode="ee",
        action_mode="absolute_ee",
        reward_mode="success",
        ee_frame_offset=[0.0, -0.4, 0.78],
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        auto_reset=False,
        max_episode_steps=max_episode_steps,
        max_steps_per_rollout_epoch=max_episode_steps,
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
        video_cfg=SimpleNamespace(save_video=False),
    )


def assert_obs(obs, batch: int):
    assert isinstance(obs, dict)
    assert set(["main_images", "states", "task_descriptions"]).issubset(obs)
    assert isinstance(obs["main_images"], torch.Tensor)
    assert isinstance(obs["states"], torch.Tensor)
    assert obs["main_images"].shape == (batch, 256, 256, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["states"].shape == (batch, 7)
    assert obs["states"].dtype == torch.float32
    assert isinstance(obs["task_descriptions"], list)
    assert len(obs["task_descriptions"]) == batch
    assert all(isinstance(item, str) for item in obs["task_descriptions"])


def assert_info(info, batch: int):
    assert isinstance(info, dict)
    for key in [
        "success",
        "success_once",
        "elapsed_steps",
        "episode_return",
        "task_name",
        "instruction",
        "ik_success",
    ]:
        assert key in info, key
    assert len(info["task_name"]) == batch
    assert len(info["instruction"]) == batch
    assert info["success"].shape == (batch,)
    assert info["success_once"].shape == (batch,)
    assert info["elapsed_steps"].shape == (batch,)
    assert info["episode_return"].shape == (batch,)
    assert info["ik_success"].shape == (batch,)


def main():
    batch = 2
    cfg = make_cfg(num_envs=batch, max_episode_steps=5)
    env_cls = get_env_cls(cfg.env_type, cfg)
    env = env_cls(cfg=cfg, num_envs=batch, seed_offset=0, total_num_processes=1)
    try:
        print("[1] reset sync vector env")
        obs, info = env.reset()
        assert_obs(obs, batch)
        assert_info(info, batch)
        assert len(set(info["task_name"])) >= 1
        print("tasks:", info["task_name"])

        print("[2] step action shape [B, 7]")
        actions = np.zeros((batch, 7), dtype=np.float32)
        actions[:, 2] = 0.05
        obs, reward, terminated, truncated, info = env.step(actions)
        assert_obs(obs, batch)
        assert reward.shape == (batch,)
        assert reward.dtype == np.float32
        assert terminated.shape == (batch,)
        assert truncated.shape == (batch,)
        assert_info(info, batch)
        assert info["elapsed_steps"].tolist() == [1, 1]
        print("step reward:", reward, "terminated:", terminated, "truncated:", truncated)

        print("[3] chunk_step action shape [B, T, 7]")
        chunk = torch.zeros((batch, 3, 7), dtype=torch.float32)
        chunk[:, :, 2] = 0.05
        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(chunk)
        assert len(obs_list) == 3
        assert len(infos_list) == 3
        assert rewards.shape == (batch, 3)
        assert terminations.shape == (batch, 3)
        assert truncations.shape == (batch, 3)
        assert rewards.dtype == torch.float32
        assert terminations.dtype == torch.bool
        assert truncations.dtype == torch.bool
        assert_obs(obs_list[-1], batch)
        assert_info(infos_list[-1], batch)
        assert isinstance(obs_list[-1]["task_descriptions"], list)
        print("chunk rewards:", rewards)
        print("chunk truncations:", truncations)

        print("[4] done latch after max_episode_steps")
        _, rewards2, terminations2, truncations2, infos2 = env.chunk_step(
            np.zeros((batch, 3, 7), dtype=np.float32)
        )
        assert rewards2.shape == (batch, 3)
        assert truncations2.any().item()
        elapsed = infos2[-1]["elapsed_steps"].clone()
        _, rewards3, terminations3, truncations3, infos3 = env.chunk_step(
            np.zeros((batch, 2, 7), dtype=np.float32)
        )
        assert rewards3.sum().item() == 0.0
        assert truncations3.all().item()
        assert torch.equal(infos3[-1]["elapsed_steps"], elapsed)
        assert_info(infos3[-1], batch)
        print("latched elapsed:", infos3[-1]["elapsed_steps"])

        print("[5] render first env")
        image = env.render()
        assert isinstance(image, np.ndarray)
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
    finally:
        env.close()

    print("VLABench sync vector smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
