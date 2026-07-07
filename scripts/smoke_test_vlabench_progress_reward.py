#!/usr/bin/env python3
"""Smoke tests for VLABench success_plus_progress_delta reward mode."""

from __future__ import annotations

import json
import os
import shutil
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
        task_names=None,
        robot="franka",
        num_envs=1,
        total_num_envs=1,
        group_size=1,
        seed=31,
        vector_mode="sync",
        control_mode="ee",
        action_mode="absolute_ee",
        reward_mode="success",
        success_reward=1.0,
        progress_reward_coef=0.5,
        progress_delta_negative=False,
        progress_delta_clip_min=0.0,
        progress_delta_clip_max=1.0,
        step_penalty=0.0,
        ik_failure_penalty=0.0,
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        auto_reset=False,
        max_episode_steps=3,
        max_steps_per_rollout_epoch=3,
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


def assert_reward_info(info, batch: int | None = None):
    required = [
        "reward_mode",
        "reward_success",
        "reward_progress",
        "reward_step_penalty",
        "reward_ik_penalty",
        "progress_delta",
        "progress_delta_clipped",
        "progress_available",
        "episode_progress_reward",
        "episode_success_reward",
        "episode_total_reward",
        "final_progress_score",
        "total_progress_delta",
    ]
    for key in required:
        assert key in info, key
    if batch is not None:
        assert info["reward_success"].shape == (batch,)
        assert info["reward_progress"].shape == (batch,)
        assert info["progress_delta"].shape == (batch,)
        assert info["progress_available"].shape == (batch,)
        assert len(info["reward_mode"]) == batch


def run_success_regression():
    print("[A] success reward regression")
    cfg = make_cfg(reward_mode="success")
    env = make_env(cfg)
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        assert reward in (0.0, 1.0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["reward_mode"] == "success"
        assert_reward_info(info)
        assert info["reward_progress"] == 0.0
        print("success reward:", reward)
    finally:
        env.close()


def run_single_progress():
    print("[B] progress reward single-env")
    cfg = make_cfg(reward_mode="success_plus_progress_delta", max_episode_steps=4)
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_reward_info(info)
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        assert isinstance(reward, float)
        assert_reward_info(info)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 7), dtype=np.float32)
        )
        assert rewards.shape == (1, 2)
        assert_reward_info(infos[-1])
        print("single rewards:", rewards)
    finally:
        env.close()


def run_vector(vector_mode: str):
    print(f"[C/D] progress reward {vector_mode} vector")
    cfg = make_cfg(
        reward_mode="success_plus_progress_delta",
        task_names=["select_fruit", "select_drink"],
        num_envs=2,
        total_num_envs=2,
        vector_mode=vector_mode,
        max_episode_steps=3,
    )
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_reward_info(info, batch=2)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            torch.zeros((2, 2, 7), dtype=torch.float32)
        )
        assert rewards.shape == (2, 2)
        assert terminations.shape == (2, 2)
        assert truncations.shape == (2, 2)
        assert_reward_info(infos[-1], batch=2)
        assert infos[-1]["reward_success"].shape == (2,)
        print(vector_mode, "rewards:", rewards)
    finally:
        env.close()


def run_export():
    print("[E] progress reward eval export")
    result_dir = Path("/tmp/vlabench_progress_reward_export")
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(
        reward_mode="success_plus_progress_delta",
        task_names=["select_fruit", "select_drink"],
        num_envs=2,
        total_num_envs=2,
        max_episode_steps=2,
        vlabench_eval=SimpleNamespace(
            export_results=True,
            result_path=str(result_dir / "vlabench_results.jsonl"),
            summary_path=str(result_dir / "vlabench_summary.json"),
            summary_csv_path=str(result_dir / "vlabench_summary.csv"),
            export_format="jsonl",
        ),
    )
    env = make_env(cfg)
    try:
        env.reset()
        _, rewards, _, truncations, infos = env.chunk_step(np.zeros((2, 3, 7), dtype=np.float32))
        assert rewards.shape == (2, 3)
        assert truncations[:, 1:].all().item()
        result_path = result_dir / "vlabench_results.jsonl"
        summary_path = result_dir / "vlabench_summary.json"
        csv_path = result_dir / "vlabench_summary.csv"
        assert result_path.exists()
        assert summary_path.exists()
        assert csv_path.exists()
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        assert len(rows) >= 2
        for row in rows:
            for key in [
                "episode_progress_reward",
                "episode_success_reward",
                "final_progress_score",
                "total_progress_delta",
                "reward_mode",
            ]:
                assert key in row, key
            assert row["reward_mode"] == "success_plus_progress_delta"
        summary = json.loads(summary_path.read_text())
        for key in [
            "avg_episode_progress_reward",
            "avg_episode_success_reward",
            "avg_final_progress_score",
        ]:
            assert key in summary["overall"], key
        assert "avg_episode_progress_reward" in csv_path.read_text().splitlines()[0]
        print("export rows:", len(rows), "summary:", summary["overall"])
    finally:
        env.close()


def main():
    run_success_regression()
    run_single_progress()
    run_vector("sync")
    run_vector("subprocess")
    run_export()
    print("VLABench progress reward smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
