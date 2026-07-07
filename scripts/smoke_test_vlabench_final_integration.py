#!/usr/bin/env python3
"""Final integration smoke for RLinf VLABench Gym-style wrapper."""

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
from rlinf.envs.vlabench.utils import get_joint_control_dims  # noqa: E402


def make_cfg(**overrides):
    cfg = dict(
        env_type="vlabench",
        task_name="select_fruit",
        task_names=None,
        robot="franka",
        num_envs=1,
        total_num_envs=1,
        group_size=1,
        seed=41,
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
        delta_position_scale=1.0,
        delta_rotation_scale=1.0,
        delta_position_clip=0.05,
        delta_rotation_clip=0.25,
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


def assert_obs(obs, batch: int, *, tensors: bool):
    assert isinstance(obs, dict)
    for key in ["main_images", "states", "task_descriptions"]:
        assert key in obs, key
    assert isinstance(obs["task_descriptions"], list)
    assert len(obs["task_descriptions"]) == batch
    assert all(isinstance(item, str) for item in obs["task_descriptions"])
    if tensors:
        assert isinstance(obs["main_images"], torch.Tensor)
        assert isinstance(obs["states"], torch.Tensor)
        assert obs["main_images"].shape[0] == batch
        assert obs["states"].shape == (batch, 7)
        assert obs["main_images"].dtype == torch.uint8
        assert obs["states"].dtype == torch.float32
    else:
        assert isinstance(obs["main_images"], np.ndarray)
        assert isinstance(obs["states"], np.ndarray)
        assert obs["main_images"].shape[0] == batch
        assert obs["states"].shape == (batch, 7)
        assert obs["main_images"].dtype == np.uint8
        assert obs["states"].dtype == np.float32


def assert_info(info, batch: int | None = None):
    required = [
        "task_name",
        "instruction",
        "success",
        "success_once",
        "elapsed_steps",
        "episode_return",
        "ik_success",
        "reward_mode",
        "reward_success",
        "reward_progress",
        "progress_delta",
        "progress_available",
    ]
    for key in required:
        assert key in info, key
    if batch is not None:
        assert len(info["task_name"]) == batch
        assert len(info["instruction"]) == batch
        assert info["success"].shape == (batch,)
        assert info["elapsed_steps"].shape == (batch,)
        assert info["reward_success"].shape == (batch,)


def current_joint_action(env, gripper=0.0):
    qpos = np.asarray(env.envs[0].robot.get_qpos(env.envs[0].physics), dtype=np.float32).reshape(-1)
    return np.concatenate([qpos, np.asarray([gripper], dtype=np.float32)]).astype(np.float32)


def run_single_gym():
    print("[A] Gym-style single env")
    for return_tensors in (False, True):
        cfg = make_cfg(num_envs=1, total_num_envs=1, return_tensors=return_tensors)
        env = make_env(cfg)
        try:
            assert env.action_space.shape == (7,)
            assert "main_images" in env.observation_space.spaces
            obs, info = env.reset()
            assert_obs(obs, 1, tensors=return_tensors)
            assert_info(info)
            out = env.step(np.zeros(7, dtype=np.float32))
            assert isinstance(out, tuple) and len(out) == 5
            obs, reward, terminated, truncated, info = out
            assert_obs(obs, 1, tensors=return_tensors)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert_info(info)
        finally:
            env.close()


def run_sync_vector():
    print("[B] Gym-style sync vector env")
    cfg = make_cfg(num_envs=2, total_num_envs=2, task_names=["select_fruit", "select_drink"])
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 2, tensors=True)
        assert_info(info, batch=2)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 2, 7), dtype=np.float32)
        )
        assert len(obs_list) == 2
        assert rewards.shape == (2, 2)
        assert terminations.shape == (2, 2)
        assert truncations.shape == (2, 2)
        assert_info(infos[-1], batch=2)
    finally:
        env.close()


def run_subprocess_vector():
    print("[C] Gym-style subprocess vector env")
    cfg = make_cfg(
        num_envs=2,
        total_num_envs=2,
        vector_mode="subprocess",
        task_names=["select_fruit", "select_drink"],
    )
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 2, tensors=True)
        assert_info(info, batch=2)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 2, 7), dtype=np.float32)
        )
        assert rewards.shape == (2, 2)
        assert_info(infos[-1], batch=2)
    finally:
        env.close()
        env.close()
        for process in env._subproc_processes:
            assert not process.is_alive(), f"worker still alive pid={process.pid}"


def run_action_modes():
    print("[D] action mode coverage")
    cases = [
        make_cfg(control_mode="ee", action_mode="absolute_ee"),
        make_cfg(control_mode="ee", action_mode="delta_ee"),
        make_cfg(control_mode="joint", action_mode="absolute_joint", joint_action_dim=8),
    ]
    for cfg in cases:
        env = make_env(cfg)
        try:
            obs, info = env.reset()
            if cfg.control_mode == "joint":
                qpos_dim, ctrl_dim, gripper_dim = get_joint_control_dims(env.envs[0])
                assert qpos_dim + 1 == env.policy_action_dim == cfg.joint_action_dim
                assert ctrl_dim == qpos_dim + gripper_dim
                action = current_joint_action(env)
            else:
                action = np.zeros(7, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert_info(info)
            obs_list, rewards, terminations, truncations, infos = env.chunk_step(action.reshape(1, -1))
            assert rewards.shape == (1, 1)
            assert_info(infos[-1])
            print("  ", cfg.control_mode, cfg.action_mode, "ok")
        finally:
            env.close()


def run_reward_modes():
    print("[E] reward mode coverage")
    for reward_mode in ("success", "success_plus_progress_delta"):
        cfg = make_cfg(reward_mode=reward_mode)
        env = make_env(cfg)
        try:
            env.reset()
            _, reward, _, _, info = env.step(np.zeros(7, dtype=np.float32))
            assert info["reward_mode"] == reward_mode
            assert "reward_success" in info and "reward_progress" in info
            assert isinstance(reward, float)
        finally:
            env.close()


def run_eval_export():
    print("[F] eval/export coverage")
    result_dir = Path("/tmp/vlabench_final_integration_export")
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(
        num_envs=2,
        total_num_envs=2,
        task_names=["select_fruit", "select_drink"],
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
    finally:
        env.close()
    result_path = result_dir / "vlabench_results.jsonl"
    summary_path = result_dir / "vlabench_summary.json"
    csv_path = result_dir / "vlabench_summary.csv"
    assert result_path.exists()
    assert summary_path.exists()
    assert csv_path.exists()
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) >= 2
    summary = json.loads(summary_path.read_text())
    assert "overall" in summary and "tasks" in summary
    assert csv_path.read_text().splitlines()[0].startswith("task_name,num_episodes")


def run_runner_config_audit():
    print("[G] runner config audit")
    required = [
        "examples/embodiment/config/vlabench_eval_export_smoke.yaml",
        "examples/embodiment/config/vlabench_delta_ee_smoke.yaml",
        "examples/embodiment/config/vlabench_joint_control_smoke.yaml",
        "examples/embodiment/config/vlabench_progress_reward_smoke.yaml",
    ]
    for item in required:
        path = REPO_ROOT / item
        assert path.exists(), path
        print("  exists", item)


def main():
    run_single_gym()
    run_sync_vector()
    run_subprocess_vector()
    run_action_modes()
    run_reward_modes()
    run_eval_export()
    run_runner_config_audit()
    print("VLABench final integration smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
