#!/usr/bin/env python3
"""Smoke tests for VLABench joint control (control_mode=joint, action_mode=absolute_joint)."""

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
from rlinf.envs.vlabench.utils import (  # noqa: E402
    get_joint_control_dims,
    joint_action_to_ctrl,
)


def make_cfg(**overrides):
    cfg = dict(
        env_type="vlabench",
        task_name="select_fruit",
        robot="franka",
        num_envs=1,
        total_num_envs=1,
        group_size=1,
        seed=31,
        vector_mode="sync",
        control_mode="joint",
        action_mode="absolute_joint",
        reward_mode="success",
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
        joint_action_dim=None,
        joint_position_low=None,
        joint_position_high=None,
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
        assert isinstance(info["success"], bool)
        assert isinstance(info["elapsed_steps"], int)
    else:
        assert len(info["task_name"]) == batch
        assert info["success"].shape == (batch,)
        assert info["elapsed_steps"].shape == (batch,)


def run_dimension_discovery():
    print("[A] joint control dimension discovery")
    cfg = make_cfg(num_envs=1, total_num_envs=1)
    env = make_env(cfg)
    try:
        env.reset()
        qpos_dim, ctrl_dim, gripper_ctrl_dim = get_joint_control_dims(env.envs[0])
        policy_action_dim = env.policy_action_dim
        print(
            f"qpos_dim={qpos_dim} gripper_ctrl_dim={gripper_ctrl_dim} "
            f"ctrl_dim={ctrl_dim} policy_action_dim={policy_action_dim}"
        )
        assert ctrl_dim == qpos_dim + gripper_ctrl_dim
        assert policy_action_dim == qpos_dim + 1
        if qpos_dim == 7 and gripper_ctrl_dim == 2:
            print("Franka dims match expected qpos_dim=7, gripper_ctrl_dim=2, ctrl_dim=9, policy_action_dim=8")
        else:
            print(
                "NOTE: dims differ from the usual Franka expectation "
                f"(qpos_dim=7, gripper_ctrl_dim=2, ctrl_dim=9) -- got qpos_dim={qpos_dim}, "
                f"gripper_ctrl_dim={gripper_ctrl_dim}, ctrl_dim={ctrl_dim}"
            )
        return qpos_dim
    finally:
        env.close()


def run_single_env(qpos_dim: int):
    print("[B] single-env joint smoke")
    cfg = make_cfg(num_envs=1, total_num_envs=1)
    env = make_env(cfg)
    try:
        obs, info = env.reset()
        assert_obs(obs, 1)
        assert_info(info, 1)

        current_qpos = np.asarray(env.envs[0].robot.get_qpos(env.envs[0].physics), dtype=np.float32).reshape(-1)
        assert current_qpos.shape == (qpos_dim,)
        action = np.concatenate([current_qpos, np.asarray([1.0], dtype=np.float32)]).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert_obs(obs, 1)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert_info(info, 1)

        chunk = np.tile(action, (3, 1))
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


def run_sync_vector(qpos_dim: int):
    print("[C] sync vector joint smoke")
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

        actions = np.stack(
            [
                np.concatenate(
                    [
                        np.asarray(env.envs[i].robot.get_qpos(env.envs[i].physics), dtype=np.float32).reshape(-1),
                        np.asarray([1.0], dtype=np.float32),
                    ]
                )
                for i in range(2)
            ],
            axis=0,
        )
        assert actions.shape == (2, qpos_dim + 1)
        obs, reward, terminated, truncated, info = env.step(actions)
        assert_obs(obs, 2)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert_info(info, 2)

        chunk_actions = np.tile(actions[:, None, :], (1, 2, 1))
        assert chunk_actions.shape == (2, 2, qpos_dim + 1)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            torch.as_tensor(chunk_actions, dtype=torch.float32)
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


def run_subprocess_vector(qpos_dim: int):
    print("[D] subprocess joint smoke")
    joint_action_dim = qpos_dim + 1
    cfg = make_cfg(
        num_envs=2,
        total_num_envs=2,
        vector_mode="subprocess",
        joint_action_dim=joint_action_dim,
        task_names=["select_fruit", "select_drink"],
        task_sample_mode="sequential",
    )
    env = make_env(cfg)
    try:
        assert env.policy_action_dim == joint_action_dim
        obs, info = env.reset()
        assert_obs(obs, 2)
        assert_info(info, 2)

        actions = np.zeros((2, joint_action_dim), dtype=np.float32)
        actions[:, -1] = 1.0
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(actions)
        assert len(obs_list) == 1
        assert rewards.shape == (2, 1)
        assert terminations.shape == (2, 1)
        assert truncations.shape == (2, 1)
        assert_info(infos[-1], 2)

        chunk_actions = np.zeros((2, 2, joint_action_dim), dtype=np.float32)
        chunk_actions[..., -1] = 1.0
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            torch.as_tensor(chunk_actions, dtype=torch.float32)
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
    print("subprocess workers cleanly terminated, no residual processes")


def run_gripper_adapter_smoke(qpos_dim: int):
    print("[E] gripper adapter smoke")
    cfg = make_cfg(num_envs=1, total_num_envs=1)
    env = make_env(cfg)
    try:
        env.reset()
        raw_env = env.envs[0]
        current_qpos = np.asarray(raw_env.robot.get_qpos(raw_env.physics), dtype=np.float32).reshape(-1)
        assert current_qpos.shape == (qpos_dim,)

        open_action = np.concatenate([current_qpos, np.asarray([1.0], dtype=np.float32)]).astype(np.float32)
        close_action = np.concatenate([current_qpos, np.asarray([-1.0], dtype=np.float32)]).astype(np.float32)

        open_ctrl, open_ok = joint_action_to_ctrl(
            raw_env,
            open_action,
            gripper_open_threshold=env.gripper_open_threshold,
            gripper_open_value=env.gripper_open_value,
            joint_position_low=env.joint_position_low,
            joint_position_high=env.joint_position_high,
        )
        close_ctrl, close_ok = joint_action_to_ctrl(
            raw_env,
            close_action,
            gripper_open_threshold=env.gripper_open_threshold,
            gripper_open_value=env.gripper_open_value,
            joint_position_low=env.joint_position_low,
            joint_position_high=env.joint_position_high,
        )
        _, _, gripper_ctrl_dim = get_joint_control_dims(raw_env)

        assert open_ok and close_ok
        open_gripper_state = open_ctrl[qpos_dim:]
        close_gripper_state = close_ctrl[qpos_dim:]
        assert open_gripper_state.shape == (gripper_ctrl_dim,)
        assert close_gripper_state.shape == (gripper_ctrl_dim,)
        assert np.allclose(open_gripper_state, env.gripper_open_value)
        assert np.allclose(close_gripper_state, 0.0)
        print(
            f"gripper adapter ok: gripper_ctrl_dim={gripper_ctrl_dim} "
            f"open_value={open_gripper_state.tolist()} close_value={close_gripper_state.tolist()}"
        )
    finally:
        env.close()


def main():
    qpos_dim = run_dimension_discovery()
    run_single_env(qpos_dim)
    run_sync_vector(qpos_dim)
    run_subprocess_vector(qpos_dim)
    run_gripper_adapter_smoke(qpos_dim)
    print("VLABench joint control smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
