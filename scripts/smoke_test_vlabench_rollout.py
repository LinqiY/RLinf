#!/usr/bin/env python3
"""Smoke test for the VLABench EnvWorker-compatible rollout path.

This intentionally does not launch the full distributed RLinf worker graph. It
checks the same local path EnvWorker uses for embodied env interaction:
get_env_cls -> reset -> prepare_actions -> chunk_step -> EnvOutput.to_dict.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
VLABENCH_REPO = WORKSPACE_ROOT / "VLABench"
VLABENCH_ROOT = VLABENCH_REPO / "VLABench"

os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_ROOT))
# Keep dm_control's OpenGL backend consistent even if the parent shell still has
# MUJOCO_GL=egl from another run. This environment has been validated with OSMesa.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
for path in (str(REPO_ROOT), str(VLABENCH_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rlinf.data.embodied_io_struct import EnvOutput  # noqa: E402
from rlinf.envs import get_env_cls  # noqa: E402
from rlinf.envs.action_utils import prepare_actions  # noqa: E402


def make_cfg(max_episode_steps: int = 5):
    return SimpleNamespace(
        env_type="vlabench",
        task_name="select_fruit",
        robot="franka",
        num_envs=1,
        total_num_envs=1,
        group_size=1,
        seed=0,
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
        video_cfg=SimpleNamespace(save_video=False),
    )


def assert_obs(obs):
    assert isinstance(obs, dict)
    assert "main_images" in obs and "states" in obs and "task_descriptions" in obs
    assert isinstance(obs["main_images"], torch.Tensor)
    assert isinstance(obs["states"], torch.Tensor)
    assert obs["main_images"].shape == (1, 256, 256, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["states"].shape == (1, 7)
    assert obs["states"].dtype == torch.float32
    assert isinstance(obs["task_descriptions"], list)


def main():
    cfg = make_cfg(max_episode_steps=5)

    print("[1] Creating env through RLinf get_env_cls(env_type='vlabench')")
    env_cls = get_env_cls(cfg.env_type, cfg)
    env = env_cls(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    try:
        print("[2] EnvWorker-style reset")
        obs, info = env.reset()
        assert_obs(obs)
        print("reset info:", info)

        bootstrap = EnvOutput(
            obs=obs,
            dones=torch.zeros((1, 2), dtype=torch.bool),
            terminations=torch.zeros((1, 2), dtype=torch.bool),
            truncations=torch.zeros((1, 2), dtype=torch.bool),
        ).to_dict()
        assert_obs(bootstrap["obs"])

        print("[3] prepare_actions -> VLABench chunk shape")
        raw_actions = torch.zeros((1, 3, 7), dtype=torch.float32)
        raw_actions[0, :, 2] = 0.05
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=cfg.env_type,
            model_type="mlp_policy",
            num_action_chunks=3,
            action_dim=7,
        )
        assert chunk_actions.shape == (1, 3, 7)

        print("[4] EnvWorker-style chunk_step")
        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
            chunk_actions
        )
        assert len(obs_list) == 3 and len(infos_list) == 3
        assert_obs(obs_list[-1])
        assert rewards.shape == (1, 3)
        assert terminations.shape == (1, 3)
        assert truncations.shape == (1, 3)
        assert "success" in infos_list[-1]
        assert "ik_success" in infos_list[-1]
        assert "elapsed_steps" in infos_list[-1]
        print("chunk rewards:", rewards)
        print("chunk terminations:", terminations)
        print("chunk truncations:", truncations)
        print("last info:", infos_list[-1])

        env_output = EnvOutput(
            obs=obs_list[-1],
            rewards=rewards,
            dones=torch.logical_or(terminations, truncations),
            terminations=terminations,
            truncations=truncations,
            env_infos=infos_list[-1],
        ).to_dict()
        assert_obs(env_output["obs"])
        assert env_output["rewards"].shape == (1, 3)

        print("[5] max_episode_steps=5 termination by truncation")
        next_actions = torch.zeros((1, 3, 7), dtype=torch.float32)
        _, rewards2, terminations2, truncations2, infos2 = env.chunk_step(next_actions)
        assert rewards2.shape == (1, 3)
        assert terminations2.shape == (1, 3)
        assert truncations2.shape == (1, 3)
        assert truncations2.any().item()
        assert infos2[-1]["elapsed_steps"] == 5
        print("second chunk rewards:", rewards2)
        print("second chunk terminations:", terminations2)
        print("second chunk truncations:", truncations2)
        print("final info:", infos2[-1])

        print("[6] EnvWorker finish_rollout no-op hook")
        env.update_reset_state_ids()
    finally:
        env.close()

    print("VLABench rollout smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
