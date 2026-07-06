# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy fallback
    import gym

from rlinf.envs.vlabench.utils import (
    DEFAULT_EE_FRAME_OFFSET,
    ee_action_to_ctrl,
    ensure_vlabench_importable,
    get_cfg_value,
    validate_mvp_config,
    wrap_observation,
)

__all__ = ["VLABenchEnv"]


class VLABenchEnv(gym.Env):
    """MVP Gym-style wrapper for a single VLABench dm_control environment.

    MVP limits:
    - num_envs == 1
    - control_mode == "ee"
    - action_mode == "absolute_ee"
    - reward_mode == "success"
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        cfg,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info=None,
        **_,
    ):
        super().__init__()
        validate_mvp_config(cfg)
        if num_envs != 1:
            raise NotImplementedError("VLABenchEnv MVP only supports num_envs=1")

        ensure_vlabench_importable()
        from VLABench.envs import load_env

        self.cfg = cfg
        self.num_envs = 1
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = int(get_cfg_value(cfg, "seed", 0)) + seed_offset

        self.task_name = get_cfg_value(cfg, "task_name")
        self.robot = get_cfg_value(cfg, "robot", "franka")
        self.ignore_terminations = bool(get_cfg_value(cfg, "ignore_terminations", False))
        self.auto_reset = bool(get_cfg_value(cfg, "auto_reset", False))
        if self.auto_reset:
            raise NotImplementedError("VLABenchEnv Phase-2A does not support auto_reset")
        self.max_episode_steps = int(get_cfg_value(cfg, "max_episode_steps", 80))
        self.require_pcd = bool(get_cfg_value(cfg, "require_pcd", False))
        self.return_tensors = bool(get_cfg_value(cfg, "return_tensors", False))
        self.ee_frame_offset = np.asarray(
            get_cfg_value(cfg, "ee_frame_offset", DEFAULT_EE_FRAME_OFFSET),
            dtype=np.float32,
        )
        self.gripper_open_threshold = float(get_cfg_value(cfg, "gripper_open_threshold", 0.1))
        self.gripper_open_value = float(get_cfg_value(cfg, "gripper_open_value", 0.04))
        self.render_height = int(get_cfg_value(cfg, "render_height", 256))
        self.render_width = int(get_cfg_value(cfg, "render_width", 256))
        self.camera_id = int(get_cfg_value(cfg, "camera_id", 2))
        self.reset_wait_step = int(get_cfg_value(cfg, "reset_wait_step", 10))
        self.random_init = bool(get_cfg_value(cfg, "random_init", True))

        self.env = load_env(
            self.task_name,
            robot=self.robot,
            reset_wait_step=self.reset_wait_step,
            random_init=self.random_init,
        )
        self.env.render_options = {"height": self.render_height, "width": self.render_width}

        self.elapsed_steps = 0
        self.episode_return = 0.0
        self.success_once = False
        self.last_raw_obs = None
        self.last_obs = None
        self.last_info = None
        self._episode_done = False
        self._last_done_obs = None
        self._last_done_info = None
        self._last_done_termination = False
        self._last_done_truncation = False

        ncam = int(self.env.physics.model.ncam)
        extra_cams = max(ncam - 1, 0) if bool(get_cfg_value(cfg, "use_extra_views", True)) else 0
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "main_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, self.render_height, self.render_width, 3),
                    dtype=np.uint8,
                ),
                "extra_view_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, extra_cams, self.render_height, self.render_width, 3),
                    dtype=np.uint8,
                ),
                "states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 7),
                    dtype=np.float32,
                ),
                "task_descriptions": gym.spaces.Sequence(gym.spaces.Text(max_length=1024)),
            }
        )

    @property
    def instruction(self) -> str:
        return self.env.task.get_instruction() or ""

    @property
    def device(self):
        return "cpu"

    @property
    def total_num_group_envs(self):
        return 1

    def _get_info(self, *, success: bool, ik_success: Optional[bool] = None) -> dict:
        return {
            "task_name": self.task_name,
            "instruction": self.instruction,
            "success": bool(success),
            "success_once": bool(self.success_once),
            "ik_success": ik_success,
            "elapsed_steps": int(self.elapsed_steps),
            "episode_return": float(self.episode_return),
        }

    def _format_obs(self, obs: dict) -> dict:
        if not self.return_tensors:
            return obs

        formatted = dict(obs)
        for key in ("main_images", "extra_view_images", "states"):
            if formatted.get(key, None) is not None and not isinstance(
                formatted[key], torch.Tensor
            ):
                formatted[key] = torch.as_tensor(formatted[key], device="cpu").contiguous()
        return formatted

    def _get_wrapped_observation(self):
        raw_obs = self.env.get_observation(require_pcd=self.require_pcd)
        obs = wrap_observation(raw_obs, self.instruction, self.cfg)
        self.last_raw_obs = raw_obs
        self.last_obs = obs
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        if options:
            unsupported = sorted(options)
            raise NotImplementedError(
                f"VLABenchEnv MVP does not support reset options: {unsupported}"
            )

        self.env.reset()
        self.elapsed_steps = 0
        self.episode_return = 0.0
        self.success_once = False
        self._episode_done = False
        self._last_done_obs = None
        self._last_done_info = None
        self._last_done_termination = False
        self._last_done_truncation = False

        obs = self._get_wrapped_observation()
        info = self._get_info(success=False, ik_success=None)
        self.last_info = info
        return self._format_obs(obs), info

    def step(self, action):
        if self._episode_done:
            if self._last_done_obs is None or self._last_done_info is None:
                raise RuntimeError("VLABenchEnv is done but terminal observation is missing")
            return (
                self._last_done_obs,
                0.0,
                self._last_done_termination,
                self._last_done_truncation,
                self._last_done_info,
            )

        ctrl_action, ik_success = ee_action_to_ctrl(
            self.env,
            action,
            ee_frame_offset=self.ee_frame_offset,
            gripper_open_threshold=self.gripper_open_threshold,
            gripper_open_value=self.gripper_open_value,
        )
        self.env.step(ctrl_action)
        self.elapsed_steps += 1

        success = bool(self.env.task.should_terminate_episode(self.env.physics))
        reward = 1.0 if success else 0.0
        self.episode_return += reward
        self.success_once = bool(self.success_once or success)

        terminated = bool(success and not self.ignore_terminations)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps)

        obs = self._get_wrapped_observation()
        formatted_obs = self._format_obs(obs)
        info = self._get_info(success=success, ik_success=ik_success)
        self.last_info = info
        if terminated or truncated:
            self._episode_done = True
            self._last_done_obs = formatted_obs
            self._last_done_info = info
            self._last_done_termination = terminated
            self._last_done_truncation = truncated
        return formatted_obs, reward, terminated, truncated, info

    def _normalize_chunk_actions(self, chunk_actions) -> np.ndarray:
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if actions.shape == (7,):
            actions = actions.reshape(1, 1, 7)
        elif actions.shape == (1, 7):
            actions = actions.reshape(1, 1, 7)
        elif actions.ndim == 2 and actions.shape[-1] == 7:
            actions = actions.reshape(1, actions.shape[0], 7)
        elif actions.ndim == 3 and actions.shape[0] == 1 and actions.shape[-1] == 7:
            pass
        else:
            raise ValueError(
                "VLABenchEnv.chunk_step expects [7], [1, 7], [T, 7], or [1, T, 7], "
                f"got {actions.shape}"
            )
        return np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32,
            copy=False,
        )

    def chunk_step(self, chunk_actions):
        actions = self._normalize_chunk_actions(chunk_actions)
        if actions.shape[0] != 1:
            raise NotImplementedError("VLABenchEnv MVP only supports num_envs=1")

        chunk_size = actions.shape[1]
        obs_list = []
        infos_list = []
        rewards = []
        terminations = []
        truncations = []

        stopped = False
        last_obs = None
        last_info = None
        last_terminated = False
        last_truncated = False

        for step_idx in range(chunk_size):
            if not stopped:
                assert actions.dtype == np.float32
                obs, reward, terminated, truncated, info = self.step(actions[0, step_idx])
                last_obs = obs
                last_info = info
                last_terminated = bool(terminated)
                last_truncated = bool(truncated)
                stopped = last_terminated or last_truncated
            else:
                obs = last_obs
                reward = 0.0
                terminated = last_terminated
                truncated = last_truncated
                info = last_info

            obs_list.append(obs)
            infos_list.append(info)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            terminations.append(torch.tensor([terminated], dtype=torch.bool))
            truncations.append(torch.tensor([truncated], dtype=torch.bool))

        chunk_rewards = torch.stack(rewards, dim=1)
        chunk_terminations = torch.stack(terminations, dim=1)
        chunk_truncations = torch.stack(truncations, dim=1)
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def update_reset_state_ids(self):
        return None

    def render(self, info=None, rew=None):
        return self.env.render(
            camera_id=self.camera_id,
            height=self.render_height,
            width=self.render_width,
        )

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
