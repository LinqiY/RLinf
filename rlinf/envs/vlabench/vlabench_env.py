# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from typing import Optional

import numpy as np

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
        self.max_episode_steps = int(get_cfg_value(cfg, "max_episode_steps", 80))
        self.require_pcd = bool(get_cfg_value(cfg, "require_pcd", False))
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

        obs = self._get_wrapped_observation()
        info = self._get_info(success=False, ik_success=None)
        self.last_info = info
        return obs, info

    def step(self, action):
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
        info = self._get_info(success=success, ik_success=ik_success)
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def render(self, info=None, rew=None):
        return self.env.render(
            camera_id=self.camera_id,
            height=self.render_height,
            width=self.render_width,
        )

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
