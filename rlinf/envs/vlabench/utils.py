# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf

DEFAULT_EE_FRAME_OFFSET = np.array([0.0, -0.4, 0.78], dtype=np.float32)


def ensure_vlabench_importable():
    """Import VLABench and register its task/robot classes."""
    root = os.environ.get("VLABENCH_ROOT")
    if root:
        package_parent = os.path.dirname(root.rstrip(os.sep))
        if package_parent and package_parent not in sys.path:
            sys.path.insert(0, package_parent)

    try:
        importlib.import_module("VLABench.envs")
        importlib.import_module("VLABench.robots")
        importlib.import_module("VLABench.tasks")
    except ImportError as exc:
        raise ImportError(
            "Failed to import VLABench. Set VLABENCH_ROOT to "
            ".../VLABench/VLABench and add .../VLABench to PYTHONPATH."
        ) from exc


def get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def validate_mvp_config(cfg: Any) -> None:
    env_type = get_cfg_value(cfg, "env_type")
    if env_type != "vlabench":
        raise ValueError(f"VLABenchEnv requires env_type='vlabench', got {env_type!r}")

    task_name = get_cfg_value(cfg, "task_name")
    task_names = get_cfg_value(cfg, "task_names", None)
    eval_track = get_cfg_value(cfg, "eval_track", None)
    episode_config_path = get_cfg_value(cfg, "episode_config_path", None)
    if not task_name and not task_names and not eval_track and not episode_config_path:
        raise ValueError("VLABenchEnv requires task_name, task_names, eval_track, or episode_config_path")

    task_sample_mode = get_cfg_value(cfg, "task_sample_mode", "uniform")
    if task_sample_mode not in ("uniform", "sequential"):
        raise ValueError("VLABench task_sample_mode must be 'uniform' or 'sequential'")

    episode_config_sample_mode = get_cfg_value(cfg, "episode_config_sample_mode", "sequential")
    if episode_config_sample_mode not in ("random", "sequential"):
        raise ValueError("VLABench episode_config_sample_mode must be 'random' or 'sequential'")

    control_mode = get_cfg_value(cfg, "control_mode", "ee")
    if control_mode != "ee":
        raise NotImplementedError("VLABenchEnv MVP only supports control_mode='ee'")

    action_mode = get_cfg_value(cfg, "action_mode", "absolute_ee")
    if action_mode != "absolute_ee":
        raise NotImplementedError("VLABenchEnv MVP only supports action_mode='absolute_ee'")

    reward_mode = get_cfg_value(cfg, "reward_mode", "success")
    if reward_mode != "success":
        raise NotImplementedError("VLABenchEnv MVP only supports reward_mode='success'")

    ee_frame_offset = get_cfg_value(cfg, "ee_frame_offset", DEFAULT_EE_FRAME_OFFSET)
    if len(ee_frame_offset) != 3:
        raise ValueError("ee_frame_offset must contain exactly 3 values")

    max_episode_steps = int(get_cfg_value(cfg, "max_episode_steps", 80))
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be > 0")

    if bool(get_cfg_value(cfg, "require_pcd", False)):
        raise NotImplementedError("VLABenchEnv MVP requires require_pcd=false")


def cfg_to_container(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def normalize_task_names(cfg: Any, episode_configs: Any = None) -> list[str]:
    if isinstance(episode_configs, dict) and episode_configs:
        return [str(name) for name in episode_configs.keys()]

    task_names = cfg_to_container(get_cfg_value(cfg, "task_names", None))
    if task_names is None:
        task_name = get_cfg_value(cfg, "task_name", None)
        return [str(task_name)] if task_name else []
    if isinstance(task_names, str):
        task_names = [task_names]
    names = [str(name) for name in task_names if str(name)]
    if not names:
        raise ValueError("VLABench task_names must contain at least one task")
    return names


def resolve_episode_config_path(cfg: Any) -> str | None:
    path = get_cfg_value(cfg, "episode_config_path", None)
    if path:
        return str(path)

    eval_track = get_cfg_value(cfg, "eval_track", None)
    if not eval_track:
        return None
    eval_track = str(eval_track)
    if os.path.exists(eval_track):
        return eval_track

    root = os.environ.get("VLABENCH_ROOT")
    if root:
        candidate = os.path.join(
            root,
            "configs",
            "evaluation",
            "tracks",
            eval_track if eval_track.endswith(".json") else f"{eval_track}.json",
        )
        if os.path.exists(candidate):
            return candidate
    return None


def load_episode_configs(cfg: Any) -> tuple[list[dict] | dict[str, Any] | None, str | None]:
    path = resolve_episode_config_path(cfg)
    if path is None:
        if get_cfg_value(cfg, "eval_track", None):
            raise FileNotFoundError(
                "Could not resolve VLABench eval_track. Expected "
                "$VLABENCH_ROOT/configs/evaluation/tracks/{eval_track}.json or an explicit path."
            )
        return None, None

    if str(path).endswith(".json"):
        with open(path, "r") as f:
            container = json.load(f)
    else:
        loaded = OmegaConf.load(path)
        container = cfg_to_container(loaded)
    if isinstance(container, (list, dict)):
        return container, path
    raise ValueError(f"Unsupported VLABench episode config format at {path!r}")


def get_episode_candidates(episode_configs: Any, task_name: str):
    if episode_configs is None:
        return None
    if isinstance(episode_configs, list):
        return episode_configs
    if isinstance(episode_configs, dict):
        task_configs = episode_configs.get(task_name)
        if task_configs is None:
            return None
        if isinstance(task_configs, list):
            return task_configs
        return [task_configs]
    return None


def normalize_ee_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape == (1, 7):
        action = action[0]
    if action.shape != (7,):
        raise ValueError(f"VLABench EE action must have shape (7,) or (1, 7), got {action.shape}")
    if not np.all(np.isfinite(action)):
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return action.astype(np.float32, copy=False)


def make_gripper_state(gripper: float, threshold: float, open_value: float) -> np.ndarray:
    if gripper >= threshold:
        return np.ones(2, dtype=np.float32) * np.float32(open_value)
    return np.zeros(2, dtype=np.float32)


def current_noop_ctrl(env, gripper_open_value: float) -> np.ndarray:
    qpos = np.asarray(env.robot.get_qpos(env.physics), dtype=np.float32)
    gripper = np.ones(2, dtype=np.float32) * np.float32(gripper_open_value)
    return np.concatenate([qpos, gripper]).astype(np.float32)


def ee_action_to_ctrl(
    env,
    action: np.ndarray,
    *,
    ee_frame_offset,
    gripper_open_threshold: float,
    gripper_open_value: float,
) -> tuple[np.ndarray, bool]:
    """Convert OneTwoVLA-style 7D EE action to VLABench MuJoCo ctrl."""
    action = normalize_ee_action(action)
    target_pos = action[:3].copy() + np.asarray(ee_frame_offset, dtype=np.float32)
    target_euler = action[3:6].copy()
    gripper = float(action[6])

    from VLABench.utils.utils import euler_to_quaternion

    quat = euler_to_quaternion(*target_euler)
    gripper_state = make_gripper_state(
        gripper,
        threshold=gripper_open_threshold,
        open_value=gripper_open_value,
    )

    try:
        ik_result = env.robot.get_qpos_from_ee_pos(
            physics=env.physics,
            pos=target_pos,
            quat=quat,
        )
        if isinstance(ik_result, tuple):
            ik_success = bool(ik_result[0])
            qpos_action = ik_result[1]
        else:
            ik_success = True
            qpos_action = ik_result
        if not ik_success:
            raise ValueError("IK did not find a valid solution")
        qpos_action = np.asarray(qpos_action, dtype=np.float32).reshape(-1)
        if qpos_action.size == 0 or not np.all(np.isfinite(qpos_action)):
            raise ValueError("IK produced an invalid qpos action")
    except Exception:
        ik_success = False
        qpos_action = np.asarray(env.robot.get_qpos(env.physics), dtype=np.float32).reshape(-1)

    ctrl_action = np.concatenate([qpos_action, gripper_state]).astype(np.float32)
    return ctrl_action, ik_success


def ee_state_to_policy_state(raw_obs: dict, ee_frame_offset) -> np.ndarray:
    ee_state = np.asarray(raw_obs["ee_state"], dtype=np.float32).reshape(-1)
    pos = ee_state[:3].copy() - np.asarray(ee_frame_offset, dtype=np.float32)
    quat = ee_state[3:7].copy()
    gripper = np.asarray([ee_state[-1]], dtype=np.float32)

    from VLABench.utils.utils import quaternion_to_euler

    euler = np.asarray(quaternion_to_euler(quat), dtype=np.float32)
    return np.concatenate([pos, euler, gripper]).astype(np.float32)


def wrap_observation(raw_obs: dict, instruction: str, cfg: Any) -> dict:
    rgb = np.asarray(raw_obs["rgb"])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    camera_id = int(get_cfg_value(cfg, "camera_id", 2))
    if camera_id < 0 or camera_id >= rgb.shape[0]:
        raise ValueError(f"camera_id {camera_id} is out of range for rgb with {rgb.shape[0]} cameras")

    main_images = rgb[camera_id][None]
    extra_view_images = None
    if bool(get_cfg_value(cfg, "use_extra_views", True)):
        extra = [rgb[i] for i in range(rgb.shape[0]) if i != camera_id]
        if extra:
            extra_view_images = np.stack(extra, axis=0)[None].astype(np.uint8)

    states = ee_state_to_policy_state(
        raw_obs,
        get_cfg_value(cfg, "ee_frame_offset", DEFAULT_EE_FRAME_OFFSET),
    )[None]

    return {
        "main_images": main_images.astype(np.uint8, copy=False),
        "extra_view_images": extra_view_images,
        "states": states.astype(np.float32, copy=False),
        "task_descriptions": [instruction or ""],
    }
