# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import traceback
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy fallback
    import gym

from rlinf.envs.vlabench.utils import (
    DEFAULT_EE_FRAME_OFFSET,
    ee_action_to_ctrl,
    ensure_vlabench_importable,
    cfg_to_container,
    get_cfg_value,
    get_episode_candidates,
    load_episode_configs,
    normalize_task_names,
    validate_mvp_config,
    wrap_observation,
)

__all__ = ["VLABenchEnv"]


def _vlabench_subprocess_worker(remote, parent_remote, worker_id: int):
    parent_remote.close()
    env = None
    current_cfg = None
    try:
        while True:
            cmd, payload = remote.recv()
            try:
                if cmd == "reset":
                    cfg, seed_offset = payload
                    if env is not None:
                        env.close()
                    env = VLABenchEnv(
                        cfg=cfg,
                        num_envs=1,
                        seed_offset=seed_offset,
                        total_num_processes=1,
                        worker_info=None,
                    )
                    current_cfg = cfg
                    remote.send(("ok", env.reset()))
                elif cmd == "step":
                    if env is None:
                        raise RuntimeError("worker env is not initialized")
                    remote.send(("ok", env.step(payload)))
                elif cmd == "chunk_step":
                    if env is None:
                        raise RuntimeError("worker env is not initialized")
                    remote.send(("ok", env.chunk_step(payload)))
                elif cmd == "render":
                    if env is None:
                        raise RuntimeError("worker env is not initialized")
                    remote.send(("ok", env.render(**(payload or {}))))
                elif cmd == "close":
                    if env is not None:
                        env.close()
                        env = None
                    remote.send(("ok", None))
                    break
                elif cmd == "get_attr":
                    if env is None:
                        raise RuntimeError("worker env is not initialized")
                    remote.send(("ok", getattr(env, payload)))
                else:
                    raise RuntimeError(f"unknown VLABench subprocess command: {cmd}")
            except Exception:
                remote.send(("error", traceback.format_exc()))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        remote.close()


class VLABenchEnv(gym.Env):
    """Gym-style VLABench wrapper with sync vector-env support.

    Supported Phase-2 scope:
    - sync for-loop num_envs >= 1
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

        self.cfg = cfg
        cfg_num_envs = get_cfg_value(cfg, "num_envs", None)
        total_num_envs = get_cfg_value(cfg, "total_num_envs", None)
        self.num_envs = int(num_envs or cfg_num_envs or total_num_envs or 1)
        if self.num_envs < 1:
            raise ValueError("VLABenchEnv requires num_envs >= 1")

        self.vector_mode = get_cfg_value(cfg, "vector_mode", None)
        if self.vector_mode is None:
            self.vector_mode = "subprocess" if bool(get_cfg_value(cfg, "use_subprocess_env", False)) else "sync"
        if self.vector_mode not in ("sync", "subprocess"):
            raise ValueError("VLABenchEnv vector_mode must be 'sync' or 'subprocess'")

        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = int(get_cfg_value(cfg, "seed", 0)) + seed_offset
        self.rng = np.random.default_rng(self.seed)

        self.episode_configs, self.episode_config_source = load_episode_configs(cfg)
        self.task_names = normalize_task_names(cfg, self.episode_configs)
        self.default_task_name = self.task_names[0]
        self.task_sample_mode = get_cfg_value(cfg, "task_sample_mode", "uniform")
        self.episode_config_sample_mode = get_cfg_value(cfg, "episode_config_sample_mode", "sequential")
        self._task_cursor = 0
        self._episode_cursors = {task_name: 0 for task_name in self.task_names}
        self.robot = get_cfg_value(cfg, "robot", "franka")
        self.ignore_terminations = bool(get_cfg_value(cfg, "ignore_terminations", False))
        self.auto_reset = bool(get_cfg_value(cfg, "auto_reset", False))
        if self.auto_reset:
            raise NotImplementedError("VLABenchEnv Phase-2 does not support auto_reset")
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
        self.eval_track = get_cfg_value(cfg, "eval_track", None)
        self._init_eval_export()

        if self.vector_mode == "subprocess":
            self._init_subprocess_vector()
            return

        ensure_vlabench_importable()
        from VLABench.envs import load_env

        self._load_env = load_env
        self.envs = []
        self.env_task_names = []
        self.env_episode_configs = []
        self.env_episode_config_ids = []
        for env_idx in range(self.num_envs):
            task_name, episode_config, episode_config_id = self._sample_task_for_env(env_idx)
            self.envs.append(self._make_env(task_name, episode_config))
            self.env_task_names.append(task_name)
            self.env_episode_configs.append(episode_config)
            self.env_episode_config_ids.append(episode_config_id)
        self.env = self.envs[0]

        self.elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.last_raw_obs = [None] * self.num_envs
        self.last_obs = None
        self.last_info = None
        self._episode_done = np.zeros(self.num_envs, dtype=bool)
        self._last_done_obs = [None] * self.num_envs
        self._last_done_info = [None] * self.num_envs
        self._last_done_termination = np.zeros(self.num_envs, dtype=bool)
        self._last_done_truncation = np.zeros(self.num_envs, dtype=bool)

        ncam = int(self.envs[0].physics.model.ncam)
        extra_cams = max(ncam - 1, 0) if bool(get_cfg_value(cfg, "use_extra_views", True)) else 0
        action_shape = (7,) if self.num_envs == 1 else (self.num_envs, 7)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=action_shape,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "main_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.num_envs, self.render_height, self.render_width, 3),
                    dtype=np.uint8,
                ),
                "extra_view_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.num_envs, extra_cams, self.render_height, self.render_width, 3),
                    dtype=np.uint8,
                ),
                "states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_envs, 7),
                    dtype=np.float32,
                ),
                "task_descriptions": gym.spaces.Sequence(gym.spaces.Text(max_length=1024)),
            }
        )

    @property
    def instruction(self) -> str:
        return self._instruction(0)

    @property
    def device(self):
        return "cpu"

    @property
    def total_num_group_envs(self):
        return self.num_envs

    def _get_nested_cfg_value(self, section: str, key: str, default=None):
        value = get_cfg_value(self.cfg, section, None)
        if value is None:
            return default
        return get_cfg_value(value, key, default)

    def _init_eval_export(self) -> None:
        self.eval_export_enabled = bool(
            self._get_nested_cfg_value("vlabench_eval", "export_results", False)
        )
        self.eval_export_format = self._get_nested_cfg_value(
            "vlabench_eval", "export_format", "jsonl"
        )
        if self.eval_export_format != "jsonl":
            raise NotImplementedError("VLABench eval export currently supports export_format='jsonl'")
        default_result_path = os.path.join(".", "vlabench_results.jsonl")
        self.eval_result_path = self._get_nested_cfg_value(
            "vlabench_eval", "result_path", default_result_path
        )
        self.eval_summary_path = self._get_nested_cfg_value(
            "vlabench_eval", "summary_path", None
        )
        self.eval_summary_csv_path = self._get_nested_cfg_value(
            "vlabench_eval", "summary_csv_path", None
        )
        if self.eval_summary_path is None:
            root, _ = os.path.splitext(str(self.eval_result_path))
            self.eval_summary_path = f"{root}_summary.json"
        if self.eval_summary_csv_path is None:
            root, _ = os.path.splitext(str(self.eval_summary_path))
            self.eval_summary_csv_path = f"{root}.csv"
        self._episode_id_counter = 0
        self._exported_done = np.zeros(self.num_envs, dtype=bool)
        self._ik_attempts = np.zeros(self.num_envs, dtype=np.int32)
        self._ik_failures = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_records = []
        if self.eval_export_enabled:
            result_dir = os.path.dirname(str(self.eval_result_path))
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            summary_dir = os.path.dirname(str(self.eval_summary_path))
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            csv_dir = os.path.dirname(str(self.eval_summary_csv_path))
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)

    def _to_jsonable(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        return str(value)

    def _as_scalar(self, value, env_idx: int = 0):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value[env_idx].item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value[env_idx].item()
        if isinstance(value, (list, tuple)):
            return value[env_idx]
        return value

    def _update_ik_stats(self, env_idx: int, ik_success) -> None:
        if ik_success is None:
            return
        self._ik_attempts[env_idx] += 1
        if not bool(ik_success):
            self._ik_failures[env_idx] += 1

    def _build_episode_record(self, env_idx: int, info: dict, final_reward: float) -> dict:
        attempts = int(self._ik_attempts[env_idx])
        failures = int(self._ik_failures[env_idx])
        record = {
            "episode_id": int(self._episode_id_counter),
            "env_id": int(env_idx),
            "task_name": self._as_scalar(info.get("task_name"), env_idx),
            "instruction": self._as_scalar(info.get("instruction"), env_idx),
            "episode_config_id": self._as_scalar(info.get("episode_config_id"), env_idx),
            "success": bool(self._as_scalar(info.get("success", False), env_idx)),
            "success_once": bool(self._as_scalar(info.get("success_once", False), env_idx)),
            "episode_return": float(self._as_scalar(info.get("episode_return", 0.0), env_idx)),
            "elapsed_steps": int(self._as_scalar(info.get("elapsed_steps", 0), env_idx)),
            "terminated": bool(self._as_scalar(info.get("terminated", False), env_idx)),
            "truncated": bool(self._as_scalar(info.get("truncated", False), env_idx)),
            "ik_success_rate": (float(attempts - failures) / attempts if attempts > 0 else None),
            "ik_failure_count": failures,
            "ik_failure_rate": (float(failures) / attempts if attempts > 0 else None),
            "final_reward": float(final_reward),
            "vector_mode": self.vector_mode,
            "eval_track": self._to_jsonable(self.eval_track),
            "episode_config_path": self._to_jsonable(self.episode_config_source),
            "seed": int(self.seed),
        }
        for key in ("progress_score", "intention_score"):
            if key in info:
                value = self._as_scalar(info.get(key), env_idx)
                if value is not None:
                    record[key] = float(value)
        return record

    def _write_episode_record(self, record: dict) -> None:
        if not self.eval_export_enabled:
            return
        with open(self.eval_result_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_eval_summary()

    def _write_eval_summary(self) -> None:
        summary = self._compute_eval_summary()
        with open(self.eval_summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        rows = summary.get("tasks", {})
        fieldnames = [
            "task_name",
            "num_episodes",
            "success_rate",
            "avg_episode_return",
            "avg_elapsed_steps",
            "avg_progress_score",
            "avg_intention_score",
            "ik_failure_rate",
        ]
        with open(self.eval_summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for task_name, metrics in rows.items():
                row = {"task_name": task_name}
                row.update({key: metrics.get(key) for key in fieldnames if key != "task_name"})
                writer.writerow(row)

    def _mean_optional(self, values: list):
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return None
        return float(np.mean(filtered))

    def _compute_eval_summary(self) -> dict:
        by_task = {}
        for record in self._episode_records:
            by_task.setdefault(record["task_name"], []).append(record)
        task_summary = {}
        for task_name, records in by_task.items():
            task_summary[task_name] = {
                "num_episodes": len(records),
                "success_rate": float(np.mean([record["success"] for record in records])),
                "avg_episode_return": float(np.mean([record["episode_return"] for record in records])),
                "avg_elapsed_steps": float(np.mean([record["elapsed_steps"] for record in records])),
                "avg_progress_score": self._mean_optional([record.get("progress_score") for record in records]),
                "avg_intention_score": self._mean_optional([record.get("intention_score") for record in records]),
                "ik_failure_rate": self._mean_optional([record.get("ik_failure_rate") for record in records]),
            }
        all_records = self._episode_records
        overall = {
            "num_episodes": len(all_records),
            "success_rate": float(np.mean([record["success"] for record in all_records])) if all_records else 0.0,
            "avg_episode_return": float(np.mean([record["episode_return"] for record in all_records])) if all_records else 0.0,
            "avg_elapsed_steps": float(np.mean([record["elapsed_steps"] for record in all_records])) if all_records else 0.0,
            "avg_progress_score": self._mean_optional([record.get("progress_score") for record in all_records]),
            "avg_intention_score": self._mean_optional([record.get("intention_score") for record in all_records]),
            "ik_failure_rate": self._mean_optional([record.get("ik_failure_rate") for record in all_records]),
        }
        return {"overall": overall, "tasks": task_summary}

    def _episode_metrics_from_records(self, records: list[dict]) -> dict:
        if not records:
            return {}
        return {
            "vlabench/success_rate": torch.tensor([record["success"] for record in records], dtype=torch.float32),
            "vlabench/avg_episode_return": torch.tensor([record["episode_return"] for record in records], dtype=torch.float32),
            "vlabench/avg_elapsed_steps": torch.tensor([record["elapsed_steps"] for record in records], dtype=torch.float32),
            "vlabench/ik_failure_rate": torch.tensor([record.get("ik_failure_rate") or 0.0 for record in records], dtype=torch.float32),
            "vlabench/avg_progress_score": torch.tensor([record.get("progress_score") or 0.0 for record in records], dtype=torch.float32),
            "vlabench/avg_intention_score": torch.tensor([record.get("intention_score") or 0.0 for record in records], dtype=torch.float32),
        }

    def _maybe_record_done_episode(self, env_idx: int, info: dict, final_reward: float):
        done = bool(self._as_scalar(info.get("terminated", False), env_idx)) or bool(
            self._as_scalar(info.get("truncated", False), env_idx)
        )
        if not done or self._exported_done[env_idx]:
            return None
        record = self._build_episode_record(env_idx, info, final_reward)
        self._episode_records.append(record)
        self._episode_id_counter += 1
        self._exported_done[env_idx] = True
        self._write_episode_record(record)
        return record

    def _attach_episode_metrics(self, info: dict, records: list[dict]) -> dict:
        if records:
            info["episode"] = self._episode_metrics_from_records(records)
        return info

    def _init_subprocess_vector(self) -> None:
        self._load_env = None
        self.envs = []
        self.env = None
        self.env_task_names = [None] * self.num_envs
        self.env_episode_configs = [None] * self.num_envs
        self.env_episode_config_ids = [None] * self.num_envs
        self.elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.last_raw_obs = [None] * self.num_envs
        self.last_obs = None
        self.last_info = None
        self._episode_done = np.zeros(self.num_envs, dtype=bool)
        self._last_done_obs = [None] * self.num_envs
        self._last_done_info = [None] * self.num_envs
        self._last_done_termination = np.zeros(self.num_envs, dtype=bool)
        self._last_done_truncation = np.zeros(self.num_envs, dtype=bool)
        self._closed = False

        self._subproc_ctx = mp.get_context("spawn")
        self._subproc_remotes = []
        self._subproc_processes = []
        for worker_id in range(self.num_envs):
            parent_remote, child_remote = self._subproc_ctx.Pipe()
            process = self._subproc_ctx.Process(
                target=_vlabench_subprocess_worker,
                args=(child_remote, parent_remote, worker_id),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self._subproc_remotes.append(parent_remote)
            self._subproc_processes.append(process)

        action_shape = (7,) if self.num_envs == 1 else (self.num_envs, 7)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=action_shape,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "main_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.num_envs, self.render_height, self.render_width, 3),
                    dtype=np.uint8,
                ),
                "extra_view_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.num_envs, 0, self.render_height, self.render_width, 3),
                    dtype=np.uint8,
                ),
                "states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_envs, 7),
                    dtype=np.float32,
                ),
                "task_descriptions": gym.spaces.Sequence(gym.spaces.Text(max_length=1024)),
            }
        )

    def _assert_subprocess_alive(self, env_idx: int) -> None:
        process = self._subproc_processes[env_idx]
        if not process.is_alive():
            raise RuntimeError(
                f"VLABench subprocess worker {env_idx} is not alive; exitcode={process.exitcode}"
            )

    def _subprocess_call(self, env_idx: int, cmd: str, payload=None):
        if self._closed:
            raise RuntimeError("VLABench subprocess vector env is already closed")
        self._assert_subprocess_alive(env_idx)
        remote = self._subproc_remotes[env_idx]
        try:
            remote.send((cmd, payload))
            status, result = remote.recv()
        except (EOFError, BrokenPipeError) as exc:
            raise RuntimeError(f"VLABench subprocess worker {env_idx} pipe failed") from exc
        if status == "error":
            raise RuntimeError(f"VLABench subprocess worker {env_idx} failed:\n{result}")
        return result

    def _subprocess_send_all(self, cmd: str, payloads: list):
        if self._closed:
            raise RuntimeError("VLABench subprocess vector env is already closed")
        for env_idx, payload in enumerate(payloads):
            self._assert_subprocess_alive(env_idx)
            self._subproc_remotes[env_idx].send((cmd, payload))

    def _subprocess_recv_all(self, cmd: str):
        results = []
        for env_idx, remote in enumerate(self._subproc_remotes):
            try:
                status, result = remote.recv()
            except (EOFError, BrokenPipeError) as exc:
                raise RuntimeError(
                    f"VLABench subprocess worker {env_idx} pipe failed during {cmd}"
                ) from exc
            if status == "error":
                raise RuntimeError(f"VLABench subprocess worker {env_idx} failed during {cmd}:\n{result}")
            results.append(result)
        return results

    def _make_child_cfg(self, task_name: str, episode_config: Optional[dict]) -> dict:
        base = cfg_to_container(self.cfg)
        if base is None:
            base = {}
        if not isinstance(base, dict):
            if hasattr(base, "__dict__"):
                base = dict(vars(base))
            else:
                base = dict(base)
        base.update(
            {
                "num_envs": 1,
                "total_num_envs": 1,
                "vector_mode": "sync",
                "use_subprocess_env": False,
                "task_name": task_name,
                "task_names": None,
                "eval_track": None,
                "episode_config_path": None,
                "return_tensors": False,
                "task_sample_mode": "sequential",
                "episode_config_sample_mode": "sequential",
                "vlabench_eval": {"export_results": False},
            }
        )
        if episode_config is not None:
            base["episode_configs"] = {task_name: [episode_config]}
            base["random_init"] = False
        else:
            base.pop("episode_configs", None)
        return base

    def _subprocess_reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed = int(seed) + self.seed_offset
            self.rng = np.random.default_rng(self.seed)
            np.random.seed(self.seed)
        else:
            np.random.seed(self.seed)
        if options:
            unsupported = sorted(options)
            raise NotImplementedError(
                f"VLABenchEnv subprocess mode does not support partial reset options: {unsupported}"
            )

        payloads = []
        sampled = []
        for env_idx in range(self.num_envs):
            task_name, episode_config, episode_config_id = self._sample_task_for_env(env_idx)
            child_cfg = self._make_child_cfg(task_name, episode_config)
            payloads.append((child_cfg, self.seed_offset + env_idx))
            sampled.append((task_name, episode_config, episode_config_id))

        self._subprocess_send_all("reset", payloads)
        results = self._subprocess_recv_all("reset")
        obs_list = []
        infos = []
        for env_idx, ((obs, info), (task_name, episode_config, episode_config_id)) in enumerate(zip(results, sampled)):
            self.env_task_names[env_idx] = task_name
            self.env_episode_configs[env_idx] = episode_config
            self.env_episode_config_ids[env_idx] = episode_config_id
            self.elapsed_steps[env_idx] = 0
            self.episode_return[env_idx] = 0.0
            self.success_once[env_idx] = False
            self._exported_done[env_idx] = False
            self._ik_attempts[env_idx] = 0
            self._ik_failures[env_idx] = 0
            self._episode_done[env_idx] = False
            self._last_done_obs[env_idx] = None
            self._last_done_info[env_idx] = None
            self._last_done_termination[env_idx] = False
            self._last_done_truncation[env_idx] = False
            info["task_name"] = task_name
            info["episode_config_id"] = episode_config_id
            obs_list.append(obs)
            infos.append(info)

        obs = self._merge_obs(obs_list)
        formatted_obs = self._format_obs(obs)
        self.last_obs = obs
        self.last_info = infos[0] if self.num_envs == 1 else self._batch_info(infos)
        return formatted_obs, self.last_info

    def _subprocess_step(self, action):
        actions, input_was_single = self._normalize_step_actions(action)
        self._subprocess_send_all("step", [actions[i] for i in range(self.num_envs)])
        results = self._subprocess_recv_all("step")
        obs_list = []
        infos = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        for env_idx, (obs, reward, terminated, truncated, info) in enumerate(results):
            self._update_ik_stats(env_idx, info.get("ik_success"))
            record = self._maybe_record_done_episode(env_idx, info, reward)
            self._attach_episode_metrics(info, [record] if record is not None else [])
            obs_list.append(obs)
            infos.append(info)
            rewards[env_idx] = reward
            terminations[env_idx] = terminated
            truncations[env_idx] = truncated
        formatted_obs = self._format_obs(self._merge_obs(obs_list))
        self.last_info = infos[0] if self.num_envs == 1 else self._batch_info(infos)
        return self._maybe_unbatch_step_return(
            formatted_obs,
            rewards,
            terminations,
            truncations,
            infos,
            input_was_single,
        )

    def _subprocess_chunk_step(self, chunk_actions):
        actions = self._normalize_chunk_actions(chunk_actions)
        self._subprocess_send_all("chunk_step", [actions[i] for i in range(self.num_envs)])
        results = self._subprocess_recv_all("chunk_step")
        chunk_size = actions.shape[1]
        obs_list = []
        infos_list = []
        rewards = torch.zeros((self.num_envs, chunk_size), dtype=torch.float32)
        terminations = torch.zeros((self.num_envs, chunk_size), dtype=torch.bool)
        truncations = torch.zeros((self.num_envs, chunk_size), dtype=torch.bool)
        for step_idx in range(chunk_size):
            step_obs = []
            step_infos = []
            for env_idx, (worker_obs_list, worker_rewards, worker_terms, worker_truncs, worker_infos) in enumerate(results):
                info = worker_infos[step_idx]
                reward_value = float(worker_rewards[0, step_idx].item())
                self._update_ik_stats(env_idx, info.get("ik_success"))
                record = self._maybe_record_done_episode(env_idx, info, reward_value)
                self._attach_episode_metrics(info, [record] if record is not None else [])
                step_obs.append(worker_obs_list[step_idx])
                step_infos.append(info)
                rewards[env_idx, step_idx] = worker_rewards[0, step_idx]
                terminations[env_idx, step_idx] = worker_terms[0, step_idx]
                truncations[env_idx, step_idx] = worker_truncs[0, step_idx]
            obs_list.append(self._format_obs(self._merge_obs(step_obs)))
            infos_list.append(step_infos[0] if self.num_envs == 1 else self._batch_info(step_infos))
        return obs_list, rewards, terminations, truncations, infos_list

    def _subprocess_render(self, *, mode: str = "rgb_array", env_idx: int = 0, tile: Optional[bool] = None):
        if mode != "rgb_array":
            raise NotImplementedError("VLABenchEnv only supports render(mode='rgb_array')")
        if tile:
            images = [self._subprocess_call(i, "render", {"mode": mode}) for i in range(self.num_envs)]
            return self._tile_images(images)
        env_idx = int(env_idx)
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx {env_idx} out of range for num_envs={self.num_envs}")
        return self._subprocess_call(env_idx, "render", {"mode": mode})

    def _sample_task_for_env(self, env_idx: int) -> tuple[str, Optional[dict], Optional[str]]:
        del env_idx
        if self.task_sample_mode == "sequential":
            task_idx = self._task_cursor % len(self.task_names)
            self._task_cursor += 1
        else:
            task_idx = int(self.rng.integers(0, len(self.task_names)))
        task_name = self.task_names[task_idx]

        episode_configs = get_episode_candidates(self.episode_configs, task_name)
        if not episode_configs:
            return task_name, None, None

        if self.episode_config_sample_mode == "random":
            episode_idx = int(self.rng.integers(0, len(episode_configs)))
        else:
            episode_idx = self._episode_cursors.get(task_name, 0) % len(episode_configs)
            self._episode_cursors[task_name] = episode_idx + 1
        episode_config_id = f"{self.episode_config_source or 'inline'}:{task_name}:{episode_idx}"
        return task_name, episode_configs[episode_idx], episode_config_id

    def _make_env(self, task_name: str, episode_config: Optional[dict]):
        kwargs = {}
        if episode_config is not None:
            kwargs["run_mode"] = "eval"
        env = self._load_env(
            task_name,
            robot=self.robot,
            reset_wait_step=self.reset_wait_step,
            random_init=(False if episode_config is not None else self.random_init),
            episode_config=episode_config,
            **kwargs,
        )
        env.render_options = {"height": self.render_height, "width": self.render_width}
        return env

    def _maybe_resample_env(self, env_idx: int) -> None:
        task_name, episode_config, episode_config_id = self._sample_task_for_env(env_idx)
        if task_name == self.env_task_names[env_idx] and episode_config == self.env_episode_configs[env_idx]:
            self.env_episode_config_ids[env_idx] = episode_config_id
            return
        old_env = self.envs[env_idx]
        if hasattr(old_env, "close"):
            old_env.close()
        self.envs[env_idx] = self._make_env(task_name, episode_config)
        self.env_task_names[env_idx] = task_name
        self.env_episode_configs[env_idx] = episode_config
        self.env_episode_config_ids[env_idx] = episode_config_id
        if env_idx == 0:
            self.env = self.envs[0]

    def _instruction(self, env_idx: int) -> str:
        return self.envs[env_idx].task.get_instruction() or ""

    def _safe_metric(self, env_idx: int, method_name: str):
        env = self.envs[env_idx]
        method = getattr(env, method_name, None)
        if method is None:
            return None
        try:
            if method_name == "get_intention_score":
                value = method(threshold=float(get_cfg_value(self.cfg, "intention_score_threshold", 0.1)))
            else:
                value = method()
        except Exception:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def _get_info(
        self,
        env_idx: int,
        *,
        success: bool,
        ik_success: Optional[bool] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> dict:
        info = {
            "task_name": self.env_task_names[env_idx],
            "instruction": self._instruction(env_idx),
            "episode_config_id": self.env_episode_config_ids[env_idx],
            "success": bool(success),
            "success_once": bool(self.success_once[env_idx]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "ik_success": ik_success,
            "elapsed_steps": int(self.elapsed_steps[env_idx]),
            "episode_return": float(self.episode_return[env_idx]),
        }
        progress_score = self._safe_metric(env_idx, "get_task_progress")
        if progress_score is not None:
            info["progress_score"] = progress_score
        intention_score = self._safe_metric(env_idx, "get_intention_score")
        if intention_score is not None:
            info["intention_score"] = intention_score
        return info

    def _format_obs(self, obs: dict) -> dict:
        if not self.return_tensors:
            return obs

        formatted = dict(obs)
        for key in ("main_images", "extra_view_images", "states"):
            if formatted.get(key, None) is not None and not isinstance(formatted[key], torch.Tensor):
                formatted[key] = torch.as_tensor(formatted[key], device="cpu").contiguous()
        return formatted

    def _merge_obs(self, obs_list: list[dict]) -> dict:
        main_images = np.concatenate([obs["main_images"] for obs in obs_list], axis=0)
        states = np.concatenate([obs["states"] for obs in obs_list], axis=0)
        task_descriptions = []
        for obs in obs_list:
            task_descriptions.extend(obs["task_descriptions"])

        extra_items = [obs.get("extra_view_images") for obs in obs_list]
        extra_view_images = None
        if all(item is not None for item in extra_items):
            extra_view_images = np.concatenate(extra_items, axis=0)

        return {
            "main_images": main_images.astype(np.uint8, copy=False),
            "extra_view_images": extra_view_images,
            "states": states.astype(np.float32, copy=False),
            "task_descriptions": task_descriptions,
        }

    def _get_wrapped_observation_one(self, env_idx: int):
        raw_obs = self.envs[env_idx].get_observation(require_pcd=self.require_pcd)
        obs = wrap_observation(raw_obs, self._instruction(env_idx), self.cfg)
        self.last_raw_obs[env_idx] = raw_obs
        return obs

    def _get_wrapped_observation(self):
        obs = self._merge_obs([self._get_wrapped_observation_one(i) for i in range(self.num_envs)])
        self.last_obs = obs
        return obs

    def _batch_info(self, infos: list[dict]) -> dict:
        batched = {
            "task_name": [info["task_name"] for info in infos],
            "instruction": [info["instruction"] for info in infos],
            "episode_config_id": [info.get("episode_config_id") for info in infos],
            "success": torch.tensor([info["success"] for info in infos], dtype=torch.bool),
            "success_once": torch.tensor([info["success_once"] for info in infos], dtype=torch.bool),
            "terminated": torch.tensor([info.get("terminated", False) for info in infos], dtype=torch.bool),
            "truncated": torch.tensor([info.get("truncated", False) for info in infos], dtype=torch.bool),
            "ik_success": torch.tensor([bool(info.get("ik_success", False)) for info in infos], dtype=torch.bool),
            "elapsed_steps": torch.tensor([info["elapsed_steps"] for info in infos], dtype=torch.int32),
            "episode_return": torch.tensor([info["episode_return"] for info in infos], dtype=torch.float32),
        }
        for key in ("progress_score", "intention_score"):
            if any(key in info for info in infos):
                batched[key] = [info.get(key) for info in infos]
        episode_infos = [info.get("episode") for info in infos if isinstance(info.get("episode"), dict)]
        if episode_infos:
            episode = {}
            for episode_info in episode_infos:
                for key, value in episode_info.items():
                    value = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                    episode.setdefault(key, []).append(value.reshape(-1).float())
            batched["episode"] = {key: torch.cat(values, dim=0) for key, values in episode.items()}
        return batched

    def _maybe_unbatch_step_return(self, obs, rewards, terminations, truncations, infos, input_was_single: bool):
        if self.num_envs == 1 and input_was_single:
            return obs, float(rewards[0]), bool(terminations[0]), bool(truncations[0]), infos[0]
        return (
            obs,
            rewards.astype(np.float32, copy=False),
            terminations.astype(bool, copy=False),
            truncations.astype(bool, copy=False),
            self._batch_info(infos),
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.vector_mode == "subprocess":
            return self._subprocess_reset(seed=seed, options=options)

        if seed is not None:
            self.seed = int(seed) + self.seed_offset
            self.rng = np.random.default_rng(self.seed)
            np.random.seed(self.seed)
        else:
            np.random.seed(self.seed)

        if options:
            unsupported = sorted(options)
            raise NotImplementedError(
                f"VLABenchEnv Phase-2 does not support partial reset options: {unsupported}"
            )

        obs_list = []
        infos = []
        for env_idx in range(self.num_envs):
            self._maybe_resample_env(env_idx)
            self.envs[env_idx].reset()
            self.elapsed_steps[env_idx] = 0
            self.episode_return[env_idx] = 0.0
            self.success_once[env_idx] = False
            self._exported_done[env_idx] = False
            self._ik_attempts[env_idx] = 0
            self._ik_failures[env_idx] = 0
            self._episode_done[env_idx] = False
            self._last_done_obs[env_idx] = None
            self._last_done_info[env_idx] = None
            self._last_done_termination[env_idx] = False
            self._last_done_truncation[env_idx] = False
            obs_list.append(self._get_wrapped_observation_one(env_idx))
            infos.append(self._get_info(env_idx, success=False, ik_success=None))

        obs = self._merge_obs(obs_list)
        formatted_obs = self._format_obs(obs)
        self.last_obs = obs
        self.last_info = infos[0] if self.num_envs == 1 else self._batch_info(infos)
        return formatted_obs, self.last_info

    def _step_one(self, env_idx: int, action):
        if self._episode_done[env_idx]:
            if self._last_done_obs[env_idx] is None or self._last_done_info[env_idx] is None:
                raise RuntimeError("VLABenchEnv is done but terminal observation is missing")
            return (
                self._last_done_obs[env_idx],
                0.0,
                bool(self._last_done_termination[env_idx]),
                bool(self._last_done_truncation[env_idx]),
                self._last_done_info[env_idx],
            )

        ctrl_action, ik_success = ee_action_to_ctrl(
            self.envs[env_idx],
            action,
            ee_frame_offset=self.ee_frame_offset,
            gripper_open_threshold=self.gripper_open_threshold,
            gripper_open_value=self.gripper_open_value,
        )
        self.envs[env_idx].step(ctrl_action)
        self.elapsed_steps[env_idx] += 1

        success = bool(self.envs[env_idx].task.should_terminate_episode(self.envs[env_idx].physics))
        reward = 1.0 if success else 0.0
        self.episode_return[env_idx] += reward
        self.success_once[env_idx] = bool(self.success_once[env_idx] or success)

        terminated = bool(success and not self.ignore_terminations)
        truncated = bool(self.elapsed_steps[env_idx] >= self.max_episode_steps)

        self._update_ik_stats(env_idx, ik_success)
        obs = self._get_wrapped_observation_one(env_idx)
        info = self._get_info(
            env_idx,
            success=success,
            ik_success=ik_success,
            terminated=terminated,
            truncated=truncated,
        )
        record = self._maybe_record_done_episode(env_idx, info, reward)
        self._attach_episode_metrics(info, [record] if record is not None else [])
        if terminated or truncated:
            self._episode_done[env_idx] = True
            self._last_done_obs[env_idx] = obs
            self._last_done_info[env_idx] = info
            self._last_done_termination[env_idx] = terminated
            self._last_done_truncation[env_idx] = truncated
        return obs, reward, terminated, truncated, info

    def _normalize_step_actions(self, action) -> tuple[np.ndarray, bool]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        actions = np.asarray(action, dtype=np.float32)
        input_was_single = actions.shape == (7,)
        if self.num_envs == 1:
            if actions.shape == (7,):
                actions = actions.reshape(1, 7)
            elif actions.shape == (1, 7):
                pass
            else:
                raise ValueError(f"VLABenchEnv.step expects [7] or [1, 7], got {actions.shape}")
        else:
            if actions.shape != (self.num_envs, 7):
                raise ValueError(
                    f"VLABenchEnv.step expects [B, 7] for B={self.num_envs}, got {actions.shape}"
                )
        actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return actions, input_was_single

    def step(self, action):
        if self.vector_mode == "subprocess":
            return self._subprocess_step(action)

        actions, input_was_single = self._normalize_step_actions(action)
        obs_list = []
        infos = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)

        for env_idx in range(self.num_envs):
            obs, reward, terminated, truncated, info = self._step_one(env_idx, actions[env_idx])
            obs_list.append(obs)
            infos.append(info)
            rewards[env_idx] = reward
            terminations[env_idx] = terminated
            truncations[env_idx] = truncated

        formatted_obs = self._format_obs(self._merge_obs(obs_list))
        self.last_info = infos[0] if self.num_envs == 1 else self._batch_info(infos)
        return self._maybe_unbatch_step_return(
            formatted_obs,
            rewards,
            terminations,
            truncations,
            infos,
            input_was_single,
        )

    def _normalize_chunk_actions(self, chunk_actions) -> np.ndarray:
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if self.num_envs == 1:
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
                    "VLABenchEnv.chunk_step expects [7], [1, 7], [T, 7], or [1, T, 7] "
                    f"for num_envs=1, got {actions.shape}"
                )
        else:
            if actions.ndim == 2 and actions.shape == (self.num_envs, 7):
                actions = actions.reshape(self.num_envs, 1, 7)
            elif actions.ndim == 3 and actions.shape[0] == self.num_envs and actions.shape[-1] == 7:
                pass
            else:
                raise ValueError(
                    f"VLABenchEnv.chunk_step expects [B, 7] or [B, T, 7] for B={self.num_envs}, "
                    f"got {actions.shape}"
                )
        return np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def chunk_step(self, chunk_actions):
        if self.vector_mode == "subprocess":
            return self._subprocess_chunk_step(chunk_actions)

        actions = self._normalize_chunk_actions(chunk_actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"chunk action batch {actions.shape[0]} != num_envs {self.num_envs}")

        chunk_size = actions.shape[1]
        obs_list = []
        infos_list = []
        rewards = []
        terminations = []
        truncations = []

        for step_idx in range(chunk_size):
            step_obs = []
            step_infos = []
            step_rewards = np.zeros(self.num_envs, dtype=np.float32)
            step_terminations = np.zeros(self.num_envs, dtype=bool)
            step_truncations = np.zeros(self.num_envs, dtype=bool)
            for env_idx in range(self.num_envs):
                obs, reward, terminated, truncated, info = self._step_one(env_idx, actions[env_idx, step_idx])
                step_obs.append(obs)
                step_infos.append(info)
                step_rewards[env_idx] = reward
                step_terminations[env_idx] = terminated
                step_truncations[env_idx] = truncated

            obs_list.append(self._format_obs(self._merge_obs(step_obs)))
            infos_list.append(step_infos[0] if self.num_envs == 1 else self._batch_info(step_infos))
            rewards.append(torch.as_tensor(step_rewards, dtype=torch.float32))
            terminations.append(torch.as_tensor(step_terminations, dtype=torch.bool))
            truncations.append(torch.as_tensor(step_truncations, dtype=torch.bool))

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

    def _render_one(self, env_idx: int):
        return self.envs[env_idx].render(
            camera_id=self.camera_id,
            height=self.render_height,
            width=self.render_width,
        ).astype(np.uint8, copy=False)

    def _tile_images(self, images: list[np.ndarray]) -> np.ndarray:
        if len(images) == 1:
            return images[0]
        height, width, channels = images[0].shape
        cols = int(np.ceil(np.sqrt(len(images))))
        rows = int(np.ceil(len(images) / cols))
        canvas = np.zeros((rows * height, cols * width, channels), dtype=np.uint8)
        for idx, image in enumerate(images):
            row = idx // cols
            col = idx % cols
            canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
        return canvas

    def render(self, info=None, rew=None, mode: str = "rgb_array", env_idx: int = 0, tile: Optional[bool] = None):
        if self.vector_mode == "subprocess":
            return self._subprocess_render(mode=mode, env_idx=env_idx, tile=tile)

        if mode != "rgb_array":
            raise NotImplementedError("VLABenchEnv only supports render(mode='rgb_array')")
        if tile is None:
            tile = False
        if tile:
            return self._tile_images([self._render_one(i) for i in range(self.num_envs)])

        env_idx = int(env_idx)
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx {env_idx} out of range for num_envs={self.num_envs}")
        return self._render_one(env_idx)

    def close(self):
        if getattr(self, "vector_mode", "sync") == "subprocess":
            if getattr(self, "_closed", False):
                return
            self._closed = True
            for env_idx, remote in enumerate(getattr(self, "_subproc_remotes", [])):
                try:
                    if self._subproc_processes[env_idx].is_alive():
                        remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for env_idx, remote in enumerate(getattr(self, "_subproc_remotes", [])):
                try:
                    if self._subproc_processes[env_idx].is_alive() and remote.poll(5.0):
                        remote.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
                try:
                    remote.close()
                except OSError:
                    pass
            for process in getattr(self, "_subproc_processes", []):
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2.0)
            return

        for env in getattr(self, "envs", []):
            if env is not None and hasattr(env, "close"):
                env.close()
