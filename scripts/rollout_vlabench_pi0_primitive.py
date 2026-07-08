#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.vlabench.vlabench_env import VLABenchEnv
from rlinf.models.embodiment.openpi import get_model


LOGGER = logging.getLogger("vlabench_pi0_rollout")


def scalar(value: Any, idx: int = 0):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.reshape(-1)[idx].item()
    if isinstance(value, (list, tuple)):
        item = value[idx]
        if isinstance(item, (torch.Tensor, np.ndarray, list, tuple)):
            return scalar(item, 0)
        return item
    return value


def jsonable(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_model_cfg(checkpoint_dir: str, num_steps: int):
    return OmegaConf.create(
        {
            "model_path": checkpoint_dir,
            "precision": None,
            "num_action_chunks": 10,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": num_steps,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi0_ft_vlabench_primitive",
                "num_images_in_input": 3,
                "noise_level": 0.5,
                "action_chunk": 10,
                "num_steps": num_steps,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "detach_critic_input": False,
            },
            "openpi_data": {
                "repo_id": "vlabench/vlabench_ft_primitive",
            },
        }
    )


def build_env_cfg(task_name: str, max_episode_steps: int, seed: int, num_envs: int, vector_mode: str):
    return OmegaConf.create(
        {
            "env_type": "vlabench",
            "task_name": task_name,
            "robot": "franka",
            "num_envs": num_envs,
            "total_num_envs": num_envs,
            "group_size": num_envs,
            "seed": seed,
            "vector_mode": vector_mode,
            "control_mode": "ee",
            "action_mode": "absolute_ee",
            "reward_mode": "success",
            "delta_position_scale": 1.0,
            "delta_rotation_scale": 1.0,
            "delta_position_clip": 0.05,
            "delta_rotation_clip": 0.25,
            "ee_frame_offset": [0.0, -0.4, 0.78],
            "gripper_open_threshold": 0.1,
            "gripper_open_value": 0.04,
            "ignore_terminations": False,
            "max_episode_steps": max_episode_steps,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "auto_reset": False,
            "return_tensors": True,
            "require_pcd": False,
            "reset_wait_step": 10,
            "random_init": True,
            "camera_id": 2,
            "use_extra_views": True,
            "render_height": 256,
            "render_width": 256,
            "init_params": {},
        }
    )


def ensure_openpi_obs_keys(obs):
    # NOTE: the VLABench openpi config derives observation/second_image and
    # observation/wrist_image directly from env_obs["extra_view_images"]
    # inside OpenPi0ForRLActionPrediction.obs_processor (see
    # rlinf/models/embodiment/openpi/openpi_action_model.py), so no key
    # faking is needed here anymore. Keep this as a no-op passthrough so the
    # call site below doesn't need to change.
    if "extra_view_images" not in obs:
        obs["extra_view_images"] = None
    return obs


class ActionSafetyError(RuntimeError):
    """Raised when a predicted or final env action looks physically invalid."""


def check_model_dims(model, expected_action_env_dim: int = 7):
    action_env_dim = model.config.action_env_dim
    if action_env_dim != expected_action_env_dim:
        raise ActionSafetyError(
            f"model.config.action_env_dim={action_env_dim} != expected "
            f"{expected_action_env_dim} (xyz + euler + gripper). Refusing to roll out."
        )
    if "vlabench" not in model.config.config_name.lower():
        raise ActionSafetyError(
            f"model.config.config_name={model.config.config_name!r} does not "
            "reference vlabench; refusing to roll out with a possibly mismatched "
            "openpi config."
        )


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi].

    Absolute Euler-angle targets are only meaningful modulo 2*pi (they feed
    into `euler_to_quaternion`, which is periodic), so a raw arithmetic
    delta between two absolute angles near the +-pi boundary can read as
    ~2*pi even though the physical rotation is tiny. Always compare wrapped
    deltas, not raw ones.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def check_final_action(
    env: VLABenchEnv,
    action: np.ndarray,
    prev_action: np.ndarray | None,
    *,
    max_abs_xyz: float = 2.0,
    max_step_xyz_delta: float = 0.5,
    max_abs_euler: float = 4 * np.pi,
    max_step_euler_delta: float = np.pi + 1e-3,
):
    if not np.all(np.isfinite(action)):
        raise ActionSafetyError(f"final env action is not finite: {action.tolist()}")

    low, high = env.action_space.low, env.action_space.high
    if low.shape != action.shape:
        low = low.reshape(-1, action.shape[-1])[0]
        high = high.reshape(-1, action.shape[-1])[0]
    if np.any(action < low) or np.any(action > high):
        raise ActionSafetyError(
            f"final env action {action.tolist()} is outside env.action_space "
            f"[{low.tolist()}, {high.tolist()}]"
        )

    xyz, euler, gripper = action[:3], action[3:6], action[6]
    if np.any(np.abs(xyz) > max_abs_xyz):
        raise ActionSafetyError(
            f"final env action xyz={xyz.tolist()} exceeds max_abs_xyz={max_abs_xyz}"
        )
    if np.any(np.abs(euler) > max_abs_euler):
        raise ActionSafetyError(
            f"final env action euler={euler.tolist()} exceeds max_abs_euler={max_abs_euler} "
            "(unwrapped rotation this large usually indicates a delta/absolute action mixup)"
        )
    if not (-1.0 <= gripper <= 2.0):
        raise ActionSafetyError(f"final env action gripper={gripper} out of expected range")

    if prev_action is not None:
        d_xyz = xyz - prev_action[:3]
        d_euler = wrap_to_pi(euler - prev_action[3:6])
        if np.any(np.abs(d_xyz) > max_step_xyz_delta):
            raise ActionSafetyError(
                f"single-step xyz delta {d_xyz.tolist()} exceeds max_step_xyz_delta="
                f"{max_step_xyz_delta}"
            )
        if np.any(np.abs(d_euler) > max_step_euler_delta):
            raise ActionSafetyError(
                f"single-step wrapped euler delta {d_euler.tolist()} exceeds "
                f"max_step_euler_delta={max_step_euler_delta}"
            )


def render_frame(env: VLABenchEnv):
    frame = env.render(tile=(env.num_envs > 1))
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def debug_action_layers(model, result, raw_state_7d, num_dims: int = 7):
    """Best-effort per-layer trace of the action pipeline for one sample (batch 0).

    Returns a dict with the raw model action and the action after each output
    transform stage (see model.debug_output_stages, set in
    rlinf/models/embodiment/openpi/__init__.py::get_model). `raw_state_7d` is
    the raw (unnormalized) env state; it is only used by the AbsoluteActions
    stage, which -- by the time it runs in the real pipeline -- has already
    seen its "state" input round-tripped through Unnormalize(Normalize(.)),
    so the raw env state is the correct value to feed here too. Never raises:
    on any shape mismatch it just records the error message so it doesn't
    block the rollout.
    """
    try:
        action_dim = model.config.action_dim
        model_action_flat = _to_numpy(result["forward_inputs"]["model_action"][0])
        model_action = model_action_flat.reshape(-1, action_dim)
        # Unnormalize() pads norm_stats.mean/std (already action_dim-long, incl.
        # zero-padding beyond the real 7 dims) to `state.shape[-1]` without ever
        # truncating them, so `state` must already be action_dim-long here too
        # (mirroring VLABenchInputs' pad_to_dim(ee_state, action_dim) on the
        # real input path), or the broadcast against a 32-long mean/std fails.
        state_7d = np.asarray(raw_state_7d, dtype=np.float32).reshape(-1)
        state = np.zeros((action_dim,), dtype=np.float32)
        state[: state_7d.shape[0]] = state_7d
        layers = {"raw_model_action": model_action[:, :num_dims].tolist()}
        sample = {"actions": model_action.copy(), "state": state.copy()}
        for name, fn in getattr(model, "debug_output_stages", []):
            sample = fn(sample)
            layers[name] = np.asarray(sample["actions"])[:, :num_dims].tolist()
        return layers
    except Exception as exc:  # noqa: BLE001 - diagnostic only, must not crash the rollout
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_episode(model, args, episode_idx: int):
    env_cfg = build_env_cfg(
        args.task_name,
        args.max_episode_steps,
        args.seed + episode_idx,
        args.num_envs,
        args.vector_mode,
    )
    env = VLABenchEnv(env_cfg, num_envs=args.num_envs, seed_offset=episode_idx)
    frames = []
    step_records = [[] for _ in range(args.num_envs)]
    layer_traces = [[] for _ in range(args.num_envs)]
    try:
        obs, info = env.reset()
        obs = ensure_openpi_obs_keys(obs)
        frames.append(render_frame(env))
        done = np.zeros(args.num_envs, dtype=bool)
        total_reward = np.zeros(args.num_envs, dtype=np.float32)
        step_count = np.zeros(args.num_envs, dtype=np.int32)
        last_info = info
        prev_actions = [None] * args.num_envs
        LOGGER.info(
            "env.action_space low=%s high=%s shape=%s",
            np.asarray(env.action_space.low).tolist(),
            np.asarray(env.action_space.high).tolist(),
            env.action_space.shape,
        )
        while not bool(done.all()) and int(step_count.max()) < args.max_episode_steps:
            with torch.inference_mode():
                actions, result = model.predict_action_batch(obs, mode="eval", compute_values=False)
            action_np = actions.detach().cpu().numpy()
            if action_np.ndim == 2:
                action_chunk = action_np[:, None, :]
            else:
                action_chunk = action_np

            for env_idx in range(args.num_envs):
                if args.debug_layers and step_count[env_idx] < 10:
                    trace = debug_action_layers(model, result, obs["states"][env_idx])
                    trace["env_idx"] = env_idx
                    trace["chunk_start_step"] = int(step_count[env_idx])
                    trace["obs_state"] = jsonable(obs["states"][env_idx])
                    layer_traces[env_idx].append(trace)
                    LOGGER.info("env=%d step=%d layer trace: %s", env_idx, step_count[env_idx], trace)

            for chunk_i in range(action_chunk.shape[1]):
                if bool(done.all()) or int(step_count.max()) >= args.max_episode_steps:
                    break
                batch_action = action_chunk[:, chunk_i, :].astype(np.float32)
                for env_idx, action in enumerate(batch_action):
                    if done[env_idx]:
                        continue
                    check_final_action(env, action, prev_actions[env_idx])
                    prev_actions[env_idx] = action.copy()
                obs, reward, terminated, truncated, info = env.step(batch_action)
                obs = ensure_openpi_obs_keys(obs)
                last_info = info
                frames.append(render_frame(env))
                for env_idx, action in enumerate(batch_action):
                    if done[env_idx]:
                        continue
                    reward_value = float(scalar(reward, env_idx))
                    term_value = bool(scalar(terminated, env_idx))
                    trunc_value = bool(scalar(truncated, env_idx))
                    done[env_idx] = term_value or trunc_value
                    total_reward[env_idx] += reward_value
                    step_count[env_idx] += 1
                    step_records[env_idx].append(
                        {
                            "step": int(step_count[env_idx]),
                            "chunk_index": chunk_i,
                            "reward": reward_value,
                            "terminated": term_value,
                            "truncated": trunc_value,
                            "success": bool(scalar(info.get("success", False), env_idx)),
                            "success_once": bool(scalar(info.get("success_once", False), env_idx)),
                            "elapsed_steps": int(scalar(info.get("elapsed_steps", step_count), env_idx)),
                            "action": action.astype(float).tolist(),
                        }
                    )
        env_summaries = []
        for env_idx in range(args.num_envs):
            env_summaries.append(
                {
                    "env_idx": env_idx,
                    "steps": int(step_count[env_idx]),
                    "total_reward": float(total_reward[env_idx]),
                    "terminated": bool(scalar(last_info.get("terminated", False), env_idx)) if isinstance(last_info, dict) else bool(done[env_idx]),
                    "truncated": bool(scalar(last_info.get("truncated", False), env_idx)) if isinstance(last_info, dict) else int(step_count[env_idx]) >= args.max_episode_steps,
                    "success": bool(scalar(last_info.get("success", False), env_idx)) if isinstance(last_info, dict) else False,
                    "success_once": bool(scalar(last_info.get("success_once", False), env_idx)) if isinstance(last_info, dict) else False,
                    "instruction": scalar(last_info.get("instruction", ""), env_idx) if isinstance(last_info, dict) else "",
                    "episode_config_id": scalar(last_info.get("episode_config_id", None), env_idx) if isinstance(last_info, dict) else None,
                    "step_records": step_records[env_idx],
                    "layer_traces": layer_traces[env_idx],
                }
            )
        summary = {
            "episode": episode_idx,
            "task_name": args.task_name,
            "checkpoint_dir": args.checkpoint_dir,
            "num_envs": args.num_envs,
            "vector_mode": args.vector_mode,
            "max_episode_steps": args.max_episode_steps,
            "steps": int(step_count.max()),
            "env_summaries": env_summaries,
            "final_info": jsonable(last_info),
            "action_space_low": np.asarray(env.action_space.low).tolist(),
            "action_space_high": np.asarray(env.action_space.high).tolist(),
        }
        return frames, summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", default="select_fruit")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vector-mode", choices=["sync", "subprocess"], default="sync")
    parser.add_argument("--debug-layers", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading RLinf OpenPI model from %s", args.checkpoint_dir)
    model = get_model(build_model_cfg(args.checkpoint_dir, args.num_steps))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    check_model_dims(model)
    LOGGER.info(
        "Loaded model on %s: config_name=%s action_dim=%s horizon=%s chunk=%s env_dim=%s",
        device,
        model.config.config_name,
        model.config.action_dim,
        model.config.action_horizon,
        model.config.action_chunk,
        model.config.action_env_dim,
    )

    all_summaries = []
    for episode_idx in range(args.episodes):
        LOGGER.info("Starting episode %d task=%s", episode_idx, args.task_name)
        frames, summary = run_episode(model, args, episode_idx)
        video_path = output_dir / f"episode_{episode_idx:03d}_{args.task_name}.mp4"
        json_path = output_dir / f"episode_{episode_idx:03d}_{args.task_name}.json"
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=1)
        summary["video_path"] = str(video_path)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        all_summaries.append(summary)
        LOGGER.info(
            "Episode %d done: env_success=%s env_rewards=%s steps=%d video=%s",
            episode_idx,
            [item["success"] for item in summary["env_summaries"]],
            [item["total_reward"] for item in summary["env_summaries"]],
            summary["steps"],
            video_path,
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"episodes": all_summaries}, f, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
