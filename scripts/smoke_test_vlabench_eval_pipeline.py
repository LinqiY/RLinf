#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
VLABENCH_REPO = WORKSPACE_ROOT / "VLABench"
VLABENCH_ROOT = VLABENCH_REPO / "VLABench"

os.environ.setdefault("EMBODIED_PATH", str(REPO_ROOT / "examples" / "embodiment"))
os.environ.setdefault("VLABENCH_REPO_PATH", str(VLABENCH_REPO))
os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_ROOT))
os.environ.setdefault(
    "VLABENCH_PI0_PRIMITIVE_CKPT",
    str(WORKSPACE_ROOT / "checkpoints" / "pi0-primitive-10task-torch"),
)
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
for path in (str(REPO_ROOT), str(VLABENCH_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rlinf.config import validate_cfg  # noqa: E402
from rlinf.envs.vlabench.vlabench_env import VLABenchEnv  # noqa: E402


def compose_eval_cfg(overrides: list[str] | None = None):
    config_dir = REPO_ROOT / "evaluations" / "vlabench"
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(
            config_name="vlabench_pi0_primitive_eval",
            overrides=overrides or [],
        )
    return validate_cfg(cfg)


def tiny_env_cfg(tmp_dir: Path, *, num_envs: int, vector_mode: str):
    cfg = compose_eval_cfg(
        [
            "env.eval.task_names=[select_fruit]",
            "env.eval.rollout_epoch=1",
            f"env.eval.total_num_envs={num_envs}",
            f"env.eval.num_envs={num_envs}",
            f"env.eval.group_size={num_envs}",
            f"env.eval.vector_mode={vector_mode}",
            "env.eval.max_episode_steps=2",
            "env.eval.max_steps_per_rollout_epoch=10",
            "env.eval.video_cfg.save_video=false",
            f"env.eval.vlabench_eval.result_path={tmp_dir / (vector_mode + '_results.jsonl')}",
            f"env.eval.vlabench_eval.summary_path={tmp_dir / (vector_mode + '_summary.json')}",
            f"env.eval.vlabench_eval.summary_csv_path={tmp_dir / (vector_mode + '_summary.csv')}",
            f"env.eval.vlabench_eval.debug_dir={tmp_dir / (vector_mode + '_debug')}",
        ]
    )
    return cfg.env.eval


def assert_episode_config_id(info):
    value = info["episode_config_id"]
    if isinstance(value, list):
        ids = value
    else:
        ids = [value]
    assert all(item is not None and "sha1=" in str(item) for item in ids), ids


def run_env_smoke(tmp_dir: Path, *, num_envs: int, vector_mode: str):
    env_cfg = tiny_env_cfg(tmp_dir, num_envs=num_envs, vector_mode=vector_mode)
    env = VLABenchEnv(env_cfg, num_envs=num_envs)
    try:
        obs, info = env.reset()
        assert_episode_config_id(info)
        assert obs["main_images"].shape[0] == num_envs
        assert obs["states"].shape == (num_envs, 7)
        assert obs["extra_view_images"] is not None
        assert obs["extra_view_images"].shape[1] >= 2
        action = np.asarray(obs["states"], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert_episode_config_id(info)
        _, _, _, _, _ = env.step(action)
    finally:
        env.close()

    summary_path = tmp_dir / f"{vector_mode}_summary.json"
    assert summary_path.exists(), summary_path
    summary = json.loads(summary_path.read_text())
    assert summary["overall"]["num_episodes"] >= 1
    assert summary["episode_config_path"]
    assert "failed_episodes" in summary


def run_action_guard_smoke(tmp_dir: Path):
    env_cfg = tiny_env_cfg(tmp_dir, num_envs=1, vector_mode="sync")
    env = VLABenchEnv(env_cfg, num_envs=1)
    try:
        env.reset()
        bad = np.zeros((7,), dtype=np.float32)
        bad[0] = 99.0
        try:
            env.step(bad)
        except ValueError as exc:
            assert "xyz" in str(exc) or "action" in str(exc)
        else:
            raise AssertionError("VLABench action guard did not reject an invalid EE action")
    finally:
        env.close()


def run_eval_runner_smoke(tmp_dir: Path):
    env = os.environ.copy()
    env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    cmd = [
        "bash",
        "evaluations/run_eval.sh",
        "vlabench",
        "vlabench_pi0_primitive_eval",
        "env.eval.task_names=[select_fruit]",
        "env.eval.rollout_epoch=1",
        "env.eval.total_num_envs=1",
        "env.eval.num_envs=1",
        "env.eval.group_size=1",
        "env.eval.vector_mode=sync",
        "env.eval.max_episode_steps=10",
        "env.eval.max_steps_per_rollout_epoch=10",
        "env.eval.video_cfg.save_video=false",
        f"runner.logger.log_path={tmp_dir / 'eval_runner'}",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    assert (tmp_dir / "eval_runner" / "summary.json").exists()
    assert (tmp_dir / "eval_runner" / "results.jsonl").exists()
    assert (tmp_dir / "eval_runner" / "debug" / "checkpoint_config_debug.json").exists()
    assert (tmp_dir / "eval_runner" / "config.yaml").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp-dir", default="/tmp/rlinf_vlabench_eval_smoke")
    parser.add_argument("--run-eval-runner", action="store_true")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_eval_cfg()
    assert cfg.env.eval.env_type == "vlabench"
    assert cfg.env.eval.require_episode_config
    assert cfg.rollout.model.openpi.config_name == "pi0_ft_vlabench_primitive"
    assert cfg.env.eval.max_steps_per_rollout_epoch % cfg.rollout.model.num_action_chunks == 0
    print("config validation smoke passed")

    run_env_smoke(tmp_dir, num_envs=1, vector_mode="sync")
    print("single env smoke passed")
    run_env_smoke(tmp_dir, num_envs=2, vector_mode="sync")
    print("sync vector env smoke passed")
    run_env_smoke(tmp_dir, num_envs=2, vector_mode="subprocess")
    print("subprocess vector env smoke passed")
    run_action_guard_smoke(tmp_dir)
    print("action validation smoke passed")

    if args.run_eval_runner:
        run_eval_runner_smoke(tmp_dir)
        print("evaluation runner smoke passed")


if __name__ == "__main__":
    main()
