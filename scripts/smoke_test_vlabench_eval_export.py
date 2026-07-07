#!/usr/bin/env python3
"""Smoke tests for VLABench eval result export."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


def make_cfg(result_dir: Path, *, vector_mode: str):
    return SimpleNamespace(
        env_type="vlabench",
        task_name="select_fruit",
        task_names=["select_fruit", "select_drink"],
        robot="franka",
        num_envs=2,
        total_num_envs=2,
        group_size=1,
        seed=23,
        vector_mode=vector_mode,
        control_mode="ee",
        action_mode="absolute_ee",
        reward_mode="success",
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        auto_reset=False,
        max_episode_steps=2,
        max_steps_per_rollout_epoch=2,
        require_pcd=False,
        return_tensors=True,
        camera_id=2,
        use_extra_views=True,
        render_height=256,
        render_width=256,
        reset_wait_step=10,
        random_init=True,
        intention_score_threshold=0.1,
        vlabench_eval=SimpleNamespace(
            export_results=True,
            result_path=str(result_dir / "vlabench_results.jsonl"),
            summary_path=str(result_dir / "vlabench_summary.json"),
            summary_csv_path=str(result_dir / "vlabench_summary.csv"),
            export_format="jsonl",
        ),
        video_cfg=SimpleNamespace(save_video=False),
    )


def make_env(cfg):
    env_cls = get_env_cls(cfg.env_type, cfg)
    return env_cls(cfg=cfg, num_envs=cfg.num_envs, seed_offset=0, total_num_processes=1)


def check_exports(result_dir: Path):
    result_path = result_dir / "vlabench_results.jsonl"
    summary_path = result_dir / "vlabench_summary.json"
    csv_path = result_dir / "vlabench_summary.csv"
    assert result_path.exists(), result_path
    assert summary_path.exists(), summary_path
    assert csv_path.exists(), csv_path

    rows = []
    with result_path.open("r") as f:
        for line in f:
            rows.append(json.loads(line))
    assert len(rows) >= 2
    required = {
        "episode_id",
        "env_id",
        "task_name",
        "instruction",
        "episode_config_id",
        "success",
        "success_once",
        "episode_return",
        "elapsed_steps",
        "terminated",
        "truncated",
        "ik_failure_count",
        "final_reward",
        "vector_mode",
        "seed",
    }
    for row in rows:
        assert required.issubset(row.keys()), row.keys()
        assert row["task_name"] in {"select_fruit", "select_drink"}
        assert row["elapsed_steps"] == 2
        assert row["truncated"] is True

    summary = json.loads(summary_path.read_text())
    assert "overall" in summary and "tasks" in summary
    assert summary["overall"]["num_episodes"] >= 2
    assert "success_rate" in summary["overall"]
    assert "avg_episode_return" in summary["overall"]
    assert "avg_elapsed_steps" in summary["overall"]
    assert csv_path.read_text().splitlines()[0].startswith("task_name,num_episodes")
    return rows, summary


def run_case(name: str, vector_mode: str):
    print(f"[{name}] {vector_mode} export smoke")
    result_dir = Path("/tmp") / f"vlabench_eval_export_{vector_mode}"
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(result_dir, vector_mode=vector_mode)
    env = make_env(cfg)
    try:
        env.reset()
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 3, 7), dtype=np.float32)
        )
        assert rewards.shape == (2, 3)
        assert truncations[:, 1:].all().item()
        assert "episode" in infos[1] or "episode" in infos[-1]
        rows, summary = check_exports(result_dir)
        print("rows:", len(rows), "summary:", summary["overall"])
    finally:
        env.close()


def main():
    run_case("A", "sync")
    run_case("B", "subprocess")
    print("VLABench eval export smoke test passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
