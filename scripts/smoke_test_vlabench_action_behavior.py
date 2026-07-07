#!/usr/bin/env python3
"""Unified VLABench action-behavior validation.

Validates that the *physical* behavior of each supported (control_mode,
action_mode) pair matches its documented semantics -- not just that shapes
and runner smoke tests pass. Covers:

  - control_mode=ee,    action_mode=absolute_ee
  - control_mode=ee,    action_mode=delta_ee
  - control_mode=joint, action_mode=absolute_joint

For each mode this script drives the real MuJoCo simulation with concrete
actions (move +/-x/y/z, perturb individual joints, open/close gripper),
reads back the resulting pose/qpos/gripper state, and checks that the
observed change is in the expected direction. It also saves PNG frames (and
best-effort mp4) per test segment under /tmp/vlabench_action_behavior/, and
prints a final pass/warn/fail report.

This script does not touch the training runner and does not modify action
implementation; it is a read-only diagnostic on top of the existing
VLABenchEnv wrapper.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

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
from rlinf.envs.vlabench.utils import (  # noqa: E402
    ee_action_to_ctrl,
    ee_state_to_policy_state,
    get_joint_control_dims,
    joint_action_to_ctrl,
)

OUT_ROOT = Path("/tmp/vlabench_action_behavior")
TASK_NAME = "select_fruit"
ROBOT = "franka"
STEP_MOVE = 0.03
STEP_JOINT = 0.03
REPEATS = 8
DELTA_POSITION_CLIP = 0.05  # default clip; 0.03 must not be clipped
GRIPPER_OPEN_SCALAR = 1.0
GRIPPER_CLOSE_SCALAR = -1.0


@dataclass
class TestResult:
    control_mode: str
    action_mode: str
    test_name: str
    verdict: str  # "PASS", "WARN", "FAIL"
    detail: str
    observed_delta: Optional[np.ndarray] = None
    ik_success: int = 0
    ik_failure: int = 0
    save_path: Optional[str] = None


ALL_RESULTS: list[TestResult] = []


def make_cfg(control_mode: str, action_mode: str, **overrides):
    cfg = dict(
        env_type="vlabench",
        task_name=TASK_NAME,
        robot=ROBOT,
        num_envs=1,
        total_num_envs=1,
        group_size=1,
        seed=101,
        vector_mode="sync",
        control_mode=control_mode,
        action_mode=action_mode,
        reward_mode="success",
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
        delta_position_scale=1.0,
        delta_rotation_scale=1.0,
        delta_position_clip=DELTA_POSITION_CLIP,
        delta_rotation_clip=0.25,
        joint_action_dim=None,
        joint_position_low=None,
        joint_position_high=None,
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        auto_reset=False,
        max_episode_steps=80,
        max_steps_per_rollout_epoch=80,
        require_pcd=False,
        return_tensors=False,
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


def save_frames(frames: list[np.ndarray], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"png_count": 0, "mp4_path": None, "mp4_error": None, "dir": str(out_dir)}
    try:
        from PIL import Image

        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(out_dir / f"frame_{idx:03d}.png")
        result["png_count"] = len(frames)
    except Exception:
        result["png_error"] = traceback.format_exc(limit=2)
    try:
        import imageio

        mp4_path = out_dir / "clip.mp4"
        imageio.mimsave(str(mp4_path), frames, fps=4)
        result["mp4_path"] = str(mp4_path)
    except Exception:
        result["mp4_error"] = traceback.format_exc(limit=2)
    return result


def get_local_ee_pose(env) -> np.ndarray:
    """Return [x, y, z, roll, pitch, yaw, gripper] in the local (offset) frame."""
    raw_obs = env.last_raw_obs[0]
    return ee_state_to_policy_state(raw_obs, env.ee_frame_offset)


def get_current_qpos(env) -> np.ndarray:
    raw_env = env.envs[0]
    return np.asarray(raw_env.robot.get_qpos(raw_env.physics), dtype=np.float32).reshape(-1)


def record(control_mode, action_mode, test_name, verdict, detail, observed_delta=None,
           ik_success=0, ik_failure=0, save_path=None):
    result = TestResult(
        control_mode=control_mode,
        action_mode=action_mode,
        test_name=test_name,
        verdict=verdict,
        detail=detail,
        observed_delta=observed_delta,
        ik_success=ik_success,
        ik_failure=ik_failure,
        save_path=save_path,
    )
    ALL_RESULTS.append(result)
    print(f"  [{verdict}] {test_name}: {detail}")
    return result


# ---------------------------------------------------------------------------
# absolute_ee / delta_ee direction tests
# ---------------------------------------------------------------------------

DIRECTIONS = {
    "pos_x": (np.array([STEP_MOVE, 0.0, 0.0]), 0, +1),
    "neg_x": (np.array([-STEP_MOVE, 0.0, 0.0]), 0, -1),
    "pos_y": (np.array([0.0, STEP_MOVE, 0.0]), 1, +1),
    "neg_y": (np.array([0.0, -STEP_MOVE, 0.0]), 1, -1),
    "pos_z": (np.array([0.0, 0.0, STEP_MOVE]), 2, +1),
    "neg_z": (np.array([0.0, 0.0, -STEP_MOVE]), 2, -1),
}


def run_ee_direction_tests(control_mode: str, action_mode: str) -> None:
    print(f"\n=== {control_mode}/{action_mode}: direction tests ===")
    cfg = make_cfg(control_mode, action_mode)
    env = make_env(cfg)
    try:
        for name, (delta, axis, expected_sign) in DIRECTIONS.items():
            env.reset()
            initial = get_local_ee_pose(env)
            frames = [env.render(env_idx=0)]
            ik_ok, ik_fail = 0, 0
            for _ in range(REPEATS):
                if action_mode == "absolute_ee":
                    target_local = initial[:3] + delta
                    action = np.concatenate(
                        [target_local, initial[3:6], [GRIPPER_OPEN_SCALAR]]
                    ).astype(np.float32)
                else:  # delta_ee
                    action = np.concatenate(
                        [delta, np.zeros(3, dtype=np.float32), [GRIPPER_OPEN_SCALAR]]
                    ).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                ik_success = info.get("ik_success")
                if ik_success is not None:
                    if ik_success:
                        ik_ok += 1
                    else:
                        ik_fail += 1
                frames.append(env.render(env_idx=0))

            final = get_local_ee_pose(env)
            observed_delta = final[:3] - initial[:3]
            save_info = save_frames(frames, OUT_ROOT / f"{action_mode}_move_{name}")

            observed_axis = observed_delta[axis]
            if abs(observed_axis) < 1e-4:
                verdict = "WARN"
                detail = f"observed_delta={observed_delta.tolist()} (near-zero response on axis {axis})"
            elif np.sign(observed_axis) == expected_sign:
                verdict = "PASS"
                detail = f"observed_delta={observed_delta.tolist()} (direction correct)"
            else:
                verdict = "FAIL"
                detail = (
                    f"observed_delta={observed_delta.tolist()} "
                    f"(AXIS REVERSED: expected sign {expected_sign} on axis {axis})"
                )
            record(
                control_mode, action_mode, f"move_{name}", verdict, detail,
                observed_delta=observed_delta, ik_success=ik_ok, ik_failure=ik_fail,
                save_path=save_info["dir"],
            )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# absolute_joint direction tests
# ---------------------------------------------------------------------------

def run_joint_direction_tests(num_joints_to_test: int = 3) -> None:
    print("\n=== joint/absolute_joint: direction tests ===")
    cfg = make_cfg("joint", "absolute_joint")
    env = make_env(cfg)
    try:
        env.reset()
        qpos_dim, ctrl_dim, gripper_ctrl_dim = get_joint_control_dims(env.envs[0])
        print(f"  joint dims: qpos_dim={qpos_dim} gripper_ctrl_dim={gripper_ctrl_dim} ctrl_dim={ctrl_dim}")
        n_joints = min(num_joints_to_test, qpos_dim)

        responsive_joints = 0
        for j in range(n_joints):
            for sign_name, sign in (("pos", +1), ("neg", -1)):
                env.reset()
                initial_qpos = get_current_qpos(env)
                target_qpos = initial_qpos.copy()
                target_qpos[j] += sign * STEP_JOINT
                action = np.concatenate(
                    [target_qpos, [GRIPPER_OPEN_SCALAR]]
                ).astype(np.float32)

                frames = [env.render(env_idx=0)]
                for _ in range(REPEATS):
                    env.step(action)
                    frames.append(env.render(env_idx=0))

                final_qpos = get_current_qpos(env)
                observed_delta_q = final_qpos - initial_qpos
                save_info = save_frames(frames, OUT_ROOT / f"joint_q{j}_{sign_name}")

                observed = observed_delta_q[j]
                if abs(observed) < 1e-4:
                    verdict = "WARN"
                    detail = f"observed_delta_q[{j}]={observed:.5f} (weak/no response)"
                elif np.sign(observed) == sign:
                    verdict = "PASS"
                    detail = f"observed_delta_q[{j}]={observed:.5f} (direction correct)"
                    responsive_joints += 1
                else:
                    verdict = "FAIL"
                    detail = f"observed_delta_q[{j}]={observed:.5f} (AXIS REVERSED: expected sign {sign})"
                record(
                    "joint", "absolute_joint", f"joint_q{j}_{sign_name}", verdict, detail,
                    observed_delta=observed_delta_q, save_path=save_info["dir"],
                )

        if responsive_joints == 0:
            record(
                "joint", "absolute_joint", "joint_overall_responsiveness", "FAIL",
                "all tested joints showed no directional response",
            )
        else:
            record(
                "joint", "absolute_joint", "joint_overall_responsiveness", "PASS",
                f"{responsive_joints}/{n_joints * 2} joint direction tests responded correctly",
            )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# gripper adapter + physics validation
# ---------------------------------------------------------------------------

def run_gripper_test_ee(control_mode: str, action_mode: str) -> None:
    print(f"\n=== {control_mode}/{action_mode}: gripper test ===")
    cfg = make_cfg(control_mode, action_mode)
    env = make_env(cfg)
    try:
        env.reset()
        raw_env = env.envs[0]

        # adapter-level check (no stepping): must produce correctly-sized,
        # correctly-valued gripper ctrl for open vs close scalars.
        pose = get_local_ee_pose(env)
        open_action = np.concatenate([pose[:6], [GRIPPER_OPEN_SCALAR]]).astype(np.float32)
        close_action = np.concatenate([pose[:6], [GRIPPER_CLOSE_SCALAR]]).astype(np.float32)
        open_ctrl, open_ik = ee_action_to_ctrl(
            raw_env, open_action, ee_frame_offset=env.ee_frame_offset,
            gripper_open_threshold=env.gripper_open_threshold,
            gripper_open_value=env.gripper_open_value, action_mode="absolute_ee",
        )
        close_ctrl, close_ik = ee_action_to_ctrl(
            raw_env, close_action, ee_frame_offset=env.ee_frame_offset,
            gripper_open_threshold=env.gripper_open_threshold,
            gripper_open_value=env.gripper_open_value, action_mode="absolute_ee",
        )
        open_gripper_ctrl = open_ctrl[-2:]
        close_gripper_ctrl = close_ctrl[-2:]
        adapter_ok = (
            open_gripper_ctrl.shape == (2,)
            and close_gripper_ctrl.shape == (2,)
            and np.allclose(open_gripper_ctrl, env.gripper_open_value)
            and np.allclose(close_gripper_ctrl, 0.0)
        )
        if not adapter_ok:
            record(
                control_mode, action_mode, "gripper_adapter", "FAIL",
                f"open_ctrl={open_gripper_ctrl.tolist()} close_ctrl={close_gripper_ctrl.tolist()} "
                f"(expected open={env.gripper_open_value}, close=0.0, dim=2)",
            )
        else:
            record(
                control_mode, action_mode, "gripper_adapter", "PASS",
                f"open_ctrl={open_gripper_ctrl.tolist()} close_ctrl={close_gripper_ctrl.tolist()}",
            )

        # physics-level: drive open then close, observe ee_state gripper feature.
        frames = [env.render(env_idx=0)]
        for _ in range(REPEATS):
            env.step(open_action)
            frames.append(env.render(env_idx=0))
        gripper_after_open = get_local_ee_pose(env)[6]
        for _ in range(REPEATS):
            env.step(close_action)
            frames.append(env.render(env_idx=0))
        gripper_after_close = get_local_ee_pose(env)[6]
        save_info = save_frames(frames, OUT_ROOT / f"gripper_{action_mode}")

        if abs(gripper_after_open - gripper_after_close) < 1e-4:
            record(
                control_mode, action_mode, "gripper_physics", "WARN",
                f"gripper feature open={gripper_after_open:.4f} close={gripper_after_close:.4f} "
                "(physics state did not change noticeably)",
                save_path=save_info["dir"],
            )
        else:
            record(
                control_mode, action_mode, "gripper_physics", "PASS",
                f"gripper feature open={gripper_after_open:.4f} close={gripper_after_close:.4f} (state changed)",
                save_path=save_info["dir"],
            )
    finally:
        env.close()


def run_gripper_test_joint() -> None:
    print("\n=== joint/absolute_joint: gripper test ===")
    cfg = make_cfg("joint", "absolute_joint")
    env = make_env(cfg)
    try:
        env.reset()
        raw_env = env.envs[0]
        qpos = get_current_qpos(env)

        open_action = np.concatenate([qpos, [GRIPPER_OPEN_SCALAR]]).astype(np.float32)
        close_action = np.concatenate([qpos, [GRIPPER_CLOSE_SCALAR]]).astype(np.float32)
        open_ctrl, _ = joint_action_to_ctrl(
            raw_env, open_action,
            gripper_open_threshold=env.gripper_open_threshold,
            gripper_open_value=env.gripper_open_value,
        )
        close_ctrl, _ = joint_action_to_ctrl(
            raw_env, close_action,
            gripper_open_threshold=env.gripper_open_threshold,
            gripper_open_value=env.gripper_open_value,
        )
        qpos_dim, _, gripper_ctrl_dim = get_joint_control_dims(raw_env)
        open_gripper_ctrl = open_ctrl[qpos_dim:]
        close_gripper_ctrl = close_ctrl[qpos_dim:]
        adapter_ok = (
            open_gripper_ctrl.shape == (gripper_ctrl_dim,)
            and close_gripper_ctrl.shape == (gripper_ctrl_dim,)
            and np.allclose(open_gripper_ctrl, env.gripper_open_value)
            and np.allclose(close_gripper_ctrl, 0.0)
        )
        if not adapter_ok:
            record(
                "joint", "absolute_joint", "gripper_adapter", "FAIL",
                f"open_ctrl={open_gripper_ctrl.tolist()} close_ctrl={close_gripper_ctrl.tolist()} "
                f"(expected open={env.gripper_open_value}, close=0.0, dim={gripper_ctrl_dim})",
            )
        else:
            record(
                "joint", "absolute_joint", "gripper_adapter", "PASS",
                f"open_ctrl={open_gripper_ctrl.tolist()} close_ctrl={close_gripper_ctrl.tolist()} "
                f"(gripper_ctrl_dim={gripper_ctrl_dim}, matches EE-mode semantics)",
            )

        frames = [env.render(env_idx=0)]
        for _ in range(REPEATS):
            env.step(open_action)
            frames.append(env.render(env_idx=0))
        gripper_after_open = get_local_ee_pose(env)[6]
        for _ in range(REPEATS):
            env.step(close_action)
            frames.append(env.render(env_idx=0))
        gripper_after_close = get_local_ee_pose(env)[6]
        save_info = save_frames(frames, OUT_ROOT / "gripper_absolute_joint")

        if abs(gripper_after_open - gripper_after_close) < 1e-4:
            record(
                "joint", "absolute_joint", "gripper_physics", "WARN",
                f"gripper feature open={gripper_after_open:.4f} close={gripper_after_close:.4f} "
                "(physics state did not change noticeably)",
                save_path=save_info["dir"],
            )
        else:
            record(
                "joint", "absolute_joint", "gripper_physics", "PASS",
                f"gripper feature open={gripper_after_open:.4f} close={gripper_after_close:.4f} (state changed)",
                save_path=save_info["dir"],
            )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# IK diagnostics + final report
# ---------------------------------------------------------------------------

def compute_ik_summary(control_mode: str, action_mode: str) -> tuple[int, int, float]:
    attempts = sum(
        r.ik_success + r.ik_failure for r in ALL_RESULTS
        if r.control_mode == control_mode and r.action_mode == action_mode
    )
    failures = sum(
        r.ik_failure for r in ALL_RESULTS
        if r.control_mode == control_mode and r.action_mode == action_mode
    )
    rate = (failures / attempts) if attempts > 0 else 0.0
    return attempts, failures, rate


def print_final_report() -> bool:
    print("\n" + "=" * 78)
    print("VLABench action behavior validation -- final report")
    print("=" * 78)

    overall_fail = False
    for control_mode, action_mode in (("ee", "absolute_ee"), ("ee", "delta_ee"), ("joint", "absolute_joint")):
        rows = [r for r in ALL_RESULTS if r.control_mode == control_mode and r.action_mode == action_mode]
        if not rows:
            continue
        print(f"\n--- {control_mode}/{action_mode} ---")
        for r in rows:
            print(f"  [{r.verdict}] {r.test_name}: {r.detail}")
            if r.save_path:
                print(f"           frames: {r.save_path}")
            if r.verdict == "FAIL":
                overall_fail = True

        if action_mode in ("absolute_ee", "delta_ee"):
            attempts, failures, rate = compute_ik_summary(control_mode, action_mode)
            print(f"  IK: attempts={attempts} failures={failures} failure_rate={rate:.2f}")
            if rate > 0.8:
                print("  [FAIL] IK failure rate exceeds 0.8")
                overall_fail = True
            elif failures > 0:
                print("  [WARN] occasional IK failures observed")

    no_frames_saved = all(r.save_path is None for r in ALL_RESULTS)
    if no_frames_saved:
        print("\n[FAIL] no visualization frames were saved by any test")
        overall_fail = True

    print("\n" + "=" * 78)
    print(f"Output directory: {OUT_ROOT}")
    print("OVERALL RESULT:", "FAIL" if overall_fail else "PASS")
    print("=" * 78)
    return not overall_fail


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_ee_direction_tests("ee", "absolute_ee")
    run_ee_direction_tests("ee", "delta_ee")
    run_joint_direction_tests(num_joints_to_test=3)

    run_gripper_test_ee("ee", "absolute_ee")
    run_gripper_test_ee("ee", "delta_ee")
    run_gripper_test_joint()

    ok = print_final_report()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
