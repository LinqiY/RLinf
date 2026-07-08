#!/usr/bin/env python3
"""EGL scale stress test for RLinf VLABench subprocess vector env."""

from __future__ import annotations

import ctypes.util
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
VLABENCH_REPO = WORKSPACE_ROOT / "VLABench"
VLABENCH_ROOT = VLABENCH_REPO / "VLABench"

os.environ.setdefault("VLABENCH_ROOT", str(VLABENCH_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
for path in (str(REPO_ROOT), str(VLABENCH_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rlinf.envs import get_env_cls  # noqa: E402


def print_env() -> None:
    print("[0] environment")
    for key in [
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_EGL_DEVICE_ID",
        "VLABENCH_ROOT",
        "PYTHONPATH",
    ]:
        print(f"{key}={os.environ.get(key)}")
    print("find_library(EGL)=", ctypes.util.find_library("EGL"))
    print("find_library(OpenGL)=", ctypes.util.find_library("OpenGL"))
    print("find_library(GLX)=", ctypes.util.find_library("GLX"))
    assert os.environ.get("MUJOCO_GL") == "egl", "MUJOCO_GL must be egl"
    assert os.environ.get("PYOPENGL_PLATFORM") == "egl", "PYOPENGL_PLATFORM must be egl"
    assert os.environ.get("VLABENCH_ROOT"), "VLABENCH_ROOT is required"


def check_minimal_mujoco_render() -> None:
    print("[1] minimal mujoco EGL render")
    child_code = r"""
import mujoco, numpy as np, os
xml = '<mujoco><worldbody><light pos="0 0 2"/><geom type="plane" size="1 1 .1"/><body pos="0 0 .2"><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    image = renderer.render()
print('child raw mujoco render ok', image.shape, image.dtype, int(image.mean()))
"""
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None:
        exact_env = os.environ.copy()
        probe = subprocess.run(
            [sys.executable, "-c", child_code],
            env=exact_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            print(probe.stdout.strip())
        else:
            print(
                "[warning] raw mujoco EGL probe failed with MUJOCO_EGL_DEVICE_ID="
                f"{os.environ.get('MUJOCO_EGL_DEVICE_ID')}; stderr follows."
            )
            print(probe.stderr.strip())
            print("[warning] unsetting MUJOCO_EGL_DEVICE_ID for the main stress process")
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="1 1 .1" rgba=".2 .3 .4 1"/>
        <body pos="0 0 .2">
          <geom type="sphere" size=".1" rgba="1 .2 .2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        image = renderer.render()
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    print("minimal render ok", image.shape, image.dtype, "mean=", int(image.mean()))
    print("effective MUJOCO_EGL_DEVICE_ID=", os.environ.get("MUJOCO_EGL_DEVICE_ID"))


def make_cfg(num_envs: int, **overrides):
    cfg = dict(
        env_type="vlabench",
        task_name=None,
        task_names=["select_fruit", "select_drink", "select_toy", "select_book"],
        robot="franka",
        num_envs=num_envs,
        total_num_envs=num_envs,
        group_size=1,
        seed=73,
        vector_mode="subprocess",
        control_mode="ee",
        action_mode="absolute_ee",
        reward_mode="success_plus_progress_delta",
        success_reward=1.0,
        progress_reward_coef=0.5,
        progress_delta_negative=False,
        progress_delta_clip_min=0.0,
        progress_delta_clip_max=1.0,
        step_penalty=0.0,
        ik_failure_penalty=0.0,
        task_sample_mode="sequential",
        episode_config_sample_mode="sequential",
        ee_frame_offset=[0.0, -0.4, 0.78],
        gripper_open_threshold=0.1,
        gripper_open_value=0.04,
        ignore_terminations=False,
        auto_reset=False,
        max_episode_steps=50,
        max_steps_per_rollout_epoch=20,
        require_pcd=False,
        return_tensors=True,
        camera_id=2,
        use_extra_views=True,
        render_height=128,
        render_width=128,
        reset_wait_step=10,
        random_init=True,
        intention_score_threshold=0.1,
        video_cfg=SimpleNamespace(save_video=False),
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def make_env(cfg):
    env_cls = get_env_cls(cfg.env_type, cfg)
    return env_cls(cfg=cfg, num_envs=cfg.num_envs, seed_offset=0, total_num_processes=1)


def assert_obs(obs, batch: int) -> None:
    assert isinstance(obs, dict)
    assert isinstance(obs["main_images"], torch.Tensor)
    assert obs["main_images"].shape[0] == batch, obs["main_images"].shape
    assert obs["main_images"].dtype == torch.uint8
    assert isinstance(obs["states"], torch.Tensor)
    assert obs["states"].shape == (batch, 7), obs["states"].shape
    assert obs["states"].dtype == torch.float32
    assert isinstance(obs["task_descriptions"], list)
    assert len(obs["task_descriptions"]) == batch
    assert all(isinstance(item, str) for item in obs["task_descriptions"])


def assert_info(info, batch: int) -> None:
    required = [
        "task_name",
        "instruction",
        "episode_config_id",
        "success",
        "success_once",
        "episode_return",
        "elapsed_steps",
        "terminated",
        "truncated",
        "ik_success",
        "reward_mode",
        "reward_success",
        "reward_progress",
        "progress_available",
    ]
    for key in required:
        assert key in info, key
    assert len(info["task_name"]) == batch
    assert len(info["instruction"]) == batch
    assert len(info["episode_config_id"]) == batch
    for key in ["success", "elapsed_steps", "episode_return", "terminated", "truncated"]:
        assert tuple(info[key].shape) == (batch,), (key, info[key].shape)
    assert all(isinstance(name, str) for name in info["task_name"])
    assert all(isinstance(text, str) for text in info["instruction"])


def assert_chunk(obs_list, rewards, terminations, truncations, infos, batch: int, steps: int) -> None:
    assert len(obs_list) == steps, len(obs_list)
    assert tuple(rewards.shape) == (batch, steps), rewards.shape
    assert tuple(terminations.shape) == (batch, steps), terminations.shape
    assert tuple(truncations.shape) == (batch, steps), truncations.shape
    assert len(infos) == steps
    assert_obs(obs_list[-1], batch)
    assert_info(infos[-1], batch)


def check_process_hint() -> None:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "stress_test_vlabench_egl_scale|VLABench|vlabench"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [line for line in out.stdout.splitlines() if "pgrep" not in line]
        print("[process hint] matching processes after close:")
        if lines:
            for line in lines[:20]:
                print(" ", line)
        else:
            print("  none")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print("[process hint] unable to inspect processes:", exc)


def run_scale(batch: int) -> dict:
    print(f"[2] VLABench subprocess scale num_envs={batch}")
    cfg = make_cfg(batch)
    env = make_env(cfg)
    timings: list[float] = []
    try:
        assert len(env._subproc_processes) == batch
        obs, info = env.reset()
        assert_obs(obs, batch)
        assert_info(info, batch)

        action = np.zeros((batch, 7), dtype=np.float32)
        start = time.perf_counter()
        for _ in range(20):
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            timings.append(time.perf_counter() - t0)
            assert_obs(obs, batch)
            assert_info(info, batch)
            assert np.asarray(reward).shape == (batch,), np.asarray(reward).shape
            assert np.asarray(terminated).shape == (batch,), np.asarray(terminated).shape
            assert np.asarray(truncated).shape == (batch,), np.asarray(truncated).shape
        total_step_time = time.perf_counter() - start

        obs_list, rewards, terminations, truncations, infos = env.chunk_step(action)
        assert_chunk(obs_list, rewards, terminations, truncations, infos, batch, 1)

        chunk_action = np.zeros((batch, 3, 7), dtype=np.float32)
        obs_list, rewards, terminations, truncations, infos = env.chunk_step(chunk_action)
        assert_chunk(obs_list, rewards, terminations, truncations, infos, batch, 3)

        image = env.render(mode="rgb_array", tile=True)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3
        print("render tile ok", image.shape, image.dtype)

        avg = float(np.mean(timings))
        fps = float(batch * len(timings) / total_step_time)
        result = {"num_envs": batch, "avg_step_time": avg, "fps": fps, "passed": True}
        print(f"num_envs={batch} passed avg_step_time={avg:.4f}s fps={fps:.2f}")
        return result
    finally:
        env.close()
        env.close()
        for proc in env._subproc_processes:
            if proc.is_alive():
                raise AssertionError(f"worker still alive after close pid={proc.pid}")
        check_process_hint()


def main() -> None:
    results = []
    try:
        print_env()
        check_minimal_mujoco_render()
        for batch in (2, 4, 8):
            results.append(run_scale(batch))
    except Exception:
        traceback.print_exc()
        raise
    finally:
        print("[summary]")
        for result in results:
            print(result)
    print("VLABench EGL scale stress test passed")


if __name__ == "__main__":
    main()
