# VLABench RLinf Integration

## Integration Summary

VLABench is integrated into RLinf as a Gym/Gymnasium-style embodied environment wrapper through `rlinf.envs.vlabench.VLABenchEnv`.

The wrapper exposes:

- `reset() -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `chunk_step(actions) -> (obs_list, rewards, terminations, truncations, infos_list)`
- `render(mode="rgb_array")`
- `close()`

It is registered as `env_type: vlabench` in `rlinf.envs.get_env_cls`, so RLinf EnvWorker and runner configs can construct it through the normal environment registry.

## Basic Usage

Use OSMesa on the current machine:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export VLABENCH_ROOT=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench/VLABench
export PYTHONPATH=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/RLinf:/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench:$PYTHONPATH
export EMBODIED_PATH=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/RLinf/examples/embodiment
```

A minimal environment config is available at:

```text
examples/embodiment/config/env/vlabench_select_fruit.yaml
```

## Supported Matrix

### Vector Modes

- `vector_mode: sync`
- `vector_mode: subprocess`

The sync mode is the default. Subprocess mode isolates VLABench instances in worker processes, but it is not guaranteed to improve throughput under OSMesa.

### Control / Action Modes

Supported combinations:

- `control_mode: ee`, `action_mode: absolute_ee`
- `control_mode: ee`, `action_mode: delta_ee`
- `control_mode: joint`, `action_mode: absolute_joint`

EE action format:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

Joint action format:

```text
[q1, q2, ..., qN, gripper_scalar]
```

The wrapper dynamically maps joint action to MuJoCo ctrl:

```text
policy_action_dim = qpos_dim + 1
ctrl_dim = qpos_dim + gripper_ctrl_dim
ctrl = concat(joint_qpos, expanded_gripper_ctrl)
```

For Franka/Panda, current validation observes:

```text
qpos_dim = 7
gripper_ctrl_dim = 2
model.nu = 9
policy joint_action_dim = 8
```

### Reward Modes

- `reward_mode: success`
- `reward_mode: success_plus_progress_delta`

The default remains success reward:

```python
reward = 1.0 if success else 0.0
```

Progress reward uses progress deltas rather than raw progress score:

```python
success_part = success_reward * float(success)
progress_delta = progress_t - progress_prev
if not progress_delta_negative:
    progress_delta = max(progress_delta, 0.0)
progress_delta = clip(progress_delta, progress_delta_clip_min, progress_delta_clip_max)
reward = success_part + progress_reward_coef * progress_delta - step_penalty - ik_failure_penalty * float(not ik_success)
```

If VLABench progress is unavailable, `progress_available=false` and progress reward contribution is zero.

### Benchmark / Eval

Supported task sources:

- `task_name`
- `task_names`
- `eval_track`
- `episode_config_path`

Sampling controls:

- `task_sample_mode: uniform | sequential`
- `episode_config_sample_mode: random | sequential`

### Result Export

Optional eval export supports:

- JSONL episode records
- summary JSON
- summary CSV

Configure through:

```yaml
vlabench_eval:
  export_results: true
  result_path: ${runner.logger.log_path}/vlabench_results.jsonl
  summary_path: ${runner.logger.log_path}/vlabench_summary.json
  summary_csv_path: ${runner.logger.log_path}/vlabench_summary.csv
  export_format: jsonl
```

## API Shape

### Reset

```python
obs, info = env.reset()
```

`obs` is a dict with:

- `main_images`: `[B, H, W, C]`, uint8, numpy or torch depending on `return_tensors`
- `extra_view_images`: optional `[B, N, H, W, C]`
- `states`: `[B, 7]`, float32
- `task_descriptions`: `list[str]`, length `B`

`task_descriptions` intentionally remains a Python list and is not converted to a tensor.

### Step

```python
obs, reward, terminated, truncated, info = env.step(action)
```

For single-env scalar input, reward and done values are Python scalars. For vector envs, reward and done values are batched arrays.

The wrapper keeps `terminated` and `truncated` separate. After an environment reaches done, a done latch prevents further real VLABench stepping until the next reset.

### Chunk Step

```python
obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(actions)
```

Supported shapes:

- single env: `[action_dim]`, `[1, action_dim]`, `[T, action_dim]`, `[1, T, action_dim]`
- vector env: `[B, action_dim]`, `[B, T, action_dim]`

Returned tensors:

- `rewards`: `[B, T]`
- `terminations`: `[B, T]`
- `truncations`: `[B, T]`
- `infos_list`: length `T`

## Gym Spaces

`VLABenchEnv` exposes minimal Gym/Gymnasium spaces:

- `action_space`: `Box(shape=(action_dim,))` for single env or `Box(shape=(B, action_dim))` for vector env
- `observation_space`: `Dict` with image, state, and task description entries

RLinf runner primarily relies on returned obs/action tensors rather than Gym sampling, but the spaces are present for Gym-style compatibility.

## Known Limitations

- OSMesa works on the current machine; EGL is not fixed.
- Subprocess vector mode improves isolation, not necessarily throughput.
- Reward model / VLM reward is not implemented.
- Raw VLABench reward is not used unless VLABench exposes a stable raw reward interface.
- Large-scale training performance is not guaranteed by smoke tests.
- Franka/Panda joint dimensions are validated; other robots are dynamically checked but not deeply benchmarked.
- Auto-reset and partial reset remain intentionally unsupported in this integration path.

## Final Validation Commands

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export VLABENCH_ROOT=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench/VLABench
export PYTHONPATH=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/RLinf:/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench:$PYTHONPATH
export EMBODIED_PATH=/inspire/hdd/global_user/yinlinqi-p-yinlinqi/RLinf/examples/embodiment

.venv/bin/python scripts/smoke_test_vlabench_rollout.py
.venv/bin/python scripts/smoke_test_vlabench_vector.py
.venv/bin/python scripts/smoke_test_vlabench_benchmark.py
.venv/bin/python scripts/smoke_test_vlabench_subprocess.py
.venv/bin/python scripts/smoke_test_vlabench_eval_export.py
.venv/bin/python scripts/smoke_test_vlabench_delta_ee.py
.venv/bin/python scripts/smoke_test_vlabench_joint_control.py
.venv/bin/python scripts/smoke_test_vlabench_action_behavior.py
.venv/bin/python scripts/smoke_test_vlabench_progress_reward.py
.venv/bin/python scripts/smoke_test_vlabench_final_integration.py

.venv/bin/python examples/embodiment/train_embodied_agent.py --config-name vlabench_eval_export_smoke
.venv/bin/python examples/embodiment/train_embodied_agent.py --config-name vlabench_delta_ee_smoke
.venv/bin/python examples/embodiment/train_embodied_agent.py --config-name vlabench_joint_control_smoke
.venv/bin/python examples/embodiment/train_embodied_agent.py --config-name vlabench_progress_reward_smoke
```
