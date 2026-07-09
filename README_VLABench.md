# VLABench Evaluation in RLinf

This document describes the current VLABench integration in RLinf: how to run it, what the wrapper does, what reward modes are available, what artifacts are saved, and what is still not fully validated.

## Status

VLABench is currently integrated as a lightweight RLinf evaluation/rollout target. The standard RLinf evaluation entrypoint works, official VLABench episode configs can be loaded, videos and metrics are saved, and the OpenPI/PyTorch policy path has explicit checkpoint/config/action/observation debug output.

Current verified scope:

- Standard entrypoint: `evaluations/run_eval.sh`.
- Single-env evaluation.
- Sync vector evaluation, including `num_envs=2`.
- Env-level subprocess vector smoke test.
- Official primitive in-distribution track loading via `track_1_in_distribution`.
- `select_fruit` 200-step sparse reward rollout.
- `select_fruit` 2-env shaped reward rollout with cumulative reward overlay.
- Video, `results.jsonl`, `summary.json`, `summary.csv`, config snapshot, action debug, observation debug, and checkpoint metadata export.

Not yet fully validated:

- Full primitive benchmark run: 10 tasks x 50 configs = 500 episodes.
- Other VLABench tracks beyond `track_1_in_distribution`.
- Long-running RLinf evaluation with `vector_mode=subprocess`.
- Exact per-episode mp4 file mapping. Results currently record the video directory such as `videos/seed_0`.
- The checkpoint currently reports one missing PaliGemma language embedding key during load; see [Checkpoint Notes](#checkpoint-notes).

## Files

Main RLinf-side files:

- `evaluations/vlabench/vlabench_pi0_primitive_eval.yaml`
- `evaluations/run_eval.sh`
- `evaluations/eval_embodied_agent.py`
- `rlinf/envs/vlabench/vlabench_env.py`
- `rlinf/envs/vlabench/utils.py`
- `rlinf/envs/vlabench/eval_utils.py`
- `rlinf/envs/action_utils.py`
- `rlinf/models/embodiment/openpi/__init__.py`
- `scripts/smoke_test_vlabench_eval_pipeline.py`

No VLABench source code is modified by this integration.

## Environment

From the RLinf repo:

```bash
cd /inspire/hdd/global_user/yinlinqi-p-yinlinqi/RLinf
source .venv/bin/activate
```

`evaluations/run_eval.sh` sets defaults for:

```bash
VLABENCH_REPO_PATH=${REPO_PATH}/../VLABench
VLABENCH_ROOT=${VLABENCH_REPO_PATH}/VLABench
VLABENCH_PI0_PRIMITIVE_CKPT=${REPO_PATH}/../checkpoints/pi0-primitive-10task-torch
PYTHONPATH=${VLABENCH_REPO_PATH}:${REPO_PATH}:${PYTHONPATH}
```

You can override them before running if needed:

```bash
export VLABENCH_ROOT=/path/to/VLABench/VLABench
export VLABENCH_PI0_PRIMITIVE_CKPT=/path/to/pi0-primitive-10task-torch
```

For headless rendering, use OSMesa if needed:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

## Basic Usage

Full configured primitive evaluation target:

```bash
bash evaluations/run_eval.sh vlabench vlabench_pi0_primitive_eval
```

The benchmark can also be inferred from the config name:

```bash
bash evaluations/run_eval.sh vlabench_pi0_primitive_eval
```

The default yaml is configured for primitive in-distribution evaluation:

- `eval_track: track_1_in_distribution`
- `task_names`: 10 primitive tasks
- `rollout_epoch: 50`
- `num_envs: 10`
- `vector_mode: subprocess`
- `max_episode_steps: 200`

This default intends to cover 10 tasks x 50 official configs = 500 episodes, but the full run has not yet been completed as part of the current validation.

## Lightweight Examples

### Single-env sparse success evaluation

```bash
bash evaluations/run_eval.sh vlabench vlabench_pi0_primitive_eval \
  env.eval.task_names=[select_fruit] \
  env.eval.rollout_epoch=1 \
  env.eval.total_num_envs=1 \
  env.eval.num_envs=1 \
  env.eval.group_size=1 \
  env.eval.vector_mode=sync \
  env.eval.max_episode_steps=200 \
  env.eval.max_steps_per_rollout_epoch=200 \
  env.eval.video_cfg.save_video=true \
  runner.logger.log_path=outputs/vlabench/select_fruit_200step_eval_video
```

Example verified output:

```text
outputs/vlabench/select_fruit_200step_eval_video/videos/seed_0/0.mp4
```

This run used sparse success reward and completed `select_fruit` successfully in 170 steps.

### Two-env shaped reward evaluation with cumulative reward overlay

```bash
bash evaluations/run_eval.sh vlabench vlabench_pi0_primitive_eval \
  env.eval.task_names=[select_fruit] \
  env.eval.rollout_epoch=1 \
  env.eval.total_num_envs=2 \
  env.eval.num_envs=2 \
  env.eval.group_size=2 \
  env.eval.vector_mode=sync \
  env.eval.max_episode_steps=200 \
  env.eval.max_steps_per_rollout_epoch=200 \
  env.eval.video_cfg.save_video=true \
  env.eval.reward_mode=success_plus_progress_delta \
  +env.eval.progress_reward_coef=0.5 \
  '+env.eval.video_cfg.extra_info_on_video=[episode_total_reward,progress_score,reward_progress,success]' \
  runner.logger.log_path=outputs/vlabench/select_fruit_env2_200step_shaping_cumreward_video
```

Example verified output:

```text
outputs/vlabench/select_fruit_env2_200step_shaping_cumreward_video/videos/seed_0/0.mp4
```

The extra video fields are:

- `episode_total_reward`: cumulative episode return up to the current frame.
- `progress_score`: VLABench task progress score if available.
- `reward_progress`: progress component of the current step reward.
- `success`: current task success flag. Some video overlay code may warn about `numpy.bool_`; this does not affect metrics or reward display.

## Wrapper Overview

`VLABenchEnv` adapts VLABench to RLinf/Gymnasium-style APIs:

- `reset()`
- `step()`
- `chunk_step()`
- `render()`
- `close()`

Supported vector modes:

- `vector_mode=sync`
- `vector_mode=subprocess`

The wrapper is responsible for:

- Importing VLABench without modifying VLABench source.
- Creating VLABench envs through `VLABench.envs.load_env`.
- Loading official episode configs from `episode_config_path` or `eval_track`.
- Sampling tasks and episode configs sequentially or randomly.
- Recording task name, instruction, episode config id, config hash, source path, and track.
- Converting VLABench observations into RLinf/OpenPI inputs.
- Converting OpenPI policy actions into VLABench MuJoCo control actions.
- Computing RLinf-side sparse or shaped rewards.
- Saving per-episode metrics and summaries.
- Writing debug artifacts for action, observation, and checkpoint/config loading.

## Episode Configs and Tracks

The evaluation yaml uses:

```yaml
env:
  eval:
    eval_track: track_1_in_distribution
    episode_config_path: null
    require_episode_config: true
    task_sample_mode: sequential
    episode_config_sample_mode: sequential
```

If `episode_config_path` is null and `eval_track` is set, RLinf resolves:

```text
$VLABENCH_ROOT/configs/evaluation/tracks/<eval_track>.json
```

For the default environment this resolves to:

```text
/inspire/hdd/global_user/yinlinqi-p-yinlinqi/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json
```

`require_episode_config: true` means RLinf will refuse to silently fall back to random VLABench resets if official configs cannot be loaded.

Each episode records an id like:

```text
/path/to/track_1_in_distribution.json:select_fruit:1:sha1=<config_hash>
```

## Action Interface

Current default action interface:

```yaml
control_mode: ee
action_mode: absolute_ee
action_dim: 7
```

The final policy action is interpreted as:

```text
[xyz_local, euler_rad, gripper]
```

Details:

- `xyz_local` is converted to world-space target position by adding `ee_frame_offset`.
- Euler angles are in radians and converted to quaternion before IK.
- The wrapper calls VLABench robot IK to get joint qpos control.
- Gripper uses `gripper_open_threshold` and `gripper_open_value`.
- Policy actions are treated as unnormalized final OpenPI outputs after OpenPI output transforms.
- No silent clipping is applied by default.

Action validation checks:

- action dimension matches expected env action dimension;
- action is finite;
- EE xyz and Euler values are within configured guard limits;
- step-to-step EE movement is not unusually large;
- gripper is in configured range;
- final MuJoCo control action has the expected dimension and finite values;
- actuator range is checked for actuator dimensions that declare a nonzero control range.

Relevant yaml block:

```yaml
action_validation:
  enabled: true
  max_abs_xyz: 2.0
  max_step_xyz_delta: 0.5
  max_abs_euler: 12.566370614359172
  max_step_euler_delta: 3.142592653589793
  gripper_min: -1.0
  gripper_max: 2.0
  allow_clip: false
```

## Observation Interface

The wrapper exposes observations with these keys:

- `main_images`: selected camera view, shape `[B, H, W, 3]`.
- `extra_view_images`: remaining camera views, shape `[B, N, H, W, 3]`.
- `states`: 7D state, shape `[B, 7]`.
- `task_descriptions`: language instruction list.

State semantics:

```text
xyz_local + euler + gripper
```

where:

```text
xyz_local = ee_pos - ee_frame_offset
```

The VLABench OpenPI policy expects at least two extra views. The current mapping is recorded in `debug/checkpoint_config_debug.json`.

## Reward Modes

### `success`

Default official-style evaluation reward:

```text
reward = 1.0 if task_success else 0.0
```

This is sparse. Moving closer to the object or making partial progress does not necessarily change reward. Official evaluation should primarily use success/progress/intention metrics, not shaped return.

### `success_plus_progress_delta`

Debug/training-style shaped reward:

```text
reward = success_reward * success
       + progress_reward_coef * progress_delta
       - step_penalty
       - ik_failure_penalty
```

Defaults:

```text
success_reward = 1.0
progress_reward_coef = 0.5 if set in override
step_penalty = 0.0
ik_failure_penalty = 0.0
```

Important semantics:

- This uses `progress_delta`, not raw distance to the object.
- If progress does not change on a step, the step reward is 0 even if the robot is moving closer visually.
- `episode_total_reward` is the cumulative reward within the episode. It can exceed 1 because progress shaping is added on top of success reward.
- For official benchmark reporting, use `success_rate`, `progress_score`, `intention_score`, and `ik_failure_rate` rather than shaped return.

## Output Layout

A typical output directory:

```text
outputs/vlabench/<run_name>/
  videos/
    seed_0/
      0.mp4
  debug/
    checkpoint_config_debug.json
    action_debug.jsonl
    observation_debug.json
  results.jsonl
  summary.json
  summary.csv
  config.yaml
  tensorboard/
```

`results.jsonl` contains one record per episode, including:

- `episode_id`
- `env_id`
- `task_name`
- `instruction`
- `episode_config_id`
- `episode_config_hash`
- `episode_config_source`
- `success`
- `success_once`
- `episode_return`
- `elapsed_steps`
- `terminated`
- `truncated`
- `ik_failure_rate`
- `progress_score`
- `intention_score`
- `final_progress_score`
- `video_path`
- `failure_reason`

`summary.json` contains:

- overall metrics;
- per-task metrics;
- track-level summary if `eval_track` is set;
- failed episode list;
- episode config path and eval track.

## Checkpoint Notes

The VLABench primitive checkpoint currently loads with:

```text
missing=1
unexpected=0
```

Missing key:

```text
paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
```

The rest of the checkpoint/config alignment has been verified in metadata:

```text
requested_config_name = pi0_ft_vlabench_primitive
actor_train_config_name = pi0_ft_vlabench_primitive
data_config_asset_id = vlabench/vlabench_ft_primitive
norm_stats_path = <checkpoint>/vlabench/vlabench_ft_primitive/norm_stats.json
action_env_dim = 7
action_chunk = 10
```

This missing key does not prevent inference or rollout, and `select_fruit` has completed successfully in a 200-step sparse reward run. However, it is still a checkpoint completeness question. Before reporting full benchmark results, confirm whether the missing embedding is intentionally initialized from a base model or whether the checkpoint export should include/remap that key.

## Smoke Tests

Run basic VLABench integration smoke tests:

```bash
.venv/bin/python scripts/smoke_test_vlabench_eval_pipeline.py \
  --tmp-dir /tmp/rlinf_vlabench_eval_smoke
```

Run smoke tests including the standard evaluation runner:

```bash
.venv/bin/python scripts/smoke_test_vlabench_eval_pipeline.py \
  --tmp-dir /tmp/rlinf_vlabench_eval_smoke_runner \
  --run-eval-runner
```

The smoke script checks:

- Hydra config validation;
- single env rollout;
- sync vector env rollout;
- subprocess vector env rollout;
- action validation guard;
- optional `evaluations/run_eval.sh` runner path;
- result/debug artifact creation.

## Known Warnings

VLABench currently emits this import-time logging error:

```text
TypeError: not all arguments converted during string formatting
```

It comes from VLABench source logging code and does not break RLinf evaluation. RLinf does not modify VLABench source, so the warning remains visible.
