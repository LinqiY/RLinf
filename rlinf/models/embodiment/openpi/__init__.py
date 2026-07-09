# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import json
import os
import pathlib

import torch
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

_LOGGER = get_logger()


def _transform_names(transforms_list):
    return [type(item).__name__ for item in transforms_list]


def _write_model_debug_metadata(path, payload):
    if not path:
        return
    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)

    model_path_str = str(cfg.model_path)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )
    _LOGGER.info(f"[openpi.get_model] config_name (requested)      = {config_name!r}")
    _LOGGER.info(f"[openpi.get_model] actor_train_config.name       = {actor_train_config.name!r}")
    _LOGGER.info(f"[openpi.get_model] model_path                    = {model_path_str!r}")

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    _LOGGER.info(
        "[openpi.get_model] model config: "
        f"action_dim={actor_model_config.action_dim} "
        f"action_horizon={actor_model_config.action_horizon} "
        f"max_token_len={actor_model_config.max_token_len} "
        f"pi05={actor_model_config.pi05} "
        f"action_chunk={actor_model_config.action_chunk} "
        f"action_env_dim={actor_model_config.action_env_dim}"
    )

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Guard against silently loading a checkpoint with a mismatched openpi
    # config (e.g. a VLABench checkpoint loaded with `pi0_libero`/`pi0_maniskill`
    # /`pi0_metaworld`). This previously caused the model's raw delta-space
    # action prediction to be returned unconverted (no AbsoluteActions), which
    # produced physically invalid actions (e.g. an Euler component near 2*pi).
    on_disk_asset_dirs = sorted(
        glob.glob(os.path.join(checkpoint_dir, "assets", "*", "*"))
        + glob.glob(os.path.join(checkpoint_dir, "*", "*", "norm_stats.json"))
    )
    looks_like_vlabench = any("vlabench" in path.lower() for path in on_disk_asset_dirs)
    if looks_like_vlabench and "vlabench" not in str(config_name).lower():
        raise ValueError(
            f"Refusing to load checkpoint at {checkpoint_dir!r} with "
            f"openpi.config_name={config_name!r}. The checkpoint directory "
            f"contains VLABench asset paths ({on_disk_asset_dirs}) but the "
            "requested config name does not reference vlabench (it would "
            "fall back to pi0_libero/pi0_maniskill/pi0_metaworld transforms "
            "and possibly mismatched norm stats). Use a config such as "
            "'pi0_ft_vlabench_primitive' instead."
        )

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        weight_source = full_weights_path
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        load_result = model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        weight_source = actor_full_weights_path
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        load_result = model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        weight_source = weight_paths
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        load_result = model.load_state_dict(all_state_dict, strict=False)

    num_model_params = sum(1 for _ in model.state_dict())
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    _LOGGER.info(f"[openpi.get_model] weight_source                = {weight_source!r}")
    _LOGGER.info(
        f"[openpi.get_model] load_state_dict: missing={len(missing_keys)} "
        f"unexpected={len(unexpected_keys)} out of {num_model_params} model params"
    )
    if missing_keys:
        _LOGGER.warning(
            f"[openpi.get_model] missing_keys (first 20 of {len(missing_keys)}): "
            f"{missing_keys[:20]}"
        )
    if unexpected_keys:
        _LOGGER.warning(
            f"[openpi.get_model] unexpected_keys (first 20 of {len(unexpected_keys)}): "
            f"{unexpected_keys[:20]}"
        )
    # Missing/unexpected keys are expected for value heads, DSRL modules, etc.
    # that are intentionally absent from the base checkpoint, but a large
    # fraction of missing keys means the checkpoint essentially failed to
    # load (e.g. wrong architecture/config) and silently falling back to
    # randomly-initialized weights is worse than failing loudly.
    if num_model_params > 0 and len(missing_keys) / num_model_params > 0.5:
        raise RuntimeError(
            f"load_state_dict left {len(missing_keys)}/{num_model_params} model "
            f"parameters missing when loading {weight_source!r}. Refusing to "
            "continue with a mostly-uninitialized model."
        )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    _LOGGER.info(
        "[openpi.get_model] data_config: "
        f"asset_id={data_config.asset_id!r} repo_id={data_config.repo_id!r} "
        f"use_quantile_norm={data_config.use_quantile_norm}"
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # Load checkpoint-local stats so inference uses the training normalization.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats_path = os.path.join(
            checkpoint_dir, data_config.asset_id, "norm_stats.json"
        )
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"norm_stats.json not found at {norm_stats_path!r} for "
                f"asset_id={data_config.asset_id!r}. Refusing to fall back to "
                "default/uninitialized normalization stats."
            )
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    _LOGGER.info(f"[openpi.get_model] norm_stats_path              = {norm_stats_path!r}")
    for key in ("state", "actions"):
        if key in norm_stats:
            stat = norm_stats[key]
            _LOGGER.info(
                f"[openpi.get_model] norm_stats[{key}] mean[:7]={list(stat.mean[:7])} "
                f"std[:7]={list(stat.std[:7])} "
                f"q01[:7]={None if stat.q01 is None else list(stat.q01[:7])} "
                f"q99[:7]={None if stat.q99 is None else list(stat.q99[:7])}"
            )
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    # Expose the individual output-transform stages (for batch_size==1 debug
    # logging only) so callers can inspect the action at each layer:
    # raw model action -> model_transforms.outputs -> Unnormalize ->
    # data_transforms.outputs -> repack_transforms.outputs (final env action).
    model.debug_output_stages = [
        ("model_transforms.outputs", transforms.compose(data_config.model_transforms.outputs)),
        (
            "unnormalize",
            transforms.compose(
                [transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)]
            ),
        ),
        ("data_transforms.outputs", transforms.compose(data_config.data_transforms.outputs)),
        ("repack_transforms.outputs", transforms.compose(repack_transforms.outputs)),
    ]
    debug_metadata = {
        "model_path": model_path_str,
        "checkpoint_dir": checkpoint_dir,
        "weight_source": weight_source,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "requested_config_name": config_name,
        "actor_train_config_name": actor_train_config.name,
        "data_config_asset_id": data_config.asset_id,
        "data_config_repo_id": data_config.repo_id,
        "norm_stats_path": norm_stats_path,
        "action_dim": actor_model_config.action_dim,
        "action_horizon": actor_model_config.action_horizon,
        "action_chunk": actor_model_config.action_chunk,
        "action_env_dim": actor_model_config.action_env_dim,
        "state_dim": actor_model_config.action_dim,
        "image_key_mapping": {
            "main_images": "observation/image -> base_0_rgb",
            "extra_view_images[:, 0]": "observation/second_image -> left_wrist_0_rgb",
            "extra_view_images[:, -1]": "observation/wrist_image -> right_wrist_0_rgb",
        },
        "state_key_mapping": {"states": "observation/state -> padded OpenPI state"},
        "action_key_mapping": {
            "raw_model_action": "padded delta action",
            "final_action": "first 7 dims xyz_local + euler(rad) + gripper",
        },
        "input_transforms": _transform_names(repack_transforms.inputs)
        + _transform_names(data_config.data_transforms.inputs)
        + ["Normalize"]
        + _transform_names(data_config.model_transforms.inputs),
        "output_transforms": _transform_names(data_config.model_transforms.outputs)
        + ["Unnormalize"]
        + _transform_names(data_config.data_transforms.outputs)
        + _transform_names(repack_transforms.outputs),
    }
    model.vlabench_debug_metadata = debug_metadata
    _write_model_debug_metadata(getattr(cfg, "debug_metadata_path", None), debug_metadata)

    return model
