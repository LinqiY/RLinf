# Copyright 2026 The RLinf Authors.
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
#
# Ported from the original OpenPi-VLABench training config
# (src/openpi/training/config.py: LeRobotVLABenchDataConfig) so RLinf loads
# the `vlabench/vlabench_ft_primitive` checkpoint with the same transforms it
# was trained with. In particular, the raw VLABench dataset stores *absolute*
# EE-pose actions (see VLABench/utils/rlds_builder.py: `trajectory[i]`); the
# model is trained on the *delta* (action - state) via `DeltaActions`, and at
# inference `AbsoluteActions` must add the state back to recover an absolute
# action before it is sent to `env.step`. Skipping this output transform (as
# happened when this checkpoint was loaded with the generic `pi0_libero`
# config) leaves the raw delta-space prediction unconverted, which is why the
# arm target could land on values like a ~2*pi Euler component and thrash.
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import vlabench_policy


@dataclasses.dataclass(frozen=True)
class LeRobotVLABenchDataConfig(DataConfigFactory):
    """Data config for the VLABench primitive pi0 checkpoints (franka, EE control)."""

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/second_image": "second_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                vlabench_policy.VLABenchInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[vlabench_policy.VLABenchOutputs()],
        )
        # Raw VLABench actions are absolute EE pose (xyz + euler + gripper).
        # Convert to delta (relative to state) for training; the matching
        # AbsoluteActions output transform reconstructs the absolute action
        # from the model's delta prediction at inference time. Gripper (last
        # dim) is left absolute.
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
