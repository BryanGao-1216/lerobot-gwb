# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.parser import normalize_cli_aliases


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


def test_dataset_config_rlds_defaults_are_valid():
    config = DatasetConfig(repo_id="actionmem_mix")

    assert config.rlds_data_mix is None
    assert config.rlds_camera_views == ("primary", "secondary", "wrist")
    assert config.rlds_storage_format == "auto"
    assert config.rlds_action_transform == "oxe"
    assert config.rlds_target_control_hz == 10.0
    assert not hasattr(config, "rlds_backend_path")


def test_dataset_config_accepts_hybrid_rlds_storage():
    config = DatasetConfig(repo_id="actionmem_mix", rlds_storage_format="hybrid")

    assert config.rlds_storage_format == "hybrid"


def test_rlds_storage_format_cli_alias_matches_mystudy():
    assert normalize_cli_aliases(["--rlds-storage-format=hybrid"]) == [
        "--dataset.rlds_storage_format=hybrid"
    ]


def test_target_control_hz_cli_alias_matches_mystudy():
    assert normalize_cli_aliases(["--target-control-hz=10"]) == [
        "--dataset.rlds_target_control_hz=10"
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rlds_shuffle_buffer_size": 0}, "shuffle_buffer_size"),
        ({"rlds_resize_size": (256, 0)}, "resize_size"),
        ({"rlds_action_transform": "unknown"}, "action_transform"),
        ({"rlds_storage_format": "pickle"}, "storage_format"),
        ({"rlds_q0_device": "mps"}, "q0_device"),
        ({"rlds_target_control_hz": -1}, "target_control_hz"),
        ({"rlds_camera_views": ("primary", "overhead")}, "camera_views"),
    ],
)
def test_dataset_config_rejects_invalid_rlds_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DatasetConfig(repo_id="actionmem_mix", **kwargs)
