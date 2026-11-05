"""Small local datasets exercise the actual reader without loading model weights."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import draccus
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import default_collate

from lerobot.common.action_chunk_eval import full_horizon_indices
from lerobot.configs.dataset_mixture import DatasetSourceConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.configured_dataset import _gripper_indices, _map_discrete_gripper
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import WeightedDatasetSampler, compute_sampler_state
from lerobot.policies.common.masked_loss import reduce_action_loss
from lerobot.policies.pi05_lap.configuration_pi05_lap import LAM_IMAGE_PAIR, PI05LAPConfig
from lerobot.processor import DataProcessorPipeline, NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.converters import batch_to_transition, identity_transition, transition_to_batch
from lerobot.types import TransitionKey
from lerobot.utils.feature_utils import dataset_to_policy_features


@pytest.fixture(scope="module")
def source_payload(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixture")
    sources = {}
    for i, (action_dim, state_dim, fps) in enumerate(((3, 2, 10), (5, 4, 20))):
        # Different names, vector dimensions, image sizes, frame rates and camera counts.
        ak, sk, ck = f"control{i}", f"state{i}", f"view{i}"
        features = {
            ak: {"dtype": "float32", "shape": (action_dim,), "names": None},
            sk: {"dtype": "float32", "shape": (state_dim,), "names": None},
            "gripper": {"dtype": "float32", "shape": (1,), "names": None},
            ck: {"dtype": "image", "shape": (12 + i * 4, 16, 3), "names": ["height", "width", "channels"]},
        }
        cameras = {"observation.images.main": ck}
        if i == 0:
            features["wrist"] = deepcopy(features[ck])
            cameras["observation.images.wrist"] = "wrist"
        dataset = LeRobotDataset.create(
            f"test/source{i}",
            fps=fps,
            root=root / str(i),
            features=features,
            use_videos=False,
        )
        for ep in range(2):
            for frame in range(4):
                value = ep * 40 + frame * 10 + i
                row = {
                    ak: np.arange(action_dim, dtype=np.float32) + value,
                    sk: np.arange(state_dim, dtype=np.float32) + value,
                    "gripper": np.array([frame % 2], dtype=np.float32),
                    ck: np.full(features[ck]["shape"], value, dtype=np.uint8),
                    "task": f"task{i}",
                }
                if i == 0:
                    row["wrist"] = row[ck].copy()
                dataset.add_frame(row)
            dataset.save_episode()
        dataset.finalize()
        sources[f"source{i}"] = {
            "repo_id": f"test/source{i}",
            "root": str(root / str(i)),
            "weight": 1 + 2 * i,
            "action_fields": [{"key": ak, "indices": list(reversed(range(action_dim)))}, {"key": "gripper"}],
            "state_fields": [{"key": sk}],
            "camera_fields": cameras,
            "lam_camera_key": ck,
        }
    return {"return_uint8": True, "video_backend": "pyav", "datasets": sources}


def config(payload, stage="pre-training"):
    return TrainPipelineConfig(
        dataset=draccus.decode(DatasetConfig, deepcopy(payload)),
        policy=PI05LAPConfig(
            device="cpu",
            training_stage=stage,
            chunk_size=2,
            n_action_steps=2,
            image_resolution=(16, 16),
            push_to_hub=False,
        ),
    )


@pytest.fixture(scope="module")
def missing_episode_tasks_payload(tmp_path_factory):
    """A v3 dataset with per-frame task labels but no episode-level tasks column."""
    root = tmp_path_factory.mktemp("missing_episode_tasks") / "libero"
    camera = "observation.images.image"
    dataset = LeRobotDataset.create(
        "test/no_episode_tasks",
        fps=10,
        root=root,
        use_videos=False,
        features={
            "action": {"dtype": "float32", "shape": (7,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            camera: {"dtype": "image", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
        },
    )
    for episode, task in enumerate(("task A", "task A", "task B", "task B")):
        for frame in range(3):
            dataset.add_frame(
                {
                    "action": np.full(7, episode + frame, dtype=np.float32),
                    "observation.state": np.full(7, episode + frame, dtype=np.float32),
                    camera: np.full((16, 16, 3), episode + frame, dtype=np.uint8),
                    "task": task,
                }
            )
        dataset.save_episode()
    dataset.finalize()
    for path in (root / "meta" / "episodes").glob("*/*.parquet"):
        table = pq.read_table(path)
        pq.write_table(table.drop(["tasks"]), path)
    return {
        "eval_split": 0.5,
        "video_backend": "pyav",
        "datasets": {
            "libero": {
                "repo_id": "test/no_episode_tasks",
                "root": str(root),
                "camera_fields": {camera: camera},
                "lam_camera_key": camera,
            }
        },
    }


@pytest.mark.parametrize("stage", ["pre-training", "post-training"])
@pytest.mark.parametrize("configured", [True, False])
def test_split_without_episode_tasks(missing_episode_tasks_payload, stage, configured, tmp_path):
    payload = deepcopy(missing_episode_tasks_payload)
    if configured:
        # Exercise the training-script JSON entry point, even for a single source.
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(payload))
        cfg = config({"config_path": str(path)}, stage)
    else:
        source = payload["datasets"]["libero"]
        cfg = config({"repo_id": source["repo_id"], "root": source["root"], "eval_split": 0.5}, stage)
    train, evaluation = make_train_eval_datasets(cfg)
    train_child = train.children[0][1] if configured else train
    eval_child = evaluation.children[0][1] if configured else evaluation
    assert "tasks" not in train_child.meta.episodes.column_names
    # Splitting all episodes as one group would incorrectly select [0, 1] for training.
    assert train_child.episodes == [0, 2]
    assert eval_child.episodes == [1, 3]
    assert len(train) == len(evaluation) == 6
    assert [evaluation[i]["task"] for i in (0, 3)] == ["task A", "task B"]
    assert (LAM_IMAGE_PAIR in train[0]) == (stage == "pre-training")
    assert full_horizon_indices(evaluation, 2) == [0, 1, 3, 4]
    if configured:
        assert evaluation.eval_indices(4) == [0, 1, 3, 4]


def test_episode_task_fallback_reads_only_scalar_columns(missing_episode_tasks_payload, monkeypatch):
    source = missing_episode_tasks_payload["datasets"]["libero"]
    meta = LeRobotDatasetMetadata(source["repo_id"], root=source["root"])
    projections = []
    original = pq.ParquetFile.iter_batches

    def read_batches(self, *args, **kwargs):
        projections.append(kwargs["columns"])
        # Small batches exercise episodes that span multiple batches in one shared data file.
        kwargs["batch_size"] = 2
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", read_batches)
    monkeypatch.setattr(DatasetReader, "get_item", lambda *args: pytest.fail("must not decode frames"))
    assert meta.get_episode_task_keys([0, 3]) == {0: "task A", 3: "task B"}
    assert projections == [["episode_index", "task_index"]]


def test_episode_task_summaries_do_not_read_frames(source_payload, monkeypatch):
    source = source_payload["datasets"]["source0"]
    meta = LeRobotDatasetMetadata(source["repo_id"], root=source["root"])
    monkeypatch.setattr(pq, "ParquetFile", lambda *args: pytest.fail("must not scan frame parquet"))
    assert meta.get_episode_task_keys([0, 1]) == {0: "task0", 1: "task0"}
    meta.episodes = meta.episodes.remove_columns("tasks")
    # A single task in the task table also needs no frame scan.
    assert meta.get_episode_task_keys([0, 1]) == {0: "task0", 1: "task0"}


@pytest.fixture(scope="module")
def multicamera_payload(source_payload, tmp_path_factory):
    payload = deepcopy(source_payload)
    root = tmp_path_factory.mktemp("cameras") / "three_views"
    camera_values = {"front": 40, "wrist": 80, "side": 120, "teacher": 160, "unused": 200}
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        **{
            key: {"dtype": "image", "shape": (16, 16, 3), "names": ["height", "width", "channels"]}
            for key in camera_values
        },
    }
    dataset = LeRobotDataset.create(
        "test/three_views", fps=30, features=features, root=root, use_videos=False
    )
    for ep in range(2):
        for frame in range(4):
            dataset.add_frame(
                {
                    "action": np.full(6, frame + ep, dtype=np.float32),
                    "observation.state": np.full(4, frame + ep, dtype=np.float32),
                    "task": "third task",
                    **{
                        key: np.full((16, 16, 3), value + frame * 5 + ep, dtype=np.uint8)
                        for key, value in camera_values.items()
                    },
                }
            )
        dataset.save_episode()
    dataset.finalize()
    payload["datasets"]["source2"] = {
        "repo_id": "test/three_views",
        "root": str(root),
        "weight": 1,
        "camera_fields": {
            "observation.images.main": "front",
            "observation.images.wrist": "wrist",
            "observation.images.side": "side",
        },
        "lam_camera_key": "front",
    }
    return payload


def test_mixed_batch_and_episode_clamping(source_payload):
    cfg = config(source_payload)
    ds, _ = make_train_eval_datasets(cfg)
    assert len(ds) == 16
    assert all(child.absolute_to_relative_idx is None for _, child, _ in ds.children)
    assert ds.schema.action_dim == 6
    assert ds.schema.state_dim == 4
    a, b = ds[3], ds[8]
    assert a["action"].shape == b["action"].shape == (2, 6)
    assert a["action"][0].tolist() == [32, 31, 30, 1, 0, 0]
    assert a["action"][1].tolist() == a["action"][0].tolist()
    assert a["action_is_pad"].tolist() == [False, True]
    assert a["action_dim_is_pad"].tolist() == [False] * 4 + [True] * 2
    assert a["observation.state"].tolist() == [30, 31, 0, 0]
    assert b["observation.images.wrist_is_pad"].item()
    assert a[LAM_IMAGE_PAIR][0].mean().item() == pytest.approx(30 / 255)
    assert a[LAM_IMAGE_PAIR][1].mean().item() == pytest.approx(30 / 255)
    assert b[LAM_IMAGE_PAIR][1].mean().item() == pytest.approx(21 / 255)
    assert ds[1][LAM_IMAGE_PAIR][1].mean().item() == pytest.approx(30 / 255)
    batch = default_collate([a, b])
    assert batch["observation.images.main"].dtype == torch.uint8
    assert batch["observation.images.main"].shape == (2, 3, 16, 16)
    assert batch["dataset_index"].tolist() == [0, 1]
    assert a["episode_index"] != b["episode_index"]
    assert transition_to_batch(batch_to_transition(batch))["action_dim_is_pad"].equal(
        batch["action_dim_is_pad"]
    )
    assert transition_to_batch(batch_to_transition(batch))["dataset_index"].tolist() == [0, 1]


def test_configured_float_image_uses_official_truncation(source_payload, monkeypatch):
    cfg = config(source_payload, stage="post-training")
    ds, _ = make_train_eval_datasets(cfg)
    original = LeRobotDataset.__getitem__

    def decoded_float(self, index):
        row = original(self, index)
        for key in self.meta.camera_keys:
            row[key] = torch.full_like(row[key], 127.75 / 255, dtype=torch.float32)
        return row

    monkeypatch.setattr(LeRobotDataset, "__getitem__", decoded_float)
    item = ds[0]
    assert item["observation.images.main"].max() == 127
    assert item["observation.images.main"].dtype == torch.uint8


def test_per_source_normalization_and_checkpoint(source_payload, tmp_path):
    cfg = config(source_payload)
    ds, _ = make_train_eval_datasets(cfg)
    batch = default_collate([ds[0], ds[8]])
    normalizer = NormalizerProcessorStep(
        features=dataset_to_policy_features(ds.meta.features),
        norm_map=cfg.policy.normalization_mapping,
        stats=ds.meta.stats,
    )
    transition = batch_to_transition(batch)
    normalized = normalizer(transition)
    # Each source's first sample is near its own lower quantile despite different values/dimensions.
    result = transition_to_batch(normalized)
    torch.testing.assert_close(result["action"][0, :, 0], result["action"][1, :, 0])
    assert torch.all(result["action"][..., -1][0] == 0)
    assert torch.all(result["observation.state"][0, 2:] == 0)
    unnormalizer = UnnormalizerProcessorStep(**normalizer.get_config(), stats=ds.meta.stats)
    recovered = transition_to_batch(unnormalizer(normalized))
    torch.testing.assert_close(recovered["action"], batch["action"])
    torch.testing.assert_close(recovered["observation.state"], batch["observation.state"])
    pipeline = DataProcessorPipeline(
        [normalizer],
        name="test",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    pipeline.save_pretrained(tmp_path)
    restored = DataProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="test.json",
        to_transition=identity_transition,
        to_output=identity_transition,
    )
    torch.testing.assert_close(transition_to_batch(restored(transition))["action"], result["action"])
    without_id = batch_to_transition({k: v for k, v in batch.items() if k != "dataset_index"})
    with pytest.raises(ValueError, match="dataset_index"):
        restored(without_id)
    deployed = NormalizerProcessorStep(**normalizer.get_config(), stats=ds.meta.stats, dataset_index=1)
    deployed(without_id)


@pytest.mark.parametrize("mode", [NormalizationMode.QUANTILES, NormalizationMode.MEAN_STD])
def test_discrete_gripper_sources_share_raw_and_normalized_commands(source_payload, mode):
    payload = deepcopy(source_payload)
    payload["datasets"]["source0"]["gripper_representation"] = {"type": "discrete", "open": "1", "close": "0"}
    payload["datasets"]["source1"]["gripper_representation"] = {"type": "discrete", "open": 0, "close": 1}
    cfg = config(payload, "post-training")
    cfg.policy.normalization_mapping[FeatureType.ACTION.value] = mode
    ds, _ = make_train_eval_datasets(cfg)
    # Both examples are open at t and closed at t+1, despite opposite raw encodings.
    batch = default_collate([ds[1], ds[8]])
    expected = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(batch["action"][0, :, 3], expected)
    torch.testing.assert_close(batch["action"][1, :, 5], expected)
    assert batch["observation.state"][0, :2].tolist() == [10, 11]
    assert batch["action"][0, 0, :3].tolist() == [12, 11, 10]
    assert batch["action"][0, :, 4:].count_nonzero() == 0
    # Source metadata statistics stay untouched; only the mapped schema uses canonical stats.
    assert ds.children[0][1].meta.stats["gripper"]["mean"][0] == 0.5
    normalizer = NormalizerProcessorStep(
        features=dataset_to_policy_features(ds.meta.features),
        norm_map=cfg.policy.normalization_mapping,
        stats=ds.meta.stats,
    )
    normalized = normalizer(batch_to_transition(batch))
    action = transition_to_batch(normalized)["action"]
    torch.testing.assert_close(action[0, :, 3], expected)
    torch.testing.assert_close(action[1, :, 5], expected)
    inverse = UnnormalizerProcessorStep(**normalizer.get_config(), stats=ds.meta.stats)
    torch.testing.assert_close(transition_to_batch(inverse(normalized))["action"], batch["action"])


@pytest.mark.parametrize("dimension", [7, 14])
def test_arm_gripper_mapping_rejects_unknown_commands_and_leaves_continuous_values(dimension):
    source = draccus.decode(
        DatasetSourceConfig,
        {"repo_id": "robot", "gripper_representation": {"type": "discrete", "open": "1", "close": "-1"}},
    )
    meta = SimpleNamespace(features={"action": {"shape": (dimension,)}})
    indices = _gripper_indices(source, meta, [("action", list(range(dimension)))])
    action = torch.full((2, dimension), 0.25)
    action[0, indices], action[1, indices] = 1, -1
    mapped = _map_discrete_gripper(action.clone(), source, indices)
    assert (mapped[0, indices] == -1).all() and (mapped[1, indices] == 1).all()
    assert (mapped[:, [i for i in range(dimension) if i not in indices]] == 0.25).all()
    action[0, indices] = 0.5
    with pytest.raises(ValueError, match="discrete gripper labels"):
        _map_discrete_gripper(action.clone(), source, indices)
    continuous = draccus.decode(
        DatasetSourceConfig, {"repo_id": "robot", "gripper_representation": {"type": "continuous"}}
    )
    torch.testing.assert_close(_map_discrete_gripper(action.clone(), continuous, []), action)


@pytest.mark.parametrize("policy_type", ["pi05", "pi05_lap"])
def test_source_action_settings_survive_mixed_batches_and_reload(source_payload, tmp_path, policy_type):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.processor import RelativeActionsProcessorStep

    payload = deepcopy(source_payload)
    payload["datasets"]["source0"].update(use_relative_actions=False, image_augmentation=False)
    payload["datasets"]["source1"].update(use_relative_actions=True, image_augmentation=True)
    payload["datasets"]["source1"]["action_fields"][0]["indices"] = [3, 2, 1, 0]
    cfg = config(payload, "post-training")
    if policy_type == "pi05":
        cfg.policy = PI05Config(device="cpu", chunk_size=2, n_action_steps=2)
    # JSON source flags are authoritative even if the global defaults differ.
    cfg.policy.use_relative_actions = True
    cfg.policy.image_augmentation = False
    ds, _ = make_train_eval_datasets(cfg)
    features = dataset_to_policy_features(ds.meta.features)
    cfg.policy.input_features = {key: feature for key, feature in features.items() if key != "action"}
    cfg.policy.output_features = {"action": features["action"]}
    cfg.policy._runtime_dataset_meta = ds.meta
    pre, post = make_pre_post_processors(cfg.policy, dataset_stats=ds.meta.stats)
    post = replace(post, to_transition=batch_to_transition)
    batch = default_collate([ds[1], ds[9]])
    result = pre(batch)
    assert result["image_augmentation"].tolist() == [False, True]
    relative = next(step for step in pre.steps if isinstance(step, RelativeActionsProcessorStep))
    expected = batch["action"].clone()
    expected[1, :, :4] -= batch["observation.state"][1, None, :4]
    converted = relative(batch_to_transition(batch))[TransitionKey.ACTION]
    torch.testing.assert_close(converted, expected)
    recovered = post({"action": result["action"], "dataset_index": batch["dataset_index"]})
    for row, dims in enumerate((4, 5)):
        torch.testing.assert_close(recovered[row, :, :dims], batch["action"][row, :, :dims])
    assert torch.count_nonzero(result["action"][0, :, 4:]) == 0
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    del cfg.policy._runtime_dataset_meta
    cfg.policy.use_relative_actions = False
    loaded_pre, loaded_post = make_pre_post_processors(cfg.policy, pretrained_path=str(tmp_path))
    loaded_post = replace(loaded_post, to_transition=batch_to_transition)
    loaded = loaded_pre(batch)
    torch.testing.assert_close(loaded["action"], result["action"])
    restored = loaded_post({"action": loaded["action"], "dataset_index": batch["dataset_index"]})
    torch.testing.assert_close(restored, recovered)
    # Inference has no action labels; preprocessing must still cache the correct source and state.
    inference = {key: value for key, value in batch.items() if key != "action"}
    loaded_pre(inference)
    torch.testing.assert_close(
        loaded_post({"action": loaded["action"], "dataset_index": batch["dataset_index"]}), recovered
    )
    relative = next(step for step in loaded_pre.steps if isinstance(step, RelativeActionsProcessorStep))
    relative.dataset_index = 1
    source1 = {key: value[1:] for key, value in inference.items()}
    source1.pop("dataset_index")
    relative(batch_to_transition(source1))
    assert relative.get_action_mask(5).tolist() == [[True, True, True, True, False]]


def test_relative_source_requires_matching_state(source_payload):
    payload = deepcopy(source_payload)
    payload["datasets"]["source1"]["use_relative_actions"] = True
    with pytest.raises(ValueError, match="matching state components"):
        make_train_eval_datasets(config(payload, "post-training"))
    source = payload["datasets"]["source1"]
    source["action_fields"][0]["indices"] = [0, 1, 2, 3]
    source["relative_exclude_joints"] = []
    source["gripper_representation"] = {"type": "discrete", "open": 0, "close": 1}
    ds, _ = make_train_eval_datasets(config(payload, "post-training"))
    # Canonical grippers are excluded even when the user converts every other action to relative.
    assert ds.meta.relative_action_masks["1"] == [True, True, True, True, False]


@pytest.mark.parametrize("training", [False, True])
def test_source_image_augmentation_is_per_sample_and_train_only(training):
    from lerobot.policies.common.openpi_io import prepare_openpi_images

    key = "observation.images.base_0_rgb"
    policy = SimpleNamespace(
        training=training,
        parameters=lambda: iter([torch.zeros(1)]),
        config=SimpleNamespace(image_features=[key], image_resolution=(224, 224), image_augmentation=False),
    )
    batch = {
        key: torch.full((2, 3, 224, 224), 128, dtype=torch.uint8),
        "observation.state": torch.zeros(2, 32),
        "image_augmentation": torch.tensor([False, True]),
    }
    torch.manual_seed(47)
    before = torch.random.get_rng_state()
    images, masks = prepare_openpi_images(policy, batch)
    expected = torch.full_like(images[0][0], 128).div(255).mul(2).sub(1)
    assert torch.equal(images[0][0], expected)
    assert masks[0].tolist() == [True, True]
    if training:
        assert not torch.equal(images[0][1], expected)
    else:
        assert torch.equal(images[0][1], expected)
        assert torch.equal(torch.random.get_rng_state(), before)


def test_post_training_has_no_teacher_and_split_is_episode_local(source_payload):
    payload = deepcopy(source_payload)
    payload["eval_split"] = 0.5
    cfg = config(payload, "post-training")
    train, evaluation = make_train_eval_datasets(cfg)
    assert len(train) == len(evaluation) == 8
    assert LAM_IMAGE_PAIR not in train[0]
    assert train[3]["action"][0, 0] == 32
    assert evaluation[0]["action"][0, 0] == 42
    assert len(evaluation.eval_indices(3)) == 3
    assert evaluation[0]["episode_index"] != train[0]["episode_index"]


def test_full_horizon_evaluation_filters_each_held_out_source_without_decoding(
    multicamera_payload, monkeypatch
):
    payload = deepcopy(multicamera_payload)
    payload["eval_split"] = 0.5
    _, evaluation = make_train_eval_datasets(config(payload, "post-training"))
    # Two episodes per source: the second episode is held out, with 4 frames.
    assert len(evaluation) == 12
    indices = full_horizon_indices(evaluation, 2)
    assert indices == [0, 1, 2, 4, 5, 6, 8, 9, 10]
    assert all(not evaluation[i]["action_is_pad"].any() for i in indices)
    for _, child, _ in evaluation.children:
        assert full_horizon_indices(child, 2) == [0, 1, 2]
    monkeypatch.setattr(
        type(evaluation), "__getitem__", lambda *args: pytest.fail("selection must not decode frames")
    )
    assert full_horizon_indices(evaluation, 4) == [0, 4, 8]
    assert full_horizon_indices(evaluation, 5) == []


def test_eval_interval_validates_effective_source_splits(source_payload, monkeypatch):
    payload = deepcopy(source_payload)
    payload["eval_split"] = 0.5
    for source in payload["datasets"].values():
        source["eval_split"] = 0.0
    cfg = config(payload, "post-training")
    cfg.eval_steps = 50
    monkeypatch.setattr(TrainPipelineConfig, "_resolve_pretrained_from_cli", lambda self: None)
    monkeypatch.setattr("lerobot.configs.train.parser.parse_arg", lambda key: None)
    with pytest.raises(ValueError, match="eval_split"):
        cfg.validate()
    cfg.dataset.datasets["source1"].eval_split = None
    cfg.validate()
    cfg.eval_steps = -1
    with pytest.raises(ValueError, match="nonnegative"):
        cfg.validate()


def test_manifest_snapshot_and_reload(source_payload, tmp_path):
    source_payload = deepcopy(source_payload)
    source_payload["datasets"]["source0"]["gripper_representation"] = {
        "type": "discrete",
        "open": "1",
        "close": "0",
    }
    source_payload["datasets"]["source1"]["gripper_representation"] = "continuous"
    path = tmp_path / "train.json"
    path.write_text(json.dumps(source_payload))
    loaded = DatasetConfig(config_path=str(path)).resolve_file()
    assert loaded.datasets["source1"].weight == 3
    assert loaded.datasets["source0"].gripper_type == "discrete"
    saved = draccus.decode(DatasetConfig, draccus.encode(loaded))
    assert saved.datasets["source1"].gripper_representation == "continuous"
    assert saved.datasets["source0"].gripper_mapping.open == 1
    assert saved.datasets["source0"].gripper_mapping.close == 0
    payload = deepcopy(source_payload)
    payload["datasets"]["source1"]["weight"] = 9
    path.write_text(json.dumps(payload))
    assert saved.resolve_file().datasets["source1"].weight == 3
    assert saved.resolve_file(reload=True).datasets["source1"].weight == 9
    path.unlink()
    assert saved.resolve_file().datasets["source1"].weight == 3


def test_weights_and_resume_independent_of_dataset_size():
    sampler = WeightedDatasetSampler([30000, 10000], [1, 3], seed=24)
    order = list(sampler)
    assert sum(i >= 30000 for i in order) / len(order) == pytest.approx(0.75, abs=0.01)
    second_rank = WeightedDatasetSampler([30000, 10000], [1, 3], seed=24)
    assert list(second_rank) == order
    resumed = WeightedDatasetSampler([30000, 10000], [1, 3], seed=24)
    resumed.load_state_dict(compute_sampler_state(step=103, num_frames=40000, batch_size=8, num_processes=2))
    assert list(resumed) == order[103 * 16 :]
    epoch1 = list(sampler)
    resumed.load_state_dict({"epoch": 1, "start_index": 4097})
    assert list(resumed) == epoch1[4097:]


def test_accelerate_sharding_and_resume_at_nonzero_epoch():
    from accelerate.data_loader import prepare_data_loader

    # Eleven frames round up to twelve draws, so two ranks each receive three full batches.
    for rank in range(2):
        sampler = WeightedDatasetSampler([7, 4], [1, 3], seed=5, batch_multiple=4)
        assert len(sampler) == 12
        loader = prepare_data_loader(
            torch.utils.data.DataLoader(list(range(11)), batch_size=2, sampler=sampler),
            num_processes=2,
            process_index=rank,
        )
        loader.set_epoch(3)
        full = list(loader)
        resumed_sampler = WeightedDatasetSampler([7, 4], [1, 3], seed=5, batch_multiple=4)
        state = compute_sampler_state(step=10, num_frames=12, batch_size=2, num_processes=2)
        assert state == {"epoch": 3, "start_index": 4}
        resumed_sampler.load_state_dict(state)
        resumed = prepare_data_loader(
            torch.utils.data.DataLoader(list(range(11)), batch_size=2, sampler=resumed_sampler),
            num_processes=2,
            process_index=rank,
        )
        resumed.set_epoch(state["epoch"])
        suffix = list(resumed)
        assert len(suffix) == len(full) - 1
        for actual, expected in zip(suffix, full[1:], strict=True):
            assert actual.equal(expected)
        # Both loaders must advance to the same next epoch too.
        for actual, expected in zip(list(resumed), list(loader), strict=True):
            assert actual.equal(expected)


def test_padded_action_dimensions_have_zero_gradient():
    losses = torch.tensor([[[2.0, 2.0, 100.0]], [[4.0, 4.0, 4.0]]], requires_grad=True)
    loss, _ = reduce_action_loss(losses, torch.tensor([[False, False, True], [False, False, False]]), "mean")
    assert loss.item() == 3
    loss.backward()
    assert losses.grad[0, 0, 2] == 0


@pytest.mark.parametrize("policy_type", ["pi05", "pi05_lap"])
def test_policy_masks_absent_views(multicamera_payload, policy_type):
    if policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Policy
    else:
        from lerobot.policies.pi05_lap.modeling_pi05_lap import PI05LAPPolicy as Policy

    cfg = config(multicamera_payload)
    ds, _ = make_train_eval_datasets(cfg)
    batch = default_collate([ds[0], ds[8], ds[16]])
    for key in ds.meta.camera_keys:
        batch[key] = batch[key].float() / 255
    dummy = SimpleNamespace(
        parameters=lambda: iter([torch.zeros(1)]),
        training=False,
        config=SimpleNamespace(
            image_features=ds.meta.camera_keys, image_resolution=(16, 16), image_augmentation=True
        ),
    )
    images, masks = Policy._preprocess_images(dummy, batch)
    assert len(images) == 3
    assert masks[0].tolist() == [True, True, True]
    assert masks[1].tolist() == [True, False, True]
    assert masks[2].tolist() == [False, False, True]


def test_lam_camera_is_selected_student_view(multicamera_payload):
    ds, _ = make_train_eval_datasets(config(multicamera_payload))
    _, child, _ = ds.children[2]
    assert set(child.reader.camera_keys) == {"front", "wrist", "side"}
    assert "unused" not in child.hf_dataset.column_names
    assert "teacher" not in child.hf_dataset.column_names
    assert len(ds.meta.camera_keys) == 3
    item = ds[16]
    assert "teacher" not in item
    assert item[LAM_IMAGE_PAIR][0].mean().item() == pytest.approx(40 / 255)
    assert item[LAM_IMAGE_PAIR][1].mean().item() == pytest.approx(50 / 255)
    assert item["observation.images.main"].float().mean().item() == 40
    assert item["observation.images.side"].float().mean().item() == 120
    tail = ds[19]
    torch.testing.assert_close(tail[LAM_IMAGE_PAIR][0], tail[LAM_IMAGE_PAIR][1])

    # Teacher labels must be captured before random student image augmentation.
    ds.image_transforms = lambda image: torch.zeros_like(image)
    augmented = ds[16]
    torch.testing.assert_close(augmented[LAM_IMAGE_PAIR], item[LAM_IMAGE_PAIR])
    assert not augmented["observation.images.main"].any()


@pytest.mark.parametrize("camera_count", [1, 2, 3])
def test_loads_only_configured_cameras_in_post_training(multicamera_payload, camera_count):
    payload = deepcopy(multicamera_payload)
    source = payload["datasets"]["source2"]
    source["camera_fields"] = dict(list(source["camera_fields"].items())[:camera_count])
    payload["datasets"] = {"source2": source}
    ds, _ = make_train_eval_datasets(config(payload, "post-training"))
    _, child, _ = ds.children[0]
    assert len(ds.meta.camera_keys) == camera_count
    assert set(child.reader.camera_keys) == set(source["camera_fields"].values())
    assert set(child.meta.camera_keys).intersection(child.hf_dataset.column_names) == set(
        child.reader.camera_keys
    )
    assert "teacher" not in child.hf_dataset.column_names
    assert "unused" not in child.hf_dataset.column_names
    assert LAM_IMAGE_PAIR not in ds[0]


def test_lam_camera_defaults_aliases_and_validation(multicamera_payload):
    payload = deepcopy(multicamera_payload)
    payload["datasets"] = {"source2": payload["datasets"]["source2"]}
    source = payload["datasets"]["source2"]
    for choice in (None, "observation.images.main", "front"):
        source["lam_camera_key"] = choice
        ds, _ = make_train_eval_datasets(config(payload))
        assert ds[0][LAM_IMAGE_PAIR][0].mean().item() == pytest.approx(40 / 255)
        assert set(ds.children[0][1].reader.camera_keys) == {"front", "wrist", "side"}
    for choice in ("teacher", "missing_camera", "action"):
        source["lam_camera_key"] = choice
        with pytest.raises(draccus.utils.ParsingError) as error:
            config(payload)
        assert "nonempty VLM view" in str(error.value.__cause__)
    for choice in ("", "front,side", "[front,side]"):
        source["lam_camera_key"] = choice
        with pytest.raises(draccus.utils.ParsingError):
            config(payload)


def test_empty_slots_and_default_lam_camera(multicamera_payload):
    payload = deepcopy(multicamera_payload)
    source = payload["datasets"]["source2"]
    source["camera_fields"] = {
        "observation.images.main": None,
        "observation.images.wrist": "wrist",
        "observation.images.side": "",
    }
    source["lam_camera_key"] = None
    cfg = config(payload)
    ds, _ = make_train_eval_datasets(cfg)
    assert ds.children[2][1].reader.camera_keys == ["wrist"]
    item = ds[16]
    assert item["observation.images.main_is_pad"].item()
    assert item["observation.images.side_is_pad"].item()
    assert not item["observation.images.wrist_is_pad"].item()
    assert not item["observation.images.main"].any()
    assert item[LAM_IMAGE_PAIR][0].mean().item() == pytest.approx(80 / 255)
    assert item[LAM_IMAGE_PAIR][1].mean().item() == pytest.approx(90 / 255)
    default_collate([ds[0], ds[8], item])
    saved = draccus.decode(DatasetConfig, draccus.encode(cfg.dataset))
    assert saved.datasets["source2"].camera_fields["observation.images.side"] is None
    for choice in ("front", "observation.images.main"):
        source["lam_camera_key"] = choice
        with pytest.raises(draccus.utils.ParsingError) as error:
            config(payload)
        assert "nonempty VLM view" in str(error.value.__cause__)


@pytest.mark.parametrize("policy_type", ["pi05", "pi05_lap"])
@pytest.mark.parametrize("camera_fields", [{}, {"observation.images.main": None}])
def test_no_cameras_in_post_training(multicamera_payload, policy_type, camera_fields):
    if policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Policy
    else:
        from lerobot.policies.pi05_lap.modeling_pi05_lap import PI05LAPPolicy as Policy

    payload = deepcopy(multicamera_payload)
    source = payload["datasets"]["source2"]
    source["camera_fields"] = camera_fields
    source["lam_camera_key"] = None
    payload["datasets"] = {"source2": source}
    ds, _ = make_train_eval_datasets(config(payload, "post-training"))
    child = ds.children[0][1]
    assert child.reader.camera_keys == []
    assert not set(child.meta.camera_keys).intersection(child.hf_dataset.column_names)
    batch = default_collate([ds[0], ds[1]])
    dummy = SimpleNamespace(
        parameters=lambda: iter([torch.zeros(1)]),
        training=False,
        config=SimpleNamespace(
            image_features=ds.meta.camera_keys, image_resolution=(16, 16), image_augmentation=True
        ),
    )
    images, masks = Policy._preprocess_images(dummy, batch)
    assert len(images) == 3
    assert all(not mask.any() for mask in masks)
    assert LAM_IMAGE_PAIR not in batch
    with pytest.raises(ValueError, match="nonempty camera_fields"):
        make_train_eval_datasets(config(payload))


def test_video_queries_and_cache_checks_ignore_unselected_cameras(multicamera_payload, tmp_path, monkeypatch):
    # Reuse real scalar parquet data but supply video metadata and a decoder spy.
    # Only the two selected video files exist; excluded views must not trigger downloads.
    source = multicamera_payload["datasets"]["source2"]
    original_meta = LeRobotDatasetMetadata(source["repo_id"], root=source["root"])
    features = deepcopy(original_meta.features)
    video_keys = list(original_meta.camera_keys)
    for key in video_keys:
        features[key]["dtype"] = "video"
    episodes = [dict(row) for row in original_meta.episodes]
    for row in episodes:
        for key in video_keys:
            row[f"videos/{key}/from_timestamp"] = 0.0

    def video_path(ep, key):
        return tmp_path / f"{key}-{ep}.mp4"

    for ep in range(2):
        for key in ("front", "teacher"):
            video_path(ep, key).touch()
    meta = SimpleNamespace(
        features=features,
        camera_keys=video_keys,
        video_keys=video_keys,
        image_keys=[],
        depth_keys=[],
        fps=30,
        episodes=episodes,
        total_frames=8,
        total_episodes=2,
        tasks=original_meta.tasks,
        get_video_file_path=video_path,
        get_data_file_path=original_meta.get_data_file_path,
    )
    reader = DatasetReader(
        meta,
        original_meta.root,
        episodes=None,
        tolerance_s=1e-4,
        video_backend="pyav",
        delta_timestamps={"teacher": [0.0, 2 / 30]},
        image_transforms=None,
        camera_keys=["front", "teacher"],
    )
    assert reader.try_load()
    calls = []

    def decode(path, timestamps, *args, **kwargs):
        calls.append((path.name, timestamps))
        return torch.zeros(len(timestamps), 3, 16, 16)

    monkeypatch.setattr("lerobot.datasets.dataset_reader.decode_video_frames", decode)
    item = reader.get_item(3)
    assert set(video_keys).intersection(item) == {"front", "teacher"}
    calls = dict(calls)
    assert set(calls) == {"front-0.mp4", "teacher-0.mp4"}
    assert calls["front-0.mp4"] == pytest.approx([3 / 30])
    assert calls["teacher-0.mp4"] == pytest.approx([3 / 30, 3 / 30])
    files = reader.get_episodes_file_paths()
    assert not any("unused" in path or "side" in path for path in files)
    # A missing selected camera still fails the cache check.
    video_path(0, "teacher").unlink()
    assert not reader._check_cached_episodes_sufficient()


def test_json_cli_and_resume_snapshot(source_payload, tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(source_payload))
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            f"--dataset.config_path={path}",
            "--policy.type=pi05_lap",
            "--policy.device=cpu",
            "--policy.push_to_hub=false",
            "--env_eval_freq=0",
        ],
    )
    cfg.validate()
    assert len(cfg.dataset.datasets) == 2
    cfg.save_pretrained(tmp_path / "checkpoint")
    cfg.resume = True
    # Avoid resolving an actual model checkpoint; only check data configuration behavior.
    monkeypatch.setattr(TrainPipelineConfig, "_resolve_pretrained_from_cli", lambda self: None)
    monkeypatch.setattr("lerobot.configs.train.parser.parse_arg", lambda key: None)
    cfg.validate()
    assert not cfg._reset_data_order
    changed = deepcopy(source_payload)
    changed["datasets"] = dict(reversed(list(changed["datasets"].items())))
    path.write_text(json.dumps(changed))
    monkeypatch.setattr("lerobot.configs.train.parser.parse_arg", lambda key: str(path))
    cfg.validate()
    assert cfg._reset_data_order
    assert list(cfg.dataset.datasets) == ["source1", "source0"]


def test_invalid_source_and_missing_field_errors(source_payload):
    payload = deepcopy(source_payload)
    payload["datasets"]["source0"]["weight"] = -1
    with pytest.raises(draccus.utils.ParsingError):
        config(payload)
    for representation in (
        {"type": "discrete", "open": -1},
        {"type": "discrete", "open": 1, "close": 1},
        {"type": "discrete", "open": "nan", "close": 1},
    ):
        payload = deepcopy(source_payload)
        payload["datasets"]["source0"]["gripper_representation"] = representation
        with pytest.raises(draccus.utils.ParsingError):
            config(payload)
    payload = deepcopy(source_payload)
    payload["datasets"]["source0"]["gripper_representation"] = "binary"
    with pytest.raises(draccus.utils.ParsingError):
        config(payload)
    cfg = config(source_payload)
    cfg.dataset.datasets["source0"].action_fields[0].key = "does_not_exist"
    with pytest.raises(ValueError, match="Missing vector field"):
        make_train_eval_datasets(cfg)
    cfg = config(source_payload)
    cfg.policy.max_action_dim = 3
    with pytest.raises(ValueError, match="exceed policy capacity"):
        make_train_eval_datasets(cfg)
