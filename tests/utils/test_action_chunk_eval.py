"""Check inference evaluation with analytic actions and real processor pipelines."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.common.action_chunk_eval import (
    action_chunk_metrics,
    evaluate_action_chunks,
    full_horizon_indices,
)
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import DataProcessorPipeline, NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action


def test_metrics_average_arms_and_exclude_padded_dimensions():
    target = torch.zeros(2, 2, 14)
    prediction = torch.zeros_like(target)
    target[0, :, 6] = prediction[0, :, 6] = 1
    prediction[0, :, :6] = 2
    # Ignored dimensions must not affect either metric or the finite-value check.
    prediction[0, :, 7:] = float("nan")
    target[0, :, 7:] = float("nan")
    prediction[1, :, :6] = 2
    prediction[1, :, 7:13] = 4
    target[1, :, 6] = prediction[1, :, 6] = -1
    target[1, :, 13] = 1
    prediction[1, :, 13] = -1
    padding = torch.tensor([[False] * 7 + [True] * 7, [False] * 14])
    actual = action_chunk_metrics(prediction, target, padding)
    torch.testing.assert_close(actual, torch.tensor([[2.0, 1.0, 4.0], [3.0, 0.5, 6.0]]))


def test_cumulative_error_sums_before_taking_absolute_value():
    target = torch.zeros(1, 2, 7)
    prediction = target.clone()
    prediction[0, 0, :6] = 2
    prediction[0, 1, :6] = -2
    actual = action_chunk_metrics(prediction, target)
    torch.testing.assert_close(actual, torch.tensor([[2.0, 1.0, 0.0]]))


def test_metrics_reject_unsupported_or_noncontiguous_dimensions():
    with pytest.raises(ValueError, match="7 or 14"):
        action_chunk_metrics(torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
    padding = torch.tensor([[False] * 6 + [True, False]])
    with pytest.raises(ValueError, match="suffix padding"):
        action_chunk_metrics(torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), padding)


def test_complete_horizons_in_selected_native_episodes():
    dataset = SimpleNamespace(
        episodes=[1, 2],
        meta=SimpleNamespace(
            total_episodes=3,
            episodes={"dataset_from_index": [0, 2, 6], "dataset_to_index": [2, 6, 11]},
        ),
        absolute_to_relative_idx={i: i - 2 for i in range(2, 11)},
    )
    assert full_horizon_indices(dataset, 3) == [0, 1, 4, 5, 6]
    assert full_horizon_indices(dataset, 6) == []


class InferencePolicy(torch.nn.Module):
    def __init__(self, normalized_actions, *, fail=False):
        super().__init__()
        self.register_buffer("actions", normalized_actions)
        self.config = SimpleNamespace(
            chunk_size=2, max_action_dim=14, image_features={"observation.images.main": None}
        )
        self.seen_noise = []
        self.fail = fail

    def forward(self, batch, *, action_inference=False, noise=None):
        assert action_inference and not self.training and not torch.is_grad_enabled()
        assert "action" not in batch
        assert batch["observation.images.main"].dtype == torch.float32
        assert batch["observation.images.main"].max() == 1
        self.seen_noise.extend(noise.detach().clone())
        # Exercising RNG preservation, including when inference raises.
        torch.rand(1)
        if self.fail:
            raise RuntimeError("inference failed")
        return self.actions[batch["dataset_index"]]


def inference_fixture():
    truth = torch.zeros(2, 2, 14)
    truth[0, :, 6] = 1
    truth[1, :, 6], truth[1, :, 13] = -1, 1
    prediction = truth.clone()
    prediction[0, :, :6] = 2
    prediction[1, :, :6] = 2
    prediction[1, :, 7:13] = 4
    prediction[1, :, 13] = -1
    features = {
        "action": PolicyFeature(FeatureType.ACTION, (14,)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    stats, normalized = {}, []
    for source, scale in enumerate((2.0, 4.0)):
        mean, std = torch.zeros(14), torch.full((14,), scale)
        # Normalized gripper signs deliberately differ from raw open/closed signs.
        mean[6], mean[13] = (10, 10) if source == 0 else (-10, -10)
        stats[f"dataset_{source}/action"] = {"mean": mean, "std": std}
        normalized.append((prediction[source] - mean) / std)
    normalization = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    preprocessor = DataProcessorPipeline(
        [NormalizerProcessorStep(features=features, norm_map=normalization, stats=stats)]
    )
    postprocessor = DataProcessorPipeline(
        [UnnormalizerProcessorStep(features=features, norm_map=normalization, stats=stats)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    rows = [
        {
            "index": torch.tensor(source + 12),
            "dataset_index": torch.tensor(source),
            "observation.state": torch.zeros(2),
            "observation.images.main": torch.full((3, 4, 4), 255, dtype=torch.uint8),
            "action": truth[source],
            "action_is_pad": torch.zeros(2, dtype=torch.bool),
            "action_dim_is_pad": torch.tensor([False] * 7 + [source == 0] * 7),
        }
        for source in range(2)
    ]
    return torch.stack(normalized), rows, preprocessor, postprocessor


def test_evaluation_uses_full_inference_and_source_stats_without_changing_training_rng():
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    normalized, rows, preprocessor, postprocessor = inference_fixture()
    policy = InferencePolicy(normalized)
    results, noises = [], []
    for batch_size in (1, 2):
        loader = accelerator.prepare(DataLoader(rows, batch_size=batch_size))
        rng_state = torch.get_rng_state()
        policy.seen_noise = []
        results.append(
            evaluate_action_chunks(
                policy, preprocessor, postprocessor, loader, accelerator, ["single", "dual"], seed=5
            )
        )
        noises.append(torch.stack(policy.seen_noise))
        assert torch.equal(torch.get_rng_state(), rng_state)
        assert policy.training
        assert postprocessor.to_transition is policy_action_to_transition
    assert results[0] == results[1]
    assert results[0]["action_rmse"] == 2.5
    assert results[0]["gripper_accuracy"] == 0.75
    assert results[0]["cumulative_action_error"] == 5
    assert results[0]["num_action_chunks"] == 2
    assert results[0]["single/action_rmse"] == 2
    assert results[0]["dual/action_rmse"] == 3
    torch.testing.assert_close(noises[0], noises[1])


def test_evaluation_restores_mode_and_rng_after_failure():
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    normalized, rows, preprocessor, postprocessor = inference_fixture()
    policy = InferencePolicy(normalized, fail=True)
    loader = accelerator.prepare(DataLoader(rows, batch_size=2))
    rng_state = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluate_action_chunks(policy, preprocessor, postprocessor, loader, accelerator, ["single", "dual"])
    assert policy.training
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_distributed_last_batch_duplicates_are_excluded(monkeypatch):
    from accelerate import Accelerator
    from accelerate.data_loader import prepare_data_loader

    accelerator = Accelerator(cpu=True)
    normalized, rows, preprocessor, postprocessor = inference_fixture()
    # Three real samples distributed into a global batch of four. The fourth
    # sample duplicates the first; simulate only the cross-rank tensor exchange.
    loader = prepare_data_loader(
        DataLoader([rows[0], rows[1], rows[0]], batch_size=2), num_processes=2, process_index=0
    )

    def gather(data):
        metrics, source_ids = data
        return torch.cat([metrics, metrics[[0, 0]]]), torch.cat([source_ids, source_ids[[0, 0]]])

    monkeypatch.setattr(accelerator, "gather", gather)
    result = evaluate_action_chunks(
        InferencePolicy(normalized), preprocessor, postprocessor, loader, accelerator, ["single", "dual"]
    )
    assert result["num_action_chunks"] == 3
    assert result["single/num_action_chunks"] == 2
    assert result["dual/num_action_chunks"] == 1
    assert result["action_rmse"] == pytest.approx(7 / 3)


@pytest.mark.parametrize("policy_type", ["pi05", "pi05_lap"])
def test_policy_forward_dispatches_to_complete_chunk_inference(policy_type):
    if policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as Policy
    else:
        from lerobot.policies.pi05_lap.modeling_pi05_lap import PI05LAPPolicy as Policy

    noise = torch.randn(1, 2, 14)
    dummy = SimpleNamespace(predict_action_chunk=lambda batch, **kwargs: kwargs["noise"])
    assert Policy.forward(dummy, {}, action_inference=True, noise=noise) is noise


def test_lap_action_inference_remains_disabled_during_latent_pretraining():
    from lerobot.policies.pi05_lap.modeling_pi05_lap import PI05LAPPolicy

    dummy = SimpleNamespace(config=SimpleNamespace(training_stage="pre-training"), eval=lambda: None)
    with pytest.raises(RuntimeError, match="requires post-training"):
        PI05LAPPolicy.predict_action_chunk(dummy, {})
