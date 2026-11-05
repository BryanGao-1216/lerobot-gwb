"""Numerical checks for accumulated optimizer updates, configuration and data resume."""

from copy import deepcopy

import draccus
import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.common.train_utils import (
    load_training_data_batches_seen,
    load_training_state,
    save_training_state,
    save_training_step,
)
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.sampler import WeightedDatasetSampler, compute_sampler_state
from lerobot.policies.pi05_lap.configuration_pi05_lap import PI05LAPConfig
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.constants import TRAINING_STATE_DIR
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.utils import cycle


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.3, -0.2]]))
            self.linear.bias.fill_(0.1)
        self.weights_seen = []
        self.sync_seen = []
        self.on_forward = lambda: True
        self.update_count = 0

    def forward(self, batch):
        self.weights_seen.append(self.linear.weight.detach().clone())
        self.sync_seen.append(self.on_forward())
        loss = (self.linear(batch["x"]) - batch["y"]).square().mean()
        return loss, {"action_loss": loss.detach()}

    def update(self):
        self.update_count += 1


def accelerator_with_accumulation(steps):
    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin

    return Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=steps, sync_with_dataloader=False),
        step_scheduler_with_optimizer=False,
    )


def tracker(accelerator, batch_size):
    return MetricsTracker(
        batch_size=batch_size,
        num_frames=100,
        num_episodes=10,
        metrics={key: AverageMeter(key) for key in ("loss", "grad_norm", "lr", "update_s")},
        accelerator=accelerator,
    )


def data():
    x = torch.tensor([[1.0, 2], [3, 4], [-2, 1], [3, -1], [0, 3], [2, 5]])
    y = torch.tensor([[1.0], [-1], [3], [2], [4], [-2]])
    return x, y


@pytest.mark.parametrize("clip", [0.0, 0.25])
@pytest.mark.parametrize("sizes", [[6], [2, 2, 2], [2, 1, 3]])
def test_accumulation_matches_large_batch_and_updates_scheduler_once(sizes, clip, monkeypatch):
    accelerator = accelerator_with_accumulation(len(sizes))
    model, reference = TinyPolicy(), TinyPolicy()
    initial_weight = model.linear.weight.detach().clone()
    raw_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(raw_optimizer, step_size=1, gamma=0.5)
    model, optimizer, scheduler = accelerator.prepare(model, raw_optimizer, scheduler)
    model.on_forward = lambda: accelerator.sync_gradients
    metrics = tracker(accelerator, 6)
    x, y = data()
    batches, start = [], 0
    for size in sizes:
        batches.append({"x": x[start : start + size], "y": y[start : start + size]})
        start += size

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference_loss, _ = reference({"x": x, "y": y})
    reference_loss.backward()
    reference_norm = torch.nn.utils.clip_grad_norm_(
        reference.parameters(), clip if clip > 0 else float("inf")
    )
    reference_optimizer.step()
    calls, clip_grad = [], accelerator.clip_grad_norm_

    def count_clip(*args, **kwargs):
        calls.append(accelerator.sync_gradients)
        return clip_grad(*args, **kwargs)

    monkeypatch.setattr(accelerator, "clip_grad_norm_", count_clip)
    supplied = batches[0] if len(sizes) == 1 else iter(batches)
    metrics, output = update_policy(metrics, model, supplied, optimizer, clip, accelerator, scheduler)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
    assert calls == [True]
    assert model.update_count == 1
    assert scheduler.scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == 0.05
    assert model.sync_seen == [False] * (len(sizes) - 1) + [True]
    assert all(torch.equal(weight, initial_weight) for weight in model.weights_seen)
    assert metrics.loss.val == pytest.approx(reference_loss.item(), rel=1e-6)
    assert metrics.grad_norm.val == pytest.approx(reference_norm.item(), rel=1e-6)
    assert output["action_loss"] == pytest.approx(reference_loss.item(), rel=1e-6)


def test_window_crosses_epoch_boundary_without_early_update():
    accelerator = accelerator_with_accumulation(4)
    model = TinyPolicy()
    raw_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = data()
    rows = [{"x": a, "y": b} for a, b in zip(x[:5], y[:5], strict=True)]
    model, optimizer, loader = accelerator.prepare(model, raw_optimizer, DataLoader(rows, batch_size=2))
    model.on_forward = lambda: accelerator.sync_gradients
    batches = cycle(loader)
    reference = TinyPolicy()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss, _ = reference({"x": torch.cat([x[:5], x[:2]]), "y": torch.cat([y[:5], y[:2]])})
    loss.backward()
    reference_optimizer.step()
    update_policy(
        tracker(accelerator, 8), model, (next(batches) for _ in range(4)), optimizer, 0, accelerator
    )
    assert model.sync_seen == [False, False, False, True]
    assert model.update_count == 1
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("stage", ["pre-training", "post-training"])
def test_cli_and_saved_config_preserve_accumulation(stage):
    config = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=test/dataset",
            "--policy.type=pi05_lap",
            "--policy.device=cpu",
            f"--policy.training_stage={stage}",
            "--policy.gradient_accumulation=4",
        ],
    )
    assert config.policy.gradient_accumulation == 4
    saved = draccus.decode(TrainPipelineConfig, draccus.encode(config))
    assert saved.policy.gradient_accumulation == 4
    old = deepcopy(draccus.encode(config))
    old["policy"].pop("gradient_accumulation")
    assert draccus.decode(TrainPipelineConfig, old).policy.gradient_accumulation == 1


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_accumulation_requires_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        PI05LAPConfig(device="cpu", gradient_accumulation=value)


def test_overflow_does_not_advance_scheduler_or_model_update(monkeypatch):
    accelerator = accelerator_with_accumulation(2)
    model = TinyPolicy()
    initial = deepcopy(model.state_dict())
    raw_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(raw_optimizer, step_size=1)
    model, optimizer, scheduler = accelerator.prepare(model, raw_optimizer, scheduler)
    monkeypatch.setattr(optimizer, "step", lambda: setattr(optimizer, "_is_overflow", True))
    x, y = data()
    batches = iter([{"x": x[:3], "y": y[:3]}, {"x": x[3:], "y": y[3:]}])
    update_policy(tracker(accelerator, 6), model, batches, optimizer, 0, accelerator, scheduler)
    assert accelerator.optimizer_step_was_skipped
    assert model.update_count == 0
    assert scheduler.scheduler.last_epoch == 0
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, initial[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_resume_uses_consumed_batches_when_accumulation_changes(tmp_path):
    state_dir = tmp_path / TRAINING_STATE_DIR
    state_dir.mkdir()
    save_training_step(2, state_dir, num_processes=1, batch_size=2)
    assert load_training_data_batches_seen(tmp_path) is None
    save_training_state(tmp_path, 2, num_processes=1, batch_size=2, data_batches_seen=4)
    sampler = WeightedDatasetSampler([7, 4], [1, 3], seed=5, batch_multiple=2)
    original = list(sampler)
    sampler.load_state_dict(
        compute_sampler_state(load_training_data_batches_seen(tmp_path), len(sampler), 2, 1)
    )
    assert list(sampler) == original[8:]
    # One further optimizer update with N=3 consumes three batches, not two.
    save_training_state(tmp_path, 3, num_processes=1, batch_size=2, data_batches_seen=7)
    assert load_training_data_batches_seen(tmp_path) == 7
    assert compute_sampler_state(load_training_data_batches_seen(tmp_path), len(sampler), 2, 1) == {
        "epoch": 1,
        "start_index": 2,
    }


@pytest.mark.parametrize("micro_steps", [1, 3, 8])
def test_metric_readback_happens_once_after_update_without_retaining_graphs(micro_steps, monkeypatch):
    import lerobot.scripts.lerobot_train as training

    class MetricPolicy(TinyPolicy):
        def forward(self, batch):
            loss, metrics = super().forward(batch)
            metrics.update(
                {
                    "loss_per_dim": self.linear.weight.square().flatten(),
                    "python_metric": 1.5,
                    "enabled": True,
                    "label": "diagnostic",
                }
            )
            return loss, metrics

    accelerator = accelerator_with_accumulation(micro_steps)
    model = MetricPolicy()
    expected_per_dim = model.linear.weight.detach().square().flatten().tolist()
    model, optimizer = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    x, y = data()
    batches = ({"x": x[:2], "y": y[:2]} for _ in range(micro_steps))
    readbacks, cpu_calls = [], []
    materialize = training._metrics_to_python
    original_cpu = torch.Tensor.cpu

    def snapshot(values):
        assert len(model.weights_seen) == micro_steps
        assert model.update_count == 1
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        assert all(not value.requires_grad and value.grad_fn is None for value in tensors)
        # Statistics occupy the same fixed-size buffers, regardless of accumulation length.
        readbacks.append(sum(value.numel() for value in tensors))
        return materialize(values)

    def cpu(tensor, *args, **kwargs):
        cpu_calls.append(len(model.weights_seen))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(training, "_metrics_to_python", snapshot)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    # Scalar reads inside forward/backward or the accumulation loop must not be needed.
    monkeypatch.setattr(torch.Tensor, "item", lambda *args: pytest.fail("unexpected scalar readback"))
    metrics, output = update_policy(
        tracker(accelerator, micro_steps * 2), model, batches, optimizer, 0, accelerator
    )
    assert readbacks == [5]  # optimized loss, gradient norm, action loss and two per-dimension values
    assert cpu_calls == [micro_steps]
    assert output["loss_per_dim"] == pytest.approx(expected_per_dim)
    assert isinstance(output["action_loss"], float)
    assert output["action_loss"] == metrics.loss.val
    assert output["python_metric"] == 1.5
    assert output["enabled"] is True and output["label"] == "diagnostic"


def test_legacy_python_scalar_policy_metrics_still_work():
    class LegacyPolicy(TinyPolicy):
        def forward(self, batch):
            loss, metrics = super().forward(batch)
            return loss, {key: value.item() for key, value in metrics.items()}

    accelerator = accelerator_with_accumulation(3)
    model = LegacyPolicy()
    model, optimizer = accelerator.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    x, y = data()
    batches = ({"x": x[i : i + 2], "y": y[i : i + 2]} for i in range(0, 6, 2))
    metrics, output = update_policy(tracker(accelerator, 6), model, batches, optimizer, 0, accelerator)
    assert output["action_loss"] == pytest.approx(metrics.loss.val)


def test_existing_adamw_checkpoint_resumes_to_same_weights_and_optimizer_state(tmp_path):
    from safetensors.torch import load_file, save_file

    # Produce a checkpoint with the previous scalar-logging behavior and unchanged
    # weights/optimizer/scheduler layout, then continue it through update_policy.
    reference = TinyPolicy()
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.03)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.9)
    x, y = data()

    def reference_update():
        loss, _ = reference({"x": x, "y": y})
        _ = loss.detach().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25)
        reference_optimizer.step()
        reference_optimizer.zero_grad()
        reference_scheduler.step()

    reference_update()
    save_file(reference.state_dict(), tmp_path / "model.safetensors")
    save_training_state(
        tmp_path,
        1,
        reference_optimizer,
        reference_scheduler,
        num_processes=1,
        batch_size=2,
        data_batches_seen=3,
    )
    reference_update()

    model = TinyPolicy()
    model.load_state_dict(load_file(tmp_path / "model.safetensors"), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    step, optimizer, scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert step == 1 and load_training_data_batches_seen(tmp_path) == 3
    accelerator = accelerator_with_accumulation(3)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    batches = iter([{"x": x[:2], "y": y[:2]}, {"x": x[2:3], "y": y[2:3]}, {"x": x[3:], "y": y[3:]}])
    update_policy(tracker(accelerator, 6), model, batches, optimizer, 0.25, accelerator, scheduler)
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(optimizer.state_dict(), reference_optimizer.state_dict(), atol=1e-7, rtol=1e-6)
    assert scheduler.state_dict() == reference_scheduler.state_dict()
