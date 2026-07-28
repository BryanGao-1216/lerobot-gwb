from types import SimpleNamespace

import torch
from torch import nn

from lerobot.common.tensorboard_utils import TensorBoardLogger


class _FakeSummaryWriter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.text = []
        self.scalars = []
        self.histograms = []
        self.flushed = False
        self.closed = False

    def add_text(self, *args, **kwargs):
        self.text.append((args, kwargs))

    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args, kwargs))

    def add_histogram(self, *args, **kwargs):
        self.histograms.append((args, kwargs))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def _make_cfg(tmp_path):
    policy = SimpleNamespace(
        tensorboard_log_dir="events",
        tensorboard_log_freq=5,
        tensorboard_histogram_freq=10,
        tensorboard_log_parameters=True,
        tensorboard_log_gradients=True,
        tensorboard_max_queue=7,
        tensorboard_flush_secs=11,
        tensorboard_filename_suffix=".actionmem",
    )
    return SimpleNamespace(
        policy=policy,
        output_dir=tmp_path,
        to_dict=lambda: {"policy": {"type": "actionmem"}},
    )


def test_tensorboard_logger_writes_scalars_and_trainable_histograms(tmp_path):
    logger = TensorBoardLogger(_make_cfg(tmp_path), purge_step=20, writer_cls=_FakeSummaryWriter)

    assert logger.log_dir == (tmp_path / "events").resolve()
    assert logger.writer.kwargs["purge_step"] == 20
    assert logger.should_log_scalars(5)
    assert not logger.should_log_scalars(6)
    assert logger.should_log_histograms(10)

    logger.log_scalars(
        {
            "loss": 1.25,
            "accuracy": 0.5,
            "ignored_bool": True,
            "ignored_text": "vlm_only",
            "ignored_nan": float("nan"),
        },
        step=5,
    )
    scalar_names = [args[0] for args, _ in logger.writer.scalars]
    assert scalar_names == ["train/loss", "train/accuracy"]

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model[1].weight.requires_grad_(False)
    model[1].bias.requires_grad_(False)

    logger.log_model_histograms(model, step=10)
    histogram_names = [args[0] for args, _ in logger.writer.histograms]
    assert histogram_names == [
        "parameters/0.weight",
        "gradients/0.weight",
        "parameters/0.bias",
        "gradients/0.bias",
    ]

    logger.flush()
    logger.close()
    assert logger.writer.flushed
    assert logger.writer.closed
