"""W&B logger initialization stays compatible without making network requests."""

import logging
import sys
from types import SimpleNamespace

import pytest

from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.default import WandBConfig


@pytest.mark.parametrize("mode", ["online", "offline", "disabled", "no_url"])
def test_logger_uses_run_url_property_and_preserves_metric_logging(tmp_path, monkeypatch, caplog, mode):
    class Run:
        # Deliberately no get_url(): reproduce the newer W&B Run API.
        id = "run123"
        disabled = mode == "disabled"
        offline = mode == "offline"

        @property
        def url(self):
            assert not (self.disabled or self.offline), "Offline/disabled runs have no online URL."
            return None if mode == "no_url" else "https://wandb.ai/team/project/runs/run123"

    run = Run()
    init_kwargs = {}
    logged = []

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    # Exercise the Run returned by init, independently of wandb.run's global state.
    wandb = SimpleNamespace(init=init, run=None, log=lambda **kwargs: logged.append(kwargs))
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setenv("WANDB_SILENT", "True")
    cfg = SimpleNamespace(
        wandb=WandBConfig(enable=True, project="pi05-lap", mode="online" if mode == "no_url" else mode),
        output_dir=tmp_path,
        job_name="pi05-lap-pretrain",
        env=None,
        dataset=SimpleNamespace(repo_id="libero"),
        policy=SimpleNamespace(type="pi05_lap"),
        is_reward_model_training=False,
        seed=42,
        resume=False,
        to_dict=lambda: {"policy": {"type": "pi05_lap"}},
    )
    with caplog.at_level(logging.INFO):
        logger = WandBLogger(cfg)
    assert cfg.wandb.run_id == run.id
    assert init_kwargs["project"] == "pi05-lap"
    assert init_kwargs["mode"] == cfg.wandb.mode
    assert ("Track this run -->" in caplog.text) == (mode == "online")
    if mode == "offline":
        assert str(tmp_path / "wandb") in caplog.text
    logger.log_dict({"loss": 1.25, "latent_token_accuracy": 0.5}, step=10)
    assert logged == [{"data": {"train/loss": 1.25, "train/latent_token_accuracy": 0.5}, "step": 10}]
