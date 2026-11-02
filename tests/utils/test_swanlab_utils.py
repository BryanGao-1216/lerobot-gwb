"""SwanLab migration, checkpoint resume and metric logging without cloud access."""

import json
import logging
import signal
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import draccus
import pytest

from lerobot.common.swanlab_utils import SwanLabLogger, experiment_logging, make_experiment_logger
from lerobot.configs.default import DatasetConfig, SwanLabConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05_lap.configuration_pi05_lap import PI05LAPConfig


@pytest.fixture
def sdk(monkeypatch):
    class Run:
        id = "swanlab-run-123"

        @property
        def url(self):
            assert self.kwargs["mode"] == "online", "Offline and disabled runs have no cloud URL."
            return "https://swanlab.cn/@user/pi05-lap/runs/swanlab-run-123"

        def log(self, data, step):
            self.logged.append((step, data))

        def finish(self, *, state="success", error=None):
            self.finished = True
            self.alive = False
            self.finish_calls.append((state, error))

    run = Run()
    run.kwargs = {}
    run.logged = []
    run.finished = False
    run.alive = True
    run.finish_calls = []

    def init(**kwargs):
        run.kwargs = kwargs
        run.alive = True
        return run

    # Match SwanLab >=0.10's Run API: url property, no get_url() method.
    fake = SimpleNamespace(
        init=init,
        Text=lambda text: ("text", text),
        Video=lambda path: ("video", path),
        run=run,
        __spec__=ModuleSpec("swanlab", loader=None),
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    # Any accidental initialization of W&B must fail, including on resume.
    monkeypatch.setitem(sys.modules, "wandb", None)
    return fake


def make_config(tmp_path, mode="online"):
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="libero_only", video_backend="pyav"),
        policy=PI05LAPConfig(device="cpu", training_stage="pre-training"),
        output_dir=tmp_path,
        job_name="pi05-lap-pretrain",
        swanlab=SwanLabConfig(enable=True, project="pi05-lap", mode=mode),
    )


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_modes_log_optimizer_steps_and_finish(tmp_path, sdk, caplog, mode):
    cfg = make_config(tmp_path, mode)
    with caplog.at_level(logging.INFO):
        logger = make_experiment_logger(cfg)
    assert ("https://swanlab.cn" in caplog.text) == (mode == "online")
    assert sdk.run.kwargs["name"] == "pi05-lap-pretrain"
    assert sdk.run.kwargs["log_dir"] == str(tmp_path / "swanlog")
    assert sdk.run.kwargs["tags"] == ["policy:pi05_lap", "seed:1000", "dataset:libero_only"]
    logger.log_dict({"loss": 1.2, "latent_token_accuracy": 0.75, "lr": 1e-5}, step=100010)
    # Evaluation may be logged separately at the same completed optimizer step.
    logger.log_dict({"eval_loss": 1.1}, step=100010, mode="eval")
    logger.finish()
    if mode == "disabled":
        assert cfg.swanlab.run_id is None
        assert sdk.run.logged == []
    else:
        assert cfg.swanlab.run_id == sdk.run.id
        assert sdk.run.logged == [
            (100010, {"train/loss": 1.2, "train/latent_token_accuracy": 0.75, "train/lr": 1e-5}),
            (100010, {"eval/eval_loss": 1.1}),
        ]
    assert sdk.run.finished


def test_old_wandb_config_migrates_and_swanlab_id_survives_checkpoint_roundtrip(tmp_path, sdk):
    cfg = make_config(tmp_path)
    cfg.wandb = WandBConfig(enable=True, project="old-wandb-project", run_id="old-wandb-id")
    cfg.swanlab.enable = False
    legacy = cfg.to_dict()
    legacy.pop("swanlab")  # Real old checkpoints have no SwanLab fields at all.
    legacy_path = tmp_path / "train_config.json"
    legacy_path.write_text(json.dumps(legacy))

    migrated = TrainPipelineConfig.from_pretrained(
        legacy_path,
        cli_args=["--resume=true", "--steps=500000", "--swanlab.enable=true", "--swanlab.project=pi05-lap"],
    )
    assert migrated.resume and migrated.steps == 500000
    logger = make_experiment_logger(migrated)
    assert isinstance(logger, SwanLabLogger)
    assert not migrated.wandb.enable
    assert migrated.wandb.run_id == "old-wandb-id"
    assert sdk.run.kwargs["id"] is None  # Never resume SwanLab with a W&B ID.
    assert sdk.run.kwargs["resume"] is None
    migrated._save_pretrained(tmp_path)
    logger.finish()

    resumed = TrainPipelineConfig.from_pretrained(legacy_path)
    assert resumed.swanlab.run_id == sdk.run.id
    assert resumed.swanlab.project == "pi05-lap"
    logger = make_experiment_logger(resumed)
    assert sdk.run.kwargs["id"] == sdk.run.id
    assert sdk.run.kwargs["resume"] == "allow"
    logger.log_dict({"loss": 0.5}, step=200010)
    assert sdk.run.logged[-1] == (200010, {"train/loss": 0.5})


def test_disabled_logging_does_not_import_sdks(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg.swanlab.enable = False
    monkeypatch.setitem(sys.modules, "swanlab", None)
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert make_experiment_logger(cfg) is None


def test_swanlab_reports_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    with pytest.raises(ImportError, match="training extra"):
        make_experiment_logger(make_config(tmp_path))


def test_mp4_eval_video_is_logged_as_gif_without_uploading_checkpoints(tmp_path, sdk):
    import av
    import numpy as np
    from PIL import Image

    video_path = tmp_path / "eval.mp4"
    # A real short MP4 exercises the same PyAV decode path used for robot rollouts.
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), index * 80, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    logger = make_experiment_logger(make_config(tmp_path))
    logger.log_video(str(video_path), step=100000, mode="eval")
    step, logged = sdk.run.logged[-1]
    assert step == 100000
    gif_path = Path(logged["eval/video"][1])
    assert gif_path.suffix == ".gif" and video_path.exists()
    with Image.open(gif_path) as gif:
        assert gif.n_frames == 3
        assert gif.info["duration"] == 100
    # The public cloud does not support model file uploads; this needs no save/Artifact API.
    logger.log_policy(tmp_path / "checkpoints" / "100000")


def test_swanlab_config_cli_validation():
    with pytest.raises(draccus.utils.ParsingError) as exc:
        draccus.parse(SwanLabConfig, args=["--mode=invalid"])
    assert "swanlab.mode" in str(exc.value.__cause__)
    with pytest.raises(ValueError, match="swanlab.project"):
        SwanLabConfig(enable=True, project=" ")


@pytest.mark.parametrize(
    "error,expected_state",
    [
        (KeyboardInterrupt(), "aborted"),
        (RuntimeError("model initialization failed"), "crashed"),
        (SystemExit(0), "success"),
        (SystemExit(1), "crashed"),
    ],
)
def test_context_submits_exit_state_and_preserves_training_exception(tmp_path, sdk, error, expected_state):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(type(error)) as caught, experiment_logging(make_config(tmp_path)):
        raise error
    assert caught.value is error
    assert sdk.run.finish_calls[0][0] == expected_state
    assert type(error).__name__ in sdk.run.finish_calls[0][1]
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_actual_signals_unwind_training_before_finishing(tmp_path, sdk, signum):
    previous = signal.getsignal(signum)
    unwound = []
    sdk_finish = sdk.run.finish

    def finish(**kwargs):
        assert unwound == [True], "Finish must run after unwinding the interrupted training operation."
        sdk_finish(**kwargs)

    sdk.run.finish = finish
    exception_type = KeyboardInterrupt if signum == signal.SIGINT else SystemExit
    with pytest.raises(exception_type), experiment_logging(make_config(tmp_path)):
        try:
            signal.raise_signal(signum)
        finally:
            unwound.append(True)
    assert sdk.run.finish_calls[0][0] == "aborted"
    assert signal.getsignal(signum) == previous


def test_context_finishes_once_on_success(tmp_path, sdk):
    with experiment_logging(make_config(tmp_path)) as logger:
        logger.log_dict({"loss": 1.0}, step=10)
    assert sdk.run.finish_calls == [("success", None)]
    logger.finish()
    assert len(sdk.run.finish_calls) == 1


def test_shutdown_failure_preserves_training_error(tmp_path, sdk, caplog):
    def fail_finish(**kwargs):
        raise ConnectionError("offline during shutdown")

    sdk.run.finish = fail_finish
    error = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError) as caught, experiment_logging(make_config(tmp_path)):
        raise error
    assert caught.value is error
    assert "SwanLab shutdown did not complete" in caplog.text


def test_non_main_process_never_initializes_or_closes_sdk(tmp_path, sdk):
    with experiment_logging(make_config(tmp_path), enabled=False) as logger:
        assert logger is None
    assert sdk.run.kwargs == {} and sdk.run.finish_calls == []


def test_training_initialization_is_inside_logger_context(tmp_path, sdk, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    import lerobot.scripts.lerobot_train as training

    cfg = make_config(tmp_path)
    cfg.seed = None
    monkeypatch.setattr(cfg, "validate", lambda: None)
    monkeypatch.setattr(training, "init_logging", lambda **kwargs: None)
    error = RuntimeError("dataset initialization failed")

    def fail_dataset(cfg):
        raise error

    monkeypatch.setattr(training, "make_train_eval_datasets", fail_dataset)
    accelerator = SimpleNamespace(
        is_main_process=True, device="cpu", gradient_state=SimpleNamespace(plugin_kwargs={})
    )
    with pytest.raises(RuntimeError) as caught:
        training.train(cfg, accelerator=accelerator)
    assert caught.value is error
    assert sdk.run.finish_calls[0][0] == "crashed"
