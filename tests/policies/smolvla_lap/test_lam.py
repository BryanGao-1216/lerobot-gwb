"""CPU software checks with tiny untrained models; no CUDA training or downloads."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def lam_package():
    # Load the optional inference package on its own, without importing all
    # registered policies or requiring a full training environment.
    name = "_test_smolvla_lap_lam"
    root = Path(__file__).resolve().parents[3] / "src/lerobot/policies/smolvla_lap/lam"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield package
    torch.set_num_threads(previous_threads)
    for module in list(sys.modules):
        if module == name or module.startswith(f"{name}."):
            del sys.modules[module]


@pytest.mark.parametrize(
    "name,size,codebook_size",
    [("univla", 224, 16), ("lapa", 256, 8), ("mvp_lam", 224, 16), ("mvp-lam", 224, 16)],
)
def test_teacher_spec_and_preprocessing(lam_package, name, size, codebook_size):
    specs = importlib.import_module(f"{lam_package.__name__}.specs")
    preprocessing = importlib.import_module(f"{lam_package.__name__}.preprocessing")
    spec = specs.get_lam_spec(name)
    assert (spec.image_size, spec.num_tokens, spec.codebook_size) == (size, 4, codebook_size)
    pair = torch.stack(
        [torch.zeros(3, 80, 96, dtype=torch.uint8), torch.full((3, 80, 96), 255, dtype=torch.uint8)]
    )
    before = pair.clone()
    result = preprocessing.preprocess_lam_pair(pair, spec.image_size)
    assert result.shape == (2, 3, size, size)
    assert result.dtype == torch.float32
    assert torch.all(result[0] == 0) and torch.all(result[1] == 1)
    assert torch.equal(pair, before)


@pytest.mark.parametrize(
    "name,module,class_name,codebook_size",
    [
        ("univla", "univla", "UniVLALAMEncoder", 16),
        ("lapa", "lapa", "LAPALAMEncoder", 8),
        ("mvp_lam", "mvp_lam", "MVPLAMEncoder", 16),
        ("mvp-lam", "mvp_lam", "MVPLAMEncoder", 16),
    ],
)
def test_factory_dispatch(lam_package, monkeypatch, name, module, class_name, codebook_size):
    backend = importlib.import_module(f"{lam_package.__name__}.{module}")
    expected = object()
    calls = []

    def create(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(backend, class_name, create)
    config = SimpleNamespace(
        lam_type=name,
        lam_model_path="teacher.pt",
        lam_batch_size=2,
        lam_num_tokens=4,
        lam_codebook_size=codebook_size,
    )
    assert lam_package.make_lam_encoder(config, "cuda:1") is expected
    assert calls == [("teacher.pt", "cuda:1", 2)]
    config.lam_codebook_size += 1
    with pytest.raises(ValueError, match="token specification"):
        lam_package.make_lam_encoder(config, "cuda:1")


@pytest.mark.parametrize("wrapper", [None, "state_dict", "model_state_dict"])
@pytest.mark.parametrize("zip_format", [False, True])
def test_checkpoint_wrappers_and_prefixes(lam_package, tmp_path, wrapper, zip_format):
    common = importlib.import_module(f"{lam_package.__name__}.common")
    weights = {"module.lam.weight": torch.arange(4, dtype=torch.float32)}
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {wrapper: weights} if wrapper else weights, checkpoint, _use_new_zipfile_serialization=zip_format
    )
    restored = common.load_lam_state(checkpoint)
    assert list(restored) == ["weight"]
    assert torch.equal(restored["weight"], weights["module.lam.weight"])


def test_lapa_original_inference_path(lam_package):
    model_module = importlib.import_module(f"{lam_package.__name__}.lapa_model")
    backend = importlib.import_module(f"{lam_package.__name__}.lapa")
    specs = importlib.import_module(f"{lam_package.__name__}.specs")
    torch.manual_seed(42)
    model = (
        model_module.LatentActionQuantization(
            dim=32,
            quant_dim=8,
            codebook_size=8,
            image_size=256,
            patch_size=32,
            spatial_depth=1,
            temporal_depth=1,
            heads=2,
            dim_head=16,
            code_seq_len=4,
        )
        .eval()
        .requires_grad_(False)
    )
    encoder = backend.LAPALAMEncoder.__new__(backend.LAPALAMEncoder)
    encoder.model, encoder.device = model, torch.device("cpu")
    encoder.spec, encoder.batch_size = specs.get_lam_spec("lapa"), 1
    frames = torch.rand(2, 2, 3, 256, 256)
    usage = model.vq.codebooks_used.clone()
    with torch.no_grad():
        reference = model.inference(frames.permute(0, 2, 1, 3, 4).contiguous(), return_only_codebook_ids=True)
    actual = encoder.encode(frames)
    assert actual.shape == (2, 4) and actual.dtype == torch.long
    assert torch.equal(actual, reference)
    assert torch.equal(actual, encoder.encode(frames))
    assert torch.equal(usage, model.vq.codebooks_used)
    assert actual.min() >= 0 and actual.max() < 8
    with pytest.raises(ValueError, match="expects LAM frames"):
        encoder.encode(torch.zeros(1, 2, 3, 224, 224))


@pytest.mark.parametrize(
    "name,class_name,model_class",
    [
        ("univla", "UniVLALAMEncoder", "ControllableDINOLatentActionModel"),
        ("mvp_lam", "MVPLAMEncoder", "DINOLatentActionModel"),
    ],
)
def test_dino_encoder_inference_paths(lam_package, monkeypatch, name, class_name, model_class):
    monkeypatch.setenv("XFORMERS_DISABLED", "1")
    model_name = "mvp_model" if name == "mvp_lam" else "univla_model"
    module = importlib.import_module(f"{lam_package.__name__}.{model_name}")
    backend = importlib.import_module(f"{lam_package.__name__}.{name}")
    specs = importlib.import_module(f"{lam_package.__name__}.specs")

    class TinyDINO(nn.Module):
        def forward_features(self, video):
            signal = video.mean(dim=(1, 2, 3))
            return {"x_norm_patchtokens": signal[:, None, None].expand(-1, 2, 768)}

    monkeypatch.setattr(module, "vit_base", lambda **kwargs: TinyDINO())
    torch.manual_seed(42)
    model = (
        getattr(module, model_class)(
            in_dim=3,
            model_dim=24,
            latent_dim=8,
            num_latents=16,
            patch_size=14,
            enc_blocks=1,
            dec_blocks=1,
            num_heads=3,
            dropout=0.0,
        )
        .eval()
        .requires_grad_(False)
    )
    cls = getattr(backend, class_name)
    encoder = cls.__new__(cls)
    encoder.model, encoder.device = model, torch.device("cpu")
    encoder.spec, encoder.batch_size = specs.get_lam_spec(name), 1
    frames = torch.rand(2, 2, 3, 224, 224)
    usage = model.vq.usage.clone()
    with torch.no_grad():
        reference = model.vq_encode(frames)["indices"].reshape(2, 4)
    actual = encoder.encode(frames)
    assert torch.equal(actual, reference)
    assert torch.equal(actual, encoder.encode(frames))
    assert torch.equal(usage, model.vq.usage)
    assert actual.min() >= 0 and actual.max() < 16


def test_reject_wrong_checkpoint_kind(lam_package):
    lapa = importlib.import_module(f"{lam_package.__name__}.lapa")
    univla = importlib.import_module(f"{lam_package.__name__}.univla")
    specs = importlib.import_module(f"{lam_package.__name__}.specs")
    teacher = lapa.LAPALAMEncoder.__new__(lapa.LAPALAMEncoder)
    teacher.spec = specs.get_lam_spec("lapa")
    with pytest.raises(ValueError, match="vq.codebooks"):
        teacher._validate_state({"vq.codebook.weight": torch.zeros(16, 128)})
    teacher = univla.UniVLALAMEncoder.__new__(univla.UniVLALAMEncoder)
    teacher.spec = specs.get_lam_spec("univla")
    with pytest.raises(ValueError, match="stage-2"):
        teacher._validate_state({"vq_action.codebook.weight": torch.zeros(16, 128)})
