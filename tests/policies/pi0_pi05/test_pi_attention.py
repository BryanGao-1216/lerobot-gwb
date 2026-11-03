"""Attention equations, padding, gradients and real CUDA Flash kernel checks."""

import copy
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

pytest.importorskip("transformers")

from transformers import GemmaConfig

from lerobot.policies.common import pi_attention
from lerobot.policies.common.pi_attention import PiAttentionLayout, pi_attention_forward, prepare_pi_attention
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi_gemma import PiGemmaModel


def packed_math_reference(query, key, value, layout, scale):
    """CPU oracle for the packed kernel boundary; does not exercise CUDA Flash."""
    result = []
    q_start = k_start = 0
    for q_len, k_len in zip(layout.query_lengths, layout.key_lengths, strict=True):
        q = query[q_start : q_start + q_len].transpose(0, 1).unsqueeze(0)
        k = key[k_start : k_start + k_len].transpose(0, 1).unsqueeze(0)
        v = value[k_start : k_start + k_len].transpose(0, 1).unsqueeze(0)
        result.append(
            F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True).squeeze(0).transpose(0, 1)
        )
        q_start += q_len
        k_start += k_len
    return torch.cat(result)


def masks(device="cpu", kind="action"):
    # Internal camera padding, different prompt lengths, and an entirely padded prefix.
    pad = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    starts = torch.zeros_like(pad)
    if kind == "action":
        starts[:, 7] = True
    elif kind == "state_action":
        starts[:, 7:9] = True
    else:
        starts[:, 7:] = True
    return pad, starts


@pytest.mark.parametrize("kind", ["action", "state_action", "latent"])
@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
def test_valid_outputs_and_qkv_gradients_match_eager(monkeypatch, kind, backend):
    monkeypatch.setattr(pi_attention, "_flash_varlen", packed_math_reference)
    torch.manual_seed(47)
    pad, starts = masks(kind=kind)
    dense = prepare_attention_masks_4d(make_att_2d_masks(pad, starts))
    mask, layout = prepare_pi_attention(pad, starts, backend)
    tensors = [torch.randn(2, heads, 11, 32, requires_grad=True) for heads in (8, 1, 1)]
    reference = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    module = SimpleNamespace(num_key_value_groups=8, training=True)
    expected = pi_attention_forward(module, *reference, dense, 32**-0.5)
    actual = pi_attention_forward(module, *tensors, mask, 32**-0.5, backend=backend, layout=layout)
    # Padded queries are not training targets and are never visible as keys.
    torch.testing.assert_close(actual[pad], expected[pad], atol=5e-7, rtol=1e-5)
    upstream = torch.randn_like(expected[pad])
    (actual[pad] * upstream).sum().backward()
    (expected[pad] * upstream).sum().backward()
    for tensor, ref in zip(tensors, reference, strict=True):
        torch.testing.assert_close(tensor.grad, ref.grad, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize("policy_name", ["pi0", "pi05", "pi05_lap"])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
def test_joint_decoder_loss_and_parameter_gradients(monkeypatch, policy_name, checkpoint, backend):
    """Real PI blocks with RoPE, GQA, gated AdaRMS, residuals and recomputation."""
    monkeypatch.setattr(pi_attention, "_flash_varlen", packed_math_reference)
    policy_module = import_module(f"lerobot.policies.{policy_name}.modeling_{policy_name}")
    torch.manual_seed(24)
    model = policy_module.PaliGemmaWithExpertModel.__new__(policy_module.PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.freeze_vision_encoder = model.train_expert_only = False
    cfg = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "attention_dropout": 0.0,
        "use_adarms": False,
        "adarms_cond_dim": None,
    }
    language = PiGemmaModel(GemmaConfig(**cfg))
    expert = PiGemmaModel(GemmaConfig(**{**cfg, "use_adarms": True, "adarms_cond_dim": 32}))
    model.paligemma = nn.Module()
    model.paligemma.model = nn.Module()
    model.paligemma.model.language_model = language
    model.gemma_expert = nn.Module()
    model.gemma_expert.model = expert
    # Nonzero gates ensure the attention branch participates in the loss.
    for layer in expert.layers:
        nn.init.normal_(layer.input_layernorm.dense.weight, std=0.02)
        nn.init.normal_(layer.post_attention_layernorm.dense.weight, std=0.02)
    model.gradient_checkpointing = checkpoint
    model.train()
    reference = copy.deepcopy(model)
    pad, starts = masks(kind="state_action" if policy_name == "pi0" else "action")
    dense = prepare_attention_masks_4d(make_att_2d_masks(pad, starts))
    mask, layout = prepare_pi_attention(pad, starts, backend)
    inputs = [torch.randn(2, 7, 32), torch.randn(2, 4, 32)]
    condition = [None, torch.randn(2, 32)]
    positions = pad.cumsum(-1) - 1
    expected, _ = reference(
        attention_mask=dense, position_ids=positions, inputs_embeds=inputs, adarms_cond=condition
    )
    actual, _ = model(
        attention_mask=mask,
        position_ids=positions,
        inputs_embeds=inputs,
        adarms_cond=condition,
        attention_backend=backend,
        attention_layout=layout,
    )
    target = torch.randn_like(expected[1])
    ref_loss = F.mse_loss(expected[1], target)
    loss = F.mse_loss(actual[1], target)
    torch.testing.assert_close(loss, ref_loss, atol=2e-6, rtol=2e-5)
    loss.backward()
    ref_loss.backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        if parameter.grad is None:
            assert ref_parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=3e-6, rtol=2e-4, msg=name)


def test_flash_refuses_silent_precision_reduction():
    pad, starts = masks()
    layout = PiAttentionLayout.from_masks(pad, starts)
    q, k, v = [torch.randn(2, heads, 11, 32) for heads in (8, 1, 1)]
    with pytest.raises(ValueError, match="never downcasts"):
        pi_attention_forward(None, q, k, v, None, 32**-0.5, backend="flash_attention_2", layout=layout)


@pytest.mark.parametrize("policy_name", ["pi0", "pi05", "pi05_lap"])
def test_backend_config_validation_and_roundtrip(policy_name, tmp_path):
    config_module = import_module(f"lerobot.policies.{policy_name}.configuration_{policy_name}")
    config_class = getattr(
        config_module, {"pi0": "PI0Config", "pi05": "PI05Config", "pi05_lap": "PI05LAPConfig"}[policy_name]
    )
    assert config_class(device="cpu").attention_backend == "eager"
    with pytest.raises(ValueError, match="attention_backend"):
        config_class(device="cpu", attention_backend="causal")
    config = config_class(device="cpu", attention_backend="flash_attention_2")
    config.save_pretrained(tmp_path)
    assert config_class.from_pretrained(tmp_path).attention_backend == "flash_attention_2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA FlashAttention GPU")
@pytest.mark.parametrize("kind", ["action", "state_action", "latent"])
@pytest.mark.parametrize("force_sdpa", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_cuda_flash_bf16_forward_backward(monkeypatch, kind, force_sdpa, amp):
    """Actual native varlen and SDPA Flash kernels, using production Gemma head_dim=256."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 FlashAttention requires Ampere or newer")
    if force_sdpa:
        monkeypatch.setattr(pi_attention, "varlen_attn", None)
    torch.manual_seed(47)
    pad, starts = masks(device="cuda", kind=kind)
    dense = prepare_attention_masks_4d(make_att_2d_masks(pad, starts))
    layout = PiAttentionLayout.from_masks(pad, starts)
    dtype = torch.float32 if amp else torch.bfloat16
    tensors = [
        torch.randn(2, heads, 11, 256, device="cuda", dtype=dtype, requires_grad=True) for heads in (8, 1, 1)
    ]
    reference = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    module = SimpleNamespace(num_key_value_groups=8, training=True)
    with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
        expected = pi_attention_forward(module, *reference, dense, 256**-0.5)
        actual = pi_attention_forward(
            module, *tensors, None, 256**-0.5, backend="flash_attention_2", layout=layout
        )
    # BF16 kernels use different reduction orders; compare absolute error near zero too.
    torch.testing.assert_close(actual[pad], expected[pad], atol=1.5e-2, rtol=2e-2)
    upstream = torch.randn_like(actual[pad])
    (actual[pad] * upstream).float().sum().backward()
    (expected[pad] * upstream).float().sum().backward()
    for tensor, ref in zip(tensors, reference, strict=True):
        torch.testing.assert_close(tensor.grad, ref.grad, atol=8e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA FlashAttention GPU")
def test_cuda_flash_reduces_attention_peak_memory():
    """Attention-only forward/backward measurement, not a full-model VRAM claim."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 FlashAttention requires Ampere or newer")
    batch_size, sequence, action_length = 2, 1024, 50
    pad = torch.ones(batch_size, sequence, dtype=torch.bool, device="cuda")
    pad[:, 800:950] = False
    starts = torch.zeros_like(pad)
    starts[:, -action_length] = True
    module = SimpleNamespace(num_key_value_groups=8, training=True)

    def peak(backend):
        tensors = [
            torch.randn(
                batch_size, heads, sequence, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for heads in (8, 1, 1)
        ]
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        mask, layout = prepare_pi_attention(pad, starts, backend)
        output = pi_attention_forward(module, *tensors, mask, 256**-0.5, backend=backend, layout=layout)
        output[:, -action_length:].float().square().mean().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - baseline

    peak("flash_attention_2")  # Warm native kernel/module initialization.
    eager_peak = peak("eager")
    flash_peak = peak("flash_attention_2")
    print(
        f"Attention forward/backward peak: eager={eager_peak / 2**20:.1f} MiB, "
        f"flash={flash_peak / 2**20:.1f} MiB"
    )
    assert flash_peak < eager_peak * 0.75
