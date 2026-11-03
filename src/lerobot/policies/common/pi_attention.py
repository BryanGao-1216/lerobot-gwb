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

"""Memory-efficient training attention for PI policies, including LAP's prefix LM.

Flash kernels cannot consume PI's arbitrary dense mask. Instead, each block of
valid queries attends to all valid keys in that block and earlier blocks. These
independent, noncausal attentions are packed into one variable-length Flash call.
RoPE, scaling, GQA and the caller's precision/AMP settings are preserved.
"""

from dataclasses import dataclass
from inspect import signature

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # torch < 2.10: use public SDPA Flash kernels per block.
    varlen_attn = None


PI_ATTENTION_BACKENDS = ("eager", "sdpa", "flash_attention_2")


def prepare_pi_attention(padding: Tensor, block_starts: Tensor, backend: str):
    """Avoid constructing the quadratic mask when packed Flash is selected."""
    if backend == "flash_attention_2":
        return None, PiAttentionLayout.from_masks(padding, block_starts)
    from .vla_utils import make_att_2d_masks, prepare_attention_masks_4d

    return prepare_attention_masks_4d(make_att_2d_masks(padding, block_starts)), None


@dataclass(frozen=True)
class PiAttentionLayout:
    """O(tokens * blocks) packing metadata, built once and reused by every layer.

    Indices refer to flattened [batch, sequence] axes. Padding queries are not
    packed: they cannot contribute to valid queries or the PI/LAP training loss.
    """

    query_indices: Tensor
    key_indices: Tensor
    cu_query: Tensor
    cu_key: Tensor
    query_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]

    @classmethod
    @torch.compiler.disable
    def from_masks(cls, padding: Tensor, block_starts: Tensor) -> "PiAttentionLayout":
        # Copy only O(B*S) metadata to CPU, never the quadratic attention mask.
        # Keep data-dependent packing outside compiled decoder/checkpoint graphs.
        valid = padding.detach().to(device="cpu", dtype=torch.bool)
        blocks = block_starts.detach().to(device="cpu", dtype=torch.int64).cumsum(-1)
        if valid.ndim != 2 or blocks.shape != valid.shape:
            raise ValueError(
                "PI attention padding and block starts must have the same [batch, sequence] shape."
            )
        query_indices, key_indices, query_lengths, key_lengths = [], [], [], []
        sequence_length = valid.shape[1]
        for batch_index in range(valid.shape[0]):
            positions = valid[batch_index].nonzero().flatten()
            block_ids = blocks[batch_index, positions]
            for block_id in block_ids.unique_consecutive():
                queries = positions[block_ids == block_id]
                keys = positions[block_ids <= block_id]
                query_indices.extend((queries + batch_index * sequence_length).tolist())
                key_indices.extend((keys + batch_index * sequence_length).tolist())
                query_lengths.append(len(queries))
                key_lengths.append(len(keys))

        def cumulative(lengths):
            return torch.tensor([0, *lengths], device=padding.device, dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            )

        return cls(
            torch.tensor(query_indices, device=padding.device, dtype=torch.long),
            torch.tensor(key_indices, device=padding.device, dtype=torch.long),
            cumulative(query_lengths),
            cumulative(key_lengths),
            tuple(query_lengths),
            tuple(key_lengths),
        )


def _flash_varlen(query: Tensor, key: Tensor, value: Tensor, layout: PiAttentionLayout, scale: float):
    """Run only Flash kernels; an unsupported device/dtype must never fall back to dense attention."""
    if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "PI flash_attention_2 requires CUDA float16/bfloat16 Q/K/V. "
            "Use BF16 autocast for training, or choose attention_backend='sdpa'/'eager'. "
            "The attention backend never downcasts float32 inputs outside autocast."
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("PI FlashAttention requires matching Q/K/V dtypes; no implicit casts are performed.")
    if varlen_attn is not None:
        parameters = signature(varlen_attn).parameters
        kwargs = {}
        if "scale" in parameters:
            kwargs["scale"] = scale
        elif scale != query.shape[-1] ** -0.5:
            raise ValueError("This PyTorch varlen API only supports the default attention scaling.")
        if "window_size" in parameters:
            kwargs["window_size"] = (-1, -1)
        else:
            kwargs["is_causal"] = False
        return varlen_attn(
            query,
            key,
            value,
            layout.cu_query,
            layout.cu_key,
            max(layout.query_lengths),
            max(layout.key_lengths),
            **kwargs,
        )

    # Same equations on older torch versions, with one Flash launch per block.
    outputs = []
    query_start = key_start = 0
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for query_length, key_length in zip(layout.query_lengths, layout.key_lengths, strict=True):
            q = query[query_start : query_start + query_length].transpose(0, 1).unsqueeze(0)
            k = key[key_start : key_start + key_length].transpose(0, 1).unsqueeze(0)
            v = value[key_start : key_start + key_length].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
            outputs.append(out.squeeze(0).transpose(0, 1))
            query_start += query_length
            key_start += key_length
    return torch.cat(outputs)


def pi_attention_forward(
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    *,
    backend: str = "eager",
    layout: PiAttentionLayout | None = None,
    dropout: float = 0.0,
) -> Tensor:
    """Return [batch, query, heads, head_dim], matching Gemma's eager interface."""
    if backend == "eager":
        # Preserve the original operation order and precision, including under AMP.
        from transformers.models.gemma.modeling_gemma import eager_attention_forward

        return eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=dropout)[0]
    if backend == "sdpa":
        return (
            F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout,
                scale=scaling,
                enable_gqa=query.shape[1] != key.shape[1],
            )
            .transpose(1, 2)
            .contiguous()
        )
    if backend != "flash_attention_2":
        raise ValueError(f"Unknown PI attention backend: {backend!r}")
    if layout is None:
        raise ValueError(
            "PI FlashAttention requires block/padding metadata; a causal flag is not equivalent."
        )
    if dropout:
        raise ValueError("PI packed FlashAttention requires attention_dropout=0.")
    if query.shape[2] != key.shape[2]:
        raise ValueError("PI packed FlashAttention is for training without a KV cache.")

    # Native varlen does not apply autocast itself. Match eager's matmul autocast,
    # including the LAP language path where RoPE can promote Q/K to float32.
    # Outside AMP, _flash_varlen rejects float32 rather than silently downcasting.
    if torch.is_autocast_enabled(query.device.type):
        compute_dtype = torch.get_autocast_dtype(query.device.type)
        query, key, value = (tensor.to(compute_dtype) for tensor in (query, key, value))

    batch, heads, sequence, head_dim = query.shape
    q = query.transpose(1, 2).reshape(batch * sequence, heads, head_dim)
    k = key.transpose(1, 2).reshape(batch * sequence, key.shape[1], head_dim)
    v = value.transpose(1, 2).reshape(batch * sequence, value.shape[1], head_dim)
    if not layout.query_lengths:
        # A batch with no valid tokens has no training targets, but keep autograd connected.
        return (query * 0 + (key.sum() + value.sum()) * 0).transpose(1, 2)
    packed = _flash_varlen(
        q.index_select(0, layout.query_indices),
        k.index_select(0, layout.key_indices),
        v.index_select(0, layout.key_indices),
        layout,
        scaling,
    )
    # Padded queries are zeroed (as in standard unpadding), not trained/scored.
    # They remain masked as keys at every layer, so cannot change valid outputs.
    output = torch.zeros_like(q).index_copy(0, layout.query_indices, packed)
    return output.view(batch, sequence, heads, head_dim)
