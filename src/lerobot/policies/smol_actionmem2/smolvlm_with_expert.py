#!/usr/bin/env python

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

"""SmolVLM/action-expert wrapper with layer-wise gradient checkpointing."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ..smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


class SmolActionMem2VLMWithExpertModel(SmolVLMWithExpertModel):
    """Keep SmolVLA attention semantics while enabling memory-efficient training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True
        vision_model = self.get_vlm_model().vision_model
        enable_vision_checkpointing = getattr(vision_model, "gradient_checkpointing_enable", None)
        if callable(enable_vision_checkpointing):
            enable_vision_checkpointing(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "preserve_rng_state": False,
                }
            )

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        vision_model = self.get_vlm_model().vision_model
        disable_vision_checkpointing = getattr(vision_model, "gradient_checkpointing_disable", None)
        if callable(disable_vision_checkpointing):
            disable_vision_checkpointing()

    def _forward_training_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        fill_kv_cache,
    ):
        """Run one cache-free SmolVLA layer for checkpointed training."""
        if (
            fill_kv_cache
            or "cross" not in self.attention_mode
            or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
        ):
            att_outputs, _ = self.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=False,
                fill_kv_cache=fill_kv_cache,
                past_key_values=None,
            )
        else:
            att_outputs, _ = self.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=False,
                fill_kv_cache=fill_kv_cache,
                past_key_values=None,
            )

        outputs_embeds = []
        start = 0
        for stream_index, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[stream_index][layer_idx]
            att_output = att_outputs[stream_index] if stream_index < len(att_outputs) else att_outputs[0]
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            if layer is None:
                outputs_embeds.append(hidden_states)
                continue

            end = start + hidden_states.shape[1]
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output[:, start:end])
            out_emb = out_emb + hidden_states
            residual = out_emb
            out_emb = layer.post_attention_layernorm(out_emb)
            out_emb = layer.mlp(out_emb)
            outputs_embeds.append(out_emb + residual)
            start = end if len(att_outputs) == 1 else 0

        return outputs_embeds

    def _checkpointed_forward(
        self,
        attention_mask,
        position_ids,
        inputs_embeds,
        fill_kv_cache,
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        batch_size = next(hidden.shape[0] for hidden in inputs_embeds if hidden is not None)
        head_dim = self.vlm.config.text_config.head_dim
        prefix_embs, suffix_embs = inputs_embeds

        for layer_idx in range(self.num_vlm_layers):
            if suffix_embs is None:

                def prefix_layer(prefix, current_layer=layer_idx):
                    return self._forward_training_layer(
                        model_layers,
                        [prefix, None],
                        current_layer,
                        position_ids,
                        attention_mask,
                        batch_size,
                        head_dim,
                        fill_kv_cache,
                    )[0]

                prefix_embs = checkpoint(
                    prefix_layer,
                    prefix_embs,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:

                def joint_layer(prefix, suffix, current_layer=layer_idx):
                    outputs = self._forward_training_layer(
                        model_layers,
                        [prefix, suffix],
                        current_layer,
                        position_ids,
                        attention_mask,
                        batch_size,
                        head_dim,
                        fill_kv_cache,
                    )
                    return outputs[0], outputs[1]

                prefix_embs, suffix_embs = checkpoint(
                    joint_layer,
                    prefix_embs,
                    suffix_embs,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

        outputs = [
            models[0].norm(prefix_embs) if prefix_embs is not None else None,
            models[1].norm(suffix_embs) if suffix_embs is not None else None,
        ]
        return outputs, None

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if self.gradient_checkpointing and self.training and not use_cache:
            return self._checkpointed_forward(
                attention_mask,
                position_ids,
                inputs_embeds,
                fill_kv_cache,
            )
        return super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
