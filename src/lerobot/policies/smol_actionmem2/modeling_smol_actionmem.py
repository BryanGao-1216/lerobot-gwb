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

"""A lightweight SmolVLA-backed implementation of ActionMem."""

from __future__ import annotations

import logging
import math
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKEN_Q0_DISTANCES,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
    pad_tensor,
)
from .configuration_smol_actionmem import SmolActionMem2Config
from .smolvlm_with_expert import SmolActionMem2VLMWithExpertModel
from .tokenization_smol_actionmem import (
    SmolActionMem2TokenMap,
    default_smol_actionmem2_token_map_path,
)


class SmolActionMem2FlowMatching(VLAFlowMatching):
    """Gaussian flow expert conditioned on an independent discrete action code."""

    def __init__(self, config: SmolActionMem2Config, rtc_processor=None):
        # VLAFlowMatching is deliberately not initialized through super(): its
        # constructor hardcodes SmolVLMWithExpertModel. The modules below mirror
        # SmolVLA while selecting the checkpoint-capable ActionMem wrapper.
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor

        token_map_path = config.action_token_map_path or str(default_smol_actionmem2_token_map_path())
        self.action_code_map = SmolActionMem2TokenMap.from_json(token_map_path)
        config.action_token_map_path = self.action_code_map.path
        if config.chunk_size != self.action_code_map.action_horizon:
            raise ValueError(
                f"Smol ActionMem 2 chunk_size ({config.chunk_size}) must match the VQ-VAE action horizon "
                f"({self.action_code_map.action_horizon})."
            )
        if config.max_action_dim < self.action_code_map.action_dim:
            raise ValueError(
                f"max_action_dim ({config.max_action_dim}) must be at least the VQ-VAE action dimension "
                f"({self.action_code_map.action_dim})."
            )

        self.vlm_with_expert = SmolActionMem2VLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            # Stage-specific freezing is applied below and again after PEFT.
            train_expert_only=False,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device=config.device if config.device is not None else "auto",
        )

        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.action_code_embedding = nn.Embedding(
            self.action_code_map.context_vocab_size,
            hidden_size,
            padding_idx=self.action_code_map.padding_id,
        )
        self.action_classifier = nn.Linear(hidden_size, self.action_code_map.codebook_size)
        nn.init.normal_(self.action_code_embedding.weight, mean=0.0, std=config.action_code_init_std)
        nn.init.normal_(self.action_classifier.weight, mean=0.0, std=config.action_code_init_std)
        nn.init.zeros_(self.action_classifier.bias)
        with torch.no_grad():
            self.action_code_embedding.weight[self.action_code_map.padding_id].zero_()

        self.state_proj = nn.Linear(
            config.max_state_dim,
            self.vlm_with_expert.config.text_config.hidden_size,
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2,
            self.vlm_with_expert.expert_hidden_size,
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            self.vlm_with_expert.expert_hidden_size,
        )

        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token],
            dtype=torch.long,
        )
        self.add_image_special_tokens = config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = config.prefix_length

        self._training_stage_configured = False
        self.configure_training_stage()
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    @staticmethod
    def _is_vlm_parameter(name: str) -> bool:
        # The pretrained language vocabulary is deliberately excluded. Action
        # classification and action memory use their own trainable modules.
        is_vlm_backbone = name.startswith("vlm_with_expert.vlm.") and not name.startswith(
            (
                "vlm_with_expert.vlm.lm_head.",
                "vlm_with_expert.vlm.model.text_model.embed_tokens.",
            )
        )
        return is_vlm_backbone or name.startswith(
            (
                "state_proj.",
                "action_code_embedding.",
                "action_classifier.",
            )
        )

    @staticmethod
    def _is_action_expert_parameter(name: str) -> bool:
        return name.startswith("vlm_with_expert.lm_expert.") or name.startswith(
            (
                "state_proj.",
                "action_code_embedding.",
                "action_in_proj.",
                "action_out_proj.",
                "action_time_mlp_in.",
                "action_time_mlp_out.",
            )
        )

    def configure_training_stage(self) -> int:
        """Apply full-training defaults once, then safely filter injected adapters."""
        stage = self.config.training_stage
        first_configuration = not self._training_stage_configured
        for name, parameter in self.named_parameters():
            is_vlm = self._is_vlm_parameter(name)
            is_expert = self._is_action_expert_parameter(name)
            is_frozen_vision = self.config.freeze_vision_encoder and name.startswith(
                "vlm_with_expert.vlm.model.vision_model."
            )
            included = (stage in {"vlm_only", "joint"} and is_vlm) or (
                stage in {"action_expert_only", "joint"} and is_expert
            )
            if first_configuration:
                parameter.requires_grad_(included and not is_frozen_vision)
            elif not included or is_frozen_vision:
                parameter.requires_grad_(False)

        self._training_stage_configured = True
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def gradient_checkpointing_enable(self) -> None:
        self.vlm_with_expert.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.vlm_with_expert.gradient_checkpointing_disable()

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(
            bsize,
            device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )

    def _make_training_flow_source(self, actions: Tensor) -> Tensor:
        """Sample the standard Gaussian source used by conditional flow matching."""
        return self.sample_noise(actions.shape, actions.device)

    def _validate_action_token_sequence(self, action_tokens, action_token_masks) -> None:
        if action_tokens is None or action_token_masks is None:
            raise ValueError("Smol ActionMem 2 requires action_tokens and action_token_masks.")
        if action_tokens.shape != action_token_masks.shape or action_tokens.ndim != 2:
            raise ValueError(
                "action_tokens and action_token_masks must have shape [B, T], got "
                f"{tuple(action_tokens.shape)} and {tuple(action_token_masks.shape)}."
            )
        if action_tokens.shape[1] < 2:
            raise ValueError("Smol ActionMem 2 requires ACTION_QUERY plus a target slot.")
        invalid_query = (~action_token_masks[:, -2].bool()) | (
            action_tokens[:, -2] != self.action_code_map.action_query_id
        )
        if torch.any(invalid_query):
            raise ValueError(
                "The penultimate action token must be the valid ACTION_QUERY token "
                f"{self.action_code_map.action_query_id}."
            )
        targets = action_tokens[:, -1]
        target_masks = action_token_masks[:, -1].bool()
        invalid_targets = target_masks & (
            (targets < self.action_code_map.action_class_min)
            | (targets > self.action_code_map.action_class_max)
        )
        if torch.any(invalid_targets):
            raise ValueError(
                f"Action-class targets must be in [{self.action_code_map.action_class_min}, "
                f"{self.action_code_map.action_class_max}]."
            )

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens,
        action_token_masks,
        state=None,
    ):
        """Embed image/task/action-memory/state/query/target in that order.

        ACTION_QUERY is causal and therefore cannot observe the current target
        appended after it, but it can observe action memory and the current
        robot state placed immediately before it.
        """
        embs = []
        pad_masks = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                image_start = (
                    self.vlm_with_expert.embed_language_tokens(self.global_image_start_token.to(img.device))
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(image_start[:, :, 0], dtype=torch.bool)
                embs.append(image_start)
                pad_masks.append(image_start_mask)
                att_masks += [0] * image_start.shape[1]

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb * math.sqrt(img_emb.shape[-1])
            batch_size, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end = (
                    self.vlm_with_expert.embed_language_tokens(self.image_end_token.to(img.device))
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(image_end[:, :, 0], dtype=torch.bool)
                embs.append(image_end)
                pad_masks.append(image_end_mask)
                att_masks += [0] * image_end.shape[1]

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks.bool())
        att_masks += [0] * lang_emb.shape[1]

        action_token_emb = self.action_code_embedding(action_tokens)
        action_token_emb = action_token_emb * math.sqrt(action_token_emb.shape[-1])
        action_token_emb = action_token_emb.to(dtype=lang_emb.dtype)
        query_offsets = (
            (action_tokens == self.action_code_map.action_query_id) & action_token_masks.bool()
        ).to(torch.int64)
        if not torch.all(query_offsets.sum(dim=1) == 1):
            raise ValueError("Each Smol ActionMem 2 prefix must contain exactly one valid ACTION_QUERY.")
        query_indices = query_offsets.argmax(dim=1)
        if not torch.all(query_indices == query_indices[0]):
            raise ValueError("ACTION_QUERY must have the same sequence position across a batch.")
        query_index = int(query_indices[0].item())

        # Keep memory/control tokens after the task, then insert the current
        # state immediately before ACTION_QUERY.
        if query_index > 0:
            embs.append(action_token_emb[:, :query_index])
            pad_masks.append(action_token_masks[:, :query_index].bool())
            att_masks += [1] * query_index

        if state is not None:
            state = state.to(dtype=self.state_proj.weight.dtype)
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            state_emb = state_emb.to(dtype=lang_emb.dtype)
            embs.append(state_emb)
            pad_masks.append(
                torch.ones(
                    state_emb.shape[:2],
                    dtype=torch.bool,
                    device=state_emb.device,
                )
            )
            att_masks += [1] * state_emb.shape[1]

        action_base_position = sum(embedding.shape[1] for embedding in embs)
        query_positions = torch.full_like(query_indices, action_base_position)
        embs.append(action_token_emb[:, query_index:])
        pad_masks.append(action_token_masks[:, query_index:].bool())
        att_masks += [1] * (action_token_emb.shape[1] - query_index)

        embeddings = torch.cat(embs, dim=1)
        padding_masks = torch.cat(pad_masks, dim=1)
        attention_blocks = torch.tensor(att_masks, dtype=torch.bool, device=embeddings.device)[None, :]
        attention_blocks = attention_blocks.expand(embeddings.shape[0], -1)

        if self.prefix_length > 0 and embeddings.shape[1] < self.prefix_length:
            embeddings = pad_tensor(embeddings, self.prefix_length, pad_value=0)
            padding_masks = pad_tensor(padding_masks, self.prefix_length, pad_value=0)
            attention_blocks = pad_tensor(attention_blocks, self.prefix_length, pad_value=0)

        return embeddings, padding_masks, attention_blocks, query_positions

    def _compute_action_token_objective(
        self,
        prefix_out: Tensor,
        query_positions: Tensor,
        action_tokens: Tensor,
        action_token_masks: Tensor,
        q0_distances: Tensor | None,
    ) -> dict[str, Tensor]:
        batch_indices = torch.arange(prefix_out.shape[0], device=prefix_out.device)
        query_hidden = prefix_out[batch_indices, query_positions]
        query_hidden = query_hidden.to(dtype=self.action_classifier.weight.dtype)
        logits = self.action_classifier(query_hidden)
        target_mask = action_token_masks[:, -1].bool()
        safe_targets = torch.where(target_mask, action_tokens[:, -1], 0)
        if torch.any(target_mask):
            if q0_distances is None:
                raise ValueError(
                    "Smol ActionMem 2 soft-token training requires "
                    f"'{ACTION_TOKEN_Q0_DISTANCES}' in the processed batch. Use the ActionMem RLDS "
                    "collator or provide squared distances from the frozen VQ encoder latent to "
                    "all q0 codebook centers."
                )
            if q0_distances.shape != logits.shape:
                raise ValueError(
                    f"Expected q0 distances with shape {tuple(logits.shape)}, got "
                    f"{tuple(q0_distances.shape)}."
                )
            distances = q0_distances.to(device=logits.device, dtype=torch.float32)
            if not torch.isfinite(distances).all():
                raise ValueError("q0 latent distances must contain only finite values.")
            soft_targets = torch.softmax(
                -distances / self.config.action_token_soft_target_temperature,
                dim=-1,
            )
            log_predictions = F.log_softmax(logits.float(), dim=-1)
            per_sample = F.kl_div(log_predictions, soft_targets, reduction="none").sum(dim=-1)
            target_entropy_per_sample = -(
                soft_targets * soft_targets.clamp_min(torch.finfo(soft_targets.dtype).tiny).log()
            ).sum(dim=-1)
            target_peak_per_sample = soft_targets.max(dim=-1).values
        else:
            per_sample = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
            target_entropy_per_sample = torch.zeros_like(per_sample)
            target_peak_per_sample = torch.zeros_like(per_sample)

        mask = target_mask.to(dtype=per_sample.dtype)
        valid_count = mask.sum().clamp_min(1)
        per_sample = per_sample * mask
        mean_loss = per_sample.sum() / valid_count
        accuracy = ((logits.argmax(dim=-1) == safe_targets) & target_mask).float().sum() / valid_count
        target_entropy = (target_entropy_per_sample * mask).sum() / valid_count
        target_peak_probability = (target_peak_per_sample * mask).sum() / valid_count
        return {
            "action_token_loss_per_sample": per_sample,
            "action_token_target_mask": target_mask,
            "action_token_kl_loss": mean_loss,
            "action_token_accuracy": accuracy,
            "action_token_soft_target_entropy": target_entropy,
            "action_token_soft_target_peak_probability": target_peak_probability,
        }

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens,
        action_token_masks,
        state=None,
        actions=None,
        noise=None,
        time=None,
        *,
        action_token_q0_distances: Tensor | None = None,
        compute_flow: bool = True,
        compute_action_token: bool = True,
    ) -> dict[str, Tensor]:
        if not compute_flow and not compute_action_token:
            raise ValueError("At least one Smol ActionMem 2 objective must be enabled.")
        self._validate_action_token_sequence(action_tokens, action_token_masks)
        if state is None:
            raise ValueError("Smol ActionMem 2 requires state for action-token and flow conditioning.")

        if not compute_flow:
            prefix_embs, prefix_pad_masks, prefix_att_masks, query_positions = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                action_tokens,
                action_token_masks,
                state=state,
            )
            prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            (prefix_out, _), _ = self.vlm_with_expert.forward(
                attention_mask=prefix_attention,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                # Token-only execution must use all VLM self-attention layers.
                fill_kv_cache=True,
            )
            return self._compute_action_token_objective(
                prefix_out,
                query_positions,
                action_tokens,
                action_token_masks,
                action_token_q0_distances,
            )

        if state is None or actions is None or time is None:
            raise ValueError("Flow training requires state, actions, and time tensors.")
        if noise is None:
            noise = self._make_training_flow_source(actions)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # The final valid local class is the ground-truth q0 code condition.
        # It is embedded in the prefix, so every flow suffix token can attend
        # to it while predicting the Gaussian-to-action vector field.
        prefix_embs, prefix_pad_masks, prefix_att_masks, query_positions = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_tokens,
            action_token_masks,
            state=state,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = super().embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        flow_losses = F.mse_loss(u_t, self.action_out_proj(suffix_out), reduction="none")

        output = {"flow_losses": flow_losses}
        if compute_action_token:
            output.update(
                self._compute_action_token_objective(
                    prefix_out,
                    query_positions,
                    action_tokens,
                    action_token_masks,
                    action_token_q0_distances,
                )
            )
        return output

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens,
        action_token_masks,
        state,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        self._validate_action_token_sequence(action_tokens, action_token_masks)
        num_steps = num_steps or self.config.num_inference_steps

        # First pass: generate from ACTION_QUERY conditioned on the current state.
        action_prompt_tokens = action_tokens[:, :-1]
        action_prompt_masks = action_token_masks[:, :-1]
        prefix_embs, prefix_pad_masks, prefix_att_masks, query_positions = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_prompt_tokens,
            action_prompt_masks,
            state=state,
        )
        prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        batch_indices = torch.arange(prefix_out.shape[0], device=prefix_out.device)
        query_hidden = prefix_out[batch_indices, query_positions]
        logits = self.action_classifier(query_hidden.to(dtype=self.action_classifier.weight.dtype))
        generated_action_token = logits.argmax(dim=-1, keepdim=True)

        # Rebuild and cache the complete prefix with the generated q0 class.
        # This cache is the code condition read by every Euler denoising step.
        generated_sequence = torch.cat([action_prompt_tokens, generated_action_token], dim=1)
        generated_masks = torch.cat(
            [
                action_prompt_masks,
                torch.ones_like(generated_action_token, dtype=torch.bool),
            ],
            dim=1,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            generated_sequence,
            generated_masks,
            state=state,
        )
        prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        if noise is None:
            noise = self.sample_noise(
                (state.shape[0], self.config.chunk_size, self.config.max_action_dim),
                state.device,
            )
        return euler_integrate(
            lambda input_x_t, timestep: super(SmolActionMem2FlowMatching, self).denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )


class SmolActionMem2Policy(SmolVLAPolicy):
    """LeRobot policy wrapper exposing the same contract as ActionMem."""

    config_class = SmolActionMem2Config
    name = "smol_actionmem2"

    def __init__(self, config: SmolActionMem2Config, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = SmolActionMem2FlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def configure_training_stage(self) -> int:
        num_trainable = self.model.configure_training_stage()
        if num_trainable == 0:
            raise RuntimeError(
                f"Smol ActionMem 2 training_stage={self.config.training_stage!r} left no trainable "
                "parameters. Check PEFT target_modules for the selected branch."
            )
        logging.info(
            "Configured Smol ActionMem 2 training_stage=%s with %d trainable parameters.",
            self.config.training_stage,
            num_trainable,
        )
        return num_trainable

    def get_optim_params(self):
        self.configure_training_stage()
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        action_tokens = batch.get(ACTION_TOKENS)
        action_token_masks = batch.get(ACTION_TOKEN_MASK)
        if action_tokens is None or action_token_masks is None:
            raise ValueError(
                f"Smol ActionMem 2 requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the processed batch."
            )
        actions = self.model.sample_actions(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            action_tokens,
            action_token_masks,
            state,
            noise=noise,
            **kwargs,
        )
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'none'.")
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        action_tokens = batch.get(ACTION_TOKENS)
        action_token_masks = batch.get(ACTION_TOKEN_MASK)
        if action_tokens is None or action_token_masks is None:
            raise ValueError(
                f"Smol ActionMem 2 requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the processed batch."
            )

        stage = self.config.training_stage
        compute_flow = stage in {"action_expert_only", "joint"}
        compute_action_token = stage in {"vlm_only", "joint"}
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch) if compute_flow else None
        if compute_flow and time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        output = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            action_tokens,
            action_token_masks,
            state,
            actions,
            noise,
            time,
            action_token_q0_distances=batch.get(ACTION_TOKEN_Q0_DISTANCES),
            compute_flow=compute_flow,
            compute_action_token=compute_action_token,
        )

        scalar_terms = []
        per_sample_terms = []
        metrics: dict[str, float | list[float]] = {}
        if compute_flow:
            action_dim = self.config.action_feature.shape[0]
            flow_losses = output["flow_losses"][:, :, :action_dim]
            actions_is_pad = batch.get("action_is_pad")
            if actions_is_pad is None:
                flow_loss = flow_losses.mean()
                per_sample_flow = flow_losses.mean(dim=(1, 2))
            else:
                valid = (~actions_is_pad).unsqueeze(-1).to(flow_losses.dtype)
                masked_flow = flow_losses * valid
                flow_loss = masked_flow.sum() / (valid.sum() * flow_losses.shape[-1]).clamp_min(1)
                per_sample_flow = masked_flow.sum(dim=(1, 2)) / (
                    valid.sum(dim=(1, 2)) * flow_losses.shape[-1]
                ).clamp_min(1)
            scalar_terms.append(self.config.flow_loss_weight * flow_loss)
            per_sample_terms.append(self.config.flow_loss_weight * per_sample_flow)
            metrics.update(
                {
                    "loss_per_dim": flow_losses.mean(dim=(0, 1)).detach().cpu().tolist(),
                    "flow_loss": flow_loss.item(),
                    "weighted_flow_loss": (self.config.flow_loss_weight * flow_loss).item(),
                }
            )

        if compute_action_token:
            token_loss = output["action_token_kl_loss"]
            target_mask = output["action_token_target_mask"]
            valid_count = target_mask.sum().clamp(min=1)
            token_scale = target_mask.numel() / valid_count
            per_sample_token = output["action_token_loss_per_sample"] * token_scale
            scalar_terms.append(self.config.action_token_loss_weight * token_loss)
            per_sample_terms.append(self.config.action_token_loss_weight * per_sample_token)
            metrics.update(
                {
                    "action_token_kl_loss": token_loss.item(),
                    "weighted_action_token_kl_loss": (
                        self.config.action_token_loss_weight * token_loss
                    ).item(),
                    "action_token_accuracy": output["action_token_accuracy"].item(),
                    "action_token_soft_target_entropy": output[
                        "action_token_soft_target_entropy"
                    ].item(),
                    "action_token_soft_target_peak_probability": output[
                        "action_token_soft_target_peak_probability"
                    ].item(),
                }
            )

        if reduction == "none":
            loss = torch.stack(per_sample_terms).sum(dim=0)
            metrics["loss"] = loss.mean().item()
            return loss, metrics
        loss = torch.stack(scalar_terms).sum()
        metrics["loss"] = loss.item()
        return loss, metrics

    def _get_default_peft_targets(self) -> dict[str, object]:
        """Create both branches so staged LoRA training can reuse one adapter."""
        vlm_targets = (
            r".*\.vlm_with_expert\.vlm\."
            r"model\.text_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj"
        )
        expert_targets = r".*\.vlm_with_expert\.lm_expert\.layers\.[0-9]+\.self_attn\.(q|v)_proj"
        projections = (
            r"model\.(state_proj|action_in_proj|action_out_proj|"
            r"action_time_mlp_in|action_time_mlp_out)"
        )
        return {
            "target_modules": rf"({vlm_targets}|{expert_targets}|{projections})",
            # These modules are randomly initialized for this policy and must
            # be trained and serialized in full even when the backbone uses LoRA.
            "modules_to_save": ["action_code_embedding", "action_classifier"],
        }
