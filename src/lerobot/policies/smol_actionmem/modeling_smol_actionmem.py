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
    ACTION_TOKEN_DISTANCES,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import require_package

from ..action_code import (
    ActionCodeLayout,
    compute_action_code_objective,
    condition_flow_hidden,
    validate_action_code_sequence,
)
from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)
from .configuration_smol_actionmem import SmolActionMemConfig
from .smolvlm_with_expert import SmolActionMemVLMWithExpertModel


class SmolActionMemFlowMatching(VLAFlowMatching):
    """Gaussian flow expert conditioned on continuous VLM action logits."""

    def __init__(self, config: SmolActionMemConfig, rtc_processor=None):
        # VLAFlowMatching is deliberately not initialized through super(): its
        # constructor hardcodes SmolVLMWithExpertModel. The modules below mirror
        # SmolVLA while selecting the checkpoint-capable ActionMem wrapper.
        nn.Module.__init__(self)
        self.config = config
        self.rtc_processor = rtc_processor

        self.action_code_layout = ActionCodeLayout(
            codebook_size=config.action_codebook_size,
            invalid_value=config.action_code_invalid_value,
        )
        self.action_codebook_size = self.action_code_layout.codebook_size
        self.action_query_id = self.action_code_layout.action_query_id
        self.action_padding_id = self.action_code_layout.padding_id

        self.vlm_with_expert = SmolActionMemVLMWithExpertModel(
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
            self.action_code_layout.context_size,
            hidden_size,
            padding_idx=self.action_padding_id,
        )
        self.action_classifier = nn.Linear(hidden_size, self.action_codebook_size)
        self.action_condition_proj = nn.Sequential(
            nn.LayerNorm(self.action_codebook_size),
            nn.Linear(self.action_codebook_size, config.action_condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(
                config.action_condition_hidden_dim,
                self.vlm_with_expert.expert_hidden_size * 2,
            ),
        )
        nn.init.normal_(self.action_code_embedding.weight, mean=0.0, std=config.action_code_init_std)
        nn.init.normal_(self.action_classifier.weight, mean=0.0, std=config.action_code_init_std)
        nn.init.zeros_(self.action_classifier.bias)
        # Residual FiLM starts as an exact identity. This also makes checkpoints
        # created before the continuous-condition change safe to resume with
        # strict=False (the LeRobot default).
        nn.init.zeros_(self.action_condition_proj[-1].weight)
        nn.init.zeros_(self.action_condition_proj[-1].bias)
        with torch.no_grad():
            self.action_code_embedding.weight[self.action_padding_id].zero_()

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
                "action_condition_proj.",
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

    def _validate_action_token_sequence(self, action_tokens, action_token_masks) -> None:
        validate_action_code_sequence(
            action_tokens,
            action_token_masks,
            self.action_code_layout,
            policy_name="Smol ActionMem",
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
        """Embed image/task/action-memory/state/query in that order.

        The current action code is supervision only and is removed by the
        caller. ACTION_QUERY can observe action memory and the current robot
        state and is always the final prefix token.
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
        query_offsets = ((action_tokens == self.action_query_id) & action_token_masks.bool()).to(torch.int64)
        if not torch.all(query_offsets.sum(dim=1) == 1):
            raise ValueError("Each Smol ActionMem prefix must contain exactly one valid ACTION_QUERY.")
        query_indices = query_offsets.argmax(dim=1)
        if not torch.all(query_indices == query_indices[0]):
            raise ValueError("ACTION_QUERY must have the same sequence position across a batch.")
        query_index = int(query_indices[0].item())
        if query_index != action_token_emb.shape[1] - 1:
            raise ValueError("ACTION_QUERY must be the final Smol ActionMem prefix token.")

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

        embs.append(action_token_emb[:, query_index:])
        pad_masks.append(action_token_masks[:, query_index:].bool())
        att_masks += [1] * (action_token_emb.shape[1] - query_index)

        embeddings = torch.cat(embs, dim=1)
        padding_masks = torch.cat(pad_masks, dim=1)
        attention_blocks = torch.tensor(att_masks, dtype=torch.bool, device=embeddings.device)[None, :]
        attention_blocks = attention_blocks.expand(embeddings.shape[0], -1)

        if self.prefix_length > 0 and embeddings.shape[1] < self.prefix_length:
            # Keep ACTION_QUERY physically last even with a fixed prefix
            # length. Insert masked padding immediately before it rather than
            # appending padding after it.
            padding_length = self.prefix_length - embeddings.shape[1]
            embedding_padding = torch.zeros(
                embeddings.shape[0],
                padding_length,
                embeddings.shape[2],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            mask_padding = torch.zeros(
                padding_masks.shape[0],
                padding_length,
                dtype=padding_masks.dtype,
                device=padding_masks.device,
            )
            attention_padding = torch.zeros(
                attention_blocks.shape[0],
                padding_length,
                dtype=attention_blocks.dtype,
                device=attention_blocks.device,
            )
            embeddings = torch.cat([embeddings[:, :-1], embedding_padding, embeddings[:, -1:]], dim=1)
            padding_masks = torch.cat([padding_masks[:, :-1], mask_padding, padding_masks[:, -1:]], dim=1)
            attention_blocks = torch.cat(
                [attention_blocks[:, :-1], attention_padding, attention_blocks[:, -1:]], dim=1
            )

        return embeddings, padding_masks, attention_blocks

    def _compute_action_logits(self, prefix_out: Tensor) -> Tensor:
        query_hidden = prefix_out[:, -1, :].to(dtype=self.action_classifier.weight.dtype)
        return self.action_classifier(query_hidden)

    def _condition_flow_hidden(
        self,
        suffix_out: Tensor,
        action_logits: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return condition_flow_hidden(
            suffix_out,
            action_logits,
            self.action_condition_proj,
            scale=self.config.action_condition_scale,
        )

    def _compute_action_token_objective(
        self,
        logits: Tensor,
        action_tokens: Tensor,
        action_token_masks: Tensor,
        action_code_distances: Tensor | None,
    ) -> dict[str, Tensor]:
        return compute_action_code_objective(
            logits,
            action_tokens,
            action_token_masks,
            action_code_distances,
            temperature=self.config.action_token_soft_target_temperature,
            policy_name="Smol ActionMem",
        )

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
        action_token_distances: Tensor | None = None,
        compute_flow: bool = True,
        compute_action_token: bool = True,
    ) -> dict[str, Tensor]:
        if not compute_flow and not compute_action_token:
            raise ValueError("At least one Smol ActionMem objective must be enabled.")
        self._validate_action_token_sequence(action_tokens, action_token_masks)
        if state is None:
            raise ValueError("Smol ActionMem requires state for action-token and flow conditioning.")

        if not compute_flow:
            # The target slot is supervision only. It is never embedded into the
            # condition path, even though causal masking would hide it from the
            # ACTION_QUERY position.
            action_prompt_tokens = action_tokens[:, :-1]
            action_prompt_masks = action_token_masks[:, :-1]
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                action_prompt_tokens,
                action_prompt_masks,
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
            logits = self._compute_action_logits(prefix_out)
            return self._compute_action_token_objective(
                logits,
                action_tokens,
                action_token_masks,
                action_token_distances,
            )

        if state is None or actions is None or time is None:
            raise ValueError("Flow training requires state, actions, and time tensors.")
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # The ground-truth action code remains an auxiliary label only. Flow sees the prompt
        # through ACTION_QUERY, then receives the VLM's complete predicted logit
        # vector through a stop-gradient continuous residual FiLM path.
        action_prompt_tokens = action_tokens[:, :-1]
        action_prompt_masks = action_token_masks[:, :-1]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_prompt_tokens,
            action_prompt_masks,
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
        logits = self._compute_action_logits(prefix_out)
        suffix_out, condition_metrics = self._condition_flow_hidden(suffix_out, logits)
        flow_losses = F.mse_loss(u_t, self.action_out_proj(suffix_out), reduction="none")

        output = {"flow_losses": flow_losses, **condition_metrics}
        if compute_action_token:
            output.update(
                self._compute_action_token_objective(
                    logits,
                    action_tokens,
                    action_token_masks,
                    action_token_distances,
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
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
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
        (prefix_out, _), past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        logits = self._compute_action_logits(prefix_out)

        if noise is None:
            noise = self.sample_noise(
                (state.shape[0], self.config.chunk_size, self.config.max_action_dim),
                state.device,
            )
        return euler_integrate(
            lambda input_x_t, timestep: self._denoise_step_with_action_condition(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=timestep,
                action_logits=logits,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def _denoise_step_with_action_condition(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
        action_logits: Tensor,
    ) -> Tensor:
        """Apply one Euler vector-field evaluation with the predicted logit FiLM."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = super().embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        suffix_out, _ = self._condition_flow_hidden(suffix_out, action_logits)
        return self.action_out_proj(suffix_out)


class SmolActionMemPolicy(SmolVLAPolicy):
    """LeRobot policy wrapper exposing the same contract as ActionMem."""

    config_class = SmolActionMemConfig
    name = "smol_actionmem"

    def __init__(self, config: SmolActionMemConfig, **kwargs):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = SmolActionMemFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def configure_training_stage(self) -> int:
        num_trainable = self.model.configure_training_stage()
        if num_trainable == 0:
            raise RuntimeError(
                f"Smol ActionMem training_stage={self.config.training_stage!r} left no trainable "
                "parameters. Check PEFT target_modules for the selected branch."
            )
        logging.info(
            "Configured Smol ActionMem training_stage=%s with %d trainable parameters.",
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
                f"Smol ActionMem requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the processed batch."
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
                f"Smol ActionMem requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the processed batch."
            )

        stage = self.config.training_stage
        compute_flow = stage in {"action_expert_only", "joint"}
        compute_action_token = stage in {"vlm_only", "joint"} and self.config.action_token_loss_weight > 0
        state = self.prepare_state(batch)
        # Keep the original SmolVLA flow coordinate system: ACTION has already
        # been normalized by the policy preprocessor (MEAN_STD by default).
        # The per-source BOUNDS_Q99 tensor remains separate and is consumed only
        # by the data collator to produce effect-code labels and soft targets.
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
            action_token_distances=batch.get(ACTION_TOKEN_DISTANCES),
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
                    "action_condition_gamma_rms": output["action_condition_gamma_rms"].item(),
                    "action_condition_beta_rms": output["action_condition_beta_rms"].item(),
                    "action_condition_logit_std": output["action_condition_logit_std"].item(),
                    "action_condition_predicted_entropy": output["action_condition_predicted_entropy"].item(),
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
                    "action_token_target_rank": output["action_token_target_rank"].item(),
                    "action_token_soft_target_entropy": output["action_token_soft_target_entropy"].item(),
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
            "modules_to_save": [
                "action_code_embedding",
                "action_classifier",
                "action_condition_proj",
            ],
        }
