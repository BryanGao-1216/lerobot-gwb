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

"""SmolW: SmolVLA conditioned on predicted VidTwin latent motion."""

from __future__ import annotations

import logging
from collections import deque
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
)
from ..utils import populate_queues
from .configuration_smolw import SmolWConfig
from .vidtwin_motion_encoder import VidTwinMotionExtractor


class SmolWFlowMatching(VLAFlowMatching):
    """Original SmolVLA action expert with a future-motion suffix condition."""

    def __init__(self, config: SmolWConfig, rtc_processor=None):
        # The parent compiles its methods inside __init__. Delay compilation
        # until all SmolW modules exist so torch.compile sees the final graph.
        compile_model = config.compile_model
        train_expert_only = config.train_expert_only
        config.compile_model = False
        # Mode-specific freezing is applied after all SmolW modules exist.
        config.train_expert_only = False
        try:
            super().__init__(config, rtc_processor=rtc_processor)
        finally:
            config.compile_model = compile_model
            config.train_expert_only = train_expert_only

        self.config = config
        vlm_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size

        self.mt_query_embedding = nn.Embedding(1, vlm_hidden_size)
        self.past_motion_projector = nn.Sequential(
            nn.LayerNorm(config.motion_latent_dim),
            nn.Linear(config.motion_latent_dim, config.motion_projector_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.motion_projector_hidden_dim, vlm_hidden_size),
        )
        self.future_motion_head = nn.Sequential(
            nn.LayerNorm(vlm_hidden_size),
            nn.Linear(vlm_hidden_size, config.motion_projector_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.motion_projector_hidden_dim, config.motion_latent_dim),
        )
        # Flow matching operates directly in VidTwin's fixed, normalized
        # per-temporal-slot space. Only the noisy z *input* is projected into
        # the expert width; the regression target itself has no learned map.
        self.z_token_in_proj = nn.Linear(config.motion_token_dim, expert_hidden_size)
        self.z_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.z_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)
        self.z_token_out_proj = nn.Linear(expert_hidden_size, config.motion_token_dim)
        self.register_buffer("z_condition_step", torch.zeros((), dtype=torch.long), persistent=True)

        nn.init.normal_(self.mt_query_embedding.weight, mean=0.0, std=0.02)
        self._train_mode_configured = False
        self.configure_train_mode()

        if compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    @staticmethod
    def _is_motion_parameter(name: str) -> bool:
        is_vlm = name.startswith("vlm_with_expert.vlm.") and not name.startswith(
            "vlm_with_expert.vlm.lm_head."
        )
        return is_vlm or name.startswith(
            (
                "state_proj.",
                "mt_query_embedding.",
                "past_motion_projector.",
                "future_motion_head.",
            )
        )

    @staticmethod
    def _is_action_expert_parameter(name: str) -> bool:
        return name.startswith("vlm_with_expert.lm_expert.") or name.startswith(
            (
                "state_proj.",
                "action_in_proj.",
                "action_out_proj.",
                "action_time_mlp_in.",
                "action_time_mlp_out.",
                "z_token_in_proj.",
                "z_time_mlp_in.",
                "z_time_mlp_out.",
                "z_token_out_proj.",
            )
        )

    def configure_train_mode(self) -> int:
        """Freeze everything outside the selected train mode before optimizer setup."""
        mode = self.config.train_mode
        first_configuration = not self._train_mode_configured
        for name, parameter in self.named_parameters():
            is_motion = self._is_motion_parameter(name)
            is_expert = self._is_action_expert_parameter(name)
            is_frozen_vision = self.config.freeze_vision_encoder and name.startswith(
                "vlm_with_expert.vlm.model.vision_model."
            )
            included = (mode in {"motion_only", "jointly"} and is_motion) or (
                mode in {"action_only", "jointly"} and is_expert
            )
            if name.startswith("state_proj.") and not self.config.train_state_proj:
                included = False
            if first_configuration:
                # Preserve tensors that the original SmolVLA implementation
                # freezes because they are unused by its truncated VLM/expert
                # layer pairing. Re-enabling them breaks DDP without
                # find_unused_parameters=True.
                parameter.requires_grad_(parameter.requires_grad and included and not is_frozen_vision)
            elif not included or is_frozen_vision:
                # Re-run after PEFT injection without re-enabling base tensors
                # that PEFT deliberately froze.
                parameter.requires_grad_(False)

        self._train_mode_configured = True
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vlm_with_expert.get_vlm_model().vision_model.eval()
        if self.config.train_mode == "action_only":
            # action_only consumes an oracle future-motion condition and keeps
            # the complete motion/VLM path frozen.
            self.vlm_with_expert.vlm.eval()
        return self

    def embed_prefix_with_motion(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        past_motion: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append M_t after the unchanged SmolVLA image/language/state prefix."""
        if past_motion.ndim != 2 or past_motion.shape[-1] != self.config.motion_latent_dim:
            raise ValueError(
                f"past_motion must have shape [B, motion_latent_dim], got {tuple(past_motion.shape)}."
            )
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )

        projector_dtype = self.past_motion_projector[1].weight.dtype
        projected_motion = self.past_motion_projector(past_motion.to(dtype=projector_dtype))
        query_ids = torch.zeros(past_motion.shape[0], 1, dtype=torch.long, device=past_motion.device)
        mt_query = self.mt_query_embedding(query_ids) + projected_motion[:, None, :]
        mt_query = mt_query.to(dtype=prefix_embs.dtype)

        query_mask = torch.ones(
            past_motion.shape[0],
            1,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )
        query_attention = torch.ones_like(query_mask)
        return (
            torch.cat([prefix_embs, mt_query], dim=1),
            torch.cat([prefix_pad_masks, query_mask], dim=1),
            torch.cat([prefix_att_masks, query_attention], dim=1),
        )

    def predict_future_motion(self, prefix_out: Tensor) -> Tensor:
        """Decode future VidTwin motion from the final M_t hidden state."""
        query_hidden = prefix_out[:, -1, :]
        head_dtype = self.future_motion_head[1].weight.dtype
        return self.future_motion_head(query_hidden.to(dtype=head_dtype)).float()

    def motion_to_z(self, future_motion: Tensor) -> Tensor:
        """Split and normalize flattened VidTwin motion into 16 fixed z targets."""
        if future_motion.ndim != 2 or future_motion.shape[-1] != self.config.motion_latent_dim:
            raise ValueError(
                f"future_motion must have shape [B, motion_latent_dim], got {tuple(future_motion.shape)}."
            )
        motion_for_z = future_motion.detach() if self.config.detach_motion_condition else future_motion
        motion_tokens = motion_for_z.reshape(
            motion_for_z.shape[0],
            self.config.vidtwin_num_frames,
            self.config.motion_token_dim,
        )
        # Layer normalization has no affine parameters here: target semantics
        # stay fixed throughout training and across checkpoints.
        z = F.layer_norm(motion_tokens.float(), (self.config.motion_token_dim,))
        return z * self.config.motion_condition_scale

    def embed_action_z_suffix(
        self,
        noisy_actions: Tensor,
        noisy_z: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed the suffix ``[z_1,...,z_16,a_1,...,a_H]``."""
        action_embs, action_pad_masks, action_att_masks = super().embed_suffix(
            noisy_actions,
            timestep,
        )
        expected_z_shape = (
            noisy_actions.shape[0],
            self.config.vidtwin_num_frames,
            self.config.motion_token_dim,
        )
        if noisy_z.shape != expected_z_shape:
            raise ValueError(f"noisy_z must have shape {expected_z_shape}, got {tuple(noisy_z.shape)}.")
        z_dtype = self.z_token_in_proj.weight.dtype
        z_emb = self.z_token_in_proj(noisy_z.to(dtype=z_dtype))
        z_time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=z_emb.device,
        ).to(dtype=z_emb.dtype)
        z_time_emb = z_time_emb[:, None, :].expand_as(z_emb)
        z_emb = self.z_time_mlp_out(F.silu(self.z_time_mlp_in(torch.cat([z_emb, z_time_emb], dim=-1))))
        z_emb = z_emb.to(dtype=action_embs.dtype)
        z_pad_mask = torch.ones(
            noisy_actions.shape[0],
            self.config.vidtwin_num_frames,
            dtype=torch.bool,
            device=action_pad_masks.device,
        )
        return (
            torch.cat([z_emb, action_embs], dim=1),
            z_pad_mask,
            action_pad_masks,
            action_att_masks,
        )

    def run_motion_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        past_motion: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_with_motion(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            past_motion,
        )
        prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        (prefix_out, _), past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        predicted_future_motion = self.predict_future_motion(prefix_out)
        return (
            predicted_future_motion,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            past_key_values,
        )

    @staticmethod
    def make_action_attention(
        prefix_pad_masks: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build suffix attention while hiding M_t from the action expert.

        M_t is always the final cached prefix token. The action expert receives
        no direct motion condition through its action rows; current image,
        language, and state prefix tokens retain the original SmolVLA path.
        """
        suffix_len = suffix_pad_masks.shape[1]
        action_prefix_masks = prefix_pad_masks.clone()
        action_prefix_masks[:, -1] = False
        prefix_attention = action_prefix_masks[:, None, :].expand(
            action_prefix_masks.shape[0], suffix_len, action_prefix_masks.shape[1]
        )
        suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention = torch.cat([prefix_attention, suffix_attention], dim=2)

        # The masked M_t occupies no action-expert position. This keeps the
        # original SmolVLA suffix positions unchanged.
        prefix_offsets = torch.sum(action_prefix_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        return attention, position_ids

    @classmethod
    def make_action_z_attention(
        cls,
        prefix_pad_masks: Tensor,
        z_pad_masks: Tensor,
        action_pad_masks: Tensor,
        action_att_masks: Tensor,
        action_can_see_z: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Build attention for ``[z_1,...,z_16,a_1,...,a_H]``.

        All z tokens attend bidirectionally within the z block and never read
        actions. Every action reads the complete z block. The action-to-action
        and action-to-prefix submatrices remain exactly the original SmolVLA
        masks, except that M_t stays hidden from actions so z is the explicit
        motion-conditioning path.
        """
        action_attention, action_position_ids = cls.make_action_attention(
            prefix_pad_masks,
            action_pad_masks,
            action_att_masks,
        )
        batch_size, horizon = action_pad_masks.shape
        z_count = z_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = z_count + horizon

        # z rows: complete valid prefix + all z tokens + no action tokens.
        z_prefix_attention = prefix_pad_masks[:, None, :].expand(batch_size, z_count, prefix_len)
        z_self_attention = z_pad_masks[:, None, :].expand(batch_size, z_count, z_count)
        z_action_attention = torch.zeros(
            batch_size,
            z_count,
            horizon,
            dtype=torch.bool,
            device=action_pad_masks.device,
        )
        z_rows = torch.cat([z_prefix_attention, z_self_attention, z_action_attention], dim=2)

        # action rows: original prefix + optional z condition + original
        # action mask. Warmup disables action->z; z flow remains in the graph
        # but its effective loss weight is zeroed by the policy wrapper.
        action_z_attention = (
            z_pad_masks[:, None, :].expand(batch_size, horizon, z_count)
            if action_can_see_z
            else torch.zeros(
                batch_size,
                horizon,
                z_count,
                dtype=torch.bool,
                device=action_pad_masks.device,
            )
        )
        action_rows = torch.cat(
            [
                action_attention[:, :, :prefix_len],
                action_z_attention,
                action_attention[:, :, prefix_len:],
            ],
            dim=2,
        )
        attention = torch.cat([z_rows, action_rows], dim=1)

        suffix_pad_masks = torch.cat([z_pad_masks, action_pad_masks], dim=1)
        attention &= suffix_pad_masks[:, :, None]
        attention &= torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)[:, None, :]

        # Preserve the original action positions. z gets its own ordered range
        # immediately after M_t, encoding VidTwin temporal order without
        # shifting the pretrained action RoPE coordinates.
        z_offsets = prefix_pad_masks.sum(dim=1, keepdim=True)
        z_position_ids = z_offsets + torch.arange(z_count, device=prefix_pad_masks.device)[None, :]
        position_ids = torch.cat([z_position_ids, action_position_ids], dim=1)
        if attention.shape != (batch_size, suffix_len, prefix_len + suffix_len):
            raise RuntimeError(f"Unexpected z/action attention shape {tuple(attention.shape)}.")
        return attention, position_ids

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        past_motion: Tensor,
        actions: Tensor | None = None,
        future_motion_target: Tensor | None = None,
        z_motion_source: Tensor | None = None,
        noise: Tensor | None = None,
        z_noise: Tensor | None = None,
        time: Tensor | None = None,
        compute_motion_loss: bool = True,
        compute_flow: bool = True,
    ) -> dict[str, Tensor | bool | int]:
        """Run the selected motion-regression and/or action objectives."""
        (
            predicted_future_motion,
            prefix_pad_masks,
            _,
            _,
            past_key_values,
        ) = self.run_motion_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            past_motion,
        )
        output = {
            "predicted_future_motion": predicted_future_motion,
        }

        if compute_motion_loss:
            if future_motion_target is None:
                raise ValueError("Motion training requires future_motion_target.")
            output["motion_losses"] = F.smooth_l1_loss(
                predicted_future_motion,
                future_motion_target.float(),
                reduction="none",
            ).mean(dim=-1)

        if compute_flow:
            if actions is None:
                raise ValueError("Action-expert training requires actions.")
            if z_motion_source is None:
                z_motion_source = predicted_future_motion
            elif z_motion_source.shape != predicted_future_motion.shape:
                raise ValueError(
                    "z_motion_source must match predicted future-motion shape; got "
                    f"{tuple(z_motion_source.shape)} and {tuple(predicted_future_motion.shape)}."
                )
            z_target = self.motion_to_z(z_motion_source)
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if z_noise is None:
                z_noise = self.sample_noise(z_target.shape, z_target.device)
            elif z_noise.shape != z_target.shape:
                raise ValueError(
                    f"z_noise must have shape {tuple(z_target.shape)}, got {tuple(z_noise.shape)}."
                )
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            action_x_t = time_expanded * noise + (1 - time_expanded) * actions
            action_u_t = noise - actions
            z_time = time[:, None, None]
            z_x_t = z_time * z_noise + (1 - z_time) * z_target
            z_u_t = z_noise - z_target
            suffix_embs, z_pad_masks, action_pad_masks, action_att_masks = self.embed_action_z_suffix(
                action_x_t,
                z_x_t,
                time,
            )
            z_condition_step = int(self.z_condition_step.item())
            action_can_see_z = not self.training or z_condition_step >= self.config.z_condition_warmup_steps
            attention, position_ids = self.make_action_z_attention(
                prefix_pad_masks,
                z_pad_masks,
                action_pad_masks,
                action_att_masks,
                action_can_see_z=action_can_see_z,
            )
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=attention,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
            )
            suffix_length = self.config.vidtwin_num_frames + self.config.chunk_size
            suffix_out = outputs_embeds[1][:, -suffix_length:].float()
            z_velocity = self.z_token_out_proj(suffix_out[:, : self.config.vidtwin_num_frames])
            action_velocity = self.action_out_proj(suffix_out[:, self.config.vidtwin_num_frames :])
            output["flow_losses"] = F.mse_loss(action_u_t, action_velocity, reduction="none")
            output["z_flow_losses"] = F.mse_loss(
                z_u_t.float(),
                z_velocity.float(),
                reduction="none",
            ).mean(dim=(1, 2))
            output["z_motion_source"] = z_motion_source
            output["z_target"] = z_target
            output["z_condition_active"] = action_can_see_z
            output["z_condition_step"] = z_condition_step
            if self.training:
                # One increment per optimizer-bound forward. All DDP ranks run
                # the same number of forwards, and this persistent buffer is
                # restored when resuming from a checkpoint.
                self.z_condition_step.add_(1)

        return output

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        past_motion: Tensor,
        noise: Tensor | None = None,
        z_noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Jointly denoise 16 temporal z tokens and H actions, then return actions."""
        batch_size = state.shape[0]
        device = state.device
        _, prefix_pad_masks, _, _, past_key_values = self.run_motion_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            past_motion,
        )

        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                device,
            )
        if z_noise is None:
            z_noise = self.sample_noise(
                (
                    batch_size,
                    self.config.vidtwin_num_frames,
                    self.config.motion_token_dim,
                ),
                device,
            )
        expected_z_shape = (
            batch_size,
            self.config.vidtwin_num_frames,
            self.config.motion_token_dim,
        )
        if z_noise.shape != expected_z_shape:
            raise ValueError(f"z_noise must have shape {expected_z_shape}, got {tuple(z_noise.shape)}.")
        if self._rtc_enabled():
            raise NotImplementedError("RTC is not supported by SmolW's joint (z, action) denoising.")

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        action_x_t = noise
        z_x_t = z_noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)
            action_v_t, z_v_t = self.denoise_step_with_z(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                action_x_t=action_x_t,
                z_x_t=z_x_t,
                timestep=time_tensor,
            )
            action_x_t = action_x_t + dt * action_v_t
            z_x_t = z_x_t + dt * z_v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=action_x_t, v_t=action_v_t)
        return action_x_t

    def denoise_step_with_z(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        action_x_t: Tensor,
        z_x_t: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor]:
        suffix_embs, z_pad_masks, action_pad_masks, action_att_masks = self.embed_action_z_suffix(
            action_x_t,
            z_x_t,
            timestep,
        )
        attention, position_ids = self.make_action_z_attention(
            prefix_pad_masks,
            z_pad_masks,
            action_pad_masks,
            action_att_masks,
        )
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_length = self.config.vidtwin_num_frames + self.config.chunk_size
        suffix_out = outputs_embeds[1][:, -suffix_length:].float()
        z_velocity = self.z_token_out_proj(suffix_out[:, : self.config.vidtwin_num_frames])
        action_velocity = self.action_out_proj(suffix_out[:, self.config.vidtwin_num_frames :])
        return action_velocity, z_velocity


class SmolWPolicy(SmolVLAPolicy):
    """LeRobot policy wrapper for SmolW."""

    config_class = SmolWConfig
    name = "smolw"

    def __init__(
        self,
        config: SmolWConfig,
        *,
        motion_extractor: VidTwinMotionExtractor | None = None,
        **kwargs,
    ) -> None:
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = SmolWFlowMatching(config, rtc_processor=self.rtc_processor)

        image_keys = list(config.image_features)
        if not image_keys:
            raise ValueError("SmolW requires at least one visual input feature.")
        self.motion_camera_key = config.motion_camera_key or image_keys[0]
        if self.motion_camera_key not in image_keys:
            raise ValueError(f"motion_camera_key={self.motion_camera_key!r} is not one of {image_keys}.")
        self.motion_extractor = (
            motion_extractor
            if motion_extractor is not None
            else VidTwinMotionExtractor(
                checkpoint_path=config.vidtwin_checkpoint_path,
                num_frames=config.vidtwin_num_frames,
                input_height=config.vidtwin_input_height,
                input_width=config.vidtwin_input_width,
                dtype=config.vidtwin_dtype,
                sample_posterior=config.vidtwin_sample_posterior,
                expected_latent_dim=config.motion_latent_dim,
            )
        )
        self.reset()

    def configure_train_mode(self) -> int:
        num_trainable = self.model.configure_train_mode()
        if num_trainable == 0:
            raise RuntimeError(f"SmolW train_mode={self.config.train_mode!r} left no trainable parameters.")
        logging.info(
            "Configured SmolW train_mode=%s with %d trainable parameters.",
            self.config.train_mode,
            num_trainable,
        )
        return num_trainable

    def get_optim_params(self):
        self.configure_train_mode()
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def reset(self) -> None:
        super().reset()
        history_span = (self.config.motion_horizon - 1) * self.config.memory_stride + 1
        self._motion_history: deque[Tensor] = deque(maxlen=history_span)

    def _select_current_observation(self, tensor: Tensor) -> Tensor:
        if tensor.ndim in {3, 5}:
            position = self.config.current_observation_position
            if tensor.shape[1] <= position:
                raise ValueError(
                    f"Temporal observation has {tensor.shape[1]} frames, but current position is {position}."
                )
            return tensor[:, position]
        return tensor

    def _current_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        current = dict(batch)
        for key in self.config.image_features:
            if key not in current:
                continue
            current[key] = self._select_current_observation(current[key])

            padding_key = f"{key}_padding_mask"
            if padding_key in current and current[padding_key].ndim >= 2:
                current[padding_key] = current[padding_key][:, self.config.current_observation_position]

            # LeRobot delta timestamps expose *_is_pad (True means invalid),
            # while SmolVLA consumes *_padding_mask (True means valid).
            is_pad_key = f"{key}_is_pad"
            if is_pad_key in current:
                is_pad = current[is_pad_key]
                if is_pad.ndim >= 2:
                    is_pad = is_pad[:, self.config.current_observation_position]
                current[padding_key] = ~is_pad.bool()

        if OBS_STATE in current:
            current[OBS_STATE] = self._select_current_observation(current[OBS_STATE])
        return current

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply the inherited Aloha state transform along the feature axis."""
        if not self.config.adapt_to_pi_aloha:
            return batch
        prepared = dict(batch)
        state = batch[OBS_STATE].clone()
        original_shape = state.shape
        prepared[OBS_STATE] = self._pi_aloha_decode_state(state.reshape(-1, original_shape[-1])).reshape(
            original_shape
        )
        return prepared

    def prepare_images(self, batch):
        return super().prepare_images(self._current_batch(batch))

    def prepare_state(self, batch):
        state = self._current_batch(batch)[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def _temporal_motion_frames(self, batch: dict[str, Tensor]) -> Tensor:
        frames = batch.get(self.motion_camera_key)
        if frames is None:
            raise ValueError(f"SmolW motion camera {self.motion_camera_key!r} is missing from the batch.")
        if frames.ndim != 5:
            raise ValueError(
                "SmolW training requires LeRobot delta-timestamp images [B,T,C,H,W]; got "
                f"{tuple(frames.shape)} for {self.motion_camera_key!r}."
            )
        if frames.shape[1] != len(self.config.observation_delta_indices):
            raise ValueError(
                f"Expected exactly {len(self.config.observation_delta_indices)} temporal observations, "
                f"got {frames.shape[1]}."
            )
        return frames

    def prepare_past_motion_clip(self, batch: dict[str, Tensor]) -> Tensor:
        frames = self._temporal_motion_frames(batch)
        past_positions = torch.tensor(
            self.config.past_motion_positions,
            dtype=torch.long,
            device=frames.device,
        )
        return frames.index_select(1, past_positions)

    def prepare_motion_clips(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        frames = self._temporal_motion_frames(batch)
        past_positions = torch.tensor(
            self.config.past_motion_positions,
            dtype=torch.long,
            device=frames.device,
        )
        future_positions = torch.tensor(
            self.config.future_motion_positions,
            dtype=torch.long,
            device=frames.device,
        )
        future_pad_key = f"{self.motion_camera_key}_is_pad"
        if future_pad_key in batch:
            future_is_pad = batch[future_pad_key].index_select(1, future_positions).bool()
            if torch.any(future_is_pad):
                raise ValueError(
                    "SmolW received padded future frames. Ensure the LeRobot sampler uses "
                    "config.drop_n_last_frames >= motion_horizon."
                )
        return frames.index_select(1, past_positions), frames.index_select(1, future_positions)

    def _append_current_motion_frame(self, batch: dict[str, Tensor]) -> None:
        frames = batch.get(self.motion_camera_key)
        if frames is None:
            raise ValueError(f"SmolW motion camera {self.motion_camera_key!r} is missing from the batch.")
        current = self._select_current_observation(frames)
        if current.ndim != 4:
            raise ValueError(f"Expected current motion image [B,C,H,W], got {tuple(current.shape)}.")
        self._motion_history.append(current.detach().clone())

    def _history_motion_clip(self) -> Tensor:
        if not self._motion_history:
            raise RuntimeError("SmolW motion history is empty; append the current observation first.")
        history = list(self._motion_history)
        frame_count = self.config.motion_horizon
        stride = self.config.memory_stride
        sampled = []
        for index in range(frame_count):
            history_index = len(history) - 1 - (frame_count - 1 - index) * stride
            sampled.append(history[max(0, history_index)])
        return torch.stack(sampled, dim=1)

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        past_motion = self.motion_extractor.encode(self._history_motion_clip())
        actions = self.model.sample_actions(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            past_motion,
            noise=noise,
            **kwargs,
        )
        actions = actions[:, :, : self.config.action_feature.shape[0]]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        self._append_current_motion_frame(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch, noise, **kwargs)

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if self._rtc_enabled():
            raise AssertionError("RTC is not supported for select_action; use predict_action_chunk.")
        self.eval()
        batch = self._prepare_batch(batch)
        # History must advance on every environment step, including steps that
        # consume an already-predicted action queue.
        self._append_current_motion_frame(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        z_noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float | list[float]]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'none'.")
        batch = self._prepare_batch(batch)
        if self.config.adapt_to_pi_aloha:
            batch = dict(batch)
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION].clone())

        train_mode = self.config.train_mode
        compute_motion_loss = train_mode in {"motion_only", "jointly"}
        compute_flow = train_mode in {"action_only", "jointly"}
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch) if compute_flow else None

        past_frames, future_frames = self.prepare_motion_clips(batch)
        past_motion, future_motion_target = self.motion_extractor.encode_pair(
            past_frames,
            future_frames,
        )
        # action_only isolates action learning by pretending the motion
        # predictor is perfect. jointly must use its own prediction so action
        # gradients can train the complete causal chain.
        z_motion_source = future_motion_target if train_mode == "action_only" else None

        output = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            past_motion,
            actions=actions,
            future_motion_target=future_motion_target,
            z_motion_source=z_motion_source,
            noise=noise,
            z_noise=z_noise,
            time=time,
            compute_motion_loss=compute_motion_loss,
            compute_flow=compute_flow,
        )

        scalar_terms: list[Tensor] = []
        per_sample_terms: list[Tensor] = []
        metrics: dict[str, float | list[float]] = {}
        if compute_flow:
            action_dim = self.config.action_feature.shape[0]
            flow_losses = output["flow_losses"][:, :, :action_dim]
            actions_is_pad = batch.get("action_is_pad")
            if actions_is_pad is None:
                per_sample_flow = flow_losses.mean(dim=(1, 2))
                flow_loss = per_sample_flow.mean()
                loss_per_dim = flow_losses.mean(dim=(0, 1))
            else:
                valid = (~actions_is_pad.bool()).to(
                    dtype=flow_losses.dtype,
                    device=flow_losses.device,
                )
                masked_flow = flow_losses * valid.unsqueeze(-1)
                valid_steps = valid.sum().clamp_min(1)
                per_sample_steps = valid.sum(dim=1).clamp_min(1)
                per_sample_flow = masked_flow.sum(dim=(1, 2)) / (per_sample_steps * action_dim)
                flow_loss = masked_flow.sum() / (valid_steps * action_dim)
                loss_per_dim = masked_flow.sum(dim=(0, 1)) / valid_steps
            per_sample_z_flow = output["z_flow_losses"]
            z_flow_loss = per_sample_z_flow.mean()
            # During action warmup, keep the z branch in the graph (important
            # for DDP) but give it exactly zero gradient. At the same boundary
            # action->z attention and the configured z objective turn on.
            effective_z_loss_weight = self.config.z_loss_weight if output["z_condition_active"] else 0.0
            weighted_z_flow = effective_z_loss_weight * z_flow_loss
            combined_flow_loss = flow_loss + weighted_z_flow
            scalar_terms.append(combined_flow_loss)
            per_sample_terms.append(per_sample_flow + effective_z_loss_weight * per_sample_z_flow)
            metrics.update(
                {
                    "flow_loss": combined_flow_loss.item(),
                    "action_flow_loss": flow_loss.item(),
                    "z_flow_loss": z_flow_loss.item(),
                    "weighted_z_flow_loss": weighted_z_flow.item(),
                    "effective_z_loss_weight": float(effective_z_loss_weight),
                    "loss_per_dim": loss_per_dim.detach().cpu().tolist(),
                    "z_target_rms": output["z_target"].float().square().mean().sqrt().item(),
                    "z_condition_active": float(output["z_condition_active"]),
                    "z_condition_step": float(output["z_condition_step"]),
                }
            )

        if compute_motion_loss:
            per_sample_motion = output["motion_losses"]
            motion_loss = per_sample_motion.mean()
            weighted_motion = self.config.motion_loss_weight * motion_loss
            scalar_terms.append(weighted_motion)
            per_sample_terms.append(self.config.motion_loss_weight * per_sample_motion)
            metrics.update(
                {
                    "motion_loss": motion_loss.item(),
                    "weighted_motion_loss": weighted_motion.item(),
                    "target_motion_rms": future_motion_target.float().square().mean().sqrt().item(),
                }
            )

        if not scalar_terms:
            raise RuntimeError(f"SmolW train_mode {train_mode!r} did not produce any training objective.")
        total_loss = torch.stack(scalar_terms).sum()
        per_sample_total = torch.stack(per_sample_terms, dim=0).sum(dim=0)
        metrics.update(
            {
                "loss": total_loss.item(),
                "predicted_motion_rms": output["predicted_future_motion"]
                .float()
                .square()
                .mean()
                .sqrt()
                .item(),
            }
        )
        if train_mode == "action_only":
            metrics["oracle_motion_rms"] = future_motion_target.float().square().mean().sqrt().item()
        if reduction == "none":
            return per_sample_total, metrics
        return total_loss, metrics

    def _get_default_peft_targets(self) -> dict[str, object]:
        defaults = super()._get_default_peft_targets()
        defaults["modules_to_save"] = [
            "mt_query_embedding",
            "past_motion_projector",
            "future_motion_head",
            "z_token_in_proj",
            "z_time_mlp_in",
            "z_time_mlp_out",
            "z_token_out_proj",
        ]
        return defaults
