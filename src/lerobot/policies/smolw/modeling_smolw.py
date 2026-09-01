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
    make_att_2d_masks,
    pad_vector,
)
from ..utils import populate_queues
from .configuration_smolw import SmolWConfig
from .vidtwin_motion_encoder import VidTwinMotionExtractor


class SmolWFlowMatching(VLAFlowMatching):
    """Original SmolVLA action expert with a predicted-motion suffix condition."""

    def __init__(self, config: SmolWConfig, rtc_processor=None):
        # The parent compiles its methods inside __init__. Delay compilation
        # until all SmolW modules exist so torch.compile sees the final graph.
        compile_model = config.compile_model
        config.compile_model = False
        try:
            super().__init__(config, rtc_processor=rtc_processor)
        finally:
            config.compile_model = compile_model

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
        self.future_motion_condition_proj = nn.Sequential(
            nn.LayerNorm(config.motion_latent_dim),
            nn.Linear(config.motion_latent_dim, config.motion_condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.motion_condition_hidden_dim, expert_hidden_size),
        )

        nn.init.normal_(self.mt_query_embedding.weight, mean=0.0, std=0.02)
        # Preserve the original SmolVLA vector field at initialization: the
        # action expert cannot attend M_t directly, and this explicit predicted
        # motion residual starts at zero.
        nn.init.zeros_(self.future_motion_condition_proj[-1].weight)
        nn.init.zeros_(self.future_motion_condition_proj[-1].bias)

        if compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

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

    def embed_suffix_with_motion(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        predicted_future_motion: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Add predicted future motion to every original action/time suffix token."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = super().embed_suffix(noisy_actions, timestep)
        motion_for_condition = (
            predicted_future_motion.detach()
            if self.config.detach_motion_condition
            else predicted_future_motion
        )
        projection_dtype = self.future_motion_condition_proj[1].weight.dtype
        condition = self.future_motion_condition_proj(motion_for_condition.to(dtype=projection_dtype))
        condition = torch.tanh(condition) * self.config.motion_condition_scale
        suffix_embs = suffix_embs + condition[:, None, :].to(dtype=suffix_embs.dtype)
        return suffix_embs, suffix_pad_masks, suffix_att_masks

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

        M_t is always the final cached prefix token.  The action expert receives
        its information only through ``predicted_future_motion``; current image,
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

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        actions: Tensor,
        past_motion: Tensor,
        future_motion_target: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict future motion first, then evaluate one flow-matching timestep."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_with_motion(
            x_t,
            time,
            predicted_future_motion,
        )

        attention, position_ids = self.make_action_attention(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].float()
        velocity = self.action_out_proj(suffix_out)
        flow_losses = F.mse_loss(u_t, velocity, reduction="none")
        motion_losses = F.mse_loss(
            predicted_future_motion,
            future_motion_target.float(),
            reduction="none",
        ).mean(dim=-1)
        return {
            "flow_losses": flow_losses,
            "motion_losses": motion_losses,
            "predicted_future_motion": predicted_future_motion,
        }

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
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Run one motion prediction followed by the original Euler denoising loop."""
        batch_size = state.shape[0]
        device = state.device
        predicted_future_motion, prefix_pad_masks, _, _, past_key_values = self.run_motion_prefix(
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

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step_with_motion(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    predicted_future_motion=predicted_future_motion,
                )

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise_step_partial_call(x_t)
            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        return x_t

    def denoise_step_with_motion(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
        predicted_future_motion: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_with_motion(
            x_t,
            timestep,
            predicted_future_motion,
        )
        attention, position_ids = self.make_action_attention(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
        )
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].float()
        return self.action_out_proj(suffix_out)


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
                repo_path=config.vidtwin_repo_path,
                config_path=config.vidtwin_config_path,
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

    def prepare_motion_clips(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        frames = batch.get(self.motion_camera_key)
        if frames is None:
            raise ValueError(f"SmolW motion camera {self.motion_camera_key!r} is missing from the batch.")
        if frames.ndim != 5:
            raise ValueError(
                "SmolW training requires LeRobot delta-timestamp images [B,T,C,H,W]; got "
                f"{tuple(frames.shape)} for {self.motion_camera_key!r}."
            )
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
        if frames.shape[1] != len(self.config.observation_delta_indices):
            raise ValueError(
                f"Expected exactly {len(self.config.observation_delta_indices)} temporal observations, "
                f"got {frames.shape[1]}."
            )

        future_pad_key = f"{self.motion_camera_key}_is_pad"
        if future_pad_key in batch:
            future_is_pad = batch[future_pad_key].index_select(1, future_positions).bool()
            if torch.any(future_is_pad):
                raise ValueError(
                    "SmolW received padded future frames. Ensure the LeRobot sampler uses "
                    "config.drop_n_last_frames >= motion_horizon - 1."
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
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float | list[float]]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'none'.")
        batch = self._prepare_batch(batch)
        if self.config.adapt_to_pi_aloha:
            batch = dict(batch)
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION].clone())

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        past_frames, future_frames = self.prepare_motion_clips(batch)
        past_motion, future_motion_target = self.motion_extractor.encode_pair(past_frames, future_frames)

        output = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            actions,
            past_motion,
            future_motion_target,
            noise=noise,
            time=time,
        )

        action_dim = self.config.action_feature.shape[0]
        flow_losses = output["flow_losses"][:, :, :action_dim]
        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is None:
            per_sample_flow = flow_losses.mean(dim=(1, 2))
            flow_loss = per_sample_flow.mean()
            loss_per_dim = flow_losses.mean(dim=(0, 1))
        else:
            valid = (~actions_is_pad.bool()).to(dtype=flow_losses.dtype, device=flow_losses.device)
            masked_flow = flow_losses * valid.unsqueeze(-1)
            valid_steps = valid.sum().clamp_min(1)
            per_sample_steps = valid.sum(dim=1).clamp_min(1)
            per_sample_flow = masked_flow.sum(dim=(1, 2)) / (per_sample_steps * action_dim)
            flow_loss = masked_flow.sum() / (valid_steps * action_dim)
            loss_per_dim = masked_flow.sum(dim=(0, 1)) / valid_steps

        per_sample_motion = output["motion_losses"]
        motion_loss = per_sample_motion.mean()
        per_sample_total = per_sample_flow + self.config.motion_loss_weight * per_sample_motion
        total_loss = flow_loss + self.config.motion_loss_weight * motion_loss
        metrics: dict[str, float | list[float]] = {
            "loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "motion_loss": motion_loss.item(),
            "weighted_motion_loss": (self.config.motion_loss_weight * motion_loss).item(),
            "loss_per_dim": loss_per_dim.detach().cpu().tolist(),
            "predicted_motion_rms": output["predicted_future_motion"].float().square().mean().sqrt().item(),
            "target_motion_rms": future_motion_target.float().square().mean().sqrt().item(),
        }
        if reduction == "none":
            return per_sample_total, metrics
        return total_loss, metrics

    def _get_default_peft_targets(self) -> dict[str, object]:
        defaults = super()._get_default_peft_targets()
        defaults["modules_to_save"] = [
            "mt_query_embedding",
            "past_motion_projector",
            "future_motion_head",
            "future_motion_condition_proj",
        ]
        return defaults
