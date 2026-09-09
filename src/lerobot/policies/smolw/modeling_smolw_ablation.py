"""Controlled B/C/D additions to the original SmolVLA action path.

A uses VLAFlowMatching itself. This module leaves the legacy SmolW graph intact
for checkpoints with no train_version, and reuses its motion projections only.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ..smolvla.modeling_smolvla import make_att_2d_masks
from .modeling_smolw import SmolWFlowMatching


class SmolWAblationFlowMatching(SmolWFlowMatching):
    def __init__(self, config, rtc_processor=None):
        compile_model = config.compile_model
        config.compile_model = False
        try:
            super().__init__(config, rtc_processor=rtc_processor)
        finally:
            config.compile_model = compile_model

        if config.train_version == "B":
            for name in ("z_token_in_proj", "z_time_mlp_in", "z_time_mlp_out", "z_token_out_proj"):
                delattr(self, name)
        elif config.train_version == "D":
            # An independent attention normalization avoids changing the base
            # action attention's softmax denominator. At gate=0, D equals C.
            # Keep C and D's subsequent random draws aligned at initialization.
            with torch.random.fork_rng(devices=[]):
                self.motion_condition_attention = nn.MultiheadAttention(
                    self.vlm_with_expert.expert_hidden_size, num_heads=1, batch_first=True
                )
                self.motion_condition_gate = nn.Embedding(1, 1)
                nn.init.zeros_(self.motion_condition_gate.weight)

        if compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    @staticmethod
    def make_ablation_attention(prefix_pad_masks, action_pad_masks, action_att_masks, z_pad_masks=None):
        """All actions see M_t; z is isolated from actions in the shared backbone.

        Action position IDs are identical in B/C/D. z uses a separate temporal
        stream with the same origin, so adding z never shifts action positions.
        Only D's explicit residual attention connects the two streams.
        """
        batch_size, horizon = action_pad_masks.shape
        prefix_len = prefix_pad_masks.shape[1]
        action_prefix = prefix_pad_masks[:, None, :].expand(batch_size, horizon, prefix_len)
        action_self = make_att_2d_masks(action_pad_masks, action_att_masks)
        offsets = prefix_pad_masks.sum(dim=1, keepdim=True)
        action_positions = offsets + action_pad_masks.cumsum(dim=1) - 1
        if z_pad_masks is None:
            return torch.cat([action_prefix, action_self], dim=2), action_positions

        z_count = z_pad_masks.shape[1]
        z_prefix = prefix_pad_masks[:, None, :].expand(batch_size, z_count, prefix_len)
        z_self = z_pad_masks[:, None, :].expand(batch_size, z_count, z_count)
        no_z_to_action = torch.zeros(batch_size, z_count, horizon, dtype=torch.bool, device=offsets.device)
        z_rows = torch.cat([z_prefix, z_self, no_z_to_action], dim=2)
        action_rows = torch.cat([action_prefix, no_z_to_action.transpose(1, 2), action_self], dim=2)
        attention = torch.cat([z_rows, action_rows], dim=1)
        suffix_pad = torch.cat([z_pad_masks, action_pad_masks], dim=1)
        attention &= suffix_pad[:, :, None]
        attention &= torch.cat([prefix_pad_masks, suffix_pad], dim=1)[:, None, :]
        z_positions = offsets + z_pad_masks.cumsum(dim=1) - 1
        return attention, torch.cat([z_positions, action_positions], dim=1)

    def _embed_ablation_suffix(self, prefix_pad_masks, actions, timestep, z=None):
        if z is None:
            suffix, action_pad, action_att = self.embed_suffix(actions, timestep)
            z_pad = None
        else:
            suffix, z_pad, action_pad, action_att = self.embed_action_z_suffix(actions, z, timestep)
        attention, positions = self.make_ablation_attention(prefix_pad_masks, action_pad, action_att, z_pad)
        return suffix, attention, positions

    def _project_ablation_output(self, suffix_out: Tensor, *, has_z: bool):
        suffix_out = suffix_out.float()
        if not has_z:
            return self.action_out_proj(suffix_out), None
        z_hidden = suffix_out[:, : self.config.vidtwin_num_frames]
        action_hidden = suffix_out[:, self.config.vidtwin_num_frames :]
        z_velocity = self.z_token_out_proj(z_hidden)
        if self.config.train_version == "D":
            motion_update, _ = self.motion_condition_attention(
                action_hidden, z_hidden, z_hidden, need_weights=False
            )
            gate = self.motion_condition_gate.weight.tanh().to(action_hidden.dtype)
            action_hidden = action_hidden + gate * motion_update
        return self.action_out_proj(action_hidden), z_velocity

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        past_motion,
        actions,
        future_motion=None,
        noise=None,
        z_noise=None,
        time=None,
    ):
        # Keep the original action noise/time sampling order. z is additional.
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        t = time[:, None, None]
        action_x = t * noise + (1 - t) * actions

        z_target = z_x = None
        if self.config.uses_future_motion:
            if future_motion is None:
                raise ValueError("C/D training requires future motion targets.")
            z_target = self.motion_to_z(future_motion)
            if z_noise is None:
                z_noise = self.sample_noise(z_target.shape, z_target.device)
            if z_noise.shape != z_target.shape:
                raise ValueError(
                    f"Expected z_noise shape {tuple(z_target.shape)}, got {tuple(z_noise.shape)}."
                )
            z_x = t * z_noise + (1 - t) * z_target

        prefix, prefix_pad, prefix_att = self.embed_prefix_with_motion(
            images, img_masks, lang_tokens, lang_masks, state, past_motion
        )
        suffix, suffix_attention, suffix_positions = self._embed_ablation_suffix(
            prefix_pad, action_x, time, z_x
        )
        prefix_attention = make_att_2d_masks(prefix_pad, prefix_att)
        # Like original SmolVLA, training runs prefix and suffix together,
        # without a KV cache. Prefix queries cannot read either suffix stream.
        prefix_rows = F.pad(prefix_attention, (0, suffix.shape[1]), value=False)
        attention = torch.cat([prefix_rows, suffix_attention], dim=1)
        positions = torch.cat([prefix_pad.cumsum(dim=1) - 1, suffix_positions], dim=1)
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            fill_kv_cache=False,
        )
        action_v, z_v = self._project_ablation_output(suffix_out, has_z=z_target is not None)
        output = {"flow_losses": F.mse_loss(noise - actions, action_v, reduction="none")}
        if z_target is not None:
            output["z_flow_losses"] = F.mse_loss(z_noise - z_target, z_v, reduction="none").mean(dim=(1, 2))
            output["z_target"] = z_target
        return output

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        past_motion,
        noise=None,
        z_noise=None,
        **kwargs,
    ):
        if self._rtc_enabled():
            raise NotImplementedError("RTC is not supported by SmolW B/C/D.")
        batch_size, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )
        prefix_pad, cache = self.run_condition_prefix(
            images, img_masks, lang_tokens, lang_masks, state, past_motion
        )
        z_x = None
        # C's z objective is auxiliary: action inference is exactly the B graph.
        if self.config.train_version == "D":
            z_shape = (batch_size, self.config.vidtwin_num_frames, self.config.motion_token_dim)
            if z_noise is None:
                z_noise = self.sample_noise(z_shape, device)
            if z_noise.shape != z_shape:
                raise ValueError(f"Expected z_noise shape {z_shape}, got {tuple(z_noise.shape)}.")
            z_x = z_noise

        action_x = noise
        dt = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            time = torch.full((batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device)
            suffix, attention, positions = self._embed_ablation_suffix(prefix_pad, action_x, time, z_x)
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=attention,
                position_ids=positions,
                past_key_values=cache,
                inputs_embeds=[None, suffix],
                use_cache=True,
                fill_kv_cache=False,
            )
            action_v, z_v = self._project_ablation_output(suffix_out, has_z=z_x is not None)
            action_x = action_x + dt * action_v
            if z_x is not None:
                z_x = z_x + dt * z_v
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time[0].item(), x_t=action_x, v_t=action_v)
        return action_x
