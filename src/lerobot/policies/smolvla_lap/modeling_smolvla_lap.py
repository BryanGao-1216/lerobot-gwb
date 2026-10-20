"""SmolVLM latent-action pretraining followed by native SmolVLA flow matching."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_smolvla_lap import SmolVLALAPConfig
from .dataset import LAM_IMAGE_PAIR


class LAPFlowMatching(VLAFlowMatching):
    def __init__(self, config: SmolVLALAPConfig, rtc_processor=None):
        # On checkpoint reload, construct the architecture without downloading/loading
        # the original VLM weights a second time. The policy loader restores them.
        backbone_config = replace(
            config, load_vlm_weights=config.load_vlm_weights and not config.pretrained_path
        )
        super().__init__(backbone_config, rtc_processor=rtc_processor)
        self.config = config
        self._init_latent_vocabulary()
        if config.training_stage == "pre-training":
            # Override the expert-only and unused-parameter freezes of native SmolVLA.
            self.requires_grad_(False)
            self.vlm_with_expert.vlm.requires_grad_(True)
            self.state_proj.requires_grad_(True)
            self.vlm_with_expert.train_expert_only = False
            if config.freeze_vision_encoder:
                self.vlm_with_expert.get_vlm_model().vision_model.requires_grad_(False)
        else:
            # Resizing creates embedding/head parameters; reapply native freezing rules.
            self.vlm_with_expert.set_requires_grad()

    def _init_latent_vocabulary(self) -> None:
        tokenizer = self.vlm_with_expert.processor.tokenizer
        names = [f"<|latent_action_{i}|>" for i in range(self.config.lam_codebook_size)]
        bos_name = "<|latent_action_start|>"
        # Preserve the processor's existing image special-token registrations.
        # add_tokens(..., special_tokens=True) also works across Transformers 5.x
        # versions that renamed additional_special_tokens to extra_special_tokens.
        tokenizer.add_tokens([bos_name, *names], special_tokens=True)
        self.vlm_with_expert.vlm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        ids = tokenizer.convert_tokens_to_ids(names)
        bos_id = tokenizer.convert_tokens_to_ids(bos_name)
        if self.config.latent_action_token_ids and self.config.latent_action_token_ids != ids:
            raise ValueError("Checkpoint latent-action token IDs do not match the VLM tokenizer.")
        if self.config.latent_action_bos_token_id not in (None, bos_id):
            raise ValueError("Checkpoint latent-action start token does not match the VLM tokenizer.")
        self.config.latent_action_token_ids = ids
        self.config.latent_action_bos_token_id = bos_id
        self.register_buffer("latent_token_ids", torch.tensor(ids, dtype=torch.long), persistent=False)

    def latent_action_logits(self, images, image_masks, language, language_masks, state, previous_codes):
        prefix, pad_mask, ar_mask = self.embed_prefix(images, image_masks, language, language_masks, state)
        batch_size = prefix.shape[0]
        bos = torch.full(
            (batch_size, 1), self.config.latent_action_bos_token_id, device=prefix.device, dtype=torch.long
        )
        input_ids = torch.cat([bos, self.latent_token_ids[previous_codes]], dim=1)
        suffix = self.vlm_with_expert.embed_language_tokens(input_ids) * math.sqrt(prefix.shape[-1])
        embeddings = torch.cat([prefix, suffix], dim=1)
        suffix_mask = torch.ones_like(input_ids, dtype=torch.bool)
        pad_mask = torch.cat([pad_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([ar_mask, suffix_mask], dim=1)
        # Prefix retains SmolVLA's image/language/state visibility. Every suffix
        # position is causal, so no label can see itself or any future action code.
        attention_mask = make_att_2d_masks(pad_mask, ar_mask)
        position_ids = (pad_mask.long().cumsum(dim=1) - 1).clamp_min(0)
        (hidden, _), _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[embeddings, None],
            use_cache=False,
            # Route every layer through VLM self-attention, without the action expert.
            fill_kv_cache=True,
        )
        head = self.vlm_with_expert.vlm.get_output_embeddings()
        return head(hidden[:, -input_ids.shape[1] :].to(dtype=head.weight.dtype))


class SmolVLALAPPolicy(SmolVLAPolicy):
    config_class = SmolVLALAPConfig
    name = "smolvla_lap"

    def __init__(self, config: SmolVLALAPConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.init_rtc_processor()
        self.model = LAPFlowMatching(config, rtc_processor=self.rtc_processor)
        # A plain encoder object loaded lazily on the correct per-rank CUDA device.
        # No teacher tensors are registered or written into the policy checkpoint.
        self._lam_encoder = None
        self.reset()

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        # Both stages have identical saved architecture. Reject incomplete or
        # ordinary SmolVLA checkpoints instead of silently leaving random weights.
        return super()._load_as_safetensor(model, model_file, map_location, strict=True)

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        if self.config.training_stage == "post-training":
            return super().forward(batch, noise=noise, time=time, reduction=reduction)
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'.")
        if LAM_IMAGE_PAIR not in batch:
            raise ValueError("Missing raw LAM frame pairs; use the LAP LeRobot dataset loader.")
        frames = batch[LAM_IMAGE_PAIR]
        if self._lam_encoder is None:
            from .lam import make_lam_encoder

            self._lam_encoder = make_lam_encoder(self.config, frames.device)
        target_codes = self._lam_encoder.encode(frames)
        images, image_masks = self.prepare_images(batch)
        logits = self.model.latent_action_logits(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch),
            target_codes[:, :-1],
        )
        targets = self.model.latent_token_ids[target_codes]
        losses = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
        per_sample_loss = losses.mean(dim=1)
        correct = logits.detach().argmax(dim=-1) == targets
        metrics = {
            "loss": per_sample_loss.detach().mean().item(),
            "latent_token_accuracy": correct.float().mean().item(),
            "latent_sequence_accuracy": correct.all(dim=1).float().mean().item(),
        }
        return (per_sample_loss if reduction == "none" else per_sample_loss.mean()), metrics

    @torch.no_grad()
    def generate_latent_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Greedily generate the ordered LAM code IDs from current observations only."""
        self.eval()
        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        codes = torch.empty((state.shape[0], 0), device=state.device, dtype=torch.long)
        for _ in range(self.config.lam_num_tokens):
            logits = self.model.latent_action_logits(
                images,
                image_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                state,
                codes,
            )
            next_code = logits[:, -1, self.model.latent_token_ids].argmax(dim=-1, keepdim=True)
            codes = torch.cat([codes, next_code], dim=1)
        return codes

    def select_action(self, batch, **kwargs):
        if self.config.training_stage == "pre-training":
            raise RuntimeError(
                "Use generate_latent_actions() for pre-training checkpoints, or switch to post-training."
            )
        return super().select_action(batch, **kwargs)

    def predict_action_chunk(self, batch, **kwargs):
        if self.config.training_stage == "pre-training":
            raise RuntimeError("Continuous action prediction requires training_stage='post-training'.")
        return super().predict_action_chunk(batch, **kwargs)
