#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None


from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_DISTANCES,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ..action_code import (
    ActionCodeLayout,
    compute_action_code_objective,
    condition_flow_hidden,
    fill_missing_initialized_state,
    reduce_flow_losses,
    validate_action_code_sequence,
)
from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import (
    clone_past_key_values,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    prepare_attention_masks_4d,
    resize_with_pad_torch,
)
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi05_actionmem import DEFAULT_IMAGE_SIZE, PI05ActionMemConfig


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI0.5 ActionMem."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Align with PI05.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32). Align with PI05.
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05ActionMemPytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05ActionMem PyTorch model."""

    def __init__(self, config: PI05ActionMemConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self.action_code_layout = ActionCodeLayout(
            codebook_size=config.action_codebook_size,
            invalid_value=config.action_code_invalid_value,
        )

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.training_stage == "action_expert_only",
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.action_code_embedding = nn.Embedding(
            self.action_code_layout.context_size,
            paligemma_config.width,
            padding_idx=self.action_code_layout.padding_id,
        )
        self.action_classifier = nn.Linear(paligemma_config.width, self.action_code_layout.codebook_size)
        self.action_condition_proj = nn.Sequential(
            nn.LayerNorm(self.action_code_layout.codebook_size),
            nn.Linear(self.action_code_layout.codebook_size, config.action_condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.action_condition_hidden_dim, action_expert_config.width * 2),
        )
        nn.init.normal_(self.action_code_embedding.weight, mean=0.0, std=config.action_code_init_std)
        nn.init.normal_(self.action_classifier.weight, mean=0.0, std=config.action_code_init_std)
        nn.init.zeros_(self.action_classifier.bias)
        nn.init.zeros_(self.action_condition_proj[-1].weight)
        nn.init.zeros_(self.action_condition_proj[-1].bias)
        with torch.no_grad():
            self.action_code_embedding.weight[self.action_code_layout.padding_id].zero_()

        # Native PI0.5 puts state in the text prefix and conditions its AdaRMS
        # action expert on a standalone timestep embedding.
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.configure_training_stage()

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    @staticmethod
    def _is_vlm_parameter(name: str) -> bool:
        return name.startswith(
            (
                "paligemma_with_expert.paligemma.",
                "action_code_embedding.",
                "action_classifier.",
            )
        )

    @staticmethod
    def _is_action_expert_parameter(name: str) -> bool:
        return name.startswith("paligemma_with_expert.gemma_expert.") or name.startswith(
            (
                "action_in_proj.",
                "action_out_proj.",
                "action_code_embedding.",
                "action_condition_proj.",
                "time_mlp_in.",
                "time_mlp_out.",
            )
        )

    def configure_training_stage(self) -> int:
        """Freeze the branch excluded by the configured training stage.

        This is intentionally safe to call again after PEFT adapters have been
        injected: adapter parameters inherit the same module-name prefixes and
        are filtered along with the base parameters.
        """
        stage = self.config.training_stage
        for name, parameter in self.named_parameters():
            is_excluded = (stage == "vlm_only" and not self._is_vlm_parameter(name)) or (
                stage == "action_expert_only" and not self._is_action_expert_parameter(name)
            )
            if is_excluded:
                parameter.requires_grad_(False)
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        # PiGemma decoder layers inherit Transformers' GradientCheckpointingLayer.
        # Use the public API so every decoder layer receives both the flag and
        # checkpoint function. Setting only `language_model.gradient_checkpointing`
        # leaves its layers uncheckpointed and makes the VLM-only backward
        # recompute the whole 2B model in one memory-heavy graph.
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": False,
            }
        )
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05ActionMemPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing_disable()
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05ActionMemPytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

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
            policy_name="PI05ActionMem",
        )

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens=None,
        action_token_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images, PI0.5 task/state text, memory, query, and target."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if action_tokens is not None:
            if action_token_masks is None or action_tokens.shape != action_token_masks.shape:
                raise ValueError(
                    "action_tokens and action_token_masks must have the same shape, got "
                    f"{tuple(action_tokens.shape)} and "
                    f"{None if action_token_masks is None else tuple(action_token_masks.shape)}"
                )

            def action_token_embed_func(action_tokens):
                return self.action_code_embedding(action_tokens)

            action_token_emb = self._apply_checkpoint(action_token_embed_func, action_tokens)
            embs.append(action_token_emb)
            pad_masks.append(action_token_masks.bool())
            # Each memory/query/target position begins a causal block. In
            # particular, ACTION_QUERY cannot attend to its following target.
            att_masks += [1] * action_token_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy actions and native PI0.5 AdaRMS timestep conditioning."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)

        embs.append(action_emb)
        bsize, action_dim = action_emb.shape[:2]
        action_mask = torch.ones(bsize, action_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens,
        action_token_masks,
        actions=None,
        noise=None,
        time=None,
        *,
        action_token_distances: Tensor | None = None,
        compute_flow: bool = True,
        compute_action_token: bool = True,
    ) -> dict[str, Tensor]:
        """Compute the objectives required by the selected training stage."""
        if not compute_flow and not compute_action_token:
            raise ValueError("At least one PI05ActionMem training objective must be enabled.")
        self._validate_action_token_sequence(action_tokens, action_token_masks)

        action_prompt_tokens = action_tokens[:, :-1]
        action_prompt_masks = action_token_masks[:, :-1]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_prompt_tokens,
            action_prompt_masks,
        )

        if not compute_flow:
            return self._forward_action_token_only(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                action_tokens,
                action_token_masks,
                action_token_distances,
            )

        if actions is None or time is None:
            raise ValueError("Flow training requires actions and time tensors.")
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # 增加一个 head 维度，对齐多头自注意力计算
        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        action_logits = self._compute_action_logits(prefix_out)
        suffix_out, condition_metrics = condition_flow_hidden(
            suffix_out,
            action_logits,
            self.action_condition_proj,
            scale=self.config.action_condition_scale,
        )

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        flow_losses = F.mse_loss(u_t, v_t, reduction="none")

        output = {"flow_losses": flow_losses, **condition_metrics}
        if compute_action_token:
            output.update(
                self._compute_action_token_objective(
                    action_logits,
                    action_tokens,
                    action_token_masks,
                    action_token_distances,
                )
            )
        return output

    def _forward_action_token_only(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        action_tokens: Tensor,
        action_token_masks: Tensor,
        action_token_distances: Tensor | None,
    ) -> dict[str, Tensor]:
        """Run only PaliGemma and the token head, skipping the flow branch."""
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        def forward_func(prefix_embs, attention_mask, position_ids):
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
            )
            return prefix_out

        # Do not wrap the entire PaliGemma forward in one checkpoint. When
        # gradient checkpointing is enabled, gradient_checkpointing_enable()
        # configures each PiGemma decoder layer independently. A whole-model
        # checkpoint causes backward recomputation to build the complete VLM
        # graph at once and can use more memory than the normal forward.
        prefix_out = forward_func(
            prefix_embs,
            prefix_att_2d_masks_4d,
            prefix_position_ids,
        )
        return self._compute_action_token_objective(
            self._compute_action_logits(prefix_out),
            action_tokens,
            action_token_masks,
            action_token_distances,
        )

    def _compute_action_logits(self, prefix_out: Tensor) -> Tensor:
        query_hidden = prefix_out[:, -1, :].to(dtype=self.action_classifier.weight.dtype)
        return self.action_classifier(query_hidden)

    def _compute_action_token_objective(
        self,
        action_logits: Tensor,
        action_tokens: Tensor,
        action_token_masks: Tensor,
        action_token_distances: Tensor | None,
    ) -> dict[str, Tensor]:
        return compute_action_code_objective(
            action_logits,
            action_tokens,
            action_token_masks,
            action_token_distances,
            temperature=self.config.action_token_soft_target_temperature,
            policy_name="PI05ActionMem",
        )

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_tokens,
        action_token_masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        self._validate_action_token_sequence(action_tokens, action_token_masks)

        # The processor reserves the final position for the current action-token
        # target. At inference it is padding, so prefill only through ACTION_QUERY.
        action_prompt_tokens = action_tokens[:, :-1]
        action_prompt_masks = action_token_masks[:, :-1]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_prompt_tokens,
            action_prompt_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        action_logits = self._compute_action_logits(prefix_out)

        if noise is None:
            noise = self.sample_noise(
                (lang_tokens.shape[0], self.config.chunk_size, self.config.max_action_dim),
                lang_tokens.device,
            )

        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                action_logits=action_logits,
                x_t=input_x_t,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        action_logits,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out, _ = condition_flow_hidden(
            suffix_out,
            action_logits,
            self.action_condition_proj,
            scale=self.config.action_condition_scale,
        )
        return self.action_out_proj(suffix_out)


class PI05ActionMemPolicy(PreTrainedPolicy):
    """PI05ActionMem Policy for LeRobot."""

    config_class = PI05ActionMemConfig
    name = "pi05_actionmem"

    def __init__(
        self,
        config: PI05ActionMemConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05ActionMem model
        self.init_rtc_processor()
        self.model = PI05ActionMemPytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def configure_training_stage(self) -> int:
        """Apply stage-specific freezing, including to injected PEFT adapters."""
        num_trainable = self.model.configure_training_stage()
        if num_trainable == 0:
            raise RuntimeError(
                f"PI05ActionMem training_stage={self.config.training_stage!r} left no trainable parameters. "
                "Check the PEFT target_modules for the selected branch."
            )
        logging.info(
            "Configured PI05ActionMem training_stage=%s with %d trainable parameters.",
            self.config.training_stage,
            num_trainable,
        )
        return num_trainable

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load PI0.5/PI05ActionMem weights while initializing new ActionMem modules."""
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            initialized_action_code_keys = fill_missing_initialized_state(
                remapped_state_dict,
                model,
                (
                    "model.action_code_embedding.",
                    "model.action_classifier.",
                    "model.action_condition_proj.",
                ),
            )
            if initialized_action_code_keys:
                print(
                    "Initialized PI05ActionMem effect-tokenizer heads absent from checkpoint: "
                    f"{len(initialized_action_code_keys)} keys"
                )

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # PI0.5 expects time_mlp_* and has no suffix state projection.
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            if key.startswith(("state_proj.", "state_token_proj.")):
                logging.warning(f"Skipping PI0-style state projection key in PI0.5 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self):
        self.configure_training_stage()
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        action_tokens = batch.get(ACTION_TOKENS)
        action_token_masks = batch.get(ACTION_TOKEN_MASK)
        if action_tokens is None or action_token_masks is None:
            raise ValueError(
                f"PI05ActionMem requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the processed batch."
            )
        # Sample actions using the model (pass through RTC kwargs)
        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_tokens,
            action_token_masks,
            **kwargs,
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'none'.")

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        action_tokens = batch.get(ACTION_TOKENS)
        action_token_masks = batch.get(ACTION_TOKEN_MASK)
        if action_tokens is None or action_token_masks is None:
            raise ValueError(
                f"PI05ActionMem requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the processed batch.\n"
                f"Got {batch.keys()}"
            )

        stage = self.config.training_stage
        compute_flow = stage in {"action_expert_only", "joint"}
        compute_action_token = stage in {"vlm_only", "joint"}
        actions = self.prepare_action(batch) if compute_flow else None
        time = self.model.sample_time(actions.shape[0], actions.device) if compute_flow else None

        # Compute loss
        model_output = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            action_tokens,
            action_token_masks,
            actions,
            None,
            time,
            action_token_distances=batch.get(ACTION_TOKEN_DISTANCES),
            compute_flow=compute_flow,
            compute_action_token=compute_action_token,
        )

        scalar_terms: list[Tensor] = []
        per_sample_terms: list[Tensor] = []
        loss_dict: dict[str, float | list[float]] = {}

        if compute_flow:
            # Truncate losses to actual action dimensions.
            original_action_dim = self.config.output_features[ACTION].shape[0]
            flow_losses = model_output["flow_losses"][:, :, :original_action_dim]
            flow_loss, per_sample_flow_loss, loss_per_dim = reduce_flow_losses(
                flow_losses,
                batch.get("action_is_pad"),
            )
            scalar_terms.append(self.config.flow_loss_weight * flow_loss)
            per_sample_terms.append(self.config.flow_loss_weight * per_sample_flow_loss)
            loss_dict.update(
                {
                    "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
                    "flow_loss": flow_loss.item(),
                    "weighted_flow_loss": (self.config.flow_loss_weight * flow_loss).item(),
                    "action_condition_gamma_rms": model_output["action_condition_gamma_rms"].item(),
                    "action_condition_beta_rms": model_output["action_condition_beta_rms"].item(),
                    "action_condition_logit_std": model_output["action_condition_logit_std"].item(),
                    "action_condition_predicted_entropy": model_output[
                        "action_condition_predicted_entropy"
                    ].item(),
                }
            )

        if compute_action_token:
            action_token_kl_loss = model_output["action_token_kl_loss"]
            scalar_terms.append(self.config.action_token_loss_weight * action_token_kl_loss)
            # Normalize masked token losses so their batch mean matches the
            # valid-target mean used by the scalar reduction.
            target_mask = model_output["action_token_target_mask"]
            valid_count = target_mask.sum().clamp(min=1)
            token_loss_scale = target_mask.numel() / valid_count
            per_sample_token_loss = model_output["action_token_loss_per_sample"] * token_loss_scale
            per_sample_terms.append(self.config.action_token_loss_weight * per_sample_token_loss)
            loss_dict.update(
                {
                    "action_token_kl_loss": action_token_kl_loss.item(),
                    "weighted_action_token_kl_loss": (
                        self.config.action_token_loss_weight * action_token_kl_loss
                    ).item(),
                    "action_token_accuracy": model_output["action_token_accuracy"].item(),
                    "action_token_target_rank": model_output["action_token_target_rank"].item(),
                    "action_token_soft_target_entropy": model_output[
                        "action_token_soft_target_entropy"
                    ].item(),
                    "action_token_soft_target_peak_probability": model_output[
                        "action_token_soft_target_peak_probability"
                    ].item(),
                }
            )

        if reduction == "none":
            per_sample_loss = torch.stack(per_sample_terms, dim=0).sum(dim=0)
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss = torch.stack(scalar_terms).sum()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Return stage-aware default PEFT targets for PI05ActionMem fine-tuning."""
        paligemma_targets = (
            r".*\.paligemma_with_expert\."
            r"paligemma\.model\.language_model\.layers\.[0-9]+\.self_attn\.(q|v)_proj"
        )
        vlm_targets = paligemma_targets
        common_projections = "action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
        action_expert_targets = (
            rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        )
        if self.config.training_stage == "vlm_only":
            target_modules = vlm_targets
        elif self.config.training_stage == "action_expert_only":
            target_modules = action_expert_targets
        else:
            target_modules = rf"({vlm_targets}|{action_expert_targets})"
        return {
            "target_modules": target_modules,
            "modules_to_save": [
                "action_code_embedding",
                "action_classifier",
                "action_condition_proj",
            ],
        }
