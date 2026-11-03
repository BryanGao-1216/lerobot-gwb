# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""PI05 input/output transforms matching OpenPI 215abfb217dbac7d5f1273282331b9b1866c0479."""

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    NormalizerProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE

TOKENIZER_FILE = "openpi_tokenizer.model"
TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
IMAGE_ROLES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _verify_tokenizer_model(model: Path) -> Path:
    if hashlib.sha256(model.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError(f"Official PaliGemma tokenizer checksum mismatch: {model}")
    return model


def resolve_tokenizer_model(
    path: str | None = None, checkpoint: str | None = None, *, revision: str | None = None
) -> Path:
    """Prefer the self-contained checkpoint asset, then an explicit path, then the pinned public asset."""
    if checkpoint is not None:
        local = Path(checkpoint) / TOKENIZER_FILE
        if local.is_file():
            return _verify_tokenizer_model(local)
        if not Path(checkpoint).is_dir():
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            try:
                return _verify_tokenizer_model(
                    Path(hf_hub_download(checkpoint, TOKENIZER_FILE, revision=revision))
                )
            except EntryNotFoundError:
                pass  # Native PI05-base has no SentencePiece asset.
    if path is not None:
        model = Path(path).expanduser()
        if not model.is_file():
            raise FileNotFoundError(f"PaliGemma SentencePiece model does not exist: {model}")
        return _verify_tokenizer_model(model)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lerobot/openpi"
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / "paligemma_tokenizer.model"
    if model.exists() and hashlib.sha256(model.read_bytes()).hexdigest() == TOKENIZER_SHA256:
        return model
    # Atomic replacement also works when several distributed workers initialize together.
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
        target = Path(temporary.name)
    try:
        with urllib.request.urlopen(TOKENIZER_URL, timeout=60) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(target.read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("Official PaliGemma tokenizer checksum mismatch.")
        target.replace(model)
    finally:
        target.unlink(missing_ok=True)
    return model


def save_tokenizer_model(config, directory: Path) -> None:
    source = resolve_tokenizer_model(config.tokenizer_model_path)
    destination = directory / TOKENIZER_FILE
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


@ProcessorStepRegistry.register(name="openpi_tokenizer_processor")
@dataclass
class OpenPITokenizerProcessorStep(TokenizerProcessorStep):
    """Use OpenPI's actual SentencePiece encoding, including separately encoded newline."""

    tokenizer_model_path: str | None = None
    discrete_state_input: bool = True

    def __post_init__(self):
        if self.tokenizer is not None:
            self.input_tokenizer = self.tokenizer
        else:
            from lerobot.utils.import_utils import require_package

            require_package("sentencepiece", extra="pi")
            import sentencepiece

            model = resolve_tokenizer_model(self.tokenizer_model_path)
            self.input_tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(model))

    def _tokenize_text(self, text):
        observation = self.transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE)
        if self.discrete_state_input and state is None:
            raise ValueError("OpenPI discrete_state_input requires observation.state.")
        rows, masks = [], []
        for index, prompt in enumerate(text):
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            if self.discrete_state_input:
                values = state[index].detach().cpu().numpy()
                padding = observation.get(f"{OBS_STATE}_dim_is_pad")
                if padding is not None:
                    values = values[~padding[index].cpu().numpy().astype(bool)]
                bins = np.digitize(values, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                full = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
                ids = self.input_tokenizer.encode(full, add_bos=True)
            else:
                ids = self.input_tokenizer.encode(cleaned, add_bos=True)
                ids += self.input_tokenizer.encode("\n")
            if len(ids) > self.max_length:
                logging.warning("OpenPI prompt length %d exceeds %d; truncating.", len(ids), self.max_length)
            valid = min(len(ids), self.max_length)
            rows.append(ids[: self.max_length] + [0] * (self.max_length - valid))
            masks.append([True] * valid + [False] * (self.max_length - valid))
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.bool),
        }

    def get_config(self):
        return {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "discrete_state_input": self.discrete_state_input,
            "tokenizer_model_path": self.tokenizer_model_path,
        }


def make_openpi_processors(config, dataset_stats=None):
    steps = make_default_policy_processor_steps(config, dataset_stats)
    steps.normalize.quantile_epsilon = steps.unnormalize.quantile_epsilon = 1e-6
    steps.normalize.normalize_before_padding = True
    relative = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    tokenizer = OpenPITokenizerProcessorStep(
        tokenizer_model_path=config.tokenizer_model_path,
        max_length=config.tokenizer_max_length,
        discrete_state_input=config.discrete_state_input,
    )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            relative,
            steps.normalize,
            tokenizer,
            steps.to_device,
        ],
        output_steps=[
            steps.unnormalize,
            AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative),
            steps.to_cpu,
        ],
    )


def reconcile_openpi_processors(config, preprocessor, postprocessor, tokenizer):
    """Apply active policy settings and replace the native PI05-base state-to-prompt step."""
    steps = []
    for step in preprocessor.steps:
        if type(step).__name__ == "Pi05PrepareStateTokenizerProcessorStep":
            continue
        if isinstance(step, TokenizerProcessorStep):
            step = tokenizer
        steps.append(step)
    preprocessor.steps = steps
    for pipeline in (preprocessor, postprocessor):
        for step in pipeline.steps:
            if hasattr(step, "quantile_epsilon"):
                step.quantile_epsilon = 1e-6
            if isinstance(step, NormalizerProcessorStep):
                step.normalize_before_padding = True
            if isinstance(step, (RelativeActionsProcessorStep, AbsoluteActionsProcessorStep)):
                step.enabled = config.use_relative_actions
    return preprocessor, postprocessor


def resize_openpi_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """LiberoInputs -> PIL bilinear resize/pad, preserving the original uint8 conversion."""
    device, dtype = image.device, image.dtype
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = (255 * array).astype(np.uint8)
    pil = Image.fromarray(array)
    ratio = max(pil.width / width, pil.height / height)
    resized = pil.resize((int(pil.width / ratio), int(pil.height / ratio)), Image.Resampling.BILINEAR)
    output = Image.new(resized.mode, (width, height), 0)
    output.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    result = torch.from_numpy(np.array(output)).permute(2, 0, 1).to(device=device)
    return result if dtype == torch.uint8 else result.float().div(255)


def prepare_openpi_images(policy, batch):
    """Three ordered OpenPI image slots, then the official PyTorch train/eval transforms."""
    from types import SimpleNamespace

    from .openpi_preprocessing import preprocess_observation_pytorch

    device = next(policy.parameters()).device
    keys = list(policy.config.image_features)
    if len(keys) > 3:
        raise ValueError("OpenPI PI05 supports at most three image slots.")
    anchor = batch.get(OBS_STATE, batch.get("observation.language.tokens"))
    if anchor is None:
        anchor = batch[next(key for key in keys if key in batch)]
    batch_size = len(anchor)
    images, masks = {}, {}
    for index, key in enumerate(keys):
        role = key.removeprefix("observation.images.")
        role = role if role in IMAGE_ROLES else IMAGE_ROLES[index]
        if role in images:
            raise ValueError(f"Duplicate OpenPI camera role: {role}")
        if key not in batch:
            continue
        image = batch[key]
        if image.shape[-1] == 3 and image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        if tuple(image.shape[-2:]) != policy.config.image_resolution:
            image = torch.stack([resize_openpi_image(row, *policy.config.image_resolution) for row in image])
        image = image.to(device=device, dtype=torch.float32)
        if batch[key].dtype == torch.uint8:
            image = image / 255
        else:
            image = (image * 255).to(torch.uint8).float().div(255)
        # Observation.from_dict converts contiguous BHWC pixels to BCHW without
        # copying. Match that layout too: CPU interpolate/reductions have different
        # floating-point operation orders for contiguous vs channels-last tensors.
        images[role] = (image * 2 - 1).contiguous(memory_format=torch.channels_last)
        padding = batch.get(f"{key}_is_pad", torch.zeros(batch_size, dtype=torch.bool))
        masks[role] = ~padding.to(device=device, dtype=torch.bool).reshape(batch_size)
    for role in IMAGE_ROLES:
        if role not in images:
            images[role] = torch.full((batch_size, 3, *policy.config.image_resolution), -1.0, device=device)
            masks[role] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    observation = SimpleNamespace(
        images={key: images[key] for key in IMAGE_ROLES},
        image_masks=masks,
        state=torch.empty(batch_size, 32, device=device),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    output = preprocess_observation_pytorch(
        observation,
        train=policy.training and policy.config.image_augmentation,
        image_resolution=policy.config.image_resolution,
    )
    return list(output.images.values()), list(output.image_masks.values())
