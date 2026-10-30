"""The latent vocabulary is saved with each policy, without modifying base tokenizer files."""

import json
from pathlib import Path

import torch
from torch import nn

TOKENIZER_DIR = "latent_tokenizer"
TOKEN_MAP_FILE = "latent_action_tokens.json"
LATENT_START = "<latent_action_start>"
LATENT_CODES = tuple(f"<latent_action_{i}>" for i in range(16))
NUM_LATENT_TOKENS = 4


class LatentActionVocabulary:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        symbols = (LATENT_START, *LATENT_CODES)
        ids = tokenizer.convert_tokens_to_ids(list(symbols))
        if len(set(ids)) != len(symbols) or any(
            tokenizer.encode(s, add_special_tokens=False) != [i] for s, i in zip(symbols, ids, strict=True)
        ):
            raise ValueError("Latent action symbols must each map to a distinct, single tokenizer ID.")
        self.start_id = ids[0]
        self.code_ids = tuple(ids[1:])

    @classmethod
    def create(cls, tokenizer_path: str, **hub_args):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **hub_args)
        tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_START, *LATENT_CODES]})
        return cls(tokenizer)

    def metadata(self) -> dict:
        return {
            "num_tokens": NUM_LATENT_TOKENS,
            "vocab_size": len(self.tokenizer),
            "start_id": self.start_id,
            "code_ids": list(self.code_ids),
        }

    def save_pretrained(self, directory: Path):
        self.tokenizer.save_pretrained(directory / TOKENIZER_DIR)
        (directory / TOKEN_MAP_FILE).write_text(json.dumps(self.metadata(), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, source, **hub_args):
        from transformers import AutoTokenizer

        from .checkpoint import checkpoint_file

        tokenizer = AutoTokenizer.from_pretrained(str(source), subfolder=TOKENIZER_DIR, **hub_args)
        vocab = cls(tokenizer)
        saved = json.loads(checkpoint_file(source, TOKEN_MAP_FILE, **hub_args).read_text())
        if saved != vocab.metadata():
            raise ValueError("Saved latent token IDs do not match the checkpoint's tokenizer.")
        return vocab

    def targets(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dtype != torch.long or codes.ndim != 2 or codes.shape[1] != NUM_LATENT_TOKENS:
            raise ValueError("UniVLA labels must be a LongTensor [B,4].")
        if (codes < 0).any() or (codes >= len(self.code_ids)).any():
            raise ValueError("UniVLA codebook indices must be in [0,15].")
        return torch.tensor(self.code_ids, device=codes.device, dtype=torch.long)[codes]


def extend_vlm_vocabulary(vlm, size: int):
    """Resize input/output separately, preserving PI05's original weights and tying semantics.

    Avoid HF's mean-resizing and implicit re-tying, which can overwrite an independently
    trained output head in a LAP checkpoint.
    """
    embedding = vlm.get_input_embeddings()
    head = vlm.get_output_embeddings()
    size = max(size, embedding.num_embeddings)
    if size == embedding.num_embeddings:
        return
    tied = head.weight is embedding.weight
    new_embedding = nn.Embedding(
        size,
        embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    new_head = nn.Linear(
        head.in_features,
        size,
        bias=head.bias is not None,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    std = getattr(vlm.config.text_config, "initializer_range", 0.02)
    with torch.no_grad():
        nn.init.normal_(new_embedding.weight, std=std)
        new_embedding.weight[: embedding.num_embeddings].copy_(embedding.weight)
        nn.init.normal_(new_head.weight, std=std)
        new_head.weight[: head.out_features].copy_(head.weight)
        if head.bias is not None:
            new_head.bias.zero_()
            new_head.bias[: head.out_features].copy_(head.bias)
    if tied:
        new_head.weight = new_embedding.weight
    vlm.set_input_embeddings(new_embedding)
    vlm.set_output_embeddings(new_head)
    vlm.config.text_config.vocab_size = size
    vlm.config._vocab_size = size
    vlm.vocab_size = size
