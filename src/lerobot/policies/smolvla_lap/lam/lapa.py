"""Frozen official Open-X LAPA LAQ teacher, without its training dependency stack."""

import torch

from .common import FrozenLAMEncoder

LAPA_ARCHITECTURE = {
    "dim": 1024,
    "quant_dim": 32,
    "codebook_size": 8,
    "image_size": 256,
    "patch_size": 32,
    "spatial_depth": 8,
    "temporal_depth": 8,
    "dim_head": 64,
    "heads": 16,
    "code_seq_len": 4,
}


class LAPALAMEncoder(FrozenLAMEncoder):
    lam_type = "lapa"
    codebook_key = "vq.codebooks"

    def _build_model(self) -> torch.nn.Module:
        from .lapa_model import LatentActionQuantization

        return LatentActionQuantization(**LAPA_ARCHITECTURE)

    def _after_load(self) -> None:
        # Upstream keeps these outside parameters/buffers; inference must follow
        # the current rank's CUDA device after strict CPU checkpoint loading.
        self.model.vq.device = self.device
        self.model.vq.codebooks_used = self.model.vq.codebooks_used.to(self.device)

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        video = frames.permute(0, 2, 1, 3, 4).contiguous()
        return self.model.inference(video, return_only_codebook_ids=True).long()
