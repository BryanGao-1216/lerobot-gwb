# Adapted for LeRobot LAP: package-local imports and device-safe frozen inference.
"""UniVLA stage-2 model; no T5, Lightning, torch.hub, or external repository."""
from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from torchvision import transforms
from .univla_blocks import SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer
from .dinov2.vision_transformer import vit_base

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(ControllableDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = vit_base(img_size=518, patch_size=14, init_values=1.0, block_chunks=0, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0)
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches)

        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])

        # Decode
        video_action_patches = torch.cat([self.action_up(outputs["z_q"]),
                                          self.action_up_uncontrol(outputs["z_q_uncontrol"]),
                                          video_patches],
                                          dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2]:]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
