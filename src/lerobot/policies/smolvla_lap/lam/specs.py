"""Teacher metadata shared by configuration, dataset preprocessing and inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LAMSpec:
    name: str
    image_size: int
    num_tokens: int
    codebook_size: int
    latent_dim: int


LAM_SPECS = {
    "univla": LAMSpec("univla", image_size=224, num_tokens=4, codebook_size=16, latent_dim=128),
    "lapa": LAMSpec("lapa", image_size=256, num_tokens=4, codebook_size=8, latent_dim=32),
    "mvp_lam": LAMSpec("mvp_lam", image_size=224, num_tokens=4, codebook_size=16, latent_dim=128),
}


def get_lam_spec(lam_type: str) -> LAMSpec:
    name = lam_type.lower().replace("-", "_")
    if name not in LAM_SPECS:
        raise ValueError(f"Unsupported lam_type={lam_type!r}; choose univla, lapa or mvp_lam (mvp-lam).")
    return LAM_SPECS[name]
