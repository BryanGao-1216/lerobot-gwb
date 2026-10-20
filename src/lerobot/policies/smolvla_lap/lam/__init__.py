"""Lazy factory for frozen latent-action teachers; keep optional imports local."""


def make_lam_encoder(config, device):
    if config.lam_type == "univla":
        from .univla import UniVLALAMEncoder

        return UniVLALAMEncoder(config.lam_model_path, device, config.lam_batch_size)
    raise ValueError(f"Unsupported LAM type: {config.lam_type!r}.")
