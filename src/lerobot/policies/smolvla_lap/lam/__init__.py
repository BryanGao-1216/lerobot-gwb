"""Lazy factory for frozen latent-action teachers; keep optional imports local."""


def make_lam_encoder(config, device):
    from .specs import get_lam_spec

    spec = get_lam_spec(config.lam_type)
    if config.lam_num_tokens != spec.num_tokens or config.lam_codebook_size != spec.codebook_size:
        raise ValueError(f"LAM config does not match the token specification for {spec.name}.")
    if spec.name == "univla":
        from .univla import UniVLALAMEncoder

        return UniVLALAMEncoder(config.lam_model_path, device, config.lam_batch_size)
    if spec.name == "lapa":
        from .lapa import LAPALAMEncoder

        return LAPALAMEncoder(config.lam_model_path, device, config.lam_batch_size)
    if spec.name == "mvp_lam":
        from .mvp_lam import MVPLAMEncoder

        return MVPLAMEncoder(config.lam_model_path, device, config.lam_batch_size)
    raise ValueError(f"Unsupported LAM type: {config.lam_type!r}.")
