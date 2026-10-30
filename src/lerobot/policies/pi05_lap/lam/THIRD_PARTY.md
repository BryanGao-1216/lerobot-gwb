# Bundled inference components

Only the UniVLA stage-2 LAM and its DINOv2 dependency are included. No external
UniVLA repository, T5 model, Lightning module, or `torch.hub` download is used.
The LAM checkpoint (including DINOv2 weights) must be supplied separately.

- UniVLA: https://github.com/OpenDriveLab/UniVLA — `univla_model.py` and
  `univla_blocks.py`, under `LICENSE-UniVLA` (Apache-2.0).
- DINOv2: https://github.com/facebookresearch/dinov2 — `dinov2/`, under
  `LICENSE-DINOv2` (Apache-2.0).

Local adaptations use package-relative imports and device-safe frozen inference.
The quantizer's evaluation path does not update codebook usage or restart codes.
