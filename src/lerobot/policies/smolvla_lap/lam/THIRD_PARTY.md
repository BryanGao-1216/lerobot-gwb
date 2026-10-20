# Bundled inference sources

- UniVLA: `https://github.com/OpenDriveLab/UniVLA`, from `latent_action_model/genie/modules/lam.py` and `blocks.py`. Apache-2.0; see `LICENSE-UniVLA`.
- DINOv2: `https://github.com/facebookresearch/dinov2`, from `dinov2/models/vision_transformer.py` and its required `dinov2/layers/` modules. Apache-2.0; see `LICENSE-DINOv2`. Original copyright headers are retained.

These sources were copied from the local UniVLA/DINOv2 checkouts used by the provided `LAM_test/test_target/univla` reference implementation. No external repository or DINOv2 weight download is needed. The full DINOv2 weights are restored from the UniVLA stage-2 checkpoint.

Changes to UniVLA: retain only stage-2 and its required blocks; remove unused T5, multiview, patchification and rotary-attention paths; use package-relative imports and a directly constructed DINOv2 ViT-B/14 with four registers; make positional encodings follow the input device; disable diagnostic VQ usage updates in evaluation. The encoder, normalization, quantization, codebook IDs, and checkpoint parameter names are retained.

Changes to DINOv2: package-relative imports and a minimal layer export list. Inference uses the original architecture, including the 518-pixel positional embedding grid, four registers, and antialiased positional interpolation.
