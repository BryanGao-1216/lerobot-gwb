# Bundled inference sources

- UniVLA: `https://github.com/OpenDriveLab/UniVLA`, from `latent_action_model/genie/modules/lam.py` and `blocks.py`. Apache-2.0; see `LICENSE-UniVLA`.
- LAPA: from the supplied `vendor/LAPA/laq/laq_model/{latent_action_quantization,attention,nsvq}.py`. MIT; see `LICENSE-LAPA`, including the original Seonghyeon Ye / Joel Jang copyright notice. NSVQ attribution is retained in `lapa_nsvq.py`.
- MVP-LAM: from the supplied `vendor/mvp_lam/latent_action_model/genie/modules/{lam,blocks}.py`; see `NOTICE-MVP-LAM` for attribution and the license metadata present in that checkout.
- DINOv2: `https://github.com/facebookresearch/dinov2`, from `dinov2/models/vision_transformer.py` and its required `dinov2/layers/` modules. Apache-2.0; see `LICENSE-DINOv2`. Original copyright headers are retained.

These sources were copied from the local checkouts used by the provided `LAM_test/test_target/{univla,lapa,mvp_lam}` reference implementations. No external repository or DINOv2 weight download is needed. The full DINOv2 weights are restored from the selected UniVLA stage-2 or MVP-LAM checkpoint.

Changes to UniVLA: retain only stage-2 and its required blocks; remove unused T5, multiview, patchification and rotary-attention paths; use package-relative imports and a directly constructed DINOv2 ViT-B/14 with four registers; make positional encodings follow the input device; disable diagnostic VQ usage updates in evaluation. The encoder, normalization, quantization, codebook IDs, and checkpoint parameter names are retained.

Changes to DINOv2: package-relative imports and a minimal layer export list. Inference uses the original architecture, including the 518-pixel positional embedding grid, four registers, and antialiased positional interpolation.

Changes to LAPA: retain the complete inference architecture (including decoder parameters for strict checkpoint loading) and the original `inference`, encoder, attention and nearest-code mathematics; remove training methods, permissive load overrides and optional beartype decorators; use package-relative imports; construct NSVQ on the construction context's device instead of hard-coding CUDA. The adapter moves its non-buffer device/counter fields after loading. The official Open-X architecture has dim=1024, quant_dim=32, codebook_size=8, image_size=256, patch_size=32, spatial_depth=8, temporal_depth=8, dim_head=64, heads=16 and code_seq_len=4.

Changes to MVP-LAM: retain its own encoder/VQ and transformer blocks, including the upstream attention-return conventions; omit unused multiview reconstruction methods, patchification and rotary-attention paths; use package-relative imports and the bundled DINOv2; make positional encodings follow the input device; disable diagnostic VQ usage updates in evaluation. Its checkpoint parameter names, encoder mathematics and code IDs are retained.
