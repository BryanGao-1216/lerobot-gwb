# Vendored VidTwin runtime

These files were copied from `scripts/CoWVLA/vidtwin` so SmolW does not import
or execute source code from the external CoWVLA checkout at runtime.

Copied runtime files:

- `models/autoencoder.py`
- `models/vidtwin_ae.py`
- `modules/distributions.py`
- `modules/ema.py`
- `modules/qformer.py`
- `modules/regularizers.py`
- `modules/st_transformer.py`
- `modules/util.py`
- `configs/vidtwin_structure_7_7_8_dynamics_7_8.yaml`

Only package-relative imports, inference-only dependency cleanup, and a local
copy of the Q-Former head-pruning helper removed in Transformers 5 were made.
The latent extraction wrapper remains in the parent `vidtwin_motion_encoder.py`.
