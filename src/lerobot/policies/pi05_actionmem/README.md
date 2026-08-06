# PI0.5 ActionMem

`pi05_actionmem` keeps the native PI0.5 architecture and adds the ActionMem
training and inference protocol:

1. Quantile-normalized state is discretized into the PI0.5 text prompt.
2. Empty action memory and `ACTION_QUERY` are appended after that prompt.
3. The PaliGemma VLM predicts the current q0 action token.
4. The frozen residual VQ-VAE decodes q0 into the initial action chunk.
5. The PI0.5 AdaRMS action expert refines that chunk with flow matching.

The policy supports `vlm_only`, `action_expert_only`, and `joint` training,
LeRobot datasets, and weighted RLDS/OXE mixtures. Its `token_map.json` is
resolved from the policy model directory. The VQ-VAE checkpoint can be set in
that token map or with `--policy.action_vqvae_checkpoint_path`.

Use `type: "pi05_actionmem"` in the model's `config.json`. The configured
`chunk_size` must match `vqvae.action_horizon` in `token_map.json`.
