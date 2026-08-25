# PI0.5 ActionMem

`pi05_actionmem` keeps the native PI0.5 architecture and uses the same
endpoint-effect ActionMem protocol as `smol_actionmem`:

1. Quantile-normalized state is discretized into the PI0.5 text prompt.
2. Empty action memory and a model-local `ACTION_QUERY` are appended after it.
3. A separate 256-way head predicts the frozen effectTokenizer code
   distribution with a prototype-distance KL objective.
4. Detached code logits condition the PI0.5 AdaRMS action expert through a
   bounded FiLM adapter; the VLM prefix/KV path remains intact.
5. Flow matching starts from Gaussian noise and predicts the final action chunk.

The policy supports `vlm_only`, `action_expert_only`, and `joint` training,
LeRobot datasets, and weighted RLDS/OXE mixtures. RLDS training requires the
same artifact-version-3 effectTokenizer checkpoint used by `smol_actionmem`;
set it with `--policy.effect_tokenizer_checkpoint_path` or
`--dataset.rlds_effect_tokenizer_checkpoint_path`.

Use `type: "pi05_actionmem"` in the model's `config.json`. `chunk_size`, action
dimension, codebook size, and target control rate must match the checkpoint.
