# Train ActionMem from multiple RLDS datasets

The RLDS path is optional and isolated from the normal LeRobot dataset path. Select it with
`--dataset_type=rlds`; omitting that argument keeps the original `LeRobotDataset` behavior.

## Data dependencies

The VQ-VLA RLDS/OXE implementation is vendored under `lerobot.datasets.rlds`; no external VQ-VLA
checkout or `PYTHONPATH` entry is needed. Install LeRobot's optional RLDS dependencies from this
checkout:

For Python 3.12, use a TensorFlow release that supports Python 3.12, plus:

```bash
pip install -e '.[training,rlds]'
pip install --no-deps 'dlimp @ git+https://github.com/moojink/dlimp_openvla.git'
```

`--no-deps` is required because dlimp's package metadata pins TensorFlow 2.15, which conflicts with
LeRobot's Python 3.12+ and NumPy 2 environment; the `rlds` extra installs a compatible TensorFlow.
If PyPI is slow, append `-i https://pypi.tuna.tsinghua.edu.cn/simple` to the first command. The Git
dependency still comes from GitHub.

## Mixture

Pass either a VQ-VLA named OXE mixture via `--dataset.rlds_data_mix`, or a JSON file via
`--dataset.rlds_mixture_path`. The JSON format is shown in
[`actionmem_rlds_mix.json`](actionmem_rlds_mix.json). Source weights are positive relative weights.
With `rlds_balance_weights=true` (the default), they are multiplied by each dataset's usable frame
count, matching VQ-VLA/OpenVLA size-balanced mixing. Set it to false to use the JSON weights as the
final sampling proportions.

## Example command

```bash
lerobot-train \
  --dataset_type=rlds \
  --dataset.repo_id=actionmem_research_mix \
  --dataset.root=/data/open_x_embodiment \
  --dataset.rlds_mixture_path=/path/to/lerobot-gwb/examples/training/actionmem_rlds_mix.json \
  --dataset.rlds_action_vqvae_checkpoint_path=/path/to/action_vqvae.pt \
  --dataset.rlds_q0_device=cpu \
  --dataset.rlds_state_dim=32 \
  --dataset.rlds_camera_views='["primary","secondary","wrist"]' \
  --policy.path=/path/to/smol-actionmem-base \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --output_dir=/path/to/output \
  --batch_size=8 \
  --steps=100000 \
  --eval_steps=0 \
  --num_workers=0 \
  --policy.push_to_hub=false
```

The adapter uses the vendored OXE registry and trajectory transforms, then emits the same
keys that the existing ActionMem/SmolActionMem preprocessors consume. It forms complete 16-step
chunks, removes each episode's padded tail, and encodes q0 online in the collate function. The
VQ-VAE is frozen and is not part of the policy optimizer or checkpoint.

Camera keys are unified by OXE (missing views become masked padding), and proprio vectors are padded
to `rlds_state_dim` before interleaving. Padding aligns tensor shapes; it does not change the native
OXE state semantics (for example, quaternion versus Euler state encodings). If an experiment requires
one shared semantic state representation, add a dataset-specific state conversion before mixing.

`rlds_action_transform=actionmem` is intentionally restricted to DROID, RLBench, and LIBERO because
those are the action conventions currently aligned to the trained VQ-VAE. A new dataset needs an
explicit 7-D action conversion before adding it to the mixture. `identity` is available as an escape
hatch only when the dataset has already been verified to use exactly the VQ-VAE's action convention.

RLDS training is an infinite stream and currently has no LeRobot episode holdout. Keep
`dataset.eval_split=0` and `eval_steps=0`. Checkpoint saving/resume, TensorBoard/W&B, PEFT, and the
policy optimizer/scheduler continue to use the normal `lerobot-train` path.
