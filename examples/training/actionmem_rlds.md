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

## Local OpenX tar shards

The `jxu124/OpenX-Embodiment` WebDataset release can be read directly without extraction. Keep the
downloaded shards in their original layout:

```text
/data/OpenX/
├── austin_buds_dataset_converted_externally_to_rlds/
│   └── austin_buds_dataset_converted_externally_to_rlds_00000.tar
├── bc_z/
│   ├── bc_z_00000.tar
│   └── ...
└── cmu_stretch/
    └── cmu_stretch_00000.tar
```

Set `--dataset.root=/data/OpenX`. Both `rlds_storage_format=auto` and `hybrid` select WebDataset for
a source when `root/<dataset_name>/*.tar` exists and otherwise use prepared TFDS/RLDS. When one
mixture contains both kinds, LeRobot preserves the configured per-dataset weights while sampling
the tar and TFDS streams together. Set `--dataset.rlds_storage_format=webdataset` to require tar
shards for every source, or `tfds` to disable tar auto-detection.
Only use trusted local tar files: the release stores episodes as Python pickle objects.
Tar episodes are standardized with the same OXE functions as the TFDS path, mixed using the
configured weights, and shuffled before image decoding.
An all-tar stream fills `rlds_shuffle_buffer_size` compressed frames (default 100,000) before its
first tar sample. In hybrid mode, the tar and TFDS shuffle capacities are divided according to their
total sampling weights; reduce this setting for a quick smoke test.

Tar shards do not include the TFDS statistics consumed by VQ-VLA. On the first run, LeRobot scans
each selected source and writes a `dataset_statistics_<hash>.json` cache beside its tar files. This
can take a while for large sources. Later runs with the same shards and OXE transform load the cache
directly. Do not interrupt or delete these cache files after training starts, because q01/q99 must
stay fixed across runs.

## Example command

```bash
lerobot-train \
  --dataset_type=rlds \
  --dataset.repo_id=actionmem_research_mix \
  --dataset.root=/data/open_x_embodiment \
  --rlds-storage-format=hybrid \
  --dataset.rlds_mixture_path=/path/to/lerobot-gwb/examples/training/actionmem_rlds_mix.json \
  --dataset.rlds_effect_tokenizer_checkpoint_path=/path/to/effect_vqvae.pt \
  --dataset.rlds_action_tokenizer_device=cpu \
  --dataset.rlds_target_control_hz=10 \
  --dataset.rlds_state_dim=32 \
  --dataset.rlds_camera_views='["primary","secondary","wrist"]' \
  --policy.path=/path/to/smol-actionmem-base \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --output_dir=/path/to/output \
  --batch_size=8 \
  --steps=100000 \
  --eval_steps=0 \
  --num_workers=0 \
  --policy.push_to_hub=false
```

For a fixed 100-sample overfit check, add:

```bash
--dataset.rlds_overfit_num_samples=100
```

This debugging option caches the first fixed set of fully preprocessed samples and
reshuffles/repeats only that cache, so `epch` advances once per 100 global samples. Remove the
option for normal infinite-mixture training.

The adapter uses the vendored OXE registry and trajectory transforms, then emits the same
keys that the ActionMem preprocessors consume. It forms complete chunks at the horizon stored in
the effectTokenizer checkpoint, removes each episode's padded tail, and encodes endpoint-effect
codes plus prototype distances online in the collate function. The effectTokenizer is frozen and
is not part of the policy optimizer or policy checkpoint.

Action preprocessing follows VQ-VLA exactly: each source first uses its registered OXE
standardizer, the non-gripper EEF dimensions are normalized with that source's q01/q99 statistics,
and dimensions excluded by OXE's action normalization mask (the gripper for EEF actions) pass
through unchanged. No legacy DROID/RLBench target-pose delta or LIBERO `0/1 -> -1/+1` conversion is
applied. `rlds_action_transform=oxe` is the default; `identity` remains only as a backwards-compatible
name for the same behavior.

Camera keys are unified by OXE (missing views become masked padding), and proprio vectors are padded
to `rlds_state_dim` before interleaving. Padding aligns tensor shapes; it does not change the native
OXE state semantics (for example, quaternion versus Euler state encodings). If an experiment requires
one shared semantic state representation, add a dataset-specific state conversion before mixing.

RLDS training is an infinite stream and currently has no LeRobot episode holdout. Keep
`dataset.eval_split=0` and `eval_steps=0`. Checkpoint saving/resume, TensorBoard/W&B, PEFT, and the
policy optimizer/scheduler continue to use the normal `lerobot-train` path.
