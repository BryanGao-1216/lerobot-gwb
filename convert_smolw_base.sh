#!/usr/bin/env bash

set -euo pipefail

# All model and checkpoint locations are launch-time choices. Override any of
# these variables from the environment when running on another machine.
SOURCE_SMOLVLA_DIR="${SOURCE_SMOLVLA_DIR:-/data1/gaowenbing/WorkSpace/models/smolvla}"
SMOLW_MODEL_DIR="${SMOLW_MODEL_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-base}"
VIDTWIN_CHECKPOINT_PATH="${VIDTWIN_CHECKPOINT_PATH:-/data1/gaowenbing/WorkSpace/models/vidtwin-libero/checkpoint-best.ckpt}"
HORIZON="${HORIZON:-16}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"
MOTION_CAMERA_KEY="${MOTION_CAMERA_KEY:-}"

CONVERT_ARGS=(
  --source "${SOURCE_SMOLVLA_DIR}"
  --output-dir "${SMOLW_MODEL_DIR}"
  --vidtwin-checkpoint-path "${VIDTWIN_CHECKPOINT_PATH}"
  --motion-horizon "${HORIZON}"
  --memory-stride "${MEMORY_STRIDE}"
)

# During conversion the available image keys come from the source SmolVLA
# artifact, not from the later training dataset. Leaving this unset keeps the
# saved SmolW base camera-agnostic and selects the source model's first camera
# only for construction. Training may override it with
# --policy.motion_camera_key for the actual LeRobot dataset.
if [[ -n "${MOTION_CAMERA_KEY}" ]]; then
  CONVERT_ARGS+=(--motion-camera-key "${MOTION_CAMERA_KEY}")
fi

python -m lerobot.policies.smolw.convert_smolvla_checkpoint "${CONVERT_ARGS[@]}"
