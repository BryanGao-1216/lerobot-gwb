#!/usr/bin/env bash

set -euo pipefail

# All model and VidTwin locations are launch-time choices. Override any of
# these variables from the environment when running on another machine.
SMOLW_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SMOLVLA_DIR="${SOURCE_SMOLVLA_DIR:-/data1/gaowenbing/WorkSpace/models/smolvla}"
SMOLW_MODEL_DIR="${SMOLW_MODEL_DIR:-/data1/gaowenbing/WorkSpace/models/smolw-base}"
VIDTWIN_REPO_DIR="${VIDTWIN_REPO_DIR:-${SMOLW_SCRIPT_DIR}/../CoWVLA}"
VIDTWIN_CONFIG_PATH="${VIDTWIN_CONFIG_PATH:-${VIDTWIN_REPO_DIR}/vidtwin/configs/vidtwin_structure_7_7_8_dynamics_7_8.yaml}"
VIDTWIN_CHECKPOINT_PATH="${VIDTWIN_CHECKPOINT_PATH:-/mnt/models/microsoft__vidtwin/main/checkpoints/vidtwin_structure_7_7_8_dynamics_7_8.ckpt}"
HORIZON="${HORIZON:-20}"
MEMORY_STRIDE="${MEMORY_STRIDE:-1}"

MOTION_CAMERA_ARGS=()
if [[ -n "${MOTION_CAMERA_KEY:-}" ]]; then
  MOTION_CAMERA_ARGS+=(--motion-camera-key "${MOTION_CAMERA_KEY}")
fi

python -m lerobot.policies.smolw.convert_smolvla_checkpoint \
  --source "${SOURCE_SMOLVLA_DIR}" \
  --output-dir "${SMOLW_MODEL_DIR}" \
  --vidtwin-repo-path "${VIDTWIN_REPO_DIR}" \
  --vidtwin-config-path "${VIDTWIN_CONFIG_PATH}" \
  --vidtwin-checkpoint-path "${VIDTWIN_CHECKPOINT_PATH}" \
  --motion-horizon "${HORIZON}" \
  --memory-stride "${MEMORY_STRIDE}" \
  "${MOTION_CAMERA_ARGS[@]}"
