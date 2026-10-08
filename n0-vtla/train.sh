#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONFIG_NAME="${CONFIG_NAME:-vtla_tactile_posttrain}"
EXP_NAME="${EXP_NAME:-tactile_posttrain}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
  [[ -z "$NPROC_PER_NODE" || "$NPROC_PER_NODE" == "0" ]] && NPROC_PER_NODE=8
fi
CHECK_ONLY="${CHECK_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash train.sh [additional training arguments]

Environment variables:
  CONFIG_NAME=vtla_tactile_posttrain
  EXP_NAME=tactile_posttrain
  NPROC_PER_NODE=<auto-detected via nvidia-smi, falls back to 8>
  CHECK_ONLY=0
  VTLA_DATASET_PATH=/path/to/dataset
  VTLA_PRETRAINED_CHECKPOINT=/path/to/checkpoint

Examples:
  CHECK_ONLY=1 bash train.sh
  bash train.sh
  EXP_NAME=my_experiment bash train.sh
  EXP_NAME=my_experiment bash train.sh --resume
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

EXTRA_TRAIN_ARGS=("$@")

cd "$REPO_ROOT"

PYTHON_BIN="$(command -v python)"
TORCHRUN_BIN="$(command -v torchrun)"
if [[ -z "$PYTHON_BIN" || -z "$TORCHRUN_BIN" ]]; then
  echo "python or torchrun is unavailable in the active environment" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export N0VTLA_DATA_HOME="${N0VTLA_DATA_HOME:-$REPO_ROOT/models}"
export HF_HOME="${HF_HOME:-$N0VTLA_DATA_HOME/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VTLA_ATTN_IMPL="${VTLA_ATTN_IMPL:-eager}"  # Match the attention implementation used during pretraining.
export VTLA_PREFIX_CACHE="${VTLA_PREFIX_CACHE:-1}"
export VTLA_PREDICTOR_LR_SCALE="${VTLA_PREDICTOR_LR_SCALE:-0.1}"

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}.log"

echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TORCHRUN_BIN=$TORCHRUN_BIN"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "VTLA_ATTN_IMPL=$VTLA_ATTN_IMPL"
echo "VTLA_PREFIX_CACHE=$VTLA_PREFIX_CACHE"
echo "VTLA_PREDICTOR_LR_SCALE=$VTLA_PREDICTOR_LR_SCALE"
echo "LOG_FILE=$REPO_ROOT/$LOG_FILE"

"$PYTHON_BIN" - "$CONFIG_NAME" "$NPROC_PER_NODE" <<'PY'
import pathlib
import sys

import torch
from transformers import AutoConfig

from n0vtla.training import config as config_module

config_name = sys.argv[1]
required_gpus = int(sys.argv[2])
cfg = config_module.get_config(config_name)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this container")
visible_gpus = torch.cuda.device_count()
if visible_gpus < required_gpus:
    raise SystemExit(f"only {visible_gpus} GPUs are visible; NPROC_PER_NODE={required_gpus}")

checkpoint = pathlib.Path(cfg.pytorch_weight_path or "") / "model.safetensors"
dataset = pathlib.Path(cfg.data.repo_id)
norm = pathlib.Path(cfg.assets_base_dir) / cfg.name / cfg.data.assets.asset_id / "norm_stats.json"
for label, path in (("pretrained checkpoint", checkpoint), ("dataset", dataset), ("norm stats", norm)):
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")

AutoConfig.from_pretrained("facebook/dinov2-base", local_files_only=True)

print(f"environment OK: torch={torch.__version__}, GPUs={visible_gpus}")
print(f"config OK: {cfg.name}")
print(f"checkpoint OK: {checkpoint}")
print(f"dataset OK: {dataset}")
print(f"norm OK: {norm}")
print("DINOv2 cache OK")
PY

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "CHECK_ONLY=1; all checks passed."
  exit 0
fi

echo "$(date '+%F %T') starting training"
"$TORCHRUN_BIN" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train_n0vtla.py \
  "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
echo "$(date '+%F %T') training completed"
