#!/usr/bin/env bash
set -euo pipefail

# Stage-1 predictor-grounding pretraining launcher (paper Sec 4.2, action-free) --
# sibling to train.sh (post-train). Mirrors train.sh's structure exactly, but points
# at the ONLINE loader (scripts/train_stage1_online.py), which reads raw itw episode
# directories directly (no LeRobot materialization step, no norm_stats.json, no
# cfg.data.repo_id) -- see scripts/train_stage1_online.py's own docstring for why
# config.data is unused on this path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONFIG_NAME="${CONFIG_NAME:-vtla_stage1_predictor_pretrain}"
EXP_NAME="${EXP_NAME:-stage1_online}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')"
  [[ -z "$NPROC_PER_NODE" || "$NPROC_PER_NODE" == "0" ]] && NPROC_PER_NODE=8
fi
CHECK_ONLY="${CHECK_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash train_stage1.sh [additional training arguments]

Environment variables:
  CONFIG_NAME=vtla_stage1_predictor_pretrain
  EXP_NAME=stage1_online
  NPROC_PER_NODE=<auto-detected via nvidia-smi, falls back to 8>
  CHECK_ONLY=0
  VTLA_ITW_RAW_ROOT=/path/to/raw_itw_root          (required)
  VTLA_PRETRAINED_CHECKPOINT=/path/to/checkpoint    (required)
  VTLA_ITW_NORMALIZATION=/path/to/per_task_scale.json  (required; build it with scripts/build_per_task_scale_normalization.py)
  VTLA_ITW_DATES, VTLA_ITW_MAX_EPISODES, VTLA_STAGE1_FUTURE_OFFSET, VTLA_DEFAULT_PROMPT,
  VTLA_STAGE1_RECON_GRID, VTLA_STAGE1_LAMBDA_REC, VTLA_STAGE1_TEMPERATURE  (all optional)

Examples:
  CHECK_ONLY=1 bash train_stage1.sh
  VTLA_ITW_RAW_ROOT=/data/itw_raw VTLA_PRETRAINED_CHECKPOINT=checkpoints/n0-vtla-base bash train_stage1.sh
  EXP_NAME=my_experiment bash train_stage1.sh --resume
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
export VTLA_ATTN_IMPL="${VTLA_ATTN_IMPL:-eager}"
export VTLA_PREFIX_CACHE="${VTLA_PREFIX_CACHE:-1}"
if [[ -z "${VTLA_ITW_NORMALIZATION:-}" || ! -f "${VTLA_ITW_NORMALIZATION}" ]]; then
  echo "VTLA_ITW_NORMALIZATION must point to an existing normalization JSON." >&2
  echo "Build one from your episodes: python scripts/build_per_task_scale_normalization.py --raw-root <itw root> --out <file>" >&2
  exit 1
fi
export VTLA_ITW_NORMALIZATION

mkdir -p logs
LOG_FILE="logs/${EXP_NAME}.log"

echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "TORCHRUN_BIN=$TORCHRUN_BIN"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "VTLA_ITW_RAW_ROOT=${VTLA_ITW_RAW_ROOT:-<unset>}"
echo "VTLA_ITW_NORMALIZATION=$VTLA_ITW_NORMALIZATION"
echo "VTLA_PRETRAINED_CHECKPOINT=${VTLA_PRETRAINED_CHECKPOINT:-<unset>}"
echo "LOG_FILE=$REPO_ROOT/$LOG_FILE"

"$PYTHON_BIN" - "$CONFIG_NAME" "$NPROC_PER_NODE" <<'PY'
import os
import pathlib
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, "scripts")
from itw_pressure import load_normalization  # noqa: E402

import n0vtla.training.itw_online_dataset as _online  # noqa: E402
from n0vtla.training import config as config_module

config_name = sys.argv[1]
required_gpus = int(sys.argv[2])
cfg = config_module.get_config(config_name)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this container")
visible_gpus = torch.cuda.device_count()
if visible_gpus < required_gpus:
    raise SystemExit(f"only {visible_gpus} GPUs are visible; NPROC_PER_NODE={required_gpus}")

raw_root = os.environ.get("VTLA_ITW_RAW_ROOT")
if not raw_root or not pathlib.Path(raw_root).is_dir():
    raise SystemExit(f"missing/invalid VTLA_ITW_RAW_ROOT: {raw_root!r}")
dates_env = os.environ.get("VTLA_ITW_DATES")
episode_dirs = _online.list_episode_dirs(raw_root, date_dirs=dates_env.split(",") if dates_env else None)
if not episode_dirs:
    raise SystemExit(f"VTLA_ITW_RAW_ROOT has no recognizable episode dirs: {raw_root}")

norm_path = os.environ.get("VTLA_ITW_NORMALIZATION")
if not norm_path or not pathlib.Path(norm_path).is_file():
    raise SystemExit(f"missing VTLA_ITW_NORMALIZATION: {norm_path!r}")
load_normalization(pathlib.Path(norm_path))  # raises on malformed JSON

checkpoint = pathlib.Path(cfg.pytorch_weight_path or "") / "model.safetensors"
if not checkpoint.exists():
    raise SystemExit(f"missing pretrained checkpoint: {checkpoint}")

AutoConfig.from_pretrained("facebook/dinov2-base", local_files_only=True)

print(f"environment OK: torch={torch.__version__}, GPUs={visible_gpus}")
print(f"config OK: {cfg.name}")
print(f"raw episodes OK: found >=1 under {raw_root}")
print(f"normalization OK: {norm_path}")
print(f"checkpoint OK: {checkpoint}")
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
  scripts/train_stage1_online.py \
  "$CONFIG_NAME" \
  --exp-name="$EXP_NAME" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
echo "$(date '+%F %T') training completed"
