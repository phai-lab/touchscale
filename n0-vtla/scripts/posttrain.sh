#!/usr/bin/env bash
# Post-train N0-VTLA on your own robot data, from either
#   INIT=original    the original N0-VTLA checkpoint, or
#   INIT=touchscale  the original checkpoint merged with the TouchScale mid-training checkpoint
#                    (the tactile branch trained on human data, see docs/MID_TRAIN.md).
#
# Both modes run the same recipe (config vtla_tactile_posttrain through train.sh); they differ only in
# the weights that training starts from. Run from anywhere; paths are relative to the n0-vtla/ directory.
#
# Required:
#   DATASET=<dir>     canonical LeRobot dataset built by scripts/build_robot_dataset.py (or the dual-arm builder)
#   ASSET_ID=<name>   directory name holding the state/action statistics:
#                     assets/vtla_tactile_posttrain/<ASSET_ID>/norm_stats.json
#                     (create it with scripts/compute_canonical_norm.py, see docs/POST_TRAIN.md)
#   INIT=original|touchscale
# Optional:
#   BASE_CKPT=../checkpoints/n0-vtla-base     original N0-VTLA weights
#   EXP_NAME=posttrain_<INIT>                 CONFIG_NAME, NPROC_PER_NODE, CHECK_ONLY ... (see train.sh)
#   For INIT=touchscale (needs a finished mid-training run, see docs/MID_TRAIN.md; train_stage1.sh writes
#   checkpoints/vtla_stage1_predictor_pretrain/<EXP_NAME>/<step>/):
#     MIDTRAIN_EXP=stage1_online   the EXP_NAME used in train_stage1.sh (its default)
#     MIDTRAIN_STEP=<step>         default: the latest step found
#     MIDTRAIN_CKPT=<dir>          explicit step directory, overrides MIDTRAIN_EXP / MIDTRAIN_STEP
#     MERGED_CKPT=<dir>            where base + mid-training delta are merged
#                                  (default ../checkpoints/n0-vtla-base_plus_<MIDTRAIN_EXP>_<step>)
#
# Examples:
#   INIT=original   DATASET=/data/my_robot_train ASSET_ID=my_robot bash scripts/posttrain.sh
#   INIT=touchscale DATASET=/data/my_robot_train ASSET_ID=my_robot bash scripts/posttrain.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INIT="${INIT:-}"
DATASET="${DATASET:-}"
ASSET_ID="${ASSET_ID:-}"
BASE_CKPT="${BASE_CKPT:-../checkpoints/n0-vtla-base}"

if [[ "$INIT" != "original" && "$INIT" != "touchscale" ]]; then
  echo "Set INIT=original or INIT=touchscale (see the header of this script)." >&2; exit 1
fi
[[ -d "$DATASET" ]] || { echo "DATASET must be an existing canonical dataset directory: '$DATASET'" >&2; exit 1; }
[[ -n "$ASSET_ID" ]] || { echo "Set ASSET_ID (directory with norm_stats.json under assets/vtla_tactile_posttrain/)." >&2; exit 1; }
[[ -f "$BASE_CKPT/model.safetensors" ]] || {
  echo "Original checkpoint not found at $BASE_CKPT. Download it first:" >&2
  echo "  hf download NeoteAI/n0-vtla-base --local-dir $BASE_CKPT" >&2; exit 1; }

if [[ "$INIT" == "touchscale" ]]; then
  if [[ -z "${MIDTRAIN_CKPT:-}" ]]; then
    MIDTRAIN_EXP="${MIDTRAIN_EXP:-stage1_online}"
    MIDTRAIN_DIR="checkpoints/vtla_stage1_predictor_pretrain/$MIDTRAIN_EXP"
    MIDTRAIN_STEP="${MIDTRAIN_STEP:-}"
    if [[ -z "$MIDTRAIN_STEP" && -d "$MIDTRAIN_DIR" ]]; then
      MIDTRAIN_STEP="$(ls "$MIDTRAIN_DIR" | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
    fi
    MIDTRAIN_CKPT="$MIDTRAIN_DIR/$MIDTRAIN_STEP"
  else
    MIDTRAIN_STEP="$(basename "$MIDTRAIN_CKPT")"
    MIDTRAIN_EXP="$(basename "$(dirname "$MIDTRAIN_CKPT")")"
  fi
  if [[ -z "$MIDTRAIN_STEP" || ! -f "$MIDTRAIN_CKPT/model.safetensors" ]]; then
    echo "No mid-training checkpoint found at '$MIDTRAIN_CKPT'." >&2
    echo "INIT=touchscale needs a finished mid-training run first (bash train_stage1.sh, docs/MID_TRAIN.md)," >&2
    echo "or point MIDTRAIN_CKPT / MIDTRAIN_EXP / MIDTRAIN_STEP at an existing one." >&2
    exit 1
  fi
  MERGED_CKPT="${MERGED_CKPT:-../checkpoints/n0-vtla-base_plus_${MIDTRAIN_EXP}_${MIDTRAIN_STEP}}"
  if [[ ! -f "$MERGED_CKPT/model.safetensors" ]]; then
    python scripts/merge_stage1_into_base_checkpoint.py \
      --base-checkpoint "$BASE_CKPT" --stage1-checkpoint "$MIDTRAIN_CKPT" --output "$MERGED_CKPT"
  else
    echo "Reusing merged checkpoint $MERGED_CKPT (delete it to re-merge)."
  fi
  export VTLA_PRETRAINED_CHECKPOINT="$MERGED_CKPT"
else
  export VTLA_PRETRAINED_CHECKPOINT="$BASE_CKPT"
fi

export VTLA_DATASET_PATH="$DATASET"
export VTLA_ASSET_ID="$ASSET_ID"
export EXP_NAME="${EXP_NAME:-posttrain_${INIT}}"

exec bash "$REPO_ROOT/train.sh" "$@"
