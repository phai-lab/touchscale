#!/usr/bin/env python
"""Materializes the "no-tactile" control arm's Stage-1 output -- NOT a training script.

Context: a "no-tactile" control arm runs alongside the real human-data Stage-1
(`vtla_stage1_predictor_pretrain` / train_stage1_online.py) to isolate whether tactile
*signal* itself (vs. just more training compute / more human RGB exposure) drives any
downstream gain. That control arm should never see real tactile input -- the model
supports this natively: `N0VTLAPolicy._build_g` returns `g=None` whenever a sample's
tactile dict has no keys, and the predictor substitutes a learned null token for `g` in
that case (as in the paper: "a learned null token stands in for g").

Stage 1's own loss, however, has NO well-defined objective when tactile is *always*
absent: `_forward_stage1` requires a real `.future` tactile frame and raises
`RuntimeError` if `_build_future_target` returns `z_star=None`, and the training loop
in both train_stage1_predictor.py/train_stage1_online.py aborts after a full epoch of
"empty" (invalid) batches by design (`RuntimeError("A full epoch had no valid future
tactile targets...")`) -- that guard exists to catch a broken data path and should not be
bypassed. There is no principled gradient for "predict a future tactile change that never
exists," so there is nothing for a no-tactile Stage 1 to train.

The correct null hypothesis is therefore zero perturbation: this arm's "Stage-1 output"
IS `n0-vtla-base`'s tactile pathway, untouched. This script only materializes that as a
checkpoint directory in the exact `trainable_only_v1` format `save_stage1_checkpoint`
produces (see train_stage1_predictor.py), so downstream tooling can point at
`checkpoints/vtla_stage1_predictor_pretrain/<no-tactile exp name>/0/` exactly the way it
points at the real arm's checkpoints, without special-casing "this arm skipped Stage 1."
The actual null-tactile intervention belongs at Stage 2 (never loading real tactile
arrays for this arm's data loader is sufficient -- `_build_g`'s empty-keys path already
falls back to the null token with no model-code changes needed there).

Usage (CPU only, no GPU/DDP needed -- this never runs the model, just slices a
safetensors file):
    python scripts/materialize_no_tactile_stage1_checkpoint.py --exp-name=no_tactile_control

Env vars (same convention as train_stage1_predictor.py):
    VTLA_PRETRAINED_CHECKPOINT   required. Directory containing n0-vtla-base's model.safetensors.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stage1_predictor import STAGE1_TRAINABLE_PREFIXES  # noqa: E402

import n0vtla.training.config as _config  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True, help="No-tactile arm's experiment name, "
                         "e.g. no_tactile_control -- kept under the SAME config name "
                         "(vtla_stage1_predictor_pretrain) as the real arm so directory layout "
                         "stays uniform for downstream tooling.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_ckpt = os.environ.get("VTLA_PRETRAINED_CHECKPOINT")
    if not base_ckpt:
        raise ValueError("VTLA_PRETRAINED_CHECKPOINT is required (n0-vtla-base checkpoint dir)")
    base_path = Path(base_ckpt)
    weights_file = base_path / "model.safetensors" if base_path.is_dir() else base_path
    if not weights_file.is_file():
        raise FileNotFoundError(f"Base checkpoint missing: {weights_file}")

    base_config = _config.get_config("vtla_stage1_predictor_pretrain")
    checkpoint_dir = (Path(base_config.checkpoint_base_dir) / base_config.name / args.exp_name).resolve()
    final_dir = checkpoint_dir / "0"
    if final_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{final_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(final_dir)
    tmp_dir = checkpoint_dir / "tmp_0"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with safetensors.safe_open(weights_file, framework="pt", device="cpu") as weights:
        trainable_state = {
            name: weights.get_tensor(name)
            for name in weights.keys()
            if name.startswith(STAGE1_TRAINABLE_PREFIXES)
        }
    if not trainable_state:
        raise ValueError(
            f"No tensors under {STAGE1_TRAINABLE_PREFIXES} found in {weights_file} -- "
            "is this really the n0-vtla-base checkpoint (tactile pathway already "
            "trained)?"
        )
    safetensors.torch.save_file(trainable_state, tmp_dir / "model.safetensors")
    # An unstepped AdamW's state_dict has empty per-param state (only param_groups) --
    # exactly right here, since nothing was ever optimized; a later resume from this
    # checkpoint starts momentum/variance from scratch, same as any fresh run would.
    dummy_params = [torch.nn.Parameter(t.clone()) for t in trainable_state.values()]
    dummy_optim = torch.optim.AdamW(dummy_params, lr=base_config.lr_schedule.peak_lr)
    torch.save(dummy_optim.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {
            "global_step": 0,
            "step_format": "completed_updates",
            "config": dataclasses.asdict(base_config),
            "timestamp": time.time(),
            "checkpoint_format": "trainable_only_v1",
            "source": "no_tactile_control_identity_copy",
            "note": (
                "Zero-perturbation control: tensors copied verbatim from "
                f"{weights_file}, no training performed. See this script's module "
                "docstring for why Stage 1 has no defined objective for this arm."
            ),
        },
        tmp_dir / "metadata.pt",
    )
    tmp_dir.rename(final_dir)
    logging.info(
        f"Materialized no-tactile control checkpoint ({len(trainable_state)} tensors, "
        f"identical to {weights_file}) at {final_dir}"
    )


if __name__ == "__main__":
    main()
