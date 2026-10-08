#!/usr/bin/env python
"""Merges a Stage-1 trainable-only checkpoint on top of a full base checkpoint, producing
ONE complete model.safetensors that `scripts/train_pytorch.py` (post-training, e.g.
`vtla_tactile_posttrain`) can load unmodified via its existing weight-loading path.

Why this is needed: `save_stage1_checkpoint` (train_stage1_predictor.py) saves only the
~123M trainable tactile params (STAGE1_TRAINABLE_PREFIXES), not the full ~3.8B-param policy
-- see that module's docstring for why (the frozen base never changes during Stage 1, so it is
not duplicated into every checkpoint). The Stage-1 training scripts load the base and this delta
as two separate steps (load_stage1_policy_weights then load_stage1_checkpoint).
`train_pytorch.py`'s post-train path does NOT know about this two-step format -- it does one
`safetensors.torch.load_model(model, "<pytorch_weight_path>/model.safetensors", strict=False)`
call and nothing else. Pointing VTLA_PRETRAINED_CHECKPOINT directly at a Stage-1 trainable-only
checkpoint would leave the entire frozen VLM + action expert (everything NOT in
STAGE1_TRAINABLE_PREFIXES) at random init, since strict=False lets missing keys through
without erroring.

So: merge base + Stage-1 delta into one file HERE, once, offline, and point
VTLA_PRETRAINED_CHECKPOINT at the merged output directory for post-training.

Usage:
    python scripts/merge_stage1_into_base_checkpoint.py \\
        --base-checkpoint ../checkpoints/n0-vtla-base \\
        --stage1-checkpoint checkpoints/vtla_stage1_predictor_pretrain/stage1_online/20000 \\
        --output ../checkpoints/n0-vtla-base_plus_stage1_online_20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import safetensors.torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stage1_predictor import STAGE1_TRAINABLE_PREFIXES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, help="Dir containing the FULL base "
                         "policy's model.safetensors (e.g. n0-vtla-base).")
    parser.add_argument("--stage1-checkpoint", required=True, help="Dir containing a Stage-1 "
                         "trainable-only checkpoint's model.safetensors (e.g. "
                         "checkpoints/vtla_stage1_predictor_pretrain/<exp>/<step>).")
    parser.add_argument("--output", required=True, help="Output dir for the merged, complete "
                         "model.safetensors. Must not already exist unless --overwrite.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_checkpoint)
    base_weights_file = base_path / "model.safetensors" if base_path.is_dir() else base_path
    stage1_path = Path(args.stage1_checkpoint)
    stage1_weights_file = stage1_path / "model.safetensors" if stage1_path.is_dir() else stage1_path
    if not base_weights_file.is_file():
        raise FileNotFoundError(f"Base checkpoint missing: {base_weights_file}")
    if not stage1_weights_file.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint missing: {stage1_weights_file}")

    output_dir = Path(args.output)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base checkpoint: {base_weights_file}")
    base_state = safetensors.torch.load_file(base_weights_file)
    print(f"  {len(base_state)} tensors")

    print(f"Loading Stage-1 delta: {stage1_weights_file}")
    stage1_state = safetensors.torch.load_file(stage1_weights_file)
    print(f"  {len(stage1_state)} tensors")

    # Sanity: every Stage-1 tensor must fall under the known trainable prefixes, so nothing
    # unexpected is merged in.
    bad_prefix = [name for name in stage1_state if not name.startswith(STAGE1_TRAINABLE_PREFIXES)]
    if bad_prefix:
        raise ValueError(f"Stage-1 checkpoint has tensors outside STAGE1_TRAINABLE_PREFIXES: {bad_prefix}")

    # tactile_recon_head is constructed ONLY when stage1_pretrain_enabled=True (see
    # N0VTLAPolicy.__init__ in n0vtla_policy.py) -- it's Stage-1-training-only scaffolding for
    # the auxiliary L1 reconstruction loss and is NEVER part of n0-vtla-base or any post-train
    # config's model (vtla_tactile_posttrain doesn't set stage1_pretrain_enabled). So it has no
    # home to merge into downstream -- expected and dropped, not an error. Anything else missing
    # from base (tactile_encoder.tactile_proj.* / tactile_predictor.*) IS a real mismatch, since
    # both of those ARE part of every config that has tactile_predictor_enabled=True.
    missing_in_base = [name for name in stage1_state if name not in base_state]
    unexpected_missing = [name for name in missing_in_base if not name.startswith("tactile_recon_head.")]
    if unexpected_missing:
        raise ValueError(f"Stage-1 tensors not present in base checkpoint (architecture mismatch?): {unexpected_missing}")
    if missing_in_base:
        print(f"Dropping {len(missing_in_base)} tactile_recon_head.* tensors (Stage-1-only, not part of "
              f"the post-train model architecture, no home to merge into): {missing_in_base}")

    mergeable_state = {name: t for name, t in stage1_state.items() if name in base_state}
    shape_mismatch = [
        name for name, t in mergeable_state.items() if tuple(t.shape) != tuple(base_state[name].shape)
    ]
    if shape_mismatch:
        raise ValueError(f"Shape mismatch between base and Stage-1 for: {shape_mismatch}")

    merged_state = dict(base_state)
    merged_state.update(mergeable_state)
    print(f"Merged: {len(mergeable_state)} tensors overridden, {len(merged_state) - len(mergeable_state)} kept from base "
          f"({len(merged_state)} total).")

    out_file = output_dir / "model.safetensors"
    safetensors.torch.save_file(merged_state, out_file)
    (output_dir / "merge_provenance.json").write_text(json.dumps({
        "base_checkpoint": str(base_weights_file),
        "stage1_checkpoint": str(stage1_weights_file),
        "overridden_tensor_count": len(mergeable_state),
        "dropped_stage1_only_tensors": missing_in_base,
        "total_tensor_count": len(merged_state),
        "timestamp": time.time(),
    }, indent=2))
    print(f"Wrote merged checkpoint: {out_file}")
    print("Point VTLA_PRETRAINED_CHECKPOINT at this output directory for post-training "
          "(train_pytorch.py's weight loader reads <dir>/model.safetensors directly, strict=False).")


if __name__ == "__main__":
    main()
