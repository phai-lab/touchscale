"""Replay check of a trained dual-arm checkpoint through the REAL inference path.

Loads the checkpoint with create_trained_policy (the call serve_policy.py makes: checkpoint's own
norm stats, input transforms, Unnormalize + RelRotAbsoluteActions on the way out), feeds it
observations built exactly like a client would (HWC uint8 RGB, tactile [baseline, current] stacks,
no action keys), and compares the returned ABSOLUTE 50-step chunk with the dataset's own absolute
action chunk, for both arms and both hands, in physical units. Also prints the trivial
"hold the current command" baseline so the numbers have a reference.

This is a data-contract check (does the decode/normalization chain give physical, valid commands for
both arms?), not a performance metric: the frames here are TRAINING
frames, and the sampled chunk is stochastic (flow matching).

Run from the N0-VTLA repo root, e.g.
  JAX_PLATFORMS=cpu python scripts/check_dual_policy_replay.py \
      --checkpoint checkpoints/vtla_tactile_posttrain/<exp name>/<step> \
      --dataset DATA/canonical_dual_arm_train --asset-id my_dual_arm_task
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ARMS = {"left": (0, 3, 9), "right": (10, 13, 19)}      # xyz lo, rot lo, rot hi
HANDS = {"right": (20, 26), "left": (26, 32)}


def to_hwc_uint8(x) -> np.ndarray:
    a = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 + 1e-3 else a, 0, 255).astype(np.uint8)
    return a


def to_tactile_stack(x) -> np.ndarray:
    a = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
    if a.ndim == 4 and a.shape[1] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (0, 2, 3, 1))
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 + 1e-3 else a, 0, 255).astype(np.uint8)
    return a


def geodesic_deg(r6_a: np.ndarray, r6_b: np.ndarray) -> np.ndarray:
    from n0vtla.policies.rotation_utils import rot6d_to_matrix
    ra, rb = rot6d_to_matrix(r6_a.astype(np.float64)), rot6d_to_matrix(r6_b.astype(np.float64))
    tr = np.einsum("...ij,...ij->...", ra, rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Dir with model.safetensors + assets/")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    os.environ["VTLA_DATASET_PATH"] = str(args.dataset)
    os.environ["VTLA_ASSET_ID"] = args.asset_id
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch

    from n0vtla.policies import policy_config as _pc
    from n0vtla.policies.rotation_utils import rot6d_to_matrix
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    config = _config.get_config("vtla_tactile_posttrain")
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = _pc.create_trained_policy(config, args.checkpoint, default_prompt=None)
    ds = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    episodes = [json.loads(l) for l in (args.dataset / "meta/episodes.jsonl").open()]
    starts = np.concatenate([[0], np.cumsum([e["length"] for e in episodes])])
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(int(starts[-1]), size=min(args.num_samples, int(starts[-1])), replace=False))

    err = {k: [] for k in ("xyz_L", "xyz_R", "rot_L", "rot_R", "hand_L", "hand_R",
                           "hold_xyz_L", "hold_xyz_R", "hold_rot_L", "hold_rot_R", "hold_hand_L", "hold_hand_R")}
    sanity = dict(rot_col_norm_dev=[], hand_out_of_range=[], nonfinite=0)
    for n, idx in enumerate(picks, start=1):
        item = ds[int(idx)]
        raw = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in item.items()}
        gt = np.asarray(raw["action"], np.float32)                         # (50, 32) absolute
        pad = np.asarray(raw["action_is_pad"]).astype(bool)
        state = np.asarray(raw["observation.state"], np.float32)
        obs = {"observation.state": state, "prompt": "Perform the task."}
        for view in ("third_view", "left_wrist_view", "right_wrist_view"):
            obs[f"observation.image.{view}"] = to_hwc_uint8(raw[f"observation.image.{view}"])
        for view in ("left_wrist_left_tactile", "right_wrist_right_tactile"):
            obs[f"observation.image.{view}"] = to_tactile_stack(raw[f"observation.image.{view}"])
        torch.manual_seed(int(args.seed) + n)
        pred = np.asarray(policy.infer(obs)["actions"], np.float32)       # (50, 32) absolute, physical
        if pred.shape != (50, 32) or not np.isfinite(pred).all():
            sanity["nonfinite"] += 1
            continue
        ok = ~pad
        hold = np.broadcast_to(state, gt.shape)
        for side, (lo, rlo, rhi) in ARMS.items():
            tag = side[0].upper()
            err[f"xyz_{tag}"].append(np.linalg.norm(pred[ok, lo:lo + 3] - gt[ok, lo:lo + 3], axis=1).mean())
            err[f"rot_{tag}"].append(geodesic_deg(pred[ok, rlo:rhi], gt[ok, rlo:rhi]).mean())
            err[f"hold_xyz_{tag}"].append(np.linalg.norm(hold[ok, lo:lo + 3] - gt[ok, lo:lo + 3], axis=1).mean())
            err[f"hold_rot_{tag}"].append(geodesic_deg(hold[ok, rlo:rhi], gt[ok, rlo:rhi]).mean())
            R = rot6d_to_matrix(pred[ok, rlo:rhi].astype(np.float64))
            sanity["rot_col_norm_dev"].append(float(np.abs(np.linalg.norm(pred[ok, rlo:rlo + 3], axis=1) - 1).max()))
            sanity.setdefault("rot_det_dev", []).append(float(np.abs(np.linalg.det(R) - 1).max()))
        for side, (a, b) in HANDS.items():
            tag = side[0].upper()
            err[f"hand_{tag}"].append(np.abs(pred[ok, a:b] - gt[ok, a:b]).mean())
            err[f"hold_hand_{tag}"].append(np.abs(hold[ok, a:b] - gt[ok, a:b]).mean())
            sanity["hand_out_of_range"].append(float(((pred[ok, a:b] < -1) | (pred[ok, a:b] > 1001)).mean()))
        print(f"[{n}/{len(picks)}] frame {idx}", flush=True)

    summary = {k: float(np.mean(v)) for k, v in err.items() if v}
    summary["n_samples"] = len(err["xyz_L"])
    summary["rot_col_norm_max_dev"] = float(np.max(sanity["rot_col_norm_dev"])) if sanity["rot_col_norm_dev"] else None
    summary["rot_det_max_dev"] = float(np.max(sanity["rot_det_dev"])) if sanity.get("rot_det_dev") else None
    summary["hand_out_of_0_1000_frac"] = float(np.mean(sanity["hand_out_of_range"])) if sanity["hand_out_of_range"] else None
    summary["nonfinite_samples"] = sanity["nonfinite"]
    summary["note"] = ("TRAINING frames, stochastic sampling: a contract check, not a generalization metric. "
                       "hold_* = predicting the current command for all 50 steps.")
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
