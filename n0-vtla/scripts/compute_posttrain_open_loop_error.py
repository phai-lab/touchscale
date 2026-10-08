#!/usr/bin/env python
"""Open-loop SAMPLED-action error of ONE post-train checkpoint on its held-out split.

Complements compute_posttrain_held_out_loss.py (flow-matching loss = how well the denoising field
fits). Here the policy actually samples a 50-step action chunk from noise (model.sample_actions,
10 denoising steps) for each scored row and the chunk is compared with the demonstrated chunk, in
PHYSICAL units: xyz in mm, hand in raw motor counts (0..1000), rot6d dimensionless. Both chunks are
inverse-normalized (quantile stats of the run's own training asset) and live in the delta-EEF space
the model predicts, so the error equals the absolute-action error (the current state cancels).

Per row and horizon step h: xyz error = L2 norm over the 3 dims (mm); rot6d and hand = mean
absolute error over their dims. Reported as macro means over episodes (with episode SE), overall
and by horizon, for (a) individual samples (expected error) and (b) the mean of the samples, plus
context baselines: predicting zero motion (xyz/rot6d) and predicting the training-set mean hand
command. Noise is seeded per row so every checkpoint sees the same draws.

Run from the repo root with the n0vtla env (same conventions as compute_posttrain_held_out_loss.py):
  python scripts/compute_posttrain_open_loop_error.py \
    --checkpoint <ckpt dir with model.safetensors> --checkpoint-label midtrain_ckpt \
    --dataset-root <.../canonical_robot_dataset_val> --asset-id my_task --output out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

GROUPS = {"xyz": slice(10, 13), "rot6d": slice(13, 19), "hand": slice(20, 26)}


def load_action_stats(asset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = json.loads((asset_dir / "norm_stats.json").read_text())
    st = d.get("norm_stats", d)["actions"]
    return np.asarray(st["q01"], np.float64), np.asarray(st["q99"], np.float64), np.asarray(st["mean"], np.float64)


def unnormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    # inverse of Normalize(use_quantiles=True): (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
    d = x.shape[-1]
    return (x + 1.0) / 2.0 * (q99[:d] - q01[:d] + 1e-6) + q01[:d]


def chunk_errors(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """(H, D) physical-unit chunks -> per-horizon errors."""
    e = pred - gt
    return {
        "xyz_mm": np.linalg.norm(e[:, GROUPS["xyz"]], axis=1),
        "rot6d_mae": np.abs(e[:, GROUPS["rot6d"]]).mean(axis=1),
        "hand_mae": np.abs(e[:, GROUPS["hand"]]).mean(axis=1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="vtla_tactile_posttrain")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--asset-id", required=True, help="asset id whose norm_stats.json the checkpoint was trained with")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=10, help="score every Nth dataset row")
    p.add_argument("--samples", type=int, default=2, help="independent noise draws per scored row")
    p.add_argument("--num-steps", type=int, default=10, help="denoising steps")
    p.add_argument("--max-rows", type=int, default=0, help="quick check: stop after this many rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    os.environ["VTLA_DATASET_PATH"] = str(args.dataset_root)
    os.environ["VTLA_ASSET_ID"] = args.asset_id
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import jax
    import safetensors.torch
    import torch

    import scripts.train_n0vtla  # noqa: F401  (side effect: patches PI0Pytorch -> N0VTLAPolicy)
    import n0vtla.models.pi0_config
    import n0vtla.models_pytorch.pi0_pytorch
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = [ep["length"] for ep in episodes]
    total_rows = sum(lengths)
    boundaries = np.concatenate([[0], np.cumsum(lengths)])

    config = _config.get_config(args.config)
    object.__setattr__(config, "batch_size", 1)
    asset_dir = Path(config.assets_base_dir) / config.name / args.asset_id
    q01, q99, a_mean = load_action_stats(asset_dir)

    model_cfg = config.model
    if not isinstance(model_cfg, n0vtla.models.pi0_config.Pi0Config):
        raise TypeError(f"unexpected model config type {type(model_cfg)}")
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    device = torch.device(args.device)
    print(f"=== loading {args.checkpoint} ===", flush=True)
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(model, str(args.checkpoint / "model.safetensors"), device=str(device))
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch -- missing={missing}, unexpected={unexpected}")
    model.eval()

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    if len(loader._data_loader.torch_loader.dataset) != total_rows:
        raise RuntimeError("dataset length != episodes.jsonl total")
    print(f"=== rows={total_rows} episodes={len(episodes)} stride={args.stride} samples={args.samples} steps={args.num_steps} ===", flush=True)

    keys = ["xyz_mm", "rot6d_mae", "hand_mae"]
    per_ep: dict[int, dict[str, list[np.ndarray]]] = {
        i: {f"{v}_{k}": [] for v in ("sample", "mean", "zero") for k in keys} for i in range(len(episodes))
    }
    n_scored = 0
    for row, (observation, actions) in enumerate(loader):
        if row >= total_rows or (args.max_rows and row >= args.max_rows):
            break
        if row % args.stride:
            continue
        ep = int(np.searchsorted(boundaries, row, side="right") - 1)
        observation = jax.tree.map(lambda x: x.to(device), observation)
        gt = unnormalize(actions[0].float().cpu().numpy().astype(np.float64), q01, q99)
        samples = []
        for r in range(args.samples):
            torch.manual_seed(args.seed * 100_000 + row * 10 + r)
            with torch.no_grad():
                out = model.sample_actions(device, observation, num_steps=args.num_steps)
            samples.append(unnormalize(out[0].float().cpu().numpy().astype(np.float64), q01, q99))
        d = per_ep[ep]
        for k, v in chunk_errors(np.mean(samples, axis=0), gt).items():
            d[f"mean_{k}"].append(v)
        for k in keys:
            d[f"sample_{k}"].append(np.mean([chunk_errors(s, gt)[k] for s in samples], axis=0))
        zero = np.zeros_like(gt)
        zero[:, GROUPS["hand"]] = a_mean[GROUPS["hand"]]  # zero delta motion; hand = training-set mean command
        for k, v in chunk_errors(zero, gt).items():
            d[f"zero_{k}"].append(v)
        n_scored += 1
        if n_scored % 50 == 0:
            print(f"  scored {n_scored} rows (row {row}/{total_rows})", flush=True)

    # macro over episodes: per-episode mean curve (H,), then mean/SE across episodes
    metrics = {}
    for name in per_ep[0]:
        curves = [np.mean(per_ep[i][name], axis=0) for i in per_ep if per_ep[i][name]]
        arr = np.stack(curves)  # (episodes, H)
        metrics[name] = dict(
            mean=float(arr.mean()),
            episode_se=float(arr.mean(axis=1).std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
            by_horizon=[float(x) for x in arr.mean(axis=0)],
        )
    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_label=args.checkpoint_label, dataset_root=str(args.dataset_root),
        asset_id=args.asset_id, stride=args.stride, samples=args.samples, num_steps=args.num_steps, seed=args.seed,
        n_episodes=len(episodes), n_scored_rows=n_scored,
        units=dict(xyz_mm="mm (L2 over xyz)", rot6d_mae="mean abs error (rot6d dims)", hand_mae="raw motor counts 0..1000"),
        metrics=metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    m = metrics
    print(f"{args.checkpoint_label}: xyz={m['sample_xyz_mm']['mean']:.2f}mm (zero-motion {m['zero_xyz_mm']['mean']:.2f}) "
          f"hand={m['sample_hand_mae']['mean']:.1f} (mean-cmd {m['zero_hand_mae']['mean']:.1f}) ({n_scored} rows)", flush=True)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
