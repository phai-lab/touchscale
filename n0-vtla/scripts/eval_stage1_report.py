"""Held-out, observed-history future tactile prediction; never action rollout.

Uses the production observation transforms and model methods, with augmentation
disabled. Samples every fifth valid timestamp in every validation episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np


def sample_records(root, horizon=50, stride=5):
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    records, offset = [], 0
    for ep in episodes:
        for frame in range(0, ep["length"] - horizon, stride):
            records.append((ep["episode_index"], frame, offset + frame))
        offset += ep["length"]
    return np.asarray(records, np.int64).reshape(-1, 3)


def summarize(pred, target, records):
    error = np.abs(pred - target).mean(axis=(1, 2))
    zero = np.abs(target).mean(axis=(1, 2))
    # Fixed threshold in model image-difference units, not fitted on validation.
    active = zero > 1 / 255
    result = {"samples": len(pred), "mae": float(error.mean()), "zero_mae": float(zero.mean()),
              "active_threshold": 1 / 255, "active_samples": int(active.sum()),
              "active_mae": float(error[active].mean()) if active.any() else None,
              "active_zero_mae": float(zero[active].mean()) if active.any() else None}
    per_episode = []
    for ep in np.unique(records[:, 0]):
        mask = records[:, 0] == ep
        per_episode.append({"episode": int(ep), "mae": float(error[mask].mean()),
                            "zero_mae": float(zero[mask].mean()), "samples": int(mask.sum())})
    result["episodes"] = per_episode
    result["macro_episode_mae"] = float(np.mean([r["mae"] for r in per_episode]))
    return result


def evaluate(args):
    os.environ["VTLA_DATASET_PATH"] = str(args.dataset)
    os.environ["VTLA_ASSET_ID"] = args.dataset.name
    import jax
    import torch
    import torch.nn.functional as F
    from n0vtla.models.model import Observation
    from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy
    from n0vtla.training import config, data_loader
    from train_stage1_predictor import load_stage1_policy_weights, set_seed

    selection = json.loads((args.dataset / "meta/selection.json").read_text())
    if selection["split"] != "validation":
        raise ValueError("Reporting requires the held-out validation split")
    cfg = config.get_config("vtla_stage1_predictor_pretrain")
    set_seed(cfg.seed, 0)
    object.__setattr__(cfg.model, "dtype", cfg.pytorch_training_precision)
    loader = data_loader.create_data_loader(cfg, framework="pytorch", shuffle=False, skip_norm_stats=True)
    dataset = loader._data_loader.torch_loader.dataset
    policy = N0VTLAPolicy(cfg.model).to("cuda")
    checkpoint = args.checkpoint / "model.safetensors" if args.checkpoint.is_dir() else args.checkpoint
    missing, unexpected = load_stage1_policy_weights(policy, checkpoint, "cuda", strict=False)
    allowed = ("tactile_recon_head.",) if args.label == "before" else ()
    if unexpected or any(not k.startswith(allowed) for k in missing):
        raise ValueError(f"Incompatible evaluation checkpoint: {missing}, {unexpected}")
    policy.eval().requires_grad_(False)
    records = sample_records(args.dataset)
    if not len(records):
        raise ValueError("No valid future timestamps")
    predictions, targets = [], []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            batch = data_loader._collate_fn([dataset[int(r[2])] for r in batch_records])
            batch = jax.tree.map(lambda x: torch.as_tensor(x).to("cuda"), batch)
            obs = Observation.from_dict(batch)
            images, masks, tokens, token_masks, _, _ = policy._preprocess_observation(obs, train=False)
            context, _, prefix_masks, _, _ = policy._prefix_forward(
                images, masks, tokens, token_masks, use_cache=False)
            z, _, has_tac = policy._compute_z(context, prefix_masks)
            _, field, has_future = policy._build_future_target(
                policy._last_tac_f or {}, policy._last_tac_t or {}, policy._last_tac_mask_f)
            if field is None or not (has_tac & has_future).all():
                raise ValueError("Invalid future target in the evaluation timestamp manifest")
            target = F.adaptive_avg_pool2d(field.mean(dim=1, keepdim=True), policy.tactile_recon_head.grid)[:, 0]
            pred = policy.tactile_recon_head(z)
            if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
                raise ValueError("Nonfinite evaluation prediction/target")
            predictions.append(pred.float().cpu().numpy())
            targets.append(target.float().cpu().numpy())
            print(f"{args.label}: {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "predictions.npz", prediction=pred, target=target, records=records)
    metrics = summarize(pred, target, records)
    metadata_path = checkpoint.parent / "metadata.pt"
    completed_updates = 0
    if args.label == "after":
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        if metadata.get("step_format") != "completed_updates":
            raise ValueError("Ambiguous checkpoint step format")
        completed_updates = int(metadata["global_step"])
    metrics.update(label=args.label, checkpoint=str(checkpoint), dataset=str(args.dataset), stride=5,
                   missing_keys=sorted(missing), seed=cfg.seed, completed_updates=completed_updates,
                   normalization_sha256=hashlib.sha256((args.dataset / "meta/tactile_normalization.json").read_bytes()).hexdigest())
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def signed_heatmap(field, limit=0.1):
    value = np.clip(field / limit, -1, 1)
    rgb = np.ones((*field.shape, 3), np.float32)
    rgb[..., 0] -= np.maximum(-value, 0)
    rgb[..., 1] -= np.abs(value)
    rgb[..., 2] -= np.maximum(value, 0)
    return cv2.resize(np.rint(rgb * 255).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)


def plot_report(root, report, log_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for line in log_path.read_text().splitlines():
        if "stage1_nce=" not in line:
            continue
        fields = dict(re.findall(r"(step|stage1_nce|stage1_recon|stage1_total)=([0-9.eE+-]+)", line))
        if len(fields) == 4:
            entries.append({k: float(v) for k, v in fields.items()})
    (root / "training_metrics.json").write_text(json.dumps(entries, indent=2))
    if entries:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for ax, key, title in zip(axes, ("stage1_nce", "stage1_recon"), ("Training InfoNCE", "Training reconstruction L1")):
            ax.plot([e["step"] for e in entries], [e[key] for e in entries], color="#267f77")
            ax.set(title=title, xlabel="Completed updates", ylabel="Logged interval mean")
            ax.grid(alpha=.2)
        fig.savefig(root / "training_curves.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, key, zero_key, title in zip(axes, ("mae", "active_mae"), ("zero_mae", "active_zero_mae"),
                                      ("All validation samples", "Active-change subset")):
        vals = [report["before"][key], report["after"][key], report["after"][zero_key]]
        if any(v is None for v in vals):
            ax.text(.5, .5, "No active samples", ha="center", transform=ax.transAxes)
            continue
        ax.bar(["Before", "After", "Zero change"], vals, color=["#7b8490", "#267f77", "#bf6b39"])
        ax.set(title=title, ylabel="MAE (log scale; lower is better)", yscale="log")
        for i, value in enumerate(vals):
            ax.annotate(f"{value:.5f}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=.2)
    fig.savefig(root / "validation_mae.png", dpi=160)
    plt.close(fig)


def compare(args):
    import imageio.v2 as imageio
    before_meta = json.loads((args.before / "metrics.json").read_text())
    after_meta = json.loads((args.after / "metrics.json").read_text())
    for key in ("dataset", "normalization_sha256", "seed", "stride"):
        if before_meta[key] != after_meta[key]:
            raise ValueError(f"Before/after provenance differs: {key}")
    with np.load(args.before / "predictions.npz") as b, np.load(args.after / "predictions.npz") as a:
        records, target = a["records"], a["target"]
        before, after = b["prediction"], a["prediction"]
        if not np.array_equal(records, b["records"]) or not np.allclose(target, b["target"], atol=1e-7):
            raise ValueError("Before/after timestamps or targets differ")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"before": summarize(before, target, records), "after": summarize(after, target, records),
              "display_limit": 0.1, "mode": "observed-history +50-frame prediction, not closed-loop rollout"}
    energy = np.abs(target).mean(axis=(1, 2))
    peak = int(np.argmax(energy))
    centers = {"contact": peak, "ordinary": 0}
    report["clips"] = {}
    video_keys = ("third_view", "left_wrist_left_tactile", "right_wrist_right_tactile")
    for name, center in centers.items():
        ep = int(records[center, 0])
        candidates = np.flatnonzero(records[:, 0] == ep)
        location = int(np.flatnonzero(candidates == center)[0])
        start = max(0, min(location - 24, len(candidates) - 48))
        selected = candidates[start:start + 48]
        report["clips"][name] = {"episode": ep, "first_frame": int(records[selected[0], 1]),
                                    "selection": "maximum ground-truth change" if name == "contact" else "first validation episode"}
        caps = [cv2.VideoCapture(str(args.dataset / "videos" / f"chunk-{ep // 1000:03d}" /
                                    f"observation.image.{key}" / f"episode_{ep:06d}.mp4")) for key in video_keys]
        try:
            with imageio.get_writer(str(args.output / f"{name}.mp4"), fps=6, codec="libx264",
                                    pixelformat="yuv420p", macro_block_size=1,
                                    ffmpeg_params=["-crf", "18", "-movflags", "+faststart"]) as writer:
                for i in selected:
                    panels = []
                    for cap in caps:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(records[i, 1]))
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError("Could not decode source preview")
                        panels.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    panels += [signed_heatmap(target[i]), signed_heatmap(before[i]), signed_heatmap(after[i])]
                    canvas = np.full((556, 672, 3), 245, np.uint8)
                    titles = ["Current RGB", "Current left pressure", "Current right pressure",
                              "True future change", "Before training", f"After {after_meta['completed_updates']} updates"]
                    for j, (panel, title) in enumerate(zip(panels, titles)):
                        y, x = 36 + (j // 3) * 254, (j % 3) * 224
                        cv2.putText(canvas, title, (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1)
                        canvas[y + 26:y + 250, x:x + 224] = panel
                    cv2.putText(canvas, f"Offline +1.67s | t={records[i, 1] / 30:.2f}s | signed range +/-0.1",
                                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .46, (20, 20, 20), 1)
                    cv2.putText(canvas, "8x8 hand-averaged change | blue: decrease | red: increase | fixed scale",
                                (8, 553), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1)
                    writer.append_data(canvas)
                    if i == selected[len(selected) // 2]:
                        imageio.imwrite(args.output / f"{name}.png", canvas)
        finally:
            for cap in caps:
                cap.release()
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    log_path = args.after.parent / "training.log"
    if log_path.is_file():
        plot_report(args.output, report, log_path)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--label", choices=["before", "after"], required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--batch-size", type=int, default=16, help="Evaluation only; does not alter training")
    cp = sub.add_parser("compare")
    for key in ("dataset", "before", "after", "output"):
        cp.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args) if args.mode == "evaluate" else compare(args)


if __name__ == "__main__":
    main()
