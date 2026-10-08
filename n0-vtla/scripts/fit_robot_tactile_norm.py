"""Fit one shared tactile normalization across the robot-data TRAIN split (single-arm release).

Replaces robot_single_arm_adapter.py's per-episode fit_smoke_pressure (a one-recording
ad hoc fit meant only for a single-episode pipeline check) with a fit over every
train episode's live tactile file, matching the itw_pressure.py P5/P99.9 method so
downstream normalize_pressure()/pressure_rgb() from itw_pressure.py can consume it
directly. See docs/POST_TRAIN.md (section 3.4) for why a shared
fit (rather than a per-episode one) is required before the encoded tactile video
means anything.

Tactile data always comes from left_hand_data.npz (in a single-arm release
right_hand_data.npz carries no signal). Both output
hand slots are fit from that same live file -- the "left" slot is never applied to
real data downstream (only hand="right" video is ever written), it just needs to
be finite/positive to satisfy itw_pressure.load_normalization's schema check.

Usage:
  python scripts/fit_robot_tactile_norm.py \
    /path/to/raw_release \
    SPLIT.json \
    /path/to/tactile_norm.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from itw_pressure import HANDS, PAD_IDS, SCHEMA
from robot_single_arm_adapter import resolve_own_tactile_frames

SAMPLES_PER_RECORDING = 40
SEED = 42
LIVE_STD_THRESHOLD = 1e-4


def fit_robot_normalization(episode_dirs: list[Path], *, samples_per_recording: int, seed: int,
                              task_name: str | None = None, require_success_label: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    values = [[] for _ in PAD_IDS]
    for ep in episode_dirs:
        # Restricted to THIS episode's own qualifying window(s), not the whole
        # capture file it may share with other trials (including ones assigned
        # to val or holdout); sampling from the full file range would leak
        # across the split.
        npz_path, own_frames = resolve_own_tactile_frames(
            ep, task_name=task_name, require_success_label=require_success_label)
        with np.load(npz_path, allow_pickle=False) as z:
            if len(own_frames) == 0:
                raise ValueError(f"No qualifying frames for {ep}")
            pad_std = max(float(np.std(np.asarray(z[f"tactile_{p}"], np.float64)[own_frames])) for p in PAD_IDS)
            if pad_std < LIVE_STD_THRESHOLD:
                raise ValueError(f"{npz_path} looks dead within {ep.name}'s own window "
                                 f"(max per-pad std {pad_std:.2e}); do not fit normalization from it")
            take = own_frames[rng.integers(0, len(own_frames), size=min(samples_per_recording, len(own_frames)))]
            for slot, pad in enumerate(PAD_IDS):
                normal = np.asarray(z[f"tactile_{pad}"][take], np.float32)
                finite = normal[np.isfinite(normal)]
                if len(finite):
                    values[slot].append(finite.reshape(-1))

    baselines, scales, thresholds = [], [], []
    for pad, pieces in enumerate(values):
        if not pieces:
            raise ValueError(f"No finite training values for pad {pad}")
        pooled = np.concatenate(pieces).astype(np.float64)
        base, high = np.percentile(pooled, [5.0, 99.9])
        scale = max(float(high - base), 1e-6)
        med = float(np.median(pooled))
        mad = float(np.median(np.abs(pooled - med)))
        threshold = max(med + 4 * mad, float(base) + 0.10 * scale)
        baselines.append(float(base))
        scales.append(scale)
        thresholds.append(float(threshold))
    positive = [x for x in scales if x > 1e-5]
    floor = max(float(np.median(positive)) * 0.05 if positive else 1e-3, 1e-5)
    scales = [max(x, floor) for x in scales]
    thresholds = [max((t - b) / s, 0.10) for t, b, s in zip(thresholds, baselines, scales)]

    # Both hand slots come from the same live ("left"-named) file: the "left"
    # slot is inert downstream (this rig never writes a left-hand tactile video) and
    # only needs to satisfy load_normalization's finite/positive-scale check.
    assert HANDS == ("left", "right")
    return dict(schema=SCHEMA, fit_split="train",
                normal_baseline=baselines * 2, normal_scale=scales * 2, contact_threshold=thresholds * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-recording", type=int, default=SAMPLES_PER_RECORDING)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--task-name", default=None,
                         help="Expected robot/manifest.json task.name for every train episode.")
    parser.add_argument("--skip-success-label-check", action="store_true",
                         help="Skip manifest trial.label=='success'; use when this release's "
                              "admission is authoritatively decided elsewhere (see robot_single_arm_adapter).")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new normalization version; do not overwrite statistics")

    manifest = json.loads(args.split_manifest.read_text())
    episode_dirs = [args.raw_dir / uuid for uuid in manifest["train"]]
    missing = [str(p) for p in episode_dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing train episodes: {missing}")

    norm = fit_robot_normalization(episode_dirs, samples_per_recording=args.samples_per_recording, seed=args.seed,
                                     task_name=args.task_name,
                                     require_success_label=not args.skip_success_label_check)
    norm["provenance"] = dict(
        raw_dir=str(args.raw_dir),
        split_manifest=str(args.split_manifest),
        train_episode_count=len(episode_dirs),
        samples_per_recording=args.samples_per_recording,
        seed=args.seed,
        tactile_source="left_hand_data.npz (right_hand_data.npz carries no signal on this rig)",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(norm, indent=2))
    print(f"Fitted {len(episode_dirs)} train episodes -> {args.output}")
    print(f"Scale range (right/live slot): {min(norm['normal_scale'][15:]):.6g} .. {max(norm['normal_scale'][15:]):.6g}")


if __name__ == "__main__":
    main()
