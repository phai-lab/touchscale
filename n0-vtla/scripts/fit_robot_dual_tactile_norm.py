"""Fit ONE shared pad30 tactile normalization for the dual-arm TRAIN split.

Slots follow itw_pressure.HANDS = ("left", "right"): slots 0..14 = the robot LEFT hand's
glove (stored as right_hand_data.npz), slots 15..29 = the robot RIGHT hand's glove (stored as
left_hand_data.npz) -- the tactile files are crossed in the input format, see robot_dual_arm_adapter.GLOVE_FILE.
Each hand is fitted independently (the two gloves can differ in response magnitude, so a shared
fit would suppress the weaker one). Same P5/P99.9 method as fit_robot_tactile_norm.py; samples are drawn only
from each episode's own legal dual-arm rows (robot_dual_arm_adapter.resolve_dual_window).

Usage:
  python scripts/fit_robot_dual_tactile_norm.py RAW_DIR SPLIT_JSON OUT.json
    (SPLIT_JSON: {"train": [episode uuids], ...} -- uses its "train" uuids)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import robot_dual_arm_adapter as nd
from itw_pressure import PAD_IDS, SCHEMA

SAMPLES_PER_RECORDING = 40
SEED = 42
LIVE_STD_THRESHOLD = 1e-4


def fit(episode_dirs: list[Path], *, samples_per_recording: int, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    pieces = {s: [[] for _ in PAD_IDS] for s in nd.SIDES}
    dead = {s: [] for s in nd.SIDES}
    for ep in episode_dirs:
        w = nd.resolve_dual_window(ep)
        if not w["runs"]:
            raise ValueError(f"{ep.name}: no legal dual-arm window")
        rows = np.concatenate(w["runs"])
        for s in nd.SIDES:
            frames = np.unique(w["tactile"][s][rows])
            take = frames[rng.integers(0, len(frames), size=min(samples_per_recording, len(frames)))]
            with np.load(ep / nd.GLOVE_FILE[s], allow_pickle=False) as z:
                std = max(float(np.std(np.asarray(z[f"tactile_{p}"], np.float64)[frames])) for p in PAD_IDS)
                if std < LIVE_STD_THRESHOLD:
                    dead[s].append(ep.name)
                    continue
                for slot, pad in enumerate(PAD_IDS):
                    v = np.asarray(z[f"tactile_{pad}"][take], np.float32)
                    v = v[np.isfinite(v)]
                    if len(v):
                        pieces[s][slot].append(v.reshape(-1))
    baseline, scale, threshold = [0.0] * 30, [1.0] * 30, [0.1] * 30
    for s, offset in (("left", 0), ("right", 15)):
        b, sc, th = [], [], []
        for slot, p in enumerate(pieces[s]):
            if not p:
                raise ValueError(f"no live training values for {s} hand pad slot {slot}")
            pooled = np.concatenate(p).astype(np.float64)
            lo, hi = np.percentile(pooled, [5.0, 99.9])
            med = float(np.median(pooled))
            mad = float(np.median(np.abs(pooled - med)))
            b.append(float(lo))
            sc.append(max(float(hi - lo), 1e-6))
            th.append(max(med + 4 * mad, float(lo) + 0.10 * sc[-1]))
        positive = [x for x in sc if x > 1e-5]
        floor = max(float(np.median(positive)) * 0.05 if positive else 1e-3, 1e-5)
        sc = [max(x, floor) for x in sc]
        th = [max((t - bb) / x, 0.10) for t, bb, x in zip(th, b, sc)]
        baseline[offset:offset + 15], scale[offset:offset + 15], threshold[offset:offset + 15] = b, sc, th
    return dict(schema=SCHEMA, fit_split="train", normal_baseline=baseline, normal_scale=scale,
                contact_threshold=threshold), dead


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("split_json", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--samples-per-recording", type=int, default=SAMPLES_PER_RECORDING)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new normalization version; do not overwrite statistics")
    uuids = json.loads(args.split_json.read_text())["train"]
    dirs = [args.raw_dir / u for u in uuids]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing train episodes: {missing[:5]}")
    norm, dead = fit(dirs, samples_per_recording=args.samples_per_recording, seed=args.seed)
    norm["provenance"] = dict(raw_dir=str(args.raw_dir), split_json=str(args.split_json), train_episode_count=len(dirs),
                              samples_per_recording=args.samples_per_recording, seed=args.seed,
                              slots="0..14 robot LEFT hand (right_hand_data.npz), 15..29 robot RIGHT hand (left_hand_data.npz)",
                              dead_glove_episodes=dead)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(norm, indent=2))
    for name, lo in (("left hand", 0), ("right hand", 15)):
        s = norm["normal_scale"][lo:lo + 15]
        print(f"{name}: scale {min(s):.4g} .. {max(s):.4g}")
    if any(dead.values()):
        print(f"WARNING dead-glove episodes (excluded from fit): {dead}")
    print(f"Fitted {len(dirs)} train episodes -> {args.output}")


if __name__ == "__main__":
    main()
