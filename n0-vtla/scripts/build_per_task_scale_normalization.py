#!/usr/bin/env python
"""Build a per-task-scale pad30 tactile normalization file from TouchScale-format episodes.

Mid-training (`train_stage1.sh` / `scripts/train_stage1_online.py`) turns every glove frame into a
224x224 pressure image with `scripts/itw_pressure.py::normalize_pressure`. For the per-task-scale schema
(`SCHEMA_PER_TASK`) that function needs, besides one baseline per pad, ONE force scale per task name,
because force magnitude differs a lot between tasks (a light stamp press vs. squeezing a cup) and a single
global scale would compress the light ones. This script writes that file from your own data:

    <raw_root>/<date>/<episode>/left_hand_data.npz   timestamps + tactile_<pad> arrays (T, h, w), 15 pads
                               /right_hand_data.npz
                               /task_info.json       {"name": "<task label>", ...}   (the key of the table)

Method (per pad slot, 0-14 = left hand, 15-29 = right hand; pad ids from `itw_pressure.PAD_IDS`):
  1. sample `--frames-per-episode` random frames per hand and episode, keep at most `--values-per-pad`
     random taxel values per (episode, pad);
  2. baseline[pad]  = 5th percentile of the pooled values (same as `itw_pressure.fit_normalization`);
  3. dev = value - baseline[pad]; the scale of a task is the 99.9th percentile of `dev` pooled over all 30
     pads and all episodes of that task; `default_scale` is the same pooled over every episode (used by
     `normalize_pressure` for tasks that are missing from the table);
  4. tasks with fewer than `--min-episodes` episodes are left out of the table (they fall back to
     `default_scale`); `normal_scale` is filled with `default_scale` and `contact_threshold` with 0.10 (both are
     ignored by `normalize_pressure` once `task_scale` is present, they only have to be valid numbers).

Fit on TRAINING episodes only (`--episode-list` / `--dates`), as the file records `fit_split: train`.

NOTE: this implements the schema and the semantics `itw_pressure.py` consumes. The resulting table depends on the
sampling settings and on the episodes used, so tables built with different settings or data will differ numerically.

    python scripts/build_per_task_scale_normalization.py --raw-root /data/touchscale_raw \\
        --out per_task_scale.json --workers 16
    export VTLA_ITW_NORMALIZATION=$PWD/per_task_scale.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itw_pressure import HANDS, PAD_IDS, SCHEMA_PER_TASK, load_normalization  # noqa: E402

LO_PCT, HI_PCT = 5.0, 99.9
MIN_SCALE = 1e-6


def read_task_name(ep: Path) -> str | None:
    path = ep / "task_info.json"
    if not path.is_file():
        return None
    name = json.loads(path.read_text(encoding="utf-8")).get("name")
    return str(name) if name else None


def sample_episode(args: tuple[str, int, int, int]) -> tuple[str | None, list[np.ndarray | None]]:
    """-> (task name, 30 arrays of sampled taxel values or None when that hand file is absent)."""
    ep_dir, frames, per_pad, seed = args
    ep = Path(ep_dir)
    rng = np.random.default_rng(seed)
    out: list[np.ndarray | None] = [None] * (len(HANDS) * len(PAD_IDS))
    for hand_i, hand in enumerate(HANDS):
        path = ep / f"{hand}_hand_data.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            n = len(z["timestamps"])
            if n == 0:
                continue
            take = rng.integers(0, n, size=min(frames, n))
            for slot, pad in enumerate(PAD_IDS):
                values = np.asarray(z[f"tactile_{pad}"][take], np.float32).reshape(-1)
                values = values[np.isfinite(values)]
                if len(values) > per_pad:
                    values = rng.choice(values, size=per_pad, replace=False)
                out[hand_i * len(PAD_IDS) + slot] = values
    return read_task_name(ep), out


def build_normalization(episode_dirs: list[Path], *, frames_per_episode: int = 4, values_per_pad: int = 64,
                        min_episodes: int = 3, seed: int = 42, workers: int = 1) -> dict:
    jobs = [(str(ep), frames_per_episode, values_per_pad, seed * 1_000_003 + i) for i, ep in enumerate(episode_dirs)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(sample_episode, jobs, chunksize=64))
    else:
        samples = [sample_episode(j) for j in jobs]

    n_pads = len(HANDS) * len(PAD_IDS)
    per_pad = [[] for _ in range(n_pads)]
    for _, arrays in samples:
        for i, a in enumerate(arrays):
            if a is not None and len(a):
                per_pad[i].append(a)
    missing = [i for i, p in enumerate(per_pad) if not p]
    if missing:
        raise ValueError(f"no finite training values for pad slots {missing}; check the episode layout")
    baseline = np.array([np.percentile(np.concatenate(p).astype(np.float64), LO_PCT) for p in per_pad])

    def deviations(arrays) -> np.ndarray:
        parts = [a.astype(np.float64) - baseline[i] for i, a in enumerate(arrays) if a is not None and len(a)]
        return np.concatenate(parts) if parts else np.empty(0)

    by_task: dict[str, list[np.ndarray]] = defaultdict(list)
    episodes_per_task: dict[str, int] = defaultdict(int)
    all_dev = []
    n_unnamed = 0
    for name, arrays in samples:
        dev = deviations(arrays)
        if not len(dev):
            continue
        all_dev.append(dev)
        if name is None:
            n_unnamed += 1
            continue
        by_task[name].append(dev)
        episodes_per_task[name] += 1

    default_scale = max(float(np.percentile(np.concatenate(all_dev), HI_PCT)), MIN_SCALE)
    task_scale = {name: max(float(np.percentile(np.concatenate(devs), HI_PCT)), MIN_SCALE)
                  for name, devs in sorted(by_task.items()) if episodes_per_task[name] >= min_episodes}
    if not task_scale:
        raise ValueError(f"no task has >= {min_episodes} episodes with a task_info.json 'name'; "
                         "lower --min-episodes or check task_info.json")
    result = dict(
        schema=SCHEMA_PER_TASK, fit_split="train",
        normal_baseline=[float(b) for b in baseline],
        normal_scale=[default_scale] * n_pads,
        contact_threshold=[0.10] * n_pads,
        task_scale=task_scale, default_scale=default_scale,
        provenance=dict(
            script="scripts/build_per_task_scale_normalization.py", n_episodes=len(episode_dirs),
            n_episodes_without_task_name=n_unnamed, n_tasks_in_table=len(task_scale),
            n_tasks_seen=len(by_task), min_episodes=min_episodes, frames_per_episode=frames_per_episode,
            values_per_pad=values_per_pad, percentiles=dict(baseline=LO_PCT, scale=HI_PCT), seed=seed,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    return result


def discover_episodes(raw_root: Path, dates: list[str] | None, episode_list: Path | None) -> list[Path]:
    if episode_list is not None:
        rels = [l.strip() for l in episode_list.read_text().splitlines() if l.strip() and not l.startswith("#")]
        return [raw_root / r for r in rels]
    names = dates if dates else sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    eps: list[Path] = []
    for d in names:
        if (raw_root / d).is_dir():
            eps.extend(sorted(p for p in (raw_root / d).iterdir() if p.is_dir()))
    return eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, required=True, help="<raw_root>/<date>/<episode>/ ... layout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dates", nargs="*", default=None, help="restrict to these date folders (training dates)")
    ap.add_argument("--episode-list", type=Path, default=None,
                    help="text file, one episode dir per line, relative to --raw-root (overrides --dates)")
    ap.add_argument("--frames-per-episode", type=int, default=4)
    ap.add_argument("--values-per-pad", type=int, default=64)
    ap.add_argument("--min-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; choose a new name instead of overwriting statistics")
    eps = discover_episodes(args.raw_root, args.dates, args.episode_list)
    if not eps:
        raise FileNotFoundError(f"no episode directories found under {args.raw_root}")
    norm = build_normalization(eps, frames_per_episode=args.frames_per_episode, values_per_pad=args.values_per_pad,
                               min_episodes=args.min_episodes, seed=args.seed, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(norm, indent=2, ensure_ascii=False))
    load_normalization(args.out)  # validate with the exact loader the training code uses
    s = np.array(list(norm["task_scale"].values()))
    p = norm["provenance"]
    print(f"{p['n_episodes']} episodes, {p['n_tasks_in_table']}/{p['n_tasks_seen']} tasks in the table "
          f"({p['n_episodes_without_task_name']} episodes without a task name)")
    print(f"default_scale {norm['default_scale']:.4g}; task_scale min/median/max {s.min():.4g}/{np.median(s):.4g}/{s.max():.4g}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
