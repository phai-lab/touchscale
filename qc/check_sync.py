#!/usr/bin/env python3
"""Timestamp synchronization gate for a batch of recordings.

Finds every recording directory under --root (any directory containing
`rgb_head.csv`) and, for each one, simulates resampling all six streams onto a
uniform 30 Hz grid over their common time window, taking for every stream the
sample nearest in time to each grid tick:

    6 streams = head RGB, head depth, left wrist, right wrist, left glove, right glove

A recording PASSES iff:
  (1) all six timestamp streams are present, the four video files exist, and the
      glove time window covers the camera window;
  (2) ALIGNED:    >= --align-thresh of grid ticks have all six streams within
                  --align-tol-ms of the tick (primary metric);
  (3) FEW HOLES:  <= --max-hole-frac of ticks are holes (nearest sample of some
                  stream more than one frame, 33.3 ms, away);
  (4) NO BLACKOUT: no nearest-sample gap exceeds --max-gap-ms.

Isolated one-frame hiccups are tolerated (they can be masked or interpolated);
multi-frame blackouts are not, because fast contact events would be lost.

The default 17.5 ms tolerance is slightly above the geometric half-frame (16.7 ms):
the gloves sample at ~35-56 Hz and their software timestamps jitter by ~1 ms, so
grazing 16.7 ms is not a real defect. Only timestamps are read (no video decoding),
so this runs quickly on any machine.

    python check_sync.py --root <batch_dir> [--json sync.json]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

GRID_HZ = 30.0
ONE_FRAME = 1.0 / GRID_HZ          # nearest-sample gap beyond this = a hole
DEFAULT_ALIGN_TOL_MS = 17.5
DEFAULT_ALIGN_THRESH = 0.95
DEFAULT_MAX_HOLE_FRAC = 0.01
DEFAULT_MAX_GAP_MS = 66.0          # ~2 frames
STREAM_NAMES = {"rgb": "head RGB", "depth": "head depth", "wristL": "left wrist",
                "wristR": "right wrist", "tacL": "left glove", "tacR": "right glove"}


def csv_ts(p):
    """Timestamps (last column) from a CSV; tolerates a header row and bad lines."""
    if not p or not os.path.exists(p):
        return np.array([])
    rows = [x.strip() for x in open(p, errors="replace").read().splitlines() if x.strip()]
    if rows and any(c.isalpha() for c in rows[0]):
        rows = rows[1:]
    out = []
    for r in rows:
        try:
            out.append(float(r.split(",")[-1]))
        except ValueError:
            pass
    return np.array(out, float)


def _first(*patterns):
    """First existing path among glob patterns (supports flat and nested layouts)."""
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def glove_ts(ed, side):
    p = _first(f"{ed}/glove/*/{side}_hand_data.npz", f"{ed}/{side}_hand_data.npz")
    if not p:
        return np.array([])
    try:
        return np.load(p)["timestamps"].astype(float)
    except Exception:                      # truncated/corrupt upload -> reported as missing
        return np.array([])


def streams_of(ed):
    return {
        "rgb": csv_ts(f"{ed}/rgb_head.csv"),
        "depth": csv_ts(f"{ed}/depth_head.csv"),
        "wristL": csv_ts(_first(f"{ed}/realsense/wrist_left.csv", f"{ed}/wrist_left.csv")),
        "wristR": csv_ts(_first(f"{ed}/realsense/wrist_right.csv", f"{ed}/wrist_right.csv")),
        "tacL": glove_ts(ed, "left"),
        "tacR": glove_ts(ed, "right"),
    }


def video_files(ed):
    return {
        "rgb_head.mp4": f"{ed}/rgb_head.mp4",
        "depth_head.mkv": f"{ed}/depth_head.mkv",
        "wrist_left.mp4": _first(f"{ed}/realsense/wrist_left.mp4", f"{ed}/wrist_left.mp4"),
        "wrist_right.mp4": _first(f"{ed}/realsense/wrist_right.mp4", f"{ed}/wrist_right.mp4"),
    }


def nearest_gap(ts, grid):
    """|nearest sample - tick| for every grid tick, and the chosen sample indices."""
    idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    take_left = (grid - ts[idx - 1]) <= (ts[idx] - grid)
    nidx = np.where(take_left, idx - 1, idx)
    return np.abs(ts[nidx] - grid), nidx


def check_recording(ed, align_thresh=DEFAULT_ALIGN_THRESH, max_hole_frac=DEFAULT_MAX_HOLE_FRAC,
                    max_gap_ms=DEFAULT_MAX_GAP_MS, align_tol_ms=DEFAULT_ALIGN_TOL_MS,
                    check_videos=True):
    """Apply the sync gate to one recording directory; returns a result dict."""
    S = streams_of(ed)
    present = {k: len(v) >= 2 for k, v in S.items()}
    missing_streams = [STREAM_NAMES[k] for k, ok in present.items() if not ok]
    missing_files = ([n for n, p in video_files(ed).items() if not (p and os.path.exists(p))]
                     if check_videos else [])
    dur_s = round(float(S["rgb"][-1] - S["rgb"][0]), 2) if present["rgb"] else None
    r = dict(missing_streams=missing_streams, missing_files=missing_files, dur_s=dur_s)
    if missing_streams:
        reason = f"missing timestamp streams: {missing_streams}"
        if missing_files:
            reason += f"; missing video files: {missing_files}"
        r.update(PASS=False, reason=reason, aligned_frac=0.0, n_holes=-1)
        return r
    t0 = max(v[0] for v in S.values())
    t1 = min(v[-1] for v in S.values())
    if t1 - t0 < 1.0:
        r.update(PASS=False, reason="common time window shorter than 1 s",
                 aligned_frac=0.0, n_holes=-1)
        return r

    grid = np.arange(t0, t1, ONE_FRAME)
    gaps, per = {}, {}
    for k, ts in S.items():
        g, nidx = nearest_gap(ts, grid)
        gaps[k] = g
        per[k] = dict(dup=float(np.mean(nidx[1:] == nidx[:-1])), holes=int(np.sum(g > ONE_FRAME)),
                      maxgap_ms=round(float(g.max() * 1000), 1))
    allgap = np.max(np.stack([gaps[k] for k in S]), axis=0)
    align_tol = align_tol_ms / 1000.0
    aligned_frac = float(np.mean(allgap <= align_tol))
    n_holes = int(np.sum(allgap > ONE_FRAME))
    hole_frac = n_holes / len(grid)
    maxgap_ms = float(allgap.max() * 1000)
    worst = max(per, key=lambda k: per[k]["maxgap_ms"])
    cover = bool(min(S["tacL"][0], S["tacR"][0]) <= S["rgb"][0] + 0.05 and
                 max(S["tacL"][-1], S["tacR"][-1]) >= S["rgb"][-1] - 0.05)
    checks = {"files": not missing_files, "aligned": aligned_frac >= align_thresh,
              "holes": hole_frac <= max_hole_frac, "blackout": maxgap_ms <= max_gap_ms,
              "cover": cover}

    reasons = []
    if missing_files:
        reasons.append(f"missing video files: {missing_files}")
    if not checks["aligned"]:
        reasons.append(f"aligned {aligned_frac:.1%} < {align_thresh:.0%} "
                       f"(tol {align_tol_ms:.1f}ms; worst stream: {STREAM_NAMES[worst]})")
    if not checks["holes"]:
        reasons.append(f"holes {hole_frac:.2%} > {max_hole_frac:.0%} "
                       f"({n_holes} ticks; worst stream: {STREAM_NAMES[worst]})")
    if not checks["blackout"]:
        reasons.append(f"blackout: worst gap {maxgap_ms:.0f}ms > {max_gap_ms:.0f}ms "
                       f"(worst stream: {STREAM_NAMES[worst]})")
    if not cover:
        reasons.append("glove window does not cover the camera window")
    r.update(PASS=all(checks.values()), reason="; ".join(reasons) or "ok", n_grid=len(grid),
             aligned_frac=round(aligned_frac, 4), n_holes=n_holes, hole_frac=round(hole_frac, 4),
             cover=cover, per_stream=per, maxgap_ms=round(maxgap_ms, 1), worst_stream=worst)
    return r


def task_of(ed):
    """Task name from task_info.json, else the folder name without a trailing rep number."""
    p = os.path.join(ed, "task_info.json")
    try:
        name = json.load(open(p)).get("name")
        if name:
            return str(name)
    except (OSError, ValueError, AttributeError):
        pass
    base = os.path.basename(ed)
    return re.sub(r"[0-9]+$", "", base).rstrip("_-") or base


def find_recordings(root):
    return sorted({os.path.dirname(p) for p in glob.glob(f"{root}/**/rgb_head.csv", recursive=True)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--root", required=True, help="batch root (searched recursively)")
    ap.add_argument("--align-tol-ms", type=float, default=DEFAULT_ALIGN_TOL_MS,
                    help="all six streams must be within this of a grid tick (default 17.5)")
    ap.add_argument("--align-thresh", type=float, default=DEFAULT_ALIGN_THRESH,
                    help="min fraction of aligned grid ticks (default 0.95)")
    ap.add_argument("--max-hole-frac", type=float, default=DEFAULT_MAX_HOLE_FRAC,
                    help="max fraction of hole ticks (default 0.01)")
    ap.add_argument("--max-gap-ms", type=float, default=DEFAULT_MAX_GAP_MS,
                    help="max single nearest-sample gap in ms (default 66)")
    ap.add_argument("--no-video-check", action="store_true",
                    help="skip the video-file existence check (e.g. when only timestamp files "
                         "were downloaded)")
    ap.add_argument("--json", default=None, help="write machine-readable results here")
    a = ap.parse_args()

    recs = find_recordings(a.root) if os.path.isdir(a.root) else []
    if not os.path.isdir(a.root):
        print(f"not a directory: {a.root}")
        return 1
    if not recs:
        print(f"no recording directories (with rgb_head.csv) under {a.root}")
        return 1
    print(f"recordings: {len(recs)}")
    print(f"gate: {GRID_HZ:.0f} Hz grid; all streams and videos present, glove covers cameras, "
          f">= {a.align_thresh:.0%} ticks within {a.align_tol_ms:.1f} ms, "
          f"holes <= {a.max_hole_frac:.0%}, worst gap <= {a.max_gap_ms:.0f} ms\n")
    print("%-38s %-4s %8s %6s %8s  %s" % ("recording", "PASS", "aligned", "holes", "maxgap", "reason"))
    print("-" * 110)
    results, per_task = {}, defaultdict(lambda: [0, 0])
    for ed in recs:
        nm = os.path.relpath(ed, a.root)
        r = check_recording(ed, a.align_thresh, a.max_hole_frac, a.max_gap_ms,
                            a.align_tol_ms, check_videos=not a.no_video_check)
        r["task"] = task_of(ed)
        results[nm] = r
        tp = per_task[r["task"]]
        tp[0] += r["PASS"]
        tp[1] += 1
        print("%-38s %-4s %7.1f%% %6s %6sms  %s" % (
            nm[:38], "ok" if r["PASS"] else "FAIL", r.get("aligned_frac", 0) * 100,
            r.get("n_holes", "-"), r.get("maxgap_ms", "-"), r["reason"]))
    n_pass = sum(r["PASS"] for r in results.values())
    print("-" * 110)
    print(f"\npass rate: {n_pass}/{len(recs)} = {100.0 * n_pass / len(recs):.0f}%")
    fails = [(nm, r["reason"]) for nm, r in results.items() if not r["PASS"]]
    if fails:
        print(f"\nfailures ({len(fails)}):")
        for nm, reason in fails:
            print(f"  {nm}: {reason}")
    print(f"\nper task (pass / total), {len(per_task)} tasks:")
    for t, (p, n) in sorted(per_task.items()):
        print(f"  {t}: {p}/{n}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"n_recordings": len(recs), "n_tasks": len(per_task), "n_pass": n_pass,
                       "align_tol_ms": a.align_tol_ms, "align_thresh": a.align_thresh,
                       "max_hole_frac": a.max_hole_frac, "max_gap_ms": a.max_gap_ms,
                       "results": results}, f, ensure_ascii=False, indent=1)
        print(f"\nJSON -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
