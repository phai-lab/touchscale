#!/usr/bin/env python3
"""Recording-length distribution for a batch.

Measures each recording's length (head-RGB timestamp span, falling back to the
`duration` field of task_info.json), buckets the lengths, and draws a bar chart in
which buckets at or above a recommended cap are red. The cap is a recommendation,
not a pass/fail criterion: long recordings are usable, but short single-action
clips dilute contact-rich segments less.

    python duration_dist.py --root <batch_dir> --out durations.png [--bucket 5] [--cap 30]
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from check_sync import csv_ts, find_recordings, task_of  # noqa: E402


def rec_duration(ed):
    """Recording length in seconds, or None if it cannot be determined."""
    ts = csv_ts(os.path.join(ed, "rgb_head.csv"))
    if len(ts) >= 2:
        return float(ts[-1] - ts[0])
    try:
        return float(json.load(open(os.path.join(ed, "task_info.json"))).get("duration") or 0) or None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="batch root (searched recursively)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--bucket", type=float, default=5.0, help="bucket width in seconds (default 5)")
    ap.add_argument("--cap", type=float, default=30.0, help="recommended max length in s (default 30)")
    ap.add_argument("--json", default=None, help="write per-recording durations here")
    a = ap.parse_args()

    durs, names, over = [], {}, []
    for ed in find_recordings(a.root):
        d = rec_duration(ed)
        if d is None:
            continue
        nm = os.path.basename(ed)
        durs.append(d)
        names[nm] = d
        if d > a.cap:
            over.append((nm, task_of(ed), d))
    if not durs:
        print(f"no recordings with a readable duration under {a.root}")
        return 1
    durs = np.array(durs)
    n, n_over = len(durs), int(np.sum(durs > a.cap))

    print(f"recordings: {n}")
    print(f"length (s): min={durs.min():.0f}  median={np.median(durs):.0f}  mean={durs.mean():.0f}  "
          f"p90={np.percentile(durs, 90):.0f}  max={durs.max():.0f}")
    print(f"total: {durs.sum() / 60:.0f} min = {durs.sum() / 3600:.2f} h")
    print(f"longer than {a.cap:.0f}s: {n_over}/{n} = {100 * n_over / n:.0f}%\n")

    top = (int(durs.max() // a.bucket) + 1) * a.bucket
    edges = np.arange(0, top + a.bucket, a.bucket)
    counts, _ = np.histogram(durs, bins=edges)
    lefts = edges[:-1]
    for lo, c in zip(lefts, counts):
        if c:
            print(f"  {int(lo):3d}-{int(lo + a.bucket):3d}s : {c}{'  (over cap)' if lo >= a.cap else ''}")
    if over:
        print(f"\nlongest recordings over {a.cap:.0f}s (top 20 of {len(over)}):")
        for nm, task, d in sorted(over, key=lambda x: -x[2])[:20]:
            print(f"  {nm}  {d:5.0f}s  {task}")

    fig, ax = plt.subplots(figsize=(max(9.0, len(lefts) * 0.30), 4.8))
    colors = ["#c0392b" if lo >= a.cap else "#27ae60" for lo in lefts]
    ax.bar(lefts + a.bucket / 2, counts, width=a.bucket * 0.92, color=colors,
           edgecolor="white", linewidth=0.4)
    ax.axvline(a.cap, color="#c0392b", ls="--", lw=1.6, label=f"recommended cap {a.cap:.0f}s")
    for lo, c in zip(lefts, counts):
        if c:
            ax.text(lo + a.bucket / 2, c + max(counts) * 0.01, str(int(c)),
                    ha="center", va="bottom", fontsize=7)
    ax.set_xlabel(f"recording length (s), {a.bucket:.0f}s buckets")
    ax.set_ylabel("recordings")
    ax.set_xticks(edges[::max(1, len(edges) // 24)])
    ax.set_ylim(0, max(counts) * 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"Recording length ({n} recordings, median {np.median(durs):.0f}s, "
                 f"{n_over} = {100 * n_over / n:.0f}% over {a.cap:.0f}s)", fontsize=12, weight="bold")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(f"\nchart -> {a.out}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"n": n, "cap_s": a.cap, "bucket_s": a.bucket, "n_over": n_over,
                       "median_s": float(np.median(durs)), "durations": names},
                      f, ensure_ascii=False, indent=1)
        print(f"json  -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
