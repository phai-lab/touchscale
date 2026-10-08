"""Summarize how the built dual-arm dataset was cut into episodes (from meta/per_source_audit.jsonl).

Each source trajectory is split into contiguous legal runs; every run becomes one dataset episode,
and the loader's tactile "baseline" is the FIRST frame of that episode, not of the trajectory.
Usage: python scripts/analyze_dual_dataset_segments.py DATASET_ROOT
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
audit = [json.loads(l) for l in (root / "meta/per_source_audit.jsonl").open()]
runs = np.array([a["n_runs"] for a in audit])
rows = sum(a["rows"] for a in audit)
legal = sum(a["legal_rows"] for a in audit)
first = [a["run_lengths"][0] for a in audit]
later = [x for a in audit for x in a["run_lengths"][1:]]
alls = first + later
print(f"trajectories {len(audit)}  episodes(runs) {int(runs.sum())}  runs/trajectory min/med/max {runs.min()}/{np.median(runs):.0f}/{runs.max()}")
print(f"rows {rows}  legal {legal} ({legal / rows:.1%})  dropped {rows - legal}")
print(f"episodes starting at a trajectory's first legal run: {len(first)} ({sum(first)} frames); later runs: {len(later)} ({sum(later)} frames = {sum(later) / sum(alls):.1%} of frames)")
pc = lambda x: np.percentile(x, [0, 10, 50, 90, 100]).astype(int).tolist()
print("run length pct [min,10,50,90,max] first:", pc(first), " later:", pc(later) if later else None)
print("trajectories with >1 run:", int((runs > 1).sum()))
both = sum(a["rows_both_engaged"] for a in audit)
print(f"legal rows with both arms engaged: {both / legal:.1%}")
for s in ("left", "right"):
    print(f"{s} arm: holding-last-command rows {sum(a['rows_arm_holding'][s] for a in audit) / legal:.1%}, before-first-command rows {sum(a['rows_before_first_cmd'][s] for a in audit) / legal:.1%}")
