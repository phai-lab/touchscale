"""How many episodes / frames the dual-arm builder yields for different bridge-gap sizes.

Usage: python scripts/analyze_dual_bridging.py RAW_DIR SPLIT_JSON [N ...]   (default N: 0 1 2 3 5 8 15 30)
Uses only the non-video files of each package (H5, csv, glove npz, robot jsonl, manifests).
"""
import json
import sys
from pathlib import Path

import numpy as np

import dual_gap_bridge as gb
import robot_dual_arm_adapter as nd

raw, split = Path(sys.argv[1]), Path(sys.argv[2])
ns = [int(x) for x in sys.argv[3:]] or [0, 1, 2, 3, 5, 8, 15, 30]
uuids = json.loads(split.read_text())["train"]
wins = [nd.resolve_dual_window(raw / u) for u in uuids]
print(f"{'N':>3} {'episodes':>8} {'frames':>7} {'bridged':>7} {'short-dropped':>13} {'runs/traj min/med/max':>21} {'traj>1run':>9} {'frames in later runs':>20}")
for n in ns:
    eps = frames = bridged = short = later = 0
    rpt = []
    for w in wins:
        v, add = gb.bridge_short_gaps(w, n)
        runs = nd.contiguous_runs(v)
        kept = sum(len(r) for r in runs)
        eps += len(runs); frames += kept; bridged += add
        short += int(v.sum()) - kept
        rpt.append(len(runs))
        later += sum(len(r) for r in runs[1:])
    rpt = np.array(rpt)
    print(f"{n:>3} {eps:>8} {frames:>7} {bridged:>7} {short:>13} {f'{rpt.min()}/{int(np.median(rpt))}/{rpt.max()}':>21} {int((rpt > 1).sum()):>9} {later / frames:>19.1%}")
