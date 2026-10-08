"""Sanity-check a state/action norm_stats.json produced by compute_canonical_norm.py (single-arm layout).

Usage:
  python scripts/check_norm_stats.py assets/vtla_tactile_posttrain/<asset id>/norm_stats.json

Checks that the RIGHT-arm action statistics (dims 10:19) look like deltas relative to the current state
(small xyz, near-identity rot6d), not absolute poses, i.e. that the rotation-aware delta mode was used.
"""
import json
import sys

d = json.load(open(sys.argv[1]))
state = d["norm_stats"]["state"]["mean"][10:19]
action = d["norm_stats"]["actions"]["mean"][10:19]
print("state[10:19] mean:", state)
print("action[10:19] mean:", action)
xyz_small = all(abs(v) < 50 for v in action[0:3])
rot_near_identity = abs(action[3] - 1) < 0.3 and abs(action[7] - 1) < 0.3
print("action xyz looks like a small delta (not an absolute position in mm):", xyz_small)
print("action rot6d looks near-identity (correct-delta signature):", rot_near_identity)
