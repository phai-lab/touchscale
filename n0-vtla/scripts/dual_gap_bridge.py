"""Bridge short illegal gaps between legal rows of a dual-arm window (robot_dual_arm_adapter.resolve_dual_window).

The adapter marks a row illegal when an engaged arm/hand command is older than 150 ms, a glove sample
older than 50 ms, a camera is out of tolerance, or neither arm is engaged. A single such row cuts a
trajectory into separate episodes (and stretches under 50 rows are dropped). This fills a gap of at
most `max_gap` rows between two legal rows when at least one arm is engaged on every gap row and the
arm/hand targets there are finite. The physical robot holds its last command through such a gap, which
is what the resolved targets already contain; no value is invented. Gaps where neither arm is engaged
(operator released both clutches) are never filled.
"""
from __future__ import annotations

import numpy as np


def bridge_short_gaps(w: dict, max_gap: int) -> tuple[np.ndarray, int]:
    """Return (new_valid, n_rows_added). max_gap<=0 returns the window's own valid mask unchanged."""
    valid = np.asarray(w["valid"], bool).copy()
    if max_gap <= 0:
        return valid, 0
    anyeng = np.asarray(w["engaged"]["right"], bool) | np.asarray(w["engaged"]["left"], bool)
    finite = np.ones(len(valid), bool)
    for s in ("left", "right"):
        finite &= np.isfinite(np.asarray(w["arm"][s], float)).all(1) & np.isfinite(np.asarray(w["hand"][s], float)).all(1)
    ids = np.flatnonzero(valid)
    added = 0
    for a, b in zip(ids[:-1], ids[1:]):
        n = b - a - 1
        if 0 < n <= max_gap:
            gap = slice(a + 1, b)
            if anyeng[gap].all() and finite[gap].all():
                valid[gap] = True
                added += n
    return valid, added
