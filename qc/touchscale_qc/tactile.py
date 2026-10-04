"""Per-hand tactile signals used by the QC logic.

All "has signal" decisions use the same `ZERO_FLOOR` as the renderer, so what the
model and reviewers see in the video matches what the numeric checks measure.
"""
import os

import numpy as np

from . import config as C
from .episode_io import SLOTS, load_hand

ZERO_FLOOR = C.ZERO_FLOOR
PADS = sorted(SLOTS)
NAME = {s: SLOTS[s][2] for s in SLOTS}                     # slot -> "Thumb tip", ...
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
PARTS = FINGERS + ["palm"]
# thumb -> index -> middle -> ring -> pinky -> palm, each finger tip -> base
ROW_ORDER = sorted(PADS, key=lambda s: (SLOTS[s][0] if SLOTS[s][0] >= 0 else 99, SLOTS[s][1]))


def part_of(slot):
    """Hand part ("thumb", ..., "palm") a pad slot belongs to."""
    return NAME[slot].split()[0].lower()


def load(ep, hand):
    """Load `<ep>/<hand>_hand_data.npz` -> (timestamps, slots, shear)."""
    return load_hand(os.path.join(ep, f"{hand}_hand_data.npz"))


def pad_series(slots, n=None):
    """Visible force per pad per frame: taxels with |force| < ZERO_FLOOR count as 0.

    Returns (pad ids in ROW_ORDER, array of shape (n_pads, n_frames)); each value
    is the sum over the pad's taxels.
    """
    T = n or min(len(a) for a in slots.values())
    out = []
    for s in ROW_ORDER:
        if s not in slots:
            out.append(np.zeros(T))
            continue
        a = slots[s][:T].reshape(T, -1).copy()
        a[np.abs(a) < ZERO_FLOOR] = 0.0
        out.append(a.sum(1))
    return ROW_ORDER, np.stack(out)


def finger_force(ep, hand):
    """(time axis from 0, {part: per-frame force}, slots, n_frames).

    Pads are merged per finger by taking the max over tip/mid/base; taxels below
    ZERO_FLOOR are ignored, matching the renderer.
    """
    ts, slots, _ = load(ep, hand)
    n = min(len(v) for v in slots.values())
    t = ts[:n] - ts[0]
    per = {p: np.zeros(n) for p in PARTS}
    for s in PADS:
        a = slots[s][:n].reshape(n, -1)
        a = np.where(a >= ZERO_FLOOR, a, 0.0).sum(1)
        per[part_of(s)] = np.maximum(per[part_of(s)], a)
    return t, per, slots, n


def peak_per_part(ep, hand):
    """Peak visible force of each hand part over the whole episode."""
    _, per, _, _ = finger_force(ep, hand)
    return {p: float(v.max()) for p, v in per.items()}
