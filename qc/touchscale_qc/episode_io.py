"""Readers for one recorded episode: tactile gloves, timestamp CSVs, and head depth.

Tactile NPZ (one per hand): `timestamps (N,)` (UNIX epoch seconds) plus, for each
pad slot `s`, `tactile_{s} (N, h_s, w_s)` (normal force) and optionally
`tf_tactile_x_{s}` / `tf_tactile_y_{s}` (shear). 15 slots are used:
0-5, 7-9, 11-13, 15-16 and 18; see `SLOTS` for the slot -> hand-part mapping.
"""
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

# slot -> (finger column 0..4 = thumb..pinky, segment row 0=tip..2=base, label); 18 = palm
SLOTS = {
    0: (0, 0, "Thumb tip"), 1: (0, 1, "Thumb mid"), 2: (0, 2, "Thumb base"),
    3: (1, 0, "Index tip"), 4: (1, 1, "Index mid"), 5: (1, 2, "Index base"),
    7: (2, 0, "Middle tip"), 8: (2, 1, "Middle mid"), 9: (2, 2, "Middle base"),
    11: (3, 0, "Ring tip"), 12: (3, 1, "Ring mid"), 13: (3, 2, "Ring base"),
    15: (4, 0, "Pinky tip"), 16: (4, 1, "Pinky mid"),
    18: (-1, -1, "Palm"),
}


def load_hand(npz_path):
    """Load one glove. Returns (timestamps, {slot: (N,h,w) force}, {slot: (shear_x, shear_y)})."""
    d = np.load(npz_path)
    ts = d["timestamps"].astype(float)
    slots, shear = {}, {}
    for s in SLOTS:
        k = f"tactile_{s}"
        if k in d.files:
            slots[s] = d[k].astype(np.float32)            # (Ns, h, w); Ns may be < len(ts)
            kx, ky = f"tf_tactile_x_{s}", f"tf_tactile_y_{s}"
            if kx in d.files and ky in d.files:
                fx = d[kx].astype(np.float32)
                fy = d[ky].astype(np.float32)
                n = len(fx)
                shear[s] = (fx.reshape(n, -1).mean(1), fy.reshape(n, -1).mean(1))
    return ts, slots, shear


def csv_ts(path):
    """Timestamps from a `frame_index,timestamp_s` CSV with a header row."""
    rows = [l.strip().split(",") for l in open(path).read().splitlines()[1:] if l.strip()]
    return np.array([float(r[1]) for r in rows], float)


def depth_reader(mkv, start_frame=0, fps=30.0):
    """Stream 16-bit depth frames from an FFV1 .mkv through an ffmpeg gray16 pipe.

    Returns (process, width, height, bytes_per_frame); read frames from
    `process.stdout`. FFV1 is all-intra, so input seeking is frame-accurate.
    """
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cap = cv2.VideoCapture(mkv)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    cmd = [exe, "-loglevel", "error", "-threads", "1"]
    if start_frame > 0:
        cmd += ["-ss", f"{start_frame / fps:.6f}"]
    cmd += ["-i", mkv, "-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return proc, w, h, w * h * 2


def colorize_depth(d16):
    """Percentile-normalised Turbo colouring of a uint16 depth frame (BGR)."""
    v = d16.astype(np.float32)
    m = v[v > 0]
    lo, hi = (np.percentile(m, 2), np.percentile(m, 98)) if m.size else (0, 1)
    v = np.clip((v - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d16 == 0] = (40, 40, 40)
    return img


def nearest(ts, t):
    """Index of the sample in sorted `ts` closest to time `t`."""
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    return i if (ts[i] - t) < (t - ts[i - 1]) else i - 1


def total_force(slots, n):
    """Per-frame sum of raw normal force over all pads of one hand."""
    tot = np.zeros(n)
    for a in slots.values():
        m = min(len(a), n)
        tot[:m] += a[:m].reshape(m, -1).sum(1)
    return tot
