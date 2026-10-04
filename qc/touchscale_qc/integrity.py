"""Per-episode file integrity checks (code only, no model calls).

Checks:
  * video frame count vs. number of rows in the timestamp CSV
  * abnormally large frame intervals (dropped frames)
  * consistent duration across the camera streams
  * gaps in the tactile sample stream
"""
import os
import subprocess

import numpy as np

CAMERAS = ["rgb_head", "depth_head", "wrist_left", "wrist_right"]


def nframes(path):
    """Frame count from container metadata (fast), falling back to decoding (slow)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        pass
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def check(ep):
    """Return a list of human-readable issues for one episode (empty = clean)."""
    out, ts = [], {}
    for s in CAMERAS:
        csv = os.path.join(ep, f"{s}.csv")
        vid = os.path.join(ep, f"{s}.mp4")
        if not os.path.exists(vid):
            vid = os.path.join(ep, f"{s}.mkv")
        if not os.path.exists(csv) or not os.path.exists(vid):
            out.append(f"{s}: missing file")
            continue
        t = np.loadtxt(csv, delimiter=",", skiprows=1)[:, 1]
        ts[s] = t
        n = nframes(vid)
        if n < 0:
            out.append(f"{s}: could not read the frame count (unreadable video?)")
        elif abs(n - len(t)) > 2:
            out.append(f"{s}: video has {n} frames vs {len(t)} timestamp rows (diff {n - len(t)})")
        d = np.diff(t)
        med = np.median(d)
        big = np.where(d > med * 2.5)[0]
        if len(big):
            out.append(f"{s}: {len(big)} abnormal frame interval(s), max {d[big].max()*1000:.0f}ms "
                       f"(median {med*1000:.0f}ms), first at t={t[big[0]] - t[0]:.1f}s")
    if len(ts) > 1:
        durs = {s: v[-1] - v[0] for s, v in ts.items()}
        if max(durs.values()) - min(durs.values()) > 1.0:
            out.append("stream durations differ: " + ", ".join(f"{s}={d:.1f}s" for s, d in durs.items()))
    for hand in ["left", "right"]:
        p = os.path.join(ep, f"{hand}_hand_data.npz")
        if not os.path.exists(p):
            out.append(f"{hand}_hand: missing file")
            continue
        t = np.load(p)["timestamps"]
        d = np.diff(t)
        big = np.where(d > np.median(d) * 3)[0]
        if len(big):
            out.append(f"{hand}_hand tactile: {len(big)} gap(s), longest {d[big].max()*1000:.0f}ms")
    return out
