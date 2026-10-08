"""Dual-arm recording package -> N0-VTLA canonical LeRobot v3 rows (both arms filled).

Extends the single-arm `scripts/robot_single_arm_adapter.py` to the
canonical bimanual layout it reserves (n0vtla/policies/canonical_schema.py):

  [0:3]  left eef xyz (mm)      [3:9]   left eef rot6d (first two matrix columns)
  [9]    left gripper (unused, 0, masked)
  [10:13] right eef xyz (mm)    [13:19] right eef rot6d
  [19]   right gripper (unused, 0, masked)
  [20:26] right Revo2 6 motor targets (0..1000)  -- same slots as the single-arm releases
  [26:32] left Revo2 6 motor targets (0..1000)    -- the remaining reserved dims

Images: third_view <- rgb_head, left_wrist_view <- wrist_left, right_wrist_view <- wrist_right
(224x224 aspect-preserving letterbox, as the single-arm adapter). Tactile: the tactile glove on
the robot RIGHT hand is stored as left_hand_data.npz (crossed) -> right_wrist_right_tactile,
hand="right"; the glove on the robot LEFT hand is right_hand_data.npz -> left_wrist_left_tactile,
hand="left" (mirrored layout). Normalization slots: left hand 0..14, right hand 15..29.

State and action follow the single-arm contract: absolute last-issued commands (state = the
command in force at t, action = the commands at t..t+49; the loader's delta transform subtracts
the state). Each arm has its OWN clutch. An engaged arm needs a fresh accepted command
(<=150 ms) and hand request (<=150 ms) exactly as in the single-arm adapter. An arm that is NOT
engaged holds its last issued command, or, before its first command, its measured pose — what
the robot physically does — so both arms always carry valid targets and both are active in
action_mask (the training loss does not read the mask; zeros would teach "go to zero").
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from itw_pressure import PAD_IDS, SCHEMA, video_writer, write_pressure_video
from itw_tactile_adapter import (_fixed_size_list_array, _hf_schema_metadata, _info_json,
                                       _quantile_stats, _write_jsonl)

TASK = "Perform the task."   # fixed prompt (the policy is trained and served with this default prompt)
LAYOUT = "dual_eef_mm_columns6d_left_0_9_right_10_19_revo2_right_20_26_left_26_32_v1"
SIDES = ("right", "left")
ARM_SLOT = {"left": (0, 3, 9), "right": (10, 13, 19)}          # xyz lo, rot lo, rot hi
HAND_SLOT = {"right": (20, 26), "left": (26, 32)}
GLOVE_FILE = {"right": "left_hand_data.npz", "left": "right_hand_data.npz"}   # robot hand -> archive
TACTILE_KEY = {"right": "observation.image.right_wrist_right_tactile",
               "left": "observation.image.left_wrist_left_tactile"}
RGB_KEY = {"rgb_head": "observation.image.third_view", "wrist_left": "observation.image.left_wrist_view",
           "wrist_right": "observation.image.right_wrist_view"}
CMD_AGE_NS, TACTILE_AGE_NS, CAMERA_TOL_NS = 150_000_000, 50_000_000, 17_500_000


def causal_indices(source_ns, query_ns, max_age_ns=None):
    source = np.asarray(source_ns, np.int64)
    query = np.asarray(query_ns, np.int64)
    if not len(source):
        return np.zeros(len(query), np.int64), np.zeros(len(query), np.int64), np.zeros(len(query), bool)
    if np.any(np.diff(source) < 0):
        raise ValueError("Source timestamps must be nondecreasing")
    index = np.searchsorted(source, query, side="right") - 1
    age = query - source[np.maximum(index, 0)]
    valid = (index >= 0) & (age >= 0)
    if max_age_ns is not None:
        valid &= age <= max_age_ns
    return np.maximum(index, 0), age, valid


def contiguous_runs(valid, min_length=50):
    ids = np.flatnonzero(valid)
    return [g for g in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(g) >= min_length]


def pack_dual(arm_aa, hand):
    """arm_aa/hand: {side: (N,6)} absolute targets (mm, deg axis-angle / motor counts) -> (N,32)."""
    n = len(arm_aa["right"])
    out = np.zeros((n, 32), np.float32)
    for side in SIDES:
        arm, h = np.asarray(arm_aa[side], np.float64), np.asarray(hand[side], np.float64)
        if arm.shape != (n, 6) or h.shape != (n, 6) or not np.isfinite(arm).all() or not np.isfinite(h).all():
            raise ValueError(f"bad {side} targets")
        if np.any((h < 0) | (h > 1000)):
            raise ValueError(f"{side} motor target outside 0..1000")
        lo, rlo, rhi = ARM_SLOT[side]
        rot = Rotation.from_rotvec(np.deg2rad(arm[:, 3:])).as_matrix()
        out[:, lo:lo + 3] = arm[:, :3]
        out[:, rlo:rhi] = rot[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        a, b = HAND_SLOT[side]
        out[:, a:b] = h
    return out


def decode_dual(x):
    from n0vtla.policies.rotation_utils import rot6d_to_matrix
    x = np.asarray(x)
    arms, hands = {}, {}
    for side in SIDES:
        lo, rlo, rhi = ARM_SLOT[side]
        rot = rot6d_to_matrix(x[:, rlo:rhi].astype(np.float64))
        arms[side] = np.concatenate([x[:, lo:lo + 3], np.rad2deg(Rotation.from_matrix(rot).as_rotvec())], axis=1)
        a, b = HAND_SLOT[side]
        hands[side] = x[:, a:b].copy()
    return arms, hands


def action_mask(n):
    mask = np.zeros((n, 32), bool)
    mask[:, 0:9] = mask[:, 10:19] = mask[:, 20:32] = True
    return mask


def measured_aa(f, side):
    pos = f[f"obs/robot/{side}/tcp_pos"][:].astype(np.float64)
    q = f[f"obs/robot/{side}/tcp_quat"][:].astype(np.float64)        # w,x,y,z
    ok = np.isfinite(pos).all(1) & np.isfinite(q).all(1) & (np.linalg.norm(q, axis=1) > 1e-8)
    rv = np.zeros((len(q), 3))
    rv[ok] = np.rad2deg(Rotation.from_quat(q[ok][:, [1, 2, 3, 0]]).as_rotvec())
    return np.column_stack([pos, rv]), ok


def resolve_dual_window(source, expected_task=None):
    source = Path(source)
    manifest = json.loads((source / "robot/manifest.json").read_text())
    note = json.loads((source / "operator_note.json").read_text())
    if expected_task is not None and manifest["task"]["name"] != expected_task:
        raise ValueError(f"Expected task {expected_task!r}, got {manifest['task']['name']!r}")
    if note.get("success") is not True:
        raise ValueError("Expected a trial the operator marked successful")
    with h5py.File(source / "episode_30hz.h5", "r") as f:
        t = f["time/timestamp_ns"][:]
        valid = (t >= manifest["trial"]["engaged_host_ns"]) & (t < manifest["trial"]["released_host_ns"])
        engaged = {s: f[f"action/{s}/clutch"][:].astype(bool) for s in SIDES}
        valid &= engaged["right"] | engaged["left"]
        arm, hand, src = {}, {}, {}
        for s in SIDES:
            rows = [json.loads(l) for l in (source / f"robot/{s}/arm_cmd.jsonl").open()]
            cmds = [r for r in rows if r.get("accepted") is True and r.get("rc") == 0 and r.get("clutch")]
            hrows = [json.loads(l) for l in (source / f"robot/{s}/hand.jsonl").open()]
            hrows = [r for r in hrows if r.get("t_cmd_ns") is not None and r.get("target") is not None]
            ct = [r["host_timestamp_ns"] for r in cmds]
            ai, aa, av = causal_indices(ct, t, CMD_AGE_NS)
            zi, _, zv = causal_indices(ct, t)                                   # last command ever
            ht = [max(r["t_cmd_ns"], r["host_timestamp_ns"]) for r in hrows]
            hi, ha, hv = causal_indices(ht, t, CMD_AGE_NS)
            if hrows:
                age = t - np.array([r["t_cmd_ns"] for r in hrows], np.int64)[hi]
                hv &= (age >= 0) & (age <= CMD_AGE_NS)
            hzi, _, hzv = causal_indices(ht, t)
            meas, meas_ok = measured_aa(f, s)
            meas_hand = f[f"obs/robot/{s}/hand_pos"][:].astype(np.float64)
            targets = np.array([r["target_aa"] for r in cmds], np.float64) if cmds else np.zeros((1, 6))
            htargets = np.array([r["target"] for r in hrows], np.float64) if hrows else np.zeros((1, 6))
            e = engaged[s]
            # engaged: fresh issued command; not engaged: holding the last command, else the start pose
            a = np.where((e | zv)[:, None], targets[np.where(e, ai, zi)], meas)
            h = np.where((e | hzv)[:, None], htargets[np.where(e, hi, hzi)], meas_hand)
            ok = np.where(e, av & hv, (zv | meas_ok) & (hzv | np.isfinite(meas_hand).all(1)))
            valid &= ok & np.isfinite(a).all(1) & np.isfinite(h).all(1)
            arm[s], hand[s] = a, h
            src[s] = dict(engaged=e, fresh_cmd=ai, last_cmd=zi, hand_row=hi, last_hand=hzi,
                          held=~e, before_first_cmd=~e & ~zv)
        video = {}
        for name in RGB_KEY:
            csv = np.genfromtxt(source / f"{name}.csv", delimiter=",", names=True)
            key = next(k for k in ("timestamp_s", "epoch_s", "epoch", "timestamp") if k in csv.dtype.names)
            csv_t = np.rint(csv[key] * 1e9).astype(np.int64)
            ids = f[f"obs/video/{name}/src_idx"][:]
            if np.any(ids < 0) or np.any(ids >= len(csv_t)):
                raise ValueError(f"{name}: video index outside its timestamp table")
            valid &= f[f"valid/{name}"][:].astype(bool) & (np.abs(t - csv_t[ids]) <= CAMERA_TOL_NS)
            video[name] = ids
        tactile = {}
        for s in SIDES:
            with np.load(source / GLOVE_FILE[s], allow_pickle=False) as z:
                ti, _, tv = causal_indices(np.rint(z["timestamps"] * 1e9), t, TACTILE_AGE_NS)
            valid &= tv
            tactile[s] = ti
    return dict(manifest=manifest, t=t, valid=valid, runs=contiguous_runs(valid), arm=arm, hand=hand,
                src=src, video=video, tactile=tactile, engaged=engaged)


def fit_dual_pressure(source, window, rows):
    """Per-episode quick-check normalization for both hands (slots: left 0..14, right 15..29).
    A full run fits ONE normalization on the train split (fit_robot_dual_tactile_norm.py)."""
    baseline, scale, threshold = [0.0] * 30, [1.0] * 30, [0.1] * 30
    rng = np.random.default_rng(42)
    for s, offset in (("left", 0), ("right", 15)):
        frames = np.unique(window["tactile"][s][rows])
        chosen = frames[rng.integers(0, len(frames), size=min(4, len(frames)))]
        with np.load(source / GLOVE_FILE[s], allow_pickle=False) as z:
            for slot, pad in enumerate(PAD_IDS):
                a = np.asarray(z[f"tactile_{pad}"][chosen], np.float64).reshape(-1)
                lo, hi = np.percentile(a, [5, 99.9])
                sc = max(float(hi - lo), 1e-6)
                med = np.median(a)
                baseline[offset + slot], scale[offset + slot] = float(lo), sc
                threshold[offset + slot] = max(float(med + 4 * np.median(np.abs(a - med))), float(lo) + .1 * sc)
    positive = [x for x in scale if x > 1e-5]
    floor = max(float(np.median(positive)) * .05 if positive else 1e-3, 1e-5)
    scale = [max(x, floor) for x in scale]
    threshold = [max((th - b) / s, .1) for th, b, s in zip(threshold, baseline, scale)]
    return dict(schema=SCHEMA, fit_split="train", smoke_only=True, normal_baseline=baseline,
                normal_scale=scale, contact_threshold=threshold, seed=42, samples_per_recording=4)


def write_rgb(source_mp4, destination, indices):
    """Same 224 letterbox as itw_pressure.write_aligned_rgb, but seeks to the window start."""
    cap = cv2.VideoCapture(str(source_mp4))
    if not cap.isOpened() or np.any(np.diff(indices) < 0) or indices[0] < 0:
        raise ValueError(f"Invalid video/index mapping: {source_mp4}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    current, frame = int(indices[0]) - 1, None
    try:
        with video_writer(destination) as writer:
            for index in indices:
                while current < index:
                    ok, frame = cap.read()
                    current += 1
                    if not ok:
                        raise ValueError(f"Video shorter than timestamp indices: {source_mp4}")
                h, w = frame.shape[:2]
                ratio = 224 / max(h, w)
                resized = cv2.resize(frame, (max(1, round(w * ratio)), max(1, round(h * ratio))))
                canvas = np.zeros((224, 224, 3), np.uint8)
                y, x = (224 - resized.shape[0]) // 2, (224 - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return Path(destination).stat().st_size


def write_windows(source, window, groups, output, first_episode, first_index, norm):
    """Write each row group as one canonical episode; returns (episodes, stats, frames, bytes)."""
    source, output = Path(source), Path(output)
    rows = np.arange(len(window["t"]))
    actions = pack_dual({s: window["arm"][s][rows] for s in SIDES}, {s: window["hand"][s][rows] for s in SIDES})
    episodes, stats, total, nbytes = [], [], 0, 0
    for k, g in enumerate(groups):
        e, n = first_episode + k, len(g)
        x = actions[g]
        cols = {"observation.state": x, "action": x.copy(), "action_mask": action_mask(n),
                "timestamp": np.arange(n, dtype=np.float32) / 30, "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, e, np.int64),
                "index": np.arange(first_index + total, first_index + total + n, dtype=np.int64),
                "task_index": np.zeros(n, np.int64)}
        arrays = [_fixed_size_list_array(v, 32, pa.bool_() if c == "action_mask" else pa.float32())
                  if v.ndim == 2 else pa.array(v) for c, v in cols.items()]
        table = pa.Table.from_arrays(arrays, names=list(cols)).replace_schema_metadata(_hf_schema_metadata())
        dst = output / "data/chunk-000" / f"episode_{e:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst, compression="zstd")
        nbytes += dst.stat().st_size
        for name, key in RGB_KEY.items():
            nbytes += write_rgb(source / f"{name}.mp4", output / "videos/chunk-000" / key / f"episode_{e:06d}.mp4",
                                window["video"][name][g])
        for s in SIDES:
            nbytes += write_pressure_video(source / GLOVE_FILE[s],
                                           output / "videos/chunk-000" / TACTILE_KEY[s] / f"episode_{e:06d}.mp4",
                                           window["tactile"][s][g], norm, hand=s)
        episodes.append(dict(episode_index=e, tasks=[TASK], length=n))
        stats.append(dict(episode_index=e, stats={c: _quantile_stats(v) for c, v in cols.items()}))
        total += n
    return episodes, stats, total, nbytes


VIDEO_KEYS = list(RGB_KEY.values()) + [TACTILE_KEY[s] for s in SIDES]


def write_meta(output, episodes, stats, total, nbytes, robot_type):
    meta = Path(output) / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    _write_jsonl(meta / "episodes.jsonl", episodes)
    _write_jsonl(meta / "episodes_stats.jsonl", stats)
    _write_jsonl(meta / "tasks.jsonl", [dict(task_index=0, task=TASK)])
    info = _info_json(len(episodes), total, nbytes, 0, VIDEO_KEYS)
    info["robot_type"] = robot_type
    (meta / "info.json").write_text(json.dumps(info, indent=2))
