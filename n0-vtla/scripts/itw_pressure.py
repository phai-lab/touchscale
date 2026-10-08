"""Pressure-only normalization and alignment of human (TouchScale) glove recordings.

Statistics are per hand/pad, fitted on train recordings only, and fixed across episodes. The per-pad schema
(`SCHEMA`) uses one scale per pad; the per-task schema (`SCHEMA_PER_TASK`) uses one scale per task name, see below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

PAD_IDS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18)
HANDS = ("left", "right")
RGB_VIEWS = ("rgb_head", "wrist_left", "wrist_right")
SCHEMA = "touchscale_pad30_train_only_normalization"
# Maximum offset between any sensor stream (cameras, both gloves) and the master timeline for a frame to count as
# aligned. Applied uniformly to every stream.
QC_ALIGNMENT_NS = 17_500_000
# An episode is accepted when at least this fraction of frames of every stream is within QC_ALIGNMENT_NS (a pass-rate
# threshold, not zero tolerance for any single frame). See aligned_timeline below.
QC_PASS_RATE = 0.95
# Per-task force scale: same per-pad baseline, but the scale is looked up by task name instead of fit per pad.
# A per-pad scale distorts the relative force between pads for a corpus of many different tasks, and a single
# shared/global scale over-compresses light-touch tasks (task scales can be much smaller than the global one), so one
# scale per task is used instead. `scripts/build_per_task_scale_normalization.py` builds such a file from your data.
SCHEMA_PER_TASK = "touchscale_pad30_train_only_per_task_scale_normalization"


def load_normalization(path):
    result = json.loads(Path(path).read_text())
    schema = result.get("schema")
    if schema not in (SCHEMA, SCHEMA_PER_TASK):
        raise ValueError(
            f"Expected a train-only pad30 normalization file "
            f"(schema {SCHEMA!r} or {SCHEMA_PER_TASK!r}), got {schema!r}"
        )
    if result.get("fit_split") != "train":
        raise ValueError("Expected a train-only pad30 normalization file (fit_split == 'train')")
    for key in ("normal_baseline", "normal_scale", "contact_threshold"):
        array = np.asarray(result[key], dtype=np.float64)
        if array.shape != (30,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}: expected 30 finite numbers")
    if np.any(np.asarray(result["normal_scale"]) <= 0):
        raise ValueError("All scales must be positive")
    if schema == SCHEMA_PER_TASK:
        task_scale = result.get("task_scale")
        if not isinstance(task_scale, dict) or not task_scale:
            raise ValueError("Per-task-scale normalization file must have a non-empty task_scale table")
        for name, scale in task_scale.items():
            if not (isinstance(scale, (int, float)) and np.isfinite(scale) and scale > 0):
                raise ValueError(f"Invalid task_scale for {name!r}: must be a positive finite number")
        default_scale = result.get("default_scale")
        if not (isinstance(default_scale, (int, float)) and np.isfinite(default_scale) and default_scale > 0):
            raise ValueError("Per-task-scale normalization file must have a positive finite default_scale")
    return result


def fit_normalization(episodes, *, samples_per_recording=4, seed=42):
    if samples_per_recording < 1:
        raise ValueError("samples_per_recording must be positive")
    rng = np.random.default_rng(seed)
    values = [[] for _ in range(30)]
    for episode in episodes:
        for hand_i, hand in enumerate(HANDS):
            with np.load(Path(episode) / f"{hand}_hand_data.npz", allow_pickle=False) as z:
                n = len(z["timestamps"])
                if n == 0:
                    raise ValueError(f"Empty tactile stream: {episode}/{hand}")
                take = rng.integers(0, n, size=min(samples_per_recording, n))
                for slot, pad in enumerate(PAD_IDS):
                    normal = np.asarray(z[f"tactile_{pad}"][take], np.float32)
                    finite = normal[np.isfinite(normal)]
                    if len(finite):
                        values[hand_i * 15 + slot].append(finite.reshape(-1))
    baselines, scales, thresholds = [], [], []
    for pad, pieces in enumerate(values):
        if not pieces:
            raise ValueError(f"No finite training values for pad {pad}")
        n = np.concatenate(pieces).astype(np.float64)
        base, high = np.percentile(n, [5.0, 99.9])
        scale = max(float(high - base), 1e-6)
        med = float(np.median(n))
        mad = float(np.median(np.abs(n - med)))
        threshold = max(med + 4 * mad, float(base) + 0.10 * scale)
        baselines.append(float(base))
        scales.append(scale)
        thresholds.append(float(threshold))
    positive = [x for x in scales if x > 1e-5]
    floor = max(float(np.median(positive)) * 0.05 if positive else 1e-3, 1e-5)
    scales = [max(x, floor) for x in scales]
    thresholds = [max((t - b) / s, 0.10) for t, b, s in zip(thresholds, baselines, scales)]
    return dict(normal_baseline=baselines, normal_scale=scales, contact_threshold=thresholds,
                schema=SCHEMA, fit_split="train")


def normalize_pressure(raw, norm, hand, pad, *, task_name=None):
    """Normalize raw pressure of one pad to the fixed `[-1, 8]` range.

    `task_name` only matters for a SCHEMA_PER_TASK `norm` (ignored otherwise). When `norm` has a `task_scale`
    table, ALL 30 pads share one scale for a given task (force magnitude varies by TASK, not by pad, so per-pad
    indexing into task_scale would defeat the purpose). `task_name=None` (no task_info.json, or no "name" field)
    and an explicit but unmatched name are treated identically -- both fall back to `default_scale`, and a
    missing task_name never raises. Callers are responsible for passing the task name; it is advisable to check
    the match rate against the task_scale table before a real run.
    """
    index = HANDS.index(hand) * 15 + PAD_IDS.index(int(pad))
    raw = np.asarray(raw, np.float32)
    if not np.isfinite(raw).all():
        raise ValueError(f"Nonfinite pressure: {hand}/pad{pad}; repair or exclude before conversion")
    if "task_scale" in norm:
        scale = norm["task_scale"].get(task_name, norm["default_scale"]) if task_name is not None else norm["default_scale"]
    else:
        scale = norm["normal_scale"][index]
    return np.clip((raw - norm["normal_baseline"][index]) / max(scale, 1e-6), -1.0, 8.0)


def pressure_rgb(normal):
    # Fixed transport mapping, not a second fitted normalization. Zero maps to 28.
    # Replicated channels keep the reconstruction head's RGB mean pressure-only.
    gray = np.rint((np.clip(normal, -1, 8) + 1) * (255 / 9)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def nearest_indices(source, query):
    source = np.asarray(source, np.int64)
    if source.ndim != 1 or len(source) < 2 or np.any(np.diff(source) < 0):
        raise ValueError("Timestamps must be nondecreasing and contain at least two samples")
    right = np.clip(np.searchsorted(source, query), 1, len(source) - 1)
    left = right - 1
    index = np.where(query - source[left] <= source[right] - query, left, right)
    return index, np.abs(source[index] - query)


def aligned_timeline(episode):
    episode = Path(episode)
    streams, frame_ids = {}, {}
    for view in RGB_VIEWS:
        with (episode / f"{view}.csv").open() as f:
            rows = list(csv.DictReader(f, skipinitialspace=True))
        seconds = np.array([float(row["timestamp_s"]) for row in rows])
        if not np.isfinite(seconds).all():
            raise ValueError(f"Nonfinite camera timestamps: {view}")
        frame_ids[view] = np.array([int(row["frame_index"]) for row in rows], np.int64)
        streams[view] = np.rint(seconds * 1e9).astype(np.int64)
    for hand in HANDS:
        with np.load(episode / f"{hand}_hand_data.npz", allow_pickle=False) as z:
            seconds = z["timestamps"]
            if not np.isfinite(seconds).all():
                raise ValueError(f"Nonfinite tactile timestamps: {hand}")
            streams[hand] = np.rint(seconds * 1e9).astype(np.int64)
    if any(len(t) < 2 for t in streams.values()):
        raise ValueError("Each stream must contain at least two timestamps")
    start, end = max(t[0] for t in streams.values()), min(t[-1] for t in streams.values())
    count = max(0, int(np.ceil((end - start) * 30 / 1e9)))
    master = start + np.rint(np.arange(count) * 1e9 / 30).astype(np.int64)
    master = master[master < end]
    if len(master) <= 50:
        raise ValueError("Common camera/glove coverage is too short for H=50")
    mapping, audit = {"master_timestamp_ns": master}, {}
    for name, timestamps in streams.items():
        index, error = nearest_indices(timestamps, master)
        within_tolerance = error <= QC_ALIGNMENT_NS
        pass_rate = float(within_tolerance.mean())
        # A single bad tick does not fail the whole episode: only when fewer than QC_PASS_RATE of the frames
        # are within QC_ALIGNMENT_NS is the episode rejected. Frames that fail are NOT dropped from the episode;
        # they are kept and flagged via `<name>_within_tolerance`, so a consumer can mask individual bad frames
        # instead of losing the recording.
        if pass_rate < QC_PASS_RATE:
            raise ValueError(
                f"{episode.name}/{name}: only {pass_rate:.1%} of frames within "
                f"{QC_ALIGNMENT_NS / 1e6} ms (need >={QC_PASS_RATE:.0%})"
            )
        mapping[name + "_index"] = index
        mapping[name + "_error_ns"] = error
        mapping[name + "_within_tolerance"] = within_tolerance
        if name in frame_ids:
            mapping[name + "_frame_index"] = frame_ids[name][index]
        audit[name] = {
            "max_error_ms": float(error.max() / 1e6),
            "invalid_frames": int((~within_tolerance).sum()),
            "pass_rate": pass_rate,
        }
    return mapping, audit


def video_writer(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=30, codec="libx264", pixelformat="yuv420p",
                             ffmpeg_params=["-profile:v", "baseline", "-crf", "18", "-movflags", "+faststart"])


def write_aligned_rgb(source, destination, indices):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened() or np.any(np.diff(indices) < 0) or indices[0] < 0:
        raise ValueError(f"Invalid video/index mapping: {source}")
    current, frame = -1, None
    try:
        with video_writer(destination) as writer:
            for index in indices:
                while current < index:
                    ok, frame = cap.read()
                    current += 1
                    if not ok:
                        raise ValueError(f"Video shorter than timestamp indices: {source}")
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


def load_hand_pressure_arrays(npz_path, norm, *, hand, task_name=None):
    """Per-pad normalized pressure for one hand, for the WHOLE episode (all frames).

    Shared by the offline video writer and the online dataset (n0vtla/training/
    itw_online_dataset.py) so both rasterize from the exact same normalized values.
    `task_name` is forwarded to normalize_pressure (only used when `norm` is a
    per-task-scale file; see its docstring).
    """
    with np.load(npz_path, allow_pickle=False) as z:
        return {
            str(p): normalize_pressure(z[f"tactile_{p}"], norm, hand, p, task_name=task_name)
            for p in PAD_IDS
        }


def rasterize_pressure_frame(arrays, index, *, hand, layout="hand"):
    """One 224x224 hand-pressure canvas at `index` into per-pad normalized arrays.

    `arrays` is `load_hand_pressure_arrays`'s return value (or an equivalent dict of
    pad-id -> normalized-pressure-over-time array). Shared with `write_pressure_video`
    so the online dataset renders pixel-identical frames without going through a video
    file.
    """
    from itw_tactile_adapter import TACTILE_SLOT_LAYOUT, _mirror_box, _put_resized

    canvas = np.zeros((224, 224, 3), np.uint8)
    for j, (pad, array) in enumerate(arrays.items()):
        rgb = pressure_rgb(array[index])
        if layout == "hand":
            box = TACTILE_SLOT_LAYOUT[pad]
            if hand == "left":
                box = _mirror_box(box)
            if pad == "18":
                rgb = np.rot90(rgb)
        else:
            row, col = divmod(j, 4)
            box = ((col + .5) / 4, (row + .5) / 4, .25, .25)
        _put_resized(canvas, rgb, box)
    return canvas


def write_pressure_video(npz_path, destination, indices, norm, *, hand, layout="hand", task_name=None):
    arrays = load_hand_pressure_arrays(npz_path, norm, hand=hand, task_name=task_name)
    with video_writer(destination) as writer:
        for index in indices:
            writer.append_data(rasterize_pressure_frame(arrays, index, hand=hand, layout=layout))
    return Path(destination).stat().st_size


def main():
    parser = argparse.ArgumentParser(description="Fit train-only pressure statistics")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-recording", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new normalization version; do not overwrite statistics")
    manifest = json.loads(args.split_manifest.read_text())
    train_ids = {r["recording_uuid"] for r in manifest["records"] if r["split"] == "train"}
    episodes = [p for p in sorted(args.raw_root.iterdir()) if p.is_dir() and p.name in train_ids
                and all((p / f"{h}_hand_data.npz").is_file() for h in HANDS)]
    if not episodes:
        raise ValueError("No local train episodes matched the split manifest")
    norm = fit_normalization(episodes, samples_per_recording=args.samples_per_recording, seed=args.seed)
    norm["provenance"] = dict(raw_root=str(args.raw_root), episode_ids=[p.name for p in episodes],
                              samples_per_recording=args.samples_per_recording, seed=args.seed,
                              split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(norm, indent=2))
    print(f"Fitted {len(episodes)} train episodes -> {args.output}")
    print(f"Scale range: {min(norm['normal_scale']):.6g} .. {max(norm['normal_scale']):.6g}")


if __name__ == "__main__":
    main()
