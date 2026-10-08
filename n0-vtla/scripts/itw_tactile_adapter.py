#!/usr/bin/env python3
"""Convert ITW pressure arrays into canonical N0-VTLA tactile videos.

This is intentionally a small adapter, not a training-ready converter.  It takes
already-downloaded ITW episode folders and writes a LeRobot-like canonical
dataset with:

* head/wrist RGB and tactile resampled to a common 30 Hz timeline
* pressure-only hand videos with fixed train-only normalization
* zero state/action placeholders so the N0-VTLA data loader can inspect shapes

ITW has human hand/glove signals rather than robot EEF state/actions, so the
result should be used to validate tactile encoding and data plumbing only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import hashlib
import math
from pathlib import Path
import shutil
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CANONICAL_ACTION_DIM = 32
DATASET_FPS = 30.0
CHUNK_SIZE = 1000
RGB_KEYS = {
    "observation.image.third_view": "rgb_head.mp4",
    "observation.image.left_wrist_view": "wrist_left.mp4",
    "observation.image.right_wrist_view": "wrist_right.mp4",
}
TACTILE_KEYS = {
    "observation.image.left_wrist_left_tactile": "left_hand_data.npz",
    "observation.image.right_wrist_right_tactile": "right_hand_data.npz",
}
TACTILE_SLOT_LAYOUT = {
    "0": (0.18, 0.20, 0.10, 0.16),
    "1": (0.22, 0.34, 0.10, 0.16),
    "2": (0.27, 0.48, 0.10, 0.16),
    "3": (0.40, 0.11, 0.10, 0.20),
    "4": (0.40, 0.30, 0.10, 0.18),
    "5": (0.40, 0.48, 0.10, 0.18),
    "7": (0.52, 0.06, 0.10, 0.22),
    "8": (0.52, 0.27, 0.10, 0.19),
    "9": (0.52, 0.47, 0.10, 0.19),
    "11": (0.64, 0.10, 0.10, 0.20),
    "12": (0.64, 0.30, 0.10, 0.18),
    "13": (0.64, 0.48, 0.10, 0.18),
    "15": (0.76, 0.20, 0.10, 0.18),
    "16": (0.76, 0.39, 0.10, 0.18),
    "18": (0.50, 0.73, 0.34, 0.22),
}
NEUTRAL_TAC_RGB = np.array([0, 127, 127], dtype=np.uint8)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _fixed_size_list_array(values: np.ndarray, width: int, value_type: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1).tolist(), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _hf_schema_metadata() -> dict[bytes, bytes]:
    features = {
        "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "action_mask": {"feature": {"dtype": "bool", "_type": "Value"}, "length": 32, "_type": "Sequence"},
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}


def _quantile_stats(values: np.ndarray) -> dict[str, list[float | int | bool]]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr64 = arr.astype(np.float64, copy=False)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr64.mean(axis=0).tolist(),
        "std": arr64.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _read_task(ep_dir: Path) -> str:
    path = ep_dir / "task_info.json"
    if not path.exists():
        return "Perform the task."
    info = json.loads(path.read_text(encoding="utf-8"))
    steps = info.get("steps") or []
    if steps:
        return str(steps[0])
    return str(info.get("name") or "Perform the task.")


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise ValueError(f"empty video: {path}")
    return n


def _load_video_epochs(ep_dir: Path, stream: str, fallback_len: int) -> np.ndarray:
    csv_path = ep_dir / f"{stream}.csv"
    if csv_path.exists():
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        if data.size > 0:
            names = data.dtype.names or ()
            for key in ("epoch", "epoch_s", "timestamp", "timestamp_s", "time"):
                if key in names:
                    arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
                    if arr.size:
                        return arr
    return np.arange(fallback_len, dtype=np.float64) / DATASET_FPS


def _pad_ids(npz: Mapping[str, np.ndarray]) -> list[str]:
    ids = []
    for key in npz:
        if key.startswith("tactile_"):
            ids.append(key.split("_", 1)[1])
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def _robust_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    if not arrays:
        return 0.0, 1.0
    flat = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays])
    lo, hi = np.nanpercentile(flat, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _to_u8_unsigned(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    y = (np.asarray(x, dtype=np.float32) - lo) / (hi - lo)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _to_u8_signed(x: np.ndarray, mag: float) -> np.ndarray:
    if mag <= 0 or not np.isfinite(mag):
        mag = 1.0
    y = 127.5 + np.asarray(x, dtype=np.float32) / mag * 127.5
    return np.clip(y, 0, 255).astype(np.uint8)


def _put_resized(canvas: np.ndarray, image: np.ndarray, box: tuple[float, float, float, float]) -> None:
    height, width = canvas.shape[:2]
    cx, cy, bw, bh = box
    x0 = int(round((cx - bw / 2.0) * width))
    x1 = int(round((cx + bw / 2.0) * width))
    y0 = int(round((cy - bh / 2.0) * height))
    y1 = int(round((cy + bh / 2.0) * height))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = cv2.resize(image, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def _mirror_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = box
    return (1.0 - cx, cy, bw, bh)


def _slot_rgb(
    npz: Mapping[str, np.ndarray],
    pid: str,
    idx: int,
    pressure_limits: tuple[float, float],
    shear_mag: float,
) -> np.ndarray:
    pressure = _to_u8_unsigned(npz[f"tactile_{pid}"][idx], *pressure_limits)
    sx_key, sy_key = f"tf_tactile_x_{pid}", f"tf_tactile_y_{pid}"
    sx = _to_u8_signed(npz[sx_key][idx], shear_mag) if sx_key in npz else np.full_like(pressure, 127)
    sy = _to_u8_signed(npz[sy_key][idx], shear_mag) if sy_key in npz else np.full_like(pressure, 127)
    return np.stack([pressure, sx, sy], axis=-1)


def _rasterize_tactile_npz(
    npz_path: Path,
    frame_epochs: np.ndarray,
    out_path: Path,
    *,
    tile: int = 32,
    layout: str = "grid",
    canvas_size: int = 224,
) -> int:
    # NPZ indexing decompresses a whole array. Cache each array once per episode,
    # rather than decompressing every pad/channel again for every output frame.
    with np.load(npz_path) as archive:
        npz = {key: archive[key] for key in archive.files
               if key == "timestamps" or key.startswith(("tactile_", "tf_tactile_"))}
    tac_epochs = np.asarray(npz["timestamps"], dtype=np.float64)
    if not len(tac_epochs) or not np.isfinite(tac_epochs).all() or np.any(np.diff(tac_epochs) < 0):
        raise ValueError(f"invalid/nonmonotonic tactile timestamps: {npz_path}")
    if not np.isfinite(frame_epochs).all() or np.any(np.diff(frame_epochs) < 0):
        raise ValueError("invalid/nonmonotonic video timestamps")
    if frame_epochs[-1] < tac_epochs[0] or frame_epochs[0] > tac_epochs[-1]:
        raise ValueError(f"video and tactile clocks do not overlap: {npz_path}")
    pad_ids = _pad_ids(npz)
    if not pad_ids:
        raise ValueError(f"no tactile_* arrays in {npz_path}")

    sample_ids = pad_ids[: min(len(pad_ids), 16)]
    pressure_limits = _robust_limits([npz[f"tactile_{pid}"][:: max(1, len(tac_epochs) // 200)] for pid in sample_ids])
    shear_arrays = []
    for pid in sample_ids:
        for axis in ("x", "y"):
            key = f"tf_tactile_{axis}_{pid}"
            if key in npz:
                shear_arrays.append(npz[key][:: max(1, len(tac_epochs) // 200)])
    shear_mag = float(np.nanpercentile(np.abs(np.concatenate([a.reshape(-1) for a in shear_arrays])), 99.0)) if shear_arrays else 1.0

    grid_cols = 4
    grid_rows = math.ceil(len(pad_ids) / grid_cols)
    frames = []
    nearest = np.searchsorted(tac_epochs, frame_epochs, side="left")
    nearest = np.clip(nearest, 0, len(tac_epochs) - 1)
    prev = np.clip(nearest - 1, 0, len(tac_epochs) - 1)
    choose_prev = np.abs(tac_epochs[prev] - frame_epochs) < np.abs(tac_epochs[nearest] - frame_epochs)
    nearest[choose_prev] = prev[choose_prev]

    for idx in nearest:
        if layout == "hand":
            canvas = np.broadcast_to(NEUTRAL_TAC_RGB, (canvas_size, canvas_size, 3)).copy()
            mirror = "left_hand" in npz_path.name
            for pid in pad_ids:
                if pid not in TACTILE_SLOT_LAYOUT:
                    continue
                box = _mirror_box(TACTILE_SLOT_LAYOUT[pid]) if mirror else TACTILE_SLOT_LAYOUT[pid]
                tile_img = _slot_rgb(npz, pid, idx, pressure_limits, shear_mag)
                if pid == "18":
                    tile_img = np.rot90(tile_img)
                _put_resized(canvas, tile_img, box)
        else:
            canvas = np.broadcast_to(NEUTRAL_TAC_RGB, (grid_rows * tile, grid_cols * tile, 3)).copy()
            for j, pid in enumerate(pad_ids):
                r, c = divmod(j, grid_cols)
                tile_img = _slot_rgb(npz, pid, idx, pressure_limits, shear_mag)
                tile_img = cv2.resize(tile_img, (tile, tile), interpolation=cv2.INTER_NEAREST)
                canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = tile_img
        frames.append(canvas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path,
        np.asarray(frames),
        fps=DATASET_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=[
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
        ],
    )
    return out_path.stat().st_size


def _video_feature() -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(DATASET_FPS),
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _write_parquet(path: Path, *, length: int, episode_index: int, global_start: int, task_index: int = 0) -> int:
    state = np.zeros((length, CANONICAL_ACTION_DIM), dtype=np.float32)
    action = np.zeros((length, CANONICAL_ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros((length, CANONICAL_ACTION_DIM), dtype=bool)
    timestamp = np.arange(length, dtype=np.float32) / np.float32(DATASET_FPS)
    frame_index = np.arange(length, dtype=np.int64)
    index = np.arange(global_start, global_start + length, dtype=np.int64)
    table = pa.Table.from_arrays(
        [
            _fixed_size_list_array(state, 32, pa.float32()),
            _fixed_size_list_array(action, 32, pa.float32()),
            _fixed_size_list_array(action_mask, 32, pa.bool_()),
            pa.array(timestamp.tolist(), type=pa.float32()),
            pa.array(frame_index.tolist(), type=pa.int64()),
            pa.array([episode_index] * length, type=pa.int64()),
            pa.array(index.tolist(), type=pa.int64()),
            pa.array([task_index] * length, type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "action_mask",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    ).replace_schema_metadata(_hf_schema_metadata())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path.stat().st_size


def _copy_video(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def _info_json(total_episodes: int, total_frames: int, data_bytes: int, video_bytes: int, video_keys: list[str]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [32], "names": None},
        "action": {"dtype": "float32", "shape": [32], "names": None},
        "action_mask": {"dtype": "bool", "shape": [32], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in video_keys:
        features[key] = _video_feature()
    return {
        "codebase_version": "v2.1",
        "robot_type": "itw_tactile",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": math.ceil(data_bytes / (1024 * 1024)),
        "video_files_size_in_mb": math.ceil(video_bytes / (1024 * 1024)),
        "fps": DATASET_FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def main() -> None:
    from itw_pressure import aligned_timeline, load_normalization, write_aligned_rgb, write_pressure_video

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tactile-layout", choices=["grid", "hand"], default="hand")
    parser.add_argument("--normalization", type=Path, required=True,
                        help="Fixed pad30 train-only pressure statistics JSON")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--extra-raw-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest and --split must be supplied together")
    norm = load_normalization(args.normalization)

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output_dir}; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    episodes = [
        p for root in [args.raw_root, *args.extra_raw_root] for p in sorted(root.iterdir())
        if p.is_dir()
        and all((p / name).exists() for name in (*RGB_KEYS.values(), *TACTILE_KEYS.values()))
    ]
    rejected = []
    if args.split_manifest:
        split_data = json.loads(args.split_manifest.read_text())
        allowed = {r["recording_uuid"] for r in split_data["records"] if r["split"] == args.split}
        accepted = []
        for episode in episodes:
            if episode.name not in allowed:
                continue
            try:
                aligned_timeline(episode)
            except (ValueError, KeyError, OSError) as exc:
                rejected.append({"episode": episode.name, "reason": str(exc)})
                continue
            accepted.append(episode)
            if len(accepted) == args.max_episodes:
                break
        episodes = accepted
        if len(episodes) < args.max_episodes:
            (args.output_dir / "selection_failure.json").write_text(json.dumps({
                "split": args.split, "accepted": [str(p) for p in episodes], "rejected": rejected,
            }, indent=2))
            raise ValueError(f"Only {len(episodes)} aligned {args.split} episodes available")
    else:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        raise SystemExit(f"no complete episodes found under {args.raw_root}")

    video_keys = [*RGB_KEYS.keys(), *TACTILE_KEYS.keys()]
    episodes_rows = []
    episodes_stats_rows = []
    source_map = []
    data_bytes = 0
    video_bytes = 0
    total_frames = 0
    tasks = []
    alignment_reports = []

    for ep_idx, ep_dir in enumerate(episodes):
        chunk = ep_idx // CHUNK_SIZE
        mapping, alignment = aligned_timeline(ep_dir)
        n_frames = len(mapping["master_timestamp_ns"])
        task = _read_task(ep_dir)
        if task not in tasks:
            tasks.append(task)
        task_index = tasks.index(task)
        audit_dir = args.output_dir / "alignment"
        audit_dir.mkdir(exist_ok=True)
        np.savez_compressed(audit_dir / f"episode_{ep_idx:06d}.npz", **mapping)
        alignment_reports.append(dict(episode_index=ep_idx, source=ep_dir.name, streams=alignment))

        parquet_path = args.output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        data_bytes += _write_parquet(parquet_path, length=n_frames, episode_index=ep_idx,
                                    global_start=total_frames, task_index=task_index)

        for key, filename in RGB_KEYS.items():
            dst = args.output_dir / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
            video_bytes += write_aligned_rgb(ep_dir / filename, dst, mapping[Path(filename).stem + "_frame_index"])
        for key, filename in TACTILE_KEYS.items():
            dst = args.output_dir / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
            hand = "left" if filename.startswith("left") else "right"
            video_bytes += write_pressure_video(ep_dir / filename, dst, mapping[hand + "_index"],
                                                norm, hand=hand, layout=args.tactile_layout)

        timestamp = np.arange(n_frames, dtype=np.float32) / np.float32(DATASET_FPS)
        frame_index = np.arange(n_frames, dtype=np.int64)
        episodes_rows.append({"episode_index": ep_idx, "tasks": [task], "length": n_frames})
        episodes_stats_rows.append(
            {
                "episode_index": ep_idx,
                "stats": {
                    "observation.state": _quantile_stats(np.zeros((n_frames, 32), dtype=np.float32)),
                    "action": _quantile_stats(np.zeros((n_frames, 32), dtype=np.float32)),
                    "action_mask": _quantile_stats(np.zeros((n_frames, 32), dtype=bool)),
                    "timestamp": _quantile_stats(timestamp),
                    "frame_index": _quantile_stats(frame_index),
                    "episode_index": _quantile_stats(np.full(n_frames, ep_idx, dtype=np.int64)),
                    "index": _quantile_stats(np.arange(total_frames, total_frames + n_frames, dtype=np.int64)),
                    "task_index": _quantile_stats(np.full(n_frames, task_index, dtype=np.int64)),
                },
            }
        )
        source_map.append({"episode_index": ep_idx, "source_episode": ep_dir.name, "length": n_frames})
        total_frames += n_frames

    meta = args.output_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    _write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
    _write_jsonl(meta / "episodes.jsonl", episodes_rows)
    _write_jsonl(meta / "episodes_stats.jsonl", episodes_stats_rows)
    info = _info_json(len(episodes), total_frames, data_bytes, video_bytes, video_keys)
    info["total_tasks"] = len(tasks)
    (meta / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (meta / "source_episodes.json").write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (meta / "tactile_normalization.json").write_text(json.dumps(norm, indent=2))
    (meta / "selection.json").write_text(json.dumps({
        "split": args.split, "rejected": rejected,
        "raw_roots": [str(p) for p in [args.raw_root, *args.extra_raw_root]],
        "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
        if args.split_manifest else None,
    }, indent=2))
    (meta / "tactile_encoding.json").write_text(json.dumps({
        "version": "pressure_touchscale_v1", "channels": "R=G=B=pressure_gray", "shear_used": False,
        "normalized_range": [-1, 8], "gray_mapping": "round((clip(normal,-1,8)+1)*255/9)",
        "zero_gray": 28, "background_rgb": [0, 0, 0], "lossy_video": True,
        "normalization_sha256": hashlib.sha256(args.normalization.read_bytes()).hexdigest(),
        "alignment": alignment_reports,
    }, indent=2))
    print(f"wrote {len(episodes)} episodes / {total_frames} frames to {args.output_dir}")
    print(f"video keys: {', '.join(video_keys)}")
    print("note: state/action are zero placeholders; the output carries tactile/RGB videos only")


if __name__ == "__main__":
    main()
