"""Render a side-by-side preview video for one canonical robot episode.

head cam | wrist cam | tactile pressure | arm xyz trace. This is the cheapest
end-to-end check that the tactile stream (see docs/POST_TRAIN.md, section 2, glove
files) is *aligned* to real contact events, not just *live* -- something a raw
pad-std or encoded-video pixel-statistics comparison cannot show by itself.

Usage:
  python scripts/visualize_robot_episode.py \
    /path/to/canonical_robot_train \
    --episode-index 0 \
    --output /path/to/viz/episode_000000.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from itw_pressure import video_writer

PANEL = 224
TRACE_H = 80
LABEL_COLOR = (255, 255, 255)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.putText(panel, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_COLOR, 1, cv2.LINE_AA)
    return panel


def _read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _trace_panel(xyz: np.ndarray, width: int) -> np.ndarray:
    """A scrolling |dxyz| step-size trace, one bar per frame, showing motion
    smoothness (step size per tick, in mm)."""
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    step = np.concatenate([[0.0], step])
    scale = max(float(step.max()), 1.0)

    def render(up_to: int) -> np.ndarray:
        panel = np.zeros((TRACE_H, width, 3), np.uint8)
        n = min(up_to + 1, width)
        s = step[max(0, up_to - width + 1):up_to + 1]
        for x, v in enumerate(s):
            h = int(np.clip(v / scale, 0, 1) * (TRACE_H - 12))
            cv2.line(panel, (x, TRACE_H - 4), (x, TRACE_H - 4 - h), (0, 200, 255), 1)
        cv2.putText(panel, f"|d xyz| step, mm (scale max={scale:.1f})", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, LABEL_COLOR, 1, cv2.LINE_AA)
        return panel
    return render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    row = next(e for e in episodes if e["episode_index"] == args.episode_index)
    idx = args.episode_index

    streams = {
        "head": args.dataset_root / f"videos/chunk-000/observation.image.third_view/episode_{idx:06d}.mp4",
        "wrist": args.dataset_root / f"videos/chunk-000/observation.image.right_wrist_view/episode_{idx:06d}.mp4",
        "tactile": args.dataset_root / f"videos/chunk-000/observation.image.right_wrist_right_tactile/episode_{idx:06d}.mp4",
    }
    frames = {name: _read_video_frames(path) for name, path in streams.items()}
    n = min(len(v) for v in frames.values())

    table = pq.read_table(args.dataset_root / f"data/chunk-000/episode_{idx:06d}.parquet",
                           columns=["observation.state"])
    state = np.asarray(table["observation.state"].to_pylist(), np.float32)
    xyz = state[:n, 10:13]
    trace_render = _trace_panel(xyz, PANEL * 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # imageio/libx264 (yuv420p, baseline profile) rather than cv2.VideoWriter's
    # mp4v fourcc, which most browsers/players can't decode; this is the same
    # video_writer() helper used by itw_pressure.write_aligned_rgb/write_pressure_video.
    with video_writer(args.output) as writer:
        for i in range(n):
            panels = [
                _label(frames["head"][i], f"head  ep{idx} f{i}/{n}"),
                _label(frames["wrist"][i], "wrist"),
                _label(frames["tactile"][i], "tactile (left_hand_data.npz, labeled right)"),
            ]
            top = np.concatenate(panels, axis=1)
            bottom = trace_render(i)
            canvas = np.concatenate([top, bottom], axis=0)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    print(f"wrote {n} frames -> {args.output}  "
          f"(source_uuid={row['source_uuid']}, block={row['block']}, split={row['split']})")


if __name__ == "__main__":
    main()
