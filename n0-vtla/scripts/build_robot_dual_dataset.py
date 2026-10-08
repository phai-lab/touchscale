"""Build the canonical LeRobot dataset for a dual-arm robot-data release.

Dual-arm counterpart of build_robot_dataset.py: converts every package with
robot_dual_arm_adapter.write_windows (both arms filled in the canonical 32-dim layout, 3 RGB + 2
tactile views), one worker per package, then merges the per-package outputs into one dataset
with globally renumbered episode/frame indices. Each contiguous legal run (>=50 rows) is one
episode, exactly as the single-arm builder does.

This builds one split at a time (pass the SPLIT_JSON of the split you want). Statistics
(tactile normalization -> fit_robot_dual_tactile_norm.py, state/action norm ->
compute_canonical_norm.py --robot canonical_dual_arm) must be fit on this same set.

Usage:
  python scripts/build_robot_dual_dataset.py RAW_DIR SPLIT_JSON TACTILE_NORM.json OUT_DIR --workers 16
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import dual_gap_bridge as gb
import robot_dual_arm_adapter as nd
from itw_pressure import load_normalization
from itw_tactile_adapter import _info_json, _write_jsonl


def _convert_one(uuid: str, source: Path, scratch: Path, norm_path: Path,
                 bridge_gap_rows: int = 0) -> tuple[str, str | None, dict | None]:
    try:
        w = nd.resolve_dual_window(source)
        runs_before = len(w["runs"])
        w["valid"], bridged = gb.bridge_short_gaps(w, bridge_gap_rows)
        if bridge_gap_rows > 0:
            w["runs"] = nd.contiguous_runs(w["valid"])
        if not w["runs"]:
            return uuid, "no legal 50-row dual-arm window", None
        norm = load_normalization(norm_path)
        episodes, stats, frames, nbytes = nd.write_windows(source, w, w["runs"], scratch, 0, 0, norm)
        nd.write_meta(scratch, episodes, stats, frames, nbytes, "xarm6_revo2_dual")
        legal = w["valid"]
        audit = dict(
            uuid=uuid, rows=int(len(w["t"])), bridge_gap_rows=bridge_gap_rows, bridged_rows=int(bridged),
            n_runs_without_bridging=runs_before, legal_rows=int(legal.sum()), n_runs=len(w["runs"]),
            run_lengths=[int(len(g)) for g in w["runs"]], frames_written=int(frames),
            rows_both_engaged=int((legal & w["engaged"]["right"] & w["engaged"]["left"]).sum()),
            rows_arm_holding={s: int((legal & w["src"][s]["held"]).sum()) for s in nd.SIDES},
            rows_before_first_cmd={s: int((legal & w["src"][s]["before_first_cmd"]).sum()) for s in nd.SIDES},
        )
        return uuid, None, audit
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return uuid, f"{type(exc).__name__}: {exc}", None


def merge(per_source: list[tuple[str, Path, dict]], output: Path) -> dict:
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)
    ep_out, st_out, audits = [], [], []
    g_ep = g_idx = data_bytes = video_bytes = 0
    for uuid, src, audit in per_source:
        local = [json.loads(l) for l in (src / "meta/episodes.jsonl").open()]
        stats = {s["episode_index"]: s for s in (json.loads(l) for l in (src / "meta/episodes_stats.jsonl").open())}
        for ep in local:
            li = ep["episode_index"]
            table = pq.read_table(src / "data/chunk-000" / f"episode_{li:06d}.parquet")
            n = table.num_rows
            table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index",
                                     pa.array([g_ep] * n, pa.int64()))
            table = table.set_column(table.schema.get_field_index("index"), "index",
                                     pa.array(range(g_idx, g_idx + n), pa.int64()))
            dst = output / "data/chunk-000" / f"episode_{g_ep:06d}.parquet"
            pq.write_table(table, dst, compression="zstd")
            data_bytes += dst.stat().st_size
            for key in nd.VIDEO_KEYS:
                d = output / "videos/chunk-000" / key
                d.mkdir(parents=True, exist_ok=True)
                v = d / f"episode_{g_ep:06d}.mp4"
                shutil.move(src / "videos/chunk-000" / key / f"episode_{li:06d}.mp4", v)
                video_bytes += v.stat().st_size
            ep_out.append(dict(episode_index=g_ep, tasks=ep["tasks"], length=n, source_uuid=uuid, split="train"))
            st = stats[li]
            st["episode_index"] = g_ep
            st_out.append(st)
            g_ep += 1
            g_idx += n
        audits.append(audit)
    meta = output / "meta"
    _write_jsonl(meta / "episodes.jsonl", ep_out)
    _write_jsonl(meta / "episodes_stats.jsonl", st_out)
    _write_jsonl(meta / "tasks.jsonl", [dict(task_index=0, task=nd.TASK)])
    _write_jsonl(meta / "per_source_audit.jsonl", audits)
    info = _info_json(g_ep, g_idx, data_bytes, video_bytes, nd.VIDEO_KEYS)
    info["robot_type"] = "xarm6_revo2_dual"
    info["layout"] = nd.LAYOUT
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    return dict(n_source_episodes=len(per_source), n_episodes=g_ep, n_frames=g_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("split_json", type=Path)
    ap.add_argument("norm_path", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scratch-dir", type=Path, default=None)
    ap.add_argument("--bridge-gap-rows", type=int, default=0,
                    help="Fill illegal gaps of at most this many rows between legal rows when an arm is engaged "
                         "(see dual_gap_bridge.py). 0 = off (default).")
    ap.add_argument("--limit", type=int, default=0, help="Convert only the first N train uuids (quick pipeline check).")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    split = json.loads(args.split_json.read_text())
    uuids = sorted(set(split["train"]))
    if args.limit:
        uuids = uuids[:args.limit]
    scratch_ctx = tempfile.TemporaryDirectory(dir=args.scratch_dir) if args.scratch_dir else tempfile.TemporaryDirectory()
    scratch = Path(scratch_ctx.name)
    jobs = {u: scratch / u for u in uuids}
    results: dict[str, tuple[str | None, dict | None]] = {}

    def report(uuid, err, audit):
        results[uuid] = (err, audit)
        ok = sum(1 for e, _ in results.values() if e is None)
        msg = f"SKIP {uuid}: {err}" if err else f"converted {uuid} ({audit['n_runs']} runs, {audit['frames_written']} frames)"
        print(f"{msg}  [{ok} ok, {len(results) - ok} skipped, {len(results)}/{len(uuids)}]", flush=True)

    if args.workers <= 1:
        for u in uuids:
            report(*_convert_one(u, args.raw_dir / u, jobs[u], args.norm_path, args.bridge_gap_rows))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_convert_one, u, args.raw_dir / u, jobs[u], args.norm_path, args.bridge_gap_rows) for u in uuids]
            for f in as_completed(futs):
                report(*f.result())

    per_source = [(u, jobs[u], results[u][1]) for u in uuids if results[u][0] is None]
    skipped = [dict(uuid=u, reason=results[u][0]) for u in uuids if results[u][0] is not None]
    summary = merge(per_source, args.output)
    summary["skipped"] = skipped
    lens = np.array([e["length"] for e in map(json.loads, (args.output / "meta/episodes.jsonl").open())])
    summary["episode_length_min_med_max"] = [int(lens.min()), int(np.median(lens)), int(lens.max())]
    (args.output / "meta/build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    scratch_ctx.cleanup()


if __name__ == "__main__":
    main()
