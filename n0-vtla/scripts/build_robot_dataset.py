"""Build a canonical robot dataset for exactly one split (train/val/holdout).

Converts every raw episode in the requested split independently with
robot_single_arm_adapter.convert() (using a single shared tactile normalization --
see fit_robot_tactile_norm.py, and docs/POST_TRAIN.md section 3.4
for why a per-episode ad hoc fit isn't enough beyond a single-episode check), then
merges the resulting per-source mini LeRobot datasets into one dataset with
globally renumbered episode/frame indices.

IMPORTANT: this builds ONE PHYSICAL DIRECTORY PER SPLIT, not one combined
directory with a "split" tag on each episode. n0vtla/training/data_loader.py's
LocalLeRobotV3Dataset has no concept of a split tag -- it loads every episode
under the given root unconditionally, and meta/info.json's "splits" field is
not read by it either. A combined directory tagged per episode would therefore
train on all of it, including val and holdout episodes. Run this once per split
and point VTLA_DATASET_PATH at the TRAIN output only for actual training.

A raw episode that produces no qualifying contiguous window (contiguous_runs
finds nothing) is skipped and logged, not silently dropped -- see the
printed/returned skip list.

Usage:
  python scripts/build_robot_dataset.py \
    /path/to/raw_release \
    SPLIT.json \
    /path/to/tactile_norm.json \
    /path/to/canonical_robot_train \
    --which train --workers 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import robot_single_arm_adapter as adapter
from itw_tactile_adapter import _info_json, _write_jsonl


def _convert_one(uuid: str, source: Path, source_output: Path, norm_path: Path, *,
                  task_name: str, task_description: str | None,
                  require_success_label: bool) -> tuple[str, str | None]:
    """Runs in a worker process (must be a top-level function to be picklable).
    Returns (uuid, None) on success or (uuid, error message) on a skip."""
    try:
        adapter.convert(source, source_output, allow_unverified_sync=True, norm_path=norm_path,
                         task_name=task_name, task_description=task_description,
                         require_success_label=require_success_label)
    except ValueError as exc:
        return uuid, str(exc)
    return uuid, None


def split_uuids(manifest: dict, which: str) -> list[str]:
    if which == "holdout":
        uuids = manifest["block_holdout_v1"]["holdout_episodes"]
    else:
        uuids = manifest[which]
    holdout = set(manifest["block_holdout_v1"]["holdout_episodes"])
    train = set(manifest["train"])
    val = set(manifest["val"])
    if not (holdout.isdisjoint(train) and holdout.isdisjoint(val) and train.isdisjoint(val)):
        raise ValueError("split_manifest's train/val/holdout are not mutually exclusive -- "
                          "fix the manifest before building any split from it")
    return sorted(set(uuids))


def merge(which: str, per_source: list[tuple[str, Path]], split_manifest: dict, output: Path) -> dict:
    output_data = output / "data/chunk-000"
    output_meta = output / "meta"
    output_data.mkdir(parents=True)
    output_meta.mkdir(parents=True)

    block_of = split_manifest["block_of_episode"]

    global_episode = 0
    global_index = 0
    episodes_out, stats_out, per_episode_meta = [], [], []
    data_bytes = video_bytes = 0
    video_keys: list[str] | None = None

    for uuid, source_output in per_source:
        local_episodes = [json.loads(l) for l in (source_output / "meta" / "episodes.jsonl").open()]
        local_stats = [json.loads(l) for l in (source_output / "meta" / "episodes_stats.jsonl").open()]
        local_info = json.loads((source_output / "meta" / "info.json").read_text())
        if video_keys is None:
            video_keys = [k for k in local_info["features"] if local_info["features"][k].get("dtype") == "video"]

        for local in local_episodes:
            local_idx = local["episode_index"]
            table = pq.read_table(source_output / "data/chunk-000" / f"episode_{local_idx:06d}.parquet")
            n = table.num_rows
            table = table.set_column(table.schema.get_field_index("episode_index"),
                                      "episode_index", pa.array([global_episode] * n, pa.int64()))
            table = table.set_column(table.schema.get_field_index("index"),
                                      "index", pa.array(range(global_index, global_index + n), pa.int64()))
            dst = output_data / f"episode_{global_episode:06d}.parquet"
            pq.write_table(table, dst, compression="zstd")
            data_bytes += dst.stat().st_size

            for key in video_keys:
                src_video = source_output / "videos/chunk-000" / key / f"episode_{local_idx:06d}.mp4"
                dst_dir = output / "videos/chunk-000" / key
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_video = dst_dir / f"episode_{global_episode:06d}.mp4"
                shutil.copy2(src_video, dst_video)
                video_bytes += dst_video.stat().st_size

            episodes_out.append(dict(episode_index=global_episode, tasks=local["tasks"], length=n,
                                      source_uuid=uuid, block=block_of.get(uuid, "unknown"), split=which))
            stat = local_stats[[s["episode_index"] for s in local_stats].index(local_idx)]
            stat["episode_index"] = global_episode
            stats_out.append(stat)
            global_episode += 1
            global_index += n

        per_episode_meta.append(dict(uuid=uuid, audit=json.loads((source_output / "meta" / "audit.json").read_text())))

    _write_jsonl(output_meta / "episodes.jsonl", episodes_out)
    _write_jsonl(output_meta / "episodes_stats.jsonl", stats_out)
    _write_jsonl(output_meta / "tasks.jsonl", [dict(task_index=0, task=adapter.TASK)])
    (output_meta / "per_source_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in per_episode_meta))
    info = _info_json(global_episode, global_index, data_bytes, video_bytes, video_keys or [])
    info["robot_type"] = "xarm6_revo2"
    # Informational only -- n0vtla/training/data_loader.py does not read "splits" at
    # all, it loads every episode under the given root. Physical separation (this
    # script builds one directory per split) is what actually keeps them apart.
    info["splits"] = {which: f"0:{global_episode}"}
    (output_meta / "info.json").write_text(json.dumps(info, indent=2))
    return dict(which=which, n_source_episodes=len(per_source), n_episodes=global_episode, n_frames=global_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("norm_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--which", choices=("train", "val", "holdout"), required=True,
                         help="Which mutually-exclusive partition of split_manifest to build. Run this "
                              "script once per split you need -- each writes a separate directory.")
    parser.add_argument("--scratch-dir", type=Path, default=None,
                         help="Where to write per-source intermediate conversions. "
                              "Defaults to a temp dir removed after merging.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Parallel worker processes for the independent per-episode "
                              "conversions (pure CPU/IO work -- video decode/encode, no GPU "
                              "involved). merge() still runs single-threaded afterwards.")
    parser.add_argument("--task-name", default=None,
                         help="If given, robot/manifest.json task.name must equal it for every "
                              "episode (None = no check).")
    parser.add_argument("--task-description", default=None,
                         help="LeRobot task string written into tasks.jsonl/episodes.jsonl. "
                              "Defaults to robot_single_arm_adapter.TASK.")
    parser.add_argument("--skip-success-label-check", action="store_true",
                         help="Skip the manifest trial.label=='success' gate -- use when the "
                              "manifest label is empty or not authoritative and admission is "
                              "decided by a separate list. "
                              "Only safe when split_manifest already lists ONLY admitted uuids.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = json.loads(args.split_manifest.read_text())
    all_uuids = split_uuids(manifest, args.which)

    scratch_ctx = tempfile.TemporaryDirectory() if args.scratch_dir is None else None
    scratch = args.scratch_dir if args.scratch_dir is not None else Path(scratch_ctx.name)
    scratch.mkdir(parents=True, exist_ok=True)

    jobs = {uuid: scratch / uuid for uuid in all_uuids}
    results: dict[str, str | None] = {}
    if args.workers <= 1:
        for uuid, source_output in jobs.items():
            _, error = _convert_one(uuid, args.raw_dir / uuid, source_output, args.norm_path,
                                     task_name=args.task_name, task_description=args.task_description,
                                     require_success_label=not args.skip_success_label_check)
            results[uuid] = error
            done = sum(1 for e in results.values() if e is None)
            print(f"{'SKIP ' + uuid + ': ' + error if error else 'converted ' + uuid} "
                  f"({done}/{len(all_uuids)} converted so far, {len(results) - done} skipped)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_convert_one, uuid, args.raw_dir / uuid, source_output, args.norm_path,
                                    task_name=args.task_name, task_description=args.task_description,
                                    require_success_label=not args.skip_success_label_check): uuid
                       for uuid, source_output in jobs.items()}
            for future in as_completed(futures):
                uuid, error = future.result()
                results[uuid] = error
                done = sum(1 for e in results.values() if e is None)
                print(f"{'SKIP ' + uuid + ': ' + error if error else 'converted ' + uuid} "
                      f"({done}/{len(all_uuids)} converted, {len(results) - done} skipped, "
                      f"{len(results)}/{len(all_uuids)} attempted)")

    per_source = [(uuid, jobs[uuid]) for uuid in all_uuids if results[uuid] is None]
    skipped = [dict(uuid=uuid, reason=results[uuid]) for uuid in all_uuids if results[uuid] is not None]

    summary = merge(args.which, per_source, manifest, args.output)
    summary["skipped"] = skipped
    (args.output / "meta" / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if scratch_ctx is not None:
        scratch_ctx.cleanup()


if __name__ == "__main__":
    main()
