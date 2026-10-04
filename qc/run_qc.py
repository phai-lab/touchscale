#!/usr/bin/env python3
"""Automatic tactile QC for a batch of episodes.

    python run_qc.py --data <batch_dir> --out <out_dir>

Every sub-directory of <batch_dir> is one episode and must contain (episodes with
any file missing are skipped and listed):

    rgb_head.mp4 / rgb_head.csv        head RGB and frame timestamps
    depth_head.mkv / depth_head.csv    head depth (FFV1, 16-bit)
    wrist_left.mp4 / wrist_left.csv    left wrist camera
    wrist_right.mp4 / wrist_right.csv  right wrist camera
    left_hand_data.npz                 left glove
    right_hand_data.npz                right glove
    task_info.json                     task metadata

Pipeline: integrity check -> render review video -> 3-step verdict -> results and
review material. Everything is cached; re-running skips finished work.
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from touchscale_qc import config as C
from touchscale_qc import integrity

REQUIRED = ["rgb_head.mp4", "rgb_head.csv", "depth_head.mkv", "depth_head.csv",
            "wrist_left.mp4", "wrist_left.csv", "wrist_right.mp4", "wrist_right.csv",
            "left_hand_data.npz", "right_hand_data.npz", "task_info.json"]


def scan(data_dir):
    """Split episode directories into complete and incomplete ones."""
    good, bad = [], []
    for name in sorted(os.listdir(data_dir)):
        ep = os.path.join(data_dir, name)
        if not os.path.isdir(ep):
            continue
        missing = [f for f in REQUIRED if not os.path.exists(os.path.join(ep, f))]
        (bad if missing else good).append((name, ep, missing))
    return good, bad


def run_integrity(good, jobs):
    issues = {}
    print(f"[integrity] checking {len(good)} episodes, {jobs} workers", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(integrity.check, ep): name for name, ep, _ in good}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = [f"integrity check crashed: {type(e).__name__}: {str(e)[:80]}"]
            if r:
                issues[futs[fut]] = r
    print(f"[integrity] {len(issues)} episodes with timestamp / frame-drop issues")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--data", required=True, help="batch directory; one sub-directory per episode")
    ap.add_argument("--out", default="qc_out", help="output directory (default: qc_out)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel workers for rendering and integrity checks (CPU-bound; default 4)")
    ap.add_argument("--qc-jobs", type=int, default=1,
                    help="parallel episodes for the model-based verdict (API-bound; 8-16 is typical)")
    ap.add_argument("--stride", type=int, default=2,
                    help="render every N-th frame; 2 = 15 fps (default 2)")
    ap.add_argument("--skip-render", action="store_true",
                    help="reuse existing videos in <out>/videos instead of rendering")
    ap.add_argument("--no-review", action="store_true", help="do not build review material")
    ap.add_argument("--integrity-only", action="store_true",
                    help="only run the code-only integrity check (no rendering, no model calls)")
    a = ap.parse_args()
    if not os.path.isdir(a.data):
        ap.error(f"--data is not a directory: {a.data}")
    if a.jobs < 1 or a.qc_jobs < 1 or a.stride < 1:
        ap.error("--jobs, --qc-jobs and --stride must be >= 1")

    os.makedirs(a.out, exist_ok=True)
    good, bad = scan(a.data)
    print(f"[scan] {len(good)} complete episodes, {len(bad)} incomplete")
    for name, _, missing in bad[:5]:
        print(f"       skip {name}: missing {', '.join(missing[:3])}")
    if not good:
        print("No complete episodes found; check the directory layout described in --help.")
        return 1

    # Outputs and caches are keyed by the first 8 characters of the episode name
    # (a UUID prefix in TouchScale), so those prefixes must be unique.
    prefixes = Counter(name[:8] for name, _, _ in good)
    clashes = sorted(p for p, k in prefixes.items() if k > 1)
    if clashes:
        print(f"Episode names must be unique in their first 8 characters; clashing prefixes: "
              f"{', '.join(clashes[:10])}. Rename the directories (UUIDs work).")
        return 1

    issues = run_integrity(good, a.jobs)
    with open(os.path.join(a.out, "integrity.json"), "w") as f:
        json.dump(issues, f, ensure_ascii=False, indent=1)
    if a.integrity_only:
        for name, r in sorted(issues.items()):
            print(f"\n{name}\n" + "\n".join(f"    {x}" for x in r))
        return 0

    try:
        C.require_api_key()                 # fail fast, before hours of rendering
    except RuntimeError as e:
        print(f"error: {e}")
        return 1
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    from touchscale_qc import render, verdict

    viz_dir = os.path.join(a.out, "videos")
    if a.skip_render:
        rendered = {n[:8]: os.path.join(viz_dir, f"{n[:8]}.mp4") for n, _, _ in good}
        rendered = {k: v for k, v in rendered.items() if os.path.exists(v)}
        print(f"[render] skipped; reusing {len(rendered)} existing videos")
    else:
        rendered = render.render([(n[:8], ep) for n, ep, _ in good], viz_dir,
                                 jobs=a.jobs, stride=a.stride)
        print(f"[render] {len(rendered)}/{len(good)} done")

    def qc_one(name, ep, viz):
        try:
            return name[:8], verdict.run_episode(ep, viz, name)
        except Exception as e:
            return name[:8], dict(verdict="ERROR", episode=name,
                                  error=f"{type(e).__name__}: {str(e)[:200]}")

    todo = []
    for name, ep, _ in good:
        if name[:8] in rendered:
            todo.append((name, ep, rendered[name[:8]]))
        else:
            print(f"[qc] {name[:8]} has no rendered video, skipped")

    results = {}
    workers = max(1, a.qc_jobs)
    print(f"[qc] {len(todo)} episodes to judge, {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(qc_one, *x) for x in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            short, r = fut.result()
            results[short] = r
            if r["verdict"] == "ERROR":
                print(f"[qc {i}/{len(todo)}] {short}  ERROR   {r['error'][:60]}", flush=True)
            else:
                why = "; ".join(d["reason"] for d in r["hands"].values() if d["reason"])
                print(f"[qc {i}/{len(todo)}] {short}  {r['verdict']:<7} "
                      f"noise {r['noise_seg']} seg / {r['noise_s']:5.1f}s "
                      f"/ {r['noise_pct']:3.0f}% of free time  {why[:60]}", flush=True)

    out_json = os.path.join(a.out, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    c = Counter(r["verdict"] for r in results.values())
    n = len(results)
    print(f"\n{'=' * 64}\n{n} episodes   REJECT {c['REJECT']}   REVIEW {c['REVIEW']}   "
          f"PASS {c['PASS']}   ERROR {c['ERROR']}")
    if n:
        auto = c["REJECT"] + c["PASS"]
        print(f"decided automatically {auto}/{n} = {100 * auto / n:.0f}%   "
              f"needs human review {c['REVIEW']}/{n} = {100 * c['REVIEW'] / n:.0f}%")
    print(f"results written to {out_json}")

    if not a.no_review and (c["REJECT"] or c["REVIEW"]):
        # results.json is already written; a failure here must not lose paid-for verdicts
        try:
            from touchscale_qc import review
            review.build(results, viz_dir, os.path.join(a.out, "review"), data_dir=a.data)
            print(f"review material written to {os.path.join(a.out, 'review')}/README.md")
        except Exception as e:
            print(f"[warn] building review material failed (verdicts unaffected): "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
