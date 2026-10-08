"""Convert a train / validation split file into the split manifest the dataset builders read.

Input: any JSON with top-level ``train`` and ``validation`` lists of episode ids (uuids), and optionally a ``test`` list:

    {"train": ["<uuid>", ...], "validation": ["<uuid>", ...], "test": []}

Output (what ``build_robot_dataset.py::split_uuids()`` expects): ``train``, ``val``,
``block_holdout_v1.holdout_episodes`` (= the ``test`` list, empty when there is none) and ``block_of_episode``
(every episode id maps to itself; it is only used for the builder's disjointness bookkeeping).

The script refuses overlapping splits and duplicate ids. If your recording tool stores the split in another shape,
adapt ``translate`` or write the four keys above yourself.

Usage:
  python scripts/convert_release_split.py training_split.json split_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def translate(source: dict) -> dict:
    for key in ("train", "validation"):
        if key not in source:
            raise ValueError(f"split file has no {key!r} list")
    train = list(source["train"])
    val = list(source["validation"])
    test = list(source.get("test") or [])
    all_uuids = train + val + test
    if len(set(all_uuids)) != len(all_uuids):
        raise ValueError("an episode id appears more than once across train/validation/test -- refusing to convert")
    return {
        "train": train,
        "val": val,
        "block_holdout_v1": {"holdout_episodes": test},
        "block_of_episode": {u: u for u in all_uuids},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_split_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = json.loads(args.training_split_json.read_text())
    manifest = translate(source)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"train={len(manifest['train'])} val={len(manifest['val'])} "
          f"holdout={len(manifest['block_holdout_v1']['holdout_episodes'])} -> {args.output}")


if __name__ == "__main__":
    main()
