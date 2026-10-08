#!/usr/bin/env python3
"""Compute training normalization statistics for a canonical LeRobot dataset.

The script reads 32-dimensional ``observation.state`` and ``action`` columns,
constructs a 50-step action horizon, applies the selected robot's element-wise
delta convention, and writes an OpenPI-compatible ``norm_stats.json``.

Example (single dataset):
  python scripts/compute_canonical_norm.py \
    --repo-id /path/to/dataset \
    --robot flexiv \
    --train-config-name my_flexiv_predictor \
    --asset-id my_dataset

Multiple datasets (e.g. a single-arm and a bimanual collection) can be combined into ONE
norm_stats.json by passing --repo-id/--robot more than once, in the SAME order --
each repo gets its OWN delta mask (a mixed single-arm + bimanual set of platforms
needs different masks per platform, not one shared --robot value):
  python scripts/compute_canonical_norm.py \
    --repo-id /path/to/flexiv_slice --repo-id /path/to/umi_slice \
    --robot flexiv --robot aloha \
    --train-config-name vtla_tactile_posttrain --asset-id mixed_collection
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from n0vtla.policies import canonical_schema as _canonical_schema
from n0vtla.policies.rotation_utils import rot6d_delta_world


ACTION_HORIZON = 50
NUM_QUANTILE_BINS = 5000

# A dimension whose observed spread is below this is treated as constant (reserved padding or an
# unused arm block) and gets identity normalisation stats. See RunningStats.as_json.
_CONSTANT_DIM_TOL = 1e-8

# NOTE on "flexiv"/"aloha": these masks are element-wise, applied to actions[..., :dims]
# starting at index 0 -- appropriate only for a dataset whose single-arm (or first-arm) EEF
# block itself starts at index 0. The robot canonical layout (see
# scripts/robot_single_arm_adapter.py's LAYOUT) instead puts its one active arm in the RIGHT
# slot, canonical_schema.ACTION_SLOTS["right_eef_xyz"/"right_eef_rot6d"] = indices
# [10:13]/[13:19]. These masks only touch indices [0:10] (the layout's always-zero left-arm
# padding), so they would leave the action stats as absolute values instead of deltas.
#
# Also, "flexiv"/"aloha" subtract element-wise across the whole 9-dim eef block, including
# the 6 rot6d dims. rot6d is not a vector space -- element-wise subtraction of two rot6d
# vectors is not a valid relative rotation. Use ROBOT_ROTATION_AWARE for any dataset built by
# robot_single_arm_adapter.py -- it targets the correct (right-arm) indices via
# canonical_schema.ACTION_SLOTS and composes rotation matrices (matches
# n0vtla.transforms.ChunkDeltaToCurrentState's math) instead of subtracting rot6d vectors.
ROBOT_DELTA_MASKS = {
    "flexiv": [True] * 9 + [False],
    "aloha": [True] * 9 + [False] + [True] * 9 + [False],
}

# Sentinel delta_mask value routed to apply_delta_rotation_aware() instead of apply_delta().
ROBOT_ROTATION_AWARE = "canonical_single_arm_rotation_aware"
ROBOT_DELTA_MASKS["canonical_single_arm"] = ROBOT_ROTATION_AWARE
# Dual-arm robot data (robot_dual_arm_adapter.py layout): BOTH eef blocks, left [0:9] and right [10:19],
# are active, so both get the rotation-aware delta. canonical_single_arm would leave the left arm's
# stats as absolute values while training sees it as a delta (ChunkDeltaToCurrentState).
ROBOT_ROTATION_AWARE_DUAL = "canonical_dual_arm_rotation_aware"
ROBOT_DELTA_MASKS["canonical_dual_arm"] = ROBOT_ROTATION_AWARE_DUAL


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = None
        self.mean_of_squares = None
        self.min = None
        self.max = None
        self.histograms = None
        self.bin_edges = None

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, batch.shape[-1])
        if batch.shape[0] == 0:
            return
        num_elements, vector_length = batch.shape
        if self.count == 0:
            self.mean = np.mean(batch, axis=0)
            self.mean_of_squares = np.mean(batch**2, axis=0)
            self.min = np.min(batch, axis=0)
            self.max = np.max(batch, axis=0)
            self.histograms = [np.zeros(NUM_QUANTILE_BINS, dtype=np.float64) for _ in range(vector_length)]
            self.bin_edges = [
                np.linspace(self.min[i] - 1e-10, self.max[i] + 1e-10, NUM_QUANTILE_BINS + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self.mean.size:
                raise ValueError("Vector length changed.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self.max)
            min_changed = np.any(new_min < self.min)
            self.max = np.maximum(self.max, new_max)
            self.min = np.minimum(self.min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self.count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self.mean += (batch_mean - self.mean) * (num_elements / self.count)
        self.mean_of_squares += (batch_mean_of_squares - self.mean_of_squares) * (num_elements / self.count)
        self._update_histograms(batch)

    def _adjust_histograms(self) -> None:
        for i in range(len(self.histograms)):
            old_edges = self.bin_edges[i]
            new_edges = np.linspace(self.min[i], self.max[i], NUM_QUANTILE_BINS + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self.histograms[i])
            self.histograms[i] = new_hist
            self.bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self.bin_edges[i])
            self.histograms[i] += hist

    def _quantile(self, q: float) -> np.ndarray:
        target_count = q * self.count
        values = []
        for hist, edges in zip(self.histograms, self.bin_edges, strict=True):
            cumsum = np.cumsum(hist)
            idx = min(np.searchsorted(cumsum, target_count), len(edges) - 1)
            values.append(edges[idx])
        return np.asarray(values)

    def as_json(self) -> dict[str, list[float]]:
        if self.count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self.mean_of_squares - self.mean**2
        std = np.sqrt(np.maximum(0, variance))
        q01 = self._quantile(0.01)
        q99 = self._quantile(0.99)

        # Identity stats for constant dimensions.
        #
        # The canonical 32-dim container reserves dims 20:32, and a single-arm dataset also
        # leaves the whole second-arm block constant at zero. Statistics measured over a
        # constant-zero dimension come out as q01 == q99 == 0 and std == 0, and quantile
        # normalisation then maps the constant 0 to -1 rather than to 0. The pretrained
        # checkpoints normalise those same dimensions with identity stats, so 0 maps to 0.
        #
        # Mixing the two conventions is not fatal -- training pulls the padded outputs back and
        # converges to the same place -- but it starts post-training with a step-0 loss around
        # 2.5 instead of 0.04, which reads as a broken run. Emit identity stats for constant
        # dimensions so a freshly computed asset agrees with the released weights.
        constant = (q99 - q01 <= _CONSTANT_DIM_TOL) & (std <= _CONSTANT_DIM_TOL)
        mean = np.where(constant, 0.0, self.mean)
        std = np.where(constant, 1.0, std)
        q01 = np.where(constant, -1.0, q01)
        q99 = np.where(constant, 1.0, q99)
        if constant.any():
            logging.info(
                "identity stats applied to %d constant dimension(s): %s",
                int(constant.sum()),
                np.flatnonzero(constant).tolist(),
            )

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def read_fixed_list(table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def action_horizon_sequence(actions: np.ndarray, horizon: int = ACTION_HORIZON) -> np.ndarray:
    frame_count = actions.shape[0]
    offsets = np.arange(horizon, dtype=np.int64)
    base = np.arange(frame_count, dtype=np.int64)[:, None]
    indices = np.minimum(base + offsets[None, :], frame_count - 1)
    return actions[indices]


def apply_delta(actions: np.ndarray, state: np.ndarray, delta_mask: list[bool]) -> np.ndarray:
    actions = actions.copy()
    mask = np.asarray(delta_mask, dtype=bool)
    dims = mask.shape[0]
    subtract = np.where(mask, state[:, :dims], 0.0).astype(actions.dtype)
    actions[..., :dims] -= subtract[:, None, :]
    return actions


def apply_delta_rotation_aware(actions: np.ndarray, state: np.ndarray,
                               arms: tuple[str, ...] = ("right",)) -> np.ndarray:
    """Delta relative to the current state for the canonical layout's RIGHT eef
    block only (the one arm robot_single_arm_adapter.py ever writes). xyz: element-wise
    subtraction (a valid delta). rot6d: world-frame relative rotation via
    n0vtla.policies.rotation_utils.rot6d_delta_world (R_action @ R_state^T -- composing
    rotation matrices, not subtracting rot6d vectors, which is not a vector space and
    does not support element-wise delta). Matches n0vtla.transforms.
    ChunkDeltaToCurrentState's math (the transform actually used at train/serve time),
    vectorized over (N, horizon) so it's fast enough for fitting norm stats over a
    whole split.

    actions: (N, horizon, 32); state: (N, 32). The gripper [19:20] and hand [20:26]
    dims, and the always-zero left-arm block [0:10], are left untouched (already
    absolute / already zero either way).
    """
    actions = actions.copy()
    for arm in arms:
        xyz_lo, xyz_hi = _canonical_schema.ACTION_SLOTS[f"{arm}_eef_xyz"]
        rot_lo, rot_hi = _canonical_schema.ACTION_SLOTS[f"{arm}_eef_rot6d"]

        actions[..., xyz_lo:xyz_hi] -= state[:, None, xyz_lo:xyz_hi]

        ref = np.broadcast_to(state[:, None, rot_lo:rot_hi], actions[..., rot_lo:rot_hi].shape)
        actions[..., rot_lo:rot_hi] = rot6d_delta_world(ref, actions[..., rot_lo:rot_hi])
    return actions


def _train_episode_indices(repo: Path) -> set[int] | None:
    """Episode indices tagged split=="train" in meta/episodes.jsonl, or None if the
    dataset has no split tags -- callers should then use every file."""
    episodes_path = repo / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return None
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    if not rows or "split" not in rows[0]:
        return None
    return {row["episode_index"] for row in rows if row["split"] == "train"}


def compute(specs: list[dict], max_frames: int | None, train_only: bool = False) -> tuple[dict, int]:
    """specs: one dict per dataset ({"repo_id", "delta_mask"}), combined into ONE set of
    stats (shared RunningStats accumulators across all of them) -- see this module's
    docstring for why each dataset needs its own delta_mask rather than one shared value."""
    state_stats = RunningStats()
    action_stats = RunningStats()
    processed = 0

    for spec in specs:
        repo = Path(spec["repo_id"])
        if not (repo / "data").is_dir():
            raise FileNotFoundError(f"dataset data directory not found: {repo / 'data'}")

        train_indices = _train_episode_indices(repo) if train_only else None
        if train_only and train_indices is None:
            raise ValueError(f"--train-only requested but {repo} has no split tags in meta/episodes.jsonl")

        for parquet_path in sorted((repo / "data").rglob("*.parquet")):
            if train_indices is not None and int(parquet_path.stem.split("_")[-1]) not in train_indices:
                continue
            table = pq.read_table(parquet_path, columns=["observation.state", "action"])
            state = read_fixed_list(table, "observation.state")
            action = read_fixed_list(table, "action")

            if max_frames is not None:
                remaining = max_frames - processed
                if remaining <= 0:
                    return (
                        {"norm_stats": {"state": state_stats.as_json(), "actions": action_stats.as_json()}},
                        processed,
                    )
                state = state[:remaining]
                action = action[:remaining]

            actions = action_horizon_sequence(action)
            if spec["delta_mask"] == ROBOT_ROTATION_AWARE:
                actions = apply_delta_rotation_aware(actions, state)
            elif spec["delta_mask"] == ROBOT_ROTATION_AWARE_DUAL:
                actions = apply_delta_rotation_aware(actions, state, arms=("left", "right"))
            else:
                actions = apply_delta(actions, state, spec["delta_mask"])
            state_stats.update(state)
            action_stats.update(actions)
            processed += state.shape[0]

            if processed % 50000 < state.shape[0]:
                print(f"processed {processed} frames (repo={repo.name})")

    return {"norm_stats": {"state": state_stats.as_json(), "actions": action_stats.as_json()}}, processed


def resolve_specs(args: argparse.Namespace) -> tuple[str, list[dict], str]:
    repo_ids: list[str] = args.repo_id
    robots: list[str] = args.robot
    if len(robots) == 1:
        robots = robots * len(repo_ids)
    if len(robots) != len(repo_ids):
        raise ValueError(
            f"--robot given {len(robots)} time(s) but --repo-id given {len(repo_ids)} time(s) -- "
            "pass --robot once (applies to all repos) or once per --repo-id, in the same order."
        )
    specs = [
        {"repo_id": str(Path(r)), "delta_mask": ROBOT_DELTA_MASKS[robot]}
        for r, robot in zip(repo_ids, robots)
    ]
    asset_id = args.asset_id or Path(repo_ids[0]).name
    return args.train_config_name, specs, asset_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, action="append",
                         help="Local canonical LeRobot dataset root. Repeat to combine multiple "
                              "datasets into one norm_stats.json (pass --robot the same number of "
                              "times, in the same order, or once to apply to all of them).")
    parser.add_argument(
        "--robot",
        required=True,
        action="append",
        choices=sorted(ROBOT_DELTA_MASKS),
        help="Robot layout used to select the element-wise delta mask. Repeat in the same order "
             "as --repo-id for a mixed single-arm/bimanual combination, or pass once to apply "
             "to every --repo-id.",
    )
    parser.add_argument(
        "--train-config-name",
        required=True,
        help="TrainConfig.name used to choose assets/<train-config-name>/<asset-id>/norm_stats.json.",
    )
    parser.add_argument("--asset-id", help="Asset id directory. Defaults to basename of the first --repo-id.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--train-only", action="store_true",
                         help="Only fit on episodes tagged split==\"train\" in meta/episodes.jsonl "
                              "(requires a dataset built with split tags, e.g. "
                              "build_robot_dataset.py). Errors if the dataset has no tags.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root for the assets directory. Defaults to the parent of scripts/.",
    )
    args = parser.parse_args()

    train_config_name, specs, asset_id = resolve_specs(args)
    payload, processed = compute(specs, args.max_frames, train_only=args.train_only)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_dir = repo_root / "assets" / train_config_name / asset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Processed frames: {processed}")
    print(f"Writing stats to: {output_path}")


if __name__ == "__main__":
    main()
