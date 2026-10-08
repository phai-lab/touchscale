"""Held-out action-prediction loss vs. tactile activity, per episode.

Computes the model's OWN flow-matching training loss -- the exact quantity
train_pytorch.py optimizes -- on held-out episodes with NO gradient step, then
correlates the per-episode mean against the same external tactile-activity
metrics used by compute_tactile_action_correlation.py (temporal variance,
contact-event rate). Unlike that script's self-consistency proxy (predicted
step size / reversal rate, never compared to ground truth), this is a real
prediction error against the ground-truth actions.

DESIGN:

- Model construction reuses train_pytorch.py's exact path: `import
  scripts.train_n0vtla` first (the same module train.sh launches), which
  monkeypatches PI0Pytorch to the tactile-aware N0VTLAPolicy subclass as a side
  effect -- so the model class here is the one training uses, not a
  reimplemented copy that could drift.
- The data loader is `n0vtla.training.data.create_data_loader(config,
  framework="pytorch", shuffle=False)` -- the same factory train_pytorch.py's
  build_datasets() calls -- so batch construction, transforms, and the
  action-horizon/delta encoding are identical to training.
- batch_size is forced to 1 so each yielded sample maps unambiguously to
  exactly one dataset row; the dataset length is checked against the episode
  metadata to guard the "shuffle=False preserves dataset order" assumption.
- The flow-matching loss is stochastic in its own (noise, time) draw per
  forward call (not just dropout). A single draw per sample would be a noisy
  point estimate, so this averages --repeats independent draws (distinct
  seeds) per sample.
- model.eval() + torch.no_grad(): no dropout, no autograd graph, no
  optimizer step -- a pure measurement pass. missing/unexpected keys from
  the checkpoint load are asserted empty, not silently ignored.
- The correlation statistics (Pearson r, Spearman rho, exact permutation
  test, Holm correction, bootstrap CI) are the same functions
  compute_tactile_action_correlation.py uses (imported, not re-derived), so
  the two analyses are directly comparable.

LIMITATIONS:
  - A small held-out set (few episodes); hypothesis-generating, not confirmatory.
  - The correlation is with a real prediction error (not a self-consistency
    proxy), but is still correlational, not causal -- pair with an ablation
    for direction.
  - The flow-matching loss average over --repeats draws reduces but does not
    eliminate stochastic-draw noise in the per-sample estimate.

Usage:
  python scripts/compute_held_out_action_loss.py \
    --config vtla_tactile_posttrain \
    --checkpoint checkpoints/vtla_tactile_posttrain/<exp name>/<step> \
    --checkpoint-label base \
    --dataset-root /path/to/canonical_robot_holdout \
    --raw-source-root /path/to/raw_release \
    --norm-path /path/to/tactile_norm.json \
    --output /path/to/held_out_loss_base.json

  # quick pipeline check on a subset of rows:
  python scripts/compute_held_out_action_loss.py ... --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_tactile_action_correlation import (  # noqa: E402
    bootstrap_ci, exact_permutation_p, holm_correction, match_runs_to_episodes,
    pearson, spearman, tactile_activity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="vtla_tactile_posttrain")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-source-root", type=Path, required=True)
    parser.add_argument("--norm-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--stride", type=int, default=5, help="only compute loss on every Nth dataset row")
    parser.add_argument("--repeats", type=int, default=4, help="independent (noise,time) draws averaged per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="quick pipeline check: stride=50, repeats=1, no correlation statistics")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    import safetensors.torch
    import jax

    import scripts.train_n0vtla  # noqa: F401  (side effect: patches PI0Pytorch -> N0VTLAPolicy)
    import n0vtla.models.model
    import n0vtla.models.pi0_config
    import n0vtla.models_pytorch.pi0_pytorch
    import robot_single_arm_adapter as adapter
    from itw_pressure import PAD_IDS
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    norm = json.loads(args.norm_path.read_text())
    baseline = np.asarray(norm["normal_baseline"][15:30], dtype=np.float64)
    scale = np.asarray(norm["normal_scale"][15:30], dtype=np.float64)
    threshold = np.asarray(norm["contact_threshold"][15:30], dtype=np.float64)

    # Always ALL episodes, even in --smoke: the loader reads the whole configured dataset
    # regardless of what this script does on the Python side, so total_rows/boundaries must
    # match the loader's full-dataset row count or the row-count verification below would
    # fail. --smoke stays fast via stride=50/repeats=1, not by truncating this list.
    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = [ep["length"] for ep in episodes]
    total_rows = sum(lengths)
    boundaries = np.concatenate([[0], np.cumsum(lengths)])  # boundaries[i]..boundaries[i+1] = episode i's rows

    def episode_of(running_index: int) -> int | None:
        if running_index >= total_rows:
            return None
        return int(np.searchsorted(boundaries, running_index, side="right") - 1)

    print(f"=== resolving tactile activity for {len(episodes)} episode(s) ===")
    by_uuid: dict[str, list[dict]] = {}
    for ep in episodes:
        by_uuid.setdefault(ep["source_uuid"], []).append(ep)
    for v in by_uuid.values():
        v.sort(key=lambda e: e["episode_index"])
    tactile_by_episode: dict[int, dict] = {}
    for uuid, uuid_episodes in by_uuid.items():
        source = args.raw_source_root / uuid
        window = adapter.resolve_episode_window(source, task_name=args.task_name, require_success_label=True)
        run_map = match_runs_to_episodes(window["runs"], uuid_episodes)
        with np.load(window["tactile_source"], allow_pickle=False) as z:
            for ep in uuid_episodes:
                ti = window["ti"][run_map[ep["episode_index"]]]
                grids = [np.asarray(z[f"tactile_{p}"], np.float64)[ti] for p in PAD_IDS]
                tactile_by_episode[ep["episode_index"]] = tactile_activity(grids, baseline, scale, threshold)
        print(f"  uuid={uuid}: {len(uuid_episodes)} episode(s) matched by exact run length")

    device = torch.device(args.device)
    config = _config.get_config(args.config)
    if args.smoke:
        object.__setattr__(config, "batch_size", 1)
    else:
        object.__setattr__(config, "batch_size", 1)

    if not isinstance(config.model, n0vtla.models.pi0_config.Pi0Config):
        model_cfg = n0vtla.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            use_tactile=getattr(config.model, "use_tactile", False),
            expert_vision_type=getattr(config.model, "expert_vision_type", None),
            expert_vision_path=getattr(config.model, "expert_vision_path", None),
            tactile_single_token=getattr(config.model, "tactile_single_token", False),
            tactile_image_keys=getattr(config.model, "tactile_image_keys", n0vtla.models.model.TACTILE_IMAGE_KEYS),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    print(f"=== loading model from {args.checkpoint} ===")
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(
        model, str(args.checkpoint / "model.safetensors"), device=str(device)
    )
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch -- missing={missing}, unexpected={unexpected}")
    model.eval()

    print("=== building held-out data loader (shuffle=False, batch_size=1) ===")
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)

    # NOTE on what this does and doesn't verify: the loader applies the full training-time
    # transform pipeline (repack/data/model transforms, normalization, DeltaActions) before
    # yielding (observation, actions) -- its output is not numerically comparable to
    # LocalLeRobotV3Dataset's raw untransformed rows, and the (observation, actions) tuple
    # carries no passthrough episode_index/row-index field to check directly. What CAN be
    # verified without instrumenting the dataset internals: the underlying dataset's own
    # reported length (O(1), no video decode -- a map-style Dataset's __len__ reads
    # metadata, it doesn't decode frames) must equal total_rows (episodes.jsonl's summed
    # length). This catches silent frame-dropping, windowing, or duplication in how the
    # dataset was configured. Strict row-to-row order is a standard guarantee of a
    # sequential (non-shuffled) sampler and is not re-checked here.
    #
    # The check uses len() rather than iterating the loader to count samples: iterating
    # forces a full video decode of every held-out frame just to obtain a count.
    print("=== verifying dataset length (O(1), no decode) matches total_rows ===")
    # Two layers of wrapping to unwrap: DataLoaderImpl (n0vtla.training.data_loader) only
    # exposes __iter__/data_config() publicly and holds a TorchDataLoader on ._data_loader;
    # TorchDataLoader itself is NOT a torch.utils.data.DataLoader (it wraps one, with its own
    # __iter__ that loops the wrapped loader forever for training's infinite-epoch style, so
    # iterating it to count samples would never terminate). The real
    # torch.utils.data.DataLoader is exposed via TorchDataLoader's public `.torch_loader`
    # property, and its `.dataset` has the instant, no-decode __len__ this check needs.
    inner = loader._data_loader
    underlying = inner.torch_loader.dataset
    counted = len(underlying)
    print(f"  dataset reports {counted} samples; episodes.jsonl sums to {total_rows}")
    if counted != total_rows:
        raise RuntimeError(
            f"dataset length {counted} != episodes.jsonl total {total_rows} -- "
            f"per-episode attribution below would be WRONG (frames dropped/added somewhere "
            f"in the pipeline). Aborting rather than reporting misattributed results."
        )

    stride = 50 if args.smoke else args.stride
    repeats = 1 if args.smoke else args.repeats
    print(f"=== computing held-out action loss (stride={stride}, repeats={repeats}) ===")
    per_episode_losses: dict[int, list[float]] = {ep["episode_index"]: [] for ep in episodes}
    running_index = 0
    n_scored = 0
    for observation, actions in loader:
        ep_idx = episode_of(running_index)
        if ep_idx is None:
            break
        if running_index % stride == 0:
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions_t = actions.to(torch.float32).to(device)
            draws = []
            for r in range(repeats):
                torch.manual_seed(args.seed * 100_000 + running_index * 10 + r)
                with torch.no_grad():
                    losses = model(observation, actions_t)
                per_sample = losses.reshape(losses.shape[0], -1).mean(dim=1)
                draws.append(float(per_sample.item()))
            per_episode_losses[ep_idx].append(float(np.mean(draws)))
            n_scored += 1
        running_index += 1
        if running_index >= total_rows:
            break

    per_episode = []
    for ep in episodes:
        idx = ep["episode_index"]
        losses = per_episode_losses[idx]
        per_episode.append(dict(
            episode_index=idx, source_uuid=ep["source_uuid"], length=ep["length"],
            n_samples=len(losses), mean_action_loss=float(np.mean(losses)) if losses else float("nan"),
            **tactile_by_episode[idx],
        ))
        print(f"  episode {idx}: {len(losses)} samples, mean_action_loss={per_episode[-1]['mean_action_loss']:.5f}")

    results = []
    raw_pvalues = []
    if not args.smoke:
        for tm in ["variance", "contact_rate"]:
            x = np.array([row[tm] for row in per_episode])
            y = np.array([row["mean_action_loss"] for row in per_episode])
            r = pearson(x, y)
            rho = spearman(x, y)
            p_pearson = exact_permutation_p(x, y, pearson)
            p_spearman = exact_permutation_p(x, y, spearman)
            ci = bootstrap_ci(x, y, pearson, seed=args.seed)
            results.append(dict(tactile_metric=tm, quality_channel="mean_action_loss", n=len(x),
                                 pearson_r=r, pearson_p_exact=p_pearson, pearson_ci95=ci,
                                 spearman_rho=rho, spearman_p_exact=p_spearman))
            raw_pvalues.append(p_pearson)
        holm = holm_correction(raw_pvalues)
        for res, adj in zip(results, holm):
            res["pearson_p_holm_corrected_across_all_tests_this_checkpoint"] = adj

    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_label=args.checkpoint_label,
        dataset_root=str(args.dataset_root), norm_path=str(args.norm_path),
        smoke=args.smoke, stride=stride, repeats=repeats, n_episodes=len(episodes),
        n_scored_rows=n_scored, per_episode=per_episode, correlations=results,
        limitations=[
            "Real held-out flow-matching loss (no gradient), not a self-consistency proxy -- "
            "but still correlational, not causal.",
            f"n={len(episodes)} holdout episodes -- hypothesis-generating, not confirmatory.",
            f"Loss averaged over {repeats} independent (noise,time) draws per scored sample to "
            "reduce flow-matching's own stochastic-draw noise; not eliminated.",
            "2 tests per checkpoint (variance, contact_rate vs. mean_action_loss); Holm-corrected "
            "p-values are the ones to trust.",
        ] if not args.smoke else ["SMOKE RUN -- pipeline validation only, stats not meaningful."],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(checkpoint_label=args.checkpoint_label, correlations=results), indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
