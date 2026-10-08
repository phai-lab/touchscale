"""Per-episode tactile-LATENT prediction quality (InfoNCE) vs. held-out action loss.

For a post-train checkpoint, using that checkpoint's own tactile_encoder /
tactile_predictor / z_proj weights, measure how well the predicted future-tactile
latent z matches the real future-tactile latent z* on held-out episodes, then
correlate that per-episode quality against the same checkpoint's held-out action
loss (the report written by compute_held_out_action_loss.py).

WHAT z / z* / INFONCE ARE (see docs/MID_TRAIN.md, n0vtla_policy.py):
  - At every forward pass -- training, post-train, and rollout alike -- the model
    calls `_compute_z()`: it builds `g` from the current and past real tactile signal
    (tac_t, tac_0; no future info involved) and runs `g` through `tactile_predictor` to
    get `z`, a latent that is architecturally meant to anticipate near-future tactile
    state. `z` is projected (z_proj) and scaled by the learned scalar `z_gate` before
    being injected as extra conditioning tokens into the action expert. So tactile
    prediction runs on every rollout step and feeds the action head (assuming
    z_gate != 0).
  - At rollout there is no ground-truth future tactile frame to check `z` against.
    It can only be checked on logged episodes where a future frame exists, i.e.
    held-out data. `_build_future_target()` builds z* from the real
    (tac_{t+H} - tac_t) diff, and `_stage1_infonce_loss` scores whether z picks its own
    z* out of a pool of other samples' z* (symmetric InfoNCE / contrastive retrieval,
    same math as CLIP): low loss = z reliably identifies "its" real future, high loss =
    z carries no useful future-tactile information.
  - `tactile_recon_head` (pixel-space reconstruction of the future tactile frame) is a
    separate, disjoint component of the Stage-1 loss. It is only trained by a Stage-1
    run; a post-train checkpoint does not carry trained weights for it. So this script
    measures the InfoNCE/latent-retrieval component only -- the part of "tactile
    prediction quality" every post-train checkpoint can be scored on.

WHY A WHOLE-SPLIT POOL, NOT A PER-FORWARD-CALL SCORE: `_stage1_infonce_loss` (the
model's own Stage-1 loss function) scores one DDP-gathered training batch at a time,
pooling valid samples across every rank/GPU as its negative set. That in-training
pooling isn't available to a standalone single-GPU eval script, so instead this script
runs `_compute_z`/`_build_future_target` once per stride-sampled frame to collect
(z, z*) for the whole held-out split first, then builds one joint (N x N)
cosine-similarity matrix over all of them as a mutual negative pool, and takes the
per-row/per-column cross-entropy term (`reduction="none"`) as each sample's own score.
This is the same InfoNCE math (mean-pooled, L2-normalized, temperature-scaled,
symmetric i2t/t2i cross-entropy) the model itself uses -- just with a negative pool
sized to the eval split instead of one training minibatch, which is also the more
standard design for a retrieval eval. The computation is deterministic (model.eval()
disables all stochastic layers), so no repeated-draw averaging is needed here (unlike
the flow-matching loss in compute_held_out_action_loss.py, which has its own
(noise, time) stochasticity).

LIMITATIONS:
  - A small held-out set (few episodes) per checkpoint.
  - One test per checkpoint (InfoNCE-latent-error vs. mean_action_loss); no Holm
    correction is needed for a single test, but read it alongside the other
    tactile/action analyses, not in isolation.
  - Measures the latent-retrieval component of tactile prediction only (no
    checkpoint here has a trained reconstruction head to measure pixel-space
    accuracy with).
  - The negative pool for a checkpoint is drawn only from its own held-out split --
    pool composition (episode count/diversity) can differ between checkpoints that
    were evaluated on different held-out sets, so InfoNCE difficulty is not
    guaranteed identical across them; read absolute values with that in mind, but each
    checkpoint's own per-episode ranks/correlations are still an internally valid
    comparison against its own action loss.

Usage:
  python scripts/compute_held_out_tactile_pred_error.py \
    --checkpoint checkpoints/vtla_tactile_posttrain/<exp name>/<step> \
    --checkpoint-label base \
    --action-loss-json /path/to/held_out_loss_base.json \
    --dataset-root /path/to/canonical_robot_holdout \
    --asset-id <asset id> \
    --output /path/to/tactile_latent_base.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_tactile_action_correlation import (
    bootstrap_ci,
    exact_permutation_p,
    pearson,
    spearman,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1-config", default="vtla_stage1_predictor_pretrain",
                         help="TrainConfig name used only to build a model skeleton with "
                              "tactile_predictor_enabled=True, stage1_pretrain_enabled=True, "
                              "and a future_frame_offset>0 DataConfig; weights come from --checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="Post-train checkpoint dir (its own post-train run).")
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--action-loss-json", type=Path, required=True,
                         help="Held-out action-loss report (compute_held_out_action_loss.py) for this SAME checkpoint.")
    parser.add_argument("--dataset-root", type=Path, required=True,
                         help="This checkpoint's own native held-out set (matches --action-loss-json's).")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    import torch.nn.functional as F
    import safetensors.torch
    import jax
    import scripts.train_n0vtla  # noqa: F401 -- patches PI0Pytorch -> N0VTLAPolicy as an import side effect
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    action_loss_report = json.loads(args.action_loss_json.read_text())
    if action_loss_report.get("checkpoint_label") != args.checkpoint_label:
        raise RuntimeError(
            f"--checkpoint-label {args.checkpoint_label!r} does not match "
            f"--action-loss-json's checkpoint_label {action_loss_report.get('checkpoint_label')!r} -- "
            f"these must be the SAME checkpoint's two reports"
        )
    action_loss_by_episode = {row["episode_index"]: row["mean_action_loss"]
                               for row in action_loss_report["per_episode"]}

    episodes = [json.loads(l) for l in (args.dataset_root / "meta" / "episodes.jsonl").open()]
    lengths = [ep["length"] for ep in episodes]
    total_rows = sum(lengths)
    boundaries = np.concatenate([[0], np.cumsum(lengths)])

    def episode_of(running_index):
        if running_index >= total_rows:
            return None
        return int(np.searchsorted(boundaries, running_index, side="right") - 1)

    # VTLA_ASSET_ID / VTLA_DATASET_PATH are read via os.environ.get(...) at
    # config-construction time inside n0vtla/training/config.py, which already ran at
    # the `from n0vtla.training import config as _config` import above -- setting them
    # here would be too late. The caller must export them in the shell BEFORE invoking
    # this script.
    import os
    for var in ("VTLA_ASSET_ID", "VTLA_DATASET_PATH"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} must be exported in the shell before running this script "
                                f"(config-construction reads it at import time)")

    device = torch.device(args.device)
    config = _config.get_config(args.stage1_config)
    object.__setattr__(config, "batch_size", 1)
    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    print(f"=== loading checkpoint ({args.checkpoint_label}) from {args.checkpoint} ===")
    import n0vtla.models_pytorch.pi0_pytorch
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    missing, unexpected = safetensors.torch.load_model(
        model, str(args.checkpoint / "model.safetensors"), device=str(device)
    )
    # Two expected, harmless gaps (see module docstring): this
    # checkpoint has no tactile_recon_head (not trained under post-train), and the
    # Stage-1 model skeleton doesn't instantiate z_gate (it never touches the action
    # expert) even though the post-train checkpoint has one. Neither is touched by the
    # InfoNCE-only computation below.
    missing_ok = [k for k in missing if "tactile_recon_head" not in k]
    unexpected_ok = [k for k in unexpected if k != "z_gate"]
    if missing_ok or unexpected_ok:
        raise RuntimeError(f"checkpoint load mismatch beyond the expected gaps -- "
                            f"missing={missing_ok}, unexpected={unexpected_ok}")
    if missing:
        print(f"  (expected) tactile_recon_head not in checkpoint, {len(missing)} key(s) "
              f"randomly initialized and unused -- this script never calls the reconstruction head")
    if unexpected:
        print(f"  (expected) checkpoint has {unexpected} not used by the Stage-1 model skeleton -- ignored")
    model.eval()

    print("=== building held-out data loader (stage1 DataConfig, shuffle=False, batch_size=1) ===")
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    inner = loader._data_loader
    underlying = inner.torch_loader.dataset
    counted = len(underlying)
    print(f"  dataset reports {counted} samples; episodes.jsonl sums to {total_rows}")
    if counted != total_rows:
        raise RuntimeError(f"dataset length {counted} != episodes.jsonl total {total_rows} -- aborting")

    stride = 50 if args.smoke else args.stride
    print(f"=== collecting per-sample (z, z*) embeddings (stride={stride}) ===")
    torch.manual_seed(args.seed)

    hz_list, hzs_list, ep_idx_list = [], [], []
    running_index = 0
    n_considered = n_skipped_no_future = 0
    for observation, _actions in loader:
        ep_idx = episode_of(running_index)
        if ep_idx is None:
            break
        if running_index % stride == 0:
            n_considered += 1
            observation = jax.tree.map(lambda x: x.to(device), observation)
            with torch.no_grad():
                images, img_masks, lang_tokens, lang_masks, _s, _e = model._preprocess_observation(
                    observation, train=True
                )
                vl_ctx, _pe, prefix_pad_masks, _pam, _pkv = model._prefix_forward(
                    images, img_masks, lang_tokens, lang_masks, use_cache=False
                )
                z, _g, has_tac = model._compute_z(vl_ctx, prefix_pad_masks)
                tac_f = model._last_tac_f or {}
                tac_t = model._last_tac_t or {}
                tac_mask_f = getattr(model, "_last_tac_mask_f", None)
                z_star, _dbar, has_future = model._build_future_target(tac_f, tac_t, tac_mask_f)
                if z_star is None or not bool((has_tac & has_future).all()):
                    n_skipped_no_future += 1
                else:
                    hz = F.normalize(z.float().mean(dim=1), dim=-1)[0]        # (D,)
                    hzs = F.normalize(z_star.detach().float().mean(dim=1), dim=-1)[0]  # (D,)
                    hz_list.append(hz.cpu())
                    hzs_list.append(hzs.cpu())
                    ep_idx_list.append(ep_idx)
        running_index += 1
        if running_index >= total_rows:
            break
    if n_skipped_no_future:
        print(f"  skipped {n_skipped_no_future}/{n_considered} sample(s) with no valid future-tactile target")

    n_pool = len(hz_list)
    print(f"=== building joint {n_pool}x{n_pool} contrastive pool over the whole held-out split ===")
    if n_pool < 2:
        raise RuntimeError(f"only {n_pool} valid sample(s) collected -- cannot form a contrastive pool (need >=2)")

    hz_mat = torch.stack(hz_list)    # (N, D)
    hzs_mat = torch.stack(hzs_list)  # (N, D)
    temp = float(getattr(model_cfg, "stage1_temperature", 1.0))
    logits = (hz_mat @ hzs_mat.t()) / temp   # (N, N)
    labels = torch.arange(n_pool)
    loss_i2t = F.cross_entropy(logits, labels, reduction="none")
    loss_t2i = F.cross_entropy(logits.t(), labels, reduction="none")
    per_sample_error = (0.5 * (loss_i2t + loss_t2i)).numpy()

    per_episode_errs = {ep["episode_index"]: [] for ep in episodes}
    for ep_idx, err in zip(ep_idx_list, per_sample_error):
        per_episode_errs[ep_idx].append(float(err))

    per_episode = []
    for ep in episodes:
        idx = ep["episode_index"]
        errs = per_episode_errs[idx]
        if idx not in action_loss_by_episode:
            raise RuntimeError(f"episode {idx} missing from --action-loss-json -- "
                                f"is it the SAME dataset-root the action-loss run used?")
        per_episode.append(dict(
            episode_index=idx, length=ep["length"], n_samples=len(errs),
            tactile_latent_error=float(np.mean(errs)) if errs else float("nan"),
            mean_action_loss=action_loss_by_episode[idx],
        ))
        print(f"  episode {idx}: {len(errs)} samples, "
              f"tactile_latent_error={per_episode[-1]['tactile_latent_error']:.5f}, "
              f"mean_action_loss={per_episode[-1]['mean_action_loss']:.5f}")

    x = np.array([row["tactile_latent_error"] for row in per_episode])
    y = np.array([row["mean_action_loss"] for row in per_episode])
    result = dict(n=len(x), pearson_r=pearson(x, y), spearman_rho=spearman(x, y))
    if not args.smoke:
        result["pearson_p_exact"] = exact_permutation_p(x, y, pearson)
        result["spearman_p_exact"] = exact_permutation_p(x, y, spearman)
        result["pearson_ci95"] = bootstrap_ci(x, y, pearson, seed=args.seed)

    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_label=args.checkpoint_label,
        action_loss_source=str(args.action_loss_json), dataset_root=str(args.dataset_root),
        smoke=args.smoke, stride=stride, n_pool=n_pool,
        per_episode=per_episode, correlation=result,
        limitations=[
            "a small held-out set (few episodes) -- hypothesis-generating, not confirmatory.",
            "One test (tactile_latent_error vs. mean_action_loss) -- no Holm correction needed alone; "
            "read alongside the other tactile/action analyses.",
            "Measures the InfoNCE latent-retrieval component of tactile prediction only -- no checkpoint "
            "here has a trained reconstruction head.",
            "Negative pool is this checkpoint's own held-out split only -- pool composition differs from "
            "the other checkpoints' pools, so cross-checkpoint absolute-value comparison is weaker than "
            "each checkpoint's own within-checkpoint correlation.",
        ] if not args.smoke else ["SMOKE RUN -- pipeline validation only, stats not meaningful."],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(checkpoint_label=args.checkpoint_label, n_pool=n_pool, correlation=result), indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
