#!/usr/bin/env python
"""Stage-1 predictor-grounding pretraining, ONLINE-LOADER variant (see
scripts/train_stage1_predictor.py's docstring for the paper reference and what Stage 1
trains). Identical training loop, loss, checkpointing, and warm-start behavior --
the ONLY difference from train_stage1_predictor.py is the data source: this script
reads directly from a raw TouchScale-format episode tree via
n0vtla.training.itw_online_dataset.ITWOnlineTactileDataset instead of a materialized
LeRobot dataset, so it can train over a corpus too large to duplicate on disk
(see docs/MID_TRAIN.md).

Reuses train_stage1_predictor.py's checkpoint save/load, the DDP-forward wrapper, and
warm-start logic UNCHANGED (imported, not copied) so those stay in exactly one place.

Usage (single node):
    torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \\
        scripts/train_stage1_online.py vtla_stage1_predictor_pretrain \\
        --exp-name=my_online_run

Data-source env vars (in addition to train_stage1_predictor.py's VTLA_PRETRAINED_CHECKPOINT,
VTLA_STAGE1_*, VTLA_DEFAULT_PROMPT -- config.data.repo_id/VTLA_DATASET_PATH are UNUSED here,
this script never touches config.data):
    VTLA_ITW_RAW_ROOT          required. Raw TouchScale-format tree root, e.g.
                                /path/to/touchscale_raw
    VTLA_ITW_NORMALIZATION     required. Path to a pad30 normalization JSON (either
                                per-pad or per-task-scale schema; see scripts/itw_pressure.py).
    VTLA_ITW_DATES             optional, comma-separated date subdirs (default: every date
                                under VTLA_ITW_RAW_ROOT). Ignored if VTLA_ITW_EPISODE_LIST_JSON is set.
    VTLA_ITW_MAX_EPISODES      optional int. Caps the episode count via a config.seed-seeded
                                sample -- for a quick pipeline check on a subset of the corpus.
                                Ignored if VTLA_ITW_EPISODE_LIST_JSON is set.
    VTLA_ITW_EPISODE_LIST_JSON optional path to a fixed episode-list manifest
                                (a {"raw_root", "splits": {"<key>": [relative episode dirs]}} JSON).
                                When set, this OVERRIDES VTLA_ITW_DATES/VTLA_ITW_MAX_EPISODES with a
                                FIXED episode list (no re-sampling per launch) -- for data-scaling
                                experiments where every run at a given fraction must see the exact
                                same episodes. Requires VTLA_ITW_EPISODE_LIST_KEY.
    VTLA_ITW_EPISODE_LIST_KEY  required if VTLA_ITW_EPISODE_LIST_JSON is set -- which key under the
                                manifest's "splits" to use, e.g. "20pct".
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import sys
from pathlib import Path

import jax
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pytorch import (  # noqa: E402
    get_latest_checkpoint_step,
    init_logging,
    init_wandb,
    set_seed,
    setup_ddp,
)
from train_stage1_predictor import (  # noqa: E402
    STAGE1_TRAINABLE_PREFIXES,
    _Stage1Wrapper,
    load_stage1_checkpoint,
    load_stage1_policy_weights,
    save_stage1_checkpoint,
)

import n0vtla.training.config as _config  # noqa: E402
import n0vtla.training.itw_online_dataset as _online  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402


def _online_episode_dirs(seed: int) -> list[Path]:
    raw_root = os.environ.get("VTLA_ITW_RAW_ROOT")
    if not raw_root:
        raise ValueError("VTLA_ITW_RAW_ROOT is required for train_stage1_online.py")

    episode_list_json = os.environ.get("VTLA_ITW_EPISODE_LIST_JSON")
    if episode_list_json:
        list_key = os.environ.get("VTLA_ITW_EPISODE_LIST_KEY")
        if not list_key:
            raise ValueError("VTLA_ITW_EPISODE_LIST_KEY is required when VTLA_ITW_EPISODE_LIST_JSON is set")
        manifest = json.loads(Path(episode_list_json).read_text())
        try:
            rel_paths = manifest["splits"][list_key]
        except KeyError as e:
            raise ValueError(f"key {list_key!r} not found in {episode_list_json}'s splits "
                              f"(available: {sorted(manifest.get('splits', {}))})") from e
        episodes = [Path(raw_root) / rel for rel in rel_paths]
        missing = [str(p) for p in episodes if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} episode dirs from manifest missing under "
                                     f"{raw_root}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        logging.info(f"VTLA_ITW_EPISODE_LIST_JSON={episode_list_json} key={list_key}: "
                      f"{len(episodes)} fixed episode dirs (no resampling)")
        return episodes

    dates_env = os.environ.get("VTLA_ITW_DATES")
    dates = dates_env.split(",") if dates_env else None
    episodes = _online.list_episode_dirs(raw_root, date_dirs=dates)
    if not episodes:
        raise ValueError(f"No episode dirs found under {raw_root} (dates={dates})")
    max_episodes = os.environ.get("VTLA_ITW_MAX_EPISODES")
    if max_episodes is not None:
        import random

        n = int(max_episodes)
        episodes = random.Random(seed).sample(episodes, min(n, len(episodes)))
        logging.info(f"VTLA_ITW_MAX_EPISODES={n}: sampled {len(episodes)} of the available episodes")
    return episodes


def train_loop_stage1_online(config: _config.TrainConfig) -> None:
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    global_rank = dist.get_rank() if use_ddp else 0
    set_seed(config.seed, global_rank)

    model_cfg = config.model
    assert getattr(model_cfg, "stage1_pretrain_enabled", False), (
        "train_stage1_online.py requires a config with stage1_pretrain_enabled=True "
        "(use the 'vtla_stage1_predictor_pretrain' TrainConfig or a variant of it)"
    )
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    if not model_cfg.tactile_predictor_enabled or model_cfg.tactile_mode != "latent":
        raise ValueError("Stage 1 requires the latent tactile predictor")
    # Needed on EVERY launch, resume included: trainable-only checkpoints (see
    # save_stage1_checkpoint in train_stage1_predictor.py) don't carry the frozen base,
    # so it's always reloaded from here first, with a resumed run's own checkpoint
    # applied on top afterward.
    if not config.pytorch_weight_path:
        raise ValueError("Stage 1 requires a pretrained base policy checkpoint")
    ckpt_path = Path(config.pytorch_weight_path)
    weights_file = ckpt_path / "model.safetensors" if ckpt_path.is_dir() else ckpt_path
    if not weights_file.is_file():
        raise FileNotFoundError(f"Pretrained base policy checkpoint missing: {weights_file}")

    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(config.checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(f"Resuming stage-1 (online) run from step {latest_step}")
            else:
                raise FileNotFoundError(f"No valid checkpoints in {config.checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"{config.checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        import shutil

        if is_main:
            shutil.rmtree(config.checkpoint_dir)
    elif config.checkpoint_dir.exists():
        raise FileExistsError(f"{config.checkpoint_dir} exists; use --resume or a new exp-name")

    if use_ddp:
        dist.barrier()

    if not resuming:
        from train_pytorch import wait_for_path_state

        ready_file = config.checkpoint_dir / ".ddp_fs_ready"
        if is_main:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ready_file.write_text("ready\n", encoding="utf-8")
        elif use_ddp:
            wait_for_path_state(ready_file, should_exist=True)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size < 2 or config.batch_size % world_size:
        raise ValueError("Stage-1 global batch_size must be >=2 and divisible by world_size")
    logging.info(f"stage1 (online): batch_size={config.batch_size} (per-GPU={config.batch_size // world_size}) world_size={world_size}")

    episode_dirs = _online_episode_dirs(config.seed)
    logging.info(f"stage1 (online): {len(episode_dirs)} episode dirs from VTLA_ITW_RAW_ROOT")
    normalization_path = os.environ.get("VTLA_ITW_NORMALIZATION")
    if not normalization_path:
        raise ValueError("VTLA_ITW_NORMALIZATION is required for train_stage1_online.py")
    future_frame_offset = int(os.environ.get("VTLA_STAGE1_FUTURE_OFFSET", "50"))
    default_prompt = os.environ.get("VTLA_DEFAULT_PROMPT", "Perform the task.")

    loader = _online.create_stage1_data_loader(
        config, episode_dirs, normalization_path,
        future_frame_offset=future_frame_offset, default_prompt=default_prompt,
    )
    # Bound the shared loader's infinite iterator to one epoch so DDP actually reshuffles.
    torch_loader = loader._data_loader.torch_loader
    batches_per_epoch = len(torch_loader)
    if batches_per_epoch == 0:
        raise ValueError("No complete batch available for Stage 1 (online)")
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        logging.warning(
            "Stage 1 (online) is predictor grounding (Sec 4.2), not base pretraining (Sec 4.1). "
            "Retained experimental settings: n_latent=%s (paper: 10), temperature=%s "
            "(Eq. 4: 1). EMA is not implemented in this trainer.",
            model_cfg.n_latent, model_cfg.stage1_temperature,
        )

    policy = N0VTLAPolicy(model_cfg).to(device)

    # Warm-start from the released/adapted pretrained checkpoint ALWAYS, resume included
    # (see train_stage1_predictor.py's train_loop_stage1 for the full rationale -- both
    # scripts share the same trainable-only checkpoint format via save_stage1_checkpoint/
    # load_stage1_checkpoint). A resumed run's own checkpoint is applied on top below.
    missing, unexpected = load_stage1_policy_weights(policy, weights_file, device, strict=False)
    allowed_missing = ("tactile_encoder.", "tactile_predictor.", "tactile_recon_head.", "z_proj.", "z_gate")
    missing_base = [name for name in missing if not name.startswith(allowed_missing)]
    if missing_base or unexpected:
        raise ValueError(f"Incompatible pretrained checkpoint: missing_base={missing_base}, unexpected={unexpected}")
    if is_main:
        logging.info(f"Loaded pretrained base weights from {weights_file}: missing={missing}")

    trainable_prefixes = STAGE1_TRAINABLE_PREFIXES
    n_trainable, n_frozen = 0, 0
    for name, p in policy.named_parameters():
        if name.startswith(trainable_prefixes):
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    if is_main:
        logging.info(f"stage1 (online): {n_trainable:,} trainable params, {n_frozen:,} frozen params")

    model = _Stage1Wrapper(policy)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # large frozen VLM subgraph is legitimately unused
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    peak_lr = config.lr_schedule.peak_lr
    warmup_steps = config.lr_schedule.warmup_steps
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    optim = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_stage1_checkpoint(model, optim, config.checkpoint_dir, device)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    policy.paligemma_with_expert.eval()
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Stage1Online", disable=not is_main)
    infos = []
    epoch = global_step // batches_per_epoch
    empty_batches = 0
    while global_step < config.num_train_steps:
        if use_ddp:
            torch_loader.sampler.set_epoch(epoch)
        for observation, _actions in itertools.islice(loader, batches_per_epoch):
            if global_step >= config.num_train_steps:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            optim.zero_grad()
            loss = model(observation)
            loss.backward()
            raw_policy = model.module.policy if use_ddp else model.policy
            if raw_policy._last_loss_parts["stage1_valid_count"] == 0:
                # Backward must still run on every rank to complete DDP collectives.
                # Skip AdamW too: zero gradients otherwise still trigger weight decay.
                empty_batches += 1
                if empty_batches >= batches_per_epoch:
                    raise RuntimeError("A full epoch had no valid future tactile targets; check episode length/masks")
                continue
            empty_batches = 0
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite Stage-1 loss")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.optimizer.clip_gradient_norm)
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError("Non-finite Stage-1 gradient")
            optim.step()
            global_step += 1

            raw_policy = model.module.policy if use_ddp else model.policy
            info = dict(raw_policy._last_loss_parts)
            info["grad_norm"] = float(grad_norm)
            info["lr"] = optim.param_groups[0]["lr"]
            infos.append(info)

            if is_main and (global_step == 1 or global_step % config.log_interval == 0
                            or global_step == config.num_train_steps):
                mean_info = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
                logging.info(f"step={global_step} " + " ".join(f"{k}={v:.4f}" for k, v in mean_info.items()))
                if config.wandb_enabled:
                    import wandb

                    wandb.log(mean_info, step=global_step)
                infos = []
            elif not is_main:
                infos = []

            save_stage1_checkpoint(model, optim, global_step, config, is_main)
            if is_main:
                pbar.update(1)
        epoch += 1

    if is_main:
        pbar.close()
        logging.info("Stage-1 (online) training complete.")


def main() -> None:
    init_logging()
    config = _config.cli()
    try:
        train_loop_stage1_online(config)
    finally:
        # A barrier here can deadlock if another rank failed before its next collective.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
