#!/usr/bin/env python
"""Stage-1 predictor-grounding pretraining (paper arXiv:2607.23782 Sec 4.2) -- ACTION-FREE.

See docs/MID_TRAIN.md for the design. This script trains ONLY
``tactile_encoder.tactile_proj`` + ``tactile_predictor`` + ``tactile_recon_head`` against the
paper's L_1 = L_NCE + lambda_rec * L_rec objective (N0VTLAPolicy.forward_stage1); the rest of
the base policy (PaliGemma VLM + Gemma action expert) is frozen and never called with grad.
No ground-truth robot actions are read or required, so this can train on data that has none
(e.g. ITW hand/glove tactile episodes) as long as the canonical dataset has a valid
``.future`` tactile frame (``future_frame_offset>0`` on the DataConfig).

Usage (single node):
    torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \\
        scripts/train_stage1_predictor.py vtla_stage1_predictor_pretrain \\
        --exp-name=my_stage1_run

Relies on the SAME env vars as train.sh's post-training path (VTLA_DATASET_PATH,
VTLA_PRETRAINED_CHECKPOINT, VTLA_ASSET_ID, ...) plus VTLA_STAGE1_* knobs -- see the
``vtla_stage1_predictor_pretrain`` TrainConfig in n0vtla/training/config.py and
docs/MID_TRAIN.md.
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import tqdm

# Sibling scripts live in the same directory; import by path so this works whether invoked as
# `python scripts/train_stage1_predictor.py` or `torchrun ... scripts/train_stage1_predictor.py`
# (neither puts the repo root on sys.path the way `-m` would).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pytorch import (  # noqa: E402
    get_latest_checkpoint_step,
    init_logging,
    init_wandb,
    set_seed,
    setup_ddp,
    wait_for_path_state,
)

import n0vtla.training.config as _config  # noqa: E402
import n0vtla.training.data_loader as _data  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402


class _Stage1Wrapper(nn.Module):
    """Routes DDP's ``forward()`` to ``N0VTLAPolicy.forward_stage1``.

    DDP's gradient-sync hooks (``prepare_for_backward`` / bucket rebuild) are wired into
    ``DistributedDataParallel.forward`` specifically -- calling a differently-named method on a
    DDP-wrapped module (even via the ``nn.Module.__getattr__`` passthrough to ``.module``) skips
    that setup and desyncs gradients across ranks. Wrapping the stage-1 loss as this submodule's
    ``forward`` lets ``model(observation)`` be correct whether or not ``model`` is DDP-wrapped.
    """

    def __init__(self, policy: N0VTLAPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy.forward_stage1(observation)


def load_stage1_policy_weights(policy, weights_file, device, *, strict):
    # Released robot checkpoints may carry an action-conditioning gate even when the
    # Stage-1 config does not construct it. Preserve it unchanged for later action stages.
    with safetensors.safe_open(weights_file, framework="pt", device="cpu") as weights:
        if "z_gate" in weights.keys() and not hasattr(policy, "z_gate"):
            policy.register_parameter("z_gate", nn.Parameter(weights.get_tensor("z_gate").to(device),
                                                             requires_grad=False))
            policy.z_gate_zero_init = True
    return safetensors.torch.load_model(policy, weights_file, strict=strict, device=str(device))


# Must match train_loop_stage1's/train_loop_stage1_online's own freeze-selection exactly
# (paper: "With the entire base policy frozen, we train only the predictor, the tactile
# projection, and a lightweight reconstruction head") -- this is the ONLY thing a
# trainable-only checkpoint saves, so a mismatch would drop trained state.
STAGE1_TRAINABLE_PREFIXES = ("tactile_encoder.tactile_proj.", "tactile_predictor.", "tactile_recon_head.")


def save_stage1_checkpoint(model, optimizer, global_step, config, is_main):
    """Trainable-only Stage-1 checkpoint (~123M params: tactile_encoder.tactile_proj +
    tactile_predictor + tactile_recon_head), NOT a full policy snapshot.

    The frozen ~3.7B-param base (PaliGemma VLM + Gemma action expert + z_gate) never
    changes during Stage-1 and is always reloaded from config.pytorch_weight_path
    on every launch, resume included (see train_loop_stage1/train_loop_stage1_online),
    so it is not duplicated into every checkpoint. Only the trainable delta is saved
    (plus optimizer state) -- a resumed run reconstructs the full model by loading the
    base checkpoint first, then this file on top (see load_stage1_checkpoint).
    global_step counts completed optimizer updates.
    """
    if not is_main:
        return
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or (
        global_step == config.num_train_steps
    )
    if not should_save:
        return
    final_dir = config.checkpoint_dir / f"{global_step}"
    tmp_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module.policy if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.policy
    trainable_state = {
        name: param.detach().to("cpu").contiguous()
        for name, param in raw_model.named_parameters()
        if name.startswith(STAGE1_TRAINABLE_PREFIXES)
    }
    safetensors.torch.save_file(trainable_state, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(
        {"global_step": global_step, "step_format": "completed_updates",
         "config": dataclasses.asdict(config), "timestamp": time.time(),
         "checkpoint_format": "trainable_only_v1"},
        tmp_dir / "metadata.pt",
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved stage-1 checkpoint (trainable-only, {len(trainable_state)} tensors) at step {global_step} -> {final_dir}")


def load_stage1_checkpoint(model, optimizer, checkpoint_dir, device):
    """Loads the trainable-only delta (see save_stage1_checkpoint) ON TOP of whatever
    base weights the caller already loaded into `model`. A trainable-only checkpoint
    alone cannot reconstruct the whole model -- the caller MUST have already run
    load_stage1_policy_weights(base) before calling this (both train_loop_stage1 and
    train_loop_stage1_online do this unconditionally, resume included).
    """
    steps = [
        int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(steps)
    ckpt_dir = checkpoint_dir / f"{latest}"
    raw_model = model.module.policy if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.policy
    missing, unexpected = safetensors.torch.load_model(
        raw_model, ckpt_dir / "model.safetensors", strict=False, device=str(device)
    )
    if unexpected:
        raise ValueError(f"Unexpected keys in trainable-only checkpoint {ckpt_dir}: {unexpected}")
    missing_trainable = [name for name in missing if name.startswith(STAGE1_TRAINABLE_PREFIXES)]
    if missing_trainable:
        raise ValueError(f"Trainable-only checkpoint {ckpt_dir} is missing trainable params: {missing_trainable}")
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=False))
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
    step = metadata.get("global_step", latest)
    return step if metadata.get("step_format") == "completed_updates" else step + 1


def train_loop_stage1(config: _config.TrainConfig) -> None:
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    global_rank = dist.get_rank() if use_ddp else 0
    set_seed(config.seed, global_rank)

    model_cfg = config.model
    assert getattr(model_cfg, "stage1_pretrain_enabled", False), (
        "train_stage1_predictor.py requires a config with stage1_pretrain_enabled=True "
        "(use the 'vtla_stage1_predictor_pretrain' TrainConfig or a variant of it)"
    )
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    if not model_cfg.tactile_predictor_enabled or model_cfg.tactile_mode != "latent":
        raise ValueError("Stage 1 requires the latent tactile predictor")
    # Needed on EVERY launch, resume included: trainable-only checkpoints (see
    # save_stage1_checkpoint) don't carry the frozen base, so it's always reloaded from
    # here first, with a resumed run's own checkpoint applied on top afterward.
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
                logging.info(f"Resuming stage-1 run from step {latest_step}")
            else:
                raise FileNotFoundError(f"No valid checkpoints in {config.checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"{config.checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
    elif config.checkpoint_dir.exists():
        raise FileExistsError(f"{config.checkpoint_dir} exists; use --resume or a new exp-name")

    if use_ddp:
        dist.barrier()

    if not resuming:
        ready_file = config.checkpoint_dir / ".ddp_fs_ready"
        if is_main:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ready_file.write_text("ready\n", encoding="utf-8")
        elif use_ddp:
            wait_for_path_state(ready_file, should_exist=True)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size < 2 or config.batch_size % world_size:
        raise ValueError("Stage-1 global batch_size must be >=2 and divisible by world_size")
    logging.info(f"stage1: batch_size={config.batch_size} (per-GPU={config.batch_size // world_size}) world_size={world_size}")

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True, skip_norm_stats=True)
    # Bound the shared loader's infinite iterator to one epoch so DDP actually reshuffles.
    torch_loader = loader._data_loader.torch_loader
    batches_per_epoch = len(torch_loader)
    if batches_per_epoch == 0:
        raise ValueError("No complete batch available for Stage 1")
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        logging.warning(
            "Stage 1 is predictor grounding (Sec 4.2), not base pretraining (Sec 4.1). "
            "Retained experimental settings: n_latent=%s (paper: 10), temperature=%s "
            "(Eq. 4: 1). EMA is not implemented in this trainer.",
            model_cfg.n_latent, model_cfg.stage1_temperature,
        )

    policy = N0VTLAPolicy(model_cfg).to(device)

    # Warm-start from the released/adapted pretrained checkpoint ALWAYS, resume included
    # (the predictor weights if present; tactile_recon_head is a new module that is not
    # in a base checkpoint, so it always comes up randomly initialized here -- expected,
    # not an error). A resumed run's own trainable-only checkpoint is
    # applied ON TOP of this further down, overwriting just the trained submodules.
    missing, unexpected = load_stage1_policy_weights(policy, weights_file, device, strict=False)
    allowed_missing = ("tactile_encoder.", "tactile_predictor.", "tactile_recon_head.", "z_proj.", "z_gate")
    missing_base = [name for name in missing if not name.startswith(allowed_missing)]
    if missing_base or unexpected:
        raise ValueError(f"Incompatible pretrained checkpoint: missing_base={missing_base}, unexpected={unexpected}")
    if is_main:
        logging.info(f"Loaded pretrained base weights from {weights_file}: missing={missing}")

    # Freeze everything except the three Stage-1-trainable submodules (paper: "With the entire
    # base policy frozen, we train only the predictor, the tactile projection, and a lightweight
    # reconstruction head").
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
        logging.info(f"stage1: {n_trainable:,} trainable params, {n_frozen:,} frozen params")

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
    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Stage1", disable=not is_main)
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
        logging.info("Stage-1 training complete.")


def main() -> None:
    init_logging()
    config = _config.cli()
    try:
        train_loop_stage1(config)
    finally:
        # A barrier here can deadlock if another rank failed before its next collective.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
