"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `n0vtla/training/config.py` and `n0vtla/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py vtla_tactile_posttrain --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py vtla_tactile_posttrain --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import faulthandler
import gc
import logging
import os
import platform
import shutil
import signal
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import n0vtla.models.model
import n0vtla.models.pi0_config
import n0vtla.models_pytorch.gemma_pytorch
import n0vtla.models_pytorch.pi0_pytorch
import n0vtla.shared.normalize as _normalize
import n0vtla.training.config as _config
import n0vtla.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

    if os.environ.get("N0VTLA_ENABLE_FAULTHANDLER", "0") == "1":
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, rank: int):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def wait_for_path_state(path, *, should_exist: bool, timeout_s: float = 120.0, poll_s: float = 0.1):
    deadline = time.time() + timeout_s
    while path.exists() != should_exist:
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {path} to exist={should_exist}")
        time.sleep(poll_s)


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    rank = dist.get_rank() if use_ddp else 0
    debug_step0 = os.environ.get("N0VTLA_DDP_DEBUG_STEP0", "0") == "1"
    debug_trace_file = os.environ.get("N0VTLA_DDP_DEBUG_FILE")

    def append_trace_file(message: str) -> None:
        if debug_trace_file:
            with open(f"{debug_trace_file}.rank{rank}", "a", encoding="utf-8") as f:
                f.write(f"[rank {rank}] {message}\n")

    def trace_debug(message: str) -> None:
        if not (debug_step0 and use_ddp):
            return
        append_trace_file(message)
        logging.info("[rank %d] %s", rank, message)

    # Seed by GLOBAL rank: seed+local_rank gives every node's rank-k the same RNG stream
    # (flow-matching noise duplicated 8-way across nodes on 64 GPUs).
    global_rank = dist.get_rank() if use_ddp else int(os.environ.get("RANK", local_rank))
    set_seed(config.seed, global_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory.
        # In DDP only rank 0 should mutate the filesystem, then other ranks wait.
        exp_checkpoint_dir = config.checkpoint_dir
        ready_file = exp_checkpoint_dir / ".ddp_fs_ready"
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ready_file.write_text("ready\n", encoding="utf-8")
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        elif use_ddp:
            wait_for_path_state(ready_file, should_exist=True)
        if use_ddp:
            append_trace_file("after_checkpoint_barrier")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process). Skip initialization entirely when disabled:
    # even "mode=disabled" can block rank 0 and stall multi-GPU startup.
    if is_main and config.wandb_enabled:
        init_wandb(config, resuming=resuming, enabled=True)
        trace_debug("after_wandb_init")

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    append_trace_file("before_batch_size_log")
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )
    append_trace_file("after_batch_size_log")

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    trace_debug("before_build_datasets")
    loader, data_config = build_datasets(config)
    trace_debug("loader_ready")

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, n0vtla.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
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
            tactile_image_keys=getattr(
                config.model, "tactile_image_keys", n0vtla.models.model.TACTILE_IMAGE_KEYS
            ),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    trace_debug("before_model_init")
    model = n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)
    trace_debug("after_model_init")
    model = model.to(device)
    trace_debug("model_ready")

    # VTLA_NO_CKPT=1 skips gradient checkpointing entirely — REQUIRED for use_prefix_cache
    # (HF silently drops the prefix KV under checkpointing; equiv-verified lossless without it).
    # Trade: more activation memory, less recompute — pair with a batch size probe.
    if os.environ.get("VTLA_NO_CKPT") == "1":
        enable_gradient_checkpointing = False
        logging.info("VTLA_NO_CKPT=1: gradient checkpointing DISABLED (prefix-KV-cache mode)")
    elif hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    if is_main:
        logging.info(
            "Effective config: VTLA_ATTN_IMPL=%s VTLA_PREFIX_CACHE=%s gradient_checkpointing=%s",
            getattr(n0vtla.models_pytorch.gemma_pytorch, "_VTLA_ATTN_IMPL", "eager"),
            getattr(model, "use_prefix_cache", os.environ.get("VTLA_PREFIX_CACHE") == "1"),
            "on" if enable_gradient_checkpointing else "off",
        )

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # VTLA_PREDICTOR_FREEZE=1: freeze the tactile perception stack (tactile_predictor.* + tactile_encoder.*)
    # in e2e — z becomes a FIXED function of inputs, killing the predictor-drift feedback loop by
    # exponential grad ramp from ~step 2000, same signature as random init — so the drift source is
    # "predictor trainable with no anchor in e2e", not init quality). z_proj (linear adapter into the
    # expert) and z_gate stay trainable. Default off = unchanged behavior.
    if os.environ.get("VTLA_PREDICTOR_FREEZE", "0") == "1":
        raw_m = model
        frozen_n = 0
        for n, p in raw_m.named_parameters():
            if n.startswith(("tactile_predictor.", "tactile_encoder.")):
                p.requires_grad = False
                frozen_n += 1
        if is_main:
            logging.info(f"VTLA_PREDICTOR_FREEZE=1: froze {frozen_n} tactile_predictor/tactile_encoder tensors "
                         f"(z_proj + z_gate stay trainable)")


    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )
        trace_debug("ddp_ready")

    # Load weights from weight_loader if specified (for fine-tuning). DDP's default
    # init_sync=True (above) already broadcasts rank 0's initial state to every rank before
    # this runs, so a strict=False load's missing keys (left at their pre-load, already-synced
    # random init) stay identical across ranks with no extra broadcast needed -- every rank
    # loading the same file from disk independently is enough.
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        missing, unexpected = safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
            model_path,
            strict=False,
        )
        if missing or unexpected:
            logging.info(f"Loaded base weights with missing keys={missing}, unexpected keys={unexpected}")
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    trace_debug("weights_ready")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters. VTLA_PREDICTOR_LR_SCALE puts the tactile-predictor heads
    # (freshly-initialized, fastest-moving params) in their own group at scale*lr — damper for the
    # predictor-drift feedback loop at peak lr (default 1.0 = single group, unchanged behavior).
    #
    # first N steps with the tactile FEATURE stack (tactile_predictor + tactile_encoder) frozen via
    # lr=0 (DDP-safe: requires_grad untouched, buckets intact) and vl_dropout forced to 1.0 (all
    # samples' suffix cut from the VL prefix — task info reaches the expert ONLY through z, since
    # arch C's z still sees the detached vl_ctx as query conditioner). z_proj (the random "wiring")
    # trains at FULL lr in phase A. After the boundary the vl mask ramps back to the config value
    # over VTLA_PHASE_RAMP_STEPS (default 500) and z_proj drops to predictor_lr_scale. This replaces
    # the zero-init z_gate's implicit 10k-step slow opening (instability suspect #2) with a short
    # explicit alignment phase followed by CONSTANT dynamics.
    predictor_lr_scale = float(os.environ.get("VTLA_PREDICTOR_LR_SCALE", "1.0"))
    phase_a_steps = int(os.environ.get("VTLA_PHASE_A_STEPS", "0"))
    phase_ramp_steps = int(os.environ.get("VTLA_PHASE_RAMP_STEPS", "500"))
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if phase_a_steps > 0:
        feat_prefixes = ("tactile_predictor.", "tactile_encoder.")
        zwire_prefixes = ("z_proj.",)
        base_params, feat_params, zwire_params = [], [], []
        for n, p in raw_model.named_parameters():
            if n.startswith(feat_prefixes):
                feat_params.append(p)
            elif n.startswith(zwire_prefixes):
                zwire_params.append(p)
            else:
                base_params.append(p)
        param_groups = [
            {"params": base_params},
            {"params": feat_params, "lr_scale": predictor_lr_scale, "phase_frozen": True},
            {"params": zwire_params, "lr_scale": predictor_lr_scale, "phase_zwire": True},
        ]
        if is_main:
            logging.info(
                f"phase-A curriculum ENABLED: steps<{phase_a_steps} freeze feat "
                f"({len(feat_params)} tensors, lr=0) + vl_dropout=1.0; z_proj "
                f"({len(zwire_params)} tensors) at FULL lr in phase A -> {predictor_lr_scale}x after; "
                f"vl mask ramps to config over {phase_ramp_steps} steps"
            )
    elif predictor_lr_scale != 1.0:
        predictor_prefixes = ("tactile_predictor.", "z_proj.", "tactile_encoder.")
        base_params = [p for n, p in raw_model.named_parameters() if not n.startswith(predictor_prefixes)]
        predictor_params = [p for n, p in raw_model.named_parameters() if n.startswith(predictor_prefixes)]
        param_groups = [
            {"params": base_params},
            {"params": predictor_params, "lr_scale": predictor_lr_scale},
        ]
        if is_main:
            logging.info(
                f"Predictor param group: {len(predictor_params)} tensors at lr_scale={predictor_lr_scale} "
                f"(base group: {len(base_params)} tensors)"
            )
    else:
        param_groups = model.parameters()
    optim = torch.optim.AdamW(
        param_groups,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval

    def trace_step0(message: str) -> None:
        if not (debug_step0 and use_ddp and (global_step == 0 or global_step == 1)):
            return
        trace_debug(message)

    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    nonfinite_grad_skips = 0
    sample_clip_skips = 0
    sample_loss_clip = float(os.environ.get("VTLA_SAMPLE_LOSS_CLIP", "0"))
    if is_main and sample_loss_clip > 0:
        logging.info(f"sample-loss-clip ENABLED (threshold={sample_loss_clip})")
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        trace_step0("before_outer_while")
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))
            trace_step0("after_set_epoch")

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            trace_step0("step0 batch fetched")
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901
            trace_step0("step0 batch moved_to_device")

            # Update LR (per-group scale for the predictor group, 1.0 otherwise). Phase-A curriculum:
            # feat group lr=0 and vl mask 1.0 during phase A; z_proj full lr in A, scaled after;
            # mask ramps 1.0 -> config value over phase_ramp_steps at the boundary.
            in_phase_a = phase_a_steps > 0 and global_step < phase_a_steps
            for pg in optim.param_groups:
                scale = pg.get("lr_scale", 1.0)
                if phase_a_steps > 0:
                    if pg.get("phase_frozen") and in_phase_a:
                        scale = 0.0
                    elif pg.get("phase_zwire") and in_phase_a:
                        scale = 1.0
                pg["lr"] = lr_schedule(global_step) * scale
            if phase_a_steps > 0:
                if in_phase_a:
                    raw_model._vl_dropout_override = 1.0
                elif global_step < phase_a_steps + phase_ramp_steps:
                    frac = (global_step - phase_a_steps) / max(phase_ramp_steps, 1)
                    cfg_p = float(getattr(raw_model.config, "vl_dropout_prob", 0.0))
                    raw_model._vl_dropout_override = 1.0 + (cfg_p - 1.0) * frac
                else:
                    raw_model._vl_dropout_override = None
                if is_main and global_step in (phase_a_steps, phase_a_steps + phase_ramp_steps):
                    logging.info(
                        f"phase-A curriculum: boundary at step={global_step} "
                        f"(feat unfrozen, vl mask {'ramping' if global_step == phase_a_steps else 'at config value'})"
                    )

            # Forward pass
            losses = model(observation, actions)
            trace_step0("step0 forward_done")
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            # Sample-loss clip (opt-in via VTLA_SAMPLE_LOSS_CLIP>0): zero-weight outlier samples
            # in the batch mean. Real manipulation corpora have heavy tails — state-jump chunks
            # and vigorous motion, amplified when a low-variance dimension is normalised — which
            # can reach tens of sigma. Once the healthy loss has floored (order 0.04) those few
            # samples' gradient can dominate the batch direction and the loss climbs steadily
            # from there. This is a circuit breaker, not data cleaning: which samples exceed the
            # threshold is stochastic in the flow-matching (t, eps) draw, so nothing is
            # permanently excluded.
            if sample_loss_clip > 0 and losses.dim() >= 1 and losses.shape[0] > 1:
                per_sample = losses.reshape(losses.shape[0], -1).mean(dim=1)
                # Non-finite samples must be masked via where(): NaN * 0 is still NaN and would
                # poison the whole batch instead of just the bad sample.
                keep = (per_sample <= sample_loss_clip) & torch.isfinite(per_sample)
                kept_sum = torch.where(keep, per_sample, torch.zeros_like(per_sample)).sum()
                # Global kept count: per-rank kept-mean + DDP equal-rank averaging would
                # over-weight samples on high-skip ranks; also makes skip stats globally
                # meaningful (rank0's local 128-slice has no verdict power at bs 9216).
                n_kept = keep.sum().float()
                n_skip_t = (~keep).sum().float()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    counts = torch.stack([n_kept, n_skip_t])
                    torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                    global_kept, global_skip = counts[0], counts[1]
                else:
                    world_size = 1
                    global_kept, global_skip = n_kept, n_skip_t
                n_skip = int(global_skip.item())
                if n_skip > 0:
                    sample_clip_skips += n_skip
                    # All ranks agree on global_skip (post all-reduce), so this collective is
                    # consistent across ranks — no deadlock.
                    local_max = torch.where(
                        torch.isfinite(per_sample), per_sample, torch.zeros_like(per_sample)
                    ).max()
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
                    if is_main:
                        global_batch = int(losses.shape[0]) * world_size
                        logging.info(
                            f"sample-loss-clip: skipped {n_skip}/{global_batch} samples this step "
                            f"(global max per-sample loss {local_max.item():.2f}, total skipped {sample_clip_skips})"
                        )
                # DDP mean-averages rank losses; scaling by world_size/global_kept makes the
                # effective loss the exact global kept-mean regardless of per-rank skip counts.
                loss = kept_sum * world_size / global_kept.clamp(min=1.0)
            else:
                loss = losses.mean()

            # Backward pass
            loss.backward()
            trace_step0("step0 backward_done")

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available() and not use_ddp:
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
            trace_step0("step0 clip_done")

            # Non-finite guard: a bf16 overflow makes clip_coef 0 or NaN, and one poisoned
            # optim.step() is enough to wreck the rest of the run. grad_norm is identical on all
            # ranks (it is computed on allreduced grads), so skipping the step is DDP-consistent.
            if not torch.isfinite(grad_norm):
                nonfinite_grad_skips += 1
                if is_main:
                    logging.warning(
                        f"non-finite grad_norm={grad_norm} at step {global_step}; skipping optimizer step "
                        f"(total skips={nonfinite_grad_skips})"
                    )
                optim.zero_grad(set_to_none=True)
                trace_step0("step0 optimizer_skipped_nonfinite")
            else:
                # Optimizer step
                optim.step()
                optim.zero_grad(set_to_none=True)
            trace_step0("step0 optimizer_done")

            # Collect stats
            if is_main:
                trace_step0("step0 before_collect_stats")
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )
                trace_step0("step0 after_collect_stats")
                # Per-step loss line, so two runs can be compared step for step. Independent
                # of the log_interval averaging below, which is left untouched.
                print(
                    f"STEP {global_step} loss={loss.item()} lr={optim.param_groups[0]['lr']}",
                    flush=True,
                )

            if is_main and (global_step % config.log_interval == 0):
                trace_step0("step0 before_log_interval_block")
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection
                trace_step0("step0 after_log_interval_block")

            global_step += 1
            # Save checkpoint using the new mechanism
            trace_step0("step0 before_save_checkpoint")
            save_checkpoint(model, optim, global_step, config, is_main, data_config)
            trace_step0("step0 after_save_checkpoint")

            # Update progress bar
            if pbar is not None:
                trace_step0("step0 before_pbar")
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )
                trace_step0("step0 after_pbar")

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
