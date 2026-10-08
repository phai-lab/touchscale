import dataclasses
import logging
import pathlib
import socket

import tyro

from n0vtla.policies import policy as _policy
from n0vtla.policies import policy_config as _policy_config
from n0vtla.serving import websocket_policy_server
from n0vtla.training import checkpoints as _checkpoints
from n0vtla.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "vtla_tactile_posttrain").
    config: str
    # Checkpoint directory (e.g., "checkpoints/vtla_tactile_posttrain/exp/20000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Checkpoint to serve.
    policy: Checkpoint

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Inference-only memory optimization: construct the PyTorch model's backbone on the meta
    # device and load checkpoint weights via assign=True instead of building a full fp32 CPU
    # copy first. This roughly halves the peak system RAM during loading without affecting
    # results (see docs/REAL_ROBOT_INFERENCE.md §2.2). Off by default; only applies to
    # N0VTLAConfig-based configs (e.g. vtla_tactile_posttrain) and is a no-op for the JAX
    # loading path.
    low_cpu_mem_usage: bool = False


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    train_config = _config.get_config(args.policy.config)
    if args.low_cpu_mem_usage:
        if not hasattr(train_config.model, "low_cpu_mem_usage"):
            raise ValueError(
                "--low-cpu-mem-usage is only supported for N0VTLAConfig-based configs, but "
                f"'{args.policy.config}' uses {type(train_config.model).__name__}."
            )
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, low_cpu_mem_usage=True)
        )

    # The config preset's asset_id (used to locate norm_stats.json under the checkpoint's
    # assets/ dir) is the preset's default, which can differ from the asset_id a training run
    # used if it was overridden at launch time (e.g. a run trained with asset_id="my_task_train"
    # while the vtla_tactile_posttrain presets default to "canonical_tactile_task"). If the
    # preset's asset_id isn't present under this checkpoint but there is exactly one asset
    # directory, that directory is unambiguously the checkpoint's own norm_stats, so use it
    # instead of failing.
    norm_stats = None
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    assets_root = pathlib.Path(args.policy.dir) / "assets"
    if data_config.asset_id and not (assets_root / data_config.asset_id).exists() and assets_root.is_dir():
        candidates = [p.name for p in assets_root.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            logging.info(
                "Config asset_id '%s' not found under %s; falling back to this checkpoint's own "
                "asset dir '%s'.",
                data_config.asset_id,
                assets_root,
                candidates[0],
            )
            norm_stats = _checkpoints.load_norm_stats(assets_root, candidates[0])

    return _policy_config.create_trained_policy(
        train_config, args.policy.dir, default_prompt=args.default_prompt, norm_stats=norm_stats
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        # Only used for the log line below; an unresolvable hostname must not take down a
        # server that has already paid for loading the checkpoint.
        local_ip = "unresolved"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
