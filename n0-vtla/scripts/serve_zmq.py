#!/usr/bin/env python
"""N0-VTLA inference server: one ZMQ REP + msgpack serve for every shipped config.

Serves an N0-VTLA policy with or without the latent tactile predictor. The tactile path is
auto-detected from the training config, so the same serve and the same wire protocol drive all
of them:

  * ``flexiv_vision_reference``   base policy, no tactile    (FlexivEEFInputs)
  * ``flexiv_tactile_reference``  base + tactile predictor   (FlexivTactileInputs)
  * ``sim_single_arm_tactile``    UniVTAC single-arm, joint  (FlexivTactileInputs)
  * ``sim_dual_arm_tactile``      NeoSim dual-arm, joint     (FlexivTactileInputs)

Why the data dict is built here (not via a repack transform)
------------------------------------------------------------
``create_trained_policy`` does NOT run the DataConfig's RepackTransform (it only applies the
``repack_transforms`` argument, which is empty by default). The first transform the policy
actually applies is ``data_config.data_transforms.inputs[0]`` — i.e. ``FlexivEEFInputs`` for
control and ``FlexivTactileInputs`` for prior. So this serve constructs exactly the dict those
transforms consume:

    data["images"] = {"cam_high": HWC, "cam_left_wrist": HWC, "cam_right_wrist": HWC}
    data["state"]  = float32[32]
    data["prompt"] = str            (optional; InjectDefaultPrompt fills the default)
    data["tactile"] = {<view>: uint8[2, H, W, 3]}   # prior only; [baseline, current]

``FlexivTactileInputs`` splits each tactile stack into two model image keys ``<view>`` (current,
= stack[1]) and ``<view>.baseline`` (= stack[0]); ``N0VTLAPolicy._preprocess_observation``
pops both back out before SigLIP and feeds ``DINOv2(tac_t - tac_0)`` to the prior.

Per-episode tactile baseline (tac_0)
------------------------------------
The prior's contact signal is ``tac_t - tac_0`` where ``tac_0`` is the FIRST frame of the
episode (training loads it via ``delta_timestamps = [-BIG, 0]``, which LeRobot clamps to
frame 0). This serve reproduces that at inference:

  * ``reset``   clears the stored baseline. Call it at the START of every episode.
  * ``predict`` on the first call after a reset captures the current tactile as the baseline
    (so frame 0 has ``tac_t == tac_0`` -> zero contact, matching training); every subsequent
    predict pairs the live tactile (tac_t) with that stored baseline (tac_0).

Wire protocol (msgpack, ZMQ REP)
--------------------------------
request  {"cmd": "reset"}                       -> {"status": "ok"}
request  {"cmd": "predict",                      -> {"status": "ok",
          "state": [float]*<=32,                     "actions": [[float]*32]*horizon,
          "prompt": str (optional),                  "infer_time_ms": float}
          "observation/image":          img,     # -> cam_high        (base)
          "observation/wrist_image":    img,     # -> cam_left_wrist  (left wrist)
          "observation/secondary_image": img,    # -> cam_right_wrist (right wrist)
          # tactile keys ONLY for tactile configs (ignored by the base policy):
          "observation/left_tactile":   img,     # -> tactile view 0
          "observation/right_tactile":  img,     # -> tactile view 1
          ...}                                   # 4 views for sim_dual_arm_tactile
where ``img`` is PNG/JPG bytes OR an HWC/CHW uint8 array (as nested lists), at its NATIVE
resolution -- do not pre-resize, the model transform letterboxes to 224x224.
On any failure: {"status": "error", "message": str}.

The camera and tactile key sets are per-config, so the protocol is not identical across
configs. Send one tactile key per model view: 2 for ``sim_single_arm_tactile``, 4 for
``sim_dual_arm_tactile`` (the serve logs the view list at startup). Under-supplying silently
drops the tail views. ``sim_single_arm_tactile`` has no right-wrist camera and ignores
``observation/secondary_image``, warning once.

Actions are returned padded to 32 dims. What occupies the leading dims depends on the config:
the end-effector configs emit 10 dims (``[xyz(3), rot6d(6), grip(1)]``), while the released
simulation configs emit JOINT actions, 8 (single-arm) or 16 (dual-arm). Read the first
``raw_action_dim`` columns and ignore the zero padding.

DEPLOYMENT — the client MUST execute the FULL action horizon
------------------------------------------------------------
``predict`` returns ``action_horizon`` action steps — 50 for most configs, 16 for
``sim_dual_arm_tactile``; the serve logs the value at startup. The client MUST execute ALL of
them before requesting the next chunk. A shorter execution stride drops the TAIL of every
chunk, and the gripper-close commands live in that tail, so the arm reaches the right position
but the gripper never closes. Execute the full horizon.

Usage
-----
  # a policy you post-trained yourself
  python scripts/serve_zmq.py \
      --config flexiv_tactile_reference \
      --ckpt checkpoints/flexiv_tactile_reference/<exp>/<step> \
      --addr "tcp://*:5557" --default-prompt "do the task"

  # a released simulation policy (VTLA_ASSET_ID names the dir under the ckpt's assets/)
  VTLA_ASSET_ID=n0_insert_hole_norm python scripts/serve_zmq.py \
      --config sim_single_arm_tactile \
      --ckpt checkpoints/n0_VTLA_insert_hole \
      --addr "tcp://*:5557" --default-prompt "insert hole"
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
import time

import cv2
import msgpack
import numpy as np
import zmq

from n0vtla.policies import policy_config as _policy_config
from n0vtla.training import config as _config

# Client wire key -> model camera key consumed by FlexivEEFInputs / FlexivTactileInputs.
# All 3 RGB views are always sent. The model camera keys correspond, in the LeRobot
# dataset, to third_view / left_wrist_view / second_third_view respectively.
CAM_MAP = {
    "observation/image": "cam_high",                   # third_view (base / overhead)
    "observation/wrist_image": "cam_left_wrist",       # left_wrist_view
    "observation/secondary_image": "cam_right_wrist",  # second_third_view
}

# Preferred ordering of tactile wire keys. Client tactile frames are matched POSITIONALLY onto
# the model's ``tactile_image_keys`` (the authoritative, ordered view list from the training
# config): after sorting the client's tactile keys by this preference (then alphabetically for
# any not listed), key i feeds tactile view i. Single-arm gripper = 2 fingers; dual-arm = 4.
TAC_WIRE_ORDER = (
    "observation/left_tactile",
    "observation/right_tactile",
    "observation/left_wrist_left_tactile",
    "observation/left_wrist_right_tactile",
    "observation/right_wrist_left_tactile",
    "observation/right_wrist_right_tactile",
)

# A tactile wire key is any request key of this shape.
_TAC_PREFIX = "observation/"
_TAC_SUFFIX = "_tactile"

# Escape hatch for datasets that were themselves built by squashing native frames to 224x224.
# Off by default: the released checkpoints were trained on native-resolution frames that
# ResizeImages letterboxed, so squashing here would break the train/serve geometry match.
_SQUASH = os.environ.get("VTLA_SERVE_SQUASH", "").lower() in ("1", "true", "yes")


def _decode_image(v) -> np.ndarray:
    """bytes(PNG/JPG) | HWC | CHW -> RGB uint8 HWC, at the frame's native resolution.

    Deliberately does NOT resize. The policy's ``model_transforms`` end with
    ``ResizeImages(224, 224)``, which is ``resize_with_pad``: aspect-preserving, letterboxed
    with black bars. Handing it the native frame is what training did, so serve and train agree.

    Squashing here instead would make that step a no-op: the simulator streams 270x480 RGB and
    240x320 tactile, whose letterboxes are 44% and 25% black bars, so the model would receive a
    geometry it never saw. Set ``VTLA_SERVE_SQUASH=1`` only if your dataset was itself built by
    squashing native frames to 224x224, in which case training saw squashed frames too.
    """
    if isinstance(v, (bytes, bytearray)):
        im = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("image decode failed")
        arr = im[..., ::-1].copy()  # BGR -> RGB
    else:
        arr = np.asarray(v, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if _SQUASH and arr.shape[:2] != (224, 224):
        arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR)
    return arr


def _pad_actions(actions, width: int = 32) -> np.ndarray:
    """Zero-pad the last action dim up to ``width`` (FlexivEEFOutputs emits 10; client reads 0:10)."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[-1] < width:
        pad = np.zeros((*a.shape[:-1], width - a.shape[-1]), dtype=a.dtype)
        a = np.concatenate([a, pad], axis=-1)
    return a


class VtlaZmqServer:
    """ZMQ REP server driving a control or tactile-conditioned policy over msgpack."""

    def __init__(
        self,
        policy,
        addr: str,
        action_horizon: int,
        default_prompt: str | None,
        tactile_views: list[str],
        state_dim: int = 32,
        cameras: Sequence[str] | None = None,
    ):
        self._policy = policy
        self._addr = addr
        # Model camera slots this checkpoint was trained with. A single-arm embodiment has no
        # right wrist camera: its dataset has no wrist_r column, so that slot was a zero image
        # with image_mask=False for the whole run. Accepting the client's secondary_image there
        # would hand the action expert 256 SigLIP tokens it never saw, silently. None => accept
        # every slot (the real-robot configs, which do carry three cameras).
        self._cameras = set(CAM_MAP.values()) if cameras is None else set(cameras)
        self._warned_cameras: set[str] = set()
        # Proprio width the config's norm_stats were computed at. Flexiv real-robot configs
        # keep the canonical padded-32 layout; the sim configs are raw_action_dim wide
        # (10 single-arm EEF, 20 dual-arm EEF, 8/16 joint). Padding to the wrong width makes
        # Normalize fail with a broadcast error against the stats.
        self._state_dim = int(state_dim)
        # Model action horizon. Informational only (logged at
        # startup); the server returns whatever the policy emits and does NOT truncate. The
        # client MUST execute all action_horizon steps — see the module DEPLOYMENT note.
        self._horizon = action_horizon
        self._default_prompt = default_prompt
        # tactile_views is the ordered list of model view names (empty => control/base model).
        self._tactile_views = list(tactile_views)
        self._tactile_enabled = bool(tactile_views)
        # Per-episode tactile baseline (tac_0): {view_name: HWC uint8}. None until first predict
        # after a reset.
        self._baseline: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ reset
    def _reset(self) -> dict:
        """Start a new episode: drop the tactile baseline so the next predict re-captures it."""
        self._baseline = None
        if self._tactile_enabled:
            logging.info("reset: cleared tactile baseline (%d views)", len(self._tactile_views))
        else:
            logging.info("reset: no-op (control model has no tactile)")
        return {"status": "ok"}

    # -------------------------------------------------------------- tactile
    def _collect_current_tactile(self, msg: dict) -> dict[str, np.ndarray]:
        """Decode the tactile frames the client sent, keyed by MODEL view name.

        Client tactile keys are ``observation/<x>_tactile``; they are ordered by
        ``TAC_WIRE_ORDER`` (then alphabetically) and matched positionally onto
        ``self._tactile_views``. Views the client did not send are simply omitted. Generalizes
        to N views (2 single-arm / 4 dual-arm) — no hardcoded count.
        """
        wire_keys = [
            k
            for k in msg
            if isinstance(k, str) and k.startswith(_TAC_PREFIX) and k.endswith(_TAC_SUFFIX)
        ]
        wire_keys.sort(
            key=lambda k: (TAC_WIRE_ORDER.index(k) if k in TAC_WIRE_ORDER else len(TAC_WIRE_ORDER), k)
        )
        if len(wire_keys) > len(self._tactile_views):
            logging.warning(
                "client sent %d tactile views but model expects %d (%s); using the first %d",
                len(wire_keys),
                len(self._tactile_views),
                self._tactile_views,
                len(self._tactile_views),
            )
        return {view: _decode_image(msg[wk]) for view, wk in zip(self._tactile_views, wire_keys)}

    # ------------------------------------------------------------- predict
    def _predict(self, msg: dict) -> dict:
        """Run one inference and return the reply dict.

        Builds exactly the dict FlexivEEFInputs/FlexivTactileInputs consume (images, state,
        optional prompt, optional tactile), runs the policy, and zero-pads actions to 32 dims.
        For prior models, pairs the live tactile (tac_t) with the per-episode baseline (tac_0),
        capturing that baseline on the FIRST predict after a reset.

        Returns the FULL model action horizon (``action_horizon`` steps); the client MUST
        execute every step — see the DEPLOYMENT note in the module docstring.
        """
        t0 = time.monotonic()

        # state -> self._state_dim (must match the width norm_stats were computed at;
        # FlexivEEFInputs/FlexivTactileInputs pass state through as-is).
        state = np.zeros(self._state_dim, dtype=np.float32)
        raw = np.asarray(msg.get("state", []), dtype=np.float32).reshape(-1)
        n = min(raw.size, self._state_dim)
        state[:n] = raw[:n]

        images: dict = {}
        for src, cam in CAM_MAP.items():
            if src not in msg:
                continue
            if cam not in self._cameras:
                if cam not in self._warned_cameras:
                    self._warned_cameras.add(cam)
                    logging.warning(
                        "ignoring client key %r: this checkpoint has no %s (trained with that "
                        "slot zeroed and masked off)",
                        src,
                        cam,
                    )
                continue
            images[cam] = _decode_image(msg[src])

        data: dict = {"images": images, "state": state}

        # Only forward a NON-empty client prompt; an empty one would shadow --default-prompt
        # (InjectDefaultPrompt only fills when "prompt" is absent).
        prompt = msg.get("prompt")
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode("utf-8")
        if prompt:
            data["prompt"] = prompt

        # Tactile predictor: pair the live frame (tac_t) with the per-episode baseline (tac_0).
        if self._tactile_enabled:
            tac_now = self._collect_current_tactile(msg)
            if tac_now:
                if self._baseline is None:
                    # First predict after reset: the current frame IS the episode baseline.
                    self._baseline = {v: img.copy() for v, img in tac_now.items()}
                    logging.info("captured tactile baseline for views: %s", list(tac_now))
                tactile: dict = {}
                for view, cur in tac_now.items():
                    base = self._baseline.get(view, cur)  # missing baseline for a late view -> no-contact
                    # (2, H, W, 3): index 0 = baseline (tac_0), index 1 = current (tac_t).
                    tactile[view] = np.stack([base, cur], axis=0)
                data["tactile"] = tactile
            else:
                logging.warning(
                    "tactile-prior model but request carried no tactile views; "
                    "running prior with no contact signal"
                )

        actions = np.asarray(self._policy.infer(data)["actions"], dtype=np.float32)
        actions = _pad_actions(actions, 32)
        # actions is [action_horizon, 32]; the client MUST execute the FULL horizon — a shorter
        # execution stride drops the chunk-tail gripper close (see module DEPLOYMENT note).
        return {
            "status": "ok",
            "actions": actions.tolist(),
            "infer_time_ms": (time.monotonic() - t0) * 1000.0,
        }

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        """Bind the ZMQ REP socket and serve reset/predict requests forever (one reply per request)."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(self._addr)
        logging.info(
            "N0-VTLA ZMQ server on %s (horizon=%d, tactile=%s, views=%s)",
            self._addr,
            self._horizon,
            self._tactile_enabled,
            self._tactile_views or "-",
        )
        while True:
            try:
                msg = msgpack.unpackb(sock.recv(), raw=False)
                cmd = msg.get("cmd", "predict")
                if cmd == "reset":
                    reply = self._reset()
                elif cmd == "predict":
                    reply = self._predict(msg)
                else:
                    reply = {"status": "error", "message": f"unknown cmd: {cmd}"}
            except Exception as e:  # noqa: BLE001
                logging.exception("request failed")
                reply = {"status": "error", "message": str(e)}
            sock.send(msgpack.packb(reply, use_bin_type=True))


def main() -> None:
    """CLI entry point: load the trained policy, auto-detect the tactile path from the config, serve over ZMQ."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="train config name (e.g. flexiv_tactile_reference)")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing model.safetensors + assets/")
    ap.add_argument("--addr", default="tcp://*:5556")
    ap.add_argument("--default-prompt", default=None)
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    train_config = _config.get_config(cli.config)
    policy = _policy_config.create_trained_policy(
        train_config, cli.ckpt, default_prompt=cli.default_prompt
    )

    model = train_config.model
    # Auto-detect the tactile path from the config: tactile configs set tactile_predictor_enabled=True
    # and carry the ordered view list in tactile_image_keys; control/base configs have neither.
    tactile_enabled = bool(getattr(model, "tactile_predictor_enabled", False))
    tactile_views = list(getattr(model, "tactile_image_keys", ()) or ()) if tactile_enabled else []
    if tactile_enabled and not tactile_views:
        logging.warning("tactile_predictor_enabled but tactile_image_keys is empty; tactile disabled")

    logging.info(
        "model loaded: config=%s ckpt=%s tactile=%s views=%s",
        cli.config,
        cli.ckpt,
        tactile_enabled,
        tactile_views or "-",
    )
    # The sim configs compute norm_stats at raw_action_dim; Flexiv real-robot configs
    # keep the canonical padded-32 layout and have no such field.
    state_dim = int(getattr(train_config.data, "raw_action_dim", 0) or 32)
    logging.info("proprio width: %d (from %s)", state_dim,
                 "data.raw_action_dim" if getattr(train_config.data, "raw_action_dim", 0) else "canonical-32 default")

    # Which camera slots this checkpoint actually saw. The simulation configs carry
    # raw_action_dim (8 = single arm, 16 = dual arm); a single-arm dataset has no wrist_r
    # column, so its right-wrist slot was zero + image_mask=False for the whole run. Real-robot
    # configs have no raw_action_dim and do carry three cameras, so they keep every slot.
    cameras = None
    if getattr(train_config.data, "raw_action_dim", 0) and state_dim < 16:
        cameras = [c for c in CAM_MAP.values() if c != "cam_right_wrist"]
        logging.info("single-arm config: accepting cameras %s (no right wrist)", cameras)
    server = VtlaZmqServer(
        policy,
        cli.addr,
        train_config.model.action_horizon,
        cli.default_prompt,
        tactile_views,
        state_dim,
        cameras,
    )
    server.run()


if __name__ == "__main__":
    main()
