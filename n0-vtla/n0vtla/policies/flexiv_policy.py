import dataclasses

import numpy as np

from n0vtla import transforms


def _parse_image(image) -> np.ndarray:
    """Return ``image`` as HWC uint8: scale floating [0,1] -> [0,255] and move a leading
    channel axis (CHW) to trailing (HWC). No-op for images already HWC uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _split_tactile_stack(
    tactile: dict, image: dict, image_mask: dict, baseline_suffix: str, future_suffix: str
) -> None:
    """Split each latent-tactile frame-stack into model-image keys (in place).

    Each ``tactile[key]`` is a delta_timestamps stack with a leading time axis:
      * length 2: ``[baseline(0), current(1)]``            (v0 / e2e).
      * length 3: ``[baseline(0), current(1), future(2)]`` (when a future frame is loaded).
    Emits ``<key>`` (current), ``<key><baseline_suffix>`` (frame 0), and — only when a 3rd
    frame is present — ``<key><future_suffix>`` (frame t+H). A stack with no time axis
    (delta_timestamps not applied) yields current-only. Shared by ``FlexivTactileInputs`` and
    the canonical tactile inputs so the two paths can never diverge.
    """
    for key, raw in tactile.items():
        arr = np.asarray(raw)
        if arr.ndim == 4 and arr.shape[0] >= 2:
            image[key] = _parse_image(arr[1])
            image_mask[key] = np.True_
            image[key + baseline_suffix] = _parse_image(arr[0])
            image_mask[key + baseline_suffix] = np.True_
            if arr.shape[0] >= 3:
                # future frame at t+H (index 2). Only emitted for a length-3 stack,
                # so 2-frame v0 datasets stay byte-identical (no future key).
                image[key + future_suffix] = _parse_image(arr[2])
                image_mask[key + future_suffix] = np.True_
        else:
            # No time axis (e.g. delta_timestamps not applied): current only, no baseline.
            image[key] = _parse_image(arr)
            image_mask[key] = np.True_


def _build_inputs(data: dict) -> dict:
    """Assemble the canonical model-input dict from one repacked LeRobot sample.

    Shared by every Flexiv input transform. Always emits the 3 base RGB slots
    (base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb) plus ``state``. The right-wrist
    camera is optional: when absent, its slot is a zero image with ``image_mask=False``.
    Optionally passes through, when present in ``data``:
      * ``actions``  - training target;
      * ``expert_images`` -> ``expert_image`` / ``expert_image_mask`` - the DIRECT-INJECTION
        tactile path (the ``use_tactile=True`` configs). NOT used by the reference control/predictor
        configs, and NOT how the action-predictor routes tactile (that is ``FlexivTactileInputs``,
        which adds tactile as extra ``image`` keys);
      * ``prompt``   - decoded to str if bytes.

    Returns the model-side input dict.
    """
    in_images = data["images"]
    unexpected = set(in_images) - set(FlexivEEFInputs.EXPECTED_CAMERAS)
    if unexpected:
        raise ValueError(f"Unexpected Flexiv image keys: {tuple(sorted(unexpected))}")
    base_image = _parse_image(in_images["cam_high"])
    left_wrist_image = _parse_image(in_images["cam_left_wrist"])
    if "cam_right_wrist" in in_images:
        right_wrist_image = _parse_image(in_images["cam_right_wrist"])
        right_wrist_mask = np.True_
    else:
        right_wrist_image = np.zeros_like(base_image)
        right_wrist_mask = np.False_

    inputs = {
        "state": np.asarray(data["state"]),
        "image": {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist_image,
            "right_wrist_0_rgb": right_wrist_image,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": right_wrist_mask,
        },
    }

    if "actions" in data:
        inputs["actions"] = np.asarray(data["actions"])

    if "expert_images" in data:
        expert_image = {key: _parse_image(image) for key, image in data["expert_images"].items()}
        inputs["expert_image"] = expert_image
        inputs["expert_image_mask"] = {key: np.True_ for key in expert_image}

    if "prompt" in data:
        prompt = data["prompt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        inputs["prompt"] = prompt

    return inputs


@dataclasses.dataclass(frozen=True)
class FlexivEEFInputs(transforms.DataTransformFn):
    """Base EEF input transform: 3 RGB cameras + state, NO tactile.

    This is the tactile-OFF path; its output must stay byte-identical to stock n0vtla for the
    control/parity runs (iron rule). ``FlexivTactileInputs`` is the tactile action-predictor variant
    that adds latent-tactile keys on top of this same base.
    """

    EXPECTED_CAMERAS: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        return _build_inputs(data)


@dataclasses.dataclass(frozen=True)
class FlexivEEFOutputs(transforms.DataTransformFn):
    """EEF output transform: keep the first ``action_dim`` (10) action dims - 9 EEF-pose dims
    + 1 gripper - and drop the zero-padding up to the 32-dim action space."""

    action_dim: int = 10

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class FlexivTactileInputs(transforms.DataTransformFn):
    """FlexivEEFInputs + latent-tactile routing for the tactile action-predictor (v0) path.

    Builds the standard 3 RGB slots via ``_build_inputs`` and additionally splits each
    latent-tactile stack in ``data["tactile"]`` into model-image keys:
      ``<view>`` (current), ``<view>.baseline`` (frame 0), and ``<view>.future`` (frame t+H,
      only when a future frame is loaded).

    Stack layout depends on ``extra_delta_timestamps``:
      * v0 / e2e:      ``[-BIG, 0]``      -> 2 frames ``[baseline, current]`` (no future).
      * with a future frame: ``[-BIG, 0, +H]``  -> ``[baseline, current, future]``.
    LeRobot clamps the -BIG offset to frame 0 and clamps +H past the episode end to the last
    frame. The number of emitted keys follows the stack length, so a v0 dataset stays
    byte-identical (the future branch only fires on a length-3 stack).

    Ports the tactile split of the reference ``CanonicalVTLAInputs`` (canonical_vtla_policy.py
    :114-144). The keys land in the model ``image`` dict;
    ``N0VTLAPolicy._preprocess_observation`` pops them back out before SigLIP.
    """

    BASELINE_SUFFIX: str = ".baseline"
    FUTURE_SUFFIX: str = ".future"

    def __call__(self, data: dict) -> dict:
        inputs = _build_inputs(data)
        tactile = data.get("tactile")
        if not tactile:
            return inputs
        _split_tactile_stack(tactile, inputs["image"], inputs["image_mask"],
                             self.BASELINE_SUFFIX, self.FUTURE_SUFFIX)
        return inputs




@dataclasses.dataclass(frozen=True)
class FlexivJointOutputs(transforms.DataTransformFn):
    """Joint-space output transform: keep the first ``action_dim`` (8) joint action dims."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
