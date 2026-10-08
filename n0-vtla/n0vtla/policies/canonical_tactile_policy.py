"""CanonicalTactileInputs: map a canonical LeRobot sample -> N0VTLAPolicy model input.

Adapted from ``CanonicalVTLAInputs`` for canonical tactile-predictor pretraining. The differences
below preserve the expected tactile input semantics without affecting the tactile-disabled path:

  1. SHORT tactile keys (``cs.short_name``): the model's ``tactile_image_keys`` are SHORT, and
     ``N0VTLAPolicy._tactile_keys`` returns only configured keys present in the image dict. Emitting
     FULL canonical names would make ``_tactile_keys`` return [] → g=None (no tactile in z) AND the
     unpopped full-name tactile would leak into SigLIP as bogus RGB. So we emit SHORT keys.
  2. No center crop or de-letterboxing. Tactile frames retain their source geometry. We do not
     strip constant borders here,
     because (a) the tactile signal is a *diff* and a constant border cancels
     (both frames share it) → a retained border does NOT poison the predictor, and (b) train/serve stay
     consistent only if both skip de-bordering. So tactile frames pass through ``_parse_image``
     untouched and ``model_transforms``' ``resize_with_pad`` does the final 224 sizing. (An earlier
     draft center-cropped to a square — WRONG for 480x640: it cropped left/right CONTENT and kept
     the top/bottom border. Removed.)
  3. uniform placeholder emission: EVERY one of the 4 canonical tactile views gets
     ``<short>`` / ``<short>.baseline`` / ``<short>.future`` (placeholder + mask False when a
     platform lacks the sensor), so mixed-embodiment batches collate (jax.tree.map needs identical
     keys). NB: a placeholder tactile carries ``image_mask=False``; the model side must treat
     tactile validity by MASK, not key-presence (see N0VTLAPolicy _build_g mask handling) so a
     no-tactile sample is not mistaken for a real one.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping

import numpy as np

from n0vtla import transforms
from n0vtla.policies import canonical_schema as cs
from n0vtla.policies.flexiv_policy import _parse_image

_BASE_RGB = "base_0_rgb"
_LEFT_RGB = "left_wrist_0_rgb"
_RIGHT_RGB = "right_wrist_0_rgb"


@dataclasses.dataclass(frozen=True)
class Stage1ObservationOnly(transforms.DataTransformFn):
    """Supply interface placeholders for human data, without reading robot targets.

    The inherited Pi0.5 tokenizer still sees a constant zero state. This is an explicit
    human-data adaptation, not measured robot proprioception or action supervision.
    """

    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data.pop("actions", None)
        data.pop("action_mask", None)
        data["observation.state"] = np.zeros(self.action_dim, dtype=np.float32)
        data["action"] = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class CanonicalTactileInputs(transforms.DataTransformFn):
    """Canonical → model input for the tactile action-predictor path.

    ``latent_tactile``: each tactile column arrives as a delta_timestamps
    stack ``(T, H, W, C)`` = ``[baseline(0), current(1), future(2)]`` (future absent → length 2).
    ``flip_wrist_180`` rotates the (real) wrist RGB 180° for UMI entries to match the arm-cam
    convention; tactile is orientation-invariant (a contact diff) so it is NOT flipped.
    Tactile frames are passed through untouched (no de-letterbox) — see module docstring diff #2.
    """

    latent_tactile: bool = True
    flip_wrist_180: bool = False
    # Per-sample variant of flip_wrist_180 for the mixed canonical corpus: flip wrist RGB only
    # for UMI entries (task_key ends with "_UMI"), leaving arm-platform samples untouched.
    # 1.0 applied this per-group (flip_wrist=group_id.startswith("umi")); the all-or-nothing
    # flip_wrist_180 cannot express that in a ConcatDataset, which is why 2.0 shipped with the
    # flip silently off — UMI wrist views trained upside-down relative to the arm convention.
    flip_wrist_umi: bool = False
    BASELINE_SUFFIX: str = ".baseline"
    FUTURE_SUFFIX: str = ".future"

    def __call__(self, data: dict) -> dict:
        # placeholder shape source = first present canonical image. Native shape is fine: every
        # image (real or placeholder) is resize_with_pad'd to 224 by model_transforms before collate,
        # so a placeholder's shape need not match the real tactile's — only the mask matters.
        first = None
        for k in cs.IMAGE_KEYS:
            if k in data:
                first = _parse_image(data[k])
                break
        if first is None:
            raise ValueError("CanonicalTactileInputs: no canonical image key present in sample")
        rgb_placeholder = np.zeros_like(first)
        tac_placeholder = rgb_placeholder  # no center-crop (diff #2)

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}

        # --- 3 RGB slots (fill from canonical views; missing -> placeholder + mask False) ---
        def _put_rgb(out_key: str, src_keys: tuple[str, ...]) -> None:
            for sk in src_keys:
                if sk in data:
                    images[out_key] = _parse_image(data[sk])
                    image_masks[out_key] = np.True_
                    return
            images[out_key] = rgb_placeholder
            image_masks[out_key] = np.False_

        _put_rgb(_BASE_RGB, ("observation.image.third_view",))
        _put_rgb(_LEFT_RGB, ("observation.image.left_wrist_view",))
        _put_rgb(_RIGHT_RGB, ("observation.image.right_wrist_view", "observation.image.second_third_view"))

        flip = self.flip_wrist_180
        if not flip and self.flip_wrist_umi:
            flip = str(data.get("task_key", "")).lower().endswith("_umi")
        if flip:
            for k in (_LEFT_RGB, _RIGHT_RGB):
                if bool(image_masks.get(k, np.False_)):
                    images[k] = np.ascontiguousarray(images[k][::-1, ::-1])

        # --- 4 tactile views: SHORT keys, passed through (no de-letterbox), uniform placeholder ---
        for full_key in sorted(cs.TACTILE_KEYS):
            short = cs.short_name(full_key)
            bk, fk = short + self.BASELINE_SUFFIX, short + self.FUTURE_SUFFIX
            arr = np.asarray(data[full_key]) if full_key in data else None
            if arr is not None and arr.ndim == 4 and arr.shape[0] >= 2:
                images[short] = _parse_image(arr[1])
                image_masks[short] = np.True_
                images[bk] = _parse_image(arr[0])
                image_masks[bk] = np.True_
                if arr.shape[0] >= 3:
                    # Episode-tail clamp: the loader clamps t+50 past the episode end to the LAST
                    # frame and flags it in <key>_is_pad[2]. A clamped frame is NOT a valid z*
                    # target (at the very tail z* = encoder(0), a batch-constant that poisons the
                    # InfoNCE pool). Keep the pixels (batch shape uniformity) but mask False so
                    # the row is dropped from the tactile signal. NB: is_pad[0] (baseline)
                    # is True BY DESIGN (the -100000-frame offset clamps to frame 0 — that IS the
                    # baseline) and must NOT be consulted.
                    pad = data.get(full_key + "_is_pad")
                    fut_clamped = bool(np.asarray(pad).reshape(-1)[2]) if pad is not None else False
                    images[fk] = _parse_image(arr[2])
                    image_masks[fk] = np.bool_(not fut_clamped)
                else:  # no-future mode: placeholder keeps batch keys uniform
                    images[fk] = tac_placeholder
                    image_masks[fk] = np.False_
            elif arr is not None and arr.ndim == 3:  # single frame, no delta_timestamps
                images[short] = _parse_image(arr)
                image_masks[short] = np.True_
                images[bk] = tac_placeholder
                image_masks[bk] = np.False_
                images[fk] = tac_placeholder
                image_masks[fk] = np.False_
            else:  # platform lacks this sensor -> placeholder all 3 (mask False)
                images[short] = tac_placeholder
                image_masks[short] = np.False_
                images[bk] = tac_placeholder
                image_masks[bk] = np.False_
                images[fk] = tac_placeholder
                image_masks[fk] = np.False_

        if "observation.state" not in data:
            raise ValueError(
                "CanonicalTactileInputs requires 'observation.state'; got keys " f"{sorted(data)}"
            )
        out = {
            "state": np.asarray(data["observation.state"], dtype=np.float32),
            "image": images,
            "image_mask": image_masks,
        }
        # Actions are the training target and are absent at inference time, where the
        # server is handed an observation only. Mirrors FlexivEEFInputs, which has always
        # treated them as optional.
        action = data.get("actions", data.get("action"))
        if action is not None:
            out["actions"] = np.asarray(action, dtype=np.float32)
        if "action_mask" in data:
            out["action_mask"] = np.asarray(data["action_mask"], dtype=bool)
        if "norm_group_id" in data:
            # Per-repo routing key injected by the data loader (repo_group_ids); consumed and
            # popped by GroupedNormalize after ChunkDeltaToCurrentState.
            out["norm_group_id"] = data["norm_group_id"]
        if "observation.state_mask" in data:
            out["state_mask"] = np.asarray(data["observation.state_mask"], dtype=bool)
        if "prompt" in data:
            p = data["prompt"]
            out["prompt"] = p.decode("utf-8") if isinstance(p, bytes) else p
        return out


@dataclasses.dataclass(frozen=True)
class OptionalRepack(transforms.RepackTransform):
    """RepackTransform that DROPS keys absent from the sample.

    Mixed-embodiment canonical repos vary in which views exist (single-arm repos lack
    second_third_view / right_wrist_* tactile), and the loader already skips missing videos
    (_extract_referenced_video_keys filters referenced keys by availability). So the repack must
    tolerate the gap instead of KeyError'ing on ``flat_item[missing]``. Subclasses RepackTransform
    so ``_extract_referenced_video_keys`` (isinstance check) still discovers the referenced video
    keys. Assumes a FLAT ``{new: old}`` structure — canonical uses exactly that.
    """

    def __call__(self, data: dict) -> dict:
        flat_item = transforms.flatten_dict(data)
        return {new: flat_item[old] for new, old in self.structure.items() if old in flat_item}
