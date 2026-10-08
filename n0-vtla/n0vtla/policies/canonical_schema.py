"""Canonical observation/action schema shared by all VTLA platforms (umi + non_umi).

Pure constants for the canonical umi + non_umi data path — no behaviour. Key names are
preserved as-is from the source datasets so the transforms are mostly identity; only
``image_masks`` / ``action_mask`` differ per platform.

Additive, base-config-unreachable (nothing here is imported by a tactile-OFF / flexiv
config), so the tactile-OFF byte-identity gate is unaffected.
"""
from __future__ import annotations


# Ordered tuple; index 0..7 stable across all platforms. Missing views get
# image_masks[key]=False but the key still exists in the dict so downstream
# model / sequence positions stay consistent across a mixed-embodiment batch.
IMAGE_KEYS: tuple[str, ...] = (
    "observation.image.third_view",
    "observation.image.second_third_view",
    "observation.image.left_wrist_view",
    "observation.image.right_wrist_view",
    "observation.image.left_wrist_left_tactile",
    "observation.image.left_wrist_right_tactile",
    "observation.image.right_wrist_left_tactile",
    "observation.image.right_wrist_right_tactile",
)

# Which of IMAGE_KEYS are tactile (fed through the frozen DINOv2 tactile encoder, NOT SigLIP).
TACTILE_KEYS: frozenset[str] = frozenset(k for k in IMAGE_KEYS if k.endswith("_tactile"))

# Optical-flow variant (observation.image.flow_<view>_tactile). NOT used by the raw-gel predictor
# path; kept only for parity with the fork's flow_tactile option.
FLOW_TACTILE_KEYS: frozenset[str] = frozenset(
    k.replace("observation.image.", "observation.image.flow_") for k in TACTILE_KEYS
)

STATE_KEYS: tuple[str, ...] = (
    "observation.state.joint_position",
    "observation.state.joint_velocity",
    "observation.state.eef_pose",
    "observation.state.gripper",
)

# 32-dim canonical action layout — CONTIGUOUS EEF (matches the on-disk data whose action_mask
# is True on [0:10] single-arm / [0:20] dual-arm).
#   Left arm  [0:10]:  xyz [0:3] + rot6d [3:9] + gripper [9]
#   Right arm [10:20]: xyz [10:13] + rot6d [13:19] + gripper [19]
# dims [20:32] are reserved/unused (masked False in the current all-EEF data).
ACTION_DIM: int = 32

ACTION_SLOTS: dict[str, tuple[int, int]] = {
    "left_eef_xyz": (0, 3),
    "left_eef_rot6d": (3, 9),
    "left_gripper": (9, 10),
    "right_eef_xyz": (10, 13),
    "right_eef_rot6d": (13, 19),
    "right_gripper": (19, 20),
}


def short_name(image_key: str) -> str:
    """Drop the ``observation.image.`` prefix → the SHORT model-side key.

    The tactile naming contract (migration bug #1): the model's ``tactile_image_keys`` are
    SHORT (e.g. ``left_wrist_left_tactile``) and ``N0VTLAPolicy._tactile_keys`` returns only
    configured keys present in the image dict. If the input transform emitted FULL canonical
    names, ``_tactile_keys`` would return [] → tactile never reaches the predictor AND the unpopped
    full-name tactile leaks into SigLIP as bogus RGB. So ``CanonicalTactileInputs`` emits SHORT
    keys via this helper, and the tactile configs set ``tactile_image_keys`` to these short names.
    """
    return image_key.replace("observation.image.", "")


# The 4 SHORT tactile view names — set the model's tactile_image_keys to exactly this.
TACTILE_SHORT_KEYS: tuple[str, ...] = tuple(short_name(k) for k in IMAGE_KEYS if k.endswith("_tactile"))
