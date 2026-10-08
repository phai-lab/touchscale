from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from n0vtla_client import image_tools
import scipy.ndimage

from n0vtla.models import tokenizer as _tokenizer
from n0vtla.policies import canonical_schema as _canonical_schema
from n0vtla.shared import array_typing as at
from n0vtla.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


def _resize_single_channel(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    zoom_factors = (height / image.shape[0], width / image.shape[1])
    return scipy.ndimage.zoom(image.astype(np.float32, copy=False), zoom_factors, order=1)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame rank 3, got shape {frame.shape}")
    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame with 3 channels, got shape {frame.shape}")
    if frame.max(initial=0.0) > 1.0:
        frame = frame / 255.0
    gray = frame[..., 0] * 0.2989 + frame[..., 1] * 0.5870 + frame[..., 2] * 0.1140
    return gray.astype(np.float32, copy=False)


def _horn_schunck_flow(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    alpha: float,
    num_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel_x = np.array([[-0.25, 0.25], [-0.25, 0.25]], dtype=np.float32)
    kernel_y = np.array([[-0.25, -0.25], [0.25, 0.25]], dtype=np.float32)
    kernel_t = np.full((2, 2), 0.25, dtype=np.float32)
    avg_kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6, 0.0, 1 / 6],
            [1 / 12, 1 / 6, 1 / 12],
        ],
        dtype=np.float32,
    )

    prev = previous_frame.astype(np.float32, copy=False)
    curr = current_frame.astype(np.float32, copy=False)

    grad_x = scipy.ndimage.convolve(prev, kernel_x, mode="nearest") + scipy.ndimage.convolve(
        curr, kernel_x, mode="nearest"
    )
    grad_y = scipy.ndimage.convolve(prev, kernel_y, mode="nearest") + scipy.ndimage.convolve(
        curr, kernel_y, mode="nearest"
    )
    grad_t = scipy.ndimage.convolve(curr, kernel_t, mode="nearest") - scipy.ndimage.convolve(
        prev, kernel_t, mode="nearest"
    )

    flow_x = np.zeros_like(prev, dtype=np.float32)
    flow_y = np.zeros_like(prev, dtype=np.float32)
    denom = alpha * alpha + grad_x * grad_x + grad_y * grad_y + 1e-6

    for _ in range(num_iterations):
        avg_x = scipy.ndimage.convolve(flow_x, avg_kernel, mode="nearest")
        avg_y = scipy.ndimage.convolve(flow_y, avg_kernel, mode="nearest")
        update = (grad_x * avg_x + grad_y * avg_y + grad_t) / denom
        flow_x = avg_x - grad_x * update
        flow_y = avg_y - grad_y * update

    return flow_x, flow_y


def _hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    hue = np.mod(hue, 1.0)
    idx = np.floor(hue * 6.0).astype(np.int32)
    frac = hue * 6.0 - idx
    p = value * (1.0 - saturation)
    q = value * (1.0 - frac * saturation)
    t = value * (1.0 - (1.0 - frac) * saturation)
    idx = idx % 6

    red = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5], [value, q, p, p, t, value])
    green = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5], [t, value, value, q, p, p])
    blue = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5], [p, p, t, value, value, q])
    return np.stack((red, green, blue), axis=-1).astype(np.float32, copy=False)


def _flow_to_rgb(flow_x: np.ndarray, flow_y: np.ndarray, magnitude_percentile: float) -> np.ndarray:
    magnitude = np.sqrt(flow_x * flow_x + flow_y * flow_y)
    scale = float(np.percentile(magnitude, magnitude_percentile))
    if scale <= 1e-6:
        scale = float(np.max(magnitude, initial=0.0))
    if scale <= 1e-6:
        scale = 1.0

    hue = (np.arctan2(flow_y, flow_x) + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue, dtype=np.float32)
    value = np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32, copy=False)
    rgb = _hsv_to_rgb(hue.astype(np.float32, copy=False), saturation, value)
    return np.moveaxis(rgb, -1, 0)


def compute_tactile_optical_flow_rgb(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    flow_size: tuple[int, int] = (96, 128),
    alpha: float = 0.25,
    num_iterations: int = 20,
    magnitude_percentile: float = 95.0,
) -> np.ndarray:
    """Compute pseudo-RGB optical flow from two tactile frames.

    Returns a `C,H,W` float32 image in `[0, 1]`.
    """
    flow_height, flow_width = flow_size
    previous_gray = _resize_single_channel(_to_grayscale(previous_frame), flow_height, flow_width)
    current_gray = _resize_single_channel(_to_grayscale(current_frame), flow_height, flow_width)
    flow_x, flow_y = _horn_schunck_flow(
        previous_gray,
        current_gray,
        alpha=alpha,
        num_iterations=num_iterations,
    )
    return _flow_to_rgb(flow_x, flow_y, magnitude_percentile)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class OptionalRepackTransform(RepackTransform):
    """Repacks data while omitting explicitly optional source fields when absent.

    Required fields retain ``RepackTransform`` semantics and raise ``KeyError`` when missing.
    Optional leaves are removed from their containing mapping, allowing schemas with optional
    sensors to share one data configuration.
    """

    optional_source_keys: frozenset[str] = frozenset()

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        missing = object()

        def repack(node):
            if isinstance(node, Mapping):
                result = {}
                for key, value in node.items():
                    repacked = repack(value)
                    if repacked is not missing:
                        result[key] = repacked
                return result
            if isinstance(node, str):
                if node in flat_item:
                    return flat_item[node]
                if node in self.optional_source_keys:
                    return missing
                raise KeyError(node)
            raise TypeError(f"OptionalRepackTransform only supports mappings with string leaves, got {type(node)}")

        return repack(self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        if "expert_image" in data:
            data["expert_image"] = {
                k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["expert_image"].items()
            }
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation representation to a 3x3 rotation matrix.

    d6: (..., 6) -> (..., 3, 3). Columns of the returned matrix are b1, b2, b3.
    Matches the converter convention (rot6d = first two columns of the matrix).
    """
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.where(n1 < 1e-8, 1.0, n1)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(a2p, axis=-1, keepdims=True)
    b2 = a2p / np.where(n2 < 1e-8, 1.0, n2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)  # columns are b1, b2, b3
    # Degenerate rot6d (zero or collinear columns — e.g. missing/placeholder frames such
    # as a corrupt UR episode whose 2nd column is all-zero) would divide by zero and yield
    # NaN, poisoning norm stats and the training loss. Fall back to identity on those rows
    # so the pipeline stays finite. Well-formed rot6d is unaffected (n1, n2 are ~1).
    degen = (n1[..., 0] < 1e-6) | (n2[..., 0] < 1e-6)
    if np.any(degen):
        R[degen] = np.eye(3, dtype=R.dtype)
    return R


def _matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to its 6D representation (first two columns).

    R: (..., 3, 3) -> (..., 6).
    """
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


@dataclasses.dataclass(frozen=True)
class ChunkDeltaToCurrentState(DataTransformFn):
    """Rewrite an ABSOLUTE-pose action chunk into Pi0-style deltas relative to the
    current proprioceptive state ``data["state"]``.

    On-disk canonical chunks store ABSOLUTE eef targets (action == next absolute pose;
    schema ``*_eef_10`` / ``*_eef_20``, i.e. action ≈ state in range). For each ACTIVE
    arm this makes the eef dims relative to the current state s0 = data["state"]
    (broadcast over the action horizon):
      xyz:   d_xyz[t] = action_xyz[t] - s0_xyz                  (element-wise, like DeltaActions)
      rot6d: d_R[t]   = R_action[t] @ R_s0^T                    (world-frame relative rotation)
      grip:  unchanged (already absolute)

    This is the EXACT inverse of ``RelRotAbsoluteActions`` (the serve-time transform:
    abs_xyz = s0 + d_xyz, abs_R = d_R @ R_s0), so the policy's predicted deltas restore
    to absolute poses at deploy. ``_rot6d_to_matrix`` is degenerate-safe (identity on
    zero/collinear rows), keeping NaNs out of the norm stats and loss.

    The canonical converter writes absolute poses. This transform therefore computes
    relative rotations directly from the current state and each absolute target; it does
    not apply a cumulative sum over the action chunk.

    An arm is only transformed if it is ACTIVE (its eef_xyz dims are all True in
    ``action_mask``); inactive/masked arms keep placeholder values. Requires
    ``data["state"]`` (the (D,) current proprioceptive state). A single-step chunk
    (ndim == 1) and the absence of ``"actions"`` / ``"state"`` are no-ops.
    """

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or "state" not in data:
            return data

        actions = np.asarray(data["actions"], dtype=np.float32)
        # Single-step case (shape (ACTION_DIM,)): nothing to make relative across a chunk.
        if actions.ndim == 1:
            return data

        actions = actions.copy()
        state = np.asarray(data["state"], dtype=np.float32)  # (D,) current proprioception
        slots = _canonical_schema.ACTION_SLOTS

        action_mask = data.get("action_mask")
        if action_mask is not None:
            action_mask = np.asarray(action_mask)

        for xyz_name, rot_name in (
            ("left_eef_xyz", "left_eef_rot6d"),
            ("right_eef_xyz", "right_eef_rot6d"),
        ):
            xyz_lo, xyz_hi = slots[xyz_name]
            rot_lo, rot_hi = slots[rot_name]

            # An arm is active iff its eef_xyz dims are all True in the mask.
            if action_mask is not None and not bool(np.all(action_mask[xyz_lo:xyz_hi])):
                continue

            # xyz: relative to the current state (broadcast s0 over the horizon).
            actions[:, xyz_lo:xyz_hi] -= state[None, xyz_lo:xyz_hi]

            # rot6d: world-frame relative rotation R_action @ R_s0^T (== rot6d_delta_world,
            # the inverse of RelRotAbsoluteActions' rot6d_apply_delta_world).
            ref = np.broadcast_to(state[None, rot_lo:rot_hi], actions[:, rot_lo:rot_hi].shape)
            R_state = _rot6d_to_matrix(ref)                          # (H, 3, 3) degenerate-safe
            R_action = _rot6d_to_matrix(actions[:, rot_lo:rot_hi])   # (H, 3, 3)
            R_rel = np.einsum("...ij,...kj->...ik", R_action, R_state)  # R_action @ R_state^T
            actions[:, rot_lo:rot_hi] = _matrix_to_rot6d(R_rel)

        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class RelRotAbsoluteActions(DataTransformFn):
    """Rewrite a chunk of RELATIVE eef targets (as produced by ChunkDeltaToCurrentState)
    back into ABSOLUTE poses, using the current proprioceptive state ``data["state"]``.

    This is the EXACT inverse of ChunkDeltaToCurrentState:
      xyz:   action_xyz[t] = d_xyz[t] + s0_xyz                  (element-wise, like AbsoluteActions)
      rot6d: R_action[t]   = d_R[t] @ R_s0                      (undoes d_R = R_action @ R_s0^T)
      grip:  unchanged (already absolute)

    Naively element-wise adding a predicted rot6d delta to the current rot6d state (what
    plain ``AbsoluteActions`` does) does not recover a valid rotation: rot6d is not a vector
    space, so its components cannot be composed by addition. This transform instead composes
    the actual rotation matrices, mirroring ChunkDeltaToCurrentState's encode-side math.

    An arm is only transformed if it is ACTIVE (its eef_xyz dims are all True in
    ``action_mask``); inactive/masked arms keep placeholder values. Requires
    ``data["state"]`` (the (D,) current proprioceptive state). A single-step chunk
    (ndim == 1) and the absence of ``"actions"`` / ``"state"`` are no-ops.
    """

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or "state" not in data:
            return data

        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 1:
            return data

        actions = actions.copy()
        state = np.asarray(data["state"], dtype=np.float32)
        slots = _canonical_schema.ACTION_SLOTS

        action_mask = data.get("action_mask")
        if action_mask is not None:
            action_mask = np.asarray(action_mask)

        for xyz_name, rot_name in (
            ("left_eef_xyz", "left_eef_rot6d"),
            ("right_eef_xyz", "right_eef_rot6d"),
        ):
            xyz_lo, xyz_hi = slots[xyz_name]
            rot_lo, rot_hi = slots[rot_name]

            if action_mask is not None and not bool(np.all(action_mask[xyz_lo:xyz_hi])):
                continue

            # xyz: undo the subtraction, relative to the current state.
            actions[:, xyz_lo:xyz_hi] += state[None, xyz_lo:xyz_hi]

            # rot6d: R_action = R_rel @ R_state (undoes R_rel = R_action @ R_state^T).
            ref = np.broadcast_to(state[None, rot_lo:rot_hi], actions[:, rot_lo:rot_hi].shape)
            R_state = _rot6d_to_matrix(ref)                          # (H, 3, 3) degenerate-safe
            R_rel = _rot6d_to_matrix(actions[:, rot_lo:rot_hi])      # (H, 3, 3)
            R_action = np.einsum("...ij,...jk->...ik", R_rel, R_state)  # R_rel @ R_state
            actions[:, rot_lo:rot_hi] = _matrix_to_rot6d(R_action)

        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
