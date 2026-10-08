"""rot6d <-> rotation matrix + world-frame relative-rotation delta.

Used by the `eef_delta_relrot` action mode: rotation dims are encoded as the
world-frame relative rotation  R_rel = R_{t+1} @ R_t^T  (instead of naive
element-wise rot6d subtraction), and restored as  R_{t+1} = R_rel @ R_t.

Roundtrip verified to machine precision (max matrix err ~4e-16).
"""
from __future__ import annotations

import numpy as np


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(...,6) -> (...,3,3). Gram-Schmidt (Zhou et al. 2019). Columns = basis."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,6). First two columns."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_delta_world(rot6d_t: np.ndarray, rot6d_t1: np.ndarray) -> np.ndarray:
    """Encode delta = matrix_to_rot6d( R_{t+1} @ R_t^T ).  (...,6) inputs/outputs."""
    Rt = rot6d_to_matrix(rot6d_t)
    Rt1 = rot6d_to_matrix(rot6d_t1)
    R_rel = np.einsum("...ij,...kj->...ik", Rt1, Rt)  # R_t1 @ R_t^T
    return matrix_to_rot6d(R_rel)


def rot6d_apply_delta_world(rot6d_t: np.ndarray, delta6d: np.ndarray) -> np.ndarray:
    """Restore: R_{t+1} = R_rel @ R_t ; return rot6d_{t+1}.  (...,6) inputs/outputs."""
    Rt = rot6d_to_matrix(rot6d_t)
    R_rel = rot6d_to_matrix(delta6d)
    Rt1 = np.einsum("...ij,...jk->...ik", R_rel, Rt)
    return matrix_to_rot6d(Rt1)
