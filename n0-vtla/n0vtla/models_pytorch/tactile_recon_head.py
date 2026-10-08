"""TactileReconHead: r_psi, the auxiliary L1-reconstruction head from the paper's Stage 1
predictor-grounding objective (arXiv:2607.23782 Sec 4.2, Eq. 5):

    L_1 = L_NCE + lambda_rec * L_rec,   L_rec = || r_psi(z) - Dbar_{t->t+H} ||_1

This head implements the paper's Sec 4.2 recipe so the tactile predictor can be pretrained on
data with no ground-truth robot actions (e.g. in-the-wild hand/glove tactile); see
docs/MID_TRAIN.md.

The paper does not publish r_psi's architecture or Dbar's resolution, so both are design
choices made here:
  * Dbar lives in PIXEL space (not DINOv2 feature space): the per-view current->future tactile
    difference image, averaged over the active (real, non-placeholder) views, downsampled to a
    small (grid x grid) grid via adaptive average pooling. See N0VTLAPolicy._build_future_target.
  * r_psi is a 2-layer MLP over the mean-pooled z, matching the paper's description of the head
    as "a lightweight reconstruction head".
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TactileReconHead(nn.Module):
    """Decodes latent tactile tokens z -> a coarse (grid x grid) contact-change field.

    Args:
        hidden_dim: width of z's tokens (llm_dim, matching TactileActionPredictor's output).
        grid: side length of the coarse output field. Dbar is downsampled to the same size
            (see N0VTLAPolicy._build_future_target) so the L1 term compares equal shapes.
        mlp_ratio: hidden width multiplier for the 2-layer MLP.
    """

    def __init__(self, hidden_dim: int, grid: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.grid = grid
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, grid * grid),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, n_latent, hidden_dim) -> (B, grid, grid)."""
        pooled = z.mean(dim=1)  # (B, hidden_dim) -- same pooling convention as the InfoNCE h(.)
        field = self.mlp(pooled)  # (B, grid*grid)
        return field.view(z.shape[0], self.grid, self.grid)
