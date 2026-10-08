"""Frozen DINOv2 tactile encoder + trainable projection.

This encoder does NOT compute the tactile difference: it encodes whatever image it is
given. In the VTLA pipeline the caller (n0vtla_policy.py::_build_g) forms the
tac_t - tac_0 difference and passes it in, so g = DINOv2(tac_t - tac_0) per view.

Output: 10 tokens per image (1 CLS + 3x3 avg-pooled spatial), each of
`llm_dim` dimensionality (matching the Gemma VLM hidden size).

The DINOv2 backbone is frozen for three reasons (see spec §2.8.3):
  1. Preserves general-purpose self-supervised features.
  2. Enables external-user migration: their adapter only trains `tactile_proj`.
  3. Saves ~350MB/GPU of activation+optimizer memory.

External users can subclass this or provide their own class satisfying
the `TactileEncoder` Protocol.

Phase 2 upgrade (NOT this plan): swap to `facebook/dinov3-vitb16-pretrain-lvd1689m`
once n0vtla's `transformers==4.53.2` pin is lifted to >=4.56.0. The Protocol +
3x3 pool layout work for both backbones (DINOv3 ViT-B/16 produces 14x14=196 spatial
tokens at 224 input vs DINOv2's 16x16=256; `adaptive_avg_pool2d(grid, 3)` handles both).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@runtime_checkable
class TactileEncoder(Protocol):
    """Public interface. out_dim must match the VLM hidden dimension."""
    out_dim: int

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Accept either uint8 ``(B, 3, H, W)`` or pre-normalized float tensor.

        Concrete implementations handle both dtypes internally:
        - ``torch.uint8``: apply full preprocessing (e.g. ImageNet normalize)
        - ``torch.float32`` / ``torch.bfloat16``: passed through untouched. The default
          impl applies NO normalization to float input. In the VTLA pipeline the float
          input is a pre-computed tac_t - tac_0 diff, which is intentionally NOT
          ImageNet-normalized (see FrozenDINOv2TactileEncoder._dinov2_preprocess).

        Returns: (B, N, out_dim) where N == 10 for the default impl.
        """
        ...


class FrozenDINOv2TactileEncoder(nn.Module):
    """Default tactile encoder: frozen DINOv2-base + CLS+3x3 pool + linear proj.

    Backbone: facebook/dinov2-base (ViT-B/14, 768-dim, ~86M params, frozen).
    Projection: trainable nn.Linear(768, llm_dim).
    Tokens out: 10 per image = 1 CLS + 3x3 avg-pooled spatial grid.

    Phase 2 upgrade note: to switch to DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`),
    update BACKBONE_NAME and rename the class. Blocked until n0vtla's transformers pin
    is lifted from ==4.53.2 to >=4.56.0. Architecture and token layout are identical.
    """

    BACKBONE_NAME = "facebook/dinov2-base"
    BACKBONE_DIM = 768
    N_TOKENS = 10  # == 1 + pool_grid**2 at the default pool_grid=3; use the n_tokens property for the runtime count

    def __init__(self, llm_dim: int, pool_grid: int = 3):
        super().__init__()
        self.pool_grid = pool_grid
        self.backbone = AutoModel.from_pretrained(self.BACKBONE_NAME)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.tactile_proj = nn.Linear(self.BACKBONE_DIM, llm_dim)
        self.out_dim = llm_dim
        # ImageNet normalization constants — registered as buffers so they
        # move to GPU automatically with .to(device) / .cuda().
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @property
    def n_tokens(self) -> int:
        return 1 + self.pool_grid * self.pool_grid

    def train(self, mode: bool = True) -> "FrozenDINOv2TactileEncoder":
        """Keep backbone in eval mode even when module is set to train.
        Prevents DINOv2 dropout layers from activating during training.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def _pool_spatial(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Reshape (B, S, D) patch tokens back to a (B, D, side, side) grid and
        adaptive-avg-pool to pool_grid x pool_grid, returning (B, pool_grid**2, D).

        Assumes S is a perfect square (DINOv2 emits a square patch grid, e.g. 16x16=256).
        """
        B, S, D = spatial_tokens.shape
        side = int(S ** 0.5)
        assert side * side == S, f"non-square spatial: {S}"
        grid = spatial_tokens.transpose(1, 2).reshape(B, D, side, side)
        pooled = F.adaptive_avg_pool2d(grid, self.pool_grid)
        return pooled.flatten(2).transpose(1, 2)

    def _dinov2_preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to uint8 input; pass float input through unchanged.

        uint8 (B, 3, H, W) -> float32 / 255, then (x - imagenet_mean) / imagenet_std.
        float32 / bfloat16 -> returned AS-IS (this method applies NO normalization).

        NOTE: the production path (n0vtla_policy.py::_build_g) feeds a FLOAT
        tac_t - tac_0 DIFFERENCE image, so it takes the pass-through branch and the
        frozen DINOv2 sees the raw diff WITHOUT ImageNet normalization. This is
        intentional for v0 (the diff, not a natural image, is the conditioning signal)
        and the resulting model is robot-validated. The uint8 branch exists for
        standalone / test callers that pass a single natural image.
        """
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
            imgs = (imgs - self.imagenet_mean) / self.imagenet_std
        return imgs

    def encode_backbone(self, imgs: torch.Tensor) -> torch.Tensor:
        """Frozen backbone -> n_tokens x 768 features (CLS + pool_grid^2 pooled), BEFORE proj.

        Split from ``project`` so the frozen-backbone features can be precomputed and
        cached offline (the OPTIONAL latent-cache training speedup, OFF by default). The
        online path never calls this directly: it calls ``forward``, which chains
        encode_backbone -> project. The backbone is frozen so its output is constant
        across training, while ``tactile_proj`` keeps training end-to-end and is applied
        by ``project`` at train time.

        Returns (B, n_tokens, BACKBONE_DIM=768); n_tokens == 1 + pool_grid^2 (10 default).
        """
        imgs = self._dinov2_preprocess(imgs)
        with torch.no_grad():
            hs = self.backbone(pixel_values=imgs).last_hidden_state
        cls = hs[:, :1]
        pooled = self._pool_spatial(hs[:, 1:])
        return torch.cat([cls, pooled], dim=1)

    def project(self, backbone_tokens: torch.Tensor) -> torch.Tensor:
        """Apply the trainable projection to backbone tokens (online or cached).

        (B, N_TOKENS, 768) → (B, N_TOKENS, llm_dim). Lets the latent path consume
        cached 768-dim features without re-running DINOv2.
        """
        return self.tactile_proj(backbone_tokens)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.tactile_proj(self.encode_backbone(imgs))

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.forward(imgs)


class TactileCrossAttnLayer(nn.Module):
    """Pre-norm cross-attention + FFN block: queries q attend over key/value kv.

    Per call: q = q + MHA(LN(q), LN(kv), LN(kv)); q = q + FFN(LN(q)). q and kv are
    layer-normed separately because they come from different sources (action/latent
    queries vs. tactile/context tokens).

    Args:
        hidden_dim: token width (shared by q and kv).
        n_heads:    number of attention heads.

    kv_mask (in forward): optional (B, N_kv) bool, True = valid kv token. It is inverted
    to MultiheadAttention's key_padding_mask (True = ignore). A row whose kv are ALL
    masked makes softmax over all -inf produce NaN; such rows are sanitized to 0.
    """
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # MultiheadAttention: key_padding_mask True means "ignore"
        kp_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
            key_padding_mask=kp_mask,
            need_weights=False,
        )
        if kv_mask is not None:
            # A sample with no valid kv has an all-True key_padding_mask, so softmax over
            # all -inf yields NaN. Zero those NaNs and hard-zero the attn output for such
            # rows (has_kv=False) so a fully-masked sample contributes nothing.
            has_kv = kv_mask.any(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            attn_out = torch.nan_to_num(attn_out) * has_kv.to(attn_out.dtype)
        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x

    def forward_query_cond(
        self,
        residual_q: torch.Tensor,
        attn_q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Arch-C variant of ``forward`` (tactile-KV predictor): the conditioned query ``attn_q``
        drives the attention logits, but the residual stream walks the PLAIN ``residual_q``:

            x = residual_q + MHA(LN(attn_q), LN(kv), LN(kv));  x = x + FFN(LN(x))

        so whatever conditioned ``attn_q`` (e.g. the pooled-VL adapter) shapes WHERE the
        queries look in kv but never adds content to the output. Reuses this layer's existing
        parameters — no new state_dict keys. ``kv_mask`` semantics match ``forward``.
        """
        kp_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.attn(
            self.norm_q(attn_q), self.norm_kv(kv), self.norm_kv(kv),
            key_padding_mask=kp_mask,
            need_weights=False,
        )
        if kv_mask is not None:
            # Same fully-masked-row NaN sanitization as ``forward``.
            has_kv = kv_mask.any(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            attn_out = torch.nan_to_num(attn_out) * has_kv.to(attn_out.dtype)
        x = residual_q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


