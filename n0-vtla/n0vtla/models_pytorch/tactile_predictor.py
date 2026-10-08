"""TactileActionPredictor: maps (VL context, tactile-diff g) -> latent action z.

Two config-gated architectures (``predictor_arch``):

* ``"joint_kv"`` (A, DEFAULT — the running baseline, byte-identical to before this flag
  existed): z = Predictor(queries, kv=[VL_ctx ; g]) via cross-attention. VL context and
  tactile g are BOTH key/value content sources.
* ``"tactile_kv"`` (C): g is the ONLY key/value source; vl_ctx is demoted to a query
  conditioner that shapes the attention logits ONLY and never enters z's residual
  stream (no VL value path, no FiLM beta) — so VL cannot leak content into z. Per-layer:

      q_attn = q + vl_query_adapter(masked_pool(vl_ctx))   # VL drives attention only
      q = q + CrossAttn(query=q_attn, key=g, value=g)      # residual walks q, NOT q_attn
      q = q + FFN(norm(q))

  The C path also fixes two known A-path defects (frozen in A on purpose — A is the
  running baseline): the VL pool is a MASKED mean over valid prefix tokens (A attends
  padding rows via ctx_mask=ones), and no-tactile rows get a PER-SAMPLE null_g (A only
  substitutes null_g when the whole batch has g is None).

Used by Pi0VTLAPytorch in `latent` mode (the tactile-predictor training path). When no tactile is present, a learned
`null_g` token stands in for g so z is always produced.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from n0vtla.models_pytorch.tactile_encoder import TactileCrossAttnLayer


class TactileActionPredictor(nn.Module):
    """Cross-attention predictor distilling (VL context + tactile g) into n_latent action tokens z.

    n_latent learned query tokens attend over kv = [VL context ; tactile g] through
    n_layers cross-attention layers; the output z (B, n_latent, hidden_dim) is prepended
    to the flow-matching action suffix by Pi0VTLAPytorch (latent mode).

    Args:
        hidden_dim: shared token width of vl_ctx, tactile g, the latent queries, and z.
        n_latent:   number of learned query tokens = number of z tokens emitted.
        n_layers:   number of stacked cross-attention layers.
        n_heads:    attention heads per layer.
        predictor_arch: "joint_kv" (A, default) | "tactile_kv" (C). See module docstring.
                    The ``vl_query_adapter`` submodule exists ONLY in tactile_kv mode, so
                    joint_kv state_dict keys are byte-identical to before this flag existed
                    (A-mode checkpoints load unchanged).

    ``null_g`` is a learned stand-in used when no tactile g is available (tactile OFF or
    no tactile views this batch) so the predictor always emits z of the same shape.
    """
    def __init__(
        self,
        hidden_dim: int,
        n_latent: int = 10,
        n_layers: int = 2,
        n_heads: int = 8,
        predictor_arch: str = "joint_kv",
    ):
        super().__init__()
        if predictor_arch not in ("joint_kv", "tactile_kv"):
            raise ValueError(f"unknown predictor_arch={predictor_arch!r} (expected 'joint_kv' or 'tactile_kv')")
        self.predictor_arch = predictor_arch
        self.n_latent = n_latent
        self.latent_queries = nn.Parameter(torch.randn(n_latent, hidden_dim) * 0.02)
        # Learned stand-in for tactile g when g is None (tactile OFF or no tactile views
        # this batch); guarantees the predictor still emits z of shape (B, n_latent, D).
        self.null_g = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [TactileCrossAttnLayer(hidden_dim, n_heads) for _ in range(n_layers)]
        )
        if predictor_arch == "tactile_kv":
            # Arch C ONLY: VL -> query conditioner (shapes attention logits; NOT in the
            # residual / value path, so no VL content reaches z). Constructed only here so
            # A-mode state_dict keys stay byte-identical (ckpt compat).
            self.vl_query_adapter = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        vl_ctx: torch.Tensor,
        g: torch.Tensor | None,
        g_mask: torch.Tensor | None = None,
        vl_ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vl_ctx: (B, N_ctx, D); g: (B, N_g, D) or None -> z: (B, n_latent, D).

        g_mask: optional (B, N_g) bool, True for valid g tokens. Needed when g is
        zero-padded across a variable number of tactile views (the optional latent-cache path); VL
        context tokens are always valid. None → attend all kv (online path).

        vl_ctx_mask: optional (B, N_ctx) bool, True for valid (non-padding) prefix tokens
        (the caller's prefix_pad_masks). USED ONLY by arch C's masked VL pool; arch A
        deliberately IGNORES it — A is the running baseline and its behavior (ctx_mask=ones,
        padding rows attended: known defect S5) is frozen. The S5 fix lands in the C path.
        """
        if self.predictor_arch == "tactile_kv":
            return self._forward_tactile_kv(vl_ctx, g, g_mask, vl_ctx_mask)
        # ---------------- arch A ("joint_kv") — byte-identical frozen baseline ----------------
        B, N_ctx = vl_ctx.shape[0], vl_ctx.shape[1]
        if g is None:
            # No tactile for the whole batch: substitute the learned null_g token so kv is
            # never empty and z is always produced. (Per-sample missing views are handled
            # by g_mask below, not here.)
            g = self.null_g.unsqueeze(0).expand(B, -1, -1).to(vl_ctx.dtype)
            g_mask = None
        kv = torch.cat([vl_ctx, g.to(vl_ctx.dtype)], dim=1)        # (B, N_ctx+N_g, D)
        kv_mask = None
        if g_mask is not None:
            ctx_mask = torch.ones(B, N_ctx, dtype=torch.bool, device=vl_ctx.device)
            kv_mask = torch.cat([ctx_mask, g_mask.to(vl_ctx.device)], dim=1)
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1).to(vl_ctx.dtype)
        for layer in self.layers:
            q = layer(q, kv, kv_mask=kv_mask)
        return q

    def _forward_tactile_kv(
        self,
        vl_ctx: torch.Tensor,
        g: torch.Tensor | None,
        g_mask: torch.Tensor | None,
        vl_ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Arch C forward: g is the ONLY K/V source; VL conditions the attention query only.

        Residual discipline (the soul of C — per Codex adversarial review):
            q_attn = q + vl_query_adapter(masked_pool(vl_ctx))
            q = q + CrossAttn(query=q_attn, key=g, value=g)   # residual walks q, NOT q_attn
            q = q + FFN(norm(q))
        vl_ctx never appears as a value and has no additive path into z (no FiLM beta).
        """
        B = vl_ctx.shape[0]
        if g is None:
            # No tactile for the whole batch: learned null_g stand-in so kv is never empty.
            g = self.null_g.unsqueeze(0).expand(B, -1, -1).to(vl_ctx.dtype)
            g_mask = None
        elif g_mask is not None:
            # PER-SAMPLE null_g (fixes the A-path defect where null_g only covers the whole-
            # batch g-is-None case): a row with NO valid tactile token gets null_g at position
            # 0 with mask=True, so its cross-attn has a real key instead of the all-masked
            # NaN->zero sanitization path.
            no_tac = ~g_mask.any(dim=1)                                   # (B,)
            if bool(no_tac.any()):
                g = g.clone()
                g[no_tac, 0] = self.null_g.to(g.dtype)
                g_mask = g_mask.clone()
                g_mask[no_tac, 0] = True

        # Residual base = plain learned queries (NO VL content here).
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1).to(vl_ctx.dtype)
        # VL -> pooled query conditioner. MASKED mean over valid prefix tokens (fixes S5:
        # padding rows are excluded); a missing mask -> plain mean (all tokens valid).
        if vl_ctx_mask is not None:
            m = vl_ctx_mask.to(device=vl_ctx.device, dtype=torch.bool)
            mf = m.to(vl_ctx.dtype)[:, :, None]                            # (B, N_ctx, 1)
            pooled = (vl_ctx * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            pooled = vl_ctx.mean(dim=1, keepdim=True)                      # (B, 1, D)
        # dtype-robust: adapter params may be fp32 while activations are bf16 (or vice versa).
        vl_q_cond = self.vl_query_adapter(pooled.to(self.vl_query_adapter.weight.dtype)).to(q.dtype)
        kv = g.to(q.dtype)
        for layer in self.layers:
            q = layer.forward_query_cond(residual_q=q, attn_q=q + vl_q_cond, kv=kv, kv_mask=g_mask)
        return q
