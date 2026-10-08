"""N0VTLAPolicy: tactile action-predictor on the clean upstream PI0Pytorch base.

The predictor is trained end-to-end: z is shaped purely by the action-MSE gradient, with no
separate supervised objective. (The paper's supervised predictor-grounding stage, which
regresses z against a future-touch target, is not part of this post-training repository.)

This is a SEPARATE predictor path, NOT the upstream direct-injection tactile hooks that
already live on ``PI0Pytorch`` (``_encode_tactile`` / ``embed_prefix(tactile_vlm_tokens=)``
/ ``embed_suffix(tactile_expert_tokens=)``). Those stay untouched; the predictor config runs
with ``use_tactile=False`` so none of the direct-injection modules are even constructed.

Gating (HARD REQUIREMENT):
  ``N0VTLAConfig.tactile_predictor_enabled`` defaults to ``False``. When False:
    * ``__init__`` constructs NO extra submodules  -> state_dict identical to the base.
    * ``forward`` / ``sample_actions`` / ``_preprocess_observation`` delegate straight to
      ``super()`` -> byte-identical to the clean base (this is the CI gate).
  Only when True does the predictor path activate.

Predictor path (v0 — the unsupervised variant: no z* target and no separate predictor loss):

  Phase 1  clean prefix (RGB + language, NO tactile) forward through PaliGemma -> vl_ctx.
  Phase 2  g = concat_over_views DINOv2(tac_t - tac_0);   z = Predictor(vl_ctx, g)   [n_latent].
  Phase 3  action expert over [z ; state ; noisy_actions] -> action flow-matching MSE. By
           default a single joint forward re-encodes the prefix together with the suffix (the
           safe path; the prefix is forwarded TWICE per step). With ``use_prefix_cache=True``
           Phase 1 also emits the prefix K/V (use_cache=True) and Phase 3 forwards ONLY the
           suffix against it (mirrors base ``PI0Pytorch.sample_actions``/``denoise_step``),
           skipping the 2nd prefix encode. Flag is OFF by default; verify cache-on == cache-off
           with scripts/verify_prefix_cache.py before enabling.
  loss = action_mse ONLY. The Predictor is trained purely by the action gradient flowing back
  through z (unsupervised).

"""
from __future__ import annotations

import math

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from n0vtla.models.pi0_config import Pi0Config
from n0vtla.models_pytorch.pi0_pytorch import PI0Pytorch
from n0vtla.models_pytorch.pi0_pytorch import make_att_2d_masks


# ---------------------------------------------------------------------------
# Config  (subclass keeps the base Pi0Config byte-identical)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class N0VTLAConfig(Pi0Config):
    """Pi0Config + the v0 tactile-predictor knobs.

    All base Pi0Config fields are inherited. New fields are additive with defaults that
    reproduce the base model exactly when ``tactile_predictor_enabled=False``.

    Training is unsupervised: there is no z* target and no predictor loss, so ``n_latent`` is
    independent of the tactile encoder's token count (``tactile_pool_grid`` only sizes g,
    the Predictor's key/value). ``predictor_loss_weight`` is reserved for the later supervised phase
    and is inert here.
    """

    tactile_predictor_enabled: bool = False
    # How tactile evidence reaches the policy. One of:
    #   "latent"        — the predictor distils the DINOv2 difference tokens into n_latent latent
    #                     tokens z, injected into the action expert (the N0-VTLA path),
    #   "vlm_concat"    — DINOv2 difference tokens appended to the VLM prefix, full prefix-LM
    #                     attention,
    #   "expert_concat" — DINOv2 difference tokens through a ZERO-INIT g_proj straight into the
    #                     expert suffix, with no gate.
    # The two concat modes require tactile_predictor_enabled=False.
    tactile_mode: str = "latent"
    n_latent: int = 5
    tactile_pool_grid: int = 3          # 1 + pool_grid**2 tokens for g (default DINOv2 layout)
    predictor_n_layers: int = 2
    predictor_n_heads: int = 8
    predictor_loss_weight: float = 0.5      # reserved (anchored-unfreeze λ); inert
    # Predictor architecture (see tactile_predictor.py module docstring):
    #   "joint_kv"   (A, DEFAULT) — cross-attn over kv=[vl_ctx ; g]. The running baseline;
    #                behavior and state_dict are byte-identical to before this field existed.
    #   "tactile_kv" (C) — g is the ONLY K/V source; vl_ctx only conditions the attention
    #                query (masked-pooled adapter), so VL content cannot leak into z. The C
    #                path also fixes the A-path S5 padding defect (masked VL pool) and adds
    #                per-sample null_g for no-tactile rows. Adds ONE new submodule
    #                (tactile_predictor.vl_query_adapter) — only present in C-mode checkpoints.
    predictor_arch: str = "joint_kv"

    # Gate the latent tokens behind a zero-initialised scalar. True creates the
    # Parameter self.z_gate (torch.zeros(1)); _embed_suffix_with_z multiplies z_w by it at
    # the single entry point (covers BOTH the training forward AND the sampler, which both route
    # z_w through _embed_suffix_with_z). At init 0 the z block is muted (suffix == the base-policy
    # suffix, bitwise), so the action expert starts exactly at the pretrained base-policy behaviour and
    # the gate opens under gradient. Pure scalar, NO tanh, init 0. The Parameter is created ONLY
    # when this flag is True, so a gate-OFF checkpoint's state_dict key set is byte-unchanged.
    z_gate_zero_init: bool = False
    # Vision-language dropout. During training each sample in the batch, chosen
    # independently with probability p, has its suffix→prefix attention block fully masked (action
    # AND z tokens see NO prefix KV), so the action expert must lean on z. z's OWN computation is
    # unaffected: the predictor still consumes the full vl_ctx. Position
    # ids are NOT changed (only attention is masked). Consumes the model RNG (torch.rand on device),
    # so disable for bitwise reproducibility. eval/serve (self.training=False) NEVER triggers it.
    vl_dropout_prob: float = 0.0
    # Also feed the RAW tactile tokens g into the action expert alongside z
    # (``g_w = self.g_proj(g); cond = torch.cat([z_w, g_w], dim=1)``). This gives the expert a
    # tactile channel that does NOT route through the trainable predictor, so a predictor that
    # drifts during training cannot take the expert's only tactile signal down with it — a
    # structural stabiliser worth having if end-to-end training proves unstable.
    # When True, __init__ creates ``g_proj``
    # (llm_dim -> action-expert width, mirroring z_proj) plus a zero-init scalar ``g_gate``
    # _embed_suffix_with_z injects suffix = [z ; g_proj(g)*g_gate ; action tokens]. The g tokens
    # join the z condition block; a g_mask=False token (placeholder view) gets pad_masks=False so
    # the expert never attends it. Default False -> NEITHER Parameter exists (state_dict key set
    # byte-unchanged) and the g kwarg stays unused, leaving the suffix bitwise unchanged.
    g_to_expert: bool = False

    # Perf (Phase-3 KV cache): skip the prefix re-encode by caching the Phase-1 prefix K/V and
    # forwarding ONLY the suffix against it (mirrors base PI0Pytorch.sample_actions). OFF by
    # default so the running-training path is byte-unchanged; enable only AFTER
    # scripts/verify_prefix_cache.py confirms cache-on == cache-off (bf16-noise level).
    use_prefix_cache: bool = False

    # Inference-only memory optimization: construct the large PaliGemma+action-expert backbone
    # (the `super().__init__(config)` call in N0VTLAPolicy.__init__) on torch.device("meta")
    # instead of materializing it in PyTorch's default fp32 CPU dtype first. Weights are then
    # ASSIGNED in directly from the checkpoint (N0VTLAConfig.load_pytorch: load_state_dict(...,
    # assign=True)), which replaces each meta parameter with the checkpoint's own tensor (already
    # in its final mixed fp32/bf16 dtype) instead of copy_()-ing checkpoint values into a
    # pre-allocated fp32 tensor. Avoids ever materializing a full fp32 copy of the ~4.1B-param
    # model, roughly halving the load-time system RAM peak (see
    # docs/REAL_ROBOT_INFERENCE.md §2.2). Submodules built AFTER `super().__init__`
    # (e.g. FrozenDINOv2TactileEncoder, which uses AutoModel.from_pretrained and is incompatible
    # with a blanket meta-device context without `accelerate`) are NOT affected -- they continue
    # to construct normally (real CPU memory, ~344MB, not the main cost). OFF by default so
    # training and every existing caller is unaffected; opt in for inference only.
    low_cpu_mem_usage: bool = False

    # --- Stage-1 predictor-grounding pretraining (paper Sec 4.2; see docs/MID_TRAIN.md) ---
    # Gates construction of ``tactile_recon_head`` and enables ``forward_stage1``. Requires
    # tactile_predictor_enabled=True. Off by default, in which case the state_dict key set is
    # unchanged.
    stage1_pretrain_enabled: bool = False
    # Side length of the coarse (grid x grid) reconstruction target/output. Not specified by the
    # paper (see TactileReconHead / N0VTLAPolicy._build_future_target).
    stage1_recon_grid: int = 8
    # lambda_rec in the paper's L_1 = L_NCE + lambda_rec * L_rec (Eq. 5). Paper states
    # lambda_rec > 0 but does not publish a value, so this is a tunable hyperparameter.
    stage1_lambda_rec: float = 0.5
    # InfoNCE temperature. Eq. 3-4 use RAW cosine similarity as the logit (s_ij itself, no
    # scaling term anywhere in the equations) -- i.e. temperature=1, which is the default.
    # A lower temperature is the standard SimCLR/CLIP-style choice, but deviates from the
    # paper's equations.
    stage1_temperature: float = 1.0

    def load_pytorch(self, train_config, weight_path: str):
        """Serve/eval load path: build N0VTLAPolicy and load the trained weights.

        Overrides BaseModelConfig.load_pytorch so the predictor submodules are instantiated and
        populated on serve/eval (create_trained_policy does not run the train-script
        monkey-patch); see the comment below for why the base hardcoded-PI0Pytorch path fails.
        """
        # Base BaseModelConfig.load_pytorch (models/model.py:286) hardcodes PI0Pytorch, which
        # for a predictor checkpoint would silently leave the predictor submodules unloaded on the
        # serve/eval path (create_trained_policy does NOT run the train-script monkey-patch).
        # Build the predictor class here so serve/eval load the trained predictor weights.
        import safetensors.torch as _st

        model = N0VTLAPolicy(config=train_config.model)

        if bool(getattr(self, "low_cpu_mem_usage", False)):
            # Model was constructed with its backbone on torch.device("meta") (see
            # N0VTLAConfig.low_cpu_mem_usage / N0VTLAPolicy.__init__). assign=True REPLACES each
            # meta parameter/buffer with the checkpoint's own tensor (no fp32 intermediate),
            # rather than safetensors.torch.load_model's plain load_state_dict, which does an
            # in-place copy_() that requires the target to already be a real (non-meta) tensor.
            state_dict = _st.load_file(weight_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
            leftover_meta = _repair_meta_leftovers_after_low_mem_load(model)
            # Anything the checkpoint AND the repair pass above didn't cover is still on the meta
            # device (no real data) -- fail loudly rather than silently run inference with
            # garbage/uninitialized weights.
            if leftover_meta:
                raise RuntimeError(
                    "N0VTLAConfig.low_cpu_mem_usage=True: checkpoint did not cover every "
                    f"parameter/buffer -- {len(leftover_meta)} left on the meta device with no "
                    f"real data: {leftover_meta[:10]}{'...' if len(leftover_meta) > 10 else ''}. "
                    "Set low_cpu_mem_usage=False to use the normal (higher-peak-RAM) load path."
                )
        else:
            missing, unexpected = _st.load_model(model, weight_path, strict=False)

        if missing or unexpected:
            import logging

            logging.info(f"Loaded N0VTLAPolicy ckpt with missing={missing}, unexpected={unexpected}")
        return model


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# The tactile predictor submodule used to be called ``tactile_prior``, so its parameters were
# stored under "tactile_prior.*". Checkpoints written back then are still valid weights, and the
# loaders run with strict=False, which would silently leave the renamed submodule randomly
# initialized. Rewriting the keys as they come in keeps those checkpoints loadable.
_LEGACY_KEY_PREFIXES = (("tactile_prior.", "tactile_predictor."),)


def _repair_meta_leftovers_after_low_mem_load(model: "N0VTLAPolicy") -> list[str]:
    """Fix up the handful of tensors a checkpoint never covers, after a meta+assign load.

    ``load_state_dict(..., assign=True)`` only touches keys present in the checkpoint file, so
    two categories of tensor are left stranded on the meta device (no real data) even on a
    checkpoint that fully matches the model:

      1. Non-persistent buffers (``register_buffer(..., persistent=False)``) -- e.g. each Gemma
         attention block's RoPE ``inv_freq`` and SigLIP's vision-embedding ``position_ids``. These
         are pure functions of config, are NEVER written to a state_dict (persistent=False), and
         get computed at construction time -- which, under the meta context, computed a *meta*
         result instead of a real one. Recomputed here for real (tiny; negligible memory).
      2. Tied weights whose alias was dropped at save time. safetensors.torch.load_model's
         strict=False path auto-resolves these via ``_remove_duplicate_names`` (it diffs
         ``model.state_dict()`` for shared storage and skips the dropped alias, since copying into
         the kept name already updates both -- they're the same tensor in a normally-constructed
         model). ``assign=True`` breaks that aliasing (it replaces tensors wholesale rather than
         copying in place), so PaliGemma's ``embed_tokens.weight`` -- tied to, and dropped from the
         checkpoint in favor of, ``lm_head.weight`` -- needs to be re-pointed at it explicitly.

    Returns the names of any parameter/buffer still on the meta device after these repairs (should
    be empty for a checkpoint that otherwise fully matches the model).
    """
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.is_meta:
            real_inv_freq, _ = module.rope_init_fn(module.config, torch.device("cpu"))
            module.inv_freq = real_inv_freq
            module.original_inv_freq = real_inv_freq
        position_ids = getattr(module, "position_ids", None)
        if isinstance(position_ids, torch.Tensor) and position_ids.is_meta:
            module.position_ids = torch.arange(module.num_positions).expand((1, -1))

    paligemma = model.paligemma_with_expert.paligemma
    lm_embed_tokens = paligemma.model.language_model.embed_tokens
    if lm_embed_tokens.weight.is_meta and not paligemma.lm_head.weight.is_meta:
        lm_embed_tokens.weight = paligemma.lm_head.weight

    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    leftover += [n for n, b in model.named_buffers() if b.is_meta]
    return leftover


def _remap_legacy_predictor_keys(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
) -> None:
    """``load_state_dict`` pre-hook: rename legacy tactile-predictor keys in place."""
    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    for old, new in _LEGACY_KEY_PREFIXES:
        for key in [k for k in state_dict if k.startswith(prefix + old)]:
            state_dict[prefix + new + key[len(prefix + old) :]] = state_dict.pop(key)


class N0VTLAPolicy(PI0Pytorch):
    """PI0Pytorch + a config-gated tactile action-predictor path.

    When ``config.tactile_predictor_enabled`` is False every override falls through to the base,
    so the model is a drop-in, byte-identical replacement for ``PI0Pytorch``.
    """

    def __init__(self, config) -> None:
        if bool(getattr(config, "low_cpu_mem_usage", False)):
            # Meta-construct ONLY the large PaliGemma+action-expert backbone. PI0Pytorch.__init__
            # builds everything from `config` alone -- no from_pretrained calls -- so it is safe
            # under a blanket meta-device context with no `accelerate` dependency. Real weights
            # are assigned in from the checkpoint afterward (N0VTLAConfig.load_pytorch,
            # assign=True). Everything constructed below this point (tactile encoder/predictor,
            # ...) is NOT wrapped -- FrozenDINOv2TactileEncoder uses AutoModel.from_pretrained,
            # which is incompatible with a blanket meta context without `accelerate`, and it's
            # small (~344MB) anyway, so it just constructs normally.
            with torch.device("meta"):
                super().__init__(config)
        else:
            super().__init__(config)
        self.tactile_predictor_enabled = bool(getattr(config, "tactile_predictor_enabled", False))
        # Phase-3 prefix-KV-cache optimization (see N0VTLAConfig.use_prefix_cache). Read as
        # an INSTANCE flag so a verify/bench harness can flip it at runtime on a fixed model.
        # Harmless when the predictor is gated off (only the predictor forward/sample paths read it).
        import os as _os
        self.use_prefix_cache = bool(getattr(config, "use_prefix_cache", False)) or _os.environ.get("VTLA_PREFIX_CACHE") == "1"
        # Training stage — read BEFORE the gate-off early return so ``forward`` can route on it

        # Baseline-frame key suffix (data contract): the input pipeline
        # (policies/flexiv_policy.py:87, same ".baseline") emits "<view>_tactile.baseline" for
        # frame 0 alongside the current "<view>_tactile". v0 loads only [baseline(frame0),
        # current]; there is NO future frame.
        self.BASELINE_SUFFIX = ".baseline"
        # Future frame (tac_{t+H}) key suffix — emitted by the input transform only for a
        # length-3 delta_timestamps stack (policies/flexiv_policy.py::_split_tactile_stack). Absent
        # in v0/e2e (2-frame stack), so _last_tac_f stays None there.
        self.FUTURE_SUFFIX = ".future"
        # Tactile stashes: written by _preprocess_observation, read by the predictor forward/sample
        # paths (_compute_z).
        self._last_tac_t: dict[str, torch.Tensor] | None = None
        self._last_tac_0: dict[str, torch.Tensor] | None = None
        self._last_tac_f: dict[str, torch.Tensor] | None = None  # stage-1 future frames (else None)
        self._last_g_mask: torch.Tensor | None = None  # (B, N_g) from _build_g; g→expert pad mask
        # Per-step loss-parts stash for EXTERNAL logging: written by _forward_predictor, read by the
        # trainer (not by this class). v0 holds only {"act": ...}; RESERVED to also carry the
        # Phase-2 predictor-loss parts. Safe to ignore if you are not wiring up loss logging.
        self._last_loss_parts: dict[str, float] = {}

        # --- baseline tactile-injection modes (ablation; see N0VTLAConfig.tactile_mode) ---
        self.tactile_mode = str(getattr(config, "tactile_mode", "latent"))
        if self.tactile_mode != "latent":
            assert not self.tactile_predictor_enabled, "concat modes exclude the z/predictor path"
            from n0vtla.models_pytorch.tactile_encoder import FrozenDINOv2TactileEncoder
            llm_dim = self.paligemma_with_expert.paligemma.config.hidden_size
            pool_grid = int(getattr(config, "tactile_pool_grid", 3))
            self.tactile_encoder = FrozenDINOv2TactileEncoder(llm_dim=llm_dim, pool_grid=pool_grid)
            assert self.tactile_encoder.out_dim == llm_dim
            if self.tactile_mode == "expert_concat":
                # ZERO-INIT projection, NO gate: step 0 == the base policy (g contributes exactly
                # nothing) while grad_W = delta·g^T != 0 from the first step — the one-sided-zero
                # unlock (codex review), avoiding both the zero-gate multiplicative deadlock and
                # a 0.1x-random noise injection.
                self.g_proj = nn.Linear(llm_dim, self.action_in_proj.out_features)
                nn.init.zeros_(self.g_proj.weight)
                nn.init.zeros_(self.g_proj.bias)
            return

        if not self.tactile_predictor_enabled:
            # Gate OFF: construct nothing extra -> identical parameters to the base.
            return

        # --- predictor submodules (only when enabled) ---
        from n0vtla.models_pytorch.tactile_encoder import FrozenDINOv2TactileEncoder
        from n0vtla.models_pytorch.tactile_predictor import TactileActionPredictor

        llm_dim = self.paligemma_with_expert.paligemma.config.hidden_size
        n_latent = int(getattr(config, "n_latent", 5))
        pool_grid = int(getattr(config, "tactile_pool_grid", 3))

        # Frozen DINOv2 tactile encoder -> (B, n_tokens, llm_dim). n_tokens = 1 + pool_grid**2.
        # v0 has NO z* target, so n_latent is INDEPENDENT of the encoder token count (no pairing
        # constraint). g (encoder tokens) is only the Predictor's key/value; z stays n_latent.
        self.tactile_encoder = FrozenDINOv2TactileEncoder(llm_dim=llm_dim, pool_grid=pool_grid)
        assert self.tactile_encoder.out_dim == llm_dim

        # Predictor: n_latent learned queries -> z (B, n_latent, llm_dim). predictor_arch selects the
        # kv layout: "joint_kv" (A) cross-attends [vl_ctx ; g]; "tactile_kv" (C) cross-attends
        # g ONLY with vl_ctx demoted to a query conditioner (see tactile_predictor.py).
        self.tactile_predictor = TactileActionPredictor(
            hidden_dim=llm_dim,
            n_latent=n_latent,
            n_layers=int(getattr(config, "predictor_n_layers", 2)),
            n_heads=int(getattr(config, "predictor_n_heads", 8)),
            predictor_arch=str(getattr(config, "predictor_arch", "joint_kv")),
        )
        # Checkpoints written before the tactile_prior -> tactile_predictor rename carry
        # "tactile_prior.*" keys. Remap them on the way in so older weights still load, whichever
        # entrypoint does the loading (safetensors.load_model routes through load_state_dict too).
        self.register_load_state_dict_pre_hook(_remap_legacy_predictor_keys)
        # z is produced in llm_dim (to match z* for the cosine loss); project to the
        # action-expert width before injecting as suffix tokens.
        self.z_proj = nn.Linear(llm_dim, self.action_in_proj.out_features)

        # state_dict key set is byte-unchanged. At init 0 -> z_w muted -> suffix == the base-policy
        # suffix (bitwise), so stage-2 starts exactly at the pretrained base-policy action behavior and
        # the gate opens under gradient. Applied at the single _embed_suffix_with_z entry point.
        self.z_gate_zero_init = bool(getattr(config, "z_gate_zero_init", False))
        if self.z_gate_zero_init:
            self.z_gate = nn.Parameter(torch.zeros(1))

        # g→expert raw-tactile anchor: project the
        # encoder tokens g DIRECTLY into the action expert, bypassing the trainable predictor, so the
        # expert keeps a tactile signal that cannot drift with the predictor. Both Parameters are
        # created ONLY when enabled (off -> state_dict key set byte-unchanged); g_gate zero-init
        # suffix injection: see _build_g/_last_g_mask.
        self.g_to_expert = bool(getattr(config, "g_to_expert", False))
        if self.g_to_expert:
            self.g_proj = nn.Linear(llm_dim, self.action_in_proj.out_features)
            self.g_gate = nn.Parameter(torch.zeros(1))

        # Stage-1 predictor-grounding pretraining (see N0VTLAConfig.stage1_pretrain_enabled).
        # Constructed ONLY when enabled, so every other config's state_dict key set is unaffected.
        self.stage1_pretrain_enabled = bool(getattr(config, "stage1_pretrain_enabled", False))
        if self.stage1_pretrain_enabled:
            from n0vtla.models_pytorch.tactile_recon_head import TactileReconHead

            self.tactile_recon_head = TactileReconHead(
                hidden_dim=llm_dim, grid=int(getattr(config, "stage1_recon_grid", 8))
            )


    # ------------------------------------------------------------------
    # Tactile key discovery (generalize to N views; never hardcode 2)
    # ------------------------------------------------------------------
    def _tactile_keys(self, images: dict) -> list[str]:
        """The tactile view keys present as CURRENT frames in the observation image dict.

        Prefers the configured ``tactile_image_keys`` (authoritative, ordered), else falls
        back to any key ending in ``_tactile``. Generalizes to N views (2 single-arm,
        4 dual-arm) — no hardcoded count.
        """
        configured = tuple(getattr(self.config, "tactile_image_keys", ()) or ())
        if configured:
            return [k for k in configured if k in images]
        return [k for k in images if isinstance(k, str) and k.endswith("_tactile")]

    # ------------------------------------------------------------------
    # Override: pop tactile (current/baseline/future) BEFORE base preprocessing
    # ------------------------------------------------------------------
    def _preprocess_observation(self, observation, *, train: bool = True):
        """Pop tactile (current + baseline) from ``observation.images`` by key naming, then
        delegate to the base.

        Implements the latent branch, minus the future
        frame (v0 is unsupervised). The tactile tensors are POPPED before
        ``super()._preprocess_observation`` so they never reach SigLIP; the predictor path
        re-encodes them via the frozen DINOv2 encoder.

        Gate OFF -> straight delegation (byte-identical to base).
        """
        if not self.tactile_predictor_enabled and getattr(self, "tactile_mode", "latent") == "latent":
            return super()._preprocess_observation(observation, train=train)

        if hasattr(observation, "images"):
            # Work on dictionary copies so repeated forwards receive the same tactile inputs.
            # Observation stores plain dictionary references, and popping from them would mutate
            # the caller's object.
            observation = observation.replace(
                images=dict(observation.images),
                image_masks=dict(observation.image_masks)
                if getattr(observation, "image_masks", None) is not None
                else observation.image_masks,
            )
            # NHWC -> NCHW guard for float inputs; the base
            # preprocessing only auto-permutes uint8. DINOv2 expects (B, 3, H, W).
            for k, img in list(observation.images.items()):
                if img.ndim == 4 and img.shape[-1] == 3 and img.shape[1] != 3:
                    observation.images[k] = img.permute(0, 3, 1, 2).contiguous()

            tac_t: dict[str, torch.Tensor] = {}
            tac_0: dict[str, torch.Tensor] = {}
            tac_f: dict[str, torch.Tensor] = {}
            # Per-view validity masks. CanonicalTactileInputs emits a zero PLACEHOLDER (image_mask=
            # False) for any tactile view a platform lacks, to keep batch keys uniform across mixed
            # embodiments. We MUST retain those masks (not discard them) so _build_g/
            # _future_flow can exclude placeholder views by MASK, not key-presence — else a no-tactile
            # row (e.g. a umi entry, or single-arm's 2 absent right-hand views) is treated as real
            # tactile and pollutes InfoNCE/recon (migration-review critical fix). mask is (B,) per
            # view; a missing mask -> treat that view as all-real.
            tac_mask: dict[str, torch.Tensor] = {}
            tac_mask_f: dict[str, torch.Tensor] = {}
            has_masks = hasattr(observation, "image_masks")
            for k in self._tactile_keys(observation.images):
                tac_t[k] = observation.images.pop(k)
                mk = observation.image_masks.pop(k, None) if has_masks else None
                if mk is not None:
                    tac_mask[k] = mk
                bk = k + self.BASELINE_SUFFIX
                if bk in observation.images:
                    tac_0[k] = observation.images.pop(bk)
                    if has_masks:
                        observation.image_masks.pop(bk, None)
                # Stage-1 future frame (present only for a 3-frame stack); absent in v0/e2e.
                fk = k + self.FUTURE_SUFFIX
                if fk in observation.images:
                    tac_f[k] = observation.images.pop(fk)
                    mfk = observation.image_masks.pop(fk, None) if has_masks else None
                    if mfk is not None:
                        tac_mask_f[k] = mfk
            self._last_tac_t = tac_t or None
            self._last_tac_0 = tac_0 or None
            self._last_tac_f = tac_f or None
            self._last_tac_mask = tac_mask or None
            self._last_tac_mask_f = tac_mask_f or None
        else:
            self._last_tac_t = self._last_tac_0 = self._last_tac_f = None
            self._last_tac_mask = self._last_tac_mask_f = None

        return super()._preprocess_observation(observation, train=train)

    # ------------------------------------------------------------------
    # Predictor helpers
    # ------------------------------------------------------------------
    def _build_g(self, tac_t: dict, tac_0: dict):
        """g = concat over views of DINOv2(tac_t - tac_0). Returns (g|None, g_mask|None, has_tac(B,)).

        ``g_mask`` (B, n_tokens*V) bool marks tokens from REAL (non-placeholder) tactile views. A
        placeholder view (image_mask=False, see _preprocess_observation) is STILL encoded into g so
        the token layout stays batch-uniform across mixed embodiments, but g_mask=False there so
        ``tactile_predictor``'s cross-attn does NOT attend it (migration-review fix; by MASK not
        key-presence). A missing per-view mask (all-real) -> all-True (back-compat).
        ``has_tac`` (B,) = the row has >=1 real tactile view (reserved for the Phase-2 contact gate).

        Includes the per-view mask.

        Side effect: stashes ``self._last_g_mask = g_mask`` for the g→expert suffix injection
        (_forward_predictor / _sample_actions_predictor read it right after _compute_z). A stash, not a
        return-signature change, because _compute_z's 3-tuple unpack is frozen by external
        scripts (eval_stage1_predictor.py / probe_z_tactile_dependence.py) that must not be touched.
        """
        keys = [k for k in tac_t if k in tac_0]
        if not keys:
            self._last_g_mask = None
            any_img = next(iter(tac_t.values()), None)
            B = any_img.shape[0] if any_img is not None else 0
            return None, None, torch.zeros(B, dtype=torch.bool)
        view_masks = getattr(self, "_last_tac_mask", None) or {}
        toks, tmask = [], []
        for k in keys:
            diff = tac_t[k].float() - tac_0[k].float()
            tok = self.tactile_encoder(diff)                # (B, n_tokens, D)
            toks.append(tok)
            rk = view_masks.get(k)
            if rk is None:
                rk = torch.ones(tok.shape[0], dtype=torch.bool, device=tok.device)
            else:
                rk = rk.to(device=tok.device, dtype=torch.bool).reshape(-1)
            tmask.append(rk[:, None].expand(-1, tok.shape[1]))   # (B, n_tokens)
        g = torch.cat(toks, dim=1)                          # (B, n_tokens * V, D)
        g_mask = torch.cat(tmask, dim=1)                    # (B, n_tokens * V)
        has_tac = g_mask.any(dim=1)                         # (B,)
        self._last_g_mask = g_mask
        return g, g_mask, has_tac

    # ------------------------------------------------------------------
    # Both ride the BASE forward/sample paths untouched: embed_prefix/embed_suffix are the
    # hooks those paths already call, so training AND sampling inherit the injection for free.
    # Mode-gated -> "latent" (default) is byte-identical to before.
    # ------------------------------------------------------------------
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, tactile_vlm_tokens=None):
        """vlm_concat: append DINOv2 tactile
        DIFF tokens after the RGB+language prefix with full prefix-LM attention (att 0, same as
        image tokens). Improvements over 1.0-b1: diff (tac_t - tac_0) instead of the raw current
        frame (the gel signal lives in the diff), and per-view validity masks as pad_masks so
        placeholder views are never attended."""
        embs, pad_masks, att_masks = super().embed_prefix(
            images, img_masks, lang_tokens, lang_masks, tactile_vlm_tokens=tactile_vlm_tokens
        )
        if getattr(self, "tactile_mode", "latent") != "vlm_concat" or not self._last_tac_t:
            return embs, pad_masks, att_masks
        g, g_mask, _ = self._build_g(self._last_tac_t, self._last_tac_0 or {})
        if g is None:
            return embs, pad_masks, att_masks
        tac = g.to(dtype=embs.dtype)
        B, n_t = tac.shape[0], tac.shape[1]
        pad = (
            g_mask.to(device=embs.device, dtype=pad_masks.dtype)
            if g_mask is not None
            else torch.ones(B, n_t, dtype=pad_masks.dtype, device=embs.device)
        )
        att = torch.zeros(B, n_t, dtype=att_masks.dtype, device=embs.device)
        return (
            torch.cat([embs, tac], dim=1),
            torch.cat([pad_masks, pad], dim=1),
            torch.cat([att_masks, att], dim=1),
        )

    def embed_suffix(self, state, noisy_actions, timestep, tactile_expert_tokens=None):
        """expert_concat: suffix = [g_proj(g)(zero-init, NO gate) ; base suffix]. The g block is
        a new attention block (first token att 1, rest 0 — same layout as the z block); action
        outputs are sliced from the tail (-H:) by the caller so lead tokens never reach
        action_out_proj. latent/vlm_concat -> straight delegation."""
        a_embs, a_pad, a_att, adarms_cond = super().embed_suffix(
            state, noisy_actions, timestep, tactile_expert_tokens=tactile_expert_tokens
        )
        if getattr(self, "tactile_mode", "latent") != "expert_concat" or not self._last_tac_t:
            return a_embs, a_pad, a_att, adarms_cond
        g, g_mask, _ = self._build_g(self._last_tac_t, self._last_tac_0 or {})
        if g is None:
            return a_embs, a_pad, a_att, adarms_cond
        g_w = self.g_proj(g.to(self.g_proj.weight.dtype)).to(a_embs.dtype)
        B, n_g = g_w.shape[0], g_w.shape[1]
        device = a_embs.device
        g_pad = (
            g_mask.to(device=device, dtype=a_pad.dtype)
            if g_mask is not None
            else torch.ones(B, n_g, dtype=a_pad.dtype, device=device)
        )
        g_att = torch.zeros(B, n_g, dtype=a_att.dtype, device=device)
        g_att[:, 0] = 1  # new attention block after the prefix
        return (
            torch.cat([g_w, a_embs], dim=1),
            torch.cat([g_pad, a_pad], dim=1),
            torch.cat([g_att, a_att], dim=1),
            adarms_cond,
        )

    def _embed_suffix_with_z(self, state, noisy_actions, timestep, z, g=None, g_mask=None):
        """suffix = [z(n_latent) ; g(optional 1.0-parity anchor) ; base action suffix]. Ported
        for z and, when enabled, g_to_expert.

        z: (B, n_latent, W) where W == action-expert width. The z block is prepended; action_out
        is later sliced as suffix_out[:, -H:] so the z/g tokens never reach action_out_proj.
        g: (B, N_g, llm_dim) RAW tactile encoder tokens (from _build_g) or None. Injected ONLY
        when config.g_to_expert (else the kwarg is ignored — default None keeps the signature
        change behavior-neutral): projected by g_proj and muted by the zero-init g_gate, appended
        to the z condition block (att all-0 -> same attention block; z's first token stays the
        block start). g_mask (B, N_g) bool: False tokens (placeholder views) get pad_masks=False
        so the expert never attends them; None -> all attended.
        """
        a_embs, a_pad, a_att, adarms_cond = self.embed_suffix(state, noisy_actions, timestep)
        B, M = z.shape[0], z.shape[1]
        device = a_embs.device
        z = z.to(a_embs.dtype)
        # (_forward_predictor training + _denoise_step_with_z sampling), so gating here covers both.
        # At init 0 -> z is fully muted -> embs == the base-policy suffix (bitwise). dtype-aligned
        # to z_w. Absent (gate off) -> no-op, byte-unchanged.
        gate = getattr(self, "z_gate", None)
        if gate is not None:
            z = z * gate.to(z.dtype)
        z_pad = torch.ones(B, M, dtype=a_pad.dtype, device=device)
        # z is a new attention block after the prefix: first token 1 (block start), rest 0.
        z_att = torch.zeros(B, M, dtype=a_att.dtype, device=device)
        z_att[:, 0] = 1
        cond_embs, cond_pad, cond_att = [z], [z_pad], [z_att]
        # extra condition tokens the trainable predictor cannot drift. Same single choke point as the
        # z gate, so training (_forward_predictor) and sampling (_denoise_step_with_z) can't diverge.
        if g is not None and getattr(self, "g_to_expert", False):
            g_w = self.g_proj(g.to(self.g_proj.weight.dtype)).to(a_embs.dtype)  # llm_dim -> W
            g_w = g_w * self.g_gate.to(g_w.dtype)
            n_g = g_w.shape[1]
            if g_mask is not None:
                g_pad = g_mask.to(device=device, dtype=a_pad.dtype)   # placeholder views -> False
            else:
                g_pad = torch.ones(B, n_g, dtype=a_pad.dtype, device=device)
            g_att = torch.zeros(B, n_g, dtype=a_att.dtype, device=device)  # joins the z block
            cond_embs.append(g_w)
            cond_pad.append(g_pad)
            cond_att.append(g_att)
        embs = torch.cat([*cond_embs, a_embs], dim=1)
        pad_masks = torch.cat([*cond_pad, a_pad], dim=1)
        att_masks = torch.cat([*cond_att, a_att], dim=1)
        return embs, pad_masks, att_masks, adarms_cond

    def _prefix_forward(self, images, img_masks, lang_tokens, lang_masks, *, use_cache: bool):
        """Clean RGB+lang prefix forward -> (vl_ctx, prefix_pad_masks, past_key_values).

        v0: an explicit, correct second prefix forward (no forward_prefix_cache). Phase-2
        perf optimization: cache the prefix K/V so Phase 3 skips the re-encode.
        """
        # tactile_vlm_tokens omitted -> clean prefix, NO tactile in the VLM (base embed_prefix).
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_4d = self._prepare_attention_masks_4d(prefix_att_2d)
        # sdpa requires the attention bias dtype to match the query dtype. The prefix-only
        # forward ([prefix_embs, None]) drives query dtype from prefix_embs (bf16 when the
        # model is bf16); cast the 4D mask to match, else "invalid dtype for bias".
        prefix_4d = prefix_4d.to(dtype=prefix_embs.dtype)
        if use_cache:
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (vl_ctx, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
        )
        return vl_ctx, prefix_embs, prefix_pad_masks, prefix_att_masks, past_key_values

    def _compute_z(self, vl_ctx, prefix_pad_masks=None):
        """z (B, n_latent, llm_dim) from the current observation's tactile + vl_ctx.

        NB: z is returned in llm_dim (to match z* for the Phase-2 cosine loss). Callers apply
        ``self.z_proj`` to project it to the action-expert width before injecting it as suffix
        tokens; do NOT drop that projection.

        ``prefix_pad_masks`` (B, N_ctx) bool marks valid (non-padding) prefix tokens. It is
        ALWAYS passed through to the predictor, but only arch C ("tactile_kv") consumes it (masked
        VL pool, the S5 fix); arch A ("joint_kv") deliberately ignores it — A is the running
        baseline and its ctx_mask=ones behavior is frozen (see TactileActionPredictor.forward).

        Returns (z, g, has_tac); g and has_tac come straight from _build_g.
        """
        tac_t = self._last_tac_t or {}
        tac_0 = self._last_tac_0 or {}
        g, g_mask, has_tac = self._build_g(tac_t, tac_0)
        g_f = g.to(torch.float32) if g is not None else None
        z = self.tactile_predictor(
            vl_ctx.to(torch.float32), g_f, g_mask=g_mask, vl_ctx_mask=prefix_pad_masks
        )   # (B, n_latent, llm_dim)
        return z, g, has_tac

    # ------------------------------------------------------------------
    # Stage-1 predictor-grounding pretraining (paper Sec 4.2; see
    # docs/MID_TRAIN.md)
    # ------------------------------------------------------------------
    def _build_future_target(self, tac_f: dict, tac_t: dict, tac_mask_f: dict | None):
        """z* (Eq. 2) + Dbar (Eq. 5 target), both from the SAME (tac_{t+H} - tac_t) diff.

        Mirrors ``_build_g``'s per-view masking, but AVERAGES the per-view encodings instead of
        concatenating them -- the paper defines z* as the mean over active views of
        f_enc(tac_{t+H}^k - tac_t^k), not a per-view token layout (there is no attention over
        views for the target side, unlike g's role as the predictor's key/value).

        Target stop-gradient is an implementation assumption, not specified in the paper.
        The shared projection updates through the current branch, so these targets still
        change between optimizer steps; this is not a fixed or EMA target encoder.

        Returns (z_star | None, dbar_field | None, has_future(B,) bool). ``dbar_field`` is the
        (B, C, H, W) masked-mean pixel-space diff (BEFORE the grid downsample and the recon
        head's grid choice are applied by the caller), so the caller decides the target
        resolution. Rows with no valid future view (has_future=False) are still shaped
        correctly (zeros) -- the caller must exclude them via the mask, not skip them, so batch
        shapes stay uniform across a mixed-embodiment / partially-clamped batch.
        """
        keys = [k for k in tac_f if k in tac_t]
        if not keys:
            any_img = next(iter(tac_f.values()), None)
            if any_img is None:
                return None, None, None
            B = any_img.shape[0]
            return None, None, torch.zeros(B, dtype=torch.bool, device=any_img.device)

        view_masks = tac_mask_f or {}
        with torch.no_grad():
            enc_list, field_list, valid_list = [], [], []
            for k in keys:
                diff = tac_f[k].float() - tac_t[k].float()          # (B, C, H, W)
                enc_list.append(self.tactile_encoder(diff))          # (B, n_tokens, D)
                field_list.append(diff)
                rk = view_masks.get(k)
                if rk is None:
                    rk = torch.ones(diff.shape[0], dtype=torch.bool, device=diff.device)
                else:
                    rk = rk.to(device=diff.device, dtype=torch.bool).reshape(-1)
                current_mask = (getattr(self, "_last_tac_mask", None) or {}).get(k)
                if current_mask is not None:
                    rk = rk & current_mask.to(device=diff.device, dtype=torch.bool).reshape(-1)
                valid_list.append(rk)

            valid = torch.stack(valid_list, dim=0)                      # (V, B)
            has_future = valid.any(dim=0)                               # (B,)
            w = valid.to(torch.float32)
            w = w / w.sum(dim=0, keepdim=True).clamp(min=1.0)           # (V, B), masked mean weights

            enc_stack = torch.stack(enc_list, dim=0)                    # (V, B, n_tokens, D)
            z_star = (enc_stack * w[:, :, None, None]).sum(dim=0)       # (B, n_tokens, D)

            field_stack = torch.stack(field_list, dim=0)                # (V, B, C, H, W)
            dbar_field = (field_stack * w[:, :, None, None, None]).sum(dim=0)  # (B, C, H, W)

        return z_star.detach(), dbar_field.detach(), has_future

    def _stage1_infonce_loss(self, z: torch.Tensor, z_star: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE (paper Eq. 3-4): h(.) = mean-pool over tokens + L2-normalize,
        applied independently to z (n_latent tokens) and z* (encoder's n_tokens) -- the two
        sides need not share a token count since both collapse to one vector per sample before
        the cosine-similarity matrix. ``valid`` rows lack a real future-tactile target (e.g. a
        tail-clamped episode end or a platform missing every tactile view that step) and are
        DROPPED from the pool rather than zero-filled, so they cannot supply a degenerate
        (batch-constant) target as a negative -- see canonical_tactile_policy.py's clamp-mask
        comment, which this loss is the consumer of.
        """
        import torch.distributed as dist
        from torch.distributed.nn.functional import all_gather

        # Gather before masking so ranks with different valid counts have equal shapes.
        # Autograd-aware gathering plus DDP averaging gives the global-batch gradient.
        z = z.float().mean(dim=1)
        z_star = z_star.detach().float().mean(dim=1)
        if dist.is_initialized():
            z = torch.cat(all_gather(z), dim=0)
            targets = [torch.empty_like(z_star) for _ in range(dist.get_world_size())]
            masks = [torch.empty_like(valid) for _ in range(dist.get_world_size())]
            dist.all_gather(targets, z_star.contiguous())
            dist.all_gather(masks, valid.contiguous())
            z_star, valid = torch.cat(targets), torch.cat(masks)
        n_valid = int(valid.sum())
        if n_valid < 2:
            # Not enough real targets this batch to form a contrastive pool (>=2 needed for a
            # meaningful softmax over negatives); contribute no gradient rather than a
            # misleading number.
            return z.sum() * 0.0
        z = z[valid]
        z_star = z_star[valid]
        hz = F.normalize(z, dim=-1)
        hzs = F.normalize(z_star, dim=-1)
        temp = float(getattr(self.config, "stage1_temperature", 1.0))
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError("stage1_temperature must be finite and positive")
        logits = (hz @ hzs.t()) / temp                                    # (B', B')
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)

    def forward_stage1(self, observation) -> torch.Tensor:
        """Public entry point for scripts/train_stage1_predictor.py. See _forward_stage1."""
        return self._forward_stage1(observation)

    def _forward_stage1(self, observation) -> torch.Tensor:
        """Stage-1 forward (paper Sec 4.2): L_1 = L_NCE + lambda_rec * L_rec.

        Action-free by construction: takes NO ``actions`` argument, never builds the action
        suffix, and never touches ``action_in_proj``/the action expert/``action_out_proj``.
        Requires ``config.stage1_pretrain_enabled`` and a data pipeline that loads a ``.future``
        tactile frame (``future_frame_offset>0`` on the DataConfig; see
        LeRobotCanonicalTaskTactileDataConfig).

        The base policy is frozen for this stage per the paper ("With the entire base policy
        frozen, we train only the predictor, the tactile projection, and a lightweight
        reconstruction head") -- enforced by the CALLER (the training script only unfreezes
        ``tactile_encoder.tactile_proj`` / ``tactile_predictor`` / ``tactile_recon_head``), but
        the prefix forward is ALSO wrapped in ``torch.no_grad()`` here so the ~3B-param VLM
        backbone never builds an autograd graph for this loss regardless of what the caller
        freezes.
        """
        assert self.stage1_pretrain_enabled, "forward_stage1 requires config.stage1_pretrain_enabled=True"
        images, img_masks, lang_tokens, lang_masks, _state, _expert_images = self._preprocess_observation(
            observation, train=True
        )
        with torch.no_grad():
            vl_ctx, _prefix_embs, prefix_pad_masks, _prefix_att_masks, _pkv = self._prefix_forward(
                images, img_masks, lang_tokens, lang_masks, use_cache=False
            )
        vl_ctx = vl_ctx.detach()

        z, _g, has_tac = self._compute_z(vl_ctx, prefix_pad_masks)

        tac_f = self._last_tac_f or {}
        tac_t = self._last_tac_t or {}
        tac_mask_f = getattr(self, "_last_tac_mask_f", None)
        z_star, dbar_field, has_future = self._build_future_target(tac_f, tac_t, tac_mask_f)
        if z_star is None:
            raise RuntimeError(
                "forward_stage1 requires a '.future' tactile frame in the observation; set "
                "future_frame_offset>0 on the DataConfig (see LeRobotCanonicalTaskTactileDataConfig)."
            )

        valid = has_tac & has_future
        nce_loss = self._stage1_infonce_loss(z, z_star, valid)

        grid = self.tactile_recon_head.grid
        gray = dbar_field.mean(dim=1, keepdim=True)                              # (B, 1, H, W)
        dbar = F.adaptive_avg_pool2d(gray, grid).squeeze(1)    # (B, grid, grid)
        pred_field = self.tactile_recon_head(z)                                  # (B, grid, grid)
        import torch.distributed as dist

        valid_count = valid.sum().detach()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if dist.is_initialized():
            dist.all_reduce(valid_count)
        # Sum valid rows, then normalize by the GLOBAL count, including empty ranks.
        # Empty indexing retains the graph without incorporating invalid target pixels.
        recon_sum = (pred_field[valid].float() - dbar[valid]).abs().sum()
        recon_loss = recon_sum * world_size / (valid_count.clamp(min=1) * grid * grid)

        lam = float(getattr(self.config, "stage1_lambda_rec", 0.5))
        total = nce_loss + lam * recon_loss
        self._last_loss_parts = {
            "stage1_nce": float(nce_loss.detach()),
            "stage1_recon": float(recon_loss.detach()),
            "stage1_total": float(total.detach()),
            "stage1_valid_frac": float(valid.float().mean().detach()),
            "stage1_valid_count": int(valid_count),
        }
        return total

    def _vl_dropout_keep(self, batch_size: int, device) -> torch.Tensor | None:
        """Per-sample keep mask for VL-dropout. True => the sample KEEPS suffix→prefix attention;
        False => its suffix (action + z tokens) is cut off from all prefix key/value states.

        Returns None (no-op) unless TRAINING and vl_dropout_prob>0 — so eval/serve never drops.
        CONSUMES the model RNG (torch.rand on device): for a bitwise-reproducible run set
        vl_dropout_prob=0. Only the attention mask is touched by the caller; position ids and z's
        own computation are untouched (z still sees the full vl_ctx — architecture C conditioning).
        """
        # Phase-A curriculum override (train_pytorch sets _vl_dropout_override per step):
        # 1.0 during phase A (ALL samples cut from prefix — task info reaches the expert only
        # through z, maximal z↔expert alignment pressure), ramped back to the config value
        # after the phase boundary, None = no override (config value, default behavior).
        ovr = getattr(self, "_vl_dropout_override", None)
        p = float(ovr) if ovr is not None else float(getattr(self.config, "vl_dropout_prob", 0.0))
        if not self.training or p <= 0.0:
            return None
        return torch.rand(batch_size, device=device) >= p   # keep w.p. (1 - p), per sample

    # ------------------------------------------------------------------
    # forward (gated)
    # ------------------------------------------------------------------
    def forward(self, observation, actions, noise=None, time=None):
        """Training forward. Gate OFF -> base PI0Pytorch.forward (byte-identical, the iron-rule
        path). Gate ON -> the tactile-predictor forward: ``predictor_pretrain`` runs the stage-1
        supervised predictor objective (no action loss); ``e2e`` runs the v0 action forward."""
        if not self.tactile_predictor_enabled:
            return super().forward(observation, actions, noise=noise, time=time)
        return self._forward_predictor(observation, actions, noise=noise, time=time)

    def _forward_predictor(self, observation, actions, noise=None, time=None):
        """Three-phase predictor forward (v0 unsupervised). Returns unreduced action MSE (B,H,D);
        the trainer reduces via ``.mean()``. The Predictor is trained PURELY by the action gradient
        flowing back through z (no z*, no predictor loss). Mirrors the reference implementation
        minus the predictor-loss / future-target machinery.
        """
        images, img_masks, lang_tokens, lang_masks, state, _expert_images = self._preprocess_observation(
            observation, train=True
        )

        cached_k = cached_v = None
        if self.use_prefix_cache:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(prefix_embs.dtype)
            vl_ctx, cached_k, cached_v = self.paligemma_with_expert.forward_prefix_cache(
                prefix_embs, prefix_4d, prefix_pos
            )
        else:
            vl_ctx, prefix_embs, prefix_pad_masks, prefix_att_masks, _prefix_kv = self._prefix_forward(
                images, img_masks, lang_tokens, lang_masks, use_cache=False
            )

        # Phase 2: g, z  (no z*, no predictor loss in v0). prefix_pad_masks feeds arch C's masked
        # VL pool only; arch A ignores it (frozen baseline).
        # vl_ctx DETACHED into the predictor: the predictor has
        # no anchor loss in e2e, so letting its action-grad flow back into the whole VLM adds a
        # predictor<->VLM feedback loop the stable 1.0 run never had (e2e56 grad spiral at peak lr,
        # gradient via the expert's prefix-KV attention. Side benefit: no predictor->VLM backward.
        # g is kept for the g→expert anchor (its pad mask was stashed by _build_g inside
        # _compute_z); both are ignored downstream unless config.g_to_expert.
        z, g, _has_tac = self._compute_z(vl_ctx.detach(), prefix_pad_masks)

        # Phase 3: action expert over [z ; state ; noisy_actions].
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        z_w = self.z_proj(z).to(self.action_in_proj.weight.dtype)     # llm_dim -> action-expert width
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self._embed_suffix_with_z(
            state, x_t, time, z_w, g=g, g_mask=getattr(self, "_last_g_mask", None)
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        if self.use_prefix_cache:
            suffix_len = suffix_pad.shape[1]
            bsz = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, suffix_len, prefix_len)
            # VL-dropout: zero the whole prefix column-block for dropped rows so their suffix
            # (z + action tokens) attends NO prefix KV. `& keep` yields a fresh contiguous mask
            # (prefix_pad_2d is an expand view). suffix_att_2d (suffix→suffix) is untouched.
            vl_keep = self._vl_dropout_keep(bsz, prefix_pad_masks.device)
            if vl_keep is not None:
                prefix_pad_2d = prefix_pad_2d & vl_keep[:, None, None]
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d).to(suffix_embs.dtype)
            suffix_out_full = self.paligemma_with_expert.forward_suffix_cached(
                suffix_embs, full_att_4d, position_ids, cached_k, cached_v, adarms_cond
            )
            suffix_out = suffix_out_full[:, -self.config.action_horizon :].to(torch.float32)
        else:
            # Phase 3 (DEFAULT, UNCACHED): single joint forward over [prefix ; suffix]
            # (recompute-prefix; the 2x prefix forward is the v0 cost the cache removes). This is
            # the EXACT current path / safe fallback -- byte-unchanged.
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att], dim=1)
            full_att_2d = make_att_2d_masks(pad_masks, att_masks)
            # VL-dropout: for dropped rows, cut the suffix→prefix block (suffix ROWS = last
            # suffix_len; prefix COLS = first prefix_len). prefix↔prefix and suffix↔suffix stay
            # intact; position_ids below are DELIBERATELY unchanged (positions kept, only attn
            # masked). full_att_2d is fresh from make_att_2d_masks so the in-place edit is safe.
            bsz = pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            vl_keep = self._vl_dropout_keep(bsz, pad_masks.device)
            if vl_keep is not None:
                full_att_2d[~vl_keep, prefix_len:, :prefix_len] = False
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d)
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(torch.float32)

        v_t = self.action_out_proj(suffix_out)

        action_mse = F.mse_loss(u_t, v_t, reduction="none")          # (B, H, D)  — base convention
        self._last_loss_parts = {"act": float(action_mse.mean().detach())}
        return action_mse


    # ------------------------------------------------------------------
    # sample_actions (gated)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        """Inference sampler. Gate OFF -> base PI0Pytorch.sample_actions (byte-identical, the
        iron-rule path). Gate ON -> the predictor sampler (_sample_actions_predictor)."""
        if not self.tactile_predictor_enabled:
            return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        return self._sample_actions_predictor(device, observation, noise=noise, num_steps=num_steps)

    @torch.no_grad()
    def _sample_actions_predictor(self, device, observation, noise=None, num_steps=10):
        """Predictor-mode sampling. z is predicted from the CURRENT observation (no future tactile
        at inference; no z* / loss).

        NB: inference is ALREADY prefix-KV-cached and is INDEPENDENT of ``use_prefix_cache``: the
        single ``_prefix_forward(use_cache=True)`` below yields BOTH vl_ctx (-> z) AND the prefix
        K/V that every denoise step reuses (mirrors base sample_actions), so there is no redundant
        prefix encode to remove. The flag only gates the TRAINING forward (_forward_predictor). This
        method is byte-unchanged; verify_prefix_cache runs it off-vs-on to confirm the flag never
        perturbs inference.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((bsize, self.config.action_horizon, self.config.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state, _expert_images = self._preprocess_observation(
            observation, train=False
        )
        vl_ctx, _prefix_embs, prefix_pad_masks, _prefix_att_masks, past_key_values = self._prefix_forward(
            images, img_masks, lang_tokens, lang_masks, use_cache=True
        )
        z, g, _has_tac = self._compute_z(vl_ctx, prefix_pad_masks)
        # g→expert anchor: the SAME g/g_mask injection as the training forward (train/infer
        # consistency); inert unless config.g_to_expert (see _embed_suffix_with_z).
        g_mask = getattr(self, "_last_g_mask", None)
        z = self.z_proj(z).to(self.action_in_proj.weight.dtype)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self._denoise_step_with_z(
                state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize), z, g=g, g_mask=g_mask
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def _suffix_forward_cached(
        self, prefix_pad_masks, past_key_values, suffix_embs, suffix_pad, suffix_att, adarms_cond
    ):
        """Forward ONLY the suffix against a cached prefix K/V. Returns the raw suffix hidden
        states (B, action_horizon, W) in float32, BEFORE action_out_proj.

        The mask / position setup is copied VERBATIM from base ``PI0Pytorch.denoise_step``
        (pi0_pytorch.py:514-538) -- the proven-correct cached suffix step:
          * ``prefix_pad_2d`` (pi0_pytorch.py:518): every suffix row attends to ALL valid prefix
            columns -> full attention over the cached prefix.
          * ``suffix_att_2d`` (pi0_pytorch.py:520): block/causal attention WITHIN the suffix.
          * ``position_ids`` (pi0_pytorch.py:524-525): suffix positions are OFFSET BY THE PREFIX
            LENGTH (sum(prefix_pad) + cumsum(suffix_pad) - 1) so the suffix rotary positions match
            the joint [prefix ; suffix] layout.
          * expert attn forced to eager (pi0_pytorch.py:529), matching the joint path's eager expert.
        Shared by the training cached forward (_forward_predictor, use_prefix_cache=True) and by
        _denoise_step_with_z so the two cached paths can never diverge.
        """
        suffix_len = suffix_pad.shape[1]
        B = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(B, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
        full_att_4d = self._prepare_attention_masks_4d(full_att_2d)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return outputs_embeds[1][:, -self.config.action_horizon :].to(torch.float32)

    def _denoise_step_with_z(self, state, prefix_pad_masks, past_key_values, x_t, timestep, z, g=None, g_mask=None):
        """One denoising step with the z block (and, when config.g_to_expert, the g anchor block)
        prepended. The cached suffix forward is
        shared with the training cached path via _suffix_forward_cached (identical mask/position
        math -> the two never diverge).
        """
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self._embed_suffix_with_z(
            state, x_t, timestep, z, g=g, g_mask=g_mask
        )
        suffix_out = self._suffix_forward_cached(
            prefix_pad_masks, past_key_values, suffix_embs, suffix_pad, suffix_att, adarms_cond
        )
        return self.action_out_proj(suffix_out)
