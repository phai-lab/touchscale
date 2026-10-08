#!/usr/bin/env python
"""VERIFY-PREFIX-CACHE: prove N0VTLAPolicy(use_prefix_cache=True) == (use_prefix_cache=False).

The predictor's Phase-3 optimization (n0vtla_policy.py) caches the Phase-1 prefix K/V and
forwards ONLY the suffix against it, instead of re-encoding the whole prefix jointly with the
suffix. Correctness is the hard requirement: the cached path must reproduce the uncached joint
path to bf16-noise level. This is flag-gated (``use_prefix_cache``, OFF by default) and stays off
until THIS script says OK.

What it checks (ONE model, ONE set of weights, ONE fixed input -- only the flag flips):

  (1) TRAINING forward loss:   ``forward`` -> _forward_predictor with the flag OFF (joint recompute-
      prefix) vs ON (cached suffix). This is the REAL test -- two genuinely different computations
      that must agree. Reports torch.equal, max|Δ|, and a relative delta.

  (2) TRAINING gradient:       d(loss)/d(prefix q_proj weight), flag OFF vs ON. The value check
      (1) CANNOT catch a cache-detachment regression (the base policy only uses the cache under no_grad
      at inference); this proves gradients flow through the cached prefix K/V into the paligemma
      prefix weights, matching the joint path.

  (3) INFERENCE sample_actions: flag OFF vs ON. Inference is ALREADY prefix-KV-cached and is
      flag-INDEPENDENT (see _sample_actions_predictor docstring), so this is expected to be exactly
      torch.equal -- it confirms the flag does not perturb the (unchanged) inference path. Reports
      torch.equal, overall max|Δ|, and the per-action-dim ("y-dim") max diff.

Correctness rationale for (1) (the load-bearing detail): the cached suffix step uses the mask /
position setup copied VERBATIM from base ``PI0Pytorch.denoise_step`` (pi0_pytorch.py:514-538) via
``_suffix_forward_cached`` -- suffix attends to the full valid prefix + block/causal within the
suffix, and suffix positions are OFFSET BY THE PREFIX LENGTH. The cached prefix K/V equals the
joint forward's recomputed prefix K/V because the prefix self-attention is causally isolated from
the suffix (prefix att_masks all 0; first z/suffix block-start 1). The residual difference is only
the bf16 kernel-path difference between the joint two-stream forward (compute_layer_complete) and
the cached HF language_model forward -- the SAME difference the base policy already tolerates between
its ``forward`` and ``sample_actions``.

Run (single GPU, on a CUDA-capable GPU -- do NOT run on CPU: bf16 kernels are slow/limited and the tactile
DINOv2 backbone downloads from HF on construction):
    python scripts/verify_prefix_cache.py

Prints per-check numbers then a final ``PREFIX_CACHE OK  ...`` or ``PREFIX_CACHE FAIL  ...``.
Exit code 0 on OK, 1 on FAIL. Tolerance note is printed so a human can adjudicate borderline bf16.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import traceback

# --- make `import n0vtla` / `import scripts` work when run as `python scripts/verify_prefix_cache.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import n0vtla.training.config as _config  # noqa: E402
from n0vtla.models.model import Observation  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAConfig  # noqa: E402
from n0vtla.shared import array_typing as at  # noqa: E402

# The predictor config carries tactile_predictor_enabled=True + the reference predictor hparams (config.py:1823).
BASE_CONFIG_NAME = "flexiv_tactile_reference"

# Deterministic seeds. SEED is re-applied right before every forward/sample so the global RNG
# (CPU + CUDA) is identical for the OFF and ON runs -- the (hardcoded train=True) image
# augmentation in _preprocess_observation consumes the global RNG, so identical RNG + identical
# weights + identical input is required for a clean OFF-vs-ON delta.
SEED = 0
OBS_SEED = 1234       # fixes the synthetic Observation (RGB + tactile current + tactile baseline)
TENSOR_SEED = 4321    # fixes actions / noise / time / sampling noise
BATCH = 2

# Pass/fail tolerance on max|Δ|. fp32 -> essentially exact (two kernel paths); bf16 -> ~1e-2 noise
# on the action-scale outputs is expected and acceptable. A mask/position bug produces a GROSS
# delta (order-1+ / relative >> 5%), which trips the threshold regardless.
TOL_FP32 = 1e-4
TOL_BF16 = 3e-2
REL_TOL = 0.05        # 5% relative -- a coarse gross-error trip independent of absolute scale


# ---------------------------------------------------------------------------
# Config construction -- pull the REAL predictor training config (highest fidelity)
# ---------------------------------------------------------------------------
def build_predictor_cfg() -> tuple[N0VTLAConfig, str]:
    """Build the predictor ``model_cfg`` exactly as the predictor training path does.

    ``get_config(flexiv_tactile_reference).model`` is a ``N0VTLAConfig`` with
    ``tactile_predictor_enabled=True`` (config.py:1824-1836). Mirror train_pytorch's dtype override
    (dtype := pytorch_training_precision) via ``dataclasses.replace`` on a fresh instance.
    ``use_prefix_cache`` stays at its default False on the config; the harness flips the INSTANCE
    flag ``model.use_prefix_cache`` at runtime instead, so OFF and ON share identical weights.
    """
    train_cfg = _config.get_config(BASE_CONFIG_NAME)
    src_model = train_cfg.model
    assert isinstance(src_model, N0VTLAConfig), f"expected N0VTLAConfig, got {type(src_model)}"
    assert src_model.tactile_predictor_enabled, "predictor config must have tactile_predictor_enabled=True"
    precision = train_cfg.pytorch_training_precision          # "bfloat16"
    model_cfg = dataclasses.replace(src_model, dtype=precision)
    return model_cfg, precision


# ---------------------------------------------------------------------------
# Synthetic, fixed inputs (WITH tactile: 2 views + per-view baseline)
# ---------------------------------------------------------------------------
def make_observation(device: torch.device, model_cfg: N0VTLAConfig) -> Observation:
    """One fixed synthetic Observation with RGB + tactile-current + tactile-baseline keys.

    The predictor's ``_preprocess_observation`` POPS the tactile keys (current ``k`` and baseline
    ``k + '.baseline'``) out of ``observation.images`` before the base SigLIP preprocessing, so a
    FRESH observation must be built for every forward (the pop mutates in place). RGB layout mirrors
    the real pipeline (NCHW float32 in [-1, 1], like gate_c_check.py); tactile is provided as
    float32 NCHW so the frozen DINOv2 encoder passes it through untouched (tactile_encoder.py:109-112).
    """
    g = torch.Generator(device="cpu").manual_seed(OBS_SEED)
    b = BATCH
    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}

    # RGB views (go through SigLIP prefix).
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        img = torch.rand((b, 3, 224, 224), generator=g, dtype=torch.float32) * 2.0 - 1.0  # [-1, 1]
        images[key] = img.to(device)
        image_masks[key] = torch.ones(b, dtype=torch.bool, device=device)

    # Tactile views: current frame + baseline (frame 0). Keys come from the config
    # (tactile_image_keys), baseline key = current + N0VTLAPolicy.BASELINE_SUFFIX (".baseline").
    baseline_suffix = ".baseline"
    for key in model_cfg.tactile_image_keys:
        cur = torch.rand((b, 3, 224, 224), generator=g, dtype=torch.float32)
        base = torch.rand((b, 3, 224, 224), generator=g, dtype=torch.float32)
        images[key] = cur.to(device)
        images[key + baseline_suffix] = base.to(device)
        image_masks[key] = torch.ones(b, dtype=torch.bool, device=device)
        image_masks[key + baseline_suffix] = torch.ones(b, dtype=torch.bool, device=device)

    state = (torch.rand((b, model_cfg.action_dim), generator=g, dtype=torch.float32) * 2.0 - 1.0).to(device)
    # Valid token ids in [0, vocab). vocab = 257152. Use long for nn.Embedding.
    tokenized_prompt = torch.randint(0, 257152, (b, model_cfg.max_token_len), generator=g, dtype=torch.long).to(device)
    tokenized_prompt_mask = torch.ones((b, model_cfg.max_token_len), dtype=torch.bool, device=device)

    with at.disable_typechecking():
        obs = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
    return obs


def make_flow_tensors(device: torch.device, model_cfg: N0VTLAConfig):
    """Fixed actions / noise / time / sampling-noise (all float32)."""
    g = torch.Generator(device="cpu").manual_seed(TENSOR_SEED)
    b, h, d = BATCH, model_cfg.action_horizon, model_cfg.action_dim
    actions = torch.randn((b, h, d), generator=g, dtype=torch.float32).to(device)
    noise = torch.randn((b, h, d), generator=g, dtype=torch.float32).to(device)
    time = (torch.rand((b,), generator=g, dtype=torch.float32) * 0.998 + 0.001).to(device)  # (0.001, 0.999)
    sample_noise = torch.randn((b, h, d), generator=g, dtype=torch.float32).to(device)
    return actions, noise, time, sample_noise


def _delta_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float, float]:
    """(torch.equal, max|Δ|, relative max|Δ|) computed in float64."""
    eq = bool(torch.equal(a, b))
    af, bf = a.to(torch.float64), b.to(torch.float64)
    abs_delta = (af - bf).abs().max().item()
    scale = af.abs().max().item()
    rel_delta = abs_delta / scale if scale > 0 else abs_delta
    return eq, abs_delta, rel_delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("VERIFY-PREFIX-CACHE: N0VTLAPolicy use_prefix_cache=True  ==  use_prefix_cache=False")
    print("=" * 78)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found. The model casts paligemma to bfloat16 internally and "
              "the tactile DINOv2 backbone downloads from HF; run this on the CUDA-capable GPU for a valid check.")

    model_cfg, precision = build_predictor_cfg()
    tol = TOL_FP32 if precision == "float32" else TOL_BF16
    print(f"[cfg] source config      : {BASE_CONFIG_NAME}")
    print(f"[cfg] precision (dtype)  : {precision}")
    print(f"[cfg] pi05               : {model_cfg.pi05}")
    print(f"[cfg] action_dim         : {model_cfg.action_dim}")
    print(f"[cfg] action_horizon     : {model_cfg.action_horizon}")
    print(f"[cfg] tactile_predictor      : {model_cfg.tactile_predictor_enabled}  n_latent={model_cfg.n_latent}")
    print(f"[cfg] tactile_image_keys : {model_cfg.tactile_image_keys}")
    print(f"[cfg] compile_mode       : {model_cfg.pytorch_compile_mode}")
    print(f"[cfg] device             : {device}")
    print(f"[cfg] max|Δ| tolerance   : {tol:.1e}  (rel {REL_TOL:.0%})   [fp32 is near-exact; bf16 ~1e-2 expected]")

    ok_loss = ok_grad = ok_actions = False
    fail_reason = ""
    loss_abs = loss_rel = grad_abs = grad_rel = act_abs = act_rel = float("nan")

    try:
        # -- Build ONE model with fixed (seeded) weights. Both OFF and ON runs use THIS instance,
        #    so weights are byte-identical across the comparison; only self.use_prefix_cache flips.
        torch.manual_seed(SEED)
        model = N0VTLAPolicy(model_cfg).to(device)
        model.eval()
        # Force eager attention for the prefix language_model so the OFF-path Phase-1 vl_ctx and the
        # ON-path Phase-1 vl_ctx are produced by the SAME kernel (use_cache=True already forces
        # eager). This isolates the quantity under test -- whether the cached prefix K/V reproduces
        # the joint forward's recomputed prefix K/V -- from any sdpa-vs-eager Phase-1 drift, so z is
        # identical in both runs and the whole delta is attributable to the cache mechanism.
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        actions, noise, time, sample_noise = make_flow_tensors(device, model_cfg)

        # -- (1) TRAINING forward loss: OFF (joint) vs ON (cached suffix) --------------------------
        # Fresh obs per run (the predictor pops tactile keys in place). Re-seed the global RNG right
        # before each forward so the train=True image augmentation draws identical randoms.
        print("\n--- (1) training forward loss: use_prefix_cache OFF (joint) vs ON (cached) ---")
        with torch.no_grad():
            obs_off = make_observation(device, model_cfg)
            model.use_prefix_cache = False
            torch.manual_seed(SEED)
            loss_off = model.forward(obs_off, actions, noise=noise, time=time)

            obs_on = make_observation(device, model_cfg)
            model.use_prefix_cache = True
            torch.manual_seed(SEED)
            loss_on = model.forward(obs_on, actions, noise=noise, time=time)

        ok_loss_eq, loss_abs, loss_rel = _delta_stats(loss_off, loss_on)
        ok_loss = ok_loss_eq or (loss_abs <= tol) or (loss_rel <= REL_TOL)
        print(f"    loss shape={tuple(loss_off.shape)} dtype={loss_off.dtype}")
        print(f"    loss_off.mean()={loss_off.float().mean().item():.8e}  "
              f"loss_on.mean()={loss_on.float().mean().item():.8e}")
        print(f"    torch.equal={ok_loss_eq}  max|Δ|={loss_abs:.3e}  rel={loss_rel:.3e}")
        print("    PASS: cached training loss matches joint within tolerance." if ok_loss
              else "    FAIL: cached training loss diverges beyond tolerance.")
        if not ok_loss:
            fail_reason = fail_reason or f"training loss differs (max|Δ|={loss_abs:.3e}, rel={loss_rel:.3e})"

        # -- (2) TRAINING gradient equality: OFF vs ON --------------------------------------------
        # The forward-VALUE check above cannot catch a cache-detachment regression: the base policy only
        # ever consumes the prefix K/V under no_grad (inference), so grad-through-cache is a NEW
        # property the training cached path relies on. modeling_gemma.py:308-310 does
        # ``torch.cat([past_k, new_k])`` with NO detach, so grad SHOULD flow suffix -> cached prefix
        # K/V -> paligemma prefix weights, matching the joint path. This check proves it: compare
        # d(loss.mean())/d(prefix q_proj.weight) between OFF and ON. If the cache detached, the ON
        # grad would MISS the K/V->prefix contribution (present in OFF's joint forward) -> gross diff.
        # Run in eval() (no gradient checkpointing) -- the ONLY regime where the cache is usable in
        # training anyway (the _forward_predictor guard blocks cache+checkpointing+training).
        print("\n--- (2) training gradient equality: d(loss)/d(prefix q_proj) OFF vs ON ---")
        prefix_w = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight
        z_w_param = model.z_proj.weight

        def _grad_of(use_cache_flag: bool):
            model.use_prefix_cache = use_cache_flag
            model.zero_grad(set_to_none=True)
            obs_g = make_observation(device, model_cfg)
            torch.manual_seed(SEED)
            loss_g = model.forward(obs_g, actions, noise=noise, time=time).float().mean()
            loss_g.backward()
            gp = None if prefix_w.grad is None else prefix_w.grad.detach().clone()
            gz = None if z_w_param.grad is None else z_w_param.grad.detach().clone()
            return gp, gz

        gp_off, gz_off = _grad_of(False)
        gp_on, gz_on = _grad_of(True)
        model.zero_grad(set_to_none=True)

        ok_grad = True
        if gp_off is None or gp_on is None:
            ok_grad = False
            print(f"    FAIL: prefix q_proj grad missing (off={gp_off is not None}, on={gp_on is not None}) "
                  "-- grad did NOT flow through the cached prefix K/V.")
            fail_reason = fail_reason or "prefix grad missing (cache detached the prefix K/V)"
            grad_abs = grad_rel = float("nan")
        else:
            _, grad_abs, grad_rel = _delta_stats(gp_off, gp_on)
            ok_grad = (grad_abs <= tol) or (grad_rel <= REL_TOL)
            _, gz_abs, gz_rel = _delta_stats(gz_off, gz_on)
            print(f"    prefix q_proj grad: max|Δ|={grad_abs:.3e}  rel={grad_rel:.3e}  "
                  f"(||g_off||={gp_off.float().norm().item():.3e})")
            print(f"    z_proj     grad: max|Δ|={gz_abs:.3e}  rel={gz_rel:.3e}")
            print("    PASS: gradients flow through the cache and match the joint path." if ok_grad
                  else "    FAIL: prefix gradients diverge -- grad-through-cache is WRONG.")
            if not ok_grad:
                fail_reason = fail_reason or f"prefix grad differs (max|Δ|={grad_abs:.3e}, rel={grad_rel:.3e})"

        # -- (3) INFERENCE sample_actions: OFF vs ON (flag-independent -> expect exact equal) ------
        print("\n--- (3) sample_actions: use_prefix_cache OFF vs ON (inference is flag-independent) ---")
        obs_s_off = make_observation(device, model_cfg)
        model.use_prefix_cache = False
        torch.manual_seed(SEED)
        a_off = model.sample_actions(device, obs_s_off, noise=sample_noise)

        obs_s_on = make_observation(device, model_cfg)
        model.use_prefix_cache = True
        torch.manual_seed(SEED)
        a_on = model.sample_actions(device, obs_s_on, noise=sample_noise)

        ok_act_eq, act_abs, act_rel = _delta_stats(a_off, a_on)
        ok_actions = ok_act_eq or (act_abs <= tol) or (act_rel <= REL_TOL)
        # per-action-dim ("y-dim") max diff: max over (batch, horizon) for each action channel.
        per_dim = (a_off.to(torch.float64) - a_on.to(torch.float64)).abs().amax(dim=(0, 1))
        print(f"    actions shape={tuple(a_off.shape)} dtype={a_off.dtype}")
        print(f"    torch.equal={ok_act_eq}  max|Δ|={act_abs:.3e}  rel={act_rel:.3e}")
        print(f"    per-action-dim max|Δ|: min={per_dim.min().item():.3e}  "
              f"max={per_dim.max().item():.3e}  mean={per_dim.mean().item():.3e}")
        print("    PASS: sampled actions unchanged by the flag (inference already cached)." if ok_actions
              else "    FAIL: sampled actions changed -- the flag must NOT affect inference.")
        if not ok_actions:
            fail_reason = fail_reason or f"sampled actions differ (max|Δ|={act_abs:.3e}, rel={act_rel:.3e})"

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        fail_reason = fail_reason or f"exception: {type(exc).__name__}: {exc}"

    # -- verdict ---------------------------------------------------------------------------------
    all_ok = ok_loss and ok_grad and ok_actions
    print("\n" + "=" * 78)
    print(f"    (1) training loss  OFF==ON  : {ok_loss}  (max|Δ|={loss_abs:.3e}, rel={loss_rel:.3e})")
    print(f"    (2) training grad  OFF==ON  : {ok_grad}  (max|Δ|={grad_abs:.3e}, rel={grad_rel:.3e})")
    print(f"    (3) sample_actions OFF==ON  : {ok_actions}  (max|Δ|={act_abs:.3e}, rel={act_rel:.3e})")
    print("    NOTE: torch.equal is ideal; under bfloat16 the joint (compute_layer_complete) and")
    print("          cached (HF language_model) kernels differ at ~1e-2 -- the SAME noise the base policy")
    print("          tolerates between forward and sample_actions. If max|Δ| is small (<= tol) but")
    print("          torch.equal is False, that is bf16 noise, NOT a bug. A mask/position error")
    print("          produces a GROSS delta (order-1 / rel >> 5%).")
    print("=" * 78)
    if all_ok:
        print(f"PREFIX_CACHE OK  max|Δ|_loss={loss_abs:.3e}  max|Δ|_act={act_abs:.3e}")
        return 0
    print(f"PREFIX_CACHE FAIL  max|Δ|_loss={loss_abs:.3e}  max|Δ|_act={act_abs:.3e}  reason={fail_reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
