#!/usr/bin/env python
"""GATE-C: prove N0VTLAPolicy(tactile_predictor_enabled=False) == base PI0Pytorch.

Hard CI requirement: with the tactile predictor GATED OFF, the subclass
``N0VTLAPolicy`` must be BYTE-IDENTICAL to the clean upstream ``PI0Pytorch``:

  (A) identical state_dict keys                (no extra / missing params or buffers)
  (B) base weights load into the subclass      (strict=True, zero missing/unexpected)
  (C) identical forward/flow-matching loss      (torch.equal, exact)
  (D) identical sampled actions                 (torch.equal, exact)

The comparison runs on the REAL training code path: ``model_cfg`` is built exactly
the way ``scripts/train_pytorch.py`` builds it (mirrored line refs inline), the same
internal bf16 precision cast is applied (via ``config.dtype`` -> gemma_pytorch), and
the same ``.to(device)`` move is used.

Run (single GPU, on a CUDA-capable GPU):
    python scripts/gate_c_check.py

Prints each sub-check, then a final unambiguous ``GATE_C PASS`` or ``GATE_C FAIL: ...``.
Exit code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import traceback

# --- make `import n0vtla` / `import scripts` work when run as `python scripts/gate_c_check.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import n0vtla.training.config as _config  # noqa: E402
from n0vtla.models.model import Observation  # noqa: E402
from n0vtla.models.pi0_config import Pi0Config  # noqa: E402
from n0vtla.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy  # noqa: E402
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAConfig  # noqa: E402
from n0vtla.shared import array_typing as at  # noqa: E402

# Config used to source the *exact* reference hyper-params (config.py:1788-1815).
BASE_CONFIG_NAME = "flexiv_vision_reference"

# Deterministic seeds. SEED is re-applied right before every forward/sample so that the
# global (CPU + CUDA) RNG state is identical for `base` and `off` — see note below on why
# this matters even for a "deterministic" comparison.
SEED = 0
OBS_SEED = 1234       # fixes the synthetic Observation
TENSOR_SEED = 4321    # fixes actions / noise / time
BATCH = 2


# ---------------------------------------------------------------------------
# Config construction — mirrors scripts/train_pytorch.py:437-458
# ---------------------------------------------------------------------------
def build_model_cfg() -> tuple[Pi0Config, str]:
    """Build the base ``model_cfg`` exactly as train_pytorch does.

    train_pytorch.py:437-458: ``config.model`` for ``flexiv_vision_reference`` IS a
    plain ``Pi0Config`` (config.py:1789), so the ELSE branch (train_pytorch.py:455-458) runs:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    We reproduce that with ``dataclasses.replace`` (fresh instance, don't mutate the registry
    object) so ``dtype`` == ``pytorch_training_precision`` ("bfloat16", config.py:1311).
    """
    train_cfg = _config.get_config(BASE_CONFIG_NAME)          # a TrainConfig
    src_model = train_cfg.model                               # a plain Pi0Config (reference hparams)
    assert isinstance(src_model, Pi0Config), f"expected Pi0Config, got {type(src_model)}"
    precision = train_cfg.pytorch_training_precision          # "bfloat16"
    model_cfg = dataclasses.replace(src_model, dtype=precision)
    return model_cfg, precision


def build_predictor_cfg_off(model_cfg: Pi0Config) -> N0VTLAConfig:
    """A ``N0VTLAConfig`` with base fields IDENTICAL to ``model_cfg`` but the predictor GATED OFF.

    Copies every ``Pi0Config`` field verbatim (dtype, pi05, action_dim/horizon, variants,
    max_token_len, discrete_state_input, compile_mode, tactile knobs) and only adds
    ``tactile_predictor_enabled=False`` (n0vtla_policy.py:58). The other predictor-only fields
    (n_latent, ...) take defaults and are INERT while gated off (n0vtla_policy.py:103-105).
    """
    base_fields = {f.name: getattr(model_cfg, f.name) for f in dataclasses.fields(Pi0Config)}
    return N0VTLAConfig(**base_fields, tactile_predictor_enabled=False)


# ---------------------------------------------------------------------------
# Synthetic, fixed inputs
# ---------------------------------------------------------------------------
def make_observation(device: torch.device, model_cfg: Pi0Config) -> Observation:
    """One fixed synthetic Observation with NO tactile keys (the tactile-off path).

    Layout mirrors what the real data pipeline hands the model (model.py:127-149): RGB images
    are float32, channels-first NCHW [B, 3, 224, 224], normalized to [-1, 1]. The model's
    preprocessing auto-detects channels-first (preprocessing_pytorch.py:30) and returns NCHW,
    which SigLIP's ``get_image_features`` expects. State is [B, action_dim]; for pi05 its value
    is numerically inert (state is folded into the discrete language tokens, so embed_suffix
    skips ``state_proj`` — pi0_pytorch.py:309-327), but the tensor is still required for shape.
    """
    g = torch.Generator(device="cpu").manual_seed(OBS_SEED)
    b = BATCH
    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        img = torch.rand((b, 3, 224, 224), generator=g, dtype=torch.float32) * 2.0 - 1.0  # [-1, 1]
        images[key] = img.to(device)
        image_masks[key] = torch.ones(b, dtype=torch.bool, device=device)

    state = (torch.rand((b, model_cfg.action_dim), generator=g, dtype=torch.float32) * 2.0 - 1.0).to(device)
    # Valid token ids in [0, vocab). vocab = 257152 (gemma_pytorch.py:24). Use long for nn.Embedding.
    tokenized_prompt = torch.randint(0, 257152, (b, model_cfg.max_token_len), generator=g, dtype=torch.long).to(device)
    tokenized_prompt_mask = torch.ones((b, model_cfg.max_token_len), dtype=torch.bool, device=device)

    # Construct directly; disable jaxtyping so the torch-tensor NCHW layout is never rejected.
    with at.disable_typechecking():
        obs = Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
    return obs


def make_flow_tensors(device: torch.device, model_cfg: Pi0Config):
    """Fixed actions / noise / time / sampling-noise (all float32)."""
    g = torch.Generator(device="cpu").manual_seed(TENSOR_SEED)
    b, h, d = BATCH, model_cfg.action_horizon, model_cfg.action_dim
    actions = torch.randn((b, h, d), generator=g, dtype=torch.float32).to(device)
    noise = torch.randn((b, h, d), generator=g, dtype=torch.float32).to(device)
    time = (torch.rand((b,), generator=g, dtype=torch.float32) * 0.998 + 0.001).to(device)  # (0.001, 0.999)
    sample_noise = torch.randn((b, h, d), generator=g, dtype=torch.float32).to(device)
    return actions, noise, time, sample_noise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("GATE-C: N0VTLAPolicy(tactile_predictor_enabled=False)  ==  base PI0Pytorch")
    print("=" * 78)

    # Determinism knobs (cheap; applied identically to both models). PI0Pytorch.__init__ also
    # sets torch.set_float32_matmul_precision('high') globally (pi0_pytorch.py:132).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found. The models cast paligemma to bfloat16 internally; "
              "bf16 kernels are slow/limited on CPU. Run this on the CUDA-capable GPU for a valid gate.")

    # -- Config (mirror train_pytorch.py:437-458) --------------------------------------------
    model_cfg, precision = build_model_cfg()
    predictor_cfg = build_predictor_cfg_off(model_cfg)
    print(f"[cfg] source config      : {BASE_CONFIG_NAME}")
    print(f"[cfg] precision (dtype)  : {precision}")
    print(f"[cfg] pi05               : {model_cfg.pi05}")
    print(f"[cfg] action_dim         : {model_cfg.action_dim}")
    print(f"[cfg] action_horizon     : {model_cfg.action_horizon}")
    print(f"[cfg] max_token_len      : {model_cfg.max_token_len}")
    print(f"[cfg] paligemma_variant  : {model_cfg.paligemma_variant}")
    print(f"[cfg] action_expert      : {model_cfg.action_expert_variant}")
    print(f"[cfg] compile_mode       : {model_cfg.pytorch_compile_mode}")
    print(f"[cfg] use_tactile        : {model_cfg.use_tactile}")
    print(f"[cfg] predictor gated OFF    : tactile_predictor_enabled={predictor_cfg.tactile_predictor_enabled}")
    print(f"[cfg] device             : {device}")

    # -- Build both models (mirror train_pytorch.py:461-463: build, then .to(device)) ---------
    # NOTE: train_pytorch does NOT apply any top-level `.to(bfloat16)`. The bf16 precision is
    # applied INTERNALLY by PaliGemmaWithExpertModel(precision=config.dtype) which casts only
    # paligemma_with_expert to bf16 (keeping select layernorms/embeddings fp32) — gemma_pytorch.py:60-82.
    # The top-level heads (action_in_proj / action_out_proj / time_mlp_*) stay fp32, so the
    # returned loss and actions are fp32. We mirror exactly: build, then `.to(device)`.
    torch.manual_seed(SEED)
    base = PI0Pytorch(model_cfg).to(device)
    torch.manual_seed(SEED)
    off = N0VTLAPolicy(predictor_cfg).to(device)

    # Same mode for both (the load-bearing requirement is an IDENTICAL code path). eval() disables
    # dropout (0 here anyway) and the training-only forced-grad-checkpointing branch, giving a
    # clean deterministic forward. NB: PI0Pytorch.forward hardcodes _preprocess_observation(train=True)
    # (pi0_pytorch.py:384), so image augmentation (preprocessing_pytorch.py:38-95) runs regardless of
    # eval() and consumes the global RNG — which is exactly why we re-seed before each forward below.
    base.eval()
    off.eval()

    ok_keys = ok_load = ok_loss = ok_actions = False
    fail_reason = ""

    try:
        # -- (A) state_dict key parity -------------------------------------------------------
        base_sd = base.state_dict()
        off_sd = off.state_dict()
        base_keys, off_keys = set(base_sd.keys()), set(off_sd.keys())
        missing_in_off = base_keys - off_keys
        extra_in_off = off_keys - base_keys
        ok_keys = (not missing_in_off) and (not extra_in_off)
        print("\n--- (A) state_dict key parity ---")
        print(f"    base keys: {len(base_keys)} | off keys: {len(off_keys)}")
        if ok_keys:
            print("    PASS: identical key sets (no extra/missing).")
        else:
            print(f"    FAIL: missing_in_off={sorted(missing_in_off)[:10]} "
                  f"(+{max(0, len(missing_in_off) - 10)} more)")
            print(f"          extra_in_off={sorted(extra_in_off)[:10]} "
                  f"(+{max(0, len(extra_in_off) - 10)} more)")
            fail_reason = fail_reason or "state_dict keys differ"

        # -- (B) load identical base weights into the subclass -------------------------------
        print("\n--- (B) load base weights into N0VTLAPolicy (strict=True) ---")
        incompat = off.load_state_dict(base.state_dict(), strict=True)
        n_missing = len(getattr(incompat, "missing_keys", []))
        n_unexpected = len(getattr(incompat, "unexpected_keys", []))
        ok_load = (n_missing == 0) and (n_unexpected == 0)
        print(f"    missing={n_missing}  unexpected={n_unexpected}")
        print("    PASS: base weights load with zero missing/unexpected." if ok_load
              else "    FAIL: non-empty missing/unexpected on strict load.")
        if not ok_load:
            fail_reason = fail_reason or "strict load_state_dict incomplete"

        # -- fixed inputs --------------------------------------------------------------------
        obs = make_observation(device, model_cfg)
        actions, noise, time, sample_noise = make_flow_tensors(device, model_cfg)

        # -- (C) forward / flow-matching loss equality ---------------------------------------
        # Re-seed to the SAME state before each forward so the (hardcoded train=True) image
        # augmentation draws identical random numbers for base and off. Identical weights +
        # identical inputs + identical RNG + identical code path => bit-identical loss.
        # no_grad: forward() is written for training and would otherwise retain an autograd
        # graph for EACH of the two full models (memory blow-up / OOM risk). GATE-C needs
        # no gradients; no_grad leaves the forward math bit-identical while freeing activations.
        # It does not alter any branch (the grad-checkpointing paths key off self.training=False).
        print("\n--- (C) forward loss equality (torch.equal) ---")
        with torch.no_grad():
            torch.manual_seed(SEED)
            loss_base = base.forward(obs, actions, noise=noise, time=time)
            torch.manual_seed(SEED)
            loss_off = off.forward(obs, actions, noise=noise, time=time)
        ok_loss = bool(torch.equal(loss_base, loss_off))
        loss_delta = (loss_base.to(torch.float64) - loss_off.to(torch.float64)).abs().max().item()
        print(f"    loss shape={tuple(loss_base.shape)} dtype={loss_base.dtype}")
        print(f"    loss_base.mean()={loss_base.float().mean().item():.8e}  "
              f"loss_off.mean()={loss_off.float().mean().item():.8e}")
        print(f"    max|Δ|={loss_delta:.3e}  torch.equal={ok_loss}")
        print("    PASS: losses are exactly equal." if ok_loss else "    FAIL: losses differ.")
        if not ok_loss:
            fail_reason = fail_reason or f"forward loss differs (max|Δ|={loss_delta:.3e})"

        # -- (D) sample_actions equality -----------------------------------------------------
        print("\n--- (D) sample_actions equality (torch.equal) ---")
        torch.manual_seed(SEED)
        a_base = base.sample_actions(device, obs, noise=sample_noise)
        torch.manual_seed(SEED)
        a_off = off.sample_actions(device, obs, noise=sample_noise)
        ok_actions = bool(torch.equal(a_base, a_off))
        act_delta = (a_base.to(torch.float64) - a_off.to(torch.float64)).abs().max().item()
        print(f"    actions shape={tuple(a_base.shape)} dtype={a_base.dtype}")
        print(f"    max|Δ|={act_delta:.3e}  torch.equal={ok_actions}")
        print("    PASS: sampled actions are exactly equal." if ok_actions else "    FAIL: actions differ.")
        if not ok_actions:
            fail_reason = fail_reason or f"sampled actions differ (max|Δ|={act_delta:.3e})"

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        fail_reason = fail_reason or f"exception: {type(exc).__name__}: {exc}"

    # -- verdict ---------------------------------------------------------------------------------
    all_ok = ok_keys and ok_load and ok_loss and ok_actions
    print("\n" + "=" * 78)
    print(f"    (A) state_dict keys identical : {ok_keys}")
    print(f"    (B) strict weight load        : {ok_load}")
    print(f"    (C) forward loss torch.equal  : {ok_loss}")
    print(f"    (D) sampled actions torch.equal: {ok_actions}")
    print("=" * 78)
    if all_ok:
        print("GATE_C PASS")
        return 0
    print(f"GATE_C FAIL: {fail_reason or 'one or more sub-checks failed'}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
