#!/usr/bin/env python
"""Tactile causal probe — does the model's tactile latent z actually READ the tactile input?

WHAT THIS MEASURES
------------------
A tactile action-predictor encodes the tactile signal into a small latent `z` (n_latent tokens)
that is injected into the policy's action expert. A common failure mode we call "VL-router
disease" is that `z` learns to IGNORE the tactile input and instead reconstruct the
tactile-correlated part of the vision+language (VL) context, because the action / InfoNCE loss
can be lowered that way without ever looking at the sensor. The policy then looks fine on paper
(loss goes down) but is effectively blind to touch. This tool detects that.

It loads ONE checkpoint, runs a batch of real observations through the model's OWN
z-computation path (deterministic: train=False, no augmentation), and re-runs it under five
controlled input perturbations. Comparing how much `z` moves under a TACTILE perturbation vs a
VL perturbation tells you whether z is tactile-driven or VL-driven.

THE FIVE z GROUPS (each recomputed on the same batch via _preprocess_observation ->
_prefix_forward -> _compute_z, so every variant walks the identical path):

  z_real     unperturbed input — the reference.
  z_null     tactile current frame := baseline frame, so the tactile diff (tac_t - tac_0) is
             exactly zero. Measures z's response to REMOVING the tactile signal.
  z_shuffle  the whole tactile group (frames + masks) is rolled by one within the batch, so
             sample i gets sample (i+1)'s tactile; VL context untouched. This is the TACTILE
             sensitivity — how much z changes when only the touch changes.
  z_vlswap   the mirror control: RGB + language prompt rolled by one, tactile untouched. This
             is the VL sensitivity — how much z changes when only vision+language change.
  z_padpert  language mask gets N extra tail tokens masked (padding-length perturbation). A
             sanity control — z should be roughly invariant to how much padding is masked.

METRICS (each group vs z_real, per sample, z flattened):
  cos        cosine similarity mean±std. Also reported CENTERED (cos_cent): the batch-mean of
             z_real is subtracted from both sides first. The learned latent queries add a large
             shared constant to every z, which pushes RAW cosine toward 1 and destroys
             resolution — JUDGE ON THE CENTERED NUMBERS.
  relL2      ||z_v - z_real|| / ||z_real|| mean±std.
  z_xsample  cos(z_real[i], z_real[i+1]) — collapse indicator. If z barely varies across
             samples (centered cos ≈ 1) then nothing drives z and every sensitivity is moot.

THE HEADLINE NUMBER: R
  R = (1 - cos(z_real, z_shuffle)) / (1 - cos(z_real, z_vlswap))   [centered, tactile subset]
    = tactile sensitivity / VL sensitivity.

  R >> 1  (>= 3)   TACTILE-DOMINANT  — z responds far more to touch than to VL. Healthy.
  R ~ 1   (0.3-3)  MIXED             — z reads both; tactile has not clearly won.
  R << 1  (< 0.3)  VL-ROUTER DISEASE — z is essentially blind to touch, driven by VL.

  Reference values we have measured: a joint-KV predictor reaches R ≈ 3.9 (healthy); the tactile-KV
  variant reaches far higher tactile causality in simulation. See docs/TACTILE_CAUSAL_PROBE.md
  for the full interpretation guide and decision tree.

RUN (single GPU):
    python scripts/probe_z_tactile_dependence.py \\
        --config <train-config-name> --ckpt <path/to/model.safetensors> \\
        --batches 8 --batch-size 16 --out-json probe.json --markdown

The checkpoint must be a FULL-key state_dict (what scripts/train_pytorch.py saves); the probe
fails loudly if any key is missing or unexpected.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="flexiv_tactile_reference",
                   help="training config name; defines the model AND the data loader used to draw the probe batch")
    p.add_argument(
        "--ckpt",
        default=(
            "/path/to/checkpoints/vtla_tactile_posttrain/<experiment>/<step>/model.safetensors"
        ),
        help="path to model.safetensors — must be a FULL-key state_dict (as saved by scripts/train_pytorch.py)",
    )
    p.add_argument("--batches", type=int, default=8, help="number of batches to average the metrics over")
    p.add_argument("--batch-size", type=int, default=16,
                   help="samples per batch (shuffle/vlswap roll tactile/VL within a batch, so larger is more robust)")
    p.add_argument("--num-workers", type=int, default=4, help="data loader workers")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the shuffled data loader — fixes which batches are drawn (reproducible)")
    p.add_argument("--pad-extra", type=int, default=30, help="z_padpert: extra lang-mask tokens masked at the tail")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--dtype", default=None, choices=(None, "bfloat16", "float32"),
                   help="model dtype; default = config.pytorch_training_precision (matches the ckpt)")
    p.add_argument("--out-json", default=None, help="if set, write the full result dict as JSON here")
    p.add_argument("--markdown", action="store_true",
                   help="also print a Markdown report (tables + R + verdict) to stdout — paste into an issue or paper")
    p.add_argument("--out-md", default=None, help="if set, write the Markdown report to this path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build model + loader (mirrors scripts/train_pytorch.py, single GPU, eval)
# ---------------------------------------------------------------------------
def build(args):
    import n0vtla.training.config as _config
    import n0vtla.training.data_loader as _data
    from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy, N0VTLAConfig

    config = _config.get_config(args.config)
    config = dataclasses.replace(
        config, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )
    model_cfg = config.model
    assert isinstance(model_cfg, N0VTLAConfig), f"config.model is {type(model_cfg)}"
    dtype = args.dtype or config.pytorch_training_precision
    object.__setattr__(model_cfg, "dtype", dtype)

    device = torch.device(args.device)
    model = N0VTLAPolicy(model_cfg).to(device)
    model.eval()

    import safetensors.torch as _st

    missing, unexpected = _st.load_model(model, args.ckpt, strict=False, device=str(device))
    print(f"[ckpt] {args.ckpt}\n[ckpt] missing={sorted(missing)}\n[ckpt] unexpected={sorted(unexpected)}", flush=True)
    if missing or unexpected:
        raise SystemExit("FATAL: ckpt did not load cleanly (see missing/unexpected above)")

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True, num_batches=args.batches)
    return model, loader, device, dtype


# ---------------------------------------------------------------------------
# z computation: the model's own path, deterministic (train=False => no augmentation)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_z(model, obs):
    """Compute latent tactile features through the model's standard preprocessing path."""
    images, img_masks, lang_tokens, lang_masks, _state, _ = model._preprocess_observation(obs, train=False)
    vl_ctx, *_rest = model._prefix_forward(images, img_masks, lang_tokens, lang_masks, use_cache=False)
    z, _g, has_tac = model._compute_z(vl_ctx)
    return z.float(), has_tac.to(torch.bool).cpu()


def tac_family(key: str, tac_keys: tuple[str, ...]) -> bool:
    return any(key == t or key.startswith(t + ".") for t in tac_keys)


def variant_obs(obs, name: str, tac_keys, pad_extra: int):
    """Build the perturbed Observation for one probe group. Never mutates `obs`'s dicts."""
    if name == "real":
        return obs
    if name == "null":
        imgs = dict(obs.images)
        for t in tac_keys:
            bk = t + ".baseline"
            if t in imgs and bk in imgs:
                imgs[t] = imgs[bk]  # current := baseline => diff = 0
        return obs.replace(images=imgs)
    if name == "shuffle":
        imgs, masks = dict(obs.images), dict(obs.image_masks)
        for k in list(imgs):
            if tac_family(k, tac_keys):
                imgs[k] = torch.roll(imgs[k], shifts=-1, dims=0)
                if k in masks:
                    masks[k] = torch.roll(masks[k], shifts=-1, dims=0)
        return obs.replace(images=imgs, image_masks=masks)
    if name == "vlswap":
        imgs, masks = dict(obs.images), dict(obs.image_masks)
        for k in list(imgs):
            if not tac_family(k, tac_keys):
                imgs[k] = torch.roll(imgs[k], shifts=-1, dims=0)
                if k in masks:
                    masks[k] = torch.roll(masks[k], shifts=-1, dims=0)
        kw = dict(
            images=imgs,
            image_masks=masks,
            tokenized_prompt=torch.roll(obs.tokenized_prompt, shifts=-1, dims=0),
            tokenized_prompt_mask=torch.roll(obs.tokenized_prompt_mask, shifts=-1, dims=0),
        )
        return obs.replace(**kw)
    if name == "padpert":
        m = obs.tokenized_prompt_mask
        lengths = m.sum(dim=1)
        n_extra = torch.clamp(torch.minimum(torch.full_like(lengths, pad_extra), lengths - 1), min=0)
        keep = lengths - n_extra
        idx = torch.arange(m.shape[1], device=m.device)[None, :]
        return obs.replace(tokenized_prompt_mask=m & (idx < keep[:, None]))
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pair_metrics(z_a, z_b, mu=None):
    """Per-sample flatten cos + relative L2. mu: optional (n_latent, llm_dim) batch-mean of z_real
    subtracted from BOTH sides (centered variant, kills the shared latent_queries component)."""
    if mu is not None:
        z_a, z_b = z_a - mu, z_b - mu
    a, b = z_a.flatten(1), z_b.flatten(1)
    cos = F.cosine_similarity(a, b, dim=1)
    rel = (a - b).norm(dim=1) / (a.norm(dim=1) + 1e-8)
    return cos.cpu().numpy(), rel.cpu().numpy()


def agg(x: list[np.ndarray]):
    if not x or sum(v.size for v in x) == 0:
        return None
    v = np.concatenate(x)
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)}


def fmt(s):
    return "        n=0 (skip)" if s is None else f"{s['mean']:+.6f} ± {s['std']:.6f}  (n={s['n']})"


def render_markdown(out: dict) -> str:
    """Render the result dict (same object written to --out-json) as a Markdown report.

    Pure formatting over already-computed numbers — no model/metric logic here. Handy for
    pasting a probe result straight into a GitHub issue or a paper appendix."""
    m = out["meta"]

    def cell(st):
        return "n=0" if st is None else f"{st['mean']:+.4f} ± {st['std']:.4f}"

    def num(x):
        return "n/a" if x is None else (f"{x:.3e}" if abs(x) < 1e-2 else f"{x:.4f}")

    meaning = {
        "null": "tactile signal removed (diff=0)",
        "shuffle": "tactile swapped in batch — **tactile sensitivity**",
        "vlswap": "RGB+prompt swapped in batch — **VL sensitivity**",
        "padpert": "extra padding masked — sanity control",
        "xsample": "z[i] vs z[i+1] — collapse indicator",
    }

    L = []
    L.append("### Tactile causal probe")
    L.append("")
    L.append(f"- config: `{m['config']}`")
    L.append(f"- checkpoint: `{m['ckpt']}`")
    L.append(f"- dtype `{m['dtype']}` · batch_size {m['batch_size']} · batches {m['batches']} · seed {m['seed']}")
    L.append(f"- rows: {m['n_rows']} total, {m['n_tac_rows']} with real tactile "
             f"(recompute noise floor max|Δz| = {m['noise_floor_max_abs_dz']:.1e}, expect ~0)")
    L.append(f"- z_norm: {cell(out['z_norm'])}")
    L.append("")
    L.append("| group (subset) | centered cos vs z_real | raw cos | relL2 | meaning |")
    L.append("|---|---|---|---|---|")
    for g, s in (("null", "tac"), ("shuffle", "tac"), ("vlswap", "tac"),
                 ("padpert", "all"), ("xsample", "all")):
        st = out["groups"][f"{g}/{s}"]
        L.append(f"| z_{g} ({s}) | {cell(st['cos_c'])} | {cell(st['cos'])} | {cell(st['rel'])} | {meaning[g]} |")
    L.append("")
    r = out["ratios"]["centered"]
    L.append("**Sensitivity ratios (centered — the ones to trust):**")
    L.append("")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| 1 − cos(shuffle) — tactile sensitivity | {num(r['1-cos(shuffle)'])} |")
    L.append(f"| 1 − cos(vlswap) — VL sensitivity | {num(r['1-cos(vlswap)'])} |")
    L.append(f"| 1 − cos(null) — tactile-removal sensitivity | {num(r['1-cos(null)'])} |")
    L.append(f"| **R = shuffle / vlswap** | **{num(r['R_shuffle/vlswap'])}** |")
    L.append(f"| R_null = null / vlswap | {num(r['R_null/vlswap'])} |")
    L.append("")
    R = out["verdict"]["R_centered"]
    L.append(f"**Verdict (centered R = {num(R)}):** {out['verdict']['text']}")
    L.append("")
    L.append("> R ≫ 1 (≥3) tactile-dominant · R ~ 1 (0.3–3) mixed · R ≪ 1 (<0.3) VL-router disease. "
             "Use centered numbers; raw cos is inflated by the shared latent-query constant.")
    return "\n".join(L)


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    model, loader, device, dtype = build(args)
    tac_keys = tuple(getattr(model.config, "tactile_image_keys", ()) or ())
    print(f"[setup] dtype={dtype} tac_keys={tac_keys} B={args.batch_size} batches={args.batches} "
          f"pad_extra={args.pad_extra} seed={args.seed}", flush=True)

    import jax

    groups = ("null", "shuffle", "vlswap", "padpert")
    acc = {g: {s: {m: [] for m in ("cos", "cos_c", "rel")} for s in ("all", "tac")} for g in groups}
    acc["xsample"] = {"all": {m: [] for m in ("cos", "cos_c", "rel")}}
    znorm, noise_floor = [], None
    n_tac_rows = n_rows = 0

    t0 = time.time()
    for bi, (observation, _actions) in enumerate(loader):
        observation = jax.tree.map(lambda x: x.to(device), observation)
        z_real, has_tac = compute_z(model, observation)
        mu = z_real.mean(dim=0, keepdim=True)
        znorm.append(z_real.flatten(1).norm(dim=1).cpu().numpy())
        n_rows += z_real.shape[0]
        n_tac_rows += int(has_tac.sum())

        if bi == 0:
            z_again, _ = compute_z(model, observation)
            noise_floor = float((z_again - z_real).abs().max())
            print(f"[selfcheck] recompute max|dz| = {noise_floor:.3e} (expect 0.0)", flush=True)

        m_tac = has_tac.numpy()
        m_tac_pair = m_tac & np.roll(m_tac, -1)

        for g in groups:
            zv, _ = compute_z(model, variant_obs(observation, g, tac_keys, args.pad_extra))
            cos, rel = pair_metrics(z_real, zv)
            cos_c, _ = pair_metrics(z_real, zv, mu=mu)
            sub = m_tac_pair if g == "shuffle" else m_tac
            for name, arr in (("cos", cos), ("cos_c", cos_c), ("rel", rel)):
                acc[g]["all"][name].append(arr)
                acc[g]["tac"][name].append(arr[sub])

        z_roll = torch.roll(z_real, shifts=-1, dims=0)
        cos, rel = pair_metrics(z_real, z_roll)
        cos_c, _ = pair_metrics(z_real, z_roll, mu=mu)
        for name, arr in (("cos", cos), ("cos_c", cos_c), ("rel", rel)):
            acc["xsample"]["all"][name].append(arr)

        print(f"[batch {bi}] done  tac_rows={int(m_tac.sum())}/{len(m_tac)}  "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    # ---------------- report ----------------
    out = {"meta": vars(args) | {"dtype": dtype, "n_rows": n_rows, "n_tac_rows": n_tac_rows,
                                 "noise_floor_max_abs_dz": noise_floor},
           "z_norm": agg(znorm), "groups": {}}
    print("\n================ PROBE RESULTS (vs z_real) ================")
    print(f"rows total={n_rows}  with real tactile={n_tac_rows}  "
          f"z_norm={fmt(agg(znorm))}  noise_floor={noise_floor:.3e}")
    rows = [(g, s) for g in groups for s in ("tac", "all")] + [("xsample", "all")]
    for g, s in rows:
        st = {m: agg(acc[g][s][m]) for m in ("cos", "cos_c", "rel")}
        out["groups"][f"{g}/{s}"] = st
        print(f"\n z_{g:8s} [{s:3s}]  cos      = {fmt(st['cos'])}")
        print(f"                     cos_cent = {fmt(st['cos_c'])}")
        print(f"                     relL2    = {fmt(st['rel'])}")

    def one_minus(g, s, m):
        a = out["groups"][f"{g}/{s}"][m]
        return None if a is None else max(1.0 - a["mean"], 0.0)

    print("\n================ SENSITIVITY RATIOS ================")

    def _f(x):
        return "n/a" if x is None else f"{x:.3e}"

    ratios = {}
    for m, label in (("cos", "raw"), ("cos_c", "centered")):
        d_sh, d_vl = one_minus("shuffle", "tac", m), one_minus("vlswap", "tac", m)
        d_nl, d_pp = one_minus("null", "tac", m), one_minus("padpert", "all", m)
        R = (d_sh / d_vl) if (d_sh is not None and d_vl not in (None, 0.0)) else float("nan")
        R_null = (d_nl / d_vl) if (d_nl is not None and d_vl not in (None, 0.0)) else float("nan")
        S5 = (d_pp / d_vl) if (d_pp is not None and d_vl not in (None, 0.0)) else float("nan")
        ratios[label] = {"1-cos(shuffle)": d_sh, "1-cos(vlswap)": d_vl, "1-cos(null)": d_nl,
                         "1-cos(padpert)": d_pp, "R_shuffle/vlswap": R, "R_null/vlswap": R_null,
                         "S5_padpert/vlswap": S5}
        print(f" [{label:8s}] 1-cos: shuffle={_f(d_sh)} null={_f(d_nl)} vlswap={_f(d_vl)} "
              f"padpert={_f(d_pp)}  =>  R={R:.3f}  R_null={R_null:.3f}  S5={S5:.3f}")
    out["ratios"] = ratios

    R = ratios["centered"]["R_shuffle/vlswap"]
    verdict = ("INCONCLUSIVE: no valid tactile samples, or zero VL-swap sensitivity; increase --batches"
               if not np.isfinite(R)
               else "TACTILE-DOMINANT: latent features respond primarily to tactile input" if R >= 3.0
               else "MIXED: latent features respond to both tactile and vision-language input" if R > 0.3
               else "VL-DOMINANT: latent features show limited tactile sensitivity")
    out["verdict"] = {"R_centered": R, "text": verdict}
    print(f"\nVERDICT (centered R = {R:.3f}): {verdict}")
    print("Interpretation: R >> 1 indicates tactile dominance; use the centered metric because "
          "constant latent-query components can inflate raw cosine similarity.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nwrote {args.out_json}", flush=True)

    if args.markdown or args.out_md:
        md = render_markdown(out)
        if args.markdown:
            print("\n" + md, flush=True)
        if args.out_md:
            with open(args.out_md, "w") as f:
                f.write(md + "\n")
            print(f"wrote {args.out_md}", flush=True)

    print("PROBE_Z_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
