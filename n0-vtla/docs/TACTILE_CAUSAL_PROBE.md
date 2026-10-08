# Tactile Causal-Dependence Analysis

This tool applies to checkpoints whose tactile signal is encoded into a latent representation
that feeds a downstream prediction head.

## TL;DR

When you add a tactile stream to a large pretrained vision-language(-action) model, it is
possible to end up with a model that **ignores the tactile sensor** while its training loss
continues to decrease. This tool loads one checkpoint and answers a single question with a
number:

> Does the tactile latent `z` change more when I perturb the **touch** than when I perturb the
> **vision+language**?

The answer is a ratio **R = tactile-sensitivity / VL-sensitivity**:

- **R ≫ 1** is healthy: `z` is driven by touch.
- **R ≈ 1** is mixed: touch has not clearly won.
- **R ≪ 1** is sick: `z` is essentially a vision-language function; the sensor is decorative.

```bash
export PYTHONPATH=$PWD
python scripts/probe_z_tactile_dependence.py \
    --config <your-train-config> \
    --ckpt   checkpoints/<...>/model.safetensors \
    --batches 8 --batch-size 16 \
    --markdown --out-json probe.json
```

---

## 1. Failure mode: vision-language shortcutting

Setup: a tactile action-predictor encodes the tactile signal into a small latent `z` (a handful of
tokens). `z` is injected into the policy's action expert and is trained either by the action
loss (end-to-end) or by a self-supervised objective such as InfoNCE against a future-tactile
target (predictor pretraining).

A key issue is that, in real manipulation data, **what you see and what you're told predicts a lot
of what you'll feel.** The prompt says "pick up the strawberry", the RGB shows the gripper
closing on the strawberry, so the *tactile-correlated component* of the future contact is
already largely determined by the vision-language (VL) context. A gradient-descent optimizer,
handed both a VL context and a tactile stream, will happily learn to produce `z` from the **VL
context alone**, because:

- the VL features are information-rich, clean, and already well-trained (they come from a
  pretrained VLM), whereas
- your fresh tactile encoder is noisy and randomly initialized, and
- routing around it still lowers the loss.

The result is a model where `z` is a slightly-laundered copy of the VLM's guess about contact.
This is **vision-language shortcutting**: `z` has become a *router* that reads vision-language and
merely *labels* it as "tactile". Symptoms:

- **Loss goes down normally.** InfoNCE / action loss both improve. Nothing looks wrong.
- **The tactile encoder can be frozen, zeroed, or fed garbage** and downstream metrics barely
  move.
- **At deployment the policy is blind to touch**: it cannot tell a firm grasp from a slip,
  cannot react to unexpected contact, and generalizes to a new sensor no better than an RGB-only
  baseline.

You cannot catch this from the loss curve. You need a **causal** test: intervene on the input
and measure the effect on `z`.

---

## 2. How the probe works

The probe never trains anything. It loads one checkpoint, draws a batch of **real**
observations, and computes `z` through the model's own path in `eval()` mode (deterministic, no
augmentation, no dropout). Then it recomputes `z` under five controlled input perturbations and
measures how far each moved `z` away from the unperturbed reference.

### The five `z` groups

| group | what is perturbed | what it isolates |
|---|---|---|
| `z_real` | nothing | the reference |
| `z_null` | tactile current frame ← baseline frame, so the tactile diff `tac_t − tac_0` is exactly **zero** | response to **removing** the tactile signal |
| `z_shuffle` | the whole tactile group (frames + masks) is **rolled by one within the batch**; sample *i* gets sample *i+1*'s touch; VL untouched | **tactile sensitivity** |
| `z_vlswap` | the mirror control: RGB **and** language prompt rolled by one; tactile untouched | **VL sensitivity** |
| `z_padpert` | language mask gets N extra tail tokens masked (a padding-length change) | sanity control; `z` should barely move |

The key comparison is **`z_shuffle` (only touch changed) vs `z_vlswap` (only vision+language
changed)**. If `z` is tactile-driven, swapping the touch should move it a lot and swapping the
VL should move it little. With vision-language shortcutting, the relationship is reversed.

`z_null` is a second, independent read on the same question: zeroing the tactile *difference*
should collapse a healthy `z` toward a constant. `z_padpert` is a red-herring detector: a
padding-length change carries no task information, so a large `z_padpert` response means `z` is
picking up on spurious formatting artifacts.

### The metrics

For each group, per sample, versus `z_real`:

- **cos**: cosine similarity. Reported both raw and **centered** (`cos_cent`): the batch-mean
  of `z_real` is subtracted from both sides first.
  - **Always judge on the centered numbers.** The model's learned latent queries add a large
    *shared constant* to every `z`, which inflates raw cosine toward 1.0 and destroys
    resolution. Raw cosine of 0.99 can hide a centered cosine of 0.3.
- **relL2**: `‖z_v − z_real‖ / ‖z_real‖`, a magnitude cross-check.
- **z_xsample**: `cos(z_real[i], z_real[i+1])`, the **collapse indicator**. If `z` barely
  varies across *different samples* (centered cos ≈ 1), then nothing drives `z` at all and every
  sensitivity number is meaningless. Check this first.

### Attribution ratio

```
R = (1 − cos_cent(z_real, z_shuffle)) / (1 − cos_cent(z_real, z_vlswap))
  = tactile sensitivity / VL sensitivity
```

computed on the **tactile subset** (only rows that actually carry real tactile, and, for
`shuffle`, whose swap partner does too; placeholder rows are excluded).

A **self-check** runs on the first batch: it recomputes `z_real` a second time and prints the
max element-wise difference, which must be ~0 (the path is deterministic in `eval()`). A
non-zero noise floor means something stochastic leaked in and the numbers are not trustworthy.

---

## 3. Running it on your own checkpoint

Requirements:

- The checkpoint is a **full-key** `model.safetensors` (every parameter present). The probe
  loads with `strict=False` but **fails loudly** if any key is missing or unexpected, a partial
  checkpoint would probe randomly initialized modules.
- The `--config` you pass must construct the same model architecture as the checkpoint and must
  define a data loader that yields real tactile observations.

```bash
export PYTHONPATH=$PWD
python scripts/probe_z_tactile_dependence.py \
    --config  <your-train-config> \
    --ckpt    checkpoints/<...>/<step>/model.safetensors \
    --batches 8 --batch-size 16 \
    --markdown --out-json probe.json
```

Useful flags (`--help` lists all):

- `--markdown`: also print a Markdown report (tables + R + verdict), ready to paste into a
  GitHub issue or a paper appendix. `--out-md report.md` writes it to a file.
- `--out-json probe.json`: the full result dict (all groups, all metrics, ratios, meta).
- `--batches` / `--batch-size`: enlarge if the verdict is `INCONCLUSIVE` (too few real-tactile
  rows sampled, or VL sensitivity numerically 0).
- `--seed`: the data loader is shuffled but seeded, so a given seed draws reproducible batches.
- `--dtype`: defaults to the config's training precision (matches how the checkpoint was
  saved); override only if you know why.

---

## 4. Interpreting the result

Start at the top; stop at the first branch that matches.

```
1. Is z_xsample centered cos ≈ 1 (z barely varies across samples)?
   └─ YES → z has COLLAPSED. It is a near-constant; no sensitivity is meaningful.
            Do not read R. Cause is usually a dead/muted encoder, a saturated gate, or
            an over-strong shared latent-query constant. Fix the collapse first, re-probe.
   └─ NO  → continue.

2. Is the verdict INCONCLUSIVE (R = NaN)?
   └─ YES → too few real-tactile rows, or VL sensitivity was numerically 0.
            Raise --batches / --batch-size and re-run.
   └─ NO  → continue.

3. Look at R (centered):

   R ≥ 3      TACTILE-DOMINANT.  Healthy. z is driven by touch. Cross-check:
              z_null should also show a clear response (1 − cos(null) not tiny), and
              z_padpert should be near-invariant (1 − cos(padpert) ≈ 0). If padpert is
              large, z is reading spurious formatting → investigate even though R is high.

   0.3 – 3    MIXED.  z reads both touch and VL; touch has not clearly won. Common on early
              end-to-end checkpoints. Interventions that push R up: an architecture where the
              tactile tokens are the ONLY key/value for z (so VL cannot leak in), a zero-init
              gate so the model starts from the pretrained behavior and earns z gradually, and
              VL-dropout during training so the action head cannot lean on the prefix.

   R < 0.3    VL-ROUTER DISEASE.  z is essentially blind to touch. The sensor is decorative.
              Also expect a small z_null response. Do not ship this as a "tactile" model.
```

### Reference values

- A **joint-KV predictor** (latent queries attend to the concatenation `[vl_ctx ; tactile_tokens]`)
  reaches **R ≈ 3.9** after predictor pretraining: healthy, but VL is still an available shortcut
  because it sits in the same key/value stream.
- A **tactile-KV predictor** (tactile tokens are the *only* key/value; the VL context is demoted to
  a query-side conditioner so its content cannot flow into `z`) reaches **far higher** tactile
  causality in simulation: the tactile-attribution ratio jumps by orders of magnitude versus
  the joint-KV design. This is an effective architectural constraint against vision-language shortcutting: **if VL
  cannot be a key/value for `z`, `z` cannot route around the sensor.**

The reported values are reference measurements rather than universal thresholds. Interpret the
qualitative bands (≥3 / 0.3–3 / <0.3) together with the collapse and padding-perturbation checks.

---

## 5. Caveats

- **The probe measures sensitivity of `z`, not task performance.** A high R says `z` reads
  touch; it does not by itself prove the touch information improves the policy. Pair it with a
  closed-loop or held-out tactile-ablation evaluation.
- **`z` is centered per batch.** Interpret centered metrics; the raw ones exist only to expose
  how much of `z` is the shared constant.
- **Perturbations are batch-internal rolls**, so a larger batch gives more independent swap
  partners and a more stable R. Very small batches are noisy.
- **It is architecture-agnostic in spirit but path-specific in code**: it calls the model's
  `_preprocess_observation → _prefix_forward → _compute_z` path. Porting it to a different stack
  means pointing those three calls at your equivalent tactile-latent computation.
