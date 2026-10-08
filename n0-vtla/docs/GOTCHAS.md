# Implementation and Training Considerations for Multimodal VLM Extensions

This guide covers non-obvious failure modes that can occur when adding a new
input modality (here: tactile) and a new module (a tactile predictor) on top of a large pretrained
vision-language-action model. Every item below cost us real GPU-days. Most of them **do not show
up in the loss curve**. The run looks healthy while it is quietly broken.

Each entry is **Symptom → Root cause → Fix (with the code that implements it)**. Code is
referenced by file + function name so the references survive line-number drift. None of these
are exotic; they are the standard tax of extending a frozen-ish pretrained backbone, and you
will hit most of them on any similar project.

---

## 1. Synchronizing newly added modules across DDP ranks

**Symptom.** Multi-GPU training looks completely normal: loss goes down, no crash, no warning
you'd notice. But the final model behaves as if it were under-trained or noisy, and single-GPU
runs behave *differently* from multi-GPU ones. The only visible tell is a log line at startup:
`Loaded base weights with missing keys=[...]`, and **your new module's parameters are in that
missing-keys list**.

**Root cause.** A subtle interaction of three reasonable-looking choices:

1. You seed each rank differently (`set_seed(seed + rank)`) before building the model, so each
   rank's *randomly initialized* parameters start **different**.
2. You wrap in DDP with the initial parameter broadcast **disabled** (`init_sync=False`) because
   you're about to load a checkpoint anyway. In this repo that's
   `skip_ddp_init_sync = use_ddp and (config.pytorch_weight_path is not None or resuming)`.
3. You load the pretrained backbone with `strict=False`, because your new module's keys aren't
   in it.

This creates a synchronization issue: the backbone keys get loaded identically on every rank, but your **new
module's keys are not in the checkpoint**, so `strict=False` leaves them at their *per-rank
random init*. DDP only ever all-reduces **gradients**, never parameters, so those weights are
**never reconciled**. Rank 0 trains one tactile head, rank 1 trains a different one, and so on.
The averaged gradient is applied to divergent weights on every rank. It trains, it just trains
garbage, and nothing ever self-heals.

**Fix.** After the `strict=False` load, **broadcast the entire model (parameters *and* buffers)
from rank 0** so the missing-key modules inherit rank 0's init:

```python
with torch.no_grad():
    for tensor in itertools.chain(model.module.parameters(), model.module.buffers()):
        dist.broadcast(tensor.data, src=0)
```

Loaded keys are already identical, so the broadcast is a no-op for them; only the previously
de-synced new modules change. In this repo: `scripts/train_pytorch.py`, in the weight-loading
section right after `safetensors.torch.load_model(..., strict=False)`. Repeat the same
full-model re-broadcast whenever you overlay extra weights later, for exactly this reason.

**How to catch it yourself.** Grep the startup log for `missing keys` and check whether any of
*your* module's parameters are listed. If they are, and you use DDP with `init_sync=False`, you
have this bug until you broadcast.

---

## 2. Gradient feedback into the backbone can destabilize optimization

**Symptom.** Training is stable at a low learning rate, then you raise the peak LR (or reach the
peak of a warmup schedule) and `grad_norm` **grows geometrically**, roughly the same
multiplicative factor every logging interval (e.g. 0.3 → 0.6 → 1.1 → 2.1 → 3.8 → …), eventually
hitting `inf`. Crucially it is a *steady exponential ramp*, not a single spike.

**Root cause.** Your freshly-initialized module (the predictor) takes the backbone's context tensor
as input **without detaching it**, and its output flows into the loss. So the loss gradient now
has a path back through your module *into the backbone*, forming a **positive feedback loop**: a
random-init module perturbs the backbone, which changes the module's input, which enlarges the
perturbation… At a small LR the loop is contractive; past a critical LR it becomes expansive and
the grad norm compounds every step. This is a genuine dynamical instability, not bad data.

> **Distinguish it from bad data.** A corrupt sample or a label outlier produces a **single
> sharp spike** in `grad_norm` that then returns to baseline. A feedback loop produces a
> **near-constant ratio of geometric growth** across consecutive log intervals. If you plot
> `log(grad_norm)` vs step and it's a straight rising line, it's feedback, not data.

**Fix.** Two independent dampers, both cheap:

1. **Detach the backbone context where it enters your module.** The predictor consumes
   `vl_ctx.detach()`, so action gradients still train the predictor *through its own output* `z`, but
   no gradient flows predictor → backbone. The backbone still gets its gradient the normal way (via
   the action expert's attention over the prefix). See `N0VTLAPolicy._forward_predictor` in
   `n0vtla/models_pytorch/n0vtla_policy.py` (the `self._compute_z(vl_ctx.detach(), ...)`
   call).
2. **Put the new module's parameters in their own optimizer group at a lower LR.** The predictor
   heads are the freshest, fastest-moving params; scaling their LR down (here ×0.1 via
   `VTLA_PREDICTOR_LR_SCALE`) keeps them from outrunning the schedule. See the optimizer-group
   construction in `scripts/train_pytorch.py` (params whose names start with `tactile_predictor.`,
   `z_proj.`, `tactile_encoder.` get `lr_scale`).

---

## 3. Zero-initialized gates preserve pretrained behavior at initialization

**Symptom.** The instant you enable the new module, early-training loss jumps well above the
pretrained baseline and takes a long time to recover, or never fully does. The pretrained
skill is being disturbed before the new module has learned anything useful.

**Root cause.** A randomly-initialized module's output is **noise**, and you are injecting that
noise straight into a carefully-pretrained model's computation (here: prepending the latent `z`
to the action expert's suffix tokens). At step 0 the model is strictly worse than the pretrained
model it started from, because you've added a large random perturbation to a good solution.

**Fix.** Gate the new module's output behind a **learnable scalar initialized to zero**, so at
init the injection is exactly muted and the model reproduces the pretrained behavior *bitwise*;
the gate then opens gradually under gradient. This is the standard trick from **LoRA** (zero-init
`B` matrix), **ControlNet** (zero-convolutions), and **Flamingo / gated cross-attention**
(`tanh` gate initialized to 0). Do it at the **single choke point** every consumer of the
module's output routes through, so "gate off" is byte-identical to the base model.

In this repo: `z_gate_zero_init` creates `self.z_gate = nn.Parameter(torch.zeros(1))`, applied in
`N0VTLAPolicy._embed_suffix_with_z` in `n0vtla/models_pytorch/n0vtla_policy.py`. Keeping
the parameter *absent* when the feature is off also keeps the checkpoint's key set unchanged.

---

## 4. Non-finite gradients can invalidate optimizer state

**Symptom.** Loss is training fine, then at some step it **jumps to a large value and stays
remain pinned at that value**: a flat, high plateau with no recovery for the rest of the run. Restarting
from a checkpoint before the event works; continuing past it never does.

**Root cause.** In mixed precision, a `bf16`/`fp16` overflow makes a gradient `inf` or `NaN`.
When you call `clip_grad_norm_`, the total norm is then `inf`/`NaN`, the clip coefficient
becomes `0` or `NaN`, **but the code still calls `optimizer.step()`**. That step multiplies
parameters by `NaN` (or applies a `NaN`-scaled update), poisoning the weights. From then on every
forward is `NaN`, and the loss is stuck. A single bad step is enough to destroy the run
permanently.

**Fix.** After clipping, **check the grad norm is finite and skip the optimizer step if not**:

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
if not torch.isfinite(grad_norm):
    nonfinite_grad_skips += 1
    # log and SKIP: do not call optim.step() / do not update params
else:
    optim.step()
```

Under DDP the grad norm is computed on already-all-reduced gradients, so it's identical on every
rank and the skip decision is DDP-consistent (no rank divergence). In this repo: the training
loop in `scripts/train_pytorch.py`, right after the `clip_grad_norm_` call (`nonfinite_grad_skips`
counter). Skipping one step is free; a poisoned step is fatal.

---

## 5. Verify that accelerator jobs are executing on GPU

**Symptom.** A multi-node job runs, the loss curve looks completely normal, but each step is
inexplicably slow. Nothing errors. You only notice because the ETA is absurd, or you don't
notice until you've burned a week of wall-clock.

**Root cause.** In a launch/payload script you set `LD_LIBRARY_PATH` with an **overwriting**
assignment:

```bash
export LD_LIBRARY_PATH=/some/staged/libs        # WRONG: clobbers the container's paths
```

This wipes out the path the container's runtime injected for `libcuda.so`. PyTorch then can't
find the CUDA driver, `torch.cuda.is_available()` returns `False`, and because the training
code falls back to `cpu` instead of hard-failing, **the whole model trains on CPU**. Same math,
same loss curve, ~80× slower.

**Fix.** Always **preserve** the existing value when extending `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/some/staged/libs:${LD_LIBRARY_PATH:-}   # RIGHT: append, don't clobber
```

Any custom launcher should preserve the existing value with the
`:${LD_LIBRARY_PATH:-}` suffix form.

**Autopsy / how to catch it.** Grep the job log for the periodic GPU-memory lines; this repo
logs `Step N (...): GPU memory - allocated: ...` via `log_memory_usage` in
`scripts/train_pytorch.py`, which **returns early when CUDA is unavailable**. **Zero `GPU memory`
lines in the log = the job ran on CPU.** More directly, log `torch.cuda.is_available()` once at
startup and fail loudly if it's `False` on a job you expect to be on GPU.

---

## 6. Multi-timestamp video queries decode the whole episode prefix per sample

**Symptom.** Adding a tactile (or any extra) video stream that you sample at **multiple
timestamps per item** (e.g. `[baseline=frame 0, current=t, future=t+H]`) makes each training
step 20–50× slower. The GPUs sit idle; the bottleneck is the data loader / CPU video decode, not
compute.

**Root cause.** A naïve multi-timestamp video read seeks to the **earliest** requested timestamp
and then decodes **forward** frame-by-frame until it passes the **latest** one. When your
timestamps span `[0, t, t+H]` and `t` is deep into a long clip, "earliest to latest" means
decoding the **entire episode prefix** (hundreds to ~a thousand frames), per view, **per
sample**, just to grab 2–3 frames. With long GOP (few keyframes) it's even worse.

**Fix.** **Cluster the requested timestamps and seek independently per cluster.** Timestamps
that are far apart (here: gap > ~2 s, ≈ a couple of GOPs) each get their own `seek` + short
decode; only near-neighbors are read by sequential decode. The `baseline=0` frame is a keyframe,
so it's one seek + one frame; `t` and `t+H` are each ≤ one GOP of decode. In this repo this is
`_decode_video_frames_precise_pyav` in `n0vtla/training/data_loader.py` (the `cluster_gap_s`
clustering loop), which cut per-step decode time by ~24× while returning **bit-identical** frames
to the naïve nearest-neighbor read.

**How to catch it.** If steps are slow and GPU utilization is low, profile the data loader in
isolation. A decode time that scales with **where in the episode** you sample (late-in-clip items
much slower than early ones) is the fingerprint of "seek-to-earliest, decode-to-latest".

---

## 7. Equivalence tests must cover the production configuration

**Symptom.** You add an optimization or a refactor (a KV-cache, a fused kernel, a fast path), you
write an equivalence test comparing it to the reference implementation, it passes at a tight
tolerance (say `1e-4`), and then production training behaves differently, or a stricter test
later fails. The test validated a different configuration.

**Root cause.** Equivalence you verified under **eval / fp32 / eager** does **not** automatically
hold under **train / bf16 / fused-attention**, because those modes take different code paths and
carry different numerical error:

- **`train()` vs `eval()`** enable/disable behavior. Gradient checkpointing in this repo is
  gated on `self.training` (`PI0Pytorch._apply_checkpoint` only checkpoints when
  `gradient_checkpointing_enabled and self.training`), so an eval-mode test never exercises the
  recompute path that production uses.
- **`bf16` vs `fp32`**: an equivalence that holds in fp32 can breach a `1e-4` threshold in bf16
  simply from rounding.
- **TF32.** Setting `torch.set_float32_matmul_precision("high")` (which `PI0Pytorch.__init__`
  does on CUDA) makes nominal-`fp32` matmuls run in **TF32**, injecting ~`1e-3`-level,
  **order-dependent** error. Two mathematically identical computations with different matmul
  *shapes* (e.g. a joint `(P+S)×(P+S)` attention vs a suffix-only `S×(P+S)`) will then differ at
  ~`1e-3` even though both are "fp32".
- **eager vs SDPA / fused attention** are not bitwise identical (here selectable via
  `VTLA_ATTN_IMPL`; default `eager`).

**Fix / discipline.**

- **Set your test tolerance to the precision you're actually running.** Don't assert `1e-4` on a
  TF32/bf16 path; expect ~`1e-3` and justify it. This repo's `scripts/verify_prefix_cache.py`
  forces `set_float32_matmul_precision("highest")` and disables cuDNN TF32 **after** model
  construction (which sets it back to "high"), precisely so the equivalence test measures the
  algorithm and not TF32 noise, and it pins the same attention backend on both paths.
- **Test the mode you ship.** If production trains in `train()`/bf16/SDPA with gradient
  checkpointing on, at minimum run the equivalence check once in that exact configuration (with a
  loosened, precision-appropriate tolerance), not only in the clean eval/fp32/eager setup where
  the numbers are prettiest.

---

## Summary

Every bug above shares a shape: **the loss curve kept looking healthy while the model was
broken.** When you extend a pretrained model, the loss is a *lagging, low-resolution* health
signal. Add cheap, direct assertions on the things that actually matter: parameters identical
across ranks, gradients finite, CUDA actually in use, `z` actually reading the sensor (see
`docs/TACTILE_CAUSAL_PROBE.md`), and check them *at startup and periodically*, not just at the
end when the compute is already spent.
