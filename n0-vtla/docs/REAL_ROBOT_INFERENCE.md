# Real-Robot Inference: Interface Contract

For whoever is wiring up a real single-arm robot (the reference configuration is a UFACTORY xArm6 arm with a
BrainCo Revo2 hand) to a trained single-arm post-train checkpoint (dual-arm: see `REAL_ROBOT_INFERENCE_DUAL.md`). This
document specifies exactly what the inference server expects to receive and what it returns, taken from the actual
transform code (`n0vtla/policies/canonical_tactile_policy.py`). **What is NOT provided**: any
robot-specific driver (arm SDK calls, hand motor commands, camera/tactile
capture), or a safety layer (limits, watchdog, e-stop). That is what you are
adding. See §6 before sending a single command to a real arm.

## 0. Status and limitations

- The data and conversion contract is in `POST_TRAIN.md`. Offline checks (for example a smoothness check of the
  predicted trajectory against recorded demonstrations) test that contract, not real-world behaviour.
- **This server has not been exercised end-to-end against a real robot.** This
  document specifies the contract from the code. Treat the first real connection as a new
  integration test, not a formality.
- **Physical cross-host sync (camera/tactile/robot clock alignment) of the training data is
  typically NTP-level only**, and an offline check says nothing about it. It does not block live inference
  (which uses its own real-time clock), but offline results should not be expected to predict real-world success.

## 1. Getting the code, and where the data lives

**Code**: use this repository as a whole (`n0-vtla/`), not a partial copy:

```bash
cd n0-vtla
pip install -r requirements.txt   # or: see pyproject.toml
```

`serve_policy.py` needs the full `n0vtla/` package (model implementation,
training config, transforms, model loader) and `n0vtla/serving/` (the
websocket server); all of it is in this repo.

**Checkpoints and normalization files** (not in git). Paths are relative to `n0-vtla/`: the original N0-VTLA checkpoint at
`../checkpoints/n0-vtla-base` (`hf download NeoteAI/n0-vtla-base --local-dir ../checkpoints/n0-vtla-base`), your post-trained checkpoint at
`checkpoints/vtla_tactile_posttrain/<exp name>/<step>/` (written by `scripts/posttrain.sh`, see `POST_TRAIN.md`).
Deployment needs two normalization files, both fitted from your own training split at data-conversion time:
the state/action `norm_stats.json` (`scripts/compute_canonical_norm.py`; training copies it into the checkpoint's
`assets/<asset id>/`, and the **server** reads it from there) and the tactile normalization JSON
(`scripts/fit_robot_tactile_norm.py`; used by your **client** to encode live glove readings, section 4.2; the server
never reads it, so keep it next to the checkpoint).

## 2. Starting the server

### 2.1 GPU / hardware requirements (indicative numbers)

Indicative numbers on one H200 GPU, from `policy.infer()` calls identical to what `serve_policy.py` does per request
(`torch.cuda.max_memory_allocated`/`max_memory_reserved`; this repo's PyTorch models don't need JAX at inference time):

| Stage | Allocated | Reserved |
|---|---|---|
| After loading the checkpoint | about 8.3 GB | about 8.3 GB |
| After the first `infer()` call, and steady state (no growth) | about 8.4 GB | about 8.6 GB |

Latency: the first call includes a one-time warmup (about 1 s); steady state is about **0.22 s/call**. One
call returns the full 50-step chunk (1.67 s of robot motion at 30 Hz native
rate), so inference latency is not the bottleneck for a real-time control loop.

**Recommendation: one GPU with ≥16 GB VRAM** (e.g. RTX 4080/3090/A4000-class or
better) — covers the ~8.6 GB above with headroom for driver
overhead and anything else running on the same box. **No multi-GPU needed for
inference**: multi-GPU DDP is a training-only requirement (optimizer
state, gradients, backward-pass activations); none of that exists at inference
time.

These numbers are **specific to this backbone**, not a general
N0-VTLA constant, and the backbone is inherently large by design. Per the model summary in `README.md`:

| Component | Choice |
|---|---|
| Backbone | PaliGemma (gemma_2b prefix) — a ~3B-parameter vision-language model (SigLIP vision encoder + Gemma-2B) |
| Action expert | Gemma 300M — a separate, smaller transformer doing flow matching over the 50-step action chunk |
| Tactile encoder | Frozen DINOv2 (`facebook/dinov2-base`) over baseline-difference images |
| Tactile pathway | Cross-attention predictor → 5 latent tokens, injected into the action expert |
| Precision | bf16 parameters, eager attention (matches the pretraining path) |

`model.safetensors` is about 8.25 GB, matching the GPU allocation after loading
almost exactly (inference loads in bf16 without upcasting): 8.25 GB /
2 bytes-per-param ≈ 4.1B parameters, consistent with PaliGemma (~3B) + the
300M action expert + DINOv2 (~86M, frozen) + the tactile
predictor/projection modules. This is standard for this class of VLA
foundation model (the `pi05`/Pi0.5 lineage this config name refers to): the
point is inheriting broad visual-language priors from large-scale pretraining before task-specific
post-training. A different backbone or a larger pretrained checkpoint would need
correspondingly more VRAM/RAM; re-measure rather than
assuming these numbers carry over. Training used `VTLA_ATTN_IMPL=eager` (no
flash-attention or other Hopper/Ampere-specific kernels), so inference has no
unusual GPU compute-capability requirement beyond bf16 support.

### 2.2 System RAM (host, not GPU) requirements — read this even if VRAM looks fine

**TL;DR: pass `--low-cpu-mem-usage` to `serve_policy.py` (see below) unless
your host has ~17 GB of free RAM to spare purely for the loading step.**
The rest of this section explains why.

Indicative host-RAM numbers around `create_trained_policy` (the call `serve_policy.py` makes once at startup):

| Mode | Peak RSS (transient, during load) | Steady-state RSS | Load time |
|---|---|---|---|
| default (`low_cpu_mem_usage=False`) | about 16.6 GB | about 3.8 GB | about 50 s |
| `low_cpu_mem_usage=True` | **about 9.6 GB** | **about 1.8 GB** | **about 5 s** |

**Without `--low-cpu-mem-usage`, the host needs ~17 GB of free system RAM to
survive loading the checkpoint, even though steady-state usage afterward is
under 4 GB.** A machine sized only for steady-state operation (e.g. 8 GB free)
will fail during loading, before the server ever starts accepting connections.
The load step accounts for essentially all of the peak; `policy.infer()` adds no further peak.

**Cause**: `N0VTLAPolicy.__init__`
(`n0vtla/models_pytorch/n0vtla_policy.py`) builds
`PaliGemmaWithExpertModel` — the PaliGemma+action-expert backbone, the bulk of
the 4.1B params — in PyTorch's default dtype (fp32) on CPU first (4.1B ×
4 bytes ≈ 16.4 GB), then casts it down to bf16 (`to_bfloat16_for_selected_params`).
`N0VTLAConfig.load_pytorch` then copies the checkpoint's values into that model with a plain `load_state_dict`,
which does not re-introduce an fp32 copy. The spike is therefore a **construction-time** cost
(build-fp32-then-cast), not a checkpoint-loading one; the fp32 copy is freed once casting finishes, but the host
needs enough RAM to hold it at the peak.

**Opt-in fix: `N0VTLAConfig.low_cpu_mem_usage=True`.** This field
(default `False`) constructs the backbone on
`torch.device("meta")` (no real allocation) and loads the checkpoint's
tensors directly via `load_state_dict(..., assign=True)`, which assigns each
meta parameter the checkpoint's own tensor (already in its final mixed
fp32/bf16 dtype) instead of allocating-then-casting. The small
`FrozenDINOv2TactileEncoder` submodule (uses `AutoModel.from_pretrained`,
incompatible with a blanket meta context without `accelerate`) is still
constructed normally; it is only ~344 MB. A handful of
tensors are never in the checkpoint at all (non-persistent RoPE `inv_freq` /
SigLIP `position_ids` buffers, computed from config at construction time; and
PaliGemma's `embed_tokens.weight`, tied to and dropped from the checkpoint in
favor of `lm_head.weight`) — `load_pytorch` repairs these explicitly after the
assign-load and raises if anything is still on the meta device
afterward, rather than silently running with garbage weights. The loaded weights are identical to the
default path, and `policy.infer()` with the same observation and injected noise produces identical actions.

Pass `--low-cpu-mem-usage` to `serve_policy.py` to use it:

```bash
cd n0-vtla
python scripts/serve_policy.py \
  --policy.config=vtla_tactile_posttrain \
  --policy.dir=<path to the checkpoint step directory> \
  --low-cpu-mem-usage
```

The flag defaults to off (higher peak RAM). It only applies to `N0VTLAConfig`-based
configs (e.g. `vtla_tactile_posttrain`); passing it with a plain-`Pi0Config`
config raises an error instead of silently doing nothing.

To set it programmatically instead (e.g. in your own script calling
`create_trained_policy` directly, not through `serve_policy.py`), it's a
field on the frozen model-config dataclass:

```python
import dataclasses
train_config = dataclasses.replace(
    train_config, model=dataclasses.replace(train_config.model, low_cpu_mem_usage=True)
)
```

This starts a websocket server (`n0vtla/serving/websocket_policy_server.py`,
default port 8000) that loads the checkpoint once and serves inference
requests. It reads `norm_stats.json` from the checkpoint's own `assets/`
directory (not the repo's `assets/` dir, and not `tactile_norm.json`
— see §4.2 for why those are two different files), so the checkpoint
directory must be complete (`model.safetensors`, `metadata.pt`, `assets/`).

**Asset-id fallback:** the `vtla_tactile_posttrain` preset
config's own default asset_id (`canonical_tactile_task`) usually does not match what
a checkpoint's `assets/` directory contains, because a training run sets `VTLA_ASSET_ID` at
launch time. `serve_policy.py`'s `create_policy` therefore falls back to the
checkpoint's own asset directory whenever the preset's asset_id isn't present
there and exactly one asset directory exists (logged at INFO level when it fires).
If a checkpoint's `assets/` contains
more than one directory, the fallback does nothing and you need to pass
`norm_stats` explicitly by calling `create_trained_policy` yourself, since
which one is correct is ambiguous.

Run `python scripts/gate_c_check.py` first if you've changed anything about the
model assembly or tactile path (DEPLOY.md) — a compatibility regression test
against a known-good state.

## 3. Client connection

Use `n0vtla_client.websocket_client_policy.WebsocketClientPolicy`:

```python
from n0vtla_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="<server-ip>", port=8000)
policy.reset()              # call once per episode/rollout attempt -- see §4.2 (tactile baseline)
result = policy.infer(obs)  # obs: dict, see §4. result: dict, see §5.
```

`infer()` blocks on one round trip (msgpack over websocket) and returns the
full response dict. There is no batching — one call per observation.

## 4. Observation dict (what you send)

Taken from `CanonicalTactileInputs.__call__`
(`n0vtla/policies/canonical_tactile_policy.py`), the transform that
actually consumes this dict — `create_trained_policy` does **not** run any
repack step by default, so these are the literal keys the model-facing
pipeline expects, not the raw canonical LeRobot column names of the training
data (they happen to be nearly the same, which is intentional but not
guaranteed by any other layer).

| Key | Required | Shape / dtype | Notes |
|---|---|---|---|
| `observation.state` | **yes** (raises `ValueError` if absent) | `float32[32]` | See §4.1 for the layout. This is the **last commanded** arm+hand target, not measured proprioception — matches the training contract. |
| `observation.image.third_view` | recommended | `HWC uint8` RGB, any resolution | Static head camera. Resized to 224×224 internally (`ResizeImages`, `n0vtla/transforms.py`). Absent → zero placeholder, `image_mask` false for that view (the model was trained with this view always present in training, so don't omit it in practice). |
| `observation.image.right_wrist_view` | recommended | `HWC uint8` RGB, any resolution | Wrist (D405) camera. Same resize/placeholder behavior. |
| `observation.image.left_wrist_view` | omit | — | A single-arm setup has no left arm; leave absent (the model is trained with this view masked absent for single-arm data). |
| `observation.image.right_wrist_right_tactile` | recommended | `float32[2,H,W,3]` or `uint8[2,H,W,3]` stack `[baseline, current]` (or a single `HWC` frame — see §4.2) | Tactile pressure heatmap, **not raw sensor data** — see §4.2 for the required encoding. |
| `prompt` | optional | `str` | Defaults to `"Perform the task."` if omitted (`InjectDefaultPrompt`). Keep it fixed to this string — post-training uses the config's default prompt (`POST_TRAIN.md` §1.5). |
| `action` / `actions` | **must be absent** | — | If present, the relative-action transform in the same transform chain will try to subtract state from it and raise a shape error — it expects a full `(horizon, 32)` ground-truth chunk, which doesn't exist at inference time. Intentional: in `canonical_tactile_policy.py`, "Actions are the training target and are absent at inference time." |

### 4.1 `observation.state` layout (32-D)

Same layout as training (`right_eef_mm_columns6d_10_19_revo2_raw6_20_26_v1`):

| Indices | Meaning | Units |
|---|---|---|
| `[0:10]` | unused (reserved) | zero |
| `[10:13]` | right arm EEF xyz | mm |
| `[13:19]` | right arm orientation, rot6d (first two columns of the rotation matrix) | — |
| `[19]` | unused | zero |
| `[20:26]` | 6 Revo2 hand motor targets, in the same order as the hand target in your recorded data (`action/right/hand_target`) | raw motor units, 0-1000 |
| `[26:32]` | unused (reserved) | zero |

This is the **last commanded** state — whatever arm/hand target you most
recently sent to the robot, not something read back from encoders. Convert
your live orientation reading to rot6d with
`n0vtla.policies.rotation_utils` (the same utility
`scripts/robot_single_arm_adapter.py:decode_commands` uses to go the other
direction) — don't hand-roll the axis-angle-to-rot6d conversion; get the
column-vs-row convention right by using the shipped utility, not by matching
round-trip output to your own inverse (a self-consistent
round-trip test does not catch a systematic convention bug).

### 4.2 Tactile encoding — this is not raw sensor data

The model was trained on a **grayscale pressure-heatmap image**, not raw taxel
readings. Reproducing this at inference means, per taxel-pad reading:

1. Normalize: `(raw - baseline[pad]) / scale[pad]`, clipped to `[-1, 8]`
   (`itw_pressure.normalize_pressure`, reused as-is by
   `n0vtla/policies/canonical_tactile_policy.py`'s pipeline). Use the
   **per-pad `normal_baseline` / `normal_scale` from the tactile normalization
   JSON fitted on your own training split** (restricted to each
   episode's own trial window — `scripts/fit_robot_tactile_norm.py`), **not** human-corpus
   stats and not a fresh fit on your own live readings — the checkpoint's
   `norm_stats.json` and this pad-level file are two different things that
   both need to match training.
2. Map to grayscale: `gray = round((clip(normal,-1,8) + 1) * 255/9)`, replicate
   to 3 channels (`itw_pressure.pressure_rgb`).
3. Lay out all 15 pads into one 224×224 canvas at their fixed positions
   (`itw_tactile_adapter.TACTILE_SLOT_LAYOUT`, consumed via
   `itw_pressure._put_resized`) — a hand-shaped diagram, not a raster of the
   physical sensor grid.
4. **Check which sensor channel is live.** In the raw recording layout read by the
   converters, the glove file names can be crossed relative to the robot hand
   (see `POST_TRAIN.md` §2); the builders map each file to the hand it is physically on
   by signal level (`robot_single_arm_adapter.resolve_episode_window`: whichever channel has
   `std >= 1e-4` is live). Your live sensor SDK's channel ordering may differ — check it
   directly rather than assuming the same swap applies.

**Baseline frame — read this carefully, it is not "the previous frame":**
Per training (`n0vtla/training/config.py`, `_LATENT_BASELINE_FRAMES =
100_000`, deliberately larger than any episode so the LeRobot delta-timestamp
loader clamps it to frame 0), the "baseline" tactile frame is the
**first frame of the episode**, paired with the current frame at every
subsequent step — not a rolling window, not the immediately-preceding frame.
Reproduce this exactly as `scripts/serve_zmq.py`'s own docstring specifies for
its (different, Flexiv-family) config, because the mechanism is identical:

> `reset` clears the stored baseline. Call it at the START of every episode.
> `predict` on the first call after a reset captures the current tactile as the
> baseline (so frame 0 has `tac_t == tac_0` → zero contact, matching
> training); every subsequent predict pairs the live tactile (`tac_t`) with
> that stored baseline (`tac_0`).

Concretely: send `observation.image.right_wrist_right_tactile` as a
`(2, H, W, 3)` stack `[baseline_frame, current_frame]` where `baseline_frame`
is the **same array, held fixed**, captured on your first `infer()` call after
`reset()`. If you'd rather send a single current-frame-only image (3D array,
no stack), `CanonicalTactileInputs` accepts that too, but then the baseline
slot is masked absent (`image_mask` false) and the model's contact signal
degrades to "no baseline available" — not what training saw. Get the stacked
version working before trusting any tactile-conditioned behavior.

`scripts/serve_zmq.py` is a complete reference implementation of this exact
reset/baseline pattern (for a sibling config family, not this one — its image
keys and state layout don't apply here, but its control flow does). Read it
before writing your own hardware-facing serve wrapper; don't reinvent the
reset semantics from scratch.

## 5. Response dict (what you get back)

`Policy.infer()` (`n0vtla/policies/policy.py`) returns:

```python
{
    "state": <your input state, echoed back>,
    "actions": np.ndarray,  # shape (horizon=50, 32), ABSOLUTE physical units
    "policy_timing": {"infer_ms": float},
}
```

`actions` is already fully denormalized and delta-inverted (`Unnormalize` +
`AbsoluteActions`, part of the policy's output transform chain) — decode it
with the **same layout as §4.1** (`[10:13]`=xyz mm absolute, `[13:19]`=rot6d
absolute, `[20:26]`=hand motor targets absolute 0-1000). Convert rot6d back to
whatever your arm controller wants (axis-angle, quaternion, ...) with
`n0vtla.policies.rotation_utils.rot6d_to_matrix`, the same function
`robot_single_arm_adapter.decode_commands` uses — again, don't hand-roll this
conversion.

**Execute the full 50-step chunk before requesting the next prediction**
(DEPLOY.md) unless you deliberately implement receding-horizon replanning —
the model wasn't trained with a different replanning cadence in mind.
Receding-horizon replanning instead of blindly executing the entire chunk is a
deliberate real-time control decision for whoever integrates this, not something
this contract prescribes either way.

## 6. Before commanding the real arm — safety gates

This document only specifies the *data contract*. It is not a green light to
move the robot. Recommended pre-motion checklist: physical-unit held-out action
error (`scripts/compute_posttrain_held_out_loss.py`), smoothness/command-limit checks, causal sensor latency, source-action
replay through the exact deployed decoder — **before** any robot motion — and
then supervised low-speed deployment only, with workspace/joint/motor limits,
velocity/acceleration caps, stale-observation rejection, a watchdog, and an
accessible emergency stop. None of that is implemented by anything in this
repo or this document; it is rig-specific and must be built by whoever
operates the real robot.

## References

| Topic | File |
|---|---|
| Server entry point | `scripts/serve_policy.py` |
| Compatibility regression test | `scripts/gate_c_check.py` |
| Client library | `n0vtla_client/websocket_client_policy.py`, `n0vtla_client/base_policy.py` |
| The actual input contract (read this, not this doc, if they disagree) | `n0vtla/policies/canonical_tactile_policy.py` |
| Reference reset/baseline serve pattern (different config family, same mechanism) | `scripts/serve_zmq.py` |
| rot6d conversion utilities | `n0vtla.policies.rotation_utils` |
| Tactile normalization/encoding | `scripts/itw_pressure.py` (`normalize_pressure`, `pressure_rgb`), `scripts/itw_tactile_adapter.py` (`TACTILE_SLOT_LAYOUT`, `_put_resized`) |
| Data format and conversion | `POST_TRAIN.md` |
| Mid-training (how the tactile branch is trained) | `MID_TRAIN.md` |
