# Real-Robot Inference: Dual-Arm Policy — README for deployment

For whoever wires a real dual-arm rig (two arms, each with a dexterous hand, a static head camera, a camera on
each wrist, and a tactile glove on each robot hand; the reference configuration is two UFACTORY xArm6 arms with BrainCo
Revo2 hands) to a dual-arm post-trained checkpoint. It is self-contained; the single-arm `REAL_ROBOT_INFERENCE.md` is
background (same server, same safety caveats). What is **not** provided: any robot driver, camera/glove capture, or
safety layer.

## 0. Status: what the repo checks, and what it does not

Offline checks available in this repo (they run on recorded training data, so they test the data contract, not
generalization; real-robot rollouts are the evaluation):
- **Server + client end to end**: start `scripts/serve_policy.py` with the command in section 3 and replay a recorded
  training episode through the websocket with `scripts/dual_arm_client_example.py` (native-resolution RGB, raw taxel
  readings encoded live, both arms decoded). Indicative steady-state latency is about 207 ms per call on one H200
  GPU. This is a plumbing check, not a benchmark.
- **Tactile encoding**: encoding raw taxel readings live reproduces the frames stored in the training videos up to
  video-compression noise.
- **Decode chain** on random training frames (`scripts/check_dual_policy_replay.py`): predicted rotations should be
  valid, hand targets should stay inside 0-1000, and position/rotation/hand errors should be below the "hold the
  current command" baseline for both arms.

Not verified: any run on a real robot or with real sensors; the rot6d/axis-angle convention against your controller's
(only self-consistent round trips were tested); VRAM and latency on your GPU; behaviour under your camera/glove
clocks.

## 1. What to get and where it goes

**Code.** This repository (`n0-vtla/`). The serving package (`n0vtla/`, `n0vtla_client/`, `scripts/serve_policy.py`) is the
one the checkpoints are trained with. `pip install -r requirements.txt`.

**Checkpoints and normalization files.** Paths are relative to `n0-vtla/` (the original and merged checkpoints sit in the `checkpoints/`
folder of the TouchScale checkout, i.e. `../checkpoints/`):

| what | location |
|---|---|
| original N0-VTLA checkpoint | `../checkpoints/n0-vtla-base` (`hf download NeoteAI/n0-vtla-base --local-dir ../checkpoints/n0-vtla-base`) |
| TouchScale mid-train checkpoint (trainable-only delta, written by `train_stage1.sh`) | `checkpoints/vtla_stage1_predictor_pretrain/stage1_online/<step>` |
| merged initialisation for post-training (made by `scripts/posttrain.sh` with `INIT=touchscale`) | `../checkpoints/n0-vtla-base_plus_stage1_online_<step>` |
| your post-trained checkpoint (`scripts/posttrain.sh`) | `checkpoints/vtla_tactile_posttrain/<exp name>/<step>/` |

A post-trained step folder contains `model.safetensors`, `metadata.pt`, `assets/<asset id>/norm_stats.json` and
`optimizer.pt` (not needed for inference).

**Normalization files that deployment needs (two, produced at data-conversion time):**

| file | produced by | read by |
|---|---|---|
| state/action `norm_stats.json` | `scripts/compute_canonical_norm.py` (`--robot canonical_dual_arm`) | **the server**, from `<checkpoint>/assets/<asset id>/norm_stats.json` (training copies it into every checkpoint) |
| tactile normalization JSON (30 pad baselines/scales) | `scripts/fit_robot_dual_tactile_norm.py` | **your client**, to encode live glove readings exactly like the training videos; the server never reads it. Keep it next to the checkpoint, e.g. `<checkpoint>/tactile_norm_dual.json` |

Neither file is shipped with this repository: both are fitted from your own training split.

## 2. Hardware

One GPU with 16 GB or more, bf16, no multi-GPU. Memory/latency numbers of the single-arm document do not
carry over (three cameras and two tactile views make a longer prefix); measure on your machine. Indicative latency
is about 207 ms per call on one H200 GPU; VRAM was not recorded for this configuration.

## 3. Start the server

```bash
python scripts/serve_policy.py --policy.config=vtla_tactile_posttrain \
  --policy.dir=<checkpoint step dir> --low-cpu-mem-usage        # websocket, port 8000
```

`--low-cpu-mem-usage` avoids a ~17 GB host-RAM spike while loading. Do not set `VTLA_ASSET_ID`; the server
finds the checkpoint's single asset directory itself.

## 4. Client

Use `n0vtla_client.websocket_client_policy.WebsocketClientPolicy`. **`scripts/dual_arm_client_example.py` is the
reference**: `DualArmClient.reset()/infer()` build the observation below, `encode_tactile()` does the tactile
encoding, `decode_actions()` decodes the reply. Run it against a server to see the whole loop:

```bash
python scripts/dual_arm_client_example.py --package <recorded episode dir> \
  --norm <checkpoint>/tactile_norm_dual.json --port 8000
```

### 4.1 Observation (one `infer()` per call; send **no** `action`/`actions` key)

| key | content |
|---|---|
| `observation.state` | float32[32], section 4.2 |
| `observation.image.third_view` | head camera, HWC uint8 RGB, any resolution |
| `observation.image.left_wrist_view` | wrist camera on the LEFT arm |
| `observation.image.right_wrist_view` | wrist camera on the RIGHT arm |
| `observation.image.left_wrist_left_tactile` | glove on the robot LEFT hand: `(2, 224, 224, 3)` uint8 `[baseline, current]` |
| `observation.image.right_wrist_right_tactile` | glove on the robot RIGHT hand: same shape |
| `prompt` | optional; default and training value is `"Perform the task."` — keep it |

Images are letterboxed to 224x224 inside the pipeline (training data was letterboxed the same way). An omitted
view becomes a zero placeholder the model never saw in training, so send all three RGB views. Restore the camera
placements (head camera and both wrist cameras) to what the training data used; a rotated or shifted camera is a
distribution shift the policy was not trained on.

**Tactile baseline.** The model uses `current - baseline`. The baseline is the **first frame after `reset()`**
for that hand, held fixed for the whole rollout (one baseline per hand): start with the hands at rest, call
`reset()`, and keep the first encoded frame as the baseline. A single 3-D frame is accepted but masks the baseline.

**Tactile encoding** (`encode_tactile()`): per pad `(raw - baseline[pad]) / scale[pad]` clipped to [-1, 8],
grayscale `round((x + 1) * 255 / 9)`, 15 pads placed on a 224x224 canvas (`itw_pressure`), left hand mirrored.
Use your `tactile_norm_dual.json` (slots 0–14 = robot LEFT hand, 15–29 = robot RIGHT hand; fitted per
hand because the two gloves can respond very differently). In the raw recording layout read by the converters the
files are **crossed**: the glove on the robot right hand is stored as `left_hand_data.npz`, the one on the robot left
hand as `right_hand_data.npz` (`POST_TRAIN.md` §2). The model-side mapping is anatomical (robot right hand →
`right_wrist_right_tactile`, `hand="right"`); check the channel order of your own sensor SDK directly, the crossing
is a property of that raw layout, not necessarily of your live sensors.

### 4.2 State (`observation.state`, float32[32]) and the layout of the reply

| dims | content | units |
|---|---|---|
| 0–2 | LEFT arm EEF xyz, in the left arm's base frame | mm |
| 3–8 | LEFT arm rot6d (first two columns of the rotation matrix, flattened column-wise) | – |
| 9 | left gripper, unused | 0 |
| 10–12 | RIGHT arm EEF xyz, in the right arm's base frame | mm |
| 13–18 | RIGHT arm rot6d | – |
| 19 | right gripper, unused | 0 |
| 20–25 | RIGHT Revo2 hand, 6 motor targets (order of the hand `target` in the recording) | 0–1000 |
| 26–31 | LEFT Revo2 hand, 6 motor targets | 0–1000 |

- State is the **last commanded** target, not encoder readback. Before an arm's first command use its measured
  pose; an arm that is not currently engaged holds its last command (that is what the training data does).
- Axis-angle (degrees) to rot6d: `robot_dual_arm_adapter.pack_dual`. Use it and `n0vtla.policies.rotation_utils`; do not
  hand-roll the convention.
- Each arm's pose is in its own base frame. The model has no shared frame; in the reference rig the bases sit side by
  side, the left arm about 600 mm to the left of the right arm (approximate, not calibrated), so keep your arm placement
  comparable to your training data.

### 4.3 Reply

`actions`: float32[50, 32], **absolute** physical units (already denormalized; relative-to-state targets restored
for both arms). Decode with the same layout as the state: `decode_actions()` or
`robot_dual_arm_adapter.decode_dual(actions)` → per arm xyz mm and axis-angle degrees (in that arm's own base frame),
per hand 6 motor targets (clip to 0–1000). Send each arm and each hand **its own** targets. 50 steps are 1.67 s
at 30 Hz; execute the chunk or replan receding-horizon, the model was not trained for a specific cadence.
The model has no clutch input; engagement/hold logic belongs to your controller.

## 5. Before moving anything

None of the safety layer exists in this repo. Before the first motion: workspace/joint/motor limits per arm,
velocity and acceleration caps, a collision margin between the two arms and the work surface (pressing on a fixture can
trigger controller collision stops), stale-observation rejection, a watchdog and an accessible emergency stop; run
supervised at low speed first. Replay a recorded episode through the exact deployed decoder (the example client does
this) before commanding the robot.

## 6. Data-processing caveats that shape what the policy learned

- Recordings are cut into episodes at short gaps in the command stream; the tactile baseline during training is the
  first frame of each piece, so some training frames have a baseline that already contains contact, while a rollout
  starts from rest. The effect on behaviour is unmeasured; if grasp-force or "is it gripped" behaviour looks unreliable,
  suspect this first. `build_robot_dual_dataset.py --bridge-gap-rows` reduces the cutting.
- Cross-host clocks (cameras/gloves vs. robot) are typically NTP-disciplined, not hardware-synchronized.
- If the training data contains only successful demonstrations, no failure recovery is learned.
- The model has no clutch input and an arm that was not engaged during recording simply held its last command.

See `POST_TRAIN.md` for the data format and every conversion step.

## 7. Files

| | |
|---|---|
| server | `scripts/serve_policy.py` |
| reference client | `scripts/dual_arm_client_example.py` |
| decode / layout helpers | `scripts/robot_dual_arm_adapter.py` (`pack_dual`, `decode_dual`), `n0vtla.policies.rotation_utils` |
| tactile normalization | `scripts/fit_robot_dual_tactile_norm.py` |
| decode-chain check (GPU) | `scripts/check_dual_policy_replay.py` |
| server+client check | start `serve_policy.py`, then run `scripts/dual_arm_client_example.py` |
| actual input contract (read this if it disagrees with this document) | `n0vtla/policies/canonical_tactile_policy.py` |
