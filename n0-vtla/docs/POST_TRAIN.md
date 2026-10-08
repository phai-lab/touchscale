# Post-training on your own robot data

This document is the single reference for post-training N0-VTLA: the input format the model requires, the raw robot recording layout the converters read, the conversions between the two, and the exact scripts and order to run. (The original release's generic notes are in `POST_TRAINING.md`; the TouchScale mid-training stage is in `MID_TRAIN.md`.)

Post-training (`train.sh`, config `vtla_tactile_posttrain`) reads a **canonical LeRobot-style dataset**. Our robot recordings are not in that format, so a conversion step sits between them. You can start post-training from either

- `INIT=original`: the original N0-VTLA checkpoint, or
- `INIT=touchscale`: the original checkpoint merged with the **mid-training delta** produced by `train_stage1.sh`
(so mid-training must have been run first; see `MID_TRAIN.md`).

Both use the same recipe and data; only the initial weights differ.

```
raw robot recordings (per episode: H5 + mp4 + glove npz + robot logs)
   (0, INIT=touchscale only) bash train_stage1.sh      -> checkpoints/vtla_stage1_predictor_pretrain/<EXP_NAME>/<step>/
   1. tactile normalization fit (train split only)     scripts/fit_robot_tactile_norm.py | fit_robot_dual_tactile_norm.py
   2. canonical dataset (parquet + 224x224 mp4)         scripts/build_robot_dataset.py | build_robot_dual_dataset.py
   3. state/action normalization statistics             scripts/compute_canonical_norm.py   -> assets/<config>/<asset id>/norm_stats.json
   4. post-train                                        scripts/posttrain.sh  (INIT=original | touchscale)
   5. deploy                                            scripts/serve_policy.py + client (docs/REAL_ROBOT_INFERENCE*.md)
```

## 1. What N0-VTLA requires

### 1.1 Dataset layout (one directory per split; the loader reads every episode under the root)

```
<dataset_root>/
  meta/info.json            fps=30, feature list (below), robot_type
  meta/episodes.jsonl       {"episode_index", "tasks": [<text>], "length"}
  meta/episodes_stats.jsonl per-episode quantile stats (written by the builders)
  meta/tasks.jsonl          {"task_index": 0, "task": <text>}
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<image key>/episode_000000.mp4      H.264, yuv420p, 224x224, 30 fps, one frame per parquet row
```

The loader has no notion of a split tag: build **one directory per split** and point `VTLA_DATASET_PATH` at the train
one only.

### 1.2 Per-frame parquet columns (30 Hz, one row per frame)


| column                                                             | type            | meaning                                                                                   |
| ------------------------------------------------------------------ | --------------- | ----------------------------------------------------------------------------------------- |
| `observation.state`                                                | float32[32]     | canonical state, section 1.3                                                              |
| `action`                                                           | float32[32]     | canonical **absolute** command at this frame (the loader stacks the next 50 into a chunk) |
| `action_mask`                                                      | bool[32]        | which dims are real                                                                       |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | float32 / int64 | standard LeRobot bookkeeping; `timestamp = frame_index / 30`                              |




### 1.3 The canonical 32-dim state/action layout (`n0vtla/policies/canonical_schema.py`)


| dims               | content                                                                                       | units       |
| ------------------ | --------------------------------------------------------------------------------------------- | ----------- |
| 0-2                | left arm EEF xyz                                                                              | mm          |
| 3-8                | left arm orientation, rot6d = first two columns of the rotation matrix, flattened column-wise | -           |
| 9                  | left gripper (unused with Revo2 hands)                                                        | 0           |
| 10-12 / 13-18 / 19 | right arm xyz / rot6d / gripper (unused)                                                      | mm / - / 0  |
| 20-25              | right Revo2 hand, 6 motor targets (raw, 0-1000)                                               | motor units |
| 26-31              | left Revo2 hand, 6 motor targets                                                              | motor units |


Single-arm data fills only the right arm (10-18) and right hand (20-25); the left blocks stay 0 and are masked.
Dual-arm data fills everything except the grippers. **State is the command in force at that frame** (last issued
target), not encoder readback; `action[t]` is the command at `t`, and the loader builds the 50-step chunk
`action[t..t+49]` (repeating the last row past the episode end). The model predicts, and deployment returns,
absolute commands.

At train time the transforms (`n0vtla/training/config.py`, `LeRobotCanonicalTaskTactileDataConfig`) make the
chunk relative to the current state: `ChunkDeltaToCurrentState` - xyz `action - state`, rot6d as a relative rotation
`R_action @ R_state^T`, hand targets stay absolute; `RelRotAbsoluteActions` inverts it at inference. An arm is only
transformed if its xyz dims are True in `action_mask`.
*Difference from the original N0-VTLA:* original repo used element-wise `DeltaActions` on the rot6d dims too.

### 1.4 Image and tactile streams (keys are fixed canonical names)


| key (`observation.image.*`)                                                 | content                                                                       |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `third_view`                                                                | static head camera RGB                                                        |
| `left_wrist_view`, `right_wrist_view`                                       | wrist cameras RGB                                                             |
| `left_wrist_left_tactile`, `right_wrist_right_tactile`                      | tactile glove on the left / right hand, rendered as an RGB **pressure image** |
| `second_third_view`, `left_wrist_right_tactile`, `right_wrist_left_tactile` | exist as slots; leave absent if you have no such sensor                       |


A missing key is allowed: the pipeline substitutes a zero image with its mask false. Frames are stored **letterboxed to 224x224** (aspect-preserving resize, black padding), so the video frame `i` belongs to parquet row `i`.

**Tactile as a video.** The glove's raw taxel readings are converted to a 224x224 image per frame (section 3.3). The
model consumes `current - baseline`, where the baseline is the **first frame of the episode** (the loader fetches
frame 0 for every step), so an episode should start with the hands at rest. A tactile key that has a video gets
`[baseline, current]`; a tactile key without a video is masked, like any missing view.

### 1.5 Normalization inputs


| file                                                                                  | produced by                         | used for                                                                                                                                               |
| ------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `assets/<config name>/<asset id>/norm_stats.json` (state and action mean/std/q01/q99) | `scripts/compute_canonical_norm.py` | quantile-normalizing state and the (relative) action chunk in training and inference. `<asset id>` = `VTLA_ASSET_ID`                                   |
| pad30 tactile normalization JSON (30 baselines/scales/thresholds)                     | `scripts/fit_*tactile_norm.py`      | turning raw glove values into the pressure image; **not** read by the model, but needed again at deployment to encode live glove readings the same way |


Statistics are fit on the **train split only**. The prompt is the config's `default_prompt` (`VTLA_DEFAULT_PROMPT`,
default `"Perform the task."`); `tasks.jsonl` text is stored but not used unless `prompt_from_task` is enabled.

## 2. Raw robot recording layout

The converters read one folder per recorded episode (uuid), as written by a teleoperation stack with UFACTORY xArm6
arms and BrainCo Revo2 hands (one or two arms; static head RGB camera, a wrist RGB camera per arm, a tactile glove on
each robot hand):

```
<uuid>/
  episode_30hz.h5          30 Hz time grid + validity masks (see below)
  rgb_head.mp4/.csv, wrist_right.mp4/.csv [, wrist_left.mp4/.csv]   full-resolution video + per-frame timestamps
  left_hand_data.npz, right_hand_data.npz    glove: timestamps + tactile_<pad> arrays (T, h, w) for 15 pads
  robot/manifest.json      task name, trial engaged/released host timestamps, clock domains
  robot/arm_cmd.jsonl, robot/hand.jsonl      (single arm)   issued arm targets, hand targets
  robot/left/..., robot/right/...            (dual arm)     same, one folder per arm
```

Other files in the folder are ignored.

Key facts about the content:

- **H5** (`episode_30hz.h5`): `time/timestamp_ns` (the 30 Hz grid),  `obs/robot/<side>/{tcp_pos,tcp_quat,joint_pos,hand_pos}` (measured, mm / quaternion w,x,y,z), `action/<side>/{arm_target_aa,hand_request_target,clutch}`, `obs/video/<camera>/src_idx` (frame of the mp4 nearest to each grid row) and `valid/<camera>` masks.
- **Commands** are in the robot logs: arm target = xyz mm + axis-angle degrees, hand target = 6 motor values 0-1000. Each
arm has its own **clutch**: the arm only follows the operator while engaged, otherwise it holds its last command.
- **Poses are in each arm's own base frame.** In the reference dual-arm rig the bases are side by side, about 600 mm
apart (approximate, not calibrated).
- **Gloves are stored under the opposite name** in this layout: the glove on the robot RIGHT hand is `left_hand_data.npz`, the one on the robot LEFT hand is `right_hand_data.npz` (in single-arm recordings the other file is simply dead). The builders check the file's signal level and map it to the robot hand it is physically on.
- Clocks of the camera/glove computer and the robot computer are NTP-disciplined, not hardware-synchronised.

## 3. What the conversion does

### 3.1 Which rows are usable (`resolve_episode_window` / `resolve_dual_window`)

A 30 Hz grid row is kept only if **all** of these hold (thresholds in `robot_single_arm_adapter.py`,  `robot_dual_arm_adapter.py`):


| condition                                                            | threshold                            |
| -------------------------------------------------------------------- | ------------------------------------ |
| inside the trial window of `robot/manifest.json` (engaged..released) | -                                    |
| the engaged arm's clutch is on (dual arm: at least one arm engaged)  | -                                    |
| an accepted arm command already exists and is fresh                  | issued at most 150 ms before the row |
| a hand target already exists and is fresh                            | at most 150 ms                       |
| a glove sample exists and is fresh                                   | at most 50 ms                        |
| each camera frame is valid in the H5 and close to the row            | `valid/<cam>` true and `             |


Only causally available data is used (the newest command at or before the row). **Dual-arm hold rule:** an arm that is not
engaged on a row keeps its last issued command (or its measured start pose before its first command), because that is
what the robot physically does; no zero placeholders are written as targets.

### 3.2 Episodes, windows and gaps

Consecutive kept rows form a run; **each run of at least 50 rows becomes one dataset episode** (shorter runs are
dropped), so one recording can yield several episodes. Because a single stale command row cuts a run, the dual-arm
builder has `--bridge-gap-rows N` (`dual_gap_bridge.py`): gaps of at most N rows between kept rows, with at least one
arm engaged, are filled with the command already in force. `scripts/analyze_dual_bridging.py` shows how many episodes and frames each N gives. Cutting also moves the tactile baseline to the first frame of each piece, which differs from deployment (baseline = first frame of the rollout).

### 3.3 Value conversions


| quantity        | raw                                      | canonical                                                                                                                                                                                                                                   |
| --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| arm position    | mm                                       | unchanged                                                                                                                                                                                                                                   |
| arm orientation | axis-angle, degrees                      | rotation matrix, first two columns flattened column-wise -> rot6d (`pack_commands`, `pack_dual`)                                                                                                                                            |
| hand            | 6 motor targets 0-1000                   | unchanged (checked to be inside 0..1000)                                                                                                                                                                                                    |
| side -> slot    | right arm A / left arm B                 | dims 10-19 / 0-9; right hand 20-25, left hand 26-31                                                                                                                                                                                         |
| RGB             | native resolution mp4                    | frames picked by `src_idx`, letterboxed to 224x224, re-encoded H.26                                                                                                                                                                         |
| glove           | 15 pads of taxel grids (`tactile_<pad>`) | per pad `clip((raw - baseline[pad]) / scale[pad], -1, 8)`, gray = `round((x+1)*255/9)`, the 15 pad images pasted on a 224x224 canvas at fixed hand-shaped positions (`TACTILE_SLOT_LAYOUT`, mirrored for the left hand), written as a video |


Round trips are checked: the single-arm builder records position/rotation/hand round-trip errors (`decode_commands`) in
`meta/audit.json`; for dual arm `verify_robot_dual_dataset.py` repeats it with `decode_dual`.

### 3.4 Normalization fits

- **Tactile** (`fit_robot_tactile_norm.py`, `fit_robot_dual_tactile_norm.py`): per pad, baseline = 5th percentile and scale
= 99.9th percentile minus baseline over frames sampled from each training episode's own usable rows; the dual-arm
version fits each hand separately (the two gloves respond very differently). Slots 0-14 = left hand, 15-29 = right hand.
- **State/action** (`compute_canonical_norm.py`): the action chunk is made relative the same way as in training  
(`--robot canonical_single_arm` for one arm, `canonical_dual_arm` for both arms; the plain `flexiv`/`aloha` modes do element-wise  
deltas), then mean/std/quantiles are accumulated; constant dims (padding, grippers) get identity statistics.

### 3.5 Splits

Builders take a split manifest with `train`, `val`, `block_holdout_v1.holdout_episodes` and `block_of_episode`
(`build_robot_dataset.py::split_uuids`). `scripts/convert_release_split.py` turns a plain split file
`{"train": [uuid, ...], "validation": [uuid, ...], "test": [...]}` (your own
train/validation split, `test` optional) into that manifest. The dual-arm builder and tactile fit read such a file (`train` list)
directly. Fits and statistics always use the train list only. No split files are shipped with this repository.

## 4. How to run (step by step)

Notation: `RAW` = folder with one sub-folder per recorded episode, `OUT/train` = the canonical dataset you are creating.

**Step 0, only for** `INIT=touchscale`**.** Run mid-training once (`MID_TRAIN.md`): `bash train_stage1.sh`. It writes the
trainable-only delta to `checkpoints/vtla_stage1_predictor_pretrain/<EXP_NAME>/<step>/` (default `EXP_NAME=stage1_online`), which is exactly where `posttrain.sh` looks for it. Download the original checkpoint once as well: `hf download NeoteAI/n0-vtla-base --local-dir ../checkpoints/n0-vtla-base`.

**Steps 1-3, convert and compute statistics.** Single arm (layout with `robot/arm_cmd.jsonl`, `robot/hand.jsonl`; by
default only episodes whose `robot/manifest.json` has `trial.label == "success"` are used, `--skip-success-label-check` disables this):

```bash
python scripts/fit_robot_tactile_norm.py RAW SPLIT.json tactile_norm.json [--task-name <manifest task.name>] [--skip-success-label-check]
python scripts/build_robot_dataset.py RAW SPLIT.json tactile_norm.json OUT/train --which train --workers 16 \
    [--task-name <manifest task.name>] [--task-description "<instruction>"] [--skip-success-label-check]
python scripts/compute_canonical_norm.py --repo-id OUT/train --robot canonical_single_arm \
    --train-config-name vtla_tactile_posttrain --asset-id my_asset --train-only
```

Dual arm:

```bash
python scripts/fit_robot_dual_tactile_norm.py RAW training_split.json tactile_norm_dual.json
python scripts/build_robot_dual_dataset.py RAW training_split.json tactile_norm_dual.json OUT/train \
    --workers 16 --bridge-gap-rows 8
python scripts/compute_canonical_norm.py --repo-id OUT/train --robot canonical_dual_arm \
    --train-config-name vtla_tactile_posttrain --asset-id my_asset --train-only
python scripts/verify_robot_dual_dataset.py --dataset OUT/train --asset-id my_asset   # read-back through the real pipeline
```

`--task-name` is an optional safety check (the episode's `robot/manifest.json` task name must equal it) and
`--task-description` is the text stored in `tasks.jsonl`; the model is trained and served with the config's default
prompt either way (section 1.5). `scripts/visualize_robot_episode.py` renders a preview video of a converted episode.

**Step 4, post-train.**

```bash
# (a) from the original N0-VTLA checkpoint
INIT=original   DATASET=OUT/train ASSET_ID=my_asset bash scripts/posttrain.sh
# (b) from the original checkpoint + the mid-training delta (found automatically; merged on first use)
INIT=touchscale DATASET=OUT/train ASSET_ID=my_asset bash scripts/posttrain.sh
```

`posttrain.sh` sets `VTLA_DATASET_PATH` / `VTLA_ASSET_ID` / `VTLA_PRETRAINED_CHECKPOINT`, and for (b) calls
`scripts/merge_stage1_into_base_checkpoint.py` (latest step of `MIDTRAIN_EXP`, default `stage1_online`; override with
`MIDTRAIN_EXP`, `MIDTRAIN_STEP` or `MIDTRAIN_CKPT`), then runs `train.sh`, so `CONFIG_NAME`, `EXP_NAME`,
`NPROC_PER_NODE`, `CHECK_ONLY` and extra training arguments work as documented there. Checkpoints are written to
`checkpoints/vtla_tactile_posttrain/<EXP_NAME>/<step>/`, each with its `assets/<asset id>/norm_stats.json`.

**Step 5, deployment needs both normalization files.** The state/action `norm_stats.json` is copied into every checkpoint
under `assets/<asset id>/` and read by `scripts/serve_policy.py`; the tactile JSON from step 1 is read by your client
(`encode_tactile()` in `scripts/dual_arm_client_example.py`) to turn live glove readings into the same pressure images the model was trained on. Neither file is shipped: they are statistics of *your* training split, so generate them with the
commands above and keep the tactile JSON next to the checkpoint.

## 5. Adapting to another robot

Keep the output contract of sections 1.1-1.5 and replace the input side: (1) a function that returns, per 30 Hz row, the
issued arm and hand targets, camera frame indices and glove indices (`resolve_episode_window` is the model); (2) `pack_commands` for your end-effector convention (xyz in mm, orientation as rot6d, your hand/gripper values in dims 20-31 or 9/19 and unmasked in `action_mask`); (3) the tactile rendering if your sensor is not a 15-pad glove. The builders, normalization scripts, `train.sh` and the serving code do not change.