# Mid-training on human tactile data

This document describes the **action-free mid-training** stage added on top of N0-VTLA for TouchScale. It learns the
tactile branch of the model from human recordings (RGB + glove pressure, no robot actions), starting from the original
N0-VTLA checkpoint. The result is a small *trainable-only* checkpoint that you can merge back into the base model and then
post-train on robot data (see `POST_TRAIN.md`).

## 1. What is trained

For each sample the model sees three RGB views (head + two wrist cameras), a fixed prompt, zero state tokens, and the
tactile *change* of each hand, `T_t - T_0` (current glove image minus the first frame of the episode). A tactile predictor
(`tactile_kv`, 5 queries, 2 layers, 8 heads) conditioned on the vision-language context has to predict the **future**
tactile change `T_{t+50} - T_t` (50 frames, 1.67 s at 30 Hz):

```
L = L_symmetric_InfoNCE(z, z*) + 0.5 * L1(recon_8x8, target_8x8)
```

- `z` is the predictor output, `z*` the same frozen DINOv2 encoder + trainable projection applied to the future change
(stop-gradient branch). Both are token-mean-pooled and L2-normalised; the logits are the cosine similarities
(`temperature = 1.0`, i.e. exactly the form of the paper's equation); negatives come from the global (all-GPU) batch.
- The reconstruction head maps `z` to an 8x8 coarse map of the averaged future pressure change.
- No action is read, no action suffix is built, and there is no flow-matching loss (`Policy.forward_stage1`,
`Stage1ObservationOnly` provides zero state/action placeholders).

Only three parameter groups are trained (about 123.8 M parameters); everything else (PaliGemma, DINOv2, action expert) is frozen:

```
tactile_encoder.tactile_proj.   tactile_predictor.   tactile_recon_head.
```

Checkpoints therefore contain **only these parameters** (+ optimizer and metadata). Load them on top of a base checkpoint
with `scripts/merge_stage1_into_base_checkpoint.py` (section 5).

Code map: `n0vtla/models_pytorch/n0vtla_policy.py` (`forward_stage1`, `_build_future_target`, `_stage1_infonce_loss`),  `n0vtla/models_pytorch/tactile_recon_head.py`,  `n0vtla/policies/canonical_tactile_policy.py` (`Stage1ObservationOnly`), config `vtla_stage1_predictor_pretrain` in `n0vtla/training/config.py`, trainer `scripts/train_stage1_predictor.py`
(training loop, checkpoint I/O) and `scripts/train_stage1_online.py` (same loop, reads raw recordings directly).

## 2. Input layout (raw recordings)

The online trainer reads recordings in place; nothing has to be converted first.

```
<raw_root>/<date>/<episode>/
    rgb_head.mp4    rgb_head.csv          # csv: frame_index, timestamp_s
    wrist_left.mp4  wrist_left.csv
    wrist_right.mp4 wrist_right.csv
    left_hand_data.npz  right_hand_data.npz    # timestamps + tactile_<pad_id> arrays, (T, h, w)
    task_info.json                              # {"name": "<task label>", ...}
```

- The glove has 15 pads per hand (pad ids `0,1,2,3,4,5,7,8,9,11,12,13,15,16,18`; see `itw_pressure.PAD_IDS`). Only pressure is used, not shear.
- All three RGB streams and both glove streams are put on a common 30 Hz grid over their time intersection (nearest sample;`itw_pressure.aligned_timeline`). An episode is rejected unless it has more than 50 aligned frames and the fraction of frames within the 17.5 ms camera tolerance is high enough.
- RGB is letterboxed to 224x224. Each hand's pressure is normalised (section 3), rasterised onto a fixed 224x224 pad layout (left hand mirrored) and used as a grayscale image `round((clip(x, -1, 8) + 1) * 255 / 9)`.

## 3. Normalization file

Pressure is normalized with one baseline per pad and one force scale **per task** (a light touch and a hard squeeze need different scales). Generate it from your own chosen training set:

```bash
python scripts/build_per_task_scale_normalization.py --raw-root /path/to/raw_root \
    --out per_task_scale.json --workers 16
```

It samples frames, takes the 5th percentile of every pad as its baseline, and the 99.9th percentile of the baseline-
subtracted pressure over all pads of a task as that task's scale (tasks with fewer than `--min-episodes` episodes fall back
to `default_scale`). Fit on training episodes only (`--dates` / `--episode-list`) and keep the file fixed afterwards.
Post-training and deployment on a robot use a separate tactile normalization fitted on the robot gloves
(`POST_TRAIN.md` section 3.4, `REAL_ROBOT_INFERENCE.md`).

## 4. Training

```bash
export VTLA_ITW_RAW_ROOT=/path/to/raw_root
export VTLA_ITW_NORMALIZATION=$PWD/per_task_scale.json
export VTLA_PRETRAINED_CHECKPOINT=$PWD/../checkpoints/n0-vtla-base     # original N0-VTLA weights

CHECK_ONLY=1 bash train_stage1.sh        # environment / data / checkpoint sanity check only
bash train_stage1.sh                      # EXP_NAME defaults to stage1_online
```

Optional variables: `VTLA_ITW_DATES` (comma-separated date folders), `VTLA_ITW_MAX_EPISODES` (quick smoke runs),
`VTLA_STAGE1_FUTURE_OFFSET`, `VTLA_STAGE1_RECON_GRID`, `VTLA_STAGE1_LAMBDA_REC`, `VTLA_STAGE1_TEMPERATURE`,
`VTLA_DEFAULT_PROMPT`, `NPROC_PER_NODE` (default: number of visible GPUs). A fixed episode list can be given with
`VTLA_ITW_EPISODE_LIST_JSON` / `VTLA_ITW_EPISODE_LIST_KEY` (see the docstring of `scripts/train_stage1_online.py`).

Defaults (`vtla_stage1_predictor_pretrain`): global batch 64, AdamW, grad-clip 1.0, 500 warm-up steps, peak LR 1e-4 decaying to 1e-5 over 20 000 steps, bfloat16, checkpoint every 2 000 steps and at the end. `--resume` restores weights, optimizer state and the update counter (not the exact RNG / data cursor). 

Notes: samples whose tactile view or future frame is missing are masked out; with fewer than two valid samples in the global batch InfoNCE is zero and only the reconstruction term trains.

Checkpoints (the trainable-only *delta*) are written under `checkpoints/vtla_stage1_predictor_pretrain/<EXP_NAME>/<step>/`
(relative to `n0-vtla/`). `scripts/posttrain.sh` looks there for the mid-training result.

## 5. Use the result

1. **Post-train from it.** `INIT=touchscale bash scripts/posttrain.sh ...` (see `POST_TRAIN.md`) finds the latest step of
  the run above, merges the delta into the original checkpoint and post-trains from the merged weights. You can also merge manually (post-training loads one complete `model.safetensors`; pointing it at the delta alone would leave the frozen model randomly initialised).
2. **Optional evaluation.** `scripts/eval_stage1_report.py` scores held-out future-tactile prediction (before/after
  mid-training) on a canonical LeRobot-format tactile dataset; `scripts/compute_held_out_tactile_pred_error.py`  
   and the other `scripts/compute_*.py` analyses are described in their docstrings.

## 6. Tests

`tests/test_stage1.py` (loss/gradient/DDP-edge cases, needs the JAX/OpenPI dependencies of the repo),
`tests/test_itw_pressure.py`, `tests/test_itw_online_dataset.py`, `tests/test_stage1_report.py`,
`tests/test_build_per_task_scale.py`.