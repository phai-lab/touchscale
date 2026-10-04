# TouchScale data QC

Automatic quality control for egocentric visual-tactile recordings: a head-mounted
RGB-D camera, two wrist cameras, and a pair of tactile gloves. This is the
pipeline we used to screen TouchScale recordings. It has two stages:

| Stage | Script | What it catches | Cost |
|---|---|---|---|
| 1. Sync gate | `check_sync.py` | missing streams or files, bad cross-sensor time alignment, dropped samples, blackouts | timestamps only, seconds per batch |
| 2. Tactile QC | `run_qc.py` | **tactile loss** (the hand visibly grasps something but its fingers report no force) and **baseline noise** (the hand touches nothing but the glove reports force) | renders a review video and calls a video-capable VLM (Gemini) |

`duration_dist.py` also plots recording lengths for a batch.

## Setup

```bash
cd qc
pip install -r requirements.txt     # Python >= 3.9; also needs ffmpeg / ffprobe on PATH
cp .env.example .env                # then set QC_API_KEY (only needed for run_qc.py)
```

## Data layout

Each recording is one directory, named by a UUID:

```
<batch_dir>/
└── <episode-uuid>/
    ├── rgb_head.mp4    rgb_head.csv       head RGB + per-frame timestamps
    ├── depth_head.mkv  depth_head.csv     head depth (FFV1, 16-bit)
    ├── wrist_left.mp4  wrist_left.csv     left wrist camera
    ├── wrist_right.mp4 wrist_right.csv    right wrist camera
    ├── left_hand_data.npz                 left glove
    ├── right_hand_data.npz                right glove
    └── task_info.json                     task metadata ({"name": ...})
```

- Timestamp CSVs have a header and rows `frame_index,timestamp_s` (UNIX epoch seconds).
- Each glove `.npz` holds `timestamps (N,)` and, for each pad slot `s` in
  0-5, 7-9, 11-13, 15-16, 18, a normal-force array `tactile_{s} (N, h, w)` and
  optional shear arrays `tf_tactile_x_{s}` / `tf_tactile_y_{s}`. The slot-to-part
  mapping (thumb/index/middle/ring tip-mid-base, pinky tip-mid, palm) is in
  [`touchscale_qc/episode_io.py`](touchscale_qc/episode_io.py).

`run_qc.py` skips (and lists) recordings with any file missing. Outputs and caches
are keyed by the first 8 characters of the directory name, so these must be unique
(the script refuses to run otherwise), and directory names should stay unique across
batches that share a cache. `check_sync.py`
searches `--root` recursively and also accepts wrist files under `realsense/` and
glove files under `glove/*/`.

## Stage 1: sync gate

```bash
python check_sync.py --root <batch_dir> --json sync.json
```

All six streams are resampled onto a uniform 30 Hz grid over their common time
window (nearest sample per tick, as is usual when building a training dataset).
A recording **passes** if all of the following hold:

1. all streams and video files are present, and the glove time window covers the camera window;
2. at least 95% of ticks have **all six** streams within 17.5 ms (`--align-thresh`, `--align-tol-ms`);
3. at most 1% of ticks are holes, i.e. some stream's nearest sample is more than one frame (33.3 ms) away (`--max-hole-frac`);
4. no nearest-sample gap exceeds 66 ms, about two frames (`--max-gap-ms`).

The thresholds tolerate isolated one-frame hiccups, which can be masked or
interpolated, but reject multi-frame blackouts that would hide fast contact
events. An earlier "zero holes" rule rejected many recordings that were in fact
usable. The tolerance is 17.5 ms rather than the geometric half-frame of 16.7 ms
because the gloves sample at ~35-56 Hz and their software timestamps jitter by
about 1 ms. Every failing recording gets a reason that names the stream at fault.

Evaluate alignment on the time grid, not by frame index: a single dropped frame
shifts every later index and makes a good recording look badly out of sync.

## Stage 2: tactile QC

```bash
python run_qc.py --data <batch_dir> --out qc_out --jobs 16 --qc-jobs 8
```

Outputs:

```
qc_out/
├── results.json      per-episode verdicts (structured)
├── integrity.json    frame-count / frame-drop / tactile-gap issues (code-only check)
├── videos/           rendered review video per episode
└── review/
    ├── README.md     review index: REJECT, REVIEW, and noise sections
    ├── reject/       close-up clips for high-confidence losses
    └── review/       close-up clips that need a human decision
```

Useful flags: `--skip-render` (reuse `qc_out/videos`), `--no-review`,
`--integrity-only` (no rendering, no model calls), `--stride` (render frame step;
default 2 = 15 fps). Rendering and integrity checks are CPU-bound (`--jobs`); the
verdict step waits on the API (`--qc-jobs`). No GPU is needed.

Rendered videos are reused from `<out>/videos`, and every model response and clip is
cached under `qc/.cache` (`QC_CACHE_DIR`, keyed by episode, hand and model).
Interrupted runs resume, and threshold changes can be re-evaluated without new API
calls. Delete the cache to re-query the model for the same episodes.

Wrist videos are sent inline (base64) in the request body, so very long recordings
can exceed the API's request-size limit; such episodes show up as `ERROR` in
`results.json`.

### How it works

```
Step 1  VLM watches one hand's wrist-camera video  ->  grasp events
        (grasp/release times, object, fingers "confirmed" on the object vs "uncertain")
        The model sees no tactile data at this step, so the visual judgement is
        independent of the signal it is checked against.

Step 2  numpy only
        loss:  a confirmed finger is silent during its grasp
        noise: tactile signal while the hand is in free space (candidates)

Step 3  VLM looks at each noise candidate (upscaled tactile map + wrist camera)
        and answers one question: is the hand really touching nothing?
        The step-2 filter is deliberately over-sensitive; this removes candidates
        that are explained by real contact.
```

Step 1 asks for **grasp events** rather than moment-by-moment contact state. Once
a hand closes on an object, the back of the hand fills the wrist camera and the
contact is hidden, but the moments of grasping and releasing are clearly visible.
Asked for moment-by-moment states, the model abstained repeatedly on long holds.

### Verdicts

| Verdict | Condition | Meaning |
|---|---|---|
| **REJECT** | within one grasp, at least two visually confirmed fingers are all silent and never respond anywhere in the episode; or all 15 pads of a glove are silent for the whole episode although step 1 reports at least one grasp for that hand | can be rejected automatically |
| **REVIEW** | one confirmed finger never responds while other fingers in the same grasp do | a human checks whether that finger really presses the object |
| **PASS** | everything else | baseline noise is still recorded |

The tiers follow a reliability gap we measured in blind tests. The hand-level
judgement ("is this hand holding something?") made no errors on 50 recordings,
while the finger-level judgement ("which finger is on the object?") was wrong in
all 6 cases where it was decisive. REJECT therefore depends only on the
hand-level judgement, and anything that needs the finger-level one goes to a
human. Pinky and palm never trigger a loss: the pinky often rests on an object
without load and is the most occluded finger, and the palm is rarely loaded in
fine manipulation. Noise is recorded but never changes the verdict, because in
human labels the boundary between acceptable and unacceptable noise could not be
drawn (the labelled noise durations of the two classes overlapped completely).

`results.json` per episode:

```json
{
  "1a2b3c4d": {
    "verdict": "REJECT", "episode": "1a2b3c4d-....",
    "noise_seg": 5, "noise_s": 10.3, "free_s": 17.2, "noise_pct": 60.0,
    "hands": {
      "left": {
        "verdict": "REJECT", "reason": "tactile loss (high confidence): ...",
        "loss": [{"finger": "thumb", "start": 3.6, "end": 9.3, "object": "test tube", "...": "..."}],
        "weak": [], "suspect": [], "noise": [{"start": 1.2, "end": 3.0, "dur": 1.8, "fingers": ["index"]}],
        "noise_unconfirmed": 0,
        "free_s": 12.8, "noise_s": 6.5, "noise_pct": 0.51
      },
      "right": {"...": "..."}
    }
  }
}
```

| Field | Meaning |
|---|---|
| `loss` | high-confidence losses (trigger REJECT) |
| `weak` | low-confidence losses (trigger REVIEW) |
| `suspect` | finger silent in this grasp but responsive elsewhere, so it may simply not have pressed; recorded only |
| `noise` | model-confirmed baseline-noise segments; recorded only |
| `noise_unconfirmed` | noise candidates whose step-3 call failed (counted as not noise); non-zero means re-run |
| `free_s` | time the hand is visually in free space (the denominator of the noise rate) |
| `noise_pct` | noise time / free time: a **fraction** (0-1) per hand, a **percentage** (0-100) per episode |

### Internal validation

On 53 recordings (50 random recordings not used for tuning, all with no tactile
loss, plus 3 known-loss recordings):

```
50 random (no loss)    REJECT 0   REVIEW 6   PASS 44
 3 known loss          REJECT 1   REVIEW 2   PASS 0
```

85% of recordings were decided automatically with no errors, 15% went to human
review, and 8/8 spot-checked noise records were correct. The 3 positives are too
few to give a meaningful recall estimate, and 2 of them were used during tuning.

### Configuration

All settings live in [`touchscale_qc/config.py`](touchscale_qc/config.py) and can be
overridden through the environment or `qc/.env`:

| Variable | Default | |
|---|---|---|
| `QC_API_KEY` / `GEMINI_API_KEY` | (none) | required for `run_qc.py` |
| `QC_API_BASE` | `https://generativelanguage.googleapis.com` | any endpoint that speaks Gemini `generateContent` |
| `QC_MODEL` | `gemini-3.7-flash` | must accept video input |
| `QC_FPS` / `QC_FPS3` | 2 / 6 | video sampling rate sent in step 1 / step 3 |
| `QC_CACHE_DIR` | `qc/.cache` | |

The rendering constants (`VMAX=2.0`, `GAMMA=1.0`, `ZERO_FLOOR=0.03`) define both
the colour scale that reviewers see and the "has signal" threshold of the numeric
checks, so change them together or not at all.

### Cost and throughput

Per episode: one step-1 call per hand, plus one step-3 call per noise candidate
(at most `MAX_NOISE_CLIPS` = 5 per hand; about 1.4 per episode on average in our
runs). In our measurements on single-threaded episodes of up to ~30 s, the
integrity check took ~8 s and rendering ~70 s per episode; the verdict step is
dominated by API latency.

### Known limitations

- Single-finger failures cannot be decided automatically (see Verdicts); they go to REVIEW.
- Pinky and palm are not checked for loss.
- Noise is recorded, not judged.
- Very few true-loss examples were available for tuning, so we recommend collecting
  more before adjusting the loss logic.
- Model outputs vary between calls, especially for finger-level judgements; the
  cache pins results, so clear it to re-query.

## Tests

```bash
cd qc && python -m pytest
```

The tests use synthetic data and make no network calls.
