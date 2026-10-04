"""Central configuration. Change settings here (or via environment variables),
not inside the individual modules.

Environment variables are read once at import time. A `.env` file next to
`run_qc.py` (i.e. in the `qc/` directory) is loaded first if present; values
already set in the environment take precedence.
"""
import os
import re

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
QC_DIR = os.path.dirname(PKG_DIR)


def _load_dotenv(path: str) -> None:
    """Minimal `.env` loader (KEY=VALUE lines); never overrides existing env vars."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip().removeprefix("export ").strip()
            v = v.strip()
            if v[:1] in ('"', "'") and v[-1:] == v[:1] and len(v) > 1:
                v = v[1:-1]                       # quoted: keep everything inside
            else:
                v = re.split(r"\s+#", v, maxsplit=1)[0].strip()   # drop inline comment
            os.environ.setdefault(k, v)


_load_dotenv(os.path.join(QC_DIR, ".env"))

# ── Vision-language model ───────────────────────────────────────────────────
# Any endpoint that speaks the Gemini `generateContent` REST protocol works.
# Gemini is used because it accepts raw video input with a controllable
# sampling rate. The default is Google's public Gemini API.
API_BASE = os.environ.get("QC_API_BASE", "https://generativelanguage.googleapis.com")
API_KEY = os.environ.get("QC_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("QC_MODEL", "gemini-3.7-flash")

# ── Sampling rates sent to the model ────────────────────────────────────────
FPS_STEP1 = float(os.environ.get("QC_FPS", "2"))       # step 1: grasp events from wrist video
FPS_STEP3 = float(os.environ.get("QC_FPS3", "6"))      # step 3: noise-candidate confirmation

# ── Verdict parameters ──────────────────────────────────────────────────────
# Fingers that take part in tactile-loss detection. Pinky and palm are excluded:
#   pinky -- in power grasps it often rests on the object without bearing load,
#            and it is the outermost (most occluded) finger in the wrist view.
#            On 27 known-good recordings it caused 2 false REJECTs, and none of
#            the 4 known true-loss recordings were caught through it.
#   palm  -- rarely loaded in fine manipulation, yet the model often counts it
#            as part of the grasp.
LOSS_FINGERS = ["thumb", "index", "middle", "ring"]

# A high-confidence (auto-REJECT) loss requires that, within one grasp, AT LEAST
# this many visually confirmed fingers are all silent. With only one finger,
# "all silent" degenerates into "a single silent finger", which is exactly the
# finger-level visual judgement that proved unreliable in blind tests.
MIN_CONFIRMED_FOR_REJECT = 2

MAX_NOISE_CLIPS = 5      # max noise candidates per hand sent to the model for confirmation
CLIP_PAD_S = 1.5         # seconds of context added before/after each candidate clip

# ── Rendering (keep in sync with what human reviewers look at) ─────────────
VMAX = 2.0               # full-scale colour value per taxel (glove range is roughly 0-2.2)
GAMMA = 1.0              # colour-map gamma
ZERO_FLOOR = 0.03        # taxels below this render black; also the "has signal" threshold

# ── Paths ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.environ.get("QC_CACHE_DIR", os.path.join(QC_DIR, ".cache"))


def require_api_key() -> str:
    """Return the API key, or raise with an actionable message if it is unset."""
    if not API_KEY:
        raise RuntimeError(
            "No model API key configured. Set QC_API_KEY (or GEMINI_API_KEY) in the "
            "environment or in qc/.env -- see qc/.env.example.")
    return API_KEY
