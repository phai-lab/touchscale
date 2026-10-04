"""Step 1: ask a VLM for the grasp events of one hand, from its wrist camera only.

The model never sees tactile data at this step, so its visual judgement cannot be
biased by the signal it is later compared against. Responses are cached per
(episode, hand, model, fps), so re-tuning downstream thresholds costs no API calls.
"""
import base64
import json
import os
import time

import numpy as np
import requests

from . import config as C
from . import tactile as T

FPS = C.FPS_STEP1
CACHE = os.path.join(C.CACHE_DIR, "step1")

PROMPT = """This is the wrist-mounted camera of the {hand} hand of a person wearing a tactile
glove. Everything in view belongs to THIS hand - no other hand is present. Video starts
at t=0s. Watch the WHOLE clip before answering.

Report this hand's GRASP EVENTS, not moment-by-moment states.

Why: once the hand closes on something, the back of the hand fills this camera and you
usually cannot see the contact any more. But the instants of GRASPING and RELEASING are
clearly visible - the fingers close, the object starts moving with the hand, and later
the fingers open and the object stays behind. Between those two instants the object is
still held, even when you cannot see it. Judge the whole hold from its endpoints.

For each grasp, report:
  "grasp"      the moment the fingers close on the object
  "release"    the moment they let go (use the end of the video if it is never released)
  "object"     what is held
  "confirmed"  fingers you can SEE closing onto the object at the moment of the grasp,
               or that you can see pressing at any point during the hold
  "uncertain"  fingers you cannot resolve - hidden behind the hand or the object,
               out of frame, or blurred

The confirmed / uncertain split is the most important part. Downstream, a confirmed
finger that reports no force is treated as a broken sensor, so a wrong "confirmed"
creates a false alarm. Only list a finger as confirmed if you actually saw it on the
object. When in doubt, put it in uncertain.

Also report a short "grip_type" for each grasp - for example "pinch (thumb+index)",
"power grasp (whole hand around handle)", "two-finger", "palm press". This is what you
can infer about which fingers must be involved even when they are occluded; it is
recorded for context but is NOT treated as confirmed.

A finger visibly OFF the object belongs in neither list - omit it.

Also report "free": intervals in which the hand holds nothing and touches nothing.
A hand spends much of a task moving through free space, so these are expected.

Brief brushes and taps count as grasps too - use the same start/end fields.
If you cannot tell whether something was grasped at all, set "uncertain_interval": true.

JSON only:
{{"contacts":[{{"start":0.0,"end":0.0,"confirmed":[],"uncertain":[],"object":"",
                "grip_type":"","uncertain_interval":false}}],
  "free":[[0.0,0.0]]}}
Finger names: thumb,index,middle,ring,pinky,palm."""


def generate(contents, tries=3, timeout=180):
    """One `generateContent` call with retry and exponential backoff.

    A short timeout plus retries recovers from sporadic gateway stalls much faster
    than a single long timeout, which can hold up an entire batch.
    Returns (text, usage_metadata).
    """
    key = C.require_api_key()
    body = {"contents": contents,
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}
    last = "no attempt"
    for k in range(tries):
        try:
            # key goes in a header, not the URL, so it never shows up in error strings
            r = requests.post(f"{C.API_BASE}/v1beta/models/{C.MODEL}:generateContent",
                              headers={"x-goog-api-key": key}, json=body, timeout=timeout)
            if r.status_code == 200:
                d = r.json()
                return (d["candidates"][0]["content"]["parts"][0]["text"],
                        d.get("usageMetadata", {}))
            last = f"HTTP {r.status_code} {r.text[:160]}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if k < tries - 1:
            time.sleep(2 ** k * 5)
    raise RuntimeError(last)


def parse_json(txt):
    """Parse a JSON object from model output, tolerating ``` fences and stray text."""
    s = txt.strip().strip("`")
    s = s[4:] if s.startswith("json") else s
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        return json.loads(s[i:j + 1]) if i >= 0 < j else {}


def normalize_r1(r1):
    """Coerce a step-1 response into the shape downstream code assumes.

    The prompt asks for "free":[[start,end]], but the model sometimes answers
    [{"start":..,"end":..}] instead. Iterating that as `for a, b in free` yields
    the dict keys, and the next numpy comparison raises UFuncTypeError. Because
    responses are cached, such a failure would be permanent, so this is applied
    to both fresh and cached responses.
    """
    if not isinstance(r1, dict):
        return {}
    free = []
    for iv in r1.get("free") or []:
        if isinstance(iv, dict):
            a, b = iv.get("start"), iv.get("end")
        elif isinstance(iv, (list, tuple)) and len(iv) >= 2:
            a, b = iv[0], iv[1]
        else:
            continue
        try:
            free.append([float(a), float(b)])
        except (TypeError, ValueError):
            continue
    r1["free"] = free

    contacts = []
    for c in r1.get("contacts") or []:
        if not isinstance(c, dict):
            continue
        try:
            c["start"], c["end"] = float(c["start"]), float(c["end"])
        except (TypeError, ValueError, KeyError):
            continue                      # unusable without a time window
        contacts.append(c)
    r1["contacts"] = contacts
    return r1


def write_json_atomic(path, obj):
    """Write JSON via a temp file so an interrupted run never leaves a truncated cache entry."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def cache_path(ep, hand):
    return os.path.join(CACHE, f"{os.path.basename(ep)}_{hand}_{C.MODEL}_fps{FPS:g}.json")


def run(ep, hand):
    """Grasp events for one hand (cached). Returns the normalised step-1 dict."""
    ck = cache_path(ep, hand)
    if os.path.exists(ck):
        return normalize_r1(json.load(open(ck))["r1"])
    os.makedirs(CACHE, exist_ok=True)
    video = {"inline_data": {"mime_type": "video/mp4",
                             "data": base64.b64encode(
                                 open(os.path.join(ep, f"wrist_{hand}.mp4"), "rb").read()).decode()},
             "video_metadata": {"fps": FPS}}
    text, usage = generate([{"role": "user",
                             "parts": [video, {"text": PROMPT.format(hand=hand)}]}])
    r1 = normalize_r1(parse_json(text))
    write_json_atomic(ck, {"r1": r1, "tok": [usage.get("totalTokenCount", 0)]})
    return r1


def segments(mask, t, min_len=0.0):
    """Split a boolean series into [(start_s, end_s)] runs (gaps of <=2 samples bridged)."""
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    out = []
    for g in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1):
        a, b = t[g[0]], t[g[-1]]
        if b - a >= min_len:
            out.append((float(a), float(b)))
    return out


def hard_fault(slots, n):
    """Tier-0 numeric check: True if none of the hand's 15 pads ever responds.

    (An earlier rule, "negative-value frame ratio > 5% => fault", looked perfect
    on the first 10 samples but failed on later batches, where good recordings
    had a median negative ratio of 20%. Negative values are a calibration/batch
    property, not a fault signal, so that rule was dropped.)
    """
    _, P = T.pad_series(slots, n)
    return int((P.max(1) > T.ZERO_FLOOR).sum()) == 0
