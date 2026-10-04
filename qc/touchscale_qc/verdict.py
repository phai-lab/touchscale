"""Steps 2-3 and the per-hand verdict.

    Step 1  (grasp_events.py) VLM reads the wrist video -> grasp intervals, with the
            fingers it can SEE on the object ("confirmed") vs. cannot resolve.
    Step 2  numpy only:
              tactile loss -- a confirmed finger is silent during its grasp
              noise        -- tactile signal while the hand is in free space (candidates)
    Step 3  VLM binary check on each noise candidate clip: is the hand really touching
            nothing? The step-2 pre-filter is deliberately over-sensitive; this step
            removes candidates that are actually explained by contact.

Verdicts per hand:
    REJECT  high-confidence loss: within one grasp, >= MIN_CONFIRMED_FOR_REJECT confirmed
            fingers are all silent and never respond anywhere in the episode; or the whole
            glove (all 15 pads) never responds although the hand visibly grasps something.
            These rely only on the hand-level visual judgement ("is this hand holding
            something?"), which made zero errors in a 50-recording blind test.
    REVIEW  low-confidence loss: a single confirmed finger never responds while other
            fingers in the same grasp do. This relies on finger-level visual judgement,
            which was unreliable in blind tests, so it goes to a human.
    PASS    everything else. Confirmed noise and "suspect" silences are recorded as facts
            but never change the verdict: in human labels the borderline/reject boundary
            for noise was not separable (cumulative durations fully overlapped).
"""
import base64
import json
import os
import time

import numpy as np
import requests

from . import config as C
from . import grasp_events as Q
from . import tactile as T
from .clips import make_clip

CACHE = os.path.join(C.CACHE_DIR, "step3")
CLIPDIR = os.path.join(C.CACHE_DIR, "clips")

PROMPT = """This clip was cut from a tactile-glove recording because an automatic pre-filter
flagged it: during this window the {hand} hand looks like it is in FREE SPACE, yet its
tactile sensors still report some force. Confirm or reject that flag.

WHAT YOU SEE - two panels side by side:
- LEFT: the tactile map of the {hand} hand. The angled column at the side is the thumb,
  then index, middle, ring, pinky, each with tip / mid / base pads, and the palm pad at
  the bottom. Colour = force. BLACK = no force. A few small coloured specks still count.
- RIGHT: the wrist camera of the same hand over the same moments.

ONE QUESTION: during this window, is the hand actually touching or gripping anything?

- Holding a TOOL (pipette, stirring rod, wash bottle, tweezers, tube rack) IS contact,
  and the fingers on it should legitimately report force.
- A whole hand lighting up while it grips something is NORMAL.
- The clip includes about {pad:.1f}s of lead-in and lead-out, so the hand may make or
  break contact near the edges; judge the middle of the window.

If the hand is genuinely in free space and the tactile map still shows colour, this is
real baseline noise -> "noise": true.
If the force is explained by genuine contact, or the map is effectively black,
-> "noise": false.

The pre-filter is deliberately over-sensitive; "false" is a perfectly normal answer.

JSON only: {{"noise":true|false,"fingers":[],"why":"<15 words, what you saw>"}}"""


# ─────────────────────────────────────────────────────────── step 2: numpy
def check_loss(t, per, r1):
    """Confirmed fingers that are silent during their grasp.

    Returns (high_confidence, low_confidence, suspect). Only the model's
    `confirmed` list is trusted; `uncertain` fingers are never judged.
    """
    hard, weak, soft = [], [], []
    ever = {f: per[f].max() > 0 for f in per}
    for c in r1.get("contacts", []):
        if c.get("uncertain_interval") or c.get("uncertain") is True:
            continue
        conf = [f for f in (c.get("confirmed") or c.get("fingers") or [])
                if f in C.LOSS_FINGERS]
        m = (t >= c["start"]) & (t <= c["end"])
        if m.sum() < 3:
            continue
        silent = [f for f in conf if per[f][m].max() == 0]
        # All confirmed fingers silent -> only the hand-level judgement matters.
        # Requiring >= 2 fingers keeps this from degenerating into a single-finger call.
        whole = len(silent) == len(conf) and len(conf) >= C.MIN_CONFIRMED_FOR_REJECT
        for f in silent:
            rec = dict(start=float(c["start"]), end=float(c["end"]), finger=f,
                       object=c.get("object", ""), dur=float(c["end"] - c["start"]),
                       whole_hand=whole, confirmed=list(conf))
            if not ever[f] and whole:
                hard.append(rec)          # high confidence: can be rejected automatically
            elif not ever[f]:
                weak.append(rec)          # low confidence: human review
            else:
                soft.append(rec)          # responds elsewhere; maybe just not pressing
    return hard, weak, soft


def noise_candidates(t, per, slots, n, r1):
    """Free-space intervals where the tactile map still shows any visible signal."""
    vis = np.zeros(n, bool)
    for s in T.PADS:
        vis |= (slots[s][:n].reshape(n, -1) >= T.ZERO_FLOOR).any(1)
    tot = sum(per[f] for f in per)
    out = []
    for a, b in (r1.get("free") or []):
        m = (t >= a) & (t <= b)
        if m.sum() < 3:
            continue
        for sa, sb in Q.segments(m & vis, t, 0.3):
            k = (t >= sa) & (t <= sb)
            hot = [f for f in T.FINGERS if per[f][k].max() > 0]
            out.append(dict(start=float(sa), end=float(sb), dur=float(sb - sa),
                            peak=float(tot[k].max()), fingers=hot))
    out.sort(key=lambda d: -(d["dur"] * max(d["peak"], 1e-6)))
    return out[:C.MAX_NOISE_CLIPS]


# ─────────────────────────────────────────────────────────── step 3: VLM
def confirm_noise(clip, hand):
    """Ask the model whether a candidate clip is real free-space noise (cached)."""
    ck = os.path.join(CACHE, f"{os.path.basename(clip)}_{C.MODEL}.json")
    if os.path.exists(ck):
        return json.load(open(ck))
    os.makedirs(CACHE, exist_ok=True)
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "video/mp4",
                         "data": base64.b64encode(open(clip, "rb").read()).decode()},
         "video_metadata": {"fps": C.FPS_STEP3}},
        {"text": PROMPT.format(hand=hand, pad=C.CLIP_PAD_S)}]}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}
    err = "no attempt"
    for k in range(3):
        try:
            r = requests.post(f"{C.API_BASE}/v1beta/models/{C.MODEL}:generateContent",
                              headers={"x-goog-api-key": C.require_api_key()},
                              json=body, timeout=180)
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
                time.sleep(2 ** k * 5)
                continue
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().strip("`")
            txt = txt[4:] if txt.startswith("json") else txt
            i, j = txt.find("{"), txt.rfind("}")
            o = json.loads(txt[i:j + 1])          # malformed -> retry, never cached
            Q.write_json_atomic(ck, o)
            return o
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if k < 2:
                time.sleep(2 ** k * 5)
    # Noise is only recorded, never decisive: prefer a missed record over a false one.
    print(f"    [warn] noise confirmation failed, treating as not noise: {err[:70]}", flush=True)
    return {"noise": False, "why": f"API failure: {err[:70]}", "failed": True}


# ─────────────────────────────────────────────────────────── per-hand verdict
def _empty(verdict, reason):
    return dict(verdict=verdict, reason=reason, loss=[], weak=[], suspect=[], noise=[],
                free_s=0.0, noise_s=0.0, noise_pct=0.0)


def run_hand(ep, hand, viz, uid):
    """Full QC for one hand of one episode. `viz` is the rendered review video."""
    _, slots, _ = T.load(ep, hand)
    r1 = Q.run(ep, hand)

    # A dead glove is only a fault if the hand visibly grasps something: a hand that
    # never touches anything legitimately reports all zeros.
    if Q.hard_fault(slots, min(len(v) for v in slots.values())) and r1.get("contacts"):
        return _empty("REJECT", "[tier0] all 15 pads silent for the whole episode "
                                "(vision confirms this hand grasps something)")

    t, per, slots, n = T.finger_force(ep, hand)
    hard, weak, soft = check_loss(t, per, r1)
    free_s = sum(max(0.0, b - a) for a, b in (r1.get("free") or []))

    confirmed, unconfirmed = [], 0
    for k, c in enumerate(noise_candidates(t, per, slots, n, r1)):
        out = os.path.join(CLIPDIR, f"{uid[:8]}_{hand}_{k}_{c['start']:.0f}-{c['end']:.0f}.mp4")
        if not os.path.exists(out):
            make_clip(viz, hand, c["start"], c["end"], float(t[-1]), out, C.CLIP_PAD_S)
        if not (os.path.exists(out) and os.path.getsize(out)):
            continue
        r = confirm_noise(out, hand)
        unconfirmed += bool(r.get("failed"))
        if r.get("noise"):
            confirmed.append(dict(start=c["start"], end=c["end"], dur=c["dur"],
                                  fingers=r.get("fingers") or c["fingers"],
                                  why=r.get("why", "")))

    noise_s = sum(x["dur"] for x in confirmed)
    base = dict(loss=hard, weak=weak, suspect=soft, noise=confirmed,
                noise_unconfirmed=unconfirmed, free_s=free_s,
                noise_s=noise_s, noise_pct=noise_s / free_s if free_s > 1e-6 else 0.0)
    if hard:
        return dict(verdict="REJECT", reason="tactile loss (high confidence): " + "; ".join(
            f"{h['finger']} touches {h['object'] or 'object'} at {h['start']:.1f}-{h['end']:.1f}s "
            f"and every visually confirmed finger in that grasp is silent" for h in hard), **base)
    if weak:
        def _why(w):
            others = [x for x in w["confirmed"] if x != w["finger"]]
            tail = (f"while {', '.join(others)} in the same grasp respond" if others
                    else "and it is the only confirmed finger in that grasp (weak evidence)")
            return (f"{w['finger']} reported touching {w['object'] or 'object'} at "
                    f"{w['start']:.1f}-{w['end']:.1f}s but is silent, {tail}")
        return dict(verdict="REVIEW", reason="possible tactile loss (needs human check): "
                    + "; ".join(_why(w) for w in weak), **base)
    return dict(verdict="PASS", reason="", **base)


def run_episode(ep, viz, name):
    """Both hands -> episode result (worst hand verdict wins)."""
    hands = {h: run_hand(ep, h, viz, name) for h in ["left", "right"]}
    vs = [hands[h]["verdict"] for h in hands]
    v = "REJECT" if "REJECT" in vs else ("REVIEW" if "REVIEW" in vs else "PASS")
    noise_s = sum(hands[h]["noise_s"] for h in hands)
    free_s = sum(hands[h]["free_s"] for h in hands)
    return dict(verdict=v, episode=name, hands=hands,
                noise_seg=sum(len(hands[h]["noise"]) for h in hands),
                noise_s=noise_s, free_s=free_s, noise_pct=100 * noise_s / max(free_s, 1e-6))
