"""Human review material: close-up clips for REJECT / REVIEW hands plus an index README.

Each clip shows the upscaled tactile map next to the same hand's wrist camera, with a
burned-in timecode and the reason -- the same view the model was given. The reviewer
only has to answer one question: is that finger actually pressing on the object?
"""
import os
import subprocess

from . import config as C
from . import tactile as T
from .clips import filter_graph


def _esc(s):
    return str(s).replace(":", "\\:").replace("'", "").replace(",", " ")


def clip(viz, hand, start, end, text, out, pad=None):
    """Cut a review clip with a timecode and a caption burned in."""
    pad = C.CLIP_PAD_S if pad is None else pad
    ss = max(0.0, start - pad)
    overlay = (f",drawtext=text='t\\=%{{eif\\:{ss:.2f}+t\\:d\\:2}}."
               f"%{{eif\\:mod(trunc(({ss:.2f}+t)*10)\\,10)\\:d}}s':"
               f"x=12:y=12:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.6,"
               f"drawtext=text='{_esc(text)}':x=12:y=h-46:fontsize=22:fontcolor=yellow:"
               f"box=1:boxcolor=black@0.7")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.2f}",
                    "-to", f"{end + pad:.2f}", "-i", viz,
                    "-filter_complex", filter_graph(hand, overlay),
                    "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast",
                    "-movflags", "+faststart", out], check=False)
    return os.path.exists(out) and os.path.getsize(out) > 0


def build(results, viz_dir, out_dir, data_dir=None):
    """Write `<out_dir>/README.md` and per-finger clips under reject/ and review/."""
    for d in ["reject", "review"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    L = ["# Human review\n",
         "\nThe only question to answer: **is that finger actually pressing on the object "
         "in the video?**\n",
         "\n- Yes -> the tactile signal is missing; the loss is real.\n"
         "- No -> no tactile signal is expected; false alarm.\n"]

    for tier, title, hint in [
        ("REJECT", "1. Auto-reject (high confidence)",
         "Within one grasp, at least two visually confirmed fingers are all silent, or a "
         "whole glove is silent while the hand grasps something. Please confirm the loss is real."),
        ("REVIEW", "2. Needs human confirmation",
         "A single finger is silent while other fingers in the same grasp respond. "
         "Finger-level visual judgement is not reliable enough to decide automatically."),
    ]:
        items = {k: v for k, v in results.items() if v["verdict"] == tier}
        L.append(f"\n\n## {title} ({len(items)})\n\n{hint}\n")
        for uid, r in sorted(items.items()):
            viz = os.path.join(viz_dir, f"{uid}.mp4")
            ep = os.path.join(data_dir, r["episode"]) if data_dir else None
            L.append(f"\n### `{uid}`\n")
            for h, d in r["hands"].items():
                if d["verdict"] == "PASS":
                    continue
                L.append(f"\n- **{h} hand** ({d['verdict']}) {d['reason']}\n")
                for item in (d["loss"] or []) + (d.get("weak") or []):
                    o = os.path.join(out_dir, d["verdict"].lower(),
                                     f"{uid}_{h}_{item['finger']}"
                                     f"_{item['start']:.0f}-{item['end']:.0f}s.mp4")
                    if os.path.exists(viz) and clip(
                            viz, h, item["start"], item["end"],
                            f"{uid} {h} {item['finger']} claimed on "
                            f"{item.get('object', '')} but silent", o):
                        L.append(f"  - [close-up: {item['finger']} "
                                 f"{item['start']:.1f}-{item['end']:.1f}s]"
                                 f"({d['verdict'].lower()}/{os.path.basename(o)})\n")
                if ep and os.path.isdir(ep):
                    p = T.peak_per_part(ep, h)
                    L.append("  - peak force per part: "
                             + ", ".join(f"{k}={v:.2f}" for k, v in p.items()) + "\n")
            if os.path.exists(viz):
                L.append(f"- [full video]({os.path.relpath(viz, out_dir)})\n")

    # ERROR results carry only {verdict, episode, error}; tolerate missing noise fields.
    noisy = sorted(((r.get("noise_pct", 0.0), k, r) for k, r in results.items()
                    if r.get("noise_seg", 0) > 0), reverse=True)
    L.append(f"\n\n## 3. Baseline noise ({len(noisy)}) -- recorded only, does not affect verdicts\n\n")
    L.append("| episode | verdict | segments | total | share of free time |\n|---|---|---|---|---|\n")
    for pct, uid, r in noisy:
        L.append(f"| `{uid}` | {r.get('verdict', '?')} | {r.get('noise_seg', 0)} | "
                 f"{r.get('noise_s', 0.0):.1f}s | {pct:.0f}% |\n")

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.writelines(L)
