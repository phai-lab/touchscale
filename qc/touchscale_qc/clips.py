"""Cut per-hand close-up clips out of the rendered review video.

Why: the full 2048x1214 frame gets downscaled to ~768 px wide by the model, which
leaves the tactile panel at ~195x225 px. Weak baseline noise (a few hundred coloured
pixels in the original render) then shrinks to a few dozen pixels and is erased by
h264 compression -- the model cannot see it at all.

So each clip shows a single hand: the tactile panel cropped to the hand's bounding
box and upscaled 2x (nearest neighbour, so isolated taxels survive compression as
2x2 blocks), next to that hand's wrist camera.

Offsets are derived from the renderer constants in `render.py` / `hand_grid.py`:
    TITLE=30  PANEL_W=512  PANEL_H=384  hand panel 520x600
    camera row: x = i*512, y = 30..414
    tactile row: x0 = (2048 - (520+30+520)) // 2 = 489, y = 444..1044
                 left hand x 489..1009, right hand x 1039..1559
"""
import os
import subprocess

from . import hand_grid as G

TITLE, PW, PH, CW = G.TITLE, G.PANEL_W, G.PANEL_H, G.CANVAS_W
TAC_X0, TAC_Y0 = 489, 444
WRIST_X = {"left": 2 * PW, "right": 3 * PW}
TAC_X = {"left": TAC_X0, "right": TAC_X0 + CW + 30}

# Hand bounding box inside the 520x600 tactile panel (x0, y0, w, h). Left and right
# are mirror images (the abducted thumb sits on opposite sides), so the boxes differ.
HBOX = {"right": (12, 4, 376, 538),
        "left": (12, 4, 492, 538)}
UP = 2                                  # nearest-neighbour upscale factor


def filter_graph(hand, extra=""):
    """ffmpeg filter: upscaled tactile crop | wrist camera, side by side."""
    hx, hy, hw, hh = HBOX[hand]
    tw, th = hw * UP, hh * UP
    return (f"[0:v]crop={hw}:{hh}:{TAC_X[hand]+hx}:{TAC_Y0+hy},"
            f"scale={tw}:{th}:flags=neighbor[t];"
            f"[0:v]crop={PW}:{PH}:{WRIST_X[hand]}:{TITLE},scale=340:-2,"
            f"pad=340:{th}:0:0:black[w];"        # hstack needs equal heights
            f"[t][w]hstack=inputs=2{extra}")


def make_clip(viz, hand, start, end, duration, out, pad):
    """Cut [start-pad, end+pad] (clamped to the video) for one hand. Returns path or None."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ss, to = max(0.0, start - pad), min(duration, end + pad)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.2f}", "-to", f"{to:.2f}",
                    "-i", viz, "-filter_complex", filter_graph(hand), "-c:v", "libx264",
                    "-profile:v", "main", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", out], check=False)
    return out if os.path.exists(out) and os.path.getsize(out) else None
