"""Anatomical tactile hand panel: fixed per-pad geometry, heat-map pasting, and text.

Every physical pad is resized for display into a fixed anatomical segment, so the
sensor matrix dimensions do not make some fingers look longer or wider than others.
Raw values and all QC computations are unaffected by this layout.

The panel geometry is load-bearing: `clips.py` crops the rendered video using
coordinates derived from the constants below.
"""
import os

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C

PANEL_W, PANEL_H, TITLE = 512, 384, 30      # camera panel size and title bar height
CANVAS_W, CANVAS_H = 520, 600               # one hand panel
_HAND_W, _HAND_H = 430, 540
CMAP = cv2.COLORMAP_TURBO

# DejaVu Sans ships with matplotlib, so no font file needs to be vendored.
_FONT_PATH = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
_FONTS = {}

# (segments [(slot, length fraction)], centre x, base y, total length, width, angle deg)
_DIGITS = (
    (((2, .29), (1, .29), (0, .36)), .285, .305, .355, .100, 43.0),      # thumb
    (((5, .29), (4, .29), (3, .36)), .405, .430, .430, .090, 1.5),       # index
    (((9, .29), (8, .29), (7, .36)), .505, .440, .490, .094, 0.0),       # middle
    (((13, .29), (12, .29), (11, .36)), .605, .430, .455, .090, -1.5),   # ring
    (((16, .45), (15, .51)), .700, .405, .365, .084, -3.0),              # pinky
)


def _font(size):
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(_FONT_PATH, size)
    return _FONTS[size]


def draw_texts(bgr, items):
    """Draw [(text, (x, y), size, (b, g, r)), ...] onto a BGR image."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for text, (x, y), size, (b, g, r) in items:
        d.text((x, y), text, font=_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _axis_point(cx, bottom, distance, angle):
    r = np.deg2rad(angle)
    return cx - np.sin(r) * distance, bottom + np.cos(r) * distance


def geometry():
    """Normalised display region (cx, cy, width, height, angle) for every pad slot."""
    regions = {}
    gap = .010
    for segments, cx, bottom, total_h, width, angle in _DIGITS:
        usable = total_h - gap * (len(segments) - 1)
        distance = 0.0
        scale = sum(frac for _, frac in segments)
        for slot, frac in segments:
            height = usable * frac / scale
            px, py = _axis_point(cx, bottom, distance + height / 2, angle)
            regions[slot] = (px, py, width, height, angle)
            distance += height + gap
    regions[18] = (.505, .245, .325, .315, 0.0)                          # palm
    return {"grid_w": CANVAS_W, "grid_h": CANVAS_H, "regions": regions}


def _to_px(cx, cy, mirror):
    # Palm-facing view: the left thumb appears on the right, the right thumb on the left.
    if mirror:
        cx = 1.0 - cx
    return int(round(45 + cx * _HAND_W)), int(round(35 + (1.0 - cy) * _HAND_H))


def _rounded_mask(h, w):
    mask = np.zeros((h, w), np.uint8)
    radius = max(2, min(h, w) // 7)
    cv2.rectangle(mask, (radius, 0), (w - radius - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (w - 1, h - radius - 1), 255, -1)
    for x in (radius, w - radius - 1):
        for y in (radius, h - radius - 1):
            cv2.circle(mask, (x, y), radius, 255, -1, lineType=cv2.LINE_AA)
    return mask


def _composite_rotated(canvas, tile, tile_mask, pcx, pcy, angle):
    """Rotate an image tile and its mask together about a shared centre and blend."""
    h, w = tile.shape[:2]
    side = int(np.ceil(np.hypot(w, h))) + 4
    layer = np.zeros((side, side, 3), np.uint8)
    mask = np.zeros((side, side), np.uint8)
    x0, y0 = (side - w) // 2, (side - h) // 2
    layer[y0:y0 + h, x0:x0 + w] = tile
    mask[y0:y0 + h, x0:x0 + w] = tile_mask
    M = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    layer = cv2.warpAffine(layer, M, (side, side), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, M, (side, side), flags=cv2.INTER_LINEAR)
    px, py = pcx - side // 2, pcy - side // 2
    roi = canvas[py:py + side, px:px + side]
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    roi[:] = (layer * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def template(g, mirror, title):
    """Static hand panel: a neutral rounded halo for every pad, plus a title."""
    img = np.full((g["grid_h"], g["grid_w"], 3), 26, np.uint8)
    halo_px = 7
    for cx, cy, width, height, angle in g["regions"].values():
        pw, ph = max(4, int(width * _HAND_W)), max(4, int(height * _HAND_H))
        sw, sh = pw + 2 * halo_px, ph + 2 * halo_px
        layer = np.full((sh, sw, 3), (58, 58, 62), np.uint8)
        _composite_rotated(img, layer, _rounded_mask(sh, sw), *_to_px(cx, cy, mirror),
                           (-angle if mirror else angle))
    return draw_texts(img, [(title, (18, 4), 22, (240, 240, 240))])


def paste_cells(canvas, g, slots, shear, idx, mirror,
                vmax=C.VMAX, gamma=C.GAMMA, zero_floor=C.ZERO_FLOOR, show_slot=False):
    """Paint frame `idx` of every pad onto a hand panel produced by `template`.

    Each taxel is coloured by its own force; taxels below `zero_floor` stay black.
    A white arrow at each pad centre shows the mean shear vector.
    """
    for s, (cx, cy, width, height, angle) in g["regions"].items():
        if s not in slots:
            continue
        arr = slots[s]
        i = min(idx, len(arr) - 1)
        a = arr[i]
        vv = np.clip(a / vmax, 0, 1) ** gamma
        heat = cv2.applyColorMap((vv * 255).astype(np.uint8), CMAP)
        heat[a < zero_floor] = 0
        pw, ph = max(4, int(width * _HAND_W)), max(4, int(height * _HAND_H))
        big = cv2.resize(heat, (pw, ph), interpolation=cv2.INTER_LINEAR)
        if mirror:
            big = cv2.flip(big, 1)
        pcx, pcy = _to_px(cx, cy, mirror)
        _composite_rotated(canvas, big, _rounded_mask(ph, pw), pcx, pcy,
                           -angle if mirror else angle)
        if show_slot:                       # raw slot index, for mapping verification
            tp = (pcx - 8, pcy + 5)
            cv2.putText(canvas, str(s), tp, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(s), tp, cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
        if s in shear:
            fx, fy = shear[s]
            k = min(i, len(fx) - 1)
            dx, dy = float(fx[k]) * 60, float(fy[k]) * 60
            ex, ey = int(pcx + (-dx if mirror else dx)), int(pcy + dy)
            if abs(dx) + abs(dy) > 2:
                cv2.arrowedLine(canvas, (pcx, pcy), (ex, ey), (255, 255, 255), 2, tipLength=0.35)


def timeline(W, Lts, LP, Rts, RP, cam_t0, t_now, h=150):
    """Total-pressure curves for both hands with a cursor at `t_now` (seconds)."""
    img = np.full((h, W, 3), 20, np.uint8)
    x0, x1, y0, y1 = 54, W - 12, 12, h - 26
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 1)
    tmax = max(Lts[-1], Rts[-1]) - cam_t0
    pmax = max(LP.max(), RP.max(), 1)

    def poly(ts, P, col):
        xs = np.clip((ts - cam_t0) / max(tmax, 1e-6), 0, 1) * (x1 - x0) + x0
        ys = y1 - np.clip(P / pmax, 0, 1) * (y1 - y0)
        cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, col, 1)

    poly(Lts, LP, (255, 160, 60))
    poly(Rts, RP, (80, 80, 255))
    cx = int(t_now / max(tmax, 1e-6) * (x1 - x0) + x0)
    cv2.line(img, (cx, y0), (cx, y1), (240, 240, 240), 1)
    return draw_texts(img, [("Total pressure  L (blue) / R (red)", (x0, h - 22), 15, (200, 200, 200)),
                            (f"{t_now:5.1f}s", (max(x0, cx - 20), 0), 14, (240, 240, 240))])
