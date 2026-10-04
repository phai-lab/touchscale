"""Render the synchronized review video for an episode.

Layout (2048 px wide): four camera panels on top (head RGB | head depth | left wrist |
right wrist), the two anatomical tactile hand panels below, and a total-pressure
timeline at the bottom. Cameras are read in lockstep by frame index; each tactile
frame is the glove sample nearest in time to the head-RGB frame.

This video is what both the model (step 3) and human reviewers look at, so its
geometry and colour scale must stay stable; `clips.py` crops it by fixed offsets.
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np

from . import config as C
from . import episode_io as E
from . import hand_grid as G

os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
cv2.setNumThreads(1)                      # avoid decoder thread oversubscription in workers

PANEL_W, PANEL_H, TITLE = G.PANEL_W, G.PANEL_H, G.TITLE


def _find(ep, *names):
    """First existing path among candidates; supports flat and nested episode layouts."""
    for n in names:
        hits = glob.glob(os.path.join(ep, n))
        if hits:
            return hits[0]
    return None


def render_episode(ep, out_path, stride=1, name=None, vmax=C.VMAX, gamma=C.GAMMA,
                   zero_floor=C.ZERO_FLOOR, show_slot=False, quiet=True):
    """Render one episode directory to `out_path`. Returns a short status string."""
    name = name or os.path.basename(os.path.normpath(ep))
    rgb_csv = os.path.join(ep, "rgb_head.csv")
    if not os.path.exists(rgb_csv):
        return f"SKIP {name} (no rgb_head.csv)"
    rgb_ts = E.csv_ts(rgb_csv)
    if len(rgb_ts) < 2:
        return f"SKIP {name} (empty rgb_head.csv)"
    cam_t0, n_cam = rgb_ts[0], len(rgb_ts)
    lnpz = _find(ep, "left_hand_data.npz", "glove/*/left_hand_data.npz")
    rnpz = _find(ep, "right_hand_data.npz", "glove/*/right_hand_data.npz")
    if not (lnpz and rnpz):
        return f"SKIP {name} (no glove data)"

    Lts, Lslots, Lshear = E.load_hand(lnpz)
    Rts, Rslots, Rshear = E.load_hand(rnpz)
    LP, RP = E.total_force(Lslots, len(Lts)), E.total_force(Rslots, len(Rts))
    gL, gR = G.geometry(), G.geometry()
    tplL = G.template(gL, mirror=True, title="Left hand tactile")
    tplR = G.template(gR, mirror=False, title="Right hand tactile")

    top_w = 4 * PANEL_W
    tac_w = gL["grid_w"] + 30 + gR["grid_w"]
    W = max(top_w, tac_w)
    tl_h = 150
    H = TITLE + PANEL_H + 30 + max(gL["grid_h"], gR["grid_h"]) + tl_h + 20
    top_x0, tac_x0 = (W - top_w) // 2, (W - tac_w) // 2
    tac_y0 = TITLE + PANEL_H + 30

    caps = {k: cv2.VideoCapture(p) for k, p in [
        ("rgb", os.path.join(ep, "rgb_head.mp4")),
        ("wl", _find(ep, "wrist_left.mp4", "realsense/wrist_left.mp4") or ""),
        ("wr", _find(ep, "wrist_right.mp4", "realsense/wrist_right.mp4") or "")]}
    dproc, dw, dh, dsz = E.depth_reader(os.path.join(ep, "depth_head.mkv"))
    blank = np.full((PANEL_H, PANEL_W, 3), 30, np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = iio.get_writer(out_path, fps=30.0 / stride, codec="libx264", quality=8,
                            macro_block_size=None, ffmpeg_log_level="error", pixelformat="yuv420p",
                            output_params=["-threads", "2", "-profile:v", "main",
                                           "-movflags", "+faststart"])
    gamma_txt = f", gamma {gamma:g}" if gamma != 1.0 else ""
    try:
        for fi in range(n_cam):
            okr, rgb = caps["rgb"].read()
            okl, wl = caps["wl"].read()
            okrr, wr = caps["wr"].read()
            raw = dproc.stdout.read(dsz)
            if fi % stride:
                continue
            if not okr:
                break
            t_epoch = rgb_ts[fi]
            t_rel = t_epoch - cam_t0
            d16 = (np.frombuffer(raw, np.uint16).reshape(dh, dw) if len(raw) == dsz
                   else np.zeros((dh, dw), np.uint16))
            canvas = np.full((H, W, 3), 22, np.uint8)
            panels = [(cv2.resize(rgb, (PANEL_W, PANEL_H)), "Head RGB"),
                      (cv2.resize(E.colorize_depth(d16), (PANEL_W, PANEL_H)), "Head depth"),
                      (cv2.resize(wl, (PANEL_W, PANEL_H)) if okl else blank, "Left wrist"),
                      (cv2.resize(wr, (PANEL_W, PANEL_H)) if okrr else blank, "Right wrist")]
            labels = []
            for j, (img, label) in enumerate(panels):
                x = top_x0 + j * PANEL_W
                canvas[TITLE:TITLE + PANEL_H, x:x + PANEL_W] = img
                labels.append((label, (x + 8, TITLE + 4), 16, (240, 240, 240)))

            iL, iR = E.nearest(Lts, t_epoch), E.nearest(Rts, t_epoch)
            cL = tplL.copy()
            G.paste_cells(cL, gL, Lslots, Lshear, iL, True, vmax, gamma, zero_floor, show_slot)
            cR = tplR.copy()
            G.paste_cells(cR, gR, Rslots, Rshear, iR, False, vmax, gamma, zero_floor, show_slot)
            canvas[tac_y0:tac_y0 + gL["grid_h"], tac_x0:tac_x0 + gL["grid_w"]] = cL
            rx = tac_x0 + gL["grid_w"] + 30
            canvas[tac_y0:tac_y0 + gR["grid_h"], rx:rx + gR["grid_w"]] = cR
            canvas[H - tl_h:H] = G.timeline(W, Lts, LP, Rts, RP, cam_t0, t_rel)
            labels += [(f"{name}  t={t_rel:5.1f}s  frame {fi}/{n_cam}"
                        f"   L-total {LP[min(iL, len(LP) - 1)]:5.1f}"
                        f"  R-total {RP[min(iR, len(RP) - 1)]:5.1f}",
                        (top_x0 + 6, 4), 20, (245, 245, 245)),
                       (f"color = normal force (0-{vmax:g}{gamma_txt}), "
                        "white arrow = shear; normalized anatomical hand layout",
                        (top_x0 + 6, TITLE - 22 + PANEL_H + 4), 15, (200, 210, 220))]
            canvas = G.draw_texts(canvas, labels)
            writer.append_data(np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
            if not quiet and fi % 120 == 0:
                print(f"  [{name}] {fi}/{n_cam}", flush=True)
    finally:
        writer.close()
        for c in caps.values():
            c.release()
        dproc.kill()
    return f"wrote {out_path}"


def _render_job(args):
    name, ep, out_path, stride = args
    tmp = out_path[:-4] + ".partial.mp4"  # never leave a truncated video under the final name
    try:
        msg = render_episode(ep, tmp, stride=stride, name=name)
        if msg.startswith("wrote"):
            os.replace(tmp, out_path)
    except Exception as e:                # one bad episode must not stop the batch
        msg = f"{type(e).__name__}: {e}"
    if os.path.exists(tmp):
        os.remove(tmp)
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    return name, ok, "" if ok else msg[-200:]


def render(episodes, out_dir, jobs=4, stride=2):
    """Render [(name, episode_dir), ...] in parallel -> {name: mp4 path}.

    `name` becomes the output file name (the pipeline uses the first 8 characters
    of the episode UUID). Existing outputs are reused, so interrupted runs resume.
    """
    os.makedirs(out_dir, exist_ok=True)
    done, todo = {}, []
    for name, ep in episodes:
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        if os.path.exists(mp4) and os.path.getsize(mp4) > 0:
            done[name] = mp4
        else:
            todo.append((name, os.path.abspath(ep), mp4, stride))
    if todo:
        print(f"[render] {len(todo)} to render ({len(done)} already exist), {jobs} workers")
        with ProcessPoolExecutor(jobs) as ex:
            futs = {ex.submit(_render_job, t): t[0] for t in todo}
            for fut in as_completed(futs):
                try:
                    name, ok, err = fut.result()
                except Exception as e:      # e.g. BrokenProcessPool after a worker was killed
                    name, ok, err = futs[fut], False, f"{type(e).__name__}: {e}"
                if ok:
                    done[name] = os.path.join(out_dir, f"{name}.mp4")
                    print(f"  ok   {name}", flush=True)
                else:
                    print(f"  FAIL {name}  {err}", flush=True)
    return done
