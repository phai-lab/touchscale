"""Reference client for a dual-arm N0-VTLA policy (see docs/REAL_ROBOT_INFERENCE_DUAL.md).

Shows, in one place, everything a robot-side client has to get right:
  * encode_tactile():   raw taxel readings of one hand -> the 224x224 pressure canvas the model was trained on
  * DualArmClient:      reset() / infer() with per-hand held baselines, state layout, observation keys
  * decode_actions():   the 50x32 absolute response -> per-arm xyz mm + axis-angle degrees, per-hand motor targets

It does NOT drive hardware. As a runnable demo and integration test it REPLAYS a recorded episode package
(raw mp4 at native resolution, raw glove readings, the commands in force) through a running
serve_policy.py websocket server, so the live-encoding path (native-resolution RGB + raw taxel -> canvas) is
the one a real client would use, and prints what would be sent to each arm/hand next to the demonstrated one.

  python scripts/serve_policy.py --policy.config=vtla_tactile_posttrain --policy.dir=<ckpt> --low-cpu-mem-usage &
  python scripts/dual_arm_client_example.py --package <release>/<uuid> --norm tactile_norm_dual.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import itw_pressure as ip
import robot_dual_arm_adapter as nd

PROMPT = "Perform the task."
STATE_DIM, CHUNK = 32, 50


def encode_tactile(frame_by_pad: dict, norm: dict, hand: str) -> np.ndarray:
    """One hand's raw readings -> (224, 224, 3) uint8 pressure canvas.

    frame_by_pad[pad] is the 2-D taxel grid of pad `pad` for ONE time step (same grids as the recorded
    tactile_<pad> arrays, e.g. tactile_0 is 16x10). hand is "left"/"right" = the ROBOT hand the glove is on;
    `norm` is the tactile normalization JSON fitted by fit_robot_dual_tactile_norm.py.
    """
    arrays = {str(p): ip.normalize_pressure(np.asarray(frame_by_pad[p], np.float64)[None], norm, hand, p)
              for p in ip.PAD_IDS}
    return ip.rasterize_pressure_frame(arrays, 0, hand=hand)


def decode_actions(actions: np.ndarray) -> dict:
    """(50, 32) absolute response -> {"left"/"right": {"xyz_mm": (50,3), "axis_angle_deg": (50,3), "hand": (50,6)}}.
    Each arm's pose is in ITS OWN base frame; hand values are Revo2 motor targets (clip to 0..1000 before sending)."""
    arms, hands = nd.decode_dual(np.asarray(actions))
    return {s: {"xyz_mm": arms[s][:, :3], "axis_angle_deg": arms[s][:, 3:], "hand": np.clip(hands[s], 0, 1000)}
            for s in nd.SIDES}


class DualArmClient:
    def __init__(self, policy, norm: dict):
        self.policy, self.norm = policy, norm
        self.baseline: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        """Call at the start of every rollout: the next infer() stores each hand's first tactile frame as baseline."""
        self.baseline = {}

    def infer(self, state: np.ndarray, head_rgb, left_wrist_rgb, right_wrist_rgb,
              left_pads: dict, right_pads: dict) -> dict:
        """state: float32[32] = LAST COMMANDED target per arm/hand (see nd.pack_dual); images HWC uint8 RGB, any size;
        left_pads/right_pads: raw taxel grids of the glove on the robot LEFT / RIGHT hand."""
        assert np.asarray(state).shape == (STATE_DIM,)
        tactile = {}
        for side, pads in (("left", left_pads), ("right", right_pads)):
            current = encode_tactile(pads, self.norm, side)
            self.baseline.setdefault(side, current)
            tactile[side] = np.stack([self.baseline[side], current])          # (2, 224, 224, 3) [baseline, current]
        obs = {
            "observation.state": np.asarray(state, np.float32),
            "observation.image.third_view": head_rgb,
            "observation.image.left_wrist_view": left_wrist_rgb,
            "observation.image.right_wrist_view": right_wrist_rgb,
            "observation.image.left_wrist_left_tactile": tactile["left"],
            "observation.image.right_wrist_right_tactile": tactile["right"],
            "prompt": PROMPT,
        }
        out = self.policy.infer(obs)
        return {"decoded": decode_actions(out["actions"]), "raw": out["actions"], "infer_ms": out["policy_timing"]["infer_ms"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True, help="one recorded episode dir (needs the mp4/csv/npz/robot files)")
    ap.add_argument("--norm", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--queries", type=int, default=6, help="number of 50-step chunks to request")
    args = ap.parse_args()

    from n0vtla_client.websocket_client_policy import WebsocketClientPolicy

    norm = ip.load_normalization(args.norm)
    w = nd.resolve_dual_window(args.package)
    run = max(w["runs"], key=len)
    client = DualArmClient(WebsocketClientPolicy(args.host, args.port), norm)
    client.reset()
    caps = {n: cv2.VideoCapture(str(args.package / f"{n}.mp4")) for n in nd.RGB_KEY}
    glove = {s: np.load(args.package / nd.GLOVE_FILE[s], allow_pickle=False) for s in nd.SIDES}

    def frame(name, idx):
        caps[name].set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = caps[name].read()
        assert ok, f"cannot read {name} frame {idx}"
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    def pads(side, idx):
        return {p: glove[side][f"tactile_{p}"][idx] for p in ip.PAD_IDS}

    arm_aa = {s: w["arm"][s] for s in nd.SIDES}
    hand = {s: w["hand"][s] for s in nd.SIDES}
    absolute = nd.pack_dual(arm_aa, hand)                      # (N, 32) commands in force at every row
    # the first infer() call stores the rollout's first tactile frame as each hand's baseline, as after reset()
    errs = []
    for q in range(args.queries):
        row = run[min(q * CHUNK, len(run) - 1)]
        res = client.infer(absolute[row], frame("rgb_head", w["video"]["rgb_head"][row]),
                           frame("wrist_left", w["video"]["wrist_left"][row]),
                           frame("wrist_right", w["video"]["wrist_right"][row]),
                           pads("left", w["tactile"]["left"][row]), pads("right", w["tactile"]["right"][row]))
        gt = absolute[row:row + CHUNK]
        n = len(gt)
        line = f"[{q + 1}/{args.queries}] row {row} infer {res['infer_ms']:.0f} ms |"
        for side, (lo, rlo, rhi) in (("left", (0, 3, 9)), ("right", (10, 13, 19))):
            d = np.linalg.norm(res["raw"][:n, lo:lo + 3] - gt[:, lo:lo + 3], axis=1).mean()
            line += f" {side} xyz err {d:.1f} mm"
            errs.append(d)
        a0 = {s: res["decoded"][s] for s in nd.SIDES}
        line += (f" | step0 R xyz {np.round(a0['right']['xyz_mm'][0], 1).tolist()} hand {np.round(a0['right']['hand'][0]).astype(int).tolist()}"
                 f" | L xyz {np.round(a0['left']['xyz_mm'][0], 1).tolist()} hand {np.round(a0['left']['hand'][0]).astype(int).tolist()}")
        print(line, flush=True)
    print(json.dumps({"queries": args.queries, "mean_xyz_err_mm": float(np.mean(errs))}))


if __name__ == "__main__":
    main()
