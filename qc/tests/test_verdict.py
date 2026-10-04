"""Loss classification and noise-candidate logic (step 2), on synthetic signals."""
import numpy as np

from touchscale_qc import verdict as V

T = np.arange(0, 10, 0.1)


def _per(**active):
    """Per-part force; each kwarg is a list of (start, end) windows with force 1.0."""
    per = {p: np.zeros_like(T) for p in ["thumb", "index", "middle", "ring", "pinky", "palm"]}
    for part, windows in active.items():
        for a, b in windows:
            per[part][(T >= a) & (T <= b)] = 1.0
    return per


def _grasp(confirmed, start=2.0, end=5.0):
    return {"contacts": [{"start": start, "end": end, "confirmed": confirmed, "object": "cup"}]}


def test_two_dead_confirmed_fingers_is_high_confidence():
    hard, weak, soft = V.check_loss(T, _per(), _grasp(["thumb", "index"]))
    assert [h["finger"] for h in hard] == ["thumb", "index"] and not weak and not soft


def test_single_dead_finger_goes_to_review():
    hard, weak, soft = V.check_loss(T, _per(thumb=[(2, 5)]), _grasp(["thumb", "index"]))
    assert not hard and [w["finger"] for w in weak] == ["index"] and not soft


def test_finger_silent_here_but_alive_elsewhere_is_only_suspect():
    per = _per(thumb=[(2, 5)], index=[(7, 8)])
    hard, weak, soft = V.check_loss(T, per, _grasp(["thumb", "index"]))
    assert not hard and not weak and [s["finger"] for s in soft] == ["index"]


def test_pinky_palm_and_uncertain_grasps_are_never_judged():
    assert V.check_loss(T, _per(), _grasp(["pinky", "palm"])) == ([], [], [])
    r1 = _grasp(["thumb", "index"])
    r1["contacts"][0]["uncertain_interval"] = True
    assert V.check_loss(T, _per(), r1) == ([], [], [])


def test_noise_candidates_only_inside_free_time():
    n = len(T)
    slots = {s: np.zeros((n, 2, 2), np.float32) for s in V.T.PADS}
    slots[3][(T >= 1) & (T <= 2)] = 0.5    # index tip lit while "free"
    slots[3][(T >= 6) & (T <= 7)] = 0.5    # index tip lit during contact -> not noise
    per = _per(index=[(1, 2), (6, 7)])
    r1 = {"free": [[0.0, 4.0]], "contacts": [{"start": 5.0, "end": 8.0}]}
    cands = V.noise_candidates(T, per, slots, n, r1)
    assert len(cands) == 1
    assert abs(cands[0]["start"] - 1.0) < 1e-6 and abs(cands[0]["end"] - 2.0) < 1e-6
    assert cands[0]["fingers"] == ["index"]
