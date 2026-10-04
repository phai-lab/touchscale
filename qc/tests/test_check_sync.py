"""Timestamp sync gate on synthetic recordings."""
import numpy as np

import check_sync as S

T0 = 1.7e9                                  # UNIX-epoch-like start time


def _write_csv(path, ts):
    path.write_text("frame_index,timestamp_s\n"
                    + "".join(f"{i},{t:.6f}\n" for i, t in enumerate(ts)))


def _make_recording(root, glove_ts=None, drop_wrist=None):
    ed = root / "rec0"
    ed.mkdir()
    cam = T0 + np.arange(300) / 30.0        # 10 s at 30 Hz
    for s in ["rgb_head", "depth_head", "wrist_left", "wrist_right"]:
        _write_csv(ed / f"{s}.csv", cam)
        (ed / f"{s}.{'mkv' if s == 'depth_head' else 'mp4'}").write_bytes(b"x")
    g = T0 - 0.02 + np.arange(520) / 50.0 if glove_ts is None else glove_ts
    for side in ["left", "right"]:
        np.savez(ed / f"{side}_hand_data.npz", timestamps=g)
    if drop_wrist:
        (ed / "wrist_right.mp4").unlink()
    return ed


def test_clean_recording_passes(tmp_path):
    r = S.check_recording(str(_make_recording(tmp_path)))
    assert r["PASS"], r["reason"]
    assert r["aligned_frac"] >= 0.95 and r["n_holes"] == 0


def test_glove_blackout_fails(tmp_path):
    g = T0 - 0.02 + np.arange(520) / 50.0
    g = np.concatenate([g[:200], g[230:]])  # 0.6 s with no glove samples
    r = S.check_recording(str(_make_recording(tmp_path, glove_ts=g)))
    assert not r["PASS"]
    assert "blackout" in r["reason"] and r["worst_stream"] in ("tacL", "tacR")


def test_missing_video_file_fails(tmp_path):
    r = S.check_recording(str(_make_recording(tmp_path, drop_wrist=True)))
    assert not r["PASS"] and "wrist_right.mp4" in r["reason"]


def test_find_recordings_and_task_name(tmp_path):
    ed = _make_recording(tmp_path)
    (ed / "task_info.json").write_text('{"name": "wipe the table"}')
    assert S.find_recordings(str(tmp_path)) == [str(ed)]
    assert S.task_of(str(ed)) == "wipe the table"


def test_corrupt_glove_file_is_reported_not_raised(tmp_path):
    ed = _make_recording(tmp_path)
    (ed / "left_hand_data.npz").write_bytes(b"")            # truncated upload
    r = S.check_recording(str(ed))
    assert not r["PASS"] and "left glove" in r["reason"]
