"""Batch scanning and the episode-name uniqueness guard."""
import sys

import run_qc


def _episode(root, name, files=run_qc.REQUIRED):
    d = root / name
    d.mkdir()
    for f in files:
        (d / f).write_bytes(b"")


def test_scan_splits_complete_and_incomplete(tmp_path):
    _episode(tmp_path, "aaaaaaaa-1")
    _episode(tmp_path, "bbbbbbbb-1", files=run_qc.REQUIRED[:-1])
    good, bad = run_qc.scan(str(tmp_path))
    assert [g[0] for g in good] == ["aaaaaaaa-1"]
    assert bad[0][0] == "bbbbbbbb-1" and bad[0][2] == ["task_info.json"]


def test_prefix_collision_is_refused(tmp_path, monkeypatch, capsys):
    _episode(tmp_path, "pour_water_01")
    _episode(tmp_path, "pour_water_02")
    monkeypatch.setattr(sys, "argv", ["run_qc.py", "--data", str(tmp_path),
                                      "--out", str(tmp_path / "out")])
    assert run_qc.main() == 1
    assert "pour_wat" in capsys.readouterr().out
