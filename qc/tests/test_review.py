"""Review material must survive ERROR-shaped results (only {verdict, episode, error})."""
from touchscale_qc import review


def test_error_verdict_does_not_break_review(tmp_path):
    results = {
        "caffec72": {"verdict": "ERROR", "episode": "caffec72-0000",
                     "error": "JSONDecodeError: Extra data: line 2 column 1"},
        "aaaaaaaa": {"verdict": "PASS", "episode": "aaaaaaaa-0000", "noise_seg": 0,
                     "noise_s": 0.0, "noise_pct": 0.0, "hands": {}},
        "bbbbbbbb": {"verdict": "REVIEW", "episode": "bbbbbbbb-0000", "noise_seg": 2,
                     "noise_s": 3.4, "noise_pct": 15.0,
                     "hands": {"left": {"verdict": "REVIEW", "reason": "r", "loss": [],
                                        "weak": [], "suspect": [], "noise": []}}},
    }
    review.build(results, str(tmp_path / "videos"), str(tmp_path / "review"))
    body = (tmp_path / "review" / "README.md").read_text()
    noise_table = body.split("Baseline noise")[-1]
    assert "bbbbbbbb" in noise_table
    assert "caffec72" not in noise_table
