import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_stage1_report import sample_records, signed_heatmap, summarize


class ReportTests(unittest.TestCase):
    def test_sample_records_respects_horizon_and_episode_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            episodes = [{"episode_index": 0, "length": 61}, {"episode_index": 1, "length": 56}]
            (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
            np.testing.assert_array_equal(sample_records(root), [[0, 0, 0], [0, 5, 5], [0, 10, 10],
                                                                 [1, 0, 61], [1, 5, 66]])

    def test_zero_baseline_and_active_subset(self):
        target = np.stack([np.zeros((8, 8)), np.full((8, 8), .1)])
        records = np.array([[0, 0, 0], [1, 0, 51]])
        result = summarize(np.zeros_like(target), target, records)
        self.assertAlmostEqual(result["mae"], result["zero_mae"])
        self.assertEqual(result["active_samples"], 1)
        self.assertAlmostEqual(result["active_mae"], .1)
        self.assertEqual(summarize(target, target, records)["mae"], 0)

    def test_signed_scale_fixed_and_zero_white(self):
        np.testing.assert_array_equal(signed_heatmap(np.zeros((8, 8)))[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(signed_heatmap(np.full((8, 8), .1))[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(signed_heatmap(np.full((8, 8), -.1))[0, 0], [0, 0, 255])


if __name__ == "__main__":
    unittest.main()
