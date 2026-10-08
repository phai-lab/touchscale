import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from itw_pressure import (
    HANDS,
    PAD_IDS,
    SCHEMA,
    SCHEMA_PER_TASK,
    aligned_timeline,
    fit_normalization,
    load_normalization,
    normalize_pressure,
    pressure_rgb,
)


class PressureTests(unittest.TestCase):
    def test_fit_quantiles_and_corpus_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for h, hand in enumerate(HANDS):
                data = {"timestamps": np.arange(4, dtype=float)}
                for slot, pad in enumerate(PAD_IDS):
                    p = h * 15 + slot
                    scale = 0 if p == 0 else p + 1
                    data[f"tactile_{pad}"] = np.tile(np.arange(4, dtype=np.float32) * scale, (4, 1, 1))
                np.savez(root / f"{hand}_hand_data.npz", **data)
            norm = fit_normalization([root])
            np.testing.assert_array_equal(norm["normal_baseline"], np.zeros(30))
            np.testing.assert_allclose(norm["normal_scale"][1:], np.arange(2, 31) * 3)
            self.assertAlmostEqual(norm["normal_scale"][0], 2.4)
            self.assertEqual(norm, fit_normalization([root]))

    def test_fixed_per_pad_mapping_and_clipping(self):
        norm = {"normal_baseline": [1.] * 30, "normal_scale": [2.] * 30}
        norm["normal_scale"][15] = 4.
        raw = np.array([-100., 1., 3., 100.], dtype=np.float32)
        np.testing.assert_array_equal(normalize_pressure(raw, norm, "left", 0), [-1, 0, 1, 8])
        self.assertEqual(normalize_pressure(raw, norm, "right", 0)[2], .5)
        # No per-episode rescaling: extra extreme observations do not change earlier values.
        np.testing.assert_array_equal(normalize_pressure(raw[:3], norm, "left", 0), [-1, 0, 1])
        with self.assertRaises(ValueError):
            normalize_pressure(np.array([np.nan]), norm, "left", 0)

    def test_per_task_scale_overrides_per_pad_and_falls_back_to_default(self):
        # Same baseline/index math as per-pad; only the SCALE source changes when a
        # task_scale table is present -- ALL pads use the one scale for the resolved
        # task, not their individual normal_scale entries (which the manifest sets
        # uniformly to default_scale, for compatibility with callers that never pass
        # task_name).
        norm = {
            "normal_baseline": [0.] * 30,
            "normal_scale": [999.] * 30,  # must be ignored once task_scale is present
            "task_scale": {"pour water": 2.0},
            "default_scale": 4.0,
        }
        raw = np.array([2.0], dtype=np.float32)
        np.testing.assert_array_equal(normalize_pressure(raw, norm, "left", 0, task_name="pour water"), [1.0])
        # Unknown task and no task_name at all both fall back to default_scale
        # (a missing/unmatched task_name never raises).
        np.testing.assert_array_equal(normalize_pressure(raw, norm, "left", 0, task_name="unseen task"), [0.5])
        np.testing.assert_array_equal(normalize_pressure(raw, norm, "left", 0), [0.5])

    def test_load_normalization_accepts_both_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            per_pad = Path(tmp) / "per_pad.json"
            per_pad.write_text(json.dumps({
                "schema": SCHEMA, "fit_split": "train",
                "normal_baseline": [0.] * 30, "normal_scale": [1.] * 30,
                "contact_threshold": [0.5] * 30,
            }))
            result = load_normalization(per_pad)
            self.assertNotIn("task_scale", result)

            per_task = Path(tmp) / "per_task.json"
            per_task.write_text(json.dumps({
                "schema": SCHEMA_PER_TASK, "fit_split": "train",
                "normal_baseline": [0.] * 30, "normal_scale": [4.] * 30,
                "contact_threshold": [0.5] * 30,
                "task_scale": {"pour water": 2.0}, "default_scale": 4.0,
            }))
            result = load_normalization(per_task)
            self.assertEqual(result["task_scale"], {"pour water": 2.0})

            missing_table = Path(tmp) / "missing_table.json"
            missing_table.write_text(json.dumps({
                "schema": SCHEMA_PER_TASK, "fit_split": "train",
                "normal_baseline": [0.] * 30, "normal_scale": [4.] * 30,
                "contact_threshold": [0.5] * 30,
                "task_scale": {}, "default_scale": 4.0,
            }))
            with self.assertRaisesRegex(ValueError, "non-empty task_scale"):
                load_normalization(missing_table)

    def test_pressure_transport_has_no_shear_or_color(self):
        rgb = pressure_rgb(np.array([-1., 0., 8.]))
        np.testing.assert_array_equal(rgb[:, 0], [0, 28, 255])
        np.testing.assert_array_equal(rgb[:, 0], rgb[:, 1])
        np.testing.assert_array_equal(rgb[:, 1], rgb[:, 2])

    def test_common_timeline_and_gap_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def camera(name, drop=False):
                with (root / f"{name}.csv").open("w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame_index", "timestamp_s"])
                    for i in range(90):
                        if not drop or i not in range(35, 45):
                            writer.writerow([i + 2, 1000 + i / 30])
            for view in ("rgb_head", "wrist_left", "wrist_right"):
                camera(view)
            for hand in HANDS:
                np.savez(root / f"{hand}_hand_data.npz", timestamps=1000 + np.arange(120) / 40)
            mapping, audit = aligned_timeline(root)
            self.assertGreater(len(mapping["master_timestamp_ns"]), 50)
            self.assertEqual(mapping["rgb_head_frame_index"][0], 2)
            self.assertTrue(all(v["invalid_frames"] == 0 for v in audit.values()))
            self.assertTrue(all(v.all() for k, v in mapping.items() if k.endswith("_within_tolerance")))
            # A big enough gap (10/90 frames, ~11%) drops the pass rate below the
            # 95% threshold (QC_PASS_RATE) -> whole episode still rejected.
            camera("wrist_left", drop=True)
            with self.assertRaisesRegex(ValueError, "only .*% of frames within"):
                aligned_timeline(root)

    def test_per_task_scale_a_few_bad_frames_stay_within_tolerance(self):
        # The gate is a PASS-RATE threshold, not zero tolerance: a handful of
        # misaligned ticks that stay under 5% of the stream must NOT reject the whole
        # episode -- and the offending frames must be flagged in the per-frame mask,
        # not silently treated as valid.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "rgb_head.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "timestamp_s"])
                for i in range(90):
                    # One single frame (i == 40) is shifted by 25ms -- over the 17.5ms
                    # limit -- everything else is perfectly aligned. 1/90 ~= 1.1% bad.
                    jitter = 0.025 if i == 40 else 0.0
                    writer.writerow([i + 2, 1000 + i / 30 + jitter])
            for view in ("wrist_left", "wrist_right"):
                with (root / f"{view}.csv").open("w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame_index", "timestamp_s"])
                    for i in range(90):
                        writer.writerow([i + 2, 1000 + i / 30])
            for hand in HANDS:
                np.savez(root / f"{hand}_hand_data.npz", timestamps=1000 + np.arange(120) / 40)
            mapping, audit = aligned_timeline(root)
            self.assertEqual(audit["rgb_head"]["invalid_frames"], 1)
            self.assertLess(audit["rgb_head"]["pass_rate"], 1.0)
            self.assertGreaterEqual(audit["rgb_head"]["pass_rate"], 0.95)
            self.assertFalse(mapping["rgb_head_within_tolerance"].all())
            # The other streams have zero jitter -- every frame stays within tolerance.
            self.assertTrue(mapping["wrist_left_within_tolerance"].all())


if __name__ == "__main__":
    unittest.main()
