import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_per_task_scale_normalization as bn  # noqa: E402
import itw_pressure as ip  # noqa: E402


def _write_episode(root: Path, name: str, task: str | None, level: float, rng, frames: int = 20) -> Path:
    ep = root / "2026-01-01" / name
    ep.mkdir(parents=True)
    for hand in ip.HANDS:
        arrays = {f"tactile_{p}": (rng.random((frames, 4, 3)) * level).astype(np.float64) for p in ip.PAD_IDS}
        np.savez(ep / f"{hand}_hand_data.npz", timestamps=np.arange(frames) / 30.0, **arrays)
    if task is not None:
        (ep / "task_info.json").write_text(json.dumps({"name": task}))
    return ep


class PerTaskScaleTest(unittest.TestCase):
    def test_scale_follows_task_force_level_and_loads(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eps = [_write_episode(root, f"a{i}", "light press", 0.2, rng) for i in range(4)]
            eps += [_write_episode(root, f"b{i}", "hard squeeze", 5.0, rng) for i in range(4)]
            eps += [_write_episode(root, "rare", "one-off", 1.0, rng)]          # below --min-episodes
            eps += [_write_episode(root, "noname", None, 1.0, rng)]
            norm = bn.build_normalization(eps, min_episodes=3, seed=1)
            out = root / "n.json"
            out.write_text(json.dumps(norm))
            loaded = ip.load_normalization(out)                                  # exact loader used in training
            ts = loaded["task_scale"]
            self.assertEqual(set(ts), {"light press", "hard squeeze"})
            self.assertAlmostEqual(ts["light press"], 0.2, delta=0.02)
            self.assertAlmostEqual(ts["hard squeeze"], 5.0, delta=0.5)
            self.assertGreater(loaded["default_scale"], ts["light press"])
            self.assertLess(loaded["default_scale"], ts["hard squeeze"] + 1e-9)
            self.assertEqual(norm["provenance"]["n_episodes_without_task_name"], 1)
            # normalize_pressure uses the task's scale, and falls back to default for an unknown task
            raw = np.array([0.2])
            light = ip.normalize_pressure(raw, loaded, "left", PAD_IDS0, task_name="light press")
            hard = ip.normalize_pressure(raw, loaded, "left", PAD_IDS0, task_name="hard squeeze")
            unknown = ip.normalize_pressure(raw, loaded, "left", PAD_IDS0, task_name="never seen")
            self.assertGreater(light[0], hard[0])
            self.assertAlmostEqual(float(unknown[0]), float(ip.normalize_pressure(raw, loaded, "left", PAD_IDS0)[0]))

    def test_no_task_names_is_an_error(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            eps = [_write_episode(Path(tmp), f"e{i}", None, 1.0, rng) for i in range(4)]
            with self.assertRaisesRegex(ValueError, "no task has"):
                bn.build_normalization(eps)


PAD_IDS0 = ip.PAD_IDS[0]

if __name__ == "__main__":
    unittest.main()
