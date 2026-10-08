"""Read-back check of a built dual-arm canonical dataset through N0-VTLA's real input pipeline.

A. Per-episode transform check (first / middle frame of every episode): LocalLeRobotV3Dataset ->
   CanonicalTactileInputs -> ChunkDeltaToCurrentState -> RelRotAbsoluteActions round trip for
   BOTH arms, 3 RGB + 2 tactile slots present, the other 2 tactile slots masked, action_mask
   [0:9] [10:19] [20:32] True and the two gripper dims False.
B. Training-batch check: builds the real vtla_tactile_posttrain data loader (norm stats from
   assets/vtla_tactile_posttrain/<asset-id>/) and pulls one batch; prints the exact state/action/
   image shapes the model will see and the range of the normalized (training-target) actions.

Run in the n0vtla environment (jax/torch/pyav), from the N0-VTLA repo root:
  python scripts/verify_robot_dual_dataset.py --dataset DIR --asset-id my_dual_arm_task
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    os.environ["VTLA_DATASET_PATH"] = str(args.dataset)
    os.environ["VTLA_ASSET_ID"] = args.asset_id
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import json

    import robot_dual_arm_adapter as nd
    import n0vtla.transforms as T
    from n0vtla.policies.canonical_tactile_policy import CanonicalTactileInputs
    from n0vtla.training import config as _config
    from n0vtla.training import data_loader as _data

    episodes = [json.loads(l) for l in (args.dataset / "meta/episodes.jsonl").open()]
    starts = np.concatenate([[0], np.cumsum([e["length"] for e in episodes])])
    ds = _data.LocalLeRobotV3Dataset(args.dataset, delta_timestamps={"action": [i / 30 for i in range(50)]},
                                     tolerance_s=0.04, video_backend="pyav")
    to_model = CanonicalTactileInputs(latent_tactile=False)
    nonpad_end = 0
    for k, ep in enumerate(episodes):
        for frac in (0.0, 0.5):
            item = ds[int(starts[k] + frac * (ep["length"] - 1))]
            assert list(item["action"].shape) == [50, 32], item["action"].shape
            raw = {a: (b.numpy() if hasattr(b, "numpy") else b) for a, b in item.items()}
            raw["state"], raw["actions"] = raw["observation.state"], raw["action"]
            inp = to_model(raw)
            m = inp["image_mask"]
            assert m["base_0_rgb"] and m["left_wrist_0_rgb"] and m["right_wrist_0_rgb"], (k, "rgb")
            assert m["right_wrist_right_tactile"] and m["left_wrist_left_tactile"], (k, "tactile")
            assert not m["right_wrist_left_tactile"] and not m["left_wrist_right_tactile"], (k, "tactile mask")
            state, actions, mask = raw["state"].astype(np.float32), raw["actions"].astype(np.float32), raw["action_mask"]
            assert mask[0:9].all() and mask[10:19].all() and mask[20:32].all() and not mask[9] and not mask[19], k
            rel = T.ChunkDeltaToCurrentState()({"state": state.copy(), "actions": actions.copy(), "action_mask": mask})
            for name, a in (("left", 0), ("right", 10)):
                assert not np.allclose(rel["actions"][:, a:a + 3], actions[:, a:a + 3]), (k, name, "not made relative")
            back = T.RelRotAbsoluteActions()({"state": state.copy(), "actions": rel["actions"].copy(), "action_mask": mask})
            np.testing.assert_allclose(back["actions"], actions, atol=2e-4)
            arms, hands = nd.decode_dual(actions)
            np.testing.assert_allclose(nd.pack_dual(arms, hands), actions, atol=2e-4)
            assert np.isfinite(inp["state"]).all() and np.isfinite(inp["actions"]).all(), k
            nonpad_end += int(not np.asarray(item["action_is_pad"]).all())
    print(f"A. PASS: {len(episodes)} episodes x 2 frames, both arms relative-rotation round trip, "
          f"3 RGB + 2 tactile slots, masks as expected")

    config = _config.get_config("vtla_tactile_posttrain")
    object.__setattr__(config, "batch_size", args.batch_size)
    object.__setattr__(config, "num_workers", 0)
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    obs, actions = next(iter(loader))
    print("B. batch shapes:")
    print("   state", tuple(obs.state.shape), " actions", tuple(actions.shape))
    for k, v in sorted(obs.images.items()):
        print(f"   image {k:28s} {tuple(v.shape)}  mask={np.asarray(obs.image_masks[k]).astype(int).tolist()}")
    a = np.asarray(actions)
    st = np.asarray(obs.state)
    assert np.isfinite(a).all() and np.isfinite(st).all()
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("   normalized action |max| per group  L-arm[0:9]:", np.abs(a[..., 0:9]).max(),
          " R-arm[10:19]:", np.abs(a[..., 10:19]).max(), " hands[20:32]:", np.abs(a[..., 20:32]).max())
    print("   normalized state |max| per group   L-arm[0:9]:", np.abs(st[..., 0:9]).max(),
          " R-arm[10:19]:", np.abs(st[..., 10:19]).max(), " hands[20:32]:", np.abs(st[..., 20:32]).max())
    print("B. PASS")


if __name__ == "__main__":
    main()
