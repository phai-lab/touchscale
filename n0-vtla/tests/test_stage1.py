"""Small CPU regressions using production loss methods, without pretrained downloads.

Run from the repo root with the project environment installed: python -m unittest discover -s tests -p test_stage1.py -v
"""
import copy
import dataclasses
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy
from n0vtla.models_pytorch.tactile_recon_head import TactileReconHead
from n0vtla.policies.canonical_tactile_policy import CanonicalTactileInputs, Stage1ObservationOnly


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tactile_proj = nn.Linear(3, 4)

    def forward(self, x):
        return self.tactile_proj(x.mean((-2, -1))).unsqueeze(1).expand(-1, 10, -1)


class TinyPolicy(N0VTLAPolicy):
    """Replace heavyweight encoders only; execute the real Stage-1 forward/loss."""
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(stage1_temperature=0.07, stage1_lambda_rec=0.5)
        self.stage1_pretrain_enabled = True
        self.tactile_encoder = TinyEncoder()
        self.tactile_predictor = nn.Linear(4, 4)
        self.tactile_recon_head = TactileReconHead(4, grid=2)

    def _preprocess_observation(self, obs, *, train=True):
        self._last_tac_t = {"left": obs["current"]}
        self._last_tac_f = {"left": obs["future"]}
        self._last_tac_mask = {"left": torch.ones_like(obs["valid"])}
        self._last_tac_mask_f = {"left": obs["valid"]}
        return (None,) * 6

    def _prefix_forward(self, *args, **kwargs):
        return torch.zeros(1), None, None, None, None

    def _compute_z(self, *args):
        g = self.tactile_encoder(self._last_tac_t["left"])
        return self.tactile_predictor(g), g, self._last_tac_mask["left"]

    def forward(self, obs):
        return self.forward_stage1(obs)


def sample():
    torch.manual_seed(91)
    return {"current": torch.randn(4, 3, 4, 4), "future": torch.randn(4, 3, 4, 4),
            "valid": torch.tensor([False, False, True, True])}


def distributed_worker(rank, init_file):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = TinyPolicy()
    policy = copy.deepcopy(reference)
    batch = sample()
    expected_loss = reference(batch)
    expected_loss.backward()
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        ddp = DistributedDataParallel(policy)
        local = {k: v[rank * 2:(rank + 1) * 2] for k, v in batch.items()}
        loss = ddp(local)
        loss.backward()
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(mean_loss / 2, expected_loss.detach())
        for actual, expected in zip(policy.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
        # All ranks invalid must still complete the backward collectives.
        ddp.zero_grad(set_to_none=True)
        local["valid"] = torch.zeros(2, dtype=torch.bool)
        empty_loss = ddp(local)
        empty_loss.backward()
        assert empty_loss.item() == 0
    finally:
        dist.destroy_process_group()


class Stage1Tests(unittest.TestCase):
    def test_infonce_matches_equation(self):
        policy = TinyPolicy()
        policy.config.stage1_temperature = 1.0
        z = torch.randn(4, 10, 4, requires_grad=True)
        target = torch.randn(4, 10, 4)
        valid = torch.tensor([True, False, True, True])
        h = F.normalize(z[valid].mean(1), dim=-1)
        ht = F.normalize(target[valid].mean(1), dim=-1)
        logits = h @ ht.T
        expected = (F.cross_entropy(logits, torch.arange(3)) +
                    F.cross_entropy(logits.T, torch.arange(3))) / 2
        actual = policy._stage1_infonce_loss(z, target, valid)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(z.grad[1].abs().sum().item(), 0)

    def test_empty_and_singleton_pools_backward(self):
        for valid in (torch.tensor([False, False]), torch.tensor([True, False])):
            z = torch.randn(2, 10, 4, requires_grad=True)
            loss = TinyPolicy()._stage1_infonce_loss(z, z.detach(), valid)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(z.grad.abs().sum().item(), 0)

    def test_future_target_intersects_current_masks(self):
        policy = TinyPolicy()
        left = torch.ones(2, 3, 4, 4)
        right = left * 3
        policy._last_tac_mask = {"left": torch.tensor([True, False]),
                                 "right": torch.tensor([False, True])}
        future = {"left": left, "right": right}
        current = {key: torch.zeros_like(value) for key, value in future.items()}
        target, field, valid = policy._build_future_target(future, current, None)
        torch.testing.assert_close(field[0], left[0])
        torch.testing.assert_close(field[1], right[1])
        self.assertTrue(valid.all())
        self.assertFalse(target.requires_grad)

    def test_complete_forward_gradients(self):
        policy = TinyPolicy()
        loss = policy(sample())
        loss.backward()
        for name, param in policy.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        self.assertGreater(policy.tactile_encoder.tactile_proj.weight.grad.abs().sum(), 0)

    def test_observation_only_and_tail_mask(self):
        key = "observation.image.left_wrist_left_tactile"
        data = {key: np.zeros((3, 4, 4, 3), dtype=np.uint8),
                key + "_is_pad": np.array([True, False, True])}
        out = CanonicalTactileInputs()(Stage1ObservationOnly(32, 50)(data))
        self.assertEqual(out["actions"].shape, (50, 32))
        self.assertEqual(out["state"].shape, (32,))
        self.assertFalse(out["image_mask"]["left_wrist_left_tactile.future"])
        self.assertTrue(out["image_mask"]["left_wrist_left_tactile.baseline"])
        self.assertNotIn("action", data)

    def test_stage1_config_removes_action_dependencies_only(self):
        from n0vtla import transforms
        from n0vtla.training import config

        stage1 = config.get_config("vtla_stage1_predictor_pretrain")
        posttrain = config.get_config("vtla_tactile_posttrain")
        # Avoid downloading tokenizers or reading machine-specific norm assets.
        with mock.patch.object(config.ModelTransformFactory, "__call__", return_value=transforms.Group()), \
             mock.patch.object(config.DataConfigFactory, "_load_norm_stats", return_value=None):
            data = stage1.data.create(Path("unused-stage1-test-assets"), stage1.model)
            robot = posttrain.data.create(Path("unused-stage1-test-assets"), posttrain.model)
        self.assertEqual(data.action_sequence_keys, ())
        self.assertEqual(robot.action_sequence_keys, ("action",))
        self.assertFalse(any(isinstance(t, transforms.DeltaActions) for t in data.data_transforms.inputs))
        self.assertTrue(any(isinstance(t, transforms.DeltaActions) for t in robot.data_transforms.inputs))
        key = "observation.image.left_wrist_left_tactile"
        row = {key: np.zeros((3, 4, 4, 3), dtype=np.uint8)}
        for transform in (*data.repack_transforms.inputs, *data.data_transforms.inputs):
            row = transform(row)
        self.assertEqual(row["actions"].shape, (50, 32))
        self.assertEqual(len(data.extra_delta_timestamps[key]), 3)

    def test_checkpoint_resume_counts_completed_updates(self):
        from scripts.train_stage1_predictor import (
            _Stage1Wrapper, load_stage1_checkpoint, save_stage1_checkpoint,
        )

        @dataclasses.dataclass
        class CheckpointConfig:
            checkpoint_dir: Path
            num_train_steps: int = 2
            save_interval: int = 100

        policy = TinyPolicy()
        policy.register_parameter("z_gate", nn.Parameter(torch.tensor([0.3]), requires_grad=False))
        optimizer = torch.optim.AdamW(p for p in policy.parameters() if p.requires_grad)
        policy(sample()).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = CheckpointConfig(Path(tmp))
            save_stage1_checkpoint(_Stage1Wrapper(policy), optimizer, 2, cfg, True)
            restored = TinyPolicy()
            restored_optim = torch.optim.AdamW(p for p in restored.parameters() if p.requires_grad)
            step = load_stage1_checkpoint(_Stage1Wrapper(restored), restored_optim, Path(tmp), "cpu")
            self.assertEqual(step, 2)
            for a, b in zip(policy.parameters(), restored.parameters(), strict=True):
                torch.testing.assert_close(a, b)
            metadata_path = Path(tmp) / "2" / "metadata.pt"
            metadata = torch.load(metadata_path, weights_only=False)
            metadata.pop("step_format")
            torch.save(metadata, metadata_path)
            self.assertEqual(load_stage1_checkpoint(
                _Stage1Wrapper(restored), restored_optim, Path(tmp), "cpu"), 3)

    def test_ddp_global_loss_and_gradients_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_worker, args=(str(Path(tmp) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
