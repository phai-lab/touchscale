"""Entry point for VTLA tactile-predictor (v1) PyTorch training.

Monkey-patches n0vtla's ``PI0Pytorch`` class with ``N0VTLAPolicy`` (subclass) so that
upstream ``scripts.train_pytorch.main`` constructs the predictor variant WITHOUT any change to
the base training script or the base model. The predictor path only activates when the config's
model carries ``tactile_predictor_enabled=True`` (a ``N0VTLAConfig``); with the flag off,
``N0VTLAPolicy`` is byte-identical to ``PI0Pytorch``, so running ANY existing config through
this entry is a no-op. ``scripts/gate_c_check.py`` asserts that equivalence.

Usage (single node, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/train_n0vtla.py vtla_tactile_posttrain \
        --exp_name=my_experiment
"""
from __future__ import annotations

import n0vtla.models_pytorch.pi0_pytorch as _pp
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAPolicy

# Patch 1: bind the predictor subclass BEFORE importing the train entry, so
# ``train_pytorch.main`` picks it up at ``n0vtla.models_pytorch.pi0_pytorch.PI0Pytorch(cfg)``.
_pp.PI0Pytorch = N0VTLAPolicy  # type: ignore[misc]

# Patch 2: the pretrained base-checkpoint load in train_pytorch.py already uses strict=False, so
# the additive predictor keys (tactile_encoder.*, tactile_predictor.*, z_proj.*, DINOv2 buffers) are
# tolerated. Force strict=False on ALL safetensors.load_model calls anyway (e.g. resume) so a
# predictor checkpoint's extra keys are never fatal on the base symbol.
import safetensors.torch as _st  # noqa: E402

_original_load_model = _st.load_model


def _tolerant_load_model(model, filename, strict=False, device=None):  # type: ignore[no-untyped-def]
    return _original_load_model(model, filename, strict=False, device=device)


_st.load_model = _tolerant_load_model  # type: ignore[assignment]

from scripts.train_pytorch import main  # noqa: E402

if __name__ == "__main__":
    main()
