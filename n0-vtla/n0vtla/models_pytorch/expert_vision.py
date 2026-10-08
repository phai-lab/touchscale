import os
import pathlib
import sys

import torch
from torch import nn

_DEFAULT_DINOV3_REPO = "/path/to/workspace/models/dinov3/repo"


def _import_dinov3_builder():
    try:
        from dinov3.hub.backbones import dinov3_vitb16

        return dinov3_vitb16
    except ModuleNotFoundError:
        repo_path = pathlib.Path(os.environ.get("N0VTLA_DINOV3_REPO", _DEFAULT_DINOV3_REPO))
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)
            from dinov3.hub.backbones import dinov3_vitb16

            return dinov3_vitb16
    raise ImportError(
        "DINOv3 package not found. Install the official `dinov3` package or clone the repo to "
        f"`{_DEFAULT_DINOV3_REPO}` (or set `N0VTLA_DINOV3_REPO`)."
    )


class DINOv3VisionBackbone(nn.Module):
    def __init__(self, weights_path: str | None = None):
        super().__init__()
        builder = _import_dinov3_builder()
        self.model = builder(pretrained=False)

        if weights_path is not None:
            weights = pathlib.Path(weights_path)
            if not weights.is_file():
                raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)

        embed_dim = getattr(self.model, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("Failed to infer DINOv3 embed_dim from the loaded backbone.")
        self.embed_dim = embed_dim

    def forward_features(self, pixel_values: torch.Tensor):
        return self.model.forward_features(pixel_values)

