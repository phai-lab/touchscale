from collections.abc import Sequence
import logging

import torch

from n0vtla.shared import image_tools

logger = logging.getLogger("n0vtla")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def _preprocess_image_dict(
    images,
    image_masks,
    batch_shape,
    *,
    image_resolution: tuple[int, int],
    train: bool,
):
    out_images = {}
    for key, image in images.items():
        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            image = image / 2.0 + 0.5

            if "wrist" not in key:
                height, width = image.shape[1:3]
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

                angle = torch.rand(1, device=image.device) * 10 - 5
                if torch.abs(angle) > 0.1:
                    angle_rad = angle * torch.pi / 180.0
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
            image = image * brightness_factor

            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0

        if is_channels_first:
            image = image.permute(0, 3, 1, 2)

        out_images[key] = image

    out_masks = {}
    masks = image_masks or {}
    for key in out_images:
        if key not in masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=next(iter(out_images.values())).device)
        else:
            out_masks[key] = masks[key]

    return out_images, out_masks


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]
    out_images, out_masks = _preprocess_image_dict(
        {key: observation.images[key] for key in image_keys},
        observation.image_masks,
        batch_shape,
        image_resolution=image_resolution,
        train=train,
    )

    out_expert_images = None
    out_expert_masks = None
    if getattr(observation, "expert_images", None) is not None:
        out_expert_images, out_expert_masks = _preprocess_image_dict(
            observation.expert_images,
            getattr(observation, "expert_image_masks", None),
            batch_shape,
            image_resolution=image_resolution,
            train=False,
        )

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        expert_images=out_expert_images,
        expert_image_masks=out_expert_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
