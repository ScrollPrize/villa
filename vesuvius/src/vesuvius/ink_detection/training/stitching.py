"""Non-overlapping planar tiled forward for oversized flat training patches."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def run_model_forward(
    model,
    image_BCZYX: torch.Tensor,
    model_crop_size: tuple[int, int, int],
    *,
    stitched: bool = True,
    use_gradient_checkpointing: bool = True,
):
    """Return the model's ink tensor or pyramid, stitched at its own scale."""

    if not stitched:
        return model(image_BCZYX)["ink"]
    if image_BCZYX.ndim != 5:
        raise ValueError(
            f"stitched input must be BCZYX, got {tuple(image_BCZYX.shape)}"
        )

    _, _, depth, height, width = image_BCZYX.shape
    crop_depth, crop_height, crop_width = model_crop_size
    if depth != crop_depth:
        raise ValueError(
            f"stitched depth {depth} does not match model crop {crop_depth}"
        )
    if height % crop_height or width % crop_width:
        raise ValueError(
            "stitched Y/X dimensions must divide exactly into model crops: "
            f"{(height, width)!r} vs {(crop_height, crop_width)!r}"
        )
    if height < crop_height or width < crop_width:
        raise ValueError("stitched input cannot be smaller than the model crop")
    if height == crop_height and width == crop_width:
        return model(image_BCZYX)["ink"]

    def scaled_extent(total_extent: int, tile_extent: int, crop_extent: int):
        scaled, remainder = divmod(total_extent * tile_extent, crop_extent)
        if remainder:
            raise ValueError("tile output scale does not divide stitched extent")
        return scaled

    def scaled_bounds(
        start: int, end: int, tile_extent: int, crop_extent: int
    ) -> tuple[int, int]:
        scaled_start, start_remainder = divmod(
            start * tile_extent, crop_extent
        )
        scaled_end, end_remainder = divmod(end * tile_extent, crop_extent)
        if start_remainder or end_remainder:
            raise ValueError("tile output scale does not divide stitched bounds")
        return scaled_start, scaled_end

    def allocate(tile_prediction):
        if isinstance(tile_prediction, (list, tuple)):
            return [allocate(level) for level in tile_prediction]
        output_height = scaled_extent(
            height, tile_prediction.shape[-2], crop_height
        )
        output_width = scaled_extent(
            width, tile_prediction.shape[-1], crop_width
        )
        return tile_prediction.new_empty(
            *tile_prediction.shape[:-2], output_height, output_width
        )

    def write(output, tile_prediction, *, y0, y1, x0, x1):
        if isinstance(tile_prediction, (list, tuple)):
            for output_level, prediction_level in zip(
                output, tile_prediction, strict=True
            ):
                write(
                    output_level,
                    prediction_level,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
            return
        y0_scaled, y1_scaled = scaled_bounds(
            y0, y1, tile_prediction.shape[-2], crop_height
        )
        x0_scaled, x1_scaled = scaled_bounds(
            x0, x1, tile_prediction.shape[-1], crop_width
        )
        output[..., y0_scaled:y1_scaled, x0_scaled:x1_scaled] = tile_prediction

    stitched_output = None
    tile_prediction = None
    for y0 in range(0, height, crop_height):
        y1 = y0 + crop_height
        for x0 in range(0, width, crop_width):
            x1 = x0 + crop_width
            image_tile = image_BCZYX[:, :, :, y0:y1, x0:x1]

            def forward_ink(tile):
                return model(tile)["ink"]

            if use_gradient_checkpointing:
                tile_prediction = checkpoint(
                    forward_ink, image_tile, use_reentrant=False
                )
            else:
                tile_prediction = forward_ink(image_tile)
            if stitched_output is None:
                stitched_output = allocate(tile_prediction)
            write(
                stitched_output,
                tile_prediction,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
            )

    if isinstance(tile_prediction, (list, tuple)):
        return type(tile_prediction)(stitched_output)
    return stitched_output
