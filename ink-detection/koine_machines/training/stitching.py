import torch
from torch.utils.checkpoint import checkpoint


def resolve_model_and_loader_patch_sizes(config):
    model_crop_size = config['patch_size']
    stitch_factor = int(config.get('stitch_factor', 1))
    assert stitch_factor > 0, stitch_factor

    # The trainer stitches planar sub-crops back together before loss computation.
    # Keep depth aligned with the model crop unless/until 3D output stitching is needed.
    loader_patch_size = (
        model_crop_size[0],
        model_crop_size[1] * stitch_factor,
        model_crop_size[2] * stitch_factor,
    )
    return model_crop_size, loader_patch_size, stitch_factor


def run_model_forward(
    model,
    image,
    model_crop_size,
    *,
    stitched=True,
    use_gradient_checkpointing=True,
):
    if not stitched:
        return model(image)['ink']

    assert image.ndim == 5, image.shape

    _, _, depth, height, width = image.shape
    crop_depth, crop_height, crop_width = model_crop_size
    assert depth == crop_depth, (depth, crop_depth)
    assert height % crop_height == 0 and width % crop_width == 0, (
        (depth, height, width),
        model_crop_size,
    )
    assert height >= crop_height and width >= crop_width

    if height == crop_height and width == crop_width:
        return model(image)['ink']

    def _scaled_extent(total_extent, tile_extent, crop_extent):
        scaled, remainder = divmod(int(total_extent) * int(tile_extent), int(crop_extent))
        assert remainder == 0, ((total_extent, tile_extent, crop_extent), remainder)
        return scaled

    def _scaled_bounds(start, end, tile_extent, crop_extent):
        scaled_start, start_remainder = divmod(int(start) * int(tile_extent), int(crop_extent))
        scaled_end, end_remainder = divmod(int(end) * int(tile_extent), int(crop_extent))
        assert start_remainder == 0 and end_remainder == 0, (
            (start, end, tile_extent, crop_extent),
            (start_remainder, end_remainder),
        )
        return scaled_start, scaled_end

    def _allocate_stitched_output(tile_pred):
        if isinstance(tile_pred, (list, tuple)):
            return [_allocate_stitched_output(pred) for pred in tile_pred]

        scaled_height = _scaled_extent(height, tile_pred.shape[-2], crop_height)
        scaled_width = _scaled_extent(width, tile_pred.shape[-1], crop_width)
        return tile_pred.new_empty(*tile_pred.shape[:-2], scaled_height, scaled_width)

    def _write_stitched_output(stitched_output, tile_pred, *, y0, y1, x0, x1):
        if isinstance(tile_pred, (list, tuple)):
            assert isinstance(stitched_output, list), type(stitched_output)
            assert len(stitched_output) == len(tile_pred), (len(stitched_output), len(tile_pred))
            for output_level, pred_level in zip(stitched_output, tile_pred):
                _write_stitched_output(output_level, pred_level, y0=y0, y1=y1, x0=x0, x1=x1)
            return

        y0_scaled, y1_scaled = _scaled_bounds(y0, y1, tile_pred.shape[-2], crop_height)
        x0_scaled, x1_scaled = _scaled_bounds(x0, x1, tile_pred.shape[-1], crop_width)
        stitched_output[..., y0_scaled:y1_scaled, x0_scaled:x1_scaled] = tile_pred

    stitched_output = None
    for y0 in range(0, height, crop_height):
        y1 = y0 + crop_height
        for x0 in range(0, width, crop_width):
            x1 = x0 + crop_width
            image_tile = image[:, :, :, y0:y1, x0:x1]

            def forward_ink(tile):
                model_output = model(tile)
                return model_output['ink']
            if use_gradient_checkpointing:
                tile_pred = checkpoint(forward_ink, image_tile, use_reentrant=False)
            else:
                tile_pred = forward_ink(image_tile)

            if stitched_output is None:
                stitched_output = _allocate_stitched_output(tile_pred)
            _write_stitched_output(stitched_output, tile_pred, y0=y0, y1=y1, x0=x0, x1=x1)

    if isinstance(tile_pred, (list, tuple)):
        return type(tile_pred)(stitched_output)
    return stitched_output
