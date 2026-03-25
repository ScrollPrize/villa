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
            assert tile_pred.shape[-2:] == (crop_height, crop_width), (
                tile_pred.shape,
                (crop_height, crop_width),
            )

            if stitched_output is None:
                stitched_output = tile_pred.new_empty(
                    tile_pred.shape[0],
                    tile_pred.shape[1],
                    height,
                    width,
                )
            stitched_output[:, :, y0:y1, x0:x1] = tile_pred

    return stitched_output
