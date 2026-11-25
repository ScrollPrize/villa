"""Shared ink-test inputs; importing these does not construct or import models."""

def config_mapping(
    model_type: str = "vesuvius_unet",
    *,
    depth: int = 4,
    side: int = 16,
    stem_channels: int = 2,
) -> dict:
    is_base = model_type in {"vesuvius_unet", "unet"}
    dimensions = 3 if is_base else 2
    kernel = [3] * dimensions
    first_stride = [1] * dimensions
    second_stride = [2] * dimensions
    model_config = {
        "autoconfigure": True,
        "basic_encoder_block": "BasicBlockD",
        "basic_decoder_block": "ConvBlock",
        "features_per_stage": [2, 4],
        "n_blocks_per_stage": [1, 1],
        "n_conv_per_stage_decoder": [1],
        "kernel_sizes": [kernel, kernel],
        "strides": [first_stride, second_stride],
        "pool_op_kernel_sizes": [first_stride, second_stride],
        "spacing": [1, 1, 1],
    }
    if "3d_stem_2d" in model_type:
        model_config["stem_channels"] = stem_channels
    return {
        "model_type": model_type,
        "mode": "flat",
        "patch_size": [depth, side, side],
        "patch_overlap": 0.25,
        "patch_min_labeled_coverage": 0.0,
        "batch_size": 1,
        "in_channels": depth if "2p5d" in model_type else 1,
        "model_config": model_config,
        "targets": {
            "ink": {
                "out_channels": 1,
                "activation": "none",
                "z_projection_mode": "none",
            }
        },
        "datasets": [{"segments_path": "/tmp/ink", "volume_scale": 0}],
    }
