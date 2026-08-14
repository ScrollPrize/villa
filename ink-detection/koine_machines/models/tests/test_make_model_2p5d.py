import pytest
import torch

from koine_machines.models.hybrid_3d2d import Local3DStem2DUNet
from koine_machines.models.make_model import SliceChannel2DModel, make_model


def _config():
    return {
        "model_type": "vesuvius_unet_2p5d",
        "crop_size": [17, 32, 32],
        "patch_size": [17, 32, 32],
        "batch_size": 2,
        "in_channels": 17,
        "model_config": {
            "autoconfigure": True,
            "basic_encoder_block": "BasicBlockD",
            "basic_decoder_block": "ConvBlock",
            "features_per_stage": [4, 8],
            "n_blocks_per_stage": [1, 1],
            "n_conv_per_stage_decoder": [1],
            "kernel_sizes": [[3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2]],
            "pool_op_kernel_sizes": [[1, 1], [2, 2]],
        },
        "targets": {
            "ink": {
                "out_channels": 1,
                "activation": "none",
                "z_projection_mode": "none",
            }
        },
    }


def test_2p5d_treats_depth_as_2d_input_channels():
    model = make_model(_config())

    assert isinstance(model, SliceChannel2DModel)
    assert model.network.op_dims == 2
    assert model.network.in_channels == 17
    assert isinstance(model.network.shared_encoder.stem.convs[0].conv, torch.nn.Conv2d)

    output = model(torch.randn(2, 1, 17, 32, 32))

    assert tuple(output["ink"].shape) == (2, 1, 32, 32)


def test_2p5d_rejects_wrong_depth():
    model = make_model(_config())

    with pytest.raises(ValueError, match="input depth mismatch"):
        model(torch.randn(1, 1, 16, 32, 32))


def test_3d_stem_2d_unet_preserves_z_locality_before_projection():
    config = _config()
    config["model_type"] = "vesuvius_unet_3d_stem_2d"
    config["in_channels"] = 1
    config["model_config"]["stem_channels"] = 4

    model = make_model(config)

    assert isinstance(model, Local3DStem2DUNet)
    assert isinstance(model.depth_fusion.features[0], torch.nn.Conv3d)
    assert isinstance(model.depth_fusion.features[1], torch.nn.InstanceNorm3d)
    assert model.network.op_dims == 2
    assert model.network.in_channels == 8

    output = model(torch.randn(2, 1, 17, 32, 32))

    assert tuple(output["ink"].shape) == (2, 1, 32, 32)
    assert torch.isfinite(output["ink"]).all()


def test_3d_stem_2d_unet_rejects_wrong_depth():
    config = _config()
    config["model_type"] = "vesuvius_unet_3d_stem_2d"
    config["in_channels"] = 1
    model = make_model(config)

    with pytest.raises(ValueError, match="input depth mismatch"):
        model(torch.randn(1, 1, 16, 32, 32))
