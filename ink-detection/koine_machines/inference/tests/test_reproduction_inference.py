import pytest
import numpy as np
import torch
import zarr

from koine_machines.inference.infer import (
    TargetHeadWrapper,
    OmeZarrPatchReader,
    center_crop_layer_indices,
    compute_importance_map_2d,
    inference_preprocessing_from_config,
    resolve_patch_stride,
    choose_pyramid_array,
    supported_repo_model_type,
)
from koine_machines.models.input_padding import center_pad_input_depth


def test_reproduction_model_types_are_auto_detected():
    assert supported_repo_model_type("vesuvius_unet")
    assert supported_repo_model_type("vesuvius_unet_2p5d")
    assert supported_repo_model_type("vesuvius_unet_3d_stem_2d")
    assert supported_repo_model_type("unet_3d_stem_2d")


def test_hires9um_preprocessing_uses_clip200_div255_path():
    config = {
        "image_normalization": {
            "mode": "clip_divide",
            "clip_min": 0,
            "clip_max": 200,
            "divisor": 255,
        }
    }
    assert inference_preprocessing_from_config(config) == "legacy_uint8"


def test_divide_255_preprocessing_has_no_input_clip():
    config = {"image_normalization": {"mode": "divide", "divisor": 255}}
    assert inference_preprocessing_from_config(config) == "divide_255"

    reader = OmeZarrPatchReader(
        input_path="unused.zarr",
        resolution="0",
        depth_axis_first=True,
        height=1,
        width=3,
        layer_indices=np.arange(1),
        output_depth=1,
        preprocessing="divide_255",
    )
    source = np.array([[[199], [201], [255]]], dtype=np.uint8)
    reader._read_raw = lambda *args: source

    output = reader.read(0, 0, 1, 3)

    np.testing.assert_array_equal(output, source)


def test_explicit_inference_stride_takes_precedence_over_overlap():
    assert resolve_patch_stride(
        patch_size=128,
        overlap=0.25,
        explicit_stride=64,
    ) == 64
    assert resolve_patch_stride(
        patch_size=128,
        overlap=0.50,
        explicit_stride=None,
    ) == 64


def test_inference_stride_rejects_gaps_and_invalid_overlap():
    with pytest.raises(ValueError, match="must not exceed"):
        resolve_patch_stride(patch_size=128, overlap=0.0, explicit_stride=129)
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        resolve_patch_stride(patch_size=128, overlap=1.0, explicit_stride=None)


def test_hann_importance_map_is_positive_and_center_weighted():
    weights = compute_importance_map_2d(
        patch_size=(128, 128),
        mode="hann",
        sigma_scale=0.125,
    )
    assert tuple(weights.shape) == (128, 128)
    assert torch.all(weights > 0)
    assert weights[64, 64] > weights[0, 0]
    assert weights.max() == pytest.approx(1.0)


def test_inference_layer_crop_matches_training_surface_center_for_even_depth():
    indices = center_crop_layer_indices(
        np.arange(28, dtype=np.int64),
        output_depth=21,
    )
    np.testing.assert_array_equal(indices, np.arange(4, 25, dtype=np.int64))
    assert indices[21 // 2] == 28 // 2


def test_inference_layer_crop_preserves_shorter_input_for_center_padding():
    indices = center_crop_layer_indices(
        np.arange(28, dtype=np.int64),
        output_depth=31,
    )
    np.testing.assert_array_equal(indices, np.arange(28, dtype=np.int64))


def test_inference_rejects_unimplemented_clip_divide_variant():
    with pytest.raises(ValueError, match="legacy_uint8"):
        inference_preprocessing_from_config(
            {
                "image_normalization": {
                    "mode": "clip_divide",
                    "clip_min": 0,
                    "clip_max": 255,
                    "divisor": 255,
                }
            }
        )


def test_center_pad_input_depth_preserves_centered_21_layers():
    image = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
    padded = center_pad_input_depth(image, 24)

    assert tuple(padded.shape) == (1, 1, 24, 1, 1)
    torch.testing.assert_close(padded[:, :, 1:22], image)
    assert padded[:, :, 0].count_nonzero() == 0
    assert padded[:, :, 22:].count_nonzero() == 0


def test_target_wrapper_pads_before_model_forward():
    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shape = None

        def forward(self, image):
            self.shape = tuple(image.shape)
            return {"ink": image.amax(dim=2)}

    base_model = CaptureModel()
    wrapper = TargetHeadWrapper(
        base_model,
        target_name="ink",
        input_pad_depth_to=24,
    )
    output = wrapper(torch.ones(2, 1, 21, 8, 8))

    assert base_model.shape == (2, 1, 24, 8, 8)
    assert tuple(output.shape) == (2, 1, 8, 8)


def test_zarr_reader_center_pads_when_checkpoint_expects_more_layers(monkeypatch):
    reader = OmeZarrPatchReader(
        input_path="unused.zarr",
        resolution="0",
        depth_axis_first=True,
        height=2,
        width=3,
        layer_indices=np.arange(3),
        output_depth=5,
    )
    block = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    monkeypatch.setattr(reader, "_read_raw", lambda *args: block)

    output = reader.read(0, 0, 2, 3)

    assert output.shape == (2, 3, 5)
    np.testing.assert_array_equal(output[..., 1:4], block)
    assert np.count_nonzero(output[..., 0]) == 0
    assert np.count_nonzero(output[..., 4]) == 0


def test_pyramid_probe_handles_store_without_directory_listing(tmp_path, monkeypatch):
    group = zarr.open_group(tmp_path / "example.zarr", mode="w")
    expected = group.create_dataset("3", shape=(2, 4, 5), dtype="u1")
    monkeypatch.setattr(group, "array_keys", lambda: iter(()))

    level, array = choose_pyramid_array(
        group,
        preferred_key="3",
        purpose="occupancy scan",
    )

    assert level == "3"
    assert array.shape == expected.shape
