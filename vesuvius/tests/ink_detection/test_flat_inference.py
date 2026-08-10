"""Flat axes, blending, TTA, TIFF encoding, and CLI lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from torch import nn
import zarr

from vesuvius.ink_detection.config import InkConfig, NormalizationConfig
from vesuvius.ink_detection.inference.infer import (
    ChunkAccumulator,
    FlatPatchReader,
    compute_chunk_contribution_counts,
    compute_equal_length_mirror_axes,
    compute_importance_map_2d,
    convert_volume_dtype,
    flat_preprocessing_from_config,
    infer_folder,
    iter_blocks,
    iter_probability_tiles,
    main,
    normalize_flat_patch,
    normalize_inference_paths,
    parse_args,
    predict_with_mirror_tta,
    resolve_segment_zarr_path,
    select_layer_indices,
    write_output_tiff,
)
from vesuvius.ink_detection.models.model import make_model

from .test_model_foundation import _config_mapping


@pytest.mark.parametrize(
    "name",
    [
        "robust_mad",
        "robust_percentile_span",
        "minmax",
        "percentile_minmax",
        "clip_zscore",
        "none",
    ],
)
def test_flat_legacy_normalization_routes_nondivide_modes_to_robust(name):
    value: object = name
    if name == "clip_zscore":
        value = {
            "mode": name,
            "clip_min": 0,
            "clip_max": 1,
            "mean": 0,
            "std": 1,
        }
    assert (
        flat_preprocessing_from_config(NormalizationConfig.from_value(value))
        == "tifxyz_robust"
    )


def test_flat_legacy_divide_restrictions_and_uint16_quantization():
    assert flat_preprocessing_from_config(
        NormalizationConfig.from_value({"mode": "divide", "divisor": 255})
    ) == "divide_255"
    assert flat_preprocessing_from_config(
        NormalizationConfig.from_value(
            {"mode": "clip_divide", "clip_min": 0, "clip_max": 200, "divisor": 255}
        )
    ) == "legacy_uint8"
    with pytest.raises(ValueError, match="divisor=255"):
        flat_preprocessing_from_config(
            NormalizationConfig.from_value({"mode": "divide", "divisor": 256})
        )
    with pytest.raises(ValueError, match="clip_min=0"):
        flat_preprocessing_from_config(
            NormalizationConfig.from_value(
                {"mode": "clip_divide", "clip_min": 1, "clip_max": 200, "divisor": 255}
            )
        )
    np.testing.assert_array_equal(
        convert_volume_dtype(np.array([0, 255, 256, 511, 65535], dtype=np.uint16)),
        np.array([0, 0, 1, 1, 255], dtype=np.uint8),
    )
    np.testing.assert_allclose(
        normalize_flat_patch(
            np.array([0, 0, 0, 0, 10], dtype=np.float32),
            "tifxyz_robust",
        ),
        np.array([0, 0, 0, 0, 2.4], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        normalize_flat_patch(
            np.array([0, 200], dtype=np.uint8), "legacy_uint8"
        ),
        np.array([0.0, 200.0 / 255.0], dtype=np.float32),
    )


def test_flat_axes_layers_reverse_and_short_depth_padding(tmp_path):
    depth_first_path = tmp_path / "depth-first.zarr"
    depth_last_path = tmp_path / "depth-last.zarr"
    values_ZYX = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    first = zarr.open(
        depth_first_path,
        mode="w",
        shape=values_ZYX.shape,
        chunks=values_ZYX.shape,
        dtype="u1",
        zarr_format=2,
    )
    first[:] = values_ZYX
    last = zarr.open(
        depth_last_path,
        mode="w",
        shape=(5, 7, 3),
        chunks=(5, 7, 3),
        dtype="u1",
        zarr_format=2,
    )
    last[:] = np.moveaxis(values_ZYX, 0, -1)
    layers = select_layer_indices(
        3,
        layer_start=None,
        layer_end=None,
        output_depth=5,
        direction="forward",
    )
    readers = [
        FlatPatchReader(
            input_path=depth_first_path,
            resolution="0",
            depth_axis_first=True,
            height=5,
            width=7,
            layer_indices=layers,
            output_depth=5,
            preprocessing="divide_255",
        ),
        FlatPatchReader(
            input_path=depth_last_path,
            resolution="0",
            depth_axis_first=False,
            height=5,
            width=7,
            layer_indices=layers,
            output_depth=5,
            preprocessing="divide_255",
        ),
    ]
    patches = [reader.read(0, 0, 5, 7) for reader in readers]
    np.testing.assert_array_equal(patches[0], patches[1])
    assert np.count_nonzero(patches[0][..., 0]) == 0
    assert np.count_nonzero(patches[0][..., 4]) == 0
    np.testing.assert_array_equal(patches[0][..., 1:4], np.moveaxis(values_ZYX, 0, -1))

    assert select_layer_indices(
        28,
        layer_start=None,
        layer_end=None,
        output_depth=21,
        direction="forward",
    ).tolist() == list(range(4, 25))
    assert select_layer_indices(
        6,
        layer_start=5,
        layer_end=2,
        output_depth=6,
        direction="reverse",
    ).tolist() == [5, 4, 3, 2, 1, 0]


def test_blocks_occupancy_blend_accumulation_and_truncating_tiles():
    occupancy = np.array([[False, True], [False, False]])
    blocks = iter_blocks((5, 7), 4, 3, occupancy, (3, 4))
    assert [(block.y0, block.x0) for block in blocks] == [(0, 3), (1, 3)]
    assert iter_blocks((2, 3), 4, 3)[0].valid_h == 2

    hann = compute_importance_map_2d(patch_size=(3, 3), mode="hann").numpy()
    np.testing.assert_array_equal(
        hann,
        np.array(
            [[0.001, 0.001, 0.001], [0.001, 1.0, 0.001], [0.001, 0.001, 0.001]],
            dtype=np.float32,
        ),
    )

    scheduled = iter_blocks((2, 3), 2, 1)
    probability = np.zeros((2, 3), dtype=np.float32)
    weight = np.zeros((2, 3), dtype=np.float32)
    accumulator = ChunkAccumulator(
        shape=(2, 3),
        chunk_shape=(2, 3),
        prob_sum_store=probability,
        weight_sum_store=weight,
        contribution_counts=compute_chunk_contribution_counts(
            scheduled, chunk_shape=(2, 3)
        ),
    )
    accumulator.add_tile(
        y0=0,
        x0=0,
        tile=np.full((2, 2), 0.25, dtype=np.float32),
        tile_weights=np.ones((2, 2), dtype=np.float32),
    )
    accumulator.add_tile(
        y0=0,
        x0=1,
        tile=np.full((2, 2), 0.75, dtype=np.float32),
        tile_weights=np.ones((2, 2), dtype=np.float32),
    )
    encoded = next(iter_probability_tiles(probability, weight, (2, 3)))
    np.testing.assert_array_equal(
        encoded,
        np.array([[63, 127, 191], [63, 127, 191]], dtype=np.uint8),
    )


class _OrientationModel(nn.Module):
    def forward(self, image_BCZYX):
        depth_weights = torch.arange(
            1,
            image_BCZYX.shape[2] + 1,
            device=image_BCZYX.device,
            dtype=image_BCZYX.dtype,
        ).view(1, 1, -1, 1, 1)
        return (image_BCZYX * depth_weights).sum(dim=2)


def test_flat_tta_variant_counts_restoration_and_batching():
    assert compute_equal_length_mirror_axes((17, 128, 128)) == (1, 2)
    assert compute_equal_length_mirror_axes((4, 4, 4)) == (0, 1, 2)
    image = torch.arange(1 * 1 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 1, 3, 4, 4) / 30
    model = _OrientationModel()
    one_at_a_time = predict_with_mirror_tta(
        model,
        image,
        tta_axes=(0, 1, 2),
        tta_batch_size=1,
    )
    all_at_once = predict_with_mirror_tta(
        model,
        image,
        tta_axes=(0, 1, 2),
        tta_batch_size=None,
    )
    torch.testing.assert_close(one_at_a_time, all_at_once)
    assert tuple(all_at_once.shape) == (1, 1, 4, 4)


def test_tiff_export_is_tiled_lzw_truncating_and_replaceable(tmp_path):
    probability = np.zeros((16, 16), dtype=np.float32)
    weight = np.ones((16, 16), dtype=np.float32)
    probability[0, :4] = [-1, 0.5, 1, 2]
    weight[1, 0] = 0
    output = tmp_path / "prediction.tif"

    write_output_tiff(probability, weight, output, (16, 16))
    first = tifffile.imread(output)
    np.testing.assert_array_equal(first[0, :4], [0, 127, 255, 255])
    assert first[1, 0] == 0
    with tifffile.TiffFile(output) as tiff:
        assert tiff.pages[0].is_tiled
        assert tiff.pages[0].compression.name == "LZW"

    probability.fill(1)
    write_output_tiff(probability, weight, output, (16, 16))
    assert np.all(tifffile.imread(output)[0] == 255)


def test_cli_aliases_folder_shorthand_and_segment_resolution(tmp_path):
    args = parse_args(
        [
            "--num_workers",
            "0",
            "--prefetch_factor",
            "3",
            "--tta_mirror",
            "--no_compile",
        ]
    )
    assert (args.num_workers, args.prefetch_factor, args.tta_mirror) == (0, 3, True)
    assert args.compile_model is False
    shorthand = normalize_inference_paths(
        parse_args(["--folder", str(tmp_path), "weights.pth"])
    )
    assert shorthand.input_zarr is None
    assert shorthand.checkpoint == Path("weights.pth")
    with pytest.raises(SystemExit):
        parse_args(["--overlap", "1"])

    segment = tmp_path / "segment"
    segment.mkdir()
    direct = segment / "segment.ome.zarr"
    direct.mkdir()
    assert resolve_segment_zarr_path(segment) == direct


def test_folder_mode_logs_existing_prediction_and_summary(
    tmp_path, monkeypatch, caplog
):
    segment = tmp_path / "segment"
    input_zarr = segment / "surface.zarr"
    input_zarr.mkdir(parents=True)
    (input_zarr / ".zarray").touch()
    prediction_dir = segment / "preds"
    prediction_dir.mkdir()
    existing = prediction_dir / "segment_model_forward_010101.tif"
    existing.touch()
    monkeypatch.setattr(
        "vesuvius.ink_detection.inference.infer.infer_single_zarr",
        lambda **kwargs: pytest.fail("existing prediction was rerun"),
    )
    args = SimpleNamespace(
        folder=tmp_path,
        checkpoint=Path("model.pth"),
        output_prefix="",
        direction="forward",
    )

    with caplog.at_level("INFO"):
        infer_folder(args, None, device=torch.device("cpu"))

    assert "already have segment_model_forward_010101.tif" in caplog.text
    assert "segments_ran=0 segments_skipped=1" in caplog.text


def test_cpu_command_checkpoint_to_tiff_timeline(tmp_path):
    config_mapping = _config_mapping(
        "vesuvius_unet_2p5d", depth=3, side=16
    )
    config_mapping["image_normalization"] = "robust_mad"
    config = InkConfig.from_mapping(config_mapping)
    model = make_model(config)
    with torch.no_grad():
        for value in model.state_dict().values():
            if torch.is_floating_point(value):
                value.zero_()
    checkpoint = tmp_path / "model.pth"
    torch.save({"config": config_mapping, "model": model.state_dict()}, checkpoint)
    input_path = tmp_path / "surface.zarr"
    array = zarr.open(
        input_path,
        mode="w",
        shape=(3, 16, 16),
        chunks=(3, 16, 16),
        dtype="u1",
        zarr_format=2,
    )
    array[:] = 1
    output = tmp_path / "prediction.tif"

    assert main(
        [
            str(input_path),
            str(checkpoint),
            str(output),
            "--workers",
            "0",
            "--no-compile",
            "--blend-mode",
            "constant",
        ]
    ) == 0
    assert np.all(tifffile.imread(output) == 127)

    array[:] = 2
    assert main(
        [
            str(input_path),
            str(checkpoint),
            str(output),
            "--workers",
            "0",
            "--no-compile",
            "--blend-mode",
            "constant",
        ]
    ) == 0
    assert np.all(tifffile.imread(output) == 127)
    assert not any(path.name.startswith("ink_flat_infer_") for path in tmp_path.iterdir())
