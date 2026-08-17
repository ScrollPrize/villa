"""Flat axes, blending, TTA, TIFF encoding, and CLI lifecycle tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from torch import nn
import zarr

from vesuvius.ink_detection.config import InkConfig, NormalizationConfig
from vesuvius.ink_detection.inference.infer import (
    Block,
    ChunkAccumulator,
    FlatBlockDataset,
    FlatPatchReader,
    choose_pyramid_array,
    compute_chunk_contribution_counts,
    compute_equal_length_mirror_axes,
    compute_importance_map_2d,
    flat_preprocessing_from_config,
    infer_folder,
    iter_blocks,
    iter_probability_tiles,
    load_grayscale_mask,
    main,
    normalize_flat_patch,
    normalize_inference_paths,
    open_temp_zarr_array,
    parse_args,
    predict_with_mirror_tta,
    resolve_patch_stride,
    resolve_segment_zarr_path,
    select_layer_indices,
    write_output_tiff,
)
from vesuvius.ink_detection.models.model import make_model

from .test_model_foundation import _config_mapping


def test_temp_accumulation_array_is_explicit_v2(tmp_path):
    path = tmp_path / "accumulation.zarr"
    array = open_temp_zarr_array(path, shape=(3, 5), chunks=(2, 3))
    array[:] = np.arange(15, dtype=np.float32).reshape(3, 5)

    metadata = json.loads((path / ".zarray").read_text(encoding="utf-8"))
    assert metadata["zarr_format"] == 2
    assert tuple(array.shape) == (3, 5)
    assert tuple(array.chunks) == (2, 3)
    assert np.dtype(array.dtype) == np.dtype(np.float32)


def test_pyramid_substitution_and_mask_alignment_warn_factually(tmp_path, caplog):
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        group = zarr.open_group(
            tmp_path / "pyramid.zarr", mode="w", zarr_format=2
        )
        group.create_array("1", shape=(2, 2), dtype="u1")
    else:
        group = zarr.open_group(tmp_path / "pyramid.zarr", mode="w")
        group.create_dataset("1", shape=(2, 2), dtype="u1")
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.ones((2, 3), dtype=np.uint8))

    with caplog.at_level("WARNING"):
        selected, _ = choose_pyramid_array(
            group, preferred_key="0", purpose="input"
        )
        mask = load_grayscale_mask(mask_path, (3, 2))

    assert selected == "1"
    assert mask.shape == (3, 2)
    assert "Requested input level 0 was not found; using level 1" in caplog.text
    assert "Mask shape (2, 3) did not match Zarr shape (3, 2)" in caplog.text


def test_flat_preprocessing_accepts_exact_training_contracts():
    assert flat_preprocessing_from_config(
        NormalizationConfig.from_value("robust_mad")
    ) == "tifxyz_robust"
    assert flat_preprocessing_from_config(
        NormalizationConfig.from_value({"mode": "divide", "divisor": 255})
    ) == "divide_255"


@pytest.mark.parametrize(
    "value",
    [
        "robust_percentile_span",
        "minmax",
        "percentile_minmax",
        {"mode": "clip_zscore", "clip_min": 0, "clip_max": 1, "mean": 0, "std": 1},
        "none",
    ],
)
def test_flat_preprocessing_rejects_unsupported_training_modes(value):
    config = NormalizationConfig.from_value(value)
    with pytest.raises(ValueError, match=rf"mode '{config.mode}'"):
        flat_preprocessing_from_config(config)


def test_flat_preprocessing_rejects_nondefault_robust_percentiles():
    config = NormalizationConfig.from_value(
        {"mode": "robust_mad", "percentile_lower": 2, "percentile_upper": 98}
    )
    with pytest.raises(ValueError, match="percentile_lower=1"):
        flat_preprocessing_from_config(config)


def test_flat_divide_restriction_and_robust_normalization():
    with pytest.raises(ValueError, match="divisor=255"):
        flat_preprocessing_from_config(
            NormalizationConfig.from_value({"mode": "divide", "divisor": 256})
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


def test_flat_robust_normalization_excludes_reader_padding(tmp_path):
    input_path = tmp_path / "short-source.zarr"
    values_ZYX = np.arange(100, 136, dtype=np.uint8).reshape(3, 3, 4)
    source = zarr.open(
        input_path,
        mode="w",
        shape=values_ZYX.shape,
        chunks=values_ZYX.shape,
        dtype="u1",
        zarr_format=2,
    )
    source[:] = values_ZYX
    reader = FlatPatchReader(
        input_path=input_path,
        resolution="0",
        depth_axis_first=True,
        height=3,
        width=4,
        layer_indices=np.arange(3),
        output_depth=5,
        preprocessing="tifxyz_robust",
    )
    dataset = FlatBlockDataset(
        reader=reader,
        blocks=[Block(y0=0, x0=0, valid_h=3, valid_w=4)],
        patch_size=5,
        preprocessing="tifxyz_robust",
    )

    image_CZYX, metadata = dataset[0]
    expected = np.zeros((1, 5, 5, 5), dtype=np.float32)
    expected[0, 1:4, :3, :4] = normalize_flat_patch(
        values_ZYX.copy(), "tifxyz_robust"
    )

    np.testing.assert_array_equal(image_CZYX.numpy(), expected)
    np.testing.assert_array_equal(metadata.numpy(), [0, 0, 3, 4, 1])


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
    overlap = parse_args(["--overlap", "1"])
    with pytest.raises(ValueError, match="--overlap"):
        resolve_patch_stride(
            patch_size=32,
            overlap=overlap.overlap,
            explicit_stride=overlap.stride,
        )
    with pytest.raises(SystemExit):
        parse_args(["--model-type", "auto"])
    with pytest.raises(SystemExit):
        parse_args(["--metadata-json", "metadata.json"])

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


def test_cpu_command_checkpoint_to_tiff_timeline(tmp_path, caplog):
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

    mask_path = tmp_path / "empty-mask.tif"
    tifffile.imwrite(mask_path, np.zeros((16, 16), dtype=np.uint8))
    caplog.clear()
    with caplog.at_level("INFO"):
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
                "--mask-path",
                str(mask_path),
            ]
        ) == 0
    assert not tifffile.imread(output).any()
    assert "Selected source layer indices=" in caplog.text
    assert "foreground coverage 0.000%" in caplog.text
    assert "Selected 0 patches for inference" in caplog.text
    assert "all-zero output" in caplog.text

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


@pytest.mark.parametrize("workers", (0, 1))
def test_flat_inference_ignores_embedded_io_paths_and_keeps_reference_defaults(
    tmp_path, caplog, monkeypatch, workers
):
    """Published checkpoints embed foreign I/O paths; flat inference must not
    read them, and its defaults must match the reference (overlap 0.5,
    auto blend resolving to Hann below full stride). The embedded paths sit
    below a regular file, so consuming any of them fails regardless of
    filesystem privileges — in the main process and in spawned workers."""

    blocker = tmp_path / "embedded-io-blocker"
    blocker.write_bytes(b"")
    config_mapping = _config_mapping("vesuvius_unet_2p5d", depth=3, side=16)
    config_mapping["volume_cache_dir"] = str(blocker / "cache")
    config_mapping["volume_cache_max_gb"] = 120.0
    config_mapping["volume_auth_json"] = str(blocker / "auth.json")
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

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with caplog.at_level("INFO"):
        assert main(
            [
                str(input_path),
                str(checkpoint),
                str(output),
                "--workers",
                str(workers),
                "--no-compile",
            ]
        ) == 0
    assert "stride=8 requested_overlap=0.500 blend_mode=hann" in caplog.text
    assert np.all(tifffile.imread(output) == 127)
    assert not any(path.name.startswith("ink_flat_infer_") for path in tmp_path.iterdir())
