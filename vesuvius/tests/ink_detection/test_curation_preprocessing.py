"""CPU witnesses for the five ink-detection curation command leaves."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.preprocessing import clean_labels
from vesuvius.ink_detection.preprocessing import composite_from_zarr
from vesuvius.ink_detection.preprocessing import download_required_zarr_chunks
from vesuvius.ink_detection.preprocessing import merge_predictions
from vesuvius.ink_detection.preprocessing import validate_segments
from vesuvius.ink_detection.types import Segment
from vesuvius.tifxyz import Tifxyz, write_tifxyz


def _write_group(path: Path, arrays: dict[str, np.ndarray], *, chunks: tuple[int, ...]) -> None:
    kwargs: dict[str, object] = {"mode": "w"}
    zarr_v3 = int(zarr.__version__.split(".", 1)[0]) >= 3
    if zarr_v3:
        kwargs["zarr_format"] = 2
    group = zarr.open_group(path, **kwargs)
    for key, array in arrays.items():
        if zarr_v3:
            group.create_array(key, data=array, chunks=chunks)
        else:
            group.create_dataset(key, data=array, chunks=chunks)


def _set_tiny_pillow_limit() -> None:
    Image.MAX_IMAGE_PIXELS = 1


def test_clean_labels_preserves_categorical_foreground() -> None:
    source = np.array([[0, 1, 255], [0, 0, 0]], dtype=np.uint8)
    assert clean_labels.normalize_mask_image(source).tolist() == [[0, 255, 255], [0, 0, 0]]


def test_composite_projection_encodes_reference_uint8_range() -> None:
    source = np.array([[[0, 100]], [[255, 200]]], dtype=np.uint8)
    assert composite_from_zarr._project_block(source, "max").tolist() == [[255, 200]]
    assert composite_from_zarr._project_block(source, "min").tolist() == [[0, 100]]
    assert composite_from_zarr._project_block(source, "mean").tolist() == [[127.5, 150.0]]
    assert composite_from_zarr._to_uint8(np.array([[0.0, 255.0]])).tolist() == [[0, 255]]


def test_composite_center_layers_selects_middle_of_level() -> None:
    resolve = composite_from_zarr._resolve_z_range
    assert resolve(24, None, None, 8) == (8, 16)
    assert resolve(24, None, None, 7) == (8, 15)
    assert resolve(24, None, None, 24) == (0, 24)
    assert resolve(3, None, None, 1) == (1, 2)
    assert resolve(24, None, None) == (0, 24)
    assert resolve(24, -6, None) == (18, 24)
    with pytest.raises(ValueError):
        resolve(3, None, None, 4)
    args = composite_from_zarr.parse_args(["--zarr", ".", "--method", "max", "--center-layers", "8"])
    assert args.center_layers == 8 and args.start_z is None
    for bad in (["--center-layers", "0"], ["--center-layers", "4", "--start_z", "1"]):
        with pytest.raises(SystemExit):
            composite_from_zarr.parse_args(["--zarr", ".", "--method", "max", *bad])


def test_composite_tile_shape_is_chunk_aligned_and_clamped_to_image() -> None:
    def resolve(shape, chunks, tile_size):
        return composite_from_zarr._resolve_tile_geometry(shape, chunks, tile_size).tiff_tile

    assert resolve((3, 19120, 1138260), (3, 128, 128), 1024) == (1024, 1024)
    assert resolve((3, 19120, 1138260), (3, 128, 128), 1000) == (1024, 1024)
    assert resolve((3, 19120, 1138260), (3, 100, 100), 1024) == (1200, 1200)
    assert resolve((2, 16, 16), (1, 16, 16), 1024) == (16, 16)
    assert resolve((2, 20, 1000), (1, 16, 16), 1024) == (32, 1008)


def test_composite_scaled_geometry_keeps_tiles_integral() -> None:
    from fractions import Fraction

    geometry = composite_from_zarr._resolve_tile_geometry
    g = geometry((24, 19120, 1138260), (24, 128, 128), 1024, Fraction(1, 4))
    assert g.out_shape == (4780, 284565) and g.in_tile == (1024, 1024)
    assert g.out_tile == (256, 256) and g.tiff_tile == (256, 256)
    g = geometry((24, 19121, 100), (24, 128, 128), 1024, Fraction(2))
    assert g.out_shape == (38242, 200) and g.in_tile == (1024, 100)
    assert g.out_tile == (2048, 200) and g.tiff_tile == (2048, 208)
    g = geometry((24, 19120, 1138260), (24, 128, 128), 1024, Fraction(3, 10))
    assert g.in_tile == (1280, 1280) and g.out_tile == (384, 384) and g.out_shape == (5736, 341478)
    tasks = composite_from_zarr._iter_chunk_tasks(geometry((3, 70, 90), (3, 16, 16), 16, Fraction(1, 4)))
    assert tasks[0].out_y_stop == 16 and tasks[0].y_stop == 64
    assert tasks[-1].out_y_stop == 18 and tasks[-1].y_start == 64 and tasks[-1].y_stop == 72
    for text, expected in (("2", 2), ("2.0", 2), ("0.25", Fraction(1, 4)), ("1/3", Fraction(1, 3))):
        assert composite_from_zarr._parse_scale(text) == expected
    for bad in ("0", "-1", "abc", "1/0"):
        with pytest.raises(Exception):
            composite_from_zarr._parse_scale(bad)


def test_prediction_merge_and_chunk_coverage_are_deterministic() -> None:
    inputs = [np.array([[0, 255]], dtype=np.uint8), np.array([[255, 0]], dtype=np.uint8)]
    assert merge_predictions.merge_soft_mean_chunk(inputs).tolist() == [[128, 128]]
    assert download_required_zarr_chunks.collect_unique_chunk_ids(
        [{"world_bbox": (0, 1, 1, 4, 5, 5)}],
        chunk_shape_zyx=(2, 2, 2),
        array_shape_zyx=(4, 6, 6),
    ) == ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2))


def test_validate_segments_accepts_reference_binary_values() -> None:
    assert validate_segments._normalize_version_id("v2") == 2
    assert validate_segments.ALLOWED_BINARY_LABEL_VALUES == frozenset({0, 1, 255})


def test_all_curation_parsers_use_required_aliases() -> None:
    clean = clean_labels.parse_args([".", "--target_folder", "."])
    validate = validate_segments.parse_args([".", "--no_progress"])
    composite = composite_from_zarr.parse_args(
        ["--input_root", ".", "--start-z", "1", "--method", "max"]
    )
    download = download_required_zarr_chunks.parse_args(
        ["--datasets_root", ".", "--volumes_json", "volumes.json", "--output_root", "."]
    )
    assert clean.target_folder == Path(".")
    assert validate.no_progress
    assert composite.input_root == "." and composite.start_z == 1
    assert download.datasets_root == "."


def test_clean_labels_exports_backup_is_rerunnable_and_preserves_source_on_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "labels"
    root.mkdir()
    source = root / "segment_inklabels.tif"
    original = np.zeros((32, 32), dtype=np.uint8)
    original[4:12, 4:12] = 1
    tifffile.imwrite(source, original)

    with patch.object(clean_labels, "write_tiff", side_effect=OSError("injected write failure")):
        with pytest.raises(OSError, match="injected write failure"):
            clean_labels.process_one(
                root,
                source,
                min_component_size=1,
                do_fill_holes=False,
                max_hole_area=0,
                overwrite=False,
            )
    backup = clean_labels.backup_path_for_input(root, source)
    assert source.is_file()
    assert not backup.exists()

    assert clean_labels.main([str(root), "--workers", "1"]) == 0
    np.testing.assert_array_equal(tifffile.imread(source), original * 255)
    np.testing.assert_array_equal(tifffile.imread(backup), original)
    assert clean_labels.main([str(root), "--workers", "1"]) == 0
    np.testing.assert_array_equal(tifffile.imread(source), original * 255)


def test_clean_labels_publish_failure_keeps_original_in_backup(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    root.mkdir()
    source = root / "segment_inklabels.tif"
    original = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    tifffile.imwrite(source, original)
    backup = clean_labels.backup_path_for_input(root, source)

    with patch.object(
        clean_labels,
        "publish_staged_output",
        side_effect=OSError("injected publish failure"),
    ):
        with pytest.raises(OSError, match="injected publish failure"):
            clean_labels.process_one(
                root,
                source,
                min_component_size=1,
                do_fill_holes=False,
                max_hole_area=0,
                overwrite=False,
            )

    assert not source.exists()
    np.testing.assert_array_equal(tifffile.imread(backup), original)


def test_spawned_clean_worker_applies_the_same_pillow_policy(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    root.mkdir()
    source = root / "segment_supervision_mask.png"
    Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(source)

    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=_set_tiny_pillow_limit,
    ) as executor:
        result = executor.submit(
            clean_labels._process_worker,
            str(root),
            str(source),
            1,
            False,
            0,
            False,
        ).result()
    assert result["status"] == "written"
    assert source.with_suffix(".tif").is_file()


def test_validate_segments_reports_content_despite_unrelated_metadata_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "labels"
    good = root / "scroll" / "good"
    bad = root / "scroll" / "bad"
    good.mkdir(parents=True)
    bad.mkdir()
    _write_group(
        good / "good.zarr",
        {"0": np.zeros((1, 16, 16), dtype=np.uint8)},
        chunks=(1, 16, 16),
    )
    invalid = np.zeros((16, 16), dtype=np.uint8)
    invalid[0, 0] = 2
    tifffile.imwrite(good / "good_inklabels.tif", invalid)
    tifffile.imwrite(good / "good_supervision_mask.tif", np.zeros_like(invalid))

    results = validate_segments.validate_root(root, workers=1, show_progress=False)
    issues = {
        result.segment_dir.name: [issue.message for issue in result.issues]
        for result in results
    }
    assert any("missing" in message.lower() for message in issues["bad"])
    assert any("non-binary values: 2" in message for message in issues["good"])
    repeated = validate_segments.validate_root(root, workers=1, show_progress=False)
    assert [result.issues for result in repeated] == [result.issues for result in results]
    assert validate_segments.main([str(root), "--workers", "1", "--no-progress"]) == 1


def test_merge_predictions_exports_expected_values_reruns_and_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    preds = tmp_path / "preds"
    preds.mkdir()
    tifffile.imwrite(
        preds / "model_a_forward_prediction.tif",
        np.array([[0, 255], [10, 20]], dtype=np.uint8),
    )
    tifffile.imwrite(
        preds / "model_b_forward_prediction.tif",
        np.array([[255, 0], [30, 40]], dtype=np.uint8),
    )
    args = [str(preds), "--workers", "1", "--direction", "forward", "--method", "soft_mean"]
    assert merge_predictions.main(args) == 0
    output = preds / "merged_soft_mean_forward.tif"
    expected = np.array([[128, 128], [20, 30]], dtype=np.uint8)
    np.testing.assert_array_equal(tifffile.imread(output), expected)
    # A rerun must ignore its own previous output: corrupt the merged file and
    # verify the merge is recomputed from the two model predictions alone.
    tifffile.imwrite(output, np.full((2, 2), 255, dtype=np.uint8))
    assert merge_predictions.main(args) == 0
    np.testing.assert_array_equal(tifffile.imread(output), expected)

    empty = tmp_path / "empty" / "preds"
    empty.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="No matching prediction files"):
        merge_predictions.main([str(empty), "--workers", "1", "--direction", "forward"])
    with pytest.raises(ValueError, match="non-finite"):
        merge_predictions.normalize_prediction_array(
            np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32),
            path=Path("bad.tif"),
        )
    nonfinite = tmp_path / "nonfinite" / "preds"
    nonfinite.mkdir(parents=True)
    tifffile.imwrite(
        nonfinite / "model_a_forward_prediction.tif",
        np.array([[np.nan, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="non-finite"):
        merge_predictions.main(
            [str(nonfinite), "--workers", "1", "--direction", "forward"]
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("max", np.array([[255, 201], [9, 10]], dtype=np.uint8)),
        ("min", np.array([[0, 100], [5, 6]], dtype=np.uint8)),
        ("mean", np.array([[128, 150], [7, 8]], dtype=np.uint8)),
    ],
)
def test_composite_cli_exports_reference_values_and_has_safe_error_paths(
    tmp_path: Path,
    method: str,
    expected: np.ndarray,
) -> None:
    folder = tmp_path / method
    folder.mkdir()
    first = np.zeros((16, 16), dtype=np.uint8)
    second = np.zeros((16, 16), dtype=np.uint8)
    first[:2, :2] = np.array([[0, 100], [5, 6]], dtype=np.uint8)
    second[:2, :2] = np.array([[255, 201], [9, 10]], dtype=np.uint8)
    _write_group(folder / "surface.zarr", {"0": np.stack([first, second])}, chunks=(1, 16, 16))
    args = ["--input-root", str(folder), "--method", method, "--workers", "1", "--no-progress"]
    assert composite_from_zarr.main(args) == 0
    output = folder / f"{folder.name}_{method}_0_2.tif"
    np.testing.assert_array_equal(tifffile.imread(output)[:2, :2], expected)
    with pytest.raises(FileExistsError):
        composite_from_zarr.main(args)
    assert composite_from_zarr.main([*args, "--overwrite"]) == 0
    with pytest.raises(NotADirectoryError):
        composite_from_zarr.main(
            ["--input-root", str(tmp_path / "missing"), "--method", "max", "--no-progress"]
        )


@pytest.mark.parametrize("parallelism", ["threads", "processes"])
def test_composite_single_zarr_median_matches_whole_volume_filter(
    tmp_path: Path,
    parallelism: str,
) -> None:
    from scipy import ndimage

    rng = np.random.default_rng(0)
    volume = rng.integers(0, 256, size=(6, 70, 90), dtype=np.uint8)
    zarr_path = tmp_path / "volume.zarr"
    _write_group(zarr_path, {"0": volume}, chunks=(6, 16, 16))
    output = tmp_path / "out" / "composite.tif"
    output.parent.mkdir()
    # 16-pixel tiles with a radius-2 halo: every interior tile border crosses a
    # neighbour, and layers 1..4 need the Z halo from layers 0 and 5, so this checks
    # the halo read and crop in all three axes, not just the filter.
    args = [
        "--zarr", str(zarr_path), "--output", str(output), "--method", "max",
        "--median-radius", "2", "--tile-size", "16", "--workers", "2",
        "--start_z", "1", "--end_z", "5", "--parallelism", parallelism, "--no-progress",
    ]
    assert composite_from_zarr.main(args) == 0
    filtered = ndimage.median_filter(volume, size=5, mode="nearest")
    expected = filtered[1:5].max(axis=0)
    np.testing.assert_array_equal(tifffile.imread(output), expected)
    with tifffile.TiffFile(output) as tif:
        page = tif.pages[0]
        assert tif.is_bigtiff and (page.tilelength, page.tilewidth) == (16, 16)


def test_composite_volume_filter_chain_halo_and_suffix() -> None:
    filters = composite_from_zarr._build_volume_filters(median_radius=0)
    assert filters == () and composite_from_zarr._filter_halo(filters) == 0
    filters = composite_from_zarr._build_volume_filters(median_radius=3)
    assert [f.suffix for f in filters] == ["median3"] and composite_from_zarr._filter_halo(filters) == 3
    block = np.full((3, 5, 5), 9, dtype=np.uint8)
    np.testing.assert_array_equal(composite_from_zarr._apply_volume_filters(block, filters), block)
    from fractions import Fraction

    image_filters = composite_from_zarr._build_image_filters(median_2d_radius=2)
    assert [f.suffix for f in image_filters] == ["median2d2"]
    assert composite_from_zarr._build_image_filters(median_2d_radius=0) == ()
    # 2 output pixels at quarter scale need 8 input pixels, rounded up to the scale
    # denominator together with the resize halo and the 3D radius.
    halo = composite_from_zarr._resolve_halo(filters, Fraction(1, 4), "auto", image_filters)
    assert halo % 4 == 0 and halo >= 3 + 8 + 10
    assert composite_from_zarr._resolve_halo((), Fraction(1), "auto", image_filters) == 2
    assert composite_from_zarr._output_name(
        "v", method="max", filters=filters, array_key="0", start_z=0, end_z=2,
        scale=Fraction(1, 4), image_filters=image_filters,
    ) == "v_max_median3_scale0.25_area_median2d2_0_2.tif"
    assert composite_from_zarr._output_name(
        "v", method="max", filters=(), array_key="0", start_z=0, end_z=2, scale=Fraction(2),
    ) == "v_max_scale2_linear_0_2.tif"


def _scaled_reference(image: np.ndarray, scale, interpolation: int) -> np.ndarray:
    """Whole-image resize at exactly ``scale`` with replicate borders."""

    import cv2

    den = scale.denominator
    h, w = image.shape
    ph, pw = -(-h // den) * den, -(-w // den) * den
    padded = np.pad(image, ((0, ph - h), (0, pw - w)), mode="edge")
    resized = cv2.resize(padded, (int(pw * scale), int(ph * scale)), interpolation=interpolation)
    return resized[: -(-h * scale.numerator // den), : -(-w * scale.numerator // den)]


@pytest.mark.parametrize(
    ("scale", "interpolation", "median", "median_2d"),
    [
        ("0.25", "auto", 0, 0),
        ("0.5", "auto", 1, 0),
        ("2", "auto", 0, 0),
        ("2", "nearest", 2, 0),
        ("1.5", "linear", 0, 0),
        ("0.25", "auto", 0, 2),
        ("0.5", "auto", 1, 3),
        ("2", "auto", 0, 1),
        ("1", "auto", 0, 2),
    ],
)
def test_composite_output_scale_matches_whole_image_resize(
    tmp_path: Path, scale, interpolation, median, median_2d
) -> None:
    cv2 = pytest.importorskip("cv2")
    from fractions import Fraction

    rng = np.random.default_rng(3)
    volume = rng.integers(0, 256, size=(3, 70, 90), dtype=np.uint8)
    zarr_path = tmp_path / "volume.zarr"
    _write_group(zarr_path, {"0": volume}, chunks=(3, 16, 16))
    output = tmp_path / "scaled.tif"
    args = [
        "--zarr", str(zarr_path), "--output", str(output), "--method", "max",
        "--output-scale", scale, "--resize-interpolation", interpolation,
        "--median-radius", str(median), "--median-2d-radius", str(median_2d),
        "--tile-size", "16", "--workers", "2", "--no-progress",
    ]
    assert composite_from_zarr.main(args) == 0
    from scipy import ndimage

    filtered = ndimage.median_filter(volume, size=2 * median + 1, mode="nearest") if median else volume
    image = filtered.max(axis=0)
    fraction = Fraction(scale)
    flag = {
        "auto": cv2.INTER_AREA if fraction < 1 else cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
    }[interpolation]
    expected = _scaled_reference(image, fraction, flag) if fraction != 1 else image
    if median_2d:
        # The 2D stage runs after the resize, on the full output image.
        expected = cv2.medianBlur(np.ascontiguousarray(expected), 2 * median_2d + 1)
    got = tifffile.imread(output)
    assert got.shape == expected.shape == (-(-70 * fraction.numerator // fraction.denominator), -(-90 * fraction.numerator // fraction.denominator))
    if fraction.numerator == 1 or fraction.denominator == 1:
        np.testing.assert_array_equal(got, expected)
    else:
        # OpenCV rounds its fixed-point interpolation coefficients from a float source
        # coordinate, so for scales like 3/2 the whole-image and tiled results can
        # differ by a grey level or two at positions with different float rounding.
        np.testing.assert_allclose(got.astype(int), expected.astype(int), atol=2)


def test_composite_single_zarr_default_name_and_input_exclusivity(tmp_path: Path) -> None:
    zarr_path = tmp_path / "volume.zarr"
    _write_group(zarr_path, {"0": np.zeros((2, 16, 16), dtype=np.uint8)}, chunks=(1, 16, 16))
    base = ["--method", "mean", "--median-radius", "1", "--workers", "1", "--no-progress"]
    assert composite_from_zarr.main(["--zarr", str(zarr_path), *base]) == 0
    assert (tmp_path / "volume_mean_median1_0_2.tif").exists()
    assert composite_from_zarr.main(["--zarr", str(zarr_path), "--center-layers", "1", *base]) == 0
    assert (tmp_path / "volume_mean_median1_0_1.tif").exists()
    assert composite_from_zarr.main(["--zarr", str(zarr_path), "--output-scale", "0.5", *base]) == 0
    assert (tmp_path / "volume_mean_median1_scale0.5_area_0_2.tif").exists()
    assert tifffile.imread(tmp_path / "volume_mean_median1_scale0.5_area_0_2.tif").shape == (8, 8)
    assert composite_from_zarr.main(
        ["--zarr", str(zarr_path), "--output-scale", "2", "--resize-interpolation", "nearest", *base]
    ) == 0
    assert (tmp_path / "volume_mean_median1_scale2_nearest_0_2.tif").exists()
    with pytest.raises(SystemExit):
        composite_from_zarr.parse_args(["--zarr", str(zarr_path), "--input-root", str(tmp_path), *base])
    with pytest.raises(SystemExit):
        composite_from_zarr.parse_args(base)
    with pytest.raises(SystemExit):
        composite_from_zarr.parse_args(["--input-root", str(tmp_path), "--output", "x.tif", *base])


def test_composite_failure_leaves_no_published_or_partial_tiff(tmp_path: Path) -> None:
    folder = tmp_path / "dataset"
    folder.mkdir()
    _write_group(
        folder / "surface.zarr",
        {"0": np.zeros((1, 16, 16), dtype=np.uint8)},
        chunks=(1, 16, 16),
    )
    args = composite_from_zarr.parse_args(
        ["--input-root", str(folder), "--method", "max", "--workers", "1", "--no-progress"]
    )
    job = composite_from_zarr._build_jobs(
        input_root=folder,
        start_z=args.start_z,
        end_z=args.end_z,
        method=args.method,
        resolution=args.resolution,
        overwrite=args.overwrite,
        parallelism=args.parallelism,
        workers=args.workers,
        no_progress=args.no_progress,
        compression=args.compression,
    )[0]
    with patch.object(composite_from_zarr, "_project_chunk", side_effect=OSError("injected")):
        with pytest.raises(OSError, match="injected"):
            composite_from_zarr._write_projection(job)
    assert list(folder.glob("*.tif")) == []


def test_downloader_writes_v2_manifest_resumes_and_rejects_source_drift(
    tmp_path: Path,
) -> None:
    datasets_root = tmp_path / "datasets"
    segment = datasets_root / "dataset" / "segment"
    segment.mkdir(parents=True)
    grid_y, grid_x = np.mgrid[1:5, 1:5].astype(np.float32)
    surface = Tifxyz(
        _x=grid_x,
        _y=grid_y,
        _z=np.ones_like(grid_x),
        uuid="segment",
        _mask=np.ones_like(grid_x, dtype=bool),
    )
    write_tifxyz(segment, surface, overwrite=True)
    label = np.zeros((3, 4, 4), dtype=np.uint8)
    label[1] = 255
    _write_group(segment / "segment_inklabels.zarr", {"0": label}, chunks=(1, 2, 2))
    _write_group(segment / "segment_supervision_mask.zarr", {"0": label}, chunks=(1, 2, 2))

    source_path = tmp_path / "source.zarr"
    source_data = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8) % 251
    source_data = source_data.astype(np.uint8)
    _write_group(source_path, {"0": source_data}, chunks=(2, 2, 2))
    volumes_json = tmp_path / "volumes.json"
    volumes_json.write_text(
        json.dumps({"dataset": {"volume_path": str(source_path), "volume_scale": 0}})
    )
    output_root = tmp_path / "output"
    args = [
        "--datasets-root", str(datasets_root),
        "--volumes-json", str(volumes_json),
        "--output-root", str(output_root),
        "--patch-size", "2",
        "--overlap-fraction", "0.5",
        "--download-workers", "1",
    ]
    assert download_required_zarr_chunks.main(args) == 0
    output_path = output_root / "dataset.zarr"
    assert json.loads((output_path / ".zgroup").read_text()) == {"zarr_format": 2}
    output = zarr.open_group(output_path, mode="r")
    plan = output.attrs[download_required_zarr_chunks.DOWNLOAD_PLAN_ATTR]
    progress = output.attrs[download_required_zarr_chunks.DOWNLOAD_PROGRESS_ATTR]
    assert plan["source"]["volume_path"] == str(source_path.resolve())
    assert progress["completed_chunk_ids_by_scale"]["0"] == plan["chunk_ids_by_scale"]["0"]
    assert download_required_zarr_chunks.Segment is Segment
    first_export = np.asarray(output["0"])
    assert np.count_nonzero(first_export) > 0
    assert download_required_zarr_chunks.main(args) == 0
    np.testing.assert_array_equal(zarr.open_group(output_path, mode="r")["0"], first_export)

    with pytest.raises(ValueError, match="source, chunk plan, array schema"):
        download_required_zarr_chunks.copy_chunks_to_output(
            volume_path=str(source_path),
            volume_scale=0,
            volume_auth_json=None,
            output_path=output_path,
            chunk_ids_by_scale={0: ()},
            worker_count=1,
            tqdm_desc="test",
            overwrite=False,
            recompress="balanced",
        )

    array_metadata_path = output_path / "0" / ".zarray"
    array_metadata_text = array_metadata_path.read_text()
    array_metadata = json.loads(array_metadata_text)
    array_metadata["chunks"] = [1, 2, 2]
    array_metadata_path.write_text(json.dumps(array_metadata))
    with pytest.raises(ValueError, match="output schema for array"):
        download_required_zarr_chunks.main(args)
    array_metadata_path.write_text(array_metadata_text)

    other_source = tmp_path / "other.zarr"
    _write_group(other_source, {"0": np.zeros_like(source_data)}, chunks=(2, 2, 2))
    volumes_json.write_text(
        json.dumps({"dataset": {"volume_path": str(other_source), "volume_scale": 0}})
    )
    with pytest.raises(ValueError, match="source, chunk plan, array schema"):
        download_required_zarr_chunks.main(args)


@pytest.mark.parametrize(
    "dataset_name",
    [
        "",
        ".",
        "..",
        "/tmp/escape",
        "../escape",
        "family/segment",
        r"family\segment",
        r"C:\escape",
        r"C:relative",
    ],
)
@pytest.mark.parametrize("list_style", [False, True])
def test_downloader_rejects_dataset_paths_at_json_boundary(
    tmp_path: Path,
    dataset_name: str,
    list_style: bool,
) -> None:
    volumes_json = tmp_path / "volumes.json"
    entry = {"volume_path": "unused.zarr"}
    payload = [{"dataset": dataset_name, **entry}] if list_style else {dataset_name: entry}
    volumes_json.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="one directory name, not a path"):
        download_required_zarr_chunks.load_dataset_sources(volumes_json)


def test_downloader_accepts_dots_inside_dataset_basename(tmp_path: Path) -> None:
    volumes_json = tmp_path / "volumes.json"
    volumes_json.write_text(json.dumps({"dataset..v2": "source.zarr"}))

    sources = download_required_zarr_chunks.load_dataset_sources(volumes_json)

    assert sources["dataset..v2"].dataset_name == "dataset..v2"


def test_downloader_overwrite_cannot_escape_output_root(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir()
    output_root = tmp_path / "output"
    escaped_output = tmp_path / "escape.zarr"
    escaped_output.mkdir()
    sentinel = escaped_output / "sentinel"
    sentinel.write_text("preserve")
    volumes_json = tmp_path / "volumes.json"
    volumes_json.write_text(
        json.dumps({"../escape": {"volume_path": "unused.zarr"}})
    )

    with pytest.raises(ValueError, match="one directory name, not a path"):
        download_required_zarr_chunks.main(
            [
                "--datasets-root", str(datasets_root),
                "--volumes-json", str(volumes_json),
                "--output-root", str(output_root),
                "--overwrite",
            ]
        )

    assert sentinel.read_text() == "preserve"


def test_downloader_records_only_completed_writes_and_resumes_after_failure(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.zarr"
    output_path = tmp_path / "output.zarr"
    source = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    _write_group(source_path, {"0": source}, chunks=(2, 2, 2))
    kwargs = {
        "volume_path": str(source_path),
        "volume_scale": 0,
        "volume_auth_json": None,
        "output_path": output_path,
        "chunk_ids_by_scale": {0: ((0, 0, 0), (1, 1, 1))},
        "worker_count": 1,
        "tqdm_desc": "test",
        "overwrite": False,
        "recompress": "fast",
    }
    original_copy = download_required_zarr_chunks._copy_one_chunk
    call_count = 0

    def fail_second(chunk_id: tuple[int, int, int]) -> tuple[int, int, int]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected chunk failure")
        return original_copy(chunk_id)

    with patch.object(
        download_required_zarr_chunks,
        "_copy_one_chunk",
        side_effect=fail_second,
    ):
        with pytest.raises(OSError, match="injected chunk failure"):
            download_required_zarr_chunks.copy_chunks_to_output(**kwargs)
    progress = zarr.open_group(output_path, mode="r").attrs[
        download_required_zarr_chunks.DOWNLOAD_PROGRESS_ATTR
    ]
    assert progress["completed_chunk_ids_by_scale"] == {"0": [[0, 0, 0]]}

    stats = download_required_zarr_chunks.copy_chunks_to_output(**kwargs)
    assert stats == {
        "copied_counts_by_scale": {0: 1},
        "existing_counts_by_scale": {0: 1},
    }
    np.testing.assert_array_equal(
        zarr.open_group(output_path, mode="r")["0"][2:4, 2:4, 2:4],
        source[2:4, 2:4, 2:4],
    )
