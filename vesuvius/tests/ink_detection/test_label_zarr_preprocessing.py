from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.preprocessing.create_label_zarrs import (
    DEFAULT_LABEL_SLICE,
    build_pyramid_with_mode,
    convert_image,
    find_target_images,
    main,
    parse_target_image,
)


def _zarr_format(path: Path) -> int:
    metadata = path / ".zgroup"
    assert metadata.is_file()
    import json

    return int(json.loads(metadata.read_text())["zarr_format"])


def test_label_image_names_share_segment_vocabulary_and_accept_png(tmp_path):
    paths = [
        tmp_path / "Segment-A_InkLabels.TIFF",
        tmp_path / "segment-a_supervision_mask_v3.png",
        tmp_path / "segment-a_validation_mask.tif",
    ]
    for path in paths:
        path.touch()

    assert parse_target_image(paths[0]) == {
        "prefix": "Segment-A",
        "label_kind": "inklabels",
        "version_num": None,
        "extension": ".TIFF",
    }
    assert parse_target_image(paths[1])["version_num"] == 3
    assert parse_target_image(paths[2])["label_kind"] == "validation_mask"
    padded_version = tmp_path / "segment-a_inklabels_v01.tif"
    padded_version.touch()
    assert parse_target_image(padded_version)["version_num"] == 1
    assert parse_target_image(tmp_path / "missing_inklabels.tif") is None


def test_discovery_is_stable_skips_zarr_trees_and_adds_one_composite(tmp_path):
    segment = tmp_path / "segment-a"
    segment.mkdir()
    labels = [
        segment / "segment-a_inklabels.tif",
        segment / "segment-a_supervision_mask.tif",
    ]
    composites = [
        segment / "segment-a_composite-b.tif",
        segment / "segment-a_max-a.tif",
    ]
    for path in [*labels, *composites]:
        path.touch()
    hidden = segment / "old.zarr"
    hidden.mkdir()
    (hidden / "inside_inklabels.tif").touch()

    assert find_target_images(tmp_path) == [
        labels[0],
        labels[1],
        composites[0],
    ]


def test_label_and_composite_pyramids_keep_nearest_and_mean_rounding():
    image_YX = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8
    )
    nearest = build_pyramid_with_mode(image_YX, levels=2)
    mean = build_pyramid_with_mode(image_YX, levels=2, downsample_mode="mean")

    assert nearest[0].shape == (65, 3, 3)
    np.testing.assert_array_equal(nearest[0][DEFAULT_LABEL_SLICE], image_YX)
    np.testing.assert_array_equal(
        nearest[1][DEFAULT_LABEL_SLICE], np.array([[0, 2], [6, 8]])
    )
    np.testing.assert_array_equal(
        mean[1][DEFAULT_LABEL_SLICE], np.array([[2, 4], [6, 8]])
    )
    assert np.count_nonzero(nearest[1][:DEFAULT_LABEL_SLICE]) == 0


def test_conversion_writes_v2_ome_metadata_and_preserves_skip_overwrite(tmp_path):
    label_path = tmp_path / "segment-a_inklabels.tif"
    first_YX = np.arange(35, dtype=np.uint8).reshape(5, 7)
    tifffile.imwrite(label_path, first_YX)

    first = convert_image(label_path, levels=3)
    output = label_path.with_suffix(".zarr")
    assert first["status"] == "written"
    assert _zarr_format(output) == 2
    group = zarr.open_group(output, mode="r")
    assert sorted(group.array_keys()) == ["0", "1", "2"]
    assert group.attrs["multiscales"][0]["axes"] == [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    np.testing.assert_array_equal(group["0"][DEFAULT_LABEL_SLICE], first_YX)
    assert group["0"].attrs["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]

    replacement_YX = np.full((5, 7), 99, dtype=np.uint8)
    tifffile.imwrite(label_path, replacement_YX)
    assert convert_image(label_path, levels=3)["status"] == "skipped"
    np.testing.assert_array_equal(
        zarr.open_group(output, mode="r")["0"][DEFAULT_LABEL_SLICE], first_YX
    )

    assert convert_image(label_path, levels=3, overwrite=True)["status"] == "written"
    np.testing.assert_array_equal(
        zarr.open_group(output, mode="r")["0"][DEFAULT_LABEL_SLICE],
        replacement_YX,
    )


def test_tiled_tiff_streaming_matches_flat_image(tmp_path):
    label_path = tmp_path / "segment-a_supervision_mask.tif"
    image_YX = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    tifffile.imwrite(label_path, image_YX, tile=(16, 16))

    result = convert_image(label_path, levels=2)
    assert result["streamed_tiled_tiff"] == "true"
    group = zarr.open_group(label_path.with_suffix(".zarr"), mode="r")
    np.testing.assert_array_equal(group["0"][DEFAULT_LABEL_SLICE], image_YX)
    np.testing.assert_array_equal(
        group["1"][DEFAULT_LABEL_SLICE], image_YX[::2, ::2]
    )


def test_striped_tiff_streaming_matches_flat_image(tmp_path):
    """A plain ``tifffile.imwrite`` with no ``tile=`` writes a striped TIFF --
    this is the input shape #1231's second half reports OOMing on, because the
    old streaming gate checked only ``page.is_tiled``.
    """
    label_path = tmp_path / "segment-a_validation_mask.tif"
    image_YX = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    tifffile.imwrite(label_path, image_YX)  # no tile= -> striped by default

    with tifffile.TiffFile(label_path) as tif:
        assert not tif.pages[0].is_tiled, "fixture must actually be striped"

    result = convert_image(label_path, levels=2)
    assert result["streamed_tiled_tiff"] == "true"
    group = zarr.open_group(label_path.with_suffix(".zarr"), mode="r")
    np.testing.assert_array_equal(group["0"][DEFAULT_LABEL_SLICE], image_YX)
    np.testing.assert_array_equal(
        group["1"][DEFAULT_LABEL_SLICE], image_YX[::2, ::2]
    )


@pytest.mark.parametrize("compression", ["lzw", "deflate", "packbits"])
def test_striped_tiff_streaming_covers_common_codecs(tmp_path, compression):
    """Block decode must round-trip for every codec the streaming path
    claims to support, not just uncompressed strips."""
    label_path = tmp_path / f"segment-a_{compression}_supervision_mask.tif"
    image_YX = np.random.default_rng(0).integers(
        0, 255, size=(80, 120), dtype=np.uint8
    )
    tifffile.imwrite(label_path, image_YX, compression=compression)

    result = convert_image(label_path, levels=1)
    assert result["streamed_tiled_tiff"] == "true"
    group = zarr.open_group(label_path.with_suffix(".zarr"), mode="r")
    np.testing.assert_array_equal(group["0"][DEFAULT_LABEL_SLICE], image_YX)


def test_unstreamable_codec_falls_back_without_error(tmp_path):
    """A codec outside the verified set (e.g. JPEG) must fall through to the
    existing in-memory path rather than attempt an unsupported block decode."""
    label_path = tmp_path / "segment-a_jpeg_inklabels.tif"
    image_YX = np.random.default_rng(1).integers(
        0, 255, size=(64, 64), dtype=np.uint8
    )
    tifffile.imwrite(label_path, image_YX, compression="jpeg")

    result = convert_image(label_path, levels=1)
    assert result["streamed_tiled_tiff"] == "false"
    group = zarr.open_group(label_path.with_suffix(".zarr"), mode="r")
    # JPEG is lossy, so this is a sanity check on shape/dtype, not exact
    # pixel equality.
    assert group["0"][DEFAULT_LABEL_SLICE].shape == image_YX.shape


def test_multipage_tiff_is_left_to_the_existing_path_unchanged(tmp_path):
    """Multi-page TIFFs are explicitly out of scope for this change.

    The rest of this module assumes a single flat 2D label image; converting
    a genuine multi-page file already produces silently wrong output on the
    pre-existing in-memory path today (tifffile.imread stacks pages, and the
    channel-squeeze logic then mistakes the page axis for height). That bug
    is real but separate, and this PR does not fix it -- it only guarantees
    not to touch multi-page behaviour at all, so the streaming gate must
    return "false" (unchanged from before this PR) for any multi-page input,
    whether tiled or striped.
    """
    label_path = tmp_path / "segment-a_multipage_supervision_mask.tif"
    # z=5, not 3 or 4: tifffile's imwrite heuristically treats a leading axis
    # of exactly 3 or 4 on a uint8 array as RGB(A) color planes and writes ONE
    # page instead of several -- confirmed by hitting that ambiguity with
    # z=3 while writing this test, which is itself a small illustration of
    # how easy it is to end up with an unintended single-page file.
    volume_ZYX = np.random.default_rng(2).integers(
        0, 2, size=(5, 20, 30), dtype=np.uint8
    )
    tifffile.imwrite(label_path, volume_ZYX)  # writes 5 separate pages

    with tifffile.TiffFile(label_path) as tif:
        assert len(tif.pages) == 5, "fixture must actually be multi-page"

    result = convert_image(label_path, levels=1)
    assert result["streamed_tiled_tiff"] == "false"


def test_label_command_reports_failure_and_cli_module_help(tmp_path, capsys):
    bad = tmp_path / "bad_inklabels.tif"
    bad.write_bytes(b"not a tiff")
    assert main([str(tmp_path), "--workers", "1", "--levels", "2"]) == 1
    output = capsys.readouterr().out
    assert "1 failed" in output
    assert f"ERROR {bad}" in output

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vesuvius.ink_detection.preprocessing.create_label_zarrs",
            "-h",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--overwrite" in completed.stdout


def test_label_command_rerun_counts_existing_output_as_skipped(tmp_path, capsys):
    label_path = tmp_path / "segment-a_inklabels.png"
    import cv2

    assert cv2.imwrite(
        str(label_path), np.array([[0, 255], [255, 0]], dtype=np.uint8)
    )
    assert main([str(tmp_path), "--workers", "1", "--levels", "1"]) == 0
    assert main([str(tmp_path), "--workers", "1", "--levels", "1"]) == 0
    assert "0 written, 1 skipped, 0 failed" in capsys.readouterr().out


def test_label_command_validates_scan_root(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Root folder does not exist"):
        main([str(missing)])
    file_path = tmp_path / "file"
    file_path.touch()
    with pytest.raises(NotADirectoryError, match="Root path is not a directory"):
        main([str(file_path)])
