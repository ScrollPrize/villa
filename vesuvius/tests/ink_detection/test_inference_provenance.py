"""Provenance and physical scale recorded in flat ink prediction TIFFs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.inference.infer import write_output_tiff
from vesuvius.ink_detection.inference.provenance import (
    DESCRIPTION_HEADER,
    PROVENANCE_KEY,
    build_provenance,
    physical_scale_from_root,
    provenance_description,
    public_location,
    resolution_tags,
    sha256_file,
)


def _ome_root(tmp_path: Path, scale_by_level: dict[str, list[float]], unit="micrometer"):
    root = zarr.open_group(str(tmp_path / "surface.zarr"), mode="w")
    for level in scale_by_level:
        root.create_array(level, shape=(3, 8, 8), chunks=(3, 8, 8), dtype="uint8")
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "z", "type": "space", "unit": unit},
                {"name": "y", "type": "space", "unit": unit},
                {"name": "x", "type": "space", "unit": unit},
            ],
            "datasets": [
                {
                    "path": level,
                    "coordinateTransformations": [{"type": "scale", "scale": scale}],
                }
                for level, scale in scale_by_level.items()
            ],
        }
    ]
    return root


def _read(path: Path) -> dict:
    """Read the record and the resolution tags straight off the TIFF tags."""

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        description = tags.get("ImageDescription")
        record = None
        if description is not None:
            header, _, body = description.value.partition("\n")
            assert header == DESCRIPTION_HEADER
            record = json.loads(body)[PROVENANCE_KEY]
        # tifffile always writes resolution tags; without a physical scale
        # they are 1/1 with ResolutionUnit NONE, which means "unknown".
        um = None
        unit = tags.get("ResolutionUnit")
        if unit is not None and unit.value == tifffile.RESUNIT.INCH:
            (xn, xd), (yn, yd) = tags["XResolution"].value, tags["YResolution"].value
            um = {"x": 25400 * xd / xn, "y": 25400 * yd / yn}
        return {
            "record": record,
            "um_per_px": um,
            "shaped": page.is_shaped,
            "unit": None if unit is None else unit.value,
        }


def test_physical_scale_reads_the_requested_level_not_level_zero(tmp_path):
    root = _ome_root(
        tmp_path, {"0": [9.362, 9.362, 9.362], "1": [9.362, 18.724, 18.724]}
    )
    level_zero = physical_scale_from_root(root, "0", depth_axis_first=True)
    level_one = physical_scale_from_root(root, "1", depth_axis_first=True)
    assert level_zero is not None and level_one is not None
    assert level_zero["um_per_px_y"] == pytest.approx(9.362)
    assert level_one["um_per_px_y"] == pytest.approx(18.724)
    assert level_one["um_per_px_x"] == pytest.approx(18.724)
    assert level_one["unit"] == "micrometer"
    assert level_one["axes"] == ["z", "y", "x"]


def test_physical_scale_keeps_anisotropic_axes_apart_and_lowercases_names(tmp_path):
    root = _ome_root(tmp_path, {"0": [1.0, 10.0, 5.0]})
    root.attrs["multiscales"][0]["axes"] = [
        {"name": "Z", "unit": "micrometer"},
        {"name": "Y", "unit": "micrometer"},
        {"name": "X", "unit": "micrometer"},
    ]
    record = physical_scale_from_root(root, "0", depth_axis_first=True)
    assert record["um_per_px_y"] == 10.0
    assert record["um_per_px_x"] == 5.0


def test_physical_scale_is_none_for_bare_arrays_and_unknown_levels(tmp_path):
    array = zarr.open_array(
        str(tmp_path / "bare.zarr"), mode="w", shape=(3, 8, 8), dtype="uint8"
    )
    assert physical_scale_from_root(array, "0", depth_axis_first=True) is None
    root = _ome_root(tmp_path, {"0": [1.0, 2.0, 2.0]})
    assert physical_scale_from_root(root, "3", depth_axis_first=True) is None


@pytest.mark.parametrize("unit", ["nanometer", "millimeter", "parsec", None])
def test_non_micrometre_units_are_recorded_but_produce_no_resolution(tmp_path, unit):
    # vc_render_tifxyz's default --voxel-unit label is "nanometer" while the
    # value it writes comes from meta.json in micrometres; converting such a
    # label would give tags 1000x off, so only micrometre units get tags.
    root = _ome_root(tmp_path, {"0": [9.362, 9.362, 9.362]}, unit=unit)
    record = physical_scale_from_root(root, "0", depth_axis_first=True)
    assert record is not None
    assert record["scale"] == [9.362, 9.362, 9.362]
    assert record["unit"] == unit
    assert record["um_per_px_y"] is None and record["um_per_px_x"] is None
    assert resolution_tags(record["um_per_px_y"], record["um_per_px_x"]) is None


def test_ome_0_5_layout_is_read_too(tmp_path):
    root = zarr.open_group(str(tmp_path / "ome5.zarr"), mode="w")
    root.create_array("0", shape=(3, 8, 8), chunks=(3, 8, 8), dtype="uint8")
    root.attrs["ome"] = {
        "multiscales": [
            {
                "axes": [{"name": n, "unit": "micrometer"} for n in "zyx"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [2.0, 2.399, 2.399]}
                        ],
                    }
                ],
            }
        ]
    }
    record = physical_scale_from_root(root, "0", depth_axis_first=True)
    assert record["um_per_px_x"] == pytest.approx(2.399)


@pytest.mark.parametrize(
    "um",
    [9.362, 2.399, 7.91, 5.273333333333333, 6.241333333333333, 7.909999847412109, 0.5, 150.0],
)
def test_resolution_rationals_fit_uint32_and_match_pixels_per_inch(um):
    (x_res, y_res), unit = resolution_tags(um, um)
    assert unit == "INCH"
    assert x_res == y_res
    numerator, denominator = x_res
    assert 0 < numerator <= 2**32 - 1 and 0 < denominator <= 2**32 - 1
    assert numerator / denominator == pytest.approx(25400 / um, rel=1e-9)


def test_resolution_tags_are_exact_for_short_decimals_and_ordered_x_then_y():
    (x_res, y_res), _ = resolution_tags(10.0, 5.0)  # y=10 um, x=5 um
    assert x_res == (5080, 1)
    assert y_res == (2540, 1)
    (x_res, _), _ = resolution_tags(9.362, 9.362)
    assert x_res[0] * 9362 == x_res[1] * 25400 * 1000
    assert resolution_tags(None, 1.0) is None
    assert resolution_tags(0.0, 1.0) is None
    # Finer than a uint32 can express in pixels per inch: no tag, not a wrong one.
    assert resolution_tags(1e-6, 1e-6) is None


def test_public_location_strips_url_credentials_and_keeps_paths():
    assert (
        public_location("https://alice:s3cret@dl.ash2txt.org/full-scrolls/x.zarr/")
        == "https://dl.ash2txt.org/full-scrolls/x.zarr/"
    )
    assert public_location("https://host:8443/a.zarr") == "https://host:8443/a.zarr"
    assert public_location("https://u:p@[::1]:8080/a.zarr") == "https://[::1]:8080/a.zarr"
    assert public_location("https://u:p@Host.Example/x") == "https://Host.Example/x"
    assert public_location("s3://bucket/key.zarr") == "s3://bucket/key.zarr"
    assert public_location(Path("/data/w035/9um.zarr")) == "/data/w035/9um.zarr"


def _document(checkpoint: Path, input_zarr, physical_scale=None):
    return build_provenance(
        checkpoint={
            "name": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "weights": "model",
        },
        input_zarr=input_zarr,
        level="0",
        input_shape=(28, 64, 48),
        depth_axis_first=True,
        layer_indices=np.arange(5, 22),
        layer_start=None,
        layer_end=None,
        direction="reverse",
        patch_size=16,
        stride=8,
        overlap=0.5,
        blend_mode="hann",
        tta_mirror=True,
        amp_dtype="float16",
        compile_requested=False,
        compile_mode="reduce-overhead",
        batch_size=4,
        device="cuda:0",
        torch_version="2.14.0",
        mask_name=None,
        physical_scale=physical_scale,
        preprocessing="tifxyz_robust",
    )


def test_write_output_tiff_pixels_and_tile_bytes_are_unchanged_by_the_tags(tmp_path):
    rng = np.random.default_rng(0)
    height, width = 40, 56
    probability = rng.random((height, width), dtype=np.float32) * 3.0
    weight = rng.random((height, width), dtype=np.float32) * 3.0
    weight[:8, :] = 0.0  # never-covered rows stay zero

    plain = tmp_path / "plain.tif"
    # The tile iterator normalises its stores in place (they are throwaway
    # accumulation arrays in the real path), so each write gets fresh copies.
    write_output_tiff(probability.copy(), weight.copy(), plain, (16, 16))

    checkpoint = tmp_path / "step-000001.pth"
    checkpoint.write_bytes(b"not really a checkpoint")
    scale = {
        "level": "0",
        "axes": ["z", "y", "x"],
        "scale": [9.362, 10.0, 5.0],
        "unit": "micrometer",
        "um_per_px_y": 10.0,
        "um_per_px_x": 5.0,
    }
    document = _document(
        checkpoint, "https://alice:s3cret@dl.ash2txt.org/x.zarr/", physical_scale=scale
    )
    resolution, unit = resolution_tags(10.0, 5.0)
    tagged = tmp_path / "tagged.tif"
    write_output_tiff(
        probability.copy(),
        weight.copy(),
        tagged,
        (16, 16),
        description=provenance_description(document),
        resolution=resolution,
        resolutionunit=unit,
    )

    assert np.array_equal(tifffile.imread(plain), tifffile.imread(tagged))
    with tifffile.TiffFile(plain) as before, tifffile.TiffFile(tagged) as after:
        assert before.pages[0].databytecounts == after.pages[0].databytecounts
        assert before.pages[0].is_tiled and after.pages[0].is_tiled
        before_tags = {tag.name: tag.value for tag in before.pages[0].tags.values()}
        after_tags = {tag.name: tag.value for tag in after.pages[0].tags.values()}
        changed = {
            name
            for name in set(before_tags) | set(after_tags)
            if before_tags.get(name) != after_tags.get(name)
        }
        # Tile offsets move because the IFD grew; the byte counts (asserted
        # equal above) show the tile payloads themselves are unchanged.
        changed -= {"TileOffsets", "StripOffsets"}
        # Only the four provenance/scale tags differ; tifffile writes a
        # placeholder resolution (1/1, unit NONE) even without a scale.
        assert changed == {"ImageDescription", "XResolution", "YResolution", "ResolutionUnit"}
        assert before_tags["ResolutionUnit"] == tifffile.RESUNIT.NONE
        assert after_tags["ResolutionUnit"] == tifffile.RESUNIT.INCH

    before = _read(plain)
    assert before["record"] is None and before["um_per_px"] is None

    after = _read(tagged)
    assert not after["shaped"], "the header line must keep tifffile's heuristic off"
    record = after["record"]
    assert record["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert record["input"]["zarr"] == "https://dl.ash2txt.org/x.zarr/"
    assert "s3cret" not in json.dumps(record)
    assert record["run"]["direction"] == "reverse"
    assert record["run"]["layer_indices"] == list(range(5, 22))
    assert record["run"]["tta_mirror"] is True
    assert record["input"]["shape"] == [28, 64, 48]
    assert record["physical_scale"]["um_per_px_y"] == 10.0
    assert after["um_per_px"] == {"y": pytest.approx(10.0), "x": pytest.approx(5.0)}
    with tifffile.TiffFile(tagged) as tif:
        assert tif.pages[0].tags["ImageDescription"].value.isascii()


def test_created_timestamp_honours_source_date_epoch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "c.pth"
    checkpoint.write_bytes(b"x")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    first = provenance_description(_document(checkpoint, "a.zarr"))
    second = provenance_description(_document(checkpoint, "a.zarr"))
    assert first == second
    assert '"created_utc":"2023-11-14T22:13:20+00:00"' in first


def _tiny_checkpoint(tmp_path: Path) -> Path:
    import torch

    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.models.model import make_model

    from .test_model_foundation import _config_mapping

    config_mapping = _config_mapping("vesuvius_unet_2p5d", depth=3, side=16)
    config_mapping["image_normalization"] = "robust_mad"
    model = make_model(InkConfig.from_mapping(config_mapping))
    with torch.no_grad():
        for value in model.state_dict().values():
            if torch.is_floating_point(value):
                value.zero_()
    checkpoint = tmp_path / "step-000007.pth"
    torch.save(
        {"config": config_mapping, "model": model.state_dict(), "step": 7},
        checkpoint,
    )
    return checkpoint


def test_cli_output_tiff_records_checkpoint_run_and_scale(tmp_path):
    from vesuvius.ink_detection.inference.infer import main

    checkpoint = _tiny_checkpoint(tmp_path)
    root = _ome_root(tmp_path / "ome", {"0": [9.362, 9.362, 9.362]})
    root["0"][:] = 1
    ome_output = tmp_path / "ome_prediction.tif"
    common = ["--workers", "0", "--no-compile", "--blend-mode", "constant"]
    assert (
        main(
            [
                str(tmp_path / "ome" / "surface.zarr"),
                str(checkpoint),
                str(ome_output),
                "--direction",
                "both",
                *common,
            ]
        )
        == 0
    )
    forward = _read(ome_output)
    reverse = _read(ome_output.with_name("ome_prediction_reverse.tif"))
    record = forward["record"]
    assert record["checkpoint"] == {
        "name": "step-000007.pth",
        "sha256": sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "weights": "model",
        "step": 7,
    }
    assert record["input"]["shape"] == [3, 8, 8]
    assert record["input"]["level"] == "0"
    assert record["input"]["zarr"].endswith("surface.zarr")
    assert record["run"]["direction"] == "forward"
    assert record["run"]["layer_indices"] == [0, 1, 2]
    assert record["run"]["blend_mode"] == "constant"
    assert record["run"]["compile_requested"] is False
    device = record["run"]["device"]
    assert device.startswith(("cpu", "cuda", "mps"))
    if not device.startswith("cuda"):
        assert record["run"]["amp_dtype"] is None  # autocast only applies on CUDA
    assert record["run"]["torch_version"]
    assert record["run"]["preprocessing"] == "tifxyz_robust"
    assert record["physical_scale"]["um_per_px_y"] == pytest.approx(9.362)
    assert forward["um_per_px"]["y"] == pytest.approx(9.362)
    assert reverse["record"]["run"]["direction"] == "reverse"
    assert reverse["record"]["run"]["layer_indices"] == [2, 1, 0]
    assert np.all(tifffile.imread(ome_output) == 127)

    bare = zarr.open_array(
        str(tmp_path / "bare.zarr"),
        mode="w",
        shape=(3, 16, 16),
        chunks=(3, 16, 16),
        dtype="u1",
        zarr_format=2,
    )
    bare[:] = 1
    bare_output = tmp_path / "bare_prediction.tif"
    assert main([str(tmp_path / "bare.zarr"), str(checkpoint), str(bare_output), *common]) == 0
    bare_record = _read(bare_output)
    assert bare_record["record"]["physical_scale"] is None
    assert bare_record["um_per_px"] is None
    assert bare_record["unit"] == tifffile.RESUNIT.NONE
    assert np.all(tifffile.imread(bare_output) == 127)
