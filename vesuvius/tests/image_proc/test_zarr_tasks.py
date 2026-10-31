"""Each zarr task, run end to end on a small volume through the task CLI's classes."""
import numpy as np
import pytest
from vc_reader.reader import ZarrWriter, create_group, open_group, open_zarr, write_pyramid

from vesuvius.image_proc.run.zarr_tasks import get_task
from vesuvius.image_proc.run.zarr_tasks.tasks.edt_dilate import EdtDilateConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.merge import MergeConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import RecompressConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.remap import RemapConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.resize import ResizeConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.scale import ScaleConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import ThresholdConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.transpose import TransposeConfig
from vesuvius.image_proc.run.zarr_tasks.tasks.zero_range import ZeroRangeConfig

SHAPE = (8, 10, 12)


def _volume(seed=0):
    return np.random.default_rng(seed).integers(1, 250, SHAPE, dtype=np.uint8)


def _write(path, data, levels=1):
    writer = ZarrWriter.create(f"{path}/0", shape=data.shape, chunks=(4, 4, 4), dtype=data.dtype, cname="zstd", shuffle=2,
                               separator="/")
    writer[...] = data
    if levels:
        write_pyramid(path, levels)
    return str(path)


def _config(cls, source, output, **kwargs):
    return cls(input_zarr=source, output_zarr=output, num_workers=2, **kwargs)


@pytest.fixture
def source(tmp_path):
    data = _volume()
    return _write(tmp_path / "in.zarr", data), data


def _run(name, config):
    get_task(name)(config).run()


def test_threshold_writes_binary_nearest_pyramid(tmp_path, source):
    path, data = source
    out = str(tmp_path / "out.zarr")
    _run("threshold", _config(ThresholdConfig, path, out, threshold=100.0, num_levels=2))
    expected = np.where(data > 100, 255, 0).astype(np.uint8)
    group = open_group(out)
    np.testing.assert_array_equal(group["0"][...], expected)
    np.testing.assert_array_equal(group["1"][...], expected[::2, ::2, ::2])
    assert group.levels == ("0", "1")
    assert group["0"].compressor["cname"] == "zstd" and group["0"].meta.separator == "/"


def test_chunks_of_only_zero_stay_off_disk(tmp_path, source):
    path, data = source
    out = tmp_path / "out.zarr"
    _run("threshold", _config(ThresholdConfig, path, str(out), threshold=255.0, num_levels=1))
    assert not list((out / "0").glob("*/*/*"))
    assert open_zarr(out / "0")[...].max() == 0


def test_scale_and_transpose(tmp_path, source):
    path, data = source
    scaled = str(tmp_path / "scaled.zarr")
    _run("scale", _config(ScaleConfig, path, scaled, scale_factor=2.0, num_levels=1))
    np.testing.assert_array_equal(open_zarr(f"{scaled}/0")[...], data.repeat(2, 0).repeat(2, 1).repeat(2, 2))
    moved = str(tmp_path / "moved.zarr")
    _run("transpose", _config(TransposeConfig, path, moved, transpose_order="xzy", num_levels=1))
    np.testing.assert_array_equal(open_zarr(f"{moved}/0")[...], data.transpose(2, 0, 1))
    assert [axis["name"] for axis in open_group(moved).multiscales[0]["axes"]] == ["x", "z", "y"]


def test_zero_range_scales_by_level(source):
    path, data = source
    level1 = open_zarr(f"{path}/1")[...]
    _run("zero-range", _config(ZeroRangeConfig, path, None, z_start=2, z_end=3, replace_value=9))
    expected = data.copy()
    expected[2:4] = 9
    np.testing.assert_array_equal(open_zarr(f"{path}/0")[...], expected)
    level1[1] = 9
    np.testing.assert_array_equal(open_zarr(f"{path}/1")[...], level1)


def test_edt_dilate(tmp_path):
    data = np.zeros(SHAPE, np.uint8)
    data[4, 5, 6] = 1
    path = _write(tmp_path / "in.zarr", data, levels=0)
    out = str(tmp_path / "out.zarr")
    _run("edt-dilate", _config(EdtDilateConfig, path, out, distance=1.0, chunk_size=(8, 10, 12), resolution="0"))
    dilated = open_zarr(f"{out}/0")[...]
    assert dilated[4, 5, 6] == 1 and dilated[3, 5, 6] == 1 and dilated[3, 4, 6] == 0


@pytest.mark.parametrize(("interpolation", "method"), [("average", "mean"), ("nearest", "mode")])
def test_merge_takes_the_maximum_and_builds_levels(tmp_path, interpolation, method):
    first, second = _volume(1), _volume(2)
    a, b = _write(tmp_path / "a.zarr", first, 0), _write(tmp_path / "b.zarr", second, 0)
    out = str(tmp_path / "out.zarr")
    _run("merge", _config(MergeConfig, a, out, input2_zarr=b, num_levels=2, interpolation=interpolation))
    group = open_group(out)
    np.testing.assert_array_equal(group["0"][...], np.maximum(first, second))
    assert group["1"].shape == (4, 5, 6)
    assert group.multiscales[0]["metadata"]["downsampling_method"] == method


def test_remap_keeps_attributes_and_every_level(tmp_path):
    data = np.full(SHAPE, 3, np.uint8)
    data[0] = 7
    path = _write(tmp_path / "in.zarr", data)
    out = str(tmp_path / "out.zarr")
    create_group(path).attrs["note"] = "kept"
    _run("remap", _config(RemapConfig, path, out, remap_dict={7: 1}))
    result = open_group(out)
    np.testing.assert_array_equal(result["0"][...], np.where(data == 7, 1, data))
    assert result.array_keys() == ["0", "1"]
    assert result.attrs["note"] == "kept"


def test_resize_matches_the_reference(tmp_path, source):
    path, data = source
    reference = _write(tmp_path / "ref.zarr", np.ones((4, 5, 6), np.uint8))
    out = str(tmp_path / "out.zarr")
    _run("resize", _config(ResizeConfig, path, out, reference_zarr=reference, interpolation="nearest"))
    written = open_zarr(f"{out}/0")
    assert written.shape == (4, 5, 6) and written.meta.separator == "/" and written.compressor["cname"] == "zstd"
    assert written[...].any()


def test_recompress_to_output_and_in_place(tmp_path, source):
    path, data = source
    out = str(tmp_path / "out.zarr")
    _run("recompress", _config(RecompressConfig, path, out, compression_level=5, num_levels=2))
    written = open_group(out)
    np.testing.assert_array_equal(written["0"][...], data)
    assert written["0"].compressor["clevel"] == 5 and written["0"].compressor["shuffle"] == 2

    _run("recompress", RecompressConfig(input_zarr=path, output_zarr=None, num_workers=2, inplace=True, compression_level=7))
    level0 = open_zarr(f"{path}/0")
    np.testing.assert_array_equal(level0[...], data)
    assert level0.compressor["clevel"] == 7 and level0.meta.separator == "/"


def test_recompress_fills_corrupt_chunks_when_asked(tmp_path, source):
    path, data = source
    chunk = tmp_path / "in.zarr" / "0" / "0" / "0" / "0"
    chunk.write_bytes(b"not a chunk")
    out = str(tmp_path / "out.zarr")
    _run("recompress", _config(RecompressConfig, path, out, skip_corrupt=True, num_levels=1))
    expected = data.copy()
    expected[:4, :4, :4] = 0
    np.testing.assert_array_equal(open_zarr(f"{out}/0")[...], expected)
