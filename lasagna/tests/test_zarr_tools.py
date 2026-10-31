"""Smoke tests of Lasagna's Zarr command-line tools on small synthetic volumes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter, create_group, open_group, open_zarr

import convert_fit_zarr_to_vc3d_omezarr as convert
import mix_channels
import predict3d_holescan as holescan
from omezarr_pyramid import _mean_pool2x_u8, _moment_pool2x_normals, build_normal_omezarr_pyramid, build_scalar_omezarr_pyramid
from scripts import rechunk_omezarr, zarr_threshold_masks


def _volume(shape=(16, 16, 16), seed=0):
    return np.random.default_rng(seed).integers(1, 255, shape, dtype=np.uint8)


def _pyramid(root: Path, data: np.ndarray, *, chunk=8, levels=3) -> Path:
    create_group(root, attrs={"keep": 1})
    for level in range(levels):
        shape = tuple(max(1, (n + 2**level - 1) // 2**level) for n in data.shape)
        ZarrWriter.create(root / str(level), shape=shape, chunks=[min(chunk, n) for n in shape], dtype=np.uint8)
    ZarrWriter.open(root / "0")[:] = data
    return root


def _expected_mean(data: np.ndarray) -> np.ndarray:
    return _mean_pool2x_u8(data)


def test_scalar_pyramid_matches_lasagna_mean_on_odd_shapes(tmp_path):
    data = _volume((17, 9, 16))
    root = _pyramid(tmp_path / "s.zarr", data)
    build_scalar_omezarr_pyramid(root, 0, 3, workers=2, label="s")
    level1 = open_zarr(root / "1")[...]
    np.testing.assert_array_equal(level1, _expected_mean(data))
    np.testing.assert_array_equal(open_zarr(root / "2")[...], _expected_mean(level1))
    assert open_group(root).attrs["lasagna_pyramid_downsample"] == "mean_pool2x"
    assert open_group(root).attrs["keep"] == 1


def test_crop_limits_scalar_pyramid(tmp_path):
    data = _volume()
    root = _pyramid(tmp_path / "s.zarr", data)
    build_scalar_omezarr_pyramid(root, 0, 2, workers=1, crop_zyx=(0, 0, 0, 4, 4, 4))
    assert set(ZarrWriter.open(root / "1").stored_chunks()) == {(0, 0, 0)}


def test_zero_overrides_and_normal_pyramids_match_their_reductions(tmp_path):
    data = _volume((17, 9, 16))
    data[3, 3, 3] = 0
    grad = _pyramid(tmp_path / "g.zarr", data)
    build_scalar_omezarr_pyramid(grad, 0, 2, workers=2, zero_overrides=True, scan_existing_source_chunks=True)
    np.testing.assert_array_equal(open_zarr(grad / "1")[...], _mean_pool2x_u8(data, zero_overrides=True))
    assert open_group(grad).attrs["lasagna_pyramid_downsample"] == "mean_pool2x_zero_overrides"

    nx_data, ny_data = _volume((17, 9, 16), 1), _volume((17, 9, 16), 2)
    nx, ny = _pyramid(tmp_path / "nx.zarr", nx_data), _pyramid(tmp_path / "ny.zarr", ny_data)
    build_normal_omezarr_pyramid(nx, ny, 0, 2, workers=2)
    nx_down, ny_down = _moment_pool2x_normals(nx_data, ny_data)
    np.testing.assert_array_equal(open_zarr(nx / "1")[...], nx_down)
    np.testing.assert_array_equal(open_zarr(ny / "1")[...], ny_down)


def _run_rechunk(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["rechunk_omezarr.py", *map(str, args)])
    rechunk_omezarr.main()


def test_rechunk_out_of_place_and_in_place(tmp_path, monkeypatch):
    data = _volume()
    src = _pyramid(tmp_path / "s.zarr", data)
    dst = tmp_path / "d.zarr"
    _run_rechunk(monkeypatch, src, dst, "-c", "4", "-j", "2")
    assert open_zarr(dst / "0").chunks == (4, 4, 4)
    assert open_zarr(dst / "0").meta.separator == "/"
    np.testing.assert_array_equal(open_zarr(dst / "0")[...], data)
    np.testing.assert_array_equal(open_zarr(dst / "1")[...], _expected_mean(data))
    assert open_group(dst).attrs["keep"] == 1
    assert [d["path"] for d in open_group(dst).attrs["multiscales"][0]["datasets"]] == ["0", "1", "2"]
    with pytest.raises(FileExistsError):
        _run_rechunk(monkeypatch, src, dst)

    _run_rechunk(monkeypatch, src, "--in-place", "-c", "2", "-j", "2")
    assert open_zarr(src / "0").chunks == (2, 2, 2)
    np.testing.assert_array_equal(open_zarr(src / "0")[...], data)
    assert not list(tmp_path.glob("*.bak-*")) and not list(tmp_path.glob("*.tmp-rechunk-*"))


def test_rechunk_does_not_store_missing_source_chunks(tmp_path, monkeypatch):
    src = _pyramid(tmp_path / "s.zarr", np.zeros((16, 16, 16), np.uint8))
    ZarrWriter.open(src / "0")[0:8, 0:8, 0:8] = 9
    dst = tmp_path / "d.zarr"
    _run_rechunk(monkeypatch, src, dst, "-c", "8", "--levels", "1", "-j", "1")
    assert set(ZarrWriter.open(dst / "0").stored_chunks()) == {(0, 0, 0)}


def test_rechunk_rebuild_scales_of_normal_pair(tmp_path, monkeypatch):
    nx_data, ny_data = _volume(seed=1), _volume(seed=2)
    nx, ny = _pyramid(tmp_path / "nx.zarr", nx_data, levels=2), _pyramid(tmp_path / "ny.zarr", ny_data, levels=2)
    _run_rechunk(monkeypatch, nx, "--in-place", "--rebuild-scales-only", "--normal-pair", ny, "-j", "2")
    np.testing.assert_array_equal(open_zarr(nx / "1")[...], _moment_pool2x_normals(nx_data, ny_data)[0])


def _czyx(path: Path, data: np.ndarray, channels) -> Path:
    array = ZarrWriter.create(path, shape=data.shape, chunks=(1, 8, 8, 8), dtype=np.uint8)
    array[:] = data
    array.attrs["preprocess_params"] = {"scaledown": 1, "channels": channels}
    return path


def test_mix_channels_takes_named_channel_from_other(tmp_path, monkeypatch):
    base = _czyx(tmp_path / "base.zarr", np.stack([_volume(seed=3), _volume(seed=4)]), ["cos", "grad_mag"])
    other = _czyx(tmp_path / "other.zarr", np.stack([_volume(seed=5), _volume(seed=6)]), ["cos", "grad_mag"])
    output = tmp_path / "out.zarr"
    monkeypatch.setattr(sys, "argv", ["mix", "--base", str(base), "--other", str(other), "--output", str(output),
                                      "--channels-other", "grad_mag", "--workers", "2"])
    mix_channels.main()
    mixed = open_zarr(output)
    np.testing.assert_array_equal(mixed[0], open_zarr(base)[0])
    np.testing.assert_array_equal(mixed[1], open_zarr(other)[1])
    assert mixed.attrs["preprocess_params"]["channels"] == ["cos", "grad_mag"]


def test_convert_fit_builds_channel_pyramids(tmp_path):
    channels = ["cos", "grad_mag", "nx", "ny"]
    data = np.stack([_volume(seed=seed) for seed in range(4)])
    array = ZarrWriter.create(tmp_path / "fit.zarr", shape=data.shape, chunks=(1, 8, 8, 8), dtype=np.uint8)
    array[:] = data
    array.attrs["preprocess_params"] = {"scaledown": 2, "channels": channels}
    convert.run(input_path=str(tmp_path / "fit.zarr"), output_prefix=str(tmp_path / "out"), levels=3, chunk=8, workers=2,
                base_shape=(32, 32, 32))
    cos = tmp_path / "out_cos.ome.zarr"
    np.testing.assert_array_equal(open_zarr(cos / "1")[...], data[0])
    np.testing.assert_array_equal(open_zarr(cos / "2")[...], _mean_pool2x_u8(data[0]))
    np.testing.assert_array_equal(open_zarr(cos / "0")[...], data[0].repeat(2, 0).repeat(2, 1).repeat(2, 2))
    np.testing.assert_array_equal(open_zarr(tmp_path / "out_grad_mag.ome.zarr" / "2")[...],
                                  _mean_pool2x_u8(data[1], zero_overrides=True))
    nx_down, ny_down = _moment_pool2x_normals(data[2], data[3])
    np.testing.assert_array_equal(open_zarr(tmp_path / "out_nx.ome.zarr" / "2")[...], nx_down)
    np.testing.assert_array_equal(open_zarr(tmp_path / "out_ny.ome.zarr" / "2")[...], ny_down)
    attrs = open_group(cos).attrs
    assert attrs["vc3d_convert"]["first_filled_level"] == 1
    assert attrs["lasagna_pyramid_downsample"] == "mean_pool2x"
    assert open_group(tmp_path / "out_nx.ome.zarr").attrs["lasagna_pyramid_downsample"] == "normal_second_moment_mean_pool2x"


def test_holescan_lists_only_stored_chunks_of_the_channel(tmp_path):
    array = ZarrWriter.create(tmp_path / "v.zarr", shape=(2, 8, 8, 8), chunks=(1, 4, 4, 4), dtype=np.uint8)
    array[0, 0:4, 0:4, 4:8] = 1
    array[1, 4:8, 0:4, 0:4] = 1
    reader = open_zarr(tmp_path / "v.zarr")
    assert holescan._stored_chunk_coords(holescan.ScanTarget("a", tmp_path / "v.zarr", channel_index=0), reader) == {(0, 0, 1)}
    assert holescan._stored_chunk_coords(holescan.ScanTarget("b", tmp_path / "v.zarr", channel_index=1), reader) == {(1, 0, 0)}


def test_threshold_mask_is_written_and_applied(tmp_path):
    keep = np.zeros((8, 8, 8), dtype=bool)
    keep[:4] = True
    mask = tmp_path / "mask.zarr"
    zarr_threshold_masks._write_mask_zarr(out_path=mask, keep_mask=keep, source_path=Path("source"), scale_zyx=(1.0, 1.0, 1.0),
                                          z_range_full=(0, 8), z_range_array=(0, 8))
    written = open_zarr(mask)
    np.testing.assert_array_equal(written[...], keep.astype(np.uint8) * 255)
    assert written.attrs["mask_values"] == {"masked_out": 0, "keep": 255}
    assert written.compressor["id"] == "zlib"

    grad = tmp_path / "grad"
    ZarrWriter.create(grad, shape=(8, 8, 8), chunks=(4, 4, 4), dtype=np.uint8)[:] = 7
    total, skipped, changed = zarr_threshold_masks._apply_external_keep_mask(
        grad_level_path=grad, mask=written, mask_channel=None, workers=2)
    assert (total, skipped) == (8, 0)
    assert changed == 4
    expected = np.where(keep, 7, 0)
    np.testing.assert_array_equal(open_zarr(grad)[...], expected)

    sparse = tmp_path / "sparse"
    ZarrWriter.create(sparse, shape=(8, 8, 8), chunks=(4, 4, 4), dtype=np.uint8)[0:4] = 1
    missing = zarr_threshold_masks._missing_chunk_mask_for_slab(
        array_path=sparse, shape=(8, 8, 8), chunks=(4, 4, 4), z_range=(0, 8), channel=None)
    np.testing.assert_array_equal(missing, ~keep)


def test_fitted_labels_and_channel_statistics_read_vc_reader_stores(tmp_path, monkeypatch, capsys):
    import analyze_zarr_channels
    import debug_normals_extract
    import fitted_to_unet_labels

    fitted = create_group(tmp_path / "fitted.zarr")
    for name, shape in (("normal", (3, 4, 4, 4)), ("winding", (4, 4, 4)), ("validity", (4, 4, 4)), ("density", (4, 4, 4))):
        fitted.create_array(name, shape=shape, chunks=shape, dtype=np.float32)[...] = 1.0
    fitted_to_unet_labels.convert(str(tmp_path / "fitted.zarr"), str(tmp_path / "labels"))
    assert list((tmp_path / "labels").iterdir())

    path = _czyx(tmp_path / "c.zarr", np.stack([_volume(seed=7), _volume(seed=8)]), ["cos", "grad_mag"])
    monkeypatch.setattr(sys, "argv", ["analyze", str(path)])
    analyze_zarr_channels.main()
    assert "cos" in capsys.readouterr().out
    array, params = debug_normals_extract._open_zarr_array(str(path))
    assert array.shape == (2, 16, 16, 16) and params["channels"] == ["cos", "grad_mag"]


def test_label_tiff_to_cc_zarr(tmp_path, monkeypatch):
    import labels_to_cc_zarr
    from vc_reader import images

    labels = np.zeros((8, 16, 16), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    images.write_tiff(tmp_path / "labels.tif", labels)
    monkeypatch.setattr(sys, "argv", ["labels_to_cc", "--input", str(tmp_path / "labels.tif"), "--output", str(tmp_path / "cc.zarr"),
                                      "--binary-zarr", str(tmp_path / "binary.zarr"), "--chunk-size", "8"])
    labels_to_cc_zarr.main()
    cc = open_zarr(tmp_path / "cc.zarr")
    assert cc.attrs["n_components"] >= 1
    np.testing.assert_array_equal(open_group(tmp_path / "binary.zarr")["0"][...] != 0, cc[...] != 0)


def test_eval_prediction_store_is_a_zstd_v3_group(tmp_path):
    prediction = np.random.default_rng(9).random((8, 8, 8, 8), dtype=np.float32)
    group = create_group(tmp_path / "pred.zarr", version=3, overwrite=True)
    group.create_array("prediction", shape=prediction.shape, chunks=(8, 8, 8, 8), dtype=np.float32, cname="zstd", level=3, shuffle=1)[...] = prediction
    np.testing.assert_array_equal(open_group(tmp_path / "pred.zarr")["prediction"][...], prediction)
