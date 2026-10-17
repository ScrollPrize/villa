"""The prepare step writes only layers [START_LAYER, END_LAYER) into the surface-volume zarr; the inference
step must index that zarr by channel, not by absolute layer index."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


def _layer_stack(tmp_path, n=8):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:02d}.tif"
        tifffile.imwrite(p, np.full((16, 16), i, dtype=np.uint8))
        paths.append(str(p))
    return paths


def test_prepared_zarr_window_is_rebased(tmp_path):
    layers = _layer_stack(tmp_path)
    start, end = 2, 7
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(layers[start:end], out, chunk_size=16, use_compression=False)
    processing.record_layer_window(out, start, end)

    assert processing.resolve_zarr_layer_window(out, start, end) == (0, end - start)
    z = zarr.open(out, mode="r")
    assert z.shape[2] == end - start
    assert int(z[0, 0, 0]) == start  # channel 0 is source layer START_LAYER


def test_prepared_zarr_rejects_layers_it_does_not_hold(tmp_path):
    layers = _layer_stack(tmp_path)
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(layers[2:7], out, chunk_size=16, use_compression=False)
    processing.record_layer_window(out, 2, 7)
    with pytest.raises(RuntimeError):
        processing.resolve_zarr_layer_window(out, 1, 7)


def test_external_full_depth_zarr_keeps_absolute_indices(tmp_path):
    layers = _layer_stack(tmp_path)
    out = str(tmp_path / "full.zarr")
    processing.create_surface_volume_zarr(layers, out, chunk_size=16, use_compression=False)
    assert processing.resolve_zarr_layer_window(out, 2, 7) == (2, 7)


def test_layers_source_channel_mismatch_is_an_error(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    import inference

    layers = _layer_stack(tmp_path)
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(layers[1:7], out, chunk_size=16, use_compression=False)
    monkeypatch.setattr(inference.CFG, "in_chans", 6)
    with pytest.raises(ValueError):
        inference.preprocess_layers(out, None, False, start_z=1, end_z=7)


def test_layer_files_must_be_one_per_index_without_gaps():
    assert processing.contiguous_layer_window(["01.tif", "02.tif", "03.tif"]) == (1, 4)
    with pytest.raises(RuntimeError, match="missing"):
        processing.contiguous_layer_window(["01.tif", "02.tif", "04.tif"])
    with pytest.raises(RuntimeError, match="more than once"):
        processing.contiguous_layer_window(["a/layers/01.tif", "b/layers/01.tif", "a/layers/02.tif"])


def test_window_recorded_from_the_files_present(tmp_path):
    # A segment holding layers 1..5 only, asked for [1, 8): the zarr records [1, 6), so asking for layers it
    # does not hold is an error instead of a silently shorter or shifted stack.
    layers = _layer_stack(tmp_path)[1:6]
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(layers, out, chunk_size=16, use_compression=False)
    processing.record_layer_window(out, *processing.contiguous_layer_window(layers))
    assert processing.resolve_zarr_layer_window(out, 2, 6) == (1, 5)
    with pytest.raises(RuntimeError, match="not all present"):
        processing.resolve_zarr_layer_window(out, 1, 8)


def test_unreadable_attrs_stop_instead_of_falling_back(tmp_path, monkeypatch):
    layers = _layer_stack(tmp_path)
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(layers[1:7], out, chunk_size=16, use_compression=False)
    processing.record_layer_window(out, 1, 7)

    def failing_open(*args, **kwargs):
        raise OSError("simulated S3/cache read error")

    monkeypatch.setattr(processing.zarr, "open", failing_open)
    with pytest.raises(RuntimeError, match="Could not read the layer-window attrs"):
        processing.resolve_zarr_layer_window(out, 2, 6)


@pytest.mark.parametrize("kept", ["layer_start", "layer_end"])
def test_partial_window_attrs_are_an_error(tmp_path, kept):
    layers = _layer_stack(tmp_path)
    out = str(tmp_path / "sv.zarr")
    processing.create_surface_volume_zarr(layers[1:7], out, chunk_size=16, use_compression=False)
    processing.record_layer_window(out, 1, 7)
    z = zarr.open(out, mode="r+")
    for name in processing.LAYER_WINDOW_ATTRS:
        if name != kept:
            del z.attrs[name]

    with pytest.raises(RuntimeError, match="records only"):
        processing.resolve_zarr_layer_window(out, 2, 6)
