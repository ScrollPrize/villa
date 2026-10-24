"""Volume(verbose=True) must open a multiscale (group) store.

load_data printed ``data.shape`` and ``data.dtype`` after opening the store.
Scroll and segment volumes are multiscale OME-Zarr groups, which have neither,
so every ``Volume(type='scroll'|'segment', ..., verbose=True)`` raised
AttributeError: 'Group' object has no attribute 'shape'.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.data.volume import Volume

_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3


@pytest.fixture
def multiscale_path(tmp_path):
    path = str(tmp_path / "scroll.zarr")
    if _ZARR_V3:
        root = zarr.open_group(store=path, mode="w", zarr_format=2)
        create = root.create_array
    else:
        root = zarr.open_group(store=path, mode="w")
        create = root.create_dataset
    a0 = create("0", shape=(16, 16, 16), chunks=(8, 8, 8), dtype="uint8")
    a1 = create("1", shape=(8, 8, 8), chunks=(8, 8, 8), dtype="uint8")
    a0[:] = 7
    a1[:] = 7
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}, {"path": "1"}]}]
    return path


def test_verbose_scroll_volume_opens_multiscale_group(multiscale_path, monkeypatch, capsys):
    # Resolve the "scroll" to the local store instead of the YAML config.
    monkeypatch.setattr(Volume, "get_url_from_yaml", lambda self: multiscale_path)

    vol = Volume(type="scroll", scroll_id=1, domain="local", verbose=True)

    assert isinstance(vol.data, zarr.Group)
    assert np.asarray(vol[0:2, 0:2, 0:2]).sum() == 7 * 8
    out = capsys.readouterr().out
    assert "ERROR initializing Volume" not in out
    # meta() still reports every level.
    assert "Level 0 Shape: (16, 16, 16)" in out
    assert "Level 1 Shape: (8, 8, 8)" in out


def test_verbose_load_data_reports_shape_of_single_array(tmp_path, capsys):
    path = str(tmp_path / "single.zarr")
    zarr.open(path, mode="w", shape=(4, 4, 4), chunks=(4, 4, 4), dtype="uint8")
    vol = Volume.__new__(Volume)
    vol.type, vol.path, vol.url = "scroll", None, path
    vol.verbose, vol.cache, vol.cache_size_mb, vol.anon = True, False, 256, False

    data = vol.load_data()

    assert isinstance(data, zarr.Array)
    assert "Shape: (4, 4, 4), Dtype: uint8" in capsys.readouterr().out
