import numpy as np
import pytest
import zarr

_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3


@pytest.fixture
def multiscale_zarr2_group(tmp_path):
    """A multiscale OME-Zarr group written in zarr v2 format (levels '0','1').

    Group creation is version-branched because the v2-format-request API
    differs: zarr 3's ``open_group`` takes ``zarr_format=2`` and its Group
    exposes ``create_array``; zarr 2 writes v2 by default (no such kwarg)
    and its Group exposes ``create_dataset`` instead.
    """
    path = str(tmp_path / "multiscale_v2.zarr")
    rng = np.random.default_rng(0)
    full = rng.integers(0, 255, (64, 64, 64), dtype="uint8")
    if _ZARR_V3:
        root = zarr.open_group(store=path, mode="w", zarr_format=2)
        a0 = root.create_array("0", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
        a1 = root.create_array("1", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8")
    else:
        root = zarr.open_group(store=path, mode="w")
        a0 = root.create_dataset("0", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8")
        a1 = root.create_dataset("1", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8")
    a0[:] = full
    a1[:] = full[::2, ::2, ::2]
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}, {"path": "1"}]}]
    return path


@pytest.mark.unit
def test_volume_reads_multiscale_zarr2_group(multiscale_zarr2_group):
    # Regression: zarr 3 requires string keys for group member access
    # (self.data[0] raises TypeError); Volume must resolve levels through
    # a version-agnostic accessor instead of integer indexing the group.
    from vesuvius.data.volume import Volume

    vol = Volume(type="zarr", path=multiscale_zarr2_group, format="zarr", normalization_scheme="none")
    assert isinstance(vol.data, zarr.Group)
    assert np.dtype(vol.dtype) == np.dtype("uint8")
    assert vol._num_levels() == 2
    # highest-resolution level is readable through the group
    block = vol[0:16, 0:16, 0:16]
    assert np.asarray(block).shape == (16, 16, 16)
    # explicit subvolume index selects the second level
    low_res = vol[0:8, 0:8, 0:8, 1]
    assert np.asarray(low_res).shape == (8, 8, 8)


@pytest.mark.unit
def test_volume_level_accessors_memoize(multiscale_zarr2_group):
    from vesuvius.data.volume import Volume

    vol = Volume(type="zarr", path=multiscale_zarr2_group, format="zarr", normalization_scheme="none")
    assert vol._level(0) is vol._level(0)
    assert vol._num_levels() == vol._num_levels()
