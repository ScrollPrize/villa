"""The structure-tensor writers' on-disk results: Zarr v2 with the expected layout."""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr
from numcodecs import Blosc
from vc_reader.reader import create_group

from vesuvius.data.utils import create_zarr_array, open_zarr_group
from vesuvius.structure_tensor.vf_format import require_array, zarr_codec


def _is_v2_array_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, ".zarray"))


def test_require_array_keeps_what_exists(tmp_path):
    root = create_group(tmp_path / "g.zarr")
    first = require_array(root, "U", shape=(3, 2, 2, 2), chunks=(3, 2, 2, 2), dtype=np.float32)
    first[...] = 7.0
    again = require_array(root, "U", shape=(3, 2, 2, 2), chunks=(3, 1, 1, 1), dtype=np.float32)
    np.testing.assert_array_equal(again[...], 7.0)
    assert again.chunks == (3, 2, 2, 2)
    with pytest.raises(ValueError, match="shape"):
        require_array(root, "U", shape=(3, 1, 1, 1), chunks=(3, 1, 1, 1), dtype=np.float32)
    root.create_group("sub")
    with pytest.raises(TypeError, match="group, not an array"):
        require_array(root, "sub", shape=(1,), chunks=(1,), dtype=np.uint8)


def test_zarr_codec_names():
    assert zarr_codec("ZSTD", 3) == {"cname": "zstd", "level": 3, "shuffle": 1}
    assert zarr_codec("none", 3) == {"compressor": None}
    assert zarr_codec("unknown", 3) == {"cname": "zstd", "level": 1, "shuffle": 1}


def test_voxelize_rechunk_keeps_store_v2(tmp_path):
    """--chunk_size N routes level 0 through a dask rechunk; that must not leave
    a v3 array or a root zarr.json in the v2 label store (issue #1448 follow-up)."""
    from vesuvius.image_proc.run.voxelize_objs import _rechunk_with_dask

    path = str(tmp_path / "labels.zarr")
    root = open_zarr_group(path, mode="w")
    level0 = create_zarr_array(root, "0", shape=(8, 8, 8), chunks=(8, 8, 8), dtype=np.uint16, compressor=Blosc())
    expected = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    level0[...] = expected
    level0.attrs["note"] = "kept"

    rechunked = _rechunk_with_dask(level0, 4, "labels", num_workers=1)
    assert tuple(rechunked.chunks) == (4, 4, 4)
    assert not os.path.exists(tmp_path / "labels.zarr" / "zarr.json")
    assert _is_v2_array_dir(str(tmp_path / "labels.zarr" / "0"))
    reopened = zarr.open_group(path, mode="r")
    assert list(reopened.keys()) == ["0"]
    np.testing.assert_array_equal(reopened["0"][:], expected)
    np.testing.assert_array_equal(rechunked[:], expected)
    assert reopened["0"].attrs["note"] == "kept"


def test_eigenanalysis_writes_ome_and_eigen_arrays(tmp_path):
    """_finalize_structure_tensor_torch creates 3 OME vector groups, confidence and eigen* arrays."""
    from vesuvius.structure_tensor.create_st import _finalize_structure_tensor_torch

    Z, Y, X = 2, 4, 4
    st = np.zeros((6, Z, Y, X), dtype=np.float32)
    st[0] = 3.0  # Jzz
    st[3] = 2.0  # Jyy
    st[5] = 1.0  # Jxx
    zarr_path = str(tmp_path / "st.zarr")
    create_group(zarr_path).create_array("structure_tensor", shape=st.shape, chunks=(6, Z, Y, X), dtype="f4")[...] = st

    _finalize_structure_tensor_torch(
        zarr_path=zarr_path,
        chunk_size=None,
        num_workers=0,
        codec={"cname": "zstd", "level": 1},
        verbose=False,
        swap_eigenvectors=False,
        device="cpu",
        ome_out=True,
        ome_downsample=1,
        ome_scale="0",
        confidence_metric="fa",
        keep_eigen=True,
    )

    out = zarr.open_group(zarr_path, mode="r")
    for name in ("first_component", "second_component", "normal"):
        for axis in ("z", "y", "x"):
            assert tuple(out[f"{name}/{axis}/0"].shape) == (Z, Y, X)
            assert _is_v2_array_dir(os.path.join(zarr_path, name, axis, "0"))
    assert tuple(out["confidence/0"].shape) == (Z, Y, X)
    assert tuple(out["eigenvectors"].shape) == (9, Z, Y, X)
    assert tuple(out["eigenvalues"].shape) == (3, Z, Y, X)
    assert tuple(out["eigenvectors"].chunks) == (1, Z, Y, X)
    ev = np.asarray(out["eigenvalues"][:])
    assert np.isfinite(ev).all()


def test_ome_u8_vector_writer_creates_scale_arrays(tmp_path):
    from vesuvius.structure_tensor.vf_format import OMEU8VectorWriter

    writer = OMEU8VectorWriter(
        output_path=str(tmp_path / "vf.zarr"),
        group_name="normal",
        vol_shape_zyx=(4, 8, 8),
        chunks_zyx=(4, 4, 4),
        downsample=2,
        make_confidence=True,
    )
    assert tuple(writer.ds_z.shape) == (2, 4, 4)
    assert tuple(writer.ds_conf.shape) == (2, 4, 4)
    assert _is_v2_array_dir(str(tmp_path / "vf.zarr" / "normal" / "z" / "0"))
    # Re-opening must reuse the existing arrays rather than fail
    OMEU8VectorWriter(
        output_path=str(tmp_path / "vf.zarr"),
        group_name="normal",
        vol_shape_zyx=(4, 8, 8),
        chunks_zyx=(4, 4, 4),
        downsample=2,
        make_confidence=True,
    )
