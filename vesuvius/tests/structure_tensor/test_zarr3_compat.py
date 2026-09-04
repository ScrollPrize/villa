"""The structure-tensor writers must work on a zarr 3 install (issue #1448).

``pyproject.toml`` allows ``zarr>=2.18.7,<4``; on the 3.x half of that range
``Group.create_dataset`` / ``require_dataset`` / ``zarr.DirectoryStore`` no
longer exist, so ``vesuvius.compute_st`` died before writing a voxel. These
tests drive the real writers (eigenanalysis, OME vector writer, vector-field
creator) through the shared compatibility helpers and check the on-disk
result is still Zarr v2 with the same chunk layout under either zarr major.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from vesuvius.data.utils import (
    _ZARR_V3,
    create_zarr_array,
    open_zarr_group,
    require_zarr_array,
)


def _is_v2_array_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, ".zarray"))


def test_open_zarr_group_creates_v2_layout(tmp_path):
    root = open_zarr_group(str(tmp_path / "g.zarr"), mode="w")
    assert os.path.exists(tmp_path / "g.zarr" / ".zgroup")
    assert not os.path.exists(tmp_path / "g.zarr" / "zarr.json")
    if _ZARR_V3:
        assert root.metadata.zarr_format == 2


def test_open_zarr_group_append_creates_v2_when_missing(tmp_path):
    root = open_zarr_group(str(tmp_path / "new.zarr"), mode="a")
    assert os.path.exists(tmp_path / "new.zarr" / ".zgroup")
    assert not os.path.exists(tmp_path / "new.zarr" / "zarr.json")
    create_zarr_array(root, "U", shape=(2, 2), chunks=(2, 2), dtype=np.float32, compressor=Blosc())
    assert _is_v2_array_dir(str(tmp_path / "new.zarr" / "U"))


@pytest.mark.skipif(not _ZARR_V3, reason="zarr 3 stores only exist under zarr 3")
def test_open_zarr_group_append_keeps_existing_v3_store(tmp_path):
    """Appending must not write a second .zgroup beside an existing zarr.json."""
    path = str(tmp_path / "v3.zarr")
    v3 = zarr.open_group(path, mode="w", zarr_format=3)
    v3.create_array("existing", shape=(2, 2), chunks=(2, 2), dtype=np.uint8)

    root = open_zarr_group(path, mode="a")
    assert root.metadata.zarr_format == 3
    assert not os.path.exists(tmp_path / "v3.zarr" / ".zgroup")
    # numcodecs Blosc is translated to the v3 BloscCodec on this branch
    arr = create_zarr_array(
        root, "new", shape=(2, 2), chunks=(2, 2), dtype=np.float32,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
    )
    arr[...] = 1.0
    reopened = zarr.open_group(path, mode="r")
    assert sorted(reopened.keys()) == ["existing", "new"]
    np.testing.assert_array_equal(reopened["new"][:], 1.0)


@pytest.mark.skipif(not _ZARR_V3, reason="zarr 3 stores only exist under zarr 3")
def test_open_zarr_group_append_recreates_empty_v3_leftover_as_v2(tmp_path):
    """A failed run on zarr 3 leaves an empty zarr.json-only group; re-running
    on the same --output must produce the v2 store the consumers read."""
    path = str(tmp_path / "leftover.zarr")
    zarr.open_group(path, mode="w", zarr_format=3)
    assert os.path.exists(tmp_path / "leftover.zarr" / "zarr.json")

    root = open_zarr_group(path, mode="a")
    assert root.metadata.zarr_format == 2
    create_zarr_array(root, "U", shape=(2, 2), chunks=(2, 2), dtype=np.float32, compressor=Blosc())
    assert not os.path.exists(tmp_path / "leftover.zarr" / "zarr.json")
    assert os.path.exists(tmp_path / "leftover.zarr" / ".zgroup")
    assert _is_v2_array_dir(str(tmp_path / "leftover.zarr" / "U"))
    assert list(zarr.open_group(path, mode="r").keys()) == ["U"]


def test_create_zarr_array_matches_create_dataset_layout(tmp_path):
    root = open_zarr_group(str(tmp_path / "g.zarr"), mode="w")
    arr = create_zarr_array(
        root,
        "structure_tensor",
        shape=(6, 4, 8, 8),
        chunks=(6, 4, 4, 4),
        dtype=np.float32,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
        write_empty_chunks=False,
    )
    arr[:, :, :4, :4] = 1.0
    array_dir = str(tmp_path / "g.zarr" / "structure_tensor")
    assert _is_v2_array_dir(array_dir)
    meta = json.load(open(os.path.join(array_dir, ".zarray")))
    assert meta["chunks"] == [6, 4, 4, 4]
    assert meta["compressor"]["id"] == "blosc"
    assert meta["compressor"]["cname"] == "zstd"
    # write_empty_chunks=False: only the written chunk is on disk
    chunk_files = [f for f in os.listdir(array_dir) if not f.startswith(".")]
    assert chunk_files == ["0.0.0.0"]
    reopened = zarr.open_group(str(tmp_path / "g.zarr"), mode="r")
    np.testing.assert_array_equal(reopened["structure_tensor"][0, 0, :4, :4], 1.0)


def test_create_zarr_array_with_data_and_nested_separator(tmp_path):
    root = open_zarr_group(str(tmp_path / "g.zarr"), mode="w")
    data = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    create_zarr_array(
        root,
        "labels",
        data=data,
        chunks=(1, 4, 4),
        compressor=Blosc(),
        dimension_separator="/",
    )
    array_dir = tmp_path / "g.zarr" / "labels"
    assert (array_dir / "0" / "0" / "0").exists()
    np.testing.assert_array_equal(zarr.open_group(str(tmp_path / "g.zarr"), mode="r")["labels"][:], data)


def test_require_zarr_array_returns_existing_or_creates(tmp_path):
    root = open_zarr_group(str(tmp_path / "g.zarr"), mode="a")
    first = require_zarr_array(root, "U", shape=(3, 2, 2, 2), chunks=(3, 2, 2, 2), dtype=np.float32, compressor=None)
    first[...] = 7.0
    again = require_zarr_array(root, "U", shape=(3, 2, 2, 2), chunks=(3, 2, 2, 2), dtype=np.float32, compressor=None)
    np.testing.assert_array_equal(again[...], 7.0)
    with pytest.raises(TypeError):
        require_zarr_array(root, "U", shape=(3, 1, 1, 1), dtype=np.float32)
    fresh = require_zarr_array(root, "U", shape=(3, 2, 2, 2), chunks=(3, 2, 2, 2), dtype=np.float32, compressor=None, overwrite=True)
    assert float(fresh[0, 0, 0, 0]) == 0.0


def test_eigenanalysis_writes_ome_and_eigen_arrays(tmp_path):
    """_finalize_structure_tensor_torch creates 3 OME vector groups, confidence and eigen* arrays."""
    from vesuvius.structure_tensor.create_st import _finalize_structure_tensor_torch

    Z, Y, X = 2, 4, 4
    st = np.zeros((6, Z, Y, X), dtype=np.float32)
    st[0] = 3.0  # Jzz
    st[3] = 2.0  # Jyy
    st[5] = 1.0  # Jxx
    zarr_path = str(tmp_path / "st.zarr")
    root = open_zarr_group(zarr_path, mode="w")
    create_zarr_array(root, "structure_tensor", data=st, chunks=(6, Z, Y, X), dtype="f4")

    _finalize_structure_tensor_torch(
        zarr_path=zarr_path,
        chunk_size=None,
        num_workers=0,
        compressor=Blosc(cname="zstd", clevel=1),
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
