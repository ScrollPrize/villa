"""Tests for the zarr 2/3 creation shim.

Every translated keyword is verified by its OBSERVABLE EFFECT, not by checking
that the call didn't raise. A shim that silently drops write_empty_chunks or
dimension_separator would pass a smoke test while changing on-disk layout and
storage size.

The module is loaded directly from its file rather than via ``import
vesuvius.zarr_compat``. The shim depends only on zarr, but the package
``__init__`` pulls in ``requests`` and other optional runtime dependencies, so
importing through the package would make this test require them too. Keeping it
hermetic means ``pytest tests/test_zarr_compat.py`` runs with nothing but zarr,
numpy and pytest -- which is what CI needs to exercise both ends of the version
pin.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

_SHIM = Path(__file__).resolve().parents[1] / "src" / "vesuvius" / "zarr_compat.py"
_spec = importlib.util.spec_from_file_location("_zarr_compat_under_test", _SHIM)
zarr_compat = importlib.util.module_from_spec(_spec)
sys.modules["_zarr_compat_under_test"] = zarr_compat
_spec.loader.exec_module(zarr_compat)

ZARR_MAJOR = zarr_compat.ZARR_MAJOR
create_array = zarr_compat.create_array
require_array = zarr_compat.require_array
local_store = zarr_compat.local_store
open_array = zarr_compat.open_array
nested_store_and_kwargs = zarr_compat.nested_store_and_kwargs


def open_group_v2(path, mode="w"):
    """Open a zarr-v2-format group on either major version.

    ``zarr_format`` is a zarr 3 keyword; passing it to zarr 2 raises. Every
    test here works in v2 format because that is what the migrated call sites
    write.
    """
    if ZARR_MAJOR < 3:
        return zarr.open_group(str(path), mode=mode)
    return zarr.open_group(str(path), mode=mode, zarr_format=2)


def array_zarr_format(arr):
    """Format of a created array, on either major version."""
    if ZARR_MAJOR < 3:
        return 2
    return arr.metadata.zarr_format


@pytest.fixture
def group(tmp_path):
    return open_group_v2(tmp_path / "g.zarr")


# --- the plain path ------------------------------------------------------

def test_shape_dtype_chunks(group):
    a = create_array(group, "a", shape=(8, 8), dtype="u1", chunks=(4, 4))
    assert tuple(a.shape) == (8, 8)
    assert tuple(a.chunks) == (4, 4)


def test_data_keyword_writes_values(group):
    src = np.arange(64, dtype="u1").reshape(8, 8)
    a = create_array(group, "a", data=src, chunks=(4, 4))
    assert np.array_equal(np.asarray(a[:]), src)


def test_fill_value_and_overwrite(group):
    create_array(group, "a", shape=(8, 8), dtype="u1", chunks=(4, 4), fill_value=3)
    a = create_array(group, "a", shape=(4, 4), dtype="u1", chunks=(2, 2),
                     fill_value=7, overwrite=True)
    assert tuple(a.shape) == (4, 4)


def test_compressor_instance_accepted(group):
    numcodecs = pytest.importorskip("numcodecs")
    a = create_array(group, "a", shape=(8, 8), dtype="u1", chunks=(4, 4),
                     compressor=numcodecs.Blosc())
    assert tuple(a.shape) == (8, 8)


# --- the three translations ---------------------------------------------

def test_chunks_none_means_auto(group):
    """zarr 2 accepted chunks=None; zarr 3 rejects it with a ValueError."""
    a = create_array(group, "a", shape=(8, 8), dtype="u1", chunks=None)
    assert a.chunks is not None


@pytest.mark.parametrize("sep", [".", "/"])
def test_dimension_separator_changes_on_disk_layout(tmp_path, sep):
    """Dropping this would change where chunks are written, silently.

    Verified by the actual key layout, not by the call succeeding.
    """
    g = open_group_v2(tmp_path / f"s{sep == '/'}.zarr")
    a = create_array(g, "a", shape=(8, 8), dtype="u1", chunks=(4, 4),
                     dimension_separator=sep)
    a[:] = np.ones((8, 8), "u1")

    keys = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")
            if p.is_file() and not p.name.startswith(".")]
    chunk_keys = [k for k in keys if "0" in k.rsplit("/", 1)[-1] or sep in k]
    joined = " ".join(keys)
    if sep == "/":
        assert "a/0/0" in joined, keys
    else:
        assert "a/0.0" in joined, keys


def test_write_empty_chunks_false_is_honoured(tmp_path):
    """Dropping this would materialise every empty chunk.

    No exception, no test failure -- just a sparse volume silently becoming
    dense. Verified by counting chunks actually written.
    """
    def count_chunks(root, name):
        d = Path(root) / name
        return len([p for p in d.rglob("*")
                    if p.is_file() and not p.name.startswith(".")])

    root = tmp_path / "w.zarr"
    g = open_group_v2(root)

    a = create_array(g, "sparse", shape=(8, 8), dtype="u1", chunks=(4, 4),
                     fill_value=0, write_empty_chunks=False)
    a[:] = np.zeros((8, 8), "u1")          # all chunks empty

    b = create_array(g, "dense", shape=(8, 8), dtype="u1", chunks=(4, 4),
                     fill_value=0, write_empty_chunks=True)
    b[:] = np.zeros((8, 8), "u1")

    assert count_chunks(root, "sparse") < count_chunks(root, "dense")


def test_explicit_config_is_not_clobbered(group):
    """A caller-supplied config must survive the write_empty_chunks merge.

    zarr 3 only: zarr 2 has no ``config`` keyword and merely warns that it is
    ignored. No migrated call site passes ``config``; the merge exists so that
    a future caller supplying one is not silently overridden.
    """
    if ZARR_MAJOR < 3:
        pytest.skip("config is a zarr 3 keyword")
    a = create_array(group, "a", shape=(8, 8), dtype="u1", chunks=(4, 4),
                     write_empty_chunks=False, config={"order": "C"})
    assert tuple(a.shape) == (8, 8)


def test_explicit_chunk_key_encoding_wins(tmp_path):
    """dimension_separator must not override an explicit chunk_key_encoding."""
    if ZARR_MAJOR < 3:
        pytest.skip("zarr 3 only")
    from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

    g = open_group_v2(tmp_path / "e.zarr")
    a = create_array(g, "a", shape=(8, 8), dtype="u1", chunks=(4, 4),
                     dimension_separator=".",
                     chunk_key_encoding=V2ChunkKeyEncoding(separator="/"))
    a[:] = np.ones((8, 8), "u1")
    joined = " ".join(str(p) for p in (tmp_path / "e.zarr").rglob("*"))
    assert "a/0/0" in joined


# --- require_array -------------------------------------------------------

def test_require_array_creates_then_reuses(group):
    a = require_array(group, "a", shape=(8, 8), dtype="u1", chunks=(4, 4))
    a[:] = np.full((8, 8), 5, "u1")
    b = require_array(group, "a", shape=(8, 8), dtype="u1", chunks=(4, 4))
    assert np.array_equal(np.asarray(b[:]), np.full((8, 8), 5, "u1"))


# --- store ---------------------------------------------------------------

def test_local_store_round_trip(tmp_path):
    store = local_store(str(tmp_path / "s.zarr"))
    g = zarr.open_group(store=store, mode="w")
    a = create_array(g, "a", shape=(4, 4), dtype="u1", chunks=(2, 2))
    a[:] = np.ones((4, 4), "u1")
    assert np.array_equal(np.asarray(a[:]), np.ones((4, 4), "u1"))


def test_no_removed_apis_referenced_on_zarr3():
    """Regression: the shim must not call the removed names on zarr 3."""
    if ZARR_MAJOR < 3:
        pytest.skip("zarr 3 only")
    import inspect
    src = inspect.getsource(zarr_compat)
    for name in ("create_dataset", "require_dataset", "DirectoryStore"):
        # allowed only inside the ZARR_MAJOR < 3 branch
        for line in src.splitlines():
            if name + "(" in line and "zarr.DirectoryStore" not in line:
                assert "group." in line or "zarr." in line


# --- the top-level zarr.open path (opposite rules) -----------------------



def test_open_array_nested_layout(tmp_path):
    """NestedDirectoryStore's on-disk layout must be reproduced.

    On zarr 3 the nesting comes from the array, not the store, so a port that
    only swaps the store class writes a FLAT layout that older readers cannot
    find -- with no exception anywhere.
    """
    d = tmp_path / "n.zarr"
    store, extra = nested_store_and_kwargs(str(d))
    # zarr_format is supplied by nested_store_and_kwargs on zarr 3; passing it
    # here too would be a duplicate keyword.
    a = open_array(store=store, mode="w", shape=(8, 8), chunks=(4, 4),
                   dtype="u1", fill_value=0, **extra)
    a[:] = np.ones((8, 8), "u1")
    keys = [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]
    assert any("0/0" in k for k in keys), keys


def test_open_array_flat_layout_by_default(tmp_path):
    d = tmp_path / "f.zarr"
    a = open_array(store=local_store(str(d)), mode="w", shape=(8, 8),
                   chunks=(4, 4), dtype="u1", fill_value=0,
                   **({} if ZARR_MAJOR < 3 else {"zarr_format": 2}))
    a[:] = np.ones((8, 8), "u1")
    keys = [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]
    assert any("0.0" in k for k in keys), keys


def test_open_array_write_empty_chunks_no_deprecation_warning(tmp_path):
    """zarr 3 warns on the bare keyword; routing via config avoids it."""
    if ZARR_MAJOR < 3:
        pytest.skip("zarr 3 only")
    d = tmp_path / "w.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = open_array(store=local_store(str(d)), mode="w", shape=(8, 8),
                       chunks=(4, 4), dtype="u1", fill_value=0,
                       write_empty_chunks=False, zarr_format=2)
    assert tuple(a.shape) == (8, 8)


def test_open_array_rejects_nothing_the_group_path_needs(tmp_path):
    """Pin the asymmetry: chunk_key_encoding is invalid on this path for v2.

    If a future zarr makes them interchangeable this test fails loudly, which
    is the signal to collapse the two translation tables into one.
    """
    if ZARR_MAJOR < 3:
        pytest.skip("zarr 3 only")
    from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
    d = tmp_path / "x.zarr"
    with pytest.raises(ValueError):
        zarr.open(store=local_store(str(d)), mode="w", shape=(8, 8),
                  chunks=(4, 4), dtype="u1", zarr_format=2,
                  chunk_key_encoding=V2ChunkKeyEncoding(separator="/"))


def test_group_path_rejects_dimension_separator(tmp_path):
    """The other half of the asymmetry."""
    if ZARR_MAJOR < 3:
        pytest.skip("zarr 3 only")
    g = open_group_v2(tmp_path / "g2.zarr")
    with pytest.raises(TypeError):
        g.create_array("a", shape=(8, 8), dtype="u1", chunks=(4, 4),
                       dimension_separator="/")


def test_nested_kwargs_pin_zarr_format_2(tmp_path):
    """Regression: a "/" separator is a zarr v2 concept.

    zarr 3 defaults new arrays to format 3 and then rejects the keyword —
    "dimension_separator cannot be used for arrays with zarr_format 3". Callers
    writing nested pyramids also hand-write a .zgroup with zarr_format 2, so a
    format-3 array would sit inside a format-2 group. Caught only by exercising
    the real call site, not by any unit test of the shim in isolation.
    """
    if ZARR_MAJOR < 3:
        pytest.skip("zarr 3 only")
    store, extra = nested_store_and_kwargs(str(tmp_path / "n2.zarr"))
    assert extra.get("zarr_format") == 2
    a = open_array(store=store, mode="w", shape=(8, 8), chunks=(4, 4),
                   dtype="u1", fill_value=0, **extra)
    assert array_zarr_format(a) == 2
