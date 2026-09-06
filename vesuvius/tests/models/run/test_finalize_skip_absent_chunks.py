"""finalize_logits must not iterate chunk positions that have no input data.

A merged logits store is shaped like the whole scroll volume but only holds
chunks where inference ran (a --bbox run leaves everything else absent). On
main, finalising 27 real 128^3 chunks of a PHerc1447 run walked all 827,640
chunk positions of the (4, 24297, 8343, 8343) store: a 1/200 Z-slab holding
zero real chunks took 39 minutes on its own. The chunk-occupancy bitmap
answers "is there any input chunk under this output chunk" from the store
listing alone, so the work is restricted to those positions.
"""

from __future__ import annotations

import numpy as np
import zarr

from vesuvius.models.run import finalize_outputs
from vesuvius.models.run.finalize_outputs import _drop_absent_chunks, finalize_logits


def _chunk_infos(shape, chunks):
    """Same list finalize_logits builds internally (spatial chunk indices only)."""
    from itertools import product

    counts = [int(np.ceil(s / c)) for s, c in zip(shape[1:], chunks[1:])]
    return [{"indices": idx} for idx in product(*[range(n) for n in counts])]


def _write_sparse_logits(path, shape=(2, 64, 64, 64), chunks=(1, 16, 16, 16)):
    """v2 store shaped like a whole volume with data in one 16^3 corner region only."""
    kwargs = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
    arr = zarr.open(str(path), mode="w", shape=shape, chunks=chunks, dtype="f4", fill_value=0, **kwargs)
    block = np.zeros((2, 16, 16, 16), dtype=np.float32)
    block[1, :8] = 20.0  # foreground (p ~ 1.0) in the first 8 layers
    block[0, 8:] = 20.0
    arr[:, 16:32, 32:48, 0:16] = block
    return arr


def test_drop_absent_chunks_keeps_only_overlapping_output_chunks(tmp_path):
    arr = _write_sparse_logits(tmp_path / "merged.zarr")
    infos = _chunk_infos(arr.shape, (1, 16, 16, 16))
    kept = _drop_absent_chunks(
        infos, input_path=str(tmp_path / "merged.zarr"), input_shape=arr.shape,
        input_chunks=arr.chunks, output_chunks=(1, 16, 16, 16), verbose=False,
    )
    assert len(infos) == 64 and len(kept) == 1
    assert tuple(kept[0]["indices"]) == (1, 2, 0)


def test_drop_absent_chunks_with_coarser_output_chunks_keeps_any_overlap(tmp_path):
    arr = _write_sparse_logits(tmp_path / "merged.zarr")
    infos = _chunk_infos(arr.shape, (1, 32, 32, 32))
    kept = _drop_absent_chunks(
        infos, input_path=str(tmp_path / "merged.zarr"), input_shape=arr.shape,
        input_chunks=arr.chunks, output_chunks=(1, 32, 32, 32), verbose=False,
    )
    assert [tuple(k["indices"]) for k in kept] == [(0, 1, 0)]


def test_drop_absent_chunks_falls_back_when_occupancy_unavailable(tmp_path, monkeypatch):
    arr = _write_sparse_logits(tmp_path / "merged.zarr")
    infos = _chunk_infos(arr.shape, (1, 16, 16, 16))
    import vesuvius.data.zarr_chunk_index as zci

    monkeypatch.setattr(zci, "build_chunk_occupancy", lambda *a, **k: None)
    assert _drop_absent_chunks(
        infos, input_path=str(tmp_path / "merged.zarr"), input_shape=arr.shape,
        input_chunks=arr.chunks, output_chunks=(1, 16, 16, 16), verbose=False,
    ) is infos
    monkeypatch.setattr(zci, "build_chunk_occupancy", lambda *a, **k: (_ for _ in ()).throw(OSError("listing failed")))
    assert len(_drop_absent_chunks(
        infos, input_path=str(tmp_path / "merged.zarr"), input_shape=arr.shape,
        input_chunks=arr.chunks, output_chunks=(1, 16, 16, 16), verbose=False,
    )) == 64


def test_finalize_output_is_identical_with_and_without_skip(tmp_path, monkeypatch):
    src = tmp_path / "merged.zarr"
    _write_sparse_logits(src)
    finalize_logits(str(src), str(tmp_path / "skip.zarr"), mode="binary", threshold=None,
                    num_workers=1, verbose=False)
    monkeypatch.setattr(finalize_outputs, "_drop_absent_chunks", lambda infos, **kw: infos)
    finalize_logits(str(src), str(tmp_path / "full.zarr"), mode="binary", threshold=None,
                    num_workers=1, verbose=False)
    a = zarr.open(str(tmp_path / "skip.zarr"), mode="r")[:]
    b = zarr.open(str(tmp_path / "full.zarr"), mode="r")[:]
    np.testing.assert_array_equal(a, b)
    assert a[0, 16:24, 32:48, 0:16].min() == 255 and a[0, 24:32, 32:48, 0:16].max() == 0
