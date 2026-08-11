"""Tests for the CT-support audit (models/run/audit_ct_support.py).

Everything runs against synthetic local zarrs with hand-placed data, so the
expected chunk classifications and voxel counts are exact and no network is
touched.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from vesuvius.models.run.audit_ct_support import (  # noqa: E402
    _parse_chunk_key,
    audit_chunks,
    audit_voxels,
    classify_chunks,
    stored_chunk_coords,
)


def _write(path, shape, chunks, fill):
    """Create a uint8 zarr array at ``path`` and let ``fill`` populate it."""
    arr = zarr.open_array(str(path), mode="w", shape=shape, chunks=chunks, dtype="u1")
    fill(arr)
    return arr


@pytest.fixture
def volumes(tmp_path):
    """Predictions with a known layout against a CT block of the same grid.

    Grid: 4x4x4 chunks of 16 voxels. CT holds data in chunk (1,1,1) only.
    Predictions: (1,1,1) supported, (0,1,1) and (2,2,2) touch it diagonally or
    face-on within one chunk, and (3,3,3) is two chunks away -- beyond any
    one-chunk blend margin.
    """
    shape, chunks = (64, 64, 64), (16, 16, 16)
    ct_path, pred_path = tmp_path / "ct.zarr", tmp_path / "preds.zarr"

    def fill_ct(a):
        a[16:32, 16:32, 16:32] = 200

    def fill_preds(a):
        for z, y, x in [(1, 1, 1), (0, 1, 1), (2, 2, 2), (3, 3, 3)]:
            a[z * 16:z * 16 + 16, y * 16:y * 16 + 16, x * 16:x * 16 + 16] = 255

    _write(ct_path, shape, chunks, fill_ct)
    _write(pred_path, shape, chunks, fill_preds)
    return str(pred_path), str(ct_path)


def test_parse_chunk_key_layouts():
    assert _parse_chunk_key("/1/2/3", 3) == (1, 2, 3)      # v2 nested
    assert _parse_chunk_key("1.2.3", 3) == (1, 2, 3)        # v2 flat
    assert _parse_chunk_key("c/1/2/3", 3) == (1, 2, 3)      # v3
    assert _parse_chunk_key("/.zarray", 3) is None
    assert _parse_chunk_key("/zarr.json", 3) is None
    assert _parse_chunk_key("/1/2", 3) is None              # wrong rank


def test_stored_chunk_coords_lists_only_written_chunks(volumes):
    pred_path, ct_path = volumes
    assert stored_chunk_coords(ct_path, 3) == {(1, 1, 1)}
    assert stored_chunk_coords(pred_path, 3) == {(1, 1, 1), (0, 1, 1), (2, 2, 2), (3, 3, 3)}


def test_audit_chunks_buckets_by_distance(volumes):
    pred_path, ct_path = volumes
    report = audit_chunks(pred_path, ct_path)

    assert report["prediction_chunks"] == 4
    assert report["supported"] == 1              # (1,1,1)
    assert report["halo_within_1_chunk"] == 2    # (0,1,1) face-on, (2,2,2) diagonal
    assert report["beyond_blend_margin"] == 1    # (3,3,3)
    assert report["frac_supported"] == pytest.approx(0.25)
    assert report["beyond_examples"][0]["chunk"] == [3, 3, 3]
    assert report["mode"] == "chunks"


def test_classify_chunks_handles_differing_chunk_shapes():
    """Predictions on 192^3 chunks against CT on 128^3 -- the published layout."""
    shape = (768, 768, 768)
    pred_chunks, ct_chunks = (192, 192, 192), (128, 128, 128)
    # CT voxels 256..383 in every axis -> CT chunk index (2,2,2).
    ct_coords = {(2, 2, 2)}

    # Prediction chunk (1,1,1) covers 192..383: overlaps the CT box.
    overlapping = classify_chunks([(1, 1, 1)], ct_coords, shape, pred_chunks, ct_chunks)
    assert overlapping["supported"] == 1

    # (0,0,0) covers 0..191, gap to 256 is 65 voxels: inside one 192 chunk.
    near = classify_chunks([(0, 0, 0)], ct_coords, shape, pred_chunks, ct_chunks)
    assert near["halo_within_1_chunk"] == 1

    # (3,3,3) covers 576..767, gap to 383 is 193 voxels: just beyond the margin.
    far = classify_chunks([(3, 3, 3)], ct_coords, shape, pred_chunks, ct_chunks)
    assert far["beyond_blend_margin"] == 1


def test_classify_chunks_empty_ct_is_all_beyond():
    report = classify_chunks([(0, 0, 0), (1, 1, 1)], set(), (64, 64, 64),
                             (16, 16, 16), (16, 16, 16))
    assert report["beyond_blend_margin"] == 2
    assert report["frac_supported"] == 0.0
    assert report["beyond_examples"][0]["gap_voxels"] is None


def test_audit_voxels_counts_phantoms_exactly(tmp_path):
    """One plane of predictions, half of it over CT data, half over zero."""
    shape, chunks = (8, 8, 8), (4, 4, 4)
    ct_path, pred_path = tmp_path / "ct.zarr", tmp_path / "preds.zarr"
    _write(ct_path, shape, chunks, lambda a: a.__setitem__((slice(0, 8), slice(0, 4)), 100))
    _write(pred_path, shape, chunks, lambda a: a.__setitem__((slice(0, 8),), 255))

    report = audit_voxels(str(pred_path), str(ct_path), threshold=127,
                          slab_stride=1, progress=False)

    assert report["planes_measured"] == 8
    assert report["positives"] == 8 * 8 * 8
    assert report["phantom"] == 8 * 4 * 8          # the half with CT == 0
    assert report["phantom_frac"] == pytest.approx(0.5)
    assert report["support_frac"] == pytest.approx(0.5)
    assert all(r["phantom_frac"] == pytest.approx(0.5) for r in report["per_plane"])


def test_audit_voxels_threshold_is_exclusive(tmp_path):
    """Values at exactly the threshold are background, one above it foreground."""
    shape, chunks = (4, 4, 4), (4, 4, 4)
    ct_path, pred_path = tmp_path / "ct.zarr", tmp_path / "preds.zarr"
    _write(ct_path, shape, chunks, lambda a: None)                       # CT all zero
    _write(pred_path, shape, chunks, lambda a: a.__setitem__(Ellipsis, 127))

    at_threshold = audit_voxels(str(pred_path), str(ct_path), threshold=127,
                                slab_stride=1, progress=False)
    assert at_threshold["positives"] == 0
    assert at_threshold["phantom_frac"] == 0.0

    below = audit_voxels(str(pred_path), str(ct_path), threshold=126,
                         slab_stride=1, progress=False)
    assert below["positives"] == 4 * 4 * 4
    assert below["phantom_frac"] == pytest.approx(1.0)


def test_slab_stride_visits_every_nth_slab(tmp_path):
    shape, chunks = (32, 4, 4), (4, 4, 4)          # 8 slabs of 4 planes
    ct_path, pred_path = tmp_path / "ct.zarr", tmp_path / "preds.zarr"
    _write(ct_path, shape, chunks, lambda a: a.__setitem__(Ellipsis, 100))
    _write(pred_path, shape, chunks, lambda a: a.__setitem__(Ellipsis, 255))

    every = audit_voxels(str(pred_path), str(ct_path), slab_stride=1, progress=False)
    strided = audit_voxels(str(pred_path), str(ct_path), slab_stride=2, progress=False)

    assert every["planes_measured"] == 32
    assert strided["planes_measured"] == 16
    assert [r["z"] for r in strided["per_plane"]][:5] == [0, 1, 2, 3, 8]


def test_grid_mismatch_is_reported(tmp_path):
    ct_path, pred_path = tmp_path / "ct.zarr", tmp_path / "preds.zarr"
    _write(ct_path, (8, 8, 8), (4, 4, 4), lambda a: None)
    _write(pred_path, (8, 8, 16), (4, 4, 4), lambda a: None)

    with pytest.raises(ValueError, match="grid mismatch"):
        audit_chunks(str(pred_path), str(ct_path))
    with pytest.raises(ValueError, match="grid mismatch"):
        audit_voxels(str(pred_path), str(ct_path), progress=False)


def test_report_totals_are_self_consistent(volumes):
    pred_path, ct_path = volumes
    report = audit_chunks(pred_path, ct_path)
    assert (report["supported"] + report["halo_within_1_chunk"]
            + report["beyond_blend_margin"] == report["prediction_chunks"])
    assert report["frac_supported"] + report["frac_halo_within_1_chunk"] \
        + report["frac_beyond_blend_margin"] == pytest.approx(1.0)
