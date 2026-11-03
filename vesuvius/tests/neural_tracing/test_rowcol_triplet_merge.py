import numpy as np
import pytest

from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import (
    _WeightedDenseDisplacementMerger,
    _build_triplet_direction_priors_for_crop,
)


def _reference_direction_priors(
    crop_size,
    cond_vox,
    local_zyx,
    local_normals,
    local_normals_valid,
    fallback_unit_normal,
    mask_mode="cond",
):
    """Dense implementation the sparse version must match bit for bit."""
    cond = np.asarray(cond_vox, dtype=np.float32)
    priors_zyx = np.zeros(crop_size + (3,), dtype=np.float32)
    counts = np.zeros(crop_size, dtype=np.uint32)
    local_arr = np.asarray(local_zyx, dtype=np.float32)
    normals_arr = np.asarray(local_normals, dtype=np.float32)
    normals_valid = np.asarray(local_normals_valid, dtype=bool)
    finite = normals_valid & np.isfinite(local_arr).all(axis=1) & np.isfinite(normals_arr).all(axis=1)
    if bool(finite.any()):
        ijk = np.rint(local_arr[finite]).astype(np.int64, copy=False)
        in_bounds = np.all((ijk >= 0) & (ijk < np.asarray(crop_size)), axis=1)
        if bool(in_bounds.any()):
            ijk = ijk[in_bounds]
            n = normals_arr[finite][in_bounds]
            for axis in range(3):
                np.add.at(priors_zyx[..., axis], (ijk[:, 0], ijk[:, 1], ijk[:, 2]), n[:, axis])
            np.add.at(counts, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)

    have_prior = counts > 0
    if bool(have_prior.any()):
        priors_zyx[have_prior] /= counts[have_prior, None].astype(np.float32, copy=False)
        norms = np.linalg.norm(priors_zyx, axis=3)
        finite = np.isfinite(priors_zyx).all(axis=3) & np.isfinite(norms) & (norms > 1e-6)
        have_prior &= finite
        priors_zyx[have_prior] /= norms[have_prior, None].astype(np.float32, copy=False)

    fallback = np.asarray(fallback_unit_normal, dtype=np.float32).reshape(3)
    fill_mask = (cond > 0.5) & (~have_prior)
    if bool(fill_mask.any()):
        priors_zyx[fill_mask] = fallback
        have_prior[fill_mask] = True

    if mask_mode == "full":
        if bool(have_prior.any()):
            n = np.mean(priors_zyx[have_prior], axis=0, dtype=np.float64).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(n))
            n = n / norm if np.isfinite(norm) and norm > 1e-6 else fallback
        else:
            n = fallback
        priors_zyx[:, :, :] = n
    else:
        priors_zyx[cond <= 0.5] = 0.0

    priors = np.zeros((6, *crop_size), dtype=np.float32)
    for axis in range(3):
        priors[axis, ...] = priors_zyx[..., axis]
        priors[axis + 3, ...] = -priors_zyx[..., axis]
    return priors


def _random_prior_case(seed, crop_size=(12, 20, 18)):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(0, 300))
    zyx = rng.uniform(-3.0, np.asarray(crop_size) + 3.0, size=(m, 3)).astype(np.float32)
    normals = rng.normal(size=(m, 3)).astype(np.float32)
    if m:
        zyx[: m // 4] = zyx[0]  # several points per voxel
        normals[: m // 8] = 0.0  # zero-length mean at that voxel
        normals[rng.integers(0, m, 3)] = np.nan
        zyx[rng.integers(0, m, 2)] = np.inf
    cond = (rng.uniform(size=crop_size) > 0.6).astype(np.float32)
    cond[rng.uniform(size=crop_size) > 0.98] = np.nan
    fallback = rng.normal(size=3).astype(np.float32)
    fallback /= np.linalg.norm(fallback)
    valid = rng.uniform(size=m) > 0.1
    return dict(
        crop_size=crop_size,
        cond_vox=cond,
        local_zyx=zyx,
        local_normals=normals,
        local_normals_valid=valid,
        fallback_unit_normal=fallback,
    )


@pytest.mark.parametrize("mask_mode", ["cond", "full"])
@pytest.mark.parametrize("seed", range(12))
def test_direction_priors_match_dense_reference_bitwise(seed, mask_mode):
    case = _random_prior_case(seed)
    expected = _reference_direction_priors(**case, mask_mode=mask_mode)
    got = _build_triplet_direction_priors_for_crop(**case, mask_mode=mask_mode)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_async_accumulate_matches_inline_bitwise(tmp_path):
    rng = np.random.default_rng(0)
    window_shape, crop = (40, 56, 48), (16, 24, 24)
    corners = [tuple(int(rng.integers(0, w - c + 1)) for w, c in zip(window_shape, crop)) for _ in range(8)]
    batches = [rng.normal(size=(1, 6, *crop)).astype(np.float32) for _ in corners]

    reads = []
    for use_async in (False, True):
        with _WeightedDenseDisplacementMerger((0, 0, 0), window_shape, crop, temp_dir=tmp_path, chunk_size=16) as merger:
            for disp, corner in zip(batches, corners):
                items = [{"min_corner": corner}]
                if use_async:
                    merger.accumulate_batch_async(disp, items)
                else:
                    merger.accumulate_batch(disp, items)
            reads.append([merger.read_crop(c) for c in corners])
            assert merger.crop_count == len(corners)

    for inline, overlapped in zip(*reads):
        assert np.array_equal(inline.view(np.uint32), overlapped.view(np.uint32))


def test_prefetch_batch_items_matches_inline_gather(monkeypatch):
    import vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps as M

    calls = []

    def fake_gather(batch_records, **kwargs):
        calls.append(tuple(batch_records))
        return [r * 10 for r in batch_records]

    monkeypatch.setattr(M, "_gather_batch_items", fake_gather)
    records = list(range(7))
    batches = list(M._iter_bbox_batches(records, 3))
    got = list(M._prefetch_batch_items(iter(batches), {"crop_size": (1, 1, 1)}))
    assert got == [[0, 10, 20], [30, 40, 50], [60]]
    assert calls == [(0, 1, 2), (3, 4, 5), (6,)]
    assert list(M._prefetch_batch_items(iter([]), {})) == []


def test_prefetch_batch_items_propagates_errors(monkeypatch):
    import vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps as M

    def failing_gather(batch_records, **kwargs):
        if batch_records == [1]:
            raise RuntimeError("chunk read failed")
        return list(batch_records)

    monkeypatch.setattr(M, "_gather_batch_items", failing_gather)
    gen = M._prefetch_batch_items(iter([(0, [0]), (1, [1]), (2, [2])]), {})
    assert next(gen) == [0]
    with pytest.raises(RuntimeError, match="chunk read failed"):
        next(gen)
