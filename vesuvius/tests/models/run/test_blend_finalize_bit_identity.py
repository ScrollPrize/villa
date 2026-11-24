"""The faster blending and finalization paths give the results the plain ones gave.

Blending reads a patch once per batch of neighbouring chunks instead of once per chunk,
the spatial index is built and queried with NumPy, and finalization tests for constant
chunks and computes the two-channel softmax with fewer temporaries. None of that may
move a value: every test compares against the straightforward computation, bit for bit.
"""
from collections import defaultdict

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter, open_zarr

from vesuvius.models.run import blending, finalize_outputs
from vesuvius.models.run.blending import (
    SpatialPatchGrid,
    _PatchCache,
    calculate_chunks,
    generate_gaussian_map,
    group_chunk_work_items,
    merge_inference_outputs,
)
from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization, finalize_logits


# --- Spatial index ---------------------------------------------------------------------------

def reference_grid(coords_cache, non_empty_cache, patch_size, grid_size):
    """The per-patch loop that built the grid before it was vectorized."""
    grid, stats = defaultdict(list), {'total_patches': 0, 'skipped_empty': 0}
    pZ, pY, pX = patch_size
    for part_id, coords in coords_cache.items():
        mask = non_empty_cache.get(part_id)
        for patch_idx in range(len(coords)):
            if mask is not None and not mask[patch_idx]:
                stats['skipped_empty'] += 1
                continue
            z, y, x = coords[patch_idx].tolist()
            for gz in range(z // grid_size, (z + pZ) // grid_size + 1):
                for gy in range(y // grid_size, (y + pY) // grid_size + 1):
                    for gx in range(x // grid_size, (x + pX) // grid_size + 1):
                        grid[(gz, gy, gx)].append((part_id, patch_idx))
            stats['total_patches'] += 1
    return grid, stats


def reference_chunk_patches(grid, coords_cache, patch_size, grid_size, chunk):
    """The entry-by-entry lookup that `get_chunk_patches` replaced."""
    pZ, pY, pX = patch_size
    seen, result = set(), defaultdict(list)
    for gz in range(chunk['z_start'] // grid_size, (chunk['z_end'] - 1) // grid_size + 1):
        for gy in range(chunk['y_start'] // grid_size, (chunk['y_end'] - 1) // grid_size + 1):
            for gx in range(chunk['x_start'] // grid_size, (chunk['x_end'] - 1) // grid_size + 1):
                for entry in grid.get((gz, gy, gx), []):
                    if entry in seen:
                        continue
                    seen.add(entry)
                    part_id, patch_idx = entry
                    z, y, x = coords_cache[part_id][patch_idx].tolist()
                    if (z + pZ <= chunk['z_start'] or z >= chunk['z_end'] or
                            y + pY <= chunk['y_start'] or y >= chunk['y_end'] or
                            x + pX <= chunk['x_start'] or x >= chunk['x_end']):
                        continue
                    result[part_id].append((patch_idx, z, y, x))
    return dict(result)


def random_index(seed, grid_size, coord_dtype=np.int32):
    rng = np.random.default_rng(seed)
    patch_size = (12, 10, 14)
    volume = (70, 64, 90)
    coords_cache, non_empty_cache = {}, {}
    for part_id in (3, 0, 7):  # deliberately not sorted: insertion order is part of the contract
        n = int(rng.integers(20, 60))
        coords_cache[part_id] = np.stack(
            [rng.integers(-4, v - p // 2, size=n) for v, p in zip(volume, patch_size)], axis=1).astype(coord_dtype)
        non_empty_cache[part_id] = rng.random(n) < 0.7
    del non_empty_cache[7]  # a part without a mask keeps all its patches
    index = SpatialPatchGrid(patch_size=patch_size, grid_size=grid_size)
    index.coords_cache, index.non_empty_cache = coords_cache, non_empty_cache
    index._build_grid(patch_size)
    return index, volume


@pytest.mark.parametrize("grid_size", [1000, 16, 5])
@pytest.mark.parametrize("coord_dtype", [np.int32, np.int64])
def test_spatial_grid_matches_per_patch_loop(grid_size, coord_dtype):
    index, volume = random_index(0, grid_size, coord_dtype)
    grid, stats = reference_grid(index.coords_cache, index.non_empty_cache, index.patch_size, grid_size)

    assert list(index.grid.items()) == list(grid.items())  # same cells, same order, same entries
    assert all(type(v) is int for cell in index.grid for v in cell)
    assert all(type(e[1]) is int for entries in index.grid.values() for e in entries)
    assert index.stats['total_patches'] == stats['total_patches']
    assert index.stats['skipped_empty'] == stats['skipped_empty']

    for chunk in calculate_chunks(volume, (16, 24, 20)) + calculate_chunks(volume, (70, 64, 90)):
        got = index.get_chunk_patches(chunk)
        want = reference_chunk_patches(grid, index.coords_cache, index.patch_size, grid_size, chunk)
        assert list(got.items()) == list(want.items())
        assert all(type(v) is type(w) for part in got for a, b in zip(got[part], want[part]) for v, w in zip(a, b))


def test_chunk_lookup_follows_a_grid_edited_by_hand():
    index = SpatialPatchGrid(patch_size=(4, 4, 4), grid_size=1000)
    index.coords_cache[0] = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 100]], dtype=np.int32)
    index.grid[(0, 0, 0)] = [(0, 0), (0, 1), (0, 0)]  # a duplicate is reported once
    chunk = {'z_start': 0, 'z_end': 4, 'y_start': 0, 'y_end': 4, 'x_start': 0, 'x_end': 4}
    assert index.get_chunk_patches(chunk) == {0: [(0, 0, 0, 0), (1, 0, 0, 2)]}
    index.grid[(0, 0, 0)].append((0, 2))  # outside the chunk
    assert index.get_chunk_patches(chunk) == {0: [(0, 0, 0, 0), (1, 0, 0, 2)]}
    index.grid[(0, 0, 0)] = [(0, 0), (0, 2)]
    assert index.get_chunk_patches(chunk) == {0: [(0, 0, 0, 0)]}
    # Coordinates that are not an integer array are looked up entry by entry, as before.
    index.coords_cache[0] = index.coords_cache[0].astype(np.float64)
    index.grid[(0, 0, 0)] = [(0, 1), (0, 2)]
    assert index.get_chunk_patches(chunk) == {0: [(1, 0.0, 0.0, 2.0)]}


# --- Batched blending ------------------------------------------------------------------------

def test_patch_cache_reads_once_and_frees_after_the_last_use():
    reads = []

    def reader(key):
        return lambda: reads.append(key) or np.zeros(10, np.uint8)

    cache = _PatchCache(max_bytes=20)
    for key, uses in (("a", 3), ("b", 2), ("c", 2), ("d", 1)):
        for _ in range(uses):
            cache.expect(key)

    assert cache.fetch("a", reader("a"))[1] and cache.fetch("b", reader("b"))[1]
    assert cache.nbytes == 20
    assert cache.fetch("c", reader("c"))[1] and "c" not in cache.patches  # over budget: not kept
    assert not cache.fetch("a", reader("a"))[1] and cache.nbytes == 20
    assert not cache.fetch("a", reader("a"))[1] and cache.nbytes == 10  # last use frees it
    assert cache.fetch("c", reader("c"))[1]  # read again, and it was its last use
    assert cache.fetch("d", reader("d"))[1] and "d" not in cache.patches  # single use: never kept
    cache.skip("b")
    assert cache.nbytes == 0 and not cache.patches and not cache.remaining
    assert reads == ["a", "b", "c", "c", "d"] and cache.hits == 2


def test_batches_cover_every_chunk_once_in_order():
    chunks = calculate_chunks((40, 40, 80), (8, 8, 8))
    items = [{'chunk_info': c, 'chunk_patches': {0: [(i, 0, 0, 0)]}} for i, c in enumerate(chunks)]
    for workers, largest in ((1, 16), (8, 8), (10 ** 6, 1)):
        batches = group_chunk_work_items(items, (8, 8, 8), workers)
        assert sorted((id(i) for b in batches for i in b)) == sorted(id(i) for i in items)
        assert max(len(b) for b in batches) == largest
        assert len(batches) >= min(len(items), 4 * workers) or largest == 16
        for batch in batches:
            order = [i['chunk_patches'][0][0][0] for i in batch]
            assert order == sorted(order)
    assert group_chunk_work_items([], (8, 8, 8), 4) == []


def write_parts(parent, volume, patch_size, num_classes, seed, dtype=np.float16):
    """Overlapping patches (50%) split over three parts, some of them all zero."""
    rng = np.random.default_rng(seed)
    starts = [list(range(0, v - p + 1, p // 2)) for v, p in zip(volume, patch_size)]
    coords = np.array([(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]], dtype=np.int32)
    rng.shuffle(coords)
    for part_id, part_coords in enumerate(np.array_split(coords, 3)):
        patches = rng.normal(0, 3, size=(len(part_coords), num_classes, *patch_size)).astype(dtype)
        patches[rng.random(len(part_coords)) < 0.2] = 0  # skipped by inference, not stored
        logits = ZarrWriter.create(f"{parent}/logits_part_{part_id}.zarr", shape=patches.shape,
                                   chunks=(1, *patches.shape[1:]), dtype=dtype,
                                   cname="zstd", level=1, skip_empty=True)
        logits[...] = patches
        logits.attrs.update(patch_size=list(patch_size), original_volume_shape=list(volume), overlap=0.5)
        positions = ZarrWriter.create(f"{parent}/coordinates_part_{part_id}.zarr", shape=part_coords.shape,
                                      chunks=part_coords.shape, dtype=np.int32)
        positions[...] = part_coords
    return coords


def blend_one_chunk_at_a_time(parent, output_path, volume, patch_size, num_classes, chunk_size, finalize_config):
    """What blending did before batching: every chunk on its own, every patch read per chunk."""
    blending.create_blend_output(output_path, volume, num_classes, patch_size, chunk_size,
                                 finalize_config=finalize_config, verbose=False)
    part_files = {p: {'logits': f"{parent}/logits_part_{p}.zarr",
                      'coordinates': f"{parent}/coordinates_part_{p}.zarr"} for p in range(3)}
    index = SpatialPatchGrid(patch_size=patch_size, grid_size=1000)
    chunks = calculate_chunks(volume, chunk_size or patch_size)
    index.build(parent, part_files, [0, 1, 2], patch_size, chunks=chunks)
    saved = dict(blending._worker_state)
    try:
        blending._init_worker(part_files, output_path, generate_gaussian_map(patch_size), patch_size,
                              num_classes, False, finalize_config)
        reads = 0
        for chunk in chunks:
            patches = index.get_chunk_patches(chunk)
            if patches:
                reads += blending.process_chunk(chunk, patches)['num_patch_reads']
    finally:
        blending._worker_state.clear()
        blending._worker_state.update(saved)
    return reads


@pytest.mark.parametrize("finalize_config,chunk_size", [
    (None, None),
    (None, (10, 12, 12)),  # chunks that do not line up with the patches
    (FinalizeConfig(mode="binary"), None),
    (FinalizeConfig(mode="binary", threshold=0.4), None),
    (FinalizeConfig(mode="multiclass"), (16, 8, 8)),
])
@pytest.mark.parametrize("patch_cache_mb", [256, 0])
def test_batched_blend_is_bit_identical(tmp_path, finalize_config, chunk_size, patch_cache_mb):
    parent = str(tmp_path / "parts")
    volume, patch_size = (24, 32, 40), (8, 8, 8)
    num_classes = 3 if finalize_config is not None and finalize_config.mode == "multiclass" else 2
    write_parts(parent, volume, patch_size, num_classes, seed=1)

    reference = str(tmp_path / "reference.zarr")
    reference_reads = blend_one_chunk_at_a_time(parent, reference, volume, patch_size, num_classes,
                                                chunk_size, finalize_config)
    batched = str(tmp_path / "batched.zarr")
    merge_inference_outputs(parent, batched, num_workers=2, verbose=False, chunk_size=chunk_size,
                            finalize_config=finalize_config, patch_cache_mb=patch_cache_mb)

    got, want = open_zarr(batched), open_zarr(reference)
    assert got.dtype == want.dtype and got.shape == want.shape
    values = got[...]
    assert np.array_equal(values, want[...])
    assert np.any(values != 0)
    assert reference_reads > 0


def test_batch_reads_shared_patches_once(tmp_path):
    parent = str(tmp_path / "parts")
    volume, patch_size = (16, 16, 48), (8, 8, 8)
    write_parts(parent, volume, patch_size, 2, seed=2)
    reference = str(tmp_path / "reference.zarr")
    per_chunk_reads = blend_one_chunk_at_a_time(parent, reference, volume, patch_size, 2, None, None)

    part_files = {p: {'logits': f"{parent}/logits_part_{p}.zarr",
                      'coordinates': f"{parent}/coordinates_part_{p}.zarr"} for p in range(3)}
    index = SpatialPatchGrid(patch_size=patch_size, grid_size=1000)
    chunks = calculate_chunks(volume, patch_size)
    index.build(parent, part_files, [0, 1, 2], patch_size, chunks=chunks)
    items = [{'chunk_info': c, 'chunk_patches': index.get_chunk_patches(c)} for c in chunks]
    distinct = {(part, patch[0]) for item in items for part, patches in item['chunk_patches'].items()
                for patch in patches}
    batched = str(tmp_path / "batched.zarr")
    blending.create_blend_output(batched, volume, 2, patch_size, verbose=False)
    saved = dict(blending._worker_state)
    try:
        blending._init_worker(part_files, batched, generate_gaussian_map(patch_size), patch_size, 2, False)
        result = blending.process_chunk_batch(items)  # the whole volume as one batch
    finally:
        blending._worker_state.clear()
        blending._worker_state.update(saved)

    assert result['chunks'] == len(items)
    assert result['num_patch_reads'] == len(distinct) < per_chunk_reads
    assert result['num_patch_reads'] + result['patch_cache_hits'] == per_chunk_reads
    assert np.array_equal(open_zarr(batched)[...], open_zarr(reference)[...])


# --- Finalization ----------------------------------------------------------------------------

def reference_finalization(logits_np, num_classes, config):
    """Binary / multiclass finalization as it was written before, for a single-task model."""
    if np.allclose(logits_np, logits_np.flat[0], rtol=1e-6):
        return None, True
    uncovered = np.all(logits_np == 0, axis=0) if logits_np.shape[0] >= 2 else None
    cutoff = None if config.threshold is None else float(np.log(config.threshold / (1.0 - config.threshold)))
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=0, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    if config.mode == "binary":
        if cutoff is not None:
            output = (logits_np[1] - logits_np[0] > cutoff).astype(np.float32)[np.newaxis]
        else:
            output = softmax[1:2]
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    else:
        argmax = np.argmax(logits_np, axis=0).astype(np.uint8)[np.newaxis]
        output = np.concatenate([np.clip(np.rint(softmax * 255.0), 0, 255).astype(np.uint8), argmax], axis=0)
    output[:, uncovered] = 0
    return output, False


def adversarial_logits(dtype, seed):
    """Two-channel logits around every place the two softmax formulations could part ways."""
    rng = np.random.default_rng(seed)
    info = np.finfo(dtype)
    n = 40_000
    blocks = [
        rng.normal(0, 4, size=(2, n)),
        rng.normal(0, 0.01, size=(2, n)),                       # near-ties
        rng.uniform(-100, 100, size=(2, n)),                    # saturated either way
        np.stack([rng.normal(0, 30, n), np.zeros(n)]),          # one channel exactly 0
        np.stack([np.full(n, 7.25), 7.25 + rng.normal(0, 1e-3, n)]),
    ]
    # Differences that put p * 255 right at an integer, where truncation is most sensitive.
    levels = (np.arange(1, 255) + rng.uniform(-1e-4, 1e-4, size=(160, 254))).ravel() / 255.0
    difference = np.log(levels / (1.0 - levels))
    base = rng.normal(0, 5, size=difference.shape)
    blocks.append(np.stack([base, base + difference]))
    special = np.array([0.0, -0.0, 1.0, -1.0, 20.0, -20.0, 100.0, -100.0, 745.0, -745.0, 88.72, -88.72,
                        float(info.max), float(-info.max), float(info.tiny), float(-info.tiny),
                        float(info.eps), 1e-3, 11.09, 17.33])
    a, b = np.meshgrid(special, special)
    blocks.append(np.stack([a.ravel(), b.ravel()]))
    with np.errstate(over='ignore'):
        logits = np.concatenate(blocks, axis=1).astype(dtype)
    return logits.reshape(2, 1, 1, -1)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
@pytest.mark.parametrize("seed", [0, 1])
def test_two_channel_softmax_is_bit_identical(dtype, seed):
    logits = adversarial_logits(dtype, seed)
    before = logits.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        got, got_empty = apply_finalization(logits, 2, FinalizeConfig(mode="binary"))
        want, want_empty = reference_finalization(logits, 2, FinalizeConfig(mode="binary"))
    assert got_empty is want_empty is False
    assert got.dtype == np.uint8 and got.shape == want.shape == (1, *logits.shape[1:])
    assert np.array_equal(got, want)
    assert np.array_equal(logits, before, equal_nan=True)  # the input is not modified
    assert got.min() == 0 and got.max() == 255


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_non_finite_logits_take_the_general_softmax(bad):
    logits = np.random.default_rng(3).normal(0, 3, size=(2, 2, 3, 50)).astype(np.float32)
    logits[1, 0, 0, 0] = bad
    with np.errstate(invalid='ignore'):
        want_probabilities = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        want_probabilities = (want_probabilities / np.sum(want_probabilities, axis=0, keepdims=True))[1:2]
        got = finalize_outputs._softmax_foreground(logits, all_finite=False)
        assert finalize_outputs._is_constant(logits) == (False, False)
    assert np.array_equal(got, want_probabilities, equal_nan=True)


@pytest.mark.parametrize("config,num_classes", [
    (FinalizeConfig(mode="binary", threshold=0.3), 2),
    (FinalizeConfig(mode="multiclass"), 2),
    (FinalizeConfig(mode="multiclass"), 4),
])
def test_other_modes_are_unchanged(config, num_classes):
    logits = np.random.default_rng(4).normal(0, 3, size=(num_classes, 6, 7, 8)).astype(np.float32)
    logits[:, 0] = 0  # uncovered voxels
    got, _ = apply_finalization(logits, num_classes, config)
    want, _ = reference_finalization(logits, num_classes, config)
    assert np.array_equal(got, want)


def test_multi_task_two_channel_targets_are_bit_identical():
    logits = np.concatenate([adversarial_logits(np.float32, 5), adversarial_logits(np.float32, 6)], axis=0)
    target_info = {'ink': {'start_channel': 2, 'end_channel': 4}, 'surface': {'start_channel': 0, 'end_channel': 2}}
    config = FinalizeConfig(mode="binary", is_multi_task=True, target_info=target_info)
    with np.errstate(over='ignore'):
        got, _ = apply_finalization(logits, 4, config)
        want = [reference_finalization(logits[a:b], 2, FinalizeConfig(mode="binary"))[0][0] for a, b in ((0, 2), (2, 4))]
    uncovered = np.all(logits == 0, axis=0)  # across all four channels, as apply_finalization masks
    for channel, expected in zip(got, want):
        assert np.array_equal(channel[~uncovered], expected[~uncovered])


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64])
def test_constant_chunk_test_matches_allclose(dtype):
    rng = np.random.default_rng(7)
    info = np.finfo(dtype)
    cases = []
    for first in (0.0, 1.0, -3.5, 1e-7, 250.0, 6e4, float(info.tiny), float(info.max) / 2):
        first = dtype(first)
        step = np.spacing(np.abs(first))
        tolerance = 1e-8 + 1e-6 * abs(float(first))
        for width in (0.0, float(step), 0.5 * tolerance, 0.999 * tolerance, tolerance, 1.001 * tolerance,
                      2 * tolerance, 1e-3, 1.0):
            for sign in (1, -1):
                values = np.full(64, first, dtype=dtype)
                values[1:] += (sign * width * rng.random(63)).astype(dtype)
                values[int(rng.integers(1, 64))] = first + dtype(sign * width)
                cases.append(values)
    for special in (np.nan, np.inf, -np.inf):
        for position in (0, 5):
            values = np.zeros(64, dtype=dtype)
            values[position] = special
            cases.append(values)
        cases.append(np.full(64, special, dtype=dtype))
    outcomes = set()
    for values in cases:
        logits = values.reshape(2, 2, 4, 4)
        with np.errstate(invalid='ignore', over='ignore'):
            want = bool(np.allclose(logits, logits.flat[0], rtol=1e-6))
            got, all_finite = finalize_outputs._is_constant(logits)
        assert got is want
        assert all_finite is bool(np.isfinite(logits).all())
        outcomes.add(want)
    assert outcomes == {True, False}
    assert apply_finalization(np.full((2, 2, 2, 2), 0.25, dtype=dtype), 2, FinalizeConfig()) == (None, True)


def test_finalize_reuses_worker_stores_and_matches_fresh_opens(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    logits = rng.normal(0, 3, size=(2, 16, 16, 24)).astype(np.float16)
    logits[:, :8, :8, :8] = 0  # one empty chunk
    source = str(tmp_path / "logits.zarr")
    store = ZarrWriter.create(source, shape=logits.shape, chunks=(1, 8, 8, 8), dtype=np.float16,
                              cname="zstd", level=1, skip_empty=True)
    store[...] = logits

    pooled = str(tmp_path / "pooled.zarr")
    finalize_logits(source, pooled, mode="binary", num_workers=2, verbose=False)
    want, _ = reference_finalization(logits, 2, FinalizeConfig(mode="binary"))
    want[:, :8, :8, :8] = 0  # skipped as empty, left at the fill value
    assert np.array_equal(open_zarr(pooled)[...], want)

    # Direct calls, as before the worker state existed: the stores are opened per call...
    direct = str(tmp_path / "direct.zarr")
    ZarrWriter.create(direct, shape=(1, 16, 16, 24), chunks=(1, 8, 8, 8), dtype=np.uint8, skip_empty=True)
    monkeypatch.setattr(finalize_outputs, "_worker_state", {})
    args = dict(input_path=source, output_path=direct, mode="binary", threshold=None, num_classes=2,
                spatial_shape=(16, 16, 24), output_chunks=(1, 8, 8, 8))
    for indices in np.ndindex(2, 2, 3):
        finalize_outputs.process_chunk({'indices': indices}, **args)
    assert np.array_equal(open_zarr(direct)[...], want)

    # ...and once per worker after _init_worker, for these two stores only.
    opened = []
    finalize_outputs._init_worker(source, direct)
    monkeypatch.setattr(finalize_outputs, "open_zarr", lambda *a, **k: opened.append(a) or open_zarr(*a, **k))
    finalize_outputs.process_chunk({'indices': (1, 1, 1)}, **args)
    assert opened == []
    finalize_outputs.process_chunk({'indices': (1, 1, 1)}, **{**args, 'output_path': pooled})
    assert len(opened) == 1
