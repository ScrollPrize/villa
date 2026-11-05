"""Dense block correlation: the mechanics, shared by the research scripts and the session.

Take a block of the coarse volume, resample the fine volume onto its grid under the
current transform, and measure the residual translation by phase correlation. Each block
then yields one correspondence for the same fits the landmarks use. Why the method is
worth having is measured in `research/dense_blocks.py` and `auto-refinement-research.md`
section 10; this module is only how it is done.

These things make this cheap enough to be worth doing:
  * the fine side is read from the pyramid level whose voxel size matches the coarse
    volume's, so a 128^3 coarse block costs a 128^3 fine read rather than a 500^3 one,
    and the binning that produced that level is also the anti-aliasing the resampling
    needs
  * every array that outlives a block -- the coarse blocks, the cached fine patches --
    is held in the store's own uint8, and the lattice a block is resampled on is built a
    slab at a time in float32. A 128^3 block holds two million points; carried as float64
    through the shared `apply` and `to_level` helpers, with a temporary per arithmetic
    step, that alone reached several hundred megabytes per block and multiplied by the
    worker count
  * blocks are the store's own chunks. These stores are written in uncompressed 128^3
    chunks, so a chunk-aligned 128^3 block is one request with nothing wasted, where the
    same block through the 32^3 chunk rewriter is 64 requests
    (`research/chunk_access_cost.py`). Reading the public bucket directly also keeps a
    shared service out of the loop entirely.
  * only blocks that contain texture and lie inside both masks are read at all
  * nothing is ever scanned: blocks are sampled inside the region the landmarks span

The fine side cannot be chunk-aligned as well -- the transform decides where a coarse
block lands -- so its patch is read with a small pad and kept, which lets every later
pass re-interpolate without touching the network again. The pad is voxels, not whole
chunks: rounding the request out to chunk boundaries would cost nothing on the wire,
since those chunks are fetched either way, but it makes the patch this has to hold on to
eight times larger, and eighty of those do not fit in memory.

The precision ceiling is the coarse volume. Correlating a 9.4 um image against a 9.4 um
image cannot resolve 2.4 um detail, so the target is a fraction of a coarse voxel per
block -- and then the fit averages a few hundred of those.
"""

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from .volumes import read_patch
from .deformable import choose_regularisation, fit_correction
from .transform_fit import fit_affine12
from .transform_io import matrix_swap_xyz_zyx, points_swap_xyz_zyx

# One store chunk. Bigger than the 64^3 the blob work uses, which buys both a single
# request per block and twice the capture range, at 8x the FFT cost -- still milliseconds.
BLOCK = 128
CHUNK = 128

MAX_ATTEMPTS_PER_BLOCK = 12
# Candidates are read in rounds of this many so the sampler can stop once it has enough.
# `ThreadPoolExecutor.map` submits every task before the loop body can break, so handing
# it all `wanted * MAX_ATTEMPTS_PER_BLOCK` origins at once reads all of them however few
# are needed -- 960 blocks fetched to keep 80, twelve times over. Nine candidates in ten
# are kept, so one round of a little more than `wanted` almost always suffices. Measured
# on PHerc0139, reading is latency-bound rather than bandwidth-bound: 200 full 128^3
# blocks and 200 level-2 probes a sixty-fourth the size both take about 33 s, so what
# costs time is the number of requests and nothing else.
SAMPLE_ROUND = 1.3
UPSAMPLE = 20
THREADS = 6

# A block with no texture has no correlation peak to find, and one that straddles the
# mask edge locks onto the edge instead of the contents. Both thresholds are the ones
# blob_discovery already uses on the coarse side.
MIN_BLOCK_STD = 12.0
MAX_EMPTY_FRACTION = 0.2

# Interpolation needs a voxel either side of the grid it samples. The cached patch gets
# more, so that later passes -- which move the grid by well under a voxel -- stay inside
# what was already read.
MARGIN = 3
CACHE_PAD = 8

# The lattice is built this many planes at a time. All 128^3 of it at once, with the
# temporaries each arithmetic step makes, is what put six workers over four gigabytes.
SLAB = 16

# How far a block's measured shift may sit before it is treated as a mismatch rather than
# a measurement. In coarse voxels, and wide enough to cover a starting transform that is
# a few voxels out.
MAX_SHIFT_VOXELS = 8.0

# The perturbation the sign calibration plants, in coarse voxels. Large enough that the
# wrong sign lands unmistakably far away, small enough to stay inside the capture range.
CALIBRATION_SHIFT = np.array([2.0, -3.0, 1.5])
# How far above the residual the transform already has the planted displacement must sit.
# The wrong sign misses by about twice the plant, so a factor of four leaves the two signs
# roughly eight times apart, against the threefold separation the test demands.
CALIBRATION_MARGIN = 4.0

# How far a block's correspondence may sit from the consensus affine before it is treated
# as a block that locked onto the wrong feature rather than one the volumes disagree at.
# In *coarse* voxels, because that is what the correlation measured in: a mis-lock is a
# peak on the wrong feature of a coarse-resolution image, so its scale is the coarse
# voxel and nothing else. Stating it in fixed voxels, as this first did, silently made the
# rule four times stricter on the two pairs whose fixed side is the fine volume -- 3.3 um
# against a 6.2 um field on PHerc0343P, which threw away 58% of the blocks for carrying
# exactly the displacement the measurement exists to find.
RANSAC_THRESHOLD_COARSE_VOXELS = 1.5

# A discarded block is called debris when it sits this many times further from what its
# neighbours say than those neighbours sit from each other. Measuring against the
# neighbourhood's own spread rather than against the blocks that were kept matters: a
# robust fit keeps whatever agrees with an affine, which is systematically the flattest
# part of the field, so the kept blocks are a reference that makes every steep region look
# like debris.
DEBRIS_FACTOR = 3.0
DISCARD_NEIGHBOURS = 8


def as_zyx(matrix_xyz):
    """A stored 3x4 XYZ matrix as a 4x4 ZYX one, which is what the fits work in."""
    return matrix_swap_xyz_zyx(np.vstack([np.array(matrix_xyz, float), [0, 0, 0, 1]]))


def apply(matrix, points):
    return np.asarray(points, float) @ matrix[:3, :3].T + matrix[:3, 3]


def pick_fine_level(levels, wanted_scale):
    """The pyramid level whose voxel size is closest to the coarse volume's.

    Reading the fine volume here rather than at level 0 is what makes a block affordable,
    and because these levels are box-mean reductions it is also the low-pass the coarse
    grid needs: sampling 2.4 um data on a 9.4 um lattice without it would alias.
    """
    return min(levels, key=lambda level: abs(np.log(level[1] / wanted_scale)))


def try_read(array, lo, hi):
    """A patch, or None where the store cannot produce one.

    Unwritten regions of these stores answer with a 500 that the rewriter reports as a
    404, which is indistinguishable here from any other read failure and equally fatal to
    the block. One missing chunk must not end a run over hundreds of blocks.
    """
    try:
        return read_patch(array, lo, hi)
    except Exception:  # noqa: BLE001 - an unreadable block is a skipped block
        return None


def lattice_to_fine_level(fine_to_coarse, fine_scale):
    """One affine map from coarse level-0 voxels to fine-level voxels, in float32.

    Inverting the transform and converting to a pyramid level are both affine, so they
    compose into a single matrix. Fusing them matters here only because it removes a whole
    intermediate copy of a two-million-point lattice, and the shared `to_level` helper
    works in float64.
    """
    coarse_to_fine = np.linalg.inv(fine_to_coarse)
    linear = (coarse_to_fine[:3, :3] / fine_scale).astype(np.float32)
    offset = ((coarse_to_fine[:3, 3] + 0.5) / fine_scale - 0.5).astype(np.float32)
    return linear, offset


def level_coordinates(points, fine_scale):
    """Fine level-0 coordinates to the coordinates of the level actually being read.

    The same convention `lattice_to_fine_level` folds into its matrix, kept separately
    because a bending map cannot be folded into anything.
    """
    return (points + 0.5) / fine_scale - 0.5


# Points along each axis used to bound a bending map. Eight corners are enough for an
# affine and not for a spline, whose interior can bulge past them; nine per axis is 729
# evaluations against a block's two million, and the map is smooth between them.
BEND_SAMPLES = 9


def block_bounds(linear, offset, origin, size, to_fine=None, fine_scale=1):
    """Where a coarse block lands in the fine volume.

    An affine map sends a box to the bounding box of its corners, so eight points give the
    extent exactly, without building the lattice that only the interpolation needs.

    A bending map does not obey that. The inside of a block can land outside where its
    corners did, and a point outside the patch that was read is clamped to its edge
    without complaint -- which draws a row of background through the middle of a panel
    when that edge is background. So a bend is bounded by a grid through the block rather
    than by its corners.
    """
    if to_fine is None:
        corners = np.array([[origin[0] + a * (size - 1),
                             origin[1] + b * (size - 1),
                             origin[2] + c * (size - 1)]
                            for a in (0, 1) for b in (0, 1) for c in (0, 1)],
                           dtype=np.float32)
        mapped = corners @ linear.T + offset
    else:
        steps = np.linspace(0, size - 1, BEND_SAMPLES)
        grid = np.stack(np.meshgrid(origin[0] + steps, origin[1] + steps,
                                    origin[2] + steps, indexing="ij"), -1)
        mapped = level_coordinates(to_fine(grid.reshape(-1, 3).astype(np.float32)),
                                   fine_scale)
    return (np.floor(mapped.min(0)).astype(int) - MARGIN,
            np.ceil(mapped.max(0)).astype(int) + MARGIN)


def covers(fine_array, fine_scale, fine_to_coarse, origin, size):
    """Would `resample_fine` return a block here? Answered without interpolating one.

    The same two refusals, in the same order: the block must fall inside the fine volume,
    and the patch behind it must not be mostly background. What it skips is the
    interpolation of every lattice point, which is the expensive half and which decides
    nothing -- a census asks only whether a block exists, not what it looks like.
    """
    linear, offset = lattice_to_fine_level(fine_to_coarse, fine_scale)
    want_lo, want_hi = block_bounds(linear, offset, origin, size, None, fine_scale)
    if (want_lo < 0).any() or (want_hi > np.array(fine_array.shape)).any():
        return False
    patch = try_read(fine_array, want_lo, want_hi)
    if patch is None or patch.size == 0:
        return False
    return bool((patch == 0).mean() <= MAX_EMPTY_FRACTION)


def resample_fine(fine_array, fine_scale, fine_to_coarse, origin, size, cache=None,
                  key=None, to_fine=None):
    """The fine volume on a coarse block's lattice, or None if it is not covered.

    `origin` is in coarse level-0 voxels; the returned block is that grid with fine
    intensities interpolated onto it. When a cache is given, the fine patch is read once
    with a small pad and reused by every later pass whose points still fall inside it --
    which they do, because after the first pass the transform moves by well under a voxel.

    `to_fine` replaces the affine entirely: a callable taking coarse level-0 points and
    returning fine level-0 ones, which is how a bending transform is drawn. It is far
    slower -- a dense kernel against every control point, per lattice point -- so it is
    for looking at blocks rather than measuring hundreds of them, and `fine_to_coarse` is
    still required because the cache and the bounds are sized from it.
    """
    linear, offset = lattice_to_fine_level(fine_to_coarse, fine_scale)
    want_lo, want_hi = block_bounds(linear, offset, origin, size, to_fine, fine_scale)
    if (want_lo < 0).any() or (want_hi > np.array(fine_array.shape)).any():
        return None

    held = cache.get(key) if cache is not None else None
    if held is not None and (held[0] <= want_lo).all() and (want_hi <= held[1]).all():
        lo, _hi, patch = held
    else:
        lo = np.maximum(want_lo - CACHE_PAD, 0)
        hi = np.minimum(want_hi + CACHE_PAD, np.array(fine_array.shape))
        patch = try_read(fine_array, lo, hi)
        if patch is None or patch.size == 0:
            return None
        # Held as it was stored. Casting to float here would quadruple what the cache
        # costs, for a conversion that is cheap to redo per use.
        if cache is not None:
            cache[key] = (lo, hi, patch)

    if (patch == 0).mean() > MAX_EMPTY_FRACTION:
        return None

    patch = patch.astype(np.float32)
    local_offset = offset - lo.astype(np.float32)
    shifted = lo.astype(np.float32)
    rows = np.arange(size, dtype=np.float32) + origin[1]
    columns = np.arange(size, dtype=np.float32) + origin[2]
    plane_rows, plane_columns = np.meshgrid(rows, columns, indexing="ij")
    out = np.empty((size, size, size), dtype=np.float32)
    for start in range(0, size, SLAB):
        stop = min(start + SLAB, size)
        depth = stop - start
        planes = np.arange(start, stop, dtype=np.float32) + origin[0]
        lattice = np.stack([
            np.broadcast_to(planes[:, None, None], (depth, size, size)),
            np.broadcast_to(plane_rows, (depth, size, size)),
            np.broadcast_to(plane_columns, (depth, size, size)),
        ], axis=-1).reshape(-1, 3)
        points = (lattice @ linear.T + local_offset if to_fine is None
                  else level_coordinates(to_fine(lattice), fine_scale) - shifted)
        out[start:stop] = ndimage.map_coordinates(
            patch, np.ascontiguousarray(points.T), order=1, mode="nearest"
        ).reshape(depth, size, size)
    return out


def window(block):
    """Hann along each axis, so the FFT does not see the block's own edges.

    Built per axis rather than once from the first one: the blocks are cubes, but the
    half-blocks the precision check correlates are not.
    """
    result = np.asarray(block, dtype=np.float32)
    for axis, length in enumerate(result.shape):
        shape = [1, 1, 1]
        shape[axis] = length
        result = result * np.hanning(length).astype(np.float32).reshape(shape)
    return result


def correlate(coarse_block, fine_block):
    """Sub-voxel residual translation between two blocks on the same grid.

    Phase normalisation, not plain correlation: the two scans were taken at different
    energies (78 keV and 113 keV for this pair), so their intensities are not comparable
    and only the phase of the cross-spectrum carries the alignment.
    """
    shift, error, _phase = phase_cross_correlation(
        window(coarse_block.astype(np.float32)),
        window(fine_block.astype(np.float32)),
        upsample_factor=UPSAMPLE,
        normalization="phase",
    )
    return np.asarray(shift, float), float(error)


def signal_centre(block, origin):
    """Where in the block the correlation's evidence actually is, in coarse voxels.

    A block correlation reports one translation for the whole block, weighted by wherever
    the texture is. Attributing it to the geometric centre would place the measurement at
    a spot that contributed nothing to it, so it is attributed to the intensity centroid
    -- the same estimator the landmark snap uses, applied to a block instead of a blob.
    """
    weights = window(np.clip(block.astype(np.float32) - np.median(block), 0, None))
    total = float(weights.sum())
    if total <= 0:
        return np.asarray(origin, float) + block.shape[0] / 2.0
    # Marginalised one axis at a time: three 1-D sums instead of three full 3-D grids,
    # which at 128^3 is the difference between kilobytes and 150 MB per block.
    centre = []
    for axis in range(3):
        profile = weights.sum(axis=tuple(k for k in range(3) if k != axis))
        centre.append(float((profile * np.arange(len(profile))).sum() / total))
    return np.asarray(origin, float) + np.array(centre)


def candidate_origins(coarse_array, bounds, rng, wanted):
    """Block origins worth trying, inside the region the landmarks span.

    Split from reading them because a caller that can reject an origin without looking at
    it pays nothing for the rejection. The blind review does exactly that: it drops any
    block where the two transforms it is comparing place the image within half a voxel of
    each other, which is decided by arithmetic on the origin and would otherwise cost a
    two-megabyte read per discarded block.
    """
    low, high = bounds
    shape = np.array(coarse_array.shape)
    origins = []
    for _ in range(wanted * MAX_ATTEMPTS_PER_BLOCK):
        origin = np.array([
            int(rng.integers(max(0, int(low[k])),
                             max(1, min(int(high[k]), shape[k] - BLOCK))))
            for k in range(3)
        ])
        # Snapped to the store's own chunk grid, so one block is one chunk and one
        # request. Nothing about the measurement cares where a block starts.
        origin = (origin // CHUNK) * CHUNK
        if (origin + BLOCK <= shape).all():
            origins.append(origin)
    return origins


def chunk_grid_origins(shape, bounds, rng, count):
    """Up to `count` distinct chunk-aligned block origins inside `bounds`, in random order.

    `candidate_origins` draws a voxel and snaps it down to the chunk grid, which reaches
    the last whole chunk along an axis only from the few voxels between its start and
    `shape - BLOCK` -- so the far edge of a volume is drawn a fraction of as often as the
    rest, and in a small overlap almost never. This draws the chunk positions themselves,
    each equally likely and none twice. The research runs keep the old sampler so their
    numbers stay reproducible.
    """
    low, high = bounds
    shape = np.asarray(shape)
    first = np.ceil(np.maximum(np.asarray(low, float), 0) / CHUNK).astype(int)
    last = np.floor((np.minimum(np.asarray(high, float) + 1, shape) - BLOCK)
                    / CHUNK).astype(int)
    if (last < first).any():
        return []
    sizes = last - first + 1
    total = int(np.prod(sizes))
    picks = rng.choice(total, size=min(int(count), total), replace=False)
    cells = np.stack(np.unravel_index(picks, tuple(sizes)), axis=1) + first
    return [cell * CHUNK for cell in cells]


# A block is 128 level-0 voxels, so at one thirty-second it is four voxels a side: small
# enough that a whole scroll fits in memory, large enough to tell air from tissue.
PREFILTER_SCALE = 32.0
# Only a window that is essentially all exact zero is refused. A stricter rule -- too
# little spread, too dark -- would reject real tissue that happens to be bland at one
# thirty-second, and a false negative here silently shrinks the region the review claims
# to cover, which is the bias this rebuild exists to remove. Outside the reconstructed
# field these volumes are exactly zero, and averaging zeros gives zero, so this refuses
# only what the full-resolution rule would refuse anyway.
PREFILTER_EMPTY = 0.9


def coarsest_level(levels, want=PREFILTER_SCALE):
    """The most reduced level available, preferring one at least `want` times smaller.

    None when the store has no pyramid, which means every candidate is read in full --
    the behaviour before this existed.
    """
    reduced = [level for level in levels if level[1] > 1.0]
    if not reduced:
        return None
    big = [level for level in reduced if level[1] >= want]
    return min(big, key=lambda level: level[1]) if big else max(
        reduced, key=lambda level: level[1])


class TissueMap(NamedTuple):
    """One coarse read of a whole sampling region, to answer "is anything here" for free.

    Chunking is why this is a whole region rather than a block at a time: reading four
    voxels of a level costs the chunk they sit in, so probing a thousand positions one by
    one costs a thousand chunks and saves nothing at all.
    """

    data: object
    scale: float
    low: object

    def holds_tissue(self, origin, size=BLOCK):
        lo = ((np.asarray(origin, float) - self.low) / self.scale).astype(int)
        hi = lo + max(int(size / self.scale), 1)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.data.shape))
        if (hi <= lo).any():
            return False
        patch = self.data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if patch.size == 0:
            return False
        return bool((patch == 0).mean() <= PREFILTER_EMPTY)


# Slabs the one coarse read is split into. A single read of a whole box walks its chunks
# one at a time inside zarr, which leaves fifteen of sixteen cores and most of the network
# idle; these are fetched at once and stacked.
MAP_SLABS = 16


def tissue_map(levels, low, high, want=PREFILTER_SCALE, slabs=MAP_SLABS):
    """Read `low`..`high` of the coarse volume once, reduced, as an in-memory mask."""
    level = coarsest_level(levels, want)
    if level is None:
        return None
    array, scale = level
    lo = np.maximum((np.asarray(low, float) / scale).astype(int) - 1, 0)
    hi = np.minimum((np.asarray(high, float) / scale).astype(int) + 2,
                    np.asarray(array.shape))
    if (hi <= lo).any():
        return None

    axis = int(np.argmax(hi - lo))
    edges = np.linspace(lo[axis], hi[axis], min(slabs, int(hi[axis] - lo[axis])) + 1)
    edges = sorted(set(int(round(edge)) for edge in edges))

    def slab(bounds):
        start, stop = bounds
        at_lo, at_hi = lo.copy(), hi.copy()
        at_lo[axis], at_hi[axis] = start, stop
        return try_read(array, at_lo, at_hi)

    with ThreadPoolExecutor(max_workers=min(len(edges) - 1, THREADS * 2)) as pool:
        parts = list(pool.map(slab, list(zip(edges[:-1], edges[1:]))))
    if any(part is None for part in parts):
        return None
    patch = np.concatenate([np.asarray(part) for part in parts], axis=axis)
    if patch.size == 0:
        return None
    return TissueMap(patch, scale, lo * scale)


def worth_reading(where, origin, size=None):
    """Could this block hold tissue? Answered from the map, so a no costs nothing.

    Conservative by construction: a no means the region is nearly all background at low
    resolution, and a yes decides nothing -- `textured_block` still applies the real rule
    at full resolution. True when there is no map to ask.
    """
    if where is None:
        return True
    return where.holds_tissue(origin, BLOCK if size is None else size)


def block_rejection(block):
    """Why a coarse block holds nothing to correlate, or None if it is worth correlating.

    Split from `textured_block` so a caller that keeps the rejected blocks -- the review
    session shows them to a person -- applies the same rule and can say which half failed.
    """
    if block.std() < MIN_BLOCK_STD:
        return "no-texture"
    if (block == 0).mean() > MAX_EMPTY_FRACTION:
        return "mask-edge"
    return None


def textured_block(coarse_array, origin):
    """The block at `origin`, or None if it will not read or holds nothing to correlate."""
    block = try_read(coarse_array, origin, origin + BLOCK)
    if block is None:
        return None
    # Kept as stored. Every block is held for the whole run, and at 128^3 the float
    # copy would cost four times as much for a conversion each use redoes anyway.
    if block_rejection(block) is not None:
        return None
    return block


def sample_blocks(coarse_array, bounds, rng, wanted):
    """Textured blocks of the coarse volume inside the region the landmarks span."""
    origins = candidate_origins(coarse_array, bounds, rng, wanted)

    def fetch(origin):
        block = textured_block(coarse_array, origin)
        return None if block is None else (origin, block)

    blocks, tried = [], 0
    step = max(THREADS, int(wanted * SAMPLE_ROUND))
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for start in range(0, len(origins), step):
            round_of = origins[start:start + step]
            tried += len(round_of)
            blocks.extend(r for r in pool.map(fetch, round_of) if r is not None)
            if len(blocks) >= wanted:
                break
    # The same blocks, in the same order, as reading every candidate would have kept.
    return blocks[:wanted], tried


class Measurement(NamedTuple):
    centre: np.ndarray
    fine_point: np.ndarray
    shift: np.ndarray
    error: float
    match: float | None
    check: dict | None
    octants: list | None = None


def measure_block(origin, coarse_block, fine_array, fine_scale, fine_to_coarse, sign,
                  cache=None, key=None, with_match=True, with_check=False,
                  with_octants=False):
    """One block's correspondence, or a string naming why it has none.

    Returns a `Measurement`: the coarse point the evidence sits at, the fine point it
    matches, the residual shift in coarse voxels, the correlation's own error,
    `aligned_match` of the two blocks (None without `with_match`, which costs a resample
    of the fine block), `peak_check` of the shift (None without `with_check`, which
    costs a few dozen more NCCs), and `measure_octants` (None without `with_octants`,
    about the cost of the block again). The two refusals are told apart
    because a person reviewing the blocks needs to know whether the fine volume was
    missing or the peak was out of range.
    """
    fine_block = resample_fine(fine_array, fine_scale, fine_to_coarse, origin, BLOCK,
                               cache, key)
    if fine_block is None:
        return "not-covered"
    shift, error = correlate(coarse_block, fine_block)
    if not np.isfinite(shift).all() or np.linalg.norm(shift) > MAX_SHIFT_VOXELS:
        return "shift-out-of-range"
    centre = signal_centre(coarse_block, origin)
    return Measurement(
        centre, apply(np.linalg.inv(fine_to_coarse), centre + sign * shift), shift, error,
        aligned_match(coarse_block, fine_block, shift) if with_match else None,
        peak_check(coarse_block, fine_block, shift) if with_check else None,
        measure_octants(origin, coarse_block, fine_block, fine_to_coarse, sign)
        if with_octants else None)


def measure_octants(origin, coarse_block, fine_block, fine_to_coarse, sign):
    """The block's eight half-size sub-blocks, each measured on its own.

    One shift per 128^3 block assumes the block moved as a whole. Where it did not -- the
    outer windings at the end of a scroll, a piece hanging off -- the eight halves along
    z, y and x tell a smooth gradient (the scroll deformed inside the block) from one
    half that disagrees (something in it moved) from a scatter (nothing to trust), and
    the halves that agree can still be used as points of their own.

    Returns one dict per octant, `corner_zyx` its offset in the block: either `rejected`
    with the reason, or `centre_zyx`, `fine_zyx` and `shift_zyx` as for a block, with its
    `match` and its `check` (`peak_check`).
    """
    half = BLOCK // 2
    to_fine = np.linalg.inv(fine_to_coarse)
    octants = []
    for corner in itertools.product((0, half), repeat=3):
        cut = tuple(slice(start, start + half) for start in corner)
        coarse, fine = coarse_block[cut], fine_block[cut]
        entry = {"corner_zyx": list(corner)}
        reason = block_rejection(coarse)
        shift = None
        if reason is None:
            shift, _error = correlate(coarse, fine)
            if not np.isfinite(shift).all() or np.linalg.norm(shift) > MAX_SHIFT_VOXELS:
                reason = "shift-out-of-range"
        if reason is not None:
            entry["rejected"] = reason
        else:
            centre = signal_centre(coarse, np.asarray(origin, float) + corner)
            entry.update(
                centre_zyx=centre.tolist(),
                fine_zyx=np.asarray(apply(to_fine, centre + sign * shift), float).tolist(),
                shift_zyx=np.asarray(shift, float).tolist(),
                match=aligned_match(coarse, fine, shift),
                check=peak_check(coarse, fine, shift))
        octants.append(entry)
    return octants


def aligned_match(coarse_block, fine_block, shift):
    """How alike two blocks look once the measured shift is applied: windowed NCC, -1 to 1.

    The phase correlation's own `error` says nothing here -- with phase normalisation it
    comes back as 1.0 for every block -- and the peak position alone cannot tell a block
    that lines up from one that merely has a maximum somewhere. This is the direct
    question: after moving the fine block by what the correlation found, do the two agree.
    """
    moved = ndimage.shift(np.asarray(fine_block, np.float32), shift, order=1, mode="nearest")
    a = window(np.asarray(coarse_block, np.float32) - float(np.mean(coarse_block)))
    b = window(moved - float(moved.mean()))
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / norm) if norm > 0 else 0.0


# How far `peak_check` climbs from the measured shift towards a better match, in voxels.
# The measurements the review turned up as wrong were off by one to two voxels.
PEAK_REACH = 4
OFFSETS_27 = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def peak_check(coarse_block, fine_block, shift):
    """Whether the measured shift is where the two blocks actually match best.

    The phase correlation answers with one peak and no sign of how good it is, and a block
    of good papyrus can still come back a voxel or two off. This asks the direct question
    in the image itself: starting from the measured shift, climb the windowed NCC over
    whole-voxel offsets to its maximum, then fit a quadratic over the 27 neighbours there.

    Returns a dict:

    * `offset` -- how far the NCC's own optimum is from the measured shift, in voxels. A
      measurement that is right sits at its own optimum; one that is off by more than
      about half a voxel is the measurement's fault, whatever the fit does.
    * `peak_zyx` -- the shift at that optimum, the convention `shift` is in.
    * `peak_match` -- the NCC there, on the block less a `PEAK_REACH + 1` margin.
    * `sharpness` -- how fast the NCC falls off the optimum, per voxel squared, along the
      three principal directions, weakest first. A sheet pinned across but sliding along
      itself has one or two weak directions; texture pins all three.
    """
    moved = ndimage.shift(np.asarray(fine_block, np.float32), shift, order=1, mode="nearest")
    coarse = np.asarray(coarse_block, np.float32)
    margin = PEAK_REACH + 1
    inner = tuple(slice(margin, length - margin) for length in coarse.shape)
    a = coarse[inner]
    a = window(a - float(a.mean()))
    a_norm = float(np.sqrt((a * a).sum()))
    seen = {}

    def ncc(offset):
        # moved[x + d] is the fine block shifted by `shift - d`.
        offset = tuple(int(value) for value in offset)
        if offset not in seen:
            b = moved[tuple(slice(margin + d, length - margin + d)
                            for d, length in zip(offset, moved.shape))]
            b = window(b - float(b.mean()))
            norm = a_norm * float(np.sqrt((b * b).sum()))
            seen[offset] = float((a * b).sum() / norm) if norm > 0 else 0.0
        return seen[offset]

    centre = np.zeros(3, int)
    for _ in range(PEAK_REACH):
        around = [centre + step for step in OFFSETS_27
                  if np.abs(centre + step).max() <= PEAK_REACH]
        best = max(around, key=ncc)
        if (best == centre).all():
            break
        centre = best
    values = np.array([ncc(centre + step) for step in OFFSETS_27])
    z, y, x = OFFSETS_27.T.astype(float)
    design = np.stack([np.ones_like(z), z, y, x, z * z, y * y, x * x, z * y, z * x, y * x], 1)
    c = np.linalg.lstsq(design, values, rcond=None)[0]
    gradient = c[1:4]
    hessian = np.array([[2 * c[4], c[7], c[8]],
                        [c[7], 2 * c[5], c[9]],
                        [c[8], c[9], 2 * c[6]]])
    sharpness = np.sort(np.linalg.eigvalsh(-hessian))
    within = np.zeros(3)
    if sharpness[0] > 0:
        # Past one voxel the quadratic is extrapolating; the climb already stopped at the
        # best whole voxel, so the optimum is within it.
        within = np.clip(-np.linalg.solve(hessian, gradient), -1, 1)
    best_offset = centre + within
    return {
        "offset": float(np.linalg.norm(best_offset)),
        "peak_zyx": (np.asarray(shift, float) - best_offset).tolist(),
        "peak_match": ncc(centre),
        "sharpness": sharpness.tolist(),
    }


def pairs_from(blocks, fine_array, fine_scale, fine_to_coarse, sign, cache):
    """Correlate every block and name the fine point each coarse point matches.

    The fine point is read off the same expression the sign calibration checks, so the
    two cannot drift apart.
    """
    def one(entry):
        index, (origin, coarse_block) = entry
        measured = measure_block(origin, coarse_block, fine_array, fine_scale,
                                 fine_to_coarse, sign, cache, index, with_match=False)
        return None if isinstance(measured, str) else measured[:4]

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = [r for r in pool.map(one, enumerate(blocks)) if r is not None]
    if not results:
        return (np.empty((0, 3)),) * 3 + (np.empty(0),)
    coarse_points, fine_points, shifts, errors = zip(*results)
    return (np.array(coarse_points), np.array(fine_points),
            np.array(shifts), np.array(errors))


def first_found(measure, items, wanted):
    """`measure` over `items` in order, keeping the first `wanted` results that are not
    None -- the ones a loop that stops there would keep -- but read `THREADS` at a time in
    rounds, so a stop does not wait on reads nobody uses (see SAMPLE_ROUND)."""
    found = []
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        for start in range(0, len(items), max(wanted, THREADS)):
            for result in pool.map(measure, items[start:start + max(wanted, THREADS)]):
                if result is not None and len(found) < wanted:
                    found.append(result)
            if len(found) >= wanted:
                break
    return found


def starting_residual(blocks, fine_array, fine_scale, fine_to_coarse, wanted=12, cache=None,
                      keys=None):
    """How far the unperturbed transform already is from the images, in coarse voxels.

    The sign test plants a displacement and asks which sign undoes it. That only answers
    anything if the planted displacement is large against the error already there, so the
    error already there has to be measured rather than assumed.

    With a `cache` (and a key per block), the fine patches read here are the ones the
    first measurement pass reads -- same transform, same bounds -- and it reuses them.
    """
    def one(index):
        origin, coarse_block = blocks[index]
        fine_block = resample_fine(fine_array, fine_scale, fine_to_coarse, origin, BLOCK,
                                   cache=cache, key=None if keys is None else keys[index])
        if fine_block is None:
            return None
        shift, _error = correlate(coarse_block, fine_block)
        return float(np.linalg.norm(shift)) if np.isfinite(shift).all() else None

    seen = first_found(one, list(range(min(len(blocks), 40))), wanted)
    return float(np.median(seen)) if seen else float("nan")


def calibration_plant(residual):
    """The displacement to plant, given the residual the starting transform already has.

    Both are coarse voxels. The wrong sign lands about twice the plant away from where the
    right one lands, so the two are separated by roughly 2*plant while each carries the
    residual as noise; demanding a threefold separation therefore needs the plant to be a
    small multiple of the residual, not a constant.
    """
    planted = np.array(CALIBRATION_SHIFT, dtype=float)
    if not np.isfinite(residual):
        return planted
    direction = planted / np.linalg.norm(planted)
    return direction * max(np.linalg.norm(planted), CALIBRATION_MARGIN * float(residual))


def calibrate_sign(blocks, fine_array, fine_scale, fine_to_coarse, cache=None, keys=None):
    """Decide the sign by perturbing a transform that is already close to right.

    Offsetting the transform's translation by a known amount moves every resampled block
    by that amount, and the pair built from a correlated block must undo exactly it: the
    fine point named under the perturbed transform has to be the one the unperturbed
    transform names. Only one sign does that. The other lands about twice the
    perturbation away while looking every bit as healthy, which is why this is measured
    on the expression `pairs_from` uses rather than reasoned about.

    The size of the perturbation is not a free choice. A fixed number of coarse voxels is
    a different physical displacement on every pair -- 37 um against a 9.4 um volume, 9 um
    against a 2.4 um one -- and when it falls to the size of the residual the transform
    already has, both signs miss by about the same amount and the test says nothing. It
    refused three registrations that way, all of them pairs whose coarse side is itself
    fine. So the residual is measured first and the perturbation is set from it, which
    costs one extra correlation pass over twelve blocks and makes the test mean the same
    thing on every pair.
    """
    residual = starting_residual(blocks, fine_array, fine_scale, fine_to_coarse, cache=cache,
                                 keys=keys)
    planted = calibration_plant(residual)

    perturbed = np.array(fine_to_coarse, dtype=float)
    perturbed[:3, 3] += planted
    inverse_perturbed = np.linalg.inv(perturbed)
    inverse_true = np.linalg.inv(fine_to_coarse)

    def one(block):
        origin, coarse_block = block
        fine_block = resample_fine(fine_array, fine_scale, perturbed, origin, BLOCK)
        if fine_block is None:
            return None
        shift, _error = correlate(coarse_block, fine_block)
        if not np.isfinite(shift).all():
            return None
        centre = signal_centre(coarse_block, origin)
        want = apply(inverse_true, centre)
        return {sign: float(np.linalg.norm(apply(inverse_perturbed, centre + sign * shift) - want))
                for sign in (1.0, -1.0)}

    found = first_found(one, blocks[:40], 12)
    errors = {sign: [result[sign] for result in found] for sign in (1.0, -1.0)}

    if not errors[1.0]:
        raise RuntimeError("no block could be resampled under the perturbed transform")
    scores = {sign: float(np.median(values)) for sign, values in errors.items()}
    best = min(scores, key=scores.get)
    print(f"  the transform it starts from already sits {residual:.2f} coarse voxels "
          f"from the images")
    print(f"  planted {np.round(planted, 2)} coarse voxels into the translation, "
          f"{len(errors[1.0])} blocks")
    for sign in (1.0, -1.0):
        print(f"    centre {'+' if sign > 0 else '-'} shift -> "
              f"median {scores[sign]:8.2f} fine voxels from where it should land")
    if scores[best] > 0.3 * scores[-best]:
        raise RuntimeError(f"the two signs are not distinguishable: {scores}")
    return best


def ransac(fixed, moving, threshold, iterations=400, seed=0):
    """Least-squares is not robust to a block that locked onto the wrong feature.

    `threshold` is in the units of `fixed`, and the caller is responsible for putting it
    there -- see RANSAC_THRESHOLD_COARSE_VOXELS for why that is not a detail.
    """
    rng = np.random.default_rng(seed)
    best = np.zeros(len(fixed), bool)
    for _ in range(iterations):
        sample = rng.choice(len(fixed), 6, replace=False)
        try:
            linear, offset = fit_affine12(fixed[sample], moving[sample])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(moving @ linear.T + offset - fixed, axis=1)
        inliers = residual < threshold
        if inliers.sum() > best.sum():
            best = inliers
    return np.flatnonzero(best)


def classify_discards(points, shifts, inliers, coarse_um):
    """Split the blocks a robust fit dropped into debris and deformation.

    Debris -- loose material in the case, which moves between scans and belongs to no
    registration -- is isolated: it disagrees with its neighbours as much as with the
    global affine. A region that genuinely bent disagrees with the affine and *agrees*
    with its neighbours, because a deformation is smooth. An affine-consistency rule
    cannot tell them apart, so this asks the second question afterwards.

    Returns None when there is nothing to classify.
    """
    total = len(points)
    kept = np.zeros(total, bool)
    kept[np.asarray(inliers, int)] = True
    if kept.all() or total <= DISCARD_NEIGHBOURS + 1:
        return None

    separation = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    local = np.empty(total)
    spread = np.empty(total)
    for index in range(total):
        near = np.argsort(separation[index])[1:DISCARD_NEIGHBOURS + 1]
        centre = np.median(shifts[near], axis=0)
        local[index] = np.linalg.norm(shifts[index] - centre)
        # How much the neighbourhood disagrees with itself, which is the scale a block
        # inside a smooth field is allowed to differ by.
        spread[index] = np.median(np.linalg.norm(shifts[near] - centre, axis=1))
    floor = float(np.median(spread))
    line = DEBRIS_FACTOR * np.maximum(spread, floor)
    debris = (~kept) & (local > line)
    field = (~kept) & (local <= line)
    return {
        "kept": int(kept.sum()),
        "dropped": int((~kept).sum()),
        "share_kept": float(kept.mean()),
        "debris": int(debris.sum()),
        "field": int(field.sum()),
        "line_um": float(np.median(line[~kept]) * coarse_um),
        "kept_local_um": float(np.median(local[kept]) * coarse_um),
        "neighbourhood_spread_um": floor * coarse_um,
        "field_local_um": (float(np.median(local[field]) * coarse_um)
                           if field.any() else None),
        "debris_local_um": (float(np.median(local[debris]) * coarse_um)
                            if debris.any() else None),
    }

def bend_to_fine(dense, fixed_is_coarse):
    """The stored bend as a map from coarse points to fine ones, for resampling with.

    A deformation has to be fitted the other way round to be *drawn*. The file stores
    moving -> fixed; resampling asks the opposite question -- where in the fine volume does
    this coarse voxel come from -- which a spline cannot answer by being inverted. Fitting
    the same correspondences the other way round does answer it, and is what the schema
    tells a reader to do.

    Returns `(to_fine, chosen)`. `chosen` is False when the hold-out declined a bend and
    the best level is being handed back anyway: a rule declining a bend and a bend doing
    nothing are different claims, and only the second is visible in a panel, so a figure or
    a review can show what was turned down as long as it says which it is. `to_fine` is
    None only when no level of smoothing could be fitted at all.
    """
    forward = as_zyx(dense["transformation_matrix"])
    fixed_zyx = np.array(points_swap_xyz_zyx(dense["fixed_landmarks"]), float)
    moving_zyx = np.array(points_swap_xyz_zyx(dense["moving_landmarks"]), float)
    coarse_pts = fixed_zyx if fixed_is_coarse else moving_zyx
    fine_pts = moving_zyx if fixed_is_coarse else fixed_zyx
    coarse_to_fine = (np.linalg.inv(forward) if fixed_is_coarse else forward)[:3]

    lam, edf, scores = choose_regularisation(fine_pts, coarse_pts, coarse_to_fine)
    chosen = lam is not None
    if lam is None:
        bending = [row for row in scores if row[1] is not None]
        if bending:
            _score, lam, edf = min(bending, key=lambda row: row[0].held_out_rms)
    if lam is None:
        return None, chosen, None
    reverse = fit_correction(fine_pts, coarse_pts, coarse_to_fine, lam)
    return reverse.apply, chosen, (lam, edf)


def half_block_precision(blocks, fine_array, fine_scale, fine_to_coarse, sign, cache):
    """How much of the per-block residual is the method's own noise.

    Each block is cut in two and each half correlated on its own. The halves see
    different voxels of the same piece of object, so they share whatever displacement is
    really there and do not share their noise: the spread of their *difference* measures
    the measurement, with the registration divided out.

    Halving the data costs roughly sqrt(2) in precision and differencing two independent
    estimates costs another sqrt(2), so the full-block noise is about the spread of that
    difference divided by two. Comparing it against the residual the passes settled at
    says whether that residual is the method or the volumes.

    The two halves are adjacent, so any deformation that varies within a block lands in
    their difference and is counted as noise. That biases this towards calling a real
    residual noise, which is the safe direction: a large margin here is a floor.
    """
    differences = []
    for index, (origin, coarse_block) in enumerate(blocks):
        fine_block = resample_fine(fine_array, fine_scale, fine_to_coarse, origin, BLOCK,
                                   cache, index)
        if fine_block is None:
            continue
        half = BLOCK // 2
        estimates = []
        for lower, upper in ((0, half), (half, BLOCK)):
            shift, _error = correlate(coarse_block[lower:upper], fine_block[lower:upper])
            if not np.isfinite(shift).all() or np.linalg.norm(shift) > MAX_SHIFT_VOXELS:
                estimates = []
                break
            estimates.append(shift)
        if len(estimates) == 2:
            differences.append(estimates[0] - estimates[1])
    if len(differences) < 10:
        print("\ntoo few blocks split cleanly to estimate the precision")
        return None
    differences = np.array(differences)
    spread = float(np.median(np.linalg.norm(differences, axis=1)))
    noise = spread / 2.0
    print(f"\nhalf-block agreement over {len(differences)} blocks: "
          f"median |A - B| = {spread:.3f} coarse voxels")
    print(f"  implied full-block measurement noise ~{noise:.3f} voxels")
    return noise
