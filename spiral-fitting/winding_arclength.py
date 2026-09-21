"""Measured horizontal arclength for a shared, continuous winding chart.

Angle orders the samples; it does not supply their distance. Patch triangles
measure the tangent to a constant-Z contour. Integrating its length per turn
gives a monotone horizontal coordinate shared by every source. A median across
height makes this a representative contour, not a full 2D isometric flattening.
"""
import numpy as np
from winding_topology import source_quad_mask


def source_grid(payload, source):
    start = source['byte_offset'] // 40
    return payload[start:start + source['rows'] * source['cols']].reshape(
        source['rows'], source['cols'], 5)


def _sample_indices(mask, limit):
    indices = np.flatnonzero(mask)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]
    return indices


def contour_samples(grid, turn_width, limit=256, source=None):
    """Return (turn, height, ds/dturn) from supported source triangles.

    Using both triangles includes either side of a quad's diagonal. No triangle
    crosses a missing source vertex. Synthetic fiber ribbons never enter here.
    """
    if min(grid.shape[:2]) < 2:
        return np.empty((0, 3))
    samples = []
    for a, b, c in ((grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1]),
                    (grid[1:, 1:], grid[1:, :-1], grid[:-1, 1:])):
        valid = source_quad_mask(grid, source)
        idx = _sample_indices(valid, limit)
        a, b, c = (v.reshape(-1, 5)[idx] for v in (a, b, c))
        da, db = b - a, c - a
        ta, tb = da[:, 3] / turn_width, db[:, 3] / turn_width
        za, zb = da[:, 2], db[:, 2]
        determinant = ta * zb - tb * za
        scale = np.hypot(ta, tb) * np.hypot(za, zb)
        valid = np.abs(determinant) > np.maximum(1e-12, scale * 1e-8)
        tangent = (da[valid, :3] * zb[valid, None]
                   - db[valid, :3] * za[valid, None]) / determinant[valid, None]
        speed = np.linalg.norm(tangent, axis=1)
        center = (a[valid] + b[valid] + c[valid]) / 3
        good = np.isfinite(speed) & (speed > 0)
        samples.append(np.column_stack((center[good, 3] / turn_width,
                                        center[good, 2], speed[good])))
    return np.concatenate(samples)


def fiber_samples(grid, turn_width, limit=1024):
    line = grid[1]
    delta = np.diff(line, axis=0)
    dt = delta[:, 3] / turn_width
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    # Steep fibers do not measure a horizontal contour reliably. These samples
    # fill only angular bins without patch support.
    valid = (np.abs(dt) > 1e-10) & (horizontal > 0) & (np.abs(delta[:, 2]) <= horizontal)
    idx = _sample_indices(valid, limit)
    center = (line[:-1] + line[1:])[idx] / 2
    return np.column_stack((center[:, 3] / turn_width, center[:, 2],
                            horizontal[idx] / np.abs(dt[idx])))


def _profile(samples, start, count, bins_per_turn, height_bin):
    """One median vote per height band, then one median per angular bin."""
    result = np.full(count, np.nan)
    if not len(samples):
        return result
    angular = np.floor(samples[:, 0] * bins_per_turn).astype(np.int64) - start
    height = np.floor(samples[:, 1] / height_bin).astype(np.int64)
    order = np.lexsort((height, angular))
    angular, height, speed = angular[order], height[order], samples[order, 2]
    cuts = np.r_[0, np.flatnonzero((np.diff(angular) != 0) | (np.diff(height) != 0)) + 1, len(order)]
    medians = np.array([np.median(speed[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    bins = angular[cuts[:-1]]
    cuts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(bins)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        result[bins[a]] = np.median(medians[a:b])
    return result


def measure_arclength(payload, sources, turn_width, bounds, bins_per_turn=128, height_bin=256.):
    patch, fiber = [], []
    for source in sources:
        if source.get('quarantined', False):
            continue
        grid = source_grid(payload, source)
        target = fiber if source['fiber'] else patch
        target.append(fiber_samples(grid, turn_width) if source['fiber']
                      else contour_samples(grid, turn_width, source=source))
    start = int(np.floor(bounds[0][0] / turn_width * bins_per_turn))
    end = int(np.ceil(bounds[1][0] / turn_width * bins_per_turn))
    count = max(1, end - start)
    patches = _profile(np.concatenate(patch) if patch else np.empty((0, 3)),
                       start, count, bins_per_turn, height_bin)
    fibers = _profile(np.concatenate(fiber) if fiber else np.empty((0, 3)),
                      start, count, bins_per_turn, height_bin)
    speed = np.where(np.isfinite(patches), patches, fibers)
    supported = np.isfinite(speed)
    if not supported.any():
        raise ValueError('No measured horizontal arclength in accepted geometry')
    # Interpolate uncovered angular intervals; do not invent a universal turn
    # width. Record their count so inferred distances are explicit.
    speed = np.interp(np.arange(count), np.flatnonzero(supported), speed[supported])
    turns = np.arange(start, start + count + 1, dtype=float) / bins_per_turn
    arc = np.r_[0., np.cumsum(speed / bins_per_turn)]
    arc -= np.interp(0., turns, arc)
    integers = np.arange(int(np.ceil(turns[0])), int(np.floor(turns[-1])) + 1)
    widths = np.diff(np.interp(integers, turns, arc))
    return dict(method='measured_contour_arclength', turns=turns.tolist(), arclength=arc.tolist(),
                bins_per_turn=bins_per_turn, height_bin=height_bin,
                patch_bins=int(np.isfinite(patches).sum()),
                fiber_only_bins=int((~np.isfinite(patches) & np.isfinite(fibers)).sum()),
                interpolated_bins=int((~supported).sum()),
                winding_numbers=integers[:-1].tolist(), winding_widths=widths.tolist(),
                height_reduction='median of contour speeds across supported height bands')


def apply_arclength(payload, sources, turn_width, profile):
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    for source in sources:
        if source.get('quarantined', False):
            continue
        grid = source_grid(payload, source).reshape(-1, 5)
        valid = grid[:, 0] != -1
        grid[valid, 3] = np.interp(grid[valid, 3] / turn_width,
                                 profile['turns'], profile['arclength'])
        low = np.minimum(low, grid[valid, 3:].min(axis=0))
        high = np.maximum(high, grid[valid, 3:].max(axis=0))
    return [low.tolist(), high.tolist()]
