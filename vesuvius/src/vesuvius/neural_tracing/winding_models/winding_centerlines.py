#!/usr/bin/env python3
"""Extract one ordered winding centerline per label and Z plane.

The input is a local 3-D integer Zarr array in Z,Y,X order.  Value ``-1`` is
background by default; every other value is treated as a winding label.  The
output is a sparse OME-Zarr label image with the same shape and label values,
but with each winding reduced to a one-pixel centerline on every populated Z
plane.

Every label/plane is reduced directly from its noisy pixels to one open,
ordered polyline.  Winding-shaped point sets are parameterized by angle and
interpolated from their median radius; short, elongated fragments are
parameterized along their principal axis.  Neither representation can branch,
cycle, or retrace.  Stable point indices through Z also permit optional
per-label TIFXYZ export.

The reader intentionally walks stored chunks rather than requesting whole Z
planes from Zarr.  This matters for very large sparse arrays, where a whole
plane request would perform millions of missing-chunk lookups.  Each input
chunk is decompressed once and immediately reduced to its foreground
coordinates grouped by plane and label; the dense data is discarded.  Curve
fitting runs in a process pool with one task per label and Z-chunk slab: the
fit itself only depends on that plane's points, so each task fits its planes
independently and threads the label's previous-curve orientation state through
its own Z run, which keeps results identical to a plane-by-plane sweep while
planes fit in parallel.  While one slab fits, the previous slab's curves are
rasterized and written, and the next slab is decoded.

Example
-------

.. code-block:: bash

    python -m vesuvius.neural_tracing.winding_models.winding_centerlines \
      /path/to/predictions.zarr/winding \
      /path/to/winding_centerlines.ome.zarr \
      --workers 8 --read-workers 16 \
      --tifxyz-dir /path/to/winding_centerlines_tifxyz
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import sys
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import click
import cv2
import numpy as np
import tifffile
from numcodecs import Zstd, get_codec
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from tqdm import tqdm


@dataclass(frozen=True)
class ChunkEntry:
    index: tuple[int, int, int]
    path: Path


@dataclass(frozen=True)
class Curve:
    points_yx: np.ndarray
    closed: bool


@dataclass(frozen=True)
class PreviousCurve:
    z: int
    curve: Curve


@dataclass(frozen=True)
class ExtractionParameters:
    curve_points: int
    angle_bins: int
    min_bin_points: int
    smooth_sigma: float
    min_points: int
    max_z_gap: int


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with open(temporary, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def _parse_triplet(text: str, option_name: str, cast=float) -> tuple:
    values = [cast(item.strip()) for item in text.split(',')]
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or any(not np.isfinite(value) or value <= 0 for value in values):
        raise click.UsageError(f'{option_name} must be one positive value or Z,Y,X')
    return tuple(values)


def _parse_labels(text: str | None) -> set[int] | None:
    if text is None:
        return None
    try:
        labels = {int(item.strip()) for item in text.split(',') if item.strip()}
    except ValueError as error:
        raise click.UsageError('--labels must be a comma-separated list of integers') from error
    if not labels:
        raise click.UsageError('--labels did not contain any labels')
    return labels


class LocalZarrArray:
    """Small local-filesystem Zarr v2/v3 reader optimized for stored chunks."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        v3_metadata = self.path / 'zarr.json'
        v2_metadata = self.path / '.zarray'
        if v3_metadata.exists():
            self.version = 3
            with open(v3_metadata) as stream:
                metadata = json.load(stream)
            if metadata.get('node_type') != 'array':
                raise click.UsageError(f'{self.path} is a Zarr group, not an array')
            self.shape = tuple(map(int, metadata['shape']))
            self.chunks = tuple(map(int, metadata['chunk_grid']['configuration']['chunk_shape']))
            self.fill_value = metadata.get('fill_value')
            self.order = 'C'
            self.codecs = metadata.get('codecs', [])
            self.dtype = np.dtype(metadata['data_type'])
            for codec in self.codecs:
                if codec.get('name') == 'bytes':
                    endian = codec.get('configuration', {}).get('endian')
                    if endian == 'little':
                        self.dtype = self.dtype.newbyteorder('<')
                    elif endian == 'big':
                        self.dtype = self.dtype.newbyteorder('>')
            encoding = metadata.get('chunk_key_encoding', {
                'name': 'default', 'configuration': {'separator': '/'},
            })
            self.chunk_key_name = encoding.get('name', 'default')
            self.separator = encoding.get('configuration', {}).get('separator', '/')
            self.chunk_root = self.path / ('c' if self.chunk_key_name == 'default' else '')
            self.compressor = None
            self.filters = []
        elif v2_metadata.exists():
            self.version = 2
            with open(v2_metadata) as stream:
                metadata = json.load(stream)
            self.shape = tuple(map(int, metadata['shape']))
            self.chunks = tuple(map(int, metadata['chunks']))
            self.fill_value = metadata.get('fill_value')
            self.order = metadata.get('order', 'C')
            self.dtype = np.dtype(metadata['dtype'])
            self.separator = metadata.get('dimension_separator', '.')
            self.chunk_root = self.path
            self.compressor = metadata.get('compressor')
            self.filters = metadata.get('filters') or []
            self.codecs = []
            self.chunk_key_name = 'v2'
        else:
            raise click.UsageError(f'{self.path} is not a local Zarr v2/v3 array')

        if len(self.shape) != 3 or len(self.chunks) != 3:
            raise click.UsageError(
                f'{self.path} has shape {self.shape}; expected a three-dimensional Z,Y,X array'
            )
        if not np.issubdtype(self.dtype, np.integer):
            raise click.UsageError(f'{self.path} has dtype {self.dtype}; expected an integer array')

    def stored_chunks(self) -> list[ChunkEntry]:
        entries = []
        if not self.chunk_root.exists():
            return entries
        for path in self.chunk_root.rglob('*'):
            if not path.is_file() or path.name.startswith('.'):
                continue
            relative = path.relative_to(self.chunk_root)
            try:
                if self.separator == '/':
                    parts = relative.parts
                else:
                    if len(relative.parts) != 1:
                        continue
                    parts = tuple(relative.name.split(self.separator))
                if len(parts) != 3:
                    continue
                index = tuple(map(int, parts))
            except ValueError:
                continue
            entries.append(ChunkEntry(index=index, path=path))
        entries.sort(key=lambda entry: entry.index)
        return entries

    def actual_chunk_shape(self, index: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(
            min(chunk, size - coordinate * chunk)
            for size, chunk, coordinate in zip(self.shape, self.chunks, index)
        )

    def _decode_v3(self, payload: bytes) -> bytes:
        decoded = payload
        for codec in reversed(self.codecs):
            name = codec.get('name')
            configuration = codec.get('configuration', {})
            if name == 'zstd':
                decoded = Zstd().decode(decoded)
            elif name == 'gzip':
                decoded = get_codec({'id': 'gzip', **configuration}).decode(decoded)
            elif name == 'bytes':
                pass
            else:
                raise click.ClickException(
                    f'unsupported Zarr v3 codec {name!r} in {self.path}; '
                    'bytes+zstd and bytes+gzip are supported'
                )
        return bytes(decoded)

    def _decode_v2(self, payload: bytes) -> bytes:
        decoded = payload
        if self.compressor is not None:
            decoded = get_codec(self.compressor).decode(decoded)
        if self.filters:
            raise click.ClickException(
                f'filtered Zarr v2 input is not currently supported: {self.path}'
            )
        return bytes(decoded)

    def decode_chunk(self, entry: ChunkEntry) -> tuple[ChunkEntry, bytes, tuple[int, int, int]]:
        payload = entry.path.read_bytes()
        decoded = self._decode_v3(payload) if self.version == 3 else self._decode_v2(payload)
        actual_shape = self.actual_chunk_shape(entry.index)
        actual_bytes = math.prod(actual_shape) * self.dtype.itemsize
        full_bytes = math.prod(self.chunks) * self.dtype.itemsize
        if len(decoded) == actual_bytes:
            shape = actual_shape
        elif len(decoded) == full_bytes:
            shape = self.chunks
        else:
            raise click.ClickException(
                f'{entry.path} decoded to {len(decoded)} bytes; expected '
                f'{actual_bytes} (edge chunk) or {full_bytes} (full chunk)'
            )
        return entry, decoded, tuple(map(int, shape))


def _group_block_points(
    block: np.ndarray,
    plane_numbers: np.ndarray,
    y0: int,
    x0: int,
    background: int,
    selected_labels: np.ndarray | None,
) -> list[tuple[int, int, np.ndarray]]:
    """Group a 3-D block's foreground coordinates by plane and label.

    ``plane_numbers`` maps the block's first axis to global Z planes.  Points
    within one (plane, label) group keep their row-major raster order, and
    groups are returned ordered by plane, which matches a plane-by-plane scan
    of the same data.
    """
    zz, yy, xx = np.nonzero(block != background)
    if not len(zz):
        return []
    labels = block[zz, yy, xx]
    if selected_labels is not None:
        keep = np.isin(labels, selected_labels)
        zz, yy, xx, labels = zz[keep], yy[keep], xx[keep], labels[keep]
        if not len(zz):
            return []

    order = np.lexsort((labels, zz))
    zz = zz[order]
    yy = yy[order]
    xx = xx[order]
    labels = labels[order]
    boundaries = np.flatnonzero((np.diff(zz) != 0) | (np.diff(labels) != 0))
    starts = np.r_[0, boundaries + 1]
    stops = np.r_[starts[1:], len(zz)]
    groups = []
    for start, stop in zip(starts, stops):
        points = np.column_stack((
            yy[start:stop] + y0, xx[start:stop] + x0,
        )).astype(np.int32, copy=False)
        groups.append((int(plane_numbers[zz[start]]), int(labels[start]), points))
    return groups


def _collect_chunk_points(
    reader: LocalZarrArray,
    entry: ChunkEntry,
    background: int,
    selected_labels: np.ndarray | None,
    anchor_planes: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    """Decode one stored chunk and reduce it to grouped foreground points."""
    entry, decoded, shape = reader.decode_chunk(entry)
    z0 = entry.index[0] * reader.chunks[0]
    y0 = entry.index[1] * reader.chunks[1]
    x0 = entry.index[2] * reader.chunks[2]
    valid_z = min(shape[0], reader.shape[0] - z0)
    valid_y = min(shape[1], reader.shape[1] - y0)
    valid_x = min(shape[2], reader.shape[2] - x0)
    local = anchor_planes - z0
    local = local[(local >= 0) & (local < valid_z)]
    if not len(local):
        return []
    values = np.ndarray(shape, dtype=reader.dtype, buffer=decoded, order=reader.order)
    low, high = int(local[0]), int(local[-1]) + 1
    if len(local) == high - low:
        block = values[low:high, :valid_y, :valid_x]
        plane_numbers = np.arange(z0 + low, z0 + high)
    else:
        block = values[local, :valid_y, :valid_x]
        plane_numbers = z0 + local
    return _group_block_points(block, plane_numbers, y0, x0, background, selected_labels)


class PointSlab:
    """Foreground points of one input Z-chunk slab, grouped per plane and label.

    Chunks are decompressed in parallel and immediately reduced to compact
    coordinate arrays; the dense voxel data never outlives its decode task.
    Per-group parts are kept in stored-chunk order so concatenating them
    reproduces exactly the point order of a sequential plane scan.
    """

    def __init__(
        self,
        reader: LocalZarrArray,
        entries: list[ChunkEntry],
        anchor_planes: np.ndarray,
        read_workers: int,
        background: int,
        selected_labels: np.ndarray | None,
        progress_enabled: bool,
    ):
        self._by_label: dict[int, dict[int, list[np.ndarray]]] = defaultdict(dict)
        iterator = iter(entries)
        with ThreadPoolExecutor(max_workers=read_workers) as pool:
            # Decode ahead in parallel, but consume futures in entry order so
            # concatenated point groups match a sequential chunk walk.
            pending = deque()
            for _ in range(min(read_workers * 2, len(entries))):
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(
                    _collect_chunk_points,
                    reader, entry, background, selected_labels, anchor_planes,
                ))

            with tqdm(
                total=len(entries),
                desc=f'read input z-chunk {entries[0].index[0]}',
                unit='chunk',
                leave=False,
                disable=not progress_enabled,
            ) as progress:
                while pending:
                    for z, label, points in pending.popleft().result():
                        self._by_label[label].setdefault(z, []).append(points)
                    progress.update()
                    try:
                        next_entry = next(iterator)
                    except StopIteration:
                        continue
                    pending.append(pool.submit(
                        _collect_chunk_points,
                        reader, next_entry, background, selected_labels, anchor_planes,
                    ))

    def label_runs(self) -> dict[int, list[tuple[int, list[np.ndarray]]]]:
        """Return each label's ascending-Z run of point-part lists."""
        return {
            label: sorted(planes.items())
            for label, planes in self._by_label.items()
        }


def _resample_curve(points: np.ndarray, count: int, closed: bool) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if closed:
        points = np.vstack([points, points[0]])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    points = points[keep]
    distance = distance[keep]
    if len(points) < 2 or distance[-1] <= 0:
        raise ValueError('curve has fewer than two distinct points')
    targets = np.linspace(0, distance[-1], count, endpoint=not closed)
    return interp1d(distance, points, axis=0, assume_sorted=True)(targets)


def _align_curve(
    curve: np.ndarray,
    previous: np.ndarray,
    *,
    allow_phase_shift: bool = False,
) -> np.ndarray:
    if len(curve) != len(previous):
        raise ValueError('curve alignment requires equal point counts')
    if allow_phase_shift:
        best = None
        best_cost = float('inf')
        for oriented in (curve, curve[::-1]):
            correlation = np.zeros(len(oriented), dtype=np.float64)
            for axis in range(2):
                correlation += np.fft.ifft(
                    np.conj(np.fft.fft(previous[:, axis]))
                    * np.fft.fft(oriented[:, axis])
                ).real
            shift = int(np.argmax(correlation))
            aligned = np.roll(oriented, -shift, axis=0)
            cost = float(np.mean(np.sum((aligned - previous) ** 2, axis=1)))
            if cost < best_cost:
                best_cost = cost
                best = aligned
        return np.asarray(best).copy()
    forward = np.mean(np.sum((curve - previous) ** 2, axis=1))
    reverse = np.mean(np.sum((curve[::-1] - previous) ** 2, axis=1))
    return curve if forward <= reverse else curve[::-1].copy()


# Integer-pixel neighbor offsets grouped by increasing Euclidean distance.
# The squared distances 1, 2, 4, 5, 8, 9, 10 are every representable value
# up to 10, so a point resolved in one group has no nearer neighbor.
_NEIGHBOR_GROUPS = [
    [(0, 1), (0, -1), (1, 0), (-1, 0)],
    [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    [(0, 2), (0, -2), (2, 0), (-2, 0)],
    [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)],
    [(2, 2), (2, -2), (-2, 2), (-2, -2)],
    [(0, 3), (0, -3), (3, 0), (-3, 0)],
    [(1, 3), (1, -3), (-1, 3), (-1, -3), (3, 1), (3, -1), (-3, 1), (-3, -1)],
]


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Exact nearest-neighbor distance per point for integer pixel coordinates.

    Raster points nearly always have a neighbor within a few pixels, so probing
    packed coordinate keys by increasing neighbor distance resolves almost
    every point without the O(N log N) k-d tree the general problem needs.
    The rare points with no neighbor within distance sqrt(10) fall back to one
    k-d tree query, keeping every distance identical to a full tree query.
    """
    count = len(points)
    pixels = np.asarray(np.rint(points), dtype=np.int64)
    keys = (pixels[:, 0] << 24) | (pixels[:, 1] + (1 << 20))
    sorted_keys = np.sort(keys)
    nearest = np.full(count, np.inf)
    unresolved = np.arange(count)
    for offsets in _NEIGHBOR_GROUPS:
        if not len(unresolved):
            break
        found = np.zeros(len(unresolved), dtype=bool)
        base = keys[unresolved]
        for offset_y, offset_x in offsets:
            probe = base + (offset_y << 24) + offset_x
            slot = np.searchsorted(sorted_keys, probe)
            found |= (slot < count) & (sorted_keys[np.minimum(slot, count - 1)] == probe)
        nearest[unresolved[found]] = math.hypot(*offsets[0])
        unresolved = unresolved[~found]
    if len(unresolved):
        tree = cKDTree(points)
        nearest[unresolved] = tree.query(points[unresolved], k=2)[0][:, 1]
    return nearest


def _discard_isolated_points(points: np.ndarray) -> np.ndarray:
    """Drop only unmistakably isolated raster noise before interpolation."""
    if len(points) < 4:
        return points
    nearest = _nearest_neighbor_distances(points)
    cutoff = max(2.0, 3.0 * float(np.quantile(nearest, 0.90)))
    retained = points[nearest <= cutoff]
    return retained if len(retained) >= 2 else points


def _grouped_medians(
    values: np.ndarray,
    group_indices: np.ndarray,
    group_count: int,
) -> np.ndarray:
    """Median of ``values`` per group, identical to ``np.median`` per group.

    Empty groups yield NaN.  One shared sort replaces a Python loop of
    per-group ``np.median`` calls, whose call overhead dominated extraction.
    """
    counts = np.bincount(group_indices, minlength=group_count)
    ordered = values[np.lexsort((values, group_indices))]
    offsets = np.r_[0, np.cumsum(counts)]
    medians = np.full(group_count, np.nan, dtype=np.float64)
    present = np.flatnonzero(counts > 0)
    lower = ordered[offsets[present] + (counts[present] - 1) // 2]
    upper = ordered[offsets[present] + counts[present] // 2]
    medians[present] = 0.5 * (lower + upper)
    return medians


def _principal_axis_curve(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    parameters: ExtractionParameters,
) -> np.ndarray:
    """Interpolate an elongated fragment with a monotone linear parameter."""
    normal = np.asarray([-axis[1], axis[0]])
    along = (points - center) @ axis
    across = (points - center) @ normal
    group_count = min(
        parameters.angle_bins,
        max(2, len(points) // parameters.min_bin_points),
    )
    order = np.argsort(along, kind='stable')
    groups = np.array_split(order, group_count)
    sampled_along = np.asarray([np.median(along[group]) for group in groups])
    sampled_across = np.asarray([np.median(across[group]) for group in groups])
    sampled_across = ndimage.gaussian_filter1d(
        sampled_across,
        sigma=parameters.smooth_sigma,
        mode='nearest',
    )
    ordered = (
        center
        + sampled_along[:, None] * axis
        + sampled_across[:, None] * normal
    )
    return _resample_curve(ordered, parameters.curve_points, False)


def _angular_median_curve(
    points: np.ndarray,
    center: np.ndarray,
    parameters: ExtractionParameters,
) -> np.ndarray:
    """Interpolate one radius per increasing angle as an open polyline.

    The cut is placed at the largest geometric jump between consecutive
    supported angular bins.  This handles both a genuinely unsupported angle
    interval and an open spiral whose two ends occupy adjacent angle bins at
    substantially different radii.  In particular, it must not periodically
    connect those two ends merely because the point set covers almost every
    angle.  The representation cannot branch, cycle, or retrace.
    """
    offsets = points - center
    angles = np.mod(np.arctan2(offsets[:, 0], offsets[:, 1]), 2.0 * np.pi)
    radii = np.linalg.norm(offsets, axis=1)
    bins = parameters.angle_bins
    bin_indices = np.minimum(
        (angles * bins / (2.0 * np.pi)).astype(np.intp),
        bins - 1,
    )
    counts = np.bincount(bin_indices, minlength=bins)
    positive_counts = counts[counts > 0]
    density_floor = max(
        parameters.min_bin_points,
        int(np.quantile(positive_counts, 0.10)),
    )
    reliable = np.flatnonzero(counts >= density_floor)
    if len(reliable) < 2:
        reliable = np.flatnonzero(counts > 0)
    if len(reliable) < 2:
        raise ValueError('point set occupies fewer than two parameter bins')

    bin_medians = _grouped_medians(radii, bin_indices, bins)
    median_radius = np.full(bins, np.nan, dtype=np.float64)
    median_radius[reliable] = bin_medians[reliable]

    reliable_angles = (reliable + 0.5) * (2.0 * np.pi / bins)
    reliable_points = median_radius[reliable, None] * np.column_stack((
        np.sin(reliable_angles),
        np.cos(reliable_angles),
    ))
    circular_jumps = np.linalg.norm(
        np.roll(reliable_points, -1, axis=0) - reliable_points,
        axis=1,
    )
    seam = int(np.argmax(circular_jumps))
    start = int(reliable[(seam + 1) % len(reliable)])
    relative = (reliable - start) % bins
    order = np.argsort(relative)
    known_bins = start + relative[order]
    known_radii = median_radius[reliable[order]]
    unwrapped_bins = np.arange(known_bins[0], known_bins[-1] + 1)
    sampled_radius = np.interp(unwrapped_bins, known_bins, known_radii)
    sampled_radius = ndimage.gaussian_filter1d(
        sampled_radius,
        sigma=parameters.smooth_sigma,
        mode='nearest',
    )
    sampled_angles = (unwrapped_bins + 0.5) * (2.0 * np.pi / bins)
    ordered = center + sampled_radius[:, None] * np.column_stack((
        np.sin(sampled_angles),
        np.cos(sampled_angles),
    ))
    return _resample_curve(ordered, parameters.curve_points, False)


def _interpolate_open_curve(
    points: np.ndarray,
    parameters: ExtractionParameters,
) -> np.ndarray:
    points = _discard_isolated_points(np.asarray(points, dtype=np.float64))
    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, -1]
    dimensionality = eigenvalues[0] / max(eigenvalues[-1], 1e-9)
    if dimensionality < 0.08:
        return _principal_axis_curve(points, center, principal_axis, parameters)
    return _angular_median_curve(points, center, parameters)


def _extract_curve(
    points: np.ndarray,
    previous: PreviousCurve | None,
    z: int,
    parameters: ExtractionParameters,
) -> Curve | None:
    if len(points) < parameters.min_points:
        return None
    usable_previous = (
        previous is not None and 0 < z - previous.z <= parameters.max_z_gap
    )
    previous_points = previous.curve.points_yx if usable_previous else None
    centerline = _interpolate_open_curve(points, parameters)
    if previous_points is not None:
        endpoint_gap = np.linalg.norm(centerline[-1] - centerline[0])
        previous_endpoint_gap = np.linalg.norm(previous_points[-1] - previous_points[0])
        segment_length = float(np.median(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))
        previous_segment_length = float(
            np.median(np.linalg.norm(np.diff(previous_points, axis=0), axis=1))
        )
        centerline = _align_curve(
            centerline,
            previous_points,
            allow_phase_shift=(
                endpoint_gap < 3.0 * segment_length
                and previous_endpoint_gap < 3.0 * previous_segment_length
            ),
        )
    return Curve(centerline, False)


def _fit_label_run(
    label: int,
    run: list[tuple[int, list[np.ndarray]]],
    previous: PreviousCurve | None,
    parameters: ExtractionParameters,
) -> tuple[int, list[tuple[int, Curve | None, str | None]], PreviousCurve | None]:
    """Fit one label's anchor planes in ascending Z within one slab.

    Each plane is fit from its own points; ``previous`` only orients the
    fitted polyline, so threading it through the run reproduces exactly the
    curves a global plane-by-plane sweep would produce for this label.
    """
    results = []
    for z, parts in run:
        points = np.concatenate(parts, axis=0)
        try:
            curve = _extract_curve(points, previous, z, parameters)
        except Exception as error:  # noqa: BLE001 - isolate one bad label/plane
            results.append((z, None, str(error)))
            continue
        if curve is not None:
            previous = PreviousCurve(z=z, curve=curve)
        results.append((z, curve, None))
    return label, results, previous


class OmeZarrWriter:
    """Write a sparse Zarr-v2 OME label image without read/modify/write races."""

    pyramid_levels = 6

    def __init__(
        self,
        path: Path,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        background: int,
        chunk_yx: int,
        compression_level: int,
        scale_zyx: tuple[float, float, float],
        unit: str | None,
        source: Path,
        parameters: dict,
        chunk_z: int = 128,
    ):
        self.path = path
        self.shape = shape
        self.level_shapes = [
            tuple((size + (1 << level) - 1) // (1 << level) for size in shape)
            for level in range(self.pyramid_levels)
        ]
        self.array_paths = [path / str(level) for level in range(self.pyramid_levels)]
        self.dtype = np.dtype(dtype)
        self.background = background
        self.chunk_z = chunk_z
        self.chunk_yx = chunk_yx
        self.compressor = Zstd(level=compression_level)
        self.compression_level = compression_level
        self._plane = np.full(shape[1:], background, dtype=self.dtype)
        self._buffers: list[dict[tuple[int, int], np.ndarray]] = [
            {} for _ in range(self.pyramid_levels)
        ]
        self._buffer_z_chunks: list[int | None] = [None] * self.pyramid_levels
        self.stored_chunk_count = 0
        path.mkdir(parents=True)
        for array_path in self.array_paths:
            array_path.mkdir()
        _write_json_atomic(path / '.zgroup', {'zarr_format': 2})
        axes = [{'name': axis, 'type': 'space'} for axis in 'zyx']
        if unit:
            for axis in axes:
                axis['unit'] = unit
        self.attributes = {
            'kind': 'winding_centerlines',
            'created': datetime.now(UTC).isoformat(),
            'command_line': shlex.join(sys.argv),
            'source': str(source),
            'background_value': background,
            'collision_policy': 'ascending-label first wins',
            'curve_topology': 'open',
            'pyramid': {
                'levels': self.pyramid_levels,
                'downsampling_method': 'nearest',
                'scale_factor_zyx': [2, 2, 2],
            },
            'complete': False,
            'parameters': parameters,
            'multiscales': [{
                'version': '0.4',
                'name': 'winding_centerlines',
                'axes': axes,
                'datasets': [{
                    'path': str(level),
                    'coordinateTransformations': [{
                        'type': 'scale',
                        'scale': [value * (1 << level) for value in scale_zyx],
                    }],
                } for level in range(self.pyramid_levels)],
            }],
        }
        _write_json_atomic(path / '.zattrs', self.attributes)
        for level, (array_path, level_shape) in enumerate(
            zip(self.array_paths, self.level_shapes)
        ):
            _write_json_atomic(array_path / '.zarray', {
                'zarr_format': 2,
                'shape': list(level_shape),
                'chunks': [chunk_z, chunk_yx, chunk_yx],
                'dtype': self.dtype.str,
                'compressor': {'id': 'zstd', 'level': compression_level},
                'fill_value': background,
                'order': 'C',
                'filters': None,
                'dimension_separator': '/',
            })
            array_attributes = {'_ARRAY_DIMENSIONS': ['z', 'y', 'x']}
            if level:
                array_attributes['downsampling_method'] = 'nearest'
            _write_json_atomic(array_path / '.zattrs', array_attributes)

    def _write_chunk(
        self,
        level: int,
        z_chunk: int,
        chunk_y: int,
        chunk_x: int,
        values: np.ndarray,
    ) -> None:
        destination = self.array_paths[level] / str(z_chunk) / str(chunk_y) / str(chunk_x)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes(self.compressor.encode(values.tobytes(order='C')))
        temporary = destination.with_name(f'.{destination.name}.tmp-{os.getpid()}')
        with open(temporary, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary, destination)

    def _flush_level(self, level: int) -> None:
        z_chunk = self._buffer_z_chunks[level]
        if z_chunk is None:
            return
        for (chunk_y, chunk_x), values in self._buffers[level].items():
            self._write_chunk(level, z_chunk, chunk_y, chunk_x, values)
        self._buffers[level].clear()
        self._buffer_z_chunks[level] = None

    def _select_z_chunk(self, level: int, z_chunk: int) -> None:
        current = self._buffer_z_chunks[level]
        if current is not None and current != z_chunk:
            self._flush_level(level)
        self._buffer_z_chunks[level] = z_chunk

    def write_plane(self, z: int, curves: dict[int, Curve], line_width: int) -> int:
        """Rasterize curves once, then accumulate sparse, genuinely 3-D chunks."""
        plane = self._plane
        plane.fill(self.background)
        direct_drawing = (
            self.dtype.kind == 'i' and self.dtype.itemsize <= 4
            or self.dtype.kind == 'u' and self.dtype.itemsize <= 2
        )
        for label in sorted(curves, reverse=True):
            curve = curves[label]
            xy = np.rint(curve.points_yx).astype(np.int32)[:, ::-1]
            if direct_drawing:
                cv2.polylines(
                    plane,
                    [xy.reshape(-1, 1, 2)],
                    False,
                    int(label),
                    line_width,
                    cv2.LINE_8,
                )
            else:
                ink = np.zeros(plane.shape, dtype=np.uint8)
                cv2.polylines(
                    ink, [xy.reshape(-1, 1, 2)], False,
                    1, line_width, cv2.LINE_8,
                )
                plane[ink != 0] = label

        new_chunks = 0
        for level in range(self.pyramid_levels):
            scale = 1 << level
            if z % scale:
                continue
            output_z = z // scale
            z_chunk, local_z = divmod(output_z, self.chunk_z)
            self._select_z_chunk(level, z_chunk)
            sampled = plane[::scale, ::scale]
            level_buffers = self._buffers[level]
            for y0 in range(0, sampled.shape[0], self.chunk_yx):
                chunk_y = y0 // self.chunk_yx
                for x0 in range(0, sampled.shape[1], self.chunk_yx):
                    values = sampled[y0:y0 + self.chunk_yx, x0:x0 + self.chunk_yx]
                    if not np.any(values != self.background):
                        continue
                    key = (chunk_y, x0 // self.chunk_yx)
                    output = level_buffers.get(key)
                    if output is None:
                        output = np.full(
                            (self.chunk_z, self.chunk_yx, self.chunk_yx),
                            self.background,
                            dtype=self.dtype,
                        )
                        level_buffers[key] = output
                        self.stored_chunk_count += 1
                        new_chunks += 1
                    output[local_z, :values.shape[0], :values.shape[1]] = values
        return new_chunks

    def finish(self, labels: Iterable[int], curve_count: int, _chunk_count: int) -> None:
        for level in range(self.pyramid_levels):
            self._flush_level(level)
        self.attributes.update({
            'labels': sorted(map(int, labels)),
            'curve_count': int(curve_count),
            'stored_chunk_count': int(self.stored_chunk_count),
            'complete': True,
        })
        _write_json_atomic(self.path / '.zattrs', self.attributes)


class TifxyzSpool:
    def __init__(self, directory: Path, curve_points: int, spacing: float):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.curve_points = curve_points
        self.spacing = spacing
        self.dtype = np.dtype([
            ('z', '<i4'),
            ('closed', 'u1'),
            ('padding', 'u1', (3,)),
            ('yx', '<f4', (curve_points, 2)),
        ])
        self.handles: dict[int, object] = {}

    def append(self, label: int, z: int, curve: Curve) -> None:
        handle = self.handles.get(label)
        if handle is None:
            # Handles intentionally stay open while curves stream through Z.
            handle = open(self.directory / f'label_{label}.bin', 'ab')  # noqa: SIM115
            self.handles[label] = handle
        record = np.zeros(1, dtype=self.dtype)
        record['z'][0] = z
        record['closed'][0] = curve.closed
        record['yx'][0] = curve.points_yx.astype(np.float32, copy=False)
        handle.write(record.tobytes())

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _write_label(self, spool_path: Path, output_directory: Path, source: Path) -> Path:
        label = int(spool_path.stem.split('_', 1)[1])
        records = np.fromfile(spool_path, dtype=self.dtype)
        if not len(records):
            raise ValueError(f'empty TIFXYZ spool {spool_path}')

        # Internal curves are uniformly sampled in arc length.  Resample the Z
        # direction first so the rows that will actually be exported determine
        # the final shared column count.
        source_yx = records['yx'].astype(np.float64)
        # Resample each contiguous Z run independently.  Separator rows remain
        # invalid so downstream quad construction cannot bridge a genuine gap.
        run_starts = np.r_[0, np.flatnonzero(np.diff(records['z']) > 1) + 1]
        run_stops = np.r_[run_starts[1:], len(records)]
        full_runs: list[tuple[np.ndarray, np.ndarray]] = []
        for start, stop in zip(run_starts, run_stops):
            source_z = records['z'][start:stop].astype(np.float64)
            if len(source_z) == 1:
                target_z = source_z
                sampled_yx = source_yx[start:stop]
            else:
                sample_count = max(
                    2,
                    round((source_z[-1] - source_z[0]) / self.spacing) + 1,
                )
                target_z = np.linspace(source_z[0], source_z[-1], sample_count)
                upper_rows = np.searchsorted(source_z, target_z, side='left')
                upper_rows = np.clip(upper_rows, 1, len(source_z) - 1)
                lower_rows = upper_rows - 1
                row_fraction = (
                    (target_z - source_z[lower_rows])
                    / (source_z[upper_rows] - source_z[lower_rows])
                )[:, None, None]
                sampled_yx = (
                    (1.0 - row_fraction) * source_yx[start + lower_rows]
                    + row_fraction * source_yx[start + upper_rows]
                )
            full_runs.append((target_z.astype(np.float32), sampled_yx))

        exported_lengths = np.concatenate([
            np.linalg.norm(np.diff(sampled_yx, axis=1), axis=2).sum(axis=1)
            for _, sampled_yx in full_runs
        ])
        reference_length = float(np.median(exported_lengths))
        columns = max(2, round(reference_length / self.spacing) + 1)

        def resample_columns(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
            source_positions = np.linspace(0, self.curve_points - 1, count)
            lower = np.floor(source_positions).astype(np.intp)
            upper = np.minimum(lower + 1, self.curve_points - 1)
            fraction = (source_positions - lower)[None, :, None]
            return [
                (
                    target_z,
                    (
                        (1.0 - fraction) * sampled_yx[:, lower]
                        + fraction * sampled_yx[:, upper]
                    ).astype(np.float32),
                )
                for target_z, sampled_yx in full_runs
            ]

        # Tight curves can have a much shorter Euclidean chord than their arc
        # increment.  Correct the shared count using actual exported chords.
        for _ in range(3):
            runs = resample_columns(columns)
            chord_spacing = float(np.median(np.concatenate([
                np.linalg.norm(np.diff(sampled_yx, axis=1), axis=2).ravel()
                for _, sampled_yx in runs
            ])))
            adjusted_columns = max(
                2,
                round((columns - 1) * chord_spacing / self.spacing) + 1,
            )
            if adjusted_columns == columns:
                break
            columns = adjusted_columns
        runs = resample_columns(columns)

        extra_rows = max(0, len(runs) - 1)
        rows = sum(len(target_z) for target_z, _ in runs) + extra_rows
        internally_closed = bool(np.all(records['closed']))
        x = np.full((rows, columns), -1.0, dtype=np.float32)
        y = np.full_like(x, -1.0)
        z_values = np.full_like(x, -1.0)
        output_row = 0
        for run_index, (target_z, sampled_yx) in enumerate(runs):
            if run_index:
                output_row += 1
            stop = output_row + len(target_z)
            y[output_row:stop] = sampled_yx[:, :, 0]
            x[output_row:stop] = sampled_yx[:, :, 1]
            z_values[output_row:stop] = target_z[:, None]
            output_row = stop

        name = f'label_{label}.tifxyz'
        destination = output_directory / name
        destination.mkdir(parents=True)
        tifffile.imwrite(destination / 'x.tif', x, dtype=np.float32, compression=None)
        tifffile.imwrite(destination / 'y.tif', y, dtype=np.float32, compression=None)
        tifffile.imwrite(destination / 'z.tif', z_values, dtype=np.float32, compression=None)
        valid = x >= 0
        xyz = np.column_stack([x[valid], y[valid], z_values[valid]])
        valid_right = valid[:, :-1] & valid[:, 1:]
        right_spacing = np.sqrt(
            np.diff(x, axis=1) ** 2
            + np.diff(y, axis=1) ** 2
            + np.diff(z_values, axis=1) ** 2
        )[valid_right]
        valid_down = valid[:-1] & valid[1:]
        down_spacing = np.sqrt(
            np.diff(x, axis=0) ** 2
            + np.diff(y, axis=0) ** 2
            + np.diff(z_values, axis=0) ** 2
        )[valid_down]
        _write_json_atomic(destination / 'meta.json', {
            'format': 'tifxyz',
            'type': 'seg',
            'uuid': name,
            'scale': [1.0 / self.spacing, 1.0 / self.spacing],
            'bbox': [xyz.min(axis=0).astype(float).tolist(), xyz.max(axis=0).astype(float).tolist()],
            'source': f'Gaussian winding centerlines from {source}',
            'winding_label': label,
            'curve_points': columns,
            'source_curve_points': self.curve_points,
            'grid_columns': columns,
            'grid_rows': rows,
            'closed': False,
            'internally_closed_centerlines': internally_closed,
            'topology': 'open',
            'target_spacing_voxels': self.spacing,
            'median_column_spacing_voxels': (
                float(np.median(right_spacing)) if len(right_spacing) else None
            ),
            'median_row_spacing_voxels': (
                float(np.median(down_spacing)) if len(down_spacing) else None
            ),
            'z_planes': rows - extra_rows,
            'source_z_planes': len(records),
            'invalid_gap_rows': extra_rows,
            'coordinate_order': 'ZYX values stored in z.tif, y.tif, x.tif',
        })
        return destination

    def finalize(
        self,
        output_directory: Path,
        source: Path,
        workers: int,
        progress_enabled: bool,
    ) -> list[Path]:
        self.close()
        output_directory.mkdir(parents=True)
        spool_paths = sorted(self.directory.glob('label_*.bin'))
        outputs = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._write_label, path, output_directory, source)
                for path in spool_paths
            ]
            for future in tqdm(
                futures,
                desc='write TIFXYZ labels',
                unit='label',
                disable=not progress_enabled,
            ):
                outputs.append(future.result())
        return outputs


def _prepare_destination(path: Path, overwrite: bool, description: str) -> None:
    if path.exists():
        if not overwrite:
            raise click.UsageError(f'{description} {path} already exists; pass --overwrite')
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


@dataclass
class SlabState:
    """One Z-chunk slab moving through the fit/emit pipeline."""

    plane_start: int
    block_stop: int
    anchors: list[int]
    gap_before: bool
    futures: list[Future]
    curves_by_z: dict[int, dict[int, Curve]] = field(default_factory=dict)
    errors: list[tuple[int, int, str]] = field(default_factory=list)


@click.command(context_settings={'show_default': True})
@click.argument(
    'input_array',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument('output_ome_zarr', type=click.Path(path_type=Path))
@click.option('--tifxyz-dir', type=click.Path(path_type=Path),
              help='optionally write one TIFXYZ surface per label')
@click.option('--workers', default=max(1, min(8, os.cpu_count() or 1)),
              type=click.IntRange(min=1), help='parallel curve extraction processes')
@click.option('--read-workers', default=max(1, min(16, os.cpu_count() or 1)),
              type=click.IntRange(min=1), help='parallel input chunk decompression threads')
@click.option('--tifxyz-workers', default=4, type=click.IntRange(min=1),
              help='parallel per-label TIFXYZ writers')
@click.option('--tifxyz-spacing', default=20.0, type=click.FloatRange(min=0.1),
              help='target voxel spacing in both exported TIFXYZ grid directions')
@click.option('--cache-dir', type=click.Path(file_okay=False, path_type=Path),
              help='temporary TIFXYZ spool directory; defaults to the system temp directory')
@click.option('--curve-points', default=2048, type=click.IntRange(min=32),
              help='fixed ordered samples per curve and TIFXYZ row')
@click.option('--angle-bins', default=720, type=click.IntRange(min=32),
              help='angular interpolation bins for winding-shaped point sets')
@click.option('--min-bin-points', default=2, type=click.IntRange(min=1),
              help='minimum point support for a reliable interpolation bin')
@click.option('--smooth-sigma', default=2.0, type=click.FloatRange(min=0.0),
              help='Gaussian smoothing in interpolation-bin units')
@click.option('--max-z-gap', default=2, type=click.IntRange(min=1),
              help='largest Z gap across which curve orientation is matched')
@click.option('--z-stride', default=1, type=click.IntRange(min=1),
              help='fit every Nth plane and linearly interpolate skipped planes')
@click.option('--min-points', default=32, type=click.IntRange(min=2),
              help='ignore a label on a plane when it has fewer input pixels')
@click.option('--background', default=-1, type=int, help='input/output background value')
@click.option('--labels', 'labels_text', help='optional comma-separated label subset (default: all)')
@click.option('--z-min', default=0, type=click.IntRange(min=0), help='first Z plane to process')
@click.option('--z-max', default=None, type=click.IntRange(min=1),
              help='exclusive last Z plane; defaults to input depth')
@click.option('--line-width', default=1, type=click.IntRange(min=1),
              help='centerline width in output voxels')
@click.option('--output-chunk-z', default=128, type=click.IntRange(min=1),
              help='output chunk depth')
@click.option('--output-chunk-yx', default=128, type=click.IntRange(min=16),
              help='output chunk Y/X edge')
@click.option('--compression-level', default=3, type=click.IntRange(min=1, max=22),
              help='output Zstd compression level')
@click.option('--voxel-size', default='1',
              help='OME scale as one value or Z,Y,X')
@click.option('--unit', default=None,
              help='OME spatial unit, for example micrometer; omitted means pixels')
@click.option('--overwrite', is_flag=True, help='replace existing output destinations')
@click.option('--no-progress', is_flag=True, help='disable tqdm progress bars')
def main(
    input_array: Path,
    output_ome_zarr: Path,
    tifxyz_dir: Path | None,
    workers: int,
    read_workers: int,
    tifxyz_workers: int,
    tifxyz_spacing: float,
    cache_dir: Path | None,
    curve_points: int,
    angle_bins: int,
    min_bin_points: int,
    smooth_sigma: float,
    max_z_gap: int,
    z_stride: int,
    min_points: int,
    background: int,
    labels_text: str | None,
    z_min: int,
    z_max: int | None,
    line_width: int,
    output_chunk_z: int,
    output_chunk_yx: int,
    compression_level: int,
    voxel_size: str,
    unit: str | None,
    overwrite: bool,
    no_progress: bool,
) -> None:
    """Extract centerlines from INPUT_ARRAY into OUTPUT_OME_ZARR."""
    progress_enabled = not no_progress
    reader = LocalZarrArray(input_array)
    limits = np.iinfo(reader.dtype)
    if background < limits.min or background > limits.max:
        raise click.UsageError(f'background {background} is not representable by {reader.dtype}')
    z_max = reader.shape[0] if z_max is None else z_max
    if z_min >= z_max or z_max > reader.shape[0]:
        raise click.UsageError(f'Z range [{z_min}, {z_max}) is outside [0, {reader.shape[0]})')
    selected_labels = _parse_labels(labels_text)
    if selected_labels is not None:
        selected_labels.discard(background)
    selected_array = (
        None if selected_labels is None
        else np.array(sorted(selected_labels), dtype=np.int64)
    )
    scale_zyx = _parse_triplet(voxel_size, '--voxel-size', float)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    _prepare_destination(output_ome_zarr, overwrite, 'output OME-Zarr')
    if tifxyz_dir is not None:
        _prepare_destination(tifxyz_dir, overwrite, 'TIFXYZ directory')

    parameter_metadata = {
        'curve_points': curve_points,
        'interpolation': 'angular-median-or-principal-axis',
        'angle_bins': angle_bins,
        'min_bin_points': min_bin_points,
        'smooth_sigma': smooth_sigma,
        'max_z_gap': max_z_gap,
        'z_stride': z_stride,
        'min_points': min_points,
        'z_range': [z_min, z_max],
        'line_width': line_width,
        'output_topology': 'open',
        'tifxyz_spacing': tifxyz_spacing,
        'output_chunk_zyx': [output_chunk_z, output_chunk_yx, output_chunk_yx],
    }
    writer = OmeZarrWriter(
        output_ome_zarr,
        reader.shape,
        reader.dtype,
        background,
        output_chunk_yx,
        compression_level,
        scale_zyx,
        unit,
        input_array.resolve(),
        parameter_metadata,
        output_chunk_z,
    )
    extraction_parameters = ExtractionParameters(
        curve_points=curve_points,
        angle_bins=angle_bins,
        min_bin_points=min_bin_points,
        smooth_sigma=smooth_sigma,
        min_points=min_points,
        max_z_gap=max(max_z_gap, z_stride),
    )

    entries_by_z_chunk: dict[int, list[ChunkEntry]] = defaultdict(list)
    for entry in reader.stored_chunks():
        block_start = entry.index[0] * reader.chunks[0]
        block_stop = min(reader.shape[0], block_start + reader.chunks[0])
        if block_stop > z_min and block_start < z_max:
            entries_by_z_chunk[entry.index[0]].append(entry)
    if not entries_by_z_chunk:
        writer.finish([], 0, 0)
        click.echo('No stored input chunks intersect the requested Z range.')
        return

    spool_temporary = None
    spool = None
    if tifxyz_dir is not None:
        spool_temporary = tempfile.TemporaryDirectory(
            prefix='winding-centerline-tifxyz-',
            dir=None if cache_dir is None else str(cache_dir),
        )
        spool = TifxyzSpool(Path(spool_temporary.name), curve_points, tifxyz_spacing)

    previous: dict[int, PreviousCurve] = {}
    labels_seen: set[int] = set()
    curve_count = 0
    output_chunk_count = 0
    failed_curves = 0
    cv2.setNumThreads(1)

    # Anchor planes are decided ahead of fitting; the emit side below replays
    # the same sequence, so both agree on which planes interpolate.
    submit_cursor = z_min
    submit_last_anchor: int | None = None

    def plan_slab(z_chunk: int) -> tuple[int, int, list[int], bool]:
        nonlocal submit_cursor, submit_last_anchor
        block_start = z_chunk * reader.chunks[0]
        block_stop = min(reader.shape[0], block_start + reader.chunks[0], z_max)
        plane_start = max(z_min, block_start)
        gap_before = plane_start > submit_cursor
        if gap_before:
            submit_last_anchor = None
        anchors = []
        for z in range(plane_start, block_stop):
            if (
                submit_last_anchor is None
                or z == z_max - 1
                or (z - z_min) % z_stride == 0
            ):
                anchors.append(z)
                submit_last_anchor = z
        submit_cursor = block_stop
        return plane_start, block_stop, anchors, gap_before

    # Emit-side state.
    last_anchor_z: int | None = None
    last_anchor_curves: dict[int, Curve] = {}
    pending_z: list[int] = []
    cursor = z_min

    def emit_plane(z: int, curves: dict[int, Curve]) -> None:
        nonlocal curve_count, output_chunk_count
        for label, curve in curves.items():
            labels_seen.add(label)
            curve_count += 1
            if spool is not None:
                spool.append(label, z, curve)
        output_chunk_count += writer.write_plane(z, curves, line_width)
        progress.update()
        progress.set_postfix(
            labels=len(labels_seen),
            curves=curve_count,
            failed=failed_curves,
            refresh=False,
        )

    def interpolate_pending(current_z: int, current_curves: dict[int, Curve]) -> None:
        if last_anchor_z is None:
            return
        span = current_z - last_anchor_z
        for pending_plane in pending_z:
            alpha = (pending_plane - last_anchor_z) / span
            interpolated: dict[int, Curve] = {}
            for label in last_anchor_curves.keys() | current_curves.keys():
                first = last_anchor_curves.get(label)
                second = current_curves.get(label)
                if first is None:
                    interpolated[label] = second
                elif second is None:
                    interpolated[label] = first
                elif first.closed != second.closed:
                    interpolated[label] = first if alpha < 0.5 else second
                else:
                    points = (1.0 - alpha) * first.points_yx + alpha * second.points_yx
                    interpolated[label] = Curve(points, first.closed)
            emit_plane(pending_plane, interpolated)
        pending_z.clear()

    def harvest_slab(slab: SlabState) -> None:
        for future in slab.futures:
            label, results, state = future.result()
            if state is not None:
                previous[label] = state
            for z, curve, message in results:
                if message is not None:
                    slab.errors.append((z, label, message))
                elif curve is not None:
                    slab.curves_by_z.setdefault(z, {})[label] = curve

    def emit_slab(slab: SlabState) -> None:
        nonlocal last_anchor_z, last_anchor_curves, cursor, failed_curves
        for z, label, message in sorted(slab.errors):
            failed_curves += 1
            tqdm.write(f'z={z} label={label}: extraction failed: {message}')
        if slab.gap_before:
            if pending_z and last_anchor_z is not None:
                for pending_plane in pending_z:
                    emit_plane(pending_plane, last_anchor_curves)
                pending_z.clear()
            last_anchor_z = None
            last_anchor_curves = {}
            progress.update(slab.plane_start - cursor)
            cursor = slab.plane_start
        anchor_set = set(slab.anchors)
        for z in range(slab.plane_start, slab.block_stop):
            if z not in anchor_set:
                pending_z.append(z)
                cursor = z + 1
                continue
            curves = slab.curves_by_z.get(z, {})
            interpolate_pending(z, curves)
            emit_plane(z, curves)
            last_anchor_z = z
            last_anchor_curves = curves
            cursor = z + 1

    try:
        with ProcessPoolExecutor(max_workers=workers) as fit_pool, tqdm(
            total=z_max - z_min,
            desc='extract centerlines',
            unit='plane',
            disable=not progress_enabled,
        ) as progress:
            # Pipeline slabs: while the pool fits slab N, slab N-1 is
            # rasterized and written and slab N+1 is decoded.  Fits for slab
            # N+1 are only submitted after slab N completes because each
            # label's orientation state threads through consecutive slabs.
            pending_slab: SlabState | None = None
            for z_chunk in sorted(entries_by_z_chunk):
                plane_start, block_stop, anchors, gap_before = plan_slab(z_chunk)
                point_slab = PointSlab(
                    reader,
                    entries_by_z_chunk[z_chunk],
                    np.asarray(anchors, dtype=np.int64),
                    read_workers,
                    background,
                    selected_array,
                    progress_enabled,
                )
                if pending_slab is not None:
                    harvest_slab(pending_slab)
                futures = [
                    fit_pool.submit(
                        _fit_label_run,
                        label,
                        run,
                        previous.get(label),
                        extraction_parameters,
                    )
                    for label, run in sorted(point_slab.label_runs().items())
                ]
                current_slab = SlabState(
                    plane_start, block_stop, anchors, gap_before, futures,
                )
                if pending_slab is not None:
                    emit_slab(pending_slab)
                pending_slab = current_slab
            if pending_slab is not None:
                harvest_slab(pending_slab)
                emit_slab(pending_slab)
            if pending_z and last_anchor_z is not None:
                for pending_plane in pending_z:
                    emit_plane(pending_plane, last_anchor_curves)
                pending_z.clear()
            if cursor < z_max:
                progress.update(z_max - cursor)

        writer.finish(labels_seen, curve_count, output_chunk_count)
        if spool is not None:
            spool.finalize(
                tifxyz_dir,
                input_array.resolve(),
                tifxyz_workers,
                progress_enabled,
            )
    finally:
        if spool is not None:
            spool.close()
        if spool_temporary is not None:
            spool_temporary.cleanup()

    click.echo(
        f'Wrote {curve_count} curves for {len(labels_seen)} labels to {output_ome_zarr} '
        f'({output_chunk_count} stored chunks, {failed_curves} failed extractions).'
    )
    if tifxyz_dir is not None:
        click.echo(f'Wrote per-label TIFXYZ surfaces to {tifxyz_dir}.')


if __name__ == '__main__':
    main()
