"""Smooth fiber polylines from fiber probabilities.

For each family, voxels whose probability reaches the threshold are
skeletonized and the skeleton is cut at its junctions, leaving simple chains.
Each chain is fitted with a cubic smoothing spline that must stay within
``MAX_DEVIATION`` voxels of the skeleton and inside the predicted fiber; a
chain without such a fit is split in two and each half is tried again. Parts
of the skeleton are never bridged, so a fiber interrupted in the prediction
gives several polylines.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


MIN_LENGTH = 8.0
MIN_CHAIN_VOXELS = 6
MAX_DEVIATION = 1.5
# From smoothest to nearly interpolating: a close fit follows the voxel
# staircase of the skeleton and creates false sharp turns.
SMOOTHINGS = (0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.25)
DENSE_STEP = 0.25
FAMILIES = (("V", 0), ("H", 1))
INTERSECTION = 2

Progress = Callable[[int, int], None]


def threshold_to_u8(percent: float) -> int:
    if not 0 < percent <= 100:
        raise ValueError("The threshold is a percentage in (0, 100]")
    return round(percent * 255 / 100)


def family_probability(probabilities: np.ndarray, channel: int) -> np.ndarray:
    """A family's probability plus half of the intersection probability."""
    out = np.empty(probabilities.shape[:3], dtype=np.uint8)
    for z in range(len(probabilities)):
        plane = probabilities[z, :, :, channel].astype(np.float32) + 0.5 * probabilities[z, :, :, INTERSECTION]
        out[z] = np.clip(plane, 0, 255).astype(np.uint8)
    return out


def chain_order(coords: np.ndarray) -> np.ndarray | None:
    """Order the voxels of an unbranched 26-connected chain; None for loops or clusters."""
    neighbors = cKDTree(coords).query_ball_point(coords, 1.01, p=np.inf)
    degrees = np.array([len(n) - 1 for n in neighbors])
    ends = np.flatnonzero(degrees == 1)
    if len(ends) != 2 or degrees.max() > 2:
        return None
    previous, current = -1, int(ends[0])
    ordered = [current]
    while current != ends[1]:
        following = [i for i in neighbors[current] if i != current and i != previous]
        if len(following) != 1 or len(ordered) >= len(coords):
            return None
        previous, current = current, following[0]
        ordered.append(current)
    return coords[ordered] if len(ordered) == len(coords) else None


def fit_spline(chain: np.ndarray, probability: np.ndarray, threshold: int) -> np.ndarray | None:
    """Densely sampled spline through an ordered chain, or None if no fit is acceptable.

    ``probability`` is float32 so that it can be interpolated along the curve.
    """
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    u = distance / distance[-1]
    tree = cKDTree(chain)
    upper = np.array(probability.shape) - 1
    for rms in SMOOTHINGS:
        tck, _ = splprep(chain.T, u=u, s=len(chain) * rms**2, k=3)
        if np.linalg.norm(np.asarray(splev(u, tck)).T - chain, axis=1).max() > MAX_DEVIATION:
            continue
        dense = np.asarray(splev(np.linspace(0, 1, int(np.ceil(distance[-1] / DENSE_STEP)) + 1), tck)).T
        if np.any(dense < 0) or np.any(dense > upper):
            continue
        support = probability[tuple(np.rint(dense).astype(int).T)]
        interpolated = ndi.map_coordinates(probability, dense.T, order=1, prefilter=False)
        if support.min() < threshold or interpolated.min() < max(0.0, threshold - 0.05 * 255):
            continue
        if tree.query(dense)[0].max() > MAX_DEVIATION:
            continue
        return dense
    return None


def fit_portions(chain: np.ndarray, probability: np.ndarray, threshold: int) -> Iterator[np.ndarray]:
    """Fit a chain, splitting it at half its length where no single fit is acceptable."""
    pending = [chain]
    while pending:
        portion = pending.pop()
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(portion, axis=0), axis=1))]
        if len(portion) < 4 or distance[-1] < MIN_LENGTH:
            continue
        try:
            dense = fit_spline(portion, probability, threshold)
        except (ValueError, TypeError):
            dense = None
        if dense is not None:
            yield dense
            continue
        if distance[-1] < 2 * MIN_LENGTH:
            continue
        middle = int(np.searchsorted(distance, distance[-1] / 2))
        if 3 <= middle <= len(portion) - 4:
            # Both halves keep the cut vertex.
            pending.extend([portion[middle:], portion[: middle + 1]])


def extract_family(probability: np.ndarray, threshold: int, progress: Progress | None = None) -> list[np.ndarray]:
    """Polylines, in ZYX voxels of ``probability``, of one fiber family."""
    skeleton = skeletonize(probability >= threshold, method="lee").astype(bool)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8), mode="constant") - skeleton
    junctions = skeleton & (degree > 2)
    chains = skeleton & ~ndi.binary_dilation(junctions, structure=np.ones((3, 3, 3)))
    del skeleton, degree, junctions
    labels, _ = ndi.label(chains, structure=np.ones((3, 3, 3)))
    del chains
    sizes = np.bincount(labels.ravel())
    boxes = ndi.find_objects(labels)
    candidates = [label for label in np.flatnonzero(sizes >= MIN_CHAIN_VOXELS) if label != 0]
    probability = probability.astype(np.float32)
    polylines = []
    for index, label in enumerate(candidates):
        if progress is not None and index % 100 == 0:
            progress(index, len(candidates))
        box = boxes[label - 1]
        coords = np.argwhere(labels[box] == label) + np.array([s.start for s in box])
        chain = chain_order(coords)
        if chain is None:
            continue
        for dense in fit_portions(chain, probability, threshold):
            # Half-voxel spacing, always ending at the end of the spline.
            line = dense[::2]
            if np.linalg.norm(line[-1] - dense[-1]) > 1e-6:
                line = np.concatenate([line, dense[-1:]])
            if np.linalg.norm(np.diff(line, axis=0), axis=1).sum() >= MIN_LENGTH:
                polylines.append(line)
    if progress is not None:
        progress(len(candidates), len(candidates))
    return polylines


def extract_splines(
    probabilities: np.ndarray, threshold: int, progress: Callable[[str, int, int], None] | None = None
) -> list[tuple[str, np.ndarray]]:
    """``(family, ZYX polyline)`` pairs, longest first, from ``(Z, Y, X, 3)`` probabilities.

    Channels are vertical, horizontal and intersection; ``threshold`` is on
    the 0-255 scale of the probabilities.
    """
    traces = []
    for family, channel in FAMILIES:
        report = (lambda done, total, family=family: progress(family, done, total)) if progress else None
        for line in extract_family(family_probability(probabilities, channel), threshold, report):
            traces.append((family, line))
    lengths = [np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for _, line in traces]
    return [traces[i] for i in sorted(range(len(traces)), key=lambda i: -lengths[i])]
