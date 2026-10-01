"""CT seed axes and distance-based heading updates, independent of model outputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import arclength, interp_at, frame_from_heading, normalize, block_start

SEED_HEADING_POLICY = 'ct_sheet_hv_v1'
TRACE_HEADING_POLICY = 'linear12_trusted_v1'
FRAME_POLICY = 'ct_normal_uv_v1'
HEADING_SPAN = 12.0  # trace voxels, not vertices or model calls


class SeedHeadingError(ValueError):
    """The supplied seed/family does not determine a usable CT heading."""


def fiber_family(value):
    family = str(value).strip().upper()
    family = {'HORIZONTAL': 'H', 'VERTICAL': 'V'}.get(family, family)
    if family not in ('H', 'V'):
        raise SeedHeadingError('CT seed headings require an H or V fiber family')
    return family


def ct_sheet_normal(cube, center_zyx):
    """Unsigned xyz sheet normal; native CT derivative sigma 1, integration 4."""
    image = np.asarray(cube, dtype=np.float64) / 255.
    center = np.asarray(center_zyx, dtype=np.float64)
    if image.ndim != 3 or center.shape != (3,) or not np.isfinite(image).all() or not np.isfinite(center).all():
        raise SeedHeadingError('CT seed context must be a finite 3D image and center')
    gradients = []
    for axis in range(3):
        order = [0, 0, 0]
        order[axis] = 1
        gradients.append(gaussian_filter(image, 1., order=order, mode='nearest', truncate=3.))
    weights = [np.exp(-.5*((np.arange(n)-c)/4.)**2) for n, c in zip(image.shape, center)]
    weight = weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]
    weight /= weight.sum()
    tensor = np.array([[np.sum(weight*a*b) for b in gradients] for a in gradients])
    values, vectors = np.linalg.eigh(tensor)
    if values[-1] <= 1e-12 or values[-1]-values[-2] <= 1e-6*values[-1]:
        raise SeedHeadingError('CT seed context has no identifiable sheet normal')
    return vectors[::-1, -1]  # zyx array axes -> world xyz


def sheet_heading(normal, family):
    family = fiber_family(family)
    z = np.array([0., 0., 1.])
    axis = z-(z@normal)*normal if family == 'V' else np.cross(z, normal)
    size = float(np.linalg.norm(axis))
    if size < 1e-3:
        raise SeedHeadingError('H/V is ambiguous for this sheet orientation')
    return axis/size


def ct_sheet_heading(cube, center_zyx, family):
    return sheet_heading(ct_sheet_normal(cube, center_zyx), family)


def normal_context(pos_xyz, input_scale):
    """Native CT bounds for normal estimation, also used by remote prefetch."""
    pos = np.asarray(pos_xyz, dtype=np.float64)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise SeedHeadingError('Seed position must be a finite xyz vector')
    center = pos[::-1]*input_scale
    start = np.floor(center).astype(np.int64)-32
    return start, np.array([65, 65, 65])


def ct_normal(vol, pos_xyz):
    start, size = normal_context(pos_xyz, vol.input_scale)
    if np.any(start < 0) or np.any(start+65 > np.asarray(vol.ct.shape)):
        raise SeedHeadingError('CT seed context crosses the volume boundary')
    center = np.asarray(pos_xyz)[::-1]*vol.input_scale-start
    return ct_sheet_normal(vol.ct.read(start, size), center)


def ct_seed_heading(vol, pos_xyz, family):
    """Read only CT around the unchanged seed. No presence/direction readers."""
    return sheet_heading(ct_normal(vol, pos_xyz), family)


def normal_frame(heading, normal, previous=None):
    """u crosses the sheet; v lies in it. Transport the unsigned normal's sign."""
    h = np.asarray(heading, dtype=np.float64)
    if h.shape != (3,) or not np.isfinite(h).all() or np.linalg.norm(h) < 1e-8:
        raise SeedHeadingError('Frame heading must be a finite nonzero xyz vector')
    h = normalize(h)
    normal = np.asarray(normal, dtype=np.float64)
    u = normal-(normal @ h)*h
    support = float(np.linalg.norm(u))
    if not np.isfinite(support) or support < .2:
        if previous is None:
            raise SeedHeadingError('CT normal nearly parallels the heading')
        return frame_from_heading(h, previous[:, 0])
    u /= support
    sign = u[np.argmax(np.abs(u))] if previous is None else u @ frame_from_heading(h, previous[:, 0])[:, 0]
    if sign < 0:
        u = -u
    return np.stack([u, np.cross(h, u), h], axis=1)


def ct_frame(vol, pos, heading, previous=None):
    try:
        normal = ct_normal(vol, pos)
    except SeedHeadingError:
        if previous is None:
            raise
        # A weak/degenerate CT observation cannot invent a new roll. Retain the
        # last CT-established roll while transporting it to the new heading.
        return frame_from_heading(heading, previous[:, 0])
    return normal_frame(heading, normal, previous)


def reframe_item(item, frame):
    """Rotate every local geometric quantity while preserving world geometry."""
    rotation = np.asarray(item['frame']).T @ frame
    for key in ('hist_local', 'gt_history', 'fut_local', 'end_local',
                'candidate_points', 'identity_curve'):
        if key in item:
            item[key] = np.asarray(item[key]) @ rotation
    for key in ('plane_ab', 'dense_ab'):
        if key in item:
            item[key] = np.asarray(item[key]) @ rotation[:2, :2]
    item['frame'] = frame


def orient_item(item, vol):
    """Resolve synthetic geometry once; recorded tracing frames are immutable."""
    if 'frame_policy' in item:
        if item['frame_policy'] != FRAME_POLICY:
            raise ValueError('Unsupported crop frame policy; recollect replay')
        return
    reframe_item(item, ct_frame(vol, item['pos'], np.asarray(item['frame'])[:, 2]))
    item['frame_policy'] = FRAME_POLICY


def frame_prefetch_bounds(item, crop, input_scale):
    """Cover every roll until CT is available, plus the normal's CT context."""
    from .data import tight_block
    if item.get('frame_policy') == FRAME_POLICY:
        yield tight_block(item['pos'], item['frame'], crop, input_scale)
    else:
        # block_start encloses the crop's circumsphere, independent of u/v.
        lo = np.floor(block_start(item['pos'], item['frame'], crop)*input_scale).astype(np.int64)-1
        hi = np.ceil((block_start(item['pos'], item['frame'], crop)+crop.block_size)*input_scale).astype(np.int64)+2
        yield lo, hi-lo
        yield normal_context(item['pos'], input_scale)


def linear12_heading(path, start=0):
    """Fit the last 12 trusted arclength voxels; None means hold the full frame.

    ``start`` excludes all older/untrusted vertices. Equal-arclength sampling
    makes a short commit or a densely interpolated segment no more influential
    than the same geometry committed in a different number of calls.
    """
    if start < 0 or start >= len(path)-1:
        return None
    distance = 0.
    first = len(path)-1
    while first > start and distance < HEADING_SPAN:
        distance += float(np.linalg.norm(np.asarray(path[first])-path[first-1]))
        first -= 1
    if distance < HEADING_SPAN-1e-8:
        return None
    points = np.asarray(path[first:], dtype=np.float64)
    if not np.isfinite(points).all():
        return None
    arc = arclength(points)
    times = np.arange(13., dtype=np.float64)
    samples = interp_at(points, arc, np.maximum(arc[-1]-HEADING_SPAN+times, 0))-points[-1]
    # Free intercept: fit direction without forcing the line through the head.
    direction = (times-times.mean()) @ samples
    size = float(np.linalg.norm(direction))
    return direction/size if size > 1e-8 else None
