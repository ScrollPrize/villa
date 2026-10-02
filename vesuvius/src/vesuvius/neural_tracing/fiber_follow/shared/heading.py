"""CT seed axes and distance-based heading updates, independent of model outputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .geometry import arclength, interp_at, frame_from_heading, normalize, block_start

SEED_HEADING_POLICY = 'ct_sheet_hv_v1'
TRACE_HEADING_POLICY = 'linear12_trusted_v1'
FRAME_POLICY = 'ct_transverse_uv_v2'
# CT is scaled to [0, 1]. Require both directional evidence and enough
# transverse energy to avoid orienting a crop from numerical residue.
MIN_FRAME_ENERGY = 1e-12
MIN_FRAME_ENERGY_FRACTION = 1e-4
MIN_FRAME_GAP = .05
HEADING_SPAN = 12.0  # trace voxels, not vertices or model calls


class SeedHeadingError(ValueError):
    """The supplied seed/family does not determine a usable CT heading."""


def fiber_family(value):
    family = str(value).strip().upper()
    family = {'HORIZONTAL': 'H', 'VERTICAL': 'V'}.get(family, family)
    if family not in ('H', 'V'):
        raise SeedHeadingError('CT seed headings require an H or V fiber family')
    return family


def ct_structure_tensor(cube, center_zyx):
    """XYZ gradient tensor; native CT derivative sigma 1, integration 4."""
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
    return tensor[::-1, ::-1]  # zyx array axes -> world xyz


def ct_sheet_normal(cube, center_zyx):
    """Unrestricted sheet normal for selecting an initial H/V heading."""
    # Solve in array-axis order for the seed-heading sign convention.
    values, vectors = np.linalg.eigh(ct_structure_tensor(cube, center_zyx)[::-1, ::-1])
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


def _ct_context(vol, pos_xyz):
    start, size = normal_context(pos_xyz, vol.input_scale)
    if np.any(start < 0) or np.any(start+65 > np.asarray(vol.ct.shape)):
        raise SeedHeadingError('CT seed context crosses the volume boundary')
    center = np.asarray(pos_xyz)[::-1]*vol.input_scale-start
    return vol.ct.read(start, size), center


def ct_normal(vol, pos_xyz):
    return ct_sheet_normal(*_ct_context(vol, pos_xyz))


def ct_tensor(vol, pos_xyz):
    return ct_structure_tensor(*_ct_context(vol, pos_xyz))


def ct_seed_heading(vol, pos_xyz, family):
    """Read only CT around the unchanged seed. No presence/direction readers."""
    return sheet_heading(ct_normal(vol, pos_xyz), family)


def oriented_seed_heading(vol, pos_xyz, family, direction):
    """The tracer's seed heading: the CT sheet axis for the family, signed along ``direction``.

    Only the sign comes from ``direction`` (annotation or a simulated trace's own
    path); the axis itself is CT-only, exactly as in inference and collection.
    """
    axis = ct_seed_heading(vol, pos_xyz, family)
    return -axis if np.dot(axis, direction) < 0 else axis


def transverse_frame(tensor, heading, previous=None, *, fallback=None, diagnostics=None):
    """Estimate roll in the heading's plane; weak evidence uses a held frame.

    ``previous`` also controls eigenvector sign continuity. ``fallback`` is an
    optional anchor used only for weak evidence (the first historical slab can
    borrow the current observation's roll without changing its normal sign).
    Diagnostic source codes: 0 = CT, 1 = transported, 2 = deterministic.
    """
    h = np.asarray(heading, dtype=np.float64)
    if h.shape != (3,) or not np.isfinite(h).all() or np.linalg.norm(h) < 1e-8:
        raise SeedHeadingError('Frame heading must be a finite nonzero xyz vector')
    h = normalize(h)
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.shape != (3, 3) or not np.isfinite(tensor).all():
        raise SeedHeadingError('CT frame tensor must be a finite 3x3 matrix')
    for anchor in (previous, fallback):
        if anchor is not None and (np.shape(anchor) != (3, 3) or not np.isfinite(anchor).all()):
            raise SeedHeadingError('Previous frame must be a finite 3x3 matrix')
    base = frame_from_heading(h)
    basis = base[:, :2]
    values, vectors = np.linalg.eigh(basis.T @ tensor @ basis)
    energy = max(0., float(values[-1]))
    fraction = energy/max(float(np.trace(tensor)), MIN_FRAME_ENERGY)
    gap = max(0., float(values[-1]-values[0]))/max(energy, MIN_FRAME_ENERGY)
    reliable = (energy > MIN_FRAME_ENERGY and fraction >= MIN_FRAME_ENERGY_FRACTION
                and gap >= MIN_FRAME_GAP)
    source = 0
    if reliable:
        u = basis @ vectors[:, -1]
        sign = (u[np.argmax(np.abs(u))] if previous is None
                else u @ frame_from_heading(h, previous[:, 0])[:, 0])
        if sign < 0:
            u = -u
        frame = np.stack([u, np.cross(h, u), h], axis=1)
    else:
        anchor = previous if previous is not None else fallback
        if anchor is not None:
            hint = np.asarray(anchor)[:, 0]
            if np.linalg.norm(hint-(hint @ h)*h) < 1e-3:
                anchor = None
        source = 2 if anchor is None else 1
        frame = base if anchor is None else frame_from_heading(h, np.asarray(anchor)[:, 0])
    if diagnostics is not None:
        diagnostics.update(source=source, energy=energy, energy_fraction=fraction, gap=gap)
    return frame


def ct_frame(vol, pos, heading, previous=None, *, fallback=None, diagnostics=None):
    # Invalid input and CT I/O errors remain explicit. Only weak orientation
    # evidence falls back; initial H/V heading selection still uses ct_normal.
    return transverse_frame(ct_tensor(vol, pos), heading, previous,
                            fallback=fallback, diagnostics=diagnostics)


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
    diagnostics = {}
    reframe_item(item, ct_frame(vol, item['pos'], np.asarray(item['frame'])[:, 2], diagnostics=diagnostics))
    item['ct_frame_diagnostics'] = diagnostics
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


def heading_free_bounds(pos, crop, input_scale):
    """Native CT block holding the crop for any heading and roll about ``pos``."""
    lateral = (crop.width-1)*crop.spacing/2
    reach = np.sqrt(2*lateral**2+(max(crop.behind, crop.depth-1-crop.behind)*crop.spacing)**2)+1.
    center = np.asarray(pos, dtype=np.float64)[::-1]*input_scale
    lo = np.floor(center-reach*input_scale).astype(np.int64)-1
    hi = np.ceil(center+reach*input_scale).astype(np.int64)+2
    return lo, hi-lo


def trace_heading(path, start, held):
    """Crop heading after a commit: the trusted 12-voxel fit, else the held heading.

    Shared by tracing and every training sample source. At a trace start (or after
    an untrusted boundary) the held heading is the previous one, i.e. the seed's.
    """
    tangent = linear12_heading(path, start)
    return np.asarray(held, dtype=np.float64) if tangent is None else tangent


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
