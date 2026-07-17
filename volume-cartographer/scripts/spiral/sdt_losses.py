"""Surf-SDT-driven dense losses: the ``phase`` dense-spacing bundle.

All components are driven by one capped signed-distance store of the
binarized surface prediction (positive outside, negative inside, encoded
0 = no-data). The ``phase`` bundle executes four independently weighted
components (a zero sub-weight disables one component, it does not create
another mode):

- soft-sequence phase registration: complete SDT bands detected on an
  extended ray are aligned to the ordered modeled-winding sequence with a
  differentiable monotonic pair-HMM (``soft_alignment``), and the projected,
  local-gap-normalized Huber residual is applied through the resulting match
  probabilities;
- crossing count: a soft sheet-crossing count over the central modeled
  interval, residual |count - m|, sharing the phase rays' central samples;
- native minimum spacing: a squared log-gap hinge as the exact-collapse
  recovery barrier (``get_min_spacing_loss``);
- SDT attachment: smooth-L1 of the exterior distance at dense points on
  fitted winding surfaces (``get_dense_attachment_loss``).

Observation extraction is detached (rays, bands, reference gaps); only the
sequence alignment and the modeled winding target positions are
differentiable. The count loss keeps a detached endpoint support gate with a
nominal-mass denominator floor, so it scales down (to a defined zero) when
support is scarce instead of collapsing back to an ordinary mean.

The legacy ``grad_mag`` density-integral objective lives unchanged in
``losses.iter_lasagna_losses``.
"""

import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

from ddp_helpers import get_world_size, is_distributed
from loss_maps import diagnostics_enabled, record_loss_samples
from soft_alignment import soft_align_sequences


# A sample is valid when the total trilinear weight of its valid corners is at
# least this mass (weight mass, not corner count: four valid corners can carry
# near-zero weight when the sample point sits in the invalid octant).
MIN_VALID_CORNER_WEIGHT_MASS = 0.5

_CORNER_OFFSETS = tuple(itertools.product((0, 1), repeat=3))

# Printed diagnostic state for the support-mass floor (metric is emitted every
# step regardless; the print fires only on transitions to avoid spam).
_floor_was_active = False


def fitted_winding_domain(outer_winding_idx):
    # Today the fitted domain is [1, outer_winding_idx - 1]: winding 1 is the
    # innermost by convention and there is no explicit core-radius bound.
    # Samplers must reference this helper rather than the constants so an
    # explicit inner bound can be introduced in one place later.
    return 1, int(outer_winding_idx) - 1


def _sample_windings_by_circumference(
    lowest, highest, num_samples, device, generator=None,
):
    """Integer windings in [lowest, highest] with probability proportional to
    winding circumference (weight k + 0.5). ``highest`` may be a per-sample
    tensor. Uses the analytic inverse CDF: cum-mass to k is ((k+1)^2 - a^2)/2."""
    a = float(lowest)
    b = torch.as_tensor(highest, device=device, dtype=torch.float32)
    total_mass = ((b + 1.0) ** 2 - a * a) / 2.0
    u = torch.rand(num_samples, device=device, generator=generator)
    k = torch.ceil(torch.sqrt(2.0 * u * total_mass + a * a) - 1.0)
    return torch.minimum(k.clamp(min=a), b)


def decode_sdt_values(values_u8, sdt_volume):
    """Decode encoded uint8 values per the store's own attributes.

    kind 'sdt': sd = (value - offset) * unit working voxels.
    kind 'surf': centered raw value (value - 128); no distance semantics.
    """
    centered = values_u8.to(torch.float32) - float(sdt_volume['offset'])
    if sdt_volume['kind'] == 'sdt':
        return centered * float(sdt_volume['unit'])
    return centered


def _sampling_constants(sdt_volume, device, dtype):
    # Constant tensors (scale, z-origin shift, corner offsets, shape) are
    # rebuilt on every call otherwise - a fixed host-alloc + H2D cost on a
    # per-step hot path - so cache them on the volume dict per device.
    key = (str(device), str(dtype))
    cache = sdt_volume.setdefault('_sampling_constants', {})
    if key not in cache:
        cache[key] = (
            torch.as_tensor(sdt_volume['scale_zyx'], device=device, dtype=dtype),
            torch.tensor([float(sdt_volume['z_origin']), 0.0, 0.0],
                         device=device, dtype=dtype),
            torch.tensor(_CORNER_OFFSETS, device=device, dtype=torch.long),
            torch.tensor(sdt_volume['shape'], device=device, dtype=torch.long),
        )
    return cache[key]


def sample_sdt_trilinear(sdt_volume, points_working_zyx):
    """Differentiable trilinear sample at working-voxel coordinates (N, 3).

    Corner indices are detached integers; the fractional interpolation weights
    keep their gradient with respect to the mapped sample positions. Corner
    validity = in-bounds and encoded nonzero (0 = no-data); weights are
    renormalised over the valid corners and a sample is valid when the valid
    weight mass is >= MIN_VALID_CORNER_WEIGHT_MASS. Never test the interpolated
    value against 0 for validity.

    Returns (value, sample_valid, corner_values_u8): decoded interpolated
    value (garbage where invalid), boolean validity, and the raw corners for
    diagnostics.
    """
    device = points_working_zyx.device
    scale, origin_shift, offsets, shape = _sampling_constants(
        sdt_volume, device, points_working_zyx.dtype)
    grid = points_working_zyx / scale - origin_shift

    base = grid.detach().floor().long()
    frac = grid - base.to(grid.dtype)

    corners = base[:, None, :] + offsets[None, :, :]  # (N, 8, 3)
    in_bounds = ((corners >= 0) & (corners < shape)).all(dim=-1)
    clamped = torch.minimum(corners.clamp(min=0), shape - 1)

    if sdt_volume['backend'] == 'mmap':
        values_u8 = sdt_volume['store'].gather(
            clamped.reshape(-1, 3), device).reshape(clamped.shape[:2])
    else:  # dense torch uint8 tensor (tests / tiny ROIs)
        volume = sdt_volume['volume']
        values_u8 = volume[clamped[..., 0], clamped[..., 1], clamped[..., 2]]

    corner_valid = in_bounds & (values_u8 != 0)
    corner_frac = torch.where(
        offsets[None, :, :].bool(), frac[:, None, :], 1.0 - frac[:, None, :])
    weights = corner_frac.prod(dim=-1)  # (N, 8), differentiable wrt positions
    valid_f = corner_valid.to(weights.dtype)
    mass = (weights * valid_f).sum(dim=-1)
    normalised = weights * valid_f / mass.clamp(min=1e-12)[:, None]
    value = (normalised * decode_sdt_values(values_u8, sdt_volume)).sum(dim=-1)
    sample_valid = mass.detach() >= MIN_VALID_CORNER_WEIGHT_MASS
    return value, sample_valid, values_u8


def _inside_indicator(field_value, s_count_wv):
    return torch.sigmoid(-field_value / s_count_wv)


def _saturation_threshold(sdt_volume):
    # Half an encoding step below the cap: decoded values quantize in steps of
    # `unit` working voxels, so the margin must scale with it.
    return float(sdt_volume['cap']) - 0.5 * float(sdt_volume['unit'])


def sdt_sample_fractions(field_value, sample_valid, sdt_volume):
    """No-data / inside / outside-live / saturated fractions for diagnostics."""
    with torch.no_grad():
        total = max(1, field_value.numel())
        invalid = ~sample_valid
        if sdt_volume['kind'] == 'sdt' and sdt_volume.get('cap') is not None:
            saturated = sample_valid & (field_value.abs() >= _saturation_threshold(sdt_volume))
        else:
            saturated = torch.zeros_like(sample_valid)
        inside = sample_valid & (field_value < 0) & ~saturated
        live = sample_valid & (field_value >= 0) & ~saturated
        counts = torch.stack(
            [invalid.sum(), inside.sum(), live.sum(), saturated.sum()]).tolist()
        return {
            'nodata_fraction': counts[0] / total,
            'inside_fraction': counts[1] / total,
            'outside_live_fraction': counts[2] / total,
            'saturated_fraction': counts[3] / total,
        }


def sample_pair_polylines(
    slice_to_spiral_transform, dr_per_winding, phase_start, phase_end, theta, z,
    cfg, *, no_grad=False,
):
    """Map fully sampled packed rays between modeled winding coordinates.

    Step allocation is independent for every at-most-one-winding constituent
    interval. ``phase_start``/``phase_end`` may include fractional extension
    padding. Every ray is either represented in full or marked ``too_long``;
    over-budget rays carry only a cheap diagnostic chord and must never be
    scored by a caller.
    """
    device = z.device
    dtype = phase_start.dtype
    num_pairs = phase_start.shape[0]
    dr = dr_per_winding.detach()
    target_step = float(cfg['dense_spacing_target_step_wv'])
    max_step = float(cfg['dense_spacing_max_step_wv'])
    max_steps = int(cfg['dense_spacing_max_steps'])
    oversample = float(cfg['dense_spacing_step_oversample'])

    span = phase_end - phase_start
    if bool((span <= 0).any()):
        raise ValueError('phase ray endpoints must be monotonically outward')
    max_segments = max(1, int(torch.ceil(span.max()).item()))
    segment_slot = torch.arange(max_segments, device=device)
    breakpoint_slot = torch.arange(max_segments + 1, device=device)
    breakpoint_phase = phase_start[:, None] + torch.minimum(
        breakpoint_slot[None, :].to(dtype), span[:, None])
    active_segment = segment_slot[None, :].to(dtype) < span[:, None]

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    phase_theta = theta / (2 * np.pi)
    breakpoint_radii = (breakpoint_phase + phase_theta[:, None]) * dr
    probe_pair_id = torch.arange(num_pairs, device=device).repeat_interleave(
        max_segments + 1)
    probe_radii = breakpoint_radii.reshape(-1)
    probe_spiral = torch.stack([
        z[probe_pair_id], sin_t[probe_pair_id] * probe_radii,
        cos_t[probe_pair_id] * probe_radii,
    ], dim=-1)
    with torch.no_grad():
        breakpoint_scroll = slice_to_spiral_transform.inv(probe_spiral).reshape(
            num_pairs, max_segments + 1, 3)
        segment_lengths = torch.diff(breakpoint_scroll, dim=1).norm(dim=-1)
    segment_steps = torch.where(
        active_segment,
        torch.ceil(segment_lengths * oversample / target_step).long().clamp(min=2),
        torch.zeros_like(segment_lengths, dtype=torch.long),
    )
    steps_needed = segment_steps.sum(dim=1)
    too_long = steps_needed > max_steps

    emitted_steps = torch.where(
        too_long[:, None], torch.zeros_like(segment_steps), segment_steps)
    emitted_steps[:, 0] = torch.where(
        too_long, torch.full_like(steps_needed, 2), emitted_steps[:, 0])
    n = emitted_steps.sum(dim=1)

    # Each segment emits its start and interior; the terminal block emits the
    # ray endpoint. This avoids duplicate zero-length samples at breakpoints.
    block_counts = F.pad(emitted_steps, (0, 1), value=1)
    flat_block_counts = block_counts.reshape(-1)
    block_offsets = F.pad(flat_block_counts.cumsum(0), (1, 0))
    block_id = torch.repeat_interleave(
        torch.arange(flat_block_counts.numel(), device=device), flat_block_counts)
    block_position = (
        torch.arange(int(block_offsets[-1].item()), device=device)
        - block_offsets[block_id])
    pair_id = torch.div(block_id, max_segments + 1, rounding_mode='floor')
    block_slot = block_id % (max_segments + 1)
    terminal = block_slot == max_segments
    segment_index = block_slot.clamp(max=max_segments - 1)
    block_n = flat_block_counts[block_id].clamp(min=1)
    segment_start = phase_start[pair_id] + segment_index.to(dtype)
    regular_span = torch.minimum(
        torch.ones_like(segment_start), phase_end[pair_id] - segment_start)
    emitted_span = torch.where(too_long[pair_id], span[pair_id], regular_span)
    sample_phase = segment_start + (
        block_position.to(dtype) / block_n.to(dtype)) * emitted_span
    sample_phase = torch.where(terminal, phase_end[pair_id], sample_phase)

    samples_per_pair = n + 1
    offsets = F.pad(samples_per_pair.cumsum(0), (1, 0))
    total = int(offsets[-1].item())
    position = torch.arange(total, device=device) - offsets[pair_id]
    radii = (sample_phase + phase_theta[pair_id]) * dr
    spiral_poly = torch.stack([
        z[pair_id], sin_t[pair_id] * radii, cos_t[pair_id] * radii,
    ], dim=-1)
    if no_grad:
        with torch.no_grad():
            scroll_poly = slice_to_spiral_transform.inv(spiral_poly)
    else:
        scroll_poly = slice_to_spiral_transform.inv(spiral_poly)

    has_successor = position < n[pair_id]
    with torch.no_grad():
        step_lengths = (scroll_poly[1:] - scroll_poly[:-1]).norm(dim=-1)
        step_lengths = torch.where(
            has_successor[:-1], step_lengths, torch.zeros_like(step_lengths))
        max_step_per_pair = torch.zeros(num_pairs, device=device).scatter_reduce(
            0, pair_id[:-1], step_lengths, reduce='amax', include_self=True)
    step_violation = (max_step_per_pair > max_step) & ~too_long
    return {
        'scroll_poly': scroll_poly,
        'spiral_poly': spiral_poly,
        'sample_phase': sample_phase,
        'pair_id': pair_id,
        'position': position,
        'offsets': offsets,
        'samples_per_pair': samples_per_pair,
        'has_successor': has_successor,
        'too_long': too_long,
        'step_violation': step_violation,
        'steps_needed': steps_needed,
    }


def _pair_counts_from_samples(ray, field_value, sample_valid, sdt_volume, cfg):
    """Count/support/validity for a sampled pair-polyline batch.

    Shared by :func:`compute_pair_counts` (which builds its own rays) and the
    phase bundle (which reuses the shared central-ray samples). ``field_value``
    stays differentiable through the count; support is detached.
    """
    device = field_value.device
    num_pairs = ray['too_long'].shape[0]
    pair_id = ray['pair_id']
    offsets = ray['offsets']
    has_successor = ray['has_successor']
    too_long = ray['too_long']
    step_violation = ray['step_violation']
    s_count = float(cfg['dense_spacing_count_temperature_wv'])

    indicator = _inside_indicator(field_value, s_count)
    diffs = (indicator[1:] - indicator[:-1]).abs()
    diffs = diffs * has_successor[:-1].to(diffs.dtype)
    count = 0.5 * torch.zeros(num_pairs, device=device, dtype=diffs.dtype).index_add_(
        0, pair_id[:-1], diffs)

    # Whole-segment gating: a partially covered path undercounts and unfairly
    # compares against m.
    valid_samples_per_pair = torch.zeros(
        num_pairs, device=device, dtype=torch.long).index_add_(
        0, pair_id, sample_valid.to(torch.long))
    all_samples_valid = valid_samples_per_pair == ray['samples_per_pair']
    seg_valid = all_samples_valid & ~too_long & ~step_violation

    # Detached endpoint support from the SDT's exterior distance. stop_gradient
    # is required: otherwise the optimiser can lower spacing loss by moving
    # away from the predicted sheets and eroding its own support.
    first = offsets[:-1]
    last = offsets[1:] - 1
    with torch.no_grad():
        sigma = float(cfg['dense_spacing_support_sigma'])
        if sdt_volume is None or not cfg['dense_spacing_use_support_gate']:
            support = torch.ones(num_pairs, device=device)
        else:
            sd_first, sd_last = field_value[first], field_value[last]
            valid_first, valid_last = sample_valid[first], sample_valid[last]
            support_first = torch.exp(-(F.relu(sd_first) / sigma) ** 2)
            support_last = torch.exp(-(F.relu(sd_last) / sigma) ** 2)
            if cfg['dense_spacing_support_policy'] == 'minimum':
                support = torch.minimum(support_first, support_last)
            else:
                support = support_first * support_last
            support = support * (valid_first & valid_last).to(support.dtype)

    return {
        'count': count,
        'seg_valid': seg_valid,
        'support': support,
        'too_long': too_long,
        'step_violation': step_violation,
        'field_value': field_value,
        'sample_valid': sample_valid,
    }


def compute_pair_counts(
    slice_to_spiral_transform, dr_per_winding, sdt_volume,
    winding_idx, pair_m, theta, z, cfg,
):
    """Soft crossing counts along the mapped inter-winding polylines.

    Shared by the count-only supplement and the per-pair aggregation
    diagnostic. Each pair (k, k + m) at (z, theta) becomes a radial segment in
    spiral space, inverse-mapped as a full polyline (never the endpoint
    chord). Each constituent winding gap gets its own detached length estimate
    and step allocation targeting ~1 working voxel between mapped samples, so
    one unusually wide gap is not undersampled because its neighbours are
    short. Pairs whose required step count exceeds dense_spacing_max_steps, or
    whose mapped adjacent samples still exceed dense_spacing_max_step_wv, are
    marked invalid and reported - they are never evaluated at a spacing that
    could step over a sheet.

    Returns a dict of per-pair tensors: count (differentiable), residual
    target m, seg_valid, support (detached), too_long, step_violation, plus
    the flat sampled field for diagnostics.
    """
    dr = dr_per_winding.detach()
    m_f = pair_m.to(torch.float32)
    r0 = (winding_idx + theta / (2 * np.pi)) * dr
    r1 = r0 + m_f * dr
    ray = sample_pair_polylines(
        slice_to_spiral_transform, dr_per_winding, winding_idx,
        winding_idx + m_f, theta, z, cfg)
    field_value, sample_valid, _ = sample_sdt_trilinear(
        sdt_volume, ray['scroll_poly'])
    result = _pair_counts_from_samples(
        ray, field_value, sample_valid, sdt_volume, cfg)
    result['target_m'] = m_f
    result['r_mid'] = (r0 + r1) / 2
    return result


def sample_lasagna_normals_nearest(lasagna_volume, points_working_zyx):
    """Nearest-voxel Lasagna normal lookup in working ``zyx`` coordinates.

    The two stored uint8 components encode ``nx``/``ny``. The production field
    stores the positive-``nz`` hemisphere; returned vectors are normalized and
    ordered ``[nz, ny, nx]`` to match ray directions in ``zyx`` order.
    """
    device = points_working_zyx.device
    z_size, y_size, x_size = lasagna_volume['shape']
    scale = float(lasagna_volume['lasagna_scale'])
    index = (points_working_zyx / scale).round().long()
    zi = index[:, 0] - int(lasagna_volume['z_origin'])
    yi, xi = index[:, 1], index[:, 2]
    in_bounds = (
        (zi >= 0) & (zi < z_size) & (yi >= 0) & (yi < y_size)
        & (xi >= 0) & (xi < x_size))
    zi = zi.clamp(0, z_size - 1)
    yi = yi.clamp(0, y_size - 1)
    xi = xi.clamp(0, x_size - 1)
    if lasagna_volume['backend'] == 'mmap':
        normal_indices = torch.stack([zi, yi, xi], dim=-1)
        empty = torch.zeros([0, 3], dtype=torch.long, device=device)
        normal_u8, _ = lasagna_volume['store'].gather_pair(
            normal_indices, empty, device)
        nx_u8, ny_u8 = normal_u8.unbind(dim=-1)
    else:
        volume = lasagna_volume['volume']
        nx_u8 = volume[0, zi, yi, xi]
        ny_u8 = volume[1, zi, yi, xi]
    valid = in_bounds & ((nx_u8 != 0) | (ny_u8 != 0))
    nx = (nx_u8.to(torch.float32) - 128.0) / 127.0
    ny = (ny_u8.to(torch.float32) - 128.0) / 127.0
    nz = torch.sqrt((1.0 - nx.square() - ny.square()).clamp(min=0.0))
    normal = F.normalize(torch.stack([nz, ny, nx], dim=-1), dim=-1)
    return normal, valid


def _interpolate_at_global_arclength(ray, global_arc, target_global_arc):
    right = torch.searchsorted(global_arc, target_global_arc, right=True)
    left = (right - 1).clamp(min=0, max=global_arc.numel() - 2)
    right = left + 1
    denom = (global_arc[right] - global_arc[left]).clamp(min=1e-8)
    frac = ((target_global_arc - global_arc[left]) / denom).clamp(0.0, 1.0)
    center = torch.lerp(ray['scroll_poly'][left], ray['scroll_poly'][right], frac[:, None])
    phase = torch.lerp(ray['sample_phase'][left], ray['sample_phase'][right], frac)
    direction = F.normalize(
        ray['scroll_poly'][right] - ray['scroll_poly'][left], dim=-1)
    return center, phase, direction


@torch.no_grad()
def detect_complete_sdt_bands(ray, field, sample_valid, normal_volume, cfg):
    """Detect detached complete outside->inside->outside SDT intervals.

    ``ray`` is a detection view: packed ``scroll_poly`` / ``sample_phase`` /
    ``pair_id`` / ``has_successor`` tensors plus ``num_pairs``. ``field`` and
    ``sample_valid`` are the precomputed (detached) SDT samples at the view's
    points, so central samples shared with the count loss are gathered once.

    Leading/trailing partial intervals are never closed at ray endpoints. The
    returned packed band tensors are ordered by pair then outward arclength.
    """
    scroll = ray['scroll_poly']
    pair_id = ray['pair_id']
    has_successor = ray['has_successor']
    field = field.detach()
    normal_gather_seconds = 0.0
    edge_valid = has_successor[:-1] & sample_valid[:-1] & sample_valid[1:]
    inside = field < 0
    crossing_edge = edge_valid & (inside[:-1] != inside[1:])
    crossing_index = crossing_edge.nonzero(as_tuple=False).squeeze(-1)

    empty = torch.zeros([0], device=scroll.device)
    if crossing_index.numel() < 2:
        return {
            'pair_id': empty.long(), 'ordinal': empty.long(),
            'center': torch.zeros([0, 3], device=scroll.device),
            'phase': empty, 'direction': torch.zeros([0, 3], device=scroll.device),
            'width': empty, 'depth': empty, 'normal_dot': empty,
            'normal_valid': empty.bool(), 'graze': empty.bool(),
            'ambiguous': empty.bool(), 'interior_invalid': empty.bool(),
            'unsupported_before': empty.bool(), 'merged_count': 0,
            'field_value': field, 'sample_valid': sample_valid,
            'normal_gather_seconds': normal_gather_seconds,
        }

    step = (scroll[1:] - scroll[:-1]).norm(dim=-1)
    step = torch.where(has_successor[:-1], step, torch.zeros_like(step))
    global_arc = F.pad(step.cumsum(0), (1, 0))
    f0, f1 = field[crossing_index], field[crossing_index + 1]
    frac = (f0 / (f0 - f1)).clamp(0.0, 1.0)
    cross_global_arc = torch.lerp(
        global_arc[crossing_index], global_arc[crossing_index + 1], frac)
    entry = (~inside[crossing_index]) & inside[crossing_index + 1]
    paired = (
        entry[:-1] & ~entry[1:]
        & (pair_id[crossing_index[:-1]] == pair_id[crossing_index[1:]]))
    left_cross = paired.nonzero(as_tuple=False).squeeze(-1)
    if left_cross.numel() == 0:
        return {
            'pair_id': empty.long(), 'ordinal': empty.long(),
            'center': torch.zeros([0, 3], device=scroll.device),
            'phase': empty, 'direction': torch.zeros([0, 3], device=scroll.device),
            'width': empty, 'depth': empty, 'normal_dot': empty,
            'normal_valid': empty.bool(), 'graze': empty.bool(),
            'ambiguous': empty.bool(), 'interior_invalid': empty.bool(),
            'unsupported_before': empty.bool(), 'merged_count': 0,
            'field_value': field, 'sample_valid': sample_valid,
            'normal_gather_seconds': normal_gather_seconds,
        }

    entry_edge = crossing_index[left_cross]
    exit_edge = crossing_index[left_cross + 1]
    entry_arc = cross_global_arc[left_cross]
    exit_arc = cross_global_arc[left_cross + 1]

    # Minimum decoded SDT inside each complete interval.
    sample_index = torch.arange(field.numel(), device=field.device)
    band_for_sample = torch.searchsorted(entry_edge, sample_index, right=True) - 1
    clipped_band = band_for_sample.clamp(min=0, max=entry_edge.numel() - 1)
    in_band = (
        (band_for_sample >= 0) & (sample_index > entry_edge[clipped_band])
        & (sample_index <= exit_edge[clipped_band])
        & (pair_id[sample_index] == pair_id[entry_edge[clipped_band]]))
    depth = torch.full(
        [entry_edge.numel()], float('inf'), device=field.device,
        dtype=field.dtype)
    depth.scatter_reduce_(
        0, clipped_band[in_band], field[in_band], reduce='amin', include_self=True)
    # A band whose own interior crosses invalid samples may hide an
    # exit/re-entry (two sheets read as one complete interval), so its center
    # and width cannot be trusted.
    interior_invalid = torch.zeros(
        entry_edge.numel(), dtype=torch.bool, device=field.device)
    interior_invalid[clipped_band[in_band & ~sample_valid]] = True

    def geometry(e_arc, x_arc):
        nonlocal normal_gather_seconds
        center_arc = (e_arc + x_arc) * 0.5
        center, phase, direction = _interpolate_at_global_arclength(
            ray, global_arc, center_arc)
        normal_started = time.perf_counter()
        normal, normal_valid = sample_lasagna_normals_nearest(
            normal_volume, center)
        normal_gather_seconds += time.perf_counter() - normal_started
        normal_dot = (normal * direction).sum(dim=-1).abs()
        return center_arc, center, phase, direction, normal_dot, normal_valid

    center_arc, center, phase, direction, normal_dot, normal_valid = geometry(
        entry_arc, exit_arc)
    band_pair = pair_id[entry_edge]

    # An invalid exterior fragment cannot be used to decide whether two close
    # complete intervals are a split. Mark both candidates ambiguous instead.
    invalid_index = (~sample_valid).nonzero(as_tuple=False).squeeze(-1)
    unsupported_before = torch.zeros_like(normal_valid)
    if invalid_index.numel() and center_arc.numel() > 1:
        next_band = torch.searchsorted(center_arc, global_arc[invalid_index])
        valid_next = (next_band > 0) & (next_band < center_arc.numel())
        nb = next_band[valid_next]
        ii = invalid_index[valid_next]
        same = pair_id[ii] == band_pair[nb]
        unsupported_before[nb[same]] = True

    adjacent = torch.arange(center_arc.numel() - 1, device=field.device)
    same_pair = band_pair[:-1] == band_pair[1:]
    center_separation = center_arc[1:] - center_arc[:-1]
    projected = center_separation * torch.minimum(normal_dot[:-1], normal_dot[1:])
    close = same_pair & (
        projected < float(cfg['dense_spacing_phase_min_center_gap_wv']))
    ambiguous = interior_invalid.clone()
    ambiguous_split = close & (
        unsupported_before[1:] | interior_invalid[:-1] | interior_invalid[1:]
        | ~normal_valid[:-1] | ~normal_valid[1:])
    ambiguous[:-1] |= ambiguous_split
    ambiguous[1:] |= ambiguous_split
    merge = close & ~ambiguous_split
    # Avoid overlapping merges in a three-fragment chain in one pass; the
    # remaining close pair is retained as ambiguous instead of double-counted.
    overlapping = merge & F.pad(merge[:-1], (1, 0), value=False)
    ambiguous[:-1] |= overlapping
    ambiguous[1:] |= overlapping
    merge &= ~F.pad(merge[:-1], (1, 0), value=False)
    merged_count = int(merge.sum().item())
    if merged_count:
        left = adjacent[merge]
        right = left + 1
        exit_arc[left] = exit_arc[right]
        exit_edge[left] = exit_edge[right]
        depth[left] = torch.minimum(depth[left], depth[right])
        keep = torch.ones(center_arc.numel(), dtype=torch.bool, device=field.device)
        keep[right] = False
        entry_edge, exit_edge = entry_edge[keep], exit_edge[keep]
        entry_arc, exit_arc = entry_arc[keep], exit_arc[keep]
        depth, band_pair = depth[keep], band_pair[keep]
        ambiguous = ambiguous[keep]
        interior_invalid = interior_invalid[keep]
        unsupported_before = unsupported_before[keep]
        center_arc, center, phase, direction, normal_dot, normal_valid = geometry(
            entry_arc, exit_arc)

    counts = torch.zeros(
        int(ray['num_pairs']), dtype=torch.long, device=field.device)
    counts.index_add_(0, band_pair, torch.ones_like(band_pair))
    band_offsets = F.pad(counts.cumsum(0), (1, 0))
    ordinal = torch.arange(band_pair.numel(), device=field.device) - band_offsets[band_pair]
    graze = (
        (normal_dot < float(cfg['dense_spacing_phase_graze_dot']))
        & (depth > -float(cfg['dense_spacing_phase_graze_depth_wv'])))
    return {
        'pair_id': band_pair, 'ordinal': ordinal, 'center': center,
        'center_arc': center_arc, 'phase': phase, 'direction': direction,
        'width': exit_arc - entry_arc, 'depth': depth,
        'normal_dot': normal_dot, 'normal_valid': normal_valid,
        'graze': graze,
        'ambiguous': ambiguous, 'interior_invalid': interior_invalid,
        'unsupported_before': unsupported_before,
        'merged_count': merged_count, 'field_value': field,
        'sample_valid': sample_valid,
        'normal_gather_seconds': normal_gather_seconds,
    }


def _sample_spacing_pairs(
    cfg, inner_winding, outer_winding, num_pairs, device, z_begin, z_end,
    generator=None,
):
    domain_span = outer_winding - inner_winding

    def clamped(bounds):
        hi = min(int(bounds[1]), domain_span)
        lo = min(max(1, int(bounds[0])), hi)
        return lo, hi

    short_lo, short_hi = clamped(cfg['dense_spacing_pair_m_short'])
    long_lo, long_hi = clamped(cfg['dense_spacing_pair_m_long'])
    short_m = torch.randint(
        short_lo, short_hi + 1, [num_pairs], device=device, generator=generator)
    long_m = torch.randint(
        long_lo, long_hi + 1, [num_pairs], device=device, generator=generator)
    use_long = torch.rand(
        num_pairs, device=device, generator=generator) < float(
            cfg['dense_spacing_pair_long_fraction'])
    pair_m = torch.where(use_long, long_m, short_m)
    k = _sample_windings_by_circumference(
        inner_winding, (outer_winding - pair_m).to(torch.float32), num_pairs,
        device, generator=generator)
    theta = torch.rand(
        num_pairs, device=device, generator=generator) * (2 * np.pi)
    z = torch.empty(num_pairs, device=device).uniform_(
        float(z_begin), float(z_end - 1), generator=generator)
    return k, pair_m, theta, z


def _aggregate_nominal_mass(pair_loss, weight, alpha):
    device = pair_loss.device
    numerator = (weight * pair_loss).sum()
    stats = torch.stack([
        weight.sum(), torch.tensor(float(weight.numel()), device=device)])
    world_size = 1
    if is_distributed() and torch.is_grad_enabled():
        import torch.distributed as dist
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
    denominator = torch.maximum(stats[0], float(alpha) * stats[1])
    return world_size * numerator / denominator, stats


def _phase_band_features(bands, num_pairs, device):
    """Pack detected bands into padded per-ray tensors for the aligner.

    All outputs are detached observations. The per-band local reference gap
    is the minimum of the adjacent same-ray center distances - the minimum is
    robust around insertions and deletions (the corrupted side is either
    ignored or conservatively shrinks the gap, suppressing the band rather
    than diluting the residual scale).
    """
    pair = bands['pair_id']
    n_bands = pair.numel()
    counts = torch.zeros(num_pairs, dtype=torch.long, device=device)
    if n_bands:
        counts.index_add_(0, pair, torch.ones_like(pair))
    m_max = int(counts.max().item()) if n_bands else 0
    features = {
        'counts': counts,
        'band_valid': torch.zeros(
            [num_pairs, max(1, m_max)], dtype=torch.bool, device=device),
        'phase': torch.zeros([num_pairs, max(1, m_max)], device=device),
        'center': torch.zeros([num_pairs, max(1, m_max), 3], device=device),
        'direction': torch.zeros(
            [num_pairs, max(1, m_max), 3], device=device),
        'gref': torch.ones([num_pairs, max(1, m_max)], device=device),
        'matchable': torch.zeros(
            [num_pairs, max(1, m_max)], dtype=torch.bool, device=device),
        'confidence_cost': torch.zeros(
            [num_pairs, max(1, m_max)], device=device),
    }
    if not n_bands:
        return features

    same_prev = torch.zeros(n_bands, dtype=torch.bool, device=device)
    same_prev[1:] = pair[1:] == pair[:-1]
    gap = torch.zeros(n_bands, device=device)
    gap[1:] = bands['center_arc'][1:] - bands['center_arc'][:-1]
    infinite = torch.full([n_bands], float('inf'), device=device)
    left_gap = torch.where(same_prev, gap, infinite)
    right_gap = infinite.clone()
    right_gap[:-1] = torch.where(
        same_prev[1:], gap[1:], infinite[:-1])
    gref = torch.minimum(left_gap, right_gap)
    has_reference = torch.isfinite(gref)
    matchable = (
        bands['normal_valid'] & ~bands['ambiguous'] & ~bands['graze']
        & ~bands['unsupported_before'] & has_reference)

    slot = bands['ordinal']
    features['band_valid'][pair, slot] = True
    features['phase'][pair, slot] = bands['phase']
    features['center'][pair, slot] = bands['center']
    features['direction'][pair, slot] = bands['direction']
    features['gref'][pair, slot] = torch.where(
        has_reference, gref, torch.ones_like(gref)).clamp(min=1e-3)
    features['matchable'][pair, slot] = matchable
    features['confidence_cost'][pair, slot] = 1.0 - bands['normal_dot']
    return features


def _build_phase_padding(
    slice_to_spiral_transform, dr_per_winding, sdt_volume, k, pair_m, theta,
    z, inner_winding, outer_winding, cfg,
):
    """Map the detached extension pads flanking the central interval.

    Pads exist only for band detection; they are mapped and sampled without
    autograd. Rays whose pads are over budget keep their central samples (and
    therefore their count validity) but are rejected for phase.
    """
    device = k.device
    num_pairs = k.shape[0]
    extension = float(cfg['dense_spacing_phase_extension_windings'])
    m_f = pair_m.to(k.dtype)
    inner_start = (k - extension).clamp(min=float(inner_winding))
    outer_end = torch.minimum(
        k + m_f + extension, torch.full_like(k, float(outer_winding)))
    ray_index = torch.arange(num_pairs, device=device)
    has_inner = (k - inner_start) > 1e-4
    has_outer = (outer_end - (k + m_f)) > 1e-4
    piece_start = torch.cat([inner_start[has_inner], (k + m_f)[has_outer]])
    piece_end = torch.cat([k[has_inner], outer_end[has_outer]])
    piece_ray = torch.cat([ray_index[has_inner], ray_index[has_outer]])
    piece_is_inner = torch.cat([
        torch.ones(int(has_inner.sum()), dtype=torch.bool, device=device),
        torch.zeros(int(has_outer.sum()), dtype=torch.bool, device=device),
    ])
    pad_rejected = torch.zeros(num_pairs, dtype=torch.bool, device=device)
    if piece_start.numel() == 0:
        return None, pad_rejected, outer_end
    with torch.no_grad():
        pad_ray = sample_pair_polylines(
            slice_to_spiral_transform, dr_per_winding, piece_start, piece_end,
            theta[piece_ray], z[piece_ray], cfg, no_grad=True)
        pad_field, pad_valid, _ = sample_sdt_trilinear(
            sdt_volume, pad_ray['scroll_poly'])
    rejected_piece = pad_ray['too_long'] | pad_ray['step_violation']
    pad_rejected[piece_ray[rejected_piece]] = True
    pads = {
        'ray': pad_ray,
        'field': pad_field,
        'sample_valid': pad_valid,
        'piece_ray': piece_ray,
        'piece_is_inner': piece_is_inner,
        'rejected_piece': rejected_piece,
    }
    return pads, pad_rejected, outer_end


def _assemble_detection_view(central_ray, central_field, central_valid, pads):
    """Concatenate detached pad + central samples into one detection view.

    Per ray the view is [inner pad without its terminal sample, central
    samples, outer pad without its first sample] so the shared endpoints at
    phases k and k + m are not duplicated. Central values are detached copies
    of the live count samples - the SDT is gathered once for both losses.
    """
    device = central_field.device
    num_pairs = central_ray['too_long'].shape[0]
    n_central = central_ray['samples_per_pair']
    zeros = torch.zeros(num_pairs, dtype=torch.long, device=device)
    n_inner = zeros.clone()
    n_outer = zeros.clone()
    if pads is not None:
        piece_samples = pads['ray']['samples_per_pair']
        usable = ~pads['rejected_piece']
        contribution = torch.where(
            usable, piece_samples - 1, torch.zeros_like(piece_samples))
        n_inner.index_add_(
            0, pads['piece_ray'][pads['piece_is_inner']],
            contribution[pads['piece_is_inner']])
        n_outer.index_add_(
            0, pads['piece_ray'][~pads['piece_is_inner']],
            contribution[~pads['piece_is_inner']])
    totals = n_inner + n_central + n_outer
    view_offsets = F.pad(totals.cumsum(0), (1, 0))
    total = int(view_offsets[-1].item())
    scroll = torch.zeros([total, 3], device=device)
    phase = torch.zeros([total], device=device)
    field = torch.zeros([total], device=device)
    valid = torch.zeros([total], dtype=torch.bool, device=device)

    central_pair = central_ray['pair_id']
    central_dest = (view_offsets[central_pair] + n_inner[central_pair]
                    + central_ray['position'])
    scroll[central_dest] = central_ray['scroll_poly'].detach()
    phase[central_dest] = central_ray['sample_phase'].detach()
    field[central_dest] = central_field.detach()
    valid[central_dest] = central_valid

    if pads is not None:
        pad_ray = pads['ray']
        pid = pad_ray['pair_id']
        pos = pad_ray['position']
        usable = ~pads['rejected_piece']
        ray_of = pads['piece_ray'][pid]
        piece_samples = pad_ray['samples_per_pair']
        inner_mask = (pads['piece_is_inner'][pid] & usable[pid]
                      & (pos < piece_samples[pid] - 1))
        outer_mask = (~pads['piece_is_inner'][pid] & usable[pid]
                      & (pos >= 1))
        inner_dest = view_offsets[ray_of] + pos
        outer_dest = (view_offsets[ray_of] + n_inner[ray_of]
                      + n_central[ray_of] + pos - 1)
        for mask, dest in ((inner_mask, inner_dest), (outer_mask, outer_dest)):
            index = dest[mask]
            scroll[index] = pad_ray['scroll_poly'][mask]
            phase[index] = pad_ray['sample_phase'][mask]
            field[index] = pads['field'][mask]
            valid[index] = pads['sample_valid'][mask]

    view_pair = torch.repeat_interleave(
        torch.arange(num_pairs, device=device), totals)
    view_position = torch.arange(total, device=device) - view_offsets[view_pair]
    has_successor = view_position < (totals[view_pair] - 1)
    view = {
        'scroll_poly': scroll,
        'sample_phase': phase,
        'pair_id': view_pair,
        'has_successor': has_successor,
        'num_pairs': num_pairs,
    }
    return view, field, valid


def _phase_registration_loss(
    slice_to_spiral_transform, dr_per_winding, view, view_field, view_valid,
    normal_volume, k, pair_m, theta, z, phase_rejected, cfg, *,
    compute_map_path=False,
):
    """Soft-sequence phase registration on detached detected bands.

    The modeled winding targets (mapped through the live inverse transform)
    and the sequence alignment are differentiable; band geometry, reference
    gaps, and every gate/mass are detached, so the run cannot reduce its
    objective by rotating observations or eroding its own confidence.
    """
    device = k.device
    num_pairs = k.shape[0]
    bands = detect_complete_sdt_bands(
        view, view_field, view_valid, normal_volume, cfg)
    features = _phase_band_features(bands, num_pairs, device)

    n_max = int(pair_m.max().item()) + 1
    offsets_i = torch.arange(n_max, device=device)
    model_valid = offsets_i[None, :] <= pair_m[:, None]
    winding = k[:, None] + offsets_i[None, :].to(k.dtype)
    radius = (winding + (theta / (2 * np.pi))[:, None]) * dr_per_winding.detach()
    spiral = torch.stack([
        z[:, None].expand_as(radius),
        torch.sin(theta)[:, None] * radius,
        torch.cos(theta)[:, None] * radius,
    ], dim=-1)
    # Only the modeled targets participating in phase costs go through the
    # live inverse transform; everything observation-side stays detached.
    target = slice_to_spiral_transform.inv(
        spiral.reshape(-1, 3)).reshape(num_pairs, n_max, 3)

    displacement = ((target[:, :, None, :] - features['center'][:, None, :, :])
                    * features['direction'][:, None, :, :]).sum(dim=-1)
    rho = displacement / features['gref'][:, None, :]
    huber = F.huber_loss(
        rho, torch.zeros_like(rho), reduction='none',
        delta=float(cfg['dense_spacing_phase_huber_delta']))
    window = float(cfg['dense_spacing_phase_window_windings'])
    with torch.no_grad():
        allowed = (
            (features['phase'][:, None, :] - winding[:, :, None]).abs()
            <= window)
        allowed &= features['matchable'][:, None, :]
        allowed &= model_valid[:, :, None]
        allowed &= features['band_valid'][:, None, :]
        allowed &= ~phase_rejected[:, None, None]
    confidence = (float(cfg['dense_spacing_phase_band_confidence_cost'])
                  * features['confidence_cost'])
    cost = torch.where(
        allowed, huber + confidence[:, None, :],
        torch.full_like(huber, float('inf')))

    # Semi-global end gaps as per-band skip costs: padding bands outside the
    # central interval (plus a protection margin so a slightly displaced
    # boundary band cannot be dumped into the free gap) are free to skip.
    with torch.no_grad():
        margin = float(cfg['dense_spacing_phase_end_free_margin_windings'])
        m_f = pair_m.to(k.dtype)
        out_of_model = (
            (features['phase'] < (k - margin)[:, None])
            | (features['phase'] > (k + m_f + margin)[:, None]))
        extra_open = torch.where(
            out_of_model,
            torch.zeros_like(features['phase']),
            torch.full_like(features['phase'],
                            float(cfg['dense_spacing_phase_extra_cost'])))
        extra_extend = torch.where(
            out_of_model,
            torch.zeros_like(features['phase']),
            torch.full_like(
                features['phase'],
                float(cfg['dense_spacing_phase_extra_extend_cost'])))

    alignment = soft_align_sequences(
        cost, model_valid, features['band_valid'],
        missing_open=float(cfg['dense_spacing_phase_missing_cost']),
        missing_extend=float(cfg['dense_spacing_phase_missing_extend_cost']),
        extra_open=extra_open,
        extra_extend=extra_extend,
        temperature=float(cfg['dense_spacing_phase_temperature']),
        compute_map_path=compute_map_path)
    posterior = alignment['match_posterior']

    with torch.no_grad():
        margin_threshold = float(cfg['dense_spacing_phase_top2_margin'])
        detached_posterior = posterior.detach()
        # Each winding is scored at its dominant match cell only, weighted by
        # the live marginal: sub-dominant posterior on a plausible-but-wrong
        # +-1 match next to a hole is reported (ambiguous/suppressed mass)
        # but must never pull the surface - crossing count owns the topology
        # signal. A winding participates only when its dominant match beats
        # its missing state (probability >= 0.5) and is not multimodal (the
        # top-2 margin gate also keeps the dominant-cell selection continuous
        # where marginals cross).
        masked_posterior = detached_posterior * allowed
        best = masked_posterior.argmax(dim=2, keepdim=True)
        dominant = torch.zeros_like(allowed).scatter_(2, best, True) & allowed
        dominant_mass = (masked_posterior * dominant).sum(dim=2)
        suppressed_winding = model_valid & (
            (alignment['top2_margin'] < margin_threshold)
            | (dominant_mass < 0.5))
        accept = dominant & ~suppressed_winding[:, :, None]
        total_match_mass = masked_posterior.sum()
        ambiguous_mass = (masked_posterior
                          * ~accept).sum()
        winding_mass = (detached_posterior * accept).sum(dim=2)
        useful_windings = (winding_mass >= 0.5).sum(dim=1)
        mass = winding_mass.sum(dim=1)
        ray_ok = (
            ~phase_rejected
            & (useful_windings
               >= int(cfg['dense_spacing_phase_min_matched_windings']))
            & (mass >= float(cfg['dense_spacing_phase_min_matched_mass'])))
        pair_weight = ray_ok.to(torch.float32)
        accepted_mass = (winding_mass * pair_weight[:, None]).sum()
        suppressed_mass = total_match_mass - accepted_mass

    numerator = (posterior * huber * accept).sum(dim=(1, 2))
    ray_loss = numerator / mass.clamp(min=1e-6)
    loss, mass_stats = _aggregate_nominal_mass(
        ray_loss, pair_weight, cfg['dense_spacing_support_floor_alpha'])

    with torch.no_grad():
        n_bands = int(bands['pair_id'].numel())
        scoring = ray_ok
        scoring_f = scoring.to(torch.float32)
        n_scoring = scoring_f.sum().clamp(min=1.0)
        metrics = {
            'dense_spacing_phase_match_mass': float(accepted_mass.item()),
            'dense_spacing_phase_valid_fraction': float(pair_weight.mean().item()),
            'dense_spacing_phase_rejected_ray_fraction': float(
                phase_rejected.float().mean().item()),
            'dense_spacing_phase_matches_per_ray': float(
                ((alignment['expected_matches'] * scoring_f).sum()
                 / n_scoring).item()),
            'dense_spacing_phase_missing_per_ray': float(
                ((alignment['expected_missing'] * scoring_f).sum()
                 / n_scoring).item()),
            'dense_spacing_phase_extra_per_ray': float(
                ((alignment['expected_extra'] * scoring_f).sum()
                 / n_scoring).item()),
            'dense_spacing_phase_alignment_entropy_mean': float(
                ((alignment['ray_entropy'] * scoring_f).sum()
                 / n_scoring).item()),
            'dense_spacing_phase_ambiguous_match_mass_fraction': float(
                (ambiguous_mass / total_match_mass.clamp(min=1e-6)).item()),
            'dense_spacing_phase_suppressed_match_mass_fraction': float(
                (suppressed_mass.clamp(min=0.0)
                 / total_match_mass.clamp(min=1e-6)).item()),
            'dense_spacing_phase_mean_scored_correspondences': float(
                ((winding_mass.sum(dim=1) * scoring_f).sum()
                 / n_scoring).item()),
            'dense_spacing_phase_complete_bands_per_ray': float(
                n_bands / max(1, num_pairs)),
            'dense_spacing_phase_bands_per_modeled_winding': float(
                n_bands / max(1, int(pair_m.sum().item()))),
            'dense_spacing_phase_merged_fragment_fraction': float(
                bands['merged_count']
                / max(1, n_bands + bands['merged_count'])),
            'dense_spacing_phase_unsupported_sdt_fraction': float(
                (~view_valid).float().mean().item()) if view_valid.numel()
            else 0.0,
            'dense_spacing_phase_sampled_points': float(
                view['scroll_poly'].shape[0]),
            'dense_spacing_phase_normal_gather_seconds': bands[
                'normal_gather_seconds'],
        }
        if scoring.any():
            entropy_values = alignment['ray_entropy'][scoring]
            clearance_values = alignment['map_clearance'][scoring]
            metrics['dense_spacing_phase_alignment_entropy_p90'] = float(
                torch.quantile(entropy_values, 0.9).item())
            metrics['dense_spacing_phase_top2_clearance_p10'] = float(
                torch.quantile(
                    clearance_values.clamp(max=1e4), 0.1).item())
        # Effective matches (posterior-dominant accepted cells) drive the
        # residual quantiles and the by-offset breakdown.
        effective = accept & (detached_posterior > 0.25) \
            & scoring[:, None, None]
        if effective.any():
            abs_rho = rho.abs()[effective]
            metrics['dense_spacing_phase_residual_mean_abs'] = float(
                abs_rho.mean().item())
            rho_q = torch.quantile(
                abs_rho, torch.tensor([0.5, 0.9, 0.95], device=device))
            metrics['dense_spacing_phase_residual_abs_p50'] = float(rho_q[0].item())
            metrics['dense_spacing_phase_residual_abs_p90'] = float(rho_q[1].item())
            metrics['dense_spacing_phase_residual_abs_p95'] = float(rho_q[2].item())
            offset_grid = offsets_i[None, :, None].expand_as(effective)
            effective_offset = offset_grid[effective]
            for ordinal in torch.unique(effective_offset).tolist():
                ordinal_mask = effective_offset == ordinal
                metrics[
                    f'dense_spacing_phase_residual_mean_abs_ordinal_{ordinal}'
                ] = float(abs_rho[ordinal_mask].mean().item())
        if n_bands:
            projected_width = bands['width'] * bands['normal_dot']
            width_q = torch.quantile(
                projected_width, torch.tensor([0.5, 0.9], device=device))
            metrics['dense_spacing_phase_projected_width_p50'] = float(
                width_q[0].item())
            metrics['dense_spacing_phase_projected_width_p90'] = float(
                width_q[1].item())
        if diagnostics_enabled() and effective.any():
            centers = features['center'][:, None, :, :].expand(
                -1, n_max, -1, -1)[effective]
            detected_center_spiral = (
                slice_to_spiral_transform(centers)
                if callable(slice_to_spiral_transform) else centers)
            record_loss_samples(
                'dense_spacing_phase', detected_center_spiral,
                rho.abs()[effective].detach(),
                torch.ones(int(effective.sum()), dtype=torch.bool,
                           device=device))
    if compute_map_path:
        metrics['_map_alignment'] = {
            'map_match': alignment['map_match'],
            'map_extra': alignment['map_extra'],
            'match_posterior': alignment['match_posterior'].detach(),
            'ray_ok': ray_ok,
        }
    return loss, metrics


def _count_loss_from_pairs(counts, targets, supports, seg_valids, cfg,
                           sdt_volume, diagnostics):
    """Support-gated |count - m| with the global nominal-mass floor.

    L = sum(w * |count - m|) / max(sum(w), alpha * num_pairs): the floor makes
    the loss scale down (reaching a defined zero) when total support falls
    below alpha of the batch, instead of a scale-invariant weighted mean that
    would fire at full strength on uniformly tiny support.
    """
    count = torch.cat(counts)
    target_m = torch.cat(targets)
    support = torch.cat(supports)
    seg_valid = torch.cat(seg_valids)
    weight = (support * seg_valid.to(torch.float32)).detach()
    residual = (count - target_m).abs()
    alpha = float(cfg['dense_spacing_support_floor_alpha'])
    loss, stats = _aggregate_nominal_mass(residual, weight, alpha)

    with torch.no_grad():
        valid_f = seg_valid.to(torch.float32)
        n_valid = valid_f.sum()
        n_valid_floor = n_valid.clamp(min=1.0)
        (mass_value, nominal_value, valid_fraction, count_mean,
         residual_mean, n_valid_value) = torch.stack([
            stats[0],
            stats[1],
            valid_f.mean(),
            (count * valid_f).sum() / n_valid_floor,
            (residual * valid_f).sum() / n_valid_floor,
            n_valid,
        ]).tolist()
        floor_active = mass_value < alpha * nominal_value
        mass_fraction = mass_value / max(1.0, nominal_value)
        metrics = {
            'dense_spacing_count_support_mass': mass_value,
            'dense_spacing_count_support_mass_fraction': mass_fraction,
            'dense_spacing_count_floor_active': float(floor_active),
            'dense_spacing_count_valid_fraction': valid_fraction,
        }
        metrics.update(diagnostics)
        if n_valid_value > 0:
            metrics['dense_spacing_count_mean'] = count_mean
            metrics['dense_spacing_count_residual_mean'] = residual_mean
        global _floor_was_active
        if floor_active and not _floor_was_active:
            print('dense_spacing_count: support-mass floor active '
                  f'(mass fraction {mass_fraction:.4f} < alpha {alpha}); '
                  'the count loss is effectively scaled down until support recovers')
        elif _floor_was_active and not floor_active:
            print('dense_spacing_count: support-mass floor released '
                  f'(mass fraction {mass_fraction:.4f} >= alpha {alpha})')
        _floor_was_active = floor_active
    return loss, metrics, residual, weight


def phase_bundle_component_weights(cfg, attachment_ramp=1.0):
    """The four phase-bundle component weights; zero disables a component."""
    return {
        'dense_spacing_phase': float(cfg['loss_weight_dense_spacing']),
        'dense_spacing_count': float(cfg['loss_weight_dense_spacing_count']),
        'min_spacing': float(cfg['loss_weight_min_spacing']),
        'dense_attachment': (float(cfg['loss_weight_dense_attachment'])
                             * float(attachment_ramp)),
    }


def _phase_and_count_losses(
    slice_to_spiral_transform, dr_per_winding, sdt_volume, normal_volume,
    outer_winding_idx, cfg, z_begin, z_end, weights, generator,
    compute_map_path,
):
    """Shared-ray phase registration and crossing count.

    One joint ray batch serves both components: the central ``k -> k + m``
    polyline is built with autograd and the SDT is sampled once with
    gradients on it; the count uses the live samples while phase band
    detection sees detached copies plus detached extension pads. Validity is
    tracked separately - a failed pad must not invalidate a valid central
    count, and a missing phase correspondence must not silence the count.
    """
    device = dr_per_winding.device
    count_active = weights['dense_spacing_count'] > 0
    phase_active = weights['dense_spacing_phase'] > 0
    inner, outer = fitted_winding_domain(outer_winding_idx)
    if outer - inner < 1:
        return
    started = time.perf_counter()
    num_pairs = int(cfg['dense_spacing_num_pairs'])
    k, pair_m, theta, z = _sample_spacing_pairs(
        cfg, inner, outer, num_pairs, device, z_begin, z_end, generator)
    m_f = pair_m.to(k.dtype)

    build_grad = count_active and torch.is_grad_enabled()
    central_ray = sample_pair_polylines(
        slice_to_spiral_transform, dr_per_winding, k, k + m_f, theta, z, cfg,
        no_grad=not build_grad)
    sdt_started = time.perf_counter()
    central_field, central_valid, _ = sample_sdt_trilinear(
        sdt_volume, central_ray['scroll_poly'])
    sdt_sample_seconds = time.perf_counter() - sdt_started

    view = view_field = view_valid = None
    pad_rejected = None
    if phase_active:
        pads, pad_rejected, outer_end = _build_phase_padding(
            slice_to_spiral_transform, dr_per_winding, sdt_volume, k, pair_m,
            theta, z, inner, outer, cfg)
        view, view_field, view_valid = _assemble_detection_view(
            central_ray, central_field, central_valid, pads)
        del pads

    if count_active:
        pair = _pair_counts_from_samples(
            central_ray, central_field, central_valid, sdt_volume, cfg)
        diagnostics = {
            'dense_spacing_count_too_long_fraction': float(
                pair['too_long'].float().mean().item()),
            'dense_spacing_count_step_violation_fraction': float(
                pair['step_violation'].float().mean().item()),
            'dense_spacing_count_sdt_sample_seconds': sdt_sample_seconds,
        }
        diagnostics.update({
            f'dense_spacing_count_sdt_{name}': value
            for name, value in sdt_sample_fractions(
                pair['field_value'], pair['sample_valid'],
                sdt_volume).items()
        })
        counts = [pair['count']]
        targets = [m_f]
        supports = [pair['support']]
        seg_valids = [pair['seg_valid']]
        theta_all, z_all, r_mid_all = [theta], [z], [
            (k + m_f / 2 + theta / (2 * np.pi)) * dr_per_winding.detach()]
        extra_pairs = int(cfg['dense_spacing_count_extra_pairs'])
        if extra_pairs > 0:
            # Optional count-only supplement restoring spatial coverage when
            # the shared batch alone is too sparse.
            k_e, m_e, theta_e, z_e = _sample_spacing_pairs(
                cfg, inner, outer, extra_pairs, device, z_begin, z_end,
                generator)
            extra = compute_pair_counts(
                slice_to_spiral_transform, dr_per_winding, sdt_volume,
                k_e, m_e, theta_e, z_e, cfg)
            counts.append(extra['count'])
            targets.append(extra['target_m'])
            supports.append(extra['support'])
            seg_valids.append(extra['seg_valid'])
            theta_all.append(theta_e)
            z_all.append(z_e)
            r_mid_all.append(extra['r_mid'])
        count_loss, count_metrics, residual, weight = _count_loss_from_pairs(
            counts, targets, supports, seg_valids, cfg, sdt_volume,
            diagnostics)
        spiral_mid = torch.stack([
            torch.cat(z_all),
            torch.sin(torch.cat(theta_all)) * torch.cat(r_mid_all),
            torch.cos(torch.cat(theta_all)) * torch.cat(r_mid_all),
        ], dim=-1)
        record_loss_samples(
            'dense_spacing_count', spiral_mid, residual, weight > 0)
        yield 'dense_spacing_count', count_loss, count_metrics
        del pair, counts, residual

    if phase_active:
        # Central live tensors are no longer needed once the detached view
        # exists; drop them so the count backward can free its graph.
        central_rejected = central_ray['too_long'] | central_ray['step_violation']
        phase_rejected = central_rejected | pad_rejected
        central_metrics = {
            'dense_spacing_phase_too_long_fraction': float(
                central_ray['too_long'].float().mean().item()),
            'dense_spacing_phase_step_violation_fraction': float(
                central_ray['step_violation'].float().mean().item()),
            'dense_spacing_phase_pad_rejected_fraction': float(
                pad_rejected.float().mean().item()),
            'dense_spacing_phase_extension_domain_clamped_fraction': float(
                ((k + m_f + float(
                    cfg['dense_spacing_phase_extension_windings']))
                 > float(outer)).float().mean().item()),
            'dense_spacing_phase_sdt_sample_seconds': sdt_sample_seconds,
        }
        del central_ray, central_field, central_valid
        phase_loss, phase_metrics = _phase_registration_loss(
            slice_to_spiral_transform, dr_per_winding, view, view_field,
            view_valid, normal_volume, k, pair_m, theta, z, phase_rejected,
            cfg, compute_map_path=compute_map_path)
        phase_metrics.update(central_metrics)
        phase_metrics['dense_spacing_phase_wall_seconds'] = (
            time.perf_counter() - started)
        yield 'dense_spacing_phase', phase_loss, phase_metrics


def iter_phase_bundle_losses(
    spiral_and_transform, slice_to_spiral_transform, dr_per_winding,
    sdt_volume, normal_volume, outer_winding_idx, cfg, z_begin, z_end, *,
    attachment_ramp=1.0, generator=None, compute_map_path=False,
):
    """Yield the active phase-bundle components as (name, loss, metrics).

    The bundle is the single ``phase`` dense-spacing mode: soft-sequence
    phase registration, crossing count (sharing the phase rays), native
    minimum spacing, and SDT attachment. Yielding lazily lets the training
    loop run one backward per component so at most one large graph is
    resident; the public mode stays ``phase`` regardless of this internal
    backward grouping.
    """
    weights = phase_bundle_component_weights(cfg, attachment_ramp)
    if outer_winding_idx is not None and sdt_volume is not None:
        if sdt_volume['kind'] != 'sdt':
            raise ValueError('the phase bundle requires a signed-distance store')
        if ((weights['dense_spacing_phase'] > 0 and normal_volume is not None)
                or weights['dense_spacing_count'] > 0):
            effective = dict(weights)
            if normal_volume is None:
                effective['dense_spacing_phase'] = 0.0
            yield from _phase_and_count_losses(
                slice_to_spiral_transform, dr_per_winding, sdt_volume,
                normal_volume, outer_winding_idx, cfg, z_begin, z_end,
                effective, generator, compute_map_path)
    if weights['min_spacing'] > 0 and spiral_and_transform is not None:
        min_loss, min_metrics = get_min_spacing_loss(
            spiral_and_transform, outer_winding_idx, cfg, z_begin, z_end,
            generator=generator)
        yield 'min_spacing', min_loss, min_metrics
    if weights['dense_attachment'] > 0 and sdt_volume is not None:
        attachment_loss, attachment_metrics = get_dense_attachment_loss(
            slice_to_spiral_transform, dr_per_winding, sdt_volume,
            outer_winding_idx, cfg, z_begin, z_end)
        yield 'dense_attachment', attachment_loss, attachment_metrics


def get_min_spacing_loss(
    spiral_and_transform, outer_winding_idx, cfg, z_begin, z_end, *,
    generator=None,
):
    """Squared hinge on sampled native log gaps before exponentiation."""
    device = spiral_and_transform.device
    zero = torch.zeros([], device=device)
    if outer_winding_idx is None:
        return zero, {}
    num_samples = int(cfg['min_spacing_independent_samples'])
    inner, outer = fitted_winding_domain(outer_winding_idx)
    if outer <= inner or num_samples <= 0:
        return zero, {}
    winding = _sample_windings_by_circumference(
        inner, outer - 1, num_samples, device, generator=generator).long()
    theta = torch.rand(
        num_samples, device=device, generator=generator) * (2 * np.pi)
    z = torch.empty(num_samples, device=device).uniform_(
        float(z_begin), float(z_end - 1), generator=generator)
    ell_gap = spiral_and_transform.get_native_log_gaps(winding, theta, z)
    ell_min = float(np.log(float(cfg['min_spacing_d_min_wv'])))
    deficiency = F.relu(ell_min - ell_gap)
    loss = deficiency.square().mean()
    with torch.no_grad():
        gap = torch.exp(ell_gap)
        q = torch.quantile(gap, torch.tensor([0.1, 0.5, 0.9], device=device))
        active = deficiency > 0
        metrics = {
            'min_spacing_active_fraction': float(active.float().mean().item()),
            'min_spacing_gap_p10': float(q[0].item()),
            'min_spacing_gap_p50': float(q[1].item()),
            'min_spacing_gap_p90': float(q[2].item()),
            'min_spacing_ell_gap_mean': float(ell_gap.mean().item()),
            'min_spacing_violation_depth_mean': float(
                deficiency[active].mean().item()) if active.any() else 0.0,
        }
    return loss, metrics


def get_dense_attachment_loss(
    slice_to_spiral_transform, dr_per_winding, sdt_volume, outer_winding_idx,
    cfg, z_begin, z_end,
):
    """smooth_l1(relu(sd) / attachment_scale) at dense points on fitted winding
    surfaces. Zero on or inside the predicted mask, smooth-L1 growth outside,
    with a live gradient over the SDT's full +/-cap range. Evaluated on winding
    surfaces only - never on spacing's inter-winding path points."""
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if sdt_volume is None or outer_winding_idx is None:
        return zero, {}
    if sdt_volume['kind'] != 'sdt':
        raise ValueError('attachment requires a signed-distance store, not a raw-surf volume')

    num_points = int(cfg['dense_attachment_num_points'])
    attachment_scale = float(cfg['dense_attachment_scale'])
    inner_winding, outer_winding = fitted_winding_domain(outer_winding_idx)
    if outer_winding < inner_winding:
        return zero, {}

    winding_idx = _sample_windings_by_circumference(
        inner_winding, float(outer_winding), num_points, device)
    theta = torch.rand(num_points, device=device) * (2 * np.pi)
    z = torch.empty(num_points, device=device).uniform_(float(z_begin), float(z_end - 1))
    radius = (winding_idx + theta / (2 * np.pi)) * dr_per_winding.detach()
    spiral_zyx = torch.stack(
        [z, torch.sin(theta) * radius, torch.cos(theta) * radius], dim=-1)
    scroll_zyx = slice_to_spiral_transform.inv(spiral_zyx)

    sd, sample_valid, _ = sample_sdt_trilinear(sdt_volume, scroll_zyx)
    exterior = F.relu(sd)
    residual = F.smooth_l1_loss(
        exterior / attachment_scale, torch.zeros_like(exterior), reduction='none')
    valid_f = sample_valid.to(residual.dtype)
    loss = (residual * valid_f).sum() / valid_f.sum().clamp(min=1.0)

    with torch.no_grad():
        metrics = {
            f'dense_attachment_{name}': value
            for name, value in sdt_sample_fractions(sd, sample_valid, sdt_volume).items()
        }
        if sample_valid.any():
            exterior_valid = exterior[sample_valid]
            quantiles = torch.quantile(
                exterior_valid, torch.tensor([0.5, 0.9, 0.95], device=device))
            metrics.update({
                'dense_attachment_exterior_p50': float(quantiles[0].item()),
                'dense_attachment_exterior_p90': float(quantiles[1].item()),
                'dense_attachment_exterior_p95': float(quantiles[2].item()),
            })
            # Saturated samples carry a residual but no local gradient; split
            # them out so aggregate loss is not mistaken for effective
            # attraction.
            live = sample_valid & (sd.abs() < _saturation_threshold(sdt_volume))
            metrics['dense_attachment_live_gradient_fraction'] = float(
                live.float().sum().item() / max(1, int(sample_valid.sum().item())))

    record_loss_samples('dense_attachment', spiral_zyx, residual, sample_valid)
    return loss, metrics


@torch.no_grad()
def aggregate_pair_counts(
    slice_to_spiral_transform, dr_per_winding, sdt_volume,
    outer_winding_idx, cfg, z_begin, z_end, samples_per_pair=192, pair_m=1,
    batch_pairs=64,
):
    """Per-pair aggregated crossing counts: mean_count - m per winding pair.

    With per-segment std ~ 0.5 * sqrt(m) and hundreds of samples per pair,
    off-by-one winding assignments are detectable per pair with high
    confidence. This is a measurement/diagnostic - the input for any future
    discrete insert/remove/reindex operation - not a loss.
    """
    device = dr_per_winding.device
    inner_winding, outer_winding = fitted_winding_domain(outer_winding_idx)
    rows = []
    windings = list(range(inner_winding, outer_winding - pair_m + 1))
    for batch_start in range(0, len(windings), batch_pairs):
        batch = windings[batch_start:batch_start + batch_pairs]
        k = torch.tensor(batch, device=device, dtype=torch.float32).repeat_interleave(
            samples_per_pair)
        m = torch.full_like(k, float(pair_m), dtype=torch.long)
        theta = torch.rand(k.shape[0], device=device) * (2 * np.pi)
        z = torch.empty(k.shape[0], device=device).uniform_(
            float(z_begin), float(z_end - 1))
        pair = compute_pair_counts(
            slice_to_spiral_transform, dr_per_winding, sdt_volume,
            k, m, theta, z, cfg)
        counts = pair['count'].reshape(len(batch), samples_per_pair)
        valid = pair['seg_valid'].reshape(len(batch), samples_per_pair)
        support = pair['support'].reshape(len(batch), samples_per_pair)
        for row_index, winding in enumerate(batch):
            row_valid = valid[row_index]
            n_valid = int(row_valid.sum().item())
            row = {
                'winding': int(winding),
                'pair_m': int(pair_m),
                'num_samples': samples_per_pair,
                'num_valid': n_valid,
                'mean_support': float(support[row_index].mean().item()),
            }
            if n_valid:
                row_counts = counts[row_index][row_valid]
                row.update({
                    'mean_count': float(row_counts.mean().item()),
                    'std_count': float(row_counts.std().item()) if n_valid > 1 else 0.0,
                    'mean_count_minus_m': float(row_counts.mean().item()) - pair_m,
                })
            rows.append(row)
    return rows
