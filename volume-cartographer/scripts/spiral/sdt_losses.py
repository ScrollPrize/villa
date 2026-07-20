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
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ddp_helpers import get_world_size, is_distributed
from loss_maps import diagnostics_enabled, record_loss_samples
from soft_alignment import soft_align_sequences
from transforms import ray_gap_enabled, ray_specialized_spiral_to_scroll


def _map_radial_samples(
    slice_to_spiral_transform, radii, theta, z, pair_id, sin_t, cos_t,
):
    """Spiral->scroll for radial-ray samples; per-ray gap stage when the
    production chain is recognized (see transforms.ray_specialized_spiral_to_
    scroll), generic transform otherwise."""
    if ray_gap_enabled():
        out = ray_specialized_spiral_to_scroll(
            slice_to_spiral_transform, radii, theta, z, pair_id, sin_t, cos_t)
        if out is not None:
            return out
    spiral = torch.stack([
        z[pair_id], sin_t[pair_id] * radii, cos_t[pair_id] * radii], dim=-1)
    return slice_to_spiral_transform.inv(spiral)


# A sample is valid when the total trilinear weight of its valid corners is at
# least this mass (weight mass, not corner count: four valid corners can carry
# near-zero weight when the sample point sits in the invalid octant).
MIN_VALID_CORNER_WEIGHT_MASS = 0.5

_CORNER_OFFSETS = tuple(itertools.product((0, 1), repeat=3))

# Printed diagnostic state for the support-mass floor (metric is emitted every
# step regardless; the print fires only on transitions to avoid spam).
_floor_was_active = False


_METRICS_EVERY = max(1, int(os.environ.get('FIT_SPIRAL_METRICS_EVERY', '1')))
_metrics_tick = itertools.count()
_metrics_enabled_now = True


def metrics_enabled():
    """Whether this bundle iteration computes the expensive metrics blocks.

    Every float()/.item() in a metrics dict is a device synchronization; at
    probe/DDP shapes those stalls dominate the step. With
    FIT_SPIRAL_METRICS_EVERY=K (default 1 = every step, the historical
    behavior) the sync-heavy metrics run on every K-th
    iter_phase_bundle_losses call and the skipped steps yield minimal
    dicts. Losses and gradients are identical either way.
    """
    return _metrics_enabled_now


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
            torch.tensor([float(sdt_volume['z_origin']),
                          float(sdt_volume.get('y_origin', 0)),
                          float(sdt_volume.get('x_origin', 0))],
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
    elif sdt_volume['backend'] == 'dense_cuda_paged':
        from lasagna_data import gather_paged_u8
        values_u8 = gather_paged_u8(
            sdt_volume, clamped[..., 0], clamped[..., 1], clamped[..., 2])
    else:  # dense torch uint8 tensor (tests / tiny ROIs)
        volume = sdt_volume['volume']
        values_u8 = volume[clamped[..., 0], clamped[..., 1], clamped[..., 2]]

    corner_valid = in_bounds & (values_u8 != 0)
    corner_frac = torch.where(
        offsets[None, :, :].bool(), frac[:, None, :], 1.0 - frac[:, None, :])
    # (N, 8), differentiable wrt positions. Explicit chain instead of
    # .prod(dim=-1): same left-to-right multiply (bitwise-identical values)
    # but elementwise MulBackward instead of ProdBackward's double-cumprod
    # scan, which dominated the count-term backward (2x56 ms/step on H100
    # for the [N*8, 3] scan).
    weights = (corner_frac[..., 0] * corner_frac[..., 1]
               * corner_frac[..., 2])
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
    with torch.no_grad():
        breakpoint_scroll = _map_radial_samples(
            slice_to_spiral_transform, probe_radii, theta, z, probe_pair_id,
            sin_t, cos_t).reshape(num_pairs, max_segments + 1, 3)
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
            scroll_poly = _map_radial_samples(
                slice_to_spiral_transform, radii, theta, z, pair_id,
                sin_t, cos_t)
    else:
        scroll_poly = _map_radial_samples(
            slice_to_spiral_transform, radii, theta, z, pair_id, sin_t, cos_t)

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
    yi = index[:, 1] - int(lasagna_volume.get('y_origin', 0))
    xi = index[:, 2] - int(lasagna_volume.get('x_origin', 0))
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
    elif lasagna_volume['backend'] == 'dense_cuda_paged':
        from lasagna_data import gather_paged_u8
        nx_u8 = gather_paged_u8(lasagna_volume, zi, yi, xi, channel=0)
        ny_u8 = gather_paged_u8(lasagna_volume, zi, yi, xi, channel=1)
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
            'unsupported_before': empty.bool(), 'clearance_before': empty,
            'merged_count': 0,
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
            'unsupported_before': empty.bool(), 'clearance_before': empty,
            'merged_count': 0,
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

    # Exterior clearance of the gap before each band: the maximum decoded SDT
    # between the previous same-pair band's exit and this band's entry. A
    # genuine inter-winding gap contains real air (clearance well above 0); a
    # band split off its neighbour by a shallow blip or hole barely exits the
    # sheet. First bands of a ray (no previous band) read +inf.
    clearance_before = torch.full(
        [entry_edge.numel()], float('inf'), device=field.device,
        dtype=field.dtype)
    if entry_edge.numel() > 1:
        band_of = torch.searchsorted(entry_edge, sample_index, right=True) - 1
        clipped = band_of.clamp(min=0, max=entry_edge.numel() - 2)
        in_gap = (
            (band_of >= 0) & (band_of < entry_edge.numel() - 1)
            & (sample_index > exit_edge[clipped])
            & (sample_index <= entry_edge[clipped + 1])
            & (pair_id[sample_index] == pair_id[entry_edge[clipped]])
            & sample_valid)
        gap_clearance = torch.full(
            [entry_edge.numel() - 1], float('-inf'), device=field.device,
            dtype=field.dtype)
        gap_clearance.scatter_reduce_(
            0, clipped[in_gap], field[in_gap], reduce='amax',
            include_self=True)
        same_prev_band = band_pair[1:] == band_pair[:-1]
        clearance_before[1:] = torch.where(
            same_prev_band, gap_clearance,
            torch.full_like(gap_clearance, float('inf')))

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
        'clearance_before': clearance_before,
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
        'clearance': torch.full(
            [num_pairs, max(1, m_max)], float('inf'), device=device),
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
    features['clearance'][pair, slot] = bands['clearance_before']
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


def _metric_density_loss(central, features, pair_m, rejected, cfg):
    """``|integral(lambda ds) - m|`` with a detached band-derived density.

    ``lambda`` is the inverse distance between the two detected band centers
    bracketing each live polyline step (piecewise-constant, detached), so -
    unlike the topological crossing count, whose gradient lives only where a
    sigmoid transition straddles a sample - every step of the live central
    polyline carries gradient proportional to its length. This is the metric
    analogue of the legacy grad_mag density integral, sourced from clean SDT
    band geometry instead of the noisy grad_mag volume, and with no endpoint
    support gate to self-confirm on a mis-registered model.

    Sub-resolution abstention (band-GT calibrated 2026-07-17): below ~12 wv
    detected spacing every SDT estimator under-reads winding density ~30%
    (the store cannot separate that tight a sheet pair), so optimising the
    integral there actively fights truth. Steps bracketed by such a gap are
    blind: they contribute nothing to the integral and their detached
    model-phase span is subtracted from the target, so the rest of the pair
    still carries an unbiased residual. Pairs that are mostly blind are
    dropped entirely. The criterion is observation-side (detected gap
    width); only the excluded span's size uses the model, so there is no
    restoring force inside blind zones but no self-confirmation either.
    """
    device = pair_m.device
    scroll = central['scroll_poly']
    pid = central['pair_id']
    succ = central['has_successor']
    counts = features['counts']
    if features['phase'].shape[1] < 2:
        zero = scroll.sum() * 0.0
        return zero, {'dense_spacing_density_valid_fraction': 0.0}
    with torch.no_grad():
        phase = central['sample_phase'].detach()
        bphase = features['phase']
        band_valid = features['band_valid']
        centers = features['center']
        gap_floor = float(cfg['dense_spacing_phase_min_center_gap_wv'])
        gaps = (centers[:, 1:] - centers[:, :-1]).norm(dim=-1).clamp(
            min=gap_floor)
        # step midpoint phase -> index of the bracketing gap
        step_mask = succ.clone()
        step_mask[-1] = False
        s_phase = 0.5 * (phase[:-1] + phase[1:])
        s_pid = pid[:-1]
        cmp = (bphase[s_pid] <= s_phase[:, None]) & band_valid[s_pid]
        left = cmp.sum(dim=1) - 1
        gap_idx = left.clamp(min=0)
        gap_idx = torch.minimum(
            gap_idx, (counts[s_pid] - 2).clamp(min=0))
        step_gap = gaps[s_pid, gap_idx.clamp(max=gaps.shape[1] - 1)]
        blind = (
            step_gap < float(cfg['dense_spacing_density_min_gap_wv']))
        lambda_mode = cfg.get('dense_spacing_density_lambda', 'inverse_gap')
        if (lambda_mode in ('soft_mass', 'soft_mass_wide')
                and central.get('field') is not None):
            # Mass-weighted lambda (outer-recall experiment): per detected
            # gap, lambda = (soft crossing mass)/2 per unit arclength
            # instead of 1/(center distance). A sheet the hard detection
            # missed still carries its |delta indicator| mass (2 per full
            # crossing), so the integral keeps counting it fractionally —
            # the density calibration's -12% outer under-read is exactly
            # missed detections widening gaps.
            s_count = float(cfg['dense_spacing_count_temperature_wv'])
            ind = _inside_indicator(central['field'].detach(), s_count)
            live_step = step_mask[:-1]
            dmass = (ind[1:] - ind[:-1]).abs() * live_step.to(ind.dtype)
            ds_det = (scroll[1:] - scroll[:-1]).norm(dim=-1).detach() \
                * live_step.to(ind.dtype)
            max_gaps = gaps.shape[1]
            key = s_pid * max_gaps + gap_idx.clamp(max=max_gaps - 1)
            flat = torch.zeros(
                counts.shape[0] * max_gaps, device=device, dtype=ind.dtype)
            gap_mass = flat.clone().index_add_(0, key, dmass)
            gap_len = flat.index_add_(0, key, ds_det)
            lam = (0.5 * gap_mass / gap_len.clamp(min=gap_floor))[key]
            if lambda_mode == 'soft_mass_wide':
                # Hybrid: mass only where the detected gap is wide (where
                # missed sheets live and the soft mass is reliable);
                # inverse-gap keeps the topological 1-winding-per-gap prior
                # at tight spacings, where the soft indicator under-reads
                # (the measured <12 wv estimator blindness).
                wide = step_gap > float(
                    cfg.get('dense_spacing_density_soft_mass_min_gap_wv',
                            20.0))
                lam = torch.where(wide, lam, 1.0 / step_gap)
        else:
            lam = 1.0 / step_gap
        lam = lam * (step_mask[:-1] & ~blind).to(lam.dtype)
    ds = (scroll[1:] - scroll[:-1]).norm(dim=-1)
    num_pairs = int(counts.shape[0])
    with torch.no_grad():
        # blind span in detached model windings; removed from the target so
        # the remaining path is scored against the windings it should hold
        step_dphase = (phase[1:] - phase[:-1]).clamp(min=0.0)
        blind_span = torch.zeros(num_pairs, device=device).index_add_(
            0, s_pid,
            step_dphase * (blind & step_mask[:-1]).to(step_dphase.dtype))
        blind_fraction = blind_span / pair_m.to(blind_span.dtype).clamp(min=1e-6)
        ray_valid = (
            (counts >= 2) & ~rejected
            & (blind_fraction
               <= float(cfg['dense_spacing_density_max_blind_fraction'])))
        target = (pair_m.to(ds.dtype) - blind_span).clamp(min=0.0)
    integral = torch.zeros(
        num_pairs, device=device, dtype=ds.dtype).index_add_(
        0, s_pid, lam * ds)
    residual = (integral - target).abs()
    weight = ray_valid.to(torch.float32)
    loss, _stats = _aggregate_nominal_mass(
        residual, weight, cfg['dense_spacing_support_floor_alpha'])
    if not metrics_enabled():
        return loss, {}
    with torch.no_grad():
        n_valid = weight.sum().clamp(min=1.0)
        metrics = {
            'dense_spacing_density_valid_fraction': float(weight.mean()),
            'dense_spacing_density_blind_fraction_mean': float(
                blind_fraction.mean()),
            'dense_spacing_density_integral_mean': float(
                (integral.detach() * weight).sum() / n_valid),
            'dense_spacing_density_residual_mean': float(
                (residual.detach() * weight).sum() / n_valid),
        }
    return loss, metrics


def _phase_registration_loss(
    slice_to_spiral_transform, dr_per_winding, bands, features, view,
    view_valid, k, pair_m, theta, z, phase_rejected, cfg, *,
    compute_map_path=False,
):
    """Soft-sequence phase registration on detached detected bands.

    The modeled winding targets (mapped through the live inverse transform)
    and the sequence alignment are differentiable; band geometry, reference
    gaps, and every gate/mass are detached, so the run cannot reduce its
    objective by rotating observations or eroding its own confidence.
    ``bands``/``features`` are the precomputed detached detection outputs
    (shared with the density component).
    """
    device = k.device
    num_pairs = k.shape[0]
    n_max = int(pair_m.max().item()) + 1
    offsets_i = torch.arange(n_max, device=device)
    model_valid = offsets_i[None, :] <= pair_m[:, None]
    winding = k[:, None] + offsets_i[None, :].to(k.dtype)
    radius = (winding + (theta / (2 * np.pi))[:, None]) * dr_per_winding.detach()
    # Only the modeled targets participating in phase costs go through the
    # live inverse transform; everything observation-side stays detached.
    target_pair_id = torch.arange(
        num_pairs, device=device).repeat_interleave(n_max)
    target = _map_radial_samples(
        slice_to_spiral_transform, radius.reshape(-1), theta, z,
        target_pair_id, torch.sin(theta), torch.cos(theta),
    ).reshape(num_pairs, n_max, 3)

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

    if not (metrics_enabled() or compute_map_path):
        return loss, {}

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
            'k': k,
            'm': pair_m,
            'model_valid': model_valid,
            'band_valid': features['band_valid'],
            'band_phase': features['phase'],
            'band_out_of_model': out_of_model,
            'rho': rho.detach(),
            'allowed': allowed,
            'top2_margin': alignment['top2_margin'],
            'map_clearance': alignment['map_clearance'],
            'ray_entropy': alignment['ray_entropy'],
        }
    return loss, metrics


def _sheet_walk_offsets(radius, step, device, dtype):
    count = max(1, int(np.ceil(float(radius) / float(step))))
    return torch.linspace(-float(radius), float(radius), 2 * count + 1,
                          device=device, dtype=dtype)


def _sample_sdt_normal_lines(sdt_volume, origins, axes, offsets):
    """Sample parallel per-point lines, returning [points, offsets] tensors."""
    if origins.shape[0] == 0:
        shape = (0, offsets.numel())
        return (torch.empty(shape, device=origins.device, dtype=origins.dtype),
                torch.empty(shape, device=origins.device, dtype=torch.bool))
    samples = origins[:, None, :] + offsets[None, :, None] * axes[:, None, :]
    field, valid, _ = sample_sdt_trilinear(
        sdt_volume, samples.reshape(-1, 3))
    return field.reshape(origins.shape[0], -1), valid.reshape(origins.shape[0], -1)


def _crossing_offset(field, offsets, edge):
    """Linearly interpolate the zero crossing on each selected line edge."""
    row = torch.arange(field.shape[0], device=field.device)
    edge = edge.clamp(min=0, max=field.shape[1] - 2)
    f0 = field[row, edge]
    f1 = field[row, edge + 1]
    # Replace only exact zeros; valid sign-changing edges otherwise have a
    # nonzero denominator.
    denom = f1 - f0
    safe = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
    frac = (-f0 / safe).clamp(0.0, 1.0)
    return torch.lerp(offsets[edge], offsets[edge + 1], frac)


def _transverse_normal(normal):
    """Normal within a constant-z slice; sign remains intentionally arbitrary."""
    transverse = torch.stack([
        torch.zeros_like(normal[:, 0]), normal[:, 1], normal[:, 2]], dim=-1)
    length = transverse.norm(dim=-1)
    return transverse / length.clamp(min=1e-12)[:, None], length


def _initial_sheet_band(sdt_volume, normal_volume, points, radius, sample_step):
    """Find the complete SDT-negative band nearest each source point."""
    normal, normal_valid = sample_lasagna_normals_nearest(normal_volume, points)
    axis, transverse_length = _transverse_normal(normal)
    offsets = _sheet_walk_offsets(radius, sample_step, points.device, points.dtype)
    field, valid = _sample_sdt_normal_lines(sdt_volume, points, axis, offsets)
    inside = valid & (field < 0)
    seed_cost = torch.where(
        inside, offsets.abs()[None, :], torch.full_like(field, float('inf')))
    seed_cost_min, seed = seed_cost.min(dim=1)

    edge_index = torch.arange(offsets.numel() - 1, device=points.device)
    edge_valid = valid[:, :-1] & valid[:, 1:]
    entry = edge_valid & (field[:, :-1] >= 0) & (field[:, 1:] < 0)
    exit = edge_valid & (field[:, :-1] < 0) & (field[:, 1:] >= 0)
    before_seed = edge_index[None, :] < seed[:, None]
    at_or_after_seed = edge_index[None, :] >= seed[:, None]
    left_edge = torch.where(
        entry & before_seed, edge_index[None, :],
        torch.full_like(edge_index[None, :], -1)).max(dim=1).values
    right_edge = torch.where(
        exit & at_or_after_seed, edge_index[None, :],
        torch.full_like(edge_index[None, :], offsets.numel())).min(dim=1).values

    safe_left = left_edge.clamp(min=0, max=offsets.numel() - 2)
    safe_right = right_edge.clamp(min=0, max=offsets.numel() - 2)
    left_offset = _crossing_offset(field, offsets, safe_left)
    right_offset = _crossing_offset(field, offsets, safe_right)
    left = points + left_offset[:, None] * axis
    right = points + right_offset[:, None] * axis
    center = (left + right) * 0.5
    width = (right - left).norm(dim=-1)

    sample_index = torch.arange(offsets.numel(), device=points.device)
    interior = ((sample_index[None, :] > safe_left[:, None])
                & (sample_index[None, :] <= safe_right[:, None]))
    complete_interior = ((~interior) | inside).all(dim=1)
    alive = (
        normal_valid & (transverse_length > 0.25)
        & torch.isfinite(seed_cost_min)
        & (left_edge >= 0) & (right_edge < offsets.numel() - 1)
        & complete_interior & (width > 0.5)
    )
    actual_axis = (right - left) / width.clamp(min=1e-12)[:, None]
    return center, left, right, actual_axis, width, alive


def _refine_predicted_boundary(
    sdt_volume, predicted, axis, radius, sample_step, *, entering,
):
    """Continue one oriented zero-crossing inside a tight trust bracket."""
    offsets = _sheet_walk_offsets(radius, sample_step, predicted.device,
                                  predicted.dtype)
    field, valid = _sample_sdt_normal_lines(
        sdt_volume, predicted, axis, offsets)
    edge_valid = valid[:, :-1] & valid[:, 1:]
    if entering:
        crossing = edge_valid & (field[:, :-1] >= 0) & (field[:, 1:] < 0)
    else:
        crossing = edge_valid & (field[:, :-1] < 0) & (field[:, 1:] >= 0)
    any_crossing = edge_valid & ((field[:, :-1] < 0) != (field[:, 1:] < 0))
    edge_mid = (offsets[:-1] + offsets[1:]) * 0.5
    score = torch.where(
        crossing, edge_mid.abs()[None, :],
        torch.full_like(field[:, :-1], float('inf')))
    _, edge = score.min(dim=1)
    offset = _crossing_offset(field, offsets, edge)
    point = predicted + offset[:, None] * axis
    # Any additional crossing in one small trust region indicates a hole,
    # split, or another sheet. Fail closed instead of choosing one.
    ok = (crossing.sum(dim=1) == 1) & (any_crossing.sum(dim=1) == 1)
    return point, ok


def _advance_sheet_band(
    sdt_volume, normal_volume, center, left, right, axis, width, alive,
    dz_total, *, substep_wv, boundary_search_wv, sample_step_wv,
):
    """Paired-boundary predictor/corrector; returns the complete final state."""
    max_abs = float(dz_total.abs().max()) if dz_total.numel() else 0.0
    n_substeps = max(1, int(np.ceil(max_abs / float(substep_wv))))
    dz = dz_total / n_substeps
    for _ in range(n_substeps):
        normal, normal_valid = sample_lasagna_normals_nearest(
            normal_volume, center)
        _, transverse_length = _transverse_normal(normal)
        transverse_sq = transverse_length.square()
        # Minimum-norm tangent with exactly one unit of z motion:
        # n dot tangent == 0 and tangent[z] == 1. Normal sign cancels.
        tangent = torch.stack([
            torch.ones_like(dz),
            -normal[:, 0] * normal[:, 1] / transverse_sq.clamp(min=1e-12),
            -normal[:, 0] * normal[:, 2] / transverse_sq.clamp(min=1e-12),
        ], dim=-1)
        displacement = dz[:, None] * tangent
        predicted_center = center + displacement
        predicted_left = left + displacement
        predicted_right = right + displacement

        predicted_normal, predicted_normal_valid = \
            sample_lasagna_normals_nearest(normal_volume, predicted_center)
        predicted_axis, predicted_transverse = _transverse_normal(predicted_normal)
        flip = (predicted_axis * axis).sum(dim=-1) < 0
        predicted_axis = torch.where(
            flip[:, None], -predicted_axis, predicted_axis)

        new_left, left_ok = _refine_predicted_boundary(
            sdt_volume, predicted_left, predicted_axis,
            boundary_search_wv, sample_step_wv, entering=True)
        new_right, right_ok = _refine_predicted_boundary(
            sdt_volume, predicted_right, predicted_axis,
            boundary_search_wv, sample_step_wv, entering=False)
        new_center = (new_left + new_right) * 0.5
        new_width = (new_right - new_left).norm(dim=-1)
        new_axis = ((new_right - new_left)
                    / new_width.clamp(min=1e-12)[:, None])
        center_field, center_valid, _ = sample_sdt_trilinear(
            sdt_volume, new_center)

        width_allowance = torch.maximum(
            torch.full_like(width, 2.0), width * 0.5)
        correction = (new_center - predicted_center).norm(dim=-1)
        axis_alignment = (new_axis * predicted_axis).sum(dim=-1)
        step_ok = (
            alive & normal_valid & predicted_normal_valid
            & (transverse_length > 0.25) & (predicted_transverse > 0.25)
            & left_ok & right_ok & center_valid & (center_field < 0)
            & (new_width > 0.5)
            & ((new_width - width).abs() <= width_allowance)
            & (correction <= float(boundary_search_wv))
            & (axis_alignment > 0.75)
        )
        center = torch.where(step_ok[:, None], new_center, center)
        left = torch.where(step_ok[:, None], new_left, left)
        right = torch.where(step_ok[:, None], new_right, right)
        axis = torch.where(step_ok[:, None], new_axis, axis)
        width = torch.where(step_ok, new_width, width)
        alive = step_ok
    return center, left, right, axis, width, alive


def sheet_walk(
    sdt_volume, normal_volume, points_zyx, dz_total, *, substep_wv=4.0,
    band_search_wv=8.0, boundary_search_wv=3.0, sample_step_wv=1.0,
    validate_roundtrip=True, roundtrip_tolerance_wv=1.0,
):
    """Track the medial center of the same physical SDT band through z.

    This observation-side primitive is fully detached. It initializes the
    nearest complete positive->negative->positive SDT interval along the
    in-slice component of the Lasagna normal, then continues its TWO oriented
    zero-crossing boundaries independently. Each substep uses the full
    Lasagna normal for a sheet-tangent predictor and tightly brackets the
    predicted entry and exit crossings; their midpoint is the new band center.
    Ambiguous, missing, implausibly moving, or non-closing bands die rather
    than falling back to a nearby sheet.

    ``dz_total`` may be scalar or per-point. When ``validate_roundtrip`` is
    true (the default), the final paired-boundary state is walked back by the
    opposite displacement and retained only if it closes within
    ``roundtrip_tolerance_wv``. Returns ``(endpoints, alive)``.
    """
    with torch.no_grad():
        device = points_zyx.device
        num = points_zyx.shape[0]
        if num == 0:
            return points_zyx.clone(), torch.empty(
                [0], dtype=torch.bool, device=device)
        if substep_wv <= 0 or band_search_wv <= 0 \
                or boundary_search_wv <= 0 or sample_step_wv <= 0:
            raise ValueError('sheet-walk distances and sampling step must be positive')
        if roundtrip_tolerance_wv < 0:
            raise ValueError('roundtrip_tolerance_wv must be nonnegative')
        if not torch.is_tensor(dz_total):
            dz_total = torch.full(
                (num,), float(dz_total), device=device,
                dtype=points_zyx.dtype)
        else:
            dz_total = dz_total.to(device=device, dtype=points_zyx.dtype)
            if dz_total.numel() == 1:
                dz_total = dz_total.reshape(1).expand(num)
            elif dz_total.numel() == num:
                dz_total = dz_total.reshape(num)
            else:
                raise ValueError(
                    'dz_total must be scalar or have one value per point')
        state = _initial_sheet_band(
            sdt_volume, normal_volume, points_zyx,
            float(band_search_wv), float(sample_step_wv))
        source_center = state[0]
        end_state = _advance_sheet_band(
            sdt_volume, normal_volume, *state, dz_total,
            substep_wv=float(substep_wv),
            boundary_search_wv=float(boundary_search_wv),
            sample_step_wv=float(sample_step_wv))
        endpoint, alive = end_state[0], end_state[-1]
        if validate_roundtrip and num:
            back_state = _advance_sheet_band(
                sdt_volume, normal_volume, *end_state, -dz_total,
                substep_wv=float(substep_wv),
                boundary_search_wv=float(boundary_search_wv),
                sample_step_wv=float(sample_step_wv))
            closes = ((back_state[0] - source_center).norm(dim=-1)
                      <= float(roundtrip_tolerance_wv))
            alive = alive & back_state[-1] & closes
        return endpoint, alive


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

    if not metrics_enabled():
        return loss, dict(diagnostics), residual, weight

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
    """The phase-bundle component weights; zero disables a component."""
    return {
        'dense_spacing_phase': float(cfg['loss_weight_dense_spacing']),
        'dense_spacing_count': float(cfg['loss_weight_dense_spacing_count']),
        # Metric density integral over detached band-derived inverse gaps;
        # shares the band-detection machinery with phase but not the
        # pair-HMM, so it stays cheap at high sampling.
        'dense_spacing_density': float(
            cfg.get('loss_weight_dense_spacing_density', 0.0)),
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
    density_active = weights.get('dense_spacing_density', 0) > 0
    phase_active = weights['dense_spacing_phase'] > 0
    # Density derives its detached density field from detected bands, so
    # band detection runs for either consumer; the pair-HMM alignment itself
    # runs only when phase is weighted (the density fast path).
    detection_active = phase_active or density_active
    inner, outer = fitted_winding_domain(outer_winding_idx)
    if outer - inner < 1:
        return
    started = time.perf_counter()
    num_pairs = int(cfg['dense_spacing_num_pairs'])
    k, pair_m, theta, z = _sample_spacing_pairs(
        cfg, inner, outer, num_pairs, device, z_begin, z_end, generator)
    m_f = pair_m.to(k.dtype)

    build_grad = (count_active or density_active) and torch.is_grad_enabled()
    central_ray = sample_pair_polylines(
        slice_to_spiral_transform, dr_per_winding, k, k + m_f, theta, z, cfg,
        no_grad=not build_grad)
    sdt_started = time.perf_counter()
    central_field, central_valid, _ = sample_sdt_trilinear(
        sdt_volume, central_ray['scroll_poly'])
    sdt_sample_seconds = time.perf_counter() - sdt_started

    view = view_field = view_valid = None
    pad_rejected = None
    if detection_active:
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
            'dense_spacing_count_sdt_sample_seconds': sdt_sample_seconds,
        }
        if metrics_enabled():
            diagnostics.update({
                'dense_spacing_count_too_long_fraction': float(
                    pair['too_long'].float().mean().item()),
                'dense_spacing_count_step_violation_fraction': float(
                    pair['step_violation'].float().mean().item()),
            })
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
        # The count, phase, and shared-batch density losses backpropagate
        # through one central-ray graph; the tag lets the training loop sum
        # them into a single backward instead of traversing that graph once
        # per component (metrics consumers must pop/ignore '_'-keys).
        count_metrics['_shared_graph'] = True
        spiral_mid = torch.stack([
            torch.cat(z_all),
            torch.sin(torch.cat(theta_all)) * torch.cat(r_mid_all),
            torch.cos(torch.cat(theta_all)) * torch.cat(r_mid_all),
        ], dim=-1)
        record_loss_samples(
            'dense_spacing_count', spiral_mid, residual, weight > 0)
        yield 'dense_spacing_count', count_loss, count_metrics
        del pair, counts, residual

    if detection_active:
        # Central live tensors are no longer needed once the detached view
        # exists; drop them so the count backward can free its graph. The
        # density component keeps only the live pieces it integrates over.
        central_rejected = central_ray['too_long'] | central_ray['step_violation']
        phase_rejected = central_rejected | pad_rejected
        density_central = None
        if density_active and build_grad:
            density_central = {
                'scroll_poly': central_ray['scroll_poly'],
                'sample_phase': central_ray['sample_phase'],
                'pair_id': central_ray['pair_id'],
                'has_successor': central_ray['has_successor'],
                'field': central_field.detach(),
            }
        central_metrics = {
            'dense_spacing_phase_sdt_sample_seconds': sdt_sample_seconds,
        }
        if metrics_enabled():
            central_metrics.update({
                'dense_spacing_phase_too_long_fraction': float(
                    central_ray['too_long'].float().mean().item()),
                'dense_spacing_phase_step_violation_fraction': float(
                    central_ray['step_violation'].float().mean().item()),
                'dense_spacing_phase_pad_rejected_fraction': float(
                    pad_rejected.float().mean().item()),
                'dense_spacing_phase_extension_domain_clamped_fraction':
                    float(((k + m_f + float(
                        cfg['dense_spacing_phase_extension_windings']))
                        > float(outer)).float().mean().item()),
            })
        del central_ray, central_field, central_valid
        bands = detect_complete_sdt_bands(
            view, view_field, view_valid, normal_volume, cfg)
        features = _phase_band_features(bands, num_pairs, device)
        density = None
        if density_central is not None:
            density = _metric_density_loss(
                density_central, features, pair_m, phase_rejected, cfg)
            del density_central
        if phase_active:
            phase_loss, phase_metrics = _phase_registration_loss(
                slice_to_spiral_transform, dr_per_winding, bands, features,
                view, view_valid, k, pair_m, theta, z, phase_rejected, cfg,
                compute_map_path=compute_map_path)
            phase_metrics.update(central_metrics)
            phase_metrics['dense_spacing_phase_wall_seconds'] = (
                time.perf_counter() - started)
            phase_metrics['_shared_graph'] = True
            yield 'dense_spacing_phase', phase_loss, phase_metrics
        del bands, features, view, view_field, view_valid
        if density is not None:
            density_loss, density_metrics = density
            if not phase_active:
                # the shared-batch health metrics otherwise ride with phase
                density_metrics.update(central_metrics)
            extra_pairs = int(cfg['dense_spacing_density_extra_pairs'])
            total_density_pairs = num_pairs + max(0, extra_pairs)
            share = num_pairs / total_density_pairs
            density_metrics['_shared_graph'] = True
            yield ('dense_spacing_density', density_loss * share,
                   density_metrics)
            # Density-only supplement: the fast path (polylines + band
            # detection, no pair-HMM) is cheap enough to run at several times
            # the shared-batch sampling; chunked so only one chunk's graph is
            # resident per backward.
            chunk_cap = int(cfg['dense_spacing_density_chunk_pairs'])
            remaining = max(0, extra_pairs)
            while remaining > 0:
                n_chunk = min(chunk_cap, remaining)
                remaining -= n_chunk
                chunk_loss, chunk_metrics = _density_only_batch(
                    slice_to_spiral_transform, dr_per_winding, sdt_volume,
                    normal_volume, cfg, inner, outer, n_chunk, device,
                    z_begin, z_end, generator)
                yield ('dense_spacing_density',
                       chunk_loss * (n_chunk / total_density_pairs),
                       chunk_metrics)


def _density_only_batch(
    slice_to_spiral_transform, dr_per_winding, sdt_volume, normal_volume,
    cfg, inner, outer, num_pairs, device, z_begin, z_end, generator,
):
    """One density-only pair batch: live polylines + band detection, no
    count and no pair-HMM. This is the cheap path that lets density sample
    far above the shared-batch budget."""
    k, pair_m, theta, z = _sample_spacing_pairs(
        cfg, inner, outer, num_pairs, device, z_begin, z_end, generator)
    m_f = pair_m.to(k.dtype)
    central = sample_pair_polylines(
        slice_to_spiral_transform, dr_per_winding, k, k + m_f, theta, z, cfg)
    # The field feeds only the (detached) detection view here — the density
    # loss differentiates through scroll_poly alone — so skip the trilinear
    # autograd graph; values are identical.
    with torch.no_grad():
        field, valid, _ = sample_sdt_trilinear(
            sdt_volume, central['scroll_poly'])
    pads, pad_rejected, _ = _build_phase_padding(
        slice_to_spiral_transform, dr_per_winding, sdt_volume, k, pair_m,
        theta, z, inner, outer, cfg)
    view, view_field, view_valid = _assemble_detection_view(
        central, field, valid, pads)
    del pads
    bands = detect_complete_sdt_bands(
        view, view_field, view_valid, normal_volume, cfg)
    features = _phase_band_features(bands, num_pairs, device)
    rejected = central['too_long'] | central['step_violation'] | pad_rejected
    density_central = {
        'scroll_poly': central['scroll_poly'],
        'sample_phase': central['sample_phase'],
        'pair_id': central['pair_id'],
        'has_successor': central['has_successor'],
        'field': field.detach(),
    }
    return _metric_density_loss(
        density_central, features, pair_m, rejected, cfg)


def iter_phase_bundle_losses(
    spiral_and_transform, slice_to_spiral_transform, dr_per_winding,
    sdt_volume, normal_volume, outer_winding_idx, cfg, z_begin, z_end, *,
    attachment_ramp=1.0, generator=None, compute_map_path=False,
):
    """Yield the active phase-bundle components as (name, loss, metrics).

    The bundle is the single ``phase`` dense-spacing mode: soft-sequence
    phase registration, crossing count (sharing the phase rays), native
    minimum spacing, and SDT attachment. Yielding lazily lets the training
    loop run one backward per graph so at most one large graph is resident;
    components whose metrics carry ``'_shared_graph': True`` (count, phase,
    shared-batch density) share the central-ray graph and should be summed
    into one backward. The public mode stays ``phase`` regardless of this
    internal backward grouping.
    """
    global _metrics_enabled_now
    _metrics_enabled_now = next(_metrics_tick) % _METRICS_EVERY == 0
    weights = phase_bundle_component_weights(cfg, attachment_ramp)
    if outer_winding_idx is not None and sdt_volume is not None:
        if sdt_volume['kind'] != 'sdt':
            raise ValueError('the phase bundle requires a signed-distance store')
        if (((weights['dense_spacing_phase'] > 0
              or weights['dense_spacing_density'] > 0)
             and normal_volume is not None)
                or weights['dense_spacing_count'] > 0):
            effective = dict(weights)
            if normal_volume is None:
                effective['dense_spacing_phase'] = 0.0
                effective['dense_spacing_density'] = 0.0
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
    if not metrics_enabled():
        return loss, {}
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
