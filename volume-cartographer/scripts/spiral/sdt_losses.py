"""Surf-SDT-driven dense losses (docs/spiral_pred_dt_dense_spacing.md).

Three pieces, all driven by one capped signed-distance store of the binarized
surface prediction (positive outside, negative inside, encoded 0 = no-data):

- crossing-count spacing: a soft sheet-crossing count along the mapped
  polyline between integer winding pairs (k, k + m), residual |count - m|;
- attachment: smooth-L1 of the exterior distance at dense points on fitted
  winding surfaces;
- a detached endpoint support gate on the spacing loss with a nominal-mass
  denominator floor, so spacing scales down (to a defined zero) when support
  is scarce instead of collapsing back to an ordinary mean.

The spacing and attachment losses are independent: each has its own weight,
sample count, and enablement, and neither reuses the other's sample points.
"""

import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

from ddp_helpers import get_world_size, is_distributed
from loss_maps import diagnostics_enabled, record_loss_samples


# A sample is valid when the total trilinear weight of its valid corners is at
# least this mass (weight mass, not corner count: four valid corners can carry
# near-zero weight when the sample point sits in the invalid octant).
MIN_VALID_CORNER_WEIGHT_MASS = 0.5

# Raw-surf counting fallback: indicator temperature in raw uint8 units.
# 128-centered soft counting was validated to measure identically to
# threshold-150/200 hard counting (store re-measurement, 2026-07-16).
SURF_INDICATOR_TEMPERATURE = 20.0

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


def _inside_indicator(field_value, count_volume, s_count_wv):
    if count_volume['kind'] == 'sdt':
        return torch.sigmoid(-field_value / s_count_wv)
    return torch.sigmoid(field_value / SURF_INDICATOR_TEMPERATURE)


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


def compute_pair_counts(
    slice_to_spiral_transform, dr_per_winding, count_volume, sdt_volume,
    winding_idx, pair_m, theta, z, cfg,
):
    """Soft crossing counts along the mapped inter-winding polylines.

    Shared by the training loss and the per-pair aggregation diagnostic. Each
    pair (k, k + m) at (z, theta) becomes a radial segment in spiral space,
    inverse-mapped as a full polyline (never the endpoint chord). Each
    constituent winding gap gets its own detached length estimate and step
    allocation targeting ~1 working voxel between mapped samples, so one
    unusually wide gap is not undersampled because its neighbours are short.
    Pairs whose required step count exceeds dense_spacing_max_steps, or whose
    mapped adjacent samples still exceed dense_spacing_max_step_wv, are marked
    invalid and reported - they are never evaluated at a spacing that could
    step over a sheet.

    Returns a dict of per-pair tensors: count (differentiable), residual
    target m, seg_valid, support (detached), too_long, step_violation, plus
    the flat sampled field for diagnostics.
    """
    device = z.device
    num_pairs = winding_idx.shape[0]
    dr = dr_per_winding.detach()

    s_count = float(cfg['dense_spacing_count_temperature_wv'])
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    m_f = pair_m.to(torch.float32)
    r0 = (winding_idx + theta / (2 * np.pi)) * dr
    r1 = r0 + m_f * dr
    ray = sample_pair_polylines(
        slice_to_spiral_transform, dr_per_winding, winding_idx,
        winding_idx + m_f, theta, z, cfg)
    scroll_poly = ray['scroll_poly']
    pair_id = ray['pair_id']
    offsets = ray['offsets']
    samples_per_pair = ray['samples_per_pair']
    has_successor = ray['has_successor']
    too_long = ray['too_long']
    step_violation = ray['step_violation']

    field_value, sample_valid, _ = sample_sdt_trilinear(count_volume, scroll_poly)
    indicator = _inside_indicator(field_value, count_volume, s_count)

    diffs = (indicator[1:] - indicator[:-1]).abs()
    diffs = diffs * has_successor[:-1].to(diffs.dtype)
    count = 0.5 * torch.zeros(num_pairs, device=device, dtype=diffs.dtype).index_add_(
        0, pair_id[:-1], diffs)

    # Whole-segment gating: a partially covered path undercounts and unfairly
    # compares against m.
    valid_samples_per_pair = torch.zeros(
        num_pairs, device=device, dtype=torch.long).index_add_(
        0, pair_id, sample_valid.to(torch.long))
    all_samples_valid = valid_samples_per_pair == samples_per_pair
    seg_valid = all_samples_valid & ~too_long & ~step_violation

    # Detached endpoint support from the SDT's exterior distance. stop_gradient
    # is required: otherwise the optimiser can lower spacing loss by moving
    # away from the predicted sheets and eroding its own support. Skipped
    # entirely when the gate is off - with a separate count source that would
    # otherwise cost a second full store gather per step.
    first = offsets[:-1]
    last = offsets[1:] - 1
    with torch.no_grad():
        sigma = float(cfg['dense_spacing_support_sigma'])
        if sdt_volume is None or not cfg['dense_spacing_use_support_gate']:
            support = torch.ones(num_pairs, device=device)
        else:
            if sdt_volume is count_volume:
                sd_first, sd_last = field_value[first], field_value[last]
                valid_first, valid_last = sample_valid[first], sample_valid[last]
            else:
                endpoint_points = torch.cat(
                    [scroll_poly[first], scroll_poly[last]], dim=0).detach()
                sd_e, valid_e, _ = sample_sdt_trilinear(sdt_volume, endpoint_points)
                sd_first, sd_last = sd_e[:num_pairs], sd_e[num_pairs:]
                valid_first, valid_last = valid_e[:num_pairs], valid_e[num_pairs:]
            support_first = torch.exp(-(F.relu(sd_first) / sigma) ** 2)
            support_last = torch.exp(-(F.relu(sd_last) / sigma) ** 2)
            if cfg['dense_spacing_support_policy'] == 'minimum':
                support = torch.minimum(support_first, support_last)
            else:
                support = support_first * support_last
            support = support * (valid_first & valid_last).to(support.dtype)

    return {
        'count': count,
        'target_m': m_f,
        'seg_valid': seg_valid,
        'support': support,
        'too_long': too_long,
        'step_violation': step_violation,
        'field_value': field_value,
        'sample_valid': sample_valid,
        'r_mid': (r0 + r1) / 2,
    }


def get_crossing_count_spacing_loss(
    slice_to_spiral_transform, dr_per_winding, count_volume, sdt_volume,
    outer_winding_idx, cfg, z_begin, z_end,
):
    """Support-gated soft crossing-count spacing loss.

    L = sum(w * |count - m|) / max(sum(w), alpha * num_pairs): the nominal-mass
    floor makes the loss scale down (reaching a defined zero) when total
    support falls below alpha of the batch, instead of a scale-invariant
    weighted mean that would fire at full strength on uniformly tiny support.
    Deliberately, well-supported pairs are diluted in sparse-support batches -
    spacing must not dominate when support is scarce.
    """
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if count_volume is None or outer_winding_idx is None:
        return zero, {}

    num_pairs = int(cfg['dense_spacing_num_pairs'])
    inner_winding, outer_winding = fitted_winding_domain(outer_winding_idx)
    domain_span = outer_winding - inner_winding
    if domain_span < 1:
        return zero, {}
    # m is a two-range mixture biased short: most pairs span a few windings
    # (localized residuals, cheap rays), the rest reach further so rays can
    # straddle wide unsupported regions. Both ranges are clamped so every
    # sampled pair keeps both windings inside the fitted domain
    # (k >= inner and k + m <= outer).
    def clamped_m_range(bounds):
        hi = min(int(bounds[1]), domain_span)
        lo = min(max(1, int(bounds[0])), hi)
        return lo, hi

    short_lo, short_hi = clamped_m_range(cfg['dense_spacing_pair_m_short'])
    long_lo, long_hi = clamped_m_range(cfg['dense_spacing_pair_m_long'])
    long_fraction = float(cfg['dense_spacing_pair_long_fraction'])
    short_m = torch.randint(short_lo, short_hi + 1, [num_pairs], device=device)
    long_m = torch.randint(long_lo, long_hi + 1, [num_pairs], device=device)
    use_long = torch.rand(num_pairs, device=device) < long_fraction
    pair_m = torch.where(use_long, long_m, short_m)
    highest_k = (outer_winding - pair_m).to(torch.float32)
    winding_idx = _sample_windings_by_circumference(
        inner_winding, highest_k, num_pairs, device)
    theta = torch.rand(num_pairs, device=device) * (2 * np.pi)
    z = torch.empty(num_pairs, device=device).uniform_(float(z_begin), float(z_end - 1))

    pair = compute_pair_counts(
        slice_to_spiral_transform, dr_per_winding, count_volume, sdt_volume,
        winding_idx, pair_m, theta, z, cfg)

    # `support` is all-ones when the gate is disabled (compute_pair_counts
    # skips the endpoint sampling entirely then).
    weight = (pair['support'] * pair['seg_valid'].to(torch.float32)).detach()

    residual = (pair['count'] - pair['target_m']).abs()
    numerator = (weight * residual).sum()
    stats = torch.stack([
        weight.sum(), torch.tensor(float(num_pairs), device=device)])
    world_size = 1
    if is_distributed() and torch.is_grad_enabled():
        # The all-reduce makes the nominal-mass floor global. It must never
        # run on the no-grad preview/diagnostic path, which executes on the
        # publishing rank only - an unmatched collective would deadlock the
        # other ranks.
        import torch.distributed as dist
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
    support_mass, nominal = stats[0], stats[1]
    alpha = float(cfg['dense_spacing_support_floor_alpha'])
    denominator = torch.maximum(support_mass, alpha * nominal)
    # Gradients are averaged across ranks; scaling each rank's local numerator
    # by world size makes that average equal the intended global
    # sum(w * r) / max(sum(w), alpha * N) semantics.
    loss = world_size * numerator / denominator

    with torch.no_grad():
        # One stacked device->host transfer for all scalar diagnostics; a
        # per-metric .item() would serialize ~10 GPU syncs per step.
        valid_f = pair['seg_valid'].to(torch.float32)
        n_valid = valid_f.sum()
        n_valid_floor = n_valid.clamp(min=1.0)
        (mass_value, nominal_value, valid_fraction, too_long_fraction,
         step_violation_fraction, count_mean, residual_mean, n_valid_value) = torch.stack([
            support_mass,
            nominal,
            valid_f.mean(),
            pair['too_long'].to(torch.float32).mean(),
            pair['step_violation'].to(torch.float32).mean(),
            (pair['count'] * valid_f).sum() / n_valid_floor,
            (residual * valid_f).sum() / n_valid_floor,
            n_valid,
        ]).tolist()
        floor_active = mass_value < alpha * nominal_value
        mass_fraction = mass_value / max(1.0, nominal_value)
        metrics = {
            'dense_spacing_support_mass': mass_value,
            'dense_spacing_support_mass_fraction': mass_fraction,
            'dense_spacing_floor_active': float(floor_active),
            'dense_spacing_valid_fraction': valid_fraction,
            'dense_spacing_too_long_fraction': too_long_fraction,
            'dense_spacing_step_violation_fraction': step_violation_fraction,
        }
        if n_valid_value > 0:
            metrics['dense_spacing_count_mean'] = count_mean
            metrics['dense_spacing_count_residual_mean'] = residual_mean
        metrics.update({
            f'dense_spacing_sdt_{name}': value
            for name, value in sdt_sample_fractions(
                pair['field_value'], pair['sample_valid'], count_volume).items()
        })
        global _floor_was_active
        if floor_active and not _floor_was_active:
            print('dense_spacing: support-mass floor active '
                  f'(mass fraction {mass_fraction:.4f} < alpha {alpha}); '
                  'the spacing loss is effectively scaled down until support recovers')
        elif _floor_was_active and not floor_active:
            print('dense_spacing: support-mass floor released '
                  f'(mass fraction {mass_fraction:.4f} >= alpha {alpha})')
        _floor_was_active = floor_active

    spiral_mid = torch.stack([
        z, torch.sin(theta) * pair['r_mid'], torch.cos(theta) * pair['r_mid'],
    ], dim=-1)
    record_loss_samples('dense_spacing', spiral_mid, residual, weight > 0)

    return loss, metrics


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
def detect_complete_sdt_bands(ray, sdt_volume, normal_volume, cfg):
    """Detect detached complete outside->inside->outside SDT intervals.

    Leading/trailing partial intervals are never closed at ray endpoints. The
    returned packed band tensors are ordered by pair then outward arclength.
    """
    scroll = ray['scroll_poly']
    pair_id = ray['pair_id']
    has_successor = ray['has_successor']
    field, sample_valid, _ = sample_sdt_trilinear(sdt_volume, scroll)
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
            'ambiguous': empty.bool(),
            'unsupported_before': empty.bool(), 'merged_count': 0,
            'field_value': field, 'sample_valid': sample_valid,
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
            'ambiguous': empty.bool(),
            'unsupported_before': empty.bool(), 'merged_count': 0,
            'field_value': field, 'sample_valid': sample_valid,
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

    def geometry(e_arc, x_arc):
        center_arc = (e_arc + x_arc) * 0.5
        center, phase, direction = _interpolate_at_global_arclength(
            ray, global_arc, center_arc)
        normal, normal_valid = sample_lasagna_normals_nearest(
            normal_volume, center)
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
    ambiguous = torch.zeros_like(normal_valid)
    ambiguous_split = close & (
        unsupported_before[1:] | ~normal_valid[:-1] | ~normal_valid[1:])
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
        unsupported_before = unsupported_before[keep]
        center_arc, center, phase, direction, normal_dot, normal_valid = geometry(
            entry_arc, exit_arc)

    counts = torch.zeros(
        ray['too_long'].numel(), dtype=torch.long, device=field.device)
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
        'ambiguous': ambiguous, 'unsupported_before': unsupported_before,
        'merged_count': merged_count, 'field_value': field,
        'sample_valid': sample_valid,
    }


def _select_nearest_band(bands, target_phase, num_pairs, margin, max_distance):
    device = target_phase.device
    pair = bands['pair_id']
    inf = torch.tensor(float('inf'), device=device)
    minimum = torch.full([num_pairs], float('inf'), device=device)
    if pair.numel():
        distance = (bands['phase'] - target_phase[pair]).abs()
        minimum.scatter_reduce_(0, pair, distance, reduce='amin', include_self=True)
        flat_index = torch.arange(pair.numel(), device=device)
        candidate = torch.full(
            [num_pairs], pair.numel(), dtype=torch.long, device=device)
        is_min = distance == minimum[pair]
        candidate.scatter_reduce_(
            0, pair[is_min], flat_index[is_min], reduce='amin', include_self=True)
        safe_candidate = candidate.clamp(max=max(0, pair.numel() - 1))
        second_values = torch.where(
            flat_index == safe_candidate[pair], inf, distance)
        second = torch.full([num_pairs], float('inf'), device=device)
        second.scatter_reduce_(
            0, pair, second_values, reduce='amin', include_self=True)
    else:
        candidate = torch.zeros([num_pairs], dtype=torch.long, device=device)
        second = minimum.clone()
    no_band = ~torch.isfinite(minimum)
    ambiguity = ~no_band & ((second - minimum) < margin)
    too_far = ~no_band & (minimum > max_distance)
    return candidate, minimum, no_band, ambiguity, too_far


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


def get_phase_spacing_loss(
    slice_to_spiral_transform, dr_per_winding, sdt_volume, normal_volume,
    outer_winding_idx, cfg, z_begin, z_end, *, generator=None,
):
    """Single-ray complete-band phase registration loss and diagnostics."""
    started = time.perf_counter()
    device = dr_per_winding.device
    zero = torch.zeros([], device=device)
    if sdt_volume is None or normal_volume is None or outer_winding_idx is None:
        return zero, {}
    if sdt_volume['kind'] != 'sdt':
        raise ValueError('phase spacing requires a signed-distance store')
    num_pairs = int(cfg['dense_spacing_num_pairs'])
    inner, outer = fitted_winding_domain(outer_winding_idx)
    if outer - inner < 1:
        return zero, {}

    k, pair_m, theta, z = _sample_spacing_pairs(
        cfg, inner, outer, num_pairs, device, z_begin, z_end, generator)
    extension = float(cfg['dense_spacing_phase_extension_windings'])
    phase_start = (k - extension).clamp(min=float(inner))
    phase_end = torch.minimum(
        k + pair_m.to(k.dtype) + extension,
        torch.full_like(k, float(outer)))
    ray = sample_pair_polylines(
        slice_to_spiral_transform, dr_per_winding, phase_start, phase_end,
        theta, z, cfg, no_grad=True)
    bands = detect_complete_sdt_bands(ray, sdt_volume, normal_volume, cfg)
    pair = bands['pair_id']
    n_bands = pair.numel()
    margin = float(cfg['dense_spacing_phase_anchor_margin'])
    max_distance = float(cfg['dense_spacing_phase_anchor_max_distance'])
    anchor, _, no_band, anchor_ambiguous, anchor_far = _select_nearest_band(
        bands, k, num_pairs, margin, max_distance)
    outer_candidate, _, outer_no_band, outer_ambiguous, outer_far = _select_nearest_band(
        bands, k + pair_m, num_pairs, margin, max_distance)

    ray_rejected = ray['too_long'] | ray['step_violation']
    if n_bands:
        safe_anchor = anchor.clamp(max=n_bands - 1)
        anchor_supported = (
            bands['normal_valid'][safe_anchor] & ~bands['ambiguous'][safe_anchor])
        anchor_supported &= ~no_band
        anchor_ok = (
            ~ray_rejected & ~no_band & ~anchor_ambiguous & ~anchor_far
            & anchor_supported)
        anchor_ordinal = bands['ordinal'][safe_anchor]
        j = bands['ordinal'] - anchor_ordinal[pair]
        outward = anchor_ok[pair] & (j >= 0)

        same_prev = torch.zeros(n_bands, dtype=torch.bool, device=device)
        same_prev[1:] = pair[1:] == pair[:-1]
        phase_gap = torch.zeros(n_bands, device=device)
        phase_gap[1:] = bands['phase'][1:] - bands['phase'][:-1]
        wide_before = same_prev & (
            phase_gap >= float(cfg['dense_spacing_phase_censor_gap_windings']))
        graze = bands['graze']
        bad_band = graze | ~bands['normal_valid'] | bands['ambiguous']
        censor_here = outward & (
            bad_band | ((j > 0) & (wide_before | bands['unsupported_before'])))
        censor_cumulative = censor_here.to(torch.long).cumsum(0)
        before_anchor = torch.zeros(num_pairs, dtype=torch.long, device=device)
        valid_anchor_indices = anchor_ok.nonzero(as_tuple=False).squeeze(-1)
        if valid_anchor_indices.numel():
            ai = anchor[valid_anchor_indices]
            before_anchor[valid_anchor_indices] = torch.where(
                ai > 0, censor_cumulative[(ai - 1).clamp(min=0)],
                torch.zeros_like(ai))
        censored_since_anchor = censor_cumulative - before_anchor[pair]
        target_winding = k[pair] + j.to(k.dtype)
        accepted = (
            outward & (censored_since_anchor == 0)
            & (target_winding >= inner) & (target_winding <= outer))

        accepted_gap = torch.zeros(n_bands, dtype=torch.bool, device=device)
        accepted_gap[1:] = (
            accepted[1:] & accepted[:-1] & same_prev[1:]
            & (j[1:] == j[:-1] + 1))
        gap_length = torch.zeros(n_bands, device=device)
        gap_length[1:] = bands['center_arc'][1:] - bands['center_arc'][:-1]
        gref_sum = torch.zeros(n_bands, device=device)
        gref_count = torch.zeros(n_bands, device=device)
        gap_right = accepted_gap.nonzero(as_tuple=False).squeeze(-1)
        if gap_right.numel():
            lengths = gap_length[gap_right]
            both = torch.cat([gap_right - 1, gap_right])
            twice = torch.cat([lengths, lengths])
            gref_sum.index_add_(0, both, twice)
            gref_count.index_add_(0, both, torch.ones_like(twice))
        gref = gref_sum / gref_count.clamp(min=1.0)
        correspondence = accepted & (gref_count > 0)
        corr_index = correspondence.nonzero(as_tuple=False).squeeze(-1)
    else:
        anchor_ok = torch.zeros(num_pairs, dtype=torch.bool, device=device)
        outer_ambiguous = torch.zeros_like(anchor_ok)
        outer_far = torch.zeros_like(anchor_ok)
        outer_no_band = torch.ones_like(anchor_ok)
        wide_before = graze = accepted = correspondence = torch.zeros(
            [0], dtype=torch.bool, device=device)
        j = torch.zeros([0], dtype=torch.long, device=device)
        gref = torch.zeros([0], device=device)
        corr_index = torch.zeros([0], dtype=torch.long, device=device)

    pair_loss = torch.zeros(num_pairs, device=device)
    pair_corr_count = torch.zeros(num_pairs, device=device)
    rho = torch.zeros([corr_index.numel()], device=device)
    model_physical_gaps = torch.zeros([0], device=device)
    if corr_index.numel():
        corr_pair = pair[corr_index]
        corr_j = j[corr_index]
        target_winding = k[corr_pair] + corr_j.to(k.dtype)
        radius = (
            target_winding + theta[corr_pair] / (2 * np.pi))
        radius = radius * dr_per_winding.detach()
        target_spiral = torch.stack([
            z[corr_pair], torch.sin(theta[corr_pair]) * radius,
            torch.cos(theta[corr_pair]) * radius,
        ], dim=-1)
        target_scroll = slice_to_spiral_transform.inv(target_spiral)
        adjacent_target = torch.zeros(
            corr_index.numel(), dtype=torch.bool, device=device)
        adjacent_target[1:] = (
            (corr_pair[1:] == corr_pair[:-1])
            & (corr_j[1:] == corr_j[:-1] + 1))
        model_physical_gaps = (
            target_scroll[1:] - target_scroll[:-1]).norm(dim=-1)[
                adjacent_target[1:]]
        rho = (
            (target_scroll - bands['center'][corr_index])
            * bands['direction'][corr_index]).sum(dim=-1) / gref[corr_index]
        residual = F.huber_loss(
            rho, torch.zeros_like(rho), reduction='none',
            delta=float(cfg['dense_spacing_phase_huber_delta']))
        pair_loss.index_add_(0, corr_pair, residual)
        pair_corr_count.index_add_(0, corr_pair, torch.ones_like(residual))
        pair_loss = pair_loss / pair_corr_count.clamp(min=1.0)
        if diagnostics_enabled():
            with torch.no_grad():
                detected_center_spiral = (
                    slice_to_spiral_transform(bands['center'][corr_index])
                    if callable(slice_to_spiral_transform) else target_spiral.detach())
            record_loss_samples(
                'dense_spacing_phase', detected_center_spiral, rho.abs(),
                torch.ones_like(rho, dtype=torch.bool))
    pair_weight = (anchor_ok & (pair_corr_count >= 2)).to(torch.float32).detach()
    loss, mass_stats = _aggregate_nominal_mass(
        pair_loss, pair_weight, cfg['dense_spacing_support_floor_alpha'])

    with torch.no_grad():
        metrics = {
            'dense_spacing_phase_support_mass': float(mass_stats[0].item()),
            'dense_spacing_phase_valid_fraction': float(pair_weight.mean().item()),
            'dense_spacing_phase_rejected_ray_fraction': float(ray_rejected.float().mean().item()),
            'dense_spacing_phase_too_long_fraction': float(ray['too_long'].float().mean().item()),
            'dense_spacing_phase_step_violation_fraction': float(ray['step_violation'].float().mean().item()),
            'dense_spacing_phase_anchor_no_band_fraction': float(no_band.float().mean().item()),
            'dense_spacing_phase_anchor_ambiguity_fraction': float(anchor_ambiguous.float().mean().item()),
            'dense_spacing_phase_anchor_max_distance_fraction': float(anchor_far.float().mean().item()),
            'dense_spacing_phase_outer_no_band_fraction': float(outer_no_band.float().mean().item()),
            'dense_spacing_phase_outer_ambiguity_fraction': float(outer_ambiguous.float().mean().item()),
            'dense_spacing_phase_outer_max_distance_fraction': float(outer_far.float().mean().item()),
            'dense_spacing_phase_no_reference_gap_fraction': float((anchor_ok & (pair_corr_count < 2)).float().mean().item()),
            'dense_spacing_phase_complete_bands_per_ray': float(n_bands / max(1, num_pairs)),
            'dense_spacing_phase_bands_per_modeled_winding': float(n_bands / max(1, int(pair_m.sum().item()))),
            'dense_spacing_phase_merged_fragment_fraction': float(bands['merged_count'] / max(1, n_bands + bands['merged_count'])),
            'dense_spacing_phase_unsupported_sdt_fraction': float((~bands['sample_valid']).float().mean().item()),
            'dense_spacing_phase_sampled_points': float(ray['scroll_poly'].shape[0]),
            'dense_spacing_phase_censor_domain_boundary_fraction': float(
                (phase_end >= float(outer)).float().mean().item()),
            'dense_spacing_phase_mean_enumerated_ordinals': float(
                pair_corr_count.mean().item()),
        }
        if n_bands:
            graze_fraction = float(graze.float().mean().item())
            normal_bad_fraction = float((~bands['normal_valid']).float().mean().item())
            wide_fraction = float(wide_before.float().mean().item())
            metrics.update({
                'dense_spacing_phase_censor_wide_gap_fraction': wide_fraction,
                'dense_spacing_phase_missing_band_wide_gap_fraction': wide_fraction,
                'dense_spacing_phase_censor_graze_fraction': graze_fraction,
                'dense_spacing_phase_censor_unsupported_normal_fraction': normal_bad_fraction,
                'dense_spacing_phase_unsupported_normal_fraction': normal_bad_fraction,
                'dense_spacing_phase_missing_observation_fraction': wide_fraction,
                'dense_spacing_phase_unsupported_observation_fraction': float(
                    ((~bands['normal_valid']) | bands['unsupported_before']).float().mean().item()),
                'dense_spacing_phase_censored_action_fraction': float(
                    (wide_before | graze | ~bands['normal_valid']
                     | bands['unsupported_before']).float().mean().item()),
            })
            projected_width = bands['width'] * bands['normal_dot']
            q = torch.quantile(projected_width, torch.tensor(
                [0.5, 0.9], device=device))
            metrics['dense_spacing_phase_projected_width_p50'] = float(q[0].item())
            metrics['dense_spacing_phase_projected_width_p90'] = float(q[1].item())
            gap_values = gap_length[accepted_gap]
            if gap_values.numel():
                gap_q = torch.quantile(gap_values, torch.tensor(
                    [0.1, 0.5, 0.9], device=device))
                metrics['dense_spacing_phase_physical_gap_p10'] = float(gap_q[0].item())
                metrics['dense_spacing_phase_physical_gap_p50'] = float(gap_q[1].item())
                metrics['dense_spacing_phase_physical_gap_p90'] = float(gap_q[2].item())
        if rho.numel():
            abs_rho = rho.abs()
            metrics['dense_spacing_phase_residual_mean_abs'] = float(abs_rho.mean().item())
            corr_j = j[corr_index]
            for ordinal in torch.unique(corr_j).tolist():
                ordinal_mask = corr_j == ordinal
                metrics[f'dense_spacing_phase_residual_mean_abs_ordinal_{ordinal}'] = float(
                    abs_rho[ordinal_mask].mean().item())
            corr_gref = gref[corr_index]
            boundaries = (0.0, 4.0, 8.0, 16.0, float('inf'))
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                mask = (corr_gref >= lo) & (corr_gref < hi)
                if mask.any():
                    label = f'{lo:g}_{hi:g}' if np.isfinite(hi) else f'{lo:g}_inf'
                    metrics[f'dense_spacing_phase_signed_rho_gref_{label}'] = float(
                        rho[mask].mean().item())
        if model_physical_gaps.numel():
            model_gap_q = torch.quantile(model_physical_gaps, torch.tensor(
                [0.1, 0.5, 0.9], device=device))
            metrics['dense_spacing_phase_model_adjacent_gap_p10'] = float(
                model_gap_q[0].item())
            metrics['dense_spacing_phase_model_adjacent_gap_p50'] = float(
                model_gap_q[1].item())
            metrics['dense_spacing_phase_model_adjacent_gap_p90'] = float(
                model_gap_q[2].item())
        metrics['dense_spacing_phase_span_valid_fraction'] = 0.0
        if n_bands:
            safe_outer = outer_candidate.clamp(max=n_bands - 1)
            outer_supported = bands['normal_valid'][safe_outer] & ~bands['ambiguous'][safe_outer]
            outer_ok = ~outer_no_band & ~outer_ambiguous & ~outer_far & outer_supported
            # A span is valid only when its candidate lies in the accepted prefix.
            outer_reached = torch.zeros(num_pairs, dtype=torch.bool, device=device)
            valid_outer_pair = outer_ok.nonzero(as_tuple=False).squeeze(-1)
            if valid_outer_pair.numel():
                outer_reached[valid_outer_pair] = accepted[outer_candidate[valid_outer_pair]]
            span_valid = anchor_ok & outer_ok & outer_reached
            if span_valid.any():
                p = span_valid.nonzero(as_tuple=False).squeeze(-1)
                span = (
                    bands['ordinal'][outer_candidate[p]]
                    - bands['ordinal'][anchor[p]])
                metrics['dense_spacing_phase_span_minus_m_mean'] = float(
                    (span - pair_m[p]).to(torch.float32).mean().item())
                metrics['dense_spacing_phase_span_valid_fraction'] = float(
                    span_valid.float().mean().item())
        # Keep the shipped soft crossing count as a detached rollout
        # diagnostic on the unextended modeled interval.
        indicator = _inside_indicator(
            bands['field_value'], sdt_volume,
            float(cfg['dense_spacing_count_temperature_wv']))
        edge_mid_phase = (ray['sample_phase'][:-1] + ray['sample_phase'][1:]) * 0.5
        base_edge = (
            ray['has_successor'][:-1]
            & bands['sample_valid'][:-1] & bands['sample_valid'][1:]
            & (edge_mid_phase >= k[ray['pair_id'][:-1]])
            & (edge_mid_phase <= (k + pair_m)[ray['pair_id'][:-1]]))
        soft_diff = (indicator[1:] - indicator[:-1]).abs() * base_edge.to(indicator.dtype)
        soft_count = 0.5 * torch.zeros(num_pairs, device=device).index_add_(
            0, ray['pair_id'][:-1], soft_diff)
        non_rejected = ~ray_rejected
        if non_rejected.any():
            metrics['dense_spacing_phase_soft_count_mean'] = float(
                soft_count[non_rejected].mean().item())
            metrics['dense_spacing_phase_soft_count_residual_mean'] = float(
                (soft_count[non_rejected] - pair_m[non_rejected]).abs().mean().item())
        metrics.update({
            f'dense_spacing_phase_sdt_{name}': value
            for name, value in sdt_sample_fractions(
                bands['field_value'], bands['sample_valid'], sdt_volume).items()
        })
        metrics['dense_spacing_phase_wall_seconds'] = time.perf_counter() - started
    return loss, metrics


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
    slice_to_spiral_transform, dr_per_winding, count_volume, sdt_volume,
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
            slice_to_spiral_transform, dr_per_winding, count_volume, sdt_volume,
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
