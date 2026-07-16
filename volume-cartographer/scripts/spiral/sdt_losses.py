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

import numpy as np
import torch
import torch.nn.functional as F

from ddp_helpers import get_world_size, is_distributed
from loss_maps import record_loss_samples


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


def _sample_windings_by_circumference(lowest, highest, num_samples, device):
    """Integer windings in [lowest, highest] with probability proportional to
    winding circumference (weight k + 0.5). ``highest`` may be a per-sample
    tensor. Uses the analytic inverse CDF: cum-mass to k is ((k+1)^2 - a^2)/2."""
    a = float(lowest)
    b = torch.as_tensor(highest, device=device, dtype=torch.float32)
    total_mass = ((b + 1.0) ** 2 - a * a) / 2.0
    u = torch.rand(num_samples, device=device)
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


def compute_pair_counts(
    slice_to_spiral_transform, dr_per_winding, count_volume, sdt_volume,
    winding_idx, pair_m, theta, z, cfg,
):
    """Soft crossing counts along the mapped inter-winding polylines.

    Shared by the training loss and the per-pair aggregation diagnostic. Each
    pair (k, k + m) at (z, theta) becomes a radial segment in spiral space,
    inverse-mapped as a full polyline (never the endpoint chord) with a
    per-pair step count targeting ~1 working voxel between mapped samples.
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
    target_step = float(cfg['dense_spacing_target_step_wv'])
    max_step = float(cfg['dense_spacing_max_step_wv'])
    max_steps = int(cfg['dense_spacing_max_steps'])
    oversample = float(cfg['dense_spacing_step_oversample'])

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    m_f = pair_m.to(torch.float32)
    r0 = (winding_idx + theta / (2 * np.pi)) * dr
    r1 = r0 + m_f * dr

    # Detached endpoint mapping estimates each pair's mapped length, from which
    # the per-pair step count is derived (chord underestimates a curved path,
    # hence the conservative oversample; the mapped result is still validated
    # against the max-step bound below).
    def spiral_points(radii):
        return torch.stack([z, sin_t * radii, cos_t * radii], dim=-1)

    with torch.no_grad():
        endpoint_scroll = slice_to_spiral_transform.inv(
            torch.cat([spiral_points(r0), spiral_points(r1)], dim=0))
        chord = (endpoint_scroll[num_pairs:] - endpoint_scroll[:num_pairs]).norm(dim=-1)
    steps_needed = torch.ceil(chord * oversample / target_step).long().clamp(min=2)
    too_long = steps_needed > max_steps
    # Too-long pairs stay in the batch (their invalidity is reported) but are
    # mapped with a minimal polyline so they cost almost nothing.
    n = torch.where(too_long, torch.full_like(steps_needed, 2), steps_needed)

    samples_per_pair = n + 1
    offsets = F.pad(samples_per_pair.cumsum(0), (1, 0))
    total = int(offsets[-1].item())
    pair_id = torch.repeat_interleave(torch.arange(num_pairs, device=device), samples_per_pair)
    position = torch.arange(total, device=device) - offsets[pair_id]
    t = position.to(torch.float32) / n[pair_id].to(torch.float32)
    radii = r0[pair_id] + t * (r1 - r0)[pair_id]
    spiral_poly = torch.stack(
        [z[pair_id], sin_t[pair_id] * radii, cos_t[pair_id] * radii], dim=-1)
    scroll_poly = slice_to_spiral_transform.inv(spiral_poly)

    # Validate the actual mapped step sizes, not just length / step count.
    has_successor = position < n[pair_id]
    with torch.no_grad():
        step_lengths = (scroll_poly[1:] - scroll_poly[:-1]).norm(dim=-1)
        step_lengths = torch.where(has_successor[:-1], step_lengths,
                                   torch.zeros_like(step_lengths))
        max_step_per_pair = torch.zeros(num_pairs, device=device).scatter_reduce(
            0, pair_id[:-1], step_lengths, reduce='amax', include_self=True)
    step_violation = (max_step_per_pair > max_step) & ~too_long

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
    # Clamp m to the domain span so every sampled pair keeps both windings
    # inside the fitted domain (k >= inner and k + m <= outer).
    max_m = min(max(1, int(cfg['dense_spacing_pair_max_m'])), domain_span)

    pair_m = torch.randint(1, max_m + 1, [num_pairs], device=device)
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
