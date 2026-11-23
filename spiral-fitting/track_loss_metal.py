"""Fused Metal kernels for the track radius loss on Apple GPUs (MPS).

On MPS the eager track radius loss (tracks.iter_track_losses) is a chain of
about fifty small ops forward and as many backward: theta and shifted radii,
the ragged per-row unwrap (a global cumsum), the crossing-partner alignment,
the grouped mean target (an inf-padded [groups, width, points] tensor), the
hinge and the grouped p-norm. Each op costs MPS a host dispatch of tens of
microseconds while the GPU work per op is a few microseconds, and every new
point count compiles another MPSGraph, so the eager loss is host-bound and its
graph caches grow host memory every step.

Here one thread owns one group (a primary track and its crossing partners,
which are all the rows the group's target and alignment read), so the whole
forward is one dispatch and the whole backward another, with no atomics and a
fixed, deterministic summation order. Arithmetic follows the eager path term
by term, with two differences in evaluation order that change rounding only:
- The unwrap adjustment is a running sum along each row, where the eager path
  takes one cumsum over the whole batch and subtracts the row start's value.
  Both sum the same multiples of dr; the per-row sum is the more exact.
- Group sums (target mean, p-norm numerator) run sequentially in row and
  point order instead of torch's reduction trees.

Gradients match the eager autograd's subgradient conventions: abs and relu
give zero at zero, clamp(min=0) passes the gradient where its input >= 0, and
the x component replaced by 1e-10 near the axis gets no gradient. The unwrap
adjustments, alignments and the hinge margin use detached values, as in the
eager path.

Compiled once per process with torch.mps.compile_shader (PyTorch's safe math
mode). Set FIT_SPIRAL_METAL=0, or FIT_SPIRAL_METAL_TRACKS=0 for this loss
only, to use the eager path.
"""
import os

import numpy as np
import torch

_GROUP = 64
_LIB = None

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

constant float TWO_PI_F = 6.283185307179586f;
constant float PI_F = 3.141592653589793f;

struct Polar {
    float theta;
    float shifted;
    float y;
    float x;       // x after the near-axis replacement
    bool x_live;   // false where x was replaced, so x gets no gradient
    bool clamp_live;
};

// sample_spiral.get_theta_and_radii for one point: theta in [0, 2pi) with
// torch.remainder semantics, radius, and the shifted radius clamped at 0
inline Polar polar(device const float* spiral, long i, float dr) {
    Polar p;
    p.y = spiral[3 * i + 1];
    float x = spiral[3 * i + 2];
    p.x_live = !(fabs(x) < 1.0e-10f);
    p.x = p.x_live ? x : 1.0e-10f;
    float a = precise::atan2(p.y, p.x);
    p.theta = a - TWO_PI_F * floor(a / TWO_PI_F);
    float radius = precise::sqrt(p.y * p.y + p.x * p.x);
    float raw = radius - p.theta / TWO_PI_F * dr;
    p.clamp_live = raw >= 0.0f;
    p.shifted = max(raw, 0.0f);
    return p;
}

inline float crossing_step(float theta_prev, float theta, float dr_detached) {
    float d = theta - theta_prev;
    return (float(d > PI_F) - float(d < -PI_F)) * dr_detached;
}

// Row r of group g: the primary row is g itself, partner rows follow in
// [n_groups + partner_offsets[g], n_groups + partner_offsets[g + 1])
inline long group_row(long g, long j, long n_groups,
                      device const long* partner_offsets) {
    return j == 0 ? g : n_groups + partner_offsets[g] + (j - 1);
}

// Unwrapped, aligned shifted radii for one row; returns nothing, writes
// unwrapped[] and adjust[] (the detached crossing adjustment plus alignment)
inline void unwrap_row(device const float* spiral, device float* unwrapped,
                       device float* adjust, device float* theta_out,
                       long begin, long end, float dr, float dr_detached) {
    float running = 0.0f;
    float theta_prev = 0.0f;
    for (long i = begin; i < end; ++i) {
        Polar p = polar(spiral, i, dr);
        if (i > begin) {
            running += crossing_step(theta_prev, p.theta, dr_detached);
        }
        theta_prev = p.theta;
        unwrapped[i] = p.shifted + running;
        adjust[i] = running;
        theta_out[i] = p.theta;
    }
}

kernel void track_radius_fwd(
        device const float* spiral [[buffer(0)]],
        device const long* row_starts [[buffer(1)]],
        device const long* partner_offsets [[buffer(2)]],
        device const long* primary_cross [[buffer(3)]],
        device const long* partner_cross [[buffer(4)]],
        device const long* target_flat [[buffer(5)]],
        device const float* dr_buf [[buffer(6)]],
        device float* unwrapped [[buffer(7)]],
        device float* adjust [[buffer(8)]],
        device float* theta_out [[buffer(9)]],
        device float* targets [[buffer(10)]],
        device float* group_mean [[buffer(11)]],
        device float* group_loss [[buffer(12)]],
        constant long& n_groups [[buffer(13)]],
        constant long& target_points [[buffer(14)]],
        constant float& margin [[buffer(15)]],
        constant float& within_p [[buffer(16)]],
        uint tid [[thread_position_in_grid]]) {
    long g = long(tid);
    if (g >= n_groups) {
        return;
    }
    float dr = dr_buf[0];
    long rows = 1 + partner_offsets[g + 1] - partner_offsets[g];

    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        unwrap_row(spiral, unwrapped, adjust, theta_out,
                   row_starts[r], row_starts[r + 1], dr, dr);
    }
    // each partner moves into its primary's unwrap frame by the difference at
    // their shared voxel (detached, read after every row is unwrapped)
    for (long j = 1; j < rows; ++j) {
        long q = partner_offsets[g] + (j - 1);
        long r = n_groups + q;
        float shift = unwrapped[primary_cross[q]] - unwrapped[partner_cross[q]];
        for (long i = row_starts[r]; i < row_starts[r + 1]; ++i) {
            unwrapped[i] += shift;
            adjust[i] += shift;
        }
    }

    // mean of the finite target values over every row of the group
    float total = 0.0f;
    float count = 0.0f;
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        for (long k = 0; k < target_points; ++k) {
            float v = unwrapped[target_flat[r * target_points + k]];
            if (isfinite(v)) {
                total += v;
                count += 1.0f;
            }
        }
    }
    float target = total / max(count, 1.0f);
    targets[g] = target;

    float hinge_margin = dr * margin;
    float sum_q = 0.0f;
    float points = 0.0f;
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        for (long i = row_starts[r]; i < row_starts[r + 1]; ++i) {
            float h = max(fabs(unwrapped[i] - target) - hinge_margin, 0.0f);
            sum_q += within_p == 1.0f ? h : powr(h + 1.0e-5f, within_p);
            points += 1.0f;
        }
    }
    float mean_q = sum_q / max(points, 1.0f);
    group_mean[g] = mean_q;
    group_loss[g] = within_p == 1.0f ? mean_q : powr(mean_q, 1.0f / within_p);
}

kernel void track_radius_bwd(
        device const float* spiral [[buffer(0)]],
        device const long* row_starts [[buffer(1)]],
        device const long* partner_offsets [[buffer(2)]],
        device const long* target_flat [[buffer(3)]],
        device const float* dr_buf [[buffer(4)]],
        device const float* unwrapped [[buffer(5)]],
        device const float* targets [[buffer(6)]],
        device const float* group_mean [[buffer(7)]],
        device const float* grad_group_loss [[buffer(8)]],
        device float* grad_unwrapped [[buffer(9)]],
        device float* grad_spiral [[buffer(10)]],
        device float* grad_dr_partial [[buffer(11)]],
        constant long& n_groups [[buffer(12)]],
        constant long& target_points [[buffer(13)]],
        constant float& margin [[buffer(14)]],
        constant float& within_p [[buffer(15)]],
        uint tid [[thread_position_in_grid]]) {
    long g = long(tid);
    if (g >= n_groups) {
        return;
    }
    float dr = dr_buf[0];
    long rows = 1 + partner_offsets[g + 1] - partner_offsets[g];
    float target = targets[g];
    float hinge_margin = dr * margin;

    long points = 0;
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        points += row_starts[r + 1] - row_starts[r];
    }
    float inv_points = 1.0f / max(float(points), 1.0f);
    // d group_loss / d mean_q, then per point d mean_q / d q = 1 / points
    float gl = grad_group_loss[g];
    float d_mean = within_p == 1.0f
        ? gl
        : gl * (1.0f / within_p) * powr(group_mean[g], 1.0f / within_p - 1.0f);

    // direct term, and the target's share: d loss / d target = -sum of them
    float grad_target = 0.0f;
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        for (long i = row_starts[r]; i < row_starts[r + 1]; ++i) {
            float diff = unwrapped[i] - target;
            float dev = fabs(diff);
            float pre = dev - hinge_margin;
            float gi = 0.0f;
            if (pre > 0.0f && diff != 0.0f) {
                float dq = within_p == 1.0f
                    ? 1.0f
                    : within_p * powr(pre + 1.0e-5f, within_p - 1.0f);
                gi = d_mean * inv_points * dq * (diff > 0.0f ? 1.0f : -1.0f);
            }
            grad_unwrapped[i] = gi;
            grad_target -= gi;
        }
    }
    float count = 0.0f;
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        for (long k = 0; k < target_points; ++k) {
            if (isfinite(unwrapped[target_flat[r * target_points + k]])) {
                count += 1.0f;
            }
        }
    }
    float share = grad_target / max(count, 1.0f);
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        for (long k = 0; k < target_points; ++k) {
            long i = target_flat[r * target_points + k];
            if (isfinite(unwrapped[i])) {
                grad_unwrapped[i] += share;
            }
        }
    }

    // unwrapped = shifted + detached adjustment, so d/d shifted = d/d unwrapped
    float grad_dr = 0.0f;
    for (long j = 0; j < rows; ++j) {
        long r = group_row(g, j, n_groups, partner_offsets);
        for (long i = row_starts[r]; i < row_starts[r + 1]; ++i) {
            Polar p = polar(spiral, i, dr);
            float gs = p.clamp_live ? grad_unwrapped[i] : 0.0f;
            float r2 = p.y * p.y + p.x * p.x;
            float radius = precise::sqrt(r2);
            float g_theta = -gs * dr / TWO_PI_F;
            grad_dr += -gs * p.theta / TWO_PI_F;
            float gy = gs * p.y / radius + g_theta * p.x / r2;
            float gx = gs * p.x / radius - g_theta * p.y / r2;
            grad_spiral[3 * i + 0] = 0.0f;
            grad_spiral[3 * i + 1] = gy;
            grad_spiral[3 * i + 2] = p.x_live ? gx : 0.0f;
        }
    }
    grad_dr_partial[g] = grad_dr;
}

// tracks._draw_track_sample's per-point and per-target indices for one row,
// and the point gather: row_id, the resampled points, the target positions
// round_half_even(fraction * (length - 1)) and their source indices
kernel void track_sample_rows(
        device const long* row_starts [[buffer(0)]],
        device const long* tracks [[buffer(1)]],
        device const long* offsets [[buffer(2)]],
        device const float* flat_zyx [[buffer(3)]],
        device const long* flat_source_local [[buffer(4)]],
        device const float* fractions [[buffer(5)]],
        device long* row_id [[buffer(6)]],
        device float* sampled [[buffer(7)]],
        device long* target_flat [[buffer(8)]],
        device long* target_source [[buffer(9)]],
        constant long& rows [[buffer(10)]],
        constant long& target_points [[buffer(11)]],
        uint tid [[thread_position_in_grid]]) {
    long r = long(tid);
    if (r >= rows) {
        return;
    }
    long begin = row_starts[r];
    long length = row_starts[r + 1] - begin;
    long base = offsets[tracks[r]];
    for (long j = 0; j < length; ++j) {
        long i = begin + j;
        long source = base + j;
        row_id[i] = r;
        sampled[3 * i + 0] = flat_zyx[3 * source + 0];
        sampled[3 * i + 1] = flat_zyx[3 * source + 1];
        sampled[3 * i + 2] = flat_zyx[3 * source + 2];
    }
    float span = float(length - 1);
    for (long k = 0; k < target_points; ++k) {
        long local = target_points == 1 ? 0 : long(rint(fractions[k] * span));
        target_flat[r * target_points + k] = begin + local;
        target_source[r * target_points + k] = flat_source_local[base + local];
    }
}
'''


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SOURCE)
    return _LIB


def track_radius_metal_available(spiral, dr_per_winding, cfg):
    """True when the fused kernels can replace the eager radius loss here."""
    if not _enabled():
        return False
    if cfg['track_radius_target'] != 'mean':
        return False
    return (spiral.device.type == 'mps' and spiral.dtype == torch.float32
            and dr_per_winding.dtype == torch.float32)


class _TrackRadiusMetal(torch.autograd.Function):
    # Differentiable in the sampled spiral coordinates and dr_per_winding.
    # The other outputs are detached intermediates the DT loss reuses.

    @staticmethod
    def forward(ctx, spiral, dr, row_starts, partner_offsets, primary_cross,
                partner_cross, target_flat, margin, within_p):
        n_groups = partner_offsets.shape[0] - 1
        target_points = target_flat.shape[1]
        unwrapped = torch.empty(spiral.shape[0], device=spiral.device,
                                dtype=spiral.dtype)
        adjust = torch.empty_like(unwrapped)
        theta = torch.empty_like(unwrapped)
        targets = torch.empty(n_groups, device=spiral.device,
                              dtype=spiral.dtype)
        group_mean = torch.empty_like(targets)
        group_loss = torch.empty_like(targets)
        if n_groups > 0:
            _lib().track_radius_fwd(
                spiral, row_starts, partner_offsets, primary_cross,
                partner_cross, target_flat, dr, unwrapped, adjust, theta,
                targets, group_mean, group_loss, int(n_groups),
                int(target_points), float(margin), float(within_p),
                threads=n_groups, group_size=_GROUP)
        ctx.save_for_backward(spiral, dr, row_starts, partner_offsets,
                              target_flat, unwrapped, targets, group_mean)
        ctx.consts = (int(n_groups), int(target_points), float(margin),
                      float(within_p))
        ctx.mark_non_differentiable(unwrapped, adjust, theta, targets)
        return group_loss, unwrapped, adjust, theta, targets

    @staticmethod
    def backward(ctx, grad_group_loss, _g_unwrapped, _g_adjust, _g_theta,
                 _g_targets):
        (spiral, dr, row_starts, partner_offsets, target_flat, unwrapped,
         targets, group_mean) = ctx.saved_tensors
        n_groups = ctx.consts[0]
        grad_spiral = torch.zeros_like(spiral)
        grad_unwrapped = torch.zeros_like(unwrapped)
        grad_dr_partial = torch.empty(n_groups, device=spiral.device,
                                      dtype=spiral.dtype)
        if grad_group_loss is not None and n_groups > 0:
            _lib().track_radius_bwd(
                spiral, row_starts, partner_offsets, target_flat, dr,
                unwrapped, targets, group_mean,
                grad_group_loss.contiguous(), grad_unwrapped, grad_spiral,
                grad_dr_partial, *ctx.consts, threads=n_groups,
                group_size=_GROUP)
            grad_dr = grad_dr_partial.sum().reshape(dr.shape)
        else:
            grad_dr = torch.zeros_like(dr)
        return (grad_spiral, grad_dr, None, None, None, None, None, None,
                None)


def track_radius_loss_metal(sampled_spiral, dr_per_winding, sample, cfg):
    """Fused track radius loss on MPS.

    Returns ``(loss, unwrapped, adjustments, theta, group_targets)``: the
    scalar loss (mean of the per-group p-norms, as the eager
    tracks._grouped_same_radius_loss), plus detached per-point unwrapped and
    aligned shifted radii, their crossing adjustments (alignment included),
    theta, and the per-group radius targets the DT loss reuses.
    """
    num_groups = int(sample['num_groups'])
    partner_offsets = sample.get('partner_offsets')
    if partner_offsets is None:
        # partner rows follow in group order (tracks._draw_track_sample)
        partner_group = sample['group_id'][num_groups:]
        partner_offsets = torch.searchsorted(
            partner_group.contiguous(),
            torch.arange(num_groups + 1, device=partner_group.device))
    dr = dr_per_winding.reshape(1)
    primary_cross = sample['primary_cross_flat']
    partner_cross = sample['partner_cross_flat']
    if primary_cross.numel() == 0:
        # an empty tensor binds no Metal buffer; with no partner rows the
        # kernel never reads these
        primary_cross = partner_cross = torch.zeros(
            1, dtype=torch.int64, device=sampled_spiral.device)
    group_loss, unwrapped, adjust, theta, targets = _TrackRadiusMetal.apply(
        sampled_spiral.contiguous(), dr.contiguous(),
        sample['row_starts'].contiguous(), partner_offsets.contiguous(),
        primary_cross.contiguous(), partner_cross.contiguous(),
        sample['target_flat_idx'].contiguous(),
        cfg['track_radius_loss_margin'], cfg['track_radius_within_norm_p'])
    return group_loss.mean(), unwrapped, adjust, theta, targets



def _enabled():
    return (os.environ.get('FIT_SPIRAL_METAL', '1') != '0'
            and os.environ.get('FIT_SPIRAL_METAL_TRACKS', '1') != '0'
            and torch.backends.mps.is_available())


def host_planned_sample_available(resampled, device):
    """True when tracks._host_planned_track_sample can build this batch: MPS,
    with the resampled points resident on the device and their lengths on
    the host."""
    if device.type != 'mps' or not _enabled():
        return False
    flat_zyx = resampled.get('flat_zyx')
    source = resampled.get('flat_source_local')
    return (flat_zyx is not None and source is not None
            and 'lengths_cpu' in resampled
            and flat_zyx.dtype == torch.float32 and flat_zyx.is_contiguous()
            and resampled['offsets'].dtype == torch.int64)


# host buffers whose non-blocking uploads may still be in flight: MPS reads a
# pageable source in place when its blit runs (no staging copy), so each one
# must outlive its blit. The fit synchronizes after every step, and the next
# batch's seed read waits for the queue, so two batches back is long done
_INFLIGHT = 2


def upload_host_arrays(prepared_tracks, arrays, device):
    """Send a dict of 1-D int64 numpy arrays to ``device`` in one non-blocking
    copy; returns device views, one per key."""
    sizes = {name: int(np.asarray(a).size) for name, a in arrays.items()}
    packed = np.empty(max(sum(sizes.values()), 1), dtype=np.int64)
    views = {}
    offset = 0
    for name, a in arrays.items():
        n = sizes[name]
        packed[offset:offset + n] = np.asarray(a, dtype=np.int64).reshape(-1)
        views[name] = (offset, n)
        offset += n
    host = torch.from_numpy(packed)
    on_device = host.to(device, non_blocking=True)
    inflight = prepared_tracks.setdefault('_inflight_host_uploads', [])
    inflight.append(host)
    del inflight[:-_INFLIGHT]
    return {name: on_device[o:o + n] for name, (o, n) in views.items()}


def build_track_sample(row_starts, track_idx, resampled, target_points,
                       num_points, prepared_tracks):
    """One dispatch for the batch's row_id, gathered points and target
    indices (see track_sample_rows)."""
    device = row_starts.device
    rows = int(track_idx.shape[0])
    cache = prepared_tracks.setdefault('_track_fractions', {})
    fractions = cache.get(target_points)
    if fractions is None:
        # the eager path's own linspace, so the rounded positions agree
        fractions = torch.linspace(0.0, 1.0, target_points, device=device)
        cache[target_points] = fractions
    source = resampled['flat_source_local']
    source_i64 = source if source.dtype == torch.int64 else resampled.setdefault(
        '_flat_source_local_i64', source.to(torch.int64))
    row_id = torch.empty(num_points, dtype=torch.int64, device=device)
    sampled = torch.empty(num_points, 3, dtype=torch.float32, device=device)
    target_flat = torch.empty(rows, target_points, dtype=torch.int64,
                              device=device)
    target_source = torch.empty_like(target_flat)
    _lib().track_sample_rows(
        row_starts, track_idx, resampled['offsets'], resampled['flat_zyx'],
        source_i64, fractions, row_id, sampled, target_flat, target_source,
        int(rows), int(target_points), threads=rows, group_size=_GROUP)
    if source.dtype != torch.int64:
        target_source = target_source.to(source.dtype)
    return {
        'row_id': row_id,
        'sampled_scroll': sampled,
        'target_flat_idx': target_flat,
        'target_source_idx': target_source,
    }
