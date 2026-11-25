"""The fiber loss: one term per connected component (CC) of the winding solve.

The solve (fiber_solve_inputs.WindingSolve) gives every kept fiber point a
seam-relative continuous winding ``w_s``. Only its integer is trusted, read in
the model's own angle at the point (the "sheet" rule, branch_target):

    n(p) = round(w_s(p) - theta_model(p) / 2pi)          # detached
    rho(p) = w_s(p) - theta_model(p) / 2pi - n(p)         # |rho| > 0.4 rejected
    s(p) = (shifted(p) - offset(p) * stretch(p)) / dr - n(p)

Spiral's branch label q = shifted / dr steps at theta = 0, where n steps too,
so s is constant over a correctly placed CC: the CC's offset between Spiral's
windings and the solve's sheets. ``offset`` is the optional outward shift of
vertical fibers (the sheet's back face), converted to spiral radius by the
transform's normal stretch (sample_spiral.get_radial_normal_stretch).

Per CC, on its accepted sampled points (Huber on the error, quadratic below
fiber_huber_delta_turns, linear above):

- the largest CC: |s - c0| from step 0, c0 an integer gauge;
- every other CC: |s - median(s)| over the step's samples (zero when fewer
  than MIN_MEDIAN_SAMPLES are accepted), then, from loss_start_fiber_dt,
  |s - T_CC| with T_CC = round(median(s)) over all its accepted points,
  refreshed every fiber_dt_target_update_interval steps (stateless: computed
  again at start or resume).

The CCs are the solve's, or those left after cutting the crossings the
solve violates by more than fiber_constraint_cut_turns
(fiber_constraint_cut), which takes a mislabelled group out of the largest.

The gauge: per fiber, its sheet labels lie in [floor(w_min - c),
floor(w_max - c)] (whole fiber). As a backstop after the cut, largest-CC
fibers whose minimum lies more than fiber_outlier_margin_turns below the
fiber_largest_cc_min_percentile-th percentile of the minima (or whose
maximum lies that far above the matching high percentile of the maxima) are
dropped everywhere: a fiber left below the gauge would ask for a winding
below the innermost. ``n_low`` is the smallest minimum of the kept ones, and
c0 = fiber_largest_cc_min_winding - n_low, so the innermost kept sheet is
that winding.
"""
import math

import numpy as np
import torch

from sample_spiral import get_radial_normal_stretch, get_theta_and_radii
from loss_maps import diagnostics_enabled, record_loss_samples
from fiber_solve_inputs import FAMILY_V
from spiral_helpers import _huber_abs
from tracks import _transform_shape_bucketed

# Ported from the frozen winding_convention.py (PHerc0332 repro tree, tag
# pherc0332-s1_f327-2026-11-09).
RHO_MAX = 0.4
# Accepted samples a non-largest CC needs in a step for its median target.
MIN_MEDIAN_SAMPLES = 2
# Points (and consecutive-point pairs) the health metrics draw.
METRIC_SAMPLES = 8192
# Seed of the metrics' own generator (offset by the iteration), so measuring
# never moves the training RNG streams.
METRIC_SEED = 0x5F1BE2
# Points transformed per chunk by the DT-target refresh.
CHUNK = 65536


def branch_target(w_cont, phi):
    """(t, rho) for continuous gauged windings and wrapped angles; numpy or torch, any matching shapes."""
    if isinstance(w_cont, torch.Tensor) or isinstance(phi, torch.Tensor):
        w_cont = torch.as_tensor(w_cont)
        phi = torch.as_tensor(phi, device=w_cont.device, dtype=w_cont.dtype)
        x = w_cont - phi / (2 * math.pi)
        t = torch.round(x)
        return t, x - t
    x = np.asarray(w_cont, dtype=np.float64) - np.asarray(phi, dtype=np.float64) / (2 * np.pi)
    t = np.round(x)
    return t, x - t


def accepted(rho):
    """Phase acceptance: |rho| <= RHO_MAX (inclusive)."""
    if isinstance(rho, torch.Tensor):
        return rho.abs() <= RHO_MAX
    return np.abs(rho) <= RHO_MAX


def vertical_radial_offset(config):
    """Outward target offset (scroll voxels) of vertical fibers; 0 when off."""
    if not config['fiber_vertical_radial_offset_enabled']:
        return 0.0
    return float(config['fiber_vertical_radial_offset_voxels'])


def largest_cc_gauge(solve, config):
    """The largest CC's outlier backstop and gauge, from whole-fiber label extremes.

    Returns a dict: ``dropped`` bool [F] (largest-CC fibers dropped), ``n_low``,
    ``c0``, the percentiles, the kept label range and how many kept fibers
    have their minimum at n_low and at n_low + 1.
    """
    low = np.floor(solve.w_min - solve.c).astype(np.int64)
    high = np.floor(solve.w_max - solve.c).astype(np.int64)
    largest = solve.fiber_cc == solve.largest_cc
    percentile = float(config['fiber_largest_cc_min_percentile'])
    margin = float(config['fiber_outlier_margin_turns'])
    p_low = float(np.percentile(low[largest], percentile))
    p_high = float(np.percentile(high[largest], 100.0 - percentile))
    dropped = largest & ((low < p_low - margin) | (high > p_high + margin))
    kept = largest & ~dropped
    n_low = int(low[kept].min())
    return {
        'dropped': dropped, 'percentile': percentile, 'margin': margin,
        'p_low': p_low, 'p_high': p_high,
        'n_low': n_low, 'c0': int(config['fiber_largest_cc_min_winding']) - n_low,
        'kept_range': (n_low, int(high[kept].max())),
        'at_n_low': int((low[kept] == n_low).sum()),
        'at_n_low_plus_1': int((low[kept] == n_low + 1).sum()),
    }


def _group_median(group, values, mask, num_groups):
    """Per group, the (lower) median of ``values`` where ``mask``, and the
    masked count. Returns (median [num_groups], count [num_groups]); the
    median is meaningless where the count is 0."""
    keyed = torch.where(mask, values, torch.full_like(values, float('inf')))
    order = torch.argsort(keyed)
    order = order[torch.argsort(group[order], stable=True)]
    totals = torch.bincount(group, minlength=num_groups)
    counts = torch.bincount(group[mask], minlength=num_groups)
    starts = torch.cumsum(totals, 0) - totals
    middle = (starts + torch.div((counts - 1).clamp(min=0), 2, rounding_mode='floor')).clamp(
        max=max(int(values.numel()) - 1, 0))
    sorted_values = keyed[order]
    if not sorted_values.numel():
        return torch.zeros(num_groups, dtype=values.dtype, device=values.device), counts
    return sorted_values[middle], counts


def whole_winding_targets(group, s, ok, num_groups):
    """Per group, round(median s) over its accepted points (``ok``), NaN for
    a group with none; and the accepted count. The DT targets T_CC and the
    satisfaction pass's targets are this."""
    median, count = _group_median(group, s, ok, num_groups)
    nan = torch.full_like(median, float('nan'))
    return torch.where(count > 0, torch.round(median), nan), count


class _Table:
    """Sampling tables over items (points or consecutive-point pairs) grouped
    by fiber. A fiber with k items weighs k ** exponent (fibers without items
    never draw): a CC by its fibers' total weight, a fiber within it by its
    weight, then an item uniformly within the fiber's range. Exponent 0
    weighs every fiber alike, 1 every item alike."""

    def __init__(self, item_start, item_count, fiber_cc, num_cc, device):
        fibers = np.flatnonzero(item_count > 0)
        order = fibers[np.argsort(fiber_cc[fibers], kind='stable')]
        cc_count = np.bincount(fiber_cc[fibers], minlength=num_cc)
        self.size = int(item_count.sum())
        as_tensor = lambda values: torch.as_tensor(values, dtype=torch.int64, device=device)  # noqa: E731
        self.fiber_order = as_tensor(order)
        self.cc_start = as_tensor(np.cumsum(cc_count) - cc_count)
        self.cc_count = as_tensor(cc_count)
        self.cc_weight = torch.as_tensor(cc_count, dtype=torch.float32, device=device)
        self.item_start = as_tensor(item_start)
        self.item_count = as_tensor(item_count)
        # Per fiber in CC order, for the weighted tables (_weighted).
        self._order_counts = item_count[order].astype(np.float64)
        self._cc_end = np.cumsum(cc_count)
        self._cc_begin = self._cc_end - cc_count
        self._weighted_exponent = None

    def _weighted(self, exponent):
        """(cumulative weight at each CC's end, cumulative fiber weight, cc
        base, cc total) for ``exponent``, built on first use and kept until
        it changes. The tables are float64 (CPU on MPS, which has none): one
        point among millions weighs less than a float32 can resolve."""
        if self._weighted_exponent != exponent:
            weight = self._order_counts ** exponent
            cumulative = np.cumsum(weight)
            edges = np.concatenate([[0.0], cumulative])
            end, base = edges[self._cc_end], edges[self._cc_begin]
            device = self.cc_count.device
            table = torch.device('cpu') if device.type == 'mps' else device
            as_double = lambda values: torch.as_tensor(values, dtype=torch.float64, device=table)  # noqa: E731
            self._weighted_tables = (
                as_double(end), as_double(cumulative), as_double(base), as_double(end - base))
            # The last CC with weight: where rounding up to the total lands.
            self._last_cc = int(np.searchsorted(end, end[-1])) if len(end) else 0
            self._weighted_exponent = exponent
        return self._weighted_tables

    def _fraction(self, count, like, generator):
        """``count`` uniforms in [0, 1) as ``like``'s dtype and device, drawn
        on the sampler's device. A float32 uniform has 24 bits, too coarse
        for one point among millions: a second draw refines it below
        2 ** -24."""
        u = torch.rand((2, count), device=self.cc_count.device, generator=generator).to(like)
        return u[0] + u[1] * 2.0 ** -24

    def sample(self, draws, per_draw, generator=None, exponent=0.0):
        """``(cc, item)`` int64 [draws * per_draw]: ``draws`` CCs, then
        ``per_draw`` (fiber, item) picks in each, fibers weighted by their
        item count ** ``exponent``. Exponent 0 is the uniform-fiber draw."""
        exponent = float(exponent)
        weighted = self._weighted(exponent) if exponent != 0.0 else None
        if weighted is None:
            cc = torch.multinomial(self.cc_weight, draws, replacement=True, generator=generator)
        else:
            # Inverse CDF over the CCs (a multinomial on float32 weights
            # cannot resolve a few-point CC beside one of millions); a CC
            # without weight has an empty interval and is never drawn.
            cc_end = weighted[0]
            target = self._fraction(draws, cc_end, generator) * cc_end[-1]
            cc = torch.searchsorted(cc_end, target, right=True)
            cc = cc.clamp(max=self._last_cc).to(self.cc_count.device)
        cc = cc[:, None].expand(draws, per_draw).reshape(-1)
        u = torch.rand((2, cc.numel()), device=cc.device, generator=generator)
        count = self.cc_count[cc]
        if weighted is None:
            rank = torch.minimum((u[0] * count).long(), count - 1)
            fiber = self.fiber_order[self.cc_start[cc] + rank]
        else:
            # Inverse CDF within the CC, clamped into its range against
            # rounding at either end.
            _, cumulative, base, total = weighted
            cc_table = cc.to(cumulative.device)
            fraction = self._fraction(cc.numel(), cumulative, generator)
            target = base[cc_table] + fraction * total[cc_table]
            position = torch.searchsorted(cumulative, target, right=True).to(cc.device)
            start = self.cc_start[cc]
            position = torch.minimum(torch.maximum(position, start), start + count - 1)
            fiber = self.fiber_order[position]
        count = self.item_count[fiber]
        item = self.item_start[fiber] + torch.minimum((u[1] * count).long(), count - 1)
        return cc, item


class FiberCCLoss:
    """The fiber CC loss over one solve's kept points (see the module).

    Built once the model exists and ``solve.c`` is set (build); lives with the
    model stage. Tables index the eligible points: kept (in-window) points of
    fibers the gauge did not drop.
    """

    def __init__(self, solve, config, device, log=print):
        self.device = device
        self.log = log
        gauge = largest_cc_gauge(solve, config)
        self.gauge = gauge
        self.c0 = gauge['c0']
        keep_fiber = ~gauge['dropped']
        points = np.flatnonzero(keep_fiber[solve.point_fiber])
        self.num_points = len(points)
        fiber = solve.point_fiber[points]
        fibers, fiber_local, fiber_count = np.unique(fiber, return_inverse=True, return_counts=True)
        fiber_start = np.cumsum(fiber_count) - fiber_count
        ccs, fiber_cc = np.unique(solve.fiber_cc[fibers], return_inverse=True)
        self.num_cc = len(ccs)
        self.cc_fiber_count = np.bincount(fiber_cc, minlength=self.num_cc)
        # Per kept fiber (points are grouped by fiber, in this order) and per
        # CC, for the whole-set satisfaction pass (fiber_satisfaction).
        self.fiber_cc = fiber_cc
        self.fiber_point_count = fiber_count
        self.fiber_uuid = solve.fiber_uuid[fibers]
        self.cc_component = solve.cc_component[ccs]
        self.run_uuid = solve.run_uuid
        largest_local = np.flatnonzero(ccs == solve.largest_cc)
        self.largest_local = int(largest_local[0]) if len(largest_local) else -1

        # Pairs: consecutive kept points of one in-window piece.
        piece = np.searchsorted(solve.piece_offsets, points, side='right') - 1
        pair_first = np.flatnonzero((points[1:] == points[:-1] + 1) & (piece[1:] == piece[:-1]))
        pair_fiber = fiber_local[pair_first]
        pair_count = np.bincount(pair_fiber, minlength=len(fibers))
        pair_start = np.cumsum(pair_count) - pair_count

        as_long = lambda values: torch.as_tensor(values, dtype=torch.int64, device=device)  # noqa: E731
        self.zyx = torch.as_tensor(solve.zyx[points], dtype=torch.float32, device=device)
        self.w_s = torch.as_tensor(solve.w_s[points], dtype=torch.float32, device=device)
        self.vertical = torch.as_tensor(
            solve.family[fiber] == FAMILY_V, dtype=torch.bool, device=device)
        self.n0 = solve.n0[points]
        self.point_z = solve.zyx[points, 0]
        self.point_largest = fiber_cc[fiber_local] == self.largest_local
        self.points = _Table(fiber_start, fiber_count, fiber_cc, self.num_cc, device)
        self.pairs = _Table(pair_start, pair_count, fiber_cc, self.num_cc, device)
        self.pair_first = as_long(pair_first)
        self.cc_is_largest = torch.arange(self.num_cc, device=device) == self.largest_local
        # The other CCs' points, for the DT-target refresh.
        others = np.flatnonzero(~self.point_largest)
        self.other_points = as_long(others)
        self.other_point_cc = as_long(fiber_cc[fiber_local[others]])
        self.offset = None
        self.has_offsets = False
        self.refill_vertical_offsets(config)

    @classmethod
    def build(cls, solve, config, device, log=print):
        """The loss for a solve whose c is measured, or None (logged) when no
        fiber point lies in the z window."""
        if solve is None:
            return None
        if solve.c is None:
            log('fiber loss: off for this run: no fiber point in the z window')
            return None
        loss = cls(solve, config, device, log=log)
        for line in loss.report():
            log(line)
        if loss.num_points == 0:
            log('fiber loss: off for this run: every fiber point in the z window '
                'belongs to a dropped fiber')
            return None
        return loss

    def report(self):
        gauge = self.gauge
        dropped = int(gauge['dropped'].sum())
        low, high = gauge['kept_range']
        largest = self.cc_fiber_count[self.largest_local] if self.largest_local >= 0 else 0
        return [
            f'fiber loss: largest component: the outlier backstop dropped {dropped:,} fibers '
            f'(whole-fiber sheet '
            f'labels outside [{gauge["p_low"]:g} - {gauge["margin"]:g}, {gauge["p_high"]:g} + '
            f'{gauge["margin"]:g}], the {gauge["percentile"]:g}th-percentile minimum and '
            f'maximum); kept labels [{low}, {high}]',
            f'fiber loss: gauge n_low = {gauge["n_low"]}, c0 = {self.c0} (innermost kept sheet '
            f'at winding {self.c0 + gauge["n_low"]}); kept largest-component fibers with their '
            f'minimum at n_low: {gauge["at_n_low"]}, at n_low + 1: {gauge["at_n_low_plus_1"]}',
            f'fiber loss: {self.num_points:,} points of {int(self.cc_fiber_count.sum()):,} '
            f'fibers in {self.num_cc:,} components ({largest:,} fibers in the largest), '
            f'{self.pairs.size:,} consecutive pairs']

    def refill_vertical_offsets(self, config):
        """Per-point outward offsets from each fiber's family and the current
        settings; drops the DT targets computed against the old ones."""
        offset = vertical_radial_offset(config)
        self.offset = torch.where(
            self.vertical, torch.tensor(offset, device=self.device),
            torch.tensor(0.0, device=self.device))
        self.has_offsets = offset != 0.0 and bool(self.vertical.any())
        self.reset_targets()

    def reset_targets(self):
        self.targets = None
        self.targets_iteration = None

    # --- evaluation -------------------------------------------------------

    def _evaluate(self, transform, dr, index):
        """(spiral, theta, n, rho, s_vox) at points ``index``: s_vox = s * dr,
        n and rho detached."""
        zyx = self.zyx[index]
        spiral = _transform_shape_bucketed(transform, zyx)
        theta, _, shifted = get_theta_and_radii(spiral[..., 1:], dr)
        n, rho = branch_target(self.w_s[index], theta.detach())
        if self.has_offsets:
            # Constant with respect to the fit: the offset is data, not a
            # lever the transform could shrink by compressing the normal.
            stretch = get_radial_normal_stretch(transform, zyx, spiral_zyx=spiral.detach())
            shifted = shifted - self.offset[index] * stretch
        return spiral, theta, n, rho, shifted - n * dr

    def evaluate_points(self, transform, dr, index):
        """``(s, ok)`` at points ``index`` (int64 tensor), under no_grad in
        chunks of CHUNK points: s in turns (float32) and the phase
        acceptance (bool)."""
        values, masks = [], []
        with torch.no_grad():
            for begin in range(0, index.numel(), CHUNK):
                _, _, _, rho, s_vox = self._evaluate(transform, dr, index[begin:begin + CHUNK])
                values.append(s_vox / dr)
                masks.append(accepted(rho))
        if not values:
            return (torch.zeros(0, device=self.device),
                    torch.zeros(0, dtype=torch.bool, device=self.device))
        return torch.cat(values), torch.cat(masks)

    def refresh_targets(self, transform, dr, iteration):
        """T_CC = round(median s) over each non-largest CC's accepted points;
        NaN for a CC with none. Logs how many targets changed."""
        targets = torch.full((self.num_cc,), float('nan'), device=self.device)
        if self.other_points.numel():
            s, ok = self.evaluate_points(transform, dr, self.other_points)
            targets, _ = whole_winding_targets(self.other_point_cc, s, ok, self.num_cc)
            targets[self.cc_is_largest] = float('nan')
        previous = self.targets
        has = torch.isfinite(targets)
        if previous is None:
            changed = None
        else:
            same = (torch.isfinite(previous) == has) & (~has | (previous == targets))
            changed = int((~same).sum())
        self.targets = targets
        self.targets_iteration = iteration
        self.log(f'fiber DT targets (step {iteration}): {int(has.sum()):,} of '
                 f'{self.num_cc - (self.largest_local >= 0):,} other components'
                 + ('' if changed is None else f', {changed:,} changed'))

    def loss(self, transform, dr, config, *, iteration, dt_active, generator=None):
        """The weighted-mean Huber error of the step's sampled points, in
        scroll voxels (an error of e turns is e * dr voxels)."""
        if dt_active and (self.targets is None or iteration - self.targets_iteration
                          >= max(1, int(config['fiber_dt_target_update_interval']))):
            self.refresh_targets(transform, dr, iteration)
        cc, index = self.points.sample(
            int(config['sample_count_fiber_ccs']),
            int(config['sample_count_points_per_fiber_cc']), generator,
            exponent=config['fiber_sampling_length_exponent'])
        spiral, theta, n, rho, s_vox = self._evaluate(transform, dr, index)
        valid = accepted(rho)
        largest = self.cc_is_largest[cc]
        if dt_active:
            other_target = self.targets[cc]
            has_target = torch.isfinite(other_target)
        else:
            median, count = _group_median(cc, (s_vox / dr).detach(), valid, self.num_cc)
            other_target = median[cc]
            has_target = count[cc] >= MIN_MEDIAN_SAMPLES
        valid = valid & (largest | has_target)
        # A point without a target gets a finite stand-in and weight 0: an
        # infinite or NaN target would turn its zero weight into NaN.
        target = torch.where(largest, torch.full_like(s_vox, float(self.c0)),
                             torch.where(has_target, other_target, torch.zeros_like(s_vox)))
        residual = s_vox - target * dr
        per_point = _huber_abs(residual, float(config['fiber_huber_delta_turns']) * dr.detach())
        weight = valid.to(per_point.dtype)
        loss = (per_point * weight).sum() / weight.sum().clamp(min=1)
        if diagnostics_enabled():
            # The target radius: the point's own radius moved by the error
            # (the offset times its stretch is part of both).
            radius = (torch.linalg.norm(spiral[..., 1:], dim=-1) - residual).detach()
            display = torch.stack([spiral[..., 0], torch.sin(theta) * radius,
                                   torch.cos(theta) * radius], dim=-1).detach()
            record_loss_samples('fiber', spiral, per_point, valid, display_spiral_zyx=display)
        return loss

    # --- health metrics -----------------------------------------------------

    @torch.no_grad()
    def metrics(self, transform, dr, iteration):
        """The label health checks on a fixed-size sample drawn from the
        metrics' own generator: the rejected fraction (|rho| > RHO_MAX), the
        largest CC's mean |s - c0| (turns), its signed median and the share of
        it off by more than half a turn, and the label-step mismatch fraction of consecutive
        accepted pairs (n must step exactly where q does, at theta = 0). Fibers
        weigh alike here whatever fiber_sampling_length_exponent says, so the
        checks compare across settings."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(METRIC_SEED + int(iteration))
        cc, index = self.points.sample(METRIC_SAMPLES, 1, generator)
        _, _, _, rho, s_vox = self._evaluate(transform, dr, index)
        ok = accepted(rho)
        largest = self.cc_is_largest[cc] & ok
        signed = (s_vox / dr - self.c0)[largest]
        error = signed.abs()
        # Keys without a sample to measure are left out, never NaN.
        metrics = {'fiber_rejected_fraction': float(1.0 - ok.float().mean())}
        if error.numel():
            metrics.update({
                'fiber_largest_cc_median_offset': float(signed.median()),
                'fiber_largest_cc_mean_abs_offset': float(error.mean()),
                'fiber_largest_cc_off_fraction': float((error > 0.5).float().mean()),
            })
        if self.pairs.size:
            _, item = self.pairs.sample(METRIC_SAMPLES, 1, generator)
            first = self.pair_first[item]
            both = torch.cat([first, first + 1])
            _, theta, n, rho, _ = self._evaluate(transform, dr, both)
            half = first.numel()
            step = theta[half:] - theta[:half]
            k = (step < -math.pi).to(n.dtype) - (step > math.pi).to(n.dtype)
            pair_ok = accepted(rho[:half]) & accepted(rho[half:])
            mismatch = (n[half:] - n[:half] != k) & pair_ok
            count = int(pair_ok.sum())
            if count:
                metrics['fiber_label_step_pairs'] = count
                metrics['fiber_label_step_mismatch'] = float(mismatch.sum()) / count
        return metrics

    def metrics_lines(self, metrics, config, label):
        """The printed metrics line and any warnings."""
        parts = [f'rejected {100 * metrics["fiber_rejected_fraction"]:.2f}%']
        if 'fiber_largest_cc_mean_abs_offset' in metrics:
            parts.append(
                f'largest component mean |s - c0| = '
                f'{metrics["fiber_largest_cc_mean_abs_offset"]:.3f} turns (median s - c0 = '
                f'{metrics["fiber_largest_cc_median_offset"]:+.2f}), '
                f'{100 * metrics["fiber_largest_cc_off_fraction"]:.1f}% off by more than half a turn')
        if 'fiber_label_step_mismatch' in metrics:
            parts.append(
                f'label-step mismatch {100 * metrics["fiber_label_step_mismatch"]:.3f}% of '
                f'{metrics["fiber_label_step_pairs"]:,} pairs')
        lines = [f'fiber metrics ({label}): ' + '; '.join(parts)]
        if metrics['fiber_rejected_fraction'] > float(config['fiber_rejected_fraction_warn']):
            lines.append(
                f'WARNING: {100 * metrics["fiber_rejected_fraction"]:.2f}% of fiber points are '
                f'rejected (|rho| > {RHO_MAX}), above fiber_rejected_fraction_warn: the model\'s '
                'angle has drifted near half a turn from the raw angle there')
        if metrics.get('fiber_label_step_mismatch', 0.0) > float(
                config['fiber_label_step_mismatch_warn']):
            lines.append(
                f'WARNING: {100 * metrics["fiber_label_step_mismatch"]:.3f}% of consecutive fiber '
                'point pairs step their sheet label where Spiral\'s winding does not, above '
                'fiber_label_step_mismatch_warn: the model\'s angle has drifted past half a '
                'turn somewhere')
        return lines

    # --- winding-range and shell-index inference ----------------------------

    def range_windings(self, z_begin, z_end):
        """Spiral winding labels n0 + c0 of the largest CC's kept points in
        [z_begin, z_end) (int64), for winding-range and shell-index
        inference; the other CCs' labels have an arbitrary offset."""
        inside = self.point_largest & (self.point_z >= z_begin) & (self.point_z < z_end)
        return self.n0[inside] + self.c0

