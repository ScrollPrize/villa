"""Adaptive annotation-sampling strategies for fit_spiral.py.

All functions are pure NumPy (no torch / no fit_spiral import) so they can be
unit-tested in isolation. fit_spiral.py imports this module and wires the
helpers into its patch / PCL / track samplers, gated by config keys whose
defaults reproduce the original (uniform / sqrt-area) behaviour exactly.

Granularity: reweighting acts on the *sampled unit* (patch index, PCL index,
track index) and on stable *scan-space strata*. Within-unit points are
re-randomised each step, so per-point EMA is not maintained.

Core ideas
----------
* density-equalization (F2): bin units into scan-space strata; multiply each
  unit's selection weight by ``density(stratum)**(-tau)`` so dense regions of
  the volume stop dominating the gradient. ``tau=0`` -> no-op.
* residual-adaptive / hard-example (F3): keep an EMA of each unit's loss and
  multiply its weight by ``(eps+ema)**beta``. ``beta=0`` -> no-op.
* coverage / blue-noise (F4): farthest-point / Poisson-disk subsample of an
  oversampled candidate set to de-cluster dense regions.
"""

import numpy as np

EPS_LOG = 1e-300


# --------------------------------------------------------------------------- #
# F2 helpers: scan-space stratification + per-stratum density
# --------------------------------------------------------------------------- #
def compute_stratum_ids(
    zyxs,
    *,
    bin_scheme='voxel',
    voxel_size=256.0,
    num_z_slabs=16,
    num_radial_rings=12,
    num_angular_sectors=8,
    z_to_umbilicus_yx=None,
    z_begin=None,
    z_end=None,
):
    """Map each unit's representative scan-space coord (N,3 z,y,x) to a compact
    integer stratum id in [0, S). Deterministic; pure function of geometry."""
    zyxs = np.asarray(zyxs, dtype=np.float64)
    if zyxs.ndim != 2 or zyxs.shape[1] != 3:
        raise ValueError(f'zyxs must be (N,3); got {zyxs.shape}')
    n = zyxs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    z, y, x = zyxs[:, 0], zyxs[:, 1], zyxs[:, 2]

    if bin_scheme == 'voxel':
        key = np.stack(
            [
                np.floor(z / voxel_size),
                np.floor(y / voxel_size),
                np.floor(x / voxel_size),
            ],
            axis=1,
        )
        _, ids = np.unique(key, axis=0, return_inverse=True)
        return ids.astype(np.int64)

    if bin_scheme == 'zradang':
        zlo = float(z.min()) if z_begin is None else float(z_begin)
        zhi = float(z.max()) if z_end is None else float(z_end)
        span = max(zhi - zlo, 1e-6)
        iz = np.clip(np.floor((z - zlo) / span * num_z_slabs), 0, num_z_slabs - 1)
        if z_to_umbilicus_yx is not None:
            cyx = np.asarray(z_to_umbilicus_yx(z), dtype=np.float64).reshape(-1, 2)
            cy, cx = cyx[:, 0], cyx[:, 1]
        else:
            cy = np.full(n, float(y.mean()))
            cx = np.full(n, float(x.mean()))
        dy, dx = y - cy, x - cx
        r = np.sqrt(dy * dy + dx * dx)
        rmax = max(float(r.max()), 1e-6)
        ir = np.clip(np.floor(r / rmax * num_radial_rings), 0, num_radial_rings - 1)
        ang = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)  # [0,1)
        ia = np.clip(np.floor(ang * num_angular_sectors), 0, num_angular_sectors - 1)
        key = (iz * num_radial_rings + ir) * num_angular_sectors + ia
        _, ids = np.unique(key, return_inverse=True)
        return ids.astype(np.int64)

    raise ValueError(f'unknown bin_scheme {bin_scheme!r}')


def stratum_density_per_unit(stratum_id, measure=None, min_count=1.0):
    """Per-unit stratum density = sum of ``measure`` over units sharing the
    stratum (count if measure is None), floored at ``min_count``. Returns an
    array aligned with ``stratum_id`` (the raw density, NOT yet exponentiated)."""
    stratum_id = np.asarray(stratum_id)
    n = stratum_id.shape[0]
    if n == 0:
        return np.zeros(0)
    vals = np.ones(n, dtype=np.float64) if measure is None else np.asarray(measure, dtype=np.float64)
    s = int(stratum_id.max()) + 1
    density = np.zeros(s, dtype=np.float64)
    np.add.at(density, stratum_id, vals)
    density = np.maximum(density, float(min_count))
    return density[stratum_id]


# --------------------------------------------------------------------------- #
# F3 helper: per-unit residual EMA (online hard-example mining)
# --------------------------------------------------------------------------- #
class ResidualEMA:
    """Exponential moving average of per-unit residual/loss. Units never yet
    sampled keep ``init`` so they look average (not zero)."""

    def __init__(self, n, init=1.0, decay=0.9):
        self.ema = np.full(int(n), float(init), dtype=np.float64)
        self.seen = np.zeros(int(n), dtype=bool)
        self.decay = float(decay)

    def update(self, indices, residuals, decay=None):
        decay = self.decay if decay is None else float(decay)
        indices = np.asarray(indices).reshape(-1)
        residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        if indices.shape[0] == 0:
            return
        n = self.ema.shape[0]
        s = np.zeros(n, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        np.add.at(s, indices, residuals)          # sum residuals per index
        np.add.at(c, indices, 1.0)                # count per index (averages duplicates)
        upd = c > 0
        mean_r = np.zeros(n, dtype=np.float64)
        mean_r[upd] = s[upd] / c[upd]
        self.ema[upd] = decay * self.ema[upd] + (1.0 - decay) * mean_r[upd]
        self.seen[upd] = True

    def values(self):
        return self.ema


# --------------------------------------------------------------------------- #
# curriculum schedule
# --------------------------------------------------------------------------- #
def curriculum_factor(schedule, t, start, end):
    """Ramp in [0,1] as training fraction t goes start->end."""
    if end <= start:
        return 1.0 if t >= end else 0.0
    u = float(np.clip((t - start) / (end - start), 0.0, 1.0))
    if schedule == 'linear':
        return u
    if schedule == 'cosine':
        return float(0.5 - 0.5 * np.cos(np.pi * u))
    if schedule == 'step':
        return 1.0 if u >= 1.0 else 0.0
    return u


def resolve_tau_beta(cfg, iteration, num_steps):
    """Current (tau, beta) given the curriculum config. Without curriculum,
    returns the static configured values."""
    tau = float(cfg['density_equalize_tau']) if cfg.get('density_equalize_enable') else 0.0
    beta = float(cfg['residual_ema_beta']) if cfg.get('residual_adaptive_enable') else 0.0
    if not cfg.get('curriculum_enable') or iteration is None or not num_steps:
        return tau, beta
    t = float(iteration) / float(num_steps)
    sched = cfg.get('curriculum_schedule', 'linear')
    warm = float(cfg.get('curriculum_warmup_frac', 0.0))
    d_end = float(cfg.get('curriculum_density_end_frac', 1.0))
    h_start = float(cfg.get('curriculum_hard_start_frac', 0.5))
    tau_final = float(cfg.get('curriculum_tau_final', 0.0)) or tau
    beta_final = float(cfg.get('curriculum_beta_final', 0.0)) or beta
    tau = tau_final * curriculum_factor(sched, t, warm, d_end)
    beta = beta_final * curriculum_factor(sched, t, h_start, 1.0)
    return tau, beta


# --------------------------------------------------------------------------- #
# the single choke-point combiner
# --------------------------------------------------------------------------- #
def combine_sampling_weights(
    base_p,
    *,
    density=None,
    residual_ema=None,
    tau=0.0,
    beta=0.0,
    eps=1e-6,
    max_ratio=50.0,
):
    """Multiply base selection probs by density**(-tau) and (eps+ema)**beta in
    log space, clip the weight ratio, renormalise. Returns ``base_p`` unchanged
    (same object) when neither component is active (bit-identical baseline)."""
    density_active = density is not None and tau != 0.0
    residual_active = residual_ema is not None and beta != 0.0
    if not density_active and not residual_active:
        return base_p
    base_p = np.asarray(base_p, dtype=np.float64)
    logw = np.log(np.maximum(base_p, EPS_LOG))
    if density_active:
        logw = logw - tau * np.log(np.maximum(np.asarray(density, dtype=np.float64), 1e-12))
    if residual_active:
        logw = logw + beta * np.log(np.maximum(np.asarray(residual_ema, dtype=np.float64), 0.0) + eps)
    logw -= logw.max()
    w = np.exp(logw)
    if max_ratio and max_ratio > 0:
        w = np.maximum(w, w.max() / float(max_ratio))
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return base_p
    return w / total


# --------------------------------------------------------------------------- #
# F6 helper: satisfaction-frontier (metric-aware) weighting
# --------------------------------------------------------------------------- #
def frontier_weight(frac, *, threshold=0.95, peak=0.9, width=0.12, floor=0.05, hopeless=0.35,
                    retain=0.0, retain_width=0.04):
    """Per-patch sampling weight that targets the satisfaction FRONTIER.

    `frac` = per-patch satisfied-quad fraction (satisfied_areas/total_areas). A patch
    counts as "satisfied" once frac >= `threshold`, so the highest-marginal-value targets
    are the patches just BELOW threshold (one nudge from flipping). Weight peaks near
    `peak` (a Gaussian of scale `width`), is `floor` for already-satisfied (>=threshold)
    and near-hopeless (<hopeless) patches, and is normalised to sum 1. This is the right
    surrogate for the non-decomposable COUNT metric: weight examples by how close they are
    to changing the satisfied/not-satisfied indicator. Returns uniform if degenerate.

    F7 two-sided / hysteresis term (`retain` > 0): patches JUST ABOVE threshold get an extra
    `retain`-scaled Gaussian (scale `retain_width`) on top of `floor`, so freshly-satisfied
    patches keep being sampled and don't slip back below threshold while the optimizer pushes
    their neighbours over. Targets the regression/oscillation failure mode of the one-sided
    frontier (patches flipping satisfied<->unsatisfied step to step). `retain=0` is exactly
    the original one-sided behaviour.
    """
    frac = np.asarray(frac, dtype=np.float64)
    n = frac.shape[0]
    if n == 0:
        return np.zeros(0)
    bump = np.exp(-((frac - peak) / max(width, 1e-6)) ** 2)
    active = (frac < threshold) & (frac >= hopeless)
    w = float(floor) + active * bump
    if retain > 0:
        retained = (frac >= threshold) * float(retain) * np.exp(
            -((frac - threshold) / max(retain_width, 1e-6)) ** 2)
        w = w + retained
    s = w.sum()
    return w / s if s > 0 and np.isfinite(s) else np.full(n, 1.0 / n)


# --------------------------------------------------------------------------- #
# F4 helper: blue-noise / farthest-point subsample
# --------------------------------------------------------------------------- #
def blue_noise_subsample(zyxs, k, *, method='fps', min_dist=0.0, seed_idx=None):
    """Pick k spread-out indices from candidate coords (P,3). FPS = greedy
    farthest-point; poisson = min-distance rejection. Falls back to all when
    k>=P."""
    zyxs = np.asarray(zyxs, dtype=np.float64)
    p = zyxs.shape[0]
    if k >= p:
        return np.arange(p)
    if method == 'fps':
        start = int(np.random.randint(p)) if seed_idx is None else int(seed_idx)
        idx = np.empty(k, dtype=np.int64)
        idx[0] = start
        d = np.full(p, np.inf)
        for i in range(1, k):
            diff = zyxs - zyxs[idx[i - 1]]
            d = np.minimum(d, np.einsum('ij,ij->i', diff, diff))
            d[idx[:i]] = -1.0
            idx[i] = int(np.argmax(d))
        return idx
    if method == 'poisson':
        order = np.random.permutation(p)
        md2 = float(min_dist) * float(min_dist)
        chosen = []
        for i in order:
            if len(chosen) >= k:
                break
            if not chosen:
                chosen.append(int(i))
                continue
            diff = zyxs[chosen] - zyxs[i]
            if np.all(np.einsum('ij,ij->i', diff, diff) >= md2):
                chosen.append(int(i))
        if len(chosen) < k:  # top up with remaining order
            extra = [int(i) for i in order if i not in set(chosen)]
            chosen.extend(extra[: k - len(chosen)])
        return np.asarray(chosen[:k], dtype=np.int64)
    raise ValueError(f'unknown method {method!r}')
