"""Unit tests for adaptive_sampling.py (pure NumPy; no GPU / no fit_spiral)."""
import numpy as np
import adaptive_sampling as A


def test_stratum_ids_voxel():
    # two clusters far apart -> two strata
    zyxs = np.array([[0, 0, 0], [10, 10, 10], [11, 11, 11], [0, 1, 1]], float)
    ids = A.compute_stratum_ids(zyxs, bin_scheme='voxel', voxel_size=5.0)
    assert ids[0] == ids[3]      # both in voxel (0,0,0)
    assert ids[1] == ids[2]      # both in voxel (2,2,2)
    assert ids[0] != ids[1]
    assert set(ids.tolist()) == {0, 1}


def test_stratum_ids_zradang_shape_determinism():
    rng = np.random.default_rng(0)
    zyxs = rng.uniform(0, 100, size=(50, 3))
    kw = dict(bin_scheme='zradang', num_z_slabs=4, num_radial_rings=3, num_angular_sectors=4)
    a = A.compute_stratum_ids(zyxs, **kw)
    b = A.compute_stratum_ids(zyxs, **kw)
    assert a.shape == (50,)
    assert np.array_equal(a, b)               # deterministic
    assert a.min() >= 0


def test_density_per_unit_counts():
    sid = np.array([0, 0, 0, 1])              # 3 in stratum 0, 1 in stratum 1
    dens = A.stratum_density_per_unit(sid)
    assert np.allclose(dens, [3, 3, 3, 1])
    # area measure: stratum density = summed measure
    dens_a = A.stratum_density_per_unit(sid, measure=[2.0, 2.0, 2.0, 5.0])
    assert np.allclose(dens_a, [6, 6, 6, 5])


def test_combine_noop_identity():
    base = np.array([0.1, 0.2, 0.3, 0.4])
    # nothing active -> same object
    assert A.combine_sampling_weights(base, tau=0.0, beta=0.0) is base
    # tau=0 is a no-op even with density supplied
    assert A.combine_sampling_weights(base, density=np.array([3., 3., 3., 1.]), tau=0.0) is base


def test_combine_density_equalization():
    base = np.full(4, 0.25)
    dens = np.array([3., 3., 3., 1.])
    w = A.combine_sampling_weights(base, density=dens, tau=1.0, max_ratio=0)
    # tau=1 with count density equalizes strata: P(stratum)=0.5 each; w_B/w_A == 3
    assert abs((w[3] / w[0]) - 3.0) < 1e-9
    assert abs(w.sum() - 1.0) < 1e-12
    # tau=0.5 -> ratio sqrt(3)
    w2 = A.combine_sampling_weights(base, density=dens, tau=0.5, max_ratio=0)
    assert abs((w2[3] / w2[0]) - np.sqrt(3.0)) < 1e-9


def test_combine_residual():
    base = np.full(3, 1 / 3)
    ema = np.array([1.0, 2.0, 4.0])
    w = A.combine_sampling_weights(base, residual_ema=ema, beta=1.0, eps=0.0, max_ratio=0)
    assert np.allclose(w, ema / ema.sum())     # beta=1 -> w ∝ base*ema ∝ ema
    w0 = A.combine_sampling_weights(base, residual_ema=ema, beta=0.0)
    assert w0 is base                          # beta=0 no-op


def test_residual_ema_closed_form_and_dupes():
    e = A.ResidualEMA(3, init=0.0, decay=0.5)
    e.update([0], [1.0])                        # ema0 = 0.5*0 + 0.5*1 = 0.5
    assert abs(e.values()[0] - 0.5) < 1e-12
    e.update([0], [1.0])                        # 0.5*0.5 + 0.5*1 = 0.75
    assert abs(e.values()[0] - 0.75) < 1e-12
    # duplicate indices in one step are averaged before the EMA step
    e2 = A.ResidualEMA(2, init=0.0, decay=0.0)  # decay=0 -> ema == this step's mean
    e2.update([1, 1], [2.0, 4.0])
    assert abs(e2.values()[1] - 3.0) < 1e-12    # mean(2,4)=3
    assert e2.values()[0] == 0.0                # untouched


def test_blue_noise_spread_beats_random():
    rng = np.random.default_rng(1)
    # clustered cloud
    pts = np.vstack([rng.normal(0, 0.2, (200, 3)), rng.normal(5, 0.2, (10, 3))])
    np.random.seed(0)
    fps = A.blue_noise_subsample(pts, 12, method='fps', seed_idx=0)
    rnd = rng.choice(len(pts), 12, replace=False)

    def min_pair_dist(idx):
        p = pts[idx]
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        d[np.diag_indices_from(d)] = np.inf
        return d.min()

    assert min_pair_dist(fps) > min_pair_dist(rnd)


def test_curriculum_and_resolve():
    assert A.curriculum_factor('linear', 0.0, 0.0, 1.0) == 0.0
    assert A.curriculum_factor('linear', 1.0, 0.0, 1.0) == 1.0
    assert abs(A.curriculum_factor('linear', 0.5, 0.0, 1.0) - 0.5) < 1e-12
    cfg = dict(density_equalize_enable=True, density_equalize_tau=0.8,
               residual_adaptive_enable=True, residual_ema_beta=0.6,
               curriculum_enable=False)
    assert A.resolve_tau_beta(cfg, 100, 1000) == (0.8, 0.6)
    cfg2 = dict(cfg, curriculum_enable=True, curriculum_schedule='linear',
                curriculum_warmup_frac=0.0, curriculum_density_end_frac=1.0,
                curriculum_hard_start_frac=0.5, curriculum_tau_final=0.8, curriculum_beta_final=0.6)
    tau, beta = A.resolve_tau_beta(cfg2, 500, 1000)   # t=0.5
    assert abs(tau - 0.8 * 0.5) < 1e-9                # density ramps 0->1 over [0,1]
    assert abs(beta - 0.0) < 1e-9                     # hard ramps 0->1 over [0.5,1]; at t=0.5 -> 0


def test_frontier_weight():
    # patches just below the 0.95 threshold (0.88, 0.92) should dominate; already-satisfied
    # (0.96, 0.99) and hopeless (0.10) get only the floor.
    frac = np.array([0.10, 0.50, 0.88, 0.92, 0.96, 0.99])
    w = A.frontier_weight(frac, threshold=0.95, peak=0.9, width=0.12, floor=0.05)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[2] > w[0] and w[2] > w[4] and w[3] > w[5]      # frontier beats hopeless & satisfied
    assert w[4] < w[2] and w[5] < w[3]                       # satisfied patches down-weighted
    # all-satisfied (degenerate) -> safe uniform
    w2 = A.frontier_weight(np.array([0.99, 0.98, 0.97]), threshold=0.95)
    assert abs(w2.sum() - 1.0) < 1e-9 and np.allclose(w2, 1 / 3, atol=0.34)


def test_frontier_two_sided_retention():
    # F7: retain>0 boosts just-above-threshold patches (anti-regression) without disturbing the
    # below-threshold frontier; retain=0 is exactly the one-sided F6 weight.
    frac = np.array([0.10, 0.88, 0.96, 0.999])
    w0 = A.frontier_weight(frac, threshold=0.95, peak=0.9, width=0.12, floor=0.05)
    wr = A.frontier_weight(frac, threshold=0.95, peak=0.9, width=0.12, floor=0.05,
                           retain=0.5, retain_width=0.04)
    assert abs(wr.sum() - 1.0) < 1e-9
    assert wr[2] > w0[2]          # just-satisfied patch boosted vs one-sided
    assert wr[2] > wr[3]          # ...more than a comfortably-satisfied patch
    assert wr[1] > wr[2]          # the below-threshold frontier still dominates
    # retain=0 reproduces the one-sided frontier exactly
    assert np.allclose(w0, A.frontier_weight(frac, threshold=0.95, peak=0.9, width=0.12,
                                             floor=0.05, retain=0.0))


if __name__ == '__main__':
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    n_pass = 0
    for fn in fns:
        try:
            fn()
            print(f'PASS {fn.__name__}')
            n_pass += 1
        except Exception:
            print(f'FAIL {fn.__name__}')
            traceback.print_exc()
    print(f'\n{n_pass}/{len(fns)} passed')
    raise SystemExit(0 if n_pass == len(fns) else 1)
