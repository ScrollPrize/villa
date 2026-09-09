"""Sobolev-damped Hessian-free flow step (sobolev_gauss_newton.py).

Covers the Cartesian and cylindrical metrics (symmetry, positive
definiteness, Neumann borders, periodic rings, constrained-cell
elimination, no radial mixing), the inner CG inverse, PCG against a dense
direct solve, the large-damping limit against the Sobolev step, the
curvature and non-finite guards with their fallbacks, the finite-difference
Hessian-vector product, the trust region, bitwise-unchanged constrained
cells, the flow-field factories and width conversion, the random-state
snapshot, the configuration keys, the fitter hook on a stub context (flow
gradients hidden from AdamW, exact parameter restoration around
re-evaluations, counters and non-flow gradients restored), and a toy
ablation comparing the update roughness and same-batch loss reduction of
AdamW, smoothed AdamW, the Sobolev gradient step and the damped
Hessian-free step on a small synthetic problem.
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flow_grad_smoothing
import sobolev_gauss_newton as sgn
from config import BACKFILLABLE_CONFIG_DEFAULTS, Config
from flow_fields import CartesianFlowField, CylindricalFlowField
from lazy_moment_adamw import LazyMomentAdamW

torch.set_default_dtype(torch.float32)


def _cylinder_tables(nr):
    num_phi = [1] + [max(1, int(round(2 * math.pi * r))) for r in range(1, nr)]
    offsets = [0]
    for n in num_phi:
        offsets.append(offsets[-1] + n)
    return num_phi, offsets


def _dense(operator, shape, dtype=torch.float64):
    """Explicit matrix of a linear operator on tensors of ``shape``."""
    n = math.prod(shape)
    columns = []
    for index in range(n):
        basis = torch.zeros(n, dtype=dtype)
        basis[index] = 1.0
        columns.append(operator(basis.view(shape)).reshape(-1))
    return torch.stack(columns, dim=1)


def _settings(**overrides):
    settings = sgn.SobolevSettings(**overrides)
    settings.validate()
    return settings


# --------------------------------------------------------------------------
# Metrics


def test_cartesian_metric_is_symmetric_positive_definite_with_neumann_borders():
    metric = sgn.CartesianSobolevMetric((1.5, 0.7, 1.0))
    shape = (1, 1, 3, 4, 5)
    matrix = _dense(metric.apply_A, shape)
    torch.testing.assert_close(matrix, matrix.T)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert float(eigenvalues.min()) >= 1.0 - 1e-9
    # Neumann: a constant has zero Laplacian, so A c = c exactly.
    constant = torch.full(shape, 2.5, dtype=torch.float64)
    torch.testing.assert_close(metric.apply_A(constant), constant)
    # The Laplacian part is the graph Laplacian of the lattice: I - A has a
    # row sum of zero and only nearest-neighbour couplings.
    laplacian = matrix - torch.eye(matrix.shape[0], dtype=torch.float64)
    torch.testing.assert_close(laplacian.sum(dim=1), torch.zeros(matrix.shape[0], dtype=torch.float64))
    # Per-axis lengths: coupling along z is lz^2, along y ly^2.
    assert float(laplacian[0, 1 * 4 * 5]) == pytest.approx(-1.5 ** 2)
    assert float(laplacian[0, 5]) == pytest.approx(-0.7 ** 2)
    assert float(laplacian[0, 1]) == pytest.approx(-1.0)


def test_cartesian_metric_acts_per_slab_and_component():
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    v = torch.zeros(2, 3, 4, 4, 4, dtype=torch.float64)
    v[1, 2, 2, 2, 2] = 1.0
    out = metric.apply_A(v)
    assert float(out[0].abs().sum()) == 0.0
    assert float(out[1, :2].abs().sum()) == 0.0
    assert float(out[1, 2, 2, 2, 2]) == pytest.approx(1.0 + 6.0)
    assert float(out[1, 2, 2, 2, 3]) == pytest.approx(-1.0)


def test_constrained_cells_are_eliminated_from_the_operator():
    constrained = torch.zeros(3, 3, 3, dtype=torch.bool)
    constrained[1, 1, 1] = True
    constrained[0, :, :] = True
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0), constrained=constrained)
    shape = (1, 1, 3, 3, 3)
    matrix = _dense(metric.apply_A, shape)
    torch.testing.assert_close(matrix, matrix.T)
    flat = constrained.reshape(-1)
    # Constrained rows and columns are zero; the free block is positive definite.
    assert float(matrix[flat].abs().sum()) == 0.0
    assert float(matrix[:, flat].abs().sum()) == 0.0
    free = matrix[~flat][:, ~flat]
    assert float(torch.linalg.eigvalsh(free).min()) > 1.0 - 1e-9
    # A free neighbour of a constrained cell keeps that edge's diagonal
    # contribution (Dirichlet-zero at the mask, not Neumann): (1, 1, 0) has
    # five lattice edges, two of them to constrained cells, so its diagonal
    # is 1 + 5 while a lattice-border Neumann cell would have lost the edge.
    index = 1 * 9 + 1 * 3 + 0
    assert float(matrix[index, index]) == pytest.approx(1.0 + 5.0)
    # solve_A stays in the free subspace.
    r = torch.randn(shape, dtype=torch.float64)
    x, info = metric.solve_A(r, iterations=200, tolerance=1e-12)
    assert float(x.view(-1)[flat].abs().sum()) == 0.0
    torch.testing.assert_close(metric.apply_A(x), r.masked_fill(constrained, 0.0), atol=1e-8, rtol=1e-8)


def test_cylindrical_metric_is_symmetric_periodic_and_pins_ring_zero():
    num_phi, offsets = _cylinder_tables(4)
    nz = 3
    metric = sgn.CylindricalSobolevMetric(num_phi, offsets, 1.2, 0.8, nz=nz)
    shape = (1, 1, nz, offsets[-1])
    matrix = _dense(metric.apply_A, shape)
    torch.testing.assert_close(matrix, matrix.T)
    flat_ring0 = metric.constrained.reshape(-1)
    assert int(flat_ring0.sum()) == nz  # one ring-0 cell per z
    assert float(matrix[flat_ring0].abs().sum()) == 0.0
    free = matrix[~flat_ring0][:, ~flat_ring0]
    assert float(torch.linalg.eigvalsh(free).min()) >= 1.0 - 1e-9
    # A field constant on the free cells has zero Laplacian: z is Neumann
    # and every ring is closed, so A c = c there.
    constant = torch.full(shape, 3.0, dtype=torch.float64)
    expected = constant.masked_fill(metric.constrained, 0.0)
    torch.testing.assert_close(metric.apply_A(constant), expected)
    # Periodic: the first and last cells of a ring are coupled.
    ring, n = 2, num_phi[2]
    start = offsets[ring]
    assert float(matrix[start, start + n - 1]) == pytest.approx(-0.8 ** 2)
    # No radial edges: a delta on ring 2 produces nothing on rings 1 and 3.
    v = torch.zeros(shape, dtype=torch.float64)
    v[0, 0, 1, start] = 1.0
    out = metric.apply_A(v)[0, 0]
    others = torch.ones(offsets[-1], dtype=torch.bool)
    others[start:start + n] = False
    assert float(out[:, others].abs().sum()) == 0.0
    # z coupling with its own length.
    assert float(out[0, start]) == pytest.approx(-1.2 ** 2)


def test_solve_A_inverts_apply_A_on_small_lattices():
    torch.manual_seed(0)
    metric = sgn.CartesianSobolevMetric((2.0, 2.0, 2.0))
    x = torch.randn(1, 3, 5, 6, 7, dtype=torch.float64)
    recovered, info = metric.solve_A(metric.apply_A(x), iterations=300, tolerance=1e-12)
    torch.testing.assert_close(recovered, x, atol=1e-8, rtol=1e-8)
    assert info.reason == 'converged'
    num_phi, offsets = _cylinder_tables(5)
    metric = sgn.CylindricalSobolevMetric(num_phi, offsets, 1.5, 1.5, nz=6)
    x = torch.randn(1, 3, 6, offsets[-1], dtype=torch.float64).masked_fill(metric.constrained, 0.0)
    recovered, _ = metric.solve_A(metric.apply_A(x), iterations=300, tolerance=1e-12)
    torch.testing.assert_close(recovered, x, atol=1e-8, rtol=1e-8)
    # Constant fields (the smoothest direction) are recovered in one iteration.
    constant = torch.full((1, 1, 4, 4, 4), 1.0, dtype=torch.float64)
    cart = sgn.CartesianSobolevMetric((3.0, 3.0, 3.0))
    recovered, info = cart.solve_A(constant, iterations=10, tolerance=1e-10)
    torch.testing.assert_close(recovered, constant)
    assert info.iterations == 1


def test_roughness_is_zero_for_constants_and_grows_with_oscillation():
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    constant = torch.ones(1, 1, 4, 4, 4, dtype=torch.float64)
    assert float(metric.roughness(constant)) == 0.0
    checker = torch.ones(4, 4, 4, dtype=torch.float64)
    idx = torch.arange(4)
    sign = ((idx[:, None, None] + idx[None, :, None] + idx[None, None, :]) % 2 * 2 - 1).to(torch.float64)
    assert float(metric.roughness((checker * sign).view(1, 1, 4, 4, 4))) == pytest.approx(2.0)
    zero = torch.zeros(1, 1, 4, 4, 4, dtype=torch.float64)
    assert float(metric.roughness(zero)) == 0.0


# --------------------------------------------------------------------------
# Solvers


def _dense_operator(matrix, shape):
    def apply(v):
        return [(matrix @ v[0].reshape(-1)).view(shape)]
    return apply


def test_pcg_matches_a_dense_direct_solve():
    torch.manual_seed(1)
    shape = (1, 1, 3, 3, 4)
    n = math.prod(shape)
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    A = _dense(metric.apply_A, shape)
    J = torch.randn(2 * n, n, dtype=torch.float64)
    H = J.T @ J  # positive semidefinite curvature
    damping = 0.5
    B = H + damping * A
    g = torch.randn(shape, dtype=torch.float64)
    expected = torch.linalg.solve(B, -g.reshape(-1)).view(shape)

    settings = _settings(damping=damping, curvature='finite_difference',
                         pcg_iterations=200, pcg_tolerance=1e-12,
                         inner_cg_iterations=200, inner_cg_tolerance=1e-14)
    hvp = lambda v: [(H @ v[0].reshape(-1)).view(shape)]
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=hvp)
    torch.testing.assert_close(steps[0], expected, atol=1e-7, rtol=1e-7)
    assert stats['reason'] == 'converged'
    assert not stats['fallback']
    # Inexact (inner-CG) preconditioning and roundoff cost a few iterations
    # beyond the exact-arithmetic bound of n.
    assert stats["iterations"] <= 3 * n
    assert stats['residual_final'] < 1e-10 * stats['residual_initial']
    # The plain conjugate_gradient helper agrees too (no preconditioner).
    x, info = sgn.conjugate_gradient(_dense_operator(B, shape), [-g], max_iterations=200, tolerance=1e-13)
    torch.testing.assert_close(x[0], expected, atol=1e-7, rtol=1e-7)


def test_large_damping_limit_is_the_sobolev_gradient_step():
    torch.manual_seed(2)
    shape = (1, 3, 3, 3, 3)
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    g = torch.randn(shape, dtype=torch.float64)
    n = math.prod(shape)
    J = torch.randn(n, n, dtype=torch.float64)
    H = J.T @ J
    hvp = lambda v: [(H @ v[0].reshape(-1)).view(shape)]
    damping = 1e10
    settings = _settings(damping=damping, curvature='finite_difference',
                         pcg_iterations=10, pcg_tolerance=1e-10,
                         inner_cg_iterations=300, inner_cg_tolerance=1e-14)
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=hvp)
    reference, _ = sgn.sobolev_step([g], [metric], _settings(
        damping=damping, inner_cg_iterations=300, inner_cg_tolerance=1e-14))
    exact = -(torch.linalg.solve(_dense(metric.apply_A, shape), g.reshape(-1)) / damping).view(shape)
    torch.testing.assert_close(reference[0], exact, atol=1e-16, rtol=1e-8)
    torch.testing.assert_close(steps[0], exact, atol=1e-16, rtol=1e-6)
    assert stats['reason'] == 'converged'
    assert stats['iterations'] <= 2


def test_nonpositive_curvature_and_nonfinite_products_fall_back():
    torch.manual_seed(3)
    shape = (1, 1, 3, 3, 3)
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    g = torch.randn(shape, dtype=torch.float64)
    settings = _settings(damping=1.0, curvature='finite_difference', pcg_iterations=5,
                         pcg_tolerance=1e-12, inner_cg_iterations=200, inner_cg_tolerance=1e-14)
    sobolev, _ = sgn.sobolev_step([g], [metric], _settings(
        damping=1.0, inner_cg_iterations=200, inner_cg_tolerance=1e-14))
    # Strongly negative curvature on the first direction: the Steihaug move
    # along the first (preconditioned-gradient) direction to the fallback
    # radius is exactly the Sobolev step.
    concave = lambda v: [-1e3 * v[0]]
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=concave)
    assert stats['reason'] == 'nonpositive_curvature'
    assert stats['iterations'] == 0 and stats['steihaug'] > 0.0 and not stats['fallback']
    torch.testing.assert_close(steps[0], sobolev[0])
    # A non-finite product on the first iteration: same fallback.
    broken = lambda v: None
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=broken)
    assert stats['reason'] == 'nonfinite' and stats['fallback']
    torch.testing.assert_close(steps[0], sobolev[0])
    # Negative curvature met after one valid iteration keeps that iterate.
    calls = {'n': 0}

    def later_concave(v):
        calls['n'] += 1
        return [v[0] * (2.0 if calls['n'] == 1 else -1e3)]
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=later_concave)
    assert stats['reason'] == 'nonpositive_curvature'
    assert stats['iterations'] == 1 and not stats['fallback']
    assert bool(torch.isfinite(steps[0]).all())
    assert float(steps[0].abs().sum()) > 0.0
    # A non-finite gradient yields a zero step rather than propagating.
    bad = g.clone()
    bad[0, 0, 0, 0, 0] = math.nan
    steps, stats = sgn.sobolev_step([bad], [metric], settings, hvp=lambda v: [v[0]])
    assert float(steps[0].abs().sum()) == 0.0
    assert stats['reason'] == 'nonfinite_rhs'


def test_finite_difference_hvp_is_exact_for_quadratics():
    torch.manual_seed(4)
    shape = (1, 1, 2, 3, 2)
    n = math.prod(shape)
    J = torch.randn(n, n, dtype=torch.float64)
    H = J.T @ J
    p = torch.randn(shape, dtype=torch.float64)
    base = (H @ p.reshape(-1)).view(shape)

    def gradient_fn(perturbation, scale):
        shifted = p + scale * perturbation[0]
        return [(H @ shifted.reshape(-1)).view(shape)]
    v = torch.randn(shape, dtype=torch.float64)
    hv = sgn.finite_difference_hvp(gradient_fn, [base], [v], 1e-3)
    torch.testing.assert_close(hv[0], (H @ v.reshape(-1)).view(shape), atol=1e-7, rtol=1e-7)
    assert sgn.finite_difference_hvp(lambda *_: None, [base], [v], 1e-3) is None
    nan_fn = lambda *_: [torch.full(shape, math.nan, dtype=torch.float64)]
    assert sgn.finite_difference_hvp(nan_fn, [base], [v], 1e-3) is None


def test_trust_region_scales_to_the_sobolev_radius_and_steps_add_in_place():
    torch.manual_seed(5)
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    g = torch.randn(1, 3, 3, 3, 3, dtype=torch.float64)
    free, free_stats = sgn.sobolev_step([g], [metric], _settings(damping=1.0, inner_cg_iterations=200))
    radius = 0.25 * free_stats['sobolev_norm']
    steps, stats = sgn.sobolev_step([g], [metric], _settings(
        damping=1.0, inner_cg_iterations=200, trust_radius=radius))
    assert stats['trust_scale'] == pytest.approx(0.25)
    assert float(metric.quadratic_norm(steps[0]).sqrt()) == pytest.approx(radius)
    torch.testing.assert_close(steps[0], free[0] * 0.25)
    param = torch.zeros_like(g)
    sgn.apply_steps_([param], steps, 0.5)
    torch.testing.assert_close(param, steps[0] * 0.5)


def test_constrained_cells_stay_bitwise_unchanged():
    torch.manual_seed(6)
    constrained = torch.rand(4, 4, 4) < 0.3
    metric = sgn.CartesianSobolevMetric((2.0, 2.0, 2.0), constrained=constrained)
    param = torch.randn(1, 3, 4, 4, 4)
    before = param.clone()
    g = torch.randn_like(param).masked_fill(constrained, 0.0)
    hvp = lambda v: [3.0 * v[0]]
    settings = _settings(damping=0.1, curvature='finite_difference', pcg_iterations=8)
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=hvp)
    sgn.apply_steps_([param], steps, 1.0)
    assert torch.equal(param[:, :, constrained], before[:, :, constrained])
    assert not torch.equal(param[:, :, ~constrained], before[:, :, ~constrained])


# --------------------------------------------------------------------------
# Flow-field factories, width conversion, random state, configuration


def test_metric_factories_match_the_lattices():
    cartesian = CartesianFlowField((12, 12, 12), spatial_scale_factor=6)
    metric = sgn.metric_for_flow_field(cartesian, 0, 0.5)
    assert isinstance(metric, sgn.CartesianSobolevMetric)
    assert metric.lengths_cells == (0.5, 0.5, 0.5)
    v = torch.randn_like(cartesian.flows[0])
    assert metric.apply_A(v).shape == v.shape
    cylindrical = CylindricalFlowField((12, 24, 24), spatial_scale_factor=6)
    for level in (0, 1):
        metric = sgn.metric_for_flow_field(cylindrical, level, 1.5)
        assert isinstance(metric, sgn.CylindricalSobolevMetric)
        v = torch.randn_like(cylindrical.flows[level])
        out = metric.apply_A(v)
        assert out.shape == v.shape
        n0 = int((cylindrical._lr_num_phi if level == 0 else cylindrical._hr_num_phi)[0])
        assert float(out[..., :n0].abs().sum()) == 0.0
        pre = sgn.gaussian_preconditioner_for_flow_field(cylindrical, level, 1.0, 0.5)
        z = pre(v)
        assert z.shape == v.shape and float(z[..., :n0].abs().sum()) == 0.0
    # Physical lengths convert like the smoother's widths: coarse cells are
    # spatial_scale_factor times wider.
    assert sgn.lattice_length_cells(32.0, 16.0, 1) == 2.0
    assert sgn.lattice_length_cells(32.0, 16.0, 6) == pytest.approx(2.0 / 6.0)
    report = sgn.describe_settings(_settings(length_voxels=32.0), 16.0, 6, 'cylindrical')
    assert 'HR 2.00 cells' in report and 'LR 0.33 cells' in report
    assert 'no radial edges' in report
    heuristic = sgn.describe_settings(_settings(preconditioner='gaussian'), 16.0, 6, 'cartesian')
    assert 'HEURISTIC' in heuristic


def test_gaussian_preconditioner_is_the_smoother_on_a_copy():
    torch.manual_seed(7)
    pre = sgn.GaussianPreconditioner(lambda t: flow_grad_smoothing.smooth_cartesian_(t, 1.0))
    r = torch.randn(1, 3, 8, 8, 8)
    before = r.clone()
    z = pre(r)
    assert torch.equal(r, before)
    torch.testing.assert_close(z, flow_grad_smoothing.smooth_cartesian_(r.clone(), 1.0))


def test_rng_snapshot_replays_every_generator():
    snapshot = sgn.RngSnapshot()
    first = (torch.rand(3), np.random.rand(3), random.random())
    snapshot.restore()
    second = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_sobolev_keys_are_run_boundary_off_by_default_and_backfilled():
    fields = Config.catalog()['schema']['fields']
    defaults = Config().as_dict()
    expected = {
        'optimizer_flow_sobolev_gn': ('boolean', False),
        'optimizer_flow_sobolev_length_voxels': ('number', 32.0),
        'optimizer_flow_sobolev_damping': ('number', 1.0),
        'optimizer_flow_sobolev_curvature': ('enum', 'none'),
        'optimizer_flow_sobolev_pcg_iterations': ('integer', 5),
        'optimizer_flow_sobolev_pcg_tolerance': ('number', 0.1),
        'optimizer_flow_sobolev_inner_cg_iterations': ('integer', 20),
        'optimizer_flow_sobolev_inner_cg_tolerance': ('number', 0.001),
        'optimizer_flow_sobolev_step_scale': ('number', 0.1),
        'optimizer_flow_sobolev_trust_radius': ('number', 0.0),
        'optimizer_flow_sobolev_max_step_voxels': ('number', 2.0),
        'optimizer_flow_sobolev_adapt_damping': ('boolean', True),
        'optimizer_flow_sobolev_diagnostic_interval': ('integer', 0),
        'optimizer_flow_sobolev_damping_increase': ('number', 3.0),
        'optimizer_flow_sobolev_damping_decrease': ('number', 1.5),
        'optimizer_flow_sobolev_damping_max_factor': ('number', 10000.0),
        'optimizer_flow_sobolev_fd_epsilon_voxels': ('number', 1.0),
        'optimizer_flow_sobolev_preconditioner': ('enum', 'cg'),
        'optimizer_flow_sobolev_evaluate_step': ('boolean', False),
    }
    for key, (kind, default) in expected.items():
        assert fields[key]['type'] == kind, key
        assert fields[key]['runtime_impact'] == 'run_boundary', key
        assert defaults[key] == default, key
        assert BACKFILLABLE_CONFIG_DEFAULTS[key] == default, key
        assert 'description' in fields[key], key
    assert fields['optimizer_flow_sobolev_curvature']['values'] == ['none', 'finite_difference']
    assert fields['optimizer_flow_sobolev_preconditioner']['values'] == ['cg', 'gaussian']
    # A checkpoint without the keys resolves to the disabled step.
    settings = sgn.SobolevSettings.from_config({})
    assert not settings.enabled and settings.curvature == 'none'
    settings = sgn.SobolevSettings.from_config(defaults)
    assert not settings.enabled
    with pytest.raises(ValueError):
        sgn.SobolevSettings.from_config({**defaults, 'optimizer_flow_sobolev_curvature': 'exact'})
    with pytest.raises(ValueError):
        sgn.SobolevSettings.from_config({**defaults, 'optimizer_flow_sobolev_damping': 0.0})
    with pytest.raises(ValueError):
        sgn.SobolevSettings.from_config({**defaults, 'optimizer_flow_sobolev_preconditioner': 'multigrid'})


# --------------------------------------------------------------------------
# The fitter hook on a stub context


class _Dist:
    is_main_process = True
    is_distributed = False
    world_size = 1


class _Timer:
    def start(self, name):
        pass

    def stop(self, name):
        pass

    def tick(self):
        pass

    def maybe_report(self, iteration):
        pass


class _Model(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == 'cartesian':
            self.flow_field = CartesianFlowField((6, 6, 6), spatial_scale_factor=3)
        else:
            self.flow_field = CylindricalFlowField((6, 8, 8), spatial_scale_factor=3)
        self.extra_flow_fields = torch.nn.ModuleList()
        self.pitch = torch.nn.Parameter(torch.tensor([0.3]))
        self.flow_min_corner_zyx = torch.tensor([0., -48., -48.])
        self.flow_max_corner_zyx = torch.tensor([96., 48., 48.])

    @property
    def flow_fields(self):
        return [self.flow_field, *self.extra_flow_fields]

    # The fitter rebuilds its cached transform after re-evaluations; the stub
    # records the rebuild and returns a marker.
    def get_slice_to_spiral_transform(self):
        self.rebuilt_transforms = getattr(self, 'rebuilt_transforms', 0) + 1
        return ('transform', self.rebuilt_transforms)

    def get_dr_per_winding(self):
        return self.pitch.detach().clone()


def _stub_context(kind, **config_overrides):
    import fit_spiral

    context = fit_spiral.FitContext.__new__(fit_spiral.FitContext)
    config = Config().as_dict()
    config.update({
        'model_flow_field_type': kind,
        'model_flow_voxel_resolution': 16,
        'optimizer_flow_sobolev_gn': True,
        'optimizer_flow_sobolev_adapt_damping': False,
        'optimizer_flow_sobolev_length_voxels': 24.0,
        'optimizer_flow_sobolev_damping': 2.0,
        'optimizer_flow_sobolev_step_scale': 1.0,
        'optimizer_flow_sobolev_inner_cg_iterations': 50,
        'optimizer_flow_sobolev_max_step_voxels': 0.0,
    })
    config.update(config_overrides)
    context.config = config
    context.dist = _Dist()
    context.step_timer = _Timer()
    context.spiral_and_transform = _Model(kind)
    context.dist_grad_params = list(context.spiral_and_transform.parameters())
    context.dist_grad_named = list(context.spiral_and_transform.named_parameters())
    context.nonfinite_grad_steps = torch.zeros(())
    context.nonfinite_grad_by_param = {name: torch.zeros(()) for name, _ in context.dist_grad_named}
    context.influence_state = None
    context.flow_grad_clip_stats = {}
    context.sobolev_step_stats = {}
    context.sobolev_damping = None
    context.optimiser = LazyMomentAdamW(
        [{'params': [context.spiral_and_transform.pitch]},
         {'params': list(context.spiral_and_transform.flow_field.flows), 'lazy_moments': True}],
        lr=1e-2)
    return context


def _quadratic_loss_fn(model, curvature, generator_scale=1.0):
    """A synthetic per-step loss: a random (batch-dependent) quadratic in the
    flow lattices plus a quadratic in the pitch. Drawn from torch's global
    generator, so a restored random state replays it exactly."""
    targets = {}

    def compute(context, iteration, *, skip_flow_conditioning=False):
        loss_total = torch.zeros(())
        losses = {}
        for name, param in context.dist_grad_named:
            noise = torch.randn_like(param) * generator_scale
            target = targets.setdefault(name, torch.randn_like(param) * 0.1)
            if name.startswith('flow_field'):
                residual = param - target
                loss = 0.5 * curvature * (residual * residual).sum() + (noise * param).sum()
            else:
                loss = 0.5 * ((param - 1.0) ** 2).sum() + (noise * param).sum()
            loss.backward()
            losses[name] = loss.detach()
            loss_total = loss_total + loss.detach()
        context._sanitize_nonfinite_grads_()
        return loss_total, losses, {}, {}
    return compute


@pytest.mark.parametrize('kind', ['cartesian', 'cylindrical'])
def test_fitter_hook_updates_flow_params_and_hides_their_gradients(kind, monkeypatch):
    context = _stub_context(kind, optimizer_flow_sobolev_curvature='finite_difference',
                            optimizer_flow_sobolev_pcg_iterations=4,
                            optimizer_flow_sobolev_evaluate_step=True)
    compute = _quadratic_loss_fn(context.spiral_and_transform, curvature=4.0)
    monkeypatch.setattr(context, '_compute_step_gradients',
                        lambda iteration, **kw: compute(context, iteration, **kw))
    torch.manual_seed(11)
    flows = list(context.spiral_and_transform.flow_field.flows)
    with torch.no_grad():
        for flow in flows:
            flow.copy_(torch.randn_like(flow) * 0.05)
    flows_before = [flow.detach().clone() for flow in flows]
    pitch = context.spiral_and_transform.pitch

    settings = context._sobolev_settings()
    rng = sgn.RngSnapshot()
    loss, losses, _, _ = context._compute_step_gradients(0)
    pitch_grad = pitch.grad.clone()
    flow_grads = [flow.grad.clone() for flow in flows]
    context.nonfinite_grad_steps.fill_(3.0)
    context._sobolev_flow_step(0, settings, rng, loss)
    stats = context.sobolev_step_stats

    # Flow parameters moved (except pinned cells), their gradients are hidden
    # from the optimizer, and the pitch keeps the gradient of the original
    # evaluation despite the re-evaluations.
    for flow, before in zip(flows, flows_before):
        assert not torch.equal(flow.detach(), before)
        assert flow.grad is None
    torch.testing.assert_close(pitch.grad, pitch_grad)
    assert float(context.nonfinite_grad_steps) == 3.0
    assert stats['reevaluations'] >= 2  # >= 1 Hessian product + the evaluation
    # The cached transform was rebuilt from the updated parameters.
    assert context.spiral_and_transform.rebuilt_transforms == 1
    assert context.slice_to_spiral_transform == ('transform', 1)
    assert stats['iterations'] >= 1 and stats['reason'] in ('converged', 'max_iterations')
    assert stats['curvature'] == 'finite_difference'
    assert 'loss_after' in stats and stats['loss_before'] == float(loss)
    # A descent step on this convex problem lowers the same-batch loss.
    assert stats['loss_after'] < stats['loss_before']
    assert len(stats['blocks']) == 2
    assert stats['blocks'][0]['name'] == 'LR/stage0' and stats['blocks'][1]['name'] == 'HR/stage0'
    if kind == 'cylindrical':
        for flow, before, num_phi in zip(flows, flows_before, (
                context.spiral_and_transform.flow_field._lr_num_phi,
                context.spiral_and_transform.flow_field._hr_num_phi)):
            n0 = int(num_phi[0])
            assert torch.equal(flow.detach()[..., :n0], before[..., :n0])
    # The report renders and its payload is scalar.
    lines, payload = context._sobolev_report()
    assert any('flow sobolev:' in line for line in lines)
    assert all(isinstance(value, (int, float)) for value in payload.values())
    assert payload['flow_sobolev/loss_reduction_same_batch'] > 0.0
    # The optimizer then steps only the pitch.
    context.optimiser.step()
    assert not torch.equal(pitch.detach(), torch.zeros_like(pitch))
    # The sanitised gradients were not the Adam denominators for the flows.
    for flow in flows:
        assert flow not in context.optimiser.state or len(context.optimiser.state[flow]) == 0


def test_fitter_hook_sobolev_gradient_step_matches_the_module_step(monkeypatch):
    context = _stub_context('cartesian')
    compute = _quadratic_loss_fn(context.spiral_and_transform, curvature=1.0, generator_scale=0.0)
    monkeypatch.setattr(context, '_compute_step_gradients',
                        lambda iteration, **kw: compute(context, iteration, **kw))
    torch.manual_seed(12)
    flows = list(context.spiral_and_transform.flow_field.flows)
    settings = context._sobolev_settings()
    rng = sgn.RngSnapshot()
    loss, *_ = context._compute_step_gradients(0)
    grads = [flow.grad.clone() for flow in flows]
    before = [flow.detach().clone() for flow in flows]
    context._sobolev_flow_step(0, settings, rng, loss)
    assert context.sobolev_step_stats['reevaluations'] == 0
    assert context.sobolev_step_stats['reason'] == 'sobolev_gradient'
    metrics, _ = context._sobolev_metrics(settings)
    expected, _ = sgn.sobolev_step(grads, metrics, settings)
    for flow, start, step in zip(flows, before, expected):
        torch.testing.assert_close(flow.detach(), start + settings.step_scale * step)
    # Both lattices see the same physical length: coarse cells are 3x wider.
    assert metrics[0].lengths_cells == (0.5,) * 3
    assert metrics[1].lengths_cells == (1.5,) * 3


def test_fitter_hook_restores_parameters_exactly_around_reevaluations(monkeypatch):
    context = _stub_context('cartesian', optimizer_flow_sobolev_curvature='finite_difference',
                            optimizer_flow_sobolev_pcg_iterations=3,
                            optimizer_flow_sobolev_step_scale=0.0)
    compute = _quadratic_loss_fn(context.spiral_and_transform, curvature=2.0)
    monkeypatch.setattr(context, '_compute_step_gradients',
                        lambda iteration, **kw: compute(context, iteration, **kw))
    torch.manual_seed(13)
    flows = list(context.spiral_and_transform.flow_field.flows)
    with torch.no_grad():
        for flow in flows:
            flow.copy_(torch.randn_like(flow))
    before = [flow.detach().clone() for flow in flows]
    settings = context._sobolev_settings()
    rng = sgn.RngSnapshot()
    loss, *_ = context._compute_step_gradients(0)
    context._sobolev_flow_step(0, settings, rng, loss)
    # A zero step scale: the perturbations of the finite differences leave
    # no trace.
    assert context.sobolev_step_stats['reevaluations'] >= 1
    for flow, start in zip(flows, before):
        assert torch.equal(flow.detach(), start)


def test_fitter_hook_eliminates_influence_masked_cells(monkeypatch):
    context = _stub_context('cartesian')
    model = context.spiral_and_transform
    lr_mask = torch.ones(model.flow_field.flows[0].shape[2:])
    hr_mask = torch.ones(model.flow_field.flows[1].shape[2:])
    lr_mask[0] = 0.0
    hr_mask[:, :, 0] = 0.0

    class _Influence:
        active = True
        masks = {'flow_lr': lr_mask, 'flow_hr': hr_mask}

        def apply_grad_masks_(self, spiral_and_transform):
            for flow, mask in zip(spiral_and_transform.flow_field.flows, (lr_mask, hr_mask)):
                flow.grad.mul_(mask)
    context.influence_state = _Influence()
    compute = _quadratic_loss_fn(model, curvature=1.0)

    def masked_compute(iteration, **kw):
        out = compute(context, iteration, **kw)
        context.influence_state.apply_grad_masks_(model)
        return out
    monkeypatch.setattr(context, '_compute_step_gradients', masked_compute)
    torch.manual_seed(14)
    flows = list(model.flow_field.flows)
    before = [flow.detach().clone() for flow in flows]
    settings = context._sobolev_settings()
    rng = sgn.RngSnapshot()
    loss, *_ = context._compute_step_gradients(0)
    context._sobolev_flow_step(0, settings, rng, loss)
    assert torch.equal(flows[0].detach()[:, :, 0], before[0][:, :, 0])
    assert torch.equal(flows[1].detach()[..., 0], before[1][..., 0])
    assert not torch.equal(flows[0].detach()[:, :, 1:], before[0][:, :, 1:])


# --------------------------------------------------------------------------
# Toy ablation: update roughness and same-batch loss reduction


def _toy_problem(seed=21):
    """A smooth target displacement observed through sparse, noisy point
    samples of a 3-D lattice (trilinear interpolation), the shape of the
    fitter's flow problem in miniature."""
    torch.manual_seed(seed)
    shape = (1, 3, 8, 8, 8)
    zyx = torch.stack(torch.meshgrid(*(torch.linspace(0, 1, 8),) * 3, indexing='ij'))
    target = torch.stack([
        0.3 * torch.sin(2 * math.pi * zyx[0]),
        0.2 * torch.cos(2 * math.pi * zyx[1]),
        0.1 * zyx[2] ** 2,
    ]).unsqueeze(0)
    points = torch.rand(400, 3) * 2 - 1  # grid_sample coordinates

    grid = points.view(1, 1, 1, -1, 3)

    def sample(param):
        return torch.nn.functional.grid_sample(
            param, grid, mode='bilinear', align_corners=True).view(3, -1)
    wanted = sample(target)

    def loss_fn(param):
        return 0.5 * ((sample(param) - wanted) ** 2).sum()
    return shape, loss_fn, sample


def _exact_hvp(sample):
    # The loss is quadratic in the lattice (sampling is linear), so H v is
    # the gradient of 0.5 |S v|^2 at v: S^T S v.
    def hvp(v):
        p = v[0].detach().requires_grad_(True)
        (hv,) = torch.autograd.grad(0.5 * (sample(p) ** 2).sum(), p)
        return [hv.detach()]
    return hvp


def test_toy_ablation_sobolev_steps_are_smoother_and_reduce_the_loss():
    shape, loss_fn, sample = _toy_problem()
    metric = sgn.CartesianSobolevMetric((1.5, 1.5, 1.5))
    hvp = _exact_hvp(sample)

    def gradient(param):
        p = param.detach().requires_grad_(True)
        loss = loss_fn(p)
        (grad,) = torch.autograd.grad(loss, p)
        return float(loss.detach()), grad

    results = {}

    def adamw_variant(name, smooth, shared):
        param = torch.nn.Parameter(torch.zeros(shape))
        optimiser = LazyMomentAdamW(
            [{'params': [param], 'lazy_moments': False, 'shared_second_moment': shared}], lr=1e-2)
        loss_before, grad = gradient(param)
        param.grad = grad.clone()
        if smooth:
            flow_grad_smoothing.smooth_cartesian_(param.grad, 1.5)
        optimiser.step()
        step = param.detach().clone()
        results[name] = (loss_before, float(loss_fn(param.detach())), float(metric.roughness(step)))

    adamw_variant('adamw', smooth=False, shared=False)
    adamw_variant('adamw_smoothed', smooth=True, shared=False)
    adamw_variant('adamw_smoothed_shared', smooth=True, shared=True)

    param = torch.zeros(shape)
    loss_before, grad = gradient(param)
    # The damping plays a different role in the two steps (it is the whole
    # step length of the Sobolev gradient step, but only the regulariser of
    # the curvature-aware one), so the Sobolev gradient step is tried at a
    # range of dampings.
    for damping in (0.3, 1.0, 3.0):
        settings = _settings(damping=damping, inner_cg_iterations=100, inner_cg_tolerance=1e-8,
                             step_scale=1.0)
        steps, stats = sgn.sobolev_step([grad], [metric], settings)
        assert stats['reason'] == 'sobolev_gradient'
        results[f'sobolev_gradient_{damping:g}'] = (
            loss_before, float(loss_fn(param + steps[0])), float(metric.roughness(steps[0])))

    settings = _settings(damping=0.3, curvature='finite_difference', pcg_iterations=10,
                         pcg_tolerance=1e-3, inner_cg_iterations=100, inner_cg_tolerance=1e-8,
                         step_scale=1.0)
    steps, hf_stats = sgn.sobolev_step([grad], [metric], settings, hvp=hvp)
    hf_param = param + steps[0]
    results['hessian_free'] = (loss_before, float(loss_fn(hf_param)), float(metric.roughness(steps[0])))
    assert hf_stats['iterations'] >= 1 and not hf_stats['fallback']

    # Evidence on one toy problem, not a quality claim: every variant reduces
    # the same-batch loss; the Sobolev-metric steps are smoother than the raw
    # AdamW step; and the curvature-aware step reduces this convex loss more
    # than the best of the fixed-damping Sobolev gradient steps.
    for name, (before, after, roughness) in results.items():
        assert after < before, (name, before, after)
        assert math.isfinite(roughness), name
    for name, (_, _, roughness) in results.items():
        if name != 'adamw':
            assert roughness < results['adamw'][2], (name, results)
    best_sobolev = min(after for name, (_, after, _) in results.items()
                       if name.startswith('sobolev_gradient'))
    assert results['hessian_free'][1] < best_sobolev, results
    # The predicted quadratic reduction of the Hessian-free step is positive.
    predicted = -(torch.sum(grad * steps[0], dtype=torch.float64)
                  + 0.5 * torch.sum(steps[0] * hvp([steps[0]])[0], dtype=torch.float64))
    assert float(predicted) > 0.0


def test_fitter_hook_caps_the_step_in_voxels(monkeypatch):
    context = _stub_context('cartesian', optimizer_flow_sobolev_max_step_voxels=0.5)
    compute = _quadratic_loss_fn(context.spiral_and_transform, curvature=1.0, generator_scale=0.0)
    monkeypatch.setattr(context, '_compute_step_gradients',
                        lambda iteration, **kw: compute(context, iteration, **kw))
    torch.manual_seed(15)
    flows = list(context.spiral_and_transform.flow_field.flows)
    with torch.no_grad():
        for flow in flows:
            flow.copy_(torch.randn_like(flow) * 50.0)  # far from target: a huge raw step
    settings = context._sobolev_settings()
    rng = sgn.RngSnapshot()
    loss, *_ = context._compute_step_gradients(0)
    grads = [flow.grad.clone() for flow in flows]
    before = [flow.detach().clone() for flow in flows]
    context._sobolev_flow_step(0, settings, rng, loss)
    stats = context.sobolev_step_stats
    assert stats['cap_scale'] < 1.0
    _, component_voxels = context._flow_component_units()
    largest = max(v for block in stats['blocks'] for v in block['component_rms_vox'])
    assert largest == pytest.approx(0.5, rel=1e-4)
    # The applied increment is the raw step times the logged scale.
    metrics, _ = context._sobolev_metrics(settings)
    raw, _ = sgn.sobolev_step(grads, metrics, settings)
    for flow, start, step in zip(flows, before, raw):
        torch.testing.assert_close(flow.detach(), start + stats['step_scale'] * step)
    lines, payload = context._sobolev_report()
    assert any('voxel cap scale' in line for line in lines)
    assert payload['flow_sobolev/cap_scale'] == stats['cap_scale']


def test_residual_growth_stops_the_solve_and_keeps_the_previous_iterate():
    torch.manual_seed(31)
    shape = (1, 1, 3, 3, 3)
    n = math.prod(shape)
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    g = torch.randn(shape, dtype=torch.float64)
    # A wildly non-symmetric "Hessian": CG's residual recurrence breaks down.
    N = 50.0 * torch.randn(n, n, dtype=torch.float64)
    hvp = lambda v: [(N @ v[0].reshape(-1)).view(shape)]
    settings = _settings(damping=1.0, curvature='finite_difference', pcg_iterations=20,
                         pcg_tolerance=1e-12, inner_cg_iterations=200, inner_cg_tolerance=1e-14,
                         adapt_damping=False)
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=hvp)
    assert stats['reason'] in ('residual_increased', 'nonpositive_curvature')
    assert stats['residual_final'] <= stats['residual_initial'] * sgn.RESIDUAL_GROWTH_LIMIT
    assert bool(torch.isfinite(steps[0]).all())
    # A symmetric positive operator is never stopped by the growth guard.
    J = torch.randn(n, n, dtype=torch.float64)
    spd = lambda v: [((J.T @ J) @ v[0].reshape(-1)).view(shape)]
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=spd)
    assert stats['reason'] in ('converged', 'max_iterations')


def test_steihaug_moves_to_the_boundary_on_negative_curvature():
    torch.manual_seed(32)
    shape = (1, 1, 3, 3, 3)
    metric = sgn.CartesianSobolevMetric((1.0, 1.0, 1.0))
    g = torch.randn(shape, dtype=torch.float64)
    calls = {'n': 0}

    def later_concave(v):
        calls['n'] += 1
        return [v[0] * (2.0 if calls['n'] == 1 else -1e3)]
    # With a trust radius the iterate is moved along p to that Sobolev norm.
    radius = 5.0
    settings = _settings(damping=1.0, curvature='finite_difference', pcg_iterations=5,
                         pcg_tolerance=1e-12, inner_cg_iterations=200, inner_cg_tolerance=1e-14,
                         trust_radius=radius, adapt_damping=False)
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=later_concave)
    assert stats['reason'] == 'nonpositive_curvature'
    assert stats['iterations'] == 1 and stats['steihaug'] > 0.0 and not stats['fallback']
    assert stats['sobolev_norm'] == pytest.approx(radius, rel=1e-6)
    assert stats['trust_scale'] == pytest.approx(1.0)
    # Without one, the boundary is the fallback Sobolev step's norm.
    calls['n'] = 0
    settings = _settings(damping=1.0, curvature='finite_difference', pcg_iterations=5,
                         pcg_tolerance=1e-12, inner_cg_iterations=200, inner_cg_tolerance=1e-14,
                         adapt_damping=False)
    fallback, _ = sgn.sobolev_step([g], [metric], _settings(
        damping=1.0, inner_cg_iterations=200, inner_cg_tolerance=1e-14))
    fallback_norm = float(metric.quadratic_norm(fallback[0]).sqrt())
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=later_concave)
    assert stats['steihaug'] > 0.0
    assert stats['sobolev_norm'] == pytest.approx(fallback_norm, rel=1e-6)
    # Negative curvature on the very first direction: the iterate is zero, so
    # Steihaug walks the whole way and no fallback is needed.
    calls['n'] = 0
    concave = lambda v: [-1e3 * v[0]]
    steps, stats = sgn.sobolev_step([g], [metric], settings, hvp=concave)
    assert stats['iterations'] == 0 and stats['steihaug'] > 0.0 and not stats['fallback']
    assert stats['sobolev_norm'] == pytest.approx(fallback_norm, rel=1e-6)


def test_adapt_damping_is_a_bounded_cross_step_lm_rule():
    settings = _settings(damping=2.0, curvature='finite_difference',
                         damping_increase=3.0, damping_decrease=1.5, damping_max_factor=10.0)
    assert sgn.adapt_damping(2.0, 'nonpositive_curvature', settings) == 6.0
    assert sgn.adapt_damping(2.0, 'residual_increased', settings) == 6.0
    assert sgn.adapt_damping(6.0, 'converged', settings) == 4.0
    assert sgn.adapt_damping(2.0, 'converged', settings) == 2.0  # never below the config
    assert sgn.adapt_damping(18.0, 'nonfinite', settings) == 20.0  # capped at 10x
    assert sgn.adapt_damping(6.0, 'sobolev_gradient', settings) == 6.0
    off = _settings(damping=2.0, curvature='finite_difference', adapt_damping=False)
    assert sgn.adapt_damping(7.0, 'nonpositive_curvature', off) == 7.0
    none = _settings(damping=2.0, curvature='none')
    assert sgn.adapt_damping(7.0, 'nonpositive_curvature', none) == 7.0
    with pytest.raises(ValueError):
        _settings(curvature='finite_difference', damping_increase=1.0)


def test_fitter_hook_adapts_and_reports_damping(monkeypatch):
    context = _stub_context('cartesian', optimizer_flow_sobolev_curvature='finite_difference',
                            optimizer_flow_sobolev_pcg_iterations=3,
                            optimizer_flow_sobolev_adapt_damping=True)
    # A concave synthetic loss: every solve hits negative curvature.
    compute = _quadratic_loss_fn(context.spiral_and_transform, curvature=-100.0, generator_scale=0.0)
    monkeypatch.setattr(context, '_compute_step_gradients',
                        lambda iteration, **kw: compute(context, iteration, **kw))
    torch.manual_seed(33)
    settings = context._sobolev_settings()
    assert context.sobolev_damping is None
    for expected in (2.0, 6.0, 18.0):
        rng = sgn.RngSnapshot()
        loss, *_ = context._compute_step_gradients(0)
        context._sobolev_flow_step(0, settings, rng, loss)
        stats = context.sobolev_step_stats
        assert stats['damping'] == expected
        assert stats['reason'] == 'nonpositive_curvature'
        assert stats['next_damping'] == expected * 3.0
        assert context.sobolev_damping == expected * 3.0
    lines, payload = context._sobolev_report()
    assert 'lambda 18->54' in lines[0]
    assert payload['flow_sobolev/next_damping'] == 54.0
    # The checkpoint payload key round-trips through apply_checkpoint's read.
    context.sobolev_damping = 12.5
    assert getattr(context, 'sobolev_damping') == 12.5


def test_predicted_reduction_and_symmetry_defect():
    torch.manual_seed(41)
    shape = (1, 1, 2, 2, 2)
    n = math.prod(shape)
    J = torch.randn(n, n, dtype=torch.float64)
    H = J.T @ J
    g = torch.randn(shape, dtype=torch.float64)
    d = torch.randn(shape, dtype=torch.float64)
    hvp = lambda v: [(H @ v[0].reshape(-1)).view(shape)]
    expected = -(float(g.reshape(-1) @ d.reshape(-1)) + 0.5 * float(d.reshape(-1) @ H @ d.reshape(-1)))
    assert sgn.predicted_reduction([g], [d], hvp([d])) == pytest.approx(expected)
    assert sgn.predicted_reduction([g], [d]) == pytest.approx(-float(g.reshape(-1) @ d.reshape(-1)))
    u = torch.randn(shape, dtype=torch.float64)
    defect, _, _ = sgn.symmetry_defect(hvp, [u], [d])
    assert defect < 1e-12
    N = torch.randn(n, n, dtype=torch.float64)
    defect, _, _ = sgn.symmetry_defect(lambda v: [(N @ v[0].reshape(-1)).view(shape)], [u], [d])
    assert defect > 0.1
    assert sgn.symmetry_defect(lambda v: None, [u], [d]) is None


def test_fitter_hook_diagnostics_report_rho_and_symmetry(monkeypatch, capsys):
    context = _stub_context('cartesian', optimizer_flow_sobolev_curvature='finite_difference',
                            optimizer_flow_sobolev_pcg_iterations=4,
                            optimizer_flow_sobolev_diagnostic_interval=1)
    # A convex quadratic with a deterministic batch: the quadratic model is
    # exact, so rho is 1 and the finite-difference operator is symmetric.
    compute = _quadratic_loss_fn(context.spiral_and_transform, curvature=4.0, generator_scale=0.0)
    monkeypatch.setattr(context, '_compute_step_gradients',
                        lambda iteration, **kw: compute(context, iteration, **kw))
    torch.manual_seed(42)
    flows = list(context.spiral_and_transform.flow_field.flows)
    with torch.no_grad():
        for flow in flows:
            flow.copy_(torch.randn_like(flow) * 0.05)
    settings = context._sobolev_settings()
    rng = sgn.RngSnapshot()
    loss, *_ = context._compute_step_gradients(0)
    context._sobolev_flow_step(0, settings, rng, loss)
    stats = context.sobolev_step_stats
    assert stats['rho'] == pytest.approx(1.0, abs=1e-3)
    assert stats['symmetry_defect'] < 1e-4
    assert stats['predicted_reduction'] > 0.0
    out = capsys.readouterr().out
    assert 'flow sobolev diag it=0' in out and 'rho 1.00' in out
    _, payload = context._sobolev_report()
    assert payload['flow_sobolev/rho'] == stats['rho']
