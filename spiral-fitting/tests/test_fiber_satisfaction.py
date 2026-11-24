"""The whole-set fiber satisfaction pass (fiber_satisfaction) on a small
synthetic solve under the identity transform: point, fiber and component
verdicts, the largest component's exact health statistics, the per-fiber
file, the vertical-fiber offset, and the no-op when the fiber loss is off."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

DR = 10.0
C0 = 3
# Per fiber: (component index, vertical, [(s, rho), ...]); every point's
# sheet label is 1. Component 0 (three fibers) is the largest.
FIBERS = [
    (0, False, [(3.1, 0.0), (2.8, 0.1), (3.4, -0.1)]),   # A: satisfied
    (0, False, [(3.0, 0.0), (3.0, 0.45)]),               # B: one point rejected
    (0, False, [(3.0, 0.0), (4.0, 0.0)]),                # C: one point a turn off
    (1, False, [(7.2, 0.0), (6.9, 0.0)]),                # D: component 1, target 7
    (1, False, [(7.1, 0.0), (7.0, 0.0)]),                # E
    (2, False, [(0.1, 0.0), (0.0, 0.0), (1.2, 0.0)]),    # F: component 2, target 0
]


def identity(points):
    return points


def synthetic_solve(fibers=FIBERS, on_axis=False, dr=DR):
    """Points at theta = pi/2 in the spiral frame (= the scroll frame), at
    radius (s + 1 + 1/4) dr, with w_s = 1 + 1/4 + rho: label 1, phase rho.
    ``on_axis`` puts them at theta = 0 (radius (s + 1) dr, w_s = 1 + rho)
    instead, where s is exact in float32 for dyadic s and a power-of-two dr."""
    quarter = 0.0 if on_axis else 0.25
    zyx, w_s, point_fiber = [], [], []
    for index, (_, _, points) in enumerate(fibers):
        for s, rho in points:
            radius = (s + 1 + quarter) * dr
            zyx.append((float(len(zyx)), 0.0, radius) if on_axis
                       else (float(len(zyx)), radius, 0.0))
            w_s.append(1 + quarter + rho)
            point_fiber.append(index)
    count = len(zyx)
    fiber_cc = np.array([cc for cc, _, _ in fibers], dtype=np.int64)
    uuids = np.zeros((len(fibers), 16), dtype=np.uint8)
    uuids[:, 0] = np.arange(len(fibers)) + 1
    return SimpleNamespace(
        c=0.0, w_min=np.zeros(len(fibers)), w_max=np.zeros(len(fibers)),
        fiber_cc=fiber_cc, largest_cc=0, point_fiber=np.array(point_fiber, dtype=np.int64),
        piece_offsets=np.concatenate([[0], np.cumsum([len(p) for _, _, p in fibers])]),
        zyx=np.array(zyx, dtype=np.float32), w_s=np.array(w_s),
        family=np.array([int(v) for _, v, _ in fibers], dtype=np.int8),
        fiber_uuid=uuids, cc_component=np.array([10, 20, 30]), run_uuid='run-1',
        n0=np.ones(count, dtype=np.int64))


def fiber_loss(solve, **overrides):
    from config import Config
    from fiber_cc_loss import FiberCCLoss
    config = Config({'fiber_largest_cc_min_winding': C0, **overrides}).as_dict()
    loss = FiberCCLoss(solve, config, torch.device('cpu'), log=lambda line: None)
    assert loss.c0 == C0
    return loss


def evaluate(loss, dr=DR):
    from fiber_satisfaction import evaluate_fiber_satisfaction
    return evaluate_fiber_satisfaction(loss, identity, torch.tensor(dr))


def test_point_fiber_and_component_verdicts(tmp_path):
    from fiber_satisfaction import fiber_satisfaction_lines, write_fiber_satisfaction
    loss = fiber_loss(synthetic_solve())
    rng = torch.get_rng_state()
    result = evaluate(loss)
    assert torch.equal(torch.get_rng_state(), rng)
    fibers, summary = result.fibers, result.summary

    np.testing.assert_array_equal(fibers['component'], [10, 10, 10, 20, 20, 30])
    np.testing.assert_array_equal(fibers['largest'], [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(fibers['point_count'], [3, 2, 2, 2, 2, 3])
    np.testing.assert_array_equal(fibers['accepted_count'], [3, 1, 2, 2, 2, 3])
    np.testing.assert_array_equal(fibers['satisfied_count'], [3, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(fibers['satisfied'], [1, 0, 0, 1, 1, 0])
    np.testing.assert_allclose(fibers['satisfied_fraction'], [1, .5, .5, 1, 1, 2 / 3])
    # The other components' targets: the rounded medians, 7 and 0 (either
    # median of component 1's four points rounds to 7; see
    # test_half_a_turn_is_satisfied_and_the_median_is_the_lower_one).
    np.testing.assert_array_equal(fibers['target'], [3, 3, 3, 7, 7, 0])
    np.testing.assert_allclose(
        fibers['mean_offset'], [0.1, 0.0, 0.5, 0.05, 0.05, 1.3 / 3], atol=1e-5)

    assert summary['satisfied_fibers'] == 3 and summary['total_fibers'] == 6
    assert summary['satisfied_fibers_fraction'] == pytest.approx(0.5)
    assert summary['satisfied_fibers_point_weighted_fraction'] == pytest.approx(7 / 14)
    assert summary['satisfied_fiber_points'] == 11
    assert summary['satisfied_fiber_points_fraction'] == pytest.approx(11 / 14)
    assert (summary['satisfied_fibers_largest_cc'], summary['total_fibers_largest_cc']) == (1, 3)
    assert summary['satisfied_fibers_largest_cc_point_weighted_fraction'] == pytest.approx(3 / 7)
    assert (summary['satisfied_fibers_other_ccs'], summary['total_fibers_other_ccs']) == (2, 3)
    assert (summary['satisfied_fiber_other_ccs'], summary['total_fiber_other_ccs']) == (1, 2)
    assert summary['fiber_other_ccs_with_target'] == 2
    assert summary['fiber_other_ccs_nonzero_target'] == 1
    # The largest component's exact health statistics: accepted s - c0 is
    # [.1, -.2, .4, 0, 0, 1].
    assert summary['fiber_rejected_points'] == 1
    assert summary['fiber_rejected_fraction'] == pytest.approx(1 / 14)
    assert summary['fiber_largest_cc_rejected_fraction'] == pytest.approx(1 / 7)
    assert summary['fiber_largest_cc_mean_abs_offset'] == pytest.approx(1.7 / 6, abs=1e-5)
    assert summary['fiber_largest_cc_median_offset'] == pytest.approx(0.0, abs=1e-5)
    assert summary['fiber_largest_cc_off_fraction'] == pytest.approx(1 / 6)
    assert 'fiber_cached_targets' not in summary

    lines = fiber_satisfaction_lines(result)
    assert lines[0].startswith('satisfied_fibers = 3/6 (50.0%')
    path = write_fiber_satisfaction(result, str(tmp_path), 'fitted')
    assert path.endswith('fiber_satisfied_fitted.npz')
    with np.load(path) as saved:
        assert str(saved['run_uuid']) == 'run-1' and int(saved['c0']) == C0
        np.testing.assert_array_equal(saved['fiber_uuid'][:, 0], np.arange(6) + 1)
        np.testing.assert_array_equal(saved['satisfied'], fibers['satisfied'])


def test_cached_dt_targets_are_compared():
    loss = fiber_loss(synthetic_solve())
    loss.refresh_targets(identity, torch.tensor(DR), iteration=7)
    summary = evaluate(loss).summary
    assert (summary['fiber_cached_targets'], summary['fiber_cached_targets_differing']) == (2, 0)
    loss.targets[1] = 6.0
    summary = evaluate(loss).summary
    assert summary['fiber_cached_targets_iteration'] == 7
    assert (summary['fiber_cached_targets'], summary['fiber_cached_targets_differing']) == (2, 1)


def test_vertical_fibers_are_measured_at_the_loss_offset():
    """The loss's own residual: fiber A vertical with a 5-voxel (half-turn)
    outward offset moves its s by -0.5 under the identity transform."""
    fibers = [(cc, index == 0, points) for index, (cc, _, points) in enumerate(FIBERS)]
    loss = fiber_loss(synthetic_solve(fibers), fiber_vertical_radial_offset_enabled=True,
                      fiber_vertical_radial_offset_voxels=0.5 * DR)
    assert loss.has_offsets
    result = evaluate(loss)
    assert result.fibers['mean_offset'][0] == pytest.approx(0.1 - 0.5, abs=1e-4)
    # 2.3 is now 0.7 below c0.
    assert result.fibers['satisfied_count'][0] == 2 and not result.fibers['satisfied'][0]


def test_half_a_turn_is_satisfied_and_the_median_is_the_lower_one():
    """Exact float32 values at theta = 0 with dr = 8: a point exactly half a
    turn from its target is satisfied (<=), one 5/8 off is not; and a
    component with two accepted points at s = 1/4 and 3/4 gets the lower
    median's target, round(1/4) = 0 (the upper one would round to 1)."""
    dr = 8.0
    fibers = [
        (0, False, [(3.5, 0.0)]),     # c0 + 1/2: satisfied
        (0, False, [(2.5, 0.0)]),     # c0 - 1/2: satisfied
        (0, False, [(3.625, 0.0)]),   # c0 + 5/8: not
        (1, False, [(0.25, 0.0)]),
        (1, False, [(0.75, 0.0)]),
    ]
    loss = fiber_loss(synthetic_solve(fibers, on_axis=True, dr=dr))
    s, ok = loss.evaluate_points(identity, torch.tensor(dr), torch.arange(loss.num_points))
    assert s.tolist() == [3.5, 2.5, 3.625, 0.25, 0.75] and bool(ok.all())
    result = evaluate(loss, dr)
    np.testing.assert_array_equal(result.fibers['target'], [3, 3, 3, 0, 0])
    np.testing.assert_array_equal(result.fibers['satisfied'], [1, 1, 0, 1, 0])
    # The DT targets agree (the same helper).
    loss.refresh_targets(identity, torch.tensor(dr), iteration=0)
    assert float(loss.targets[1]) == 0.0


@pytest.mark.parametrize('case', ['absent', 'off', 'outside', 'dropped'])
def test_no_op_when_the_fiber_loss_is_off(case, tmp_path, capsys):
    import fit_spiral
    solve = synthetic_solve()
    if case == 'outside':
        solve.c = None
    stub = SimpleNamespace(
        fiber_solve=None if case == 'absent' else solve,
        fiber_cc_loss=fiber_loss(solve) if case == 'off' else None,
        config={'loss_weight_fiber': 0.0 if case == 'off' else 5.0},
        out_path=str(tmp_path), device=torch.device('cpu'))
    stub._fiber_loss_active = lambda: fit_spiral.FitContext._fiber_loss_active(stub)
    assert fit_spiral.FitContext._fiber_satisfaction(stub, 'fitted') is None
    assert list(tmp_path.iterdir()) == []
    printed = capsys.readouterr().out.splitlines()
    reason = {
        'off': 'the fiber loss is off: loss_weight_fiber is 0',
        'outside': 'no fiber point lies in the z window',
        'dropped': 'every fiber point in the z window belongs to a dropped fiber',
    }
    assert printed == ([] if case == 'absent' else
                       [f'fiber satisfaction: skipped ({reason[case]})'])


def test_out_of_memory_skips_the_fiber_outputs(tmp_path, capsys, monkeypatch):
    import fit_spiral
    import fiber_satisfaction

    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError('test')
    monkeypatch.setattr(fiber_satisfaction, 'evaluate_fiber_satisfaction', out_of_memory)
    solve = synthetic_solve()
    stub = SimpleNamespace(
        fiber_solve=solve, fiber_cc_loss=fiber_loss(solve),
        config={'loss_weight_fiber': 5.0}, out_path=str(tmp_path),
        device=torch.device('cpu'),
        spiral_and_transform=SimpleNamespace(
            get_slice_to_spiral_transform=lambda: identity,
            get_dr_per_winding=lambda: torch.tensor(DR)))
    stub._fiber_loss_active = lambda: fit_spiral.FitContext._fiber_loss_active(stub)
    assert fit_spiral.FitContext._fiber_satisfaction(stub, 'fitted') is None
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out.splitlines() == [
        'WARNING: skipped the fiber satisfaction metrics (CUDA OOM during fiber evaluation)']
