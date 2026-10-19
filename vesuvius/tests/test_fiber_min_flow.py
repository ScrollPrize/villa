import itertools
import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import (
    Seam, corrected_targets, integer_targets, solve)


def test_certificate_matches_bruteforce():
    rng = np.random.default_rng(31)
    for _ in range(20):
        a = np.array([0, 1, 0, 2, 1])
        b = np.array([1, 2, 2, 0, 0])
        d = rng.integers(-2, 3, len(a))
        w = rng.choice([.25, 1., 1.25], len(a))
        hinge = rng.random(len(a)) < .3
        out = solve(3, a, b, d, w, hinge)
        def cost(k):
            r = d-(k[b]-k[a])
            return np.sum(w*np.where(hinge, np.maximum(r, 0), abs(r)))
        brute = min(cost(np.array([0, *v])) for v in itertools.product(range(-6, 7), repeat=2))
        assert out['certificate']['primal'] == pytest.approx(brute)
        assert out['certificate']['gap'] == pytest.approx(0)


def test_consistent_chain_and_contradiction():
    assert solve(4, [0, 1, 2], [1, 2, 3], [0, 1, 2])['certificate']['primal'] == 0
    r = solve(4, [0, 1, 2, 3], [1, 2, 3, 0], [0, 1, 2, -2])
    assert r['certificate']['primal'] == 1
    reverse = solve(4, [1, 2, 3, 0], [0, 1, 2, 3], [0, -1, -2, 2])
    assert reverse['certificate']['primal'] == 1


def test_components_isolates_and_self_edges():
    r = solve(5, [0, 2, 0], [1, 3, 0], [2, -3, 4])
    assert r['certificate']['components'] == 3
    assert r['certificate']['primal'] == 4
    assert r['winding'][4] == 0


def test_targets_and_unsigned_evidence():
    target, valid = integer_targets([.5, .5, .1, .8], [1, 1, 0, 0], [1, -1, 0, 0], [False, False, True, True])
    np.testing.assert_array_equal(target, [0, 1, 0, 1])
    np.testing.assert_array_equal(valid, [True, True, True, False])
    rng = np.random.default_rng(5)
    d = rng.uniform(0, 5, 100)
    s = rng.choice([-1, 1], 100)
    f = rng.choice([-1, 0, 1], 100)
    p = rng.choice([True, False], 100)
    np.testing.assert_array_equal(integer_targets(d, s, f, p)[0], -integer_targets(d, -s, -f, p)[0])


def test_seam_transport_both_paths_and_endpoint_convention():
    seam = Seam([[0, 0, -2], [0, 0, 2]])
    a, b, cut = [-1, -2, 0], [1, -2, 0], [0, -2, 0]
    assert seam.segment(a, b) == 1
    assert seam.segment(b, a) == -1
    assert seam.segment(a, cut)+seam.segment(cut, b) == 1
    assert seam.segment(b, cut)+seam.segment(cut, a) == -1
    assert corrected_targets(0, 0, 1, 0) == -1  # fiber-only
    assert corrected_targets(0, 0, 0, 1) == 1   # connector-only
    assert corrected_targets(0, 0, 1, 1) == 0   # cancellation
    angles = np.linspace(0, 6*np.pi, 101)
    points = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(len(angles))])
    assert seam.polyline(points)[-1] == 3
    with pytest.raises(ValueError, match='umbilicus'):
        seam.segment([-1, 0, 0], [1, 0, 0])


def test_moving_seam_changes_gauge_not_loop_objective():
    angles = np.linspace(0, 4.7*np.pi, 47)
    points = np.column_stack([3*np.cos(angles), 3*np.sin(angles), np.zeros(len(angles))])
    costs = []
    for angle in [-np.pi/2, .33, 2.2]:
        seam = Seam([[0, 0, -1], [0, 0, 1]], angle)
        starts = [points[:17], points[17:34], points[34:]]
        transport = [seam.polyline(p)[-1] for p in starts]
        target = []
        for i in range(3):
            j = (i+1)%3
            target.append(corrected_targets(0, transport[i], 0, seam.segment(starts[i][-1], starts[j][0])))
        costs.append(solve(3, [0, 1, 2], [1, 2, 0], target)['certificate']['primal'])
    assert len(set(costs)) == 1


def test_bent_axis_segment_is_split_at_knots():
    # x relative to axis changes + -> - -> + although endpoints coincide in XY.
    seam = Seam([[0, 0, 0], [2, 0, 1], [0, 0, 2]])
    assert seam.segment([1, -1, 0], [1, -1, 2]) == 0
    assert seam.segment([1, -1, 0], [1, -1, 1]) == -1
    assert seam.segment([1, -1, 1], [1, -1, 2]) == 1


def test_adapter_fiber_connector_and_signed_hinge():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import graph
    class Paths:
        seam = Seam([[0, 0, -1], [0, 0, 1]])
        def ends(self, side):
            return (np.array([[-1, -2, 0]]), np.array([2])) if side == 'a' else (np.array([[1, -2, 0]]), np.array([3]))
    d = dict(family=np.array([0, 1]), i=np.array([0]), j=np.array([1]),
             distance=np.array([.5]), sign=np.array([1]), parallel=np.array([False]))
    g, _ = graph({}, d, Paths())
    assert g['target'][0] == 0  # +1 connector cancels +1 endpoint difference
    d['sign'][:] = -1
    g, _ = graph({}, d, Paths(), sign_only=True)
    assert g['a'][0] == 1 and g['b'][0] == 0
    assert g['target'][0] == 1  # outward V -> H requires next winding
    assert g['one_sided'][0]


def test_interpolated_angle_is_not_physical_seam_intersection():
    seam = Seam([[0, 0, -1], [0, 0, 1]])
    a, b = np.array([-10., -1., 0]), np.array([1., -1., 0])
    # Unequal radii: interpolated angle crosses first, physical segment later.
    p = .7*b+.3*a
    angle_interp = .7*np.arctan2(b[1], b[0])+.3*np.arctan2(a[1], a[0])
    assert angle_interp > -np.pi/2
    assert seam.segment(a, p) == 0
