import math

import pytest
import torch

from theta_crossing_map import ThetaCrossingMap


def _points_for_theta(theta):
    return torch.tensor([
        [0.0, math.sin(value), math.cos(value)] for value in theta
    ], dtype=torch.float32)


def _identity(value):
    return value


def test_unwrap_tree_caches_branching_multiwrap_node_potentials():
    # DFS preorder: 0 -> 1 -> 2 -> 3, then node 4 branches from node 1.
    # The main arm crosses theta=0 once; the branch does not.
    unwrapped_theta = [5.8, 6.1, 6.4, 6.7, 5.5]
    points = _points_for_theta([value % (2 * math.pi) for value in unwrapped_theta])
    crossing_map = ThetaCrossingMap('cpu', chunk_size=2)
    crossing_map.register_nodes(len(points), lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([
        [0, 1], [1, 2], [2, 3], [1, 4],
        # Non-tree edge with the same continuous lift checks cycle consistency.
        [0, 4],
    ])
    crossing_map.register_unwrap_tree(
        [0, 1, 2, 3, 4], [-1, 0, 1, 2, 1])
    crossing_map.force_refresh(_identity)

    assert crossing_map.node_winding_potential.tolist() == [0, 0, -1, -1, 0]
    assert crossing_map.potential_consistency() == {
        'checked_edges': 5,
        'inconsistent_edges': 0,
        'max_abs_residual': 0,
    }

    # Arbitrary sample order no longer needs to be a contiguous theta walk.
    sample_ids = torch.tensor([[3, 1, 4, 2]])
    sample_theta = crossing_map.node_theta[sample_ids]
    adjustments = crossing_map.adjustments_from_potentials(
        sample_ids, sample_theta, torch.tensor(12.0))
    assert torch.equal(adjustments, torch.tensor([[-12.0, 0.0, 0.0, -12.0]]))


def test_potential_inconsistencies_returns_only_bad_edge_nodes_on_host():
    # The tree walks 0.1 -> 2.5 -> 4.9 without crossing wrapped theta zero,
    # while the direct 0 -> 2 edge does cross the branch cut.  No global lift
    # can satisfy that cycle.
    points = _points_for_theta([0.1, 2.5, 4.9])
    crossing_map = ThetaCrossingMap('cpu', chunk_size=2)
    crossing_map.register_nodes(3, lambda lo, hi: points[lo:hi])
    crossing_map.register_unwrap_tree([0, 1, 2], [-1, 0, 1])
    crossing_map.register_edges([[0, 2]])
    crossing_map.force_refresh(_identity)

    report, bad_nodes = crossing_map.potential_inconsistencies()

    assert report == {
        'checked_edges': 3,
        'inconsistent_edges': 1,
        'max_abs_residual': 1,
    }
    assert bad_nodes.device.type == 'cpu'
    assert bad_nodes.tolist() == [0, 2]
    assert crossing_map.potential_consistency() == report


def test_unordered_potentials_span_forty_wraps_without_sparse_aliasing():
    unwrapped_theta = torch.arange(0.2, 40 * 2 * math.pi + 0.2, 0.2)
    wrapped_theta = unwrapped_theta.remainder(2 * math.pi)
    points = _points_for_theta(wrapped_theta.tolist())
    num_nodes = len(points)
    crossing_map = ThetaCrossingMap('cpu', chunk_size=37)
    crossing_map.register_nodes(num_nodes, lambda lo, hi: points[lo:hi])
    crossing_map.register_unwrap_tree(
        torch.arange(num_nodes),
        torch.cat([torch.tensor([-1]), torch.arange(num_nodes - 1)]),
    )
    crossing_map.force_refresh(_identity)

    expected = -torch.floor(unwrapped_theta / (2 * math.pi)).to(torch.int32)
    torch.testing.assert_close(crossing_map.node_winding_potential, expected)
    # Deliberately stride farther than half a turn between arbitrary picks.
    picked = torch.arange(num_nodes - 1, -1, -31)[:800][None, :]
    actual = crossing_map.winding_potentials(
        picked, crossing_map.node_theta[picked])
    torch.testing.assert_close(actual, expected[picked])


def test_potential_adjustments_handle_fractional_picks():
    # Patch nodes 1..3 form a tree rooted at node 1 and cross theta=0 at
    # node 2.
    points = _points_for_theta([6.0, 6.1, 0.1, 0.3])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(len(points), lambda lo, hi: points[lo:hi])
    crossing_map.register_unwrap_tree([1, 2, 3], [-1, 0, 1])
    crossing_map.force_refresh(_identity)

    sample_ids = torch.tensor([[2, 3]])
    # The first fractional point is just before theta=0 although its cell
    # centre is just after it; the local correction cancels the tree crossing.
    sampled_theta = torch.tensor([[2 * math.pi - 0.05, 0.3]])
    adjustments = crossing_map.adjustments_from_potentials(
        sample_ids, sampled_theta, torch.tensor(10.0))
    assert torch.equal(adjustments, torch.tensor([[0.0, -10.0]]))


def test_unwrap_tree_rejects_non_preorder_and_unregistered_potential_lookup():
    points = _points_for_theta([0.0, 0.1, 0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(3, lambda lo, hi: points[lo:hi])
    with pytest.raises(ValueError, match='precede'):
        crossing_map.register_unwrap_tree([0, 1, 2], [-1, 2, 0])
    crossing_map_4 = ThetaCrossingMap('cpu')
    points_4 = _points_for_theta([0.0, 0.1, 0.2, 0.3])
    crossing_map_4.register_nodes(4, lambda lo, hi: points_4[lo:hi])
    with pytest.raises(ValueError, match='depth-first preorder'):
        crossing_map_4.register_unwrap_tree([0, 1, 2, 3], [-1, 0, 0, 1])
    crossing_map.register_unwrap_tree([0, 1], [-1, 0])
    crossing_map.force_refresh(_identity)
    with pytest.raises(RuntimeError, match='no registered unwrap potential'):
        crossing_map.winding_potentials(torch.tensor([2]))


def test_single_node_unwrap_tree_needs_no_registered_edge():
    points = _points_for_theta([0.4])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(1, lambda lo, hi: points[lo:hi])
    crossing_map.register_unwrap_tree([0], [-1])
    crossing_map.force_refresh(_identity)
    assert crossing_map.edge_nodes.numel() == 0
    assert crossing_map.winding_potentials([0]).tolist() == [0]


def test_refresh_interval_chunking_force_refresh_and_interval_change():
    points = _points_for_theta([0.1, 0.2, 0.3, 0.4, 0.5])
    calls = []

    def transform(value):
        calls.append(len(value))
        return value

    crossing_map = ThetaCrossingMap('cpu', update_interval=3, chunk_size=2)
    crossing_map.register_nodes(len(points), lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1]])
    assert crossing_map.refresh_if_due(10, transform)
    assert calls == [2, 2, 1]
    assert not crossing_map.refresh_if_due(12, transform)
    assert crossing_map.refresh_if_due(13, transform)

    crossing_map.force_refresh(transform)
    num_calls = len(calls)
    # A diagnostic refresh is reused and anchors the schedule on first use.
    assert not crossing_map.refresh_if_due(20, transform)
    assert len(calls) == num_calls
    # A live interval change resets the cadence.
    assert crossing_map.refresh_if_due(21, transform, update_interval=7)


def test_unregistered_sampled_edge_fails_clearly():
    points = _points_for_theta([0.0, 0.1, 0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(3, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1]])
    with pytest.raises(RuntimeError, match='unregistered edge'):
        crossing_map.resolve_edges(torch.tensor([[1, 2]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_unset_potential_raises_at_assert_boundary():
    # On CUDA the unset-potential hard error is deferred (the verdict is
    # copied off-device asynchronously); the guarantee is that it surfaces
    # no later than assert_no_pending_potential_errors(), which the training
    # loop calls before every optimizer step.
    points = _points_for_theta([0.0, 0.1, 0.2])
    crossing_map = ThetaCrossingMap('cuda')
    crossing_map.register_nodes(3, lambda lo, hi: points[lo:hi])
    crossing_map.register_unwrap_tree([0, 1], [-1, 0])
    crossing_map.force_refresh(_identity)

    # Node 2 is outside the tree; the call itself may return the sentinel
    # without raising, but the boundary must raise.
    crossing_map.winding_potentials(torch.tensor([2], device='cuda'))
    with pytest.raises(RuntimeError, match='no registered unwrap potential'):
        crossing_map.assert_no_pending_potential_errors()

    # A healthy query afterwards passes the boundary cleanly.
    crossing_map._pending_potential_checks.clear()
    assert crossing_map.winding_potentials(
        torch.tensor([0], device='cuda')).tolist() == [0]
    crossing_map.assert_no_pending_potential_errors()


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='needs CUDA')),
])
def test_compact_source_potentials_stream_in_chunks(device):
    # Preorder tree: 0 -> 1 -> 2, 0 -> 3, and a second root 4; exits are interior.
    nodes = torch.arange(5).numpy()
    parents = torch.tensor([0, 0, 1, 0, 4]).numpy()
    exits = torch.tensor([4, 3, 3, 4, 5]).numpy()
    points = _points_for_theta([0.1, 6.2, 0.2, 6.1, 3.0])  # theta in [0, 2pi)
    crossing_map = ThetaCrossingMap(device, chunk_size=2)
    start = crossing_map.register_nodes(5, lambda lo, hi: points[lo:hi].to(device))
    crossing_map.register_potential_source(
        start, 5, lambda lo, hi: (nodes[lo:hi], parents[lo:hi], exits[lo:hi]),
        lambda cursor, limit: (cursor, torch.empty((0, 2), dtype=torch.int64).numpy()))
    crossing_map.force_refresh(_identity)
    assert crossing_map.node_winding_potential.tolist() == [0, 1, 0, 1, 0]
    assert crossing_map.node_winding_potential.device.type == device
