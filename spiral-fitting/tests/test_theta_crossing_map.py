import math

import pytest
import torch

from losses import SampledWalk, _pack_walks
from theta_crossing_map import ThetaCrossingMap


def _points_for_theta(theta):
    return torch.tensor([
        [0.0, math.sin(value), math.cos(value)] for value in theta
    ], dtype=torch.float32)


def _identity(value):
    return value


def test_crossings_reverse_reanchor_padding_and_current_dr_scaling():
    points = _points_for_theta([-0.2, 0.2, 1.0, -0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(len(points), lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1], [1, 2], [2, 3], [1, 0]])
    crossing_map.force_refresh(_identity)

    assert crossing_map.crossings.dtype == torch.int8
    assert crossing_map.node_theta.dtype == torch.float32
    assert crossing_map.edge_nodes.shape[0] == 3  # reverse duplicate removed

    edge_ids, directions = crossing_map.resolve_walks(torch.arange(4))
    reverse_ids, reverse_directions = crossing_map.resolve_walks(
        torch.arange(3, -1, -1))
    assert torch.equal(reverse_ids, edge_ids.flip(0))
    assert torch.equal(reverse_directions, -directions.flip(0))

    # Picks begin at node 1, so its adjustment is the row's zero anchor and
    # the final + crossing cancels the initial - one.
    packed = _pack_walks([
        SampledWalk(
            torch.arange(4).numpy(), torch.tensor([1, 2, 3]).numpy(), False),
    ], crossing_map)
    theta = crossing_map.node_theta[torch.tensor([[1, 2, 3]])]
    adjustment = crossing_map.adjustments(packed, theta, torch.tensor(12.0))
    assert torch.equal(adjustment, torch.tensor([[0.0, 0.0, 12.0]]))

    # Cached crossings are integer windings: scaling uses the current value,
    # not the value at refresh time.
    adjustment_20 = crossing_map.adjustments(
        packed, theta, torch.tensor(20.0))
    assert torch.equal(adjustment_20, torch.tensor([[0.0, 0.0, 20.0]]))


def test_patch_local_correction_connects_centres_to_fractional_picks():
    points = _points_for_theta([0.1, 0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(2, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1]])
    crossing_map.force_refresh(_identity)
    packed = _pack_walks([
        SampledWalk(
            torch.tensor([0, 1]).numpy(), torch.tensor([0, 1]).numpy(), True),
    ], crossing_map)
    adjustment = crossing_map.adjustments(
        packed, torch.tensor([[2 * math.pi - 0.1, 0.2]]),
        torch.tensor(12.0))
    assert torch.allclose(adjustment, torch.tensor([[0.0, -12.0]]))


def test_absolute_winding_reference_survives_unsampled_walk_origin():
    # The exact annotation and dense-walk origin are before theta=0, but the
    # first random pick is after it. Anchor-supervised winding walks must retain
    # that crossing instead of reanchoring it away at the first sparse pick.
    points = _points_for_theta([6.0, 6.1, 0.1, 0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(4, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[1, 2], [2, 3]])
    crossing_map.force_refresh(_identity)
    packed = _pack_walks([
        SampledWalk(
            torch.tensor([1, 2, 3]).numpy(),
            torch.tensor([1, 2]).numpy(), True,
            reference_node_id=0),
    ], crossing_map)
    theta = crossing_map.node_theta[torch.tensor([[2, 3]])]
    adjustment = crossing_map.adjustments(
        packed, theta, torch.tensor(12.0))

    assert torch.equal(adjustment, torch.tensor([[-12.0, -12.0]]))
    raw_shifted = torch.tensor([[60.0, 60.0]])
    absolute_target = torch.tensor(48.0)
    assert torch.equal(
        raw_shifted + adjustment,
        absolute_target.expand_as(raw_shifted))


def test_relative_winding_walks_keep_each_annotation_frame():
    # Side one crosses theta=0 before its first pick; side two does not. The
    # pooled values must still differ by the annotations' two-winding delta.
    points = _points_for_theta([
        6.0, 6.1, 0.1, 0.2,
        1.0, 1.1, 1.2, 1.3,
    ])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(8, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([
        [1, 2], [2, 3], [5, 6], [6, 7],
    ])
    crossing_map.force_refresh(_identity)
    packed = _pack_walks([
        SampledWalk(
            torch.tensor([1, 2, 3]).numpy(),
            torch.tensor([1, 2]).numpy(), True,
            reference_node_id=0),
        SampledWalk(
            torch.tensor([5, 6, 7]).numpy(),
            torch.tensor([1, 2]).numpy(), True,
            reference_node_id=4),
    ], crossing_map)
    theta = crossing_map.node_theta[torch.tensor([[2, 3], [6, 7]])]
    adjustment = crossing_map.adjustments(
        packed, theta, torch.tensor(12.0))

    raw_shifted = torch.tensor([[60.0, 60.0], [72.0, 72.0]])
    unwrapped = raw_shifted + adjustment
    assert torch.equal(unwrapped[1] - unwrapped[0], torch.tensor([24.0, 24.0]))


def test_reference_node_connects_exact_annotation_to_patch_walk_origin():
    # The PCL point and its attached patch quad can straddle theta=0 even though
    # they are spatially adjacent; transport that final local branch step too.
    points = _points_for_theta([6.1, 0.1, 0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(3, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[1, 2]])
    crossing_map.force_refresh(_identity)
    packed = _pack_walks([
        SampledWalk(
            torch.tensor([1, 2]).numpy(),
            torch.tensor([0, 1]).numpy(), True,
            reference_node_id=0),
    ], crossing_map)
    theta = crossing_map.node_theta[torch.tensor([[1, 2]])]
    adjustment = crossing_map.adjustments(
        packed, theta, torch.tensor(12.0))
    assert torch.equal(adjustment, torch.tensor([[-12.0, -12.0]]))


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


def test_potential_adjustments_handle_fractional_and_annotation_frames():
    # Reference node 0 is an exact PCL point. Patch nodes 1..3 form a tree
    # rooted at the attached quad (node 1) and cross theta=0 at node 2.
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
        sample_ids,
        sampled_theta,
        torch.tensor(10.0),
        reference_node_ids=torch.tensor([0]),
        reference_patch_node_ids=torch.tensor([1]),
    )
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
        _pack_walks([
            SampledWalk(
                torch.tensor([1, 2]).numpy(), torch.tensor([0]).numpy(), False),
        ], crossing_map)


def test_common_packer_handles_ragged_reverse_single_node_and_pick_modes():
    points = _points_for_theta([0.0, 0.1, 0.2, 0.3])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(4, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1], [1, 2], [2, 3]])
    crossing_map.force_refresh(_identity)

    packed = _pack_walks([
        SampledWalk(
            torch.tensor([0, 1, 2, 3]).numpy(),
            torch.tensor([0, 2, 2]).numpy(), True),
        SampledWalk(
            torch.tensor([3, 2, 1]).numpy(),
            torch.tensor([0, 2, 1]).numpy(), False),
        SampledWalk(
            torch.tensor([2]).numpy(),
            torch.tensor([0, 0, 0]).numpy(), False),
    ], crossing_map)

    assert packed.edge_ids.device == crossing_map.device
    assert packed.directions.device == crossing_map.device
    assert packed.edge_valid.tolist() == [
        [True, True, True], [True, True, False], [False, False, False]]
    assert packed.directions[1, :2].tolist() == [-1, -1]
    assert packed.pick_positions.tolist() == [[0, 2, 2], [0, 2, 1], [0, 0, 0]]
    assert packed.correction_node_ids.tolist() == [
        [0, 2, 2], [-1, -1, -1], [-1, -1, -1]]
    adjustment = crossing_map.adjustments(
        packed, crossing_map.node_theta[torch.tensor([
            [0, 2, 2], [3, 1, 2], [2, 2, 2],
        ])], torch.tensor(12.0))
    assert adjustment[2].tolist() == [0.0, 0.0, 0.0]


def test_new_adjustment_container_preserves_values_loss_and_gradients():
    points = _points_for_theta([-0.2, 0.2, 1.0, -0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(4, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1], [1, 2], [2, 3]])
    crossing_map.force_refresh(_identity)
    packed = _pack_walks([
        SampledWalk(
            torch.tensor([0, 1, 2, 3]).numpy(),
            torch.tensor([0, 2, 3]).numpy(), True),
        SampledWalk(
            torch.tensor([3, 2, 1]).numpy(),
            torch.tensor([0, 1, 2]).numpy(), False),
    ], crossing_map)
    sampled_theta = torch.tensor([
        [2 * math.pi - 0.1, 1.0, 2 * math.pi - 0.2],
        [2 * math.pi - 0.2, 1.0, 0.2],
    ])

    def legacy_adjustments(dr):
        steps = (
            crossing_map.crossings[packed.edge_ids].to(torch.int32)
            * packed.directions.to(torch.int32)
            * packed.edge_valid.to(torch.int32))
        cumulative = torch.cat([
            torch.zeros((2, 1), dtype=torch.int32), steps.cumsum(dim=-1),
        ], dim=-1)
        picked = torch.gather(cumulative, -1, packed.pick_positions)
        mask = packed.correction_node_ids >= 0
        centre = crossing_map.node_theta[
            packed.correction_node_ids.clamp_min(0)]
        delta = sampled_theta - centre
        local = ((delta > math.pi).to(torch.int32)
                 - (delta < -math.pi).to(torch.int32))
        picked = picked + torch.where(mask, local, torch.zeros_like(local))
        picked = picked - picked[..., :1]
        return picked.to(dr.dtype) * dr.detach()

    old_dr = torch.tensor(12.0, requires_grad=True)
    new_dr = old_dr.detach().clone().requires_grad_()
    old_raw = torch.tensor(
        [[3.0, 4.0, 5.0], [8.0, 7.0, 6.0]], requires_grad=True)
    new_raw = old_raw.detach().clone().requires_grad_()
    old_adjustment = legacy_adjustments(old_dr)
    new_adjustment = crossing_map.adjustments(
        packed, sampled_theta, new_dr)
    torch.testing.assert_close(new_adjustment, old_adjustment)

    old_unwrapped = old_raw + old_dr * 0.125 + old_adjustment
    new_unwrapped = new_raw + new_dr * 0.125 + new_adjustment
    torch.testing.assert_close(new_unwrapped, old_unwrapped)
    old_loss = (old_unwrapped - 2.5).square().mean()
    new_loss = (new_unwrapped - 2.5).square().mean()
    torch.testing.assert_close(new_loss, old_loss)
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_raw.grad, old_raw.grad)
    torch.testing.assert_close(new_dr.grad, old_dr.grad)


@pytest.mark.parametrize(
    ('nodes', 'picks', 'message'),
    [([], [0], 'nonempty'), ([0, 1], [2], 'out-of-range')],
)
def test_common_packer_rejects_invalid_walks(nodes, picks, message):
    points = _points_for_theta([0.0, 0.1])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(2, lambda lo, hi: points[lo:hi])
    crossing_map.register_edges([[0, 1]])
    with pytest.raises(ValueError, match=message):
        _pack_walks([
            SampledWalk(
                torch.tensor(nodes).numpy(), torch.tensor(picks).numpy(), False),
        ], crossing_map)


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
