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
    crossing_map.register_nodes(len(points), lambda indices: points[indices])
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
    crossing_map.register_nodes(2, lambda indices: points[indices])
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


def test_refresh_interval_chunking_force_refresh_and_interval_change():
    points = _points_for_theta([0.1, 0.2, 0.3, 0.4, 0.5])
    calls = []

    def transform(value):
        calls.append(len(value))
        return value

    crossing_map = ThetaCrossingMap('cpu', update_interval=3, chunk_size=2)
    crossing_map.register_nodes(len(points), lambda indices: points[indices])
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
    crossing_map.register_nodes(3, lambda indices: points[indices])
    crossing_map.register_edges([[0, 1]])
    with pytest.raises(RuntimeError, match='unregistered edge'):
        _pack_walks([
            SampledWalk(
                torch.tensor([1, 2]).numpy(), torch.tensor([0]).numpy(), False),
        ], crossing_map)


def test_common_packer_handles_ragged_reverse_single_node_and_pick_modes():
    points = _points_for_theta([0.0, 0.1, 0.2, 0.3])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(4, lambda indices: points[indices])
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
    crossing_map.register_nodes(4, lambda indices: points[indices])
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
    crossing_map.register_nodes(2, lambda indices: points[indices])
    crossing_map.register_edges([[0, 1]])
    with pytest.raises(ValueError, match=message):
        _pack_walks([
            SampledWalk(
                torch.tensor(nodes).numpy(), torch.tensor(picks).numpy(), False),
        ], crossing_map)
