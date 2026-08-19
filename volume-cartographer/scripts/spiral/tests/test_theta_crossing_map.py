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

    # One padded edge is ignored. Picks begin at node 1, so its adjustment is
    # the row's zero anchor and the final + crossing cancels the initial - one.
    padded_edges = torch.cat([edge_ids, edge_ids[-1:]])[None]
    padded_directions = torch.cat([directions, directions[-1:]])[None]
    valid = torch.tensor([[True, True, True, False]])
    picks = torch.tensor([[1, 2, 3]])
    nodes = torch.tensor([[1, 2, 3]])
    theta = crossing_map.node_theta[nodes]
    mask = torch.zeros_like(nodes, dtype=torch.bool)
    adjustment = crossing_map.adjustments(
        padded_edges, padded_directions, picks, nodes, theta, mask,
        torch.tensor(12.0), valid)
    assert torch.equal(adjustment, torch.tensor([[0.0, 0.0, 12.0]]))

    # Cached crossings are integer windings: scaling uses the current value,
    # not the value at refresh time.
    adjustment_20 = crossing_map.adjustments(
        padded_edges, padded_directions, picks, nodes, theta, mask,
        torch.tensor(20.0), valid)
    assert torch.equal(adjustment_20, torch.tensor([[0.0, 0.0, 20.0]]))


def test_patch_local_correction_connects_centres_to_fractional_picks():
    points = _points_for_theta([0.1, 0.2])
    crossing_map = ThetaCrossingMap('cpu')
    crossing_map.register_nodes(2, lambda indices: points[indices])
    crossing_map.register_edges([[0, 1]])
    crossing_map.force_refresh(_identity)
    edge_ids, directions = crossing_map.resolve_walks(torch.tensor([0, 1]))

    adjustment = crossing_map.adjustments(
        edge_ids[None], directions[None], torch.tensor([[0, 1]]),
        torch.tensor([[0, 1]]), torch.tensor([[2 * math.pi - 0.1, 0.2]]),
        torch.ones((1, 2), dtype=torch.bool), torch.tensor(12.0))
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
        crossing_map.resolve_edges([[1, 2]])
