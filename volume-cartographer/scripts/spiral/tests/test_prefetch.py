from dataclasses import dataclass

import torch

from prefetch import _iter_tensors


@dataclass
class WalkPayload:
    positions: torch.Tensor
    edge_ids: torch.Tensor


def test_iter_tensors_finds_walk_payload_nested_in_batch_metadata():
    direct = torch.tensor([1.0])
    edge_ids = torch.tensor([[2, 3, 4]])
    positions = torch.tensor([0])
    result = (
        direct,
        {'packed_walks': WalkPayload(positions, edge_ids)},
        [None],
    )

    tensors = list(_iter_tensors(result))
    assert tensors[0] is direct
    assert tensors[1] is positions
    assert tensors[2] is edge_ids
