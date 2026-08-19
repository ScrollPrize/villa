from dataclasses import dataclass

import torch

from prefetch import _iter_tensors


@dataclass
class WalkPayload:
    positions: torch.Tensor
    walk_zyxs: torch.Tensor


def test_iter_tensors_finds_walk_payload_nested_in_batch_metadata():
    direct = torch.tensor([1.0])
    walk_zyxs = torch.tensor([[2.0, 3.0, 4.0]])
    positions = torch.tensor([0])
    result = (
        direct,
        {'packed_dense_walks': WalkPayload(positions, walk_zyxs)},
        [None],
    )

    tensors = list(_iter_tensors(result))
    assert tensors[0] is direct
    assert tensors[1] is positions
    assert tensors[2] is walk_zyxs
