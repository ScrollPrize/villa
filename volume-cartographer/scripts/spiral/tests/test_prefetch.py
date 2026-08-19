import torch

from prefetch import _iter_tensors


def test_iter_tensors_finds_walk_payload_nested_in_batch_metadata():
    direct = torch.tensor([1.0])
    walk_zyxs = torch.tensor([[2.0, 3.0, 4.0]])
    positions = torch.tensor([0])
    result = (
        direct,
        {'dense_walk_info': {
            'positions': positions,
            'walk_zyxs': walk_zyxs,
        }},
        [None],
    )

    tensors = list(_iter_tensors(result))
    assert tensors[0] is direct
    assert tensors[1] is positions
    assert tensors[2] is walk_zyxs
