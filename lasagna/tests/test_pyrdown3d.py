import torch

from lasagna import tiled_predict3d as tiled


def test_pyrdown3d_returns_contiguous_cpu_tensor():
    # The native accumulator (accumulator_add.cpp add_inplace) needs contiguous
    # rows. On a CPU device .cpu().numpy() keeps the strides of a sliced view, so
    # predict3d --device cpu failed with "X rows must be contiguous".
    torch.manual_seed(0)
    x = torch.rand(3, 32, 32, 32)
    out = tiled._pyrdown3d(x, factor=4)
    assert out.shape == (3, 8, 8, 8)
    assert out.is_contiguous()
    assert out.cpu().numpy().flags["C_CONTIGUOUS"]
