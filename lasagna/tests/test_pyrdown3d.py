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


def _pyrdown3d_conv_reference(t, *, factor):
    # The grouped-conv3d version used before the shifted-adds rewrite.
    import torch.nn.functional as F
    k = torch.tensor([1, 4, 6, 4, 1], dtype=t.dtype, device=t.device) / 16.0
    f = int(factor)
    while f > 1:
        C = t.shape[0]
        for dim, pad_arg in enumerate([(0,0,0,0,2,2), (0,0,2,2,0,0), (2,2,0,0,0,0)]):
            shape = [1, 1, 1, 1, 1]
            shape[dim + 2] = 5
            kd = k.view(*shape).expand(C, 1, *shape[2:])
            t = F.conv3d(F.pad(t.unsqueeze(0), pad_arg, mode='reflect'), kd, groups=C)[0]
        t = t[:, ::2, ::2, ::2]
        f //= 2
    return t


def test_pyrdown3d_matches_conv_reference():
    torch.manual_seed(0)
    for shape, factor in [((3, 32, 32, 32), 2), ((3, 32, 32, 32), 4), ((2, 17, 24, 13), 2)]:
        x = torch.rand(*shape)
        out = tiled._pyrdown3d(x, factor=factor)
        ref = _pyrdown3d_conv_reference(x, factor=factor)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-6, rtol=0)
