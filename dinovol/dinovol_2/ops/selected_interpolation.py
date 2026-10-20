"""Optional forward-only Triton interpolation for EMA-teacher cross-view targets."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def available():
    return triton is not None


if triton is not None:
    @triton.jit
    def _sample(Grid, Coords, Indices, Rows, Out, P: tl.constexpr,
                C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
                S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                S3: tl.constexpr, S4: tl.constexpr, BLOCK: tl.constexpr):
        point = tl.program_id(0)
        channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        index = tl.load(Indices + point)
        row = tl.load(Rows + index // P)
        z = tl.minimum(tl.maximum((tl.load(Coords + index * 3) + 0.5) * (D / ID) - 0.5, 0.), D - 1.)
        y = tl.minimum(tl.maximum((tl.load(Coords + index * 3 + 1) + 0.5) * (H / IH) - 0.5, 0.), H - 1.)
        x = tl.minimum(tl.maximum((tl.load(Coords + index * 3 + 2) + 0.5) * (W / IW) - 0.5, 0.), W - 1.)
        z0, y0, x0 = tl.floor(z).to(tl.int32), tl.floor(y).to(tl.int32), tl.floor(x).to(tl.int32)
        dz, dy, dx = z - z0, y - y0, x - x0
        result = tl.full((BLOCK,), 0, tl.float32)
        for corner in tl.static_range(8):
            iz = tl.minimum(z0 + (corner // 4), D - 1)
            iy = tl.minimum(y0 + ((corner // 2) % 2), H - 1)
            ix = tl.minimum(x0 + (corner % 2), W - 1)
            weight = (dz if corner // 4 else 1. - dz) * (dy if (corner // 2) % 2 else 1. - dy)
            weight = weight * (dx if corner % 2 else 1. - dx)
            value = tl.load(Grid + row * S0 + iz * S1 + iy * S2 + ix * S3 + channel * S4,
                            channel < C, other=0).to(tl.float32)
            result = result + value * weight
        tl.store(Out + point * C + channel, result, channel < C)


def sample_selected(grid, coordinates, input_shape, indices, source_rows):
    output = grid.new_empty((indices.numel(), grid.shape[-1]))
    if indices.numel():
        _sample[(indices.numel(), triton.cdiv(grid.shape[-1], 256))](
            grid, coordinates.float().contiguous(), indices.contiguous(), source_rows.contiguous(), output,
            coordinates.shape[1], grid.shape[-1], *grid.shape[1:4], *input_shape,
            *grid.stride(), BLOCK=256, num_warps=4, enable_fp_fusion=False,
        )
    return output
