"""Optional forward-only Triton interpolation for teacher targets and volume inference."""
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
    def _corner(Grid, row, channel, mask, z0, y0, x0, dz, dy, dx,
                corner: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                S3: tl.constexpr, S4: tl.constexpr, CUDA_ORDER: tl.constexpr):
        iz = tl.minimum(z0 + (corner // 4), D - 1)
        iy = tl.minimum(y0 + ((corner // 2) % 2), H - 1)
        ix = tl.minimum(x0 + (corner % 2), W - 1)
        wz = dz if corner // 4 else 1. - dz
        wy = dy if (corner // 2) % 2 else 1. - dy
        wx = dx if corner % 2 else 1. - dx
        # CUDA torch.prod of three contiguous elements reduces lanes 0,2 then 1.
        weight = (wz * wx) * wy if CUDA_ORDER else (wz * wy) * wx
        value = tl.load(Grid + row * S0 + iz * S1 + iy * S2 + ix * S3 + channel * S4,
                        mask, other=0).to(tl.float32)
        return value * weight

    @triton.jit
    def _sample(Grid, Coords, Indices, Rows, Out, P: tl.constexpr,
                C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
                S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                S3: tl.constexpr, S4: tl.constexpr, BLOCK: tl.constexpr,
                DENSE: tl.constexpr = False, N: tl.constexpr = 0):
        if DENSE:
            offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            point, channel = offset // C, offset % C
            index, row = point, point // P
            mask = point < N
            coordinate_mask = mask
        else:
            point = tl.program_id(0)
            channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            index = tl.load(Indices + point)
            row = tl.load(Rows + index // P)
            mask = channel < C
            coordinate_mask = True
        z = tl.minimum(tl.maximum((tl.load(Coords + index * 3, coordinate_mask, other=0) + 0.5) * (D / ID) - 0.5, 0.), D - 1.)
        y = tl.minimum(tl.maximum((tl.load(Coords + index * 3 + 1, coordinate_mask, other=0) + 0.5) * (H / IH) - 0.5, 0.), H - 1.)
        x = tl.minimum(tl.maximum((tl.load(Coords + index * 3 + 2, coordinate_mask, other=0) + 0.5) * (W / IW) - 0.5, 0.), W - 1.)
        z0, y0, x0 = tl.floor(z).to(tl.int32), tl.floor(y).to(tl.int32), tl.floor(x).to(tl.int32)
        dz, dy, dx = z - z0, y - y0, x - x0
        result = tl.full((BLOCK,), 0, tl.float32)
        if DENSE:
            # Match torch.sum's four accumulators over the eight gathered corners.
            for corner in tl.static_range(4):
                a = _corner(Grid, row, channel, mask, z0, y0, x0, dz, dy, dx,
                            corner, D, H, W, S0, S1, S2, S3, S4, True)
                b = _corner(Grid, row, channel, mask, z0, y0, x0, dz, dy, dx,
                            corner + 4, D, H, W, S0, S1, S2, S3, S4, True)
                result = result + (a + b)
        else:
            for corner in tl.static_range(8):
                result = result + _corner(Grid, row, channel, mask, z0, y0, x0, dz, dy, dx,
                                          corner, D, H, W, S0, S1, S2, S3, S4, False)
        tl.store(Out + point * C + channel, result, mask)


def sample_selected(grid, coordinates, input_shape, indices, source_rows):
    output = grid.new_empty((indices.numel(), grid.shape[-1]))
    if indices.numel():
        _sample[(indices.numel(), triton.cdiv(grid.shape[-1], 256))](
            grid, coordinates.float().contiguous(), indices.contiguous(), source_rows.contiguous(), output,
            coordinates.shape[1], grid.shape[-1], *grid.shape[1:4], *input_shape,
            *grid.stride(), BLOCK=256, num_warps=4, enable_fp_fusion=False,
        )
    return output


def sample_batched(grid, coordinates, input_shape):
    """FP32 interpolation without materializing [B,P,8,C] corners or products.

    Uses four corner accumulators to keep rounding close to the eager sampler.
    Callers retain the differentiable PyTorch path for training and unsupported inputs.
    """
    output = torch.empty((*coordinates.shape[:2], grid.shape[-1]), device=grid.device, dtype=torch.float32)
    if output.numel():
        _sample[(triton.cdiv(output.numel(), 256),)](
            grid, coordinates.contiguous(), coordinates, coordinates, output,
            coordinates.shape[1], grid.shape[-1], *grid.shape[1:4], *input_shape,
            *grid.stride(), BLOCK=256, DENSE=True, N=coordinates.shape[0] * coordinates.shape[1],
            num_warps=4, enable_fp_fusion=False,
        )
    return output
