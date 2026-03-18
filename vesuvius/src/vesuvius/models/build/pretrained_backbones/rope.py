import math
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


RopeEmbedding = tuple[Tensor, Tensor]


def rope_rotate_half(x: Tensor) -> Tensor:
    x_first, x_second = x.chunk(2, dim=-1)
    return torch.cat((-x_second, x_first), dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


def apply_rotary_embedding(x: Tensor, rope: RopeEmbedding, *, prefix_tokens: int = 0) -> Tensor:
    sin, cos = rope
    if prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be non-negative, got {prefix_tokens}")
    if x.shape[-2] - prefix_tokens != sin.shape[-2]:
        raise ValueError(
            "rope token count does not match the rotated suffix: "
            f"x.shape[-2]={x.shape[-2]}, prefix_tokens={prefix_tokens}, rope_tokens={sin.shape[-2]}"
        )

    x_dtype = x.dtype
    rope_dtype = sin.dtype
    x_prefix = x[..., :prefix_tokens, :]
    x_suffix = x[..., prefix_tokens:, :].to(dtype=rope_dtype)
    x_suffix = rope_apply(x_suffix, sin, cos).to(dtype=x_dtype)
    if prefix_tokens == 0:
        return x_suffix
    return torch.cat((x_prefix, x_suffix), dim=-2)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        *,
        ndim: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = 2.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError(f"RopePositionEmbedding only supports 2D or 3D inputs, got ndim={ndim}")
        if head_dim % (2 * ndim) != 0:
            raise ValueError(
                f"head_dim must be divisible by 2 * ndim for RoPE, got head_dim={head_dim}, ndim={ndim}"
            )

        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.head_dim = int(head_dim)
        self.ndim = int(ndim)
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype or torch.float32
        self.freqs_per_axis = self.head_dim // (2 * self.ndim)
        self.axis_dim = self.head_dim // self.ndim

        self.register_buffer(
            "periods",
            torch.empty(self.freqs_per_axis, device=device, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.freqs_per_axis, device=self.periods.device, dtype=self.periods.dtype)
                / self.axis_dim
            )
        else:
            assert self.min_period is not None
            assert self.max_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.freqs_per_axis, device=self.periods.device, dtype=self.periods.dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data.copy_(periods)

    def _normalized_axes(self, shape: tuple[int, ...]) -> list[Tensor]:
        device = self.periods.device
        dtype = self.periods.dtype
        if self.normalize_coords == "max":
            denominator = max(shape)
            axes = [torch.arange(0.5, size, device=device, dtype=dtype) / denominator for size in shape]
        elif self.normalize_coords == "min":
            denominator = min(shape)
            axes = [torch.arange(0.5, size, device=device, dtype=dtype) / denominator for size in shape]
        elif self.normalize_coords == "separate":
            axes = [torch.arange(0.5, size, device=device, dtype=dtype) / size for size in shape]
        else:
            raise ValueError(f"Unknown normalize_coords={self.normalize_coords!r}")
        return axes

    def _apply_coord_augmentations(self, coords: Tensor) -> Tensor:
        if self.training and self.shift_coords is not None:
            shift = torch.empty(self.ndim, device=coords.device, dtype=coords.dtype).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords = coords + shift[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_log = math.log(self.jitter_coords)
            jitter = torch.empty(self.ndim, device=coords.device, dtype=coords.dtype).uniform_(
                -jitter_log, jitter_log
            ).exp()
            coords = coords * jitter[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_log = math.log(self.rescale_coords)
            rescale = torch.empty(1, device=coords.device, dtype=coords.dtype).uniform_(
                -rescale_log, rescale_log
            ).exp()
            coords = coords * rescale

        return coords

    def get_embed(self, shape: Sequence[int]) -> RopeEmbedding:
        spatial_shape = tuple(int(size) for size in shape)
        if len(spatial_shape) != self.ndim:
            raise ValueError(f"expected a {self.ndim}D shape and got {spatial_shape}")
        if any(size <= 0 for size in spatial_shape):
            raise ValueError(f"shape dimensions must be positive, got {spatial_shape}")

        axes = self._normalized_axes(spatial_shape)
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, self.ndim)
        coords = (2.0 * coords) - 1.0
        coords = self._apply_coord_augmentations(coords)

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos

    def forward(self, shape: Sequence[int]) -> RopeEmbedding:
        return self.get_embed(shape)
