import math
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


RopeEmbedding = tuple[Tensor, Tensor]
RopeCoords = Tensor


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

    # Equal to concatenating the prefix with rope_apply of the suffix, without
    # concatenating activations: the prefix gets the identity rotation (cos 1, sin 0)
    # and rotate_half is a gather with the sign folded into sin. Gathers and
    # pointwise ops can be recomputed in backward by compiled graphs instead of saved.
    if prefix_tokens:
        sin = torch.nn.functional.pad(sin, (0, 0, prefix_tokens, 0))
        cos = torch.nn.functional.pad(cos, (0, 0, prefix_tokens, 0), value=1.0)
    half = x.shape[-1] // 2
    rotation = torch.arange(x.shape[-1], device=x.device).roll(half)
    signed_sin = torch.cat((-sin[..., :half], sin[..., half:]), dim=-1)
    x_rope = x.to(dtype=sin.dtype)
    return ((x_rope * cos) + (x_rope[..., rotation] * signed_sin)).to(dtype=x.dtype)


class _BaseRopePositionEmbedding(nn.Module):
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

    def _build_periods(self, count: int, denominator_dim: int, *, device: torch.device | None = None) -> Tensor:
        device = device or torch.device("cpu")
        if self.base is not None:
            return self.base ** (
                2 * torch.arange(count, device=device, dtype=self.dtype) / denominator_dim
            )

        assert self.min_period is not None
        assert self.max_period is not None
        base = self.max_period / self.min_period
        exponents = torch.linspace(0, 1, count, device=device, dtype=self.dtype)
        periods = base**exponents
        periods = periods / base
        periods = periods * self.max_period
        return periods

    def _get_coords(self, shape: Sequence[int]) -> Tensor:
        spatial_shape = tuple(int(size) for size in shape)
        if len(spatial_shape) != self.ndim:
            raise ValueError(f"expected a {self.ndim}D shape and got {spatial_shape}")
        if any(size <= 0 for size in spatial_shape):
            raise ValueError(f"shape dimensions must be positive, got {spatial_shape}")

        axes = self._normalized_axes(spatial_shape)
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, self.ndim)
        coords = (2.0 * coords) - 1.0
        coords = self._apply_coord_augmentations(coords)
        return coords

    def get_coords(self, shape: Sequence[int]) -> RopeCoords:
        return self._get_coords(shape)


class RopePositionEmbedding(_BaseRopePositionEmbedding):
    def __init__(
        self,
        head_dim: int,
        *,
        ndim: int,
        num_heads: int | None = None,
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
        if head_dim % (2 * ndim) != 0:
            raise ValueError(
                f"head_dim must be divisible by 2 * ndim for RoPE, got head_dim={head_dim}, ndim={ndim}"
            )
        super().__init__(
            head_dim,
            ndim=ndim,
            base=base,
            min_period=min_period,
            max_period=max_period,
            normalize_coords=normalize_coords,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
            dtype=dtype,
            device=device,
        )

        self.freqs_per_axis = self.head_dim // (2 * self.ndim)
        self.axis_dim = self.head_dim // self.ndim
        self.register_buffer(
            "periods",
            torch.empty(self.freqs_per_axis, device=device, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        periods = self._build_periods(
            self.freqs_per_axis,
            self.axis_dim,
            device=self.periods.device,
        )
        self.periods.data.copy_(periods)

    def get_embed_from_coords(self, coords: RopeCoords) -> RopeEmbedding:
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos

    def get_embed(self, shape: Sequence[int]) -> RopeEmbedding:
        coords = self.get_coords(shape)
        return self.get_embed_from_coords(coords)

    def forward(self, shape: Sequence[int]) -> RopeEmbedding:
        return self.get_embed(shape)


class MixedRopePositionEmbedding(_BaseRopePositionEmbedding):
    def __init__(
        self,
        head_dim: int,
        *,
        ndim: int,
        num_heads: int,
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
        if head_dim % (2 * ndim) != 0:
            raise ValueError(
                f"mixed RoPE warm-start requires head_dim divisible by 2 * ndim, got head_dim={head_dim}, ndim={ndim}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive for mixed RoPE, got {num_heads}")
        super().__init__(
            head_dim,
            ndim=ndim,
            base=base,
            min_period=min_period,
            max_period=max_period,
            normalize_coords=normalize_coords,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
            dtype=dtype,
            device=device,
        )

        self.num_heads = int(num_heads)
        self.freqs_per_axis = self.head_dim // (2 * self.ndim)
        self.axis_dim = self.head_dim // self.ndim
        self.num_pairs = self.head_dim // 2
        self.register_buffer(
            "periods",
            torch.empty(self.freqs_per_axis, device=device, dtype=self.dtype),
            persistent=True,
        )
        self.mix_frequencies = nn.Parameter(
            torch.empty(self.num_heads, self.num_pairs, self.ndim, device=device, dtype=self.dtype)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        periods = self._build_periods(
            self.freqs_per_axis,
            self.axis_dim,
            device=self.periods.device,
        )
        self.periods.data.copy_(periods)
        self.reset_mixed_frequencies_to_random_oriented()

    @torch.no_grad()
    def reset_mixed_frequencies_to_axial(self) -> None:
        mix_frequencies = torch.zeros_like(self.mix_frequencies)
        inv_periods = self.periods.reciprocal()
        for axis in range(self.ndim):
            start = axis * self.freqs_per_axis
            end = start + self.freqs_per_axis
            mix_frequencies[:, start:end, axis] = inv_periods
        self.mix_frequencies.copy_(mix_frequencies)

    @torch.no_grad()
    def reset_mixed_frequencies_to_random_oriented(self) -> None:
        mix_frequencies = torch.empty_like(self.mix_frequencies)
        inv_periods = self.periods.reciprocal()

        if self.ndim == 2:
            angles = torch.empty(self.num_heads, device=self.periods.device, dtype=self.periods.dtype).uniform_(
                0.0, 2.0 * math.pi
            )
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            basis = torch.stack(
                (
                    torch.stack((cos, sin), dim=-1),
                    torch.stack((-sin, cos), dim=-1),
                ),
                dim=-1,
            )
        else:
            basis = torch.randn(
                self.num_heads,
                self.ndim,
                self.ndim,
                device=self.periods.device,
                dtype=self.periods.dtype,
            )
            basis, r = torch.linalg.qr(basis)
            diag = torch.diagonal(r, dim1=-2, dim2=-1)
            signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
            basis = basis * signs.unsqueeze(-2)
            negative_det = torch.linalg.det(basis) < 0
            if negative_det.any():
                basis[negative_det, :, 0] *= -1

        for axis in range(self.ndim):
            start = axis * self.freqs_per_axis
            end = start + self.freqs_per_axis
            mix_frequencies[:, start:end, :] = inv_periods[None, :, None] * basis[:, None, :, axis]

        self.mix_frequencies.copy_(mix_frequencies)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"mix_frequencies"}

    def get_embed_from_coords(self, coords: RopeCoords) -> RopeEmbedding:
        # Autocast would run the einsum, and so the angles and sin/cos, in half precision.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            angles = 2 * math.pi * torch.einsum("td,hpd->htp", coords, self.mix_frequencies)
        angles = angles.tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos

    def get_embed(self, shape: Sequence[int]) -> RopeEmbedding:
        coords = self.get_coords(shape)
        return self.get_embed_from_coords(coords)

    def forward(self, shape: Sequence[int]) -> RopeEmbedding:
        return self.get_embed(shape)


class CompassRopePositionEmbedding(_BaseRopePositionEmbedding):
    """Compass-RoPE (Sec. 5 / Appendix F, https://openreview.net/pdf?id=kbIXDF7PD0).

    In 2D, signed radii and stratified polar angles form complex frequencies,
    mixed by a unitary DFT within each head. Cartesian phase evaluation is
    equivalent to the paper's polar formula, without atan2 at the grid origin.

    The paper is 2D only. Our 3D extension uses spherical angles and the real
    orthogonal Hartley transform (Re(DFT) - Im(DFT)) on each spatial component.
    Unlike discarding the imaginary part of a complex FFT, this preserves
    frequency energy and commutes with spatial rotations, without favoring an
    axis. Both paths retain Dinovol's normalized coordinates and 2*pi scaling.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        ndim: int,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = 2.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        use_dft: bool = True,
        radial_jitter: float = 0.01,
    ) -> None:
        if head_dim <= 0 or head_dim % 2:
            raise ValueError(f"Compass RoPE requires a positive even head_dim, got {head_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive for Compass RoPE, got {num_heads}")
        if not 0 <= radial_jitter < 1:
            raise ValueError(f"radial_jitter must be in [0, 1), got {radial_jitter}")
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError("Compass RoPE requires float32 or float64 phase precision")
        super().__init__(
            head_dim, ndim=ndim, base=base, min_period=min_period, max_period=max_period,
            normalize_coords=normalize_coords, shift_coords=shift_coords,
            jitter_coords=jitter_coords, rescale_coords=rescale_coords, dtype=dtype, device=device,
        )
        self.num_heads = int(num_heads)
        self.num_pairs = self.head_dim // 2
        self.use_dft = bool(use_dft)
        self.radial_jitter = float(radial_jitter)
        self.register_buffer("periods", self._build_periods(self.num_pairs, self.head_dim, device=device))
        self.radii = nn.Parameter(torch.empty(num_heads, self.num_pairs, device=device, dtype=self.dtype))
        self.angles = nn.Parameter(torch.empty(num_heads, self.num_pairs, ndim - 1, device=device, dtype=self.dtype))

        # Real matrix products implement the small per-head transforms without
        # complex tensors, including under torch.compile and mixed precision.
        indices = torch.arange(self.num_pairs, device=device, dtype=self.dtype)
        phase = (2 * math.pi / self.num_pairs) * indices[:, None] * indices[None, :]
        self.register_buffer("_dft_cos", phase.cos() / math.sqrt(self.num_pairs), persistent=False)
        self.register_buffer("_dft_sin", phase.sin() / math.sqrt(self.num_pairs), persistent=False)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        self.periods.copy_(self._build_periods(self.num_pairs, self.head_dim, device=self.periods.device))
        self.radii.uniform_(1 - self.radial_jitter, 1 + self.radial_jitter)
        self.radii.mul_(self.periods.reciprocal())
        bins = torch.arange(self.num_pairs, device=self.periods.device, dtype=self.periods.dtype)
        azimuth = (bins + torch.rand_like(self.radii)) * (2 * math.pi / self.num_pairs)
        self.angles[..., 0].copy_(azimuth)
        if self.ndim == 3:
            # Uniform polar angles would concentrate directions at the poles.
            # Stratify cos(polar) instead, and independently permute its bins
            # per head so latitude is not correlated with azimuth or radius.
            for head in range(self.num_heads):
                order = torch.randperm(self.num_pairs, device=self.periods.device)
                z = 2 * (bins + torch.rand_like(bins)) / self.num_pairs - 1
                self.angles[head, :, 1].copy_(z[order].acos())

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"radii", "angles"}

    def _mix_frequencies(self, frequencies: Tensor) -> Tensor:
        if not self.use_dft:
            return frequencies
        if self.ndim == 2:
            x, y = frequencies.unbind(-1)
            # (C - iS) @ (x + iy), exactly fft(..., norm="ortho").
            return torch.stack((x @ self._dft_cos.T + y @ self._dft_sin.T,
                                y @ self._dft_cos.T - x @ self._dft_sin.T), dim=-1)
        return (self._dft_cos + self._dft_sin) @ frequencies

    def get_embed_from_coords(self, coords: RopeCoords) -> RopeEmbedding:
        with torch.autocast(device_type=coords.device.type, enabled=False):
            azimuth = self.angles[..., 0]
            if self.ndim == 2:
                directions = torch.stack((azimuth.cos(), azimuth.sin()), dim=-1)
            else:
                polar = self.angles[..., 1]
                directions = torch.stack((polar.sin() * azimuth.cos(),
                                          polar.sin() * azimuth.sin(), polar.cos()), dim=-1)
            frequencies = self._mix_frequencies(self.radii[..., None] * directions)
            phases = 2 * math.pi * torch.einsum("td,hpd->htp", coords.to(self.radii.dtype), frequencies)
            phases = phases.tile(2)
            return phases.sin(), phases.cos()

    def get_embed(self, shape: Sequence[int]) -> RopeEmbedding:
        return self.get_embed_from_coords(self.get_coords(shape))

    def forward(self, shape: Sequence[int]) -> RopeEmbedding:
        return self.get_embed(shape)
