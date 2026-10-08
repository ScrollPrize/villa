from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def ceil_div(value: int, divisor: int) -> int:
    value = int(value)
    divisor = int(divisor)
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value >= 0 and divisor > 0")
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class IsotropicOutputLattice:
    """Stride-aligned output lattice for padded odd-kernel 3D convolutions."""

    stride: int

    def __post_init__(self) -> None:
        if int(self.stride) <= 0:
            raise ValueError("output stride must be > 0")

    def shape(self, input_shape: Sequence[int]) -> tuple[int, ...]:
        return tuple(ceil_div(int(size), int(self.stride)) for size in input_shape)

    def residual_to(self, total_stride: int) -> int:
        total_stride = int(total_stride)
        if total_stride <= 0 or total_stride % int(self.stride):
            raise ValueError(
                f"total output stride {total_stride} must be a positive multiple of "
                f"native output stride {self.stride}"
            )
        return total_stride // int(self.stride)


def cumulative_isotropic_strides(
    stage_strides: Iterable[Sequence[int]],
) -> tuple[int, ...]:
    cumulative = [1, 1, 1]
    resolved: list[int] = []
    for stage_stride in stage_strides:
        values = tuple(int(value) for value in stage_stride)
        if len(values) != 3 or any(value <= 0 for value in values):
            raise ValueError("encoder stage strides must contain three positive values")
        cumulative = [left * right for left, right in zip(cumulative, values)]
        if not (cumulative[0] == cumulative[1] == cumulative[2]):
            raise ValueError(
                "output_stride currently requires isotropic cumulative encoder strides"
            )
        resolved.append(cumulative[0])
    return tuple(resolved)


def output_stage_index(stage_strides: Iterable[Sequence[int]], output_stride: int) -> int:
    output_stride = int(output_stride)
    if output_stride <= 0:
        raise ValueError("output_stride must be > 0")
    cumulative = cumulative_isotropic_strides(stage_strides)
    if output_stride == 1:
        return 0
    try:
        return cumulative.index(output_stride)
    except ValueError as exc:
        raise ValueError(
            f"output_stride={output_stride} is not represented by encoder "
            f"resolutions {list(cumulative)}"
        ) from exc


__all__ = [
    "IsotropicOutputLattice",
    "ceil_div",
    "cumulative_isotropic_strides",
    "output_stage_index",
]
