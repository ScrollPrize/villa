"""Transpose task for zarr arrays.

Transposes array axes according to a specified permutation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import numpy as np
import zarr

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import (
    build_pyramid,
    create_level_dataset,
    get_chunk_coords,
    inverse_permutation,
    write_multiscales_metadata,
)


@dataclass
class TransposeConfig(TaskConfig):
    """Configuration for transpose task."""

    transpose_order: str = "xzy"
    num_levels: int = 6


def _transpose_worker(args: Tuple) -> Tuple[Tuple[int, int], ...]:
    """Multiprocessing worker for transposing chunks."""
    input_path, output_path, perm, out_chunk_coords = args
    in_z = zarr.open(input_path, mode="r")
    out_z = zarr.open(output_path, mode="r+")

    out_slices = tuple(slice(start, stop) for start, stop in out_chunk_coords)

    inv = inverse_permutation(perm)
    in_slices = [None] * in_z.ndim
    for in_ax in range(in_z.ndim):
        out_ax = inv[in_ax]
        o_start, o_stop = out_chunk_coords[out_ax]
        in_slices[in_ax] = slice(o_start, o_stop)

    in_block = in_z[tuple(in_slices)]
    out_block = np.transpose(in_block, axes=perm)

    out_z[out_slices] = out_block

    return out_chunk_coords


@register_task("transpose")
class TransposeTask(ZarrTask):
    """Transpose zarr array axes."""

    def __init__(self, config: TransposeConfig):
        super().__init__(config)
        self.config: TransposeConfig = config
        self._lvl0_path: str = ""
        self._axes_names: List[str] = []
        self._out_shape: Tuple[int, ...] = ()
        self._out_chunks: Tuple[int, ...] = ()
        self._perm: List[int] = []

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add transpose-specific arguments."""
        parser.add_argument(
            "--transpose-order",
            type=str,
            default="xzy",
            help="Output axes order for transpose, as a permutation of xyz (default: xzy)",
        )
        parser.add_argument(
            "--num-levels",
            type=int,
            default=6,
            help="Number of pyramid levels to generate (default: 6)",
        )

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Validate transpose arguments."""
        order_str = args.transpose_order.lower()
        if sorted(order_str) != ["x", "y", "z"] or len(order_str) != 3:
            raise SystemExit(
                "Error: --transpose-order must be a permutation of xyz, e.g., xzy"
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TransposeTask":
        """Create task from parsed arguments."""
        cls.validate_args(args)
        base_config = make_task_config(args)
        config = TransposeConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            transpose_order=args.transpose_order,
            num_levels=getattr(args, "num_levels", 6),
        )
        return cls(config)

    def prepare(self) -> None:
        """Create output OME-Zarr structure with transposed shape."""
        input_z = self._get_input_array(
            str(self.config.level) if self.config.level is not None else "0"
        )

        # Parse transpose order string to permutation
        valid_axes = {"x": 2, "y": 1, "z": 0}
        order_str = self.config.transpose_order.lower()

        # perm is list such that out axis i = in axis perm[i]
        self._perm = [valid_axes[c] for c in order_str]

        # Compute output shape/chunks by permuting input dims
        self._out_shape = tuple(input_z.shape[p] for p in self._perm)
        self._out_chunks = tuple(input_z.chunks[p] for p in self._perm)

        # Determine axes names in output order
        axis_name_for_in = ["z", "y", "x"]
        self._axes_names = [axis_name_for_in[p] for p in self._perm]

        print(f"Input array shape: {input_z.shape}")
        print(f"Input array chunks: {input_z.chunks}")
        print(f"Input array dtype: {input_z.dtype}")
        print(f"Task: transpose | order={order_str} | perm={self._perm}")
        print(f"Output array shape: {self._out_shape}")
        print(f"Output array chunks: {self._out_chunks}")

        root_group_path = self.config.output_zarr
        self._lvl0_path = f"{root_group_path}/0"

        create_level_dataset(
            root_group_path,
            "0",
            self._out_shape,
            self._out_chunks,
            input_z.dtype,
            input_z.compressor,
        )

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items for parallel processing."""
        out_chunk_coords = get_chunk_coords(self._out_shape, self._out_chunks)

        for coords in out_chunk_coords:
            yield (
                self.config.input_zarr,
                self._lvl0_path,
                self._perm,
                coords,
            )

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _transpose_worker(args)

    def finalize(self) -> None:
        """Build pyramid levels and write metadata."""
        root_group_path = self.config.output_zarr

        build_pyramid(
            root_group_path,
            self._axes_names,
            num_levels=self.config.num_levels,
            num_workers=self.config.num_workers,
        )
        write_multiscales_metadata(
            root_group_path, self._axes_names, num_levels=self.config.num_levels
        )

        print(
            f"Transpose complete. OME-Zarr pyramid saved to: {self.config.output_zarr}"
        )
