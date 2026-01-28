"""Scale task for zarr arrays.

Scales arrays by a uniform factor using nearest-neighbor interpolation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import numpy as np
import zarr

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import (
    build_pyramid,
    create_level_dataset,
    get_chunk_coords,
    write_multiscales_metadata,
)


@dataclass
class ScaleConfig(TaskConfig):
    """Configuration for scale task."""

    scale_factor: float = 1.0
    num_levels: int = 6


def _scale_worker(args: Tuple) -> Tuple[Tuple[int, int], ...]:
    """Multiprocessing worker for scaling chunks."""
    input_path, output_path, scale, out_chunk_coords = args
    in_z = zarr.open(input_path, mode="r")
    out_z = zarr.open(output_path, mode="r+")

    out_slices = tuple(slice(start, stop) for start, stop in out_chunk_coords)

    in_slices = []
    for (o_start, o_stop), dim in zip(out_chunk_coords, range(in_z.ndim)):
        in_start = int(np.floor(o_start / scale))
        in_stop = int(np.ceil(o_stop / scale))
        in_start = max(0, in_start)
        in_stop = min(in_z.shape[dim], max(in_start + 1, in_stop))
        in_slices.append(slice(in_start, in_stop))

    in_block = in_z[tuple(in_slices)]

    idx_list = []
    for ax, (o_start, o_stop), s in zip(
        range(in_z.ndim), out_chunk_coords, [scale] * in_z.ndim
    ):
        o_coords = np.arange(o_start, o_stop, dtype=np.float64)
        in_idx = np.floor(o_coords / s).astype(np.int64) - in_slices[ax].start
        in_idx[in_idx < 0] = 0
        max_valid = in_block.shape[ax] - 1
        if max_valid >= 0:
            in_idx[in_idx > max_valid] = max_valid
        idx_list.append(in_idx)

    ix = np.ix_(*idx_list)
    out_block = in_block[ix]

    out_z[out_slices] = out_block

    return out_chunk_coords


@register_task("scale")
class ScaleTask(ZarrTask):
    """Scale zarr array by a uniform factor."""

    def __init__(self, config: ScaleConfig):
        super().__init__(config)
        self.config: ScaleConfig = config
        self._lvl0_path: str = ""
        self._axes_names = ["z", "y", "x"]
        self._out_shape: Tuple[int, ...] = ()
        self._out_chunks: Tuple[int, ...] = ()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add scale-specific arguments."""
        parser.add_argument(
            "--scale-factor",
            type=float,
            default=None,
            help="Uniform scale factor for all dimensions",
        )
        parser.add_argument(
            "--num-levels",
            type=int,
            default=6,
            help="Number of pyramid levels to generate (default: 6)",
        )

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Validate scale arguments."""
        if args.scale_factor is None or args.scale_factor <= 0:
            raise SystemExit(
                "Error: --scale-factor must be provided and > 0 for task=scale"
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ScaleTask":
        """Create task from parsed arguments."""
        cls.validate_args(args)
        base_config = make_task_config(args)
        config = ScaleConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            scale_factor=args.scale_factor,
            num_levels=getattr(args, "num_levels", 6),
        )
        return cls(config)

    def prepare(self) -> None:
        """Create output OME-Zarr structure with scaled shape."""
        input_z = self._get_input_array(
            str(self.config.level) if self.config.level is not None else "0"
        )

        sf = self.config.scale_factor
        self._out_shape = tuple(max(1, int(round(d * sf))) for d in input_z.shape)
        self._out_chunks = tuple(
            min(c, s) for c, s in zip(input_z.chunks, self._out_shape)
        )

        print(f"Input array shape: {input_z.shape}")
        print(f"Input array chunks: {input_z.chunks}")
        print(f"Input array dtype: {input_z.dtype}")
        print(f"Task: scale | factor={sf}")
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
                self.config.scale_factor,
                coords,
            )

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _scale_worker(args)

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

        print(f"Scaling complete. OME-Zarr pyramid saved to: {self.config.output_zarr}")
