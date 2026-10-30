"""Threshold task for zarr arrays.

Thresholds values and outputs a binary mask (0 or 255).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple
from urllib.parse import urlsplit
from urllib.request import url2pathname

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
class ThresholdConfig(TaskConfig):
    """Configuration for threshold task."""

    threshold: float = 127.0
    erase_blank: bool = False
    num_levels: int = 6


def _threshold_worker(args: Tuple) -> Tuple[Tuple[int, int], ...]:
    """Multiprocessing worker for thresholding chunks."""
    input_path, output_path, threshold, erase_blank, chunk_coords = args
    input_z = zarr.open(input_path, mode="r")
    output_z = zarr.open(output_path, mode="r+")

    slices = tuple(slice(start, stop) for start, stop in chunk_coords)
    chunk_data = input_z[slices]

    if erase_blank and len(np.unique(chunk_data)) < 5:
        output_z[slices] = np.zeros_like(chunk_data, dtype=np.uint8)
    else:
        thresholded = np.where(chunk_data > threshold, 255, 0).astype(np.uint8)
        output_z[slices] = thresholded

    return chunk_coords


def _validate_threshold_paths(input_path, output_path):
    """Keep this non-inplace operation outside its read-only local input tree."""
    if not output_path:
        raise ValueError("threshold requires a separate output path")
    source = str(input_path)
    # A Windows drive is a local path, including the accepted C:// spelling.
    is_drive = (len(source) >= 3 and source[0].isalpha() and source[1] == ":"
                and source[2] in "/\\")
    if "://" in source and not is_drive:
        parsed = urlsplit(source)
        if parsed.scheme != "file":
            # The creator writes a local Path; a remote input is a separate store.
            return
        source = url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            source = "//" + parsed.netloc + source
    source = Path(source).resolve()
    destination = Path(output_path).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("threshold output overlaps its input; use a separate output directory")


@register_task("threshold")
class ThresholdTask(ZarrTask):
    """Threshold zarr values to create a binary mask."""

    def __init__(self, config: ThresholdConfig):
        super().__init__(config)
        self.config: ThresholdConfig = config
        self._lvl0_path: str = ""
        self._axes_names = ["z", "y", "x"]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add threshold-specific arguments."""
        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            default=127,
            help="Threshold value (default: 127)",
        )
        parser.add_argument(
            "--erase-blank",
            action="store_true",
            help="Erase homogeneous chunks (less than 5 unique values) by setting them to 0",
        )
        parser.add_argument(
            "--num-levels",
            type=int,
            default=6,
            help="Number of pyramid levels to generate (default: 6)",
        )

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Reject implicit local in-place thresholding before overwrite prompts."""
        try:
            _validate_threshold_paths(args.input_zarr, args.output_zarr)
        except ValueError as error:
            raise SystemExit(f"Error: {error}") from None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ThresholdTask":
        """Create task from parsed arguments."""
        cls.validate_args(args)
        base_config = make_task_config(args)
        config = ThresholdConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            threshold=args.threshold,
            erase_blank=args.erase_blank,
            num_levels=getattr(args, "num_levels", 6),
        )
        return cls(config)

    def prepare(self) -> None:
        """Create output OME-Zarr structure."""
        _validate_threshold_paths(self.config.input_zarr, self.config.output_zarr)
        input_z = self._get_input_array(
            str(self.config.level) if self.config.level is not None else "0"
        )

        print(f"Input array shape: {input_z.shape}")
        print(f"Input array chunks: {input_z.chunks}")
        print(f"Input array dtype: {input_z.dtype}")
        print(
            f"Task: threshold | value={self.config.threshold} | "
            f"erase_blank={self.config.erase_blank}"
        )

        root_group_path = self.config.output_zarr
        self._lvl0_path = f"{root_group_path}/0"

        create_level_dataset(
            root_group_path,
            "0",
            input_z.shape,
            input_z.chunks,
            np.uint8,
            input_z.compressor,
        )

        self._input_shape = input_z.shape
        self._input_chunks = input_z.chunks

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items for parallel processing."""
        chunk_coords = get_chunk_coords(self._input_shape, self._input_chunks)

        for coords in chunk_coords:
            yield (
                self.config.input_zarr,
                self._lvl0_path,
                self.config.threshold,
                self.config.erase_blank,
                coords,
            )

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _threshold_worker(args)

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
            f"Thresholding complete. OME-Zarr pyramid saved to: {self.config.output_zarr}"
        )
