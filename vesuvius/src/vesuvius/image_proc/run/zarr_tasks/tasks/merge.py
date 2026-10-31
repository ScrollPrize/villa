"""Merge task for zarr arrays.

Merges two OME-Zarr files by taking the maximum value at overlapping positions.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from vc_reader.reader import create_group, ome_multiscales, write_pyramid

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import get_chunk_slices, open_input, open_output


@dataclass
class MergeConfig(TaskConfig):
    """Configuration for merge task."""

    input2_zarr: str = ""
    num_levels: int = 6
    compression_level: int = 3
    interpolation: str = "average"  # "average" or "nearest"


def _merge_chunk_worker(args: Tuple) -> str:
    """Process a single chunk: read both, take maximum, write output.

    Returns:
        'empty': neither input had data
        'copy1': only input1 had data
        'copy2': only input2 had data
        'merge': both had data, merged with maximum
    """
    chunk_idx, chunk_slice, input1_path, input2_path, output_path = args

    # Open stores (each worker opens its own handle)
    arr1 = open_input(f"{input1_path}/0")
    arr2 = open_input(f"{input2_path}/0")
    out_arr = open_output(f"{output_path}/0")

    # Read both chunks
    data1 = arr1[chunk_slice]
    data2 = arr2[chunk_slice]

    # Handle sparse data efficiently
    has_data1 = np.any(data1 != 0)
    has_data2 = np.any(data2 != 0)

    if not has_data1 and not has_data2:
        return "empty"
    elif has_data1 and not has_data2:
        out_arr[chunk_slice] = data1
        return "copy1"
    elif has_data2 and not has_data1:
        out_arr[chunk_slice] = data2
        return "copy2"
    else:
        out_arr[chunk_slice] = np.maximum(data1, data2)
        return "merge"


@register_task("merge")
class MergeTask(ZarrTask):
    """Merge two OME-Zarr files by taking the maximum at overlapping positions."""

    def __init__(self, config: MergeConfig):
        super().__init__(config)
        self.config: MergeConfig = config
        self._shape: Tuple[int, ...] = ()
        self._chunks: Tuple[int, ...] = ()
        self._dtype: np.dtype = np.dtype("uint8")
        self._merge_stats: Counter = Counter()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add merge-specific arguments."""
        parser.add_argument(
            "--input2",
            type=str,
            required=True,
            help="Path to second input OME-Zarr",
        )
        parser.add_argument(
            "--num-levels",
            type=int,
            default=6,
            help="Number of pyramid levels to generate (default: 6)",
        )
        parser.add_argument(
            "--compression-level",
            type=int,
            default=3,
            choices=range(1, 10),
            help="Blosc compression level (1-9, default: 3)",
        )
        parser.add_argument(
            "--interpolation",
            type=str,
            default="average",
            choices=["average", "nearest"],
            help="Downsampling method: 'average' for intensity data, 'nearest' (mode of non-zero) for labels (default: average)",
        )

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Validate merge arguments."""
        if not args.input2:
            raise SystemExit("Error: --input2 is required for merge task")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MergeTask":
        """Create task from parsed arguments."""
        cls.validate_args(args)
        base_config = make_task_config(args)
        config = MergeConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            input2_zarr=args.input2,
            num_levels=getattr(args, "num_levels", 6),
            compression_level=getattr(args, "compression_level", 3),
            interpolation=getattr(args, "interpolation", "average"),
        )
        return cls(config)

    def _validate_compatibility(self) -> None:
        """Verify shapes match between inputs."""
        arr1 = open_input(f"{self.config.input_zarr}/0")
        arr2 = open_input(f"{self.config.input2_zarr}/0")

        if arr1.shape != arr2.shape:
            raise ValueError(
                f"Shape mismatch: input1 has shape {arr1.shape}, input2 has shape {arr2.shape}. "
                f"Use resize_zarr.py to match shapes before merging."
            )

        if arr1.dtype != arr2.dtype:
            raise ValueError(
                f"Dtype mismatch: input1 has dtype {arr1.dtype}, input2 has dtype {arr2.dtype}."
            )

        self._shape = arr1.shape
        self._chunks = arr1.chunks
        self._dtype = arr1.dtype

    def prepare(self) -> None:
        """Validate inputs and create output zarr."""
        print(f"Input 1: {self.config.input_zarr}")
        print(f"Input 2: {self.config.input2_zarr}")
        print(f"Output: {self.config.output_zarr}")
        print(f"Workers: {self.config.num_workers}")
        print(f"Pyramid levels: {self.config.num_levels}")
        print(f"Compression level: {self.config.compression_level}")

        print("\nValidating input compatibility...")
        self._validate_compatibility()
        print(f"Shape: {self._shape}")
        print(f"Chunks: {self._chunks}")
        print(f"Dtype: {self._dtype}")

        # Create output zarr with level 0
        output_path = Path(self.config.output_zarr)
        print(f"\nCreating output zarr at {output_path}")

        root = create_group(str(output_path), overwrite=True, auth=None)
        root.create_array(
            "0",
            shape=self._shape,
            chunks=self._chunks,
            dtype=self._dtype,
            cname="zstd",
            level=self.config.compression_level,
            shuffle=2,
            fill_value=0,
        )

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items for parallel processing."""
        chunk_slices = get_chunk_slices(self._shape, self._chunks)

        for idx, chunk_slice in enumerate(chunk_slices):
            yield (
                idx,
                chunk_slice,
                self.config.input_zarr,
                self.config.input2_zarr,
                self.config.output_zarr,
            )

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _merge_chunk_worker(args)

    @property
    def _reduction(self) -> str:
        # "nearest" here has always meant the mode of each block's nonzero values, for labels.
        return "mode" if self.config.interpolation == "nearest" else "mean"

    def finalize(self) -> None:
        """Build pyramid and write metadata."""
        # Collect merge statistics
        results = getattr(self, "_results", [])
        self._merge_stats = Counter(results)

        print(f"\nMerge statistics:")
        print(f"  Empty (both inputs): {self._merge_stats['empty']}")
        print(f"  Copied from input1: {self._merge_stats['copy1']}")
        print(f"  Copied from input2: {self._merge_stats['copy2']}")
        print(f"  Merged (maximum): {self._merge_stats['merge']}")

        # Generate pyramid levels
        levels = self.config.num_levels - 1
        if levels > 0:
            print(f"\nGenerating {levels} pyramid levels...")
            write_pyramid(
                self.config.output_zarr,
                levels,
                reduction=self._reduction,
                threads=self.config.num_workers,
                overwrite=True,
            )

        # Write OME-Zarr metadata
        print("\nWriting OME-Zarr metadata...")
        output_path = Path(self.config.output_zarr)
        create_group(str(output_path), auth=None).attrs["multiscales"] = [
            ome_multiscales(name="merged", levels=levels, unit="micrometer", reduction=self._reduction)
        ]

        # Copy metadata.json if it exists in input1
        metadata_src = Path(self.config.input_zarr) / "metadata.json"
        if metadata_src.exists():
            metadata_dst = output_path / "metadata.json"
            shutil.copy2(metadata_src, metadata_dst)
            print("Copied metadata.json from input1")

        print(f"\nDone! Output saved to {self.config.output_zarr}")
