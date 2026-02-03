"""Merge task for zarr arrays.

Merges two OME-Zarr files by taking the maximum value at overlapping positions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import get_chunk_slices


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
    store1 = zarr.open(input1_path, mode="r")
    store2 = zarr.open(input2_path, mode="r")
    out_store = zarr.open(output_path, mode="r+")

    arr1 = store1["0"]
    arr2 = store2["0"]
    out_arr = out_store["0"]

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


def _downsample_chunk_worker(args: Tuple) -> str:
    """Downsample a single chunk from the previous level."""
    (
        out_chunk_idx,
        zarr_path,
        prev_level,
        out_level,
        prev_shape,
        out_shape,
        out_chunks,
        interpolation,
    ) = args
    cz, cy, cx = out_chunk_idx

    # Calculate output chunk bounds
    z_out_start = cz * out_chunks[0]
    y_out_start = cy * out_chunks[1]
    x_out_start = cx * out_chunks[2]
    z_out_end = min(z_out_start + out_chunks[0], out_shape[0])
    y_out_end = min(y_out_start + out_chunks[1], out_shape[1])
    x_out_end = min(x_out_start + out_chunks[2], out_shape[2])

    # Calculate corresponding input bounds (2x)
    z_in_start = z_out_start * 2
    y_in_start = y_out_start * 2
    x_in_start = x_out_start * 2
    z_in_end = min(z_out_end * 2, prev_shape[0])
    y_in_end = min(y_out_end * 2, prev_shape[1])
    x_in_end = min(x_out_end * 2, prev_shape[2])

    # Read only the required input region
    root = zarr.open(zarr_path, mode="r+")
    prev_arr = root[str(prev_level)]
    input_chunk = prev_arr[z_in_start:z_in_end, y_in_start:y_in_end, x_in_start:x_in_end]

    # Skip empty chunks
    if input_chunk.max() == 0:
        return "empty"

    # Output dimensions for this chunk
    out_z = z_out_end - z_out_start
    out_y = y_out_end - y_out_start
    out_x = x_out_end - x_out_start

    # Downsample using configured method
    if interpolation == "nearest":
        # Mode-based: most common non-zero value in each 2x2x2 block
        sz, sy, sx = input_chunk.shape
        pad_z = sz % 2
        pad_y = sy % 2
        pad_x = sx % 2
        if pad_z or pad_y or pad_x:
            input_chunk = np.pad(input_chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            sz, sy, sx = input_chunk.shape

        # Reshape to (n_blocks, 8)
        blocks = input_chunk.reshape(sz // 2, 2, sy // 2, 2, sx // 2, 2)
        blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(-1, 8)

        # Compute mode of non-zero values for each block
        result = np.zeros(blocks.shape[0], dtype=input_chunk.dtype)
        for i, vals in enumerate(blocks):
            nonzero = vals[vals != 0]
            if len(nonzero) > 0:
                unique, counts = np.unique(nonzero, return_counts=True)
                result[i] = unique[np.argmax(counts)]

        downsampled = result.reshape(sz // 2, sy // 2, sx // 2)
    else:  # average
        # Pad to even dimensions if needed
        sz, sy, sx = input_chunk.shape
        pad_z = sz % 2
        pad_y = sy % 2
        pad_x = sx % 2
        if pad_z or pad_y or pad_x:
            input_chunk = np.pad(input_chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='edge')
            sz, sy, sx = input_chunk.shape

        downsampled = input_chunk.reshape(sz // 2, 2, sy // 2, 2, sx // 2, 2).mean(axis=(1, 3, 5)).astype(input_chunk.dtype)

    # Ensure correct output size (handle edge cases)
    if downsampled.shape != (out_z, out_y, out_x):
        result = np.zeros((out_z, out_y, out_x), dtype=input_chunk.dtype)
        sz = min(downsampled.shape[0], out_z)
        sy = min(downsampled.shape[1], out_y)
        sx = min(downsampled.shape[2], out_x)
        result[:sz, :sy, :sx] = downsampled[:sz, :sy, :sx]
        downsampled = result

    # Write directly to output
    out_arr = root[str(out_level)]
    out_arr[
        z_out_start:z_out_end, y_out_start:y_out_end, x_out_start:x_out_end
    ] = downsampled

    return "written"


@register_task("merge")
class MergeTask(ZarrTask):
    """Merge two OME-Zarr files by taking the maximum at overlapping positions."""

    def __init__(self, config: MergeConfig):
        super().__init__(config)
        self.config: MergeConfig = config
        self._shape: Tuple[int, ...] = ()
        self._chunks: Tuple[int, ...] = ()
        self._dtype: np.dtype = np.dtype("uint8")
        self._compressor: Optional[Blosc] = None
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
        store1 = zarr.open(self.config.input_zarr, mode="r")
        store2 = zarr.open(self.config.input2_zarr, mode="r")

        arr1 = store1["0"]
        arr2 = store2["0"]

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

        # Create compressor
        self._compressor = Blosc(
            cname="zstd",
            clevel=self.config.compression_level,
            shuffle=Blosc.BITSHUFFLE,
        )

        # Create output zarr with level 0
        output_path = Path(self.config.output_zarr)
        print(f"\nCreating output zarr at {output_path}")

        root = zarr.open(str(output_path), mode="w")
        root.create_dataset(
            "0",
            shape=self._shape,
            chunks=self._chunks,
            dtype=self._dtype,
            compressor=self._compressor,
            fill_value=0,
            write_empty_chunks=False,
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

    def _generate_pyramid_levels(self) -> None:
        """Generate downsampled pyramid levels from level 0."""
        zarr_path = self.config.output_zarr
        root = zarr.open(zarr_path, mode="r+")

        for level in range(1, self.config.num_levels):
            prev_level = level - 1
            prev_arr = root[str(prev_level)]
            prev_shape = prev_arr.shape

            # Output shape is halved
            out_shape = (
                (prev_shape[0] + 1) // 2,
                (prev_shape[1] + 1) // 2,
                (prev_shape[2] + 1) // 2,
            )

            print(f"Level {level}: {prev_shape} -> {out_shape}")

            # Create output array
            root.create_dataset(
                str(level),
                shape=out_shape,
                chunks=self._chunks,
                dtype=self._dtype,
                compressor=self._compressor,
                overwrite=True,
            )

            # Generate chunk tasks
            tasks = []
            for cz in range((out_shape[0] + self._chunks[0] - 1) // self._chunks[0]):
                for cy in range(
                    (out_shape[1] + self._chunks[1] - 1) // self._chunks[1]
                ):
                    for cx in range(
                        (out_shape[2] + self._chunks[2] - 1) // self._chunks[2]
                    ):
                        tasks.append(
                            (
                                (cz, cy, cx),
                                zarr_path,
                                prev_level,
                                level,
                                prev_shape,
                                out_shape,
                                self._chunks,
                                self.config.interpolation,
                            )
                        )

            # Process chunks in parallel
            with Pool(processes=self.config.num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_downsample_chunk_worker, tasks),
                    total=len(tasks),
                    desc=f"Level {level}",
                ):
                    pass

    def _create_ome_zarr_metadata(self) -> dict:
        """Create OME-Zarr multiscales metadata."""
        datasets = []
        for level in range(self.config.num_levels):
            scale = 2**level
            datasets.append(
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [float(scale), float(scale), float(scale)],
                        }
                    ],
                }
            )

        return {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": "merged",
                    "axes": [
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": datasets,
                    "type": self.config.interpolation,
                    "metadata": {
                        "method": f"{self.config.interpolation} 2x downsampling",
                    },
                }
            ]
        }

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
        if self.config.num_levels > 1:
            print(f"\nGenerating {self.config.num_levels - 1} pyramid levels...")
            self._generate_pyramid_levels()

        # Write OME-Zarr metadata
        print("\nWriting OME-Zarr metadata...")
        output_path = Path(self.config.output_zarr)
        metadata = self._create_ome_zarr_metadata()
        attrs_path = output_path / ".zattrs"
        with open(attrs_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Copy metadata.json if it exists in input1
        metadata_src = Path(self.config.input_zarr) / "metadata.json"
        if metadata_src.exists():
            import shutil

            metadata_dst = output_path / "metadata.json"
            shutil.copy2(metadata_src, metadata_dst)
            print("Copied metadata.json from input1")

        print(f"\nDone! Output saved to {self.config.output_zarr}")
