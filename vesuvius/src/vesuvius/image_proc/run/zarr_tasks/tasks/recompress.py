"""Recompress task for zarr arrays.

Recompresses chunks with a new compressor (default: zstd level 1).
Supports in-place recompression with optional corrupt chunk handling.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from multiprocessing import Pool

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import (
    build_pyramid,
    create_level_dataset,
    delete_chunk_file,
    get_chunk_coords,
    write_multiscales_metadata,
)


@dataclass
class RecompressConfig(TaskConfig):
    """Configuration for recompress task."""

    skip_corrupt: bool = False
    compression_level: int = 1
    num_levels: int = 6


def _recompress_worker(args: Tuple) -> Tuple[Tuple[Tuple[int, int], ...], bool]:
    """Multiprocessing worker for recompressing chunks."""
    input_path, output_path, chunk_coords, delete_source_path, chunks, skip_corrupt = (
        args
    )
    input_z = zarr.open(input_path, mode="r")
    output_z = zarr.open(output_path, mode="r+")

    slices = tuple(slice(start, stop) for start, stop in chunk_coords)
    corrupted = False

    try:
        chunk_data = input_z[slices]
        output_z[slices] = chunk_data
    except (ValueError, Exception) as e:
        if skip_corrupt:
            # Fill with zeros for corrupted chunks
            expected_shape = tuple(stop - start for start, stop in chunk_coords)
            output_z[slices] = np.zeros(expected_shape, dtype=output_z.dtype)
            corrupted = True
        else:
            raise

    if delete_source_path is not None and chunks is not None:
        delete_chunk_file(delete_source_path, chunk_coords, chunks)

    return (chunk_coords, corrupted)


@register_task("recompress")
class RecompressTask(ZarrTask):
    """Recompress zarr array with a new compressor."""

    def __init__(self, config: RecompressConfig):
        super().__init__(config)
        self.config: RecompressConfig = config
        self._lvl0_path: str = ""
        self._axes_names = ["z", "y", "x"]
        self._compressor: Optional[Blosc] = None
        self._is_ome_zarr: bool = False
        self._levels_to_process: List[str] = []
        self._all_corrupted_chunks: List[Tuple[str, Tuple]] = []

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add recompress-specific arguments."""
        parser.add_argument(
            "--skip-corrupt",
            action="store_true",
            help="Skip corrupted chunks instead of failing (fills with zeros)",
        )
        parser.add_argument(
            "--compression-level",
            type=int,
            default=1,
            choices=range(1, 10),
            help="Blosc zstd compression level (1-9, default: 1)",
        )
        parser.add_argument(
            "--num-levels",
            type=int,
            default=6,
            help="Number of pyramid levels to generate (default: 6)",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RecompressTask":
        """Create task from parsed arguments."""
        base_config = make_task_config(args)
        config = RecompressConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            skip_corrupt=args.skip_corrupt,
            compression_level=getattr(args, "compression_level", 1),
            num_levels=getattr(args, "num_levels", 6),
        )
        return cls(config)

    def prepare(self) -> None:
        """Detect OME-Zarr structure and prepare compressor."""
        self._compressor = Blosc(
            cname="zstd",
            clevel=self.config.compression_level,
            shuffle=Blosc.BITSHUFFLE,
        )

        # Detect OME-Zarr structure
        self._is_ome_zarr, all_levels = self._detect_ome_zarr(self.config.input_zarr)

        if self._is_ome_zarr:
            if self.config.level is not None:
                self._levels_to_process = [str(self.config.level)]
                if str(self.config.level) not in all_levels:
                    raise SystemExit(
                        f"Error: Level {self.config.level} not found. "
                        f"Available levels: {all_levels}"
                    )
            else:
                self._levels_to_process = all_levels

            print(
                f"Detected OME-Zarr with {len(all_levels)} resolution levels: {all_levels}"
            )
            if len(self._levels_to_process) > 1:
                print(f"Will process levels: {self._levels_to_process}")
        else:
            self._levels_to_process = [None]

        input_z = self._get_input_array(
            self._levels_to_process[0] if self._is_ome_zarr else None
        )

        print(f"Input array shape: {input_z.shape}")
        print(f"Input array chunks: {input_z.chunks}")
        print(f"Input array dtype: {input_z.dtype}")
        print(f"Task: recompress | zstd level {self.config.compression_level}")
        if self.config.skip_corrupt:
            print("  --skip-corrupt enabled: corrupted chunks will be filled with zeros")

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items - handled in custom run() for recompress."""
        # Recompress uses custom run() method
        return []

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _recompress_worker(args)

    def run(self) -> None:
        """Execute recompression with special handling for inplace mode."""
        self.prepare()

        if self.config.inplace:
            self._run_inplace()
        else:
            self._run_to_output()

        self.finalize()

    def _run_inplace(self) -> None:
        """Run in-place recompression."""
        for level in self._levels_to_process:
            if self._is_ome_zarr:
                level_path = f"{self.config.input_zarr}/{level}"
                print(f"\nProcessing level {level}...")
            else:
                level_path = self.config.input_zarr

            temp_path = f"{level_path}_tmp"

            # Read original zarr
            read_z = zarr.open(level_path, mode="r")
            print(f"  Shape: {read_z.shape}, Chunks: {read_z.chunks}")

            # Create temp zarr with new compressor
            temp_z = zarr.open(
                temp_path,
                mode="w",
                shape=read_z.shape,
                chunks=read_z.chunks,
                dtype=read_z.dtype,
                compressor=self._compressor,
            )

            # Copy .zattrs if it exists
            zattrs_path = Path(level_path) / ".zattrs"
            if zattrs_path.exists():
                shutil.copy2(zattrs_path, Path(temp_path) / ".zattrs")

            chunk_coords = get_chunk_coords(read_z.shape, read_z.chunks)
            chunks = read_z.chunks
            print(f"  Total chunks to process: {len(chunk_coords)}")

            # Copy chunks to temp and delete from original
            work_items = [
                (
                    level_path,
                    temp_path,
                    coords,
                    level_path,
                    chunks,
                    self.config.skip_corrupt,
                )
                for coords in chunk_coords
            ]

            desc = (
                f"Recompressing level {level}"
                if self._is_ome_zarr
                else "Recompressing chunks in-place"
            )
            level_corrupted = []
            try:
                with Pool(processes=self.config.num_workers) as pool:
                    for result in tqdm(
                        pool.imap_unordered(_recompress_worker, work_items),
                        total=len(work_items),
                        desc=desc,
                    ):
                        coords, corrupted = result
                        if corrupted:
                            level_corrupted.append((level, coords))

                # Replace original with temp
                shutil.rmtree(level_path)
                os.rename(temp_path, level_path)

                if level_corrupted:
                    self._all_corrupted_chunks.extend(level_corrupted)
                    print(
                        f"  Warning: {len(level_corrupted)} corrupted chunks filled with zeros"
                    )
            except Exception as e:
                # Clean up temp on failure
                if Path(temp_path).exists():
                    shutil.rmtree(temp_path)
                raise RuntimeError(f"Recompression failed for {level_path}: {e}") from e

    def _run_to_output(self) -> None:
        """Run recompression to a new output location."""
        input_z = self._get_input_array(
            self._levels_to_process[0] if self._is_ome_zarr else None
        )

        root_group_path = self.config.output_zarr
        self._lvl0_path = f"{root_group_path}/0"

        create_level_dataset(
            root_group_path,
            "0",
            input_z.shape,
            input_z.chunks,
            input_z.dtype,
            self._compressor,
        )

        # Process chunks
        chunk_coords = get_chunk_coords(input_z.shape, input_z.chunks)
        print(f"Total chunks to process: {len(chunk_coords)}")

        work_items = [
            (
                self.config.input_zarr,
                self._lvl0_path,
                coords,
                None,
                None,
                self.config.skip_corrupt,
            )
            for coords in chunk_coords
        ]

        corrupted_chunks = []
        with Pool(processes=self.config.num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_recompress_worker, work_items),
                total=len(work_items),
                desc="Recompressing chunks",
            ):
                coords, corrupted = result
                if corrupted:
                    corrupted_chunks.append(coords)

        if corrupted_chunks:
            self._all_corrupted_chunks = [(None, c) for c in corrupted_chunks]

        # Build pyramid and write metadata
        build_pyramid(
            root_group_path,
            self._axes_names,
            num_levels=self.config.num_levels,
            num_workers=self.config.num_workers,
        )
        write_multiscales_metadata(
            root_group_path, self._axes_names, num_levels=self.config.num_levels
        )

    def finalize(self) -> None:
        """Print completion message and corrupted chunk summary."""
        if self.config.inplace:
            print(f"In-place recompression complete: {self.config.input_zarr}")
        else:
            print(
                f"Recompression complete. OME-Zarr pyramid saved to: {self.config.output_zarr}"
            )

        if self._all_corrupted_chunks:
            print(
                f"\nSummary: {len(self._all_corrupted_chunks)} corrupted chunks "
                "were skipped and filled with zeros:"
            )
            for level, coords in self._all_corrupted_chunks:
                if level is not None:
                    print(f"  Level {level}: {coords}")
                else:
                    print(f"  {coords}")
