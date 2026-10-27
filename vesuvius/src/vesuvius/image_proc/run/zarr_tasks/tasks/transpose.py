"""Transpose task for zarr arrays.

Transposes array axes according to a specified permutation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
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
    read_level_scale,
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


def _transpose_metadata(group, array_path, perm):
    """Keep the selected dataset's coordinate system when axes are reordered."""
    multiscales = group.attrs.get("multiscales", [])
    if not multiscales:
        return None
    matches = [(ms, ds) for ms in multiscales for ds in ms.get("datasets", [])
               if ds.get("path", "").strip("/") == array_path.strip("/")]
    if len(matches) != 1:
        raise ValueError("Transpose needs one multiscales entry for the selected array")
    ms, dataset = matches[0]
    axes = ms.get("axes", [])
    if len(axes) != 3 or {a.get("name") for a in axes} != {"z", "y", "x"}:
        raise ValueError("Transpose supports three named spatial axes: z, y, x")
    if any(a.get("type", "space") != "space" for a in axes):
        raise ValueError("Transpose requires spatial axes")

    def reordered(transforms, required):
        if required and (not transforms or transforms[0].get("type") != "scale"):
            raise ValueError("Dataset coordinates must start with a scale transform")
        result = []
        for transform in transforms:
            kind = transform.get("type")
            if kind not in {"scale", "translation"}:
                raise ValueError("Transpose supports scale and translation transforms only")
            values = transform.get(kind)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError("Coordinate transform must have three values")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and np.isfinite(v) for v in values):
                raise ValueError("Coordinate transform values must be finite numbers")
            copied = deepcopy(transform)
            copied[kind] = [values[p] for p in perm]
            result.append(copied)
        return result

    result = deepcopy(ms)
    # The output pyramid uses nearest-neighbor decimation, not the source filter.
    result.pop("type", None)
    result.pop("metadata", None)
    result["axes"] = [deepcopy(axes[p]) for p in perm]
    result["datasets"] = []
    if "coordinateTransformations" in ms:
        result["coordinateTransformations"] = reordered(ms["coordinateTransformations"], False)
    transformed = reordered(dataset.get("coordinateTransformations"), True)
    return result, transformed


def _transpose_spec(input_path, level, order, ndim, num_levels):
    """One read-only named-axis/frame contract for planning and preparation."""
    order = order.lower()
    if ndim != 3 or num_levels < 1:
        raise ValueError("Transpose requires a 3D array and at least one output level")
    if len(order) != 3 or sorted(order) != ["x", "y", "z"]:
        raise ValueError("Transpose order must be a permutation of xyz")
    opened = zarr.open(input_path, mode="r")
    group, array_path = None, ""
    if isinstance(opened, zarr.Group):
        group, array_path = opened, level
    else:
        parent = Path(input_path).parent
        if (parent / ".zgroup").is_file() or (parent / "zarr.json").is_file():
            group = zarr.open_group(str(parent), mode="r")
            array_path = Path(input_path).name
    axis_names = ["z", "y", "x"]
    if group is not None and group.attrs.get("multiscales"):
        matches = [ms for ms in group.attrs["multiscales"]
                   if any(ds.get("path", "").strip("/") == array_path.strip("/")
                          for ds in ms.get("datasets", []))]
        if len(matches) != 1:
            raise ValueError("Transpose needs one multiscales entry for the selected array")
        axes = matches[0].get("axes", [])
        if len(axes) != 3 or {a.get("name") for a in axes} != {"z", "y", "x"}:
            raise ValueError("Transpose supports three named spatial axes: z, y, x")
        axis_names = [a["name"] for a in axes]
    perm = [axis_names.index(c) for c in order]
    metadata = _transpose_metadata(group, array_path, perm) if group is not None else None
    return perm, [axis_names[p] for p in perm], metadata


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
        self._metadata = None

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
    def planned_output(cls, args, shape, dtype):
        level = str(args.level) if args.level is not None else "0"
        perm, _, _ = _transpose_spec(args.input_zarr, level, args.transpose_order,
                                     len(shape), args.num_levels)
        return tuple(shape[p] for p in perm), dtype

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
        level = str(self.config.level) if self.config.level is not None else "0"
        input_z = self._get_input_array(level)
        self._input_path = self._input_array_path(level)
        self._source_scale = read_level_scale(self.config.input_zarr, level)

        self._perm, self._axes_names, self._metadata = _transpose_spec(
            self.config.input_zarr, level, self.config.transpose_order,
            input_z.ndim, self.config.num_levels,
        )
        order_str = self.config.transpose_order.lower()
        self._out_shape = tuple(input_z.shape[p] for p in self._perm)
        self._out_chunks = tuple(input_z.chunks[p] for p in self._perm)

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
                self._input_path,
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
        if self._metadata is None:
            base_scale, units = self._source_scale or (None, None)
            if base_scale is not None:
                base_scale = [base_scale[p] for p in self._perm]
                units = [units[p] for p in self._perm]
            write_multiscales_metadata(
                root_group_path, self._axes_names, num_levels=self.config.num_levels,
                base_scale=base_scale, units=units,
            )

        else:
            metadata, transforms = self._metadata
            for level in range(self.config.num_levels):
                level_transforms = deepcopy(transforms)
                level_transforms[0]["scale"] = [
                    value * 2**level for value in transforms[0]["scale"]
                ]
                metadata["datasets"].append({
                    "path": str(level), "coordinateTransformations": level_transforms
                })
            root = zarr.open_group(root_group_path, mode="a")
            root.attrs["multiscales"] = [metadata]
            for level in range(self.config.num_levels):
                root[str(level)].attrs["_ARRAY_DIMENSIONS"] = self._axes_names

        print(
            f"Transpose complete. OME-Zarr pyramid saved to: {self.config.output_zarr}"
        )
