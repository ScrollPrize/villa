#!/usr/bin/env python3
"""Resize multiscale Zarr volumes to a target spatial shape with padding.

This utility pads (or crops) OME-Zarr multiscale volumes so that their spatial
axes match a user-provided shape. Existing data is positioned using the
translation metadata from the multiscales coordinate transformations. Any
additional (non-spatial) axes are left unchanged. Padding uses a configurable
fill value (default 0).

The script copies the source Zarr hierarchy into a new output store so the
original data remains untouched. Data movement is done chunk-wise with Dask
running on the multiprocessing scheduler; the Dask progress bar is enabled for
visibility. At no point is the full array materialized in memory.

Example:
    python resize_zarr_spatial.py \\
        --input /path/to/Scroll1_8um_surface_frame.zarr \\
        --output /path/to/Scroll1_8um_surface_frame_full.zarr \\
        --target-shape 14376 7888 8096
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import dask.array as da
from dask.diagnostics import ProgressBar
import dask
import numpy as np
import zarr


@dataclass
class AxisLayout:
    names: Tuple[str, ...]
    spatial_indices: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize OME-Zarr multiscale volumes to a new spatial shape")
    parser.add_argument("--input", required=True, help="Path to the source Zarr directory")
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Desired spatial shape (full-resolution) as three integers",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Destination Zarr directory (will be created). Defaults to <input>_resized.zarr",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Fill value used for padded regions (default: 0.0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for Dask (default: Dask/processes default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned operations without writing data",
    )
    return parser.parse_args()


def load_multiscales(root: zarr.Group) -> Tuple[Dict, AxisLayout]:
    attrs = root.attrs.asdict()
    multiscales = attrs.get("multiscales")
    if not multiscales:
        raise ValueError("No 'multiscales' metadata found in root attributes")

    multiscale = multiscales[0]
    axes = multiscale.get("axes")
    if not axes:
        raise ValueError("Multiscales entry is missing 'axes' metadata")

    axis_names = tuple(str(axis["name"]) for axis in axes)
    spatial_names = {"z", "y", "x"}
    spatial_indices = {}
    for name in spatial_names:
        if name not in axis_names:
            raise ValueError(f"Required axis '{name}' not present in multiscales axes {axis_names}")
        spatial_indices[name] = axis_names.index(name)

    return attrs, AxisLayout(names=axis_names, spatial_indices=spatial_indices)


def compute_scale_and_translation(dataset_meta: Dict, ndim: int) -> Tuple[List[float], List[float]]:
    scale = [1.0] * ndim
    translation = [0.0] * ndim

    for transform in dataset_meta.get("coordinateTransformations", []):
        t_type = transform.get("type")
        if t_type == "scale":
            values = transform.get("scale") or []
            if len(values) != ndim:
                raise ValueError(
                    "Scale vector length %s does not match axes dimensionality %s" % (len(values), ndim)
                )
            scale = [float(v) for v in values]
        elif t_type == "translation":
            values = transform.get("translation") or []
            if len(values) != ndim:
                raise ValueError(
                    "Translation vector length %s does not match axes dimensionality %s"
                    % (len(values), ndim)
                )
            translation = [float(v) for v in values]
    return scale, translation


def target_shape_for_level(
    target_spatial: Dict[str, int],
    layout: AxisLayout,
    scale: Sequence[float],
    original_shape: Sequence[int],
) -> Tuple[int, ...]:
    shape = list(original_shape)
    for axis_name, axis_idx in layout.spatial_indices.items():
        base = target_spatial[axis_name]
        scale_factor = scale[axis_idx] if axis_idx < len(scale) else 1.0
        if scale_factor <= 0:
            raise ValueError(f"Invalid scale factor {scale_factor} for axis '{axis_name}'")
        scaled = int(math.ceil(base / scale_factor))
        shape[axis_idx] = scaled
    return tuple(int(v) for v in shape)


def offsets_from_translation(
    layout: AxisLayout,
    scale: Sequence[float],
    translation: Sequence[float],
) -> Tuple[int, ...]:
    offsets: List[int] = []
    for axis_idx, axis_name in enumerate(layout.names):
        if axis_name in layout.spatial_indices:
            scale_factor = scale[axis_idx] if axis_idx < len(scale) else 1.0
            offset = int(math.floor((translation[axis_idx] or 0.0) / scale_factor + 1e-8))
            offsets.append(offset)
        else:
            offsets.append(0)
    return tuple(offsets)


def validate_region(
    offsets: Sequence[int],
    original_shape: Sequence[int],
    target_shape: Sequence[int],
    axis_names: Sequence[str],
) -> None:
    for axis, (off, src_len, dst_len) in enumerate(zip(offsets, original_shape, target_shape)):
        if off < 0:
            raise ValueError(f"Negative offset {off} for axis '{axis_names[axis]}' is not supported")
        if off + src_len > dst_len:
            raise ValueError(
                f"Source data (len={src_len}) with offset {off} exceeds target length {dst_len} on axis '{axis_names[axis]}'"
            )


def create_output_store(path: Path) -> zarr.Group:
    store = zarr.DirectoryStore(str(path))
    return zarr.group(store=store, overwrite=True)


def copy_dataset(
    layout: AxisLayout,
    old_array: zarr.Array,
    new_group: zarr.Group,
    dataset_meta: Dict,
    target_spatial: Dict[str, int],
    fill_value: float,
    num_workers: int | None,
    dry_run: bool,
) -> None:
    scale, translation = compute_scale_and_translation(dataset_meta, old_array.ndim)
    new_shape = target_shape_for_level(target_spatial, layout, scale, old_array.shape)
    offsets = offsets_from_translation(layout, scale, translation)
    validate_region(offsets, old_array.shape, new_shape, layout.names)

    print(
        f"  - dataset '{dataset_meta.get('path')}' scale={scale} translation={translation} -> new_shape={new_shape} offsets={offsets}"
    )

    if dry_run:
        return

    new_array = new_group.create_dataset(
        dataset_meta["path"],
        shape=new_shape,
        chunks=old_array.chunks,
        dtype=old_array.dtype,
        compressor=old_array.compressor,
        fill_value=fill_value,
        filters=old_array.filters,
        order=old_array.order,
        overwrite=True,
    )

    # Copy dataset-level attributes
    ds_attrs = old_array.attrs.asdict()
    if ds_attrs:
        new_array.attrs.put(ds_attrs)

    region = []
    for axis_idx in range(old_array.ndim):
        length = old_array.shape[axis_idx]
        offset = offsets[axis_idx]
        region.append(slice(offset, offset + length))
    region_tuple = tuple(region)

    source = da.from_zarr(old_array)

    dask_config = {"scheduler": "processes"}
    if num_workers is not None:
        dask_config["num_workers"] = int(num_workers)

    with dask.config.set(**dask_config):
        with ProgressBar():
            da.store(
                source,
                new_array,
                regions=region_tuple,
                lock=True,
            )


def update_multiscales_translations(
    multiscales: Dict,
    layout: AxisLayout,
    spatial_names: Iterable[str] = ("z", "y", "x"),
) -> Dict:
    updated = json.loads(json.dumps(multiscales))  # deep copy via JSON
    for dataset in updated[0].get("datasets", []):
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "translation":
                values = list(transform.get("translation", []))
                for axis_name in spatial_names:
                    idx = layout.spatial_indices[axis_name]
                    if idx < len(values):
                        values[idx] = 0.0
                transform["translation"] = values
    return updated


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    output_path = Path(args.output).resolve() if args.output else input_path.with_name(input_path.name + "_resized.zarr")
    if output_path.exists() and not args.dry_run:
        raise SystemExit(f"Output path already exists: {output_path}")

    target_spatial = {axis: size for axis, size in zip(["z", "y", "x"], args.target_shape)}

    input_store = zarr.DirectoryStore(str(input_path))
    root_in = zarr.open(input_store, mode="r")

    root_attrs, layout = load_multiscales(root_in)
    multiscales_meta = root_attrs.get("multiscales")
    datasets_meta = multiscales_meta[0].get("datasets", [])

    print(f"Resizing '{input_path}' -> '{output_path}'")
    print(f"Target spatial shape (z, y, x): {args.target_shape}")

    if args.dry_run:
        print("-- DRY RUN --")
    else:
        root_out = create_output_store(output_path)

    for array_name in root_in.array_keys():
        old_array = root_in[array_name]
        dataset_meta = next((d for d in datasets_meta if d.get("path") == array_name), None)
        if dataset_meta is None:
            raise ValueError(f"Dataset metadata for path '{array_name}' not found in multiscales attributes")

        copy_dataset(
            layout=layout,
            old_array=old_array,
            new_group=root_in if args.dry_run else root_out,
            dataset_meta=dataset_meta,
            target_spatial=target_spatial,
            fill_value=args.fill_value,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("Dry run complete; no data written.")
        return

    # Propagate group attributes, updating translation to zeroed values for spatial axes
    updated_multiscales = update_multiscales_translations(root_attrs["multiscales"], layout)
    root_attrs["multiscales"] = updated_multiscales
    root_out.attrs.put(root_attrs)

    # Copy any additional groups (unlikely for surface frame, but keep generic)
    for subname in root_in.group_keys():
        if subname in root_out:
            continue
        print(f"Copying nested group '{subname}'")
        source_group = root_in[subname]
        target_group = root_out.create_group(subname)
        target_group.attrs.put(source_group.attrs.asdict())

    print("Resizing complete.")


if __name__ == "__main__":
    main()
