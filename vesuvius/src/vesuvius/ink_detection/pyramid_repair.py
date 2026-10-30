"""Rebuild ink pyramid levels that cropped depth instead of pooling it.

Some published surface volumes declare a Z scale of 1 and a mean downsample,
then store only the first pooled-depth slices. Level 2 of ``841/w00`` is the
successive 2× XY mean of level-0 slices 0..16, not a mean through all 65 slices. This
module rewrites those levels with the same successive 2× rounded mean extended
across Z, and sets the Z scale to match.
"""

from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import zarr


class PyramidRepairError(ValueError):
    """The pyramid cannot be rebuilt by the depth-pooling repair."""


def rounded_mean_downsample(array: np.ndarray) -> np.ndarray:
    """Pool every axis by 2 with one rounded mean of the whole 2× neighborhood.

    A full block of ``2**ndim`` voxels becomes ``(sum + count / 2) // count``.
    An edge block uses only the voxels that exist, so a leftover voxel is
    copied. Output length on each axis is ``ceil(n / 2)``.
    """

    src = np.asarray(array)
    if src.ndim == 0:
        raise PyramidRepairError("Cannot downsample a scalar")
    if not np.issubdtype(src.dtype, np.integer):
        raise PyramidRepairError(
            f"Rounded-mean downsampling expects an integer volume, got {src.dtype}"
        )
    padded_shape = tuple(length + (length % 2) for length in src.shape)
    values = np.zeros(padded_shape, dtype=np.int64)
    counts = np.zeros(padded_shape, dtype=np.int64)
    original = tuple(slice(0, length) for length in src.shape)
    values[original] = src
    counts[original] = 1
    pair_shape: list[int] = []
    for length in padded_shape:
        pair_shape.extend((length // 2, 2))
    pair_axes = tuple(range(1, len(pair_shape), 2))
    summed = values.reshape(pair_shape).sum(axis=pair_axes)
    count = counts.reshape(pair_shape).sum(axis=pair_axes)
    return ((summed + count // 2) // np.maximum(count, 1)).astype(src.dtype, copy=False)


def successive_rounded_mean(array: np.ndarray, steps: int) -> np.ndarray:
    """Apply :func:`rounded_mean_downsample` ``steps`` times."""

    if steps < 1:
        raise PyramidRepairError("Downsample steps must be positive")
    reduced = array
    for _ in range(steps):
        reduced = rounded_mean_downsample(reduced)
    return reduced


def repair_pyramid(source: str | Path, destination: str | Path) -> tuple[str, ...]:
    """Write a new pyramid, rebuilding truncated mean levels from level 0.

    The destination must not already exist. Levels whose depth already agrees
    with the declared Z scale are copied. Returns the rebuilt level names.
    """

    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"Refusing to replace {destination_path}")
    root = _open_group(source, mode="r")
    entry = _multiscale_entry(root)
    if entry is None:
        raise PyramidRepairError(f"{source} has no multiscale metadata")
    datasets = _datasets(entry)
    if "0" not in datasets or "0" not in root:
        raise PyramidRepairError(f"{source} has no level 0")
    reference = root["0"]
    plans = [
        _plan_level(entry, datasets, reference, root, name)
        for name in datasets
        if name != "0"
    ]
    rebuilt = tuple(plan["name"] for plan in plans if plan["rebuild"])
    if not rebuilt:
        raise PyramidRepairError(f"{source} has no truncated mean level to rebuild")

    destination_group = _open_group(destination_path, mode="w")
    _copy_array(reference, destination_group, "0")
    written = {"0"}
    for plan in sorted(plans, key=lambda item: (item["factor"], item["name"])):
        if not plan["rebuild"]:
            _copy_array(root[plan["name"]], destination_group, plan["name"])
            written.add(plan["name"])
            continue
        parent_name, steps = _parent_steps(plan["factor"], written, plans)
        _pool_array(
            destination_group[parent_name],
            destination_group,
            plan["name"],
            steps=steps,
        )
        written.add(plan["name"])
    destination_group.attrs.update(_repaired_attributes(root, entry, rebuilt))
    return rebuilt


def main(argv: list[str] | None = None) -> None:
    """Repair one pyramid and print the levels that were rebuilt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source OME-Zarr group, local path or URL")
    parser.add_argument("destination", help="New OME-Zarr group. Refuses to overwrite.")
    args = parser.parse_args(argv)
    rebuilt = repair_pyramid(args.source, args.destination)
    print(f"rebuilt levels {', '.join(rebuilt)} -> {args.destination}")


def _open_group(path: str | Path, *, mode: str):
    kwargs: dict[str, Any] = {"mode": mode}
    if mode == "w" and int(zarr.__version__.split(".", 1)[0]) >= 3:
        kwargs["zarr_format"] = 2
    return zarr.open_group(path, **kwargs)


def _create_array(group, name: str, *, shape, chunks, dtype):
    kwargs = {
        "shape": tuple(int(value) for value in shape),
        "chunks": tuple(int(value) for value in chunks),
        "dtype": dtype,
        "fill_value": 0,
    }
    if hasattr(group, "create_array"):
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def _chunk_tuple(array) -> tuple[int, ...]:
    chunks = tuple(int(value) for value in array.chunks)
    return tuple(min(chunk, int(length)) for chunk, length in zip(chunks, array.shape))


def _iter_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    ranges = [
        [slice(start, min(length, start + chunk)) for start in range(0, length, chunk)]
        for length, chunk in zip(shape, chunks)
    ]
    yield from product(*ranges)


def _copy_array(source, group, name: str) -> None:
    chunks = _chunk_tuple(source)
    destination = _create_array(
        group, name, shape=source.shape, chunks=chunks, dtype=source.dtype
    )
    for slices in _iter_slices(tuple(int(n) for n in source.shape), chunks):
        destination[slices] = np.asarray(source[slices])


def _pool_array(source, group, name: str, *, steps: int) -> None:
    out_shape = tuple(int(n) for n in source.shape)
    for _ in range(steps):
        out_shape = tuple((n + 1) // 2 for n in out_shape)
    in_chunks = _chunk_tuple(source)
    out_chunks = tuple(
        max(1, min(in_chunks[axis], out_shape[axis])) for axis in range(len(out_shape))
    )
    destination = _create_array(
        group, name, shape=out_shape, chunks=out_chunks, dtype=source.dtype
    )
    span = 2**steps
    for slices in _iter_slices(out_shape, out_chunks):
        source_slices = tuple(
            slice(
                item.start * span,
                min(int(source.shape[axis]), item.stop * span),
            )
            for axis, item in enumerate(slices)
        )
        block = np.asarray(source[source_slices])
        destination[slices] = successive_rounded_mean(block, steps)


def _parent_steps(
    factor: int,
    written: set[str],
    plans: list[dict[str, Any]],
) -> tuple[str, int]:
    candidates = [("0", 1)]
    for plan in plans:
        if plan["name"] in written and plan["factor"] > 1 and factor % plan["factor"] == 0:
            candidates.append((plan["name"], plan["factor"]))
    parent_name, parent_factor = max(candidates, key=lambda item: item[1])
    steps = int(round(math.log2(factor / parent_factor)))
    if 2**steps != factor // parent_factor:
        raise PyramidRepairError(
            f"Level factor {factor} is not a power-of-two step from {parent_name}"
        )
    return parent_name, steps


def _plan_level(entry, datasets, reference, root, name: str) -> dict[str, Any]:
    factor = _xy_factor(datasets["0"], datasets[name], _axis_index(entry, "y"), _axis_index(entry, "x"))
    array = root[name]
    rebuild = False
    if factor is not None:
        z_axis = _axis_index(entry, "z")
        ratio = _scale_component(datasets[name], z_axis) / _scale_component(datasets["0"], z_axis)
        expected = tuple((int(n) + factor - 1) // factor for n in reference.shape)
        same_shape = tuple(int(n) for n in array.shape) == expected
        kept_depth = int(array.shape[z_axis]) == int(reference.shape[z_axis])
        rebuild = abs(ratio - 1.0) <= 1e-3 and same_shape and not kept_depth
    return {"name": name, "factor": factor or 1, "rebuild": rebuild}


def _xy_factor(reference: dict[str, Any], selected: dict[str, Any], y_axis: int, x_axis: int) -> int | None:
    factors = []
    for axis in (y_axis, x_axis):
        ratio = _scale_component(selected, axis) / _scale_component(reference, axis)
        factor = int(round(ratio))
        if factor < 2 or abs(ratio - factor) > 1e-3 or factor & (factor - 1):
            return None
        factors.append(factor)
    if factors[0] != factors[1]:
        return None
    return factors[0]


def _scale_component(dataset: dict[str, Any], axis: int) -> float:
    for transform in dataset.get("coordinateTransformations") or []:
        if not isinstance(transform, dict):
            continue
        scale = transform.get("scale")
        if not isinstance(scale, (list, tuple)) or axis >= len(scale):
            continue
        value = float(scale[axis])
        if math.isfinite(value) and value > 0.0:
            return value
    raise PyramidRepairError(f"Dataset {dataset.get('path')!r} has no positive scale on axis {axis}")


def _axis_index(entry: dict[str, Any], name: str) -> int:
    for index, axis in enumerate(entry.get("axes") or []):
        if isinstance(axis, dict) and str(axis.get("name", "")).lower() == name:
            return index
    raise PyramidRepairError(f"Multiscale metadata has no {name} axis")


def _multiscale_entry(root) -> dict[str, Any] | None:
    attrs = dict(root.attrs)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales or not isinstance(multiscales[0], dict):
        return None
    return multiscales[0]


def _datasets(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    found = {}
    for dataset in entry.get("datasets") or []:
        if isinstance(dataset, dict) and dataset.get("path") is not None:
            found[str(dataset["path"])] = dataset
    return found


def _repaired_attributes(root, entry, rebuilt: tuple[str, ...]) -> dict[str, Any]:
    attrs = dict(root.attrs)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list):
        return attrs
    updated = []
    for item in multiscales:
        if not isinstance(item, dict):
            updated.append(item)
            continue
        item = dict(item)
        item_datasets = []
        for dataset in item.get("datasets") or []:
            if not isinstance(dataset, dict):
                item_datasets.append(dataset)
                continue
            dataset = dict(dataset)
            name = str(dataset.get("path"))
            if name in rebuilt:
                dataset["coordinateTransformations"] = _with_z_scale(
                    dataset.get("coordinateTransformations"),
                    _axis_index(entry, "z"),
                    _axis_index(entry, "y"),
                )
            item_datasets.append(dataset)
        item["datasets"] = item_datasets
        metadata = dict(item.get("metadata") or {})
        metadata["downsampling_method"] = "successive 2x rounded mean"
        metadata["rebuilt_levels"] = list(rebuilt)
        item["metadata"] = metadata
        updated.append(item)
    attrs["multiscales"] = updated
    return attrs


def _with_z_scale(transforms, z_axis: int, y_axis: int):
    copied = []
    replaced = False
    for transform in transforms or []:
        if (
            not replaced
            and isinstance(transform, dict)
            and isinstance(transform.get("scale"), (list, tuple))
        ):
            scale = list(transform["scale"])
            scale[z_axis] = scale[y_axis]
            transform = dict(transform)
            transform["scale"] = scale
            replaced = True
        copied.append(transform)
    if not replaced:
        raise PyramidRepairError("Rebuilt level is missing a scale transform")
    return copied


if __name__ == "__main__":
    main()
