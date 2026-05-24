import argparse
from pathlib import Path

import numpy as np


TIFF_SUFFIXES = {".tif", ".tiff"}
ZARR_SUFFIX = ".zarr"


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated XYZ values")
    return tuple(float(part) for part in parts)


def _require_positive_xyz(values, name: str) -> None:
    if any(float(value) <= 0.0 for value in values):
        raise ValueError(f"{name} values must be positive")


def _parse_positive_xyz(value: str, name: str) -> tuple[float, float, float]:
    parsed = _parse_xyz(value)
    try:
        _require_positive_xyz(parsed, name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed


def _parse_non_negative_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{name} must be non-negative")
    return parsed


def _is_tiff_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TIFF_SUFFIXES


def _is_npy_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".npy"


def _is_zarr_path(path: str | Path) -> bool:
    return _split_zarr_path(path)[0] is not None


def _split_zarr_path(path: str | Path) -> tuple[Path | None, str | None]:
    path = Path(path)
    parts = path.parts
    zarr_index = next((index for index, part in enumerate(parts) if Path(part).suffix.lower() == ZARR_SUFFIX), None)
    if zarr_index is None:
        return None, None
    store_path = Path(*parts[: zarr_index + 1])
    array_parts = parts[zarr_index + 1 :]
    array_key = "/".join(array_parts) if array_parts else None
    return store_path, array_key


def _import_tifffile():
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required for .tif/.tiff volume I/O") from exc
    return tifffile


def _import_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr is required for .zarr volume I/O") from exc
    return zarr


def _resolve_zarr_array(opened, array_key: str | None = None):
    if hasattr(opened, "shape") and hasattr(opened, "ndim"):
        return opened
    key = "0" if array_key is None else str(array_key)
    try:
        return opened[key]
    except (KeyError, TypeError) as exc:
        available = ", ".join(str(item) for item in opened.keys()) if hasattr(opened, "keys") else "unknown"
        raise ValueError(f"zarr group does not contain array key {key!r}; available keys: {available}") from exc


def load_volume(path: str | Path, zarr_array_key: str | None = None) -> np.ndarray:
    path = Path(path)
    if _is_tiff_path(path):
        return np.asarray(_import_tifffile().imread(str(path)))
    zarr_store_path, path_array_key = _split_zarr_path(path)
    if zarr_store_path is not None:
        array_key = zarr_array_key if zarr_array_key is not None else path_array_key
        return _resolve_zarr_array(_import_zarr().open(str(zarr_store_path), mode="r"), array_key)
    return np.load(path)


def save_volume(path: str | Path, volume: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_tiff_path(path):
        _import_tifffile().imwrite(str(path), volume)
        return
    if _is_zarr_path(path):
        output = _import_zarr().open(
            str(path),
            mode="w",
            shape=volume.shape,
            dtype=volume.dtype,
            chunks=True,
        )
        output[:] = volume
        return
    np.save(path, volume)


def _read_values_zyx(volume, z_index: np.ndarray, y_index: np.ndarray, x_index: np.ndarray) -> np.ndarray:
    if hasattr(volume, "vindex"):
        return volume.vindex[z_index, y_index, x_index]
    try:
        return volume[z_index, y_index, x_index]
    except (IndexError, TypeError, ValueError):
        return np.asarray(
            [volume[int(z), int(y), int(x)] for z, y, x in zip(z_index, y_index, x_index)],
            dtype=volume.dtype,
        )


def _volume_size(volume) -> int:
    size = getattr(volume, "size", None)
    if size is not None:
        return int(size)
    return int(np.prod(volume.shape))


def trilinear_sample_zyx(
    volume: np.ndarray,
    points_zyx: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    if points_zyx.ndim != 2 or points_zyx.shape[1] != 3:
        raise ValueError(f"expected sample points with shape (N, 3), got {points_zyx.shape}")

    z_size, y_size, x_size = volume.shape
    z = points_zyx[:, 0]
    y = points_zyx[:, 1]
    x = points_zyx[:, 2]
    inside = (
        (z >= 0.0)
        & (z <= z_size - 1)
        & (y >= 0.0)
        & (y <= y_size - 1)
        & (x >= 0.0)
        & (x <= x_size - 1)
    )

    samples = np.full(points_zyx.shape[0], fill_value, dtype=np.float64)
    if not inside.any():
        return samples.astype(volume.dtype, copy=False)

    z_inside = z[inside]
    y_inside = y[inside]
    x_inside = x[inside]
    z0 = np.floor(z_inside).astype(np.int64)
    y0 = np.floor(y_inside).astype(np.int64)
    x0 = np.floor(x_inside).astype(np.int64)
    z1 = np.minimum(z0 + 1, z_size - 1)
    y1 = np.minimum(y0 + 1, y_size - 1)
    x1 = np.minimum(x0 + 1, x_size - 1)
    dz = z_inside - z0
    dy = y_inside - y0
    dx = x_inside - x0

    c000 = _read_values_zyx(volume, z0, y0, x0)
    c001 = _read_values_zyx(volume, z0, y0, x1)
    c010 = _read_values_zyx(volume, z0, y1, x0)
    c011 = _read_values_zyx(volume, z0, y1, x1)
    c100 = _read_values_zyx(volume, z1, y0, x0)
    c101 = _read_values_zyx(volume, z1, y0, x1)
    c110 = _read_values_zyx(volume, z1, y1, x0)
    c111 = _read_values_zyx(volume, z1, y1, x1)

    c00 = c000 * (1.0 - dx) + c001 * dx
    c01 = c010 * (1.0 - dx) + c011 * dx
    c10 = c100 * (1.0 - dx) + c101 * dx
    c11 = c110 * (1.0 - dx) + c111 * dx
    c0 = c00 * (1.0 - dy) + c01 * dy
    c1 = c10 * (1.0 - dy) + c11 * dy
    samples[inside] = c0 * (1.0 - dz) + c1 * dz
    return samples.astype(volume.dtype, copy=False)


def _warp_volume_z_range(
    volume: np.ndarray,
    displacement_xyz: np.ndarray,
    origin_xyz: tuple[float, float, float] | np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    volume_origin_xyz: tuple[float, float, float] | np.ndarray | None = None,
    volume_spacing_xyz: tuple[float, float, float] | np.ndarray | None = None,
    fill_value: float = 0.0,
    z_start: int = 0,
    z_stop: int | None = None,
) -> np.ndarray:
    if displacement_xyz.ndim != 4 or displacement_xyz.shape[3] != 3:
        raise ValueError(f"expected displacement field shape (Z, Y, X, 3), got {displacement_xyz.shape}")
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

    field_origin = np.asarray(origin_xyz, dtype=np.float64)
    field_spacing = np.asarray(spacing_xyz, dtype=np.float64)
    volume_origin = field_origin if volume_origin_xyz is None else np.asarray(volume_origin_xyz, dtype=np.float64)
    volume_spacing = field_spacing if volume_spacing_xyz is None else np.asarray(volume_spacing_xyz, dtype=np.float64)
    _require_positive_xyz(field_spacing, "field spacing")
    _require_positive_xyz(volume_spacing, "volume spacing")
    z_size, y_size, x_size = volume.shape
    z_stop = z_size if z_stop is None else z_stop
    if z_start < 0 or z_stop < z_start or z_stop > z_size:
        raise ValueError(f"invalid z range [{z_start}, {z_stop}) for volume depth {z_size}")
    zz, yy, xx = np.meshgrid(
        np.arange(z_start, z_stop, dtype=np.float64),
        np.arange(y_size, dtype=np.float64),
        np.arange(x_size, dtype=np.float64),
        indexing="ij",
    )
    grid_xyz = np.stack(
        [
            volume_origin[0] + xx * volume_spacing[0],
            volume_origin[1] + yy * volume_spacing[1],
            volume_origin[2] + zz * volume_spacing[2],
        ],
        axis=-1,
    )
    field_x = (grid_xyz[..., 0] - field_origin[0]) / field_spacing[0]
    field_y = (grid_xyz[..., 1] - field_origin[1]) / field_spacing[1]
    field_z = (grid_xyz[..., 2] - field_origin[2]) / field_spacing[2]
    field_points_zyx = np.stack([field_z.ravel(), field_y.ravel(), field_x.ravel()], axis=1)
    sampled_components = [
        trilinear_sample_zyx(displacement_xyz[..., channel], field_points_zyx, fill_value=0.0)
        for channel in range(3)
    ]
    chunk_shape = (z_stop - z_start, y_size, x_size)
    sampled_displacement_xyz = np.stack(sampled_components, axis=1).reshape(chunk_shape + (3,))
    source_xyz = grid_xyz - sampled_displacement_xyz.astype(np.float64)
    source_x = (source_xyz[..., 0] - volume_origin[0]) / volume_spacing[0]
    source_y = (source_xyz[..., 1] - volume_origin[1]) / volume_spacing[1]
    source_z = (source_xyz[..., 2] - volume_origin[2]) / volume_spacing[2]
    points_zyx = np.stack([source_z.ravel(), source_y.ravel(), source_x.ravel()], axis=1)
    warped = trilinear_sample_zyx(volume, points_zyx, fill_value=fill_value)
    return warped.reshape(chunk_shape)


def warp_sampling_diagnostics(
    volume: np.ndarray,
    displacement_xyz: np.ndarray,
    origin_xyz: tuple[float, float, float] | np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    volume_origin_xyz: tuple[float, float, float] | np.ndarray | None = None,
    volume_spacing_xyz: tuple[float, float, float] | np.ndarray | None = None,
    chunk_depth: int = 0,
    sample_step: int = 1,
) -> dict:
    if displacement_xyz.ndim != 4 or displacement_xyz.shape[3] != 3:
        raise ValueError(f"expected displacement field shape (Z, Y, X, 3), got {displacement_xyz.shape}")
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

    field_origin = np.asarray(origin_xyz, dtype=np.float64)
    field_spacing = np.asarray(spacing_xyz, dtype=np.float64)
    volume_origin = field_origin if volume_origin_xyz is None else np.asarray(volume_origin_xyz, dtype=np.float64)
    volume_spacing = field_spacing if volume_spacing_xyz is None else np.asarray(volume_spacing_xyz, dtype=np.float64)
    _require_positive_xyz(field_spacing, "field spacing")
    _require_positive_xyz(volume_spacing, "volume spacing")
    z_size, y_size, x_size = volume.shape
    chunk_depth = z_size if int(chunk_depth) <= 0 else int(chunk_depth)
    sample_step = int(sample_step)
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")
    y_indices = np.arange(0, y_size, sample_step, dtype=np.float64)
    x_indices = np.arange(0, x_size, sample_step, dtype=np.float64)
    z_indices = np.arange(0, z_size, sample_step, dtype=np.float64)
    sample_count = int(len(z_indices) * len(y_indices) * len(x_indices))
    in_bounds_count = 0

    for z_start in range(0, z_size, chunk_depth):
        z_stop = min(z_start + chunk_depth, z_size)
        z_chunk_indices = z_indices[(z_indices >= z_start) & (z_indices < z_stop)]
        if len(z_chunk_indices) == 0:
            continue
        zz, yy, xx = np.meshgrid(
            z_chunk_indices,
            y_indices,
            x_indices,
            indexing="ij",
        )
        grid_xyz = np.stack(
            [
                volume_origin[0] + xx * volume_spacing[0],
                volume_origin[1] + yy * volume_spacing[1],
                volume_origin[2] + zz * volume_spacing[2],
            ],
            axis=-1,
        )
        field_x = (grid_xyz[..., 0] - field_origin[0]) / field_spacing[0]
        field_y = (grid_xyz[..., 1] - field_origin[1]) / field_spacing[1]
        field_z = (grid_xyz[..., 2] - field_origin[2]) / field_spacing[2]
        field_points_zyx = np.stack([field_z.ravel(), field_y.ravel(), field_x.ravel()], axis=1)
        sampled_components = [
            trilinear_sample_zyx(displacement_xyz[..., channel], field_points_zyx, fill_value=0.0)
            for channel in range(3)
        ]
        chunk_shape = (len(z_chunk_indices), len(y_indices), len(x_indices))
        sampled_displacement_xyz = np.stack(sampled_components, axis=1).reshape(chunk_shape + (3,))
        source_xyz = grid_xyz - sampled_displacement_xyz.astype(np.float64)
        source_x = (source_xyz[..., 0] - volume_origin[0]) / volume_spacing[0]
        source_y = (source_xyz[..., 1] - volume_origin[1]) / volume_spacing[1]
        source_z = (source_xyz[..., 2] - volume_origin[2]) / volume_spacing[2]
        inside = (
            (source_z >= 0.0)
            & (source_z <= z_size - 1)
            & (source_y >= 0.0)
            & (source_y <= y_size - 1)
            & (source_x >= 0.0)
            & (source_x <= x_size - 1)
        )
        in_bounds_count += int(inside.sum())

    magnitude = np.linalg.norm(displacement_xyz.astype(np.float64), axis=-1)
    in_bounds_fraction = 1.0 if sample_count == 0 else in_bounds_count / sample_count
    return {
        "schema_version": "1.0.0",
        "volume_shape_zyx": [int(value) for value in volume.shape],
        "displacement_field_shape_zyx": [int(value) for value in displacement_xyz.shape[:3]],
        "metrics_sample_step": int(sample_step),
        "volume_voxel_count": _volume_size(volume),
        "sample_count": sample_count,
        "in_bounds_sample_count": int(in_bounds_count),
        "in_bounds_fraction": float(in_bounds_fraction),
        "out_of_bounds_fraction": float(1.0 - in_bounds_fraction),
        "displacement_magnitude": {
            "min": float(magnitude.min()),
            "max": float(magnitude.max()),
            "mean": float(magnitude.mean()),
        },
    }


def warp_volume_with_displacement_field(
    volume: np.ndarray,
    displacement_xyz: np.ndarray,
    origin_xyz: tuple[float, float, float] | np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    volume_origin_xyz: tuple[float, float, float] | np.ndarray | None = None,
    volume_spacing_xyz: tuple[float, float, float] | np.ndarray | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    return _warp_volume_z_range(
        volume,
        displacement_xyz,
        origin_xyz=origin_xyz,
        spacing_xyz=spacing_xyz,
        volume_origin_xyz=volume_origin_xyz,
        volume_spacing_xyz=volume_spacing_xyz,
        fill_value=fill_value,
    )


def warp_volume_with_displacement_field_chunked(
    volume: np.ndarray,
    displacement_xyz: np.ndarray,
    origin_xyz: tuple[float, float, float] | np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    volume_origin_xyz: tuple[float, float, float] | np.ndarray | None = None,
    volume_spacing_xyz: tuple[float, float, float] | np.ndarray | None = None,
    fill_value: float = 0.0,
    chunk_depth: int = 16,
) -> np.ndarray:
    chunk_depth = int(chunk_depth)
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")

    warped = np.empty_like(volume)
    for z_start in range(0, volume.shape[0], chunk_depth):
        z_stop = min(z_start + chunk_depth, volume.shape[0])
        warped[z_start:z_stop] = _warp_volume_z_range(
            volume,
            displacement_xyz,
            origin_xyz=origin_xyz,
            spacing_xyz=spacing_xyz,
            volume_origin_xyz=volume_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            fill_value=fill_value,
            z_start=z_start,
            z_stop=z_stop,
        )
    return warped


def warp_volume_to_npy_memmap(
    output_path: str | Path,
    volume: np.ndarray,
    displacement_xyz: np.ndarray,
    origin_xyz: tuple[float, float, float] | np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    volume_origin_xyz: tuple[float, float, float] | np.ndarray | None = None,
    volume_spacing_xyz: tuple[float, float, float] | np.ndarray | None = None,
    fill_value: float = 0.0,
    chunk_depth: int = 16,
) -> Path:
    chunk_depth = int(chunk_depth)
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=volume.dtype,
        shape=volume.shape,
    )
    for z_start in range(0, volume.shape[0], chunk_depth):
        z_stop = min(z_start + chunk_depth, volume.shape[0])
        output[z_start:z_stop] = _warp_volume_z_range(
            volume,
            displacement_xyz,
            origin_xyz=origin_xyz,
            spacing_xyz=spacing_xyz,
            volume_origin_xyz=volume_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            fill_value=fill_value,
            z_start=z_start,
            z_stop=z_stop,
        )
    output.flush()
    del output
    return output_path


def warp_volume_to_zarr(
    output_path: str | Path,
    volume: np.ndarray,
    displacement_xyz: np.ndarray,
    origin_xyz: tuple[float, float, float] | np.ndarray,
    spacing_xyz: tuple[float, float, float] | np.ndarray,
    volume_origin_xyz: tuple[float, float, float] | np.ndarray | None = None,
    volume_spacing_xyz: tuple[float, float, float] | np.ndarray | None = None,
    fill_value: float = 0.0,
    chunk_depth: int = 16,
) -> Path:
    chunk_depth = int(chunk_depth)
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (min(chunk_depth, volume.shape[0]), volume.shape[1], volume.shape[2])
    output = _import_zarr().open(
        str(output_path),
        mode="w",
        shape=volume.shape,
        dtype=volume.dtype,
        chunks=chunks,
    )
    for z_start in range(0, volume.shape[0], chunk_depth):
        z_stop = min(z_start + chunk_depth, volume.shape[0])
        output[z_start:z_stop] = _warp_volume_z_range(
            volume,
            displacement_xyz,
            origin_xyz=origin_xyz,
            spacing_xyz=spacing_xyz,
            volume_origin_xyz=volume_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            fill_value=fill_value,
            z_start=z_start,
            z_stop=z_stop,
        )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a coarse displacement field to a .npy, .tif, .tiff, or .zarr volume."
    )
    parser.add_argument("--volume", required=True, help="Input 3D .npy/.tif/.tiff/.zarr volume in z,y,x order.")
    parser.add_argument("--field", required=True, help=".npz displacement field from build_deformation_field.py.")
    parser.add_argument("--output", required=True, help="Output warped .npy/.tif/.tiff/.zarr volume.")
    parser.add_argument(
        "--zarr-array-key",
        default=None,
        help="Array key to read when --volume points at a Zarr group. Defaults to '0'.",
    )
    parser.add_argument("--volume-origin", type=_parse_xyz, default=None, help="Input volume origin as x,y,z. Defaults to field origin.")
    parser.add_argument(
        "--volume-spacing",
        type=lambda value: _parse_positive_xyz(value, "volume spacing"),
        default=None,
        help="Input volume spacing as x,y,z. Defaults to field spacing.",
    )
    parser.add_argument("--fill-value", type=float, default=0.0, help="Value used outside the input volume.")
    parser.add_argument(
        "--chunk-depth",
        type=lambda value: _parse_non_negative_int(value, "chunk-depth"),
        default=0,
        help="Warp this many z-slices at a time; 0 warps the whole volume at once.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    volume = load_volume(args.volume, zarr_array_key=args.zarr_array_key)
    with np.load(args.field) as field_data:
        displacement_xyz = field_data["displacement_xyz"]
        origin_xyz = field_data["origin_xyz"]
        spacing_xyz = field_data["spacing_xyz"]
    if args.chunk_depth > 0 and _is_npy_path(args.output):
        warp_volume_to_npy_memmap(
            args.output,
            volume,
            displacement_xyz,
            origin_xyz=origin_xyz,
            spacing_xyz=spacing_xyz,
            volume_origin_xyz=args.volume_origin,
            volume_spacing_xyz=args.volume_spacing,
            fill_value=args.fill_value,
            chunk_depth=args.chunk_depth,
        )
        return
    if args.chunk_depth > 0 and _is_zarr_path(args.output):
        warp_volume_to_zarr(
            args.output,
            volume,
            displacement_xyz,
            origin_xyz=origin_xyz,
            spacing_xyz=spacing_xyz,
            volume_origin_xyz=args.volume_origin,
            volume_spacing_xyz=args.volume_spacing,
            fill_value=args.fill_value,
            chunk_depth=args.chunk_depth,
        )
        return

    warp_fn = warp_volume_with_displacement_field_chunked if args.chunk_depth > 0 else warp_volume_with_displacement_field
    warp_kwargs = {"chunk_depth": args.chunk_depth} if args.chunk_depth > 0 else {}
    warped = warp_fn(
        volume,
        displacement_xyz,
        origin_xyz=origin_xyz,
        spacing_xyz=spacing_xyz,
        volume_origin_xyz=args.volume_origin,
        volume_spacing_xyz=args.volume_spacing,
        fill_value=args.fill_value,
        **warp_kwargs,
    )
    save_volume(args.output, warped)


if __name__ == "__main__":
    main()
