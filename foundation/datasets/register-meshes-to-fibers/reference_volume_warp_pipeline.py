import argparse
import json
from pathlib import Path

import numpy as np

from build_deformation_field import build_grid_field, load_control_points
from export_deformation_control_points import export_control_points
from warp_volume_with_field import (
    _is_npy_path,
    _is_zarr_path,
    load_volume,
    save_volume,
    warp_sampling_diagnostics,
    warp_volume_with_displacement_field,
    warp_volume_with_displacement_field_chunked,
    warp_volume_to_npy_memmap,
    warp_volume_to_zarr,
)


def _parse_xyz(value: str, cast=float) -> tuple:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated XYZ values")
    return tuple(cast(part) for part in parts)


def _require_positive(values: tuple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")


def _parse_positive_xyz(value: str, cast=float, name: str = "values") -> tuple:
    parsed = _parse_xyz(value, cast)
    try:
        _require_positive(parsed, name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed


def _parse_positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive")
    return parsed


def _parse_non_negative_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{name} must be non-negative")
    return parsed


def _parse_grid_shape(value: str) -> tuple[int, int, int] | str:
    if value.lower() == "auto":
        return "auto"
    return _parse_positive_xyz(value, int, "grid shape")


def infer_grid_shape_from_volume(
    volume_shape_zyx: tuple[int, int, int],
    field_spacing_xyz: tuple[float, float, float],
    volume_spacing_xyz: tuple[float, float, float],
) -> tuple[int, int, int]:
    z_size, y_size, x_size = volume_shape_zyx
    volume_shape_xyz = (x_size, y_size, z_size)
    shape_xyz = []
    for size, volume_spacing, field_spacing in zip(
        volume_shape_xyz,
        volume_spacing_xyz,
        field_spacing_xyz,
    ):
        if field_spacing <= 0.0:
            raise ValueError("field spacing values must be positive")
        if volume_spacing <= 0.0:
            raise ValueError("volume spacing values must be positive")
        extent = max(0.0, (size - 1) * volume_spacing)
        shape_xyz.append(int(np.ceil(extent / field_spacing)) + 1)
    return tuple(shape_xyz)


def _resolve_output_path(output_dir: Path, output_name: str) -> Path:
    path = Path(output_name)
    if path.is_absolute():
        return path
    return output_dir / path


def _as_json_list(values) -> list:
    return [float(value) if isinstance(value, (float, np.floating)) else int(value) for value in values]


def _write_manifest(
    manifest_path: Path,
    mesh_pairs: list[tuple[Path, Path]],
    volume_path: Path,
    volume: np.ndarray,
    output_paths: dict[str, Path],
    requested_grid_shape,
    actual_grid_shape_xyz: tuple[int, int, int],
    field_spacing_xyz: tuple[float, float, float],
    field_origin_xyz: tuple[float, float, float],
    volume_spacing_xyz: tuple[float, float, float] | None,
    volume_origin_xyz: tuple[float, float, float] | None,
    effective_volume_spacing_xyz: tuple[float, float, float],
    effective_volume_origin_xyz: tuple[float, float, float],
    zarr_array_key: str | None,
    max_points_per_mesh: int | None,
    k: int,
    power: float,
    field_query_chunk_size: int | None,
    field_control_chunk_size: int | None,
    fill_value: float,
    chunk_depth: int,
    output_write_mode: str,
    metrics: dict,
) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "coordinate_order": "xyz",
        "input_volume": str(volume_path),
        "volume_shape_zyx": [int(value) for value in volume.shape],
        "volume_dtype": str(volume.dtype),
        "mesh_pairs": [
            {
                "source_mesh": str(source),
                "registered_mesh": str(registered),
            }
            for source, registered in mesh_pairs
        ],
        "output_paths": {name: str(path) for name, path in output_paths.items()},
        "requested_grid_shape": requested_grid_shape,
        "actual_grid_shape_xyz": [int(value) for value in actual_grid_shape_xyz],
        "field_spacing_xyz": _as_json_list(field_spacing_xyz),
        "field_origin_xyz": _as_json_list(field_origin_xyz),
        "volume_spacing_xyz": None if volume_spacing_xyz is None else _as_json_list(volume_spacing_xyz),
        "volume_origin_xyz": None if volume_origin_xyz is None else _as_json_list(volume_origin_xyz),
        "effective_volume_spacing_xyz": _as_json_list(effective_volume_spacing_xyz),
        "effective_volume_origin_xyz": _as_json_list(effective_volume_origin_xyz),
        "zarr_array_key": zarr_array_key,
        "max_points_per_mesh": max_points_per_mesh,
        "idw_k": int(k),
        "idw_power": float(power),
        "field_query_chunk_size": field_query_chunk_size,
        "field_control_chunk_size": field_control_chunk_size,
        "fill_value": float(fill_value),
        "chunk_depth": int(chunk_depth),
        "output_write_mode": output_write_mode,
        "metrics": metrics,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def run_pipeline(
    mesh_pairs: list[tuple[Path, Path]],
    volume_path: Path,
    output_dir: Path,
    controls_output: str = "deformation-controls.json",
    field_output: str = "deformation-field.npz",
    warped_output: str = "warped-volume.npy",
    manifest_output: str = "deformation-run-manifest.json",
    metrics_output: str = "deformation-run-metrics.json",
    grid_shape_xyz: tuple[int, int, int] = (1, 1, 1),
    field_spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    field_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    volume_spacing_xyz: tuple[float, float, float] | None = None,
    volume_origin_xyz: tuple[float, float, float] | None = None,
    zarr_array_key: str | None = None,
    max_points_per_mesh: int | None = None,
    k: int = 8,
    power: float = 2.0,
    field_query_chunk_size: int | None = None,
    field_control_chunk_size: int | None = None,
    fill_value: float = 0.0,
    chunk_depth: int = 0,
    metrics_sample_step: int = 1,
) -> dict[str, Path]:
    if grid_shape_xyz != "auto":
        _require_positive(grid_shape_xyz, "grid shape")
    _require_positive(field_spacing_xyz, "field spacing")
    if volume_spacing_xyz is not None:
        _require_positive(volume_spacing_xyz, "volume spacing")
    if chunk_depth < 0:
        raise ValueError("chunk_depth must be non-negative")
    if metrics_sample_step <= 0:
        raise ValueError("metrics_sample_step must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if field_query_chunk_size is not None and field_query_chunk_size <= 0:
        raise ValueError("field_query_chunk_size must be positive")
    if field_control_chunk_size is not None and field_control_chunk_size <= 0:
        raise ValueError("field_control_chunk_size must be positive")
    if max_points_per_mesh is not None and max_points_per_mesh < 0:
        raise ValueError("max_points_per_mesh must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    controls_path = _resolve_output_path(output_dir, controls_output)
    field_path = _resolve_output_path(output_dir, field_output)
    warped_path = _resolve_output_path(output_dir, warped_output)
    manifest_path = _resolve_output_path(output_dir, manifest_output)
    metrics_path = _resolve_output_path(output_dir, metrics_output)
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    field_path.parent.mkdir(parents=True, exist_ok=True)
    warped_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    volume = load_volume(volume_path, zarr_array_key=zarr_array_key)
    resolved_volume_spacing = volume_spacing_xyz if volume_spacing_xyz is not None else field_spacing_xyz
    resolved_volume_origin = volume_origin_xyz if volume_origin_xyz is not None else field_origin_xyz
    requested_grid_shape = "auto" if grid_shape_xyz == "auto" else [int(value) for value in grid_shape_xyz]
    if grid_shape_xyz == "auto":
        grid_shape_xyz = infer_grid_shape_from_volume(
            volume.shape,
            field_spacing_xyz=field_spacing_xyz,
            volume_spacing_xyz=resolved_volume_spacing,
        )

    controls = export_control_points(
        mesh_pairs,
        max_points_per_mesh=max_points_per_mesh,
    )
    with controls_path.open("w", encoding="utf-8") as f:
        json.dump(controls, f, indent=2)
        f.write("\n")

    controls_xyz, displacements_xyz = load_control_points(controls_path)
    displacement_field, field_origin, field_spacing = build_grid_field(
        controls_xyz,
        displacements_xyz,
        grid_shape_xyz=grid_shape_xyz,
        spacing_xyz=field_spacing_xyz,
        origin_xyz=field_origin_xyz,
        k=k,
        power=power,
        query_chunk_size=field_query_chunk_size,
        control_chunk_size=field_control_chunk_size,
    )
    np.savez_compressed(
        field_path,
        displacement_xyz=displacement_field,
        origin_xyz=field_origin,
        spacing_xyz=field_spacing,
        coordinate_order=np.array("zyx_grid_xyz_vectors"),
        controls_path=np.array(str(controls_path)),
    )
    metrics = warp_sampling_diagnostics(
        volume,
        displacement_field,
        origin_xyz=field_origin,
        spacing_xyz=field_spacing,
        volume_origin_xyz=volume_origin_xyz,
        volume_spacing_xyz=volume_spacing_xyz,
        chunk_depth=chunk_depth,
        sample_step=metrics_sample_step,
    )
    metrics["mesh_pair_count"] = len(mesh_pairs)
    metrics["control_point_count"] = len(controls["control_points"])

    if chunk_depth > 0 and _is_npy_path(warped_path):
        warp_volume_to_npy_memmap(
            warped_path,
            volume,
            displacement_field,
            origin_xyz=field_origin,
            spacing_xyz=field_spacing,
            volume_origin_xyz=volume_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            fill_value=fill_value,
            chunk_depth=chunk_depth,
        )
        output_paths = {
            "controls": controls_path,
            "field": field_path,
            "warped": warped_path,
            "manifest": manifest_path,
            "metrics": metrics_path,
        }
        metrics["output_write_mode"] = "chunked_npy_memmap"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        _write_manifest(
            manifest_path,
            mesh_pairs=mesh_pairs,
            volume_path=volume_path,
            volume=volume,
            output_paths=output_paths,
            requested_grid_shape=requested_grid_shape,
            actual_grid_shape_xyz=grid_shape_xyz,
            field_spacing_xyz=field_spacing_xyz,
            field_origin_xyz=field_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            volume_origin_xyz=volume_origin_xyz,
            effective_volume_spacing_xyz=resolved_volume_spacing,
            effective_volume_origin_xyz=resolved_volume_origin,
            zarr_array_key=zarr_array_key,
            max_points_per_mesh=max_points_per_mesh,
            k=k,
            power=power,
            field_query_chunk_size=field_query_chunk_size,
            field_control_chunk_size=field_control_chunk_size,
            fill_value=fill_value,
            chunk_depth=chunk_depth,
            output_write_mode="chunked_npy_memmap",
            metrics=metrics,
        )
        return {
            "controls": controls_path,
            "field": field_path,
            "warped": warped_path,
            "manifest": manifest_path,
            "metrics": metrics_path,
        }
    if chunk_depth > 0 and _is_zarr_path(warped_path):
        warp_volume_to_zarr(
            warped_path,
            volume,
            displacement_field,
            origin_xyz=field_origin,
            spacing_xyz=field_spacing,
            volume_origin_xyz=volume_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            fill_value=fill_value,
            chunk_depth=chunk_depth,
        )
        output_paths = {
            "controls": controls_path,
            "field": field_path,
            "warped": warped_path,
            "manifest": manifest_path,
            "metrics": metrics_path,
        }
        metrics["output_write_mode"] = "chunked_zarr"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        _write_manifest(
            manifest_path,
            mesh_pairs=mesh_pairs,
            volume_path=volume_path,
            volume=volume,
            output_paths=output_paths,
            requested_grid_shape=requested_grid_shape,
            actual_grid_shape_xyz=grid_shape_xyz,
            field_spacing_xyz=field_spacing_xyz,
            field_origin_xyz=field_origin_xyz,
            volume_spacing_xyz=volume_spacing_xyz,
            volume_origin_xyz=volume_origin_xyz,
            effective_volume_spacing_xyz=resolved_volume_spacing,
            effective_volume_origin_xyz=resolved_volume_origin,
            zarr_array_key=zarr_array_key,
            max_points_per_mesh=max_points_per_mesh,
            k=k,
            power=power,
            field_query_chunk_size=field_query_chunk_size,
            field_control_chunk_size=field_control_chunk_size,
            fill_value=fill_value,
            chunk_depth=chunk_depth,
            output_write_mode="chunked_zarr",
            metrics=metrics,
        )
        return {
            "controls": controls_path,
            "field": field_path,
            "warped": warped_path,
            "manifest": manifest_path,
            "metrics": metrics_path,
        }

    warp_fn = warp_volume_with_displacement_field_chunked if chunk_depth > 0 else warp_volume_with_displacement_field
    warp_kwargs = {"chunk_depth": chunk_depth} if chunk_depth > 0 else {}
    warped = warp_fn(
        volume,
        displacement_field,
        origin_xyz=field_origin,
        spacing_xyz=field_spacing,
        volume_origin_xyz=volume_origin_xyz,
        volume_spacing_xyz=volume_spacing_xyz,
        fill_value=fill_value,
        **warp_kwargs,
    )
    save_volume(warped_path, warped)
    output_write_mode = "chunked_volume_save" if chunk_depth > 0 else "full_volume_save"
    output_paths = {
        "controls": controls_path,
        "field": field_path,
        "warped": warped_path,
        "manifest": manifest_path,
        "metrics": metrics_path,
    }
    metrics["output_write_mode"] = output_write_mode
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    _write_manifest(
        manifest_path,
        mesh_pairs=mesh_pairs,
        volume_path=volume_path,
        volume=volume,
        output_paths=output_paths,
        requested_grid_shape=requested_grid_shape,
        actual_grid_shape_xyz=grid_shape_xyz,
        field_spacing_xyz=field_spacing_xyz,
        field_origin_xyz=field_origin_xyz,
        volume_spacing_xyz=volume_spacing_xyz,
        volume_origin_xyz=volume_origin_xyz,
        effective_volume_spacing_xyz=resolved_volume_spacing,
        effective_volume_origin_xyz=resolved_volume_origin,
        zarr_array_key=zarr_array_key,
        max_points_per_mesh=max_points_per_mesh,
        k=k,
        power=power,
        field_query_chunk_size=field_query_chunk_size,
        field_control_chunk_size=field_control_chunk_size,
        fill_value=fill_value,
        chunk_depth=chunk_depth,
        output_write_mode=output_write_mode,
        metrics=metrics,
    )
    return {
        "controls": controls_path,
        "field": field_path,
        "warped": warped_path,
        "manifest": manifest_path,
        "metrics": metrics_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reference mesh-to-volume deformation workflow: export sparse "
            "controls, build a coarse field, and warp a .npy, .tif, .tiff, or .zarr volume."
        )
    )
    parser.add_argument(
        "--mesh-pair",
        action="append",
        nargs=2,
        metavar=("SOURCE_OBJ", "REGISTERED_OBJ"),
        required=True,
        help="Source and registered OBJ pair. May be provided more than once.",
    )
    parser.add_argument("--volume", required=True, help="Input 3D .npy/.tif/.tiff/.zarr volume in z,y,x order.")
    parser.add_argument(
        "--zarr-array-key",
        default=None,
        help="Array key to read when --volume points at a Zarr group. Defaults to '0'.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs.")
    parser.add_argument(
        "--controls-output",
        default="deformation-controls.json",
        help="Control-point JSON filename or absolute output path.",
    )
    parser.add_argument(
        "--field-output",
        default="deformation-field.npz",
        help="Displacement-field .npz filename or absolute output path.",
    )
    parser.add_argument(
        "--warped-output",
        default="warped-volume.npy",
        help="Warped .npy/.tif/.tiff/.zarr filename or absolute output path.",
    )
    parser.add_argument(
        "--manifest-output",
        default="deformation-run-manifest.json",
        help="Run manifest JSON filename or absolute output path.",
    )
    parser.add_argument(
        "--metrics-output",
        default="deformation-run-metrics.json",
        help="Run metrics JSON filename or absolute output path.",
    )
    parser.add_argument(
        "--grid-shape",
        required=True,
        type=_parse_grid_shape,
        help="Displacement field shape as x,y,z, or 'auto' to cover the input volume extent.",
    )
    parser.add_argument(
        "--field-spacing",
        required=True,
        type=lambda value: _parse_positive_xyz(value, float, "field spacing"),
        help="Displacement field spacing as x,y,z.",
    )
    parser.add_argument(
        "--field-origin",
        default="0,0,0",
        type=lambda value: _parse_xyz(value, float),
        help="Displacement field origin as x,y,z.",
    )
    parser.add_argument(
        "--volume-spacing",
        default=None,
        type=lambda value: _parse_positive_xyz(value, float, "volume spacing"),
        help="Input volume spacing as x,y,z. Defaults to field spacing.",
    )
    parser.add_argument(
        "--volume-origin",
        default=None,
        type=lambda value: _parse_xyz(value, float),
        help="Input volume origin as x,y,z. Defaults to field origin.",
    )
    parser.add_argument(
        "--max-points-per-mesh",
        type=lambda value: _parse_non_negative_int(value, "max-points-per-mesh"),
        default=0,
        help="Uniformly sample at most this many vertices per mesh; 0 keeps all vertices.",
    )
    parser.add_argument("--k", type=lambda value: _parse_positive_int(value, "k"), default=8, help="Nearest controls used for IDW interpolation.")
    parser.add_argument("--power", type=float, default=2.0, help="Inverse-distance power.")
    parser.add_argument(
        "--field-query-chunk-size",
        type=lambda value: _parse_positive_int(value, "field-query-chunk-size"),
        default=None,
        help="Interpolate at most this many displacement-field grid points per IDW batch.",
    )
    parser.add_argument(
        "--field-control-chunk-size",
        type=lambda value: _parse_positive_int(value, "field-control-chunk-size"),
        default=None,
        help="Compare at most this many deformation controls per IDW sub-batch.",
    )
    parser.add_argument("--fill-value", type=float, default=0.0, help="Value used outside the input volume.")
    parser.add_argument(
        "--chunk-depth",
        type=lambda value: _parse_non_negative_int(value, "chunk-depth"),
        default=0,
        help="Warp this many z-slices at a time; 0 warps the whole volume at once.",
    )
    parser.add_argument(
        "--metrics-sample-step",
        type=lambda value: _parse_positive_int(value, "metrics-sample-step"),
        default=1,
        help="Sample every Nth voxel per axis for run metrics; 1 evaluates every voxel.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(
        mesh_pairs=[(Path(source), Path(registered)) for source, registered in args.mesh_pair],
        volume_path=Path(args.volume),
        output_dir=Path(args.output_dir),
        controls_output=args.controls_output,
        field_output=args.field_output,
        warped_output=args.warped_output,
        manifest_output=args.manifest_output,
        metrics_output=args.metrics_output,
        grid_shape_xyz=args.grid_shape,
        field_spacing_xyz=args.field_spacing,
        field_origin_xyz=args.field_origin,
        volume_spacing_xyz=args.volume_spacing,
        volume_origin_xyz=args.volume_origin,
        zarr_array_key=args.zarr_array_key,
        max_points_per_mesh=args.max_points_per_mesh,
        k=args.k,
        power=args.power,
        field_query_chunk_size=args.field_query_chunk_size,
        field_control_chunk_size=args.field_control_chunk_size,
        fill_value=args.fill_value,
        chunk_depth=args.chunk_depth,
        metrics_sample_step=args.metrics_sample_step,
    )


if __name__ == "__main__":
    main()
