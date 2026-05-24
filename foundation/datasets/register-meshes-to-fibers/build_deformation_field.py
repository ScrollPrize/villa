import argparse
import json
from pathlib import Path

import numpy as np


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


def load_control_points(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("coordinate_order") != "xyz":
        raise ValueError("deformation controls must use coordinate_order='xyz'")

    positions = []
    displacements = []
    for control in data.get("control_points", []):
        source = np.asarray(control["source_xyz"], dtype=np.float64)
        if "displacement_xyz" in control:
            displacement = np.asarray(control["displacement_xyz"], dtype=np.float64)
        else:
            displacement = np.asarray(control["target_xyz"], dtype=np.float64) - source
        positions.append(source)
        displacements.append(displacement)

    if not positions:
        raise ValueError("control file does not contain any control_points")

    return np.vstack(positions), np.vstack(displacements)


def _select_nearest_by_distance_then_index(
    dists: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_dists = np.empty((dists.shape[0], k), dtype=dists.dtype)
    nearest_indices = np.empty((indices.shape[0], k), dtype=indices.dtype)
    for row in range(dists.shape[0]):
        order = np.lexsort((indices[row], dists[row]))[:k]
        nearest_dists[row] = dists[row, order]
        nearest_indices[row] = indices[row, order]
    return nearest_dists, nearest_indices


def interpolate_idw(
    query_xyz: np.ndarray,
    controls_xyz: np.ndarray,
    displacements_xyz: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    epsilon: float = 1e-12,
    query_chunk_size: int | None = None,
    control_chunk_size: int | None = None,
) -> np.ndarray:
    if controls_xyz.shape != displacements_xyz.shape:
        raise ValueError("control positions and displacements must have matching shape")
    if controls_xyz.ndim != 2 or controls_xyz.shape[1] != 3:
        raise ValueError(f"expected controls with shape (N, 3), got {controls_xyz.shape}")
    if query_xyz.ndim != 2 or query_xyz.shape[1] != 3:
        raise ValueError(f"expected query points with shape (N, 3), got {query_xyz.shape}")
    if controls_xyz.shape[0] == 0:
        raise ValueError("at least one control point is required")
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if query_chunk_size is not None and int(query_chunk_size) <= 0:
        raise ValueError("query_chunk_size must be positive")
    if control_chunk_size is not None and int(control_chunk_size) <= 0:
        raise ValueError("control_chunk_size must be positive")

    k = min(int(k), controls_xyz.shape[0])
    output = np.empty((query_xyz.shape[0], 3), dtype=np.float64)
    if query_xyz.shape[0] == 0:
        return output
    step = query_xyz.shape[0] if query_chunk_size is None else int(query_chunk_size)
    for start in range(0, query_xyz.shape[0], step):
        stop = min(start + step, query_xyz.shape[0])
        chunk = query_xyz[start:stop]
        if control_chunk_size is None:
            diff = chunk[:, None, :] - controls_xyz[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            control_indices = np.broadcast_to(np.arange(controls_xyz.shape[0], dtype=np.int64), dists.shape)
            nearest_dists, nearest_indices = _select_nearest_by_distance_then_index(dists, control_indices, k)
        else:
            control_step = int(control_chunk_size)
            nearest_dists = np.full((chunk.shape[0], k), np.inf, dtype=np.float64)
            nearest_indices = np.full((chunk.shape[0], k), -1, dtype=np.int64)
            for control_start in range(0, controls_xyz.shape[0], control_step):
                control_stop = min(control_start + control_step, controls_xyz.shape[0])
                control_chunk = controls_xyz[control_start:control_stop]
                diff = chunk[:, None, :] - control_chunk[None, :, :]
                candidate_dists = np.concatenate([nearest_dists, np.linalg.norm(diff, axis=2)], axis=1)
                candidate_indices = np.concatenate(
                    [
                        nearest_indices,
                        np.arange(control_start, control_stop, dtype=np.int64)[None, :].repeat(chunk.shape[0], axis=0),
                    ],
                    axis=1,
                )
                nearest_dists, nearest_indices = _select_nearest_by_distance_then_index(
                    candidate_dists,
                    candidate_indices,
                    k,
                )

        for row, row_nearest_indices in enumerate(nearest_indices):
            row_dists = nearest_dists[row]
            exact = np.where(row_dists <= epsilon)[0]
            if exact.size:
                output[start + row] = displacements_xyz[row_nearest_indices[exact[0]]]
                continue
            weights = 1.0 / np.power(row_dists, power)
            output[start + row] = (
                weights[:, None] * displacements_xyz[row_nearest_indices]
            ).sum(axis=0) / weights.sum()
    return output


def build_grid_field(
    controls_xyz: np.ndarray,
    displacements_xyz: np.ndarray,
    grid_shape_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    k: int = 8,
    power: float = 2.0,
    query_chunk_size: int | None = None,
    control_chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_positive(grid_shape_xyz, "grid shape")
    _require_positive(spacing_xyz, "spacing")

    x_count, y_count, z_count = grid_shape_xyz
    x = origin_xyz[0] + np.arange(x_count, dtype=np.float64) * spacing_xyz[0]
    y = origin_xyz[1] + np.arange(y_count, dtype=np.float64) * spacing_xyz[1]
    z = origin_xyz[2] + np.arange(z_count, dtype=np.float64) * spacing_xyz[2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    query_xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    interpolated = interpolate_idw(
        query_xyz,
        controls_xyz,
        displacements_xyz,
        k=k,
        power=power,
        query_chunk_size=query_chunk_size,
        control_chunk_size=control_chunk_size,
    )
    field = interpolated.reshape((z_count, y_count, x_count, 3))
    return (
        field.astype(np.float32),
        np.asarray(origin_xyz, dtype=np.float64),
        np.asarray(spacing_xyz, dtype=np.float64),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a coarse displacement field from sparse deformation controls."
    )
    parser.add_argument("--controls", required=True, help="Control-point JSON from export_deformation_control_points.py.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument("--grid-shape", required=True, type=lambda value: _parse_positive_xyz(value, int, "grid shape"), help="Grid shape as x,y,z.")
    parser.add_argument("--spacing", required=True, type=lambda value: _parse_positive_xyz(value, float, "spacing"), help="Grid spacing as x,y,z.")
    parser.add_argument("--origin", default="0,0,0", type=lambda value: _parse_xyz(value, float), help="Grid origin as x,y,z.")
    parser.add_argument("--k", type=lambda value: _parse_positive_int(value, "k"), default=8, help="Number of nearest controls used for IDW interpolation.")
    parser.add_argument("--power", type=float, default=2.0, help="Inverse-distance power.")
    parser.add_argument(
        "--query-chunk-size",
        type=lambda value: _parse_positive_int(value, "query-chunk-size"),
        default=None,
        help="Interpolate at most this many grid points per IDW batch; default processes the grid in one batch.",
    )
    parser.add_argument(
        "--control-chunk-size",
        type=lambda value: _parse_positive_int(value, "control-chunk-size"),
        default=None,
        help="Compare at most this many control points per IDW sub-batch; default compares all controls at once.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    controls_xyz, displacements_xyz = load_control_points(args.controls)
    field, origin, spacing = build_grid_field(
        controls_xyz,
        displacements_xyz,
        grid_shape_xyz=args.grid_shape,
        spacing_xyz=args.spacing,
        origin_xyz=args.origin,
        k=args.k,
        power=args.power,
        query_chunk_size=args.query_chunk_size,
        control_chunk_size=args.control_chunk_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        displacement_xyz=field,
        origin_xyz=origin,
        spacing_xyz=spacing,
        coordinate_order=np.array("zyx_grid_xyz_vectors"),
        controls_path=np.array(str(args.controls)),
    )


if __name__ == "__main__":
    main()
