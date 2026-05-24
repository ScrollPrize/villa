import argparse
import json
from pathlib import Path

import numpy as np


def read_obj_vertices(path: str | Path) -> np.ndarray:
    vertices = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4 or parts[0] != "v":
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64)


def _sample_indices(vertex_count: int, max_points: int | None) -> np.ndarray:
    if max_points is not None and max_points < 0:
        raise ValueError("max_points must be non-negative")
    if max_points is None or max_points == 0 or vertex_count <= max_points:
        return np.arange(vertex_count, dtype=np.int64)
    return np.linspace(0, vertex_count - 1, max_points, dtype=np.int64)


def build_control_points(
    source_vertices: np.ndarray,
    registered_vertices: np.ndarray,
    max_points: int | None = None,
) -> list[dict]:
    if source_vertices.shape != registered_vertices.shape:
        raise ValueError(
            "Source and registered meshes must have the same number of vertices "
            f"and coordinates: {source_vertices.shape} != {registered_vertices.shape}"
        )
    if source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N, 3), got {source_vertices.shape}")

    controls = []
    for vertex_index in _sample_indices(source_vertices.shape[0], max_points):
        source = source_vertices[vertex_index]
        target = registered_vertices[vertex_index]
        displacement = target - source
        controls.append(
            {
                "vertex_index": int(vertex_index),
                "source_xyz": [float(value) for value in source],
                "target_xyz": [float(value) for value in target],
                "displacement_xyz": [float(value) for value in displacement],
            }
        )
    return controls


def export_control_points(
    mesh_pairs: list[tuple[Path, Path]],
    max_points_per_mesh: int | None = None,
) -> dict:
    output_pairs = []
    output_controls = []
    for mesh_pair_index, (source_path, registered_path) in enumerate(mesh_pairs):
        source_vertices = read_obj_vertices(source_path)
        registered_vertices = read_obj_vertices(registered_path)
        controls = build_control_points(
            source_vertices,
            registered_vertices,
            max_points=max_points_per_mesh,
        )
        output_pairs.append(
            {
                "source_mesh": str(source_path),
                "registered_mesh": str(registered_path),
                "source_vertex_count": int(source_vertices.shape[0]),
                "registered_vertex_count": int(registered_vertices.shape[0]),
                "sampled_control_points": len(controls),
            }
        )
        for control in controls:
            control_with_pair = dict(control)
            control_with_pair["mesh_pair_index"] = mesh_pair_index
            output_controls.append(control_with_pair)

    return {
        "schema_version": "1.0.0",
        "coordinate_order": "xyz",
        "description": (
            "Sparse displacement controls derived from source and registered OBJ vertex pairs. "
            "source_xyz + displacement_xyz = target_xyz."
        ),
        "mesh_pairs": output_pairs,
        "control_points": output_controls,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export sparse displacement control points from original and registered OBJ meshes."
        )
    )
    parser.add_argument(
        "--mesh-pair",
        action="append",
        nargs=2,
        metavar=("SOURCE_OBJ", "REGISTERED_OBJ"),
        required=True,
        help="Source and registered OBJ pair.",
    )
    def parse_non_negative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("max-points-per-mesh must be non-negative")
        return parsed

    parser.add_argument("--output", required=True, help="Path for the output JSON file.")
    parser.add_argument(
        "--max-points-per-mesh",
        type=parse_non_negative_int,
        default=0,
        help="Uniformly sample at most this many vertices per mesh; 0 keeps all vertices.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    data = export_control_points(
        [(Path(source), Path(registered)) for source, registered in args.mesh_pair],
        max_points_per_mesh=args.max_points_per_mesh,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
