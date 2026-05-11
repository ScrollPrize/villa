from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vesuvius.data import affine
from vesuvius.neural_tracing.autoreg_fiber import fiber_geometry as geom
from vesuvius.neural_tracing.autoreg_fiber.webknossos_annotations import (
    _flattened_trees,
    _load_webknossos_module,
    skeleton_trees_from_tracing_proto,
)


def _load_skeleton_proto_zip(path: Path) -> list[Any]:
    trees: list[Any] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".SkeletonTracing.pb"):
                trees.extend(skeleton_trees_from_tracing_proto(zf.read(name)))
    return trees


def _load_annotation_trees(path: Path) -> list[Any]:
    if path.name.endswith(".skeleton-v0.zip") or path.name.endswith(".skeleton-v1.zip"):
        return _load_skeleton_proto_zip(path)
    wk = _load_webknossos_module()
    annotation = wk.Annotation.load(path)
    return _flattened_trees(annotation)


def _tree_id(tree: Any) -> str:
    return str(getattr(tree, "id", ""))


def _find_tree(trees: Sequence[Any], tree_id: str | int | None) -> Any | None:
    wanted = str(tree_id)
    for tree in trees:
        if _tree_id(tree) == wanted:
            return tree
    return None


def _load_transforms_by_marker(records: Sequence[dict[str, Any]]) -> dict[str, affine.TransformDocument]:
    markers = sorted({str(record["marker"]) for record in records})
    docs: dict[str, affine.TransformDocument] = {}
    for marker in markers:
        if marker not in geom.TRANSFORM_URL_BY_MARKER:
            raise ValueError(f"unknown fiber marker {marker!r}")
        docs[marker] = affine.read_transform_json(geom.TRANSFORM_URL_BY_MARKER[marker])
    return docs


def build_fiber_cache(
    *,
    inventory_json: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    densify_step: float | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    inventory_path = Path(inventory_json).expanduser().resolve()
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = [
        dict(record)
        for record in payload.get("records", [])
        if record.get("status") == "accepted" and record.get("downloaded_path") and record.get("tree_id") is not None
    ]
    if limit is not None:
        records = records[: int(limit)]
    transforms = _load_transforms_by_marker(records)
    out_dir = Path(output_dir).expanduser().resolve()
    manifest_path = Path(manifest_json).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    trees_by_path: dict[Path, list[Any]] = {}
    cache_paths: list[str] = []
    warnings: list[str] = []
    target_volume_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    failed = 0

    for record in records:
        annotation_path = Path(str(record["downloaded_path"])).expanduser()
        if not annotation_path.is_absolute():
            annotation_path = (inventory_path.parent / annotation_path).resolve()
        if annotation_path not in trees_by_path:
            try:
                trees_by_path[annotation_path] = _load_annotation_trees(annotation_path)
            except Exception as exc:
                failed += 1
                warnings.append(f"failed to load {annotation_path}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
        tree = _find_tree(trees_by_path[annotation_path], record.get("tree_id"))
        if tree is None:
            failed += 1
            warnings.append(
                f"missing tree_id={record.get('tree_id')!r} in {annotation_path}; "
                f"available={[_tree_id(tree) for tree in trees_by_path[annotation_path]][:10]!r}"
            )
            continue
        marker = str(record["marker"])
        try:
            fiber = geom.tree_to_fiber_path(
                tree,
                annotation_id=str(record["annotation_id"]),
                marker=marker,
                target_volume=str(record["target_volume"]),
                new_to_old_matrix_xyz=transforms[marker].matrix_xyz,
                densify_step=densify_step,
            )
            path = geom.write_fiber_cache(fiber, out_dir)
        except Exception as exc:
            failed += 1
            warnings.append(
                f"failed to convert annotation={record.get('annotation_id')} tree={record.get('tree_id')}: "
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )
            continue
        cache_paths.append(str(path))
        target_volume_counts[str(record["target_volume"])] += 1
        marker_counts[marker] += 1

    elapsed = time.perf_counter() - started
    manifest = {
        "inventory_json": str(inventory_path),
        "output_dir": str(out_dir),
        "fiber_cache_paths": sorted(cache_paths),
        "counts": {
            "records_considered": len(records),
            "fibers_written": len(cache_paths),
            "failed": int(failed),
            "target_volume_counts": dict(sorted(target_volume_counts.items())),
            "marker_counts": dict(sorted(marker_counts.items())),
        },
        "transforms": {
            marker: {
                "url": geom.TRANSFORM_URL_BY_MARKER[marker],
                "checksum": affine.matrix_checksum(doc.matrix_xyz),
                "fixed_volume": doc.fixed_volume,
            }
            for marker, doc in sorted(transforms.items())
        },
        "benchmark": {
            "elapsed_seconds": float(elapsed),
            "fibers_per_second": 0.0 if elapsed <= 0.0 else float(len(cache_paths)) / float(elapsed),
        },
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def resample_fiber_cache(
    *,
    source_manifest_json: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    target_spacing: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    src_path = Path(source_manifest_json).expanduser().resolve()
    src_payload = json.loads(src_path.read_text(encoding="utf-8"))
    out_dir = Path(output_dir).expanduser().resolve()
    manifest_path = Path(manifest_json).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cache_paths: list[str] = []
    warnings: list[str] = []
    target_volume_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    point_count_in = 0
    point_count_out = 0
    skipped_short = 0
    failed_load = 0
    convention = f"wk_xyz_to_new_zyx_resampled_{float(target_spacing):.4g}"

    for raw_path in src_payload.get("fiber_cache_paths", []):
        try:
            points_zyx, metadata = geom.load_fiber_cache(raw_path)
        except Exception as exc:
            failed_load += 1
            warnings.append(f"failed to load {raw_path}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        point_count_in += int(points_zyx.shape[0])
        resampled = geom.resample_polyline_uniform(
            points_zyx.astype(np.float64),
            target_spacing=float(target_spacing),
        )
        if resampled.shape[0] < 2:
            skipped_short += 1
            continue
        point_count_out += int(resampled.shape[0])
        fiber = geom.FiberPath(
            annotation_id=str(metadata["annotation_id"]),
            tree_id=str(metadata["tree_id"]),
            target_volume=str(metadata["target_volume"]),
            marker=str(metadata["marker"]),
            source_points_xyz=geom.zyx_to_xyz(resampled),
            points_zyx=resampled,
            transform_checksum=str(metadata["transform_checksum"]),
            densify_step=float(target_spacing),
            coordinate_convention=convention,
        )
        written_path = geom.write_fiber_cache(fiber, out_dir)
        cache_paths.append(str(written_path))
        target_volume_counts[str(metadata.get("target_volume", "__unknown__"))] += 1
        marker_counts[str(metadata.get("marker", "__unknown__"))] += 1

    elapsed = time.perf_counter() - started
    manifest = {
        "source_manifest_json": str(src_path),
        "output_dir": str(out_dir),
        "fiber_cache_paths": sorted(cache_paths),
        "resample": {
            "target_spacing": float(target_spacing),
            "coordinate_convention": convention,
            "point_count_in": int(point_count_in),
            "point_count_out": int(point_count_out),
            "skipped_short": int(skipped_short),
        },
        "counts": {
            "records_considered": len(src_payload.get("fiber_cache_paths", [])),
            "fibers_written": len(cache_paths),
            "failed": int(skipped_short + failed_load),
            "target_volume_counts": dict(sorted(target_volume_counts.items())),
            "marker_counts": dict(sorted(marker_counts.items())),
        },
        "transforms": src_payload.get("transforms", {}),
        "benchmark": {
            "elapsed_seconds": float(elapsed),
            "fibers_per_second": 0.0 if elapsed <= 0.0 else float(len(cache_paths)) / float(elapsed),
        },
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build autoreg_fiber .npz caches from a WebKnossos inventory.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--inventory-json")
    src.add_argument("--resample-from-cache", help="Path to an existing manifest.json to resample.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--densify-step", type=float, default=None)
    parser.add_argument("--target-spacing", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.resample_from_cache is not None:
        if args.target_spacing is None or float(args.target_spacing) <= 0.0:
            raise SystemExit("--target-spacing > 0 is required with --resample-from-cache")
        manifest = resample_fiber_cache(
            source_manifest_json=args.resample_from_cache,
            output_dir=args.output_dir,
            manifest_json=args.manifest_json,
            target_spacing=float(args.target_spacing),
        )
        counts = manifest["counts"]
        rs = manifest["resample"]
        print(
            "Resampled fiber cache: "
            f"{counts['fibers_written']} fibers written, "
            f"{counts['failed']} failed/skipped, "
            f"points {rs['point_count_in']} -> {rs['point_count_out']}, "
            f"target_spacing={rs['target_spacing']}"
        )
        return
    manifest = build_fiber_cache(
        inventory_json=args.inventory_json,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        densify_step=args.densify_step,
        limit=args.limit,
    )
    counts = manifest["counts"]
    print(
        "Autoreg fiber cache: "
        f"{counts['fibers_written']} fibers written, "
        f"{counts['failed']} failed, "
        f"targets={counts['target_volume_counts']}"
    )


if __name__ == "__main__":
    main()


__all__ = ["build_fiber_cache", "resample_fiber_cache", "main"]
