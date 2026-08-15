#!/usr/bin/env python3
"""Transfer categorical labels between TIFXYZ surface renders."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Optional, Sequence

import numpy as np

from vesuvius.utils.cli import HyphenUnderscoreParser

from .core import (
    AffineChoice,
    choose_affine_direction,
    estimate_surface_spacing,
    infer_output_shape,
    load_affine,
    SurfaceMapper,
    transfer_array,
)
from .io import (
    TemporaryRaster,
    StreamingTiffOutputs,
    load_surface,
    read_image,
    read_image_shape,
    sidecar_path,
    write_image,
)
from .native import resolve_rasterizer
from .planar import transfer_array_planar


def _distance_arg(value: str) -> Optional[float]:
    if value.lower() == "auto":
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "distance must be a positive number or 'auto'"
        )
    return parsed


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _coverage_arg(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("coverage must be between 0 and 1")
    return parsed


def _shape_arg(values: Optional[Sequence[int]]) -> Optional[tuple[int, int]]:
    if values is None:
        return None
    shape = int(values[0]), int(values[1])
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"invalid output shape: {shape}")
    return shape


def _jsonable_affine_choice(choice: AffineChoice) -> dict:
    data = asdict(choice)
    data["matrix"] = choice.matrix.tolist()
    return data


class ProgressPrinter:
    def __init__(self, label: str) -> None:
        self.label = label
        self.last_percent = -1
        self.started = time.monotonic()

    def __call__(self, complete: int, total: int) -> None:
        percent = int(100 * complete / max(1, total))
        if percent == self.last_percent and complete != total:
            return
        self.last_percent = percent
        elapsed = time.monotonic() - self.started
        rate = complete / elapsed if elapsed > 0 else 0.0
        remaining = (total - complete) / rate if rate > 0 else 0.0
        print(
            f"\r{self.label}: {complete}/{total} steps "
            f"({percent:3d}%, ETA {remaining:,.0f}s)",
            end="\n" if complete == total else "",
            flush=True,
        )


def _resolve_output_shape(args, source, label, target) -> tuple[int, int]:
    explicit = _shape_arg(args.output_shape)
    if args.target_reference is not None:
        reference_shape = read_image_shape(args.target_reference)
        if explicit is not None and explicit != reference_shape:
            raise ValueError(
                f"--output-shape {explicit} disagrees with target reference "
                f"shape {reference_shape}"
            )
        explicit = reference_shape
    return infer_output_shape(
        source, label.shape, target, explicit_shape=explicit
    )


def _choose_affine(args, source, target) -> AffineChoice:
    if args.affine is None:
        return AffineChoice(matrix=np.eye(4), direction="identity")
    matrix = load_affine(args.affine)
    return choose_affine_direction(
        source,
        target,
        matrix,
        direction=args.affine_direction,
        sample_limit=args.affine_sample_points,
    )


def _resolve_max_distance(
    requested: Optional[float], source, target, affine: np.ndarray
) -> tuple[float, dict]:
    source_spacing = estimate_surface_spacing(source, affine)
    target_spacing = estimate_surface_spacing(target)
    automatic = max(1e-3, 0.75 * min(source_spacing, target_spacing))
    return (
        automatic if requested is None else requested,
        {
            "source_spacing_in_target_coordinates": source_spacing,
            "target_spacing": target_spacing,
            "automatic_max_distance": automatic,
        },
    )


def _mapping_preflight(
    source,
    target,
    affine: np.ndarray,
    max_distance: float,
    *,
    nearest_vertices: int,
    sample_limit: int,
    vertex_index: str = "kdtree",
) -> dict:
    """Sample exact point-to-triangle coverage before allocating outputs."""

    assert target.valid is not None
    target_indices = np.flatnonzero(target.valid)
    if target_indices.size == 0:
        raise ValueError(f"target TIFXYZ {target.name!r} has no valid vertices")
    if target_indices.size > sample_limit:
        selection = np.linspace(
            0,
            target_indices.size - 1,
            num=sample_limit,
            dtype=np.int64,
        )
        target_indices = target_indices[selection]
    target_points = np.column_stack(
        (
            target.x.ravel()[target_indices],
            target.y.ravel()[target_indices],
            target.z.ravel()[target_indices],
        )
    )
    mapper = SurfaceMapper(
        source,
        affine=affine,
        nearest_vertices=nearest_vertices,
        vertex_index=vertex_index,
        index_max_distance=max_distance,
    )
    _, _, _, distances = mapper.locate(
        target_points,
        max_distance=math.inf,
    )
    finite = distances[np.isfinite(distances)]
    within_distance = np.isfinite(distances) & (distances <= max_distance)
    report = {
        "sampled_target_valid_vertices": int(target_indices.size),
        "mapped_within_distance": int(within_distance.sum()),
        "mapping_coverage": float(within_distance.mean()),
        "max_distance": float(max_distance),
        "distance_p50": None,
        "distance_p95": None,
        "distance_max": None,
    }
    if finite.size:
        report["distance_p50"] = float(np.percentile(finite, 50))
        report["distance_p95"] = float(np.percentile(finite, 95))
        report["distance_max"] = float(finite.max())
    return report


def _enforce_mapping_coverage(
    *,
    stage_name: str,
    coverage: float,
    minimum: float,
    context: str,
) -> None:
    if coverage >= minimum:
        return
    affine_option = (
        "--stage-one-affine"
        if stage_name == "old-to-updated"
        else "--affine"
    )
    raise ValueError(
        f"{stage_name} {context} mapping coverage {coverage:.6f} is below "
        f"--minimum-mapping-coverage {minimum:.6f}. The TIFXYZ surfaces "
        "likely use different volume frames, or the supplied affine/direction "
        f"is wrong. Supply or correct the registration with "
        f"{affine_option}; automatic affine estimation is intentionally not "
        "performed. Set --minimum-mapping-coverage 0 only for a deliberately "
        "partial-overlap transfer."
    )


def _check_output_paths(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "outputs already exist (pass --overwrite to replace): "
            + ", ".join(str(path) for path in existing)
        )


def run_single(args, stage_name: str = "transfer") -> dict:
    source_path = Path(args.source_tifxyz)
    target_path = Path(args.target_tifxyz)
    label_path = Path(args.label)
    output_path = Path(args.output)

    print(f"Loading source TIFXYZ: {source_path}")
    source = load_surface(source_path, use_mask=not args.ignore_tifxyz_mask)
    print(f"Loading target TIFXYZ: {target_path}")
    target = load_surface(target_path, use_mask=not args.ignore_tifxyz_mask)
    print(f"Loading label: {label_path}")
    label = read_image(label_path)
    additional_specs = [
        (Path(source_value), Path(output_value))
        for source_value, output_value in getattr(
            args, "additional_labels", ()
        )
    ]
    additional_labels = []
    for additional_path, _ in additional_specs:
        print(f"Loading additional label: {additional_path}")
        additional_label = read_image(additional_path)
        if additional_label.shape != label.shape:
            raise ValueError(
                "all labels in one geometry pass must have the same shape; "
                f"got {additional_label.shape} and {label.shape}"
            )
        additional_labels.append(additional_label)
    source_validity_path = (
        Path(args.source_validity)
        if getattr(args, "source_validity", None) is not None
        else None
    )
    source_validity = None
    if source_validity_path is not None:
        print(f"Loading source validity: {source_validity_path}")
        source_validity = read_image(source_validity_path)
        if source_validity.shape != label.shape:
            raise ValueError(
                "source validity shape must match source label shape; "
                f"got {source_validity.shape} and {label.shape}"
            )
    if not np.issubdtype(label.dtype, np.integer) and label.dtype != np.bool_:
        raise ValueError(
            f"categorical label must have integer/bool dtype; got {label.dtype}"
        )
    for additional_label in additional_labels:
        if (
            not np.issubdtype(additional_label.dtype, np.integer)
            and additional_label.dtype != np.bool_
        ):
            raise ValueError(
                "categorical label must have integer/bool dtype; got "
                f"{additional_label.dtype}"
            )
    if label.dtype == np.bool_:
        fill_in_range = args.fill_value in (0, 1)
    else:
        limits = np.iinfo(label.dtype)
        fill_in_range = limits.min <= args.fill_value <= limits.max
    if not fill_in_range:
        raise ValueError(
            f"--fill-value {args.fill_value} is outside label dtype "
            f"{label.dtype}"
        )
    fill_value = np.asarray(args.fill_value, dtype=label.dtype).item()
    additional_fill_values = []
    for additional_label in additional_labels:
        if additional_label.dtype == np.bool_:
            additional_fill_in_range = args.fill_value in (0, 1)
        else:
            additional_limits = np.iinfo(additional_label.dtype)
            additional_fill_in_range = (
                additional_limits.min
                <= args.fill_value
                <= additional_limits.max
            )
        if not additional_fill_in_range:
            raise ValueError(
                f"--fill-value {args.fill_value} is outside label dtype "
                f"{additional_label.dtype}"
            )
        additional_fill_values.append(
            np.asarray(args.fill_value, dtype=additional_label.dtype).item()
        )

    output_shape = _resolve_output_shape(args, source, label, target)
    affine_choice = _choose_affine(args, source, target)
    max_distance, spacing_report = _resolve_max_distance(
        args.max_distance, source, target, affine_choice.matrix
    )
    minimum_mapping_coverage = float(
        getattr(args, "minimum_mapping_coverage", 0.01)
    )
    preflight_sample_points = int(
        getattr(args, "preflight_sample_points", 4096)
    )
    canvas_offset = getattr(args, "label_canvas_offset", None)
    if canvas_offset is None:
        offset_full = (0.0, 0.0)
        label_offset_yx = (0.0, 0.0)
    else:
        offset_full = float(canvas_offset[0]), float(canvas_offset[1])
        if not all(math.isfinite(value) for value in offset_full):
            raise ValueError(
                f"--label-canvas-offset must be finite; got {offset_full}"
            )
        full_height, full_width = source.full_resolution_shape
        # The offset is declared in full-resolution source-canvas pixels;
        # convert it to the units of the label raster actually provided.
        label_offset_yx = (
            offset_full[0] * label.shape[0] / full_height,
            offset_full[1] * label.shape[1] / full_width,
        )

    planar = bool(getattr(args, "planar", False))
    if planar and additional_labels:
        raise ValueError("additional labels are not supported with --planar")
    if planar and args.distance_output is not None:
        raise ValueError(
            "--distance-output is not available with --planar; the global "
            "affine has no per-pixel surface distances"
        )
    uv_cache_prefix = getattr(args, "uv_cache", None)
    uv_cache_path = (
        None
        if uv_cache_prefix is None
        else f"{uv_cache_prefix}.{stage_name}.npz"
    )
    valid_path = (
        Path(args.valid_output)
        if args.valid_output is not None
        else sidecar_path(output_path, "valid", ".tif")
    )
    distance_path = (
        Path(args.distance_output)
        if args.distance_output is not None
        else None
    )
    report_path = (
        Path(args.report_output)
        if args.report_output is not None
        else sidecar_path(output_path, "report", ".json")
    )
    all_outputs = [output_path, valid_path, report_path]
    additional_paths = []
    for additional_source, additional_output in additional_specs:
        additional_valid = sidecar_path(additional_output, "valid", ".tif")
        additional_report = sidecar_path(
            additional_output, "report", ".json"
        )
        additional_paths.append(
            (
                additional_source,
                additional_output,
                additional_valid,
                additional_report,
            )
        )
        all_outputs.extend(
            [additional_output, additional_valid, additional_report]
        )
    if distance_path is not None:
        all_outputs.append(distance_path)
    if not args.dry_run:
        _check_output_paths(all_outputs, args.overwrite)

    logical_output_bytes = (
        output_shape[0]
        * output_shape[1]
        * (
            label.dtype.itemsize
            + sum(item.dtype.itemsize for item in additional_labels)
            + 1
            + (4 if distance_path else 0)
        )
    )
    stream_output = (
        not planar
        and distance_path is None
        and not bool(getattr(args, "no_stream_output", False))
        and args.tile_size % 16 == 0
        and all(
            path.suffix.lower() in {".tif", ".tiff"}
            for path in [output_path, valid_path]
            + [item[1] for item in additional_paths]
        )
    )
    requested_rasterizer = getattr(args, "rasterizer", "auto")
    selected_rasterizer = (
        "planar" if planar else resolve_rasterizer(requested_rasterizer)
    )
    if stream_output:
        tile_rows = math.ceil(output_shape[0] / args.tile_size)
        tile_cols = math.ceil(output_shape[1] / args.tile_size)
        tile_count = tile_rows * tile_cols
        requested_workers = getattr(args, "workers", None)
        if requested_workers is None:
            requested_workers = os.cpu_count() or 1
        effective_workers = min(max(1, requested_workers), tile_count)
        bytes_per_pixel = (
            label.dtype.itemsize
            + sum(item.dtype.itemsize for item in additional_labels)
            + 1
        )
        tile_payload_bytes = args.tile_size**2 * bytes_per_pixel
        # The ordered worker window holds at most 2 * workers completed tile
        # payloads. Encoder queues add at most two more payloads in aggregate.
        estimated_buffered_output_bytes = (
            2 * effective_workers + 2
        ) * tile_payload_bytes
        temporary_full_raster_bytes = 0
    else:
        estimated_buffered_output_bytes = logical_output_bytes
        temporary_full_raster_bytes = logical_output_bytes
    report = {
        "stage": stage_name,
        "source_tifxyz": str(source_path),
        "target_tifxyz": str(target_path),
        "source_label": str(label_path),
        "source_validity": (
            str(source_validity_path)
            if source_validity_path is not None
            else None
        ),
        "output": str(output_path),
        "valid_output": str(valid_path),
        "distance_output": (
            str(distance_path) if distance_path is not None else None
        ),
        "report_output": str(report_path),
        "source_stored_shape": list(source.shape),
        "source_full_resolution_shape": list(source.full_resolution_shape),
        "source_scale_yx": list(source.scale_yx),
        "source_label_shape": list(label.shape),
        "target_stored_shape": list(target.shape),
        "target_full_resolution_shape": list(target.full_resolution_shape),
        "target_scale_yx": list(target.scale_yx),
        "output_shape": list(output_shape),
        "logical_output_bytes": logical_output_bytes,
        # Preserve this legacy field's full-raster meaning for report readers.
        "estimated_working_output_bytes": logical_output_bytes,
        "estimated_buffered_output_bytes": estimated_buffered_output_bytes,
        "temporary_full_raster_bytes": temporary_full_raster_bytes,
        "output_storage_mode": (
            "streamed-compressed-tiles" if stream_output else "temporary-rasters"
        ),
        "rasterizer_requested": requested_rasterizer,
        "rasterizer": selected_rasterizer,
        "affine": _jsonable_affine_choice(affine_choice),
        "label_canvas_offset_full_resolution_px": list(offset_full),
        "label_canvas_offset_label_px": list(label_offset_yx),
        "max_distance": max_distance,
        "minimum_mapping_coverage": minimum_mapping_coverage,
        "preflight_sample_points": preflight_sample_points,
        **spacing_report,
        "nearest_vertices": args.nearest_vertices,
        "vertex_index": getattr(args, "vertex_index", "kdtree"),
        "tile_size": args.tile_size,
        "query_batch_size": args.query_batch_size,
        "fill_value": args.fill_value,
        "planar": planar,
        "fill_seams": bool(getattr(args, "fill_seams", False)),
        "workers": getattr(args, "workers", None),
        "uv_cache": uv_cache_path,
    }
    if planar:
        report["planar_sample_vertices"] = int(
            getattr(args, "planar_sample_vertices", 200_000)
        )

    print(
        f"Inferred output: {output_shape[0]}x{output_shape[1]} "
        f"({logical_output_bytes / 1024**3:.2f} GiB logical outputs; "
        + ("streamed tiles" if stream_output else "temporary rasters")
        + ")"
    )
    print(
        f"Affine direction: {affine_choice.direction}; "
        f"maximum surface distance: {max_distance:.4g} target voxels"
    )
    preflight = _mapping_preflight(
        source,
        target,
        affine_choice.matrix,
        max_distance,
        nearest_vertices=args.nearest_vertices,
        sample_limit=preflight_sample_points,
        vertex_index=getattr(args, "vertex_index", "kdtree"),
    )
    report["preflight_mapping"] = preflight
    print(
        "Geometry preflight: "
        f"{preflight['mapped_within_distance']}/"
        f"{preflight['sampled_target_valid_vertices']} sampled target "
        f"vertices accepted ({preflight['mapping_coverage']:.2%})"
    )
    _enforce_mapping_coverage(
        stage_name=stage_name,
        coverage=float(preflight["mapping_coverage"]),
        minimum=minimum_mapping_coverage,
        context="preflight",
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for _, additional_output, _, _ in additional_paths:
        additional_output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if stream_output:
            streaming = stack.enter_context(
                StreamingTiffOutputs(
                    [
                        (output_path, label.dtype, fill_value),
                        *[
                            (path_info[1], additional_label.dtype, additional_fill)
                            for additional_label, additional_fill, path_info in zip(
                                additional_labels,
                                additional_fill_values,
                                additional_paths,
                            )
                        ],
                        (valid_path, np.dtype(np.uint8), 0),
                    ],
                    output_shape,
                    args.tile_size,
                )
            )
            output_raster = None
            additional_rasters = []
            valid_raster = None
        else:
            streaming = None
            output_raster = stack.enter_context(
                TemporaryRaster(
                    output_path.parent,
                    output_shape,
                    label.dtype,
                    fill_value,
                    ".label-transfer-",
                )
            )
            additional_rasters = [
                stack.enter_context(
                    TemporaryRaster(
                        additional_output.parent,
                        output_shape,
                        additional_label.dtype,
                        additional_fill,
                        ".label-transfer-",
                    )
                )
                for additional_label, additional_fill, (
                    _,
                    additional_output,
                    _,
                    _,
                ) in zip(
                    additional_labels,
                    additional_fill_values,
                    additional_paths,
                )
            ]
            valid_raster = stack.enter_context(
                TemporaryRaster(
                    output_path.parent,
                    output_shape,
                    np.dtype(np.uint8),
                    0,
                    ".label-valid-",
                )
            )
        if distance_path is None:
            distance_array = None
        else:
            distance_array = stack.enter_context(
                TemporaryRaster(
                    output_path.parent,
                    output_shape,
                    np.dtype(np.float32),
                    np.inf,
                    ".label-distance-",
                )
            ).array

        if planar:
            print("Fitting global planar affine from sampled correspondences...")
            _, _, _, mapping_report = transfer_array_planar(
                source,
                target,
                label,
                source_validity=source_validity,
                output_shape=output_shape,
                label_offset_yx=label_offset_yx,
                affine=affine_choice.matrix,
                max_distance=max_distance,
                nearest_vertices=args.nearest_vertices,
                vertex_index=getattr(args, "vertex_index", "kdtree"),
                sample_vertices=int(
                    getattr(args, "planar_sample_vertices", 200_000)
                ),
                fill_value=fill_value,
                output=output_raster.array,
                valid_output=valid_raster.array,
            )
            residual = mapping_report["residual_label_px"]
            print(
                "Planar fit residual (label px): "
                f"p50 {residual['p50']:.3f}, p95 {residual['p95']:.3f}, "
                f"max {residual['max']:.3f} over "
                f"{mapping_report['fit_inliers']} inliers"
            )
        else:
            _, _, _, stats = transfer_array(
                source,
                target,
                label,
                source_validity=source_validity,
                output_shape=output_shape,
                label_offset_yx=label_offset_yx,
                affine=affine_choice.matrix,
                max_distance=max_distance,
                nearest_vertices=args.nearest_vertices,
                vertex_index=getattr(args, "vertex_index", "kdtree"),
                fill_seams=getattr(args, "fill_seams", False),
                workers=getattr(args, "workers", None),
                uv_cache=uv_cache_path,
                tile_size=args.tile_size,
                query_batch_size=args.query_batch_size,
                fill_value=fill_value,
                output=(None if output_raster is None else output_raster.array),
                additional_source_labels=additional_labels,
                additional_outputs=[
                    item.array for item in additional_rasters
                ],
                valid_output=(None if valid_raster is None else valid_raster.array),
                distance_output=distance_array,
                materialize_output=not stream_output,
                rasterizer=selected_rasterizer,
                tile_callback=(
                    None
                    if streaming is None
                    else lambda bounds, labels, validity: streaming.write_tile(
                        bounds, [*labels, validity]
                    )
                ),
                progress=ProgressPrinter(stage_name),
            )
            mapping_report = stats.as_dict()
            _enforce_mapping_coverage(
                stage_name=stage_name,
                coverage=float(mapping_report["mapping_coverage"]),
                minimum=minimum_mapping_coverage,
                context="final",
            )
        if stream_output:
            print(f"Finishing streamed label outputs: {output_path}")
        else:
            assert output_raster is not None
            assert valid_raster is not None
            print(f"Writing label: {output_path}")
            write_image(output_path, output_raster.array)
            for additional_raster, (
                _,
                additional_output,
                _,
                _,
            ) in zip(additional_rasters, additional_paths):
                print(f"Writing additional label: {additional_output}")
                write_image(additional_output, additional_raster.array)
            print(f"Writing mapping validity: {valid_path}")
            write_image(valid_path, valid_raster.array)
        if distance_path is not None and distance_array is not None:
            print(f"Writing mapping distances: {distance_path}")
            write_image(distance_path, distance_array)

    report["mapping"] = mapping_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Writing report: {report_path}")
    additional_reports = []
    for (
        additional_source,
        additional_output,
        additional_valid,
        additional_report_path,
    ) in additional_paths:
        print(f"Writing mapping validity: {additional_valid}")
        shutil.copyfile(valid_path, additional_valid)
        additional_report = dict(report)
        additional_report.update(
            {
                "source_label": str(additional_source),
                "output": str(additional_output),
                "valid_output": str(additional_valid),
                "report_output": str(additional_report_path),
            }
        )
        additional_report_path.parent.mkdir(parents=True, exist_ok=True)
        with additional_report_path.open("w", encoding="utf-8") as handle:
            json.dump(additional_report, handle, indent=2)
            handle.write("\n")
        print(f"Writing report: {additional_report_path}")
        additional_reports.append(additional_report)
    if additional_reports:
        report["additional_reports"] = additional_reports
    return report


def _single_namespace_from_pipeline(
    args,
    *,
    source_tifxyz: str,
    target_tifxyz: str,
    label: str,
    output: str,
    affine: Optional[str],
    affine_direction: str,
    max_distance: Optional[float],
    target_reference: Optional[str],
    output_shape: Optional[Sequence[int]],
    source_validity: Optional[str] = None,
    label_canvas_offset: Optional[Sequence[float]] = None,
    additional_labels: Optional[Sequence[Sequence[str]]] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        source_tifxyz=source_tifxyz,
        target_tifxyz=target_tifxyz,
        label=label,
        additional_labels=list(additional_labels or ()),
        source_validity=source_validity,
        label_canvas_offset=label_canvas_offset,
        output=output,
        affine=affine,
        affine_direction=affine_direction,
        affine_sample_points=args.affine_sample_points,
        preflight_sample_points=getattr(args, "preflight_sample_points", 4096),
        minimum_mapping_coverage=getattr(
            args, "minimum_mapping_coverage", 0.01
        ),
        max_distance=max_distance,
        nearest_vertices=args.nearest_vertices,
        vertex_index=getattr(args, "vertex_index", "kdtree"),
        fill_seams=getattr(args, "fill_seams", False),
        workers=getattr(args, "workers", None),
        uv_cache=getattr(args, "uv_cache", None),
        tile_size=args.tile_size,
        query_batch_size=args.query_batch_size,
        fill_value=args.fill_value,
        output_shape=output_shape,
        target_reference=target_reference,
        valid_output=None,
        distance_output=None,
        report_output=None,
        ignore_tifxyz_mask=args.ignore_tifxyz_mask,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


def run_pipeline(args) -> dict:
    intermediate = (
        Path(args.intermediate_output)
        if args.intermediate_output
        else sidecar_path(Path(args.output), "updated-2.4um", ".tif")
    )
    final_output = Path(args.output)
    additional_transfers = [
        (str(source), str(intermediate_value), str(final_value))
        for source, intermediate_value, final_value in getattr(
            args, "additional_labels", ()
        )
    ]
    pipeline_outputs = [
        intermediate,
        sidecar_path(intermediate, "valid", ".tif"),
        sidecar_path(intermediate, "report", ".json"),
        final_output,
        sidecar_path(final_output, "valid", ".tif"),
        sidecar_path(final_output, "report", ".json"),
    ]
    for _, intermediate_value, final_value in additional_transfers:
        for value in (Path(intermediate_value), Path(final_value)):
            pipeline_outputs.extend(
                [
                    value,
                    sidecar_path(value, "valid", ".tif"),
                    sidecar_path(value, "report", ".json"),
                ]
            )
    canonical_outputs = [
        path.expanduser().resolve(strict=False) for path in pipeline_outputs
    ]
    if len(set(canonical_outputs)) != len(canonical_outputs):
        raise ValueError(
            "pipeline output paths collide: "
            + ", ".join(str(path) for path in pipeline_outputs)
        )
    if not args.dry_run:
        _check_output_paths(pipeline_outputs, args.overwrite)

    stage_one_args = _single_namespace_from_pipeline(
        args,
        source_tifxyz=args.old_tifxyz,
        target_tifxyz=args.updated_tifxyz,
        label=args.label,
        output=str(intermediate),
        affine=getattr(args, "same_volume_affine", None),
        affine_direction=getattr(
            args, "same_volume_affine_direction", "forward"
        ),
        max_distance=args.same_volume_max_distance,
        target_reference=args.updated_reference,
        output_shape=args.intermediate_shape,
        label_canvas_offset=getattr(args, "label_canvas_offset", None),
        additional_labels=[
            (source, intermediate_value)
            for source, intermediate_value, _ in additional_transfers
        ],
    )
    stage_one = run_single(stage_one_args, stage_name="old-to-updated")
    if args.dry_run:
        # The final shape and affine direction can be resolved without writing
        # the intermediate label; only its inferred shape is needed.
        updated = load_surface(
            args.updated_tifxyz, use_mask=not args.ignore_tifxyz_mask
        )
        target = load_surface(
            args.target_tifxyz, use_mask=not args.ignore_tifxyz_mask
        )
        intermediate_shape = tuple(stage_one["output_shape"])
        final_explicit = _shape_arg(args.output_shape)
        if args.target_reference:
            reference_shape = read_image_shape(args.target_reference)
            if final_explicit is not None and final_explicit != reference_shape:
                raise ValueError(
                    f"--output-shape {final_explicit} disagrees with target "
                    f"reference shape {reference_shape}"
                )
            final_explicit = reference_shape
        final_shape = infer_output_shape(
            updated, intermediate_shape, target, explicit_shape=final_explicit
        )
        affine_choice = choose_affine_direction(
            updated,
            target,
            load_affine(args.affine),
            direction=args.affine_direction,
            sample_limit=args.affine_sample_points,
        )
        final_distance, spacing_report = _resolve_max_distance(
            args.cross_volume_max_distance,
            updated,
            target,
            affine_choice.matrix,
        )
        preflight = _mapping_preflight(
            updated,
            target,
            affine_choice.matrix,
            final_distance,
            nearest_vertices=args.nearest_vertices,
            sample_limit=getattr(args, "preflight_sample_points", 4096),
            vertex_index=getattr(args, "vertex_index", "kdtree"),
        )
        _enforce_mapping_coverage(
            stage_name="updated-to-target",
            coverage=float(preflight["mapping_coverage"]),
            minimum=float(getattr(args, "minimum_mapping_coverage", 0.01)),
            context="preflight",
        )
        result = {
            "stage_one": stage_one,
            "stage_two": {
                "source_label_shape": list(intermediate_shape),
                "source_validity": stage_one["valid_output"],
                "output_shape": list(final_shape),
                "affine": _jsonable_affine_choice(affine_choice),
                "max_distance": final_distance,
                "minimum_mapping_coverage": float(
                    getattr(args, "minimum_mapping_coverage", 0.01)
                ),
                "preflight_mapping": preflight,
                **spacing_report,
            },
        }
        print(json.dumps(result["stage_two"], indent=2))
        return result

    stage_two_args = _single_namespace_from_pipeline(
        args,
        source_tifxyz=args.updated_tifxyz,
        target_tifxyz=args.target_tifxyz,
        label=str(intermediate),
        output=args.output,
        affine=args.affine,
        affine_direction=args.affine_direction,
        max_distance=args.cross_volume_max_distance,
        target_reference=args.target_reference,
        output_shape=args.output_shape,
        source_validity=stage_one["valid_output"],
        additional_labels=[
            (intermediate_value, final_value)
            for _, intermediate_value, final_value in additional_transfers
        ],
    )
    stage_two = run_single(stage_two_args, stage_name="updated-to-target")
    return {"stage_one": stage_one, "stage_two": stage_two}


def _add_common_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nearest-vertices",
        type=_positive_int_arg,
        default=8,
        help="nearest source vertices used to find candidate triangles (default: 8)",
    )
    parser.add_argument(
        "--vertex-index",
        choices=("grid", "kdtree"),
        default="kdtree",
        help=(
            "nearest-vertex search structure: 'kdtree' uses scipy's "
            "parallel cKDTree (default; fastest in benchmarks), 'grid' "
            "buckets the near-uniformly spaced TIFXYZ vertices into a "
            "uniform 3D grid"
        ),
    )
    parser.add_argument(
        "--fill-seams",
        action="store_true",
        help=(
            "fill target pixels whose geometry mapping was rejected (fold "
            "seams, distance failures) by smoothly continuing the measured "
            "UV field across the gaps; filled pixels get validity 128 "
            "instead of 255 so measured and interpolated stay "
            "distinguishable"
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int_arg,
        default=None,
        help=(
            "threads for the mapping batches and output tiles (default: all "
            "cores); outputs and reports are identical for any value"
        ),
    )
    parser.add_argument(
        "--no-stream-output",
        action="store_true",
        help=(
            "materialize complete temporary rasters before writing TIFFs; "
            "mainly useful for benchmarking or non-streaming compatibility"
        ),
    )
    parser.add_argument(
        "--rasterizer",
        choices=("auto", "native", "python"),
        default="auto",
        help=(
            "per-pixel rasterizer: use the compiled kernel when available "
            "(auto), require it (native), or use the NumPy reference (python)"
        ),
    )
    parser.add_argument(
        "--uv-cache",
        default=None,
        help=(
            "path prefix for caching the mapped UV field, which depends only "
            "on the surface pair and matching parameters — not on the label. "
            "The file is written as <prefix>.<stage>.npz; labels of the same "
            "segment reuse it and skip the mapping phase entirely. A cache "
            "that does not match the current configuration is recomputed "
            "and rewritten"
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=_positive_int_arg,
        default=512,
        help="square output processing tile size (default: 512)",
    )
    parser.add_argument(
        "--query-batch-size",
        type=_positive_int_arg,
        default=65_536,
        help="target TIFXYZ vertices matched per batch (default: 65536)",
    )
    parser.add_argument(
        "--preflight-sample-points",
        type=_positive_int_arg,
        default=4096,
        help=(
            "target TIFXYZ vertices checked before allocating full-resolution "
            "outputs (default: 4096)"
        ),
    )
    parser.add_argument(
        "--minimum-mapping-coverage",
        type=_coverage_arg,
        default=0.01,
        help=(
            "abort when sampled or final valid-surface coverage is below this "
            "fraction (default: 0.01; use 0 only for intentional partial overlap)"
        ),
    )
    parser.add_argument(
        "--fill-value",
        type=int,
        default=0,
        help="label value for unmapped pixels (default: 0)",
    )
    parser.add_argument(
        "--ignore-tifxyz-mask",
        action="store_true",
        help="ignore optional mask.tif files and use XYZ validity only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and report inferred geometry without writing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = HyphenUnderscoreParser(
        description=(
            "Transfer categorical labels between TIFXYZ surfaces using their "
            "3D geometry; render logs are not required for full canvases."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="run one surface-to-surface transfer")
    single.add_argument("--source-tifxyz", required=True)
    single.add_argument("--target-tifxyz", required=True)
    single.add_argument("--label", required=True)
    single.add_argument(
        "--additional-label",
        dest="additional_labels",
        action="append",
        nargs=2,
        default=[],
        metavar=("SOURCE", "OUTPUT"),
        help=(
            "transfer another same-shape categorical raster in the same "
            "geometry pass; may be repeated"
        ),
    )
    single.add_argument(
        "--source-validity",
        help=(
            "optional source-label validity raster; invalid samples remain "
            "invalid in the output"
        ),
    )
    single.add_argument(
        "--label-canvas-offset",
        nargs=2,
        type=float,
        metavar=("DY", "DX"),
        help=(
            "constant offset between the label raster and the source TIFXYZ "
            "canvas in full-resolution canvas pixels: label pixel (i, j) "
            "depicts canvas position (i + DY, j + DX); measure it with "
            "estimate_canvas_offset.py"
        ),
    )
    single.add_argument("--output", required=True)
    single.add_argument("--affine", help="optional registration JSON")
    single.add_argument(
        "--affine-direction",
        choices=("auto", "forward", "inverse"),
        default="auto",
        help="forward means target=fixed and source=moving (default: auto)",
    )
    single.add_argument(
        "--affine-sample-points",
        type=_positive_int_arg,
        default=100_000,
        help="surface points used for affine auto-direction (default: 100000)",
    )
    single.add_argument(
        "--max-distance",
        type=_distance_arg,
        default=None,
        metavar="VOXELS|auto",
        help="reject matches farther than this in target voxels (default: auto)",
    )
    single.add_argument(
        "--output-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="override automatically inferred target label dimensions",
    )
    single.add_argument(
        "--target-reference",
        help="image whose dimensions define the exact output shape",
    )
    single.add_argument("--valid-output", help="mapping validity TIFF path")
    single.add_argument("--distance-output", help="optional float distance TIFF path")
    single.add_argument("--report-output", help="JSON report path")
    single.add_argument(
        "--planar",
        action="store_true",
        help=(
            "fit one global 2D affine between the two parameterizations "
            "from sampled correspondences and warp the whole label at once "
            "instead of mapping per pixel; no fold-seam holes, but any "
            "non-affine drift is ignored (see residual_label_px in the "
            "report)"
        ),
    )
    single.add_argument(
        "--planar-sample-vertices",
        type=_positive_int_arg,
        default=200_000,
        help=(
            "target vertices sampled to fit the planar affine "
            "(default: 200000)"
        ),
    )
    _add_common_transfer_arguments(single)

    pipeline = subparsers.add_parser(
        "pipeline", help="old 2.4 -> updated 2.4 -> target volume"
    )
    pipeline.add_argument("--old-tifxyz", required=True)
    pipeline.add_argument("--updated-tifxyz", required=True)
    pipeline.add_argument("--target-tifxyz", required=True)
    pipeline.add_argument("--label", required=True)
    pipeline.add_argument(
        "--additional-label",
        dest="additional_labels",
        action="append",
        nargs=3,
        default=[],
        metavar=("SOURCE", "INTERMEDIATE", "FINAL"),
        help=(
            "transfer another same-shape categorical raster through both "
            "stages in the shared geometry passes; may be repeated"
        ),
    )
    pipeline.add_argument("--affine", required=True)
    pipeline.add_argument("--output", required=True)
    pipeline.add_argument(
        "--intermediate-output",
        help="updated-2.4 label TIFF (default: beside final output)",
    )
    pipeline.add_argument(
        "--updated-reference",
        help="optional updated-2.4 render whose shape defines the intermediate",
    )
    pipeline.add_argument(
        "--target-reference",
        help="optional target render whose shape defines the final output",
    )
    pipeline.add_argument(
        "--intermediate-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
    )
    pipeline.add_argument(
        "--output-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
    )
    pipeline.add_argument(
        "--affine-direction",
        choices=("auto", "forward", "inverse"),
        default="auto",
    )
    pipeline.add_argument(
        "--affine-sample-points",
        type=_positive_int_arg,
        default=100_000,
    )
    pipeline.add_argument(
        "--label-canvas-offset",
        nargs=2,
        type=float,
        metavar=("DY", "DX"),
        help=(
            "constant offset between the old label raster and the old "
            "TIFXYZ canvas in full-resolution canvas pixels, applied in "
            "stage one; measure it with estimate_canvas_offset.py"
        ),
    )
    pipeline.add_argument(
        "--stage-one-affine",
        "--same-volume-affine",
        dest="same_volume_affine",
        help=(
            "optional registration JSON mapping source-volume XYZ into "
            "updated-volume XYZ for stage one; "
            "default assumes both TIFXYZs share one volume frame"
        ),
    )
    pipeline.add_argument(
        "--stage-one-affine-direction",
        "--same-volume-affine-direction",
        dest="same_volume_affine_direction",
        choices=("auto", "forward", "inverse"),
        default="forward",
        help=(
            "direction for --stage-one-affine (default: forward; 'auto' "
            "cannot distinguish small tangential translations because "
            "point-to-surface distances are blind to them)"
        ),
    )
    pipeline.add_argument(
        "--same-volume-max-distance",
        type=_distance_arg,
        default=None,
        metavar="VOXELS|auto",
    )
    pipeline.add_argument(
        "--cross-volume-max-distance",
        type=_distance_arg,
        default=None,
        metavar="VOXELS|auto",
    )
    _add_common_transfer_arguments(pipeline)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "single":
            run_single(args)
        else:
            run_pipeline(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
