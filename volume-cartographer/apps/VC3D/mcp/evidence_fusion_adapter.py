#!/usr/bin/env python3
"""Bounded perturbation stability and transparent evidence fusion.

Scores produced here are deterministic ranking heuristics, not calibrated
probabilities and not assertions of writing or surface correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

MAX_PIXELS = 1_048_576
MAX_VARIANTS = 7
MAX_CANDIDATES = 16


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def normalize_signal(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = values[valid & np.isfinite(values)]
    if selected.size < 16:
        raise ValueError("registered signal has insufficient valid dynamic range")
    low, high = np.percentile(selected, [1, 99])
    if high <= low:
        return np.zeros(values.shape, np.float32)
    result = np.clip((values.astype(np.float32) - low) / (high - low), 0, 1)
    result[~valid] = 0
    return result


def normals(xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] < 2 or xyz.shape[1] < 2:
        raise ValueError("stability requires at least a 2x2 UV grid")
    du = np.zeros_like(xyz, np.float32); dv = np.zeros_like(xyz, np.float32)
    du[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) * 0.5
    du[:, 0] = xyz[:, 1] - xyz[:, 0]; du[:, -1] = xyz[:, -1] - xyz[:, -2]
    dv[1:-1] = (xyz[2:] - xyz[:-2]) * 0.5
    dv[0] = xyz[1] - xyz[0]; dv[-1] = xyz[-1] - xyz[-2]
    support = valid.copy()
    support[:, 1:] &= valid[:, :-1]; support[:, :-1] &= valid[:, 1:]
    support[1:] &= valid[:-1]; support[:-1] &= valid[1:]
    result = np.cross(du, dv); lengths = np.linalg.norm(result, axis=-1)
    support &= np.isfinite(lengths) & (lengths > 1e-8)
    result[~support] = 0; result[support] /= lengths[support, None]
    return result, support


def load_registered(path: Path) -> dict:
    import zarr

    root = zarr.open_group(str(path / "surface.zarr"), mode="r")
    xyz = np.asarray(root["geometry/xyz"], np.float32)
    valid = np.asarray(root["geometry/valid"]) != 0
    signal = np.asarray(root["renders/raw"], np.float32)
    if xyz.shape != (*valid.shape, 3) or signal.shape != valid.shape or valid.size > MAX_PIXELS:
        raise ValueError("registered surface exceeds stability layout or pixel limits")
    registered = dict(root.attrs.get("registered_volume", {}))
    if registered.get("voxel_spacing_unit", "um") != "um" or not registered.get("voxel_spacing_explicit", False):
        raise ValueError("stability requires explicit voxel spacing in micrometres")
    spacing = np.asarray(registered.get("voxel_spacing"), np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("registered surface has invalid voxel spacing")
    unit_normals, normal_support = normals(xyz, valid)
    return {
        "xyz": xyz,
        "valid": valid,
        "signal": normalize_signal(signal, valid),
        "normals": unit_normals,
        "normal_support": normal_support,
        "spacing_um": spacing,
        "coordinate_space": root.attrs.get("coordinate_space"),
    }


def image_preview(values: np.ndarray, *, bounded: bool = False) -> np.ndarray:
    values = np.asarray(values, np.float32); finite = np.isfinite(values)
    output = np.zeros(values.shape, np.uint8)
    if not finite.any():
        return output
    if bounded:
        output[finite] = np.clip(values[finite], 0, 1) * 255
        return output
    low, high = np.percentile(values[finite], [1, 99])
    if high <= low: high = low + 1
    output[finite] = np.clip((values[finite] - low) * 255 / (high - low), 0, 255)
    return output


def write_png(path: Path, values: np.ndarray, *, bounded: bool = False) -> None:
    from PIL import Image

    Image.fromarray(image_preview(values, bounded=bounded)).save(path)


def save_map(output: Path, name: str, values: np.ndarray, *, bounded: bool = False) -> dict:
    import tifffile

    values = np.asarray(values, np.float32)
    np.save(output / f"{name}.npy", values, allow_pickle=False)
    tifffile.imwrite(output / f"{name}.tif", values)
    write_png(output / f"{name}.png", values, bounded=bounded)
    return {"npy": f"{name}.npy", "tiff": f"{name}.tif", "preview": f"{name}.png"}


def perturbation_stability(request: dict, baseline_path: Path, variant_paths: list[Path], output: Path) -> dict:
    if not 1 <= len(variant_paths) <= MAX_VARIANTS:
        raise ValueError("stability requires from 1 to 7 variants")
    displacement_scale = float(request.get("displacement_scale_mm", 0.05))
    angle_scale = float(request.get("normal_angle_scale_degrees", 10.0))
    signal_scale = float(request.get("signal_scale", 0.10))
    if min(displacement_scale, angle_scale, signal_scale) <= 0:
        raise ValueError("stability scales must be positive")
    baseline = load_registered(baseline_path)
    shape = baseline["valid"].shape
    maximum_displacement = np.zeros(shape, np.float32)
    maximum_angle = np.zeros(shape, np.float32)
    maximum_signal = np.zeros(shape, np.float32)
    minimum_local_stability = np.ones(shape, np.float32)
    comparisons = []
    baseline_valid = baseline["valid"]
    for index, path in enumerate(variant_paths):
        variant = load_registered(path)
        if variant["valid"].shape != shape:
            raise ValueError("stability surfaces must have identical VU dimensions")
        if variant["coordinate_space"] != baseline["coordinate_space"] or not np.allclose(variant["spacing_um"], baseline["spacing_um"], rtol=0, atol=1e-9):
            raise ValueError("stability surfaces must share coordinate space and physical spacing")
        overlap = baseline_valid & variant["valid"]
        normal_overlap = baseline["normal_support"] & variant["normal_support"]
        displacement = np.zeros(shape, np.float32)
        delta_mm = (variant["xyz"] - baseline["xyz"]) * (baseline["spacing_um"] / 1000.0)
        displacement[overlap] = np.linalg.norm(delta_mm[overlap], axis=-1)
        angle = np.zeros(shape, np.float32)
        dots = np.clip(np.sum(baseline["normals"] * variant["normals"], axis=-1), -1, 1)
        angle[normal_overlap] = np.degrees(np.arccos(np.abs(dots[normal_overlap]))).astype(np.float32)
        signal_difference = np.zeros(shape, np.float32)
        signal_difference[overlap] = np.abs(baseline["signal"][overlap] - variant["signal"][overlap])
        valid_union = baseline_valid | variant["valid"]
        valid_iou = float(overlap.sum() / max(1, valid_union.sum()))
        local = np.zeros(shape, np.float32)
        local[overlap] = np.exp(
            -displacement[overlap] / displacement_scale
            -angle[overlap] / angle_scale
            -signal_difference[overlap] / signal_scale
        )
        maximum_displacement = np.maximum(maximum_displacement, displacement)
        maximum_angle = np.maximum(maximum_angle, angle)
        maximum_signal = np.maximum(maximum_signal, signal_difference)
        minimum_local_stability = np.minimum(minimum_local_stability, local)
        comparisons.append(
            {
                "variant_index": index,
                "artifact": request["variants"][index],
                "valid_iou": valid_iou,
                "overlap_pixels": int(overlap.sum()),
                "median_displacement_mm": float(np.median(displacement[overlap])) if overlap.any() else math.inf,
                "p95_displacement_mm": float(np.percentile(displacement[overlap], 95)) if overlap.any() else math.inf,
                "median_normal_angle_degrees": float(np.median(angle[normal_overlap])) if normal_overlap.any() else math.inf,
                "median_signal_difference": float(np.median(signal_difference[overlap])) if overlap.any() else math.inf,
                "median_local_stability": float(np.median(local[overlap])) if overlap.any() else 0.0,
            }
        )
    minimum_local_stability[~baseline_valid] = 0
    overall = float(min(item["valid_iou"] * item["median_local_stability"] for item in comparisons))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "stability-valid.npy", baseline_valid.astype(np.uint8), allow_pickle=False)
    import zarr
    group = zarr.open_group(str(output / "surface-stability.zarr"), mode="w", zarr_format=2)
    chunks = (min(256, shape[0]), min(256, shape[1]))
    for name, values in (("maximum_displacement_mm", maximum_displacement), ("maximum_normal_angle_degrees", maximum_angle),
                         ("maximum_signal_difference", maximum_signal), ("minimum_local_stability", minimum_local_stability)):
        group.create_array(name, data=values, chunks=chunks)
    maps = {
        "displacement": save_map(output, "stability-displacement-mm", maximum_displacement),
        "normal_angle": save_map(output, "stability-normal-angle", maximum_angle),
        "signal_difference": save_map(output, "stability-signal-difference", maximum_signal),
        "local_stability": save_map(output, "stability-local-score", minimum_local_stability, bounded=True),
    }
    manifest = {
        "kind": "vc_surface_perturbation_stability_v1",
        "score_semantics": "deterministic_perturbation_robustness_not_surface_correctness",
        "baseline_artifact": request["baseline"],
        "variant_artifacts": request["variants"],
        "surface_shape_vu": list(shape),
        "scales": {"displacement_mm": displacement_scale, "normal_angle_degrees": angle_scale, "signal_difference": signal_scale},
        "overall_stability_score": overall,
        "comparisons": comparisons,
        "stability_zarr": "surface-stability.zarr",
        "validity_map": "stability-valid.npy",
        "maps": maps,
        "limitations": [
            "stability only measures supplied perturbations and does not prove correct layer selection",
            "normal direction sign is ignored when measuring angular change",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fuse_ink_scores(request: dict, output: Path) -> dict:
    """Fuse registered Villa and DinoVol UV scores without hiding components."""
    import tifffile
    import zarr

    villa_path = Path(request["villa_path"])
    dinovol_path = Path(request["dinovol_path"])
    villa_manifest = read_json(villa_path / "manifest.json")
    dinovol_manifest = read_json(dinovol_path / "manifest.json")
    if villa_manifest.get("kind") != "villa_ink_prediction_v1":
        raise ValueError("villa artifact manifest has the wrong kind")
    if dinovol_manifest.get("kind") != "dinovol_registered_exemplar_v1":
        raise ValueError("DinoVol artifact manifest has the wrong kind")

    villa_raw = np.load(villa_path / "ink-probability.npy", allow_pickle=False).astype(np.float32)
    dinovol_raw = np.load(dinovol_path / "exemplar-similarity-surface.npy", allow_pickle=False).astype(np.float32)
    if villa_raw.ndim != 2 or villa_raw.shape != dinovol_raw.shape or villa_raw.size > MAX_PIXELS:
        raise ValueError("Villa and DinoVol UV maps must have one matching bounded 2D shape")
    shape = villa_raw.shape
    if villa_manifest.get("output_shape_hw") != list(shape) or dinovol_manifest.get("surface_shape_vu") != list(shape):
        raise ValueError("component manifest dimensions do not match the UV score maps")

    valid = np.isfinite(villa_raw) & np.isfinite(dinovol_raw)
    villa_support_path = villa_path / "ink-valid.npy"
    dinovol_support_path = dinovol_path / "surface-support.npy"
    if villa_support_path.is_file():
        villa_support = np.load(villa_support_path, allow_pickle=False)
        if villa_support.shape != shape:
            raise ValueError("Villa support map does not match the UV score maps")
        valid &= villa_support != 0
    if dinovol_support_path.is_file():
        dinovol_support = np.load(dinovol_support_path, allow_pickle=False)
        if dinovol_support.shape != shape:
            raise ValueError("DinoVol support map does not match the UV score maps")
        valid &= dinovol_support != 0

    villa = np.clip(villa_raw, 0, 1)
    selected = dinovol_raw[valid]
    if selected.size < 16:
        raise ValueError("DinoVol map has insufficient registered support")
    low, high = np.percentile(selected, (1, 99))
    dinovol = np.zeros(shape, np.float32)
    if high > low:
        dinovol[valid] = np.clip((dinovol_raw[valid] - low) / (high - low), 0, 1)

    stability_artifact = request.get("stability")
    stability_raw = None
    stability = None
    stability_manifest = None
    if stability_artifact is not None:
        stability_path = Path(request["stability_path"])
        stability_manifest = read_json(stability_path / "manifest.json")
        if stability_manifest.get("kind") != "vc_surface_perturbation_stability_v1":
            raise ValueError("stability artifact manifest has the wrong kind")
        stability_raw = np.load(stability_path / "stability-local-score.npy", allow_pickle=False).astype(np.float32)
        if stability_raw.shape != shape or stability_manifest.get("surface_shape_vu") != list(shape):
            raise ValueError("stability map does not match Villa/DinoVol UV shape")
        stability_support_path = stability_path / "stability-valid.npy"
        if stability_support_path.is_file():
            stability_support = np.load(stability_support_path, allow_pickle=False)
            if stability_support.shape != shape:
                raise ValueError("stability support map does not match the UV score maps")
            valid &= stability_support != 0
        valid &= np.isfinite(stability_raw)
        stability = np.clip(stability_raw, 0, 1)

    source_artifacts = {
        "villa": villa_manifest.get("source_surface_artifact"),
        "dinovol": dinovol_manifest.get("source_surface_artifact"),
        "stability": stability_manifest.get("baseline_artifact") if stability_manifest else None,
    }
    known_sources = [source for source in source_artifacts.values() if source is not None]
    if known_sources and any(source != known_sources[0] for source in known_sources[1:]):
        raise ValueError("fusion artifacts do not refer to the same registered surface")

    supplied_weights = request.get("weights", {})
    if not isinstance(supplied_weights, dict) or any(key not in {"villa", "dinovol", "stability"} for key in supplied_weights):
        raise ValueError("unknown fusion weight")
    weights = {"villa": 1.0, "dinovol": 1.0, "stability": 0.5 if stability is not None else 0.0}
    weights.update({key: float(value) for key, value in supplied_weights.items()})
    if any(not math.isfinite(value) or value < 0 or value > 100 for value in weights.values()):
        raise ValueError("fusion weights must be finite and from 0 to 100")
    if stability is None and weights["stability"] != 0:
        raise ValueError("a non-zero stability weight requires a stability artifact")
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("fusion weights cannot all be zero")
    combined = weights["villa"] * villa + weights["dinovol"] * dinovol
    if stability is not None:
        combined += weights["stability"] * stability
    combined /= denominator
    combined[~valid] = 0

    output.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(output / "ink-fusion.zarr"), mode="w", zarr_format=2)
    chunks = (min(256, shape[0]), min(256, shape[1]))
    map_values = [
        ("villa_probability_raw", villa_raw, False),
        ("villa_probability", villa, True),
        ("dinovol_similarity_raw", dinovol_raw, False),
        ("dinovol_similarity_normalized", dinovol, True),
    ]
    if stability is not None:
        map_values.extend([
            ("stability_local_score_raw", stability_raw, False),
            ("stability_local_score", stability, True),
        ])
    map_values.append(("combined_score", combined, True))
    maps = {}
    for name, values, bounded in map_values:
        group.create_array(name, data=values, chunks=chunks)
        maps[name] = save_map(output, name.replace("_", "-"), values, bounded=bounded)
    valid_u8 = valid.astype(np.uint8)
    group.create_array("valid", data=valid_u8, chunks=chunks)
    np.save(output / "fusion-valid.npy", valid_u8, allow_pickle=False)
    tifffile.imwrite(output / "fusion-valid.tif", valid_u8)
    write_png(output / "fusion-valid.png", valid_u8, bounded=True)

    component_maps = [name for name, _, _ in map_values if name != "combined_score"]
    manifest = {
        "kind": "vc_registered_ink_fusion_v1",
        "score_semantics": "transparent_review_priority_not_calibrated_ink_probability",
        "villa_artifact": request["villa"],
        "dinovol_artifact": request["dinovol"],
        "stability_artifact": stability_artifact,
        "source_surface_artifacts": source_artifacts,
        "source_surface_consistency": (
            "verified" if len(known_sources) >= 2 else
            "partially_available" if known_sources else
            "unavailable_in_source_manifests"
        ),
        "shape_uv": list(shape),
        "valid_pixels": int(valid.sum()),
        "weights": weights,
        "formula": "(w_villa*clip(villa_raw,0,1) + w_dinovol*percentile_normalized_dinovol + w_stability*clip(stability_raw,0,1)) / sum(weights)",
        "villa_normalization": {"method": "clip", "range": [0, 1]},
        "dinovol_normalization": {"method": "linear_percentile_clip", "percentiles": [1, 99], "low": float(low), "high": float(high)},
        "stability_normalization": {"method": "clip", "range": [0, 1]} if stability is not None else None,
        "component_maps_preserved": component_maps,
        "combined_score_summary": {"minimum": float(combined[valid].min()) if valid.any() else 0.0,
                                   "maximum": float(combined[valid].max()) if valid.any() else 0.0,
                                   "mean": float(combined[valid].mean()) if valid.any() else 0.0},
        "fusion_zarr": "ink-fusion.zarr",
        "validity_map": {"npy": "fusion-valid.npy", "tiff": "fusion-valid.tif", "preview": "fusion-valid.png"},
        "maps": maps,
        "limitations": [
            "Villa probabilities are uncalibrated on PHerc0332",
            "DinoVol percentile normalization is ROI-relative and not a probability",
            "stability is a perturbation-robustness measurement, not independent ink evidence",
            "the combined score is a transparent review-priority heuristic, not proof of ink or text",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def bounded_score(value: float) -> float:
    return float(np.clip(value, 0, 1))


def candidate_components(candidate: dict) -> tuple[dict, tuple]:
    geometry = read_json(Path(candidate["resolved"]["geometry"]) / "manifest.json")
    alignment = read_json(Path(candidate["resolved"]["alignment"]) / "manifest.json")
    grid = read_json(Path(candidate["resolved"]["grid"]) / "manifest.json")
    sources = [geometry.get("source_artifact"), alignment.get("source_artifact"), grid.get("source_artifact")]
    if any(source is None for source in sources) or not all(source == sources[0] for source in sources[1:]):
        raise ValueError(f"candidate {candidate['id']} evidence does not refer to one registered surface")
    shapes = [tuple(value.get("surface_shape_vu", [])) for value in (geometry, alignment, grid)]
    if not shapes[0] or not all(shape == shapes[0] for shape in shapes[1:]):
        raise ValueError(f"candidate {candidate['id']} evidence has inconsistent VU dimensions")
    valid = max(1, int(geometry.get("valid_pixels", 1)))
    fold_fraction = float(geometry.get("fold_or_degenerate_cells", 0)) / valid
    geometry_score = math.exp(
        -float(geometry.get("p95_stretch_log_ratio", 0)) / 0.25
        -float(geometry.get("p95_normal_change_degrees", 0)) / 30.0
        -fold_fraction * 100.0
        -max(0, int(geometry.get("connected_components", 1)) - 1) * 0.1
        -int(geometry.get("enclosed_holes", 0)) * 0.1
    )
    alignment_score = (
        float(alignment.get("support_fraction", 0))
        * float(alignment.get("median_confidence", 0))
        * math.exp(-abs(float(alignment.get("median_peak_offset_voxels", 0))) / 2.0)
    )
    median_grid = max(0.0, float(grid.get("score_summary", {}).get("median", 0)))
    p_value = bounded_score(float(grid.get("significance", {}).get("empirical_p_value", 1)))
    grid_score = (1 - math.exp(-median_grid)) * (0.5 + 0.5 * (1 - p_value))
    components = {
        "geometry": bounded_score(geometry_score),
        "alignment": bounded_score(alignment_score),
        "grid": bounded_score(grid_score),
    }
    if "stability" in candidate["resolved"]:
        stability = read_json(Path(candidate["resolved"]["stability"]) / "manifest.json")
        if stability.get("baseline_artifact") != sources[0]:
            raise ValueError(f"candidate {candidate['id']} stability baseline does not match its registered surface")
        components["stability"] = bounded_score(float(stability.get("overall_stability_score", 0)))
    signature = (shapes[0], tuple(round(float(value), 12) for value in grid.get("pixel_spacing_mm_uv", [])))
    return components, signature


def ranking_image(ranked: list[dict], width: int = 780) -> np.ndarray:
    from PIL import Image, ImageDraw

    row_height = 44; height = 32 + row_height * len(ranked)
    image = Image.new("RGB", (width, height), (23, 21, 18)); draw = ImageDraw.Draw(image)
    for row, candidate in enumerate(ranked):
        y = 20 + row * row_height; score = candidate["combined_score"]
        draw.text((12, y + 7), f"{row + 1:02d}  {candidate['id']}", fill=(240, 231, 210))
        draw.rectangle((220, y + 5, width - 20, y + 29), outline=(57, 51, 41))
        draw.rectangle((221, y + 6, 221 + int((width - 242) * score), y + 28), fill=(242, 169, 59))
        draw.text((width - 66, y + 8), f"{score:.3f}", fill=(23, 21, 18) if score > 0.12 else (240, 231, 210))
    return np.asarray(image)


def rank_evidence(request: dict, output: Path) -> dict:
    candidates = request["resolved_candidates"]
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("evidence ranking requires from 1 to 16 candidates")
    weights = {"geometry": 1.0, "alignment": 1.0, "grid": 1.0, "stability": 1.0}
    weights.update({key: float(value) for key, value in request.get("weights", {}).items()})
    if any(not math.isfinite(value) or value < 0 or value > 100 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("evidence weights must be finite, non-negative, and not all zero")
    ranked = []
    comparison_signature = None
    for candidate in candidates:
        components, signature = candidate_components(candidate)
        if comparison_signature is None:
            comparison_signature = signature
        elif signature != comparison_signature:
            raise ValueError("ranked candidates must share VU dimensions and physical pixel spacing")
        active = [(key, score) for key, score in components.items() if weights.get(key, 0) > 0]
        if not active:
            combined = 0.0
        else:
            denominator = sum(weights[key] for key, _ in active)
            combined = math.exp(sum(weights[key] * math.log(score + 0.01) for key, score in active) / denominator) - 0.01
        ranked.append({"id": candidate["id"], "combined_score": bounded_score(combined), "components": components})
    ranked.sort(key=lambda item: (-item["combined_score"], item["id"]))
    for index, candidate in enumerate(ranked, 1): candidate["rank"] = index
    output.mkdir(parents=True, exist_ok=True)
    write_png(output / "evidence-ranking.png", ranking_image(ranked))
    with (output / "evidence-ranking.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["rank", "id", "combined_score", "geometry", "alignment", "grid", "stability"])
        for item in ranked:
            writer.writerow([item["rank"], item["id"], item["combined_score"], item["components"].get("geometry"),
                             item["components"].get("alignment"), item["components"].get("grid"), item["components"].get("stability")])
    manifest = {
        "kind": "vc_transparent_evidence_ranking_v1",
        "score_semantics": "review_priority_heuristic_not_probability_or_truth",
        "formula": "weighted geometric mean of bounded component scores with +0.01 numerical floor",
        "weights": weights,
        "component_formulas": {
            "geometry": "exp(-p95_stretch/0.25 - p95_normal_change/30 - fold_fraction*100 - 0.1*extra_components - 0.1*holes)",
            "alignment": "support_fraction * median_confidence * exp(-abs(median_peak_offset)/2)",
            "grid": "(1-exp(-median_grid_score)) * (0.5 + 0.5*(1-empirical_p))",
            "stability": "overall_stability_score from supplied perturbation artifact",
        },
        "ranked_candidates": ranked,
        "ranking_preview": "evidence-ranking.png",
        "ranking_csv": "evidence-ranking.csv",
        "limitations": [
            "weights and transforms are explicit heuristics and are not calibrated against papyrological truth",
            "candidate ranking is only comparable when inputs cover the same registered physical region",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    stability = commands.add_parser("stability"); stability.add_argument("--request", required=True, type=Path); stability.add_argument("--baseline", required=True, type=Path); stability.add_argument("--variants", required=True, type=Path); stability.add_argument("--output", required=True, type=Path)
    ranking = commands.add_parser("rank"); ranking.add_argument("--request", required=True, type=Path); ranking.add_argument("--output", required=True, type=Path)
    fusion = commands.add_parser("ink-fuse"); fusion.add_argument("--request", required=True, type=Path); fusion.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); request = read_json(args.request); args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "stability":
        variants = read_json(args.variants).get("paths", [])
        result = perturbation_stability(request, args.baseline, [Path(path) for path in variants], args.output)
    elif args.command == "ink-fuse":
        result = fuse_ink_scores(request, args.output)
    else:
        result = rank_evidence(request, args.output)
    print(json.dumps(result), flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
