#!/usr/bin/env python3
"""Bounded structural evidence for registered VC surface artifacts.

The grid-coherence and comparison concepts are adapted from Diego's
MIT-licensed vesuvius-topological-grid project, pinned for design reference at
commit 1b117880f17d4e639376dae26b80d3f7b755a62e. This is a local reimplementation
for typed MCP artifacts; it has no runtime dependency on that repository.

All outputs measure periodic structure or agreement. They do not establish ink,
text, transcription, model independence, or correct surface geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

MAX_PIXELS = 4_194_304
MAX_WINDOWS = 4096
MAX_NULL_TRIALS = 64
REFERENCE_COMMIT = "1b117880f17d4e639376dae26b80d3f7b755a62e"
REFERENCE_REPOSITORY = "https://github.com/Diego-dcv/vesuvius-topological-grid"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_png(path: Path, values: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(values).save(path)


def preview(values: np.ndarray, *, diverging: bool = False) -> np.ndarray:
    values = np.asarray(values, np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros((*values.shape, 3), np.uint8) if diverging else np.zeros(values.shape, np.uint8)
    if diverging:
        limit = float(np.percentile(np.abs(values[finite]), 99))
        limit = max(limit, 1e-9)
        scaled = np.clip(values / limit, -1, 1)
        image = np.zeros((*values.shape, 3), np.uint8)
        image[..., 0] = np.where(scaled >= 0, scaled * 255, (1 + scaled) * 52).astype(np.uint8)
        image[..., 2] = np.where(scaled <= 0, -scaled * 255, (1 - scaled) * 52).astype(np.uint8)
        image[..., 1] = ((1 - np.abs(scaled)) * 38).astype(np.uint8)
        return image
    low, high = np.percentile(values[finite], [1, 99])
    if high <= low:
        high = low + 1
    output = np.zeros(values.shape, np.uint8)
    output[finite] = np.clip((values[finite] - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    return output


def save_map(output: Path, name: str, values: np.ndarray, *, diverging: bool = False) -> dict:
    import tifffile

    values = np.asarray(values, np.float32)
    np.save(output / f"{name}.npy", values, allow_pickle=False)
    tifffile.imwrite(output / f"{name}.tif", values)
    write_png(output / f"{name}.png", preview(values, diverging=diverging))
    return {"npy": f"{name}.npy", "tiff": f"{name}.tif", "preview": f"{name}.png"}


def load_registered(artifact: Path, polarity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, float, float]:
    import zarr

    root = zarr.open_group(str(artifact / "surface.zarr"), mode="r")
    image = np.asarray(root["renders/raw"], dtype=np.float32)
    valid = np.asarray(root["geometry/valid"]) != 0
    xyz = np.asarray(root["geometry/xyz"], dtype=np.float32)
    if image.ndim != 2 or image.shape != valid.shape or xyz.shape != (*image.shape, 3):
        raise ValueError("registered surface arrays do not share the required VU grid")
    if image.size > MAX_PIXELS:
        raise ValueError("structural evidence is limited to 4,194,304 registered pixels")
    selected = image[valid & np.isfinite(image)]
    if selected.size < 64:
        raise ValueError("registered surface has fewer than 64 valid signal pixels")
    low, high = np.percentile(selected, [1, 99])
    if high <= low:
        raise ValueError("registered signal has no usable dynamic range")
    normalized = np.clip((image - low) / (high - low), 0, 1)
    if polarity == "dark":
        normalized = 1 - normalized
    elif polarity != "bright":
        raise ValueError("polarity must be bright or dark")
    normalized[~valid] = float(np.median(normalized[valid]))

    attrs = dict(root.attrs)
    registered = attrs.get("registered_volume", {})
    if registered.get("voxel_spacing_unit", "um") != "um":
        raise ValueError("structural evidence requires voxel_spacing_unit=um")
    if not registered.get("voxel_spacing_explicit", False):
        raise ValueError("structural evidence requires explicit physical voxel_spacing metadata")
    spacing = np.asarray(registered.get("voxel_spacing", [1.0, 1.0, 1.0]), dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("registered surface has invalid XYZ voxel spacing")
    du = (xyz[:, 1:] - xyz[:, :-1]) * spacing
    dv = (xyz[1:] - xyz[:-1]) * spacing
    valid_u = valid[:, 1:] & valid[:, :-1]
    valid_v = valid[1:] & valid[:-1]
    lengths_u = np.linalg.norm(du, axis=-1)[valid_u]
    lengths_v = np.linalg.norm(dv, axis=-1)[valid_v]
    lengths_u = lengths_u[np.isfinite(lengths_u) & (lengths_u > 0)]
    lengths_v = lengths_v[np.isfinite(lengths_v) & (lengths_v > 0)]
    if not lengths_u.size or not lengths_v.size:
        raise ValueError("cannot derive physical UV spacing from surface geometry")
    # VC volume spacing is expressed in micrometres; structural periods use mm.
    pixel_u_mm = float(np.median(lengths_u) / 1000.0)
    pixel_v_mm = float(np.median(lengths_v) / 1000.0)
    return normalized.astype(np.float32), valid, xyz, attrs, pixel_u_mm, pixel_v_mm


def dominant_period(signal: np.ndarray, pixel_mm: float, minimum: float, maximum: float) -> float | None:
    if signal.size < 16:
        return None
    detrended = signal - gaussian_filter1d(signal, sigma=max(2, signal.size / 20))
    spectrum = np.abs(np.fft.rfft(detrended * np.hanning(signal.size)))
    frequency = np.fft.rfftfreq(signal.size, d=pixel_mm)
    with np.errstate(divide="ignore"):
        period = np.where(frequency > 0, 1 / frequency, np.inf)
    selected = (period >= minimum) & (period <= maximum) & np.isfinite(period)
    if not selected.any():
        return None
    indices = np.flatnonzero(selected)
    return float(period[indices[np.argmax(spectrum[selected])]])


def peak_prominence(signal: np.ndarray, pixel_mm: float, target: float | None, tolerance: float = 0.20) -> float:
    if target is None or signal.size < 16:
        return 0.0
    detrended = signal - gaussian_filter1d(signal, sigma=max(2, signal.size / 10))
    spectrum = np.abs(np.fft.rfft(detrended * np.hanning(signal.size)))
    frequency = np.fft.rfftfreq(signal.size, d=pixel_mm)
    with np.errstate(divide="ignore"):
        period = np.where(frequency > 0, 1 / frequency, np.inf)
    peak = (period >= target * (1 - tolerance)) & (period <= target * (1 + tolerance))
    background = (period >= target * (1 - 3 * tolerance)) & (period <= target * (1 + 3 * tolerance)) & ~peak
    if not peak.any() or not background.any():
        return 0.0
    height = float(spectrum[peak].max())
    baseline = float(np.median(spectrum[background])) + 1e-9
    return max(0.0, (height - baseline) / baseline)


def line_gaps(image: np.ndarray, pixel_v_mm: float, minimum: float, maximum: float) -> list[float]:
    profile = image.sum(axis=1)
    if profile.std() < 1e-6:
        return []
    baseline = gaussian_filter1d(profile, sigma=max(3, int(maximum / pixel_v_mm)))
    filtered = gaussian_filter1d(profile - baseline, sigma=max(1, int(0.25 / pixel_v_mm)))
    peaks, _ = find_peaks(
        filtered,
        distance=max(1, int(minimum / pixel_v_mm)),
        prominence=max(1e-9, filtered.std() * 0.4),
    )
    gaps = np.diff(peaks) * pixel_v_mm
    return [float(value) for value in gaps[(gaps >= minimum) & (gaps <= maximum)]]


def calibrate(image: np.ndarray, pixel_u_mm: float, pixel_v_mm: float, request: dict) -> dict:
    letter = request.get("letter_period_mm")
    column = request.get("column_period_mm")
    line = request.get("line_period_mm")
    if letter is None:
        letter = dominant_period(image.sum(axis=0), pixel_u_mm, 2.0, 4.5)
    if column is None:
        column = dominant_period(image.sum(axis=0), pixel_u_mm, 40.0, 80.0)
    gaps = line_gaps(image, pixel_v_mm, 2.0, 8.0)
    if line is None and len(gaps) >= 3:
        line = float(np.median(gaps))
    line_iqr = [float(value) for value in np.percentile(gaps, [25, 75])] if len(gaps) >= 3 else None
    return {
        "letters": float(letter) if letter is not None else None,
        "columns": float(column) if column is not None else None,
        "lines": float(line) if line is not None else None,
        "line_gap_iqr_mm": line_iqr,
        "line_gap_count": len(gaps),
    }


def window_score(window: np.ndarray, pixel_u_mm: float, pixel_v_mm: float, periods: dict, minimum_cycles: float) -> float:
    height_mm = window.shape[0] * pixel_v_mm
    width_mm = window.shape[1] * pixel_u_mm
    components = (
        (periods.get("lines"), window.sum(axis=1), height_mm, pixel_v_mm),
        (periods.get("letters"), window.sum(axis=0), width_mm, pixel_u_mm),
        (periods.get("columns"), window.sum(axis=0), width_mm, pixel_u_mm),
    )
    values = []
    for period, profile, extent, spacing in components:
        if period is None or extent / period < minimum_cycles:
            continue
        values.append(peak_prominence(profile - profile.mean(), spacing, period))
    if not values:
        return 0.0
    return float(np.exp(np.mean(np.log(np.asarray(values) + 0.01))))


def coherence_map(image: np.ndarray, pixel_u_mm: float, pixel_v_mm: float, periods: dict, request: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    width_mm = float(request.get("window_width_mm", 30.0))
    height_mm = float(request.get("window_height_mm", 8.0))
    step_mm = float(request.get("step_mm", 3.0))
    minimum_cycles = float(request.get("minimum_cycles", 3.0))
    width = min(image.shape[1], max(16, int(round(width_mm / pixel_u_mm))))
    height = min(image.shape[0], max(16, int(round(height_mm / pixel_v_mm))))
    step_u = max(1, int(round(step_mm / pixel_u_mm)))
    step_v = max(1, int(round(step_mm / pixel_v_mm)))
    xs = np.arange(width // 2, max(width // 2 + 1, image.shape[1] - (width - width // 2) + 1), step_u)
    ys = np.arange(height // 2, max(height // 2 + 1, image.shape[0] - (height - height // 2) + 1), step_v)
    if not len(xs): xs = np.array([image.shape[1] // 2])
    if not len(ys): ys = np.array([image.shape[0] // 2])
    if len(xs) * len(ys) > MAX_WINDOWS:
        raise ValueError(f"coherence sweep would create {len(xs) * len(ys)} windows; increase step_mm")
    scores = np.zeros((len(ys), len(xs)), np.float32)
    for row, center_v in enumerate(ys):
        v0 = max(0, center_v - height // 2); v1 = min(image.shape[0], v0 + height)
        for column, center_u in enumerate(xs):
            u0 = max(0, center_u - width // 2); u1 = min(image.shape[1], u0 + width)
            scores[row, column] = window_score(image[v0:v1, u0:u1], pixel_u_mm, pixel_v_mm, periods, minimum_cycles)
    parameters = {
        "window_width_mm": width_mm,
        "window_height_mm": height_mm,
        "effective_window_pixels_vu": [height, width],
        "step_mm": step_mm,
        "minimum_cycles": minimum_cycles,
    }
    return scores, xs, ys, parameters


def null_significance(image: np.ndarray, pixel_u_mm: float, pixel_v_mm: float, periods: dict, minimum_cycles: float, trials: int, seed: int) -> dict:
    observed = window_score(image, pixel_u_mm, pixel_v_mm, periods, minimum_cycles)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(trials):
        shuffled = image[rng.permutation(image.shape[0])].copy()
        shifts = rng.integers(0, image.shape[1], size=image.shape[0])
        for row, shift in enumerate(shifts):
            shuffled[row] = np.roll(shuffled[row], int(shift))
        null.append(window_score(shuffled, pixel_u_mm, pixel_v_mm, periods, minimum_cycles))
    null_values = np.asarray(null, np.float32)
    return {
        "observed_global_score": observed,
        "null_scores": [float(value) for value in null_values],
        "null_trials": trials,
        "null_seed": seed,
        "empirical_p_value": float((1 + np.count_nonzero(null_values >= observed)) / (trials + 1)),
        "null_kind": "row_permutation_plus_independent_circular_row_shift",
    }


def provenance() -> dict:
    return {
        "adapted_concepts_from": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "reference_license": "MIT",
        "implementation": "local_bounded_reimplementation",
    }


def grid_coherence(request: dict, artifact: Path, output: Path) -> dict:
    import zarr

    polarity = request.get("polarity", "bright")
    image, valid, _, _, pixel_u_mm, pixel_v_mm = load_registered(artifact, polarity)
    periods = calibrate(image, pixel_u_mm, pixel_v_mm, request)
    if all(periods.get(key) is None for key in ("letters", "columns", "lines")):
        raise ValueError("no structural period could be calibrated; provide an expected period")
    scores, xs, ys, parameters = coherence_map(image, pixel_u_mm, pixel_v_mm, periods, request)
    trials = int(request.get("null_trials", 16)); seed = int(request.get("null_seed", 0))
    if trials < 4 or trials > MAX_NULL_TRIALS:
        raise ValueError("null_trials must be from 4 to 64")
    significance = null_significance(image, pixel_u_mm, pixel_v_mm, periods, parameters["minimum_cycles"], trials, seed)
    output.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(output / "grid-coherence.zarr"), mode="w", zarr_format=2)
    group.create_array("score", data=scores, chunks=(min(128, scores.shape[0]), min(128, scores.shape[1])))
    group.create_array("centers_u", data=xs.astype(np.int32), chunks=(min(1024, len(xs)),))
    group.create_array("centers_v", data=ys.astype(np.int32), chunks=(min(1024, len(ys)),))
    maps = {"coherence": save_map(output, "grid-coherence", scores)}
    with (output / "grid-coherence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["u", "v", "u_mm", "v_mm", "score"])
        for row, v in enumerate(ys):
            for column, u in enumerate(xs):
                writer.writerow([int(u), int(v), float(u * pixel_u_mm), float(v * pixel_v_mm), float(scores[row, column])])
    manifest = {
        "kind": "vc_structural_grid_coherence_v1",
        "score_semantics": "structural_periodicity_not_ink_probability_or_text_truth",
        "source_artifact": request["surface"],
        "surface_shape_vu": list(image.shape),
        "valid_pixels": int(valid.sum()),
        "polarity": polarity,
        "pixel_spacing_mm_uv": [pixel_u_mm, pixel_v_mm],
        "periods_mm": periods,
        "sweep": parameters,
        "score_summary": {"minimum": float(scores.min()), "median": float(np.median(scores)), "maximum": float(scores.max())},
        "significance": significance,
        "grid_coherence_zarr": "grid-coherence.zarr",
        "maps": maps,
        "provenance": provenance(),
        "limitations": [
            "periodicity is not specific to writing and may arise from fibers, rendering, tiling, or model artifacts",
            "the empirical null disrupts row organization but does not represent every nuisance process",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def compare_registered(request: dict, artifact_a: Path, artifact_b: Path, output: Path) -> dict:
    import zarr

    polarity = request.get("polarity", "bright")
    a, valid_a, xyz_a, attrs_a, pixel_u_a, pixel_v_a = load_registered(artifact_a, polarity)
    b, valid_b, xyz_b, attrs_b, pixel_u_b, pixel_v_b = load_registered(artifact_b, polarity)
    if a.shape != b.shape or not np.array_equal(valid_a, valid_b):
        raise ValueError("registered surfaces must have identical VU shape and validity")
    coordinate_difference = float(np.max(np.abs(xyz_a[valid_a] - xyz_b[valid_b]))) if valid_a.any() else math.inf
    if coordinate_difference > 1e-4:
        raise ValueError("registered surfaces do not share the same UV-to-XYZ geometry")
    if abs(pixel_u_a - pixel_u_b) > 1e-9 or abs(pixel_v_a - pixel_v_b) > 1e-9:
        raise ValueError("registered surfaces have different physical UV spacing")
    periods_a = calibrate(a, pixel_u_a, pixel_v_a, request)
    periods_b = calibrate(b, pixel_u_b, pixel_v_b, request)
    common = {}
    for key in ("letters", "columns", "lines"):
        values = [value for value in (periods_a.get(key), periods_b.get(key)) if value is not None]
        common[key] = float(np.mean(values)) if values else None
    score_a, xs_a, ys_a, parameters = coherence_map(a, pixel_u_a, pixel_v_a, common, request)
    score_b, xs_b, ys_b, _ = coherence_map(b, pixel_u_b, pixel_v_b, common, request)
    if not np.array_equal(xs_a, xs_b) or not np.array_equal(ys_a, ys_b):
        raise ValueError("coherence sweeps did not produce the same registered grid")
    scale_a = max(float(np.percentile(score_a, 95)), 1e-9); scale_b = max(float(np.percentile(score_b, 95)), 1e-9)
    agreement = np.clip(np.minimum(score_a / scale_a, score_b / scale_b), 0, 1)
    divergence = score_a - score_b
    priority = np.abs(divergence) * np.maximum(score_a / scale_a, score_b / scale_b)
    pixel_difference = np.abs(a - b).astype(np.float32); pixel_difference[~valid_a] = 0
    if score_a.size > 1 and score_a.std() > 0 and score_b.std() > 0:
        correlation = float(np.corrcoef(score_a.ravel(), score_b.ravel())[0, 1])
    else:
        correlation = 1.0 if np.array_equal(score_a, score_b) else 0.0
    output.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(output / "structural-comparison.zarr"), mode="w", zarr_format=2)
    chunks = (min(128, score_a.shape[0]), min(128, score_a.shape[1]))
    for name, values in (("score_a", score_a), ("score_b", score_b), ("agreement", agreement), ("divergence", divergence), ("review_priority", priority)):
        group.create_array(name, data=values.astype(np.float32), chunks=chunks)
    group.create_array("centers_u", data=xs_a.astype(np.int32), chunks=(min(1024, len(xs_a)),))
    group.create_array("centers_v", data=ys_a.astype(np.int32), chunks=(min(1024, len(ys_a)),))
    group.create_array("pixel_difference", data=pixel_difference, chunks=(min(256, a.shape[0]), min(256, a.shape[1])))
    maps = {
        "agreement": save_map(output, "structural-agreement", agreement),
        "divergence": save_map(output, "structural-divergence", divergence, diverging=True),
        "review_priority": save_map(output, "structural-review-priority", priority),
        "pixel_difference": save_map(output, "registered-pixel-difference", pixel_difference),
    }
    manifest = {
        "kind": "vc_registered_structural_comparison_v1",
        "score_semantics": "agreement_and_divergence_not_truth_or_model_independence",
        "source_artifact_a": request["surface_a"],
        "source_artifact_b": request["surface_b"],
        "surface_shape_vu": list(a.shape),
        "maximum_xyz_difference": coordinate_difference,
        "pixel_spacing_mm_uv": [pixel_u_a, pixel_v_a],
        "periods_a_mm": periods_a,
        "periods_b_mm": periods_b,
        "common_periods_mm": common,
        "sweep": parameters,
        "map_correlation": correlation,
        "a_greater_fraction": float(np.mean(divergence > 0)),
        "b_greater_fraction": float(np.mean(divergence < 0)),
        "structural_comparison_zarr": "structural-comparison.zarr",
        "maps": maps,
        "provenance": provenance(),
        "limitations": [
            "agreement may reflect correlated errors or shared preprocessing and is not consensus truth",
            "divergence prioritizes review but cannot identify which input is correct",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fold_profile(signal: np.ndarray, period_pixels: float, bins: int) -> np.ndarray:
    phase = np.mod(np.arange(signal.size, dtype=np.float64), period_pixels) / period_pixels
    indices = np.minimum((phase * bins).astype(np.int64), bins - 1)
    sums = np.bincount(indices, weights=signal, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def fold_statistic(profile: np.ndarray) -> float:
    return float((profile.max() - profile.min()) / (np.std(np.diff(profile)) + 1e-9))


def period_fold_search(signal: np.ndarray, center_period_pixels: float, tolerance: float, steps: int, bins: int) -> tuple[float, np.ndarray, np.ndarray]:
    periods = np.linspace(center_period_pixels * (1 - tolerance), center_period_pixels * (1 + tolerance), steps)
    statistics = np.zeros(steps, np.float32)
    for index, period in enumerate(periods):
        statistics[index] = fold_statistic(fold_profile(signal, float(period), bins))
    best = int(np.argmax(statistics))
    return float(periods[best]), statistics, periods.astype(np.float32)


def profile_image(profile: np.ndarray, width: int = 720, height: int = 260) -> np.ndarray:
    from PIL import Image, ImageDraw

    values = np.asarray(profile, np.float64)
    low, high = float(values.min()), float(values.max())
    normalized = (values - low) / max(high - low, 1e-9)
    points = [(int(index * (width - 1) / max(1, len(values) - 1)), int((1 - value) * (height - 25)) + 10) for index, value in enumerate(normalized)]
    image = Image.new("RGB", (width, height), (23, 21, 18)); draw = ImageDraw.Draw(image)
    for y in range(10, height - 10, 40): draw.line((0, y, width, y), fill=(57, 51, 41))
    if len(points) > 1: draw.line(points, fill=(102, 217, 208), width=3)
    return np.asarray(image)


def epoch_fold(request: dict, surface_artifact: Path, grid_artifact: Path, output: Path) -> dict:
    polarity = request.get("polarity", "bright")
    image, valid, _, _, _, pixel_v_mm = load_registered(surface_artifact, polarity)
    grid_manifest = read_json(grid_artifact / "manifest.json")
    line_period_mm = grid_manifest.get("periods_mm", {}).get("lines")
    if line_period_mm is None:
        raise ValueError("grid artifact has no line period; rerun grid coherence with line_period_mm")
    profile = image.mean(axis=1)
    center_pixels = float(line_period_mm) / pixel_v_mm
    if center_pixels < 2 or center_pixels > image.shape[0] / 2:
        raise ValueError("line period is not measurable within the registered image height")
    tolerance = float(request.get("period_tolerance", 0.10)); steps = int(request.get("period_steps", 41)); bins = int(request.get("phase_bins", 64))
    trials = int(request.get("null_trials", 16)); seed = int(request.get("null_seed", 0))
    if tolerance <= 0 or tolerance > 0.25 or steps < 9 or steps > 101 or bins < 16 or bins > 256 or trials < 4 or trials > MAX_NULL_TRIALS:
        raise ValueError("invalid bounded epoch-fold search parameters")
    best_pixels, statistics, searched = period_fold_search(profile, center_pixels, tolerance, steps, bins)
    folded = fold_profile(profile, best_pixels, bins); observed = float(statistics.max())
    rng = np.random.default_rng(seed); null_best = []
    for _ in range(trials):
        shuffled = profile[rng.permutation(profile.size)]
        _, null_statistics, _ = period_fold_search(shuffled, center_pixels, tolerance, steps, bins)
        null_best.append(float(null_statistics.max()))
    p_value = float((1 + sum(value >= observed for value in null_best)) / (trials + 1))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "folded-phase-profile.npy", folded, allow_pickle=False)
    np.save(output / "period-search-statistic.npy", statistics, allow_pickle=False)
    write_png(output / "folded-phase-profile.png", profile_image(folded))
    search_map = np.tile(statistics[None, :], (32, 1)); write_png(output / "period-search.png", preview(search_map))
    with (output / "period-search.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["period_pixels", "period_mm", "statistic"])
        for period, statistic in zip(searched, statistics): writer.writerow([float(period), float(period * pixel_v_mm), float(statistic)])
    manifest = {
        "kind": "vc_epoch_fold_structure_v1",
        "score_semantics": "periodic_line_structure_detection_not_text_or_transcription",
        "source_surface_artifact": request["surface"],
        "source_grid_artifact": request["grid"],
        "polarity": polarity,
        "input_line_period_mm": float(line_period_mm),
        "best_period_mm": float(best_pixels * pixel_v_mm),
        "best_period_pixels": best_pixels,
        "period_tolerance": tolerance,
        "period_steps": steps,
        "phase_bins": bins,
        "fold_statistic": observed,
        "look_elsewhere_corrected_empirical_p_value": p_value,
        "null_best_statistics": null_best,
        "null_trials": trials,
        "null_seed": seed,
        "null_kind": "row_projection_permutation_with_full_period_search",
        "folded_profile": "folded-phase-profile.npy",
        "folded_profile_preview": "folded-phase-profile.png",
        "period_search_preview": "period-search.png",
        "provenance": provenance(),
        "limitations": [
            "folding averages the horizontal signal into a one-dimensional phase profile and cannot transcribe text",
            "periodic fibers, acquisition artifacts, or render artifacts can also produce significant folding",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("grid", "compare", "fold"):
        command = subparsers.add_parser(name)
        command.add_argument("--request", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "compare":
            command.add_argument("--artifact-a", required=True, type=Path)
            command.add_argument("--artifact-b", required=True, type=Path)
        else:
            command.add_argument("--artifact", required=True, type=Path)
            if name == "fold": command.add_argument("--grid-artifact", required=True, type=Path)
    arguments = parser.parse_args(); request = read_json(arguments.request); arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.command == "grid": result = grid_coherence(request, arguments.artifact, arguments.output)
    elif arguments.command == "compare": result = compare_registered(request, arguments.artifact_a, arguments.artifact_b, arguments.output)
    else: result = epoch_fold(request, arguments.artifact, arguments.grid_artifact, arguments.output)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
