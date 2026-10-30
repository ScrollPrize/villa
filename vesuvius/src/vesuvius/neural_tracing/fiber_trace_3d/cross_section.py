"""Evaluation-only fiber-presence cross-section accumulation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .loader import FiberTrace3DBatch
from .model import presence_outputs


@dataclass(frozen=True)
class CrossSectionConfig:
    """Physical sampling and bounded histogram settings.

    The image has normal offsets on rows and in-sheet transverse offsets on
    columns. Inclusive endpoints retain the exact fiber center.
    """

    width_base: int = 160
    height_base: int = 80
    spacing_base: float = 2.0
    histogram_bins: int = 1024
    quantiles: tuple[float, ...] = (0.001, 0.01, 0.10, 0.20, 0.50)

    def __post_init__(self) -> None:
        if self.width_base <= 0 or self.height_base <= 0:
            raise ValueError("cross-section width and height must be positive")
        if self.width_base % 2 or self.height_base % 2:
            raise ValueError("cross-section width and height must be even")
        if not np.isfinite(self.spacing_base) or self.spacing_base <= 0:
            raise ValueError("cross-section base spacing must be positive and finite")
        if self.histogram_bins < 2:
            raise ValueError("cross-section histogram needs at least two bins")
        if not self.quantiles or any(not 0.0 <= value <= 1.0 for value in self.quantiles):
            raise ValueError("cross-section quantiles must be nonempty and in [0,1]")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height_base + 1, self.width_base + 1

    @property
    def normal_offsets_base(self) -> np.ndarray:
        return np.arange(-self.height_base // 2, self.height_base // 2 + 1, dtype=np.float32)

    @property
    def width_offsets_base(self) -> np.ndarray:
        return np.arange(-self.width_base // 2, self.width_base // 2 + 1, dtype=np.float32)


def _closest_segment(
    point: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    vectors = ends - starts
    length2 = torch.sum(vectors * vectors, dim=1).clamp_min(1.0e-12)
    fraction = (torch.sum((point[None, :] - starts) * vectors, dim=1) / length2).clamp(0.0, 1.0)
    projections = starts + fraction[:, None] * vectors
    index = int(torch.argmin(torch.sum((projections - point[None, :]) ** 2, dim=1)))
    return projections[index], vectors[index], index


def segment_reference_width(batch, sample, center, segment):
    """Target full width at a selected segment projection, in input voxels."""
    if batch.target_segment_widths_input is None:
        raise ValueError('width-filtered visualization requires target segment widths')
    index = int(batch.target_segment_offsets[sample]) + segment
    start = batch.target_segment_starts_zyx[index].to(center)
    delta = batch.target_segment_ends_zyx[index].to(center) - start
    fraction = ((center-start)*delta).sum()/delta.square().sum().clamp_min(1e-12)
    w0, w1 = batch.target_segment_widths_input[index].to(center)
    return w0 + fraction.clamp(0, 1)*(w1-w0)


class CrossSectionAccumulator:
    """Prediction observer with deterministic, mergeable CPU accumulators."""

    def __init__(self, config: CrossSectionConfig, *, width_filtered=False):
        self.config = config
        self.sum = np.zeros(config.shape, dtype=np.float64)
        self.count = np.zeros(config.shape, dtype=np.uint32)
        self.histogram = np.zeros((*config.shape, config.histogram_bins), dtype=np.uint32)
        self.sections = 0
        self.skipped = 0
        self.width_filtered = CrossSectionAccumulator(config) if width_filtered else None
        self.filter_metadata = None

    def observe_annotations(self, batch, output, annotation_output, annotation_config):
        self(batch, output, annotation_output=annotation_output, annotation_config=annotation_config)

    def state_dict(self):
        """Array views; the shared handoff capture owns the immutable copy."""
        return dict(config=asdict(self.config),
                    arrays={name: getattr(self, name)
                            for name in ('sum', 'count', 'histogram')},
                    sections=self.sections, skipped=self.skipped,
                    filter_metadata=self.filter_metadata,
                    width_filtered=(self.width_filtered.state_dict()
                                    if self.width_filtered is not None else None))

    @classmethod
    def from_state_dict(cls, state):
        result = cls(CrossSectionConfig(**state['config']))
        for name, array in state['arrays'].items():
            getattr(result, name)[...] = array
        result.sections, result.skipped = state['sections'], state['skipped']
        result.filter_metadata = state['filter_metadata']
        if state['width_filtered'] is not None:
            result.width_filtered = cls.from_state_dict(state['width_filtered'])
        return result

    def clear(self):
        """Reuse bounded wave storage after merging its observations."""
        for name in ('sum', 'count', 'histogram'):
            getattr(self, name).fill(0)
        self.sections = self.skipped = 0
        if self.width_filtered is not None:
            self.width_filtered.clear()

    @torch.no_grad()
    def __call__(self, batch: FiberTrace3DBatch, output: torch.Tensor, *,
                 annotation_output=None, annotation_config=None) -> None:
        if batch.target_segment_normals_zyx is None:
            raise ValueError("cross-section evaluation requires Lasagna segment normals")
        if int(output.shape[0]) != int(batch.volume.shape[0]):
            raise ValueError("cross-section batch/output size mismatch")
        input_shape = tuple(int(value) for value in batch.volume.shape[-3:])
        output_shape = tuple(int(value) for value in output.shape[-3:])
        strides = tuple(i / o for i, o in zip(input_shape, output_shape, strict=True))
        if any(abs(value - round(value)) > 1.0e-6 for value in strides) or len(set(round(v) for v in strides)) != 1:
            raise ValueError(f"cross-section requires an isotropic integer output stride, got {strides}")
        output_stride = int(round(strides[0]))
        if output_stride < 1:
            raise ValueError("cross-section output stride must be positive")

        presences = presence_outputs(output).amax(dim=1)  # B,1,Z,Y,X
        predicted_widths = None
        if self.width_filtered is not None:
            from .annotations import predicted_width, width_similarity
            if (annotation_output is None or annotation_config is None
                    or 'width' not in annotation_config.names or batch.target_segment_widths_input is None):
                raise ValueError('width-filtered cross-section requires width head and segment targets')
            predicted_widths = predicted_width(annotation_output, annotation_config)
            self.filter_metadata = {
                'width_fraction': annotation_config.sideband_width_fraction,
                'falloff': annotation_config.sideband_width_gate,
                'reference': 'refined_target_segment_width_at_section_center',
                'direction_filter': False,
            }
        normal_offsets = torch.arange(
            -self.config.height_base // 2,
            self.config.height_base // 2 + 1,
            dtype=torch.float32,
            device=output.device,
        )
        width_offsets = torch.arange(
            -self.config.width_base // 2,
            self.config.width_base // 2 + 1,
            dtype=torch.float32,
            device=output.device,
        )
        normal_grid, width_grid = torch.meshgrid(normal_offsets, width_offsets, indexing="ij")
        normal_grid = normal_grid / float(self.config.spacing_base)
        width_grid = width_grid / float(self.config.spacing_base)

        planes: list[torch.Tensor] = []
        valid_planes: list[torch.Tensor] = []
        filtered_planes, filtered_valid = [], []
        for sample in range(int(output.shape[0])):
            offset = int(batch.target_segment_offsets[sample])
            count = int(batch.target_segment_counts[sample])
            if count <= 0:
                self.skipped += 1
                continue
            starts = batch.target_segment_starts_zyx[offset : offset + count].to(
                device=output.device, dtype=torch.float32
            )
            ends = batch.target_segment_ends_zyx[offset : offset + count].to(
                device=output.device, dtype=torch.float32
            )
            normals = batch.target_segment_normals_zyx[offset : offset + count].to(
                device=output.device, dtype=torch.float32
            )
            center, tangent, segment = _closest_segment(
                batch.cp_local_zyx[sample].to(device=output.device, dtype=torch.float32),
                starts,
                ends,
            )
            tangent = tangent / torch.linalg.vector_norm(tangent).clamp_min(1.0e-6)
            normal = normals[segment] - torch.dot(normals[segment], tangent) * tangent
            normal_length = torch.linalg.vector_norm(normal)
            if not bool(torch.isfinite(normal_length)) or float(normal_length) <= 1.0e-6:
                self.skipped += 1
                continue
            normal = normal / normal_length
            width = torch.linalg.cross(tangent, normal, dim=0)
            width = width / torch.linalg.vector_norm(width).clamp_min(1.0e-6)
            coordinates = (
                center[None, None, :]
                + normal_grid[..., None] * normal[None, None, :]
                + width_grid[..., None] * width[None, None, :]
            ) / float(output_stride)
            valid = torch.ones(self.config.shape, dtype=torch.bool, device=output.device)
            for axis, size in enumerate(output_shape):
                valid &= (coordinates[..., axis] >= 0.0) & (coordinates[..., axis] <= float(size - 1))
            z, y, x = coordinates.unbind(dim=-1)
            normalized = torch.stack(
                (
                    2.0 * x / max(1, output_shape[2] - 1) - 1.0,
                    2.0 * y / max(1, output_shape[1] - 1) - 1.0,
                    2.0 * z / max(1, output_shape[0] - 1) - 1.0,
                ),
                dim=-1,
            ).view(1, 1, *self.config.shape, 3)
            plane = F.grid_sample(
                presences[sample : sample + 1].float(),
                normalized,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0, 0]
            planes.append(plane)
            valid_planes.append(valid)
            if predicted_widths is not None:
                widths = F.grid_sample(predicted_widths[sample:sample+1].float(), normalized,
                    mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0, 0]
                reference = segment_reference_width(batch, sample, center, segment)
                gate = width_similarity(widths, reference,
                    fraction=annotation_config.sideband_width_fraction,
                    falloff=annotation_config.sideband_width_gate)
                filtered_planes.append(plane * gate)
                filtered_valid.append(valid & torch.isfinite(widths) & torch.isfinite(reference) & (reference > 0))

        if not planes:
            if self.width_filtered is not None:
                self.width_filtered.skipped += int(output.shape[0])
            return
        self._accumulate_planes(planes, valid_planes)
        if self.width_filtered is not None:
            self.width_filtered.skipped += int(output.shape[0]) - len(planes)
            self.width_filtered._accumulate_planes(filtered_planes, filtered_valid)

    def _accumulate_planes(self, planes, valid_planes):
        values = torch.stack(planes).detach().cpu().numpy().astype(np.float32, copy=False)
        validity = torch.stack(valid_planes).detach().cpu().numpy()
        for plane, valid in zip(values, validity, strict=True):
            finite = valid & np.isfinite(plane)
            if not finite.any():
                self.skipped += 1
                continue
            clipped = np.clip(plane, 0.0, 1.0)
            self.sum[finite] += clipped[finite]
            self.count[finite] += 1
            bins = np.minimum(
                (clipped[finite] * self.config.histogram_bins).astype(np.int64),
                self.config.histogram_bins - 1,
            )
            pixels = np.flatnonzero(finite)
            np.add.at(self.histogram.reshape(-1, self.config.histogram_bins), (pixels, bins), 1)
            self.sections += 1

    def merge(self, other: "CrossSectionAccumulator") -> None:
        if other.config != self.config:
            raise ValueError("cannot merge cross-section accumulators with different settings")
        if (self.width_filtered is None) != (other.width_filtered is None):
            raise ValueError('cannot merge raw-only and paired cross-sections')
        if self.filter_metadata is not None and other.filter_metadata is not None and self.filter_metadata != other.filter_metadata:
            raise ValueError('cross-section width gates differ')
        self.sum += other.sum
        self.count += other.count
        self.histogram += other.histogram
        self.sections += other.sections
        self.skipped += other.skipped
        if self.width_filtered is not None:
            self.filter_metadata = self.filter_metadata or other.filter_metadata
            self.width_filtered.merge(other.width_filtered)

    def mean(self) -> np.ndarray:
        return np.divide(
            self.sum,
            self.count,
            out=np.full(self.config.shape, np.nan, dtype=np.float64),
            where=self.count > 0,
        ).astype(np.float32)

    def quantile(self, value: float) -> np.ndarray:
        if not 0.0 <= value <= 1.0:
            raise ValueError("quantile must be in [0,1]")
        cumulative = np.cumsum(self.histogram, axis=-1, dtype=np.uint64)
        total = cumulative[..., -1]
        # Nearest-rank quantiles make q=0 the observed minimum rather than an
        # unobserved histogram boundary.
        rank = np.maximum(1, np.ceil(value * total).astype(np.uint64))
        indices = np.argmax(cumulative >= rank[..., None], axis=-1)
        result = (indices.astype(np.float64) + 0.5) / float(self.config.histogram_bins)
        result[total == 0] = np.nan
        return result.astype(np.float32)

    def exceedance_threshold(self, fraction: float) -> np.ndarray:
        """Return the value exceeded by ``fraction`` of observations.

        Thus 0.001 selects the 99.9th percentile and retains only the
        brightest 0.1% tail, while 0.1 selects the 90th percentile.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("exceedance fraction must be in [0,1]")
        return self.quantile(1.0 - fraction)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        filtered = {} if self.width_filtered is None else {
            'filtered_sum': self.width_filtered.sum,
            'filtered_count': self.width_filtered.count,
            'filtered_histogram': self.width_filtered.histogram,
            'filtered_sections': self.width_filtered.sections,
            'filtered_skipped': self.width_filtered.skipped,
            'filter_metadata': np.asarray(json.dumps(self.filter_metadata)),
        }
        np.savez_compressed(
            path,
            **filtered,
            sum=self.sum,
            count=self.count,
            histogram=self.histogram,
            sections=np.asarray(self.sections, dtype=np.int64),
            skipped=np.asarray(self.skipped, dtype=np.int64),
            config=np.asarray(json.dumps({
                "width_base": self.config.width_base,
                "height_base": self.config.height_base,
                "spacing_base": self.config.spacing_base,
                "histogram_bins": self.config.histogram_bins,
                "quantiles": self.config.quantiles,
            })),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "CrossSectionAccumulator":
        with np.load(path, allow_pickle=False) as data:
            raw = json.loads(str(data["config"]))
            raw["quantiles"] = tuple(raw["quantiles"])
            result = cls(CrossSectionConfig(**raw), width_filtered='filtered_sum' in data)
            result.sum[...] = data["sum"]
            result.count[...] = data["count"]
            result.histogram[...] = data["histogram"]
            result.sections = int(data["sections"])
            result.skipped = int(data["skipped"])
            if result.width_filtered is not None:
                for name in ('sum', 'count', 'histogram'):
                    getattr(result.width_filtered, name)[...] = data['filtered_' + name]
                result.width_filtered.sections = int(data['filtered_sections'])
                result.width_filtered.skipped = int(data['filtered_skipped'])
                result.filter_metadata = json.loads(str(data['filter_metadata']))
        return result


def merge_shards(paths: Iterable[str | Path]) -> CrossSectionAccumulator:
    iterator = iter(paths)
    try:
        result = CrossSectionAccumulator.load_npz(next(iterator))
    except StopIteration as exc:
        raise ValueError("no cross-section shards to merge") from exc
    for path in iterator:
        result.merge(CrossSectionAccumulator.load_npz(path))
    return result


def fit_mean_gaussian(
    mean: np.ndarray,
    config: CrossSectionConfig,
    *,
    fit_width_pixels: int = 51,
    fit_height_pixels: int = 25,
) -> tuple[np.ndarray, dict[str, float | int | list[int]]]:
    """Fit an axis-aligned 2-D Gaussian to the central mean cross-section."""
    from scipy.optimize import least_squares

    if mean.shape != config.shape:
        raise ValueError(f"mean shape {mean.shape} does not match {config.shape}")
    if fit_width_pixels < 3 or fit_height_pixels < 3:
        raise ValueError("Gaussian fit window must be at least 3x3 pixels")
    cy, cx = np.asarray(mean.shape) // 2
    half_y, half_x = fit_height_pixels // 2, fit_width_pixels // 2
    ys = slice(max(0, cy - half_y), min(mean.shape[0], cy + half_y + 1))
    xs = slice(max(0, cx - half_x), min(mean.shape[1], cx + half_x + 1))
    values = np.asarray(mean[ys, xs], dtype=np.float64)
    normal = config.normal_offsets_base[ys].astype(np.float64)
    width = config.width_offsets_base[xs].astype(np.float64)
    yy, xx = np.meshgrid(normal, width, indexing="ij")
    finite = np.isfinite(values)
    if int(finite.sum()) < 9:
        raise ValueError("not enough finite mean pixels for Gaussian fit")

    border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1]))
    background0 = float(np.nanmedian(border))
    amplitude0 = max(float(np.nanmax(values)) - background0, 1.0e-4)
    weights = np.maximum(np.where(finite, values - background0, 0.0), 0.0)
    weight_sum = float(weights.sum())
    if weight_sum > 0.0:
        center_n0 = float((weights * yy).sum() / weight_sum)
        center_w0 = float((weights * xx).sum() / weight_sum)
        sigma_n0 = max(float(np.sqrt((weights * (yy - center_n0) ** 2).sum() / weight_sum)), 1.0)
        sigma_w0 = max(float(np.sqrt((weights * (xx - center_w0) ** 2).sum() / weight_sum)), 1.0)
    else:
        center_n0 = center_w0 = 0.0
        sigma_n0 = sigma_w0 = 2.0

    def evaluate(parameters: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        background, amplitude, center_n, center_w, sigma_n, sigma_w = parameters
        exponent = -0.5 * (((y - center_n) / sigma_n) ** 2 + ((x - center_w) / sigma_w) ** 2)
        return background + amplitude * np.exp(exponent)

    initial = np.asarray((background0, amplitude0, center_n0, center_w0, sigma_n0, sigma_w0))
    lower = np.asarray((0.0, 0.0, normal[0], width[0], 0.25, 0.25))
    upper = np.asarray((1.0, 1.5, normal[-1], width[-1], 2.0 * np.ptp(normal), 2.0 * np.ptp(width)))
    initial = np.clip(initial, lower + 1.0e-8, upper - 1.0e-8)
    result = least_squares(
        lambda parameters: (evaluate(parameters, yy, xx) - values)[finite],
        initial,
        bounds=(lower, upper),
        method="trf",
    )
    background, amplitude, center_n, center_w, sigma_n, sigma_w = map(float, result.x)
    full_y, full_x = np.meshgrid(
        config.normal_offsets_base.astype(np.float64),
        config.width_offsets_base.astype(np.float64),
        indexing="ij",
    )
    fitted = evaluate(result.x, full_y, full_x).astype(np.float32)
    residual = (evaluate(result.x, yy, xx) - values)[finite]
    params: dict[str, float | int | list[int]] = {
        "fit_window_pixels": [int(values.shape[1]), int(values.shape[0])],
        "background": background,
        "amplitude": amplitude,
        "peak": background + amplitude,
        "center_normal_base": center_n,
        "center_width_base": center_w,
        "sigma_normal_base": sigma_n,
        "sigma_width_base": sigma_w,
        "fwhm_normal_base": 2.354820045 * sigma_n,
        "fwhm_width_base": 2.354820045 * sigma_w,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "optimizer_evaluations": int(result.nfev),
    }
    return fitted, params


def write_cross_section_artifacts(
    accumulator: CrossSectionAccumulator,
    directory: str | Path,
    *,
    metadata: dict | None = None,
) -> dict[str, str]:
    import tifffile

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mean = accumulator.mean()
    gaussian_fit, gaussian_parameters = fit_mean_gaussian(mean, accumulator.config)
    arrays = {"mean": mean, "mean_gaussian_fit": gaussian_fit}
    for fraction in accumulator.config.quantiles:
        arrays[f"q{fraction:g}"] = accumulator.exceedance_threshold(fraction)
    paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = directory / f"cross_section_{name}.tif"
        tifffile.imwrite(path, array, dtype=np.float32, photometric="minisblack")
        paths[name] = str(path)
    description = {
        "format": "fiber_presence_cross_section_v2",
        "axes": ["surface_normal_base", "in_sheet_transverse_base"],
        "shape": list(accumulator.config.shape),
        "normal_offsets_base": accumulator.config.normal_offsets_base.tolist(),
        "width_offsets_base": accumulator.config.width_offsets_base.tolist(),
        "histogram_bins": accumulator.config.histogram_bins,
        "histogram_range": [0.0, 1.0],
        "exceedance_fractions": list(accumulator.config.quantiles),
        "mean_gaussian_fit": gaussian_parameters,
        "sections": accumulator.sections,
        "skipped": accumulator.skipped,
        "valid_count_min": int(accumulator.count.min()),
        "valid_count_max": int(accumulator.count.max()),
        "files": paths,
        **dict(metadata or {}),
    }
    metadata_path = directory / "cross_section.json"
    metadata_path.write_text(json.dumps(description, indent=2) + "\n")
    paths["metadata"] = str(metadata_path)
    return paths
