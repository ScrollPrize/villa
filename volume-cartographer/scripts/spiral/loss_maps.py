"""Surface-aligned diagnostic loss maps for interactive Spiral previews.

Training losses are intentionally reduced to scalars.  During the paused
preview export window, this module can be enabled as a lightweight sink for
the per-sample residuals already computed by the loss functions.  Samples are
binned into the exact disconnected winding layout used by the VC3D preview.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image


_active_recorder = None


def diagnostics_enabled() -> bool:
    return _active_recorder is not None


def record_loss_samples(name, spiral_zyx, residual, mask=None) -> None:
    recorder = _active_recorder
    if recorder is not None:
        recorder.record(name, spiral_zyx, residual, mask)


@contextmanager
def capture_loss_maps(recorder, *, suppress_errors=False):
    global _active_recorder
    if _active_recorder is not None:
        raise RuntimeError("Nested Spiral loss-map capture is not supported")
    _active_recorder = recorder
    try:
        yield recorder
    except Exception as error:
        if not suppress_errors:
            raise
        recorder.error = error
    finally:
        _active_recorder = None


class LossMapRecorder:
    def __init__(self, preview_manifest, generation_path, *, z0, grid_spacing,
                 dr_per_winding, weights):
        self.generation_path = Path(generation_path)
        self.z0 = float(z0)
        self.grid_spacing = float(grid_spacing)
        self.dr_per_winding = float(dr_per_winding.detach().item()
                                    if hasattr(dr_per_winding, "detach")
                                    else dr_per_winding)
        self.weights = {str(name): float(value) for name, value in weights.items()}
        self.error = None
        self.components = {
            int(winding): (int(bounds[0]), int(bounds[1]))
            for winding, bounds in zip(preview_manifest["winding_ids"],
                                       preview_manifest["components"])
        }
        surface = self.generation_path / preview_manifest["surface_id"] / "x.tif"
        with Image.open(surface) as image:
            self.width, self.height = image.size
        self._maps = {}

    @staticmethod
    def _numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    def record(self, name, spiral_zyx, residual, mask=None):
        if name not in self.weights or self.weights[name] == 0:
            return
        points = self._numpy(spiral_zyx).reshape(-1, 3)
        values = self._numpy(residual)
        if values.size == 1:
            values = np.full(points.shape[0], float(values.reshape(-1)[0]),
                             dtype=np.float32)
        else:
            values = np.broadcast_to(values, np.asarray(spiral_zyx).shape[:-1]
                                     if not hasattr(spiral_zyx, "detach")
                                     else tuple(spiral_zyx.shape[:-1])).reshape(-1)
        if points.shape[0] != values.shape[0]:
            raise ValueError(f"Loss map {name}: point/residual shape mismatch")

        valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
        if mask is not None:
            mask_np = self._numpy(mask)
            if mask_np.size == 1:
                valid &= bool(mask_np.reshape(-1)[0])
            else:
                valid &= np.broadcast_to(mask_np, tuple(spiral_zyx.shape[:-1])).reshape(-1).astype(bool)
        if not valid.any():
            return

        points = points[valid]
        values = np.maximum(values[valid].astype(np.float64, copy=False), 0.0)
        theta = np.mod(np.arctan2(points[:, 1], points[:, 2]), 2.0 * np.pi)
        radius = np.linalg.norm(points[:, 1:], axis=1)
        shifted = np.maximum(radius - theta / (2.0 * np.pi) * self.dr_per_winding, 0.0)
        windings = np.rint(shifted / self.dr_per_winding).astype(np.int64)
        rows = np.rint((points[:, 0] - self.z0) / self.grid_spacing).astype(np.int64)

        index_chunks, value_chunks = self._maps.setdefault(name, ([], []))
        for winding in np.unique(windings):
            component = self.components.get(int(winding))
            if component is None:
                continue
            first, end = component
            component_width = end - first
            selected = windings == winding
            selected_rows = rows[selected]
            selected_cols = first + np.minimum(
                (theta[selected] / (2.0 * np.pi) * component_width).astype(np.int64),
                component_width - 1,
            )
            in_bounds = ((selected_rows >= 0) & (selected_rows < self.height)
                         & (selected_cols >= first) & (selected_cols < end))
            if not in_bounds.any():
                continue
            index_chunks.append(
                selected_rows[in_bounds] * self.width + selected_cols[in_bounds])
            value_chunks.append(values[selected][in_bounds])

    @staticmethod
    def _splat(indices, values, height, width):
        # One-pixel splat makes sparse patch/PCL/track samples visible without
        # implying support over unsampled parts of the surface.  Keep this
        # sparse until the final RGBA allocation: preview images can be large.
        rows = indices // width
        cols = indices % width
        splat_indices = []
        splat_values = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted_rows = rows + dy
                shifted_cols = cols + dx
                valid = ((shifted_rows >= 0) & (shifted_rows < height)
                         & (shifted_cols >= 0) & (shifted_cols < width))
                splat_indices.append(shifted_rows[valid] * width + shifted_cols[valid])
                splat_values.append(values[valid])
        splat_indices = np.concatenate(splat_indices)
        splat_values = np.concatenate(splat_values)
        order = np.argsort(splat_indices)
        splat_indices = splat_indices[order]
        splat_values = splat_values[order]
        unique, starts = np.unique(splat_indices, return_index=True)
        return unique, np.maximum.reduceat(splat_values, starts)

    @staticmethod
    def _rgba(values, display_maximum):
        intensity = np.zeros_like(values, dtype=np.float32)
        if display_maximum > 0:
            intensity = np.clip(values / display_maximum, 0.0, 1.0).astype(np.float32)
        # Blue -> cyan -> yellow -> red, with opacity increasing with loss.
        stops = np.array([[32, 64, 220], [20, 210, 235], [255, 220, 35], [255, 45, 20]],
                         dtype=np.float32)
        scaled = intensity * (len(stops) - 1)
        lower = np.minimum(scaled.astype(np.int32), len(stops) - 2)
        frac = (scaled - lower)[..., None]
        rgb = stops[lower] * (1.0 - frac) + stops[lower + 1] * frac
        rgba = np.empty((values.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(
            28.0 + 207.0 * np.sqrt(intensity), 0, 235).astype(np.uint8)
        return rgba

    def finish(self):
        output = self.generation_path / "loss-maps"
        output.mkdir(exist_ok=True)
        entries = []
        for name in sorted(self._maps):
            index_chunks, value_chunks = self._maps[name]
            if not index_chunks:
                continue
            sample_indices = np.concatenate(index_chunks)
            sample_values = np.concatenate(value_chunks)
            unique_indices, inverse = np.unique(sample_indices, return_inverse=True)
            sums = np.bincount(inverse, weights=sample_values)
            counts = np.bincount(inverse)
            raw = sums / counts
            weighted = raw * abs(self.weights[name])
            positive = weighted[weighted > 0]
            display_maximum = float(np.percentile(positive, 95)) if positive.size else 0.0
            if not math.isfinite(display_maximum) or display_maximum <= 0:
                display_maximum = float(weighted.max(initial=0.0))
            display_indices, display_values = self._splat(
                unique_indices, weighted, self.height, self.width)
            rgba = np.zeros((self.height * self.width, 4), dtype=np.uint8)
            rgba[display_indices] = self._rgba(display_values, display_maximum)
            rgba = rgba.reshape(self.height, self.width, 4)
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            relative_path = f"loss-maps/{safe_name}.png"
            Image.fromarray(rgba).save(self.generation_path / relative_path)
            entries.append({
                "name": name,
                "path": relative_path,
                "weight": self.weights[name],
                "sample_count": int(len(sample_indices)),
                "supported_pixels": int(len(unique_indices)),
                "p50": float(np.percentile(weighted, 50)),
                "p95": float(np.percentile(weighted, 95)),
                "maximum": float(weighted.max()),
                "display_maximum": display_maximum,
                "values": "weighted_per_sample_residual",
            })
        return entries


def attach_loss_maps_to_manifest(manifest, generation_path, entries):
    manifest = dict(manifest)
    manifest["loss_maps"] = list(entries)
    path = Path(generation_path) / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return manifest
