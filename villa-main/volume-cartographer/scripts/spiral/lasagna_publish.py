"""Flattened-preview publication through a temporary Lasagna fit service.

``LasagnaPublisher`` takes one raw Spiral preview generation, flattens it with
Lasagna, maps the winding membership and the loss/run-difference overlays onto
the flattened grid, and returns the published manifest path.

Everything the publisher reports goes through exactly one progress path
(``_report``): the stage boundaries it declares itself, the poll updates from
the Lasagna job, and the one stage the Lasagna console output tells us about.
There is no nested stage callback, no direct mutation of the service's
preview-publication status, and no second event emitter — the orchestrator
receives one stream of progress records and decides what to do with it.

The publisher never sees ``ServiceState``: the session, its lock, the status
snapshot, the event buffer, the artifact registry and the upload state are all
outside its interface. What it needs arrives as constructor callbacks
(progress, subprocess registration, session validity, and the previous raw
manifest it must diff against and then replace).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import numpy as np
from PIL import Image
import scipy.ndimage

from service_http import TRANSFER_CHUNK_BYTES


LASAGNA_PREVIEW_OUTPUT_STEP_VX = 20.0
LASAGNA_CONFIG_NAME = "flatten_fast_nofilter.json"
# Lasagna prints this once the optimizer loop is done and the flatten model is
# being written; the job status does not distinguish that phase.
_LASAGNA_SAVING_MARKER = "[fit] peak GPU memory:"
_LASAGNA_PORT_LINE = re.compile(r"listening on http://[^:]+:(\d+)")


def _find_lasagna_service():
    configured = str(os.environ.get("LASAGNA_SERVICE_PATH") or "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    here = Path(__file__).resolve()
    candidates.extend([
        here.parents[3] / "lasagna" / "fit_service.py",
        Path.home() / "villa" / "lasagna" / "fit_service.py",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot find Lasagna fit_service.py on the Spiral host; "
        "set LASAGNA_SERVICE_PATH")


def _md5_file(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(TRANSFER_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return f"md5:{digest.hexdigest()}"


def _prepare_cleaned_lasagna_surface(surface_dir, destination,
                                     erosion_cells=3):
    """Stage a Lasagna-only TIFXYZ with ragged/disconnected support removed."""
    surface_dir = Path(surface_dir).resolve(strict=True)
    destination = Path(destination)
    required = ("meta.json", "x.tif", "y.tif", "z.tif")
    missing = [name for name in required if not (surface_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Spiral preview is missing: {', '.join(missing)}")

    coordinates = []
    shape = None
    valid = None
    for name in ("x.tif", "y.tif", "z.tif"):
        with Image.open(surface_dir / name) as image:
            coordinate = np.asarray(image, dtype=np.float32)
        if coordinate.ndim != 2:
            raise RuntimeError(
                f"Spiral preview {name} must be a two-dimensional TIFF")
        if shape is None:
            shape = coordinate.shape
        elif coordinate.shape != shape:
            raise RuntimeError(
                "Spiral preview coordinate TIFF dimensions do not match")
        coordinate = coordinate.copy()
        coordinates.append(coordinate)
        coordinate_valid = coordinate != -1.0
        valid = (coordinate_valid if valid is None
                 else valid | coordinate_valid)

    cleaned = scipy.ndimage.binary_erosion(
        valid, iterations=int(erosion_cells), border_value=0)
    labels, component_count = scipy.ndimage.label(
        cleaned, structure=scipy.ndimage.generate_binary_structure(2, 1))
    if component_count == 0:
        raise RuntimeError(
            f"Lasagna input cleanup removed every valid TIFXYZ vertex "
            f"after {int(erosion_cells)}-cell erosion")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    cleaned = labels == int(np.argmax(component_sizes))

    shutil.copytree(surface_dir, destination)
    for coordinate, name in zip(coordinates, ("x.tif", "y.tif", "z.tif")):
        coordinate[~cleaned] = -1.0
        Image.fromarray(coordinate).save(destination / name)

    metadata_path = destination / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["bbox"] = [
        [float(coordinate[cleaned].min()) for coordinate in coordinates],
        [float(coordinate[cleaned].max()) for coordinate in coordinates],
    ]
    valid_quad = (
        cleaned[:-1, :-1]
        & cleaned[1:, :-1]
        & cleaned[:-1, 1:]
        & cleaned[1:, 1:]
    )
    scale = metadata.get("scale")
    if (isinstance(scale, list) and len(scale) >= 2
            and all(isinstance(value, (int, float)) and value > 0
                    for value in scale[:2])):
        area_vx2 = float(valid_quad.sum()) / (
            float(scale[0]) * float(scale[1]))
        old_area_vx2 = metadata.get("area_vx2")
        old_area_cm2 = metadata.get("area_cm2")
        metadata["area_vx2"] = area_vx2
        if (isinstance(old_area_vx2, (int, float))
                and old_area_vx2 > 0
                and isinstance(old_area_cm2, (int, float))):
            metadata["area_cm2"] = (
                area_vx2 * float(old_area_cm2) / float(old_area_vx2))
    metadata["lasagna_input_cleanup"] = {
        "erosion_cells": int(erosion_cells),
        "component_connectivity": 4,
        "components_after_erosion": int(component_count),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=4) + "\n", encoding="utf-8")
    return destination


def _prepare_lasagna_surface_object(surface_dir, object_store):
    surface_dir = Path(surface_dir).resolve(strict=True)
    required = ("meta.json", "x.tif", "y.tif", "z.tif")
    missing = [name for name in required if not (surface_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Spiral preview is missing: {', '.join(missing)}")
    lines = []
    for path in sorted(p for p in surface_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(surface_dir).as_posix()
        lines.append(f"{relative}\t{_md5_file(path)}\n")
    manifest = hashlib.md5(
        "".join(lines).encode("utf-8"), usedforsecurity=False).hexdigest()
    ref = {"type": "tifxyz_segment", "name": surface_dir.name,
           "hash": f"md5:{manifest}"}
    destination = (Path(object_store) / ref["type"] / manifest
                   / quote(ref["name"], safe=""))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "segment").symlink_to(surface_dir, target_is_directory=True)
    (destination / "object.json").write_text(
        json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    return ref


def _surface_xyz(surface_dir):
    """Load one TIFXYZ grid as HxWx3 float32 plus its validity mask."""
    coordinates = []
    for name in ("x.tif", "y.tif", "z.tif"):
        with Image.open(Path(surface_dir) / name) as image:
            coordinates.append(np.asarray(image, dtype=np.float32).copy())
    if not coordinates or any(value.shape != coordinates[0].shape
                              for value in coordinates[1:]):
        raise RuntimeError("TIFXYZ coordinate TIFF dimensions do not match")
    xyz = np.stack(coordinates, axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & ~np.all(xyz == -1.0, axis=-1)
    return xyz, valid


def _validate_tifxyz_output_step(metadata, expected_step):
    """Require exported TIFXYZ scale to match the requested grid step."""
    expected = float(expected_step)
    scale = metadata.get("scale") if isinstance(metadata, dict) else None
    if (not math.isfinite(expected) or expected <= 0.0
            or not isinstance(scale, list) or len(scale) < 2):
        raise RuntimeError(
            "Lasagna output metadata has no valid two-axis scale")
    expected_scale = 1.0 / expected
    values = []
    for raw in scale[:2]:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise RuntimeError(
                "Lasagna output metadata has a non-numeric scale") from None
        if (not math.isfinite(value) or value <= 0.0
                or not math.isclose(
                    value, expected_scale, rel_tol=1.0e-6, abs_tol=1.0e-9)):
            raise RuntimeError(
                "Lasagna output scale does not match requested preview step "
                f"{expected:g}")
        values.append(value)
    return values


def _load_flatten_correspondence(checkpoint_path=None, map_path=None):
    """Read Lasagna's flattened-output -> Spiral-source grid map."""
    if map_path is not None and Path(map_path).is_file():
        mapping = np.load(str(map_path), mmap_mode="r", allow_pickle=False)
        mapping = np.asarray(mapping, dtype=np.float32)
        if mapping.ndim != 3 or mapping.shape[-1] != 2:
            raise RuntimeError(
                "Lasagna output-to-source map must have shape (rows, columns, 2)")
        if not np.isfinite(mapping).all():
            raise RuntimeError(
                "Lasagna output-to-source map contains non-finite values")
        return mapping

    if checkpoint_path is None:
        raise RuntimeError("Lasagna produced no flatten correspondence")
    import torch

    state = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "flatten_map_flat" not in state:
        raise RuntimeError(
            "Lasagna flatten checkpoint contains no output-to-source map")
    value = state["flatten_map_flat"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mapping = np.asarray(value, dtype=np.float32)
    if mapping.ndim != 3 or mapping.shape[-1] != 2:
        raise RuntimeError(
            "Lasagna output-to-source map must have shape (rows, columns, 2)")
    if not np.isfinite(mapping).all():
        raise RuntimeError("Lasagna output-to-source map contains non-finite values")
    return mapping


def _sample_rgba_through_map(source_rgba, source_yx, output_valid, *,
                             executor=None):
    """Bilinearly warp RGBA using premultiplied alpha."""
    source = np.asarray(source_rgba, dtype=np.float32) / 255.0
    if source.ndim != 3 or source.shape[-1] != 4:
        raise RuntimeError("Mapped preview overlay must be RGBA")
    alpha = source[..., 3]
    premultiplied = source[..., :3] * alpha[..., None]
    coordinates = [source_yx[..., 0], source_yx[..., 1]]

    def sample(channel):
        values = alpha if channel == 3 else premultiplied[..., channel]
        return scipy.ndimage.map_coordinates(
            values, coordinates, order=1, mode="constant", cval=0.0,
            prefilter=False)

    if executor is None:
        sampled = [sample(channel) for channel in range(4)]
    else:
        # Each channel is independent and scipy releases the GIL here.  Mapping
        # them concurrently preserves the exact interpolation and output bytes.
        sampled = list(executor.map(sample, range(4)))
    sampled_alpha = sampled[3]
    sampled_rgb = np.stack(sampled[:3], axis=-1)
    nonzero = sampled_alpha > 1.0e-8
    sampled_rgb[nonzero] /= sampled_alpha[nonzero, None]
    sampled_rgb[~nonzero] = 0.0
    sampled_alpha = np.where(output_valid, sampled_alpha, 0.0)
    sampled_rgb = np.where(output_valid[..., None], sampled_rgb, 0.0)
    result = np.empty((*sampled_alpha.shape, 4), dtype=np.uint8)
    result[..., :3] = np.clip(
        np.rint(sampled_rgb * 255.0), 0, 255).astype(np.uint8)
    result[..., 3] = np.clip(
        np.rint(sampled_alpha * 255.0), 0, 255).astype(np.uint8)
    return result


def _mapped_winding_ids(source_manifest, source_shape, source_yx,
                        output_valid):
    """Categorically map source winding membership onto a flattened grid."""
    ranges = source_manifest.get("winding_column_ranges")
    windings = source_manifest.get("winding_ids")
    if (not isinstance(ranges, list) or not isinstance(windings, list)
            or len(ranges) != len(windings) or not ranges):
        raise RuntimeError("Spiral source preview has no winding mapping")
    winding_values = sorted({int(winding) for winding in windings})
    winding_to_dense = {
        winding: index + 1 for index, winding in enumerate(winding_values)
    }
    source_labels = np.zeros(source_shape, dtype=np.int32)
    for bounds, winding in zip(ranges, windings):
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise RuntimeError("Malformed Spiral winding column range")
        begin, end = int(bounds[0]), int(bounds[1])
        if begin < 0 or end <= begin or end > source_shape[1]:
            raise RuntimeError("Spiral winding column range is out of bounds")
        source_labels[:, begin:end] = winding_to_dense[int(winding)]

    rows = np.rint(source_yx[..., 0]).astype(np.int64)
    columns = np.rint(source_yx[..., 1]).astype(np.int64)
    in_bounds = (
        output_valid
        & (rows >= 0) & (rows < source_shape[0])
        & (columns >= 0) & (columns < source_shape[1])
    )
    dense_result = np.zeros(output_valid.shape, dtype=np.int32)
    dense_result[in_bounds] = source_labels[
        rows[in_bounds], columns[in_bounds]]
    bounds = []
    objects = scipy.ndimage.find_objects(
        dense_result, max_label=len(winding_values))
    for dense_label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        row_slice, column_slice = slices
        winding = winding_values[dense_label - 1]
        bounds.append({
            "winding": winding,
            "row_begin": int(row_slice.start),
            "row_end": int(row_slice.stop),
            "column_begin": int(column_slice.start),
            "column_end": int(column_slice.stop),
        })
    if not bounds:
        raise RuntimeError("Lasagna correspondence mapped no preview windings")
    lookup = np.asarray([-1, *winding_values], dtype=np.int32)
    result = lookup[dense_result]
    return result, bounds


def _raw_run_diff_rgba(previous_manifest, current_manifest, *,
                       current_surface_data=None):
    """Build a current-source-grid displacement overlay by winding identity."""
    if current_surface_data is None:
        current_surface = Path(current_manifest["surface_path"])
        current_xyz, current_valid = _surface_xyz(current_surface)
    else:
        current_xyz, current_valid = current_surface_data
    rgba = np.zeros((*current_valid.shape, 4), dtype=np.uint8)
    if previous_manifest is None:
        return rgba, 0
    previous_xyz, previous_valid = _surface_xyz(
        Path(previous_manifest["surface_path"]))
    previous_by_winding = {
        int(winding): bounds
        for winding, bounds in zip(
            previous_manifest.get("winding_ids", []),
            previous_manifest.get("winding_column_ranges", []))
    }
    magnitudes = np.full(current_valid.shape, np.nan, dtype=np.float32)
    for winding, current_bounds in zip(
            current_manifest.get("winding_ids", []),
            current_manifest.get("winding_column_ranges", [])):
        previous_bounds = previous_by_winding.get(int(winding))
        if previous_bounds is None:
            continue
        current_begin, current_end = map(int, current_bounds)
        previous_begin, previous_end = map(int, previous_bounds)
        rows = min(current_xyz.shape[0], previous_xyz.shape[0])
        width = min(current_end - current_begin,
                    previous_end - previous_begin)
        if rows <= 0 or width <= 0:
            continue
        current_region = current_xyz[:rows, current_begin:current_begin + width]
        previous_region = previous_xyz[
            :rows, previous_begin:previous_begin + width]
        valid = (
            current_valid[:rows, current_begin:current_begin + width]
            & previous_valid[:rows, previous_begin:previous_begin + width]
        )
        delta = np.linalg.norm(current_region - previous_region, axis=-1)
        target = magnitudes[:rows, current_begin:current_begin + width]
        target[valid] = delta[valid]
    finite = np.isfinite(magnitudes) & (magnitudes > 1.0e-6)
    if not finite.any():
        return rgba, 0
    display_maximum = float(np.percentile(magnitudes[finite], 95))
    if not math.isfinite(display_maximum) or display_maximum <= 0.0:
        display_maximum = float(np.nanmax(magnitudes))
    intensity = np.zeros_like(magnitudes)
    intensity[finite] = np.clip(
        magnitudes[finite] / display_maximum, 0.0, 1.0)
    # Match VC3D's blue/cyan/yellow/red diagnostic palette closely.
    stops = np.asarray(
        [[32, 64, 220], [20, 210, 235], [255, 220, 35], [255, 45, 20]],
        dtype=np.float32)
    scaled = intensity * (len(stops) - 1)
    lower = np.minimum(scaled.astype(np.int32), len(stops) - 2)
    fraction = (scaled - lower)[..., None]
    rgba[..., :3] = np.clip(
        stops[lower] * (1.0 - fraction)
        + stops[lower + 1] * fraction,
        0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(
        finite,
        np.clip(28.0 + 207.0 * np.sqrt(intensity), 0, 235),
        0).astype(np.uint8)
    return rgba, int(finite.sum())


def _fit_service_json(port, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}{path}", data=data,
        method="GET" if body is None else "POST",
        headers={"X-Fit-Service-API-Version": "2",
                 "Content-Type": "application/json",
                 "X-VC3D-Source": "Spiral host service"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {}
        raise RuntimeError(
            str(payload.get("error") or f"Lasagna HTTP {exc.code}")) from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def stop_process_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


class PreviewPublication:
    """The one record of preview publication state a service keeps.

    Preview publication used to be tracked by three service-side generation
    counters (registered/processed/publishing), a progress dictionary that
    carried a fourth copy of the generation, a separate error field, a stage
    start time and a subprocess handle. They all describe one thing: which
    raw preview generation is being published, how far it has got, and what
    it produced. That is this record.

    The authoritative generation number is still the session's
    ``preview_generation``; this record only tracks what the host has done
    with it. Every field is guarded by the owning service's lock.
    """

    __slots__ = ("generation", "completed_generation", "session_id",
                 "artifact", "error", "process", "previous_raw_manifest",
                 "stage_started", "progress")

    def __init__(self):
        #: Generation currently being published, or 0 when nothing is.
        self.generation = 0
        #: Highest generation the host has finished handling (published or
        #: failed); a generation is never published twice.
        self.completed_generation = 0
        self.session_id = None
        #: Registered artifact reference for the newest published generation.
        self.artifact = None
        self.error = None
        #: Temporary Lasagna subprocess, so shutdown can stop it.
        self.process = None
        #: Newest successful raw generation's manifest; the next
        #: run-difference overlay is built against it.
        self.previous_raw_manifest = None
        self.stage_started = None
        self.progress = None

    def reset_session_scope(self):
        """Forget everything session-scoped; return the raw manifest to drop."""
        previous_raw = self.previous_raw_manifest
        self.generation = 0
        self.completed_generation = 0
        self.session_id = None
        self.artifact = None
        self.error = None
        self.previous_raw_manifest = None
        self.stage_started = None
        self.progress = None
        return previous_raw

    def claim(self, session_id, generation):
        """Take ownership of one generation, or refuse a stale/duplicate one."""
        if (not generation
                or generation <= self.completed_generation
                or generation <= self.generation):
            return False
        self.generation = generation
        self.session_id = session_id
        self.error = None
        return True

    def owns(self, generation):
        return self.generation == generation and generation != 0

    def record_progress(self, generation, values):
        """Merge one progress report; return the snapshot, or None if stale."""
        if not self.owns(generation):
            return None
        current = dict(self.progress or {})
        next_stage = values.get("stage_name", current.get("stage_name"))
        if next_stage != current.get("stage_name"):
            self.stage_started = time.monotonic()
        current.update(values)
        current["generation"] = generation
        self.progress = current
        return dict(current)

    def finish(self, generation):
        """Mark one generation handled, whatever the outcome."""
        self.completed_generation = max(self.completed_generation, generation)
        if self.generation == generation:
            self.generation = 0
        self.progress = None
        self.stage_started = None

    def status_progress(self):
        """The ``publishing_preview`` progress block, or None."""
        if not self.progress:
            return None
        stage_name = str(self.progress.get("stage_name") or "").strip()
        if not stage_name:
            return None
        elapsed = (max(0.0, time.monotonic() - self.stage_started)
                   if self.stage_started is not None else 0.0)
        step = self.progress.get("step")
        total = self.progress.get("total_steps")
        return {
            "operation": "publishing_preview",
            "stage_name": stage_name,
            "detail": None,
            "step": int(step) if step is not None else None,
            "total_steps": int(total) if total is not None else None,
            "unit": "steps",
            "elapsed_seconds": elapsed,
        }


class LasagnaPublisher:
    """Publish one flattened preview generation.

    Parameters are callbacks into the lifecycle orchestrator, all optional
    except ``progress``:

    ``progress(state=..., stage_name=..., **values)``
        The single progress path. Every stage boundary, every Lasagna job
        poll, and the one console-derived stage all arrive here, in order.
    ``attach_process(process)`` / ``detach_process(process)``
        Publish the temporary Lasagna subprocess so a service shutdown can
        stop it, and withdraw it again when it is gone.
    ``session_valid()``
        False once the session that asked for this preview is gone; the
        publication then aborts instead of writing into a stale session.
    ``previous_raw_manifest()``
        Path of the previous successful raw generation's manifest, or None.
    ``adopt_raw_manifest(path)``
        Record ``path`` as the newest raw generation and return the one it
        replaces, whose directory the publisher then removes.
    """

    def __init__(self, *, progress, attach_process=None, detach_process=None,
                 session_valid=None, previous_raw_manifest=None,
                 adopt_raw_manifest=None):
        self._progress = progress
        self._attach_process = attach_process or (lambda process: None)
        self._detach_process = detach_process or (lambda process: None)
        self._session_valid = session_valid or (lambda: True)
        self._previous_raw_manifest = previous_raw_manifest or (lambda: None)
        self._adopt_raw_manifest = adopt_raw_manifest or (lambda path: None)
        self._generation = 0
        self._timing_stage = None
        self._timing_started = 0.0

    # -- the one progress path -----------------------------------------

    def _report(self, state, stage_name, **values):
        self._progress(state=state, stage_name=stage_name, **values)

    def _stage(self, state, stage_name, **values):
        """Declare a publisher-owned stage boundary and report it."""
        now = time.perf_counter()
        if self._timing_stage is not None:
            print(
                "SPIRAL_PREVIEW_TIMING "
                f"generation={self._generation} stage={self._timing_stage!r} "
                f"seconds={now - self._timing_started:.6f}",
                flush=True)
        self._timing_stage = stage_name
        self._timing_started = now
        self._report(state, stage_name, **values)

    # -- publication ---------------------------------------------------

    def publish(self, preview_manifest_path, *, session_id, generation,
                output_directory, voxel_size_um=None):
        """Flatten one raw preview generation and return its manifest path."""
        self._generation = generation
        self._timing_stage = None
        self._timing_started = time.perf_counter()
        process = None
        publish_root = None
        try:
            fit_service = _find_lasagna_service()
            config_path = fit_service.parent / "configs" / LASAGNA_CONFIG_NAME
            if not config_path.is_file():
                raise RuntimeError(
                    f"Cannot find Lasagna flatten config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(preview_manifest_path.read_text(encoding="utf-8"))
            surface_path = Path(str(manifest.get("surface_path") or ""))
            if not surface_path.is_dir():
                raise RuntimeError(
                    f"Spiral preview surface does not exist: {surface_path}")

            args = config.get("args")
            if not isinstance(args, dict):
                args = {}
            args["flatten_output_step"] = LASAGNA_PREVIEW_OUTPUT_STEP_VX
            args["flatten_output_margin"] = 0.0
            config["args"] = args
            if isinstance(voxel_size_um, (int, float)) and float(voxel_size_um) > 0.0:
                config["voxel_size_um"] = float(voxel_size_um)

            output_root = Path(output_directory).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            publish_parent = output_root / ".spiral-published" / session_id
            publish_parent.mkdir(parents=True, exist_ok=True)
            final_root = publish_parent / f"generation-{generation}"
            if final_root.exists():
                raise RuntimeError(
                    f"Refusing to overwrite published preview: {final_root}")
            publish_root = publish_parent / (
                f".generation-{generation}.incoming-{secrets.token_hex(5)}")
            publish_root.mkdir(parents=True, exist_ok=False)
            source_id = str(manifest.get("surface_id") or "")
            if not source_id:
                raise RuntimeError("Spiral preview manifest has no surface id")
            surface_id = f"{source_id}-lasagna.tifxyz"
            flattened_surface = publish_root / surface_id
            model_output = publish_root / "flatten-model.pt"

            with tempfile.TemporaryDirectory(prefix="spiral_lasagna_") as temporary:
                temporary = Path(temporary)
                object_store = temporary / "objects"
                self._stage(
                    "preparing", "Preparing Lasagna input surface",
                    output_step_vx=LASAGNA_PREVIEW_OUTPUT_STEP_VX)
                metadata = json.loads(
                    (surface_path / "meta.json").read_text(encoding="utf-8"))
                cleanup = metadata.get("lasagna_input_cleanup")
                if (not isinstance(cleanup, dict)
                        or cleanup.get("erosion_cells") != 3
                        or cleanup.get("component_connectivity") != 4
                        or not isinstance(cleanup.get("components_after_erosion"), int)
                        or cleanup["components_after_erosion"] < 1
                        or manifest.get("schema_version") != 2
                        or "components" in metadata):
                    raise RuntimeError(
                        "Spiral preview was not published with authoritative "
                        "connected-surface cleanup")
                surface_ref = _prepare_lasagna_surface_object(
                    surface_path, object_store)
                ready = threading.Event()
                port_holder = {}

                process = subprocess.Popen(
                    [sys.executable, str(fit_service), "--port", "0",
                     "--allow-no-data-dir", "--object-store-dir",
                     str(object_store)],
                    cwd=str(fit_service.parent),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=(os.name == "posix"))
                self._attach_process(process)

                def relay_output():
                    assert process.stdout is not None
                    for line in process.stdout:
                        text = line.rstrip()
                        if text:
                            print(f"SPIRAL_LASAGNA {text}", flush=True)
                        match = _LASAGNA_PORT_LINE.search(text)
                        if match:
                            port_holder["port"] = int(match.group(1))
                            ready.set()
                        if _LASAGNA_SAVING_MARKER in text:
                            # The job status has no phase for this; the
                            # console is the only source, so it is reported
                            # through the same progress path as everything
                            # else instead of touching service state.
                            self._report(
                                "saving", "Saving optimized flatten model",
                                step=0, total_steps=0, overall_progress=0.0)

                relay = threading.Thread(
                    target=relay_output, name="spiral-lasagna-log",
                    daemon=True)
                relay.start()
                if not ready.wait(60):
                    code = process.poll()
                    raise RuntimeError(
                        "Temporary Lasagna service failed to start"
                        + (f" (exit {code})" if code is not None else ""))
                port = port_holder["port"]
                config["external_surfaces"] = [surface_ref]
                request_body = {
                    "config": config,
                    "job_spec": {
                        "config": config,
                        "linked_surfaces": [surface_ref],
                    },
                    "single_segment": True,
                    "config_name": LASAGNA_CONFIG_NAME,
                    "output_name": surface_id,
                    "output_dir": str(publish_root),
                    "model_output": str(model_output),
                    "embed_job_metadata": False,
                    "omit_model": True,
                    "export_flatten_map": True,
                    "source": "Spiral host service",
                }
                accepted = _fit_service_json(
                    port, "/optimize", request_body, timeout=60)
                fit_job_id = str(accepted.get("job_id") or "")
                if not fit_job_id:
                    raise RuntimeError(
                        "Temporary Lasagna service returned no job id")
                self._stage(
                    "running", "Flattening preview surface",
                    step=0, total_steps=0, overall_progress=0.0)

                while True:
                    if not self._session_valid():
                        raise RuntimeError(
                            "The Spiral session changed while publishing its preview")
                    fit_status = _fit_service_json(
                        port, f"/jobs/{fit_job_id}", timeout=15)
                    state = str(fit_status.get("state") or "")
                    self._report(
                        state or "running",
                        str(fit_status.get("stage_name")
                            or fit_status.get("stage") or "Flattening"),
                        step=int(fit_status.get("step") or 0),
                        total_steps=int(fit_status.get("total_steps") or 0),
                        overall_progress=float(
                            fit_status.get("overall_progress") or 0.0),
                        loss=fit_status.get("loss"))
                    if state == "finished":
                        break
                    if state == "cancelled":
                        raise RuntimeError("Lasagna preview flatten was cancelled")
                    if state == "error":
                        raise RuntimeError(
                            str(fit_status.get("error")
                                or "Lasagna flatten failed"))
                    time.sleep(0.5)

                flattened_metadata_path = flattened_surface / "meta.json"
                if not flattened_metadata_path.is_file():
                    raise RuntimeError(
                        "Lasagna reported success but produced no tifxyz output")
                flatten_map_output = publish_root / ".flatten-map.npy"
                self._stage(
                    "loading", "Loading flattened preview output",
                    step=0, total_steps=0, overall_progress=0.0)
                correspondence = _load_flatten_correspondence(
                    model_output, flatten_map_output)
                flattened_xyz, flattened_valid = _surface_xyz(flattened_surface)
                if correspondence.shape[:2] != flattened_xyz.shape[:2]:
                    raise RuntimeError(
                        "Lasagna correspondence dimensions do not match "
                        "the flattened surface")
                source_xyz, source_valid = _surface_xyz(surface_path)
                self._stage(
                    "mapping", "Mapping preview winding membership",
                    step=0, total_steps=0, overall_progress=0.0)
                winding_ids, winding_bounds = _mapped_winding_ids(
                    manifest, source_xyz.shape[:2],
                    correspondence, flattened_valid)
                winding_map_name = "winding-ids.tif"
                # OpenCV/Qt do not portably decode signed-int TIFF samples.
                # IEEE float32 represents every supported winding id exactly
                # and is converted back to int32 after validation in VC3D.
                Image.fromarray(
                    winding_ids.astype(np.float32), mode="F").save(
                    publish_root / winding_map_name)

                mapped_loss_maps = []
                loss_output = publish_root / "loss-maps"
                loss_output.mkdir(exist_ok=True)
                loss_entries = [
                    entry for entry in manifest.get("loss_maps", [])
                    if isinstance(entry, dict)
                    and (preview_manifest_path.parent
                         / str(entry.get("path") or "")).is_file()
                ]
                remap_total = len(loss_entries)
                self._stage(
                    "mapping", (
                        f"Remapping preview loss maps (0/{remap_total})"
                        if remap_total else "No preview loss maps to remap"),
                    step=0, total_steps=remap_total,
                    overall_progress=0.0)
                with ThreadPoolExecutor(
                        max_workers=4,
                        thread_name_prefix="spiral-overlay-channel") as remap_executor:
                    for loss_index, entry in enumerate(loss_entries, start=1):
                        relative = str(entry.get("path") or "")
                        self._report(
                            "mapping",
                            (f"Remapping preview loss maps "
                             f"({loss_index}/{remap_total}): "
                             f"{Path(relative).name}"),
                            step=loss_index - 1, total_steps=remap_total,
                            overall_progress=(
                                float(loss_index - 1) / float(remap_total)
                                if remap_total else 1.0))
                        source_overlay = preview_manifest_path.parent / relative
                        with Image.open(source_overlay) as image:
                            source_rgba = np.asarray(
                                image.convert("RGBA"), dtype=np.uint8)
                        mapped = _sample_rgba_through_map(
                            source_rgba, correspondence, flattened_valid,
                            executor=remap_executor)
                        destination = loss_output / Path(relative).name
                        Image.fromarray(mapped, mode="RGBA").save(destination)
                        mapped_entry = dict(entry)
                        mapped_entry["path"] = (
                            Path("loss-maps") / destination.name).as_posix()
                        mapped_entry["supported_pixels"] = int(
                            np.count_nonzero(mapped[..., 3]))
                        mapped_loss_maps.append(mapped_entry)

                    self._stage(
                        "mapping", "Building preview run difference",
                        step=0, total_steps=0, overall_progress=0.0)
                    previous_manifest = None
                    previous_path = self._previous_raw_manifest()
                    if previous_path and Path(previous_path).is_file():
                        previous_manifest = json.loads(
                            Path(previous_path).read_text(encoding="utf-8"))
                    raw_diff, changed_pixels = _raw_run_diff_rgba(
                        previous_manifest, manifest,
                        current_surface_data=(source_xyz, source_valid))
                    mapped_diff = _sample_rgba_through_map(
                        raw_diff, correspondence, flattened_valid,
                        executor=remap_executor)
                run_diff = None
                if changed_pixels:
                    run_diff_name = "run-diff.png"
                    Image.fromarray(mapped_diff, mode="RGBA").save(
                        publish_root / run_diff_name)
                    run_diff = {
                        "path": run_diff_name,
                        "changed_source_pixels": changed_pixels,
                        "supported_pixels": int(
                            np.count_nonzero(mapped_diff[..., 3])),
                    }

                self._stage(
                    "finalizing", "Finalizing preview metadata",
                    step=0, total_steps=0, overall_progress=0.0)
                flattened_metadata = json.loads(
                    flattened_metadata_path.read_text(encoding="utf-8"))
                _validate_tifxyz_output_step(
                    flattened_metadata, LASAGNA_PREVIEW_OUTPUT_STEP_VX)
                flattened_metadata.pop("components", None)
                flattened_metadata.pop("winding_column_ranges", None)
                flattened_metadata.pop("model_source", None)
                flattened_metadata["uuid"] = surface_id
                flattened_metadata["name"] = surface_id
                flattened_metadata["grid_shape"] = [
                    int(flattened_xyz.shape[0]), int(flattened_xyz.shape[1])]
                flattened_metadata["output_step_vx"] = (
                    LASAGNA_PREVIEW_OUTPUT_STEP_VX)
                flattened_metadata["winding_id_map"] = winding_map_name
                flattened_metadata["winding_id_dtype"] = "float32_integer"
                flattened_metadata["winding_bounds"] = winding_bounds
                flattened_metadata["component_winding_ids"] = [
                    item["winding"] for item in winding_bounds]
                flattened_metadata_path.write_text(
                    json.dumps(flattened_metadata, indent=4) + "\n",
                    encoding="utf-8")

                published = dict(manifest)
                published["schema_version"] = 3
                published["surface_id"] = surface_id
                published["surface_path"] = str(final_root / surface_id)
                published["manifest_path"] = str(final_root / "manifest.json")
                published["output_step_vx"] = LASAGNA_PREVIEW_OUTPUT_STEP_VX
                published["grid_shape"] = [
                    int(flattened_xyz.shape[0]), int(flattened_xyz.shape[1])]
                published["winding_ids"] = [
                    item["winding"] for item in winding_bounds]
                published["winding_id_map"] = winding_map_name
                published["winding_id_dtype"] = "float32_integer"
                published["winding_bounds"] = winding_bounds
                published["loss_maps"] = mapped_loss_maps
                published.pop("winding_column_ranges", None)
                published.pop("components", None)
                if run_diff is None:
                    published.pop("run_diff", None)
                else:
                    published["run_diff"] = run_diff
                (publish_root / "manifest.json").write_text(
                    json.dumps(published, indent=2) + "\n",
                    encoding="utf-8")

                # These are transient transport files.  The interactive Spiral
                # preview never exposes a Lasagna model to VC3D.
                del correspondence
                model_output.unlink(missing_ok=True)
                flatten_map_output.unlink(missing_ok=True)
                (flattened_surface / "model.pt").unlink(missing_ok=True)
                os.replace(publish_root, final_root)
                publish_root = None

                old_raw = self._adopt_raw_manifest(str(preview_manifest_path))
                if old_raw and old_raw != str(preview_manifest_path):
                    shutil.rmtree(Path(old_raw).parent, ignore_errors=True)
                self._stage(
                    "finalizing", "Preparing preview artifact index",
                    step=0, total_steps=0, overall_progress=0.0)
                return final_root / "manifest.json"
        finally:
            stop_process_group(process)
            if publish_root is not None:
                shutil.rmtree(publish_root, ignore_errors=True)
            self._detach_process(process)
