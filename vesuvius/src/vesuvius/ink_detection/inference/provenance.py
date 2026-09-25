"""Provenance and physical scale for flat ink prediction TIFFs.

A prediction TIFF written by :mod:`vesuvius.ink_detection.inference.infer`
used to carry nothing but a ``Software`` tag, so nobody reading the file
later could tell which checkpoint, layer window, direction or input volume
produced it (villa issue #1691), and it carried no physical scale even
though the rendered surface volume it came from knows its voxel size (the
C++ renderers already write DPI since #785).

This module builds the JSON record for the TIFF ``ImageDescription`` tag
and the ``XResolution``/``YResolution`` tags. Reading them back needs no
code: ``tiffcomment out.tif`` (shipped with tifffile) prints the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

PROVENANCE_KEY = "vesuvius_ink_inference"
SCHEMA_VERSION = 1
TOOL_NAME = "vesuvius.ink_detection.inference.infer"
# First line of the ImageDescription tag; the JSON document follows on the
# next line. The header exists so the description never starts with "{":
# tifffile treats a description that starts with "{" and contains
# '"shape":' as its own shaped-array metadata and then fails to read the
# file (TiffPage.shaped_description).
DESCRIPTION_HEADER = f"{PROVENANCE_KEY} v{SCHEMA_VERSION}"

# TIFF resolution is written in pixels per inch, the convention
# vc_render_tifxyz / vc_zarr_to_tiff use (25400 / voxel size in um).
_MICROMETRES_PER_INCH = 25400
# Resolution tags are written only for these unit spellings. Other units
# are recorded in the JSON as declared and produce no tags: the renderer's
# default `--voxel-unit` label has not always matched the value it writes,
# and a wrong tag is worse than none.
_MICROMETRE_UNITS = frozenset({"micrometer", "micrometre", "micron", "um", "µm", "μm"})
_MAX_RATIONAL = 2**32 - 1


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> str | None:
    """``git describe --always --dirty`` of the checkout this runs from."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            try:
                completed = subprocess.run(
                    ["git", "-C", str(parent), "describe", "--always", "--dirty"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            revision = completed.stdout.strip()
            return revision or None
    return None


def package_version() -> str | None:
    """Installed ``vesuvius`` distribution version, if resolvable."""

    try:
        from importlib.metadata import version

        return version("vesuvius")
    except Exception:  # pragma: no cover - metadata absent in odd installs
        return None


def created_timestamp() -> str:
    """UTC creation time; honours ``SOURCE_DATE_EPOCH`` for reproducible files."""

    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        try:
            moment = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            moment = datetime.now(timezone.utc)
    else:
        moment = datetime.now(timezone.utc)
    return moment.replace(microsecond=0).isoformat()


def public_location(location: str | Path) -> str:
    """A path or URL safe to publish: URL userinfo (``user:pass@``) removed."""

    text = str(location)
    parts = urlsplit(text)
    if parts.scheme and parts.netloc and "@" in parts.netloc:
        host = parts.netloc.rpartition("@")[2]
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return text


def _multiscales(attrs: Any) -> Sequence[Any] | None:
    try:
        entry = attrs.get("multiscales")
        if not entry:
            ome = attrs.get("ome")
            entry = ome.get("multiscales") if isinstance(ome, Mapping) else None
    except Exception:
        return None
    return entry or None


def physical_scale_from_root(
    root: Any,
    level: str,
    *,
    depth_axis_first: bool,
) -> dict[str, Any] | None:
    """Read the OME-Zarr ``scale`` transform for one pyramid level.

    Returns ``None`` when the input is a bare array or the multiscale
    metadata does not describe the requested level. The result records the
    per-axis scale and unit as declared, and the in-plane micrometres per
    pixel only when the declared unit is a micrometre spelling.
    """

    attrs = getattr(root, "attrs", None)
    if attrs is None:
        return None
    multiscales = _multiscales(attrs)
    if not multiscales:
        return None
    entry = multiscales[0]
    if not isinstance(entry, Mapping):
        return None
    axes = entry.get("axes") or []
    axis_names = [
        str(axis.get("name")).lower() if isinstance(axis, Mapping) else str(axis).lower()
        for axis in axes
    ]
    axis_units = [
        axis.get("unit") if isinstance(axis, Mapping) else None for axis in axes
    ]
    dataset = None
    for candidate in entry.get("datasets") or []:
        if isinstance(candidate, Mapping) and str(candidate.get("path")) == str(level):
            dataset = candidate
            break
    if dataset is None:
        return None
    scale = None
    for transform in dataset.get("coordinateTransformations") or []:
        if isinstance(transform, Mapping) and transform.get("type") == "scale":
            scale = [float(value) for value in transform.get("scale") or []]
            break
    if not scale:
        return None

    if not axis_names or len(axis_names) != len(scale):
        axis_names = (["z", "y", "x"] if depth_axis_first else ["y", "x", "z"])[: len(scale)]
        axis_units = [None] * len(scale)
    by_axis = dict(zip(axis_names, scale))
    unit_by_axis = dict(zip(axis_names, axis_units))
    unit = unit_by_axis.get("y") or unit_by_axis.get("x") or None
    result: dict[str, Any] = {
        "level": str(level),
        "axes": axis_names,
        "scale": scale,
        "unit": unit,
        "um_per_px_y": None,
        "um_per_px_x": None,
    }
    if unit and str(unit).lower() in _MICROMETRE_UNITS and "y" in by_axis and "x" in by_axis:
        result["um_per_px_y"] = float(by_axis["y"])
        result["um_per_px_x"] = float(by_axis["x"])
    return result


def _pixels_per_inch(um: float) -> tuple[int, int] | None:
    """``25400 / um`` as a TIFF RATIONAL (two uint32), exact for short decimals.

    Returns ``None`` when the value itself exceeds uint32 (pixels finer than
    about 6e-6 um), rather than writing a truncated tag.
    """

    value = Fraction(_MICROMETRES_PER_INCH) / Fraction(repr(float(um)))
    if value > _MAX_RATIONAL:
        return None
    # Cap the denominator by the value so the numerator fits as well;
    # limit_denominator(2**32 - 1) alone leaves numerators near 1e13 for
    # any pixel size that is not a short decimal.
    cap = max(1, min(_MAX_RATIONAL, int(_MAX_RATIONAL / value)))
    approximation = value.limit_denominator(cap)
    if approximation.numerator > _MAX_RATIONAL:
        approximation = value.limit_denominator(max(1, cap // 2))
    return int(approximation.numerator), int(approximation.denominator)


def resolution_tags(
    um_per_px_y: float | None, um_per_px_x: float | None
) -> tuple[tuple[tuple[int, int], tuple[int, int]], str] | None:
    """TIFF ``resolution=(x, y)`` and ``resolutionunit`` for a pixel size in um.

    Returns ``None`` when the scale is unknown or not positive.
    """

    if not um_per_px_y or not um_per_px_x:
        return None
    if um_per_px_y <= 0 or um_per_px_x <= 0:
        return None
    x_res = _pixels_per_inch(um_per_px_x)
    y_res = _pixels_per_inch(um_per_px_y)
    if x_res is None or y_res is None:
        return None
    return (x_res, y_res), "INCH"


def build_provenance(
    *,
    checkpoint: Mapping[str, Any],
    input_zarr: str | Path,
    level: str,
    input_shape: Sequence[int],
    depth_axis_first: bool,
    layer_indices: Sequence[int],
    layer_start: int | None,
    layer_end: int | None,
    direction: str,
    patch_size: int,
    stride: int,
    overlap: float,
    blend_mode: str,
    tta_mirror: bool,
    amp_dtype: str | None,
    compile_requested: bool,
    compile_mode: str | None,
    batch_size: int,
    device: str,
    torch_version: str,
    mask_name: str | None,
    physical_scale: Mapping[str, Any] | None,
    preprocessing: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON document written to ``ImageDescription``."""

    return {
        PROVENANCE_KEY: {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "vesuvius_version": package_version(),
            "source_revision": source_revision(),
            "created_utc": created_timestamp(),
            "checkpoint": dict(checkpoint),
            "input": {
                "zarr": public_location(input_zarr),
                "level": str(level),
                "shape": [int(value) for value in input_shape],
                "depth_axis_first": bool(depth_axis_first),
            },
            "run": {
                "layer_indices": [int(value) for value in layer_indices],
                "layer_start": layer_start,
                "layer_end": layer_end,
                "direction": str(direction),
                "patch_size": int(patch_size),
                "stride": int(stride),
                "overlap": float(overlap),
                "blend_mode": str(blend_mode),
                "tta_mirror": bool(tta_mirror),
                "amp_dtype": amp_dtype,
                "compile_requested": bool(compile_requested),
                "compile_mode": compile_mode,
                "batch_size": int(batch_size),
                "device": str(device),
                "torch_version": str(torch_version),
                "mask": mask_name,
                "preprocessing": preprocessing,
            },
            "physical_scale": None if physical_scale is None else dict(physical_scale),
        }
    }


def provenance_description(document: Mapping[str, Any]) -> str:
    """Serialise the document for the ASCII ``ImageDescription`` tag.

    One header line (:data:`DESCRIPTION_HEADER`) then the JSON document.
    """

    body = json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{DESCRIPTION_HEADER}\n{body}"
