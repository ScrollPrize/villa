#!/usr/bin/env python3
"""Prepare auditable canvas-offset CT evidence.

The source annotation render is copied as one object when the ink dataset
ships ``*_max_<first>_<last>.tif``. Exact-center images and target comparison
slabs are read from surface-volume Zarrs through vc_reader.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image
from vc_reader import images
from vc_reader.reader import ObjectStore, open_group, open_zarr

from vesuvius.utils.cli import HyphenUnderscoreParser


# The ink dataset is private: there is no usable default, every run must
# name the location of the caller's own copy explicitly.
DEFAULT_INK_ROOT: Optional[str] = None
DEFAULT_OPEN_DATA_ROOT = "s3://vesuvius-challenge-open-data"
DEFAULT_AWS_CREDENTIALS_FILE: Optional[Path] = None
def _remote_join(root: str, *parts: str) -> str:
    return root.rstrip("/") + "/" + "/".join(
        part.strip("/") for part in parts if part
    )


def _load_aws_credentials(path: Optional[Path]) -> Optional[Path]:
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return None
    if path is None or not path.is_file():
        return None
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"AWS_[A-Z0-9_]+", key):
            continue
        parsed = shlex.split(raw_value, posix=True)
        if len(parsed) != 1:
            raise ValueError(
                f"cannot parse {key} in AWS credentials file {path}"
            )
        os.environ[key] = parsed[0]
        loaded += 1
    if not loaded:
        raise ValueError(f"no AWS_* assignments found in {path}")
    print(f"Loaded AWS environment from {path}", flush=True)
    return path


@dataclass(frozen=True)
class ZarrLevel:
    remote: str
    level: str
    attrs: dict[str, Any]
    metadata: dict[str, Any]
    level_zero_metadata: dict[str, Any]
    scale_zyx: tuple[float, float, float]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.metadata["shape"])

    @property
    def chunks(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.metadata["chunks"])

    def open(self, **reader_options):
        return open_zarr(_remote_join(self.remote, self.level), **reader_options)


def _array_metadata(array) -> dict[str, Any]:
    return {"shape": list(array.shape), "chunks": list(array.chunks), "dtype": str(array.dtype)}


def inspect_zarr(remote: str, preferred_level: int) -> ZarrLevel:
    group = open_group(remote)
    attrs = group.attrs
    datasets = [
        dataset
        for multiscale in attrs.get("multiscales") or []
        for dataset in multiscale.get("datasets") or []
    ]
    numeric = sorted(
        int(str(dataset["path"]))
        for dataset in datasets
        if str(dataset.get("path", "")).isdigit()
    )
    if not numeric:
        raise ValueError(f"{remote} has no numeric multiscale datasets")
    candidates = [value for value in numeric if value <= preferred_level]
    level = str(max(candidates) if candidates else min(numeric))
    dataset = next(
        item for item in datasets if str(item["path"]) == level
    )
    scale = (1.0, 1.0, 1.0)
    for transform in dataset.get("coordinateTransformations") or []:
        if transform.get("type") == "scale":
            values = tuple(float(value) for value in transform["scale"])
            if len(values) == 3:
                scale = values
    return ZarrLevel(
        remote=remote.rstrip("/"),
        level=level,
        attrs=attrs,
        metadata=_array_metadata(group.array(level)),
        level_zero_metadata=_array_metadata(group.array("0")),
        scale_zyx=scale,
    )


def extract_composite(
    info: ZarrLevel,
    z_indices: Sequence[int],
    output_path: Path,
    *,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Extract a maximum composite, reading one band of rows of the slab at a time."""

    audit_path = output_path.with_suffix(".json")
    if output_path.exists() and audit_path.exists() and not overwrite:
        print(f"Cache hit: {output_path}", flush=True)
        return json.loads(audit_path.read_text(encoding="utf-8"))

    shape = info.shape
    chunks = info.chunks
    dtype = np.dtype(info.metadata["dtype"])
    indices = np.unique(
        np.clip(np.asarray(z_indices, dtype=np.int64), 0, shape[0] - 1)
    )
    if not np.all(np.diff(indices) == 1):
        raise ValueError("selected Z layers must be contiguous")
    if len(shape) != 3:
        raise ValueError(f"expected a z/y/x array, got shape {shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.partial.tif"
    )
    partial_output.unlink(missing_ok=True)
    # Paris4 level-2 composites exceed 200 MiB.  Back the output directly by
    # its final TIFF instead of briefly retaining another complete raster in
    # RAM while Napari may still be releasing the previous case.
    output = images.memmap(partial_output, (shape[1], shape[2]), dtype)
    print(
        f"Extracting {info.remote} level {info.level} Z "
        f"{indices[0]}..{indices[-1]} -> {output_path}",
        flush=True,
    )
    reader = info.open(threads=workers)
    try:
        for row in range(0, shape[1], chunks[1]):
            band = reader[int(indices[0]) : int(indices[-1]) + 1, row : row + chunks[1], :]
            output[row : row + band.shape[1]] = band.max(axis=0)
        output.flush()
        transferred = int(reader.stats().get("remote_bytes", 0))
    except BaseException:
        del output
        partial_output.unlink(missing_ok=True)
        raise
    finally:
        reader.close()

    del output
    partial_output.replace(output_path)
    audit = {
        "transport": "vc_reader",
        "source": info.remote,
        "pyramid_level": int(info.level),
        "scale_zyx": list(info.scale_zyx),
        "full_resolution_shape_zyx": info.level_zero_metadata["shape"],
        "selected_level_shape_zyx": list(shape),
        "chunk_shape_zyx": list(chunks),
        "selected_z_indices": indices.tolist(),
        "composite": "single" if len(indices) == 1 else "maximum",
        "output": str(output_path),
        "transferred_bytes": transferred,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Cached {output_path} ({transferred / 1024**2:.1f} MiB read)",
        flush=True,
    )
    return audit


def _preserves_z_planes(info: ZarrLevel) -> bool:
    """Return whether a pyramid level retains level-0 Z plane indices."""

    level_zero_shape = info.level_zero_metadata.get("shape")
    return (
        isinstance(level_zero_shape, (list, tuple))
        and len(level_zero_shape) == 3
        and int(level_zero_shape[0]) == int(info.shape[0])
    )


def _copy_annotation_render(
    dataset_remote: str, output_dir: Path, overwrite: bool
) -> tuple[Optional[Path], Optional[tuple[int, int]]]:
    store = ObjectStore(dataset_remote)
    candidates = sorted(
        name for name in store.keys()
        if "/" not in name and re.search(r"_max_.*_.*\.tif$", name)
    )
    if not candidates:
        return None, None
    if len(candidates) != 1:
        raise ValueError(
            f"expected at most one shipped max TIFF in {dataset_remote}, "
            f"found {candidates}"
        )
    name = candidates[0]
    match = re.search(r"_max_(\d+)_(\d+)\.tif$", name)
    if match is None:
        raise ValueError(f"cannot parse slab indices from {name}")
    output = output_dir / f"source-annotation-{name}"
    if overwrite or not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.{os.getpid()}.partial")
        try:
            partial.write_bytes(store.read(name))
            partial.replace(output)
        finally:
            partial.unlink(missing_ok=True)
    return output, (int(match.group(1)), int(match.group(2)))


def _resize_annotation_render(
    source: Path,
    shape_yx: tuple[int, int],
    output: Path,
    overwrite: bool,
) -> Path:
    with images.open(source) as reader:
        source_shape = tuple(int(value) for value in reader.shape)
        source_dtype = np.dtype(reader.dtype)
    if source_shape == shape_yx:
        return source
    if len(source_shape) != 2:
        raise ValueError(f"{source} is not a 2D render: {source_shape}")
    source_aspect = source_shape[1] / source_shape[0]
    target_aspect = shape_yx[1] / shape_yx[0]
    if abs(source_aspect / target_aspect - 1.0) > 0.01:
        raise ValueError(
            f"annotation render {source_shape} is not the same canvas "
            f"aspect as selected Zarr level {shape_yx}"
        )
    if not overwrite and output.exists():
        return output
    factor_y = source_shape[0] // shape_yx[0]
    factor_x = source_shape[1] // shape_yx[1]
    integer_downsample = (
        factor_y >= 1
        and factor_x >= 1
        and source_shape[0] == shape_yx[0] * factor_y
        and source_shape[1] == shape_yx[1] * factor_x
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if integer_downsample:
        if output.exists():
            output.unlink()
        destination = images.memmap(output, shape_yx, source_dtype)
        band = factor_y * max(1, 1024 // factor_y)
        with images.open(source) as reader:
            for row in range(0, source_shape[0], band):
                block = reader[row : row + band, :]
                usable_y = block.shape[0] // factor_y * factor_y
                usable_x = block.shape[1] // factor_x * factor_x
                reduced = block[:usable_y, :usable_x].reshape(
                    usable_y // factor_y,
                    factor_y,
                    usable_x // factor_x,
                    factor_x,
                ).mean(axis=(1, 3))
                if np.issubdtype(destination.dtype, np.integer):
                    reduced = np.rint(reduced)
                destination[
                    row // factor_y : row // factor_y + reduced.shape[0],
                    : reduced.shape[1],
                ] = reduced.astype(destination.dtype)
        destination.flush()
    else:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(source) as image:
            resized = image.resize(
                (shape_yx[1], shape_yx[0]), Image.Resampling.BOX
            )
            resized.save(output)
    return output


def _source_resolution_um(
    selection: dict[str, Any],
    source: ZarrLevel,
    target_resolution_um: float,
    override: Optional[float],
) -> tuple[float, str]:
    if override is not None:
        return float(override), "command line"
    source_zarr = str(source.attrs.get("source_zarr") or "")
    match = re.search(
        r"(?:^|[_.\/-])(\d+(?:\.\d+)?)um(?=[_.\/-]|$)", source_zarr
    )
    if match is not None:
        return float(match.group(1)), "source surface Zarr provenance"
    volume_id = str(selection["segment"]["original_volume_id"])
    matches = [
        float(volume["resolution_um"])
        for volume in selection.get("surface_volumes") or []
        if str(volume.get("volume_id")) == volume_id
    ]
    if len(matches) == 1:
        return matches[0], "matching source volume in selection.json"
    # Older ink selections often name only the newly rendered target volume.
    # For same-resolution segmentation updates its physical Z step is the
    # best available default, but the provenance remains explicit.
    return (
        float(target_resolution_um),
        "target volume resolution fallback; override with "
        "--source-resolution-um for a different source scan",
    )


def _level_zero_z_scale(info: ZarrLevel) -> Optional[float]:
    """Return the level-0 dataset's Z scale factor, if declared."""

    for multiscale in info.attrs.get("multiscales") or []:
        for dataset in multiscale.get("datasets") or []:
            if str(dataset.get("path")) != "0":
                continue
            for transform in dataset.get("coordinateTransformations") or []:
                if transform.get("type") == "scale":
                    values = [float(value) for value in transform["scale"]]
                    if len(values) == 3:
                        return values[0]
    return None


def _surface_z_step_um(
    info: ZarrLevel, resolution_um: float, level_zero: bool = False
) -> float:
    """Return one surface-render Z-index step in physical micrometers.

    ``level_zero=True`` returns the step between level-0 Z indices (the
    frame annotation provenance indices are expressed in); otherwise the
    step is in the selected level's index frame.
    """

    axes = [
        axis
        for multiscale in info.attrs.get("multiscales") or []
        for axis in (multiscale.get("axes") or [])[:1]
    ]
    unit = str(axes[0].get("unit", "")).lower() if axes else ""
    zero_scale = _level_zero_z_scale(info)
    if unit in {"micrometer", "micrometre", "µm", "um"}:
        if level_zero and zero_scale is not None:
            return float(zero_scale)
        return float(info.scale_zyx[0])
    slice_step = float(info.attrs.get("slice_step", 1.0))
    if not math.isfinite(slice_step) or slice_step <= 0:
        raise ValueError(f"invalid surface slice_step: {slice_step}")
    step = slice_step * float(resolution_um)
    if not level_zero and zero_scale:
        step *= float(info.scale_zyx[0]) / float(zero_scale)
    return step


def _source_dataset_remote(selection: dict[str, Any], root: str) -> str:
    path = selection["source_surface_zarrs"][0]["path"].strip("/")
    if path.startswith("ink/"):
        path = path[len("ink/") :]
    return _remote_join(root, str(Path(path).parent))


def _target_zarr_remote(
    selection: dict[str, Any], case_dir: Path, root: str
) -> str:
    tifxyz = sorted((case_dir / "open-data").glob("*um-*.tifxyz"))
    if not tifxyz:
        raise ValueError(f"{case_dir} has no open-data TIFXYZ")
    resolution = float(tifxyz[0].name.split("um-", 1)[0])
    matches = [
        volume
        for volume in selection["surface_volumes"]
        if math.isclose(
            float(volume["resolution_um"]), resolution, abs_tol=1e-6
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot select {resolution:g}um surface volume from selection"
        )
    return _remote_join(root, matches[0]["path"])


def _surface_resolution_for_remote(
    selection: dict[str, Any], remote: str
) -> float:
    matches = [
        float(volume["resolution_um"])
        for volume in selection.get("surface_volumes") or []
        if remote.rstrip("/").endswith(volume["path"].strip("/"))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one surface-volume resolution for {remote}, got "
            f"{matches}"
        )
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = HyphenUnderscoreParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--zarr-level", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--exact-center-only",
        action="store_true",
        help=(
            "prepare only the inexpensive exact-center comparison; use this "
            "as a first pass for very large canvases before fetching the "
            "physically matched annotation slab"
        ),
    )
    parser.add_argument(
        "--source-resolution-um",
        type=float,
        help=(
            "physical source Z step used to match a shipped max slab; "
            "normally inferred, but required for differing scan resolutions"
        ),
    )
    parser.add_argument(
        "--ink-root",
        default=DEFAULT_INK_ROOT,
        help=(
            "location (s3://, a URL or a path) of your copy of the private ink dataset "
            "(required, e.g. myremote:bucket/datasets/ink/ink_YYYYMM)"
        ),
    )
    parser.add_argument(
        "--open-data-root",
        default=DEFAULT_OPEN_DATA_ROOT,
        help=(
            "location of the public Vesuvius open-data bucket"
        ),
    )
    parser.add_argument(
        "--aws-credentials-file",
        type=Path,
        default=DEFAULT_AWS_CREDENTIALS_FILE,
        help=(
            "optional shell-format AWS exports loaded only when "
            "AWS_ACCESS_KEY_ID is absent (default: none; authenticate via "
            "the environment or ~/.aws)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ink_root:
        raise SystemExit(
            "error: --ink-root is required; the ink dataset is "
            "private, so point it at your own copy "
            "(e.g. s3://bucket/datasets/ink/ink_YYYYMM)"
        )
    credentials_path = _load_aws_credentials(
        args.aws_credentials_file.expanduser()
        if args.aws_credentials_file
        else None
    )
    case_dir = args.case_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else case_dir / "renders" / "offset-evidence"
    )
    selection = json.loads(
        (case_dir / "selection.json").read_text(encoding="utf-8")
    )
    print(f"Preparing canvas-offset evidence for {case_dir}", flush=True)
    dataset_remote = _source_dataset_remote(
        selection, args.ink_root
    )
    source_zarr_name = Path(
        selection["source_surface_zarrs"][0]["path"]
    ).name
    source_zarr_remote = _remote_join(dataset_remote, source_zarr_name)
    target_zarr_remote = _target_zarr_remote(
        selection, case_dir, args.open_data_root
    )
    target_resolution = _surface_resolution_for_remote(
        selection, target_zarr_remote
    )
    print(f"Inspecting source Zarr: {source_zarr_remote}", flush=True)
    source = inspect_zarr(source_zarr_remote, args.zarr_level)
    print(f"Inspecting target Zarr: {target_zarr_remote}", flush=True)
    target = inspect_zarr(target_zarr_remote, args.zarr_level)

    target_middle = target.shape[0] // 2
    source_center_path = output_dir / "source-center.tif"
    target_center_path = output_dir / "target-center.tif"
    source_preserves_z = _preserves_z_planes(source)
    if source_preserves_z:
        source_middle = source.shape[0] // 2
        source_center = extract_composite(
            source,
            [source_middle],
            source_center_path,
            workers=args.workers,
            overwrite=args.overwrite,
        )
    else:
        source_center = {
            "available": False,
            "reason": (
                f"source pyramid level {source.level} has "
                f"{source.shape[0]} Z planes versus "
                f"{source.level_zero_metadata.get('shape', ['?'])[0]} at "
                "level 0; it is not an exact-center plane"
            ),
        }
        print(
            "Skipping exact-center evidence: " + source_center["reason"],
            flush=True,
        )
    target_center = extract_composite(
        target,
        [target_middle],
        target_center_path,
        workers=args.workers,
        overwrite=args.overwrite,
    )

    annotation_path = None
    annotation_indices = None
    if not args.exact_center_only:
        annotation_path, annotation_indices = _copy_annotation_render(
            dataset_remote, output_dir, args.overwrite
        )
    comparisons: list[dict[str, Any]] = []
    if source_preserves_z:
        comparisons.append(
            {
                "name": "exact-center",
                "source_render": str(source_center_path),
                "target_render": str(target_center_path),
                "purpose": "geometrically pure center-layer comparison",
            }
        )
    matched_target = None
    if annotation_path is not None and annotation_indices is not None:
        annotation_path = _resize_annotation_render(
            annotation_path,
            source.shape[1:],
            output_dir
            / f"source-annotation-level{source.level}.tif",
            args.overwrite,
        )
        source_resolution, resolution_provenance = _source_resolution_um(
            selection, source, target_resolution, args.source_resolution_um
        )
        half_layers = (annotation_indices[1] - annotation_indices[0]) / 2
        source_z_um = _surface_z_step_um(
            source, source_resolution, level_zero=True
        )
        half_thickness_um = half_layers * source_z_um
        target_z_um = _surface_z_step_um(target, target_resolution)
        radius = max(0, int(round(half_thickness_um / target_z_um)))
        selected = range(
            max(0, target_middle - radius),
            min(target.shape[0], target_middle + radius + 1),
        )
        matched_target_path = output_dir / "target-matched-max.tif"
        matched_target = extract_composite(
            target,
            list(selected),
            matched_target_path,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        comparisons.append(
            {
                "name": "annotation-matched-slab",
                "source_render": str(annotation_path),
                "target_render": str(matched_target_path),
                "source_annotation_z_indices": list(annotation_indices),
                "target_z_indices": list(selected),
                "matched_half_thickness_um": half_thickness_um,
                "source_resolution_um": source_resolution,
                "source_resolution_provenance": resolution_provenance,
                "source_z_step_um": source_z_um,
                "target_z_step_um": target_z_um,
                "purpose": (
                    "actual annotation canvas versus a physically matched "
                    "target maximum slab"
                ),
            }
        )

    manifest = {
        "tool": "prepare_canvas_offset_evidence.py",
        "transport": "vc_reader",
        "aws_credentials_file": (
            str(credentials_path) if credentials_path else None
        ),
        "case_dir": str(case_dir),
        "source_dataset_remote": dataset_remote,
        "source_zarr_remote": source_zarr_remote,
        "target_zarr_remote": target_zarr_remote,
        "source_center": source_center,
        "target_center": target_center,
        "target_matched_max": matched_target,
        "annotation_render": (
            str(annotation_path) if annotation_path else None
        ),
        "comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
