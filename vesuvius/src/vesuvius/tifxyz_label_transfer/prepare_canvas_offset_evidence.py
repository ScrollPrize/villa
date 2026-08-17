#!/usr/bin/env python3
"""Prepare auditable canvas-offset CT evidence using rclone only.

The source annotation render is copied as one object when the ink dataset
ships ``*_max_<first>_<last>.tif``. Exact-center images and target comparison
slabs are extracted from surface-volume Zarrs with ``rclone cat``. Uncompressed
chunks use byte ranges; compressed chunks are fetched whole and decoded
locally. No HTTP or Hugging Face client is used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image
import tifffile

from vesuvius.utils.cli import HyphenUnderscoreParser


# The ink dataset is private: there is no usable default, every run must
# name the caller's own rclone mirror explicitly.
DEFAULT_INK_ROOT: Optional[str] = None
# The open-data bucket is public; this inline rclone remote reads it
# anonymously without requiring any local rclone configuration.
DEFAULT_OPEN_DATA_ROOT = (
    ":s3,provider=AWS,env_auth=false,region=us-east-1:"
    "vesuvius-challenge-open-data"
)
DEFAULT_AWS_CREDENTIALS_FILE: Optional[Path] = None
RCLONE_PROCESS_TIMEOUT_SECONDS = 10 * 60
ANNOTATION_COPY_TIMEOUT_SECONDS = 30 * 60


def _remote_join(root: str, *parts: str) -> str:
    return root.rstrip("/") + "/" + "/".join(
        part.strip("/") for part in parts if part
    )


def _run_rclone(
    arguments: list[str], timeout_seconds: int = RCLONE_PROCESS_TIMEOUT_SECONDS
) -> bytes:
    rclone_command = [
        "rclone",
        *arguments,
        "--contimeout",
        "10s",
        "--timeout",
        "60s",
        "--retries",
        "2",
    ]
    # A SIGKILL/OOM kill cannot execute Python cleanup.  On Linux, ask the
    # kernel to terminate rclone when this process dies so a failed Napari
    # extraction cannot continue downloading orphaned multi-gigabyte chunks.
    setpriv = (
        shutil.which("setpriv")
        if sys.platform.startswith("linux")
        else None
    )
    command = (
        [setpriv, "--pdeathsig", "TERM", *rclone_command]
        if setpriv is not None
        else rclone_command
    )
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"rclone timed out while running {arguments[0]!r}; check AWS "
            "credentials and object-store connectivity"
        ) from error
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"rclone failed ({completed.returncode}): "
            f"rclone {arguments[0]}: {message}"
        )
    return completed.stdout


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


def _cat_json(remote: str) -> dict[str, Any]:
    return json.loads(_run_rclone(["cat", remote]))


def _cat_range(remote: str, offset: int, count: int) -> bytes:
    return _run_rclone(
        [
            "cat",
            remote,
            "--offset",
            str(offset),
            "--count",
            str(count),
        ]
    )


def _list_files(remote: str, include: str) -> list[str]:
    output = _run_rclone(
        [
            "lsf",
            remote,
            "--max-depth",
            "1",
            "--files-only",
            "--include",
            include,
        ]
    )
    return sorted(
        line for line in output.decode("utf-8").splitlines() if line
    )


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


def inspect_zarr(remote: str, preferred_level: int) -> ZarrLevel:
    attrs = _cat_json(_remote_join(remote, ".zattrs"))
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
        metadata=_cat_json(_remote_join(remote, level, ".zarray")),
        level_zero_metadata=_cat_json(_remote_join(remote, "0", ".zarray")),
        scale_zyx=scale,
    )


def _chunk_key(
    info: ZarrLevel, z_chunk: int, y_chunk: int, x_chunk: int
) -> str:
    separator = info.metadata.get("dimension_separator", ".")
    indices = separator.join(
        str(value) for value in (z_chunk, y_chunk, x_chunk)
    )
    return _remote_join(info.remote, info.level, indices)


def _batched_range_payloads(
    info: ZarrLevel,
    z_chunk: int,
    byte_offset: int,
    byte_count: int,
    workers: int,
) -> list[tuple[int, int, bytes]]:
    """Read uncompressed chunks with one rclone process per chunk row."""

    separator = info.metadata.get("dimension_separator", ".")
    if separator == "/":
        root = _remote_join(info.remote, info.level, str(z_chunk))
        prefix = ""
    else:
        root = _remote_join(info.remote, info.level)
        prefix = f"{z_chunk}."
    listed = _run_rclone(
        ["lsf", root, "--recursive", "--files-only"]
    ).decode("utf-8").splitlines()
    rows: dict[int, list[tuple[str, int]]] = {}
    for relative in listed:
        if separator == "/":
            parts = relative.split("/")
            if len(parts) != 2:
                continue
            y_chunk, x_chunk = int(parts[0]), int(parts[1])
        else:
            if not relative.startswith(prefix):
                continue
            parts = relative.split(".")
            if len(parts) != 3 or int(parts[0]) != z_chunk:
                continue
            y_chunk, x_chunk = int(parts[1]), int(parts[2])
        rows.setdefault(y_chunk, []).append((relative, x_chunk))

    def fetch_row(
        item: tuple[int, list[tuple[str, int]]]
    ) -> list[tuple[int, int, bytes]]:
        y_chunk, files = item
        # rclone cat concatenates matching files in remote lexical order.
        files = sorted(files, key=lambda value: value[0])
        arguments = ["cat", root]
        for relative, _ in files:
            arguments.extend(["--include", f"/{relative}"])
        arguments.extend(
            [
                "--offset",
                str(byte_offset),
                "--count",
                str(byte_count),
            ]
        )
        combined = _run_rclone(arguments)
        expected = len(files) * byte_count
        if len(combined) != expected:
            raise ValueError(
                f"batched rclone row {y_chunk} returned "
                f"{len(combined)} bytes, expected {expected}"
            )
        return [
            (
                y_chunk,
                x_chunk,
                combined[index * byte_count : (index + 1) * byte_count],
            )
            for index, (_, x_chunk) in enumerate(files)
        ]

    payloads: list[tuple[int, int, bytes]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, max(len(rows), 1))
    ) as executor:
        for row_payloads in executor.map(fetch_row, sorted(rows.items())):
            payloads.extend(row_payloads)
    return payloads


def _bulk_copy_compressed_chunks(
    info: ZarrLevel,
    z_chunk: int,
    positions: Sequence[tuple[int, int]],
    destination: Path,
    workers: int,
) -> list[tuple[int, int, Path]]:
    """Fetch compressed chunks with one parallel rclone copy process."""

    level_remote = _remote_join(info.remote, info.level)
    listed = set(
        _run_rclone(
            ["lsf", level_remote, "--recursive", "--files-only"]
        )
        .decode("utf-8")
        .splitlines()
    )
    separator = info.metadata.get("dimension_separator", ".")
    wanted = {
        separator.join(str(value) for value in (z_chunk, y_chunk, x_chunk)): (
            y_chunk,
            x_chunk,
        )
        for y_chunk, x_chunk in positions
    }
    present = sorted(relative for relative in wanted if relative in listed)
    if not present:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    files_path = destination / "files-from.txt"
    files_path.write_text("\n".join(present) + "\n", encoding="utf-8")
    try:
        _run_rclone(
            [
                "copy",
                level_remote,
                str(destination),
                "--files-from",
                str(files_path),
                "--no-traverse",
                "--transfers",
                str(workers),
                "--checkers",
                str(workers),
            ],
            timeout_seconds=1800,
        )
    finally:
        files_path.unlink(missing_ok=True)
    return [
        (*wanted[relative], destination / relative) for relative in present
    ]


def extract_composite(
    info: ZarrLevel,
    z_indices: Sequence[int],
    output_path: Path,
    *,
    workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Extract a maximum composite while transferring only required bytes."""

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
    z_chunks = indices // chunks[0]
    if not np.all(z_chunks == z_chunks[0]):
        raise ValueError(
            "selected Z layers cross chunk boundaries; split extraction "
            "is not implemented"
        )
    local_z = indices % chunks[0]
    if not np.all(np.diff(local_z) == 1):
        raise ValueError("selected Z layers must be contiguous")
    if (
        len(shape) != 3
        or len(chunks) != 3
        or info.metadata.get("filters") is not None
        or info.metadata.get("order") != "C"
    ):
        raise ValueError(
            "requires an unfiltered, C-order 3D Zarr; "
            f"got {info.metadata}"
        )

    compressor = info.metadata.get("compressor")
    codec = None
    if compressor is not None:
        from numcodecs import get_codec

        codec = get_codec(compressor)
    slice_bytes = chunks[1] * chunks[2] * dtype.itemsize
    byte_offset = int(local_z[0]) * slice_bytes
    byte_count = len(indices) * slice_bytes
    full_chunk_bytes = math.prod(chunks) * dtype.itemsize
    y_chunks = math.ceil(shape[1] / chunks[1])
    x_chunks = math.ceil(shape[2] / chunks[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.partial.tif"
    )
    partial_output.unlink(missing_ok=True)
    # Paris4 level-2 composites exceed 200 MiB.  Back the output directly by
    # its final TIFF instead of briefly retaining another complete raster in
    # RAM while Napari may still be releasing the previous case.
    output = tifffile.memmap(
        partial_output,
        shape=(shape[1], shape[2]),
        dtype=dtype,
        photometric="minisblack",
        metadata=None,
    )
    output[...] = 0
    print(
        f"Extracting {info.remote} level {info.level} Z "
        f"{indices[0]}..{indices[-1]} -> {output_path}",
        flush=True,
    )

    def fetch_one(
        position: tuple[int, int]
    ) -> tuple[int, int, Optional[bytes]]:
        y_chunk, x_chunk = position
        remote = _chunk_key(
            info, int(z_chunks[0]), y_chunk, x_chunk
        )
        payload = _cat_range(remote, byte_offset, byte_count)
        # Sparse Zarrs omit all-fill chunks. rclone cat represents a missing
        # object as an empty result for this S3 backend.
        return y_chunk, x_chunk, payload or None

    positions = [
        (y_chunk, x_chunk)
        for y_chunk in range(y_chunks)
        for x_chunk in range(x_chunks)
    ]
    transferred = 0
    present_chunks = 0
    missing_chunks = 0
    compressed_cache: Optional[tempfile.TemporaryDirectory] = None
    if codec is not None:
        compressed_cache = tempfile.TemporaryDirectory(
            prefix=".tifxyz-compressed-chunks-",
            dir=output_path.parent,
        )
        fetched = _bulk_copy_compressed_chunks(
            info,
            int(z_chunks[0]),
            positions,
            Path(compressed_cache.name),
            workers,
        )
        missing_chunks = len(positions) - len(fetched)
    elif workers > 1:
        fetched = _batched_range_payloads(
            info, int(z_chunks[0]), byte_offset, byte_count, workers
        )
        missing_chunks = len(positions) - len(fetched)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            fetched = list(executor.map(fetch_one, positions))
    try:
        for y_chunk, x_chunk, payload_value in fetched:
            payload = (
                payload_value.read_bytes()
                if isinstance(payload_value, Path)
                else payload_value
            )
            if payload is None:
                missing_chunks += 1
                continue
            present_chunks += 1
            transferred += len(payload)
            if codec is None:
                expected = byte_count
                if len(payload) != expected:
                    raise ValueError(
                        f"short range for chunk {y_chunk}/{x_chunk}: "
                        f"{len(payload)} != {expected}"
                    )
                layers = np.frombuffer(payload, dtype=dtype).reshape(
                    len(indices), chunks[1], chunks[2]
                )
            else:
                decoded = codec.decode(payload)
                if len(decoded) != full_chunk_bytes:
                    raise ValueError(
                        f"decoded chunk {y_chunk}/{x_chunk} has "
                        f"{len(decoded)} bytes, expected {full_chunk_bytes}"
                    )
                layers = np.frombuffer(decoded, dtype=dtype).reshape(
                    chunks
                )[int(local_z[0]) : int(local_z[-1]) + 1]
            block = layers.max(axis=0)
            row = y_chunk * chunks[1]
            col = x_chunk * chunks[2]
            row_end = min(row + chunks[1], shape[1])
            col_end = min(col + chunks[2], shape[2])
            output[row:row_end, col:col_end] = block[
                : row_end - row, : col_end - col
            ]
        output.flush()
    except BaseException:
        del output
        partial_output.unlink(missing_ok=True)
        raise
    finally:
        if compressed_cache is not None:
            compressed_cache.cleanup()

    del output
    partial_output.replace(output_path)
    audit = {
        "transport": "rclone",
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
        "present_chunks": present_chunks,
        "missing_chunks": missing_chunks,
        "whole_compressed_chunks": codec is not None,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Cached {output_path} ({transferred / 1024**2:.1f} MiB via rclone)",
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
    candidates = _list_files(dataset_remote, "*_max_*_*.tif")
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
            _run_rclone(
                [
                    "copyto",
                    _remote_join(dataset_remote, name),
                    str(partial),
                ],
                timeout_seconds=ANNOTATION_COPY_TIMEOUT_SECONDS,
            )
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
    with tifffile.TiffFile(source) as handle:
        page = handle.pages[0]
        source_shape = tuple(int(value) for value in page.shape)
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
        destination = tifffile.memmap(
            output, shape=shape_yx, dtype=page.dtype, metadata=None
        )
        with tifffile.TiffFile(source) as handle:
            for segment, indices, _ in handle.pages[0].segments():
                row = int(indices[2])
                col = int(indices[3])
                block = np.squeeze(segment)
                block = block[
                    : min(block.shape[0], source_shape[0] - row),
                    : min(block.shape[1], source_shape[1] - col),
                ]
                usable_y = block.shape[0] // factor_y * factor_y
                usable_x = block.shape[1] // factor_x * factor_x
                if usable_y == 0 or usable_x == 0:
                    continue
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
                    col // factor_x : col // factor_x + reduced.shape[1],
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
        "--ink-rclone-root",
        default=DEFAULT_INK_ROOT,
        help=(
            "rclone remote:path of your mirror of the private ink dataset "
            "(required, e.g. myremote:bucket/datasets/ink/ink_YYYYMM)"
        ),
    )
    parser.add_argument(
        "--open-data-rclone-root",
        default=DEFAULT_OPEN_DATA_ROOT,
        help=(
            "rclone remote:path of the public Vesuvius open-data bucket "
            "(default: anonymous inline S3 remote, no rclone config needed)"
        ),
    )
    parser.add_argument(
        "--aws-credentials-file",
        type=Path,
        default=DEFAULT_AWS_CREDENTIALS_FILE,
        help=(
            "optional shell-format AWS exports loaded only when "
            "AWS_ACCESS_KEY_ID is absent (default: none; authenticate via "
            "the environment or your rclone config)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ink_rclone_root:
        raise SystemExit(
            "error: --ink-rclone-root is required; the ink dataset is "
            "private, so point it at your own rclone mirror "
            "(e.g. myremote:bucket/datasets/ink/ink_YYYYMM)"
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
        selection, args.ink_rclone_root
    )
    source_zarr_name = Path(
        selection["source_surface_zarrs"][0]["path"]
    ).name
    source_zarr_remote = _remote_join(dataset_remote, source_zarr_name)
    target_zarr_remote = _target_zarr_remote(
        selection, case_dir, args.open_data_rclone_root
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
        "transport": "rclone",
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
