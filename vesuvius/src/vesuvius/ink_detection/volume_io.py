"""Local/remote Zarr opening, resolution selection, padding, and disk caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import aiohttp
import numpy as np
import zarr

from vesuvius.data.utils import open_zarr as open_vesuvius_zarr


_PUBLIC_S3_VOLUME_SUBSTRING = "vesuvius-challenge-open-data"
ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _cache_snapshot(cache_dir: Path) -> list[tuple[int, int, Path]]:
    snapshot = []
    for directory, _, filenames in os.walk(cache_dir):
        for filename in filenames:
            if filename.endswith(".partial"):
                continue
            path = Path(directory) / filename
            try:
                stat = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not path.is_file():
                continue
            snapshot.append((stat.st_mtime_ns, stat.st_size, path))
    snapshot.sort()
    return snapshot


def _evict_to_watermark(
    snapshot: list[tuple[int, int, Path]], max_bytes: int
) -> int:
    total = sum(size for _, size, _ in snapshot)
    target_bytes = 0.9 * max_bytes
    for _, size, path in snapshot:
        if total <= target_bytes:
            break
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        total -= size
    return total


def load_volume_auth(auth_json_path: str | Path | None) -> tuple[str, str] | None:
    """Read the exact username/password JSON boundary used for HTTPS volumes."""
    if auth_json_path is None:
        return None
    with Path(auth_json_path).open("r", encoding="utf-8") as stream:
        authored = json.load(stream)
    if not isinstance(authored, dict) or "username" not in authored or "password" not in authored:
        raise ValueError("volume auth JSON requires username and password")
    return str(authored["username"]), str(authored["password"])


def disk_cache_subdir(source_path: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(source_path).encode()).hexdigest()[:12]
    return Path(cache_dir) / digest


def _available_top_level_keys(root: Any) -> tuple[str, ...]:
    if not hasattr(root, "keys"):
        return ()
    return tuple(sorted(str(key) for key in root.keys()))


def _missing_node_error(message: str) -> Exception:
    error_type = getattr(zarr.errors, "NodeNotFoundError", None)
    if error_type is None:
        error_type = getattr(zarr.errors, "PathNotFoundError", KeyError)
    return error_type(message)


def open_volume_root(
    path: str | Path,
    auth_json_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_max_gb: float | None = None,
):
    """Open a Zarr root with process-owned remote transport and optional cache."""
    path_text = str(path)
    storage_options: dict[str, Any] = {}
    is_public_s3 = (
        path_text.startswith("s3://")
        and _PUBLIC_S3_VOLUME_SUBSTRING in path_text
    )
    if is_public_s3:
        storage_options["anon"] = True
    auth = load_volume_auth(auth_json_path)
    if not is_public_s3 and path_text.startswith("https://") and auth is not None:
        storage_options["client_kwargs"] = {
            "auth": aiohttp.BasicAuth(auth[0], auth[1])
        }
    is_remote = path_text.startswith(("s3://", "http://", "https://"))

    if cache_dir is not None:
        if not ZARR_V3:
            raise NotImplementedError(
                "volume disk cache requires zarr 3; "
                f"installed zarr is {zarr.__version__}"
            )
        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import LocalStore

        maximum_bytes = (
            None if cache_max_gb is None else int(float(cache_max_gb) * 1e9)
        )
        if maximum_bytes is not None and maximum_bytes < 0:
            raise ValueError("cache_max_gb must be nonnegative or None")
        cache_path = disk_cache_subdir(path_text, Path(cache_dir))
        cache_path.mkdir(parents=True, exist_ok=True)
        if maximum_bytes is not None:
            snapshot = _cache_snapshot(cache_path)
            if sum(size for _, size, _ in snapshot) > maximum_bytes:
                _evict_to_watermark(snapshot, maximum_bytes)
        if is_remote:
            remote_options = dict(storage_options)
            remote_options["skip_instance_cache"] = True
            source_store = zarr.storage.FsspecStore.from_url(
                path_text,
                storage_options=remote_options,
                read_only=True,
            )
        else:
            source_store = LocalStore(path_text, read_only=True)
        store = CacheStore(
            store=source_store,
            cache_store=LocalStore(cache_path),
            max_size=maximum_bytes,
        )
        return zarr.open(store=store, mode="r")

    if is_remote and ZARR_V3:
        storage_options["skip_instance_cache"] = True
        store = zarr.storage.FsspecStore.from_url(
            path_text,
            storage_options=storage_options,
            read_only=True,
        )
        return zarr.open(store=store, mode="r")

    return open_vesuvius_zarr(
        path_text, mode="r", storage_options=storage_options
    )


def select_volume_level(
    root: Any,
    resolution: int | str,
    *,
    source: str,
    root_array_is_requested_level: bool = False,
) -> Any:
    """Select one resolution from an already opened array or group root."""

    if hasattr(root, "shape"):
        if not root_array_is_requested_level and str(resolution) not in {"0", ""}:
            raise _missing_node_error(
                f"{source.rstrip('/')}/{resolution} (resolution {str(resolution)!r} "
                f"in zarr array {source!r})"
            )
        return root
    try:
        return root[str(resolution)]
    except KeyError as exc:
        message = (
            f"{source.rstrip('/')}/{resolution} (resolution {str(resolution)!r} "
            f"in zarr store {source!r})"
        )
        try:
            available = _available_top_level_keys(root)
        except Exception:
            available = ()
        if available:
            message += "; available top-level keys: " + ", ".join(available[:20])
        raise _missing_node_error(message) from exc


def open_volume(
    path: str | Path,
    resolution: int | str,
    auth_json_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_max_gb: float | None = None,
    root_array_is_requested_level: bool = False,
):
    """Open one Zarr pyramid level through the shared root boundary."""

    root = open_volume_root(
        path,
        auth_json_path,
        cache_dir=cache_dir,
        cache_max_gb=cache_max_gb,
    )
    return select_volume_level(
        root,
        resolution,
        source=str(path),
        root_array_is_requested_level=root_array_is_requested_level,
    )


def read_bbox_with_padding(
    volume: Any,
    bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    fill_value: int | float = 0,
) -> tuple[np.ndarray, tuple[slice, slice, slice] | None]:
    """Read a positive ZYX bbox, padding only outside the array bounds."""
    z0, y0, x0, z1, y1, x1 = (int(value) for value in bbox_zyx)
    expected_shape = z1 - z0, y1 - y0, x1 - x0
    if any(size <= 0 for size in expected_shape):
        raise ValueError(f"bbox must define a positive crop, got {bbox_zyx!r}")
    shape = tuple(int(value) for value in volume.shape[:3])
    starts = max(0, z0), max(0, y0), max(0, x0)
    stops = min(shape[0], z1), min(shape[1], y1), min(shape[2], x1)
    output = np.full(expected_shape, fill_value, dtype=np.dtype(volume.dtype))
    if any(stop <= start for start, stop in zip(starts, stops)):
        return output, None
    crop = np.asarray(
        volume[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ]
    )
    destination_starts = starts[0] - z0, starts[1] - y0, starts[2] - x0
    destination = tuple(
        slice(start, start + size)
        for start, size in zip(destination_starts, crop.shape)
    )
    output[destination] = crop
    return output, destination


# `prepare_9um_isotropic_input` stamps `format`, `source` and `source_level` on every
# input it writes, unconditionally: the tag names the recipe, and `source`/`source_level`
# say which store the input came from and at which pyramid level. Nothing downstream
# currently reads any of it.
#
# The tag does not name a single scale. The shipped recipe is trained on two documented
# representation families: 2.399 um renders pooled at level 2, and native 9.362 um
# renders used at level 0. Both are legitimate inputs, so both are accepted.
FLAT_INPUT_RECIPES = {
    # format tag -> (in-plane micrometres the recipe trains at, level the tag names)
    "level2-zmean4-21slice-v1": ((9.596, 9.362), "2"),
}
FLAT_INPUT_SCALE_TOLERANCE = 0.02


class InputScaleRefused(ValueError):
    """Raised by `check_flat_input_scale(strict=True)`; a folder run skips and records it."""

# Mirrors the order `Volume::normalize()` uses when a store records its scale outside
# OME metadata: a numeric `voxelsize`, then the alternative spellings, then the beamline
# `samplePixelSize`, which is recorded in millimetres.
_VOXEL_SIZE_KEYS = (
    "voxelsize",
    "voxel_size_um",
    "voxelSizeUm",
    "pixel_size_um",
    "pixelSizeUm",
    "resolution_um",
)
_NESTED_METADATA_KEYS = ("scan", "volume", "properties", "metadata")
_SAMPLE_PIXEL_SIZE_PATH = ("scan", "tomo", "acquisition", "detector")


def _positive_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _number_from_object(obj, keys):
    if not isinstance(obj, dict):
        return None
    for key in keys:
        found = _positive_number(obj.get(key))
        if found is not None:
            return found
    return None


def inplane_um_from_ome(attrs, level):
    """In-plane micrometres for one pyramid level of an OME-Zarr store.

    Rendered surface volumes record this directly - axes in micrometre and a scale per
    dataset - so the level an input was built from names its own in-plane sampling.
    """

    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return None
    entry = multiscales[0]
    if not isinstance(entry, dict):
        return None
    axes = entry.get("axes")
    if not isinstance(axes, list):
        return None
    names = [str(axis.get("name")) for axis in axes if isinstance(axis, dict)]
    units = {
        str(axis.get("name")): str(axis.get("unit", ""))
        for axis in axes
        if isinstance(axis, dict)
    }
    if any(units.get(name) not in ("micrometer", "micrometre") for name in ("y", "x")):
        return None
    for dataset in entry.get("datasets") or ():
        if not isinstance(dataset, dict) or str(dataset.get("path")) != str(level):
            continue
        for transform in dataset.get("coordinateTransformations") or ():
            if not isinstance(transform, dict) or transform.get("type") != "scale":
                continue
            scale = transform.get("scale")
            if not isinstance(scale, list) or len(scale) != len(names):
                continue
            values = [
                _positive_number(scale[index])
                for index, name in enumerate(names)
                if name in ("y", "x")
            ]
            if len(values) == 2 and all(values) and values[0] == values[1]:
                return values[0]
    return None


def voxel_size_um_from_document(document):
    """Scale from a store's metadata document, in the order `Volume::normalize()` uses."""

    if not isinstance(document, dict):
        return None
    merged = dict(document)
    scan = merged.get("scan")
    if isinstance(scan, dict):
        merged.update(scan)
    found = _number_from_object(merged, _VOXEL_SIZE_KEYS)
    if found is not None:
        return found
    for key in _NESTED_METADATA_KEYS:
        found = _number_from_object(merged.get(key), _VOXEL_SIZE_KEYS)
        if found is not None:
            return found
    current = document
    for key in _SAMPLE_PIXEL_SIZE_PATH:
        if not isinstance(current, dict) or key not in current:
            current = None
            break
        current = current[key]
    millimetres = _number_from_object(current, ("samplePixelSize",))
    return None if millimetres is None else millimetres * 1000.0


def resolve_source_inplane_um(source, level, *, attrs_reader=None):
    """In-plane micrometres of `source` at `level`, with the reason when unknown.

    Only the store's metadata is read, never its arrays. "unreachable" is kept distinct
    from "records no scale" so a source that cannot be opened is never reported as a
    scale error.
    """

    if not source:
        return None, "no source recorded"
    reader = attrs_reader
    if reader is None:

        def reader(path):
            return dict(zarr.open(path, mode="r").attrs)

    try:
        attrs = reader(source)
    except Exception as exc:  # unreachable source, stale path, no permission
        return None, f"source unreachable ({type(exc).__name__})"
    if not isinstance(attrs, Mapping):
        return None, "source metadata unreadable"
    found = inplane_um_from_ome(attrs, level)
    if found is None:
        found = voxel_size_um_from_document(dict(attrs))
        if found is not None and str(level).isdigit():
            found *= 2 ** int(level)
    if found is None:
        return None, "source records no scale"
    return found, "ok"


def describe_flat_input_scale_mismatch(root, *, source, attrs_reader=None):
    """Compare a prepared input's actual in-plane scale with what its recipe trains at.

    Returns a description, or None when the input is consistent or carries no recipe tag
    to check. Inputs without a known `format` tag - published surface volumes, or inputs
    prepared by other means - are left alone.
    """

    attrs = getattr(root, "attrs", None)
    if attrs is None:
        return None
    try:
        recorded = dict(attrs)
    except Exception:
        return None

    tag = recorded.get("format")
    recipe = None if tag is None else FLAT_INPUT_RECIPES.get(str(tag))
    if recipe is None:
        return None
    trained_scales, expected_level = recipe

    source_level = recorded.get("source_level")
    if source_level is None:
        return (
            f"{source}: input declares format {str(tag)!r}, which is prepared from "
            f"pyramid level {expected_level}, but records no source_level, so the scale "
            "it was built at cannot be checked"
        )
    recorded_level = str(source_level)
    recorded_source = str(recorded.get("source") or "")
    actual_um, reason = resolve_source_inplane_um(
        recorded_source, recorded_level, attrs_reader=attrs_reader
    )
    if actual_um is not None:
        nearest = min(trained_scales, key=lambda um: abs(actual_um - um) / um)
        if abs(actual_um - nearest) / nearest <= FLAT_INPUT_SCALE_TOLERANCE:
            return None
        trained = " or ".join(f"{um:g}" for um in trained_scales)
        return (
            f"{source}: input is {actual_um:g} um in-plane; this recipe trains at "
            f"{trained} um (nearest {nearest:g}, factor {actual_um / nearest:.3g}). "
            f"Prepared from {recorded_source} at level {recorded_level}."
        )

    # The source could not be consulted. The tag and the recorded level still have to
    # agree with each other, and that comparison needs nothing but the input itself.
    if recorded_level == expected_level:
        return None
    message = (
        f"{source}: input declares format {str(tag)!r}, which is prepared from pyramid "
        f"level {expected_level}, but records source_level {recorded_level!r}"
    )
    if recorded_level.isdigit():
        ratio = 2.0 ** (int(expected_level) - int(recorded_level))
        message += (
            f"; its in-plane sampling is {ratio:g}x finer than that format implies"
            if ratio > 1
            else f"; its in-plane sampling is {1 / ratio:g}x coarser than that format implies"
        )
    return (
        f"{message} (absolute scale not checked: {reason}; a native "
        f"{min(trained_scales):g} um render prepared at level 0 is legitimate and would "
        "be reported here)"
    )


def check_flat_input_scale(root, *, source, strict=False, attrs_reader=None):
    """Report a prepared input whose scale does not match its recipe.

    Reporting rather than refusing is the default: an input prepared by other means is a
    legitimate thing to hand this code, and a message carrying the numbers is enough to
    act on. `strict` turns the same finding into a refusal.
    """

    message = describe_flat_input_scale_mismatch(
        root, source=source, attrs_reader=attrs_reader
    )
    if message is not None and strict:
        raise InputScaleRefused(
            message
            + " Re-prepare the input at the level this recipe expects, or drop "
            "--strict-input-scale to continue with a warning."
        )
    return message
