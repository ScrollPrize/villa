import aiohttp
import json
import re
import warnings
from pathlib import Path

import fsspec
import numpy as np
import tifffile
import zarr
from numba import njit
from vesuvius.image_proc.intensity.normalization import normalize_robust


def load_volume_auth(auth_json_path):
    if auth_json_path is None:
        return None, None

    with open(str(auth_json_path), "r", encoding="utf-8") as f:
        auth = json.load(f)
    return str(auth["username"]), str(auth["password"])


def flat_patch_cache_path(config):
    explicit_cache_path = config.get("patch_cache_filename")
    if explicit_cache_path not in (None, ""):
        return Path(str(explicit_cache_path))

    patch_size = config.get("patch_size", ())
    if isinstance(patch_size, int):
        patch_size_key = str(int(patch_size))
    else:
        patch_size_key = "x".join(str(int(v)) for v in patch_size)
    version_key = label_version_cache_token(config.get("label_version"))
    return Path(config.get("out_dir", ".")) / f"flat_ink_patches_ps-{patch_size_key}_labels-{version_key}.json"


def save_flat_patch_cache(path, patches):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "dataset_idx": int(patch.segment.dataset_idx),
                    "segment_relpath": str(patch.segment.segment_relpath),
                    "scale": patch.segment.scale,
                    "inklabels_path": str(patch.segment.inklabels),
                    "supervision_mask_path": str(patch.segment.supervision_mask),
                    "validation_mask_path": (
                        "" if getattr(patch.segment, "validation_mask", None) is None
                        else str(patch.segment.validation_mask)
                    ),
                    "active_supervision_mask_path": str(patch.supervision_mask),
                    "is_validation": bool(getattr(patch, "is_validation", False)),
                    "bbox": list(patch.bbox),
                }
                for patch in patches
            ],
            f,
        )


def load_flat_patch_cache(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def open_zarr(path, resolution, auth=None):
    path_str = str(path)
    user, password = load_volume_auth(auth)
    use_https_auth = path_str.startswith("https://") and bool(user) and bool(password)
    if use_https_auth:
        fs = fsspec.filesystem(
            "https",
            client_kwargs={"auth": aiohttp.BasicAuth(user, password)},
        )
        store = zarr.storage.FSStore(
            path_str.rstrip("/"),
            fs=fs,
            mode="r",
            check=False,
            create=False,
            exceptions=(KeyError, FileNotFoundError, PermissionError, OSError, aiohttp.ClientResponseError),
        )
        return zarr.open(store, path=str(resolution), mode="r")
    return zarr.open(path_str, path=str(resolution), mode="r")


_LABEL_ASSET_NAME_RE = re.compile(
    r"^(?P<prefix>.*)_(?P<label_kind>inklabels|supervision_mask|validation_mask)"
    r"(?:_v(?P<version_num>\d+))?(?P<extension>\.(?:tif|tiff|zarr))$",
    re.IGNORECASE,
)


def normalize_label_version(label_version):
    if label_version in (None, ""):
        return None
    if isinstance(label_version, str):
        value = label_version.strip().lower()
        if value in {"", "auto", "latest"}:
            return None
        if value in {"base", "unversioned", "v1"}:
            return 1
        if value.startswith("v") and value[1:].isdigit():
            version_num = int(value[1:])
            if version_num < 1:
                raise ValueError(f"label_version must be >= v1, got {label_version!r}")
            return version_num
        raise ValueError(
            f"label_version must be one of None/'auto', 'base', or 'vN', got {label_version!r}"
        )
    if isinstance(label_version, (int, np.integer)):
        version_num = int(label_version)
        if version_num < 1:
            raise ValueError(f"label_version must be >= 1, got {label_version!r}")
        return version_num
    raise ValueError(
        f"label_version must be None, a string like 'v2', or an integer, got {type(label_version).__name__}"
    )


def label_version_cache_token(label_version):
    version_num = normalize_label_version(label_version)
    if version_num is None:
        return "auto"
    if version_num <= 1:
        return "base"
    return f"v{version_num}"


def _split_path_dir_and_name(path):
    path_str = str(path).rstrip("/")
    last_sep = max(path_str.rfind("/"), path_str.rfind("\\"))
    if last_sep < 0:
        return "", path_str
    return path_str[: last_sep + 1], path_str[last_sep + 1 :]


def parse_label_asset_path(path):
    dir_prefix, name = _split_path_dir_and_name(path)
    match = _LABEL_ASSET_NAME_RE.match(name)
    if match is None:
        return None
    version_num_raw = match.group("version_num")
    version_num = 1 if version_num_raw is None else int(version_num_raw)
    return {
        "path": str(path),
        "dir_prefix": dir_prefix,
        "name": name,
        "prefix": match.group("prefix"),
        "label_kind": match.group("label_kind").lower(),
        "version_num": version_num,
        "extension": match.group("extension"),
    }


def build_matching_label_asset_path(path, *, label_kind):
    parsed = parse_label_asset_path(path)
    assert parsed is not None, f"Label path has unexpected format: {path}"
    version_suffix = "" if int(parsed["version_num"]) <= 1 else f"_v{int(parsed['version_num'])}"
    return (
        f"{parsed['dir_prefix']}{parsed['prefix']}_{str(label_kind)}"
        f"{version_suffix}{parsed['extension']}"
    )


def resolve_versioned_label_path(paths, *, label_kind, label_version=None, context="labels"):
    requested_version = normalize_label_version(label_version)
    candidates = {}
    for path in paths:
        parsed = parse_label_asset_path(path)
        if parsed is None or parsed["label_kind"] != str(label_kind):
            continue
        candidates[int(parsed["version_num"])] = str(path)

    assert candidates, f"{context} must contain at least one {label_kind} path."

    if requested_version is not None:
        resolved = candidates.get(int(requested_version))
        requested_name = "base" if int(requested_version) <= 1 else f"v{int(requested_version)}"
        assert resolved is not None, (
            f"{context} does not contain {label_kind} version {requested_name}."
        )
        return resolved

    return candidates[max(candidates)]


def resolve_segment_inklabel_path(segment, *, label_version=None):
    segment_uuid = str(segment.uuid)
    ink_label_paths = [
        str(label["path"])
        for label in segment.list_labels()
        if label.get("name") == "inklabels" and label.get("path") is not None
    ]
    return resolve_versioned_label_path(
        ink_label_paths,
        label_kind="inklabels",
        label_version=label_version,
        context=f"Segment {segment_uuid!r}",
    )


def resolve_local_label_pair_paths(segment_dir, segment_name, *, label_version=None, extension=".zarr"):
    inklabels, supervision_mask, _ = resolve_local_label_paths(
        segment_dir,
        segment_name,
        label_version=label_version,
        extension=extension,
    )
    return inklabels, supervision_mask


def resolve_local_label_paths(segment_dir, segment_name, *, label_version=None, extension=".zarr"):
    segment_dir = Path(segment_dir)
    normalized_extension = str(extension).lower()
    requested_version = normalize_label_version(label_version)
    candidates_by_version = {}

    for path in segment_dir.iterdir():
        parsed = parse_label_asset_path(path.name)
        if parsed is None:
            continue
        if parsed["prefix"] != str(segment_name):
            continue
        if parsed["extension"].lower() != normalized_extension:
            continue
        version_entry = candidates_by_version.setdefault(int(parsed["version_num"]), {})
        version_entry[parsed["label_kind"]] = path

    available_versions = sorted(
        version_num
        for version_num, record in candidates_by_version.items()
        if "inklabels" in record and "supervision_mask" in record
    )
    assert available_versions, (
        f"{segment_dir} must contain matching inklabels and supervision_mask {normalized_extension} assets."
    )

    if requested_version is None:
        chosen_version = available_versions[-1]
    else:
        chosen_version = int(requested_version)
        requested_name = "base" if chosen_version <= 1 else f"v{chosen_version}"
        assert chosen_version in available_versions, (
            f"{segment_dir} does not contain matching {normalized_extension} labels for version {requested_name}."
        )

    selected = candidates_by_version[chosen_version]
    return (
        selected["inklabels"],
        selected["supervision_mask"],
        selected.get("validation_mask"),
    )

def to_uint8_image(image_2d):
    image_2d = np.nan_to_num(np.asarray(image_2d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image_2d.min())
    max_value = float(image_2d.max())
    if max_value > min_value:
        image_2d = (image_2d - min_value) / (max_value - min_value)
    else:
        image_2d = np.zeros_like(image_2d, dtype=np.float32)
    return np.clip(np.rint(image_2d * 255.0), 0, 255).astype(np.uint8)

def to_uint8_label(label_2d, ignore_mask_2d=None):
    label_2d = np.asarray(label_2d, dtype=np.float32)
    label_vis = np.zeros(label_2d.shape, dtype=np.uint8)
    if ignore_mask_2d is not None:
        ignore_mask_2d = np.asarray(ignore_mask_2d, dtype=np.float32) > 0
        label_vis[ignore_mask_2d] = 127
    label_vis[label_2d == 0] = 0
    label_vis[label_2d > 0] = 255
    if ignore_mask_2d is not None:
        label_vis[ignore_mask_2d] = 127
    return label_vis

def to_uint8_probability(probability_2d, lower_percentile=1.0, upper_percentile=99.0):
    probability_2d = np.nan_to_num(np.asarray(probability_2d, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    probability_2d = np.clip(probability_2d, 0.0, 1.0)
    lo = float(np.percentile(probability_2d, lower_percentile))
    hi = float(np.percentile(probability_2d, upper_percentile))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        probability_2d = np.clip(probability_2d, lo, hi)
        probability_2d = (probability_2d - lo) / (hi - lo)
    return np.clip(np.rint(probability_2d * 255.0), 0, 255).astype(np.uint8)

def build_preview_montage(input_tiles, label_tiles, probability_tiles, gap_size=4):
    if not input_tiles:
        return None

    rows = []
    for input_tile, label_tile, probability_tile in zip(input_tiles, label_tiles, probability_tiles):
        column_gap = np.zeros((input_tile.shape[0], gap_size), dtype=np.uint8)
        rows.append(
            np.concatenate(
                [input_tile, column_gap, label_tile, column_gap, probability_tile],
                axis=1,
            )
        )

    row_gap = np.zeros((gap_size, rows[0].shape[1]), dtype=np.uint8)
    montage = []
    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            montage.append(row_gap)
        montage.append(row)
    return np.concatenate(montage, axis=0)

def save_val_preview_tif(output_path, input_tiles, label_tiles, probability_tiles, gap_size=4):
    montage = build_preview_montage(
        input_tiles,
        label_tiles,
        probability_tiles,
        gap_size=gap_size,
    )
    if montage is None:
        return
    tifffile.imwrite(output_path, montage, compression="lzw")

def _normalize_patch_size_zyx(patch_size):
    patch_size_zyx = np.asarray(patch_size, dtype=np.int32).reshape(-1)
    if patch_size_zyx.size == 1:
        patch_size_zyx = np.repeat(patch_size_zyx, 3)
    if patch_size_zyx.size != 3 or np.any(patch_size_zyx <= 0):
        raise ValueError(
            f"patch_size must be a positive int or [z, y, x], got {patch_size!r}"
        )
    return patch_size_zyx


def _get_segment_stored_grid(dataset, segment):
    segment_uuid = str(segment.uuid)
    cached = dataset._segment_grid_cache.get(segment_uuid)
    if cached is not None:
        return cached

    segment.use_stored_resolution()
    x_stored, y_stored, z_stored, valid_stored = segment[:, :]

    x_stored = np.asarray(x_stored, dtype=np.float32)
    y_stored = np.asarray(y_stored, dtype=np.float32)
    z_stored = np.asarray(z_stored, dtype=np.float32)
    valid_mask = np.asarray(valid_stored, dtype=bool)
    valid_mask &= np.isfinite(x_stored)
    valid_mask &= np.isfinite(y_stored)
    valid_mask &= np.isfinite(z_stored)

    cached = {
        "x": x_stored,
        "y": y_stored,
        "z": z_stored,
        "valid": valid_mask,
        "shape": (int(x_stored.shape[0]), int(x_stored.shape[1])),
    }
    dataset._segment_grid_cache[segment_uuid] = cached
    return cached


def _read_bbox_with_padding(volume, bbox, *, fill_value=0):
    z0, y0, x0, z1, y1, x1 = (int(v) for v in bbox)
    expected_shape = (z1 - z0, y1 - y0, x1 - x0)
    if any(size <= 0 for size in expected_shape):
        raise ValueError(f"bbox must define a positive crop, got {bbox!r}")

    volume_shape = tuple(int(v) for v in volume.shape[:3])
    src_starts = (
        max(0, z0),
        max(0, y0),
        max(0, x0),
    )
    src_stops = (
        min(volume_shape[0], z1),
        min(volume_shape[1], y1),
        min(volume_shape[2], x1),
    )

    dtype = np.asarray(volume[(slice(0, 1),) * 3]).dtype
    output = np.full(expected_shape, fill_value, dtype=dtype)

    if any(stop <= start for start, stop in zip(src_starts, src_stops)):
        return output, None

    crop = np.asarray(
        volume[
            src_starts[0]:src_stops[0],
            src_starts[1]:src_stops[1],
            src_starts[2]:src_stops[2],
        ]
    )
    dst_starts = (
        src_starts[0] - z0,
        src_starts[1] - y0,
        src_starts[2] - x0,
    )
    dst_stops = tuple(dst_start + size for dst_start, size in zip(dst_starts, crop.shape))
    dst_slices = tuple(slice(start, stop) for start, stop in zip(dst_starts, dst_stops))
    output[dst_slices] = crop
    return output, dst_slices

def _normalize_vectors_last_axis(vectors, eps=1e-6):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    good = np.isfinite(vectors).all(axis=-1, keepdims=True)
    good &= np.isfinite(norms) & (norms > float(eps))
    out = np.zeros_like(vectors, dtype=np.float32)
    good_mask = good[..., 0]
    out[good_mask] = vectors[good_mask] / norms[good_mask]
    return out, good_mask


def _reconcile_label_mask_shape(
    mask,
    expected_shape,
    *,
    segment_uuid,
    label_name,
    max_axis_mismatch=1,
):
    expected_shape = tuple(int(v) for v in expected_shape)
    actual_shape = tuple(int(v) for v in mask.shape[:2])
    if actual_shape == expected_shape:
        return mask

    axis_mismatches = tuple(
        int(actual - expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )
    if any(abs(mismatch) > int(max_axis_mismatch) for mismatch in axis_mismatches):
        raise AssertionError(
            f"Segment {segment_uuid!r} {label_name} label shape {actual_shape} does not match "
            f"tifxyz full-resolution shape {expected_shape}."
        )

    row_extent = min(actual_shape[0], expected_shape[0])
    col_extent = min(actual_shape[1], expected_shape[1])
    mask = np.asarray(mask[:row_extent, :col_extent])

    pad_rows = expected_shape[0] - row_extent
    pad_cols = expected_shape[1] - col_extent
    if pad_rows > 0 or pad_cols > 0:
        pad_width = [(0, pad_rows), (0, pad_cols)]
        pad_width.extend([(0, 0)] * max(0, mask.ndim - 2))
        mask = np.pad(mask, pad_width, mode="constant", constant_values=0)

    warnings.warn(
        f"Segment {segment_uuid!r} {label_name} label shape {actual_shape} adjusted to "
        f"{expected_shape} to absorb a <=1 pixel edge mismatch.",
        stacklevel=2,
    )
    return mask

def load_segment_label_masks(segment, shape, label_version=None):
    import cv2

    segment_uuid = str(segment.uuid)
    shape = tuple(int(v) for v in shape)

    def read_mask(path, label_name):
        assert path, f"Segment {segment_uuid!r} must contain {label_name} mask."
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert mask is not None, f"Segment {segment_uuid!r} failed to read {label_name} mask: {path}"
        if mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = _reconcile_label_mask_shape(
            mask,
            shape,
            segment_uuid=segment_uuid,
            label_name=label_name,
        )
        return np.asarray(mask > 0, dtype=bool)

    ink_label_path = resolve_segment_inklabel_path(
        segment,
        label_version=label_version,
    )
    supervision_path = build_matching_label_asset_path(
        ink_label_path,
        label_kind="supervision_mask",
    )

    ink_mask = read_mask(ink_label_path, "ink")
    supervision_mask = read_mask(supervision_path, "supervision")
    return ink_mask, np.asarray(supervision_mask & ~ink_mask, dtype=bool), ink_label_path

def _get_labels_and_mask(dataset, segment):
    segment_uuid = str(segment.uuid)
    cached = dataset._segment_labels_and_mask_cache.get(segment_uuid)
    if cached is not None:
        return cached

    shape = tuple(int(v) for v in segment.full_resolution_shape)
    ink_mask, supervision_mask, _ = load_segment_label_masks(
        segment,
        shape,
        label_version=getattr(dataset, "label_version", None),
    )
    out = (ink_mask, supervision_mask)
    dataset._segment_labels_and_mask_cache[segment_uuid] = out
    return out

def _sample_patch_supervision_grid(
    dataset,
    segment,
    min_corner,
    max_corner,
    extra_bbox_pad=0.0,
    stored_rowcol_bounds=None,
):
    from vesuvius.tifxyz import interpolate_at_points

    grid = _get_segment_stored_grid(dataset, segment)
    x_stored = grid["x"]
    y_stored = grid["y"]
    z_stored = grid["z"]
    valid_mask = grid["valid"]
    n_rows_stored, n_cols_stored = x_stored.shape
    crop_size_tuple = tuple(int(v) for v in dataset.patch_size_zyx)
    min_corner_f = np.asarray(min_corner, dtype=np.float32).reshape(3)
    max_corner_f = np.asarray(max_corner, dtype=np.float32).reshape(3)

    bbox_pad = max(float(dataset.surface_bbox_pad), float(extra_bbox_pad))
    interp_method = "catmull_rom"

    segment.use_stored_resolution()
    scale_y, scale_x = getattr(segment, "_scale", (1.0, 1.0))
    scale_y = float(scale_y) if np.isfinite(scale_y) and float(scale_y) > 0.0 else 1.0
    scale_x = float(scale_x) if np.isfinite(scale_x) and float(scale_x) > 0.0 else 1.0

    kernel_pad = 2 if interp_method in {"catmull_rom", "bspline"} else 1
    if stored_rowcol_bounds is None:
        expanded_min = min_corner_f - bbox_pad
        expanded_max = max_corner_f + bbox_pad
        in_bbox = (
            valid_mask
            & (z_stored >= expanded_min[0])
            & (z_stored < expanded_max[0])
            & (y_stored >= expanded_min[1])
            & (y_stored < expanded_max[1])
            & (x_stored >= expanded_min[2])
            & (x_stored < expanded_max[2])
        )
        rows, cols = np.where(in_bbox)
        assert rows.size > 0 and cols.size > 0, "patch must intersect stored tifxyz points"
        row_start = max(0, int(rows.min()) - kernel_pad)
        row_stop = min(n_rows_stored, int(rows.max()) + kernel_pad + 1)
        col_start = max(0, int(cols.min()) - kernel_pad)
        col_stop = min(n_cols_stored, int(cols.max()) + kernel_pad + 1)
    else:
        row_start, row_stop, col_start, col_stop = (
            int(stored_rowcol_bounds[0]),
            int(stored_rowcol_bounds[1]),
            int(stored_rowcol_bounds[2]),
            int(stored_rowcol_bounds[3]),
        )
        row_start = max(0, row_start - kernel_pad)
        row_stop = min(n_rows_stored, row_stop + kernel_pad)
        col_start = max(0, col_start - kernel_pad)
        col_stop = min(n_cols_stored, col_stop + kernel_pad)
        assert row_stop > row_start and col_stop > col_start, "stored_rowcol_bounds must define a non-empty slice"

    x_local = x_stored[row_start:row_stop, col_start:col_stop]
    y_local = y_stored[row_start:row_stop, col_start:col_stop]
    z_local = z_stored[row_start:row_stop, col_start:col_stop]
    valid_local = valid_mask[row_start:row_stop, col_start:col_stop]

    n_rows_local = row_stop - row_start
    n_cols_local = col_stop - col_start
    query_h = 1 if n_rows_local <= 1 else max(n_rows_local, int(round(n_rows_local / scale_y)))
    query_w = 1 if n_cols_local <= 1 else max(n_cols_local, int(round(n_cols_local / scale_x)))
    query_rows_global = np.linspace(row_start, row_stop - 1, query_h, dtype=np.float32)
    query_cols_global = np.linspace(col_start, col_stop - 1, query_w, dtype=np.float32)
    query_y_global, query_x_global = np.meshgrid(query_rows_global, query_cols_global, indexing="ij")
    query_y_local = query_y_global - float(row_start)
    query_x_local = query_x_global - float(col_start)

    x_int, y_int, z_int, int_valid = interpolate_at_points(
        x_local,
        y_local,
        z_local,
        valid_local,
        query_y_local,
        query_x_local,
        scale=(1.0, 1.0),
        method=interp_method,
        invalid_value=-1.0,
    )
    world_grid = np.stack([z_int, y_int, x_int], axis=-1).astype(np.float32, copy=False)
    valid_interp = np.asarray(int_valid, dtype=bool)
    valid_interp &= np.isfinite(world_grid).all(axis=-1)
    in_patch = (
        valid_interp
        & (world_grid[..., 0] >= min_corner_f[0])
        & (world_grid[..., 0] < max_corner_f[0])
        & (world_grid[..., 1] >= min_corner_f[1])
        & (world_grid[..., 1] < max_corner_f[1])
        & (world_grid[..., 2] >= min_corner_f[2])
        & (world_grid[..., 2] < max_corner_f[2])
    )

    normals_grid = _get_segment_normals_zyx(dataset, segment)
    nz_int, ny_int, nx_int, normals_valid = interpolate_at_points(
        normals_grid[row_start:row_stop, col_start:col_stop, 0],
        normals_grid[row_start:row_stop, col_start:col_stop, 1],
        normals_grid[row_start:row_stop, col_start:col_stop, 2],
        valid_local,
        query_y_local,
        query_x_local,
        scale=(1.0, 1.0),
        method=interp_method,
        invalid_value=np.nan,
    )
    normals_zyx = np.stack([nz_int, ny_int, nx_int], axis=-1).astype(np.float32, copy=False)
    normals_valid = np.asarray(normals_valid, dtype=bool)
    normals_zyx, normals_nonzero = _normalize_vectors_last_axis(normals_zyx)
    normals_valid &= normals_nonzero

    class_codes = np.full(query_y_global.shape, 100, dtype=np.uint8)
    ink_mask_full, supervision_mask_full = _get_labels_and_mask(dataset, segment)
    if ink_mask_full.size > 0:
        full_h, full_w = ink_mask_full.shape
        query_rows_full = query_y_global / scale_y
        query_cols_full = query_x_global / scale_x
        label_rows = np.rint(query_rows_full).astype(np.int64, copy=False)
        label_cols = np.rint(query_cols_full).astype(np.int64, copy=False)
        in_label_bounds = (
            (label_rows >= 0)
            & (label_rows < int(full_h))
            & (label_cols >= 0)
            & (label_cols < int(full_w))
        )
        if bool(np.any(in_label_bounds)):
            rows_in = label_rows[in_label_bounds]
            cols_in = label_cols[in_label_bounds]
            class_codes[in_label_bounds] = 100
            class_codes[in_label_bounds] = np.where(
                supervision_mask_full[rows_in, cols_in],
                0,
                class_codes[in_label_bounds],
            )
            class_codes[in_label_bounds] = np.where(
                ink_mask_full[rows_in, cols_in],
                1,
                class_codes[in_label_bounds],
            )

    local_grid = world_grid - min_corner_f.reshape(1, 1, 3)
    return {
        "local_grid": local_grid,
        "world_grid": world_grid,
        "valid_interp": valid_interp,
        "in_patch": in_patch,
        "class_codes": class_codes,
        "normals_zyx": normals_zyx,
        "normals_valid": normals_valid,
        "crop_size": crop_size_tuple,
    }


def _project_points_along_normals(
    dataset,
    points_world,
    normals_zyx,
    min_corner,
    crop_size,
    label_distance,
    require_points=False,
):
    crop_size_tuple = tuple(int(v) for v in crop_size)
    pos_distance = float(label_distance[0])
    neg_distance = float(label_distance[1])
    points_world = np.asarray(points_world, dtype=np.float32)
    normals_zyx = np.asarray(normals_zyx, dtype=np.float32)

    if require_points:
        assert points_world.shape[0] > 0, "points_world must contain at least one point"

    return _build_normal_offset_mask_from_labeled_points(
        points_world,
        normals_zyx,
        min_corner=min_corner,
        crop_size=crop_size_tuple,
        label_distance=(pos_distance, neg_distance),
        sample_step=float(dataset.normal_sample_step),
        trilinear_threshold=1e-4,
    )


def _project_label_from_sampled_grid(
    dataset,
    sampled_grid,
    min_corner,
    max_corner,
    crop_size,
    class_value,
    label_distance,
    require_points=False,
):
    pos_distance = float(label_distance[0])
    neg_distance = float(label_distance[1])
    point_mask = (
        sampled_grid["valid_interp"]
        & sampled_grid["normals_valid"]
        & (sampled_grid["class_codes"] == int(class_value))
    )
    points_world = sampled_grid["world_grid"][point_mask].astype(np.float32, copy=False)
    normals_zyx = sampled_grid["normals_zyx"][point_mask].astype(np.float32, copy=False)

    expand = max(pos_distance, neg_distance) + 1.0
    expanded_min = np.asarray(min_corner, dtype=np.float32) - expand
    expanded_max = np.asarray(max_corner, dtype=np.float32) + expand
    in_expanded = (
        (points_world[:, 0] >= expanded_min[0]) & (points_world[:, 0] < expanded_max[0]) &
        (points_world[:, 1] >= expanded_min[1]) & (points_world[:, 1] < expanded_max[1]) &
        (points_world[:, 2] >= expanded_min[2]) & (points_world[:, 2] < expanded_max[2])
    )

    return _project_points_along_normals(
        dataset,
        points_world[in_expanded],
        normals_zyx[in_expanded],
        min_corner=min_corner,
        crop_size=crop_size,
        label_distance=(pos_distance, neg_distance),
        require_points=require_points,
    )

@njit(cache=True)
def _splat_points_trilinear_numba(points, size_z, size_y, size_x):
    vox = np.zeros((size_z, size_y, size_x), dtype=np.float32)
    n_points = points.shape[0]
    for i in range(n_points):
        pz = points[i, 0]
        py = points[i, 1]
        px = points[i, 2]
        if not (np.isfinite(pz) and np.isfinite(py) and np.isfinite(px)):
            continue

        z0 = int(np.floor(pz))
        y0 = int(np.floor(py))
        x0 = int(np.floor(px))
        dz = pz - z0
        dy = py - y0
        dx = px - x0

        for oz in range(2):
            zi = z0 + oz
            if zi < 0 or zi >= size_z:
                continue
            wz = (1.0 - dz) if oz == 0 else dz
            if wz <= 0.0:
                continue
            for oy in range(2):
                yi = y0 + oy
                if yi < 0 or yi >= size_y:
                    continue
                wy = (1.0 - dy) if oy == 0 else dy
                if wy <= 0.0:
                    continue
                for ox in range(2):
                    xi = x0 + ox
                    if xi < 0 or xi >= size_x:
                        continue
                    wx = (1.0 - dx) if ox == 0 else dx
                    if wx <= 0.0:
                        continue
                    vox[zi, yi, xi] += wz * wy * wx
    return vox


def _points_to_voxels(points_local, crop_size, threshold=1e-4):
    crop_size_arr = np.asarray(crop_size, dtype=np.int64).reshape(3)
    crop_size_tuple = tuple(int(v) for v in crop_size_arr.tolist())
    points = np.asarray(points_local, dtype=np.float32)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    vox_accum = _splat_points_trilinear_numba(
        points,
        int(crop_size_arr[0]),
        int(crop_size_arr[1]),
        int(crop_size_arr[2]),
    )
    return (vox_accum > float(threshold)).astype(np.float32, copy=False)


def _estimate_surface_normals_zyx(x_grid, y_grid, z_grid, valid_mask, eps=1e-6):
    x = np.asarray(x_grid, dtype=np.float32)
    y = np.asarray(y_grid, dtype=np.float32)
    z = np.asarray(z_grid, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)

    p = np.stack([z, y, x], axis=-1).astype(np.float32, copy=False)
    p_prev_r = np.roll(p, 1, axis=0)
    p_next_r = np.roll(p, -1, axis=0)
    p_prev_c = np.roll(p, 1, axis=1)
    p_next_c = np.roll(p, -1, axis=1)

    v_prev_r = np.roll(valid, 1, axis=0)
    v_next_r = np.roll(valid, -1, axis=0)
    v_prev_c = np.roll(valid, 1, axis=1)
    v_next_c = np.roll(valid, -1, axis=1)
    v_prev_r[0, :] = False
    v_next_r[-1, :] = False
    v_prev_c[:, 0] = False
    v_next_c[:, -1] = False

    tangent_r = np.zeros_like(p, dtype=np.float32)
    tangent_c = np.zeros_like(p, dtype=np.float32)

    center_r = v_prev_r & v_next_r & valid
    forward_r = (~v_prev_r) & v_next_r & valid
    backward_r = v_prev_r & (~v_next_r) & valid
    tangent_r[center_r] = 0.5 * (p_next_r[center_r] - p_prev_r[center_r])
    tangent_r[forward_r] = p_next_r[forward_r] - p[forward_r]
    tangent_r[backward_r] = p[backward_r] - p_prev_r[backward_r]

    center_c = v_prev_c & v_next_c & valid
    forward_c = (~v_prev_c) & v_next_c & valid
    backward_c = v_prev_c & (~v_next_c) & valid
    tangent_c[center_c] = 0.5 * (p_next_c[center_c] - p_prev_c[center_c])
    tangent_c[forward_c] = p_next_c[forward_c] - p[forward_c]
    tangent_c[backward_c] = p[backward_c] - p_prev_c[backward_c]

    normals = np.cross(tangent_r, tangent_c).astype(np.float32, copy=False)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    good = valid & np.isfinite(norm[..., 0]) & (norm[..., 0] > float(eps))
    out = np.zeros_like(normals, dtype=np.float32)
    out[good] = normals[good] / norm[good]
    return out


def _get_segment_normals_zyx(dataset, segment):
    segment_uuid = str(segment.uuid)
    cached = dataset._segment_normal_cache.get(segment_uuid)
    if cached is not None:
        return cached

    grid = _get_segment_stored_grid(dataset, segment)
    normals = _estimate_surface_normals_zyx(
        grid["x"],
        grid["y"],
        grid["z"],
        grid["valid"],
    )
    dataset._segment_normal_cache[segment_uuid] = normals
    return normals


def _build_normal_offset_mask_from_labeled_points(
    points_world_zyx,
    normals_zyx,
    min_corner,
    crop_size,
    label_distance,
    sample_step=0.5,
    trilinear_threshold=1e-4,
):
    points = np.asarray(points_world_zyx, dtype=np.float32)
    normals = np.asarray(normals_zyx, dtype=np.float32)
    crop_size_arr = np.asarray(crop_size, dtype=np.int64).reshape(3)
    crop_size_tuple = tuple(int(v) for v in crop_size_arr.tolist())
    assert points.ndim == 2 and points.shape[1] == 3 and points.shape[0] > 0, (
        f"points_world_zyx must have shape [n, 3] with n > 0, got {points.shape!r}"
    )
    assert normals.ndim == 2 and normals.shape == points.shape, (
        f"normals_zyx must match points_world_zyx shape {points.shape!r}, got {normals.shape!r}"
    )

    pos_distance = float(label_distance[0])
    neg_distance = float(label_distance[1])
    pos_distance = max(0.0, pos_distance)
    neg_distance = max(0.0, neg_distance)

    sample_step = float(sample_step)

    n_norm = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (n_norm > 1e-6)
    assert bool(np.any(valid)), "points_world_zyx/normals_zyx must contain at least one finite point with a non-zero normal"

    points = points[valid]
    normals = normals[valid] / n_norm[valid, None]
    min_corner = np.asarray(min_corner, dtype=np.float32).reshape(1, 3)

    n_samples = max(2, int(np.ceil((pos_distance + neg_distance) / sample_step)) + 1)
    offsets = np.linspace(-neg_distance, pos_distance, num=n_samples, dtype=np.float32)
    sampled = points[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    local_points = sampled.reshape(-1, 3) - min_corner
    return _points_to_voxels(
        local_points,
        crop_size_tuple,
        threshold=trilinear_threshold,
    )

def _read_volume_crop_from_patch_dict(patch, crop_size, min_corner, max_corner):
    """Read a [z, y, x] crop from a patch dict and robust-normalize it."""
    volume = patch["volume"]
    if not hasattr(volume, "shape"):
        volume = volume[str(int(patch["scale"]))]

    crop_size = tuple(int(v) for v in crop_size)
    min_corner = np.asarray(min_corner, dtype=np.int64).reshape(3)
    max_corner = np.asarray(max_corner, dtype=np.int64).reshape(3)

    vol_crop = np.zeros(crop_size, dtype=volume.dtype)
    vol_shape = np.asarray(volume.shape, dtype=np.int64)
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, vol_shape)
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)

    if np.all(src_ends > src_starts):
        vol_crop[
            dst_starts[0]:dst_ends[0],
            dst_starts[1]:dst_ends[1],
            dst_starts[2]:dst_ends[2],
        ] = volume[
            src_starts[0]:src_ends[0],
            src_starts[1]:src_ends[1],
            src_starts[2]:src_ends[2],
        ]
    return normalize_robust(vol_crop)
