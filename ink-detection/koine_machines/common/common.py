import aiohttp
import json
from pathlib import Path

import fsspec
import numpy as np
import zarr

_FLAT_PATCH_FINDING_CACHE_VERSION = "v3"


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
    patch_finding_key = flat_patch_finding_cache_token(config)
    discovery_mode = str(config.get("patch_discovery_mode", "labeled")).strip().lower() or "labeled"
    return Path(config.get("out_dir", ".")) / (
        f"flat_ink_patches_dm-{discovery_mode}_pf-{patch_finding_key}_ps-{patch_size_key}_labels-{version_key}.json"
    )


def flat_patch_finding_cache_token(config):
    patch_finding_type = str(config.get("patch_finding_type", "default")).strip().lower() or "default"
    discovery_mode = str(config.get("patch_discovery_mode", "labeled")).strip().lower() or "labeled"
    patch_size = config.get("patch_size", ())
    if isinstance(patch_size, int):
        patch_dims = [int(patch_size)]
    else:
        patch_dims = [int(v) for v in patch_size]

    patch_size_y = patch_dims[1] if len(patch_dims) >= 2 else (patch_dims[0] if patch_dims else 0)
    default_stride = int(patch_size_y * float(config.get("patch_overlap", 0)))

    if patch_finding_type == "subtiling":
        tile_size = int(config.get("patch_finding_tile_size", patch_size_y))
        stride = int(config.get("patch_finding_stride", default_stride))
        filter_empty_tile = int(bool(config.get("patch_finding_filter_empty_tile", False)))
        return (
            f"{discovery_mode}-subtiling-{_FLAT_PATCH_FINDING_CACHE_VERSION}"
            f"-ts-{tile_size}_st-{stride}_fe-{filter_empty_tile}"
        )

    if discovery_mode == "unlabeled":
        return (
            f"unlabeled-default-{_FLAT_PATCH_FINDING_CACHE_VERSION}"
            f"-po-{config.get('patch_overlap', '')}"
            f"-mdc-{config.get('unlabeled_patch_min_data_coverage', 0.15)}"
        )
    return f"labeled-default-{_FLAT_PATCH_FINDING_CACHE_VERSION}-po-{config.get('patch_overlap', '')}"


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
                    "inklabels_path": (
                        "" if getattr(patch.segment, "inklabels", None) is None
                        else str(patch.segment.inklabels)
                    ),
                    "supervision_mask_path": (
                        "" if getattr(patch.segment, "supervision_mask", None) is None
                        else str(patch.segment.supervision_mask)
                    ),
                    "validation_mask_path": (
                        "" if getattr(patch.segment, "validation_mask", None) is None
                        else str(patch.segment.validation_mask)
                    ),
                    "active_supervision_mask_path": (
                        "" if patch.supervision_mask is None else str(patch.supervision_mask)
                    ),
                    "is_validation": bool(getattr(patch, "is_validation", False)),
                    "is_unlabeled": bool(getattr(patch, "is_unlabeled", False)),
                    "patch_finding_key": flat_patch_finding_cache_token(
                        getattr(patch.segment, "config", {})
                    ),
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

def label_version_cache_token(label_version):
    if label_version in (None, ""):
        return "auto"
    return str(label_version).strip()


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

    dtype = np.dtype(volume.dtype)
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
