import os
import os.path as osp
import re
from functools import lru_cache

from train_resnet3d_lib.config import CFG


_IMAGE_EXTENSIONS = (".png", ".tiff", ".tif")


def _looks_like_zarr_store(path: str) -> bool:
    if not osp.exists(path):
        return False
    if osp.isfile(path):
        return path.endswith(".zarr")
    if osp.isdir(path):
        if osp.exists(osp.join(path, ".zarray")):
            return True
        if osp.exists(osp.join(path, ".zgroup")):
            return True
        if osp.exists(osp.join(path, "0", ".zarray")):
            return True
    return False


def _normalize_layout_token(value):
    token = osp.basename(str(value or "").strip())
    token = re.sub(r"[^0-9a-z]+", "", token.lower())
    if not token:
        return ""
    match = re.match(r"0*([0-9]+)(.*)$", token)
    if match:
        token = f"{int(match.group(1))}{match.group(2)}"
    return token


def _name_match_aliases(value):
    raw = osp.basename(str(value or "").strip())
    if not raw:
        return set()

    if raw.lower().endswith(".zarr"):
        stem = raw[:-5]
    else:
        stem = osp.splitext(raw)[0]

    pending = [stem]
    aliases = set()
    while pending:
        current = pending.pop()
        if not current or current in aliases:
            continue
        aliases.add(current)

        stripped = re.sub(r"_(?:inklabels|mask|supervision_mask)(?:_val(?:_\d+)?)?$", "", current, flags=re.IGNORECASE)
        if stripped != current:
            pending.append(stripped)

        stripped = re.sub(r"_max_[0-9_]+$", "", current, flags=re.IGNORECASE)
        if stripped != current:
            pending.append(stripped)

        stripped = re.sub(r"^\d{8,}[-_]+", "", current)
        if stripped != current:
            pending.append(stripped)

        stripped = re.sub(r"_\d+um$", "", current, flags=re.IGNORECASE)
        if stripped != current:
            pending.append(stripped)

    normalized = {_normalize_layout_token(alias) for alias in aliases}
    return {alias for alias in aliases | normalized if alias}


def _request_aliases(fragment_id, original_path):
    aliases = set(_name_match_aliases(fragment_id))
    aliases.update(_name_match_aliases(original_path))
    return aliases


def _path_aliases(path):
    aliases = set(_name_match_aliases(osp.basename(path)))
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                aliases.update(_name_match_aliases(entry.name))
    except FileNotFoundError:
        pass
    return aliases


def _score_alias_match(request_aliases, candidate_aliases, base_aliases):
    if request_aliases & base_aliases:
        return 300
    if request_aliases & candidate_aliases:
        return 200

    best = 0
    for request_alias in request_aliases:
        if len(request_alias) < 8:
            continue
        for candidate_alias in base_aliases:
            if request_alias in candidate_alias or candidate_alias in request_alias:
                best = max(best, 100 + min(len(request_alias), len(candidate_alias)))
        for candidate_alias in candidate_aliases:
            if request_alias in candidate_alias or candidate_alias in request_alias:
                best = max(best, 50 + min(len(request_alias), len(candidate_alias)))
    return best


def _pick_best_path(request_aliases, candidates, *, context):
    best_score = 0
    best_paths = []
    for candidate in sorted(set(candidates)):
        base_aliases = _name_match_aliases(osp.basename(candidate))
        candidate_aliases = _path_aliases(candidate) if osp.isdir(candidate) else _name_match_aliases(candidate)
        score = _score_alias_match(request_aliases, candidate_aliases, base_aliases)
        if score > best_score:
            best_score = score
            best_paths = [candidate]
        elif score == best_score and score > 0:
            best_paths.append(candidate)

    if best_score <= 0:
        return None
    if len(best_paths) == 1:
        return best_paths[0]
    raise ValueError(f"Ambiguous {context}: matched multiple candidates {best_paths}")


def _is_segment_dir(root, dirs, files):
    if "layers" in dirs:
        return True
    lower_files = {name.lower() for name in files}
    if {"x.tif", "y.tif", "z.tif"}.issubset(lower_files):
        return True
    if any(name.lower().endswith(".zarr") for name in dirs):
        return True
    return any(
        lower_name.endswith(_IMAGE_EXTENSIONS) and any(tag in lower_name for tag in ("_inklabels", "_mask", "_supervision_mask"))
        for lower_name in lower_files
    )


@lru_cache(maxsize=32)
def _scan_segment_dirs(dataset_root):
    dataset_root = osp.normpath(str(dataset_root))
    segment_dirs = []
    for root, dirs, files in os.walk(dataset_root, topdown=True):
        rel = osp.relpath(root, dataset_root)
        rel_parts = [] if rel == "." else rel.split(os.sep)
        if "unused" in rel_parts:
            dirs[:] = []
            continue
        if root != dataset_root and _is_segment_dir(root, dirs, files):
            segment_dirs.append(root)
            dirs[:] = []
    return tuple(sorted(set(segment_dirs)))


@lru_cache(maxsize=128)
def _resolve_group_dirs(dataset_root, base_path):
    target = _normalize_layout_token(base_path)
    if not target or not osp.isdir(dataset_root):
        return tuple()

    matches = []
    with os.scandir(dataset_root) as entries:
        for entry in entries:
            if entry.is_dir() and _normalize_layout_token(entry.name) == target:
                matches.append(entry.path)
    return tuple(sorted(matches))


@lru_cache(maxsize=512)
def _resolve_segment_dir_cached(dataset_root, fragment_id, base_path, original_path):
    fragment_id = str(fragment_id).strip()
    if not fragment_id:
        raise ValueError("segment id must be a non-empty string")

    direct = osp.normpath(osp.join(dataset_root, fragment_id))
    if osp.isdir(direct):
        return direct

    original_basename = osp.basename(str(original_path or "").rstrip("/"))
    if original_basename:
        direct = osp.normpath(osp.join(dataset_root, original_basename))
        if osp.isdir(direct):
            return direct

    request_aliases = _request_aliases(fragment_id, original_path)
    group_dirs = _resolve_group_dirs(dataset_root, base_path)
    if group_dirs:
        group_segment_dirs = []
        for group_dir in group_dirs:
            direct = osp.join(group_dir, fragment_id)
            if osp.isdir(direct):
                return direct
            if original_basename:
                direct = osp.join(group_dir, original_basename)
                if osp.isdir(direct):
                    return direct
            prefix = group_dir + os.sep
            group_segment_dirs.extend(path for path in _scan_segment_dirs(dataset_root) if path.startswith(prefix))

        group_segment_dirs = sorted(set(group_segment_dirs))
        if len(group_segment_dirs) == 1:
            return group_segment_dirs[0]

        picked = _pick_best_path(
            request_aliases,
            group_segment_dirs,
            context=f"segment directory for {fragment_id!r} under base_path={base_path!r}",
        )
        if picked is not None:
            return picked

    picked = _pick_best_path(
        request_aliases,
        _scan_segment_dirs(dataset_root),
        context=f"segment directory for {fragment_id!r}",
    )
    if picked is not None:
        return picked

    raise FileNotFoundError(
        f"Could not resolve segment directory for {fragment_id!r} under dataset_root={dataset_root!r}."
    )


def resolve_segment_dir(fragment_id, *, seg_meta=None):
    seg_meta = dict(seg_meta or {})
    dataset_root = osp.normpath(str(getattr(CFG, "dataset_root", "train_scrolls")))
    return _resolve_segment_dir_cached(
        dataset_root,
        str(fragment_id),
        str(seg_meta.get("base_path", "") or ""),
        str(seg_meta.get("original_path", "") or ""),
    )


def resolve_segment_layers_dir(fragment_id, *, seg_meta=None):
    segment_dir = resolve_segment_dir(fragment_id, seg_meta=seg_meta)
    layers_dir = osp.join(segment_dir, "layers")
    if osp.isdir(layers_dir):
        return layers_dir
    raise FileNotFoundError(f"Could not resolve layers directory for {fragment_id!r}: {layers_dir!r}")


def _resolve_named_image_path(fragment_id, *, seg_meta=None, artifact_names, suffix=""):
    segment_dir = resolve_segment_dir(fragment_id, seg_meta=seg_meta)
    original_path = str((seg_meta or {}).get("original_path", "") or "")
    request_aliases = _request_aliases(fragment_id, original_path)

    prefix_candidates = [str(fragment_id), osp.basename(segment_dir), osp.basename(original_path.rstrip("/"))]
    for prefix in [candidate for candidate in prefix_candidates if candidate]:
        for artifact_name in artifact_names:
            for ext in _IMAGE_EXTENSIONS:
                candidate = osp.join(segment_dir, f"{prefix}_{artifact_name}{suffix}{ext}")
                if osp.isfile(candidate):
                    return candidate

    matches = []
    with os.scandir(segment_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            lower_name = entry.name.lower()
            if not lower_name.endswith(_IMAGE_EXTENSIONS):
                continue
            stem_lower = osp.splitext(lower_name)[0]
            for artifact_name in artifact_names:
                if stem_lower.endswith(f"_{artifact_name}{suffix}".lower()):
                    matches.append(entry.path)
                    break

    if len(matches) == 1:
        return matches[0]

    picked = _pick_best_path(
        request_aliases,
        matches,
        context=f"{artifact_names[0]} file for {fragment_id!r}",
    )
    if picked is not None:
        return picked

    suffix_display = "/".join(artifact_names)
    raise FileNotFoundError(
        f"Could not resolve {suffix_display} file for {fragment_id!r} inside {segment_dir!r}."
    )


def resolve_segment_label_path(fragment_id, *, seg_meta=None, suffix=""):
    return _resolve_named_image_path(
        fragment_id,
        seg_meta=seg_meta,
        artifact_names=("inklabels",),
        suffix=suffix,
    )


def resolve_segment_mask_path(fragment_id, *, seg_meta=None, suffix=""):
    return _resolve_named_image_path(
        fragment_id,
        seg_meta=seg_meta,
        artifact_names=("mask", "supervision_mask"),
        suffix=suffix,
    )


def resolve_segment_zarr_path(fragment_id, *, seg_meta=None):
    dataset_root = osp.normpath(str(getattr(CFG, "dataset_root", "train_scrolls")))
    fragment_id = str(fragment_id).strip()
    if not fragment_id:
        raise ValueError("segment id must be a non-empty string")

    direct_candidate = osp.normpath(osp.join(dataset_root, f"{fragment_id}.zarr"))
    if _looks_like_zarr_store(direct_candidate):
        return direct_candidate

    original_path = str((seg_meta or {}).get("original_path", "") or "")
    original_basename = osp.basename(original_path.rstrip("/"))
    if original_basename:
        direct_candidate = osp.normpath(osp.join(dataset_root, f"{original_basename}.zarr"))
        if _looks_like_zarr_store(direct_candidate):
            return direct_candidate

    segment_dir = resolve_segment_dir(fragment_id, seg_meta=seg_meta)
    request_aliases = _request_aliases(fragment_id, original_path)
    prefix_candidates = [fragment_id, osp.basename(segment_dir), original_basename]
    for prefix in [candidate for candidate in prefix_candidates if candidate]:
        candidate = osp.join(segment_dir, f"{prefix}.zarr")
        if _looks_like_zarr_store(candidate):
            return candidate

    matches = []
    with os.scandir(segment_dir) as entries:
        for entry in entries:
            lower_name = entry.name.lower()
            if not lower_name.endswith(".zarr"):
                continue
            if lower_name.endswith(("_inklabels.zarr", "_mask.zarr", "_supervision_mask.zarr")):
                continue
            if "_max_" in lower_name:
                continue
            if _looks_like_zarr_store(entry.path):
                matches.append(entry.path)

    if len(matches) == 1:
        return matches[0]

    picked = _pick_best_path(
        request_aliases,
        matches,
        context=f"zarr volume for {fragment_id!r}",
    )
    if picked is not None:
        return picked

    raise FileNotFoundError(
        f"Could not resolve zarr volume path for segment={fragment_id}. "
        f"Expected zarr store under dataset_root={dataset_root!r} or inside {segment_dir!r}."
    )
