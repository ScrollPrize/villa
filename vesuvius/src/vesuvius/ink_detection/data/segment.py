"""Discover segment directories and select versioned label assets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.types import Segment


_LABEL_SUFFIXES = (
    ("_inklabels", "inklabels"),
    ("_supervision_mask", "supervision_mask"),
    ("_validation_mask", "validation_mask"),
)
_LABEL_EXTENSIONS = {".tif", ".tiff", ".zarr"}
_REMOTE_VOLUME_PREFIXES = ("s3://", "http://", "https://")


def parse_label_asset_path(path: str | Path) -> dict[str, object] | None:
    """Parse the label basename vocabulary; unversioned means v1."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in _LABEL_EXTENSIONS:
        return None
    stem = path.stem
    version_num = 1
    stem_parts = stem.rsplit("_", 1)
    if (
        len(stem_parts) == 2
        and stem_parts[1].startswith("v")
        and stem_parts[1][1:].isdigit()
    ):
        version_num = int(stem_parts[1][1:])
        stem = stem_parts[0]
    for suffix, label_kind in _LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return {
                "path": path,
                "dir_prefix": path.parent,
                "name": path.name,
                "prefix": stem[: -len(suffix)],
                "label_kind": label_kind,
                "version_num": version_num,
                "extension": extension,
            }
    return None


def build_matching_label_asset_path(
    path: str | Path, *, label_kind: str
) -> Path:
    parsed = parse_label_asset_path(path)
    if parsed is None:
        raise ValueError(f"Label path has unexpected format: {path}")
    version_num = int(parsed["version_num"])
    version_suffix = "" if version_num <= 1 else f"_v{version_num}"
    return Path(parsed["dir_prefix"]) / (
        f"{parsed['prefix']}_{label_kind}{version_suffix}{parsed['extension']}"
    )


def resolve_versioned_label_path(
    paths: Iterable[str | Path],
    *,
    label_kind: str,
    label_version: str | None = None,
    context: str = "labels",
) -> Path:
    """Select one label kind with traversal-order overwrite semantics."""
    requested = None if label_version in (None, "") else str(label_version).strip()
    candidates: dict[int, Path] = {}
    for path in paths:
        parsed = parse_label_asset_path(path)
        if parsed is None or parsed["label_kind"] != label_kind:
            continue
        candidates[int(parsed["version_num"])] = Path(path)
    if not candidates:
        raise ValueError(f"{context} must contain at least one {label_kind} path")
    if requested:
        for version_num, candidate in candidates.items():
            if f"v{version_num}" == requested:
                return candidate
        raise ValueError(
            f"{context} does not contain {label_kind} version {requested}"
        )
    return candidates[max(candidates)]


def discover_segment_labels(
    segment: Segment, *, extension: str = ".zarr", required: bool = True
) -> Segment:
    """Resolve independently greatest versions, or one matching explicit version."""
    normalized_extension = str(extension).lower()
    requested = segment.data_config.label_version
    candidates_by_version: dict[int, dict[str, Path]] = {}
    candidates_by_kind: dict[str, dict[int, Path]] = {
        "inklabels": {},
        "supervision_mask": {},
        "validation_mask": {},
    }
    for path in segment.segment_dir.iterdir():
        parsed = parse_label_asset_path(path)
        if parsed is None:
            continue
        if parsed["prefix"] != segment.segment_name:
            continue
        if str(parsed["extension"]).lower() != normalized_extension:
            continue
        version_num = int(parsed["version_num"])
        kind = str(parsed["label_kind"])
        candidates_by_version.setdefault(version_num, {})[kind] = path
        candidates_by_kind[kind][version_num] = path

    if requested:
        selected = None
        for version_num, record in candidates_by_version.items():
            if (
                "inklabels" in record
                and "supervision_mask" in record
                and f"v{version_num}" == requested
            ):
                selected = record
                break
        if selected is None:
            if required:
                raise ValueError(
                    f"{segment.segment_dir} does not contain matching "
                    f"{normalized_extension} labels for version {requested}"
                )
            return segment.with_labels(
                inklabels=None, supervision_mask=None, validation_mask=None
            )
        return segment.with_labels(
            inklabels=selected["inklabels"],
            supervision_mask=selected["supervision_mask"],
            validation_mask=selected.get("validation_mask"),
        )

    if required:
        if not candidates_by_kind["inklabels"]:
            raise ValueError(
                f"{segment.segment_dir} must contain at least one inklabels "
                f"{normalized_extension} asset"
            )
        if not candidates_by_kind["supervision_mask"]:
            raise ValueError(
                f"{segment.segment_dir} must contain at least one supervision_mask "
                f"{normalized_extension} asset"
            )
    selected_paths: dict[str, Path | None] = {}
    for kind, candidates in candidates_by_kind.items():
        selected_paths[kind] = candidates[max(candidates)] if candidates else None
    return segment.with_labels(
        inklabels=selected_paths["inklabels"],
        supervision_mask=selected_paths["supervision_mask"],
        validation_mask=selected_paths["validation_mask"],
    )


def gather_segments(config: InkDataConfig) -> tuple[Segment, ...]:
    """Discover usable segment representations in stable directory-name order."""
    segments: list[Segment] = []
    native_mode = config.mode in {"full_3d", "full_3d_single_wrap"}
    for dataset_idx, source in enumerate(config.active_datasets):
        allowlist = set(source.segment_names)
        for segment_dir in sorted(source.segments_path.iterdir()):
            if not segment_dir.is_dir() or segment_dir.name == "unused":
                continue
            if allowlist and segment_dir.name not in allowlist:
                continue
            explicit_surface = bool(
                source.surface_volume_paths or source.surface_volume_path is not None
            )
            if (native_mode or not explicit_surface) and not any(
                segment_dir.rglob("x.tif")
            ):
                continue
            relpath = segment_dir.relative_to(source.segments_path).as_posix()
            if native_mode:
                if source.volume_path is None:
                    raise ValueError(
                        f"datasets[{dataset_idx}].volume_path is required for {config.mode}"
                    )
                image_volume: str | Path = source.volume_path
            elif source.surface_volume_paths:
                explicit = source.surface_volume_paths.get(
                    relpath, source.surface_volume_paths.get(segment_dir.name)
                )
                if explicit is None:
                    raise KeyError(
                        "surface_volume_paths has no entry for "
                        f"segment {relpath!r}"
                    )
                image_volume = explicit
            elif source.surface_volume_path is not None:
                if len(allowlist) != 1:
                    raise ValueError(
                        "surface_volume_path is only unambiguous with exactly one "
                        "allowlisted segment; use surface_volume_paths for multiple segments"
                    )
                image_volume = source.surface_volume_path
            else:
                image_volume = segment_dir / f"{segment_dir.name}.zarr"

            segment = Segment(
                data_config=config,
                source=source,
                dataset_idx=dataset_idx,
                segment_relpath=relpath,
                segment_dir=segment_dir,
                segment_name=segment_dir.name,
                image_volume=image_volume,
            )
            segment = discover_segment_labels(
                segment,
                extension=".zarr",
                required=config.discovery_mode != "unlabeled",
            )
            image_is_remote = str(image_volume).lower().startswith(
                _REMOTE_VOLUME_PREFIXES
            )
            if not image_is_remote and not Path(image_volume).exists():
                if config.discovery_mode == "unlabeled":
                    continue
                raise ValueError(
                    f"{segment_dir.name} is missing its image volume {image_volume}"
                )
            if config.discovery_mode != "unlabeled":
                if not (
                    segment.supervision_mask is not None
                    and segment.inklabels is not None
                    and segment.supervision_mask.exists()
                    and segment.inklabels.exists()
                ):
                    raise ValueError(
                        f"{segment_dir.name} is missing required image, supervision, or labels"
                    )
            segments.append(segment)
    return tuple(segments)
