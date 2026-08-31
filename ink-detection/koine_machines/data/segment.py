import hashlib
import os
from pathlib import Path

_LABEL_SUFFIXES = (
    ("_inklabels", "inklabels"),
    ("_supervision_mask", "supervision_mask"),
    ("_validation_mask", "validation_mask"),
)
_LABEL_EXTENSIONS = {".tif", ".tiff", ".zarr"}


def _parse_label_asset_path(name):
    path = Path(name)
    extension = path.suffix.lower()
    if extension not in _LABEL_EXTENSIONS:
        return None

    stem = path.stem
    version_num = 1
    stem_parts = stem.rsplit("_", 1)
    if len(stem_parts) == 2 and stem_parts[1].startswith("v") and stem_parts[1][1:].isdigit():
        version_num = int(stem_parts[1][1:])
        stem = stem_parts[0]

    for suffix, label_kind in _LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return {
                "prefix": stem[: -len(suffix)],
                "label_kind": label_kind,
                "version_num": version_num,
                "extension": extension,
            }
    return None


def label_asset_fingerprint(paths):
    """Return a short digest of the label assets' on-disk layout.

    The patch cache identifies a segment's labels by *path*, which catches pointing at a
    different tree and misses regenerating a mask in place -- the split is then silently
    reused against labels it no longer describes. Hashing each asset's relative file names
    and sizes catches that: rewriting a zarr changes chunk sizes, and dropping annotation
    deletes chunks outright when the store does not write empty ones.

    It reads no chunk contents, so it costs one directory walk (8 ms for a 6,429-file
    label array here, via ``os.scandir``, which carries the size on Windows and Linux
    alike) and it does not distinguish two different labels that compress to identical
    sizes under identical names. A copied tree fingerprints the same as its source, which
    is the wanted behaviour.
    """
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in paths if p):
        root = Path(path)
        digest.update(root.name.encode())
        if not root.exists():
            digest.update(b"\0missing")
            continue
        stack = [("", str(root))]
        entries = []
        while stack:
            relative, current = stack.pop()
            try:
                children = sorted(os.scandir(current), key=lambda entry: entry.name)
            except OSError:
                continue
            for child in children:
                name = f"{relative}/{child.name}" if relative else child.name
                if child.is_dir(follow_symlinks=False):
                    stack.append((name, child.path))
                else:
                    try:
                        size = child.stat().st_size
                    except OSError:
                        size = -1
                    entries.append((name, size))
        for name, size in sorted(entries):
            digest.update(name.encode())
            digest.update(str(size).encode())
    return digest.hexdigest()[:16]

class Segment:
    def __init__(
        self,
        config,
        image_volume=None,
        image_volume_3d=None,
        supervision_mask=None,
        validation_mask=None,
        inklabels=None,
        scale=None,
        dataset_idx=None,
        segment_relpath=None,
        segment_dir=None,
        segment_name=None,
    ):
        self.config = config
        self.scale = scale
        self.image_volume = image_volume
        self.supervision_mask = supervision_mask
        self.validation_mask = validation_mask
        self.inklabels = inklabels
        self.dataset_idx = dataset_idx
        self.segment_relpath = segment_relpath
        self.segment_dir = None if segment_dir is None else Path(segment_dir)
        self.segment_name = segment_name
        self.patch_size = config["patch_size"]

    @staticmethod
    def parse_label_asset_path(path):
        path = Path(path)
        parsed = _parse_label_asset_path(path.name)
        if parsed is None:
            return None
        return {
            "path": path,
            "dir_prefix": path.parent,
            "name": path.name,
            **parsed,
        }

    @classmethod
    def build_matching_label_asset_path(cls, path, *, label_kind):
        parsed = cls.parse_label_asset_path(path)
        assert parsed is not None, f"Label path has unexpected format: {path}"
        version_suffix = (
            "" if int(parsed["version_num"]) <= 1 else f"_v{int(parsed['version_num'])}"
        )
        return parsed["dir_prefix"] / (
            f"{parsed['prefix']}_{str(label_kind)}{version_suffix}{parsed['extension']}"
        )

    @classmethod
    def resolve_versioned_label_path(
        cls,
        paths,
        *,
        label_kind,
        label_version=None,
        context="labels",
    ):
        requested_version = (
            None if label_version in (None, "") else str(label_version).strip()
        )
        candidates = {}
        for path in paths:
            parsed = cls.parse_label_asset_path(path)
            if parsed is None or parsed["label_kind"] != str(label_kind):
                continue
            candidates[int(parsed["version_num"])] = Path(path)

        assert candidates, f"{context} must contain at least one {label_kind} path."

        if requested_version:
            for version_num, candidate_path in candidates.items():
                if f"v{version_num}" == requested_version:
                    return candidate_path
            raise AssertionError(
                f"{context} does not contain {label_kind} version {requested_version}."
            )

        return candidates[max(candidates)]

    @classmethod
    def resolve_segment_inklabel_path(cls, segment, *, label_version=None):
        segment_uuid = str(segment.uuid)
        segment_path = getattr(segment, "path", None)
        if segment_path is not None:
            segment_dir = Path(str(segment_path))
            if segment_dir.is_dir():
                local_segment = cls(
                    config={
                        "patch_size": [1, 1, 1],
                        "label_version": label_version,
                    },
                    segment_dir=segment_dir,
                    segment_name=segment_dir.name,
                )
                inklabels, _, _ = local_segment.discover_labels(
                    label_version=label_version,
                    extension=".zarr",
                )
                return inklabels

        ink_label_paths = [
            Path(str(label["path"]))
            for label in segment.list_labels()
            if label.get("name") == "inklabels" and label.get("path") is not None
        ]
        return cls.resolve_versioned_label_path(
            ink_label_paths,
            label_kind="inklabels",
            label_version=label_version,
            context=f"Segment {segment_uuid!r}",
        )

    @property
    def label_fingerprint(self):
        """Digest of the label assets as they are on disk right now (computed once)."""
        cached = getattr(self, "_label_fingerprint", None)
        if cached is None:
            cached = label_asset_fingerprint(
                (self.inklabels, self.supervision_mask, self.validation_mask)
            )
            object.__setattr__(self, "_label_fingerprint", cached)
        return cached

    @property
    def cache_key(self):
        return (
            int(self.dataset_idx),
            str(self.segment_relpath),
            self.scale,
            "" if self.inklabels is None else str(self.inklabels),
            "" if self.supervision_mask is None else str(self.supervision_mask),
            "" if self.validation_mask is None else str(self.validation_mask),
            self.label_fingerprint,
        )
    
    def discover_labels(self, *, label_version=None, extension=".zarr", required=True):
        if self.segment_dir is None:
            raise ValueError("segment_dir is required for local label discovery")

        segment_name = self.segment_name or self.segment_dir.name
        resolved_label_version = self.config.get("label_version") if label_version is None else label_version
        normalized_extension = str(extension).lower()
        requested_version = (
            None if resolved_label_version in (None, "") else str(resolved_label_version).strip()
        )
        candidates_by_version = {}
        candidates_by_kind = {
            "inklabels": {},
            "supervision_mask": {},
            "validation_mask": {},
        }

        for path in self.segment_dir.iterdir():
            parsed = self.parse_label_asset_path(path)
            if parsed is None:
                continue
            if parsed["prefix"] != str(segment_name):
                continue
            if parsed["extension"].lower() != normalized_extension:
                continue
            version_num = int(parsed["version_num"])
            version_entry = candidates_by_version.setdefault(version_num, {})
            version_entry[parsed["label_kind"]] = path
            candidates_by_kind[parsed["label_kind"]][version_num] = path

        if requested_version:
            selected = None
            for version_num, record in candidates_by_version.items():
                if "inklabels" not in record or "supervision_mask" not in record:
                    continue
                if f"v{version_num}" == requested_version:
                    selected = record
                    break
            if selected is None:
                if required:
                    raise AssertionError(
                        f"{self.segment_dir} does not contain matching {normalized_extension} labels "
                        f"for version {requested_version}."
                    )
                inklabels = None
                supervision_mask = None
                validation_mask = None
            else:
                inklabels = selected["inklabels"]
                supervision_mask = selected["supervision_mask"]
                validation_mask = selected.get("validation_mask")
        else:
            if required:
                assert candidates_by_kind["inklabels"], (
                    f"{self.segment_dir} must contain at least one inklabels {normalized_extension} asset."
                )
                assert candidates_by_kind["supervision_mask"], (
                    f"{self.segment_dir} must contain at least one supervision_mask {normalized_extension} asset."
                )
            validation_candidates = candidates_by_kind["validation_mask"]
            inklabels = (
                candidates_by_kind["inklabels"][max(candidates_by_kind["inklabels"])]
                if candidates_by_kind["inklabels"]
                else None
            )
            supervision_mask = (
                candidates_by_kind["supervision_mask"][max(candidates_by_kind["supervision_mask"])]
                if candidates_by_kind["supervision_mask"]
                else None
            )
            validation_mask = (
                validation_candidates[max(validation_candidates)] if validation_candidates else None
            )

        self.inklabels = inklabels
        self.supervision_mask = supervision_mask
        self.validation_mask = validation_mask
        return inklabels, supervision_mask, validation_mask
    
    def _find_patches(self):
        from koine_machines.data.patch_finding.default import (
            find_segment_patches,
            find_segment_unlabeled_patches,
        )
        from koine_machines.data.patch import Patch

        patch_discovery_mode = str(self.config.get("patch_discovery_mode", "labeled")).strip().lower()
        if patch_discovery_mode == "unlabeled":
            training_patches, validation_patches = find_segment_unlabeled_patches(self, Patch)
        else:
            training_patches, validation_patches = find_segment_patches(self, Patch)
        self.training_patches = training_patches
        self.validation_patches = validation_patches
        self.patches = training_patches + validation_patches
