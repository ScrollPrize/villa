from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re

from ink.recipes.data.masks import SUPERVISION_MASK_NAME, normalize_mask_names


@dataclass(frozen=True)
class SegmentPaths:
    segment_dir: Path
    volume_path: Path
    inklabels_path: Path
    mask_path: Path

    @property
    def supervision_mask_path(self) -> Path:
        return self.mask_path


def resolve_segment_artifact_path(
    segment_dir: str | Path,
    segment_id: str,
    *,
    artifact_name: str,
    suffix: str = "",
    required: bool = True,
) -> Path:
    segment_dir = Path(segment_dir)
    if str(suffix):
        path = segment_dir / f"{str(segment_id)}_{str(artifact_name)}{str(suffix)}.zarr"
    else:
        path = _resolve_latest_segment_artifact_path(
            segment_dir,
            segment_id,
            artifact_name=artifact_name,
        )
    if required and not path.exists():
        raise FileNotFoundError(
            f"Could not resolve {artifact_name} zarr for {segment_id!r} inside {str(segment_dir)!r}. "
            f"Expected {str(path)!r}."
        )
    return path


@dataclass(frozen=True)
class NestedZarrLayout:
    dataset_root: str | Path
    _index_built: bool = field(default=False, init=False, repr=False, compare=False)
    _segment_dirs: dict[str, Path] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _ensure_segment_index(self) -> None:
        if self._index_built:
            return

        dataset_root = Path(self.dataset_root)
        segment_dirs: dict[str, Path] = {}
        for group_dir in sorted(dataset_root.iterdir(), key=lambda path: path.name):
            if not group_dir.is_dir():
                continue
            for segment_dir in sorted(group_dir.iterdir(), key=lambda path: path.name):
                if not segment_dir.is_dir():
                    continue
                segment_dirs.setdefault(segment_dir.name, segment_dir)

        object.__setattr__(self, "_segment_dirs", segment_dirs)
        object.__setattr__(self, "_index_built", True)

    def resolve_segment_dir(self, segment_id: str) -> Path:
        """Resolve a segment id across grouped dataset folders."""
        segment_id = str(segment_id).strip()
        if not segment_id:
            raise ValueError("segment id must be a non-empty string")

        self._ensure_segment_index()
        segment_dir = self._segment_dirs.get(segment_id)
        if segment_dir is None:
            raise FileNotFoundError(
                f"Could not resolve segment directory for {segment_id!r} under dataset_root={str(self.dataset_root)!r}."
            )
        return Path(segment_dir)

    def resolve_group_name(self, segment_id: str) -> str:
        group_name = self.resolve_segment_dir(segment_id).parent.name
        if not group_name:
            raise ValueError(f"could not resolve group name for segment_id={str(segment_id)!r}")
        return group_name

    def resolve_paths(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_name: str = SUPERVISION_MASK_NAME,
    ) -> SegmentPaths:
        """Resolve the canonical volume, label, and mask paths for one segment."""
        segment_id = str(segment_id).strip()
        segment_dir = self.resolve_segment_dir(segment_id)
        volume_path = segment_dir / f"{segment_id}.zarr"
        if not volume_path.exists():
            raise FileNotFoundError(
                f"Could not resolve zarr volume for {segment_id!r} inside {str(segment_dir)!r}. "
                f"Expected {str(volume_path)!r}."
            )

        inklabels_path = resolve_segment_artifact_path(
            segment_dir,
            segment_id,
            artifact_name="inklabels",
            suffix=label_suffix,
        )
        mask_path = resolve_segment_artifact_path(
            segment_dir,
            segment_id,
            artifact_name=mask_name,
            suffix=mask_suffix,
        )

        return SegmentPaths(
            segment_dir=Path(segment_dir),
            volume_path=volume_path,
            inklabels_path=inklabels_path,
            mask_path=mask_path,
        )

    def _mask_signature_payload(
        self,
        segment_id: str,
        *,
        mask_suffix: str,
        mask_names,
        signature_fn,
    ) -> tuple[tuple[str, object], ...]:
        segment_dir = self.resolve_segment_dir(segment_id)
        return tuple(
            (
                str(current_mask_name),
                signature_fn(
                    resolve_segment_artifact_path(
                        segment_dir,
                        segment_id,
                        artifact_name=current_mask_name,
                        suffix=mask_suffix,
                    )
                ),
            )
            for current_mask_name in mask_names
        )

    def label_mask_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_name: str = SUPERVISION_MASK_NAME,
        mask_names=None,
    ) -> str:
        """Hash label and supervision-mask contents so caches can detect dataset changes."""
        resolved_mask_names = normalize_mask_names(mask_name=mask_name, mask_names=mask_names)
        paths = self.resolve_paths(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            mask_name=resolved_mask_names[0],
        )
        payload = (
            ("segment_id", str(segment_id)),
            ("label_suffix", str(label_suffix)),
            ("mask_suffix", str(mask_suffix)),
            ("mask_names", resolved_mask_names),
            ("inklabels", _path_tree_signature(paths.inklabels_path)),
            (
                "masks",
                self._mask_signature_payload(
                    segment_id,
                    mask_suffix=mask_suffix,
                    mask_names=resolved_mask_names,
                    signature_fn=_path_tree_signature,
                ),
            ),
        )
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def label_mask_metadata_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_name: str = SUPERVISION_MASK_NAME,
        mask_names=None,
    ) -> str:
        """Hash only zarr metadata files for label and supervision-mask sources."""
        resolved_mask_names = normalize_mask_names(mask_name=mask_name, mask_names=mask_names)
        paths = self.resolve_paths(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            mask_name=resolved_mask_names[0],
        )
        payload = (
            ("segment_id", str(segment_id)),
            ("label_suffix", str(label_suffix)),
            ("mask_suffix", str(mask_suffix)),
            ("mask_names", resolved_mask_names),
            ("inklabels", _zarr_metadata_signature(paths.inklabels_path)),
            (
                "masks",
                self._mask_signature_payload(
                    segment_id,
                    mask_suffix=mask_suffix,
                    mask_names=resolved_mask_names,
                    signature_fn=_zarr_metadata_signature,
                ),
            ),
        )
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def segment_source_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_name: str = SUPERVISION_MASK_NAME,
        mask_names=None,
    ) -> str:
        """Hash the canonical volume, label, and supervision-mask sources for one segment."""
        return self._segment_source_fingerprint(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            mask_name=mask_name,
            mask_names=mask_names,
            signature_fn=_path_tree_signature,
        )

    def segment_source_metadata_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_name: str = SUPERVISION_MASK_NAME,
        mask_names=None,
    ) -> str:
        """Hash only zarr metadata for the canonical volume, label, and mask sources."""
        return self._segment_source_fingerprint(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            mask_name=mask_name,
            mask_names=mask_names,
            signature_fn=_zarr_metadata_signature,
        )

    def _segment_source_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
        mask_name: str = SUPERVISION_MASK_NAME,
        mask_names=None,
        signature_fn,
    ) -> str:
        resolved_mask_names = normalize_mask_names(mask_name=mask_name, mask_names=mask_names)
        paths = self.resolve_paths(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
            mask_name=resolved_mask_names[0],
        )
        payload = (
            ("segment_id", str(segment_id)),
            ("label_suffix", str(label_suffix)),
            ("mask_suffix", str(mask_suffix)),
            ("mask_names", resolved_mask_names),
            ("volume", signature_fn(paths.volume_path)),
            ("inklabels", signature_fn(paths.inklabels_path)),
            (
                "masks",
                self._mask_signature_payload(
                    segment_id,
                    mask_suffix=mask_suffix,
                    mask_names=resolved_mask_names,
                    signature_fn=signature_fn,
                ),
            ),
        )
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _resolve_latest_segment_artifact_path(
    segment_dir: Path,
    segment_id: str,
    *,
    artifact_name: str,
) -> Path:
    segment_id = str(segment_id)
    artifact_name = str(artifact_name)
    exact_name = f"{segment_id}_{artifact_name}.zarr"
    version_pattern = re.compile(
        rf"^{re.escape(segment_id)}_{re.escape(artifact_name)}_v(?P<version>\d+)\.zarr$"
    )

    best_path = segment_dir / exact_name
    best_version = 0 if best_path.exists() else -1
    for candidate in segment_dir.iterdir():
        if not candidate.is_dir():
            continue
        match = version_pattern.match(candidate.name)
        if match is None:
            continue
        version = int(match.group("version"))
        if version > best_version:
            best_path = candidate
            best_version = version
    return best_path


def _hash_file_contents(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _zarr_metadata_signature(path: Path) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    """Summarize the zarr metadata files we actually use, never chunk payloads."""
    root = Path(path)
    if not root.exists():
        return (root.name, (("missing", -1, "missing"),))

    entries = []
    for child in (
        root / ".zattrs",
        root / ".zgroup",
        root / "0" / ".zarray",
        root / "0" / ".zattrs",
    ):
        if child.is_file():
            entries.append((child.relative_to(root).as_posix(), int(child.stat().st_size), _hash_file_contents(child)))
    return (root.name, tuple(entries))


def _path_tree_signature(path: Path) -> tuple[str, int, int, str]:
    """Summarize a path tree by name, size, file count, and content hash."""
    root = Path(path)
    if root.is_file():
        stat = root.stat()
        return (
            root.name,
            int(stat.st_size),
            1,
            _hash_file_contents(root),
        )
    if not root.exists():
        return (root.name, -1, 0, "missing")

    digest = hashlib.sha1()
    total_size = 0
    file_count = 0
    for child in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not child.is_file():
            continue
        relative_name = child.relative_to(root).as_posix()
        stat = child.stat()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(stat.st_size)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file_contents(child).encode("utf-8"))
        digest.update(b"\0")
        total_size += int(stat.st_size)
        file_count += 1
    return (
        root.name,
        int(total_size),
        int(file_count),
        digest.hexdigest(),
    )

__all__ = ["NestedZarrLayout", "SegmentPaths", "resolve_segment_artifact_path"]
