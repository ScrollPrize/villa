from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class SegmentPaths:
    segment_dir: Path
    volume_path: Path
    inklabels_path: Path
    supervision_mask_path: Path


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

        inklabels_path = segment_dir / f"{segment_id}_inklabels{str(label_suffix)}.zarr"
        if not inklabels_path.exists():
            raise FileNotFoundError(
                f"Could not resolve inklabels zarr for {segment_id!r} inside {str(segment_dir)!r}. "
                f"Expected {str(inklabels_path)!r}."
            )

        supervision_mask_path = segment_dir / f"{segment_id}_supervision_mask{str(mask_suffix)}.zarr"
        if not supervision_mask_path.exists():
            raise FileNotFoundError(
                f"Could not resolve supervision_mask zarr for {segment_id!r} inside {str(segment_dir)!r}. "
                f"Expected {str(supervision_mask_path)!r}."
            )

        return SegmentPaths(
            segment_dir=Path(segment_dir),
            volume_path=volume_path,
            inklabels_path=inklabels_path,
            supervision_mask_path=supervision_mask_path,
        )

    def label_mask_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
    ) -> str:
        """Hash label and supervision-mask contents so caches can detect dataset changes."""
        paths = self.resolve_paths(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
        )
        payload = (
            ("segment_id", str(segment_id)),
            ("label_suffix", str(label_suffix)),
            ("mask_suffix", str(mask_suffix)),
            ("inklabels", _path_tree_signature(paths.inklabels_path)),
            ("supervision_mask", _path_tree_signature(paths.supervision_mask_path)),
        )
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def segment_source_fingerprint(
        self,
        segment_id: str,
        *,
        label_suffix: str = "",
        mask_suffix: str = "",
    ) -> str:
        """Hash the canonical volume, label, and supervision-mask sources for one segment."""
        paths = self.resolve_paths(
            segment_id,
            label_suffix=label_suffix,
            mask_suffix=mask_suffix,
        )
        payload = (
            ("segment_id", str(segment_id)),
            ("label_suffix", str(label_suffix)),
            ("mask_suffix", str(mask_suffix)),
            ("volume", _path_tree_signature(paths.volume_path)),
            ("inklabels", _path_tree_signature(paths.inklabels_path)),
            ("supervision_mask", _path_tree_signature(paths.supervision_mask_path)),
        )
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _hash_file_contents(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

__all__ = ["NestedZarrLayout", "SegmentPaths"]
