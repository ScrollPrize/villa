from __future__ import annotations

from dataclasses import dataclass, field
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

__all__ = ["NestedZarrLayout", "SegmentPaths"]
