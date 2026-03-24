from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ink.core.types import DataBundle
from ink.recipes.data.patch_bundle.writer import PatchBundleWriter
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe


@dataclass(frozen=True)
class GeneratedPatchBundleDataRecipe:
    bundle_root: str
    source: ZarrPatchDataRecipe

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        if not isinstance(self.source, ZarrPatchDataRecipe):
            raise TypeError("GeneratedPatchBundleDataRecipe.source must be ZarrPatchDataRecipe")

        bundle_root = Path(self.bundle_root).expanduser().resolve()
        PatchBundleWriter(self.source).ensure(out_root=bundle_root)

        bundled_segment_ids = tuple(
            dict.fromkeys(
                (
                    *(str(segment_id) for segment_id in self.source.train_segment_ids),
                    *(str(segment_id) for segment_id in self.source.val_segment_ids),
                )
            )
        )
        bundled_source = replace(
            self.source,
            dataset_root=str(bundle_root),
            segments={str(segment_id): {} for segment_id in bundled_segment_ids},
            dataset_version="",
            label_suffix="",
            mask_suffix="",
            patch_index_cache_dir=str(bundle_root / ".patch_index_cache"),
        )
        return bundled_source.build(runtime=runtime, augment=augment)


__all__ = ["GeneratedPatchBundleDataRecipe"]
