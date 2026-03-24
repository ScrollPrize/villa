from __future__ import annotations

from dataclasses import dataclass, field

from ink.recipes.data.samplers import GroupBalancedSampler, GroupStratifiedSampler, ShuffleSampler
from ink.recipes.data.zarr_data import (
    ZarrDataContext,
    ZarrPatchDataRecipe,
    ZarrPatchDataset,
    build_patch_data_bundle,
    build_zarr_split_samples,
    collate_grouped_batch,
    count_group_idxs,
    log_data_progress,
)


class GroupedZarrPatchDataset(ZarrPatchDataset):
    def __init__(self, samples, **kwargs):
        self._group_idxs = [int(group_idx) for _, _, _, group_idx in samples]
        super().__init__(
            [
                (segment_id, xyxy, bbox_index)
                for segment_id, xyxy, bbox_index, _group_idx in samples
            ],
            **kwargs,
        )

    @property
    def sample_groups(self) -> list[int]:
        return list(self._group_idxs)

    def __getitem__(self, idx):
        idx = int(idx)
        image, label, valid_mask, xyxy, segment_id = super().__getitem__(idx)
        return image, label, valid_mask, xyxy, segment_id, self._group_idxs[idx]


@dataclass(frozen=True)
class GroupedZarrPatchDataRecipe(ZarrPatchDataRecipe):
    sampler: ShuffleSampler | GroupBalancedSampler | GroupStratifiedSampler = field(default_factory=ShuffleSampler)

    def build(self, *, runtime=None, augment=None):
        assert augment is not None
        patch_size = int(self.patch_size)
        in_channels = int(self.in_channels)
        tile_size = patch_size if self.tile_size is None else int(self.tile_size)
        stride = patch_size if self.stride is None else int(self.stride)
        context = ZarrDataContext.from_recipe(self)
        log_data_progress(f"[data] dataset_root={self.dataset_root}")

        split_segment_ids = self.split_segment_ids()
        base_samples_by_split = {
            split: build_zarr_split_samples(
                context,
                segment_ids=segment_ids,
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                split_name=split,
                build_workers=max(0, int(self.num_workers)),
            )
            for split, segment_ids in split_segment_ids.items()
        }
        samples_by_split = group_samples_by_split(
            layout=context.layout,
            split_segment_ids=split_segment_ids,
            base_samples_by_split=base_samples_by_split,
        )

        log_data_progress(f"[data] train segments={len(self.train_segment_ids)} patches={len(samples_by_split['train'])}")
        log_data_progress(f"[data] eval segments={len(self.val_segment_ids)} patches={len(samples_by_split['valid'])}")

        extras = dict(self.extras or {})
        normalization_stats = extras.pop("normalization_stats", None)
        normalization = self.normalization.build(normalization_stats=normalization_stats)
        augment_recipe = augment.build(patch_size=patch_size, runtime=runtime)
        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size

        dataset_kwargs = {
            "layout": context.layout,
            "segments": context.segments,
            "augment": augment_recipe,
            "normalization": normalization,
            "patch_size": patch_size,
            "tile_size": tile_size,
            "stride": stride,
            "in_channels": in_channels,
            "label_suffix": context.label_suffix,
            "mask_suffix": context.mask_suffix,
            "volume_cache": context.volume_cache,
            "label_mask_store_cache": context.label_mask_store_cache,
            "patch_index_cache_dir": context.patch_index_cache_dir,
        }
        train_dataset = GroupedZarrPatchDataset(
            samples_by_split["train"],
            split="train",
            segment_ids=split_segment_ids["train"],
            cache_patches_in_memory=bool(self.cache_train_patches_in_memory),
            include_valid_mask=bool(self.include_train_valid_mask),
            **dataset_kwargs,
        )
        eval_dataset = GroupedZarrPatchDataset(
            samples_by_split["valid"],
            split="valid",
            segment_ids=split_segment_ids["valid"],
            cache_patches_in_memory=False,
            include_valid_mask=True,
            **dataset_kwargs,
        )

        train_patch_ram_cache = "disabled"
        if train_dataset.cache_patches_in_memory:
            train_patch_ram_cache = (
                "preload_then_fork_share" if int(self.num_workers) > 0 else "lazy_single_process"
            )
        log_data_progress(
            "[data] caches "
            f"volume_entries={len(context.volume_cache)} "
            f"label_mask_store_entries={len(context.label_mask_store_cache)} "
            f"volume_cache_scope=in_process "
            f"label_mask_store_cache_scope=in_process "
            f"train_patch_ram_cache={train_patch_ram_cache}"
        )
        return build_patch_data_bundle(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_batch_size=int(self.train_batch_size),
            valid_batch_size=int(valid_batch_size),
            num_workers=max(0, int(self.num_workers)),
            shuffle=bool(self.shuffle),
            sampler=self.sampler,
            collate_fn=collate_grouped_batch,
            in_channels=in_channels,
            augment_recipe=augment_recipe,
            group_counts=count_group_idxs(sample[3] for sample in samples_by_split["train"]),
            extras=extras,
        )


def _group_name_to_idx(layout, split_segment_ids: dict[str, tuple[str, ...]]) -> dict[str, int]:
    return {
        group_name: idx
        for idx, group_name in enumerate(
            sorted(
                {
                    layout.resolve_group_name(segment_id)
                    for segment_ids in split_segment_ids.values()
                    for segment_id in segment_ids
                }
            )
        )
    }


def group_samples_by_split(*, layout, split_segment_ids: dict[str, tuple[str, ...]], base_samples_by_split: dict[str, list]) -> dict[str, list]:
    group_name_to_idx = _group_name_to_idx(layout, split_segment_ids)
    return {
        split: [
            (
                segment_id,
                xyxy,
                bbox_index,
                int(group_name_to_idx[layout.resolve_group_name(segment_id)]),
            )
            for segment_id, xyxy, bbox_index in samples
        ]
        for split, samples in base_samples_by_split.items()
    }


__all__ = ["GroupedZarrPatchDataRecipe", "group_samples_by_split"]
