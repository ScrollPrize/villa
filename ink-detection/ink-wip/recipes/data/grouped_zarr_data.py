from __future__ import annotations

from dataclasses import dataclass, field

from ink.recipes.data.samplers import GroupBalancedSampler, GroupStratifiedSampler, ShuffleSampler
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe, ZarrPatchDataset, _batch_from_parts


def _collate_grouped_batch(samples):
    images, labels, valid_masks, xyxys, segment_ids, group_idxs = zip(*samples)
    return _batch_from_parts(
        images=images,
        labels=labels,
        valid_masks=valid_masks,
        xyxys=xyxys,
        segment_ids=segment_ids,
        group_idxs=group_idxs,
    )


class GroupedZarrPatchDataset(ZarrPatchDataset):
    def __init__(self, samples, **kwargs):
        self._group_idxs = [int(group_idx) for _, _, _, _, group_idx in samples]
        super().__init__(
            [
                (segment_id, layer_range, reverse_layers, xyxy)
                for segment_id, layer_range, reverse_layers, xyxy, _group_idx in samples
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

    def _build_samples_by_split(
        self,
        *,
        layout,
        segments,
        in_channels: int,
        patch_size: int,
        tile_size: int,
        stride: int,
        volume_cache,
        mask_cache,
    ) -> dict[str, list]:
        base_samples_by_split = super()._build_samples_by_split(
            layout=layout,
            segments=segments,
            in_channels=in_channels,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            volume_cache=volume_cache,
            mask_cache=mask_cache,
        )
        group_name_to_idx = {
            group_name: idx
            for idx, group_name in enumerate(
                sorted(
                    {
                        layout.resolve_group_name(segment_id)
                        for segment_ids in self._split_segment_ids().values()
                        for segment_id in segment_ids
                    }
                )
            )
        }
        return {
            split: [
                (
                    segment_id,
                    layer_range,
                    reverse_layers,
                    xyxy,
                    int(group_name_to_idx[layout.resolve_group_name(segment_id)]),
                )
                for segment_id, layer_range, reverse_layers, xyxy in samples
            ]
            for split, samples in base_samples_by_split.items()
        }

    def _build_group_counts(
        self,
        *,
        samples_by_split: dict[str, list],
    ) -> list[int] | None:
        group_counts = [0]
        for _, _, _, _, group_idx in samples_by_split["train"]:
            while int(group_idx) >= len(group_counts):
                group_counts.append(0)
            group_counts[int(group_idx)] += 1
        return group_counts if any(group_counts) else None

    def _dataset_class(self):
        return GroupedZarrPatchDataset

    def _collate_fn(self):
        return _collate_grouped_batch


__all__ = ["GroupedZarrPatchDataRecipe"]
