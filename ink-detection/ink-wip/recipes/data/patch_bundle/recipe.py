from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset
import zarr

from ink.core.types import DataBundle
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patch_bundle.writer import load_patch_bundle_manifest
from ink.recipes.data.samplers import GroupBalancedSampler, GroupStratifiedSampler, ShuffleSampler
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)
from ink.recipes.data.zarr_data import _batch_from_parts, _collate_batch


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


def _manifest_value(manifest: dict[str, Any], *path: str) -> Any:
    current = manifest
    for part in path:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"patch bundle manifest missing key path {path!r}")
        current = current[part]
    return current


class PatchBundleDataset(Dataset):
    def __init__(
        self,
        *,
        bundle_root: str | Path,
        split: str,
        augment,
        normalization,
    ):
        self.bundle_root = Path(bundle_root).expanduser().resolve()
        self.split = str(split).strip().lower()
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")
        self.manifest = load_patch_bundle_manifest(self.bundle_root, split=self.split)
        self.augment = augment
        self.normalization = normalization
        self.patch_size = int(self.manifest["extraction"]["patch_size"])
        self.in_channels = int(self.manifest["extraction"]["in_channels"])
        self.segment_ids = tuple(str(segment_id) for segment_id in self.manifest["segment_ids"])

        patches_root = self.bundle_root / self.split / "patches.zarr"
        if not patches_root.exists():
            raise FileNotFoundError(f"Could not resolve patch bundle arrays at {str(patches_root)!r}")
        store = zarr.open_group(str(patches_root), mode="r")
        self._x = store["x"]
        self._y = store["y"]
        self._valid_mask = store["valid_mask"]
        self._xyxy = store["xyxy"]
        self._segment_index = store["segment_index"]
        self.transform = build_joint_transform(
            self.split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def _raw_item(self, idx: int):
        idx = int(idx)
        image = np.asarray(self._x[idx], dtype=np.uint8)
        label = np.asarray(self._y[idx], dtype=np.uint8)
        valid_mask = np.asarray(self._valid_mask[idx], dtype=np.uint8)
        xyxy = np.asarray(self._xyxy[idx], dtype=np.int64)
        segment_idx = int(np.asarray(self._segment_index[idx]).item())
        segment_id = self.segment_ids[segment_idx]
        return image, label, valid_mask, xyxy, segment_id

    def __getitem__(self, idx):
        image, label, valid_mask, xyxy, segment_id = self._raw_item(int(idx))
        if self.split == "train":
            image, label, valid_mask = apply_train_sample_transforms(
                image,
                label,
                augment=self.augment,
                patch_size=self.patch_size,
                transform=self.transform,
                valid_mask=valid_mask,
            )
        else:
            image, label, valid_mask = apply_eval_sample_transforms(
                image,
                label,
                patch_size=self.patch_size,
                transform=self.transform,
                valid_mask=valid_mask,
            )
        return image, label, valid_mask, xyxy, segment_id


class GroupedPatchBundleDataset(PatchBundleDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        patches_root = self.bundle_root / self.split / "patches.zarr"
        store = zarr.open_group(str(patches_root), mode="r")
        if "group_idx" not in store:
            raise ValueError("grouped patch bundle requires group_idx array")
        self._group_idx = store["group_idx"]

    @property
    def sample_groups(self) -> list[int]:
        return [int(np.asarray(self._group_idx[idx]).item()) for idx in range(len(self))]

    def __getitem__(self, idx):
        image, label, valid_mask, xyxy, segment_id = super().__getitem__(int(idx))
        return image, label, valid_mask, xyxy, segment_id, int(np.asarray(self._group_idx[int(idx)]).item())


@dataclass(frozen=True)
class PatchBundleDataRecipe:
    bundle_root: str
    train_batch_size: int = 1
    valid_batch_size: int | None = None
    num_workers: int = 0
    shuffle: bool = True
    sampler: ShuffleSampler = field(default_factory=ShuffleSampler)
    normalization: Any = field(default_factory=ClipMaxDiv255Normalization)
    extras: dict[str, Any] = field(default_factory=dict)

    def _load_manifests(self) -> tuple[dict[str, Any], dict[str, Any]]:
        train_manifest = load_patch_bundle_manifest(self.bundle_root, split="train")
        valid_manifest = load_patch_bundle_manifest(self.bundle_root, split="valid")
        self._validate_recipe_family(train_manifest)
        self._validate_recipe_family(valid_manifest)
        return train_manifest, valid_manifest

    def _validate_recipe_family(self, manifest: dict[str, Any]) -> None:
        family = str(manifest.get("recipe_family"))
        if family != "patch":
            raise ValueError(f"PatchBundleDataRecipe requires recipe_family='patch', got {family!r}")

    def _validate_split_contract(self, train_manifest: dict[str, Any], valid_manifest: dict[str, Any]) -> tuple[int, int]:
        train_in_channels = int(_manifest_value(train_manifest, "extraction", "in_channels"))
        valid_in_channels = int(_manifest_value(valid_manifest, "extraction", "in_channels"))
        if train_in_channels != valid_in_channels:
            raise ValueError(
                f"patch bundle train/valid in_channels mismatch: {train_in_channels} vs {valid_in_channels}"
            )
        train_patch_size = int(_manifest_value(train_manifest, "extraction", "patch_size"))
        valid_patch_size = int(_manifest_value(valid_manifest, "extraction", "patch_size"))
        if train_patch_size != valid_patch_size:
            raise ValueError(
                f"patch bundle train/valid patch_size mismatch: {train_patch_size} vs {valid_patch_size}"
            )
        return train_in_channels, train_patch_size

    def _dataset_class(self):
        return PatchBundleDataset

    def _collate_fn(self):
        return _collate_batch

    def _build_group_counts(self, *, train_dataset, train_manifest: dict[str, Any]) -> list[int] | None:
        del train_dataset, train_manifest
        return None

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None
        train_manifest, valid_manifest = self._load_manifests()
        in_channels, patch_size = self._validate_split_contract(train_manifest, valid_manifest)

        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size
        extras = dict(self.extras or {})
        normalization_stats = extras.pop("normalization_stats", None)
        normalization = self.normalization.build(normalization_stats=normalization_stats)
        augment_recipe = augment.build(patch_size=patch_size, runtime=runtime)

        dataset_kwargs = {
            "bundle_root": Path(self.bundle_root).expanduser().resolve(),
            "augment": augment_recipe,
            "normalization": normalization,
        }
        dataset_class = self._dataset_class()
        collate_fn = self._collate_fn()
        train_dataset = dataset_class(split="train", **dataset_kwargs)
        val_dataset = dataset_class(split="valid", **dataset_kwargs)
        num_workers = max(0, int(self.num_workers))
        train_loader = self.sampler.build_loader(
            train_dataset,
            batch_size=int(self.train_batch_size),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            shuffle=bool(self.shuffle),
        )
        eval_loader = DataLoader(
            val_dataset,
            batch_size=int(valid_batch_size),
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        group_counts = self._build_group_counts(train_dataset=train_dataset, train_manifest=train_manifest)
        return DataBundle(
            train_loader=train_loader,
            eval_loader=eval_loader,
            in_channels=int(in_channels),
            augment=augment_recipe,
            group_counts=group_counts,
            extras=extras,
        )


__all__ = ["PatchBundleDataRecipe"]


@dataclass(frozen=True)
class GroupedPatchBundleDataRecipe(PatchBundleDataRecipe):
    sampler: ShuffleSampler | GroupBalancedSampler | GroupStratifiedSampler = field(default_factory=ShuffleSampler)

    def _validate_recipe_family(self, manifest) -> None:
        family = str(manifest.get("recipe_family"))
        if family != "grouped_patch":
            raise ValueError(f"GroupedPatchBundleDataRecipe requires recipe_family='grouped_patch', got {family!r}")

    def _dataset_class(self):
        return GroupedPatchBundleDataset

    def _collate_fn(self):
        return _collate_grouped_batch

    def _build_group_counts(self, *, train_dataset, train_manifest) -> list[int] | None:
        group_counts = list(train_manifest.get("group_counts") or [])
        if group_counts:
            return [int(value) for value in group_counts]
        counts = [0]
        for group_idx in train_dataset.sample_groups:
            while int(group_idx) >= len(counts):
                counts.append(0)
            counts[int(group_idx)] += 1
        return counts if any(counts) else None


__all__ = [
    "GroupedPatchBundleDataRecipe",
    "GroupedPatchBundleDataset",
    "PatchBundleDataRecipe",
    "PatchBundleDataset",
]
