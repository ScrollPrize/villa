from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
import zarr

from ink.core.types import DataBundle
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patch_bundle.writer import load_patch_bundle_manifest
from ink.recipes.data.samplers import ShuffleSampler
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)
from ink.recipes.data.zarr_data import (
    build_patch_data_bundle,
    collate_patch_batch,
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
        self._store = store
        self._x = store["x"]
        self._y = store["y"]
        self._valid_mask = store["valid_mask"]
        self._xyxy = store["xyxy"]
        self._segment_index = store["segment_index"]
        self._group_idx = store["group_idx"] if "group_idx" in store else None
        self.transform = build_joint_transform(
            self.split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    @property
    def sample_groups(self) -> list[int]:
        if self._group_idx is None:
            raise ValueError("patch bundle does not provide group_idx")
        return [int(np.asarray(self._group_idx[idx]).item()) for idx in range(len(self))]

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
        if self._group_idx is None:
            return image, label, valid_mask, xyxy, segment_id
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
        return train_manifest, valid_manifest

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

    def build(self, *, runtime=None, augment=None) -> DataBundle:
        assert augment is not None
        train_manifest, valid_manifest = self._load_manifests()
        in_channels, patch_size = self._validate_split_contract(train_manifest, valid_manifest)

        valid_batch_size = self.train_batch_size if self.valid_batch_size is None else self.valid_batch_size
        extras = dict(self.extras or {})
        group_counts = extras.pop("group_counts", None)
        if group_counts is not None:
            group_counts = [int(value) for value in group_counts]
            if not group_counts:
                group_counts = None
        normalization_stats = extras.pop("normalization_stats", None)
        if normalization_stats is None:
            normalization_stats = dict(train_manifest.get("normalization_stats") or {})
        normalization = self.normalization.build(normalization_stats=normalization_stats)
        augment_recipe = augment.build(patch_size=patch_size, runtime=runtime)

        dataset_kwargs = {
            "bundle_root": Path(self.bundle_root).expanduser().resolve(),
            "augment": augment_recipe,
            "normalization": normalization,
        }
        train_dataset = PatchBundleDataset(split="train", **dataset_kwargs)
        valid_dataset = PatchBundleDataset(split="valid", **dataset_kwargs)
        if group_counts is None:
            manifest_group_counts = train_manifest.get("group_counts")
            if manifest_group_counts is not None:
                group_counts = [int(value) for value in manifest_group_counts]
                if not group_counts:
                    group_counts = None
        num_workers = max(0, int(self.num_workers))
        return build_patch_data_bundle(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_batch_size=int(self.train_batch_size),
            valid_batch_size=int(valid_batch_size),
            num_workers=num_workers,
            shuffle=bool(self.shuffle),
            sampler=self.sampler,
            collate_fn=collate_patch_batch,
            in_channels=int(in_channels),
            augment_recipe=augment_recipe,
            group_counts=group_counts,
            extras=extras,
        )


__all__ = [
    "PatchBundleDataRecipe",
    "PatchBundleDataset",
]
