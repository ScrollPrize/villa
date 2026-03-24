from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset
import zarr

from ink.recipes.data.patch_bundle.writer import load_patch_bundle_manifest
from ink.recipes.data.transforms import (
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
    build_joint_transform,
)
from ink.recipes.data.zarr_data import collate_patch_batch
from ink.recipes.stitch.config import StitchSegmentSpec


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
        self._sample_segment_ids = tuple(
            self.segment_ids[int(segment_idx)]
            for segment_idx in np.asarray(self._segment_index[:], dtype=np.int32).tolist()
        )
        self.transform = self._build_transform(self.split)

    @property
    def sample_groups(self) -> list[int]:
        if self._group_idx is None:
            raise ValueError("patch bundle does not provide group_idx")
        return [int(np.asarray(self._group_idx[idx]).item()) for idx in range(len(self))]

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def segment_sample_indices(self, segment_id: str) -> tuple[int, ...]:
        segment_id = str(segment_id)
        return tuple(
            sample_idx
            for sample_idx, sample_segment_id in enumerate(self._sample_segment_ids)
            if sample_segment_id == segment_id
        )

    def build_segment_dataset(self, *, split: str, segment_id: str) -> "_PatchBundleSegmentDataset":
        segment_id = str(segment_id)
        return _PatchBundleSegmentDataset(
            base_dataset=self,
            split=split,
            sample_indices=self.segment_sample_indices(segment_id),
            segment_ids=(segment_id,),
        )

    def build_segment_eval_loaders(
        self,
        *,
        segment_ids,
        batch_size: int,
        num_workers: int = 0,
    ) -> list[DataLoader]:
        requested_segment_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
        return [
            DataLoader(
                self.build_segment_dataset(split="valid", segment_id=segment_id),
                batch_size=max(1, int(batch_size)),
                shuffle=False,
                num_workers=max(0, int(num_workers)),
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_patch_batch,
            )
            for segment_id in requested_segment_ids
        ]

    def build_segment_infer_loaders(
        self,
        *,
        segment_ids,
        batch_size: int,
        num_workers: int = 0,
    ) -> list[DataLoader]:
        del segment_ids, batch_size, num_workers
        raise ValueError("patch bundle stitch runtime does not support log_only loaders")

    def stitch_segment_specs(
        self,
        *,
        segment_ids,
        downsample: int,
        mode: str,
        use_roi: bool,
    ) -> list[StitchSegmentSpec]:
        mode_name = str(mode).strip().lower()
        if mode_name == "log_only":
            raise ValueError("patch bundle stitch runtime does not support log_only segment derivation")
        if int(downsample) != 1:
            raise ValueError("patch bundle stitch segment specs currently require downsample=1")
        raw_specs = self.manifest.get("segment_specs")
        if not isinstance(raw_specs, list):
            raise ValueError("patch bundle manifest is missing segment_specs")
        specs_by_id = {}
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, dict):
                continue
            segment_id = raw_spec.get("segment_id")
            shape = raw_spec.get("shape")
            if segment_id is None or shape is None:
                continue
            bbox = None
            if bool(use_roi) and raw_spec.get("bbox") is not None:
                bbox = tuple(tuple(int(value) for value in row) for row in raw_spec.get("bbox") or ())
            specs_by_id[str(segment_id)] = StitchSegmentSpec(
                segment_id=str(segment_id),
                shape=tuple(int(value) for value in shape),
                bbox=bbox,
            )
        requested_segment_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
        missing_segment_ids = [segment_id for segment_id in requested_segment_ids if segment_id not in specs_by_id]
        if missing_segment_ids:
            raise ValueError(f"patch bundle manifest is missing segment_specs for {missing_segment_ids}")
        return [specs_by_id[segment_id] for segment_id in requested_segment_ids]

    def _raw_item(self, idx: int):
        idx = int(idx)
        image = np.asarray(self._x[idx], dtype=np.uint8)
        label = np.asarray(self._y[idx], dtype=np.uint8)
        valid_mask = np.asarray(self._valid_mask[idx], dtype=np.uint8)
        xyxy = np.asarray(self._xyxy[idx], dtype=np.int64)
        segment_idx = int(np.asarray(self._segment_index[idx]).item())
        segment_id = self.segment_ids[segment_idx]
        return image, label, valid_mask, xyxy, segment_id

    def _build_transform(self, split: str):
        return build_joint_transform(
            split,
            augment=self.augment,
            normalization=self.normalization,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
        )

    def _apply_transforms(self, *, split: str, transform, image, label, valid_mask):
        if split == "train":
            return apply_train_sample_transforms(
                image,
                label,
                augment=self.augment,
                patch_size=self.patch_size,
                transform=transform,
                valid_mask=valid_mask,
            )
        return apply_eval_sample_transforms(
            image,
            label,
            patch_size=self.patch_size,
            transform=transform,
            valid_mask=valid_mask,
        )

    def _group_value(self, idx: int) -> int | None:
        if self._group_idx is None:
            return None
        return int(np.asarray(self._group_idx[int(idx)]).item())

    def _item_for_split(self, idx: int, *, split: str, transform):
        image, label, valid_mask, xyxy, segment_id = self._raw_item(int(idx))
        image, label, valid_mask = self._apply_transforms(
            split=split,
            transform=transform,
            image=image,
            label=label,
            valid_mask=valid_mask,
        )
        group_idx = self._group_value(int(idx))
        if group_idx is None:
            return image, label, valid_mask, xyxy, segment_id
        return image, label, valid_mask, xyxy, segment_id, group_idx

    def __getitem__(self, idx):
        return self._item_for_split(int(idx), split=self.split, transform=self.transform)


class _PatchBundleSegmentDataset(Dataset):
    def __init__(self, *, base_dataset: PatchBundleDataset, split: str, sample_indices, segment_ids=()):
        self.base_dataset = base_dataset
        self.split = str(split).strip().lower()
        if self.split not in {"train", "valid"}:
            raise ValueError(f"unknown split: {split!r}")
        self.augment = base_dataset.augment
        self.normalization = base_dataset.normalization
        self.patch_size = int(base_dataset.patch_size)
        self.in_channels = int(base_dataset.in_channels)
        self.sample_indices = tuple(int(sample_idx) for sample_idx in sample_indices)
        self.segment_ids = tuple(str(segment_id) for segment_id in (segment_ids or ()))
        self.transform = base_dataset._build_transform(self.split)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx):
        source_idx = int(self.sample_indices[int(idx)])
        return self.base_dataset._item_for_split(source_idx, split=self.split, transform=self.transform)


__all__ = ["PatchBundleDataset"]
