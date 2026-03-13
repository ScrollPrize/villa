import numpy as np
from torch.utils.data import DataLoader, Dataset

from train_resnet3d_lib.config import CFG
from train_resnet3d_lib.data.augmentations import (
    _apply_image_transform,
    _xy_to_bounds,
    apply_eval_sample_transforms,
    apply_train_sample_transforms,
)
from train_resnet3d_lib.data.patching import _read_mask_patch

SUPPORTED_DATA_BACKENDS = ("zarr", "tiff")


def normalize_data_backend(data_backend):
    normalized = str(data_backend).strip().lower()
    if normalized not in SUPPORTED_DATA_BACKENDS:
        raise ValueError(
            f"Unknown training.data_backend: {data_backend!r}. "
            f"Expected one of {SUPPORTED_DATA_BACKENDS!r}."
        )
    return normalized


def build_eval_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_stitch_train_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=int(getattr(CFG, "stitch_patch_batch_size", getattr(CFG, "valid_batch_size", 1))),
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _first_item(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) != 1:
        raise ValueError(f"expected a single-item batch, got {type(batch).__name__} len={len(batch) if hasattr(batch, '__len__') else 'n/a'}")
    return batch[0]


def _identity_batch(batch):
    return list(batch)


class SegmentIdDataset(Dataset):
    def __init__(self, segment_ids):
        self.segment_ids = [str(segment_id) for segment_id in (segment_ids or [])]
        if not self.segment_ids:
            raise ValueError("SegmentIdDataset requires at least one segment id")

    def __len__(self):
        return len(self.segment_ids)

    def __getitem__(self, idx):
        return self.segment_ids[int(idx)]


class ComponentKeyDataset(Dataset):
    def __init__(self, component_keys):
        self.component_keys = []
        for item in (component_keys or []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"component key must be a pair (segment_id, component_idx), got {item!r}")
            self.component_keys.append((str(item[0]), int(item[1])))
        if not self.component_keys:
            raise ValueError("ComponentKeyDataset requires at least one component key")

    def __len__(self):
        return len(self.component_keys)

    def __getitem__(self, idx):
        return self.component_keys[int(idx)]


def build_segment_id_train_loader(segment_ids):
    return DataLoader(
        SegmentIdDataset(segment_ids),
        batch_size=1,
        shuffle=str(getattr(CFG, "sampler", "shuffle")).strip().lower() == "shuffle",
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=_first_item,
    )


def build_component_id_train_loader(component_keys):
    return DataLoader(
        ComponentKeyDataset(component_keys),
        batch_size=int(getattr(CFG, "train_batch_size", 1)),
        shuffle=str(getattr(CFG, "sampler", "shuffle")).strip().lower() == "shuffle",
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=_identity_batch,
    )


def _require_dict(value, *, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def _flatten_segment_patch_index(
    xyxys_by_segment,
    groups_by_segment=None,
    sample_bbox_indices_by_segment=None,
):
    xyxys_by_segment = _require_dict(xyxys_by_segment, name="xyxys_by_segment")
    if groups_by_segment is not None:
        groups_by_segment = _require_dict(groups_by_segment, name="groups_by_segment")
    if sample_bbox_indices_by_segment is not None:
        sample_bbox_indices_by_segment = _require_dict(
            sample_bbox_indices_by_segment,
            name="sample_bbox_indices_by_segment",
        )

    segment_ids = []
    seg_indices = []
    xy_chunks = []
    group_chunks = []
    bbox_idx_chunks = []

    for segment_id, xyxys in xyxys_by_segment.items():
        xy = np.asarray(xyxys, dtype=np.int64)
        if xy.ndim != 2 or xy.shape[1] != 4:
            raise ValueError(
                f"{segment_id}: expected xyxys shape (N, 4), got {tuple(xy.shape)}"
            )
        if xy.shape[0] == 0:
            continue

        seg_idx = len(segment_ids)
        segment_ids.append(str(segment_id))
        xy_chunks.append(xy)
        seg_indices.append(np.full((xy.shape[0],), seg_idx, dtype=np.int32))
        if sample_bbox_indices_by_segment is not None:
            seg_id_key = str(segment_id)
            if seg_id_key not in sample_bbox_indices_by_segment:
                raise KeyError(f"sample_bbox_indices_by_segment missing segment id: {seg_id_key!r}")
            bbox_idx = np.asarray(sample_bbox_indices_by_segment[seg_id_key], dtype=np.int32).reshape(-1)
            if bbox_idx.shape[0] != xy.shape[0]:
                raise ValueError(
                    f"{segment_id}: sample_bbox_indices length {bbox_idx.shape[0]} "
                    f"does not match xyxys length {xy.shape[0]}"
                )
            bbox_idx_chunks.append(bbox_idx)
        if groups_by_segment is not None:
            seg_id_key = str(segment_id)
            if seg_id_key not in groups_by_segment:
                raise KeyError(f"groups_by_segment missing segment id: {seg_id_key!r}")
            group_id = int(groups_by_segment[seg_id_key])
            group_chunks.append(np.full((xy.shape[0],), group_id, dtype=np.int64))

    if len(segment_ids) == 0:
        empty_xy = np.zeros((0, 4), dtype=np.int64)
        empty_seg = np.zeros((0,), dtype=np.int32)
        empty_bbox_idx = np.zeros((0,), dtype=np.int32) if sample_bbox_indices_by_segment is not None else None
        if groups_by_segment is None:
            return [], empty_seg, empty_xy, None, empty_bbox_idx
        return [], empty_seg, empty_xy, np.zeros((0,), dtype=np.int64), empty_bbox_idx

    flat_xy = np.concatenate(xy_chunks, axis=0)
    flat_seg = np.concatenate(seg_indices, axis=0)
    flat_bbox_idx = None
    if sample_bbox_indices_by_segment is not None:
        flat_bbox_idx = np.concatenate(bbox_idx_chunks, axis=0)
    if groups_by_segment is None:
        return segment_ids, flat_seg, flat_xy, None, flat_bbox_idx
    flat_groups = np.concatenate(group_chunks, axis=0)
    return segment_ids, flat_seg, flat_xy, flat_groups, flat_bbox_idx


def _init_flat_segment_index(
    xyxys_by_segment,
    groups_by_segment,
    dataset_name,
    *,
    sample_bbox_indices_by_segment=None,
):
    segment_ids, sample_segment_indices, sample_xyxys, sample_groups, sample_bbox_indices = _flatten_segment_patch_index(
        xyxys_by_segment,
        groups_by_segment,
        sample_bbox_indices_by_segment=sample_bbox_indices_by_segment,
    )
    if sample_xyxys.shape[0] == 0:
        raise ValueError(f"{dataset_name} has no samples")
    return segment_ids, sample_segment_indices, sample_xyxys, sample_groups, sample_bbox_indices


def _validate_segment_data(segment_ids, volumes, masks=None):
    for segment_id in segment_ids:
        if segment_id not in volumes:
            raise ValueError(f"Missing volume for segment={segment_id}")
        if masks is not None and segment_id not in masks:
            raise ValueError(f"Missing mask for segment={segment_id}")


def _group_id_for_index(groups, idx):
    if groups is None:
        return 0
    return int(groups[idx])


def _prepare_xy_label_group_sample(image, label, xy, group_id, *, transform, cfg):
    image, label = apply_eval_sample_transforms(
        image,
        label,
        transform=transform,
        cfg=cfg,
    )
    return image, label, xy, int(group_id)


class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, groups=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.groups = groups

        self.transform = transform
        self.xyxys = xyxys

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        group_id = _group_id_for_index(self.groups, idx)
        if self.xyxys is not None:
            return _prepare_xy_label_group_sample(
                self.images[idx],
                self.labels[idx],
                self.xyxys[idx],
                group_id,
                transform=self.transform,
                cfg=self.cfg,
            )
        image = self.images[idx]
        label = self.labels[idx]
        image, label = apply_train_sample_transforms(
            image,
            label,
            transform=self.transform,
            cfg=self.cfg,
        )
        return image, label, group_id


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        image = _apply_image_transform(self.transform, image)
        return image, xy


class LazyZarrTrainDataset(Dataset):
    def __init__(
        self,
        volumes_by_segment,
        masks_by_segment,
        xyxys_by_segment,
        groups_by_segment,
        cfg,
        transform=None,
        sample_bbox_indices_by_segment=None,
    ):
        self.volumes = dict(_require_dict(volumes_by_segment, name="volumes_by_segment"))
        self.masks = dict(_require_dict(masks_by_segment, name="masks_by_segment"))
        self.cfg = cfg
        self.transform = transform

        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            self.sample_groups,
            self.sample_bbox_indices,
        ) = _init_flat_segment_index(
            xyxys_by_segment,
            groups_by_segment,
            "LazyZarrTrainDataset",
            sample_bbox_indices_by_segment=sample_bbox_indices_by_segment,
        )
        _validate_segment_data(self.segment_ids, self.volumes, self.masks)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        x1, y1, x2, y2 = _xy_to_bounds(self.sample_xyxys[idx])
        group_id = int(self.sample_groups[idx])
        bbox_idx = int(self.sample_bbox_indices[idx]) if self.sample_bbox_indices is not None else None

        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        label = _read_mask_patch(self.masks[segment_id], y1=y1, y2=y2, x1=x1, x2=x2, bbox_index=bbox_idx)[..., None]
        image, label = apply_train_sample_transforms(
            image,
            label,
            transform=self.transform,
            cfg=self.cfg,
        )

        return image, label, group_id


class LazyZarrXyLabelDataset(Dataset):
    def __init__(
        self,
        volumes_by_segment,
        masks_by_segment,
        xyxys_by_segment,
        groups_by_segment,
        cfg,
        transform=None,
        sample_bbox_indices_by_segment=None,
    ):
        self.volumes = dict(_require_dict(volumes_by_segment, name="volumes_by_segment"))
        self.masks = dict(_require_dict(masks_by_segment, name="masks_by_segment"))
        self.cfg = cfg
        self.transform = transform
        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            self.sample_groups,
            self.sample_bbox_indices,
        ) = _init_flat_segment_index(
            xyxys_by_segment,
            groups_by_segment,
            "LazyZarrXyLabelDataset",
            sample_bbox_indices_by_segment=sample_bbox_indices_by_segment,
        )
        _validate_segment_data(self.segment_ids, self.volumes, self.masks)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        xy = self.sample_xyxys[idx]
        x1, y1, x2, y2 = _xy_to_bounds(xy)
        group_id = _group_id_for_index(self.sample_groups, idx)
        bbox_idx = int(self.sample_bbox_indices[idx]) if self.sample_bbox_indices is not None else None

        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        label = _read_mask_patch(self.masks[segment_id], y1=y1, y2=y2, x1=x1, x2=x2, bbox_index=bbox_idx)[..., None]
        return _prepare_xy_label_group_sample(
            image,
            label,
            xy,
            group_id,
            transform=self.transform,
            cfg=self.cfg,
        )


class LazyZarrXyOnlyDataset(Dataset):
    def __init__(
        self,
        volumes_by_segment,
        xyxys_by_segment,
        cfg,
        transform=None,
    ):
        self.volumes = dict(_require_dict(volumes_by_segment, name="volumes_by_segment"))
        self.cfg = cfg
        self.transform = transform
        (
            self.segment_ids,
            self.sample_segment_indices,
            self.sample_xyxys,
            _,
            _,
        ) = _init_flat_segment_index(
            xyxys_by_segment,
            groups_by_segment=None,
            dataset_name="LazyZarrXyOnlyDataset",
            sample_bbox_indices_by_segment=None,
        )
        _validate_segment_data(self.segment_ids, self.volumes)

    def __len__(self):
        return int(self.sample_xyxys.shape[0])

    def __getitem__(self, idx):
        idx = int(idx)
        seg_idx = int(self.sample_segment_indices[idx])
        segment_id = self.segment_ids[seg_idx]
        xy = self.sample_xyxys[idx]
        x1, y1, x2, y2 = _xy_to_bounds(xy)
        image = self.volumes[segment_id].read_patch(y1, y2, x1, x2)
        image = _apply_image_transform(self.transform, image)
        return image, xy
