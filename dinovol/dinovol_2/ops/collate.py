# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol's three-dimensional training pipeline.

from __future__ import annotations

from typing import Any, Mapping, NotRequired, TypedDict

import torch


class MultiCropBatch(TypedDict):
    """Views stacked as [views * batch, C, D, H, W], independently by resolution.

    Row = view_index * batch_size + sample_index. Global and local rows with
    the same sample_index come from the same source. Zero locals is supported;
    consumers must use n_local_views rather than the empty tensor's spatial shape.
    Masks are deliberately not part of this contract. Optional annotations keep
    their original geometry and the same global row ordering.
    """
    collated_global_crops: torch.Tensor
    collated_local_crops: torch.Tensor
    n_global_views: int
    n_local_views: int
    batch_size: int
    collated_point_coordinates: NotRequired[torch.Tensor]
    collated_point_type_ids: NotRequired[torch.Tensor]
    collated_point_rows: NotRequired[torch.Tensor]
    collated_gram_teacher_crops: NotRequired[torch.Tensor]
    global_token_targets: NotRequired[torch.Tensor]
    object_point_coordinates: NotRequired[torch.Tensor]
    object_point_valid: NotRequired[torch.Tensor]
    object_point_source: NotRequired[torch.Tensor]
    object_point_objects: NotRequired[torch.Tensor]
    object_point_kinds: NotRequired[torch.Tensor]
    object_negatives: NotRequired[torch.Tensor]
    object_point_ink: NotRequired[torch.Tensor]


def collate_multicrop_batch(samples: list[Mapping[str, Any]], *, dtype=torch.float32) -> MultiCropBatch:
    """Stack crops and annotations without imposing a training objective."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    n_global = len(samples[0]["global_views"])
    n_local = len(samples[0]["local_views"])
    if n_global < 1:
        raise ValueError("At least one global view is required")
    for sample in samples:
        if len(sample["global_views"]) != n_global or len(sample["local_views"]) != n_local:
            raise ValueError("Samples in a batch must have the same global/local view counts")
        if bool(sample.get("gram_teacher_views")) != bool(samples[0].get("gram_teacher_views")):
            raise ValueError("Auxiliary views must be present for every sample or none")
        if sample.get("gram_teacher_views") and len(sample["gram_teacher_views"]) != n_global:
            raise ValueError("Auxiliary Gram views must correspond to the global views")
    n_global_views = len(samples[0]["global_views"])
    n_local_views = len(samples[0]["local_views"])

    global_crops = torch.stack(
        [sample["global_views"][i] for i in range(n_global_views) for sample in samples]
    ).to(dtype)

    if n_local_views:
        local_crops = torch.stack(
            [sample["local_views"][i] for i in range(n_local_views) for sample in samples]
        ).to(dtype)
    else:
        local_shape = samples[0]["global_views"][0].shape
        local_crops = global_crops.new_empty((0, *local_shape), dtype=dtype)

    if samples[0].get("gram_teacher_views"):
        collated_gram_teacher_crops = torch.stack(
            [sample["gram_teacher_views"][i] for i in range(n_global_views) for sample in samples]
        ).to(dtype)
    else:
        collated_gram_teacher_crops = None

    point_coordinates_parts: list[torch.Tensor] = []
    point_type_parts: list[torch.Tensor] = []
    point_row_parts: list[torch.Tensor] = []
    for view_index in range(n_global_views):
        for sample_index, sample in enumerate(samples):
            coordinates_by_view = sample.get("global_point_coordinates")
            types_by_view = sample.get("global_point_type_ids")
            if coordinates_by_view is None:
                coordinates = torch.empty((0, 3), dtype=torch.float32)
                type_ids = torch.empty((0,), dtype=torch.long)
            else:
                coordinates = torch.as_tensor(coordinates_by_view[view_index], dtype=torch.float32)
                type_ids = torch.as_tensor(types_by_view[view_index], dtype=torch.long)
            if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or type_ids.shape != (coordinates.shape[0],):
                raise ValueError("Each global view must provide matching Nx3 point coordinates and N type IDs.")
            if coordinates.shape[0]:
                row_index = view_index * len(samples) + sample_index
                point_coordinates_parts.append(coordinates)
                point_type_parts.append(type_ids)
                point_row_parts.append(torch.full((coordinates.shape[0],), row_index, dtype=torch.long))

    if point_coordinates_parts:
        collated_point_coordinates = torch.cat(point_coordinates_parts)
        collated_point_type_ids = torch.cat(point_type_parts)
        collated_point_rows = torch.cat(point_row_parts)
    else:
        collated_point_coordinates = torch.empty((0, 3), dtype=torch.float32)
        collated_point_type_ids = torch.empty((0,), dtype=torch.long)
        collated_point_rows = torch.empty((0,), dtype=torch.long)

    batch = {
        "collated_global_crops": global_crops,
        "collated_local_crops": local_crops,
        "n_global_views": n_global_views,
        "n_local_views": n_local_views,
        "batch_size": len(samples),
        "collated_point_coordinates": collated_point_coordinates,
        "collated_point_type_ids": collated_point_type_ids,
        "collated_point_rows": collated_point_rows,
    }
    if collated_gram_teacher_crops is not None:
        batch["collated_gram_teacher_crops"] = collated_gram_teacher_crops
    if "global_token_targets" in samples[0]:
        # Each token's center in the other global view, rows stacked like the global crops.
        batch["global_token_targets"] = torch.stack(
            [sample["global_token_targets"][i] for i in range(n_global_views) for sample in samples]
        )
    if "object_point_coordinates" in samples[0]:
        # Views are stacked like the global crops: row = view_index * batch + sample.
        batch["object_point_coordinates"] = torch.stack(
            [sample["object_point_coordinates"][i] for i in range(n_global_views) for sample in samples]
        )
        batch["object_point_valid"] = torch.stack(
            [sample["object_point_valid"][i] for i in range(n_global_views) for sample in samples]
        )
        for key in ("object_point_source", "object_point_objects", "object_point_kinds", "object_negatives"):
            batch[key] = torch.stack([sample[key] for sample in samples])
        if "object_point_ink" in samples[0]:
            batch["object_point_ink"] = torch.stack([sample["object_point_ink"] for sample in samples])
    return batch


def collate_dino_ibot_batch(samples, **kwargs):
    """Compatibility import; DINO batch preparation lives with the method."""
    from dinovol_2.methods.dino_data import collate_dino_ibot_batch as collate
    return collate(samples, **kwargs)


def build_dino_ibot_collate_fn(config):
    """Compatibility import for existing readers and scripts."""
    from dinovol_2.methods.dino_data import build_dino_ibot_collate_fn as build
    return build(config)
