from __future__ import annotations

from itertools import product
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


def point_to_patch_coordinates(
    coordinates_zyx: torch.Tensor,
    patch_size: Sequence[int],
    feature_map_shape: Sequence[int],
) -> torch.Tensor:
    """Convert output-voxel centers to continuous patch-token grid coordinates."""
    if coordinates_zyx.ndim != 2 or coordinates_zyx.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point coordinates, got {tuple(coordinates_zyx.shape)}.")
    patch = coordinates_zyx.new_tensor(tuple(float(value) for value in patch_size))
    maximum = coordinates_zyx.new_tensor(tuple(float(value - 1) for value in feature_map_shape))
    patch_coordinates = (coordinates_zyx + 0.5) / patch - 0.5
    return torch.minimum(torch.maximum(patch_coordinates, torch.zeros_like(patch_coordinates)), maximum)


def interpolation_support_indices(
    coordinates_zyx: torch.Tensor,
    patch_size: Sequence[int],
    feature_map_shape: Sequence[int],
) -> torch.Tensor:
    """Return the (up to eight) flattened tokens supporting every point."""
    if coordinates_zyx.numel() == 0:
        return torch.empty((0, 8), dtype=torch.long, device=coordinates_zyx.device)
    patch_coordinates = point_to_patch_coordinates(coordinates_zyx, patch_size, feature_map_shape)
    lower = torch.floor(patch_coordinates).long()
    upper = torch.minimum(
        lower + 1,
        lower.new_tensor(tuple(int(value - 1) for value in feature_map_shape)),
    )
    depth, height, width = (int(value) for value in feature_map_shape)
    del depth
    support = []
    for use_upper in product((False, True), repeat=3):
        index = torch.stack(
            [upper[:, axis] if use_upper[axis] else lower[:, axis] for axis in range(3)],
            dim=1,
        )
        support.append(index[:, 0] * (height * width) + index[:, 1] * width + index[:, 2])
    return torch.stack(support, dim=1)


def sample_normalized_patch_embeddings(
    patch_tokens: torch.Tensor,
    row_indices: torch.Tensor,
    coordinates_zyx: torch.Tensor,
    patch_size: Sequence[int],
    feature_map_shape: Sequence[int],
) -> torch.Tensor:
    """Trilinearly sample a normalized patch-token grid without duplicating grids."""
    if coordinates_zyx.numel() == 0:
        return patch_tokens.reshape(-1, patch_tokens.shape[-1])[:0]
    if row_indices.ndim != 1 or row_indices.shape[0] != coordinates_zyx.shape[0]:
        raise ValueError("Point row indices must be one-dimensional and match the point count.")

    patch_coordinates = point_to_patch_coordinates(coordinates_zyx, patch_size, feature_map_shape)
    lower = torch.floor(patch_coordinates).long()
    upper = torch.minimum(
        lower + 1,
        lower.new_tensor(tuple(int(value - 1) for value in feature_map_shape)),
    )
    fraction = patch_coordinates - lower.to(patch_coordinates.dtype)
    normalized_tokens = F.normalize(patch_tokens.float(), dim=-1)
    height, width = int(feature_map_shape[1]), int(feature_map_shape[2])
    result = normalized_tokens.new_zeros((coordinates_zyx.shape[0], patch_tokens.shape[-1]))
    for use_upper in product((False, True), repeat=3):
        index = torch.stack(
            [upper[:, axis] if use_upper[axis] else lower[:, axis] for axis in range(3)],
            dim=1,
        )
        flat_index = index[:, 0] * (height * width) + index[:, 1] * width + index[:, 2]
        weight = torch.ones_like(fraction[:, 0])
        for axis in range(3):
            axis_weight = fraction[:, axis] if use_upper[axis] else 1.0 - fraction[:, axis]
            weight = weight * axis_weight
        result = result + normalized_tokens[row_indices, flat_index] * weight[:, None]
    return result


def gather_variable_points(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-aware all-gather for uneven point counts."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return embeddings, labels
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Point embeddings and labels must have matching leading dimensions.")

    world_size = dist.get_world_size()
    local_count = torch.tensor([embeddings.shape[0]], device=embeddings.device, dtype=torch.long)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, local_count)
    counts = [int(count.item()) for count in gathered_counts]
    padded_count = max(max(counts), 1)

    embedding_padding = embeddings.new_zeros((padded_count - embeddings.shape[0], embeddings.shape[1]))
    padded_embeddings = torch.cat((embeddings, embedding_padding), dim=0)
    label_padding = labels.new_full((padded_count - labels.shape[0],), -1)
    padded_labels = torch.cat((labels, label_padding), dim=0)

    from torch.distributed.nn.functional import all_gather as differentiable_all_gather

    gathered_embeddings = differentiable_all_gather(padded_embeddings)
    gathered_labels = [torch.empty_like(padded_labels) for _ in range(world_size)]
    dist.all_gather(gathered_labels, padded_labels)
    return (
        torch.cat([rank_embeddings[:count] for rank_embeddings, count in zip(gathered_embeddings, counts)]),
        torch.cat([rank_labels[:count] for rank_labels, count in zip(gathered_labels, counts)]),
    )
