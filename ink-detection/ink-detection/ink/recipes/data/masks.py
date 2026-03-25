from __future__ import annotations

SUPERVISION_MASK_NAME = "supervision_mask"
VALIDATION_MASK_NAME = "validation_mask"


def normalize_mask_names(*, mask_name: str = SUPERVISION_MASK_NAME, mask_names=None) -> tuple[str, ...]:
    if mask_names is None:
        mask_names = (mask_name,)
    normalized = tuple(str(name).strip() for name in tuple(mask_names))
    normalized = tuple(name for name in normalized if name)
    if not normalized:
        raise ValueError("mask_names must include at least one non-empty mask name")
    return normalized


def default_mask_name_for_split(split_name: str) -> str:
    split = str(split_name).strip().lower()
    if split == "valid":
        return VALIDATION_MASK_NAME
    return SUPERVISION_MASK_NAME


def resolve_segment_mask_names(
    *,
    split_name: str,
    segment_id: str,
    train_segment_ids=(),
    default_mask_name: str = SUPERVISION_MASK_NAME,
    available_mask_names=None,
) -> tuple[str, ...]:
    default_mask_names = normalize_mask_names(mask_name=default_mask_name)
    split = str(split_name).strip().lower()
    if split != "valid":
        return default_mask_names

    if isinstance(train_segment_ids, frozenset):
        train_ids = train_segment_ids
    else:
        train_ids = frozenset(str(current_segment_id) for current_segment_id in train_segment_ids)
    if not train_ids:
        return default_mask_names
    if str(segment_id) in train_ids:
        return (VALIDATION_MASK_NAME,)
    preferred_mask_names = (SUPERVISION_MASK_NAME, VALIDATION_MASK_NAME)
    normalized_available_mask_names = tuple(
        str(mask_name).strip()
        for mask_name in tuple(available_mask_names or ())
        if str(mask_name).strip()
    )
    if not normalized_available_mask_names:
        return preferred_mask_names
    available_mask_name_set = frozenset(normalized_available_mask_names)
    selected_mask_names = tuple(
        mask_name for mask_name in preferred_mask_names if mask_name in available_mask_name_set
    )
    return selected_mask_names or preferred_mask_names


def resolve_stitch_roi_mask_names(*, available_mask_names=None) -> tuple[str, ...]:
    preferred_mask_names = (SUPERVISION_MASK_NAME, VALIDATION_MASK_NAME)
    normalized_available_mask_names = tuple(
        str(mask_name).strip()
        for mask_name in tuple(available_mask_names or ())
        if str(mask_name).strip()
    )
    if not normalized_available_mask_names:
        return preferred_mask_names
    available_mask_name_set = frozenset(normalized_available_mask_names)
    selected_mask_names = tuple(
        mask_name for mask_name in preferred_mask_names if mask_name in available_mask_name_set
    )
    return selected_mask_names or preferred_mask_names


__all__ = [
    "SUPERVISION_MASK_NAME",
    "VALIDATION_MASK_NAME",
    "default_mask_name_for_split",
    "normalize_mask_names",
    "resolve_segment_mask_names",
    "resolve_stitch_roi_mask_names",
]
