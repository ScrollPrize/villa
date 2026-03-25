from .default import find_segment_patches as find_default_segment_patches
from .subtiling import (
    build_patch_index,
    find_segment_patches as find_subtiling_segment_patches,
)


def resolve_patch_finding_type(config) -> str:
    patch_finding_type = str(config.get("patch_finding_type", "default")).strip().lower()
    if patch_finding_type not in {"default", "subtiling"}:
        raise ValueError(
            "patch_finding_type must be one of {'default', 'subtiling'}, "
            f"got {config.get('patch_finding_type')!r}"
        )
    return patch_finding_type


def find_segment_patches(segment, patch_cls):
    patch_finding_type = resolve_patch_finding_type(segment.config)
    if patch_finding_type == "default":
        return find_default_segment_patches(segment, patch_cls)
    return find_subtiling_segment_patches(segment, patch_cls)


__all__ = [
    "build_patch_index",
    "find_default_segment_patches",
    "find_segment_patches",
    "find_subtiling_segment_patches",
    "resolve_patch_finding_type",
]
