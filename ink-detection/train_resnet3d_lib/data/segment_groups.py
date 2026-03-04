from train_resnet3d_lib.data.image_readers import build_group_mappings
from train_resnet3d_lib.data.segment_metadata import get_segment_meta as _segment_meta


def build_group_metadata(fragment_ids, segments_metadata, group_key):
    group_names, _group_name_to_idx, fragment_to_group_idx = build_group_mappings(
        fragment_ids,
        segments_metadata,
        group_key=group_key,
    )
    return group_names, fragment_to_group_idx


def segment_group_context(fragment_id, segments_metadata, fragment_to_group_idx, group_names):
    seg_meta = _segment_meta(segments_metadata, fragment_id)
    group_idx = int(fragment_to_group_idx[fragment_id])
    group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)
    return seg_meta, group_idx, group_name


__all__ = [
    "build_group_metadata",
    "segment_group_context",
]
