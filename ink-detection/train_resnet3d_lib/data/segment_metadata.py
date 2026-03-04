def parse_layer_range_value(layer_range, *, context):
    start_idx, end_idx = [int(v) for v in layer_range]
    if end_idx <= start_idx:
        raise ValueError(f"{context} requires end_idx > start_idx")
    return start_idx, end_idx


def get_segment_meta(segments_metadata, segment_id):
    return dict(segments_metadata[segment_id])


def get_segment_layer_range(seg_meta, segment_id):
    layer_range = seg_meta["layer_range"]
    return parse_layer_range_value(layer_range, context=f"segments[{segment_id!r}].layer_range")


def get_segment_reverse_layers(seg_meta, segment_id):
    reverse_layers = seg_meta["reverse_layers"]
    if not isinstance(reverse_layers, bool):
        raise TypeError(
            f"segments[{segment_id!r}].reverse_layers must be boolean, got {type(reverse_layers).__name__}"
        )
    return reverse_layers
