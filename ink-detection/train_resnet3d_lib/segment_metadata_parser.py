def get_segment_meta(segments_metadata, segment_id):
    if segment_id not in segments_metadata:
        raise KeyError(f"segments metadata missing fragment id: {segment_id!r}")
    seg_meta = segments_metadata[segment_id]
    if not isinstance(seg_meta, dict):
        raise TypeError(f"segments[{segment_id!r}] must be an object, got {type(seg_meta).__name__}")
    return seg_meta


def get_segment_layer_range(seg_meta, segment_id):
    if "layer_range" not in seg_meta:
        raise KeyError(f"segments[{segment_id!r}] missing required key: 'layer_range'")
    layer_range = seg_meta["layer_range"]
    if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
        raise TypeError(
            f"segments[{segment_id!r}].layer_range must be [start_idx, end_idx], got {layer_range!r}"
        )
    start_idx = int(layer_range[0])
    end_idx = int(layer_range[1])
    if end_idx <= start_idx:
        raise ValueError(
            f"segments[{segment_id!r}].layer_range must satisfy end_idx > start_idx, got {layer_range!r}"
        )
    return start_idx, end_idx


def get_segment_reverse_layers(seg_meta, segment_id):
    if "reverse_layers" not in seg_meta:
        raise KeyError(f"segments[{segment_id!r}] missing required key: 'reverse_layers'")
    reverse_layers = seg_meta["reverse_layers"]
    if not isinstance(reverse_layers, bool):
        raise TypeError(
            f"segments[{segment_id!r}].reverse_layers must be boolean, got {type(reverse_layers).__name__}"
        )
    return reverse_layers
