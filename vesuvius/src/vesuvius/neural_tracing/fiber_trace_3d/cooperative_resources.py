"""Native worker cache initialization shared with cooperative training."""

def initialize_worker_volume_cache(raw):
    """Initialize actual native worker counts before the first cache operation."""
    if "datasets" in raw and not raw["datasets"]:
        return
    from vc import volume
    initialize = getattr(volume, "initialize_chunk_cache", None)
    if initialize is None:
        raise RuntimeError("cooperative loading needs rebuilt vc.volume with initialize_chunk_cache")
    from .loader import _parse_volume_cache_memory_mib
    mib = _parse_volume_cache_memory_mib(raw.get("volume_cache_memory_mib"))
    initialize(int(mib * 1024**2), 1, 1, 1)
