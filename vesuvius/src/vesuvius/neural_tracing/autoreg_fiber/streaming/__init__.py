"""Streaming inference for the autoregressive fiber tracer.

Top-level surface exposed by this package:

* :class:`ChunkLRUCache` and :func:`open_streaming_volume` — chunk-level LRU
  cache plus thread-pool prefetch in front of a (typically S3-backed) Zarr.
* :class:`WindowedVolumeReader` — slides a fixed-size 128^3 crop across a
  large remote volume, re-anchoring on the trace as it leaves the window.
* (added in later milestones) :class:`FiberTracer` — drives
  :meth:`AutoregFiberModel.init_kv_cache` + :meth:`step_from_encoded_cached`
  over the windowed reader.
"""

from vesuvius.neural_tracing.autoreg_fiber.streaming.chunk_cache import (
    ChunkLRUCache,
    open_streaming_volume,
)
from vesuvius.neural_tracing.autoreg_fiber.streaming.window import WindowedVolumeReader

__all__ = [
    "ChunkLRUCache",
    "WindowedVolumeReader",
    "open_streaming_volume",
]
