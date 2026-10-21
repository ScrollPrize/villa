"""CPU-only sample partitioning and native worker cache initialization."""

from dataclasses import dataclass


def partition_offsets(effective_batch_size, weights):
    if type(effective_batch_size) is not int or effective_batch_size < 1 or not weights:
        raise ValueError("partition requires a positive effective batch and weights")
    if any(type(value) is not int or value < 1 for value in weights):
        raise ValueError("partition weights must be positive integers")
    total = sum(weights)
    prefix = 0
    offsets = [0]
    for value in weights:
        prefix += value
        offsets.append(effective_batch_size * prefix // total)
    return tuple(offsets)


@dataclass(frozen=True)
class EffectiveBatchSampleIndex:
    """Rank-local consecutive sample stream, independent of physical batches."""

    effective_batch_size: int
    microbatch_sizes: tuple[int, ...]
    rank: int
    start_step: int = 0

    def __post_init__(self):
        partition_offsets(self.effective_batch_size, self.microbatch_sizes)
        if type(self.rank) is not int or not 0 <= self.rank < len(self.microbatch_sizes):
            raise ValueError("invalid partition rank")
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("invalid start step")

    @property
    def bounds(self):
        offsets = partition_offsets(self.effective_batch_size, self.microbatch_sizes)
        return offsets[self.rank:self.rank + 2]

    @property
    def local_samples(self):
        start, stop = self.bounds
        return stop - start

    @property
    def batch_sizes(self):
        size = self.microbatch_sizes[self.rank]
        return tuple(min(size, self.local_samples - start)
                     for start in range(0, self.local_samples, size))

    def __call__(self, index):
        if type(index) is not int or index < 0 or not self.local_samples:
            raise ValueError("invalid local sample index (or zero-work rank)")
        step, offset = divmod(index, self.local_samples)
        return (self.start_step + step) * self.effective_batch_size + self.bounds[0] + offset

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
