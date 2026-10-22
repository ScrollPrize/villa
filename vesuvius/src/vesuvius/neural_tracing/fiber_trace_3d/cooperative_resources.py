"""CPU-only resource planning for cooperative Fiber training.

No Torch import: this module is safe in a supervisor that must never own CUDA.
Memory observations must include initialization, optimizer, preparation and test
peaks, not just activations. No estimate is supplied implicitly.
"""

from dataclasses import dataclass


def configure_cuda_memory_budget(allowance, reserve, device=0):
    """Worker-only allocator limit, shared by production and calibration.

    Leave reserve outside PyTorch for the context/NCCL. Driver-visible usage is
    checked by the supervisor too; PyTorch's limit alone is not a driver quota.
    """
    if type(allowance) is not int or type(reserve) is not int or not 0 <= reserve < allowance:
        raise ValueError("VRAM grant must exceed its nonnegative driver reserve")
    import torch
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(min(1.0, (allowance - reserve) / properties.total_memory), device)
    return properties


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
    group_size: int = 1

    def __post_init__(self):
        partition_offsets(self.effective_batch_size, self.microbatch_sizes)
        if (type(self.group_size) is not int or self.group_size < 1 or
                self.effective_batch_size % self.group_size or
                any(s % self.group_size for s in self.microbatch_sizes)):
            raise ValueError("effective batch and microbatches must contain whole sample groups")
        if type(self.rank) is not int or not 0 <= self.rank < len(self.microbatch_sizes):
            raise ValueError("invalid partition rank")
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("invalid start step")

    @property
    def bounds(self):
        offsets = tuple(v * self.group_size for v in
                        partition_offsets(self.effective_batch_size // self.group_size, self.microbatch_sizes))
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


@dataclass(frozen=True)
class MemoryCalibration:
    fixed_bytes: int
    bytes_per_sample: int
    preparation_bytes_per_sample: int
    preparation_depth: int
    reserve_bytes: int
    max_microbatch: int
    # Measured sizes bound admissible interpolation. Never extrapolate above them.
    measured_peaks: tuple[tuple[int, int], ...]

    def __post_init__(self):
        for name in ("fixed_bytes", "bytes_per_sample", "preparation_bytes_per_sample",
                     "preparation_depth", "reserve_bytes", "max_microbatch"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_microbatch < 1 or not self.measured_peaks:
            raise ValueError("memory calibration needs measured microbatch peaks")
        previous = 0
        for size, peak in self.measured_peaks:
            if type(size) is not int or type(peak) is not int or size <= previous or peak <= 0:
                raise ValueError("measured peaks must have increasing positive batch sizes")
            previous = size
            if self.estimated_bytes(size) < peak + self.reserve_bytes:
                raise ValueError("memory estimate underestimates a measured peak")
        if self.max_microbatch > previous:
            raise ValueError("cannot extrapolate beyond measured microbatch range")

    def estimated_bytes(self, batch_size):
        return (self.fixed_bytes + self.bytes_per_sample * batch_size
                + self.preparation_bytes_per_sample * batch_size * self.preparation_depth
                + self.reserve_bytes)

    def microbatch_for(self, vram_bytes):
        if type(vram_bytes) is not int or vram_bytes < 0:
            raise ValueError("VRAM allowance must be nonnegative integer bytes")
        candidates = [size for size, _ in self.measured_peaks
                      if size <= self.max_microbatch and self.estimated_bytes(size) <= vram_bytes]
        return max(candidates, default=0)


@dataclass(frozen=True)
class WorkPartition:
    gpu_uuid: str
    microbatch_size: int
    start: int
    stop: int

    def batches(self):
        return tuple(tuple(range(i, min(i + self.microbatch_size, self.stop)))
                     for i in range(self.start, self.stop, self.microbatch_size))


def partition_grant(grant, calibration, *, step, effective_batch_size):
    """Exact canonical effective batch under unequal GPU VRAM allowances."""
    if type(step) is not int or step < 0 or type(effective_batch_size) is not int or effective_batch_size < 1:
        raise ValueError("invalid optimizer step/effective batch size")
    devices = sorted(grant["gpus"].items())
    if not devices:
        return ()
    sizes = [calibration.microbatch_for(vram) for _, vram in devices]
    if min(sizes) < 1:
        raise ValueError("grant cannot fit a calibrated single-sample microbatch")
    # Device identities fix order. Weight sample assignment by physical batch
    # capacity, but leave the canonical batch membership untouched.
    offsets = partition_offsets(effective_batch_size, sizes)
    result = []
    for rank, ((gpu, _), size) in enumerate(zip(devices, sizes, strict=True)):
        start = effective_batch_size * step + offsets[rank]
        stop = effective_batch_size * step + offsets[rank + 1]
        result.append(WorkPartition(gpu, size, start, stop))
    return tuple(result)
