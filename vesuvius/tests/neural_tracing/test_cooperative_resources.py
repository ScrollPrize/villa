import pytest

from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import (
    EffectiveBatchSampleIndex, MemoryCalibration, partition_grant,
)


def calibration():
    return MemoryCalibration(fixed_bytes=100, bytes_per_sample=10,
                             preparation_bytes_per_sample=5, preparation_depth=2,
                             reserve_bytes=20, max_microbatch=16,
                             measured_peaks=((1, 120), (4, 180), (16, 420)))


def test_memory_grants_and_no_extrapolation():
    c = calibration()
    assert c.microbatch_for(139) == 0
    assert c.microbatch_for(140) == 1
    assert c.microbatch_for(199) == 1
    assert c.microbatch_for(200) == 4
    assert c.microbatch_for(10000) == 16
    with pytest.raises(ValueError, match="underestimates"):
        MemoryCalibration(10, 1, 0, 0, 0, 1, ((1, 100),))


def test_unequal_vram_exact_batch():
    grant = {"cpu_cores": 16, "gpus": {"GPU-c": 440, "GPU-a": 140, "GPU-b": 200}}
    partitions = partition_grant(grant, calibration(), step=7, effective_batch_size=128)
    assert [p.microbatch_size for p in partitions] == [1, 4, 16]
    assert [i for p in partitions for batch in p.batches() for i in batch] == list(range(896, 1024))
    assert partition_grant({"gpus": {}}, calibration(), step=0, effective_batch_size=128) == ()


def test_insufficient_grant_fails_not_silently_shrinks_batch():
    with pytest.raises(ValueError, match="single-sample"):
        partition_grant({"gpus": {"GPU-a": 30}}, calibration(), step=0, effective_batch_size=128)


@pytest.mark.parametrize("sizes,batch", [((1, 4, 16), 128), ((16,) * 8, 128), ((1, 1, 1), 2), ((3, 5), 17)])
def test_canonical_index_partial_and_zero_work_ranks(sizes, batch):
    maps = [EffectiveBatchSampleIndex(batch, sizes, rank, start_step=4) for rank in range(len(sizes))]
    for step in range(3):
        samples = []
        for mapping in maps:
            assert sum(mapping.batch_sizes) == mapping.local_samples
            samples.extend(mapping(step * mapping.local_samples + i) for i in range(mapping.local_samples))
        assert samples == list(range((step + 4) * batch, (step + 5) * batch))
