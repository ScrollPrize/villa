from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.arc_sampling import MixedSourceArcSampler


def records():
    return [SimpleNamespace(dataset_config={"scroll_id": scroll, "supervision_mode": mode},
                            fiber=SimpleNamespace(line_points_xyz=np.array([[0., 0, 0], [length, 0, 0]]),
                                                  path=f"{mode}/{scroll}/{length}"))
            for mode in ("precise", "coarse_area")
            for scroll in ("A", "B") for length in (10., 30.)]


def test_exact_half_independent_of_rank_order_and_resume():
    rows = records()
    sampler = MixedSourceArcSampler(rows, global_batch_size=128, seed=13)
    expected = [sampler.source(i) for i in range(128 * 10)]
    for begin in range(0, len(expected), 128):
        assert Counter(expected[begin:begin+128]) == {"precise": 64, "coarse_area": 64}
    assert expected[:128] != expected[128:256]
    for ranks in (1, 2, 6, 8):
        for rank in range(ranks):
            resumed = MixedSourceArcSampler(rows, global_batch_size=128, seed=13)
            for i in range(rank, len(expected), ranks):
                assert resumed.source(i) == expected[i]
    for i in np.random.default_rng(5).permutation(len(expected)):
        assert sampler.source(int(i)) == expected[i]


def test_source_scroll_and_fiber_distance_probabilities():
    rows = records()
    sampler = MixedSourceArcSampler(rows, global_batch_size=128, seed=13)
    rng = np.random.default_rng(19)
    counts = Counter()
    for i in range(128 * 128):
        record, index, location = sampler.sample(i, rng)
        assert record is rows[index]
        assert record.dataset_config["supervision_mode"] == sampler.source(i)
        assert 0 <= location.fraction < 1
        counts[index] += 1
    for start in (0, 4):
        total = sum(counts[i] for i in range(start, start+4))
        assert total == 8192
        for scroll in (start, start+2):
            n = counts[scroll] + counts[scroll+1]
            assert n / total == pytest.approx(.5, abs=.025)
            assert counts[scroll+1] / n == pytest.approx(.75, abs=.025)


@pytest.mark.parametrize("size", [0, 1, 127, -2, True])
def test_reject_non_even_batch(size):
    with pytest.raises(ValueError, match="positive even"):
        MixedSourceArcSampler(records(), global_batch_size=size, seed=1)


def test_reject_missing_source():
    with pytest.raises(ValueError, match="both"):
        MixedSourceArcSampler(records()[:4], global_batch_size=128, seed=1)
