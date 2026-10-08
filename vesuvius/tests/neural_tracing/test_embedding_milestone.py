import pytest
from vesuvius.neural_tracing.fiber_trace_3d.embedding_milestone import check_checkpoint


def test_exact_raw_milestone():
    assert not check_checkpoint({'step': 5300}, 5400)
    assert check_checkpoint({'step': 5400, 'samples_seen': 691200}, 5400)
    with pytest.raises(RuntimeError, match='Missed'):
        check_checkpoint({'step': 5500}, 5400)
    with pytest.raises(ValueError):
        check_checkpoint({'step': 5400, 'samples_seen': 691200, 'snapshot_kind': 'ema'}, 5400)
    with pytest.raises(ValueError):
        check_checkpoint({'step': 5400, 'samples_seen': 345600}, 5400)
