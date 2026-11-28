"""Host batches dropped right after ``to_device`` on MPS must still arrive intact: on torch 2.11 and 2.12 a non_blocking
host-to-MPS copy can read its source after it was freed (pytorch/pytorch#189690)."""
import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.tracers.sheets.data.episodes import pack_blocks
from vesuvius.neural_tracing.tracers.sheets.evaluation.rollouts import to_device


def _batch(i):
    g = torch.Generator().manual_seed(i)
    blocks = [np.random.default_rng(i).integers(0, 255, (1, 4, 5, 6), dtype=np.uint8)]
    return dict(gt=torch.rand(1, 5, 4, 3, generator=g), valid=torch.rand(1, 5, 4, generator=g) > .5,
                block=pack_blocks(blocks), origin=torch.randint(0, 9, (1, 3), generator=g), meta=[{}])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_mps_batches_survive_dropped_host_batches():
    bad = 0
    for _ in range(20):
        got = [to_device(_batch(i), 'mps') for i in range(3)]  # each host batch is freed right after its copy
        for i, b in enumerate(got):
            want = to_device(_batch(i), 'cpu')
            bad += not all(torch.equal(b[k].cpu(), want[k]) for k in ('gt', 'valid', 'block', 'origin'))
    assert bad == 0, f'{bad} of 60 MPS batches differ from their host batch'
