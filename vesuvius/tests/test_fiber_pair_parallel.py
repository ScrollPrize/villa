import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_pair_parallel import ParallelPairs
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as imp


@pytest.mark.parametrize('workers', [2, 3])
def test_uneven_pair_partition_values_gradients(workers):
    torch.manual_seed(7)
    a = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    widths = torch.full((10,), 100., dtype=torch.float64, requires_grad=True)
    fa = torch.arange(7) % 2
    args = (a, b, fa, 1-fa, torch.arange(7.), torch.arange(7.), widths,
            torch.ones(7), torch.ones(7), torch.zeros(7, dtype=torch.bool),
            dict(winding_sigma=.5, sigma_max=8., pair_sigma_groups=True,
                 soft_pair_weight=1., probability_pair_weight=.3))
    kw = dict(sigma_a=torch.full((7,), .5, requires_grad=True), sigma_b=torch.full((7,), .8, requires_grad=True))
    expected = imp.pair_cost(*args, **kw)
    parallel = ParallelPairs(imp.pair_cost, ['cpu'] * workers)
    try:
        actual = parallel(*args, **kw)
        for x, y in zip(expected, actual): torch.testing.assert_close(x, y)
        p = (a, b, widths, kw['sigma_a'], kw['sigma_b'])
        eg = torch.autograd.grad(sum(v.sum() for v in expected), p)
        ag = torch.autograd.grad(sum(v.sum() for v in actual), p)
        for x, y in zip(eg, ag): torch.testing.assert_close(x, y)
    finally:
        parallel.close()
