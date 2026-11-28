"""sparse_pool_metal on any machine, MPS or not (CPU-only machines included).

The kernels themselves need an Apple GPU (tests/test_sparse_pool_metal.py). What runs anywhere is
checked here: the module imports without MPS, compile() refuses without MPS (so sparse_pool_cache
falls back to the torch ops and says why), calls on CPU tensors run on the torch ops with one
RuntimeWarning per process (or raise under LASAGNA_METAL=1), and the argument checks are the
torch ops'.
"""
import os
import sys
import warnings

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sparse_pool_metal as metal  # noqa: E402
import sparse_pool_ops as ops  # noqa: E402
from sparse_pool_cases import make_case  # noqa: E402

FUNCTIONS = ["sample_u8", "sample_f32", "sample_grad", "missing_chunks"]


@pytest.fixture(autouse=True)
def _auto_mode(monkeypatch):
    monkeypatch.delenv("LASAGNA_METAL", raising=False)
    monkeypatch.setattr(metal, "_WARNED", set())


def _call(module, name, args, go, origin, spacing):
    if name == "missing_chunks":
        return module.missing_chunks(args[2], args[1], origin, spacing)
    if name == "sample_grad":
        return module.sample_grad(*args, go)
    return getattr(module, name)(*args)


def _cpu_case():
    case = make_case(seed=80, C=2)
    go = torch.randn(2, *case.grid.shape[:3], generator=torch.Generator().manual_seed(1))
    return case.to("cpu"), go, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 0.75])


def test_compile_refuses_without_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(metal, "_LIB", None)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        metal.compile()
    assert metal._LIB is None


@pytest.mark.parametrize("name", FUNCTIONS)
def test_cpu_tensors_use_the_torch_ops_with_one_warning(name):
    args, go, origin, spacing = _cpu_case()
    want = _call(ops, name, args, go, origin, spacing)
    with pytest.warns(RuntimeWarning, match="not on MPS"):
        got = _call(metal, name, args, go, origin, spacing)
    assert got.device.type == "cpu" and torch.equal(got, want)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert torch.equal(_call(metal, name, args, go, origin, spacing), want)
    assert not [w for w in caught if "sparse_pool_metal" in str(w.message)]  # once per process


@pytest.mark.parametrize("name", FUNCTIONS)
def test_lasagna_metal_1_refuses_cpu_tensors(name, monkeypatch):
    monkeypatch.setenv("LASAGNA_METAL", "1")
    args, go, origin, spacing = _cpu_case()
    with pytest.raises(RuntimeError, match="LASAGNA_METAL=1.*not on MPS"):
        _call(metal, name, args, go, origin, spacing)


def test_argument_checks_are_the_torch_ops():
    pool, table, grid, offset, inv_scale = make_case(seed=81).to("cpu")
    bad = [
        ("grid", (pool, table, grid.reshape(-1, 3), offset, inv_scale)),
        ("chunk_table", (pool, table.long(), grid, offset, inv_scale)),
        ("pool", (pool.float(), table, grid, offset, inv_scale)),
        ("grid", (pool, table, grid.double(), offset, inv_scale)),
        ("offset", (pool, table, grid, offset[:2], inv_scale)),
    ]
    for match, args in bad:
        for module in (metal, ops):
            with pytest.raises(ValueError, match=match):
                module.sample_f32(*args)
            with pytest.raises(ValueError, match=match):
                module.sample_u8(*args)
    go = torch.zeros(1, *grid.shape[:3])
    for module in (metal, ops):
        with pytest.raises(ValueError, match="grad_output"):
            module.sample_grad(pool, table, grid, offset, inv_scale, go[:, :1])
        with pytest.raises(ValueError, match="xyz"):
            module.missing_chunks(grid[..., :2], table, torch.zeros(3), torch.ones(3))
