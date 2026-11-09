"""GHBP: CPU == CUDA, and agreement with an independent NumPy implementation (tests/reference/ghbp_ref.py)."""

import numpy as np
import pytest

import helirec
from reference import ghbp_ref

CASES = [  # (n_angles, N, n_bins, rot_center, legs, steps, clip, full_turn)
    (8000, 129, 90, 61.25, 8, 2, True, True),    # production-like hierarchy
    (8000, 129, 90, 61.25, 4, 3, False, True),   # with an intermediate level
    (900, 129, 140, 70.5, 3, 2, True, False),    # 180 degrees
]


def make_case(n, nb, seed=0):
    rng = np.random.default_rng(seed)
    sino = rng.standard_normal((n, nb)).astype(np.float32)
    return sino


@pytest.mark.parametrize("case", CASES)
def test_matches_numpy_port(case):
    n, N, nb, rc, legs, steps, clip, full = case
    sino = make_case(n, nb)
    A = (np.arange(n) * ((2 if full else 1) * np.pi / n)).astype(np.float32)
    ref = ghbp_ref.ghbp(sino, -A, N, rc, 1e9, 8.64, steps, legs, 2.0, clip)
    out = helirec.ghbp(sino, A, N, rc, scale_factor=2.0, legs=legs, reduction_steps=steps,
                       clip_outer_circle=clip, voxel_size_microns=8.64, device="cpu")
    rel = np.sqrt(((out - ref) ** 2).mean()) / np.sqrt((ref**2).mean())
    assert rel < 2e-5, rel


@pytest.mark.skipif(not helirec.cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("case", CASES)
def test_cpu_cuda_bitwise(case):
    n, N, nb, rc, legs, steps, clip, full = case
    sino = make_case(n, nb, 1)
    A = (np.arange(n) * ((2 if full else 1) * np.pi / n)).astype(np.float32)
    kw = dict(scale_factor=3.0, legs=legs, reduction_steps=steps, clip_outer_circle=clip)
    a = helirec.ghbp(sino, A, N, rc, device="cpu", **kw)
    b = helirec.ghbp(sino, A, N, rc, device="cuda", **kw)
    assert np.array_equal(a, b), np.abs(a - b).max()
