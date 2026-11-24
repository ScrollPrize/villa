"""Regression tests for sparse 3x3 structure tensor eigensolve and AMD ROCm solver fallback."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from vesuvius.structure_tensor.create_st import (
    _compute_eigenvectors,
    _eigh_and_sanitize,
    _is_gpu_solver_error,
    _orient_eigenvectors_right_handed,
    _quantize_unit_u8,
)


def _reference_eigh_block(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, dz, dy, dx = block.shape
    N = dz * dy * dx
    x = torch.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).reshape(6, N)
    M = torch.empty((N, 3, 3), dtype=torch.float64, device=block.device)
    M[:, 0, 0] = x[0]
    M[:, 0, 1] = x[1]
    M[:, 0, 2] = x[2]
    M[:, 1, 0] = x[1]
    M[:, 1, 1] = x[3]
    M[:, 1, 2] = x[4]
    M[:, 2, 0] = x[2]
    M[:, 2, 1] = x[4]
    M[:, 2, 2] = x[5]
    zero_mask = M.abs().sum(dim=(1, 2)) == 0
    w, v = torch.linalg.eigh(M)
    w = torch.nan_to_num(w.float(), nan=0.0, posinf=0.0, neginf=0.0).masked_fill(zero_mask.unsqueeze(-1), 0.0)
    v = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0).masked_fill(
        zero_mask.unsqueeze(-1).unsqueeze(-1), 0.0
    )
    return (
        w.transpose(0, 1).view(3, dz, dy, dx),
        v.permute(0, 2, 1).reshape(N, 9).transpose(0, 1).view(9, dz, dy, dx),
    )


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("cuSOLVER error: CUSOLVER_STATUS_INTERNAL_ERROR", True),
        ("hipsolver error: HIPSOLVER_STATUS_EXECUTION_FAILED", True),
        ("rocSOLVER syevd failed on gfx1100", True),
        ("ROCm linear algebra backend failure", True),
        ("shape mismatch", False),
        ("out of memory", False),
    ],
)
def test_is_gpu_solver_error_detects_cuda_and_rocm(msg, expected):
    assert _is_gpu_solver_error(RuntimeError(msg)) is expected


@pytest.mark.parametrize(
    "error_msg",
    [
        "CUSOLVER_STATUS_INTERNAL_ERROR",
        "HIPSOLVER_STATUS_EXECUTION_FAILED",
        "rocsolver failed",
    ],
)
def test_eigh_and_sanitize_falls_back_to_cpu_on_cuda_and_rocm_errors(error_msg):
    M = torch.tensor([[[3.0, 0.5, 0.1], [0.5, 2.0, 0.2], [0.1, 0.2, 1.0]]], dtype=torch.float64)
    real_eigh = torch.linalg.eigh
    calls = {"n": 0}

    def flaky_eigh(tensor):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(error_msg)
        return real_eigh(tensor)

    with mock.patch("torch.linalg.eigh", side_effect=flaky_eigh):
        w, v = _eigh_and_sanitize(M)
    assert calls["n"] == 2
    assert torch.isfinite(w).all() and torch.isfinite(v).all()
    assert torch.all(w[:, 1:] >= w[:, :-1])


def test_eigh_and_sanitize_falls_back_to_numpy_when_cpu_torch_lacks_lapack():
    M = torch.tensor([[[3.0, 0.5, 0.1], [0.5, 2.0, 0.2], [0.1, 0.2, 1.0]]], dtype=torch.float64)
    calls = {"n": 0}

    def no_lapack_eigh(tensor):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rocSOLVER syevd failed on gfx906")
        raise RuntimeError("linalg.eigh: Calling torch.linalg.eigh on a CPU tensor requires compiling PyTorch with LAPACK.")

    with mock.patch("torch.linalg.eigh", side_effect=no_lapack_eigh):
        w, v = _eigh_and_sanitize(M)
    assert calls["n"] == 2
    assert torch.isfinite(w).all() and torch.isfinite(v).all()
    assert torch.all(w[:, 1:] >= w[:, :-1])
    w_ref, _ = torch.linalg.eigh(M)
    assert torch.allclose(w, w_ref.float(), atol=1e-5, rtol=1e-5)


def test_eigh_and_sanitize_reraises_non_solver_runtime_error():
    M = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    with mock.patch("torch.linalg.eigh", side_effect=RuntimeError("unrelated failure")):
        with pytest.raises(RuntimeError, match="unrelated failure"):
            _eigh_and_sanitize(M)


def test_compute_eigenvectors_empty_chunk_exact_zero():
    block = torch.zeros((6, 8, 8, 8), dtype=torch.float32)
    w, v = _compute_eigenvectors(block)
    assert w.shape == (3, 8, 8, 8)
    assert v.shape == (9, 8, 8, 8)
    assert torch.equal(w, torch.zeros_like(w))
    assert torch.equal(v, torch.zeros_like(v))


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e5])
def test_compute_eigenvectors_accuracy_and_orthonormality_across_scales(scale):
    gen = torch.Generator().manual_seed(2026)
    dz, dy, dx = 8, 8, 8
    N = dz * dy * dx
    # Build random PSD structure tensors A^T A across multiple scales + zero mask
    A = torch.randn((N, 3, 3), generator=gen, dtype=torch.float32) * (scale ** 0.5)
    M = torch.bmm(A.transpose(1, 2), A)
    # Inject zeros on 40% of voxels
    zero_mask = torch.rand((N,), generator=gen) < 0.4
    M[zero_mask] = 0.0
    block = torch.stack(
        (M[:, 0, 0], M[:, 0, 1], M[:, 0, 2], M[:, 1, 1], M[:, 1, 2], M[:, 2, 2]),
        dim=0,
    ).view(6, dz, dy, dx)

    w, v = _compute_eigenvectors(block)
    rw, rv = _reference_eigh_block(block)

    # Active voxels keep main's per-voxel eigh output, including eigenvector signs
    assert torch.equal(w, rw)
    assert torch.equal(v, rv)

    # Exact zero match
    nz = ~zero_mask.view(dz, dy, dx)
    assert torch.equal(w[:, ~nz], torch.zeros_like(w[:, ~nz]))
    assert torch.equal(v[:, ~nz], torch.zeros_like(v[:, ~nz]))

    # Eigenvalues ascending & accurate far beyond float32 precision
    assert torch.all(w[1] >= w[0] - 1e-6 * scale)
    assert torch.all(w[2] >= w[1] - 1e-6 * scale)
    rel_err = ((w - rw).abs() / rw.abs().amax(dim=0, keepdim=True).clamp_min(1e-30)).max().item()
    assert rel_err < 1e-5, f"rel_err={rel_err}"

    # Verify eigenpair residual ||M v_k - w_k v_k|| / ||M|| and V^T V == I on all nonzero voxels
    nz_flat = ~zero_mask
    Mnz = M[nz_flat].double()
    Vnz = v.view(3, 3, N)[:, :, nz_flat].permute(2, 1, 0).double()  # [K, comp, eigvec]
    Wnz = w.view(3, N)[:, nz_flat].T.double()  # [K, 3]
    gram = torch.bmm(Vnz.transpose(1, 2), Vnz)
    eye = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand_as(gram)
    assert (gram - eye).abs().max().item() < 1e-5
    lhs = torch.bmm(Mnz, Vnz)
    rhs = Vnz * Wnz.unsqueeze(1)
    m_norm = Mnz.flatten(1).norm(dim=1).clamp_min(1e-30)
    res = ((lhs - rhs).flatten(1).norm(dim=1) / m_norm).max().item()
    assert res < 1e-5, f"eigen residual={res}"


def test_compute_eigenvectors_degenerate_and_nans():
    # Repeated eigenvalues, rank-1, isotropic, NaN/Inf, and zero voxels in one block.
    mats = [
        [2.0, 0.0, 0.0, 2.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [5.0, 0.0, 0.0, 5.0, 0.0, 5.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [float("nan"), 1.0, 0.0, 2.0, float("inf"), 3.0],
        [float("-inf"), 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 1.0, 0.5, 3.0, 0.25, 2.0],
    ]
    block = torch.tensor(mats, dtype=torch.float32).T.view(6, 2, 2, 2)
    w, v = _compute_eigenvectors(block)
    rw, _ = _reference_eigh_block(block)
    assert torch.isfinite(w).all() and torch.isfinite(v).all()
    assert torch.allclose(w, rw, atol=1e-5, rtol=1e-5)


def test_orient_eigenvectors_right_handed_matches_linalg_det():
    gen = torch.Generator().manual_seed(99)
    A = torch.randn((64, 3, 3), generator=gen, dtype=torch.float32)
    M = torch.bmm(A.transpose(1, 2), A)
    _, V = torch.linalg.eigh(M)
    # Deliberately flip half the frames left-handed
    V[:32, :, 2] *= -1
    eigvecs = V.permute(0, 2, 1).reshape(64, 9).transpose(0, 1).view(9, 4, 4, 4)

    # Reference implementation from original create_st.py
    V_flat = eigvecs.reshape(3, 3, -1).permute(2, 0, 1).clone()
    det_ref = torch.linalg.det(V_flat)
    V_flat[det_ref < 0, 2, :] *= -1
    v_ref = V_flat.permute(1, 2, 0).reshape(3, 3, 4, 4, 4)
    s = torch.sign(v_ref[0, 0].mean())
    s = torch.tensor(1.0) if s == 0 else s
    v_ref[0] *= s
    v_ref[2] *= s

    v_got = _orient_eigenvectors_right_handed(eigvecs)
    assert torch.allclose(v_got, v_ref, atol=1e-6)
    # Check det > 0 everywhere
    det_got = torch.linalg.det(v_got.reshape(3, 3, -1).permute(2, 0, 1))
    assert torch.all(det_got > 0.999)


def test_fa_confidence_quantization_matches_reference():
    gen = torch.Generator().manual_seed(7)
    A = torch.randn((512, 3, 3), generator=gen, dtype=torch.float32)
    M = torch.bmm(A.transpose(1, 2), A)
    block = torch.stack(
        (M[:, 0, 0], M[:, 0, 1], M[:, 0, 2], M[:, 1, 1], M[:, 1, 2], M[:, 2, 2]),
        dim=0,
    ).view(6, 8, 8, 8)
    w, _ = _compute_eigenvectors(block)
    rw, _ = _reference_eigh_block(block)
    for arr in (w.numpy(), rw.numpy()):
        assert np.isfinite(arr).all()

    def _fa(wnp):
        l1, l2, l3 = wnp[0], wnp[1], wnp[2]
        lbar = (l1 + l2 + l3) / 3.0
        num = (l1 - lbar) ** 2 + (l2 - lbar) ** 2 + (l3 - lbar) ** 2
        den = (l1 ** 2 + l2 ** 2 + l3 ** 2) + 1e-12
        return _quantize_unit_u8(np.sqrt(1.5) * np.sqrt(num) / np.sqrt(den))

    np.testing.assert_array_equal(_fa(w.numpy()), _fa(rw.numpy()))

