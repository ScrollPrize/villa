"""Unit tests for the closed-form analytical 3x3 symmetric eigensolver.

Covers `_analytical_eigh_3x3_6ch` (Cardano eigenvalues + cross-product
eigenvectors with eigh fallback) and the end-to-end `_compute_eigenvectors_impl`
path that uses it, on CPU and CUDA.
"""

from __future__ import annotations

import pytest
import torch

from vesuvius.structure_tensor.create_st import (
    _analytical_eigh_3x3_6ch,
    _compute_eigenvectors_impl,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _to_6ch(M: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) symmetric matrices -> (6, N) channel layout [a00, a01, a02, a11, a12, a22]."""
    return torch.stack(
        [M[:, 0, 0], M[:, 0, 1], M[:, 0, 2], M[:, 1, 1], M[:, 1, 2], M[:, 2, 2]],
        dim=0,
    )


def _rotated_diag(dvals: list[float], Q: torch.Tensor) -> torch.Tensor:
    """Q diag(dvals) Q^T for a batch of rotations Q (N, 3, 3)."""
    N = Q.shape[0]
    d = torch.tensor(dvals, dtype=torch.float64, device=Q.device).expand(N, 3).clone()
    return torch.bmm(Q, torch.bmm(torch.diag_embed(d), Q.transpose(1, 2)))


def _make_matrices(kind: str, N: int, gen: torch.Generator) -> torch.Tensor:
    if kind == "general_symmetric":
        M = torch.randn((N, 3, 3), generator=gen, dtype=torch.float64)
        return 0.5 * (M + M.transpose(1, 2))
    if kind == "psd":
        A = torch.randn((N, 3, 3), generator=gen, dtype=torch.float64)
        return torch.bmm(A.transpose(1, 2), A)
    if kind == "axisymmetric":  # l1 == l2 < l3, randomly rotated
        Q, _ = torch.linalg.qr(torch.randn((N, 3, 3), generator=gen, dtype=torch.float64))
        return _rotated_diag([1.0, 1.0, 3.0], Q)
    if kind == "axisymmetric_l2_l3":  # l1 < l2 == l3, randomly rotated
        Q, _ = torch.linalg.qr(torch.randn((N, 3, 3), generator=gen, dtype=torch.float64))
        return _rotated_diag([1.0, 3.0, 3.0], Q)
    if kind == "degenerate":  # l1 == l2 == l3
        return torch.diag_embed(torch.full((N, 3), 5.0, dtype=torch.float64))
    if kind == "zero":
        return torch.zeros((N, 3, 3), dtype=torch.float64)
    if kind == "tiny":  # entries ~1e-12
        A = torch.randn((N, 3, 3), generator=gen, dtype=torch.float64) * 1e-6
        return torch.bmm(A.transpose(1, 2), A)
    if kind == "diagonal":
        d = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64).expand(N, 3).clone()
        return torch.diag_embed(d)
    raise ValueError(kind)


def _check_against_eigh(M: torch.Tensor, x6: torch.Tensor, device: str, check_alignment: bool):
    """Assert parity with torch.linalg.eigh plus orthonormality/eigen-residual invariants."""
    w, v = _analytical_eigh_3x3_6ch(x6)
    assert w.shape == (M.shape[0], 3) and v.shape == (M.shape[0], 3, 3)
    assert w.dtype == torch.float32 and v.dtype == torch.float32
    assert w.device.type == device and v.device.type == device
    assert torch.isfinite(w).all() and torch.isfinite(v).all()

    # Ascending eigenvalues.
    assert torch.all(w[:, 1:] >= w[:, :-1] - 1e-6 * M.abs().amax(dim=(1, 2)).unsqueeze(1))

    wr, vr = torch.linalg.eigh(M)
    wr = wr.float()
    vr = vr.float()

    # Eigenvalue parity: <= 1e-6 relative to each matrix's scale.
    scale = M.abs().amax(dim=(1, 2)).clamp(min=1e-30)
    rel_w = ((w - wr).abs() / scale.unsqueeze(1)).max().item()
    assert rel_w <= 1e-6, f"eigenvalue rel err {rel_w:.2e}"

    # Orthonormality ||V^T V - I|| < 1e-5 (V columns are the eigenvectors).
    gram = torch.bmm(v.transpose(1, 2), v)
    orth = (gram - torch.eye(3, dtype=torch.float32, device=v.device).unsqueeze(0)).abs().max().item()
    assert orth < 1e-5, f"orthonormality {orth:.2e}"

    # Eigen-residual ||M v_k - w_k v_k|| < 1e-5 * ||M||.
    lhs = torch.bmm(M.float(), v)
    rhs = v * w.unsqueeze(1)
    m_norm = M.float().flatten(1).norm(dim=1).clamp(min=1e-30)
    res = ((lhs - rhs).flatten(1).norm(dim=1) / m_norm).max().item()
    assert res < 1e-5, f"eigen-residual {res:.2e}"

    if check_alignment:
        # For simple spectra the eigenvector directions must match eigh up to sign.
        dots = (v * vr).sum(dim=1).abs()  # (N, 3)
        assert dots.min().item() > 1.0 - 1e-5, f"min alignment {dots.min().item():.8f}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["general_symmetric", "psd", "tiny"])
def test_analytical_parity_with_eigh_simple_spectra(device, kind):
    gen = torch.Generator().manual_seed(1234)
    N = 4096
    M = _make_matrices(kind, N, gen).to(device)
    _check_against_eigh(M, _to_6ch(M), device, check_alignment=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kind", ["axisymmetric", "axisymmetric_l2_l3", "degenerate", "zero", "diagonal"]
)
def test_analytical_special_matrices(device, kind):
    # Degenerate subspaces allow any orthonormal basis, so only eigenvalues,
    # orthonormality and residuals are checked (not direction alignment).
    gen = torch.Generator().manual_seed(99)
    N = 2048
    M = _make_matrices(kind, N, gen).to(device)
    _check_against_eigh(M, _to_6ch(M), device, check_alignment=False)


@pytest.mark.parametrize("device", DEVICES)
def test_analytical_low_fallback_rate(device, monkeypatch):
    """Verify _eigh_and_sanitize fallback stays <1% on generic PSD and near-isotropic 5*I + 1e-3*S."""
    import vesuvius.structure_tensor.create_st as create_st_mod

    orig_eigh = create_st_mod._eigh_and_sanitize
    fallback_count = 0

    def _counting_eigh(M: torch.Tensor):
        nonlocal fallback_count
        fallback_count += M.shape[0]
        return orig_eigh(M)

    monkeypatch.setattr(create_st_mod, "_eigh_and_sanitize", _counting_eigh)

    gen = torch.Generator().manual_seed(42)
    N = 2048

    M_psd = _make_matrices("psd", N, gen).to(device)
    fallback_count = 0
    _check_against_eigh(M_psd, _to_6ch(M_psd), device, check_alignment=True)
    assert fallback_count < 0.01 * N, f"PSD fallback count {fallback_count}/{N}"

    S = _make_matrices("general_symmetric", N, gen).to(device)
    eye = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0)
    M_near_iso = 5.0 * eye + 1e-3 * S
    fallback_count = 0
    _check_against_eigh(M_near_iso, _to_6ch(M_near_iso), device, check_alignment=True)
    assert fallback_count < 0.01 * N, f"near-isotropic fallback count {fallback_count}/{N}"


@pytest.mark.parametrize("device", DEVICES)
def test_analytical_mixed_batch_with_fallback_and_nans(device):
    """Mixed batch: normal analytical + diagonal + isotropic + NaN rows."""
    gen = torch.Generator().manual_seed(777)
    M_norm = _make_matrices("psd", 256, gen)
    M_diag = _make_matrices("diagonal", 64, gen)
    M_iso = _make_matrices("degenerate", 64, gen)
    M_nan = torch.full((32, 3, 3), float("nan"), dtype=torch.float64)
    M_nan[:, 0, 1] = 1.0  # partial NaN row

    M = torch.cat([M_norm, M_diag, M_iso, M_nan], dim=0).to(device)
    valid = torch.isfinite(M).all(dim=(1, 2))

    w, v = _analytical_eigh_3x3_6ch(_to_6ch(M))
    assert w.shape == (M.shape[0], 3) and v.shape == (M.shape[0], 3, 3)
    assert torch.isfinite(w).all() and torch.isfinite(v).all()

    Mv = M[valid].float()
    wv = w[valid]
    vv = v[valid]

    gram = torch.bmm(vv.transpose(1, 2), vv)
    eye = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    orth = (gram - eye).abs().max().item()
    assert orth < 1e-5, f"orthonormality {orth:.2e}"

    lhs = torch.bmm(Mv, vv)
    rhs = vv * wv.unsqueeze(1)
    m_norm = Mv.flatten(1).norm(dim=1).clamp(min=1e-30)
    res = ((lhs - rhs).flatten(1).norm(dim=1) / m_norm).max().item()
    assert res < 1e-5, f"eigen-residual {res:.2e}"


@pytest.mark.parametrize("device", DEVICES)
def test_compute_eigenvectors_impl_end_to_end_parity(device):
    """End-to-end `_compute_eigenvectors_impl` on a synthetic 3D block vs float64 eigh."""
    gen = torch.Generator().manual_seed(2026)
    dz, dy, dx = 8, 8, 8
    N = dz * dy * dx
    A = torch.randn((N, 3, 3), generator=gen, dtype=torch.float64)
    M = torch.bmm(A.transpose(1, 2), A).float()  # float32 structure tensors
    zero_mask = torch.rand((N,), generator=gen) < 0.3
    M[zero_mask] = 0.0
    block = _to_6ch(M).view(6, dz, dy, dx).to(device)

    w, v = _compute_eigenvectors_impl(block)
    assert w.shape == (3, dz, dy, dx) and v.shape == (9, dz, dy, dx)
    assert w.dtype == torch.float32 and v.dtype == torch.float32

    # Reference: the previous implementation (float64 eigh on the (N,3,3) block).
    Mr = M.to(device).double()
    wr, vr = torch.linalg.eigh(Mr)
    wr = torch.nan_to_num(wr.float(), nan=0.0, posinf=0.0, neginf=0.0)
    vr = torch.nan_to_num(vr.float(), nan=0.0, posinf=0.0, neginf=0.0)
    wr = wr.transpose(0, 1).view(3, dz, dy, dx)
    vr = vr.permute(0, 2, 1).reshape(N, 9).transpose(0, 1).view(9, dz, dy, dx)
    zero_flat = zero_mask.view(dz, dy, dx).to(device)
    wr = wr.masked_fill(zero_flat.unsqueeze(0), 0.0)
    vr = vr.masked_fill(zero_flat.unsqueeze(0), 0.0)

    # Eigenvalue parity <= 1e-6 relative on all voxels; zero voxels stay exactly zero.
    scale = M.abs().amax(dim=(1, 2)).clamp(min=1e-30).view(dz, dy, dx).to(device)
    rel_w = ((w - wr).abs() / scale.unsqueeze(0)).max().item()
    assert rel_w <= 1e-6, f"eigenvalue rel err {rel_w:.2e}"
    assert torch.equal(w[:, zero_flat], torch.zeros_like(w[:, zero_flat]))
    assert torch.equal(v[:, zero_flat], torch.zeros_like(v[:, zero_flat]))

    # Eigenvector parity (up to sign) and residuals on non-degenerate voxels.
    V = v.view(3, 3, N).permute(2, 1, 0)  # (N, comp, eigvec)
    Vr = vr.view(3, 3, N).permute(2, 1, 0)
    wf = w.view(3, N).T  # (N, 3)
    scale_flat = scale.view(N)
    simple3 = (wf[:, 2] - wf[:, 1]) > 1e-3 * scale_flat  # l3 simple
    dots3 = (V[:, :, 2] * Vr[:, :, 2]).sum(dim=1).abs()
    assert dots3[simple3].min().item() > 1.0 - 1e-4

    nz = ~zero_flat.view(N).to(device)
    Mnz = Mr[nz]
    Vnz = V[nz].double()
    Wnz = wf[nz].double()
    gram = torch.bmm(Vnz.transpose(1, 2), Vnz)
    eye = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).expand_as(gram)
    assert (gram - eye).abs().max().item() < 1e-5
    lhs = torch.bmm(Mnz, Vnz)
    rhs = Vnz * Wnz.unsqueeze(1)
    m_norm = Mnz.flatten(1).norm(dim=1).clamp(min=1e-30)
    res = ((lhs - rhs).flatten(1).norm(dim=1) / m_norm).max().item()
    assert res < 1e-5, f"eigen-residual {res:.2e}"
