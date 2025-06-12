import torch
import numpy as np
import pytest

from optimize_fields.optimizer import (
    _init_identity,
    _init_random,
    _init_svd,
    PatchOptimizer
)


def test_init_identity():
    D, H, W = 2, 3, 4
    q = _init_identity(D, H, W, device=torch.device('cpu'), dtype=torch.float32)
    # shape
    assert q.shape == (4, D, H, W)
    # all voxels should be [1,0,0,0]
    expected = torch.zeros_like(q)
    expected[0] = 1.0
    assert torch.allclose(q, expected, atol=0.0)


def test_init_random_unit_norm():
    D, H, W = 5, 6, 7
    eps = 1e-8
    q = _init_random(D, H, W, device=torch.device('cpu'), dtype=torch.float32, eps=eps)
    # shape
    assert q.shape == (4, D, H, W)
    # each quaternion should be unit‐norm
    norms = torch.norm(q, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_init_svd_identity_and_zero():
    # Case A: perfect canonical UVN → should produce identity quaternion
    D, H, W = 3, 4, 5
    U = torch.zeros(3, D, H, W)
    V = torch.zeros_like(U)
    N = torch.zeros_like(U)
    # build canonical basis everywhere
    U[0].fill_(1.0)
    V[1].fill_(1.0)
    N[2].fill_(1.0)
    q = _init_svd(U, V, N, eps=1e-8)
    assert q.shape == (4, D, H, W)
    # expect [1,0,0,0] at every voxel
    assert torch.allclose(q[0], torch.ones((D, H, W)), atol=1e-6)
    assert torch.allclose(q[1:], torch.zeros((3, D, H, W)), atol=1e-6)

    # Case B: totally zero UVN → also identity
    Uz = torch.zeros_like(U)
    Vz = torch.zeros_like(V)
    Nz = torch.zeros_like(N)
    qz = _init_svd(Uz, Vz, Nz, eps=1e-8)
    assert torch.allclose(qz[0], torch.ones((D, H, W)), atol=1e-6)
    assert torch.allclose(qz[1:], torch.zeros((3, D, H, W)), atol=1e-6)


def test_patchoptimizer_zero_patch_returns_identity():
    D, H, W = 4, 4, 4
    # Create zero inputs
    U = torch.zeros(3, D, H, W)
    V = torch.zeros_like(U)
    N = torch.zeros_like(U)
    # Build optimizer with any hyperparams
    opt = PatchOptimizer(
        mu0=1e-3, mu1=1e-2, mu4=1e-2, mu5=1e-3,
        lr=0.1, iters=5,
        device=torch.device('cpu'),
        optimizer_type='adam',
        init_type='svd'
    )
    q_out = opt(U, V, N, chunk_idx=0)
    # Should be identity field
    assert q_out.shape == (4, D, H, W)
    assert torch.allclose(q_out[0], torch.ones((D, H, W)), atol=1e-6)
    assert torch.allclose(q_out[1:], torch.zeros((3, D, H, W)), atol=1e-6)
