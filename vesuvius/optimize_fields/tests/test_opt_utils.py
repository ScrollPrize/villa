import pytest
import torch
import numpy as np
import wandb

# Stub out wandb.log so optimizer tests don’t require wandb.init()
wandb.log = lambda *args, **kwargs: None

from optimize_fields.optimizer import (
    _init_identity,
    _init_random,
    _init_svd,
    PatchOptimizer,
)
from optimize_fields.utils import (
    rotate_by_quaternion,
    matrix_to_quaternion,
    gradient
)


def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    """Your re‐labeled axes: Z→x, Y→y, X→z."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'Z':
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]], dtype=np.float64)
    if axis == 'Y':
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float64)
    if axis == 'X':
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,   0, 1]], dtype=np.float64)
    raise ValueError(f"Unknown axis {axis}")


def test_init_identity():
    D, H, W = 2, 3, 4
    q = _init_identity(D, H, W, device=torch.device('cpu'), dtype=torch.float32)
    assert q.shape == (4, D, H, W)
    expected = torch.zeros_like(q)
    expected[0] = 1.0
    assert torch.allclose(q, expected, atol=0.0)


def test_init_random_unit_norm():
    D, H, W = 5, 6, 7
    eps = 1e-8
    q = _init_random(D, H, W, device=torch.device('cpu'),
                     dtype=torch.float32, eps=eps)
    assert q.shape == (4, D, H, W)
    norms = torch.norm(q, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_init_svd_identity_and_zero():
    D, H, W = 3, 4, 5
    # Perfect canonical UVN → identity
    U = torch.zeros(3, D, H, W)
    V = torch.zeros_like(U)
    N = torch.zeros_like(U)
    U[0].fill_(1.0)
    V[1].fill_(1.0)
    N[2].fill_(1.0)
    q = _init_svd(U, V, N, eps=1e-8)
    assert q.shape == (4, D, H, W)
    assert torch.allclose(q[0], torch.ones((D, H, W)), atol=1e-6)
    assert torch.allclose(q[1:], torch.zeros((3, D, H, W)), atol=1e-6)

    # All-zero UVN → identity
    qz = _init_svd(U*0, V*0, N*0, eps=1e-8)
    assert torch.allclose(qz[0], torch.ones((D, H, W)), atol=1e-6)
    assert torch.allclose(qz[1:], torch.zeros((3, D, H, W)), atol=1e-6)


def test_init_svd_recovers_uniform_rotation():
    # 45° about Z (→ x‐axis): quaternion [w,x,y,z] = [c,s,0,0]
    theta = np.pi / 4
    c, s = np.cos(theta/2), np.sin(theta/2)
    q_true = torch.tensor([c, s, 0.0, 0.0], dtype=torch.float32)
    D, H, W = 4, 4, 4

    # Expand quaternion field
    qf = q_true.view(1, 1, 1, 4).expand(D, H, W, 4)

    # Expand canonical axes to (D,H,W,3)
    u0 = torch.tensor([1, 0, 0], dtype=torch.float32)
    v0 = torch.tensor([0, 1, 0], dtype=torch.float32)
    n0 = torch.tensor([0, 0, 1], dtype=torch.float32)
    u0e = u0.view(1, 1, 1, 3).expand(D, H, W, 3)
    v0e = v0.view(1, 1, 1, 3).expand(D, H, W, 3)
    n0e = n0.view(1, 1, 1, 3).expand(D, H, W, 3)

    # Rotate
    U_nhwc = rotate_by_quaternion(qf, u0e)
    V_nhwc = rotate_by_quaternion(qf, v0e)
    N_nhwc = rotate_by_quaternion(qf, n0e)

    # To (3,D,H,W)
    U = U_nhwc.permute(3, 0, 1, 2)
    V = V_nhwc.permute(3, 0, 1, 2)
    N = N_nhwc.permute(3, 0, 1, 2)

    q_out = _init_svd(U, V, N, eps=1e-6)

    # Dot‐product |⟨q_out, q_true⟩| ≈ 1
    q_flat = q_out.permute(1, 2, 3, 0).reshape(-1, 4)
    q_t_flat = q_true.expand(q_flat.shape[0], 4)
    dots = (q_flat * q_t_flat).sum(dim=1).abs()
    assert torch.allclose(dots, torch.ones_like(dots), atol=1e-5)


def test_gradient():
    # Ramp along x in channel 0
    D, H, W = 1, 1, 5
    f = torch.zeros(3, D, H, W)
    ramp = torch.linspace(0, 4, W)
    f[0, 0, 0, :] = ramp

    grad = gradient(f)
    # grad shape is (channels=3, axes=3, D,H,W)
    assert grad.shape == (3, 3, D, H, W)

    # 1) No Z- or Y-gradient anywhere:
    for ch in range(3):
        zero_zy = torch.zeros_like(grad[ch, 0])
        assert torch.allclose(grad[ch, 0], zero_zy, atol=1e-6)  # ∂/∂z
        assert torch.allclose(grad[ch, 1], zero_zy, atol=1e-6)  # ∂/∂y

    # 2) Only channel 0 has an X-derivative
    zero_x = torch.zeros_like(grad[1, 2])
    assert torch.allclose(grad[1, 2], zero_x, atol=1e-6)
    assert torch.allclose(grad[2, 2], zero_x, atol=1e-6)

    dx = grad[0, 2, 0, 0, :]  # channel0, ∂/∂x, flatten
    assert dx.shape == (W,)

    # 3a) First W-1 entries positive
    assert torch.all(dx[:-1] > 0)

    # 3b) Last entry negative
    assert dx[-1] < 0

    # 3c) Maximum at center index
    center = W // 2
    assert dx[center].item() == pytest.approx(dx.max().item(), rel=1e-5)


def test_rotate_then_inverse():
    q = torch.randn(4)
    q = q / q.norm()
    v = torch.randn(3)
    qc = torch.tensor([q[0], -q[1], -q[2], -q[3]])
    v1 = rotate_by_quaternion(q, v)
    v2 = rotate_by_quaternion(qc, v1)
    assert torch.allclose(v2, v, atol=1e-6)


def test_matrix_to_quaternion_w_nonnegative():
    for _ in range(10):
        A = torch.randn(3, 3)
        Q, R = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        q = matrix_to_quaternion(Q.unsqueeze(0)).squeeze(0)
        assert q[0].item() >= 0
        assert abs(q.norm().item() - 1.0) < 1e-6


def test_patchoptimizer_zero_patch_returns_identity():
    D, H, W = 4, 4, 4
    U = torch.zeros(3, D, H, W)
    V = torch.zeros_like(U)
    N = torch.zeros_like(U)
    opt = PatchOptimizer(
        mu0=1e-3, mu1=1e-2, mu4=1e-2, mu5=1e-3,
        lr=0.1, iters=5,
        device=torch.device('cpu'),
        optimizer_type='adam',
        init_type='svd',
    )
    q_out = opt(U, V, N, chunk_idx=0)
    assert q_out.shape == (4, D, H, W)
    assert torch.allclose(q_out[0], torch.ones((D, H, W)), atol=1e-6)
    assert torch.allclose(q_out[1:], torch.zeros((3, D, H, W)), atol=1e-6)


def test_optimizer_identity_skips():
    D, H, W = 3, 3, 3
    U = torch.zeros(3, D, H, W); U[0].fill_(1.0)
    V = torch.zeros_like(U);      V[1].fill_(1.0)
    N = torch.zeros_like(U);      N[2].fill_(1.0)
    opt = PatchOptimizer(
        mu0=1e-3, mu1=1e-2, mu4=1e-2, mu5=1e-3,
        lr=0.1, iters=10,
        device=torch.device('cpu'),
        optimizer_type='adam',
        init_type='identity',
    )
    q_out = opt(U, V, N, chunk_idx=0)
    # it should still be a valid quaternion field: shape (4,D,H,W) and unit‐norm everywhere
    assert q_out.shape == (4, D, H, W)
    norms = torch.sqrt((q_out ** 2).sum(dim=0))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
