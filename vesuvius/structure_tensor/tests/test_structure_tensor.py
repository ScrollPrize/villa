# tests/test_structure_tensor.py

import pytest
import torch
import numpy as np

from structure_tensor.create_st import (
    StructureTensorInferer,
    _eigh_and_sanitize,
    _compute_eigenvectors
)


@pytest.fixture
def cpu_inferer():
    """A small, CPU‐only inferer with no smoothing, fixed patch size."""
    return StructureTensorInferer(
        model_path="dummy_model",
        input_dir="dummy_input",
        output_dir="dummy_output",
        sigma=0.0,
        smooth_components=False,
        volume=None,
        num_parts=1,
        part_id=0,
        overlap=0.0,
        step_size=1.0,
        batch_size=1,
        patch_size=(5, 5, 5),
        device="cpu",
        verbose=False,
        compressor_name="none",
        compression_level=1,
        num_dataloader_workers=0,
    )


def test_gaussian_kernel_normalizes_to_one():
    # Only applies when sigma > 0
    inferer = StructureTensorInferer(
        model_path="dummy",
        input_dir="dummy",
        output_dir="dummy",
        sigma=1.0,
        smooth_components=False,
        volume=None,
        num_parts=1,
        part_id=0,
        overlap=0.0,
        step_size=1.0,
        batch_size=1,
        patch_size=(5, 5, 5),
        device="cpu",
        verbose=False,
        compressor_name="none",
        compression_level=1,
        num_dataloader_workers=0,
    )
    # The 3D Gaussian kernel buffer should sum to 1
    g3 = inferer._gauss3d.squeeze()  # [D,H,W]
    total = float(g3.sum().item())
    assert total == pytest.approx(1.0, rel=1e-6)



def test_pavel_kernel_shapes(cpu_inferer):
    # Depth kernel is 9×5×5, height 5×9×5, width 5×5×9
    assert tuple(cpu_inferer.pavel_kz.shape) == (1, 1, 9, 5, 5)
    assert tuple(cpu_inferer.pavel_ky.shape) == (1, 1, 5, 9, 5)
    assert tuple(cpu_inferer.pavel_kx.shape) == (1, 1, 5, 5, 9)


def test_compute_structure_tensor_all_zero(cpu_inferer):
    # Zero input ⇒ zero tensor
    x = torch.zeros((1, 1, 5, 5, 5), dtype=torch.float32)
    J = cpu_inferer.compute_structure_tensor(x)
    assert J.shape == (1, 6, 5, 5, 5)
    assert torch.allclose(J, torch.zeros_like(J))


def test_compute_structure_tensor_linear_x(cpu_inferer):
    # Use a volume large enough that padding never reaches the center
    D, H, W = 9, 9, 9
    # Pure X‐ramp
    ramp = torch.arange(W, dtype=torch.float32)
    x = ramp.view(1, 1, 1, 1, W).expand(1, 1, D, H, W)

    # Compute structure tensor, drop batch dim → [6, D, H, W]
    J = cpu_inferer.compute_structure_tensor(x)[0]

    # Channel 5 is Jxx
    Jxx = J[5]
    cz, cy, cx = D//2, H//2, W//2

    # 1) At the exact center, Jxx must be positive
    center_Jxx = Jxx[cz, cy, cx]
    assert center_Jxx > 0, f"Jxx at center should be positive, got {center_Jxx.item()}"

    # 2) And it must exceed the absolute value of every other component there
    for c in range(5):
        other = J[c, cz, cy, cx].abs()
        assert center_Jxx > other, (
            f"At center voxel, Jxx={center_Jxx:.4g} must exceed |J[{c}]|={other:.4g}"
        )




@pytest.mark.parametrize("mat", [
    torch.tensor([[[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]]),
    torch.tensor([[[2., 1., 0.], [1., 2., 1.], [0., 1., 2.]]])
])
def test_eigh_and_sanitize_symmetric(mat):
    # Should return real eigenvalues and no NaNs/Infs
    w, v = _eigh_and_sanitize(mat)
    assert (~torch.isnan(w)).all() and (~torch.isnan(v)).all()
    assert (~torch.isinf(w)).all() and (~torch.isinf(v)).all()
    # w sorted ascending
    assert torch.all(w[:, 1:] >= w[:, :-1])


def test_compute_eigenvectors_constant_block():
    # build a block whose structure tensor is diagonal [1,2,3]
    dz, dy, dx = 2, 2, 2
    block = torch.zeros((6, dz, dy, dx), dtype=torch.float32)
    # channels: [Jzz, Jzy, Jzx, Jyy, Jyx, Jxx]
    block[0].fill_(3.0)  # Jzz
    block[3].fill_(2.0)  # Jyy
    block[5].fill_(1.0)  # Jxx

    eigvals, eigvecs = _compute_eigenvectors(block)
    # shapes are correct
    assert eigvals.shape == (3, dz, dy, dx)
    assert eigvecs.shape == (9, dz, dy, dx)

    # eigenvalues should be [1,2,3] for every voxel
    v0 = eigvals[0]
    v1 = eigvals[1]
    v2 = eigvals[2]
    assert torch.allclose(v0, torch.full_like(v0, 1.0), atol=1e-6)
    assert torch.allclose(v1, torch.full_like(v1, 2.0), atol=1e-6)
    assert torch.allclose(v2, torch.full_like(v2, 3.0), atol=1e-6)

    # eigenvectors must be unit‐norm and orthogonal
    # reshape to (3 eigenvectors, 3 components, ...)
    v = eigvecs.view(3, 3, dz, dy, dx)
    # check each eigenvector has unit norm everywhere
    norms = torch.sqrt((v ** 2).sum(dim=1))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # check sorted: smallest eigenvalue has eigenvector along X axis, etc.
    # For this diagonal M, eigenvectors are the coordinate axes.
    # So v[0] ≃ [1,0,0], v[1] ≃ [0,1,0], v[2] ≃ [0,0,1]
    # we can test that the largest component of each is on the correct axis
    # e.g. for first EV v[0], max index is component 0
    comps = torch.argmax(v.abs(), dim=1)
    # For M=diag(Jzz,Jyy,Jxx)=[3,2,1], ascending eigenvalues [1,2,3] correspond to axes [x=2, y=1, z=0]
    assert torch.all(comps[0] == 2), "smallest eigenvalue → x‐axis (index 2)"
    assert torch.all(comps[1] == 1), "middle eigenvalue → y‐axis (index 1)"
    assert torch.all(comps[2] == 0), "largest eigenvalue → z‐axis (index 0)"
