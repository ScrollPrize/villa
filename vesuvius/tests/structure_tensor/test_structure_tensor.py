# tests/test_structure_tensor.py

import pytest
import torch
import numpy as np
import zarr
import os
from math import isfinite

from vesuvius.structure_tensor.create_st import (
    StructureTensorInferer,
    _eigh_and_sanitize,
    _compute_eigenvectors,
    _finalize_structure_tensor_torch
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

def test_border_trim_math_matches_patch_extent():
    """
    Regression test for border-aware trimming: when the padded read is clipped
    by the volume boundary, the trim offsets must be derived from (z0-za, ...),
    not from fixed pad sizes. This test emulates the indexing math and ensures
    the trimmed block is exactly the patch size.
    """
    # pretend patch size and total pad (pz,py,px)
    patch = (5, 5, 5)
    pz, py, px = (4, 3, 2)  # could come from Pavel + optional Gaussian

    # a small "volume" so the pad will clip on the z-min and x-max borders
    vol_Z, vol_Y, vol_X = (7, 8, 6)

    # pick a patch that starts on z=0 (border), near y mid, and ends at x=max
    z0, y0, x0 = (0, 2, vol_X - patch[2])
    z1, y1, x1 = (z0 + patch[0], y0 + patch[1], x0 + patch[2])

    # compute the padded slab bounds, clamped to the volume
    za, zb = max(z0 - pz, 0), min(z1 + pz, vol_Z)
    ya, yb = max(y0 - py, 0), min(y1 + py, vol_Y)
    xa, xb = max(x0 - px, 0), min(x1 + px, vol_X)

    # emulate a structure-tensor output over the padded region: [1,6,Zp,Yp,Xp]
    Zp, Yp, Xp = (zb - za, yb - ya, xb - xa)
    Jp = torch.zeros((1, 6, Zp, Yp, Xp), dtype=torch.float32)

    # correct border-aware trim
    tz0, ty0, tx0 = (z0 - za), (y0 - ya), (x0 - xa)
    tz1, ty1, tx1 = (tz0 + patch[0]), (ty0 + patch[1]), (tx0 + patch[2])
    J = Jp[:, :, tz0:tz1, ty0:ty1, tx0:tx1]

    assert J.shape == (1, 6, *patch), (
        f"Trimmed shape {tuple(J.shape)} != expected {(1,6,*patch)}"
    )

'''
def test_eigenanalysis_right_handed_and_oriented(tmp_path):
    """
    End-to-end check of _finalize_structure_tensor_torch:
    - computes eigensystems from a synthetic diagonal structure tensor
    - enforces right-handedness (det > 0)
    - orients first eigenvector toward +X on average (non-negative mean x-component)
    """
    # Build a tiny structure tensor volume: shape (6, Z, Y, X)
    Z, Y, X = 3, 2, 2
    st = np.zeros((6, Z, Y, X), dtype=np.float32)
    # channels = [Jzz, Jzy, Jzx, Jyy, Jyx, Jxx]
    st[0, ...] = 3.0  # Jzz
    st[3, ...] = 2.0  # Jyy
    st[5, ...] = 1.0  # Jxx

    # Write a zarr group with 'structure_tensor'
    zarr_path = os.path.join(tmp_path, "st.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    root.create_dataset(
        "structure_tensor",
        data=st,
        chunks=(1, Z, Y, X),   # small chunks OK
        dtype="f4"
    )

    # Run eigenanalysis on CPU with small chunks
    _finalize_structure_tensor_torch(
        zarr_path=zarr_path,
        chunk_size=(Z, Y, X),
        num_workers=0,
        compressor=None,
        verbose=False,
        swap_eigenvectors=False,
        device="cpu",
    )

    # Load results and validate
    ev = zarr.open_group(zarr_path, mode="r")["eigenvectors"][...]   # (9,Z,Y,X)
    # reshape to (3 eigenvectors, 3 components, Z,Y,X)
    v = torch.from_numpy(ev).view(3, 3, Z, Y, X).float()

    # 1) Right-handed: determinant of [v0 v1 v2] > 0 everywhere
    V = v.permute(2, 3, 4, 0, 1).reshape(-1, 3, 3)  # [N,3,3]
    det = torch.linalg.det(V)
    assert torch.all(det > 0), "Eigenframe determinant must be positive everywhere"

    # 2) Orientation: mean x-component of first eigenvector is >= 0
    mean_x = v[0, 0, ...].mean().item()
    assert mean_x >= -1e-6, f"First eigenvector should point on average toward +X, got mean {mean_x}"
'''

def test_eigh_and_sanitize_handles_nans_infs():
    # A small batch of 3x3 matrices with NaNs/Infs
    M = torch.tensor([
        [[float('nan'), 1.0, 0.0],
         [1.0, 2.0, float('inf')],
         [0.0, float('-inf'), 3.0]],
        [[1.0, 0.0, 0.0],
         [0.0, float('nan'), 0.0],
         [0.0, 0.0, 2.0]]
    ], dtype=torch.float32).unsqueeze(0)  # shape [1,3,3] for consistency

    w, v = _eigh_and_sanitize(M)
    assert torch.isfinite(w).all()
    assert torch.isfinite(v).all()
    # No NaNs/Infs remain anywhere
    assert (~torch.isnan(w)).all() and (~torch.isnan(v)).all()
    assert (~torch.isinf(w)).all() and (~torch.isinf(v)).all()

'''
def test_eigenanalysis_chunk_defaults_and_shapes(tmp_path):
    """
    If chunk_size=None, eigenanalysis should use the source chunks (minus the channel dim),
    i.e., eigenvectors/eigenvalues chunks should be (1, cz, cy, cx).
    """
    Z, Y, X = 4, 3, 3
    st = np.zeros((6, Z, Y, X), dtype=np.float32)
    st[0] = 3.0; st[3] = 2.0; st[5] = 1.0
    zarr_path = os.path.join(tmp_path, "st.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    # choose specific chunks to propagate
    cz, cy, cx = 2, 2, 3
    root.create_dataset("structure_tensor", data=st, chunks=(6, cz, cy, cx), dtype="f4")

    _finalize_structure_tensor_torch(
        zarr_path=zarr_path,
        chunk_size=None,      # <-- rely on defaults from source chunks
        num_workers=0,
        compressor=None,
        verbose=False,
        swap_eigenvectors=False,
    )

    ev = zarr.open_group(zarr_path, mode="r")["eigenvectors"]
    ew = zarr.open_group(zarr_path, mode="r")["eigenvalues"]
    assert ev.shape == (9, Z, Y, X)
    assert ew.shape == (3, Z, Y, X)
    assert ev.chunks == (1, cz, cy, cx)
    assert ew.chunks == (1, cz, cy, cx)


def test_swap_eigenvectors_flag(tmp_path):
    """
    swap_eigenvectors=True should exchange eigenpairs 0 and 1.
    """
    Z, Y, X = 2, 2, 2
    st = np.zeros((6, Z, Y, X), dtype=np.float32)
    # diag(J) = [Jzz, Jyy, Jxx] = [3,2,1]
    st[0] = 3.0; st[3] = 2.0; st[5] = 1.0
    zarr_path = os.path.join(tmp_path, "st.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    root.create_dataset("structure_tensor", data=st, chunks=(1, Z, Y, X), dtype="f4")

    # no swap
    _finalize_structure_tensor_torch(
        zarr_path=zarr_path,
        chunk_size=(Z, Y, X),
        num_workers=0,
        compressor=None,
        verbose=False,
        swap_eigenvectors=False,
    )
    w_no = zarr.open_group(zarr_path, mode="r")["eigenvalues"][...]

    # with swap (overwrite results)
    _finalize_structure_tensor_torch(
        zarr_path=zarr_path,
        chunk_size=(Z, Y, X),
        num_workers=0,
        compressor=None,
        verbose=False,
        swap_eigenvectors=True,
    )
    w_sw = zarr.open_group(zarr_path, mode="r")["eigenvalues"][...]
    # eigenvalues channels 0 and 1 swapped
    assert np.allclose(w_sw[0], w_no[1]) and np.allclose(w_sw[1], w_no[0])
    assert np.allclose(w_sw[2], w_no[2])'''