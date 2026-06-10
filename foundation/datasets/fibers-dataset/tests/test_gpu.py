import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
def test_eigenvalues_analytical():
    """Test the analytical 3x3 eigensolver against NumPy's eigvalsh."""
    np.random.seed(42)
    # Generate random symmetric matrices
    A = np.random.rand(10, 10, 3, 3).astype(np.float32)
    for i in range(3):
        for j in range(i+1, 3):
            A[..., i, j] = A[..., j, i]
            
    A_cp = cp.array(A)
    
    eigvals_cp = tools.compute_eigenvalues_3x3_batch(A_cp)
    eigvals_np = np.linalg.eigvalsh(A)
    
    np.testing.assert_allclose(cp.asnumpy(eigvals_cp), eigvals_np, rtol=1e-3, atol=1e-4)

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
def test_vesselness_parity():
    """Ensure CPU (NumPy) and GPU (CuPy) backends produce similar results."""
    np.random.seed(42)
    volume_np = np.random.rand(32, 32, 32).astype(np.float32)
    volume_cp = cp.array(volume_np)
    
    res_np = tools.detect_vesselness(volume_np)
    res_cp = tools.detect_vesselness(volume_cp)
    
    np.testing.assert_allclose(res_np, cp.asnumpy(res_cp), rtol=1e-3, atol=1e-4)

def test_tiled_parity():
    """Tiled execution matches the dense path when the halo covers the filter support."""
    np.random.seed(42)
    volume_np = np.random.rand(64, 64, 64).astype(np.float32)

    # Default gauss_sigma=2 -> smoothing support 4*sigma=8 voxels (+2 for the
    # finite-difference Hessian), fully inside halo=16.
    res_dense = tools.detect_vesselness(volume_np.copy())
    res_tiled = tools.detect_vesselness_tiled(volume_np.copy(), block_size=32, halo=16)

    np.testing.assert_allclose(res_dense, res_tiled, rtol=1e-3, atol=1e-4)

def test_tiled_parity_ridges():
    """Same parity check for the ridge filter."""
    np.random.seed(7)
    volume_np = np.random.rand(64, 64, 64).astype(np.float32)

    res_dense = tools.detect_ridges(volume_np.copy())
    res_tiled = tools.detect_ridges_tiled(volume_np.copy(), block_size=32, halo=16)

    np.testing.assert_allclose(res_dense, res_tiled, rtol=1e-3, atol=1e-4)
