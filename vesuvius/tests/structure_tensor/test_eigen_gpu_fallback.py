"""GPU eigen solves that run out of memory must fall back to the CPU solve.

With torch 2.12.1/CUDA 13 cuSOLVER's batched eigh asks ~0.5 MB of workspace per
3x3 float64 matrix, so create_st's 1,048,576-voxel batch requests ~530 GiB and
vesuvius.compute_st failed on every GPU tried (T4, L4, A100).
"""

from vesuvius.structure_tensor import create_st


def test_gpu_eigen_oom_falls_back_to_cpu():
    # cuSOLVER's batched eigh asks ~0.5 MB per 3x3 float64 matrix (torch 2.12/CUDA 13),
    # so the 1,048,576-voxel batch requests ~530 GiB and raised OutOfMemoryError on
    # T4/L4/A100 alike; only "cusolver"-message errors used to reach the CPU fallback.
    import torch

    cuda = torch.device("cuda")
    oom = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 530.22 GiB.")
    assert create_st._should_solve_on_cpu(oom, cuda)
    assert create_st._should_solve_on_cpu(RuntimeError("cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR"), cuda)
    assert not create_st._should_solve_on_cpu(RuntimeError("shape mismatch"), cuda)
    assert not create_st._should_solve_on_cpu(oom, torch.device("cpu"))
