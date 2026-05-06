#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace {

constexpr int kChunkSize = 32;

__global__ void mark_needed_chunks_kernel(
    const float* __restrict__ xyz,
    int64_t n_points,
    const float* __restrict__ origin,
    const float* __restrict__ spacing,
    uint8_t* __restrict__ needed,
    int cZ,
    int cY,
    int cX) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_points) {
        return;
    }

    const float x = xyz[idx * 3 + 0];
    const float y = xyz[idx * 3 + 1];
    const float z = xyz[idx * 3 + 2];

    int cx = isfinite(x) ? static_cast<int>((x - origin[0]) / spacing[0] / kChunkSize) : 0;
    int cy = isfinite(y) ? static_cast<int>((y - origin[1]) / spacing[1] / kChunkSize) : 0;
    int cz = isfinite(z) ? static_cast<int>((z - origin[2]) / spacing[2] / kChunkSize) : 0;

    cx = max(0, min(cX - 1, cx));
    cy = max(0, min(cY - 1, cy));
    cz = max(0, min(cZ - 1, cz));

    // Mark the 26-neighborhood directly. This matches the Python prefetch
    // dilation while keeping the decision on the GPU.
    for (int dz = -1; dz <= 1; ++dz) {
        const int zz = cz + dz;
        if (zz < 0 || zz >= cZ) {
            continue;
        }
        for (int dy = -1; dy <= 1; ++dy) {
            const int yy = cy + dy;
            if (yy < 0 || yy >= cY) {
                continue;
            }
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = cx + dx;
                if (xx < 0 || xx >= cX) {
                    continue;
                }
                needed[(zz * cY + yy) * cX + xx] = 1;
            }
        }
    }
}

}  // namespace

torch::Tensor missing_chunks(
    torch::Tensor xyz_fullres,
    torch::Tensor chunk_table,
    torch::Tensor origin,
    torch::Tensor spacing) {
    TORCH_CHECK(xyz_fullres.is_cuda(), "xyz_fullres must be CUDA");
    TORCH_CHECK(chunk_table.is_cuda(), "chunk_table must be CUDA");
    TORCH_CHECK(origin.is_cuda(), "origin must be CUDA");
    TORCH_CHECK(spacing.is_cuda(), "spacing must be CUDA");
    TORCH_CHECK(xyz_fullres.dtype() == torch::kFloat32, "xyz_fullres must be float32");
    TORCH_CHECK(origin.dtype() == torch::kFloat32, "origin must be float32");
    TORCH_CHECK(spacing.dtype() == torch::kFloat32, "spacing must be float32");
    TORCH_CHECK(chunk_table.dtype() == torch::kInt64, "chunk_table must be int64");
    TORCH_CHECK(xyz_fullres.size(-1) == 3, "xyz_fullres last dimension must be 3");
    TORCH_CHECK(chunk_table.dim() == 3, "chunk_table must be 3D");
    TORCH_CHECK(origin.numel() == 3, "origin must have 3 elements");
    TORCH_CHECK(spacing.numel() == 3, "spacing must have 3 elements");

    auto xyz = xyz_fullres.contiguous();
    auto origin_c = origin.contiguous();
    auto spacing_c = spacing.contiguous();

    const int cZ = static_cast<int>(chunk_table.size(0));
    const int cY = static_cast<int>(chunk_table.size(1));
    const int cX = static_cast<int>(chunk_table.size(2));
    const int64_t n_points = xyz.numel() / 3;

    auto needed = torch::zeros({cZ, cY, cX},
                               torch::TensorOptions().device(xyz.device()).dtype(torch::kUInt8));
    if (n_points > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((n_points + threads - 1) / threads);
        mark_needed_chunks_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            xyz.data_ptr<float>(),
            n_points,
            origin_c.data_ptr<float>(),
            spacing_c.data_ptr<float>(),
            needed.data_ptr<uint8_t>(),
            cZ,
            cY,
            cX);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto missing_mask = needed.to(torch::kBool).logical_and(chunk_table.eq(0));
    return torch::nonzero(missing_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("missing_chunks", &missing_chunks,
          "Compute missing sparse-cache chunk coordinates from sample positions");
}
