#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward: trilinear sample from sparse chunks -> float32 output (no clamping)
__global__ void sparse_grid_sample_3d_u8_diff_fwd_kernel(
    const long long* __restrict__ chunk_table,  // (cZ, cY, cX) — device ptrs
    int cZ, int cY, int cX,
    int C,
    const float* __restrict__ grid,
    int N,
    long long grid_stride_d,
    long long grid_stride_h,
    long long grid_stride_w,
    long long grid_stride_c,
    int W_grid, int H_grid,
    const float* __restrict__ offset,
    const float* __restrict__ inv_scale,
    float* __restrict__ out                     // (C, N) contiguous float32
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int w = n % W_grid;
    int tmp = n / W_grid;
    int h = tmp % H_grid;
    int d = tmp / H_grid;

    long long grid_base = d * grid_stride_d + h * grid_stride_h + w * grid_stride_w;
    float px = (grid[grid_base + 0 * grid_stride_c] - offset[0]) * inv_scale[0];
    float py = (grid[grid_base + 1 * grid_stride_c] - offset[1]) * inv_scale[1];
    float pz = (grid[grid_base + 2 * grid_stride_c] - offset[2]) * inv_scale[2];

    // NaN/Inf guard: non-finite coords → output 0
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) {
        for (int c = 0; c < C; c++) out[(long long)c * N + n] = 0.0f;
        return;
    }

    int ci_x = (int)floorf(px / 32.0f);
    int ci_y = (int)floorf(py / 32.0f);
    int ci_z = (int)floorf(pz / 32.0f);

    if (ci_x < 0 || ci_x >= cX || ci_y < 0 || ci_y >= cY || ci_z < 0 || ci_z >= cZ) {
        for (int c = 0; c < C; c++) out[(long long)c * N + n] = 0.0f;
        return;
    }

    long long ptr_val = chunk_table[(long long)ci_z * cY * cX + ci_y * cX + ci_x];
    if (ptr_val == 0) {
        for (int c = 0; c < C; c++) out[(long long)c * N + n] = 0.0f;
        return;
    }
    const uint8_t* chunk = (const uint8_t*)ptr_val;

    float lx = px - ci_x * 32.0f + 1.0f;
    float ly = py - ci_y * 32.0f + 1.0f;
    float lz = pz - ci_z * 32.0f + 1.0f;

    float lx0f = floorf(lx), ly0f = floorf(ly), lz0f = floorf(lz);
    float fx = lx - lx0f, fy = ly - ly0f, fz = lz - lz0f;

    int ix0 = max(0, min(33, (int)lx0f));
    int iy0 = max(0, min(33, (int)ly0f));
    int iz0 = max(0, min(33, (int)lz0f));
    int ix1 = max(0, min(33, ix0 + 1));
    int iy1 = max(0, min(33, iy0 + 1));
    int iz1 = max(0, min(33, iz0 + 1));

    float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    float w100 = fx          * (1.0f - fy) * (1.0f - fz);
    float w010 = (1.0f - fx) * fy          * (1.0f - fz);
    float w110 = fx          * fy          * (1.0f - fz);
    float w001 = (1.0f - fx) * (1.0f - fy) * fz;
    float w101 = fx          * (1.0f - fy) * fz;
    float w011 = (1.0f - fx) * fy          * fz;
    float w111 = fx          * fy          * fz;

    const int S_z = 34 * 34;
    const int S_y = 34;

    for (int c = 0; c < C; c++) {
        const uint8_t* ch = chunk + (long long)c * 34 * 34 * 34;
        float val = 0.0f;
        val += w000 * (float)ch[iz0 * S_z + iy0 * S_y + ix0];
        val += w100 * (float)ch[iz0 * S_z + iy0 * S_y + ix1];
        val += w010 * (float)ch[iz0 * S_z + iy1 * S_y + ix0];
        val += w110 * (float)ch[iz0 * S_z + iy1 * S_y + ix1];
        val += w001 * (float)ch[iz1 * S_z + iy0 * S_y + ix0];
        val += w101 * (float)ch[iz1 * S_z + iy0 * S_y + ix1];
        val += w011 * (float)ch[iz1 * S_z + iy1 * S_y + ix0];
        val += w111 * (float)ch[iz1 * S_z + iy1 * S_y + ix1];
        out[(long long)c * N + n] = val;
    }
}


// Backward: compute grad w.r.t. grid positions
__global__ void sparse_grid_sample_3d_u8_diff_bwd_kernel(
    const long long* __restrict__ chunk_table,
    int cZ, int cY, int cX,
    int C,
    const float* __restrict__ grid,
    int N,
    long long grid_stride_d,
    long long grid_stride_h,
    long long grid_stride_w,
    long long grid_stride_c,
    int W_grid, int H_grid,
    const float* __restrict__ offset,
    const float* __restrict__ inv_scale,
    const float* __restrict__ grad_output,       // (C, N) contiguous
    float* __restrict__ grad_grid                // (N, 3) contiguous
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int w = n % W_grid;
    int tmp = n / W_grid;
    int h = tmp % H_grid;
    int d = tmp / H_grid;

    long long grid_base = d * grid_stride_d + h * grid_stride_h + w * grid_stride_w;
    float isx = inv_scale[0], isy = inv_scale[1], isz = inv_scale[2];
    float px = (grid[grid_base + 0 * grid_stride_c] - offset[0]) * isx;
    float py = (grid[grid_base + 1 * grid_stride_c] - offset[1]) * isy;
    float pz = (grid[grid_base + 2 * grid_stride_c] - offset[2]) * isz;

    // NaN/Inf guard
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) {
        grad_grid[n * 3 + 0] = 0.0f;
        grad_grid[n * 3 + 1] = 0.0f;
        grad_grid[n * 3 + 2] = 0.0f;
        return;
    }

    int ci_x = (int)floorf(px / 32.0f);
    int ci_y = (int)floorf(py / 32.0f);
    int ci_z = (int)floorf(pz / 32.0f);

    if (ci_x < 0 || ci_x >= cX || ci_y < 0 || ci_y >= cY || ci_z < 0 || ci_z >= cZ) {
        grad_grid[n * 3 + 0] = 0.0f;
        grad_grid[n * 3 + 1] = 0.0f;
        grad_grid[n * 3 + 2] = 0.0f;
        return;
    }

    long long ptr_val = chunk_table[(long long)ci_z * cY * cX + ci_y * cX + ci_x];
    if (ptr_val == 0) {
        grad_grid[n * 3 + 0] = 0.0f;
        grad_grid[n * 3 + 1] = 0.0f;
        grad_grid[n * 3 + 2] = 0.0f;
        return;
    }
    const uint8_t* chunk = (const uint8_t*)ptr_val;

    float lx = px - ci_x * 32.0f + 1.0f;
    float ly = py - ci_y * 32.0f + 1.0f;
    float lz = pz - ci_z * 32.0f + 1.0f;

    float lx0f = floorf(lx), ly0f = floorf(ly), lz0f = floorf(lz);
    float fx = lx - lx0f, fy = ly - ly0f, fz = lz - lz0f;

    int ix0 = max(0, min(33, (int)lx0f));
    int iy0 = max(0, min(33, (int)ly0f));
    int iz0 = max(0, min(33, (int)lz0f));
    int ix1 = max(0, min(33, ix0 + 1));
    int iy1 = max(0, min(33, iy0 + 1));
    int iz1 = max(0, min(33, iz0 + 1));

    const int S_z = 34 * 34;
    const int S_y = 34;

    float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;

    for (int c = 0; c < C; c++) {
        const uint8_t* ch = chunk + (long long)c * 34 * 34 * 34;
        float go = grad_output[(long long)c * N + n];

        float v000 = (float)ch[iz0 * S_z + iy0 * S_y + ix0];
        float v100 = (float)ch[iz0 * S_z + iy0 * S_y + ix1];
        float v010 = (float)ch[iz0 * S_z + iy1 * S_y + ix0];
        float v110 = (float)ch[iz0 * S_z + iy1 * S_y + ix1];
        float v001 = (float)ch[iz1 * S_z + iy0 * S_y + ix0];
        float v101 = (float)ch[iz1 * S_z + iy0 * S_y + ix1];
        float v011 = (float)ch[iz1 * S_z + iy1 * S_y + ix0];
        float v111 = (float)ch[iz1 * S_z + iy1 * S_y + ix1];

        // d(interp)/d(fx)
        float dfx = (1.0f - fy) * (1.0f - fz) * (v100 - v000)
                   + fy          * (1.0f - fz) * (v110 - v010)
                   + (1.0f - fy) * fz          * (v101 - v001)
                   + fy          * fz          * (v111 - v011);

        // d(interp)/d(fy)
        float dfy = (1.0f - fx) * (1.0f - fz) * (v010 - v000)
                   + fx          * (1.0f - fz) * (v110 - v100)
                   + (1.0f - fx) * fz          * (v011 - v001)
                   + fx          * fz          * (v111 - v101);

        // d(interp)/d(fz)
        float dfz = (1.0f - fx) * (1.0f - fy) * (v001 - v000)
                   + fx          * (1.0f - fy) * (v101 - v100)
                   + (1.0f - fx) * fy          * (v011 - v010)
                   + fx          * fy          * (v111 - v110);

        // Chain rule: d(local)/d(fullres) = inv_scale
        dgx += go * dfx * isx;
        dgy += go * dfy * isy;
        dgz += go * dfz * isz;
    }

    grad_grid[n * 3 + 0] = dgx;
    grad_grid[n * 3 + 1] = dgy;
    grad_grid[n * 3 + 2] = dgz;
}


// ---- C++ wrappers ----

torch::Tensor sparse_grid_sample_3d_u8_diff_fwd(
    torch::Tensor chunk_table,
    int C,
    torch::Tensor grid,
    torch::Tensor offset,
    torch::Tensor inv_scale
) {
    TORCH_CHECK(chunk_table.is_cuda() && chunk_table.dtype() == torch::kInt64,
                "chunk_table must be CUDA int64");
    TORCH_CHECK(chunk_table.dim() == 3, "chunk_table must be (cZ, cY, cX)");
    TORCH_CHECK(chunk_table.is_contiguous(), "chunk_table must be contiguous");
    TORCH_CHECK(grid.is_cuda() && grid.dtype() == torch::kFloat32, "grid must be CUDA float32");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 3, "grid must be (D, H, W, 3)");

    int cZ = chunk_table.size(0);
    int cY = chunk_table.size(1);
    int cX = chunk_table.size(2);
    int D = grid.size(0);
    int H = grid.size(1);
    int W = grid.size(2);
    int N = D * H * W;

    long long grid_stride_d = grid.stride(0);
    long long grid_stride_h = grid.stride(1);
    long long grid_stride_w = grid.stride(2);
    long long grid_stride_c = grid.stride(3);

    auto out = torch::empty({C, N}, torch::TensorOptions().dtype(torch::kFloat32).device(chunk_table.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sparse_grid_sample_3d_u8_diff_fwd_kernel<<<blocks, threads>>>(
        (long long*)chunk_table.data_ptr<int64_t>(),
        cZ, cY, cX, C,
        grid.data_ptr<float>(),
        N,
        grid_stride_d, grid_stride_h, grid_stride_w, grid_stride_c,
        W, H,
        offset.data_ptr<float>(),
        inv_scale.data_ptr<float>(),
        out.data_ptr<float>()
    );

    return out.reshape({C, D, H, W});
}


torch::Tensor sparse_grid_sample_3d_u8_diff_bwd(
    torch::Tensor chunk_table,
    int C,
    torch::Tensor grid,
    torch::Tensor offset,
    torch::Tensor inv_scale,
    torch::Tensor grad_output  // (C, D, H, W)
) {
    TORCH_CHECK(chunk_table.is_cuda() && chunk_table.dtype() == torch::kInt64);
    TORCH_CHECK(chunk_table.dim() == 3 && chunk_table.is_contiguous());
    TORCH_CHECK(grid.is_cuda() && grid.dtype() == torch::kFloat32);
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 3);
    TORCH_CHECK(grad_output.is_cuda() && grad_output.dtype() == torch::kFloat32);
    TORCH_CHECK(grad_output.is_contiguous());

    int cZ = chunk_table.size(0);
    int cY = chunk_table.size(1);
    int cX = chunk_table.size(2);
    int D = grid.size(0);
    int H = grid.size(1);
    int W = grid.size(2);
    int N = D * H * W;

    long long grid_stride_d = grid.stride(0);
    long long grid_stride_h = grid.stride(1);
    long long grid_stride_w = grid.stride(2);
    long long grid_stride_c = grid.stride(3);

    auto grad_grid = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(chunk_table.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sparse_grid_sample_3d_u8_diff_bwd_kernel<<<blocks, threads>>>(
        (long long*)chunk_table.data_ptr<int64_t>(),
        cZ, cY, cX, C,
        grid.data_ptr<float>(),
        N,
        grid_stride_d, grid_stride_h, grid_stride_w, grid_stride_c,
        W, H,
        offset.data_ptr<float>(),
        inv_scale.data_ptr<float>(),
        grad_output.reshape({C, N}).data_ptr<float>(),
        grad_grid.data_ptr<float>()
    );

    return grad_grid.reshape({D, H, W, 3});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_grid_sample_3d_u8_diff_fwd", &sparse_grid_sample_3d_u8_diff_fwd,
          "Sparse trilinear 3D grid_sample uint8 -> float32 (CUDA forward)");
    m.def("sparse_grid_sample_3d_u8_diff_bwd", &sparse_grid_sample_3d_u8_diff_bwd,
          "Gradient w.r.t. grid for sparse trilinear 3D grid_sample (CUDA backward)");
}
