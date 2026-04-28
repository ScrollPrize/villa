#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward: trilinear sample uint8 volume -> float32 output (no clamping)
__global__ void grid_sample_3d_u8_diff_fwd_kernel(
    const uint8_t* __restrict__ volume,  // (C, Z, Y, X) contiguous
    int C, int Z, int Y, int X,
    const float* __restrict__ grid,      // (D, H, W, 3) possibly non-contiguous
    int N,                               // D*H*W
    long long grid_stride_d,
    long long grid_stride_h,
    long long grid_stride_w,
    long long grid_stride_c,
    int W_grid, int H_grid,
    const float* __restrict__ offset,    // (3,)
    const float* __restrict__ inv_scale, // (3,)
    float* __restrict__ out              // (C, N) contiguous float32
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

    float fx0 = floorf(px);
    float fy0 = floorf(py);
    float fz0 = floorf(pz);
    float fx = px - fx0;
    float fy = py - fy0;
    float fz = pz - fz0;

    int ix0 = (int)fx0, iy0 = (int)fy0, iz0 = (int)fz0;
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    bool x0_ok = (ix0 >= 0 && ix0 < X);
    bool x1_ok = (ix1 >= 0 && ix1 < X);
    bool y0_ok = (iy0 >= 0 && iy0 < Y);
    bool y1_ok = (iy1 >= 0 && iy1 < Y);
    bool z0_ok = (iz0 >= 0 && iz0 < Z);
    bool z1_ok = (iz1 >= 0 && iz1 < Z);

    float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    float w100 = fx          * (1.0f - fy) * (1.0f - fz);
    float w010 = (1.0f - fx) * fy          * (1.0f - fz);
    float w110 = fx          * fy          * (1.0f - fz);
    float w001 = (1.0f - fx) * (1.0f - fy) * fz;
    float w101 = fx          * (1.0f - fy) * fz;
    float w011 = (1.0f - fx) * fy          * fz;
    float w111 = fx          * fy          * fz;

    long long stride_z = (long long)Y * X;
    long long stride_y = X;

    for (int c = 0; c < C; c++) {
        const uint8_t* vol_c = volume + (long long)c * Z * stride_z;
        float val = 0.0f;

        if (z0_ok && y0_ok && x0_ok) val += w000 * (float)vol_c[(long long)iz0 * stride_z + iy0 * stride_y + ix0];
        if (z0_ok && y0_ok && x1_ok) val += w100 * (float)vol_c[(long long)iz0 * stride_z + iy0 * stride_y + ix1];
        if (z0_ok && y1_ok && x0_ok) val += w010 * (float)vol_c[(long long)iz0 * stride_z + iy1 * stride_y + ix0];
        if (z0_ok && y1_ok && x1_ok) val += w110 * (float)vol_c[(long long)iz0 * stride_z + iy1 * stride_y + ix1];
        if (z1_ok && y0_ok && x0_ok) val += w001 * (float)vol_c[(long long)iz1 * stride_z + iy0 * stride_y + ix0];
        if (z1_ok && y0_ok && x1_ok) val += w101 * (float)vol_c[(long long)iz1 * stride_z + iy0 * stride_y + ix1];
        if (z1_ok && y1_ok && x0_ok) val += w011 * (float)vol_c[(long long)iz1 * stride_z + iy1 * stride_y + ix0];
        if (z1_ok && y1_ok && x1_ok) val += w111 * (float)vol_c[(long long)iz1 * stride_z + iy1 * stride_y + ix1];

        out[(long long)c * N + n] = val;
    }
}


// Backward: compute grad w.r.t. grid positions
__global__ void grid_sample_3d_u8_diff_bwd_kernel(
    const uint8_t* __restrict__ volume,  // (C, Z, Y, X) contiguous
    int C, int Z, int Y, int X,
    const float* __restrict__ grid,      // (D, H, W, 3) possibly non-contiguous
    int N,
    long long grid_stride_d,
    long long grid_stride_h,
    long long grid_stride_w,
    long long grid_stride_c,
    int W_grid, int H_grid,
    const float* __restrict__ offset,    // (3,)
    const float* __restrict__ inv_scale, // (3,)
    const float* __restrict__ grad_output, // (C, N) contiguous
    float* __restrict__ grad_grid          // (N, 3) contiguous
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

    float fx0 = floorf(px);
    float fy0 = floorf(py);
    float fz0 = floorf(pz);
    float fx = px - fx0;
    float fy = py - fy0;
    float fz = pz - fz0;

    int ix0 = (int)fx0, iy0 = (int)fy0, iz0 = (int)fz0;
    int ix1 = ix0 + 1, iy1 = iy0 + 1, iz1 = iz0 + 1;

    bool x0_ok = (ix0 >= 0 && ix0 < X);
    bool x1_ok = (ix1 >= 0 && ix1 < X);
    bool y0_ok = (iy0 >= 0 && iy0 < Y);
    bool y1_ok = (iy1 >= 0 && iy1 < Y);
    bool z0_ok = (iz0 >= 0 && iz0 < Z);
    bool z1_ok = (iz1 >= 0 && iz1 < Z);

    long long stride_z = (long long)Y * X;
    long long stride_y = X;

    float dgx = 0.0f, dgy = 0.0f, dgz = 0.0f;

    for (int c = 0; c < C; c++) {
        const uint8_t* vol_c = volume + (long long)c * Z * stride_z;
        float go = grad_output[(long long)c * N + n];

        // Read 8 corner values (0 if out of bounds)
        float v000 = (z0_ok && y0_ok && x0_ok) ? (float)vol_c[(long long)iz0 * stride_z + iy0 * stride_y + ix0] : 0.0f;
        float v100 = (z0_ok && y0_ok && x1_ok) ? (float)vol_c[(long long)iz0 * stride_z + iy0 * stride_y + ix1] : 0.0f;
        float v010 = (z0_ok && y1_ok && x0_ok) ? (float)vol_c[(long long)iz0 * stride_z + iy1 * stride_y + ix0] : 0.0f;
        float v110 = (z0_ok && y1_ok && x1_ok) ? (float)vol_c[(long long)iz0 * stride_z + iy1 * stride_y + ix1] : 0.0f;
        float v001 = (z1_ok && y0_ok && x0_ok) ? (float)vol_c[(long long)iz1 * stride_z + iy0 * stride_y + ix0] : 0.0f;
        float v101 = (z1_ok && y0_ok && x1_ok) ? (float)vol_c[(long long)iz1 * stride_z + iy0 * stride_y + ix1] : 0.0f;
        float v011 = (z1_ok && y1_ok && x0_ok) ? (float)vol_c[(long long)iz1 * stride_z + iy1 * stride_y + ix0] : 0.0f;
        float v111 = (z1_ok && y1_ok && x1_ok) ? (float)vol_c[(long long)iz1 * stride_z + iy1 * stride_y + ix1] : 0.0f;

        // d(interp)/d(fx): differentiate trilinear weights w.r.t. fx
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

        // Chain rule: d(grid_coord)/d(fullres_coord) = inv_scale
        dgx += go * dfx * isx;
        dgy += go * dfy * isy;
        dgz += go * dfz * isz;
    }

    grad_grid[n * 3 + 0] = dgx;
    grad_grid[n * 3 + 1] = dgy;
    grad_grid[n * 3 + 2] = dgz;
}


// ---- C++ wrappers ----

torch::Tensor grid_sample_3d_u8_diff_fwd(
    torch::Tensor volume,     // (C, Z, Y, X) uint8 cuda
    torch::Tensor grid,       // (D, H, W, 3) float32 cuda
    torch::Tensor offset,     // (3,) float32 cuda
    torch::Tensor inv_scale   // (3,) float32 cuda
) {
    TORCH_CHECK(volume.is_cuda() && volume.dtype() == torch::kUInt8, "volume must be CUDA uint8");
    TORCH_CHECK(volume.dim() == 4, "volume must be (C, Z, Y, X)");
    TORCH_CHECK(volume.is_contiguous(), "volume must be contiguous");
    TORCH_CHECK(grid.is_cuda() && grid.dtype() == torch::kFloat32, "grid must be CUDA float32");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 3, "grid must be (D, H, W, 3)");
    TORCH_CHECK(offset.is_cuda() && offset.numel() == 3, "offset must be CUDA (3,)");
    TORCH_CHECK(inv_scale.is_cuda() && inv_scale.numel() == 3, "inv_scale must be CUDA (3,)");

    int C = volume.size(0);
    int Z = volume.size(1);
    int Y = volume.size(2);
    int X = volume.size(3);
    int D = grid.size(0);
    int H = grid.size(1);
    int W = grid.size(2);
    int N = D * H * W;

    long long grid_stride_d = grid.stride(0);
    long long grid_stride_h = grid.stride(1);
    long long grid_stride_w = grid.stride(2);
    long long grid_stride_c = grid.stride(3);

    auto out = torch::empty({C, N}, torch::TensorOptions().dtype(torch::kFloat32).device(volume.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    grid_sample_3d_u8_diff_fwd_kernel<<<blocks, threads>>>(
        volume.data_ptr<uint8_t>(),
        C, Z, Y, X,
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


torch::Tensor grid_sample_3d_u8_diff_bwd(
    torch::Tensor volume,       // (C, Z, Y, X) uint8 cuda
    torch::Tensor grid,         // (D, H, W, 3) float32 cuda
    torch::Tensor offset,       // (3,) float32 cuda
    torch::Tensor inv_scale,    // (3,) float32 cuda
    torch::Tensor grad_output   // (C, D, H, W) float32 cuda
) {
    TORCH_CHECK(volume.is_cuda() && volume.dtype() == torch::kUInt8, "volume must be CUDA uint8");
    TORCH_CHECK(volume.dim() == 4, "volume must be (C, Z, Y, X)");
    TORCH_CHECK(volume.is_contiguous(), "volume must be contiguous");
    TORCH_CHECK(grid.is_cuda() && grid.dtype() == torch::kFloat32, "grid must be CUDA float32");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 3, "grid must be (D, H, W, 3)");
    TORCH_CHECK(grad_output.is_cuda() && grad_output.dtype() == torch::kFloat32, "grad_output must be CUDA float32");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");

    int C = volume.size(0);
    int Z = volume.size(1);
    int Y = volume.size(2);
    int X = volume.size(3);
    int D = grid.size(0);
    int H = grid.size(1);
    int W = grid.size(2);
    int N = D * H * W;

    long long grid_stride_d = grid.stride(0);
    long long grid_stride_h = grid.stride(1);
    long long grid_stride_w = grid.stride(2);
    long long grid_stride_c = grid.stride(3);

    auto grad_grid = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(volume.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    grid_sample_3d_u8_diff_bwd_kernel<<<blocks, threads>>>(
        volume.data_ptr<uint8_t>(),
        C, Z, Y, X,
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
    m.def("grid_sample_3d_u8_diff_fwd", &grid_sample_3d_u8_diff_fwd,
          "Trilinear 3D grid_sample on uint8 volume -> float32 (CUDA forward)");
    m.def("grid_sample_3d_u8_diff_bwd", &grid_sample_3d_u8_diff_bwd,
          "Gradient w.r.t. grid positions for trilinear 3D grid_sample (CUDA backward)");
}
