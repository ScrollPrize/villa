#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void grid_sample_3d_u8_kernel(
    const uint8_t* __restrict__ volume,  // (C, Z, Y, X) contiguous
    int C, int Z, int Y, int X,
    const float* __restrict__ grid,      // (D, H, W, 3) possibly non-contiguous
    int N,                               // D*H*W
    long long grid_stride_d,             // stride along D dim (in floats)
    long long grid_stride_h,             // stride along H dim
    long long grid_stride_w,             // stride along W dim
    long long grid_stride_c,             // stride along last (xyz) dim
    int W_grid,                          // grid width for d,h,w recovery
    int H_grid,                          // grid height
    const float* __restrict__ offset,    // (3,)
    const float* __restrict__ inv_scale, // (3,)
    uint8_t* __restrict__ out            // (C, N) contiguous
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Recover (d, h, w) from flat index n
    int w = n % W_grid;
    int tmp = n / W_grid;
    int h = tmp % H_grid;
    int d = tmp / H_grid;

    // Index into non-contiguous grid using strides (all 4 dims)
    long long grid_base = d * grid_stride_d + h * grid_stride_h + w * grid_stride_w;
    float px = (grid[grid_base + 0 * grid_stride_c] - offset[0]) * inv_scale[0];
    float py = (grid[grid_base + 1 * grid_stride_c] - offset[1]) * inv_scale[1];
    float pz = (grid[grid_base + 2 * grid_stride_c] - offset[2]) * inv_scale[2];

    // Floor and fractional
    float fx0 = floorf(px);
    float fy0 = floorf(py);
    float fz0 = floorf(pz);
    float fx = px - fx0;
    float fy = py - fy0;
    float fz = pz - fz0;

    int ix0 = (int)fx0;
    int iy0 = (int)fy0;
    int iz0 = (int)fz0;
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;

    // Bounds check flags
    bool x0_ok = (ix0 >= 0 && ix0 < X);
    bool x1_ok = (ix1 >= 0 && ix1 < X);
    bool y0_ok = (iy0 >= 0 && iy0 < Y);
    bool y1_ok = (iy1 >= 0 && iy1 < Y);
    bool z0_ok = (iz0 >= 0 && iz0 < Z);
    bool z1_ok = (iz1 >= 0 && iz1 < Z);

    // Trilinear weights
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

        int ival = (int)roundf(val);
        if (ival < 0) ival = 0;
        if (ival > 255) ival = 255;
        out[(long long)c * N + n] = (uint8_t)ival;
    }
}


torch::Tensor grid_sample_3d_u8(
    torch::Tensor volume,     // (C, Z, Y, X) uint8 cuda
    torch::Tensor grid,       // (D, H, W, 3) float32 cuda — fullres coords
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

    // Get grid strides (in number of floats) for non-contiguous support
    long long grid_stride_d = grid.stride(0);
    long long grid_stride_h = grid.stride(1);
    long long grid_stride_w = grid.stride(2);
    long long grid_stride_c = grid.stride(3);

    auto out = torch::empty({C, N}, torch::TensorOptions().dtype(torch::kUInt8).device(volume.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    grid_sample_3d_u8_kernel<<<blocks, threads>>>(
        volume.data_ptr<uint8_t>(),
        C, Z, Y, X,
        grid.data_ptr<float>(),
        N,
        grid_stride_d, grid_stride_h, grid_stride_w, grid_stride_c,
        W, H,
        offset.data_ptr<float>(),
        inv_scale.data_ptr<float>(),
        out.data_ptr<uint8_t>()
    );

    return out.reshape({C, D, H, W});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_sample_3d_u8", &grid_sample_3d_u8,
          "Trilinear 3D grid_sample on uint8 volume (CUDA)");
}
