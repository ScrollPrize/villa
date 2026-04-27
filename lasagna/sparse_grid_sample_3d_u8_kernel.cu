#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sparse_grid_sample_3d_u8_kernel(
    const long long* __restrict__ chunk_table,  // (cZ, cY, cX) contiguous — device ptrs (0=empty)
    int cZ, int cY, int cX,
    int C,                                       // channels per chunk
    const float* __restrict__ grid,              // (D, H, W, 3) possibly non-contiguous
    int N,                                       // D*H*W
    long long grid_stride_d,
    long long grid_stride_h,
    long long grid_stride_w,
    long long grid_stride_c,
    int W_grid, int H_grid,
    const float* __restrict__ offset,            // (3,)
    const float* __restrict__ inv_scale,         // (3,)
    uint8_t* __restrict__ out                    // (C, N) contiguous
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Recover (d, h, w) from flat index
    int w = n % W_grid;
    int tmp = n / W_grid;
    int h = tmp % H_grid;
    int d = tmp / H_grid;

    // Grid position -> local volume coords
    long long grid_base = d * grid_stride_d + h * grid_stride_h + w * grid_stride_w;
    float px = (grid[grid_base + 0 * grid_stride_c] - offset[0]) * inv_scale[0];
    float py = (grid[grid_base + 1 * grid_stride_c] - offset[1]) * inv_scale[1];
    float pz = (grid[grid_base + 2 * grid_stride_c] - offset[2]) * inv_scale[2];

    // Chunk coord
    int ci_x = (int)floorf(px / 32.0f);
    int ci_y = (int)floorf(py / 32.0f);
    int ci_z = (int)floorf(pz / 32.0f);

    // Bounds check chunk grid
    if (ci_x < 0 || ci_x >= cX || ci_y < 0 || ci_y >= cY || ci_z < 0 || ci_z >= cZ) {
        for (int c = 0; c < C; c++) out[(long long)c * N + n] = 0;
        return;
    }

    // Look up device pointer
    long long ptr_val = chunk_table[(long long)ci_z * cY * cX + ci_y * cX + ci_x];
    if (ptr_val == 0) {
        for (int c = 0; c < C; c++) out[(long long)c * N + n] = 0;
        return;
    }
    const uint8_t* chunk = (const uint8_t*)ptr_val;

    // Local coords within padded chunk (+1 for margin)
    float lx = px - ci_x * 32.0f + 1.0f;
    float ly = py - ci_y * 32.0f + 1.0f;
    float lz = pz - ci_z * 32.0f + 1.0f;

    // Floor and fractional
    float lx0f = floorf(lx);
    float ly0f = floorf(ly);
    float lz0f = floorf(lz);
    float fx = lx - lx0f;
    float fy = ly - ly0f;
    float fz = lz - lz0f;

    int ix0 = (int)lx0f;
    int iy0 = (int)ly0f;
    int iz0 = (int)lz0f;
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;

    // Clamp to [0, 33] (padded chunk bounds)
    ix0 = max(0, min(33, ix0)); ix1 = max(0, min(33, ix1));
    iy0 = max(0, min(33, iy0)); iy1 = max(0, min(33, iy1));
    iz0 = max(0, min(33, iz0)); iz1 = max(0, min(33, iz1));

    // Trilinear weights
    float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
    float w100 = fx          * (1.0f - fy) * (1.0f - fz);
    float w010 = (1.0f - fx) * fy          * (1.0f - fz);
    float w110 = fx          * fy          * (1.0f - fz);
    float w001 = (1.0f - fx) * (1.0f - fy) * fz;
    float w101 = fx          * (1.0f - fy) * fz;
    float w011 = (1.0f - fx) * fy          * fz;
    float w111 = fx          * fy          * fz;

    // Strides within a single padded chunk (34x34x34)
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

        int ival = (int)roundf(val);
        if (ival < 0) ival = 0;
        if (ival > 255) ival = 255;
        out[(long long)c * N + n] = (uint8_t)ival;
    }
}


torch::Tensor sparse_grid_sample_3d_u8(
    torch::Tensor chunk_table,   // (cZ, cY, cX) int64 cuda — device pointers
    int C,                       // channels per chunk
    torch::Tensor grid,          // (D, H, W, 3) float32 cuda — fullres coords
    torch::Tensor offset,        // (3,) float32 cuda
    torch::Tensor inv_scale      // (3,) float32 cuda
) {
    TORCH_CHECK(chunk_table.is_cuda() && chunk_table.dtype() == torch::kInt64,
                "chunk_table must be CUDA int64");
    TORCH_CHECK(chunk_table.dim() == 3, "chunk_table must be (cZ, cY, cX)");
    TORCH_CHECK(chunk_table.is_contiguous(), "chunk_table must be contiguous");
    TORCH_CHECK(grid.is_cuda() && grid.dtype() == torch::kFloat32, "grid must be CUDA float32");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 3, "grid must be (D, H, W, 3)");
    TORCH_CHECK(offset.is_cuda() && offset.numel() == 3, "offset must be CUDA (3,)");
    TORCH_CHECK(inv_scale.is_cuda() && inv_scale.numel() == 3, "inv_scale must be CUDA (3,)");

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

    auto out = torch::empty({C, N}, torch::TensorOptions().dtype(torch::kUInt8).device(chunk_table.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sparse_grid_sample_3d_u8_kernel<<<blocks, threads>>>(
        (long long*)chunk_table.data_ptr<int64_t>(),
        cZ, cY, cX, C,
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
    m.def("sparse_grid_sample_3d_u8", &sparse_grid_sample_3d_u8,
          "Trilinear 3D grid_sample from sparse chunk cache (CUDA)");
}
