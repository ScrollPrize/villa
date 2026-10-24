"""Fused CUDA/HIP kernels for render_gpu.py: one thread per (slice, pixel), the same
fp32 fmaf sequence as ChunkSampler::sampleTrilinear, so it matches the CPU sampler bit for bit.
Built once per GPU arch with torch load_inline (cache: $TORCH_EXTENSIONS_DIR)."""
import os
import torch

CPP = r"""
#include <torch/extension.h>
void sample_cuda(torch::Tensor base, torch::Tensor dirs, torch::Tensor offs, torch::Tensor lut,
                 torch::Tensor pool, int64_t sz, int64_t sy, int64_t sx, torch::Tensor out);
void mark_cuda(torch::Tensor base, torch::Tensor dirs, torch::Tensor offs,
               int64_t sz, int64_t sy, int64_t sx, int64_t gz, int64_t gy, int64_t gx, torch::Tensor mark);
"""

CU = r"""
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#define CS 128

struct Vol { const int* lut; const uint8_t* pool; int sz, sy, sx, gy, gx; };

__device__ __forceinline__ float sampleInt(const Vol& V, int iz, int iy, int ix) {
    if ((unsigned)iz >= (unsigned)V.sz || (unsigned)iy >= (unsigned)V.sy || (unsigned)ix >= (unsigned)V.sx) return 0.f;
    int cz = iz / CS, cy = iy / CS, cx = ix / CS;
    int slot = V.lut[((long long)cz * V.gy + cy) * V.gx + cx];
    if (slot < 0) return 0.f;
    long long o = (long long)slot * (CS * CS * CS) + (long long)(iz - cz * CS) * (CS * CS) + (iy - cy * CS) * CS + (ix - cx * CS);
    return (float)V.pool[o];
}

__global__ void sample_k(const float* __restrict__ base, const float* __restrict__ dirs,
                         const float* __restrict__ offs, int S, int N, Vol V, uint8_t* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)S * N) return;
    int s = (int)(t / N), n = (int)(t % N);
    float off = offs[s];
    float vx = __fmaf_rn(dirs[3 * n + 0], off, base[3 * n + 0]);
    float vy = __fmaf_rn(dirs[3 * n + 1], off, base[3 * n + 1]);
    float vz = __fmaf_rn(dirs[3 * n + 2], off, base[3 * n + 2]);
    uint8_t val = 0;
    if (vz >= 0 && vy >= 0 && vx >= 0 && vz < (float)V.sz && vy < (float)V.sy && vx < (float)V.sx) {
        int iz = (int)vz, iy = (int)vy, ix = (int)vx;
        float c000 = sampleInt(V, iz, iy, ix),         c100 = sampleInt(V, iz + 1, iy, ix);
        float c010 = sampleInt(V, iz, iy + 1, ix),     c110 = sampleInt(V, iz + 1, iy + 1, ix);
        float c001 = sampleInt(V, iz, iy, ix + 1),     c101 = sampleInt(V, iz + 1, iy, ix + 1);
        float c011 = sampleInt(V, iz, iy + 1, ix + 1), c111 = sampleInt(V, iz + 1, iy + 1, ix + 1);
        float fz = __fsub_rn(vz, (float)iz), fy = __fsub_rn(vy, (float)iy), fx = __fsub_rn(vx, (float)ix);
        float c00 = __fmaf_rn(fx, __fsub_rn(c001, c000), c000);
        float c01 = __fmaf_rn(fx, __fsub_rn(c011, c010), c010);
        float c10 = __fmaf_rn(fx, __fsub_rn(c101, c100), c100);
        float c11 = __fmaf_rn(fx, __fsub_rn(c111, c110), c110);
        float c0 = __fmaf_rn(fy, __fsub_rn(c01, c00), c00);
        float c1 = __fmaf_rn(fy, __fsub_rn(c11, c10), c10);
        float v = __fmaf_rn(fz, __fsub_rn(c1, c0), c0);
        if (v < 0.f) v = 0.f;
        if (v > 255.f) v = 255.f;
        val = (uint8_t)v;
    }
    out[t] = val;
}

__global__ void mark_k(const float* __restrict__ base, const float* __restrict__ dirs,
                       const float* __restrict__ offs, int S, int N, int sz, int sy, int sx,
                       int gz, int gy, int gx, uint8_t* __restrict__ mark) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)S * N) return;
    int s = (int)(t / N), n = (int)(t % N);
    float off = offs[s];
    float vx = __fmaf_rn(dirs[3 * n + 0], off, base[3 * n + 0]);
    float vy = __fmaf_rn(dirs[3 * n + 1], off, base[3 * n + 1]);
    float vz = __fmaf_rn(dirs[3 * n + 2], off, base[3 * n + 2]);
    if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < (float)sz && vy < (float)sy && vx < (float)sx)) return;
    int iz = (int)vz, iy = (int)vy, ix = (int)vx;
    for (int d = 0; d < 8; d++) {
        int z = (iz + (d >> 2)) / CS, y = (iy + ((d >> 1) & 1)) / CS, x = (ix + (d & 1)) / CS;
        if (z < gz && y < gy && x < gx) mark[((long long)z * gy + y) * gx + x] = 1;
    }
}

static inline int blocks(long long n) { return (int)((n + 255) / 256); }

void sample_cuda(at::Tensor base, at::Tensor dirs, at::Tensor offs, at::Tensor lut,
                 at::Tensor pool, int64_t sz, int64_t sy, int64_t sx, at::Tensor out) {
    int S = offs.size(0), N = base.size(0);
    Vol V{lut.data_ptr<int>(), pool.data_ptr<uint8_t>(), (int)sz, (int)sy, (int)sx, (int)lut.size(1), (int)lut.size(2)};
    sample_k<<<blocks((long long)S * N), 256>>>(base.data_ptr<float>(), dirs.data_ptr<float>(), offs.data_ptr<float>(), S, N, V, out.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mark_cuda(at::Tensor base, at::Tensor dirs, at::Tensor offs,
               int64_t sz, int64_t sy, int64_t sx, int64_t gz, int64_t gy, int64_t gx, at::Tensor mark) {
    int S = offs.size(0), N = base.size(0);
    mark_k<<<blocks((long long)S * N), 256>>>(base.data_ptr<float>(), dirs.data_ptr<float>(), offs.data_ptr<float>(), S, N,
        (int)sz, (int)sy, (int)sx, (int)gz, (int)gy, (int)gx, mark.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_mod = None


def load(dev):
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        p = torch.cuda.get_device_properties(dev)
        arch = getattr(p, "gcnArchName", "").split(":")[0] if torch.version.hip else f"sm{p.major}{p.minor}"
        if torch.version.hip:
            os.environ.setdefault("PYTORCH_ROCM_ARCH", arch)
        else:
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{p.major}.{p.minor}")
        tv = (torch.version.hip or torch.version.cuda or "x").replace(".", "_")
        _mod = load_inline(name=f"rgpu_{arch}_{tv}", cpp_sources=CPP, cuda_sources=CU,
                           functions=["sample_cuda", "mark_cuda"], verbose=False,
                           extra_cuda_cflags=["-O3", "-ffp-contract=off"] if torch.version.hip else ["-O3", "--fmad=false"])
    return _mod
