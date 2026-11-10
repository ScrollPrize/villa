// CUDA backend of the slice methods (src/common/tomo_algorithms.hpp) + gridrec on cuFFT.
#include <cufft.h>

#include <climits>
#include <cmath>
#include <complex>
#include <map>
#include <memory>
#include <vector>

#include "common/gridrec_tables.hpp"
#include "common/tomo_algorithms.hpp"
#include "common/tomo_core.hpp"
#include "cuda_utils.cuh"
#include "helirec/tomo.hpp"
#include "helirec/sino.hpp"

namespace helirec::cuda {

namespace {
using tomo_core::AngleC;
using tomo_core::Geom;
constexpr int kTile = 16, kAngles = 256;
unsigned grid_for(std::size_t n) { return static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 65535u * 16u)); }

// Chord model: one thread per ray (neighbouring threads: neighbouring rays of one angle)
template <class E>
__global__ void project_kernel(const float* __restrict__ x, const float* __restrict__ xT, const int32_t* sel, int n_sel,
                               const AngleC* __restrict__ ak, Geom g, E epi) {
    const std::size_t total = static_cast<std::size_t>(n_sel) * g.n_det;
    for (std::size_t q = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; q < total;
         q += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int a = sel[q / g.n_det], d = static_cast<int>(q % g.n_det);
        const AngleC k = ak[a];
        const float t0 = static_cast<float>(d) - g.rot;
        int s0, s1;
        tomo_core::ray_span(g, k, t0, s0, s1);
        // the warp walks the union of its rays' spans in step (same major index -> neighbouring addresses)
        const unsigned mask = __activemask();
        const int mlo = __reduce_min_sync(mask, s1 > s0 ? s0 : INT_MAX), mhi = __reduce_max_sync(mask, s1 > s0 ? s1 : INT_MIN);
        const float v = s1 > s0 ? tomo_core::project_ray_steps<true, false>(x, xT, g, k, t0, s0, s1, mlo, mhi, 1, nullptr) : 0.f;
        epi(static_cast<std::size_t>(a) * g.n_det + d, v);
    }
}

// a warp per ray (few rays: one thread per ray would be latency bound)
template <class E>
__global__ void project_warp_kernel(const float* __restrict__ x, const float* __restrict__ xT, const int32_t* sel, int n_sel,
                                    const AngleC* __restrict__ ak, Geom g, E epi) {
    const std::size_t total = static_cast<std::size_t>(n_sel) * g.n_det;
    const int lane = threadIdx.x & 31;
    const std::size_t w0 = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 5;
    const std::size_t nw = (static_cast<std::size_t>(gridDim.x) * blockDim.x) >> 5;
    for (std::size_t q = w0; q < total; q += nw) {
        const int a = sel[q / g.n_det], d = static_cast<int>(q % g.n_det);
        const AngleC k = ak[a];
        float v = tomo_core::project_ray<true, false>(x, xT, g, k, d, nullptr, lane, 32);
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) epi(static_cast<std::size_t>(a) * g.n_det + d, v);
    }
}

template <int L2>
__global__ void len2_kernel(const AngleC* __restrict__ ak, int n_angles, Geom g, float* l2) {
    const std::size_t total = static_cast<std::size_t>(n_angles) * g.n_det;
    for (std::size_t q = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; q < total;
         q += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        tomo_core::project_ray<false, L2>(nullptr, nullptr, g, ak[q / g.n_det], static_cast<int>(q % g.n_det), &l2[q]);
}

// SF model (tomo_core.hpp): one thread per cell boundary. A warp walks 32 consecutive boundaries of one angle in
// step and writes the 31 cells between them (warps overlap by one boundary). The warps of a block take the same
// boundaries at kSfGroup consecutive angles of sel: their walks read nearly the same image lines (L1). wpa: warps per
// angle.
constexpr int kSfCells = 31, kSfGroup = 8;
template <class E>
__global__ void sf_project_kernel(tomo_core::SfImage im, const int32_t* sel, int n_sel, const AngleC* __restrict__ ak,
                                  const tomo_core::SfAngle* __restrict__ sfa, Geom g, int wpa, E epi) {
    const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;   // warp of the block: the angle within the group
    const std::size_t total = static_cast<std::size_t>((n_sel + kSfGroup - 1) / kSfGroup) * wpa;   // blocks of work
    for (std::size_t q = blockIdx.x; q < total; q += gridDim.x) {
        const int ia = static_cast<int>(q / wpa) * kSfGroup + wb;
        if (ia >= n_sel) continue;
        const int a = sel[ia], beta = static_cast<int>(q % wpa) * kSfCells + lane;
        const AngleC k = ak[a];
        int s0 = 0, s1 = 0;
        if (beta <= g.n_det) tomo_core::ray_span(g, k, static_cast<float>(beta) - 0.5f - g.rot, s0, s1);
        const int mlo = __reduce_min_sync(0xffffffffu, s1 > s0 ? s0 : INT_MAX);
        const int mhi = __reduce_max_sync(0xffffffffu, s1 > s0 ? s1 : INT_MIN);
        float A = 0.f, Eb = 0.f;
        tomo_core::sf_boundary_steps(im, g, k, tomo_core::sf_walk(g, k, sfa[a], beta), s0, s1, mlo, mhi, A, Eb);
        const float An = __shfl_down_sync(0xffffffffu, A, 1);
        if (lane < kSfCells && beta < g.n_det) epi(static_cast<std::size_t>(a) * g.n_det + beta, An + Eb);
    }
}

// the box image -> the SF walks' copies (SfImage): zero outside the disk; the padding stays zero
__global__ void sf_image_kernel(const float* __restrict__ x, float* rows, float* cols, Geom g) {
    __shared__ float t[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32, P = tomo_core::kSfPad;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int ix = bx + threadIdx.x, iy = by + j;
        if (ix < g.W && iy < g.H) {
            const float v = tomo_core::sf_image_value(x, g, ix, iy);
            t[j][threadIdx.x] = v;
            rows[static_cast<std::size_t>(iy) * (g.W + 2 * P) + P + ix] = v;
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int iy = by + threadIdx.x, ix = bx + j;
        if (ix < g.W && iy < g.H) cols[static_cast<std::size_t>(ix) * (g.H + 2 * P) + P + iy] = t[threadIdx.x][j];
    }
}

// box pixels x the angles of sel (angle constants staged in shared memory); a block takes a tile of 2 kTile x 2 kTile
// pixels, four per thread (kTile apart on each axis)
constexpr int kPix = 4;
template <int PJ>
__global__ void backproject_kernel(const float* __restrict__ y, float* x, const int32_t* sel, int n_sel,
                                   const AngleC* __restrict__ ak, Geom g, bool accumulate) {
    __shared__ AngleC s_k[kAngles];
    __shared__ int s_a[kAngles];
    const int q0 = blockIdx.x * kTile * 2 + threadIdx.x, r0 = blockIdx.y * kTile * 2 + threadIdx.y;
    bool in[kPix];
    float X[kPix], Y[kPix], s[kPix], inner[kPix], reach[kPix];
    bool any = false;
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int q = q0 + (j & 1) * kTile, r = r0 + (j >> 1) * kTile;
        X[j] = g.ox + q - g.c;
        Y[j] = g.oy + r - g.c;
        in[j] = q < g.W && r < g.H && (!g.clip || tomo_core::in_disk(X[j], Y[j], g));
        any |= in[j];
        s[j] = 0.f;
        // SF: the pixel's detector positions with all four cells on the detector [1, inner), or some (-2, reach)
        inner[j] = in[j] ? static_cast<float>(g.n_det - 2) : -1e30f;
        reach[j] = in[j] ? static_cast<float>(g.n_det + 1) : -1e30f;
    }
    const int tid = threadIdx.y * kTile + threadIdx.x;
    for (int k0 = 0; k0 < n_sel; k0 += kAngles) {
        const int nk = min(kAngles, n_sel - k0);
        __syncthreads();
        for (int t = tid; t < nk; t += kTile * kTile) {
            const int a = sel[k0 + t];
            s_a[t] = a;
            s_k[t] = ak[a];
        }
        __syncthreads();
        if (any)
            for (int t = 0; t < nk; ++t) {
                const AngleC k = s_k[t];
                const float* yr = y + static_cast<std::size_t>(s_a[t]) * g.n_det;
#pragma unroll
                for (int j = 0; j < kPix; ++j) {
                    if (PJ == tomo_core::SF) {
                        const float u0 = X[j] * k.ca + Y[j] * k.sa + g.rot;
                        if (u0 >= 1.f && u0 < inner[j]) s[j] = tomo_core::sf_backproject_inner(yr, u0, k, s[j]);
                        else if (u0 < reach[j]) s[j] = tomo_core::sf_backproject_edge(yr, g.n_det, u0, k, s[j]);
                    } else if (in[j]) {
                        s[j] += tomo_core::backproject_pixel<PJ>(yr, g, X[j], Y[j], k);
                    }
                }
            }
    }
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int q = q0 + (j & 1) * kTile, r = r0 + (j >> 1) * kTile;
        if (q < g.W && r < g.H) {
            const std::size_t i = static_cast<std::size_t>(r) * g.W + q;
            x[i] = accumulate ? x[i] + s[j] : s[j];
        }
    }
}

__global__ void transpose_kernel(const float* __restrict__ in, float* out, int W, int H) {  // out[ix][iy] = in[iy][ix]
    __shared__ float t[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int ix = bx + threadIdx.x, iy = by + j;
        if (ix < W && iy < H) t[j][threadIdx.x] = in[static_cast<std::size_t>(iy) * W + ix];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int iy = by + threadIdx.x, ix = bx + j;
        if (ix < W && iy < H) out[static_cast<std::size_t>(ix) * H + iy] = t[threadIdx.x][j];
    }
}
void transpose(const float* in, float* out, int W, int H) {
    transpose_kernel<<<dim3((W + 31) / 32, (H + 31) / 32), dim3(32, 8)>>>(in, out, W, H);
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

template <class F>
__global__ void each_kernel(std::size_t n, F f) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        f(i);
}

struct FillF {
    float* d; float v;
    HR_HD void operator()(std::size_t i) const { d[i] = v; }
};
struct WeightF {
    const float *r, *w; float* o;
    HR_HD void operator()(std::size_t i) const { o[i] = r[i] * w[i]; }
};

// ---------------------------------------------------------------- gridrec
__global__ void pad_complex_kernel(const float* rows, cufftComplex* out, int n_rows, int nd, int pdim) {
    const std::size_t total = static_cast<std::size_t>(n_rows) * pdim;
    for (std::size_t k = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>(k % pdim);
        out[k] = make_cuFloatComplex(j < nd ? rows[(k / pdim) * nd + j] : 0.f, 0.f);
    }
}

// convolution of each filtered line onto the Cartesian grid H (and its conjugate-symmetric mirror)
__global__ void spread_kernel(const cufftComplex* sino, int n_rows, int pdim, const float2* filphase, const float2* cs,
                              int p0, const float* wtbl, float L2, float tblspcg, float w0, float* H) {
    const int pd2 = pdim / 2;
    const std::size_t total = static_cast<std::size_t>(n_rows) * (pd2 - 1);
    for (std::size_t q = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; q < total;
         q += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(q / (pd2 - 1)), j = 1 + static_cast<int>(q % (pd2 - 1));
        const cufftComplex* s = sino + static_cast<std::size_t>(r) * pdim;
        const float2 t = cs[p0 + r];
        float2 f = filphase[j];
        if (w0 != 0.f) {  // even grids: the half-pixel image shift as a phase (see the CPU gridrec)
            float sn, cn;
            sincosf(w0 * (t.x + t.y) * j, &sn, &cn);
            f = make_float2(f.x * cn - f.y * sn, f.x * sn + f.y * cn);
        }
        const cufftComplex a = s[j], b = s[pdim - j];
        const float c1r = f.x * a.x - f.y * a.y, c1i = f.x * a.y + f.y * a.x;
        const float c2r = f.x * b.x + f.y * b.y, c2i = f.x * b.y - f.y * b.x;  // conj(f) * b
        const float U = j * t.x + pd2, V = j * t.y + pd2;
        const int iul = max(static_cast<int>(ceilf(U - L2)), 1), iuh = min(static_cast<int>(floorf(U + L2)), pdim - 1);
        const int ivl = max(static_cast<int>(ceilf(V - L2)), 1), ivh = min(static_cast<int>(floorf(V + L2)), pdim - 1);
        for (int iu = iul; iu <= iuh; ++iu) {
            const float wu = wtbl[static_cast<int>(roundf(fabsf(U - iu) * tblspcg))];
            for (int iv = ivl; iv <= ivh; ++iv) {
                const float w = wu * wtbl[static_cast<int>(roundf(fabsf(V - iv) * tblspcg))];
                float* h1 = H + 2 * (static_cast<std::size_t>(iu) * pdim + iv);
                float* h2 = H + 2 * (static_cast<std::size_t>(pdim - iu) * pdim + (pdim - iv));
                atomicAdd(h1, w * c1r);
                atomicAdd(h1 + 1, w * c1i);
                atomicAdd(h2, w * c2r);
                atomicAdd(h2 + 1, w * c2i);
            }
        }
    }
}

__global__ void extract_kernel(const cufftComplex* H, const float* winv, int pdim, int N, float* out) {
    const int o = N / 2, linv = pdim / 2 - 1;  // FFT origin (even N: + the half-pixel phase; see the CPU gridrec)
    const std::size_t total = static_cast<std::size_t>(N) * N;
    for (std::size_t k = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; k < total;
         k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int ix = static_cast<int>(k % N), iy = static_cast<int>(k / N);
        const int iu = ((ix - o) % pdim + pdim) % pdim, iv = ((iy - o) % pdim + pdim) % pdim;
        out[k] = winv[linv + ix - o] * winv[linv + iy - o] * H[static_cast<std::size_t>(iu) * pdim + iv].x;
    }
}

void cufft_check(cufftResult r, const char* w) {
    if (r != CUFFT_SUCCESS) throw Error(std::string("cuFFT error in gridrec ") + w);
}

struct CudaBackend final : TomoWorkspace {
    Geom g, gfull;
    TomoGeom key;
    int n_angles = 0;
    std::vector<float> angles;
    DeviceBuffer<AngleC> ak;
    DeviceBuffer<tomo_core::SfAngle> sfa;
    std::map<int, DeviceBuffer<float>> bufs;
    std::vector<std::unique_ptr<DeviceBuffer<int32_t>>> keep;
    DeviceBuffer<float> l2, xT, sf_rows, sf_cols, ones;
    int have_l2 = -1;    // projector the cached ray norms belong to
    int pj = tomo_core::Chord;
    int norm = 1;        // SIRT ray norm: 1 |a_i|^2, 2 row sum
    // gridrec state
    int gr_pdim = 0, gr_chunk = 0;
    DeviceBuffer<float> gr_wtbl, gr_winv;
    DeviceBuffer<float2> gr_fp;
    TomoArgs gr_args;
    bool gr_have_fp = false;
    DeviceBuffer<cufftComplex> gr_H, gr_rows;
    std::map<int, cufftHandle> gr_plans;
    cufftHandle gr_plan2 = 0;

    CudaBackend(const TomoGeom& t, const float* ang) : key(t), n_angles(t.n_angles), angles(ang, ang + t.n_angles) {
        gfull.N = t.N; gfull.n_det = t.n_det; gfull.c = 0.5f * (t.N - 1); gfull.rot = t.rot; gfull.R = t.R;
        gfull = tomo_core::box_geom(gfull, 0, t.N, 0, t.N);
        g = gfull;
        std::vector<AngleC> h(n_angles);
        for (int i = 0; i < n_angles; ++i) h[i] = tomo_core::angle_constants(std::cos(ang[i]), std::sin(ang[i]));
        ak.upload(h.data(), h.size());
        std::vector<tomo_core::SfAngle> hs(n_angles);
        for (int i = 0; i < n_angles; ++i) hs[i] = tomo_core::sf_angle_constants(gfull, h[i]);
        sfa.upload(hs.data(), hs.size());
    }
    ~CudaBackend() override {
        for (auto& [n, p] : gr_plans) cufftDestroy(p);
        if (gr_plan2) cufftDestroy(gr_plan2);
    }
    bool matches(const TomoGeom& t, const float* ang) const {
        return t.n_angles == key.n_angles && t.n_det == key.n_det && t.N == key.N && t.rot == key.rot && t.R == key.R &&
               std::equal(angles.begin(), angles.end(), ang);
    }
    float* buf(int slot, std::size_t n) {
        auto& v = bufs[slot];
        v.ensure(n);
        return v.data();
    }
    const int32_t* ints(const std::vector<int32_t>& v) {
        keep.push_back(std::make_unique<DeviceBuffer<int32_t>>());
        keep.back()->upload(v.data(), v.size());
        return keep.back()->data();
    }
    template <class E>
    void project(const float* x, const int32_t* sel, int n_sel, const E& epi) {
        if (pj == tomo_core::SF) {
            const std::size_t nr = static_cast<std::size_t>(g.H) * (g.W + 2 * tomo_core::kSfPad);
            const std::size_t nc = static_cast<std::size_t>(g.W) * (g.H + 2 * tomo_core::kSfPad);
            if (sf_rows.size() != nr || sf_cols.size() != nc) {
                sf_rows.resize(nr);
                sf_cols.resize(nc);
                HELIREC_CUDA_CHECK(cudaMemsetAsync(sf_rows.data(), 0, nr * sizeof(float)));
                HELIREC_CUDA_CHECK(cudaMemsetAsync(sf_cols.data(), 0, nc * sizeof(float)));
            }
            sf_image_kernel<<<dim3((g.W + 31) / 32, (g.H + 31) / 32), dim3(32, 8)>>>(x, sf_rows.data(), sf_cols.data(), g);
            const int wpa = g.n_det / kSfCells + 1;   // boundaries 0 .. n_det
            const std::size_t work = static_cast<std::size_t>((n_sel + kSfGroup - 1) / kSfGroup) * wpa;
            sf_project_kernel<<<static_cast<unsigned>(std::min<std::size_t>(work, 1u << 20)), 32 * kSfGroup>>>(
                tomo_core::SfImage{sf_rows.data(), sf_cols.data()}, sel, n_sel, ak.data(), sfa.data(), g, wpa, epi);
            HELIREC_CUDA_CHECK(cudaGetLastError());
            return;
        }
        const float* xt = nullptr;
        if (n_sel >= 16) {  // columns-walking rays read the transposed box (coalesced)
            xT.ensure(static_cast<std::size_t>(g.W) * g.H);
            transpose(x, xT.data(), g.W, g.H);
            xt = xT.data();
        }
        const std::size_t rays = static_cast<std::size_t>(n_sel) * g.n_det;
        if (rays < (1u << 16)) project_warp_kernel<<<grid_for(rays * 32), 256>>>(x, xt, sel, n_sel, ak.data(), g, epi);
        else project_kernel<<<grid_for(rays), 256>>>(x, xt, sel, n_sel, ak.data(), g, epi);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    // ray norms over the disk; SF row sums: the projection of the all-ones image
    const float* len2() {
        if (have_l2 != pj * 4 + norm) {
            const std::size_t rays = static_cast<std::size_t>(n_angles) * g.n_det;
            l2.ensure(rays);
            if (pj == tomo_core::SF) {
                if (norm != 2) throw Error("sf projector: only row-sum ray norms");
                ones.resize(static_cast<std::size_t>(g.W) * g.H);
                fill(ones.data(), 1.f, ones.size());
                std::vector<int32_t> all(n_angles);
                for (int i = 0; i < n_angles; ++i) all[i] = i;
                project(ones.data(), ints(all), n_angles, tomo_algos::StoreEpi{l2.data()});
                HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
                ones.reset();
            } else if (norm == 2) {
                len2_kernel<2><<<grid_for(rays), 256>>>(ak.data(), n_angles, gfull, l2.data());
            } else {
                len2_kernel<1><<<grid_for(rays), 256>>>(ak.data(), n_angles, gfull, l2.data());
            }
            HELIREC_CUDA_CHECK(cudaGetLastError());
            have_l2 = pj * 4 + norm;
        }
        return l2.data();
    }
    void backproject(const float* y, float* x, const int32_t* sel, int n_sel, bool accumulate) {
        dim3 block(kTile, kTile), grid((g.W + 2 * kTile - 1) / (2 * kTile), (g.H + 2 * kTile - 1) / (2 * kTile));
        if (pj == tomo_core::SF) backproject_kernel<tomo_core::SF><<<grid, block>>>(y, x, sel, n_sel, ak.data(), g, accumulate);
        else backproject_kernel<tomo_core::Chord><<<grid, block>>>(y, x, sel, n_sel, ak.data(), g, accumulate);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    template <class F>
    void each(std::size_t n, const F& f) {
        each_kernel<<<grid_for(n), 256>>>(n, f);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    void fill(float* d, float v, std::size_t n) { each(n, FillF{d, v}); }
    void weighted_rows(const float* rows, const float* w, float* out, std::size_t n) { each(n, WeightF{rows, w, out}); }
    void initial_rows(const float* rows, const float* weights, float* out, float* iw, std::size_t n) {
        each(n, tomo_algos::InitialCopyF{rows, weights, out, iw});
        cuda::rebalance(out, iw, n_angles, 1, g.n_det, g.rot, false);
        weighted_rows(out, iw, out, n);
    }

    void gridrec(const float* wrows, float* out, const TomoArgs& a) {
        using namespace tomo_gridrec;
        const int NG = n_angles, ND = g.n_det, N = g.N;
        const int pdim = pdim_for(ND, N), pd2 = pdim / 2;
        if (gr_pdim != pdim) {
            std::vector<float> wtbl, winv;
            pswf_tables(pdim, wtbl, winv);
            gr_wtbl.upload(wtbl.data(), wtbl.size());
            gr_winv.upload(winv.data(), winv.size());
            gr_H.resize(static_cast<std::size_t>(pdim) * pdim);
            gr_chunk = std::max(1, std::min(NG, static_cast<int>((512u << 20) / (static_cast<std::size_t>(pdim) * 8))));
            gr_rows.resize(static_cast<std::size_t>(gr_chunk) * pdim);
            if (gr_plan2) cufftDestroy(gr_plan2);
            cufft_check(cufftPlan2d(&gr_plan2, pdim, pdim, CUFFT_C2C), "plan 2d");
            gr_pdim = pdim;
            gr_have_fp = false;
        }
        if (!gr_have_fp || gr_args.filter_name != a.filter_name || gr_args.filter_par != a.filter_par) {
            const auto fp = filter_phase(pdim, NG, g.rot, a, span_factor(angles.data(), NG));
            std::vector<float2> fph(fp.size());
            for (std::size_t i = 0; i < fp.size(); ++i) fph[i] = make_float2(fp[i].real(), fp[i].imag());
            gr_fp.upload(fph.data(), fph.size());
            gr_args = a;
            gr_have_fp = true;
        }
        const std::size_t H_n = static_cast<std::size_t>(pdim) * pdim;
        HELIREC_CUDA_CHECK(cudaMemsetAsync(gr_H.data(), 0, H_n * sizeof(cufftComplex)));
        const float L2 = static_cast<float>(static_cast<int>(kC / M_PI));
        const int L = kernel_cells();
        const float tblspcg = 2.f * kLtbl / L;
        for (int p0 = 0; p0 < NG; p0 += gr_chunk) {
            const int n = std::min(gr_chunk, NG - p0);
            auto it = gr_plans.find(n);
            if (it == gr_plans.end()) {
                cufftHandle h;
                int dims[1] = {pdim};
                cufft_check(cufftPlanMany(&h, 1, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, n), "plan rows");
                it = gr_plans.emplace(n, h).first;
            }
            pad_complex_kernel<<<grid_for(static_cast<std::size_t>(n) * pdim), 256>>>(wrows + static_cast<std::size_t>(p0) * ND,
                                                                                    gr_rows.data(), n, ND, pdim);
            cufft_check(cufftExecC2C(it->second, gr_rows.data(), gr_rows.data(), CUFFT_INVERSE), "rows");
            spread_kernel<<<grid_for(static_cast<std::size_t>(n) * (pd2 - 1)), 256>>>(
                gr_rows.data(), n, pdim, gr_fp.data(), cs_of_angles(), p0, gr_wtbl.data(), L2, tblspcg,
                -2.f * static_cast<float>(M_PI) * grid_shift(N) / pdim, reinterpret_cast<float*>(gr_H.data()));
            HELIREC_CUDA_CHECK(cudaGetLastError());
        }
        cufft_check(cufftExecC2C(gr_plan2, gr_H.data(), gr_H.data(), CUFFT_FORWARD), "2d");
        extract_kernel<<<grid_for(static_cast<std::size_t>(N) * N), 256>>>(gr_H.data(), gr_winv.data(), pdim, N, out);
        HELIREC_CUDA_CHECK(cudaGetLastError());
    }
    DeviceBuffer<float2> cs;
    const float2* cs_of_angles() {
        if (!cs.size()) {
            std::vector<float2> t(n_angles);
            for (int i = 0; i < n_angles; ++i) t[i] = make_float2(std::cos(angles[i]), std::sin(angles[i]));
            cs.upload(t.data(), t.size());
        }
        return cs.data();
    }
};

CudaBackend& backend_for(const TomoGeom& t, const float* angles, std::unique_ptr<TomoWorkspace>* ws,
                         std::unique_ptr<TomoWorkspace>& local) {
    std::unique_ptr<TomoWorkspace>& slot = ws ? *ws : local;
    auto* b = dynamic_cast<CudaBackend*>(slot.get());
    if (!b || !b->matches(t, angles)) {
        slot.reset();
        slot = std::make_unique<CudaBackend>(t, angles);
        b = static_cast<CudaBackend*>(slot.get());
    }
    b->keep.clear();
    return *b;
}

TomoGeom whole(int n_angles, int n_det, int N, float rot) {
    TomoGeom t;
    t.n_angles = n_angles; t.n_det = n_det; t.N = N; t.rot = rot;
    return t;
}
}  // namespace

void tomo_reconstruct(const TomoArgs& a, const TomoGeom& t, const float* rows, const float* weights,
                      const float* angles_host, float* out, std::unique_ptr<TomoWorkspace>* ws) {
    std::unique_ptr<TomoWorkspace> local;
    tomo_algos::run(backend_for(t, angles_host, ws, local), a, rows, weights, out);
    HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
}

void chord_project(const float* img, float* sino, const float* angles_host, int n_angles, int n_det, int N, float rot, int projector) {
    std::unique_ptr<TomoWorkspace> local;
    auto& b = backend_for(whole(n_angles, n_det, N, rot), angles_host, nullptr, local);
    b.pj = projector;
    std::vector<int32_t> all(n_angles);
    for (int i = 0; i < n_angles; ++i) all[i] = i;
    b.project(img, b.ints(all), n_angles, tomo_algos::StoreEpi{sino});
    HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
}

void chord_backproject(const float* sino, float* img, const float* angles_host, int n_angles, int n_det, int N, float rot, int projector) {
    std::unique_ptr<TomoWorkspace> local;
    auto& b = backend_for(whole(n_angles, n_det, N, rot), angles_host, nullptr, local);
    b.pj = projector;
    std::vector<int32_t> all(n_angles);
    for (int i = 0; i < n_angles; ++i) all[i] = i;
    b.backproject(sino, img, b.ints(all), n_angles, false);
    HELIREC_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace helirec::cuda
