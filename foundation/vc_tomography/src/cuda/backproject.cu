#include <climits>
#include <cmath>
#include <vector>

#include "common/interp_core.hpp"
#include "cuda_utils.cuh"
#include "vc_tomography/kernels.hpp"

namespace vc_tomography::cuda {

namespace {
constexpr int kSide = kBackprojectTile;   // a block computes one tile of kSide x kSide pixels
constexpr int kTX = 16, kTY = 16;         // its threads, each with the pixels kTX / kTY apart on the two axes
constexpr int kPX = kSide / kTX, kPY = kSide / kTY, kPix = kPX * kPY, kThreads = kTX * kTY;
static_assert(kPX * kTX == kSide && kPY * kTY == kSide, "the threads' pixels fill the tile");
constexpr int kMaxAngles = kThreads;      // angles staged in shared memory per pass, at most (a thread places one)
// of a block's shared memory for the staged bins. More holds more angles a pass, but fewer blocks then run at once on
// a multiprocessor: 14 KB measured fastest on an RTX 5090 (8 to 24 KB within 4 %, 44 KB a third slower)
constexpr int kSharedBytes = 14 << 10;
constexpr int kNoWindow = INT_MIN;          // an angle none of the tile's pixels reads

// The widest stretch of a row's bins the pixels of a tile read at one angle: the tile's own extent along the row,
// `guard` bins either side (what the beam's divergence and the rounding of a position move a pixel by), the
// kernel's taps.
template <int K>
int window_bins(float us, float guard) {
    return static_cast<int>(std::ceil(static_cast<double>(us) * (kSide - 1) * 1.41421357 + 2.0 * guard)) + interp_core::tap_count<K>() + 2;
}

// A block computes one tile of 32 x 32 pixels, G angles at a pass. For each angle it first copies the bins of the
// angle's row that the tile's pixels read into shared memory (their positions lie between those of the tile's corner
// pixels, to `guard` bins: W bins hold them), then every pixel takes its value from there: the bins are fetched from
// the sinogram once for the tile rather than once for every pixel, and an angle at which the whole tile lies off
// the detector is skipped. The order of the sums is that of one pixel at a time.
// What is staged of a bin: the row's value there (a pixel combines the kernel's taps: combine_taps), or for the cubic
// B-spline the polynomial of the interval that starts at the bin (bspline3_polynomial: a pixel evaluates it at its
// fraction, three multiply-adds; equal to the taps' combination to rounding).
// FAN: a diverging beam (BackprojectParams::fan; `fan` its n_fan columns on the device)
template <int K, bool FAN>
__global__ void backproject_kernel(const float* __restrict__ sino, const float2* __restrict__ cs_sn,
                                   float* __restrict__ slice, int n_angles, int n_det, int ny, int nx,
                                   float rot_center, float scale, float c, int x0, int y0, bool clip,
                                   float r2, float us, const int32_t* __restrict__ tiles, int ntx,
                                   const float* __restrict__ fan, int n_fan, int G, int W, float guard) {
    constexpr bool poly = K == interp_core::BSpline3;
    constexpr int E = poly ? 4 : 1;     // floats staged per bin
    extern __shared__ float4 s_mem[];   // (G) cos, sin; (G) first bin of the window; (G, W) the windows (staged_bytes)
    float2* const s_cs = reinterpret_cast<float2*>(s_mem);
    int* const s_first = reinterpret_cast<int*>(s_cs + G);
    float* const s_win = reinterpret_cast<float*>(s_mem + (static_cast<size_t>(G) * (sizeof(float2) + sizeof(int)) + sizeof(float4) - 1) / sizeof(float4));
    // the block's tile: (blockIdx.x, blockIdx.y), or entry blockIdx.x of the tile list
    const int bx = tiles ? tiles[blockIdx.x] % ntx : blockIdx.x, by = tiles ? tiles[blockIdx.x] / ntx : blockIdx.y;
    const int q0 = bx * kSide + threadIdx.x, r0 = by * kSide + threadIdx.y;
    float x[kPix], y[kPix], acc[kPix], umax[kPix];
    bool any = false;
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int ix = q0 + (j % kPX) * kTX, iy = r0 + (j / kPX) * kTY;
        x[j] = static_cast<float>(ix + x0) - c;
        y[j] = static_cast<float>(iy + y0) - c;
        const bool inside = ix < nx && iy < ny && (!clip || x[j] * x[j] + y[j] * y[j] <= r2);
        any |= inside;
        acc[j] = 0.f;
        // detector positions (bins) within the kernel's reach: [reach_lo, umax)
        umax[j] = inside ? static_cast<float>(n_det) + interp_core::reach_hi<K>() : -1e30f;
    }
    const int tid = threadIdx.y * kTX + threadIdx.x;
    // (the same for every thread of the block: no pixel of the tile is computed)
    {
        __shared__ int s_any;
        if (tid == 0) s_any = 0;
        __syncthreads();
        if (any) s_any = 1;
        __syncthreads();
        if (!s_any) {
#pragma unroll
            for (int j = 0; j < kPix; ++j) {
                const int ix = q0 + (j % kPX) * kTX, iy = r0 + (j / kPX) * kTY;
                if (ix < nx && iy < ny) slice[static_cast<size_t>(iy) * nx + ix] = acc[j] * scale;
            }
            return;
        }
    }
    // the tile's corner pixels
    const float xa = static_cast<float>(bx * kSide + x0) - c, xb = static_cast<float>(bx * kSide + kSide - 1 + x0) - c;
    const float ya = static_cast<float>(by * kSide + y0) - c, yb = static_cast<float>(by * kSide + kSide - 1 + y0) - c;
    const float reach_end = static_cast<float>(n_det) + interp_core::reach_hi<K>();
    for (int a0 = 0; a0 < n_angles; a0 += G) {
        const int na = min(G, n_angles - a0);
        __syncthreads();
        if (tid < na) {   // the angle's window: from the lowest position of the tile's pixels
            const float2 t = cs_sn[a0 + tid];
            s_cs[tid] = t;
            float lo = 1e30f, hi = -1e30f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float u = __fmul_rn(__fadd_rn(__fmaf_rn(q & 1 ? xb : xa, t.x, __fmul_rn(q & 2 ? yb : ya, t.y)), rot_center), us);
                lo = fminf(lo, u);
                hi = fmaxf(hi, u);
            }
            lo -= guard;
            hi += guard;
            // (the first bin staged: the first tap of the lowest position, or the interval that holds it)
            s_first[tid] = hi < interp_core::reach_lo<K>() || lo >= reach_end ? kNoWindow
                                                                              : static_cast<int>(floorf(lo)) + (poly ? 0 : interp_core::tap_lo<K>());
        }
        __syncthreads();
        for (int i = tid; i < na * W; i += kThreads) {   // bins outside the row are 0, as its taps there
            const int k = i / W, first = s_first[k];
            if (first == kNoWindow) continue;
            const int b = first + (i - k * W);
            const float* row = sino + static_cast<size_t>(a0 + k) * n_det;
            if (poly) {
                if (b >= 1 && b + 2 < n_det) interp_core::bspline3_polynomial(row[b - 1], row[b], row[b + 1], row[b + 2], s_win + static_cast<size_t>(i) * E);
                else interp_core::bspline3_polynomial(interp_core::tap(row, n_det, b - 1), interp_core::tap(row, n_det, b), interp_core::tap(row, n_det, b + 1),
                                                      interp_core::tap(row, n_det, b + 2), s_win + static_cast<size_t>(i) * E);
            } else {
                s_win[i] = interp_core::tap(row, n_det, b);
            }
        }
        __syncthreads();
        if (!any) continue;
        for (int k = 0; k < na; ++k) {
            const int first = s_first[k];
            if (first == kNoWindow) continue;
            const float2 t = s_cs[k];
            const float* win = s_win + (static_cast<ptrdiff_t>(k) * W - first) * E;   // (bin b of the row is at win + b E)
#pragma unroll
            for (int j = 0; j < kPix; ++j) {
                float u;
                if (FAN) {
                    const float at = x[j] * t.x + y[j] * t.y + rot_center;
                    u = (at + fan[interp_core::fan_column(at, n_fan)] * (x[j] * t.y - y[j] * t.x)) * us;
                } else {
                    // (the contraction nvcc applies to (x cos + y sin + rot) us, spelled out)
                    u = __fmul_rn(__fadd_rn(__fmaf_rn(x[j], t.x, __fmul_rn(y[j], t.y)), rot_center), us);
                }
                if (u >= interp_core::reach_lo<K>() && u < umax[j]) {
                    const float fu = floorf(u), w = u - fu;
                    if (poly) {
                        const float4 a = *reinterpret_cast<const float4*>(win + static_cast<int>(fu) * E);
                        acc[j] += __fmaf_rn(__fmaf_rn(__fmaf_rn(a.w, w, a.z), w, a.y), w, a.x);
                    } else {
                        acc[j] += interp_core::combine_taps<K>(win + static_cast<int>(fu) + interp_core::tap_lo<K>(), w);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int ix = q0 + (j % kPX) * kTX, iy = r0 + (j / kPX) * kTY;
        if (ix < nx && iy < ny) slice[static_cast<size_t>(iy) * nx + ix] = acc[j] * scale;
    }
}
}  // namespace

void backproject(const float* sino, const float* angles_host, float* slice, int n_angles,
                 int n_det, int ny, int nx, const BackprojectParams& p, void* stream) {
    const float c = p.center >= 0.f ? p.center : 0.5f * static_cast<float>(std::max(nx, ny) - 1);
    const float r = p.clip_radius >= 0.f ? p.clip_radius : c;
    std::vector<float2> cs(n_angles);
    for (int a = 0; a < n_angles; ++a)
        cs[a] = make_float2(std::cos(angles_host[a]), std::sin(angles_host[a]));
    cudaStream_t s = as_stream(stream);
    // (kept between calls: one slice after another on the calling thread, each finished before it returns)
    thread_local DeviceBuffer<float2> d_cs;
    thread_local DeviceBuffer<int32_t> d_tiles;
    thread_local DeviceBuffer<float> d_fan;
    thread_local ThreadScratch scratch(d_cs, d_tiles, d_fan);
    if (p.fan && p.n_fan < 1) throw Error("backproject: a fan without columns");
    if (p.fan) {
        d_fan.ensure(static_cast<size_t>(p.n_fan));
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_fan.data(), p.fan, p.n_fan * sizeof(float), cudaMemcpyHostToDevice, s));
    }
    d_cs.ensure(cs.size());
    VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_cs.data(), cs.data(), cs.size() * sizeof(float2), cudaMemcpyHostToDevice, s));
    const int ntx = (nx + kSide - 1) / kSide, nty = (ny + kSide - 1) / kSide;
    dim3 block(kTX, kTY), grid(ntx, nty);
    const int32_t* tiles = nullptr;
    if (p.tiles) {   // the listed tiles alone; the others are 0
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemsetAsync(slice, 0, static_cast<size_t>(ny) * nx * sizeof(float), s));
        if (p.n_tiles <= 0) {
            VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamSynchronize(s));
            return;
        }
        d_tiles.ensure(static_cast<size_t>(p.n_tiles));
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_tiles.data(), p.tiles, p.n_tiles * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        tiles = d_tiles.data();
        grid = dim3(p.n_tiles, 1);
    }
    // How far a pixel's position may lie from where the parallel beam's exact arithmetic puts it, in bins: the
    // fan's term (the largest fan x the farthest pixel from the axis), and the rounding of the position where it is
    // not computed operation by operation as the tile's corners are.
    double far = 0, fan_max = 0;
    for (int q = 0; q < 4; ++q)
        far = std::max(far, std::hypot(static_cast<double>((q & 1 ? nx - 1 : 0) + p.x0) - c, static_cast<double>((q & 2 ? ny - 1 : 0) + p.y0) - c));
    for (int i = 0; i < (p.fan ? p.n_fan : 0); ++i) fan_max = std::max(fan_max, std::fabs(static_cast<double>(p.fan[i])));
    const double reach = (far + std::fabs(static_cast<double>(p.rot_center)) + fan_max * far) * p.u_scale;
    const float guard = static_cast<float>(fan_max * far * p.u_scale + 16 * 1.2e-7 * reach + 0.01);
    if (!(p.u_scale > 0.f) || !std::isfinite(guard)) throw Error("backproject: a row scale that is not positive, or a fan that is not finite");
#define VC_TOMOGRAPHY_BP_OF(K, FAN) \
    do { \
        const int W = window_bins<K>(p.u_scale, guard); \
        const int per_bin = (K == interp_core::BSpline3 ? 4 : 1) * static_cast<int>(sizeof(float)); \
        const int G = std::min(kMaxAngles, (kSharedBytes - static_cast<int>(sizeof(float4))) / (W * per_bin + static_cast<int>(sizeof(float2) + sizeof(int)))); \
        if (G < 1) throw Error("backproject: the beam's divergence moves a pixel by more bins of a row than a tile can stage"); \
        const size_t shared = (static_cast<size_t>(G) * (sizeof(float2) + sizeof(int)) + sizeof(float4) - 1) / sizeof(float4) * sizeof(float4) + \
                              static_cast<size_t>(G) * W * per_bin; \
        backproject_kernel<K, FAN><<<grid, block, shared, s>>>(sino, d_cs.data(), slice, n_angles, n_det, ny, nx, \
                                           p.rot_center, p.scale, c, p.x0, p.y0, p.clip_outer_circle, r * r, p.u_scale, \
                                           tiles, ntx, p.fan ? d_fan.data() : nullptr, p.n_fan, G, W, guard); \
    } while (0)
#define VC_TOMOGRAPHY_BP(K) \
    do { \
        if (p.fan) VC_TOMOGRAPHY_BP_OF(K, true); \
        else VC_TOMOGRAPHY_BP_OF(K, false); \
    } while (0)
    interp_core::dispatch_interp(p.interp, "backproject", [&](auto k) { VC_TOMOGRAPHY_BP(decltype(k)::value); });
#undef VC_TOMOGRAPHY_BP
#undef VC_TOMOGRAPHY_BP_OF
    VC_TOMOGRAPHY_CUDA_CHECK(cudaGetLastError());
    VC_TOMOGRAPHY_CUDA_CHECK(cudaStreamSynchronize(s));  // the slice is complete on return
}

}  // namespace vc_tomography::cuda
