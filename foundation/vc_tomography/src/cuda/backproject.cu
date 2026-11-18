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
static_assert(kTX % 8 == 0 && kTY % 4 == 0, "a warp's threads take a patch of 8 x 4 pixels");
constexpr int kMaxAngles = kThreads;      // angles staged in shared memory per pass, at most (a thread places one)
// of a block's shared memory for the staged bins. More holds more angles a pass, but fewer blocks then run at once on
// a multiprocessor: 14 KB measured fastest on an RTX 5090 (10 and 20 KB take 3 to 6 % longer)
constexpr int kSharedBytes = 14 << 10;
constexpr int kNoWindow = INT_MIN;          // an angle none of the tile's pixels reads

// The widest stretch of a row's bins the pixels of a tile read at one angle: the tile's own extent along the row,
// `guard` bins either side (what the beam's divergence and the rounding of a position move a pixel by), the
// kernel's taps.
template <int K>
int window_bins(float us, float guard) {
    return static_cast<int>(std::ceil(static_cast<double>(us) * (kSide - 1) * 1.41421357 + 2.0 * guard)) + interp_core::tap_count<K>() + 2;
}

// What a block keeps of an angle of a pass, in shared memory. The positions are counted from the tile's first pixel
// and from the first bin of the angle's window, so that they are small numbers: a pixel's position is then right to
// 1e-5 of a bin, where the same sum on the slice's own coordinates (thousands of bins) rounds it to 5e-4.
struct AngleHead {
    float a, b;   // a parallel beam: cos, sin x the bins per column; a fan: cos, sin
    float u0;     // the tile's first pixel, in bins from the bin `base` (floor of the window's lowest position)
    int first;    // the first bin staged (base, or for the kernels of taps base + the first tap), or kNoWindow
};
struct FanHead {
    float p0;     // the tile's first pixel along the beam (x sin - y cos)
    float f0;     // its column + 1/2 from col0: floor(f0 + a pixel's column from the first pixel's) is the pixel's place in the window
    int col0;     // the first column of the fan's window
    int pad;
};
constexpr int kFanBins = 48;   // columns of the fan staged per angle: a tile's extent along a row (44 at most) and its ends
// bytes of an angle in shared memory beside its window of the row: its heads, of this pass and of the next, and its
// window of the fan
template <bool FAN>
constexpr int kHeadBytes = 2 * static_cast<int>(sizeof(AngleHead)) + (FAN ? 2 * static_cast<int>(sizeof(FanHead)) + kFanBins * static_cast<int>(sizeof(float)) : 0);

// What the heads of an angle are made from, the same for every tile (angle_consts, on the host in double precision).
// Positions are 64-bit integers in units of kUnit of a bin (or of a column): the first pixel of the tile (bx, by) lies
// at bx ax + by ay + b exactly, where the same sum in single precision would round it to 5e-4 of a bin and in double
// precision would wait for the few double precision units a consumer GPU has.
constexpr double kFixed = 4294967296.0;                // units per bin
constexpr float kUnit = 2.3283064365386963e-10f;       // 1 / kFixed
struct AngleConst {
    long long ax, ay, b;     // bins: of a tile along x, along y; of the tile (0, 0)'s first pixel
    long long lo, hi;        // bins: from a tile's first pixel to the lowest and highest position of its pixels, the guard included
    long long cx, cy, c;     // a fan, columns: as ax, ay, b, the last with the lowest pixel's offset and 1/2 added
    long long dmin;          // a fan, columns: the lowest pixel's offset from the first
    float a, s;              // AngleHead::a, b
};

// the heads of an angle for the tile (bx, by), whose first pixel is (xa, ya) from the axis
template <int K, bool FAN>
__device__ __forceinline__ void angle_heads(const AngleConst& q, int bx, int by, float xa, float ya, long long reach_lo, long long reach_end,
                                            AngleHead& h, FanHead& f) {
    constexpr bool poly = K == interp_core::BSpline3;
    const long long u = bx * q.ax + by * q.ay + q.b;   // the first pixel's bin
    const long long lo = u + q.lo, hi = u + q.hi;
    h.a = q.a;
    h.b = q.s;
    if (hi < reach_lo || lo >= reach_end) {   // none of the tile's pixels reads the row
        h.u0 = 0.f;
        h.first = kNoWindow;
    } else {
        const long long base = lo >> 32;   // (floor)
        h.u0 = static_cast<float>(u - base * 4294967296LL) * kUnit;
        // (the first bin staged: the interval that holds the lowest position, or its first tap)
        h.first = static_cast<int>(base) + (poly ? 0 : interp_core::tap_lo<K>());
    }
    if (FAN) {
        // the table's column of a pixel: floor(its column + 1/2), counted from col0, one below the lowest
        const long long at = bx * q.cx + by * q.cy + q.c;
        const long long col0 = (at >> 32) - 1;
        f.p0 = __fmaf_rn(xa, q.s, -__fmul_rn(ya, q.a));
        f.f0 = static_cast<float>(at - q.dmin - col0 * 4294967296LL) * kUnit;
        f.col0 = static_cast<int>(col0);
        f.pad = 0;
    }
}

// A block computes one tile of 32 x 32 pixels, G angles at a pass. For each angle it first copies the bins of the
// angle's row that the tile's pixels read into shared memory (their positions lie between those of the tile's corner
// pixels, to `guard` bins: W bins hold them), then every pixel takes its value from there: the bins are fetched from
// the sinogram once for the tile rather than once for every pixel, and an angle at which the whole tile lies off
// the detector is skipped. The order of the sums is that of one pixel at a time.
// What is staged of a bin: the row's value there (a pixel combines the kernel's taps: combine_taps), or for the cubic
// B-spline the polynomial of the interval that starts at the bin (bspline3_polynomial: a pixel evaluates it at its
// fraction, three multiply-adds; equal to the taps' combination to rounding). Bins beyond the row's ends are staged
// as zeros (the row's taps there), so a pixel reads its window without asking whether it is on the detector.
// A pixel's position is that of the tile's first pixel (angle_heads, exact) plus the pixel's offset in the tile
// times the angle's cosine and sine. The heads of a pass are computed by a few threads while the others compute the
// pixels of the pass before it.
// FAN: a diverging beam (BackprojectParams::fan; `fan` its n_fan columns on the device): the columns of the table
// that the tile reads at the angle are staged as well (FanHead), times the bins per column.
template <int K, bool FAN>
__global__ void backproject_kernel(const float* __restrict__ sino, const AngleConst* __restrict__ consts,
                                   float* __restrict__ slice, int n_angles, int n_det, int ny, int nx,
                                   float scale, float c, int x0, int y0, bool clip,
                                   float r2, float us, const int32_t* __restrict__ tiles, int ntx, int tx0, int ty0,
                                   const float* __restrict__ fan, int n_fan, int G, int W) {
    constexpr bool poly = K == interp_core::BSpline3;
    constexpr int E = poly ? 4 : 1;     // floats staged per bin
    // (2, G) AngleHead; a fan: (2, G) FanHead, (G, kFanBins) the fan's windows; (G, W) the rows' windows
    extern __shared__ float4 s_mem[];
    AngleHead* const s_head = reinterpret_cast<AngleHead*>(s_mem);
    FanHead* const s_fhead = reinterpret_cast<FanHead*>(s_head + 2 * G);
    float* const s_fwin = reinterpret_cast<float*>(s_fhead + (FAN ? 2 * G : 0));
    float* const s_win = reinterpret_cast<float*>(s_mem + (static_cast<size_t>(G) * kHeadBytes<FAN> + sizeof(float4) - 1) / sizeof(float4));
    // the block's tile: (blockIdx.x, blockIdx.y) of the slice's tiles (BackprojectTiles), or entry blockIdx.x of the
    // tile list; its first pixel in the grid
    const int bx = tiles ? tiles[blockIdx.x] % ntx : blockIdx.x, by = tiles ? tiles[blockIdx.x] / ntx : blockIdx.y;
    const int gx = (tx0 + bx) * kSide, gy = (ty0 + by) * kSide;
    // The thread's first pixel in the tile, (px, py) of its first kTX x kTY pixels: the 32 threads of a warp take a
    // patch of 8 x 4 pixels, not a strip of them (the bins a warp reads at an angle then lie closer together).
    const int tid = threadIdx.y * kTX + threadIdx.x;
    const int px = tid / 32 % (kTX / 8) * 8 + tid % 8, py = tid / 32 / (kTX / 8) * 4 + tid % 32 / 8;
    float dx[kPix], dy[kPix], acc[kPix];   // the thread's pixels from the tile's first one; their sums
    ptrdiff_t at[kPix];                    // their places in the slice (-1: outside it)
    unsigned inside = 0;                   // bit j: pixel j is computed
#pragma unroll
    for (int j = 0; j < kPix; ++j) {
        const int ox = px + (j % kPX) * kTX, oy = py + (j / kPX) * kTY;
        const int ix = gx + ox - x0, iy = gy + oy - y0;   // in the slice
        const float x = static_cast<float>(gx + ox) - c, y = static_cast<float>(gy + oy) - c;
        const bool in_slice = ix >= 0 && ix < nx && iy >= 0 && iy < ny;
        at[j] = in_slice ? static_cast<ptrdiff_t>(iy) * nx + ix : -1;
        if (in_slice && (!clip || x * x + y * y <= r2)) inside |= 1u << j;
        dx[j] = static_cast<float>(ox);
        dy[j] = static_cast<float>(oy);
        acc[j] = 0.f;
    }
    const bool any = inside != 0;
    // (the same for every thread of the block: no pixel of the tile is computed)
    {
        __shared__ int s_any;
        if (tid == 0) s_any = 0;
        __syncthreads();
        if (any) s_any = 1;
        __syncthreads();
        if (!s_any) {
#pragma unroll
            for (int j = 0; j < kPix; ++j)
                if (at[j] >= 0) slice[at[j]] = 0.f;
            return;
        }
    }
    // the tile's first pixel from the axis; the row's reach in units
    const float xa = static_cast<float>(gx) - c, ya = static_cast<float>(gy) - c;
    const long long reach_lo = static_cast<long long>(interp_core::reach_lo<K>()) * 4294967296LL;
    const long long reach_end = (n_det + static_cast<long long>(interp_core::reach_hi<K>())) * 4294967296LL;
    const auto heads = [&](int a, int slot) {   // of angle a, into `slot` of the heads
        AngleHead h;
        FanHead f;
        angle_heads<K, FAN>(consts[a], bx, by, xa, ya, reach_lo, reach_end, h, f);
        s_head[slot] = h;
        if (FAN) s_fhead[slot] = f;
    };
    if (tid < min(G, n_angles)) heads(tid, tid);
    const float inv_w = 1.f / static_cast<float>(W);
    for (int a0 = 0, cur = 0; a0 < n_angles; a0 += G, cur ^= G) {   // (cur: where this pass's heads are, 0 or G)
        const int na = min(G, n_angles - a0);
        __syncthreads();   // (the heads of this pass are there, and the pixels of the pass before are done)
        for (int i = tid; i < na * W; i += kThreads) {   // bins outside the row are 0, as its taps there
            const int k = static_cast<int>((static_cast<float>(i) + 0.5f) * inv_w), first = s_head[cur + k].first;   // (i / W)
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
        if (FAN) {
            for (int i = tid; i < na * kFanBins; i += kThreads) {   // (the ends' values beyond the table: fan_column)
                const int k = i / kFanBins;
                if (s_head[cur + k].first == kNoWindow) continue;
                const int col = s_fhead[cur + k].col0 + (i - k * kFanBins);
                s_fwin[i] = fan[col < 0 ? 0 : (col >= n_fan ? n_fan - 1 : col)] * us;
            }
        }
        __syncthreads();
        if (tid < min(G, n_angles - a0 - G)) heads(a0 + G + tid, (cur ^ G) + tid);   // the next pass's
        if (!any) continue;
        for (int k = 0; k < na; ++k) {
            const AngleHead h = s_head[cur + k];
            if (h.first == kNoWindow) continue;
            const float* win = s_win + static_cast<ptrdiff_t>(k) * W * E;   // (bin base + i of the row is at win + i E)
            const FanHead f = FAN ? s_fhead[cur + k] : FanHead{};
            const float* fw = s_fwin + (FAN ? k * kFanBins : 0);
#pragma unroll
            for (int j = 0; j < kPix; ++j) {
                float u;   // the pixel's position in bins from `base`
                if (FAN) {
                    const float d = __fmaf_rn(dx[j], h.a, __fmul_rn(dy[j], h.b));    // its column from the first pixel's
                    const float e = __fmaf_rn(dx[j], h.b, -__fmul_rn(dy[j], h.a));   // its place along the beam
                    u = __fmaf_rn(fw[static_cast<int>(f.f0 + d)], f.p0 + e, __fmaf_rn(d, us, h.u0));
                } else {
                    u = __fmaf_rn(dx[j], h.a, __fmaf_rn(dy[j], h.b, h.u0));
                }
                const int i = static_cast<int>(u);
                const float w = u - static_cast<float>(i);
                if (poly) {
                    const float4 p = *reinterpret_cast<const float4*>(win + i * E);
                    acc[j] += __fmaf_rn(__fmaf_rn(__fmaf_rn(p.w, w, p.z), w, p.y), w, p.x);
                } else {
                    acc[j] += interp_core::combine_taps<K>(win + i, w);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kPix; ++j)
        if (at[j] >= 0) slice[at[j]] = (inside >> j & 1u) ? acc[j] * scale : 0.f;
}

// AngleConst of every angle: `b` for the grid's tile that holds the slice's first pixel, (p.x0, p.y0) of a grid centred at c
std::vector<AngleConst> angle_consts(const float* angles, int n_angles, const BackprojectParams& p, float c, float guard) {
    std::vector<AngleConst> out(n_angles);
    const BackprojectTiles tiles(1, 1, p.x0, p.y0);   // (the grid's tile of the slice's first pixel)
    const double us = p.u_scale, xf = static_cast<double>(tiles.tx0 * kSide) - c, yf = static_cast<double>(tiles.ty0 * kSide) - c;
    const auto units = [](double v) { return std::llround(v * kFixed); };
    for (int i = 0; i < n_angles; ++i) {
        const double cs = std::cos(static_cast<double>(angles[i])), sn = std::sin(static_cast<double>(angles[i]));
        AngleConst& q = out[i];
        q.a = static_cast<float>(p.fan ? cs : cs * us);
        q.s = static_cast<float>(p.fan ? sn : sn * us);
        const double at = xf * cs + yf * sn + static_cast<double>(p.rot_center);   // the first pixel's column
        if (!std::isfinite(at)) {   // (an angle that is not a number: no pixel reads its row)
            q.ax = q.ay = q.b = q.cx = q.cy = q.c = q.dmin = 0;
            q.lo = LLONG_MAX / 4;
            q.hi = LLONG_MIN / 4;
            continue;
        }
        // the offsets of a tile's pixels along the row: between those of its corners
        const double ex = (kSide - 1) * cs, ey = (kSide - 1) * sn;
        const double dmin = std::min(ex, 0.0) + std::min(ey, 0.0), dmax = std::max(ex, 0.0) + std::max(ey, 0.0);
        q.ax = units(kSide * cs * us);
        q.ay = units(kSide * sn * us);
        q.b = units(at * us);
        q.lo = units(dmin * us - guard);
        q.hi = units(dmax * us + guard);
        q.cx = units(kSide * cs);
        q.cy = units(kSide * sn);
        q.c = units(at + dmin + 0.5);
        q.dmin = units(dmin);
    }
    return out;
}
}  // namespace

void backproject(const float* sino, const float* angles_host, float* slice, int n_angles,
                 int n_det, int ny, int nx, const BackprojectParams& p, void* stream) {
    const float c = p.center >= 0.f ? p.center : 0.5f * static_cast<float>(std::max(nx, ny) - 1);
    const float r = p.clip_radius >= 0.f ? p.clip_radius : c;
    cudaStream_t s = as_stream(stream);
    // (kept between calls: one slice after another on the calling thread, each finished before it returns)
    thread_local DeviceBuffer<AngleConst> d_cs;
    thread_local DeviceBuffer<int32_t> d_tiles;
    thread_local DeviceBuffer<float> d_fan;
    thread_local ThreadScratch scratch(d_cs, d_tiles, d_fan);
    if (p.fan && p.n_fan < 1) throw Error("backproject: a fan without columns");
    if (p.fan) {
        d_fan.ensure(static_cast<size_t>(p.n_fan));
        VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_fan.data(), p.fan, p.n_fan * sizeof(float), cudaMemcpyHostToDevice, s));
    }
    const BackprojectTiles all(ny, nx, p.x0, p.y0);
    const int ntx = all.ntx, nty = all.nty;
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
    // fan's term (the largest fan x the farthest pixel of the slice's tiles from the axis), and the rounding of the position where it is
    // not computed operation by operation as the tile's corners are.
    double far = 0, fan_max = 0;
    for (int q = 0; q < 4; ++q)
        far = std::max(far, std::hypot(static_cast<double>((all.tx0 + (q & 1 ? ntx : 0)) * kSide) - c, static_cast<double>((all.ty0 + (q & 2 ? nty : 0)) * kSide) - c));
    for (int i = 0; i < (p.fan ? p.n_fan : 0); ++i) fan_max = std::max(fan_max, std::fabs(static_cast<double>(p.fan[i])));
    const double reach = (far + std::fabs(static_cast<double>(p.rot_center)) + fan_max * far) * p.u_scale;
    const float guard = static_cast<float>(fan_max * far * p.u_scale + 16 * 1.2e-7 * reach + 0.01);
    if (!(p.u_scale > 0.f) || !std::isfinite(guard)) throw Error("backproject: a row scale that is not positive, or a fan that is not finite");
    const std::vector<AngleConst> consts = angle_consts(angles_host, n_angles, p, c, guard);
    d_cs.ensure(consts.size());
    VC_TOMOGRAPHY_CUDA_CHECK(cudaMemcpyAsync(d_cs.data(), consts.data(), consts.size() * sizeof(AngleConst), cudaMemcpyHostToDevice, s));
#define VC_TOMOGRAPHY_BP_OF(K, FAN) \
    do { \
        const int W = window_bins<K>(p.u_scale, guard); \
        const int per_angle = kHeadBytes<FAN> + W * (K == interp_core::BSpline3 ? 4 : 1) * static_cast<int>(sizeof(float)); \
        const int G = std::min(kMaxAngles, (kSharedBytes - static_cast<int>(sizeof(float4))) / per_angle); \
        if (G < 1) throw Error("backproject: the beam's divergence moves a pixel by more bins of a row than a tile can stage"); \
        const size_t shared = (static_cast<size_t>(G) * kHeadBytes<FAN> + sizeof(float4) - 1) / sizeof(float4) * sizeof(float4) + \
                              static_cast<size_t>(G) * (per_angle - kHeadBytes<FAN>); \
        backproject_kernel<K, FAN><<<grid, block, shared, s>>>(sino, d_cs.data(), slice, n_angles, n_det, ny, nx, \
                                           p.scale, c, p.x0, p.y0, p.clip_outer_circle, r * r, p.u_scale, \
                                           tiles, ntx, all.tx0, all.ty0, p.fan ? d_fan.data() : nullptr, p.n_fan, G, W); \
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
