// The tracers' CropCNN (models/blocks.py) on Metal 4 tensor ops, for models/mps_cnn.py
//
// Activations are channels-last, [B, D, H, W, C]. A conv's input is kept zero-padded by one voxel on every side,
// [B, D + 2, H + 2, W + 2, C], followed by `tail` zero rows of C (2 (H + 2)(W + 2) + 2 (W + 2) + 2), so every tap of
// the last sample stays inside the buffer. Producers write only the interior; the borders are zeroed once, when the
// host allocates the buffer.
//
// A stride-1 3x3x3 conv is 9 accumulating matmul2d per output tile, one per (dz, dy) tap: in the padded layout each
// tap is a constant row shift, and the three dx taps are one K = 3C view whose rows overlap (row r is padded voxels r,
// r + 1, r + 2 back to back). Output rows that land on padding are computed and dropped. A stride-2 conv tiles one
// output line at a time, with rows 2C apart. Each conv's epilogue writes its output once and the InstanceNorm partials
// (n, mean, M2) of each tile's columns, which merge Chan-style in tile order, so a second pass reproduces the first bit
// for bit. Conv biases before an InstanceNorm are left out: the norm removes them. Accumulation and statistics are
// float. The design follows inkrs's Metal conv kernels

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename T>
using mat = tensor<device T, dextents<int32_t, 2>, tensor_inline>;

// activation and weight storage: float, or half when the host prepends `#define CNN_HALF`
#ifdef CNN_HALF
typedef half real;
typedef half4 real4;
#else
typedef float real;
typedef float4 real4;
#endif

static inline float4 chan_merge(float4 a, float4 b) {
    if (b.x == 0.0f) return a;
    if (a.x == 0.0f) return b;
    float n = a.x + b.x;
    float d = b.y - a.y;
    float f = b.x / n;
    return float4(n, a.y + d * f, a.z + b.z + d * d * a.x * f, 0.0f);
}

// The partial (n, mean, M2, 0) of every column below `ncols` of one tile staged row-major in `tile` [ROWS, COLS]:
// THREADS / COLS threads per column each take a contiguous run of rows (mean, then squared deviations), and the first
// merges the runs in order into dst[col]. valid(j) says whether tile row j is a kept output. Every thread calls it
template <int ROWS, int COLS, int THREADS, typename Valid>
inline void tile_stats(threadgroup const real* tile, threadgroup float4* part, uint tid, Valid valid,
                       device float4* dst, int ncols) {
    static_assert(THREADS % COLS == 0 && ROWS % (THREADS / COLS) == 0, "tile_stats layout");
    constexpr int TPC = THREADS / COLS, SPAN = ROWS / TPC;
    int col = tid % COLS, k = tid / COLS;
    float n = 0.0f, s = 0.0f;
    for (int j = k * SPAN; j < (k + 1) * SPAN; j++) {
        if (valid(j)) {
            s += float(tile[j * COLS + col]);
            n += 1.0f;
        }
    }
    float mean = n > 0.0f ? s / n : 0.0f;
    float m2 = 0.0f;
    for (int j = k * SPAN; j < (k + 1) * SPAN; j++) {
        if (valid(j)) {
            float d = float(tile[j * COLS + col]) - mean;
            m2 += d * d;
        }
    }
    part[tid] = float4(n, mean, m2, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0 && col < ncols) {
        float4 a = part[col];
        for (int q = 1; q < TPC; q++) a = chan_merge(a, part[q * COLS + col]);
        dst[col] = a;
    }
}

// Stride-1 conv, kernel 3 (pad 1) or 1, on the padded input xp of C channels: y [B, D, H, W, O] and, unless ints[7]
// is 0, partials part [B, tiles, O]. Weights [O, KS, KS, KS, C]. ints: B, D, H, W, C, O, KS, stats. Grid (O tiles,
// B * tiles) of 128 threads
//
// Output (b, z, y, x) is padded row b P + z PLANE + y WP + x (P = (D + 2) PLANE), whose 3x3x3 taps are rows + dz
// PLANE + dy WP + dx; a 1x1 conv reads the centre row + PLANE + WP + 1. The grid tiles each sample's D PLANE rows
// separately, so every tile stays inside one sample
template <int BM, int BN>
kernel void conv_s1(device real* p_xp [[buffer(0)]], device real* p_w [[buffer(1)]], device real* p_y [[buffer(2)]],
                    device float4* p_part [[buffer(3)]], device const int* p_ints [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup real tile[BM * BN];
    threadgroup float4 part[128];
    const int B = p_ints[0], D = p_ints[1], H = p_ints[2], W = p_ints[3], C = p_ints[4], O = p_ints[5];
    const int KS = p_ints[6];
    const bool stats = p_ints[7] != 0;
    const int WP = W + 2, PLANE = (H + 2) * WP, per = (D + 2) * PLANE;
    const int xrows = B * per + 2 * PLANE + 2 * WP + 2;
    const int span = D * PLANE, tps = (span + BM - 1) / BM;
    const int b = tg.y / tps, chunk = tg.y % tps;
    const int base = b * per, end = base + span;
    const int m0 = base + chunk * BM, n0 = tg.x * BN;
    const int kc = KS * C, wrow = KS * KS * KS * C;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    // the last two rows of a K = 3C view would read past the buffer; no kept output reaches them
    mat<real> tx(p_xp, dextents<int32_t, 2>(kc, xrows - 2), array<int, 2>{1, C});
    mat<real> tw0(p_w, dextents<int32_t, 2>(kc, O), array<int, 2>{1, wrow});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    if (KS == 3) {
        for (int t = 0; t < 9; t++) {
            mat<real> tw(p_w + t * 3 * C, dextents<int32_t, 2>(3 * C, O), array<int, 2>{1, wrow});
            auto mx = tx.slice(0, m0 + (t / 3) * PLANE + (t % 3) * WP);
            auto mw = tw.slice(0, n0);
            op.run(mx, mw, acc);
        }
    } else {
        auto mx = tx.slice(0, m0 + PLANE + WP + 1);
        auto mw = tw0.slice(0, n0);
        op.run(mx, mw, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], r = m0 + idx[1];
        int rem = r - base, z = rem / PLANE, yx = rem % PLANE, y = yx / WP, x = yx % WP;
        if (r < end && y < H && x < W && col < O) {
            real v = real(acc[i]);
            p_y[((((long)b * D + z) * H + y) * W + x) * O + col] = v;
            tile[idx[1] * BN + idx[0]] = v;
        }
    }
    if (!stats) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<BM, BN, 128>(tile, part, tid,
                            [&](int j) {
                                int rem = m0 + j - base, yx = rem % PLANE;
                                return m0 + j < end && yx / WP < H && yx % WP < W;
                            },
                            p_part + ((long)b * tps + chunk) * O + n0, O - n0);
}

#define CONV_S1(BM, BN)                                                                                              \
    template [[host_name("conv_s1_" #BM "x" #BN)]] kernel void conv_s1<BM, BN>(                                     \
        device real*, device real*, device real*, device float4*, device const int*, uint2, uint);
CONV_S1(64, 64)
CONV_S1(64, 32)
CONV_S1(64, 16)

// Stride-2 conv on the padded input xp [B, D + 2, H + 2, W + 2, C] (D, H, W even): kernel 3 with pad 1, or kernel 2
// over voxels 2o, 2o + 1 (an AvgPool3d(2) and 1x1 conv folded into one, weights w / 8). y [B, D/2, H/2, W/2, O] and
// partials part [B, tiles, O]. Weights [O, KS, KS, KS, C]. ints: B, D, H, W, C, O, KS. Each tile is up to BM outputs
// of one output line (b, z, y), whose rows read 3C (2C) contiguous values 2C apart. Grid (O tiles, B * tiles) of 128
// threads
template <int BM, int BN>
kernel void conv_s2(device real* p_xp [[buffer(0)]], device real* p_w [[buffer(1)]], device real* p_y [[buffer(2)]],
                    device float4* p_part [[buffer(3)]], device const int* p_ints [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup real tile[BM * BN];
    threadgroup float4 part[128];
    const int D = p_ints[1], H = p_ints[2], W = p_ints[3], C = p_ints[4], O = p_ints[5], KS = p_ints[6];
    const int Do = D / 2, Ho = H / 2, Wo = W / 2;
    const int WP = W + 2, PLANE = (H + 2) * WP, per = (D + 2) * PLANE;
    const int lines = (Wo + BM - 1) / BM, tps = Do * Ho * lines;
    const int b = tg.y / tps, t = tg.y % tps;
    const int chunk = t % lines, oy = (t / lines) % Ho, oz = t / lines / Ho;
    const int ox0 = chunk * BM, n0 = tg.x * BN;
    const int off = KS == 3 ? 0 : 1;  // padded coordinate of the first voxel each output reads, minus 2o
    const int wrow = KS * KS * KS * C;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    mat<real> tw0(p_w, dextents<int32_t, 2>(KS * C, O), array<int, 2>{1, wrow});
    device real* line0 = p_xp + (long)b * per * C;
    mat<real> tx0(line0, dextents<int32_t, 2>(KS * C, Wo - ox0), array<int, 2>{1, 2 * C});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx0.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int dz = 0; dz < KS; dz++) {
        for (int dy = 0; dy < KS; dy++) {
            long row = (long)(2 * oz + dz + off) * PLANE + (2 * oy + dy + off) * WP + 2 * ox0 + off;
            mat<real> tx(line0 + row * C, dextents<int32_t, 2>(KS * C, Wo - ox0), array<int, 2>{1, 2 * C});
            mat<real> tw(p_w + (dz * KS + dy) * KS * C, dextents<int32_t, 2>(KS * C, O), array<int, 2>{1, wrow});
            auto mx = tx.slice(0, 0);
            auto mw = tw.slice(0, n0);
            op.run(mx, mw, acc);
        }
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], ox = ox0 + idx[1];
        if (ox < Wo && col < O) {
            real v = real(acc[i]);
            p_y[((((long)b * Do + oz) * Ho + oy) * Wo + ox) * O + col] = v;
            tile[idx[1] * BN + idx[0]] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<BM, BN, 128>(tile, part, tid, [&](int j) { return ox0 + j < Wo; },
                            p_part + ((long)b * tps + t) * O + n0, O - n0);
}

#define CONV_S2(BM, BN)                                                                                              \
    template [[host_name("conv_s2_" #BM "x" #BN)]] kernel void conv_s2<BM, BN>(                                     \
        device real*, device real*, device real*, device float4*, device const int*, uint2, uint);
CONV_S2(32, 64)
CONV_S2(32, 32)

// Stage 0's first block on the input crops x [B, 2, D, H, W] (float, channels first): its stride-2 3x3x3 conv (pad
// 1) and its skip, AvgPool3d(2) then a 1x1 conv, both 2 -> 12 channels, one thread per output voxel. Writes y and ys
// [B, D/2, H/2, W/2, 12] with partials part and parts [B, tiles, 12]. Weights w [12, 2, 27] (torch's layout) and
// ws [12, 2]. ints: B, D, H, W. Grid (tiles * 192, B), threadgroups of 192 voxels of one sample
kernel void stem(device const float* p_x [[buffer(0)]], device const real* p_w [[buffer(1)]],
                 device const real* p_ws [[buffer(2)]], device real* p_y [[buffer(3)]], device real* p_ys [[buffer(4)]],
                 device float4* p_part [[buffer(5)]], device float4* p_parts [[buffer(6)]],
                 device const int* p_ints [[buffer(7)]], uint2 gid [[thread_position_in_grid]],
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int CI = 2, CO = 12, T = 192;
    threadgroup real tile[T * CO];
    threadgroup float4 part[T];
    const int D = p_ints[1], H = p_ints[2], W = p_ints[3];
    const int Do = D / 2, Ho = H / 2, Wo = W / 2, count = Do * Ho * Wo, tiles = (count + T - 1) / T;
    const int b = gid.y, v = gid.x;
    const bool inside = v < count;
    const int ox = v % Wo, oy = (v / Wo) % Ho, oz = v / Wo / Ho;
    float acc[CO], skip[CO];
    for (int o = 0; o < CO; ++o) acc[o] = skip[o] = 0.0f;
    if (inside) {
        const long plane = (long)D * H * W;
        device const float* xb = p_x + (long)b * CI * plane;
        float pooled[CI] = {0.0f, 0.0f};
        for (int kd = 0; kd < 3; ++kd) {
            int z = 2 * oz - 1 + kd;
            if (z < 0 || z >= D) continue;
            for (int kh = 0; kh < 3; ++kh) {
                int y = 2 * oy - 1 + kh;
                if (y < 0 || y >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    int x = 2 * ox - 1 + kw;
                    if (x < 0 || x >= W) continue;
                    const long at = ((long)z * H + y) * W + x;
                    const int tap = (kd * 3 + kh) * 3 + kw;
                    const bool pool = kd > 0 && kh > 0 && kw > 0;  // voxels 2o, 2o + 1
                    for (int c = 0; c < CI; ++c) {
                        const float value = xb[c * plane + at];
                        if (pool) pooled[c] += value;
                        for (int o = 0; o < CO; ++o) acc[o] = fma(value, float(p_w[(o * CI + c) * 27 + tap]), acc[o]);
                    }
                }
            }
        }
        for (int c = 0; c < CI; ++c) {
            pooled[c] *= 0.125f;
            for (int o = 0; o < CO; ++o) skip[o] = fma(pooled[c], float(p_ws[o * CI + c]), skip[o]);
        }
        const long at = ((long)b * count + v) * CO;
        for (int o = 0; o < CO; ++o) {
            p_y[at + o] = real(acc[o]);
            p_ys[at + o] = real(skip[o]);
        }
    }
    const int first = tg.x * T;
    auto valid = [&](int j) { return first + j < count; };
    for (int o = 0; o < CO; ++o) tile[tid * CO + o] = real(acc[o]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<T, CO, T>(tile, part, tid, valid, p_part + ((long)b * tiles + tg.x) * CO, CO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int o = 0; o < CO; ++o) tile[tid * CO + o] = real(skip[o]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_stats<T, CO, T>(tile, part, tid, valid, p_parts + ((long)b * tiles + tg.x) * CO, CO);
}

// InstanceNorm statistics from conv partials [B, tiles, O]: stats [B, O] (mean, 1 / sqrt(biased var + eps)).
// ints: tiles, O. Grid (ceil(O / 16) * 256, B) of 256 threads: 16 per column, each merging a contiguous run of
// tiles, then the runs merge in order
kernel void in_finalize(device float2* p_stats [[buffer(0)]], device const float4* p_part [[buffer(1)]],
                        device const int* p_ints [[buffer(2)]], constant float& eps [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 part[256];
    const int tiles = p_ints[0], O = p_ints[1];
    const int c = tg.x * 16 + tid % 16, k = tid / 16, b = tg.y;
    const int stripe = (tiles + 15) / 16;
    const int lo = min(k * stripe, tiles), hi = min(lo + stripe, tiles);
    float4 a = 0.0f;
    if (c < O) {
        device const float4* p = p_part + (long)b * tiles * O + c;
        for (int t = lo; t < hi; t++) a = chan_merge(a, p[(long)t * O]);
    }
    part[tid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0 && c < O) {
        for (int q = 1; q < 16; q++) a = chan_merge(a, part[q * 16 + tid]);
        float var = max(a.z / a.x, 0.0f);
        p_stats[b * O + c] = float2(a.y, 1.0f / sqrt(var + eps));
    }
}

static inline float4 normed(float4 v, device const float2* s, device const float* g, device const float* beta, int c) {
    float4 mean = float4(s[c].x, s[c + 1].x, s[c + 2].x, s[c + 3].x);
    float4 rstd = float4(s[c].y, s[c + 1].y, s[c + 2].y, s[c + 3].y);
    return (v - mean) * rstd * float4(g[c], g[c + 1], g[c + 2], g[c + 3]) + float4(beta[c], beta[c + 1], beta[c + 2], beta[c + 3]);
}

// out = act(norm(y) + residual) into the interior of the padded out [B, D + 2, H + 2, W + 2, O], from a conv's y
// [B, D, H, W, O] and its stats. Residual by mode: 0 none; 1 the interior of a padded tensor shaped as out; 2 an
// unpadded [B, D, H, W, O] tensor normalized by its own stats, gamma and beta (a projection skip); 3 the 2x2x2 mean
// of a padded tensor of twice the size (an AvgPool3d(2) skip). LeakyReLU with `slope` when act is 1. O is a multiple
// of 4. ints: B, D, H, W, O, mode, act. One thread per 4 channels of a voxel
kernel void in_apply(device const real4* p_y [[buffer(0)]], device const float2* p_stats [[buffer(1)]],
                     device const float* p_gamma [[buffer(2)]], device const float* p_beta [[buffer(3)]],
                     device const real4* p_r [[buffer(4)]], device const float2* p_rstats [[buffer(5)]],
                     device const float* p_rgamma [[buffer(6)]], device const float* p_rbeta [[buffer(7)]],
                     device real4* p_out [[buffer(8)]], device const int* p_ints [[buffer(9)]],
                     constant float& slope [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    const int B = p_ints[0], D = p_ints[1], H = p_ints[2], W = p_ints[3], O = p_ints[4];
    const int mode = p_ints[5], act = p_ints[6];
    const int G = O / 4;
    if (i >= uint(B * D * H * W * G)) return;
    const int g = i % G;
    // 32-bit index math: every buffer here has under 2^31 elements, and Apple GPUs emulate 64-bit division, which
    // made this decomposition the kernel's bottleneck (~38% of read bandwidth before)
    const uint v = i / uint(G), vw = v / uint(W), vh = vw / uint(H);
    const int x = int(v - vw * uint(W)), y = int(vw - vh * uint(H)), z = int(vh % uint(D)), b = int(vh / uint(D));
    const int c = 4 * g;
    float4 val = normed(float4(p_y[i]), p_stats + b * O, p_gamma, p_beta, c);
    const long po = ((((long)b * (D + 2) + z + 1) * (H + 2) + y + 1) * (W + 2) + x + 1) * G + g;
    if (mode == 1) {
        val += float4(p_r[po]);
    } else if (mode == 2) {
        val += normed(float4(p_r[i]), p_rstats + b * O, p_rgamma, p_rbeta, c);
    } else if (mode == 3) {
        const int D2 = 2 * D, H2 = 2 * H, W2 = 2 * W;
        float4 sum = 0.0f;
        for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                    sum += float4(p_r[((((long)b * (D2 + 2) + 2 * z + dz + 1) * (H2 + 2) + 2 * y + dy + 1) * (W2 + 2) +
                                       2 * x + dx + 1) * G + g]);
        val += sum * 0.125f;
    }
    if (act) val = select(val, val * slope, val < 0.0f);
    p_out[po] = real4(val);
}

// The CropCNN's context fuse: out [B, D, H, W, O] = x + trilinear(f) + bias, x the interior of the padded
// [B, D + 2, H + 2, W + 2, O] last-stage map and f [B, D/2, H/2, W/2, O] the 1x1 fuse conv of the context map without
// its bias, upsampled as F.interpolate(mode='trilinear', align_corners=False) does (the 1x1 conv and the upsampling
// commute). ints: B, D, H, W, O. One thread per 4 channels of a voxel
kernel void upsample_add(device const real4* p_x [[buffer(0)]], device const real4* p_f [[buffer(1)]],
                         device const float* p_bias [[buffer(2)]], device float4* p_out [[buffer(3)]],
                         device const int* p_ints [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const int B = p_ints[0], D = p_ints[1], H = p_ints[2], W = p_ints[3], O = p_ints[4];
    const int G = O / 4, Dl = D / 2, Hl = H / 2, Wl = W / 2;
    if (i >= uint(B * D * H * W * G)) return;
    const int g = i % G;
    // 32-bit index math: every buffer here has under 2^31 elements, and Apple GPUs emulate 64-bit division, which
    // made this decomposition the kernel's bottleneck (~38% of read bandwidth before)
    const uint v = i / uint(G), vw = v / uint(W), vh = vw / uint(H);
    const int x = int(v - vw * uint(W)), y = int(vw - vh * uint(H)), z = int(vh % uint(D)), b = int(vh / uint(D));
    // source index (dst + 0.5) * in / out - 0.5, clamped at 0, as PyTorch's area_pixel_compute_source_index
    const float sz = max((z + 0.5f) * 0.5f - 0.5f, 0.0f), sy = max((y + 0.5f) * 0.5f - 0.5f, 0.0f),
                sx = max((x + 0.5f) * 0.5f - 0.5f, 0.0f);
    const int z0 = int(sz), y0 = int(sy), x0 = int(sx);
    const int z1 = min(z0 + 1, Dl - 1), y1 = min(y0 + 1, Hl - 1), x1 = min(x0 + 1, Wl - 1);
    const float lz = sz - z0, ly = sy - y0, lx = sx - x0;
    auto at = [&](int zz, int yy, int xx) { return float4(p_f[((((long)b * Dl + zz) * Hl + yy) * Wl + xx) * G + g]); };
    const float4 up = (1.0f - lz) * ((1.0f - ly) * ((1.0f - lx) * at(z0, y0, x0) + lx * at(z0, y0, x1)) +
                                     ly * ((1.0f - lx) * at(z0, y1, x0) + lx * at(z0, y1, x1))) +
                      lz * ((1.0f - ly) * ((1.0f - lx) * at(z1, y0, x0) + lx * at(z1, y0, x1)) +
                            ly * ((1.0f - lx) * at(z1, y1, x0) + lx * at(z1, y1, x1)));
    const long px = ((((long)b * (D + 2) + z + 1) * (H + 2) + y + 1) * (W + 2) + x + 1) * G + g;
    const int c = 4 * g;
    p_out[i] = float4(p_x[px]) + up + float4(p_bias[c], p_bias[c + 1], p_bias[c + 2], p_bias[c + 3]);
}
