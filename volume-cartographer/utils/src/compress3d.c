#include "utils/compress3d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __aarch64__
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

static inline float hsum_ps_sse(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#ifdef __AVX2__
static inline float hsum_ps_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    return hsum_ps_sse(lo);
}
#endif
#endif

/* Convert float array (centered at 0) to uint8 with +128 bias, clamp, and rounding */
static void float_to_u8_biased(const float *vol, uint8_t *output, int count) {
    int i = 0;
#ifdef __aarch64__
    float32x4_t bias = vdupq_n_f32(128.5f);
    float32x4_t lo = vdupq_n_f32(0.0f);
    float32x4_t hi = vdupq_n_f32(255.0f);
    for (; i + 15 < count; i += 16) {
        float32x4_t a0 = vminq_f32(vmaxq_f32(vaddq_f32(vld1q_f32(vol+i),    bias), lo), hi);
        float32x4_t a1 = vminq_f32(vmaxq_f32(vaddq_f32(vld1q_f32(vol+i+4),  bias), lo), hi);
        float32x4_t a2 = vminq_f32(vmaxq_f32(vaddq_f32(vld1q_f32(vol+i+8),  bias), lo), hi);
        float32x4_t a3 = vminq_f32(vmaxq_f32(vaddq_f32(vld1q_f32(vol+i+12), bias), lo), hi);
        uint32x4_t u0 = vcvtq_u32_f32(a0);
        uint32x4_t u1 = vcvtq_u32_f32(a1);
        uint32x4_t u2 = vcvtq_u32_f32(a2);
        uint32x4_t u3 = vcvtq_u32_f32(a3);
        uint16x4_t h0 = vmovn_u32(u0);
        uint16x4_t h1 = vmovn_u32(u1);
        uint16x4_t h2 = vmovn_u32(u2);
        uint16x4_t h3 = vmovn_u32(u3);
        uint16x8_t p0 = vcombine_u16(h0, h1);
        uint16x8_t p1 = vcombine_u16(h2, h3);
        uint8x8_t b0 = vmovn_u16(p0);
        uint8x8_t b1 = vmovn_u16(p1);
        vst1q_u8(output + i, vcombine_u8(b0, b1));
    }
#elif defined(__x86_64__) || defined(_M_X64)
    __m128 bias = _mm_set1_ps(128.5f);
    __m128 lo = _mm_setzero_ps();
    __m128 hi = _mm_set1_ps(255.0f);
    for (; i + 7 < count; i += 8) {
        __m128 a0 = _mm_min_ps(_mm_max_ps(_mm_add_ps(_mm_loadu_ps(vol+i),   bias), lo), hi);
        __m128 a1 = _mm_min_ps(_mm_max_ps(_mm_add_ps(_mm_loadu_ps(vol+i+4), bias), lo), hi);
        __m128i i0 = _mm_cvtps_epi32(a0);
        __m128i i1 = _mm_cvtps_epi32(a1);
        __m128i packed = _mm_packs_epi32(i0, i1);
        __m128i bytes = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64((__m128i *)(output + i), bytes);
    }
#endif
    for (; i < count; i++) {
        float v = vol[i] + 128.0f;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        output[i] = (uint8_t)roundf(v);
    }
}

#define N C3D_BLOCK_SIZE
#define N3 C3D_BLOCK_VOXELS

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 1: Fast DCT via precomputed matrix
 * ════════════════════════════════════════════════════════════════════════════ */

static uint16_t zigzag_order[N3];
static float dct_matrix[N][N];
static float idct_matrix[N][N];
static float freq_scale_table[N3];
static float freq_dist_table[N3];
static pthread_once_t tables_once = PTHREAD_ONCE_INIT;

/* Quality step lookup table - initialized in do_init_tables */
static float quality_step_table[102];

static void init_quality_step_table(void) {
    for (int q = 0; q <= 101; q++) {
        int cq = q < 1 ? 1 : (q > 100 ? 100 : q);
        quality_step_table[q] = 50.0f * powf(0.01f, (cq - 1) / 99.0f);
    }
}

/* VLQ lookup table - initialized in do_init_tables */
typedef struct { uint8_t bytes[3]; uint8_t len; } vlq_entry_t;
static vlq_entry_t vlq_table[512];
static int vlq_table_inited = 0;

typedef struct { uint16_t idx; uint32_t dist; } zentry;

static int zentry_cmp(const void *a, const void *b) {
    uint32_t da = ((const zentry *)a)->dist;
    uint32_t db = ((const zentry *)b)->dist;
    return (da > db) - (da < db);
}

static void do_init_tables(void) {
    for (int k = 0; k < N; k++) {
        float alpha = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        for (int n = 0; n < N; n++) {
            float val = alpha * cosf((float)M_PI * (2*n + 1) * k / (2*N));
            dct_matrix[k][n] = val;
            idct_matrix[n][k] = val;
        }
    }

    zentry *entries = (zentry *)malloc(N3 * sizeof(zentry));
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int i = z * N * N + y * N + x;
                entries[i].idx = (uint16_t)i;
                entries[i].dist = x*x + y*y + z*z;
            }
    qsort(entries, N3, sizeof(zentry), zentry_cmp);
    for (int i = 0; i < N3; i++)
        zigzag_order[i] = entries[i].idx;
    free(entries);

    /* Pre-compute powf(fd/53.7, 1.5) lookup indexed by integer squared distance.
     * Max squared distance is 31^2+31^2+31^2 = 2883. */
    #define MAX_SQ_DIST (3 * (N-1) * (N-1))
    float pow_lut[MAX_SQ_DIST + 1];
    for (int sd = 0; sd <= MAX_SQ_DIST; sd++) {
        float fd = sqrtf((float)sd);
        pow_lut[sd] = powf(fd / 53.7f, 1.5f);
    }

    /* Precompute frequency-dependent scale and distance tables.
     * These only depend on (x,y,z) position and never change. */
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int idx = z * N * N + y * N + x;
                int sd = x*x + y*y + z*z;
                float fd = sqrtf((float)sd);
                freq_scale_table[idx] = (fd < 0.01f) ? 0.5f : 1.0f + 3.0f * pow_lut[sd];
                freq_dist_table[idx] = fd / 53.7f;
            }
    #undef MAX_SQ_DIST

    /* Initialize quality step lookup table */
    init_quality_step_table();

    /* Initialize VLQ lookup table */
    for (int f = 0; f < 512; f++) {
        uint32_t v = (uint32_t)f;
        int n = 0;
        if (v <= 127) {
            vlq_table[f].bytes[0] = (uint8_t)v;
            vlq_table[f].len = 1;
        } else {
            vlq_table[f].bytes[n++] = 0x80 | (uint8_t)(v & 0x7F);
            v >>= 7;
            while (v >= 128) {
                vlq_table[f].bytes[n++] = 0x80 | (uint8_t)(v & 0x7F);
                v >>= 7;
            }
            vlq_table[f].bytes[n++] = (uint8_t)v;
            vlq_table[f].len = (uint8_t)n;
        }
    }
    vlq_table_inited = 1;
}

static void init_tables(void) {
    pthread_once(&tables_once, do_init_tables);
}

/* Transpose helpers for cache-friendly axis processing. */

#ifdef __aarch64__

static void transpose_xy(const float * restrict src, float * restrict dst) {
    for (int z = 0; z < N; z++) {
        const float *s = src + z * N * N;
        float *d = dst + z * N * N;
        for (int by = 0; by < N; by += 4) {
            for (int bx = 0; bx < N; bx += 4) {
                float32x4_t r0 = vld1q_f32(s + (by + 0) * N + bx);
                float32x4_t r1 = vld1q_f32(s + (by + 1) * N + bx);
                float32x4_t r2 = vld1q_f32(s + (by + 2) * N + bx);
                float32x4_t r3 = vld1q_f32(s + (by + 3) * N + bx);

                float32x4_t t0 = vtrn1q_f32(r0, r1);
                float32x4_t t1 = vtrn2q_f32(r0, r1);
                float32x4_t t2 = vtrn1q_f32(r2, r3);
                float32x4_t t3 = vtrn2q_f32(r2, r3);

                float64x2_t u0 = vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
                float64x2_t u1 = vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));
                float64x2_t u2 = vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
                float64x2_t u3 = vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));

                vst1q_f32(d + (bx + 0) * N + by, vreinterpretq_f32_f64(u0));
                vst1q_f32(d + (bx + 1) * N + by, vreinterpretq_f32_f64(u1));
                vst1q_f32(d + (bx + 2) * N + by, vreinterpretq_f32_f64(u2));
                vst1q_f32(d + (bx + 3) * N + by, vreinterpretq_f32_f64(u3));
            }
        }
    }
}

static void transpose_xz(const float * restrict src, float * restrict dst) {
    for (int y = 0; y < N; y++) {
        for (int bz = 0; bz < N; bz += 4) {
            for (int bx = 0; bx < N; bx += 4) {
                float32x4_t r0 = vld1q_f32(src + (bz + 0) * N * N + y * N + bx);
                float32x4_t r1 = vld1q_f32(src + (bz + 1) * N * N + y * N + bx);
                float32x4_t r2 = vld1q_f32(src + (bz + 2) * N * N + y * N + bx);
                float32x4_t r3 = vld1q_f32(src + (bz + 3) * N * N + y * N + bx);

                float32x4_t t0 = vtrn1q_f32(r0, r1);
                float32x4_t t1 = vtrn2q_f32(r0, r1);
                float32x4_t t2 = vtrn1q_f32(r2, r3);
                float32x4_t t3 = vtrn2q_f32(r2, r3);

                float64x2_t u0 = vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
                float64x2_t u1 = vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));
                float64x2_t u2 = vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
                float64x2_t u3 = vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));

                vst1q_f32(dst + (bx + 0) * N * N + y * N + bz, vreinterpretq_f32_f64(u0));
                vst1q_f32(dst + (bx + 1) * N * N + y * N + bz, vreinterpretq_f32_f64(u1));
                vst1q_f32(dst + (bx + 2) * N * N + y * N + bz, vreinterpretq_f32_f64(u2));
                vst1q_f32(dst + (bx + 3) * N * N + y * N + bz, vreinterpretq_f32_f64(u3));
            }
        }
    }
}

#elif defined(__x86_64__) || defined(_M_X64)

static void transpose_xy(const float * restrict src, float * restrict dst) {
    for (int z = 0; z < N; z++) {
        const float *s = src + z * N * N;
        float *d = dst + z * N * N;
        for (int by = 0; by < N; by += 4) {
            for (int bx = 0; bx < N; bx += 4) {
                __m128 r0 = _mm_loadu_ps(s + (by + 0) * N + bx);
                __m128 r1 = _mm_loadu_ps(s + (by + 1) * N + bx);
                __m128 r2 = _mm_loadu_ps(s + (by + 2) * N + bx);
                __m128 r3 = _mm_loadu_ps(s + (by + 3) * N + bx);
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                _mm_storeu_ps(d + (bx + 0) * N + by, r0);
                _mm_storeu_ps(d + (bx + 1) * N + by, r1);
                _mm_storeu_ps(d + (bx + 2) * N + by, r2);
                _mm_storeu_ps(d + (bx + 3) * N + by, r3);
            }
        }
    }
}

static void transpose_xz(const float * restrict src, float * restrict dst) {
    for (int y = 0; y < N; y++) {
        for (int bz = 0; bz < N; bz += 4) {
            for (int bx = 0; bx < N; bx += 4) {
                __m128 r0 = _mm_loadu_ps(src + (bz + 0) * N * N + y * N + bx);
                __m128 r1 = _mm_loadu_ps(src + (bz + 1) * N * N + y * N + bx);
                __m128 r2 = _mm_loadu_ps(src + (bz + 2) * N * N + y * N + bx);
                __m128 r3 = _mm_loadu_ps(src + (bz + 3) * N * N + y * N + bx);
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                _mm_storeu_ps(dst + (bx + 0) * N * N + y * N + bz, r0);
                _mm_storeu_ps(dst + (bx + 1) * N * N + y * N + bz, r1);
                _mm_storeu_ps(dst + (bx + 2) * N * N + y * N + bz, r2);
                _mm_storeu_ps(dst + (bx + 3) * N * N + y * N + bz, r3);
            }
        }
    }
}

#else

static void transpose_xy(const float * restrict src, float * restrict dst) {
    for (int z = 0; z < N; z++) {
        const float *s = src + z * N * N;
        float *d = dst + z * N * N;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                d[x * N + y] = s[y * N + x];
    }
}

static void transpose_xz(const float * restrict src, float * restrict dst) {
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++) {
            if (y + 1 < N)
                __builtin_prefetch(src + z * N * N + (y + 1) * N, 0, 1);
            for (int x = 0; x < N; x++)
                dst[x * N * N + y * N + z] = src[z * N * N + y * N + x];
        }
}

#endif

/* Transform N contiguous rows in-place using matrix mat.
 * Register blocking: 2 output coefficients at a time share input loads. */
static void transform_rows(float * restrict vol, const float mat[N][N], int nrows) {
    float line[N];
    for (int r = 0; r < nrows; r++) {
        float *rp = vol + r * N;
        if (r + 1 < nrows)
            __builtin_prefetch(vol + (r + 1) * N, 0, 1);
        memcpy(line, rp, N * sizeof(float));
        int k = 0;
        for (; k + 1 < N; k += 2) {
            const float *m0 = mat[k], *m1 = mat[k + 1];
#ifdef __aarch64__
            float32x4_t a0 = vmulq_f32(vld1q_f32(m0+ 0), vld1q_f32(line+ 0));
            float32x4_t a1 = vmulq_f32(vld1q_f32(m0+ 4), vld1q_f32(line+ 4));
            float32x4_t b0 = vmulq_f32(vld1q_f32(m1+ 0), vld1q_f32(line+ 0));
            float32x4_t b1 = vmulq_f32(vld1q_f32(m1+ 4), vld1q_f32(line+ 4));
            for (int i = 8; i < N; i += 8) {
                float32x4_t l0 = vld1q_f32(line+i), l1 = vld1q_f32(line+i+4);
                a0 = vfmaq_f32(a0, vld1q_f32(m0+i),   l0);
                a1 = vfmaq_f32(a1, vld1q_f32(m0+i+4), l1);
                b0 = vfmaq_f32(b0, vld1q_f32(m1+i),   l0);
                b1 = vfmaq_f32(b1, vld1q_f32(m1+i+4), l1);
            }
            rp[k]   = vaddvq_f32(vaddq_f32(a0, a1));
            rp[k+1] = vaddvq_f32(vaddq_f32(b0, b1));
#elif defined(__x86_64__) || defined(_M_X64)
#ifdef __AVX2__
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(m0+ 0), _mm256_loadu_ps(line+ 0));
            __m256 b0 = _mm256_mul_ps(_mm256_loadu_ps(m1+ 0), _mm256_loadu_ps(line+ 0));
            for (int i = 8; i < N; i += 8) {
                __m256 l = _mm256_loadu_ps(line+i);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(m0+i), l, a0);
                b0 = _mm256_fmadd_ps(_mm256_loadu_ps(m1+i), l, b0);
            }
            rp[k]   = hsum_ps_avx(a0);
            rp[k+1] = hsum_ps_avx(b0);
#else
            __m128 a0 = _mm_mul_ps(_mm_loadu_ps(m0+ 0), _mm_loadu_ps(line+ 0));
            __m128 a1 = _mm_mul_ps(_mm_loadu_ps(m0+ 4), _mm_loadu_ps(line+ 4));
            __m128 b0 = _mm_mul_ps(_mm_loadu_ps(m1+ 0), _mm_loadu_ps(line+ 0));
            __m128 b1 = _mm_mul_ps(_mm_loadu_ps(m1+ 4), _mm_loadu_ps(line+ 4));
#ifdef __FMA__
            for (int i = 8; i < N; i += 8) {
                __m128 l0 = _mm_loadu_ps(line+i), l1 = _mm_loadu_ps(line+i+4);
                a0 = _mm_fmadd_ps(_mm_loadu_ps(m0+i),   l0, a0);
                a1 = _mm_fmadd_ps(_mm_loadu_ps(m0+i+4), l1, a1);
                b0 = _mm_fmadd_ps(_mm_loadu_ps(m1+i),   l0, b0);
                b1 = _mm_fmadd_ps(_mm_loadu_ps(m1+i+4), l1, b1);
            }
#else
            for (int i = 8; i < N; i += 8) {
                __m128 l0 = _mm_loadu_ps(line+i), l1 = _mm_loadu_ps(line+i+4);
                a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(m0+i),   l0));
                a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(m0+i+4), l1));
                b0 = _mm_add_ps(b0, _mm_mul_ps(_mm_loadu_ps(m1+i),   l0));
                b1 = _mm_add_ps(b1, _mm_mul_ps(_mm_loadu_ps(m1+i+4), l1));
            }
#endif
            rp[k]   = hsum_ps_sse(_mm_add_ps(a0, a1));
            rp[k+1] = hsum_ps_sse(_mm_add_ps(b0, b1));
#endif
#else
            float s0 = 0.0f, s1 = 0.0f;
            for (int n = 0; n < N; n++) { s0 += m0[n]*line[n]; s1 += m1[n]*line[n]; }
            rp[k] = s0; rp[k+1] = s1;
#endif
        }
        if (k < N) {
            const float *m0 = mat[k];
#ifdef __aarch64__
            float32x4_t a0 = vmulq_f32(vld1q_f32(m0+ 0), vld1q_f32(line+ 0));
            float32x4_t a1 = vmulq_f32(vld1q_f32(m0+ 4), vld1q_f32(line+ 4));
            for (int i = 8; i < N; i += 8) {
                a0 = vfmaq_f32(a0, vld1q_f32(m0+i),   vld1q_f32(line+i));
                a1 = vfmaq_f32(a1, vld1q_f32(m0+i+4), vld1q_f32(line+i+4));
            }
            rp[k] = vaddvq_f32(vaddq_f32(a0, a1));
#elif defined(__x86_64__) || defined(_M_X64)
#ifdef __AVX2__
            __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(m0+ 0), _mm256_loadu_ps(line+ 0));
            for (int i = 8; i < N; i += 8) {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(m0+i), _mm256_loadu_ps(line+i), a0);
            }
            rp[k] = hsum_ps_avx(a0);
#else
            __m128 a0 = _mm_mul_ps(_mm_loadu_ps(m0+ 0), _mm_loadu_ps(line+ 0));
            __m128 a1 = _mm_mul_ps(_mm_loadu_ps(m0+ 4), _mm_loadu_ps(line+ 4));
#ifdef __FMA__
            for (int i = 8; i < N; i += 8) {
                a0 = _mm_fmadd_ps(_mm_loadu_ps(m0+i),   _mm_loadu_ps(line+i),   a0);
                a1 = _mm_fmadd_ps(_mm_loadu_ps(m0+i+4), _mm_loadu_ps(line+i+4), a1);
            }
#else
            for (int i = 8; i < N; i += 8) {
                a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(m0+i),   _mm_loadu_ps(line+i)));
                a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(m0+i+4), _mm_loadu_ps(line+i+4)));
            }
#endif
            rp[k] = hsum_ps_sse(_mm_add_ps(a0, a1));
#endif
#else
            float s0 = 0.0f;
            for (int n = 0; n < N; n++) s0 += m0[n]*line[n];
            rp[k] = s0;
#endif
        }
    }
}

/* Fused 3-axis DCT: shares a single tmp buffer across all axes instead of
 * allocating a separate N3-sized stack array per axis call. */
static void dct3d_forward_all(float * restrict vol) {
    float tmp[N3];
    transform_rows(vol, dct_matrix, N * N);
    transpose_xy(vol, tmp); transform_rows(tmp, dct_matrix, N * N); transpose_xy(tmp, vol);
    transpose_xz(vol, tmp); transform_rows(tmp, dct_matrix, N * N); transpose_xz(tmp, vol);
}

static void dct3d_inverse_all(float * restrict vol) {
    float tmp[N3];
    transpose_xz(vol, tmp); transform_rows(tmp, idct_matrix, N * N); transpose_xz(tmp, vol);
    transpose_xy(vol, tmp); transform_rows(tmp, idct_matrix, N * N); transpose_xy(tmp, vol);
    transform_rows(vol, idct_matrix, N * N);
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 2: Dead-zone quantization
 * ════════════════════════════════════════════════════════════════════════════ */


static float quality_to_step(int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    return quality_step_table[quality];
}

static void compute_quant_table(float base_step, float *qtable,
                                float *dz_table, float *bias_table) {
    for (int i = 0; i < N3; i++) {
        qtable[i] = base_step * freq_scale_table[i];
        float dz = 0.3f + 0.5f * freq_dist_table[i];
        dz_table[i] = dz;
        bias_table[i] = 1.0f - dz;
    }
}

/* Dequantize-only path: only computes qtable, skipping dz/bias (not needed for decompress). */
static void compute_dequant_table(float base_step, float *qtable) {
#ifdef __aarch64__
    float32x4_t vbs = vdupq_n_f32(base_step);
    for (int i = 0; i < N3; i += 8) {
        float32x4_t a = vld1q_f32(freq_scale_table + i);
        float32x4_t b = vld1q_f32(freq_scale_table + i + 4);
        vst1q_f32(qtable + i,     vmulq_f32(vbs, a));
        vst1q_f32(qtable + i + 4, vmulq_f32(vbs, b));
    }
#elif defined(__x86_64__) || defined(_M_X64)
    __m128 vbs = _mm_set1_ps(base_step);
    for (int i = 0; i < N3; i += 8) {
        __m128 a = _mm_loadu_ps(freq_scale_table + i);
        __m128 b = _mm_loadu_ps(freq_scale_table + i + 4);
        _mm_storeu_ps(qtable + i,     _mm_mul_ps(vbs, a));
        _mm_storeu_ps(qtable + i + 4, _mm_mul_ps(vbs, b));
    }
#else
    for (int i = 0; i < N3; i++)
        qtable[i] = base_step * freq_scale_table[i];
#endif
}

static void quantize_volume(const float *coeffs, int32_t *quant,
                            const float *qtable, const float *dz_table,
                            const float *bias_table) {
#ifdef __aarch64__
    for (int i = 0; i < N3; i += 4) {
        float32x4_t c = vld1q_f32(coeffs + i);
        float32x4_t q = vld1q_f32(qtable + i);
        float32x4_t recip = vrecpeq_f32(q);
        recip = vmulq_f32(recip, vrecpsq_f32(q, recip));
        float32x4_t v = vmulq_f32(c, recip);
        float32x4_t vdz = vld1q_f32(dz_table + i);
        float32x4_t vndz = vnegq_f32(vdz);
        float32x4_t vbias = vld1q_f32(bias_table + i);
        float32x4_t vnbias = vnegq_f32(vbias);
        uint32x4_t pos_mask = vcgtq_f32(v, vdz);
        uint32x4_t neg_mask = vcltq_f32(v, vndz);
        float32x4_t pos_val = vaddq_f32(v, vbias);
        float32x4_t neg_val = vaddq_f32(v, vnbias);
        int32x4_t pos_i = vcvtq_s32_f32(pos_val);
        int32x4_t neg_i = vcvtq_s32_f32(neg_val);
        int32x4_t zero = vdupq_n_s32(0);
        int32x4_t result = vbslq_s32(pos_mask, pos_i, zero);
        result = vbslq_s32(neg_mask, neg_i, result);
        vst1q_s32(quant + i, result);
    }
#elif defined(__x86_64__) || defined(_M_X64)
    for (int i = 0; i < N3; i += 4) {
        __m128 c = _mm_loadu_ps(coeffs + i);
        __m128 q = _mm_loadu_ps(qtable + i);
        __m128 recip = _mm_rcp_ps(q);
        recip = _mm_mul_ps(recip, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(q, recip)));
        __m128 v = _mm_mul_ps(c, recip);
        __m128 vdz = _mm_loadu_ps(dz_table + i);
        __m128 vndz = _mm_sub_ps(_mm_setzero_ps(), vdz);
        __m128 vbias = _mm_loadu_ps(bias_table + i);
        __m128 vnbias = _mm_sub_ps(_mm_setzero_ps(), vbias);
        __m128 pos_mask = _mm_cmpgt_ps(v, vdz);
        __m128 neg_mask = _mm_cmplt_ps(v, vndz);
        __m128 pos_val = _mm_add_ps(v, vbias);
        __m128 neg_val = _mm_add_ps(v, vnbias);
        __m128i pos_i = _mm_cvttps_epi32(pos_val);
        __m128i neg_i = _mm_cvttps_epi32(neg_val);
        __m128i zero = _mm_setzero_si128();
        __m128i result = _mm_blendv_epi8(zero, pos_i, _mm_castps_si128(pos_mask));
        result = _mm_blendv_epi8(result, neg_i, _mm_castps_si128(neg_mask));
        _mm_storeu_si128((__m128i *)(quant + i), result);
    }
#else
    for (int i = 0; i < N3; i++) {
        float q = qtable[i];
        float v = coeffs[i] / q;
        float dz = dz_table[i];
        float bias = bias_table[i];
        if (v > dz)
            quant[i] = (int32_t)(v + bias);
        else if (v < -dz)
            quant[i] = (int32_t)(v - bias);
        else
            quant[i] = 0;
    }
#endif
}

/* Estimate the byte cost of encoding a single quantized value */
static float vlq_byte_cost(int32_t val) {
    if (val == 0) return 0.0f;
    uint32_t folded = (val > 0) ? (uint32_t)(2 * val) : (uint32_t)(-2 * val - 1);
    if (folded <= 127) return 1.0f;
    float bytes = 1.0f;
    folded >>= 7;
    while (folded >= 128) { bytes += 1.0f; folded >>= 7; }
    bytes += 1.0f;
    return bytes;
}

/*
 * Reverse-scan trellis RD optimization.
 * Scans coefficients in reverse zigzag order tracking zero-run state.
 * For each coefficient considers: keep, alternative rounding, or zero.
 * Chooses the option minimizing D + lambda * R.
 */
static void rd_optimize_coeffs(const float *coeffs, int32_t *quant,
                                const float *qtable, float lambda) {
    int trailing_zeros = 0;
    for (int zi = N3 - 1; zi >= 0; zi--) {
        if (quant[zigzag_order[zi]] == 0)
            trailing_zeros++;
        else
            break;
    }

    /* Reverse scan from the last non-zero coefficient */
    int last_nz = N3 - 1 - trailing_zeros;
    for (int zi = last_nz; zi >= 0; zi--) {
        int i = zigzag_order[zi];
        int32_t q = quant[i];
        float orig = coeffs[i];
        float qstep = qtable[i];

        /* Frequency-aware lambda: more aggressive at high frequencies.
         * freq_dist_table[i] = sqrt(x^2+y^2+z^2) / 53.7, precomputed in init_tables. */
        float freq_lambda = lambda * (1.0f + 0.5f * freq_dist_table[i]);

        /* Check if next coeff in zigzag order is zero */
        int next_is_zero = (zi >= last_nz) ? 1 :
                           (quant[zigzag_order[zi + 1]] == 0);

        /* Option A: keep current value */
        float recon_a = q * qstep;
        float dist_a = (orig - recon_a) * (orig - recon_a);
        float rate_a = vlq_byte_cost(q) * 8.0f;
        if (next_is_zero && q != 0)
            rate_a += 16.0f;
        float cost_a = dist_a + freq_lambda * rate_a;

        /* Option B: alternative rounding */
        int32_t q_alt;
        if (q == 0) {
            q_alt = (orig >= 0) ? 1 : -1;
        } else {
            float recon_up = (q + 1) * qstep;
            float recon_dn = (q - 1) * qstep;
            float d_up = (orig - recon_up) * (orig - recon_up);
            float d_dn = (orig - recon_dn) * (orig - recon_dn);
            q_alt = (d_up < d_dn) ? q + 1 : q - 1;
        }
        float recon_b = q_alt * qstep;
        float dist_b = (orig - recon_b) * (orig - recon_b);
        float rate_b;
        if (q_alt == 0) {
            rate_b = 0.0f;
        } else {
            rate_b = vlq_byte_cost(q_alt) * 8.0f;
            if (next_is_zero)
                rate_b += 16.0f;
        }
        float cost_b = dist_b + freq_lambda * rate_b;

        /* Option C: set to zero */
        float dist_c = orig * orig;
        float rate_c = 0.0f;
        float cost_c = dist_c + freq_lambda * rate_c;

        /* Choose best option and update state */
        if (cost_c <= cost_a && cost_c <= cost_b) {
            quant[i] = 0;
        } else if (cost_b < cost_a) {
            quant[i] = q_alt;
            if (q == 0) last_nz = (zi > last_nz) ? zi : last_nz;
        }
        /* else keep q */
    }
}

static void dequantize_volume(const int32_t *quant, float *coeffs, const float *qtable) {
#ifdef __aarch64__
    for (int i = 0; i < N3; i += 4) {
        int32x4_t qi = vld1q_s32(quant + i);
        float32x4_t qf = vcvtq_f32_s32(qi);
        float32x4_t qt = vld1q_f32(qtable + i);
        vst1q_f32(coeffs + i, vmulq_f32(qf, qt));
    }
#elif defined(__x86_64__) || defined(_M_X64)
    for (int i = 0; i < N3; i += 4) {
        __m128i qi = _mm_loadu_si128((const __m128i *)(quant + i));
        __m128 qf = _mm_cvtepi32_ps(qi);
        __m128 qt = _mm_loadu_ps(qtable + i);
        _mm_storeu_ps(coeffs + i, _mm_mul_ps(qf, qt));
    }
#else
    for (int i = 0; i < N3; i++)
        coeffs[i] = quant[i] * qtable[i];
#endif
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 4: rANS entropy coder
 * ════════════════════════════════════════════════════════════════════════════ */

#define RANS_PROB_BITS 14
#define RANS_PROB_SCALE (1 << RANS_PROB_BITS)
#define RANS_BYTE_L (1 << 23)

typedef struct {
    uint32_t freq[256];
    uint32_t cum_freq[257];
    uint32_t total;
    uint64_t rcp[256];
} rans_model_t;

typedef struct {
    uint16_t freq;
    uint16_t cum_freq;
    uint8_t  sym;
    uint8_t  pad[3];
} rans_decode_entry_t;

static void rans_precompute_rcp(rans_model_t *model) {
    for (int i = 0; i < 256; i++) {
        if (model->freq[i] > 0)
            model->rcp[i] = ((1ULL << 32) + model->freq[i] - 1) / model->freq[i];
        else
            model->rcp[i] = 0;
    }
}

static void rans_build_model(const uint8_t *data, int len, rans_model_t *model) {
    memset(model->freq, 0, sizeof(model->freq));
    for (int i = 0; i < len; i++)
        model->freq[data[i]]++;

    uint32_t total = 0;
    for (int i = 0; i < 256; i++)
        total += model->freq[i];

    if (total == 0) {
        model->freq[0] = RANS_PROB_SCALE;
        model->total = RANS_PROB_SCALE;
        model->cum_freq[0] = 0;
        for (int i = 0; i < 256; i++)
            model->cum_freq[i+1] = model->cum_freq[i] + ((i == 0) ? RANS_PROB_SCALE : 0);
        rans_precompute_rcp(model);
        return;
    }

    uint32_t scaled[256];
    uint32_t scaled_total = 0;
    for (int i = 0; i < 256; i++) {
        if (model->freq[i] > 0) {
            scaled[i] = (uint32_t)((uint64_t)model->freq[i] * RANS_PROB_SCALE / total);
            if (scaled[i] == 0) scaled[i] = 1;
        } else {
            scaled[i] = 0;
        }
        scaled_total += scaled[i];
    }

    int diff = (int)RANS_PROB_SCALE - (int)scaled_total;
    int max_sym = 0;
    for (int i = 1; i < 256; i++)
        if (scaled[i] > scaled[max_sym]) max_sym = i;
    scaled[max_sym] += diff;

    for (int i = 0; i < 256; i++)
        model->freq[i] = scaled[i];
    model->cum_freq[0] = 0;
    for (int i = 0; i < 256; i++)
        model->cum_freq[i+1] = model->cum_freq[i] + model->freq[i];
    model->total = RANS_PROB_SCALE;
    rans_precompute_rcp(model);
}

static void rans_build_decode_table(const rans_model_t *model, uint8_t *sym_table) {
    for (int s = 0; s < 256; s++)
        for (uint32_t j = model->cum_freq[s]; j < model->cum_freq[s+1]; j++)
            sym_table[j] = (uint8_t)s;
}

static void rans_build_decode_table_fast(const rans_model_t *model, rans_decode_entry_t *dtable) {
    for (int s = 0; s < 256; s++) {
        uint32_t f = model->freq[s];
        uint32_t cf = model->cum_freq[s];
        uint32_t end = model->cum_freq[s+1];
        if (end > RANS_PROB_SCALE) end = RANS_PROB_SCALE;
        if (cf > end) cf = end;
        for (uint32_t j = cf; j < end; j++) {
            dtable[j].sym = (uint8_t)s;
            dtable[j].freq = (uint16_t)f;
            dtable[j].cum_freq = (uint16_t)cf;
        }
    }
}

static void rans_encode_buf(const uint8_t *data, int len, const rans_model_t *model,
                            uint8_t *buf, size_t buf_cap, size_t *out_len) {
    int buf_pos = 0;
    int cap = (int)buf_cap;
    uint32_t state = RANS_BYTE_L;

    for (int i = len - 1; i >= 0; i--) {
        uint8_t s = data[i];
        uint32_t freq = model->freq[s];
        uint32_t start = model->cum_freq[s];
        uint64_t upper64 = (uint64_t)freq * ((uint64_t)(RANS_BYTE_L >> RANS_PROB_BITS) << 8);
        while (state >= upper64) {
            if (buf_pos >= cap) { *out_len = 0; return; }
            buf[buf_pos++] = (uint8_t)(state & 0xFF);
            state >>= 8;
        }
        uint64_t rcp = model->rcp[s];
        uint32_t q = (uint32_t)((uint64_t)state * rcp >> 32);
        uint32_t r = state - q * freq;
        if (r >= freq) { q--; r += freq; }
        state = q * RANS_PROB_SCALE + r + start;
    }

    if (buf_pos + 4 > cap) { *out_len = 0; return; }
    buf[buf_pos++] = (uint8_t)(state & 0xFF);
    buf[buf_pos++] = (uint8_t)((state >> 8) & 0xFF);
    buf[buf_pos++] = (uint8_t)((state >> 16) & 0xFF);
    buf[buf_pos++] = (uint8_t)((state >> 24) & 0xFF);

    for (int i = 0; i < buf_pos / 2; i++) {
        uint8_t tmp = buf[i];
        buf[i] = buf[buf_pos - 1 - i];
        buf[buf_pos - 1 - i] = tmp;
    }

    *out_len = buf_pos;
}

static void rans_encode(const uint8_t *data, int len, const rans_model_t *model,
                         uint8_t **out, size_t *out_len) {
    size_t cap = (size_t)len * 2 + 256;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) { *out = NULL; *out_len = 0; return; }
    rans_encode_buf(data, len, model, buf, cap, out_len);
    *out = buf;
}

/* Core rANS decode using a caller-provided dtable (no malloc). */
static void rans_decode_core(const uint8_t *compressed, int comp_len,
                              const rans_model_t *model,
                              rans_decode_entry_t *dtable,
                              uint8_t *output, int orig_len) {
    if (comp_len < 4 || orig_len <= 0 || !dtable) {
        if (orig_len > 0) memset(output, 0, (size_t)orig_len);
        return;
    }
    rans_build_decode_table_fast(model, dtable);

    int pos = 0;
    uint32_t state = (uint32_t)compressed[pos] << 24
                   | (uint32_t)compressed[pos+1] << 16
                   | (uint32_t)compressed[pos+2] << 8
                   | (uint32_t)compressed[pos+3];
    pos += 4;

    for (int i = 0; i < orig_len; i++) {
        uint32_t slot = state & (RANS_PROB_SCALE - 1);
        rans_decode_entry_t e = dtable[slot];
        output[i] = e.sym;
        state = (uint32_t)e.freq * (state >> RANS_PROB_BITS) + slot - (uint32_t)e.cum_freq;
        while (state < RANS_BYTE_L && pos < comp_len)
            state = (state << 8) | compressed[pos++];
        if (i + 1 < orig_len) {
            uint32_t ns = state & (RANS_PROB_SCALE - 1);
            __builtin_prefetch(&dtable[ns], 0, 3);
        }
    }
}

static void rans_decode(const uint8_t *compressed, int comp_len,
                         const rans_model_t *model, const uint8_t *sym_table,
                         uint8_t *output, int orig_len) {
    (void)sym_table;
    rans_decode_entry_t dtable[RANS_PROB_SCALE];
    rans_decode_core(compressed, comp_len, model, dtable, output, orig_len);
}

/* ── 4-stream interleaved rANS for decode ILP ── */

#define RANS_NUM_INTERLEAVE 4

static void rans_encode_interleaved(const uint8_t *symbols, int count,
    const rans_model_t *model, uint8_t *output, size_t *out_len) {
    /* Split symbols round-robin into 4 sub-streams */
    int sub_counts[RANS_NUM_INTERLEAVE];
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++)
        sub_counts[s] = (count + RANS_NUM_INTERLEAVE - 1 - s) / RANS_NUM_INTERLEAVE;

    /* Encode each sub-stream separately */
    uint8_t *sub_bufs[RANS_NUM_INTERLEAVE];
    size_t sub_lens[RANS_NUM_INTERLEAVE];
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++) {
        int sc = sub_counts[s];
        uint8_t *sub_syms = (uint8_t *)malloc(sc > 0 ? sc : 1);
        for (int i = 0; i < sc; i++)
            sub_syms[i] = symbols[i * RANS_NUM_INTERLEAVE + s];
        size_t sub_cap = (size_t)sc * 2 + 256;
        sub_bufs[s] = (uint8_t *)malloc(sub_cap);
        rans_encode_buf(sub_syms, sc, model, sub_bufs[s], sub_cap, &sub_lens[s]);
        free(sub_syms);
    }

    /* Pack: [4 x stream_len (16 bytes)] [stream0] [stream1] [stream2] [stream3] */
    size_t pos = 0;
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++) {
        uint32_t sl = (uint32_t)sub_lens[s];
        output[pos++] = (uint8_t)(sl);
        output[pos++] = (uint8_t)(sl >> 8);
        output[pos++] = (uint8_t)(sl >> 16);
        output[pos++] = (uint8_t)(sl >> 24);
    }
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++) {
        memcpy(output + pos, sub_bufs[s], sub_lens[s]);
        pos += sub_lens[s];
        free(sub_bufs[s]);
    }
    *out_len = pos;
}

/* Core interleaved decode using caller-provided dtable (no malloc). */
static void rans_decode_interleaved_core(const uint8_t *data, int data_len,
    const rans_model_t *model, rans_decode_entry_t *dtable,
    uint8_t *symbols, int count) {
    /* Parse 4 stream lengths */
    if (data_len < RANS_NUM_INTERLEAVE * 4 || count <= 0 || !dtable) return;

    uint32_t sub_lens[RANS_NUM_INTERLEAVE];
    int pos = 0;
    uint64_t total_sub = 0;
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++) {
        sub_lens[s] = (uint32_t)data[pos] | ((uint32_t)data[pos+1] << 8)
                    | ((uint32_t)data[pos+2] << 16) | ((uint32_t)data[pos+3] << 24);
        total_sub += sub_lens[s];
        pos += 4;
    }

    /* Validate sub-stream lengths don't overflow available data */
    if (total_sub > (uint64_t)(data_len - pos)) return;

    int sub_counts[RANS_NUM_INTERLEAVE];
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++)
        sub_counts[s] = (count + RANS_NUM_INTERLEAVE - 1 - s) / RANS_NUM_INTERLEAVE;

    /* Build decode table once, shared by all 4 streams */
    rans_build_decode_table_fast(model, dtable);

    /* Decode each sub-stream, then interleave results */
    int stream_off = pos;
    for (int s = 0; s < RANS_NUM_INTERLEAVE; s++) {
        int slen = (int)sub_lens[s];
        int sc = sub_counts[s];

        if (slen < 4 || sc <= 0) {
            stream_off += slen;
            continue;
        }

        const uint8_t *sdata = data + stream_off;

        /* Inline decode to reuse dtable */
        int sp = 0;
        uint32_t state = (uint32_t)sdata[sp] << 24
                       | (uint32_t)sdata[sp+1] << 16
                       | (uint32_t)sdata[sp+2] << 8
                       | (uint32_t)sdata[sp+3];
        sp += 4;

        for (int i = 0; i < sc; i++) {
            uint32_t slot = state & (RANS_PROB_SCALE - 1);
            rans_decode_entry_t e = dtable[slot];
            symbols[i * RANS_NUM_INTERLEAVE + s] = e.sym;
            state = (uint32_t)e.freq * (state >> RANS_PROB_BITS) + slot - (uint32_t)e.cum_freq;
            while (state < RANS_BYTE_L && sp < slen)
                state = (state << 8) | sdata[sp++];
        }

        stream_off += slen;
    }
}

static void rans_decode_interleaved(const uint8_t *data, int data_len,
    const rans_model_t *model,
    uint8_t *symbols, int count) {
    rans_decode_entry_t dtable[RANS_PROB_SCALE];
    rans_decode_interleaved_core(data, data_len, model, dtable, symbols, count);
}

/* ── Order-1 context modeling with 8 context groups ── */

#define NUM_CTX_GROUPS 8

static const uint8_t ctx_group_lut[256] = {
    0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4,
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
};

static inline int ctx_group(uint8_t prev_sym) {
    return ctx_group_lut[prev_sym];
}

static void rans_build_ctx_models(const uint8_t *data, int len, rans_model_t models[NUM_CTX_GROUPS]) {
    uint32_t counts[NUM_CTX_GROUPS][256];
    memset(counts, 0, sizeof(counts));

    for (int i = 0; i < len; i++) {
        int g = (i == 0) ? 7 : ctx_group(data[i-1]);
        counts[g][data[i]]++;
    }

    for (int g = 0; g < NUM_CTX_GROUPS; g++) {
        uint32_t total = 0;
        for (int i = 0; i < 256; i++)
            total += counts[g][i];

        if (total == 0) {
            for (int i = 0; i < 256; i++) {
                models[g].freq[i] = (i == 0) ? RANS_PROB_SCALE : 0;
                models[g].cum_freq[i] = (i == 0) ? 0 : RANS_PROB_SCALE;
            }
            models[g].cum_freq[256] = RANS_PROB_SCALE;
            models[g].total = RANS_PROB_SCALE;
            rans_precompute_rcp(&models[g]);
            continue;
        }

        uint32_t scaled[256];
        uint32_t scaled_total = 0;
        for (int i = 0; i < 256; i++) {
            if (counts[g][i] > 0) {
                scaled[i] = (uint32_t)((uint64_t)counts[g][i] * RANS_PROB_SCALE / total);
                if (scaled[i] == 0) scaled[i] = 1;
            } else {
                scaled[i] = 0;
            }
            scaled_total += scaled[i];
        }

        int diff = (int)RANS_PROB_SCALE - (int)scaled_total;
        int max_sym = 0;
        for (int i = 1; i < 256; i++)
            if (scaled[i] > scaled[max_sym]) max_sym = i;
        scaled[max_sym] += diff;

        for (int i = 0; i < 256; i++)
            models[g].freq[i] = scaled[i];
        models[g].cum_freq[0] = 0;
        for (int i = 0; i < 256; i++)
            models[g].cum_freq[i+1] = models[g].cum_freq[i] + models[g].freq[i];
        models[g].total = RANS_PROB_SCALE;
        rans_precompute_rcp(&models[g]);
    }
}

static void rans_encode_ctx_buf(const uint8_t *data, int len,
                                 const rans_model_t models[NUM_CTX_GROUPS],
                                 uint8_t *buf, size_t buf_cap, size_t *out_len) {
    int buf_pos = 0;
    int cap = (int)buf_cap;
    uint32_t state = RANS_BYTE_L;

    for (int i = len - 1; i >= 0; i--) {
        uint8_t s = data[i];
        int g = (i == 0) ? 7 : ctx_group(data[i-1]);
        const rans_model_t *model = &models[g];
        uint32_t freq = model->freq[s];
        uint32_t start = model->cum_freq[s];
        uint64_t upper64 = (uint64_t)freq * ((uint64_t)(RANS_BYTE_L >> RANS_PROB_BITS) << 8);
        while (state >= upper64) {
            if (buf_pos >= cap) { *out_len = 0; return; }
            buf[buf_pos++] = (uint8_t)(state & 0xFF);
            state >>= 8;
        }
        uint64_t rcp = model->rcp[s];
        uint32_t q = (uint32_t)((uint64_t)state * rcp >> 32);
        uint32_t r = state - q * freq;
        if (r >= freq) { q--; r += freq; }
        state = q * RANS_PROB_SCALE + r + start;
    }

    if (buf_pos + 4 > cap) { *out_len = 0; return; }
    buf[buf_pos++] = (uint8_t)(state & 0xFF);
    buf[buf_pos++] = (uint8_t)((state >> 8) & 0xFF);
    buf[buf_pos++] = (uint8_t)((state >> 16) & 0xFF);
    buf[buf_pos++] = (uint8_t)((state >> 24) & 0xFF);

    for (int i = 0; i < buf_pos / 2; i++) {
        uint8_t tmp = buf[i];
        buf[i] = buf[buf_pos - 1 - i];
        buf[buf_pos - 1 - i] = tmp;
    }

    *out_len = buf_pos;
}

static void rans_encode_ctx(const uint8_t *data, int len,
                             const rans_model_t models[NUM_CTX_GROUPS],
                             uint8_t **out, size_t *out_len) {
    size_t cap = (size_t)len * 2 + 256;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) { *out = NULL; *out_len = 0; return; }
    rans_encode_ctx_buf(data, len, models, buf, cap, out_len);
    *out = buf;
}

/* Core contextual decode using caller-provided dtables (no malloc).
 * dtables must have room for NUM_CTX_GROUPS * RANS_PROB_SCALE entries. */
static void rans_decode_ctx_core(const uint8_t *compressed, int comp_len,
                                  const rans_model_t models[NUM_CTX_GROUPS],
                                  rans_decode_entry_t *dtables,
                                  uint8_t *output, int orig_len) {
    if (comp_len < 4 || orig_len <= 0 || !dtables) {
        if (orig_len > 0) memset(output, 0, (size_t)orig_len);
        return;
    }
    for (int g = 0; g < NUM_CTX_GROUPS; g++)
        rans_build_decode_table_fast(&models[g], dtables + g * RANS_PROB_SCALE);

    int pos = 0;
    uint32_t state = (uint32_t)compressed[pos] << 24
                   | (uint32_t)compressed[pos+1] << 16
                   | (uint32_t)compressed[pos+2] << 8
                   | (uint32_t)compressed[pos+3];
    pos += 4;

    for (int i = 0; i < orig_len; i++) {
        int g = (i == 0) ? 7 : ctx_group(output[i-1]);
        rans_decode_entry_t *dt = dtables + g * RANS_PROB_SCALE;
        uint32_t slot = state & (RANS_PROB_SCALE - 1);
        rans_decode_entry_t e = dt[slot];
        output[i] = e.sym;
        state = (uint32_t)e.freq * (state >> RANS_PROB_BITS) + slot - (uint32_t)e.cum_freq;
        while (state < RANS_BYTE_L && pos < comp_len)
            state = (state << 8) | compressed[pos++];
    }
}

static void rans_decode_ctx(const uint8_t *compressed, int comp_len,
                              const rans_model_t models[NUM_CTX_GROUPS],
                              uint8_t sym_tables[][RANS_PROB_SCALE],
                              uint8_t *output, int orig_len) {
    (void)sym_tables;
    rans_decode_entry_t *dtables = (rans_decode_entry_t *)malloc(
        NUM_CTX_GROUPS * RANS_PROB_SCALE * sizeof(rans_decode_entry_t));
    if (!dtables) { if (orig_len > 0) memset(output, 0, (size_t)orig_len); return; }
    rans_decode_ctx_core(compressed, comp_len, models, dtables, output, orig_len);
    free(dtables);
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 5: VLQ symbol encoding with zero-run RLE
 *
 * Converts quantized coefficients (zigzag scan order) to a byte stream.
 * Much better than raw byteshuffle for sparse data (many zeros after deadzone).
 *   0x00 + len:        zero run (len+1 zeros, max 256)
 *   0x01-0x7F:         single-byte sign-folded value (fold: 1→2,-1→1,2→4,-2→3)
 *   0x80|low7 + VLQ:   multi-byte value using variable-length quantity
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t *symbols;
    int count;
    int cap;
} symstream_t;

static void sym_init(symstream_t *s) {
    s->cap = N3;
    s->symbols = (uint8_t *)malloc(s->cap);
    s->count = 0;
}

static void sym_push(symstream_t *s, uint8_t val) {
    if (s->count >= s->cap) {
        s->cap *= 2;
        s->symbols = (uint8_t *)realloc(s->symbols, s->cap);
    }
    s->symbols[s->count++] = val;
}

/* VLQ table type/data/init moved to top of file and merged into do_init_tables().
 * Keep a fallback init_vlq_table() for any code paths that call it directly. */
static void init_vlq_table(void) {
    /* Already initialized by do_init_tables via pthread_once */
    if (vlq_table_inited) return;
    for (int f = 0; f < 512; f++) {
        uint32_t v = (uint32_t)f;
        int n = 0;
        if (v <= 127) {
            vlq_table[f].bytes[0] = (uint8_t)v;
            vlq_table[f].len = 1;
        } else {
            vlq_table[f].bytes[n++] = 0x80 | (uint8_t)(v & 0x7F);
            v >>= 7;
            while (v >= 128) {
                vlq_table[f].bytes[n++] = 0x80 | (uint8_t)(v & 0x7F);
                v >>= 7;
            }
            vlq_table[f].bytes[n++] = (uint8_t)v;
            vlq_table[f].len = (uint8_t)n;
        }
    }
    vlq_table_inited = 1;
}

static void emit_vlq(symstream_t *s, uint32_t folded) {
    if (folded < 512) {
        const vlq_entry_t *e = &vlq_table[folded];
        for (int i = 0; i < e->len; i++)
            sym_push(s, e->bytes[i]);
        return;
    }
    uint32_t v = folded;
    sym_push(s, 0x80 | (uint8_t)(v & 0x7F));
    v >>= 7;
    while (v >= 128) {
        sym_push(s, 0x80 | (uint8_t)(v & 0x7F));
        v >>= 7;
    }
    sym_push(s, (uint8_t)v);
}

/* Delta-code the first COEFF_PRED_BAND coefficients in zigzag order */
#define COEFF_PRED_BAND 64

static void coeff_predict_forward(int32_t *quant) {
    for (int i = COEFF_PRED_BAND - 1; i >= 1; i--)
        quant[zigzag_order[i]] -= quant[zigzag_order[i - 1]];
}

static void coeff_predict_inverse(int32_t *quant) {
    for (int i = 1; i < COEFF_PRED_BAND; i++)
        quant[zigzag_order[i]] += quant[zigzag_order[i - 1]];
}

static void coeffs_to_symbols(const int32_t *quant, symstream_t *s) {
    init_vlq_table();
    int i = 0;
    while (i < N3) {
        if (quant[zigzag_order[i]] == 0) {
            int run = 0;
            while (i < N3 && quant[zigzag_order[i]] == 0) { run++; i++; }
            /* All zero runs use RLE escape */
            while (run > 0) {
                int r = (run > 256) ? 256 : run;
                sym_push(s, 0x00);
                sym_push(s, (uint8_t)(r - 1));
                run -= r;
            }
        } else {
            int32_t val = quant[zigzag_order[i]];
            uint32_t folded = (val > 0) ? (uint32_t)(2 * val) : (uint32_t)(-2 * val - 1);
            if (folded < 512) {
                const vlq_entry_t *e = &vlq_table[folded];
                for (int j = 0; j < e->len; j++)
                    sym_push(s, e->bytes[j]);
            } else {
                emit_vlq(s, folded);
            }
            i++;
        }
    }
}

static int symbols_to_coeffs(const uint8_t *syms, int nsyms, int32_t *quant) {
    memset(quant, 0, N3 * sizeof(int32_t));
    int si = 0, ci = 0;
    while (si < nsyms && ci < N3) {
        uint8_t b = syms[si++];
        if (b == 0x00) {
            if (si >= nsyms) return -1;
            int run = (int)syms[si++] + 1;
            for (int j = 0; j < run && ci < N3; j++)
                quant[zigzag_order[ci++]] = 0;
        } else if (b & 0x80) {
            uint32_t folded = (b & 0x7F);
            int shift = 7;
            while (si < nsyms) {
                uint8_t cb = syms[si++];
                folded |= (uint32_t)(cb & 0x7F) << shift;
                shift += 7;
                if (!(cb & 0x80)) break;
            }
            int32_t val = (int32_t)(folded >> 1) ^ -(int32_t)(folded & 1);
            quant[zigzag_order[ci++]] = val;
        } else {
            uint32_t folded = (uint32_t)b;
            int32_t val = (int32_t)(folded >> 1) ^ -(int32_t)(folded & 1);
            quant[zigzag_order[ci++]] = val;
        }
    }
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 5b: Lossless compression via 3D prediction + rANS
 * ════════════════════════════════════════════════════════════════════════════ */

#define LOSSLESS_HEADER_SIZE (4 + 4 + 4 + 4 + 512)  /* 528 bytes */
#define LOSSLESS_HEADER_CTX_SIZE (4 + 4 + 4 + 4 + NUM_CTX_GROUPS * 512)  /* 4112 bytes */
#define C3D_FLAG_CTX_LOSSLESS 0x01  /* bit 0 of byte 7: order-1 context mode */

/* Forward declarations for sparse freq table functions (defined in Section 6) */
static int sparse_freq_write(uint8_t *p, const rans_model_t *model);
static int sparse_freq_read(const uint8_t *p, size_t avail, rans_model_t *model);

static inline uint8_t loco3d_predict(const uint8_t *data, int x, int y, int z) {
    int a = (x > 0) ? data[z*N*N + y*N + (x-1)] : -1;       /* left */
    int b = (y > 0) ? data[z*N*N + (y-1)*N + x] : -1;       /* above */
    int c = (x > 0 && y > 0) ? data[z*N*N + (y-1)*N + (x-1)] : -1; /* diag */
    int d = (z > 0) ? data[(z-1)*N*N + y*N + x] : -1;       /* behind */

    int p2d;
    if (a >= 0 && b >= 0 && c >= 0) {
        /* LOCO-I / MED predictor */
        int min_ab = a < b ? a : b;
        int max_ab = a > b ? a : b;
        if (c >= max_ab)      p2d = min_ab;
        else if (c <= min_ab) p2d = max_ab;
        else                  p2d = a + b - c;
    } else if (a >= 0 && b >= 0) {
        p2d = (a + b) / 2;
    } else if (a >= 0) {
        p2d = a;
    } else if (b >= 0) {
        p2d = b;
    } else if (d >= 0) {
        return (uint8_t)d;
    } else {
        return 128;
    }

    /* Blend 2D prediction with behind-neighbor */
    if (d >= 0) {
        int p = (p2d * 2 + d) / 3;
        return (uint8_t)p;
    }
    return (uint8_t)p2d;
}

static void predict_residuals(const uint8_t *input, uint8_t *residuals) {
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int idx = z * N * N + y * N + x;
                uint8_t predict = loco3d_predict(input, x, y, z);
                residuals[idx] = input[idx] - predict;
            }
}

static void unpredict_residuals(const uint8_t *residuals, uint8_t *output) {
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int idx = z * N * N + y * N + x;
                uint8_t predict = loco3d_predict(output, x, y, z);
                output[idx] = residuals[idx] + predict;
            }
}

static size_t lossless_ctx_header_size(const rans_model_t models[NUM_CTX_GROUPS]) {
    size_t sz = 17; /* 16 byte fixed header + 1 byte num_groups */
    for (int g = 0; g < NUM_CTX_GROUPS; g++) {
        int nz = 0;
        for (int i = 0; i < 256; i++)
            if (models[g].freq[i] > 0) nz++;
        sz += 1 + nz * 3;
    }
    return sz;
}

static size_t lossless_compress_to(const uint8_t *input, uint8_t *output, size_t output_cap) {
    uint8_t *residuals = (uint8_t *)malloc(N3);
    predict_residuals(input, residuals);

    /* Order-0 encoding */
    rans_model_t model_o0;
    rans_build_model(residuals, N3, &model_o0);

    uint8_t *rans_o0;
    size_t rans_o0_len;
    rans_encode(residuals, N3, &model_o0, &rans_o0, &rans_o0_len);
    size_t total_o0 = LOSSLESS_HEADER_SIZE + rans_o0_len;

    /* Order-1 (context) encoding with sparse freq tables */
    rans_model_t models_ctx[NUM_CTX_GROUPS];
    rans_build_ctx_models(residuals, N3, models_ctx);

    uint8_t *rans_ctx;
    size_t rans_ctx_len;
    rans_encode_ctx(residuals, N3, models_ctx, &rans_ctx, &rans_ctx_len);
    size_t ctx_hdr_size = lossless_ctx_header_size(models_ctx);
    size_t total_ctx = ctx_hdr_size + rans_ctx_len;

    free(residuals);

    int use_ctx = (total_ctx < total_o0);
    size_t total = use_ctx ? total_ctx : total_o0;

    if (total > output_cap) {
        free(rans_o0); free(rans_ctx);
        return 0;
    }

    if (use_ctx) {
        /* Context mode: sparse freq tables */
        memset(output, 0, 17);
        output[0] = 'C'; output[1] = '3'; output[2] = 'D'; output[3] = 0x04;
        output[4] = 101;  /* quality = lossless */
        output[5] = 0;    /* transform (unused) */
        output[6] = 1;    /* mode: lossless */
        output[7] = C3D_FLAG_CTX_LOSSLESS;  /* context flag */

        uint32_t ol = (uint32_t)N3;
        output[8] = ol; output[9] = ol >> 8; output[10] = ol >> 16; output[11] = ol >> 24;
        uint32_t rl = (uint32_t)rans_ctx_len;
        output[12] = rl; output[13] = rl >> 8; output[14] = rl >> 16; output[15] = rl >> 24;

        output[16] = (uint8_t)NUM_CTX_GROUPS;
        size_t off = 17;
        for (int g = 0; g < NUM_CTX_GROUPS; g++)
            off += sparse_freq_write(output + off, &models_ctx[g]);

        memcpy(output + off, rans_ctx, rans_ctx_len);
    } else {
        /* Order-0 header (original format, byte 7 = 0) */
        memset(output, 0, LOSSLESS_HEADER_SIZE);
        output[0] = 'C'; output[1] = '3'; output[2] = 'D'; output[3] = 0x04;
        output[4] = 101;  /* quality = lossless */
        output[5] = 0;    /* transform (unused) */
        output[6] = 1;    /* mode: lossless */

        uint32_t ol = (uint32_t)N3;
        output[8] = ol; output[9] = ol >> 8; output[10] = ol >> 16; output[11] = ol >> 24;
        uint32_t rl = (uint32_t)rans_o0_len;
        output[12] = rl; output[13] = rl >> 8; output[14] = rl >> 16; output[15] = rl >> 24;

        for (int i = 0; i < 256; i++) {
            uint16_t f = (uint16_t)model_o0.freq[i];
            output[16 + i * 2] = (uint8_t)(f & 0xFF);
            output[16 + i * 2 + 1] = (uint8_t)(f >> 8);
        }

        memcpy(output + LOSSLESS_HEADER_SIZE, rans_o0, rans_o0_len);
    }

    free(rans_o0); free(rans_ctx);
    return total;
}

static int lossless_decompress_to(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    int use_ctx = (compressed[7] & C3D_FLAG_CTX_LOSSLESS);

    if (compressed_size < 16) return -1;

    uint32_t orig_len = compressed[8] | ((uint32_t)compressed[9]<<8)
                      | ((uint32_t)compressed[10]<<16) | ((uint32_t)compressed[11]<<24);
    uint32_t rl = compressed[12] | ((uint32_t)compressed[13]<<8)
                | ((uint32_t)compressed[14]<<16) | ((uint32_t)compressed[15]<<24);

    if (orig_len != (uint32_t)N3) return -1;

    uint8_t *residuals = (uint8_t *)malloc(N3);
    if (!residuals) return -1;

    if (use_ctx) {
        /* Order-1 context decode with sparse freq tables */
        if (compressed_size < 17) { free(residuals); return -1; }
        int ngroups = compressed[16];
        if (ngroups != NUM_CTX_GROUPS) { free(residuals); return -1; }

        rans_model_t models[NUM_CTX_GROUPS];
        size_t off = 17;
        for (int g = 0; g < NUM_CTX_GROUPS; g++) {
            int consumed = sparse_freq_read(compressed + off, compressed_size - off, &models[g]);
            if (consumed < 0) { free(residuals); return -1; }
            off += (size_t)consumed;
        }

        if (off + rl > compressed_size) { free(residuals); return -1; }
        rans_decode_ctx(compressed + off, (int)rl, models, NULL, residuals, N3);
    } else {
        /* Order-0 decode (original path) */
        if (compressed_size < LOSSLESS_HEADER_SIZE) { free(residuals); return -1; }
        if (LOSSLESS_HEADER_SIZE + rl > compressed_size) { free(residuals); return -1; }

        rans_model_t model;
        uint32_t freq_total = 0;
        for (int i = 0; i < 256; i++) {
            model.freq[i] = compressed[16 + i * 2] | ((uint32_t)compressed[16 + i * 2 + 1] << 8);
            freq_total += model.freq[i];
        }
        if (freq_total != RANS_PROB_SCALE) { free(residuals); return -1; }

        model.cum_freq[0] = 0;
        for (int i = 0; i < 256; i++)
            model.cum_freq[i + 1] = model.cum_freq[i] + model.freq[i];
        model.total = RANS_PROB_SCALE;

        uint8_t *sym_table = (uint8_t *)malloc(RANS_PROB_SCALE);
        if (!sym_table) { free(residuals); return -1; }
        rans_build_decode_table(&model, sym_table);

        rans_decode(compressed + LOSSLESS_HEADER_SIZE, (int)rl, &model, sym_table, residuals, N3);
        free(sym_table);
    }

    unpredict_residuals(residuals, output);
    free(residuals);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 6: Public API
 *
 * Format v4: quantize → VLQ+RLE symbols → 4-stream interleaved rANS
 * Header: magic(4) + quality+reserved(4) + sym_len(4) + rans_len[4](16) + freq_table(512) = 540
 * ════════════════════════════════════════════════════════════════════════════ */

#define HEADER_CTX (4 + 4 + 4 + 4 + NUM_CTX_GROUPS * 512)  /* 2064 bytes (contextual) */
#define HEADER_O0  (4 + 4 + 4 + 4 + 512)                    /* 528 bytes (order-0) */
#define HEADER_SIZE HEADER_CTX  /* max of the two, for bounds */

size_t c3d_compress_bound(void) {
    return HEADER_SIZE + N3 * 2 + 1024;
}

/* Sparse frequency table: write non-zero entries as (sym, freq_lo, freq_hi).
 * Returns number of bytes written starting at p. */
static int sparse_freq_write(uint8_t *p, const rans_model_t *model) {
    int nz = 0;
    for (int i = 0; i < 256; i++)
        if (model->freq[i] > 0) nz++;
    p[0] = (uint8_t)nz;  /* 0 means 256 */
    int off = 1;
    for (int i = 0; i < 256; i++) {
        if (model->freq[i] > 0) {
            p[off++] = (uint8_t)i;
            p[off++] = (uint8_t)(model->freq[i] & 0xFF);
            p[off++] = (uint8_t)(model->freq[i] >> 8);
        }
    }
    return off;
}

/* Sparse frequency table: read. Returns bytes consumed, or -1 on error. */
static int sparse_freq_read(const uint8_t *p, size_t avail, rans_model_t *model) {
    if (avail < 1) return -1;
    int nz = p[0];
    if (nz == 0) nz = 256;
    size_t needed = 1 + (size_t)nz * 3;
    if (avail < needed) return -1;
    memset(model->freq, 0, sizeof(model->freq));
    int off = 1;
    for (int i = 0; i < nz; i++) {
        uint8_t sym = p[off++];
        model->freq[sym] = p[off] | ((uint32_t)p[off+1] << 8);
        off += 2;
    }
    uint32_t total = 0;
    for (int i = 0; i < 256; i++) total += model->freq[i];
    if (total != RANS_PROB_SCALE) return -1;
    model->cum_freq[0] = 0;
    for (int i = 0; i < 256; i++)
        model->cum_freq[i+1] = model->cum_freq[i] + model->freq[i];
    model->total = RANS_PROB_SCALE;
    return off;
}

/* Compute sparse header size for a set of models */
static size_t sparse_header_size(const rans_model_t *models, int nmodels) {
    size_t sz = 17; /* 16 byte fixed header + 1 byte num_groups */
    for (int g = 0; g < nmodels; g++) {
        int nz = 0;
        for (int i = 0; i < 256; i++)
            if (models[g].freq[i] > 0) nz++;
        sz += 1 + nz * 3;
    }
    return sz;
}

/* Pack with sparse frequency tables. Returns total size. */
static size_t pack_sparse(uint8_t *output, int quality, uint8_t flags,
                           const symstream_t *syms,
                           const rans_model_t *models, int nmodels,
                           const uint8_t *rans_data, size_t rans_len) {
    size_t hdr_sz = sparse_header_size(models, nmodels);
    size_t total = hdr_sz + rans_len;
    memset(output, 0, 17);
    output[0] = 'C'; output[1] = '3'; output[2] = 'D'; output[3] = 0x04;
    output[4] = (uint8_t)quality;
    output[7] = flags | C3D_FLAG_SPARSE;
    uint32_t sl = (uint32_t)syms->count;
    output[8] = sl; output[9] = sl>>8; output[10] = sl>>16; output[11] = sl>>24;
    uint32_t rl = (uint32_t)rans_len;
    output[12] = rl; output[13] = rl>>8; output[14] = rl>>16; output[15] = rl>>24;
    output[16] = (uint8_t)nmodels;
    size_t off = 17;
    for (int g = 0; g < nmodels; g++)
        off += sparse_freq_write(output + off, &models[g]);
    memcpy(output + off, rans_data, rans_len);
    return total;
}

/* Pack order-0 encoded output. Returns total size. */
static size_t pack_order0(uint8_t *output, int quality, const symstream_t *syms,
                           const rans_model_t *model, const uint8_t *rans_data, size_t rans_len) {
    size_t total = HEADER_O0 + rans_len;
    memset(output, 0, HEADER_O0);
    output[0] = 'C'; output[1] = '3'; output[2] = 'D'; output[3] = 0x04;
    output[4] = (uint8_t)quality;
    output[7] = C3D_FLAG_INTERLEAVED;  /* order-0 + interleaved */
    uint32_t sl = (uint32_t)syms->count;
    output[8] = sl; output[9] = sl>>8; output[10] = sl>>16; output[11] = sl>>24;
    uint32_t rl = (uint32_t)rans_len;
    output[12] = rl; output[13] = rl>>8; output[14] = rl>>16; output[15] = rl>>24;
    for (int i = 0; i < 256; i++) {
        uint16_t f = (uint16_t)model->freq[i];
        output[16+i*2] = (uint8_t)(f & 0xFF);
        output[16+i*2+1] = (uint8_t)(f >> 8);
    }
    memcpy(output + HEADER_O0, rans_data, rans_len);
    return total;
}

/* Pack contextual encoded output. Returns total size. */
static size_t pack_ctx(uint8_t *output, int quality, const symstream_t *syms,
                        const rans_model_t models[NUM_CTX_GROUPS],
                        const uint8_t *rans_data, size_t rans_len) {
    size_t total = HEADER_CTX + rans_len;
    memset(output, 0, HEADER_CTX);
    output[0] = 'C'; output[1] = '3'; output[2] = 'D'; output[3] = 0x04;
    output[4] = (uint8_t)quality;
    output[7] = 1;  /* contextual flag */
    uint32_t sl = (uint32_t)syms->count;
    output[8] = sl; output[9] = sl>>8; output[10] = sl>>16; output[11] = sl>>24;
    uint32_t rl = (uint32_t)rans_len;
    output[12] = rl; output[13] = rl>>8; output[14] = rl>>16; output[15] = rl>>24;
    for (int g = 0; g < NUM_CTX_GROUPS; g++)
        for (int i = 0; i < 256; i++) {
            uint16_t f = (uint16_t)models[g].freq[i];
            int off = 16 + g*512 + i*2;
            output[off] = (uint8_t)(f & 0xFF);
            output[off+1] = (uint8_t)(f >> 8);
        }
    memcpy(output + HEADER_CTX, rans_data, rans_len);
    return total;
}

/* Encode symbols and pack into output buffer. Tries all 4 encoding modes,
 * picks smallest. Sets appropriate flags in output[7].
 * If rans_buf is non-NULL, it is used as scratch space (workspace path);
 * otherwise buffers are malloc'd internally.
 * Returns total compressed size, or 0 on error.
 */
static size_t encode_and_pack(const symstream_t *syms, int quality,
                               uint8_t extra_flags, uint8_t *output, size_t output_cap,
                               uint8_t *rans_buf, size_t rans_buf_cap) {
    /* Try order-0 */
    rans_model_t model_o0;
    rans_build_model(syms->symbols, syms->count, &model_o0);

    size_t rans_needed = (size_t)syms->count * 2 + 256;
    uint8_t *rans_o0;
    int o0_alloced = 0;
    if (rans_buf) {
        rans_o0 = rans_buf;
    } else {
        rans_o0 = (uint8_t *)malloc(rans_needed);
        if (!rans_o0) return 0;
        o0_alloced = 1;
    }
    size_t rans_o0_len;
    rans_encode_interleaved(syms->symbols, syms->count, &model_o0, rans_o0, &rans_o0_len);
    size_t total_o0 = HEADER_O0 + rans_o0_len;

    /* Try contextual */
    rans_model_t models_ctx[NUM_CTX_GROUPS];
    rans_build_ctx_models(syms->symbols, syms->count, models_ctx);

    uint8_t *rans_ctx;
    int ctx_alloced = 0;
    size_t rans_ctx_len;
    if (rans_buf) {
        rans_ctx = rans_buf + rans_needed;
        size_t ctx_cap = (rans_buf_cap > rans_needed) ? rans_buf_cap - rans_needed : 0;
        rans_encode_ctx_buf(syms->symbols, syms->count, models_ctx, rans_ctx, ctx_cap, &rans_ctx_len);
    } else {
        rans_encode_ctx(syms->symbols, syms->count, models_ctx, &rans_ctx, &rans_ctx_len);
        ctx_alloced = 1;
    }
    size_t total_ctx = HEADER_CTX + rans_ctx_len;

    /* Also compute sparse header sizes */
    size_t sparse_o0_total = sparse_header_size(&model_o0, 1) + rans_o0_len;
    size_t sparse_ctx_total = sparse_header_size(models_ctx, NUM_CTX_GROUPS) + rans_ctx_len;

    /* Pick the smallest of all 4 options */
    size_t best = total_o0;
    int best_mode = 0;
    if (total_ctx < best) { best = total_ctx; best_mode = 1; }
    if (sparse_o0_total < best) { best = sparse_o0_total; best_mode = 2; }
    if (sparse_ctx_total < best) { best = sparse_ctx_total; best_mode = 3; }

    if (best > output_cap) {
        if (o0_alloced) free(rans_o0);
        if (ctx_alloced) free(rans_ctx);
        return 0;
    }

    size_t total;
    switch (best_mode) {
    case 0: total = pack_order0(output, quality, syms, &model_o0, rans_o0, rans_o0_len); break;
    case 1: total = pack_ctx(output, quality, syms, models_ctx, rans_ctx, rans_ctx_len); break;
    case 2: total = pack_sparse(output, quality, C3D_FLAG_INTERLEAVED, syms, &model_o0, 1, rans_o0, rans_o0_len); break;
    case 3: total = pack_sparse(output, quality, 1, syms, models_ctx, NUM_CTX_GROUPS, rans_ctx, rans_ctx_len); break;
    default: total = 0; break;
    }
    output[7] |= extra_flags;

    if (o0_alloced) free(rans_o0);
    if (ctx_alloced) free(rans_ctx);
    return total;
}

size_t c3d_compress_to(const uint8_t *input, int quality, uint8_t *output, size_t output_cap) {
    if (!input || !output) return 0;
    init_tables();

    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;

    /* Lossless mode */
    if (quality == 101)
        return lossless_compress_to(input, output, output_cap);

    /* Float conversion + centering */
    float *vol = (float *)malloc(N3 * sizeof(float));
    if (!vol) return 0;
    for (int i = 0; i < N3; i++)
        vol[i] = (float)input[i] - 128.0f;

    /* 3D DCT */
    dct3d_forward_all(vol);

    /* Dead-zone quantization */
    float step = quality_to_step(quality);
    float *qtable = (float *)malloc(N3 * sizeof(float));
    float *dz_table = (float *)malloc(N3 * sizeof(float));
    float *bias_table = (float *)malloc(N3 * sizeof(float));
    int32_t *quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    if (!qtable || !dz_table || !bias_table || !quant) {
        free(vol); free(qtable); free(dz_table); free(bias_table); free(quant);
        return 0;
    }
    compute_quant_table(step, qtable, dz_table, bias_table);
    quantize_volume(vol, quant, qtable, dz_table, bias_table);

    /* RD-optimized thresholding of small coefficients. */
    rd_optimize_coeffs(vol, quant, qtable, step * step * 0.8f);

    /* AC coefficient prediction (delta coding on low-frequency band) */
    coeff_predict_forward(quant);

    /* VLQ+RLE symbol encoding */
    symstream_t syms;
    sym_init(&syms);
    coeffs_to_symbols(quant, &syms);

    size_t total = encode_and_pack(&syms, quality, C3D_FLAG_COEFF_PRED,
                                    output, output_cap, NULL, 0);

    free(vol); free(dz_table); free(bias_table); free(qtable); free(quant); free(syms.symbols);
    return total;
}

c3d_compressed_t c3d_compress(const uint8_t *input, int quality) {
    if (!input) return (c3d_compressed_t){ .data = NULL, .size = 0 };
    size_t bound = c3d_compress_bound();
    uint8_t *buf = (uint8_t *)malloc(bound);
    if (!buf) return (c3d_compressed_t){ .data = NULL, .size = 0 };

    size_t actual = c3d_compress_to(input, quality, buf, bound);
    if (actual == 0) {
        free(buf);
        return (c3d_compressed_t){ .data = NULL, .size = 0 };
    }

    buf = (uint8_t *)realloc(buf, actual);
    return (c3d_compressed_t){ .data = buf, .size = actual };
}

static int c3d_decompress_wavelet_internal(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

static int c3d_decompress_roi(const uint8_t *compressed, size_t compressed_size, uint8_t *output);

/* ── CRC32 and Metadata support ── */

static uint32_t c3d_crc32_table[4][256];
static int c3d_crc32_table_init = 0;

static void c3d_crc32_init(void) {
    if (c3d_crc32_table_init) return;
    /* Build base table (slice 0) */
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++)
            c = (c >> 1) ^ (0xEDB88320 & (-(c & 1)));
        c3d_crc32_table[0][i] = c;
    }
    /* Build extended tables for slicing-by-4 (slices 1-3) */
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = c3d_crc32_table[0][i];
        for (int s = 1; s < 4; s++) {
            c = (c >> 8) ^ c3d_crc32_table[0][c & 0xFF];
            c3d_crc32_table[s][i] = c;
        }
    }
    c3d_crc32_table_init = 1;
}

static uint32_t c3d_crc32(const uint8_t *data, size_t len) {
    c3d_crc32_init();
    uint32_t crc = 0xFFFFFFFF;

    /* Process 4 bytes at a time using slicing-by-4 */
    while (len >= 4) {
        crc ^= (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
               ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
        crc = c3d_crc32_table[3][crc & 0xFF] ^
              c3d_crc32_table[2][(crc >> 8) & 0xFF] ^
              c3d_crc32_table[1][(crc >> 16) & 0xFF] ^
              c3d_crc32_table[0][(crc >> 24) & 0xFF];
        data += 4;
        len -= 4;
    }

    /* Handle remaining bytes */
    while (len--)
        crc = (crc >> 8) ^ c3d_crc32_table[0][(crc ^ *data++) & 0xFF];

    return crc ^ 0xFFFFFFFF;
}

#define C3D_CRC_SECTION_SIZE  8
#define C3D_META_SECTION_SIZE 36
#define C3D_META_DATA_SIZE    28

static void c3d_write_le32(uint8_t *p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
static uint32_t c3d_read_le32(const uint8_t *p) {
    return p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static void c3d_write_le16(uint8_t *p, uint16_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8);
}
static uint16_t c3d_read_le16(const uint8_t *p) {
    return p[0]|((uint16_t)p[1]<<8);
}
static void c3d_write_le_float(uint8_t *p, float v) {
    uint32_t u; memcpy(&u,&v,4); c3d_write_le32(p,u);
}
static float c3d_read_le_float(const uint8_t *p) {
    uint32_t u=c3d_read_le32(p); float v; memcpy(&v,&u,4); return v;
}

static size_t c3d_payload_end(const uint8_t *d, size_t sz) {
    if (sz<8) return sz;
    if (d[0]!='C'||d[1]!='3'||d[2]!='D'||d[3]!=0x04) return sz;
    if (d[6]==1) {
        if (d[7] & C3D_FLAG_CTX_LOSSLESS) {
            /* Lossless context mode: sparse freq tables */
            if (sz < 17) return sz;
            int ngroups = d[16];
            size_t off = 17;
            for (int g = 0; g < ngroups; g++) {
                if (off >= sz) return sz;
                int nz = d[off]; if (nz == 0) nz = 256;
                off += 1 + (size_t)nz * 3;
            }
            size_t end = off + c3d_read_le32(d+12);
            return end <= sz ? end : sz;
        }
        if (sz<(size_t)LOSSLESS_HEADER_SIZE) return sz;
        size_t end=LOSSLESS_HEADER_SIZE+c3d_read_le32(d+12);
        return end<=sz?end:sz;
    }
    size_t tr = c3d_read_le32(d+12);  /* single rans_len field */
    if (d[7] & C3D_FLAG_SPARSE) {
        /* Sparse header: scan freq tables to find where rANS data starts */
        if (sz < 17) return sz;
        int ngroups = d[16];
        size_t off = 17;
        for (int g = 0; g < ngroups; g++) {
            if (off >= sz) return sz;
            int nz = d[off]; if (nz == 0) nz = 256;
            off += 1 + (size_t)nz * 3;
        }
        size_t end = off + tr;
        return end <= sz ? end : sz;
    }
    int use_ctx = (d[7] & 1);
    size_t hdr = use_ctx ? (size_t)HEADER_CTX : (size_t)HEADER_O0;
    if (sz<hdr) return sz;
    size_t end=hdr+tr;
    return end<=sz?end:sz;
}

static void c3d_parse_sections(const uint8_t *d, size_t sz,
    uint32_t *out_crc, int *hc, c3d_metadata_t *out_meta, int *hm) {
    *hc=0; *hm=0;
    size_t off=c3d_payload_end(d,sz);
    while (off+8<=sz) {
        if (d[off]=='C'&&d[off+1]=='R'&&d[off+2]=='C'&&d[off+3]==0) {
            *out_crc=c3d_read_le32(d+off+4); *hc=1; off+=C3D_CRC_SECTION_SIZE;
        } else if (d[off]=='M'&&d[off+1]=='E'&&d[off+2]=='T'&&d[off+3]=='A') {
            uint32_t ml=c3d_read_le32(d+off+4);
            if (off+8+ml<=sz && ml>=C3D_META_DATA_SIZE && out_meta) {
                const uint8_t *md=d+off+8;
                for(int i=0;i<3;i++) out_meta->voxel_size[i]=c3d_read_le_float(md+i*4);
                for(int i=0;i<3;i++) out_meta->origin[i]=c3d_read_le_float(md+12+i*4);
                out_meta->modality=c3d_read_le16(md+24);
                out_meta->bits_per_voxel=c3d_read_le16(md+26);
                *hm=1;
            }
            off+=8+ml;
        } else break;
    }
}

static int c3d_verify_crc(const uint8_t *compressed, size_t size) {
    if (size<8 || !(compressed[7]&C3D_FLAG_HAS_CRC)) return 0;
    uint32_t stored; int hc,hm; c3d_metadata_t dm;
    c3d_parse_sections(compressed,size,&stored,&hc,&dm,&hm);
    if (!hc) return 0;
    size_t pe=c3d_payload_end(compressed,size);
    return (c3d_crc32(compressed,pe)==stored)?0:C3D_ERR_CHECKSUM;
}

c3d_compressed_t c3d_compress_meta(const uint8_t *input, int quality, const c3d_metadata_t *meta) {
    if (!input) return (c3d_compressed_t){NULL, 0};
    c3d_compressed_t base=c3d_compress(input,quality);
    if (!base.data) return base;
    int hm=(meta!=NULL);
    size_t extra=C3D_CRC_SECTION_SIZE+(hm?C3D_META_SECTION_SIZE:0);
    uint8_t *p=(uint8_t*)realloc(base.data,base.size+extra);
    if (!p){free(base.data);return (c3d_compressed_t){NULL,0};}
    p[7]|=C3D_FLAG_HAS_CRC|(hm?C3D_FLAG_HAS_META:0);
    uint32_t crc=c3d_crc32(p,base.size);
    size_t off=base.size;
    p[off]='C';p[off+1]='R';p[off+2]='C';p[off+3]=0;
    c3d_write_le32(p+off+4,crc); off+=C3D_CRC_SECTION_SIZE;
    if (hm) {
        p[off]='M';p[off+1]='E';p[off+2]='T';p[off+3]='A';
        c3d_write_le32(p+off+4,C3D_META_DATA_SIZE);
        uint8_t *md=p+off+8;
        for(int i=0;i<3;i++) c3d_write_le_float(md+i*4,meta->voxel_size[i]);
        for(int i=0;i<3;i++) c3d_write_le_float(md+12+i*4,meta->origin[i]);
        c3d_write_le16(md+24,meta->modality);
        c3d_write_le16(md+26,meta->bits_per_voxel);
        off+=C3D_META_SECTION_SIZE;
    }
    return (c3d_compressed_t){.data=p,.size=base.size+extra};
}

int c3d_get_metadata(const uint8_t *compressed, size_t size, c3d_metadata_t *meta) {
    if (!compressed||size<8||!meta) return -1;
    memset(meta,0,sizeof(*meta));
    if (!(compressed[7]&C3D_FLAG_HAS_META)) return -1;
    uint32_t cv; int hc,hm;
    c3d_parse_sections(compressed,size,&cv,&hc,meta,&hm);
    return hm?0:-1;
}

/* Unified rANS decode dispatch. Reads flags/lengths from header bytes 7,12-15.
 * Decodes sym_len symbols into the provided buffer.
 * Returns 0 on success, -1 on error. */
static int decode_symbols(const uint8_t *compressed, size_t compressed_size,
                          uint8_t *symbols, uint32_t sym_len) {
    if (compressed_size < 16) return -1;
    int use_ctx = (compressed[7] & 1);
    int is_sparse = (compressed[7] & C3D_FLAG_SPARSE) != 0;

    uint32_t rans_len_total = compressed[12] | ((uint32_t)compressed[13]<<8)
                            | ((uint32_t)compressed[14]<<16) | ((uint32_t)compressed[15]<<24);

    if (is_sparse) {
        if (compressed_size < 17) return -1;
        int ngroups = compressed[16];
        size_t off = 17;
        if (ngroups == 1) {
            rans_model_t model;
            int consumed = sparse_freq_read(compressed + off, compressed_size - off, &model);
            if (consumed < 0) return -1;
            off += consumed;
            if (off + rans_len_total > compressed_size) return -1;
            if (compressed[7] & C3D_FLAG_INTERLEAVED) {
                rans_decode_interleaved(compressed + off, (int)rans_len_total,
                                        &model, symbols, (int)sym_len);
            } else {
                rans_decode(compressed + off, (int)rans_len_total,
                            &model, NULL, symbols, (int)sym_len);
            }
        } else {
            rans_model_t models[NUM_CTX_GROUPS];
            int gread = ngroups < NUM_CTX_GROUPS ? ngroups : NUM_CTX_GROUPS;
            for (int g = 0; g < gread; g++) {
                int consumed = sparse_freq_read(compressed + off, compressed_size - off, &models[g]);
                if (consumed < 0) return -1;
                off += consumed;
            }
            for (int g = gread; g < NUM_CTX_GROUPS; g++)
                models[g] = models[gread > 0 ? gread - 1 : 0];
            if (off + rans_len_total > compressed_size) return -1;
            rans_decode_ctx(compressed + off, (int)rans_len_total,
                            models, NULL, symbols, (int)sym_len);
        }
    } else if (use_ctx) {
        size_t hdr_size = (size_t)HEADER_CTX;
        if (compressed_size < hdr_size) return -1;
        if (hdr_size + rans_len_total > compressed_size) return -1;
        rans_model_t models[NUM_CTX_GROUPS];
        for (int g = 0; g < NUM_CTX_GROUPS; g++) {
            uint32_t freq_total = 0;
            for (int i = 0; i < 256; i++) {
                models[g].freq[i] = compressed[16 + g*512 + i*2]
                                  | ((uint32_t)compressed[16 + g*512 + i*2 + 1] << 8);
                freq_total += models[g].freq[i];
            }
            if (freq_total != RANS_PROB_SCALE) return -1;
            models[g].cum_freq[0] = 0;
            for (int i = 0; i < 256; i++)
                models[g].cum_freq[i+1] = models[g].cum_freq[i] + models[g].freq[i];
            models[g].total = RANS_PROB_SCALE;
        }
        rans_decode_ctx(compressed + HEADER_CTX, (int)rans_len_total,
                        models, NULL, symbols, (int)sym_len);
    } else {
        size_t hdr_size = (size_t)HEADER_O0;
        if (compressed_size < hdr_size) return -1;
        if (hdr_size + rans_len_total > compressed_size) return -1;
        rans_model_t model;
        uint32_t freq_total = 0;
        for (int i = 0; i < 256; i++) {
            model.freq[i] = compressed[16 + i*2] | ((uint32_t)compressed[16 + i*2 + 1] << 8);
            freq_total += model.freq[i];
        }
        if (freq_total != RANS_PROB_SCALE) return -1;
        model.cum_freq[0] = 0;
        for (int i = 0; i < 256; i++)
            model.cum_freq[i+1] = model.cum_freq[i] + model.freq[i];
        model.total = RANS_PROB_SCALE;
        if (compressed[7] & C3D_FLAG_INTERLEAVED) {
            rans_decode_interleaved(compressed + HEADER_O0, (int)rans_len_total,
                                    &model, symbols, (int)sym_len);
        } else {
            rans_decode(compressed + HEADER_O0, (int)rans_len_total,
                        &model, NULL, symbols, (int)sym_len);
        }
    }
    return 0;
}

/* Workspace-aware decode_symbols: reuses pre-allocated dtable from workspace. */
static int decode_symbols_ws(const uint8_t *compressed, size_t compressed_size,
                              uint8_t *symbols, uint32_t sym_len,
                              rans_decode_entry_t *dtable) {
    if (compressed_size < 16) return -1;
    int use_ctx = (compressed[7] & 1);
    int is_sparse = (compressed[7] & C3D_FLAG_SPARSE) != 0;

    uint32_t rans_len_total = compressed[12] | ((uint32_t)compressed[13]<<8)
                            | ((uint32_t)compressed[14]<<16) | ((uint32_t)compressed[15]<<24);

    if (is_sparse) {
        if (compressed_size < 17) return -1;
        int ngroups = compressed[16];
        size_t off = 17;
        if (ngroups == 1) {
            rans_model_t model;
            int consumed = sparse_freq_read(compressed + off, compressed_size - off, &model);
            if (consumed < 0) return -1;
            off += consumed;
            if (off + rans_len_total > compressed_size) return -1;
            if (compressed[7] & C3D_FLAG_INTERLEAVED) {
                rans_decode_interleaved_core(compressed + off, (int)rans_len_total,
                                              &model, dtable, symbols, (int)sym_len);
            } else {
                rans_decode_core(compressed + off, (int)rans_len_total,
                                  &model, dtable, symbols, (int)sym_len);
            }
        } else {
            rans_model_t models[NUM_CTX_GROUPS];
            int gread = ngroups < NUM_CTX_GROUPS ? ngroups : NUM_CTX_GROUPS;
            for (int g = 0; g < gread; g++) {
                int consumed = sparse_freq_read(compressed + off, compressed_size - off, &models[g]);
                if (consumed < 0) return -1;
                off += consumed;
            }
            for (int g = gread; g < NUM_CTX_GROUPS; g++)
                models[g] = models[gread > 0 ? gread - 1 : 0];
            if (off + rans_len_total > compressed_size) return -1;
            rans_decode_ctx_core(compressed + off, (int)rans_len_total,
                                  models, dtable, symbols, (int)sym_len);
        }
    } else if (use_ctx) {
        size_t hdr_size = (size_t)HEADER_CTX;
        if (compressed_size < hdr_size) return -1;
        if (hdr_size + rans_len_total > compressed_size) return -1;
        rans_model_t models[NUM_CTX_GROUPS];
        for (int g = 0; g < NUM_CTX_GROUPS; g++) {
            uint32_t freq_total = 0;
            for (int i = 0; i < 256; i++) {
                models[g].freq[i] = compressed[16 + g*512 + i*2]
                                  | ((uint32_t)compressed[16 + g*512 + i*2 + 1] << 8);
                freq_total += models[g].freq[i];
            }
            if (freq_total != RANS_PROB_SCALE) return -1;
            models[g].cum_freq[0] = 0;
            for (int i = 0; i < 256; i++)
                models[g].cum_freq[i+1] = models[g].cum_freq[i] + models[g].freq[i];
            models[g].total = RANS_PROB_SCALE;
        }
        rans_decode_ctx_core(compressed + HEADER_CTX, (int)rans_len_total,
                              models, dtable, symbols, (int)sym_len);
    } else {
        size_t hdr_size = (size_t)HEADER_O0;
        if (compressed_size < hdr_size) return -1;
        if (hdr_size + rans_len_total > compressed_size) return -1;
        rans_model_t model;
        uint32_t freq_total = 0;
        for (int i = 0; i < 256; i++) {
            model.freq[i] = compressed[16 + i*2] | ((uint32_t)compressed[16 + i*2 + 1] << 8);
            freq_total += model.freq[i];
        }
        if (freq_total != RANS_PROB_SCALE) return -1;
        model.cum_freq[0] = 0;
        for (int i = 0; i < 256; i++)
            model.cum_freq[i+1] = model.cum_freq[i] + model.freq[i];
        model.total = RANS_PROB_SCALE;
        if (compressed[7] & C3D_FLAG_INTERLEAVED) {
            rans_decode_interleaved_core(compressed + HEADER_O0, (int)rans_len_total,
                                          &model, dtable, symbols, (int)sym_len);
        } else {
            rans_decode_core(compressed + HEADER_O0, (int)rans_len_total,
                              &model, dtable, symbols, (int)sym_len);
        }
    }
    return 0;
}

int c3d_decompress_to(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    if (!compressed || !output) return -1;
    init_tables();
    if (compressed_size < 8) return -1;

    /* ROI format: "C3R\x01" */
    if (compressed[0] == 'C' && compressed[1] == '3' &&
        compressed[2] == 'R' && compressed[3] == 0x01)
        return c3d_decompress_roi(compressed, compressed_size, output);

    if (compressed[0] != 'C' || compressed[1] != '3' || compressed[2] != 'D' || compressed[3] != 0x04)
        return -1;

    /* CRC verification before decoding */
    { int rc = c3d_verify_crc(compressed, compressed_size); if (rc) return rc; }

    /* Lossless mode: byte 6 == 1 */
    if (compressed[6] == 1)
        return lossless_decompress_to(compressed, compressed_size, output);

    /* Auto-detect transform type from header byte 5 */
    if (compressed[5] == C3D_TRANSFORM_WAVELET)
        return c3d_decompress_wavelet_internal(compressed, compressed_size, output);

    if (compressed_size < 16) return -1;

    int quality = compressed[4];
    uint32_t sym_len = compressed[8] | ((uint32_t)compressed[9]<<8)
                     | ((uint32_t)compressed[10]<<16) | ((uint32_t)compressed[11]<<24);

    if (sym_len > (uint32_t)N3 * 6) return -1;

    uint8_t *syms = (uint8_t *)malloc(sym_len > 0 ? sym_len : 1);
    if (!syms) return -1;

    if (decode_symbols(compressed, compressed_size, syms, sym_len) != 0) {
        free(syms);
        return -1;
    }

    /* VLQ decode -> quantized coefficients */
    int32_t *quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    if (!quant) { free(syms); return -1; }
    if (symbols_to_coeffs(syms, (int)sym_len, quant) != 0) {
        free(syms); free(quant);
        return -1;
    }

    /* Undo AC coefficient prediction if enabled */
    if (compressed[7] & C3D_FLAG_COEFF_PRED)
        coeff_predict_inverse(quant);

    /* Dequantize */
    float step = quality_to_step(quality);
    float *qtable = (float *)malloc(N3 * sizeof(float));
    float *vol = (float *)malloc(N3 * sizeof(float));
    if (!qtable || !vol) { free(syms); free(quant); free(qtable); free(vol); return -1; }
    compute_dequant_table(step, qtable);
    dequantize_volume(quant, vol, qtable);

    /* Inverse 3D DCT */
    dct3d_inverse_all(vol);

    /* Convert back to uint8 */
    float_to_u8_biased(vol, output, N3);

    free(syms); free(quant); free(qtable); free(vol);
    return 0;
}

int c3d_decompress(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    return c3d_decompress_to(compressed, compressed_size, output);
}

int c3d_get_size(const uint8_t *compressed, size_t compressed_size) {
    if (!compressed || compressed_size < 4) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3') return -1;

    /* Shard format "C3S\x01" — not a single cube */
    if (compressed[2] == 'S' && compressed[3] == 0x01) return -1;

    /* Lossy/lossless "C3D\x04", ROI "C3R\x01", Progressive "C3P\x01" */
    if ((compressed[2] == 'D' && compressed[3] == 0x04) ||
        (compressed[2] == 'R' && compressed[3] == 0x01) ||
        (compressed[2] == 'P' && compressed[3] == 0x01))
        return C3D_BLOCK_SIZE;

    return -1;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 6b: Progressive bitstream API
 *
 * Format: "C3P\x01" (4) + quality (1) + raw VLQ+RLE symbol bytes
 * Truncatable at any byte boundary after the 5-byte header.
 * ════════════════════════════════════════════════════════════════════════════ */

#define PROG_HEADER_SIZE 5

c3d_compressed_t c3d_compress_progressive(const uint8_t *input, int quality) {
    if (!input) return (c3d_compressed_t){ .data = NULL, .size = 0 };
    init_tables();

    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;

    float *vol = (float *)malloc(N3 * sizeof(float));
    for (int i = 0; i < N3; i++)
        vol[i] = (float)input[i] - 128.0f;

    dct3d_forward_all(vol);

    float step = quality_to_step(quality);
    float *qtable = (float *)malloc(N3 * sizeof(float));
    float *dz_table = (float *)malloc(N3 * sizeof(float));
    float *bias_table = (float *)malloc(N3 * sizeof(float));
    compute_quant_table(step, qtable, dz_table, bias_table);
    int32_t *quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    quantize_volume(vol, quant, qtable, dz_table, bias_table);
    rd_optimize_coeffs(vol, quant, qtable, step * step * 0.8f);

    symstream_t syms;
    sym_init(&syms);
    coeffs_to_symbols(quant, &syms);

    size_t total = PROG_HEADER_SIZE + (size_t)syms.count;
    uint8_t *out = (uint8_t *)malloc(total);
    if (!out) {
        free(vol); free(dz_table); free(bias_table); free(qtable); free(quant); free(syms.symbols);
        return (c3d_compressed_t){ .data = NULL, .size = 0 };
    }

    out[0] = 'C'; out[1] = '3'; out[2] = 'P'; out[3] = 0x01;
    out[4] = (uint8_t)quality;
    memcpy(out + PROG_HEADER_SIZE, syms.symbols, syms.count);

    free(vol); free(dz_table); free(bias_table); free(qtable); free(quant); free(syms.symbols);
    return (c3d_compressed_t){ .data = out, .size = total };
}

static int symbols_to_coeffs_partial(const uint8_t *syms, int nsyms, int32_t *quant) {
    memset(quant, 0, N3 * sizeof(int32_t));
    int si = 0, ci = 0;
    while (si < nsyms && ci < N3) {
        uint8_t b = syms[si++];
        if (b == 0x00) {
            if (si >= nsyms) break;
            int run = (int)syms[si++] + 1;
            for (int j = 0; j < run && ci < N3; j++)
                quant[zigzag_order[ci++]] = 0;
        } else if (b & 0x80) {
            uint32_t folded = (b & 0x7F);
            int shift = 7;
            int complete = 0;
            while (si < nsyms) {
                uint8_t cb = syms[si++];
                folded |= (uint32_t)(cb & 0x7F) << shift;
                shift += 7;
                if (!(cb & 0x80)) { complete = 1; break; }
            }
            if (!complete) break;
            int32_t val = (int32_t)(folded >> 1) ^ -(int32_t)(folded & 1);
            quant[zigzag_order[ci++]] = val;
        } else {
            uint32_t folded = (uint32_t)b;
            int32_t val = (int32_t)(folded >> 1) ^ -(int32_t)(folded & 1);
            quant[zigzag_order[ci++]] = val;
        }
    }
    return 0;
}

int c3d_decompress_progressive(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    if (!compressed || !output) return -1;
    init_tables();
    if (compressed_size < PROG_HEADER_SIZE) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3' || compressed[2] != 'P' || compressed[3] != 0x01)
        return -1;

    int quality = compressed[4];
    if (quality < 1 || quality > 100) return -1;

    int nsyms = (int)(compressed_size - PROG_HEADER_SIZE);
    int32_t *quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    float *qtable = (float *)malloc(N3 * sizeof(float));
    float *vol = (float *)malloc(N3 * sizeof(float));
    if (!quant || !qtable || !vol) { free(quant); free(qtable); free(vol); return -1; }
    symbols_to_coeffs_partial(compressed + PROG_HEADER_SIZE, nsyms, quant);

    float step = quality_to_step(quality);
    compute_dequant_table(step, qtable);
    dequantize_volume(quant, vol, qtable);

    dct3d_inverse_all(vol);

    float_to_u8_biased(vol, output, N3);

    free(quant); free(qtable); free(vol);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 7: Shard API — inter-chunk DC delta coding
 *
 * Format: "C3S\x01" (4) | nx (1) | ny (1) | nz (1) | quality (1)
 *       | per-chunk DC delta (float32 LE) x nchunks
 *       | per-chunk compressed size (uint32 LE) x nchunks
 *       | concatenated per-chunk compressed blobs
 * ════════════════════════════════════════════════════════════════════════════ */

#define SHARD_HEADER_SIZE 8

static float compute_mean(const uint8_t *chunk) {
    uint32_t sum = 0;
#ifdef __aarch64__
    uint32x4_t acc = vdupq_n_u32(0);
    for (int i = 0; i < N3; i += 16) {
        uint8x16_t v = vld1q_u8(chunk + i);
        uint16x8_t w = vpaddlq_u8(v);
        acc = vpadalq_u16(acc, w);
    }
    sum = vaddvq_u32(acc);
#elif defined(__x86_64__) || defined(_M_X64)
    __m128i acc = _mm_setzero_si128();
    __m128i zero = _mm_setzero_si128();
    for (int i = 0; i < N3; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(chunk + i));
        acc = _mm_add_epi32(acc, _mm_sad_epu8(v, zero));
    }
    sum = (uint32_t)_mm_extract_epi32(acc, 0) + (uint32_t)_mm_extract_epi32(acc, 2);
#else
    for (int i = 0; i < N3; i++)
        sum += chunk[i];
#endif
    return (float)sum / N3;
}

static int shard_pred_index(int cx, int cy, int cz, int nx, int ny) {
    if (cx > 0) return cz * ny * nx + cy * nx + (cx - 1);
    if (cy > 0) return cz * ny * nx + (cy - 1) * nx + cx;
    if (cz > 0) return (cz - 1) * ny * nx + cy * nx + cx;
    return -1;
}

c3d_compressed_t c3d_compress_shard(const uint8_t **chunks, int nx, int ny, int nz, int quality, int num_threads) {
    c3d_compressed_t result = {NULL, 0};
    if (!chunks) return result;
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > 255 || ny > 255 || nz > 255)
        return result;
    int64_t nchunks64 = (int64_t)nx * (int64_t)ny * (int64_t)nz;
    if (nchunks64 > INT32_MAX)
        return result;
    int nchunks = (int)nchunks64;

    init_tables();
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;

    float *means = (float *)malloc(nchunks * sizeof(float));
    float *deltas = (float *)malloc(nchunks * sizeof(float));
    c3d_compressed_t *comp = (c3d_compressed_t *)malloc(nchunks * sizeof(c3d_compressed_t));
    if (!means || !deltas || !comp) {
        free(means); free(deltas); free(comp);
        return result;
    }

    /* Sequential: compute means and deltas (depends on neighbor means) */
    for (int cz = 0; cz < nz; cz++)
        for (int cy = 0; cy < ny; cy++)
            for (int cx = 0; cx < nx; cx++) {
                int idx = cz * ny * nx + cy * nx + cx;
                means[idx] = compute_mean(chunks[idx]);
                int pred = shard_pred_index(cx, cy, cz, nx, ny);
                float predicted = (pred >= 0) ? means[pred] : 128.0f;
                deltas[idx] = means[idx] - predicted;
            }

    /* Sequential: build shifted chunks, then compress in parallel */
    uint8_t **shifted_chunks = (uint8_t **)malloc(nchunks * sizeof(uint8_t *));
    for (int i = 0; i < nchunks; i++) {
        shifted_chunks[i] = (uint8_t *)malloc(N3);
        int cz = i / (ny * nx), cy = (i / nx) % ny, cx = i % nx;
        int pred = shard_pred_index(cx, cy, cz, nx, ny);
        float predicted = (pred >= 0) ? means[pred] : 128.0f;
        float shift = means[i] - predicted;

        for (int j = 0; j < N3; j++) {
            float v = (float)chunks[i][j] - shift;
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            shifted_chunks[i][j] = (uint8_t)roundf(v);
        }
    }

    /* Parallel: compress all shifted chunks */
    const uint8_t **const_shifted = (const uint8_t **)shifted_chunks;
    c3d_compress_batch(const_shifted, nchunks, quality, num_threads, comp);

    for (int i = 0; i < nchunks; i++)
        free(shifted_chunks[i]);
    free(shifted_chunks);

    size_t total = SHARD_HEADER_SIZE + nchunks * (sizeof(float) + 4);
    for (int i = 0; i < nchunks; i++)
        total += comp[i].size;

    uint8_t *out = (uint8_t *)malloc(total);
    if (!out) {
        for (int i = 0; i < nchunks; i++) free(comp[i].data);
        free(comp); free(means); free(deltas);
        return result;
    }
    uint8_t *p = out;
    p[0] = 'C'; p[1] = '3'; p[2] = 'S'; p[3] = 0x01;
    p[4] = (uint8_t)nx; p[5] = (uint8_t)ny; p[6] = (uint8_t)nz;
    p[7] = (uint8_t)quality;
    p += SHARD_HEADER_SIZE;

    for (int i = 0; i < nchunks; i++) {
        memcpy(p, &deltas[i], sizeof(float));
        p += sizeof(float);
    }
    for (int i = 0; i < nchunks; i++) {
        uint32_t sz = (uint32_t)comp[i].size;
        p[0] = sz; p[1] = sz >> 8; p[2] = sz >> 16; p[3] = sz >> 24;
        p += 4;
    }
    for (int i = 0; i < nchunks; i++) {
        memcpy(p, comp[i].data, comp[i].size);
        p += comp[i].size;
        free(comp[i].data);
    }

    free(means); free(deltas); free(comp);
    result.data = out;
    result.size = total;
    return result;
}

int c3d_decompress_shard(const uint8_t *compressed, size_t compressed_size,
                          uint8_t **chunks, int nx, int ny, int nz, int num_threads) {
    if (!compressed || !chunks) return -1;
    if (compressed_size < SHARD_HEADER_SIZE) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3' || compressed[2] != 'S' || compressed[3] != 0x01)
        return -1;
    if (compressed[4] != (uint8_t)nx || compressed[5] != (uint8_t)ny || compressed[6] != (uint8_t)nz)
        return -1;

    int nchunks = nx * ny * nz;
    const uint8_t *p = compressed + SHARD_HEADER_SIZE;
    size_t remaining = compressed_size - SHARD_HEADER_SIZE;

    size_t meta_size = nchunks * (sizeof(float) + 4);
    if (remaining < meta_size) return -1;

    float *deltas = (float *)malloc(nchunks * sizeof(float));
    uint32_t *sizes = (uint32_t *)malloc(nchunks * sizeof(uint32_t));
    if (!deltas || !sizes) { free(deltas); free(sizes); return -1; }
    for (int i = 0; i < nchunks; i++) {
        memcpy(&deltas[i], p, sizeof(float));
        p += sizeof(float);
    }

    for (int i = 0; i < nchunks; i++) {
        sizes[i] = p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        p += 4;
    }
    remaining -= meta_size;

    /* Compute per-chunk offsets and validate sizes */
    size_t off = 0;
    c3d_compressed_t *comp_array = (c3d_compressed_t *)malloc(nchunks * sizeof(c3d_compressed_t));
    if (!comp_array) { free(deltas); free(sizes); return -1; }
    for (int i = 0; i < nchunks; i++) {
        if (sizes[i] > remaining - off) {
            free(deltas); free(sizes); free(comp_array);
            return -1;
        }
        comp_array[i].data = (uint8_t *)(p + off);
        comp_array[i].size = sizes[i];
        off += sizes[i];
    }

    /* Parallel: decompress all chunks */
    int ret = c3d_decompress_batch(comp_array, nchunks, num_threads, chunks);
    free(comp_array);

    if (ret != 0) {
        free(deltas); free(sizes);
        return -1;
    }

    /* Sequential: reconstruct means and apply DC correction */
    float *means = (float *)malloc(nchunks * sizeof(float));
    if (!means) { free(deltas); free(sizes); return -1; }
    for (int cz = 0; cz < nz; cz++)
        for (int cy = 0; cy < ny; cy++)
            for (int cx = 0; cx < nx; cx++) {
                int idx = cz * ny * nx + cy * nx + cx;
                int pred = shard_pred_index(cx, cy, cz, nx, ny);
                float predicted = (pred >= 0) ? means[pred] : 128.0f;
                means[idx] = predicted + deltas[idx];
            }

    for (int i = 0; i < nchunks; i++) {
        int cz = i / (ny * nx), cy = (i / nx) % ny, cx = i % nx;
        int pred = shard_pred_index(cx, cy, cz, nx, ny);
        float predicted = (pred >= 0) ? means[pred] : 128.0f;
        float shift = means[i] - predicted;

        for (int j = 0; j < N3; j++) {
            float v = (float)chunks[i][j] + shift;
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            chunks[i][j] = (uint8_t)roundf(v);
        }
    }

    free(deltas); free(sizes); free(means);
    return 0;
}
int c3d_shard_chunk_count(const uint8_t *compressed, size_t compressed_size) {
    if (!compressed) return -1;
    if (compressed_size < SHARD_HEADER_SIZE) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3' || compressed[2] != 'S' || compressed[3] != 0x01)
        return -1;
    int nx = compressed[4], ny = compressed[5], nz = compressed[6];
    if (nx <= 0 || ny <= 0 || nz <= 0) return -1;
    return nx * ny * nz;
}

int c3d_decompress_shard_chunk(const uint8_t *compressed, size_t compressed_size,
                                int chunk_index, uint8_t *output) {
    if (!compressed || !output) return -1;
    if (compressed_size < SHARD_HEADER_SIZE) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3' || compressed[2] != 'S' || compressed[3] != 0x01)
        return -1;

    int nx = compressed[4], ny = compressed[5], nz = compressed[6];
    int nchunks = nx * ny * nz;
    if (nchunks <= 0 || chunk_index < 0 || chunk_index >= nchunks)
        return -1;

    const uint8_t *p = compressed + SHARD_HEADER_SIZE;
    size_t remaining = compressed_size - SHARD_HEADER_SIZE;
    size_t meta_size = nchunks * (sizeof(float) + 4);
    if (remaining < meta_size) return -1;

    float *deltas = (float *)malloc(nchunks * sizeof(float));
    uint32_t *sizes = (uint32_t *)malloc(nchunks * sizeof(uint32_t));
    if (!deltas || !sizes) { free(deltas); free(sizes); return -1; }
    for (int i = 0; i < nchunks; i++) {
        memcpy(&deltas[i], p + i * sizeof(float), sizeof(float));
    }
    p += nchunks * sizeof(float);

    for (int i = 0; i < nchunks; i++) {
        sizes[i] = p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        p += 4;
    }
    remaining -= meta_size;

    size_t offset = 0;
    for (int i = 0; i < chunk_index; i++) {
        if (sizes[i] > remaining - offset) {
            free(deltas); free(sizes);
            return -1;
        }
        offset += sizes[i];
    }
    if (sizes[chunk_index] > remaining - offset) {
        free(deltas); free(sizes);
        return -1;
    }

    int ret = c3d_decompress_to(p + offset, sizes[chunk_index], output);
    if (ret != 0) {
        free(deltas); free(sizes);
        return -1;
    }

    /* Reconstruct means from deltas up to chunk_index */
    float *means = (float *)malloc(nchunks * sizeof(float));
    if (!means) { free(deltas); free(sizes); return -1; }
    for (int cz = 0; cz < nz; cz++)
        for (int cy = 0; cy < ny; cy++)
            for (int cx = 0; cx < nx; cx++) {
                int idx = cz * ny * nx + cy * nx + cx;
                int pred_idx = shard_pred_index(cx, cy, cz, nx, ny);
                float predicted = (pred_idx >= 0) ? means[pred_idx] : 128.0f;
                means[idx] = predicted + deltas[idx];
                if (idx == chunk_index) goto means_done;
            }
means_done:;

    {
        int ci = chunk_index;
        int czi = ci / (ny * nx), cyi = (ci / nx) % ny, cxi = ci % nx;
        int pi = shard_pred_index(cxi, cyi, czi, nx, ny);
        float predicted = (pi >= 0) ? means[pi] : 128.0f;
        float shift = means[ci] - predicted;

        for (int j = 0; j < N3; j++) {
            float v = (float)output[j] + shift;
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            output[j] = (uint8_t)roundf(v);
        }
    }

    free(deltas); free(sizes); free(means);
    return 0;
}


/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 8: Batch parallel compression/decompression
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const uint8_t **inputs;
    c3d_compressed_t *outputs;
    int quality;
    int count;
    volatile int next_item;
    volatile int error;
} batch_compress_ctx_t;

typedef struct {
    const c3d_compressed_t *compressed;
    uint8_t **outputs;
    int count;
    volatile int next_item;
    volatile int error;
} batch_decompress_ctx_t;

static void *batch_compress_worker(void *arg) {
    batch_compress_ctx_t *ctx = (batch_compress_ctx_t *)arg;
    for (;;) {
        int idx = __atomic_fetch_add(&ctx->next_item, 1, __ATOMIC_SEQ_CST);
        if (idx >= ctx->count) break;
        ctx->outputs[idx] = c3d_compress(ctx->inputs[idx], ctx->quality);
        if (ctx->outputs[idx].data == NULL)
            __atomic_store_n(&ctx->error, 1, __ATOMIC_SEQ_CST);
    }
    return NULL;
}

static void *batch_decompress_worker(void *arg) {
    batch_decompress_ctx_t *ctx = (batch_decompress_ctx_t *)arg;
    for (;;) {
        int idx = __atomic_fetch_add(&ctx->next_item, 1, __ATOMIC_SEQ_CST);
        if (idx >= ctx->count) break;
        int ret = c3d_decompress(ctx->compressed[idx].data,
                                 ctx->compressed[idx].size,
                                 ctx->outputs[idx]);
        if (ret != 0)
            __atomic_store_n(&ctx->error, 1, __ATOMIC_SEQ_CST);
    }
    return NULL;
}

static int get_num_threads(int num_threads, int count) {
    if (num_threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        num_threads = (n > 0) ? (int)n : 1;
    }
    if (num_threads > count) num_threads = count;
    return num_threads;
}

int c3d_compress_batch(const uint8_t **inputs, int count, int quality,
                       int num_threads, c3d_compressed_t *outputs) {
    if (!inputs || !outputs) return -1;
    if (count <= 0) return 0;
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;
    init_tables();

    num_threads = get_num_threads(num_threads, count);

    batch_compress_ctx_t ctx = {
        .inputs = inputs,
        .outputs = outputs,
        .quality = quality,
        .count = count,
        .next_item = 0,
        .error = 0,
    };

    pthread_t *threads = (pthread_t *)malloc((num_threads - 1) * sizeof(pthread_t));
    if (!threads && num_threads > 1) return -1;
    int created = 0;
    for (int i = 0; i < num_threads - 1; i++) {
        if (pthread_create(&threads[i], NULL, batch_compress_worker, &ctx) != 0) {
            ctx.error = 1;
            break;
        }
        created++;
    }

    batch_compress_worker(&ctx);

    for (int i = 0; i < created; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    return ctx.error ? -1 : 0;
}

int c3d_decompress_batch(const c3d_compressed_t *compressed, int count,
                         int num_threads, uint8_t **outputs) {
    if (!compressed || !outputs) return -1;
    if (count <= 0) return 0;
    init_tables();

    num_threads = get_num_threads(num_threads, count);

    batch_decompress_ctx_t ctx = {
        .compressed = compressed,
        .outputs = outputs,
        .count = count,
        .next_item = 0,
        .error = 0,
    };

    pthread_t *threads = (pthread_t *)malloc((num_threads - 1) * sizeof(pthread_t));
    if (!threads && num_threads > 1) return -1;
    int created = 0;
    for (int i = 0; i < num_threads - 1; i++) {
        if (pthread_create(&threads[i], NULL, batch_decompress_worker, &ctx) != 0) {
            ctx.error = 1;
            break;
        }
        created++;
    }

    batch_decompress_worker(&ctx);

    for (int i = 0; i < created; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    return ctx.error ? -1 : 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 9: 3D SSIM (Structural Similarity Index)
 * ════════════════════════════════════════════════════════════════════════════ */

#define SSIM_WIN 4
#define SSIM_STEP 4
#define SSIM_C1 6.5025   /* (0.01*255)^2 */
#define SSIM_C2 58.5225  /* (0.03*255)^2 */

double c3d_ssim(const uint8_t *original, const uint8_t *reconstructed) {
    if (!original || !reconstructed) return 0.0;
    double total_ssim = 0.0;
    int num_windows = 0;

    for (int z = 0; z <= N - SSIM_WIN; z += SSIM_STEP) {
        for (int y = 0; y <= N - SSIM_WIN; y += SSIM_STEP) {
            for (int x = 0; x <= N - SSIM_WIN; x += SSIM_STEP) {
                double sum_x = 0, sum_y = 0;
                double sum_xx = 0, sum_yy = 0, sum_xy = 0;
                int count = SSIM_WIN * SSIM_WIN * SSIM_WIN;

                for (int dz = 0; dz < SSIM_WIN; dz++)
                    for (int dy = 0; dy < SSIM_WIN; dy++)
                        for (int dx = 0; dx < SSIM_WIN; dx++) {
                            double vx = original[(z+dz)*N*N + (y+dy)*N + (x+dx)];
                            double vy = reconstructed[(z+dz)*N*N + (y+dy)*N + (x+dx)];
                            sum_x += vx;
                            sum_y += vy;
                            sum_xx += vx * vx;
                            sum_yy += vy * vy;
                            sum_xy += vx * vy;
                        }

                double mu_x = sum_x / count;
                double mu_y = sum_y / count;
                double sigma_xx = sum_xx / count - mu_x * mu_x;
                double sigma_yy = sum_yy / count - mu_y * mu_y;
                double sigma_xy = sum_xy / count - mu_x * mu_y;

                double num = (2.0 * mu_x * mu_y + SSIM_C1) * (2.0 * sigma_xy + SSIM_C2);
                double den = (mu_x * mu_x + mu_y * mu_y + SSIM_C1) * (sigma_xx + sigma_yy + SSIM_C2);

                total_ssim += num / den;
                num_windows++;
            }
        }
    }

    return total_ssim / num_windows;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 10: Reusable workspace API
 * ════════════════════════════════════════════════════════════════════════════ */

#define WS_SYM_BUF_INIT 32768
#define WS_RANS_BUF_SIZE 65536

struct c3d_workspace {
    float *vol;
    float *qtable;
    float *dz_table;
    float *bias_table;
    int32_t *quant;
    uint8_t *sym_buf;
    size_t sym_buf_cap;
    uint8_t *rans_buf;
    size_t rans_buf_cap;
    rans_decode_entry_t *dtable;
};

c3d_workspace_t *c3d_workspace_create(void) {
    c3d_workspace_t *ws = (c3d_workspace_t *)calloc(1, sizeof(c3d_workspace_t));
    if (!ws) return NULL;
    ws->vol = (float *)malloc(N3 * sizeof(float));
    ws->qtable = (float *)malloc(N3 * sizeof(float));
    ws->dz_table = (float *)malloc(N3 * sizeof(float));
    ws->bias_table = (float *)malloc(N3 * sizeof(float));
    ws->quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    ws->sym_buf_cap = WS_SYM_BUF_INIT;
    ws->sym_buf = (uint8_t *)malloc(ws->sym_buf_cap);
    ws->rans_buf_cap = WS_RANS_BUF_SIZE;
    ws->rans_buf = (uint8_t *)malloc(ws->rans_buf_cap);
    ws->dtable = (rans_decode_entry_t *)malloc(
        NUM_CTX_GROUPS * RANS_PROB_SCALE * sizeof(rans_decode_entry_t));
    if (!ws->vol || !ws->qtable || !ws->dz_table || !ws->bias_table ||
        !ws->quant || !ws->sym_buf || !ws->rans_buf || !ws->dtable) {
        c3d_workspace_free(ws);
        return NULL;
    }
    return ws;
}

void c3d_workspace_free(c3d_workspace_t *ws) {
    if (!ws) return;
    free(ws->vol);
    free(ws->qtable);
    free(ws->dz_table);
    free(ws->bias_table);
    free(ws->quant);
    free(ws->sym_buf);
    free(ws->rans_buf);
    free(ws->dtable);
    free(ws);
}

static void sym_init_ws(symstream_t *s, c3d_workspace_t *ws) {
    s->symbols = ws->sym_buf;
    s->cap = (int)ws->sym_buf_cap;
    s->count = 0;
}

static void sym_push_ws(symstream_t *s, uint8_t val, c3d_workspace_t *ws) {
    if (s->count >= s->cap) {
        int new_cap = s->cap * 2;
        uint8_t *tmp = (uint8_t *)realloc(ws->sym_buf, (size_t)new_cap);
        if (!tmp) return;
        s->cap = new_cap;
        ws->sym_buf_cap = new_cap;
        ws->sym_buf = tmp;
        s->symbols = tmp;
    }
    s->symbols[s->count++] = val;
}

static void emit_vlq_ws(symstream_t *s, uint32_t folded, c3d_workspace_t *ws) {
    if (folded < 512) {
        const vlq_entry_t *e = &vlq_table[folded];
        for (int i = 0; i < e->len; i++)
            sym_push_ws(s, e->bytes[i], ws);
        return;
    }
    uint32_t v = folded;
    sym_push_ws(s, 0x80 | (uint8_t)(v & 0x7F), ws);
    v >>= 7;
    while (v >= 128) {
        sym_push_ws(s, 0x80 | (uint8_t)(v & 0x7F), ws);
        v >>= 7;
    }
    sym_push_ws(s, (uint8_t)v, ws);
}

static void coeffs_to_symbols_ws(const int32_t *qdata, symstream_t *s, c3d_workspace_t *ws) {
    init_vlq_table();
    int i = 0;
    while (i < N3) {
        if (qdata[zigzag_order[i]] == 0) {
            int run = 0;
            while (i < N3 && qdata[zigzag_order[i]] == 0) { run++; i++; }
            /* All zero runs use RLE escape */
            while (run > 0) {
                int r = (run > 256) ? 256 : run;
                sym_push_ws(s, 0x00, ws);
                sym_push_ws(s, (uint8_t)(r - 1), ws);
                run -= r;
            }
        } else {
            int32_t val = qdata[zigzag_order[i]];
            uint32_t folded = (val > 0) ? (uint32_t)(2 * val) : (uint32_t)(-2 * val - 1);
            if (folded < 512) {
                const vlq_entry_t *e = &vlq_table[folded];
                for (int j = 0; j < e->len; j++)
                    sym_push_ws(s, e->bytes[j], ws);
            } else {
                emit_vlq_ws(s, folded, ws);
            }
            i++;
        }
    }
}

static void ws_ensure_rans_buf(c3d_workspace_t *ws, size_t needed) {
    if (needed > ws->rans_buf_cap) {
        uint8_t *tmp = (uint8_t *)realloc(ws->rans_buf, needed);
        if (!tmp) return;
        ws->rans_buf_cap = needed;
        ws->rans_buf = tmp;
    }
}

size_t c3d_compress_ws(const uint8_t *input, int quality, uint8_t *output, size_t output_cap, c3d_workspace_t *ws) {
    if (!input || !output || !ws) return 0;
    init_tables();
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;

    for (int i = 0; i < N3; i++)
        ws->vol[i] = (float)input[i] - 128.0f;

    dct3d_forward_all(ws->vol);

    float step = quality_to_step(quality);
    compute_quant_table(step, ws->qtable, ws->dz_table, ws->bias_table);
    quantize_volume(ws->vol, ws->quant, ws->qtable, ws->dz_table, ws->bias_table);
    rd_optimize_coeffs(ws->vol, ws->quant, ws->qtable, step * step * 0.8f);

    coeff_predict_forward(ws->quant);

    symstream_t syms;
    sym_init_ws(&syms, ws);
    coeffs_to_symbols_ws(ws->quant, &syms, ws);

    size_t rans_needed = (size_t)syms.count * 4 + 512;
    ws_ensure_rans_buf(ws, rans_needed);

    size_t total = encode_and_pack(&syms, quality, C3D_FLAG_COEFF_PRED,
                                   output, output_cap, ws->rans_buf, ws->rans_buf_cap);
    return total;
}

int c3d_decompress_ws(const uint8_t *compressed, size_t compressed_size, uint8_t *output, c3d_workspace_t *ws) {
    if (!compressed || !output || !ws) return -1;
    init_tables();
    if (compressed_size < 8) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3' || compressed[2] != 'D' || compressed[3] != 0x04)
        return -1;

    int quality = compressed[4];
    uint32_t sym_len = compressed[8] | ((uint32_t)compressed[9]<<8)
                     | ((uint32_t)compressed[10]<<16) | ((uint32_t)compressed[11]<<24);

    if (sym_len > (uint32_t)N3 * 6) return -1;

    if (sym_len > ws->sym_buf_cap) {
        ws->sym_buf_cap = sym_len;
        ws->sym_buf = (uint8_t *)realloc(ws->sym_buf, ws->sym_buf_cap);
    }

    if (decode_symbols_ws(compressed, compressed_size, ws->sym_buf, sym_len, ws->dtable) != 0)
        return -1;

    if (symbols_to_coeffs(ws->sym_buf, (int)sym_len, ws->quant) != 0)
        return -1;

    if (compressed[7] & C3D_FLAG_COEFF_PRED)
        coeff_predict_inverse(ws->quant);

    float step = quality_to_step(quality);
    compute_dequant_table(step, ws->qtable);
    dequantize_volume(ws->quant, ws->vol, ws->qtable);

    dct3d_inverse_all(ws->vol);

    float_to_u8_biased(ws->vol, output, N3);

    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION 10: CDF 5/3 (Le Gall) Wavelet Transform
 * ════════════════════════════════════════════════════════════════════════════ */

#define WAVELET_LEVELS 4

static void cdf53_forward_1d(int32_t *data, int32_t *tmp, int len, int stride) {
    int half = len / 2;
    int32_t *even = tmp;
    int32_t *odd = tmp + half;

#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
    if (stride == 1 && half >= 4) {
        /* Deinterleave: split data into even/odd samples */
        int i = 0;
#ifdef __aarch64__
        for (; i + 3 < half; i += 4) {
            int32x4x2_t pair = vld2q_s32(data + 2 * i);
            vst1q_s32(even + i, pair.val[0]);
            vst1q_s32(odd + i, pair.val[1]);
        }
#else
        for (; i + 3 < half; i += 4) {
            __m128i a0 = _mm_loadu_si128((const __m128i *)(data + 2 * i));
            __m128i a1 = _mm_loadu_si128((const __m128i *)(data + 2 * i + 4));
            /* deinterleave: a0 = [e0,o0,e1,o1], a1 = [e2,o2,e3,o3] */
            __m128i e = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(a0), _mm_castsi128_ps(a1), _MM_SHUFFLE(2,0,2,0)));
            __m128i o = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(a0), _mm_castsi128_ps(a1), _MM_SHUFFLE(3,1,3,1)));
            _mm_storeu_si128((__m128i *)(even + i), e);
            _mm_storeu_si128((__m128i *)(odd + i), o);
        }
#endif
        for (; i < half; i++) {
            even[i] = data[2 * i];
            odd[i]  = data[2 * i + 1];
        }

        /* Predict: odd[i] -= (even[i] + even[i+1]) >> 1 */
        i = 0;
#ifdef __aarch64__
        for (; i + 3 < half - 1; i += 4) {
            int32x4_t ve = vld1q_s32(even + i);
            int32x4_t ve_next = vld1q_s32(even + i + 1);
            int32x4_t vo = vld1q_s32(odd + i);
            int32x4_t sum = vaddq_s32(ve, ve_next);
            sum = vshrq_n_s32(sum, 1);
            vst1q_s32(odd + i, vsubq_s32(vo, sum));
        }
#else
        for (; i + 3 < half - 1; i += 4) {
            __m128i ve = _mm_loadu_si128((const __m128i *)(even + i));
            __m128i ve_next = _mm_loadu_si128((const __m128i *)(even + i + 1));
            __m128i vo = _mm_loadu_si128((const __m128i *)(odd + i));
            __m128i sum = _mm_add_epi32(ve, ve_next);
            sum = _mm_srai_epi32(sum, 1);
            _mm_storeu_si128((__m128i *)(odd + i), _mm_sub_epi32(vo, sum));
        }
#endif
        for (; i < half; i++) {
            int32_t e_next = (i + 1 < half) ? even[i + 1] : even[i];
            odd[i] = odd[i] - ((even[i] + e_next) >> 1);
        }

        /* Update: even[i] += (odd[i-1] + odd[i] + 2) >> 2 */
        /* i=0 is a boundary case (odd[-1] = odd[0]) */
        even[0] = even[0] + ((odd[0] + odd[0] + 2) >> 2);
        i = 1;
#ifdef __aarch64__
        int32x4_t vtwo = vdupq_n_s32(2);
        for (; i + 3 < half; i += 4) {
            int32x4_t vd_prev = vld1q_s32(odd + i - 1);
            int32x4_t vd = vld1q_s32(odd + i);
            int32x4_t ve = vld1q_s32(even + i);
            int32x4_t sum = vaddq_s32(vaddq_s32(vd_prev, vd), vtwo);
            sum = vshrq_n_s32(sum, 2);
            vst1q_s32(even + i, vaddq_s32(ve, sum));
        }
#else
        __m128i vtwo = _mm_set1_epi32(2);
        for (; i + 3 < half; i += 4) {
            __m128i vd_prev = _mm_loadu_si128((const __m128i *)(odd + i - 1));
            __m128i vd = _mm_loadu_si128((const __m128i *)(odd + i));
            __m128i ve = _mm_loadu_si128((const __m128i *)(even + i));
            __m128i sum = _mm_add_epi32(_mm_add_epi32(vd_prev, vd), vtwo);
            sum = _mm_srai_epi32(sum, 2);
            _mm_storeu_si128((__m128i *)(even + i), _mm_add_epi32(ve, sum));
        }
#endif
        for (; i < half; i++) {
            int32_t d_prev = odd[i - 1];
            even[i] = even[i] + ((d_prev + odd[i] + 2) >> 2);
        }

        /* Write back */
        memcpy(data, even, half * sizeof(int32_t));
        memcpy(data + half, odd, half * sizeof(int32_t));
        return;
    }
#endif

    for (int i = 0; i < half; i++) {
        even[i] = data[2 * i * stride];
        odd[i]  = data[(2 * i + 1) * stride];
    }
    for (int i = 0; i < half; i++) {
        int32_t e_next = (i + 1 < half) ? even[i + 1] : even[i];
        odd[i] = odd[i] - ((even[i] + e_next) >> 1);
    }
    for (int i = 0; i < half; i++) {
        int32_t d_prev = (i > 0) ? odd[i - 1] : odd[0];
        even[i] = even[i] + ((d_prev + odd[i] + 2) >> 2);
    }
    for (int i = 0; i < half; i++)
        data[i * stride] = even[i];
    for (int i = 0; i < half; i++)
        data[(half + i) * stride] = odd[i];
}

static void cdf53_inverse_1d(int32_t *data, int32_t *tmp, int len, int stride) {
    int half = len / 2;
    int32_t *even = tmp;
    int32_t *odd = tmp + half;

#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
    if (stride == 1 && half >= 4) {
        memcpy(even, data, half * sizeof(int32_t));
        memcpy(odd, data + half, half * sizeof(int32_t));

        /* Inverse update: even[i] -= (odd[i-1] + odd[i] + 2) >> 2 */
        even[0] = even[0] - ((odd[0] + odd[0] + 2) >> 2);
        int i = 1;
#ifdef __aarch64__
        int32x4_t vtwo_inv = vdupq_n_s32(2);
        for (; i + 3 < half; i += 4) {
            int32x4_t vd_prev = vld1q_s32(odd + i - 1);
            int32x4_t vd = vld1q_s32(odd + i);
            int32x4_t ve = vld1q_s32(even + i);
            int32x4_t sum = vaddq_s32(vaddq_s32(vd_prev, vd), vtwo_inv);
            sum = vshrq_n_s32(sum, 2);
            vst1q_s32(even + i, vsubq_s32(ve, sum));
        }
#else
        __m128i vtwo_inv = _mm_set1_epi32(2);
        for (; i + 3 < half; i += 4) {
            __m128i vd_prev = _mm_loadu_si128((const __m128i *)(odd + i - 1));
            __m128i vd = _mm_loadu_si128((const __m128i *)(odd + i));
            __m128i ve = _mm_loadu_si128((const __m128i *)(even + i));
            __m128i sum = _mm_add_epi32(_mm_add_epi32(vd_prev, vd), vtwo_inv);
            sum = _mm_srai_epi32(sum, 2);
            _mm_storeu_si128((__m128i *)(even + i), _mm_sub_epi32(ve, sum));
        }
#endif
        for (; i < half; i++) {
            int32_t d_prev = odd[i - 1];
            even[i] = even[i] - ((d_prev + odd[i] + 2) >> 2);
        }

        /* Inverse predict: odd[i] += (even[i] + even[i+1]) >> 1 */
        i = 0;
#ifdef __aarch64__
        for (; i + 3 < half - 1; i += 4) {
            int32x4_t ve = vld1q_s32(even + i);
            int32x4_t ve_next = vld1q_s32(even + i + 1);
            int32x4_t vo = vld1q_s32(odd + i);
            int32x4_t sum = vaddq_s32(ve, ve_next);
            sum = vshrq_n_s32(sum, 1);
            vst1q_s32(odd + i, vaddq_s32(vo, sum));
        }
#else
        for (; i + 3 < half - 1; i += 4) {
            __m128i ve = _mm_loadu_si128((const __m128i *)(even + i));
            __m128i ve_next = _mm_loadu_si128((const __m128i *)(even + i + 1));
            __m128i vo = _mm_loadu_si128((const __m128i *)(odd + i));
            __m128i sum = _mm_add_epi32(ve, ve_next);
            sum = _mm_srai_epi32(sum, 1);
            _mm_storeu_si128((__m128i *)(odd + i), _mm_add_epi32(vo, sum));
        }
#endif
        for (; i < half; i++) {
            int32_t e_next = (i + 1 < half) ? even[i + 1] : even[i];
            odd[i] = odd[i] + ((even[i] + e_next) >> 1);
        }

        /* Interleave back */
        i = 0;
#ifdef __aarch64__
        for (; i + 3 < half; i += 4) {
            int32x4x2_t pair;
            pair.val[0] = vld1q_s32(even + i);
            pair.val[1] = vld1q_s32(odd + i);
            vst2q_s32(data + 2 * i, pair);
        }
#else
        for (; i + 3 < half; i += 4) {
            __m128i e = _mm_loadu_si128((const __m128i *)(even + i));
            __m128i o = _mm_loadu_si128((const __m128i *)(odd + i));
            __m128i lo = _mm_unpacklo_epi32(e, o);
            __m128i hi = _mm_unpackhi_epi32(e, o);
            _mm_storeu_si128((__m128i *)(data + 2 * i), lo);
            _mm_storeu_si128((__m128i *)(data + 2 * i + 4), hi);
        }
#endif
        for (; i < half; i++) {
            data[2 * i]     = even[i];
            data[2 * i + 1] = odd[i];
        }
        return;
    }
#endif

    for (int i = 0; i < half; i++) {
        even[i] = data[i * stride];
        odd[i]  = data[(half + i) * stride];
    }
    for (int i = 0; i < half; i++) {
        int32_t d_prev = (i > 0) ? odd[i - 1] : odd[0];
        even[i] = even[i] - ((d_prev + odd[i] + 2) >> 2);
    }
    for (int i = 0; i < half; i++) {
        int32_t e_next = (i + 1 < half) ? even[i + 1] : even[i];
        odd[i] = odd[i] + ((even[i] + e_next) >> 1);
    }
    for (int i = 0; i < half; i++) {
        data[2 * i * stride]       = even[i];
        data[(2 * i + 1) * stride] = odd[i];
    }
}

static void wavelet3d_forward_axis(int32_t *vol, int axis, int size) {
    int stride = (axis == 0) ? 1 : (axis == 1) ? N : N * N;
    int32_t tmp[N];
    for (int a = 0; a < size; a++)
        for (int b = 0; b < size; b++) {
            int base;
            if (axis == 0)      base = a * N * N + b * N;
            else if (axis == 1) base = a * N * N + b;
            else                base = a * N + b;
            cdf53_forward_1d(vol + base, tmp, size, stride);
        }
}

static void wavelet3d_inverse_axis(int32_t *vol, int axis, int size) {
    int stride = (axis == 0) ? 1 : (axis == 1) ? N : N * N;
    int32_t tmp[N];
    for (int a = 0; a < size; a++)
        for (int b = 0; b < size; b++) {
            int base;
            if (axis == 0)      base = a * N * N + b * N;
            else if (axis == 1) base = a * N * N + b;
            else                base = a * N + b;
            cdf53_inverse_1d(vol + base, tmp, size, stride);
        }
}

static void wavelet3d_forward(int32_t *vol) {
    int size = N;
    for (int level = 0; level < WAVELET_LEVELS; level++) {
        wavelet3d_forward_axis(vol, 0, size);
        wavelet3d_forward_axis(vol, 1, size);
        wavelet3d_forward_axis(vol, 2, size);
        size /= 2;
    }
}

static void wavelet3d_inverse(int32_t *vol) {
    int size = N >> (WAVELET_LEVELS - 1);
    for (int level = WAVELET_LEVELS - 1; level >= 0; level--) {
        wavelet3d_inverse_axis(vol, 2, size);
        wavelet3d_inverse_axis(vol, 1, size);
        wavelet3d_inverse_axis(vol, 0, size);
        size *= 2;
    }
}

/* Determine the subband level for a single coordinate.
 * Returns 0 for finest detail (level-0 high-pass, positions [N/2..N-1]),
 * up to WAVELET_LEVELS for the coarsest approximation band. */
static int wavelet_subband_level(int coord) {
    int size = N;
    for (int level = 0; level < WAVELET_LEVELS; level++) {
        int half = size / 2;
        if (coord >= half)
            return level;
        size = half;
    }
    return WAVELET_LEVELS; /* coarsest approximation */
}

/* Build a per-coefficient quantization step table for the wavelet domain. */
static void wavelet_compute_qtable(float base_step, float *qtable) {
    /* Aggressive scale factors for low quality (q=1, base_step=50) */
    static const float lo_scale[WAVELET_LEVELS + 1] = {
        4.0f, 2.0f, 1.5f, 1.0f, 0.5f
    };

    /* Quality-dependent spread: as base_step shrinks (high quality), narrow
     * toward uniform 1.0 so that fine detail subbands aren't over-quantized.
     * base_step ranges from 50.0 (q=1) to 0.5 (q=100).  We use a normalized
     * quality factor t in [0,1] where t=0 means low quality, t=1 means high. */
    float t = 1.0f - logf(base_step / 0.5f) / logf(50.0f / 0.5f);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    /* Sharpen the curve so spread narrows faster at high quality */
    t = t * t;

    float level_scale[WAVELET_LEVELS + 1];
    for (int i = 0; i <= WAVELET_LEVELS; i++)
        level_scale[i] = 1.0f + (lo_scale[i] - 1.0f) * (1.0f - t);

    for (int z = 0; z < N; z++) {
        int lz = wavelet_subband_level(z);
        for (int y = 0; y < N; y++) {
            int ly = wavelet_subband_level(y);
            for (int x = 0; x < N; x++) {
                int lx = wavelet_subband_level(x);
                int idx = z * N * N + y * N + x;
                /* Use the finest (lowest-numbered) subband among the 3 axes */
                int min_level = lx < ly ? lx : ly;
                if (lz < min_level) min_level = lz;
                qtable[idx] = base_step * level_scale[min_level];
            }
        }
    }
}

static void wavelet_quantize(const int32_t *coeffs, int32_t *quant, const float *qtable) {
    for (int z = 0; z < N; z++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int i = z * N * N + y * N + x;
                float qstep = qtable[i];
                float v = (float)coeffs[i] / qstep;
                float dz = (qstep > 1.0f) ? 0.5f : 0.3f;
                float bias = 1.0f - dz;
                if (v > dz)
                    quant[i] = (int32_t)(v + bias);
                else if (v < -dz)
                    quant[i] = (int32_t)(v - bias);
                else
                    quant[i] = 0;
            }
}

static void wavelet_dequantize(const int32_t *quant, int32_t *coeffs, const float *qtable) {
    for (int i = 0; i < N3; i++)
        coeffs[i] = (int32_t)(quant[i] * qtable[i] + (quant[i] > 0 ? 0.5f : (quant[i] < 0 ? -0.5f : 0.0f)));
}

/* RD optimization adapted for wavelet coefficients (int32_t originals). */
static void wavelet_rd_optimize(const int32_t *coeffs, int32_t *quant,
                                const float *qtable, float lambda) {
    int trailing_zeros = 0;
    for (int zi = N3 - 1; zi >= 0; zi--) {
        if (quant[zigzag_order[zi]] == 0)
            trailing_zeros++;
        else
            break;
    }

    int last_nz = N3 - 1 - trailing_zeros;
    for (int zi = last_nz; zi >= 0; zi--) {
        int i = zigzag_order[zi];
        int32_t q = quant[i];
        float orig = (float)coeffs[i];
        float qstep = qtable[i];

        /* Subband-aware lambda: scale by relative step size */
        float freq_lambda = lambda * (qstep / qtable[0]);

        int next_is_zero = (zi >= last_nz) ? 1 :
                           (quant[zigzag_order[zi + 1]] == 0);

        /* Option A: keep current value */
        float recon_a = q * qstep;
        float dist_a = (orig - recon_a) * (orig - recon_a);
        float rate_a = vlq_byte_cost(q) * 8.0f;
        if (next_is_zero && q != 0)
            rate_a += 16.0f;
        float cost_a = dist_a + freq_lambda * rate_a;

        /* Option B: alternative rounding */
        int32_t q_alt;
        if (q == 0) {
            q_alt = (orig >= 0) ? 1 : -1;
        } else {
            float recon_up = (q + 1) * qstep;
            float recon_dn = (q - 1) * qstep;
            float d_up = (orig - recon_up) * (orig - recon_up);
            float d_dn = (orig - recon_dn) * (orig - recon_dn);
            q_alt = (d_up < d_dn) ? q + 1 : q - 1;
        }
        float recon_b = q_alt * qstep;
        float dist_b = (orig - recon_b) * (orig - recon_b);
        float rate_b;
        if (q_alt == 0) {
            rate_b = 0.0f;
        } else {
            rate_b = vlq_byte_cost(q_alt) * 8.0f;
            if (next_is_zero)
                rate_b += 16.0f;
        }
        float cost_b = dist_b + freq_lambda * rate_b;

        /* Option C: set to zero */
        float dist_c = orig * orig;
        float rate_c = 0.0f;
        float cost_c = dist_c + freq_lambda * rate_c;

        if (cost_c <= cost_a && cost_c <= cost_b) {
            quant[i] = 0;
        } else if (cost_b < cost_a) {
            quant[i] = q_alt;
            if (q == 0) last_nz = (zi > last_nz) ? zi : last_nz;
        }
    }
}

size_t c3d_compress_wavelet_to(const uint8_t *input, int quality, uint8_t *output, size_t output_cap) {
    if (!input || !output) return 0;
    init_tables();
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;

    int32_t *vol = (int32_t *)malloc(N3 * sizeof(int32_t));
    float *qtable = (float *)malloc(N3 * sizeof(float));
    int32_t *quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    if (!vol || !qtable || !quant) { free(vol); free(qtable); free(quant); return 0; }
    for (int i = 0; i < N3; i++)
        vol[i] = (int32_t)input[i] - 128;

    wavelet3d_forward(vol);

    float step = quality_to_step(quality);
    wavelet_compute_qtable(step, qtable);
    wavelet_quantize(vol, quant, qtable);

    /* RD-optimized thresholding of small coefficients */
    wavelet_rd_optimize(vol, quant, qtable, step * step * 0.15f);

    /* Coefficient prediction (delta coding on low-frequency band) */
    coeff_predict_forward(quant);

    /* VLQ+RLE symbol encoding */
    symstream_t syms;
    sym_init(&syms);
    coeffs_to_symbols(quant, &syms);

    uint8_t flags = C3D_FLAG_COEFF_PRED;
    size_t total = encode_and_pack(&syms, quality, flags, output, output_cap, NULL, 0);
    if (total > 0) output[5] = C3D_TRANSFORM_WAVELET;

    free(vol); free(quant); free(qtable); free(syms.symbols);
    return total;
}

c3d_compressed_t c3d_compress_wavelet(const uint8_t *input, int quality) {
    if (!input) return (c3d_compressed_t){ .data = NULL, .size = 0 };
    size_t bound = c3d_compress_bound();
    uint8_t *buf = (uint8_t *)malloc(bound);
    if (!buf) return (c3d_compressed_t){ .data = NULL, .size = 0 };

    size_t actual = c3d_compress_wavelet_to(input, quality, buf, bound);
    if (actual == 0) {
        free(buf);
        return (c3d_compressed_t){ .data = NULL, .size = 0 };
    }

    buf = (uint8_t *)realloc(buf, actual);
    return (c3d_compressed_t){ .data = buf, .size = actual };
}

static int c3d_decompress_wavelet_internal(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    init_tables();

    int quality = compressed[4];

    uint32_t sym_len = compressed[8] | ((uint32_t)compressed[9]<<8)
                     | ((uint32_t)compressed[10]<<16) | ((uint32_t)compressed[11]<<24);

    if (sym_len > (uint32_t)N3 * 6) return -1;

    uint8_t *syms = (uint8_t *)malloc(sym_len > 0 ? sym_len : 1);
    if (!syms) return -1;

    if (decode_symbols(compressed, compressed_size, syms, sym_len) != 0) {
        free(syms);
        return -1;
    }

    /* VLQ decode -> quantized coefficients */
    int32_t *quant = (int32_t *)malloc(N3 * sizeof(int32_t));
    if (!quant) { free(syms); return -1; }
    if (symbols_to_coeffs(syms, (int)sym_len, quant) != 0) {
        free(syms); free(quant);
        return -1;
    }

    /* Undo coefficient prediction if enabled */
    if (compressed[7] & C3D_FLAG_COEFF_PRED)
        coeff_predict_inverse(quant);

    float step = quality_to_step(quality);
    float *qtable = (float *)malloc(N3 * sizeof(float));
    int32_t *vol = (int32_t *)malloc(N3 * sizeof(int32_t));
    if (!qtable || !vol) { free(syms); free(quant); free(qtable); free(vol); return -1; }
    wavelet_compute_qtable(step, qtable);
    wavelet_dequantize(quant, vol, qtable);
    free(qtable);

    wavelet3d_inverse(vol);

    for (int i = 0; i < N3; i++) {
        int32_t v = vol[i] + 128;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        output[i] = (uint8_t)v;
    }

    free(syms); free(quant); free(vol);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Automatic transform selection: DCT vs wavelet
 * ════════════════════════════════════════════════════════════════════════════ */

static float compute_edge_strength(const uint8_t *input) {
    uint64_t grad_sum = 0;
    int count = 0;
    for (int z = 0; z < N; z += 2)
        for (int y = 0; y < N; y += 2)
            for (int x = 0; x < N; x += 2) {
                int idx = z * N * N + y * N + x;
                int gx = 0, gy = 0, gz = 0;
                if (x + 1 < N) gx = (int)input[idx + 1] - (int)input[idx];
                if (y + 1 < N) gy = (int)input[idx + N] - (int)input[idx];
                if (z + 1 < N) gz = (int)input[idx + N * N] - (int)input[idx];
                grad_sum += (unsigned)(gx < 0 ? -gx : gx)
                          + (unsigned)(gy < 0 ? -gy : gy)
                          + (unsigned)(gz < 0 ? -gz : gz);
                count++;
            }
    return (float)grad_sum / (float)count;
}

#define C3D_EDGE_THRESHOLD 12.0f

c3d_compressed_t c3d_compress_auto(const uint8_t *input, int quality) {
    if (!input) return (c3d_compressed_t){ .data = NULL, .size = 0 };
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;
    if (quality == 101)
        return c3d_compress(input, quality);

    float edge = compute_edge_strength(input);

    /* Fast path: very low edge strength means DCT almost certainly wins */
    if (edge < 4.0f)
        return c3d_compress(input, quality);

    /* Fast path: very high edge strength means wavelet almost certainly wins */
    if (edge > 30.0f)
        return c3d_compress_wavelet(input, quality);

    /* Middle range: trial both transforms, keep whichever is smaller */
    c3d_compressed_t dct = c3d_compress(input, quality);
    c3d_compressed_t wav = c3d_compress_wavelet(input, quality);

    if (wav.size < dct.size) {
        free(dct.data);
        return wav;
    } else {
        free(wav.data);
        return dct;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Auto-quality: binary search for target PSNR / size / SSIM
 * ════════════════════════════════════════════════════════════════════════════ */

static double auto_compute_psnr(const uint8_t *a, const uint8_t *b) {
    double mse = 0;
    for (int i = 0; i < N3; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= N3;
    if (mse < 1e-10) return 999.0;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

c3d_compressed_t c3d_compress_target_psnr(const uint8_t *input, double target_psnr) {
    if (!input) return (c3d_compressed_t){NULL, 0};
    c3d_workspace_t *ws = c3d_workspace_create();
    if (!ws) return (c3d_compressed_t){NULL, 0};

    size_t bound = c3d_compress_bound();
    uint8_t *comp_buf = (uint8_t *)malloc(bound);
    uint8_t *dec_buf = (uint8_t *)malloc(N3);
    if (!comp_buf || !dec_buf) {
        free(comp_buf); free(dec_buf); c3d_workspace_free(ws);
        return (c3d_compressed_t){NULL, 0};
    }

    int lo = 1, hi = 100, best_q = 100;

    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        size_t sz = c3d_compress_ws(input, mid, comp_buf, bound, ws);
        if (sz == 0) { lo = mid + 1; continue; }
        c3d_decompress_ws(comp_buf, sz, dec_buf, ws);
        double psnr = auto_compute_psnr(input, dec_buf);
        if (psnr >= target_psnr - 0.5) {
            best_q = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    size_t best_size = c3d_compress_ws(input, best_q, comp_buf, bound, ws);
    free(dec_buf);
    c3d_workspace_free(ws);

    if (best_size == 0) { free(comp_buf); return (c3d_compressed_t){NULL, 0}; }
    comp_buf = (uint8_t *)realloc(comp_buf, best_size);
    return (c3d_compressed_t){comp_buf, best_size};
}

c3d_compressed_t c3d_compress_target_size(const uint8_t *input, size_t target_size) {
    if (!input) return (c3d_compressed_t){NULL, 0};
    c3d_workspace_t *ws = c3d_workspace_create();
    if (!ws) return (c3d_compressed_t){NULL, 0};

    size_t bound = c3d_compress_bound();
    uint8_t *comp_buf = (uint8_t *)malloc(bound);
    if (!comp_buf) { c3d_workspace_free(ws); return (c3d_compressed_t){NULL, 0}; }

    int lo = 1, hi = 100, best_q = 1;
    size_t best_size = 0;

    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        size_t sz = c3d_compress_ws(input, mid, comp_buf, bound, ws);
        if (sz == 0) { hi = mid - 1; continue; }
        if (sz <= target_size + target_size / 20) {
            best_q = mid;
            best_size = sz;
            if (sz >= target_size - target_size / 20) break;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    best_size = c3d_compress_ws(input, best_q, comp_buf, bound, ws);
    c3d_workspace_free(ws);
    if (best_size == 0) { free(comp_buf); return (c3d_compressed_t){NULL, 0}; }
    comp_buf = (uint8_t *)realloc(comp_buf, best_size);
    return (c3d_compressed_t){comp_buf, best_size};
}

c3d_compressed_t c3d_compress_target_ssim(const uint8_t *input, double target_ssim) {
    if (!input) return (c3d_compressed_t){NULL, 0};
    c3d_workspace_t *ws = c3d_workspace_create();
    if (!ws) return (c3d_compressed_t){NULL, 0};

    size_t bound = c3d_compress_bound();
    uint8_t *comp_buf = (uint8_t *)malloc(bound);
    uint8_t *dec_buf = (uint8_t *)malloc(N3);
    if (!comp_buf || !dec_buf) {
        free(comp_buf); free(dec_buf); c3d_workspace_free(ws);
        return (c3d_compressed_t){NULL, 0};
    }

    int lo = 1, hi = 100, best_q = 100;

    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        size_t sz = c3d_compress_ws(input, mid, comp_buf, bound, ws);
        if (sz == 0) { lo = mid + 1; continue; }
        c3d_decompress_ws(comp_buf, sz, dec_buf, ws);
        double ssim = c3d_ssim(input, dec_buf);
        if (ssim >= target_ssim - 0.005) {
            best_q = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    size_t best_size = c3d_compress_ws(input, best_q, comp_buf, bound, ws);
    free(dec_buf);
    c3d_workspace_free(ws);
    if (best_size == 0) { free(comp_buf); return (c3d_compressed_t){NULL, 0}; }
    comp_buf = (uint8_t *)realloc(comp_buf, best_size);
    return (c3d_compressed_t){comp_buf, best_size};
}

/* ════════════════════════════════════════════════════════════════════════════
 * SECTION: ROI (Region of Interest) coding
 * ════════════════════════════════════════════════════════════════════════════ */

#define C3D_ROI_HEADER 32

c3d_compressed_t c3d_compress_roi(const uint8_t *input, int quality, const c3d_roi_t *roi) {
    c3d_compressed_t fail = { .data = NULL, .size = 0 };

    if (!input || !roi) return fail;
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;
    if (roi->x0 < 0 || roi->y0 < 0 || roi->z0 < 0) return fail;
    if (roi->x1 > N || roi->y1 > N || roi->z1 > N) return fail;
    if (roi->x0 >= roi->x1 || roi->y0 >= roi->y1 || roi->z0 >= roi->z1) return fail;

    c3d_compressed_t bg = c3d_compress(input, quality);
    if (!bg.data) return fail;

    uint8_t *bg_recon = (uint8_t *)malloc(N3);
    if (!bg_recon) { free(bg.data); return fail; }
    if (c3d_decompress(bg.data, bg.size, bg_recon) != 0) {
        free(bg.data); free(bg_recon); return fail;
    }

    uint8_t *residual = (uint8_t *)malloc(N3);
    if (!residual) { free(bg.data); free(bg_recon); return fail; }
    memset(residual, 128, N3);

    for (int z = roi->z0; z < roi->z1; z++)
        for (int y = roi->y0; y < roi->y1; y++)
            for (int x = roi->x0; x < roi->x1; x++) {
                int idx = z * N * N + y * N + x;
                int diff = (int)input[idx] - (int)bg_recon[idx];
                int mapped = diff + 128;
                if (mapped < 0) mapped = 0;
                if (mapped > 255) mapped = 255;
                residual[idx] = (uint8_t)mapped;
            }

    free(bg_recon);

    c3d_compressed_t roi_comp = c3d_compress(residual, roi->roi_quality);
    free(residual);
    if (!roi_comp.data) { free(bg.data); return fail; }

    size_t total = C3D_ROI_HEADER + bg.size + roi_comp.size;
    uint8_t *out = (uint8_t *)malloc(total);
    if (!out) { free(bg.data); free(roi_comp.data); return fail; }

    size_t pos = 0;
    out[pos++] = 'C'; out[pos++] = '3'; out[pos++] = 'R'; out[pos++] = 0x01;
    uint32_t bg_sz = (uint32_t)bg.size;
    out[pos++] = (uint8_t)(bg_sz & 0xFF);
    out[pos++] = (uint8_t)((bg_sz >> 8) & 0xFF);
    out[pos++] = (uint8_t)((bg_sz >> 16) & 0xFF);
    out[pos++] = (uint8_t)((bg_sz >> 24) & 0xFF);
    int roi_params[6] = { roi->x0, roi->y0, roi->z0, roi->x1, roi->y1, roi->z1 };
    for (int i = 0; i < 6; i++) {
        uint32_t v = (uint32_t)roi_params[i];
        out[pos++] = (uint8_t)(v & 0xFF);
        out[pos++] = (uint8_t)((v >> 8) & 0xFF);
        out[pos++] = (uint8_t)((v >> 16) & 0xFF);
        out[pos++] = (uint8_t)((v >> 24) & 0xFF);
    }
    memcpy(out + pos, bg.data, bg.size);
    pos += bg.size;
    memcpy(out + pos, roi_comp.data, roi_comp.size);

    free(bg.data);
    free(roi_comp.data);

    return (c3d_compressed_t){ .data = out, .size = total };
}

static int c3d_decompress_roi(const uint8_t *compressed, size_t compressed_size, uint8_t *output) {
    if (compressed_size < C3D_ROI_HEADER) return -1;
    if (compressed[0] != 'C' || compressed[1] != '3' ||
        compressed[2] != 'R' || compressed[3] != 0x01) return -1;

    uint32_t bg_sz = (uint32_t)compressed[4]
                   | ((uint32_t)compressed[5] << 8)
                   | ((uint32_t)compressed[6] << 16)
                   | ((uint32_t)compressed[7] << 24);

    int roi_p[6];
    for (int i = 0; i < 6; i++) {
        int off = 8 + i * 4;
        roi_p[i] = (int)((uint32_t)compressed[off]
                       | ((uint32_t)compressed[off+1] << 8)
                       | ((uint32_t)compressed[off+2] << 16)
                       | ((uint32_t)compressed[off+3] << 24));
    }
    int x0 = roi_p[0], y0 = roi_p[1], z0 = roi_p[2];
    int x1 = roi_p[3], y1 = roi_p[4], z1 = roi_p[5];

    if (C3D_ROI_HEADER + bg_sz > compressed_size) return -1;

    const uint8_t *bg_blob = compressed + C3D_ROI_HEADER;
    const uint8_t *roi_blob = bg_blob + bg_sz;
    size_t roi_sz = compressed_size - C3D_ROI_HEADER - bg_sz;

    if (c3d_decompress(bg_blob, bg_sz, output) != 0) return -1;

    uint8_t *res_decoded = (uint8_t *)malloc(N3);
    if (!res_decoded) return -1;
    if (c3d_decompress(roi_blob, roi_sz, res_decoded) != 0) {
        free(res_decoded); return -1;
    }

    for (int z = z0; z < z1; z++)
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) {
                int idx = z * N * N + y * N + x;
                int res = (int)res_decoded[idx] - 128;
                int val = (int)output[idx] + res;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                output[idx] = (uint8_t)val;
            }

    free(res_decoded);
    return 0;
}

/* ── Memory-mapped shard access ── */
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

struct c3d_shard_map {
    uint8_t *data;
    size_t   file_size;
    int      nchunks;
};

c3d_shard_map_t *c3d_shard_mmap_open(const char *path) {
    if (!path) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    size_t fsize = (size_t)st.st_size;

    if (fsize < SHARD_HEADER_SIZE) { close(fd); return NULL; }

    uint8_t *mapped = (uint8_t *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return NULL;

    if (mapped[0] != 'C' || mapped[1] != '3' || mapped[2] != 'S' || mapped[3] != 0x01) {
        munmap(mapped, fsize);
        return NULL;
    }

    c3d_shard_map_t *m = (c3d_shard_map_t *)malloc(sizeof(c3d_shard_map_t));
    m->data = mapped;
    m->file_size = fsize;
    m->nchunks = mapped[4] * mapped[5] * mapped[6];
    return m;
}

int c3d_shard_mmap_chunk_count(const c3d_shard_map_t *map) {
    if (!map) return -1;
    return map->nchunks;
}

int c3d_shard_mmap_read_chunk(const c3d_shard_map_t *map, int chunk_index, uint8_t *output) {
    if (!map || !output) return -1;
    return c3d_decompress_shard_chunk(map->data, map->file_size, chunk_index, output);
}

void c3d_shard_mmap_close(c3d_shard_map_t *map) {
    if (!map) return;
    munmap(map->data, map->file_size);
    free(map);
}
#endif /* !_WIN32 */

/* ── Streaming shard writer ── */
#include <stdio.h>

struct c3d_shard_writer {
    FILE    *fp;
    int      nx, ny, nz, quality;
    int      nchunks;
    int      chunks_written;
    float   *means;
    float   *deltas;
    uint32_t *sizes;
    uint8_t **comp_data;
};

c3d_shard_writer_t *c3d_shard_writer_open(const char *path, int nx, int ny, int nz, int quality) {
    if (!path) return NULL;
    if (nx <= 0 || ny <= 0 || nz <= 0 || nx > 255 || ny > 255 || nz > 255)
        return NULL;
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;

    FILE *fp = fopen(path, "wb");
    if (!fp) return NULL;

    int nchunks = nx * ny * nz;
    c3d_shard_writer_t *w = (c3d_shard_writer_t *)calloc(1, sizeof(c3d_shard_writer_t));
    if (!w) { fclose(fp); return NULL; }
    w->fp = fp;
    w->nx = nx; w->ny = ny; w->nz = nz;
    w->quality = quality;
    w->nchunks = nchunks;
    w->chunks_written = 0;
    w->means = (float *)calloc(nchunks, sizeof(float));
    w->deltas = (float *)calloc(nchunks, sizeof(float));
    w->sizes = (uint32_t *)calloc(nchunks, sizeof(uint32_t));
    w->comp_data = (uint8_t **)calloc(nchunks, sizeof(uint8_t *));
    if (!w->means || !w->deltas || !w->sizes || !w->comp_data) {
        free(w->means); free(w->deltas); free(w->sizes); free(w->comp_data);
        fclose(fp); free(w);
        return NULL;
    }
    return w;
}

int c3d_shard_writer_add_chunk(c3d_shard_writer_t *w, const uint8_t *chunk) {
    if (!w || !chunk || w->chunks_written >= w->nchunks) return -1;

    init_tables();

    int idx = w->chunks_written;
    int cz = idx / (w->ny * w->nx);
    int cy = (idx / w->nx) % w->ny;
    int cx = idx % w->nx;

    w->means[idx] = compute_mean(chunk);
    int pred = shard_pred_index(cx, cy, cz, w->nx, w->ny);
    float predicted = (pred >= 0) ? w->means[pred] : 128.0f;
    w->deltas[idx] = w->means[idx] - predicted;
    float shift = w->deltas[idx];

    uint8_t shifted[N3];
    for (int j = 0; j < N3; j++) {
        float v = (float)chunk[j] - shift;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        shifted[j] = (uint8_t)roundf(v);
    }

    size_t bound = c3d_compress_bound();
    uint8_t *buf = (uint8_t *)malloc(bound);
    size_t csz = c3d_compress_to(shifted, w->quality, buf, bound);
    if (csz == 0) { free(buf); return -1; }

    w->comp_data[idx] = buf;
    w->sizes[idx] = (uint32_t)csz;
    w->chunks_written++;
    return 0;
}

int c3d_shard_writer_finish(c3d_shard_writer_t *w) {
    if (!w) return -1;
    int ret = -1;

    if (w->chunks_written != w->nchunks) goto cleanup;

    uint8_t hdr[SHARD_HEADER_SIZE];
    hdr[0] = 'C'; hdr[1] = '3'; hdr[2] = 'S'; hdr[3] = 0x01;
    hdr[4] = (uint8_t)w->nx; hdr[5] = (uint8_t)w->ny; hdr[6] = (uint8_t)w->nz;
    hdr[7] = (uint8_t)w->quality;
    if (fwrite(hdr, 1, SHARD_HEADER_SIZE, w->fp) != SHARD_HEADER_SIZE) goto cleanup;

    for (int i = 0; i < w->nchunks; i++) {
        if (fwrite(&w->deltas[i], sizeof(float), 1, w->fp) != 1) goto cleanup;
    }

    for (int i = 0; i < w->nchunks; i++) {
        uint8_t sb[4];
        sb[0] = w->sizes[i]; sb[1] = w->sizes[i] >> 8;
        sb[2] = w->sizes[i] >> 16; sb[3] = w->sizes[i] >> 24;
        if (fwrite(sb, 1, 4, w->fp) != 4) goto cleanup;
    }

    for (int i = 0; i < w->nchunks; i++) {
        if (fwrite(w->comp_data[i], 1, w->sizes[i], w->fp) != w->sizes[i]) goto cleanup;
    }

    ret = 0;

cleanup:
    fclose(w->fp);
    for (int i = 0; i < w->nchunks; i++)
        free(w->comp_data[i]);
    free(w->means);
    free(w->deltas);
    free(w->sizes);
    free(w->comp_data);
    free(w);
    return ret;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Post-decompression deblocking filter
 * ════════════════════════════════════════════════════════════════════════════ */

void c3d_deblock(uint8_t *volume, int vx, int vy, int vz, float strength) {
    if (!volume || strength <= 0.0f) return;
    if (strength > 1.0f) strength = 1.0f;

    float w1 = 0.30f * strength;
    float w2 = 0.15f * strength;

    /* X boundaries */
    for (int bx = C3D_BLOCK_SIZE; bx < vx; bx += C3D_BLOCK_SIZE) {
        if (bx - 2 < 0 || bx + 1 >= vx) continue;
        for (int z = 0; z < vz; z++) {
            for (int y = 0; y < vy; y++) {
                int base = z * vx * vy + y * vx;
                float v0 = volume[base + bx - 2], v1 = volume[base + bx - 1];
                float v2 = volume[base + bx],     v3 = volume[base + bx + 1];
                int c;
                c = (int)(v0 + w2 * (v2 - v0) + 0.5f); volume[base + bx - 2] = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v1 + w1 * (v2 - v1) + 0.5f); volume[base + bx - 1] = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v2 + w1 * (v1 - v2) + 0.5f); volume[base + bx]     = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v3 + w2 * (v1 - v3) + 0.5f); volume[base + bx + 1] = c < 0 ? 0 : c > 255 ? 255 : c;
            }
        }
    }

    /* Y boundaries */
    for (int by = C3D_BLOCK_SIZE; by < vy; by += C3D_BLOCK_SIZE) {
        if (by - 2 < 0 || by + 1 >= vy) continue;
        for (int z = 0; z < vz; z++) {
            for (int x = 0; x < vx; x++) {
                int zbase = z * vx * vy;
                float v0 = volume[zbase + (by - 2) * vx + x];
                float v1 = volume[zbase + (by - 1) * vx + x];
                float v2 = volume[zbase + by * vx + x];
                float v3 = volume[zbase + (by + 1) * vx + x];
                int c;
                c = (int)(v0 + w2 * (v2 - v0) + 0.5f); volume[zbase + (by - 2) * vx + x] = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v1 + w1 * (v2 - v1) + 0.5f); volume[zbase + (by - 1) * vx + x] = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v2 + w1 * (v1 - v2) + 0.5f); volume[zbase + by * vx + x]        = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v3 + w2 * (v1 - v3) + 0.5f); volume[zbase + (by + 1) * vx + x]  = c < 0 ? 0 : c > 255 ? 255 : c;
            }
        }
    }

    /* Z boundaries */
    for (int bz = C3D_BLOCK_SIZE; bz < vz; bz += C3D_BLOCK_SIZE) {
        if (bz - 2 < 0 || bz + 1 >= vz) continue;
        for (int y = 0; y < vy; y++) {
            for (int x = 0; x < vx; x++) {
                int yx = y * vx + x;
                float v0 = volume[(bz - 2) * vx * vy + yx];
                float v1 = volume[(bz - 1) * vx * vy + yx];
                float v2 = volume[bz * vx * vy + yx];
                float v3 = volume[(bz + 1) * vx * vy + yx];
                int c;
                c = (int)(v0 + w2 * (v2 - v0) + 0.5f); volume[(bz - 2) * vx * vy + yx] = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v1 + w1 * (v2 - v1) + 0.5f); volume[(bz - 1) * vx * vy + yx] = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v2 + w1 * (v1 - v2) + 0.5f); volume[bz * vx * vy + yx]        = c < 0 ? 0 : c > 255 ? 255 : c;
                c = (int)(v3 + w2 * (v1 - v3) + 0.5f); volume[(bz + 1) * vx * vy + yx]  = c < 0 ? 0 : c > 255 ? 255 : c;
            }
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Streaming ring-buffer compression API
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t data[N3];
    int      chunk_index;
    int      occupied;
} c3d_stream_slot_t;

struct c3d_stream {
    c3d_stream_slot_t *slots;
    int                num_slots;
    int                head;
    int                tail;
    int                count;

    int                quality;
    c3d_write_cb       write_cb;
    void              *userdata;

    pthread_t         *threads;
    int                num_threads;
    pthread_mutex_t    mtx;
    pthread_cond_t     not_full;
    pthread_cond_t     not_empty;
    int                shutdown;
    int                error;
    int                next_index;
};

static void *stream_worker(void *arg) {
    c3d_stream_t *s = (c3d_stream_t *)arg;
    c3d_workspace_t *ws = c3d_workspace_create();
    if (!ws) { __sync_fetch_and_or(&s->error, 1); return NULL; }
    size_t bound = c3d_compress_bound();
    uint8_t *comp_buf = (uint8_t *)malloc(bound);
    if (!comp_buf) { c3d_workspace_free(ws); __sync_fetch_and_or(&s->error, 1); return NULL; }

    for (;;) {
        pthread_mutex_lock(&s->mtx);
        while (s->count == 0 && !s->shutdown)
            pthread_cond_wait(&s->not_empty, &s->mtx);
        if (s->count == 0 && s->shutdown) {
            pthread_mutex_unlock(&s->mtx);
            break;
        }
        c3d_stream_slot_t *slot = &s->slots[s->tail];
        int chunk_index = slot->chunk_index;
        uint8_t chunk_copy[N3];
        memcpy(chunk_copy, slot->data, N3);
        slot->occupied = 0;
        s->tail = (s->tail + 1) % s->num_slots;
        s->count--;
        pthread_cond_signal(&s->not_full);
        pthread_mutex_unlock(&s->mtx);

        size_t comp_size = c3d_compress_ws(chunk_copy, s->quality, comp_buf, bound, ws);
        if (comp_size == 0)
            __sync_fetch_and_or(&s->error, 1);
        else
            s->write_cb(comp_buf, comp_size, chunk_index, s->userdata);
    }

    free(comp_buf);
    c3d_workspace_free(ws);
    return NULL;
}

c3d_stream_t *c3d_stream_compress_create(int quality, int num_threads,
                                          c3d_write_cb write_cb, void *userdata) {
    init_tables();
    if (!write_cb) return NULL;
    if (quality < 1) quality = 1;
    if (quality > 101) quality = 101;
    if (num_threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        num_threads = (n > 0) ? (int)n : 1;
    }
    c3d_stream_t *s = (c3d_stream_t *)calloc(1, sizeof(c3d_stream_t));
    if (!s) return NULL;
    s->quality = quality;
    s->write_cb = write_cb;
    s->userdata = userdata;
    s->num_threads = num_threads;
    s->num_slots = 2 * num_threads;
    s->slots = (c3d_stream_slot_t *)calloc(s->num_slots, sizeof(c3d_stream_slot_t));
    if (!s->slots) { free(s); return NULL; }
    pthread_mutex_init(&s->mtx, NULL);
    pthread_cond_init(&s->not_full, NULL);
    pthread_cond_init(&s->not_empty, NULL);
    s->threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (!s->threads) { free(s->slots); free(s); return NULL; }
    int created = 0;
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&s->threads[i], NULL, stream_worker, s) != 0)
            break;
        created++;
    }
    if (created == 0) {
        free(s->threads); free(s->slots); free(s);
        return NULL;
    }
    s->num_threads = created;
    return s;
}

int c3d_stream_push(c3d_stream_t *stream, const uint8_t *chunk) {
    if (!stream || !chunk) return -1;
    pthread_mutex_lock(&stream->mtx);
    while (stream->count == stream->num_slots)
        pthread_cond_wait(&stream->not_full, &stream->mtx);
    c3d_stream_slot_t *slot = &stream->slots[stream->head];
    memcpy(slot->data, chunk, N3);
    slot->chunk_index = stream->next_index++;
    slot->occupied = 1;
    stream->head = (stream->head + 1) % stream->num_slots;
    stream->count++;
    pthread_cond_signal(&stream->not_empty);
    pthread_mutex_unlock(&stream->mtx);
    return 0;
}

int c3d_stream_flush(c3d_stream_t *stream) {
    if (!stream) return -1;
    pthread_mutex_lock(&stream->mtx);
    stream->shutdown = 1;
    pthread_cond_broadcast(&stream->not_empty);
    pthread_mutex_unlock(&stream->mtx);
    for (int i = 0; i < stream->num_threads; i++)
        pthread_join(stream->threads[i], NULL);
    return stream->error ? -1 : 0;
}

void c3d_stream_free(c3d_stream_t *stream) {
    if (!stream) return;
    if (!stream->shutdown)
        c3d_stream_flush(stream);
    pthread_mutex_destroy(&stream->mtx);
    pthread_cond_destroy(&stream->not_full);
    pthread_cond_destroy(&stream->not_empty);
    free(stream->threads);
    free(stream->slots);
    free(stream);
}
