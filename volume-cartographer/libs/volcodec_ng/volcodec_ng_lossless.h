/* volcodec_ng_lossless.h: lossless 128^3 uint8 chunk codec (volcodec-ng, format revision 2, the lossless
 * modes only).
 *
 * One chunk is 128^3 uint8 voxels, z-major, as a zarr v3 store with sharding_indexed keeps them: every
 * chunk decodes on its own. Each voxel is predicted from 12 voxels of the two previous rows and the
 * two previous slices with weights fitted per chunk by least squares (Q12 integers in the stream);
 * the residual is coded with static canonical Huffman codes, one per activity class of the previous
 * rows, into 4 interleaved bit streams. The decoder is integer only, so every platform and build
 * decodes the same voxels; the encoder's fit uses double precision with floating-point contraction
 * turned off, so every build writes the same bytes.
 *
 * Stream: 12-byte header "VCNG", revision (2), mode, then 6 zero bytes; the payload follows.
 *   mode 2: raw chunk (2 MiB), written when coding would not make the chunk smaller;
 *   mode 3: constant chunk, one byte;
 *   mode 5: the lossless Huffman mode described above.
 * Modes 0, 1 and 4 of the volcodec-ng format are not part of this file: vcn_decode returns
 * VCN_ERR_MODE for them. Comments copied from the full codec that mention "mode 4" refer to its
 * predictor, which mode 5 shares.
 *
 * Generated from Chase Hendrick's volcodec_ng.h (sha256 9a06846f3613d5de...ca2d9fd9) by lossless_header.py in
 * the codec source; the function bodies are verbatim copies.
 *
 * MIT License, Copyright (c) 2026 Chase Hendrick (see LICENSE next to this file). */
#ifndef VOLCODEC_NG_LOSSLESS_H
#define VOLCODEC_NG_LOSSLESS_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VCN_DIM 128u
#define VCN_VOXELS 2097152u
#define VCN_HDR 12u
#define VCN_REVISION 2u /* the revision this encoder writes; the decoder reads 1 and 2 */
/* the encoder falls back to a raw chunk, so a stream never exceeds this */
#define VCN_ENCODE_BOUND ((size_t)VCN_HDR + VCN_VOXELS)

#define VCN_MODE_RAW 2u
#define VCN_MODE_CONST 3u
#define VCN_MODE_LLHUF 5u

typedef enum {
  VCN_OK = 0,
  VCN_ERR_ARG = -1,
  VCN_ERR_CAP = -2,
  VCN_ERR_CORRUPT = -3,
  VCN_ERR_NOMEM = -4,
  VCN_ERR_MODE = -5 /* a volcodec-ng chunk in a mode this file does not read (0, 1 or 4) */
} vcn_status;

/* Encode one chunk losslessly into dst (cap bytes; VCN_ENCODE_BOUND always suffices). */
static inline vcn_status vcn_encode_lossless_chunk(const uint8_t *src, uint8_t *dst, size_t cap, size_t *out_n);
/* Decode one chunk of n bytes into dst (dst_n >= VCN_VOXELS). */
static inline vcn_status vcn_decode(const uint8_t *enc, size_t n, uint8_t *dst, size_t dst_n);
/* The chunk mode in a stream's header, or -1 when enc is not a volcodec-ng stream. */
static inline int vcn_chunk_mode(const uint8_t *enc, size_t n) {
  if (!enc || n < VCN_HDR || enc[0] != 'V' || enc[1] != 'C' || enc[2] != 'N' || enc[3] != 'G' || enc[4] < 1 ||
      enc[4] > VCN_REVISION)
    return -1;
  return enc[5];
}
static inline const char *vcn_status_string(vcn_status s) {
  switch (s) {
  case VCN_OK: return "ok";
  case VCN_ERR_ARG: return "bad argument";
  case VCN_ERR_CAP: return "output buffer too small";
  case VCN_ERR_CORRUPT: return "corrupt stream";
  case VCN_ERR_NOMEM: return "out of memory";
  case VCN_ERR_MODE: return "chunk mode not supported (only the lossless modes 2, 3 and 5 are)";
  }
  return "unknown";
}

/* ======================================================================== */
/*                               implementation                             */
/* ======================================================================== */

/* No floating-point contraction in the implementation. The encoder's decisions are float, and a
 * compiler that fuses a * b + c into one multiply-add (one rounding instead of two) changes some of them,
 * so the streams would depend on the flags: GCC fuses by default in its GNU C modes whenever FMA is
 * available (-mfma, -march=native, every aarch64 target); before this region existed, a GCC -mfma or
 * -march=native build of the tests gave a different stream digest. GCC: optimize("fp-contract=off")
 * for every function defined here; Clang: FP_CONTRACT OFF, restored at the end of the header. Not covered:
 * Clang's -ffp-contract=fast (it ignores the pragma; -ffp-contract=fast-honor-pragmas keeps it), and
 * -ffast-math or -Ofast with any compiler. The decoders are integer only and unaffected. */
#if defined(__clang__)
#pragma float_control(push)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

/* Process-wide flags that are set on first use (CPU features, the decoder choice): relaxed atomic loads
 * and stores, so threads that make their first call at the same time do not race on them. */
#if defined(__GNUC__) || defined(__clang__)
#define VCN_LD(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define VCN_ST(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#else
#define VCN_LD(p) (*(p))
#define VCN_ST(p, v) (*(p) = (v))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VCN_INLINE __attribute__((always_inline)) static inline
#else
#define VCN_INLINE static inline
#endif
/* Hot loops are plain C written for auto-vectorization; on x86-64 Linux they are compiled twice
 * (AVX2 and baseline) and picked at load time. Integer results are identical either way.
 * -DVCN_NO_CLONES builds the baseline only. */
#if !defined(VCN_NO_CLONES) && defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(__SANITIZE_ADDRESS__)
#define VCN_CLONES __attribute__((target_clones("avx2", "default")))
/* the same with BMI1 and BMI2 as well (one-micro-op variable shifts), for the bit coders of mode 5 */
#if defined(__clang__)
#define VCN_CLONES_BMI VCN_CLONES
#else
#define VCN_CLONES_BMI __attribute__((target_clones("arch=haswell", "default")))
#endif
#define VCN_KERNELS "auto-avx2"
#else
#define VCN_CLONES
#define VCN_CLONES_BMI
#define VCN_KERNELS "c"
#endif

/* ---- small helpers ---- */
VCN_INLINE int32_t vcn_clamp32(int64_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : (int32_t)v);
}
VCN_INLINE uint8_t vcn_u8(int32_t v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }
static inline void vcn_wr16(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
static inline uint32_t vcn_rd16(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
/* little-endian 32-bit loads and stores (one move on little-endian targets) */
static inline void vcn_wr32(uint8_t *p, uint32_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  memcpy(p, &v, 4);
#else
  for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
#endif
}
static inline uint32_t vcn_rd32(const uint8_t *p) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
#else
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
#endif
}

/* small dense least-squares solve (encoder only) */
static int vcn_solve(double *A, double *b, int n) { /* Cholesky; A n x n SPD, solves in place into b */
  for (int j = 0; j < n; j++) {
    double d = A[j * n + j];
    for (int k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (d <= 1e-12) return -1;
    d = sqrt(d);
    A[j * n + j] = d;
    for (int i = j + 1; i < n; i++) {
      double s2 = A[i * n + j];
      for (int k = 0; k < j; k++) s2 -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s2 / d;
    }
  }
  for (int i = 0; i < n; i++) {
    double s2 = b[i];
    for (int k = 0; k < i; k++) s2 -= A[i * n + k] * b[k];
    b[i] = s2 / A[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double s2 = b[i];
    for (int k = i + 1; k < n; k++) s2 -= A[k * n + i] * b[k];
    b[i] = s2 / A[i * n + i];
  }
  return 0;
}

/* AVX2 kernels on x86-64 with GCC or Clang, chosen at run time; the plain C kernels give the same
 * integers. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(VCN_NO_AVX2)
#include <immintrin.h>
#define VCN_HAVE_AVX2 1
static inline int vcn_use_avx2(void) {
  static int v = -1;
  int r = VCN_LD(&v);
  if (r < 0) {
    r = __builtin_cpu_supports("avx2") ? 1 : 0;
    VCN_ST(&v, r);
  }
  return r;
}
#else
#define VCN_HAVE_AVX2 0
static inline int vcn_use_avx2(void) { return 0; }
#endif
static inline const char *vcn_kernels(void) { return vcn_use_avx2() ? "avx2" : "c"; }

/* ======================================================================== */
/*          predictor rows and least-squares fit (shared with mode 4)       */
/* ======================================================================== */
#define VCN_LF_TAPS 12
#define VCN_LF_NCTX 8
#define VCN_LF_ST 8
#define VCN_LF_HDR (2 * VCN_LF_TAPS + 4 + 2 * (VCN_LF_NCTX - 1))

/* the six source rows of output row (z, y), padded: [N, NN, U, UN, US, UU][x + 2] */
static void vcn_lf_rows(const uint8_t *v, unsigned z, unsigned y, uint8_t pad[6][132]) {
  const size_t D = VCN_DIM, D2 = D * D;
  static const uint8_t flat[VCN_DIM] = {
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
      128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
  const uint8_t *N = y ? v + z * D2 + (y - 1) * D : (z ? v + (z - 1) * D2 + y * D : flat);
  const uint8_t *NN = y >= 2 ? v + z * D2 + (y - 2) * D : N;
  const uint8_t *U = z ? v + (z - 1) * D2 + y * D : N;
  const uint8_t *UN = z && y ? v + (z - 1) * D2 + (y - 1) * D : U;
  const uint8_t *US = z && y + 1 < VCN_DIM ? v + (z - 1) * D2 + (y + 1) * D : U;
  const uint8_t *UU = z >= 2 ? v + (z - 2) * D2 + y * D : U;
  const uint8_t *src[6] = {N, NN, U, UN, US, UU};
  for (unsigned r = 0; r < 6; r++) {
    memcpy(pad[r] + 2, src[r], VCN_DIM);
    pad[r][0] = pad[r][1] = src[r][0];
    pad[r][130] = pad[r][131] = src[r][VCN_DIM - 1];
  }
}

VCN_INLINE void vcn_lf_taps(const uint8_t pad[6][132], int x, double *f) {
  const uint8_t *n = pad[0] + 2, *nn = pad[1] + 2, *u = pad[2] + 2, *un = pad[3] + 2, *us = pad[4] + 2, *uu = pad[5] + 2;
  f[0] = n[x], f[1] = n[x - 1], f[2] = n[x + 1], f[3] = nn[x], f[4] = n[x - 2], f[5] = n[x + 2];
  f[6] = u[x], f[7] = u[x - 1], f[8] = u[x + 1], f[9] = un[x], f[10] = us[x], f[11] = uu[x];
}

/* weights, bias and activity thresholds of modes 4 and 5, from the source chunk */
/* zs: slice step of the sample grid (4 in mode 4; 16 in mode 5: a quarter of the samples, a faster encoder) */
static void vcn_lf_fit(const uint8_t *v, unsigned zs, int32_t wq[VCN_LF_TAPS], int32_t *bias_out,
                       uint16_t thr[VCN_LF_NCTX - 1]) {
  uint8_t pad[6][132];
  /* least-squares weights on a sparse grid of rows and voxels */
  enum { NF = VCN_LF_TAPS + 1 };
  double A[NF * NF], bvec[NF];
  memset(A, 0, sizeof A), memset(bvec, 0, sizeof bvec);
  for (unsigned z = 2; z < VCN_DIM; z += zs)
    for (unsigned y = 2 + (z & 4u); y + 1 < VCN_DIM; y += 8) {
      vcn_lf_rows(v, z, y, pad);
      const uint8_t *row = v + ((size_t)z * VCN_DIM + y) * VCN_DIM;
      for (int x = (int)((y + z) & 3u); x < (int)VCN_DIM; x += 4) {
        double fd[NF];
        vcn_lf_taps(pad, x, fd);
        fd[VCN_LF_TAPS] = 1.0;
        for (int a = 0; a < NF; a++) {
          bvec[a] += fd[a] * row[x];
          for (int c = 0; c <= a; c++) A[a * NF + c] += fd[a] * fd[c];
        }
      }
    }
  for (int a = 0; a < NF; a++)
    for (int c = a + 1; c < NF; c++) A[a * NF + c] = A[c * NF + a];
  const int NT = VCN_LF_TAPS;
  const double cntv = A[NT * NF + NT] > 0 ? A[NT * NF + NT] : 1.0;
  double C[VCN_LF_TAPS * VCN_LF_TAPS], cb[VCN_LF_TAPS], mf[VCN_LF_TAPS], mt = bvec[NT] / cntv, tr = 0;
  for (int a = 0; a < NT; a++) mf[a] = A[a * NF + NT] / cntv;
  for (int a = 0; a < NT; a++) {
    cb[a] = bvec[a] / cntv - mf[a] * mt;
    for (int c = 0; c < NT; c++) C[a * NT + c] = A[a * NF + c] / cntv - mf[a] * mf[c];
    tr += C[a * NT + a];
  }
  for (int a = 0; a < NT; a++) C[a * NT + a] += 1e-7 * tr + 1e-9;
  int32_t bias = 0;
  if (vcn_solve(C, cb, NT) == 0) {
    double bb = mt;
    for (int k = 0; k < NT; k++) {
      double q = floor(cb[k] * 4096.0 + 0.5);
      wq[k] = (int32_t)(q < -32768 ? -32768 : (q > 32767 ? 32767 : q));
      bb -= wq[k] / 4096.0 * mf[k];
    }
    bb = floor(bb * 4096.0 + 0.5);
    bias = (int32_t)(bb < -(1 << 22) ? -(1 << 22) : (bb > (1 << 22) ? (1 << 22) : bb));
  } else { /* degenerate data: N + U - UN */
    static const int32_t L[VCN_LF_TAPS] = {4096, 0, 0, 0, 0, 0, 4096, 0, 0, -4096, 0, 0};
    memcpy(wq, L, sizeof L);
  }
  /* activity thresholds: quantiles on the same grid */
  {
    uint32_t hist[1531], ns = 0;
    memset(hist, 0, sizeof hist);
    for (unsigned z = 2; z < VCN_DIM; z += zs)
      for (unsigned y = 2 + (z & 2u); y + 1 < VCN_DIM; y += 4) {
        vcn_lf_rows(v, z, y, pad);
        const uint8_t *n = pad[0] + 2, *u = pad[2] + 2, *un = pad[3] + 2, *us = pad[4] + 2;
        for (int x = (int)((y + z) & 3u); x < (int)VCN_DIM; x += 4) {
          const int a = abs((int)n[x] - n[x - 1]) + abs((int)n[x] - n[x + 1]) + abs((int)u[x] - un[x]) +
                        abs((int)u[x] - us[x]) + abs((int)u[x] - u[x - 1]) + abs((int)u[x] - u[x + 1]);
          hist[a]++, ns++;
        }
      }
    uint32_t acc = 0;
    unsigned k = 0;
    for (unsigned a = 0; a < 1531 && k < VCN_LF_NCTX - 1; a++) {
      acc += hist[a];
      while (k < VCN_LF_NCTX - 1 && (uint64_t)acc * VCN_LF_NCTX >= (uint64_t)(k + 1) * ns) thr[k++] = (uint16_t)a;
    }
    while (k < VCN_LF_NCTX - 1) thr[k++] = 1530;
  }
  *bias_out = bias;
}

/* ======================================================================== */
/*                       Huffman lossless mode (mode 5)                      */
/* ======================================================================== */
/* Mode 5. The predictor, weights, bias and activity classes of mode 4 (the same row-parallel prediction
 * from the two previous rows and slices), with the residual coded by static canonical Huffman codes,
 * one per activity class, at most VCN_LH_L bits long, into VCN_LH_NS bit streams (voxel x of every row
 * goes to stream x mod VCN_LH_NS), so the decoder follows independent dependency chains. Bits are read
 * least significant first; each stream ends at a byte boundary with zero padding.
 * Payload: weights (12 x int16), bias (int32), thresholds (7 x u16), for each class the count n of
 * coded symbols (u16, 0 to 256) and n code lengths (4 bits each, low nibble first, 0 = absent), the
 * byte sizes of the first VCN_LH_NS - 1 streams (u32 each), then the streams. */
#define VCN_LH_L 11
#define VCN_LH_NS 4 /* the coder below is written for exactly 4 */
#define VCN_LH_TAB (1u << VCN_LH_L)
#define VCN_LH_RS 136 /* stride of a padded 16-bit row: voxel x at [x + 4], x = -2 .. 129 valid */

/* mode 4's prediction and context classes from rows already widened to 16 bits and padded (the same
 * integers as vcn_lf_pred; no widening in the loop). r: N, NN, U, UN, US, UU, each at its voxel 0 */
static void vcn_lf_pred16_c(const int16_t *const r[6], const int32_t *restrict wq, int32_t bias,
                            const uint16_t *restrict thr, uint8_t *restrict pred, uint8_t *restrict ctx) {
  const int16_t *n = r[0], *nn = r[1], *u = r[2], *un = r[3], *us = r[4], *uu = r[5];
  for (int x = 0; x < (int)VCN_DIM; x++) {
    const int32_t p = bias + wq[0] * n[x] + wq[1] * n[x - 1] + wq[2] * n[x + 1] + wq[3] * nn[x] + wq[4] * n[x - 2] +
                      wq[5] * n[x + 2] + wq[6] * u[x] + wq[7] * u[x - 1] + wq[8] * u[x + 1] + wq[9] * un[x] +
                      wq[10] * us[x] + wq[11] * uu[x];
    const int32_t q = (p + 2048) >> 12;
    pred[x] = (uint8_t)(q < 0 ? 0 : (q > 255 ? 255 : q));
    const int a = abs(n[x] - n[x - 1]) + abs(n[x] - n[x + 1]) + abs(u[x] - un[x]) + abs(u[x] - us[x]) +
                  abs(u[x] - u[x - 1]) + abs(u[x] - u[x + 1]);
    unsigned c = 0;
    for (unsigned k = 0; k < VCN_LF_NCTX - 1; k++) c += (unsigned)a > thr[k];
    ctx[x] = (uint8_t)c;
  }
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void vcn_lf_pred16_avx2(const int16_t *const r[6], const int32_t *restrict wq,
                                                               int32_t bias, const uint16_t *restrict thr,
                                                               uint8_t *restrict pred, uint8_t *restrict ctx) {
  const int16_t *n = r[0], *nn = r[1], *u = r[2], *un = r[3], *us = r[4], *uu = r[5];
#define VCN_W2(a, b) _mm256_set1_epi32((int32_t)((uint32_t)(uint16_t)wq[a] | (uint32_t)(uint16_t)wq[b] << 16))
  const __m256i k01 = VCN_W2(0, 1), k23 = VCN_W2(2, 3), k45 = VCN_W2(4, 5), k67 = VCN_W2(6, 7), k89 = VCN_W2(8, 9),
                kab = VCN_W2(10, 11);
#undef VCN_W2
  const __m256i vb = _mm256_set1_epi32(bias + 2048);
  __m256i th[VCN_LF_NCTX - 1];
  for (unsigned k = 0; k < VCN_LF_NCTX - 1; k++) th[k] = _mm256_set1_epi16((short)thr[k]);
  for (int x = 0; x < (int)VCN_DIM; x += 16) {
#define VCN_L(p) _mm256_loadu_si256((const __m256i *)(p))
    const __m256i N = VCN_L(n + x), NW = VCN_L(n + x - 1), NE = VCN_L(n + x + 1), NN = VCN_L(nn + x),
                  NWW = VCN_L(n + x - 2), NEE = VCN_L(n + x + 2), U = VCN_L(u + x), UW = VCN_L(u + x - 1),
                  UE = VCN_L(u + x + 1), UN = VCN_L(un + x), US = VCN_L(us + x), UU = VCN_L(uu + x);
#undef VCN_L
    __m256i lo = vb, hi = vb;
#define VCN_P(A, B, K)                                                                                    \
  lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(A, B), K));                            \
  hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(A, B), K))
    VCN_P(N, NW, k01);
    VCN_P(NE, NN, k23);
    VCN_P(NWW, NEE, k45);
    VCN_P(U, UW, k67);
    VCN_P(UE, UN, k89);
    VCN_P(US, UU, kab);
#undef VCN_P
    const __m256i q = _mm256_packs_epi32(_mm256_srai_epi32(lo, 12), _mm256_srai_epi32(hi, 12));
    const __m256i q8 = _mm256_packus_epi16(q, q);
    _mm_storeu_si128((__m128i *)(pred + x), _mm256_castsi256_si128(_mm256_permute4x64_epi64(q8, 0x08)));
    __m256i a = _mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(N, NW)), _mm256_abs_epi16(_mm256_sub_epi16(N, NE)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, UN)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, US)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, UW)));
    a = _mm256_add_epi16(a, _mm256_abs_epi16(_mm256_sub_epi16(U, UE)));
    __m256i c = _mm256_setzero_si256();
    for (unsigned k = 0; k < VCN_LF_NCTX - 1; k++) c = _mm256_sub_epi16(c, _mm256_cmpgt_epi16(a, th[k]));
    const __m256i c8 = _mm256_packus_epi16(c, c);
    _mm_storeu_si128((__m128i *)(ctx + x), _mm256_castsi256_si128(_mm256_permute4x64_epi64(c8, 0x08)));
  }
}
#endif
static inline void vcn_lf_pred16(const int16_t *const r[6], const int32_t *restrict wq, int32_t bias,
                                 const uint16_t *restrict thr, uint8_t *restrict pred, uint8_t *restrict ctx) {
#if VCN_HAVE_AVX2
  if (vcn_use_avx2()) {
    vcn_lf_pred16_avx2(r, wq, bias, thr, pred, ctx);
    return;
  }
#endif
  vcn_lf_pred16_c(r, wq, bias, thr, pred, ctx);
}
/* widen a row of 128 voxels into a padded 16-bit row (two replicated voxels on each side) */
static inline void vcn_lf_widen(const uint8_t *src, int16_t *dst) {
  for (unsigned x = 0; x < VCN_DIM; x++) dst[x + 4] = src[x];
  dst[2] = dst[3] = src[0];
  dst[VCN_DIM + 4] = dst[VCN_DIM + 5] = src[VCN_DIM - 1];
}
/* the six source rows of output row (z, y), as vcn_lf_rows chooses them; s0, s1, s2: the padded 16-bit
 * slices z, z - 1, z - 2 (rows VCN_LH_RS apart), flat: a padded row of 128 */
static inline void vcn_lf_src16(const int16_t *s0, const int16_t *s1, const int16_t *s2, const int16_t *flat,
                                unsigned z, unsigned y, const int16_t *r[6]) {
  const size_t R = VCN_LH_RS;
  const int16_t *N = y ? s0 + (y - 1) * R : (z ? s1 + y * R : flat);
  const int16_t *U = z ? s1 + y * R : N;
  r[0] = N + 4;
  r[1] = (y >= 2 ? s0 + (y - 2) * R : N) + 4;
  r[2] = U + 4;
  r[3] = (z && y ? s1 + (y - 1) * R : U) + 4;
  r[4] = (z && y + 1 < VCN_DIM ? s1 + (y + 1) * R : U) + 4;
  r[5] = (z >= 2 ? s2 + y * R : U) + 4;
}

static inline uint64_t vcn_rd64(const uint8_t *p) {
  uint64_t x;
  memcpy(&x, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  x = __builtin_bswap64(x);
#endif
  return x;
}
static inline void vcn_wr64(uint8_t *p, uint64_t x) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  x = __builtin_bswap64(x);
#endif
  memcpy(p, &x, 8);
}

/* code lengths of a length-limited Huffman code for cnt (0 = absent); every present symbol gets a code */
static void vcn_lh_lengths(const uint32_t cnt[256], uint8_t len[256]) {
  unsigned idx[256], m = 0;
  memset(len, 0, 256);
  for (unsigned s = 0; s < 256; s++)
    if (cnt[s]) idx[m++] = s;
  if (m == 0) return;
  if (m == 1) {
    len[idx[0]] = 1;
    return;
  }
  /* sort by count ascending, then symbol (insertion sort; m <= 256, once per class) */
  for (unsigned i = 1; i < m; i++) {
    const unsigned t = idx[i];
    unsigned j = i;
    while (j > 0 && (cnt[idx[j - 1]] > cnt[t] || (cnt[idx[j - 1]] == cnt[t] && idx[j - 1] > t))) idx[j] = idx[j - 1], j--;
    idx[j] = t;
  }
  /* two-queue Huffman: leaves in idx order, internal nodes in creation order (nondecreasing weights) */
  uint64_t w[512];
  int par[512];
  for (unsigned i = 0; i < m; i++) w[i] = cnt[idx[i]];
  unsigned ql = 0, qi = m, nn = m;
  for (unsigned k = 0; k + 1 < m; k++) {
    unsigned pick[2];
    for (int c = 0; c < 2; c++) {
      if (ql < m && (qi >= nn || w[ql] <= w[qi])) pick[c] = ql++;
      else pick[c] = qi++;
    }
    w[nn] = w[pick[0]] + w[pick[1]];
    par[pick[0]] = par[pick[1]] = (int)nn;
    nn++;
  }
  unsigned depth[512];
  depth[nn - 1] = 0;
  for (int i = (int)nn - 2; i >= 0; i--) depth[i] = depth[par[i]] + 1;
  /* limit to VCN_LH_L: clamp, then lengthen the least frequent codes until the Kraft sum fits, then
   * shorten the most frequent codes while it still fits (units of 2^-L) */
  unsigned l[256];
  uint64_t K = 0;
  for (unsigned i = 0; i < m; i++) {
    l[i] = depth[i] > VCN_LH_L ? VCN_LH_L : depth[i];
    K += 1ull << (VCN_LH_L - l[i]);
  }
  while (K > VCN_LH_TAB) {
    unsigned best = m;
    for (unsigned i = 0; i < m; i++) /* least frequent first: the longest code below the limit */
      if (l[i] < VCN_LH_L && (best == m || l[i] > l[best])) best = i;
    l[best]++;
    K -= 1ull << (VCN_LH_L - l[best]);
  }
  for (unsigned i = m; i-- > 0;)
    while (l[i] > 1 && K + (1ull << (VCN_LH_L - l[i])) <= VCN_LH_TAB) K += 1ull << (VCN_LH_L - l[i]), l[i]--;
  for (unsigned i = 0; i < m; i++) len[idx[i]] = (uint8_t)l[i];
}

/* canonical codes, bit-reversed for least-significant-first reading; returns 0 if the lengths are valid */
static int vcn_lh_codes(const uint8_t len[256], uint16_t code[256]) {
  unsigned nl[VCN_LH_L + 1] = {0};
  uint64_t K = 0;
  for (unsigned s = 0; s < 256; s++) {
    if (len[s] > VCN_LH_L) return 1;
    if (len[s]) nl[len[s]]++, K += 1ull << (VCN_LH_L - len[s]);
  }
  if (K > VCN_LH_TAB) return 1;
  unsigned next[VCN_LH_L + 2], c = 0;
  for (unsigned b = 1; b <= VCN_LH_L; b++) c = (c + nl[b - 1]) << 1, next[b] = c;
  for (unsigned s = 0; s < 256; s++) {
    const unsigned b = len[s];
    if (!b) continue;
    unsigned v = next[b]++, r = 0;
    for (unsigned k = 0; k < b; k++) r = (r << 1) | ((v >> k) & 1u);
    code[s] = (uint16_t)r;
  }
  return 0;
}

#define VCN_LH_SCAP ((size_t)VCN_VOXELS / VCN_LH_NS * VCN_LH_L / 8 + 16)

VCN_CLONES_BMI static vcn_status vcn_decode_llhuf(const uint8_t *p, size_t n, uint8_t *v) {
  const uint8_t *end = p + n;
  if (n < VCN_LF_HDR) return VCN_ERR_CORRUPT;
  int32_t wq[VCN_LF_TAPS];
  for (unsigned i = 0; i < VCN_LF_TAPS; i++) wq[i] = (int32_t)(int16_t)vcn_rd16(p + 2 * i);
  p += 2 * VCN_LF_TAPS;
  const int32_t bias = (int32_t)vcn_rd32(p);
  p += 4;
  if (bias < -(1 << 22) || bias > (1 << 22)) return VCN_ERR_CORRUPT;
  uint16_t thr[VCN_LF_NCTX - 1];
  for (unsigned i = 0; i < VCN_LF_NCTX - 1; i++) thr[i] = (uint16_t)vcn_rd16(p + 2 * i);
  p += 2 * (VCN_LF_NCTX - 1);
  /* the code tables, then the last three slices widened to 16 bits and a flat row */
  uint16_t *blk = (uint16_t *)malloc(sizeof(uint16_t) * (VCN_LF_NCTX * VCN_LH_TAB + (3 * VCN_DIM + 1) * VCN_LH_RS));
  if (!blk) return VCN_ERR_NOMEM;
  uint16_t *restrict T = blk;
  for (unsigned t = 0; t < VCN_LF_NCTX; t++) {
    uint8_t len[256];
    uint16_t code[256];
    if (end - p < 2) goto bad;
    const unsigned m = vcn_rd16(p);
    p += 2;
    if (m > 256 || (size_t)(end - p) < (m + 1) / 2) goto bad;
    memset(len, 0, sizeof len);
    for (unsigned s = 0; s < m; s++) len[s] = (uint8_t)((p[s >> 1] >> (4 * (s & 1u))) & 15u);
    p += (m + 1) / 2;
    if (vcn_lh_codes(len, code)) goto bad;
    uint16_t *tt = T + (size_t)t * VCN_LH_TAB;
    for (unsigned i = 0; i < VCN_LH_TAB; i++) tt[i] = (uint16_t)(0x8000u | VCN_LH_L << 8); /* not a code */
    for (unsigned s = 0; s < 256; s++)
      if (len[s])
        for (unsigned j = code[s]; j < VCN_LH_TAB; j += 1u << len[s]) tt[j] = (uint16_t)(s | (unsigned)len[s] << 8);
  }
  if ((size_t)(end - p) < 4 * (VCN_LH_NS - 1)) goto bad;
  {
    const uint8_t *sp[VCN_LH_NS], *se[VCN_LH_NS];
    const uint8_t *q = p + 4 * (VCN_LH_NS - 1);
    for (unsigned k = 0; k < VCN_LH_NS; k++) {
      const size_t sz = k + 1 < VCN_LH_NS ? vcn_rd32(p + 4 * k) : (size_t)(end - q);
      if (sz > (size_t)(end - q)) goto bad;
      sp[k] = q, se[k] = q + sz, q += sz;
    }
    const uint8_t *p0 = sp[0], *p1 = sp[1], *p2 = sp[2], *p3 = sp[3];
    const uint8_t *e0 = se[0], *e1 = se[1], *e2 = se[2], *e3 = se[3];
    uint64_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0, v0 = 0, v1 = 0, v2 = 0, v3 = 0, acc = 0;
    /* refill to at least 56 bits (5 codes of at most 11); near the end byte by byte, zeros past it
     * counted in v */
#define VCN_LH_FILL(k)                                                                                     \
  do {                                                                                                     \
    if (e##k - p##k >= 8) {                                                                                \
      f##k |= vcn_rd64(p##k) << c##k;                                                                      \
      p##k += (63u - c##k) >> 3;                                                                           \
      c##k |= 56u;                                                                                         \
    } else                                                                                                 \
      for (; c##k <= 56u; c##k += 8) {                                                                     \
        if (p##k < e##k) f##k |= (uint64_t)*p##k++ << c##k;                                                \
        else v##k++;                                                                                       \
      }                                                                                                    \
  } while (0)
#define VCN_LH_GET(k, x)                                                                                   \
  do {                                                                                                     \
    const unsigned e_ = T[(size_t)ctx[x] << VCN_LH_L | (unsigned)(f##k & (VCN_LH_TAB - 1u))];              \
    const unsigned b_ = (e_ >> 8) & 15u;                                                                   \
    sym[x] = (uint8_t)e_, acc |= e_, f##k >>= b_, c##k -= b_;                                              \
  } while (0)
    uint8_t pred[VCN_DIM], ctx[VCN_DIM], sym[VCN_DIM];
    /* the last three slices widened to 16 bits, and a flat row */
    int16_t *restrict ring = (int16_t *)(blk + VCN_LF_NCTX * VCN_LH_TAB), *flat = ring + 3 * VCN_DIM * VCN_LH_RS;
    for (unsigned x = 0; x < VCN_LH_RS; x++) flat[x] = 128;
    for (unsigned z = 0; z < VCN_DIM; z++)
      for (unsigned y = 0; y < VCN_DIM; y++) {
        const size_t SL = (size_t)VCN_DIM * VCN_LH_RS;
        const int16_t *r6[6];
        int16_t *s0 = ring + (z % 3) * SL;
        vcn_lf_src16(s0, ring + ((z + 2) % 3) * SL, ring + ((z + 1) % 3) * SL, flat, z, y, r6);
        vcn_lf_pred16(r6, wq, bias, thr, pred, ctx);
        for (unsigned x0 = 0; x0 < VCN_DIM; x0 += 4 * VCN_LH_NS) {
          VCN_LH_FILL(0); VCN_LH_FILL(1); VCN_LH_FILL(2); VCN_LH_FILL(3);
          for (unsigned x = x0; x < x0 + 4 * VCN_LH_NS; x += 4) {
            VCN_LH_GET(0, x); VCN_LH_GET(1, x + 1); VCN_LH_GET(2, x + 2); VCN_LH_GET(3, x + 3);
          }
        }
        uint8_t *row = v + ((size_t)z * VCN_DIM + y) * VCN_DIM;
        for (unsigned x = 0; x < VCN_DIM; x++)
          row[x] = (uint8_t)(pred[x] + ((sym[x] >> 1) ^ (uint8_t)(0u - (sym[x] & 1u))));
        vcn_lf_widen(row, s0 + y * VCN_LH_RS);
      }
#undef VCN_LH_FILL
#undef VCN_LH_GET
    const uint8_t *pe[VCN_LH_NS] = {p0, p1, p2, p3};
    const uint64_t buf[VCN_LH_NS] = {f0, f1, f2, f3};
    const unsigned cnt[VCN_LH_NS] = {c0, c1, c2, c3}, virt[VCN_LH_NS] = {v0, v1, v2, v3};
    for (unsigned k = 0; k < VCN_LH_NS; k++) sp[k] = pe[k];
    free(blk);
    if (acc & 0x8000u) return VCN_ERR_CORRUPT;
    /* every stream used up to its last byte, not past it: the bits left are the final byte's zero
     * padding (and the zeros read past the end) */
    for (unsigned k = 0; k < VCN_LH_NS; k++) {
      const int real = (int)cnt[k] - 8 * (int)virt[k];
      if (sp[k] != se[k] || real < 0 || real >= 8) return VCN_ERR_CORRUPT;
      if (cnt[k] < 64 && (buf[k] & ((1ull << cnt[k]) - 1u))) return VCN_ERR_CORRUPT;
      if (cnt[k] >= 64 && buf[k]) return VCN_ERR_CORRUPT;
    }
    return VCN_OK;
  }
bad:
  free(blk);
  return VCN_ERR_CORRUPT;
}

VCN_CLONES_BMI static vcn_status vcn_encode_llhuf(const uint8_t *v, uint8_t *out, size_t cap, size_t *out_n) {
  int32_t wq[VCN_LF_TAPS], bias;
  uint16_t thr[VCN_LF_NCTX - 1];
  vcn_lf_fit(v, 16, wq, &bias, thr);
  uint16_t *restrict ts = (uint16_t *)malloc(sizeof(uint16_t) * VCN_VOXELS);
  int16_t *ring = (int16_t *)malloc(sizeof(int16_t) * (3 * VCN_DIM + 1) * VCN_LH_RS);
  if (!ts || !ring) {
    free(ts), free(ring);
    return VCN_ERR_NOMEM;
  }
  /* counts by class << 8 | symbol in four tables (voxel x mod 4), 64 words more than their size apart
   * so that the counters of neighbouring voxels never sit at the same offset in a 4 KiB page */
  enum { CW = VCN_LF_NCTX * 256 + 64 };
  uint32_t cnt[4 * CW];
  memset(cnt, 0, sizeof cnt);
  {
    const size_t SL = (size_t)VCN_DIM * VCN_LH_RS;
    int16_t *flat = ring + 3 * SL;
    for (unsigned x = 0; x < VCN_LH_RS; x++) flat[x] = 128;
    uint8_t pred[VCN_DIM], cr[VCN_DIM];
    uint32_t *c0 = cnt, *c1 = cnt + CW, *c2 = cnt + 2 * CW, *c3 = cnt + 3 * CW;
    for (unsigned z = 0; z < VCN_DIM; z++) {
      int16_t *s0 = ring + (z % 3) * SL;
      for (unsigned y = 0; y < VCN_DIM; y++) vcn_lf_widen(v + ((size_t)z * VCN_DIM + y) * VCN_DIM, s0 + y * VCN_LH_RS);
      for (unsigned y = 0; y < VCN_DIM; y++) {
        const size_t r0 = ((size_t)z * VCN_DIM + y) * VCN_DIM;
        const int16_t *r6[6];
        vcn_lf_src16(s0, ring + ((z + 2) % 3) * SL, ring + ((z + 1) % 3) * SL, flat, z, y, r6);
        vcn_lf_pred16(r6, wq, bias, thr, pred, cr);
        uint16_t *t = ts + r0;
        for (unsigned x = 0; x < VCN_DIM; x++) { /* zigzag mod 256: (2r) ^ -(r >> 7) */
          const uint8_t r = (uint8_t)(v[r0 + x] - pred[x]);
          t[x] = (uint16_t)(cr[x] << 8 | (uint8_t)((uint8_t)(r << 1) ^ (uint8_t)(0u - (r >> 7))));
        }
        for (unsigned x = 0; x < VCN_DIM; x += 4) c0[t[x]]++, c1[t[x + 1]]++, c2[t[x + 2]]++, c3[t[x + 3]]++;
      }
    }
  }
  free(ring);
  uint8_t len[VCN_LF_NCTX][256];
  uint32_t ct[VCN_LF_NCTX * 256]; /* code | length << 16 by (class << 8 | symbol) */
  for (unsigned t = 0; t < VCN_LF_NCTX; t++) {
    uint32_t c[256];
    uint16_t code[256];
    for (unsigned s = 0; s < 256; s++) {
      const unsigned j = t * 256 + s;
      c[s] = cnt[j] + cnt[CW + j] + cnt[2 * CW + j] + cnt[3 * CW + j];
    }
    vcn_lh_lengths(c, len[t]);
    (void)vcn_lh_codes(len[t], code);
    for (unsigned s = 0; s < 256; s++) ct[t * 256 + s] = len[t][s] ? (uint32_t)code[s] | (uint32_t)len[t][s] << 16 : 0;
  }
  uint8_t hdr[VCN_LF_HDR + VCN_LF_NCTX * (2 + 128) + 4 * VCN_LH_NS];
  size_t pos = 0;
  for (unsigned i = 0; i < VCN_LF_TAPS; i++) vcn_wr16(hdr + pos, (uint32_t)(uint16_t)(int16_t)wq[i]), pos += 2;
  vcn_wr32(hdr + pos, (uint32_t)bias), pos += 4;
  for (unsigned i = 0; i < VCN_LF_NCTX - 1; i++) vcn_wr16(hdr + pos, thr[i]), pos += 2;
  for (unsigned t = 0; t < VCN_LF_NCTX; t++) {
    unsigned m = 256;
    while (m > 0 && len[t][m - 1] == 0) m--;
    vcn_wr16(hdr + pos, m), pos += 2;
    for (unsigned s = 0; s < m; s += 2)
      hdr[pos++] = (uint8_t)(len[t][s] | (s + 1 < m ? len[t][s + 1] : 0u) << 4);
  }
  /* the streams, each into its own buffer (8 bytes of slack for the 64-bit stores) */
  const size_t scap = VCN_LH_SCAP;
  uint8_t *restrict sb = (uint8_t *)malloc(scap * VCN_LH_NS);
  if (!sb) {
    free(ts);
    return VCN_ERR_NOMEM;
  }
  uint8_t *o[VCN_LH_NS];
  uint64_t bb[VCN_LH_NS];
  unsigned nb[VCN_LH_NS];
  for (unsigned k = 0; k < VCN_LH_NS; k++) o[k] = sb + k * scap, bb[k] = 0, nb[k] = 0;
  {
    /* four codes per stream (at most 44 bits) between flushes, the streams' state in registers */
    uint8_t *o0 = o[0], *o1 = o[1], *o2 = o[2], *o3 = o[3];
    uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    unsigned n0 = 0, n1 = 0, n2 = 0, n3 = 0;
#define VCN_LH_PUT(k, i)                                                                                   \
  do {                                                                                                     \
    const uint32_t e_ = ct[t[i]];                                                                          \
    b##k |= (uint64_t)(e_ & 0xFFFFu) << n##k;                                                              \
    n##k += e_ >> 16;                                                                                      \
  } while (0)
#define VCN_LH_FLUSH(k)                                                                                    \
  do {                                                                                                     \
    vcn_wr64(o##k, b##k);                                                                                  \
    o##k += n##k >> 3, b##k >>= n##k & ~7u, n##k &= 7u;                                                    \
  } while (0)
    for (size_t i0 = 0; i0 < VCN_VOXELS; i0 += 4 * VCN_LH_NS) {
      const uint16_t *t = ts + i0;
      for (unsigned j = 0; j < 16; j += 4) {
        VCN_LH_PUT(0, j); VCN_LH_PUT(1, j + 1); VCN_LH_PUT(2, j + 2); VCN_LH_PUT(3, j + 3);
      }
      VCN_LH_FLUSH(0); VCN_LH_FLUSH(1); VCN_LH_FLUSH(2); VCN_LH_FLUSH(3);
    }
#undef VCN_LH_PUT
#undef VCN_LH_FLUSH
    o[0] = o0, o[1] = o1, o[2] = o2, o[3] = o3;
    bb[0] = b0, bb[1] = b1, bb[2] = b2, bb[3] = b3;
    nb[0] = n0, nb[1] = n1, nb[2] = n2, nb[3] = n3;
  }
  free(ts);
  size_t sz[VCN_LH_NS], total = pos + 4 * (VCN_LH_NS - 1);
  for (unsigned k = 0; k < VCN_LH_NS; k++) {
    if (nb[k]) vcn_wr64(o[k], bb[k]), o[k]++;
    sz[k] = (size_t)(o[k] - (sb + k * scap));
    total += sz[k];
  }
  vcn_status st = VCN_OK;
  if (total > cap) st = VCN_ERR_CAP;
  else {
    for (unsigned k = 0; k + 1 < VCN_LH_NS; k++) vcn_wr32(hdr + pos, (uint32_t)sz[k]), pos += 4;
    memcpy(out, hdr, pos);
    uint8_t *w = out + pos;
    for (unsigned k = 0; k < VCN_LH_NS; k++) memcpy(w, sb + k * scap, sz[k]), w += sz[k];
    *out_n = total;
  }
  free(sb);
  return st;
}

/* ======================================================================== */
/*                                 top level                                */
/* ======================================================================== */
static inline void vcn_header(uint8_t *h, unsigned mode) {
  h[0] = 'V', h[1] = 'C', h[2] = 'N', h[3] = 'G', h[4] = (uint8_t)VCN_REVISION, h[5] = (uint8_t)mode;
  memset(h + 6, 0, VCN_HDR - 6);
}

static inline vcn_status vcn_encode_lossless_chunk(const uint8_t *src, uint8_t *dst, size_t cap, size_t *out_n) {
  if (!src || !dst || !out_n) return VCN_ERR_ARG;
  /* constant chunk */
  {
    size_t i = 1;
    while (i < VCN_VOXELS && src[i] == src[0]) i++;
    if (i == VCN_VOXELS) {
      if (cap < VCN_HDR + 1) return VCN_ERR_CAP;
      vcn_header(dst, VCN_MODE_CONST);
      dst[VCN_HDR] = src[0];
      *out_n = VCN_HDR + 1;
      return VCN_OK;
    }
  }
  {
    size_t n = 0;
    vcn_status st = VCN_ERR_CAP;
    if (cap >= VCN_HDR) st = vcn_encode_llhuf(src, dst + VCN_HDR, cap - VCN_HDR, &n);
    if (st == VCN_OK && VCN_HDR + n < VCN_HDR + VCN_VOXELS) {
      vcn_header(dst, VCN_MODE_LLHUF);
      *out_n = VCN_HDR + n;
      return VCN_OK;
    }
    if (st != VCN_OK && st != VCN_ERR_CAP) return st;
  }
  /* incompressible: a raw chunk */
  if (cap < VCN_HDR + VCN_VOXELS) return VCN_ERR_CAP;
  vcn_header(dst, VCN_MODE_RAW);
  memcpy(dst + VCN_HDR, src, VCN_VOXELS);
  *out_n = VCN_HDR + VCN_VOXELS;
  return VCN_OK;
}

static inline vcn_status vcn_decode(const uint8_t *enc, size_t n, uint8_t *dst, size_t dst_n) {
  if (!enc || !dst || dst_n < VCN_VOXELS) return VCN_ERR_ARG;
  const int mode = vcn_chunk_mode(enc, n);
  if (mode < 0) return VCN_ERR_CORRUPT;
  if (mode == (int)VCN_MODE_CONST) {
    if (n != VCN_HDR + 1) return VCN_ERR_CORRUPT;
    memset(dst, enc[VCN_HDR], VCN_VOXELS);
    return VCN_OK;
  }
  if (mode == (int)VCN_MODE_RAW) {
    if (n != VCN_HDR + VCN_VOXELS) return VCN_ERR_CORRUPT;
    memcpy(dst, enc + VCN_HDR, VCN_VOXELS);
    return VCN_OK;
  }
  if (mode == (int)VCN_MODE_LLHUF) {
    if (enc[6] | enc[7] | enc[8] | enc[9] | enc[10] | enc[11]) return VCN_ERR_CORRUPT;
    return vcn_decode_llhuf(enc + VCN_HDR, n - VCN_HDR, dst);
  }
  if (mode == 0 || mode == 1 || mode == 4) return VCN_ERR_MODE;
  return VCN_ERR_CORRUPT;
}

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif /* VOLCODEC_NG_LOSSLESS_H */
