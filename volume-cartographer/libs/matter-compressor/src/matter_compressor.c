#define _GNU_SOURCE                  // pthread_setname_np (linux)
#include <pthread.h>
#include <stdatomic.h>
// ============================================================================
// matter_compressor.c — single-file implementation: codec + archive + cache.
// Unified from mc_dct.h / mc_rangecoder.h / mc_xxhash.h / mc_codec.c /
// mc_archive.h / mc_archive_read.h / mc_archive.c / mc_cache.c.
// ============================================================================
#include "matter_compressor.h"

// ============================================================================
// mc_dct.h — integer separable 3D DCT-16 (matter-compressor).
//
// Q14 fixed-point, even/odd partial-butterfly 1D core, 3 separable passes with a
// cache-blocked rotate between them. 16^3 only (the codec's fixed block size).
// Range-safe in i32 for 16^3. Lossy (integer rounding). Call mc_dct_init() once.
// ============================================================================
#ifndef MC_DCT_H
#define MC_DCT_H
#include <stdint.h>
#include <math.h>
// SIMD kernel selection. ARM: NEON always (Graviton, Apple M, X1 Elite).
// x86: AVX2 selected at compile time (x86-64-v3 is the project's floor, so
// build with -march=x86-64-v3 or newer); an AVX-512 line-pair variant is
// runtime-dispatched when compiled in (Zen4/5 have it; many Intel parts have
// it fused off, so never assume at compile time). SVE was evaluated and
// intentionally skipped: every hot kernel here is a fixed 8-lane i32 problem
// (half a DCT-16 line) which two NEON q-regs / one AVX2 ymm already saturate;
// scalable vectors add overhead, not lanes, at this block size.
#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define MC_SIMD_NEON 1
#elif defined(MC_ENABLE_AVX512) && defined(__AVX512F__)
  #include <immintrin.h>
  #define MC_SIMD_AVX512 1   // opt-in (-DMC_ENABLE_AVX512, -march=x86-64-v4/znver4);
                             // compile-tested only — default x86 builds use the
                             // measured AVX2 path below.
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define MC_SIMD_AVX2 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MC_DCT_N    16              // block edge
#define MC_DCT_Q    14              // fixed-point fraction bits
#define MC_DCT_ALIGN 64
typedef int32_t mc_fi32;

// Q14 cosine matrix for N=16, built once. Packed even/odd variants:
//   g_cm_eT[n][j] = cm[2j][n], g_cm_oT[n][j] = cm[2j+1][n]   (forward, k-contig)
//   g_cm_e [j][n] = cm[2j][n], g_cm_o [j][n] = cm[2j+1][n]   (inverse, n-contig)
static mc_fi32 g_mc_cm[MC_DCT_N][MC_DCT_N] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_eT[8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_oT[8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_e [8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_o [8][8] __attribute__((aligned(MC_DCT_ALIGN)));
static mc_fi32 g_cm_eo[8][16] __attribute__((aligned(MC_DCT_ALIGN)));   // [e row | o row] for zmm
static int g_mc_cm_ready = 0;
static void mc_dct_init(void){
    if(g_mc_cm_ready) return;
    double scale=(double)((int64_t)1<<MC_DCT_Q);
    for(int k=0;k<MC_DCT_N;++k){ double ck=(k==0)?sqrt(1.0/MC_DCT_N):sqrt(2.0/MC_DCT_N);
        for(int n=0;n<MC_DCT_N;++n){ double v=ck*cos(M_PI*(2.0*n+1.0)*k/(2.0*MC_DCT_N));
            g_mc_cm[k][n]=(mc_fi32)llround(v*scale); } }
    for(int j=0;j<8;++j)for(int n=0;n<8;++n){
        g_cm_e[j][n]=g_mc_cm[2*j][n];   g_cm_eT[n][j]=g_mc_cm[2*j][n];
        g_cm_o[j][n]=g_mc_cm[2*j+1][n]; g_cm_oT[n][j]=g_mc_cm[2*j+1][n];
        g_cm_eo[j][n]=g_cm_e[j][n];     g_cm_eo[j][n+8]=g_cm_o[j][n];
    }
    g_mc_cm_ready=1;
}

// 1D forward DCT-II (even/odd partial butterfly). NEON/AVX2: packed transposed
// tables (g_cm_eT/oT) give contiguous 8-lane MACs; scalar keeps the k-parallel
// form (measured faster under autovectorization). (AVX-512 defines fwd+inv
// together above.)
#if MC_SIMD_AVX512
/* defined alongside the inverse above */
#elif MC_SIMD_NEON
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[8], d[8];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    int32x4_t ae0=vdupq_n_s32(rnd), ae1=vdupq_n_s32(rnd);
    int32x4_t ao0=vdupq_n_s32(rnd), ao1=vdupq_n_s32(rnd);
    for(int n=0;n<H;++n){
        int32x4_t sn=vdupq_n_s32(s[n]), dn=vdupq_n_s32(d[n]);
        ae0=vmlaq_s32(ae0,vld1q_s32(&g_cm_eT[n][0]),sn);
        ae1=vmlaq_s32(ae1,vld1q_s32(&g_cm_eT[n][4]),sn);
        ao0=vmlaq_s32(ao0,vld1q_s32(&g_cm_oT[n][0]),dn);
        ao1=vmlaq_s32(ao1,vld1q_s32(&g_cm_oT[n][4]),dn);
    }
    mc_fi32 e[8],o[8];
    vst1q_s32(e,vshrq_n_s32(ae0,MC_DCT_Q)); vst1q_s32(e+4,vshrq_n_s32(ae1,MC_DCT_Q));
    vst1q_s32(o,vshrq_n_s32(ao0,MC_DCT_Q)); vst1q_s32(o+4,vshrq_n_s32(ao1,MC_DCT_Q));
    for(int j=0;j<8;++j){ out[2*j]=e[j]; out[2*j+1]=o[j]; }
}
#elif MC_SIMD_AVX2
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[8], d[8];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    __m256i ae=_mm256_set1_epi32(rnd), ao=_mm256_set1_epi32(rnd);
    for(int n=0;n<H;++n){
        ae=_mm256_add_epi32(ae,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_eT[n][0]),_mm256_set1_epi32(s[n])));
        ao=_mm256_add_epi32(ao,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_oT[n][0]),_mm256_set1_epi32(d[n])));
    }
    mc_fi32 e[8],o[8];
    _mm256_storeu_si256((__m256i*)e,_mm256_srai_epi32(ae,MC_DCT_Q));
    _mm256_storeu_si256((__m256i*)o,_mm256_srai_epi32(ao,MC_DCT_Q));
    for(int j=0;j<8;++j){ out[2*j]=e[j]; out[2*j+1]=o[j]; }
}
#else
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[H], d[H];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    mc_fi32 acc[MC_DCT_N]; for(int k=0;k<S;++k) acc[k]=rnd;
    for(int n=0;n<H;++n){ mc_fi32 sn=s[n], dn=d[n];
        for(int k=0;k<S;k+=2) acc[k]+=g_mc_cm[k][n]*sn;
        for(int k=1;k<S;k+=2) acc[k]+=g_mc_cm[k][n]*dn; }
    for(int k=0;k<S;++k) out[k]=acc[k]>>MC_DCT_Q;
}
#endif
// 1D inverse, sparse-aware: skips zero coefficients (most lines have only a few
// nonzero low-frequency entries after dequant). NEON/AVX2 kernels measured
// ~1.6x over the autovectorized scalar form; scalar fallback kept.
#if MC_SIMD_NEON
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1);
    const mc_fi32 *ce=&g_cm_e[0][0], *co=&g_cm_o[0][0];   // one base each: the
    // unrolled loop below indexes with immediates instead of re-materializing
    // adrp+add per table row (objdump finding).
    int32x4_t ae0=vdupq_n_s32(rnd), ae1=vdupq_n_s32(rnd);
    int32x4_t ao0=vdupq_n_s32(0),  ao1=vdupq_n_s32(0);
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve){
            int32x4_t v=vdupq_n_s32(ve);
            ae0=vmlaq_s32(ae0,vld1q_s32(ce+j*8),v);
            ae1=vmlaq_s32(ae1,vld1q_s32(ce+j*8+4),v);
        }
        mc_fi32 vo=in[2*j+1];
        if(vo){
            int32x4_t v=vdupq_n_s32(vo);
            ao0=vmlaq_s32(ao0,vld1q_s32(co+j*8),v);
            ao1=vmlaq_s32(ao1,vld1q_s32(co+j*8+4),v);
        }
    }
    int32x4_t s0=vshrq_n_s32(vaddq_s32(ae0,ao0),MC_DCT_Q);
    int32x4_t s1=vshrq_n_s32(vaddq_s32(ae1,ao1),MC_DCT_Q);
    int32x4_t d0=vshrq_n_s32(vsubq_s32(ae0,ao0),MC_DCT_Q);
    int32x4_t d1=vshrq_n_s32(vsubq_s32(ae1,ao1),MC_DCT_Q);
    vst1q_s32(out,s0); vst1q_s32(out+4,s1);
    int32x4_t r1=vrev64q_s32(d1); r1=vextq_s32(r1,r1,2);   // reverse lanes
    int32x4_t r0=vrev64q_s32(d0); r0=vextq_s32(r0,r0,2);
    vst1q_s32(out+8,r1); vst1q_s32(out+12,r0);
}
#elif MC_SIMD_AVX512
// even+odd accumulators in one zmm: lanes [ae0..ae7 | ao0..ao7].
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1);
    __m512i acc=_mm512_inserti64x4(_mm512_set1_epi32(rnd),_mm256_setzero_si256(),1);
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j], vo=in[2*j+1];
        if(!(ve|vo)) continue;
        __m512i val=_mm512_inserti64x4(_mm512_set1_epi32(ve),_mm256_set1_epi32(vo),1);
        acc=_mm512_add_epi32(acc,_mm512_mullo_epi32(_mm512_load_si512(&g_cm_eo[j][0]),val));
    }
    __m256i ae=_mm512_extracti64x4_epi64(acc,0), ao=_mm512_extracti64x4_epi64(acc,1);
    __m256i sm=_mm256_srai_epi32(_mm256_add_epi32(ae,ao),MC_DCT_Q);
    __m256i d =_mm256_srai_epi32(_mm256_sub_epi32(ae,ao),MC_DCT_Q);
    _mm256_storeu_si256((__m256i*)out,sm);
    __m256i rev=_mm256_shuffle_epi32(d,_MM_SHUFFLE(0,1,2,3));
    rev=_mm256_permute2x128_si256(rev,rev,1);
    _mm256_storeu_si256((__m256i*)(out+8),rev);
}
static inline void mc_dct1d_fwd(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 s[8], d[8];
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
    __m256i ae=_mm256_set1_epi32(rnd), ao=_mm256_set1_epi32(rnd);
    for(int n=0;n<H;++n){
        ae=_mm256_add_epi32(ae,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_eT[n][0]),_mm256_set1_epi32(s[n])));
        ao=_mm256_add_epi32(ao,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_oT[n][0]),_mm256_set1_epi32(d[n])));
    }
    mc_fi32 e[8],o[8];
    _mm256_storeu_si256((__m256i*)e,_mm256_srai_epi32(ae,MC_DCT_Q));
    _mm256_storeu_si256((__m256i*)o,_mm256_srai_epi32(ao,MC_DCT_Q));
    for(int j=0;j<8;++j){ out[2*j]=e[j]; out[2*j+1]=o[j]; }
}
#elif MC_SIMD_AVX2
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1);
    __m256i ae=_mm256_set1_epi32(rnd), ao=_mm256_setzero_si256();
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve) ae=_mm256_add_epi32(ae,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_e[j][0]),_mm256_set1_epi32(ve)));
        mc_fi32 vo=in[2*j+1];
        if(vo) ao=_mm256_add_epi32(ao,_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i*)&g_cm_o[j][0]),_mm256_set1_epi32(vo)));
    }
    __m256i s=_mm256_srai_epi32(_mm256_add_epi32(ae,ao),MC_DCT_Q);
    __m256i d=_mm256_srai_epi32(_mm256_sub_epi32(ae,ao),MC_DCT_Q);
    _mm256_storeu_si256((__m256i*)out,s);
    // reverse d into out[8..15]
    __m256i rev=_mm256_shuffle_epi32(d,_MM_SHUFFLE(0,1,2,3));
    rev=_mm256_permute2x128_si256(rev,rev,1);
    _mm256_storeu_si256((__m256i*)(out+8),rev);
}
#else
static inline void mc_dct1d_inv(const mc_fi32 *restrict in, mc_fi32 *restrict out){
    const mc_fi32 rnd=(mc_fi32)1<<(MC_DCT_Q-1); const int S=MC_DCT_N, H=S/2;
    mc_fi32 acc_e[8], acc_o[8];
    for(int n=0;n<H;++n){ acc_e[n]=rnd; acc_o[n]=0; }
    for(int j=0;j<8;++j){
        mc_fi32 ve=in[2*j];
        if(ve) for(int n=0;n<H;++n) acc_e[n]+=g_cm_e[j][n]*ve;
        mc_fi32 vo=in[2*j+1];
        if(vo) for(int n=0;n<H;++n) acc_o[n]+=g_cm_o[j][n]*vo;
    }
    for(int n=0;n<H;++n){ out[n]=(acc_e[n]+acc_o[n])>>MC_DCT_Q; out[S-1-n]=(acc_e[n]-acc_o[n])>>MC_DCT_Q; }
}
#endif

// cache-blocked rotate (z,y,x)->(x,z,y): dst[(x*S+z)*S+y] = src[(z*S+y)*S+x].
// Per fixed z this is a 16x16 i32 transpose (src rows y, stride S -> dst rows
// x, stride S*S); done as 4x4 register tiles on NEON, scalar tiles otherwise.
#if MC_SIMD_NEON
static inline void mc_rot(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N;
    for(int z=0;z<S;++z){
        const mc_fi32 *sp=src+(size_t)z*S*S;
        mc_fi32 *dp=dst+(size_t)z*S;
        for(int y=0;y<S;y+=4)
        for(int x=0;x<S;x+=4){
            int32x4_t r0=vld1q_s32(sp+(size_t)(y+0)*S+x);
            int32x4_t r1=vld1q_s32(sp+(size_t)(y+1)*S+x);
            int32x4_t r2=vld1q_s32(sp+(size_t)(y+2)*S+x);
            int32x4_t r3=vld1q_s32(sp+(size_t)(y+3)*S+x);
            int32x4_t t0=vtrn1q_s32(r0,r1), t1=vtrn2q_s32(r0,r1);
            int32x4_t t2=vtrn1q_s32(r2,r3), t3=vtrn2q_s32(r2,r3);
            int32x4_t c0=vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t0),vreinterpretq_s64_s32(t2)));
            int32x4_t c1=vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(t1),vreinterpretq_s64_s32(t3)));
            int32x4_t c2=vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t0),vreinterpretq_s64_s32(t2)));
            int32x4_t c3=vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(t1),vreinterpretq_s64_s32(t3)));
            vst1q_s32(dp+(size_t)(x+0)*S*S+y, c0);
            vst1q_s32(dp+(size_t)(x+1)*S*S+y, c1);
            vst1q_s32(dp+(size_t)(x+2)*S*S+y, c2);
            vst1q_s32(dp+(size_t)(x+3)*S*S+y, c3);
        }
    }
}
#else
#define MC_ROT_TILE 8
static inline void mc_rot(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N;
    for(int zt=0; zt<S; zt+=MC_ROT_TILE)
    for(int xt=0; xt<S; xt+=MC_ROT_TILE)
        for(int z=zt; z<zt+MC_ROT_TILE; ++z)
        for(int x=xt; x<xt+MC_ROT_TILE; ++x){
            const mc_fi32 *sp = src + ((size_t)z*S)*S + x;
            mc_fi32 *dp = dst + ((size_t)x*S+z)*S;
            for(int y=0;y<S;++y) dp[y]=sp[(size_t)y*S];
        }
}
#endif
// transform all contiguous lines (skipping all-zero lines), in place or out-of-place.
static inline void mc_lines_fwd_to(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ const mc_fi32 *v=src+(size_t)off*S; mc_fi32 *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0; continue; }
        mc_dct1d_fwd(v,ol); for(int i=0;i<S;++i) o[i]=ol[i]; }
}
static inline void mc_lines_fwd(mc_fi32 *restrict blk){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ mc_fi32 *v=blk+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;} if(!nz) continue;
        mc_dct1d_fwd(v,ol); for(int i=0;i<S;++i) v[i]=ol[i]; }
}
static inline void mc_lines_inv_to(const mc_fi32 *restrict src, mc_fi32 *restrict dst){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ const mc_fi32 *v=src+(size_t)off*S; mc_fi32 *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0; continue; }
        mc_dct1d_inv(v,ol); for(int i=0;i<S;++i) o[i]=ol[i]; }
}
static inline void mc_lines_inv(mc_fi32 *restrict blk){
    const int S=MC_DCT_N; mc_fi32 ol[MC_DCT_N];
    for(int off=0;off<S*S;++off){ mc_fi32 *v=blk+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;} if(!nz) continue;
        mc_dct1d_inv(v,ol); for(int i=0;i<S;++i) v[i]=ol[i]; }
}

// 3D forward/inverse on a 16^3 block (float in/out for the codec's quant path).
// Each pass: transform contiguous lines, then rotate. 3 rotates return to (z,y,x).
typedef struct {
    mc_fi32 in[16*16*16] __attribute__((aligned(MC_DCT_ALIGN)));
    mc_fi32 a [16*16*16] __attribute__((aligned(MC_DCT_ALIGN)));
    mc_fi32 b [16*16*16] __attribute__((aligned(MC_DCT_ALIGN)));
} mc_dct_tls_t;
static void mc_dct3_fwd(mc_dct_tls_t *D, const float *restrict blk, float *restrict coef){
    const int n=MC_DCT_N*MC_DCT_N*MC_DCT_N;
    mc_fi32 *in=D->in, *a=D->a, *b=D->b;
    for(int i=0;i<n;++i) in[i]=(mc_fi32)lrintf(blk[i]);
    mc_lines_fwd_to(in,a); mc_rot(a,b);
    mc_lines_fwd(b);       mc_rot(b,a);
    mc_lines_fwd(a);       mc_rot(a,b);
    for(int i=0;i<n;++i) coef[i]=(float)b[i];
}
static void mc_dct3_inv(mc_dct_tls_t *D, const float *restrict coef, float *restrict blk){
    const int n=MC_DCT_N*MC_DCT_N*MC_DCT_N;
    mc_fi32 *in=D->in, *a=D->a, *b=D->b;
    for(int i=0;i<n;++i) in[i]=(mc_fi32)lrintf(coef[i]);
    mc_lines_inv_to(in,a); mc_rot(a,b);
    mc_lines_inv(b);       mc_rot(b,a);
    mc_lines_inv(a);       mc_rot(a,b);
    for(int i=0;i<n;++i) blk[i]=(float)b[i];
}
// variant taking PREPARED i32 coefficients (decoder fuses dequantization into
// the input conversion) and returning the raw i32 spatial result.
static void mc_dct3_inv_i32(mc_dct_tls_t *D, const mc_fi32 *restrict in, mc_fi32 *restrict out){
    mc_fi32 *a=D->a;
    mc_lines_inv_to(in,a); mc_rot(a,out);
    mc_lines_inv(out);     mc_rot(out,a);
    mc_lines_inv(a);       mc_rot(a,out);
}

#endif

// ============================================================================
// mc_rangecoder.h — CABAC-style binary range coder + DCT coefficient context coder
// (matter-compressor). Adaptive bit model + bypass bits; coefficients coded in
// ascending-frequency scan order with an EOB, per-band significance contexts, and a
// unary+Exp-Golomb magnitude coder.
// ============================================================================
#ifndef MC_RANGECODER_H
#define MC_RANGECODER_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

typedef uint8_t  rc_u8;  typedef int16_t rc_i16; typedef int32_t rc_i32;
typedef uint32_t rc_u32; typedef uint64_t rc_u64;

typedef struct { rc_u8 *buf; size_t cap, len; rc_u64 low; rc_u32 range; rc_u8 cache; rc_u64 cache_size; } rc_enc;
typedef struct { const rc_u8 *buf; size_t len, pos; rc_u32 code, range; } rc_dec;
typedef struct { uint16_t p0; } ctx_t;
static inline void ctx_init(ctx_t *c){ c->p0 = 1u<<11; }
static inline void ctx_init_p(ctx_t *c, uint16_t p0){ c->p0 = p0; }
#define RC_TOP (1u<<24)

// ---- trained context priors -------------------------------------------------
// Per-block contexts reset every block, so without priors every adaptive bin
// starts at p=0.5 and is no better than bypass for the first ~32 bins. These
// tables are trained on PHercParis4 2.4um (fysics-masked) at q in {1,3,6,12}
// via tools/mc_train (build with -DMC_TRAIN to retrain).
// Context classes (training bucket ids):
enum { RCC_SIG=0, RCC_MAG=1, RCC_EOB=2, RCC_MASK=3, RCC_MASKU=4, RCC_MASKA=5, RCC_FLAG=6, RCC_DC=7, RCC_NCLS=8 };
#define RCC_SLOTS 32
#ifdef MC_TRAIN
extern long mc_tr_n[RCC_NCLS][RCC_SLOTS], mc_tr_z[RCC_NCLS][RCC_SLOTS];
#define RC_TRAIN(cls,slot,bit) (mc_tr_n[cls][slot]++, mc_tr_z[cls][slot]+=((bit)==0))
#else
#define RC_TRAIN(cls,slot,bit) ((void)0)
#endif
// priors (p0 = P(bit==0) in 1/4096 units), trained at q=1 and q=12 on
// PHercParis4 2.4um (fysics-masked) via tools/mc_train; rc_prior_build_into()
// interpolates in log2(q) into the ctx's pri[][] (the decoder knows q, so this
// costs no side information). 2048 = untrained slot.
#define RC_NSLOT 32
static const uint16_t RC_PLO[8][RC_NSLOT]={
  /*SIG*/ {115,110,112,144,3866,3522,3330,948,3939,3723,3483,2383,4007,3795,3575,3130,4042,3912,3805,3686,4064,4064,4028,4064,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MAG*/ {1996,1234,948,743,595,539,441,421,333,358,297,305,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*EOB*/ {32,32,32,32,32,32,32,32,32,92,285,2295,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MASK*/ {4025,3064,2995,827,2107,473,576,50,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKU*/ {250,1713,2961,2991,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKA*/ {3936,32,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*FLAG*/ {1023,4064,4064,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*DC*/ {3873,366,1820,2056,2040,2055,2041,2045,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048}
};
static const uint16_t RC_PHI[8][RC_NSLOT]={
  /*SIG*/ {456,335,535,1133,3950,3755,3590,2892,4064,4051,3990,3686,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MAG*/ {2121,1410,1081,863,746,631,569,508,465,431,399,370,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*EOB*/ {59,32,32,32,32,63,92,315,2273,4064,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MASK*/ {4025,3064,2995,827,2107,473,576,50,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKU*/ {250,1713,2961,2991,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*MSKA*/ {3936,32,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*FLAG*/ {1023,4064,4064,4064,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048},
  /*DC*/ {3873,366,1820,2056,2040,2055,2041,2045,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048}
};
// Per-volume prior override (format v6: a trained-prior blob stored in the
// archive replaces the baked corpus tables). Process-global config, set once at
// open before decode threads start; a generation counter forces per-ctx rebuild.
static uint16_t g_plo_ovr[8][RC_NSLOT], g_phi_ovr[8][RC_NSLOT];
static int g_pri_ovr = 0;
static int g_pri_gen = 1;
static void rc_set_priors(const uint16_t *plo, const uint16_t *phi){
    if(plo&&phi){
        memcpy(g_plo_ovr,plo,sizeof g_plo_ovr);
        memcpy(g_phi_ovr,phi,sizeof g_phi_ovr);
        g_pri_ovr=1;
    } else g_pri_ovr=0;
    g_pri_gen++;
}
// Build the per-ctx interpolated prior table from the process-global config.
static void rc_prior_build_into(uint16_t pri[8][RC_NSLOT], float *pri_q, int *pri_seen, float q){
    if(*pri_q==q && *pri_seen==g_pri_gen) return;
    *pri_seen=g_pri_gen;
    float lo=0.0f, hi=3.585f;                       // log2(1) .. log2(12)
    float w=(q<=1.0f)?0.0f:((float)(log(q)/log(2.0))-lo)/(hi-lo);
    if(w<0)w=0; if(w>1)w=1;
    const uint16_t (*tlo)[RC_NSLOT] = g_pri_ovr ? (const uint16_t(*)[RC_NSLOT])g_plo_ovr : RC_PLO;
    const uint16_t (*thi)[RC_NSLOT] = g_pri_ovr ? (const uint16_t(*)[RC_NSLOT])g_phi_ovr : RC_PHI;
    for(int c=0;c<8;++c)for(int s=0;s<RC_NSLOT;++s)
        pri[c][s]=(uint16_t)(tlo[c][s]+(thi[c][s]-tlo[c][s])*w+0.5f);
    *pri_q=q;
}
// Prior-table accessors: pri is the ctx's interpolated uint16_t[8][RC_NSLOT].
#define RC_PRIOR_SIG(pri)   ((pri)[0])
#define RC_PRIOR_MAG(pri)   ((pri)[1])
#define RC_PRIOR_EOB(pri)   ((pri)[2])
#define RC_PRIOR_MASK(pri)  ((pri)[3])
#define RC_PRIOR_MASKU(pri) ((pri)[4])
#define RC_PRIOR_MASKA(pri) ((pri)[5])
#define RC_PRIOR_FLAG(pri)  ((pri)[6])
#define RC_PRIOR_DC(pri)    ((pri)[7])

static void enc_init(rc_enc *e, rc_u8 *buf, size_t cap){ e->buf=buf;e->cap=cap;e->len=0;e->low=0;e->range=0xFFFFFFFFu;e->cache=0;e->cache_size=1; }
static void enc_putbyte(rc_enc *e, rc_u8 b){ if(e->len<e->cap) e->buf[e->len++]=b; else e->len++; }
static void enc_shift_low(rc_enc *e){
    if((rc_u32)(e->low>>32)!=0 || e->low<0xFF000000ull){
        rc_u8 carry=(rc_u8)(e->low>>32);
        do{ enc_putbyte(e,(rc_u8)(e->cache+carry)); e->cache=0xFF; }while(--e->cache_size);
        e->cache=(rc_u8)(e->low>>24);
    }
    e->cache_size++; e->low=(e->low<<8)&0xFFFFFFFFull;
}
static void enc_bit(rc_enc *e, ctx_t *c, int bit){
    rc_u32 r0=(e->range>>12)*c->p0;
    if(bit==0){ e->range=r0; c->p0=(uint16_t)(c->p0+((4096-c->p0)>>4)); }
    else { e->low+=r0; e->range-=r0; c->p0=(uint16_t)(c->p0-(c->p0>>4)); }
    while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; }
}
static void enc_bypass(rc_enc *e, int bit){ e->range>>=1; if(bit) e->low+=e->range; while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; } }
static void enc_flush(rc_enc *e){ for(int i=0;i<5;++i) enc_shift_low(e); }

static void dec_init(rc_dec *d,const rc_u8*buf,size_t len){ d->buf=buf;d->len=len;d->pos=0;d->code=0;d->range=0xFFFFFFFFu; for(int i=0;i<5;++i){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; } }
static int dec_bit(rc_dec *d, ctx_t *c){
    rc_u32 r0=(d->range>>12)*c->p0; int bit;
    if(d->code<r0){ d->range=r0;bit=0; c->p0=(uint16_t)(c->p0+((4096-c->p0)>>4)); }
    else { d->code-=r0; d->range-=r0; bit=1; c->p0=(uint16_t)(c->p0-(c->p0>>4)); }
    while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; }
    return bit;
}
static int dec_bypass(rc_dec *d){ d->range>>=1; int bit=(d->code>=d->range); if(bit)d->code-=d->range; while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; } return bit; }

// batched bypass: k equiprobable bits in one renorm round (bit-compatible
// with k single bypasses only when both sides batch identically — they do).
static void enc_bypass_n(rc_enc *e, rc_u32 v, int k){
    while(k>16){ enc_bypass_n(e,(v>>(k-16))&0xFFFF,16); k-=16; }
    if(!k) return;
    e->range>>=k;
    e->low+=(rc_u64)(v&((1u<<k)-1))*e->range;
    while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; }
}
static rc_u32 dec_bypass_n(rc_dec *d, int k){
    rc_u32 v=0;
    while(k>16){ v=(v<<16)|dec_bypass_n(d,16); k-=16; }
    if(!k) return v;
    d->range>>=k;
    rc_u32 q=d->code/d->range;
    rc_u32 m=(1u<<k)-1; if(q>m)q=m;
    d->code-=q*d->range;
    while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; }
    return (v<<k)|q;
}

// Exp-Golomb in bypass bits (order 0), v >= 0.
static void enc_eg(rc_enc*e,rc_u32 v){
    rc_u32 nb=0,t=v+1; while(t>1){t>>=1;nb++;}
    for(rc_u32 i=0;i<nb;++i)enc_bypass(e,1); enc_bypass(e,0);
    if(nb) enc_bypass_n(e,(v+1)&((1u<<nb)-1),(int)nb);
}
static rc_u32 dec_eg(rc_dec*d){
    rc_u32 nb=0; while(dec_bypass(d))nb++;
    if(!nb) return 0;
    return ((1u<<nb)|dec_bypass_n(d,(int)nb))-1;
}

// --- coefficient context coder (block size S) ---
// Ascending-band scan with adaptive last-sig (EOB), significance conditioned on
// (band, recent significance density), and an adaptive-unary + Exp-Golomb
// magnitude ladder. Group-skip flags and bypass Rice remainders were tried
// (HEVC-style) and measured WORSE on scroll data: the EOB already truncates the
// sparse tail, and the skewed magnitude distribution wants adaptive bins, not
// bypass remainders.
#define NB_BANDS 8       // L1-frequency band buckets
#define MAGCTX   12      // unary magnitude ladder contexts
typedef struct {
    ctx_t sig[NB_BANDS*4];     // significance: band x min(recent sig count,3)
    ctx_t mag[MAGCTX];         // unary magnitude ladder (band-conditioning the
                               // first rungs was measured ratio-neutral: per-
                               // block adaptation already learns the block's
                               // own magnitude distribution)
} atom_ctx;
static void atom_ctx_init(atom_ctx *a, const uint16_t (*pri)[RC_NSLOT]){
    for(int i=0;i<NB_BANDS*4;++i) ctx_init_p(&a->sig[i],RC_PRIOR_SIG(pri)[i]);
    for(int i=0;i<MAGCTX;++i)     ctx_init_p(&a->mag[i],RC_PRIOR_MAG(pri)[i]);
}
static void enc_magnitude(rc_enc*e,atom_ctx*ac,rc_u32 m){
    ctx_t*mag=ac->mag; rc_u32 v=m-1,k=0;
    while(k<(rc_u32)(MAGCTX-1)&&v>0){ RC_TRAIN(RCC_MAG,k,1); enc_bit(e,&mag[k],1); v-=1;k++; if(v==0){RC_TRAIN(RCC_MAG,k,0); enc_bit(e,&mag[k],0);return;} }
    if(v==0){ RC_TRAIN(RCC_MAG,k,0); enc_bit(e,&mag[k],0); return; }
    RC_TRAIN(RCC_MAG,k,1); enc_bit(e,&mag[k],1); rc_u32 x=v,nbits=0,tt=x+1; while(tt>1){tt>>=1;nbits++;}
    for(rc_u32 i=0;i<nbits;++i)enc_bypass(e,1); enc_bypass(e,0);
    if(nbits) enc_bypass_n(e,(x+1)&((1u<<nbits)-1),(int)nbits);
}
static rc_u32 dec_magnitude(rc_dec*d,atom_ctx*ac){
    ctx_t*mag=ac->mag; rc_u32 v=0,k=0;
    while(k<(rc_u32)(MAGCTX-1)){ if(dec_bit(d,&mag[k])){v+=1;k++;} else return v+1; }
    if(!dec_bit(d,&mag[k])) return v+1;
    rc_u32 nbits=0; while(dec_bypass(d))nbits++;
    rc_u32 x = nbits ? ((1u<<nbits)|dec_bypass_n(d,(int)nbits))-1 : 0;
    return v+x+1;
}

// per-size ascending-L1-frequency scan tables, built lazily (indexed by log2 S).
// Build is serialized: concurrent encode workers used to race the ready flag,
// double-build, and leak the losing table (caught by LeakSanitizer).
static uint16_t *g_scanS[6]; static _Atomic int g_scanS_ready[6];
static pthread_mutex_t g_scanS_mu = PTHREAD_MUTEX_INITIALIZER;
static int scanS_cmp_S;
static int scanS_cmp(const void*pa,const void*pb){
    rc_u32 a=*(const rc_u32*)pa,b=*(const rc_u32*)pb; int S=scanS_cmp_S;
    rc_u32 fa=(a/(S*S))+((a/S)%S)+(a%S), fb=(b/(S*S))+((b/S)%S)+(b%S);
    if(fa!=fb) return (int)fa-(int)fb; return (int)a-(int)b;
}
static void scanS_build(int S){
    int l=0,t=S; while(t>1){t>>=1;l++;}
    if(atomic_load_explicit(&g_scanS_ready[l],memory_order_acquire)) return;
    pthread_mutex_lock(&g_scanS_mu);
    if(!atomic_load_explicit(&g_scanS_ready[l],memory_order_relaxed)){
        int n=S*S*S; rc_u32 *ord=malloc(n*sizeof(rc_u32));
        uint16_t *tab=ord?malloc(n*sizeof(uint16_t)):NULL;
        if(!ord||!tab){ free(ord); free(tab); pthread_mutex_unlock(&g_scanS_mu); return; }
        for(int i=0;i<n;++i)ord[i]=i;
        scanS_cmp_S=S; qsort(ord,n,sizeof(rc_u32),scanS_cmp);
        for(int i=0;i<n;++i)tab[i]=(uint16_t)ord[i];
        free(ord);
        g_scanS[l]=tab;
        atomic_store_explicit(&g_scanS_ready[l],1,memory_order_release);
    }
    pthread_mutex_unlock(&g_scanS_mu);
}
static inline int band_of_S(rc_u32 idx,int S){
    rc_u32 cz=idx/(S*S),cy=(idx/S)%S,cx=idx%S, freq=cz+cy+cx;
    int b=(int)(freq*NB_BANDS/(3u*S)); if(b>=NB_BANDS)b=NB_BANDS-1; return b;
}
// bit-length of eob (well-predicted: most blocks have similar sparsity), then the
// low bits in bypass. v in [0, n]; prefix k = MSB position + 1 (0 -> v==0).
#define EOB_CTX 14
typedef struct { ctx_t pfx[EOB_CTX]; } eob_ctx;
static void eob_ctx_init(eob_ctx*c, const uint16_t (*pri)[RC_NSLOT]){ for(int i=0;i<EOB_CTX;++i) ctx_init_p(&c->pfx[i],RC_PRIOR_EOB(pri)[i]); }
static void enc_eob(rc_enc*e,eob_ctx*c,rc_u32 v,int n){
    int kmax=0; while((1u<<kmax)<=(rc_u32)n) kmax++;          // 13 for n=4096
    int k=0; while((1u<<k)<=v) k++;                            // MSB+1 (0 for v=0)
    for(int i=0;i<k;++i){ RC_TRAIN(RCC_EOB,i,1); enc_bit(e,&c->pfx[i],1); }
    if(k<kmax){ RC_TRAIN(RCC_EOB,k,0); enc_bit(e,&c->pfx[k],0); }
    if(k>1) enc_bypass_n(e,v&((1u<<(k-1))-1),k-1);             // suffix below the MSB
}
static rc_u32 dec_eob(rc_dec*d,eob_ctx*c,int n){
    int kmax=0; while((1u<<kmax)<=(rc_u32)n) kmax++;
    int k=0; while(k<kmax && dec_bit(d,&c->pfx[k])) k++;
    if(k==0) return 0;
    if(k==1) return 1;
    return (1u<<(k-1))|dec_bypass_n(d,k-1);
}

// encode/decode quantized levels q[S^3] (raster).
static void enc_block_coefs(rc_enc*e,const rc_i16*q,int S, const uint16_t (*pri)[RC_NSLOT]){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac,pri);
    eob_ctx ec; eob_ctx_init(&ec,pri);
    rc_u32 eob=0; for(rc_u32 p=n;p-->0;){ if(q[scan[p]]!=0){eob=p+1;break;} }
    enc_eob(e,&ec,eob,n);
    rc_u32 hist=0;                                  // last 16 sig decisions
    for(rc_u32 p=0;p<eob;++p){
        rc_u32 idx=scan[p]; int b=band_of_S(idx,S); rc_i16 v=q[idx];
        int dens=__builtin_popcount(hist&0xFFFFu); dens=dens<3?dens:3;
        if(p!=eob-1){                               // last sig position is nonzero by definition
            int sctx=b*4+dens;
            RC_TRAIN(RCC_SIG,sctx,v!=0); enc_bit(e,&ac.sig[sctx],v!=0);
        }
        hist=(hist<<1)|(v!=0);
        if(!v) continue;
        rc_u32 m=(rc_u32)(v<0?-v:v);
        enc_magnitude(e,&ac,m);
        enc_bypass(e,v<0?1:0);
    }
}
// decodes levels; also reports the per-axis extent of nonzero coefficients
// (ext[0..2] = max z,y,x; -1 if none) so the inverse DCT can skip empty space.
static void dec_block_coefs_ext(rc_dec*d,rc_i16*q,int S,int ext[3], const uint16_t (*pri)[RC_NSLOT]){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac,pri); memset(q,0,n*sizeof(rc_i16));
    eob_ctx ec; eob_ctx_init(&ec,pri);
    rc_u32 eob=dec_eob(d,&ec,n); if(eob>(rc_u32)n)eob=n;
    rc_u32 hist=0;
    int ez=-1,ey=-1,ex=-1;
    for(rc_u32 p=0;p<eob;++p){
        rc_u32 idx=scan[p]; int b=band_of_S(idx,S);
        int dens=__builtin_popcount(hist&0xFFFFu); dens=dens<3?dens:3;
        int sig;
        if(p==eob-1) sig=1;
        else { int sctx=b*4+dens; sig=dec_bit(d,&ac.sig[sctx]); }
        hist=(hist<<1)|sig;
        if(!sig) continue;
        rc_u32 m=dec_magnitude(d,&ac);
        int neg=dec_bypass(d);
        q[idx]=(rc_i16)(neg?-(rc_i32)m:(rc_i32)m);
        int cz=(int)(idx/(rc_u32)(S*S)), cy=(int)((idx/(rc_u32)S)%(rc_u32)S), cx=(int)(idx%(rc_u32)S);
        if(cz>ez)ez=cz; if(cy>ey)ey=cy; if(cx>ex)ex=cx;
    }
    ext[0]=ez; ext[1]=ey; ext[2]=ex;
}
static void dec_block_coefs(rc_dec*d,rc_i16*q,int S, const uint16_t (*pri)[RC_NSLOT]){
    int ext[3]; dec_block_coefs_ext(d,q,S,ext,pri);
}
#endif

// mc_xxhash.h — minimal XXH64 (Yann Collet's xxHash, 64-bit variant) for
// chunk-blob integrity checksums. Public-domain-style reimplementation.
#include <stdint.h>
#include <string.h>

#define XXP1 0x9E3779B185EBCA87ULL
#define XXP2 0xC2B2AE3D27D4EB4FULL
#define XXP3 0x165667B19E3779F9ULL
#define XXP4 0x85EBCA77C2B2AE63ULL
#define XXP5 0x27D4EB2F165667C5ULL
static inline uint64_t xx_rotl(uint64_t x,int r){ return (x<<r)|(x>>(64-r)); }
static inline uint64_t xx_round(uint64_t acc,uint64_t in){ acc+=in*XXP2; acc=xx_rotl(acc,31); return acc*XXP1; }
static inline uint64_t xx_merge(uint64_t acc,uint64_t v){ acc^=xx_round(0,v); return acc*XXP1+XXP4; }
static inline uint64_t xx_read64(const uint8_t*p){ uint64_t v; memcpy(&v,p,8); return v; }
static inline uint32_t xx_read32(const uint8_t*p){ uint32_t v; memcpy(&v,p,4); return v; }
static uint64_t mc_xxh64(const void *data, size_t len, uint64_t seed){
    const uint8_t *p=(const uint8_t*)data, *end=p+len;
    uint64_t h;
    if(len>=32){
        uint64_t v1=seed+XXP1+XXP2, v2=seed+XXP2, v3=seed, v4=seed-XXP1;
        const uint8_t *limit=end-32;
        do{
            v1=xx_round(v1,xx_read64(p));    p+=8;
            v2=xx_round(v2,xx_read64(p));    p+=8;
            v3=xx_round(v3,xx_read64(p));    p+=8;
            v4=xx_round(v4,xx_read64(p));    p+=8;
        }while(p<=limit);
        h=xx_rotl(v1,1)+xx_rotl(v2,7)+xx_rotl(v3,12)+xx_rotl(v4,18);
        h=xx_merge(h,v1); h=xx_merge(h,v2); h=xx_merge(h,v3); h=xx_merge(h,v4);
    } else h=seed+XXP5;
    h+=(uint64_t)len;
    while(p+8<=end){ h^=xx_round(0,xx_read64(p)); h=xx_rotl(h,27)*XXP1+XXP4; p+=8; }
    if(p+4<=end){ h^=(uint64_t)xx_read32(p)*XXP1; h=xx_rotl(h,23)*XXP2+XXP3; p+=4; }
    while(p<end){ h^=(*p)*XXP5; h=xx_rotl(h,11)*XXP1; ++p; }
    h^=h>>33; h*=XXP2; h^=h>>29; h*=XXP3; h^=h>>32;
    return h;
}

// ============================================================================
// mc_codec.c — matter-compressor block codec implementation. See mc_codec.h.
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N3 (MC_BLK*MC_BLK*MC_BLK)
#define MC_GRID3_F 4096

// Per-operation codec scratch, formerly a bundle of _Thread_local globals. The
// dynamic-TLS model (PIC .so) compiled every _Thread_local access to a
// __tls_get_addr CALL (~14% of decode CPU); scattered through the hot functions
// those calls also forced the range-coder state to spill to the stack around
// each one (objdump: 8 blr + 102 sp-stores in mc_dec_block). Folding everything
// into one heap-allocated mc_codec_ctx passed by pointer turns those into plain
// member loads off one register; hot loops run call-free with the coder state
// held in registers. One ctx is owned per worker thread (decode/encode/cache
// pools each allocate their own), so concurrent operations stay race-free.
typedef struct {
    // decode
    mc_u8  air[N3];
    rc_i16 ql[N3];
    float  coef[N3], blk[N3];
    mc_fi32 qin[N3] __attribute__((aligned(64)));
    mc_fi32 qout[N3] __attribute__((aligned(64)));
    // encode
    float  eblk[N3], ecoef[N3];
    rc_u8  scratch[N3*4+1024];
    uint16_t cpos[N3]; mc_i32 cdel[N3];
    float  rcoef[N3], rblk[N3];
} mc_tls_t;

// red-black SOR air-fill scratch (18^3 padded). Block-independent PM/CNT tables
// are built once per ctx (pm_init guard).
enum { MC_FILL_PS=MC_BLK+2, MC_FILL_PN=MC_FILL_PS*MC_FILL_PS*MC_FILL_PS };
typedef struct {
    float P[MC_FILL_PN];          // pads stay 0
    float W6[2][MC_FILL_PN];      // pads stay 0
    float PM[MC_FILL_PN], CNT[MC_FILL_PN];
    int   pm_init;
} mc_fill_tls_t;

// chunk-level encode scratch (encode_chunk_blob / append paths). MC_GRID3 (=4096,
// the 16^3 block grid) is #defined later in the archive section; use the literal
// here since this struct precedes it.
#define MC_CTX_GRID3 4096
typedef struct {
    mc_buf   tmp;                 // concatenated block payloads (growable)
    uint8_t  frac[MC_CTX_GRID3];
    uint8_t  vox[N3];
    uint16_t lens16[MC_CTX_GRID3];
    uint8_t  fmap[MC_CTX_GRID3/2+64];
} mc_chunk_tls_t;

// Explicit per-thread codec context (replaces ~30 _Thread_local globals).
typedef struct mc_codec_ctx {
    float    quality;
    int      max_err;             // 0 = corrections off
    float    step_tab[N3];        // quality*hf_weight per coefficient
    float    rstep_tab[N3];       // 1/step: quant uses mul, not div
    float    step_q;              // cache guard for step tables
    uint16_t pri[8][RC_NSLOT];    // interpolated trained priors for this q
    float    pri_q;               // cache guard for priors
    int      pri_seen;            // priors generation seen
    mc_tls_t      scratch;        // hot enc/dec block scratch
    mc_dct_tls_t  dct;            // DCT line buffers
    mc_fill_tls_t fill;           // air-fill SOR scratch
    mc_chunk_tls_t chunk;         // chunk-blob encode scratch
} mc_codec_ctx;

static void step_tab_build(mc_codec_ctx *C);

mc_codec_ctx *mc_codec_ctx_new(void){
    mc_codec_ctx *C = calloc(1,sizeof *C);
    if(!C) return NULL;
    C->quality = 8.0f;
    C->max_err = 0;
    C->step_q  = -1.0f;
    C->pri_q   = -1.0f;
    C->pri_seen = 0;
    step_tab_build(C);
    return C;
}
void mc_codec_ctx_free(mc_codec_ctx *C){
    if(!C) return;
    free(C->chunk.tmp.p);
    free(C);
}
void  mc_codec_ctx_set_quality(mc_codec_ctx *C, float q){ if(C){ C->quality=q; step_tab_build(C); } }
float mc_codec_ctx_get_quality(mc_codec_ctx *C){ return C?C->quality:0.0f; }
void  mc_codec_ctx_set_max_error(mc_codec_ctx *C, int tau){ if(C) C->max_err = tau<0?0:tau; }
int   mc_codec_ctx_get_max_error(mc_codec_ctx *C){ return C?C->max_err:0; }

// ---- calibrated preset ladder (see header; bench/RESULTS.md for the data) --
static const struct { float q; int tau; const char *name; } g_presets[8] = {
    { 0.5f,   1, "archival"  },
    { 0.5f,   2, "master"    },
    { 1.0f,   4, "high"      },
    { 2.5f,   8, "balanced"  },
    { 6.0f,  16, "streaming" },
    {16.0f,  32, "fast"      },
    {32.0f,  64, "ultrafast" },
    {64.0f, 128, "preview"   },
};
float mc_preset_quality(mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) p=MC_PRESET_STREAMING;
    return g_presets[p].q;
}
int mc_preset_tau(mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) p=MC_PRESET_STREAMING;
    return g_presets[p].tau;
}
const char *mc_preset_name(mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) return "?";
    return g_presets[p].name;
}
float mc_apply_preset(mc_codec_ctx *C, mc_preset p){
    if((unsigned)p>=MC_PRESET_COUNT) p=MC_PRESET_STREAMING;
    mc_codec_ctx_set_quality(C,g_presets[p].q);
    mc_codec_ctx_set_max_error(C,g_presets[p].tau);
    return g_presets[p].q;
}
void  mc_codec_init(void){ mc_dct_init(); }
void  mc_codec_set_priors(const uint16_t *plo, const uint16_t *phi){ rc_set_priors(plo,phi); }

void mc_buf_put(mc_buf *b, const void *s, size_t n){
    if(b->len+n > b->cap){ size_t nc=b->cap?b->cap*2:1<<16; while(nc<b->len+n)nc*=2;
        void *tmp=realloc(b->p,nc); if(!tmp) return; b->p=tmp; b->cap=nc; }
    memcpy(b->p+b->len,s,n); b->len+=n;
}

// frozen quant: dead-zone, step = quality*(1+L1freq)^MC_HF_EXP
static inline float hf_weight(int cz,int cy,int cx){ return powf(1.0f+(float)(cz+cy+cx), MC_HF_EXP); }
static void step_tab_build(mc_codec_ctx *C){
    rc_prior_build_into(C->pri,&C->pri_q,&C->pri_seen,C->quality);
    if(C->step_q==C->quality) return;
    for(int cz=0;cz<MC_BLK;++cz)for(int cy=0;cy<MC_BLK;++cy)for(int cx=0;cx<MC_BLK;++cx){
        int i=(cz*MC_BLK+cy)*MC_BLK+cx;
        C->step_tab[i]=C->quality*hf_weight(cz,cy,cx);
        C->rstep_tab[i]=1.0f/C->step_tab[i];
    }
    C->step_q=C->quality;
}
static inline mc_i32 quant_one(float c, float step){
    float dz=MC_DZ_FRAC*step, a=fabsf(c); mc_i32 lv=0;
    if(a>=dz) lv=(mc_i32)((a-dz)/step+1.0f);
    return c<0?-lv:lv;
}
static inline float deq_one(mc_i32 lv, float step){
    if(!lv) return 0.0f;
    float a=(float)(lv<0?-lv:lv); float r=(a-1.0f)*step+MC_DZ_FRAC*step+0.40f*step;
    return lv<0?-r:r;
}

// block-mask surface coder: 3-neighbor (z-1,y-1,x-1) context bit coder over the
// block's 16^3 air mask (air = vox==0). Out-of-block neighbors read as 0. Codes
// into the block's single range-coder stream (shared with the coefficients).
// Two-level mask: per 4^3 subcube a class (uniform-air / uniform-material /
// mixed; ~2 adaptive bins), then per-voxel 3-neighbor context bins only inside
// mixed subcubes. Subcubes and voxels both scan in raster order so the causal
// (z-1,y-1,x-1) context always reads already-coded mask values. Cuts mask bins
// ~3-6x on boundary blocks (most blocks of a masked scroll volume).
#define MSUB 4
static void enc_blockmask(rc_enc *e, const mc_u8 *vox, const uint16_t (*pri)[RC_NSLOT]){
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init_p(&ctx[i],RC_PRIOR_MASK(pri)[i]);
    ctx_t cu[4];  for(int i=0;i<4;++i) ctx_init_p(&cu[i],RC_PRIOR_MASKU(pri)[i]);
    ctx_t ca[2];  for(int i=0;i<2;++i) ctx_init_p(&ca[i],RC_PRIOR_MASKA(pri)[i]);
    const int S=MC_BLK, G=S/MSUB;
    mc_u8 air[N3];
    mc_u8 sc[4*4*4];                       // subcube class: 0=material,1=air,2=mixed
    for(int sz=0;sz<G;++sz)for(int sy=0;sy<G;++sy)for(int sx=0;sx<G;++sx){
        int si=(sz*G+sy)*G+sx;
        int nair_s=0;
        for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x)
            nair_s += !vox[((sz*MSUB+z)*S+(sy*MSUB+y))*S+(sx*MSUB+x)];
        int uni = (nair_s==0 || nair_s==MSUB*MSUB*MSUB);
        int nmix=0, nairn=0, nn=0;
        if(sz){ int c=sc[si-G*G]; nn++; nmix+=c==2; nairn+=c==1; }
        if(sy){ int c=sc[si-G];   nn++; nmix+=c==2; nairn+=c==1; }
        if(sx){ int c=sc[si-1];   nn++; nmix+=c==2; nairn+=c==1; }
        int uctx=nmix<3?nmix:3, actx=nairn?1:0;
        RC_TRAIN(RCC_MASKU,uctx,uni); enc_bit(e,&cu[uctx],uni);
        if(uni){
            int isair = nair_s>0;
            RC_TRAIN(RCC_MASKA,actx,isair); enc_bit(e,&ca[actx],isair);
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x)
                air[((sz*MSUB+z)*S+(sy*MSUB+y))*S+(sx*MSUB+x)]=(mc_u8)isair;
        } else {
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x){
                int gz=sz*MSUB+z, gy=sy*MSUB+y, gx=sx*MSUB+x;
                int i=(gz*S+gy)*S+gx;
                int a=!vox[i];
                int nz_= gz?air[i-S*S]:0, ny_= gy?air[i-S]:0, nx_= gx?air[i-1]:0;
                int cc=(nz_<<2)|(ny_<<1)|nx_;
                RC_TRAIN(RCC_MASK,cc,a);
                enc_bit(e,&ctx[cc],a);
                air[i]=(mc_u8)a;
            }
        }
        sc[si]=(mc_u8)(uni? (nair_s>0?1:0) : 2);
    }
}
static void dec_blockmask(rc_dec *d, mc_u8 *air, const uint16_t (*pri)[RC_NSLOT]){
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init_p(&ctx[i],RC_PRIOR_MASK(pri)[i]);
    ctx_t cu[4];  for(int i=0;i<4;++i) ctx_init_p(&cu[i],RC_PRIOR_MASKU(pri)[i]);
    ctx_t ca[2];  for(int i=0;i<2;++i) ctx_init_p(&ca[i],RC_PRIOR_MASKA(pri)[i]);
    const int S=MC_BLK, G=S/MSUB;
    mc_u8 sc[4*4*4];
    for(int sz=0;sz<G;++sz)for(int sy=0;sy<G;++sy)for(int sx=0;sx<G;++sx){
        int si=(sz*G+sy)*G+sx;
        int nmix=0, nairn=0;
        if(sz){ int c=sc[si-G*G]; nmix+=c==2; nairn+=c==1; }
        if(sy){ int c=sc[si-G];   nmix+=c==2; nairn+=c==1; }
        if(sx){ int c=sc[si-1];   nmix+=c==2; nairn+=c==1; }
        int uctx=nmix<3?nmix:3, actx=nairn?1:0;
        int uni=dec_bit(d,&cu[uctx]);
        int isair_u=0;
        if(uni){
            int isair=dec_bit(d,&ca[actx]); isair_u=isair;
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x)
                air[((sz*MSUB+z)*S+(sy*MSUB+y))*S+(sx*MSUB+x)]=(mc_u8)isair;
        } else {
            for(int z=0;z<MSUB;++z)for(int y=0;y<MSUB;++y)for(int x=0;x<MSUB;++x){
                int gz=sz*MSUB+z, gy=sy*MSUB+y, gx=sx*MSUB+x;
                int i=(gz*S+gy)*S+gx;
                int nz_= gz?air[i-S*S]:0, ny_= gy?air[i-S]:0, nx_= gx?air[i-1]:0;
                air[i]=(mc_u8)dec_bit(d,&ctx[(nz_<<2)|(ny_<<1)|nx_]);
            }
        }
        sc[si]=(mc_u8)(uni? (isair_u?1:0) : 2);
    }
}

// ---- decode-side deblocking (optional, no format change) ---------------------
// Clamped 2-tap filter across every 16-aligned block face, per axis. Gated on
// a quality-scaled flatness test (only smooth true block seams, keep edges),
// and never touches air (0) voxels.
static inline void mc_db_pair(mc_u8 *p1, mc_u8 *p0, mc_u8 *q0, mc_u8 *q1, int beta, int tc){
    int a=*p0, b=*q0;
    if(!a||!b) return;
    int d=b-a; if(d<0?-d>=beta:d>=beta) return;            // real edge: keep
    int pp=*p1?*p1:a, qq=*q1?*q1:b;
    int delta=(4*(b-a)+(pp-qq)+4)>>3;
    if(delta>tc)delta=tc; if(delta<-tc)delta=-tc;
    int na=a+delta, nb=b-delta;
    if(na<1)na=1; if(na>255)na=255; if(nb<1)nb=1; if(nb>255)nb=255;
    *p0=(mc_u8)na; *q0=(mc_u8)nb;
}
void mc_deblock(mc_u8 *v, int nz, int ny, int nx, float quality){
    int beta=(int)(3.0f*quality+6.0f);                     // flatness gate
    int tc  =(int)(0.5f*quality+1.0f);                     // max correction
    size_t sy=(size_t)nx, szp=(size_t)ny*nx;
    for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)             // X faces
        for(int x=MC_BLK;x<nx;x+=MC_BLK){
            mc_u8 *p=v+(size_t)z*szp+(size_t)y*sy+x;
            mc_db_pair(p-2,p-1,p,p+1,beta,tc);
        }
    for(int z=0;z<nz;++z)for(int y=MC_BLK;y<ny;y+=MC_BLK)  // Y faces
        for(int x=0;x<nx;++x){
            mc_u8 *p=v+(size_t)z*szp+(size_t)y*sy+x;
            mc_db_pair(p-2*sy,p-sy,p,p+sy,beta,tc);
        }
    for(int z=MC_BLK;z<nz;z+=MC_BLK)for(int y=0;y<ny;++y)  // Z faces
        for(int x=0;x<nx;++x){
            mc_u8 *p=v+(size_t)z*szp+(size_t)y*sy+x;
            mc_db_pair(p-2*szp,p-szp,p,p+szp,beta,tc);
        }
}

// ---- per-chunk material-fraction map (4096 nibbles) -------------------------
// Smooth field: each nibble coded as 4 adaptive bins conditioned on the
// previous nibble bucket (0 / 1-14 / 15). ~0.1-0.3% of chunk bytes.
uint32_t mc_enc_fracmap(const uint8_t *frac, uint8_t *out, size_t cap){
    rc_enc e; enc_init(&e,out,cap);
    ctx_t cx[3][4]; for(int b=0;b<3;++b)for(int i=0;i<4;++i) ctx_init(&cx[b][i]);
    int prev=0;
    for(int i=0;i<MC_GRID3_F;++i){
        int v=frac[i]&15, pb=prev==0?0:prev==15?2:1;
        for(int b=3;b>=0;--b) enc_bit(&e,&cx[pb][b],(v>>b)&1);
        prev=v;
    }
    enc_flush(&e);
    return (uint32_t)e.len;
}
void mc_dec_fracmap(const uint8_t *in, uint32_t len, uint8_t *frac){
    rc_dec d; dec_init(&d,in,len);
    ctx_t cx[3][4]; for(int b=0;b<3;++b)for(int i=0;i<4;++i) ctx_init(&cx[b][i]);
    int prev=0;
    for(int i=0;i<MC_GRID3_F;++i){
        int v=0, pb=prev==0?0:prev==15?2:1;
        for(int b=3;b>=0;--b) v|=dec_bit(&d,&cx[pb][b])<<b;
        frac[i]=(uint8_t)v; prev=v;
    }
}

// block payload layout: ONE range-coded stream, nothing else. The stream starts
// with [mixed bit][corr bit][dc 8 bits] all context-coded with
// trained priors (the old raw dc+flags bytes were ~5% of an average payload),
// then the mask bins (if mixed), the coefficients, and the corrections. flags bit0 =
// mixed block; the stream carries the mask bins (if mixed) then the coefficients.
// One stream = one flush (~5B) instead of two streams + a 2B mask length.
int mc_enc_block(mc_codec_ctx *C, const mc_u8 *vox, mc_buf *out, uint32_t *len_out){
    int n=N3, any=0;
    mc_tls_t *T=&C->scratch;
    step_tab_build(C);
    float *blk=T->eblk, *coef=T->ecoef;
    long sum=0,cnt=0;
#if MC_SIMD_NEON
    {   uint32x4_t s32=vdupq_n_u32(0), c32=vdupq_n_u32(0);
        for(int i=0;i<n;i+=16){
            uint8x16_t v=vld1q_u8(vox+i);
            s32=vpadalq_u16(s32,vpaddlq_u8(v));
            uint8x16_t one=vminq_u8(v,vdupq_n_u8(1));
            c32=vpadalq_u16(c32,vpaddlq_u8(one));
        }
        sum=(long)vaddvq_u32(s32); cnt=(long)vaddvq_u32(c32);
        any=cnt>0;
    }
#else
    // branchless so gcc auto-vectorizes (the old guarded sum/cnt loop was
    // scalar and ~6% of encode on x86)
    {   int s_=0, c_=0;
        for(int i=0;i<n;++i){ s_+=vox[i]; c_+=vox[i]!=0; }
        sum=s_; cnt=c_; any=c_>0;
    }
#endif
    if(!any||!cnt){ *len_out=0; return 0; }
    int dc = (int)((sum+cnt/2)/cnt);                  // DC over material only
    int nair = n-(int)cnt;                            // air = vox==0
#if MC_SIMD_NEON
    {   int16x8_t vdc=vdupq_n_s16((int16_t)dc);
        for(int i=0;i<n;i+=16){
            uint8x16_t v=vld1q_u8(vox+i);
            uint8x16_t nz=vtstq_u8(v,v);
            int16x8_t lo=vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
            int16x8_t hi=vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v)));
            lo=vsubq_s16(lo,vdc); hi=vsubq_s16(hi,vdc);
            int16x8_t mlo=vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(nz)));
            int16x8_t mhi=vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(nz)));
            // mask: 0x00FF per nonzero after widen -> turn into all-ones/all-zero
            mlo=vreinterpretq_s16_u16(vtstq_u16(vreinterpretq_u16_s16(mlo),vreinterpretq_u16_s16(mlo)));
            mhi=vreinterpretq_s16_u16(vtstq_u16(vreinterpretq_u16_s16(mhi),vreinterpretq_u16_s16(mhi)));
            lo=vandq_s16(lo,mlo); hi=vandq_s16(hi,mhi);   // air -> 0 (== dc-dc)
            vst1q_f32(blk+i   ,vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
            vst1q_f32(blk+i+4 ,vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
            vst1q_f32(blk+i+8 ,vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
            vst1q_f32(blk+i+12,vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
        }
    }
#else
    for(int i=0;i<n;++i) blk[i]=(float)((vox[i]?vox[i]:dc)-dc);
#endif
    // harmonic air-fill: relax air voxels toward the 6-neighbor mean (material
    // fixed) so the masked region carries no spurious DCT energy. Perf on real
    // masked-scroll exports showed the original raster-order Gauss-Seidel/SOR
    // over an air-voxel index list was the #1 hot spot of mc_enc_block (~31%
    // of export compute): a strictly serial scalar dependency chain. The fill
    // only shapes values UNDER the air mask — decode forces them to 0 — so its
    // exact values are free to change slightly; only encode speed and archive
    // size matter. Rewritten as: coarse 4^3 seed + RED-BLACK SOR in a dense,
    // branch-free, auto-vectorizing form (see below).
    // Measured (8x 256^3 mixed material/air chunks of a real masked scroll,
    // q=8, best-of-5 process-CPU time incl. the vectorized stats/quant loops
    // above/below): encode 0.345s -> 0.253s (-26.7%), archive size
    // 1126163 -> 1126019 bytes (-0.013%), material max-abs-diff unchanged
    // (41), air voxels still decode to exactly 0.
    if(nair>0){
        // CROSS-ISA DETERMINISM: the floats computed here feed the DCT, so
        // they must round identically on every target. The build is strict
        // IEEE (no -ffast-math — see CMakeLists); under fast-math the
        // per-target reassociation/reciprocal choices in these loops broke
        // bitstream identity (caught by CI), for zero measured speedup.
#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif
        const int S=MC_BLK;
        // (b) skip the fine SOR sweeps on nearly-pure blocks (<5% or >95% air):
        // the coarse 4^3 seed already lands within quantization noise there
        // (thin slivers / almost-all-masked blocks), so refinement is an
        // invisible cost.
        int do_fine = (nair >= n/20) && (nair <= n - n/20);
        // Coarse-to-fine init: solve the fill on the 4^3 subcube grid first
        // (each cell = mean of its material voxels, air cells relaxed), then
        // seed fine air voxels from their cell before the fine SOR sweeps.
        // Lands much closer than a flat dc start, so few sweeps converge.
        // Accumulation runs per (z,y) row with 4-wide unrolled segment sums
        // (SLP-vectorizable; air contributes 0 to the sum because blk[]==0
        // there) — no per-voxel div/mod or branches.
        {
            float cs[64]; int cm[64]; const int G=4;
            for(int c=0;c<64;++c){ cs[c]=0.0f; cm[c]=0; }
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                const float *bp=blk+(size_t)(z*S+y)*S;
                const mc_u8 *vp=vox+(size_t)(z*S+y)*S;
                int cb=((z>>2)*G+(y>>2))*G;
                for(int sx=0;sx<G;++sx){
                    const float *b4=bp+4*sx; const mc_u8 *v4=vp+4*sx;
                    cs[cb+sx]+=b4[0]+b4[1]+b4[2]+b4[3];
                    cm[cb+sx]+=(v4[0]!=0)+(v4[1]!=0)+(v4[2]!=0)+(v4[3]!=0);
                }
            }
            for(int c=0;c<64;++c) cs[c]=cm[c]?cs[c]/(float)cm[c]:0.0f;
            for(int it=0;it<6;++it){                      // coarse relax (air cells)
                for(int cz=0;cz<G;++cz)for(int cy=0;cy<G;++cy)for(int cx=0;cx<G;++cx){
                    int c=(cz*G+cy)*G+cx; if(cm[c]) continue;
                    float a=0; int k=0;
                    if(cz){a+=cs[c-G*G];k++;} if(cz<G-1){a+=cs[c+G*G];k++;}
                    if(cy){a+=cs[c-G];k++;}   if(cy<G-1){a+=cs[c+G];k++;}
                    if(cx){a+=cs[c-1];k++;}   if(cx<G-1){a+=cs[c+1];k++;}
                    if(k) cs[c]=a/k;
                }
            }
            // seed air voxels from their cell: expand the 4 cell values of a
            // subcube row into a 16-float row pattern once per 4 rows, then a
            // dense branchless select per row (auto-vectorizes).
            float vrow[16]={0};   // always set at y==0 (init quiets -Wmaybe-uninitialized)
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                if((y&3)==0){
                    const float *cr=cs+((z>>2)*G+(y>>2))*G;
                    for(int x=0;x<S;++x) vrow[x]=cr[x>>2];
                }
                float *bp=blk+(size_t)(z*S+y)*S;
                const mc_u8 *vp=vox+(size_t)(z*S+y)*S;
                for(int x=0;x<S;++x) bp[x]=vp[x]?bp[x]:vrow[x];
            }
        }
        // (a) RED-BLACK SOR (two-color Gauss-Seidel, omega=1.6) in a DENSE
        // vectorizable form replacing the serial scalar chain:
        //   - copy the block into an 18^3 zero-padded buffer P (pad cells are
        //     never written, so out-of-block neighbors read as 0 == dc);
        //   - fold air mask and color ((z+y+x)&1) into two per-color weight
        //     arrays (w6 = omega/6 on this color's air voxels, else 0) so a
        //     color pass is a UNIFORM branch-free stencil over the whole
        //     padded array:   P[i] += w6[i]*(nbsum[i] - cnt[i]*P[i])
        //     where cnt[i] (in-block 6-neighbor count, = the old serial
        //     code's divisor scaled into the relaxation step) is a static
        //     block-independent table, built once per thread like PM below.
        //   - within one color no voxel neighbors another, so the neighbor-sum
        //     and update loops are data-parallel and auto-vectorize (AVX/
        //     NEON); updated reds are visible to blacks (true Gauss-Seidel
        //     convergence, same omega, same sweep count as the serial code).
        if(do_fine){
            enum { PS=MC_FILL_PS, PP=PS*PS, PN=MC_FILL_PN };
            float *P=C->fill.P;                            // pads stay 0
            float (*W6)[PN]=C->fill.W6;                    // pads stay 0
            // parity mask (voxel color) and in-block neighbor count are both
            // block-independent: build once per ctx.
            float *PM=C->fill.PM, *CNT=C->fill.CNT;
            if(!C->fill.pm_init){
                for(int z=0;z<PS;++z)for(int y=0;y<PS;++y)for(int x=0;x<PS;++x){
                    int i=(z*PS+y)*PS+x;
                    PM[i]=(float)((z+y+x)&1);
                    CNT[i]=(float)((z>1)+(z<S)+(y>1)+(y<S)+(x>1)+(x<S));
                }
                C->fill.pm_init=1;
            }
            // rows: copy P + build per-color weights in one vectorized pass.
            // Only real-voxel lanes are ever written, so pad/gap lanes of P
            // and W6 keep their static-zero values across blocks.
            const float O6=1.6f/6.0f;
            for(int z=0;z<S;++z)for(int y=0;y<S;++y){
                const mc_u8 *vp=vox+(size_t)(z*S+y)*S;
                const float *bp=blk+(size_t)(z*S+y)*S;
                int pb=((z+1)*PS+(y+1))*PS+1;
                const float *pm=PM+pb;
                for(int x=0;x<S;++x){
                    P[pb+x]=bp[x];
                    float w=vp[x]?0.0f:O6, a=w*pm[x];
                    W6[1][pb+x]=a; W6[0][pb+x]=w-a;
                }
            }
            // each color pass = per z-plane: (1) dense neighbor sums into a
            // small plane buffer NB (pure reads of P), (2) masked update of
            // the plane. Exact red-black Gauss-Seidel: this color's neighbors
            // are all the OTHER color, untouched within the pass, so the
            // snapshot in NB is the live value. Splitting removes the
            // read-after-write dependence that kept the fused in-place loop
            // scalar; both loops auto-vectorize (AVX/NEON). Plane blocking
            // keeps NB and the three active P planes L1-resident instead of
            // streaming a full-volume NB array through L2 every pass.
            // Pad/material lanes are killed by w6=0.
            // (c) the coarse seed does most of the work, so few fine sweeps
            // are needed: on the benchmark below 1 red-black sweep gives a
            // marginally SMALLER archive than the old 3 serial sweeps
            // (-0.013%), 2 sweeps +0.016% — the rate effect of sweep count is
            // already in the quantization noise (values are masked out at
            // decode anyway), so take the cheapest.
            int nsweep=MC_FILL_SWEEPS<1?MC_FILL_SWEEPS:1;
            for(int it=0; it<nsweep; ++it){
                for(int col=0; col<2; ++col){
                    const float *restrict w6=W6[col];
                    for(int pz=1; pz<PS-1; ++pz){
                        float NB[PP]; const int b=pz*PP;
                        for(int k=0;k<PP;++k)
                            NB[k]=P[b+k-1]+P[b+k+1]+P[b+k-PS]+P[b+k+PS]+P[b+k-PP]+P[b+k+PP];
                        for(int k=0;k<PP;++k)
                            P[b+k]+=w6[b+k]*(NB[k]-CNT[b+k]*P[b+k]);
                    }
                }
            }
            for(int z=0;z<S;++z)for(int y=0;y<S;++y)       // material rows are
                memcpy(blk+(size_t)(z*S+y)*S,              // bit-identical (w=0)
                       P+((z+1)*PS+(y+1))*PS+1, S*sizeof(float));
        }
    }
    mc_dct3_fwd(&C->dct,blk,coef);
    rc_i16 *ql=T->ql;
    rc_u8 *scratch=T->scratch;
    const size_t scratch_cap=sizeof T->scratch;
    // fused branchless quant+clamp (same math as quant_one up to fp rounding:
    // t = |c|/step - dzfrac + 1; for |c|>=dz, t>=1 truncates to the level,
    // for |c|<dz, t<1 so max(t,0) truncates to 0). The branchy quant_one
    // loop was scalar; this one auto-vectorizes, with a reciprocal step
    // table instead of a per-coefficient divide.
    for(int idx=0;idx<N3;++idx){
        float c=coef[idx];
        float t=fabsf(c)*C->rstep_tab[idx]+(1.0f-MC_DZ_FRAC);
        t=t>0.0f?t:0.0f; t=t<32767.0f?t:32767.0f;
        mc_i32 v=(mc_i32)t;
        ql[idx]=(rc_i16)(c<0.0f?-v:v);
    }

    // max-error corrections: locally reconstruct and list voxels with |err| > tau.
    uint16_t *cpos=T->cpos; mc_i32 *cdel=T->cdel;
    int ncorr=0;
    if(C->max_err>0){
        float *rcoef=T->rcoef, *rblk=T->rblk;
        for(int idx=0;idx<N3;++idx) rcoef[idx]=deq_one(ql[idx],C->step_tab[idx]);
        mc_dct3_inv(&C->dct,rcoef,rblk);
        for(int i=0;i<n;++i){
            if(!vox[i]) continue;                          // air decodes to exactly 0
            int v=(int)lrintf(rblk[i])+dc; if(v<0)v=0; if(v>255)v=255;
            int err=(int)vox[i]-v;
            int ae=err<0?-err:err;
            if(ae>C->max_err){ cpos[ncorr]=(uint16_t)i; cdel[ncorr]= err<0 ? -(ae-C->max_err) : (ae-C->max_err); ncorr++; }
        }
    }

    rc_enc e; enc_init(&e,scratch,scratch_cap);
    {   // header bins: mixed, has-corr, dc (trained priors)
        const uint16_t (*pri)[RC_NSLOT]=C->pri;
        ctx_t cf[2]; for(int i=0;i<2;++i) ctx_init_p(&cf[i],RC_PRIOR_FLAG(pri)[i]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_p(&cd[i],RC_PRIOR_DC(pri)[i]);
        RC_TRAIN(RCC_FLAG,0,nair>0);  enc_bit(&e,&cf[0],nair>0);
        RC_TRAIN(RCC_FLAG,1,ncorr>0); enc_bit(&e,&cf[1],ncorr>0);
        for(int b=7;b>=0;--b){ int bit=(dc>>b)&1; RC_TRAIN(RCC_DC,7-b,bit); enc_bit(&e,&cd[7-b],bit); }
    }
    if(nair>0) enc_blockmask(&e,vox,C->pri);
    enc_block_coefs(&e,ql,MC_BLK,C->pri);
    if(ncorr>0){                                           // [eg count][gap, sign, eg(|d|-1)]*
        enc_eg(&e,(rc_u32)(ncorr-1));
        rc_u32 prev=0;
        for(int c=0;c<ncorr;++c){
            enc_eg(&e,(rc_u32)cpos[c]-prev); prev=cpos[c];
            mc_i32 D=cdel[c]; enc_bypass(&e,D<0); rc_u32 m=(rc_u32)(D<0?-D:D);
            enc_eg(&e,m-1);
        }
    }
    enc_flush(&e);
    uint32_t slen=(uint32_t)e.len;
    if(slen>scratch_cap){ fprintf(stderr,"mc_enc_block: scratch overflow (%u)\n",slen); abort(); }

    mc_buf_put(out,scratch,slen);
    *len_out = slen;
    return 1;
}

void mc_dec_block(mc_codec_ctx *C, const mc_u8 *p, uint32_t plen, mc_u8 *dst){
    int n=N3, dc=0, flags=0;
    mc_tls_t *T=&C->scratch;
    step_tab_build(C);                  // before hot loops
    mc_u8 *air=T->air;
    rc_i16 *ql=T->ql;
    rc_dec d; dec_init(&d,p,plen);
    {   // header bins (must mirror the encoder exactly)
        const uint16_t (*pri)[RC_NSLOT]=C->pri;
        ctx_t cf[2]; for(int i=0;i<2;++i) ctx_init_p(&cf[i],RC_PRIOR_FLAG(pri)[i]);
        ctx_t cd[8]; for(int i=0;i<8;++i) ctx_init_p(&cd[i],RC_PRIOR_DC(pri)[i]);
        flags |= dec_bit(&d,&cf[0]) ? 1 : 0;
        flags |= dec_bit(&d,&cf[1]) ? 2 : 0;
        for(int b=0;b<8;++b) dc=(dc<<1)|dec_bit(&d,&cd[b]);
    }
    if(flags&1) dec_blockmask(&d,air,C->pri);
    else        memset(air,0,n);
    int ext[3]; dec_block_coefs_ext(&d,ql,MC_BLK,ext,C->pri);
    float *coef=T->coef, *blk=T->blk;
    int ez=ext[0],ey=ext[1],ex=ext[2];
    if(ez<0 && !(flags&1) && !(flags&2)){                   // constant block: dc fill
        memset(dst,(mc_u8)dc,n); return;
    }
    (void)ey;(void)ex;
#if MC_SIMD_NEON
    {   // fused dequant -> i32 DCT input (no float coefficient pass), then
        // integer iDCT and vectorized clamp+dc+air store.
        mc_fi32 *qin=T->qin, *qout=T->qout;
        float32x4_t bias=vdupq_n_f32(MC_DZ_FRAC-1.0f+0.40f);
        for(int i=0;i<N3;i+=4){
            int32x4_t lv=vmovl_s16(vld1_s16(ql+i));
            uint32x4_t nz=vmvnq_u32(vceqzq_s32(lv));
            uint32x4_t neg=vcltzq_s32(lv);
            float32x4_t a=vcvtq_f32_s32(vabsq_s32(lv));
            float32x4_t r=vmulq_f32(vaddq_f32(a,bias),vld1q_f32(C->step_tab+i));
            int32x4_t ri=vcvtnq_s32_f32(r);
            ri=vbslq_s32(neg,vnegq_s32(ri),ri);
            ri=vandq_s32(ri,vreinterpretq_s32_u32(nz));
            vst1q_s32(qin+i,ri);
        }
        mc_dct3_inv_i32(&C->dct,qin,qout);
        int32x4_t vdc=vdupq_n_s32(dc);
        for(int i=0;i<n;i+=16){
            int32x4_t a0=vaddq_s32(vld1q_s32(qout+i),vdc);
            int32x4_t a1=vaddq_s32(vld1q_s32(qout+i+4),vdc);
            int32x4_t a2=vaddq_s32(vld1q_s32(qout+i+8),vdc);
            int32x4_t a3=vaddq_s32(vld1q_s32(qout+i+12),vdc);
            uint16x8_t p0=vcombine_u16(vqmovun_s32(a0),vqmovun_s32(a1));
            uint16x8_t p1=vcombine_u16(vqmovun_s32(a2),vqmovun_s32(a3));
            uint8x16_t v8=vcombine_u8(vqmovn_u16(p0),vqmovn_u16(p1));
            uint8x16_t am=vld1q_u8(air+i);
            v8=vbicq_u8(v8,vtstq_u8(am,am));               // air -> 0
            vst1q_u8(dst+i,v8);
        }
        (void)coef;(void)blk;
    }
#else
    for(int idx=0;idx<N3;++idx) coef[idx]=deq_one(ql[idx],C->step_tab[idx]);
    mc_dct3_inv(&C->dct,coef,blk);
    for(int i=0;i<n;++i){
        int v = air[i] ? 0 : (int)lrintf(blk[i])+dc;  // mask-restore: air -> exactly 0
        if(v<0)v=0; if(v>255)v=255; dst[i]=(mc_u8)v;
    }
#endif
    if(flags&2){                                      // sparse max-error corrections
        // HARDENED: ncorr and positions are attacker-controlled on corrupted
        // input — clamp both so a flipped bit can never write outside dst.
        rc_u32 ncorr=dec_eg(&d)+1, pos=0;
        if(ncorr>(rc_u32)N3) ncorr=N3;
        for(rc_u32 c=0;c<ncorr;++c){
            pos+=dec_eg(&d);
            int neg=dec_bypass(&d); rc_u32 m=dec_eg(&d)+1;
            if(pos>=(rc_u32)N3) break;                // corrupt stream: stop
            int v=(int)dst[pos]+(neg?-(int)m:(int)m);
            if(v<0)v=0; if(v>255)v=255; dst[pos]=(mc_u8)v;
        }
    }
}

// ============================================================================
// mc_archive.h — matter-compressor on-disk archive format constants.
//
// An APPENDABLE, crash-safe, persistent archive: a node tree of dense 256^3 chunks,
// each a 16^3 grid of DCT blocks coded by mc_codec. 8 independent LODs (LOD0 = full
// res), each its own tree. LODs are independently fetchable AND independently
// decodable — no cross-LOD dependency (a hard design constraint).
//
// HIERARCHY (per LOD), all grids are 16x16x16 (4 bits/axis):
//   chunk coord (voxel>>8, up to 12 bits/axis) = [ region 4b | subregion 4b | shard 4b ]
//   CHUNK (256^3 voxels = 16^3 DCT blocks) is the dense leaf. Above it: a dense 16^3
//   SHARD table of chunk offsets, then 2 NODE levels. Covers 2^12 chunks/axis =
//   2^20 voxels/axis.
//
// NODE / SHARD = a DENSE flat MC_GRID3 (=4096) array of u64 child offsets, slot value
// 0 = absent child. Directly indexed by the chunk-coord nibble; no bitmap, no
// rank-packing -> every node + shard is UPDATABLE IN PLACE. This is what makes the
// archive appendable + crash-safe: chunk payloads append at EOF; the index path
// (root -> node -> node -> shard) is created/updated in place with the chunk offset
// published LAST as the commit word. The file is a fully valid, decodable archive
// after every appended chunk and PERSISTS across process runs (reopen + append).
//
// CHUNK blob (v2) = [512B block-bitmap][present-block u32 lens][block payloads];
// one range-GET fetches a whole chunk. Blocks are self-contained (per-block air
// mask lives in each block payload), so one block decode needs only the bitmap,
// the lens, and its own payload.
//
// LAYOUT: [256B header][metadata region up to 128KB][archive data from 128KB:
//          node/shard tables + chunk blobs, both appended at EOF].
// ============================================================================
#ifndef MC_ARCHIVE_H
#define MC_ARCHIVE_H
#include <stdint.h>
#include <string.h>

// ---- node / shard geometry (dense 16^3 child map of u64 offsets) ----
#define MC_GRID         16
#define MC_GRID3        4096          // 16^3 slots
#define MC_BITMAP_BYTES 512           // 4096 bits — chunk-blob block bitmap (NOT node index)
#define MC_NODE_BYTES   (MC_GRID3*8)  // dense node table: 4096 u64 child offsets (32KB)
#define MC_SHARD_BYTES  (MC_GRID3*8)  // dense shard table: 4096 u64 chunk offsets (32KB)
#define MC_TREE_LEVELS  3             // root node, 1 inner node, 1 shard (indexed by nibbles 2,1,0)

// ---- header (256B) ----
#define MC_MAGIC   0x0043434Du      // "MCC\0"
#define MCH_MAGIC   0               // u32 magic
#define MCH_VER     4               // u32 format-version field
#define MCH_NX      12              // u32 volume dims (x fastest)
#define MCH_NY      16
#define MCH_NZ      20
#define MCH_ROOTOFF 24              // u64[8] per-LOD root-node file offset (0 = empty LOD)
#define MCH_TOTLEN  88              // u64 total archive length (= append cursor / EOF)
#define MCH_METAOFF 96              // u64 metadata region start (= MC_HDR = 256)
#define MCH_METACAP 104             // u64 metadata region capacity (= MC_META_END - MC_HDR)
#define MCH_METALEN 112             // u64 metadata bytes actually written
#define MCH_QUALITY 120             // f32 quality the archive was built at (writer stamps it)
#define MCH_PRIOROFF 128            // u64 offset of an optional per-volume prior blob (0 = none)
#define MC_PRIORS_MAGIC 0x53524950u // "PRIS"
#define MC_PRIORS_BYTES (8 + 2*8*32*2)  // magic+ver, RC_PLO + RC_PHI as u16[8][32]
#define MC_HDR      256u            // header size; metadata region begins here
#define MC_META_END (128u*1024u)    // archive data begins at this offset (128KB)
#define MC_META_CAP (MC_META_END - MC_HDR)
#define MC_VERSION  7u              // format version (v7: per-chunk material-fraction map)

#define MC_CHUNK_ALIGN 256          // volume dim must be a multiple of this

// chunk-blob block-bitmap + chunk-coord nibble helpers
static inline int  mc_bit_get(const uint8_t*bm,int i){ return (bm[i>>3]>>(i&7))&1; }
static inline void mc_bit_set(uint8_t*bm,int i){ bm[i>>3]|=(uint8_t)(1u<<(i&7)); }
static inline int  mc_nib(int chunkcoord,int level){ return (chunkcoord>>(4*level))&15; }
#endif

// ============================================================================
// mc_archive_read.h — resolve a chunk by coord through the DENSE node tree, then
// locate a block within the dense chunk. Absent child = empty region (decodes to 0).
//
// The index is 3 dense levels (root node, inner node, shard), each a flat MC_GRID3
// array of u64 offsets indexed by the chunk-coord nibble (2, then 1, then 0). A slot
// value of 0 means absent. No bitmap / no rank-packing at the index level, so the
// resolve is three direct array reads.
// ============================================================================
#ifndef MC_ARCHIVE_READ_H
#define MC_ARCHIVE_READ_H
#include <stdint.h>

// number of set bits in bm[0..idx) — used for the chunk-blob block bitmap only.
static inline int mc_rank(const uint8_t*bm,int idx){
    int r=0, full=idx>>3;
    for(int i=0;i<full;++i) r+=__builtin_popcount(bm[i]);
    int rem=idx&7; if(rem) r+=__builtin_popcount(bm[full] & ((1u<<rem)-1));
    return r;
}

// resolve chunk (cz,cy,cx) -> chunk-blob offset (0 = empty/absent). Walks the dense
// node tree: nibble 2 (root node) -> nibble 1 (inner node) -> nibble 0 (shard) ->
// chunk-blob offset. Each level is a direct u64 array index.
static uint64_t mc_resolve_chunk(const uint8_t*arc, uint64_t root_off,int cz,int cy,int cx){
    uint64_t node = root_off;
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;
        int dz=mc_nib(cz,nib), dy=mc_nib(cy,nib), dx=mc_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        uint64_t childoff; memcpy(&childoff, arc+node + (size_t)idx*8, 8);
        node = childoff;
    }
    return node;   // chunk-blob offset (0 if absent)
}

// chunk blob (v7): [f32 q][u64 xxh64][u16 fmaplen][fmap][512B block-bitmap]
// [present u16 lens][payloads]. q = the chunk's own quality; the hash covers
// fmap+bitmap+lens+payloads; fmap = rc-coded per-block material fractions
// (4096 nibbles, 0..15 ~= 0..100%) for rejection-free ML sampling.
#define MC_BLOB_HDR 14
// Slot sentinel: a chunk that was VISITED and decodes to all-zero (air). Real
// blob offsets are always >= MC_HDR (blobs append after the header), so 1 is a
// safe sentinel. Distinguishes "air, fetched" from "never fetched" (slot 0).
#define MC_SLOT_ZERO 1ull
static inline float mc_chunk_q(const uint8_t*arc, uint64_t chunk_off){
    float q; memcpy(&q,arc+chunk_off,4); return q;
}
static inline uint64_t mc_chunk_stored_hash(const uint8_t*arc, uint64_t chunk_off){
    uint64_t h; memcpy(&h,arc+chunk_off+4,8); return h;
}
static inline uint16_t mc_chunk_fmaplen(const uint8_t*arc, uint64_t chunk_off){
    uint16_t l; memcpy(&l,arc+chunk_off+12,2); return l;
}
// block (bz,by,bx) present? -> 1 + its payload (abs_off, len). 0 = ZERO block. Offsets
// are implicit (cumulative len of present blocks before it); ZERO blocks cost 1 bitmap bit.
// Per-block payload (abs_off,len) within a chunk blob. The on-disk layout stores
// only present-block lengths (offsets implicit = prefix sum). Computing that prefix
// sum + rank per call is O(block-index): summed over a chunk's 4096 blocks it is
// O(n^2) (~16M ops/chunk) -- the render-path cost when a frame samples many blocks
// of one chunk. We cache, thread-locally, a full per-chunk index built in ONE
// O(4096) pass: bi -> abs_off (0 = absent block). Render samples are spatially
// coherent (consecutive blocks share a chunk), so the table is built once per
// chunk and every block lookup is O(1).
typedef struct { const uint8_t*arc; uint64_t chunk_off, tag; uint64_t off[MC_GRID3]; uint16_t len[MC_GRID3]; } mc_chunk_idx;
static const mc_chunk_idx *mc_chunk_index(const uint8_t*arc, uint64_t chunk_off){
    static _Thread_local mc_chunk_idx idx = { .arc=NULL, .chunk_off=~0ull, .tag=0 };
    // Key on (arc base, chunk_off, content hash). chunk_off alone is unsafe: it is
    // reused across archives (same tree position -> same EOF offset) and after a
    // re-append, so the stored xxh64 disambiguates content. Otherwise a stale index
    // from a different chunk is served (caught by mc_v6 par-vs-serial).
    uint64_t tag = mc_chunk_stored_hash(arc, chunk_off);
    if(idx.arc==arc && idx.chunk_off==chunk_off && idx.tag==tag) return &idx;   // hot
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + mc_chunk_fmaplen(arc,chunk_off);
    const uint8_t*bm = arc + bm_off;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t pay = bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2;   // first payload
    int slot=0;
    for(int bi=0; bi<MC_GRID3; ++bi){
        if(mc_bit_get(bm,bi)){
            uint16_t l; memcpy(&l, lens+(size_t)slot*2, 2);
            idx.off[bi]=pay; idx.len[bi]=l; pay+=l; ++slot;
        } else { idx.off[bi]=0; idx.len[bi]=0; }
    }
    idx.arc=arc; idx.chunk_off=chunk_off; idx.tag=tag;
    return &idx;
}
static int mc_block_range(const uint8_t*arc, uint64_t chunk_off, int bz,int by,int bx,
                          uint64_t *abs_off, uint32_t *len){
    const mc_chunk_idx *ix = mc_chunk_index(arc, chunk_off);
    int bi=(bz*16+by)*16+bx;
    if(!ix->off[bi]) return 0;                        // absent (ZERO) block
    *abs_off = ix->off[bi]; *len = ix->len[bi];
    return 1;
}

// total byte length of a chunk blob — used to copy a whole compressed chunk verbatim
// (mc_append_chunk_compressed) and for chunk-blob range queries. chunk_off must be valid.
static uint64_t mc_chunk_blob_len(const uint8_t*arc, uint64_t chunk_off){
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + mc_chunk_fmaplen(arc,chunk_off);
    const uint8_t*bm = arc + bm_off;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + MC_BITMAP_BYTES;
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    return bm_off + MC_BITMAP_BYTES + (uint64_t)npresent*2 + paybytes - chunk_off;
}
#endif

// ============================================================================
// mc_archive.c — matter-compressor archive build + APPENDABLE writer + decode.
// See mc_archive_api.h. Source-agnostic: no zarr/S3 here.
//
// Index is a DENSE node tree (root node -> inner node -> shard), each a flat
// MC_GRID3 array of u64 offsets, updatable in place. Chunk payloads append at EOF.
//
// Three faces:
//   - mc_build / mc_build_to_file : one-shot build of a whole volume (RAM-materialized).
//   - mc_archive_* : ONE persistent, crash-safe, appendable on-disk handle that both
//                    APPENDS chunks and DECODES them (no writer/reader split). mmap +
//                    atomic append cursor + in-place dense-node index + release-published
//                    commit, modeled on volume-compressor's writer; decode reads the live
//                    mmap.
//   - mc_open / mc_open_streaming + decode : read an already-built archive from a buffer
//                    or a byte-source (streaming / S3).
// ============================================================================
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <pthread.h>
  #define MC_HAVE_MMAP 1
#else
  #define MC_HAVE_MMAP 0
#endif

typedef uint8_t u8;
static void priors_load(const uint8_t *base);

// ---------------------------------------------------------------- volume (one LOD)
// A contiguous dim^3 u8 buffer. LOD0 is materialized from the source callback; coarser
// LODs are box-decimated from the finer one.
typedef struct { const u8 *v; int nz, ny, nx; } vol_t;
static inline u8 vget(const vol_t *V, int z,int y,int x){
    if((unsigned)z>=(unsigned)V->nz||(unsigned)y>=(unsigned)V->ny||(unsigned)x>=(unsigned)V->nx) return 0;
    return V->v[((size_t)z*V->ny+y)*V->nx+x];
}
// 2x box-decimate: mean of NONZERO children; all-zero stays 0 (inherited zero-mask).
// Per-axis dims; odd source dims round up (the edge voxel decimates alone).
static u8 *decimate(const u8 *src,int DZ,int DY,int DX,int *HZ,int *HY,int *HX){
    int hz=(DZ+1)/2, hy=(DY+1)/2, hx=(DX+1)/2;
    u8 *o=calloc((size_t)hz*hy*hx,1);
    for(int z=0;z<hz;++z)for(int y=0;y<hy;++y)for(int x=0;x<hx;++x){ int s=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            int sz=2*z+dz, sy=2*y+dy, sx=2*x+dx;
            if(sz>=DZ||sy>=DY||sx>=DX) continue;
            u8 v=src[((size_t)sz*DY+sy)*DX+sx]; if(v){s+=v;c++;}}
        o[((size_t)z*hy+y)*hx+x]=c?(u8)((s+c/2)/c):0; }
    *HZ=hz; *HY=hy; *HX=hx; return o; }

// gather a 16^3 block from a contiguous 256^3 chunk buffer (chunk is dense, no edges).
static int gather_blk256(const u8 *chunk,int bz,int by,int bx,u8 *dst){
    int z0=bz*MC_BLK,y0=by*MC_BLK,x0=bx*MC_BLK,any=0;
    for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)for(int x=0;x<MC_BLK;++x){
        u8 v=chunk[((size_t)(z0+z)*MC_CHUNK+(y0+y))*MC_CHUNK+(x0+x)]; dst[(z*MC_BLK+y)*MC_BLK+x]=v; any|=v; }
    return any;
}

// ---------------------------------------------------------------- chunk-blob encoder
// Encode one dense 256^3 chunk buffer into a compressed blob in `out` (a growable
// byte sink: out_put(out, ptr, n)). Returns the blob length, or 0 if the chunk is all
// air (no blob — caller leaves the slot absent). Shared by build + writer.
// blob (v6) = [f32 q][u64 xxh64][512B block-bitmap][present-block u16 lens]
// [block payloads]; blocks are self-contained (mask in payload). q is the
// chunk's own quality; the hash covers bitmap+lens+payloads.
typedef void (*out_put_fn)(void *out, const void *s, size_t n);

static size_t encode_chunk_blob(mc_codec_ctx *C, const u8 *chunk256, out_put_fn put, void *out){
    mc_buf *tmp=&C->chunk.tmp; tmp->len=0;
    uint8_t bm[MC_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint16_t blen[MC_GRID3]; int npresent=0;
    uint8_t *frac=C->chunk.frac;
    memset(frac,0,MC_GRID3);
    u8 *vox=C->chunk.vox;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk256(chunk256,bz,by,bx,vox)) continue;
        int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=vox[i]!=0;
        frac[bi]=(uint8_t)((cnt*15+2048)/4096);              // nibble 0..15
        if(cnt&&!frac[bi]) frac[bi]=1;                        // any material -> >=1
        uint32_t len=0; if(mc_enc_block(C,vox,tmp,&len)){ mc_bit_set(bm,bi); blen[bi]=(uint16_t)len; npresent++; }
    }
    if(!npresent) return 0;   // all air -> no blob
    // v7 blob header: [f32 q][u64 xxh64][u16 fmaplen][fmap]
    float q=C->quality;
    uint16_t *lens16=C->chunk.lens16; int nl=0;
    for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(bm,bi)) lens16[nl++]=blen[bi];
    uint8_t *fmap=C->chunk.fmap;
    uint16_t fml=(uint16_t)mc_enc_fracmap(frac,fmap,sizeof C->chunk.fmap);
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens16,(size_t)nl*2,h);
    h^=mc_xxh64(tmp->p,tmp->len,h);
    size_t total=MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(size_t)npresent*2+tmp->len;
    put(out,&q,4); put(out,&h,8); put(out,&fml,2);
    put(out,fmap,fml);
    put(out,bm,MC_BITMAP_BYTES);
    put(out,lens16,(size_t)nl*2);
    put(out,tmp->p,tmp->len);
    return total;
}

// recompute a chunk blob's hash from its bytes (mc_verify / verify-on-decode).
uint64_t mc_chunk_compute_hash(const uint8_t *blob, uint64_t blob_len){
    uint16_t fml; memcpy(&fml,blob+12,2);
    const uint8_t *fmap=blob+MC_BLOB_HDR;
    const uint8_t *bm=fmap+fml;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t *lens=bm+MC_BITMAP_BYTES;
    uint64_t pay=(uint64_t)MC_BLOB_HDR+fml+MC_BITMAP_BYTES+(uint64_t)npresent*2;
    uint64_t h=mc_xxh64(fmap,fml,0x6D636368756E6Bull);
    h^=mc_xxh64(bm,MC_BITMAP_BYTES,h);
    h^=mc_xxh64(lens,(size_t)npresent*2,h);
    h^=mc_xxh64(blob+pay,(size_t)(blob_len-pay),h);
    return h;
}

// ---------------------------------------------------------------- one-shot build (abuf)
typedef struct { u8 *p; size_t len, cap; } abuf;
static void a_reserve(abuf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; void*tmp=realloc(b->p,nc); if(!tmp)return; b->p=tmp; b->cap=nc; }
static size_t a_put_at(abuf*b,const void*s,size_t n){ a_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t a_zero(abuf*b,size_t n){ a_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
static void a_u32(abuf*b,size_t at,uint32_t v){ memcpy(b->p+at,&v,4); }
static void a_u64(abuf*b,size_t at,uint64_t v){ memcpy(b->p+at,&v,8); }
static void abuf_put(void *out, const void *s, size_t n){ a_put_at((abuf*)out,s,n); }

// gather a contiguous 256^3 chunk out of a LOD volume (zero-padded at the volume edge).
static int gather_chunk256(const vol_t *V,int cz,int cy,int cx,u8 *dst){
    int z0=cz*256,y0=cy*256,x0=cx*256,any=0;
    for(int z=0;z<MC_CHUNK;++z)for(int y=0;y<MC_CHUNK;++y)for(int x=0;x<MC_CHUNK;++x){
        u8 v=vget(V,z0+z,y0+y,x0+x); dst[((size_t)z*MC_CHUNK+y)*MC_CHUNK+x]=v; any|=v; }
    return any;
}

// dense one-shot build: write the dense node tree for one LOD volume. Allocate the
// dense tables top-down with deferred offset back-patch (we know child offsets only
// after writing children, but the tables are dense + fixed-size so we reserve then fill).
// Simpler: build bottom-up into an in-RAM dense tree, then emit. We emit chunks first,
// recording offsets in a dense shard map, then emit shards, inner nodes, root.

static uint64_t build_lod_dense(mc_codec_ctx *C, abuf*b, const vol_t *V, int ncz,int ncy,int ncx){
    // dense maps over chunk grid (sparse population). We allocate per populated node.
    // root: 16^3 inner-node offsets; inner: 16^3 shard offsets; shard: 16^3 chunk offsets.
    // chunk coord nibbles: 2 (root), 1 (inner), 0 (shard).
    // First pass: emit chunk blobs, fill shard tables in RAM keyed by (n1=hi, n0).
    // To keep it simple + correct for any nchunks up to 4096/axis, use hash-free dense
    // allocation only where present.
    // We build a 3-level RAM tree of u64 tables, emit chunk blobs to `b`, then emit the
    // tables bottom-up and return the root offset.

    // RAM node: 4096 u64 child *table-indices* during build, resolved to file offsets on emit.
    // Use dynamic arrays of tables.
    typedef struct { uint64_t slot[MC_GRID3]; } table_t;   // 32KB each
    // shards keyed by (n2,n1); inners keyed by n2; root single.
    // We accumulate present shards/inners in vectors with their grid index.
    table_t *root = calloc(1,sizeof *root);
    // inner tables: up to 4096; allocate lazily.
    table_t **inner = calloc(MC_GRID3,sizeof *inner);
    // for each inner, its shard tables lazily.
    table_t ***shard = calloc(MC_GRID3,sizeof *shard);

    u8 *chunkbuf=malloc((size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK);
    if(!chunkbuf){ free(root); free(inner); free(shard); return 0; }

    for(int cz=0;cz<ncz;++cz)for(int cy=0;cy<ncy;++cy)for(int cx=0;cx<ncx;++cx){
        if(!gather_chunk256(V,cz,cy,cx,chunkbuf)) continue;
        size_t at=b->len;
        if(!encode_chunk_blob(C,chunkbuf,abuf_put,b)) continue;   // all air
        int n2=(mc_nib(cz,2)*16+mc_nib(cy,2))*16+mc_nib(cx,2);
        int n1=(mc_nib(cz,1)*16+mc_nib(cy,1))*16+mc_nib(cx,1);
        int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
        if(!inner[n2]) inner[n2]=calloc(1,sizeof(table_t));
        if(!shard[n2]) shard[n2]=calloc(MC_GRID3,sizeof(table_t*));
        if(!shard[n2][n1]) shard[n2][n1]=calloc(1,sizeof(table_t));
        shard[n2][n1]->slot[n0]=(uint64_t)at;   // chunk-blob offset
    }
    // emit bottom-up: shards, then inners, then root.
    int any_root=0;
    for(int n2=0;n2<MC_GRID3;++n2){
        if(!inner[n2]) continue;
        int any_inner=0;
        for(int n1=0;n1<MC_GRID3;++n1){
            if(!shard[n2] || !shard[n2][n1]) continue;
            uint64_t soff=a_put_at(b,shard[n2][n1],MC_SHARD_BYTES);
            inner[n2]->slot[n1]=soff; any_inner=1;
        }
        if(any_inner){ uint64_t ioff=a_put_at(b,inner[n2],MC_NODE_BYTES); root->slot[n2]=ioff; any_root=1; }
    }
    uint64_t root_off=0;
    if(any_root) root_off=a_put_at(b,root,MC_NODE_BYTES);

    // free RAM tree
    for(int n2=0;n2<MC_GRID3;++n2){
        if(shard[n2]){ for(int n1=0;n1<MC_GRID3;++n1) free(shard[n2][n1]); free(shard[n2]); }
        free(inner[n2]);
    }
    free(shard); free(inner); free(root);
    free(chunkbuf);
    return root_off;
}

uint8_t *mc_build(mc_voxel_fn src, void *ud, const mc_build_opts *opts, size_t *out_len){
    int NX = opts->nx>0?opts->nx:opts->dim;
    int NY = opts->ny>0?opts->ny:opts->dim;
    int NZ = opts->nz>0?opts->nz:opts->dim;
    if(NX<=0||NY<=0||NZ<=0){ fprintf(stderr,"mc_build: bad dims %dx%dx%d\n",NX,NY,NZ); return NULL; }
    // pad each axis to the chunk boundary (zero padding is nearly free)
    int PX=(NX+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    int PY=(NY+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    int PZ=(NZ+MC_CHUNK_ALIGN-1)/MC_CHUNK_ALIGN*MC_CHUNK_ALIGN;
    mc_codec_init();
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C){ fprintf(stderr,"mc_build: OOM allocating codec ctx\n"); return NULL; }
    mc_codec_ctx_set_quality(C,opts->quality);
    u8 *lod0=calloc((size_t)PZ*PY,(size_t)PX);
    if(!lod0){ fprintf(stderr,"mc_build: OOM allocating %dx%dx%d\n",PZ,PY,PX); mc_codec_ctx_free(C); return NULL; }
    for(int z=0;z<NZ;++z)for(int y=0;y<NY;++y)for(int x=0;x<NX;++x)
        lod0[((size_t)z*PY+y)*PX+x]=src(ud,x,y,z);

    abuf b={0}; a_zero(&b,MC_META_END);
    size_t mlen=0;
    if(opts->metadata && opts->meta_len){
        mlen=opts->meta_len;
        if(mlen>MC_META_CAP){ fprintf(stderr,"mc_build: metadata %zu B > %u cap, truncating\n",mlen,(unsigned)MC_META_CAP); mlen=MC_META_CAP; }
        memcpy(b.p+MC_HDR, opts->metadata, mlen);
    }
    uint64_t roots[8]={0};
    const u8 *cur=lod0; u8 *owned=NULL;
    int dz=PZ, dy=PY, dx=PX;
    for(int lod=0; lod<8 && (dz>=1&&dy>=1&&dx>=1) && (dz>=MC_CHUNK||dy>=MC_CHUNK||dx>=MC_CHUNK||lod==0); ++lod){
        vol_t vv={cur,dz,dy,dx};
        int ncz=(dz+255)/256, ncy=(dy+255)/256, ncx=(dx+255)/256;
        roots[lod]=build_lod_dense(C,&b,&vv,ncz,ncy,ncx);
        if(dz/2<1||dy/2<1||dx/2<1) break;
        if(dz/2<MC_CHUNK&&dy/2<MC_CHUNK&&dx/2<MC_CHUNK&&lod>=1) break;
        int hz,hy,hx;
        u8 *next=decimate(cur,dz,dy,dx,&hz,&hy,&hx);
        if(owned) free(owned); owned=next; cur=next; dz=hz; dy=hy; dx=hx;
    }
    if(owned && owned!=lod0) free(owned);
    free(lod0);
    float q=opts->quality;
    a_u32(&b,MCH_MAGIC,MC_MAGIC); a_u32(&b,MCH_VER,MC_VERSION);
    a_u32(&b,MCH_NX,(uint32_t)NX); a_u32(&b,MCH_NY,(uint32_t)NY); a_u32(&b,MCH_NZ,(uint32_t)NZ);
    for(int l=0;l<8;++l) a_u64(&b,MCH_ROOTOFF+l*8,roots[l]);
    a_u64(&b,MCH_TOTLEN,b.len);
    a_u64(&b,MCH_METAOFF,MC_HDR); a_u64(&b,MCH_METACAP,MC_META_CAP); a_u64(&b,MCH_METALEN,mlen);
    memcpy(b.p+MCH_QUALITY,&q,4);
    mc_codec_ctx_free(C);
    if(out_len) *out_len=b.len;
    return b.p;
}

int mc_build_to_file(mc_voxel_fn src, void *ud, const mc_build_opts *opts, const char *outpath){
    size_t len=0; uint8_t *buf=mc_build(src,ud,opts,&len);
    if(!buf) return 1;
    FILE *of=fopen(outpath,"wb"); if(!of){ perror("fopen out"); free(buf); return 1; }
    fwrite(buf,1,len,of); fclose(of); free(buf);
    return 0;
}

// ============================================================================
// APPENDABLE WRITER — persistent mmap'd archive, modeled on volume-compressor.
// ============================================================================
#if MC_HAVE_MMAP

#define MC_RESERVE   (10ull*1024*1024*1024*1024)   // 10 TiB virtual reservation (NORESERVE)
#define MC_GROW_STEP (1ull*1024*1024*1024)          // grow the file 1 GiB at a time

// Coverage memo: an O(1) resident-region set so frozen render reads never walk
// the node tree (mc_resolve_chunk) per absent block. Keyed by a region key that
// packs (lod,cz,cy,cx); value bit distinguishes PRESENT from ZERO(air). Open
// addressing, power-of-two, atomic slots — lock-free inserts from decode threads,
// lock-free probes from render. Slot 0 = empty. Sized generously; never resizes
// (a render archive's covered-region count is bounded by the volume's region
// count, and we cap fill at the reserve anyway).
#define MC_COV_CAP (1u<<20)            // 1M slots -> up to ~700k regions at 0.7 load
// memo value = state(2 bits) | packed region key. Region coords are 256^3-chunk
// indices (<= ~4096 even for a 77824^3 volume) so 12 bits/axis is ample; the key
// fits in 39 bits, leaving the top free for state. state 0 = empty slot.
#define MC_COV_PRESENT 1ull
#define MC_COV_ZERO    2ull
#define MC_COV_ABSENT  3ull            // VISITED-and-absent: memoized so re-probes
                                       // of a not-yet-downloaded region are O(1)
                                       // (overwritten -> PRESENT when it lands).
struct mc_archive {
    int fd;
    u8 *base;                  // fixed mmap base (never moves)
    _Atomic uint64_t cursor;   // append EOF (bytes used)
    _Atomic uint64_t file_len; // current ftruncate'd file size
    int dim;
    float quality;
    uint64_t reserve;          // mmap reservation size (dims-derived, <= MC_RESERVE)
    pthread_mutex_t grow_mu;   // serializes ftruncate only; decode is lock-free
    _Atomic uint64_t *cov;     // coverage memo slots (region key | flags), 0 = empty
    _Atomic uint64_t gen;      // bumped on every publish; invalidates per-thread
                               // chunk_off memos when a chunk is re-appended
};

// region key for the coverage memo: state in the top 2 bits, then lod(3)+12/axis.
static inline uint64_t mc_covkey(int lod,int cz,int cy,int cx){
    return ((uint64_t)(lod & 7) << 36) | ((uint64_t)(cz & 0xFFF) << 24) |
           ((uint64_t)(cy & 0xFFF) << 12) | (uint64_t)(cx & 0xFFF);
}
#define MC_COV_STATE_SHIFT 62
#define MC_COV_KEYMASK ((1ull<<MC_COV_STATE_SHIFT)-1)
// probe the memo. PRESENT/ZERO/ABSENT if memoized, MC_ABSENT(== not found) only
// when the slot is empty or the probe run is exhausted (caller then tree-walks).
static int mc_cov_probe(mc_archive *a,int lod,int cz,int cy,int cx){
    if(!a->cov) return -1;
    uint64_t key = mc_covkey(lod,cz,cy,cx);
    uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>44);
    for(int p=0;p<32;++p){
        uint32_t i=(h+(uint32_t)p)&(MC_COV_CAP-1);
        uint64_t cur=atomic_load_explicit(&a->cov[i],memory_order_acquire);
        if(cur==0) return -1;                  // empty -> not memoized
        if((cur & MC_COV_KEYMASK)==key){
            uint64_t st = cur >> MC_COV_STATE_SHIFT;
            return st==MC_COV_PRESENT?MC_PRESENT : st==MC_COV_ZERO?MC_ZERO : MC_ABSENT;
        }
    }
    return -1;                                 // probe exhausted -> tree-walk
}
// insert/update the memo with an explicit state (PRESENT/ZERO/ABSENT). A later
// PRESENT publish overwrites a prior ABSENT/ZERO for the same region in place.
static void mc_cov_put_state(mc_archive *a,int lod,int cz,int cy,int cx,uint64_t state){
    if(!a->cov) return;
    uint64_t key = mc_covkey(lod,cz,cy,cx);
    uint64_t val = key | (state<<MC_COV_STATE_SHIFT);
    uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>44);
    for(int p=0;p<32;++p){
        uint32_t i=(h+(uint32_t)p)&(MC_COV_CAP-1);
        uint64_t cur=atomic_load_explicit(&a->cov[i],memory_order_relaxed);
        if((cur & MC_COV_KEYMASK)==key && cur!=0){   // same region: update state
            if(cur==val) return;
            atomic_store_explicit(&a->cov[i],val,memory_order_release); return;
        }
        if(cur==0){
            uint64_t exp=0;
            if(atomic_compare_exchange_strong_explicit(&a->cov[i],&exp,val,
                   memory_order_release,memory_order_relaxed)) return;
            if((exp & MC_COV_KEYMASK)==key){  // lost race to same region
                if(exp!=val) atomic_store_explicit(&a->cov[i],val,memory_order_release);
                return; }
        }
    }
}
static inline void mc_cov_put(mc_archive *a,int lod,int cz,int cy,int cx,int is_zero){
    mc_cov_put_state(a,lod,cz,cy,cx, is_zero?MC_COV_ZERO:MC_COV_PRESENT);
}

static int w_ensure(mc_archive *w, uint64_t need){
    uint64_t fl = atomic_load_explicit(&w->file_len, memory_order_acquire);
    if(need <= fl) return 0;
    pthread_mutex_lock(&w->grow_mu);
    fl = atomic_load_explicit(&w->file_len, memory_order_relaxed);
    if(need > fl){
        uint64_t nf = fl;
        while(nf < need) nf += MC_GROW_STEP;
        if(nf > w->reserve){   // past the mmap reservation: fail cleanly, not SIGBUS
            fprintf(stderr,"mc_archive: grow beyond reservation (%llu > %llu)\n",
                    (unsigned long long)nf,(unsigned long long)w->reserve);
            pthread_mutex_unlock(&w->grow_mu); return -1;
        }
        if(ftruncate(w->fd, (off_t)nf) != 0){ pthread_mutex_unlock(&w->grow_mu); return -1; }
        atomic_store_explicit(&w->file_len, nf, memory_order_release);
    }
    pthread_mutex_unlock(&w->grow_mu);
    return 0;
}
// reserve a disjoint [off, off+n) range at EOF, growing the file as needed.
static uint64_t w_alloc(mc_archive *w, uint64_t n){
    uint64_t off = atomic_fetch_add_explicit(&w->cursor, n, memory_order_relaxed);
    if(w_ensure(w, off+n)!=0) return ~0ull;
    return off;
}
static void w_write_u64(mc_archive *w, uint64_t at, uint64_t v){ memcpy(w->base+at,&v,8); }
static uint64_t w_read_u64(mc_archive *w, uint64_t at){ uint64_t v; memcpy(&v,w->base+at,8); return v; }
// Atomically publish a node offset into a tree slot iff still empty (0). Index nodes
// are 32KB-aligned (w_alloc bumps by MC_NODE_BYTES) so every u64 slot is 8-aligned ->
// safe to treat as _Atomic. Returns the offset now in the slot: `want` if we won the
// CAS, else the offset another thread published first (caller abandons its node).
static uint64_t w_publish_child(mc_archive *w, uint64_t at, uint64_t want){
    _Atomic uint64_t *slot = (_Atomic uint64_t*)(void*)(w->base+at);
    uint64_t exp = 0;
    if(atomic_compare_exchange_strong_explicit(slot,&exp,want,
           memory_order_release,memory_order_acquire)) return want;
    return exp;   // lost the race: someone else's node is live; abandon ours
}

// growable sink wrapping a writer EOF append (used by encode_chunk_blob via a staging
// buffer; we encode to RAM first then memcpy the whole blob into one EOF range so the
// chunk payload is contiguous + committed atomically).
typedef struct { u8 *p; size_t len, cap; } stage_t;
static void stage_put(void *out, const void *s, size_t n){
    stage_t *st=(stage_t*)out;
    if(st->len+n>st->cap){ size_t nc=st->cap?st->cap*2:1<<16; while(nc<st->len+n)nc*=2;
        void *tmp=realloc(st->p,nc); if(!tmp) return; st->p=tmp; st->cap=nc; }
    memcpy(st->p+st->len,s,n); st->len+=n;
}

// ensure the index path root->inner->shard exists for (lod,cz,cy,cx); return the file
// offset of the SHARD-table slot that will hold the chunk offset. Creates dense node
// tables in place as needed (allocated zeroed at EOF, parent slot published last).
static uint64_t w_ensure_shard_slot(mc_archive *w, int lod, int cz,int cy,int cx){
    // Decoder/dl threads append concurrently; node creation must be race-free. We
    // allocate a zeroed node then CAS-publish it into the parent slot. If we lose the
    // CAS another thread's node is live and we abandon ours (a bounded 32KB bump-alloc
    // leak, only on genuine concurrent first-touch of the same parent). No lock.
    uint64_t root = w_read_u64(w, MCH_ROOTOFF + (uint64_t)lod*8);
    if(!root){
        uint64_t no = w_alloc(w, MC_NODE_BYTES); if(no==~0ull) return ~0ull;
        memset(w->base+no, 0, MC_NODE_BYTES);
        root = w_publish_child(w, MCH_ROOTOFF+(uint64_t)lod*8, no);
    }
    // walk nibble 2 (root) -> nibble 1 (inner) -> nibble 0 (shard slot).
    uint64_t node = root;
    for(int nib=MC_TREE_LEVELS-1; nib>=1; --nib){
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        // acquire-load to pair with w_publish_child's release: a child offset becomes
        // visible only after its node's zero-fill, so we never walk into stale bytes.
        uint64_t child = atomic_load_explicit(
            (_Atomic uint64_t*)(void*)(w->base + node + (uint64_t)idx*8),
            memory_order_acquire);
        if(!child){
            uint64_t no = w_alloc(w, MC_NODE_BYTES); if(no==~0ull) return ~0ull;
            memset(w->base+no, 0, MC_NODE_BYTES);
            child = w_publish_child(w, node + (uint64_t)idx*8, no);
        }
        node = child;
    }
    int n0=(mc_nib(cz,0)*16+mc_nib(cy,0))*16+mc_nib(cx,0);
    return node + (uint64_t)n0*8;   // address of the shard slot for this chunk
}

// append a finished compressed blob at EOF + publish it in the shard slot (commit word).
// Mark a chunk's slot as VISITED-but-all-zero (air). Lets a re-run / prefetch
// tell "fetched, it was air" from "never fetched" — no blob is written.
static int w_mark_zero(mc_archive *w,int lod,int cz,int cy,int cx){
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull) return -1;
    w_write_u64(w, slot, MC_SLOT_ZERO);
    mc_cov_put(w, lod, cz, cy, cx, 1 /*air*/);
    atomic_fetch_add_explicit(&w->gen, 1, memory_order_release);
    return 0;
}

static int w_install_blob(mc_archive *w,int lod,int cz,int cy,int cx,const u8 *blob,size_t len){
    uint64_t slot = w_ensure_shard_slot(w,lod,cz,cy,cx);
    if(slot==~0ull) return -1;
    uint64_t off = w_alloc(w, len);
    if(off==~0ull) return -1;
    memcpy(w->base+off, blob, len);
    // release fence so the payload bytes are visible before the commit word.
    atomic_thread_fence(memory_order_release);
    w_write_u64(w, slot, off);   // publish chunk offset = commit
    mc_cov_put(w, lod, cz, cy, cx, 0 /*present*/);
    atomic_fetch_add_explicit(&w->gen, 1, memory_order_release);
    // keep the header's total length current so the file is valid if reopened now.
    uint64_t cur = atomic_load_explicit(&w->cursor, memory_order_relaxed);
    w_write_u64(w, MCH_TOTLEN, cur);
    return 0;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality);
mc_archive *mc_archive_open(const char *path, int dim, float quality){
    return mc_archive_open_dims(path,dim,dim,dim,quality);
}
// reservation sized from the volume: worst-case compressed bytes are bounded by
// ~raw size; 1.5x headroom + 1 GiB floor, capped at MC_RESERVE. A blanket 10 TiB
// map breaks sanitizer shadow memory and small-volume test runners.
static uint64_t reserve_for_dims(int nx,int ny,int nz){
    uint64_t need=0, dz=(uint64_t)nz, dy=(uint64_t)ny, dx=(uint64_t)nx;
    for(int l=0;l<8;++l){
        uint64_t pz=(dz+255)/256*256, py=(dy+255)/256*256, px=(dx+255)/256*256;
        need += pz*py*px;
        dz=(dz+1)/2; dy=(dy+1)/2; dx=(dx+1)/2;
    }
    uint64_t r = need + need/2 + (1ull<<30);
    return r > MC_RESERVE ? MC_RESERVE : r;
}

mc_archive *mc_archive_open_dims(const char *path, int nx, int ny, int nz, float quality){
    if(nx<=0||ny<=0||nz<=0){ fprintf(stderr,"mc_archive_open: bad dims\n"); return NULL; }
    int dim=nx;  // legacy field below; per-axis dims live in the header
    mc_codec_init();   // quality is per-chunk; each worker sets it on its own ctx
    int fd = open(path, O_RDWR|O_CREAT, 0644);
    if(fd<0){ perror("mc_archive_open: open"); return NULL; }
    struct stat sb; if(fstat(fd,&sb)!=0){ perror("fstat"); close(fd); return NULL; }
    int fresh = (sb.st_size==0);
    uint64_t reserve = reserve_for_dims(nx,ny,nz);
    uint64_t init_len;
    if(fresh){
        init_len = MC_META_END;   // header + metadata region; data appends after.
        if(ftruncate(fd,(off_t)init_len)!=0){ perror("ftruncate"); close(fd); return NULL; }
    } else {
        init_len = (uint64_t)sb.st_size;
    }
    u8 *base = mmap(NULL, reserve, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_NORESERVE, fd, 0);
    if(base==MAP_FAILED){ perror("mmap"); close(fd); return NULL; }

    mc_archive *w = calloc(1,sizeof *w);
    w->fd=fd; w->base=base; w->dim=dim; w->quality=quality; w->reserve=reserve;
    pthread_mutex_init(&w->grow_mu,NULL);
    atomic_store(&w->file_len, init_len);
    w->cov = calloc(MC_COV_CAP, sizeof *w->cov);   // coverage memo (0 = empty)

    if(fresh){
        memset(base,0,MC_HDR);
        uint32_t magic=MC_MAGIC, ver=MC_VERSION;
        uint32_t ux=(uint32_t)nx, uy=(uint32_t)ny, uz=(uint32_t)nz;
        memcpy(base+MCH_MAGIC,&magic,4); memcpy(base+MCH_VER,&ver,4);
        memcpy(base+MCH_NX,&ux,4); memcpy(base+MCH_NY,&uy,4); memcpy(base+MCH_NZ,&uz,4);
        uint64_t z=0; for(int l=0;l<8;++l) memcpy(base+MCH_ROOTOFF+l*8,&z,8);
        uint64_t metaoff=MC_HDR, metacap=MC_META_CAP, totlen=MC_META_END;
        memcpy(base+MCH_METAOFF,&metaoff,8); memcpy(base+MCH_METACAP,&metacap,8);
        memcpy(base+MCH_METALEN,&z,8); memcpy(base+MCH_TOTLEN,&totlen,8);
        memcpy(base+MCH_QUALITY,&quality,4);
        atomic_store(&w->cursor, MC_META_END);
    } else {
        uint32_t magic; memcpy(&magic,base+MCH_MAGIC,4);
        uint32_t ver;   memcpy(&ver,base+MCH_VER,4);
        uint32_t ux,uy,uz; memcpy(&ux,base+MCH_NX,4); memcpy(&uy,base+MCH_NY,4); memcpy(&uz,base+MCH_NZ,4);
        if(magic!=MC_MAGIC || ver!=MC_VERSION || (int)ux!=nx || (int)uy!=ny || (int)uz!=nz){
            fprintf(stderr,"mc_archive_open: %s is not a matching mc archive (magic/ver/dims)\n",path);
            munmap(base,reserve); close(fd); free(w); return NULL;
        }
        uint64_t totlen; memcpy(&totlen,base+MCH_TOTLEN,8);
        if(totlen < MC_META_END) totlen=MC_META_END;
        atomic_store(&w->cursor, totlen);
        priors_load(base);
    }
    return w;
}

int mc_archive_append_chunk_compressed(mc_archive *a, int lod, int cz,int cy,int cx,
                                       const uint8_t *blob, size_t len){
    if(!a||lod<0||lod>7||!blob||!len) return -1;
    return w_install_blob(a,lod,cz,cy,cx,blob,len);
}

int mc_archive_reserve_index(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return -1;
    return w_ensure_shard_slot(a,lod,cz,cy,cx)==~0ull ? -1 : 0;
}

int mc_archive_append_chunk_ctx(mc_archive *a, mc_codec_ctx *C,
                                int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256]){
    if(!a||!C||lod<0||lod>7||!vox) return -1;
    stage_t st={0};
    size_t blen = encode_chunk_blob(C, vox, stage_put, &st);
    int rc = 0;
    if(blen) rc = w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
    else     rc = w_mark_zero(a,lod,cz,cy,cx);   // air, but record it as VISITED
    free(st.p);
    return rc;
}
int mc_archive_append_chunk_raw_q(mc_archive *a, int lod, int cz,int cy,int cx,
                                  const mc_u8 vox[256*256*256], float q){
    if(!a||lod<0||lod>7||!vox) return -1;
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C) return -1;
    mc_codec_ctx_set_quality(C,q);
    int rc = mc_archive_append_chunk_ctx(a,C,lod,cz,cy,cx,vox);
    mc_codec_ctx_free(C);
    return rc;
}
int mc_archive_append_chunk_raw(mc_archive *a, int lod, int cz,int cy,int cx, const mc_u8 vox[256*256*256]){
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,a?a->quality:8.0f);
}

// rate control: sample-encode 256 diagonally-spread blocks at base q, scale q
// once by the empirical bytes ~ q^-GAMMA law, then encode the chunk for real.
#define MC_RC_GAMMA 0.75f
int mc_archive_append_chunk_target(mc_archive *a, int lod, int cz,int cy,int cx,
                                   const mc_u8 vox[256*256*256], float target_ratio,
                                   float *q_out){
    if(!a||lod<0||lod>7||!vox||target_ratio<=1.0f) return -1;
    float q0=a->quality;
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C) return -1;
    mc_codec_ctx_set_quality(C,q0);
    mc_buf samp={0};
    u8 blk[MC_BLK*MC_BLK*MC_BLK];
    size_t sample_bytes=0; int sampled=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by){
        int bx=(bz+by)&15;                       // diagonal spread, 256 blocks
        if(!gather_blk256(vox,bz,by,bx,blk)) { sampled++; continue; }
        uint32_t len=0; samp.len=0;
        if(mc_enc_block(C,blk,&samp,&len)) sample_bytes+=len;
        sampled++;
    }
    free(samp.p);
    mc_codec_ctx_free(C);
    float q=q0;
    if(sample_bytes){
        double est_total=(double)sample_bytes*(4096.0/sampled);
        double want_total=(double)(256.0*256.0*256.0)/target_ratio;
        q=(float)(q0*pow(est_total/want_total,1.0/MC_RC_GAMMA));
        if(q<0.5f)q=0.5f; if(q>24.0f)q=24.0f;
    }
    if(q_out)*q_out=q;
    return mc_archive_append_chunk_raw_q(a,lod,cz,cy,cx,vox,q);
}

int mc_archive_set_priors(struct mc_archive *a,
                          const uint16_t plo[8][32], const uint16_t phi[8][32]){
    if(!a||!plo||!phi) return -1;
    uint64_t off=w_alloc(a,MC_PRIORS_BYTES);
    if(off==~0ull) return -1;
    uint32_t magic=MC_PRIORS_MAGIC, ver=1;
    memcpy(a->base+off,&magic,4); memcpy(a->base+off+4,&ver,4);
    memcpy(a->base+off+8,plo,8*32*2);
    memcpy(a->base+off+8+8*32*2,phi,8*32*2);
    atomic_thread_fence(memory_order_release);
    w_write_u64(a,MCH_PRIOROFF,off);
    mc_codec_set_priors((const uint16_t*)plo,(const uint16_t*)phi);
    return 0;
}

// load priors from a header offset if present (open paths call this).
static void priors_load(const uint8_t *base){
    uint64_t off; memcpy(&off,base+MCH_PRIOROFF,8);
    if(!off) { mc_codec_set_priors(NULL,NULL); return; }
    uint32_t magic; memcpy(&magic,base+off,4);
    if(magic!=MC_PRIORS_MAGIC){ mc_codec_set_priors(NULL,NULL); return; }
    mc_codec_set_priors((const uint16_t*)(base+off+8),(const uint16_t*)(base+off+8+8*32*2));
}

uint64_t mc_archive_data_len(mc_archive *a){
    return a ? atomic_load_explicit(&a->cursor, memory_order_relaxed) : 0;
}

int mc_archive_set_metadata(mc_archive *a, const void *data, size_t len){
    if(!a || (len && !data)) return -1;
    if(len > MC_META_CAP) return -1;
    if(len) memcpy(a->base + MC_HDR, data, len);
    // publish the length AFTER the bytes so a concurrent flat read never sees
    // a length covering unwritten content (same commit-word discipline as blobs).
    atomic_thread_fence(memory_order_release);
    uint64_t l = len; memcpy(a->base + MCH_METALEN, &l, 8);
    return 0;
}

const char *mc_archive_metadata(mc_archive *a, size_t *out_len){
    if(!a){ if(out_len) *out_len = 0; return NULL; }
    return mc_metadata(a->base, out_len);
}

mc_cover mc_archive_chunk_coverage(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return MC_ABSENT;
    // Fast path: the coverage memo. A hit is O(1) and never touches the node tree
    // (the per-block tree walk on the render worker was the 49ms cost). Regions
    // made resident this session are always in the memo. A memo-miss falls back to
    // the tree (covers disk-loaded archives committed in a prior session) and
    // backfills the memo so the next probe is O(1).
    int m = mc_cov_probe(a, lod, cz, cy, cx);
    if(m >= 0) return (mc_cover)m;           // memoized (incl ABSENT) -> O(1)
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    uint64_t off = mc_resolve_chunk(a->base, root, cz,cy,cx);
    if(off==0){ mc_cov_put_state(a,lod,cz,cy,cx,MC_COV_ABSENT); return MC_ABSENT; }
    int zero = (off==MC_SLOT_ZERO);
    mc_cov_put(a, lod, cz, cy, cx, zero);   // backfill for next time
    return zero ? MC_ZERO : MC_PRESENT;
}

uint64_t mc_archive_chunk_offset(mc_archive *a, int lod, int cz,int cy,int cx){
    if(!a||lod<0||lod>7) return 0;
    uint64_t root = w_read_u64(a, MCH_ROOTOFF+(uint64_t)lod*8);
    return mc_resolve_chunk(a->base, root, cz,cy,cx);
}

// Decode one 16^3 block from the live mmap. LOCK-FREE: the codec scratch lives in
// a caller-owned per-thread mc_codec_ctx (quality set per chunk on that ctx), so
// concurrent decodes are safe without serialization. Blocks are fully
// self-contained (v2: per-block air mask in the payload), so a single block decode
// touches only the bitmap + its own payload. The mmap is read-only here; appends
// publish via a release fence so a resolved chunk_off always points at fully-written
// bytes.
// Internal: decode using a caller-owned ctx (worker pools own one ctx and reuse
// it across the chunk's 4096 blocks; sets the per-chunk q on that ctx).
static void mc_archive_decode_block_ctx(mc_codec_ctx *C, mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!a||chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_codec_ctx_set_quality(C,mc_chunk_q(a->base,chunk_off));   // per-chunk q
    uint64_t boff; uint32_t bl;
    if(!mc_block_range(a->base,chunk_off,bz,by,bx,&boff,&bl)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    // HARDENED: offsets derive from the on-disk length table; on a corrupt
    // archive they could point past the mapped file (SIGBUS). Bound against
    // the live append cursor. (For untrusted archives run mc_verify first —
    // the per-chunk xxh64 covers bitmap+lens+payloads.)
    uint64_t end=atomic_load_explicit(&a->cursor,memory_order_acquire);
    if(boff+bl>end){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(C,a->base+boff,bl,dst);
}
void mc_archive_decode_block(mc_archive *a, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(!a||chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    // This is the HOT render-path read (mc_cache miss -> src_archive -> here, per
    // block). A fresh ctx per call would re-run step_tab_build's 4096-powf loop
    // every block (~4% of render CPU). Keep one ctx per thread; step_tab_build
    // caches on quality, so same-q blocks skip the rebuild.
    static _Thread_local mc_codec_ctx *C = NULL;
    if(!C){ C=mc_codec_ctx_new(); if(!C){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; } }
    mc_archive_decode_block_ctx(C,a,chunk_off,bz,by,bx,dst);
}

// ---- parallel whole-chunk helpers ------------------------------------------
typedef struct {
    mc_archive *a; uint64_t chunk_off; mc_u8 *out; float q;
    _Atomic uint32_t next;
} dchunk_ctx;
static void *dchunk_worker(void *p){
    dchunk_ctx *c=p;
    mc_codec_ctx *C=mc_codec_ctx_new();     // one ctx per worker, reused per block
    if(!C) return NULL;
    mc_codec_ctx_set_quality(C,c->q);
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t bi=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(bi>=MC_GRID3) break;
        int bz=bi>>8, by=(bi>>4)&15, bx=bi&15;
        mc_archive_decode_block_ctx(C,c->a,c->chunk_off,bz,by,bx,blk);
        for(int z=0;z<MC_BLK;++z)for(int y=0;y<MC_BLK;++y)
            memcpy(c->out+((size_t)(bz*16+z)*MC_CHUNK+(by*16+y))*MC_CHUNK+(size_t)bx*16,
                   blk+((size_t)z*16+y)*16,16);
    }
    mc_codec_ctx_free(C);
    return NULL;
}
static int auto_threads(int nthreads){
    if(nthreads>0) return nthreads>16?16:nthreads;
    long nc=sysconf(_SC_NPROCESSORS_ONLN);
    int nt=(int)(nc>0?nc:4); return nt>16?16:nt;
}
void mc_archive_decode_chunk(mc_archive *a, uint64_t chunk_off, mc_u8 *out, int nthreads){
    if(!a||!out) return;
    if(chunk_off<=MC_SLOT_ZERO){ memset(out,0,(size_t)MC_CHUNK*MC_CHUNK*MC_CHUNK); return; }
    dchunk_ctx c={.a=a,.chunk_off=chunk_off,.out=out,.q=mc_chunk_q(a->base,chunk_off)};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads);
    if(nt<=1){ dchunk_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,dchunk_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// parallel encode: stripes of blocks into per-worker buffers, stitched in
// bitmap order so the blob is byte-identical to the serial path.
#define ENC_STRIPES 16
typedef struct {
    const mc_u8 *vox; float q;
    mc_buf bufs[ENC_STRIPES];
    uint16_t blen[MC_GRID3];
    uint8_t  bm[MC_BITMAP_BYTES];
    uint8_t  frac[MC_GRID3];
    _Atomic uint32_t next;
} echunk_ctx;
static void *echunk_worker(void *p){
    echunk_ctx *c=p;
    mc_codec_ctx *C=mc_codec_ctx_new();
    if(!C) return NULL;
    mc_codec_ctx_set_quality(C,c->q);
    u8 blk[MC_BLK*MC_BLK*MC_BLK];
    for(;;){
        uint32_t s=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(s>=ENC_STRIPES) break;
        uint32_t b0=s*(MC_GRID3/ENC_STRIPES), b1=b0+(MC_GRID3/ENC_STRIPES);
        for(uint32_t bi=b0;bi<b1;++bi){
            int bz=(int)(bi>>8), by=(int)((bi>>4)&15), bx=(int)(bi&15);
            if(!gather_blk256(c->vox,bz,by,bx,blk)) continue;
            { int cnt=0; for(int i=0;i<MC_BLK*MC_BLK*MC_BLK;++i) cnt+=blk[i]!=0;
              c->frac[bi]=(uint8_t)((cnt*15+2048)/4096);
              if(cnt&&!c->frac[bi]) c->frac[bi]=1; }
            uint32_t len=0;
            if(mc_enc_block(C,blk,&c->bufs[s],&len)){
                c->blen[bi]=(uint16_t)len;
                // No lock: each stripe spans MC_GRID3/ENC_STRIPES=256 blocks = 32
                // whole bitmap bytes, so stripes set DISJOINT bytes (8 blocks/byte).
                mc_bit_set(c->bm,bi);
            }
        }
    }
    mc_codec_ctx_free(C);
    return NULL;
}
int mc_archive_append_chunk_par(mc_archive *a, int lod, int cz,int cy,int cx,
                                const mc_u8 vox[256*256*256], float q, int nthreads){
    if(!a||lod<0||lod>7||!vox) return -1;
    echunk_ctx *c=calloc(1,sizeof *c);
    c->vox=vox; c->q=q>0?q:a->quality;
    atomic_store(&c->next,0);
    int nt=auto_threads(nthreads); if(nt>ENC_STRIPES)nt=ENC_STRIPES;
    if(nt<=1) echunk_worker(c);
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,echunk_worker,c);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
    }
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(c->bm[i]);
    int rc=0;
    if(npresent){
        // stitch: stripes hold payloads in ascending-bi order within the stripe,
        // so concatenating stripes in order matches the serial blob layout.
        stage_t st={0};
        uint16_t lens16[MC_GRID3];
        int nl=0;
        for(int bi=0;bi<MC_GRID3;++bi) if(mc_bit_get(c->bm,bi)) lens16[nl++]=c->blen[bi];
        float qq=c->q; uint64_t h=0;
        uint8_t fmap[MC_GRID3/2+64];
        uint32_t fml32=mc_enc_fracmap(c->frac,fmap,sizeof fmap);
        if(fml32>sizeof fmap) fml32=sizeof fmap;            // always true; gives the optimizer the bound
        uint16_t fml=(uint16_t)fml32;
        stage_put(&st,&qq,4); stage_put(&st,&h,8);          // hash patched below
        stage_put(&st,&fml,2); stage_put(&st,fmap,fml);
        stage_put(&st,c->bm,MC_BITMAP_BYTES);
        stage_put(&st,lens16,(size_t)nl*2);
        for(int s=0;s<ENC_STRIPES;++s) if(c->bufs[s].len) stage_put(&st,c->bufs[s].p,c->bufs[s].len);
        h=mc_chunk_compute_hash(st.p,(uint64_t)st.len);     // same bytes => same hash as serial
        memcpy(st.p+4,&h,8);
        rc=w_install_blob(a,lod,cz,cy,cx,st.p,st.len);
        free(st.p);
    }
    for(int s=0;s<ENC_STRIPES;++s) free(c->bufs[s].p);
    free(c);
    return rc;
}

int mc_archive_block_present(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(co<=MC_SLOT_ZERO) return 0;
    const u8 *bm=a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(a->base,co);
    return mc_bit_get(bm,((bz&15)*16+(by&15))*16+(bx&15));
}

float mc_archive_block_fraction(mc_archive *a, int lod, int bz, int by, int bx){
    if(!a||lod<0||lod>7||bz<0||by<0||bx<0) return 0.0f;
    uint64_t co=mc_archive_chunk_offset(a,lod,bz>>4,by>>4,bx>>4);
    if(co<=MC_SLOT_ZERO) return 0.0f;
    uint8_t fr[MC_GRID3];
    uint16_t fml=mc_chunk_fmaplen(a->base,co);
    if(!fml) return 0.0f;
    mc_dec_fracmap(a->base+co+MC_BLOB_HDR,fml,fr);
    return (float)fr[((bz&15)*16+(by&15))*16+(bx&15)]/15.0f;
}

static inline uint64_t mc_rng64(uint64_t *s){
    uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; *s=x; return x;
}
int mc_archive_sample_boxes(mc_archive *a, int lod, uint64_t seed, int count,
                            long dz, long dy, long dx, float min_frac,
                            mc_box *out){
    if(!a||!out||count<=0||dz<=0||dy<=0||dx<=0) return 0;
    uint32_t unx,uny,unz;
    memcpy(&unx,a->base+MCH_NX,4); memcpy(&uny,a->base+MCH_NY,4); memcpy(&unz,a->base+MCH_NZ,4);
    long NXl=(long)unx>>lod, NYl=(long)uny>>lod, NZl=(long)unz>>lod;
    if(NZl<dz||NYl<dy||NXl<dx) return 0;
    uint64_t s=seed?seed:0x9E3779B97F4A7C15ull;
    int got=0; long attempts=0, max_attempts=(long)count*256;
    while(got<count && attempts++<max_attempts){
        long z0=(long)(mc_rng64(&s)%(uint64_t)(NZl-dz+1));
        long y0=(long)(mc_rng64(&s)%(uint64_t)(NYl-dy+1));
        long x0=(long)(mc_rng64(&s)%(uint64_t)(NXl-dx+1));
        // mean block fraction over the touched block grid
        long bz0=z0>>4, bz1=(z0+dz-1)>>4, by0=y0>>4, by1=(y0+dy-1)>>4, bx0=x0>>4, bx1=(x0+dx-1)>>4;
        double fsum=0; long nb=0;
        for(long bz=bz0;bz<=bz1;++bz)for(long by=by0;by<=by1;++by)for(long bx=bx0;bx<=bx1;++bx){
            fsum+=mc_archive_block_fraction(a,lod,(int)bz,(int)by,(int)bx); nb++;
        }
        if(nb && fsum/nb>=min_frac){ out[got].z0=z0; out[got].y0=y0; out[got].x0=x0; got++; }
    }
    return got;
}

typedef struct {
    mc_archive *a; int lod;
    const mc_box *boxes; int n;
    long dz,dy,dx;
    mc_u8 *out; size_t bstride;
    _Atomic uint32_t next;
} crops_ctx;
static void *crops_worker(void *p){
    crops_ctx *c=p;
    for(;;){
        uint32_t i=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(i>=(uint32_t)c->n) break;
        mc_archive_read_region(c->a,c->lod,c->boxes[i].z0,c->boxes[i].y0,c->boxes[i].x0,
                               c->dz,c->dy,c->dx,
                               c->out+(size_t)i*c->bstride,
                               (size_t)c->dy*c->dx,(size_t)c->dx,1);
    }
    return NULL;
}
void mc_archive_read_regions(mc_archive *a, int lod, const mc_box *boxes, int n,
                             long dz, long dy, long dx,
                             mc_u8 *out, size_t batch_stride, int nthreads){
    if(!a||!boxes||n<=0||!out) return;
    crops_ctx c={.a=a,.lod=lod,.boxes=boxes,.n=n,.dz=dz,.dy=dy,.dx=dx,
                 .out=out,.bstride=batch_stride};
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads); if(nt>n)nt=n;
    if(nt<=1){ crops_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,crops_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

// ---- region read ------------------------------------------------------------
typedef struct {
    mc_archive *a; int lod;
    long z0,y0,x0,dz,dy,dx;
    mc_u8 *out; size_t sz,sy;
    int nbz,nby,nbx; long bz0,by0,bx0;     // touched block range
    _Atomic uint32_t next;
} region_ctx;
static void *region_worker(void *p){
    region_ctx *c=p;
    mc_u8 blk[MC_BLK*MC_BLK*MC_BLK];
    uint32_t nb=(uint32_t)(c->nbz*c->nby*c->nbx);
    for(;;){
        uint32_t w=atomic_fetch_add_explicit(&c->next,1,memory_order_relaxed);
        if(w>=nb) break;
        long bz=c->bz0+w/(c->nby*c->nbx);
        long by=c->by0+(w/c->nbx)%c->nby;
        long bx=c->bx0+w%c->nbx;
        uint64_t co=mc_archive_chunk_offset(c->a,c->lod,(int)(bz>>4),(int)(by>>4),(int)(bx>>4));
        long gz=bz*16, gy=by*16, gx=bx*16;            // block origin in voxels
        // intersection of this block with the region
        long iz0=gz>c->z0?gz:c->z0, iz1=(gz+16<c->z0+c->dz)?gz+16:c->z0+c->dz;
        long iy0=gy>c->y0?gy:c->y0, iy1=(gy+16<c->y0+c->dy)?gy+16:c->y0+c->dy;
        long ix0=gx>c->x0?gx:c->x0, ix1=(gx+16<c->x0+c->dx)?gx+16:c->x0+c->dx;
        if(iz0>=iz1||iy0>=iy1||ix0>=ix1) continue;
        int present=0;
        if(co){
            const u8 *bm=c->a->base+co+MC_BLOB_HDR+mc_chunk_fmaplen(c->a->base,co);
            present=mc_bit_get(bm,(int)(((bz&15)*16+(by&15))*16+(bx&15)));
        }
        if(present) mc_archive_decode_block(c->a,co,(int)(bz&15),(int)(by&15),(int)(bx&15),blk);
        long nrow=ix1-ix0;
        for(long z=iz0;z<iz1;++z)for(long y=iy0;y<iy1;++y){
            mc_u8 *dst=c->out+(size_t)(z-c->z0)*c->sz+(size_t)(y-c->y0)*c->sy+(size_t)(ix0-c->x0);
            if(present) memcpy(dst,blk+((size_t)(z-gz)*16+(y-gy))*16+(ix0-gx),(size_t)nrow);
            else        memset(dst,0,(size_t)nrow);
        }
    }
    return NULL;
}
void mc_archive_read_region(mc_archive *a, int lod,
                            long z0, long y0, long x0,
                            long dz, long dy, long dx,
                            mc_u8 *out, size_t sz, size_t sy, int nthreads){
    if(!a||lod<0||lod>7||!out||dz<=0||dy<=0||dx<=0) return;
    region_ctx c={.a=a,.lod=lod,.z0=z0,.y0=y0,.x0=x0,.dz=dz,.dy=dy,.dx=dx,
                  .out=out,.sz=sz,.sy=sy};
    c.bz0=z0>>4; c.by0=y0>>4; c.bx0=x0>>4;
    c.nbz=(int)(((z0+dz+15)>>4)-c.bz0);
    c.nby=(int)(((y0+dy+15)>>4)-c.by0);
    c.nbx=(int)(((x0+dx+15)>>4)-c.bx0);
    atomic_store(&c.next,0);
    int nt=auto_threads(nthreads);
    uint32_t nb=(uint32_t)(c.nbz*c.nby*c.nbx);
    if((uint32_t)nt>nb) nt=(int)nb;
    if(nt<=1){ region_worker(&c); return; }
    pthread_t th[16];
    for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,region_worker,&c);
    for(int t=0;t<nt;++t) pthread_join(th[t],NULL);
}

void mc_archive_close(mc_archive *a){
    if(!a) return;
    uint64_t cur = atomic_load(&a->cursor);
    w_write_u64(a, MCH_TOTLEN, cur);
    msync(a->base, cur, MS_SYNC);
    munmap(a->base, a->reserve);
    if(ftruncate(a->fd,(off_t)cur)!=0) perror("mc_archive_close: ftruncate");
    close(a->fd);
    pthread_mutex_destroy(&a->grow_mu);
    free(a->cov);
    free(a);
}

#else // !MC_HAVE_MMAP — the appendable archive requires mmap/ftruncate.
mc_archive *mc_archive_open(const char *p,int d,float q){ (void)p;(void)d;(void)q;
    fprintf(stderr,"mc_archive: requires a POSIX mmap platform\n"); return NULL; }
int mc_archive_append_chunk_raw(mc_archive*a,int l,int z,int y,int x,const mc_u8*v){ (void)a;(void)l;(void)z;(void)y;(void)x;(void)v; return -1; }
int mc_archive_append_chunk_compressed(mc_archive*a,int l,int z,int y,int x,const uint8_t*b,size_t n){ (void)a;(void)l;(void)z;(void)y;(void)x;(void)b;(void)n; return -1; }
mc_cover mc_archive_chunk_coverage(mc_archive*a,int l,int z,int y,int x){ (void)a;(void)l;(void)z;(void)y;(void)x; return MC_ABSENT; }
uint64_t mc_archive_chunk_offset(mc_archive*a,int l,int z,int y,int x){ (void)a;(void)l;(void)z;(void)y;(void)x; return 0; }
void mc_archive_decode_block(mc_archive*a,uint64_t o,int z,int y,int x,mc_u8*d){ (void)a;(void)o;(void)z;(void)y;(void)x; memset(d,0,MC_BLK*MC_BLK*MC_BLK); }
void mc_archive_close(mc_archive*a){ (void)a; }
#endif

// ============================================================================
// VERIFY — walk every chunk of every LOD, recompute xxh64, compare to stored.
// ============================================================================
long mc_verify_archive(const uint8_t *arc, size_t len, int verbose){
    (void)len;
    long bad=0, total=0;
    for(int lod=0;lod<8;++lod){
        uint64_t root; memcpy(&root,arc+MCH_ROOTOFF+(uint64_t)lod*8,8);
        if(!root) continue;
        for(int n2=0;n2<MC_GRID3;++n2){
            uint64_t inner; memcpy(&inner,arc+root+(size_t)n2*8,8);
            if(!inner) continue;
            for(int n1=0;n1<MC_GRID3;++n1){
                uint64_t shard; memcpy(&shard,arc+inner+(size_t)n1*8,8);
                if(!shard) continue;
                for(int n0=0;n0<MC_GRID3;++n0){
                    uint64_t co; memcpy(&co,arc+shard+(size_t)n0*8,8);
                    if(co<=MC_SLOT_ZERO) continue;
                    total++;
                    uint64_t blen=mc_chunk_blob_len(arc,co);
                    uint64_t want=mc_chunk_stored_hash(arc,co);
                    uint64_t got=mc_chunk_compute_hash(arc+co,blen);
                    if(want!=got){
                        bad++;
                        if(verbose) fprintf(stderr,"mc_verify: lod %d node %d/%d/%d chunk@%llu CORRUPT\n",
                            lod,n2,n1,n0,(unsigned long long)co);
                    }
                }
            }
        }
    }
    if(verbose) fprintf(stderr,"mc_verify: %ld chunks checked, %ld corrupt\n",total,bad);
    return bad;
}

// ============================================================================
// READER (flat buffer + streaming byte-source)
// ============================================================================
const char *mc_metadata(const uint8_t *arc, size_t *out_len){
    uint64_t off,len; memcpy(&off,arc+MCH_METAOFF,8); memcpy(&len,arc+MCH_METALEN,8);
    if(!off) off=MC_HDR;
    if(out_len) *out_len=(size_t)len;
    return (const char*)(arc+off);
}

#define MC_RD_NODE_CACHE 512   // cached node tables per streaming reader (512*32KB = 16MB):
                               // a pan across a large volume touches many subtrees; a
                               // churned-out table costs a serial ranged GET to re-fetch
struct mc_reader {
    const uint8_t *arc;        // flat mode: archive buffer; streaming: NULL
    size_t len;
    uint64_t roots[8];
    int nx, ny, nz;            // LOD0 dims from the header
    float quality;             // quality the archive was built at
    u8 priors[MC_PRIORS_BYTES]; int has_priors;   // raw per-volume prior blob
    mc_read_fn read; void *read_ud;   // streaming mode
    // streaming scratch: a fetched window of the current chunk blob.
    u8 *cbuf; uint64_t cbuf_off; uint64_t cbuf_len;
    // partial-fetch mode: per-chunk header cache (bitmap + lens) + one payload.
    int partial;
    u8 hdr[MC_BLOB_HDR + MC_BITMAP_BYTES + MC_GRID3*2]; uint64_t hdr_off; int hdr_np; uint16_t hdr_fml;
    u8 *pbuf; size_t pbuf_cap;
    // streaming node-table cache: resolving a chunk needs 3 dependent 32KB
    // table reads; without a cache every resolve re-fetches them (3 GETs per
    // chunk over S3). FIFO of the last MC_RD_NODE_CACHE tables.
    uint64_t ntab_off[MC_RD_NODE_CACHE];
    u8 *ntab[MC_RD_NODE_CACHE];
    int ntab_next;
    // owned per-reader codec context (a reader's decode scratch — cbuf/pbuf/hdr
    // — is already non-reentrant, so one codec ctx per reader is the right scope).
    mc_codec_ctx *codec;
};

static void reader_hdr_load(mc_reader *r, const u8 *hdr){
    for(int l=0;l<8;++l) memcpy(&r->roots[l], hdr+MCH_ROOTOFF+l*8, 8);
    uint32_t ux,uy,uz;
    memcpy(&ux,hdr+MCH_NX,4); memcpy(&uy,hdr+MCH_NY,4); memcpy(&uz,hdr+MCH_NZ,4);
    r->nx=(int)ux; r->ny=(int)uy; r->nz=(int)uz;
    memcpy(&r->quality,hdr+MCH_QUALITY,4);
}

mc_reader *mc_open(const uint8_t *arc, size_t len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->arc=arc; r->len=len;
    r->codec=mc_codec_ctx_new();
    reader_hdr_load(r, arc);
    uint64_t poff; memcpy(&poff,arc+MCH_PRIOROFF,8);
    if(poff){ memcpy(r->priors,arc+poff,MC_PRIORS_BYTES); r->has_priors=1; }
    priors_load(arc);
    return r;
}

// streaming: fetch exactly len bytes at off into dst (via callback).
static int sread(mc_reader *r, uint64_t off, uint32_t len, u8 *dst){
    return r->read(r->read_ud, off, len, dst);
}

mc_reader *mc_open_streaming(mc_read_fn read, void *ud, uint64_t total_len){
    mc_codec_init();
    mc_reader *r=calloc(1,sizeof *r); r->read=read; r->read_ud=ud; r->len=(size_t)total_len;
    r->codec=mc_codec_ctx_new();
    u8 hdr[MC_HDR];
    if(read(ud,0,MC_HDR,hdr)!=0){ mc_codec_ctx_free(r->codec); free(r); return NULL; }
    uint32_t magic; memcpy(&magic,hdr+MCH_MAGIC,4);
    if(magic!=MC_MAGIC){ mc_codec_ctx_free(r->codec); free(r); return NULL; }
    reader_hdr_load(r, hdr);
    // per-volume priors: flat open gets them via priors_load(arc); a streaming
    // reader must pull the blob through the callback or HF decode is wrong.
    uint64_t poff; memcpy(&poff,hdr+MCH_PRIOROFF,8);
    if(poff){
        u8 pb[MC_PRIORS_BYTES];
        uint32_t pm=0;
        if(read(ud,poff,MC_PRIORS_BYTES,pb)==0) memcpy(&pm,pb,4);
        if(pm==MC_PRIORS_MAGIC){
            memcpy(r->priors,pb,MC_PRIORS_BYTES); r->has_priors=1;
            mc_codec_set_priors((const uint16_t*)(pb+8),(const uint16_t*)(pb+8+8*32*2));
        } else {
            mc_codec_set_priors(NULL,NULL);
        }
    } else {
        mc_codec_set_priors(NULL,NULL);
    }
    return r;
}

// Raw per-volume prior arrays (plo/phi as u16[8][32]); 0 if the archive has none.
// Feed into mc_archive_set_priors to make a local mirror decode identically.
int mc_reader_priors(mc_reader *r, const uint16_t **plo, const uint16_t **phi){
    if(!r||!r->has_priors) return 0;
    if(plo)*plo=(const uint16_t*)(r->priors+8);
    if(phi)*phi=(const uint16_t*)(r->priors+8+8*32*2);
    return 1;
}

// Total byte length of the chunk blob at `chunk_off` (flat or streaming reader).
// 0 on error. Pair with mc_chunk_offset to range-copy compressed chunks verbatim.
uint64_t mc_reader_chunk_blob_len(mc_reader *r, uint64_t chunk_off){
    if(!r||!chunk_off) return 0;
    if(r->arc) return mc_chunk_blob_len(r->arc, chunk_off);
    u8 h[MC_BLOB_HDR];
    if(sread(r,chunk_off,MC_BLOB_HDR,h)!=0) return 0;
    uint16_t fml; memcpy(&fml,h+MC_BLOB_HDR-2,2);
    const uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    u8 bm[MC_BITMAP_BYTES];
    if(sread(r,bm_off,MC_BITMAP_BYTES,bm)!=0) return 0;
    int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm[i]);
    if(!np) return bm_off + MC_BITMAP_BYTES - chunk_off;
    u8 *lens=malloc((size_t)np*2);
    if(!lens||sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)((size_t)np*2),lens)!=0){ free(lens); return 0; }
    uint64_t pay=0; for(int i=0;i<np;++i){ uint16_t l; memcpy(&l,lens+(size_t)i*2,2); pay+=l; }
    free(lens);
    return bm_off + MC_BITMAP_BYTES + (uint64_t)np*2 + pay - chunk_off;
}

// Copy `len` bytes of the chunk blob at `chunk_off` into `dst` (flat or streaming
// reader). For .mca -> .mca verbatim copy: mc_reader_chunk_blob_len then this then
// mc_archive_append_chunk_compressed. NOT thread-safe on a streaming reader (single
// cbuf/codec ctx) -- caller serializes. Returns 0 on success.
int mc_reader_read_blob(mc_reader *r, uint64_t chunk_off, size_t len, uint8_t *dst){
    if(!r||!chunk_off||!len||!dst) return -1;
    if(r->arc){ memcpy(dst, r->arc + chunk_off, len); return 0; }
    // streaming: range-read in <=2^31 chunks (sread takes a uint32 len).
    size_t done=0;
    while(done<len){
        uint32_t n = (len-done > 0x40000000u) ? 0x40000000u : (uint32_t)(len-done);
        if(sread(r, chunk_off+done, n, dst+done)!=0) return -1;
        done += n;
    }
    return 0;
}

void mc_reader_dims(mc_reader *r, int *nx, int *ny, int *nz){
    if(nx)*nx=r?r->nx:0; if(ny)*ny=r?r->ny:0; if(nz)*nz=r?r->nz:0;
}
float mc_reader_quality(mc_reader *r){ return r?r->quality:0.f; }
int mc_reader_nlods(mc_reader *r){
    if(!r) return 0;
    int n=0; for(int l=0;l<8;++l) if(r->roots[l]) n=l+1;
    return n;
}

void mc_close(mc_reader *r){ if(!r)return;
    for(int i=0;i<MC_RD_NODE_CACHE;++i) free(r->ntab[i]);
    mc_codec_ctx_free(r->codec);
    free(r->cbuf); free(r->pbuf); free(r); }
// Partial-fetch mode (streaming readers only): decode a block by fetching just
// the chunk's bitmap+length table (cached per chunk, <=8.7KB) plus that block's
// own payload, instead of the whole chunk blob. Wins cold random-access latency
// over high-latency byte sources (S3); leave OFF when scanning whole chunks.
void mc_reader_set_partial_fetch(mc_reader *r, int on){ if(r){ r->partial=on; r->hdr_off=~0ull; } }
void mc_reader_set_quality(mc_reader *r, float q){ if(r&&r->codec) mc_codec_ctx_set_quality(r->codec,q); }

// streaming chunk-offset resolve: pull node tables on demand (each is
// MC_NODE_BYTES), memoized in the reader's FIFO node-table cache so repeated
// resolves (scans, neighborhoods) cost zero extra reads.
static const u8 *sfetch_node(mc_reader *r, uint64_t off){
    for(int i=0;i<MC_RD_NODE_CACHE;++i)
        if(r->ntab[i] && r->ntab_off[i]==off) return r->ntab[i];
    int slot=r->ntab_next; r->ntab_next=(slot+1)%MC_RD_NODE_CACHE;
    if(!r->ntab[slot]) r->ntab[slot]=malloc(MC_NODE_BYTES);
    if(sread(r,off,MC_NODE_BYTES,r->ntab[slot])!=0){ free(r->ntab[slot]); r->ntab[slot]=0; return NULL; }
    r->ntab_off[slot]=off;
    return r->ntab[slot];
}
static uint64_t sresolve_chunk(mc_reader *r,int lod,int cz,int cy,int cx,int *err){
    uint64_t node = r->roots[lod];
    for(int nib=MC_TREE_LEVELS-1; nib>=0; --nib){
        if(!node) return 0;                       // genuinely absent (air)
        const u8 *tbl=sfetch_node(r,node);
        if(!tbl){ if(err)*err=1; return 0; }      // FETCH FAILED -- not absent!
        int idx=(mc_nib(cz,nib)*16+mc_nib(cy,nib))*16+mc_nib(cx,nib);
        uint64_t child; memcpy(&child,tbl+(size_t)idx*8,8);
        node=child;
    }
    return node;
}

// As mc_chunk_offset, but distinguishes "resolved to absent" (ret 0, *err 0)
// from "node-table read FAILED" (ret 0, *err 1). A streaming caller that maps
// offset 0 to permanent air MUST use this: a transient network error (expired
// creds, timeout) otherwise poisons the region as ZERO forever.
uint64_t mc_chunk_offset_chk(mc_reader *r,int lod,int cz,int cy,int cx,int *err){
    if(err)*err=0;
    if(!r||lod<0||lod>7) return 0;
    if(r->arc) return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx);  // flat: no I/O
    return sresolve_chunk(r,lod,cz,cy,cx,err);
}

uint64_t mc_chunk_offset(mc_reader *r,int lod,int cz,int cy,int cx){
    if(lod<0||lod>7) return 0;
    if(r->arc) return mc_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx);
    return sresolve_chunk(r,lod,cz,cy,cx,NULL);
}

// streaming: ensure the whole chunk blob at chunk_off is cached in r->cbuf, return ptr.
static const u8 *sfetch_chunk(mc_reader *r, uint64_t chunk_off){
    if(r->cbuf && r->cbuf_off==chunk_off) return r->cbuf;
    // fetch header (q/hash/fmaplen), then bitmap + lens for total length, then
    // the full blob in one GET.
    u8 bh[MC_BLOB_HDR];
    if(sread(r,chunk_off,MC_BLOB_HDR,bh)!=0) return NULL;
    uint16_t fml; memcpy(&fml,bh+12,2);
    uint64_t bm_off = chunk_off + MC_BLOB_HDR + fml;
    u8 bm[MC_BITMAP_BYTES];
    if(sread(r,bm_off,MC_BITMAP_BYTES,bm)!=0) return NULL;
    int npresent=0; for(int i=0;i<MC_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    u8 *lens=malloc((size_t)npresent*2);
    if(npresent && !lens) return NULL;
    if(npresent && sread(r,bm_off+MC_BITMAP_BYTES,(uint32_t)(npresent*2),lens)!=0){ free(lens); return NULL; }
    uint64_t paybytes=0; for(int s=0;s<npresent;++s){ uint16_t l; memcpy(&l,lens+(size_t)s*2,2); paybytes+=l; }
    free(lens);
    uint64_t total = (bm_off+MC_BITMAP_BYTES+(uint64_t)npresent*2+paybytes) - chunk_off;
    { void *tmp=realloc(r->cbuf,total); if(!tmp) return NULL; r->cbuf=tmp; }
    if(sread(r,chunk_off,(uint32_t)total,r->cbuf)!=0) return NULL;
    r->cbuf_off=chunk_off; r->cbuf_len=total;
    return r->cbuf;
}

// partial-fetch path: header cache + single-payload range read.
static int spartial_decode(mc_reader *r, uint64_t chunk_off, int bi, mc_u8 *dst){
    if(r->hdr_off!=chunk_off){
        if(sread(r,chunk_off,MC_BLOB_HDR,r->hdr)!=0) return -1;
        uint16_t fml; memcpy(&fml,r->hdr+12,2);
        r->hdr_fml=fml;
        if(sread(r,chunk_off+MC_BLOB_HDR+fml,MC_BITMAP_BYTES,r->hdr+MC_BLOB_HDR)!=0) return -1;
        const u8 *bm0=r->hdr+MC_BLOB_HDR;
        int np=0; for(int i=0;i<MC_BITMAP_BYTES;++i) np+=__builtin_popcount(bm0[i]);
        if(np && sread(r,chunk_off+MC_BLOB_HDR+fml+MC_BITMAP_BYTES,(uint32_t)(np*2),r->hdr+MC_BLOB_HDR+MC_BITMAP_BYTES)!=0) return -1;
        r->hdr_np=np; r->hdr_off=chunk_off;
    }
    { float q; memcpy(&q,r->hdr,4); mc_codec_ctx_set_quality(r->codec,q); }
    const u8 *bm=r->hdr+MC_BLOB_HDR;
    if(!mc_bit_get(bm,bi)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return 0; }
    int slot=mc_rank(bm,bi);
    const u8 *lens=bm+MC_BITMAP_BYTES;
    uint64_t cum=0; for(int s2=0;s2<slot;++s2){ uint16_t l; memcpy(&l,lens+(size_t)s2*2,2); cum+=l; }
    uint16_t mylen; memcpy(&mylen,lens+(size_t)slot*2,2);
    uint64_t pay=chunk_off+MC_BLOB_HDR+r->hdr_fml+MC_BITMAP_BYTES+(uint64_t)r->hdr_np*2+cum;
    if(r->pbuf_cap<mylen){ void *tmp=realloc(r->pbuf,mylen); if(!tmp) return -1; r->pbuf=tmp; r->pbuf_cap=mylen; }
    if(sread(r,pay,mylen,r->pbuf)!=0) return -1;
    mc_dec_block(r->codec,r->pbuf,mylen,dst);
    return 0;
}

void mc_decode_block(mc_reader *r, uint64_t chunk_off, int bz,int by,int bx, mc_u8 *dst){
    if(chunk_off<=MC_SLOT_ZERO){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    if(!r->arc && r->partial){
        if(spartial_decode(r,chunk_off,(bz*16+by)*16+bx,dst)==0) return;
        memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return;
    }
    // resolve the chunk-blob base pointer (flat mmap or streamed window).
    const u8 *blob_base;       // points at the chunk blob start
    uint64_t blob_origin;      // absolute archive offset of that blob start
    if(r->arc){ blob_base=r->arc; blob_origin=0; }            // absolute offsets index into arc
    else {
        const u8 *cb = sfetch_chunk(r,chunk_off);
        if(!cb){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
        blob_base = cb - chunk_off;   // so blob_base + abs_off lands inside the window
        blob_origin = 0;
    }
    (void)blob_origin;
    mc_codec_ctx_set_quality(r->codec,mc_chunk_q(blob_base,chunk_off));
    uint64_t boff; uint32_t blen;
    if(!mc_block_range(blob_base,chunk_off,bz,by,bx,&boff,&blen)){ memset(dst,0,MC_BLK*MC_BLK*MC_BLK); return; }
    mc_dec_block(r->codec,blob_base+boff,blen,dst);
}

// ============================================================================
// mc_cache.c — sharded CLOCK/NRU decoded-block cache. See mc_cache.h.
// ============================================================================
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <unistd.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <sys/mman.h>
  #include <pthread.h>
  #define MC_CACHE_MMAP 1
#else
  #define MC_CACHE_MMAP 0
#endif

#define BLK_BYTES 4096u
#define NSHARD    64          // power of two
#define EMPTY_KEY 0ull

// key: 1 (always-set guard) | lod(3) | bz(20) | by(20) | bx(20)
static inline uint64_t bkey(int lod,int bz,int by,int bx){
    return (1ull<<63) | ((uint64_t)(lod&7)<<60)
         | ((uint64_t)(bz&0xFFFFF)<<40) | ((uint64_t)(by&0xFFFFF)<<20) | (uint64_t)(bx&0xFFFFF);
}
static inline uint64_t khash(uint64_t k){
    k^=k>>33; k*=0xFF51AFD7ED558CCDull; k^=k>>33; k*=0xC4CEB9FE1A85EC53ull; k^=k>>33;
    return k;
}

// one shard: its own slice of the arena, its own map, eviction state. No lock:
// all mutation is single-owner (THAW partitions fill by shard), reads are pure.
typedef struct {
    uint64_t *map_key;        // open addressing, linear probe, backward-shift delete
    uint32_t *map_slot;
    uint32_t  map_cap;        // power of two
    uint64_t *slot_key;       // per-slot reverse key (EMPTY_KEY = free)
    uint8_t  *slot_ref;       // CLOCK ref bit / S3-FIFO 2-bit freq
    uint32_t  nslot, used, hand;
    mc_u8    *arena;          // nslot * 4KB
    uint64_t  hits, misses, evictions;
    uint32_t *slot_epoch;            // pin: slot used by the current epoch
    // S3-FIFO state: two slot-id rings + a ghost fingerprint ring w/ set.
    uint32_t *fs, fs_head, fs_tail, fs_cap;     // small queue (ring)
    uint32_t *fm, fm_head, fm_tail, fm_cap;     // main queue (ring)
    uint32_t *gfp, g_head, g_cap;               // ghost fingerprints (ring)
    int32_t  *gset; uint32_t gset_cap;          // fp -> ring idx, open addressing
    uint8_t  *slot_inmain;                      // which queue a slot lives in
} shard_t;

#define FREQ_MAX 3

#define MISSQ_CAP 65536
struct mc_cache {
    int policy;               // mc_cache_policy
    _Atomic int frozen;
    _Atomic int phase_mode;          // set by the first freeze(); until then the
                                     // cache is the plain always-thread-safe
                                     // multi-reader/multi-writer cache and pins
                                     // are inert (no behavior change for
                                     // clients that never tick)
    _Atomic uint32_t epoch;          // bumped at thaw; pins compare against it
    // Miss set: open-addressing dedup table (slot 0 == empty). A thin slice
    // re-probes the same 16^3 block from many pixels/bands; recording each unique
    // absent block ONCE (not once per pixel) keeps the per-frame fill set ~= the
    // real working set instead of ~100x larger. Lock-free insert via CAS.
    _Atomic uint64_t missq[MISSQ_CAP];
    _Atomic uint32_t miss_n;          // approx live entries (for drain early-out)
    shard_t sh[NSHARD];
    mc_cache_src_fn src; void *src_ud;
    size_t arena_bytes;
    void *arena_base;
    // reader binding (single-owner decode; no lock)
    struct mc_reader *rd;
    struct mc_archive *ar;
};

static uint32_t pow2_at_least(uint32_t v){ uint32_t p=1; while(p<v)p<<=1; return p; }
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot);
static inline int cache_frozen(mc_cache *c){ return atomic_load_explicit(&c->frozen,memory_order_acquire); }
// Dedup insert into the miss set. Lock-free: hash, linear-probe a bounded run,
// CAS an empty slot to `key`. If the key is already present (or lands in the
// probe run), do nothing — recording the same absent block from another pixel is
// free. Bounded probe (table is generously sized vs the per-frame working set);
// on a full run we just drop the record (it re-records next frame).
static void miss_record(mc_cache *c, uint64_t key){
    if(!key) return;
    uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>40);
    for(int p=0;p<8;++p){
        uint32_t i=(h+(uint32_t)p)&(MISSQ_CAP-1);
        uint64_t cur=atomic_load_explicit(&c->missq[i],memory_order_relaxed);
        if(cur==key) return;                          // already recorded
        if(cur==0){
            uint64_t exp=0;
            if(atomic_compare_exchange_strong_explicit(&c->missq[i],&exp,key,
                   memory_order_relaxed,memory_order_relaxed)){
                atomic_fetch_add_explicit(&c->miss_n,1,memory_order_relaxed);
                return;
            }
            if(exp==key) return;                      // racer inserted same key
        }
    }
}

static void shard_free_tables(shard_t *sh);
static int  shard_init_tables(shard_t *sh, mc_u8 *arena_slice, uint32_t per);

mc_cache *mc_cache_new(size_t bytes, mc_cache_src_fn src, void *src_ud){
    mc_cache *c=calloc(1,sizeof *c);
    if(!c) return NULL;
    c->src=src; c->src_ud=src_ud;
    size_t nslot_total = bytes/BLK_BYTES; if(nslot_total<NSHARD) nslot_total=NSHARD;
    uint32_t per = (uint32_t)(nslot_total/NSHARD); if(per<1)per=1;
    c->arena_bytes = (size_t)per*NSHARD*BLK_BYTES;
#if MC_CACHE_MMAP
    c->arena_base = mmap(NULL,c->arena_bytes,PROT_READ|PROT_WRITE,
                         MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(c->arena_base==MAP_FAILED){ free(c); return NULL; }
#else
    c->arena_base = malloc(c->arena_bytes);
    if(!c->arena_base){ free(c); return NULL; }
#endif
    for(int s=0;s<NSHARD;++s){
        if(!shard_init_tables(&c->sh[s], (mc_u8*)c->arena_base + (size_t)s*per*BLK_BYTES, per)){
            for(int j=0;j<s;++j) shard_free_tables(&c->sh[j]);
#if MC_CACHE_MMAP
            munmap(c->arena_base,c->arena_bytes);
#else
            free(c->arena_base);
#endif
            free(c); return NULL;
        }
    }
    atomic_store(&c->epoch,1);
    return c;
}
void mc_cache_set_policy(mc_cache *c, mc_cache_policy p){ if(c) c->policy=(int)p; }

// free a shard's lookup/eviction tables (NOT its mutex, NOT the shared arena).
static void shard_free_tables(shard_t *sh){
    free(sh->map_key); free(sh->map_slot); free(sh->slot_key); free(sh->slot_ref);
    free(sh->fs); free(sh->fm); free(sh->gfp); free(sh->gset); free(sh->slot_inmain);
    free(sh->slot_epoch);
}

// (re)allocate a shard's tables for `per` slots pointing at `arena_slice`.
// Resets the shard to empty. Mutex assumed already init'd + held during resize.
// Returns 0 on allocation failure (shard left freeable: every table is NULL or owned).
static int shard_init_tables(shard_t *sh, mc_u8 *arena_slice, uint32_t per){
    sh->nslot=per; sh->hand=0; sh->used=0;
    sh->arena=arena_slice;
    sh->map_cap=pow2_at_least(per*2);
    sh->map_key=calloc(sh->map_cap,8);
    sh->map_slot=calloc(sh->map_cap,4);
    sh->slot_key=calloc(per,8);
    sh->slot_ref=calloc(per,1);
    sh->fs_cap=per+1; sh->fm_cap=per+1;
    sh->fs=malloc(4u*sh->fs_cap); sh->fm=malloc(4u*sh->fm_cap);
    sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
    sh->g_cap=per; sh->gfp=calloc(sh->g_cap,4);
    sh->gset_cap=pow2_at_least(per*2); sh->gset=malloc(4u*sh->gset_cap);
    sh->slot_inmain=calloc(per,1);
    sh->slot_epoch=calloc(per,4);
    if(!sh->map_key||!sh->map_slot||!sh->slot_key||!sh->slot_ref||!sh->fs||!sh->fm||
       !sh->gfp||!sh->gset||!sh->slot_inmain||!sh->slot_epoch) return 0;
    memset(sh->gset,0xFF,4u*sh->gset_cap);
    return 1;
}

void mc_cache_free(mc_cache *c){
    if(!c) return;
    for(int s=0;s<NSHARD;++s) shard_free_tables(&c->sh[s]);
#if MC_CACHE_MMAP
    munmap(c->arena_base,c->arena_bytes);
#else
    free(c->arena_base);
#endif
    free(c);
}

// ---- runtime budget control -------------------------------------------------
// Live-resize the decoded-block cache to `new_bytes`. The cache is just a cache,
// so resizing DISCARDS resident blocks (re-decode on demand) rather than
// migrating them. Single-owner: call only between ticks (no fill in flight).
// Returns the byte budget actually installed (rounded to whole slots), or 0.
size_t mc_cache_resize(mc_cache *c, size_t new_bytes){
    if(!c) return 0;
    size_t nslot_total = new_bytes/BLK_BYTES; if(nslot_total<NSHARD) nslot_total=NSHARD;
    uint32_t per = (uint32_t)(nslot_total/NSHARD); if(per<1)per=1;
    size_t new_arena = (size_t)per*NSHARD*BLK_BYTES;
#if MC_CACHE_MMAP
    void *na = mmap(NULL,new_arena,PROT_READ|PROT_WRITE,
                    MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
    if(na==MAP_FAILED) return 0;
#else
    void *na = malloc(new_arena);
    if(!na) return 0;
#endif
    // Build the new shard tables into temporaries first; only commit (free old +
    // swap) if every shard succeeds, so an OOM leaves the old cache fully intact.
    shard_t ns[NSHARD]; memset(ns,0,sizeof ns);
    int ok=1;
    for(int s=0;s<NSHARD && ok;++s)
        ok = shard_init_tables(&ns[s], (mc_u8*)na + (size_t)s*per*BLK_BYTES, per);
    if(!ok){
        for(int s=0;s<NSHARD;++s) shard_free_tables(&ns[s]);
#if MC_CACHE_MMAP
        munmap(na,new_arena);
#else
        free(na);
#endif
        return 0;
    }
    for(int s=0;s<NSHARD;++s){ shard_free_tables(&c->sh[s]); c->sh[s]=ns[s]; }
    void *old = c->arena_base; size_t old_bytes = c->arena_bytes;
    c->arena_base = na; c->arena_bytes = new_arena;
    atomic_fetch_add(&c->epoch,1);   // invalidate outstanding pins
#if MC_CACHE_MMAP
    munmap(old,old_bytes);
#else
    free(old);
#endif
    return new_arena;
}

size_t mc_cache_capacity_bytes(const mc_cache *c){ return c ? c->arena_bytes : 0; }

size_t mc_cache_used_bytes(mc_cache *c){
    if(!c) return 0;
    size_t used=0;
    for(int s=0;s<NSHARD;++s) used += (size_t)c->sh[s].used;   // racy read ok (stat)
    return used*BLK_BYTES;
}

double mc_cache_usage_fraction(mc_cache *c){
    if(!c||c->arena_bytes==0) return 0.0;
    return (double)mc_cache_used_bytes(c)/(double)c->arena_bytes;
}

// ---- shard map ops (shard lock held) ----
static inline uint32_t map_find(shard_t *sh, uint64_t key){   // -> map index or UINT32_MAX
    uint32_t m=sh->map_cap-1, i=(uint32_t)khash(key)&m;
    for(;;){
        if(sh->map_key[i]==key) return i;
        if(sh->map_key[i]==EMPTY_KEY) return UINT32_MAX;
        i=(i+1)&m;
    }
}
static void map_insert(shard_t *sh, uint64_t key, uint32_t slot){
    uint32_t m=sh->map_cap-1, i=(uint32_t)khash(key)&m;
    while(sh->map_key[i]!=EMPTY_KEY) i=(i+1)&m;
    sh->map_key[i]=key; sh->map_slot[i]=slot;
}
static void map_delete(shard_t *sh, uint64_t key){            // backward-shift deletion
    uint32_t m=sh->map_cap-1, i=map_find(sh,key);
    if(i==UINT32_MAX) return;
    uint32_t j=i;
    for(;;){
        j=(j+1)&m;
        uint64_t kj=sh->map_key[j];
        if(kj==EMPTY_KEY) break;
        uint32_t home=(uint32_t)khash(kj)&m;
        // can kj move into the hole at i? (it must not cross its home slot)
        uint32_t dist_ij=(i-home)&m, dist_jj=(j-home)&m;
        if(dist_ij<=dist_jj){ sh->map_key[i]=kj; sh->map_slot[i]=sh->map_slot[j]; i=j; }
    }
    sh->map_key[i]=EMPTY_KEY;
}
// CLOCK sweep: find a victim slot (clearing ref bits as we pass).
static uint32_t reclaim_slot(shard_t *sh){
    if(sh->used<sh->nslot){                       // free slot exists: linear scan from hand
        for(uint32_t k=0;k<sh->nslot;++k){
            uint32_t i=(sh->hand+k)%sh->nslot;
            if(sh->slot_key[i]==EMPTY_KEY){ sh->hand=(i+1)%sh->nslot; sh->used++; return i; }
        }
    }
    for(;;){
        uint32_t i=sh->hand; sh->hand=(sh->hand+1)%sh->nslot;
        if(sh->slot_key[i]==EMPTY_KEY) return i;            // invalidated: reuse
        if(sh->slot_ref[i]){ sh->slot_ref[i]=0; continue; }
        map_delete(sh,sh->slot_key[i]);
        sh->evictions++;
        return i;
    }
}

// ---- S3-FIFO (SOSP'23) ----
static inline uint32_t s3_fp(uint64_t key){ uint32_t f=(uint32_t)(khash(key)>>17); return f?f:1; }
static inline uint32_t ring_len(uint32_t h,uint32_t t,uint32_t cap){ return (t+cap-h)%cap; }
static void ghost_put(shard_t *sh, uint32_t fp){
    uint32_t old=sh->gfp[sh->g_head];
    if(old){  // remove the overwritten fingerprint from the set
        uint32_t m=sh->gset_cap-1, i=(old*0x9E3779B1u)&m;
        while(sh->gset[i]!=-1){ if(sh->gfp[sh->gset[i]]==old && (uint32_t)sh->gset[i]==sh->g_head){ sh->gset[i]=-1; break; } i=(i+1)&m; }
    }
    sh->gfp[sh->g_head]=fp;
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    while(sh->gset[i]!=-1) i=(i+1)&m;
    sh->gset[i]=(int32_t)sh->g_head;
    sh->g_head=(sh->g_head+1)%sh->g_cap;
}
static int ghost_has(shard_t *sh, uint32_t fp){
    uint32_t m=sh->gset_cap-1, i=(fp*0x9E3779B1u)&m;
    for(int probes=0;probes<64;++probes){
        int32_t v=sh->gset[i];
        if(v==-1) return 0;
        if(sh->gfp[v]==fp) return 1;
        i=(i+1)&m;
    }
    return 0;
}
// reclaim one slot under S3-FIFO rules; the freed slot's key is unmapped.
static uint32_t s3_reclaim(mc_cache *c_pin, shard_t *sh){
    uint32_t spins=0, budget=2*sh->nslot+8;
    for(;;){
        int force = ++spins>budget;        // all pinned/hot: evict regardless
        uint32_t small_len=ring_len(sh->fs_head,sh->fs_tail,sh->fs_cap);
        if(small_len >= sh->nslot/10+1){
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            if(!force && (sh->slot_ref[s]>0 || slot_pinned(c_pin,sh,s))){   // promote to main
                sh->slot_ref[s]=0; sh->slot_inmain[s]=1;
                sh->fm[sh->fm_tail]=s; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap;
                continue;
            }
            ghost_put(sh,s3_fp(sh->slot_key[s]));
            map_delete(sh,sh->slot_key[s]); sh->evictions++;
            return s;
        }
        if(ring_len(sh->fm_head,sh->fm_tail,sh->fm_cap)==0){   // degenerate: force small
            uint32_t s=sh->fs[sh->fs_head]; sh->fs_head=(sh->fs_head+1)%sh->fs_cap;
            if(sh->slot_key[s]==EMPTY_KEY) return s;        // invalidated: reuse
            ghost_put(sh,s3_fp(sh->slot_key[s]));
            map_delete(sh,sh->slot_key[s]); sh->evictions++;
            return s;
        }
        uint32_t s=sh->fm[sh->fm_head]; sh->fm_head=(sh->fm_head+1)%sh->fm_cap;
        if(sh->slot_key[s]==EMPTY_KEY) return s;            // invalidated: reuse
        if(!force && (sh->slot_ref[s]>0 || slot_pinned(c_pin,sh,s))){   // pinned: keep
            if(sh->slot_ref[s]>0) sh->slot_ref[s]--;
            sh->fm[sh->fm_tail]=s; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap;
            continue;
        }
        map_delete(sh,sh->slot_key[s]); sh->evictions++;
        return s;
    }
}
// allocate a slot for `key` under the active policy and enqueue it.
static uint32_t cache_alloc_slot(mc_cache *c, shard_t *sh, uint64_t key){
    uint32_t slot;
    if(c->policy==MC_CACHE_CLOCK){
        slot=reclaim_slot(sh);
        return slot;
    }
    if(sh->used<sh->nslot){
        for(uint32_t k=0;k<sh->nslot;++k){
            uint32_t i=(sh->hand+k)%sh->nslot;
            if(sh->slot_key[i]==EMPTY_KEY){ sh->hand=(i+1)%sh->nslot; sh->used++; slot=i; goto have; }
        }
    }
    slot=s3_reclaim(c,sh);
have:;
    int to_main = ghost_has(sh,s3_fp(key));
    sh->slot_inmain[slot]=(uint8_t)to_main;
    sh->slot_ref[slot]=0;
    if(to_main){ sh->fm[sh->fm_tail]=slot; sh->fm_tail=(sh->fm_tail+1)%sh->fm_cap; }
    else       { sh->fs[sh->fs_tail]=slot; sh->fs_tail=(sh->fs_tail+1)%sh->fs_cap; }
    return slot;
}
static inline void cache_touch(mc_cache *c, shard_t *sh, uint32_t slot){
    if(c->policy==MC_CACHE_CLOCK) sh->slot_ref[slot]=1;
    else if(sh->slot_ref[slot]<FREQ_MAX) sh->slot_ref[slot]++;
    sh->slot_epoch[slot]=atomic_load_explicit(&c->epoch,memory_order_relaxed);
}
static inline int slot_pinned(mc_cache *c, shard_t *sh, uint32_t slot){
    if(!atomic_load_explicit(&c->phase_mode,memory_order_relaxed)) return 0;
    return sh->slot_epoch[slot]==atomic_load_explicit(&c->epoch,memory_order_relaxed);
}


static inline shard_t *shard_of(mc_cache *c, uint64_t key){
    return &c->sh[(khash(key)>>56)&(NSHARD-1)];
}

// LOCK-FREE. FROZEN (render): bare probe; miss -> record + return NULL (caller
// falls to coarser LOD). UNFROZEN (THAW / single-owner CLI): probe; miss ->
// decode + insert. No lock: every mutation is single-owner by contract (THAW
// partitions by shard; CLI is single-threaded). The cache arena is an immutable
// snapshot during a frozen frame, so the probe needs no synchronization.
const mc_u8 *mc_cache_get(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        if(!cache_frozen(c)){ cache_touch(c,sh,sh->map_slot[mi]); sh->hits++; }
        return sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES;
    }
    if(cache_frozen(c)){ miss_record(c,key); return NULL; }
    sh->misses++;
    uint32_t slot=cache_alloc_slot(c,sh,key);
    sh->slot_key[slot]=key;
    map_insert(sh,key,slot);
    c->src(c->src_ud,lod,bz,by,bx,sh->arena+(size_t)slot*BLK_BYTES);   // decode in place
    cache_touch(c,sh,slot);
    return sh->arena+(size_t)slot*BLK_BYTES;
}

void mc_cache_get_copy(mc_cache *c, int lod, int bz, int by, int bx, mc_u8 *dst){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        if(!cache_frozen(c)){ cache_touch(c,sh,sh->map_slot[mi]); sh->hits++; }
        memcpy(dst,sh->arena+(size_t)sh->map_slot[mi]*BLK_BYTES,BLK_BYTES);
        return;
    }
    if(cache_frozen(c)){
        miss_record(c,key);
        c->src(c->src_ud,lod,bz,by,bx,dst);   // read-through, no insert
        return;
    }
    sh->misses++;
    uint32_t slot=cache_alloc_slot(c,sh,key);
    sh->slot_key[slot]=key;
    map_insert(sh,key,slot);
    c->src(c->src_ud,lod,bz,by,bx,sh->arena+(size_t)slot*BLK_BYTES);
    memcpy(dst,sh->arena+(size_t)slot*BLK_BYTES,BLK_BYTES);
    cache_touch(c,sh,slot);
}

int mc_cache_contains(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    return map_find(sh,key)!=UINT32_MAX;   // lock-free probe
}

// lookup-or-decode-insert; returns 1 if a decode happened. LOCK-FREE: the caller
// guarantees single-owner access to this block's shard (THAW partitions fill work
// by shard so no two workers touch one shard). All shard mutation -- map, arena,
// eviction rings -- happens here, only during THAW, only from the shard's owner.
static int cache_fill_one(mc_cache *c, int lod, int bz, int by, int bx){
    uint64_t key=bkey(lod,bz,by,bx);
    shard_t *sh=shard_of(c,key);
    uint32_t mi=map_find(sh,key);
    if(mi!=UINT32_MAX){
        cache_touch(c,sh,sh->map_slot[mi]); sh->hits++;
        return 0;
    }
    sh->misses++;
    uint32_t slot=cache_alloc_slot(c,sh,key);
    sh->slot_key[slot]=key;
    map_insert(sh,key,slot);
    c->src(c->src_ud,lod,bz,by,bx,sh->arena+(size_t)slot*BLK_BYTES);   // decode in place
    cache_touch(c,sh,slot);
    return 1;
}

// THAW batch fill, partitioned by SHARD ownership. Each block is bucketed by its
// shard; worker t owns shards { t, t+nt, t+2nt, ... }. Because a shard is touched
// by exactly one worker, every shard mutation (map insert, slot alloc, eviction,
// arena write) is single-owner and needs NO lock. This is the partitioned phase
// update: parallelism by disjoint ownership, not by shared queue. The only sync
// is the join at the end (the phase barrier). Caller must be UNFROZEN (THAW).
typedef struct {
    mc_cache *c;
    const mc_block_id *ids;     // all blocks (unsorted)
    const uint32_t *bucket;     // bucket[s] = head index into `link` for shard s (-1 end)
    const uint32_t *link;       // link[i] = next block index in the same shard (-1 end)
    int nt, t;                  // this worker owns shards s where s % nt == t
    size_t decoded;             // written by this worker only
} upd_ctx;
static void *upd_worker(void *p){
    upd_ctx *u=p;
    size_t dec=0;
    for(int s=u->t; s<NSHARD; s+=u->nt){            // this worker's disjoint shards
        for(uint32_t i=u->bucket[s]; i!=UINT32_MAX; i=u->link[i]){
            const mc_block_id *b=&u->ids[i];
            dec+=(size_t)cache_fill_one(u->c,b->lod,b->bz,b->by,b->bx);
        }
    }
    u->decoded=dec;
    return NULL;
}
size_t mc_cache_update(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return 0;
    // Bucket block indices by shard via per-shard singly-linked lists (no sort,
    // no shared structure). bucket[s] heads shard s's chain through link[].
    uint32_t *bucket=malloc(NSHARD*sizeof *bucket);
    uint32_t *link=malloc(n*sizeof *link);
    if(!bucket||!link){ free(bucket); free(link); return 0; }
    for(int s=0;s<NSHARD;++s) bucket[s]=UINT32_MAX;
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(ids[i].lod,ids[i].bz,ids[i].by,ids[i].bx);
        int s=(int)((khash(key)>>56)&(NSHARD-1));
        link[i]=bucket[s]; bucket[s]=(uint32_t)i;
    }
    int nt=nthreads;
    if(nt<=0){ long nc=sysconf(_SC_NPROCESSORS_ONLN); nt=(int)(nc>0?nc:4); }
    if(nt>16)nt=16; if(nt>NSHARD)nt=NSHARD; if(nt<1)nt=1;
    upd_ctx u[16];
    for(int t=0;t<nt;++t) u[t]=(upd_ctx){.c=c,.ids=ids,.bucket=bucket,.link=link,.nt=nt,.t=t,.decoded=0};
    if(nt<=1){ upd_worker(&u[0]); }
    else {
        pthread_t th[16];
        for(int t=0;t<nt;++t) pthread_create(&th[t],NULL,upd_worker,&u[t]);
        for(int t=0;t<nt;++t) pthread_join(th[t],NULL);   // phase barrier (join)
    }
    size_t dec=0; for(int t=0;t<nt;++t) dec+=u[t].decoded;
    free(bucket); free(link);
    return dec;
}

void mc_cache_freeze(mc_cache *c){
    if(!c) return;
    atomic_store_explicit(&c->phase_mode,1,memory_order_relaxed);
    atomic_store_explicit(&c->frozen,1,memory_order_release);
}
void mc_cache_thaw(mc_cache *c){
    if(!c) return;
    atomic_store_explicit(&c->frozen,0,memory_order_release);
    atomic_fetch_add_explicit(&c->epoch,1,memory_order_relaxed);
}
// Record a block into the dedup miss set from outside the frozen read path
// (e.g. mc_volume_try_block marking an ABSENT block for the downloader). Same
// lock-free dedup insert; safe to call concurrently with frozen reads.
void mc_cache_miss_mark(mc_cache *c, int lod, int bz, int by, int bx){
    if(c) miss_record(c, bkey(lod,bz,by,bx));
}

// Drain the dedup miss set: emit each unique block once and clear its slot. Must
// run while no frozen reads are inserting (thaw window) — consistent with the
// game loop (thaw drains, then freeze reopens reads).
size_t mc_cache_misses_drain(mc_cache *c, mc_block_id *out, size_t cap){
    if(!c||!out) return 0;
    if(atomic_load_explicit(&c->miss_n,memory_order_relaxed)==0) return 0;
    size_t m=0;
    for(uint32_t i=0;i<MISSQ_CAP;++i){
        uint64_t k=atomic_load_explicit(&c->missq[i],memory_order_relaxed);
        if(!k) continue;
        atomic_store_explicit(&c->missq[i],0,memory_order_relaxed);   // clear slot
        if(m<cap){
            out[m].lod=(int)((k>>60)&7);
            out[m].bz=(int)((k>>40)&0xFFFFF);
            out[m].by=(int)((k>>20)&0xFFFFF);
            out[m].bx=(int)(k&0xFFFFF);
            m++;
        }
    }
    atomic_store_explicit(&c->miss_n,0,memory_order_relaxed);
    return m;
}

int mc_cache_best_lod(mc_cache *c, int finest_lod, int bz, int by, int bx){
    for(int l=finest_lod;l<8;++l){
        uint64_t key=bkey(l,bz,by,bx);
        shard_t *sh=shard_of(c,key);
        if(map_find(sh,key)!=UINT32_MAX) return l;   // lock-free probe
        bz>>=1; by>>=1; bx>>=1;
    }
    return -1;
}

// ---- async update tickets ---------------------------------------------------
// Async ticket: same shard-partitioned, lock-free fill as mc_cache_update, but on
// detached worker threads. Worker t owns shards { t, t+nth, ... } via per-shard
// linked buckets -> single-owner, no lock. Cancel sets a flag workers poll.
struct mc_cache_ticket {
    mc_cache *c;
    mc_block_id *ids;            // owned copy
    uint32_t *bucket;           // bucket[s] = head index for shard s (UINT32_MAX end)
    uint32_t *link;             // link[i]   = next block index in same shard
    _Atomic uint32_t workers_done;
    _Atomic int cancel;
    pthread_t th[16]; int nth; int t_id[16];
    int joined;
};
static void *aupd_worker(void *p){
    mc_cache_ticket *t = ((void**)p)[0];
    int me = (int)(intptr_t)((void**)p)[1];
    free(p);
    for(int s=me; s<NSHARD; s+=t->nth){
        if(atomic_load_explicit(&t->cancel,memory_order_relaxed)) break;
        for(uint32_t i=t->bucket[s]; i!=UINT32_MAX; i=t->link[i]){
            const mc_block_id *b=&t->ids[i];
            cache_fill_one(t->c,b->lod,b->bz,b->by,b->bx);
        }
    }
    atomic_fetch_add_explicit(&t->workers_done,1,memory_order_release);
    return NULL;
}
mc_cache_ticket *mc_cache_update_async(mc_cache *c, const mc_block_id *ids, size_t n, int nthreads){
    if(!c||!ids||!n||cache_frozen(c)) return NULL;
    mc_cache_ticket *t=calloc(1,sizeof *t);
    t->c=c;
    t->ids=malloc(n*sizeof *t->ids);
    memcpy(t->ids,ids,n*sizeof *t->ids);
    t->bucket=malloc(NSHARD*sizeof *t->bucket);
    t->link=malloc(n*sizeof *t->link);
    for(int s=0;s<NSHARD;++s) t->bucket[s]=UINT32_MAX;
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(t->ids[i].lod,t->ids[i].bz,t->ids[i].by,t->ids[i].bx);
        int s=(int)((khash(key)>>56)&(NSHARD-1));
        t->link[i]=t->bucket[s]; t->bucket[s]=(uint32_t)i;
    }
    atomic_store(&t->workers_done,0); atomic_store(&t->cancel,0);
    int nt=nthreads;
    if(nt<=0){ long nc=sysconf(_SC_NPROCESSORS_ONLN); nt=(int)(nc>0?nc:4); }
    if(nt>16)nt=16; if(nt>NSHARD)nt=NSHARD; if(nt<1)nt=1;
    t->nth=nt;
    for(int i=0;i<nt;++i){
        void **arg=malloc(2*sizeof(void*)); arg[0]=t; arg[1]=(void*)(intptr_t)i;
        pthread_create(&t->th[i],NULL,aupd_worker,arg);
    }
    return t;
}
int mc_cache_ticket_done(mc_cache_ticket *t){
    if(!t) return 1;
    return atomic_load_explicit(&t->workers_done,memory_order_acquire)>=(uint32_t)t->nth;
}
void mc_cache_ticket_cancel(mc_cache_ticket *t){
    if(t) atomic_store_explicit(&t->cancel,1,memory_order_release);
}
static void ticket_join(mc_cache_ticket *t){
    if(t->joined) return;
    for(int i=0;i<t->nth;++i) pthread_join(t->th[i],NULL);
    t->joined=1;
}
void mc_cache_ticket_wait(mc_cache_ticket *t){ if(t) ticket_join(t); }
void mc_cache_ticket_free(mc_cache_ticket *t){
    if(!t) return;
    ticket_join(t);
    free(t->ids); free(t->bucket); free(t->link); free(t);
}

// resolve: ensure resident (parallel via mc_cache_update), then fill the pointer
// table; cache_touch stamps the current epoch so these slots are pinned against
// eviction until the next thaw(). Lock-free: single-owner (THAW / CLI) contract.
size_t mc_cache_resolve(mc_cache *c, const mc_block_id *ids, size_t n,
                        const mc_u8 **ptrs, int nthreads){
    if(!c||!ids||!n||!ptrs||cache_frozen(c)) return 0;
    size_t dec=mc_cache_update(c,ids,n,nthreads);
    for(size_t i=0;i<n;++i){
        uint64_t key=bkey(ids[i].lod,ids[i].bz,ids[i].by,ids[i].bx);
        shard_t *sh=shard_of(c,key);
        uint32_t mi=map_find(sh,key);
        if(mi!=UINT32_MAX){
            uint32_t slot=sh->map_slot[mi];
            cache_touch(c,sh,slot);
            ptrs[i]=sh->arena+(size_t)slot*BLK_BYTES;
        } else ptrs[i]=NULL;   // evicted by same-batch pressure (set > capacity)
    }
    return dec;
}

void mc_cache_prefetch_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int gz=cz*16+bz, gy=cy*16+by, gx=cx*16+bx;
        uint64_t key=bkey(lod,gz,gy,gx);
        if(map_find(shard_of(c,key),key)==UINT32_MAX) (void)mc_cache_get(c,lod,gz,gy,gx);
    }
}

// remove one key if present. Single-owner (UNFROZEN/THAW) — no lock.
static void shard_remove_key(shard_t *sh, uint64_t key){
    uint32_t mi=map_find(sh,key);
    if(mi==UINT32_MAX) return;
    uint32_t slot=sh->map_slot[mi];
    map_delete(sh,key);
    sh->slot_key[slot]=EMPTY_KEY;       // slot stays in its FIFO ring; the
    sh->slot_ref[slot]=0;               // reclaim path skips empty keys
    sh->evictions++;
}
void mc_cache_invalidate_chunk(mc_cache *c, int lod, int cz, int cy, int cx){
    if(cache_frozen(c)) return;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        uint64_t key=bkey(lod,cz*16+bz,cy*16+by,cx*16+bx);
        shard_remove_key(shard_of(c,key),key);
    }
}

void mc_cache_clear(mc_cache *c){
    for(int s=0;s<NSHARD;++s){            // single-owner: call only while no fill runs
        shard_t *sh=&c->sh[s];
        memset(sh->map_key,0,(size_t)sh->map_cap*8);
        memset(sh->slot_key,0,(size_t)sh->nslot*8);
        memset(sh->slot_ref,0,sh->nslot);
        sh->used=0; sh->hand=0;
        sh->fs_head=sh->fs_tail=sh->fm_head=sh->fm_tail=0;
        sh->g_head=0; memset(sh->gfp,0,4u*sh->g_cap);
        memset(sh->gset,0xFF,4u*sh->gset_cap);
        memset(sh->slot_inmain,0,sh->nslot);
    }
}

void mc_cache_get_stats(mc_cache *c, mc_cache_stats *out){
    memset(out,0,sizeof *out);
    for(int s=0;s<NSHARD;++s){
        shard_t *sh=&c->sh[s];
        out->hits+=sh->hits; out->misses+=sh->misses; out->evictions+=sh->evictions;
        out->slots+=sh->nslot; out->used+=sh->used;
    }
}

// ---- bindings ----
static void src_archive(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    struct mc_archive *a=ud;
    // The tree walk (mc_resolve_chunk) is identical for all 4096 blocks of a chunk;
    // memoize the last (lod,cz,cy,cx)->chunk_off thread-locally. Render samples are
    // chunk-coherent so this collapses the per-block walk to one walk per chunk.
    int cz=bz>>4, cy=by>>4, cx=bx>>4;
    static _Thread_local const struct mc_archive *la=NULL;
    static _Thread_local int llod=-1,lcz=-1,lcy=-1,lcx=-1; static _Thread_local uint64_t lco=0,lgen=0;
    uint64_t gen=atomic_load_explicit(&a->gen,memory_order_acquire);
    uint64_t co;
    if(la==a && lgen==gen && llod==lod && lcz==cz && lcy==cy && lcx==cx) co=lco;
    else { co=mc_archive_chunk_offset(a,lod,cz,cy,cx);   // re-resolve if archive grew
           la=a; lgen=gen; llod=lod; lcz=cz; lcy=cy; lcx=cx; lco=co; }
    mc_archive_decode_block(a,co,bz&15,by&15,bx&15,dst);
}
mc_cache *mc_cache_new_archive(size_t bytes, struct mc_archive *a){
    mc_cache *c=mc_cache_new(bytes,src_archive,a);
    if(c) c->ar=a;
    return c;
}
typedef struct { mc_cache *c; } rdwrap_t;
static void src_reader(void *ud, int lod, int bz,int by,int bx, mc_u8 *dst){
    mc_cache *c=ud;                                 // single-owner (THAW/CLI): no lock
    uint64_t co=mc_chunk_offset(c->rd,lod,bz>>4,by>>4,bx>>4);
    mc_decode_block(c->rd,co,bz&15,by&15,bx&15,dst);
}
mc_cache *mc_cache_new_reader(size_t bytes, struct mc_reader *r){
    mc_cache *c=mc_cache_new(bytes,NULL,NULL);
    if(!c) return NULL;
    c->rd=r; c->src=src_reader; c->src_ud=c;
    return c;
}

// ============================================================================
// mc_sample — point sampling over blocked u8 volumes (see header)
// ============================================================================
#include <unistd.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE4_1__)
#include <immintrin.h>
#endif

#define MC_S_MEMO 256   // covers an oblique 1024-px row's block working set

// Pointer-only memo entry (16B, was 4112B). The block() source returns a STABLE
// pointer the memo caches directly: the cache/arena source (mc_volume interactive
// path) returns an arena/zero pointer valid for the frozen frame; dense bypasses
// mc_s_block entirely. The only source that synthesizes into a scratch buffer is
// the CLI blocking path -- for THAT, owns_ptr is 0 and we use a per-sampler buf[]
// (allocated only then) so cached pointers don't alias one shared scratch.
typedef struct {
    int bz, by, bx;             // -1 = empty
    const uint8_t *ptr;         // NULL = known-absent (sampled as 0)
} mc_s_memo;

struct mc_sampler {
    mc_sample_src src;
    int nbz, nby, nbx;
    int lbz, lby, lbx;          // last block touched (ray-coherence cache)
    const uint8_t *lptr;
    int rbz, rby, rbx, rres;    // last residency probe (ray-coherence cache)
    uint8_t (*scratch)[4096];   // per-entry synth buffers, only if !owns_ptr (else NULL)
    mc_s_memo m[MC_S_MEMO];
};

static inline const uint8_t *mc_s_block(mc_sampler *s, int bz, int by, int bx) {
    if (bz == s->lbz && by == s->lby && bx == s->lbx) return s->lptr;
    unsigned h = ((unsigned)bz * 73856093u) ^ ((unsigned)by * 19349663u) ^
                 ((unsigned)bx * 83492791u);
    unsigned slot = h & (MC_S_MEMO - 1);
    mc_s_memo *e = &s->m[slot];
    if (!(e->bz == bz && e->by == by && e->bx == bx)) {
        e->bz = bz; e->by = by; e->bx = bx;
        // owns_ptr sources return a stable pointer (cache arena); cache it directly.
        // Otherwise synthesize into this slot's own scratch buffer.
        uint8_t *tmp = s->scratch ? s->scratch[slot] : NULL;
        e->ptr = s->src.block(&s->src, bz, by, bx, tmp);
    }
    s->lbz = bz; s->lby = by; s->lbx = bx; s->lptr = e->ptr;
    return e->ptr;
}

static inline float mc_s_voxel(mc_sampler *s, int z, int y, int x) {
    if ((unsigned)z >= (unsigned)s->src.nz ||
        (unsigned)y >= (unsigned)s->src.ny ||
        (unsigned)x >= (unsigned)s->src.nx) return 0.0f;
    if (s->src.dense)
        return (float)s->src.dense[(size_t)z * s->src.dsy +
                                   (size_t)y * s->src.dsx + (size_t)x];
    const uint8_t *b = mc_s_block(s, z >> 4, y >> 4, x >> 4);
    return b ? (float)b[((z & 15) << 8) | ((y & 15) << 4) | (x & 15)] : 0.0f;
}

static inline float mc_s_nearest(mc_sampler *s, float z, float y, float x) {
    return mc_s_voxel(s, (int)floorf(z + 0.5f), (int)floorf(y + 0.5f),
                      (int)floorf(x + 0.5f));
}

// Interior cell: all 8 corners inside ONE block; caller verified bounds and
// no-straddle. Ints + fractions precomputed (no refloor / rebounds).
static inline float mc_s_cell_in(mc_sampler *s, int z0, int y0, int x0,
                                 float dz, float dy, float dx) {
    const uint8_t *b = mc_s_block(s, z0 >> 4, y0 >> 4, x0 >> 4);
    if (!b) return 0.0f;
    const uint8_t *p = b + (((z0 & 15) << 8) | ((y0 & 15) << 4) | (x0 & 15));
    float c00 = (float)p[0]   + ((float)p[1]   - (float)p[0])   * dx;
    float c01 = (float)p[16]  + ((float)p[17]  - (float)p[16])  * dx;
    float c10 = (float)p[256] + ((float)p[257] - (float)p[256]) * dx;
    float c11 = (float)p[272] + ((float)p[273] - (float)p[272]) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

// Straddling cell: crosses >=1 block face (caller verified bounds). Fetch each
// DISTINCT block once -- 2 for a face straddle (the common case), 4 edge, 8
// corner -- instead of 8 per-corner lookups.
static inline float mc_s_cell_straddle(mc_sampler *s, int z0, int y0, int x0,
                                       float dz, float dy, float dx) {
    const int sz = (z0 & 15) == 15, sy = (y0 & 15) == 15, sx = (x0 & 15) == 15;
    const int bz = z0 >> 4, by = y0 >> 4, bx = x0 >> 4;
    // Single-axis x straddle: the dominant case on scanlines (consecutive lanes
    // walk x). Two block fetches, column x=15 of B and column x=0 of B+1 --
    // no 8-pointer dedup table.
    if (sx && !sy && !sz) {
        const uint8_t *b0 = mc_s_block(s, bz, by, bx);
        const uint8_t *b1 = mc_s_block(s, bz, by, bx + 1);
        const int o00 = ((z0 & 15) << 8) | ((y0 & 15) << 4);
        const int o01 = o00 + 16, o10 = o00 + 256, o11 = o00 + 272;
        float c000 = b0 ? (float)b0[o00 | 15] : 0.0f;
        float c001 = b1 ? (float)b1[o00]      : 0.0f;
        float c010 = b0 ? (float)b0[o01 | 15] : 0.0f;
        float c011 = b1 ? (float)b1[o01]      : 0.0f;
        float c100 = b0 ? (float)b0[o10 | 15] : 0.0f;
        float c101 = b1 ? (float)b1[o10]      : 0.0f;
        float c110 = b0 ? (float)b0[o11 | 15] : 0.0f;
        float c111 = b1 ? (float)b1[o11]      : 0.0f;
        float c00 = c000 + (c001 - c000) * dx;
        float c01 = c010 + (c011 - c010) * dx;
        float c10 = c100 + (c101 - c100) * dx;
        float c11 = c110 + (c111 - c110) * dx;
        float c0 = c00 + (c01 - c00) * dy;
        float c1 = c10 + (c11 - c10) * dy;
        return c0 + (c1 - c0) * dz;
    }
    // B[a][b][c] = block holding corner (z0+a, y0+b, x0+c); reuse pointers
    // along non-straddling axes so each distinct block is fetched once.
    const uint8_t *B[2][2][2];
    B[0][0][0] = mc_s_block(s, bz, by, bx);
    B[0][0][1] = sx ? mc_s_block(s, bz, by, bx + 1) : B[0][0][0];
    B[0][1][0] = sy ? mc_s_block(s, bz, by + 1, bx) : B[0][0][0];
    B[0][1][1] = sy ? (sx ? mc_s_block(s, bz, by + 1, bx + 1) : B[0][1][0])
                    : B[0][0][1];
    if (sz) {
        B[1][0][0] = mc_s_block(s, bz + 1, by, bx);
        B[1][0][1] = sx ? mc_s_block(s, bz + 1, by, bx + 1) : B[1][0][0];
        B[1][1][0] = sy ? mc_s_block(s, bz + 1, by + 1, bx) : B[1][0][0];
        B[1][1][1] = sy ? (sx ? mc_s_block(s, bz + 1, by + 1, bx + 1) : B[1][1][0])
                        : B[1][0][1];
    } else {
        B[1][0][0] = B[0][0][0]; B[1][0][1] = B[0][0][1];
        B[1][1][0] = B[0][1][0]; B[1][1][1] = B[0][1][1];
    }
    const int oz0 = (z0 & 15) << 8, oz1 = ((z0 + 1) & 15) << 8;
    const int oy0 = (y0 & 15) << 4, oy1 = ((y0 + 1) & 15) << 4;
    const int ox0 = x0 & 15,        ox1 = (x0 + 1) & 15;
    #define MC_S_C(a, b, c, oz, oy, ox) \
        (B[a][b][c] ? (float)B[a][b][c][(oz) | (oy) | (ox)] : 0.0f)
    float c000 = MC_S_C(0, 0, 0, oz0, oy0, ox0);
    float c001 = MC_S_C(0, 0, 1, oz0, oy0, ox1);
    float c010 = MC_S_C(0, 1, 0, oz0, oy1, ox0);
    float c011 = MC_S_C(0, 1, 1, oz0, oy1, ox1);
    float c100 = MC_S_C(1, 0, 0, oz1, oy0, ox0);
    float c101 = MC_S_C(1, 0, 1, oz1, oy0, ox1);
    float c110 = MC_S_C(1, 1, 0, oz1, oy1, ox0);
    float c111 = MC_S_C(1, 1, 1, oz1, oy1, ox1);
    #undef MC_S_C
    float c00 = c000 + (c001 - c000) * dx;
    float c01 = c010 + (c011 - c010) * dx;
    float c10 = c100 + (c101 - c100) * dx;
    float c11 = c110 + (c111 - c110) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

// In-bounds blocked cell, ints + fracs precomputed: dispatch straddle/interior.
static inline float mc_s_cell(mc_sampler *s, int z0, int y0, int x0,
                              float dz, float dy, float dx) {
    return ((z0 & 15) == 15 || (y0 & 15) == 15 || (x0 & 15) == 15)
        ? mc_s_cell_straddle(s, z0, y0, x0, dz, dy, dx)
        : mc_s_cell_in(s, z0, y0, x0, dz, dy, dx);
}

static inline float mc_s_trilinear(mc_sampler *s, float z, float y, float x) {
    float zf = floorf(z), yf = floorf(y), xf = floorf(x);
    int z0 = (int)zf, y0 = (int)yf, x0 = (int)xf;
    float dz = z - zf, dy = y - yf, dx = x - xf;
    // dense fast path: direct strided gather, only a bounds check
    if (s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1)) {
        const size_t sy = s->src.dsy, sx = s->src.dsx;
        const uint8_t *p = s->src.dense + (size_t)z0 * sy + (size_t)y0 * sx + x0;
        float c00 = (float)p[0]      + ((float)p[1]        - (float)p[0])      * dx;
        float c01 = (float)p[sx]     + ((float)p[sx + 1]   - (float)p[sx])     * dx;
        float c10 = (float)p[sy]     + ((float)p[sy + 1]   - (float)p[sy])     * dx;
        float c11 = (float)p[sy + sx] + ((float)p[sy + sx + 1] - (float)p[sy + sx]) * dx;
        float c0 = c00 + (c01 - c00) * dy;
        float c1 = c10 + (c11 - c10) * dy;
        return c0 + (c1 - c0) * dz;
    }
    // blocked in-bounds cell: interior (one block) or straddle (dedup'd blocks)
    if (!s->src.dense &&
        (unsigned)z0 < (unsigned)(s->src.nz - 1) &&
        (unsigned)y0 < (unsigned)(s->src.ny - 1) &&
        (unsigned)x0 < (unsigned)(s->src.nx - 1))
        return mc_s_cell(s, z0, y0, x0, dz, dy, dx);
    // slow path (volume edge / dense edge): block/bounds handled per corner
    float c000 = mc_s_voxel(s, z0, y0, x0);
    float c001 = mc_s_voxel(s, z0, y0, x0 + 1);
    float c010 = mc_s_voxel(s, z0, y0 + 1, x0);
    float c011 = mc_s_voxel(s, z0, y0 + 1, x0 + 1);
    float c100 = mc_s_voxel(s, z0 + 1, y0, x0);
    float c101 = mc_s_voxel(s, z0 + 1, y0, x0 + 1);
    float c110 = mc_s_voxel(s, z0 + 1, y0 + 1, x0);
    float c111 = mc_s_voxel(s, z0 + 1, y0 + 1, x0 + 1);
    float c00 = c000 + (c001 - c000) * dx;
    float c01 = c010 + (c011 - c010) * dx;
    float c10 = c100 + (c101 - c100) * dx;
    float c11 = c110 + (c111 - c110) * dx;
    float c0 = c00 + (c01 - c00) * dy;
    float c1 = c10 + (c11 - c10) * dy;
    return c0 + (c1 - c0) * dz;
}

// NaN test that survives -ffast-math (where compilers delete x!=x and isnan):
// sign-cleared bits above the +inf pattern <=> exponent all-ones, mantissa != 0.
static inline int mc_s_isnan(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

static inline float mc_s_sample(mc_sampler *s, float z, float y, float x,
                                mc_filter f) {
    if (mc_s_isnan(z) || mc_s_isnan(y) || mc_s_isnan(x)) return 0.0f;
    return f == MC_FILTER_NEAREST ? mc_s_nearest(s, z, y, x)
                                  : mc_s_trilinear(s, z, y, x);
}

// All-zero block for masked SIMD lanes: a straddling lane points here during
// the group gather (its lerp result is overwritten scalar after), and an
// absent-block lane's lerp over zeros IS its correct value (samples as 0).
__attribute__((unused)) static const uint8_t mc_s_zero4k[4096];   // unused in no-SIMD builds

// ---------------------------------------------------------------------------
// 4-wide trilinear (ray-step batching for the compositors)
// ---------------------------------------------------------------------------
// Sample 4 positions at once. Lanes that qualify for a fast path are
// gathered and lerped with NEON; anything else (edges, absent blocks,
// non-aarch64) falls back to the scalar sampler per lane. Uses separate
// mul+add (no fma), so every lane is bit-identical to mc_s_trilinear.
#if defined(__aarch64__)
static inline float32x4_t mc_s_lerp8x4(const uint8_t *p0, const uint8_t *p1,
                                       const uint8_t *p2, const uint8_t *p3,
                                       size_t sy, size_t sx,
                                       float32x4_t dz, float32x4_t dy,
                                       float32x4_t dx) {
    uint16x4_t g00 = vdup_n_u16(0), g01 = vdup_n_u16(0);
    uint16x4_t g10 = vdup_n_u16(0), g11 = vdup_n_u16(0);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p0, g00, 0);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p1, g00, 1);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p2, g00, 2);
    g00 = vld1_lane_u16((const uint16_t *)(const void *)p3, g00, 3);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sx), g01, 0);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sx), g01, 1);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sx), g01, 2);
    g01 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sx), g01, 3);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sy), g10, 0);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sy), g10, 1);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sy), g10, 2);
    g10 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sy), g10, 3);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p0 + sy + sx), g11, 0);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p1 + sy + sx), g11, 1);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p2 + sy + sx), g11, 2);
    g11 = vld1_lane_u16((const uint16_t *)(const void *)(p3 + sy + sx), g11, 3);
    uint16x8_t w00 = vmovl_u8(vreinterpret_u8_u16(g00));
    uint16x8_t w01 = vmovl_u8(vreinterpret_u8_u16(g01));
    uint16x8_t w10 = vmovl_u8(vreinterpret_u8_u16(g10));
    uint16x8_t w11 = vmovl_u8(vreinterpret_u8_u16(g11));
#define MC_S_F32E(w) vcvtq_f32_u32(vmovl_u16(vuzp1_u16(vget_low_u16(w), vget_high_u16(w))))
#define MC_S_F32O(w) vcvtq_f32_u32(vmovl_u16(vuzp2_u16(vget_low_u16(w), vget_high_u16(w))))
    float32x4_t f000 = MC_S_F32E(w00), f001 = MC_S_F32O(w00);
    float32x4_t f010 = MC_S_F32E(w01), f011 = MC_S_F32O(w01);
    float32x4_t f100 = MC_S_F32E(w10), f101 = MC_S_F32O(w10);
    float32x4_t f110 = MC_S_F32E(w11), f111 = MC_S_F32O(w11);
#undef MC_S_F32E
#undef MC_S_F32O
    float32x4_t c00 = vaddq_f32(f000, vmulq_f32(vsubq_f32(f001, f000), dx));
    float32x4_t c01 = vaddq_f32(f010, vmulq_f32(vsubq_f32(f011, f010), dx));
    float32x4_t c10 = vaddq_f32(f100, vmulq_f32(vsubq_f32(f101, f100), dx));
    float32x4_t c11 = vaddq_f32(f110, vmulq_f32(vsubq_f32(f111, f110), dx));
    float32x4_t c0 = vaddq_f32(c00, vmulq_f32(vsubq_f32(c01, c00), dy));
    float32x4_t c1 = vaddq_f32(c10, vmulq_f32(vsubq_f32(c11, c10), dy));
    return vaddq_f32(c0, vmulq_f32(vsubq_f32(c1, c0), dz));
}
#elif defined(__SSE4_1__)
static inline uint16_t mc_s_ld16(const uint8_t *p) {
    uint16_t v; __builtin_memcpy(&v, p, 2); return v;
}
static inline __m128 mc_s_lerp8x4(const uint8_t *p0, const uint8_t *p1,
                                  const uint8_t *p2, const uint8_t *p3,
                                  size_t sy, size_t sx,
                                  __m128 dz, __m128 dy, __m128 dx) {
    // per corner-row: 4 samples' (c0,c1) byte pairs in u16 lanes 0..3
    __m128i z = _mm_setzero_si128();
    __m128i g00 = _mm_insert_epi16(z, mc_s_ld16(p0), 0);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p1), 1);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p2), 2);
    g00 = _mm_insert_epi16(g00, mc_s_ld16(p3), 3);
    __m128i g01 = _mm_insert_epi16(z, mc_s_ld16(p0 + sx), 0);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p1 + sx), 1);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p2 + sx), 2);
    g01 = _mm_insert_epi16(g01, mc_s_ld16(p3 + sx), 3);
    __m128i g10 = _mm_insert_epi16(z, mc_s_ld16(p0 + sy), 0);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p1 + sy), 1);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p2 + sy), 2);
    g10 = _mm_insert_epi16(g10, mc_s_ld16(p3 + sy), 3);
    __m128i g11 = _mm_insert_epi16(z, mc_s_ld16(p0 + sy + sx), 0);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p1 + sy + sx), 1);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p2 + sy + sx), 2);
    g11 = _mm_insert_epi16(g11, mc_s_ld16(p3 + sy + sx), 3);
    // split even bytes (x0 corner) / odd bytes (x1 corner) -> u32 -> f32
    const __m128i me = _mm_set_epi8(-1, -1, -1, 6, -1, -1, -1, 4,
                                    -1, -1, -1, 2, -1, -1, -1, 0);
    const __m128i mo = _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 5,
                                    -1, -1, -1, 3, -1, -1, -1, 1);
#define MC_S_F32E(g) _mm_cvtepi32_ps(_mm_shuffle_epi8(g, me))
#define MC_S_F32O(g) _mm_cvtepi32_ps(_mm_shuffle_epi8(g, mo))
    __m128 f000 = MC_S_F32E(g00), f001 = MC_S_F32O(g00);
    __m128 f010 = MC_S_F32E(g01), f011 = MC_S_F32O(g01);
    __m128 f100 = MC_S_F32E(g10), f101 = MC_S_F32O(g10);
    __m128 f110 = MC_S_F32E(g11), f111 = MC_S_F32O(g11);
#undef MC_S_F32E
#undef MC_S_F32O
    __m128 c00 = _mm_add_ps(f000, _mm_mul_ps(_mm_sub_ps(f001, f000), dx));
    __m128 c01 = _mm_add_ps(f010, _mm_mul_ps(_mm_sub_ps(f011, f010), dx));
    __m128 c10 = _mm_add_ps(f100, _mm_mul_ps(_mm_sub_ps(f101, f100), dx));
    __m128 c11 = _mm_add_ps(f110, _mm_mul_ps(_mm_sub_ps(f111, f110), dx));
    __m128 c0 = _mm_add_ps(c00, _mm_mul_ps(_mm_sub_ps(c01, c00), dy));
    __m128 c1 = _mm_add_ps(c10, _mm_mul_ps(_mm_sub_ps(c11, c10), dy));
    return _mm_add_ps(c0, _mm_mul_ps(_mm_sub_ps(c1, c0), dz));
}

#if defined(__AVX2__) && !defined(MC_S_NO_TRI8)
// 8-wide variant for the x86-64-v3 fleet (Zen 3/4/5, 12th-gen+ Intel).
#define MC_S_HAVE_TRI8 1
static inline __m256i mc_s_g8(const uint8_t *const p[8], size_t off) {
    __m128i lo = _mm_setzero_si128(), hi = lo;
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[0] + off), 0);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[1] + off), 1);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[2] + off), 2);
    lo = _mm_insert_epi16(lo, mc_s_ld16(p[3] + off), 3);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[4] + off), 0);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[5] + off), 1);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[6] + off), 2);
    hi = _mm_insert_epi16(hi, mc_s_ld16(p[7] + off), 3);
    return _mm256_set_m128i(hi, lo);
}
static inline __m256 mc_s_lerp8x8(const uint8_t *const p[8],
                                  size_t sy, size_t sx,
                                  __m256 dz, __m256 dy, __m256 dx) {
    __m256i g00 = mc_s_g8(p, 0),  g01 = mc_s_g8(p, sx);
    __m256i g10 = mc_s_g8(p, sy), g11 = mc_s_g8(p, sy + sx);
    // even byte of each u16 pair = x0 corner, odd = x1 (per 128-bit half)
    const __m256i me = _mm256_broadcastsi128_si256(
        _mm_set_epi8(-1, -1, -1, 6, -1, -1, -1, 4,
                     -1, -1, -1, 2, -1, -1, -1, 0));
    const __m256i mo = _mm256_broadcastsi128_si256(
        _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 5,
                     -1, -1, -1, 3, -1, -1, -1, 1));
#define MC_S_F32E(g) _mm256_cvtepi32_ps(_mm256_shuffle_epi8(g, me))
#define MC_S_F32O(g) _mm256_cvtepi32_ps(_mm256_shuffle_epi8(g, mo))
    __m256 f000 = MC_S_F32E(g00), f001 = MC_S_F32O(g00);
    __m256 f010 = MC_S_F32E(g01), f011 = MC_S_F32O(g01);
    __m256 f100 = MC_S_F32E(g10), f101 = MC_S_F32O(g10);
    __m256 f110 = MC_S_F32E(g11), f111 = MC_S_F32O(g11);
#undef MC_S_F32E
#undef MC_S_F32O
    __m256 c00 = _mm256_add_ps(f000, _mm256_mul_ps(_mm256_sub_ps(f001, f000), dx));
    __m256 c01 = _mm256_add_ps(f010, _mm256_mul_ps(_mm256_sub_ps(f011, f010), dx));
    __m256 c10 = _mm256_add_ps(f100, _mm256_mul_ps(_mm256_sub_ps(f101, f100), dx));
    __m256 c11 = _mm256_add_ps(f110, _mm256_mul_ps(_mm256_sub_ps(f111, f110), dx));
    __m256 c0 = _mm256_add_ps(c00, _mm256_mul_ps(_mm256_sub_ps(c01, c00), dy));
    __m256 c1 = _mm256_add_ps(c10, _mm256_mul_ps(_mm256_sub_ps(c11, c10), dy));
    return _mm256_add_ps(c0, _mm256_mul_ps(_mm256_sub_ps(c1, c0), dz));
}
#endif  /* __AVX2__ */
#endif

static inline void mc_s_tri4(mc_sampler *s, const float *pz, const float *py,
                             const float *px, float *out) {
#if defined(__aarch64__)
    float32x4_t zv = vld1q_f32(pz), yv = vld1q_f32(py), xv = vld1q_f32(px);
    float32x4_t zf = vrndmq_f32(zv), yf = vrndmq_f32(yv), xf = vrndmq_f32(xv);
    int32x4_t zi = vcvtq_s32_f32(zf), yi = vcvtq_s32_f32(yf),
              xi = vcvtq_s32_f32(xf);
    // all-lanes in-bounds check: 0 <= c < n-1 per axis
    uint32x4_t ok = vcltq_u32(vreinterpretq_u32_s32(zi),
                              vdupq_n_u32((unsigned)(s->src.nz - 1)));
    ok = vandq_u32(ok, vcltq_u32(vreinterpretq_u32_s32(yi),
                                 vdupq_n_u32((unsigned)(s->src.ny - 1))));
    ok = vandq_u32(ok, vcltq_u32(vreinterpretq_u32_s32(xi),
                                 vdupq_n_u32((unsigned)(s->src.nx - 1))));
    if (vminvq_u32(ok) != 0) {
        float32x4_t dz = vsubq_f32(zv, zf), dy = vsubq_f32(yv, yf),
                    dx = vsubq_f32(xv, xf);
        if (s->src.dense) {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            const uint8_t *base = s->src.dense;
            vst1q_f32(out, mc_s_lerp8x4(
                base + (size_t)z0[0] * sy + (size_t)y0[0] * sx + x0[0],
                base + (size_t)z0[1] * sy + (size_t)y0[1] * sx + x0[1],
                base + (size_t)z0[2] * sy + (size_t)y0[2] * sx + x0[2],
                base + (size_t)z0[3] * sy + (size_t)y0[3] * sx + x0[3],
                sy, sx, dz, dy, dx));
            return;
        }
        // Masked group (see mc_s_tri8): one SIMD lerp for everyone; absent ->
        // zero block (correct as-is), straddlers -> zero block + scalar fixup.
        {
            int32_t z0[4], y0[4], x0[4];
            vst1q_s32(z0, zi); vst1q_s32(y0, yi); vst1q_s32(x0, xi);
            const uint8_t *b[4];
            unsigned strad = 0;
            for (int k = 0; k < 4; k++) {
                if ((z0[k] & 15) == 15 || (y0[k] & 15) == 15 ||
                    (x0[k] & 15) == 15) {
                    strad |= 1u << k;
                    b[k] = mc_s_zero4k;
                    continue;
                }
                const uint8_t *bk =
                    mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
                b[k] = bk ? bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                                  (x0[k] & 15))
                          : mc_s_zero4k;
            }
            vst1q_f32(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                        256, 16, dz, dy, dx));
            if (strad) {
                float dzs[4], dys[4], dxs[4];
                vst1q_f32(dzs, dz); vst1q_f32(dys, dy); vst1q_f32(dxs, dx);
                for (int k = 0; k < 4; k++)
                    if (strad & (1u << k))
                        out[k] = mc_s_cell_straddle(s, z0[k], y0[k], x0[k],
                                                    dzs[k], dys[k], dxs[k]);
            }
        }
        return;
    }
#endif
#if defined(__SSE4_1__) && !defined(__aarch64__)
    __m128 zv = _mm_loadu_ps(pz), yv = _mm_loadu_ps(py), xv = _mm_loadu_ps(px);
    __m128 zf = _mm_floor_ps(zv), yf = _mm_floor_ps(yv), xf = _mm_floor_ps(xv);
    __m128i zi = _mm_cvttps_epi32(zf), yi = _mm_cvttps_epi32(yf),
            xi = _mm_cvttps_epi32(xf);
    // all-lanes 0 <= c < n-1 (signed compares; negatives fail the >= 0 side)
    __m128i ok = _mm_and_si128(
        _mm_cmpgt_epi32(zi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.nz - 1), zi));
    ok = _mm_and_si128(ok, _mm_and_si128(
        _mm_cmpgt_epi32(yi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.ny - 1), yi)));
    ok = _mm_and_si128(ok, _mm_and_si128(
        _mm_cmpgt_epi32(xi, _mm_set1_epi32(-1)),
        _mm_cmpgt_epi32(_mm_set1_epi32(s->src.nx - 1), xi)));
    if (_mm_movemask_ps(_mm_castsi128_ps(ok)) == 0xF) {
        __m128 dz = _mm_sub_ps(zv, zf), dy = _mm_sub_ps(yv, yf),
               dx = _mm_sub_ps(xv, xf);
        int32_t z0[4], y0[4], x0[4];
        _mm_storeu_si128((__m128i *)z0, zi);
        _mm_storeu_si128((__m128i *)y0, yi);
        _mm_storeu_si128((__m128i *)x0, xi);
        if (s->src.dense) {
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            const uint8_t *base = s->src.dense;
            _mm_storeu_ps(out, mc_s_lerp8x4(
                base + (size_t)z0[0] * sy + (size_t)y0[0] * sx + x0[0],
                base + (size_t)z0[1] * sy + (size_t)y0[1] * sx + x0[1],
                base + (size_t)z0[2] * sy + (size_t)y0[2] * sx + x0[2],
                base + (size_t)z0[3] * sy + (size_t)y0[3] * sx + x0[3],
                sy, sx, dz, dy, dx));
            return;
        }
        // Masked group (see mc_s_tri8): one SIMD lerp for everyone; absent ->
        // zero block (correct as-is), straddlers -> zero block + scalar fixup.
        const uint8_t *b[4];
        unsigned strad = 0;
        for (int k = 0; k < 4; k++) {
            if ((z0[k] & 15) == 15 || (y0[k] & 15) == 15 || (x0[k] & 15) == 15) {
                strad |= 1u << k;
                b[k] = mc_s_zero4k;
                continue;
            }
            const uint8_t *bk =
                mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
            b[k] = bk ? bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                              (x0[k] & 15))
                      : mc_s_zero4k;
        }
        _mm_storeu_ps(out, mc_s_lerp8x4(b[0], b[1], b[2], b[3],
                                        256, 16, dz, dy, dx));
        if (strad) {
            float dzs[4], dys[4], dxs[4];
            _mm_storeu_ps(dzs, dz);
            _mm_storeu_ps(dys, dy);
            _mm_storeu_ps(dxs, dx);
            for (int k = 0; k < 4; k++)
                if (strad & (1u << k))
                    out[k] = mc_s_cell_straddle(s, z0[k], y0[k], x0[k],
                                                dzs[k], dys[k], dxs[k]);
        }
        return;
    }
#endif
    out[0] = mc_s_trilinear(s, pz[0], py[0], px[0]);
    out[1] = mc_s_trilinear(s, pz[1], py[1], px[1]);
    out[2] = mc_s_trilinear(s, pz[2], py[2], px[2]);
    out[3] = mc_s_trilinear(s, pz[3], py[3], px[3]);
}

#ifdef MC_S_HAVE_TRI8
// 8 positions at once (AVX2). Same fallback discipline as mc_s_tri4.
static inline void mc_s_tri8(mc_sampler *s, const float *pz, const float *py,
                             const float *px, float *out) {
    __m256 zv = _mm256_loadu_ps(pz), yv = _mm256_loadu_ps(py),
           xv = _mm256_loadu_ps(px);
    __m256 zf = _mm256_floor_ps(zv), yf = _mm256_floor_ps(yv),
           xf = _mm256_floor_ps(xv);
    __m256i zi = _mm256_cvttps_epi32(zf), yi = _mm256_cvttps_epi32(yf),
            xi = _mm256_cvttps_epi32(xf);
    __m256i ok = _mm256_and_si256(
        _mm256_cmpgt_epi32(zi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.nz - 1), zi));
    ok = _mm256_and_si256(ok, _mm256_and_si256(
        _mm256_cmpgt_epi32(yi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.ny - 1), yi)));
    ok = _mm256_and_si256(ok, _mm256_and_si256(
        _mm256_cmpgt_epi32(xi, _mm256_set1_epi32(-1)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(s->src.nx - 1), xi)));
    if (_mm256_movemask_ps(_mm256_castsi256_ps(ok)) == 0xFF) {
        __m256 dz = _mm256_sub_ps(zv, zf), dy = _mm256_sub_ps(yv, yf),
               dx = _mm256_sub_ps(xv, xf);
        int32_t z0[8], y0[8], x0[8];
        _mm256_storeu_si256((__m256i *)z0, zi);
        _mm256_storeu_si256((__m256i *)y0, yi);
        _mm256_storeu_si256((__m256i *)x0, xi);
        const uint8_t *b[8];
        if (s->src.dense) {
            const size_t sy = s->src.dsy, sx = s->src.dsx;
            for (int k = 0; k < 8; k++)
                b[k] = s->src.dense + (size_t)z0[k] * sy +
                       (size_t)y0[k] * sx + x0[k];
            _mm256_storeu_ps(out, mc_s_lerp8x8(b, sy, sx, dz, dy, dx));
            return;
        }
        // Masked group: every lane goes through ONE SIMD gather+lerp. Interior
        // lanes use their real block pointer (absent block -> the zero block,
        // whose lerp IS the correct 0). Straddling lanes also point at the zero
        // block for the gather and are overwritten scalar after -- so a group
        // with one straddler costs one SIMD lerp + one scalar cell, not 8
        // scalar cells (the old all-or-nothing fallback).
        unsigned strad = 0;
        for (int k = 0; k < 8; k++) {
            if ((z0[k] & 15) == 15 || (y0[k] & 15) == 15 || (x0[k] & 15) == 15) {
                strad |= 1u << k;
                b[k] = mc_s_zero4k;
                continue;
            }
            const uint8_t *bk =
                mc_s_block(s, z0[k] >> 4, y0[k] >> 4, x0[k] >> 4);
            b[k] = bk ? bk + (((z0[k] & 15) << 8) | ((y0[k] & 15) << 4) |
                              (x0[k] & 15))
                      : mc_s_zero4k;
        }
        _mm256_storeu_ps(out, mc_s_lerp8x8(b, 256, 16, dz, dy, dx));
        if (strad) {
            float dzs[8], dys[8], dxs[8];
            _mm256_storeu_ps(dzs, dz);
            _mm256_storeu_ps(dys, dy);
            _mm256_storeu_ps(dxs, dx);
            for (int k = 0; k < 8; k++)
                if (strad & (1u << k))
                    out[k] = mc_s_cell_straddle(s, z0[k], y0[k], x0[k],
                                                dzs[k], dys[k], dxs[k]);
        }
        return;
    }
    mc_s_tri4(s, pz, py, px, out);
    mc_s_tri4(s, pz + 4, py + 4, px + 4, out + 4);
}
#endif


#define BLK 16
#define BLKB 4096

// ---------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------
static const uint8_t *cache_block(const mc_sample_src *src,
                                  int bz, int by, int bx, uint8_t *tmp) {
    (void)tmp;
    // mc_cache_get decodes misses synchronously and returns an arena pointer
    // (non-owning; same stability contract as VC's BlockCache). In frozen
    // tick-phase a miss returns NULL — sampled as 0, recorded as feedback.
    return mc_cache_get((mc_cache *)src->ud, src->aux, bz, by, bx);
}

mc_sample_src mc_sample_src_cache(struct mc_cache *c, int lod,
                                  int nz, int ny, int nx) {
    mc_sample_src s = {0};
    s.ud = c; s.aux = lod; s.block = cache_block;
    s.owns_ptr = 1;                       // cache_block returns stable arena pointers
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

static const uint8_t *dense_block(const mc_sample_src *src,
                                  int bz, int by, int bx, uint8_t *tmp) {
    const uint8_t *vox = src->ud;
    int z0 = bz * BLK, y0 = by * BLK, x0 = bx * BLK;
    if (z0 >= src->nz || y0 >= src->ny || x0 >= src->nx) return NULL;
    int dz = src->nz - z0 < BLK ? src->nz - z0 : BLK;
    int dy = src->ny - y0 < BLK ? src->ny - y0 : BLK;
    int dx = src->nx - x0 < BLK ? src->nx - x0 : BLK;
    if (dz < BLK || dy < BLK || dx < BLK) memset(tmp, 0, BLKB);
    for (int z = 0; z < dz; z++)
        for (int y = 0; y < dy; y++)
            memcpy(tmp + ((z << 8) | (y << 4)),
                   vox + ((size_t)(z0 + z) * src->ny + (y0 + y)) * src->nx + x0,
                   (size_t)dx);
    return tmp;
}

mc_sample_src mc_sample_src_dense(const uint8_t *vox, int nz, int ny, int nx) {
    mc_sample_src s = {0};
    s.ud = (void *)(uintptr_t)vox;
    s.block = dense_block;                // kept for completeness; the direct
    s.dense = vox;                        // path below short-circuits it
    s.dsy = (size_t)ny * nx; s.dsx = (size_t)nx;
    s.nz = nz; s.ny = ny; s.nx = nx;
    return s;
}

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------
mc_sampler *mc_sampler_new(const mc_sample_src *src) {
    if (!src || !src->block) return NULL;
    mc_sampler *s = malloc(sizeof *s);
    if (!s) return NULL;
    s->src = *src;
    s->nbz = (src->nz + BLK - 1) / BLK;
    s->nby = (src->ny + BLK - 1) / BLK;
    s->nbx = (src->nx + BLK - 1) / BLK;
    // Per-entry 4KB scratch only for sources that synthesize into tmp (!owns_ptr).
    // The interactive cache path (owns_ptr) caches stable arena pointers -> the
    // sampler is ~5KB instead of ~1MB (no per-frame alloc/page-fault storm).
    s->scratch = (src->owns_ptr || src->dense) ? NULL
                 : malloc((size_t)MC_S_MEMO * 4096);
    mc_sampler_reset(s);
    return s;
}

void mc_sampler_free(mc_sampler *s) { if (s) { free(s->scratch); free(s); } }

void mc_sampler_reset(mc_sampler *s) {
    if (!s) return;
    for (int i = 0; i < MC_S_MEMO; i++) s->m[i].bz = -1;
    s->lbz = s->lby = s->lbx = -1;
    s->lptr = NULL;
    s->rbz = s->rby = s->rbx = -1; s->rres = 0;
}

float mc_sample_point(mc_sampler *s, float z, float y, float x, mc_filter f) {
    return mc_s_sample(s, z, y, x, f);
}

// Is the block covering voxel (z,y,x) resident at this sampler's level?
// Mirrors mc_s_trilinear's block lookup; NULL = absent (frozen miss / air).
static inline int mc_s_block_resident(mc_sampler *s, float z, float y, float x) {
    int z0 = (int)floorf(z), y0 = (int)floorf(y), x0 = (int)floorf(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 ||
        z0 >= s->src.nz || y0 >= s->src.ny || x0 >= s->src.nx) return 0;
    if (s->src.dense) return 1;                    // dense source: always resident
    int bz = z0 >> 4, by = y0 >> 4, bx = x0 >> 4;
    if (bz == s->rbz && by == s->rby && bx == s->rbx) return s->rres;  // ray-coherent
    // CHEAP probe if the source provides one: this must NOT decode (the LOD
    // fallback's whole point is to skip the fine-level decode-on-render-thread).
    int r = s->src.resident ? s->src.resident(&s->src, bz, by, bx)
                            : (mc_s_block(s, bz, by, bx) != NULL);   // may decode
    s->rbz = bz; s->rby = by; s->rbx = bx; s->rres = r;
    return r;
}

// ---------------------------------------------------------------------------
// LOD sampler: one sub-sampler per pyramid level + coarser-LOD fallback.
// ---------------------------------------------------------------------------
struct mc_lod_sampler {
    int nlods;
    mc_sampler *lv[8];          // lv[i] = sampler for level i (NULL if empty)
};

mc_lod_sampler *mc_lod_sampler_new(const mc_sample_lods *ls) {
    if (!ls || ls->nlods <= 0) return NULL;
    mc_lod_sampler *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->nlods = ls->nlods < 8 ? ls->nlods : 8;
    for (int i = 0; i < s->nlods; i++)
        if (ls->lods[i].block && ls->lods[i].nz > 0)
            s->lv[i] = mc_sampler_new(&ls->lods[i]);   // NULL ok: treated as empty
    return s;
}

void mc_lod_sampler_free(mc_lod_sampler *s) {
    if (!s) return;
    for (int i = 0; i < s->nlods; i++) mc_sampler_free(s->lv[i]);
    free(s);
}

void mc_lod_sampler_reset(mc_lod_sampler *s) {
    if (!s) return;
    for (int i = 0; i < s->nlods; i++) mc_sampler_reset(s->lv[i]);
}

float mc_lod_sample(mc_lod_sampler *s, int lod, int lod_fallback,
                    float z, float y, float x, mc_filter f) {
    if (!s) return 0.0f;
    if (mc_s_isnan(z) || mc_s_isnan(y) || mc_s_isnan(x)) return 0.0f;
    if (lod < 0) lod = 0;
    // L0 -> requested level: c_L = (c_0 + 0.5) * 2^-lod - 0.5
    if (lod > 0) {
        const float inv = 1.0f / (float)(1 << lod);
        z = (z + 0.5f) * inv - 0.5f;
        y = (y + 0.5f) * inv - 0.5f;
        x = (x + 0.5f) * inv - 0.5f;
    }
    for (int L = lod; L < s->nlods; L++) {
        mc_sampler *sub = s->lv[L];
        // Sample this level only if its block for the point is resident. (At the
        // requested level a resident block samples full quality; an absent one
        // either falls through to coarser or, without fallback, returns 0.)
        if (sub && mc_s_block_resident(sub, z, y, x))
            return mc_s_sample(sub, z, y, x, f);
        if (!lod_fallback) break;          // no walk: requested level only
        // descend to next coarser level: c' = (c + 0.5)*0.5 - 0.5
        z = (z + 0.5f) * 0.5f - 0.5f;
        y = (y + 0.5f) * 0.5f - 0.5f;
        x = (x + 0.5f) * 0.5f - 0.5f;
    }
    return 0.0f;   // nothing resident along the chain
}

static inline int pt_valid(const float *p) {
    if (p[0] != p[0] || p[1] != p[1] || p[2] != p[2]) return 0;   // NaN
    return p[0] >= 0.0f && p[1] >= 0.0f && p[2] >= 0.0f;
}

void mc_sample_points(mc_sampler *s, const float *zyx, size_t n,
                      mc_filter f, float *out) {
    if (f == MC_FILTER_NEAREST) {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            out[i] = pt_valid(p) ? mc_s_nearest(s, p[0], p[1], p[2]) : 0.0f;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            out[i] = pt_valid(p) ? mc_s_trilinear(s, p[0], p[1], p[2]) : 0.0f;
        }
    }
}

void mc_sample_points_u8(mc_sampler *s, const float *zyx, size_t n,
                         mc_filter f, uint8_t *out) {
    if (f == MC_FILTER_NEAREST) {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            float v = pt_valid(p) ? mc_s_nearest(s, p[0], p[1], p[2]) : 0.0f;
            out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            const float *p = zyx + i * 3;
            float v = pt_valid(p) ? mc_s_trilinear(s, p[0], p[1], p[2]) : 0.0f;
            out[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
        }
    }
}

// ============================================================================
// mc_render — surface rendering, compositing, LOD, 3D resampling
// ============================================================================
// ---------------------------------------------------------------------------
// core
// ---------------------------------------------------------------------------
static inline uint8_t to_u8(float v) {
    return (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255 : (int)(v + 0.5f));
}

// Per-image constants hoisted out of the pixel loop.
typedef struct {
    mc_filter filter;
    mc_comp comp;
    float t0, dt;
    int nsteps;                 // iterations of the [t0, t1] dt walk
    float a_min, a_op;          // alpha params, clamped
    // MC_COMP_SHADED (defaults resolved here; see mc_render_params docs)
    float Lz, Ly, Lx;           // unit light dir, toward the light
    int headlight;              // light[] was zero: use -ray dir per pixel
    float ka, kd, ks, shin;     // ambient / diffuse / specular / exponent
    float sigma;                // extinction per unit density per voxel
    float shadow, sss;          // shadow strength, translucency weight
    float g0sq;                 // grad_g0^2 (surface-ness knee)
    int sh_steps;               // secondary-march steps toward the light
    float sh_dt;                // secondary-march step, voxels
    float curv;                 // ridge/valley shading weight
    float pct;                  // percentile rank (0,1]
    int ink;                    // MC_COMP_INK: transmission + cone SSS
    float trans;                // backlight weight [0,1]
    int lrel;                   // light[] is in the per-pixel surface frame
    int lock_steps;             // INK sheet-lock: composite this many steps from
                                // the sheet entry (0 = whole slab). The [t0,t1]
                                // slab becomes a SEARCH range; the dense run
                                // nearest t=0 is the sheet.
    float lock_thr;             // density (post alpha_min scale) that counts as
                                // sheet material during the search
    float alut[256];            // per-step opacity 1-exp(-sigma*d*dt), d=i/255
} rcfg_t;

static rcfg_t make_cfg(const mc_render_params *p) {
    rcfg_t c;
    c.filter = p->filter;
    c.comp = p->comp;
    c.dt = p->dt > 0.0f ? p->dt : 1.0f;
    float t0 = p->t0, t1 = p->t1;
    if (t1 < t0) { float tmp = t0; t0 = t1; t1 = tmp; }
    c.t0 = t0;
    c.nsteps = 0;
    for (float t = t0; t <= t1 + 1e-4f; t += c.dt) c.nsteps++;
    c.a_min = p->alpha_min < 0.0f ? 0.0f :
              p->alpha_min > 0.99f ? 0.99f : p->alpha_min;
    c.a_op  = p->alpha_opacity <= 0.0f ? 1.0f :
              p->alpha_opacity > 1.0f ? 1.0f : p->alpha_opacity;
    float ll = p->light[0] * p->light[0] + p->light[1] * p->light[1] +
               p->light[2] * p->light[2];
    c.headlight = ll < 1e-12f;
    if (c.headlight) { c.Lz = c.Ly = 0.0f; c.Lx = 1.0f; }
    else {
        float inv = 1.0f / sqrtf(ll);
        c.Lz = p->light[0] * inv; c.Ly = p->light[1] * inv;
        c.Lx = p->light[2] * inv;
    }
    c.ka   = p->ambient   > 0.0f ? p->ambient   : 0.25f;
    c.kd   = p->diffuse   > 0.0f ? p->diffuse   : 0.75f;
    c.ks   = p->specular  > 0.0f ? p->specular  : 0.20f;
    c.shin = p->shininess > 0.0f ? p->shininess : 24.0f;
    c.sigma = p->absorption > 0.0f ? p->absorption : 1.0f;
    c.shadow = p->shadow < 0.0f ? 0.0f : p->shadow > 1.0f ? 1.0f : p->shadow;
    c.sss = p->sss < 0.0f ? 0.0f : p->sss;
    float g0 = p->grad_g0 > 0.0f ? p->grad_g0 : 8.0f;
    c.g0sq = g0 * g0;
    c.sh_steps = 12;            // 24 voxels of reach at sh_dt = 2: enough to
    c.sh_dt = 2.0f;             // self-shadow a sheet, cheap enough per ray
    c.curv = p->curvature;
    c.pct = (p->percentile > 0.0f && p->percentile <= 1.0f) ? p->percentile
                                                            : 0.9f;
    c.ink = p->comp == MC_COMP_INK;
    c.trans = p->transmission > 0.0f ? (p->transmission > 1.0f ? 1.0f : p->transmission)
                                     : (c.ink ? 0.35f : 0.0f);
    if (c.ink && p->sss <= 0.0f) c.sss = 0.5f;     // ink default: SSS on
    c.lock_steps = 0;
    if (c.ink && p->ink_lock > 0.0f) {
        c.lock_steps = (int)(p->ink_lock / c.dt + 0.5f);
        if (c.lock_steps < 1) c.lock_steps = 1;
    }
    c.lock_thr = 0.35f;        // "solidly sheet" density for the search
    c.lrel = p->light_surface_rel != 0;
    for (int i = 0; i < 256; i++)
        c.alut[i] = 1.0f - expf(-c.sigma * ((float)i * (1.0f / 255.0f)) * c.dt);
    return c;
}

// Hoare quickselect: value at rank k of a[0..n-1] (a is scrambled in place).
static float rank_select(float *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float p = a[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] < p) i++;
            while (a[j] > p) j--;
            if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
    }
    return a[k];
}

// MC_COMP_PERCENTILE: the ray's samples are collected (strided down to a
// 1024 cap for absurdly deep slabs) and the rank-`pct` value returned.
static uint8_t pct_ray(mc_sampler *s, const float *P, float nz, float ny,
                       float nx, const rcfg_t *cfg) {
    float buf[1024];
    int stride = (cfg->nsteps + 1023) / 1024;
    if (stride < 1) stride = 1;
    float sz_ = cfg->dt * nz * stride, sy_ = cfg->dt * ny * stride,
          sx_ = cfg->dt * nx * stride;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    int n = 0;
    for (int it = 0; it < cfg->nsteps; it += stride) {
        buf[n++] = mc_s_sample(s, pz, py, px, cfg->filter);
        pz += sz_; py += sy_; px += sx_;
    }
    if (!n) return 0;
    int k = (int)(cfg->pct * (float)(n - 1) + 0.5f);
    return to_u8(rank_select(buf, n, k));
}

// MC_COMP_DEPTH: first t where the value crosses alpha_min, mapped 1..255
// over the slab (0 = no hit).
static uint8_t depth_ray(mc_sampler *s, const float *P, float nz, float ny,
                         float nx, const rcfg_t *cfg) {
    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    for (int it = 0; it < cfg->nsteps; it++) {
        float v = mc_s_sample(s, pz, py, px, cfg->filter);
        if (v * (1.0f / 255.0f) > cfg->a_min) {
            float frac = cfg->nsteps > 1 ? (float)it / (float)(cfg->nsteps - 1)
                                         : 0.0f;
            return (uint8_t)(1.0f + 254.0f * frac + 0.5f);
        }
        pz += sz_; py += sy_; px += sx_;
    }
    return 0;
}

// Batched point sampling: SIMD tri8/tri4 chunks + scalar tail. The shaded /
// ink march samples rays, gradient taps and cone taps through this instead of
// per-point scalar trilinear (which was 50% of the ink profile).
static inline void mc_s_batchN(mc_sampler *s, const float *bz, const float *by,
                               const float *bx, float *out, int n, mc_filter f) {
    int k = 0;
    if (f == MC_FILTER_TRILINEAR) {
#ifdef MC_S_HAVE_TRI8
        for (; k + 8 <= n; k += 8) mc_s_tri8(s, bz + k, by + k, bx + k, out + k);
#endif
#if defined(__aarch64__) || defined(__SSE4_1__)
        for (; k + 4 <= n; k += 4) mc_s_tri4(s, bz + k, by + k, bx + k, out + k);
#endif
    }
    for (; k < n; k++) out[k] = mc_s_sample(s, bz[k], by[k], bx[k], f);
}

// MC_COMP_SHADED / MC_COMP_INK: front-to-back emission-absorption along
// P + t*N with gradient-normal lighting (see the header docs). All volume
// taps run through mc_s_batchN: the slab walk in 64-step chunks, the 6
// gradient taps as one 8-wide batch, and the (INK) 3-tap cone march as one
// 36-position batch. Per-step opacity 1-exp(-sigma*d*dt) comes from a 256-
// entry LUT built per image (sigma*dt is constant); the cone march drops the
// tau early-break (bounded extra samples, batching wins more).
#define MC_SH_CHUNK 256
static uint8_t shade_ray(mc_sampler *s, const float *P, float nz, float ny,
                         float nx, const rcfg_t *cfg, const float *Lp) {
    const mc_filter f = cfg->filter;
    const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
    float Lz, Ly, Lx;
    if (Lp) { Lz = Lp[0]; Ly = Lp[1]; Lx = Lp[2]; }          // surface-relative
    else if (cfg->headlight) { Lz = -nz; Ly = -ny; Lx = -nx; }
    else { Lz = cfg->Lz; Ly = cfg->Ly; Lx = cfg->Lx; }
    // view = toward the camera = -ray dir; half vector for Blinn-Phong
    float hz = Lz - nz, hy = Ly - ny, hx = Lx - nx;
    float hl = hz * hz + hy * hy + hx * hx;
    if (hl > 1e-12f) {
        hl = 1.0f / sqrtf(hl);
        hz *= hl; hy *= hl; hx *= hl;
    }
    // cone basis for the INK subsurface taps (two perpendiculars to L)
    const int ntap = cfg->ink ? 3 : 1;
    float e1z = 0, e1y = 0, e1x = 0, e2z = 0, e2y = 0, e2x = 0;
    float dirz[3], diry[3], dirx[3];
    dirz[0] = Lz; diry[0] = Ly; dirx[0] = Lx;
    if (ntap > 1) {
        float az = fabsf(Lz), ay = fabsf(Ly);
        float bz_ = 0, by_ = 0, bx_ = 0;
        if (az < 0.9f) bz_ = 1; else if (ay < 0.9f) by_ = 1; else bx_ = 1;
        e1z = Ly * bx_ - Lx * by_; e1y = Lx * bz_ - Lz * bx_; e1x = Lz * by_ - Ly * bz_;
        float il = 1.0f / sqrtf(e1z * e1z + e1y * e1y + e1x * e1x + 1e-12f);
        e1z *= il; e1y *= il; e1x *= il;
        e2z = Ly * e1x - Lx * e1y; e2y = Lx * e1z - Lz * e1x; e2x = Lz * e1y - Ly * e1z;
        const float k = 0.35f;
        for (int t = 1; t < 3; t++) {
            float dz_ = Lz + (t == 1 ? k * e1z : -k * e2z);
            float dy_ = Ly + (t == 1 ? k * e1y : -k * e2y);
            float dx_ = Lx + (t == 1 ? k * e1x : -k * e2x);
            float il2 = 1.0f / sqrtf(dz_ * dz_ + dy_ * dy_ + dx_ * dx_);
            dirz[t] = dz_ * il2; diry[t] = dy_ * il2; dirx[t] = dx_ * il2;
        }
    }
    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float acc = 0.0f, T = 1.0f;
    const int want_tau = cfg->shadow > 0.0f || cfg->sss > 0.0f;

    float vz[MC_SH_CHUNK], vy[MC_SH_CHUNK], vx[MC_SH_CHUNK], vals[MC_SH_CHUNK];
    float tb_z[64], tb_y[64], tb_x[64], tb_v[64];   // taps: 8 grad + <=36 cone

    const int total = cfg->lock_steps > 0 && cfg->nsteps > MC_SH_CHUNK
                          ? MC_SH_CHUNK : cfg->nsteps;   // lock searches chunk 0 only
    for (int base = 0; base < total && T >= 0.02f; base += MC_SH_CHUNK) {
        const int m = cfg->nsteps - base < MC_SH_CHUNK ? cfg->nsteps - base
                                                       : MC_SH_CHUNK;
        // batch-sample this chunk of the slab walk
        {
            float pz = P[0] + (cfg->t0 + (float)base * cfg->dt) * nz;
            float py = P[1] + (cfg->t0 + (float)base * cfg->dt) * ny;
            float px = P[2] + (cfg->t0 + (float)base * cfg->dt) * nx;
            for (int k = 0; k < m; k++) {
                vz[k] = pz; vy[k] = py; vx[k] = px;
                pz += sz_; py += sy_; px += sx_;
            }
            mc_s_batchN(s, vz, vy, vx, vals, m, f);
        }
        int k0 = 0, k1 = m;
        if (cfg->lock_steps > 0 && base == 0) {
            // SHEET LOCK: the slab is a SEARCH range. Surfaces wander -- the
            // sheet may sit in front of, behind, or across t=0 -- so find the
            // dense run (>=2 consecutive samples above lock_thr) whose center
            // is nearest the segmentation surface, then composite only
            // lock_steps from its entry face (the ink-bearing crust). Material
            // beyond that -- sheet interior, gaps, the neighboring wrap -- is
            // excluded instead of overlaying papyrus texture on the ink.
            const int i_surf = cfg->t0 < 0.0f ? (int)(-cfg->t0 / cfg->dt + 0.5f) : 0;
            int best_s = -1, best_d = 1 << 30;
            int run_s = -1;
            for (int k = 0; k <= m; k++) {
                float d = k < m ? (vals[k] * (1.0f / 255.0f) - cfg->a_min) *
                                      (cfg->a_op / (1.0f - cfg->a_min))
                                : 0.0f;
                if (k < m && d >= cfg->lock_thr) {
                    if (run_s < 0) run_s = k;
                } else if (run_s >= 0) {
                    if (k - run_s >= 2) {                 // a real run, not a fleck
                        const int center = (run_s + k - 1) >> 1;
                        const int dist = center > i_surf ? center - i_surf
                                                         : i_surf - center;
                        if (dist < best_d) { best_d = dist; best_s = run_s; }
                    }
                    run_s = -1;
                }
            }
            if (best_s < 0)
                return 0;                                 // no sheet in the search slab
            k0 = best_s > 0 ? best_s - 1 : 0;             // include the entry gradient
            k1 = best_s + cfg->lock_steps;
            if (k1 > m) k1 = m;
        }
        for (int k = k0; k < k1; k++) {
            float v = vals[k];
            float d = (v * (1.0f / 255.0f) - a_th) * a_sc;
            if (d <= 0.0f) continue;                // air: free skip
            if (d > 1.0f) d = 1.0f;
            float a = cfg->alut[(int)(d * 255.0f + 0.5f)];
            const float pz = vz[k], py = vy[k], px = vx[k];
            // 6 gradient taps in ONE 8-wide batch (2 dummy lanes)
            tb_z[0] = pz + 1; tb_y[0] = py;     tb_x[0] = px;
            tb_z[1] = pz - 1; tb_y[1] = py;     tb_x[1] = px;
            tb_z[2] = pz;     tb_y[2] = py + 1; tb_x[2] = px;
            tb_z[3] = pz;     tb_y[3] = py - 1; tb_x[3] = px;
            tb_z[4] = pz;     tb_y[4] = py;     tb_x[4] = px + 1;
            tb_z[5] = pz;     tb_y[5] = py;     tb_x[5] = px - 1;
            tb_z[6] = tb_z[7] = -1; tb_y[6] = tb_y[7] = -1; tb_x[6] = tb_x[7] = -1;
            mc_s_batchN(s, tb_z, tb_y, tb_x, tb_v, 8, f);
            const float vzp = tb_v[0], vzm = tb_v[1], vyp = tb_v[2],
                        vym = tb_v[3], vxp = tb_v[4], vxm = tb_v[5];
            float gz = vzp - vzm, gy = vyp - vym, gx = vxp - vxm;
            float g2 = 0.25f * (gz * gz + gy * gy + gx * gx);
            float w = g2 / (g2 + cfg->g0sq);        // surface-ness
            float diff = 0.0f, spec = 0.0f;
            if (g2 > 1e-8f) {
                float gi = 1.0f / sqrtf(gz * gz + gy * gy + gx * gx);
                float uz = gz * gi, uy = gy * gi, ux = gx * gi;
                diff = fabsf(uz * Lz + uy * Ly + ux * Lx);   // two-sided
                float ndh = fabsf(uz * hz + uy * hy + ux * hx);
                spec = powf(ndh, cfg->shin);
            }
            float lit = cfg->ka + (1.0f - w) * cfg->kd;     // interior: emissive
            float Tl = 1.0f;
            if (want_tau && w > 0.05f) {
                // cone subsurface march: all taps' positions in ONE batch
                int n = 0;
                for (int t = 0; t < ntap; t++) {
                    float qz = pz, qy = py, qx = px;
                    for (int j = 0; j < cfg->sh_steps; j++) {
                        qz += cfg->sh_dt * dirz[t];
                        qy += cfg->sh_dt * diry[t];
                        qx += cfg->sh_dt * dirx[t];
                        tb_z[n] = qz; tb_y[n] = qy; tb_x[n] = qx; n++;
                    }
                }
                mc_s_batchN(s, tb_z, tb_y, tb_x, tb_v, n, f);
                float tl_sum = 0.0f, glow_sum = 0.0f;
                for (int t = 0; t < ntap; t++) {
                    float tau = 0.0f;
                    const float *tv = tb_v + t * cfg->sh_steps;
                    for (int j = 0; j < cfg->sh_steps; j++) {
                        float sd = (tv[j] * (1.0f / 255.0f) - a_th) * a_sc;
                        if (sd > 0.0f) {
                            if (sd > 1.0f) sd = 1.0f;
                            tau += cfg->sigma * sd * cfg->sh_dt;
                        }
                    }
                    if (tau > 6.0f) tau = 6.0f;
                    tl_sum += expf(-tau);
                    glow_sum += expf(-0.3f * tau);
                }
                const float inv_n = 1.0f / (float)ntap;
                Tl = tl_sum * inv_n;
                lit += cfg->sss * w * glow_sum * inv_n;     // translucent glow
            }
            float shfac = 1.0f - cfg->shadow + cfg->shadow * Tl;
            lit += w * cfg->kd * diff * shfac;
            if (cfg->curv != 0.0f) {
                // density Laplacian, free from the gradient taps: negative at
                // ridges/crests (brighten), positive in cracks/pits (darken)
                float lap = vzp + vzm + vyp + vym + vxp + vxm - 6.0f * v;
                float cc = -lap * (1.0f / 510.0f);
                if (cc > 1.0f) cc = 1.0f; else if (cc < -1.0f) cc = -1.0f;
                lit += cfg->curv * cc * w;
                if (lit < 0.0f) lit = 0.0f;
            }
            float shade = v * lit + 255.0f * cfg->ks * spec * shfac * w;
            acc += T * a * shade;
            T *= 1.0f - a;
            if (T < 0.02f) break;
        }
    }
    // INK: backlit transmission -- the light that survives the slab. Papyrus is
    // translucent, carbon ink denser, so ink reads locally dark in transmission.
    // Gate on material actually traversed (1-T): a pure-air ray accumulates nothing
    // and must stay 0 (air -> black), not glow at trans*255.
    if (cfg->trans > 0.0f)
        acc += T * cfg->trans * 255.0f * (1.0f - T);
    return to_u8(acc);
}


// Composite one ray. Trilinear rays are consumed in chunks of 4 steps via
// mc_s_tri4 (NEON gather+lerp on aarch64, ~1.4x the scalar core); the
// accumulation itself stays sequential per chunk, which keeps ALPHA\'s
// front-to-back order and early-out exact. Positions are P + t*N with t
// advanced additively, as before.
static uint8_t render_pixel(mc_sampler *s, const float *P, const float *N,
                            const rcfg_t *cfg, const float *Tx) {
    if (!pt_valid(P)) return 0;
    if (cfg->comp == MC_COMP_NONE || !N)
        return to_u8(mc_s_sample(s, P[0], P[1], P[2], cfg->filter));
    float nz = N[0], ny = N[1], nx = N[2];
    float n2 = nz * nz + ny * ny + nx * nx;
    if (n2 < 1e-12f)
        return to_u8(mc_s_sample(s, P[0], P[1], P[2], cfg->filter));
    if (n2 < 0.9998f || n2 > 1.0002f) {     // gen paths emit unit normals
        float nl = 1.0f / sqrtf(n2);
        nz *= nl; ny *= nl; nx *= nl;
    }
    if (cfg->comp == MC_COMP_SHADED || cfg->comp == MC_COMP_INK) {
        // Surface-relative raking: build the per-pixel frame from the surface
        // normal + the screen-x tangent and express light[] in it. light[0] =
        // elevation component (along -N, toward the camera side), light[2] =
        // along screen-x (t1), light[1] = along N x t1 (t2). Falls back to the
        // volume frame when the tangent degenerates.
        float L[3];
        const float *Lp = NULL;
        if (cfg->lrel && !cfg->headlight && Tx) {
            float tz = Tx[0], ty = Tx[1], tx = Tx[2];
            const float dn = tz * nz + ty * ny + tx * nx;
            tz -= dn * nz; ty -= dn * ny; tx -= dn * nx;     // project off N
            float tl = tz * tz + ty * ty + tx * tx;
            if (tl > 1e-12f) {
                tl = 1.0f / sqrtf(tl);
                tz *= tl; ty *= tl; tx *= tl;
                // t2 = N x t1, computed in (x,y,z) then stored (z,y,x)
                const float ux = ny * tz - nz * ty;   // n_y t_z - n_z t_y
                const float uy = nz * tx - nx * tz;   // n_z t_x - n_x t_z
                const float uz = nx * ty - ny * tx;   // n_x t_y - n_y t_x
                L[0] = cfg->Lz * -nz + cfg->Ly * uz + cfg->Lx * tz;
                L[1] = cfg->Lz * -ny + cfg->Ly * uy + cfg->Lx * ty;
                L[2] = cfg->Lz * -nx + cfg->Ly * ux + cfg->Lx * tx;
                Lp = L;
            }
        }
        return shade_ray(s, P, nz, ny, nx, cfg, Lp);
    }
    if (cfg->comp == MC_COMP_PERCENTILE)
        return pct_ray(s, P, nz, ny, nx, cfg);
    if (cfg->comp == MC_COMP_DEPTH)
        return depth_ray(s, P, nz, ny, nx, cfg);

    const float sz_ = cfg->dt * nz, sy_ = cfg->dt * ny, sx_ = cfg->dt * nx;
    float pz = P[0] + cfg->t0 * nz, py = P[1] + cfg->t0 * ny,
          px = P[2] + cfg->t0 * nx;
    const float a_th = cfg->a_min, a_sc = cfg->a_op / (1.0f - cfg->a_min);
    float acc = 0.0f, A = 0.0f, mn = 255.0f, mx = 0.0f, sum = 0.0f,
          sum2 = 0.0f;
    int it = 0, done = 0;

    if (cfg->filter == MC_FILTER_TRILINEAR) {
// NOTE: composites deliberately stay 4-wide. Measured on Zen 3 (EPYC
        // 7763): 8-wide ray chunks ran 1.6x SLOWER than two independent 4-wide
        // chunks (the 8-long insert-gather dependency chain over z-strided
        // addresses serializes); 8-wide only wins for adjacent-pixel loads
        // (the slice path below).
        for (; it + 4 <= cfg->nsteps && !done; it += 4) {
            float bz[4], by[4], bx[4], v4[4];
            for (int k = 0; k < 4; k++) {
                bz[k] = pz; by[k] = py; bx[k] = px;
                pz += sz_; py += sy_; px += sx_;
            }
            mc_s_tri4(s, bz, by, bx, v4);
            switch (cfg->comp) {
            case MC_COMP_MIN:
                for (int k = 0; k < 4; k++) if (v4[k] < mn) mn = v4[k];
                break;
            case MC_COMP_MAX:
                for (int k = 0; k < 4; k++) if (v4[k] > mx) mx = v4[k];
                if (mx >= 255.0f) done = 1;     // saturated
                break;
            case MC_COMP_MEAN:
                sum += v4[0] + v4[1] + v4[2] + v4[3];
                break;
            case MC_COMP_STDDEV:
                for (int k = 0; k < 4; k++) {
                    sum += v4[k]; sum2 += v4[k] * v4[k];
                }
                break;
            default:                            // ALPHA
                for (int k = 0; k < 4 && !done; k++) {
                    float a = (v4[k] * (1.0f / 255.0f) - a_th) * a_sc;
                    if (a > 0.0f) {
                        if (a > 1.0f) a = 1.0f;
                        acc += (1.0f - A) * a * v4[k];
                        A   += (1.0f - A) * a;
                        if (A >= 0.98f) done = 1;
                    }
                }
                break;
            }
        }
    }
    for (; it < cfg->nsteps && !done; it++) {
        float v = mc_s_sample(s, pz, py, px, cfg->filter);
        switch (cfg->comp) {
        case MC_COMP_MIN:  if (v < mn) mn = v; break;
        case MC_COMP_MAX:  if (v > mx) mx = v; break;
        case MC_COMP_MEAN: sum += v; break;
        case MC_COMP_STDDEV: sum += v; sum2 += v * v; break;
        default: {                              // ALPHA
            float a = (v * (1.0f / 255.0f) - a_th) * a_sc;
            if (a > 0.0f) {
                if (a > 1.0f) a = 1.0f;
                acc += (1.0f - A) * a * v;
                A   += (1.0f - A) * a;
                if (A >= 0.98f) done = 1;
            }
            break;
        }
        }
        pz += sz_; py += sy_; px += sx_;
    }
    switch (cfg->comp) {
    case MC_COMP_MIN:  return to_u8(mn);
    case MC_COMP_MAX:  return to_u8(mx);
    case MC_COMP_MEAN:
        return to_u8(cfg->nsteps ? sum / (float)cfg->nsteps : 0.0f);
    case MC_COMP_STDDEV: {
        if (!cfg->nsteps) return 0;
        float m = sum / (float)cfg->nsteps;
        float var = sum2 / (float)cfg->nsteps - m * m;
        return to_u8(var > 0.0f ? sqrtf(var) : 0.0f);
    }
    case MC_COMP_ALPHA: return to_u8(acc);
    default:           return 0;
    }
}

void mc_render_points(mc_sampler *s,
                      const float *pts, const float *normals,
                      int w, int h, const mc_render_params *p, uint8_t *out) {
    rcfg_t cfg = make_cfg(p);
    size_t n = (size_t)w * h;
    if (cfg.comp == MC_COMP_NONE || !normals) {
        // slice fast path: no per-pixel normal handling, branch on the
        // filter once
        if (cfg.filter == MC_FILTER_NEAREST) {
            for (size_t k = 0; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_nearest(s, P[0], P[1], P[2])) : 0;
            }
        } else {
            // 4/8 pixels per mc_s_tri4/8 call (SIMD gather+lerp)
            size_t k = 0;
#ifdef MC_S_HAVE_TRI8
            for (; k + 8 <= n; k += 8) {
                const float *P = pts + k * 3;
                int allv = 1;
                for (int q = 0; q < 8; q++) allv &= pt_valid(P + q * 3);
                if (allv) {
                    float bz[8], by[8], bx[8], v8[8];
                    for (int q = 0; q < 8; q++) {
                        bz[q] = P[q * 3]; by[q] = P[q * 3 + 1];
                        bx[q] = P[q * 3 + 2];
                    }
                    mc_s_tri8(s, bz, by, bx, v8);
                    for (int q = 0; q < 8; q++) out[k + q] = to_u8(v8[q]);
                } else {
                    for (size_t q = k; q < k + 8; q++) {
                        const float *Q = pts + q * 3;
                        out[q] = pt_valid(Q)
                            ? to_u8(mc_s_trilinear(s, Q[0], Q[1], Q[2])) : 0;
                    }
                }
            }
#endif
            for (; k + 4 <= n; k += 4) {
                const float *P = pts + k * 3;
                if (pt_valid(P) && pt_valid(P + 3) &&
                    pt_valid(P + 6) && pt_valid(P + 9)) {
                    float bz[4] = { P[0], P[3], P[6], P[9]  };
                    float by[4] = { P[1], P[4], P[7], P[10] };
                    float bx[4] = { P[2], P[5], P[8], P[11] };
                    float v4[4];
                    mc_s_tri4(s, bz, by, bx, v4);
                    out[k]     = to_u8(v4[0]);
                    out[k + 1] = to_u8(v4[1]);
                    out[k + 2] = to_u8(v4[2]);
                    out[k + 3] = to_u8(v4[3]);
                } else {
                    for (size_t q = k; q < k + 4; q++) {
                        const float *Q = pts + q * 3;
                        out[q] = pt_valid(Q)
                            ? to_u8(mc_s_trilinear(s, Q[0], Q[1], Q[2])) : 0;
                    }
                }
            }
            for (; k < n; k++) {
                const float *P = pts + k * 3;
                out[k] = pt_valid(P)
                             ? to_u8(mc_s_trilinear(s, P[0], P[1], P[2])) : 0;
            }
        }
        return;
    }
    const int needT = cfg.lrel && !cfg.headlight &&
                      (cfg.comp == MC_COMP_SHADED || cfg.comp == MC_COMP_INK);
    for (size_t k = 0; k < n; k++) {
        float T[3];
        const float *Tx = NULL;
        if (needT && w > 1) {
            // screen-x tangent from the neighboring pixel's position
            const size_t col = k % (size_t)w;
            const float *Pc = pts + k * 3;
            const float *Pn = pts + (col + 1 < (size_t)w ? k + 1 : k - 1) * 3;
            if (pt_valid(Pc) && pt_valid(Pn)) {
                T[0] = Pn[0] - Pc[0]; T[1] = Pn[1] - Pc[1]; T[2] = Pn[2] - Pc[2];
                if (col + 1 >= (size_t)w) { T[0] = -T[0]; T[1] = -T[1]; T[2] = -T[2]; }
                Tx = T;
            }
        }
        out[k] = render_pixel(s, pts + k * 3, normals + k * 3, &cfg, Tx);
    }
}

// ---------------------------------------------------------------------------
// parallel core: row bands, one sampler per worker
// ---------------------------------------------------------------------------
// rowgen fills one row of points (+normals) into band-local scratch; plane
// and quad renders go through this so no W*H grid is ever materialized
// (a 1024^2 trilinear frame otherwise mallocs and touches 24 MB of points).
typedef void (*rowgen_fn)(const void *ud, int row, int w,
                          float *pts, float *normals);

typedef struct {
    const mc_sample_src *src;
    const float *pts, *normals;     // dense mode (rowgen == NULL)
    rowgen_fn rowgen;               // strip mode
    const void *rg_ud;
    int w, h;
    const mc_render_params *p;
    uint8_t *out;
    int row0, row1;
    // LOD-fallback mode: when ls != NULL, sample via a per-band mc_lod_sampler
    // at level `lod` with coarser-LOD fallback instead of the single `src`.
    const mc_sample_lods *ls;
    int lod;
} band_t;

// LOD-fallback point render: coords are native level-0 voxel space; each pixel
// samples the requested level (`lod`) with coarser-LOD fallback. Slice path
// (comp==NONE) only — the interactive nav case; composites stay single-level.
//
// Fast path: the common case is "all blocks resident at `lod`" (steady state and
// air, since air decodes to a resident zero block). For a group of 8 points whose
// level-`lod` blocks are ALL resident, sample with the 8-wide SIMD kernel on the
// level-`lod` sampler — same speed as the non-fallback render. Only groups that
// touch an ABSENT block fall to the per-lane coarse-LOD walk (transient, while a
// freshly-entered level streams in).
static void render_points_lod(mc_lod_sampler *ls, int lod,
                              const float *pts, int w, int h,
                              const mc_render_params *p, uint8_t *out) {
    const mc_filter f = (mc_filter)p->filter;
    const size_t n = (size_t)w * h;
    mc_sampler *L = (lod >= 0 && lod < ls->nlods) ? ls->lv[lod] : NULL;
    const float inv = lod > 0 ? 1.0f / (float)(1 << lod) : 1.0f;
    // L0 -> level-`lod` voxel coord (half-voxel-center correct).
    #define MC_L0_TO_L(c) ((c + 0.5f) * inv - 0.5f)

    size_t k = 0;
#ifdef MC_S_HAVE_TRI8
    if (L && f == MC_FILTER_TRILINEAR) {
        float pz[8], py[8], px[8], v8[8];
        for (; k + 8 <= n; k += 8) {
            int allv = 1, allres = 1;
            for (int q = 0; q < 8; q++) {
                const float *P = pts + (k + q) * 3;
                if (!pt_valid(P)) { allv = 0; break; }
                float z = MC_L0_TO_L(P[0]), y = MC_L0_TO_L(P[1]), x = MC_L0_TO_L(P[2]);
                pz[q] = z; py[q] = y; px[q] = x;
                // resident block at L? (covers air: air is a resident zero block)
                if (!mc_s_block_resident(L, z, y, x)) { allres = 0; }
            }
            if (allv && allres) {
                mc_s_tri8(L, pz, py, px, v8);
                for (int q = 0; q < 8; q++) out[k + q] = to_u8(v8[q]);
            } else {
                for (int q = 0; q < 8; q++) {
                    const float *P = pts + (k + q) * 3;
                    out[k + q] = pt_valid(P)
                        ? to_u8(mc_lod_sample(ls, lod, 1, P[0], P[1], P[2], f)) : 0;
                }
            }
        }
    }
#endif
    for (; k < n; k++) {
        const float *P = pts + k * 3;
        out[k] = pt_valid(P)
            ? to_u8(mc_lod_sample(ls, lod, 1, P[0], P[1], P[2], f)) : 0;
    }
    #undef MC_L0_TO_L
}

static void *band_main(void *ud) {
    band_t *b = ud;
    if (b->ls) {
        mc_lod_sampler *ls = mc_lod_sampler_new(b->ls);
        if (!ls) return NULL;
        float *row = malloc((size_t)b->w * 3 * sizeof(float));
        if (row) {
            for (int i = b->row0; i < b->row1; i++) {
                b->rowgen(b->rg_ud, i, b->w, row, NULL);   // L0 coords, no normals
                render_points_lod(ls, b->lod, row, b->w, 1, b->p,
                                  b->out + (size_t)i * b->w);
            }
            free(row);
        } else memset(b->out + (size_t)b->row0 * b->w, 0,
                      (size_t)(b->row1 - b->row0) * b->w);
        mc_lod_sampler_free(ls);
        return NULL;
    }
    mc_sampler *s = mc_sampler_new(b->src);
    if (!s) return NULL;
    if (!b->rowgen) {
        size_t off = (size_t)b->row0 * b->w;
        mc_render_points(s, b->pts + off * 3,
                         b->normals ? b->normals + off * 3 : NULL,
                         b->w, b->row1 - b->row0, b->p, b->out + off);
    } else {
        int need_n = b->p->comp != MC_COMP_NONE;
        float *row = malloc((size_t)b->w * 3 * sizeof(float) * (need_n ? 2 : 1));
        if (row) {
            float *nrm = need_n ? row + (size_t)b->w * 3 : NULL;
            for (int i = b->row0; i < b->row1; i++) {
                b->rowgen(b->rg_ud, i, b->w, row, nrm);
                mc_render_points(s, row, nrm, b->w, 1, b->p,
                                 b->out + (size_t)i * b->w);
            }
            free(row);
        }
        else memset(b->out + (size_t)b->row0 * b->w, 0,
                    (size_t)(b->row1 - b->row0) * b->w);
    }
    mc_sampler_free(s);
    return NULL;
}

// ls != NULL selects LOD-fallback mode (rowgen produces level-0 coords, each
// band builds an mc_lod_sampler at `lod`). Otherwise single-source mode.
static void render_bands_ex(const mc_sample_src *src,
                            const float *pts, const float *normals,
                            rowgen_fn rowgen, const void *rg_ud,
                            const mc_sample_lods *ls, int lod,
                            int w, int h, const mc_render_params *p,
                            uint8_t *out, int nthreads) {
    if (w <= 0 || h <= 0) return;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    pthread_t th[16];
    band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (band_t){ src, pts, normals, rowgen, rg_ud,
                              w, h, p, out, r0, r1, ls, lod };
        if (nthreads == 1) { band_main(&bands[nb]); continue; }
        if (pthread_create(&th[nb], NULL, band_main, &bands[nb]) != 0) {
            band_main(&bands[nb]);          // degrade to inline
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
}

static void render_bands(const mc_sample_src *src,
                         const float *pts, const float *normals,
                         rowgen_fn rowgen, const void *rg_ud,
                         int w, int h, const mc_render_params *p,
                         uint8_t *out, int nthreads) {
    render_bands_ex(src, pts, normals, rowgen, rg_ud, NULL, 0,
                    w, h, p, out, nthreads);
}

void mc_render_points_par(const mc_sample_src *src,
                          const float *pts, const float *normals,
                          int w, int h, const mc_render_params *p,
                          uint8_t *out, int nthreads) {
    render_bands(src, pts, normals, NULL, NULL, w, h, p, out, nthreads);
}

// dense-points renderer (one band-local row scratch fed from the caller's
// level-0 point grid) with LOD-fallback sampling at `lod`.
static void densepts_rowgen(const void *ud, int row, int w,
                            float *pts, float *normals) {
    (void)normals;
    const float *base = (const float *)ud;
    memcpy(pts, base + (size_t)row * w * 3, (size_t)w * 3 * sizeof(float));
}

void mc_render_points_par_lod(const mc_sample_lods *ls, int lod,
                              const float *ptsL0, int w, int h,
                              const mc_render_params *p,
                              uint8_t *out, int nthreads) {
    if (!ls || !ptsL0 || !out || w <= 0 || h <= 0) return;
    if (lod < 0) lod = 0;
    if (lod >= ls->nlods) lod = ls->nlods - 1;
    render_bands_ex(NULL, NULL, NULL, densepts_rowgen, ptsL0, ls, lod,
                    w, h, p, out, nthreads);
}

// ===========================================================================
// mc_colormap — window/level + colormap LUT (moved out of volume-cartographer).
// mc_render emits u8; this maps u8 -> ARGB32 for display. EVERY colormap is a
// static [256][3] grayscale->RGB table (palettes baked from the originals; tints
// generated as v*channel). mc stays dependency-free. Ids match the GUI strings.
// ===========================================================================
static const uint8_t MC_CMAP_VIRIDIS[256][3] = {
{68,1,84},{68,2,86},{69,4,87},{69,5,89},{70,7,90},{70,8,92},{70,10,93},{70,11,94},
{71,13,96},{71,14,97},{71,16,99},{71,17,100},{71,19,101},{72,20,103},{72,22,104},{72,23,105},
{72,24,106},{72,26,108},{72,27,109},{72,28,110},{72,29,111},{72,31,112},{72,32,113},{72,33,115},
{72,35,116},{72,36,117},{72,37,118},{72,38,119},{72,40,120},{72,41,121},{71,42,122},{71,44,122},
{71,45,123},{71,46,124},{71,47,125},{70,48,126},{70,50,126},{70,51,127},{70,52,128},{69,53,129},
{69,55,129},{69,56,130},{68,57,131},{68,58,131},{68,59,132},{67,61,132},{67,62,133},{66,63,133},
{66,64,134},{66,65,134},{65,66,135},{65,68,135},{64,69,136},{64,70,136},{63,71,136},{63,72,137},
{62,73,137},{62,74,137},{62,76,138},{61,77,138},{61,78,138},{60,79,138},{60,80,139},{59,81,139},
{59,82,139},{58,83,139},{58,84,140},{57,85,140},{57,86,140},{56,88,140},{56,89,140},{55,90,140},
{55,91,141},{54,92,141},{54,93,141},{53,94,141},{53,95,141},{52,96,141},{52,97,141},{51,98,141},
{51,99,141},{50,100,142},{50,101,142},{49,102,142},{49,103,142},{49,104,142},{48,105,142},{48,106,142},
{47,107,142},{47,108,142},{46,109,142},{46,110,142},{46,111,142},{45,112,142},{45,113,142},{44,113,142},
{44,114,142},{44,115,142},{43,116,142},{43,117,142},{42,118,142},{42,119,142},{42,120,142},{41,121,142},
{41,122,142},{41,123,142},{40,124,142},{40,125,142},{39,126,142},{39,127,142},{39,128,142},{38,129,142},
{38,130,142},{38,130,142},{37,131,142},{37,132,142},{37,133,142},{36,134,142},{36,135,142},{35,136,142},
{35,137,142},{35,138,141},{34,139,141},{34,140,141},{34,141,141},{33,142,141},{33,143,141},{33,144,141},
{33,145,140},{32,146,140},{32,146,140},{32,147,140},{31,148,140},{31,149,139},{31,150,139},{31,151,139},
{31,152,139},{31,153,138},{31,154,138},{30,155,138},{30,156,137},{30,157,137},{31,158,137},{31,159,136},
{31,160,136},{31,161,136},{31,161,135},{31,162,135},{32,163,134},{32,164,134},{33,165,133},{33,166,133},
{34,167,133},{34,168,132},{35,169,131},{36,170,131},{37,171,130},{37,172,130},{38,173,129},{39,173,129},
{40,174,128},{41,175,127},{42,176,127},{44,177,126},{45,178,125},{46,179,124},{47,180,124},{49,181,123},
{50,182,122},{52,182,121},{53,183,121},{55,184,120},{56,185,119},{58,186,118},{59,187,117},{61,188,116},
{63,188,115},{64,189,114},{66,190,113},{68,191,112},{70,192,111},{72,193,110},{74,193,109},{76,194,108},
{78,195,107},{80,196,106},{82,197,105},{84,197,104},{86,198,103},{88,199,101},{90,200,100},{92,200,99},
{94,201,98},{96,202,96},{99,203,95},{101,203,94},{103,204,92},{105,205,91},{108,205,90},{110,206,88},
{112,207,87},{115,208,86},{117,208,84},{119,209,83},{122,209,81},{124,210,80},{127,211,78},{129,211,77},
{132,212,75},{134,213,73},{137,213,72},{139,214,70},{142,214,69},{144,215,67},{147,215,65},{149,216,64},
{152,216,62},{155,217,60},{157,217,59},{160,218,57},{162,218,55},{165,219,54},{168,219,52},{170,220,50},
{173,220,48},{176,221,47},{178,221,45},{181,222,43},{184,222,41},{186,222,40},{189,223,38},{192,223,37},
{194,223,35},{197,224,33},{200,224,32},{202,225,31},{205,225,29},{208,225,28},{210,226,27},{213,226,26},
{216,226,25},{218,227,25},{221,227,24},{223,227,24},{226,228,24},{229,228,25},{231,228,25},{234,229,26},
{236,229,27},{239,229,28},{241,229,29},{244,230,30},{246,230,32},{248,230,33},{251,231,35},{253,231,37},
};
static const uint8_t MC_CMAP_MAGMA[256][3] = {
{0,0,4},{1,0,5},{1,1,6},{1,1,8},{2,1,9},{2,2,11},{2,2,13},{3,3,15},
{3,3,18},{4,4,20},{5,4,22},{6,5,24},{6,5,26},{7,6,28},{8,7,30},{9,7,32},
{10,8,34},{11,9,36},{12,9,38},{13,10,41},{14,11,43},{16,11,45},{17,12,47},{18,13,49},
{19,13,52},{20,14,54},{21,14,56},{22,15,59},{24,15,61},{25,16,63},{26,16,66},{28,16,68},
{29,17,71},{30,17,73},{32,17,75},{33,17,78},{34,17,80},{36,18,83},{37,18,85},{39,18,88},
{41,17,90},{42,17,92},{44,17,95},{45,17,97},{47,17,99},{49,17,101},{51,16,103},{52,16,105},
{54,16,107},{56,16,108},{57,15,110},{59,15,112},{61,15,113},{63,15,114},{64,15,116},{66,15,117},
{68,15,118},{69,16,119},{71,16,120},{73,16,120},{74,16,121},{76,17,122},{78,17,123},{79,18,123},
{81,18,124},{82,19,124},{84,19,125},{86,20,125},{87,21,126},{89,21,126},{90,22,126},{92,22,127},
{93,23,127},{95,24,127},{96,24,128},{98,25,128},{100,26,128},{101,26,128},{103,27,128},{104,28,129},
{106,28,129},{107,29,129},{109,29,129},{110,30,129},{112,31,129},{114,31,129},{115,32,129},{117,33,129},
{118,33,129},{120,34,129},{121,34,130},{123,35,130},{124,35,130},{126,36,130},{128,37,130},{129,37,129},
{131,38,129},{132,38,129},{134,39,129},{136,39,129},{137,40,129},{139,41,129},{140,41,129},{142,42,129},
{144,42,129},{145,43,129},{147,43,128},{148,44,128},{150,44,128},{152,45,128},{153,45,128},{155,46,127},
{156,46,127},{158,47,127},{160,47,127},{161,48,126},{163,48,126},{165,49,126},{166,49,125},{168,50,125},
{170,51,125},{171,51,124},{173,52,124},{174,52,123},{176,53,123},{178,53,123},{179,54,122},{181,54,122},
{183,55,121},{184,55,121},{186,56,120},{188,57,120},{189,57,119},{191,58,119},{192,58,118},{194,59,117},
{196,60,117},{197,60,116},{199,61,115},{200,62,115},{202,62,114},{204,63,113},{205,64,113},{207,64,112},
{208,65,111},{210,66,111},{211,67,110},{213,68,109},{214,69,108},{216,69,108},{217,70,107},{219,71,106},
{220,72,105},{222,73,104},{223,74,104},{224,76,103},{226,77,102},{227,78,101},{228,79,100},{229,80,100},
{231,82,99},{232,83,98},{233,84,98},{234,86,97},{235,87,96},{236,88,96},{237,90,95},{238,91,94},
{239,93,94},{240,95,94},{241,96,93},{242,98,93},{242,100,92},{243,101,92},{244,103,92},{244,105,92},
{245,107,92},{246,108,92},{246,110,92},{247,112,92},{247,114,92},{248,116,92},{248,118,92},{249,120,93},
{249,121,93},{249,123,93},{250,125,94},{250,127,94},{250,129,95},{251,131,95},{251,133,96},{251,135,97},
{252,137,97},{252,138,98},{252,140,99},{252,142,100},{252,144,101},{253,146,102},{253,148,103},{253,150,104},
{253,152,105},{253,154,106},{253,155,107},{254,157,108},{254,159,109},{254,161,110},{254,163,111},{254,165,113},
{254,167,114},{254,169,115},{254,170,116},{254,172,118},{254,174,119},{254,176,120},{254,178,122},{254,180,123},
{254,182,124},{254,183,126},{254,185,127},{254,187,129},{254,189,130},{254,191,132},{254,193,133},{254,194,135},
{254,196,136},{254,198,138},{254,200,140},{254,202,141},{254,204,143},{254,205,144},{254,207,146},{254,209,148},
{254,211,149},{254,213,151},{254,215,153},{254,216,154},{253,218,156},{253,220,158},{253,222,160},{253,224,161},
{253,226,163},{253,227,165},{253,229,167},{253,231,169},{253,233,170},{253,235,172},{252,236,174},{252,238,176},
{252,240,178},{252,242,180},{252,244,182},{252,246,184},{252,247,185},{252,249,187},{252,251,189},{252,253,191},
};
static const uint8_t MC_CMAP_FIRE[256][3] = {
{0,0,0},{2,0,0},{5,0,0},{8,0,0},{10,0,0},{12,0,0},{15,0,0},{18,0,0},
{20,0,0},{22,0,0},{25,0,0},{27,0,0},{30,0,0},{32,0,0},{35,0,0},{38,0,0},
{40,0,0},{42,0,0},{45,0,0},{48,0,0},{50,0,0},{52,0,0},{55,0,0},{57,0,0},
{60,0,0},{62,0,0},{65,0,0},{68,0,0},{70,0,0},{72,0,0},{75,0,0},{78,0,0},
{80,0,0},{82,0,0},{85,0,0},{88,0,0},{90,0,0},{92,0,0},{95,0,0},{98,0,0},
{100,0,0},{102,0,0},{105,0,0},{108,0,0},{110,0,0},{112,0,0},{115,0,0},{117,0,0},
{120,0,0},{122,0,0},{125,0,0},{128,0,0},{130,0,0},{132,0,0},{135,0,0},{138,0,0},
{140,0,0},{142,0,0},{145,0,0},{148,0,0},{150,0,0},{152,0,0},{155,0,0},{158,0,0},
{160,0,0},{162,0,0},{165,0,0},{168,0,0},{170,0,0},{172,0,0},{175,0,0},{178,0,0},
{180,0,0},{182,0,0},{185,0,0},{188,0,0},{190,0,0},{192,0,0},{195,0,0},{198,0,0},
{200,0,0},{202,0,0},{205,0,0},{208,0,0},{210,0,0},{212,0,0},{215,0,0},{218,0,0},
{220,0,0},{223,0,0},{225,0,0},{228,0,0},{230,0,0},{232,0,0},{235,0,0},{238,0,0},
{240,0,0},{243,0,0},{245,0,0},{248,0,0},{250,0,0},{252,0,0},{253,2,0},{254,4,0},
{254,6,0},{255,8,0},{255,10,0},{255,13,0},{255,15,0},{255,18,0},{255,20,0},{255,22,0},
{255,25,0},{255,28,0},{255,30,0},{255,32,0},{255,35,0},{255,38,0},{255,40,0},{255,42,0},
{255,45,0},{255,48,0},{255,50,0},{255,52,0},{255,55,0},{255,58,0},{255,60,0},{255,62,0},
{255,65,0},{255,68,0},{255,70,0},{255,72,0},{255,75,0},{255,78,0},{255,80,0},{255,82,0},
{255,85,0},{255,88,0},{255,90,0},{255,92,0},{255,95,0},{255,98,0},{255,100,0},{255,102,0},
{255,105,0},{255,108,0},{255,110,0},{255,112,0},{255,115,0},{255,118,0},{255,120,0},{255,122,0},
{255,125,0},{255,128,0},{255,130,0},{255,132,0},{255,135,0},{255,138,0},{255,140,0},{255,142,0},
{255,145,0},{255,148,0},{255,150,0},{255,152,0},{255,155,0},{255,158,0},{255,160,0},{255,162,0},
{255,165,0},{255,168,0},{255,170,0},{255,172,0},{255,175,0},{255,178,0},{255,180,0},{255,182,0},
{255,185,0},{255,188,0},{255,190,0},{255,192,0},{255,195,0},{255,198,0},{255,200,0},{255,202,0},
{255,205,0},{255,208,0},{255,210,0},{255,212,0},{255,215,0},{255,218,0},{255,220,0},{255,222,0},
{255,225,0},{255,228,0},{255,230,0},{255,232,0},{255,235,0},{255,238,0},{255,240,0},{255,242,0},
{255,245,0},{255,248,0},{255,250,0},{255,252,2},{255,253,5},{255,254,8},{255,255,11},{255,255,15},
{255,255,20},{255,255,25},{255,255,30},{255,255,35},{255,255,40},{255,255,45},{255,255,50},{255,255,55},
{255,255,60},{255,255,65},{255,255,70},{255,255,75},{255,255,80},{255,255,85},{255,255,90},{255,255,95},
{255,255,100},{255,255,105},{255,255,110},{255,255,115},{255,255,120},{255,255,125},{255,255,130},{255,255,135},
{255,255,140},{255,255,145},{255,255,150},{255,255,155},{255,255,160},{255,255,165},{255,255,170},{255,255,175},
{255,255,180},{255,255,185},{255,255,190},{255,255,195},{255,255,200},{255,255,205},{255,255,210},{255,255,215},
{255,255,220},{255,255,225},{255,255,230},{255,255,235},{255,255,240},{255,255,245},{255,255,250},{255,255,255},
};
static const uint8_t MC_CMAP_GRAY[256][3] = {
{0,0,0},{1,1,1},{2,2,2},{3,3,3},{4,4,4},{5,5,5},{6,6,6},{7,7,7},
{8,8,8},{9,9,9},{10,10,10},{11,11,11},{12,12,12},{13,13,13},{14,14,14},{15,15,15},
{16,16,16},{17,17,17},{18,18,18},{19,19,19},{20,20,20},{21,21,21},{22,22,22},{23,23,23},
{24,24,24},{25,25,25},{26,26,26},{27,27,27},{28,28,28},{29,29,29},{30,30,30},{31,31,31},
{32,32,32},{33,33,33},{34,34,34},{35,35,35},{36,36,36},{37,37,37},{38,38,38},{39,39,39},
{40,40,40},{41,41,41},{42,42,42},{43,43,43},{44,44,44},{45,45,45},{46,46,46},{47,47,47},
{48,48,48},{49,49,49},{50,50,50},{51,51,51},{52,52,52},{53,53,53},{54,54,54},{55,55,55},
{56,56,56},{57,57,57},{58,58,58},{59,59,59},{60,60,60},{61,61,61},{62,62,62},{63,63,63},
{64,64,64},{65,65,65},{66,66,66},{67,67,67},{68,68,68},{69,69,69},{70,70,70},{71,71,71},
{72,72,72},{73,73,73},{74,74,74},{75,75,75},{76,76,76},{77,77,77},{78,78,78},{79,79,79},
{80,80,80},{81,81,81},{82,82,82},{83,83,83},{84,84,84},{85,85,85},{86,86,86},{87,87,87},
{88,88,88},{89,89,89},{90,90,90},{91,91,91},{92,92,92},{93,93,93},{94,94,94},{95,95,95},
{96,96,96},{97,97,97},{98,98,98},{99,99,99},{100,100,100},{101,101,101},{102,102,102},{103,103,103},
{104,104,104},{105,105,105},{106,106,106},{107,107,107},{108,108,108},{109,109,109},{110,110,110},{111,111,111},
{112,112,112},{113,113,113},{114,114,114},{115,115,115},{116,116,116},{117,117,117},{118,118,118},{119,119,119},
{120,120,120},{121,121,121},{122,122,122},{123,123,123},{124,124,124},{125,125,125},{126,126,126},{127,127,127},
{128,128,128},{129,129,129},{130,130,130},{131,131,131},{132,132,132},{133,133,133},{134,134,134},{135,135,135},
{136,136,136},{137,137,137},{138,138,138},{139,139,139},{140,140,140},{141,141,141},{142,142,142},{143,143,143},
{144,144,144},{145,145,145},{146,146,146},{147,147,147},{148,148,148},{149,149,149},{150,150,150},{151,151,151},
{152,152,152},{153,153,153},{154,154,154},{155,155,155},{156,156,156},{157,157,157},{158,158,158},{159,159,159},
{160,160,160},{161,161,161},{162,162,162},{163,163,163},{164,164,164},{165,165,165},{166,166,166},{167,167,167},
{168,168,168},{169,169,169},{170,170,170},{171,171,171},{172,172,172},{173,173,173},{174,174,174},{175,175,175},
{176,176,176},{177,177,177},{178,178,178},{179,179,179},{180,180,180},{181,181,181},{182,182,182},{183,183,183},
{184,184,184},{185,185,185},{186,186,186},{187,187,187},{188,188,188},{189,189,189},{190,190,190},{191,191,191},
{192,192,192},{193,193,193},{194,194,194},{195,195,195},{196,196,196},{197,197,197},{198,198,198},{199,199,199},
{200,200,200},{201,201,201},{202,202,202},{203,203,203},{204,204,204},{205,205,205},{206,206,206},{207,207,207},
{208,208,208},{209,209,209},{210,210,210},{211,211,211},{212,212,212},{213,213,213},{214,214,214},{215,215,215},
{216,216,216},{217,217,217},{218,218,218},{219,219,219},{220,220,220},{221,221,221},{222,222,222},{223,223,223},
{224,224,224},{225,225,225},{226,226,226},{227,227,227},{228,228,228},{229,229,229},{230,230,230},{231,231,231},
{232,232,232},{233,233,233},{234,234,234},{235,235,235},{236,236,236},{237,237,237},{238,238,238},{239,239,239},
{240,240,240},{241,241,241},{242,242,242},{243,243,243},{244,244,244},{245,245,245},{246,246,246},{247,247,247},
{248,248,248},{249,249,249},{250,250,250},{251,251,251},{252,252,252},{253,253,253},{254,254,254},{255,255,255},
};
static const uint8_t MC_CMAP_RED[256][3] = {
{0,0,0},{1,0,0},{2,0,0},{3,0,0},{4,0,0},{5,0,0},{6,0,0},{7,0,0},
{8,0,0},{9,0,0},{10,0,0},{11,0,0},{12,0,0},{13,0,0},{14,0,0},{15,0,0},
{16,0,0},{17,0,0},{18,0,0},{19,0,0},{20,0,0},{21,0,0},{22,0,0},{23,0,0},
{24,0,0},{25,0,0},{26,0,0},{27,0,0},{28,0,0},{29,0,0},{30,0,0},{31,0,0},
{32,0,0},{33,0,0},{34,0,0},{35,0,0},{36,0,0},{37,0,0},{38,0,0},{39,0,0},
{40,0,0},{41,0,0},{42,0,0},{43,0,0},{44,0,0},{45,0,0},{46,0,0},{47,0,0},
{48,0,0},{49,0,0},{50,0,0},{51,0,0},{52,0,0},{53,0,0},{54,0,0},{55,0,0},
{56,0,0},{57,0,0},{58,0,0},{59,0,0},{60,0,0},{61,0,0},{62,0,0},{63,0,0},
{64,0,0},{65,0,0},{66,0,0},{67,0,0},{68,0,0},{69,0,0},{70,0,0},{71,0,0},
{72,0,0},{73,0,0},{74,0,0},{75,0,0},{76,0,0},{77,0,0},{78,0,0},{79,0,0},
{80,0,0},{81,0,0},{82,0,0},{83,0,0},{84,0,0},{85,0,0},{86,0,0},{87,0,0},
{88,0,0},{89,0,0},{90,0,0},{91,0,0},{92,0,0},{93,0,0},{94,0,0},{95,0,0},
{96,0,0},{97,0,0},{98,0,0},{99,0,0},{100,0,0},{101,0,0},{102,0,0},{103,0,0},
{104,0,0},{105,0,0},{106,0,0},{107,0,0},{108,0,0},{109,0,0},{110,0,0},{111,0,0},
{112,0,0},{113,0,0},{114,0,0},{115,0,0},{116,0,0},{117,0,0},{118,0,0},{119,0,0},
{120,0,0},{121,0,0},{122,0,0},{123,0,0},{124,0,0},{125,0,0},{126,0,0},{127,0,0},
{128,0,0},{129,0,0},{130,0,0},{131,0,0},{132,0,0},{133,0,0},{134,0,0},{135,0,0},
{136,0,0},{137,0,0},{138,0,0},{139,0,0},{140,0,0},{141,0,0},{142,0,0},{143,0,0},
{144,0,0},{145,0,0},{146,0,0},{147,0,0},{148,0,0},{149,0,0},{150,0,0},{151,0,0},
{152,0,0},{153,0,0},{154,0,0},{155,0,0},{156,0,0},{157,0,0},{158,0,0},{159,0,0},
{160,0,0},{161,0,0},{162,0,0},{163,0,0},{164,0,0},{165,0,0},{166,0,0},{167,0,0},
{168,0,0},{169,0,0},{170,0,0},{171,0,0},{172,0,0},{173,0,0},{174,0,0},{175,0,0},
{176,0,0},{177,0,0},{178,0,0},{179,0,0},{180,0,0},{181,0,0},{182,0,0},{183,0,0},
{184,0,0},{185,0,0},{186,0,0},{187,0,0},{188,0,0},{189,0,0},{190,0,0},{191,0,0},
{192,0,0},{193,0,0},{194,0,0},{195,0,0},{196,0,0},{197,0,0},{198,0,0},{199,0,0},
{200,0,0},{201,0,0},{202,0,0},{203,0,0},{204,0,0},{205,0,0},{206,0,0},{207,0,0},
{208,0,0},{209,0,0},{210,0,0},{211,0,0},{212,0,0},{213,0,0},{214,0,0},{215,0,0},
{216,0,0},{217,0,0},{218,0,0},{219,0,0},{220,0,0},{221,0,0},{222,0,0},{223,0,0},
{224,0,0},{225,0,0},{226,0,0},{227,0,0},{228,0,0},{229,0,0},{230,0,0},{231,0,0},
{232,0,0},{233,0,0},{234,0,0},{235,0,0},{236,0,0},{237,0,0},{238,0,0},{239,0,0},
{240,0,0},{241,0,0},{242,0,0},{243,0,0},{244,0,0},{245,0,0},{246,0,0},{247,0,0},
{248,0,0},{249,0,0},{250,0,0},{251,0,0},{252,0,0},{253,0,0},{254,0,0},{255,0,0},
};
static const uint8_t MC_CMAP_GREEN[256][3] = {
{0,0,0},{0,1,0},{0,2,0},{0,3,0},{0,4,0},{0,5,0},{0,6,0},{0,7,0},
{0,8,0},{0,9,0},{0,10,0},{0,11,0},{0,12,0},{0,13,0},{0,14,0},{0,15,0},
{0,16,0},{0,17,0},{0,18,0},{0,19,0},{0,20,0},{0,21,0},{0,22,0},{0,23,0},
{0,24,0},{0,25,0},{0,26,0},{0,27,0},{0,28,0},{0,29,0},{0,30,0},{0,31,0},
{0,32,0},{0,33,0},{0,34,0},{0,35,0},{0,36,0},{0,37,0},{0,38,0},{0,39,0},
{0,40,0},{0,41,0},{0,42,0},{0,43,0},{0,44,0},{0,45,0},{0,46,0},{0,47,0},
{0,48,0},{0,49,0},{0,50,0},{0,51,0},{0,52,0},{0,53,0},{0,54,0},{0,55,0},
{0,56,0},{0,57,0},{0,58,0},{0,59,0},{0,60,0},{0,61,0},{0,62,0},{0,63,0},
{0,64,0},{0,65,0},{0,66,0},{0,67,0},{0,68,0},{0,69,0},{0,70,0},{0,71,0},
{0,72,0},{0,73,0},{0,74,0},{0,75,0},{0,76,0},{0,77,0},{0,78,0},{0,79,0},
{0,80,0},{0,81,0},{0,82,0},{0,83,0},{0,84,0},{0,85,0},{0,86,0},{0,87,0},
{0,88,0},{0,89,0},{0,90,0},{0,91,0},{0,92,0},{0,93,0},{0,94,0},{0,95,0},
{0,96,0},{0,97,0},{0,98,0},{0,99,0},{0,100,0},{0,101,0},{0,102,0},{0,103,0},
{0,104,0},{0,105,0},{0,106,0},{0,107,0},{0,108,0},{0,109,0},{0,110,0},{0,111,0},
{0,112,0},{0,113,0},{0,114,0},{0,115,0},{0,116,0},{0,117,0},{0,118,0},{0,119,0},
{0,120,0},{0,121,0},{0,122,0},{0,123,0},{0,124,0},{0,125,0},{0,126,0},{0,127,0},
{0,128,0},{0,129,0},{0,130,0},{0,131,0},{0,132,0},{0,133,0},{0,134,0},{0,135,0},
{0,136,0},{0,137,0},{0,138,0},{0,139,0},{0,140,0},{0,141,0},{0,142,0},{0,143,0},
{0,144,0},{0,145,0},{0,146,0},{0,147,0},{0,148,0},{0,149,0},{0,150,0},{0,151,0},
{0,152,0},{0,153,0},{0,154,0},{0,155,0},{0,156,0},{0,157,0},{0,158,0},{0,159,0},
{0,160,0},{0,161,0},{0,162,0},{0,163,0},{0,164,0},{0,165,0},{0,166,0},{0,167,0},
{0,168,0},{0,169,0},{0,170,0},{0,171,0},{0,172,0},{0,173,0},{0,174,0},{0,175,0},
{0,176,0},{0,177,0},{0,178,0},{0,179,0},{0,180,0},{0,181,0},{0,182,0},{0,183,0},
{0,184,0},{0,185,0},{0,186,0},{0,187,0},{0,188,0},{0,189,0},{0,190,0},{0,191,0},
{0,192,0},{0,193,0},{0,194,0},{0,195,0},{0,196,0},{0,197,0},{0,198,0},{0,199,0},
{0,200,0},{0,201,0},{0,202,0},{0,203,0},{0,204,0},{0,205,0},{0,206,0},{0,207,0},
{0,208,0},{0,209,0},{0,210,0},{0,211,0},{0,212,0},{0,213,0},{0,214,0},{0,215,0},
{0,216,0},{0,217,0},{0,218,0},{0,219,0},{0,220,0},{0,221,0},{0,222,0},{0,223,0},
{0,224,0},{0,225,0},{0,226,0},{0,227,0},{0,228,0},{0,229,0},{0,230,0},{0,231,0},
{0,232,0},{0,233,0},{0,234,0},{0,235,0},{0,236,0},{0,237,0},{0,238,0},{0,239,0},
{0,240,0},{0,241,0},{0,242,0},{0,243,0},{0,244,0},{0,245,0},{0,246,0},{0,247,0},
{0,248,0},{0,249,0},{0,250,0},{0,251,0},{0,252,0},{0,253,0},{0,254,0},{0,255,0},
};
static const uint8_t MC_CMAP_BLUE[256][3] = {
{0,0,0},{0,0,1},{0,0,2},{0,0,3},{0,0,4},{0,0,5},{0,0,6},{0,0,7},
{0,0,8},{0,0,9},{0,0,10},{0,0,11},{0,0,12},{0,0,13},{0,0,14},{0,0,15},
{0,0,16},{0,0,17},{0,0,18},{0,0,19},{0,0,20},{0,0,21},{0,0,22},{0,0,23},
{0,0,24},{0,0,25},{0,0,26},{0,0,27},{0,0,28},{0,0,29},{0,0,30},{0,0,31},
{0,0,32},{0,0,33},{0,0,34},{0,0,35},{0,0,36},{0,0,37},{0,0,38},{0,0,39},
{0,0,40},{0,0,41},{0,0,42},{0,0,43},{0,0,44},{0,0,45},{0,0,46},{0,0,47},
{0,0,48},{0,0,49},{0,0,50},{0,0,51},{0,0,52},{0,0,53},{0,0,54},{0,0,55},
{0,0,56},{0,0,57},{0,0,58},{0,0,59},{0,0,60},{0,0,61},{0,0,62},{0,0,63},
{0,0,64},{0,0,65},{0,0,66},{0,0,67},{0,0,68},{0,0,69},{0,0,70},{0,0,71},
{0,0,72},{0,0,73},{0,0,74},{0,0,75},{0,0,76},{0,0,77},{0,0,78},{0,0,79},
{0,0,80},{0,0,81},{0,0,82},{0,0,83},{0,0,84},{0,0,85},{0,0,86},{0,0,87},
{0,0,88},{0,0,89},{0,0,90},{0,0,91},{0,0,92},{0,0,93},{0,0,94},{0,0,95},
{0,0,96},{0,0,97},{0,0,98},{0,0,99},{0,0,100},{0,0,101},{0,0,102},{0,0,103},
{0,0,104},{0,0,105},{0,0,106},{0,0,107},{0,0,108},{0,0,109},{0,0,110},{0,0,111},
{0,0,112},{0,0,113},{0,0,114},{0,0,115},{0,0,116},{0,0,117},{0,0,118},{0,0,119},
{0,0,120},{0,0,121},{0,0,122},{0,0,123},{0,0,124},{0,0,125},{0,0,126},{0,0,127},
{0,0,128},{0,0,129},{0,0,130},{0,0,131},{0,0,132},{0,0,133},{0,0,134},{0,0,135},
{0,0,136},{0,0,137},{0,0,138},{0,0,139},{0,0,140},{0,0,141},{0,0,142},{0,0,143},
{0,0,144},{0,0,145},{0,0,146},{0,0,147},{0,0,148},{0,0,149},{0,0,150},{0,0,151},
{0,0,152},{0,0,153},{0,0,154},{0,0,155},{0,0,156},{0,0,157},{0,0,158},{0,0,159},
{0,0,160},{0,0,161},{0,0,162},{0,0,163},{0,0,164},{0,0,165},{0,0,166},{0,0,167},
{0,0,168},{0,0,169},{0,0,170},{0,0,171},{0,0,172},{0,0,173},{0,0,174},{0,0,175},
{0,0,176},{0,0,177},{0,0,178},{0,0,179},{0,0,180},{0,0,181},{0,0,182},{0,0,183},
{0,0,184},{0,0,185},{0,0,186},{0,0,187},{0,0,188},{0,0,189},{0,0,190},{0,0,191},
{0,0,192},{0,0,193},{0,0,194},{0,0,195},{0,0,196},{0,0,197},{0,0,198},{0,0,199},
{0,0,200},{0,0,201},{0,0,202},{0,0,203},{0,0,204},{0,0,205},{0,0,206},{0,0,207},
{0,0,208},{0,0,209},{0,0,210},{0,0,211},{0,0,212},{0,0,213},{0,0,214},{0,0,215},
{0,0,216},{0,0,217},{0,0,218},{0,0,219},{0,0,220},{0,0,221},{0,0,222},{0,0,223},
{0,0,224},{0,0,225},{0,0,226},{0,0,227},{0,0,228},{0,0,229},{0,0,230},{0,0,231},
{0,0,232},{0,0,233},{0,0,234},{0,0,235},{0,0,236},{0,0,237},{0,0,238},{0,0,239},
{0,0,240},{0,0,241},{0,0,242},{0,0,243},{0,0,244},{0,0,245},{0,0,246},{0,0,247},
{0,0,248},{0,0,249},{0,0,250},{0,0,251},{0,0,252},{0,0,253},{0,0,254},{0,0,255},
};
static const uint8_t MC_CMAP_CYAN[256][3] = {
{0,0,0},{0,1,1},{0,2,2},{0,3,3},{0,4,4},{0,5,5},{0,6,6},{0,7,7},
{0,8,8},{0,9,9},{0,10,10},{0,11,11},{0,12,12},{0,13,13},{0,14,14},{0,15,15},
{0,16,16},{0,17,17},{0,18,18},{0,19,19},{0,20,20},{0,21,21},{0,22,22},{0,23,23},
{0,24,24},{0,25,25},{0,26,26},{0,27,27},{0,28,28},{0,29,29},{0,30,30},{0,31,31},
{0,32,32},{0,33,33},{0,34,34},{0,35,35},{0,36,36},{0,37,37},{0,38,38},{0,39,39},
{0,40,40},{0,41,41},{0,42,42},{0,43,43},{0,44,44},{0,45,45},{0,46,46},{0,47,47},
{0,48,48},{0,49,49},{0,50,50},{0,51,51},{0,52,52},{0,53,53},{0,54,54},{0,55,55},
{0,56,56},{0,57,57},{0,58,58},{0,59,59},{0,60,60},{0,61,61},{0,62,62},{0,63,63},
{0,64,64},{0,65,65},{0,66,66},{0,67,67},{0,68,68},{0,69,69},{0,70,70},{0,71,71},
{0,72,72},{0,73,73},{0,74,74},{0,75,75},{0,76,76},{0,77,77},{0,78,78},{0,79,79},
{0,80,80},{0,81,81},{0,82,82},{0,83,83},{0,84,84},{0,85,85},{0,86,86},{0,87,87},
{0,88,88},{0,89,89},{0,90,90},{0,91,91},{0,92,92},{0,93,93},{0,94,94},{0,95,95},
{0,96,96},{0,97,97},{0,98,98},{0,99,99},{0,100,100},{0,101,101},{0,102,102},{0,103,103},
{0,104,104},{0,105,105},{0,106,106},{0,107,107},{0,108,108},{0,109,109},{0,110,110},{0,111,111},
{0,112,112},{0,113,113},{0,114,114},{0,115,115},{0,116,116},{0,117,117},{0,118,118},{0,119,119},
{0,120,120},{0,121,121},{0,122,122},{0,123,123},{0,124,124},{0,125,125},{0,126,126},{0,127,127},
{0,128,128},{0,129,129},{0,130,130},{0,131,131},{0,132,132},{0,133,133},{0,134,134},{0,135,135},
{0,136,136},{0,137,137},{0,138,138},{0,139,139},{0,140,140},{0,141,141},{0,142,142},{0,143,143},
{0,144,144},{0,145,145},{0,146,146},{0,147,147},{0,148,148},{0,149,149},{0,150,150},{0,151,151},
{0,152,152},{0,153,153},{0,154,154},{0,155,155},{0,156,156},{0,157,157},{0,158,158},{0,159,159},
{0,160,160},{0,161,161},{0,162,162},{0,163,163},{0,164,164},{0,165,165},{0,166,166},{0,167,167},
{0,168,168},{0,169,169},{0,170,170},{0,171,171},{0,172,172},{0,173,173},{0,174,174},{0,175,175},
{0,176,176},{0,177,177},{0,178,178},{0,179,179},{0,180,180},{0,181,181},{0,182,182},{0,183,183},
{0,184,184},{0,185,185},{0,186,186},{0,187,187},{0,188,188},{0,189,189},{0,190,190},{0,191,191},
{0,192,192},{0,193,193},{0,194,194},{0,195,195},{0,196,196},{0,197,197},{0,198,198},{0,199,199},
{0,200,200},{0,201,201},{0,202,202},{0,203,203},{0,204,204},{0,205,205},{0,206,206},{0,207,207},
{0,208,208},{0,209,209},{0,210,210},{0,211,211},{0,212,212},{0,213,213},{0,214,214},{0,215,215},
{0,216,216},{0,217,217},{0,218,218},{0,219,219},{0,220,220},{0,221,221},{0,222,222},{0,223,223},
{0,224,224},{0,225,225},{0,226,226},{0,227,227},{0,228,228},{0,229,229},{0,230,230},{0,231,231},
{0,232,232},{0,233,233},{0,234,234},{0,235,235},{0,236,236},{0,237,237},{0,238,238},{0,239,239},
{0,240,240},{0,241,241},{0,242,242},{0,243,243},{0,244,244},{0,245,245},{0,246,246},{0,247,247},
{0,248,248},{0,249,249},{0,250,250},{0,251,251},{0,252,252},{0,253,253},{0,254,254},{0,255,255},
};
static const uint8_t MC_CMAP_MAGENTA[256][3] = {
{0,0,0},{1,0,1},{2,0,2},{3,0,3},{4,0,4},{5,0,5},{6,0,6},{7,0,7},
{8,0,8},{9,0,9},{10,0,10},{11,0,11},{12,0,12},{13,0,13},{14,0,14},{15,0,15},
{16,0,16},{17,0,17},{18,0,18},{19,0,19},{20,0,20},{21,0,21},{22,0,22},{23,0,23},
{24,0,24},{25,0,25},{26,0,26},{27,0,27},{28,0,28},{29,0,29},{30,0,30},{31,0,31},
{32,0,32},{33,0,33},{34,0,34},{35,0,35},{36,0,36},{37,0,37},{38,0,38},{39,0,39},
{40,0,40},{41,0,41},{42,0,42},{43,0,43},{44,0,44},{45,0,45},{46,0,46},{47,0,47},
{48,0,48},{49,0,49},{50,0,50},{51,0,51},{52,0,52},{53,0,53},{54,0,54},{55,0,55},
{56,0,56},{57,0,57},{58,0,58},{59,0,59},{60,0,60},{61,0,61},{62,0,62},{63,0,63},
{64,0,64},{65,0,65},{66,0,66},{67,0,67},{68,0,68},{69,0,69},{70,0,70},{71,0,71},
{72,0,72},{73,0,73},{74,0,74},{75,0,75},{76,0,76},{77,0,77},{78,0,78},{79,0,79},
{80,0,80},{81,0,81},{82,0,82},{83,0,83},{84,0,84},{85,0,85},{86,0,86},{87,0,87},
{88,0,88},{89,0,89},{90,0,90},{91,0,91},{92,0,92},{93,0,93},{94,0,94},{95,0,95},
{96,0,96},{97,0,97},{98,0,98},{99,0,99},{100,0,100},{101,0,101},{102,0,102},{103,0,103},
{104,0,104},{105,0,105},{106,0,106},{107,0,107},{108,0,108},{109,0,109},{110,0,110},{111,0,111},
{112,0,112},{113,0,113},{114,0,114},{115,0,115},{116,0,116},{117,0,117},{118,0,118},{119,0,119},
{120,0,120},{121,0,121},{122,0,122},{123,0,123},{124,0,124},{125,0,125},{126,0,126},{127,0,127},
{128,0,128},{129,0,129},{130,0,130},{131,0,131},{132,0,132},{133,0,133},{134,0,134},{135,0,135},
{136,0,136},{137,0,137},{138,0,138},{139,0,139},{140,0,140},{141,0,141},{142,0,142},{143,0,143},
{144,0,144},{145,0,145},{146,0,146},{147,0,147},{148,0,148},{149,0,149},{150,0,150},{151,0,151},
{152,0,152},{153,0,153},{154,0,154},{155,0,155},{156,0,156},{157,0,157},{158,0,158},{159,0,159},
{160,0,160},{161,0,161},{162,0,162},{163,0,163},{164,0,164},{165,0,165},{166,0,166},{167,0,167},
{168,0,168},{169,0,169},{170,0,170},{171,0,171},{172,0,172},{173,0,173},{174,0,174},{175,0,175},
{176,0,176},{177,0,177},{178,0,178},{179,0,179},{180,0,180},{181,0,181},{182,0,182},{183,0,183},
{184,0,184},{185,0,185},{186,0,186},{187,0,187},{188,0,188},{189,0,189},{190,0,190},{191,0,191},
{192,0,192},{193,0,193},{194,0,194},{195,0,195},{196,0,196},{197,0,197},{198,0,198},{199,0,199},
{200,0,200},{201,0,201},{202,0,202},{203,0,203},{204,0,204},{205,0,205},{206,0,206},{207,0,207},
{208,0,208},{209,0,209},{210,0,210},{211,0,211},{212,0,212},{213,0,213},{214,0,214},{215,0,215},
{216,0,216},{217,0,217},{218,0,218},{219,0,219},{220,0,220},{221,0,221},{222,0,222},{223,0,223},
{224,0,224},{225,0,225},{226,0,226},{227,0,227},{228,0,228},{229,0,229},{230,0,230},{231,0,231},
{232,0,232},{233,0,233},{234,0,234},{235,0,235},{236,0,236},{237,0,237},{238,0,238},{239,0,239},
{240,0,240},{241,0,241},{242,0,242},{243,0,243},{244,0,244},{245,0,245},{246,0,246},{247,0,247},
{248,0,248},{249,0,249},{250,0,250},{251,0,251},{252,0,252},{253,0,253},{254,0,254},{255,0,255},
};

// id -> the static palette table. All maps are a uniform [256][3] lookup.
static const uint8_t (*mc_cmap_table(int id))[3] {
    switch(id){
        case 1: return MC_CMAP_VIRIDIS;
        case 2: return MC_CMAP_MAGMA;
        case 3: return MC_CMAP_FIRE;
        case 4: return MC_CMAP_RED;
        case 5: return MC_CMAP_GREEN;
        case 6: return MC_CMAP_BLUE;
        case 7: return MC_CMAP_CYAN;
        case 8: return MC_CMAP_MAGENTA;
        default: return MC_CMAP_GRAY;
    }
}
int mc_colormap_id(const char *name){
    if(!name||!*name) return 0;
    if(!strcmp(name,"viridis")) return 1;
    if(!strcmp(name,"magma"))   return 2;
    if(!strcmp(name,"fire"))    return 3;
    if(!strcmp(name,"red"))     return 4;
    if(!strcmp(name,"green"))   return 5;
    if(!strcmp(name,"blue"))    return 6;
    if(!strcmp(name,"cyan"))    return 7;
    if(!strcmp(name,"magenta")) return 8;
    return 0;
}
// 256-entry ARGB32 LUT: window/level ramp then the colormap table. Matches VC3D's
// old buildWindowLevelColormapLut (lut[0]=opaque black for non-gray maps).
void mc_colormap_lut(uint32_t lut[256], float win_low, float win_high, int cmap_id){
    int lo=(int)(win_low<0?0:win_low>255?255:win_low);
    int hi=(int)(win_high<lo+1?lo+1:win_high>255?255:win_high);
    float span=(hi-lo)>1?(float)(hi-lo):1.0f;
    const uint8_t (*pal)[3]=mc_cmap_table(cmap_id);
    for(int i=0;i<256;++i){
        float g=((float)i-(float)lo)/span*255.0f;
        uint8_t v=(uint8_t)(g<0?0:g>255?255:g);
        lut[i]=0xFF000000u|((uint32_t)pal[v][0]<<16)|((uint32_t)pal[v][1]<<8)|(uint32_t)pal[v][2];
    }
    // A composite value of 0 means "no material" (air / block edge / air-bitmask),
    // independent of how it was zeroed -> always black. The colormap (incl. window/
    // level for grayscale) applies to real data, 1..255, only.
    lut[0]=0xFF000000u;
}
// Apply a 256-ARGB LUT to a w*h u8 image -> ARGB32 (out_stride in pixels).
void mc_colormap_apply(const uint8_t *vals, int w, int h, const uint32_t lut[256],
                       uint32_t *out, int out_stride){
    for(int y=0;y<h;++y){
        const uint8_t *s=vals+(size_t)y*w; uint32_t *d=out+(size_t)y*out_stride;
        for(int x=0;x<w;++x) d[x]=lut[s[x]];
    }
}


// ---------------------------------------------------------------------------
// plane surface
// ---------------------------------------------------------------------------
static inline void v3_norm(float *v) {
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-12f) { v[0] /= l; v[1] /= l; v[2] /= l; }
}
static inline void v3_cross(const float *a, const float *b, float *o) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

void mc_plane_basis(mc_plane *pl) {
    float *n = pl->normal;
    v3_norm(n);
    // pick the world axis least aligned with n as the seed
    float az = fabsf(n[0]), ay = fabsf(n[1]), ax = fabsf(n[2]);
    float e[3] = {0, 0, 0};
    if (az <= ay && az <= ax) e[0] = 1.0f;
    else if (ay <= ax)        e[1] = 1.0f;
    else                      e[2] = 1.0f;
    v3_cross(n, e, pl->u); v3_norm(pl->u);
    v3_cross(n, pl->u, pl->v); v3_norm(pl->v);
}

void mc_plane_gen(const mc_plane *pl, int w, int h, float scale,
                  float *pts, float *normals) {
    float cx = (float)w * 0.5f, cy = (float)h * 0.5f;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            float du = ((float)j - cx) * scale;
            float dv = ((float)i - cy) * scale;
            float *P = pts + ((size_t)i * w + j) * 3;
            for (int k = 0; k < 3; k++)
                P[k] = pl->origin[k] + du * pl->u[k] + dv * pl->v[k];
            if (normals) {
                float *N = normals + ((size_t)i * w + j) * 3;
                N[0] = pl->normal[0]; N[1] = pl->normal[1]; N[2] = pl->normal[2];
            }
        }
}

// ---------------------------------------------------------------------------
// quad surface
// ---------------------------------------------------------------------------
static inline int qvalid(const float *p) {
    return !(p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f) &&
           p[0] == p[0] && p[1] == p[1] && p[2] == p[2];
}

void mc_quad_gen(const mc_quad *q, float x0, float y0, float step,
                 int w, int h, float *pts, float *normals) {
    const int gw = q->gw, gh = q->gh;
    for (int i = 0; i < h; i++) {
        float *Prow = pts + (size_t)i * w * 3;
        float *Nrow = normals ? normals + (size_t)i * w * 3 : NULL;
        float gy = y0 + (float)i * step;
        // row-invalid fast fill
        if (!(gy >= 0.0f) || gy > (float)(gh - 1)) {
            for (int j = 0; j < w; j++) {
                Prow[j * 3] = Prow[j * 3 + 1] = Prow[j * 3 + 2] = -1.0f;
                if (Nrow) Nrow[j * 3] = Nrow[j * 3 + 1] = Nrow[j * 3 + 2] = 0.0f;
            }
            continue;
        }
        int y0i = (int)gy;
        if (y0i > gh - 2) y0i = gh - 2;
        if (y0i < 0) y0i = 0;               // gh == 1
        float fy = gy - (float)y0i;
        const float *r0 = q->grid + (size_t)y0i * gw * 3;
        const float *r1 = q->grid + (size_t)(y0i + (gh > 1)) * gw * 3;

        // per-cell state, reloaded only when the pixel crosses a grid cell
        int cell = -2, cell_ok = 0;
        float A[3], B[3], du[3], dv0[3], dv1[3];
        for (int j = 0; j < w; j++) {
            float *P = Prow + j * 3;
            float *N = Nrow ? Nrow + j * 3 : NULL;
            P[0] = P[1] = P[2] = -1.0f;
            if (N) N[0] = N[1] = N[2] = 0.0f;
            float gx = x0 + (float)j * step;
            if (!(gx >= 0.0f) || gx > (float)(gw - 1)) continue;
            int x0i = (int)gx;
            if (x0i > gw - 2) x0i = gw - 2;
            if (x0i < 0) x0i = 0;           // gw == 1
            if (x0i != cell) {
                cell = x0i;
                const float *p00 = r0 + (size_t)x0i * 3;
                const float *p01 = r0 + (size_t)(x0i + (gw > 1)) * 3;
                const float *p10 = r1 + (size_t)x0i * 3;
                const float *p11 = r1 + (size_t)(x0i + (gw > 1)) * 3;
                cell_ok = qvalid(p00) && qvalid(p01) &&
                          qvalid(p10) && qvalid(p11);
                if (cell_ok)
                    for (int k = 0; k < 3; k++) {
                        // y-lerped column endpoints: P = A + (B-A)*fx
                        A[k] = p00[k] + (p10[k] - p00[k]) * fy;
                        B[k] = p01[k] + (p11[k] - p01[k]) * fy;
                        // bilinear tangents (du constant per cell row)
                        du[k] = (p01[k] - p00[k]) * (1.0f - fy) +
                                (p11[k] - p10[k]) * fy;
                        dv0[k] = p10[k] - p00[k];
                        dv1[k] = p11[k] - p01[k];
                    }
            }
            if (!cell_ok) continue;
            float fx = gx - (float)x0i;
            P[0] = A[0] + (B[0] - A[0]) * fx;
            P[1] = A[1] + (B[1] - A[1]) * fx;
            P[2] = A[2] + (B[2] - A[2]) * fx;
            if (N) {
                float dv[3] = {
                    dv0[0] + (dv1[0] - dv0[0]) * fx,
                    dv0[1] + (dv1[1] - dv0[1]) * fx,
                    dv0[2] + (dv1[2] - dv0[2]) * fx,
                };
                v3_cross(du, dv, N);
                v3_norm(N);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// conveniences (row-strip rendering, no W*H grid)
// ---------------------------------------------------------------------------
typedef struct {
    mc_plane pl;                // normal pre-normalized
    float scale, cx, cy;
} plane_rg;

static void plane_rowgen(const void *ud, int row, int w,
                         float *pts, float *normals) {
    const plane_rg *g = ud;
    float dv = ((float)row - g->cy) * g->scale;
    float base[3], du[3];
    for (int k = 0; k < 3; k++) {
        base[k] = g->pl.origin[k] + dv * g->pl.v[k]
                  - g->cx * g->scale * g->pl.u[k];
        du[k] = g->scale * g->pl.u[k];
    }
    for (int j = 0; j < w; j++) {
        pts[j * 3 + 0] = base[0] + (float)j * du[0];
        pts[j * 3 + 1] = base[1] + (float)j * du[1];
        pts[j * 3 + 2] = base[2] + (float)j * du[2];
    }
    if (normals)
        for (int j = 0; j < w; j++) {
            normals[j * 3 + 0] = g->pl.normal[0];
            normals[j * 3 + 1] = g->pl.normal[1];
            normals[j * 3 + 2] = g->pl.normal[2];
        }
}

int mc_render_plane(const mc_sample_src *src, const mc_plane *pl,
                    int w, int h, float scale,
                    const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    plane_rg g = { *pl, scale, (float)w * 0.5f, (float)h * 0.5f };
    v3_norm(g.pl.normal);
    render_bands(src, NULL, NULL, plane_rowgen, &g, w, h, p, out, nthreads);
    return 0;
}

typedef struct {
    const mc_quad *q;
    float x0, y0, step;
} quad_rg;

static void quad_rowgen(const void *ud, int row, int w,
                        float *pts, float *normals) {
    const quad_rg *g = ud;
    mc_quad_gen(g->q, g->x0, g->y0 + (float)row * g->step, g->step,
                w, 1, pts, normals);
}

int mc_render_quad(const mc_sample_src *src, const mc_quad *q,
                   float x0, float y0, float step, int w, int h,
                   const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    quad_rg g = { q, x0, y0, step };
    render_bands(src, NULL, NULL, quad_rowgen, &g, w, h, p, out, nthreads);
    return 0;
}

// ---------------------------------------------------------------------------
// screen-space ambient occlusion over a MC_COMP_DEPTH image
// ---------------------------------------------------------------------------
// 12 fixed disk taps (two rings of 6, inner ring rotated 30deg); a pixel is
// occluded by neighbors that sit nearer the camera (smaller depth) by more
// than a small bias, with a range falloff so a distant other sheet doesn't
// read as an occluder.
// Stroke-scale local-contrast band-pass: img += gain * (img - blur(img)).
// 3-pass separable box blur of width ~sigma approximates a Gaussian; zero
// pixels (invalid/no-data) are excluded from the blur so holes don't bleed
// darkness into their surroundings.
void mc_image_dog(uint8_t *img, int w, int h, float sigma_px, float gain) {
    if (!img || w <= 0 || h <= 0 || sigma_px < 1.0f || gain == 0.0f) return;
    const size_t n = (size_t)w * h;
    // box radius for 3 passes ~= gaussian sigma: r = sigma * sqrt(12/3)/2
    int r = (int)(sigma_px + 0.5f);
    if (r < 1) r = 1;
    if (r > (w < h ? w : h) / 2) r = (w < h ? w : h) / 2;
    float *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b);
    float *wa = malloc(n * sizeof *wa), *wb = malloc(n * sizeof *wb);
    if (!a || !b || !wa || !wb) { free(a); free(b); free(wa); free(wb); return; }
    for (size_t i = 0; i < n; i++) { a[i] = (float)img[i]; wa[i] = img[i] ? 1.0f : 0.0f; }
    for (size_t i = 0; i < n; i++) a[i] *= wa[i];
    for (int pass = 0; pass < 3; pass++) {
        // horizontal box
        for (int y = 0; y < h; y++) {
            const float *src = a + (size_t)y * w, *sw = wa + (size_t)y * w;
            float *dst = b + (size_t)y * w, *dw = wb + (size_t)y * w;
            float sum = 0, wsum = 0;
            for (int x = 0; x < r && x < w; x++) { sum += src[x]; wsum += sw[x]; }
            for (int x = 0; x < w; x++) {
                if (x + r < w) { sum += src[x + r]; wsum += sw[x + r]; }
                if (x - r - 1 >= 0) { sum -= src[x - r - 1]; wsum -= sw[x - r - 1]; }
                dst[x] = sum; dw[x] = wsum;
            }
        }
        // vertical box (b -> a)
        for (int x = 0; x < w; x++) {
            float sum = 0, wsum = 0;
            for (int y = 0; y < r && y < h; y++) { sum += b[(size_t)y * w + x]; wsum += wb[(size_t)y * w + x]; }
            for (int y = 0; y < h; y++) {
                if (y + r < h) { sum += b[(size_t)(y + r) * w + x]; wsum += wb[(size_t)(y + r) * w + x]; }
                if (y - r - 1 >= 0) { sum -= b[(size_t)(y - r - 1) * w + x]; wsum -= wb[(size_t)(y - r - 1) * w + x]; }
                a[(size_t)y * w + x] = sum;
                wa[(size_t)y * w + x] = wsum;
            }
        }
        // normalize between passes so weights stay box-counts
        if (pass < 2) {
            for (size_t i = 0; i < n; i++) {
                float ww = wa[i];
                a[i] = ww > 0.0f ? a[i] / ww * (img[i] ? 1.0f : 0.0f) : 0.0f;
                wa[i] = img[i] ? 1.0f : 0.0f;
            }
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (!img[i]) continue;                  // keep holes black
        float blur = wa[i] > 0.0f ? a[i] / wa[i] : (float)img[i];
        float v = (float)img[i] + gain * ((float)img[i] - blur);
        img[i] = (uint8_t)(v < 1.0f ? 1.0f : v > 255.0f ? 255.0f : v);
    }
    free(a); free(b); free(wa); free(wb);
}

void mc_image_ssao(const uint8_t *depth, int w, int h,
                   float radius_px, float strength, uint8_t *img) {
    if (!depth || !img || w <= 0 || h <= 0) return;
    const float R = radius_px > 0.0f ? radius_px : 8.0f;
    const float S = strength <= 0.0f ? 0.7f : strength > 1.0f ? 1.0f
                                                              : strength;
    static const float taps[12][2] = {
        {1.0f, 0.0f},   {0.5f, 0.866f},   {-0.5f, 0.866f},
        {-1.0f, 0.0f},  {-0.5f, -0.866f}, {0.5f, -0.866f},
        {0.433f, 0.25f},  {0.0f, 0.5f},  {-0.433f, 0.25f},
        {-0.433f, -0.25f}, {0.0f, -0.5f}, {0.433f, -0.25f},
    };
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            int dc = depth[(size_t)i * w + j];
            if (!dc) continue;
            float occ = 0.0f; int n = 0;
            for (int t = 0; t < 12; t++) {
                int jj = j + (int)(taps[t][0] * R + (taps[t][0] < 0 ? -0.5f : 0.5f));
                int ii = i + (int)(taps[t][1] * R + (taps[t][1] < 0 ? -0.5f : 0.5f));
                if (ii < 0 || ii >= h || jj < 0 || jj >= w) continue;
                int dn = depth[(size_t)ii * w + jj];
                if (!dn) continue;
                float dd = (float)dc - (float)dn - 2.0f;    // nearer by > bias
                if (dd <= 0.0f) { n++; continue; }
                float o = dd * (1.0f / 8.0f);
                if (o > 1.0f) o = 1.0f;
                if (dd > 24.0f) o *= 24.0f / dd;            // range falloff
                occ += o; n++;
            }
            if (!n) continue;
            float ao = 1.0f - S * (occ / (float)n);
            float v = (float)img[(size_t)i * w + j] * ao;
            img[(size_t)i * w + j] = to_u8(v);
        }
}

// ---------------------------------------------------------------------------
// LOD-matched rendering
// ---------------------------------------------------------------------------
int mc_render_pick_lod(const mc_sample_lods *ls, float vox_per_pixel) {
    if (!ls || ls->nlods <= 1) return 0;
    int L = 0;
    float v = vox_per_pixel;
    while (v >= 2.0f && L < ls->nlods - 1) { v *= 0.5f; L++; }
    // skip levels the caller left empty
    while (L > 0 && (ls->lods[L].nz <= 0 || !ls->lods[L].block)) L--;
    return L;
}

float mc_quad_spacing(const mc_quad *q) {
    if (!q || !q->grid || q->gw < 2 || q->gh < 1) return 1.0f;
    // probe up to 32 horizontal neighbor pairs along the grid diagonal
    double sum = 0.0;
    int n = 0;
    int probes = q->gh < 32 ? q->gh : 32;
    for (int i = 0; i < probes; i++) {
        int gy = (int)(((int64_t)i * (q->gh - 1)) / (probes > 1 ? probes - 1 : 1));
        int gx = (int)(((int64_t)i * (q->gw - 2)) / (probes > 1 ? probes - 1 : 1));
        const float *a = q->grid + ((size_t)gy * q->gw + gx) * 3;
        const float *b = a + 3;
        if (!qvalid(a) || !qvalid(b)) continue;
        float dz = b[0] - a[0], dy = b[1] - a[1], dx = b[2] - a[2];
        sum += sqrtf(dz * dz + dy * dy + dx * dx);
        n++;
    }
    return n ? (float)(sum / n) : 1.0f;
}

// wrap a rowgen: remap generated LOD-0 points into LOD-L voxel space.
// c_L = c_0 * 2^-L + (0.5 * 2^-L - 0.5); border points that map a fraction
// below 0 clamp to 0 (they are inside voxel 0 of the coarse level) instead
// of tripping the <0 invalid convention.
typedef struct {
    rowgen_fn inner;
    const void *inner_ud;
    float a, b;
} lod_rg;

static void lod_rowgen(const void *ud, int row, int w,
                       float *pts, float *normals) {
    const lod_rg *g = ud;
    g->inner(g->inner_ud, row, w, pts, normals);
    for (int j = 0; j < w; j++) {
        float *P = pts + (size_t)j * 3;
        if (!pt_valid(P)) continue;
        for (int k = 0; k < 3; k++) {
            float v = P[k] * g->a + g->b;
            P[k] = v < 0.0f ? 0.0f : v;
        }
    }
    // normals are directions: unchanged under uniform scaling
}

static int render_lod(const mc_sample_lods *ls, int L,
                      rowgen_fn inner, const void *inner_ud,
                      int w, int h, const mc_render_params *p,
                      uint8_t *out, int nthreads) {
    const float inv = 1.0f / (float)(1 << L);
    lod_rg g = { inner, inner_ud, inv, 0.5f * inv - 0.5f };
    mc_render_params pl_ = *p;
    pl_.t0 = p->t0 * inv;       // same physical slab ...
    pl_.t1 = p->t1 * inv;       // ... stepped at the coarse level's pitch
    render_bands(&ls->lods[L], NULL, NULL, lod_rowgen, &g, w, h, &pl_,
                 out, nthreads);
    return 0;
}

int mc_render_plane_lod(const mc_sample_lods *ls, const mc_plane *pl,
                        int w, int h, float scale,
                        const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!ls || !pl || !p || !out || w <= 0 || h <= 0) return -1;
    int L = mc_render_pick_lod(ls, scale);
    if (L == 0)
        return mc_render_plane(&ls->lods[0], pl, w, h, scale, p, out, nthreads);
    plane_rg g = { *pl, scale, (float)w * 0.5f, (float)h * 0.5f };
    v3_norm(g.pl.normal);
    return render_lod(ls, L, plane_rowgen, &g, w, h, p, out, nthreads);
}

int mc_render_quad_lod(const mc_sample_lods *ls, const mc_quad *q,
                       float x0, float y0, float step, int w, int h,
                       const mc_render_params *p, uint8_t *out, int nthreads) {
    if (!ls || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !p || !out || w <= 0 || h <= 0) return -1;
    int L = mc_render_pick_lod(ls, step * mc_quad_spacing(q));
    if (L == 0)
        return mc_render_quad(&ls->lods[0], q, x0, y0, step, w, h, p, out,
                              nthreads);
    quad_rg g = { q, x0, y0, step };
    return render_lod(ls, L, quad_rowgen, &g, w, h, p, out, nthreads);
}

// ---------------------------------------------------------------------------
// 3D resampling (surface-aligned volumes)
// ---------------------------------------------------------------------------
typedef struct {
    const mc_sample_src *src;
    const mc_quad *q;
    float x0, y0, step;
    float t0, dt;
    int w, h, nlayers;
    mc_filter f;
    uint8_t *out;
    int row0, row1;
} qvol_band_t;

static void *qvol_band_main(void *ud) {
    qvol_band_t *b = ud;
    mc_sampler *s = mc_sampler_new(b->src);
    float *row = malloc((size_t)b->w * 6 * sizeof(float));
    const size_t layer = (size_t)b->w * b->h;
    if (!s || !row) {
        for (int k = 0; k < b->nlayers; k++)
            memset(b->out + layer * k + (size_t)b->row0 * b->w, 0,
                   (size_t)(b->row1 - b->row0) * b->w);
        free(row); mc_sampler_free(s);
        return NULL;
    }
    float *nrm = row + (size_t)b->w * 3;
    quad_rg g = { b->q, b->x0, b->y0, b->step };
    for (int i = b->row0; i < b->row1; i++) {
        quad_rowgen(&g, i, b->w, row, nrm);
        for (int j = 0; j < b->w; j++) {
            const float *P = row + (size_t)j * 3;
            const float *N = nrm + (size_t)j * 3;
            uint8_t *o = b->out + (size_t)i * b->w + j;
            if (!pt_valid(P)) {
                for (int k = 0; k < b->nlayers; k++) o[layer * k] = 0;
                continue;
            }
            float nz = N[0], ny = N[1], nx = N[2];
            float n2 = nz * nz + ny * ny + nx * nx;
            if (n2 >= 1e-12f && (n2 < 0.9998f || n2 > 1.0002f)) {
                float nl = 1.0f / sqrtf(n2);
                nz *= nl; ny *= nl; nx *= nl;
            }
            float pz = P[0] + b->t0 * nz, py = P[1] + b->t0 * ny,
                  px = P[2] + b->t0 * nx;
            const float sz_ = b->dt * nz, sy_ = b->dt * ny, sx_ = b->dt * nx;
            int k = 0;
            if (b->f == MC_FILTER_TRILINEAR) {
                for (; k + 4 <= b->nlayers; k += 4) {
                    float bz[4], by[4], bx[4], v4[4];
                    for (int t = 0; t < 4; t++) {
                        bz[t] = pz; by[t] = py; bx[t] = px;
                        pz += sz_; py += sy_; px += sx_;
                    }
                    mc_s_tri4(s, bz, by, bx, v4);
                    for (int t = 0; t < 4; t++)
                        o[layer * (k + t)] = to_u8(v4[t]);
                }
            }
            for (; k < b->nlayers; k++) {
                o[layer * k] = to_u8(mc_s_sample(s, pz, py, px, b->f));
                pz += sz_; py += sy_; px += sx_;
            }
        }
    }
    free(row);
    mc_sampler_free(s);
    return NULL;
}

int mc_sample_quad_volume(const mc_sample_src *src, const mc_quad *q,
                          float x0, float y0, float step, int w, int h,
                          float t0, float dt, int nlayers,
                          mc_filter f, uint8_t *out, int nthreads) {
    if (!src || !q || !q->grid || q->gw < 1 || q->gh < 1 ||
        !out || w <= 0 || h <= 0 || nlayers <= 0) return -1;
    if (nthreads <= 0) {
        long nc = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = nc > 0 ? (int)nc : 1;
    }
    if (nthreads > 16) nthreads = 16;
    if (nthreads > h)  nthreads = h;
    pthread_t th[16];
    qvol_band_t bands[16];
    int per = (h + nthreads - 1) / nthreads;
    int nb = 0;
    for (int t = 0; t < nthreads; t++) {
        int r0 = t * per, r1 = r0 + per > h ? h : r0 + per;
        if (r0 >= r1) break;
        bands[nb] = (qvol_band_t){ src, q, x0, y0, step, t0, dt,
                                   w, h, nlayers, f, out, r0, r1 };
        if (nthreads == 1 ||
            pthread_create(&th[nb], NULL, qvol_band_main, &bands[nb]) != 0) {
            qvol_band_main(&bands[nb]);
            continue;
        }
        nb++;
    }
    for (int t = 0; t < nb; t++) pthread_join(th[t], NULL);
    return 0;
}

int mc_sample_box(const mc_sample_src *src,
                  const float origin[3], const float du[3],
                  const float dv[3], const float dw[3],
                  int w, int h, int d,
                  mc_filter f, uint8_t *out, int nthreads) {
    if (!src || !origin || !du || !dv || !dw || !out ||
        w <= 0 || h <= 0 || d <= 0) return -1;
    // each depth slice is a plane render with the layer offset folded into
    // the origin; comp NONE so no normals are needed
    mc_render_params p = { .filter = f, .comp = MC_COMP_NONE };
    for (int k = 0; k < d; k++) {
        mc_plane pl;
        for (int c = 0; c < 3; c++) {
            // mc_plane_gen centers the image; sample with corner semantics
            pl.origin[c] = origin[c] + (float)k * dw[c] +
                           ((float)w * 0.5f) * du[c] + ((float)h * 0.5f) * dv[c];
            pl.normal[c] = 0.0f;
            pl.u[c] = du[c];
            pl.v[c] = dv[c];
        }
        if (mc_render_plane(src, &pl, w, h, 1.0f, &p,
                            out + (size_t)k * w * h, nthreads) != 0)
            return -1;
    }
    return 0;
}

// ============================================================================
// mc_zarr — zarr v3-sharded-c3d + v2-flat reader (dep: zstd)
// ============================================================================
#include <zstd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <zstd.h>

// ---------------------------------------------------------------------------
// tiny JSON scraping — enough for zarr.json / .zarray (no general parser).
// ---------------------------------------------------------------------------

// First integer array of `n` ints after the literal `"key"`. -1 on miss.
static int json_int_array(const char *j, const char *key, long out[], int n) {
    const char *p = strstr(j, key);
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    ++p;
    for (int i = 0; i < n; ++i) {
        char *end;
        out[i] = strtol(p, &end, 10);
        if (end == p) return -1;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p;
    }
    return 0;
}

// Is `needle` present anywhere in `j` (substring)?
static int json_has(const char *j, const char *needle) {
    return strstr(j, needle) != NULL;
}

// ---------------------------------------------------------------------------
// blosc1 decode (shuffle=0, typesize=1) — lifted from tools/mc_fetch.c.
// header: ver verlz flags typesize nbytes(4) blocksize(4) cbytes(4), LE.
// ---------------------------------------------------------------------------
static uint8_t *blosc_decode(const uint8_t *src, size_t srclen, size_t *out_len) {
    if (srclen < 16) return NULL;
    uint8_t flags = src[2];
    uint32_t nbytes, blocksize, cbytes;
    memcpy(&nbytes, src + 4, 4);
    memcpy(&blocksize, src + 8, 4);
    memcpy(&cbytes, src + 12, 4);
    if (cbytes > srclen || !nbytes) return NULL;
    if (flags & 0x1 || flags & 0x4) { fprintf(stderr, "mc_zarr: blosc shuffle unsupported\n"); return NULL; }
    uint8_t *out = malloc(nbytes);
    if (!out) return NULL;
    if (flags & 0x2) {                              // memcpyed: raw payload follows header
        if (16 + (size_t)nbytes > srclen) { free(out); return NULL; }
        memcpy(out, src + 16, nbytes);
        *out_len = nbytes;
        return out;
    }
    uint32_t nblocks = (nbytes + blocksize - 1) / blocksize;
    const uint8_t *bstarts = src + 16;
    if (16 + (size_t)nblocks * 4 > srclen) { free(out); return NULL; }
    size_t off = 0;
    for (uint32_t b = 0; b < nblocks; ++b) {
        uint32_t bs;
        memcpy(&bs, bstarts + (size_t)b * 4, 4);
        uint32_t neblock = (b == nblocks - 1) ? nbytes - b * blocksize : blocksize;
        if ((size_t)bs + 4 > srclen) { free(out); return NULL; }
        int32_t cb;
        memcpy(&cb, src + bs, 4);
        const uint8_t *payload = src + bs + 4;
        if (cb <= 0 || (size_t)bs + 4 + (size_t)cb > srclen) { free(out); return NULL; }
        if ((uint32_t)cb == neblock) {
            memcpy(out + off, payload, neblock);    // stored uncompressed
        } else {
            size_t got = ZSTD_decompress(out + off, neblock, payload, (size_t)cb);
            if (ZSTD_isError(got) || got != neblock) { free(out); return NULL; }
        }
        off += neblock;
    }
    *out_len = nbytes;
    return out;
}

// ---------------------------------------------------------------------------
// mc_zarr handle
// ---------------------------------------------------------------------------
enum { ZV3 = 3, ZV2 = 2 };

// Cached shard index footer: the (offset,len) table is immutable per shard, so
// read the 64KB footer ONCE and reuse it for all of a shard's inner chunks.
// Without this, fetching a shard's 4096 chunks one-at-a-time re-reads 64KB *4096.
#define MC_FOOTER_CACHE 64
typedef struct {
    uint64_t shard_id;     // (sz<<40)|(sy<<20)|sx, ~0 = empty slot
    uint8_t *idx;          // malloc'd footer bytes (n_inner*16)
    uint64_t lru;          // last-use tick
} footer_ent;

struct mc_zarr {
    mc_zarr_read_fn read;
    void *ud;
    int version;            // ZV3 | ZV2
    int shape[3];           // voxels (z,y,x)
    int shard_edge;         // v3: chunk_grid chunk_shape; v2: == inner_edge
    int inner_edge;         // v3: sharding inner chunk; v2: .zarray chunks
    int inner_grid[3];      // ceil(shape/inner_edge) per axis (z,y,x)
    int per;                // inner chunks per shard axis = shard_edge/inner_edge
    char codec[16];         // "c3d" | "blosc" | "raw"
    char sep;               // v2 dimension separator ('.' default, or '/')

    pthread_mutex_t fmu;    // guards the footer cache
    footer_ent fcache[MC_FOOTER_CACHE];
    uint64_t ftick;
};

// fetch a whole object by key; returns malloc'd buf or NULL (sets *len).
// `err` (optional): *err=1 if the read FAILED (transient — retry, do NOT treat
// as air); *err=0 if the object is genuinely absent (clean 404 -> confirmed
// air) or present. Distinguishes z->read's -1 (error) from 0+NULL (404).
static uint8_t *fetch_all(mc_zarr *z, const char *key, size_t *len, int *err) {
    if (err) *err = 0;
    uint8_t *buf = NULL;
    size_t n = 0;
    if (z->read(z->ud, key, 0, 0, &buf, &n) < 0) {   // transient error
        free(buf); *len = 0; if (err) *err = 1; return NULL;
    }
    *len = n;                                         // 0 + NULL == clean 404 (air)
    return buf;
}

mc_zarr *mc_zarr_open(mc_zarr_read_fn read, void *ud) {
    if (!read) return NULL;
    size_t jl = 0;
    uint8_t *jb = NULL;
    int v3 = 1;
    if (read(ud, "zarr.json", 0, 0, &jb, &jl) < 0 || !jb || !jl) {
        free(jb);
        jb = NULL;
        jl = 0;
        v3 = 0;
        if (read(ud, ".zarray", 0, 0, &jb, &jl) < 0 || !jb || !jl) { free(jb); return NULL; }
    }
    char *j = malloc(jl + 1);
    if (!j) { free(jb); return NULL; }
    memcpy(j, jb, jl);
    j[jl] = 0;
    free(jb);

    mc_zarr *z = calloc(1, sizeof *z);
    if (!z) { free(j); return NULL; }
    pthread_mutex_init(&z->fmu, NULL);
    z->read = read;
    z->ud = ud;

    long shp[3];
    if (json_int_array(j, "\"shape\"", shp, 3) != 0) { free(j); free(z); return NULL; }
    z->shape[0] = (int)shp[0];
    z->shape[1] = (int)shp[1];
    z->shape[2] = (int)shp[2];

    if (v3) {
        z->version = ZV3;
        if (!json_has(j, "sharding_indexed")) { free(j); free(z); return NULL; }
        // chunk_grid.chunk_shape = shard edge (first int array after "chunk_grid").
        const char *cg = strstr(j, "\"chunk_grid\"");
        long shard[3];
        if (!cg || json_int_array(cg, "\"chunk_shape\"", shard, 3) != 0) { free(j); free(z); return NULL; }
        // sharding_indexed configuration.chunk_shape = inner edge (after that codec).
        const char *sh = strstr(j, "sharding_indexed");
        // chunk_shape appears in the sharding config BEFORE the codec name in the
        // emitted json; search from the codecs array start instead.
        const char *cc = strstr(j, "\"codecs\"");
        long inner[3];
        if (!cc || json_int_array(cc, "\"chunk_shape\"", inner, 3) != 0) { free(j); free(z); return NULL; }
        (void)sh;
        z->shard_edge = (int)shard[0];
        z->inner_edge = (int)inner[0];
        // inner codec: these archives are always c3d.
        if (json_has(j, "\"c3d\"")) snprintf(z->codec, sizeof z->codec, "c3d");
        else { fprintf(stderr, "mc_zarr: v3 inner codec not c3d (unsupported)\n"); free(j); free(z); return NULL; }
    } else {
        z->version = ZV2;
        long ch[3];
        if (json_int_array(j, "\"chunks\"", ch, 3) != 0) { free(j); free(z); return NULL; }
        z->inner_edge = (int)ch[0];
        z->shard_edge = (int)ch[0];           // a v2 chunk is its own shard
        // compressor: null -> raw, else blosc (the standardized scroll zarrs).
        if (json_has(j, "\"compressor\": null") || json_has(j, "\"compressor\":null"))
            snprintf(z->codec, sizeof z->codec, "raw");
        else snprintf(z->codec, sizeof z->codec, "blosc");
        // dimension_separator default '.'
        z->sep = json_has(j, "\"dimension_separator\": \"/\"") ? '/' : '.';
    }

    free(j);
    if (z->inner_edge <= 0 || z->shard_edge <= 0 || z->shard_edge % z->inner_edge) {
        free(z);
        return NULL;
    }
    z->per = z->shard_edge / z->inner_edge;
    for (int d = 0; d < 3; ++d)
        z->inner_grid[d] = (z->shape[d] + z->inner_edge - 1) / z->inner_edge;
    return z;
}

void mc_zarr_free(mc_zarr *z) {
    if (!z) return;
    for (int i = 0; i < MC_FOOTER_CACHE; ++i) free(z->fcache[i].idx);
    pthread_mutex_destroy(&z->fmu);
    free(z);
}

void mc_zarr_shape(const mc_zarr *z, int *nz, int *ny, int *nx) {
    if (nz) *nz = z->shape[0];
    if (ny) *ny = z->shape[1];
    if (nx) *nx = z->shape[2];
}
int mc_zarr_inner_edge(const mc_zarr *z) { return z->inner_edge; }
int mc_zarr_shard_edge(const mc_zarr *z) { return z->shard_edge; }
const char *mc_zarr_inner_codec(const mc_zarr *z) { return z->codec; }
void mc_zarr_inner_grid(const mc_zarr *z, int *nz, int *ny, int *nx) {
    if (nz) *nz = z->inner_grid[0];
    if (ny) *ny = z->inner_grid[1];
    if (nx) *nx = z->inner_grid[2];
}

// ---------------------------------------------------------------------------
// keys
// ---------------------------------------------------------------------------

// v3 shard key for the shard containing global inner chunk (cz,cy,cx): "c/sz/sy/sx".
// v2 chunk key for inner chunk (cz,cy,cx): "cz<sep>cy<sep>cx".
static void chunk_key(const mc_zarr *z, int cz, int cy, int cx, char out[64]) {
    if (z->version == ZV3) {
        int sz = cz / z->per, sy = cy / z->per, sx = cx / z->per;
        snprintf(out, 64, "c/%d/%d/%d", sz, sy, sx);
    } else {
        snprintf(out, 64, "%d%c%d%c%d", cz, z->sep, cy, z->sep, cx);
    }
}

// ---------------------------------------------------------------------------
// v3 shard index: n_inner entries of (offset:u64, nbytes:u64) LE at shard start.
// missing == both == 0xFFFF...F. Linear order row-major, z slowest.
// ---------------------------------------------------------------------------
static int index_entry(const uint8_t *idx, size_t n_inner, size_t lin,
                       uint64_t *off, uint64_t *nb) {
    if (lin >= n_inner) return -1;
    memcpy(off, idx + lin * 16, 8);
    memcpy(nb, idx + lin * 16 + 8, 8);
    if (*off == ~(uint64_t)0 && *nb == ~(uint64_t)0) return 1;   // missing
    return 0;
}

// shard-relative linear inner index, row-major (z slowest, x fastest).
static size_t inner_linear(const mc_zarr *z, int cz, int cy, int cx) {
    int rz = cz % z->per, ry = cy % z->per, rx = cx % z->per;
    return ((size_t)rz * z->per + ry) * z->per + rx;
}

// Get the shard's index footer (n_inner*16 bytes), cached. Reads it once per
// shard via one ranged GET, then serves all the shard's chunks from RAM.
// Returns a borrowed pointer valid until the entry is evicted; copy out the
// (off,len) you need while you still need them (callers do this immediately).
// NULL if the shard is absent (all air) or on error.
// `err` (optional) distinguishes the two NULL returns: *err=1 means the footer
// READ FAILED (transient — caller must retry, NOT treat as air); *err=0 means
// the shard is genuinely absent (a clean 404 — confirmed air).
static const uint8_t *footer_get_ex(mc_zarr *z, int cz, int cy, int cx, int *err) {
    if (err) *err = 0;
    if (z->version != ZV3) return NULL;
    uint64_t sid = ((uint64_t)(cz / z->per) << 40) |
                   ((uint64_t)(cy / z->per) << 20) | (uint64_t)(cx / z->per);
    pthread_mutex_lock(&z->fmu);
    for (int i = 0; i < MC_FOOTER_CACHE; ++i)
        if (z->fcache[i].idx && z->fcache[i].shard_id == sid) {
            z->fcache[i].lru = ++z->ftick;
            const uint8_t *p = z->fcache[i].idx;
            pthread_mutex_unlock(&z->fmu);
            return p;
        }
    pthread_mutex_unlock(&z->fmu);

    // miss: fetch the footer (outside the lock — it's one ranged GET).
    char key[64];
    chunk_key(z, cz, cy, cx, key);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    size_t idx_bytes = n_inner * 16;
    uint8_t *idx = NULL;
    size_t got = 0;
    int rc = z->read(z->ud, key, 0, idx_bytes, &idx, &got);
    if (rc < 0) {                       // transient read error -> retry, NOT air
        free(idx);
        if (err) *err = 1;
        return NULL;
    }
    if (!idx || got < idx_bytes) {      // clean 404 / short -> genuinely absent shard
        free(idx);
        return NULL;                    // *err stays 0
    }

    pthread_mutex_lock(&z->fmu);
    // re-check (another thread may have inserted it); if so, drop ours.
    int victim = 0;
    uint64_t oldest = ~0ull;
    for (int i = 0; i < MC_FOOTER_CACHE; ++i) {
        if (z->fcache[i].idx && z->fcache[i].shard_id == sid) {
            free(idx);
            z->fcache[i].lru = ++z->ftick;
            const uint8_t *p = z->fcache[i].idx;
            pthread_mutex_unlock(&z->fmu);
            return p;
        }
        if (!z->fcache[i].idx) { victim = i; oldest = 0; }
        else if (z->fcache[i].lru < oldest) { oldest = z->fcache[i].lru; victim = i; }
    }
    free(z->fcache[victim].idx);
    z->fcache[victim].idx = idx;
    z->fcache[victim].shard_id = sid;
    z->fcache[victim].lru = ++z->ftick;
    pthread_mutex_unlock(&z->fmu);
    return idx;
}
static const uint8_t *footer_get(mc_zarr *z, int cz, int cy, int cx) {
    return footer_get_ex(z, cz, cy, cx, NULL);
}

int mc_zarr_shard_all_air(mc_zarr *z, int cz, int cy, int cx) {
    if (z->version != ZV3) {
        // v2: "all air" == the single chunk object is absent.
        char key[64];
        chunk_key(z, cz, cy, cx, key);
        uint8_t *b = NULL;
        size_t n = 0;
        if (z->read(z->ud, key, 0, 1, &b, &n) < 0) return -1;
        free(b);
        return n == 0 ? 1 : 0;
    }
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 1;   // absent shard = air
    for (size_t i = 0; i < n_inner; ++i) {
        uint64_t off, nb;
        if (index_entry(idx, n_inner, i, &off, &nb) == 0) return 0;
    }
    return 1;
}

// decode a v2 chunk blob (blosc/raw) into a fresh dense buffer.
static uint8_t *v2_decode(const mc_zarr *z, uint8_t *blob, size_t blen, size_t *out_len) {
    if (strcmp(z->codec, "raw") == 0) { *out_len = blen; return blob; }   // takes ownership
    size_t dl = 0;
    uint8_t *dense = blosc_decode(blob, blen, &dl);
    free(blob);
    if (!dense) return NULL;
    *out_len = dl;
    return dense;
}

// Locate one inner chunk WITHOUT fetching: fill its object key + byte range from
// the cached shard footer. Returns 0 (found: *off/*nb set, c3d raw range), 1
// (absent/air), <0 (error). v2 chunks are whole objects -> off=0,nb=0 (full GET).
// Lets a caller batch many chunks' ranged GETs into one s3_get_batch.
// 0 found (off/nb set), 1 confirmed air (clean 404 / missing index entry),
// <0 transient READ ERROR (caller must retry, NOT mark the region air).
int mc_zarr_chunk_locate(mc_zarr *z, int cz, int cy, int cx,
                         char key_out[64], uint64_t *off, uint64_t *nb) {
    chunk_key(z, cz, cy, cx, key_out);
    if (z->version == ZV2) { *off = 0; *nb = 0; return 0; }
    size_t n_inner = (size_t)z->per * z->per * z->per;
    int err = 0;
    const uint8_t *idx = footer_get_ex(z, cz, cy, cx, &err);
    if (!idx) return err ? -1 : 1;                          // read error -> retry; else air
    size_t lin = inner_linear(z, cz, cy, cx);
    int st = index_entry(idx, n_inner, lin, off, nb);
    if (st != 0) return 1;                                   // missing entry -> confirmed air
    return 0;
}

int mc_zarr_read_inner(mc_zarr *z, int cz, int cy, int cx, uint8_t **raw, size_t *len) {
    *raw = NULL;
    *len = 0;
    char key[64];
    chunk_key(z, cz, cy, cx, key);

    if (z->version == ZV2) {
        // v2: a 404 (chunk object doesn't exist on S3) == confirmed air. A read
        // error must retry, not be recorded as air.
        size_t blen = 0; int err = 0;
        uint8_t *blob = fetch_all(z, key, &blen, &err);
        if (err) { free(blob); return -1; }               // transient -> retry
        if (!blob || !blen) { free(blob); return 1; }     // clean 404 -> confirmed air
        size_t dl = 0;
        uint8_t *dense = v2_decode(z, blob, blen, &dl);
        if (!dense) return -1;
        *raw = dense;
        *len = dl;
        return 0;
    }

    // v3: get the (cached) index footer, then the one inner chunk's payload range.
    size_t n_inner = (size_t)z->per * z->per * z->per;
    int ferr = 0;
    const uint8_t *idx = footer_get_ex(z, cz, cy, cx, &ferr);
    if (!idx) return ferr ? -1 : 1;                         // read error -> retry; else air
    size_t lin = inner_linear(z, cz, cy, cx);
    uint64_t off, nb;
    int st = index_entry(idx, n_inner, lin, &off, &nb);
    if (st != 0) return 1;                                   // index air marker -> confirmed air
    uint8_t *payload = NULL;
    size_t plen = 0;
    if (z->read(z->ud, key, off, nb, &payload, &plen) < 0) { return -1; }  // transient
    if (!payload || plen < nb) { free(payload); return -1; }               // short -> retry
    *raw = payload;
    *len = nb;
    return 0;   // c3d raw bytes — caller decodes.
}

int mc_zarr_read_shard(mc_zarr *z, int cz, int cy, int cx,
                       mc_zarr_chunk_fn sink, void *sink_ud) {
    if (z->version == ZV2) {
        // a v2 "shard" is one chunk.
        uint8_t *dense = NULL;
        size_t dl = 0;
        int st = mc_zarr_read_inner(z, cz, cy, cx, &dense, &dl);
        if (st < 0) return -1;
        if (st == 0) { sink(sink_ud, cz, cy, cx, dense, dl); free(dense); }
        return 0;
    }

    // v3: read the index footer ONCE, then range-GET each present inner chunk
    // individually (no whole-shard buffering). Each chunk is sunk + freed before
    // the next, so RAM stays at one chunk and disk grows per-chunk.
    char key[64];
    chunk_key(z, cz, cy, cx, key);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 0;   // absent shard = all air
    int sz0 = (cz / z->per) * z->per, sy0 = (cy / z->per) * z->per, sx0 = (cx / z->per) * z->per;
    for (int iz = 0; iz < z->per; ++iz)
        for (int iy = 0; iy < z->per; ++iy)
            for (int ix = 0; ix < z->per; ++ix) {
                size_t lin = ((size_t)iz * z->per + iy) * z->per + ix;
                uint64_t off, nb;
                if (index_entry(idx, n_inner, lin, &off, &nb) != 0) continue;
                int gz = sz0 + iz, gy = sy0 + iy, gx = sx0 + ix;
                if (gz >= z->inner_grid[0] || gy >= z->inner_grid[1] || gx >= z->inner_grid[2])
                    continue;
                uint8_t *payload = NULL;
                size_t plen = 0;
                if (z->read(z->ud, key, off, nb, &payload, &plen) < 0) return -1;
                if (payload && plen >= nb)
                    sink(sink_ud, gz, gy, gx, payload, (size_t)nb);   // c3d raw bytes
                free(payload);
            }
    return 0;
}

int mc_zarr_shard_index(mc_zarr *z, int cz, int cy, int cx,
                        char key_out[64], mc_zarr_range **out, int *n) {
    *out = NULL;
    *n = 0;
    if (z->version == ZV2) {
        // a v2 "shard" is one chunk object; whole-object fetch (off/len = 0).
        chunk_key(z, cz, cy, cx, key_out);
        mc_zarr_range *r = malloc(sizeof *r);
        if (!r) return -1;
        r[0] = (mc_zarr_range){cz, cy, cx, 0, 0};
        *out = r;
        *n = 1;
        return 0;
    }
    chunk_key(z, cz, cy, cx, key_out);
    size_t n_inner = (size_t)z->per * z->per * z->per;
    const uint8_t *idx = footer_get(z, cz, cy, cx);
    if (!idx) return 0;                                     // absent shard = all air
    mc_zarr_range *arr = malloc(n_inner * sizeof *arr);
    if (!arr) return -1;
    int sz0 = (cz / z->per) * z->per, sy0 = (cy / z->per) * z->per, sx0 = (cx / z->per) * z->per;
    int cnt = 0;
    for (int iz = 0; iz < z->per; ++iz)
        for (int iy = 0; iy < z->per; ++iy)
            for (int ix = 0; ix < z->per; ++ix) {
                size_t lin = ((size_t)iz * z->per + iy) * z->per + ix;
                uint64_t off, nb;
                if (index_entry(idx, n_inner, lin, &off, &nb) != 0) continue;
                int gz = sz0 + iz, gy = sy0 + iy, gx = sx0 + ix;
                if (gz >= z->inner_grid[0] || gy >= z->inner_grid[1] || gx >= z->inner_grid[2])
                    continue;
                arr[cnt++] = (mc_zarr_range){gz, gy, gx, off, nb};
            }
    *out = arr;
    *n = cnt;
    return 0;
}

// ============================================================================
// mc_s3 — s3://-backed mc_reader glue (dep: libs3)
// ============================================================================
#include "libs3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mc_s3 {
    s3_client *cl;
    char *url;
    mc_reader *r;
    _Atomic uint64_t net_bytes;   // bytes pulled over S3 (for the volume's rate readout)
};

static int s3_read_cb(void *ud, uint64_t off, uint32_t len, uint8_t *dst){
    mc_s3 *s=ud;
    s3_response resp={0};
    if(s3_get_range(s->cl,s->url,off,len,&resp)!=S3_OK || !s3_response_ok(&resp)){
        s3_response_free(&resp);
        return -1;
    }
    int rc=-1;
    if(resp.status==206 && resp.body_len>=len){
        memcpy(dst,resp.body,len); rc=0;          // proper ranged reply
    } else if(resp.status==200 && resp.body_len>=off+len){
        memcpy(dst,resp.body+off,len); rc=0;      // server ignored Range and
    }                                             // sent the whole object
    if(rc==0) atomic_fetch_add_explicit(&s->net_bytes,len,memory_order_relaxed);
    s3_response_free(&resp);
    return rc;
}

uint64_t mc_s3_net_bytes(mc_s3 *s){
    return s ? atomic_load_explicit(&s->net_bytes,memory_order_relaxed) : 0;
}

mc_s3 *mc_s3_open(const char *url){
    if(!url) return NULL;
    mc_s3 *s=calloc(1,sizeof *s);
    // Full credential resolution (profile/IMDS/SSO/env), else anonymous -- same as
    // the zarr transcode path, so a private bucket (philodemos) authenticates.
    s3_config cfg={0};
    s3_credentials creds={0};
    if(s3_credentials_load(NULL,&creds)==S3_OK) cfg.creds=creds;
    s->cl=s3_client_new(&cfg);
    s3_credentials_free(&creds);
    if(!s->cl){ free(s); return NULL; }
    s->url=strdup(url);
    s3_response head={0};
    uint64_t total=0;
    if(s3_head(s->cl,url,&head)==S3_OK && s3_response_ok(&head))
        total=head.content_length;
    s3_response_free(&head);
    if(!total){ mc_s3_close(s); return NULL; }
    s->r=mc_open_streaming(s3_read_cb,s,total);
    if(!s->r){ mc_s3_close(s); return NULL; }
    mc_reader_set_partial_fetch(s->r,1);
    return s;
}
mc_reader *mc_s3_reader(mc_s3 *s){ return s?s->r:NULL; }
void mc_s3_close(mc_s3 *s){
    if(!s) return;
    if(s->r) mc_close(s->r);
    if(s->cl) s3_client_free(s->cl);
    free(s->url);
    free(s);
}

// ============================================================================
// mc_volume — remote zarr -> local .mca (deps: mc_zarr, libs3, c3d)
// ============================================================================
#include "c3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define MAXLOD 8
#define CHUNK 256
#define PER (CHUNK / BLK)   // 16 blocks per chunk axis

static void *decoder_main(void *ud);
static void *dl_main(void *ud);
static void fill_pool_init(mc_volume *v, int n);
static void fill_pool_stop(mc_volume *v);
static void fill_pool_start(mc_volume *v);
static void fill_pool_wait(mc_volume *v);
static void mc_volume_prefetch_region(mc_volume *v, int lod, int cz, int cy, int cx);
static int  inflight_has(mc_volume *v, uint64_t key);   // single-flight (v->mu held)
static void inflight_add(mc_volume *v, uint64_t key);
static void inflight_del(mc_volume *v, uint64_t key);
static void vol_mark_region(mc_volume *v, int lod, int cz, int cy, int cx);
#define MC_RGEN_SLOTS (1u << 16)   // per-region change-gen table (512KB)

// portable thread naming: macOS names the calling thread (1 arg), glibc
// takes (thread, name)
static void mc_thread_setname(const char *name) {
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}
static const uint8_t *zero256(void);   // shared 32-aligned 256^3 zero buffer

// ---- timing log (MCV_LOG=1 to enable) -------------------------------------
static int g_log = -1;
static double mcv_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;   // ms
}
#define MCVLOG(...) do { \
    if (g_log < 0) g_log = getenv("MCV_LOG") ? 1 : 0; \
    if (g_log) { fprintf(stderr, "[mcv %10.1f] ", mcv_now()); \
                 fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fflush(stderr); } \
} while (0)

// ---------------------------------------------------------------------------
// per-level source (one mc_zarr + its S3 key prefix)
// ---------------------------------------------------------------------------
typedef struct {
    mc_volume *vol;     // back-pointer (for the shared s3 client + net counter)
    char prefix[1024];  // e.g. "s3://bucket/root/0" (no trailing slash)
    mc_zarr *z;
} level_t;

struct mc_volume {
    s3_client *s3;             // NULL for a local-filesystem source
    int local;                 // 1 => root is a local dir, read via file_read
    char root[1024];           // s3://bucket/root, or /local/dir (no trailing slash)
    int nlods;
    level_t lv[MAXLOD];
    mc_archive *arc;           // ONE archive, all LODs
    mc_cache *cache;
    float quality;

    // Streaming mode: the source is an ALREADY-BUILT .mca. The download thread
    // resolves a region's offset in the source reader and copies its compressed blob
    // VERBATIM onto the LOCAL archive (v->arc) -- no decode, no re-encode. Everything
    // else is the normal local-archive path: coverage, THAW decode-from-local, cache,
    // the LIFO request stack. lv[] / decode pool are unused; one dl thread owns the
    // (non-reentrant) source reader.
    int streaming;             // 1 => source is a pre-built .mca, copied verbatim
    mc_s3 *s3mca;              // remote source reader handle (s3/https), or NULL
    struct mc_reader *rd;      // source reader: chunk offsets + verbatim blob bytes
    uint8_t *s_map;            // local-file source: whole .mca mmap'd read-only
    size_t s_map_len;
    size_t s_blob_ema;         // EMA of blob sizes -> adaptive round-A GET length
                               // (dl thread only; fixed over-read either wastes
                               // bandwidth on small blobs or two-trips big ones)
    int s_nz[MAXLOD], s_ny[MAXLOD], s_nx[MAXLOD];   // per-LOD voxel dims (n0>>lod)

    atomic_uint_fast64_t net_bytes;

    pthread_mutex_t mu;        // guards the decode queue + request stack
    pthread_cond_t cv;         // request-stack not-empty (wakes download threads)

    // Decode pipeline: download threads enqueue raw payloads here; a pool of
    // decode workers drains them (decode -> re-encode -> append). This keeps the
    // network saturated (downloaders never wait on CPU) and CPU saturated
    // (decoders run in parallel), instead of serializing download+decode.
    pthread_t decoders[32];
    int ndecoders;
    // Persistent archive->cache fill pool. thaw() hands it the frame's PRESENT
    // block set (shard-partitioned, lock-free single-owner writes) and returns;
    // the workers decode while the frame renders. freeze() waits for them (the
    // barrier) before flipping frozen, preserving no-mutation-while-frozen.
    pthread_t fillers[32];
    int nfillers;
    mc_block_id *fill_ids;           // OWNED copy of this fill's block set (NOT the
                                     // thaw thread-local: multiple caches thaw on the
                                     // same thread and would clobber a shared borrow)
    size_t fill_ids_cap;             // capacity of fill_ids
    uint32_t *fill_bucket;           // bucket[s] = head block index for shard s
    uint32_t *fill_link;             // link[i] = next block index in same shard
    size_t fill_link_cap;            // capacity of fill_link
    size_t fill_n;                   // blocks in the current fill set
    _Atomic uint64_t fill_decoded;   // blocks actually decoded this fill (summed)
    _Atomic uint64_t fill_epoch;     // bumped by thaw to start a fill
    _Atomic uint64_t fill_done;      // workers bump after finishing an epoch
    pthread_mutex_t fill_mu;
    pthread_cond_t fill_start;       // thaw signals; workers wait
    pthread_cond_t fill_fin;         // workers signal; freeze waits
    int fill_stop;
    struct decode_item *dq;    // ring of pending decode items (slot bound is high;
    int dq_cap, dq_head, dq_tail;        // the real backpressure is dq_bytes below)
    size_t dq_bytes;           // compressed bytes currently queued (staging size)
    size_t dq_byte_budget;     // block producers above this (RAM-budgeted staging)
    pthread_cond_t dq_ne;      // not-empty (wake a decoder)
    pthread_cond_t dq_nf;      // not-full  (wake a blocked producer)
    int stop;
    _Atomic int dec_active;    // items currently inside decode->re-encode->append

    // Interactive download-request stack (LIFO): a render miss / prefetch pushes
    // "fetch region R". Download threads pop the NEWEST (current view) first; when
    // full, the OLDEST (stalest, camera moved on) is dropped. DEDUPING: an open-
    // addressing membership set (rs_set) gives O(1) push-dedup instead of an O(n)
    // stack scan -- the prefetch blasts the predicted set every tick, so the stack
    // absorbs duplicates natively without the caller deduping.
    uint64_t *reqstk;          // region keys (the LIFO order)
    uint64_t *rs_set;          // membership set, power-of-two, 0 = empty slot
    int rs_set_mask;           // rs_set capacity - 1
    int rs_cap, rs_n;
    pthread_t dlthreads[16];
    int ndl;

    // Single-flight: region keys currently popped-and-being-fetched/decoded (not
    // yet appended to the .mca, so coverage is still ABSENT). req_push and the
    // download path skip keys here so a region in flight is NOT re-requested and
    // re-decoded every frame during its ~500ms fetch+decode window (the cause of
    // a measured ~15x re-decode). Flat key array (in-flight count is bounded by
    // the decode-queue depth); guarded by the same v->mu. Cleared on append/fail.
    uint64_t *inflight;
    int inflight_cap, inflight_n;

    mc_volume_ready_fn ready_cb;   // fired when a region becomes serveable
    void *ready_ud;

    // Two frozen snapshots, collated once per THAW, read during the frozen render.
    // Same tick thread writes + reads -> plain fields, no atomic, no lock.
    //  net_inflight: ACTUAL download/decode pipeline work = queued downloads +
    //    downloading + decode-queue depth. This is the user-facing "downloading N"
    //    -- stable, reflects real network/transcode work.
    //  work_pending: net_inflight + this frame's undrained misses. The render gate
    //    keys off this ("keep ticking while anything is still settling"). The miss
    //    term swings 0..thousands per frame, so it must NOT leak into the status bar.
    uint64_t net_inflight;
    uint64_t work_pending;
    uint64_t change_gen;       // bumped at THAW when the fill changed cache content
    // Per-REGION change gens (direct-map, no stored keys): slot = hash(region).
    // Writers (decode pool / dl thread / THAW) store the current render gen;
    // collisions only make an unrelated region look changed (a harmless extra
    // render, never a missed one). Lets a viewer skip the streaming re-render
    // when nothing in ITS viewport changed -- the gate was volume-global before.
    _Atomic uint64_t *rgen;
    // Per-stage breakdown of net_inflight (same THAW-collated snapshot pattern):
    // download stack / on-the-wire / decode-queue wait / active decode->encode->
    // append (+ the staging RAM the decode queue holds). Append itself is a
    // synchronous memcpy into the mmap -- there is no archive queue.
    uint64_t snap_queued, snap_downloading, snap_decq, snap_encoding, snap_staging_bytes;
};

// One unit of decode work: the sub^3 cube of source chunks covering one 256^3
// region. For c3d (sub=1) nsub==1; for v2 (sub=2) up to 8. Owns the raw bytes.
typedef struct decode_item {
    int lod, rz, ry, rx;       // target 256^3 region coords
    int sub;                   // 1 (c3d) or 2 (v2)
    int nsub;                  // number of valid sub-chunks
    int oz[8], oy[8], ox[8];   // sub-chunk voxel offsets within the region
    uint8_t *raw[8];           // owned compressed bytes (freed by the decoder)
    size_t rlen[8];
} decode_item;

// RAM the item occupies in the staging queue: its buffered source bytes —
// c3d compressed, or v2 blosc/zstd/raw decoded-dense, whatever was read. The
// byte budget thus naturally holds fewer of the larger (decoded) v2 items.
static size_t decode_item_bytes(const decode_item *it) {
    size_t b = 0;
    for (int k = 0; k < it->nsub; ++k) b += it->rlen[k];
    return b ? b : 1;                                   // ZERO/air items count as 1
}

// ---------------------------------------------------------------------------
// s3 byte source for mc_zarr (prepends the level prefix to the object key)
// ---------------------------------------------------------------------------
static int s3_read(void *ud, const char *key, uint64_t off, uint64_t len,
                   uint8_t **out, size_t *out_len) {
    level_t *lv = ud;
    char url[1280];
    snprintf(url, sizeof url, "%s/%s", lv->prefix, key);
    s3_response resp = {0};
    s3_status st;
    if (len == 0) st = s3_get(lv->vol->s3, url, &resp);
    else          st = s3_get_range(lv->vol->s3, url, off, len, &resp);
    if (st != S3_OK) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    if (s3_response_not_found(&resp)) { s3_response_free(&resp); *out = NULL; *out_len = 0; return 0; }
    if (!s3_response_ok(&resp)) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    // honor a server that ignored Range and sent the whole object.
    const uint8_t *src = resp.body;
    size_t n = resp.body_len;
    if (len != 0 && resp.status == 200 && n >= off + len) { src += off; n = len; }
    uint8_t *buf = malloc(n ? n : 1);
    if (!buf) { s3_response_free(&resp); *out = NULL; *out_len = 0; return -1; }
    memcpy(buf, src, n);
    s3_response_free(&resp);
    atomic_fetch_add_explicit(&lv->vol->net_bytes, n, memory_order_relaxed);
    *out = buf;
    *out_len = n;
    return 0;
}

// Local-filesystem source: mirror of s3_read but reads "<prefix>/<key>" from
// disk. Same contract: return 0 with *out=NULL on a missing key (so the level
// probe / air detection behaves like a 404), <0 on real I/O error, 0 with a
// malloc'd buffer otherwise. `off`/`len` honor ranged reads (footer/index).
static int file_read(void *ud, const char *key, uint64_t off, uint64_t len,
                     uint8_t **out, size_t *out_len) {
    level_t *lv = ud;
    char path[1280];
    snprintf(path, sizeof path, "%s/%s", lv->prefix, key);
    *out = NULL; *out_len = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;                              // missing key == 404
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsz = ftell(f);
    if (fsz < 0) { fclose(f); return -1; }
    uint64_t start = off;
    uint64_t want  = (len == 0) ? (uint64_t)fsz : len;
    if (start > (uint64_t)fsz) { fclose(f); return 0; }      // past EOF
    if (start + want > (uint64_t)fsz) want = (uint64_t)fsz - start;
    if (fseek(f, (long)start, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *buf = malloc(want ? want : 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = want ? fread(buf, 1, want, f) : 0;
    fclose(f);
    if (got != want) { free(buf); return -1; }
    atomic_fetch_add_explicit(&lv->vol->net_bytes, got, memory_order_relaxed);
    *out = buf; *out_len = got;
    return 0;
}

// pack a region (lod,cz,cy,cx) into a 64-bit key.
static uint64_t rkey(int lod, int cz, int cy, int cx) {
    return ((uint64_t)(lod & 7) << 60) | ((uint64_t)(cz & 0xFFFFF) << 40) |
           ((uint64_t)(cy & 0xFFFFF) << 20) | (uint64_t)(cx & 0xFFFFF);
}

// ---------------------------------------------------------------------------
// transcode one 256^3 region (cz,cy,cx) of lod into the .mca. caller ensures
// single-flight. returns 1 transcoded data, 0 air, <0 error.
// ---------------------------------------------------------------------------
// decode one source inner-chunk's raw bytes into `dst` (edge^3, edge = source
// inner_edge: 256 for c3d, 128 for v2). dst need not be 32-aligned for v2; c3d
// needs a 32-aligned 256^3 (the v3 case always passes the region buffer).
// `dec` is a caller-owned, reusable c3d decoder (one per decode-pool thread):
// c3d_decoder_new allocates a 64MB coeff buffer + 2MB symbol buffer, so making
// one per chunk costs ~14% of a decode in alloc/first-touch/free. Reusing the
// thread's decoder across chunks is bit-identical (it carries only scratch).
static void decode_inner(c3d_decoder *dec, const char *codec,
                         const uint8_t *raw, size_t rlen, uint8_t *dst, int edge) {
    size_t vox = (size_t)edge * edge * edge;
    if (strcmp(codec, "c3d") == 0) {
        c3d_decoder_chunk_decode(dec, raw, rlen, dst);  // c3d edge is always 256
    } else {                                            // blosc/raw: already dense u8
        if (rlen >= vox) memcpy(dst, raw, vox);
        else { memset(dst, 0, vox); memcpy(dst, raw, rlen); }
    }
}

// blit a src (edge^3) into the 256^3 region buffer at sub-offset (oz,oy,ox) voxels.
static void blit_sub(uint8_t *region, const uint8_t *src, int edge,
                     int oz, int oy, int ox) {
    for (int z = 0; z < edge; ++z)
        for (int y = 0; y < edge; ++y)
            memcpy(region + (((size_t)(oz + z) * CHUNK + (oy + y)) * CHUNK + ox),
                   src + ((size_t)z * edge + y) * edge, (size_t)edge);
}

// Decode one item (the sub^3 cube for a region) -> assemble 256^3 -> append.
// Frees the item's raw buffers. Runs on a decode-pool thread (off the download
// thread). The c3d decode + mc re-encode are the CPU cost we keep off the net.
// `dense` and `tile` are PERSISTENT per-decoder-thread scratch (each CHUNK^3 =
// 16MB), allocated once in decoder_main and reused. Per-call posix_memalign of a
// 16MB buffer across N decode threads under heavy streaming drove the kernel into
// direct page reclaim (native_queued_spin_lock_slowpath storm in profiling).
static void decode_one(mc_volume *v, c3d_decoder *dec, decode_item *it,
                       uint8_t *dense, uint8_t *tile) {
    const char *codec = mc_zarr_inner_codec(v->lv[it->lod].z);
    const int edge = CHUNK / it->sub;
    const uint64_t key = rkey(it->lod, it->rz, it->ry, it->rx);
    // Skip if this region became resident while the item sat in the queue (a
    // duplicate that slipped past the single-flight claim) — don't redo decode.
    if (mc_archive_chunk_coverage(v->arc, it->lod, it->rz, it->ry, it->rx) != MC_ABSENT) {
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
        pthread_mutex_lock(&v->mu); inflight_del(v, key); pthread_mutex_unlock(&v->mu);
        return;
    }
    if (it->nsub == 0) {                               // all air -> ZERO
        if (mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, zero256()) == 0)
            vol_mark_region(v, it->lod, it->rz, it->ry, it->rx);   // else leave absent -> refetch
        pthread_mutex_lock(&v->mu); inflight_del(v, key); pthread_mutex_unlock(&v->mu);
        return;
    }
    double t_dec0 = mcv_now();
    if (it->sub == 1) {                                // c3d: chunk == region
        decode_inner(dec, codec, it->raw[0], it->rlen[0], dense, CHUNK);
    } else {                                           // v2: blit the cube
        memset(dense, 0, (size_t)CHUNK * CHUNK * CHUNK);
        for (int k = 0; k < it->nsub; ++k) {
            decode_inner(dec, codec, it->raw[k], it->rlen[k], tile, edge);
            blit_sub(dense, tile, edge, it->oz[k], it->oy[k], it->ox[k]);
        }
    }
    double t_enc0 = mcv_now();
    if (mc_archive_append_chunk_raw(v->arc, it->lod, it->rz, it->ry, it->rx, dense) == 0)
        vol_mark_region(v, it->lod, it->rz, it->ry, it->rx);       // else leave absent -> refetch
    double t_end = mcv_now();
    MCVLOG("decoded   lod%d region(%d,%d,%d) codec=%s decode=%.0fms encode=%.0fms",
           it->lod, it->rz, it->ry, it->rx, codec,
           t_enc0 - t_dec0, t_end - t_enc0);
    for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
    pthread_mutex_lock(&v->mu); inflight_del(v, key); pthread_mutex_unlock(&v->mu);  // single-flight clear
}

// Decode-pool worker: drain decode items, decode off the download thread.
// Owns ONE reusable c3d decoder for its lifetime (the 64MB coeff buffer alloc
// is ~14% of a decode; reuse amortizes it to once per thread).
static void *decoder_main(void *ud) {
    mc_volume *v = ud;
    mc_thread_setname("mc-decode");        // distinguish in profilers
    c3d_decoder *dec = c3d_decoder_new();
    if (dec) c3d_decoder_set_denoise(dec, false);   // render cache: no denoise pass
    // Persistent per-thread decode scratch (16MB each), allocated once and reused
    // across every decode -- per-call 16MB allocs across the pool drove kernel page
    // reclaim under heavy streaming. tile holds the sub-cube for v2 (edge<=128).
    uint8_t *dense = NULL, *tile = NULL;
    if (posix_memalign((void **)&dense, 64, (size_t)CHUNK * CHUNK * CHUNK) ||
        posix_memalign((void **)&tile,  64, (size_t)CHUNK * CHUNK * CHUNK)) {
        free(dense); free(tile);
        if (dec) c3d_decoder_free(dec);
        return NULL;                                 // OOM at startup: this worker bows out
    }
    for (;;) {
        pthread_mutex_lock(&v->mu);
        while (v->dq_head == v->dq_tail && !v->stop) pthread_cond_wait(&v->dq_ne, &v->mu);
        if (v->stop && v->dq_head == v->dq_tail) { pthread_mutex_unlock(&v->mu); break; }
        // LIFO: decode the NEWEST queued region first (pop from the producer's end).
        // Like the download stack, interactive coords go stale as the user moves --
        // the freshest region is what's on screen now, so decode it before the
        // backlog. The ring is used as a deque: producer pushes at dq_tail, we pop
        // dq_tail-1.
        v->dq_tail = (v->dq_tail + v->dq_cap - 1) % v->dq_cap;
        decode_item it = v->dq[v->dq_tail];
        v->dq_bytes -= decode_item_bytes(&it);         // free staging budget
        pthread_cond_signal(&v->dq_nf);                // wake a blocked producer
        pthread_mutex_unlock(&v->mu);
        atomic_fetch_add_explicit(&v->dec_active, 1, memory_order_relaxed);
        decode_one(v, dec, &it, dense, tile);
        atomic_fetch_sub_explicit(&v->dec_active, 1, memory_order_relaxed);
        if (v->ready_cb) v->ready_cb(v->ready_ud);     // region became serveable
    }
    free(dense); free(tile);
    if (dec) c3d_decoder_free(dec);
    return NULL;
}

// Producer: push a decode item. Backpressure is BYTE-budgeted (RAM-budgeted
// staging): the download thread only blocks when the queued compressed bytes
// exceed dq_byte_budget — so downloads run far ahead of the CPU-bound decode
// pool and the network stays saturated, instead of stalling on a slot count.
// (A secondary slot-full guard covers the unlikely ring-wrap.) Takes ownership
// of the item's raw buffers.
static void decode_push(mc_volume *v, const decode_item *it) {
    const size_t ib = decode_item_bytes(it);
    pthread_mutex_lock(&v->mu);
    int next = (v->dq_tail + 1) % v->dq_cap;
    int blocked = 0;
    while (!v->stop &&
           (next == v->dq_head ||                       // ring full (rare; cap is huge)
            (v->dq_bytes + ib > v->dq_byte_budget && v->dq_head != v->dq_tail))) {
        blocked = 1;
        pthread_cond_wait(&v->dq_nf, &v->mu);
        next = (v->dq_tail + 1) % v->dq_cap;
    }
    if (v->stop) { pthread_mutex_unlock(&v->mu);
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]); return; }
    v->dq[v->dq_tail] = *it;
    v->dq_tail = next;
    v->dq_bytes += ib;
    pthread_cond_signal(&v->dq_ne);
    size_t qb = v->dq_bytes;
    pthread_mutex_unlock(&v->mu);
    if (blocked) MCVLOG("decode_q  FULL (staging budget hit) queued=%.0fMB", qb / 1048576.0);
}

// unpack a region key.
static void runpack(uint64_t k, int *lod, int *cz, int *cy, int *cx) {
    *lod = (int)((k >> 60) & 7);
    *cz = (int)((k >> 40) & 0xFFFFF);
    *cy = (int)((k >> 20) & 0xFFFFF);
    *cx = (int)(k & 0xFFFFF);
}

// Push an interactive download request (region key) onto the LIFO stack. Newest
// on top. If full, drop the BOTTOM (stalest). Deduped against the stack. Wakes a
// download thread. (cv doubles as the stack's not-empty signal.)
// Single-flight helpers. ALL require v->mu held.
static int inflight_has(mc_volume *v, uint64_t key) {
    for (int i = 0; i < v->inflight_n; ++i) if (v->inflight[i] == key) return 1;
    return 0;
}
static void inflight_add(mc_volume *v, uint64_t key) {
    if (v->inflight_n == v->inflight_cap) {            // grow
        int nc = v->inflight_cap ? v->inflight_cap * 2 : 256;
        uint64_t *p = realloc(v->inflight, (size_t)nc * sizeof *p);
        if (!p) return;                                // OOM: skip tracking (dup possible, not fatal)
        v->inflight = p; v->inflight_cap = nc;
    }
    v->inflight[v->inflight_n++] = key;
}
static void inflight_del(mc_volume *v, uint64_t key) {
    for (int i = 0; i < v->inflight_n; ++i)
        if (v->inflight[i] == key) {                  // swap-remove
            v->inflight[i] = v->inflight[--v->inflight_n];
            return;
        }
}

// Deduping-stack membership set (open addressing, linear probe). key != 0 always
// (rkey packs lod in the high nibble; region 0,0,0 at lod>0 is fine, and lod0
// 0,0,0 -> key 0 would alias empty; guard below treats key 0 specially -- but
// rkey for (0,0,0,0) is 0, which never occurs as a real interactive request).
static int rs_set_has(mc_volume *v, uint64_t key) {
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (uint32_t)v->rs_set_mask;
    for (int p = 0; p <= v->rs_set_mask; ++p) {
        uint64_t cur = v->rs_set[i];
        if (cur == 0) return 0;
        if (cur == key) return 1;
        i = (i + 1) & (uint32_t)v->rs_set_mask;
    }
    return 0;
}
static void rs_set_add(mc_volume *v, uint64_t key) {
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (uint32_t)v->rs_set_mask;
    for (int p = 0; p <= v->rs_set_mask; ++p) {
        if (v->rs_set[i] == 0 || v->rs_set[i] == key) { v->rs_set[i] = key; return; }
        i = (i + 1) & (uint32_t)v->rs_set_mask;
    }
}
// Rebuild the set from the stack (after a removal leaves probe-chain holes).
static void rs_set_rebuild(mc_volume *v) {
    memset(v->rs_set, 0, ((size_t)v->rs_set_mask + 1) * sizeof(uint64_t));
    for (int i = 0; i < v->rs_n; ++i) rs_set_add(v, v->reqstk[i]);
}

static void req_push(mc_volume *v, int lod, int cz, int cy, int cx) {
    uint64_t key = rkey(lod, cz, cy, cx);
    pthread_mutex_lock(&v->mu);
    if (inflight_has(v, key)) { pthread_mutex_unlock(&v->mu); return; }  // already fetching/decoding
    if (rs_set_has(v, key)) { pthread_mutex_unlock(&v->mu); return; }    // already queued (O(1))
    if (v->rs_n == v->rs_cap) {                         // full -> drop bottom (stalest)
        memmove(&v->reqstk[0], &v->reqstk[1], (size_t)(v->rs_cap - 1) * sizeof(uint64_t));
        v->rs_n--;
        rs_set_rebuild(v);                              // membership shifted; rebuild
    }
    v->reqstk[v->rs_n++] = key;                         // push on top
    rs_set_add(v, key);
    MCVLOG("req_push  lod%d region(%d,%d,%d) stack_depth=%d", lod, cz, cy, cx, v->rs_n);
    pthread_cond_signal(&v->cv);
    pthread_mutex_unlock(&v->mu);
}

enum { DL_BATCH = 64 };   // per-batch regions; streaming GETs run this deep

// Streaming fetch of ONE 256^3 region via the reader (serial; the local-file
// source path and the fallback for a header window too small to parse).
static void mc_stream_fetch_region(mc_volume *v, int lod, int rz, int ry, int rx) {
    if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) return;  // already have it
    int rerr = 0;
    uint64_t off = mc_chunk_offset_chk(v->rd, lod, rz, ry, rx, &rerr);
    if (rerr) return;                                  // transient: leave ABSENT, retry
    if (off <= MC_SLOT_ZERO) {                          // CONFIRMED air (absent OR remote
        // ZERO slot) -> record a local ZERO region. Without this, a remote MC_SLOT_ZERO
        // (==1) chunk -- the masked margin at mid LODs -- stayed ABSENT locally, so the
        // render's LOD fallback walked PAST it to a coarser PRESENT level and sampled
        // the neighbor data bled in by unmasked pyramid downsampling (junk in the void).
        if (mc_archive_append_chunk_raw(v->arc, lod, rz, ry, rx, zero256()) == 0)
            vol_mark_region(v, lod, rz, ry, rx);
        return;
    }
    uint64_t blen = mc_reader_chunk_blob_len(v->rd, off);
    if (blen == 0) return;                             // transient: leave ABSENT, retry
    uint8_t *blob = malloc((size_t)blen);
    if (!blob) return;
    if (mc_reader_read_blob(v->rd, off, (size_t)blen, blob) != 0) { free(blob); return; }
    atomic_fetch_add_explicit(&v->net_bytes, blen, memory_order_relaxed);
    if (mc_archive_append_chunk_compressed(v->arc, lod, rz, ry, rx, blob, (size_t)blen) == 0)
        vol_mark_region(v, lod, rz, ry, rx);           // else leave absent -> refetch
    free(blob);
}

// Blob total length parsed from its LEADING bytes (header + fmap + bitmap + len
// table must all be inside `buf`). 0 = window too short (caller falls back to the
// exact serial path). Mirrors mc_reader_chunk_blob_len, but over one buffer.
static uint64_t mc_blob_len_parse(const uint8_t *buf, size_t len) {
    if (len < MC_BLOB_HDR) return 0;
    uint16_t fml; memcpy(&fml, buf + MC_BLOB_HDR - 2, 2);
    uint64_t bm_off = (uint64_t)MC_BLOB_HDR + fml;
    if (len < bm_off + MC_BITMAP_BYTES) return 0;
    int np = 0;
    for (int i = 0; i < MC_BITMAP_BYTES; ++i) np += __builtin_popcount(buf[bm_off + i]);
    if (!np) return bm_off + MC_BITMAP_BYTES;
    if (len < bm_off + MC_BITMAP_BYTES + (size_t)np * 2) return 0;
    uint64_t pay = 0;
    for (int i = 0; i < np; ++i) {
        uint16_t l; memcpy(&l, buf + bm_off + MC_BITMAP_BYTES + (size_t)i * 2, 2);
        pay += l;
    }
    return bm_off + MC_BITMAP_BYTES + (uint64_t)np * 2 + pay;
}

// Streaming fetch of a BATCH of regions -- the throughput path. The serial
// per-chunk flow (resolve, header probe, length probe, blob read) is 3-4 S3
// round-trips each; one thread doing that tops out ~2MB/s. Instead: resolve all
// offsets via the reader (node tables memoized, cheap), then TWO rounds of
// s3_get_batch (32-way concurrent ranged GETs, like the zarr path): round A pulls
// each blob's leading bytes -- sized at ~2x the running average blob size, so most
// blobs complete in this single GET (exact length parsed from the leading bytes)
// without over-reading small ones; round B pulls the occasional tail.
#define MC_STREAM_HDR_MIN (64u << 10)
#define MC_STREAM_HDR_MAX (2u << 20)
static void mc_stream_fetch_batch(mc_volume *v, int m, const int *lods,
                                  const int *rz, const int *ry, const int *rx) {
    uint64_t off[DL_BATCH];
    int act[DL_BATCH], na = 0;                 // regions that still need bytes
    for (int i = 0; i < m; ++i) {
        if (mc_archive_chunk_coverage(v->arc, lods[i], rz[i], ry[i], rx[i]) != MC_ABSENT)
            continue;                                          // already resident
        int rerr = 0;
        uint64_t o = mc_chunk_offset_chk(v->rd, lods[i], rz[i], ry[i], rx[i], &rerr);
        if (rerr) continue;                                    // transient: leave ABSENT, retry
        if (o <= MC_SLOT_ZERO) {                               // CONFIRMED air (absent OR remote
            // ZERO slot) -> local ZERO region. A remote MC_SLOT_ZERO must record locally
            // as ZERO, else the masked margin stays ABSENT and the LOD fallback bleeds
            // coarser PRESENT data into the void.
            if (mc_archive_append_chunk_raw(v->arc, lods[i], rz[i], ry[i], rx[i], zero256()) == 0)
                vol_mark_region(v, lods[i], rz[i], ry[i], rx[i]);  // else leave absent -> refetch
            continue;
        }
        off[i] = o; act[na++] = i;
    }
    if (!na) return;

    if (v->local || !v->s3mca) {               // local-file source: disk is fast, go serial
        for (int k = 0; k < na; ++k)
            mc_stream_fetch_region(v, lods[act[k]], rz[act[k]], ry[act[k]], rx[act[k]]);
        return;
    }

    // Chunks are appended contiguously in the source archive, and the requested
    // regions are spatially coherent (a viewport) -- so their blobs cluster.
    // Sort by offset and COALESCE into a few LARGE sequential ranges (read through
    // small gaps), then carve each blob out of the run buffer. A handful of
    // multi-MB GETs reaches S3 large-object throughput where per-chunk ~200KB
    // GETs stall in TCP slow-start. Interior blob lengths are exact (next_off -
    // off bounds it; the parse is authoritative); each run's last blob gets an
    // EMA-sized margin, with a (rare) follow-up GET if the parse says longer.
    if (!v->s_blob_ema) v->s_blob_ema = 256u << 10;          // first batch: 256KB guess
    uint64_t margin = v->s_blob_ema * 2;
    if (margin < MC_STREAM_HDR_MIN) margin = MC_STREAM_HDR_MIN;
    if (margin > MC_STREAM_HDR_MAX) margin = MC_STREAM_HDR_MAX;

    // sort act[] by offset (insertion sort; na <= DL_BATCH).
    for (int a = 1; a < na; ++a) {
        int t = act[a]; int b = a - 1;
        while (b >= 0 && off[act[b]] > off[t]) { act[b + 1] = act[b]; --b; }
        act[b + 1] = t;
    }

    enum { RUN_GAP = 512 << 10, RUN_MAX = 64 << 20 };  // read-through gap / run size cap:
    // merge only near-adjacent blobs (raster archives put y/z neighbors MBs apart;
    // reading through those gaps wasted ~3x the useful bytes)
    s3_range_req runq[DL_BATCH]; s3_response runr[DL_BATCH];
    int rs[DL_BATCH], re[DL_BATCH], nrun = 0;          // act[] index span of each run
    for (int k = 0; k < na;) {
        int s = k, e = k;
        uint64_t end = off[act[k]] + margin;
        while (e + 1 < na &&
               off[act[e + 1]] <= end + RUN_GAP &&
               off[act[e + 1]] + margin - off[act[s]] <= RUN_MAX) {
            ++e;
            end = off[act[e]] + margin;
        }
        runq[nrun] = (s3_range_req){v->s3mca->url, off[act[s]], end - off[act[s]]};
        rs[nrun] = s; re[nrun] = e; ++nrun;
        k = e + 1;
    }

    memset(runr, 0, sizeof(s3_response) * (size_t)nrun);
    s3_get_batch(v->s3mca->cl, runq, (size_t)nrun, 16, runr);

    for (int r = 0; r < nrun; ++r) {
        if (!s3_response_ok(&runr[r]) || !runr[r].body_len) {  // transient: retry later
            s3_response_free(&runr[r]);
            continue;
        }
        atomic_fetch_add_explicit(&v->net_bytes, runr[r].body_len, memory_order_relaxed);
        for (int k = rs[r]; k <= re[r]; ++k) {
            int i = act[k];
            uint64_t rel = off[i] - runq[r].offset;
            if (rel >= runr[r].body_len) continue;             // run came back short
            size_t avail = runr[r].body_len - (size_t)rel;
            uint64_t bl = mc_blob_len_parse(runr[r].body + rel, avail);
            if (!bl) {                                         // window short of the header
                mc_stream_fetch_region(v, lods[i], rz[i], ry[i], rx[i]);
                continue;
            }
            v->s_blob_ema = (v->s_blob_ema * 7 + (size_t)bl) / 8;   // dl thread only
            if (bl <= avail) {                                 // whole blob in the run
                if (mc_archive_append_chunk_compressed(v->arc, lods[i], rz[i], ry[i], rx[i],
                                                       runr[r].body + rel, (size_t)bl) == 0)
                    vol_mark_region(v, lods[i], rz[i], ry[i], rx[i]);   // else leave absent -> refetch
            } else {                                           // run-edge tail: one follow-up GET
                s3_response tail = {0};
                if (s3_get_range(v->s3mca->cl, v->s3mca->url, off[i] + avail, bl - avail,
                                 &tail) == S3_OK &&
                    s3_response_ok(&tail) && tail.body_len >= bl - avail) {
                    atomic_fetch_add_explicit(&v->net_bytes, tail.body_len, memory_order_relaxed);
                    uint8_t *blob = malloc((size_t)bl);
                    if (blob) {
                        memcpy(blob, runr[r].body + rel, avail);
                        memcpy(blob + avail, tail.body, (size_t)(bl - avail));
                        if (mc_archive_append_chunk_compressed(v->arc, lods[i], rz[i], ry[i], rx[i],
                                                               blob, (size_t)bl) == 0)
                            vol_mark_region(v, lods[i], rz[i], ry[i], rx[i]);
                        free(blob);
                    }
                }
                s3_response_free(&tail);
            }
        }
        s3_response_free(&runr[r]);
    }
}

// Download thread: pop the newest request, download its shard (-> decode queue).
// Drain up to DL_BATCH region requests from the LIFO stack (newest first) and
// fetch their source chunks in ONE s3_get_batch — per-region precision (no
// whole-shard pull) AND high concurrency (32 GETs over the pooled connections),
// which is what saturates the link. c3d (sub=1) = one chunk per region; v2
// regions fall back to the per-chunk cube path. STREAMING: one thread drives
// batched concurrent GETs (mc_stream_fetch_batch) -- verbatim copy, no decode pool.
static void *dl_main(void *ud) {
    mc_volume *v = ud;
    mc_thread_setname("mc-download");      // distinguish in profilers
    for (;;) {
        int lods[DL_BATCH], rz[DL_BATCH], ry[DL_BATCH], rx[DL_BATCH];
        int m = 0;
        pthread_mutex_lock(&v->mu);
        while (v->rs_n == 0 && !v->stop) pthread_cond_wait(&v->cv, &v->mu);
        if (v->stop && v->rs_n == 0) { pthread_mutex_unlock(&v->mu); return NULL; }
        int popped = 0;
        while (m < DL_BATCH && v->rs_n > 0) {           // grab a batch, newest first
            uint64_t key = v->reqstk[--v->rs_n];
            popped = 1;
            if (inflight_has(v, key)) continue;          // another dl thread already has it
            inflight_add(v, key);                        // single-flight: claim it
            runpack(key, &lods[m], &rz[m], &ry[m], &rx[m]);
            ++m;
        }
        if (popped) rs_set_rebuild(v);                  // popped keys leave the set
        pthread_mutex_unlock(&v->mu);

        if (v->streaming) {
            // Verbatim .mca -> .mca copy, batched (two rounds of 32-way GETs).
            mc_stream_fetch_batch(v, m, lods, rz, ry, rx);
            pthread_mutex_lock(&v->mu);
            for (int i = 0; i < m; ++i)
                inflight_del(v, rkey(lods[i], rz[i], ry[i], rx[i]));
            pthread_mutex_unlock(&v->mu);
            if (m && v->ready_cb) v->ready_cb(v->ready_ud);
            continue;
        }

        // Locate each region's c3d source chunk via the cached footer; build one
        // batched ranged-GET. v2 (sub>1) and local both use the per-region path.
        s3_range_req reqs[DL_BATCH];
        s3_response  resp[DL_BATCH];
        char urls[DL_BATCH][1280];
        int ri[DL_BATCH], nq = 0;                       // ri[q] -> request index
        // Each of the m claimed regions is in-flight. It is cleared exactly when
        // it does NOT produce a decode_item here (decode_one clears the ones that
        // do). clr() drops the single-flight claim for region i so a later miss
        // re-requests it.
        #define DL_CLR(i) do { uint64_t _k = rkey(lods[i], rz[i], ry[i], rx[i]); \
            pthread_mutex_lock(&v->mu); inflight_del(v, _k); pthread_mutex_unlock(&v->mu); } while (0)
        for (int i = 0; i < m; ++i) {
            mc_zarr *z = v->lv[lods[i]].z;
            const int sub = CHUNK / mc_zarr_inner_edge(z);
            if (v->local || sub != 1) {                 // local / v2: direct per-region
                mc_volume_prefetch_region(v, lods[i], rz[i]*sub, ry[i]*sub, rx[i]*sub);
                DL_CLR(i);                               // prefetch_region appends synchronously
                continue;
            }
            if (mc_archive_chunk_coverage(v->arc, lods[i], rz[i], ry[i], rx[i]) != MC_ABSENT)
                { DL_CLR(i); continue; }                 // already resident
            char key[64]; uint64_t off, nb;
            int st = mc_zarr_chunk_locate(z, rz[i], ry[i], rx[i], key, &off, &nb);
            if (st < 0)                                 // transient read error: leave ABSENT
                { DL_CLR(i); continue; }                 // (retry on next miss), do NOT mark air
            if (st > 0) {                               // CONFIRMED air (footer ok + air marker)
                decode_item air = {lods[i], rz[i], ry[i], rx[i], 1, 0, {0},{0},{0}, {0},{0}};
                decode_push(v, &air);                    // decode_one clears in-flight
                continue;
            }
            snprintf(urls[nq], sizeof urls[nq], "%s/%s", v->lv[lods[i]].prefix, key);
            reqs[nq] = (s3_range_req){urls[nq], off, nb};
            ri[nq] = i;
            ++nq;
        }
        if (nq == 0) continue;
        MCVLOG("dl_batch  %d regions -> %d ranged GETs", m, nq);
        memset(resp, 0, sizeof(s3_response) * nq);
        s3_get_batch(v->s3, reqs, (size_t)nq, 32, resp);
        for (int q = 0; q < nq; ++q) {
            int i = ri[q];
            int pushed = 0;
            if (s3_response_ok(&resp[q]) && resp[q].body_len >= reqs[q].length && reqs[q].length) {
                uint8_t *raw = malloc(reqs[q].length);
                if (raw) {
                    memcpy(raw, resp[q].body, reqs[q].length);
                    atomic_fetch_add_explicit(&v->net_bytes, reqs[q].length, memory_order_relaxed);
                    decode_item it = {lods[i], rz[i], ry[i], rx[i], 1, 1, {0},{0},{0}, {0},{0}};
                    it.raw[0] = raw; it.rlen[0] = reqs[q].length;
                    decode_push(v, &it);                 // decode_one clears in-flight
                    pushed = 1;
                }
            }
            if (!pushed) DL_CLR(i);                      // download failed -> retry next miss
            s3_response_free(&resp[q]);
        }
        #undef DL_CLR
    }
}

// Blocking fill of one region (get_block / CLI): download its shard synchronously
// through the same decode queue, then wait for that region's coverage to resolve.
static mc_cover ensure_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    mc_cover cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
    if (cov != MC_ABSENT) return cov;
    if (v->streaming) {
        req_push(v, lod, cz, cy, cx);          // dl thread copies the blob verbatim
    } else {
        const int sub = CHUNK / mc_zarr_inner_edge(v->lv[lod].z);
        mc_volume_prefetch_region(v, lod, cz * sub, cy * sub, cx * sub);  // just this region
    }
    // wait for the downloaders/decoders to drain enough that this region is covered.
    for (int spin = 0; spin < 100000; ++spin) {
        cov = mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
        if (cov != MC_ABSENT) return cov;
        struct timespec ts = {0, 1000000};             // 1ms
        nanosleep(&ts, NULL);
    }
    return mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
}

// ---------------------------------------------------------------------------
// shared 32-aligned zero region (for air)
// ---------------------------------------------------------------------------
static uint8_t *g_zero = NULL;
static void init_zero(void) {
    if (posix_memalign((void **)&g_zero, 64, (size_t)CHUNK * CHUNK * CHUNK) == 0)
        memset(g_zero, 0, (size_t)CHUNK * CHUNK * CHUNK);
}
static const uint8_t *zero256(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init_zero);
    return g_zero;
}

// ===========================================================================
// open / discovery
// ===========================================================================

// strip a trailing '/'.
static void rstrip_slash(char *s) {
    size_t n = strlen(s);
    while (n && s[n - 1] == '/') s[--n] = 0;
}

mc_volume *mc_volume_open(const char *url, const char *cache_dir,
                          size_t cache_bytes, float quality) {
    return mc_volume_open_ex(url, cache_dir, cache_bytes, quality, NULL);
}

mc_volume *mc_volume_open_ex(const char *url, const char *cache_dir,
                             size_t cache_bytes, float quality,
                             const mc_volume_config *cfg) {
    if (!url || !cache_dir) return NULL;
    mc_volume *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    v->quality = quality;
    atomic_init(&v->net_bytes, 0);
    pthread_mutex_init(&v->mu, NULL);
    pthread_cond_init(&v->cv, NULL);

    // Local vs remote: a URL with a "scheme://" is remote (s3/https), otherwise
    // `url` is a local filesystem directory read via file_read (no S3 client).
    v->local = (strstr(url, "://") == NULL);
    if (!v->local) {
        // s3 client: full credential resolution (profile/IMDS/SSO/env), else anonymous.
        s3_config cfg = {0};
        s3_credentials creds = {0};
        if (s3_credentials_load(NULL, &creds) == S3_OK) cfg.creds = creds;
        v->s3 = s3_client_new(&cfg);
        s3_credentials_free(&creds);
        if (!v->s3) { free(v); return NULL; }
    }
    mc_zarr_read_fn read_cb = v->local ? file_read : s3_read;

    snprintf(v->root, sizeof v->root, "%s", url);
    rstrip_slash(v->root);

    // discover levels: probe "<root>/<i>/zarr.json" for i=0.. until a gap.
    for (int i = 0; i < MAXLOD; ++i) {
        level_t *lv = &v->lv[i];
        lv->vol = v;
        snprintf(lv->prefix, sizeof lv->prefix, "%s/%d", v->root, i);
        mc_zarr *z = mc_zarr_open(read_cb, lv);
        if (!z) { lv->prefix[0] = 0; break; }
        lv->z = z;
        v->nlods = i + 1;
    }
    if (v->nlods == 0) { if (v->s3) s3_client_free(v->s3); free(v); return NULL; }

    // local .mca dims from LOD0 shape (padded to 256 internally by mc).
    int nz, ny, nx;
    mc_zarr_shape(v->lv[0].z, &nz, &ny, &nx);
    char path[2048];
    // archive name from the last path component of the root.
    const char *base = strrchr(v->root, '/');
    base = base ? base + 1 : v->root;
    snprintf(path, sizeof path, "%s/%s.mca", cache_dir, base);
    v->arc = mc_archive_open_dims(path, nx, ny, nz, quality);
    if (!v->arc) {
        for (int i = 0; i < v->nlods; ++i) mc_zarr_free(v->lv[i].z);
        s3_client_free(v->s3); free(v); return NULL;
    }
    v->cache = mc_cache_new_archive(cache_bytes, v->arc);
    if (!v->cache) {
        mc_archive_close(v->arc);
        for (int i = 0; i < v->nlods; ++i) mc_zarr_free(v->lv[i].z);
        s3_client_free(v->s3); free(v); return NULL;
    }

    // Pipeline: a few download threads (network-bound, pop the LIFO request
    // stack) feed a bounded decode queue drained by a decode pool (CPU-bound).
    pthread_cond_init(&v->dq_ne, NULL);
    pthread_cond_init(&v->dq_nf, NULL);
    // Staging queue: downloaded compressed chunks wait here for the (CPU-bound)
    // decode pool. Backpressure is BYTE-budgeted (default 2GB) so downloads run
    // far ahead of decode and the network stays saturated, decoupling the two.
    // The ring slot count just needs to exceed however many items fit in the
    // budget — size it from the budget assuming small (~64KB) chunks, capped.
    v->dq_byte_budget = (cfg && cfg->staging_bytes > 0) ? cfg->staging_bytes
                                                        : (2ull << 30);   // 2 GB
    if (v->dq_byte_budget < (64ull << 20)) v->dq_byte_budget = 64ull << 20;
    v->dq_bytes = 0;
    v->dq_cap = (int)(v->dq_byte_budget / (64ull << 10)) + 8;   // ~budget/64KB slots
    if (v->dq_cap < 256) v->dq_cap = 256; if (v->dq_cap > 131072) v->dq_cap = 131072;
    v->dq = calloc((size_t)v->dq_cap, sizeof *v->dq);
    // LIFO request stack: just 8-byte region keys, so it can be large — a fast
    // navigation enqueues many thousands of on-screen region misses, and we
    // don't want to drop ones still in view. 64K keys = 512KB.
    v->rs_cap = (cfg && cfg->request_stack > 0) ? cfg->request_stack : 65536;
    if (v->rs_cap < 256) v->rs_cap = 256; if (v->rs_cap > (1<<22)) v->rs_cap = (1<<22);
    v->reqstk = calloc((size_t)v->rs_cap, sizeof *v->reqstk);
    // Membership set: next pow2 >= 2*rs_cap (load factor <= 0.5 -> short probes).
    int sc = 1; while (sc < v->rs_cap * 2) sc <<= 1;
    v->rs_set_mask = sc - 1;
    v->rs_set = calloc((size_t)sc, sizeof *v->rs_set);
    v->rgen = calloc(MC_RGEN_SLOTS, sizeof *v->rgen);

    // Decoders default to nproc/2: the c3d/wavelet decode is memory-bandwidth-
    // bound (a 256^3 decode streams ~16MB), so threads past ~half the cores
    // saturate the bus and only inflate per-decode latency (measured: 1T 103ms,
    // 8T 176ms, 16T 305ms — ~same throughput past 8 but 2x the latency). Caller
    // may override via mc_volume_config; clamp to the fixed pool arrays.
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    int nd = (cfg && cfg->decoders > 0) ? cfg->decoders
                                        : (nproc > 2 ? (int)(nproc / 2) : 2);
    if (nd < 1) nd = 1; if (nd > 32) nd = 32;          // decoders[32]
    v->ndecoders = nd;
    for (int i = 0; i < v->ndecoders; ++i)
        pthread_create(&v->decoders[i], NULL, decoder_main, v);
    fill_pool_init(v, nd);                              // archive->cache fill, off the GUI thread
    int ndl = (cfg && cfg->dl_threads > 0) ? cfg->dl_threads : 8;
    if (ndl < 1) ndl = 1; if (ndl > 16) ndl = 16;      // dlthreads[16]
    v->ndl = ndl;
    for (int i = 0; i < v->ndl; ++i)
        pthread_create(&v->dlthreads[i], NULL, dl_main, v);
    MCVLOG("open      %s  decoders=%d dl_threads=%d staging=%.1fGB(slots=%d) rs_cap=%d",
           url, v->ndecoders, v->ndl, v->dq_byte_budget / 1073741824.0, v->dq_cap, v->rs_cap);
    return v;
}

// mmap a local .mca read-only: the flat reader (mc_open) then treats it as one
// big array -- chunk resolves and blob reads are plain pointer reads, the kernel
// pages in on demand (nothing is slurped into RAM).
static uint8_t *mc_map_file_ro(const char *path, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return NULL; }
    uint8_t *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);                                   // the mapping keeps the file alive
    if (base == MAP_FAILED) return NULL;
    *out_len = (size_t)st.st_size;
    return base;
}

// Probe a built .mca's header WITHOUT opening a volume (no local archive, no
// threads): LOD0 dims, lod count, quality. url = s3://, https://, or a local
// path. Returns 0 on success.
int mc_mca_probe(const char *url, int *nx, int *ny, int *nz, int *nlods, float *quality) {
    if (!url) return -1;
    mc_s3 *s = NULL; mc_reader *r = NULL;
    uint8_t *map = NULL; size_t maplen = 0;
    if (strstr(url, "://")) {
        s = mc_s3_open(url);
        if (!s) return -1;
        r = mc_s3_reader(s);
    } else {
        map = mc_map_file_ro(url, &maplen);
        if (!map) return -1;
        r = mc_open(map, maplen);
        if (!r) { munmap(map, maplen); return -1; }
    }
    mc_reader_dims(r, nx, ny, nz);
    if (nlods) *nlods = mc_reader_nlods(r);
    if (quality) *quality = mc_reader_quality(r);
    if (s) mc_s3_close(s); else { mc_close(r); munmap(map, maplen); }
    return 0;
}

// Open an already-built remote (or local-file) .mca and stream it into a LOCAL
// .mca on demand. Same machinery as the zarr transcode path -- local archive,
// download threads, decode-from-local THAW -- but the download step COPIES the
// remote chunk's compressed blob verbatim onto the local archive (no decode, no
// re-encode): a 256^3 .mca chunk is already in the exact local format. We never
// mirror the whole remote; only the chunks the view touches get pulled+appended.
//   url        : the remote (s3://.../https://...) or local-file source .mca
//   cache_dir  : holds the local <name>.mca that fetched chunks append into
//   cache_bytes: resident RAM decoded-block budget
// Returns NULL on failure.
mc_volume *mc_volume_open_streaming(const char *url, const char *cache_dir,
                                    size_t cache_bytes) {
    if (!url || !cache_dir) return NULL;
    mc_volume *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    v->streaming = 1;
    atomic_init(&v->net_bytes, 0);
    pthread_mutex_init(&v->mu, NULL);
    pthread_cond_init(&v->cv, NULL);
    snprintf(v->root, sizeof v->root, "%s", url);
    rstrip_slash(v->root);

    // Open the SOURCE reader (remote ranged GETs, or a local file mmap'd read-only
    // and read like one big array) for chunk-offset resolves + verbatim blob reads.
    v->local = (strstr(url, "://") == NULL);
    if (v->local) {
        v->s_map = mc_map_file_ro(v->root, &v->s_map_len);
        if (!v->s_map) { free(v); return NULL; }
        v->rd = mc_open(v->s_map, v->s_map_len);
        if (!v->rd) { munmap(v->s_map, v->s_map_len); free(v); return NULL; }
    } else {
        v->s3mca = mc_s3_open(url);
        if (!v->s3mca) { free(v); return NULL; }
        v->rd = mc_s3_reader(v->s3mca);
    }

    int n0x, n0y, n0z;
    mc_reader_dims(v->rd, &n0x, &n0y, &n0z);
    v->nlods = mc_reader_nlods(v->rd);
    if (v->nlods > MAXLOD) v->nlods = MAXLOD;
    if (v->nlods <= 0) {
        if (v->s3mca) mc_s3_close(v->s3mca); else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v); return NULL;
    }
    v->quality = mc_reader_quality(v->rd);
    for (int l = 0; l < v->nlods; ++l) {
        v->s_nz[l] = n0z >> l < 1 ? 1 : n0z >> l;
        v->s_ny[l] = n0y >> l < 1 ? 1 : n0y >> l;
        v->s_nx[l] = n0x >> l < 1 ? 1 : n0x >> l;
    }

    // Local archive: fetched chunks append here verbatim. Same dims/quality as the
    // source so chunk coords + blob format line up exactly.
    char path[2048];
    const char *base = strrchr(v->root, '/');
    base = base ? base + 1 : v->root;
    snprintf(path, sizeof path, "%s/%s.local.mca", cache_dir, base);
    v->arc = mc_archive_open_dims(path, n0x, n0y, n0z, v->quality);
    if (!v->arc) {
        if (v->s3mca) mc_s3_close(v->s3mca); else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v); return NULL;
    }
    v->cache = mc_cache_new_archive(cache_bytes, v->arc);
    if (!v->cache) {
        mc_archive_close(v->arc);
        if (v->s3mca) mc_s3_close(v->s3mca); else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v); return NULL;
    }

    // Download threads only (no decode pool -- streaming appends compressed blobs
    // verbatim, there's nothing to decode off-thread). Reuse the LIFO request stack.
    v->rs_cap = 65536;
    v->reqstk = calloc((size_t)v->rs_cap, sizeof *v->reqstk);
    int sc = 1; while (sc < v->rs_cap * 2) sc <<= 1;
    v->rs_set_mask = sc - 1;
    v->rs_set = calloc((size_t)sc, sizeof *v->rs_set);
    v->rgen = calloc(MC_RGEN_SLOTS, sizeof *v->rgen);
    // ONE download thread: the source reader (cbuf + node-table cache + codec ctx)
    // is non-reentrant, so a single owner avoids any sharing. Chunks are large and
    // partial-fetch is efficient; this runs off the UI thread, which is the point.
    v->ndl = 1;
    for (int i = 0; i < v->ndl; ++i)
        pthread_create(&v->dlthreads[i], NULL, dl_main, v);
    // Archive->cache fill pool (decode local blobs off the GUI thread). Streaming has
    // no decode pool, so size from nproc/2.
    { long nproc = sysconf(_SC_NPROCESSORS_ONLN);
      fill_pool_init(v, nproc > 2 ? (int)(nproc / 2) : 2); }
    MCVLOG("open(stream) %s -> %s  nlods=%d dims=%dx%dx%d q=%.1f dl=%d fillers=%d",
           url, path, v->nlods, n0x, n0y, n0z, v->quality, v->ndl, v->nfillers);
    return v;
}

void mc_volume_free(mc_volume *v) {
    if (!v) return;
    if (v->streaming) {
        // Stop the download threads, then tear down cache + local archive + source
        // reader. No decode pool / zarr levels exist in streaming.
        pthread_mutex_lock(&v->mu);
        v->stop = 1;
        pthread_cond_broadcast(&v->cv);
        pthread_mutex_unlock(&v->mu);
        for (int i = 0; i < v->ndl; ++i) pthread_join(v->dlthreads[i], NULL);
        fill_pool_stop(v);                     // stop fillers before freeing the cache
        pthread_mutex_destroy(&v->fill_mu);
        pthread_cond_destroy(&v->fill_start); pthread_cond_destroy(&v->fill_fin);
        if (v->cache) mc_cache_free(v->cache);
        if (v->arc) mc_archive_close(v->arc);
        if (v->s3mca) mc_s3_close(v->s3mca);   // owns the remote reader
        else { mc_close(v->rd); if (v->s_map) munmap(v->s_map, v->s_map_len); }
        free(v->reqstk);
        free(v->rs_set);
        free((void *)v->rgen);
        free(v->inflight);
        pthread_mutex_destroy(&v->mu);
        pthread_cond_destroy(&v->cv);
        free(v);
        return;
    }
    // stop download + decode threads.
    pthread_mutex_lock(&v->mu);
    v->stop = 1;
    pthread_cond_broadcast(&v->cv);      // wake download threads
    pthread_cond_broadcast(&v->dq_ne);   // wake decoders
    pthread_cond_broadcast(&v->dq_nf);   // wake blocked producers
    pthread_mutex_unlock(&v->mu);
    for (int i = 0; i < v->ndl; ++i) pthread_join(v->dlthreads[i], NULL);
    for (int i = 0; i < v->ndecoders; ++i) pthread_join(v->decoders[i], NULL);
    // drain any remaining decode items (free their raw buffers).
    while (v->dq_head != v->dq_tail) {
        decode_item *it = &v->dq[v->dq_head];
        for (int k = 0; k < it->nsub; ++k) free(it->raw[k]);
        v->dq_head = (v->dq_head + 1) % v->dq_cap;
    }
    pthread_cond_destroy(&v->dq_ne);
    pthread_cond_destroy(&v->dq_nf);
    fill_pool_stop(v);                         // stop fillers before freeing the cache
    pthread_mutex_destroy(&v->fill_mu);
    pthread_cond_destroy(&v->fill_start); pthread_cond_destroy(&v->fill_fin);
    if (v->cache) mc_cache_free(v->cache);
    if (v->arc) mc_archive_close(v->arc);
    for (int i = 0; i < v->nlods; ++i) if (v->lv[i].z) mc_zarr_free(v->lv[i].z);
    if (v->s3) s3_client_free(v->s3);
    free(v->dq);
    free(v->reqstk);
    free(v->rs_set);
    free((void *)v->rgen);
    free(v->inflight);
    pthread_mutex_destroy(&v->mu);
    pthread_cond_destroy(&v->cv);
    free(v);
}

int  mc_volume_nlods(const mc_volume *v) { return v ? v->nlods : 0; }
void mc_volume_shape(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    if (v->streaming) {
        if (lod < 0 || lod >= v->nlods) { if(nz)*nz=0; if(ny)*ny=0; if(nx)*nx=0; return; }
        if (nz) *nz = v->s_nz[lod]; if (ny) *ny = v->s_ny[lod]; if (nx) *nx = v->s_nx[lod];
        return;
    }
    mc_zarr_shape(v->lv[lod].z, nz, ny, nx);
}
void mc_volume_block_grid(const mc_volume *v, int lod, int *nz, int *ny, int *nx) {
    int sz, sy, sx;
    mc_volume_shape(v, lod, &sz, &sy, &sx);
    if (nz) *nz = (sz + BLK - 1) / BLK;
    if (ny) *ny = (sy + BLK - 1) / BLK;
    if (nx) *nx = (sx + BLK - 1) / BLK;
}
int mc_volume_get_level_meta(const mc_volume *v, int lod, mc_volume_level_meta *out) {
    if (!v || !out || lod < 0 || lod >= v->nlods) return -1;
    if (v->streaming) {
        mc_volume_shape(v, lod, &out->shape[0], &out->shape[1], &out->shape[2]);
        out->inner_edge = CHUNK; out->shard_edge = CHUNK;
        snprintf(out->codec, sizeof out->codec, "mc");
        return 0;
    }
    const mc_zarr *z = v->lv[lod].z;
    mc_zarr_shape(z, &out->shape[0], &out->shape[1], &out->shape[2]);
    out->inner_edge = mc_zarr_inner_edge(z);
    out->shard_edge = mc_zarr_shard_edge(z);
    const char *c = mc_zarr_inner_codec(z);
    snprintf(out->codec, sizeof out->codec, "%s", c ? c : "");
    return 0;
}

// ---------------------------------------------------------------------------
// block serving
// ---------------------------------------------------------------------------
// Both paths (zarr transcode AND remote-.mca streaming) serve from the LOCAL
// archive, so coverage is just the local archive's: ABSENT -> the download stack
// pulls the chunk (re-encode for zarr / verbatim copy for streaming).
static inline mc_cover vol_coverage(mc_volume *v, int lod, int cz, int cy, int cx) {
    return mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx);
}

int mc_volume_try_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return 0; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = vol_coverage(v, lod, cz, cy, cx);
    if (cov == MC_ABSENT) {
        // Record the absent miss at REGION granularity (the region's corner block),
        // NOT per 16^3 block. A thin slice touches thousands of blocks across an
        // absent region but they all dedupe to one region key -- so the per-frame
        // miss set is ~tens (one per absent region) instead of ~thousands. THAW
        // drains these region keys and issues one download each. (Per-block absent
        // recording flooded the 64K miss table every frame -> the ~2ms thaw floor.)
        mc_cache_miss_mark(v->cache, lod, cz * PER, cy * PER, cx * PER);
        memset(dst, 0, BLK * BLK * BLK);
        return 0;
    }
    if (cov == MC_ZERO) { memset(dst, 0, BLK * BLK * BLK); return 1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

// Shared 16^3 zero block (air). One static, read-only -- samplers point at it
// instead of each copying zeros into a per-entry buffer.
static const uint8_t *mc_zero16(void) {
    static uint8_t z[BLK * BLK * BLK];   // zero-initialized (BSS)
    return z;
}

// Pointer-returning block accessor (NO copy). Returns a STABLE pointer valid for
// the duration of a frozen frame: into the cache arena (RAM hit), the shared zero
// block (air), or NULL (absent or present-but-not-cached -> caller falls coarser).
// This replaces the copy-into-tmp path so the sampler memo can hold pointers, not
// 4KB buffers -- killing the per-frame 8MB sampler alloc + a per-block memcpy.
const uint8_t *mc_volume_block_ptr(mc_volume *v, int lod, int bz, int by, int bx) {
    if (lod < 0 || lod >= v->nlods) return NULL;
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = vol_coverage(v, lod, cz, cy, cx);
    if (cov == MC_ZERO) return mc_zero16();
    if (cov == MC_ABSENT) {       // build path only; streaming never returns ABSENT
        mc_cache_miss_mark(v->cache, lod, cz * PER, cy * PER, cx * PER);
        return NULL;
    }
    // present: arena pointer on a RAM hit; NULL (+ recorded miss) otherwise.
    return mc_cache_get(v->cache, lod, bz, by, bx);
}

// Predictive prefetch: request the 256^3 REGION (cz,cy,cx) be downloaded +
// transcoded if it isn't already resident/in-flight. Cheap and non-blocking: a
// coverage probe (O(1) memo) + a push onto the LIFO download stack (deduped vs
// in-flight/queued). No decode, no scratch. Present/air -> no-op. Called from the
// tick BEFORE freeze, from the geometry-predicted working set, so regions stream
// in ahead of the render instead of being discovered as misses a cycle late.
void mc_volume_request_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (!v || lod < 0 || lod >= v->nlods) return;
    if (mc_archive_chunk_coverage(v->arc, lod, cz, cy, cx) != MC_ABSENT) return;
    req_push(v, lod, cz, cy, cx);   // download thread fetches+appends (zarr or stream)
}

int mc_volume_get_block(mc_volume *v, int lod, int bz, int by, int bx, uint8_t *dst) {
    if (lod < 0 || lod >= v->nlods) { memset(dst, 0, BLK * BLK * BLK); return -1; }
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = ensure_region(v, lod, cz, cy, cx);   // pulls the region if ABSENT
    if (cov == MC_ZERO || cov == MC_ABSENT) { memset(dst, 0, BLK * BLK * BLK); return cov == MC_ZERO ? 0 : -1; }
    mc_cache_get_copy(v->cache, lod, bz, by, bx, dst);
    return 1;
}

// ---------------------------------------------------------------------------
// sampling source
// ---------------------------------------------------------------------------
static const uint8_t *vol_block(const mc_sample_src *src,
                                int bz, int by, int bx, uint8_t *tmp) {
    mc_volume *v = src->ud;
    if (src->aux2) {   // blocking (CLI): keep the copy-into-tmp path
        int r = mc_volume_get_block(v, src->aux, bz, by, bx, tmp);
        return r == 1 ? tmp : NULL;
    }
    // interactive: return a STABLE arena/zero pointer (no copy). tmp unused.
    (void)tmp;
    return mc_volume_block_ptr(v, src->aux, bz, by, bx);
}

// CHEAP residency for the LOD fallback: sample-able NOW without a decode? Yes iff
// the region is air (ZERO -> samples 0 trivially) or the 16^3 block is already in
// the RAM cache. A PRESENT-but-not-cached block returns 0 here so the fallback
// samples a coarser resident level instead of decoding the fine block on the
// render thread (also records the miss so THAW fills it -> next frame sharpens).
static int vol_block_resident(const mc_sample_src *src, int bz, int by, int bx) {
    mc_volume *v = src->ud;
    int lod = src->aux;
    int cz = bz / PER, cy = by / PER, cx = bx / PER;
    mc_cover cov = vol_coverage(v, lod, cz, cy, cx);
    if (cov == MC_ZERO) return 1;                        // air: trivially sample-able
    if (cov != MC_PRESENT) {                             // absent: record + fall coarser
        mc_cache_miss_mark(v->cache, lod, cz * PER, cy * PER, cx * PER);
        return 0;
    }
    if (mc_cache_contains(v->cache, lod, bz, by, bx)) return 1;   // RAM hit
    mc_cache_miss_mark(v->cache, lod, bz, by, bx);       // present but uncached -> fill
    return 0;                                            // fall coarser this frame
}

mc_sample_src mc_volume_sample_src(mc_volume *v, int lod, int blocking) {
    mc_sample_src s = {0};
    s.ud = v; s.aux = lod; s.aux2 = blocking; s.block = vol_block;
    s.resident = vol_block_resident;
    s.owns_ptr = !blocking;   // interactive vol_block returns stable arena pointers
    mc_volume_shape(v, lod, &s.nz, &s.ny, &s.nx);
    return s;
}

mc_sample_lods mc_volume_sample_lods(mc_volume *v, int blocking) {
    mc_sample_lods ls = {0};
    ls.nlods = v->nlods < 8 ? v->nlods : 8;
    for (int l = 0; l < ls.nlods; l++)
        ls.lods[l] = mc_volume_sample_src(v, l, blocking);
    return ls;
}

// ---------------------------------------------------------------------------
// prefetch — batch a whole shard's present inner chunks in ONE parallel
// s3_get_batch (many concurrent GETs over pooled connections), then decode +
// assemble into 256^3 regions and append. This is the throughput path: the
// parallelism lives in libs3's connection pool, so a FEW prefetch driver
// threads saturate bandwidth without a thread-per-GET explosion. RAM is bounded
// by one shard's compressed chunks (a fraction of the decoded shard).
// (cz,cy,cx) is any source inner-chunk in the target shard.
// ---------------------------------------------------------------------------
// Download ONE 256^3 region's source chunk(s) and push it to the decode queue —
// the interactive path. The shard index (64KB footer) is read once and cached
// (footer_get / mc_zarr_read_inner), so this is: cached footer lookup + a single
// ranged GET per source chunk (1 for c3d, up to 8 for a v2 sub^3 cube). NO
// whole-shard download. (cz,cy,cx) = the source inner-chunk coord of the region.
static void mc_volume_prefetch_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    mc_zarr *z = v->lv[lod].z;
    const int edge = mc_zarr_inner_edge(z);            // 256 (c3d) or 128 (v2)
    const int sub = CHUNK / edge;                      // source chunks per region axis
    const int rz = cz / sub, ry = cy / sub, rx = cx / sub;   // 256^3 region coords
    if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) return;  // already have it

    if (sub == 1) {                                    // c3d / v2-256: one chunk == region
        uint8_t *raw = NULL; size_t rlen = 0;
        int st = mc_zarr_read_inner(z, cz, cy, cx, &raw, &rlen);
        if (st < 0) { free(raw); return; }             // transient error -> leave ABSENT, retry
        if (st > 0 || !raw || !rlen) {                 // CONFIRMED air -> ZERO region
            free(raw);
            decode_item air = {lod, rz, ry, rx, 1, 0, {0},{0},{0}, {0},{0}};
            decode_push(v, &air);
            return;
        }
        atomic_fetch_add_explicit(&v->net_bytes, rlen, memory_order_relaxed);
        decode_item it = {lod, rz, ry, rx, 1, 1, {0},{0},{0}, {0},{0}};
        it.raw[0] = raw; it.rlen[0] = rlen;
        decode_push(v, &it);
        return;
    }

    // v2 sub^3 cube: fetch each present source chunk of the region's cube. A
    // transient read error on ANY sub-chunk aborts the whole region (leave it
    // ABSENT to retry) rather than recording a partial/air result.
    int sz0 = rz * sub, sy0 = ry * sub, sx0 = rx * sub;
    decode_item it = {lod, rz, ry, rx, sub, 0, {0},{0},{0}, {0},{0}};
    for (int dz = 0; dz < sub; ++dz)
    for (int dy = 0; dy < sub; ++dy)
    for (int dx = 0; dx < sub; ++dx) {
        uint8_t *raw = NULL; size_t rlen = 0;
        int st = mc_zarr_read_inner(z, sz0 + dz, sy0 + dy, sx0 + dx, &raw, &rlen);
        if (st < 0) {                                  // transient -> abort, retry region
            for (int k = 0; k < it.nsub; ++k) free(it.raw[k]);
            free(raw);
            return;
        }
        if (st > 0 || !raw || !rlen) { free(raw); continue; }   // this sub-chunk confirmed air
        atomic_fetch_add_explicit(&v->net_bytes, rlen, memory_order_relaxed);
        int k = it.nsub++;
        it.raw[k] = raw; it.rlen[k] = rlen;
        it.oz[k] = dz * edge; it.oy[k] = dy * edge; it.ox[k] = dx * edge;
    }
    decode_push(v, &it);                               // nsub==0 -> all sub-chunks air -> ZERO
}

// Download a shard's present chunks (one parallel s3_get_batch) and PUSH each
// region's raw payload(s) to the decode queue — NO decode on this (download)
// thread. Decoders drain the queue in parallel, so the network stays saturated.
// Backpressure in decode_push bounds RAM. (cz,cy,cx) = source inner-chunk coord.
// NOTE: bulk path (CLI prefetch). The interactive render path uses
// mc_volume_prefetch_region (one region) so navigation never pulls a whole shard.
void mc_volume_prefetch_shard(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (lod < 0 || lod >= v->nlods) return;
    if (v->streaming) return;   // no shard/zarr layer; THAW read-through fills on miss
    level_t *lv = &v->lv[lod];
    mc_zarr *z = lv->z;
    const int edge = mc_zarr_inner_edge(z);            // 256 (c3d) or 128 (v2)
    const int sub = CHUNK / edge;                      // source chunks per region axis

    char shard_key[64];
    mc_zarr_range *ranges = NULL;
    int nr = 0;
    double t0 = mcv_now();
    if (mc_zarr_shard_index(z, cz, cy, cx, shard_key, &ranges, &nr) < 0) {
        MCVLOG("shard_idx lod%d src(%d,%d,%d) FAILED", lod, cz, cy, cx); return;
    }
    MCVLOG("shard_idx lod%d src(%d,%d,%d) -> %d present chunks (footer %.0fms)",
           lod, cz, cy, cx, nr, mcv_now() - t0);
    if (nr == 0) { free(ranges); return; }             // all air

    char shard_url[1280];
    snprintf(shard_url, sizeof shard_url, "%s/%s", lv->prefix, shard_key);
    uint64_t got = 0;
    int nbatch = 0;

    // Download the shard's chunks in batches of MC_BATCH (bounded buffering),
    // then hand each region's raw bytes to the decode pool. v2 groups the sub^3
    // cube per region; c3d is 1:1.
    enum { MC_BATCH = 48 };
    s3_range_req reqs[MC_BATCH];
    s3_response resp[MC_BATCH];
    int idx[MC_BATCH];
    for (int base = 0; base < nr; ) {
        int nq = 0;
        while (base < nr && nq < MC_BATCH) {
            mc_zarr_range *rg = &ranges[base++];
            int rz = rg->cz / sub, ry = rg->cy / sub, rx = rg->cx / sub;
            if (mc_archive_chunk_coverage(v->arc, lod, rz, ry, rx) != MC_ABSENT) continue;
            reqs[nq] = (s3_range_req){shard_url, rg->off, rg->len};
            idx[nq] = base - 1;
            ++nq;
        }
        if (nq == 0) continue;
        memset(resp, 0, sizeof resp);
        double tb = mcv_now();
        if (v->local) {
            // Local: each req is a ranged read of the shard file. No network, no
            // batching win — just pread each range into an s3_response so the
            // decode-push path below is identical to the remote case.
            FILE *lf = fopen(shard_url, "rb");
            for (int i = 0; i < nq; ++i) {
                if (!lf) { resp[i].status = 404; continue; }
                if (fseek(lf, (long)reqs[i].offset, SEEK_SET) != 0) { resp[i].status = 500; continue; }
                uint8_t *b = malloc(reqs[i].length ? reqs[i].length : 1);
                if (!b) { resp[i].status = 500; continue; }
                size_t g = reqs[i].length ? fread(b, 1, reqs[i].length, lf) : 0;
                if (g != reqs[i].length) { free(b); resp[i].status = 500; continue; }
                resp[i].status = 200; resp[i].body = b; resp[i].body_len = g;
            }
            if (lf) fclose(lf);
        } else {
            s3_get_batch(v->s3, reqs, (size_t)nq, 32, resp);   // partial ok; check each
        }
        { int ok = 0; uint64_t bytes = 0;
          for (int i = 0; i < nq; ++i) if (s3_response_ok(&resp[i])) { ok++; bytes += resp[i].body_len; }
          // Count ACTUAL transferred bytes per batch (the real network/disk
          // throughput), not just the kept-chunk bytes accumulated in `got`
          // below — and do it per batch so a download-rate readout updates
          // promptly instead of only when the whole shard finishes.
          atomic_fetch_add_explicit(&v->net_bytes, bytes, memory_order_relaxed);
          MCVLOG("batch#%d  lod%d nq=%d ok=%d %.2fMB in %.0fms = %.1f MB/s",
                 nbatch++, lod, nq, ok, bytes/1048576.0, mcv_now()-tb,
                 bytes/1048576.0/((mcv_now()-tb)/1000.0)); }

        if (sub == 1) {                                // c3d: one chunk == one region
            for (int i = 0; i < nq; ++i) {
                mc_zarr_range *rg = &ranges[idx[i]];
                if (s3_response_ok(&resp[i]) && rg->len && resp[i].body_len >= rg->len) {
                    decode_item it = {lod, rg->cz, rg->cy, rg->cx, 1, 1, {0},{0},{0}, {0},{0}};
                    it.raw[0] = malloc(rg->len);
                    if (it.raw[0]) { memcpy(it.raw[0], resp[i].body, rg->len); it.rlen[0] = rg->len;
                        got += rg->len; decode_push(v, &it); }
                }
                s3_response_free(&resp[i]);
            }
        } else {                                       // v2: regroup the cube per region
            // Build one decode_item per distinct region in this batch.
            for (int i = 0; i < nq; ++i) {
                if (idx[i] < 0) continue;              // already consumed into a cube
                mc_zarr_range *r0 = &ranges[idx[i]];
                int rz = r0->cz / sub, ry = r0->cy / sub, rx = r0->cx / sub;
                decode_item it = {lod, rz, ry, rx, sub, 0, {0},{0},{0}, {0},{0}};
                for (int j = i; j < nq; ++j) {
                    if (idx[j] < 0) continue;
                    mc_zarr_range *rg = &ranges[idx[j]];
                    if (rg->cz / sub != rz || rg->cy / sub != ry || rg->cx / sub != rx) continue;
                    if (s3_response_ok(&resp[j]) && resp[j].body_len >= rg->len && it.nsub < 8) {
                        size_t rlen = rg->len ? rg->len : resp[j].body_len;
                        uint8_t *buf = malloc(rlen);
                        if (buf) { memcpy(buf, resp[j].body, rlen);
                            int k = it.nsub++;
                            it.raw[k] = buf; it.rlen[k] = rlen;
                            it.oz[k] = (rg->cz % sub) * edge;
                            it.oy[k] = (rg->cy % sub) * edge;
                            it.ox[k] = (rg->cx % sub) * edge;
                            got += rlen;
                        }
                    }
                    idx[j] = -1;                       // consumed
                }
                decode_push(v, &it);                   // nsub may be 0 -> ZERO region
            }
            for (int i = 0; i < nq; ++i) s3_response_free(&resp[i]);
        }
    }
    (void)got;   // net_bytes is now counted per batch above (actual transfer)
    free(ranges);
}

void mc_volume_prefetch_level(mc_volume *v, int lod, int nthreads, volatile int *cancel) {
    (void)nthreads;   // TODO: thread team; serial walk for now.
    if (lod < 0 || lod >= v->nlods) return;
    int gz, gy, gx;
    mc_zarr_inner_grid(v->lv[lod].z, &gz, &gy, &gx);
    int per = mc_zarr_shard_edge(v->lv[lod].z) / mc_zarr_inner_edge(v->lv[lod].z);
    for (int sz = 0; sz < gz; sz += per)
        for (int sy = 0; sy < gy; sy += per)
            for (int sx = 0; sx < gx; sx += per) {
                if (cancel && *cancel) return;
                mc_volume_prefetch_shard(v, lod, sz, sy, sx);
            }
}

size_t mc_volume_set_cache_bytes(mc_volume *v, size_t bytes) {
    return (v && v->cache) ? mc_cache_resize(v->cache, bytes) : 0;
}

// Tick-phase bracket for the render game-loop (see mc_cache_freeze/thaw). The
// volume owns the resident mc_cache; expose its two-phase clock so a client
// (volume-cartographer's global render tick) can freeze the cache for the
// duration of a frame's lock-free reads, then thaw between frames to let newly
// transcoded regions land and to bump the pin epoch.
void mc_volume_freeze(mc_volume *v) {
    if (!v || !v->cache) return;
    // Barrier: wait for the fill pool to finish thaw's PRESENT set before freezing,
    // so no fill write races a frozen read. Then do the render-gen bookkeeping for
    // what was filled (deferred from thaw -- we only know decoded count after join).
    if (v->nfillers > 0 && v->fill_n) {
        fill_pool_wait(v);
        if (atomic_load_explicit(&v->fill_decoded, memory_order_acquire)) {
            v->change_gen++;                           // pixels can differ now
            uint64_t last_mk = ~0ull;                  // mark filled regions (dedup'd)
            for (size_t i = 0; i < v->fill_n; ++i) {
                const mc_block_id *b = &v->fill_ids[i];
                int cz = b->bz / PER, cy = b->by / PER, cx = b->bx / PER;
                uint64_t rk = rkey(b->lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, b->lod, cz, cy, cx); last_mk = rk; }
            }
        }
        v->fill_n = 0;                                 // consumed (keep fill_ids buffer)
    }
    mc_cache_freeze(v->cache);
}

// Thaw = the single batch-apply step of the render game loop. Between a frame's
// freeze and the next, the lock-free frozen reads recorded their misses (blocks
// the render wanted but weren't resident). Here, while UNFROZEN, we:
//   1. thaw the cache (clears frozen, epoch++) so the fill is permitted,
//   2. drain the recorded miss set,
//   3. keep only blocks whose 256^3 source region is MC_PRESENT in the archive
//      (ABSENT = still downloading -> skip; ZERO = air -> already served),
//   4. fill those from the archive (decode from disk), TIME-BOUNDED so a big
//      miss set (a zoom across a LOD that misses the whole viewport) can't stall
//      the render tick more than ~MC_THAW_BUDGET_MS. Leftover blocks re-record
//      next frame and fill progressively; meanwhile the render shows coarser
//      resident LODs (mc_lod_sample fallback).
// The archive->cache fill runs on the persistent fill pool (filler_main),
// OFF the calling (GUI) thread: thaw hands it the PRESENT set and returns;
// freeze() waits for the pool before flipping frozen. Decode overlaps the
// frame render. Still race-free: writes are shard-partitioned single-owner and
// the only frozen-vs-fill barrier is freeze() joining the pool. No network IO.

// One fill-pool worker: owns shards { me, me+nfillers, ... }. Waits for thaw to
// bump fill_epoch, decodes its shards' blocks (lock-free, single-owner), then
// signals completion. The shard partition guarantees no two workers touch the
// same shard, so cache_fill_one needs no lock.
typedef struct { mc_volume *v; int me; } filler_arg;
static void *filler_main(void *ud) {
    filler_arg *a = ud; mc_volume *v = a->v; int me = a->me; free(a);
    mc_thread_setname("mc-fill");          // distinguish from the GUI thread in profilers
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&v->fill_mu);
        while (!v->fill_stop &&
               atomic_load_explicit(&v->fill_epoch, memory_order_relaxed) == seen)
            pthread_cond_wait(&v->fill_start, &v->fill_mu);
        if (v->fill_stop) { pthread_mutex_unlock(&v->fill_mu); return NULL; }
        seen = atomic_load_explicit(&v->fill_epoch, memory_order_relaxed);
        pthread_mutex_unlock(&v->fill_mu);

        uint64_t dec = 0;
        if (v->fill_n && v->fill_bucket && v->fill_link && v->fill_ids) {
            for (int s = me; s < NSHARD; s += v->nfillers)
                for (uint32_t i = v->fill_bucket[s]; i != UINT32_MAX; i = v->fill_link[i]) {
                    const mc_block_id *b = &v->fill_ids[i];
                    dec += (uint64_t)cache_fill_one(v->cache, b->lod, b->bz, b->by, b->bx);
                }
        }
        atomic_fetch_add_explicit(&v->fill_decoded, dec, memory_order_relaxed);
        // Last worker to finish this epoch signals freeze().
        pthread_mutex_lock(&v->fill_mu);
        if (atomic_fetch_add_explicit(&v->fill_done, 1, memory_order_acq_rel) + 1
            == (uint64_t)v->nfillers)
            pthread_cond_signal(&v->fill_fin);
        pthread_mutex_unlock(&v->fill_mu);
    }
}
// Kick the pool on the current fill set (caller filled fill_ids/bucket/link/fill_n).
static void fill_pool_start(mc_volume *v) {
    if (v->nfillers <= 0) return;
    pthread_mutex_lock(&v->fill_mu);
    atomic_store_explicit(&v->fill_done, 0, memory_order_relaxed);
    atomic_store_explicit(&v->fill_decoded, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&v->fill_epoch, 1, memory_order_release);
    pthread_cond_broadcast(&v->fill_start);
    pthread_mutex_unlock(&v->fill_mu);
}
// Wait for the pool to finish the in-flight fill (the freeze barrier).
static void fill_pool_wait(mc_volume *v) {
    if (v->nfillers <= 0) return;
    pthread_mutex_lock(&v->fill_mu);
    while (atomic_load_explicit(&v->fill_done, memory_order_acquire) < (uint64_t)v->nfillers)
        pthread_cond_wait(&v->fill_fin, &v->fill_mu);
    pthread_mutex_unlock(&v->fill_mu);
}
// Spawn the persistent fill pool once at open. Sized like the decode pool (the
// fill is the same DCT decode, just from the local archive). Call after v->cache
// exists. On failure nfillers stays 0 and thaw falls back to the inline fill.
static void fill_pool_init(mc_volume *v, int n) {
    if (n < 1) n = 1; if (n > 32) n = 32;
    pthread_mutex_init(&v->fill_mu, NULL);
    pthread_cond_init(&v->fill_start, NULL);
    pthread_cond_init(&v->fill_fin, NULL);
    atomic_store(&v->fill_epoch, 0); atomic_store(&v->fill_done, n);   // idle: "epoch 0 done"
    v->fill_stop = 0; v->nfillers = 0;
    for (int i = 0; i < n; ++i) {
        filler_arg *a = malloc(sizeof *a);
        if (!a) break;
        a->v = v; a->me = i;
        if (pthread_create(&v->fillers[i], NULL, filler_main, a) != 0) { free(a); break; }
        v->nfillers++;
    }
}
static void fill_pool_stop(mc_volume *v) {
    if (v->nfillers <= 0) return;
    pthread_mutex_lock(&v->fill_mu);
    v->fill_stop = 1;
    pthread_cond_broadcast(&v->fill_start);
    pthread_mutex_unlock(&v->fill_mu);
    for (int i = 0; i < v->nfillers; ++i) pthread_join(v->fillers[i], NULL);
    v->nfillers = 0;
    free(v->fill_bucket); v->fill_bucket = NULL;
    free(v->fill_link);   v->fill_link = NULL;
    free(v->fill_ids);    v->fill_ids = NULL;
}

#define MC_THAW_BUDGET_MS 5.0
#define MC_THAW_CHUNK 256          // blocks per mc_cache_update slice (time-checked)
void mc_volume_thaw(mc_volume *v) {
    if (!v || !v->cache) return;
    mc_cache_thaw(v->cache);                                  // clears frozen, epoch++

    static _Thread_local mc_block_id *miss = NULL;
    static _Thread_local size_t miss_cap = 0;
    if (!miss) { miss_cap = MISSQ_CAP; miss = malloc(miss_cap * sizeof *miss); }
    if (!miss) return;

    size_t n = mc_cache_misses_drain(v->cache, miss, miss_cap);

    // Collate (thaw = the collator), BEFORE any early-out. net_inflight is the
    // real pipeline depth (status bar); work_pending adds this frame's undrained
    // misses so the render gate keeps ticking until everything's resident.
    int dqn = v->dq_cap ? (v->dq_tail - v->dq_head + v->dq_cap) % v->dq_cap : 0;   // 0 in streaming
    v->net_inflight = (uint64_t)(v->rs_n + v->inflight_n + dqn);
    v->work_pending = v->net_inflight + (uint64_t)n;
    // Per-stage split. inflight covers claim -> append: on the wire + waiting in
    // the decode queue + actively inside decode->re-encode->append on a worker.
    int act = (int)atomic_load_explicit(&v->dec_active, memory_order_relaxed);
    v->snap_queued = (uint64_t)v->rs_n;
    v->snap_decq = (uint64_t)dqn;
    v->snap_encoding = (uint64_t)act;
    int dl = v->inflight_n - dqn - act;
    v->snap_downloading = dl > 0 ? (uint64_t)dl : 0;
    v->snap_staging_bytes = v->dq_bytes;

    if (!n) return;

    // Split the drained misses by local-archive coverage:
    //  PRESENT -> keep for the cache fill below (decode from local disk -- fast).
    //  ABSENT  -> issue ONE download request per region (zarr re-encode OR remote-
    //             .mca verbatim copy, on the download thread -- never here).
    // The miss set is per-block; many blocks map to one 256^3 region, so collapse
    // consecutive same-region absent blocks (misses drain roughly in scan order).
    size_t keep = 0;
    uint64_t last_rq = ~0ull;
    for (size_t i = 0; i < n; ++i) {
        const mc_block_id *b = &miss[i];
        int cz = b->bz / PER, cy = b->by / PER, cx = b->bx / PER;
        mc_cover cov = mc_archive_chunk_coverage(v->arc, b->lod, cz, cy, cx);
        if (cov == MC_PRESENT) { miss[keep++] = *b; }
        else if (cov == MC_ABSENT) {
            uint64_t rq = rkey(b->lod, cz, cy, cx);
            if (rq != last_rq) { req_push(v, b->lod, cz, cy, cx); last_rq = rq; }
        }
    }
    // Hand the PRESENT set to the persistent fill pool and RETURN. The pool decodes
    // (shard-partitioned, lock-free) while the frame renders; freeze() waits for it
    // and does the change_gen/region-mark bookkeeping once the fill is complete.
    // `miss` is thaw's thread-local buffer: stable until the next thaw, which can't
    // run until the next freeze has drained the pool, so the workers' borrow is safe.
    if (!keep || v->nfillers <= 0) {
        // No fill pool (or nothing to fill): do it inline (small/degenerate case).
        size_t filled = keep ? mc_cache_update(v->cache, miss, keep, 1) : 0;
        if (filled) {
            v->change_gen++;
            uint64_t last_mk = ~0ull;
            for (size_t i = 0; i < keep; ++i) {
                int cz = miss[i].bz / PER, cy = miss[i].by / PER, cx = miss[i].bx / PER;
                uint64_t rk = rkey(miss[i].lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, miss[i].lod, cz, cy, cx); last_mk = rk; }
            }
        }
        return;
    }
    // Shard-bucket the PRESENT blocks (same partition mc_cache_update uses) so each
    // pool worker owns a disjoint set of shards. Buffers grow as needed, reused tick
    // to tick. On alloc failure fall back to a synchronous inline fill.
    if (!v->fill_bucket) v->fill_bucket = malloc(NSHARD * sizeof *v->fill_bucket);
    if (!v->fill_link || v->fill_link_cap < keep) {
        uint32_t *nl = realloc(v->fill_link, keep * sizeof *v->fill_link);
        if (nl) { v->fill_link = nl; v->fill_link_cap = keep; }
    }
    if (!v->fill_bucket || !v->fill_link || v->fill_link_cap < keep) {
        size_t filled = mc_cache_update(v->cache, miss, keep, 1);
        if (filled) { v->change_gen++;
            uint64_t last_mk = ~0ull;
            for (size_t i = 0; i < keep; ++i) {
                int cz = miss[i].bz / PER, cy = miss[i].by / PER, cx = miss[i].bx / PER;
                uint64_t rk = rkey(miss[i].lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, miss[i].lod, cz, cy, cx); last_mk = rk; } } }
        return;
    }
    // Copy the kept blocks into the volume's OWN buffer: `miss` is thaw's thread-local
    // scratch, and a sibling cache thawed later in the same tick would overwrite it
    // while this pool still reads it -> cross-cache block corruption (junk reads).
    if (!v->fill_ids || v->fill_ids_cap < keep) {
        mc_block_id *ni = realloc(v->fill_ids, keep * sizeof *v->fill_ids);
        if (ni) { v->fill_ids = ni; v->fill_ids_cap = keep; }
    }
    if (!v->fill_ids || v->fill_ids_cap < keep) {      // alloc failed: inline fallback
        size_t filled = mc_cache_update(v->cache, miss, keep, 1);
        if (filled) { v->change_gen++;
            uint64_t last_mk = ~0ull;
            for (size_t i = 0; i < keep; ++i) {
                int cz = miss[i].bz / PER, cy = miss[i].by / PER, cx = miss[i].bx / PER;
                uint64_t rk = rkey(miss[i].lod, cz, cy, cx);
                if (rk != last_mk) { vol_mark_region(v, miss[i].lod, cz, cy, cx); last_mk = rk; } } }
        return;
    }
    memcpy(v->fill_ids, miss, keep * sizeof *v->fill_ids);
    for (int s = 0; s < NSHARD; ++s) v->fill_bucket[s] = UINT32_MAX;
    for (size_t i = 0; i < keep; ++i) {
        uint64_t key = bkey(v->fill_ids[i].lod, v->fill_ids[i].bz, v->fill_ids[i].by, v->fill_ids[i].bx);
        int s = (int)((khash(key) >> 56) & (NSHARD - 1));
        v->fill_link[i] = v->fill_bucket[s]; v->fill_bucket[s] = (uint32_t)i;
    }
    v->fill_n = keep;
    fill_pool_start(v);                                // workers decode; freeze() waits
}

size_t mc_volume_set_staging_bytes(mc_volume *v, size_t bytes) {
    if (!v) return 0;
    if (bytes < (64ull << 20)) bytes = 64ull << 20;    // floor 64MB
    pthread_mutex_lock(&v->mu);
    v->dq_byte_budget = bytes;
    pthread_cond_broadcast(&v->dq_nf);                 // a raise may unblock producers
    size_t installed = v->dq_byte_budget;
    pthread_mutex_unlock(&v->mu);
    return installed;
}

// Monotonic render generation: changes whenever a frozen render could produce
// different pixels -- a THAW cache fill (change_gen) or a coverage publish
// (archive gen, bumped on every chunk/air append). Equal gens across two frames
// with an unchanged camera => provably identical frame; the caller may skip it.
uint64_t mc_volume_render_gen(const mc_volume *v) {
    if (!v) return 0;
    uint64_t g = 1 + v->change_gen;
    if (v->arc) g += atomic_load_explicit(&v->arc->gen, memory_order_acquire);
    return g;
}

static inline uint32_t rgen_slot(int lod, int cz, int cy, int cx) {
    uint64_t key = mc_covkey(lod, cz, cy, cx);
    return (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 48) & (MC_RGEN_SLOTS - 1);
}
// Mark a region changed at the CURRENT render gen (call AFTER the publish/fill
// bumped it). Racing writers both store >= the prior value; last-writer-wins.
static void vol_mark_region(mc_volume *v, int lod, int cz, int cy, int cx) {
    if (!v->rgen) return;
    atomic_store_explicit(&v->rgen[rgen_slot(lod, cz, cy, cx)],
                          mc_volume_render_gen(v), memory_order_release);
}
// Last change gen of ONE region (0 = never changed). A viewer takes the max
// over its predicted working set; if that's <= the gen of its last frame and
// the camera is unchanged, the frame is provably identical for THAT viewport.
uint64_t mc_volume_region_gen(const mc_volume *v, int lod, int cz, int cy, int cx) {
    if (!v || !v->rgen) return 0;
    return atomic_load_explicit(&v->rgen[rgen_slot(lod, cz, cy, cx)],
                                memory_order_acquire);
}

void mc_volume_set_ready_cb(mc_volume *v, mc_volume_ready_fn cb, void *ud) {
    v->ready_cb = cb;
    v->ready_ud = ud;
}

void mc_volume_get_stats(const mc_volume *v, mc_volume_stats *out) {
    mc_cache_stats cs = {0};
    if (v->cache) mc_cache_get_stats(v->cache, &cs);
    out->cache_hits = cs.hits;
    out->cache_misses = cs.misses;
    out->cache_used_blocks = cs.used;
    out->cache_cap_blocks = cs.slots;
    out->disk_bytes = v->arc ? mc_archive_data_len(v->arc) : 0;
    // Streaming pulls bytes two ways: reader callbacks (node tables / serial-path
    // blobs) counted by mc_s3, plus the batched GETs counted in v->net_bytes.
    out->net_bytes = atomic_load_explicit(&v->net_bytes, memory_order_relaxed)
                   + (v->s3mca ? mc_s3_net_bytes(v->s3mca) : 0);
    // Frozen snapshots collated at the last thaw. regions_inflight = real pipeline
    // depth (status bar); work_pending = + undrained misses (render gate).
    out->regions_inflight = v->net_inflight;
    out->work_pending = v->work_pending;
    out->regions_queued = v->snap_queued;
    out->regions_downloading = v->snap_downloading;
    out->regions_decode_queued = v->snap_decq;
    out->regions_encoding = v->snap_encoding;
    out->staging_bytes = v->snap_staging_bytes;
}
