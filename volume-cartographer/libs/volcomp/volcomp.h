/* volcomp.h — single-header lossy compressor for u8 volumetric (micro-CT) data.
 *
 * Usage: #include "volcomp.h" from any C23 translation unit. Everything is
 * static; there is no library to link, no state, no threads, no environment.
 * Every function may be called concurrently on distinct buffers.
 *
 * Units: a BLOCK is 16^3 voxels (transform / random-access unit); a CHUNK is
 * 128^3 voxels (the independently decodable unit, one zarr chunk). Voxel
 * buffers are contiguous z-major: index = (z*128 + y)*128 + x.
 *
 * Source, destination and encoded buffers passed to one call must not overlap
 * (they are declared restrict).
 * Portable: AVX2+FMA kernels are used at runtime on x86-64 CPUs that have
 * them (no compiler flags needed), plain C kernels everywhere else.
 * The only configuration is the quantiser step q: 0 for the lossless mode
 * (exact), or [1,255] (1/256 precision) for the lossy DCT codec.
 * volcomp_encode performs one VOLCOMP_MALLOC/VOLCOMP_FREE of scratch;
 * decoding allocates nothing.
 *
 * Format: see spec/format.md. Format version 1. */
#ifndef VOLCOMP_H
#define VOLCOMP_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- SIMD kernels: AVX2+FMA on x86-64 when the CPU has it, plain C otherwise ----
 * Both kernel sets are compiled into any x86-64 build (the AVX2 ones carry a
 * target attribute, so no -mavx2 is needed) and selected at runtime per call
 * with __builtin_cpu_supports; a TU built with -mavx2 -mfma calls the AVX2
 * kernels directly. Non-x86 targets get the C kernels only. Define
 * VOLCOMP_NO_AVX2 to force the C kernels (testing). The two kernel sets agree
 * to within +-1 LSB (spec "Cross-build agreement"); encoded bytes may differ. */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(VOLCOMP_NO_AVX2) && !defined(_MSC_VER)
#define VF_HAVE_AVX2 1
#include <immintrin.h>
#if defined(__AVX2__) && defined(__FMA__)
#define VF_AVX2_STATIC 1
#define VF_TARGET_AVX2
#else
#define VF_AVX2_STATIC 0
#define VF_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif
#else
#define VF_HAVE_AVX2 0
#define VF_AVX2_STATIC 0
#define VF_TARGET_AVX2
#endif

#ifndef VOLCOMP_MALLOC
#define VOLCOMP_MALLOC(n) malloc(n)
#define VOLCOMP_FREE(p) free(p)
#endif

#define VOLCOMP_VERSION_STRING "1.3.0"
#define VOLCOMP_FORMAT_VERSION 1u
/* Format revision 2 added the mask chunk mode (header byte 5 = 4); revision 3
 * added the spatially lossless mask mode (header byte 5 = 5); revision 4 adds the
 * surface chunk (header byte 5 = 6, spec/format.md §13). The version byte
 * stays 1 across revisions: every 1.0/1.1 stream is byte for byte what it always
 * was, and an older decoder rejects a newer mode cleanly (it refuses any mode
 * above the highest it knows with VOLCOMP_ERR_CORRUPT). A writer must not emit a
 * mode-5 chunk to a consumer that only reads revision 2; that consumer will
 * refuse it cleanly rather than misread it, and the same holds for a mode-6
 * chunk and a revision-3 reader. See spec/format.md §11, §12 and §13.
 */
#define VOLCOMP_FORMAT_REVISION 4u
#define VOLCOMP_BLOCK_DIM 16u
#define VOLCOMP_CHUNK_DIM 128u
#define VOLCOMP_BLOCK_VOXELS 4096u
#define VOLCOMP_CHUNK_VOXELS 2097152u
#define VOLCOMP_Q_MIN 1.0f
#define VOLCOMP_Q_MAX 255.0f
/* q == 0 selects the lossless mode (exact reconstruction; see "lossless mode"
 * below and spec/format.md §10). Any q in [1,255] is the lossy DCT codec, whose
 * bitstream is unchanged from volcomp 1.0. */
#define VOLCOMP_Q_LOSSLESS 0.0f
/* Provable upper bound on the encoded size of any chunk at any q (spec
 * "Bounds"): 8 header + 680 tables + 256 directory + tANS (<= 2 B/token, <= 8192
 * tokens/block, + 8 flush/substream) + bypass (<= 12288 B/block). Real CT
 * chunks encode to 20-200 KB; encode into a reused buffer of this size. */
#define VOLCOMP_ENCODE_BOUND ((size_t)14681264u)

typedef enum volcomp_status {
  VOLCOMP_OK = 0,
  VOLCOMP_ERR_ARG,       /* null pointer, q not 0 and outside [1,255], bad block coord */
  VOLCOMP_ERR_CORRUPT,   /* malformed, truncated, or non-canonical stream          */
  VOLCOMP_ERR_VERSION,   /* well-formed header with an unsupported version byte    */
  VOLCOMP_ERR_NOMEM,     /* allocation failed                                      */
  VOLCOMP_ERR_SHORT_BUF, /* destination capacity too small                         */
} volcomp_status;

static inline const char *volcomp_status_string(volcomp_status s);
/* Encode one 128^3 u8 chunk at quantiser step q (VOLCOMP_Q_LOSSLESS or
 * [1,255]); writes at most dst_cap bytes, *out_n = stream length. */
static inline volcomp_status volcomp_encode(const uint8_t *restrict src_zyx, float q,
                                            void *restrict dst, size_t dst_cap, size_t *out_n);
/* Decode a whole chunk into dst_zyx (dst_cap >= VOLCOMP_CHUNK_VOXELS). Contents
 * are unspecified on failure. */
static inline volcomp_status volcomp_decode(const void *restrict enc, size_t enc_n,
                                            uint8_t *restrict dst_zyx, size_t dst_cap);
/* Decode block (bz,by,bx), each in 0..7, into dst_block (>= 4096 bytes, z-major
 * 16^3). Entropy-decodes only the owning substream (<= 16 blocks of work); the
 * bytes are identical to the corresponding region of volcomp_decode. */
static inline volcomp_status volcomp_decode_block(const void *restrict enc, size_t enc_n,
                                                  uint32_t bz, uint32_t by, uint32_t bx,
                                                  uint8_t *restrict dst_block, size_t dst_cap);

/* Optional post-process: deblock an assembled decoded volume of nz*ny*nx
 * voxels (z-major) in place. Applies the quant-gated 4-tap filter across every
 * plane that is a multiple of 16 along each axis (inside the volume), so block
 * faces and chunk faces receive identical treatment; the volume's own outer
 * faces are not filtered. q must be the q the chunks were encoded with (the
 * gate strength scales with it). Not part of the format; the caller chooses
 * whether to run it. Exact zeros (masked air) are never filtered into
 * (VOLCOMP_DEBLOCK_ZERO_GUARD, below). Measured on scroll data: +0.15 dB (q8) to +0.4 dB (q32)
 * PSNR, blocking amplification 1.5x -> ~1.1x, ~5% of decode time. */
static inline void volcomp_deblock(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q);
/* volcomp_deblock with options. VOLCOMP_DEBLOCK_ZERO_GUARD: leave every face
 * position alone where one of the four voxels it reads is exactly 0 — masked CT
 * stores air as 0, and a mask edge that falls on a 16-plane is a real edge, not a
 * seam. volcomp_deblock itself now always applies the guard: without it the filter
 * blurred masked CT's chunk-aligned air edges at q >= 16 (edge error +54 %, chunk
 * faces 2.8x rougher than unfiltered); with it no chunk is ever worse than no
 * filter on the measured CT, and nothing else changes (docs/deblocking.md).
 * flags = 0 is the old, unguarded filter. */
#define VOLCOMP_DEBLOCK_ZERO_GUARD 1u
/* volcomp_decode_smooth: smooth with the gated face filter instead of a Gaussian
 * seam blur (strength then scales the filter's q) */
#define VOLCOMP_SMOOTH_GATED 2u
/* Optional decode-side deblocking of one chunk (not part of the format; the
 * stream is untouched, only this reader's reconstruction differs): decode, smooth
 * the voxels next to interior block faces (a Gaussian of `strength` voxels, or
 * with VOLCOMP_SMOOTH_GATED the gated face filter at `strength` x q), then project
 * every block back onto the quantisation cells its coefficients were decoded from,
 * so the output is still a reconstruction the stream allows. Lossy (mode 0) and
 * surface chunks only (a surface chunk keeps its exact threshold); any other chunk
 * decodes exactly as volcomp_decode. Chunk faces are left as decoded; run
 * volcomp_deblock_ex on an assembled region for those. VOLCOMP_DEBLOCK_ZERO_GUARD
 * keeps exact zeros (masked air) and the faces touching them untouched. */
static inline volcomp_status volcomp_decode_smooth(const void *restrict enc, size_t enc_n, uint8_t *restrict dst_zyx,
                                                   size_t dst_cap, float strength, unsigned flags);
static inline void volcomp_deblock_ex(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q, unsigned flags);
/* True if the stream decodes exactly (was encoded with VOLCOMP_Q_LOSSLESS).
 * False for a mask chunk, which stores a 2x downscale of its source.
 * Also validates the magic/version/mode bytes. */
static inline volcomp_status volcomp_is_lossless(const void *restrict enc, size_t enc_n, bool *out);
/* The q the stream was encoded with; 0 for a lossless stream and for a mask
 * chunk (neither is the DCT codec; volcomp_is_lossless tells them apart); the
 * base's q for a surface chunk. */
static inline volcomp_status volcomp_stream_q(const void *restrict enc, size_t enc_n, float *out);

/* ---- binary mask chunks (spec/format.md §11) ----
 * A mask chunk is the storage form of a published binary mask. The 128^3 source
 * is downscaled 2x2x2 by MAJORITY (count*2 >= 8) to a 64^3 grid, and that grid
 * is stored EXACTLY, with a context-adaptive binary arithmetic coder (12 causal
 * neighbours, 4096 contexts). There is nothing to configure.
 *
 * `volcomp_decode` returns the grid TRILINEARLY INTERPOLATED back to 128^3 as
 * u8 0..255: stored voxel i sits at full-res voxel 2i, so output voxel j samples
 * the grid at j/2 (clamped at the top edge) and the result is a continuous field
 * with a two-voxel edge ramp around the stored boundary — the training target
 * itself, not a 0/255 mask. Every existing reader (volcomp_decode_block,
 * python/volcomp_zarr, the shard readers) therefore works unchanged.
 * Measured on the 2.4 um PHercParis4 recto: 0.0057 bits per full-res voxel. */
#define VOLCOMP_MASK_DIM 64u        /* stored grid edge = VOLCOMP_CHUNK_DIM / 2 */
#define VOLCOMP_MASK_VOXELS 262144u /* VOLCOMP_MASK_DIM^3                       */
/* Encode a 128^3 mask (any nonzero source voxel is 1); never larger than
 * 9 + VOLCOMP_MASK_VOXELS/8 = 32 777 bytes. One VOLCOMP_MALLOC of scratch. */
static inline volcomp_status volcomp_mask_encode(const uint8_t *restrict src_zyx, void *restrict dst,
                                                 size_t dst_cap, size_t *out_n);

/* ---- spatially lossless mask chunks (mode 5, spec/format.md §12) ----
 * The same coder, no downscale: the full 128^3 binary mask (any nonzero source
 * voxel is 1) is coded exactly with the same 12-neighbour context model and the
 * same adaptation schedule, and `volcomp_decode` returns 0/255 exactly as
 * stored. Where mode 4 trades a two-voxel boundary ramp for 4x fewer coded bits,
 * mode 5 is the published mask itself, voxel for voxel.
 * Measured on the 2.4 um PHercParis4 recto: 0.0255 bits per voxel (0.022 is what
 * an ideal, two-pass model over the same contexts would cost).
 * Never larger than VOLCOMP_MASK_LL_BOUND. One VOLCOMP_MALLOC of scratch. */
#define VOLCOMP_MASK_LL_DIM VOLCOMP_CHUNK_DIM /* the stored grid IS the chunk */
#define VOLCOMP_MASK_LL_BOUND ((size_t)(9u + VOLCOMP_CHUNK_VOXELS / 8u)) /* 8 header + 1 flags + 262 144 */
static inline volcomp_status volcomp_mask_encode_lossless(const uint8_t *restrict src_zyx,
                                                          void *restrict dst, size_t dst_cap,
                                                          size_t *out_n);
/* True iff the stream is a mask chunk of either mode; with out_dim, the edge of
 * the grid it stores — VOLCOMP_MASK_DIM (64) for the 2x mode, VOLCOMP_CHUNK_DIM
 * (128) for the lossless one, which is how a reader tells the two apart.
 * VOLCOMP_ERR_ARG if the stream is well formed but is not a mask chunk. */
static inline volcomp_status volcomp_mask_info(const void *restrict enc, size_t enc_n, uint32_t *out_dim);
/* The STORED grid of a mask chunk, as 0/255: 64^3 for mode 4, 128^3 for mode 5
 * (dst_cap >= dim^3, i.e. VOLCOMP_MASK_VOXELS resp. VOLCOMP_CHUNK_VOXELS). */
static inline volcomp_status volcomp_mask_decode_stored(const void *restrict enc, size_t enc_n,
                                                        uint8_t *restrict dst, size_t dst_cap);
/* ---- surface chunks (mode 6, spec/format.md §13) ----
 * For smooth probability fields (surface predictions: u8 = round(255 p)) whose
 * consumers threshold them. The chunk is the DCT codec at step q PLUS a
 * refinement layer that makes the threshold exact: for every voxel,
 * (source >= thr) == (decoded >= thr), so the binary mask, its skeleton and its
 * distance field are exactly those of the source, and the soft values carry the
 * DCT error at q (a wrong-side voxel decodes to thr or thr - 1). thr = 128 is
 * p >= 0.5. `volcomp_decode` / `volcomp_decode_block` read it like any chunk
 * (decode_block decodes the whole chunk: the refinement is one adaptive pass);
 * decoding is bit-identical on every build (the base uses the strict kernels).
 * The base uses a FLAT step law (every AC step q), so q here is not mode 0's q:
 * surface q 48 is about the size of mode-0 q 16; q in [2, 255]. Measurements on
 * the recto and m7 surface teachers: docs/surface_mode.md. Never larger than
 * VOLCOMP_SURFACE_ENCODE_BOUND; one VOLCOMP_MALLOC of scratch. */
/* the flat step law's smallest step: below it an AC level of a full-swing block
 * would exceed the format's cap (|level| <= 5220, spec §4.2) */
#define VOLCOMP_SURFACE_Q_MIN 2.0f
#define VOLCOMP_SURFACE_ENCODE_BOUND (VOLCOMP_ENCODE_BOUND + 8u + VOLCOMP_CHUNK_VOXELS + 8u)
static inline volcomp_status volcomp_surface_encode(const uint8_t *restrict src_zyx, float q, uint32_t thr,
                                                    void *restrict dst, size_t dst_cap, size_t *out_n);
/* OK iff a surface chunk; its threshold and refinement margin (0 = the base
 * alone was already exact). VOLCOMP_ERR_ARG for any other well-formed chunk. */
static inline volcomp_status volcomp_surface_info(const void *restrict enc, size_t enc_n, uint32_t *out_thr,
                                                  uint32_t *out_margin);
/* Which kernel set this process will use: "avx2" or "c". */
static inline const char *volcomp_kernels(void);

/* ======================================================================== */
/*                          implementation                                  */
/* ======================================================================== */

/* ---- format constants (normative, spec/format.md) ---- */
#define VF_BLKV 4096u
#define VF_BLOCKS 512u
#define VF_BLOCKS_AXIS 8u
#define VF_NSUB 32u
#define VF_BLOCKS_PER_SUB 16u
#define VF_VERSION 1u
#define VF_HDR_BYTES 8u
/* header byte 5: which codec produced the chunk (see "lossless mode" below) */
#define VLL_MODE_LOSSY 0u
#define VLL_MODE_CODED 1u
#define VLL_MODE_RAW 2u
#define VLL_MODE_CONST 3u
#define VLL_MODE_MASK 4u
#define VLL_MODE_MASK_LL 5u
#define VLL_MODE_SURFACE 6u
#define VLL_MODE_MAX 6u
#define VF_DIR_BYTES (VF_NSUB * 8u)
#define VF_Q_RAW_MIN 256u
#define VF_Q_RAW_MAX 65280u
#define VF_NTOK 32u
#define VF_TOK_EOB 31u
#define VF_NMODELS 10u
#define VF_DC_CTX 9u
#define VF_PROB_BITS 10u /* tANS table log: 1024 states per model */
#define VF_PROB_SCALE (1u << VF_PROB_BITS)
#define VF_TOKMAX_DC 20u  /* k <= 18: |dc| <= 65536 */
#define VF_TOKMAX_RUN 13u /* k <= 11: run <= 4094  */
#define VF_TOKMAX_LVL 14u /* k <= 12: mag <= 5220  */
#define VF_DC_ABS_MAX 65536
#define VF_AC_MAG_MAX 5220u
#define VF_HF_EXP 0.65f
#define VF_DC_FINE 0.125f
#define VF_DZ_Q 0.2f       /* AC dead-zone offset (encoder) */
#define VF_DZ_Q_DC 0.5f    /* DC is rounded to nearest (encoder) */
#ifndef VF_DZ_DQ
#define VF_DZ_DQ 0.30f     /* AC reconstruction offset, |level| >= 2 */
#endif
#ifndef VF_DZ_DQ1
#define VF_DZ_DQ1 0.15f    /* AC reconstruction offset, |level| == 1 */
#endif
#define VF_NRADII 46u
#define VF_TABLES_MAX_BYTES (VF_NMODELS * (4u + VF_NTOK * 2u))
#define VF_SUB_TOK_MAX (2u * VF_BLOCKS_PER_SUB * 8192u + 8u)
#define VF_SUB_BYP_MAX (VF_BLOCKS_PER_SUB * 12288u + 8u)

/* transform kernels: allow FMA contraction and reassociation (float math, +-1 LSB tolerance) */
#if defined(__clang__)
#define VF_FASTMATH_BEGIN _Pragma("clang fp contract(fast) reassociate(on)")
#else
#define VF_FASTMATH_BEGIN
#endif

/* ---- DCT tables (orthonormal 16-point DCT-II even/odd halves; 16^3 scan) ---- */
static const float VF_EV[8][8] = {
  {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f},
  {0.346759975f, 0.293968886f, 0.196423739f, 0.0689748451f, -0.0689748451f, -0.196423739f, -0.293968886f, -0.346759975f},
  {0.326640755f, 0.135299027f, -0.135299027f, -0.326640755f, -0.326640755f, -0.135299027f, 0.135299027f, 0.326640755f},
  {0.293968886f, -0.0689748451f, -0.346759975f, -0.196423739f, 0.196423739f, 0.346759975f, 0.0689748451f, -0.293968886f},
  {0.25f, -0.25f, -0.25f, 0.25f, 0.25f, -0.25f, -0.25f, 0.25f},
  {0.196423739f, -0.346759975f, 0.0689748451f, 0.293968886f, -0.293968886f, -0.0689748451f, 0.346759975f, -0.196423739f},
  {0.135299027f, -0.326640755f, 0.326640755f, -0.135299027f, -0.135299027f, 0.326640755f, -0.326640755f, 0.135299027f},
  {0.0689748451f, -0.196423739f, 0.293968886f, -0.346759975f, 0.346759975f, -0.293968886f, 0.196423739f, -0.0689748451f},
};

static const float VF_OD[8][8] = {
  {0.351850927f, 0.338329494f, 0.311806262f, 0.273300469f, 0.224291891f, 0.166663915f, 0.102631129f, 0.0346542932f},
  {0.338329494f, 0.224291891f, 0.0346542932f, -0.166663915f, -0.311806262f, -0.351850927f, -0.273300469f, -0.102631129f},
  {0.311806262f, 0.0346542932f, -0.273300469f, -0.338329494f, -0.102631129f, 0.224291891f, 0.351850927f, 0.166663915f},
  {0.273300469f, -0.166663915f, -0.338329494f, 0.0346542932f, 0.351850927f, 0.102631129f, -0.311806262f, -0.224291891f},
  {0.224291891f, -0.311806262f, -0.102631129f, 0.351850927f, -0.0346542932f, -0.338329494f, 0.166663915f, 0.273300469f},
  {0.166663915f, -0.351850927f, 0.224291891f, 0.102631129f, -0.338329494f, 0.273300469f, 0.0346542932f, -0.311806262f},
  {0.102631129f, -0.273300469f, 0.351850927f, -0.311806262f, 0.166663915f, 0.0346542932f, -0.224291891f, 0.338329494f},
  {0.0346542932f, -0.102631129f, 0.166663915f, -0.224291891f, 0.273300469f, -0.311806262f, 0.338329494f, -0.351850927f},
};

static const uint16_t VF_SCAN16[4096] = {
  0, 1, 16, 256, 2, 17, 32, 257, 272, 512, 3, 18, 33, 48, 258, 273,
  288, 513, 528, 768, 4, 19, 34, 49, 64, 259, 274, 289, 304, 514, 529, 544,
  769, 784, 1024, 5, 20, 35, 50, 65, 80, 260, 275, 290, 305, 320, 515, 530,
  545, 560, 770, 785, 800, 1025, 1040, 1280, 6, 21, 36, 51, 66, 81, 96, 261,
  276, 291, 306, 321, 336, 516, 531, 546, 561, 576, 771, 786, 801, 816, 1026, 1041,
  1056, 1281, 1296, 1536, 7, 22, 37, 52, 67, 82, 97, 112, 262, 277, 292, 307,
  322, 337, 352, 517, 532, 547, 562, 577, 592, 772, 787, 802, 817, 832, 1027, 1042,
  1057, 1072, 1282, 1297, 1312, 1537, 1552, 1792, 8, 23, 38, 53, 68, 83, 98, 113,
  128, 263, 278, 293, 308, 323, 338, 353, 368, 518, 533, 548, 563, 578, 593, 608,
  773, 788, 803, 818, 833, 848, 1028, 1043, 1058, 1073, 1088, 1283, 1298, 1313, 1328, 1538,
  1553, 1568, 1793, 1808, 2048, 9, 24, 39, 54, 69, 84, 99, 114, 129, 144, 264,
  279, 294, 309, 324, 339, 354, 369, 384, 519, 534, 549, 564, 579, 594, 609, 624,
  774, 789, 804, 819, 834, 849, 864, 1029, 1044, 1059, 1074, 1089, 1104, 1284, 1299, 1314,
  1329, 1344, 1539, 1554, 1569, 1584, 1794, 1809, 1824, 2049, 2064, 2304, 10, 25, 40, 55,
  70, 85, 100, 115, 130, 145, 160, 265, 280, 295, 310, 325, 340, 355, 370, 385,
  400, 520, 535, 550, 565, 580, 595, 610, 625, 640, 775, 790, 805, 820, 835, 850,
  865, 880, 1030, 1045, 1060, 1075, 1090, 1105, 1120, 1285, 1300, 1315, 1330, 1345, 1360, 1540,
  1555, 1570, 1585, 1600, 1795, 1810, 1825, 1840, 2050, 2065, 2080, 2305, 2320, 2560, 11, 26,
  41, 56, 71, 86, 101, 116, 131, 146, 161, 176, 266, 281, 296, 311, 326, 341,
  356, 371, 386, 401, 416, 521, 536, 551, 566, 581, 596, 611, 626, 641, 656, 776,
  791, 806, 821, 836, 851, 866, 881, 896, 1031, 1046, 1061, 1076, 1091, 1106, 1121, 1136,
  1286, 1301, 1316, 1331, 1346, 1361, 1376, 1541, 1556, 1571, 1586, 1601, 1616, 1796, 1811, 1826,
  1841, 1856, 2051, 2066, 2081, 2096, 2306, 2321, 2336, 2561, 2576, 2816, 12, 27, 42, 57,
  72, 87, 102, 117, 132, 147, 162, 177, 192, 267, 282, 297, 312, 327, 342, 357,
  372, 387, 402, 417, 432, 522, 537, 552, 567, 582, 597, 612, 627, 642, 657, 672,
  777, 792, 807, 822, 837, 852, 867, 882, 897, 912, 1032, 1047, 1062, 1077, 1092, 1107,
  1122, 1137, 1152, 1287, 1302, 1317, 1332, 1347, 1362, 1377, 1392, 1542, 1557, 1572, 1587, 1602,
  1617, 1632, 1797, 1812, 1827, 1842, 1857, 1872, 2052, 2067, 2082, 2097, 2112, 2307, 2322, 2337,
  2352, 2562, 2577, 2592, 2817, 2832, 3072, 13, 28, 43, 58, 73, 88, 103, 118, 133,
  148, 163, 178, 193, 208, 268, 283, 298, 313, 328, 343, 358, 373, 388, 403, 418,
  433, 448, 523, 538, 553, 568, 583, 598, 613, 628, 643, 658, 673, 688, 778, 793,
  808, 823, 838, 853, 868, 883, 898, 913, 928, 1033, 1048, 1063, 1078, 1093, 1108, 1123,
  1138, 1153, 1168, 1288, 1303, 1318, 1333, 1348, 1363, 1378, 1393, 1408, 1543, 1558, 1573, 1588,
  1603, 1618, 1633, 1648, 1798, 1813, 1828, 1843, 1858, 1873, 1888, 2053, 2068, 2083, 2098, 2113,
  2128, 2308, 2323, 2338, 2353, 2368, 2563, 2578, 2593, 2608, 2818, 2833, 2848, 3073, 3088, 3328,
  14, 29, 44, 59, 74, 89, 104, 119, 134, 149, 164, 179, 194, 209, 224, 269,
  284, 299, 314, 329, 344, 359, 374, 389, 404, 419, 434, 449, 464, 524, 539, 554,
  569, 584, 599, 614, 629, 644, 659, 674, 689, 704, 779, 794, 809, 824, 839, 854,
  869, 884, 899, 914, 929, 944, 1034, 1049, 1064, 1079, 1094, 1109, 1124, 1139, 1154, 1169,
  1184, 1289, 1304, 1319, 1334, 1349, 1364, 1379, 1394, 1409, 1424, 1544, 1559, 1574, 1589, 1604,
  1619, 1634, 1649, 1664, 1799, 1814, 1829, 1844, 1859, 1874, 1889, 1904, 2054, 2069, 2084, 2099,
  2114, 2129, 2144, 2309, 2324, 2339, 2354, 2369, 2384, 2564, 2579, 2594, 2609, 2624, 2819, 2834,
  2849, 2864, 3074, 3089, 3104, 3329, 3344, 3584, 15, 30, 45, 60, 75, 90, 105, 120,
  135, 150, 165, 180, 195, 210, 225, 240, 270, 285, 300, 315, 330, 345, 360, 375,
  390, 405, 420, 435, 450, 465, 480, 525, 540, 555, 570, 585, 600, 615, 630, 645,
  660, 675, 690, 705, 720, 780, 795, 810, 825, 840, 855, 870, 885, 900, 915, 930,
  945, 960, 1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200, 1290, 1305,
  1320, 1335, 1350, 1365, 1380, 1395, 1410, 1425, 1440, 1545, 1560, 1575, 1590, 1605, 1620, 1635,
  1650, 1665, 1680, 1800, 1815, 1830, 1845, 1860, 1875, 1890, 1905, 1920, 2055, 2070, 2085, 2100,
  2115, 2130, 2145, 2160, 2310, 2325, 2340, 2355, 2370, 2385, 2400, 2565, 2580, 2595, 2610, 2625,
  2640, 2820, 2835, 2850, 2865, 2880, 3075, 3090, 3105, 3120, 3330, 3345, 3360, 3585, 3600, 3840,
  31, 46, 61, 76, 91, 106, 121, 136, 151, 166, 181, 196, 211, 226, 241, 271,
  286, 301, 316, 331, 346, 361, 376, 391, 406, 421, 436, 451, 466, 481, 496, 526,
  541, 556, 571, 586, 601, 616, 631, 646, 661, 676, 691, 706, 721, 736, 781, 796,
  811, 826, 841, 856, 871, 886, 901, 916, 931, 946, 961, 976, 1036, 1051, 1066, 1081,
  1096, 1111, 1126, 1141, 1156, 1171, 1186, 1201, 1216, 1291, 1306, 1321, 1336, 1351, 1366, 1381,
  1396, 1411, 1426, 1441, 1456, 1546, 1561, 1576, 1591, 1606, 1621, 1636, 1651, 1666, 1681, 1696,
  1801, 1816, 1831, 1846, 1861, 1876, 1891, 1906, 1921, 1936, 2056, 2071, 2086, 2101, 2116, 2131,
  2146, 2161, 2176, 2311, 2326, 2341, 2356, 2371, 2386, 2401, 2416, 2566, 2581, 2596, 2611, 2626,
  2641, 2656, 2821, 2836, 2851, 2866, 2881, 2896, 3076, 3091, 3106, 3121, 3136, 3331, 3346, 3361,
  3376, 3586, 3601, 3616, 3841, 3856, 47, 62, 77, 92, 107, 122, 137, 152, 167, 182,
  197, 212, 227, 242, 287, 302, 317, 332, 347, 362, 377, 392, 407, 422, 437, 452,
  467, 482, 497, 527, 542, 557, 572, 587, 602, 617, 632, 647, 662, 677, 692, 707,
  722, 737, 752, 782, 797, 812, 827, 842, 857, 872, 887, 902, 917, 932, 947, 962,
  977, 992, 1037, 1052, 1067, 1082, 1097, 1112, 1127, 1142, 1157, 1172, 1187, 1202, 1217, 1232,
  1292, 1307, 1322, 1337, 1352, 1367, 1382, 1397, 1412, 1427, 1442, 1457, 1472, 1547, 1562, 1577,
  1592, 1607, 1622, 1637, 1652, 1667, 1682, 1697, 1712, 1802, 1817, 1832, 1847, 1862, 1877, 1892,
  1907, 1922, 1937, 1952, 2057, 2072, 2087, 2102, 2117, 2132, 2147, 2162, 2177, 2192, 2312, 2327,
  2342, 2357, 2372, 2387, 2402, 2417, 2432, 2567, 2582, 2597, 2612, 2627, 2642, 2657, 2672, 2822,
  2837, 2852, 2867, 2882, 2897, 2912, 3077, 3092, 3107, 3122, 3137, 3152, 3332, 3347, 3362, 3377,
  3392, 3587, 3602, 3617, 3632, 3842, 3857, 3872, 63, 78, 93, 108, 123, 138, 153, 168,
  183, 198, 213, 228, 243, 303, 318, 333, 348, 363, 378, 393, 408, 423, 438, 453,
  468, 483, 498, 543, 558, 573, 588, 603, 618, 633, 648, 663, 678, 693, 708, 723,
  738, 753, 783, 798, 813, 828, 843, 858, 873, 888, 903, 918, 933, 948, 963, 978,
  993, 1008, 1038, 1053, 1068, 1083, 1098, 1113, 1128, 1143, 1158, 1173, 1188, 1203, 1218, 1233,
  1248, 1293, 1308, 1323, 1338, 1353, 1368, 1383, 1398, 1413, 1428, 1443, 1458, 1473, 1488, 1548,
  1563, 1578, 1593, 1608, 1623, 1638, 1653, 1668, 1683, 1698, 1713, 1728, 1803, 1818, 1833, 1848,
  1863, 1878, 1893, 1908, 1923, 1938, 1953, 1968, 2058, 2073, 2088, 2103, 2118, 2133, 2148, 2163,
  2178, 2193, 2208, 2313, 2328, 2343, 2358, 2373, 2388, 2403, 2418, 2433, 2448, 2568, 2583, 2598,
  2613, 2628, 2643, 2658, 2673, 2688, 2823, 2838, 2853, 2868, 2883, 2898, 2913, 2928, 3078, 3093,
  3108, 3123, 3138, 3153, 3168, 3333, 3348, 3363, 3378, 3393, 3408, 3588, 3603, 3618, 3633, 3648,
  3843, 3858, 3873, 3888, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244,
  319, 334, 349, 364, 379, 394, 409, 424, 439, 454, 469, 484, 499, 559, 574, 589,
  604, 619, 634, 649, 664, 679, 694, 709, 724, 739, 754, 799, 814, 829, 844, 859,
  874, 889, 904, 919, 934, 949, 964, 979, 994, 1009, 1039, 1054, 1069, 1084, 1099, 1114,
  1129, 1144, 1159, 1174, 1189, 1204, 1219, 1234, 1249, 1264, 1294, 1309, 1324, 1339, 1354, 1369,
  1384, 1399, 1414, 1429, 1444, 1459, 1474, 1489, 1504, 1549, 1564, 1579, 1594, 1609, 1624, 1639,
  1654, 1669, 1684, 1699, 1714, 1729, 1744, 1804, 1819, 1834, 1849, 1864, 1879, 1894, 1909, 1924,
  1939, 1954, 1969, 1984, 2059, 2074, 2089, 2104, 2119, 2134, 2149, 2164, 2179, 2194, 2209, 2224,
  2314, 2329, 2344, 2359, 2374, 2389, 2404, 2419, 2434, 2449, 2464, 2569, 2584, 2599, 2614, 2629,
  2644, 2659, 2674, 2689, 2704, 2824, 2839, 2854, 2869, 2884, 2899, 2914, 2929, 2944, 3079, 3094,
  3109, 3124, 3139, 3154, 3169, 3184, 3334, 3349, 3364, 3379, 3394, 3409, 3424, 3589, 3604, 3619,
  3634, 3649, 3664, 3844, 3859, 3874, 3889, 3904, 95, 110, 125, 140, 155, 170, 185, 200,
  215, 230, 245, 335, 350, 365, 380, 395, 410, 425, 440, 455, 470, 485, 500, 575,
  590, 605, 620, 635, 650, 665, 680, 695, 710, 725, 740, 755, 815, 830, 845, 860,
  875, 890, 905, 920, 935, 950, 965, 980, 995, 1010, 1055, 1070, 1085, 1100, 1115, 1130,
  1145, 1160, 1175, 1190, 1205, 1220, 1235, 1250, 1265, 1295, 1310, 1325, 1340, 1355, 1370, 1385,
  1400, 1415, 1430, 1445, 1460, 1475, 1490, 1505, 1520, 1550, 1565, 1580, 1595, 1610, 1625, 1640,
  1655, 1670, 1685, 1700, 1715, 1730, 1745, 1760, 1805, 1820, 1835, 1850, 1865, 1880, 1895, 1910,
  1925, 1940, 1955, 1970, 1985, 2000, 2060, 2075, 2090, 2105, 2120, 2135, 2150, 2165, 2180, 2195,
  2210, 2225, 2240, 2315, 2330, 2345, 2360, 2375, 2390, 2405, 2420, 2435, 2450, 2465, 2480, 2570,
  2585, 2600, 2615, 2630, 2645, 2660, 2675, 2690, 2705, 2720, 2825, 2840, 2855, 2870, 2885, 2900,
  2915, 2930, 2945, 2960, 3080, 3095, 3110, 3125, 3140, 3155, 3170, 3185, 3200, 3335, 3350, 3365,
  3380, 3395, 3410, 3425, 3440, 3590, 3605, 3620, 3635, 3650, 3665, 3680, 3845, 3860, 3875, 3890,
  3905, 3920, 111, 126, 141, 156, 171, 186, 201, 216, 231, 246, 351, 366, 381, 396,
  411, 426, 441, 456, 471, 486, 501, 591, 606, 621, 636, 651, 666, 681, 696, 711,
  726, 741, 756, 831, 846, 861, 876, 891, 906, 921, 936, 951, 966, 981, 996, 1011,
  1071, 1086, 1101, 1116, 1131, 1146, 1161, 1176, 1191, 1206, 1221, 1236, 1251, 1266, 1311, 1326,
  1341, 1356, 1371, 1386, 1401, 1416, 1431, 1446, 1461, 1476, 1491, 1506, 1521, 1551, 1566, 1581,
  1596, 1611, 1626, 1641, 1656, 1671, 1686, 1701, 1716, 1731, 1746, 1761, 1776, 1806, 1821, 1836,
  1851, 1866, 1881, 1896, 1911, 1926, 1941, 1956, 1971, 1986, 2001, 2016, 2061, 2076, 2091, 2106,
  2121, 2136, 2151, 2166, 2181, 2196, 2211, 2226, 2241, 2256, 2316, 2331, 2346, 2361, 2376, 2391,
  2406, 2421, 2436, 2451, 2466, 2481, 2496, 2571, 2586, 2601, 2616, 2631, 2646, 2661, 2676, 2691,
  2706, 2721, 2736, 2826, 2841, 2856, 2871, 2886, 2901, 2916, 2931, 2946, 2961, 2976, 3081, 3096,
  3111, 3126, 3141, 3156, 3171, 3186, 3201, 3216, 3336, 3351, 3366, 3381, 3396, 3411, 3426, 3441,
  3456, 3591, 3606, 3621, 3636, 3651, 3666, 3681, 3696, 3846, 3861, 3876, 3891, 3906, 3921, 3936,
  127, 142, 157, 172, 187, 202, 217, 232, 247, 367, 382, 397, 412, 427, 442, 457,
  472, 487, 502, 607, 622, 637, 652, 667, 682, 697, 712, 727, 742, 757, 847, 862,
  877, 892, 907, 922, 937, 952, 967, 982, 997, 1012, 1087, 1102, 1117, 1132, 1147, 1162,
  1177, 1192, 1207, 1222, 1237, 1252, 1267, 1327, 1342, 1357, 1372, 1387, 1402, 1417, 1432, 1447,
  1462, 1477, 1492, 1507, 1522, 1567, 1582, 1597, 1612, 1627, 1642, 1657, 1672, 1687, 1702, 1717,
  1732, 1747, 1762, 1777, 1807, 1822, 1837, 1852, 1867, 1882, 1897, 1912, 1927, 1942, 1957, 1972,
  1987, 2002, 2017, 2032, 2062, 2077, 2092, 2107, 2122, 2137, 2152, 2167, 2182, 2197, 2212, 2227,
  2242, 2257, 2272, 2317, 2332, 2347, 2362, 2377, 2392, 2407, 2422, 2437, 2452, 2467, 2482, 2497,
  2512, 2572, 2587, 2602, 2617, 2632, 2647, 2662, 2677, 2692, 2707, 2722, 2737, 2752, 2827, 2842,
  2857, 2872, 2887, 2902, 2917, 2932, 2947, 2962, 2977, 2992, 3082, 3097, 3112, 3127, 3142, 3157,
  3172, 3187, 3202, 3217, 3232, 3337, 3352, 3367, 3382, 3397, 3412, 3427, 3442, 3457, 3472, 3592,
  3607, 3622, 3637, 3652, 3667, 3682, 3697, 3712, 3847, 3862, 3877, 3892, 3907, 3922, 3937, 3952,
  143, 158, 173, 188, 203, 218, 233, 248, 383, 398, 413, 428, 443, 458, 473, 488,
  503, 623, 638, 653, 668, 683, 698, 713, 728, 743, 758, 863, 878, 893, 908, 923,
  938, 953, 968, 983, 998, 1013, 1103, 1118, 1133, 1148, 1163, 1178, 1193, 1208, 1223, 1238,
  1253, 1268, 1343, 1358, 1373, 1388, 1403, 1418, 1433, 1448, 1463, 1478, 1493, 1508, 1523, 1583,
  1598, 1613, 1628, 1643, 1658, 1673, 1688, 1703, 1718, 1733, 1748, 1763, 1778, 1823, 1838, 1853,
  1868, 1883, 1898, 1913, 1928, 1943, 1958, 1973, 1988, 2003, 2018, 2033, 2063, 2078, 2093, 2108,
  2123, 2138, 2153, 2168, 2183, 2198, 2213, 2228, 2243, 2258, 2273, 2288, 2318, 2333, 2348, 2363,
  2378, 2393, 2408, 2423, 2438, 2453, 2468, 2483, 2498, 2513, 2528, 2573, 2588, 2603, 2618, 2633,
  2648, 2663, 2678, 2693, 2708, 2723, 2738, 2753, 2768, 2828, 2843, 2858, 2873, 2888, 2903, 2918,
  2933, 2948, 2963, 2978, 2993, 3008, 3083, 3098, 3113, 3128, 3143, 3158, 3173, 3188, 3203, 3218,
  3233, 3248, 3338, 3353, 3368, 3383, 3398, 3413, 3428, 3443, 3458, 3473, 3488, 3593, 3608, 3623,
  3638, 3653, 3668, 3683, 3698, 3713, 3728, 3848, 3863, 3878, 3893, 3908, 3923, 3938, 3953, 3968,
  159, 174, 189, 204, 219, 234, 249, 399, 414, 429, 444, 459, 474, 489, 504, 639,
  654, 669, 684, 699, 714, 729, 744, 759, 879, 894, 909, 924, 939, 954, 969, 984,
  999, 1014, 1119, 1134, 1149, 1164, 1179, 1194, 1209, 1224, 1239, 1254, 1269, 1359, 1374, 1389,
  1404, 1419, 1434, 1449, 1464, 1479, 1494, 1509, 1524, 1599, 1614, 1629, 1644, 1659, 1674, 1689,
  1704, 1719, 1734, 1749, 1764, 1779, 1839, 1854, 1869, 1884, 1899, 1914, 1929, 1944, 1959, 1974,
  1989, 2004, 2019, 2034, 2079, 2094, 2109, 2124, 2139, 2154, 2169, 2184, 2199, 2214, 2229, 2244,
  2259, 2274, 2289, 2319, 2334, 2349, 2364, 2379, 2394, 2409, 2424, 2439, 2454, 2469, 2484, 2499,
  2514, 2529, 2544, 2574, 2589, 2604, 2619, 2634, 2649, 2664, 2679, 2694, 2709, 2724, 2739, 2754,
  2769, 2784, 2829, 2844, 2859, 2874, 2889, 2904, 2919, 2934, 2949, 2964, 2979, 2994, 3009, 3024,
  3084, 3099, 3114, 3129, 3144, 3159, 3174, 3189, 3204, 3219, 3234, 3249, 3264, 3339, 3354, 3369,
  3384, 3399, 3414, 3429, 3444, 3459, 3474, 3489, 3504, 3594, 3609, 3624, 3639, 3654, 3669, 3684,
  3699, 3714, 3729, 3744, 3849, 3864, 3879, 3894, 3909, 3924, 3939, 3954, 3969, 3984, 175, 190,
  205, 220, 235, 250, 415, 430, 445, 460, 475, 490, 505, 655, 670, 685, 700, 715,
  730, 745, 760, 895, 910, 925, 940, 955, 970, 985, 1000, 1015, 1135, 1150, 1165, 1180,
  1195, 1210, 1225, 1240, 1255, 1270, 1375, 1390, 1405, 1420, 1435, 1450, 1465, 1480, 1495, 1510,
  1525, 1615, 1630, 1645, 1660, 1675, 1690, 1705, 1720, 1735, 1750, 1765, 1780, 1855, 1870, 1885,
  1900, 1915, 1930, 1945, 1960, 1975, 1990, 2005, 2020, 2035, 2095, 2110, 2125, 2140, 2155, 2170,
  2185, 2200, 2215, 2230, 2245, 2260, 2275, 2290, 2335, 2350, 2365, 2380, 2395, 2410, 2425, 2440,
  2455, 2470, 2485, 2500, 2515, 2530, 2545, 2575, 2590, 2605, 2620, 2635, 2650, 2665, 2680, 2695,
  2710, 2725, 2740, 2755, 2770, 2785, 2800, 2830, 2845, 2860, 2875, 2890, 2905, 2920, 2935, 2950,
  2965, 2980, 2995, 3010, 3025, 3040, 3085, 3100, 3115, 3130, 3145, 3160, 3175, 3190, 3205, 3220,
  3235, 3250, 3265, 3280, 3340, 3355, 3370, 3385, 3400, 3415, 3430, 3445, 3460, 3475, 3490, 3505,
  3520, 3595, 3610, 3625, 3640, 3655, 3670, 3685, 3700, 3715, 3730, 3745, 3760, 3850, 3865, 3880,
  3895, 3910, 3925, 3940, 3955, 3970, 3985, 4000, 191, 206, 221, 236, 251, 431, 446, 461,
  476, 491, 506, 671, 686, 701, 716, 731, 746, 761, 911, 926, 941, 956, 971, 986,
  1001, 1016, 1151, 1166, 1181, 1196, 1211, 1226, 1241, 1256, 1271, 1391, 1406, 1421, 1436, 1451,
  1466, 1481, 1496, 1511, 1526, 1631, 1646, 1661, 1676, 1691, 1706, 1721, 1736, 1751, 1766, 1781,
  1871, 1886, 1901, 1916, 1931, 1946, 1961, 1976, 1991, 2006, 2021, 2036, 2111, 2126, 2141, 2156,
  2171, 2186, 2201, 2216, 2231, 2246, 2261, 2276, 2291, 2351, 2366, 2381, 2396, 2411, 2426, 2441,
  2456, 2471, 2486, 2501, 2516, 2531, 2546, 2591, 2606, 2621, 2636, 2651, 2666, 2681, 2696, 2711,
  2726, 2741, 2756, 2771, 2786, 2801, 2831, 2846, 2861, 2876, 2891, 2906, 2921, 2936, 2951, 2966,
  2981, 2996, 3011, 3026, 3041, 3056, 3086, 3101, 3116, 3131, 3146, 3161, 3176, 3191, 3206, 3221,
  3236, 3251, 3266, 3281, 3296, 3341, 3356, 3371, 3386, 3401, 3416, 3431, 3446, 3461, 3476, 3491,
  3506, 3521, 3536, 3596, 3611, 3626, 3641, 3656, 3671, 3686, 3701, 3716, 3731, 3746, 3761, 3776,
  3851, 3866, 3881, 3896, 3911, 3926, 3941, 3956, 3971, 3986, 4001, 4016, 207, 222, 237, 252,
  447, 462, 477, 492, 507, 687, 702, 717, 732, 747, 762, 927, 942, 957, 972, 987,
  1002, 1017, 1167, 1182, 1197, 1212, 1227, 1242, 1257, 1272, 1407, 1422, 1437, 1452, 1467, 1482,
  1497, 1512, 1527, 1647, 1662, 1677, 1692, 1707, 1722, 1737, 1752, 1767, 1782, 1887, 1902, 1917,
  1932, 1947, 1962, 1977, 1992, 2007, 2022, 2037, 2127, 2142, 2157, 2172, 2187, 2202, 2217, 2232,
  2247, 2262, 2277, 2292, 2367, 2382, 2397, 2412, 2427, 2442, 2457, 2472, 2487, 2502, 2517, 2532,
  2547, 2607, 2622, 2637, 2652, 2667, 2682, 2697, 2712, 2727, 2742, 2757, 2772, 2787, 2802, 2847,
  2862, 2877, 2892, 2907, 2922, 2937, 2952, 2967, 2982, 2997, 3012, 3027, 3042, 3057, 3087, 3102,
  3117, 3132, 3147, 3162, 3177, 3192, 3207, 3222, 3237, 3252, 3267, 3282, 3297, 3312, 3342, 3357,
  3372, 3387, 3402, 3417, 3432, 3447, 3462, 3477, 3492, 3507, 3522, 3537, 3552, 3597, 3612, 3627,
  3642, 3657, 3672, 3687, 3702, 3717, 3732, 3747, 3762, 3777, 3792, 3852, 3867, 3882, 3897, 3912,
  3927, 3942, 3957, 3972, 3987, 4002, 4017, 4032, 223, 238, 253, 463, 478, 493, 508, 703,
  718, 733, 748, 763, 943, 958, 973, 988, 1003, 1018, 1183, 1198, 1213, 1228, 1243, 1258,
  1273, 1423, 1438, 1453, 1468, 1483, 1498, 1513, 1528, 1663, 1678, 1693, 1708, 1723, 1738, 1753,
  1768, 1783, 1903, 1918, 1933, 1948, 1963, 1978, 1993, 2008, 2023, 2038, 2143, 2158, 2173, 2188,
  2203, 2218, 2233, 2248, 2263, 2278, 2293, 2383, 2398, 2413, 2428, 2443, 2458, 2473, 2488, 2503,
  2518, 2533, 2548, 2623, 2638, 2653, 2668, 2683, 2698, 2713, 2728, 2743, 2758, 2773, 2788, 2803,
  2863, 2878, 2893, 2908, 2923, 2938, 2953, 2968, 2983, 2998, 3013, 3028, 3043, 3058, 3103, 3118,
  3133, 3148, 3163, 3178, 3193, 3208, 3223, 3238, 3253, 3268, 3283, 3298, 3313, 3343, 3358, 3373,
  3388, 3403, 3418, 3433, 3448, 3463, 3478, 3493, 3508, 3523, 3538, 3553, 3568, 3598, 3613, 3628,
  3643, 3658, 3673, 3688, 3703, 3718, 3733, 3748, 3763, 3778, 3793, 3808, 3853, 3868, 3883, 3898,
  3913, 3928, 3943, 3958, 3973, 3988, 4003, 4018, 4033, 4048, 239, 254, 479, 494, 509, 719,
  734, 749, 764, 959, 974, 989, 1004, 1019, 1199, 1214, 1229, 1244, 1259, 1274, 1439, 1454,
  1469, 1484, 1499, 1514, 1529, 1679, 1694, 1709, 1724, 1739, 1754, 1769, 1784, 1919, 1934, 1949,
  1964, 1979, 1994, 2009, 2024, 2039, 2159, 2174, 2189, 2204, 2219, 2234, 2249, 2264, 2279, 2294,
  2399, 2414, 2429, 2444, 2459, 2474, 2489, 2504, 2519, 2534, 2549, 2639, 2654, 2669, 2684, 2699,
  2714, 2729, 2744, 2759, 2774, 2789, 2804, 2879, 2894, 2909, 2924, 2939, 2954, 2969, 2984, 2999,
  3014, 3029, 3044, 3059, 3119, 3134, 3149, 3164, 3179, 3194, 3209, 3224, 3239, 3254, 3269, 3284,
  3299, 3314, 3359, 3374, 3389, 3404, 3419, 3434, 3449, 3464, 3479, 3494, 3509, 3524, 3539, 3554,
  3569, 3599, 3614, 3629, 3644, 3659, 3674, 3689, 3704, 3719, 3734, 3749, 3764, 3779, 3794, 3809,
  3824, 3854, 3869, 3884, 3899, 3914, 3929, 3944, 3959, 3974, 3989, 4004, 4019, 4034, 4049, 4064,
  255, 495, 510, 735, 750, 765, 975, 990, 1005, 1020, 1215, 1230, 1245, 1260, 1275, 1455,
  1470, 1485, 1500, 1515, 1530, 1695, 1710, 1725, 1740, 1755, 1770, 1785, 1935, 1950, 1965, 1980,
  1995, 2010, 2025, 2040, 2175, 2190, 2205, 2220, 2235, 2250, 2265, 2280, 2295, 2415, 2430, 2445,
  2460, 2475, 2490, 2505, 2520, 2535, 2550, 2655, 2670, 2685, 2700, 2715, 2730, 2745, 2760, 2775,
  2790, 2805, 2895, 2910, 2925, 2940, 2955, 2970, 2985, 3000, 3015, 3030, 3045, 3060, 3135, 3150,
  3165, 3180, 3195, 3210, 3225, 3240, 3255, 3270, 3285, 3300, 3315, 3375, 3390, 3405, 3420, 3435,
  3450, 3465, 3480, 3495, 3510, 3525, 3540, 3555, 3570, 3615, 3630, 3645, 3660, 3675, 3690, 3705,
  3720, 3735, 3750, 3765, 3780, 3795, 3810, 3825, 3855, 3870, 3885, 3900, 3915, 3930, 3945, 3960,
  3975, 3990, 4005, 4020, 4035, 4050, 4065, 4080, 511, 751, 766, 991, 1006, 1021, 1231, 1246,
  1261, 1276, 1471, 1486, 1501, 1516, 1531, 1711, 1726, 1741, 1756, 1771, 1786, 1951, 1966, 1981,
  1996, 2011, 2026, 2041, 2191, 2206, 2221, 2236, 2251, 2266, 2281, 2296, 2431, 2446, 2461, 2476,
  2491, 2506, 2521, 2536, 2551, 2671, 2686, 2701, 2716, 2731, 2746, 2761, 2776, 2791, 2806, 2911,
  2926, 2941, 2956, 2971, 2986, 3001, 3016, 3031, 3046, 3061, 3151, 3166, 3181, 3196, 3211, 3226,
  3241, 3256, 3271, 3286, 3301, 3316, 3391, 3406, 3421, 3436, 3451, 3466, 3481, 3496, 3511, 3526,
  3541, 3556, 3571, 3631, 3646, 3661, 3676, 3691, 3706, 3721, 3736, 3751, 3766, 3781, 3796, 3811,
  3826, 3871, 3886, 3901, 3916, 3931, 3946, 3961, 3976, 3991, 4006, 4021, 4036, 4051, 4066, 4081,
  767, 1007, 1022, 1247, 1262, 1277, 1487, 1502, 1517, 1532, 1727, 1742, 1757, 1772, 1787, 1967,
  1982, 1997, 2012, 2027, 2042, 2207, 2222, 2237, 2252, 2267, 2282, 2297, 2447, 2462, 2477, 2492,
  2507, 2522, 2537, 2552, 2687, 2702, 2717, 2732, 2747, 2762, 2777, 2792, 2807, 2927, 2942, 2957,
  2972, 2987, 3002, 3017, 3032, 3047, 3062, 3167, 3182, 3197, 3212, 3227, 3242, 3257, 3272, 3287,
  3302, 3317, 3407, 3422, 3437, 3452, 3467, 3482, 3497, 3512, 3527, 3542, 3557, 3572, 3647, 3662,
  3677, 3692, 3707, 3722, 3737, 3752, 3767, 3782, 3797, 3812, 3827, 3887, 3902, 3917, 3932, 3947,
  3962, 3977, 3992, 4007, 4022, 4037, 4052, 4067, 4082, 1023, 1263, 1278, 1503, 1518, 1533, 1743,
  1758, 1773, 1788, 1983, 1998, 2013, 2028, 2043, 2223, 2238, 2253, 2268, 2283, 2298, 2463, 2478,
  2493, 2508, 2523, 2538, 2553, 2703, 2718, 2733, 2748, 2763, 2778, 2793, 2808, 2943, 2958, 2973,
  2988, 3003, 3018, 3033, 3048, 3063, 3183, 3198, 3213, 3228, 3243, 3258, 3273, 3288, 3303, 3318,
  3423, 3438, 3453, 3468, 3483, 3498, 3513, 3528, 3543, 3558, 3573, 3663, 3678, 3693, 3708, 3723,
  3738, 3753, 3768, 3783, 3798, 3813, 3828, 3903, 3918, 3933, 3948, 3963, 3978, 3993, 4008, 4023,
  4038, 4053, 4068, 4083, 1279, 1519, 1534, 1759, 1774, 1789, 1999, 2014, 2029, 2044, 2239, 2254,
  2269, 2284, 2299, 2479, 2494, 2509, 2524, 2539, 2554, 2719, 2734, 2749, 2764, 2779, 2794, 2809,
  2959, 2974, 2989, 3004, 3019, 3034, 3049, 3064, 3199, 3214, 3229, 3244, 3259, 3274, 3289, 3304,
  3319, 3439, 3454, 3469, 3484, 3499, 3514, 3529, 3544, 3559, 3574, 3679, 3694, 3709, 3724, 3739,
  3754, 3769, 3784, 3799, 3814, 3829, 3919, 3934, 3949, 3964, 3979, 3994, 4009, 4024, 4039, 4054,
  4069, 4084, 1535, 1775, 1790, 2015, 2030, 2045, 2255, 2270, 2285, 2300, 2495, 2510, 2525, 2540,
  2555, 2735, 2750, 2765, 2780, 2795, 2810, 2975, 2990, 3005, 3020, 3035, 3050, 3065, 3215, 3230,
  3245, 3260, 3275, 3290, 3305, 3320, 3455, 3470, 3485, 3500, 3515, 3530, 3545, 3560, 3575, 3695,
  3710, 3725, 3740, 3755, 3770, 3785, 3800, 3815, 3830, 3935, 3950, 3965, 3980, 3995, 4010, 4025,
  4040, 4055, 4070, 4085, 1791, 2031, 2046, 2271, 2286, 2301, 2511, 2526, 2541, 2556, 2751, 2766,
  2781, 2796, 2811, 2991, 3006, 3021, 3036, 3051, 3066, 3231, 3246, 3261, 3276, 3291, 3306, 3321,
  3471, 3486, 3501, 3516, 3531, 3546, 3561, 3576, 3711, 3726, 3741, 3756, 3771, 3786, 3801, 3816,
  3831, 3951, 3966, 3981, 3996, 4011, 4026, 4041, 4056, 4071, 4086, 2047, 2287, 2302, 2527, 2542,
  2557, 2767, 2782, 2797, 2812, 3007, 3022, 3037, 3052, 3067, 3247, 3262, 3277, 3292, 3307, 3322,
  3487, 3502, 3517, 3532, 3547, 3562, 3577, 3727, 3742, 3757, 3772, 3787, 3802, 3817, 3832, 3967,
  3982, 3997, 4012, 4027, 4042, 4057, 4072, 4087, 2303, 2543, 2558, 2783, 2798, 2813, 3023, 3038,
  3053, 3068, 3263, 3278, 3293, 3308, 3323, 3503, 3518, 3533, 3548, 3563, 3578, 3743, 3758, 3773,
  3788, 3803, 3818, 3833, 3983, 3998, 4013, 4028, 4043, 4058, 4073, 4088, 2559, 2799, 2814, 3039,
  3054, 3069, 3279, 3294, 3309, 3324, 3519, 3534, 3549, 3564, 3579, 3759, 3774, 3789, 3804, 3819,
  3834, 3999, 4014, 4029, 4044, 4059, 4074, 4089, 2815, 3055, 3070, 3295, 3310, 3325, 3535, 3550,
  3565, 3580, 3775, 3790, 3805, 3820, 3835, 4015, 4030, 4045, 4060, 4075, 4090, 3071, 3311, 3326,
  3551, 3566, 3581, 3791, 3806, 3821, 3836, 4031, 4046, 4061, 4076, 4091, 3327, 3567, 3582, 3807,
  3822, 3837, 4047, 4062, 4077, 4092, 3583, 3823, 3838, 4063, 4078, 4093, 3839, 4079, 4094, 4095,
};
/* inverse scan: natural index -> scan position */
static const uint16_t VF_ISCAN16[4096] = {
  0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680,
  2, 5, 11, 21, 36, 57, 85, 121, 166, 221, 287, 365, 456, 561, 681, 816,
  6, 12, 22, 37, 58, 86, 122, 167, 222, 288, 366, 457, 562, 682, 817, 966,
  13, 23, 38, 59, 87, 123, 168, 223, 289, 367, 458, 563, 683, 818, 967, 1128,
  24, 39, 60, 88, 124, 169, 224, 290, 368, 459, 564, 684, 819, 968, 1129, 1300,
  40, 61, 89, 125, 170, 225, 291, 369, 460, 565, 685, 820, 969, 1130, 1301, 1480,
  62, 90, 126, 171, 226, 292, 370, 461, 566, 686, 821, 970, 1131, 1302, 1481, 1666,
  91, 127, 172, 227, 293, 371, 462, 567, 687, 822, 971, 1132, 1303, 1482, 1667, 1856,
  128, 173, 228, 294, 372, 463, 568, 688, 823, 972, 1133, 1304, 1483, 1668, 1857, 2048,
  174, 229, 295, 373, 464, 569, 689, 824, 973, 1134, 1305, 1484, 1669, 1858, 2049, 2240,
  230, 296, 374, 465, 570, 690, 825, 974, 1135, 1306, 1485, 1670, 1859, 2050, 2241, 2430,
  297, 375, 466, 571, 691, 826, 975, 1136, 1307, 1486, 1671, 1860, 2051, 2242, 2431, 2616,
  376, 467, 572, 692, 827, 976, 1137, 1308, 1487, 1672, 1861, 2052, 2243, 2432, 2617, 2796,
  468, 573, 693, 828, 977, 1138, 1309, 1488, 1673, 1862, 2053, 2244, 2433, 2618, 2797, 2968,
  574, 694, 829, 978, 1139, 1310, 1489, 1674, 1863, 2054, 2245, 2434, 2619, 2798, 2969, 3130,
  695, 830, 979, 1140, 1311, 1490, 1675, 1864, 2055, 2246, 2435, 2620, 2799, 2970, 3131, 3280,
  3, 7, 14, 25, 41, 63, 92, 129, 175, 231, 298, 377, 469, 575, 696, 831,
  8, 15, 26, 42, 64, 93, 130, 176, 232, 299, 378, 470, 576, 697, 832, 980,
  16, 27, 43, 65, 94, 131, 177, 233, 300, 379, 471, 577, 698, 833, 981, 1141,
  28, 44, 66, 95, 132, 178, 234, 301, 380, 472, 578, 699, 834, 982, 1142, 1312,
  45, 67, 96, 133, 179, 235, 302, 381, 473, 579, 700, 835, 983, 1143, 1313, 1491,
  68, 97, 134, 180, 236, 303, 382, 474, 580, 701, 836, 984, 1144, 1314, 1492, 1676,
  98, 135, 181, 237, 304, 383, 475, 581, 702, 837, 985, 1145, 1315, 1493, 1677, 1865,
  136, 182, 238, 305, 384, 476, 582, 703, 838, 986, 1146, 1316, 1494, 1678, 1866, 2056,
  183, 239, 306, 385, 477, 583, 704, 839, 987, 1147, 1317, 1495, 1679, 1867, 2057, 2247,
  240, 307, 386, 478, 584, 705, 840, 988, 1148, 1318, 1496, 1680, 1868, 2058, 2248, 2436,
  308, 387, 479, 585, 706, 841, 989, 1149, 1319, 1497, 1681, 1869, 2059, 2249, 2437, 2621,
  388, 480, 586, 707, 842, 990, 1150, 1320, 1498, 1682, 1870, 2060, 2250, 2438, 2622, 2800,
  481, 587, 708, 843, 991, 1151, 1321, 1499, 1683, 1871, 2061, 2251, 2439, 2623, 2801, 2971,
  588, 709, 844, 992, 1152, 1322, 1500, 1684, 1872, 2062, 2252, 2440, 2624, 2802, 2972, 3132,
  710, 845, 993, 1153, 1323, 1501, 1685, 1873, 2063, 2253, 2441, 2625, 2803, 2973, 3133, 3281,
  846, 994, 1154, 1324, 1502, 1686, 1874, 2064, 2254, 2442, 2626, 2804, 2974, 3134, 3282, 3416,
  9, 17, 29, 46, 69, 99, 137, 184, 241, 309, 389, 482, 589, 711, 847, 995,
  18, 30, 47, 70, 100, 138, 185, 242, 310, 390, 483, 590, 712, 848, 996, 1155,
  31, 48, 71, 101, 139, 186, 243, 311, 391, 484, 591, 713, 849, 997, 1156, 1325,
  49, 72, 102, 140, 187, 244, 312, 392, 485, 592, 714, 850, 998, 1157, 1326, 1503,
  73, 103, 141, 188, 245, 313, 393, 486, 593, 715, 851, 999, 1158, 1327, 1504, 1687,
  104, 142, 189, 246, 314, 394, 487, 594, 716, 852, 1000, 1159, 1328, 1505, 1688, 1875,
  143, 190, 247, 315, 395, 488, 595, 717, 853, 1001, 1160, 1329, 1506, 1689, 1876, 2065,
  191, 248, 316, 396, 489, 596, 718, 854, 1002, 1161, 1330, 1507, 1690, 1877, 2066, 2255,
  249, 317, 397, 490, 597, 719, 855, 1003, 1162, 1331, 1508, 1691, 1878, 2067, 2256, 2443,
  318, 398, 491, 598, 720, 856, 1004, 1163, 1332, 1509, 1692, 1879, 2068, 2257, 2444, 2627,
  399, 492, 599, 721, 857, 1005, 1164, 1333, 1510, 1693, 1880, 2069, 2258, 2445, 2628, 2805,
  493, 600, 722, 858, 1006, 1165, 1334, 1511, 1694, 1881, 2070, 2259, 2446, 2629, 2806, 2975,
  601, 723, 859, 1007, 1166, 1335, 1512, 1695, 1882, 2071, 2260, 2447, 2630, 2807, 2976, 3135,
  724, 860, 1008, 1167, 1336, 1513, 1696, 1883, 2072, 2261, 2448, 2631, 2808, 2977, 3136, 3283,
  861, 1009, 1168, 1337, 1514, 1697, 1884, 2073, 2262, 2449, 2632, 2809, 2978, 3137, 3284, 3417,
  1010, 1169, 1338, 1515, 1698, 1885, 2074, 2263, 2450, 2633, 2810, 2979, 3138, 3285, 3418, 3536,
  19, 32, 50, 74, 105, 144, 192, 250, 319, 400, 494, 602, 725, 862, 1011, 1170,
  33, 51, 75, 106, 145, 193, 251, 320, 401, 495, 603, 726, 863, 1012, 1171, 1339,
  52, 76, 107, 146, 194, 252, 321, 402, 496, 604, 727, 864, 1013, 1172, 1340, 1516,
  77, 108, 147, 195, 253, 322, 403, 497, 605, 728, 865, 1014, 1173, 1341, 1517, 1699,
  109, 148, 196, 254, 323, 404, 498, 606, 729, 866, 1015, 1174, 1342, 1518, 1700, 1886,
  149, 197, 255, 324, 405, 499, 607, 730, 867, 1016, 1175, 1343, 1519, 1701, 1887, 2075,
  198, 256, 325, 406, 500, 608, 731, 868, 1017, 1176, 1344, 1520, 1702, 1888, 2076, 2264,
  257, 326, 407, 501, 609, 732, 869, 1018, 1177, 1345, 1521, 1703, 1889, 2077, 2265, 2451,
  327, 408, 502, 610, 733, 870, 1019, 1178, 1346, 1522, 1704, 1890, 2078, 2266, 2452, 2634,
  409, 503, 611, 734, 871, 1020, 1179, 1347, 1523, 1705, 1891, 2079, 2267, 2453, 2635, 2811,
  504, 612, 735, 872, 1021, 1180, 1348, 1524, 1706, 1892, 2080, 2268, 2454, 2636, 2812, 2980,
  613, 736, 873, 1022, 1181, 1349, 1525, 1707, 1893, 2081, 2269, 2455, 2637, 2813, 2981, 3139,
  737, 874, 1023, 1182, 1350, 1526, 1708, 1894, 2082, 2270, 2456, 2638, 2814, 2982, 3140, 3286,
  875, 1024, 1183, 1351, 1527, 1709, 1895, 2083, 2271, 2457, 2639, 2815, 2983, 3141, 3287, 3419,
  1025, 1184, 1352, 1528, 1710, 1896, 2084, 2272, 2458, 2640, 2816, 2984, 3142, 3288, 3420, 3537,
  1185, 1353, 1529, 1711, 1897, 2085, 2273, 2459, 2641, 2817, 2985, 3143, 3289, 3421, 3538, 3641,
  34, 53, 78, 110, 150, 199, 258, 328, 410, 505, 614, 738, 876, 1026, 1186, 1354,
  54, 79, 111, 151, 200, 259, 329, 411, 506, 615, 739, 877, 1027, 1187, 1355, 1530,
  80, 112, 152, 201, 260, 330, 412, 507, 616, 740, 878, 1028, 1188, 1356, 1531, 1712,
  113, 153, 202, 261, 331, 413, 508, 617, 741, 879, 1029, 1189, 1357, 1532, 1713, 1898,
  154, 203, 262, 332, 414, 509, 618, 742, 880, 1030, 1190, 1358, 1533, 1714, 1899, 2086,
  204, 263, 333, 415, 510, 619, 743, 881, 1031, 1191, 1359, 1534, 1715, 1900, 2087, 2274,
  264, 334, 416, 511, 620, 744, 882, 1032, 1192, 1360, 1535, 1716, 1901, 2088, 2275, 2460,
  335, 417, 512, 621, 745, 883, 1033, 1193, 1361, 1536, 1717, 1902, 2089, 2276, 2461, 2642,
  418, 513, 622, 746, 884, 1034, 1194, 1362, 1537, 1718, 1903, 2090, 2277, 2462, 2643, 2818,
  514, 623, 747, 885, 1035, 1195, 1363, 1538, 1719, 1904, 2091, 2278, 2463, 2644, 2819, 2986,
  624, 748, 886, 1036, 1196, 1364, 1539, 1720, 1905, 2092, 2279, 2464, 2645, 2820, 2987, 3144,
  749, 887, 1037, 1197, 1365, 1540, 1721, 1906, 2093, 2280, 2465, 2646, 2821, 2988, 3145, 3290,
  888, 1038, 1198, 1366, 1541, 1722, 1907, 2094, 2281, 2466, 2647, 2822, 2989, 3146, 3291, 3422,
  1039, 1199, 1367, 1542, 1723, 1908, 2095, 2282, 2467, 2648, 2823, 2990, 3147, 3292, 3423, 3539,
  1200, 1368, 1543, 1724, 1909, 2096, 2283, 2468, 2649, 2824, 2991, 3148, 3293, 3424, 3540, 3642,
  1369, 1544, 1725, 1910, 2097, 2284, 2469, 2650, 2825, 2992, 3149, 3294, 3425, 3541, 3643, 3732,
  55, 81, 114, 155, 205, 265, 336, 419, 515, 625, 750, 889, 1040, 1201, 1370, 1545,
  82, 115, 156, 206, 266, 337, 420, 516, 626, 751, 890, 1041, 1202, 1371, 1546, 1726,
  116, 157, 207, 267, 338, 421, 517, 627, 752, 891, 1042, 1203, 1372, 1547, 1727, 1911,
  158, 208, 268, 339, 422, 518, 628, 753, 892, 1043, 1204, 1373, 1548, 1728, 1912, 2098,
  209, 269, 340, 423, 519, 629, 754, 893, 1044, 1205, 1374, 1549, 1729, 1913, 2099, 2285,
  270, 341, 424, 520, 630, 755, 894, 1045, 1206, 1375, 1550, 1730, 1914, 2100, 2286, 2470,
  342, 425, 521, 631, 756, 895, 1046, 1207, 1376, 1551, 1731, 1915, 2101, 2287, 2471, 2651,
  426, 522, 632, 757, 896, 1047, 1208, 1377, 1552, 1732, 1916, 2102, 2288, 2472, 2652, 2826,
  523, 633, 758, 897, 1048, 1209, 1378, 1553, 1733, 1917, 2103, 2289, 2473, 2653, 2827, 2993,
  634, 759, 898, 1049, 1210, 1379, 1554, 1734, 1918, 2104, 2290, 2474, 2654, 2828, 2994, 3150,
  760, 899, 1050, 1211, 1380, 1555, 1735, 1919, 2105, 2291, 2475, 2655, 2829, 2995, 3151, 3295,
  900, 1051, 1212, 1381, 1556, 1736, 1920, 2106, 2292, 2476, 2656, 2830, 2996, 3152, 3296, 3426,
  1052, 1213, 1382, 1557, 1737, 1921, 2107, 2293, 2477, 2657, 2831, 2997, 3153, 3297, 3427, 3542,
  1214, 1383, 1558, 1738, 1922, 2108, 2294, 2478, 2658, 2832, 2998, 3154, 3298, 3428, 3543, 3644,
  1384, 1559, 1739, 1923, 2109, 2295, 2479, 2659, 2833, 2999, 3155, 3299, 3429, 3544, 3645, 3733,
  1560, 1740, 1924, 2110, 2296, 2480, 2660, 2834, 3000, 3156, 3300, 3430, 3545, 3646, 3734, 3810,
  83, 117, 159, 210, 271, 343, 427, 524, 635, 761, 901, 1053, 1215, 1385, 1561, 1741,
  118, 160, 211, 272, 344, 428, 525, 636, 762, 902, 1054, 1216, 1386, 1562, 1742, 1925,
  161, 212, 273, 345, 429, 526, 637, 763, 903, 1055, 1217, 1387, 1563, 1743, 1926, 2111,
  213, 274, 346, 430, 527, 638, 764, 904, 1056, 1218, 1388, 1564, 1744, 1927, 2112, 2297,
  275, 347, 431, 528, 639, 765, 905, 1057, 1219, 1389, 1565, 1745, 1928, 2113, 2298, 2481,
  348, 432, 529, 640, 766, 906, 1058, 1220, 1390, 1566, 1746, 1929, 2114, 2299, 2482, 2661,
  433, 530, 641, 767, 907, 1059, 1221, 1391, 1567, 1747, 1930, 2115, 2300, 2483, 2662, 2835,
  531, 642, 768, 908, 1060, 1222, 1392, 1568, 1748, 1931, 2116, 2301, 2484, 2663, 2836, 3001,
  643, 769, 909, 1061, 1223, 1393, 1569, 1749, 1932, 2117, 2302, 2485, 2664, 2837, 3002, 3157,
  770, 910, 1062, 1224, 1394, 1570, 1750, 1933, 2118, 2303, 2486, 2665, 2838, 3003, 3158, 3301,
  911, 1063, 1225, 1395, 1571, 1751, 1934, 2119, 2304, 2487, 2666, 2839, 3004, 3159, 3302, 3431,
  1064, 1226, 1396, 1572, 1752, 1935, 2120, 2305, 2488, 2667, 2840, 3005, 3160, 3303, 3432, 3546,
  1227, 1397, 1573, 1753, 1936, 2121, 2306, 2489, 2668, 2841, 3006, 3161, 3304, 3433, 3547, 3647,
  1398, 1574, 1754, 1937, 2122, 2307, 2490, 2669, 2842, 3007, 3162, 3305, 3434, 3548, 3648, 3735,
  1575, 1755, 1938, 2123, 2308, 2491, 2670, 2843, 3008, 3163, 3306, 3435, 3549, 3649, 3736, 3811,
  1756, 1939, 2124, 2309, 2492, 2671, 2844, 3009, 3164, 3307, 3436, 3550, 3650, 3737, 3812, 3876,
  119, 162, 214, 276, 349, 434, 532, 644, 771, 912, 1065, 1228, 1399, 1576, 1757, 1940,
  163, 215, 277, 350, 435, 533, 645, 772, 913, 1066, 1229, 1400, 1577, 1758, 1941, 2125,
  216, 278, 351, 436, 534, 646, 773, 914, 1067, 1230, 1401, 1578, 1759, 1942, 2126, 2310,
  279, 352, 437, 535, 647, 774, 915, 1068, 1231, 1402, 1579, 1760, 1943, 2127, 2311, 2493,
  353, 438, 536, 648, 775, 916, 1069, 1232, 1403, 1580, 1761, 1944, 2128, 2312, 2494, 2672,
  439, 537, 649, 776, 917, 1070, 1233, 1404, 1581, 1762, 1945, 2129, 2313, 2495, 2673, 2845,
  538, 650, 777, 918, 1071, 1234, 1405, 1582, 1763, 1946, 2130, 2314, 2496, 2674, 2846, 3010,
  651, 778, 919, 1072, 1235, 1406, 1583, 1764, 1947, 2131, 2315, 2497, 2675, 2847, 3011, 3165,
  779, 920, 1073, 1236, 1407, 1584, 1765, 1948, 2132, 2316, 2498, 2676, 2848, 3012, 3166, 3308,
  921, 1074, 1237, 1408, 1585, 1766, 1949, 2133, 2317, 2499, 2677, 2849, 3013, 3167, 3309, 3437,
  1075, 1238, 1409, 1586, 1767, 1950, 2134, 2318, 2500, 2678, 2850, 3014, 3168, 3310, 3438, 3551,
  1239, 1410, 1587, 1768, 1951, 2135, 2319, 2501, 2679, 2851, 3015, 3169, 3311, 3439, 3552, 3651,
  1411, 1588, 1769, 1952, 2136, 2320, 2502, 2680, 2852, 3016, 3170, 3312, 3440, 3553, 3652, 3738,
  1589, 1770, 1953, 2137, 2321, 2503, 2681, 2853, 3017, 3171, 3313, 3441, 3554, 3653, 3739, 3813,
  1771, 1954, 2138, 2322, 2504, 2682, 2854, 3018, 3172, 3314, 3442, 3555, 3654, 3740, 3814, 3877,
  1955, 2139, 2323, 2505, 2683, 2855, 3019, 3173, 3315, 3443, 3556, 3655, 3741, 3815, 3878, 3931,
  164, 217, 280, 354, 440, 539, 652, 780, 922, 1076, 1240, 1412, 1590, 1772, 1956, 2140,
  218, 281, 355, 441, 540, 653, 781, 923, 1077, 1241, 1413, 1591, 1773, 1957, 2141, 2324,
  282, 356, 442, 541, 654, 782, 924, 1078, 1242, 1414, 1592, 1774, 1958, 2142, 2325, 2506,
  357, 443, 542, 655, 783, 925, 1079, 1243, 1415, 1593, 1775, 1959, 2143, 2326, 2507, 2684,
  444, 543, 656, 784, 926, 1080, 1244, 1416, 1594, 1776, 1960, 2144, 2327, 2508, 2685, 2856,
  544, 657, 785, 927, 1081, 1245, 1417, 1595, 1777, 1961, 2145, 2328, 2509, 2686, 2857, 3020,
  658, 786, 928, 1082, 1246, 1418, 1596, 1778, 1962, 2146, 2329, 2510, 2687, 2858, 3021, 3174,
  787, 929, 1083, 1247, 1419, 1597, 1779, 1963, 2147, 2330, 2511, 2688, 2859, 3022, 3175, 3316,
  930, 1084, 1248, 1420, 1598, 1780, 1964, 2148, 2331, 2512, 2689, 2860, 3023, 3176, 3317, 3444,
  1085, 1249, 1421, 1599, 1781, 1965, 2149, 2332, 2513, 2690, 2861, 3024, 3177, 3318, 3445, 3557,
  1250, 1422, 1600, 1782, 1966, 2150, 2333, 2514, 2691, 2862, 3025, 3178, 3319, 3446, 3558, 3656,
  1423, 1601, 1783, 1967, 2151, 2334, 2515, 2692, 2863, 3026, 3179, 3320, 3447, 3559, 3657, 3742,
  1602, 1784, 1968, 2152, 2335, 2516, 2693, 2864, 3027, 3180, 3321, 3448, 3560, 3658, 3743, 3816,
  1785, 1969, 2153, 2336, 2517, 2694, 2865, 3028, 3181, 3322, 3449, 3561, 3659, 3744, 3817, 3879,
  1970, 2154, 2337, 2518, 2695, 2866, 3029, 3182, 3323, 3450, 3562, 3660, 3745, 3818, 3880, 3932,
  2155, 2338, 2519, 2696, 2867, 3030, 3183, 3324, 3451, 3563, 3661, 3746, 3819, 3881, 3933, 3976,
  219, 283, 358, 445, 545, 659, 788, 931, 1086, 1251, 1424, 1603, 1786, 1971, 2156, 2339,
  284, 359, 446, 546, 660, 789, 932, 1087, 1252, 1425, 1604, 1787, 1972, 2157, 2340, 2520,
  360, 447, 547, 661, 790, 933, 1088, 1253, 1426, 1605, 1788, 1973, 2158, 2341, 2521, 2697,
  448, 548, 662, 791, 934, 1089, 1254, 1427, 1606, 1789, 1974, 2159, 2342, 2522, 2698, 2868,
  549, 663, 792, 935, 1090, 1255, 1428, 1607, 1790, 1975, 2160, 2343, 2523, 2699, 2869, 3031,
  664, 793, 936, 1091, 1256, 1429, 1608, 1791, 1976, 2161, 2344, 2524, 2700, 2870, 3032, 3184,
  794, 937, 1092, 1257, 1430, 1609, 1792, 1977, 2162, 2345, 2525, 2701, 2871, 3033, 3185, 3325,
  938, 1093, 1258, 1431, 1610, 1793, 1978, 2163, 2346, 2526, 2702, 2872, 3034, 3186, 3326, 3452,
  1094, 1259, 1432, 1611, 1794, 1979, 2164, 2347, 2527, 2703, 2873, 3035, 3187, 3327, 3453, 3564,
  1260, 1433, 1612, 1795, 1980, 2165, 2348, 2528, 2704, 2874, 3036, 3188, 3328, 3454, 3565, 3662,
  1434, 1613, 1796, 1981, 2166, 2349, 2529, 2705, 2875, 3037, 3189, 3329, 3455, 3566, 3663, 3747,
  1614, 1797, 1982, 2167, 2350, 2530, 2706, 2876, 3038, 3190, 3330, 3456, 3567, 3664, 3748, 3820,
  1798, 1983, 2168, 2351, 2531, 2707, 2877, 3039, 3191, 3331, 3457, 3568, 3665, 3749, 3821, 3882,
  1984, 2169, 2352, 2532, 2708, 2878, 3040, 3192, 3332, 3458, 3569, 3666, 3750, 3822, 3883, 3934,
  2170, 2353, 2533, 2709, 2879, 3041, 3193, 3333, 3459, 3570, 3667, 3751, 3823, 3884, 3935, 3977,
  2354, 2534, 2710, 2880, 3042, 3194, 3334, 3460, 3571, 3668, 3752, 3824, 3885, 3936, 3978, 4012,
  285, 361, 449, 550, 665, 795, 939, 1095, 1261, 1435, 1615, 1799, 1985, 2171, 2355, 2535,
  362, 450, 551, 666, 796, 940, 1096, 1262, 1436, 1616, 1800, 1986, 2172, 2356, 2536, 2711,
  451, 552, 667, 797, 941, 1097, 1263, 1437, 1617, 1801, 1987, 2173, 2357, 2537, 2712, 2881,
  553, 668, 798, 942, 1098, 1264, 1438, 1618, 1802, 1988, 2174, 2358, 2538, 2713, 2882, 3043,
  669, 799, 943, 1099, 1265, 1439, 1619, 1803, 1989, 2175, 2359, 2539, 2714, 2883, 3044, 3195,
  800, 944, 1100, 1266, 1440, 1620, 1804, 1990, 2176, 2360, 2540, 2715, 2884, 3045, 3196, 3335,
  945, 1101, 1267, 1441, 1621, 1805, 1991, 2177, 2361, 2541, 2716, 2885, 3046, 3197, 3336, 3461,
  1102, 1268, 1442, 1622, 1806, 1992, 2178, 2362, 2542, 2717, 2886, 3047, 3198, 3337, 3462, 3572,
  1269, 1443, 1623, 1807, 1993, 2179, 2363, 2543, 2718, 2887, 3048, 3199, 3338, 3463, 3573, 3669,
  1444, 1624, 1808, 1994, 2180, 2364, 2544, 2719, 2888, 3049, 3200, 3339, 3464, 3574, 3670, 3753,
  1625, 1809, 1995, 2181, 2365, 2545, 2720, 2889, 3050, 3201, 3340, 3465, 3575, 3671, 3754, 3825,
  1810, 1996, 2182, 2366, 2546, 2721, 2890, 3051, 3202, 3341, 3466, 3576, 3672, 3755, 3826, 3886,
  1997, 2183, 2367, 2547, 2722, 2891, 3052, 3203, 3342, 3467, 3577, 3673, 3756, 3827, 3887, 3937,
  2184, 2368, 2548, 2723, 2892, 3053, 3204, 3343, 3468, 3578, 3674, 3757, 3828, 3888, 3938, 3979,
  2369, 2549, 2724, 2893, 3054, 3205, 3344, 3469, 3579, 3675, 3758, 3829, 3889, 3939, 3980, 4013,
  2550, 2725, 2894, 3055, 3206, 3345, 3470, 3580, 3676, 3759, 3830, 3890, 3940, 3981, 4014, 4040,
  363, 452, 554, 670, 801, 946, 1103, 1270, 1445, 1626, 1811, 1998, 2185, 2370, 2551, 2726,
  453, 555, 671, 802, 947, 1104, 1271, 1446, 1627, 1812, 1999, 2186, 2371, 2552, 2727, 2895,
  556, 672, 803, 948, 1105, 1272, 1447, 1628, 1813, 2000, 2187, 2372, 2553, 2728, 2896, 3056,
  673, 804, 949, 1106, 1273, 1448, 1629, 1814, 2001, 2188, 2373, 2554, 2729, 2897, 3057, 3207,
  805, 950, 1107, 1274, 1449, 1630, 1815, 2002, 2189, 2374, 2555, 2730, 2898, 3058, 3208, 3346,
  951, 1108, 1275, 1450, 1631, 1816, 2003, 2190, 2375, 2556, 2731, 2899, 3059, 3209, 3347, 3471,
  1109, 1276, 1451, 1632, 1817, 2004, 2191, 2376, 2557, 2732, 2900, 3060, 3210, 3348, 3472, 3581,
  1277, 1452, 1633, 1818, 2005, 2192, 2377, 2558, 2733, 2901, 3061, 3211, 3349, 3473, 3582, 3677,
  1453, 1634, 1819, 2006, 2193, 2378, 2559, 2734, 2902, 3062, 3212, 3350, 3474, 3583, 3678, 3760,
  1635, 1820, 2007, 2194, 2379, 2560, 2735, 2903, 3063, 3213, 3351, 3475, 3584, 3679, 3761, 3831,
  1821, 2008, 2195, 2380, 2561, 2736, 2904, 3064, 3214, 3352, 3476, 3585, 3680, 3762, 3832, 3891,
  2009, 2196, 2381, 2562, 2737, 2905, 3065, 3215, 3353, 3477, 3586, 3681, 3763, 3833, 3892, 3941,
  2197, 2382, 2563, 2738, 2906, 3066, 3216, 3354, 3478, 3587, 3682, 3764, 3834, 3893, 3942, 3982,
  2383, 2564, 2739, 2907, 3067, 3217, 3355, 3479, 3588, 3683, 3765, 3835, 3894, 3943, 3983, 4015,
  2565, 2740, 2908, 3068, 3218, 3356, 3480, 3589, 3684, 3766, 3836, 3895, 3944, 3984, 4016, 4041,
  2741, 2909, 3069, 3219, 3357, 3481, 3590, 3685, 3767, 3837, 3896, 3945, 3985, 4017, 4042, 4061,
  454, 557, 674, 806, 952, 1110, 1278, 1454, 1636, 1822, 2010, 2198, 2384, 2566, 2742, 2910,
  558, 675, 807, 953, 1111, 1279, 1455, 1637, 1823, 2011, 2199, 2385, 2567, 2743, 2911, 3070,
  676, 808, 954, 1112, 1280, 1456, 1638, 1824, 2012, 2200, 2386, 2568, 2744, 2912, 3071, 3220,
  809, 955, 1113, 1281, 1457, 1639, 1825, 2013, 2201, 2387, 2569, 2745, 2913, 3072, 3221, 3358,
  956, 1114, 1282, 1458, 1640, 1826, 2014, 2202, 2388, 2570, 2746, 2914, 3073, 3222, 3359, 3482,
  1115, 1283, 1459, 1641, 1827, 2015, 2203, 2389, 2571, 2747, 2915, 3074, 3223, 3360, 3483, 3591,
  1284, 1460, 1642, 1828, 2016, 2204, 2390, 2572, 2748, 2916, 3075, 3224, 3361, 3484, 3592, 3686,
  1461, 1643, 1829, 2017, 2205, 2391, 2573, 2749, 2917, 3076, 3225, 3362, 3485, 3593, 3687, 3768,
  1644, 1830, 2018, 2206, 2392, 2574, 2750, 2918, 3077, 3226, 3363, 3486, 3594, 3688, 3769, 3838,
  1831, 2019, 2207, 2393, 2575, 2751, 2919, 3078, 3227, 3364, 3487, 3595, 3689, 3770, 3839, 3897,
  2020, 2208, 2394, 2576, 2752, 2920, 3079, 3228, 3365, 3488, 3596, 3690, 3771, 3840, 3898, 3946,
  2209, 2395, 2577, 2753, 2921, 3080, 3229, 3366, 3489, 3597, 3691, 3772, 3841, 3899, 3947, 3986,
  2396, 2578, 2754, 2922, 3081, 3230, 3367, 3490, 3598, 3692, 3773, 3842, 3900, 3948, 3987, 4018,
  2579, 2755, 2923, 3082, 3231, 3368, 3491, 3599, 3693, 3774, 3843, 3901, 3949, 3988, 4019, 4043,
  2756, 2924, 3083, 3232, 3369, 3492, 3600, 3694, 3775, 3844, 3902, 3950, 3989, 4020, 4044, 4062,
  2925, 3084, 3233, 3370, 3493, 3601, 3695, 3776, 3845, 3903, 3951, 3990, 4021, 4045, 4063, 4076,
  559, 677, 810, 957, 1116, 1285, 1462, 1645, 1832, 2021, 2210, 2397, 2580, 2757, 2926, 3085,
  678, 811, 958, 1117, 1286, 1463, 1646, 1833, 2022, 2211, 2398, 2581, 2758, 2927, 3086, 3234,
  812, 959, 1118, 1287, 1464, 1647, 1834, 2023, 2212, 2399, 2582, 2759, 2928, 3087, 3235, 3371,
  960, 1119, 1288, 1465, 1648, 1835, 2024, 2213, 2400, 2583, 2760, 2929, 3088, 3236, 3372, 3494,
  1120, 1289, 1466, 1649, 1836, 2025, 2214, 2401, 2584, 2761, 2930, 3089, 3237, 3373, 3495, 3602,
  1290, 1467, 1650, 1837, 2026, 2215, 2402, 2585, 2762, 2931, 3090, 3238, 3374, 3496, 3603, 3696,
  1468, 1651, 1838, 2027, 2216, 2403, 2586, 2763, 2932, 3091, 3239, 3375, 3497, 3604, 3697, 3777,
  1652, 1839, 2028, 2217, 2404, 2587, 2764, 2933, 3092, 3240, 3376, 3498, 3605, 3698, 3778, 3846,
  1840, 2029, 2218, 2405, 2588, 2765, 2934, 3093, 3241, 3377, 3499, 3606, 3699, 3779, 3847, 3904,
  2030, 2219, 2406, 2589, 2766, 2935, 3094, 3242, 3378, 3500, 3607, 3700, 3780, 3848, 3905, 3952,
  2220, 2407, 2590, 2767, 2936, 3095, 3243, 3379, 3501, 3608, 3701, 3781, 3849, 3906, 3953, 3991,
  2408, 2591, 2768, 2937, 3096, 3244, 3380, 3502, 3609, 3702, 3782, 3850, 3907, 3954, 3992, 4022,
  2592, 2769, 2938, 3097, 3245, 3381, 3503, 3610, 3703, 3783, 3851, 3908, 3955, 3993, 4023, 4046,
  2770, 2939, 3098, 3246, 3382, 3504, 3611, 3704, 3784, 3852, 3909, 3956, 3994, 4024, 4047, 4064,
  2940, 3099, 3247, 3383, 3505, 3612, 3705, 3785, 3853, 3910, 3957, 3995, 4025, 4048, 4065, 4077,
  3100, 3248, 3384, 3506, 3613, 3706, 3786, 3854, 3911, 3958, 3996, 4026, 4049, 4066, 4078, 4086,
  679, 813, 961, 1121, 1291, 1469, 1653, 1841, 2031, 2221, 2409, 2593, 2771, 2941, 3101, 3249,
  814, 962, 1122, 1292, 1470, 1654, 1842, 2032, 2222, 2410, 2594, 2772, 2942, 3102, 3250, 3385,
  963, 1123, 1293, 1471, 1655, 1843, 2033, 2223, 2411, 2595, 2773, 2943, 3103, 3251, 3386, 3507,
  1124, 1294, 1472, 1656, 1844, 2034, 2224, 2412, 2596, 2774, 2944, 3104, 3252, 3387, 3508, 3614,
  1295, 1473, 1657, 1845, 2035, 2225, 2413, 2597, 2775, 2945, 3105, 3253, 3388, 3509, 3615, 3707,
  1474, 1658, 1846, 2036, 2226, 2414, 2598, 2776, 2946, 3106, 3254, 3389, 3510, 3616, 3708, 3787,
  1659, 1847, 2037, 2227, 2415, 2599, 2777, 2947, 3107, 3255, 3390, 3511, 3617, 3709, 3788, 3855,
  1848, 2038, 2228, 2416, 2600, 2778, 2948, 3108, 3256, 3391, 3512, 3618, 3710, 3789, 3856, 3912,
  2039, 2229, 2417, 2601, 2779, 2949, 3109, 3257, 3392, 3513, 3619, 3711, 3790, 3857, 3913, 3959,
  2230, 2418, 2602, 2780, 2950, 3110, 3258, 3393, 3514, 3620, 3712, 3791, 3858, 3914, 3960, 3997,
  2419, 2603, 2781, 2951, 3111, 3259, 3394, 3515, 3621, 3713, 3792, 3859, 3915, 3961, 3998, 4027,
  2604, 2782, 2952, 3112, 3260, 3395, 3516, 3622, 3714, 3793, 3860, 3916, 3962, 3999, 4028, 4050,
  2783, 2953, 3113, 3261, 3396, 3517, 3623, 3715, 3794, 3861, 3917, 3963, 4000, 4029, 4051, 4067,
  2954, 3114, 3262, 3397, 3518, 3624, 3716, 3795, 3862, 3918, 3964, 4001, 4030, 4052, 4068, 4079,
  3115, 3263, 3398, 3519, 3625, 3717, 3796, 3863, 3919, 3965, 4002, 4031, 4053, 4069, 4080, 4087,
  3264, 3399, 3520, 3626, 3718, 3797, 3864, 3920, 3966, 4003, 4032, 4054, 4070, 4081, 4088, 4092,
  815, 964, 1125, 1296, 1475, 1660, 1849, 2040, 2231, 2420, 2605, 2784, 2955, 3116, 3265, 3400,
  965, 1126, 1297, 1476, 1661, 1850, 2041, 2232, 2421, 2606, 2785, 2956, 3117, 3266, 3401, 3521,
  1127, 1298, 1477, 1662, 1851, 2042, 2233, 2422, 2607, 2786, 2957, 3118, 3267, 3402, 3522, 3627,
  1299, 1478, 1663, 1852, 2043, 2234, 2423, 2608, 2787, 2958, 3119, 3268, 3403, 3523, 3628, 3719,
  1479, 1664, 1853, 2044, 2235, 2424, 2609, 2788, 2959, 3120, 3269, 3404, 3524, 3629, 3720, 3798,
  1665, 1854, 2045, 2236, 2425, 2610, 2789, 2960, 3121, 3270, 3405, 3525, 3630, 3721, 3799, 3865,
  1855, 2046, 2237, 2426, 2611, 2790, 2961, 3122, 3271, 3406, 3526, 3631, 3722, 3800, 3866, 3921,
  2047, 2238, 2427, 2612, 2791, 2962, 3123, 3272, 3407, 3527, 3632, 3723, 3801, 3867, 3922, 3967,
  2239, 2428, 2613, 2792, 2963, 3124, 3273, 3408, 3528, 3633, 3724, 3802, 3868, 3923, 3968, 4004,
  2429, 2614, 2793, 2964, 3125, 3274, 3409, 3529, 3634, 3725, 3803, 3869, 3924, 3969, 4005, 4033,
  2615, 2794, 2965, 3126, 3275, 3410, 3530, 3635, 3726, 3804, 3870, 3925, 3970, 4006, 4034, 4055,
  2795, 2966, 3127, 3276, 3411, 3531, 3636, 3727, 3805, 3871, 3926, 3971, 4007, 4035, 4056, 4071,
  2967, 3128, 3277, 3412, 3532, 3637, 3728, 3806, 3872, 3927, 3972, 4008, 4036, 4057, 4072, 4082,
  3129, 3278, 3413, 3533, 3638, 3729, 3807, 3873, 3928, 3973, 4009, 4037, 4058, 4073, 4083, 4089,
  3279, 3414, 3534, 3639, 3730, 3808, 3874, 3929, 3974, 4010, 4038, 4059, 4074, 4084, 4090, 4093,
  3415, 3535, 3640, 3731, 3809, 3875, 3930, 3975, 4011, 4039, 4060, 4075, 4085, 4091, 4094, 4095,
};


/* ---- little-endian scalars, cursor, LEB128, zigzag ---- */
static inline void vf_wr_u16(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
static inline void vf_wr_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
static inline uint32_t vf_rd_u16(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static inline uint32_t vf_rd_u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
typedef struct vf_cur {
  const uint8_t *p, *end;
} vf_cur;
static inline bool vf_cur_has(const vf_cur *c, size_t n) { return (size_t)(c->end - c->p) >= n; }
static inline size_t vf_leb_put(uint8_t *p, uint32_t v) {
  size_t n = 0;
  while (v >= 0x80u) {
    p[n++] = (uint8_t)(v | 0x80u);
    v >>= 7;
  }
  p[n++] = (uint8_t)v;
  return n;
}
/* canonical: no trailing zero continuation, <= 5 bytes, fits 32 bits */
static inline bool vf_leb_get(vf_cur *c, uint32_t *out) {
  uint32_t v = 0, sh = 0;
  for (int i = 0; i < 5; i++) {
    if (!vf_cur_has(c, 1)) return false;
    uint32_t b = *c->p++;
    if (i == 4 && (b & 0xf0u)) return false;
    v |= (b & 0x7fu) << sh;
    if (!(b & 0x80u)) {
      if (i > 0 && (b & 0x7fu) == 0) return false;
      *out = v;
      return true;
    }
    sh += 7;
  }
  return false;
}
static inline uint32_t vf_zigzag(int32_t v) { return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31); }
static inline int32_t vf_unzigzag(uint32_t u) { return (int32_t)(u >> 1) ^ -(int32_t)(u & 1u); }

/* ---- LSB-first bypass bit writer / reader (64-bit accumulators) ---- */
typedef struct vf_bitw {
  uint8_t *buf;
  size_t cap, pos;
  uint64_t acc;
  uint32_t nbits;
} vf_bitw;
static inline void vf_bw_init(vf_bitw *w, uint8_t *buf, size_t cap) {
  w->buf = buf;
  w->cap = cap;
  w->pos = 0;
  w->acc = 0;
  w->nbits = 0;
}
__attribute__((always_inline)) static inline bool vf_bw_put(vf_bitw *w, uint32_t v, uint32_t n) { /* n <= 32, v < 2^n */
  w->acc |= (uint64_t)v << w->nbits;
  w->nbits += n;
  while (w->nbits >= 8) {
    if (w->pos >= w->cap) return false;
    w->buf[w->pos++] = (uint8_t)w->acc;
    w->acc >>= 8;
    w->nbits -= 8;
  }
  return true;
}
static inline bool vf_bw_flush(vf_bitw *w, size_t *n_out) {
  if (w->nbits > 0) {
    if (w->pos >= w->cap) return false;
    w->buf[w->pos++] = (uint8_t)w->acc;
    w->acc = 0;
    w->nbits = 0;
  }
  *n_out = w->pos;
  return true;
}
typedef struct vf_bitr {
  const uint8_t *buf;
  size_t n, pos;
  uint64_t acc;
  uint32_t nbits;
} vf_bitr;
static inline void vf_br_init(vf_bitr *r, const uint8_t *buf, size_t n) {
  r->buf = buf;
  r->n = n;
  r->pos = 0;
  r->acc = 0;
  r->nbits = 0;
}
__attribute__((always_inline)) static inline bool vf_br_get(vf_bitr *r, uint32_t n, uint32_t *out) { /* n <= 32 */
  if (r->nbits < n) {
    if (r->pos + 8 <= r->n) { /* bulk refill: whole bytes that fit in the accumulator */
      uint64_t w;
      memcpy(&w, r->buf + r->pos, 8);
      uint32_t take = (63u - r->nbits) >> 3; /* 4..7 bytes when nbits < 32 */
      r->acc |= (w & ((1ull << (take * 8u)) - 1ull)) << r->nbits;
      r->nbits += take * 8u;
      r->pos += take;
    } else {
      while (r->nbits < n) {
        if (r->pos >= r->n) return false;
        r->acc |= (uint64_t)r->buf[r->pos++] << r->nbits;
        r->nbits += 8;
      }
    }
  }
  *out = (uint32_t)(r->acc & ((n == 32u) ? 0xffffffffull : ((1ull << n) - 1ull)));
  r->acc >>= n;
  r->nbits -= n;
  return true;
}
/* exact consumption: all bytes read, remaining padding bits zero */
static inline bool vf_br_finished(const vf_bitr *r) { return r->pos == r->n && r->acc == 0; }

/* ---- HybridUint: tokens 0..3 literal; else tok = 2 + msb(u), low msb bits bypassed ---- */
__attribute__((always_inline)) static inline bool vf_hyb_emit(vf_bitw *w, uint32_t u, uint32_t *tok) {
  if (u < 4u) {
    *tok = u;
    return true;
  }
  uint32_t k = 31u - (uint32_t)__builtin_clz(u);
  *tok = 2u + k;
  return vf_bw_put(w, u & ((1u << k) - 1u), k);
}
__attribute__((always_inline)) static inline bool vf_hyb_read(vf_bitr *r, uint32_t tok, uint32_t *u) { /* tok pre-validated */
  if (tok < 4u) {
    *u = tok;
    return true;
  }
  uint32_t k = tok - 2u, extra;
  if (!vf_br_get(r, k, &extra)) return false;
  *u = (1u << k) | extra;
  return true;
}

/* ---- context models: 0..2 runs+EOB by band, 3..8 levels by band x (run==0), 9 DC ---- */
__attribute__((always_inline)) static inline uint32_t vf_band_of(uint32_t pos) { return pos < 128u ? 0u : (pos < 1024u ? 1u : 2u); }
__attribute__((always_inline)) static inline uint32_t vf_run_ctx(uint32_t pos) { return vf_band_of(pos); }
__attribute__((always_inline)) static inline uint32_t vf_level_ctx(uint32_t pos, uint32_t run) {
  return 3u + vf_band_of(pos) * 2u + (run == 0u ? 1u : 0u);
}

/* ---- entropy coder: tANS (FSE), 10-bit tables, 2 interleaved lanes ---- */
/* tANS (FSE-style) tables. Decoder: state in [0,SCALE); entry gives symbol,
 * bit count and the new-state base. */
typedef struct vf_dentry {
  uint16_t ns;
  uint8_t sym, nb;
} vf_dentry;
typedef struct vf_model {
  uint16_t freq[VF_NTOK];
  uint16_t cum[VF_NTOK + 1];
  vf_dentry dt[VF_PROB_SCALE];
} vf_model;

static inline uint32_t vf_highbit(uint32_t v) { return 31u - (uint32_t)__builtin_clz(v); }
static void vf_model_fill(vf_model *m) {
  m->cum[0] = 0;
  for (uint32_t s = 0; s < VF_NTOK; s++) m->cum[s + 1] = (uint16_t)(m->cum[s] + m->freq[s]);
  /* spread symbols over the table (FSE step), then assign next states */
  const uint32_t size = VF_PROB_SCALE, mask = size - 1u, step = (size >> 1) + (size >> 3) + 3u;
  uint32_t pos = 0;
  for (uint32_t s = 0; s < VF_NTOK; s++)
    for (uint32_t i = 0; i < m->freq[s]; i++) {
      m->dt[pos].sym = (uint8_t)s;
      pos = (pos + step) & mask;
    }
  uint16_t next[VF_NTOK];
  for (uint32_t s = 0; s < VF_NTOK; s++) next[s] = m->freq[s];
  for (uint32_t u = 0; u < size; u++) {
    uint32_t s = m->dt[u].sym;
    uint32_t nx = next[s]++;
    uint32_t nb = VF_PROB_BITS - vf_highbit(nx);
    m->dt[u].nb = (uint8_t)nb;
    m->dt[u].ns = (uint16_t)((nx << nb) - size);
  }
}
/* encoder: normalise counts to sum exactly VF_PROB_SCALE (nonzero counts keep freq >= 1) */
static bool vf_model_build(vf_model *m, const uint32_t counts[VF_NTOK]) {
  uint64_t total = 0;
  for (uint32_t s = 0; s < VF_NTOK; s++) total += counts[s];
  if (total == 0) return false;
  uint32_t assigned = 0;
  for (uint32_t s = 0; s < VF_NTOK; s++) {
    if (counts[s] == 0) {
      m->freq[s] = 0;
      continue;
    }
    uint64_t f = (uint64_t)counts[s] * VF_PROB_SCALE / total;
    if (f == 0) f = 1;
    m->freq[s] = (uint16_t)f;
    assigned += (uint32_t)f;
  }
  if (assigned != VF_PROB_SCALE) {
    uint32_t big = 0;
    for (uint32_t s = 1; s < VF_NTOK; s++)
      if (m->freq[s] > m->freq[big]) big = s;
    int32_t nf = (int32_t)m->freq[big] + ((int32_t)VF_PROB_SCALE - (int32_t)assigned);
    if (nf < 1) return false;
    m->freq[big] = (uint16_t)nf;
  }
  vf_model_fill(m);
  return true;
}
/* decoder: adopt transmitted frequencies verbatim; never renormalises */
static bool vf_model_from_freqs(vf_model *m, const uint32_t freqs[VF_NTOK]) {
  uint32_t sum = 0;
  for (uint32_t s = 0; s < VF_NTOK; s++) {
    if (freqs[s] > VF_PROB_SCALE) return false;
    sum += freqs[s];
    m->freq[s] = (uint16_t)freqs[s];
  }
  if (sum != VF_PROB_SCALE) return false;
  vf_model_fill(m);
  return true;
}
/* compact tables: per model u32 LE presence bitmap + LEB128 freq per set bit, ascending */
static size_t vf_tables_write(const vf_model *m, uint8_t *out, uint32_t nm) {
  size_t n = 0;
  for (uint32_t i = 0; i < nm; i++) {
    uint32_t bm = 0;
    for (uint32_t s = 0; s < VF_NTOK; s++)
      if (m[i].freq[s]) bm |= 1u << s;
    vf_wr_u32(out + n, bm);
    n += 4;
    for (uint32_t s = 0; s < VF_NTOK; s++)
      if (m[i].freq[s]) n += vf_leb_put(out + n, m[i].freq[s]);
  }
  return n;
}
static bool vf_tables_read(vf_cur *c, vf_model *m, uint32_t nm) {
  for (uint32_t i = 0; i < nm; i++) {
    if (!vf_cur_has(c, 4)) return false;
    uint32_t bm = vf_rd_u32(c->p);
    c->p += 4;
    if (bm == 0) return false;
    uint32_t f[VF_NTOK] = {0}, sum = 0;
    for (uint32_t s = 0; s < VF_NTOK; s++) {
      if (!(bm >> s & 1u)) continue;
      uint32_t v;
      if (!vf_leb_get(c, &v) || v == 0 || v > VF_PROB_SCALE) return false;
      sum += v;
      if (sum > VF_PROB_SCALE) return false;
      f[s] = v;
    }
    if (!vf_model_from_freqs(&m[i], f)) return false;
  }
  return true;
}
/* tANS encoder tables (FSE_buildCTable): per model a state table and per
 * symbol (deltaNbBits, deltaFindState). Encoding runs backwards over the
 * symbols; the emitted bit fields are buffered and written in decode order
 * so the decoder reads a plain forward LSB-first bit stream. */
typedef struct vf_esym {
  uint32_t delta_nb;
  int32_t delta_fs;
} vf_esym;
typedef struct vf_etab {
  uint16_t state[VF_PROB_SCALE];
  vf_esym sym[VF_NTOK];
} vf_etab;
static void vf_etabs_init(const vf_model *m, vf_etab *et, uint32_t nm) {
  const uint32_t size = VF_PROB_SCALE, mask = size - 1u, step = (size >> 1) + (size >> 3) + 3u;
  for (uint32_t c = 0; c < nm; c++) {
    uint8_t tsym[VF_PROB_SCALE];
    uint32_t pos = 0;
    for (uint32_t s = 0; s < VF_NTOK; s++)
      for (uint32_t i = 0; i < m[c].freq[s]; i++) {
        tsym[pos] = (uint8_t)s;
        pos = (pos + step) & mask;
      }
    uint32_t cum[VF_NTOK + 1];
    for (uint32_t s = 0; s <= VF_NTOK; s++) cum[s] = m[c].cum[s];
    for (uint32_t u = 0; u < size; u++) et[c].state[cum[tsym[u]]++] = (uint16_t)(size + u);
    uint32_t total = 0;
    for (uint32_t s = 0; s < VF_NTOK; s++) {
      uint32_t f = m[c].freq[s];
      if (f == 0) {
        et[c].sym[s].delta_nb = 0xffffffffu; /* marker: unusable */
        et[c].sym[s].delta_fs = 0;
        continue;
      }
      uint32_t maxbits = f == 1 ? VF_PROB_BITS : VF_PROB_BITS - vf_highbit(f - 1u);
      et[c].sym[s].delta_nb = (maxbits << 16) - (f << maxbits);
      et[c].sym[s].delta_fs = (int32_t)total - (int32_t)(f == 1 ? 1u : f);
      total += f;
    }
  }
}
/* fields: value | nbits<<27, built in reverse symbol order */
static size_t vf_tans_encode2(const vf_etab et[VF_NMODELS], const uint16_t *syms, size_t n,
                              uint32_t *fields, uint8_t *out, size_t out_cap) {
  uint32_t x[2] = {VF_PROB_SCALE, VF_PROB_SCALE};
  size_t nf = 0;
  for (size_t i = n; i-- > 0;) {
    const vf_esym *e = &et[syms[i] >> 8].sym[syms[i] & 0xffu];
    if (e->delta_nb == 0xffffffffu) return 0;
    uint32_t l = (uint32_t)i & 1u;
    uint32_t nb = (x[l] + e->delta_nb) >> 16;
    fields[nf++] = (x[l] & ((1u << nb) - 1u)) | nb << 27;
    x[l] = et[syms[i] >> 8].state[(x[l] >> nb) + (uint32_t)e->delta_fs];
  }
  /* initial decoder states (lane 1 pushed first so lane 0 is read first) */
  fields[nf++] = (x[1] - VF_PROB_SCALE) | VF_PROB_BITS << 27;
  fields[nf++] = (x[0] - VF_PROB_SCALE) | VF_PROB_BITS << 27;
  vf_bitw w;
  vf_bw_init(&w, out, out_cap);
  for (size_t i = nf; i-- > 0;)
    if (!vf_bw_put(&w, fields[i] & 0x7ffffffu, fields[i] >> 27)) return 0;
  size_t used;
  if (!vf_bw_flush(&w, &used)) return 0;
  return used;
}
typedef struct vf_rdec {
  uint32_t x[2];
  uint32_t k;
  vf_bitr br;
} vf_rdec;
static inline bool vf_rdec_init(vf_rdec *d, const uint8_t *in, size_t in_n) {
  vf_br_init(&d->br, in, in_n);
  if (!vf_br_get(&d->br, VF_PROB_BITS, &d->x[0]) || !vf_br_get(&d->br, VF_PROB_BITS, &d->x[1]))
    return false;
  d->k = 0;
  return true;
}
__attribute__((always_inline)) static inline int vf_rdec_get(vf_rdec *d, const vf_model *m) {
  uint32_t k = d->k;
  const vf_dentry *e = &m->dt[d->x[k]];
  uint32_t bits;
  if (!vf_br_get(&d->br, e->nb, &bits)) return -1;
  d->x[k] = e->ns + bits;
  d->k = k ^ 1u;
  return (int)e->sym;
}
/* exact consumption: all bytes read, zero padding, both lanes back at state 0 */
static inline bool vf_rdec_finished(const vf_rdec *d) {
  return vf_br_finished(&d->br) && d->x[0] == 0 && d->x[1] == 0;
}

/* ---- 16-point orthonormal DCT-II, separable over 16^3  ----
 * Factored (Lee's even/odd recursion, 32 multiplies per 16-point transform;
 * the inverse is the exact transpose network). One straight-line kernel on
 * eight lanes (__m256): the y and z passes feed it eight contiguous x lanes at
 * a time; the x pass transposes eight lines through 8x8 tiles so that the same
 * kernel serves all three axes. */
/* generated: 16-point DCT-II, Lee recursion (forward) and its transpose (inverse), UNNORMALISED:
 * the orthonormal scale s0 = 1/4, sk = sqrt(2)/4 per axis is folded into the quantiser tables. */
/* The kernel bodies are macros over the lane type VT so the same straight-line
 * network serves the AVX2 path (VT = __m256, eight lanes) and the C path
 * (VT = float). */
#define VF_DCT16_FWD_BODY(VT) \
  VT t1 = in[0] + in[15]; \
  VT t2 = in[1] + in[14]; \
  VT t3 = in[2] + in[13]; \
  VT t4 = in[3] + in[12]; \
  VT t5 = in[4] + in[11]; \
  VT t6 = in[5] + in[10]; \
  VT t7 = in[6] + in[9]; \
  VT t8 = in[7] + in[8]; \
  VT t9 = (in[0] - in[15]) * 5.024192862e-01f; \
  VT t10 = (in[1] - in[14]) * 5.224986149e-01f; \
  VT t11 = (in[2] - in[13]) * 5.669440348e-01f; \
  VT t12 = (in[3] - in[12]) * 6.468217834e-01f; \
  VT t13 = (in[4] - in[11]) * 7.881546235e-01f; \
  VT t14 = (in[5] - in[10]) * 1.060677686e+00f; \
  VT t15 = (in[6] - in[9]) * 1.722447098e+00f; \
  VT t16 = (in[7] - in[8]) * 5.101148619e+00f; \
  VT t17 = t1 + t8; \
  VT t18 = t2 + t7; \
  VT t19 = t3 + t6; \
  VT t20 = t4 + t5; \
  VT t21 = (t1 - t8) * 5.097955791e-01f; \
  VT t22 = (t2 - t7) * 6.013448869e-01f; \
  VT t23 = (t3 - t6) * 8.999762231e-01f; \
  VT t24 = (t4 - t5) * 2.562915448e+00f; \
  VT t25 = t17 + t20; \
  VT t26 = t18 + t19; \
  VT t27 = (t17 - t20) * 5.411961001e-01f; \
  VT t28 = (t18 - t19) * 1.306562965e+00f; \
  VT t29 = t25 + t26; \
  VT t30 = (t25 - t26) * 7.071067812e-01f; \
  VT t31 = t27 + t28; \
  VT t32 = (t27 - t28) * 7.071067812e-01f; \
  VT t33 = t31 + t32; \
  VT t34 = t21 + t24; \
  VT t35 = t22 + t23; \
  VT t36 = (t21 - t24) * 5.411961001e-01f; \
  VT t37 = (t22 - t23) * 1.306562965e+00f; \
  VT t38 = t34 + t35; \
  VT t39 = (t34 - t35) * 7.071067812e-01f; \
  VT t40 = t36 + t37; \
  VT t41 = (t36 - t37) * 7.071067812e-01f; \
  VT t42 = t40 + t41; \
  VT t43 = t38 + t42; \
  VT t44 = t42 + t39; \
  VT t45 = t39 + t41; \
  VT t46 = t9 + t16; \
  VT t47 = t10 + t15; \
  VT t48 = t11 + t14; \
  VT t49 = t12 + t13; \
  VT t50 = (t9 - t16) * 5.097955791e-01f; \
  VT t51 = (t10 - t15) * 6.013448869e-01f; \
  VT t52 = (t11 - t14) * 8.999762231e-01f; \
  VT t53 = (t12 - t13) * 2.562915448e+00f; \
  VT t54 = t46 + t49; \
  VT t55 = t47 + t48; \
  VT t56 = (t46 - t49) * 5.411961001e-01f; \
  VT t57 = (t47 - t48) * 1.306562965e+00f; \
  VT t58 = t54 + t55; \
  VT t59 = (t54 - t55) * 7.071067812e-01f; \
  VT t60 = t56 + t57; \
  VT t61 = (t56 - t57) * 7.071067812e-01f; \
  VT t62 = t60 + t61; \
  VT t63 = t50 + t53; \
  VT t64 = t51 + t52; \
  VT t65 = (t50 - t53) * 5.411961001e-01f; \
  VT t66 = (t51 - t52) * 1.306562965e+00f; \
  VT t67 = t63 + t64; \
  VT t68 = (t63 - t64) * 7.071067812e-01f; \
  VT t69 = t65 + t66; \
  VT t70 = (t65 - t66) * 7.071067812e-01f; \
  VT t71 = t69 + t70; \
  VT t72 = t67 + t71; \
  VT t73 = t71 + t68; \
  VT t74 = t68 + t70; \
  VT t75 = t58 + t72; \
  VT t76 = t72 + t62; \
  VT t77 = t62 + t73; \
  VT t78 = t73 + t59; \
  VT t79 = t59 + t74; \
  VT t80 = t74 + t61; \
  VT t81 = t61 + t70; \
  out[0] = t29; \
  out[1] = t75; \
  out[2] = t43; \
  out[3] = t76; \
  out[4] = t33; \
  out[5] = t77; \
  out[6] = t44; \
  out[7] = t78; \
  out[8] = t30; \
  out[9] = t79; \
  out[10] = t45; \
  out[11] = t80; \
  out[12] = t32; \
  out[13] = t81; \
  out[14] = t41; \
  out[15] = t70;
#define VF_DCT16_INV_BODY(VT) \
  VT t1 = in[0]; \
  VT t2 = in[1]; \
  VT t3 = in[2]; \
  VT t4 = in[3]; \
  VT t5 = in[4]; \
  VT t6 = in[5]; \
  VT t7 = in[6]; \
  VT t8 = in[7]; \
  VT t9 = in[8]; \
  VT t10 = in[9]; \
  VT t11 = in[10]; \
  VT t12 = in[11]; \
  VT t13 = in[12]; \
  VT t14 = in[13]; \
  VT t15 = in[14]; \
  VT t16 = in[15]; \
  VT t17 = t4 + t2; \
  VT t18 = t6 + t4; \
  VT t19 = t8 + t6; \
  VT t20 = t10 + t8; \
  VT t21 = t12 + t10; \
  VT t22 = t14 + t12; \
  VT t23 = t16 + t14; \
  VT t24 = t7 + t3; \
  VT t25 = t11 + t7; \
  VT t26 = t15 + t11; \
  VT t27 = t13 + t5; \
  VT t28 = t9 * 7.071067812e-01f; \
  VT t29 = t1 + t28; \
  VT t30 = t1 - t28; \
  VT t31 = t27 * 7.071067812e-01f; \
  VT t32 = t5 + t31; \
  VT t33 = t5 - t31; \
  VT t34 = t32 * 5.411961001e-01f; \
  VT t35 = t29 + t34; \
  VT t36 = t29 - t34; \
  VT t37 = t33 * 1.306562965e+00f; \
  VT t38 = t30 + t37; \
  VT t39 = t30 - t37; \
  VT t40 = t26 + t24; \
  VT t41 = t25 * 7.071067812e-01f; \
  VT t42 = t3 + t41; \
  VT t43 = t3 - t41; \
  VT t44 = t40 * 7.071067812e-01f; \
  VT t45 = t24 + t44; \
  VT t46 = t24 - t44; \
  VT t47 = t45 * 5.411961001e-01f; \
  VT t48 = t42 + t47; \
  VT t49 = t42 - t47; \
  VT t50 = t46 * 1.306562965e+00f; \
  VT t51 = t43 + t50; \
  VT t52 = t43 - t50; \
  VT t53 = t48 * 5.097955791e-01f; \
  VT t54 = t35 + t53; \
  VT t55 = t35 - t53; \
  VT t56 = t51 * 6.013448869e-01f; \
  VT t57 = t38 + t56; \
  VT t58 = t38 - t56; \
  VT t59 = t52 * 8.999762231e-01f; \
  VT t60 = t39 + t59; \
  VT t61 = t39 - t59; \
  VT t62 = t49 * 2.562915448e+00f; \
  VT t63 = t36 + t62; \
  VT t64 = t36 - t62; \
  VT t65 = t19 + t17; \
  VT t66 = t21 + t19; \
  VT t67 = t23 + t21; \
  VT t68 = t22 + t18; \
  VT t69 = t20 * 7.071067812e-01f; \
  VT t70 = t2 + t69; \
  VT t71 = t2 - t69; \
  VT t72 = t68 * 7.071067812e-01f; \
  VT t73 = t18 + t72; \
  VT t74 = t18 - t72; \
  VT t75 = t73 * 5.411961001e-01f; \
  VT t76 = t70 + t75; \
  VT t77 = t70 - t75; \
  VT t78 = t74 * 1.306562965e+00f; \
  VT t79 = t71 + t78; \
  VT t80 = t71 - t78; \
  VT t81 = t67 + t65; \
  VT t82 = t66 * 7.071067812e-01f; \
  VT t83 = t17 + t82; \
  VT t84 = t17 - t82; \
  VT t85 = t81 * 7.071067812e-01f; \
  VT t86 = t65 + t85; \
  VT t87 = t65 - t85; \
  VT t88 = t86 * 5.411961001e-01f; \
  VT t89 = t83 + t88; \
  VT t90 = t83 - t88; \
  VT t91 = t87 * 1.306562965e+00f; \
  VT t92 = t84 + t91; \
  VT t93 = t84 - t91; \
  VT t94 = t89 * 5.097955791e-01f; \
  VT t95 = t76 + t94; \
  VT t96 = t76 - t94; \
  VT t97 = t92 * 6.013448869e-01f; \
  VT t98 = t79 + t97; \
  VT t99 = t79 - t97; \
  VT t100 = t93 * 8.999762231e-01f; \
  VT t101 = t80 + t100; \
  VT t102 = t80 - t100; \
  VT t103 = t90 * 2.562915448e+00f; \
  VT t104 = t77 + t103; \
  VT t105 = t77 - t103; \
  VT t106 = t95 * 5.024192862e-01f; \
  VT t107 = t54 + t106; \
  VT t108 = t54 - t106; \
  VT t109 = t98 * 5.224986149e-01f; \
  VT t110 = t57 + t109; \
  VT t111 = t57 - t109; \
  VT t112 = t101 * 5.669440348e-01f; \
  VT t113 = t60 + t112; \
  VT t114 = t60 - t112; \
  VT t115 = t104 * 6.468217834e-01f; \
  VT t116 = t63 + t115; \
  VT t117 = t63 - t115; \
  VT t118 = t105 * 7.881546235e-01f; \
  VT t119 = t64 + t118; \
  VT t120 = t64 - t118; \
  VT t121 = t102 * 1.060677686e+00f; \
  VT t122 = t61 + t121; \
  VT t123 = t61 - t121; \
  VT t124 = t99 * 1.722447098e+00f; \
  VT t125 = t58 + t124; \
  VT t126 = t58 - t124; \
  VT t127 = t96 * 5.101148619e+00f; \
  VT t128 = t55 + t127; \
  VT t129 = t55 - t127; \
  out[0] = t107; \
  out[1] = t110; \
  out[2] = t113; \
  out[3] = t116; \
  out[4] = t119; \
  out[5] = t122; \
  out[6] = t125; \
  out[7] = t128; \
  out[8] = t129; \
  out[9] = t126; \
  out[10] = t123; \
  out[11] = t120; \
  out[12] = t117; \
  out[13] = t114; \
  out[14] = t111; \
  out[15] = t108;

/* C kernels: one 16-point transform on scalars */
static inline void vf_dct16_c_fwd(const float in[16], float out[16]) { VF_DCT16_FWD_BODY(float) }
static inline void vf_dct16_c_inv(const float in[16], float out[16]) { VF_DCT16_INV_BODY(float) }
#if VF_HAVE_AVX2
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_dct16_v8_fwd(const __m256 in[16], __m256 out[16]) { VF_DCT16_FWD_BODY(__m256) }
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_dct16_v8_inv(const __m256 in[16], __m256 out[16]) { VF_DCT16_INV_BODY(__m256) }
#endif
#if VF_HAVE_AVX2

/* in-register 8x8 transpose of v[0..7] */
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_transpose8(__m256 v[8]) {
  __m256 a0 = _mm256_unpacklo_ps(v[0], v[1]), a1 = _mm256_unpackhi_ps(v[0], v[1]);
  __m256 a2 = _mm256_unpacklo_ps(v[2], v[3]), a3 = _mm256_unpackhi_ps(v[2], v[3]);
  __m256 a4 = _mm256_unpacklo_ps(v[4], v[5]), a5 = _mm256_unpackhi_ps(v[4], v[5]);
  __m256 a6 = _mm256_unpacklo_ps(v[6], v[7]), a7 = _mm256_unpackhi_ps(v[6], v[7]);
  __m256 b0 = _mm256_shuffle_ps(a0, a2, 0x44), b1 = _mm256_shuffle_ps(a0, a2, 0xEE);
  __m256 b2 = _mm256_shuffle_ps(a1, a3, 0x44), b3 = _mm256_shuffle_ps(a1, a3, 0xEE);
  __m256 b4 = _mm256_shuffle_ps(a4, a6, 0x44), b5 = _mm256_shuffle_ps(a4, a6, 0xEE);
  __m256 b6 = _mm256_shuffle_ps(a5, a7, 0x44), b7 = _mm256_shuffle_ps(a5, a7, 0xEE);
  v[0] = _mm256_permute2f128_ps(b0, b4, 0x20);
  v[1] = _mm256_permute2f128_ps(b1, b5, 0x20);
  v[2] = _mm256_permute2f128_ps(b2, b6, 0x20);
  v[3] = _mm256_permute2f128_ps(b3, b7, 0x20);
  v[4] = _mm256_permute2f128_ps(b0, b4, 0x31);
  v[5] = _mm256_permute2f128_ps(b1, b5, 0x31);
  v[6] = _mm256_permute2f128_ps(b2, b6, 0x31);
  v[7] = _mm256_permute2f128_ps(b3, b7, 0x31);
}
/* x pass: eight consecutive lines (stride 16 floats) per call, transposed in
 * and out through two 8x8 tiles */
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_dct16_x8(float *l, bool inverse) {
  __m256 in[16], out[16];
  for (int y = 0; y < 8; y++) {
    in[y] = _mm256_loadu_ps(l + y * 16);
    in[8 + y] = _mm256_loadu_ps(l + y * 16 + 8);
  }
  vf_transpose8(in);
  vf_transpose8(in + 8);
  if (inverse)
    vf_dct16_v8_inv(in, out);
  else
    vf_dct16_v8_fwd(in, out);
  vf_transpose8(out);
  vf_transpose8(out + 8);
  for (int y = 0; y < 8; y++) {
    _mm256_storeu_ps(l + y * 16, out[y]);
    _mm256_storeu_ps(l + y * 16 + 8, out[8 + y]);
  }
}
/* y or z pass: eight contiguous x lanes, elements `es` floats apart */
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_dct16_l8(float *p, ptrdiff_t es, bool inverse) {
  __m256 in[16], out[16];
  for (int t = 0; t < 16; t++) in[t] = _mm256_loadu_ps(p + t * es);
  if (inverse)
    vf_dct16_v8_inv(in, out);
  else
    vf_dct16_v8_fwd(in, out);
  for (int t = 0; t < 16; t++) _mm256_storeu_ps(p + t * es, out[t]);
}
__attribute__((noinline)) VF_TARGET_AVX2 static void vf_dct16_fwd_avx2(float blk[VF_BLKV]) {
  VF_FASTMATH_BEGIN
  for (int z = 0; z < 16; z++) {
    float *pl = blk + z * 256;
    vf_dct16_x8(pl, false);
    vf_dct16_x8(pl + 128, false);
    vf_dct16_l8(pl, 16, false);
    vf_dct16_l8(pl + 8, 16, false);
  }
  for (int y = 0; y < 16; y++) {
    vf_dct16_l8(blk + y * 16, 256, false);
    vf_dct16_l8(blk + y * 16 + 8, 256, false);
  }
}
/* z pass of the inverse, out of place: coefficient planes at stride 256 in
 * `blk` (zero planes are not loaded) to the same layout in `out` */
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_dct16_l8z(const float *p, uint32_t zmask, float *o) {
  __m256 in[16], out[16];
  for (int t = 0; t < 16; t++)
    in[t] = zmask >> t & 1u ? _mm256_loadu_ps(p + t * 256) : _mm256_setzero_ps();
  vf_dct16_v8_inv(in, out);
  for (int t = 0; t < 16; t++) _mm256_storeu_ps(o + t * 256, out[t]);
}
/* Inverse: zmask bit z set iff coefficient plane z has any nonzero, hmask bit
 * z set iff plane z has one at y >= 8. Planes outside zmask stay zero through
 * the x and y passes (run in place on `blk`, which must be zero elsewhere), so
 * both skip them, and the x pass skips a plane's upper eight lines when they
 * are empty; the z pass reads every plane and writes the voxels to `out`, so
 * `blk` keeps only the zmask planes dirty. */
__attribute__((noinline)) VF_TARGET_AVX2 static void vf_dct16_inv_avx2(float blk[VF_BLKV], uint32_t zmask, uint32_t hmask,
                                                                   float out[VF_BLKV]) {
  VF_FASTMATH_BEGIN
  for (int z = 0; z < 16; z++) {
    if (!(zmask >> z & 1u)) continue;
    float *pl = blk + z * 256;
    vf_dct16_x8(pl, true);
    if (hmask >> z & 1u) vf_dct16_x8(pl + 128, true);
    vf_dct16_l8(pl, 16, true);
    vf_dct16_l8(pl + 8, 16, true);
  }
  for (int y = 0; y < 16; y++) {
    vf_dct16_l8z(blk + y * 16, zmask, out + y * 16);
    vf_dct16_l8z(blk + y * 16 + 8, zmask, out + y * 16 + 8);
  }
}
#endif /* VF_HAVE_AVX2 */

/* ---- C transform passes (same structure and skip logic as the AVX2 ones) ---- */
__attribute__((noinline)) static void vf_dct16_fwd_c(float blk[VF_BLKV]) {
  VF_FASTMATH_BEGIN
  float in[16], out[16];
  for (int z = 0; z < 16; z++) {
    float *pl = blk + z * 256;
    for (int y = 0; y < 16; y++) { /* x lines */
      vf_dct16_c_fwd(pl + y * 16, out);
      memcpy(pl + y * 16, out, sizeof out);
    }
    for (int x = 0; x < 16; x++) { /* y lines */
      for (int t = 0; t < 16; t++) in[t] = pl[t * 16 + x];
      vf_dct16_c_fwd(in, out);
      for (int t = 0; t < 16; t++) pl[t * 16 + x] = out[t];
    }
  }
  for (int i = 0; i < 256; i++) { /* z lines */
    for (int t = 0; t < 16; t++) in[t] = blk[t * 256 + i];
    vf_dct16_c_fwd(in, out);
    for (int t = 0; t < 16; t++) blk[t * 256 + i] = out[t];
  }
}
__attribute__((noinline)) static void vf_dct16_inv_c(float blk[VF_BLKV], uint32_t zmask, uint32_t hmask,
                                                     float out[VF_BLKV]) {
  VF_FASTMATH_BEGIN
  float in[16], o[16];
  for (int z = 0; z < 16; z++) {
    if (!(zmask >> z & 1u)) continue;
    float *pl = blk + z * 256;
    const int ny = hmask >> z & 1u ? 16 : 8; /* upper eight x lines are empty */
    for (int y = 0; y < ny; y++) {
      vf_dct16_c_inv(pl + y * 16, o);
      memcpy(pl + y * 16, o, sizeof o);
    }
    for (int x = 0; x < 16; x++) {
      for (int t = 0; t < 16; t++) in[t] = pl[t * 16 + x];
      vf_dct16_c_inv(in, o);
      for (int t = 0; t < 16; t++) pl[t * 16 + x] = o[t];
    }
  }
  for (int i = 0; i < 256; i++) {
    for (int t = 0; t < 16; t++) in[t] = zmask >> t & 1u ? blk[t * 256 + i] : 0.0f;
    vf_dct16_c_inv(in, o);
    for (int t = 0; t < 16; t++) out[t * 256 + i] = o[t];
  }
}
/* ---- kernel selection ---- */
static inline bool vf_use_avx2(void) {
#if !VF_HAVE_AVX2
  return false;
#elif VF_AVX2_STATIC
  return true;
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
static inline const char *volcomp_kernels(void) { return vf_use_avx2() ? "avx2" : "c"; }
static inline void vf_dct16_fwd(float blk[VF_BLKV]) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) { vf_dct16_fwd_avx2(blk); return; }
#endif
  vf_dct16_fwd_c(blk);
}
static inline void vf_dct16_inv(float blk[VF_BLKV], uint32_t zmask, uint32_t hmask, float out[VF_BLKV]) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) { vf_dct16_inv_avx2(blk, zmask, hmask, out); return; }
#endif
  vf_dct16_inv_c(blk, zmask, hmask, out);
}
/* re-zero the coefficient planes a block left dirty */
static inline void vf_clear_planes(float blk[VF_BLKV], uint32_t zmask) {
  for (; zmask; zmask &= zmask - 1u) memset(blk + (ptrdiff_t)__builtin_ctz(zmask) * 256, 0, 1024);
}
/* ---- quantiser: step(r) = q (1+r)^0.65 for r = z+y+x >= 1, step(0) = q/8 ---- */
__attribute__((always_inline)) static inline uint32_t vf_radius(uint32_t c) { return (c & 15u) + ((c >> 4) & 15u) + ((c >> 8) & 15u); }
/* orthonormal DCT scale folded out of the transform kernels: s(0) = 1/4,
 * s(k>0) = sqrt(2)/4 per axis, so the 3-D scale depends only on how many of
 * (x,y,z) are zero */
static inline float vf_dct_scale(uint32_t c) {
  static const float S[4] = {0.0441941738f, 0.03125f, 0.0220970869f, 0.015625f}; /* by zero count */
  uint32_t z0 = (uint32_t)((c & 15u) == 0) + (uint32_t)(((c >> 4) & 15u) == 0) + (uint32_t)(((c >> 8) & 15u) == 0);
  return S[z0];
}
static void vf_steps(float q, float step[VF_NRADII]) {
  step[0] = q * VF_DC_FINE;
  for (uint32_t r = 1; r < VF_NRADII; r++) step[r] = q * powf(1.0f + (float)r, VF_HF_EXP);
}

/* ---- the bit-exact reconstruction of a surface chunk's base (spec §13.2) ----
 * A surface chunk refines its DCT base with context-coded bits whose contexts
 * are read from the base's own reconstruction, so encoder and decoder must
 * compute that reconstruction to the last bit, on every build. The default
 * kernels are not held to that (AVX2+FMA and C agree to +-1 LSB), so the base
 * is reconstructed here in plain IEEE single precision: no contraction into
 * FMA, no reassociation, the operation order of VF_DCT16_INV_BODY, and the
 * step law from a table instead of powf (libm results differ between
 * platforms). VF_HF_POW[r] = powf(1 + r, 0.65f) as glibc computes it; the
 * steps are q * VF_HF_POW[r], one IEEE multiplication. The AVX2 version runs
 * the same scalar operation sequence on eight lines at once, so the two kernel
 * sets are bit-identical (tests/test_surfchunk.c checks it on every build). */
static const float VF_HF_POW[VF_NRADII] = {
    0x1p+0f,        0x1.91b502p+0f, 0x1.056b84p+1f, 0x1.3b2c48p+1f, 0x1.6c5e42p+1f, 0x1.9a364p+1f,
    0x1.c571acp+1f, 0x1.ee8f34p+1f, 0x1.0af468p+2f, 0x1.1de0a2p+2f, 0x1.3025eep+2f, 0x1.41d882p+2f,
    0x1.5308aep+2f, 0x1.63c3d4p+2f, 0x1.74152p+2f,  0x1.8405fep+2f, 0x1.939e8p+2f,  0x1.a2e596p+2f,
    0x1.b1e152p+2f, 0x1.c097p+2f,   0x1.cf0b54p+2f, 0x1.dd4276p+2f, 0x1.eb4022p+2f, 0x1.f907b4p+2f,
    0x1.034e16p+3f, 0x1.0a0028p+3f, 0x1.109b4cp+3f, 0x1.1720a6p+3f, 0x1.1d9142p+3f, 0x1.23ee16p+3f,
    0x1.2a3804p+3f, 0x1.306fdep+3f, 0x1.369668p+3f, 0x1.3cac54p+3f, 0x1.42b24ap+3f, 0x1.48a8e8p+3f,
    0x1.4e90cp+3f,  0x1.546a5cp+3f, 0x1.5a363cp+3f, 0x1.5ff4dap+3f, 0x1.65a6a8p+3f, 0x1.6b4c12p+3f,
    0x1.70e57cp+3f, 0x1.767346p+3f, 0x1.7bf5cep+3f, 0x1.816d66p+3f};
static void vf_steps_strict(float q, float step[VF_NRADII]) {
  step[0] = q * VF_DC_FINE;
  for (uint32_t r = 1; r < VF_NRADII; r++) step[r] = q * VF_HF_POW[r];
}
/* the surface chunk's step law (§13.1): DC as mode 0, every AC coefficient at q.
 * A smooth probability field has no perceptual reason to quantise high
 * frequencies harder, and at equal size this halves the 99th-percentile error
 * near the threshold (docs/surface_mode.md). Exact on every build. */
static void vf_steps_flat(float q, float step[VF_NRADII]) {
  step[0] = q * VF_DC_FINE;
  for (uint32_t r = 1; r < VF_NRADII; r++) step[r] = q;
}
#if defined(__clang__)
#define VF_STRICT_BEGIN _Pragma("clang fp contract(off) reassociate(off)")
#define VF_STRICT_FN
#elif defined(__GNUC__)
#define VF_STRICT_BEGIN
#define VF_STRICT_FN __attribute__((optimize("fp-contract=off", "no-fast-math")))
#else
#define VF_STRICT_BEGIN
#define VF_STRICT_FN
#endif
/* one 16-point inverse transform, `in` -> `out`, strict */
#define VF_IDCT16_STRICT(VT, IN, OUT) \
  do {                                \
    const VT *in = (IN);              \
    VT *out = (OUT);                  \
    VF_DCT16_INV_BODY(VT)             \
  } while (0)
/* Same passes, skips and order as vf_dct16_inv_c (planes outside zmask, and a
 * plane's upper eight x lines outside hmask, are zero and stay zero). */
VF_STRICT_FN __attribute__((noinline)) static void vf_dct16_inv_strict_c(float blk[VF_BLKV], uint32_t zmask,
                                                                        uint32_t hmask, float dst[VF_BLKV]) {
  VF_STRICT_BEGIN
  float li[16], lo[16];
  for (int z = 0; z < 16; z++) {
    if (!(zmask >> z & 1u)) continue;
    float *pl = blk + z * 256;
    const int ny = hmask >> z & 1u ? 16 : 8;
    for (int y = 0; y < ny; y++) {
      VF_IDCT16_STRICT(float, pl + y * 16, lo);
      memcpy(pl + y * 16, lo, sizeof lo);
    }
    for (int x = 0; x < 16; x++) {
      for (int t = 0; t < 16; t++) li[t] = pl[t * 16 + x];
      VF_IDCT16_STRICT(float, li, lo);
      for (int t = 0; t < 16; t++) pl[t * 16 + x] = lo[t];
    }
  }
  for (int i = 0; i < 256; i++) {
    for (int t = 0; t < 16; t++) li[t] = zmask >> t & 1u ? blk[t * 256 + i] : 0.0f;
    VF_IDCT16_STRICT(float, li, lo);
    for (int t = 0; t < 16; t++) dst[t * 256 + i] = lo[t];
  }
}
#if VF_HAVE_AVX2
VF_STRICT_FN __attribute__((noinline)) VF_TARGET_AVX2 static void
vf_dct16_inv_strict_avx2(float blk[VF_BLKV], uint32_t zmask, uint32_t hmask, float dst[VF_BLKV]) {
  VF_STRICT_BEGIN
  __m256 li[16], lo[16];
  for (int z = 0; z < 16; z++) {
    if (!(zmask >> z & 1u)) continue;
    float *pl = blk + z * 256;
    const int nh = hmask >> z & 1u ? 2 : 1;
    for (int h = 0; h < nh; h++) { /* x lines, eight at a time through two 8x8 transposes */
      float *l = pl + h * 128;
      for (int y = 0; y < 8; y++) {
        li[y] = _mm256_loadu_ps(l + y * 16);
        li[8 + y] = _mm256_loadu_ps(l + y * 16 + 8);
      }
      vf_transpose8(li);
      vf_transpose8(li + 8);
      VF_IDCT16_STRICT(__m256, li, lo);
      vf_transpose8(lo);
      vf_transpose8(lo + 8);
      for (int y = 0; y < 8; y++) {
        _mm256_storeu_ps(l + y * 16, lo[y]);
        _mm256_storeu_ps(l + y * 16 + 8, lo[8 + y]);
      }
    }
    for (int h = 0; h < 2; h++) { /* y lines: eight x lanes */
      for (int t = 0; t < 16; t++) li[t] = _mm256_loadu_ps(pl + t * 16 + h * 8);
      VF_IDCT16_STRICT(__m256, li, lo);
      for (int t = 0; t < 16; t++) _mm256_storeu_ps(pl + t * 16 + h * 8, lo[t]);
    }
  }
  for (int i = 0; i < 256; i += 8) { /* z lines */
    for (int t = 0; t < 16; t++)
      li[t] = zmask >> t & 1u ? _mm256_loadu_ps(blk + t * 256 + i) : _mm256_setzero_ps();
    VF_IDCT16_STRICT(__m256, li, lo);
    for (int t = 0; t < 16; t++) _mm256_storeu_ps(dst + t * 256 + i, lo[t]);
  }
}
#endif
static inline void vf_dct16_inv_strict(float blk[VF_BLKV], uint32_t zmask, uint32_t hmask, float out[VF_BLKV]) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) { vf_dct16_inv_strict_avx2(blk, zmask, hmask, out); return; }
#endif
  vf_dct16_inv_strict_c(blk, zmask, hmask, out);
}
__attribute__((always_inline)) static inline float vf_dequant_ac(uint32_t mag, float step) {
  return ((float)mag + (mag == 1u ? VF_DZ_DQ1 : VF_DZ_DQ)) * step;
}
__attribute__((always_inline)) static inline float vf_dequant_dc(uint32_t mag, float step) { return (float)mag * step; }

/* ---- block gather / scatter ---- */
static inline size_t vf_row_off(uint32_t bz, uint32_t by, uint32_t bx, uint32_t z, uint32_t y) {
  return ((size_t)(bz * 16u + z) * VOLCOMP_CHUNK_DIM + (by * 16u + y)) * VOLCOMP_CHUNK_DIM +
         (size_t)bx * 16u;
}
#if VF_HAVE_AVX2
/* u8 -> float - 128; returns true iff every voxel equals src[0] of the block */
VF_TARGET_AVX2 static bool vf_gather_block_avx2(const uint8_t *src, uint32_t bz, uint32_t by, uint32_t bx,
                                                float *blk) {
  const __m256 bias = _mm256_set1_ps(128.0f);
  __m128i mn = _mm_set1_epi8(-1), mx = _mm_setzero_si128();
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y++) {
      __m128i row = _mm_loadu_si128((const __m128i *)(src + vf_row_off(bz, by, bx, z, y)));
      mn = _mm_min_epu8(mn, row);
      mx = _mm_max_epu8(mx, row);
      float *o = blk + (size_t)z * 256 + (size_t)y * 16;
      __m256i lo = _mm256_cvtepu8_epi32(row);
      __m256i hi = _mm256_cvtepu8_epi32(_mm_srli_si128(row, 8));
      _mm256_storeu_ps(o, _mm256_sub_ps(_mm256_cvtepi32_ps(lo), bias));
      _mm256_storeu_ps(o + 8, _mm256_sub_ps(_mm256_cvtepi32_ps(hi), bias));
    }
  return _mm_movemask_epi8(_mm_cmpeq_epi8(mn, mx)) == 0xffff &&
         _mm_movemask_epi8(_mm_cmpeq_epi8(mn, _mm_set1_epi8((char)src[vf_row_off(bz, by, bx, 0, 0)]))) == 0xffff;
}
/* +128.5, clamp to [0,255], truncate: one 32-byte pack per pair of rows */
__attribute__((always_inline)) VF_TARGET_AVX2 static inline __m256i vf_round8(const float *p) {
  __m256 v = _mm256_add_ps(_mm256_loadu_ps(p), _mm256_set1_ps(128.5f));
  v = _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(255.0f));
  return _mm256_cvttps_epi32(v);
}
/* row = dst + z*zs + y*ys */
VF_TARGET_AVX2 static void vf_scatter_block_avx2(uint8_t *restrict dst, size_t zs, size_t ys,
                                                 const float *restrict blk) {
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y += 2) {
      const float *in = blk + (size_t)z * 256 + (size_t)y * 16;
      __m256i p = _mm256_packus_epi32(vf_round8(in), vf_round8(in + 8));
      __m256i q = _mm256_packus_epi32(vf_round8(in + 16), vf_round8(in + 24));
      __m256i b = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(p, q), order);
      uint8_t *row = dst + z * zs + y * ys;
      _mm_storeu_si128((__m128i *)row, _mm256_castsi256_si128(b));
      _mm_storeu_si128((__m128i *)(row + ys), _mm256_extracti128_si256(b, 1));
    }
}
#endif /* VF_HAVE_AVX2 */
static bool vf_gather_block_c(const uint8_t *src, uint32_t bz, uint32_t by, uint32_t bx, float *blk) {
  const uint8_t v0 = src[vf_row_off(bz, by, bx, 0, 0)];
  bool flat = true;
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y++) {
      const uint8_t *row = src + vf_row_off(bz, by, bx, z, y);
      float *o = blk + (size_t)z * 256 + (size_t)y * 16;
      for (uint32_t x = 0; x < 16; x++) {
        flat &= row[x] == v0;
        o[x] = (float)row[x] - 128.0f;
      }
    }
  return flat;
}
static void vf_scatter_block_c(uint8_t *restrict dst, size_t zs, size_t ys, const float *restrict blk) {
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y++) {
      const float *in = blk + (size_t)z * 256 + (size_t)y * 16;
      uint8_t *row = dst + z * zs + y * ys;
      for (uint32_t x = 0; x < 16; x++) {
        float v = in[x] + 128.5f;
        v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        row[x] = (uint8_t)(int32_t)v;
      }
    }
}
static inline bool vf_gather_block(const uint8_t *src, uint32_t bz, uint32_t by, uint32_t bx, float *blk) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) return vf_gather_block_avx2(src, bz, by, bx, blk);
#endif
  return vf_gather_block_c(src, bz, by, bx, blk);
}
static inline void vf_scatter_block(uint8_t *restrict dst, size_t zs, size_t ys, const float *restrict blk) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) { vf_scatter_block_avx2(dst, zs, ys, blk); return; }
#endif
  vf_scatter_block_c(dst, zs, ys, blk);
}
/* unnormalised IDCT of a DC-only block is the constant blk[0] (scale already folded) */
static inline uint8_t vf_flat_value(float dc) {
  float v = dc + 128.5f;
  v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
  return (uint8_t)(int32_t)v;
}

/* ---- parsed stream ---- */
typedef struct vf_parsed {
  float q;
  bool strict; /* a surface chunk's base: the bit-exact reconstruction (spec §13.2) */
  float step[VF_NRADII];
  vf_model models[VF_NMODELS];
  uint32_t tok_n[VF_NSUB], byp_n[VF_NSUB], off[VF_NSUB + 1];
  const uint8_t *payload;
} vf_parsed;

/* The body of a DCT stream — tables, directory, payload, i.e. everything after
 * the 8-byte header — in [body, end), at step q_raw / 256. A surface chunk's
 * base is such a body (spec §13); `strict` selects its bit-exact reconstruction. */
static volcomp_status vf_parse_body(const uint8_t *body, const uint8_t *end, uint32_t qraw, bool strict,
                                    bool flat, vf_parsed *restrict p) {
  if ((size_t)(end - body) < VF_DIR_BYTES) return VOLCOMP_ERR_CORRUPT;
  if (qraw < VF_Q_RAW_MIN || qraw > VF_Q_RAW_MAX) return VOLCOMP_ERR_CORRUPT;
  p->q = (float)qraw * (1.0f / 256.0f);
  p->strict = strict;
  if (flat)
    vf_steps_flat(p->q, p->step);
  else if (strict)
    vf_steps_strict(p->q, p->step);
  else
    vf_steps(p->q, p->step);
  vf_cur c = {body, end};
  if (!vf_tables_read(&c, p->models, VF_NMODELS)) return VOLCOMP_ERR_CORRUPT;
  if (!vf_cur_has(&c, VF_DIR_BYTES)) return VOLCOMP_ERR_CORRUPT;
  uint64_t sum = 0;
  for (uint32_t s = 0; s < VF_NSUB; s++) {
    uint32_t rn = vf_rd_u32(c.p + s * 8), bn = vf_rd_u32(c.p + s * 8 + 4);
    if (rn < 3u || rn > VF_SUB_TOK_MAX || bn > VF_SUB_BYP_MAX) return VOLCOMP_ERR_CORRUPT;
    p->tok_n[s] = rn;
    p->byp_n[s] = bn;
    p->off[s] = (uint32_t)sum;
    sum += (uint64_t)rn + bn;
  }
  p->off[VF_NSUB] = (uint32_t)sum;
  c.p += VF_DIR_BYTES;
  if ((uint64_t)(c.end - c.p) != sum) return VOLCOMP_ERR_CORRUPT; /* exact accounting */
  p->payload = c.p;
  return VOLCOMP_OK;
}
static volcomp_status vf_parse(const void *restrict enc, size_t enc_n, vf_parsed *restrict p) {
  const uint8_t *in = (const uint8_t *)enc;
  if (enc_n < VF_HDR_BYTES + VF_DIR_BYTES) return VOLCOMP_ERR_CORRUPT;
  if (in[0] != 'V' || in[1] != 'O' || in[2] != 'L' || in[3] != 'C') return VOLCOMP_ERR_CORRUPT;
  if (in[4] != VF_VERSION) return VOLCOMP_ERR_VERSION;
  if (in[5] != VLL_MODE_LOSSY) return VOLCOMP_ERR_CORRUPT; /* not a lossy chunk */
  return vf_parse_body(in + VF_HDR_BYTES, in + enc_n, vf_rd_u16(in + 6), false, false, p);
}

/* Decode substream s. only_block < 0: scatter its 16 blocks into the chunk dst;
 * else stop after absolute block only_block and write it into dst_block[4096]. */
static volcomp_status vf_decode_sub(const vf_parsed *restrict p, uint32_t s, uint8_t *restrict dst,
                                    int32_t only_block, uint8_t *restrict dst_block) {
  const uint8_t *rb = p->payload + p->off[s];
  vf_rdec rd;
  if (!vf_rdec_init(&rd, rb, p->tok_n[s])) return VOLCOMP_ERR_CORRUPT;
  vf_bitr br;
  vf_br_init(&br, rb + p->tok_n[s], p->byp_n[s]);
  int64_t prev_dc = 0;
  float blk[VF_BLKV] = {0}, vox[VF_BLKV]; /* blk stays zero between blocks except its zmask planes */
  for (uint32_t bi = s * VF_BLOCKS_PER_SUB; bi < (s + 1) * VF_BLOCKS_PER_SUB; bi++) {
    int t = vf_rdec_get(&rd, &p->models[VF_DC_CTX]);
    if (t < 0 || (uint32_t)t > VF_TOKMAX_DC) return VOLCOMP_ERR_CORRUPT;
    uint32_t u;
    if (!vf_hyb_read(&br, (uint32_t)t, &u)) return VOLCOMP_ERR_CORRUPT;
    int64_t dc = prev_dc + vf_unzigzag(u);
    if (dc < -VF_DC_ABS_MAX || dc > VF_DC_ABS_MAX) return VOLCOMP_ERR_CORRUPT;
    prev_dc = dc;
    if (dc != 0) {
      float mag = vf_dequant_dc((uint32_t)(dc < 0 ? -dc : dc), p->step[0]) * 0.015625f;
      blk[0] = dc < 0 ? -mag : mag;
    }
    uint32_t pos = 1, nnz_ac = 0, zmask = 1u, hmask = 0;
    for (;;) {
      t = vf_rdec_get(&rd, &p->models[vf_run_ctx(pos)]);
      if (t < 0) return VOLCOMP_ERR_CORRUPT;
      if ((uint32_t)t == VF_TOK_EOB) break;
      if ((uint32_t)t > VF_TOKMAX_RUN) return VOLCOMP_ERR_CORRUPT;
      uint32_t run;
      if (!vf_hyb_read(&br, (uint32_t)t, &run)) return VOLCOMP_ERR_CORRUPT;
      pos += run;
      if (pos >= VF_BLKV) return VOLCOMP_ERR_CORRUPT;
      int lt = vf_rdec_get(&rd, &p->models[vf_level_ctx(pos, run)]);
      if (lt < 0 || (uint32_t)lt > VF_TOKMAX_LVL) return VOLCOMP_ERR_CORRUPT;
      uint32_t mag1, sign;
      if (!vf_hyb_read(&br, (uint32_t)lt, &mag1) || !vf_br_get(&br, 1, &sign))
        return VOLCOMP_ERR_CORRUPT;
      if (mag1 + 1u > VF_AC_MAG_MAX) return VOLCOMP_ERR_CORRUPT;
      uint32_t nat = VF_SCAN16[pos];
      float stp = p->step[vf_radius(nat)];
      float v = vf_dequant_ac(mag1 + 1u, stp) * vf_dct_scale(nat);
      blk[nat] = sign ? -v : v;
      zmask |= 1u << (nat >> 8);
      hmask |= (nat >> 7 & 1u) << (nat >> 8);
      nnz_ac++;
      pos++;
    }
    if (only_block >= 0 && (uint32_t)only_block != bi) {
      vf_clear_planes(blk, zmask);
      continue;
    }
    uint32_t bz = bi >> 6, by = (bi >> 3) & 7u, bx = bi & 7u;
    if (nnz_ac == 0) { /* only blk[0] was written */
      uint8_t bv = vf_flat_value(blk[0]);
      blk[0] = 0.0f;
      if (only_block >= 0) {
        memset(dst_block, bv, VF_BLKV);
        return VOLCOMP_OK;
      }
      for (uint32_t z = 0; z < 16; z++)
        for (uint32_t y = 0; y < 16; y++) memset(dst + vf_row_off(bz, by, bx, z, y), bv, 16);
      continue;
    }
    if (p->strict)
      vf_dct16_inv_strict(blk, zmask, hmask, vox);
    else
      vf_dct16_inv(blk, zmask, hmask, vox);
    vf_clear_planes(blk, zmask);
    if (only_block >= 0) {
      vf_scatter_block(dst_block, 256, 16, vox);
      return VOLCOMP_OK;
    }
    vf_scatter_block(dst + vf_row_off(bz, by, bx, 0, 0),
                     (size_t)VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM, VOLCOMP_CHUNK_DIM, vox);
  }
  if (only_block >= 0) return VOLCOMP_ERR_ARG;
  if (!vf_rdec_finished(&rd) || !vf_br_finished(&br)) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

/* ---- encoder ---- */
typedef struct vf_enc {
  uint16_t *toks; /* packed tok | ctx<<8, all substreams back to back */
  uint8_t *byp;   /* bypass bytes, all substreams back to back        */
  uint8_t *stage; /* one substream's token-stream output              */
  float *rstep;   /* 4096 reciprocal steps, natural order             */
  size_t tok_off[VF_NSUB + 1], byp_off[VF_NSUB + 1];
  uint32_t counts[VF_NMODELS][VF_NTOK];
} vf_enc;

static inline uint32_t vf_q_raw(float q) {
  float r = q * 256.0f + 0.5f;
  if (!(r >= (float)VF_Q_RAW_MIN)) return 0;
  return r > (float)VF_Q_RAW_MAX ? VF_Q_RAW_MAX : (uint32_t)r;
}

/* quantise (reciprocal multiply + dead zone: a = |v| r + 0.2 truncates to 0
 * below 1) into qn (natural order) and list the nonzero AC positions in
 * scan coordinates, natural order (DC excluded); returns the count. */
#if VF_HAVE_AVX2
/* 32-wide bitmasks; the ctz walk touches only nonzeros (typically 50-500 per
 * block) — no 4096-wide permutation */
VF_TARGET_AVX2 static uint32_t vf_quantise_block_avx2(const float *blk, const float *rstep, int32_t *qn,
                                                      uint16_t *npos) {
  const __m256 dz = _mm256_set1_ps(VF_DZ_Q), absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const __m256i zero = _mm256_setzero_si256();
  uint32_t n = 0;
  for (uint32_t i = 0; i < VF_BLKV; i += 32) {
    uint32_t m = 0;
    for (uint32_t k = 0; k < 32; k += 8) {
      __m256 v = _mm256_loadu_ps(blk + i + k);
      __m256 a = _mm256_fmadd_ps(_mm256_and_ps(v, absmask), _mm256_loadu_ps(rstep + i + k), dz);
      __m256i l = _mm256_cvttps_epi32(a);
      __m256i s = _mm256_srai_epi32(_mm256_castps_si256(v), 31);
      l = _mm256_sub_epi32(_mm256_xor_si256(l, s), s);
      _mm256_storeu_si256((__m256i *)(qn + i + k), l);
      m |= ((uint32_t)~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(l, zero))) & 0xffu) << k;
    }
    if (i == 0) m &= ~1u; /* DC is coded separately */
    while (m) {
      uint32_t nat = i + (uint32_t)__builtin_ctz(m);
      npos[n++] = VF_ISCAN16[nat];
      m &= m - 1;
    }
  }
  return n;
}
#endif /* VF_HAVE_AVX2 */
static uint32_t vf_quantise_block_c(const float *blk, const float *rstep, int32_t *qn, uint16_t *npos) {
  uint32_t n = 0;
  for (uint32_t i = 0; i < VF_BLKV; i++) {
    float v = blk[i];
    int32_t l = (int32_t)(fabsf(v) * rstep[i] + VF_DZ_Q);
    if (v < 0.0f) l = -l;
    qn[i] = l;
    if (l != 0 && i != 0) npos[n++] = VF_ISCAN16[i];
  }
  return n;
}
static inline uint32_t vf_quantise_block(const float *blk, const float *rstep, int32_t *qn, uint16_t *npos) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) return vf_quantise_block_avx2(blk, rstep, qn, npos);
#endif
  return vf_quantise_block_c(blk, rstep, qn, npos);
}

/* transform + quantise one block into scan-order levels and the ascending
 * nonzero AC position list (single fused pass over the scan table) */
static uint32_t vf_analyse_block(const uint8_t *src, uint32_t bz, uint32_t by, uint32_t bx,
                                 const float *rstep, int32_t *lvl, uint16_t *npos) {
  /* lvl[0] = DC level; lvl[1..n] = level of npos[0..n-1] (scan positions, ascending) */
  float blk[VF_BLKV];
  int32_t qn[VF_BLKV];
  if (vf_gather_block(src, bz, by, bx, blk)) { /* constant block: DC only, skip the transform */
    float c0 = blk[0] * 4096.0f;              /* unnormalised DC of a constant block */
    float a = fabsf(c0) * rstep[0] + VF_DZ_Q_DC;
    int32_t l = (int32_t)a;
    lvl[0] = c0 < 0.0f ? -l : l;
    return 0;
  }
  vf_dct16_fwd(blk);
  { /* DC: round to nearest (a whole-block level shift is the costliest error) */
    float a = fabsf(blk[0]) * rstep[0] + VF_DZ_Q_DC;
    int32_t l = (int32_t)a;
    lvl[0] = blk[0] < 0.0f ? -l : l;
  }
  uint32_t n = vf_quantise_block(blk, rstep, qn, npos);
  /* sort scan positions ascending: insertion sort for sparse blocks, LSD
   * radix (two 6-bit passes) for dense ones */
  if (n < 64) {
    for (uint32_t a = 1; a < n; a++) {
      uint16_t v = npos[a];
      uint32_t b = a;
      while (b > 0 && npos[b - 1] > v) {
        npos[b] = npos[b - 1];
        b--;
      }
      npos[b] = v;
    }
  } else {
    uint16_t tmp[VF_BLKV];
    uint32_t c1[65] = {0}, c2[65] = {0};
    for (uint32_t a = 0; a < n; a++) c1[(npos[a] & 63u) + 1]++;
    for (uint32_t a = 0; a < 64; a++) c1[a + 1] += c1[a];
    for (uint32_t a = 0; a < n; a++) tmp[c1[npos[a] & 63u]++] = npos[a];
    for (uint32_t a = 0; a < n; a++) c2[(tmp[a] >> 6) + 1]++;
    for (uint32_t a = 0; a < 64; a++) c2[a + 1] += c2[a];
    for (uint32_t a = 0; a < n; a++) npos[c2[tmp[a] >> 6]++] = tmp[a];
  }
  for (uint32_t a = 0; a < n; a++) lvl[1 + a] = qn[VF_SCAN16[npos[a]]];
  return n;
}

static bool vf_tokenize_block(vf_enc *e, uint16_t **tp, vf_bitw *bw, const int32_t *lvl,
                              const uint16_t *npos, uint32_t nnz, int64_t *prev_dc) {
  uint16_t *tok = *tp;
  uint32_t t;
  int64_t dc = lvl[0];
  if (!vf_hyb_emit(bw, vf_zigzag((int32_t)(dc - *prev_dc)), &t)) return false;
  *tok++ = (uint16_t)(t | VF_DC_CTX << 8);
  e->counts[VF_DC_CTX][t]++;
  *prev_dc = dc;
  uint32_t prev = 0;
  for (uint32_t k = 0; k < nnz; k++) {
    uint32_t i = npos[k];
    int32_t L = lvl[1 + k];
    uint32_t run = i - prev - 1u;
    uint32_t rct = vf_run_ctx(i - run);
    if (!vf_hyb_emit(bw, run, &t)) return false;
    *tok++ = (uint16_t)(t | rct << 8);
    e->counts[rct][t]++;
    uint32_t mag1 = (uint32_t)(L < 0 ? -L : L) - 1u;
    if (!vf_hyb_emit(bw, mag1, &t)) return false;
    if (!vf_bw_put(bw, L < 0 ? 1u : 0u, 1)) return false;
    uint32_t lc = vf_level_ctx(i, run);
    *tok++ = (uint16_t)(t | lc << 8);
    e->counts[lc][t]++;
    prev = i;
  }
  uint32_t ec = vf_run_ctx(nnz > 0 ? (uint32_t)npos[nnz - 1] + 1u : 1u);
  *tok++ = (uint16_t)(VF_TOK_EOB | ec << 8);
  e->counts[ec][VF_TOK_EOB]++;
  *tp = tok;
  return true;
}

/* ======================================================================== */
/*                     lossless mode (q == 0, chunk mode 1..3)              */
/* ======================================================================== */
/* Header byte 5 selects the chunk mode; the lossy DCT format is mode 0 and is
 * unchanged (byte for byte) from format version 1 as shipped. Lossless streams
 * set qraw = 0, which the mode-0 parser rejects, and a v1.0 decoder rejects any
 * nonzero byte 5 outright (VOLCOMP_ERR_CORRUPT), so old readers never
 * misinterpret a lossless stream.
 *
 *   mode 1 VLL_MODE_CODED  tables + directory + 32 substreams (below)
 *   mode 2 VLL_MODE_RAW    8-byte header + 2097152 raw voxels
 *   mode 3 VLL_MODE_CONST  8-byte header + 1 byte (the chunk's value)
 *
 * mode 1 keeps volcomp's block/substream geometry: 512 blocks of 16^3 in
 * z-major block order, 32 substreams of 16 consecutive blocks, so
 * volcomp_decode_block() still touches one substream and reproduces exactly the
 * bytes of the full decode. Layout:
 *
 *   tables      VLL_NMODELS tANS models, same encoding as the lossy tables
 *   directory   32 x { LEB128 token bytes, LEB128 bypass bytes }
 *   payload     per substream: token bytes then bypass bytes
 *
 * Each block starts with a mode token (model VLL_M_MODE):
 *   0 CONST   4096 equal voxels; 8 bypass bits carry the value (~0 bytes)
 *   1 SPARSE  (run, level)* EOB over the residual array, models VLL_M_RUN /
 *             VLL_M_LVLS -- flat and label-like data
 *   2 DENSE   one level token per voxel, model VLL_M_LVLD + context from the
 *             previous residual -- smooth / continuous data
 *   3 RAW     4096 bypass bytes -- incompressible data
 *
 * The residual of voxel i is v[i] - pred(i) taken mod 256 and zigzagged, with
 * pred = the previous voxel along x, or along y at x == 0, or along z at
 * x == y == 0, or 128 at the block origin. Prediction never crosses a block, so
 * blocks stay independently decodable. Values are exact for every uint8 input:
 * a chunk whose coded form would not fit in 2 MiB is stored as mode 2, so the
 * output is never larger than the raw chunk plus the 8-byte header. */

#define VLL_NMODELS 8u
#define VLL_M_MODE 0u  /* block mode token                                    */
#define VLL_M_RUN 1u   /* SPARSE zero-run tokens and EOB, +0..1 by prev run   */
#define VLL_M_LVLS 3u  /* SPARSE levels (magnitude - 1), +0..1 by this run    */
#define VLL_M_LVLD 5u  /* DENSE level tokens, +0..2 by previous-plane ctx     */

#define VLL_BM_CONST 0u
#define VLL_BM_SPARSE 1u
#define VLL_BM_DENSE 2u
#define VLL_BM_RAW 3u

/* HybridUint variant for the lossless mode: values 0..15 are literal tokens, so
 * every residual of magnitude <= 7 costs one token and no bypass bit; above that
 * tok = 12 + msb(u) with the low msb(u) bits bypassed. Max token 19 for a level
 * (u <= 255) and 23 for a run (u <= 4095), both below VF_TOK_EOB. */
#define VLL_NLIT 16u
__attribute__((always_inline)) static inline bool vll_hyb_emit(vf_bitw *w, uint32_t u, uint32_t *tok) {
  if (u < VLL_NLIT) {
    *tok = u;
    return true;
  }
  uint32_t k = vf_highbit(u);
  *tok = VLL_NLIT - 4u + k;
  return vf_bw_put(w, u & ((1u << k) - 1u), k);
}
__attribute__((always_inline)) static inline bool vll_hyb_read(vf_bitr *r, uint32_t tok, uint32_t *u) {
  if (tok < VLL_NLIT) {
    *u = tok;
    return true;
  }
  uint32_t k = tok - (VLL_NLIT - 4u), extra;
  if (!vf_br_get(r, k, &extra)) return false;
  *u = (1u << k) | extra;
  return true;
}
#define VLL_TOKMAX_RUN 23u /* run <= 4095 */
#define VLL_TOKMAX_LVL 19u /* level <= 255 */
#define VLL_SPARSE_MAX_NNZ 2048u
/* per block: SPARSE emits at most 2*nnz+1 tokens, DENSE 4096 */
#define VLL_BLK_TOK_MAX (2u * VLL_SPARSE_MAX_NNZ + 1u)
/* per block bypass: RAW 4096 B; DENSE 4096*7 bits; SPARSE nnz*(11+7) bits */
#define VLL_BLK_BYP_MAX 4608u
#define VLL_SUB_TOK_MAX (2u * VF_BLOCKS_PER_SUB * VLL_BLK_TOK_MAX + 8u)
#define VLL_SUB_BYP_MAX (VF_BLOCKS_PER_SUB * VLL_BLK_BYP_MAX + 8u)

/* ---- block gather / scatter (plain bytes) ---- */
static inline void vll_gather(const uint8_t *restrict src, uint32_t bz, uint32_t by, uint32_t bx,
                              uint8_t *restrict b) {
  for (uint32_t z = 0; z < VOLCOMP_BLOCK_DIM; z++)
    for (uint32_t y = 0; y < VOLCOMP_BLOCK_DIM; y++)
      memcpy(b + (z * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_BLOCK_DIM, src + vf_row_off(bz, by, bx, z, y),
             VOLCOMP_BLOCK_DIM);
}
static inline void vll_scatter(uint8_t *restrict dst, uint32_t bz, uint32_t by, uint32_t bx,
                               const uint8_t *restrict b) {
  for (uint32_t z = 0; z < VOLCOMP_BLOCK_DIM; z++)
    for (uint32_t y = 0; y < VOLCOMP_BLOCK_DIM; y++)
      memcpy(dst + vf_row_off(bz, by, bx, z, y), b + (z * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_BLOCK_DIM,
             VOLCOMP_BLOCK_DIM);
}

/* Residual of a gathered 16^3 block: pred(i) is the voxel one step back along z,
 * or along y inside the first z-plane, or along x inside the first row, or 128
 * at the origin; the residual is (v - pred) mod 256 zigzagged into 0..255.
 * Prediction never leaves the block, so blocks stay independently decodable, and
 * every step is a plane-at-a-time (or row-at-a-time) byte operation the compiler
 * vectorises: the reconstruction is 15 vector adds over 256 lanes, 15 over 16
 * lanes and a 15-step scalar tail, not a 4096-long serial chain. */
#define VLL_ZZ(d) ((uint8_t)((uint32_t)((d) << 1) ^ (uint32_t)(uint8_t)(0u - ((uint32_t)(d) >> 7))))
#define VLL_UNZZ(u) ((uint8_t)((uint32_t)((u) >> 1) ^ (uint32_t)(uint8_t)(0u - ((u) & 1u))))

static void vll_residual(const uint8_t *restrict b, uint8_t *restrict u) {
  u[0] = VLL_ZZ((uint8_t)(b[0] - 128u));
  for (uint32_t x = 1; x < 16; x++) u[x] = VLL_ZZ((uint8_t)(b[x] - b[x - 1]));
  for (uint32_t y = 1; y < 16; y++)
    for (uint32_t x = 0; x < 16; x++)
      u[y * 16 + x] = VLL_ZZ((uint8_t)(b[y * 16 + x] - b[(y - 1) * 16 + x]));
  for (uint32_t z = 1; z < 16; z++)
    for (uint32_t j = 0; j < 256; j++)
      u[z * 256 + j] = VLL_ZZ((uint8_t)(b[z * 256 + j] - b[(z - 1) * 256 + j]));
}
/* inverse; u is consumed (unzigzagged in place) */
static void vll_rebuild(uint8_t *restrict b, uint8_t *restrict u) {
  for (uint32_t i = 0; i < VF_BLKV; i++) u[i] = VLL_UNZZ(u[i]);
  b[0] = (uint8_t)(128u + u[0]);
  for (uint32_t x = 1; x < 16; x++) b[x] = (uint8_t)(b[x - 1] + u[x]);
  for (uint32_t y = 1; y < 16; y++)
    for (uint32_t x = 0; x < 16; x++) b[y * 16 + x] = (uint8_t)(b[(y - 1) * 16 + x] + u[y * 16 + x]);
  for (uint32_t z = 1; z < 16; z++)
    for (uint32_t j = 0; j < 256; j++) b[z * 256 + j] = (uint8_t)(b[(z - 1) * 256 + j] + u[z * 256 + j]);
}
/* rough cost in bits of coding value u as a HybridUint token plus its bypass bits */
static inline uint32_t vll_bits(uint32_t u) { return u < VLL_NLIT ? 4u : vf_highbit(u) + 5u; }
/* DENSE context: the activity of the residual one z-plane back (0 in the first
 * plane). Deliberately not the immediately preceding residual -- that would put
 * a table selection on the entropy decoder's critical path; this value is known
 * 256 symbols ahead and predicts just as well. */
/* SPARSE contexts: a run following a run of zero (adjacent nonzero residuals,
 * i.e. inside a busy region) has a very different distribution from one that
 * follows a gap, and so does the level that goes with it. */
static inline uint32_t vll_run_ctx(uint32_t prev_run) { return VLL_M_RUN + (prev_run == 0u ? 0u : 1u); }
static inline uint32_t vll_lvls_ctx(uint32_t run) { return VLL_M_LVLS + (run == 0u ? 0u : 1u); }
static inline uint32_t vll_dense_ctx(const uint8_t *u, uint32_t i) {
  uint32_t a = i < 256u ? 0u : u[i - 256u];
  return VLL_M_LVLD + (a == 0u ? 0u : (a <= 2u ? 1u : 2u));
}

/* ---- decoder ---- */
typedef struct vll_parsed {
  vf_model models[VLL_NMODELS];
  uint32_t tok_n[VF_NSUB], byp_n[VF_NSUB], off[VF_NSUB + 1];
  const uint8_t *payload;
} vll_parsed;

static volcomp_status vll_parse(const uint8_t *in, size_t n, vll_parsed *restrict p) {
  vf_cur c = {in + VF_HDR_BYTES, in + n};
  if (!vf_tables_read(&c, p->models, VLL_NMODELS)) return VOLCOMP_ERR_CORRUPT;
  uint64_t sum = 0;
  for (uint32_t s = 0; s < VF_NSUB; s++) {
    uint32_t rn, bn;
    if (!vf_leb_get(&c, &rn) || !vf_leb_get(&c, &bn)) return VOLCOMP_ERR_CORRUPT;
    if (rn < 3u || rn > VLL_SUB_TOK_MAX || bn > VLL_SUB_BYP_MAX) return VOLCOMP_ERR_CORRUPT;
    p->tok_n[s] = rn;
    p->byp_n[s] = bn;
    p->off[s] = (uint32_t)sum;
    sum += (uint64_t)rn + bn;
  }
  p->off[VF_NSUB] = (uint32_t)sum;
  if ((uint64_t)(c.end - c.p) != sum) return VOLCOMP_ERR_CORRUPT; /* exact accounting */
  p->payload = c.p;
  return VOLCOMP_OK;
}

/* Decode substream s. only_block < 0: scatter its 16 blocks into dst; else stop
 * after absolute block only_block and write it into dst_block[4096]. */
static volcomp_status vll_decode_sub(const vll_parsed *restrict p, uint32_t s, uint8_t *restrict dst,
                                     int32_t only_block, uint8_t *restrict dst_block) {
  const uint8_t *rb = p->payload + p->off[s];
  vf_rdec rd;
  if (!vf_rdec_init(&rd, rb, p->tok_n[s])) return VOLCOMP_ERR_CORRUPT;
  vf_bitr br;
  vf_br_init(&br, rb + p->tok_n[s], p->byp_n[s]);
  uint8_t b[VF_BLKV], u[VF_BLKV];
  for (uint32_t bi = s * VF_BLOCKS_PER_SUB; bi < (s + 1u) * VF_BLOCKS_PER_SUB; bi++) {
    int t = vf_rdec_get(&rd, &p->models[VLL_M_MODE]);
    if (t < 0 || (uint32_t)t > VLL_BM_RAW) return VOLCOMP_ERR_CORRUPT;
    switch ((uint32_t)t) {
      case VLL_BM_CONST: {
        uint32_t v;
        if (!vf_br_get(&br, 8, &v)) return VOLCOMP_ERR_CORRUPT;
        memset(b, (int)v, VF_BLKV);
        break;
      }
      case VLL_BM_RAW: {
        for (uint32_t i = 0; i < VF_BLKV; i++) {
          uint32_t v;
          if (!vf_br_get(&br, 8, &v)) return VOLCOMP_ERR_CORRUPT;
          b[i] = (uint8_t)v;
        }
        break;
      }
      case VLL_BM_SPARSE: {
        memset(u, 0, VF_BLKV);
        uint32_t pos = 0, prev_run = 1;
        for (;;) {
          int rt = vf_rdec_get(&rd, &p->models[vll_run_ctx(prev_run)]);
          if (rt < 0) return VOLCOMP_ERR_CORRUPT;
          if ((uint32_t)rt == VF_TOK_EOB) break;
          if ((uint32_t)rt > VLL_TOKMAX_RUN) return VOLCOMP_ERR_CORRUPT;
          uint32_t run;
          if (!vll_hyb_read(&br, (uint32_t)rt, &run)) return VOLCOMP_ERR_CORRUPT;
          if (pos >= VF_BLKV || run > VF_BLKV - 1u - pos) return VOLCOMP_ERR_CORRUPT;
          pos += run;
          prev_run = run;
          int lt = vf_rdec_get(&rd, &p->models[vll_lvls_ctx(run)]);
          if (lt < 0 || (uint32_t)lt > VLL_TOKMAX_LVL) return VOLCOMP_ERR_CORRUPT;
          uint32_t uu;
          if (!vll_hyb_read(&br, (uint32_t)lt, &uu)) return VOLCOMP_ERR_CORRUPT;
          if (uu > 254u) return VOLCOMP_ERR_CORRUPT;
          u[pos++] = (uint8_t)(uu + 1u);
        }
        vll_rebuild(b, u);
        break;
      }
      default: { /* VLL_BM_DENSE */
        for (uint32_t i = 0; i < VF_BLKV; i++) {
          int lt = vf_rdec_get(&rd, &p->models[vll_dense_ctx(u, i)]);
          if (lt < 0 || (uint32_t)lt > VLL_TOKMAX_LVL) return VOLCOMP_ERR_CORRUPT;
          uint32_t uu;
          if (!vll_hyb_read(&br, (uint32_t)lt, &uu)) return VOLCOMP_ERR_CORRUPT;
          u[i] = (uint8_t)uu;
        }
        vll_rebuild(b, u);
        break;
      }
    }
    if (only_block >= 0) {
      if ((uint32_t)only_block != bi) continue;
      memcpy(dst_block, b, VF_BLKV);
      return VOLCOMP_OK;
    }
    vll_scatter(dst, bi >> 6, (bi >> 3) & 7u, bi & 7u, b);
  }
  if (only_block >= 0) return VOLCOMP_ERR_ARG;
  if (!vf_rdec_finished(&rd) || !vf_br_finished(&br)) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

/* ---- encoder ---- */
typedef struct vll_enc {
  uint16_t *toks;
  uint8_t *byp;
  size_t tok_off[VF_NSUB + 1], byp_off[VF_NSUB + 1];
  uint32_t counts[VLL_NMODELS][VF_NTOK];
} vll_enc;

#define VLL_PUT(m, tk)                             \
  do {                                             \
    uint32_t m_ = (m), t_ = (tk);                  \
    *tok++ = (uint16_t)(t_ | m_ << 8);             \
    e->counts[m_][t_]++;                           \
  } while (0)

static bool vll_encode_block(vll_enc *restrict e, uint16_t **tp, vf_bitw *bw, const uint8_t *b) {
  uint16_t *tok = *tp;
  uint32_t v0 = b[0], i;
  for (i = 1; i < VF_BLKV; i++)
    if (b[i] != v0) break;
  if (i == VF_BLKV) {
    VLL_PUT(VLL_M_MODE, VLL_BM_CONST);
    if (!vf_bw_put(bw, v0, 8)) return false;
    *tp = tok;
    return true;
  }
  uint8_t u[VF_BLKV];
  vll_residual(b, u);
  uint64_t cd = 0, cs = 8;
  uint32_t nnz = 0;
  for (i = 0; i < VF_BLKV; i++) {
    uint32_t uu = u[i];
    cd += vll_bits(uu);
    if (uu) {
      nnz++;
      cs += vll_bits(uu - 1u) + 5u;
    }
  }
  uint32_t mode = (nnz <= VLL_SPARSE_MAX_NNZ && cs < cd) ? VLL_BM_SPARSE : VLL_BM_DENSE;
  if ((mode == VLL_BM_SPARSE ? cs : cd) >= (uint64_t)VF_BLKV * 8u) mode = VLL_BM_RAW;
  VLL_PUT(VLL_M_MODE, mode);
  if (mode == VLL_BM_RAW) {
    for (i = 0; i < VF_BLKV; i++)
      if (!vf_bw_put(bw, b[i], 8)) return false;
  } else if (mode == VLL_BM_SPARSE) {
    uint32_t pos = 0, prev_run = 1;
    for (i = 0; i < VF_BLKV; i++) {
      if (!u[i]) continue;
      uint32_t t, run = i - pos;
      if (!vll_hyb_emit(bw, run, &t)) return false;
      VLL_PUT(vll_run_ctx(prev_run), t);
      if (!vll_hyb_emit(bw, (uint32_t)u[i] - 1u, &t)) return false;
      VLL_PUT(vll_lvls_ctx(run), t);
      prev_run = run;
      pos = i + 1u;
    }
    VLL_PUT(vll_run_ctx(prev_run), VF_TOK_EOB);
  } else {
    for (i = 0; i < VF_BLKV; i++) {
      uint32_t t;
      if (!vll_hyb_emit(bw, u[i], &t)) return false;
      VLL_PUT(vll_dense_ctx(u, i), t);
    }
  }
  *tp = tok;
  return true;
}
#undef VLL_PUT

static inline void vll_hdr(uint8_t *dst, uint32_t mode) {
  dst[0] = 'V';
  dst[1] = 'O';
  dst[2] = 'L';
  dst[3] = 'C';
  dst[4] = VF_VERSION;
  dst[5] = (uint8_t)mode;
  vf_wr_u16(dst + 6, 0);
}

static volcomp_status vll_encode(const uint8_t *restrict src, uint8_t *restrict dst, size_t dst_cap,
                                 size_t *out_n) {
  { /* whole-chunk constant: 9 bytes */
    size_t i = 1;
    while (i < VOLCOMP_CHUNK_VOXELS && src[i] == src[0]) i++;
    if (i == VOLCOMP_CHUNK_VOXELS) {
      if (dst_cap < VF_HDR_BYTES + 1u) return VOLCOMP_ERR_SHORT_BUF;
      vll_hdr(dst, VLL_MODE_CONST);
      dst[VF_HDR_BYTES] = src[0];
      *out_n = VF_HDR_BYTES + 1u;
      return VOLCOMP_OK;
    }
  }
  const size_t sz_toks = (size_t)VF_BLOCKS * VLL_BLK_TOK_MAX * sizeof(uint16_t);
  const size_t sz_byp = (size_t)VLL_SUB_BYP_MAX * VF_NSUB;
  const size_t sz_code = (size_t)VLL_SUB_TOK_MAX * VF_NSUB;
  const size_t sz_es = sizeof(vf_etab) * VLL_NMODELS;
  const size_t sz_fields = ((size_t)VF_BLOCKS_PER_SUB * VLL_BLK_TOK_MAX + 2u) * sizeof(uint32_t);
  uint8_t *mem = (uint8_t *)VOLCOMP_MALLOC(sz_toks + sz_byp + sz_code + sz_es + sz_fields);
  if (!mem) return VOLCOMP_ERR_NOMEM;
  vll_enc e;
  memset(e.counts, 0, sizeof e.counts);
  e.toks = (uint16_t *)(void *)mem;
  e.byp = mem + sz_toks;
  uint8_t *code = e.byp + sz_byp;
  vf_etab *es = (vf_etab *)(void *)(code + sz_code);
  uint32_t *fields = (uint32_t *)(void *)(code + sz_code + sz_es);
  volcomp_status st = VOLCOMP_ERR_CORRUPT; /* internal invariant failure */

  /* phase A: residuals and tokens, substream by substream */
  uint16_t *tp = e.toks;
  size_t byp_pos = 0;
  for (uint32_t s = 0; s < VF_NSUB; s++) {
    e.tok_off[s] = (size_t)(tp - e.toks);
    e.byp_off[s] = byp_pos;
    vf_bitw bw;
    vf_bw_init(&bw, e.byp + byp_pos, VLL_SUB_BYP_MAX);
    for (uint32_t bi = s * VF_BLOCKS_PER_SUB; bi < (s + 1u) * VF_BLOCKS_PER_SUB; bi++) {
      uint8_t b[VF_BLKV];
      vll_gather(src, bi >> 6, (bi >> 3) & 7u, bi & 7u, b);
      if (!vll_encode_block(&e, &tp, &bw, b)) goto out;
    }
    size_t bn;
    if (!vf_bw_flush(&bw, &bn)) goto out;
    byp_pos += bn;
  }
  e.tok_off[VF_NSUB] = (size_t)(tp - e.toks);
  e.byp_off[VF_NSUB] = byp_pos;

  /* phase B: models and tANS into the staging buffer; phase C: assemble */
  {
    vf_model models[VLL_NMODELS];
    for (uint32_t m = 0; m < VLL_NMODELS; m++) {
      bool any = false;
      for (uint32_t t = 0; t < VF_NTOK; t++) any |= e.counts[m][t] != 0;
      if (!any) e.counts[m][0] = 1; /* unused model: minimal valid table */
      if (!vf_model_build(&models[m], e.counts[m])) goto out;
    }
    vf_etabs_init(models, es, VLL_NMODELS);
    uint8_t tables[VF_TABLES_MAX_BYTES];
    size_t tn = vf_tables_write(models, tables, VLL_NMODELS);
    uint8_t dir[VF_NSUB * 10u];
    size_t rn[VF_NSUB], dn = 0, body = 0;
    for (uint32_t s = 0; s < VF_NSUB; s++) {
      size_t ntok = e.tok_off[s + 1] - e.tok_off[s];
      rn[s] = vf_tans_encode2(es, e.toks + e.tok_off[s], ntok, fields,
                              code + (size_t)s * VLL_SUB_TOK_MAX, VLL_SUB_TOK_MAX);
      if (rn[s] == 0) goto out;
      size_t bn = e.byp_off[s + 1] - e.byp_off[s];
      dn += vf_leb_put(dir + dn, (uint32_t)rn[s]);
      dn += vf_leb_put(dir + dn, (uint32_t)bn);
      body += rn[s] + bn;
    }
    size_t total = VF_HDR_BYTES + tn + dn + body;
    if (total >= VOLCOMP_CHUNK_VOXELS + VF_HDR_BYTES) { /* incompressible: store raw */
      if (dst_cap < VOLCOMP_CHUNK_VOXELS + VF_HDR_BYTES) {
        st = VOLCOMP_ERR_SHORT_BUF;
        goto out;
      }
      vll_hdr(dst, VLL_MODE_RAW);
      memcpy(dst + VF_HDR_BYTES, src, VOLCOMP_CHUNK_VOXELS);
      *out_n = VOLCOMP_CHUNK_VOXELS + VF_HDR_BYTES;
      st = VOLCOMP_OK;
      goto out;
    }
    if (dst_cap < total) {
      st = VOLCOMP_ERR_SHORT_BUF;
      goto out;
    }
    vll_hdr(dst, VLL_MODE_CODED);
    memcpy(dst + VF_HDR_BYTES, tables, tn);
    memcpy(dst + VF_HDR_BYTES + tn, dir, dn);
    size_t pos = VF_HDR_BYTES + tn + dn;
    for (uint32_t s = 0; s < VF_NSUB; s++) {
      size_t bn = e.byp_off[s + 1] - e.byp_off[s];
      memcpy(dst + pos, code + (size_t)s * VLL_SUB_TOK_MAX, rn[s]);
      pos += rn[s];
      memcpy(dst + pos, e.byp + e.byp_off[s], bn);
      pos += bn;
    }
    if (pos != total) goto out;
    *out_n = pos;
    st = VOLCOMP_OK;
  }
out:
  VOLCOMP_FREE(mem);
  return st;
}

/* ======================= binary mask chunks (mode 4) =======================
 * spec/format.md §11. The stored grid is 64^3 bits in z-major order (the 2x2x2
 * majority pool of the source); each bit is coded with an LZMA-style binary
 * range coder (12-bit probabilities, adaptation shift 5) under the context
 * formed by its 12 causal neighbours
 *   (0,0,-1) (0,-1,0) (-1,0,0) (0,-1,-1) (-1,0,-1) (-1,-1,0)
 *   (0,0,-2) (0,-2,0) (-2,0,0) (0,-1,1)  (-1,0,1)  (-1,1,0)      [(dz,dy,dx)]
 * (out of the grid = 0), bit j of the context being neighbour j of that list:
 * 4096 contexts, 8 KB of state, L1 resident.
 *
 * Ten of the twelve neighbours lie in earlier ROWS, so they are gathered for a
 * whole row in one straight-line pass (vmk_pre_row) and the coder loop only ORs
 * in the two bits it has just coded itself; the gather never sits on the
 * entropy coder's critical path.
 *
 * Decoding interpolates: output voxel j samples the grid at j/2, i.e. with
 * weights 1 (j even) or 1/2, 1/2 (j odd, the upper neighbour clamped at the top
 * edge), so the trilinear value is always k/8 for an integer k in 0..8 and the
 * output is VMK_LUT[k] = round(255 k / 8). */
static volcomp_status vll_chunk_mode(const void *restrict enc, size_t enc_n, uint32_t *mode);

#define VMK_FORM_CODED 0u
#define VMK_FORM_CONST 1u
#define VMK_FORM_RAW 2u
#define VMK_FORM_MAX 2u
#define VMK_DIM VOLCOMP_MASK_DIM
#define VMK_VOXELS VOLCOMP_MASK_VOXELS
#define VMK_RAW_BYTES (VMK_VOXELS / 8u)
/* mode 5 codes the chunk itself: the same model over a 128^3 grid */
#define VMK_LL_DIM VOLCOMP_CHUNK_DIM
#define VMK_LL_VOXELS VOLCOMP_CHUNK_VOXELS
#define VMK_LL_RAW_BYTES (VMK_LL_VOXELS / 8u)
#define VMK_MAX_DIM VMK_LL_DIM
#define VMK_NCTX 4096u
#define VMK_PROB_BITS 12u
#define VMK_PROB_ONE (1u << VMK_PROB_BITS)
#define VMK_PROB_INIT (VMK_PROB_ONE >> 1)
#define VMK_TOP (1u << 24)
/* Adaptation rate per context, by how many bits that context has already seen:
 * p moves by 1/2, 1/2, 1/4, 1/4 then 1/8 of the way. A context here sees ~64
 * bits per chunk at mode 4, so LZMA's fixed 1/32 never gets there — measured on
 * the recto fixture, this schedule is 0.00564 bits per full-res voxel against
 * 0.00734 for a fixed 1/32 and 0.0050 for an ideal two-pass model over the same
 * contexts. VMK_NEXT saturates the age without a branch. */
#define VMK_RATES 5u
static const uint8_t VMK_SHIFT[VMK_RATES] = {1, 1, 2, 2, 3};
static const uint8_t VMK_NEXT[VMK_RATES] = {1, 2, 3, 4, 4};
#define VMK_HDR_BYTES (VF_HDR_BYTES + 1u) /* the 8-byte chunk header + the flags byte */

static const uint8_t VMK_LUT[9] = {0, 32, 64, 96, 128, 159, 191, 223, 255};

/* one adaptive binary probability (of a zero bit) and its age, per context.
 * They stay in two arrays rather than one packed word: the probability is on the
 * coder's critical path (load -> multiply -> compare -> next context) and the age
 * is not, so splitting them keeps the unpacking off that path. */
typedef struct vmk_model {
  uint16_t p[VMK_NCTX];
  uint8_t age[VMK_NCTX];
} vmk_model;

static inline void vmk_model_init(vmk_model *m) {
  for (uint32_t i = 0; i < VMK_NCTX; i++) m->p[i] = VMK_PROB_INIT;
  memset(m->age, 0, sizeof m->age);
}
static inline uint32_t vmk_shift(vmk_model *m, uint32_t c) {
  uint32_t k = m->age[c];
  m->age[c] = VMK_NEXT[k];
  return VMK_SHIFT[k];
}

/* ---- range coder ---- */
typedef struct vmk_rc_enc {
  uint64_t low;
  uint32_t range;
  uint8_t cache;
  uint64_t cache_size;
  uint8_t *out;
  size_t cap, n;
  bool full; /* the output reached cap: the caller falls back to the raw form */
} vmk_rc_enc;

static inline void vmk_put(vmk_rc_enc *e, uint8_t b) {
  if (e->n < e->cap)
    e->out[e->n] = b;
  else
    e->full = true;
  e->n++;
}
static inline void vmk_rc_enc_init(vmk_rc_enc *e, uint8_t *out, size_t cap) {
  e->low = 0;
  e->range = 0xFFFFFFFFu;
  e->cache = 0;
  e->cache_size = 1; /* the first byte out is this 0; the decoder checks and skips it */
  e->out = out;
  e->cap = cap;
  e->n = 0;
  e->full = false;
}
static void vmk_shift_low(vmk_rc_enc *e) {
  if (e->low < 0xFF000000u || e->low > 0xFFFFFFFFu) {
    uint8_t carry = (uint8_t)(e->low >> 32);
    vmk_put(e, (uint8_t)(e->cache + carry));
    while (--e->cache_size) vmk_put(e, (uint8_t)(0xFFu + carry));
    e->cache = (uint8_t)(e->low >> 24);
  }
  e->cache_size++;
  e->low = (e->low << 8) & 0xFFFFFFFFu;
}
__attribute__((always_inline)) static inline void vmk_enc_bit(vmk_rc_enc *e, vmk_model *m, uint32_t c,
                                                              uint32_t bit) {
  uint16_t *sp = &m->p[c];
  uint32_t pr = *sp, sh = vmk_shift(m, c);
  uint32_t bound = (e->range >> VMK_PROB_BITS) * pr;
  if (!bit) {
    e->range = bound;
    pr += (VMK_PROB_ONE - pr) >> sh;
  } else {
    e->low += bound;
    e->range -= bound;
    pr -= pr >> sh;
  }
  *sp = (uint16_t)pr;
  while (e->range < VMK_TOP) {
    e->range <<= 8;
    vmk_shift_low(e);
  }
}
/* Five shifts push the whole low register out. The last of them always flushes
 * (low is 0 by then), so exactly one byte is left in the cache and it is never
 * emitted: the payload is then exactly as long as the decoder's reads (five to
 * prime it, one per renormalisation), and truncation or trailing garbage is
 * always detected. */
static void vmk_rc_enc_finish(vmk_rc_enc *e) {
  for (int i = 0; i < 5; i++) vmk_shift_low(e);
}

typedef struct vmk_rc_dec {
  uint32_t range, code;
  const uint8_t *in;
  size_t n, pos;
  bool over; /* a read past the end of the payload */
} vmk_rc_dec;

static inline uint8_t vmk_get(vmk_rc_dec *d) {
  if (d->pos >= d->n) {
    d->over = true;
    return 0;
  }
  return d->in[d->pos++];
}
static inline bool vmk_rc_dec_init(vmk_rc_dec *d, const uint8_t *in, size_t n) {
  d->in = in;
  d->n = n;
  d->pos = 0;
  d->over = false;
  d->range = 0xFFFFFFFFu;
  d->code = 0;
  if (n < 5 || in[0] != 0) return false; /* the encoder's first byte is always 0 */
  d->pos = 1;
  for (int i = 0; i < 4; i++) d->code = d->code << 8 | vmk_get(d);
  return true;
}
__attribute__((always_inline)) static inline uint32_t vmk_dec_bit(vmk_rc_dec *d, vmk_model *m, uint32_t c) {
  uint16_t *sp = &m->p[c];
  uint32_t pr = *sp, sh = vmk_shift(m, c);
  uint32_t bound = (d->range >> VMK_PROB_BITS) * pr, bit;
  if (d->code < bound) {
    d->range = bound;
    pr += (VMK_PROB_ONE - pr) >> sh;
    bit = 0;
  } else {
    d->code -= bound;
    d->range -= bound;
    pr -= pr >> sh;
    bit = 1;
  }
  *sp = (uint16_t)pr;
  while (d->range < VMK_TOP) {
    d->range <<= 8;
    d->code = d->code << 8 | vmk_get(d);
  }
  return bit;
}

/* ---- contexts ----
 * row (z,y) of the stored grid, or a zero row when it is outside it */
static inline const uint8_t *vmk_row(const uint8_t *g, const uint8_t *zero, int32_t z, int32_t y,
                                     uint32_t dim) {
  if (z < 0 || y < 0 || z >= (int32_t)dim || y >= (int32_t)dim) return zero;
  return g + ((size_t)z * dim + (uint32_t)y) * dim;
}
/* The ten context bits of row (z,y) that come from earlier rows. `ap` and `bp`
 * are scratch of dim + 2 bytes holding rows (z,y-1) and (z-1,y) with a zero
 * at each end, so the +-1 x offsets need no edge test. */
__attribute__((always_inline)) static inline void vmk_pre_row(const uint8_t *g, const uint8_t *zero,
                                                             uint32_t z, uint32_t y, uint8_t *ap,
                                                             uint8_t *bp, uint16_t *pre, uint32_t dim) {
  const uint8_t *A = vmk_row(g, zero, (int32_t)z, (int32_t)y - 1, dim);
  const uint8_t *B = vmk_row(g, zero, (int32_t)z - 1, (int32_t)y, dim);
  const uint8_t *C = vmk_row(g, zero, (int32_t)z - 1, (int32_t)y - 1, dim);
  const uint8_t *D = vmk_row(g, zero, (int32_t)z, (int32_t)y - 2, dim);
  const uint8_t *E = vmk_row(g, zero, (int32_t)z - 2, (int32_t)y, dim);
  const uint8_t *F = vmk_row(g, zero, (int32_t)z - 1, (int32_t)y + 1, dim);
  ap[0] = bp[0] = ap[dim + 1] = bp[dim + 1] = 0;
  memcpy(ap + 1, A, dim);
  memcpy(bp + 1, B, dim);
  for (uint32_t x = 0; x < dim; x++)
    pre[x] = (uint16_t)((uint32_t)ap[x + 1] << 1 | (uint32_t)bp[x + 1] << 2 | (uint32_t)ap[x] << 3 |
                        (uint32_t)bp[x] << 4 | (uint32_t)C[x] << 5 | (uint32_t)D[x] << 7 |
                        (uint32_t)E[x] << 8 | (uint32_t)ap[x + 2] << 9 | (uint32_t)bp[x + 2] << 10 |
                        (uint32_t)F[x] << 11);
}

/* ---- the 2x2x2 majority pool ---- */
static void vmk_pool(const uint8_t *restrict src, uint8_t *restrict grid) {
  const uint32_t C = VOLCOMP_CHUNK_DIM;
  for (uint32_t z = 0; z < VMK_DIM; z++)
    for (uint32_t y = 0; y < VMK_DIM; y++) {
      const uint8_t *s00 = src + ((size_t)(2 * z) * C + 2 * y) * C, *s01 = s00 + C;
      const uint8_t *s10 = s00 + (size_t)C * C, *s11 = s10 + C;
      uint8_t *d = grid + ((size_t)z * VMK_DIM + y) * VMK_DIM;
      for (uint32_t x = 0; x < VMK_DIM; x++) {
        uint32_t c = (s00[2 * x] != 0) + (s00[2 * x + 1] != 0) + (s01[2 * x] != 0) + (s01[2 * x + 1] != 0) +
                     (s10[2 * x] != 0) + (s10[2 * x + 1] != 0) + (s11[2 * x] != 0) + (s11[2 * x + 1] != 0);
        d[x] = c >= 4u; /* count * 2 >= 8 */
      }
    }
}

/* ---- the interpolating upsample ----
 * `plane` (128*128, values 0..4) is the x and y expansion of one stored plane:
 * output row jy is the sum of stored rows jy/2 and (jy+1)/2 clamped, each x
 * expanded the same way, so every value is twice a half-sum. */
static void vmk_expand_plane(const uint8_t *restrict gp, uint8_t *restrict rx, uint8_t *restrict plane) {
  const uint32_t C = VOLCOMP_CHUNK_DIM;
  for (uint32_t iy = 0; iy < VMK_DIM; iy++) {
    const uint8_t *g = gp + (size_t)iy * VMK_DIM;
    uint8_t *r = rx + (size_t)iy * C;
    for (uint32_t i = 0; i < VMK_DIM; i++) {
      uint32_t hi = i + 1 < VMK_DIM ? i + 1 : i;
      r[2 * i] = (uint8_t)(2u * g[i]);
      r[2 * i + 1] = (uint8_t)(g[i] + g[hi]);
    }
  }
  for (uint32_t jy = 0; jy < C; jy++) {
    uint32_t i0 = jy >> 1, i1 = (jy & 1u) && i0 + 1 < VMK_DIM ? i0 + 1 : i0;
    const uint8_t *a = rx + (size_t)i0 * C, *b = rx + (size_t)i1 * C;
    uint8_t *o = plane + (size_t)jy * C;
    for (uint32_t x = 0; x < C; x++) o[x] = (uint8_t)(a[x] + b[x]);
  }
}
/* scratch for the upsample: two expanded planes plus the row buffer */
#define VMK_UP_SCRATCH ((size_t)2u * VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM + (size_t)VMK_DIM * VOLCOMP_CHUNK_DIM)

/* Output planes [jz0, jz1) of the interpolated 128^3 chunk, written to
 * dst + (jz - jz0) * 128 * 128. */
static void vmk_interp(const uint8_t *restrict grid, uint32_t jz0, uint32_t jz1, uint8_t *restrict dst,
                       uint8_t *restrict scratch) {
  const uint32_t C = VOLCOMP_CHUNK_DIM;
  const size_t pl = (size_t)C * C;
  uint8_t *pa = scratch, *pb = scratch + pl, *rx = scratch + 2 * pl;
  uint32_t have_a = UINT32_MAX, have_b = UINT32_MAX;
  for (uint32_t jz = jz0; jz < jz1; jz++) {
    uint32_t i0 = jz >> 1, i1 = (jz & 1u) && i0 + 1 < VMK_DIM ? i0 + 1 : i0;
    if (have_a != i0) {
      if (have_b == i0) { /* the previous row's upper plane is this row's lower one */
        uint8_t *t = pa;
        pa = pb;
        pb = t;
        have_a = have_b;
        have_b = UINT32_MAX;
      } else {
        vmk_expand_plane(grid + (size_t)i0 * VMK_DIM * VMK_DIM, rx, pa);
        have_a = i0;
      }
    }
    const uint8_t *b = pa;
    if (i1 != i0) {
      if (have_b != i1) {
        vmk_expand_plane(grid + (size_t)i1 * VMK_DIM * VMK_DIM, rx, pb);
        have_b = i1;
      }
      b = pb;
    }
    uint8_t *o = dst + (size_t)(jz - jz0) * pl;
    for (size_t i = 0; i < pl; i++) o[i] = VMK_LUT[pa[i] + b[i]];
  }
}

/* ---- the coded form ---- */
__attribute__((always_inline)) static inline void vmk_code_grid_dim(uint8_t *restrict grid, vmk_rc_enc *e,
                                                                    vmk_rc_dec *d, vmk_model *restrict m,
                                                                    uint32_t dim) {
  uint8_t zero[VMK_MAX_DIM] = {0}, ap[VMK_MAX_DIM + 2], bp[VMK_MAX_DIM + 2];
  uint16_t pre[VMK_MAX_DIM];
  vmk_model_init(m);
  for (uint32_t z = 0; z < dim; z++)
    for (uint32_t y = 0; y < dim; y++) {
      vmk_pre_row(grid, zero, z, y, ap, bp, pre, dim);
      uint8_t *row = grid + ((size_t)z * dim + y) * dim;
      uint32_t c1 = 0, c2 = 0;
      if (e) {
        for (uint32_t x = 0; x < dim; x++) {
          uint32_t b = row[x];
          vmk_enc_bit(e, m, pre[x] | c1 | c2 << 6, b);
          c2 = c1;
          c1 = b;
        }
        if (e->full) return;
      } else {
        for (uint32_t x = 0; x < dim; x++) {
          uint32_t b = vmk_dec_bit(d, m, pre[x] | c1 | c2 << 6);
          row[x] = (uint8_t)b;
          c2 = c1;
          c1 = b;
        }
        if (d->over) return;
      }
    }
}
/* two instantiations, so the row loops see a constant edge */
__attribute__((noinline)) static void vmk_code_grid64(uint8_t *restrict grid, vmk_rc_enc *e, vmk_rc_dec *d,
                                                      vmk_model *restrict m) {
  vmk_code_grid_dim(grid, e, d, m, VMK_DIM);
}
__attribute__((noinline)) static void vmk_code_grid128(uint8_t *restrict grid, vmk_rc_enc *e, vmk_rc_dec *d,
                                                       vmk_model *restrict m) {
  vmk_code_grid_dim(grid, e, d, m, VMK_LL_DIM);
}
static inline void vmk_code_grid(uint8_t *restrict grid, vmk_rc_enc *e, vmk_rc_dec *d,
                                 vmk_model *restrict m, uint32_t dim) {
  if (dim == VMK_DIM)
    vmk_code_grid64(grid, e, d, m);
  else
    vmk_code_grid128(grid, e, d, m);
}

/* ---- stream ---- */
typedef struct vmk_parsed {
  uint32_t form;
  uint32_t dim;      /* stored grid edge: 64 (mode 4) or 128 (mode 5) */
  size_t voxels;     /* dim^3 */
  size_t raw_bytes;  /* voxels / 8 */
  bool lossless;     /* mode 5: the grid IS the chunk */
  const uint8_t *payload;
  size_t payload_n;
  uint8_t value; /* VMK_FORM_CONST: 0 or 1 */
} vmk_parsed;

static volcomp_status vmk_parse(const void *restrict enc, size_t enc_n, vmk_parsed *p) {
  const uint8_t *in = (const uint8_t *)enc;
  uint32_t mode;
  volcomp_status st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  if (mode != VLL_MODE_MASK && mode != VLL_MODE_MASK_LL) return VOLCOMP_ERR_ARG;
  if (enc_n < VMK_HDR_BYTES) return VOLCOMP_ERR_CORRUPT;
  p->lossless = mode == VLL_MODE_MASK_LL;
  p->dim = p->lossless ? VMK_LL_DIM : VMK_DIM;
  p->voxels = p->lossless ? VMK_LL_VOXELS : VMK_VOXELS;
  p->raw_bytes = p->voxels / 8u;
  uint32_t flags = in[VF_HDR_BYTES];
  if (flags >> 2) return VOLCOMP_ERR_CORRUPT; /* reserved bits */
  p->form = flags & 3u;
  if (p->form > VMK_FORM_MAX) return VOLCOMP_ERR_CORRUPT;
  p->payload = in + VMK_HDR_BYTES;
  p->payload_n = enc_n - VMK_HDR_BYTES;
  p->value = 0;
  if (p->form == VMK_FORM_CONST) {
    if (p->payload_n != 1 || p->payload[0] > 1) return VOLCOMP_ERR_CORRUPT;
    p->value = p->payload[0];
  } else if (p->form == VMK_FORM_RAW) {
    if (p->payload_n != p->raw_bytes) return VOLCOMP_ERR_CORRUPT;
  } else if (p->payload_n < 5) {
    return VOLCOMP_ERR_CORRUPT;
  }
  return VOLCOMP_OK;
}

/* the stored grid as 0/1 bytes, into `grid` (p->voxels bytes) */
static volcomp_status vmk_decode_grid(const vmk_parsed *p, uint8_t *restrict grid, vmk_model *restrict m) {
  if (p->form == VMK_FORM_CONST) {
    memset(grid, p->value, p->voxels);
    return VOLCOMP_OK;
  }
  if (p->form == VMK_FORM_RAW) {
    for (size_t i = 0; i < p->voxels; i++) grid[i] = p->payload[i >> 3] >> (i & 7u) & 1u;
    return VOLCOMP_OK;
  }
  vmk_rc_dec d;
  if (!vmk_rc_dec_init(&d, p->payload, p->payload_n)) return VOLCOMP_ERR_CORRUPT;
  vmk_code_grid(grid, NULL, &d, m, p->dim);
  if (d.over || d.pos != d.n) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

/* Emit a 0/1 grid of `dim`^3 cells under `mode`: CONST if it is uniform, else the
 * coded form when it beats the raw one, else RAW. `grid` is scratch the coder
 * reads in place. */
static volcomp_status vmk_emit(uint8_t *restrict grid, uint32_t dim, uint32_t mode, uint8_t *restrict dst,
                               size_t dst_cap, size_t *out_n, vmk_model *model) {
  const size_t voxels = (size_t)dim * dim * dim, raw_bytes = voxels / 8u;
  size_t i = 1;
  while (i < voxels && grid[i] == grid[0]) i++;
  if (i == voxels) { /* a uniform grid is one byte */
    if (dst_cap < VMK_HDR_BYTES + 1u) return VOLCOMP_ERR_SHORT_BUF;
    vll_hdr(dst, mode);
    dst[VF_HDR_BYTES] = (uint8_t)VMK_FORM_CONST;
    dst[VMK_HDR_BYTES] = grid[0];
    *out_n = VMK_HDR_BYTES + 1u;
    return VOLCOMP_OK;
  }
  if (dst_cap > VMK_HDR_BYTES) {
    /* code it, but never past the raw form's size: whichever is smaller wins */
    size_t cap = dst_cap - VMK_HDR_BYTES;
    if (cap > raw_bytes) cap = raw_bytes;
    vmk_rc_enc e;
    vmk_rc_enc_init(&e, dst + VMK_HDR_BYTES, cap);
    vmk_code_grid(grid, &e, NULL, model, dim);
    if (!e.full) {
      vmk_rc_enc_finish(&e);
      if (!e.full) {
        vll_hdr(dst, mode);
        dst[VF_HDR_BYTES] = (uint8_t)VMK_FORM_CODED;
        *out_n = VMK_HDR_BYTES + e.n;
        return VOLCOMP_OK;
      }
    }
  }
  if (dst_cap < VMK_HDR_BYTES + raw_bytes) return VOLCOMP_ERR_SHORT_BUF;
  vll_hdr(dst, mode);
  dst[VF_HDR_BYTES] = (uint8_t)VMK_FORM_RAW;
  memset(dst + VMK_HDR_BYTES, 0, raw_bytes);
  for (size_t k = 0; k < voxels; k++)
    if (grid[k]) dst[VMK_HDR_BYTES + (k >> 3)] |= (uint8_t)(1u << (k & 7u));
  *out_n = VMK_HDR_BYTES + raw_bytes;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_mask_encode(const uint8_t *restrict src_zyx, void *restrict dst_v,
                                                 size_t dst_cap, size_t *out_n) {
  if (!src_zyx || !dst_v || !out_n) return VOLCOMP_ERR_ARG;
  *out_n = 0;
  uint8_t *grid = (uint8_t *)VOLCOMP_MALLOC(VMK_VOXELS + sizeof(vmk_model));
  if (!grid) return VOLCOMP_ERR_NOMEM;
  vmk_model *model = (vmk_model *)(void *)(grid + VMK_VOXELS);
  vmk_pool(src_zyx, grid);
  volcomp_status st = vmk_emit(grid, VMK_DIM, VLL_MODE_MASK, (uint8_t *)dst_v, dst_cap, out_n, model);
  VOLCOMP_FREE(grid);
  return st;
}

static inline volcomp_status volcomp_mask_encode_lossless(const uint8_t *restrict src_zyx,
                                                          void *restrict dst_v, size_t dst_cap,
                                                          size_t *out_n) {
  if (!src_zyx || !dst_v || !out_n) return VOLCOMP_ERR_ARG;
  *out_n = 0;
  uint8_t *grid = (uint8_t *)VOLCOMP_MALLOC(VMK_LL_VOXELS + sizeof(vmk_model));
  if (!grid) return VOLCOMP_ERR_NOMEM;
  vmk_model *model = (vmk_model *)(void *)(grid + VMK_LL_VOXELS);
  for (size_t i = 0; i < VMK_LL_VOXELS; i++) grid[i] = src_zyx[i] != 0;
  volcomp_status st = vmk_emit(grid, VMK_LL_DIM, VLL_MODE_MASK_LL, (uint8_t *)dst_v, dst_cap, out_n, model);
  VOLCOMP_FREE(grid);
  return st;
}

static inline volcomp_status volcomp_mask_info(const void *restrict enc, size_t enc_n, uint32_t *out_dim) {
  if (!enc) return VOLCOMP_ERR_ARG;
  vmk_parsed p;
  volcomp_status st = vmk_parse(enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  if (out_dim) *out_dim = p.dim;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_mask_decode_stored(const void *restrict enc, size_t enc_n,
                                                        uint8_t *restrict dst, size_t dst_cap) {
  if (!enc || !dst) return VOLCOMP_ERR_ARG;
  vmk_parsed p;
  volcomp_status st = vmk_parse(enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  if (dst_cap < p.voxels) return VOLCOMP_ERR_SHORT_BUF;
  vmk_model *m = (vmk_model *)VOLCOMP_MALLOC(sizeof(vmk_model));
  if (!m) return VOLCOMP_ERR_NOMEM;
  st = vmk_decode_grid(&p, dst, m);
  VOLCOMP_FREE(m);
  if (st != VOLCOMP_OK) return st;
  for (size_t i = 0; i < p.voxels; i++) dst[i] = dst[i] ? 255u : 0u;
  return VOLCOMP_OK;
}

/* Decode a mask chunk. jz0/jz1 select the output planes: the whole chunk
 * (0, 128) for volcomp_decode, one block's 16 for volcomp_decode_block. `dst`
 * holds jz1 - jz0 planes of 128*128. Mode 4 interpolates the 64^3 grid; mode 5
 * stores those planes already and only maps 0/1 to 0/255. */
static volcomp_status vmk_decode_planes(const void *restrict enc, size_t enc_n, uint32_t jz0, uint32_t jz1,
                                        uint8_t *restrict dst) {
  vmk_parsed p;
  volcomp_status st = vmk_parse(enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  const size_t scratch_n = p.lossless ? 0u : VMK_UP_SCRATCH;
  uint8_t *mem = (uint8_t *)VOLCOMP_MALLOC(p.voxels + scratch_n + sizeof(vmk_model));
  if (!mem) return VOLCOMP_ERR_NOMEM;
  uint8_t *grid = mem, *scratch = mem + p.voxels;
  vmk_model *m = (vmk_model *)(void *)(scratch + scratch_n);
  st = vmk_decode_grid(&p, grid, m);
  if (st == VOLCOMP_OK) {
    if (p.lossless) {
      const size_t pl = (size_t)VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM;
      const uint8_t *g = grid + (size_t)jz0 * pl;
      size_t n = (size_t)(jz1 - jz0) * pl;
      for (size_t i = 0; i < n; i++) dst[i] = g[i] ? 255u : 0u;
    } else {
      vmk_interp(grid, jz0, jz1, dst, scratch);
    }
  }
  VOLCOMP_FREE(mem);
  return st;
}

/* flat: every AC step is q (the surface chunk's base, §13.1) instead of q (1+r)^0.65 */
static volcomp_status vf_encode_lossy(const uint8_t *restrict src_zyx, float q, bool flat, uint8_t *restrict dst,
                                      size_t dst_cap, size_t *out_n) {
  const uint32_t qraw = vf_q_raw(q);
  const float qe = (float)qraw * (1.0f / 256.0f); /* exactly what the decoder sees */

  const size_t sz_toks = (size_t)VF_BLOCKS * 8192u * sizeof(uint16_t);
  const size_t sz_byp = (size_t)VF_SUB_BYP_MAX * VF_NSUB;
  const size_t sz_stage = VF_SUB_TOK_MAX;
  const size_t sz_rstep = VF_BLKV * sizeof(float);
  const size_t sz_es = sizeof(vf_etab) * VF_NMODELS;
  const size_t sz_fields = ((size_t)VF_BLOCKS_PER_SUB * 8192u + 2u) * sizeof(uint32_t);
  uint8_t *mem = (uint8_t *)VOLCOMP_MALLOC(sz_toks + sz_byp + sz_stage + sz_rstep + sz_es + sz_fields);
  if (!mem) return VOLCOMP_ERR_NOMEM;
  vf_enc e;
  memset(e.counts, 0, sizeof e.counts);
  e.toks = (uint16_t *)(void *)mem;
  e.byp = mem + sz_toks;
  e.stage = e.byp + sz_byp;
  e.rstep = (float *)(void *)(e.stage + sz_stage);
  vf_etab *es = (vf_etab *)(void *)(e.stage + sz_stage + sz_rstep);
  uint32_t *fields = (uint32_t *)(void *)(e.stage + sz_stage + sz_rstep + sz_es);
  {
    float step[VF_NRADII], inv[VF_NRADII];
    if (flat)
      vf_steps_flat(qe, step);
    else
      vf_steps(qe, step);
    for (uint32_t r = 0; r < VF_NRADII; r++) inv[r] = 1.0f / step[r];
    for (uint32_t c = 0; c < VF_BLKV; c++) e.rstep[c] = inv[vf_radius(c)] * vf_dct_scale(c);
  }
  volcomp_status st = VOLCOMP_ERR_CORRUPT; /* internal invariant failure */

  /* phase A: transform, quantise, tokenise, substream by substream */
  uint16_t *tp = e.toks;
  size_t byp_pos = 0;
  for (uint32_t s = 0; s < VF_NSUB; s++) {
    e.tok_off[s] = (size_t)(tp - e.toks);
    e.byp_off[s] = byp_pos;
    vf_bitw bw;
    vf_bw_init(&bw, e.byp + byp_pos, VF_SUB_BYP_MAX);
    int64_t prev_dc = 0;
    for (uint32_t bi = s * VF_BLOCKS_PER_SUB; bi < (s + 1) * VF_BLOCKS_PER_SUB; bi++) {
      int32_t lvl[VF_BLKV + 1];
      uint16_t npos[VF_BLKV];
      uint32_t nnz = vf_analyse_block(src_zyx, bi >> 6, (bi >> 3) & 7u, bi & 7u, e.rstep, lvl, npos);
      if (!vf_tokenize_block(&e, &tp, &bw, lvl, npos, nnz, &prev_dc)) goto out;
    }
    size_t bn;
    if (!vf_bw_flush(&bw, &bn)) goto out;
    byp_pos += bn;
  }
  e.tok_off[VF_NSUB] = (size_t)(tp - e.toks);
  e.byp_off[VF_NSUB] = byp_pos;

  /* phase B: models, header, tables, directory; phase C: entropy code into dst */
  {
    vf_model models[VF_NMODELS];
    for (uint32_t m = 0; m < VF_NMODELS; m++) {
      bool any = false;
      for (uint32_t t = 0; t < VF_NTOK; t++) any |= e.counts[m][t] != 0;
      if (!any) e.counts[m][0] = 1; /* unused model: minimal valid table */
      if (!vf_model_build(&models[m], e.counts[m])) goto out;
    }
    vf_etabs_init(models, es, VF_NMODELS);
    uint8_t tables[VF_TABLES_MAX_BYTES];
    size_t tn = vf_tables_write(models, tables, VF_NMODELS);
    size_t pos = VF_HDR_BYTES + tn + VF_DIR_BYTES;
    if (dst_cap < pos) {
      st = VOLCOMP_ERR_SHORT_BUF;
      goto out;
    }
    dst[0] = 'V';
    dst[1] = 'O';
    dst[2] = 'L';
    dst[3] = 'C';
    dst[4] = VF_VERSION;
    dst[5] = (uint8_t)VLL_MODE_LOSSY;
    vf_wr_u16(dst + 6, qraw);
    memcpy(dst + VF_HDR_BYTES, tables, tn);
    uint8_t *dir = dst + VF_HDR_BYTES + tn;
    for (uint32_t s = 0; s < VF_NSUB; s++) {
      size_t ntok = e.tok_off[s + 1] - e.tok_off[s];
      size_t rn = vf_tans_encode2(es, e.toks + e.tok_off[s], ntok, fields, e.stage, sz_stage);
      if (rn == 0) goto out;
      size_t bn = e.byp_off[s + 1] - e.byp_off[s];
      if (dst_cap - pos < rn + bn) {
        st = VOLCOMP_ERR_SHORT_BUF;
        goto out;
      }
      memcpy(dst + pos, e.stage, rn);
      pos += rn;
      memcpy(dst + pos, e.byp + e.byp_off[s], bn);
      pos += bn;
      vf_wr_u32(dir + s * 8, (uint32_t)rn);
      vf_wr_u32(dir + s * 8 + 4, (uint32_t)bn);
    }
    *out_n = pos;
    st = VOLCOMP_OK;
  }
out:
  VOLCOMP_FREE(mem);
  return st;
}

/* ======================= surface chunks (mode 6) ==========================
 * spec/format.md §13. A surface chunk is a smooth probability field stored as
 *
 *   base        a DCT stream body (§2..§5, the layout mode 0 writes after its
 *               header) with a FLAT step law — every AC step q, DC q/8 —
 *               reconstructed BIT-EXACTLY (§13.2: vf_parse_body(strict),
 *               vf_dct16_inv_strict);
 *   refinement  one context-coded bit for every voxel whose base value lies
 *               within M/2 of the threshold: "does this voxel sit on the wrong
 *               side of thr?". A flipped voxel decodes to thr (source >= thr)
 *               or thr - 1 (source < thr).
 *
 * so that `source >= thr` equals `decoded >= thr` for EVERY voxel: the
 * thresholded mask is exact (dice 1, identical skeleton and distance field),
 * and the soft values carry the DCT's error at q. The refinement is cheap
 * because the base is already right almost everywhere and the bit's context
 * knows how close the base came (|2d - (2thr - 1)|, six classes), which of six
 * causal neighbours already ended up on the other side, how many of the three
 * forward face neighbours sit there in the base, and the base side itself —
 * 3072 contexts, the mask coder's range coder, probabilities adapting at 1/2,
 * 1/2, 1/4, 1/4, 1/8, 1/8 then 1/16 and clamped to [16, 4080] / 4096.
 *
 * The contexts read the base's reconstruction, which is why that has to be
 * bit-exact: a decoder on another build seeing one base value off by one LSB
 * would pick a different context and lose sync. */
#define VSF_HDR_BYTES 16u
#define VSF_LAW_FLAT 1u /* header byte 9: the base's step law; revision 4 defines only 1, flat */
#define VSF_DCLS 6u
#define VSF_NCTX (VSF_DCLS * 64u * 4u * 2u) /* voxel contexts; the far flags use 8 more */
#define VSF_NEAR 31u     /* candidates with dd <= 31 are always coded */
#define VSF_FAR_BLOCK 8u /* the rest only inside flagged 8^3 blocks */
#define VSF_PMIN 16u
#define VSF_PMAX (VMK_PROB_ONE - 16u)
#define VSF_AGE_MAX 6u /* shift = 1 + age / 2: 1, 1, 2, 2, 3, 3, then 4 */
#define VSF_M_MAX 509u /* |2d - (2thr - 1)| <= 2*255 - 1 */

typedef struct vsf_model {
  uint16_t p[VSF_NCTX + 8u];
  uint8_t age[VSF_NCTX + 8u];
} vsf_model;

static inline void vsf_model_init(vsf_model *m) {
  for (uint32_t i = 0; i < VSF_NCTX + 8u; i++) m->p[i] = VMK_PROB_INIT;
  memset(m->age, 0, sizeof m->age);
}
static inline uint32_t vsf_shift(vsf_model *m, uint32_t c) {
  uint32_t a = m->age[c];
  if (a < VSF_AGE_MAX) m->age[c] = (uint8_t)(a + 1u);
  return 1u + (a >> 1);
}
static inline uint32_t vsf_clamp_p(uint32_t p) { return p < VSF_PMIN ? VSF_PMIN : (p > VSF_PMAX ? VSF_PMAX : p); }
__attribute__((always_inline)) static inline void vsf_enc_bit(vmk_rc_enc *e, vsf_model *m, uint32_t c,
                                                              uint32_t bit) {
  uint32_t pr = m->p[c], sh = vsf_shift(m, c);
  uint32_t bound = (e->range >> VMK_PROB_BITS) * pr;
  if (!bit) {
    e->range = bound;
    pr += (VMK_PROB_ONE - pr) >> sh;
  } else {
    e->low += bound;
    e->range -= bound;
    pr -= pr >> sh;
  }
  m->p[c] = (uint16_t)vsf_clamp_p(pr);
  while (e->range < VMK_TOP) {
    e->range <<= 8;
    vmk_shift_low(e);
  }
}
__attribute__((always_inline)) static inline uint32_t vsf_dec_bit(vmk_rc_dec *d, vsf_model *m, uint32_t c) {
  uint32_t pr = m->p[c], sh = vsf_shift(m, c);
  uint32_t bound = (d->range >> VMK_PROB_BITS) * pr, bit;
  if (d->code < bound) {
    d->range = bound;
    pr += (VMK_PROB_ONE - pr) >> sh;
    bit = 0;
  } else {
    d->code -= bound;
    d->range -= bound;
    pr -= pr >> sh;
    bit = 1;
  }
  m->p[c] = (uint16_t)vsf_clamp_p(pr);
  while (d->range < VMK_TOP) {
    d->range <<= 8;
    d->code = d->code << 8 | vmk_get(d);
  }
  return bit;
}
static inline uint32_t vsf_dcls(uint32_t dd) {
  return dd <= 1u ? 0u : dd <= 3u ? 1u : dd <= 7u ? 2u : dd <= 15u ? 3u : dd <= 31u ? 4u : 5u;
}

/* bit x of out[x >> 6] set iff lo <= row[x] <= hi, for a 128-voxel row
 * (the candidate scan; integer, so both versions agree exactly) */
static inline void vsf_rowmask_c(const uint8_t *row, uint32_t lo, uint32_t hi, uint64_t out[2]) {
  const uint32_t w = hi - lo;
  for (uint32_t h = 0; h < 2; h++) {
    uint64_t m = 0;
    for (uint32_t x = 0; x < 64; x++) m |= (uint64_t)((uint32_t)(uint8_t)(row[h * 64 + x] - lo) <= w) << x;
    out[h] = m;
  }
}
#if VF_HAVE_AVX2
VF_TARGET_AVX2 static inline void vsf_rowmask_avx2(const uint8_t *row, uint32_t lo, uint32_t hi, uint64_t out[2]) {
  const __m256i l = _mm256_set1_epi8((char)lo), w = _mm256_set1_epi8((char)(hi - lo));
  for (uint32_t h = 0; h < 2; h++) {
    uint64_t m = 0;
    for (uint32_t k = 0; k < 2; k++) {
      __m256i d = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i *)(const void *)(row + h * 64 + k * 32)), l);
      __m256i in = _mm256_cmpeq_epi8(_mm256_min_epu8(d, w), d);
      m |= (uint64_t)(uint32_t)_mm256_movemask_epi8(in) << (32 * k);
    }
    out[h] = m;
  }
}
#endif
static inline void vsf_rowmask(const uint8_t *row, uint32_t lo, uint32_t hi, uint64_t out[2]) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) {
    vsf_rowmask_avx2(row, lo, hi, out);
    return;
  }
#endif
  vsf_rowmask_c(row, lo, hi, out);
}

/* The refinement pass over `v` (the base reconstruction, updated in place: a
 * flipped voxel becomes thr or thr - 1, so v[j] >= thr is the FINAL side of an
 * already visited voxel j and the BASE side of a later one). Encodes when e is
 * set (src gives the truth), decodes otherwise. Neighbours outside the chunk
 * are the voxel itself, i.e. agree with it.
 *
 * Candidates with dd <= VSF_NEAR are always coded. Those further out (dd up to
 * M) rarely flip but are most of the candidates, so they are coded only inside
 * the 8^3 blocks whose flag — coded first, one per block holding such a
 * candidate — says the block has a far flip: 2.3-3.3x fewer coded bits for 2-4 %
 * fewer bytes than coding every candidate (docs/surface_mode.md). */
static void vsf_refine(uint8_t *restrict v, const uint8_t *restrict src, uint32_t thr, uint32_t M,
                       vmk_rc_enc *e, vmk_rc_dec *d, vsf_model *restrict m) {
  const uint32_t C = VOLCOMP_CHUNK_DIM, T2 = 2u * thr - 1u, NB = C / VSF_FAR_BLOCK;
  const size_t PL = (size_t)C * C;
  uint8_t kind[256], dcl[256]; /* per base value: 0 not a candidate, 1 near, 2 far; distance class */
  for (uint32_t val = 0; val < 256; val++) {
    const uint32_t dd = 2u * val > T2 ? 2u * val - T2 : T2 - 2u * val;
    kind[val] = (uint8_t)(dd > M ? 0u : dd <= VSF_NEAR ? 1u : 2u);
    dcl[val] = (uint8_t)vsf_dcls(dd);
  }
  vsf_model_init(m);
  uint8_t flag[(VOLCOMP_CHUNK_DIM / VSF_FAR_BLOCK) * (VOLCOMP_CHUNK_DIM / VSF_FAR_BLOCK) *
               (VOLCOMP_CHUNK_DIM / VSF_FAR_BLOCK)];
  memset(flag, 0, sizeof flag);
  if (M > VSF_NEAR) /* the far flags, block raster order, before any voxel */
    for (uint32_t bz = 0; bz < NB; bz++)
      for (uint32_t by = 0; by < NB; by++)
        for (uint32_t bx = 0; bx < NB; bx++) {
          uint32_t any = 0, hit = 0;
          for (uint32_t z = bz * VSF_FAR_BLOCK; z < (bz + 1) * VSF_FAR_BLOCK; z++)
            for (uint32_t y = by * VSF_FAR_BLOCK; y < (by + 1) * VSF_FAR_BLOCK; y++) {
              const size_t row = ((size_t)z * C + y) * C + bx * VSF_FAR_BLOCK;
              for (uint32_t x = 0; x < VSF_FAR_BLOCK; x++)
                if (kind[v[row + x]] == 2u) {
                  any = 1;
                  if (src && (src[row + x] >= thr) != (v[row + x] >= thr)) hit = 1;
                }
            }
          if (!any) continue;
          const size_t bi = ((size_t)bz * NB + by) * NB + bx;
          const uint32_t c = VSF_NCTX + (uint32_t)(bz ? flag[bi - NB * NB] : 0) +
                             2u * (uint32_t)(by ? flag[bi - NB] : 0) + 4u * (uint32_t)(bx ? flag[bi - 1] : 0);
          if (e)
            vsf_enc_bit(e, m, c, hit);
          else
            hit = vsf_dec_bit(d, m, c);
          flag[bi] = (uint8_t)hit;
        }
  /* candidates: dd <= M is v in [lo, hi]; near ones (dd <= VSF_NEAR) v in [nlo, nhi] */
  const uint32_t Mn = M < VSF_NEAR ? M : VSF_NEAR;
  const uint32_t lo = T2 > M ? (T2 - M) / 2u : 0u, hi = (T2 + M) / 2u > 255u ? 255u : (T2 + M) / 2u;
  const uint32_t nlo = T2 > Mn ? (T2 - Mn) / 2u : 0u, nhi = (T2 + Mn) / 2u > 255u ? 255u : (T2 + Mn) / 2u;
  for (uint32_t z = 0; z < C; z++)
    for (uint32_t y = 0; y < C; y++) {
      const size_t row = ((size_t)z * C + y) * C;
      uint64_t all[2], near[2], far[2] = {0, 0};
      vsf_rowmask(v + row, lo, hi, all);
      if (!(all[0] | all[1])) continue;
      vsf_rowmask(v + row, nlo, nhi, near);
      const uint8_t *fr = flag + ((size_t)(z / VSF_FAR_BLOCK) * NB + y / VSF_FAR_BLOCK) * NB;
      for (uint32_t g = 0; g < NB; g++)
        if (fr[g]) far[g / 8u] |= (uint64_t)0xFFu << (8u * (g % 8u));
      const ptrdiff_t dz = z ? -(ptrdiff_t)PL : 0, dy = y ? -(ptrdiff_t)C : 0;
      const ptrdiff_t pz = z + 1 < C ? (ptrdiff_t)PL : 0, py = y + 1 < C ? (ptrdiff_t)C : 0;
      for (uint32_t h = 0; h < 2; h++)
        for (uint64_t bits = all[h] & (near[h] | far[h]); bits; bits &= bits - 1u) {
          const uint32_t x = h * 64u + (uint32_t)__builtin_ctzll(bits);
          const size_t i = row + x;
          const uint32_t dv = v[i];
          const uint32_t s = dv >= thr;
          const ptrdiff_t dx = x ? -1 : 0, px = x + 1 < C ? 1 : 0;
          const uint8_t *q = v + i;
          const uint32_t pat = (uint32_t)((q[dx] >= thr) != s) | (uint32_t)((q[dy] >= thr) != s) << 1 |
                               (uint32_t)((q[dz] >= thr) != s) << 2 | (uint32_t)((q[dy + dx] >= thr) != s) << 3 |
                               (uint32_t)((q[dz + dx] >= thr) != s) << 4 | (uint32_t)((q[dz + dy] >= thr) != s) << 5;
          const uint32_t b = (uint32_t)((q[px] >= thr) != s) + (uint32_t)((q[py] >= thr) != s) +
                             (uint32_t)((q[pz] >= thr) != s);
          const uint32_t c = (((uint32_t)dcl[dv] * 64u + pat) * 4u + b) * 2u + s;
          uint32_t f;
          if (e) {
            f = (uint32_t)(src[i] >= thr) != s;
            vsf_enc_bit(e, m, c, f);
          } else {
            f = vsf_dec_bit(d, m, c);
          }
          if (f) v[i] = (uint8_t)(s ? thr - 1u : thr);
        }
      if (e ? e->full : d->over) return;
    }
}

typedef struct vsf_parsed {
  uint32_t qraw, thr, margin;
  const uint8_t *base, *refine;
  size_t base_n, refine_n;
} vsf_parsed;

static volcomp_status vsf_parse(const void *restrict enc, size_t enc_n, vsf_parsed *p) {
  const uint8_t *in = (const uint8_t *)enc;
  uint32_t mode;
  volcomp_status st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  if (mode != VLL_MODE_SURFACE) return VOLCOMP_ERR_ARG;
  if (enc_n < VSF_HDR_BYTES) return VOLCOMP_ERR_CORRUPT;
  p->qraw = vf_rd_u16(in + 6);
  p->thr = in[8];
  p->margin = vf_rd_u16(in + 10);
  if (p->qraw < 2u * VF_Q_RAW_MIN || p->qraw > VF_Q_RAW_MAX) return VOLCOMP_ERR_CORRUPT;
  if (p->thr == 0 || in[9] != VSF_LAW_FLAT) return VOLCOMP_ERR_CORRUPT;
  if (p->margin > VSF_M_MAX || (p->margin && !(p->margin & 1u))) return VOLCOMP_ERR_CORRUPT;
  uint32_t bn = vf_rd_u32(in + 12);
  if (bn > enc_n - VSF_HDR_BYTES) return VOLCOMP_ERR_CORRUPT;
  p->base = in + VSF_HDR_BYTES;
  p->base_n = bn;
  p->refine = p->base + bn;
  p->refine_n = enc_n - VSF_HDR_BYTES - bn;
  /* exact accounting: a refinement payload iff a margin, and then at least the coder's five bytes */
  if (p->margin ? p->refine_n < 5u : p->refine_n != 0u) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

/* the bit-exact base, into dst (a whole chunk) */
static volcomp_status vsf_decode_base(const vsf_parsed *sp, uint8_t *restrict dst) {
  vf_parsed p;
  volcomp_status st = vf_parse_body(sp->base, sp->base + sp->base_n, sp->qraw, true, true, &p);
  if (st != VOLCOMP_OK) return st;
  for (uint32_t s = 0; s < VF_NSUB; s++) {
    st = vf_decode_sub(&p, s, dst, -1, NULL);
    if (st != VOLCOMP_OK) return st;
  }
  return VOLCOMP_OK;
}

static volcomp_status vsf_decode(const void *restrict enc, size_t enc_n, uint8_t *restrict dst) {
  vsf_parsed sp;
  volcomp_status st = vsf_parse(enc, enc_n, &sp);
  if (st != VOLCOMP_OK) return st;
  st = vsf_decode_base(&sp, dst);
  if (st != VOLCOMP_OK || !sp.margin) return st;
  vsf_model *m = (vsf_model *)VOLCOMP_MALLOC(sizeof(vsf_model));
  if (!m) return VOLCOMP_ERR_NOMEM;
  vmk_rc_dec d;
  if (!vmk_rc_dec_init(&d, sp.refine, sp.refine_n)) {
    VOLCOMP_FREE(m);
    return VOLCOMP_ERR_CORRUPT;
  }
  vsf_refine(dst, NULL, sp.thr, sp.margin, NULL, &d, m);
  VOLCOMP_FREE(m);
  return d.over || d.pos != d.n ? VOLCOMP_ERR_CORRUPT : VOLCOMP_OK;
}

static inline volcomp_status volcomp_surface_encode(const uint8_t *restrict src_zyx, float q, uint32_t thr,
                                                    void *restrict dst_v, size_t dst_cap, size_t *out_n) {
  if (!src_zyx || !dst_v || !out_n) return VOLCOMP_ERR_ARG;
  if (!(q >= VOLCOMP_SURFACE_Q_MIN && q <= VOLCOMP_Q_MAX) || thr < 1u || thr > 255u) return VOLCOMP_ERR_ARG;
  *out_n = 0;
  uint8_t *dst = (uint8_t *)dst_v;
  if (dst_cap < VSF_HDR_BYTES + VF_DIR_BYTES) return VOLCOMP_ERR_SHORT_BUF;
  /* The base is a mode-0 stream written 8 bytes in, so its body lands exactly
   * where the surface header ends; that header then replaces the mode-0 one. */
  size_t n0;
  volcomp_status st = vf_encode_lossy(src_zyx, q, true, dst + (VSF_HDR_BYTES - VF_HDR_BYTES),
                                      dst_cap - (VSF_HDR_BYTES - VF_HDR_BYTES), &n0);
  if (st != VOLCOMP_OK) return st;
  const uint32_t qraw = vf_rd_u16(dst + (VSF_HDR_BYTES - VF_HDR_BYTES) + 6);
  const size_t base_n = n0 - VF_HDR_BYTES;
  vll_hdr(dst, VLL_MODE_SURFACE);
  vf_wr_u16(dst + 6, qraw);
  dst[8] = (uint8_t)thr;
  dst[9] = VSF_LAW_FLAT;
  vf_wr_u16(dst + 10, 0);
  vf_wr_u32(dst + 12, (uint32_t)base_n);
  /* reconstruct the base exactly as a decoder will, and find the margin */
  uint8_t *v = (uint8_t *)VOLCOMP_MALLOC(VOLCOMP_CHUNK_VOXELS + sizeof(vsf_model));
  if (!v) return VOLCOMP_ERR_NOMEM;
  vsf_model *m = (vsf_model *)(void *)(v + VOLCOMP_CHUNK_VOXELS);
  vsf_parsed sp = {qraw, thr, 0, dst + VSF_HDR_BYTES, NULL, base_n, 0};
  st = vsf_decode_base(&sp, v);
  uint32_t M = 0;
  if (st == VOLCOMP_OK)
    for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++)
      if ((src_zyx[i] >= thr) != (v[i] >= thr)) {
        uint32_t dd = 2u * v[i] > 2u * thr - 1u ? 2u * v[i] - (2u * thr - 1u) : 2u * thr - 1u - 2u * v[i];
        if (dd > M) M = dd;
      }
  size_t n = VSF_HDR_BYTES + base_n;
  if (st == VOLCOMP_OK && M) {
    vmk_rc_enc e;
    vmk_rc_enc_init(&e, dst + n, dst_cap - n);
    vsf_refine(v, src_zyx, thr, M, &e, NULL, m);
    if (!e.full) vmk_rc_enc_finish(&e);
    if (e.full)
      st = VOLCOMP_ERR_SHORT_BUF;
    else {
      vf_wr_u16(dst + 10, M);
      n += e.n;
    }
  }
  VOLCOMP_FREE(v);
  if (st == VOLCOMP_OK) *out_n = n;
  return st;
}

static inline volcomp_status volcomp_surface_info(const void *restrict enc, size_t enc_n, uint32_t *out_thr,
                                                  uint32_t *out_margin) {
  if (!enc) return VOLCOMP_ERR_ARG;
  vsf_parsed sp;
  volcomp_status st = vsf_parse(enc, enc_n, &sp);
  if (st != VOLCOMP_OK) return st;
  if (out_thr) *out_thr = sp.thr;
  if (out_margin) *out_margin = sp.margin;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_encode(const uint8_t *restrict src_zyx, float q,
                                            void *restrict dst_v, size_t dst_cap, size_t *out_n) {
  if (!src_zyx || !dst_v || !out_n) return VOLCOMP_ERR_ARG;
  if (!(q == VOLCOMP_Q_LOSSLESS || (q >= VOLCOMP_Q_MIN && q <= VOLCOMP_Q_MAX))) return VOLCOMP_ERR_ARG;
  *out_n = 0;
  if (q == VOLCOMP_Q_LOSSLESS) return vll_encode(src_zyx, (uint8_t *)dst_v, dst_cap, out_n);
  return vf_encode_lossy(src_zyx, q, false, (uint8_t *)dst_v, dst_cap, out_n);
}

/* Common header check: magic, version, chunk mode, and (for the lossless modes)
 * that qraw is 0. */
static volcomp_status vll_chunk_mode(const void *restrict enc, size_t enc_n, uint32_t *mode) {
  const uint8_t *in = (const uint8_t *)enc;
  if (enc_n < VF_HDR_BYTES) return VOLCOMP_ERR_CORRUPT;
  if (in[0] != 'V' || in[1] != 'O' || in[2] != 'L' || in[3] != 'C') return VOLCOMP_ERR_CORRUPT;
  if (in[4] != VF_VERSION) return VOLCOMP_ERR_VERSION;
  if (in[5] > VLL_MODE_MAX) return VOLCOMP_ERR_CORRUPT;
  /* q_raw is the quantiser step of the lossy modes (0 and the surface chunk's
   * DCT base) and 0 for every other mode */
  if (in[5] != VLL_MODE_LOSSY && in[5] != VLL_MODE_SURFACE && vf_rd_u16(in + 6) != 0) return VOLCOMP_ERR_CORRUPT;
  *mode = in[5];
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_is_lossless(const void *restrict enc, size_t enc_n, bool *out) {
  if (!enc || !out) return VOLCOMP_ERR_ARG;
  uint32_t mode;
  volcomp_status st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  *out = mode != VLL_MODE_LOSSY && mode != VLL_MODE_MASK && mode != VLL_MODE_MASK_LL && mode != VLL_MODE_SURFACE;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_stream_q(const void *restrict enc, size_t enc_n, float *out) {
  if (!enc || !out) return VOLCOMP_ERR_ARG;
  uint32_t mode;
  volcomp_status st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  if (mode != VLL_MODE_LOSSY && mode != VLL_MODE_SURFACE) {
    *out = VOLCOMP_Q_LOSSLESS;
    return VOLCOMP_OK;
  }
  uint32_t qraw = vf_rd_u16((const uint8_t *)enc + 6);
  if (qraw < VF_Q_RAW_MIN || qraw > VF_Q_RAW_MAX) return VOLCOMP_ERR_CORRUPT;
  *out = (float)qraw * (1.0f / 256.0f);
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_decode(const void *restrict enc, size_t enc_n,
                                            uint8_t *restrict dst_zyx, size_t dst_cap) {
  if (!enc || !dst_zyx) return VOLCOMP_ERR_ARG;
  if (dst_cap < VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_SHORT_BUF;
  const uint8_t *in = (const uint8_t *)enc;
  uint32_t mode;
  volcomp_status st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  if (mode == VLL_MODE_CONST) {
    if (enc_n != VF_HDR_BYTES + 1u) return VOLCOMP_ERR_CORRUPT;
    memset(dst_zyx, in[VF_HDR_BYTES], VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  }
  if (mode == VLL_MODE_RAW) {
    if (enc_n != VF_HDR_BYTES + (size_t)VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_CORRUPT;
    memcpy(dst_zyx, in + VF_HDR_BYTES, VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  }
  if (mode == VLL_MODE_MASK || mode == VLL_MODE_MASK_LL)
    return vmk_decode_planes(enc, enc_n, 0, VOLCOMP_CHUNK_DIM, dst_zyx);
  if (mode == VLL_MODE_SURFACE) return vsf_decode(enc, enc_n, dst_zyx);
  if (mode == VLL_MODE_CODED) {
    vll_parsed lp;
    st = vll_parse(in, enc_n, &lp);
    if (st != VOLCOMP_OK) return st;
    for (uint32_t s = 0; s < VF_NSUB; s++) {
      st = vll_decode_sub(&lp, s, dst_zyx, -1, NULL);
      if (st != VOLCOMP_OK) return st;
    }
    return VOLCOMP_OK;
  }
  vf_parsed p;
  st = vf_parse(enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  for (uint32_t s = 0; s < VF_NSUB; s++) {
    st = vf_decode_sub(&p, s, dst_zyx, -1, NULL);
    if (st != VOLCOMP_OK) return st;
  }
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_decode_block(const void *restrict enc, size_t enc_n,
                                                  uint32_t bz, uint32_t by, uint32_t bx,
                                                  uint8_t *restrict dst_block, size_t dst_cap) {
  if (!enc || !dst_block) return VOLCOMP_ERR_ARG;
  if (bz >= VF_BLOCKS_AXIS || by >= VF_BLOCKS_AXIS || bx >= VF_BLOCKS_AXIS) return VOLCOMP_ERR_ARG;
  if (dst_cap < VOLCOMP_BLOCK_VOXELS) return VOLCOMP_ERR_SHORT_BUF;
  const uint8_t *in = (const uint8_t *)enc;
  uint32_t mode, bi = bz * 64u + by * 8u + bx;
  volcomp_status st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  if (mode == VLL_MODE_CONST) {
    if (enc_n != VF_HDR_BYTES + 1u) return VOLCOMP_ERR_CORRUPT;
    memset(dst_block, in[VF_HDR_BYTES], VOLCOMP_BLOCK_VOXELS);
    return VOLCOMP_OK;
  }
  if (mode == VLL_MODE_RAW) {
    if (enc_n != VF_HDR_BYTES + (size_t)VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_CORRUPT;
    for (uint32_t z = 0; z < VOLCOMP_BLOCK_DIM; z++)
      for (uint32_t y = 0; y < VOLCOMP_BLOCK_DIM; y++)
        memcpy(dst_block + (z * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_BLOCK_DIM,
               in + VF_HDR_BYTES + vf_row_off(bz, by, bx, z, y), VOLCOMP_BLOCK_DIM);
    return VOLCOMP_OK;
  }
  if (mode == VLL_MODE_MASK || mode == VLL_MODE_MASK_LL) {
    /* a mask chunk has no substreams: the grid decodes as a whole (0.26 MB of
     * bits at mode 4, 2.1 MB at mode 5) and only this block's 16 output planes
     * are then materialised */
    const size_t pl = (size_t)VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM;
    uint8_t *planes = (uint8_t *)VOLCOMP_MALLOC(pl * VOLCOMP_BLOCK_DIM);
    if (!planes) return VOLCOMP_ERR_NOMEM;
    st = vmk_decode_planes(enc, enc_n, bz * VOLCOMP_BLOCK_DIM, (bz + 1u) * VOLCOMP_BLOCK_DIM, planes);
    if (st == VOLCOMP_OK)
      for (uint32_t z = 0; z < VOLCOMP_BLOCK_DIM; z++)
        for (uint32_t y = 0; y < VOLCOMP_BLOCK_DIM; y++)
          memcpy(dst_block + (z * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_BLOCK_DIM,
                 planes + z * pl + ((size_t)by * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_CHUNK_DIM +
                     bx * VOLCOMP_BLOCK_DIM,
                 VOLCOMP_BLOCK_DIM);
    VOLCOMP_FREE(planes);
    return st;
  }
  if (mode == VLL_MODE_SURFACE) {
    /* the refinement is one adaptive pass over the chunk: decode it whole */
    uint8_t *full = (uint8_t *)VOLCOMP_MALLOC(VOLCOMP_CHUNK_VOXELS);
    if (!full) return VOLCOMP_ERR_NOMEM;
    st = vsf_decode(enc, enc_n, full);
    if (st == VOLCOMP_OK)
      for (uint32_t z = 0; z < VOLCOMP_BLOCK_DIM; z++)
        for (uint32_t y = 0; y < VOLCOMP_BLOCK_DIM; y++)
          memcpy(dst_block + (z * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_BLOCK_DIM, full + vf_row_off(bz, by, bx, z, y),
                 VOLCOMP_BLOCK_DIM);
    VOLCOMP_FREE(full);
    return st;
  }
  if (mode == VLL_MODE_CODED) {
    vll_parsed lp;
    st = vll_parse(in, enc_n, &lp);
    if (st != VOLCOMP_OK) return st;
    return vll_decode_sub(&lp, bi / VF_BLOCKS_PER_SUB, NULL, (int32_t)bi, dst_block);
  }
  vf_parsed p;
  st = vf_parse(enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  return vf_decode_sub(&p, bi / VF_BLOCKS_PER_SUB, NULL, (int32_t)bi, dst_block);
}


/* ---- optional decode-side deblocking by projection (not part of the format) ----
 * volcomp_decode_smooth: decode the chunk, blur (Gaussian, sigma) the voxels
 * within two of every block face, then project each 16^3 block back onto the
 * quantisation cells its coefficients were decoded from (one POCS step): the
 * result is the smoothest-looking chunk the stream is still consistent with.
 * Works inside one chunk (chunk faces are smoothed from their own side only),
 * so it needs nothing but the chunk's stream. docs/deblocking.md has the numbers. */

/* the signed level of every coefficient of the 16 blocks of substream s, natural order */
static volcomp_status vf_decode_levels_sub(const vf_parsed *restrict p, uint32_t s, int32_t *restrict lv) {
  const uint8_t *rb = p->payload + p->off[s];
  vf_rdec rd;
  if (!vf_rdec_init(&rd, rb, p->tok_n[s])) return VOLCOMP_ERR_CORRUPT;
  vf_bitr br;
  vf_br_init(&br, rb + p->tok_n[s], p->byp_n[s]);
  int64_t prev_dc = 0;
  for (uint32_t k = 0; k < VF_BLOCKS_PER_SUB; k++) {
    int32_t *b = lv + (size_t)k * VF_BLKV;
    memset(b, 0, VF_BLKV * sizeof *b);
    int t = vf_rdec_get(&rd, &p->models[VF_DC_CTX]);
    if (t < 0 || (uint32_t)t > VF_TOKMAX_DC) return VOLCOMP_ERR_CORRUPT;
    uint32_t u;
    if (!vf_hyb_read(&br, (uint32_t)t, &u)) return VOLCOMP_ERR_CORRUPT;
    int64_t dc = prev_dc + vf_unzigzag(u);
    if (dc < -VF_DC_ABS_MAX || dc > VF_DC_ABS_MAX) return VOLCOMP_ERR_CORRUPT;
    prev_dc = dc;
    b[0] = (int32_t)dc;
    uint32_t pos = 1;
    for (;;) {
      t = vf_rdec_get(&rd, &p->models[vf_run_ctx(pos)]);
      if (t < 0) return VOLCOMP_ERR_CORRUPT;
      if ((uint32_t)t == VF_TOK_EOB) break;
      if ((uint32_t)t > VF_TOKMAX_RUN) return VOLCOMP_ERR_CORRUPT;
      uint32_t run;
      if (!vf_hyb_read(&br, (uint32_t)t, &run)) return VOLCOMP_ERR_CORRUPT;
      pos += run;
      if (pos >= VF_BLKV) return VOLCOMP_ERR_CORRUPT;
      int lt = vf_rdec_get(&rd, &p->models[vf_level_ctx(pos, run)]);
      if (lt < 0 || (uint32_t)lt > VF_TOKMAX_LVL) return VOLCOMP_ERR_CORRUPT;
      uint32_t mag1, sign;
      if (!vf_hyb_read(&br, (uint32_t)lt, &mag1) || !vf_br_get(&br, 1, &sign)) return VOLCOMP_ERR_CORRUPT;
      if (mag1 + 1u > VF_AC_MAG_MAX) return VOLCOMP_ERR_CORRUPT;
      b[VF_SCAN16[pos]] = sign ? -(int32_t)(mag1 + 1u) : (int32_t)(mag1 + 1u);
      pos++;
    }
  }
  if (!vf_rdec_finished(&rd) || !vf_br_finished(&br)) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

/* separable Gaussian blur of the whole chunk (edge-replicating), radius
 * floor(4 sigma + 0.5) <= 8, in place on f (C^3 floats). `tmp` holds C^3 + 2 C^2
 * floats. Every pass runs along contiguous x rows, so it vectorises and stays in
 * cache: x taps inside a row, y taps across rows of one plane, z taps across planes. */
static void vf_blur_chunk(float *f, float sigma, float *tmp) {
  const uint32_t C = VOLCOMP_CHUNK_DIM;
  const size_t PL = (size_t)C * C;
  int rad = (int)(4.0f * sigma + 0.5f);
  if (rad > 8) rad = 8;
  float w[17], sum = 0;
  for (int k = -rad; k <= rad; k++) sum += w[k + rad] = expf(-0.5f * (float)(k * k) / (sigma * sigma));
  for (int k = 0; k <= 2 * rad; k++) w[k] /= sum;
  float *g = tmp, *line = tmp + C * PL; /* g: the z pass's output; line: one padded row */
  for (size_t r = 0; r < PL; r++) { /* x */
    float *row = f + r * C;
    for (int i = 0; i < rad; i++) line[i] = row[0], line[rad + (int)C + i] = row[C - 1];
    memcpy(line + rad, row, C * sizeof(float));
    for (uint32_t x = 0; x < C; x++) {
      float acc = 0;
      for (int k = 0; k <= 2 * rad; k++) acc += w[k] * line[x + (uint32_t)k];
      row[x] = acc;
    }
  }
  for (uint32_t z = 0; z < C; z++) { /* y, one plane at a time through g */
    float *pl = f + z * PL, *o = g;
    memcpy(o, pl, PL * sizeof(float));
    for (uint32_t y = 0; y < C; y++) {
      float *dst = pl + (size_t)y * C;
      for (uint32_t x = 0; x < C; x++) dst[x] = 0;
      for (int k = -rad; k <= rad; k++) {
        int yy = (int)y + k;
        yy = yy < 0 ? 0 : (yy >= (int)C ? (int)C - 1 : yy);
        const float *src = o + (size_t)yy * C, wk = w[k + rad];
        for (uint32_t x = 0; x < C; x++) dst[x] += wk * src[x];
      }
    }
  }
  for (uint32_t z = 0; z < C; z++) { /* z, into g */
    float *dst = g + z * PL;
    for (size_t i = 0; i < PL; i++) dst[i] = 0;
    for (int k = -rad; k <= rad; k++) {
      int zz = (int)z + k;
      zz = zz < 0 ? 0 : (zz >= (int)C ? (int)C - 1 : zz);
      const float *src = f + (size_t)zz * PL, wk = w[k + rad];
      for (size_t i = 0; i < PL; i++) dst[i] += wk * src[i];
    }
  }
  memcpy(f, g, (size_t)C * PL * sizeof(float));
}

static inline volcomp_status volcomp_decode_smooth(const void *restrict enc, size_t enc_n, uint8_t *restrict dst_zyx,
                                                   size_t dst_cap, float sigma, unsigned flags) {
  const bool zg = (flags & VOLCOMP_DEBLOCK_ZERO_GUARD) != 0;
  if (!enc || !dst_zyx) return VOLCOMP_ERR_ARG;
  if (dst_cap < VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_SHORT_BUF;
  volcomp_status st = volcomp_decode(enc, enc_n, dst_zyx, dst_cap);
  if (st != VOLCOMP_OK || !(sigma > 0.0f)) return st;
  uint32_t mode;
  st = vll_chunk_mode(enc, enc_n, &mode);
  if (st != VOLCOMP_OK) return st;
  if (mode != VLL_MODE_LOSSY && mode != VLL_MODE_SURFACE) return VOLCOMP_OK; /* nothing to project onto */
  /* below mode-0 q 4 the seams are within the voxel noise of CT: the gated
   * filter has nothing to fix there and was measured to lose 0.00-0.03 dB */
  if ((flags & VOLCOMP_SMOOTH_GATED) && mode == VLL_MODE_LOSSY && vf_rd_u16((const uint8_t *)enc + 6) < 4u * 256u)
    return VOLCOMP_OK;
  if (sigma > 4.0f) sigma = 4.0f;
  vf_parsed p;
  uint32_t thr = 0;
  if (mode == VLL_MODE_SURFACE) {
    vsf_parsed sp;
    st = vsf_parse(enc, enc_n, &sp);
    if (st == VOLCOMP_OK) st = vf_parse_body(sp.base, sp.base + sp.base_n, sp.qraw, true, true, &p);
    thr = sp.thr;
  } else {
    st = vf_parse(enc, enc_n, &p);
  }
  if (st != VOLCOMP_OK) return st;
  const size_t NV = VOLCOMP_CHUNK_VOXELS;
  const bool gauss = !(flags & VOLCOMP_SMOOTH_GATED);
  const size_t blur_n = gauss ? NV + 2u * (size_t)VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM : 0u;
  float *f = (float *)VOLCOMP_MALLOC(NV * sizeof(float) + VF_BLOCKS_PER_SUB * VF_BLKV * sizeof(int32_t) +
                                     2u * VF_BLKV * sizeof(float) + blur_n * sizeof(float));
  if (!f) return VOLCOMP_ERR_NOMEM;
  int32_t *lv = (int32_t *)(void *)(f + NV);
  float *blk = (float *)(void *)(lv + VF_BLOCKS_PER_SUB * VF_BLKV), *vox = blk + VF_BLKV, *line = vox + VF_BLKV;
  /* line: the blur's scratch (gauss only) */
  if (flags & VOLCOMP_SMOOTH_GATED) {
    /* the smoothing step is volcomp_deblock's gated face filter (strength = its q
     * multiplier) on this chunk alone: it only touches block faces whose step is
     * small against q, so real edges and fine texture are left to the projection */
    uint8_t *tmp = (uint8_t *)VOLCOMP_MALLOC(NV);
    if (!tmp) {
      VOLCOMP_FREE(f);
      return VOLCOMP_ERR_NOMEM;
    }
    memcpy(tmp, dst_zyx, NV);
    volcomp_deblock_ex(tmp, VOLCOMP_CHUNK_DIM, VOLCOMP_CHUNK_DIM, VOLCOMP_CHUNK_DIM, p.q * sigma, flags);
    for (size_t i = 0; i < NV; i++) f[i] = (float)tmp[i];
    VOLCOMP_FREE(tmp);
  } else if (zg) { /* normalised blur over the nonzero voxels only: masked air neither spreads nor receives */
    float *wgt = (float *)VOLCOMP_MALLOC(NV * sizeof(float));
    if (!wgt) {
      VOLCOMP_FREE(f);
      return VOLCOMP_ERR_NOMEM;
    }
    for (size_t i = 0; i < NV; i++) {
      f[i] = (float)dst_zyx[i];
      wgt[i] = dst_zyx[i] ? 1.0f : 0.0f;
    }
    vf_blur_chunk(f, sigma, line);
    vf_blur_chunk(wgt, sigma, line);
    for (size_t i = 0; i < NV; i++) f[i] = dst_zyx[i] && wgt[i] > 0.0f ? f[i] / wgt[i] : (float)dst_zyx[i];
    VOLCOMP_FREE(wgt);
  } else {
    for (size_t i = 0; i < NV; i++) f[i] = (float)dst_zyx[i];
    vf_blur_chunk(f, sigma, line);
  }
  /* cell bounds (orthonormal domain, units of step): L == 0 -> |X| < 1 - dz; else |X| in [|L| - dz, |L| + 1 - dz] */
  for (uint32_t s = 0; s < VF_NSUB && st == VOLCOMP_OK; s++) {
    st = vf_decode_levels_sub(&p, s, lv);
    for (uint32_t k = 0; k < VF_BLOCKS_PER_SUB && st == VOLCOMP_OK; k++) {
      const uint32_t bi = s * VF_BLOCKS_PER_SUB + k, bz = bi >> 6, by = (bi >> 3) & 7u, bx = bi & 7u;
      const int32_t *L = lv + (size_t)k * VF_BLKV;
      /* the seam-blurred block: blurred values within two voxels of an interior
       * block face, decoded ones elsewhere (a chunk face is seen from one side
       * only, so it is left as decoded) */
#define VF_SEAM(b, u) (((u) <= 1u && (b) > 0u) || ((u) >= 14u && (b) < 7u))
      uint64_t zeros = 0;
      for (uint32_t z = 0; z < 16; z++)
        for (uint32_t y = 0; y < 16; y++) {
          const size_t off = vf_row_off(bz, by, bx, z, y);
          const bool zy = VF_SEAM(bz, z) || VF_SEAM(by, y);
          for (uint32_t x = 0; x < 16; x++) {
            const bool seam = zy || VF_SEAM(bx, x);
            const uint8_t d0 = dst_zyx[off + x];
            zeros += d0 == 0;
            blk[(z * 16 + y) * 16 + x] = (seam && !(zg && !d0) ? f[off + x] : (float)d0) - 128.0f;
          }
        }
#undef VF_SEAM
      vf_dct16_fwd(blk);
      for (uint32_t c = 0; c < VF_BLKV; c++) {
        const float sc = vf_dct_scale(c), stp = p.step[vf_radius(c)];
        const float dz = c ? VF_DZ_Q : VF_DZ_Q_DC;
        float x = blk[c] * sc / stp; /* orthonormal coefficient in steps */
        const int32_t l = L[c];
        if (l == 0) {
          const float h = 1.0f - dz;
          x = x < -h ? -h : (x > h ? h : x);
        } else {
          const float a = (float)(l < 0 ? -l : l), sg = l < 0 ? -1.0f : 1.0f;
          float m = x * sg; /* magnitude on the level's side; the other side counts as 0 */
          m = m < a - dz ? a - dz : (m > a + 1.0f - dz ? a + 1.0f - dz : m);
          x = sg * m;
        }
        blk[c] = x * stp * sc; /* back to the inverse transform's input scale */
      }
      vf_dct16_inv(blk, 0xFFFFu, 0xFFFFu, vox);
      if (zg && zeros) { /* masked air stays exactly 0 */
        for (uint32_t z = 0; z < 16; z++)
          for (uint32_t y = 0; y < 16; y++) {
            const size_t off = vf_row_off(bz, by, bx, z, y);
            for (uint32_t x = 0; x < 16; x++)
              if (!dst_zyx[off + x]) vox[(z * 16 + y) * 16 + x] = -128.5f;
          }
      }
      vf_scatter_block(dst_zyx + vf_row_off(bz, by, bx, 0, 0), (size_t)VOLCOMP_CHUNK_DIM * VOLCOMP_CHUNK_DIM,
                       VOLCOMP_CHUNK_DIM, vox);
    }
  }
  if (st == VOLCOMP_OK && mode == VLL_MODE_SURFACE) {
    /* keep every voxel on the side the refinement put it: the threshold stays exact */
    uint8_t *ref = (uint8_t *)(void *)f; /* reuse: the refined chunk, re-decoded */
    st = volcomp_decode(enc, enc_n, ref, NV);
    for (size_t i = 0; st == VOLCOMP_OK && i < NV; i++) {
      if (ref[i] >= thr && dst_zyx[i] < thr) dst_zyx[i] = (uint8_t)thr;
      if (ref[i] < thr && dst_zyx[i] >= thr) dst_zyx[i] = (uint8_t)(thr - 1u);
    }
  }
  VOLCOMP_FREE(f);
  return st;
}

/* ---- optional post-decode deblock (c5d's normative face filter, generalised
 * to every 16-plane of an assembled volume) ----
 * For voxel pairs p1 p0 | q0 q1 straddling a plane, with c = clamp(0.8q+1,1,24):
 * skip if |q0-p0| == 0 or >= 4c, or |p1-p0| >= c, or |q1-q0| >= c; otherwise
 * move p0 and q0 toward each other by (3|d|+4)>>3. Axis order x, y, z. */
/* one face position: p0/q0 straddle the face, P1/Q1 one step further out.
 * delta is bounded by |Q0-P0| and moves each side toward the other, so the
 * results stay within [P0, Q0] and need no clamp. */
__attribute__((always_inline)) static inline void vf_deblock_1(uint8_t *p0, uint8_t *q0, ptrdiff_t s, int c, int zg) {
  int P1 = p0[-s], P0 = *p0, Q0 = *q0, Q1 = q0[s];
  if (zg && !(P1 && P0 && Q0 && Q1)) return; /* an exact zero: masked air, never filtered into */
  int d = Q0 - P0, ad = d < 0 ? -d : d;
  if (ad == 0 || ad >= 4 * c) return;
  int dp = P1 - P0, dq = Q1 - Q0;
  if ((dp < 0 ? -dp : dp) >= c || (dq < 0 ? -dq : dq) >= c) return;
  int delta = (3 * ad + 4) >> 3;
  if (d < 0) delta = -delta;
  *p0 = (uint8_t)(P0 + delta);
  *q0 = (uint8_t)(Q0 - delta);
}
#if VF_HAVE_AVX2
/* 16 contiguous face positions at once (lines adjacent in memory) */
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void vf_deblock_16(uint8_t *p0, uint8_t *q0, ptrdiff_t s, int c, int zg) {
  const __m256i P1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(p0 - s)));
  const __m256i P0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p0));
  const __m256i Q0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)q0));
  const __m256i Q1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(q0 + s)));
  const __m256i d = _mm256_sub_epi16(Q0, P0), ad = _mm256_abs_epi16(d);
  const __m256i adp = _mm256_abs_epi16(_mm256_sub_epi16(P1, P0));
  const __m256i adq = _mm256_abs_epi16(_mm256_sub_epi16(Q1, Q0));
  const __m256i cc = _mm256_set1_epi16((short)c), c4 = _mm256_set1_epi16((short)(4 * c));
  __m256i skip = _mm256_cmpeq_epi16(ad, _mm256_setzero_si256());
  skip = _mm256_or_si256(skip, _mm256_cmpgt_epi16(ad, _mm256_sub_epi16(c4, _mm256_set1_epi16(1))));
  skip = _mm256_or_si256(skip, _mm256_cmpgt_epi16(adp, _mm256_sub_epi16(cc, _mm256_set1_epi16(1))));
  skip = _mm256_or_si256(skip, _mm256_cmpgt_epi16(adq, _mm256_sub_epi16(cc, _mm256_set1_epi16(1))));
  if (zg) {
    const __m256i z0 = _mm256_setzero_si256();
    skip = _mm256_or_si256(skip, _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi16(P1, z0), _mm256_cmpeq_epi16(P0, z0)),
                                                 _mm256_or_si256(_mm256_cmpeq_epi16(Q0, z0), _mm256_cmpeq_epi16(Q1, z0))));
  }
  __m256i delta = _mm256_srai_epi16(_mm256_add_epi16(_mm256_mullo_epi16(ad, _mm256_set1_epi16(3)), _mm256_set1_epi16(4)), 3);
  delta = _mm256_sign_epi16(delta, d); /* negate where d < 0 (d == 0 already skipped) */
  delta = _mm256_andnot_si256(skip, delta);
  __m256i np = _mm256_add_epi16(P0, delta), nq = _mm256_sub_epi16(Q0, delta);
  __m256i pk = _mm256_permute4x64_epi64(_mm256_packus_epi16(np, nq), 0xD8);
  _mm_storeu_si128((__m128i *)p0, _mm256_castsi256_si128(pk));
  _mm_storeu_si128((__m128i *)q0, _mm256_extracti128_si256(pk, 1));
}
VF_TARGET_AVX2 static void vf_deblock_axis_avx2(uint8_t *vol, size_t n_outer, size_t n_mid, size_t n_in,
                                                size_t s_outer, size_t s_mid, size_t s_in, int c, int zg) {
  /* filters planes along the "in" axis; (outer, mid) enumerate the lines */
  for (size_t f = 16; f < n_in; f += 16)
    for (size_t o = 0; o < n_outer; o++) {
      uint8_t *base = vol + o * s_outer + f * s_in;
      size_t m = 0;
      if (s_mid == 1)
        for (; m + 16 <= n_mid; m += 16)
          vf_deblock_16(base + m - s_in, base + m, (ptrdiff_t)s_in, c, zg);
      for (; m < n_mid; m++)
        vf_deblock_1(base + m * s_mid - s_in, base + m * s_mid, (ptrdiff_t)s_in, c, zg);
    }
}
#endif /* VF_HAVE_AVX2 */
static void vf_deblock_axis_c(uint8_t *vol, size_t n_outer, size_t n_mid, size_t n_in,
                              size_t s_outer, size_t s_mid, size_t s_in, int c, int zg) {
  for (size_t f = 16; f < n_in; f += 16)
    for (size_t o = 0; o < n_outer; o++) {
      uint8_t *base = vol + o * s_outer + f * s_in;
      for (size_t m = 0; m < n_mid; m++)
        vf_deblock_1(base + m * s_mid - s_in, base + m * s_mid, (ptrdiff_t)s_in, c, zg);
    }
}
static inline void vf_deblock_axis(uint8_t *vol, size_t n_outer, size_t n_mid, size_t n_in,
                                   size_t s_outer, size_t s_mid, size_t s_in, int c, int zg) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) { vf_deblock_axis_avx2(vol, n_outer, n_mid, n_in, s_outer, s_mid, s_in, c, zg); return; }
#endif
  vf_deblock_axis_c(vol, n_outer, n_mid, n_in, s_outer, s_mid, s_in, c, zg);
}
static inline void volcomp_deblock_ex(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q, unsigned flags) {
  if (!vol || nz < 2 || ny < 2 || nx < 2) return;
  const int zg = (flags & VOLCOMP_DEBLOCK_ZERO_GUARD) != 0;
  float cf = 0.8f * q + 1.0f;
  int c = (int)(cf < 1.0f ? 1.0f : (cf > 24.0f ? 24.0f : cf));
  const size_t sy = nx, sz = nx * ny;
  vf_deblock_axis(vol, nz, ny, nx, sz, sy, 1, c, zg);  /* x planes */
  vf_deblock_axis(vol, nz, nx, ny, sz, 1, sy, c, zg);  /* y planes */
  vf_deblock_axis(vol, ny, nx, nz, sy, 1, sz, c, zg);  /* z planes */
}
static inline void volcomp_deblock(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q) {
  volcomp_deblock_ex(vol, nz, ny, nx, q, VOLCOMP_DEBLOCK_ZERO_GUARD);
}

static inline const char *volcomp_status_string(volcomp_status s) {
  switch (s) {
  case VOLCOMP_OK: return "ok";
  case VOLCOMP_ERR_ARG: return "invalid argument";
  case VOLCOMP_ERR_CORRUPT: return "corrupt stream";
  case VOLCOMP_ERR_VERSION: return "unsupported format version";
  case VOLCOMP_ERR_NOMEM: return "out of memory";
  case VOLCOMP_ERR_SHORT_BUF: return "destination buffer too small";
  }
  return "unknown status";
}

#endif /* VOLCOMP_H */
