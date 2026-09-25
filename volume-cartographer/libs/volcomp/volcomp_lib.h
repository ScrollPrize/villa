/* volcomp_lib.h — linkable C surface over the single-header volcomp codec.
 *
 * volcomp.h is header-only (every function is static); this wrapper compiles
 * it exactly once and exposes plain C entry points so C++ consumers need
 * neither the header nor any compiler flags. The codec is portable: on x86-64
 * it selects its AVX2+FMA kernels at runtime when the CPU has them and falls
 * back to plain C otherwise (arm64 always uses the C kernels);
 * volcomp_lib_kernels() reports which set is active.
 *
 * Upstream: https://github.com/SuperOptimizer/volume-compressor (volcomp.h
 * vendored verbatim beside this file). */
#ifndef VOLCOMP_LIB_H
#define VOLCOMP_LIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VOLCOMP_LIB_CHUNK_DIM 128u
#define VOLCOMP_LIB_CHUNK_BYTES 2097152u
#define VOLCOMP_LIB_Q_MIN 1.0f
#define VOLCOMP_LIB_Q_MAX 255.0f

/* Status codes: 0..5 mirror volcomp_status in volcomp.h. */
enum {
    VOLCOMP_LIB_OK = 0,
    VOLCOMP_LIB_ERR_ARG = 1,
    VOLCOMP_LIB_ERR_CORRUPT = 2,
    VOLCOMP_LIB_ERR_VERSION = 3,
    VOLCOMP_LIB_ERR_NOMEM = 4,
    VOLCOMP_LIB_ERR_SHORT_BUF = 5,
    VOLCOMP_LIB_UNSUPPORTED = 100, /* kept for ABI; never returned since the codec is portable */
};

/* Always 1: the codec is available on every target (kept for callers that
 * were written against the AVX2-gated version). */
int volcomp_lib_available(void);
/* "avx2" or "c": the kernel set this process uses. */
const char *volcomp_lib_kernels(void);
const char *volcomp_lib_status_string(int status);
/* Capacity that always suffices for volcomp_lib_encode. */
size_t volcomp_lib_encode_bound(void);
/* Encode one 128^3 u8 z-major chunk at quantiser step q (1..255). */
int volcomp_lib_encode(const uint8_t *src_zyx, float q, void *dst, size_t dst_cap, size_t *out_n);
/* Decode a chunk into dst (dst_cap >= VOLCOMP_LIB_CHUNK_BYTES). */
int volcomp_lib_decode(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap);
/* Decode block (bz,by,bx), each 0..7, into a 16^3 z-major buffer (>= 4096 bytes). */
int volcomp_lib_decode_block(const void *enc, size_t enc_n, uint32_t bz, uint32_t by, uint32_t bx,
                             uint8_t *dst_block, size_t dst_cap);
/* 1 when the buffer starts with the "VOLC" magic and a supported version. */
int volcomp_lib_is_chunk(const void *enc, size_t enc_n);
/* q recorded in a chunk header: the DCT step of a lossy chunk, the base's step
 * of a surface chunk, 0 for lossless / mask chunks and on error. */
float volcomp_lib_chunk_q(const void *enc, size_t enc_n);

/* Upstream library version ("1.3.0") and format revision (4: mode 6 surface
 * chunks). Every revision keeps header version byte 1. */
const char *volcomp_lib_version(void);
unsigned volcomp_lib_format_revision(void);

/* ---- surface chunks (mode 6, upstream spec/format.md §13) ----
 * For thresholded probability fields (u8 = round(255 p)): a DCT base at step q
 * (flat step law, q in [2, 255]; surface q 48 ~ mode-0 q 16) plus a refinement
 * that makes (source >= thr) == (decoded >= thr) exact for every voxel.
 * volcomp_lib_decode / volcomp_lib_decode_block read these like any chunk. */
#define VOLCOMP_LIB_SURFACE_Q_MIN 2.0f
#define VOLCOMP_LIB_SURFACE_THR_DEFAULT 128u
/* Capacity that always suffices for volcomp_lib_surface_encode. */
size_t volcomp_lib_surface_encode_bound(void);
int volcomp_lib_surface_encode(const uint8_t *src_zyx, float q, uint32_t thr, void *dst, size_t dst_cap,
                               size_t *out_n);
/* OK iff a surface chunk: its threshold and refinement margin (0 = the base
 * alone was already exact). VOLCOMP_LIB_ERR_ARG for any other chunk. */
int volcomp_lib_surface_info(const void *enc, size_t enc_n, uint32_t *out_thr, uint32_t *out_margin);

/* ---- optional decode-side smoothing (not part of the format) ----
 * Decode, smooth voxels next to interior 16^3 block faces, then project each
 * block back onto its quantisation cells (the result is still a reconstruction
 * the stream allows; a surface chunk keeps its exact threshold). Lossy and
 * surface chunks only; any other chunk decodes exactly as volcomp_lib_decode.
 * Opt-in: nothing in VC calls it by default. strength: Gaussian sigma in voxels
 * (upstream suggests 2.0 for CT, 0.6 for probability maps), or with
 * VOLCOMP_LIB_SMOOTH_GATED the gated face filter at strength x q. */
#define VOLCOMP_LIB_DEBLOCK_ZERO_GUARD 1u /* leave exact zeros (masked air) and faces touching them */
#define VOLCOMP_LIB_SMOOTH_GATED 2u
int volcomp_lib_decode_smooth(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap, float strength,
                              unsigned flags);

#ifdef __cplusplus
}
#endif
#endif
