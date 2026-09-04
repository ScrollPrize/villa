/* volcomp_lib.h — linkable C surface over the single-header volcomp codec.
 *
 * volcomp.h is header-only (every function is static) and requires AVX2+FMA
 * at compile time. This wrapper compiles the header exactly once, in a TU
 * built with the right flags on x86-64, and exposes plain C entry points so
 * C++ consumers need neither the header nor the flags. On targets where the
 * codec is not built (non-x86-64, MSVC) every entry point reports
 * VOLCOMP_LIB_UNSUPPORTED; volcomp_lib_available() lets callers check.
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
    VOLCOMP_LIB_UNSUPPORTED = 100, /* not compiled in, or the CPU lacks AVX2/FMA */
};

/* 1 when the codec is compiled in and the running CPU supports it. */
int volcomp_lib_available(void);
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
/* q recorded in a chunk header (0 on error). */
float volcomp_lib_chunk_q(const void *enc, size_t enc_n);

#ifdef __cplusplus
}
#endif
#endif
