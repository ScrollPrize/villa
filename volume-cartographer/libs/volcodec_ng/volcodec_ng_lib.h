/* volcodec_ng_lib.h: linkable C surface over the single-header lossless codec in
 * volcodec_ng_lossless.h.
 *
 * The header is header-only (every function is static); volcodec_ng_lib.c compiles it exactly once
 * and exports these plain C entry points for utils/volcodec_ng_codec.hpp.
 *
 * One stream codes one 128^3 uint8 chunk (z-major), losslessly. Status codes: 0 ok, -1 bad argument,
 * -2 output too small, -3 corrupt stream, -4 out of memory, -5 a VCNG chunk in a mode this build does
 * not read, -6 a mode 6 stream version this build does not read. Every function may be called from several threads at once.
 *
 * MIT License, Copyright (c) 2026 Chase Hendrick (see LICENSE in this folder). */
#ifndef VOLCODEC_NG_LIB_H
#define VOLCODEC_NG_LIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VOLCODEC_NG_LIB_OK 0
#define VOLCODEC_NG_LIB_ERR_ARG (-1)
#define VOLCODEC_NG_LIB_ERR_CAP (-2)
#define VOLCODEC_NG_LIB_ERR_CORRUPT (-3)
#define VOLCODEC_NG_LIB_ERR_NOMEM (-4)
#define VOLCODEC_NG_LIB_ERR_MODE (-5)
#define VOLCODEC_NG_LIB_ERR_VERSION (-6)

/* the stream revision the encoder writes (the decoder reads 1 up to this) */
unsigned volcodec_ng_lib_format_revision(void);
/* the largest stream the encoder writes (it falls back to a raw chunk) */
size_t volcodec_ng_lib_encode_bound(void);
const char *volcodec_ng_lib_status_string(int status);
/* "avx2", "neon" or "c": which kernel set this process uses (all give the same bytes) */
const char *volcodec_ng_lib_kernels(void);

/* Encode one chunk losslessly. */
int volcodec_ng_lib_encode(const uint8_t *src_zyx, void *dst, size_t dst_cap, size_t *out_n);
int volcodec_ng_lib_decode(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap);

/* 1 when the buffer starts with the "VCNG" magic and a revision this decoder reads */
int volcodec_ng_lib_is_chunk(const void *enc, size_t enc_n);
/* the chunk mode in the header (2 raw, 3 constant, 6 lossless; 5, an earlier lossless mode, is read
 * too), or -1 for other data */
int volcodec_ng_lib_chunk_mode(const void *enc, size_t enc_n);

#ifdef __cplusplus
}
#endif

#endif /* VOLCODEC_NG_LIB_H */
