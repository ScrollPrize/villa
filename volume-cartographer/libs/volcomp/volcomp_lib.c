#include "volcomp_lib.h"

#include "volcomp.h"

int volcomp_lib_available(void) { return 1; }

const char *volcomp_lib_kernels(void) { return volcomp_kernels(); }

const char *volcomp_lib_status_string(int status) {
    if (status == VOLCOMP_LIB_UNSUPPORTED) return "volcomp codec unavailable";
    return volcomp_status_string((volcomp_status)status);
}

size_t volcomp_lib_encode_bound(void) { return VOLCOMP_ENCODE_BOUND; }

int volcomp_lib_encode(const uint8_t *src_zyx, float q, void *dst, size_t dst_cap, size_t *out_n) {
    return (int)volcomp_encode(src_zyx, q, dst, dst_cap, out_n);
}

int volcomp_lib_decode(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap) {
    return (int)volcomp_decode(enc, enc_n, dst_zyx, dst_cap);
}

int volcomp_lib_decode_block(const void *enc, size_t enc_n, uint32_t bz, uint32_t by, uint32_t bx,
                             uint8_t *dst_block, size_t dst_cap) {
    return (int)volcomp_decode_block(enc, enc_n, bz, by, bx, dst_block, dst_cap);
}

int volcomp_lib_is_chunk(const void *enc, size_t enc_n) {
    const unsigned char *p = (const unsigned char *)enc;
    return enc && enc_n >= 8 && p[0] == 'V' && p[1] == 'O' && p[2] == 'L' && p[3] == 'C' &&
           p[4] == VOLCOMP_FORMAT_VERSION;
}

float volcomp_lib_chunk_q(const void *enc, size_t enc_n) {
    if (!volcomp_lib_is_chunk(enc, enc_n)) return 0.0f;
    float q = 0.0f;
    return volcomp_stream_q(enc, enc_n, &q) == VOLCOMP_OK ? q : 0.0f;
}

const char *volcomp_lib_version(void) { return VOLCOMP_VERSION_STRING; }

unsigned volcomp_lib_format_revision(void) { return VOLCOMP_FORMAT_REVISION; }

size_t volcomp_lib_surface_encode_bound(void) { return VOLCOMP_SURFACE_ENCODE_BOUND; }

int volcomp_lib_surface_encode(const uint8_t *src_zyx, float q, uint32_t thr, void *dst, size_t dst_cap,
                               size_t *out_n) {
    return (int)volcomp_surface_encode(src_zyx, q, thr, dst, dst_cap, out_n);
}

int volcomp_lib_surface_info(const void *enc, size_t enc_n, uint32_t *out_thr, uint32_t *out_margin) {
    return (int)volcomp_surface_info(enc, enc_n, out_thr, out_margin);
}

int volcomp_lib_decode_smooth(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap, float strength,
                              unsigned flags) {
    return (int)volcomp_decode_smooth(enc, enc_n, dst_zyx, dst_cap, strength, flags);
}

/* The lib's constants mirror the header's; fail the build if upstream moves them. */
_Static_assert(VOLCOMP_LIB_CHUNK_BYTES == VOLCOMP_CHUNK_VOXELS, "chunk size");
_Static_assert(VOLCOMP_LIB_DEBLOCK_ZERO_GUARD == VOLCOMP_DEBLOCK_ZERO_GUARD, "zero guard flag");
_Static_assert(VOLCOMP_LIB_SMOOTH_GATED == VOLCOMP_SMOOTH_GATED, "gated smoothing flag");
_Static_assert(VOLCOMP_FORMAT_REVISION == 4u, "volcomp_lib.h documents format revision 4");
