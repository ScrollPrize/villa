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
    return enc && enc_n >= 8 && p[0] == 'V' && p[1] == 'O' && p[2] == 'L' && p[3] == 'C' && p[4] == 1;
}

float volcomp_lib_chunk_q(const void *enc, size_t enc_n) {
    if (!volcomp_lib_is_chunk(enc, enc_n)) return 0.0f;
    const unsigned char *p = (const unsigned char *)enc;
    return (float)((unsigned)p[6] | (unsigned)p[7] << 8) / 256.0f;
}
