#include "volcomp_lib.h"

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER) && defined(__AVX2__) && defined(__FMA__)
#define VOLCOMP_LIB_BUILT 1
#include "volcomp.h"
#else
#define VOLCOMP_LIB_BUILT 0
#endif

#if VOLCOMP_LIB_BUILT
static int vl_cpu_ok(void) {
    static int cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
    return cached;
}
#endif

int volcomp_lib_available(void) {
#if VOLCOMP_LIB_BUILT
    return vl_cpu_ok();
#else
    return 0;
#endif
}

const char *volcomp_lib_status_string(int status) {
    if (status == VOLCOMP_LIB_UNSUPPORTED)
        return "volcomp codec unavailable (not compiled for this target, or CPU lacks AVX2/FMA)";
#if VOLCOMP_LIB_BUILT
    return volcomp_status_string((volcomp_status)status);
#else
    return "volcomp codec not compiled in";
#endif
}

size_t volcomp_lib_encode_bound(void) {
#if VOLCOMP_LIB_BUILT
    return VOLCOMP_ENCODE_BOUND;
#else
    return 0;
#endif
}

int volcomp_lib_encode(const uint8_t *src_zyx, float q, void *dst, size_t dst_cap, size_t *out_n) {
#if VOLCOMP_LIB_BUILT
    if (!vl_cpu_ok()) return VOLCOMP_LIB_UNSUPPORTED;
    return (int)volcomp_encode(src_zyx, q, dst, dst_cap, out_n);
#else
    (void)src_zyx; (void)q; (void)dst; (void)dst_cap; (void)out_n;
    return VOLCOMP_LIB_UNSUPPORTED;
#endif
}

int volcomp_lib_decode(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap) {
#if VOLCOMP_LIB_BUILT
    if (!vl_cpu_ok()) return VOLCOMP_LIB_UNSUPPORTED;
    return (int)volcomp_decode(enc, enc_n, dst_zyx, dst_cap);
#else
    (void)enc; (void)enc_n; (void)dst_zyx; (void)dst_cap;
    return VOLCOMP_LIB_UNSUPPORTED;
#endif
}

int volcomp_lib_decode_block(const void *enc, size_t enc_n, uint32_t bz, uint32_t by, uint32_t bx,
                             uint8_t *dst_block, size_t dst_cap) {
#if VOLCOMP_LIB_BUILT
    if (!vl_cpu_ok()) return VOLCOMP_LIB_UNSUPPORTED;
    return (int)volcomp_decode_block(enc, enc_n, bz, by, bx, dst_block, dst_cap);
#else
    (void)enc; (void)enc_n; (void)bz; (void)by; (void)bx; (void)dst_block; (void)dst_cap;
    return VOLCOMP_LIB_UNSUPPORTED;
#endif
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
