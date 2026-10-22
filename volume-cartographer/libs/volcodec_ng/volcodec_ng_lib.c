/* volcodec_ng_lib.c: compiles volcodec_ng_lossless.h once and exports the C surface in
 * volcodec_ng_lib.h. MIT License, Copyright (c) 2026 Chase Hendrick (see LICENSE in this folder). */
#include "volcodec_ng_lib.h"

#include "volcodec_ng_lossless.h"

_Static_assert(VCN_REVISION == 2, "volcodec_ng_lib.h documents format revision 2");
_Static_assert(VOLCODEC_NG_LIB_ERR_MODE == (int)VCN_ERR_MODE, "status codes match the header");

unsigned volcodec_ng_lib_format_revision(void) { return VCN_REVISION; }

size_t volcodec_ng_lib_encode_bound(void) { return VCN_ENCODE_BOUND; }

const char *volcodec_ng_lib_status_string(int status) { return vcn_status_string((vcn_status)status); }

const char *volcodec_ng_lib_kernels(void) { return vcn_kernels(); }

int volcodec_ng_lib_encode(const uint8_t *src_zyx, void *dst, size_t dst_cap, size_t *out_n) {
    return (int)vcn_encode_lossless_chunk(src_zyx, (uint8_t *)dst, dst_cap, out_n);
}

int volcodec_ng_lib_decode(const void *enc, size_t enc_n, uint8_t *dst_zyx, size_t dst_cap) {
    return (int)vcn_decode((const uint8_t *)enc, enc_n, dst_zyx, dst_cap);
}

int volcodec_ng_lib_is_chunk(const void *enc, size_t enc_n) {
    return vcn_chunk_mode((const uint8_t *)enc, enc_n) >= 0;
}

int volcodec_ng_lib_chunk_mode(const void *enc, size_t enc_n) {
    return vcn_chunk_mode((const uint8_t *)enc, enc_n);
}
