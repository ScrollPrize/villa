#include "decoder_context.hpp"
#include <blosc2.h>

namespace vcr {
DecoderContexts::~DecoderContexts() {
    if(blosc)blosc2_free_ctx(static_cast<blosc2_context*>(blosc));
    if(zstd)ZSTD_freeDCtx(zstd);
}
void* DecoderContexts::bloscContext() {
    if(!blosc) {
        auto params=BLOSC2_DPARAMS_DEFAULTS;params.nthreads=1;
        blosc=blosc2_create_dctx(params);
        if(!blosc)throw std::bad_alloc();
    }
    return blosc;
}
ZSTD_DCtx* DecoderContexts::zstdContext() {
    if(!zstd){zstd=ZSTD_createDCtx();if(!zstd)throw std::bad_alloc();}
    return zstd;
}
int DecoderContexts::decodeBlosc(const void* src,std::size_t n,void* dst,std::size_t size) {
    if(n>INT_MAX||size>INT_MAX)throw std::runtime_error("Blosc chunk exceeds int32 range");
    return blosc2_decompress_ctx(static_cast<blosc2_context*>(bloscContext()),src,
        static_cast<int32_t>(n),dst,static_cast<int32_t>(size));
}
} // namespace vcr
