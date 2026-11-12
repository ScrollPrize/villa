#pragma once

#include "scratch_pool.hpp"
#include <zstd.h>
#include <climits>
#include <stdexcept>

namespace vcr {
// Lazy contexts live in the existing scratch lease. No process-global codec
// mutex, per-call context allocation, or extra maintenance thread is needed.
struct DecoderContexts {
    void* blosc=nullptr;
    ZSTD_DCtx* zstd=nullptr;
    DecoderContexts()=default;
    DecoderContexts(const DecoderContexts&)=delete;
    ~DecoderContexts();
    void* bloscContext();
    ZSTD_DCtx* zstdContext();
    int decodeBlosc(const void* src,std::size_t n,void* dst,std::size_t size);
};
} // namespace vcr
