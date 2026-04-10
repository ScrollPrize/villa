#pragma once
// H.265 encode/decode for 1024x1024 × 128-frame slab videos.
// Encoder: x265 (offline import, quality-optimized).
// Decoder: libde265 (real-time viewing).

#include "shard.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <x265.h>
#include <libde265/de265.h>

namespace vc {

// Encode a slab: 128 frames of 1024x1024 u8 → H.265 bitstream.
// Input: raw[z * 1024*1024 + y * 1024 + x], z in [0,128).
inline std::vector<uint8_t> h265_encode_slab(std::span<const uint8_t> raw, int qp = 26) {
    int W = SHARD_DIM, H = SHARD_DIM, Z = SLAB_FRAMES;

    x265_param* param = x265_param_alloc();
    x265_param_default_preset(param, "slow", nullptr);
    param->sourceWidth = W;
    param->sourceHeight = H;
    param->internalCsp = X265_CSP_I400;
    param->fpsNum = 1;
    param->fpsDenom = 1;
    param->totalFrames = Z;
    param->rc.rateControlMode = X265_RC_CRF;
    param->rc.rfConstant = double(qp);
    param->bRepeatHeaders = 1;
    // Multi-threaded: 1024x1024 frames are big enough to benefit from threading
    param->bEnableWavefront = 1;
    param->frameNumThreads = 0;  // auto
    param->logLevel = X265_LOG_NONE;
    // IDR every 32 frames: 4 seek points per slab for z-random-access
    param->keyframeMax = 32;
    param->keyframeMin = 32;
    param->bOpenGOP = 0;

    x265_encoder* enc = x265_encoder_open(param);
    x265_picture* pic = x265_picture_alloc();
    x265_picture_init(param, pic);

    std::vector<uint8_t> result;
    result.reserve(raw.size() / 8);

    x265_nal* nals;
    uint32_t nalCount;

    for (int z = 0; z < Z; ++z) {
        pic->planes[0] = const_cast<uint8_t*>(raw.data() + size_t(z) * W * H);
        pic->stride[0] = W;
        pic->pts = z;
        x265_encoder_encode(enc, &nals, &nalCount, pic, nullptr);
        for (uint32_t i = 0; i < nalCount; ++i)
            result.insert(result.end(), nals[i].payload, nals[i].payload + nals[i].sizeBytes);
    }
    for (;;) {
        int ret = x265_encoder_encode(enc, &nals, &nalCount, nullptr, nullptr);
        if (ret <= 0) break;
        for (uint32_t i = 0; i < nalCount; ++i)
            result.insert(result.end(), nals[i].payload, nals[i].payload + nals[i].sizeBytes);
    }

    x265_picture_free(pic);
    x265_encoder_close(enc);
    x265_param_free(param);
    return result;
}

// Decode a slab: H.265 bitstream → 128 frames of 1024x1024 u8.
// Returns vector of Frame objects (each 1024*1024 bytes).
inline std::vector<Frame> h265_decode_slab(std::span<const uint8_t> compressed) {
    std::vector<Frame> frames;
    frames.reserve(SLAB_FRAMES);

    if (compressed.empty()) return frames;

    thread_local de265_decoder_context* ctx = [] {
        auto* c = de265_new_decoder();
        de265_set_parameter_bool(c, DE265_DECODER_PARAM_BOOL_SEI_CHECK_HASH, 0);
        de265_start_worker_threads(c, 0);
        return c;
    }();

    de265_reset(ctx);
    de265_push_data(ctx, compressed.data(), compressed.size(), 0, nullptr);
    de265_flush_data(ctx);

    int maxIter = int(compressed.size()) + SLAB_FRAMES * 10;
    for (int iter = 0; iter < maxIter && int(frames.size()) < SLAB_FRAMES; ++iter) {
        int more = 0;
        de265_decode(ctx, &more);

        const de265_image* img;
        while ((img = de265_get_next_picture(ctx)) != nullptr &&
               int(frames.size()) < SLAB_FRAMES) {
            int stride = 0;
            auto* plane = de265_get_image_plane(img, 0, &stride);
            Frame f;
            if (stride == SHARD_DIM) {
                memcpy(f.data(), plane, size_t(SHARD_DIM) * SHARD_DIM);
            } else {
                for (int y = 0; y < SHARD_DIM; ++y)
                    memcpy(f.data() + y * SHARD_DIM, plane + y * stride, SHARD_DIM);
            }
            frames.push_back(std::move(f));
        }
    }

    return frames;
}

} // namespace vc
