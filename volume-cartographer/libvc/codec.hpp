#pragma once
// H.265 encode/decode for 128^3 u8 volume chunks.
// Encoding uses x265 (ultrafast, grayscale monochrome).
// Decoding uses libde265 (thread-local context, auto-reset).

#include "shard.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <x265.h>
#include <libde265/de265.h>

namespace vc {

// 20-byte header prepended to each compressed chunk.
struct ChunkHeader {
    char magic[4] = {'V','C','3','D'};
    uint16_t codec = 1;    // 1 = H.265
    uint16_t qp = 26;
    uint32_t depth = CHUNK_DIM;
    uint32_t height = CHUNK_DIM;
    uint32_t width = CHUNK_DIM;
};
static_assert(sizeof(ChunkHeader) == 20);

// Encode 128^3 u8 voxels → compressed H.265 with VC3D header.
inline std::vector<uint8_t> h265_encode(std::span<const uint8_t> raw, int qp = 26) {
    int Z = CHUNK_DIM, Y = CHUNK_DIM, X = CHUNK_DIM;
    int padW = (X + 1) & ~1;
    int padH = (Y + 1) & ~1;
    if (padW < 16) padW = 16;
    if (padH < 16) padH = 16;

    x265_param* param = x265_param_alloc();
    x265_param_default_preset(param, "ultrafast", "zerolatency");
    param->sourceWidth = padW;
    param->sourceHeight = padH;
    param->internalCsp = X265_CSP_I400;
    param->fpsNum = 30;
    param->fpsDenom = 1;
    param->totalFrames = Z;
    param->rc.rateControlMode = X265_RC_CQP;
    param->rc.qp = qp;
    param->bRepeatHeaders = 1;
    param->bEnableWavefront = 0;
    param->frameNumThreads = 1;
    param->logLevel = X265_LOG_NONE;

    x265_encoder* enc = x265_encoder_open(param);
    x265_picture* pic = x265_picture_alloc();
    x265_picture_init(param, pic);

    std::vector<uint8_t> padded(padW * padH, 0);
    std::vector<uint8_t> result;
    result.reserve(raw.size() / 4);

    // Write header
    ChunkHeader hdr;
    hdr.qp = uint16_t(qp);
    result.resize(sizeof(hdr));
    memcpy(result.data(), &hdr, sizeof(hdr));

    x265_nal* nals;
    uint32_t nalCount;

    for (int z = 0; z < Z; ++z) {
        // Copy slice into padded buffer
        std::fill(padded.begin(), padded.end(), uint8_t(0));
        for (int y = 0; y < Y; ++y)
            memcpy(padded.data() + y * padW, raw.data() + z * Y * X + y * X, X);

        pic->planes[0] = padded.data();
        pic->stride[0] = padW;
        pic->pts = z;

        x265_encoder_encode(enc, &nals, &nalCount, pic, nullptr);
        for (uint32_t i = 0; i < nalCount; ++i)
            result.insert(result.end(), nals[i].payload, nals[i].payload + nals[i].sizeBytes);
    }

    // Flush
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

// Decode compressed H.265 chunk → 128^3 u8 voxels as ChunkData.
inline ChunkData h265_decode(std::span<const uint8_t> compressed) {
    ChunkData out;
    out.shape = {CHUNK_DIM, CHUNK_DIM, CHUNK_DIM};
    out.elem_size = 1;
    out.bytes.resize(CHUNK_DIM * CHUNK_DIM * CHUNK_DIM, 0);

    if (compressed.size() < sizeof(ChunkHeader)) return out;

    ChunkHeader hdr;
    memcpy(&hdr, compressed.data(), sizeof(hdr));

    int Z = int(hdr.depth), Y = int(hdr.height), X = int(hdr.width);
    auto* bitstream = compressed.data() + sizeof(ChunkHeader);
    auto bitstreamLen = compressed.size() - sizeof(ChunkHeader);

    // Thread-local decoder context (reused across calls)
    thread_local de265_decoder_context* ctx = [] {
        auto* c = de265_new_decoder();
        de265_set_parameter_bool(c, DE265_DECODER_PARAM_BOOL_SEI_CHECK_HASH, 0);
        de265_start_worker_threads(c, 0);
        return c;
    }();

    de265_reset(ctx);
    de265_push_data(ctx, bitstream, bitstreamLen, 0, nullptr);
    de265_flush_data(ctx);

    int frames = 0;
    int maxIter = int(bitstreamLen) + Z * 10;
    for (int iter = 0; iter < maxIter && frames < Z; ++iter) {
        int more = 0;
        de265_decode(ctx, &more);

        const de265_image* img;
        while ((img = de265_get_next_picture(ctx)) != nullptr && frames < Z) {
            int stride;
            auto* plane = de265_get_image_plane(img, 0, &stride);
            auto* dst = out.bytes.data() + frames * Y * X;
            if (stride == X) {
                memcpy(dst, plane, size_t(Y) * X);
            } else {
                for (int y = 0; y < Y; ++y)
                    memcpy(dst + y * X, plane + y * stride, X);
            }
            ++frames;
        }
    }

    return out;
}

} // namespace vc
