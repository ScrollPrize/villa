#include "utils/video_codec.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

#include <libde265/de265.h>
#include <x265.h>

namespace utils {

namespace {

constexpr char MAGIC[4] = {'V', 'C', '3', 'D'};
constexpr std::size_t HEADER_SIZE = 20;

auto pad2(int v) -> int { return std::max(16, (v + 1) & ~1); }

void write_le16(std::byte* dst, uint16_t v)
{
    dst[0] = static_cast<std::byte>(v & 0xFF);
    dst[1] = static_cast<std::byte>((v >> 8) & 0xFF);
}

void write_le32(std::byte* dst, uint32_t v)
{
    dst[0] = static_cast<std::byte>(v & 0xFF);
    dst[1] = static_cast<std::byte>((v >> 8) & 0xFF);
    dst[2] = static_cast<std::byte>((v >> 16) & 0xFF);
    dst[3] = static_cast<std::byte>((v >> 24) & 0xFF);
}

auto read_le16(const std::byte* src) -> uint16_t
{
    return static_cast<uint16_t>(static_cast<uint8_t>(src[0])) |
           (static_cast<uint16_t>(static_cast<uint8_t>(src[1])) << 8);
}

auto read_le32(const std::byte* src) -> uint32_t
{
    return static_cast<uint32_t>(static_cast<uint8_t>(src[0])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(src[1])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(src[2])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(src[3])) << 24);
}

void write_header(std::vector<std::byte>& output, int qp, int Z, int Y, int X)
{
    output.resize(HEADER_SIZE);
    std::memcpy(output.data(), MAGIC, 4);
    write_le16(output.data() + 4, 1);  // codec type = H265
    write_le16(output.data() + 6, static_cast<uint16_t>(qp));
    write_le32(output.data() + 8, static_cast<uint32_t>(Z));
    write_le32(output.data() + 12, static_cast<uint32_t>(Y));
    write_le32(output.data() + 16, static_cast<uint32_t>(X));
}

void fill_y_plane(
    std::vector<uint8_t>& yBuf, const uint8_t* src, int X, int Y, int padW, int padH)
{
    for (int y = 0; y < Y; ++y) {
        std::memcpy(yBuf.data() + y * padW, src + y * X, X);
        if (padW > X)
            std::memset(yBuf.data() + y * padW + X, 0, padW - X);
    }
    for (int y = Y; y < padH; ++y)
        std::memset(yBuf.data() + y * padW, 0, padW);
}

}  // namespace

auto video_encode(std::span<const std::byte> raw, const VideoCodecParams& params)
    -> std::vector<std::byte>
{
    const int Z = params.depth, Y = params.height, X = params.width;
    if (Z <= 0 || Y <= 0 || X <= 0)
        throw std::runtime_error("video_encode: invalid dimensions");
    if (raw.size() < static_cast<std::size_t>(Z) * Y * X)
        throw std::runtime_error("video_encode: input buffer too small");

    const int padW = pad2(X), padH = pad2(Y);

    x265_param* xparam = x265_param_alloc();
    if (!xparam) throw std::runtime_error("video_encode: param alloc failed");
    auto param_guard = std::unique_ptr<x265_param, void (*)(x265_param*)>(
        xparam, x265_param_free);

    x265_param_default_preset(xparam, "ultrafast", "zerolatency");
    xparam->sourceWidth = padW;
    xparam->sourceHeight = padH;
    xparam->internalCsp = X265_CSP_I400;
    xparam->fpsNum = 30;
    xparam->fpsDenom = 1;
    xparam->totalFrames = Z;
    xparam->bRepeatHeaders = 1;
    xparam->rc.rateControlMode = X265_RC_CQP;
    xparam->rc.qp = params.qp;
    xparam->bEnableWavefront = 0;
    xparam->frameNumThreads = 1;
    xparam->logLevel = X265_LOG_NONE;

    x265_encoder* enc = x265_encoder_open(xparam);
    if (!enc) throw std::runtime_error("video_encode: encoder open failed");
    auto enc_guard = std::unique_ptr<x265_encoder, void (*)(x265_encoder*)>(
        enc, x265_encoder_close);

    x265_picture* pic = x265_picture_alloc();
    if (!pic) throw std::runtime_error("video_encode: picture alloc failed");
    x265_picture_init(xparam, pic);
    auto pic_guard = std::unique_ptr<x265_picture, void (*)(x265_picture*)>(
        pic, x265_picture_free);

    std::vector<uint8_t> yBuf(padW * padH, 0);
    pic->planes[0] = yBuf.data();
    pic->planes[1] = nullptr;
    pic->planes[2] = nullptr;
    pic->stride[0] = padW;
    pic->stride[1] = 0;
    pic->stride[2] = 0;
    pic->colorSpace = X265_CSP_I400;

    std::vector<std::byte> output;
    write_header(output, params.qp, Z, Y, X);

    x265_nal* nals = nullptr;
    uint32_t nalCount = 0;

    for (int z = 0; z < Z; ++z) {
        const auto* src = reinterpret_cast<const uint8_t*>(raw.data()) + z * Y * X;
        fill_y_plane(yBuf, src, X, Y, padW, padH);
        pic->pts = z;

        int ret = x265_encoder_encode(enc, &nals, &nalCount, pic, nullptr);
        if (ret < 0) throw std::runtime_error("video_encode: encode failed");
        for (uint32_t i = 0; i < nalCount; ++i) {
            auto old = output.size();
            output.resize(old + nals[i].sizeBytes);
            std::memcpy(output.data() + old, nals[i].payload, nals[i].sizeBytes);
        }
    }

    while (true) {
        int ret = x265_encoder_encode(enc, &nals, &nalCount, nullptr, nullptr);
        if (ret <= 0) break;
        for (uint32_t i = 0; i < nalCount; ++i) {
            auto old = output.size();
            output.resize(old + nals[i].sizeBytes);
            std::memcpy(output.data() + old, nals[i].payload, nals[i].sizeBytes);
        }
    }

    return output;
}

auto video_decode(
    std::span<const std::byte> compressed,
    std::size_t out_size,
    const VideoCodecParams& /*params*/) -> std::vector<std::byte>
{
    if (compressed.size() < HEADER_SIZE)
        throw std::runtime_error("video_decode: input too small for header");
    if (std::memcmp(compressed.data(), MAGIC, 4) != 0)
        throw std::runtime_error("video_decode: invalid magic");

    int Z = static_cast<int>(read_le32(compressed.data() + 8));
    int Y = static_cast<int>(read_le32(compressed.data() + 12));
    int X = static_cast<int>(read_le32(compressed.data() + 16));

    if (out_size != static_cast<std::size_t>(Z) * Y * X)
        throw std::runtime_error("video_decode: out_size mismatch with header dimensions");

    const auto* bitstream =
        reinterpret_cast<const uint8_t*>(compressed.data() + HEADER_SIZE);
    const int bitstreamLen = static_cast<int>(compressed.size() - HEADER_SIZE);

    static thread_local de265_decoder_context* ctx = nullptr;
    if (!ctx) {
        ctx = de265_new_decoder();
        if (!ctx) throw std::runtime_error("video_decode: failed to create decoder");
        de265_set_parameter_bool(ctx, DE265_DECODER_PARAM_BOOL_SEI_CHECK_HASH, 0);
        de265_start_worker_threads(ctx, 0);
    }
    de265_reset(ctx);

    std::vector<std::byte> output(out_size, std::byte{0});
    int framesDecoded = 0;

    auto extractFrames = [&]() {
        const de265_image* img;
        while ((img = de265_get_next_picture(ctx)) != nullptr && framesDecoded < Z) {
            int stride = 0;
            const uint8_t* yPlane = de265_get_image_plane(img, 0, &stride);
            if (!yPlane) continue;
            auto* dst = reinterpret_cast<uint8_t*>(output.data()) + framesDecoded * Y * X;
            if (stride == X) {
                std::memcpy(dst, yPlane, static_cast<size_t>(Y) * X);
            } else {
                for (int y = 0; y < Y; ++y)
                    std::memcpy(dst + y * X, yPlane + y * stride, X);
            }
            ++framesDecoded;
        }
    };

    de265_push_data(ctx, bitstream, bitstreamLen, 0, nullptr);
    de265_flush_data(ctx);

    for (int iterations = 0; iterations < bitstreamLen + Z * 10; ++iterations) {
        int more = 0;
        de265_error err = de265_decode(ctx, &more);
        extractFrames();
        if (framesDecoded >= Z) break;
        if (!more && err != DE265_OK) break;
        if (err == DE265_ERROR_WAITING_FOR_INPUT_DATA) break;
    }

    return output;
}

}  // namespace utils
