#include "utils/video_codec.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

#ifdef UTILS_HAS_H264
#include <wels/codec_api.h>
#endif

#ifdef UTILS_HAS_H265
#include <libde265/de265.h>
#include <x265.h>
#endif

#ifdef UTILS_HAS_AV1
#include <aom/aom_encoder.h>
#include <aom/aomcx.h>
#include <dav1d/dav1d.h>
#endif

extern "C" {
#include "utils/compress3d.h"
}

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

void write_header(std::vector<std::byte>& output, VideoCodecType type, int qp, int Z, int Y, int X)
{
    output.resize(HEADER_SIZE);
    std::memcpy(output.data(), MAGIC, 4);
    write_le16(output.data() + 4, static_cast<uint16_t>(type));
    write_le16(output.data() + 6, static_cast<uint16_t>(qp));
    write_le32(output.data() + 8, static_cast<uint32_t>(Z));
    write_le32(output.data() + 12, static_cast<uint32_t>(Y));
    write_le32(output.data() + 16, static_cast<uint32_t>(X));
}

struct HeaderInfo {
    VideoCodecType codec;
    int qp, Z, Y, X;
};

auto read_header(std::span<const std::byte> compressed) -> HeaderInfo
{
    if (compressed.size() < HEADER_SIZE) {
        throw std::runtime_error("video_decode: input too small for header");
    }
    if (std::memcmp(compressed.data(), MAGIC, 4) != 0) {
        throw std::runtime_error("video_decode: invalid magic");
    }
    return {
        static_cast<VideoCodecType>(read_le16(compressed.data() + 4)),
        static_cast<int>(read_le16(compressed.data() + 6)),
        static_cast<int>(read_le32(compressed.data() + 8)),
        static_cast<int>(read_le32(compressed.data() + 12)),
        static_cast<int>(read_le32(compressed.data() + 16)),
    };
}

void fill_y_plane(
    std::vector<uint8_t>& yBuf, const uint8_t* src, int X, int Y, int padW, int padH)
{
    for (int y = 0; y < Y; ++y) {
        std::memcpy(yBuf.data() + y * padW, src + y * X, X);
        if (padW > X) {
            std::memset(yBuf.data() + y * padW + X, 0, padW - X);
        }
    }
    for (int y = Y; y < padH; ++y) {
        std::memset(yBuf.data() + y * padW, 0, padW);
    }
}

// ============================================================
// H.264 (OpenH264)
// ============================================================

#ifdef UTILS_HAS_H264

auto h264_encode(std::span<const std::byte> raw, const VideoCodecParams& params)
    -> std::vector<std::byte>
{
    const int Z = params.depth, Y = params.height, X = params.width;
    const int padW = pad2(X), padH = pad2(Y);

    ISVCEncoder* encoder = nullptr;
    if (WelsCreateSVCEncoder(&encoder) != 0 || !encoder) {
        throw std::runtime_error("h264_encode: failed to create encoder");
    }
    auto guard = std::unique_ptr<ISVCEncoder, void (*)(ISVCEncoder*)>(
        encoder, [](ISVCEncoder* e) { e->Uninitialize(); WelsDestroySVCEncoder(e); });

    SEncParamExt paramExt{};
    encoder->GetDefaultParams(&paramExt);
    paramExt.iUsageType = CAMERA_VIDEO_REAL_TIME;
    paramExt.iPicWidth = padW;
    paramExt.iPicHeight = padH;
    paramExt.fMaxFrameRate = 30.0f;
    paramExt.iRCMode = RC_OFF_MODE;
    paramExt.bEnableFrameSkip = false;
    paramExt.iMultipleThreadIdc = 1;
    paramExt.sSpatialLayers[0].iVideoWidth = padW;
    paramExt.sSpatialLayers[0].iVideoHeight = padH;
    paramExt.sSpatialLayers[0].fFrameRate = 30.0f;
    paramExt.sSpatialLayers[0].iMaxSpatialBitrate = 0;
    paramExt.sSpatialLayers[0].iSpatialBitrate = 0;
    paramExt.iMaxQp = params.qp;
    paramExt.iMinQp = params.qp;
    if (encoder->InitializeExt(&paramExt) != 0) {
        throw std::runtime_error("h264_encode: encoder init failed");
    }

    const int yPlaneSize = padW * padH;
    const int uvPlaneSize = (padW / 2) * (padH / 2);
    std::vector<uint8_t> yBuf(yPlaneSize, 0);
    std::vector<uint8_t> uBuf(uvPlaneSize, 128);
    std::vector<uint8_t> vBuf(uvPlaneSize, 128);

    std::vector<std::byte> output;
    write_header(output, params.type, params.qp, Z, Y, X);

    SSourcePicture pic{};
    pic.iColorFormat = videoFormatI420;
    pic.iPicWidth = padW;
    pic.iPicHeight = padH;
    pic.iStride[0] = padW;
    pic.iStride[1] = padW / 2;
    pic.iStride[2] = padW / 2;
    pic.pData[0] = yBuf.data();
    pic.pData[1] = uBuf.data();
    pic.pData[2] = vBuf.data();

    SFrameBSInfo info{};
    for (int z = 0; z < Z; ++z) {
        const auto* src = reinterpret_cast<const uint8_t*>(raw.data()) + z * Y * X;
        fill_y_plane(yBuf, src, X, Y, padW, padH);
        pic.uiTimeStamp = z * 33;
        if (z == 0) encoder->ForceIntraFrame(true);

        std::memset(&info, 0, sizeof(info));
        if (encoder->EncodeFrame(&pic, &info) != 0) {
            throw std::runtime_error("h264_encode: EncodeFrame failed");
        }
        if (info.eFrameType != videoFrameTypeSkip) {
            for (int layer = 0; layer < info.iLayerNum; ++layer) {
                const auto& li = info.sLayerInfo[layer];
                int layerSize = 0;
                for (int n = 0; n < li.iNalCount; ++n) layerSize += li.pNalLengthInByte[n];
                auto old = output.size();
                output.resize(old + layerSize);
                std::memcpy(output.data() + old, li.pBsBuf, layerSize);
            }
        }
    }
    return output;
}

auto h264_decode(const uint8_t* bitstream, int bitstreamLen, int Z, int Y, int X)
    -> std::vector<std::byte>
{
    ISVCDecoder* decoder = nullptr;
    if (WelsCreateDecoder(&decoder) != 0 || !decoder) {
        throw std::runtime_error("h264_decode: failed to create decoder");
    }
    auto guard = std::unique_ptr<ISVCDecoder, void (*)(ISVCDecoder*)>(
        decoder, [](ISVCDecoder* d) { d->Uninitialize(); WelsDestroyDecoder(d); });

    SDecodingParam decParam{};
    decParam.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;
    if (decoder->Initialize(&decParam) != 0) {
        throw std::runtime_error("h264_decode: decoder init failed");
    }

    std::vector<std::byte> output(static_cast<std::size_t>(Z) * Y * X, std::byte{0});
    uint8_t* yuvData[3] = {};
    SBufferInfo bufInfo{};
    int framesDecoded = 0;

    auto extractFrame = [&]() {
        if (bufInfo.iBufferStatus == 1 && yuvData[0] && framesDecoded < Z) {
            int stride = bufInfo.UsrData.sSystemBuffer.iStride[0];
            auto* dst = reinterpret_cast<uint8_t*>(output.data()) + framesDecoded * Y * X;
            for (int y = 0; y < Y; ++y)
                std::memcpy(dst + y * X, yuvData[0] + y * stride, X);
            ++framesDecoded;
        }
    };

    // Find NAL start codes and feed each separately
    std::vector<int> nalStarts;
    for (int i = 0; i + 3 < bitstreamLen; ++i) {
        if (bitstream[i] == 0 && bitstream[i + 1] == 0 &&
            bitstream[i + 2] == 0 && bitstream[i + 3] == 1) {
            nalStarts.push_back(i);
        }
    }
    if (nalStarts.empty()) nalStarts.push_back(0);

    for (std::size_t i = 0; i < nalStarts.size(); ++i) {
        int start = nalStarts[i];
        int end = (i + 1 < nalStarts.size()) ? nalStarts[i + 1] : bitstreamLen;
        std::memset(&bufInfo, 0, sizeof(bufInfo));
        yuvData[0] = yuvData[1] = yuvData[2] = nullptr;
        decoder->DecodeFrameNoDelay(bitstream + start, end - start, yuvData, &bufInfo);
        extractFrame();
    }

    while (framesDecoded < Z) {
        std::memset(&bufInfo, 0, sizeof(bufInfo));
        yuvData[0] = yuvData[1] = yuvData[2] = nullptr;
        decoder->DecodeFrameNoDelay(nullptr, 0, yuvData, &bufInfo);
        if (bufInfo.iBufferStatus != 1) break;
        extractFrame();
    }

    return output;
}

#endif  // UTILS_HAS_H264

// ============================================================
// H.265 (x265 encode, libde265 decode)
// ============================================================

#ifdef UTILS_HAS_H265

auto h265_encode(std::span<const std::byte> raw, const VideoCodecParams& params)
    -> std::vector<std::byte>
{
    const int Z = params.depth, Y = params.height, X = params.width;
    const int padW = pad2(X), padH = pad2(Y);

    x265_param* xparam = x265_param_alloc();
    if (!xparam) throw std::runtime_error("h265_encode: param alloc failed");
    auto param_guard = std::unique_ptr<x265_param, void (*)(x265_param*)>(
        xparam, x265_param_free);

    x265_param_default_preset(xparam, "ultrafast", "zerolatency");
    xparam->sourceWidth = padW;
    xparam->sourceHeight = padH;
    xparam->internalCsp = X265_CSP_I420;
    xparam->fpsNum = 30;
    xparam->fpsDenom = 1;
    xparam->totalFrames = Z;
    xparam->bRepeatHeaders = 1;
    xparam->rc.rateControlMode = X265_RC_CQP;
    xparam->rc.qp = params.qp;
    xparam->bEnableWavefront = 0;
    xparam->frameNumThreads = 1;

    x265_encoder* enc = x265_encoder_open(xparam);
    if (!enc) throw std::runtime_error("h265_encode: encoder open failed");
    auto enc_guard = std::unique_ptr<x265_encoder, void (*)(x265_encoder*)>(
        enc, x265_encoder_close);

    x265_picture* pic = x265_picture_alloc();
    if (!pic) throw std::runtime_error("h265_encode: picture alloc failed");
    x265_picture_init(xparam, pic);
    auto pic_guard = std::unique_ptr<x265_picture, void (*)(x265_picture*)>(
        pic, x265_picture_free);

    const int yPlaneSize = padW * padH;
    const int uvPlaneSize = (padW / 2) * (padH / 2);
    std::vector<uint8_t> yBuf(yPlaneSize, 0);
    std::vector<uint8_t> uBuf(uvPlaneSize, 128);
    std::vector<uint8_t> vBuf(uvPlaneSize, 128);

    pic->planes[0] = yBuf.data();
    pic->planes[1] = uBuf.data();
    pic->planes[2] = vBuf.data();
    pic->stride[0] = padW;
    pic->stride[1] = padW / 2;
    pic->stride[2] = padW / 2;
    pic->colorSpace = X265_CSP_I420;

    std::vector<std::byte> output;
    write_header(output, params.type, params.qp, Z, Y, X);

    x265_nal* nals = nullptr;
    uint32_t nalCount = 0;

    for (int z = 0; z < Z; ++z) {
        const auto* src = reinterpret_cast<const uint8_t*>(raw.data()) + z * Y * X;
        fill_y_plane(yBuf, src, X, Y, padW, padH);
        pic->pts = z;

        int ret = x265_encoder_encode(enc, &nals, &nalCount, pic, nullptr);
        if (ret < 0) throw std::runtime_error("h265_encode: encode failed");
        for (uint32_t i = 0; i < nalCount; ++i) {
            auto old = output.size();
            output.resize(old + nals[i].sizeBytes);
            std::memcpy(output.data() + old, nals[i].payload, nals[i].sizeBytes);
        }
    }

    // Flush
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

auto h265_decode(const uint8_t* bitstream, int bitstreamLen, int Z, int Y, int X)
    -> std::vector<std::byte>
{
    de265_decoder_context* ctx = de265_new_decoder();
    if (!ctx) throw std::runtime_error("h265_decode: failed to create decoder");
    auto guard = std::unique_ptr<de265_decoder_context, void (*)(de265_decoder_context*)>(
        ctx, [](de265_decoder_context* c) { de265_free_decoder(c); });

    de265_set_parameter_bool(ctx, DE265_DECODER_PARAM_BOOL_SEI_CHECK_HASH, 0);
    // Use single-threaded decoding (0 = caller thread only)
    de265_start_worker_threads(ctx, 0);

    std::vector<std::byte> output(static_cast<std::size_t>(Z) * Y * X, std::byte{0});
    int framesDecoded = 0;

    auto extractFrames = [&]() {
        const de265_image* img;
        while ((img = de265_get_next_picture(ctx)) != nullptr && framesDecoded < Z) {
            int stride = 0;
            const uint8_t* yPlane = de265_get_image_plane(img, 0, &stride);
            if (!yPlane) continue;
            auto* dst = reinterpret_cast<uint8_t*>(output.data()) + framesDecoded * Y * X;
            for (int y = 0; y < Y; ++y)
                std::memcpy(dst + y * X, yPlane + y * stride, X);
            ++framesDecoded;
        }
    };

    // Feed all data then signal end-of-stream
    de265_push_data(ctx, bitstream, bitstreamLen, 0, nullptr);
    de265_flush_data(ctx);

    // Decode loop — keep calling de265_decode until no more frames
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

#endif  // UTILS_HAS_H265

// ============================================================
// AV1 (libaom encode, dav1d decode)
// ============================================================

#ifdef UTILS_HAS_AV1

auto av1_encode(std::span<const std::byte> raw, const VideoCodecParams& params)
    -> std::vector<std::byte>
{
    const int Z = params.depth, Y = params.height, X = params.width;
    const int padW = pad2(X), padH = pad2(Y);

    aom_codec_iface_t* iface = aom_codec_av1_cx();
    aom_codec_enc_cfg_t cfg;
    if (aom_codec_enc_config_default(iface, &cfg, AOM_USAGE_REALTIME) != AOM_CODEC_OK) {
        throw std::runtime_error("av1_encode: config default failed");
    }

    cfg.g_w = padW;
    cfg.g_h = padH;
    cfg.g_timebase.num = 1;
    cfg.g_timebase.den = 30;
    cfg.rc_end_usage = AOM_Q;
    cfg.rc_min_quantizer = params.qp;
    cfg.rc_max_quantizer = params.qp;
    cfg.g_threads = 1;
    cfg.g_lag_in_frames = 0;

    aom_codec_ctx_t codec;
    if (aom_codec_enc_init(&codec, iface, &cfg, 0) != AOM_CODEC_OK) {
        throw std::runtime_error("av1_encode: init failed");
    }
    auto codec_guard = std::unique_ptr<aom_codec_ctx_t, void (*)(aom_codec_ctx_t*)>(
        &codec, [](aom_codec_ctx_t* c) { aom_codec_destroy(c); });

    aom_codec_control(&codec, AOME_SET_CPUUSED, 10);

    aom_image_t img;
    aom_img_alloc(&img, AOM_IMG_FMT_I420, padW, padH, 1);
    auto img_guard = std::unique_ptr<aom_image_t, void (*)(aom_image_t*)>(
        &img, aom_img_free);

    std::memset(img.planes[1], 128, img.stride[1] * (padH / 2));
    std::memset(img.planes[2], 128, img.stride[2] * (padH / 2));

    std::vector<std::byte> output;
    write_header(output, params.type, params.qp, Z, Y, X);

    auto collectPackets = [&]() {
        const aom_codec_cx_pkt_t* pkt;
        aom_codec_iter_t iter = nullptr;
        while ((pkt = aom_codec_get_cx_data(&codec, &iter)) != nullptr) {
            if (pkt->kind == AOM_CODEC_CX_FRAME_PKT) {
                auto old = output.size();
                output.resize(old + pkt->data.frame.sz);
                std::memcpy(output.data() + old, pkt->data.frame.buf, pkt->data.frame.sz);
            }
        }
    };

    for (int z = 0; z < Z; ++z) {
        const auto* src = reinterpret_cast<const uint8_t*>(raw.data()) + z * Y * X;
        for (int y = 0; y < Y; ++y) {
            std::memcpy(img.planes[0] + y * img.stride[0], src + y * X, X);
            if (img.stride[0] > X)
                std::memset(img.planes[0] + y * img.stride[0] + X, 0, img.stride[0] - X);
        }
        for (int y = Y; y < padH; ++y) {
            std::memset(img.planes[0] + y * img.stride[0], 0, img.stride[0]);
        }

        aom_codec_err_t ret = aom_codec_encode(&codec, &img, z, 1, 0);
        if (ret != AOM_CODEC_OK) {
            throw std::runtime_error(
                std::string("av1_encode: encode failed: ") + aom_codec_error_detail(&codec));
        }
        collectPackets();
    }

    // Flush
    aom_codec_encode(&codec, nullptr, 0, 0, 0);
    collectPackets();

    return output;
}

auto av1_decode(const uint8_t* bitstream, int bitstreamLen, int Z, int Y, int X)
    -> std::vector<std::byte>
{
    Dav1dSettings settings;
    dav1d_default_settings(&settings);
    settings.n_threads = 1;

    Dav1dContext* ctx = nullptr;
    if (dav1d_open(&ctx, &settings) < 0 || !ctx) {
        throw std::runtime_error("av1_decode: failed to open decoder");
    }
    auto guard = std::unique_ptr<Dav1dContext, void (*)(Dav1dContext*)>(
        ctx, [](Dav1dContext* c) { dav1d_close(&c); });

    std::vector<std::byte> output(static_cast<std::size_t>(Z) * Y * X, std::byte{0});
    int framesDecoded = 0;

    auto drainFrames = [&]() {
        Dav1dPicture pic{};
        while (framesDecoded < Z) {
            int ret = dav1d_get_picture(ctx, &pic);
            if (ret < 0) break;
            auto* yPlane = static_cast<const uint8_t*>(pic.data[0]);
            int stride = static_cast<int>(pic.stride[0]);
            auto* dst = reinterpret_cast<uint8_t*>(output.data()) + framesDecoded * Y * X;
            for (int y = 0; y < Y; ++y)
                std::memcpy(dst + y * X, yPlane + y * stride, X);
            ++framesDecoded;
            dav1d_picture_unref(&pic);
        }
    };

    Dav1dData data{};
    uint8_t* dataBuf = dav1d_data_create(&data, bitstreamLen);
    if (!dataBuf) throw std::runtime_error("av1_decode: data create failed");
    std::memcpy(dataBuf, bitstream, bitstreamLen);

    while (data.sz > 0) {
        int ret = dav1d_send_data(ctx, &data);
        if (ret < 0 && ret != DAV1D_ERR(EAGAIN)) {
            dav1d_data_unref(&data);
            throw std::runtime_error("av1_decode: send_data failed");
        }
        drainFrames();
    }

    // Flush
    dav1d_flush(ctx);
    drainFrames();

    return output;
}

#endif  // UTILS_HAS_AV1

// ============================================================
// C3D: compress3d 3D-DCT codec, tiled into 32³ blocks
// ============================================================

// C3T container: tiles arbitrary chunks into 32³ sub-blocks.
// Header: "C3T\x01"(4) quality(1) pad(1) Z(2) Y(2) X(2) nblocks(4) = 16 bytes
// Index:  nblocks × {offset(4), size(4)} relative to data start
// Data:   concatenated c3d_compressed blocks

constexpr char C3T_MAGIC[4] = {'C', '3', 'T', 0x01};
constexpr std::size_t C3T_HEADER_SIZE = 16;

static auto c3d_encode_chunk(std::span<const std::byte> raw, const VideoCodecParams& params)
    -> std::vector<std::byte>
{
    const int Z = params.depth, Y = params.height, X = params.width;
    // Quality: map QP 0-51 to C3D quality 100-1
    int quality = std::clamp(100 - params.qp * 2, 1, 100);

    // Calculate number of 32³ blocks per axis (round up)
    const int bz = (Z + 31) / 32, by = (Y + 31) / 32, bx = (X + 31) / 32;
    const int nblocks = bz * by * bx;

    // Compress each 32³ block
    struct BlockResult { std::vector<uint8_t> data; };
    std::vector<BlockResult> blocks(nblocks);

    uint8_t block_in[C3D_BLOCK_VOXELS];
    int bi = 0;
    for (int bzi = 0; bzi < bz; bzi++) {
        for (int byi = 0; byi < by; byi++) {
            for (int bxi = 0; bxi < bx; bxi++, bi++) {
                // Extract 32³ sub-block, zero-padding at boundaries
                std::memset(block_in, 0, C3D_BLOCK_VOXELS);
                const int z0 = bzi * 32, y0 = byi * 32, x0 = bxi * 32;
                for (int z = 0; z < 32 && z0 + z < Z; z++) {
                    for (int y = 0; y < 32 && y0 + y < Y; y++) {
                        int src_off = (z0 + z) * Y * X + (y0 + y) * X + x0;
                        int dst_off = z * 32 * 32 + y * 32;
                        int copy_x = std::min(32, X - x0);
                        std::memcpy(block_in + dst_off,
                                    reinterpret_cast<const uint8_t*>(raw.data()) + src_off,
                                    copy_x);
                    }
                }

                auto result = c3d_compress(block_in, quality);
                blocks[bi].data.assign(result.data, result.data + result.size);
                free(result.data);
            }
        }
    }

    // Build container
    std::size_t index_size = nblocks * 8;
    std::size_t data_offset = C3T_HEADER_SIZE + index_size;
    std::size_t total = data_offset;
    for (auto& b : blocks) total += b.data.size();

    std::vector<std::byte> output(total);
    auto* out = output.data();

    // Header
    std::memcpy(out, C3T_MAGIC, 4);
    out[4] = static_cast<std::byte>(quality);
    out[5] = static_cast<std::byte>(0);
    write_le16(out + 6, static_cast<uint16_t>(Z));
    write_le16(out + 8, static_cast<uint16_t>(Y));
    write_le16(out + 10, static_cast<uint16_t>(X));
    write_le32(out + 12, static_cast<uint32_t>(nblocks));

    // Index + data
    uint32_t cur_offset = 0;
    for (int i = 0; i < nblocks; i++) {
        write_le32(out + C3T_HEADER_SIZE + i * 8, cur_offset);
        write_le32(out + C3T_HEADER_SIZE + i * 8 + 4, static_cast<uint32_t>(blocks[i].data.size()));
        std::memcpy(out + data_offset + cur_offset, blocks[i].data.data(), blocks[i].data.size());
        cur_offset += static_cast<uint32_t>(blocks[i].data.size());
    }

    return output;
}

static auto c3d_decode_chunk(std::span<const std::byte> compressed, std::size_t out_size)
    -> std::vector<std::byte>
{
    if (compressed.size() < C3T_HEADER_SIZE ||
        std::memcmp(compressed.data(), C3T_MAGIC, 4) != 0)
        throw std::runtime_error("c3d_decode: invalid C3T magic");

    auto* hdr = compressed.data();
    int Z = read_le16(hdr + 6);
    int Y = read_le16(hdr + 8);
    int X = read_le16(hdr + 10);
    int nblocks = static_cast<int>(read_le32(hdr + 12));

    if (out_size != static_cast<std::size_t>(Z) * Y * X)
        throw std::runtime_error("c3d_decode: out_size mismatch");

    const int bz = (Z + 31) / 32, by = (Y + 31) / 32, bx = (X + 31) / 32;
    if (nblocks != bz * by * bx)
        throw std::runtime_error("c3d_decode: block count mismatch");

    std::size_t index_size = nblocks * 8;
    std::size_t data_start = C3T_HEADER_SIZE + index_size;

    std::vector<std::byte> output(out_size, std::byte{0});
    uint8_t block_out[C3D_BLOCK_VOXELS];

    int bi = 0;
    for (int bzi = 0; bzi < bz; bzi++) {
        for (int byi = 0; byi < by; byi++) {
            for (int bxi = 0; bxi < bx; bxi++, bi++) {
                uint32_t offset = read_le32(hdr + C3T_HEADER_SIZE + bi * 8);
                uint32_t size = read_le32(hdr + C3T_HEADER_SIZE + bi * 8 + 4);

                const auto* block_data = reinterpret_cast<const uint8_t*>(
                    compressed.data() + data_start + offset);
                if (c3d_decompress(block_data, size, block_out) != 0)
                    throw std::runtime_error("c3d_decode: decompression failed");

                // Scatter 32³ block back into output
                const int z0 = bzi * 32, y0 = byi * 32, x0 = bxi * 32;
                for (int z = 0; z < 32 && z0 + z < Z; z++) {
                    for (int y = 0; y < 32 && y0 + y < Y; y++) {
                        int dst_off = (z0 + z) * Y * X + (y0 + y) * X + x0;
                        int src_off = z * 32 * 32 + y * 32;
                        int copy_x = std::min(32, X - x0);
                        std::memcpy(reinterpret_cast<uint8_t*>(output.data()) + dst_off,
                                    block_out + src_off, copy_x);
                    }
                }
            }
        }
    }

    return output;
}

}  // namespace

// ============================================================
// Public API
// ============================================================

auto video_encode(std::span<const std::byte> raw, const VideoCodecParams& params)
    -> std::vector<std::byte>
{
    const int Z = params.depth, Y = params.height, X = params.width;
    if (Z <= 0 || Y <= 0 || X <= 0) {
        throw std::runtime_error("video_encode: invalid dimensions");
    }
    if (raw.size() < static_cast<std::size_t>(Z) * Y * X) {
        throw std::runtime_error("video_encode: input buffer too small");
    }

    switch (params.type) {
#ifdef UTILS_HAS_H264
        case VideoCodecType::H264: return h264_encode(raw, params);
#endif
#ifdef UTILS_HAS_H265
        case VideoCodecType::H265: return h265_encode(raw, params);
#endif
#ifdef UTILS_HAS_AV1
        case VideoCodecType::AV1: return av1_encode(raw, params);
#endif
        case VideoCodecType::C3D: return c3d_encode_chunk(raw, params);
        default:
            throw std::runtime_error("video_encode: requested codec not compiled in");
    }
}

auto video_decode(
    std::span<const std::byte> compressed,
    std::size_t out_size,
    const VideoCodecParams& /*params*/) -> std::vector<std::byte>
{
    // Check for C3D tiled container first
    if (is_c3d_compressed(compressed)) {
        return c3d_decode_chunk(compressed, out_size);
    }

    auto hdr = read_header(compressed);
    if (out_size != static_cast<std::size_t>(hdr.Z) * hdr.Y * hdr.X) {
        throw std::runtime_error("video_decode: out_size mismatch with header dimensions");
    }

    const auto* bitstream =
        reinterpret_cast<const uint8_t*>(compressed.data() + HEADER_SIZE);
    const int bitstreamLen = static_cast<int>(compressed.size() - HEADER_SIZE);

    switch (hdr.codec) {
#ifdef UTILS_HAS_H264
        case VideoCodecType::H264: return h264_decode(bitstream, bitstreamLen, hdr.Z, hdr.Y, hdr.X);
#endif
#ifdef UTILS_HAS_H265
        case VideoCodecType::H265: return h265_decode(bitstream, bitstreamLen, hdr.Z, hdr.Y, hdr.X);
#endif
#ifdef UTILS_HAS_AV1
        case VideoCodecType::AV1: return av1_decode(bitstream, bitstreamLen, hdr.Z, hdr.Y, hdr.X);
#endif
        default:
            throw std::runtime_error("video_decode: requested codec not compiled in");
    }
}

}  // namespace utils
