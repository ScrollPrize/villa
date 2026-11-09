#include "helirec/delta3d.hpp"

#include "helirec/common.hpp"

#ifdef HELIREC_WITH_DELTA3D
#include <vc_delta3d/codec.hpp>
#endif

namespace helirec {

#ifdef HELIREC_WITH_DELTA3D
bool delta3d_available() { return true; }

std::vector<uint8_t> delta3d_compress_u8(const uint8_t* data, int z, int y, int x, const char* preset) {
    const std::string name = preset ? preset : "fast";
    vc_delta3d::Predictor pred;
    if (name == "fast") pred = vc_delta3d::Predictor::Fast;
    else if (name == "medium") pred = vc_delta3d::Predictor::Medium;
    else if (name == "slow") pred = vc_delta3d::Predictor::Slow;
    else if (name == "slower") pred = vc_delta3d::Predictor::Slower;
    else if (name == "delta") pred = vc_delta3d::Predictor::Delta;
    else throw Error("vc-delta3d: unknown preset '" + name + "'");
    const std::size_t n = static_cast<std::size_t>(z) * y * x;
    auto enc = vc_delta3d::compress(std::span(reinterpret_cast<const std::byte*>(data), n), {z, y, x}, 1,
                                    vc_delta3d::kQuantizationLossless, pred);
    const auto* p = reinterpret_cast<const uint8_t*>(enc.data());
    return std::vector<uint8_t>(p, p + enc.size());
}

void delta3d_decompress(const uint8_t* src, std::size_t n, uint8_t* out, std::size_t out_size) {
    auto size = vc_delta3d::decodedSize(std::span(reinterpret_cast<const std::byte*>(src), n));
    if (!size || *size != out_size) throw Error("vc-delta3d: payload size does not match the chunk");
    if (!vc_delta3d::decompressInto(std::span(reinterpret_cast<const std::byte*>(src), n),
                                    std::span(reinterpret_cast<std::byte*>(out), out_size)))
        throw Error("vc-delta3d: decode failed");
}
#else
bool delta3d_available() { return false; }
std::vector<uint8_t> delta3d_compress_u8(const uint8_t*, int, int, int, const char*) { throw Error("built without vc-delta3d"); }
void delta3d_decompress(const uint8_t*, std::size_t, uint8_t*, std::size_t) { throw Error("built without vc-delta3d"); }
#endif

}  // namespace helirec
