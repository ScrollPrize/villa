#include "../../src/cpp/codecs.hpp"
#include <iostream>

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main()
{
    try {
        for (const auto shape : {std::array{32, 32, 32}, std::array{32, 32, 64},
                                 std::array{17, 19, 23}}) {
            for (std::size_t itemsize : {1u, 2u}) {
                std::vector<std::uint8_t> raw(shape[0] * shape[1] * shape[2] * itemsize);
                for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = (i * 17 + i / 113) % 256;
                const auto encoded = vcr::encode_delta3d(raw.data(), raw.size(), shape, itemsize);
                require(vcr::delta3d_lossless(encoded.data(), encoded.size()), "lossless cache rejected");
                std::vector<std::uint8_t> decoded(raw.size(),0xa5);
                vcr::decode(vcr::Codec::Delta3d, encoded.data(), encoded.size(), decoded.data(), decoded.size());
                require(decoded == raw, "roundtrip mismatch");
                const auto bytes = std::as_bytes(std::span(raw));
                for (int quant : {1, 3}) {
                    const auto old = vc_delta3d::compress(bytes, shape, itemsize, quant, vc_delta3d::Predictor::Delta);
                    require(vcr::delta3d_lossless(reinterpret_cast<const uint8_t*>(old.data()), old.size()) == (quant == 1),
                            "quantization classification");
                }
            }
        }
        // 4-D chunks, in both channel layouts; a multiple of 32 takes the preset container, others the legacy one.
        for (const std::int64_t side : {32, 64, 20})
            for (const std::string axis : {"first", "last"})
                for (std::size_t itemsize : {1u, 2u}) {
                    const std::int64_t channels = 6;
                    const std::vector<std::int64_t> shape = axis == "first" ? std::vector{channels, side, side, side}
                                                                            : std::vector{side, side, side, channels};
                    std::vector<std::uint8_t> raw(std::size_t(channels * side * side * side) * itemsize);
                    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = (i * 17 + i / 113) % 256;
                    vcr::EncodeOptions options;
                    options.channel_axis = axis;
                    const auto encoded = vcr::encode(vcr::Codec::Delta3d, options, raw.data(), raw.size(), itemsize, shape);
                    const auto frame = std::as_bytes(std::span(encoded));
                    require(vc_delta3d::channelCount(frame) == 6, "channel count");
                    require(vcr::delta3d_lossless(encoded.data(), encoded.size()), "4-D frame not lossless");
                    std::vector<std::uint8_t> decoded(raw.size(), 0xa5);
                    vcr::decode(vcr::Codec::Delta3d, encoded.data(), encoded.size(), decoded.data(), decoded.size());
                    require(decoded == raw, "4-D roundtrip mismatch");
                }
        const auto rejected = [](std::vector<std::int64_t> shape, const std::string& axis) {
            std::vector<std::uint8_t> raw(8 * 8 * 8);
            vcr::EncodeOptions options;
            options.channel_axis = axis;
            try { vcr::encode(vcr::Codec::Delta3d, options, raw.data(), raw.size(), 1, shape); }
            catch (const std::invalid_argument&) { return true; }
            return false;
        };
        require(rejected({1, 8, 8, 8}, "") && rejected({1, 8, 8, 8}, "middle") && rejected({8, 8, 8}, "first") &&
                    rejected({8, 64}, ""), "unsupported layout accepted");
        const std::array<uint8_t, 8> invalid{};
        require(!vcr::delta3d_lossless(invalid.data(), invalid.size()), "invalid frame accepted");
        std::cout << "Delta3D reader codec roundtrips and lossless gating passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
