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
                std::vector<std::uint8_t> decoded(raw.size());
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
        const std::array<uint8_t, 8> invalid{};
        require(!vcr::delta3d_lossless(invalid.data(), invalid.size()), "invalid frame accepted");
        std::cout << "Delta3D reader codec roundtrips and lossless gating passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
