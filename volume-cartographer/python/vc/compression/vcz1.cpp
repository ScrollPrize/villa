// Python bindings for the VCZ1 zarr/chunk codec (quantize + delta-zyx +
// zstd/rANS).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

#include "vc/core/util/CacheCompression.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

vc::CacheCodec parseCodec(const std::string& codec)
{
    if (codec == "rans")
        return vc::CacheCodec::Rans;
    if (codec == "zstd")
        return vc::CacheCodec::Zstd;
    throw std::invalid_argument("codec must be 'rans' or 'zstd'");
}

template <typename T>
nb::bytes compressArray(
    nb::ndarray<nb::numpy, const T, nb::ndim<3>, nb::c_contig> array,
    int quant,
    const std::string& codec)
{
    const auto z = static_cast<int>(array.shape(0));
    const auto y = static_cast<int>(array.shape(1));
    const auto x = static_cast<int>(array.shape(2));
    const auto nbytes = static_cast<std::size_t>(array.size()) * sizeof(T);
    const auto out = vc::cacheCompress(
        {reinterpret_cast<const std::byte*>(array.data()), nbytes},
        {z, y, x}, sizeof(T), vc::kCacheCompressionLevel, quant,
        parseCodec(codec));
    return nb::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

} // namespace

NB_MODULE(vcz1, m)
{
    m.doc() = "VCZ1 cache-chunk codec (quantize + delta-zyx + zstd/rANS)";

    m.def(
        "compress",
        [](nb::bytes raw, int z, int y, int x, int elem_size, int quant,
           const std::string& codec) {
            const auto out = vc::cacheCompress(
                {reinterpret_cast<const std::byte*>(raw.c_str()), raw.size()},
                {z, y, x}, static_cast<std::size_t>(elem_size),
                vc::kCacheCompressionLevel, quant, parseCodec(codec));
            return nb::bytes(reinterpret_cast<const char*>(out.data()),
                             out.size());
        },
        "raw"_a, "z"_a, "y"_a, "x"_a, "elem_size"_a, "quant"_a = 1,
        "codec"_a = "rans",
        "Compress one decoded chunk (raw C-order voxel bytes) into a "
        "self-describing VCZ1 payload.");


    m.def(
        "compress_array",
        &compressArray<std::uint8_t>,
        "array"_a, "quant"_a = 1, "codec"_a = "rans",
        "Compress a C-contiguous uint8 3D NumPy chunk without first copying it "
        "to Python bytes.");

    m.def(
        "compress_array",
        &compressArray<std::uint16_t>,
        "array"_a, "quant"_a = 1, "codec"_a = "rans",
        "Compress a C-contiguous uint16 3D NumPy chunk without first copying it "
        "to Python bytes.");

    m.def(
        "decompress",
        [](nb::bytes payload, std::size_t expected_size) {
            auto out = vc::cacheDecompress(
                {reinterpret_cast<const std::byte*>(payload.c_str()),
                 payload.size()},
                expected_size);
            if (!out) {
                throw std::runtime_error(
                    "not a valid VCZ1 payload of the expected size");
            }
            return nb::bytes(reinterpret_cast<const char*>(out->data()),
                             out->size());
        },
        "payload"_a, "expected_size"_a,
        "Decompress a VCZ1 payload (either codec) back to raw voxel bytes.");

    m.def(
        "decompress_into",
        [](nb::bytes payload,
           nb::ndarray<nb::numpy, std::uint8_t, nb::c_contig> output) {
            const auto ok = vc::cacheDecompressInto(
                {reinterpret_cast<const std::byte*>(payload.c_str()),
                 payload.size()},
                {reinterpret_cast<std::byte*>(output.data()), output.size()});
            if (!ok) {
                throw std::runtime_error(
                    "not a valid VCZ1 payload for the supplied output size");
            }
        },
        "payload"_a, "output"_a,
        "Decompress a VCZ1 payload directly into a C-contiguous uint8 buffer.");
}
