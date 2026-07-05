// Python bindings for the VCZ1 cache-chunk codec (quantize + delta-zyx +
// zstd/rANS). Used by scripts/recompress_zarr.py to read and write
// rANS-coded (codec 1) payloads, which have no pure-Python decoder.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "vc/core/util/CacheCompression.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(cache_codec, m)
{
    m.doc() = "VCZ1 cache-chunk codec (quantize + delta-zyx + zstd/rANS)";

    m.def(
        "compress",
        [](nb::bytes raw, int z, int y, int x, int elem_size, int quant,
           const std::string& codec) {
            vc::CacheCodec c;
            if (codec == "rans") {
                c = vc::CacheCodec::Rans;
            } else if (codec == "zstd") {
                c = vc::CacheCodec::Zstd;
            } else {
                throw std::invalid_argument("codec must be 'rans' or 'zstd'");
            }
            const auto out = vc::cacheCompress(
                {reinterpret_cast<const std::byte*>(raw.c_str()), raw.size()},
                {z, y, x}, static_cast<std::size_t>(elem_size),
                vc::kCacheCompressionLevel, quant, c);
            return nb::bytes(reinterpret_cast<const char*>(out.data()),
                             out.size());
        },
        "raw"_a, "z"_a, "y"_a, "x"_a, "elem_size"_a, "quant"_a = 1,
        "codec"_a = "rans",
        "Compress one decoded chunk (raw C-order voxel bytes) into a "
        "self-describing VCZ1 payload.");

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
}
