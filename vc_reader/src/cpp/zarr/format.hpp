#pragma once

#include "../reader_types.hpp"
#include <stdexcept>
#include <map>
#include <string>

namespace vcr::zarr::detail {
inline void validateShape(const Index& shape, const Index& chunks)
{
    if (shape.empty() || shape.size() != chunks.size())
        throw std::invalid_argument("Zarr shape and chunks must have the same nonzero rank");
    for (std::size_t i = 0; i < shape.size(); ++i)
        if (shape[i] < 0 || chunks[i] <= 0)
            throw std::invalid_argument("invalid Zarr shape or chunks");
}

// The same names are used when normalizing metadata and when creating arrays.
inline const std::map<std::string, std::string> v3Types{
    {"bool", "|b1"}, {"uint8", "|u1"}, {"uint16", "<u2"}, {"uint32", "<u4"},
    {"uint64", "<u8"}, {"int8", "|i1"}, {"int16", "<i2"}, {"int32", "<i4"},
    {"int64", "<i8"}, {"float16", "<f2"}, {"float32", "<f4"}, {"float64", "<f8"}};

inline const std::map<std::string, std::string> v3Compressors{
    {"blosc", "blosc"}, {"zstd", "zstd"}, {"gzip", "gzip"},
    {"numcodecs.zlib", "zlib"}, {"numcodecs.lz4", "lz4"},
    {"vc-delta3d", "vc-delta3d"}, {"volcomp", "volcomp"}};
}  // namespace vcr::zarr::detail
