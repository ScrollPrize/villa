#pragma once

#include <map>
#include <string>

namespace vcr::zarr::detail {
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
