#pragma once
#include "vc_object_store/objects/Geometry.hpp"
#include <limits>
#include <stdexcept>
namespace vc::objects::detail {
inline int64_t geometryByteLength(const std::string& dtype, const std::vector<int64_t>& shape)
{
    int64_t bytes;
    if (dtype == "|u1") bytes = 1;
    else if (dtype == "<i4" || dtype == "<f4") bytes = 4;
    else if (dtype == "<i8" || dtype == "<u8" || dtype == "<f8") bytes = 8;
    else throw std::invalid_argument("geometry: unsupported dtype " + dtype);
    if (shape.empty() || shape.size() > 8) throw std::invalid_argument("geometry: invalid array rank");
    for (auto dimension : shape) {
        if (dimension < 0 || (dimension && bytes > std::numeric_limits<int64_t>::max() / dimension))
            throw std::invalid_argument("geometry: invalid/overflowing shape");
        bytes *= dimension;
    }
    return bytes;
}

inline void requireArrayLayout(const GeometryArrayInfo& array, const char* dtype, size_t rank)
{
    if (array.dtype != dtype || array.shape.size() != rank ||
        (rank > 1 && array.shape.back() != 3) ||
        array.byteLength != geometryByteLength(array.dtype, array.shape))
        throw std::invalid_argument("invalid typed geometry layout for " + array.name);
}
}
