#pragma once
#include "vc_object_store/objects/Geometry.hpp"
#include <limits>
#include <stdexcept>
namespace vc::objects::detail {
inline int64_t geometryByteLength(const std::string& dtype, const std::vector<int64_t>& shape)
{
    int64_t bytes;
    if (dtype == "|u1") bytes = 1;
    else if (dtype == "<u2") bytes = 2;
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

namespace vc::objects::detail {
inline int64_t surfacePixelBytes(const std::string& dtype, const std::vector<int64_t>& shape)
{
    if ((shape.size() != 2 && shape.size() != 3) || (shape.size() == 3 && shape[2] <= 0))
        throw std::invalid_argument("surface channel requires [rows,columns] or [rows,columns,components]");
    return geometryByteLength(dtype, {shape.size() == 3 ? shape[2] : 1});
}
inline void validateSurfaceChannel(const GeometryArray& array, const std::vector<int64_t>& grid)
{
    surfacePixelBytes(array.dtype, array.shape);
    if (array.shape[0] != grid[0] || array.shape[1] != grid[1])
        throw std::invalid_argument("surface channel dimensions must match points: " + array.name);
    if (geometryByteLength(array.dtype, array.shape) != int64_t(array.bytes.size()))
        throw std::invalid_argument("surface channel shape does not match byte length: " + array.name);
}
}
