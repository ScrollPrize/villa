#pragma once
#include "vc_object_store/objects/Geometry.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace vc::objects::detail {
// Shared scalar encoding for storage, adapters and typed geometry. Never
// reinterpret byte buffers as aligned native arrays.
template<class T> void copyScalars(std::span<const uint8_t> bytes, std::span<T> output)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (bytes.size() != output.size_bytes()) throw std::runtime_error("invalid scalar array size");
    if (!bytes.empty()) std::memcpy(output.data(), bytes.data(), bytes.size());
    if constexpr (std::endian::native != std::endian::little) {
        auto* data = reinterpret_cast<uint8_t*>(output.data());
        for (size_t i = 0; i < bytes.size(); i += sizeof(T))
            std::reverse(data + i, data + i + sizeof(T));
    }
}

template<class T> std::vector<uint8_t> scalarBytes(std::span<const T> values)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::vector<uint8_t> bytes(values.size_bytes());
    if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    if constexpr (std::endian::native != std::endian::little)
        for (size_t i = 0; i < bytes.size(); i += sizeof(T))
            std::reverse(bytes.begin() + i, bytes.begin() + i + sizeof(T));
    return bytes;
}

template<class T> GeometryArray scalarArray(std::string name, std::string dtype,
    std::vector<int64_t> shape, std::span<const T> values)
{
    return {std::move(name), std::move(dtype), std::move(shape), scalarBytes(values)};
}

template<class T> std::vector<std::array<T, 3>> xyzValues(std::span<const uint8_t> bytes)
{
    if (bytes.size() % (3 * sizeof(T))) throw std::runtime_error("invalid XYZ array size");
    std::vector<std::array<T, 3>> result(bytes.size() / (3 * sizeof(T)));
    for (size_t i = 0; i < result.size(); ++i)
        copyScalars<T>(bytes.subspan(i * 3 * sizeof(T), 3 * sizeof(T)), result[i]);
    return result;
}

template<class T> GeometryArray xyzArray(std::string name, std::string dtype,
    std::vector<int64_t> shape, std::span<const std::array<T, 3>> points)
{
    GeometryArray result{std::move(name), std::move(dtype), std::move(shape), {}};
    result.bytes.resize(points.size() * 3 * sizeof(T));
    for (size_t i = 0; i < points.size(); ++i) {
        // Copy each array independently: no assumptions about std::array padding.
        std::memcpy(result.bytes.data() + i * 3 * sizeof(T), points[i].data(), 3 * sizeof(T));
    }
    if constexpr (std::endian::native != std::endian::little)
        for (size_t i = 0; i < result.bytes.size(); i += sizeof(T))
            std::reverse(result.bytes.begin() + i, result.bytes.begin() + i + sizeof(T));
    return result;
}
} // namespace vc::objects::detail
