#pragma once
#include "vc/core/objects/Geometry.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace vc::objects::detail {
struct GeometrySnapshot {
    GeometryRevision revision;
    std::map<std::string, std::vector<uint8_t>> arrays;
    template<class T> std::vector<T> values(const std::string& name) const {
        auto bytes = arrays.at(name);
        if (bytes.size() % sizeof(T)) throw std::runtime_error("invalid scalar array size");
        if constexpr (std::endian::native != std::endian::little)
            for (size_t i = 0; i < bytes.size(); i += sizeof(T))
                std::reverse(bytes.begin() + i, bytes.begin() + i + sizeof(T));
        std::vector<T> result(bytes.size() / sizeof(T));
        if (!bytes.empty()) std::memcpy(result.data(), bytes.data(), bytes.size());
        if constexpr (std::is_floating_point_v<T>)
            for (auto value : result)
                if (!std::isfinite(value)) throw std::runtime_error("geometry requires finite coordinates");
        return result;
    }
};
} // namespace vc::objects::detail
