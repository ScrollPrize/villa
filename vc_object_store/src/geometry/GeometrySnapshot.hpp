#pragma once
#include "geometry/ArrayBytes.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace vc::objects::detail {
struct GeometrySnapshot {
    GeometryRevision revision;
    std::map<std::string, std::vector<uint8_t>> arrays;
    template<class T> void copyValues(const std::string& name, std::span<T> result) const {
        copyScalars<T>(arrays.at(name), result);
    }
    template<class T> std::vector<T> values(const std::string& name) const {
        const auto& bytes = arrays.at(name);
        if (bytes.size() % sizeof(T)) throw std::runtime_error("invalid scalar array size");
        std::vector<T> result(bytes.size() / sizeof(T));
        copyValues<T>(name, result);
        if constexpr (std::is_floating_point_v<T>)
            for (auto value : result)
                if (!std::isfinite(value)) throw std::runtime_error("geometry requires finite coordinates");
        return result;
    }
};
} // namespace vc::objects::detail
