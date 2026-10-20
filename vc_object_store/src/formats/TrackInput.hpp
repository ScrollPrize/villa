#pragma once
// Shared source identity and measurements for ingestion and link detection.
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/PackedTrackStore.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
namespace vc::objects::detail {
inline uint64_t trackSourceId(const std::string& key)
{
    uint64_t id;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), id);
    if (error != std::errc{} || end != key.data() + key.size() || std::to_string(id) != key)
        throw std::runtime_error("track external_key must be a canonical decimal source ID: " + key);
    return id;
}
inline double trackPolylineLength(const PackedTrackStore& store, size_t row, const std::atomic_bool* cancel)
{
    const auto begin = store.offsets()[row], end = store.offsets()[row+1];
    const auto coordinates = store.coordinates();
    double length = 0;
    for (auto i = begin + 1; i < end; ++i) {
        if ((i & 1023) == 0 && cancel && cancel->load())
            throw std::runtime_error("track loading cancelled");
        double squared = 0;
        for (int axis = 0; axis < 3; ++axis) {
            // Cast before subtraction, and accumulate in source coordinates.
            const double delta = double(coordinates[3*i+axis]) - double(coordinates[3*(i-1)+axis]);
            squared += delta * delta;
        }
        length += std::sqrt(squared);
    }
    return length;
}
inline std::string trackFamily(const PackedTrackStore& store, size_t row)
{
    const auto family = store.families()[row];
    return family == 0 ? "horizontal" : family == 1 ? "vertical" : "";
}
} // namespace vc::objects::detail
