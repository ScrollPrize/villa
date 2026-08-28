#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <spiral_graph/track_io.hpp>

namespace spiral::trackio {

struct SpatialBounds {
    float z_min = 0.0f;
    float y_min = 0.0f;
    float x_min = 0.0f;
    float z_max = 0.0f;
    float y_max = 0.0f;
    float x_max = 0.0f;
};

struct SpatialIndexInfo {
    std::uint64_t point_count = 0;
    std::uint64_t cell_count = 0;
    std::uint32_t cell_size = 0;
    bool already_present = false;
};

// Exact point-id lookup with an out-of-core build. The build externally sorts
// fixed-size runs, so peak memory is bounded independently of point count.
class TrackSpatialIndex {
public:
    TrackSpatialIndex();
    ~TrackSpatialIndex();
    TrackSpatialIndex(TrackSpatialIndex&&) noexcept;
    TrackSpatialIndex& operator=(TrackSpatialIndex&&) noexcept;
    TrackSpatialIndex(const TrackSpatialIndex&) = delete;
    TrackSpatialIndex& operator=(const TrackSpatialIndex&) = delete;

    static SpatialIndexInfo build(
        const PackedTrackStore& tracks,
        const std::filesystem::path& output,
        std::uint32_t cell_size = 32,
        std::size_t memory_budget_bytes = 512ull << 20);

    void open(const std::filesystem::path& path);
    void validate_source(const PackedTrackStore& tracks) const;
    SpatialIndexInfo info() const;
    void query(const SpatialBounds& bounds, std::vector<std::uint64_t>& point_ids) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spiral::trackio
