#pragma once

// Shared exact-voxel crossing kernel used by the graph linker and vc_spiral.
// Coordinates are flat int32 ZYX triples, in source-dataset voxel units.
// Tracks must be ordered by strictly increasing source ID. The low-level
// scan requires a permutation sorted by packed 60-bit voxel key.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace vc::track_crossings {
using Progress = std::function<void(const char*, uint64_t, uint64_t)>;
struct Parameters {
    double minAngleDegrees = 30.0;
    double tangentRadius = 12.0;
    int clusterIndexRadius = 4;
};
struct Event {
    int32_t first, second, first_local, second_local;
};
struct EventBuffer {
    std::vector<Event> events;
    size_t size() const { return events.size(); }
    size_t memory_bytes() const { return events.capacity() * sizeof(Event); }
};
struct Tangent {
    double z = 0, y = 0, x = 0;
    bool valid = false;
};
struct Crossings {
    std::vector<int64_t> offsets;
    std::vector<int32_t> partners, self_local, partner_local;
    std::vector<double> positions, clearances;
    uint64_t accepted_events = 0, paired_tracks = 0;
};
void validateParameters(const Parameters& params);
void validateTracks(std::span<const int32_t> coordinates, std::span<const int64_t> offsets);
Tangent track_tangent(const int32_t* coordinates, const int64_t* offsets,
                      int32_t track, int32_t local, double radius = 12.0);
std::vector<uint32_t> parallel_argsort(std::span<const uint64_t> packed,
                                     int workers, const Progress& progress = {});
EventBuffer scan_crossing_events(
    std::span<const int32_t> coordinates, std::span<const int64_t> offsets,
    std::span<const int8_t> family_codes, std::span<const uint64_t> packed,
    std::span<const uint32_t> order, int workers, const Progress& progress = {},
    const Parameters& params = {});
// Consumes buffer.events; CSR rows are sorted by partner row (source ID).
Crossings consolidate_crossing_events(
    EventBuffer& buffer, std::span<const int32_t> coordinates,
    std::span<const int64_t> offsets, std::span<const uint64_t> source_ids,
    int workers, const Progress& progress = {}, const Parameters& params = {});
} // namespace vc::track_crossings
