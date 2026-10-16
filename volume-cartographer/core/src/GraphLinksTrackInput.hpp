#pragma once
#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/PackedTrackStore.hpp"
namespace vc::graph_links::detail {
struct TrackInput {
    std::filesystem::path packedPath;
    std::string datasetRef;
    std::vector<graph::ObjectRecord> objects;
    std::vector<size_t> registrations;
};
uint64_t trackSourceId(const std::string& key);
size_t trackRow(const PackedTrackStore& store, const graph::ObjectRecord& object);
double trackPolylineLength(const PackedTrackStore& store, size_t row, const std::atomic_bool* cancel);
std::string trackFamily(const PackedTrackStore& store, size_t row);
TrackInput loadTrackInput(graph::GraphDb& db, const std::filesystem::path& path,
    const std::vector<uint64_t>& sourceIds, const std::vector<std::string>& trackUuids,
    const std::atomic_bool* cancel);
} // namespace vc::graph_links::detail
