#pragma once
#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/PackedTrackStore.hpp"
#include "ObjectTrackInput.hpp"
#include <set>
namespace vc::graph_links::detail {
// Normalize before deduplication so casing variants select one object.
std::set<std::string> requestedTrackUuids(const std::vector<std::string>& uuids);
struct TrackInput {
    std::filesystem::path packedPath;
    std::string datasetRef;
    std::vector<graph::ObjectRecord> objects;
    std::vector<size_t> registrations;
};
using objects::detail::trackSourceId;
using objects::detail::trackRow;
using objects::detail::trackPolylineLength;
using objects::detail::trackFamily;
TrackInput loadTrackInput(graph::GraphDb& db, const std::filesystem::path& path,
    const std::vector<uint64_t>& sourceIds, const std::vector<std::string>& trackUuids,
    const std::atomic_bool* cancel, const util::ProgressCallback& progress = {});
} // namespace vc::graph_links::detail
