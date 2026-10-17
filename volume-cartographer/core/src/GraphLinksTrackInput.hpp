#pragma once
#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/PackedTrackStore.hpp"
#include "GraphTrackInput.hpp"
namespace vc::graph_links::detail {
struct TrackInput {
    std::filesystem::path packedPath;
    std::string datasetRef;
    std::vector<graph::ObjectRecord> objects;
    std::vector<size_t> registrations;
};
using graph::detail::trackSourceId;
using graph::detail::trackRow;
using graph::detail::trackPolylineLength;
using graph::detail::trackFamily;
TrackInput loadTrackInput(graph::GraphDb& db, const std::filesystem::path& path,
    const std::vector<uint64_t>& sourceIds, const std::vector<std::string>& trackUuids,
    const std::atomic_bool* cancel, const util::ProgressCallback& progress = {});
} // namespace vc::graph_links::detail
