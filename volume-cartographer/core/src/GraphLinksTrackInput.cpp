// Shared track identity/selection adapter for all graph-link detectors.
#include "GraphLinksTrackInput.hpp"
#include "vc/core/util/Uuid.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
namespace vc::graph_links::detail {
namespace {
void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("track loading cancelled");
}
}
uint64_t trackSourceId(const std::string& key)
{
    uint64_t id;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), id);
    if (error != std::errc{} || end != key.data() + key.size() || std::to_string(id) != key)
        throw std::runtime_error("track external_key must be a canonical decimal source ID: " + key);
    return id;
}
double trackPolylineLength(const PackedTrackStore& store, size_t row, const std::atomic_bool* cancel)
{
    const auto begin = store.offsets()[row], end = store.offsets()[row+1];
    const auto coordinates = store.coordinates();
    double length = 0;
    for (auto i = begin + 1; i < end; ++i) {
        if ((i & 1023) == 0) checkCancelled(cancel);
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
std::string trackFamily(const PackedTrackStore& store, size_t row)
{
    const auto family = store.families()[row];
    return family == 0 ? "horizontal" : family == 1 ? "vertical" : "";
}
size_t trackRow(const PackedTrackStore& store, const graph::ObjectRecord& object)
{
    if (!object.track) throw std::runtime_error("missing track detail: " + object.uuid);
    const auto id = trackSourceId(object.track->externalKey);
    const auto found = std::lower_bound(store.sourceIds().begin(), store.sourceIds().end(), id);
    if (found == store.sourceIds().end() || *found != id)
        throw std::runtime_error("source ID " + std::to_string(id) + " missing from " + object.track->datasetRef);
    const auto row = found - store.sourceIds().begin();
    const auto name = trackFamily(store, row);
    if (!object.track->family.empty() && object.track->family != name)
        throw std::runtime_error("registered family differs from packed track " + std::to_string(id));
    if (object.track->pointCount != 0 && object.track->pointCount != store.offsets()[row+1] - store.offsets()[row])
        throw std::runtime_error("registered point count differs from packed track " + std::to_string(id));
    return static_cast<size_t>(row);
}
TrackInput loadTrackInput(graph::GraphDb& db, const std::filesystem::path& path,
    const std::vector<uint64_t>& sourceIds, const std::vector<std::string>& trackUuids,
    const std::atomic_bool* cancel, const util::ProgressCallback& progress)
{
    checkCancelled(cancel);
    util::reportProgress(progress, "opening track dataset");
    TrackInput result;
    std::set<std::string> requested(trackUuids.begin(), trackUuids.end());
    std::vector<graph::ObjectRecord> registered;
    {
        auto read = db.read();
        registered = db.objectsByKind(read, "track");
    }
    const auto packedPath = resolvePackedTrackStore(path);
    PackedTrackStore store(packedPath);
    std::string dataset = packedPath.string();
    // Preserve existing identities when a prior caller used a relative
    // packed path, or a symlink, for this same geometry.
    std::set<std::string> matchingRefs;
    std::map<std::string, bool> sameDataset;
    for (const auto& object : registered) {
        if (!object.track) continue;
        const auto& ref = object.track->datasetRef;
        if (!sameDataset.contains(ref)) {
            const std::filesystem::path path(ref);
            std::error_code error;
            const auto resolved = std::filesystem::weakly_canonical(
                path.is_absolute() ? path : db.path().parent_path() / path, error);
            sameDataset[ref] = !error && resolved == packedPath;
        }
        if (sameDataset.at(ref)) matchingRefs.insert(ref);
    }
    if (matchingRefs.size() > 1)
        throw std::runtime_error("this packed dataset is registered under multiple dataset_ref aliases");
    if (!matchingRefs.empty()) dataset = *matchingRefs.begin();
    std::map<uint64_t, graph::ObjectRecord> existing;
    for (auto& object : registered)
        if (object.track && object.track->datasetRef == dataset)
            existing.emplace(trackSourceId(object.track->externalKey), std::move(object));
    const std::set<uint64_t> selectedIds(sourceIds.begin(), sourceIds.end());
    size_t selectedCount = 0;
    result.datasetRef = dataset;
    auto& objects = result.objects;
    for (size_t row = 0; row < store.sourceIds().size(); ++row) {
        checkCancelled(cancel);
        if ((row & 16383) == 0) util::reportProgress(progress, "selecting tracks", row, store.sourceIds().size());
        const auto id = store.sourceIds()[row];
        if (!selectedIds.empty() && !selectedIds.contains(id)) continue;
        const auto found = existing.find(id);
        if (!requested.empty() && (found == existing.end() || !requested.contains(found->second.uuid))) continue;
        ++selectedCount;
        if (found != existing.end()) {
            objects.push_back(std::move(found->second));
            continue;
        }
        graph::ObjectRecord object;
        object.uuid = vc::util::newUuid();
        object.kind = "track";
        object.name = std::to_string(id);
        object.storageKind = "dataset";
        object.storageRef = dataset;
        object.dataGeneration = 1;
        graph::TrackDetail detail;
        detail.datasetRef = dataset;
        detail.externalKey = object.name;
        detail.family = trackFamily(store, row);
        detail.pointCount = store.offsets()[row+1] - store.offsets()[row];
        object.track = std::move(detail);
        result.registrations.push_back(objects.size());
        objects.push_back(std::move(object));
    }
    if (!selectedIds.empty() && selectedCount != selectedIds.size())
        throw std::runtime_error("some requested source IDs are missing or outside the UUID selection");
    if (!requested.empty() && objects.size() != requested.size())
        throw std::runtime_error("some requested track UUIDs were not found in the selected dataset");
    util::reportProgress(progress, "selecting tracks", store.sourceIds().size(), store.sourceIds().size());
    result.packedPath = packedPath;
    return result;
}
} // namespace vc::graph_links::detail
