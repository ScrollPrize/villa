// Shared track identity/selection adapter for all graph-link detectors.
#include "linking/TrackInput.hpp"
#include "vc/core/util/Uuid.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
namespace vc::object_links::detail {
namespace {
void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("track loading cancelled");
}
}
std::set<std::string> requestedTrackUuids(const std::vector<std::string>& uuids)
{
    std::set<std::string> result;
    for (auto uuid : uuids) {
        if (!util::isUuid(uuid)) throw std::runtime_error("invalid track UUID: " + uuid);
        std::transform(uuid.begin(), uuid.end(), uuid.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        result.insert(std::move(uuid));
    }
    return result;
}
TrackInput loadTrackInput(objects::ObjectStore& db, const std::filesystem::path& path,
    const std::vector<uint64_t>& sourceIds, const std::vector<std::string>& trackUuids,
    const std::atomic_bool* cancel, const util::ProgressCallback& progress)
{
    checkCancelled(cancel);
    util::reportProgress(progress, "opening track dataset");
    TrackInput result;
    const auto requested = requestedTrackUuids(trackUuids);
    std::vector<objects::ObjectRecord> registered;
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
    std::map<uint64_t, objects::ObjectRecord> existing;
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
        objects::ObjectRecord object;
        object.uuid = vc::util::newUuid();
        object.kind = "track";
        object.name = std::to_string(id);
        object.storageKind = "dataset";
        object.storageRef = dataset;
        object.dataGeneration = 1;
        objects::TrackDetail detail;
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
} // namespace vc::object_links::detail
