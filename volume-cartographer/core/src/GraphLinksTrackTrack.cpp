#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/PackedTrackStore.hpp"
#include "vc/core/util/TrackCrossings.hpp"
#include "vc/core/util/Uuid.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace vc::graph_links {
namespace {
struct Cancelled {};
void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load(std::memory_order_relaxed)) throw Cancelled{};
}
uint64_t sourceId(const std::string& key)
{
    uint64_t id;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), id);
    if (error != std::errc{} || end != key.data() + key.size() || std::to_string(id) != key)
        throw std::runtime_error("track external_key must be a canonical decimal source ID: " + key);
    return id;
}
}

DetectorReport linkTracksToTracks(graph::GraphDb& db, const TrackTrackParams& params,
                                  const std::atomic_bool* cancel)
{
    track_crossings::Parameters crossingParams{
        params.minAngleDegrees, params.tangentRadiusVoxels, params.clusterIndexRadius};
    track_crossings::validateParameters(crossingParams);
    if (params.workers < 1 || params.edgeKind.empty())
        throw std::invalid_argument("track linking requires positive workers and a nonempty edge kind");

    if (params.tracksPath.empty() && !params.sourceIds.empty())
        throw std::invalid_argument("source_ids requires a tracks DBM or packed directory");
    if (!params.tracksPath.empty() && !params.datasetRef.empty())
        throw std::invalid_argument("use tracksPath or datasetRef, not both");
    DetectorReport report;
    std::vector<size_t> registrations;
    std::string registrationDataset;
    std::vector<LinkCandidate> candidates;
    StaleEdgeScope scope{{params.edgeKind}, {}};
    try {
        checkCancelled(cancel);
        std::map<std::string, std::vector<graph::ObjectRecord>> datasets;
        std::set<std::string> requested(params.trackUuids.begin(), params.trackUuids.end());
        std::vector<graph::ObjectRecord> registered;
        {
            auto read = db.read();
            registered = db.objectsByKind(read, "track");
        }
        if (!params.tracksPath.empty()) {
            const auto packedPath = resolvePackedTrackStore(params.tracksPath);
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
                    existing.emplace(sourceId(object.track->externalKey), std::move(object));
            const std::set<uint64_t> selectedIds(params.sourceIds.begin(), params.sourceIds.end());
            size_t selectedCount = 0;
            registrationDataset = dataset;
            auto& objects = datasets[dataset];
            for (size_t row = 0; row < store.sourceIds().size(); ++row) {
                checkCancelled(cancel);
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
                const auto family = store.families()[row];
                detail.family = family == 0 ? "horizontal" : family == 1 ? "vertical" : "";
                detail.pointCount = store.offsets()[row+1] - store.offsets()[row];
                object.track = std::move(detail);
                registrations.push_back(objects.size());
                objects.push_back(std::move(object));
            }
            if (!selectedIds.empty() && selectedCount != selectedIds.size())
                throw std::runtime_error("some requested source IDs are missing or outside the UUID selection");
            report.objectsConsidered = static_cast<int64_t>(objects.size());
        } else {
            for (auto& object : registered) {
                if (!requested.empty() && !requested.contains(object.uuid)) continue;
                if (!params.datasetRef.empty()
                    && (!object.track || object.track->datasetRef != params.datasetRef)) continue;
                ++report.objectsConsidered;
                if (!object.track || object.track->datasetRef.empty())
                    throw std::runtime_error("track has no dataset_ref: " + object.uuid);
                datasets[object.track->datasetRef].push_back(std::move(object));
            }
        }
        std::vector<graph::ObjectRecord>().swap(registered);
        if (!requested.empty() && static_cast<size_t>(report.objectsConsidered) != requested.size())
            throw std::runtime_error("some requested track UUIDs were not found in the selected dataset");
        if (!params.datasetRef.empty() && datasets.empty())
            throw std::runtime_error("no registered tracks for dataset_ref: " + params.datasetRef);

        for (auto& [dataset, objects] : datasets) {
            checkCancelled(cancel);
            const std::filesystem::path path(dataset);
            const auto resolved = path.is_absolute() ? path : db.path().parent_path() / path;
            PackedTrackStore store(resolved);
            std::map<uint64_t, const graph::ObjectRecord*> byId;
            for (const auto& object : objects) {
                const auto id = sourceId(object.track->externalKey);
                if (!byId.emplace(id, &object).second)
                    throw std::runtime_error("duplicate registered track source ID");
            }
            const bool wholeStore = byId.size() == store.sourceIds().size();
            std::vector<int32_t> coordinateBuffer;
            std::vector<int64_t> offsets{0};
            std::vector<uint64_t> sourceIds;
            std::vector<int8_t> families;
            std::vector<const graph::ObjectRecord*> rows;
            for (const auto& [id, object] : byId) {
                checkCancelled(cancel);
                const auto found = std::lower_bound(store.sourceIds().begin(), store.sourceIds().end(), id);
                if (found == store.sourceIds().end() || *found != id)
                    throw std::runtime_error("source ID " + std::to_string(id) + " missing from " + dataset);
                const auto row = found - store.sourceIds().begin();
                const int64_t begin = store.offsets()[row], end = store.offsets()[row+1];
                const int8_t family = store.families()[row];
                const std::string familyName = family == 0 ? "horizontal" : family == 1 ? "vertical" : "";
                if (!object->track->family.empty() && object->track->family != familyName)
                    throw std::runtime_error("registered family differs from packed track " + std::to_string(id));
                if (object->track->pointCount != 0 && object->track->pointCount != end - begin)
                    throw std::runtime_error("registered point count differs from packed track " + std::to_string(id));
                if (end - begin > std::numeric_limits<int32_t>::max()
                    || uint64_t(offsets.back()) + uint64_t(end - begin) > std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error("selected tracks exceed crossing kernel index limits");
                if (!wholeStore && end != begin)
                    coordinateBuffer.insert(coordinateBuffer.end(), store.coordinates().begin() + begin * 3,
                                       store.coordinates().begin() + end * 3);
                offsets.push_back(offsets.back() + end - begin);
                sourceIds.push_back(id);
                families.push_back(family);
                rows.push_back(object);
                scope.objectUuids.push_back(object->uuid);
            }
            const std::span<const int32_t> coordinates = wholeStore
                ? store.coordinates() : std::span<const int32_t>(coordinateBuffer);
            std::vector<uint64_t> packed(coordinates.size() / 3);
            for (size_t i = 0; i < packed.size(); ++i) {
                if (i % 500000 == 0) checkCancelled(cancel);
                uint64_t key = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const int32_t value = coordinates[3*i+axis];
                    if (value < 0 || value >= (1 << 20))
                        throw std::runtime_error("track voxel coordinates must be in [0, 2^20)");
                    key = (key << 20) | uint64_t(value);
                }
                packed[i] = key;
            }
            auto progress = [cancel](const char*, uint64_t, uint64_t) { checkCancelled(cancel); };
            auto order = track_crossings::parallel_argsort(packed, params.workers, progress);
            auto events = track_crossings::scan_crossing_events(
                coordinates, offsets, families, packed, order, params.workers, progress, crossingParams);
            std::vector<uint64_t>().swap(packed);
            std::vector<uint32_t>().swap(order);
            const auto crossings = track_crossings::consolidate_crossing_events(
                events, coordinates, offsets, sourceIds, params.workers, progress, crossingParams);
            for (size_t row = 0; row < rows.size(); ++row) {
                checkCancelled(cancel);
                for (auto slot = crossings.offsets[row]; slot < crossings.offsets[row+1]; ++slot) {
                    const auto partner = crossings.partners[slot];
                    if (partner <= static_cast<int64_t>(row)) continue;
                    LinkCandidate candidate;
                    candidate.kind = params.edgeKind;
                    candidate.aUuid = rows[row]->uuid;
                    candidate.bUuid = rows[partner]->uuid;
                    const auto localA = crossings.self_local[slot], localB = crossings.partner_local[slot];
                    auto anchor = [&](int32_t track, int32_t local) {
                        const auto point = offsets[track] + local;
                        const auto tangent = track_crossings::track_tangent(
                            coordinates.data(), offsets.data(), track, local, params.tangentRadiusVoxels);
                        graph::Anchor result;
                        result.index = local;
                        result.position = {double(coordinates[3*point+2]), double(coordinates[3*point+1]),
                                           double(coordinates[3*point])};
                        result.direction = {tangent.x, tangent.y, tangent.z};
                        return result;
                    };
                    candidate.aAnchor = anchor(static_cast<int32_t>(row), localA);
                    candidate.bAnchor = anchor(partner, localB);
                    const auto partnerBegin = crossings.offsets[partner];
                    const auto partnerEnd = crossings.offsets[partner+1];
                    const auto reverse = std::lower_bound(crossings.partners.begin() + partnerBegin,
                        crossings.partners.begin() + partnerEnd, static_cast<int32_t>(row));
                    const auto reverseSlot = reverse - crossings.partners.begin();
                    candidate.meta = nlohmann::json{
                        {"track_crossing", {
                            {"dataset_ref", dataset}, {"coordinate_order", "xyz"},
                            {"coordinate_units", "dataset_voxels"},
                            {"min_angle_degrees", params.minAngleDegrees},
                            {"tangent_radius_voxels", params.tangentRadiusVoxels},
                            {"cluster_index_radius", params.clusterIndexRadius},
                            {"clearance_voxels", crossings.clearances[slot]},
                            // UUID keys survive graph endpoint canonicalization.
                            {"endpoints", {
                                {candidate.aUuid, {{"source_id", std::to_string(sourceIds[row])},
                                                  {"arclength_voxels", crossings.positions[slot]}}},
                                {candidate.bUuid, {{"source_id", std::to_string(sourceIds[partner])},
                                                  {"arclength_voxels", crossings.positions[reverseSlot]}}}
                            }}
                        }}}.dump();
                    candidates.push_back(std::move(candidate));
                }
            }
        }
        checkCancelled(cancel);
        auto transaction = db.write();
        for (const auto row : registrations) {
            checkCancelled(cancel);
            db.upsertObject(transaction, datasets.at(registrationDataset).at(row));
        }
        const auto committed = commitCandidates(db, transaction, kTrackTrackDetector, 1, candidates,
                                                 params.replaceStalePending, &scope);
        checkCancelled(cancel);
        transaction.commit();
        report.candidatesFound = committed.candidatesFound;
        report.edgesCreated = committed.edgesCreated;
        report.edgesSkippedExisting = committed.edgesSkippedExisting;
        report.staleRemoved = committed.staleRemoved;
        report.problems = committed.problems;
    } catch (const Cancelled&) {
        report.problems.push_back("track-track linking cancelled; no edges changed");
    } catch (const std::exception& error) {
        report.problems.push_back(std::string("track-track linking failed; no edges changed: ") + error.what());
    }
    return report;
}
} // namespace vc::graph_links
