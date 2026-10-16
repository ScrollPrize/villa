#include "vc/core/graph_links/GraphLinks.hpp"
#include "vc/core/util/PackedTrackStore.hpp"
#include "vc/core/util/TrackCrossings.hpp"
#include "GraphLinksTrackInput.hpp"

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
        if (params.tracksPath.empty()) {
            auto read = db.read();
            registered = db.objectsByKind(read, "track");
        }
        if (!params.tracksPath.empty()) {
            auto input = detail::loadTrackInput(db, params.tracksPath, params.sourceIds, params.trackUuids, cancel);
            registrationDataset = input.datasetRef;
            registrations = std::move(input.registrations);
            report.objectsConsidered = static_cast<int64_t>(input.objects.size());
            datasets[input.datasetRef] = std::move(input.objects);
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
            std::map<uint64_t, graph::ObjectRecord*> byId;
            for (auto& object : objects) {
                const auto id = detail::trackSourceId(object.track->externalKey);
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
                const auto row = detail::trackRow(store, *object);
                object->track->length = detail::trackPolylineLength(store, row, cancel);
                object->track->family = detail::trackFamily(store, row);
                const int64_t begin = store.offsets()[row], end = store.offsets()[row+1];
                const int8_t family = store.families()[row];
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
        for (const auto& [dataset, objects] : datasets)
            for (const auto& object : objects) {
                checkCancelled(cancel);
                db.setTrackLength(transaction, object.uuid, object.track->length, object.track->family);
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
