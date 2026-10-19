#include "vc/core/object_links/ObjectLinks.hpp"
#include "vc/core/util/TrackCrossings.hpp"
#include "linking/StoredGeometry.hpp"
#include "formats/TrackInput.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace vc::object_links {
namespace {
struct Cancelled {};
void checkCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load(std::memory_order_relaxed)) throw Cancelled{};
}

}

DetectorReport linkTracksToTracks(objects::ObjectStore& db, const TrackTrackParams& params,
                                  const std::atomic_bool* cancel)
{
    track_crossings::Parameters crossingParams{
        params.minAngleDegrees, params.tangentRadiusVoxels, params.clusterIndexRadius};
    track_crossings::validateParameters(crossingParams);
    if (params.workers < 1 || params.edgeKind.empty())
        throw std::invalid_argument("track linking requires positive workers and a nonempty edge kind");

    const auto& progressCallback = params.progress;
    util::reportProgress(progressCallback, "loading track registrations");
    DetectorReport report;
    std::vector<LinkCandidate> candidates;
    StaleEdgeScope scope{{params.edgeKind}, {}};
    try {
        checkCancelled(cancel);
        std::map<std::string, std::vector<objects::ObjectRecord>> datasets;
        detail::StoredGeometry inputs(db);
        for (auto& object : inputs.select("track", params.trackUuids)) {
            if (!params.datasetRef.empty() && (!object.track || object.track->datasetRef != params.datasetRef)) {
                if (!params.trackUuids.empty()) throw std::runtime_error("selected track is outside dataset_ref");
                continue;
            }
            ++report.objectsConsidered;
            if (!object.track || object.track->datasetRef.empty())
                throw std::runtime_error("track has no dataset_ref: " + object.uuid);
            datasets[object.track->datasetRef].push_back(std::move(object));
        }
        if (!params.datasetRef.empty() && datasets.empty())
            throw std::runtime_error("no registered tracks for dataset_ref: " + params.datasetRef);

        for (auto& [dataset, objects] : datasets) {
            checkCancelled(cancel);
            std::map<uint64_t, objects::ObjectRecord*> byId;
            for (auto& object : objects) {
                const auto id = objects::detail::trackSourceId(object.track->externalKey);
                if (!byId.emplace(id, &object).second)
                    throw std::runtime_error("duplicate registered track source ID");
            }
            std::vector<int32_t> coordinateBuffer;
            std::vector<int64_t> offsets{0};
            std::vector<uint64_t> sourceIds;
            std::vector<int8_t> families;
            std::vector<const objects::ObjectRecord*> rows;
            size_t loaded = 0;
            for (const auto& [id, object] : byId) {
                util::reportProgress(progressCallback, "loading track geometry", loaded++, byId.size());
                checkCancelled(cancel);
                const auto geometry = inputs.load(*object);
                const auto xyz = geometry.values<int32_t>("points");
                const auto familyName = nlohmann::json::parse(geometry.revision.attributes).value("family", std::string());
                if (!familyName.empty() && familyName != "horizontal" && familyName != "vertical")
                    throw std::runtime_error("invalid stored track family");
                const int8_t family = familyName == "horizontal" ? 0 : familyName == "vertical" ? 1 : -1;
                const auto count = xyz.size() / 3;
                if (count > std::numeric_limits<int32_t>::max()
                    || uint64_t(offsets.back()) + count > std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error("selected tracks exceed crossing kernel index limits");
                // The crossing kernel consumes ZYX; stored geometry is always XYZ.
                for (size_t i = 0; i < count; ++i) {
                    coordinateBuffer.push_back(xyz[3*i+2]);
                    coordinateBuffer.push_back(xyz[3*i+1]);
                    coordinateBuffer.push_back(xyz[3*i]);
                }
                offsets.push_back(offsets.back() + static_cast<int64_t>(count));
                sourceIds.push_back(id);
                families.push_back(family);
                rows.push_back(object);
                scope.objectUuids.push_back(object->uuid);
            }
            util::reportProgress(progressCallback, "loading track geometry", byId.size(), byId.size());
            const std::span<const int32_t> coordinates(coordinateBuffer);
            std::vector<uint64_t> packed(coordinates.size() / 3);
            for (size_t i = 0; i < packed.size(); ++i) {
                if (i % 500000 == 0) {
                    checkCancelled(cancel);
                    util::reportProgress(progressCallback, "packing track voxels", i, packed.size());
                }
                uint64_t key = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const int32_t value = coordinates[3*i+axis];
                    if (value < 0 || value >= (1 << 20))
                        throw std::runtime_error("track voxel coordinates must be in [0, 2^20)");
                    key = (key << 20) | uint64_t(value);
                }
                packed[i] = key;
            }
            util::reportProgress(progressCallback, "packing track voxels", packed.size(), packed.size());
            auto progress = [&](const char* stage, uint64_t done, uint64_t total) {
                checkCancelled(cancel);
                util::reportProgress(progressCallback, stage, done, total);
            };
            auto order = track_crossings::parallel_argsort(packed, params.workers, progress);
            auto events = track_crossings::scan_crossing_events(
                coordinates, offsets, families, packed, order, params.workers, progress, crossingParams);
            std::vector<uint64_t>().swap(packed);
            std::vector<uint32_t>().swap(order);
            const auto crossings = track_crossings::consolidate_crossing_events(
                events, coordinates, offsets, sourceIds, params.workers, progress, crossingParams);
            for (size_t row = 0; row < rows.size(); ++row) {
                util::reportProgress(progressCallback, "preparing crossing links", row, rows.size());
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
                        objects::Anchor result;
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
        const auto committed = detail::commitStoredCandidates(db, inputs, transaction, kTrackTrackDetector, 1, candidates,
                                                 params.replaceStalePending, &scope, progressCallback);
        checkCancelled(cancel);
        util::reportProgress(progressCallback, "committing links");
        transaction.commit();
        report.candidatesFound = committed.candidatesFound;
        report.edgesCreated = committed.edgesCreated;
        report.edgesUpdated = committed.edgesUpdated;
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
} // namespace vc::object_links
