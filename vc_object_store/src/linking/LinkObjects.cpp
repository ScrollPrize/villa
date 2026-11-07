#include "vc_object_store/object_links/LinkObjects.hpp"
#include "linking/Detectors.hpp"
#include "linking/StoredGeometry.hpp"
#include "vc_object_store/Uuid.hpp"

#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace vc::object_links {
using Json = nlohmann::json;

namespace {
const Json& object(const Json& j, const char* what)
{
    if (!j.is_object()) throw std::invalid_argument(std::string("link_objects: ") + what + " is a JSON object");
    return j;
}

template <class T> T number(const Json& value, const std::string& key)
{
    if (!value.is_number()) throw std::invalid_argument("link_objects: " + key + " is a number");
    return value.get<T>();
}

std::vector<std::string> uuids(const Json& value, const std::string& key)
{
    if (!value.is_array()) throw std::invalid_argument("link_objects: " + key + " is a list of UUIDs");
    std::vector<std::string> result;
    for (const auto& uuid : value) {
        if (!uuid.is_string()) throw std::invalid_argument("link_objects: " + key + " is a list of UUIDs");
        result.push_back(uuid.get<std::string>());
    }
    return result;
}

std::string text(const Json& value, const std::string& key)
{
    if (!value.is_string()) throw std::invalid_argument("link_objects: " + key + " is a string");
    return value.get<std::string>();
}

bool flag(const Json& value, const std::string& key)
{
    if (!value.is_boolean()) throw std::invalid_argument("link_objects: " + key + " is true or false");
    return value.get<bool>();
}

[[noreturn]] void unknown(const char* detector, const std::string& key)
{
    throw std::invalid_argument(std::string("link_objects: ") + detector + ": unknown parameter '" + key + "'");
}

bool setLineSurfaceParameter(LineSurfaceParams& p, const std::string& key, const Json& value)
{
    if (key == "surface_uuids") p.surfaceUuids = uuids(value, key);
    else if (key == "surface_scale") p.surfaceScale = number<double>(value, key);
    else if (key == "ct_level") p.ctLevel = number<int>(value, key);
    else if (key == "ct_to_base_scale") p.ctToBaseScale = number<double>(value, key);
    else if (key == "working_to_base_scale") p.workingToBaseScale = number<double>(value, key);
    else if (key == "proximity") p.proximity = number<double>(value, key);
    else if (key == "min_contacts") p.minContacts = number<std::size_t>(value, key);
    else if (key == "strip_spacing") p.stripSpacing = number<double>(value, key);
    else if (key == "strip_rows_each_side") p.stripRowsEachSide = number<int>(value, key);
    else if (key == "strip_march_step") p.stripMarchStep = number<double>(value, key);
    else if (key == "contact_gap") p.contactGap = number<double>(value, key);
    else if (key == "anchor_contact") p.anchorContact = number<double>(value, key);
    else if (key == "same_sheet") p.sameSheet = number<double>(value, key);
    else if (key == "adjacent_low") p.adjacentLow = number<double>(value, key);
    else if (key == "adjacent_high") p.adjacentHigh = number<double>(value, key);
    else if (key == "same_normal") p.sameNormal = number<double>(value, key);
    else if (key == "adjacent_normal") p.adjacentNormal = number<double>(value, key);
    else if (key == "grow_same") p.growSame = number<double>(value, key);
    else if (key == "grow_ratio") p.growRatio = number<double>(value, key);
    else if (key == "strip_gate_ratio") p.stripGateRatio = number<double>(value, key);
    else if (key == "strip_gate_margin") p.stripGateMargin = number<double>(value, key);
    else if (key == "strip_margin") p.stripMargin = number<double>(value, key);
    else if (key == "max_rounds") p.maxRounds = number<int>(value, key);
    else if (key == "edge_kind") p.edgeKind = text(value, key);
    else if (key == "replace_stale_pending") p.replaceStalePending = flag(value, key);
    else if (key == "strip_weights") {
        if (!value.is_array() || value.size() != p.stripWeights.size())
            throw std::invalid_argument("link_objects: strip_weights requires four values");
        for (std::size_t i = 0; i < p.stripWeights.size(); ++i) p.stripWeights[i] = number<double>(value[i], key);
    } else return false;
    return true;
}
} // namespace

PointCollectionSurfaceParams pointCollectionSurfaceParamsFromJson(const Json& json)
{
    PointCollectionSurfaceParams p;
    for (const auto& [key, value] : object(json, "params").items()) {
        if (key == "collection_uuids") p.collectionUuids = uuids(value, key);
        else if (key == "surface_uuids") p.surfaceUuids = uuids(value, key);
        else if (key == "tolerance") p.tolerance = number<double>(value, key);
        else if (key == "point_scale") p.pointScale = number<double>(value, key);
        else if (key == "surface_scale") p.surfaceScale = number<double>(value, key);
        else if (key == "window_points") p.windowPoints = number<int>(value, key);
        else if (key == "window_min_points") p.windowMinPoints = number<int>(value, key);
        else if (key == "absolute_winding") p.absoluteWinding = value.is_null() ? std::nullopt : std::optional(flag(value, key));
        else if (key == "edge_kind") p.edgeKind = text(value, key);
        else if (key == "replace_stale_pending") p.replaceStalePending = flag(value, key);
        else unknown(kPointCollectionSurfaceDetector, key);
    }
    return p;
}

SurfaceOverlapParams surfaceOverlapParamsFromJson(const Json& json)
{
    SurfaceOverlapParams p;
    for (const auto& [key, value] : object(json, "params").items()) {
        if (key == "surface_uuids") p.surfaceUuids = uuids(value, key);
        else if (key == "workers") p.workers = number<int>(value, key);
        else if (key == "tolerance") p.tolerance = number<double>(value, key);
        else if (key == "sample_spacing") p.sampleSpacing = number<double>(value, key);
        else if (key == "thinning_spacing") p.thinningSpacing = number<double>(value, key);
        else if (key == "max_correspondences") p.maxCorrespondences = number<std::size_t>(value, key);
        else if (key == "uv_inlier_tolerance") p.uvInlierTolerance = number<double>(value, key);
        else if (key == "min_inliers") p.minInliers = number<std::size_t>(value, key);
        else if (key == "min_major_spread") p.minMajorSpread = number<double>(value, key);
        else if (key == "min_minor_spread") p.minMinorSpread = number<double>(value, key);
        else if (key == "max_refit_rms") p.maxRefitRms = number<double>(value, key);
        else if (key == "min_agreement_ratio") p.minAgreementRatio = number<double>(value, key);
        else if (key == "min_overlap_fraction") p.minOverlapFraction = number<double>(value, key);
        else if (key == "ransac_confidence") p.ransacConfidence = number<double>(value, key);
        else if (key == "ransac_max_hypotheses") p.ransacMaxHypotheses = number<std::size_t>(value, key);
        else if (key == "allow_reflection") p.allowReflection = flag(value, key);
        else if (key == "edge_kind") p.edgeKind = text(value, key);
        else if (key == "replace_stale_pending") p.replaceStalePending = flag(value, key);
        else unknown(kSurfaceOverlapDetector, key);
    }
    return p;
}

TrackTrackParams trackTrackParamsFromJson(const Json& json)
{
    TrackTrackParams p;
    for (const auto& [key, value] : object(json, "params").items()) {
        if (key == "min_angle_degrees") p.minAngleDegrees = number<double>(value, key);
        else if (key == "tangent_radius_voxels") p.tangentRadiusVoxels = number<double>(value, key);
        else if (key == "cluster_index_radius") p.clusterIndexRadius = number<int>(value, key);
        else if (key == "workers") p.workers = number<int>(value, key);
        else if (key == "dataset_ref") p.datasetRef = text(value, key);
        else if (key == "track_uuids") p.trackUuids = uuids(value, key);
        else if (key == "edge_kind") p.edgeKind = text(value, key);
        else if (key == "replace_stale_pending") p.replaceStalePending = flag(value, key);
        else unknown(kTrackTrackDetector, key);
    }
    return p;
}

FiberSurfaceParams fiberSurfaceParamsFromJson(const Json& json)
{
    FiberSurfaceParams p;
    for (const auto& [key, value] : object(json, "params").items()) {
        if (key == "fiber_uuids") p.fiberUuids = uuids(value, key);
        else if (key == "cache_directory") p.cacheDirectory = text(value, key);
        else if (key == "fiber_scale") p.fiberScale = number<double>(value, key);
        else if (key == "rebuild_strips") p.rebuildStrips = flag(value, key);
        else if (!setLineSurfaceParameter(p, key, value)) unknown(kFiberSurfaceDetector, key);
    }
    return p;
}

TrackSurfaceParams trackSurfaceParamsFromJson(const Json& json)
{
    TrackSurfaceParams p;
    for (const auto& [key, value] : object(json, "params").items()) {
        if (key == "track_scale") p.trackScale = number<double>(value, key);
        else if (key == "track_uuids") p.trackUuids = uuids(value, key);
        else if (!setLineSurfaceParameter(p, key, value)) unknown(kTrackSurfaceDetector, key);
    }
    return p;
}

Json detectorReportJson(const DetectorReport& report)
{
    return {{"objects_considered", report.objectsConsidered}, {"candidates_found", report.candidatesFound},
            {"edges_created", report.edgesCreated}, {"edges_updated", report.edgesUpdated},
            {"edges_skipped_existing", report.edgesSkippedExisting}, {"stale_removed", report.staleRemoved},
            {"problems", report.problems}};
}

std::string linkMarkerUuid() { return objects::util::nameUuid("vc_link_marker"); }

Json linkPlanJson(const LinkPlan& plan)
{
    auto steps = Json::array();
    for (const auto& step : plan.steps) {
        Json j{{"detector", step.detector}, {"params", object(step.params, "params")}};
        if (!step.ctVolume.empty()) j["ct_volume"] = step.ctVolume;
        steps.push_back(std::move(j));
    }
    return {{"detectors", steps}, {"marker", object(plan.marker, "marker")}};
}

LinkPlan linkPlanFromJson(const Json& json)
{
    LinkPlan plan;
    const auto& j = object(json, "plan");
    for (const auto& [key, value] : j.items())
        if (key != "detectors" && key != "marker") throw std::invalid_argument("link_objects: unknown plan key '" + key + "'");
    const auto& steps = j.at("detectors");
    if (!steps.is_array()) throw std::invalid_argument("link_objects: detectors is a list");
    for (const auto& step : steps) {
        const auto& s = object(step, "detector step");
        LinkStep parsed;
        for (const auto& [key, value] : s.items()) {
            if (key == "detector") parsed.detector = text(value, key);
            else if (key == "ct_volume") parsed.ctVolume = text(value, key);
            else if (key == "params") parsed.params = object(value, "params");
            else throw std::invalid_argument("link_objects: unknown detector step key '" + key + "'");
        }
        if (parsed.detector.empty()) throw std::invalid_argument("link_objects: a detector step names its detector");
        plan.steps.push_back(std::move(parsed));
    }
    if (const auto marker = j.find("marker"); marker != j.end()) plan.marker = object(*marker, "marker");
    return plan;
}

objects::LinkObjectsOp linkObjectsOperation(const LinkPlan& plan) { return {linkPlanJson(plan).dump()}; }

namespace {
// What one prepared request carries from its detection to its commit.
struct Prepared {
    std::shared_ptr<detail::StoredGeometry> inputs;
    LinkPlan plan;
    std::vector<detail::Detection> detections;
};

objects::util::ProgressCallback stepProgress(const objects::util::ProgressCallback& progress, const std::string& detector)
{
    if (!progress) return {};
    return [progress, detector](const objects::util::ProgressUpdate& update) {
        const auto stage = detector + ": " + std::string(update.stage);
        progress({stage, update.completed, update.total, update.detail});
    };
}

detail::Detection detect(detail::StoredGeometry& inputs, const LinkStep& step, const objects::PrepareOptions& options)
{
    const auto progress = stepProgress(options.progress, step.detector);
    auto normals = [&] {
        if (step.ctVolume.empty())
            throw std::invalid_argument("link_objects: " + step.detector + " needs ct_volume (its normal source)");
        return std::filesystem::path(step.ctVolume);
    };
    if (step.detector == kPointCollectionSurfaceDetector) {
        auto params = pointCollectionSurfaceParamsFromJson(step.params);
        params.progress = progress;
        return detail::detectPointCollectionsToSurfaces(inputs, params, options.cancel);
    }
    if (step.detector == kSurfaceOverlapDetector) {
        auto params = surfaceOverlapParamsFromJson(step.params);
        params.progress = progress;
        return detail::detectSurfaceOverlaps(inputs, params, options.cancel);
    }
    if (step.detector == kTrackTrackDetector) {
        auto params = trackTrackParamsFromJson(step.params);
        params.progress = progress;
        return detail::detectTracksToTracks(inputs, params, options.cancel);
    }
    if (step.detector == kFiberSurfaceDetector) {
        auto params = fiberSurfaceParamsFromJson(step.params);
        params.progress = progress;
        if (!params.cacheDirectory.empty() && !options.localPaths)
            throw std::invalid_argument("link_objects: cache_directory names a path on the client; "
                                        "the service keeps its own strip cache (vc_object_server --strip-cache)");
        if (params.cacheDirectory.empty() && !options.scratchDirectory.empty()) {
            params.cacheDirectory = options.scratchDirectory / "fiber-strips";
            std::filesystem::create_directories(params.cacheDirectory);
        }
        return detail::detectFibersToSurfaces(inputs, normals(), params, options.cancel);
    }
    if (step.detector == kTrackSurfaceDetector) {
        auto params = trackSurfaceParamsFromJson(step.params);
        params.progress = progress;
        return detail::detectTracksToSurfaces(inputs, normals(), params, options.cancel);
    }
    throw std::invalid_argument("link_objects: unknown detector '" + step.detector + "'");
}

void checkCancelled(const objects::PrepareOptions& options)
{
    if (options.cancel && options.cancel->load()) throw std::runtime_error("link_objects: cancelled; nothing was written");
}
} // namespace

objects::PreparedOperation prepareLinkObjects(objects::ObjectStore& store, const objects::LinkObjectsOp& op,
                                              const objects::PrepareOptions& options)
{
    auto prepared = std::make_shared<Prepared>();
    prepared->plan = linkPlanFromJson(Json::parse(op.planJson));
    prepared->inputs = std::make_shared<detail::StoredGeometry>(store);
    objects::PreparedOperation result;
    {
        // The marker's version under the same snapshot: concurrent link runs
        // conflict on it even when they read disjoint objects.
        result.expectedObjects[linkMarkerUuid()] = store.currentVersion(prepared->inputs->txn(), linkMarkerUuid());
    }
    // A detector that could not run (a bad normal source, missing inputs,
    // ...) fails the request: no step's links are published and the marker
    // does not move. Rejected candidates are not failures; they stay in the
    // report's problems.
    auto failed = [](const std::string& detector, const std::string& why) {
        return objects::PreparationFailed("link_objects: " + detector + " failed; nothing was written" + (why.empty() ? "" : ": " + why));
    };
    for (const auto& step : prepared->plan.steps) {
        checkCancelled(options);
        auto detection = [&] {
            try {
                return detect(*prepared->inputs, step, options);
            } catch (const objects::PreparationFailed&) {
                throw;
            } catch (const std::invalid_argument&) {
                throw; // a plan error: the request itself is malformed
            } catch (const std::exception& error) {
                // Some detectors report an execution failure by returning a
                // non-publishing detection, others by throwing; both are
                // the same failure to the caller.
                if (options.cancel && options.cancel->load()) throw;
                throw failed(step.detector, error.what());
            }
        }();
        checkCancelled(options);
        if (!detection.publish) {
            std::string why;
            for (const auto& problem : detection.report.problems) why += (why.empty() ? "" : "; ") + problem;
            throw failed(step.detector, why);
        }
        prepared->detections.push_back(std::move(detection));
    }
    prepared->inputs->close(); // the snapshot's work is done; the write checks the versions
    for (const auto& [uuid, version] : prepared->inputs->expectations()) result.expectedObjects[uuid] = version;
    const auto progress = options.progress;
    result.apply = [prepared, progress](objects::ObjectStore& db, objects::WriteTxn& txn) {
        auto steps = Json::array();
        for (std::size_t i = 0; i < prepared->detections.size(); ++i) {
            const auto& detection = prepared->detections[i];
            const auto report = detail::commitDetection(db, *prepared->inputs, txn, detection,
                                                        stepProgress(progress, detection.detector));
            Json step{{"detector", prepared->plan.steps[i].detector}, {"params", prepared->plan.steps[i].params},
                      {"report", detectorReportJson(report)}};
            if (!prepared->plan.steps[i].ctVolume.empty()) step["ct_volume"] = prepared->plan.steps[i].ctVolume;
            steps.push_back(std::move(step));
        }
        objects::util::reportProgress(progress, "publishing the link marker");
        objects::ObjectRecord marker;
        marker.uuid = linkMarkerUuid();
        marker.kind = kLinkMarkerKind;
        marker.name = "link marker";
        auto meta = prepared->plan.marker;
        // The marker's own publication advances the sequence, so this is the
        // links' commit.
        meta["commit_sequence"] = db.status(txn).commitSequence + 1;
        meta["detectors"] = std::move(steps);
        marker.meta = meta.dump();
        db.upsertObject(txn, marker);
    };
    return result;
}

void enableLinkObjects()
{
    static std::once_flag once;
    std::call_once(once, [] {
        objects::registerOperationPreparer(objects::kLinkObjectsOperation,
            [](objects::ObjectStore& store, const objects::TransactionOperation& op, const objects::PrepareOptions& options) {
                return prepareLinkObjects(store, std::get<objects::LinkObjectsOp>(op), options);
            });
    });
}

} // namespace vc::object_links
