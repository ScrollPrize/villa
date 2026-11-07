#pragma once
// Linking as a checked request: `link_objects` (objects/Transactions.hpp,
// LinkObjectsOp) runs an ordered plan of detectors over one read snapshot and
// publishes every link they found, together with the store's link marker, in
// one commit. It is prepared outside the writer lock (ObjectStore::prepare:
// the detection) and committed by ObjectStore::submit expecting every object
// the detectors read at the version they read, so a commit in between
// conflicts and nothing is written. Local stores and vc_object_server run
// the same code; a hosted client just submits the request.
//
// The plan, as JSON:
//   {"detectors": [{"detector": "<name>", "ct_volume": "<path or URL>", "params": {...}}, ...],
//    "marker": {...}}
// `detector` is a detector identity (ObjectLinks.hpp: surface-overlap,
// fiber-surface, track-surface, track-track, point-collection-surface);
// `ct_volume` is the CT normal source of the line–surface detectors (absent
// otherwise; their lasagna sources, `normal_x_volume`/`normal_y_volume` or
// `normal_pool`, are params, and with no source at all they link by the
// centerline); `params` holds that detector's parameters under the names
// the Python bindings use (surface_uuids, tolerance, ...). Every path or URL
// resolves where the detection runs: on the service for a served store.
// `marker` is kept as the marker record's metadata.
//
// The link marker is one object per store, kind kLinkMarkerKind at
// linkMarkerUuid(), published by every link_objects commit: its metadata is
// the plan's `marker` content plus `commit_sequence` (the commit that
// published it, which is the links' commit) and `detectors` (each step with
// its report). A client reads it like any object to decide whether the
// store's links are current. A detector that fails to run fails the whole
// request: nothing is published and the marker does not move.

#include "vc_object_store/object_links/ObjectLinks.hpp"
#include "vc_object_store/objects/Transactions.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace vc::object_links {

inline constexpr const char* kLinkMarkerKind = "link_marker";
// The marker's UUID, the same in every store.
std::string linkMarkerUuid();

struct LinkStep {
    std::string detector;
    std::string ctVolume;                            // line–surface detectors only; may be empty
    nlohmann::json params = nlohmann::json::object();
};
struct LinkPlan {
    std::vector<LinkStep> steps;
    nlohmann::json marker = nlohmann::json::object();
};
nlohmann::json linkPlanJson(const LinkPlan&);
LinkPlan linkPlanFromJson(const nlohmann::json&);
objects::LinkObjectsOp linkObjectsOperation(const LinkPlan&);

// Detector parameters from their JSON (the Python keyword names); an unknown
// key is an error. `progress` is left unset.
PointCollectionSurfaceParams pointCollectionSurfaceParamsFromJson(const nlohmann::json&);
SurfaceOverlapParams surfaceOverlapParamsFromJson(const nlohmann::json&);
TrackTrackParams trackTrackParamsFromJson(const nlohmann::json&);
FiberSurfaceParams fiberSurfaceParamsFromJson(const nlohmann::json&);
TrackSurfaceParams trackSurfaceParamsFromJson(const nlohmann::json&);
nlohmann::json detectorReportJson(const DetectorReport&);

// Register link_objects with the core library (registerOperationPreparer);
// call once from a program that submits such requests to local stores.
void enableLinkObjects();
// The preparer: runs the plan's detectors over one snapshot of `store`
// (PrepareOptions: progress, cancellation, the service's scratch directory
// for fiber strips, whether client paths are allowed) and returns the
// publication. Throws std::invalid_argument for a malformed plan,
// objects::PreparationFailed when a detector fails to run, and
// std::runtime_error when cancelled.
objects::PreparedOperation prepareLinkObjects(objects::ObjectStore& store, const objects::LinkObjectsOp& op,
                                              const objects::PrepareOptions& options);

} // namespace vc::object_links
