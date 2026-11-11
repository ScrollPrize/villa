#pragma once

#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/StoreReader.hpp>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

class QuadSurface;

namespace vc3d::spiral {

// The object-store inputs of a Spiral fit, read from the attached store as
// they are now and converted to reference-volume voxels, XYZ: the frame the
// workspace displays. The store is edited live; a fit read it once, so each
// input also says whether it still is what the fit read. The fit's side
// comes from its input manifest (object_store_inputs.json, published by the
// service as its input manifest artifact), compared by identities of the
// store: geometry revision UUIDs, the metadata values the adapter read,
// relationship record UUIDs, numbered versions and approval states.

// How one fitted input stands in the attached store now.
enum class InputState {
    Fitted,           // its geometry and the metadata the fit read are unchanged
    Edited,           // its geometry changed since the fit read it
    MetadataChanged,  // the same geometry, other metadata than the fit read
    Missing,          // deleted, or not in this store: nothing to show
};
const char* inputStateName(InputState state);

// One object as the manifest records it.
struct StoreInputPin {
    std::string uuid, kind, name, geometryRevision;
    // The metadata the adapter read (object_store_inputs.adapter_metadata);
    // null in manifests that do not record it, which then compare on
    // geometry only.
    nlohmann::json metadata;
    vc::objects::ObjectVersion reference;
};

struct FitSurface {
    StoreInputPin pin;
    InputState state = InputState::Missing;
    std::shared_ptr<QuadSurface> surface;  // current geometry; empty when Missing
    std::string problem;  // why it is missing or unreadable; nothing is shown then
};

struct FitPointCollection {
    StoreInputPin pin;
    InputState state = InputState::Missing;
    bool absolute = false;  // meta.source.winding_is_absolute, now
    std::vector<int64_t> ids;  // stable point IDs, ascending
    std::vector<cv::Vec3d> points;
    std::vector<std::optional<double>> windings;  // wind_a annotations
    std::string problem;
    vc::objects::ObjectVersion reference;  // the exact state an edit starts from
};

struct FitFiber {
    StoreInputPin pin;
    InputState state = InputState::Missing;
    std::string family;  // "H", "V" or ""
    std::vector<cv::Vec3d> controls, line;
    std::string problem;
};

// How one relationship the fit consumed stands now, judged as the fit's
// adapter would consume it from the current heads of the fitted objects.
enum class LinkState {
    Fitted,   // consumed at the same version, approval and endpoint geometry
    Changed,  // still consumed, but any of those differ
    Dropped,  // no longer consumed: removed, rejected, inapplicable or an endpoint gone
    Added,    // consumed now among the fitted objects, but not by the fit
};
const char* linkStateName(LinkState state);

// A track is compared, never shown: a fit may read millions of them.
struct FitTrack {
    StoreInputPin pin;
    InputState state = InputState::Missing;
    std::string problem;
};

struct FitRelationship {
    std::string uuid, family;
    LinkState state = LinkState::Dropped;
    std::string a, b;  // endpoint object UUIDs, in the adapter's order
};

struct FitInputs {
    std::string digest, storeUuid, volumeUuid;
    std::vector<FitSurface> surfaces;
    // The outer shell the fit read (the store's surface tagged
    // spiral-outer-shell), apart from the patches; unset when it read none.
    std::optional<FitSurface> outerShell;
    std::vector<FitPointCollection> pointCollections;
    std::vector<FitFiber> fibers;
    std::vector<FitTrack> tracks;
    std::vector<FitRelationship> relationships;
    // Relationships among the fitted objects that a rebuild would consume
    // and the fit did not (state Added), by record UUID.
    std::vector<FitRelationship> addedRelationships;
    // Why the manifest does not describe this store, or its relationships
    // cannot be compared; an input's own problem is on its entry.
    std::vector<std::string> problems;

    std::size_t count(InputState state) const;
    std::size_t count(LinkState state) const;
    // Every input and relationship is as the fit read it.
    bool unchanged() const
    {
        return problems.empty() && (!outerShell || outerShell->state == InputState::Fitted)
            && count(InputState::Fitted) == surfaces.size() + pointCollections.size()
                                                                   + fibers.size() + tracks.size()
            && count(LinkState::Fitted) == relationships.size() && addedRelationships.empty();
    }
};

// The objects of an input manifest document ({"digest", "manifest"}).
std::vector<StoreInputPin> manifestPins(const nlohmann::json& document);

// The metadata the fit's object-store adapter reads from one object, as
// object_store_inputs.adapter_metadata builds it.
nlohmann::json adapterMetadata(const std::string& kind, const nlohmann::json& meta,
                               const std::optional<vc::objects::FiberDetail>& fiber,
                               const std::optional<vc::objects::TrackDetail>& track = std::nullopt);

// Read the inputs of `document` ({"digest", "manifest"}) from `db` at their
// current heads, each with how it differs from what the fit read, and
// compare the relationships the fit consumed with the ones those heads
// would give. A store other than the manifest's is refused outright
// (problems).
FitInputs readFitInputs(const vc::objects::StoreReader& reader,
                        const nlohmann::json& document);

// What a store commit may have changed, read without reading the rest.
struct FitInputChanges {
    // The listed objects read again, and the relationships compared again.
    FitInputs inputs;
    // Every relationship compared again: the listed ones, the fitted ones of
    // a listed object, and the ones a listed object has now.
    std::set<std::string> checkedRelationships;
};

// The incremental form of readFitInputs for a store commit: re-read only
// the manifest objects in `objectUuids` (other UUIDs are ignored) and
// re-compare only the relationships that involve them or are listed in
// `relationshipUuids`. Its cost follows the change, not the fit.
FitInputChanges readFitInputChanges(const vc::objects::StoreReader& reader,
                                    const nlohmann::json& document, const std::set<std::string>& objectUuids,
                                    const std::set<std::string>& relationshipUuids);

// Merge changes read from the same manifest into a full read.
void applyFitInputChanges(FitInputs& inputs, FitInputChanges changes);

} // namespace vc3d::spiral
