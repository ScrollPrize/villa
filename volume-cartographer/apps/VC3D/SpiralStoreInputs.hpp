#pragma once

#include <vc_object_store/objects/ObjectStore.hpp>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class QuadSurface;

namespace vc3d::spiral {

// The object-store inputs a Spiral fit froze, read back at the revisions its
// input manifest pins (object_store_inputs.json, published by the service as
// its input manifest artifact) and converted to reference-volume voxels, XYZ:
// the frame the workspace displays. Pins are resolved by geometry revision
// UUIDs, which every replica of a store shares; the manifest's object
// revision UUIDs belong to the service's replica and are never used here.

// How the local store holds one pinned input.
enum class PinState {
    Current,          // its current geometry and fit metadata are the pinned ones
    Edited,           // the pinned geometry is history: the object changed since
    MetadataChanged,  // the same geometry, other metadata than the fit read
    Missing,          // this store does not hold the pinned geometry revision
};
const char* pinStateName(PinState state);

struct StoreInputPin {
    std::string uuid, kind, name, geometryRevision;
    std::string metadataSha256;  // empty in manifests from before the hashes
    double scaleFactor = 1.0;    // stored units per reference-volume voxel
};

struct PinnedSurface {
    StoreInputPin pin;
    PinState state = PinState::Missing;
    std::shared_ptr<QuadSurface> surface;  // empty when Missing
};

struct PinnedPointCollection {
    StoreInputPin pin;
    PinState state = PinState::Missing;
    // False when this store lacks the metadata the fit read (no revision of
    // the pinned geometry has the pinned metadata hash): the role below would
    // be guessed, so the collection carries no points and is not shown.
    bool fittedMetadata = false;
    bool absolute = false;  // meta.source.winding_is_absolute, as fitted
    std::vector<int64_t> ids;  // stable point IDs, ascending
    std::vector<cv::Vec3d> points;
    std::vector<std::optional<double>> windings;  // wind_a annotations
};

struct PinnedFiber {
    StoreInputPin pin;
    PinState state = PinState::Missing;
    std::string family;  // "H", "V" or "" as fitted; "" when that metadata is not held
    std::vector<cv::Vec3d> controls, line;
};

// How the local store holds one relationship the fit consumed, judged as
// the fit's adapter would consume it today from the fitted objects' heads.
enum class LinkState {
    Current,  // consumed with the same anchors, metadata, approval and endpoint geometry
    Changed,  // still consumed, but any of those differ
    Dropped,  // no longer consumed: removed, rejected, inapplicable or an endpoint gone
};
const char* linkStateName(LinkState state);

struct PinnedRelationship {
    std::string uuid, family;
    LinkState state = LinkState::Dropped;
};

struct PinnedStoreInputs {
    std::string digest, storeUuid, volumeUuid;
    std::vector<PinnedSurface> surfaces;
    std::vector<PinnedPointCollection> pointCollections;
    std::vector<PinnedFiber> fibers;
    // The relationships the fit consumed, by record UUID (shared by replicas).
    std::vector<PinnedRelationship> relationships;
    // Relationships among the fitted objects that a rebuild would consume
    // and the fit did not.
    std::size_t addedRelationships = 0;
    // Why the manifest does not describe this store, or a pin it cannot read.
    std::vector<std::string> problems;

    std::size_t count(PinState state) const;
    std::size_t count(LinkState state) const;
    // Every pinned input is held by this store at its pinned geometry.
    bool resolved() const { return problems.empty() && count(PinState::Missing) == 0; }
};

// The pins of an input manifest document ({"digest", "manifest"}).
std::vector<StoreInputPin> manifestPins(const nlohmann::json& manifest);

// json.dumps(value, sort_keys=True, separators=(",", ":")): the canonical
// text the Python adapter hashes (ASCII escapes, Python's float repr).
std::string pythonCanonicalJson(const nlohmann::json& value);

// SHA-256 of the metadata the fit's object-store adapter reads from one
// object, computed exactly as object_store_inputs.adapter_metadata and
// content_sha256 do (canonical JSON: sorted keys, no spaces, ASCII escapes).
std::string adapterMetadataSha256(const std::string& kind, const nlohmann::json& meta,
                                  const std::optional<vc::objects::FiberDetail>& fiber);

// The content hash of a consumed relationship (StoredRelationship.provenance's
// content_sha256): its anchors in the manifest's endpoint order, and its
// metadata.
std::string relationshipContentSha256(const vc::objects::Anchor& a, const vc::objects::Anchor& b,
                                      const nlohmann::json& meta);

// Read every pinned input of `document` ({"digest", "manifest"}) from `db`.
// A store other than the manifest's is refused outright (problems); a pin
// the store lacks is Missing and carries no geometry rather than falling back
// to a current head. The consumed relationships are compared with the ones
// the store's current heads of the fitted objects would give.
PinnedStoreInputs readPinnedStoreInputs(const vc::objects::ObjectStore& db,
                                        const vc::objects::Txn& txn,
                                        const nlohmann::json& document);

} // namespace vc3d::spiral
