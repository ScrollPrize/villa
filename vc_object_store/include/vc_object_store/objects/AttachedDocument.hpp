#pragma once
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
class ObjectStoreSession;
struct AttachmentIdentity;
// A document attached to an object (e.g. a tool's results for a surface): a
// point_collection with record meta `role`, tied to the object by an
// "attachment" relationship (data on the object, never a graph link). Each
// record of the document's `listKey` list is one point, at the record's
// `positionKey` xyz ((-1,-1,-1) without a finite one), annotated with the
// record; the rest of the document is meta.source.document.
struct DocumentLayout {
    std::string role, listKey, positionKey;
};
// Attach `document` to `objectUuid` (at its current version) as a new point
// collection named `name` whose meta.coordinates are `coordinates`, in one
// checked request; returns the collection's UUID.
std::string attachDocument(ObjectStoreSession&, const AttachmentIdentity&, const std::string& objectUuid,
                           const nlohmann::json& document, const DocumentLayout&, const std::string& name,
                           const nlohmann::json& coordinates);
// The document attached to `objectUuid` in `role` (its first live collection
// with geometry); nullopt when there is none.
std::optional<nlohmann::json> attachedDocument(const StoreReader&, const std::string& objectUuid,
                                               const std::string& role);
// Exact owner selection; attached evidence must be applicable to that state.
std::optional<nlohmann::json> attachedDocument(const StoreReader&, const ObjectVersion&, const std::string& role);
// The same, read from a session's attachment.
std::optional<nlohmann::json> attachedDocument(ObjectStoreSession&, const AttachmentIdentity&,
                                               const std::string& objectUuid, const std::string& role);
std::optional<nlohmann::json> attachedDocument(ObjectStoreSession&, const AttachmentIdentity&,
                                               const ObjectVersion&, const std::string& role);

// Editable point collections attached to an object in a role (e.g. a surface's
// corrections): each its points with stable ids, one annotation per point, and
// the rest of it as record meta.source.
struct AttachedCollection {
    std::string uuid, name;
    geometry::PointCollection points;
    nlohmann::json annotations = nlohmann::json::array();
    nlohmann::json source = nlohmann::json::object();
    std::optional<ObjectVersion> reference;
    // Mapping used when loading editable coordinates; checked under the write lock.
    std::optional<CheckVolumeMappingOp> mapping;
    // Runtime allocation floor, including IDs removed from retained revisions.
    int64_t nextPointId = 0;
};
// The live collections attached to `objectUuid` in `role`, with their points in
// `frame` when given (else as stored).
struct AttachedCollectionsSnapshot {
    ObjectVersion owner;
    StoreStatus snapshot;
    std::string role;
    std::optional<ViewFrame> frame;
    std::vector<AttachedCollection> collections;
};
AttachedCollectionsSnapshot attachedCollections(ObjectStoreSession&, const AttachmentIdentity&,
                                                    const std::string& objectUuid, const std::string& role,
                                                    const std::optional<ViewFrame>& frame = std::nullopt);
// Make the collections attached to `objectUuid` in `role` exactly
// `collections` (by UUID) in one commit: new ones are created with
// `coordinates` and attached; changed ones get a new geometry revision whose
// remaining points keep their identity, and keep their attachment; the others
// are detached and retired. The loaded snapshot guards membership, complete
// collection identities and coordinate mappings. On success it is repinned
// from the receipt, ready for another edit; on conflict it stays unchanged.
// Membership uses the store's commit sequence, so unrelated commits also
// require reloading. An inherited owner's unchanged version is not sufficient.
TransactionReceipt replaceAttachedCollections(ObjectStoreSession&, const AttachmentIdentity&, AttachedCollectionsSnapshot&,
                                const std::vector<AttachedCollection>& collections, const nlohmann::json& coordinates);

// A volume's point collections: the live point_collection objects whose
// coordinates map to `volume`'s volume (bound to it, or through a
// registration) and that are no object's attached data (no meta.role), in
// `volume`.
std::vector<AttachedCollection> volumeCollections(const StoreReader&, const VolumeVoxels& volume);
// The same, read from a session's attachment.
std::vector<AttachedCollection> volumeCollections(ObjectStoreSession&, const AttachmentIdentity&, const VolumeVoxels& volume);
// The live point collections `uuids` (any role), in `volume` and in that
// order. Throws for one that is missing or does not map to the volume.
std::vector<AttachedCollection> pointCollections(const StoreReader&, const std::vector<std::string>& uuids,
                                                 const VolumeVoxels& volume);
// In one commit: `changed` (in `volume`) are created, bound to `volume`, or
// revised keeping their points' identity; `removed` are retired. `expected`
// pins objects at the versions an edit started from (null: the edit creates
// it); the commit refuses (TransactionConflict) when one has moved on.
// Unpinned objects are expected at their current versions. The receipt
// carries each written object's version.
TransactionReceipt writeVolumeCollections(ObjectStoreSession&, const AttachmentIdentity&, const VolumeVoxels& volume,
                                          const std::vector<AttachedCollection>& changed,
                                          const std::vector<std::string>& removed,
                                          const RecordSelection& expected = {},
                                          const std::vector<ObjectVersion>& references = {});
}
