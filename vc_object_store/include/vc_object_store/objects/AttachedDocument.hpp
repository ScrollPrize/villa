#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
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
// Attach `document` to `objectUuid` as a new point collection `collectionUuid`
// named `name` whose meta.coordinates are `coordinates`. The caller's checked
// transaction expects `collectionUuid` absent and pins the object.
void attachDocument(ObjectStore&, WriteTxn&, const std::string& objectUuid, const std::string& collectionUuid,
                    const nlohmann::json& document, const DocumentLayout&, const std::string& name,
                    const nlohmann::json& coordinates);
// The document attached to `objectUuid` in `role` (its first live collection
// with geometry); nullopt when there is none.
std::optional<nlohmann::json> attachedDocument(const ObjectStore&, const Txn&, const std::string& objectUuid,
                                               const std::string& role);
// The same, read from a session's attachment.
std::optional<nlohmann::json> attachedDocument(ObjectStoreSession&, const AttachmentIdentity&,
                                               const std::string& objectUuid, const std::string& role);

// Editable point collections attached to an object in a role (e.g. a surface's
// corrections): each its points with stable ids, one annotation per point, and
// the rest of it as record meta.source.
struct AttachedCollection {
    std::string uuid, name;
    geometry::PointCollection points;
    nlohmann::json annotations = nlohmann::json::array();
    nlohmann::json source = nlohmann::json::object();
};
// The live collections attached to `objectUuid` in `role`, with their points in
// `frame` when given (else as stored).
std::vector<AttachedCollection> attachedCollections(ObjectStoreSession&, const AttachmentIdentity&,
                                                    const std::string& objectUuid, const std::string& role,
                                                    const std::optional<ViewFrame>& frame = std::nullopt);
// Make the collections attached to `objectUuid` in `role` exactly
// `collections` (by UUID) in one commit: new ones are created with
// `coordinates` and attached; changed ones get a new geometry revision whose
// remaining points keep their identity, and keep their attachment; the others
// are detached and retired. Points are in `frame` when given.
void replaceAttachedCollections(ObjectStoreSession&, const AttachmentIdentity&, const std::string& objectUuid,
                                const std::string& role, const std::vector<AttachedCollection>& collections,
                                const nlohmann::json& coordinates, const std::optional<ViewFrame>& frame = std::nullopt);

// A volume's point collections: the live point_collection objects whose
// coordinates map to `volume` (bound to it, or through a registration) and
// that are no object's attached data (no meta.role), in its voxels.
std::vector<AttachedCollection> volumeCollections(ObjectStoreSession&, const AttachmentIdentity&, const std::string& volume);
// In one commit: `changed` (in `volume`'s voxels) are created, bound to its
// voxels, or revised keeping their points' identity; `removed` are retired.
void writeVolumeCollections(ObjectStoreSession&, const AttachmentIdentity&, const std::string& volume,
                            const std::vector<AttachedCollection>& changed, const std::vector<std::string>& removed);
}
