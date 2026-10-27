#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

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
}
