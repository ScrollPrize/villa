#pragma once
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace vc3d::store {
// A document attached to an object (a tool's results for it): a point_collection
// with record meta `role`, tied to the object by an "attachment" relationship
// (data on the object, never a graph link). It holds one point per record of the
// document's meta.source.list_key list, whose annotations are the records; the
// rest of the document is meta.source.document. nullopt when there is none.
inline std::optional<nlohmann::json> attachedDocument(vc::objects::ObjectStoreSession& session,
                                                      const vc::objects::AttachmentIdentity& attachment,
                                                      const std::string& objectUuid, const std::string& role)
{
    std::optional<nlohmann::json> document;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        for (const auto& edge : db.attachments(txn, objectUuid, role)) {
            const auto uuid = edge.aUuid == objectUuid ? edge.bUuid : edge.aUuid;
            const auto object = db.objectByUuid(txn, uuid);
            const auto head = object && !object->retired ? db.currentGeometry(txn, uuid) : std::nullopt;
            if (!head) continue;
            const auto source = nlohmann::json::parse(object->meta).value("source", nlohmann::json::object());
            document = source.value("document", nlohmann::json::object());
            (*document)[source.value("list_key", std::string("records"))] =
                nlohmann::json::parse(db.geometryPointAnnotations(txn, head->uuid, "points"));
            return;
        }
    });
    return document;
}
}
