#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// A copy of an object, for moving it between stores (as a new object, or the same one):
// its current record (kind, name, tags, metadata, kind summary) and current
// geometry. meta.copied_from records the source {store_uuid, object_uuid,
// object_revision, geometry_revision}. Sequence annotations travel in meta.source
// as at ingestion. Relationships, history, approval and lifecycle are not copied.
struct ObjectCopy {
    ObjectRecord record;
    std::optional<GeometryData> geometry;
};
// `copyUuid`: the copy's UUID (default: a new one; the original's to import it as the same object).
ObjectCopy readObjectCopy(const ObjectStore&, const Txn&, const std::string& objectUuid,
                          const std::optional<std::string>& copyUuid = std::nullopt);
// Inserts the copy; the caller's checked transaction expects record.uuid to be absent.
std::string writeObjectCopy(ObjectStore&, WriteTxn&, const ObjectCopy&);
}
