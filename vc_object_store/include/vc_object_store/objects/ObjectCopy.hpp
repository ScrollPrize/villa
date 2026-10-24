#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"

namespace vc::objects {
// A copy of an object as a new object (new UUID), for moving it between stores:
// its current record (kind, name, tags, metadata, kind summary) and current
// geometry. meta.copied_from records the source {store_uuid, object_uuid,
// object_revision, geometry_revision}. Sequence annotations travel in meta.source
// as at ingestion. Relationships, history, approval and lifecycle are not copied.
struct ObjectCopy {
    ObjectRecord record;
    std::optional<GeometryData> geometry;
};
ObjectCopy readObjectCopy(const ObjectStore&, const Txn&, const std::string& objectUuid);
// Inserts the copy; the caller's checked transaction expects record.uuid to be absent.
std::string writeObjectCopy(ObjectStore&, WriteTxn&, const ObjectCopy&);
}
