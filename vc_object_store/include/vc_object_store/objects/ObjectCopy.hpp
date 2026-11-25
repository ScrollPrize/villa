#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/Transactions.hpp"

namespace vc::objects {
// A copy of an object, for moving it between stores (as a new object, or the same one):
// its current record (kind, name, tags, metadata, kind summary) and current
// geometry. meta.copied_from records the source {store_uuid, object_uuid,
// generation, version, geometry_revision}. Sequence annotations travel in meta.source
// as at ingestion. Relationships, history, approval and lifecycle are not copied.
struct ObjectCopy {
    ObjectRecord record;
    std::optional<GeometryData> geometry;
};
// `copyUuid`: the copy's UUID (default: a new one; the original's to import it as the same object).
ObjectCopy readObjectCopy(const StoreReader&, const std::string& objectUuid,
                          const std::optional<std::string>& copyUuid = std::nullopt);
// The operations creating the copy; the request expects record.uuid absent.
std::vector<TransactionOperation> objectCopyOperations(const ObjectCopy&);
// The same inside a caller's write transaction.
std::string writeObjectCopy(ObjectStore&, WriteTxn&, const ObjectCopy&);
}
