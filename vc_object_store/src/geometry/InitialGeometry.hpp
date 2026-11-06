#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include <nlohmann/json.hpp>

struct sqlite3;
namespace vc::objects::detail {

struct PreparedInitialGeometry {
    ObjectRecord record;
    std::string revisionUuid,contentHash,manifestText;
    nlohmann::json manifest;
};

// Internal initial-construction primitive. Input record is already normalized;
// caller owns one savepoint covering preparation, object insertion and payload
// publication. Preparation may insert shared annotation/identity payloads.
// The exact same GeometryData must remain unchanged until write completes.
PreparedInitialGeometry prepareInitialGeometry(sqlite3*,const ObjectRecord&,
    const GeometryData&,const GeometryStorageOptions& = {});
void writeInitialGeometry(ObjectStore&,WriteTxn&,const PreparedInitialGeometry&,
    const GeometryData&,int64_t objectId);

} // namespace vc::objects::detail
