#pragma once
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects::detail {
// Storage representation of one immutable geometry reference. The source is
// private to the geometry backend; consumers receive GeometryRevision/readers.
struct GeometryRecord {
    sqlite3* source = nullptr;
    bool archived = false;
    GeometryRevision revision;
    std::string manifest;
    nlohmann::json descriptor;
};
inline constexpr const char* geometryColumns =
    "uuid,object_uuid,parent_uuid,content_hash,manifest,object_version,created_by,created_at,created_host";
GeometryRecord decodeGeometryRecord(const Statement&,int column=0,bool verify=true);
std::optional<GeometryRecord> resolveGeometryRecord(sqlite3*, const std::string& reference);
std::vector<std::optional<GeometryRecord>> resolveGeometryRecords(sqlite3*, std::span<const std::string>,bool trustedCurrent=false);
}
