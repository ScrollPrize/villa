#include "geometry/SharedGeometry.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;
std::vector<uint8_t> readSharedPayload(sqlite3* db, const std::string& name, const std::string& hash) {
    if (readRequest) {
        Statement size(db, "SELECT length(data) FROM geometry_chunks WHERE hash=?"); size.bind(1, hash);
        if (!size.row()) throw std::runtime_error("geometry: missing shared payload");
        requestBytes(size.i64(0), &FiberRegionStats::payloadBytes);
    }
    Statement row(db, "SELECT data FROM geometry_chunks WHERE hash=?"); row.bind(1, hash);
    if (!row.row()) throw std::runtime_error("geometry: missing shared payload");
    auto* p = static_cast<const uint8_t*>(sqlite3_column_blob(row.value, 0));
    std::vector<uint8_t> bytes(p, p + sqlite3_column_bytes(row.value, 0));
    if (geometryWork) geometryWork->readBytes[name] += bytes.size();
    if (geometryPayloadHash(name, bytes) != hash) throw std::runtime_error("geometry: corrupt shared payload");
    return bytes;
}
bool identityManaged(sqlite3* db,const std::string& object) {
    Statement latest(db,"SELECT json_type(manifest,'$.sequences') FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation DESC,rowid DESC LIMIT 1");
    latest.bindBlob(1,parseUuid(object)); return latest.row() && !latest.isNull(0);
}
std::string storeSharedPayload(sqlite3* db, const std::string& name, std::span<const uint8_t> bytes) {
    GeometryPhase phase("storeSharedPayload");
    const auto hash = geometryPayloadHash(name, bytes);
    Statement insert(db, "INSERT INTO geometry_chunks VALUES(?,?) ON CONFLICT(hash) DO NOTHING");
    insert.bind(1, hash); check(sqlite3_bind_blob64(insert.value, 2, bytes.data(), bytes.size(), SQLITE_TRANSIENT), db); insert.stepDone();
    if (geometryWork && sqlite3_changes(db)) geometryWork->writtenBytes[name] += bytes.size();
    Statement verify(db, "SELECT data FROM geometry_chunks WHERE hash=?"); verify.bind(1, hash); verify.row();
    if (geometryWork) geometryWork->readBytes[name] += sqlite3_column_bytes(verify.value, 0);
    if (size_t(sqlite3_column_bytes(verify.value, 0)) != bytes.size() ||
        (bytes.size() && std::memcmp(sqlite3_column_blob(verify.value, 0), bytes.data(), bytes.size())))
        throw std::runtime_error("geometry: corrupt or colliding shared payload");
    return hash;
}
}
