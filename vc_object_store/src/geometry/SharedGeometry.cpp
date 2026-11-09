#include "geometry/SharedGeometry.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;
namespace {
StatementPool& externalPool(sqlite3* db)
{
    const auto found = activeStatementPools.find(db);
    if (found == activeStatementPools.end() || !found->second->externalGeometryChunk || !found->second->externalGeometryChunkSize)
        throw std::runtime_error("geometry: external chunk requires a pinned collection transaction");
    return *found->second;
}
int64_t externalSize(sqlite3* db, const std::string& hash)
{
    const auto size = externalPool(db).externalGeometryChunkSize(hash);
    if (size <= 0 || uint64_t(size) > geometryChunkSize)
        throw std::runtime_error("geometry: invalid external chunk size");
    return size;
}
}
bool hasExternalGeometryPayload(sqlite3* db,const std::string& hash)
{
    const auto found=activeStatementPools.find(db);
    return found!=activeStatementPools.end() && found->second->externalGeometryChunkSize &&
        found->second->externalGeometryChunkSize(hash)>0;
}
int64_t geometryPayloadSize(sqlite3* db, const std::string& hash)
{
    Statement row(db, "SELECT length(data) FROM geometry_chunks WHERE hash=?", parseDigest(hash));
    if (!row.row()) return externalSize(db, hash);
    return row.i64(0);
}
std::vector<uint8_t> readExternalGeometryPayload(sqlite3* db, const std::string& hash)
{
    const auto size = externalSize(db, hash);
    auto bytes = externalPool(db).externalGeometryChunk(hash);
    if (uint64_t(bytes.size()) != uint64_t(size)) throw std::runtime_error("geometry: external chunk size mismatch");
    return bytes;
}
std::vector<uint8_t> readGeometryPayload(sqlite3* db, const std::string& hash)
{
    Statement row(db, "SELECT data FROM geometry_chunks WHERE hash=?", parseDigest(hash));
    if (!row.row()) return readExternalGeometryPayload(db, hash);
    const auto* p = static_cast<const uint8_t*>(sqlite3_column_blob(row.value, 0));
    return std::vector<uint8_t>(p, p + sqlite3_column_bytes(row.value, 0));
}
std::vector<std::string> reachableGeometryMetadata(sqlite3* db, const SharedJson& manifest)
{
    std::vector<SharedJson> pending{manifest.value("sequences",SharedJson::object())};
    std::set<std::string> seen;
    std::vector<std::string> result;
    Statement present(db,"SELECT 1 FROM geometry_chunks WHERE hash=?");
    while(!pending.empty()) {
        auto value=std::move(pending.back()); pending.pop_back();
        if(value.is_structured()) {
            for(auto& child:value) pending.push_back(std::move(child));
        } else if(value.is_string()) {
            const auto& hash=value.get_ref<const std::string&>();
            if(hash.size()!=64 || !std::all_of(hash.begin(),hash.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}) || !seen.insert(hash).second) continue;
            present.reset(); bindDigest(present,1,hash);
            if(!present.row()&&!hasExternalGeometryPayload(db,hash)) continue; // Annotation text, not a stored reference.
            result.push_back(hash);
            const auto bytes=readSharedPayload(db,"@metadata",hash);
            auto nested=SharedJson::parse(bytes,nullptr,false);
            if(!nested.is_discarded()) pending.push_back(std::move(nested));
        }
    }
    return result;
}
std::vector<uint8_t> readSharedPayload(sqlite3* db, const std::string& name, const std::string& hash) {
    if (readRequest) requestBytes(geometryPayloadSize(db,hash), &RegionStats::payloadBytes);
    auto bytes = readGeometryPayload(db,hash);
    if (geometryWork) geometryWork->readBytes[name] += bytes.size();
    if (geometryPayloadHash(name, bytes) != hash) throw std::runtime_error("geometry: corrupt shared payload");
    return bytes;
}
std::string storeSharedPayload(sqlite3* db, const std::string& name, std::span<const uint8_t> bytes) {
    GeometryPhase phase("storeSharedPayload");
    const auto digest=geometryPayloadDigest(name,bytes);
    const auto hash=formatDigest(digest);
    if(hasExternalGeometryPayload(db,hash)) {
        const auto existing=readExternalGeometryPayload(db,hash);
        if(!std::equal(existing.begin(),existing.end(),bytes.begin(),bytes.end()))
            throw std::runtime_error("geometry: corrupt or colliding external shared payload");
        return hash;
    }
    execute(db, "INSERT INTO geometry_chunks VALUES(?,?) ON CONFLICT(hash) DO NOTHING", digest, bytes);
    if (geometryWork && sqlite3_changes(db)) geometryWork->writtenBytes[name] += bytes.size();
    Statement verify(db, "SELECT data FROM geometry_chunks WHERE hash=?", digest); verify.row();
    if (geometryWork) geometryWork->readBytes[name] += sqlite3_column_bytes(verify.value, 0);
    if (size_t(sqlite3_column_bytes(verify.value, 0)) != bytes.size() ||
        (bytes.size() && std::memcmp(sqlite3_column_blob(verify.value, 0), bytes.data(), bytes.size())))
        throw std::runtime_error("geometry: corrupt or colliding shared payload");
    return hash;
}
}
