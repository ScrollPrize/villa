#include "geometry/SharedGeometry.hpp"
#include "internal/ReferenceResolver.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;
namespace {
sqlite3* payloadSource(sqlite3* db, const std::string& hash)
{
    const auto digest = parseDigest(hash);
    return resolveReference(db, [&](sqlite3* source) {
        Statement row(source, "SELECT 1 FROM geometry_chunks WHERE hash=?", digest);
        return row.row();
    });
}
}
bool hasGeometryPayload(sqlite3* db,const std::string& hash)
{
    return payloadSource(db,hash)!=nullptr;
}
int64_t geometryPayloadSize(sqlite3* db, const std::string& hash)
{
    db=payloadSource(db,hash);
    if(!db)throw std::runtime_error("geometry: missing payload reference");
    Statement row(db, "SELECT length(data) FROM geometry_chunks WHERE hash=?", parseDigest(hash));
    row.row();
    return row.i64(0);
}
std::vector<uint8_t> readGeometryPayload(sqlite3* db, const std::string& hash)
{
    db=payloadSource(db,hash);
    if(!db)throw std::runtime_error("geometry: missing payload reference");
    Statement row(db, "SELECT data FROM geometry_chunks WHERE hash=?", parseDigest(hash));
    row.row();
    const auto* p = static_cast<const uint8_t*>(sqlite3_column_blob(row.value, 0));
    return std::vector<uint8_t>(p, p + sqlite3_column_bytes(row.value, 0));
}
void visitGeometryPayloads(sqlite3* db, std::span<const std::string> hashes,
    const std::function<void(size_t, std::span<const uint8_t>)>& visitor)
{
    if (!visitor) throw std::invalid_argument("geometry: missing payload visitor");
    std::vector<bool> found(hashes.size(), false);
    size_t remaining = hashes.size();
    if (!remaining) return;
    resolveReference(db, [&](sqlite3* source) {
        std::vector<size_t> pending;
        for (size_t i = 0; i < hashes.size(); ++i) if (!found[i]) pending.push_back(i);
        constexpr size_t window = 256;
        for (size_t begin = 0; begin < pending.size(); begin += window) {
            const auto count = std::min(window, pending.size() - begin);
            std::string sql = "WITH requested(pos,hash) AS (VALUES ";
            for (size_t j = 0; j < count; ++j)
                sql += (j ? "," : "") + std::string("(") + std::to_string(pending[begin+j]) + ",?)";
            sql += ") SELECT q.pos,c.data FROM requested q JOIN geometry_chunks c ON c.hash=q.hash";
            Statement rows(source, sql);
            for (size_t j = 0; j < count; ++j) rows.bind(int(j+1), parseDigest(hashes[pending[begin+j]]));
            while (rows.row()) {
                const auto i = size_t(rows.i64(0));
                visitor(i, std::span(static_cast<const uint8_t*>(sqlite3_column_blob(rows.value, 1)),
                                    size_t(sqlite3_column_bytes(rows.value, 1))));
                found.at(i) = true;
                --remaining;
            }
        }
        return remaining == 0;
    });
    if (remaining) throw std::runtime_error("geometry: missing payload reference");
}
std::vector<std::string> reachableGeometryMetadata(sqlite3* db, const SharedJson& manifest)
{
    std::vector<SharedJson> pending{manifest.value("sequences",SharedJson::object())};
    std::set<std::string> seen;
    std::vector<std::string> result;
    while(!pending.empty()) {
        auto value=std::move(pending.back()); pending.pop_back();
        if(value.is_structured()) {
            for(auto& child:value) pending.push_back(std::move(child));
        } else if(value.is_string()) {
            const auto& hash=value.get_ref<const std::string&>();
            if(hash.size()!=64 || !std::all_of(hash.begin(),hash.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}) || !seen.insert(hash).second) continue;
            if(!hasGeometryPayload(db,hash)) continue; // Annotation text, not a stored reference.
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
    if(hasGeometryPayload(db,hash)) {
        const auto existing=readGeometryPayload(db,hash);
        if(geometryWork)geometryWork->readBytes[name]+=existing.size();
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
