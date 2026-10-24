#include "sync/GeometryTransfer.hpp"
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/GeometryDelta.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/SequenceMetadata.hpp"

namespace vc::objects::detail {
namespace {
bool equalRows(const Statement& a, const Statement& b)
{
    const auto n = sqlite3_column_count(a.value);
    if (n != sqlite3_column_count(b.value)) return false;
    for (int i = 0; i < n; ++i) {
        if (sqlite3_column_type(a.value, i) != sqlite3_column_type(b.value, i)) return false;
        const auto size = sqlite3_column_bytes(a.value, i);
        if (size != sqlite3_column_bytes(b.value, i)) return false;
        if (size && std::memcmp(sqlite3_column_blob(a.value, i), sqlite3_column_blob(b.value, i), size)) return false;
    }
    return true;
}
void insertRow(sqlite3* destination, const Statement& row, const char* sql)
{
    Statement insert(destination, sql);
    for (int i = 0; i < sqlite3_column_count(row.value); ++i)
        check(sqlite3_bind_value(insert.value, i+1, sqlite3_column_value(row.value, i)), destination);
    insert.stepDone();
}
}
void transferGeometry(sqlite3* source, sqlite3* destination, const std::string& objectUuid, int64_t afterRow)
{
    const std::string columns = "uuid,object_uuid,parent_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host";
    Statement revisions(source, "SELECT " + columns + " FROM geometry_revisions WHERE object_uuid=? AND rowid>? ORDER BY object_generation,uuid",
        parseUuid(objectUuid), afterRow);
    while (revisions.row()) {
        const auto manifest = revisions.text(4);
        const auto parsed = loadGeometryManifest(source, formatUuid(revisions.blob16(0)), manifest, revisions.text(3));
        Statement existing(destination, "SELECT " + columns + " FROM geometry_revisions WHERE uuid=?",
            revisions.blob16(0));
        const bool present = existing.row();
        if (present && !equalRows(revisions, existing)) throw std::runtime_error("sync: geometry revision UUID collision");
        // Published revisions/chunk references are immutable. Existing identical
        // revisions need no payload I/O; verify every chunk of incoming revisions.
        if (present) continue;
        insertRow(destination, revisions, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)");
        if (parsed.contains("delta")) {
            const auto uuid = formatUuid(revisions.blob16(0));
            const auto changes = decodeGeometryDelta(source, uuid, parsed);
            if (revisions.isNull(2)) throw std::runtime_error("sync: delta has no parent");
            GeometryReader parent(source, formatUuid(revisions.blob16(2)));
            for (const auto& c : changes) {
                auto bytes = parent.chunk(c.key.first, c.key.second);
                applyGeometryDelta(bytes, c);
            }
            Statement payload(source, "SELECT revision_uuid,data,manifest_hash FROM geometry_deltas WHERE revision_uuid=?",
                revisions.blob16(0)); payload.row();
            insertRow(destination, payload, "INSERT INTO geometry_deltas VALUES(?,?,?)");
            // Resolve/validate ancestry and the resulting manifest as well.
            GeometryReader incoming(destination, uuid);
            continue;
        }
        Statement chunks(source, "SELECT r.array_name,r.chunk_index,c.hash,c.data FROM geometry_revision_chunks r "
            "JOIN geometry_chunks c ON c.hash=r.hash WHERE r.revision_uuid=? ORDER BY r.array_name,r.chunk_index",
            revisions.blob16(0));
        size_t count = 0, expected = 0;
        std::map<std::string, GeometryLayout> layouts;
        for (const auto& [name, array] : parsed.at("arrays").items()) {
            const auto layout = geometryLayout(array);
            if (!array.contains("source_revision")) expected += layout.chunkCount();
            else {
                GeometryReader inherited(destination, array.at("source_revision").get<std::string>(), false);
                auto original = inherited.manifest().at("arrays").at(name);
                auto descriptor = array; original.erase("source_revision"); descriptor.erase("source_revision");
                if (original != descriptor) throw std::runtime_error("sync: invalid inherited array");
            }
            layouts.emplace(name, layout);
        }
        if(parsed.contains("sequences"))for(const auto& state:parsed.at("sequences"))
            for(const auto* field:{"identities","allocated","annotations"})if(state.contains(field))transferMetadata(source,destination,state.at(field));
        while (chunks.row()) {
            ++count;
            const auto* data = static_cast<const uint8_t*>(sqlite3_column_blob(chunks.value, 3));
            const auto size = sqlite3_column_bytes(chunks.value, 3);
            const auto& a=parsed.at("arrays").at(chunks.text(0));
            const auto hash=a.at("chunks").at(chunks.i64(1));
            if (size != layouts.at(chunks.text(0)).chunkBytes(chunks.i64(1)) ||
                geometryHash(std::span(data, size)) != chunks.text(2) || hash != chunks.text(2))
                throw std::runtime_error("sync: corrupt geometry chunk");
            Statement old(destination, "SELECT hash,data FROM geometry_chunks WHERE hash=?", chunks.text(2));
            if (old.row()) {
                if (sqlite3_column_bytes(old.value, 1) != size || std::memcmp(sqlite3_column_blob(old.value, 1), data, size))
                    throw std::runtime_error("sync: geometry chunk hash collision");
            } else {
                execute(destination, "INSERT INTO geometry_chunks VALUES(?,?)", chunks.text(2), std::span(data, size_t(size)));
            }
            Statement binding(destination, "SELECT hash FROM geometry_revision_chunks WHERE revision_uuid=? AND array_name=? AND chunk_index=?",
                revisions.blob16(0), chunks.text(0), chunks.i64(1));
            if (binding.row()) {
                if (binding.text(0) != chunks.text(2)) throw std::runtime_error("sync: geometry chunk reference collision");
            } else {
                execute(destination, "INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)", revisions.blob16(0),
                    chunks.text(0), chunks.i64(1), chunks.text(2));
            }
        }
        if (count != expected) throw std::runtime_error("sync: missing geometry chunks");
    }
}
} // namespace vc::objects::detail
