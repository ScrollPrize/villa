#include "sync/GeometryTransfer.hpp"
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"

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
void transferGeometry(sqlite3* source, sqlite3* destination, const std::string& objectUuid)
{
    const std::string columns = "uuid,object_uuid,parent_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host";
    Statement revisions(source, "SELECT " + columns + " FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation,uuid");
    revisions.bindBlob(1, parseUuid(objectUuid));
    while (revisions.row()) {
        const auto manifest = revisions.text(4);
        if (geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size())) != revisions.text(3))
            throw std::runtime_error("sync: corrupt geometry manifest");
        Statement existing(destination, "SELECT " + columns + " FROM geometry_revisions WHERE uuid=?");
        existing.bindBlob(1, revisions.blob16(0));
        const bool present = existing.row();
        if (present && !equalRows(revisions, existing)) throw std::runtime_error("sync: geometry revision UUID collision");
        // Published revisions/chunk references are immutable. Existing identical
        // revisions need no payload I/O; verify every chunk of incoming revisions.
        if (present) continue;
        insertRow(destination, revisions, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)");
        Statement chunks(source, "SELECT r.array_name,r.chunk_index,c.hash,c.data FROM geometry_revision_chunks r "
            "JOIN geometry_chunks c ON c.hash=r.hash WHERE r.revision_uuid=? ORDER BY r.array_name,r.chunk_index");
        chunks.bindBlob(1, revisions.blob16(0));
        const auto parsed = nlohmann::json::parse(manifest);
        size_t count = 0, expected = 0;
        for (const auto& array : parsed.at("arrays")) expected += array.at("chunks").size();
        while (chunks.row()) {
            ++count;
            const auto* data = static_cast<const uint8_t*>(sqlite3_column_blob(chunks.value, 3));
            const auto size = sqlite3_column_bytes(chunks.value, 3);
            if (geometryHash(std::span(data, size)) != chunks.text(2) ||
                parsed.at("arrays").at(chunks.text(0)).at("chunks").at(chunks.i64(1)) != chunks.text(2))
                throw std::runtime_error("sync: corrupt geometry chunk");
            Statement old(destination, "SELECT hash,data FROM geometry_chunks WHERE hash=?"); old.bind(1, chunks.text(2));
            if (old.row()) {
                if (sqlite3_column_bytes(old.value, 1) != size || std::memcmp(sqlite3_column_blob(old.value, 1), data, size))
                    throw std::runtime_error("sync: geometry chunk hash collision");
            } else {
                Statement insert(destination, "INSERT INTO geometry_chunks VALUES(?,?)");
                insert.bind(1, chunks.text(2)); check(sqlite3_bind_blob(insert.value, 2, data, size, SQLITE_TRANSIENT), destination); insert.stepDone();
            }
            Statement binding(destination, "SELECT hash FROM geometry_revision_chunks WHERE revision_uuid=? AND array_name=? AND chunk_index=?");
            binding.bindBlob(1, revisions.blob16(0)); binding.bind(2, chunks.text(0)); binding.bind(3, chunks.i64(1));
            if (binding.row()) {
                if (binding.text(0) != chunks.text(2)) throw std::runtime_error("sync: geometry chunk reference collision");
            } else {
                Statement insert(destination, "INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
                insert.bindBlob(1, revisions.blob16(0)); insert.bind(2, chunks.text(0)); insert.bind(3, chunks.i64(1)); insert.bind(4, chunks.text(2)); insert.stepDone();
            }
        }
        if (count != expected) throw std::runtime_error("sync: missing geometry chunks");
    }
}
} // namespace vc::objects::detail
