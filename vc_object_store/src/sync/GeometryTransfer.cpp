#include "sync/GeometryTransfer.hpp"
#include "internal/Database.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/GeometryDelta.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/SequenceMetadata.hpp"

namespace vc::objects::detail {
namespace {
const std::string columns = "uuid,object_uuid,parent_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host";

template<class Row> Row values(const Statement& statement)
{
    Row row;
    for (int i = 0; i < sqlite3_column_count(statement.value); ++i) {
        row.emplace_back(sqlite3_value_dup(sqlite3_column_value(statement.value, i)), sqlite3_value_free);
        if (!row.back()) throw std::bad_alloc();
    }
    return row;
}
template<class Row> bool equalRows(const Row& a, const Statement& b)
{
    if (int(a.size()) != sqlite3_column_count(b.value)) return false;
    for (int i = 0; i < int(a.size()); ++i) {
        auto* value = a[i].get();
        if (sqlite3_value_type(value) != sqlite3_column_type(b.value, i)) return false;
        const auto size = sqlite3_value_bytes(value);
        if (size != sqlite3_column_bytes(b.value, i)) return false;
        if (size && std::memcmp(sqlite3_value_blob(value), sqlite3_column_blob(b.value, i), size)) return false;
    }
    return true;
}
template<class Row> void insertRow(sqlite3* destination, const Row& row, const char* sql)
{
    Statement insert(destination, sql);
    for (int i = 0; i < int(row.size()); ++i) check(sqlite3_bind_value(insert.value, i + 1, row[i].get()), destination);
    insert.stepDone();
}
}

void GeometryStage::read(sqlite3* source, sqlite3* known, const std::string& objectUuid, int64_t afterRow)
{
    std::vector<Revision> staged;
    Statement revisions(source, "SELECT " + columns + " FROM geometry_revisions WHERE object_uuid=? AND rowid>? ORDER BY object_generation,uuid",
        parseUuid(objectUuid), afterRow);
    while (revisions.row()) {
        auto& revision = staged.emplace_back(Revision{formatUuid(revisions.blob16(0)), revisions.text(4), values<Row>(revisions), {}});
        const auto parsed = loadGeometryManifest(source, revision.uuid, revision.manifest, revisions.text(3));
        // Published revisions/chunk references are immutable. Existing identical
        // revisions need no payload I/O; verify every chunk of incoming revisions.
        Statement present(known, "SELECT 1 FROM geometry_revisions WHERE uuid=?", revisions.blob16(0));
        revision.payload = !present.row();
        if (!revision.payload) continue;
        if (parsed.contains("delta")) {
            const auto changes = decodeGeometryDelta(source, revision.uuid, parsed);
            if (revisions.isNull(2)) throw std::runtime_error("sync: delta has no parent");
            GeometryReader parent(source, formatUuid(revisions.blob16(2)));
            for (const auto& c : changes) {
                auto bytes = parent.chunk(c.key.first, c.key.second);
                applyGeometryDelta(bytes, c);
            }
            Statement payload(source, "SELECT revision_uuid,data,manifest_hash FROM geometry_deltas WHERE revision_uuid=?",
                revisions.blob16(0));
            if (!payload.row()) throw std::runtime_error("sync: missing geometry delta");
            revision.delta = values<Row>(payload);
            continue;
        }
        size_t expected = 0;
        std::map<std::string, GeometryLayout> layouts;
        for (const auto& [name, array] : parsed.at("arrays").items()) {
            const auto layout = geometryLayout(array);
            if (!array.contains("source_revision")) expected += layout.chunkCount();
            layouts.emplace(name, layout);
        }
        if(parsed.contains("sequences"))for(const auto& state:parsed.at("sequences"))
            for(const auto* field:{"identities","allocated","annotations"})if(state.contains(field))
                metadataPayloads(source,state.at(field),[&](const std::vector<uint8_t>& bytes){revision.metadata.push_back(bytes);});
        Statement chunks(source, "SELECT r.array_name,r.chunk_index,c.hash,c.data FROM geometry_revision_chunks r "
            "JOIN geometry_chunks c ON c.hash=r.hash WHERE r.revision_uuid=? ORDER BY r.array_name,r.chunk_index",
            revisions.blob16(0));
        while (chunks.row()) {
            const auto* data = static_cast<const uint8_t*>(sqlite3_column_blob(chunks.value, 3));
            const auto size = sqlite3_column_bytes(chunks.value, 3);
            const auto& a=parsed.at("arrays").at(chunks.text(0));
            const auto hash=a.at("chunks").at(chunks.i64(1));
            if (size != layouts.at(chunks.text(0)).chunkBytes(chunks.i64(1)) ||
                geometryHash(std::span(data, size)) != chunks.text(2) || hash != chunks.text(2))
                throw std::runtime_error("sync: corrupt geometry chunk");
            revision.chunks.push_back({chunks.text(0), chunks.text(2), chunks.i64(1), std::vector<uint8_t>(data, data + size)});
        }
        if (revision.chunks.size() != expected) throw std::runtime_error("sync: missing geometry chunks");
    }
    objects_[objectUuid] = std::move(staged);
}

void GeometryStage::install(sqlite3* destination, const std::string& objectUuid) const
{
    const auto staged = objects_.find(objectUuid);
    if (staged == objects_.end()) throw std::logic_error("sync: geometry of " + objectUuid + " was not staged");
    for (const auto& revision : staged->second) {
        const auto uuid = parseUuid(revision.uuid);
        Statement existing(destination, "SELECT " + columns + " FROM geometry_revisions WHERE uuid=?", uuid);
        if (existing.row()) {
            if (!equalRows(revision.row, existing)) throw std::runtime_error("sync: geometry revision UUID collision");
            continue;
        }
        if (!revision.payload) throw std::runtime_error("sync: geometry revision " + revision.uuid + " left the destination while staged");
        insertRow(destination, revision.row, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)");
        if (!revision.delta.empty()) {
            insertRow(destination, revision.delta, "INSERT INTO geometry_deltas VALUES(?,?,?)");
            // Resolve/validate ancestry and the resulting manifest as well.
            GeometryReader incoming(destination, revision.uuid);
            continue;
        }
        for (const auto& [name, array] : nlohmann::json::parse(revision.manifest).at("arrays").items()) {
            if (!array.contains("source_revision")) continue;
            GeometryReader inherited(destination, array.at("source_revision").get<std::string>(), false);
            auto original = inherited.manifest().at("arrays").at(name);
            auto descriptor = array; original.erase("source_revision"); descriptor.erase("source_revision");
            if (original != descriptor) throw std::runtime_error("sync: invalid inherited array");
        }
        for (const auto& bytes : revision.metadata) storeSharedPayload(destination, "@metadata", bytes);
        for (const auto& chunk : revision.chunks) {
            const auto size = int(chunk.data.size());
            Statement old(destination, "SELECT hash,data FROM geometry_chunks WHERE hash=?", chunk.hash);
            if (old.row()) {
                if (sqlite3_column_bytes(old.value, 1) != size || std::memcmp(sqlite3_column_blob(old.value, 1), chunk.data.data(), size))
                    throw std::runtime_error("sync: geometry chunk hash collision");
            } else {
                execute(destination, "INSERT INTO geometry_chunks VALUES(?,?)", chunk.hash, std::span(chunk.data));
            }
            Statement binding(destination, "SELECT hash FROM geometry_revision_chunks WHERE revision_uuid=? AND array_name=? AND chunk_index=?",
                uuid, chunk.array, chunk.index);
            if (binding.row()) {
                if (binding.text(0) != chunk.hash) throw std::runtime_error("sync: geometry chunk reference collision");
            } else {
                execute(destination, "INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)", uuid, chunk.array, chunk.index, chunk.hash);
            }
        }
    }
}

void transferGeometry(sqlite3* source, sqlite3* destination, const std::string& objectUuid, int64_t afterRow)
{
    GeometryStage stage;
    stage.read(source, destination, objectUuid, afterRow);
    stage.install(destination, objectUuid);
}
} // namespace vc::objects::detail
