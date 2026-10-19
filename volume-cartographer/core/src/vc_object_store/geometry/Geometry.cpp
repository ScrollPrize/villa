#include "internal/RecordSql.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects {
namespace {
GeometryRevision readGeometryRevision(const Statement& s)
{
    GeometryRevision r;
    r.uuid = formatUuid(s.blob16(0)); r.objectUuid = formatUuid(s.blob16(1));
    if (!s.isNull(2)) r.parentUuid = formatUuid(s.blob16(2));
    r.contentHash = s.text(3);
    const auto text = s.text(4);
    if (detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size())) != r.contentHash)
        throw std::runtime_error("geometry: corrupt revision manifest");
    const auto manifest = nlohmann::json::parse(text);
    r.kind = manifest.at("kind"); r.formatVersion = manifest.at("format_version");
    r.coordinateSystem = manifest.at("coordinate_system"); r.attributes = manifest.at("attributes").dump();
    r.objectGeneration = s.i64(5); r.createdBy = s.text(6); r.createdAt = s.text(7); r.createdHost = s.text(8);
    for (const auto& [name, a] : manifest.at("arrays").items())
        r.arrays.push_back({name, a.at("dtype"), a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length")});
    return r;
}
constexpr const char* geometryColumns = "uuid,object_uuid,parent_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host";
}

std::optional<GeometryRevision> ObjectStore::geometryRevision(const Txn& txn, const std::string& revisionUuid) const
{
    (void)connection(txn);
    Statement s(static_cast<sqlite3*>(connection(txn)), std::string("SELECT ") + geometryColumns + " FROM geometry_revisions WHERE uuid=?");
    s.bindBlob(1, parseUuid(revisionUuid));
    return s.row() ? std::optional(readGeometryRevision(s)) : std::nullopt;
}

std::optional<GeometryRevision> ObjectStore::currentGeometry(const Txn& txn, const std::string& objectUuid) const
{
    (void)connection(txn);
    const auto head = geometryHead(static_cast<sqlite3*>(connection(txn)), objectUuid);
    return head ? geometryRevision(txn, *head) : std::nullopt;
}

std::vector<GeometryRevision> ObjectStore::geometryHistory(const Txn& txn, const std::string& objectUuid) const
{
    (void)connection(txn);
    Statement s(static_cast<sqlite3*>(connection(txn)), std::string("SELECT ") + geometryColumns + " FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation");
    s.bindBlob(1, parseUuid(objectUuid));
    std::vector<GeometryRevision> result;
    while (s.row()) result.push_back(readGeometryRevision(s));
    return result;
}

std::string ObjectStore::putGeometry(WriteTxn& txn, const std::string& objectUuid, const GeometryData& data,
    const std::optional<std::string>& expectedRevision)
{
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || object->kind != data.kind) throw std::invalid_argument("geometry: object missing or kind mismatch");
    const auto current = currentGeometry(txn, objectUuid);
    const auto expected = expectedRevision ? std::optional(formatUuid(parseUuid(*expectedRevision))) : std::nullopt;
    if ((current ? std::optional(current->uuid) : std::nullopt) != expected)
        throw std::runtime_error("geometry: current revision changed; reload before committing");
    const auto manifest = detail::geometryManifest(data);
    const auto manifestText = manifest.dump();
    const auto contentHash = detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifestText.data()), manifestText.size()));
    if (current && current->contentHash == contentHash) return current->uuid;

    exec(db, "SAVEPOINT geometry_publish;");
    try {
        const auto revisionUuid = util::newUuid();
        Statement revision(db, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)");
        revision.bindBlob(1, parseUuid(revisionUuid)); revision.bindBlob(2, parseUuid(objectUuid));
        if (current) revision.bindBlob(3, parseUuid(current->uuid)); else revision.bindNull(3);
        revision.bind(4, contentHash); revision.bind(5, manifestText); revision.bind(6, object->generation + 1);
        revision.bind(7, author()); revision.bind(8, nowIsoUtc()); revision.bind(9, hostname()); revision.stepDone();
        Statement insert(db, "INSERT INTO geometry_chunks(hash,data) VALUES(?,?) ON CONFLICT(hash) DO NOTHING");
        Statement verify(db, "SELECT data FROM geometry_chunks WHERE hash=?");
        Statement reference(db, "INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
        for (const auto& array : data.arrays) {
            size_t index = 0;
            for (size_t offset = 0; offset < array.bytes.size(); offset += detail::geometryChunkSize, ++index) {
                const auto bytes = std::span(array.bytes).subspan(offset, std::min(detail::geometryChunkSize, array.bytes.size() - offset));
                const std::string hash = manifest.at("arrays").at(array.name).at("chunks").at(index);
                insert.reset(); insert.bind(1, hash);
                check(sqlite3_bind_blob64(insert.value, 2, bytes.data(), bytes.size(), SQLITE_STATIC), db, "bind geometry chunk");
                insert.stepDone();
                // Never silently reuse corrupt data, even on a hash collision.
                verify.reset(); verify.bind(1, hash);
                if (!verify.row() || sqlite3_column_bytes(verify.value, 0) != static_cast<int>(bytes.size()) ||
                    std::memcmp(sqlite3_column_blob(verify.value, 0), bytes.data(), bytes.size()) != 0)
                    throw std::runtime_error("geometry: corrupt or colliding payload hash");
                reference.reset(); reference.bindBlob(1, parseUuid(revisionUuid)); reference.bind(2, array.name);
                reference.bind(3, static_cast<int64_t>(index)); reference.bind(4, hash); reference.stepDone();
            }
        }
        Statement head(db, "INSERT INTO geometry_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid");
        head.bindBlob(1, parseUuid(objectUuid)); head.bindBlob(2, parseUuid(revisionUuid)); head.stepDone();
        const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
        stampContentChange(db, author(), hostname(), "objects", id, parseUuid(objectUuid), "geometry");
        Statement update(db, "UPDATE objects SET storage_kind='sqlite',data_generation=generation,data_checksum=? WHERE id=?");
        update.bind(1, contentHash); update.bind(2, id); update.stepDone();
        for (const auto& edge : edgesOf(txn, objectUuid))
            stampContentChange(db, author(), hostname(), "edges",
                *edgeIdByUuid(db, parseUuid(edge.uuid)), parseUuid(edge.uuid), "geometry_invalidated");
        exec(db, "RELEASE geometry_publish;");
        return revisionUuid;
    } catch (...) {
        exec(db, "ROLLBACK TO geometry_publish; RELEASE geometry_publish;");
        throw;
    }
}

std::vector<uint8_t> ObjectStore::readGeometryArray(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, int64_t byteOffset, const std::optional<int64_t>& byteCount) const
{
    const auto revision = geometryRevision(txn, revisionUuid);
    if (!revision) throw std::invalid_argument("geometry: revision not found");
    const auto array = std::find_if(revision->arrays.begin(), revision->arrays.end(), [&](const auto& a) { return a.name == arrayName; });
    if (array == revision->arrays.end()) throw std::invalid_argument("geometry: array not found: " + arrayName);
    if (byteOffset < 0 || byteOffset > array->byteLength) throw std::out_of_range("geometry: invalid byte offset");
    const auto count = byteCount.value_or(array->byteLength - byteOffset);
    if (count < 0 || count > array->byteLength - byteOffset) throw std::out_of_range("geometry: invalid byte count");
    std::vector<uint8_t> result(static_cast<size_t>(count));
    if (!count) return result;
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement manifestQuery(db, "SELECT manifest FROM geometry_revisions WHERE uuid=?");
    manifestQuery.bindBlob(1, parseUuid(revisionUuid)); (void)manifestQuery.row();
    const auto hashes = nlohmann::json::parse(manifestQuery.text(0)).at("arrays").at(arrayName).at("chunks");
    const auto first = byteOffset / detail::geometryChunkSize, last = (byteOffset + count - 1) / detail::geometryChunkSize;
    Statement chunk(db, "SELECT c.data,r.hash FROM geometry_revision_chunks r JOIN geometry_chunks c ON c.hash=r.hash "
        "WHERE r.revision_uuid=? AND r.array_name=? AND r.chunk_index=?");
    for (auto i = first; i <= last; ++i) {
        chunk.reset(); chunk.bindBlob(1, parseUuid(revisionUuid)); chunk.bind(2, arrayName); chunk.bind(3, static_cast<int64_t>(i));
        if (!chunk.row()) throw std::runtime_error("geometry: missing payload chunk");
        const auto bytes = std::span(static_cast<const uint8_t*>(sqlite3_column_blob(chunk.value, 0)), static_cast<size_t>(sqlite3_column_bytes(chunk.value, 0)));
        const auto base = static_cast<int64_t>(i * detail::geometryChunkSize);
        if (static_cast<int64_t>(bytes.size()) != std::min<int64_t>(detail::geometryChunkSize, array->byteLength - base) ||
            chunk.text(1) != hashes.at(i).get<std::string>() || detail::geometryHash(bytes) != chunk.text(1))
            throw std::runtime_error("geometry: corrupt payload chunk");
        const auto begin = std::max(base, byteOffset), end = std::min(base + static_cast<int64_t>(bytes.size()), byteOffset + count);
        std::memcpy(result.data() + begin - byteOffset, bytes.data() + begin - base, static_cast<size_t>(end - begin));
    }
    return result;
}

} // namespace vc::objects
