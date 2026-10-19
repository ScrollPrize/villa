#include "internal/RecordSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/Dependencies.hpp"

namespace vc::objects {
namespace detail {
GeometryReader::GeometryReader(const Txn& txn, const std::string& revision, bool cacheChunks)
    : txn_(txn), revision_(parseUuid(revision)), cacheChunks_(cacheChunks)
{
    Statement row(ObjectStoreImplAccess::conn(txn_), "SELECT content_hash,manifest FROM geometry_revisions WHERE uuid=?");
    row.bindBlob(1, revision_);
    if (!row.row()) throw std::invalid_argument("geometry: revision not found");
    const auto text = row.text(1);
    if (geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size())) != row.text(0))
        throw std::runtime_error("geometry: corrupt revision manifest");
    manifest_ = nlohmann::json::parse(text);
}

std::vector<uint8_t> GeometryReader::read(const std::string& name, int64_t offset, std::optional<int64_t> count)
{
    auto* db = ObjectStoreImplAccess::conn(txn_);
    const auto& arrays = manifest_.at("arrays");
    if (!arrays.contains(name)) throw std::invalid_argument("geometry: array not found: " + name);
    const auto& array = arrays.at(name);
    const int64_t length = array.at("byte_length");
    if (offset < 0 || offset > length) throw std::out_of_range("geometry: invalid byte offset");
    const int64_t size = count.value_or(length - offset);
    if (size < 0 || size > length - offset) throw std::out_of_range("geometry: invalid byte count");
    std::vector<uint8_t> result(static_cast<size_t>(size));
    if (!size) return result;
    Statement chunk(db, "SELECT c.data,r.hash FROM geometry_revision_chunks r JOIN geometry_chunks c ON c.hash=r.hash "
                        "WHERE r.revision_uuid=? AND r.array_name=? AND r.chunk_index=?");
    for (int64_t index = offset / geometryChunkSize; index <= (offset + size - 1) / geometryChunkSize; ++index) {
        const int64_t base = index * geometryChunkSize;
        const auto key = std::make_pair(name, index);
        auto found = chunks_.find(key);
        if (found == chunks_.end()) {
            chunk.reset(); chunk.bindBlob(1, revision_); chunk.bind(2, name); chunk.bind(3, index);
            if (!chunk.row()) throw std::runtime_error("geometry: missing payload chunk");
            const auto bytes = std::span(static_cast<const uint8_t*>(sqlite3_column_blob(chunk.value, 0)),
                                        static_cast<size_t>(sqlite3_column_bytes(chunk.value, 0)));
            if (static_cast<int64_t>(bytes.size()) != std::min<int64_t>(geometryChunkSize, length - base) ||
                chunk.text(1) != array.at("chunks").at(index).get<std::string>() || geometryHash(bytes) != chunk.text(1))
                throw std::runtime_error("geometry: corrupt payload chunk");
            if (!cacheChunks_) {
                const auto begin = std::max(base, offset), end = std::min(base + int64_t(bytes.size()), offset + size);
                std::memcpy(result.data() + begin - offset, bytes.data() + begin - base, size_t(end - begin));
                continue;
            }
            // At most eight 1 MiB chunks per reader, independently of array size.
            if (chunks_.size() == 8) chunks_.erase(chunks_.begin());
            found = chunks_.emplace(key, std::vector<uint8_t>(bytes.begin(), bytes.end())).first;
        }
        const auto begin = std::max(base, offset), end = std::min(base + int64_t(found->second.size()), offset + size);
        std::memcpy(result.data() + begin - offset, found->second.data() + begin - base, size_t(end - begin));
    }
    return result;
}

} // namespace detail
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

namespace {
std::vector<EdgeRecord> affectedEdges(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
    const std::optional<GeometryRevision>& current, const detail::DependencyGeometry& after)
{
    const auto edges = store.edgesOf(txn, objectUuid);
    if (edges.empty()) return {};
    const auto before = current ? std::optional(detail::dependencyGeometry(store, txn, *current)) : std::nullopt;
    std::vector<EdgeRecord> result;
    for (const auto& edge : edges) {
        const auto changed = [&](const Anchor& anchor) {
            return !before || detail::dependenciesChanged(anchor, *before, after);
        };
        if ((edge.aUuid == objectUuid && changed(edge.aAnchor)) ||
            (edge.bUuid == objectUuid && changed(edge.bAnchor))) result.push_back(edge);
    }
    return result;
}
}

std::vector<EdgeRecord> ObjectStore::affectedGeometryEdges(const Txn& txn, const std::string& objectUuid,
                                                         const GeometryData& data) const
{
    detail::DependencyScope dependencyScope;
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || object->kind != data.kind) throw std::invalid_argument("geometry: object missing or kind mismatch");
    const auto manifest = detail::geometryManifest(data).dump();
    const auto current = currentGeometry(txn, objectUuid);
    if (current && current->contentHash == detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size())))
        return {};
    return affectedEdges(*this, txn, object->uuid, current, detail::dependencyGeometry(data));
}

std::string ObjectStore::putGeometry(WriteTxn& txn, const std::string& objectUuid, const GeometryData& data,
    const std::optional<std::string>& expectedRevision, const std::vector<GeometryEdgeEdit>& edgeEdits)
{
    detail::DependencyScope dependencyScope;
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
    if (current && current->contentHash == contentHash) {
        if (!edgeEdits.empty()) throw std::invalid_argument("geometry is unchanged; edit relationships directly");
        return current->uuid;
    }
    const auto after = detail::dependencyGeometry(data);
    const auto affected = affectedEdges(*this, txn, object->uuid, current, after);
    std::map<std::string, GeometryEdgeEdit> decisions;
    for (const auto& edit : edgeEdits) {
        const auto uuid = formatUuid(parseUuid(edit.edgeUuid));
        if (!decisions.emplace(uuid, edit).second) throw std::invalid_argument("duplicate geometry edge decision");
    }
    for (const auto& edge : affected) {
        const auto found = decisions.find(edge.uuid);
        if (found == decisions.end())
            throw std::runtime_error("geometry edit requires explicit retain/update/remove for affected edge " + edge.uuid);
        const auto& edit = found->second;
        if (edit.expectedGeneration != edge.generation)
            throw std::runtime_error("geometry edit: edge generation changed; reload decisions");
        auto canonical = [](const std::optional<std::string>& id) -> std::optional<std::string> {
            return id ? std::optional(formatUuid(parseUuid(*id))) : std::nullopt;
        };
        if (canonical(edit.aRevision) != geometryHead(db, edge.aUuid) || canonical(edit.bRevision) != geometryHead(db, edge.bUuid))
            throw std::runtime_error("geometry edit: endpoint revision changed; reload decisions");
        if (edit.action != "retain" && edit.action != "update" && edit.action != "remove")
            throw std::invalid_argument("geometry edge action must be retain, update, or remove");
        if (edit.action == "remove") continue;
        auto validate = [&](const std::string& endpoint, const Anchor& anchor) {
            if (endpoint == object->uuid) detail::validateDependencies(anchor, after);
            else if (const auto revision = currentGeometry(txn, endpoint))
                detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
            else if (!anchor.wholeGeometry || !anchor.points.empty())
                throw std::runtime_error("geometry edit: relationship endpoint has no geometry for its point references");
        };
        validate(edge.aUuid, edit.action == "retain" ? edge.aAnchor : edit.aAnchor);
        validate(edge.bUuid, edit.action == "retain" ? edge.bAnchor : edit.bAnchor);
    }
    if (decisions.size() != affected.size())
        throw std::invalid_argument("geometry decisions include an edge not affected by this edit");

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
        for (const auto& edge : affected) {
            const auto& edit = decisions.at(edge.uuid);
            if (edit.action == "remove") { removeEdge(txn, edge.uuid); continue; }
            EdgeGeometryRepair repair;
            repair.edgeUuid = edge.uuid; repair.expectedGeneration = edge.generation;
            repair.aRevision = geometryHead(db, edge.aUuid); repair.bRevision = geometryHead(db, edge.bUuid);
            repair.aAnchor = edit.action == "retain" ? edge.aAnchor : edit.aAnchor;
            repair.bAnchor = edit.action == "retain" ? edge.bAnchor : edit.bAnchor;
            repairEdgeGeometry(txn, repair);
        }
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
    (void)connection(txn);
    return detail::GeometryReader(txn, revisionUuid, false).read(arrayName, byteOffset, byteCount);
}

std::vector<std::vector<uint8_t>> ObjectStore::readGeometryRanges(const Txn& txn, const std::string& revisionUuid,
    const std::string& arrayName, const std::vector<std::array<int64_t, 2>>& ranges) const
{
    (void)connection(txn);
    detail::GeometryReader reader(txn, revisionUuid);
    std::vector<std::vector<uint8_t>> result;
    result.reserve(ranges.size());
    for (const auto& range : ranges) result.push_back(reader.read(arrayName, range[0], range[1]));
    return result;
}

} // namespace vc::objects
