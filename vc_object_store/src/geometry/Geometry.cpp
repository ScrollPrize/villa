#include "internal/RecordSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/Dependencies.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "geometry/DependencyIndex.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "geometry/GeometryDelta.hpp"
#include "internal/History.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/SequenceMetadata.hpp"
#include "geometry/InitialGeometry.hpp"

namespace vc::objects {
using namespace detail;

namespace {
GeometryRevision readGeometryRevision(const Statement& s)
{
    GeometryRevision r;
    r.uuid = readGeometryTokenColumn(s,0); r.objectUuid = formatUuid(s.blob16(1));
    if (!s.isNull(2)) r.parentUuid = formatUuid(s.blob16(2));
    r.contentHash = sqlite3_column_type(s.value,3)==SQLITE_TEXT ? s.text(3) : readDigest(s,3);
    const auto text = s.text(4);
    const auto manifest = loadGeometryManifest(sqlite3_db_handle(s.value), r.uuid, text, r.contentHash);
    r.gridOrigin = manifest.value("grid_origin", std::array<int64_t,2>{0,0});
    r.kind = manifest.at("kind").get<std::string>(); r.formatVersion = manifest.at("format_version");
    r.coordinateSystem = manifest.at("coordinate_system").get<std::string>(); r.attributes = manifest.at("attributes").dump();
    r.objectVersion = s.i64(5); r.createdBy = s.text(6); r.createdAt = s.text(7); r.createdHost = s.text(8);
    for (const auto& [name, a] : manifest.at("arrays").items())
        r.arrays.push_back({name, a.at("dtype"), a.at("shape").get<std::vector<int64_t>>(), a.at("byte_length")});
    return r;
}
constexpr const char* geometryColumns = "uuid,object_uuid,parent_uuid,content_hash,manifest,object_version,created_by,created_at,created_host";
}

std::optional<GeometryRevision> ObjectStore::geometryRevision(const Txn& txn, const std::string& revisionUuid) const
{
    (void)connection(txn);
    if(auto current=parseCurrentGeometryToken(revisionUuid)){
        if(historyEnabled(static_cast<sqlite3*>(connection(txn))))return std::nullopt;
        Statement s(static_cast<sqlite3*>(connection(txn)),"SELECT storage_ref,uuid,NULL,data_checksum,geometry_manifest,version,changed_by,changed_at,changed_host FROM objects WHERE uuid=? AND storage_ref=? AND state<>2 AND geometry_manifest IS NOT NULL",
            parseUuid(current->objectUuid),canonicalGeometryToken(revisionUuid));
        return s.row()?std::optional(readGeometryRevision(s)):std::nullopt;
    }
    if(!historyEnabled(static_cast<sqlite3*>(connection(txn))))return std::nullopt;
    Statement s(static_cast<sqlite3*>(connection(txn)), std::string("SELECT ") + geometryColumns + " FROM geometry_revisions WHERE uuid=?",
        parseUuid(revisionUuid));
    return s.row() ? std::optional(readGeometryRevision(s)) : std::nullopt;
}

std::optional<GeometryRevision> ObjectStore::currentGeometry(const Txn& txn, const std::string& objectUuid) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    Statement row(db,"SELECT storage_ref,uuid,NULL,data_checksum,geometry_manifest,version,changed_by,changed_at,changed_host "
        "FROM objects WHERE uuid=? AND state<>2 AND geometry_manifest IS NOT NULL",parseUuid(objectUuid));
    if(!row.row())return std::nullopt;
    auto result=readGeometryRevision(row);
    // Current content lives on the object in both history modes. History only
    // supplies the archival ancestry/authorship, never a second current head.
    if(historyEnabled(db)) {
        Statement archive(db,"SELECT parent_uuid,created_by,created_at,created_host,content_hash FROM geometry_revisions WHERE uuid=? AND object_uuid=?",
            parseUuid(result.uuid),parseUuid(objectUuid));
        if(!archive.row())throw std::runtime_error("geometry: current descriptor has no matching archive revision");
        if(readDigest(archive,4)!=result.contentHash)
            throw std::runtime_error("geometry: current checksum disagrees with archive revision");
        if(!archive.isNull(0))result.parentUuid=formatUuid(archive.blob16(0));
        result.createdBy=archive.text(1);result.createdAt=archive.text(2);result.createdHost=archive.text(3);
    }
    return result;
}

std::vector<GeometryRevision> ObjectStore::geometryHistory(const Txn& txn, const std::string& objectUuid) const
{
    requireHistory(static_cast<sqlite3*>(connection(txn)));
    (void)connection(txn);
    Statement s(static_cast<sqlite3*>(connection(txn)), std::string("SELECT ") + geometryColumns + " FROM geometry_revisions WHERE object_uuid=? ORDER BY rowid",
        parseUuid(objectUuid));
    std::vector<GeometryRevision> result;
    while (s.row()) result.push_back(readGeometryRevision(s));
    return result;
}

namespace {
int64_t revisionSurfaceTileSize(const Txn& txn, const GeometryRevision& revision)
{
    if (revision.kind != "surface") return 0;
    const detail::GeometryReader reader(txn, revision.uuid);
    const auto& points = reader.manifest().at("arrays").at("points");
    return points.contains("storage") ? points.at("storage").at("tile_size").get<int64_t>() : 0;
}
std::vector<EdgeRecord> affectedEdges(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
    const std::optional<GeometryRevision>& current, const detail::DependencyGeometry& after,
    const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds = std::nullopt)
{
    GeometryPhase phase("affectedEdges");
    const auto edges = changedIds ? structuralEdgeCandidates(ObjectStoreImplAccess::conn(txn),objectUuid,changedIds->first,changedIds->second) : store.edgesOf(txn, objectUuid, true);
    if (edges.empty()) return {};
    const auto before = current ? std::optional(detail::dependencyGeometry(store, txn, *current)) : std::nullopt;
    std::vector<EdgeRecord> result;
    for (const auto& edge : edges) {
        if (geometryWork) ++geometryWork->relationshipRows;
        const auto changed = [&](const Anchor& anchor) {
            return !before || detail::dependenciesChanged(anchor, *before, after);
        };
        if ((edge.aUuid == objectUuid && changed(edge.aAnchor)) ||
            (edge.bUuid == objectUuid && changed(edge.bAnchor))) result.push_back(edge);
    }
    return result;
}
// Decisions other than the keep-recorded default (which an unchanged
// geometry simply does not use).
bool explicitEdgeDecisions(const std::vector<GeometryEdgeEdit>& edits)
{
    return std::any_of(edits.begin(), edits.end(), [](const auto& e) { return e.edgeUuid != kKeepAffectedRecorded; });
}
std::map<std::string, GeometryEdgeEdit> validateEdgeEdits(const ObjectStore& store, const Txn& txn,
    sqlite3* db, const std::string& objectUuid, const detail::DependencyGeometry& after,
    const std::vector<EdgeRecord>& affected, const std::vector<GeometryEdgeEdit>& edgeEdits)
{
    std::map<std::string, GeometryEdgeEdit> decisions;
    bool keepRest = false;
    for (const auto& edit : edgeEdits) {
        if (edit.edgeUuid == kKeepAffectedRecorded) {
            if (edit.action != "keep_recorded" || keepRest)
                throw std::invalid_argument("geometry: one keep-recorded default, with action keep_recorded");
            keepRest = true;
            continue;
        }
        const auto uuid = formatUuid(parseUuid(edit.edgeUuid));
        if (!decisions.emplace(uuid, edit).second) throw std::invalid_argument("duplicate geometry edge decision");
    }
    // The default decides the rest from this checked state: the request's
    // endpoint expectations already pin the edges' endpoints.
    if (keepRest)
        for (const auto& edge : affected)
            if (!decisions.contains(edge.uuid)) {
                GeometryEdgeEdit kept;
                kept.edgeUuid = edge.uuid; kept.expectedVersion = edge.version.value_or(0); kept.action = "keep_recorded";
                kept.aRevision = geometryHead(db, edge.aUuid); kept.bRevision = geometryHead(db, edge.bUuid);
                decisions.emplace(edge.uuid, std::move(kept));
            }
    for (const auto& edge : affected) {
        const auto found = decisions.find(edge.uuid);
        if (found == decisions.end())
            throw std::runtime_error("geometry edit requires explicit retain/update/remove for affected edge " + edge.uuid);
        const auto& edit = found->second;
        if (edit.expectedVersion != edge.version.value_or(0))
            historyConflict(db, "geometry edit: edge version changed; reload decisions");
        auto canonical = [](const std::optional<std::string>& id) -> std::optional<std::string> {
            return id ? std::optional(canonicalGeometryToken(*id)) : std::nullopt;
        };
        if (canonical(edit.aRevision) != geometryHead(db, edge.aUuid) || canonical(edit.bRevision) != geometryHead(db, edge.bUuid))
            historyConflict(db, "geometry edit: endpoint revision changed; reload decisions");
        if (edit.action != "retain" && edit.action != "update" && edit.action != "remove" && edit.action != "keep_recorded")
            throw std::invalid_argument("geometry edge action must be retain, update, remove, or keep_recorded");
        if (edit.action == "remove" || edit.action == "keep_recorded") continue;
        auto validate = [&](const std::string& endpoint, const Anchor& anchor) {
            if (endpoint == objectUuid) detail::validateDependencies(anchor, after);
            else if (const auto revision = store.currentGeometry(txn, endpoint))
                detail::validateDependencies(anchor, detail::dependencyGeometry(store, txn, *revision));
            else if(const auto& validate=activeStatementPools.at(db)->externalEndpointValidateAnchor)
                validate(endpoint,anchor);
            else if (!anchor.wholeGeometry || !anchor.points.empty())
                throw std::runtime_error("geometry edit: relationship endpoint has no geometry for its point references");
        };
        validate(edge.aUuid, edit.action == "update" ? edit.aAnchor : edge.aAnchor);
        validate(edge.bUuid, edit.action == "update" ? edit.bAnchor : edge.bAnchor);
    }
    if (decisions.size() != affected.size())
        throw std::invalid_argument("geometry decisions include an edge not affected by this edit");

    return decisions;
}

void applyEdgeDecisions(ObjectStore& store, WriteTxn& txn, sqlite3* db,
                        const std::vector<EdgeRecord>& affected,
                        const std::map<std::string, GeometryEdgeEdit>& decisions)
{
    for (const auto& edge : affected) {
        const auto& edit = decisions.at(edge.uuid);
        if (edit.action == "remove") { store.removeEdge(txn, edge.uuid); continue; }
        if (edit.action == "keep_recorded") { store.keepRecordedRelationship(txn, edge.uuid); continue; }
        EdgeGeometryRepair repair;
        repair.edgeUuid = edge.uuid; repair.expectedVersion = edge.version.value_or(0);
        repair.aRevision = geometryHead(db, edge.aUuid); repair.bRevision = geometryHead(db, edge.bUuid);
        repair.aAnchor = edit.action == "retain" ? edge.aAnchor : edit.aAnchor;
        repair.bAnchor = edit.action == "retain" ? edge.bAnchor : edit.bAnchor;
        store.repairEdgeGeometry(txn, repair);
    }
}

std::string newGeometryToken(sqlite3* db,const std::string& object)
{return historyEnabled(db)?util::newUuid():currentGeometryToken(object,writeObjectVersion(db,object));}

void insertCurrentGeometry(sqlite3* db,const std::string& objectUuid,
    const std::string& uuid,const std::string& hash,const std::string& text)
{
    execute(db,"UPDATE objects SET geometry_manifest=?,storage_ref=?,data_checksum=? WHERE uuid=?",
        text,uuid,hash,parseUuid(objectUuid));
}

void writeRevision(sqlite3* db, const ObjectStore& store, const ObjectRecord& object,
    const std::string& uuid, const GeometryRevision* parent, const std::string& hash, const std::string& text,
    const std::string* currentText=nullptr)
{
    if (!historyEnabled(db)) {
        if(const auto pool=activeStatementPools.find(db);pool!=activeStatementPools.end())pool->second->geometryCache.reset();
        if(activeDependencies)activeDependencies->clear();
        // Replace the current object's descriptor; no revision or mapping rows.
        insertCurrentGeometry(db,object.uuid,uuid,hash,text);
        return;
    }
    Statement revision(db, "INSERT INTO geometry_revisions VALUES(?,?,?,?,?,?,?,?,?)", parseUuid(uuid),
        parseUuid(object.uuid));
    if (parent) revision.bind(3, parseUuid(parent->uuid)); else revision.bindNull(3);
    bindDigest(revision,4,hash); revision.bind(5, text); revision.bind(6, writeObjectVersion(db,object.uuid));
    revision.bind(7, store.author()); revision.bind(8, nowIsoUtc()); revision.bind(9, store.hostname());
    revision.stepDone();
    insertCurrentGeometry(db,object.uuid,uuid,hash,currentText ? *currentText : text);
}

void stampGeometry(sqlite3* db, const ObjectStore& store, const std::string& objectUuid,
                   int64_t id, const std::string& hash)
{
    stampContentChange(db, store.author(), store.hostname(), "objects", id, parseUuid(objectUuid), "geometry");
    execute(db, "UPDATE objects SET storage_kind='sqlite',data_checksum=? WHERE id=?", hash,
        id);
}

void writeArrays(sqlite3* db, const std::string& revisionUuid,
                 const std::vector<GeometryArray>& arrays, const nlohmann::json& manifest)
{
    const auto current=parseCurrentGeometryToken(revisionUuid);
    const auto mappingKey=parseUuid(current?current->objectUuid:revisionUuid);
    Statement insert(db, "INSERT INTO geometry_chunks(hash,data) VALUES(?,?) ON CONFLICT(hash) DO NOTHING");
    Statement verify(db, "SELECT data FROM geometry_chunks WHERE hash=?");
    std::unique_ptr<Statement> reference;
    if(historyEnabled(db))reference=std::make_unique<Statement>(db,"INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
    for (const auto& array : arrays) {
        const auto layout = detail::geometryLayout(manifest.at("arrays").at(array.name));
        detail::forEachGeometryChunk(array, layout.tile, [&](int64_t index, std::span<const uint8_t> bytes) {
            const std::string hash = manifest.at("arrays").at(array.name).at("chunks").at(index);
            const auto digest=parseDigest(hash);
            if(hasExternalGeometryPayload(db,hash)) {
                const auto existing=readExternalGeometryPayload(db,hash);
                if(!std::equal(existing.begin(),existing.end(),bytes.begin(),bytes.end()))
                    throw std::runtime_error("geometry: corrupt or colliding external payload hash");
                if(reference)reference->run(mappingKey,array.name,index,digest);
                return;
            }
            insert.reset(); insert.bind(1, digest);
            check(sqlite3_bind_blob64(insert.value, 2, bytes.data(), bytes.size(), SQLITE_STATIC), db, "bind geometry chunk");
            insert.stepDone();
            const bool inserted=sqlite3_changes(db)!=0;
            if (geometryWork && inserted) geometryWork->writtenBytes[array.name] += bytes.size();
            // A successful new INSERT has stored exactly the supplied bytes.
            // Only reuse needs readback: never silently reuse corrupt data or
            // a hash collision, including external chunks resolved via store_reference.
            if(!inserted) {
                verify.reset(); verify.bind(1, digest);
                if (!verify.row()) throw std::runtime_error("geometry: missing reused payload");
                if (sqlite3_column_bytes(verify.value, 0) != static_cast<int>(bytes.size()) ||
                    std::memcmp(sqlite3_column_blob(verify.value, 0), bytes.data(), bytes.size()) != 0)
                    throw std::runtime_error("geometry: corrupt or colliding payload hash");
                if (geometryWork) geometryWork->readBytes[array.name] += bytes.size();
            }
            if(reference)reference->run(mappingKey, array.name, static_cast<int64_t>(index), digest);
        });
    }

}


// Materialize an already-computed snapshot. Shared by history-off sparse and
// structural edits; no alternate geometry/dependency calculation.
GeometryData snapshotData(const nlohmann::json& manifest, const DependencyGeometry& source)
{
    GeometryData data;
    data.kind = manifest.at("kind"); data.formatVersion = manifest.at("format_version");
    data.coordinateSystem = manifest.at("coordinate_system"); data.attributes = manifest.at("attributes").dump();
    for (const auto& [name, array] : manifest.at("arrays").items())
        data.arrays.push_back({name, array.at("dtype"), array.at("shape"), source.read(name, 0, array.at("byte_length"))});
    return data;
}

}

PreparedInitialGeometry detail::prepareInitialGeometry(sqlite3* db,const ObjectRecord& object,
    const GeometryData& data,const GeometryStorageOptions& storage)
{
    GeometryPhase phase("initial.geometry.prepare");
    if(historyEnabled(db)||spatialEnabled(db))
        throw std::invalid_argument("initial geometry requires history-off/spatial-off storage");
    if(object.uuid.empty()||object.kind!=data.kind)
        throw std::invalid_argument("initial geometry: object missing identity or kind mismatch");
    PreparedInitialGeometry result;
    result.record=object;
    { GeometryPhase manifest("initial.geometry.manifest_and_hash");
      result.manifest=geometryManifest(data,storage.surfaceTileSize); }
    auto meta=nlohmann::json::parse(object.meta);
    normalizeObjectCoordinates(meta,object.kind,true);
    if(data.kind!="surface") {
        auto source=meta.value("source",nlohmann::json::object());
        result.manifest["sequences"]=initialSequences(db,data,source);
        meta["source"]=std::move(source);
    } else {
        // Ordinary publication maintains these derived detail fields even when
        // spatial indexing is disabled. Compute from input, before initial INSERT.
        const auto& pointArray=*std::find_if(data.arrays.begin(),data.arrays.end(),[](const auto& a){return a.name=="points";});
        const auto validArray=std::find_if(data.arrays.begin(),data.arrays.end(),[](const auto& a){return a.name==surfaceValidChannel;});
        const auto points=xyzValues<float>(pointArray.bytes);
        const auto rows=pointArray.shape[0],columns=pointArray.shape[1];
        if(!result.record.surface)result.record.surface.emplace();
        auto& surface=*result.record.surface;
        surface.gridHeight=rows;surface.gridWidth=columns;
        surface.area=vc::geometry::surfaceGridArea(rows,columns,
            [&](int64_t r,int64_t c)->const std::array<float,3>&{return points[size_t(r*columns+c)];},
            [&](int64_t r,int64_t c){const auto i=size_t(r*columns+c);return vc::geometry::validSurfacePoint(points[i])&&
                (validArray==data.arrays.end()||validArray->bytes[i]!=0);});
    }
    result.record.meta=meta.dump();
    result.manifestText=result.manifest.dump();
    if(geometryWork)geometryWork->manifestBytesWritten+=result.manifestText.size();
    result.contentHash=geometryHash(std::span(reinterpret_cast<const uint8_t*>(result.manifestText.data()),result.manifestText.size()));
    result.revisionUuid=currentGeometryToken(object.uuid,1);
    result.record.storageKind="sqlite";
    result.record.storageRef=result.revisionUuid;
    result.record.version=1;
    result.record.dataChecksum=result.contentHash;
    return result;
}

void detail::writeInitialGeometry(ObjectStore& store,WriteTxn& txn,const PreparedInitialGeometry& prepared,
    const GeometryData& data,int64_t objectId)
{
    GeometryPhase phase("initial.geometry.write");
    auto* db=ObjectStoreImplAccess::conn(txn);
    if(historyEnabled(db)||spatialEnabled(db)||objectId<=0)
        throw std::invalid_argument("initial geometry requires a new object in history-off/spatial-off storage");
    // The initial object INSERT already includes the complete current descriptor.
    // Do not rewrite its metadata/indexes merely to attach the same manifest.
    writeArrays(db,prepared.revisionUuid,data.arrays,prepared.manifest);
}

std::vector<EdgeRecord> ObjectStore::affectedGeometryEdges(const Txn& txn, const std::string& objectUuid,
                                                         const GeometryData& data) const
{
    detail::DependencyScope dependencyScope;
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || object->kind != data.kind) throw std::invalid_argument("geometry: object missing or kind mismatch");
    const auto current = currentGeometry(txn, objectUuid);
    const auto manifest = detail::geometryManifest(data, current ? revisionSurfaceTileSize(txn, *current) : GeometryStorageOptions{}.surfaceTileSize).dump();
    if (current && current->contentHash == detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size())))
        return {};
    return affectedEdges(*this, txn, object->uuid, current, detail::dependencyGeometry(data));
}

std::string ObjectStore::putGeometry(WriteTxn& txn, const std::string& objectUuid, const GeometryData& data,
    const std::optional<std::string>& expectedRevision, const std::vector<GeometryEdgeEdit>& edgeEdits,
    const GeometryStorageOptions& storage)
{
    GeometryPhase phase("ObjectStore::putGeometry");
    detail::DependencyScope dependencyScope;
    (void)connection(txn);
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || object->kind != data.kind) throw std::invalid_argument("geometry: object missing or kind mismatch");
    const auto current = currentGeometry(txn, objectUuid);
    const auto expected = expectedRevision ? std::optional(canonicalGeometryToken(*expectedRevision)) : std::nullopt;
    if ((current ? std::optional(current->uuid) : std::nullopt) != expected)
        historyConflict(db, "geometry: current revision changed; reload before committing");
    if (current && data.kind != "surface")
        throw std::invalid_argument("sequences require structural or sparse edits");
    Savepoint savepoint(db);
    auto source = nlohmann::json::parse(object->meta).value("source", nlohmann::json::object());
    auto manifest = detail::geometryManifest(data, storage.surfaceTileSize);
    if (data.kind != "surface") {
        manifest["sequences"] = initialSequences(db, data, source);
        setMeta(txn, objectUuid, "source", source.dump());
    }
    const auto manifestText = manifest.dump();
    if (geometryWork) geometryWork->manifestBytesWritten += manifestText.size();
    const auto contentHash = detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(manifestText.data()), manifestText.size()));
    bool unchanged = current && current->contentHash == contentHash;
    if (current && !unchanged && data.kind == "surface") {
        const auto oldTile = revisionSurfaceTileSize(txn, *current);
        if (oldTile != storage.surfaceTileSize) {
            const auto oldManifest = detail::geometryManifest(data, oldTile).dump();
            unchanged = current->contentHash == detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(oldManifest.data()), oldManifest.size()));
        }
    }
    if (unchanged) {
        if (explicitEdgeDecisions(edgeEdits)) throw std::invalid_argument("geometry is unchanged; edit relationships directly");
        savepoint.release();
        return current->uuid;
    }
    const auto after = detail::dependencyGeometry(data);
    const auto affected = affectedEdges(*this, txn, object->uuid, current, after);
    const auto decisions = validateEdgeEdits(*this, txn, db, object->uuid, after, affected, edgeEdits);

    const auto revisionUuid = newGeometryToken(db,objectUuid);
    writeRevision(db, *this, *objectByUuid(txn, objectUuid), revisionUuid, current ? &*current : nullptr, contentHash, manifestText);
    writeArrays(db, revisionUuid, data.arrays, manifest);

    detail::clearGeometryMaterialization(db, objectUuid);
    const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
    detail::indexGeometryBounds(db, id, revisionUuid, data);
    stampGeometry(db, *this, objectUuid, id, contentHash);
    applyEdgeDecisions(*this, txn, db, affected, decisions);
    savepoint.release();
    return revisionUuid;
}

std::string detail::publishReferencedGeometry(ObjectStore& store, WriteTxn& txn, const std::string& objectUuid,
                                              const SharedJson& manifest)
{
    auto* db=ObjectStoreImplAccess::conn(txn);
    const auto object=store.objectByUuid(txn,objectUuid);
    if(!object||object->kind!=manifest.at("kind").get<std::string>()||store.currentGeometry(txn,objectUuid))
        throw std::invalid_argument("geometry: referenced import requires an empty matching object");
    Savepoint save(db);
    const auto text=manifest.dump(), hash=hashText(text), revision=newGeometryToken(db,objectUuid);
    writeRevision(db,store,*object,revision,nullptr,hash,text);
    std::unique_ptr<Statement> reference;
    if(historyEnabled(db))reference=std::make_unique<Statement>(db,"INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
    for(const auto& [name,array]:manifest.at("arrays").items()){
        if(array.contains("source_revision"))throw std::invalid_argument("geometry: referenced import must be a full descriptor");
        int64_t index=0;for(const auto& chunk:array.at("chunks"))if(reference)reference->run(parseUuid(revision),name,index++,parseDigest(chunk.get<std::string>()));
    }
    refreshGeometryBounds(db,objectUuid);
    stampGeometry(db,store,objectUuid,*objectIdByUuid(db,parseUuid(objectUuid)),hash);
    if(!activeStatementPools.at(db)->immutableInitial)for(const auto& edge:store.edgesOf(txn,objectUuid,true)){
        auto state=store.edgeGeometryState(txn,edge.uuid);
        EdgeGeometryRepair repair;repair.edgeUuid=edge.uuid;repair.expectedVersion=edge.version.value_or(0);
        repair.aRevision=edge.aUuid==objectUuid?std::optional(revision):state.aRevision;
        repair.bRevision=edge.bUuid==objectUuid?std::optional(revision):state.bRevision;
        repair.aAnchor=edge.aAnchor;repair.bAnchor=edge.bAnchor;store.repairEdgeGeometry(txn,repair);
    }
    save.release();return revision;
}

namespace {
struct SparseGeometryEdit {
    std::shared_ptr<GeometryReader> reader;
    nlohmann::json manifest;
    std::map<GeometryChunkKey, std::vector<uint8_t>> original;
    // Shared with the dependency reader of `after` rather than copied.
    std::shared_ptr<std::map<GeometryChunkKey, std::vector<uint8_t>>> modified =
        std::make_shared<std::map<GeometryChunkKey, std::vector<uint8_t>>>();
    std::vector<GeometryChunkChange> changes;
    DependencyGeometry after;
};
std::shared_ptr<SparseGeometryEdit> prepareEdit(const ObjectStore& store, const Txn& txn,
    const std::string& objectUuid, const std::string& expected, const std::vector<GeometryArrayEdit>& edits)
{
    const auto revision = store.currentGeometry(txn, objectUuid);
    if (!revision || revision->uuid != canonicalGeometryToken(expected))
        historyConflict(ObjectStoreImplAccess::conn(txn), "geometry: current revision changed; reload before committing");
    auto plan = std::make_shared<SparseGeometryEdit>();
    plan->reader = std::make_shared<GeometryReader>(txn, revision->uuid);
    plan->manifest = plan->reader->manifest();
    for (const auto& edit : edits) {
        if (!plan->manifest.at("arrays").contains(edit.array))
            throw std::invalid_argument("geometry edit: array not found: " + edit.array);
        const auto& array = plan->manifest.at("arrays").at(edit.array);
        const auto layout = geometryLayout(array);
        const auto scalar = geometryByteLength(array.at("dtype"), {1});
        if (edit.elementOffset < 0 || edit.elementOffset > layout.length / scalar || edit.bytes.size() % scalar ||
            edit.bytes.size() > uint64_t(layout.length - edit.elementOffset * scalar))
            throw std::out_of_range("geometry edit: invalid or unaligned element range");
        int64_t offset = edit.elementOffset * scalar;
        size_t consumed = 0;
        while (consumed < edit.bytes.size()) {
            int64_t index, within, available;
            if (!layout.tile) {
                index = offset / geometryChunkSize; within = offset % geometryChunkSize;
                available = layout.chunkBytes(index) - within;
            } else {
                const auto row = offset / (layout.cols * layout.pixelBytes), col = offset % (layout.cols * layout.pixelBytes) / layout.pixelBytes;
                index = row / layout.tile * layout.tileColumns() + col / layout.tile;
                const auto [y, x, h, w] = layout.rectangle(index);
                within = (row - y) * w * layout.pixelBytes + offset % (layout.cols * layout.pixelBytes) - x * layout.pixelBytes;
                available = w * layout.pixelBytes - within % (w * layout.pixelBytes);
            }
            const GeometryChunkKey key{edit.array, index};
            auto found = plan->modified->find(key);
            if (found == plan->modified->end()) {
                auto bytes = plan->reader->chunk(edit.array, index);
                plan->original.emplace(key, bytes);
                found = plan->modified->emplace(key, std::move(bytes)).first;
            }
            const auto count = std::min<size_t>(available, edit.bytes.size() - consumed);
            std::memcpy(found->second.data() + within, edit.bytes.data() + consumed, count);
            consumed += count; offset += count;
        }
    }
    for (auto it = plan->modified->begin(); it != plan->modified->end();) {
        const auto& old = plan->original.at(it->first);
        const auto& next = it->second;
        if (old == next) { it = plan->modified->erase(it); continue; }
        // The old chunk was verified against its manifest hash when read.
        const std::string beforeHash = plan->reader->manifest().at("arrays").at(it->first.first).at("chunks").at(it->first.second);
        GeometryChunkChange change{it->first, beforeHash, geometryPayloadHash(it->first.first, next), {}, {}, {}};
        for (size_t i = 0; i < old.size();) {
            if (old[i] == next[i]) { ++i; continue; }
            const auto begin = i++;
            while (i < old.size() && old[i] != next[i]) ++i;
            change.changes.push_back({int64_t(begin), int64_t(i - begin)});
            change.before.insert(change.before.end(), old.begin() + begin, old.begin() + i);
            change.after.insert(change.after.end(), next.begin() + begin, next.begin() + i);
        }
        plan->manifest["arrays"][it->first.first]["chunks"][it->first.second] = change.afterHash;
        plan->changes.push_back(std::move(change)); ++it;
    }
    plan->after = dependencyGeometry(store, txn, *revision);
    for (const auto& [name, a] : plan->manifest.at("arrays").items()) {
        auto descriptor = a; descriptor.erase("source_revision");
        plan->after.fingerprints[name] = descriptor.dump();
    }
    plan->after.pointIndex = std::make_shared<DependencyGeometry::PointIndex>();
    // Keep the dependency reader independent of the plan's lifetime; capture
    // only buffers and the original reader, avoiding a plan -> lambda cycle.
    auto reader = plan->reader;
    auto modified = plan->modified;
    auto manifest = plan->manifest;
    plan->after.read = [reader, modified, manifest](const std::string& name, int64_t offset, int64_t count) {
        auto result = reader->read(name, offset, count);
        const auto layout = geometryLayout(manifest.at("arrays").at(name));
        for (const auto& [key, bytes] : *modified) {
            if (key.first != name) continue;
            auto copy = [&](int64_t logical, int64_t physical, int64_t length) {
                const auto begin = std::max(offset, logical), end = std::min(offset + count, logical + length);
                if (begin < end) std::memcpy(result.data() + begin - offset, bytes.data() + physical + begin - logical, end - begin);
            };
            if (!layout.tile) copy(key.second * geometryChunkSize, 0, bytes.size());
            else {
                const auto [y, x, h, w] = layout.rectangle(key.second);
                for (int64_t r = 0; r < h; ++r) copy(((y+r)*layout.cols+x)*layout.pixelBytes, r*w*layout.pixelBytes, w*layout.pixelBytes);
            }
        }
        return result;
    };
    bindSequenceIdentities(plan->after,ObjectStoreImplAccess::conn(txn),plan->manifest);
    return plan;
}
}

std::vector<EdgeRecord> ObjectStore::affectedGeometryEditEdges(const Txn& txn, const std::string& objectId,
    const std::vector<GeometryArrayEdit>& edits, const std::string& expected) const
{
    (void)connection(txn);
    const auto objectUuid = formatUuid(parseUuid(objectId));
    DependencyScope scope;
    const auto current=currentGeometry(txn,objectUuid);
    if (!current || current->uuid!=canonicalGeometryToken(expected)) historyConflict(ObjectStoreImplAccess::conn(txn),"geometry: current revision changed; reload before committing");
    const auto plan = prepareEdit(*this, txn, objectUuid, expected, edits);
    if (plan->changes.empty()) return {};
    return affectedEdges(*this, txn, objectUuid, currentGeometry(txn, objectUuid), plan->after);
}

namespace {
// Structural snapshots compare by array fingerprint: an edge whose arrays are
// untouched is unaffected, even when it records a point that no longer exists.
detail::DependencyGeometry structuralAfter(const GeometryData& data, const nlohmann::json& manifest)
{
    auto after=detail::dependencyGeometry(data);
    after.gridOrigin=manifest.value("grid_origin",std::array<int64_t,2>{0,0});
    for(const auto& [name,a]:manifest.at("arrays").items())after.fingerprints[name]=a.dump();
    return after;
}
}

std::vector<EdgeRecord> detail::structuralSnapshotEdges(const ObjectStore& store, const Txn& txn, const std::string& objectUuid,
    const GeometryData& data, const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds)
{
    DependencyScope scope;
    const auto current=store.currentGeometry(txn,objectUuid);
    if (!current) throw std::invalid_argument("structural edits require existing geometry");
    auto manifest=geometryManifest(data,revisionSurfaceTileSize(txn,*current));
    manifest["grid_origin"]=current->gridOrigin;
    return affectedEdges(store,txn,objectUuid,current,structuralAfter(data,manifest),changedIds);
}

std::string detail::publishStructuralSnapshot(ObjectStore& store, WriteTxn& txn, const std::string& objectUuid,
    const GeometryData& data, nlohmann::json previous, const std::string& expected, const std::vector<GeometryEdgeEdit>& edgeEdits,
    const std::optional<std::pair<std::string,std::vector<int64_t>>>& changedIds)
{
    GeometryPhase phase("detail::publishStructuralSnapshot");
    auto* db=ObjectStoreImplAccess::conn(txn);
    DependencyScope scope;
    const auto current=store.currentGeometry(txn,objectUuid);
    if (!current || current->uuid != canonicalGeometryToken(expected)) historyConflict(db,"geometry: current revision changed; reload before committing");
    auto manifest=geometryManifest(data,revisionSurfaceTileSize(txn,*current));
    for(const auto* field:{"sequences","grid_origin"})if(previous.contains(field))manifest[field]=previous.at(field);
    auto after=structuralAfter(data,manifest);
    bindSequenceIdentities(after,db,manifest);
    const auto affected=affectedEdges(store,txn,objectUuid,current,after,changedIds);
    const auto decisions=validateEdgeEdits(store,txn,db,objectUuid,after,affected,edgeEdits);
    const auto object=*store.objectByUuid(txn,objectUuid);
    const auto uuid=newGeometryToken(db,objectUuid); const auto id=*objectIdByUuid(db,parseUuid(objectUuid));
    const auto text=manifest.dump(), hash=hashText(text);
    if (geometryWork) geometryWork->manifestBytesWritten += text.size();
    writeRevision(db, store, object, uuid, &*current, hash, text);
    writeArrays(db,uuid,data.arrays,manifest); clearGeometryMaterialization(db,objectUuid);
    indexGeometryBounds(db,id,uuid,data);
    stampGeometry(db, store, objectUuid, id, hash);
    if (manifest.at("kind")=="point_collection" && changedIds && changedIds->first=="point_ids")
        updateSourcePointIdentities(db,objectUuid,current->uuid,uuid,manifest,changedIds->second);
    applyEdgeDecisions(store,txn,db,affected,decisions);
    return uuid;
}

std::string ObjectStore::editGeometry(WriteTxn& txn, const std::string& objectId,
    const std::vector<GeometryArrayEdit>& edits, const std::string& expected,
    const std::vector<GeometryEdgeEdit>& edgeEdits)
{
    return editGeometry(txn, objectId, edits, expected, edgeEdits, nullptr);
}

std::string ObjectStore::editGeometry(WriteTxn& txn, const std::string& objectId,
    const std::vector<GeometryArrayEdit>& edits, const std::string& expected, const EdgeDecider& decide)
{
    return editGeometry(txn, objectId, edits, expected, {}, &decide);
}

std::string ObjectStore::editGeometry(WriteTxn& txn, const std::string& objectId,
    const std::vector<GeometryArrayEdit>& edits, const std::string& expected,
    const std::vector<GeometryEdgeEdit>& edgeEdits, const EdgeDecider* decide)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto objectUuid = formatUuid(parseUuid(objectId));
    DependencyScope scope;
    const auto currentRevision=currentGeometry(txn,objectUuid);
    if (!currentRevision || currentRevision->uuid!=canonicalGeometryToken(expected)) historyConflict(db,"geometry: current revision changed; reload before committing");
    for (const auto& edit:edits) if ((edit.array=="point_ids" || edit.array=="control_point_ids") && !edit.bytes.empty())
        throw std::invalid_argument("annotated point identities require structural editing");
    const auto plan = prepareEdit(*this, txn, objectUuid, expected, edits);
    const auto current = *currentGeometry(txn, objectUuid);
    if (plan->changes.empty()) {
        if (explicitEdgeDecisions(edgeEdits)) throw std::invalid_argument("geometry is unchanged; edit relationships directly");
        return current.uuid;
    }
    const auto object = *objectByUuid(txn, objectUuid);
    const auto affected = affectedEdges(*this, txn, objectUuid, current, plan->after);
    const auto decisions = validateEdgeEdits(*this, txn, db, objectUuid, plan->after, affected,
                                             decide ? (*decide)(affected) : edgeEdits);
    if (!historyEnabled(db)) {
        const auto data = snapshotData(plan->manifest, plan->after);
        Savepoint savepoint(db);
        std::vector<GeometryEdgeEdit> resolved;
        for (const auto& [id, decision] : decisions) resolved.push_back(decision);
        const auto result = publishStructuralSnapshot(*this, txn, objectUuid, data, plan->manifest, expected, resolved);
        savepoint.release();
        return result;
    }
    // The cells whose area the edit changes, measured before it.
    const auto areaBefore = editedSurfaceCells(db, current.uuid, edits);
    if (geometryDeltaRawSize(plan->changes) > std::min(impl_->options.maxGeometryDeltaBytes, maxGeometryDeltaBytes)) {
        // Too large for one sparse delta: publish a full revision. Unchanged
        // arrays are inherited; changed arrays reference their chunks by hash,
        // storing only bytes the store does not hold yet.
        auto manifest = plan->manifest;
        std::set<std::string> changed;
        for (const auto& change : plan->changes) changed.insert(change.key.first);
        for (auto& [name, a] : manifest.at("arrays").items()) {
            if (changed.contains(name)) a.erase("source_revision");
            else if (!a.contains("source_revision")) a["source_revision"] = current.uuid;
        }
        const auto text = manifest.dump(), hash = hashText(text), uuid = util::newUuid();
        Savepoint savepoint(db);
        writeRevision(db, *this, object, uuid, &current, hash, text);
        Statement stored(db, "SELECT 1 FROM geometry_chunks WHERE hash=?");
        Statement insert(db, "INSERT INTO geometry_chunks(hash,data) VALUES(?,?)");
        Statement reference(db, "INSERT INTO geometry_revision_chunks VALUES(?,?,?,?)");
        for (const auto& name : changed) {
            const auto& array = manifest.at("arrays").at(name);
            for (int64_t index = 0; index < geometryLayout(array).chunkCount(); ++index) {
                const std::string chunk = array.at("chunks").at(index);
                const auto digest=parseDigest(chunk);
                stored.reset(); stored.bind(1, digest);
                if (!stored.row()) {
                    const auto found = plan->modified->find({name, index});
                    if (found != plan->modified->end()) insert.run(digest, found->second);
                    else insert.run(digest, plan->reader->chunk(name, index));
                }
                reference.run(parseUuid(uuid), name, index, digest);
            }
        }
        clearGeometryMaterialization(db, objectUuid);
        const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
        updateEditedGeometryBounds(db, id, uuid, edits, areaBefore);
        stampGeometry(db, *this, objectUuid, id, hash);
        applyEdgeDecisions(*this, txn, db, affected, decisions);
        savepoint.release();
        return uuid;
    }
    auto encoded = encodeGeometryDelta(plan->changes);
    Statement old(db, "SELECT manifest FROM geometry_revisions WHERE uuid=?", parseUuid(current.uuid)); old.row();
    const auto previous = loadGeometryManifest(db, current.uuid, old.text(0), current.contentHash);
    encoded.description["base"] = previous.contains("delta") ? previous.at("delta").at("base") : nlohmann::json(current.uuid);
    encoded.description["parent_hash"] = current.contentHash;
    encoded.description["result_hash"] = hashText(plan->manifest.dump());
    auto descriptor = plan->manifest;
    for (auto& a : descriptor.at("arrays")) a.erase("chunks");
    descriptor["delta"] = encoded.description;
    const auto text=descriptor.dump(),fullText=plan->manifest.dump(),hash=hashText(fullText),uuid=util::newUuid();
    Savepoint savepoint(db);
    writeRevision(db,*this,object,uuid,&current,hash,text,&fullText);
    execute(db, "INSERT INTO geometry_deltas VALUES(?,?,?)", parseUuid(uuid), encoded.bytes, parseDigest(hashText(text)));
    // A snapshot head has no mutable overlay. Delta heads preserve unchanged
    // overlay chunks; replacing changed rows never retains full old tiles.
    if (!previous.contains("delta")) clearGeometryMaterialization(db, objectUuid);
    Statement put(db, "INSERT INTO geometry_current_chunks VALUES(?,?,?,?,?) ON CONFLICT(object_uuid,array_name,chunk_index) "
        "DO UPDATE SET hash=excluded.hash,data=excluded.data");
    for (const auto& change : plan->changes) {
        const auto& bytes = plan->modified->at(change.key);
        put.run(parseUuid(objectUuid), change.key.first, change.key.second, parseDigest(change.afterHash), bytes);
    }
    execute(db, "INSERT INTO geometry_materializations VALUES(?,?,?) ON CONFLICT(object_uuid) "
        "DO UPDATE SET revision_uuid=excluded.revision_uuid,manifest=excluded.manifest", parseUuid(objectUuid),
        parseUuid(uuid),fullText);
    const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
    updateEditedGeometryBounds(db, id, uuid, edits, areaBefore);
    stampGeometry(db, *this, objectUuid, id, hash);
    applyEdgeDecisions(*this, txn, db, affected, decisions);
    savepoint.release();
    return uuid;
}

std::string ObjectStore::updateSurfaceChannels(WriteTxn& txn, const std::string& objectId,
    const std::vector<GeometryArray>& channels, const std::vector<std::string>& remove,
    const std::string& expected, const std::vector<GeometryEdgeEdit>& edgeEdits)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto objectUuid = formatUuid(parseUuid(objectId));
    DependencyScope scope;
    const auto current = currentGeometry(txn, objectUuid);
    if (!current || current->uuid != canonicalGeometryToken(expected))
        historyConflict(db, "geometry: current revision changed; reload before committing");
    if (current->kind != "surface") throw std::invalid_argument("surface channels require a surface");
    auto reader = std::make_shared<GeometryReader>(txn, current->uuid);
    const auto original = reader->manifest();
    auto manifest = original;
    auto& arrays = manifest.at("arrays");
    const auto grid = arrays.at("points").at("shape").get<std::vector<int64_t>>();
    const auto tile = revisionSurfaceTileSize(txn, *current);
    std::set<std::string> names;
    for (const auto& name : remove) {
        if (name.empty() || name == "points" || !names.insert(name).second)
            throw std::invalid_argument("invalid/duplicate surface channel name: " + name);
        if (!arrays.erase(name)) throw std::invalid_argument("surface channel not found: " + name);
    }
    for (const auto& array : channels) {
        if (array.name.empty() || array.name == "points" || !names.insert(array.name).second)
            throw std::invalid_argument("invalid/duplicate surface channel name: " + array.name);
        validateSurfaceChannel(array, grid);
        nlohmann::json descriptor = {{"dtype", array.dtype}, {"shape", array.shape},
            {"byte_length", array.bytes.size()}, {"chunks", nlohmann::json::array()}};
        if (tile) descriptor["storage"] = {{"version", 1}, {"layout", "tiles"}, {"tile_size", tile}};
        forEachGeometryChunk(array, tile, [&](int64_t, auto bytes) { descriptor["chunks"].push_back(geometryPayloadHash(array.name, bytes)); });
        arrays[array.name] = std::move(descriptor);
    }
    auto logical = [](nlohmann::json value) {
        for (auto& a : value.at("arrays")) a.erase("source_revision");
        return value;
    };
    if (logical(manifest) == logical(original)) {
        if (explicitEdgeDecisions(edgeEdits)) throw std::invalid_argument("geometry is unchanged; edit relationships directly");
        return current->uuid;
    }
    auto after = dependencyGeometry(*this, txn, *current);
    after.arrays.clear(); after.fingerprints.clear();
    for (auto& [name, a] : arrays.items()) {
        after.arrays.emplace(name, GeometryArrayInfo{name, a.at("dtype"), a.at("shape"), a.at("byte_length")});
        auto descriptor = a; descriptor.erase("source_revision"); after.fingerprints[name] = descriptor.dump();
        if (names.contains(name)) continue;
        // Flatten references when the inherited array hasn't been changed by
        // an intervening sparse delta. No payload reads are needed here.
        std::string source = current->uuid;
        if (a.contains("source_revision")) {
            GeometryReader inherited(db, a.at("source_revision").get<std::string>(), false);
            if (inherited.manifest().at("arrays").at(name).at("chunks") == a.at("chunks"))
                source = a.at("source_revision").get<std::string>();
        }
        a["source_revision"] = source;
    }
    // Point dependencies still use the unchanged points reader. Channel
    // dependencies compare descriptors/hashes, including absence.
    const auto affected = affectedEdges(*this, txn, objectUuid, current, after);
    const auto decisions = validateEdgeEdits(*this, txn, db, objectUuid, after, affected, edgeEdits);
    const auto object = *objectByUuid(txn, objectUuid);
    if (!historyEnabled(db)) {
        auto data = snapshotData(original, dependencyGeometry(*this, txn, *current));
        std::erase_if(data.arrays, [&](const GeometryArray& a) { return names.contains(a.name); });
        data.arrays.insert(data.arrays.end(), channels.begin(), channels.end());
        Savepoint savepoint(db);
        const auto result = publishStructuralSnapshot(*this, txn, objectUuid, data, original, expected, edgeEdits);
        savepoint.release();
        return result;
    }
    const auto uuid = util::newUuid(), text = manifest.dump(), hash = hashText(text);
    Savepoint savepoint(db);
    writeRevision(db, *this, object, uuid, &*current, hash, text);
    writeArrays(db, uuid, channels, manifest);
    clearGeometryMaterialization(db, objectUuid);
    const auto id = *objectIdByUuid(db, parseUuid(objectUuid));
    updateEditedGeometryBounds(db, id, uuid, {});
    stampGeometry(db, *this, objectUuid, id, hash);
    applyEdgeDecisions(*this, txn, db, affected, decisions);
    savepoint.release();
    return uuid;
}

std::string ObjectStore::editGeometryRectangle(WriteTxn& txn, const std::string& objectUuid,
    const std::string& name, int64_t row, int64_t column, int64_t rows, int64_t columns,
    const std::vector<uint8_t>& bytes, const std::string& expected, const std::vector<GeometryEdgeEdit>& edgeEdits)
{
    const auto revision = currentGeometry(txn, objectUuid);
    if (!revision || revision->uuid != canonicalGeometryToken(expected))
        historyConflict(static_cast<sqlite3*>(connection(txn)), "geometry: current revision changed; reload before committing");
    if (revision->kind != "surface") throw std::invalid_argument("rectangle edit requires a surface");
    const auto found = std::find_if(revision->arrays.begin(), revision->arrays.end(), [&](const auto& a) { return a.name == name; });
    if (found == revision->arrays.end()) throw std::invalid_argument("geometry: array not found: " + name);
    const auto pixel = surfacePixelBytes(found->dtype, found->shape);
    const auto& shape = found->shape;
    if (row < 0 || column < 0 || rows < 0 || columns < 0 || row > shape[0] || column > shape[1] ||
        rows > shape[0]-row || columns > shape[1]-column || uint64_t(rows*columns*pixel) != bytes.size())
        throw std::invalid_argument("geometry: invalid rectangle or byte length");
    const auto components = shape.size() == 3 ? shape[2] : 1;
    std::vector<GeometryArrayEdit> edits;
    if (columns) for (int64_t r = 0; r < rows; ++r)
        edits.push_back({name, ((row+r)*shape[1]+column)*components,
            {bytes.begin()+r*columns*pixel, bytes.begin()+(r+1)*columns*pixel}});
    return editGeometry(txn, objectUuid, edits, expected, edgeEdits);
}

} // namespace vc::objects

namespace vc::objects {
GeometryData ObjectStore::readGeometryData(const Txn& txn, const std::string& uuid, uint64_t budget) const
{
    return LocalStoreReader(*this, txn).readGeometryData(uuid, budget);
}
}
