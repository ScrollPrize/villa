#include "internal/RecordSql.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/RecordFields.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/Versions.hpp"
#include "internal/EdgeLayout.hpp"
#include "internal/JsonFields.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "geometry/GeometryToken.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/RecordJson.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/Project.hpp"

namespace vc::objects::detail {
bool preciseObjectTimestamp(std::string_view kind)
{
    return kind == kProjectKind || isProjectResourceKind(kind);
}

const char* kObjectCols =
    "id,uuid,kind,name,commit_sequence,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_version,format_version,storage_kind,"
    "storage_ref,version,data_checksum,meta,version_generation";

ExplicitRecordId::ExplicitRecordId(sqlite3* db,const char* keySpace,const std::string& uuid,int64_t id)
    :db_(db),keySpace_(keySpace),id_(id),save_(db)
{
    if(keySpace_!="objects"&&keySpace_!="edges")throw std::invalid_argument("object store: invalid explicit key space");
    Statement range(db_,"SELECT lower_bound,upper_bound,next_id FROM key_allocators WHERE name=?",keySpace_);
    if(!range.row()||id_<range.i64(0)||id_>range.i64(1))throw std::invalid_argument("object store: explicit ID outside reservation");
    Statement occupied(db_,"SELECT 1 FROM "+keySpace_+" WHERE id=?",id_);
    if(occupied.row())throw std::invalid_argument("object store: explicit insertion ID already exists");
    if(!uuid.empty()){
        Statement existing(db_,"SELECT 1 FROM "+keySpace_+" WHERE uuid=?",parseUuid(uuid));
        if(existing.row())throw std::invalid_argument("object store: explicit insertion UUID already exists");
    }
    if(!range.isNull(2))next_=range.i64(2);
    execute(db_,"UPDATE key_allocators SET next_id=? WHERE name=?",id_,keySpace_);
}
void ExplicitRecordId::finish()
{
    if(!next_||*next_>id_)execute(db_,"UPDATE key_allocators SET next_id=? WHERE name=?",next_,keySpace_);
    save_.release();
}

bool wants(const ObjectFields& fields, const std::string& name) {
    return !fields || fields->contains(name) || name == "uuid" || name == "kind";
}

std::string projectedObjectColumns(const ObjectFields& fields)
{
    if (!fields) return kObjectCols;
    const std::vector<std::pair<std::string, std::string>> columns{
        {"uuid", "NULL"}, {"kind", "''"}, {"name", "''"}, {"commit_sequence", "0"},
        {"created_by", "''"}, {"created_at", "''"}, {"created_host", "''"},
        {"changed_by", "''"}, {"changed_at", "''"}, {"changed_host", "''"},
        {"approval_state", "''"}, {"approved_by", "NULL"}, {"approved_at", "NULL"},
        {"approved_host", "NULL"}, {"approved_version", "NULL"}, {"format_version", "0"},
        {"storage_kind", "''"}, {"storage_ref", "''"}, {"version", "0"},
        {"data_checksum", "NULL"}, {"meta", "'{}'"}, {"version_generation", "NULL"}};
    std::set<std::string> valid{"tags", "fiber", "surface", "track", "retired", "lifecycle_epoch"};
    std::string sql = "id";
    for (const auto& [name, fallback] : columns) {
        valid.insert(name);
        // Publication identity accompanies every listing, independently of
        // the requested content projection, in the same indexed row read.
        sql += "," + ((wants(fields, name)||name=="version"||name=="version_generation") ? name : fallback);
    }
    for (const auto& field : *fields)
        if (!valid.contains(field)) throw std::invalid_argument("unknown object projection field: " + field);
    return sql;
}

void appendJournal(sqlite3* db, const std::string& author, const std::string& hostname,
                   const std::array<unsigned char, 16>& uuid, int64_t commit_sequence, const char* changeKind,
                   const std::string& deletedKind)
{
    if (!historyEnabled(db)) {
        recordCurrentChange(db, uuid, changeKind);
        return;
    }
    commit_sequence = writeCommitSequence(db);
    Statement state(db, "SELECT kind,approval_state,approved_by,approved_at,approved_host,approved_version,0 FROM objects WHERE uuid=? AND state<>2 "
        "UNION ALL SELECT kind,approval_state,approved_by,approved_at,approved_host,approved_version,1 FROM "+edgeReadSource(db)+" WHERE uuid=? AND state=0",
        uuid, uuid);
    const bool hasState = state.row();
    // Deleted entries have no current approval state. Their caller supplies
    // the record kind even though the UUID row remains as a tombstone.
    const bool edge=hasState&&state.i64(6);
    const std::string kind = hasState ? (edge?edgeKindName(state.i64(0)):state.text(0)) : deletedKind;
    if (kind.empty()) throw std::logic_error("object store: journal entry without record kind");
    Statement s(db, "INSERT INTO journal(record_uuid,record_kind,commit_sequence,author,hostname,timestamp,change_kind,"
                    "approval_state,approved_by,approved_at,approved_host,approved_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        uuid, kind, commit_sequence, author, hostname, nowIsoUtc(), std::string(changeKind));
    for (int i = 0; i < 4; ++i) {
        if (hasState && !state.isNull(1 + i)) s.bind(8 + i, edge&&i==0?edgeApprovalName(state.i64(1)):state.text(1 + i));
        else s.bindNull(8 + i);
    }
    if (hasState && !state.isNull(5)) s.bind(12, state.i64(5));
    else s.bindNull(12);
    s.stepDone();
}

void readCommon(RecordCommon& rec, const Statement& s, int uuidCol, int kindCol, int genCol)
{
    rec.uuid = formatUuid(s.blob16(uuidCol));
    rec.kind = s.text(kindCol);
    rec.commitSequence = s.i64(genCol);
    rec.createdBy = s.text(genCol + 1);
    rec.createdAt = s.text(genCol + 2);
    rec.createdHost = s.text(genCol + 3);
    rec.changedBy = s.text(genCol + 4);
    rec.changedAt = s.text(genCol + 5);
    rec.changedHost = s.text(genCol + 6);
    rec.approvalState = s.text(genCol + 7);
    rec.approvedBy = s.optText(genCol + 8);
    rec.approvedAt = s.optText(genCol + 9);
    rec.approvedHost = s.optText(genCol + 10);
    if (!s.isNull(genCol + 11))
        rec.approvedVersion = s.i64(genCol + 11);
}

std::optional<std::array<double, 6>> readBbox(const Statement& s, int firstCol)
{
    if (s.isNull(firstCol)) return std::nullopt;
    std::array<double, 6> box{};
    for (int i = 0; i < 6; ++i) box[size_t(i)] = s.f64(firstCol + i);
    return box;
}

void bindBbox(Statement& s, int firstIdx, const std::optional<std::array<double, 6>>& box)
{
    for (int i = 0; i < 6; ++i) {
        if (box) s.bind(firstIdx + i, (*box)[size_t(i)]);
        else s.bindNull(firstIdx + i);
    }
}

namespace {
template<class T> T readDetail(const Statement& row)
{
    T result;
    int column = 0;
    detailFields(result, [&](const char*, auto& value) {
        using Field = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Field, std::string>) value = row.text(column++);
        else if constexpr (std::is_same_v<Field, double>) value = row.f64(column++);
        else if constexpr (std::is_integral_v<Field>) value = row.i64(column++);
        else if constexpr (std::is_same_v<Field, std::array<double, 2>>) {
            value = {row.f64(column), row.f64(column + 1)};
            column += 2;
        } else {
            value = readBbox(row, column);
            column += 6;
        }
    });
    return result;
}

template<class Visit> void detailTables(Visit visit)
{
    visit("fiber", "fibers", "family,manual_family,control_point_count,line_point_count,length,width,"
        "width_gap_fraction,optimization_mode,bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z",
        &ObjectRecord::fiber, readDetail<FiberDetail>, 14);
    visit("surface", "surfaces", "grid_width,grid_height,scale,scale_y,area,"
        "bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z",
        &ObjectRecord::surface, readDetail<SurfaceDetail>, 11);
    visit("track", "tracks", "dataset_ref,external_key,family,point_count,length,"
        "bbox_min_x,bbox_min_y,bbox_min_z,bbox_max_x,bbox_max_y,bbox_max_z",
        &ObjectRecord::track, readDetail<TrackDetail>, 11);
}
}

namespace {
std::string readObjectVersionGeneration(sqlite3* db,const Statement& s,int generationColumn,const std::string* initialGeneration)
{
    if(!s.isNull(generationColumn))return s.uuid(generationColumn);
    if(initialGeneration)return *initialGeneration;
    Statement lineage(db,"SELECT initial_generation FROM store_manifest WHERE singleton=1");
    return lineage.row()?lineage.uuid(0):std::string{};
}
}
ObjectVersion readObjectReferenceColumns(sqlite3* db,const Statement& s,int uuidColumn,int versionColumn,int generationColumn,
    const std::string* initialGeneration,const std::string* ownerStore,const std::string* selectedUuid)
{
    return {ownerStore?*ownerStore:storeUuidText(db),readObjectVersionGeneration(db,s,generationColumn,initialGeneration),
            selectedUuid?*selectedUuid:s.uuid(uuidColumn),s.i64(versionColumn)};
}

ObjectRecord readObjectRow(sqlite3* db, const Statement& s, bool details,
                           const std::string* initialGeneration, bool lifecycle, const std::string* ownerStore)
{
    ObjectRecord rec;
    rec.versionStoreUuid=ownerStore?*ownerStore:storeUuidText(db);
    const int64_t id = s.i64(0);
    readCommon(rec, s, 1, 2, 4);
    rec.name = s.text(3);
    rec.formatVersion = s.i64(16);
    rec.storageKind = s.text(17);
    rec.storageRef = s.text(18);
    rec.version=s.i64(19);
    rec.versionGeneration=readObjectVersionGeneration(db,s,22,initialGeneration);
    rec.dataChecksum = s.optText(20);
    rec.meta = canonicalObjectJson(s.text(21));
    if (lifecycle) readLifecycle(db, id, rec);

    if (!details) return rec;
    {
        Statement tags(db, "SELECT tag FROM object_tags WHERE object_id=? ORDER BY tag", id);
        while (tags.row()) rec.tags.push_back(tags.text(0));
    }
    detailTables([&](const char* kind, const char* table, const char* columns, auto member, auto read, int) {
        if (rec.kind != kind) return;
        static const auto sql = std::string("SELECT ") + columns + " FROM " + table + " WHERE object_id=?";
        Statement detail(db, sql, id);
        if (detail.row()) rec.*member = read(detail);
    });
    return rec;
}

std::vector<ObjectRecord> readObjectRows(sqlite3* db, Statement& rows, const ObjectFields& fields)
{
    const bool hydrate = wants(fields, "tags") || wants(fields, "fiber") || wants(fields, "surface") || wants(fields, "track");
    const bool lifecycle = wants(fields, "retired") || wants(fields, "lifecycle_epoch");
    std::string initialGeneration,ownerStore;
    Statement lineage(db, "SELECT m.initial_generation,i.uuid FROM store_manifest m CROSS JOIN store_identity i WHERE m.singleton=1");
    if (!lineage.row()) throw std::runtime_error("object store: missing publication identity");
    initialGeneration = lineage.uuid(0);ownerStore = lineage.uuid(1);
    std::vector<ObjectRecord> result;
    std::vector<int64_t> ids;
    while (rows.row()) {
        if (hydrate || lifecycle) ids.push_back(rows.i64(0));
        result.push_back(readObjectRow(db, rows, false, &initialGeneration, false, &ownerStore));
    }
    hydrateObjectRows(db,result,ids,fields);
    return result;
}

void hydrateObjectRows(sqlite3* db,std::vector<ObjectRecord>& result,std::span<const int64_t> ids,
                       const ObjectFields& fields)
{
    const bool hydrate = wants(fields, "tags") || wants(fields, "fiber") || wants(fields, "surface") || wants(fields, "track");
    const bool lifecycle = wants(fields, "retired") || wants(fields, "lifecycle_epoch");
    for(auto& record:result){
        if(!wants(fields,"retired"))record.retired=false;
        if(!wants(fields,"lifecycle_epoch"))record.lifecycleEpoch.clear();
    }
    if (result.empty() || (!hydrate && !lifecycle)) return;
    if(result.size()!=ids.size())throw std::invalid_argument("object row hydration needs one ID per row");
    std::unordered_map<int64_t,size_t> positions;
    positions.reserve(ids.size());
    std::set<std::string> kinds;
    for(size_t i=0;i<ids.size();++i){
        positions.emplace(ids[i],i);
        if(hydrate)kinds.insert(result[i].kind);
    }
    const auto selection = nlohmann::json(std::vector<int64_t>(ids.begin(),ids.end())).dump();
    if (lifecycle) {
        Statement states(db, "SELECT id,state,lifecycle_epoch FROM objects WHERE id IN (SELECT value FROM json_each(?))", selection);
        while (states.row()) {
            auto& record = result.at(positions.at(states.i64(0)));
            if (wants(fields, "retired")) record.retired = states.i64(1) == 1;
            if (wants(fields, "lifecycle_epoch")) record.lifecycleEpoch = states.text(2);
        }
    }
    if (wants(fields, "tags")) {
        Statement tags(db, "SELECT object_id,tag FROM object_tags WHERE object_id IN (SELECT value FROM json_each(?)) ORDER BY object_id,tag",
            selection);
        while (tags.row()) result.at(positions.at(tags.i64(0))).tags.push_back(tags.text(1));
    }
    detailTables([&](const char* kind, const char* table, const char* columns, auto member, auto read, int idColumn) {
        if (!kinds.contains(kind) || !wants(fields, kind)) return;
        static const auto sql = std::string("SELECT ") + columns + ",object_id FROM " + table +
            " WHERE object_id IN (SELECT value FROM json_each(?))";
        Statement detail(db, sql, selection);
        while (detail.row()) result.at(positions.at(detail.i64(idColumn))).*member = read(detail);
    });
}

std::optional<int64_t> objectIdByUuid(sqlite3* db, const std::array<unsigned char, 16>& uuid)
{
    Statement s(db, "SELECT id FROM objects WHERE uuid=?", uuid);
    if (!s.row()) return std::nullopt;
    return s.i64(0);
}

namespace {
void validateObjectDetail(const ObjectRecord& rec)
{
    objectDetails(rec, [&](const char* kind, const auto& detail) {
        if (detail && rec.kind != kind)
            throw std::runtime_error(std::string("object store: ") + kind + " detail on object of kind " + rec.kind);
    });
}
}

void writeObjectDetail(sqlite3* db, int64_t id, const ObjectRecord& rec)
{
    // A surface's area is the store's (refreshSurfaceArea), kept across record
    // writes, as is its detail row when a record carries none.
    std::optional<double> storedArea;
    {
        Statement area(db, "SELECT area FROM surfaces WHERE object_id=?", id);
        if (area.row()) storedArea = area.f64(0);
    }
    for (const char* table : {"fibers", "surfaces", "tracks"}) {
        if (storedArea && rec.kind == "surface" && !rec.surface && std::string(table) == "surfaces") continue;
        execute(db, std::string("DELETE FROM ") + table + " WHERE object_id=?", id);
    }
    insertObjectDetail(db,id,rec);
    if (storedArea && rec.surface) execute(db, "UPDATE surfaces SET area=? WHERE object_id=?", *storedArea, id);
}

void insertObjectDetail(sqlite3* db,int64_t id,const ObjectRecord& rec)
{
    bool written = false;
    detailTables([&](const char*, const char* table, const char* columns, auto member, auto, int count) {
        const auto& detail = rec.*member;
        if (written || !detail) return;
        static const auto sql = [&] {
            std::string query = std::string("INSERT INTO ") + table + "(object_id," + columns + ") VALUES(?";
            for (int i = 0; i < count; ++i) query += ",?";
            return query + ")";
        }();
        Statement row(db, sql, id);
        int next = 2;
        detailFields(*detail, [&](const char*, const auto& value) {
            using Field = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Field, bool>) row.bind(next++, int64_t(value));
            else if constexpr (std::is_same_v<Field, std::array<double, 2>>)
                for (auto axis : value) row.bind(next++, axis);
            else if constexpr (std::is_same_v<Field, std::optional<std::array<double, 6>>>) {
                bindBbox(row, next, value);
                next += 6;
            } else row.bind(next++, value);
        });
        row.stepDone();
        written = true;
    });
}

void writeTags(sqlite3* db, const char* table, const char* idColumn, int64_t id, const std::vector<std::string>& tags)
{
    execute(db, std::string("DELETE FROM ") + table + " WHERE " + idColumn + "=?", id);
    insertTags(db,table,idColumn,id,tags);
}

void insertTags(sqlite3* db, const char* table, const char* idColumn, int64_t id, const std::vector<std::string>& tags)
{
    if(tags.empty()) return;
    Statement ins(db, std::string("INSERT OR IGNORE INTO ") + table + "(" + idColumn + ",tag) VALUES(?,?)");
    for (const auto& tag : tags) {
        if (tag.empty()) continue;
        ins.run(id, tag);
    }
}

void restoreCommon(sqlite3* db, const char* table, const RecordCommon& rec)
{
    // Preserve authorship/approval evidence, not another database's version
    // counter. The imported state is published here by this transaction.
    const auto commit_sequence = writeCommitSequence(db);
    const auto version = writeRecordVersion(db, table, rec.uuid);
    Statement update(db, std::string("UPDATE ") + table + " SET commit_sequence=?,created_by=?,created_at=?,created_host=?,"
        "changed_by=?,changed_at=?,changed_host=?,approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_version=? WHERE uuid=?",
        commit_sequence, rec.createdBy, rec.createdAt, rec.createdHost, rec.changedBy, rec.changedAt,
        rec.changedHost);
    if(std::string_view(table)=="edges")update.bind(8,edgeApprovalCode(rec.approvalState));else update.bind(8,rec.approvalState);
    update.bind(9,rec.approvedBy);update.bind(10,rec.approvedAt);update.bind(11,rec.approvedHost);
    update.bind(12,rec.approvedVersion?std::optional<int64_t>(version):std::nullopt);update.bind(13,parseUuid(rec.uuid));update.stepDone();
}

void validateGeometrySourceMetadata(sqlite3* db, const std::string& uuid, const nlohmann::json& source)
{
    for (const auto* key : {"point_attributes", "control_points_attributes", "line_points_attributes", "points_attributes"})
        if (source.contains(key) && geometryHead(db, uuid))
            throw std::invalid_argument("point annotations belong to geometry; use sequence editing");
}

ObjectRecord prepareObjectRecord(const ObjectRecord& input)
{
    validateObjectDetail(input);
    if (input.kind.empty())
        throw std::runtime_error("object store: object kind must be non-empty");
    if (input.surface)
        for (double axis : input.surface->scale)
            if (!std::isfinite(axis) || axis < 0)
                throw std::runtime_error("object store: surface scale must be finite and nonnegative");
    ObjectRecord rec = input;
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    rec.meta=canonicalObjectJson(rec.meta);
    (void)parseUuid(rec.uuid);
    if(const auto bounds=recordBounds(rec))validateBounds(*bounds);
    return rec;
}

void normalizeObjectCoordinates(ObjectRecord& rec,bool defaults,sqlite3* existingStore)
{
    if (hasGeometryCoordinates(rec.kind)) {
        auto meta = nlohmann::json::parse(rec.meta);
        normalizeObjectCoordinates(meta,rec.kind,defaults&&(meta.contains("coordinates")||!existingStore||!objectIdByUuid(existingStore,parseUuid(rec.uuid))));
        if(existingStore)validateGeometrySourceMetadata(existingStore,rec.uuid,meta.value("source",nlohmann::json::object()));
        rec.meta = meta.dump();
    }
}

void normalizeObjectCoordinates(nlohmann::json& meta,const std::string& kind,bool defaults)
{
    if(!hasGeometryCoordinates(kind))return;
    if(meta.contains("coordinates"))meta["coordinates"]=geometryCoordinates(meta["coordinates"]);
    else if(defaults)meta["coordinates"]=geometryCoordinates(nlohmann::json::object());
}

void insertObjectRow(sqlite3* db,const ObjectRecord& rec,int64_t id,int64_t commit_sequence,
                     const std::string& author,const std::string& host,const std::string& now,bool preserveRecord,
                     const std::string* geometryManifest)
{
    Statement s(db, "INSERT INTO objects(uuid,kind,name,commit_sequence,created_by,created_at,created_host,changed_by,changed_at,"
                    "changed_host,approval_state,approved_by,approved_at,approved_host,approved_version,format_version,"
                    "storage_kind,storage_ref,version,data_checksum,meta,id,geometry_manifest) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        parseUuid(rec.uuid), rec.kind, rec.name, commit_sequence, author, now, host, author, now, host);
    const std::string state = preserveRecord ? rec.approvalState : "pending";
    s.bind(11, state.empty() ? std::string("pending") : state);
    if (preserveRecord && rec.approvedBy) s.bind(12, *rec.approvedBy); else s.bindNull(12);
    if (preserveRecord && rec.approvedAt) s.bind(13, *rec.approvedAt); else s.bindNull(13);
    if (preserveRecord && rec.approvedHost) s.bind(14, *rec.approvedHost); else s.bindNull(14);
    if (preserveRecord && rec.approvedVersion) s.bind(15, *rec.approvedVersion); else s.bindNull(15);
    s.bind(16, rec.formatVersion); s.bind(17, rec.storageKind); s.bind(18, rec.storageRef); s.bind(19, int64_t(1));
    s.bind(20, rec.dataChecksum);if(rec.meta=="{}"||rec.meta.empty())s.bindNull(21);else s.bind(21,rec.meta);s.bind(22,id);
    if(geometryManifest)s.bindBorrowedText(23,*geometryManifest);else s.bindNull(23);
    s.stepDone();
    if (!immutableInitial(db)) execute(db,
        "UPDATE objects SET version_generation=NULLIF((SELECT generation FROM store_manifest),(SELECT initial_generation FROM store_manifest)) WHERE id=?", id);
}

std::string upsertObjectImpl(sqlite3* db, const std::string& author, const std::string& host, const ObjectRecord& input,
                             bool preserveRecord, const char* importOperation, bool restoringExtensions)
{
    Savepoint savepoint(db);
    auto rec=prepareObjectRecord(input);
    normalizeObjectCoordinates(rec,!preserveRecord,db);
    const auto original=resolveObject(db,rec.uuid);
    if (original && !preserveRecord && !restoringExtensions) {
        if (original->record.kind != rec.kind)
            throw std::runtime_error("object store: object kind cannot change");
        if (original->record.storageKind == "sqlite") {
            rec.storageKind=original->record.storageKind;
            rec.storageRef=original->record.storageRef;
            rec.dataChecksum=original->record.dataChecksum;
        }
        const auto editable=[](ObjectRecord r) {
            static const std::set<std::string> fields{"name","format_version","storage_kind","storage_ref",
                "data_checksum","meta","tags","fiber","surface","track"};
            std::sort(r.tags.begin(),r.tags.end());
            r.tags.erase(std::unique(r.tags.begin(),r.tags.end()),r.tags.end());
            if(r.meta.empty())r.meta="{}";
            return objectToJson(r,&fields);
        };
        if (editable(rec)==editable(original->record)) {savepoint.release();return rec.uuid;}
    }
    if(auto* source=objectSource(db,rec.uuid);source&&source!=db){
        createRecordOverride(db,"objects",rec.uuid);
        const auto rebased=resolveObject(db,rec.uuid)->record;
        if(rec.storageKind==original->record.storageKind&&rec.storageRef==original->record.storageRef)
            rec.storageRef=rebased.storageRef;
        if(rec.track&&original->record.track&&rec.track->datasetRef==original->record.track->datasetRef)
            rec.track->datasetRef=rebased.track->datasetRef;
    }
    const auto uuid=parseUuid(rec.uuid);
    const auto bounds = detail::recordBounds(rec);
    if (resolveReference(db, [&](sqlite3* source) {
        Statement collision(source, "SELECT 1 FROM edges WHERE uuid=?", uuid);
        return collision.row();
    })) throw std::runtime_error("object store: UUID already belongs to an edge");
    const std::string now = nowIsoUtc(preciseObjectTimestamp(rec.kind));
    Statement managed(db, "SELECT version,data_checksum,storage_ref FROM objects WHERE uuid=? AND geometry_manifest IS NOT NULL",
        uuid);
    const bool hasGeometry = managed.row();
    if (hasGeometry) {
        rec.storageKind = "sqlite";
        rec.storageRef = managed.text(2);
        rec.version = managed.i64(0);
        rec.dataChecksum = managed.text(1);
    } else if (rec.storageKind == "sqlite") {
        // A metadata override may retain, never invent/change, an existing
        // managed reference. No geometry payload or descriptor is copied.
        const auto original=resolveObject(db,rec.uuid);
        if(!original||original->record.storageKind!=rec.storageKind||original->record.storageRef!=rec.storageRef||
           original->record.dataChecksum!=rec.dataChecksum)
            throw std::invalid_argument("object store: managed geometry requires putGeometry, not a storage_kind override");
    }

    int64_t id = 0;
    const int64_t commit_sequence = writeCommitSequence(db);
    if (auto existing = objectIdByUuid(db, uuid)) {
        id = *existing;
        Statement cur(db, "SELECT kind,commit_sequence,meta,state FROM objects WHERE id=?", id);
        (void)cur.row();
        if (!restoringExtensions) checkExtensionMetadata(cur.i64(3) == 2 ? "{}" : cur.text(2), rec.meta);
        if (cur.text(0) != rec.kind)
            throw std::runtime_error("object store: object kind cannot change (" + cur.text(0) + " -> " + rec.kind + ")");
        beforeCurrentObjectEdit(db,rec.uuid);
        const auto version = writeObjectVersion(db, rec.uuid);
        execute(db, "UPDATE objects SET state=CASE WHEN state=2 THEN 0 ELSE state END,name=?,commit_sequence=?,changed_by=?,changed_at=?,changed_host=?,"
                        "approval_state='pending',approved_by=NULL,approved_at=NULL,approved_host=NULL,approved_version=NULL,"
                        "format_version=?,storage_kind=?,storage_ref=?,version=?,data_checksum=?,meta=nullif(?,'{}') WHERE id=?",
            rec.name, commit_sequence, author, now, host, rec.formatVersion, rec.storageKind, rec.storageRef,
            version, rec.dataChecksum, rec.meta, id);
    } else {
        if (!restoringExtensions) rejectInitialExtensionMetadata(rec.meta);
        if(!historyEnabled(db)&&nlohmann::json::parse(rec.meta).contains(kArtifactsMetaKey))
            beforeCurrentObjectEdit(db,rec.uuid);
        id = allocateRecordId(db, "objects");
        insertObjectRow(db,rec,id,commit_sequence,author,host,now,preserveRecord);
    }
    writeObjectDetail(db, id, rec);
    if (!hasGeometry) detail::setSpatialBounds(db, id, bounds);
    writeTags(db, "object_tags", "object_id", id, rec.tags);
    if (preserveRecord) {
        restoreCommon(db, "objects", rec);
        writeLifecycle(db, id, rec);
    }
    appendJournal(db, author, host, uuid, preserveRecord ? rec.commitSequence : commit_sequence, preserveRecord ? importOperation : "upsert");
    savepoint.release();
    return rec.uuid;
}

void stampContentChange(sqlite3* db, const std::string& author, const std::string& host, const char* table,
                        int64_t id, const std::array<unsigned char, 16>& uuid, const char* changeKind)
{
    const bool edge=std::string_view(table)=="edges";
    if(edge)requireEdgeColumns(db,{"version","version_generation","changed_by","changed_at","changed_host","approval_state",
                                 "approved_by","approved_at","approved_host","approved_version"});
    writeRecordVersion(db, table, formatUuid(uuid));
    const int64_t commit_sequence = writeCommitSequence(db);
    execute(db, std::string("UPDATE ") + table +
                        " SET commit_sequence=?,changed_by=?,changed_at=?,changed_host=?,approval_state="+(edge?"0":"'pending'")+","
                        "approved_by=NULL,approved_at=NULL,approved_host=NULL,approved_version=NULL WHERE uuid=?",
        commit_sequence, author, nowIsoUtc(), host, uuid);
    appendJournal(db, author, host, uuid, commit_sequence, changeKind);
}

std::string jsonPathForKey(const std::string& key)
{
    std::string out = "$.\"";
    for (char ch : key) {
        if (ch == '"' || ch == '\\') out.push_back('\\');
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::optional<std::string> geometryHead(sqlite3* db, const std::string& objectUuid)
{
    db=objectSource(db,objectUuid);
    if(!db)return std::nullopt;
    Statement s(db,"SELECT storage_ref FROM objects WHERE uuid=? AND state<>2 AND geometry_manifest IS NOT NULL",parseUuid(objectUuid));
    return s.row()?std::optional(s.text(0)):std::nullopt;
}
} // namespace vc::objects::detail
