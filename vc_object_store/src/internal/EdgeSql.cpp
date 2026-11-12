#include "internal/EdgeSql.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/SqlValues.hpp"
#include "internal/ReferenceResolver.hpp"
#include "internal/Versions.hpp"
#include "internal/RecordResolver.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects::detail {
namespace {
void bindDependencies(Statement& s,int index,const Anchor& anchor)
{
    if(anchor.points.empty()&&anchor.channels.empty()&&anchor.metadataKeys.empty()){
        validateAnchor(anchor);
        static const std::string empty="{}",whole=R"({"whole_geometry":true})";
        s.bindBorrowedText(index,anchor.wholeGeometry?whole:empty);
    }else s.bind(index,dependenciesToJson(anchor).dump());
}
}
std::string fiberAdjacencySql(std::optional<bool> adjacent)
{
    if (!adjacent) return {};
    // Only JSON booleans (or an absent flag) describe this relation.
    return " AND kind='fiber_branch' AND coalesce(json_type(meta,'$.adjacent'),'false')='" +
        std::string(*adjacent ? "true" : "false") + "'";
}

bool matchesFiberAdjacency(const EdgeRecord& edge, std::optional<bool> adjacent)
{
    if (!adjacent) return true;
    if (edge.kind != "fiber_branch") return false;
    const auto meta = nlohmann::json::parse(edge.meta);
    const auto flag = meta.find("adjacent");
    if (flag == meta.end()) return !*adjacent;
    return flag->is_boolean() && flag->get<bool>() == *adjacent;
}

void validateEdgeSelection(const EdgeSelection& selection)
{
    if (selection.limit && *selection.limit < 0) throw std::invalid_argument("edge limit must be nonnegative");
    for (const auto& state : selection.approvalStates)
        if (state != "pending" && state != "approved" && state != "rejected")
            throw std::invalid_argument("invalid edge approval state: " + state);
    if (selection.afterUuid) (void)parseUuid(*selection.afterUuid);
    for (const auto* ids : {&selection.endpointUuids, &selection.uuids})
        if (*ids) for (const auto& id : **ids) (void)parseUuid(id);
}

bool matchesEdgeSelection(const EdgeRecord& edge, const EdgeSelection& selection)
{
    auto contains = [](const std::vector<std::string>& ids, const std::string& id) {
        return std::any_of(ids.begin(), ids.end(), [&](const auto& value) { return formatUuid(parseUuid(value)) == id; });
    };
    if (selection.kind && edge.kind != *selection.kind) return false;
    if (!selection.attachments && edge.kind == kAttachmentEdgeKind) return false;
    if (!matchesFiberAdjacency(edge, selection.adjacent)) return false;
    if (!selection.approvalStates.empty() && std::find(selection.approvalStates.begin(), selection.approvalStates.end(),
                                                    edge.approvalState) == selection.approvalStates.end()) return false;
    if (selection.tag && std::find(edge.tags.begin(), edge.tags.end(), *selection.tag) == edge.tags.end()) return false;
    if (selection.uuids && !contains(*selection.uuids, edge.uuid)) return false;
    if (selection.afterUuid && edge.uuid <= formatUuid(parseUuid(*selection.afterUuid))) return false;
    if (selection.endpointUuids) {
        const bool a = contains(*selection.endpointUuids, edge.aUuid), b = contains(*selection.endpointUuids, edge.bUuid);
        if (selection.bothEndpoints ? !(a && b) : !(a || b)) return false;
    }
    return true;
}

const char* kEdgeCols =
    "id,uuid,a_uuid,b_uuid,kind,commit_sequence,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_version,"
    "a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
    "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,version,version_generation";

std::string projectedEdgeColumns(const ObjectFields& fields)
{
    if (!fields) return kEdgeCols;
    std::string sql;
    std::set<std::string> valid{"a", "b", "a_anchor", "b_anchor", "tags"};
    std::istringstream input(kEdgeCols);
    std::string name;
    while (std::getline(input, name, ',')) {
        std::string field = name;
        if (name.starts_with("a_anchor_") || name.starts_with("a_dir_") || name == "a_dependencies") field = "a_anchor";
        if (name.starts_with("b_anchor_") || name.starts_with("b_dir_") || name == "b_dependencies") field = "b_anchor";
        const bool identity = name == "id" || name == "uuid" || name == "kind" || name == "a_uuid" || name == "b_uuid";
        if (!identity || name == "uuid" || name == "kind") valid.insert(field);
        if (!sql.empty()) sql += ',';
        sql += identity || fields->contains(field) ? name :
            (name == "meta" || name.ends_with("_dependencies") ? "'{}'" : "NULL");
    }
    for (const auto& field : *fields)
        if (!valid.contains(field)) throw std::invalid_argument("unknown edge projection field: " + field);
    return sql;
}

Anchor readAnchor(const Statement& s, int firstCol)
{
    Anchor a;
    if (!s.isNull(firstCol)) a.position = std::array<double, 3>{s.f64(firstCol), s.f64(firstCol + 1), s.f64(firstCol + 2)};
    if (!s.isNull(firstCol + 3)) a.direction = std::array<double, 3>{s.f64(firstCol + 3), s.f64(firstCol + 4), s.f64(firstCol + 5)};
    return a;
}

void bindAnchor(Statement& s, int firstIdx, const Anchor& a)
{
    for (int i = 0; i < 3; ++i) {
        if (a.position) s.bind(firstIdx + i, (*a.position)[size_t(i)]); else s.bindNull(firstIdx + i);
    }
    for (int i = 0; i < 3; ++i) {
        if (a.direction) s.bind(firstIdx + 3 + i, (*a.direction)[size_t(i)]); else s.bindNull(firstIdx + 3 + i);
    }
}

EdgeRecord readEdgeRow(sqlite3* db, const Statement& s, bool details, const ObjectFields& fields)
{
    EdgeRecord rec;
    rec.versionStoreUuid=storeUuidText(db);
    const int64_t id = s.i64(0);
    rec.aUuid = formatUuid(s.blob16(2));
    rec.bUuid = formatUuid(s.blob16(3));
    readCommon(rec, s, 1, 4, 5);
    if (wants(fields, "a_anchor")) {
        rec.aAnchor = readAnchor(s, 17);
        dependenciesFromJson(rec.aAnchor, nlohmann::json::parse(s.text(30)));
    }
    if (wants(fields, "b_anchor")) {
        rec.bAnchor = readAnchor(s, 23);
        dependenciesFromJson(rec.bAnchor, nlohmann::json::parse(s.text(31)));
    }
    rec.meta = s.text(29);
    rec.version = s.i64(32);
    if (!s.isNull(33)) rec.versionGeneration = s.uuid(33);
    else {
        Statement lineage(db, "SELECT initial_generation FROM store_manifest WHERE singleton=1");
        if (lineage.row()) rec.versionGeneration = lineage.uuid(0);
    }
    if (!details) return rec;
    Statement tags(db, "SELECT tag FROM edge_tags WHERE edge_id=? ORDER BY tag", id);
    while (tags.row()) rec.tags.push_back(tags.text(0));
    return rec;
}

std::vector<EdgeRecord> readEdgeRows(sqlite3* db, Statement& rows, const ObjectFields& fields)
{
    std::vector<EdgeRecord> result;
    std::unordered_map<int64_t, size_t> positions;
    nlohmann::json ids = nlohmann::json::array();
    while (rows.row()) {
        positions.emplace(rows.i64(0), result.size());
        ids.push_back(rows.i64(0));
        result.push_back(readEdgeRow(db, rows, false, fields));
    }
    if (result.empty()) return result;
    const auto selection = ids.dump();
    if (wants(fields, "tags")) {
        Statement tags(db, "SELECT edge_id,tag FROM edge_tags "
            "WHERE edge_id IN (SELECT value FROM json_each(?)) ORDER BY edge_id,tag", selection);
        while (tags.row()) result.at(positions.at(tags.i64(0))).tags.push_back(tags.text(1));
    }
    return result;
}

EdgeRecord prepareNewEdge(const EdgeRecord& input)
{
    EdgeRecord rec=input;
    normalizeNewEdge(rec);
    return rec;
}

std::array<unsigned char,16> normalizeNewEdge(EdgeRecord& rec,bool* swapped)
{
    if (rec.kind.empty())
        throw std::runtime_error("object store: edge kind must be non-empty");
    const bool reversed=canonicalizeEdge(rec);
    if(swapped)*swapped=reversed;
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    if (rec.meta.empty())
        rec.meta = "{}";
    return parseUuid(rec.uuid);
}

void insertEdgeRow(sqlite3* db,const EdgeRecord& rec,int64_t id,
                   int64_t commit_sequence,const std::string& author,const std::string& host,
                   const std::string& now,bool preserveRecord)
{
    const EdgeInsertRow row{&rec,id,parseUuid(rec.aUuid),parseUuid(rec.bUuid)};
    insertEdgeRows(db,std::span(&row,1),commit_sequence,author,host,now,preserveRecord);
}

void requireUnusedObjectUuids(sqlite3* db,std::span<const EdgeInsertRow> rows,size_t sqlBatchRows)
{
    if(!sqlBatchRows)throw std::invalid_argument("UUID check batch size must be positive");
    const auto capacity=std::min(sqlBatchRows,size_t(sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,-1)));
    if(!capacity)throw std::runtime_error("SQLite variable limit cannot hold a UUID");
    for(size_t begin=0;begin<rows.size();) {
        const auto count=std::min(capacity,rows.size()-begin);
        std::string parameters;
        for(size_t i=0;i<count;++i){if(i)parameters+=',';parameters+='?'+std::to_string(i+1);}
        std::string sql="SELECT 1 FROM objects WHERE uuid IN ("+parameters+") LIMIT 1";
        if(resolveReference(db,[&](sqlite3* source){
            Statement collision(source,sql);
            for(size_t i=0;i<count;++i){const auto& row=rows[begin+i];
                collision.bind(int(i+1),row.binaryUuid?*row.binaryUuid:parseUuid(row.record->uuid));}
            return collision.row();
        }))throw std::runtime_error("object store: UUID already belongs to an object");
        begin+=count;
    }
}

void insertEdgeRows(sqlite3* db,std::span<const EdgeInsertRow> rows,int64_t commit_sequence,
                   const std::string& author,const std::string& host,const std::string& now,
                   bool preserveRecord,size_t sqlBatchRows)
{
    if(!preserveRecord){
        // New records share provenance. Approval defaults are the same as the
        // ordinary insert; avoid rebinding shared/default values per row.
        writeSqlValuesShared(db,"INSERT INTO edges(uuid,a_uuid,b_uuid,kind,commit_sequence,created_by,created_at,created_host,"
            "changed_by,changed_at,changed_host,a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
            "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,id)",
            4,20,rows.size(),sqlBatchRows,[](size_t offset){
                std::string pattern="(";
                for(size_t i=1;i<=20;++i){
                    if(i!=1)pattern+=',';
                    pattern+='?'+std::to_string(offset+i);
                    if(i==4)pattern+=",?1,?2,?3,?4,?2,?3,?4";
                }
                return pattern+')';
            },[&](Statement& s){s.bind(1,commit_sequence);s.bindBorrowedText(2,author);s.bindBorrowedText(3,now);s.bindBorrowedText(4,host);},
            [&](Statement& s,size_t index,int offset){
                const auto& row=rows[index];const auto& rec=*row.record;
                s.bind(offset+1,row.binaryUuid?*row.binaryUuid:parseUuid(rec.uuid));s.bind(offset+2,row.aUuid);s.bind(offset+3,row.bUuid);s.bindBorrowedText(offset+4,rec.kind);
                bindAnchor(s,offset+5,rec.aAnchor);bindAnchor(s,offset+11,rec.bAnchor);
                s.bindBorrowedText(offset+17,rec.meta);bindDependencies(s,offset+18,rec.aAnchor);
                bindDependencies(s,offset+19,rec.bAnchor);s.bind(offset+20,row.id);
            });
    }else{
    writeSqlValues(db,"INSERT INTO edges(uuid,a_uuid,b_uuid,kind,commit_sequence,created_by,created_at,created_host,changed_by,changed_at,"
                    "changed_host,approval_state,approved_by,approved_at,approved_host,approved_version,"
                    "a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
                    "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,id)",
        "",32,rows.size(),sqlBatchRows,[&](Statement& s,size_t index,int offset){
    const auto& row=rows[index];const auto& rec=*row.record;
    s.bind(offset+1,row.binaryUuid?*row.binaryUuid:parseUuid(rec.uuid));s.bind(offset+2,row.aUuid);s.bind(offset+3,row.bUuid);
    s.bind(offset+4,rec.kind);s.bind(offset+5,commit_sequence);
    s.bind(offset+6,author);s.bind(offset+7,now);s.bind(offset+8,host);
    s.bind(offset+9,author);s.bind(offset+10,now);s.bind(offset+11,host);
    const std::string state = preserveRecord ? rec.approvalState : "pending";
    s.bind(offset+12, state.empty() ? std::string("pending") : state);
    if (preserveRecord && rec.approvedBy) s.bind(offset+13, *rec.approvedBy); else s.bindNull(offset+13);
    if (preserveRecord && rec.approvedAt) s.bind(offset+14, *rec.approvedAt); else s.bindNull(offset+14);
    if (preserveRecord && rec.approvedHost) s.bind(offset+15, *rec.approvedHost); else s.bindNull(offset+15);
    if (preserveRecord && rec.approvedVersion) s.bind(offset+16, *rec.approvedVersion); else s.bindNull(offset+16);
    bindAnchor(s, offset+17, rec.aAnchor);
    bindAnchor(s, offset+23, rec.bAnchor);
    s.bind(offset+29, rec.meta);
    bindDependencies(s,offset+30,rec.aAnchor);
    bindDependencies(s,offset+31,rec.bAnchor);
    s.bind(offset+32, row.id);
    });
    }
    for(const auto& row:rows) insertTags(db,"edge_tags","edge_id",row.id,row.record->tags);
}

EdgeRecord addEdgeImpl(sqlite3* db, const std::string& author, const std::string& host, const EdgeRecord& input,
                        bool preserveRecord, const char* importOperation)
{
    GeometryPhase phase("native.edge.sql");
    Savepoint savepoint(db);
    auto rec=prepareNewEdge(input);
    rejectInitialExtensionMetadata(rec.meta);
    if(preserveRecord&&rec.aUuid!=formatUuid(parseUuid(input.aUuid)))swapEndpoints(rec);
    const EdgeInsertRow identity{&rec,0,{},{}};requireUnusedObjectUuids(db,std::span(&identity,1));
    if(!lifecycleBindingMatches(db,rec)) throw std::runtime_error("relationship endpoints must be active");
    // A caller explicitly inserting this UUID may replace its deletion marker.
    // Merely reviving an endpoint never reactivates relationships. Historical
    // relationship revisions retain the earlier state independently.
    int64_t nextVersion = 1;
    if(auto* source=edgeSource(db,rec.uuid,false,true)) {
        Statement previous(source, "SELECT version,state FROM edges WHERE uuid=?", parseUuid(rec.uuid));
        if (previous.row()) {
            if(previous.i64(1)!=2)throw std::runtime_error("object store: edge UUID already exists: "+rec.uuid);
            if (previous.i64(0) == INT64_MAX) throw std::overflow_error("relationship version exhausted");
            nextVersion = previous.i64(0) + 1;
        }
    }
    execute(db,"DELETE FROM edges WHERE uuid=? AND state=2",parseUuid(rec.uuid));
    const auto id=allocateRecordId(db,"edges");
    insertEdgeRow(db,rec,id,writeCommitSequence(db),author,host,nowIsoUtc(),preserveRecord);
    execute(db, "UPDATE edges SET version=?,version_generation=NULLIF((SELECT generation FROM store_manifest),(SELECT initial_generation FROM store_manifest)) WHERE id=?",
        nextVersion, id);
    if (preserveRecord) restoreCommon(db, "edges", rec);
    appendJournal(db, author, host, parseUuid(rec.uuid), preserveRecord ? rec.commitSequence : writeCommitSequence(db), preserveRecord ? importOperation : "add_edge");
    savepoint.release();
    return rec;
}
} // namespace vc::objects::detail
