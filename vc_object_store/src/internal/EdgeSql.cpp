#include "internal/EdgeSql.hpp"
#include "internal/SqlValues.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects::detail {
namespace {
void bindDependencies(Statement& s,int index,const Anchor& anchor)
{
    if(anchor.points.empty()&&anchor.channels.empty()&&anchor.metadataKeys.empty()){
        validateAnchor(anchor);
        static const std::string empty="{}",whole=R"({"whole_geometry":true})";
        s.bind(index,anchor.wholeGeometry?whole:empty);
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
    "id,uuid,a_id,b_id,kind,generation,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_generation,"
    "a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
    "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,relation_key";

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
        const bool identity = name == "id" || name == "uuid" || name == "kind" || name == "a_id" || name == "b_id";
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

std::string uuidTextByObjectId(sqlite3* db, int64_t id)
{
    Statement s(db, id>0 ? "SELECT uuid FROM objects WHERE id=?" : "SELECT uuid FROM external_object_keys WHERE id=?", id);
    if (!s.row()) throw std::runtime_error("object store: dangling edge endpoint id");
    return formatUuid(s.blob16(0));
}

std::optional<int64_t> endpointIdByUuid(sqlite3* db,const std::array<unsigned char,16>& uuid)
{
    if(auto local=objectIdByUuid(db,uuid))return local;
    Statement s(db,"SELECT id FROM external_object_keys WHERE uuid=?",uuid);
    if(!s.row())return {};
    const auto id=s.i64(0);
    if(s.row())throw std::runtime_error("object store: ambiguous endpoint identity");
    return id;
}

EdgeRecord readEdgeRow(sqlite3* db, const Statement& s, bool details, const ObjectFields& fields)
{
    EdgeRecord rec;
    const int64_t id = s.i64(0);
    const int64_t aId = s.i64(2);
    const int64_t bId = s.i64(3);
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
    rec.relationKey = s.text(32);
    if (!details) return rec;
    rec.aUuid = uuidTextByObjectId(db, aId);
    rec.bUuid = uuidTextByObjectId(db, bId);
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
    Statement endpoints(db, "SELECT e.id,a.uuid,b.uuid FROM edges e "
        "JOIN endpoint_keys a ON a.id=e.a_id JOIN endpoint_keys b ON b.id=e.b_id "
        "WHERE e.id IN (SELECT value FROM json_each(?))", selection);
    while (endpoints.row()) {
        auto& edge = result.at(positions.at(endpoints.i64(0)));
        edge.aUuid = formatUuid(endpoints.blob16(1));
        edge.bUuid = formatUuid(endpoints.blob16(2));
    }
    if (wants(fields, "tags")) {
        Statement tags(db, "SELECT edge_id,tag FROM edge_tags "
            "WHERE edge_id IN (SELECT value FROM json_each(?)) ORDER BY edge_id,tag", selection);
        while (tags.row()) result.at(positions.at(tags.i64(0))).tags.push_back(tags.text(1));
    }
    return result;
}

EdgeRecord prepareNewEdge(const EdgeRecord& input)
{
    if (input.kind.empty())
        throw std::runtime_error("object store: edge kind must be non-empty");
    EdgeRecord rec = input;
    canonicalizeEdge(rec);
    if (rec.uuid.empty())
        rec.uuid = util::newUuid();
    if (rec.meta.empty())
        rec.meta = "{}";
    (void)parseUuid(rec.uuid);
    return rec;
}

void insertEdgeRow(sqlite3* db,const EdgeRecord& rec,int64_t id,int64_t aId,int64_t bId,
                   int64_t generation,const std::string& author,const std::string& host,
                   const std::string& now,bool preserveRecord)
{
    const EdgeInsertRow row{&rec,id,aId,bId};
    insertEdgeRows(db,std::span(&row,1),generation,author,host,now,preserveRecord);
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
        std::string sql="SELECT 1 FROM objects WHERE uuid IN ("+parameters+") UNION ALL "
            "SELECT 1 FROM external_object_keys WHERE uuid IN ("+parameters+") LIMIT 1";
        Statement collision(db,sql);
        for(size_t i=0;i<count;++i)collision.bind(int(i+1),parseUuid(rows[begin+i].record->uuid));
        if(collision.row())throw std::runtime_error("object store: UUID already belongs to an object or endpoint key");
        begin+=count;
    }
}

void insertEdgeRows(sqlite3* db,std::span<const EdgeInsertRow> rows,int64_t generation,
                   const std::string& author,const std::string& host,const std::string& now,
                   bool preserveRecord,size_t sqlBatchRows)
{
    if(!preserveRecord){
        // New records share provenance. Approval defaults are the same as the
        // ordinary insert; avoid rebinding shared/default values per row.
        writeSqlValuesShared(db,"INSERT INTO edges(uuid,a_id,b_id,kind,generation,created_by,created_at,created_host,"
            "changed_by,changed_at,changed_host,a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
            "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,id,relation_key)",
            4,21,rows.size(),sqlBatchRows,[](size_t offset){
                std::string pattern="(";
                for(size_t i=1;i<=21;++i){
                    if(i!=1)pattern+=',';
                    pattern+='?'+std::to_string(offset+i);
                    if(i==4)pattern+=",?1,?2,?3,?4,?2,?3,?4";
                }
                return pattern+')';
            },[&](Statement& s){s.bind(1,generation);s.bind(2,author);s.bind(3,now);s.bind(4,host);},
            [&](Statement& s,size_t index,int offset){
                const auto& row=rows[index];const auto& rec=*row.record;
                s.bind(offset+1,parseUuid(rec.uuid));s.bind(offset+2,row.aId);s.bind(offset+3,row.bId);s.bind(offset+4,rec.kind);
                bindAnchor(s,offset+5,rec.aAnchor);bindAnchor(s,offset+11,rec.bAnchor);
                s.bind(offset+17,rec.meta);bindDependencies(s,offset+18,rec.aAnchor);
                bindDependencies(s,offset+19,rec.bAnchor);s.bind(offset+20,row.id);s.bind(offset+21,rec.relationKey);
            });
    }else{
    writeSqlValues(db,"INSERT INTO edges(uuid,a_id,b_id,kind,generation,created_by,created_at,created_host,changed_by,changed_at,"
                    "changed_host,approval_state,approved_by,approved_at,approved_host,approved_generation,"
                    "a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
                    "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,id,relation_key)",
        "",33,rows.size(),sqlBatchRows,[&](Statement& s,size_t index,int offset){
    const auto& row=rows[index];const auto& rec=*row.record;
    s.bind(offset+1,parseUuid(rec.uuid));s.bind(offset+2,row.aId);s.bind(offset+3,row.bId);
    s.bind(offset+4,rec.kind);s.bind(offset+5,generation);
    s.bind(offset+6,author);s.bind(offset+7,now);s.bind(offset+8,host);
    s.bind(offset+9,author);s.bind(offset+10,now);s.bind(offset+11,host);
    const std::string state = preserveRecord ? rec.approvalState : "pending";
    s.bind(offset+12, state.empty() ? std::string("pending") : state);
    if (preserveRecord && rec.approvedBy) s.bind(offset+13, *rec.approvedBy); else s.bindNull(offset+13);
    if (preserveRecord && rec.approvedAt) s.bind(offset+14, *rec.approvedAt); else s.bindNull(offset+14);
    if (preserveRecord && rec.approvedHost) s.bind(offset+15, *rec.approvedHost); else s.bindNull(offset+15);
    if (preserveRecord && rec.approvedGeneration) s.bind(offset+16, *rec.approvedGeneration); else s.bindNull(offset+16);
    bindAnchor(s, offset+17, rec.aAnchor);
    bindAnchor(s, offset+23, rec.bAnchor);
    s.bind(offset+29, rec.meta);
    bindDependencies(s,offset+30,rec.aAnchor);
    bindDependencies(s,offset+31,rec.bAnchor);
    s.bind(offset+32, row.id);
    s.bind(offset+33, rec.relationKey);
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
    const EdgeInsertRow identity{&rec,0,0,0};requireUnusedObjectUuids(db,std::span(&identity,1));
    auto aId=endpointIdByUuid(db,parseUuid(rec.aUuid)),bId=endpointIdByUuid(db,parseUuid(rec.bUuid));
    if(!aId) throw std::runtime_error("object store: edge endpoint not found: "+rec.aUuid);
    if(!bId) throw std::runtime_error("object store: edge endpoint not found: "+rec.bUuid);
    if(!lifecycleBindingMatches(db,rec)) throw std::runtime_error("relationship endpoints must be active");
    if(*aId>*bId) {std::swap(aId,bId);swapEndpoints(rec);}
    const auto id=allocateRecordId(db,"edges");
    insertEdgeRow(db,rec,id,*aId,*bId,writeGeneration(db),author,host,nowIsoUtc(),preserveRecord);
    if (preserveRecord) restoreCommon(db, "edges", rec);
    appendJournal(db, author, host, parseUuid(rec.uuid), preserveRecord ? rec.generation : writeGeneration(db), preserveRecord ? importOperation : "add_edge");
    savepoint.release();
    return rec;
}
} // namespace vc::objects::detail
