#include "internal/EdgeSql.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/SqlValues.hpp"
#include "internal/JsonFields.hpp"
#include "internal/ReferenceResolver.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/Versions.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
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
    return " AND kind=5 AND coalesce(json_type(nullif(meta,''),'$.adjacent'),'false')='" +
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

static constexpr const char* kEdgeCols =
    "uuid,a_uuid,b_uuid,kind,commit_sequence,created_by,created_at,created_host,changed_by,changed_at,changed_host,"
    "approval_state,approved_by,approved_at,approved_host,approved_version,"
    "a_anchor_x,a_anchor_y,a_anchor_z,a_dir_x,a_dir_y,a_dir_z,"
    "b_anchor_x,b_anchor_y,b_anchor_z,b_dir_x,b_dir_y,b_dir_z,meta,a_dependencies,b_dependencies,version,version_generation";

void validateEdgeProjection(const ObjectFields& fields)
{
    if(!fields)return;
    std::set<std::string> valid{"a","b","a_anchor","b_anchor","tags"};
    std::istringstream names(kEdgeCols);std::string name;
    while(std::getline(names,name,',')){
        if(name.starts_with("a_anchor_")||name.starts_with("a_dir_")||name=="a_dependencies")name="a_anchor";
        if(name.starts_with("b_anchor_")||name.starts_with("b_dir_")||name=="b_dependencies")name="b_anchor";
        if(name!="a_uuid"&&name!="b_uuid")valid.insert(name);
    }
    for(const auto& field:*fields)
        if(!valid.contains(field))throw std::invalid_argument("unknown edge projection field: "+field);
}

std::string projectedEdgeColumns(sqlite3* db,const ObjectFields& fields)
{
    validateEdgeProjection(fields);
    std::string sql;
    std::istringstream names(kEdgeCols);std::string name;
    while(std::getline(names,name,',')){
        std::string field=name;
        if(name.starts_with("a_anchor_")||name.starts_with("a_dir_")||name=="a_dependencies")field="a_anchor";
        if(name.starts_with("b_anchor_")||name.starts_with("b_dir_")||name=="b_dependencies")field="b_anchor";
        const bool identity=name=="uuid"||name=="a_uuid"||name=="b_uuid"||name=="kind";
        if(!sql.empty())sql+=',';
        if(!identity&&fields&&!fields->contains(field))sql+="NULL";
        else sql+=edgeReadColumn(db,name);
    }
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

EdgeRecord readEdgeRow(sqlite3* db, const Statement& s, bool details, const ObjectFields& fields,
                      const std::string* initialGeneration)
{
    EdgeRecord rec;
    rec.versionStoreUuid=storeUuidText(db);
    rec.aUuid = formatUuid(s.blob16(1));
    rec.bUuid = formatUuid(s.blob16(2));
    readCommon(rec, s, 0, 3, 4);
    rec.kind=edgeKindName(s.i64(3));
    rec.approvalState=edgeApprovalName(s.isNull(11)?0:s.i64(11));
    if (wants(fields, "a_anchor")) {
        rec.aAnchor = readAnchor(s, 16);
        dependenciesFromJson(rec.aAnchor, parseObjectJson(s.text(29)));
    }
    if (wants(fields, "b_anchor")) {
        rec.bAnchor = readAnchor(s, 22);
        dependenciesFromJson(rec.bAnchor, parseObjectJson(s.text(30)));
    }
    rec.meta = canonicalObjectJson(s.text(28));
    rec.version = s.i64(31);
    if (!s.isNull(32)) rec.versionGeneration = s.uuid(32);
    else if(initialGeneration) rec.versionGeneration=*initialGeneration;
    else {
        Statement lineage(db, "SELECT initial_generation FROM store_manifest WHERE singleton=1");
        if (lineage.row()) rec.versionGeneration = lineage.uuid(0);
    }
    if (!details) return rec;
    Statement tags(db, "SELECT tag FROM edge_tags WHERE edge_uuid=? ORDER BY tag", parseUuid(rec.uuid));
    while (tags.row()) rec.tags.push_back(tags.text(0));
    return rec;
}

std::vector<EdgeRecord> readEdgeRows(sqlite3* db, Statement& rows, const ObjectFields& fields)
{
    std::vector<EdgeRecord> result;
    std::unordered_map<std::string, size_t> positions;
    std::optional<std::string> initialGeneration;
    while (rows.row()) {
        if(rows.isNull(32)&&!initialGeneration){
            Statement lineage(db,"SELECT initial_generation FROM store_manifest WHERE singleton=1");
            if(lineage.row())initialGeneration=lineage.uuid(0);
        }
        positions.emplace(rows.uuid(0), result.size());
        result.push_back(readEdgeRow(db, rows, false, fields,initialGeneration?&*initialGeneration:nullptr));
    }
    if (result.empty()) return result;
    if (wants(fields, "tags")) {
        const auto batch=std::min(size_t(1024),size_t(sqlite3_limit(db,SQLITE_LIMIT_VARIABLE_NUMBER,-1)));
        for(size_t first=0;first<result.size();first+=batch){
            const auto n=std::min(batch,result.size()-first);
            std::string query="SELECT edge_uuid,tag FROM edge_tags WHERE edge_uuid IN (";
            for(size_t i=0;i<n;++i){if(i)query+=',';query+='?';}query+=") ORDER BY edge_uuid,tag";
            Statement tags(db,query);for(size_t i=0;i<n;++i)tags.bind(int(i+1),parseUuid(result[first+i].uuid));
            while(tags.row())result.at(positions.at(tags.uuid(0))).tags.push_back(tags.text(1));
        }
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
    rec.meta=canonicalObjectJson(rec.meta);
    return parseUuid(rec.uuid);
}

void insertEdgeRow(sqlite3* db,const EdgeRecord& rec,
                   int64_t commit_sequence,const std::string& author,const std::string& host,
                   const std::string& now,bool preserveRecord)
{
    const EdgeInsertRow row{&rec,parseUuid(rec.aUuid),parseUuid(rec.bUuid)};
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
    const auto present=edgeColumns(db);const auto specs=edgeColumnSpecs();
    std::vector<size_t> selected;std::string prefix="INSERT INTO edges(";
    for(size_t i=0;i<specs.size();++i)if(present.contains(specs[i].name)){
        if(!selected.empty())prefix+=',';prefix+=specs[i].name;selected.push_back(i);
    }
    prefix+=')';
    const auto unsupported=[&](const char* field,bool supplied){
        if(supplied&&!present.contains(field))throw std::runtime_error("edge field is not supported by this table layout: "+std::string(field));
    };
    for(const auto& row:rows){const auto& r=*row.record;
        unsupported("a_anchor_x",bool(r.aAnchor.position));unsupported("a_anchor_y",bool(r.aAnchor.position));unsupported("a_anchor_z",bool(r.aAnchor.position));
        unsupported("b_anchor_x",bool(r.bAnchor.position));unsupported("b_anchor_y",bool(r.bAnchor.position));unsupported("b_anchor_z",bool(r.bAnchor.position));
        unsupported("a_dir_x",bool(r.aAnchor.direction));unsupported("a_dir_y",bool(r.aAnchor.direction));unsupported("a_dir_z",bool(r.aAnchor.direction));
        unsupported("b_dir_x",bool(r.bAnchor.direction));unsupported("b_dir_y",bool(r.bAnchor.direction));unsupported("b_dir_z",bool(r.bAnchor.direction));
        unsupported("meta",!r.meta.empty()&&r.meta!="{}");
        const auto declared=[](const Anchor& a){return a.wholeGeometry||!a.points.empty()||!a.channels.empty()||!a.metadataKeys.empty();};
        unsupported("a_dependencies",declared(r.aAnchor));unsupported("b_dependencies",declared(r.bAnchor));
        unsupported("approval_state",edgeApprovalCode(r.approvalState)!=0);
        unsupported("approved_by",r.approvedBy.has_value());unsupported("approved_at",r.approvedAt.has_value());
        unsupported("approved_host",r.approvedHost.has_value());unsupported("approved_version",r.approvedVersion.has_value());
        if(preserveRecord){unsupported("version",r.version&&*r.version!=1);unsupported("version_generation",!r.versionGeneration.empty());}
        if(preserveRecord)for(const auto& [field,value]:{std::pair{"created_by",&r.createdBy},{"created_at",&r.createdAt},{"created_host",&r.createdHost},
                {"changed_by",&r.changedBy},{"changed_at",&r.changedAt},{"changed_host",&r.changedHost}})
            unsupported(field,!value->empty());
    }
    writeSqlValuesShared(db,prefix,1,selected.size()-1,rows.size(),sqlBatchRows,[&](size_t offset){
        std::string pattern="(";size_t bound=0;for(size_t i=0;i<selected.size();++i){
            if(i)pattern+=',';pattern+=selected[i]==5?"?1":"?"+std::to_string(offset+(++bound));
        }return pattern+')';
    },[&](Statement& s){s.bind(1,commit_sequence);},[&](Statement& s,size_t index,int offset){
        const auto& row=rows[index];const auto& r=*row.record;int at=offset;
        for(auto field:selected){if(field==5)continue;++at;
            switch(field){
            case 0:s.bind(at,row.binaryUuid?*row.binaryUuid:parseUuid(r.uuid));break;
            case 1:s.bind(at,int64_t(0));break;
            case 2:s.bind(at,row.aUuid);break;case 3:s.bind(at,row.bUuid);break;
            case 4:s.bind(at,edgeKindCode(r.kind));break;
            case 6:s.bind(at,preserveRecord?r.version.value_or(1):int64_t(1));break;
            case 7:if(!preserveRecord||r.versionGeneration.empty())s.bindNull(at);else s.bind(at,parseUuid(r.versionGeneration));break;
            case 8:s.bind(at,preserveRecord?r.createdBy:author);break;
            case 9:s.bind(at,preserveRecord?r.createdAt:now);break;
            case 10:s.bind(at,preserveRecord?r.createdHost:host);break;
            case 11:s.bind(at,preserveRecord?r.changedBy:author);break;
            case 12:s.bind(at,preserveRecord?r.changedAt:now);break;
            case 13:s.bind(at,preserveRecord?r.changedHost:host);break;
            case 14:s.bind(at,edgeApprovalCode(r.approvalState));break;
            case 15:s.bind(at,r.approvedBy);break;case 16:s.bind(at,r.approvedAt);break;
            case 17:s.bind(at,r.approvedHost);break;case 18:s.bind(at,r.approvedVersion);break;
            case 19:case 20:case 21:if(r.aAnchor.position)s.bind(at,(*r.aAnchor.position)[field-19]);else s.bindNull(at);break;
            case 22:case 23:case 24:if(r.aAnchor.direction)s.bind(at,(*r.aAnchor.direction)[field-22]);else s.bindNull(at);break;
            case 25:case 26:case 27:if(r.bAnchor.position)s.bind(at,(*r.bAnchor.position)[field-25]);else s.bindNull(at);break;
            case 28:case 29:case 30:if(r.bAnchor.direction)s.bind(at,(*r.bAnchor.direction)[field-28]);else s.bindNull(at);break;
            case 31:if(r.meta.empty()||r.meta=="{}")s.bindNull(at);else s.bind(at,r.meta);break;
            case 32:case 33:{const auto value=dependenciesToJson(field==32?r.aAnchor:r.bAnchor);if(value.empty())s.bindNull(at);else s.bind(at,value.dump());break;}
            default:throw std::logic_error("unknown edge binding column");
            }
        }
    });
    for(const auto& row:rows)for(const auto& tag:row.record->tags)if(!tag.empty())
        execute(db,"INSERT INTO edge_tags(edge_uuid,tag) VALUES(?,?) ON CONFLICT(edge_uuid,tag) DO NOTHING",row.binaryUuid?*row.binaryUuid:parseUuid(row.record->uuid),tag);
}

EdgeRecord addEdgeImpl(sqlite3* db, const std::string& author, const std::string& host, const EdgeRecord& input,
                        bool preserveRecord, const char* importOperation)
{
    GeometryPhase phase("native.edge.sql");
    Savepoint savepoint(db);
    auto rec=prepareNewEdge(input);
    rejectInitialExtensionMetadata(rec.meta);
    if(preserveRecord&&rec.aUuid!=formatUuid(parseUuid(input.aUuid)))swapEndpoints(rec);
    const EdgeInsertRow identity{&rec,{},{}};requireUnusedObjectUuids(db,std::span(&identity,1));
    if(!lifecycleBindingMatches(db,rec)) throw std::runtime_error("relationship endpoints must be active");
    // A caller explicitly inserting this UUID may replace its deletion marker.
    // Merely reviving an endpoint never reactivates relationships. Historical
    // relationship revisions retain the earlier state independently.
    int64_t nextVersion = 1;
    if(auto* source=edgeSource(db,rec.uuid,false,true)) {
        Statement previous(source, "SELECT version,state FROM "+edgeReadSource(source)+" WHERE uuid=?", parseUuid(rec.uuid));
        if (previous.row()) {
            if(previous.i64(1)!=2)throw std::runtime_error("object store: edge UUID already exists: "+rec.uuid);
            if (previous.i64(0) == INT64_MAX) throw std::overflow_error("relationship version exhausted");
            nextVersion = previous.i64(0) + 1;
        }
    }
    if(nextVersion!=1)requireEdgeColumns(db,{"version","version_generation"});
    if(hasEdgeColumn(db,"state"))execute(db,"DELETE FROM edges WHERE uuid=? AND state=2",parseUuid(rec.uuid));
    insertEdgeRow(db,rec,writeCommitSequence(db),author,host,nowIsoUtc(),preserveRecord);
    if(hasEdgeColumn(db,"version")&&hasEdgeColumn(db,"version_generation"))
        execute(db, "UPDATE edges SET version=?,version_generation=NULLIF((SELECT generation FROM store_manifest),(SELECT initial_generation FROM store_manifest)) WHERE uuid=?",
            nextVersion, parseUuid(rec.uuid));
    if (preserveRecord) restoreCommon(db, "edges", rec);
    appendJournal(db, author, host, parseUuid(rec.uuid), preserveRecord ? rec.commitSequence : writeCommitSequence(db), preserveRecord ? importOperation : "add_edge");
    savepoint.release();
    return rec;
}
} // namespace vc::objects::detail
