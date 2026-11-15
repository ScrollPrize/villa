#include "vc_object_store/objects/StoreReader.hpp"
#include "internal/Versions.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometryInternal.hpp"
#include "internal/History.hpp"
#include "internal/ExtensionHistory.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "internal/JsonFields.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/VisibleRows.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace vc::objects {
using namespace detail;
std::string ObjectStore::addEdge(WriteTxn& txn, const EdgeRecord& record)
{
    GeometryPhase phase("native.edge");
    detail::DependencyScope dependencyScope;
    {GeometryPhase geometryPhase("native.edge.geometry");
    for (const auto& [uuid, anchor] : {std::pair{record.aUuid, record.aAnchor}, std::pair{record.bUuid, record.bAnchor}}) {
        // validateDependencies immediately accepts this case. Anchor syntax
        // and finite values are still checked by canonicalizeEdge below.
        if (anchor.wholeGeometry && anchor.points.empty()) continue;
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    }
    }
    auto* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto inserted = addEdgeImpl(db, author(), impl_->hostname, record, false);
    {GeometryPhase validate("native.edge.validation");recordRelationshipValidation(db,inserted);}
    savepoint.release();
    return inserted.uuid;
}

void ObjectStore::updateEdge(WriteTxn& txn, const EdgeRecord& record)
{
    detail::DependencyScope dependencyScope;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto existing = edgeByUuid(txn, record.uuid);
    if (!existing) throw std::runtime_error("object store: edge not found: " + record.uuid);
    auto rec = record;
    rec.meta=canonicalObjectJson(rec.meta);
    canonicalizeEdge(rec);
    checkExtensionMetadata(existing->meta, rec.meta);
    if (rec.aUuid != rec.bUuid && rec.aUuid == existing->bUuid && rec.bUuid == existing->aUuid)
        swapEndpoints(rec);
    if (rec.kind != existing->kind || rec.aUuid != existing->aUuid || rec.bUuid != existing->bUuid)
        throw std::runtime_error("object store: edge kind and endpoints cannot change");
    for (const auto& [uuid, anchor] : {std::pair{rec.aUuid, rec.aAnchor}, std::pair{rec.bUuid, rec.bAnchor}})
        if (const auto revision = currentGeometry(txn, uuid))
            detail::validateDependencies(anchor, detail::dependencyGeometry(*this, txn, *revision));
    std::sort(rec.tags.begin(), rec.tags.end());
    rec.tags.erase(std::remove(rec.tags.begin(),rec.tags.end(),std::string{}),rec.tags.end());
    rec.tags.erase(std::unique(rec.tags.begin(), rec.tags.end()), rec.tags.end());
    Savepoint savepoint(db);
    if (rec.aAnchor == existing->aAnchor && rec.bAnchor == existing->bAnchor &&
        rec.tags == existing->tags && parseObjectJson(rec.meta) == parseObjectJson(existing->meta)) {
        // An unchanged record can still explicitly revalidate edited endpoints.
        // Otherwise keep the original owner, version and transaction sequence.
        Statement changed(db,"SELECT 1 FROM objects WHERE uuid IN (?,?) AND commit_sequence=? LIMIT 1",
            parseUuid(rec.aUuid),parseUuid(rec.bUuid),writeCommitSequence(db));
        if(!changed.row()&&currentRelationshipApplicable(*this,txn,rec.uuid))return;
    }
    createRecordOverride(db, "edges", rec.uuid);
    using Value=std::variant<std::monostate,double,std::string>;
    std::vector<std::pair<std::string,Value>> changes;
    const auto append=[&](const std::string& field,Value value){
        if(!hasEdgeColumn(db,field))throw std::runtime_error("edge field is not supported by this table layout: "+field);
        changes.emplace_back(field,std::move(value));
    };
    const auto vector=[&](const char* prefix,const auto& before,const auto& after){
        if(before==after)return;
        for(size_t i=0;i<3;++i)append(std::string(prefix)+"xyz"[i],after?Value((*after)[i]):Value(std::monostate{}));
    };
    vector("a_anchor_",existing->aAnchor.position,rec.aAnchor.position);
    vector("a_dir_",existing->aAnchor.direction,rec.aAnchor.direction);
    vector("b_anchor_",existing->bAnchor.position,rec.bAnchor.position);
    vector("b_dir_",existing->bAnchor.direction,rec.bAnchor.direction);
    const auto meta=canonicalObjectJson(rec.meta);
    if(meta!=canonicalObjectJson(existing->meta))append("meta",meta=="{}"?Value(std::monostate{}):Value(meta));
    const auto a=dependenciesToJson(rec.aAnchor),b=dependenciesToJson(rec.bAnchor);
    if(a!=dependenciesToJson(existing->aAnchor))append("a_dependencies",a.empty()?Value(std::monostate{}):Value(a.dump()));
    if(b!=dependenciesToJson(existing->bAnchor))append("b_dependencies",b.empty()?Value(std::monostate{}):Value(b.dump()));
    if(!changes.empty()){
        std::string sql="UPDATE edges SET ";for(size_t i=0;i<changes.size();++i){if(i)sql+=',';sql+=changes[i].first+"=?";}sql+=" WHERE uuid=?";
        Statement s(db,sql);int at=1;for(const auto& [field,value]:changes){
            std::visit([&](const auto& v){using T=std::decay_t<decltype(v)>;if constexpr(std::is_same_v<T,std::monostate>)s.bindNull(at);else s.bind(at,v);},value);++at;
        }s.bind(at,parseUuid(rec.uuid));s.stepDone();
    }
    if(rec.tags!=existing->tags){
        execute(db,"DELETE FROM edge_tags WHERE edge_uuid=?",parseUuid(rec.uuid));
        for(const auto& tag:rec.tags)execute(db,"INSERT INTO edge_tags VALUES(?,?)",parseUuid(rec.uuid),tag);
    }
    stampContentChange(db, author(), hostname(), "edges", 0, parseUuid(rec.uuid), "update_edge");
    recordRelationshipValidation(db, rec);
    savepoint.release();
}

std::optional<EdgeRecord> ObjectStore::edgeByUuid(const Txn& txn, const std::string& uuid) const
{
    const auto edge=resolveEdge(static_cast<sqlite3*>(connection(txn)),uuid);
    return edge?std::optional(edge->record):std::nullopt;
}

void ObjectStore::removeEdge(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    createRecordOverride(db, "edges", uuid);
    const auto bytes = parseUuid(uuid);
    requireEdgeColumns(db,{"state","changed_by","changed_at","changed_host"});
    Statement row(db, "SELECT uuid,kind FROM edges WHERE uuid=? AND state=0", bytes);
    if (!row.row()) throw std::runtime_error("object store: edge not found: " + uuid);
    const auto kind = edgeKindName(row.i64(1));
    writeRecordVersion(db, "edges", uuid);
    clearCurrentExtensionRows(db, "edges", uuid);
    execute(db,"UPDATE edges SET state=2,commit_sequence=?,changed_by=?,changed_at=?,changed_host=? WHERE uuid=?",
        writeCommitSequence(db),author(),nowIsoUtc(),hostname(),bytes);
    appendJournal(db, author(), impl_->hostname, bytes, 0, "remove_edge", kind);
    savepoint.release();
}

std::vector<EdgeRecord> ObjectStore::edgesOf(const Txn& txn, const std::string& objectUuid, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto key=parseUuid(objectUuid);
    requireAdjacencyIndex(db);
    if(!resolveObject(db,objectUuid))throw std::runtime_error("object store: object not found: "+objectUuid);
    if(querySources(db).size()==1){
        Statement rows(db,"SELECT "+projectedEdgeColumns(db)+" FROM "+edgeReadSource(db)+" AS edges WHERE state=0 AND (a_uuid=?1 OR b_uuid=?1)"+
            (includeRetired?std::string{}:" AND "+activeEdgeSql())+" ORDER BY uuid",key);
        return readEdgeRows(db,rows);
    }
    EdgeSelection selection;selection.endpointUuids=std::vector{objectUuid};selection.includeRetired=includeRetired;
    return selectEdges(txn,selection);
}

std::vector<EdgeRecord> ObjectStore::edgesByKind(const Txn& txn, const std::string& edgeKind, bool includeRetired) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    if(querySources(db).size()==1){
        Statement rows(db,"SELECT "+projectedEdgeColumns(db)+" FROM "+edgeReadSource(db)+" AS edges WHERE state=0 AND kind=?"+
            (includeRetired?std::string{}:" AND "+activeEdgeSql())+" ORDER BY uuid",edgeKindCode(edgeKind));
        return readEdgeRows(db,rows);
    }
    EdgeSelection selection;selection.kind=edgeKind;selection.includeRetired=includeRetired;
    return selectEdges(txn,selection);
}

std::unique_ptr<Statement> detail::edgeSelectionQuery(sqlite3* db, const EdgeSelection& selection,
    const std::string& columns, const std::optional<std::vector<int64_t>>& endpointIds)
{
    validateEdgeSelection(selection);
    const bool candidates = endpointIds || selection.endpointUuids || selection.uuids || selection.tag;
    if(endpointIds||selection.endpointUuids)requireAdjacencyIndex(db);
    std::string sql = "SELECT " + columns + " FROM "+edgeReadSource(db)+" AS edges WHERE state=0";
    if (!selection.includeRetired) sql += " AND " + activeEdgeSql();
    sql += fiberAdjacencySql(selection.adjacent);
    std::vector<std::variant<std::string,std::vector<uint8_t>,int64_t>> parameters;
    std::string prefix;
    auto parameter = [&](const auto& value) {
        parameters.push_back(value);
        return "?" + std::to_string(parameters.size());
    };
    if (selection.kind) sql += " AND kind=" + parameter(edgeKindCode(*selection.kind));
    if (!selection.attachments) sql += " AND kind<>" + parameter(edgeKindCode(kAttachmentEdgeKind));
    if (selection.approvalStates.size() == 1)
        sql += " AND approval_state=" + parameter(edgeApprovalCode(selection.approvalStates.front()));
    else if (!selection.approvalStates.empty()){
        nlohmann::json codes=nlohmann::json::array();for(const auto& state:selection.approvalStates)codes.push_back(edgeApprovalCode(state));
        sql += " AND approval_state IN (SELECT value FROM json_each(" + parameter(codes.dump()) + "))";
    }
    if (selection.tag)
        sql += " AND uuid IN (SELECT edge_uuid FROM edge_tags WHERE tag=" + parameter(*selection.tag) + ")";
    if (endpointIds || selection.endpointUuids) {
        std::string members;
        if(endpointIds) {
            members=" IN (SELECT uuid FROM objects WHERE state<>2 AND id IN (SELECT value FROM json_each("+
                parameter(nlohmann::json(*endpointIds).dump())+")))";
        } else {
            std::vector<uint8_t> keys;keys.reserve(selection.endpointUuids->size()*16);
            for(const auto& uuid:*selection.endpointUuids){const auto bytes=parseUuid(uuid);keys.insert(keys.end(),bytes.begin(),bytes.end());}
            parameters.emplace_back(std::move(keys));
            const auto blob="?"+std::to_string(parameters.size());
            prefix="WITH RECURSIVE requested_key_offsets(pos) AS (SELECT 1 WHERE length("+blob+
                ")>0 UNION ALL SELECT pos+16 FROM requested_key_offsets WHERE pos+16<=length("+blob+
                ")), requested_keys(uuid) AS MATERIALIZED (SELECT substr("+blob+",pos,16) FROM requested_key_offsets) ";
            members=" IN (SELECT uuid FROM requested_keys)";
        }
        const std::string kind = selection.kind ? " AND kind=?1" : "";
        // Materialize endpoint-index candidates, then probe by primary key.
        // Otherwise SQLite may scan an entire kind's UUID index to avoid sorting.
        sql += " AND uuid IN (SELECT uuid FROM edges INDEXED BY edges_a WHERE a_uuid" + members + kind;
        if (selection.bothEndpoints) sql += " AND b_uuid" + members;
        else sql += " UNION SELECT uuid FROM edges INDEXED BY edges_b WHERE b_uuid" + members + kind;
        sql += ")";
    }
    if (selection.uuids) {
        if(selection.uuids->empty())sql+=" AND 0";
        else{sql+=" AND uuid IN (";bool first=true;for(const auto& uuid:*selection.uuids){
            if(!first)sql+=',';first=false;const auto key=parseUuid(uuid);
            sql+=parameter(std::vector<uint8_t>(key.begin(),key.end()));}sql+=')';}
    }
    const auto cursorParameter = parameters.size() + 1;
    if (selection.afterUuid) sql += " AND uuid>?" + std::to_string(cursorParameter);
    sql += " ORDER BY uuid LIMIT ?" + std::to_string(cursorParameter + (selection.afterUuid ? 1 : 0));
    auto rows = std::make_unique<Statement>(db, prefix+sql);
    int next = 1;
    for (const auto& value : parameters) std::visit([&](const auto& bound){rows->bind(next++,bound);},value);
    if (selection.afterUuid) rows->bind(next++, parseUuid(*selection.afterUuid));
    rows->bind(next, selection.limit.value_or(-1));
    return rows;
}

std::vector<EdgeRecord> ObjectStore::selectEdges(const Txn& txn, const EdgeSelection& selection,
    const std::optional<std::vector<std::string>>& projection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const ObjectFields fields = projection ? ObjectFields(std::in_place, projection->begin(), projection->end()) : std::nullopt;
    const auto columns = projectedEdgeColumns(db,fields);
    if(querySources(db).size()>1){
        validateEdgeSelection(selection);
        if(selection.applicableOnly)requirePublishedGraphRead(txn);
        auto physical=selection;physical.limit.reset();physical.applicableOnly=false;physical.includeRetired=true;
        return selectVisibleRecords(db,[&](auto* source){
            auto rows=edgeSelectionQuery(source,physical,projectedEdgeColumns(source,fields));
            auto selected=readEdgeRows(source,*rows,fields);
            if(!txn.write_)attachRelationshipVersions(source,selected);
            return selected;
            },[&](const auto& edge,auto* source){
                if(edgeSource(db,edge.uuid)!=source)return false;
                const auto a=resolveObject(db,edge.aUuid),b=resolveObject(db,edge.bUuid);
                return a&&b&&(selection.includeRetired||(!a->record.retired&&!b->record.retired))&&
                    (!selection.applicableOnly||currentRelationshipApplicable(*this,txn,edge.uuid));
            },[](const auto& a,const auto& b){return a.uuid<b.uuid;},0,selection.limit);
    }
    if (selection.applicableOnly) {
        validateEdgeSelection(selection);
        requirePublishedGraphRead(txn);
        // Apply validity before pagination; then hydrate only accepted records.
        auto candidates = selection; candidates.limit.reset(); candidates.applicableOnly = false;
        auto rows = edgeSelectionQuery(db, candidates, "uuid");
        std::vector<std::string> accepted;
        while ((!selection.limit || accepted.size() < static_cast<size_t>(*selection.limit)) && rows->row()) {
            auto uuid = formatUuid(rows->blob16(0));
            if (currentRelationshipApplicable(*this, txn, uuid)) accepted.push_back(std::move(uuid));
        }
        EdgeSelection selected; selected.uuids = std::move(accepted); selected.includeRetired = true;
        return selectEdges(txn, selected, projection);
    }
    auto rows = edgeSelectionQuery(db, selection, columns);
    auto records = readEdgeRows(db, *rows, fields);
    if (!txn.write_) attachRelationshipVersions(db, records);
    return records;
}

EdgeRecord attachmentEdge(const std::string& dataUuid, const std::string& ownerUuid, const std::string& role)
{
    EdgeRecord edge;
    edge.kind = kAttachmentEdgeKind;
    edge.aUuid = dataUuid;
    edge.bUuid = ownerUuid;
    edge.aAnchor.wholeGeometry = true;
    edge.bAnchor.wholeGeometry = true;
    edge.meta = nlohmann::json{{"role", role}}.dump();
    return edge;
}

std::string ObjectStore::attach(WriteTxn& txn, const std::string& dataUuid, const std::string& ownerUuid, const std::string& role)
{
    return addEdge(txn, attachmentEdge(dataUuid, ownerUuid, role));
}

std::vector<EdgeRecord> ObjectStore::attachments(const Txn& txn, const std::string& ownerUuid,
                                                 const std::optional<std::string>& role) const
{
    return vc::objects::attachments(LocalStoreReader(*this, txn), ownerUuid, role);
}

} // namespace vc::objects
