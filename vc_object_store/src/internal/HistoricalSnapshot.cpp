#include "internal/HistoricalSnapshot.hpp"
#include "internal/RecordSql.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordJson.hpp"
#include "internal/History.hpp"
#include "internal/Versions.hpp"
#include "geometry/GeometryReader.hpp"

namespace vc::objects::detail {
namespace {
using Json=nlohmann::json;
struct Projection {
    sqlite3* db;
    std::vector<std::string> tables,views;
    bool spatial=false;
    explicit Projection(sqlite3* value):db(value),spatial(spatialEnabled(value)){}
    ~Projection(){
        // Readers are physically readonly; only their private TEMP schema is
        // writable. Rollback may already have removed these objects.
        sqlite3_exec(db,"PRAGMA query_only=OFF",nullptr,nullptr,nullptr);
        for(auto it=views.rbegin();it!=views.rend();++it)
            sqlite3_exec(db,("DROP VIEW IF EXISTS temp.\""+*it+"\"").c_str(),nullptr,nullptr,nullptr);
        for(auto it=tables.rbegin();it!=tables.rend();++it)
            sqlite3_exec(db,("DROP TABLE IF EXISTS temp.\""+*it+"\"").c_str(),nullptr,nullptr,nullptr);
        sqlite3_exec(db,"PRAGMA query_only=ON",nullptr,nullptr,nullptr);
        if(auto pool=activeStatementPools.find(db);pool!=activeStatementPools.end()){
            pool->second->spatialEnabled=spatial;pool->second->selectedSourceCommit.reset();
            pool->second->geometryCache.reset();pool->second->spatialCache.reset();
            pool->second->components.clear();pool->second->componentsVersion.reset();pool->second->transformedLengths.clear();
        }
    }
    void clone(const std::string& name){
        // Preserve physical column order/types/defaults and keys, not source
        // mutation triggers or foreign keys into its separate main schema.
        Statement columns(db,"SELECT name,type,dflt_value,pk FROM pragma_table_xinfo(?, 'main') ORDER BY cid",name);
        std::string sql="CREATE TEMP TABLE \""+name+"\"(";std::map<int,std::string> keys;size_t count=0;
        while(columns.row()){
            const auto column=columns.text(0);if(count++)sql+=',';
            sql+='\"'+column+"\" "+columns.text(1);
            if(!columns.isNull(2))sql+=" DEFAULT "+columns.text(2);
            if(columns.i64(3))keys.emplace(int(columns.i64(3)),column);
        }
        if(!count)throw std::runtime_error("historical snapshot: missing table "+name);
        if(!keys.empty()){sql+=",PRIMARY KEY(";bool first=true;for(const auto& [order,key]:keys){if(!first)sql+=',';first=false;sql+='\"'+key+'\"';}sql+=')';}
        sql+=')';exec(db,sql.c_str());tables.push_back(name);
        if(name=="objects")exec(db,"CREATE UNIQUE INDEX temp._vc_pinned_object_uuid ON objects(uuid)");
    }
    void view(const std::string& name,const std::string& select){
        exec(db,("CREATE TEMP VIEW \""+name+"\" AS "+select).c_str());views.push_back(name);
    }
    void overlay(const std::string& name,const std::string& predicate){
        const auto saved="_vc_pinned_"+name;
        exec(db,("ALTER TABLE temp.\""+name+"\" RENAME TO \""+saved+"\"").c_str());
        *std::find(tables.begin(),tables.end(),name)=saved;
        view(name,"SELECT * FROM main.\""+name+"\" WHERE "+predicate+" UNION ALL SELECT * FROM temp.\""+saved+"\"");
    }
};
void common(sqlite3* db,const char* table,const RecordCommon& r,int64_t sequence){
    Statement row(db,std::string("UPDATE ")+table+" SET commit_sequence=?,version=?,version_generation=?,"
        "created_by=?,created_at=?,created_host=?,changed_by=?,changed_at=?,changed_host=? WHERE uuid=?",
        sequence,r.version.value_or(1),parseUuid(r.versionGeneration),r.createdBy,r.createdAt,r.createdHost,
        r.changedBy,r.changedAt,r.changedHost,parseUuid(r.uuid));
    row.stepDone();
}
}
void requireRetainedPublication(sqlite3* db,const StoreStatus& pin){
    if(pin.commitSequence<0)throw std::invalid_argument("historical snapshot: negative source publication");
    const auto owner=util::canonicalUuid(pin.storeUuid),lineage=util::canonicalUuid(pin.generation);
    Statement head(db,"SELECT s.uuid,m.generation,m.commit_sequence,m.history_enabled FROM main.store_identity s,main.store_manifest m");
    if(!head.row()||head.uuid(0)!=owner||head.i64(2)<pin.commitSequence)
        throw StoreBusy("historical snapshot: source identity/publication is unavailable");
    if(head.i64(2)==pin.commitSequence&&head.uuid(1)==lineage)return;
    if(!head.i64(3))throw StoreBusy("historical snapshot: required source history was removed");
    Statement generation(db,"SELECT after_sequence,reason FROM main.history_generations WHERE generation=?",parseUuid(pin.generation));
    if(!generation.row()||generation.i64(0)>pin.commitSequence)
        throw StoreBusy("historical snapshot: required source lineage was removed");
    if(generation.i64(0)==0&&generation.text(1)=="history enabled"){
        // Older writers did not persist the retention boundary when enabling
        // history. Never mistake re-enabled current-state baselines for the
        // discarded earlier timeline.
        Statement first(db,"SELECT min(seq) FROM main.store_commits");first.row();
        if(first.isNull(0)||pin.commitSequence<first.i64(0)-1)
            throw StoreBusy("historical snapshot: source history coverage is unavailable");
    }
}
std::shared_ptr<void> historicalSnapshot(sqlite3* db,const StoreStatus& pin){
    requireRetainedPublication(db,pin);
    Statement head(db,"SELECT commit_sequence FROM main.store_manifest");head.row();
    if(head.i64(0)==pin.commitSequence)return {};
    if(sqlite3_db_readonly(db,"main")!=1)throw std::logic_error("historical dependencies require a physically readonly connection");
    struct Object {int64_t id;ObjectRevision revision;std::string manifest,hash;};
    struct Edge {EdgeRecord record;int64_t sequence;bool deleted;std::optional<std::string> revision;};
    std::vector<Object> objects;std::vector<Edge> edges;
    std::vector<std::pair<std::string,int64_t>> changed;
    Statement changedObjects(db,"SELECT DISTINCT o.uuid,o.id FROM main.store_changes c JOIN main.objects o ON o.uuid=c.record_uuid "
        "WHERE c.seq>? AND c.record_kind='object' ORDER BY o.uuid",pin.commitSequence);
    while(changedObjects.row()){
        const auto key=changedObjects.uuid(0);changed.emplace_back(key,changedObjects.i64(1));
        Statement version(db,"SELECT revision_uuid FROM main.object_versions WHERE object_uuid=? AND (commit_seq<=? OR commit_seq IS NULL) ORDER BY version DESC LIMIT 1",
            parseUuid(key),pin.commitSequence);
        if(!version.row()){
            Statement baseline(db,"SELECT min(version),sum(commit_seq IS NULL) FROM main.object_versions WHERE object_uuid=?",parseUuid(key));baseline.row();
            if(baseline.isNull(0)||baseline.i64(0)!=1||baseline.i64(1))throw StoreBusy("historical snapshot: object baseline unavailable");
            continue; // Created after this source publication.
        }
        const auto resolved=resolveObjectRevision(db,version.uuid(0));
        if(!resolved||!resolved->revision.version)
            throw StoreBusy("historical snapshot: object version unavailable");
        Object old{changedObjects.i64(1),resolved->revision,{},{}};
        if(!old.revision.commitSequence)old.revision.commitSequence=old.revision.record.commitSequence;
        if(*old.revision.commitSequence>pin.commitSequence){
            if(old.revision.version==1)continue;
            throw StoreBusy("historical snapshot: required pre-baseline metadata was not retained");
        }
        old.revision.record.version=old.revision.version;old.revision.record.versionGeneration=old.revision.generation;
        if(old.revision.geometryRevision&&!old.revision.deleted){
            GeometryReader geometry(db,*old.revision.geometryRevision,false,false);
            old.manifest=geometry.manifest().dump();old.hash=geometry.contentHash();
        }
        objects.push_back(std::move(old));
    }
    std::vector<std::string> changedEdges;
    Statement changedRelationships(db,"SELECT DISTINCT e.uuid FROM main.store_changes c JOIN main.edges e ON e.uuid=c.record_uuid "
        "WHERE c.seq>? AND c.record_kind='relationship' ORDER BY e.uuid",pin.commitSequence);
    while(changedRelationships.row()){
        const auto key=changedRelationships.uuid(0);changedEdges.push_back(key);
        Statement version(db,"SELECT version,generation,commit_seq,revision_uuid,deleted FROM main.relationship_versions "
            "WHERE edge_uuid=? AND (commit_seq<=? OR commit_seq IS NULL) ORDER BY version DESC LIMIT 1",parseUuid(key),pin.commitSequence);
        if(!version.row())continue;
        Edge old;old.sequence=version.i64(2);old.deleted=version.i64(4)!=0;
        if(old.deleted){
            Statement current(db,"SELECT "+projectedEdgeColumns(db)+" FROM "+edgeReadSource(db)+" WHERE uuid=?",parseUuid(key));
            if(!current.row())throw StoreBusy("historical snapshot: relationship tombstone unavailable");
            old.record=readEdgeRow(db,current);
        }else{
            old.revision=version.uuid(3);
            Statement archived(db,"SELECT record FROM main.relationship_revisions WHERE uuid=?",parseUuid(*old.revision));
            if(!archived.row())throw StoreBusy("historical snapshot: relationship version unavailable");
            old.record=edgeFromJson(Json::parse(archived.text(0)));
        }
        old.record.version=version.i64(0);old.record.versionGeneration=version.uuid(1);
        if(version.isNull(2))old.sequence=old.record.commitSequence;
        if(old.sequence>pin.commitSequence){
            if(*old.record.version==1)continue;
            throw StoreBusy("historical snapshot: required pre-baseline relationship was not retained");
        }
        edges.push_back(std::move(old));
    }
    auto projection=std::make_shared<Projection>(db);
    exec(db,"PRAGMA query_only=OFF");
    for(const auto* table:{"objects","fibers","surfaces","tracks","object_tags","edges","edge_tags",
                          "object_heads","object_head_records","object_relationship_heads","store_manifest"})projection->clone(table);
    exec(db,"CREATE TEMP TABLE _vc_changed_objects(uuid BLOB PRIMARY KEY,id INTEGER UNIQUE);CREATE TEMP TABLE _vc_changed_edges(uuid BLOB PRIMARY KEY)");
    projection->tables.push_back("_vc_changed_objects");projection->tables.push_back("_vc_changed_edges");
    for(const auto& [uuid,id]:changed)execute(db,"INSERT INTO temp._vc_changed_objects VALUES(?,?)",parseUuid(uuid),id);
    for(const auto& uuid:changedEdges)execute(db,"INSERT INTO temp._vc_changed_edges VALUES(?)",parseUuid(uuid));
    for(const auto& old:objects){
        auto record=old.revision.record;record.commitSequence=*old.revision.commitSequence;
        if(old.revision.geometryRevision){record.storageKind="sqlite";record.storageRef=*old.revision.geometryRevision;record.dataChecksum=old.hash;}
        insertObjectRow(db,record,old.id,*old.revision.commitSequence,record.createdBy,record.createdHost,record.createdAt,true,
                        old.manifest.empty()?nullptr:&old.manifest);
        common(db,"objects",record,*old.revision.commitSequence);
        execute(db,"UPDATE objects SET state=?,lifecycle_epoch=? WHERE id=?",
            int64_t(old.revision.deleted?2:record.retired?1:0),record.lifecycleEpoch,old.id);
        insertObjectDetail(db,old.id,record);insertTags(db,"object_tags","object_id",old.id,record.tags);
        execute(db,"INSERT INTO object_heads VALUES(?,?)",parseUuid(record.uuid),parseUuid(old.revision.uuid));
        execute(db,"INSERT INTO object_head_records VALUES(?,?,?)",parseUuid(record.uuid),parseUuid(old.revision.uuid),objectToJson(record).dump());
    }
    exec(db,"UPDATE temp.objects SET coordinate_volume=json_extract(nullif(meta,''),'$.coordinates.volume_uuid'),"
            "coordinate_scale=json_extract(nullif(meta,''),'$.coordinates.scale_factor')");
    for(const auto& old:edges){
        insertEdgeRow(db,old.record,old.sequence,old.record.createdBy,old.record.createdHost,old.record.createdAt,true);
        if(old.deleted)execute(db,"UPDATE edges SET state=2 WHERE uuid=?",parseUuid(old.record.uuid));
        if(old.revision)for(const auto* endpoint:{&old.record.aUuid,&old.record.bUuid})
            execute(db,"INSERT OR IGNORE INTO object_relationship_heads VALUES(?,?,?)",parseUuid(*endpoint),parseUuid(old.record.uuid),parseUuid(*old.revision));
    }
    exec(db,"INSERT INTO temp.store_manifest SELECT * FROM main.store_manifest");
    execute(db,"UPDATE temp.store_manifest SET generation=?,commit_sequence=?,spatial_enabled=0",parseUuid(pin.generation),pin.commitSequence);
    projection->overlay("objects","uuid NOT IN (SELECT uuid FROM temp._vc_changed_objects)");
    for(const auto* table:{"fibers","surfaces","tracks","object_tags"})
        projection->overlay(table,"object_id NOT IN (SELECT id FROM temp._vc_changed_objects)");
    for(const auto* table:{"object_heads","object_head_records"})
        projection->overlay(table,"object_uuid NOT IN (SELECT uuid FROM temp._vc_changed_objects)");
    for(const auto* table:{"edges","edge_tags","object_relationship_heads"})
        projection->overlay(table,std::string(table)=="edges"?"uuid NOT IN (SELECT uuid FROM temp._vc_changed_edges)":"edge_uuid NOT IN (SELECT uuid FROM temp._vc_changed_edges)");
    const auto sequence=std::to_string(pin.commitSequence);
    projection->view("store_commits","SELECT * FROM main.store_commits WHERE seq<="+sequence);
    projection->view("store_changes","SELECT * FROM main.store_changes WHERE seq<="+sequence);
    projection->view("object_versions","SELECT * FROM main.object_versions WHERE commit_seq<="+sequence+" OR commit_seq IS NULL");
    projection->view("relationship_versions","SELECT * FROM main.relationship_versions WHERE commit_seq<="+sequence+" OR commit_seq IS NULL");
    projection->view("object_revisions","SELECT * FROM main.object_revisions WHERE uuid IN (SELECT revision_uuid FROM object_versions)");
    projection->view("relationship_revisions","SELECT * FROM main.relationship_revisions WHERE uuid IN (SELECT revision_uuid FROM relationship_versions)");
    projection->view("journal","SELECT * FROM main.journal WHERE commit_sequence<="+sequence);
    auto* pool=activeStatementPools.at(db);pool->spatialEnabled=false;pool->selectedSourceCommit=pin.commitSequence;
    pool->geometryCache.reset();pool->spatialCache.reset();
    pool->components.clear();pool->componentsVersion.reset();pool->transformedLengths.clear();
    exec(db,"PRAGMA query_only=ON");
    return projection;
}
}
