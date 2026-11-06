#include "TestSupport.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/LegacyCurrentSchema.hpp"
#include "internal/Digest.hpp"
#include <bit>
#include <iostream>
#include <set>
using namespace vc::objects;
using namespace vc::objects::detail;

// A schema-17 fixture with real typed geometry and retained history. Remove
// only the v2 provisioning additions; never keep a legacy runtime code path.
void schema17(const std::filesystem::path& path)
{
    std::unique_ptr<sqlite3,decltype(&sqlite3_close)> db(openConnection(path,SQLITE_OPEN_READWRITE,{},false),sqlite3_close);
    std::vector<std::pair<std::string,std::string>> programs;
    Statement rows(db.get(),"SELECT name,sql,tbl_name FROM sqlite_schema WHERE type='trigger'");
    std::vector<std::string> names;
    const std::set<std::string> v2Tables{"table_key_columns","table_directory","database_identity","key_allocators",
        "store_required_features","store_write_state","store_manifest"};
    while(rows.row()) {
        names.push_back(rows.text(0));
        if(!v2Tables.contains(rows.text(2)))programs.emplace_back(rows.text(0),rows.text(1));
    }
    for(const auto& name:names)exec(db.get(),("DROP TRIGGER \""+name+"\"").c_str());
    std::erase_if(programs,[](const auto& program){return program.second.find("store_write_state")!=std::string::npos;});
    exec(db.get(),"CREATE TABLE geometry_heads(object_uuid BLOB PRIMARY KEY REFERENCES objects(uuid) ON DELETE RESTRICT,"
        "revision_uuid BLOB NOT NULL,FOREIGN KEY(object_uuid,revision_uuid) REFERENCES geometry_revisions(object_uuid,uuid));");
    {
        Statement objects(db.get(),"SELECT uuid,storage_ref FROM objects WHERE geometry_manifest IS NOT NULL");
        while(objects.row())execute(db.get(),"INSERT INTO geometry_heads VALUES(?,?)",objects.blob16(0),parseUuid(objects.text(1)));
    }
    exec(db.get(), "DROP TABLE table_key_columns; DROP TABLE table_directory; DROP TABLE database_identity; DROP TABLE key_allocators;"
        "DROP TABLE store_required_features; DROP TABLE store_write_state; DROP TABLE store_manifest;"
        "ALTER TABLE object_revisions DROP COLUMN generation; ALTER TABLE relationship_revisions DROP COLUMN generation;"
        "ALTER TABLE objects DROP COLUMN geometry_manifest;"
        "ALTER TABLE objects DROP COLUMN deleted;"
        "DROP INDEX IF EXISTS edges_unique;"
        "PRAGMA user_version=17;");
    // Real v1 stored current pins and a duplicate dependency index. New native
    // stores deliberately do not; reconstitute their legacy representation.
    exec(db.get(),"CREATE TABLE edge_geometry_bindings(edge_id INTEGER PRIMARY KEY REFERENCES edges(id) ON DELETE CASCADE,"
        "a_revision BLOB REFERENCES geometry_revisions(uuid),b_revision BLOB REFERENCES geometry_revisions(uuid));"
        "INSERT INTO edge_geometry_bindings SELECT e.id,ga.revision_uuid,gb.revision_uuid FROM edges e "
        "JOIN objects a ON a.uuid=e.a_uuid JOIN objects b ON b.uuid=e.b_uuid "
        "LEFT JOIN geometry_heads ga ON ga.object_uuid=a.uuid LEFT JOIN geometry_heads gb ON gb.object_uuid=b.uuid;"
        "CREATE TABLE edge_lifecycle_bindings(edge_id INTEGER PRIMARY KEY REFERENCES edges(id) ON DELETE CASCADE,a_epoch TEXT NOT NULL,b_epoch TEXT NOT NULL);"
        "INSERT INTO edge_lifecycle_bindings SELECT e.id,coalesce(a.epoch,''),coalesce(b.epoch,'') FROM edges e "
        "LEFT JOIN object_lifecycle a ON a.object_id=(SELECT id FROM objects WHERE uuid=e.a_uuid) "
        "LEFT JOIN object_lifecycle b ON b.object_id=(SELECT id FROM objects WHERE uuid=e.b_uuid);"
        "CREATE TABLE geometry_edge_dependencies(edge_id INTEGER NOT NULL REFERENCES edges(id) ON DELETE CASCADE,"
        "object_id INTEGER NOT NULL REFERENCES objects(id),array_name TEXT NOT NULL,point_id INTEGER NOT NULL,"
        "PRIMARY KEY(object_id,array_name,point_id,edge_id)) WITHOUT ROWID;"
        "CREATE INDEX geometry_dependencies_edge ON geometry_edge_dependencies(edge_id);");
    // Real v1 chunks are NOT NULL. Keep native triggers while reconstructing
    // that legacy table, before adding the extension dependency regression.
    exec(db.get(),"PRAGMA foreign_keys=OFF;");
    for(const auto& [table,digest]:std::map<std::string,std::string>{
        {"geometry_chunks","hash"},{"geometry_revisions","content_hash"},
        {"geometry_revision_chunks","hash"},{"geometry_deltas","manifest_hash"},
        {"geometry_current_chunks","hash"},{"edges",""},{"relationship_revisions",""}}) {
        Statement schema(db.get(),"SELECT sql FROM sqlite_schema WHERE name=?",table);require(schema.row(),"native digest schema");
        auto definition=schema.text(0);definition=definition.substr(definition.find('('));schema.reset();
        const auto replace=[&](const std::string& before,const std::string& after) {
            if(const auto p=definition.find(before);p!=std::string::npos)definition.replace(p,before.size(),after);
        };
        if(!digest.empty()) {
            replace(digest+" BLOB",digest+" TEXT");
            replace("CHECK(typeof("+digest+")='blob' AND length("+digest+")=32)","CHECK(length("+digest+")=64)");
        }
        if(table=="edges"){
            replace("deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0,1)),","");
            for(const auto* side:{"a","b"}){
                const auto from=std::string(side)+"_uuid",to=std::string(side)+"_id";
                replace(from+" BLOB NOT NULL CHECK(length("+from+")=16)",to+" INTEGER NOT NULL REFERENCES objects(id) ON DELETE RESTRICT");
                for(size_t p=0;(p=definition.find(from,p))!=std::string::npos;p+=to.size())definition.replace(p,from.size(),to);
            }
        }
        if(table=="relationship_revisions")for(const auto* endpoint:{"a_revision","b_revision"})
            replace(std::string(endpoint)+" TEXT NOT NULL,",std::string(endpoint)+" TEXT NOT NULL REFERENCES object_revisions(uuid) DEFERRABLE INITIALLY DEFERRED,");
        if(table=="geometry_chunks")replace("data BLOB CHECK(data IS NULL OR (length(data)>0 AND length(data)<=1048576))",
            "data BLOB NOT NULL CHECK(length(data)>0 AND length(data)<=1048576)");
        std::vector<std::string> indexes;
        Statement index(db.get(),"SELECT sql FROM sqlite_schema WHERE tbl_name=? AND type='index' AND sql IS NOT NULL",table);
        while(index.row()){
            auto sql=index.text(0);
            if(table=="edges")for(const auto* side:{"a","b"}){
                const auto from=std::string(side)+"_uuid",to=std::string(side)+"_id";
                for(size_t p=0;(p=sql.find(from,p))!=std::string::npos;p+=to.size())sql.replace(p,from.size(),to);
            }
            indexes.push_back(std::move(sql));
        }
        std::string projection="rowid",columns="rowid";
        Statement fields(db.get(),"SELECT name FROM pragma_table_xinfo(?) ORDER BY cid",table);
        while(fields.row()) {
            const auto name=fields.text(0);
            if(table=="edges"&&name=="deleted")continue;
            if(table=="edges"&&(name=="a_uuid"||name=="b_uuid")){
                columns+=name=="a_uuid"?",a_id":",b_id";
                projection+=",(SELECT id FROM objects WHERE uuid=edges."+name+")";
                continue;
            }
            columns+=",\""+name+"\"";
            projection+=name==digest?",lower(hex(\""+name+"\"))":",\""+name+"\"";
        }
        exec(db.get(),("CREATE TABLE legacy_digest"+definition+"; INSERT INTO legacy_digest("+columns+") SELECT "+projection+
            " FROM \""+table+"\"; DROP TABLE \""+table+"\"; ALTER TABLE legacy_digest RENAME TO \""+table+"\";").c_str());
        for(const auto& sql:indexes)exec(db.get(),sql.c_str());
    }
    const auto endpointTriggers=legacyEndpointHistoryTriggers();
    for(const auto& [name,definition]:programs){
        const auto legacy=endpointTriggers.find(name);
        exec(db.get(),(legacy==endpointTriggers.end()?definition:legacy->second).c_str());
    }
    for(const auto& [name,sql]:legacyCurrentTriggers())exec(db.get(),sql.c_str());
    exec(db.get(),"CREATE UNIQUE INDEX edges_unique ON edges(kind,a_id,b_id,a_dependencies,b_dependencies);");
    exec(db.get(),"PRAGMA foreign_keys=ON;"
        "CREATE VIEW extension_chunk_sizes AS SELECT hash,length(data) bytes FROM geometry_chunks;"
        "CREATE VIEW extension_chunk_total AS SELECT sum(bytes) bytes FROM extension_chunk_sizes;"
        "CREATE INDEX extension_chunk_length ON geometry_chunks(length(data));"
        "CREATE TABLE extension_marks(id INTEGER PRIMARY KEY,bytes INTEGER);"
        "CREATE TRIGGER extension_mark AFTER INSERT ON extension_marks BEGIN "
        "UPDATE extension_marks SET bytes=(SELECT bytes FROM extension_chunk_total) WHERE id=NEW.id; END;"
        "INSERT INTO extension_marks(id) VALUES(1);");
}

void rejectsStaleLegacy(const std::filesystem::path& path,const std::string& corrupt)
{
    {
        auto db=ObjectStore::create(path,OpenOptions{.author="test"});auto txn=db.write();
        ObjectRecord a;a.kind="fiber";a.meta=R"({"label":"original"})";
        const auto left=db.upsertObject(txn,a);a.uuid.clear();const auto right=db.upsertObject(txn,a);
        for(const auto& id:{left,right})db.putGeometry(txn,id,vc::geometry::encode(vc::geometry::Fiber{{},{{1,2,3},{4,5,6}}, {}}));
        EdgeRecord e;e.kind="test";e.aUuid=left;e.bUuid=right;e.aAnchor.wholeGeometry=true;e.bAnchor.wholeGeometry=true;
        e.aAnchor.metadataKeys={"label"};e.bAnchor.metadataKeys={"label"};db.addEdge(txn,e);txn.commit();
    }
    schema17(path);
    {
        std::unique_ptr<sqlite3,decltype(&sqlite3_close)> sql(openConnection(path,SQLITE_OPEN_READWRITE,{},false),sqlite3_close);
        exec(sql.get(),corrupt.c_str());
    }
    mustFail([&]{convertV1ToV2(path);});
    std::unique_ptr<sqlite3,decltype(&sqlite3_close)> sql(openConnection(path,SQLITE_OPEN_READONLY,{},true),sqlite3_close);
    require(pragmaInt(sql.get(),"user_version")==17,"rejected stale legacy conversion leaves original schema intact");
    Statement edges(sql.get(),"SELECT count(*) FROM edges");require(edges.row()&&edges.i64(0)==1,"rejected conversion retains original relation");
}

void digestHistory(const std::filesystem::path& path)
{
    std::string object,first,last;std::vector<uint8_t> expected;
    {
        OpenOptions options;options.author="test";options.maxGeometryDeltaBytes=1024;
        auto store=ObjectStore::create(path,options);
        vc::geometry::Fiber fiber;for(int i=0;i<10000;++i)fiber.linePoints.push_back({double(i),0,0});
        {auto txn=store.write();ObjectRecord record;record.kind="fiber";object=store.upsertObject(txn,record);
            first=store.putGeometry(txn,object,vc::geometry::encode(fiber));txn.commit();}
        {auto txn=store.write();GeometryArrayEdit edit{"line_points",0,std::vector<uint8_t>(8)};
            const auto bits=std::bit_cast<uint64_t>(3.5);for(int i=0;i<8;++i)edit.bytes[i]=uint8_t(bits>>(8*i));
            last=store.editGeometry(txn,object,{edit},first);txn.commit();}
        auto txn=store.read();expected=store.readGeometryArray(txn,last,"line_points");
        Statement deltas(ObjectStoreImplAccess::conn(txn),"SELECT count(*) FROM geometry_deltas");
        require(deltas.row()&&deltas.i64(0)>0,"legacy fixture includes sparse historical delta");
    }
    schema17(path);convertV1ToV2(path);
    {
        auto store=ObjectStore::open(path,OpenOptions{.readOnly=true});auto txn=store.read();
        require(store.readGeometryArray(txn,last,"line_points")==expected,"binary digest conversion preserves delta geometry");
        require(store.readGeometryArray(txn,first,"line_points")!=expected,"binary digest conversion preserves earlier full revision");
        for(const auto& [table,column]:std::map<std::string,std::string>{{"geometry_chunks","hash"},
            {"geometry_revisions","content_hash"},{"geometry_revision_chunks","hash"},
            {"geometry_deltas","manifest_hash"},{"geometry_current_chunks","hash"}}) {
            Statement invalid(ObjectStoreImplAccess::conn(txn),"SELECT count(*) FROM "+table+" WHERE typeof("+column+")!='blob' OR length("+column+")!=32");
            require(invalid.row()&&invalid.i64(0)==0,"converted SQL digests are canonical binary32");
        }
    }
    for(const auto mode:{HistoryMode::Off,HistoryMode::On}) {
        changeHistoryMode(path,mode);auto store=ObjectStore::open(path,OpenOptions{.readOnly=true});auto txn=store.read();
        const auto geometry=store.currentGeometry(txn,object);
        require(store.readGeometryArray(txn,geometry->uuid,"line_points")==expected,"policy transition preserves converted delta data");
        Statement checksum(ObjectStoreImplAccess::conn(txn),"SELECT typeof(data_checksum),length(data_checksum) FROM objects WHERE uuid=?",parseUuid(object));
        require(checksum.row()&&checksum.text(0)=="text"&&checksum.i64(1)==64,"generic object checksum remains hexadecimal text");
    }
}

int main() try
{
    TemporaryDirectory dir;
    digestHistory(dir.path/"digest-history");
    rejectsStaleLegacy(dir.path/"bad-geometry","UPDATE edge_geometry_bindings SET a_revision=NULL;");
    rejectsStaleLegacy(dir.path/"bad-lifecycle","UPDATE edge_lifecycle_bindings SET a_epoch='older';");
    rejectsStaleLegacy(dir.path/"bad-metadata","UPDATE objects SET meta=json_set(meta,'$.label','changed');");
    rejectsStaleLegacy(dir.path/"bad-retired","INSERT INTO object_lifecycle SELECT a_id,1,'legacy-retired' FROM edges;");
    rejectsStaleLegacy(dir.path/"custom-column","ALTER TABLE edge_geometry_bindings ADD COLUMN custom TEXT DEFAULT 'preserve-me';");
    rejectsStaleLegacy(dir.path/"custom-trigger","CREATE TRIGGER custom_binding_hook AFTER UPDATE ON edge_lifecycle_bindings BEGIN SELECT 1;END;");
    rejectsStaleLegacy(dir.path/"custom-view","CREATE VIEW custom_binding_view AS SELECT * FROM edge_geometry_bindings;");
    rejectsStaleLegacy(dir.path/"digest-extension","CREATE TABLE custom_digest(hash TEXT REFERENCES geometry_chunks(hash));");
    rejectsStaleLegacy(dir.path/"altered-native-trigger","DROP TRIGGER edge_geometry_bindings_history_UPDATE;"
        "CREATE TRIGGER edge_geometry_bindings_history_UPDATE AFTER UPDATE ON edge_geometry_bindings BEGIN SELECT 1;END;");
    rejectsStaleLegacy(dir.path/"altered-dependency-trigger","DROP TRIGGER geometry_dependencies_insert;"
        "CREATE TRIGGER geometry_dependencies_insert AFTER INSERT ON edges BEGIN SELECT 1;END;");
    rejectsStaleLegacy(dir.path/"altered-endpoint-history","DROP TRIGGER edges_history_UPDATE;"
        "CREATE TRIGGER edges_history_UPDATE AFTER UPDATE ON edges BEGIN SELECT 1;END;");
    rejectsStaleLegacy(dir.path/"endpoint-extension","CREATE VIEW custom_endpoint_ids AS SELECT a_id,b_id FROM edges;");
    rejectsStaleLegacy(dir.path/"altered-endpoint-index","DROP INDEX edges_a; CREATE INDEX edges_a ON edges(a_id,generation);");
    const auto path = dir.path / "legacy.vcobj";
    std::string storeId, object, geometry, exported;
    {
        auto db = ObjectStore::create(path,OpenOptions{.author="test"});
        auto w = db.write(); ObjectRecord r; r.kind="fiber"; r.fiber.emplace();
        r.meta=R"({"label":"unchanged"})";
        object=db.upsertObject(w,r);
        geometry=db.putGeometry(w,object,vc::geometry::encode(vc::geometry::Fiber{{},{{1,2,3},{4,5,6}}, {}}));
        ObjectRecord neighbor;neighbor.kind="note";const auto other=db.upsertObject(w,neighbor);
        EdgeRecord relation;relation.kind="test";relation.aUuid=object;relation.bUuid=other;
        relation.aAnchor.wholeGeometry=true;relation.bAnchor.wholeGeometry=true;relation.aAnchor.metadataKeys={"label"};
        db.addEdge(w,relation);
        w.commit();
        auto second=db.write(); db.setMeta(second,object,"retained","true"); second.commit();
        auto read=db.read(); storeId=db.uuid(read); exported=db.exportJson(read);
    }
    schema17(path);
    mustFail([&] { ObjectStore::open(path,OpenOptions{.readOnly=true}); });
    {
        std::unique_ptr<sqlite3,decltype(&sqlite3_close)> reader(openConnection(path,SQLITE_OPEN_READONLY,{},true),sqlite3_close);
        exec(reader.get(),"BEGIN; SELECT count(*) FROM objects;");
        mustFail([&] { convertV1ToV2(path); });
        require(pragmaInt(reader.get(),"user_version")==17,"busy conversion leaves original intact");
    }
    auto report=convertV1ToV2(path);
    require(report.tables>20 && report.rows>10 && report.bytes>0,"conversion validates all tables");
    {
        auto db=ObjectStore::open(path,OpenOptions{.author="test"}); auto read=db.read();
        require(db.uuid(read)==storeId,"conversion preserves logical store identity");
        require(db.exportJson(read)==exported,"conversion preserves complete current metadata");
        require(db.objectHistory(read,object).size()==2,"conversion preserves complete object history");
        require(db.currentGeometry(read,object)->uuid==geometry,"conversion preserves geometry identity");
        require(db.readGeometryArray(read,geometry,"line_points").size()==6*sizeof(double),"converted payload readable");
        auto* sql=ObjectStoreImplAccess::conn(read);
        {Statement removed(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN ('edge_geometry_bindings','edge_lifecycle_bindings','geometry_edge_dependencies')");
            require(removed.row()&&removed.i64(0)==0,"conversion removes duplicate current-state binding materialization");}
        {Statement endpoints(sql,"SELECT length(a_uuid),length(b_uuid),deleted FROM edges");
            require(endpoints.row()&&endpoints.i64(0)==16&&endpoints.i64(1)==16&&endpoints.i64(2)==0,"conversion stores original endpoint UUIDs directly");}
        {Statement triggers(sql,"SELECT sql FROM sqlite_schema WHERE type='trigger' AND (name GLOB 'edges_history_*' OR name GLOB 'edge_tags_history_*')");
            int count=0;while(triggers.row()) {const auto definition=triggers.text(0);
                require(definition.find("a_id")==std::string::npos&&definition.find("b_id")==std::string::npos,"converted history triggers use native endpoint keys");++count;}
            require(count==6,"conversion retains all endpoint history triggers");}
        { Statement checks(sql,"SELECT m.bytes=t.bytes,(SELECT count(*) FROM sqlite_schema WHERE name='extension_chunk_length') "
            "FROM extension_marks m CROSS JOIN extension_chunk_total t WHERE m.id=1");
          require(checks.row() && checks.i64(0)==1 && checks.i64(1)==1,"conversion preserves dependent views, trigger output and chunk index"); }
        read.commit();
        auto w=db.write(); db.setMeta(w,object,"post_v2","true");
        exec(ObjectStoreImplAccess::conn(w),"INSERT INTO extension_marks(id) VALUES(2)");
        { Statement checks(ObjectStoreImplAccess::conn(w),"SELECT m.bytes=t.bytes FROM extension_marks m CROSS JOIN extension_chunk_total t WHERE m.id=2");
          require(checks.row() && checks.i64(0)==1,"cross-table extension trigger remains executable after conversion"); }
        w.commit();
        auto after=db.read(); require(db.objectHistory(after,object).size()==3,"new history extends old history");
        require(db.currentObjectRevision(after,object)->generation==db.provisioning(after).generation,"new history has global publication generation");
    }
    mustFail([&] { convertV1ToV2(path); });
    int64_t previousGeneration=0;
    for (const auto mode : {HistoryMode::Off,HistoryMode::On,HistoryMode::Off}) {
        changeHistoryMode(path,mode);
        auto db=ObjectStore::open(path,OpenOptions{.author="test"});
        auto read=db.read(); const auto p=db.provisioning(read);
        require(p.history==mode && p.generation>previousGeneration,"history policy persists and never resets generation");
        previousGeneration=p.generation;
        require(db.uuid(read)==storeId,"history transition preserves logical identity");
        const auto token=db.currentGeometry(read,object)->uuid;
        require(db.readGeometryArray(read,token,"line_points").size()==6*sizeof(double),"history transition preserves geometry");
        if(mode==HistoryMode::On) require(db.objectHistory(read,object).size()==1,"history enablement begins a new retained baseline");
        else mustFail([&] { db.objectHistory(read,object); });
        read.commit();
        auto write=db.write(); db.setMeta(write,object,"after_transition","true"); write.commit();
    }
    { auto db=ObjectStore::open(path,OpenOptions{.author="test"}); auto w=db.write(); db.seal(w); w.commit(); }
    mustFail([&] { changeHistoryMode(path,HistoryMode::On); });
    for (const auto& file : std::filesystem::directory_iterator(dir.path))
        require(file.path().filename().string().find(".v2-")==std::string::npos && file.path().filename().string().find(".policy-")==std::string::npos,"staging cleaned after success and failure");
    return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
