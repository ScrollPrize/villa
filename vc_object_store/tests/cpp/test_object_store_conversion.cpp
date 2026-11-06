#include "TestSupport.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include <iostream>
using namespace vc::objects;
using namespace vc::objects::detail;

// A schema-17 fixture with real typed geometry and retained history. Remove
// only the v2 provisioning additions; never keep a legacy runtime code path.
void schema17(const std::filesystem::path& path)
{
    std::unique_ptr<sqlite3,decltype(&sqlite3_close)> db(openConnection(path,SQLITE_OPEN_READWRITE,{},false),sqlite3_close);
    std::vector<std::string> triggers;
    Statement rows(db.get(), "SELECT name FROM sqlite_schema WHERE type='trigger' AND instr(sql,'store_write_state')>0");
    while (rows.row()) triggers.push_back(rows.text(0));
    for (const auto& name : triggers) exec(db.get(), ("DROP TRIGGER \""+name+"\"").c_str());
    exec(db.get(),"DROP TRIGGER geometry_chunks_external_insert; DROP TRIGGER geometry_chunks_external_update;");
    exec(db.get(), "DROP TABLE table_key_columns; DROP TABLE table_directory; DROP TABLE database_identity; DROP TABLE key_allocators;"
        "DROP TABLE store_required_features; DROP TABLE store_write_state; DROP TABLE store_manifest;"
        "ALTER TABLE object_revisions DROP COLUMN generation; ALTER TABLE relationship_revisions DROP COLUMN generation;"
        "DROP INDEX edges_unique; ALTER TABLE edges DROP COLUMN relation_key;"
        "CREATE UNIQUE INDEX edges_unique ON edges(kind,a_id,b_id,a_dependencies,b_dependencies); PRAGMA user_version=17;");
    // Real v1 chunks are NOT NULL. Keep native triggers while reconstructing
    // that legacy table, before adding the extension dependency regression.
    std::vector<std::pair<std::string,std::string>> programs;
    { Statement schema(db.get(),"SELECT name,sql FROM sqlite_schema WHERE type='trigger'");
      while(schema.row()) programs.emplace_back(schema.text(0),schema.text(1)); }
    exec(db.get(),"PRAGMA foreign_keys=OFF;");
    for(const auto& [name,sql]:programs)exec(db.get(),("DROP TRIGGER \""+name+"\"").c_str());
    exec(db.get(),"CREATE TABLE legacy_chunks(hash TEXT PRIMARY KEY CHECK(length(hash)=64),"
        "data BLOB NOT NULL CHECK(length(data)>0 AND length(data)<=1048576));"
        "INSERT INTO legacy_chunks(rowid,hash,data) SELECT rowid,hash,data FROM geometry_chunks;"
        "DROP TABLE geometry_chunks; ALTER TABLE legacy_chunks RENAME TO geometry_chunks;");
    for(const auto& [name,sql]:programs)exec(db.get(),sql.c_str());
    exec(db.get(),"PRAGMA foreign_keys=ON;"
        "CREATE VIEW extension_chunk_sizes AS SELECT hash,length(data) bytes FROM geometry_chunks;"
        "CREATE VIEW extension_chunk_total AS SELECT sum(bytes) bytes FROM extension_chunk_sizes;"
        "CREATE INDEX extension_chunk_length ON geometry_chunks(length(data));"
        "CREATE TABLE extension_marks(id INTEGER PRIMARY KEY,bytes INTEGER);"
        "CREATE TRIGGER extension_mark AFTER INSERT ON extension_marks BEGIN "
        "UPDATE extension_marks SET bytes=(SELECT bytes FROM extension_chunk_total) WHERE id=NEW.id; END;"
        "INSERT INTO extension_marks(id) VALUES(1);");
}

int main() try
{
    TemporaryDirectory dir;
    const auto path = dir.path / "legacy.vcobj";
    std::string storeId, object, geometry, exported;
    {
        auto db = ObjectStore::create(path,OpenOptions{.author="test"});
        auto w = db.write(); ObjectRecord r; r.kind="fiber"; r.fiber.emplace();
        object=db.upsertObject(w,r);
        geometry=db.putGeometry(w,object,vc::geometry::encode(vc::geometry::Fiber{{},{{1,2,3},{4,5,6}}, {}}));
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
