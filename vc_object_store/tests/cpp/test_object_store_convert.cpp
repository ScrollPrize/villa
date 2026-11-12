// Offline conversion of upstream schema-17 through schema-20 stores.
// The fixtures (data/convert_schema17.sql, data/convert_schema18.sql) were
// written by the schema-18 runtime from one history (data/convert_schema19.sql
// is the schema-18 fixture converted by the last schema-19 converter): tracks with geometry,
// metadata edits, relationships retained and kept recorded through a geometry
// edit, a removal, retirement and deletion, copy provenance naming an old
// revision UUID, and a tiled surface with a sparse (delta) edit.
// data/convert_expected.json holds what that runtime read back: every
// revision, its numbered version and commit, the feed, relationship heads and
// every geometry descriptor and array. A converted store must read back the
// same, with the source untouched.
#include "TestSupport.hpp"
#include "vc_object_store/migration/ConvertStore.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "internal/SchemaPrograms.hpp"

#include <sqlite3.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
const std::filesystem::path kData = VC_OBJECT_STORE_TEST_DATA;

std::string bytesOf(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

void sql(const std::filesystem::path& path, const std::string& text)
{
    sqlite3* db = nullptr;
    require(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK, "open fixture");
    char* error = nullptr;
    const int rc = sqlite3_exec(db, text.c_str(), nullptr, nullptr, &error);
    const std::string message = error ? error : "";
    sqlite3_free(error);
    sqlite3_close(db);
    if (rc != SQLITE_OK) throw std::runtime_error("fixture SQL: " + message);
}

int64_t scalar(const std::filesystem::path& path, const std::string& query)
{
    sqlite3* db = nullptr;
    require(sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK, "open store");
    sqlite3_stmt* s = nullptr;
    require(sqlite3_prepare_v2(db, query.c_str(), -1, &s, nullptr) == SQLITE_OK, ("prepare " + query).c_str());
    require(sqlite3_step(s) == SQLITE_ROW, ("row " + query).c_str());
    const auto value = sqlite3_column_int64(s, 0);
    sqlite3_finalize(s);
    sqlite3_close(db);
    return value;
}

std::string hex(const std::vector<uint8_t>& bytes)
{
    std::string out;
    for (auto b : bytes) { char t[3]; std::snprintf(t, 3, "%02x", b); out += t; }
    return out;
}

// Add artifacts to an authentic frozen upstream fixture, not a current store
// whose user_version was merely changed to impersonate an old schema.
size_t fixtureArtifacts(const std::filesystem::path& path, const std::string& object,
                       const std::string& first, const std::string& later)
{
    sqlite3* db = nullptr;
    require(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK, "open artifact fixture");
    const auto close = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>(db, sqlite3_close);
    sqlite3_stmt* rows = nullptr;
    require(sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master WHERE type='trigger'", -1, &rows, nullptr) == SQLITE_OK, "fixture triggers");
    std::vector<std::string> triggers;
    while (sqlite3_step(rows) == SQLITE_ROW) triggers.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(rows, 0)));
    sqlite3_finalize(rows);
    for (const auto& trigger : triggers)
        require(sqlite3_exec(db, ("DROP TRIGGER \"" + trigger + "\"").c_str(), nullptr, nullptr, nullptr) == SQLITE_OK, "drop fixture trigger");
    const auto key = util::canonicalUuid(object);
    auto compact = key; compact.erase(std::remove(compact.begin(), compact.end(), '-'), compact.end());
    const auto query = "SELECT uuid,parent_uuid,metadata_delta FROM object_revisions WHERE object_uuid=x'" + compact + "' ORDER BY seq";
    require(sqlite3_prepare_v2(db, query.c_str(), -1, &rows, nullptr) == SQLITE_OK, "fixture revisions");
    std::vector<std::pair<std::vector<uint8_t>, Json>> rewritten;
    Json old = Json::object(), previous = Json::object(), current;
    while (sqlite3_step(rows) == SQLITE_ROW) {
        const auto delta = Json::parse(reinterpret_cast<const char*>(sqlite3_column_text(rows, 2)));
        old = delta.is_object() ? delta : old.patch(delta);
        current = old;
        current["meta"][kArtifactsMetaKey] = Json::parse(rewritten.empty() ? first : later);
        const auto* bytes = static_cast<const uint8_t*>(sqlite3_column_blob(rows, 0));
        const bool root = sqlite3_column_type(rows, 1) == SQLITE_NULL;
        rewritten.emplace_back(std::vector<uint8_t>(bytes, bytes + sqlite3_column_bytes(rows, 0)), root ? current : Json::diff(previous, current));
        previous = current;
    }
    sqlite3_finalize(rows);
    require(rewritten.size() >= 2, "fixture has two artifact versions");
    sqlite3_stmt* update = nullptr;
    require(sqlite3_prepare_v2(db, "UPDATE object_revisions SET metadata_delta=? WHERE uuid=?", -1, &update, nullptr) == SQLITE_OK, "update fixture revisions");
    for (const auto& [uuid, delta] : rewritten) {
        const auto json = delta.dump();
        sqlite3_bind_text(update, 1, json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(update, 2, uuid.data(), int(uuid.size()), SQLITE_TRANSIENT);
        require(sqlite3_step(update) == SQLITE_DONE, "write fixture revision");
        sqlite3_reset(update);
    }
    sqlite3_finalize(update);
    const auto writeJson = [&](const std::string& statement, const Json& value) {
        sqlite3_stmt* target = nullptr;
        require(sqlite3_prepare_v2(db, statement.c_str(), -1, &target, nullptr) == SQLITE_OK, "fixture current update");
        const auto text = value.dump(); sqlite3_bind_text(target, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        const auto result = sqlite3_step(target); sqlite3_finalize(target);
        require(result == SQLITE_DONE, "write fixture current record");
    };
    writeJson("UPDATE object_head_records SET record=? WHERE object_uuid=x'" + compact + "'", current);
    writeJson("UPDATE objects SET meta=? WHERE uuid=x'" + compact + "'", current.at("meta"));
    return rewritten.size();
}

// Everything the fixture's runtime read back, read again from `path`.
// `generation`: the converted store's generation replaces the recorded one
// when a schema-17 source got a new one.
void requireSameHistory(const std::filesystem::path& path, const Json& expected, const std::optional<std::string>& generation)
{
    OpenOptions options; options.readOnly = true;
    auto db = ObjectStore::open(path, options);
    auto txn = db.read();
    const auto status = db.status(txn);
    require(status.storeUuid == expected.at("store").get<std::string>() && status.commitSequence == expected.at("sequence").get<int64_t>(), "identity and sequence");
    for (const auto& object : expected.at("objects")) {
        const auto history = db.objectHistory(txn, object);
        const auto& recorded = expected.at("history").at(object.get<std::string>());
        require(history.size() == recorded.size(), "revision count");
        for (size_t i = 0; i < history.size(); ++i) {
            auto want = recorded.at(i);
            if (generation) want["generation"] = *generation;
            auto& record = want.at("record");
            record["approved_version"] = record.at("approved_generation").is_null() ? Json(nullptr) : want.at("version");
            record.erase("approved_generation"); record.erase("generation"); record.erase("data_generation");
            record["version"] = want.at("version");
            record["commit_sequence"] = want.at("commit_sequence").is_null() ? Json(0) : want.at("commit_sequence");
            record["version_generation"] = want.at("generation");
            require(Json::parse(objectRevisionToJson(history[i])) == want, ("revision " + std::to_string(i) + " of " + object.get<std::string>()).c_str());
        }
    }
    for (const auto& [uuid, geometry] : expected.at("geometry").items()) {
        const auto g = db.geometryRevision(txn, uuid);
        auto descriptor = geometry.at("descriptor");
        descriptor.erase("object_generation");
        int64_t version = 0;
        for (const auto& history : expected.at("history"))
            for (const auto& state : history)
                if (state.at("geometry_revision") == uuid && (!version || state.at("version").get<int64_t>() < version))
                    version = state.at("version").get<int64_t>();
        descriptor["object_version"] = version;
        require(g && Json::parse(geometryRevisionToJson(*g)) == descriptor, ("geometry descriptor " + uuid).c_str());
        for (const auto& [name, bytes] : geometry.at("arrays").items())
            require(hex(db.readGeometryArray(txn, uuid, name)) == bytes.get<std::string>(), ("geometry array " + name).c_str());
    }
    for (const auto& [edge, version] : expected.at("relationship_heads").items()) {
        const auto head = db.currentRelationshipVersion(txn, edge);
        require((head ? Json(*head) : Json(nullptr)) == version, "relationship head");
    }
    Json feed = Json::array();
    for (const auto& commit : db.changesSince(txn, {status.storeUuid, status.generation, 0, 0}, 1 << 20).commits)
        for (const auto& change : commit.changes) feed.push_back({commit.sequence, change.uuid, change.version, change.deleted});
    require(feed == expected.at("feed"), "feed");
}

// A converted store keeps working: a checked edit, and the old version still
// reads at its number.
void requireWritable(const std::filesystem::path& path, const std::string& object)
{
    OpenOptions options; options.author = "convert-test";
    auto db = ObjectStore::open(path, options);
    StoreStatus status; std::optional<int64_t> before; Json oldRecord;
    std::string publicationGeneration;
    { auto txn = db.read(); status = db.status(txn); before = db.currentVersion(txn, object);
      const auto revision = *db.currentObjectRevision(txn, object);
      oldRecord = Json::parse(objectRevisionToJson(revision)).at("record"); publicationGeneration = revision.generation; }
    TransactionRequest r; r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
    r.author = "convert-test"; r.expectedObjects = {{object, before}};
    r.operations = {SetMetaOp{object, "after_conversion", std::to_string(*before + 1)}};
    const auto receipt = db.submit(r);
    require(receipt.objects.at(object) == *before + 1, "edit after conversion");
    auto txn = db.read();
    const auto old = db.objectAtVersion(txn, {status.storeUuid, publicationGeneration, object, *before});
    require(old && Json::parse(objectRevisionToJson(*old)).at("record") == oldRecord, "old version reads back");
}

// A frozen schema-21 edge layout around records made by the current public
// APIs. The table is rebuilt, not merely relabelled as an older schema.
void downgradeFixture21(const std::filesystem::path& path)
{
    using namespace vc::objects::detail;
    sqlite3* db=nullptr;require(sqlite3_open(path.c_str(),&db)==SQLITE_OK,"open schema21 fixture");
    struct Close{sqlite3* db;~Close(){sqlite3_close(db);}} close{db};
    const auto programs=schemaPrograms(db);exec(db,"PRAGMA journal_mode=DELETE;PRAGMA foreign_keys=OFF;BEGIN IMMEDIATE");
    suspendPrograms(db,programs,true);
    exec(db,R"SQL(CREATE TABLE old_edges(
      id INTEGER PRIMARY KEY,uuid BLOB NOT NULL UNIQUE,state INTEGER NOT NULL DEFAULT 0,
      a_uuid BLOB NOT NULL,b_uuid BLOB NOT NULL,kind TEXT NOT NULL,
      commit_sequence INTEGER NOT NULL DEFAULT 0,version INTEGER NOT NULL DEFAULT 1,version_generation BLOB,
      created_by TEXT NOT NULL,created_at TEXT NOT NULL,created_host TEXT NOT NULL,
      changed_by TEXT NOT NULL,changed_at TEXT NOT NULL,changed_host TEXT NOT NULL,
      approval_state TEXT NOT NULL DEFAULT 'pending',approved_by TEXT,approved_at TEXT,approved_host TEXT,approved_version INTEGER,
      a_anchor_x REAL,a_anchor_y REAL,a_anchor_z REAL,a_dir_x REAL,a_dir_y REAL,a_dir_z REAL,
      b_anchor_x REAL,b_anchor_y REAL,b_anchor_z REAL,b_dir_x REAL,b_dir_y REAL,b_dir_z REAL,
      meta TEXT NOT NULL DEFAULT '{}',a_dependencies TEXT NOT NULL DEFAULT '{}',b_dependencies TEXT NOT NULL DEFAULT '{}');)SQL");
    std::string projection="row_number() OVER(ORDER BY uuid)";
    Statement fields(db,"SELECT name FROM pragma_table_info('edges') ORDER BY cid");
    while(fields.row()){
        const auto name=fields.text(0);projection+=',';
        if(name=="kind"){
            projection+="CASE kind";for(size_t i=0;i<kCoreEdgeKinds.size();++i)
                projection+=" WHEN "+std::to_string(i)+" THEN '"+std::string(kCoreEdgeKinds[i])+"'";
            projection+=" END";
        }else if(name=="approval_state")projection+="CASE approval_state WHEN 0 THEN 'pending' WHEN 1 THEN 'approved' ELSE 'rejected' END";
        else if(name=="meta"||name=="a_dependencies"||name=="b_dependencies")projection+="coalesce("+name+",'{}')";
        else projection+=name;
    }
    exec(db,("INSERT INTO old_edges SELECT "+projection+" FROM edges").c_str());
    exec(db,R"SQL(CREATE TABLE old_edge_tags(edge_id INTEGER NOT NULL REFERENCES edges(id) ON DELETE CASCADE,tag TEXT NOT NULL,PRIMARY KEY(edge_id,tag));
      INSERT INTO old_edge_tags SELECT e.id,t.tag FROM edge_tags t JOIN old_edges e ON e.uuid=t.edge_uuid;
      DROP TABLE edge_tags;DROP TABLE edges;ALTER TABLE old_edges RENAME TO edges;ALTER TABLE old_edge_tags RENAME TO edge_tags;
      CREATE INDEX edges_a ON edges(a_uuid,kind);CREATE INDEX edges_b ON edges(b_uuid,kind);CREATE INDEX edge_tags_tag ON edge_tags(tag);
      CREATE TABLE old_manifest(singleton INTEGER PRIMARY KEY,schema_version INTEGER NOT NULL CHECK(schema_version=21),
        commit_sequence INTEGER NOT NULL,generation BLOB NOT NULL,initial_generation BLOB NOT NULL,feed_start INTEGER NOT NULL,
        history_enabled INTEGER NOT NULL,sealed INTEGER NOT NULL,spatial_enabled INTEGER NOT NULL);
      INSERT INTO old_manifest SELECT singleton,21,commit_sequence,generation,initial_generation,feed_start,history_enabled,sealed,spatial_enabled FROM store_manifest;
      DROP TABLE store_manifest;ALTER TABLE old_manifest RENAME TO store_manifest;
      CREATE TABLE old_allocators(name TEXT PRIMARY KEY CHECK(name IN ('objects','edges')),namespace_uuid BLOB NOT NULL,
        lower_bound INTEGER NOT NULL,upper_bound INTEGER NOT NULL,next_id INTEGER);
      INSERT INTO old_allocators SELECT * FROM key_allocators;
      INSERT INTO old_allocators SELECT 'edges',uuid,1,9223372036854775807,(SELECT coalesce(max(id),0)+1 FROM edges) FROM store_identity;
      DROP TABLE key_allocators;ALTER TABLE old_allocators RENAME TO key_allocators;
      UPDATE table_key_columns SET name='id' WHERE table_uuid=(SELECT uuid FROM table_directory WHERE name='edges');
      UPDATE table_key_columns SET name='edge_id' WHERE table_uuid=(SELECT uuid FROM table_directory WHERE name='edge_tags') AND ordinal=1;
      UPDATE database_identity SET schema_version=21;PRAGMA user_version=21;COMMIT;)SQL");
    {Statement collection(db,"SELECT 1 FROM sqlite_schema WHERE name='collection_members'");
        if(collection.row())exec(db,"ALTER TABLE collection_members ADD COLUMN edge_low INTEGER NOT NULL DEFAULT 0;ALTER TABLE collection_members ADD COLUMN edge_high INTEGER NOT NULL DEFAULT 0;");}
    // History tag triggers are the only library trigger bodies that changed
    // with the numeric-to-UUID key transition.
    auto sourcePrograms=programs;
    for(auto& p:sourcePrograms)if(p.owner=="edge_tags"){
        for(const auto* row:{"new","old"}){
            auto replace=[&](const std::string& from,const std::string& to){size_t at=0;while((at=p.sql.find(from,at))!=std::string::npos){p.sql.replace(at,from.size(),to);at+=to.size();}};
            replace("uuid="+std::string(row)+".edge_uuid","id="+std::string(row)+".edge_id");
        }
    }
    restorePrograms(db,sourcePrograms,true);
}

void schema21Cases(const std::filesystem::path& root)
{
    for(const bool history:{false,true}){
        const auto source=root/(history?"source21-history.vcobj":"source21-direct.vcobj");
        OpenOptions options;options.author="convert21";options.history=history?HistoryMode::On:HistoryMode::Off;options.spatialIndex=false;
        std::string a,b,e,physical,storeId,lineage,artifactHash;int64_t sequence;
        const auto artifactFile=root/(history?"artifact-history.bin":"artifact-direct.bin");
        {std::ofstream bytes(artifactFile,std::ios::binary);bytes<<"abc";}
        {auto store=ObjectStore::create(source,options);auto w=store.write();ObjectRecord obj;obj.kind="note";obj.meta=R"({"fixture":true})";
            a=store.upsertObject(w,obj);b=store.upsertObject(w,obj);
            EdgeRecord edge;edge.kind="parallel";edge.aUuid=a;edge.bUuid=b;edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
            edge.aAnchor.position=std::array<double,3>{1,2,3};edge.bAnchor.position=std::array<double,3>{4,5,6};edge.tags={"retained"};edge.meta=R"({"value":4})";
            e=store.addEdge(w,edge);
            ExtensionTable ext;ext.name="ext_conversion_payload";ext.extension="conversion.test";ext.keySpace="edges";
            ext.columns={{"edge_uuid",ExtensionType::Blob},{"payload",ExtensionType::Blob}};ext.primaryKey={"edge_uuid"};
            store.createExtensionTable(w,ext);const std::vector<ExtensionRow> rows{{extensionUuid(e),std::vector<uint8_t>{0,4,255}}};
            store.appendExtensionRows(w,ext.name,rows);
            artifactHash=store.putArtifact(artifactFile);
            store.setMeta(w,a,kArtifactsMetaKey,artifactManifestJson({{"artifact.bin",artifactHash,3}}));w.commit();
            if(history){auto remove=store.write();store.removeMeta(remove,a,kArtifactsMetaKey);remove.commit();}
            auto r=store.read();physical=store.databaseIdentity(r).uuid;storeId=store.uuid(r);const auto status=store.status(r);lineage=status.generation;sequence=status.commitSequence;
        }
        downgradeFixture21(source);
        sql(source,"CREATE VIEW extension_edge_ids AS SELECT uuid FROM edges;CREATE INDEX extension_payload_lookup ON ext_conversion_payload(payload);");
        const auto before=bytesOf(source);
        const auto destination=root/(history?"converted21-history.vcobj":"converted21-direct.vcobj");
        auto converted=migration::convertStore(source,destination);
        require(bytesOf(source)==before,"schema21 source unchanged");
        require(Json::parse(converted.reportJson).at("target_schema")==22,"schema21 conversion report");
        {auto db=ObjectStore::open(destination,OpenOptions{.readOnly=true});auto r=db.read();
            require(db.uuid(r)==storeId&&db.databaseIdentity(r).uuid==physical,"schema21 identities preserved");
            require(db.status(r).generation==lineage&&db.status(r).commitSequence==sequence,"schema21 snapshot preserved");
            const auto edge=db.edgeByUuid(r,e);require(edge&&edge->kind=="parallel"&&edge->tags==std::vector<std::string>{"retained"}&&edge->aAnchor.position,"schema21 populated edge preserved");
            std::vector<ExtensionRow> rows;db.scanExtensionRows(r,"ext_conversion_payload",[&](const auto& row){rows.push_back(row);return true;});
            require(rows.size()==1&&std::get<std::vector<uint8_t>>(rows[0][1])==std::vector<uint8_t>({0,4,255}),"schema21 extension preserved");
            if(history)require(!db.objectHistory(r,a).empty(),"schema21 history preserved");
            require(db.readArtifact(artifactHash)==std::vector<uint8_t>({'a','b','c'}),"schema21 current/historical artifacts preserved");
        }
        require(scalar(destination,"SELECT count(*) FROM pragma_table_info('edges') WHERE name='id'")==0,"schema22 UUID-only edge key");
        require(scalar(destination,"SELECT count(*) FROM key_allocators WHERE name='edges'")==0,"schema22 no edge allocator");
        mustFail([&]{sql(destination,"INSERT INTO key_allocators VALUES('edges',zeroblob(16),1,100,1)");});
        require(scalar(destination,"SELECT count(*) FROM extension_edge_ids")==1&&
                scalar(destination,"SELECT count(*) FROM sqlite_schema WHERE name='extension_payload_lookup'")==1,
                "compatible extension views and indexes preserved");
        if(!history){
            for(const auto& [label,change]:std::vector<std::pair<std::string,std::string>>{
                {"unknown-kind","UPDATE edges SET kind='not_a_core_kind'"},
                {"numeric-ref","CREATE TABLE ext_numeric(edge_id INTEGER REFERENCES edges(id));INSERT INTO ext_numeric VALUES(1)"},
                {"kind-view","CREATE VIEW old_kind_test AS SELECT uuid FROM edges WHERE kind='parallel'"}}){
                const auto bad=root/(label+"21.vcobj"),output=root/(label+"22.vcobj");std::filesystem::copy_file(source,bad);sql(bad,change);
                const auto old=bytesOf(bad);mustFail([&]{migration::convertStore(bad,output);});
                require(bytesOf(bad)==old&&!std::filesystem::exists(output),"unsupported21 extension rejects without source changes or output");
            }
        }
        // Packed publication is readonly regardless of its original SQLite
        // policy, so use a sealed source snapshot for the packed fixture.
        sql(source,"UPDATE store_manifest SET sealed=1");
        const auto sealedPacked=root/(history?"sealed21-history.packed":"sealed21-direct.packed");
        {PackedStoreBuilder builder(sealedPacked);builder.importDatabase(physical,source);builder.finalize(physical);}
        const auto packedArtifact=artifactBlobPath(sealedPacked,artifactHash);
        std::filesystem::create_directories(packedArtifact.parent_path());
        std::filesystem::copy_file(artifactBlobPath(source,artifactHash),packedArtifact);
        const auto convertedPacked=root/(history?"converted21-history.packed":"converted21-direct.packed");
        migration::convertStore(sealedPacked,convertedPacked);
        {auto db=ObjectCollection::open(convertedPacked,OpenOptions{.readOnly=true});auto r=db.read();require(db.uuid(r)==storeId,"packed21 identity preserved");}
        {auto db=ObjectStore::open(packedDatabasePath(convertedPacked),OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
            require(db.readArtifact(artifactHash)==std::vector<uint8_t>({'a','b','c'}),"packed artifact paths refer to container not locator URI");}
    }
}
void schema21LinkedCase(const std::filesystem::path& root)
{
    const auto base=root/"linked21-base.vcobj",source=root/"linked21-root.vcobj";
    OpenOptions opts;opts.author="convert21-linked";opts.history=HistoryMode::Off;opts.spatialIndex=false;
    std::string a,b,e,owner,rootOwner,rootPhysical,basePhysical;
    {auto store=ObjectStore::create(base,opts);auto w=store.write();ObjectRecord object;object.kind="note";object.meta=R"({"fixture":true})";
        a=store.upsertObject(w,object);b=store.upsertObject(w,object);store.seal(w);w.commit();
        auto r=store.read();owner=store.uuid(r);basePhysical=store.databaseIdentity(r).uuid;}
    {auto collection=ObjectCollection::create(source,{base},opts);auto w=collection.write();
        EdgeRecord edge;edge.kind="perpendicular";edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
        e=collection.addEdge(w,{owner,a},{owner,b},edge).objectUuid;collection.seal(w);w.commit();
        auto r=collection.read();rootOwner=collection.uuid(r);rootPhysical=collection.members().front().databaseUuid;}
    downgradeFixture21(base);downgradeFixture21(source);const auto before=bytesOf(source),beforeBase=bytesOf(base);
    const auto destination=root/"converted21-linked.vcobj";
    auto result=migration::convertStore(source,destination);
    require(Json::parse(result.reportJson).at("physical_databases")==2,"linked21 complete closure converted");
    require(bytesOf(source)==before&&bytesOf(base)==beforeBase,"linked21 sources unchanged");
    require(scalar(destination,"SELECT count(*) FROM pragma_table_info('collection_members') WHERE name IN ('edge_low','edge_high')")==0,
            "schema22 collection drops numeric edge reservations");
    {auto collection=ObjectCollection::open(destination,OpenOptions{.readOnly=true});auto r=collection.read();
        require(collection.object(r,{owner,a}).has_value()&&collection.edge(r,{rootOwner,e}).has_value(),"converted linked21 objects and edges resolve");}
    const auto packed=root/"linked21.packed";
    {PackedStoreBuilder builder(packed);builder.importDatabase(rootPhysical,source);builder.importDatabase(basePhysical,base);builder.finalize(rootPhysical);}
    const auto convertedPacked=root/"converted21-linked.packed";
    migration::convertStore(packed,convertedPacked);
    {auto collection=ObjectCollection::open(convertedPacked,OpenOptions{.readOnly=true});auto r=collection.read();
        require(collection.members().size()==2&&collection.edge(r,{rootOwner,e}).has_value(),"packed linked21 closure resolves");}
    migration::convertStoreInPlace(packed);
    {auto collection=ObjectCollection::open(packed,OpenOptions{.readOnly=true});auto r=collection.read();
        require(collection.edge(r,{rootOwner,e}).has_value(),"packed21 in-place conversion");}
    migration::convertStoreInPlace(source);
    {auto collection=ObjectCollection::open(source,OpenOptions{.readOnly=true});auto r=collection.read();
        require(collection.object(r,{owner,a}).has_value(),"plain linked21 in-place dependencies remain available");}
}
}

int main()
try {
    TemporaryDirectory directory;
    schema21Cases(directory.path);
    schema21LinkedCase(directory.path);
    const auto expected = Json::parse(bytesOf(kData / "convert_expected.json"));
    OpenOptions options; options.author = "convert-test";
    for (const int schema : {17, 18, 19, 20}) {
        const auto source = directory.path / ("old" + std::to_string(schema) + ".vcobj");
        // Upstream schema 20 differs from 19 only by dropping replica identity.
        sql(source, bytesOf(kData / ("convert_schema" + std::to_string(std::min(schema, 19)) + ".sql")));
        if (schema == 20) sql(source, "DROP TABLE sync_replica; PRAGMA user_version=20;");
        const auto before = bytesOf(source);
        mustFail([&] { (void)ObjectStore::open(source, options); }); // the runtime refuses old schemas

        const auto destination = directory.path / ("new" + std::to_string(schema) + ".vcobj");
        const auto result = migration::convertStore(source, destination);
        require(bytesOf(source) == before, "source untouched");
        require(!std::filesystem::exists(destination.string() + ".converting"), "no staging file left");
        const auto report = Json::parse(result.reportJson);
        const auto mapping = Json::parse(result.mappingJson);
        require(report.at("validation") == "ok" && report.at("source_role") == "standalone", "report");
        require(report.at("source_schema") == schema && report.at("target_schema") == 22, "schemas reported");
        mustFail([&] { (void)migration::convertStore(source, destination); }); // never overwrites
        // Storage form: whole records for revisions without a parent, no
        // materialized head record duplicating one, 16-byte keys.
        require(scalar(destination, "SELECT count(*) FROM object_revisions WHERE parent_uuid IS NULL AND json_type(metadata_delta)<>'object'") == 0,
                "root revisions hold records");
        require(scalar(destination, "SELECT count(*) FROM object_head_records h JOIN object_revisions r ON r.uuid=h.revision_uuid "
                                    "WHERE r.parent_uuid IS NULL") == 0, "no duplicate head records");
        require(scalar(destination, "SELECT count(*) FROM object_versions WHERE length(object_uuid)<>16 OR length(revision_uuid)<>16") == 0,
                "blob keys");
        require(scalar(destination, "SELECT count(*) FROM sqlite_master WHERE name='sync_replica'") == 0,
                "no obsolete replica identity");
        if (schema < 20)
            require(std::find(report.at("removed_tables").begin(), report.at("removed_tables").end(), "sync_replica") !=
                    report.at("removed_tables").end(), "replica identity dropped");

        requireSameHistory(destination, expected,
                           schema == 17 ? std::optional(mapping.at("generation").get<std::string>()) : std::nullopt);
        {
            const auto ref = mapping.at("objects").at(expected.at("first_revision").get<std::string>());
            OpenOptions readOnly; readOnly.readOnly = true;
            auto db = ObjectStore::open(destination, readOnly);
            auto txn = db.read();
            const auto revision = db.objectAtVersion(txn, {expected.at("store"), mapping.at("generation"), ref.at("object"), ref.at("version")});
            require(revision && revision->uuid == expected.at("first_revision").get<std::string>(), "mapping resolves");
        }
        requireWritable(destination, expected.at("objects").at(1));
        if (schema != 17) continue;
        require(report.at("gaps").at("metadata_revision_references").at("count") == 2, "provenance reference reported");
        require(report.at("gaps").at("untimed_object_versions").at("count") == 0, "no untimed history");
        // Deterministic: converting again yields the same mapping.
        require(migration::convertStore(source, directory.path / "again.vcobj").mappingJson == result.mappingJson, "deterministic mapping");
        // A replica with unpublished edits is refused and leaves nothing behind.
        sql(source, "CREATE TABLE hub_upstream(singleton INTEGER PRIMARY KEY); CREATE TABLE hub_pending(record_uuid TEXT PRIMARY KEY, "
                    "counter INTEGER, geometry_after INTEGER); INSERT INTO hub_pending VALUES('x',1,0);");
        const auto refused = directory.path / "refused.vcobj";
        mustFail([&] { (void)migration::convertStore(source, refused); });
        require(!std::filesystem::exists(refused) && !std::filesystem::exists(refused.string() + ".converting"), "no partial destination");
        sql(source, "DELETE FROM hub_pending;");
        const auto settled = Json::parse(migration::convertStore(source, directory.path / "settled.vcobj").reportJson);
        require(settled.at("source_role") == "replica", "replica conversion reported");
        for (const auto* table : {"hub_pending", "hub_upstream", "sync_replica"})
            require(std::find(settled.at("removed_tables").begin(), settled.at("removed_tables").end(), table) !=
                    settled.at("removed_tables").end(), "coordination removed");
    }
    // The unpublished experimental branch reused version 18, but it is not
    // upstream schema 18 and must never be treated as a migration input.
    {
        const auto source = directory.path / "unpublished18.vcobj";
        sql(source, bytesOf(kData / "convert_schema18.sql"));
        sql(source, "CREATE TABLE store_manifest(schema_version INTEGER); INSERT INTO store_manifest VALUES(18);");
        const auto before = bytesOf(source);
        const auto destination = directory.path / "unpublished-converted.vcobj";
        mustFail([&] { (void)migration::convertStore(source, destination); });
        require(bytesOf(source) == before && !std::filesystem::exists(destination) &&
                !std::filesystem::exists(destination.string() + ".converting"), "unpublished layout rejected without changes");
    }
    {
        const auto source = directory.path / "in-place20.vcobj";
        sql(source, bytesOf(kData / "convert_schema19.sql"));
        sql(source, "DROP TABLE sync_replica; PRAGMA user_version=20;");
        const auto receipts = scalar(source, "SELECT count(*) FROM transaction_receipts");
        sql(source, "CREATE VIEW extension_current AS SELECT uuid FROM objects;");
        const auto before = bytesOf(source);
        mustFail([&] { (void)migration::convertStoreInPlace(source); });
        require(bytesOf(source) == before && scalar(source, "PRAGMA user_version") == 20,
                "unsupported extension rejects in-place conversion without modifying source");
        sql(source, "DROP VIEW extension_current;");
        const auto result = migration::convertStoreInPlace(source);
        require(Json::parse(result.reportJson).at("in_place") == true && scalar(source, "PRAGMA user_version") == 22,
                "explicit in-place conversion publishes current schema");
        requireSameHistory(source, expected, std::nullopt);
        require(scalar(source, "SELECT count(*) FROM transaction_receipts") == receipts, "in-place receipts retained");
        requireWritable(source, expected.at("objects").at(1));
    }
    // Policy transitions retain numbered current identities, including the
    // old publication lineage after an intervening history-off lineage reset.
    {
        const auto path = directory.path / "policy-converted.vcobj";
        (void)migration::convertStore(directory.path / "old20.vcobj", path);
        const auto object = expected.at("objects").at(1).get<std::string>();
        requireWritable(path, object); // Produces a request receipt to retain.
        const auto receiptCount = scalar(path, "SELECT count(*) FROM transaction_receipts");
        require(receiptCount > 0, "policy fixture contains request receipts");
        ObjectRecord before;
        StoreStatus status;
        {
            auto store = ObjectStore::open(path, options);
            auto txn = store.read(); before = *store.objectByUuid(txn, object); status = store.status(txn);
        }
        changeHistoryMode(path, HistoryMode::Off);
        {
            auto store = ObjectStore::open(path, options);
            {
                auto txn = store.read(); const auto after = *store.objectByUuid(txn, object);
                require(after.version == before.version && after.versionGeneration == before.versionGeneration &&
                        after.commitSequence == before.commitSequence, "history removal retains record identity");
                require(store.status(txn).commitSequence == status.commitSequence + 1, "history removal advances policy clock once");
            }
            store.startHistoryGeneration("conversion policy regression");
        }
        require(scalar(path, "SELECT count(*) FROM sqlite_schema WHERE name IN ('object_versions','relationship_versions','store_commits')") == 0,
                "history-off has no archival numbered tables");
        require(scalar(path, "SELECT count(*) FROM transaction_receipts") == receiptCount, "history removal preserves idempotency receipts");
        changeHistoryMode(path, HistoryMode::On);
        {
            auto store = ObjectStore::open(path, options);
            auto txn = store.read(); const auto after = *store.objectByUuid(txn, object);
            require(after.version == before.version && after.versionGeneration == before.versionGeneration &&
                    after.commitSequence == before.commitSequence, "enabling history retains original publication identity");
            require(store.status(txn).commitSequence == status.commitSequence + 3, "each policy or lineage change advances clock once");
            const auto baseline = store.objectAtVersion(txn, {status.storeUuid, before.versionGeneration, object, *before.version});
            require(baseline && !baseline->commitSequence, "history enablement archives untimed current baseline without inventing a commit");
        }
        require(scalar(path, "SELECT count(*) FROM transaction_receipts") == receiptCount, "history enablement preserves idempotency receipts");
        requireWritable(path, object);
    }
    // Schema 19 differs from 20 only by sync_replica. Preserve artifacts
    // named solely by old versions too, without modifying the source.
    {
        const auto source = directory.path / "artifacts19.vcobj";
        const auto input = directory.path / "model.pt";
        std::string object, oldHash, currentHash;
        const std::string oldBytes = "old model", currentBytes = "new model";
        {
            auto store = ObjectStore::create(directory.path / "artifact-writer.vcobj", options);
            std::ofstream(input) << oldBytes;
            oldHash = store.putArtifact(input);
            std::ofstream(input) << currentBytes;
            currentHash = store.putArtifact(input);
        }
        sql(source, bytesOf(kData / "convert_schema19.sql"));
        object = expected.at("objects").at(1).get<std::string>();
        const auto historyCount = fixtureArtifacts(source, object,
            artifactManifestJson({{"model.pt", oldHash, oldBytes.size()}}),
            artifactManifestJson({{"model.pt", currentHash, currentBytes.size()}}));
        for (const auto& [hash, content] : {std::pair{oldHash, oldBytes}, std::pair{currentHash, currentBytes}}) {
            const auto blob = artifactBlobPath(source, hash);
            std::filesystem::create_directories(blob.parent_path());
            std::ofstream(blob, std::ios::binary) << content;
        }
        const auto before = bytesOf(source);
        const auto destination = directory.path / "converted-artifacts.vcobj";
        const auto result = migration::convertStore(source, destination);
        require(Json::parse(result.reportJson).at("copied_artifacts") == 2, "both artifact versions copied");
        {
            auto store = ObjectStore::open(destination, options);
            auto txn = store.read();
            const auto history = store.objectHistory(txn, object);
            require(history.size() == historyCount, "artifact history preserved");
            for (size_t i = 0; i < history.size(); ++i) {
                const auto manifest = artifactManifest(history[i].record);
                const auto bytes = store.readArtifact(manifest.at(0).sha256);
                require(std::string(bytes.begin(), bytes.end()) == (i == 0 ? oldBytes : currentBytes), "artifact bytes preserved");
            }
        }
        require(bytesOf(source) == before && bytesOf(artifactBlobPath(source, oldHash)) == oldBytes &&
                bytesOf(artifactBlobPath(source, currentHash)) == currentBytes, "artifact source untouched");
        changeHistoryMode(destination, HistoryMode::Off);
        changeHistoryMode(destination, HistoryMode::On);
        {
            auto store = ObjectStore::open(destination, options);
            auto txn = store.read();
            const auto manifest = artifactManifest(*store.objectByUuid(txn, object));
            const auto bytes = store.readArtifact(manifest.at(0).sha256);
            require(std::string(bytes.begin(), bytes.end()) == currentBytes,
                    "history-policy staging retains artifacts at the published store location");
        }

        const auto occupied = directory.path / "occupied.vcobj";
        const auto occupiedArtifacts = artifactBlobDirectory(occupied).parent_path();
        std::filesystem::create_directories(occupiedArtifacts);
        std::ofstream(occupiedArtifacts / "keep") << "keep";
        mustFail([&] { migration::convertStore(source, occupied); });
        require(bytesOf(occupiedArtifacts / "keep") == "keep", "existing artifacts untouched");

        const auto oldBlob = artifactBlobPath(source, oldHash);
        for (const bool missing : {false, true}) {
            std::filesystem::remove(oldBlob);
            if (!missing) std::ofstream(oldBlob) << "bad model"; // same length, different hash
            const auto refused = directory.path / (missing ? "missing.vcobj" : "corrupt.vcobj");
            mustFail([&] { migration::convertStore(source, refused); });
            require(!std::filesystem::exists(refused) && !std::filesystem::exists(refused.string() + ".converting") &&
                    !std::filesystem::exists(artifactBlobDirectory(refused).parent_path()), "failed artifact conversion cleaned up");
            const auto beforeFailure = bytesOf(source);
            mustFail([&] { (void)migration::convertStoreInPlace(source); });
            require(bytesOf(source) == beforeFailure && scalar(source, "PRAGMA user_version") == 19,
                    "failed in-place artifact validation leaves upstream source untouched");
        }
        std::ofstream(oldBlob, std::ios::binary) << oldBytes;
        (void)migration::convertStoreInPlace(source);
        {
            auto store = ObjectStore::open(source, options);
            auto txn = store.read();
            require(store.objectHistory(txn, object).size() == historyCount, "in-place artifact history retained");
            require(bytesOf(artifactBlobPath(source, oldHash)) == oldBytes &&
                    bytesOf(artifactBlobPath(source, currentHash)) == currentBytes, "in-place artifacts retain their published location");
        }
        for (const auto& entry : std::filesystem::directory_iterator(directory.path))
            require(!entry.path().filename().string().starts_with(source.filename().string() + ".convert-"),
                    "in-place private conversion files cleaned on success and failure");
    }
    // Schema 16 (the patches detail table, hub sync baselines): its form of
    // the schema-17 fixture. Surface data is refused; without it the history
    // converts as from schema 17.
    {
        const auto source = directory.path / "old16.vcobj";
        sql(source, bytesOf(kData / "convert_schema17.sql"));
        sql(source, "DROP TRIGGER surfaces_history_INSERT; DROP TRIGGER surfaces_history_UPDATE; DROP TRIGGER surfaces_history_DELETE;"
                    "ALTER TABLE surfaces RENAME TO patches; ALTER TABLE patches DROP COLUMN area;"
                    "CREATE TABLE sync_baselines(seq INTEGER PRIMARY KEY AUTOINCREMENT, peer TEXT NOT NULL, token TEXT NOT NULL,"
                    " state TEXT NOT NULL, UNIQUE(peer,token));"
                    "CREATE TABLE sync_baseline_records(peer TEXT NOT NULL, record_uuid TEXT NOT NULL, baseline_seq INTEGER NOT NULL,"
                    " state TEXT NOT NULL, PRIMARY KEY(peer,record_uuid,baseline_seq));"
                    "CREATE TABLE sync_baseline_cursors(baseline_seq INTEGER PRIMARY KEY, journal_seq INTEGER NOT NULL, geometry_seq INTEGER NOT NULL);"
                    "PRAGMA user_version=16;");
        const auto refused = directory.path / "refused16.vcobj";
        mustFail([&] { (void)migration::convertStore(source, refused); }); // a surface
        require(!std::filesystem::exists(refused) && !std::filesystem::exists(refused.string() + ".converting"), "nothing left");
        sql(source, "DELETE FROM patches; UPDATE objects SET kind='test_surface' WHERE kind='surface';"
                    "DELETE FROM history_dirty_objects; DELETE FROM history_dirty_edges;"
                    "INSERT INTO sync_baselines(peer,token,state) VALUES('p','t','{}');");
        mustFail([&] { (void)migration::convertStore(source, refused); }); // unsettled baselines
        sql(source, "DELETE FROM sync_baselines;");
        const auto destination = directory.path / "new16.vcobj";
        const auto result = migration::convertStore(source, destination);
        const auto report = Json::parse(result.reportJson);
        require(report.at("source_schema") == 16 && report.at("validation") == "ok" && report.contains("upgraded_from_schema_16") &&
                scalar(destination, "SELECT count(*) FROM sqlite_master WHERE name LIKE 'sync_baseline%' OR name='patches'") == 0,
                "schema 16 upgraded");
        // The unedited schema-17 conversion.
        const auto reference = directory.path / "again.vcobj";
        for (const auto* query : {"SELECT count(*) FROM object_versions", "SELECT count(*) FROM relationship_versions",
                                  "SELECT count(*) FROM store_changes", "SELECT count(*) FROM store_commits"})
            require(scalar(destination, query) == scalar(reference, query), "same numbered history as schema 17");
        requireWritable(destination, expected.at("objects").at(1));
    }
    std::cout << "object store convert: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store convert failed: " << error.what() << "\n";
    return 1;
}
