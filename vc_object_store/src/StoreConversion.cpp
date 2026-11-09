#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/Database.hpp"
#include "internal/Schema.hpp"
#include "internal/History.hpp"
#include "internal/Versions.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/RecordJson.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/Digest.hpp"
#include "internal/ExternalEndpoint.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/DependencyIndex.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/GeometryToken.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "vc_reader/native/platform_io.hpp"
#include <memory>
#include <cctype>

namespace vc::objects {
namespace {
using namespace detail;
using Db = std::unique_ptr<sqlite3, decltype(&sqlite3_close_v2)>;
std::string quoteSqlIdentifier(const std::string& identifier)
{
    std::string result = "\"";
    for (auto c : identifier) { result += c; if (c == '"') result += c; }
    return result + "\"";
}
struct SchemaProgram { std::string type,name,owner,sql; };
std::vector<SchemaProgram> schemaPrograms(sqlite3* db)
{
    std::vector<SchemaProgram> result;
    Statement rows(db,"SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE type IN ('trigger','view') AND sql IS NOT NULL ORDER BY rowid");
    while(rows.row()) result.push_back({rows.text(0),rows.text(1),rows.text(2),rows.text(3)});
    return result;
}
void suspendPrograms(sqlite3* db,const std::vector<SchemaProgram>& programs,bool views)
{
    for(const auto* type:{"trigger","view"})
        for(const auto& program:programs)
            if(program.type==type && (views || program.type=="trigger"))
                exec(db,("DROP "+program.type+" "+quoteSqlIdentifier(program.name)).c_str());
}
void restorePrograms(sqlite3* db,const std::vector<SchemaProgram>& programs,bool views,bool validateTriggers=true)
{
    // Views may depend on one another: install every definition before preparing
    // any of them. Triggers are recreated only after their views exist.
    for(const auto* type:{"view","trigger"})
        for(const auto& program:programs)
            if(program.type==type && (views || program.type=="trigger")) exec(db,program.sql.c_str());
    std::set<std::string> owners;
    for(const auto& program:programs) {
        if(program.type=="trigger") owners.insert(program.owner);
        else { Statement check(db,"SELECT * FROM "+quoteSqlIdentifier(program.name)+" LIMIT 0"); }
    }
    if(!validateTriggers) return;
    for(const auto& owner:owners) {
        Statement kind(db,"SELECT type FROM sqlite_schema WHERE name=?",owner);
        if(!kind.row() || kind.text(0)!="table")
            throw std::runtime_error("maintenance: cannot validate extension trigger on non-table "+owner);
        std::string assignments;
        Statement columns(db,"SELECT name FROM pragma_table_xinfo(?) WHERE hidden=0",owner);
        while(columns.row()) { if(!assignments.empty()) assignments+=","; assignments+=quoteSqlIdentifier(columns.text(0))+"="+quoteSqlIdentifier(columns.text(0)); }
        Statement insert(db,"EXPLAIN INSERT INTO "+quoteSqlIdentifier(owner)+" DEFAULT VALUES");
        Statement update(db,"EXPLAIN UPDATE "+quoteSqlIdentifier(owner)+" SET "+assignments+" WHERE 0");
        Statement erase(db,"EXPLAIN DELETE FROM "+quoteSqlIdentifier(owner)+" WHERE 0");
    }
}
struct Sibling {
    std::filesystem::path path;
    ~Sibling() {
        for (auto suffix : {"", "-journal", "-wal", "-shm"}) {
            std::error_code ec; std::filesystem::remove(path.string() + suffix, ec);
        }
    }
};
void integrity(sqlite3* db)
{
    Statement check(db, "PRAGMA integrity_check");
    if (!check.row() || check.text(0) != "ok" || check.row())
        throw std::runtime_error("conversion: SQLite integrity check failed");
    Statement fk(db, "PRAGMA foreign_key_check");
    if (fk.row()) throw std::runtime_error("conversion: foreign key check failed: " + fk.text(0));
}

Db exclusive(const std::filesystem::path& path, int version)
{
    if (!std::filesystem::is_regular_file(path) || std::filesystem::is_symlink(path))
        throw std::invalid_argument("maintenance: input must be a regular non-symlink SQLite file");
    OpenOptions options; options.busyTimeoutMs=0;
    Db db(openConnection(path,SQLITE_OPEN_READWRITE,options,false),sqlite3_close_v2);
    if (pragmaInt(db.get(),"application_id")!=0x56434f42 || pragmaInt(db.get(),"user_version")!=version)
        throw std::runtime_error("maintenance: unexpected vcobj schema");
    exec(db.get(),"PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=DELETE; BEGIN EXCLUSIVE; COMMIT;");
    integrity(db.get()); return db;
}

Db backupTo(sqlite3* source, const std::filesystem::path& path)
{
    Db target(openConnection(path,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,{},false),sqlite3_close_v2);
    auto* backup=sqlite3_backup_init(target.get(),"main",source,"main");
    if (!backup) fail(target.get(),"maintenance backup");
    const int step=sqlite3_backup_step(backup,-1), finish=sqlite3_backup_finish(backup);
    if (step!=SQLITE_DONE || finish!=SQLITE_OK) fail(target.get(),"maintenance backup failed");
    return target;
}

} // namespace

void changeHistoryMode(const std::filesystem::path& input, HistoryMode mode)
{
    using namespace detail;
    const auto path=std::filesystem::absolute(input);
    OwnershipGuard ownership(path, OwnershipGuard::Mode::Service);
    auto source=exclusive(path,kSchemaVersion);
    const bool enabled=mode==HistoryMode::On;
    Statement policy(source.get(),"SELECT history_enabled,sealed,spatial_enabled,commit_sequence,generation FROM store_manifest"); policy.row();
    if (policy.i64(1)) throw std::runtime_error("history policy: sealed stores require a new export");
    if ((policy.i64(0)!=0)==enabled) return;
    if (policy.i64(3)==INT64_MAX) throw std::runtime_error("history policy: commit sequence exhausted");
    Sibling staged{path.parent_path()/(path.filename().string()+".policy-"+util::newUuid())};
    auto target=backupTo(source.get(),staged.path); auto* db=target.get();
    auto sourceChunks=attachCollectionChunkResolver(source.get(),path);
    auto targetChunks=attachCollectionChunkResolver(db,path);
    const bool hasReceipts = [&] {
        Statement receiptTable(source.get(), "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='transaction_receipts'");
        return receiptTable.row();
    }();
    // Both templates come from the real schema constructors. The source one
    // identifies library-owned triggers; a substring match must not erase an
    // extension trigger just because it reads current geometry.
    const auto layout = [&](bool history) {
        Db result(openConnection(":memory:",SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,{},false),sqlite3_close_v2);
        createProvisioning(result.get(),history,policy.i64(2)!=0);
        createSchema(result.get(),history);
        if (history) {
            createHistorySchema(result.get());
            createVersionSchema(result.get(), policy.uuid(4));
        } else if (hasReceipts) createReceiptSchema(result.get());
        createGeometryHistoryIndex(result.get());
        Statement sourceIndex(source.get(),"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='source_collections'");
        if(sourceIndex.row())createSourceIdentityIndex(result.get());
        createStorageDirectory(result.get()); createChangeTracking(result.get());
        return result;
    };
    auto desired=layout(enabled), previous=layout(!enabled);
    std::set<std::string> coreTriggers;
    { Statement rows(previous.get(),"SELECT name FROM sqlite_schema WHERE type='trigger'");
      while(rows.row()) coreTriggers.insert(rows.text(0)); }
    std::vector<SchemaProgram> extensions;
    for(const auto& program:schemaPrograms(db)) {
        Statement core(previous.get(),"SELECT 1 FROM sqlite_schema WHERE type=? AND name=?",program.type,program.name);
        if (!core.row()) extensions.push_back(program);
    }
    exec(db,"PRAGMA foreign_keys=OFF; PRAGMA synchronous=FULL; BEGIN IMMEDIATE;");
    // Disable extension triggers while rewriting physical representation. They
    // are reinstated and checked before publication, not run as user edits.
    suspendPrograms(db,extensions,false);
    exec(db,"CREATE TABLE _policy_geometry(uuid,object_uuid,parent_uuid,content_hash,manifest,object_version,created_by,created_at,created_host);"
            "CREATE TABLE _policy_chunks(object_uuid BLOB,revision_uuid BLOB,array_name TEXT,chunk_index INTEGER,hash BLOB);"
            "CREATE TABLE _policy_revision_ids(old_uuid BLOB PRIMARY KEY,new_uuid BLOB NOT NULL UNIQUE);"
            "CREATE TABLE _policy_bounds AS SELECT 0 object_id,NULL revision_uuid,0.0 min_x,0.0 min_y,0.0 min_z,0.0 max_x,0.0 max_y,0.0 max_z WHERE 0;");
    if (hasReceipts) exec(db, "CREATE TABLE _policy_receipts AS SELECT * FROM transaction_receipts;");
    if (policy.i64(2)) exec(db,"INSERT INTO _policy_bounds SELECT * FROM object_bounds;");
    {
        Statement heads(db,"SELECT storage_ref,uuid,version,changed_by,changed_at,changed_host "
                           "FROM objects WHERE geometry_manifest IS NOT NULL");
        while (heads.row()) {
            const auto uuid=readGeometryTokenColumn(heads,0); GeometryReader reader(db,uuid,false);
            const auto object=formatUuid(heads.blob16(1));
            const auto replacement=enabled?util::newUuid():currentGeometryToken(object,heads.i64(2));
            {Statement mapping(db,"INSERT INTO _policy_revision_ids VALUES(?,?)");
                bindGeometryToken(mapping,1,uuid);bindGeometryToken(mapping,2,replacement);mapping.stepDone();}
            auto manifest=reader.manifest();
            for (auto& [name,array]:manifest.at("arrays").items()) {
                array.erase("source_revision");
                const auto chunks=array.at("chunks").get<std::vector<std::string>>();
                for (size_t i=0;i<chunks.size();++i) {
                    const auto bytes=reader.chunk(name,int64_t(i));
                    const auto hash=storeSharedPayload(db,name,bytes);
                    if (hash!=chunks[i]) throw std::runtime_error("history policy: materialized chunk hash mismatch");
                    Statement chunk(db,"INSERT INTO _policy_chunks VALUES(?,?,?,?,?)");
                    chunk.bind(1,heads.blob16(1));bindGeometryToken(chunk,2,replacement);
                    chunk.bind(3,name);chunk.bind(4,int64_t(i));bindDigest(chunk,5,hash);chunk.stepDone();
                }
            }
            int64_t metadataIndex = 0;
            for (const auto& hash : reachableGeometryMetadata(db, manifest)) {
                const auto bytes = readSharedPayload(db, "@metadata", hash);
                if (storeSharedPayload(db, "@metadata", bytes) != hash)
                    throw std::runtime_error("history policy: metadata payload hash mismatch");
                Statement chunk(db,"INSERT INTO _policy_chunks VALUES(?,?,?,?,?)");
                chunk.bind(1,heads.blob16(1)); bindGeometryToken(chunk,2,replacement);
                chunk.bind(3,std::string("@metadata")); chunk.bind(4,metadataIndex++); bindDigest(chunk,5,hash); chunk.stepDone();
            }
            const auto text=manifest.dump();
            Statement geometry(db,"INSERT INTO _policy_geometry VALUES(?,?,NULL,?,?,?,?,?,?)");
            bindGeometryToken(geometry,1,replacement);geometry.bind(2,heads.blob16(1));
            bindDigest(geometry,3,geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()),text.size())));
            geometry.bind(4,text);geometry.bind(5,heads.i64(2));geometry.bind(6,heads.text(3));
            geometry.bind(7,heads.text(4));geometry.bind(8,heads.text(5));geometry.stepDone();
        }
    }
    const std::set<std::string> replaced{
        "journal","object_revisions","object_heads","object_head_records","relationship_revisions",
        "object_relationship_deltas","object_relationship_heads","history_dirty_objects","history_dirty_edges",
        "history_expected","history_checked","history_required","history_validated_edges","geometry_revisions",
        "geometry_heads","geometry_revision_chunks","geometry_deltas","geometry_materializations","geometry_current_chunks",
        "object_bounds", "history_generations", "store_commits", "store_changes", "object_versions",
        "relationship_versions", "transaction_receipts"};
    std::vector<std::pair<std::string,std::string>> dropped;
    {
        Statement schema(db,"SELECT type,name,sql FROM sqlite_schema WHERE type IN ('trigger','view','table') AND sql IS NOT NULL");
        while (schema.row()) {
            bool remove=replaced.contains(schema.text(1));
            if (schema.text(0)=="trigger" && coreTriggers.contains(schema.text(1))) {
                for (const auto& table:replaced) remove|=schema.text(2).find(table)!=std::string::npos;
                remove|=schema.text(1).starts_with("geometry_chunks_immutable_");
            }
            if (remove) dropped.emplace_back(schema.text(0),schema.text(1));
        }
    }
    for (const auto* type:{"trigger","view","table"})
        for (const auto& [kind,name]:dropped) if (kind==type) exec(db,("DROP "+kind+" IF EXISTS "+quoteSqlIdentifier(name)).c_str());
    {
        Statement schema(desired.get(),"SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'view' THEN 1 WHEN 'index' THEN 2 ELSE 3 END,rowid");
        while (schema.row()) {
            Statement present(db,"SELECT 1 FROM sqlite_schema WHERE name=?",schema.text(1));
            if (!present.row()) exec(db,schema.text(2).c_str());
        }
    }
    if (hasReceipts) exec(db, "INSERT INTO transaction_receipts SELECT * FROM _policy_receipts; DROP TABLE _policy_receipts;");
    execute(db,"UPDATE store_manifest SET history_enabled=?",int64_t(enabled));
    if (enabled) {
        execute(db, "INSERT INTO history_generations VALUES(?,0,?,?)", policy.blob16(4),
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(),
            std::string("history enabled"));
        // A lineage restart does not republish untouched objects. Preserve
        // their original version identity when seeding retained history.
        exec(db, "INSERT OR IGNORE INTO history_generations "
                 "SELECT DISTINCT coalesce(version_generation,(SELECT initial_generation FROM store_manifest)),"
                 "0,0,'retained current-state lineage; original start unrecorded' FROM objects "
                 "UNION SELECT DISTINCT coalesce(version_generation,(SELECT initial_generation FROM store_manifest)),"
                 "0,0,'retained current-state lineage; original start unrecorded' FROM edges;");
    }
    if (const auto pool=activeStatementPools.find(db); pool!=activeStatementPools.end())
        pool->second->historyEnabled=enabled;
    if(!enabled)beginCurrentWrite(db);
    if (enabled) {
        exec(db,"INSERT INTO geometry_revisions SELECT * FROM _policy_geometry;"
                "INSERT INTO geometry_revision_chunks SELECT revision_uuid,array_name,chunk_index,hash FROM _policy_chunks;"
                "INSERT INTO history_dirty_objects SELECT uuid FROM objects WHERE 1 ON CONFLICT DO NOTHING;"
                "INSERT INTO history_dirty_edges SELECT uuid FROM edges WHERE 1 ON CONFLICT DO NOTHING;");
    }
    {
        Statement rows(db,"SELECT object_uuid,uuid,content_hash,manifest FROM _policy_geometry");
        while(rows.row())execute(db,"UPDATE objects SET storage_ref=?,data_checksum=?,geometry_manifest=? WHERE uuid=?",
            readGeometryTokenColumn(rows,1),readDigest(rows,2),rows.text(3),rows.blob16(0));
    }
    {Statement sourceIndex(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='source_collections'");
        if(sourceIndex.row())exec(db,"UPDATE source_collections SET revision_uuid=coalesce("
            "(SELECT new_uuid FROM _policy_revision_ids WHERE old_uuid=revision_uuid),revision_uuid);");}
    if (policy.i64(2)) exec(db,"UPDATE _policy_bounds SET revision_uuid=(SELECT new_uuid FROM _policy_revision_ids WHERE old_uuid=revision_uuid) "
                            "WHERE revision_uuid IN (SELECT old_uuid FROM _policy_revision_ids);"
                            "DELETE FROM spatial_regions WHERE array_name=''; INSERT INTO object_bounds SELECT * FROM _policy_bounds;");
    {
        Statement edges(db,std::string("SELECT ")+kEdgeCols+" FROM edges WHERE state=0");
        while (edges.row()) recordRelationshipValidation(db,readEdgeRow(db,edges));
    }
    if (enabled) publishHistory(db,true); else validateCurrentRelationships(db);
    if (!enabled) exec(db,"DELETE FROM geometry_chunks WHERE hash NOT IN (SELECT hash FROM _policy_chunks);");
    exec(db,"DROP TABLE _policy_geometry; DROP TABLE _policy_chunks; DROP TABLE _policy_bounds; DROP TABLE _policy_revision_ids;");
    // Recreate compatible extension triggers and prepare their actions without
    // executing them. References to removed history fail on the staged copy,
    // leaving the original intact instead of publishing a broken extension.
    restorePrograms(db,extensions,false);
    registerStorageTables(db);
    exec(db,"UPDATE store_manifest SET commit_sequence=commit_sequence+1,feed_start=commit_sequence+1; "
            "UPDATE store_write_state SET dirty=0; COMMIT; PRAGMA foreign_keys=ON;");
    integrity(db);
    const bool collection=bool(targetChunks);
    targetChunks.reset(); target.reset();
    if (collection) {
        auto verified=ObjectCollection::open(staged.path,OpenOptions{.readOnly=true}); auto read=verified.read();
        if (verified.provisioning(read).history!=mode) throw std::runtime_error("history policy: collection reopen failed");
    } else {
        auto verified=ObjectStore::open(staged.path,OpenOptions{.readOnly=true}); auto read=verified.read();
        if (verified.provisioning(read).history!=mode) throw std::runtime_error("history policy: reopen failed");
    }
    vcr::platform::replace_file(staged.path,path,true);
    vcr::platform::sync_directory(path.parent_path(),"vcobj history policy");
}
}
