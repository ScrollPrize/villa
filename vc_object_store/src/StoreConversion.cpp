#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/Database.hpp"
#include "internal/Schema.hpp"
#include "internal/History.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/CollectionReads.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/DependencyIndex.hpp"
#include "geometry/SharedGeometry.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "vc_reader/native/platform_io.hpp"
#include <memory>

namespace vc::objects {
namespace {
using namespace detail;
using Db = std::unique_ptr<sqlite3, decltype(&sqlite3_close_v2)>;
std::string quoted(const std::string& identifier)
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
                exec(db,("DROP "+program.type+" "+quoted(program.name)).c_str());
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
        else { Statement check(db,"SELECT * FROM "+quoted(program.name)+" LIMIT 0"); }
    }
    if(!validateTriggers) return;
    for(const auto& owner:owners) {
        Statement kind(db,"SELECT type FROM sqlite_schema WHERE name=?",owner);
        if(!kind.row() || kind.text(0)!="table")
            throw std::runtime_error("maintenance: cannot validate extension trigger on non-table "+owner);
        std::string assignments;
        Statement columns(db,"SELECT name FROM pragma_table_xinfo(?) WHERE hidden=0",owner);
        while(columns.row()) { if(!assignments.empty()) assignments+=","; assignments+=quoted(columns.text(0))+"="+quoted(columns.text(0)); }
        Statement insert(db,"EXPLAIN INSERT INTO "+quoted(owner)+" DEFAULT VALUES");
        Statement update(db,"EXPLAIN UPDATE "+quoted(owner)+" SET "+assignments+" WHERE 0");
        Statement erase(db,"EXPLAIN DELETE FROM "+quoted(owner)+" WHERE 0");
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
}

ConversionReport convertV1ToV2(const std::filesystem::path& input)
{
    using namespace detail;
    const auto path = std::filesystem::absolute(input);
    // Switching out of WAL requires other readers/writers to leave. Exclusive
    // locking mode retains the lock after COMMIT, allowing backup on this same
    // connection without a write transaction or a race before replacement.
    auto source=exclusive(path,17);
    Sibling staged{path.parent_path() / (path.filename().string() + ".v2-" + util::newUuid())};
    auto target=backupTo(source.get(),staged.path);
    exec(target.get(), "PRAGMA synchronous=FULL; PRAGMA foreign_keys=OFF; BEGIN IMMEDIATE;");
    createProvisioning(target.get(), true, true);
    for (const auto* name : {"objects","edges"}) {
        Statement ids(target.get(),"SELECT coalesce(max(id),0) FROM "+quoted(name)); ids.row();
        Statement owner(target.get(),"SELECT uuid FROM store_identity"); owner.row();
        execute(target.get(),"INSERT INTO key_allocators VALUES(?,?,1,?,?)",std::string(name),owner.blob16(0),int64_t(INT64_MAX),
                ids.i64(0)==INT64_MAX ? std::optional<int64_t>{} : std::optional<int64_t>{ids.i64(0)+1});
    }
    // Legacy publications have no store-global generation. Zero explicitly
    // means pre-v2, not a fabricated ordering between independent old counters.
    exec(target.get(), "ALTER TABLE object_revisions ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;"
                       "ALTER TABLE relationship_revisions ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;");
    exec(target.get(), "ALTER TABLE edges ADD COLUMN relation_key TEXT NOT NULL DEFAULT '';");
    createEdgeIdentityIndex(target.get());
    {
        // SQLite cannot relax a column's NOT NULL in place. Preserve rowids,
        // blobs, and extension indexes/triggers while rebuilding this one table
        // through the same constructor as newly created stores.
        Statement columns(target.get(),"SELECT name FROM pragma_table_xinfo('geometry_chunks') ORDER BY cid");
        std::vector<std::string> names;
        while(columns.row()) names.push_back(columns.text(0));
        if(names!=std::vector<std::string>{"hash","data"})
            throw std::runtime_error("conversion: custom geometry_chunks columns require an explicit migration");
        std::vector<std::string> definitions;
        Statement schema(target.get(),"SELECT sql FROM sqlite_schema WHERE tbl_name='geometry_chunks' "
            "AND type='index' AND sql IS NOT NULL ORDER BY rowid");
        while(schema.row()) definitions.push_back(schema.text(0));
        const auto programs=schemaPrograms(target.get());
        // SQLite validates all views and triggers on ALTER TABLE. Suspend their
        // definitions while the old table name is absent, including indirect
        // references and triggers owned by other tables. No SQL-text guessing.
        suspendPrograms(target.get(),programs,true);
        createGeometryChunkTable(target.get(),"_v2_geometry_chunks");
        exec(target.get(),"INSERT INTO _v2_geometry_chunks(rowid,hash,data) SELECT rowid,hash,data FROM geometry_chunks;"
                         "DROP TABLE geometry_chunks; ALTER TABLE _v2_geometry_chunks RENAME TO geometry_chunks;");
        for(const auto& sql:definitions) exec(target.get(),sql.c_str());
        // Unlike history removal, this migration removes no columns or tables
        // referenced by the unchanged programs. Keep even INSTEAD OF triggers
        // on views, whose unsupported *other* write operations need not compile.
        restorePrograms(target.get(),programs,true,false);
        createGeometryChunkGuards(target.get());
    }
    Statement high(target.get(), "SELECT max(g) FROM (SELECT coalesce(max(generation),0) g FROM objects UNION ALL "
        "SELECT coalesce(max(generation),0) FROM edges UNION ALL SELECT coalesce(max(object_generation),0) FROM geometry_revisions)");
    high.row();
    if (high.i64(0) == INT64_MAX) throw std::runtime_error("conversion: generation exhausted");
    execute(target.get(), "UPDATE store_manifest SET generation=?", std::max<int64_t>(1, high.i64(0) + 1));
    createStorageDirectory(target.get());
    execute(target.get(), "UPDATE database_identity SET schema_version=18");
    registerStorageTables(target.get()); createChangeTracking(target.get());
    exec(target.get(), "PRAGMA user_version=18; COMMIT; PRAGMA foreign_keys=ON;");
    integrity(target.get());
    target.reset();

    // Compare the original column sets, including rowid when present. This
    // checks payload bytes and row multiplicity, not just aggregate counts.
    execute(source.get(), "ATTACH DATABASE ? AS converted", staged.path.string());
    ConversionReport report;
    Statement tables(source.get(), "SELECT name,wr FROM pragma_table_list WHERE schema='main' AND type<>'view' AND name<>'sqlite_schema' ORDER BY name");
    while (tables.row()) {
        const auto name = tables.text(0);
        if (name == "sqlite_temp_schema") continue;
        Statement columns(source.get(), "SELECT name FROM pragma_table_xinfo(?) ORDER BY cid", name);
        std::string projection = tables.i64(1) ? "" : "rowid";
        while (columns.row()) {
            if (!projection.empty()) projection += ",";
            projection += quoted(columns.text(0));
        }
        std::string order = "rowid";
        if (tables.i64(1)) {
            order.clear();
            Statement keys(source.get(), "SELECT name FROM pragma_table_info(?) WHERE pk>0 ORDER BY pk", name);
            while (keys.row()) { if (!order.empty()) order += ","; order += quoted(keys.text(0)); }
        }
        // Both ordered reads use the table's primary B-tree. Do not EXCEPT
        // entire BLOB tables into a temporary sorter during large conversions.
        Statement original(source.get(), "SELECT " + projection + " FROM main." + quoted(name) + " ORDER BY " + order);
        Statement converted(source.get(), "SELECT " + projection + " FROM converted." + quoted(name) + " ORDER BY " + order);
        while (true) {
            const bool a = original.row(), b = converted.row();
            if (a != b) throw std::runtime_error("conversion: row count differs: " + name);
            if (!a) break;
            for (int c = 0; c < sqlite3_column_count(original.value); ++c) {
                const int type = sqlite3_column_type(original.value, c);
                bool equal = type == sqlite3_column_type(converted.value, c);
                if (equal && type == SQLITE_INTEGER) equal = original.i64(c) == converted.i64(c);
                else if (equal && type == SQLITE_FLOAT) equal = sqlite3_column_double(original.value,c) == sqlite3_column_double(converted.value,c);
                else if (equal && type != SQLITE_NULL) {
                    const int n = sqlite3_column_bytes(original.value,c);
                    equal = n == sqlite3_column_bytes(converted.value,c) && (!n ||
                        std::memcmp(sqlite3_column_blob(original.value,c), sqlite3_column_blob(converted.value,c), n) == 0);
                }
                if (!equal) throw std::runtime_error("conversion: table contents differ: " + name);
            }
            ++report.rows;
        }
        ++report.tables;
    }
    // Finalize all source comparison statements before DETACH/publication.
    tables.reset();
    exec(source.get(), "DETACH DATABASE converted;");
    { auto verified = ObjectStore::open(staged.path, OpenOptions{.readOnly=true}); auto read = verified.read();
      if (verified.provisioning(read).formatVersion != 2) throw std::runtime_error("conversion: v2 reopen failed"); }
    report.bytes = std::filesystem::file_size(staged.path);
    vcr::platform::replace_file(staged.path, path, true);
    vcr::platform::sync_directory(path.parent_path(), "vcobj conversion");
    return report;
}

void changeHistoryMode(const std::filesystem::path& input, HistoryMode mode)
{
    using namespace detail;
    const auto path=std::filesystem::absolute(input);
    auto source=exclusive(path,18);
    const bool enabled=mode==HistoryMode::On;
    Statement policy(source.get(),"SELECT history_enabled,sealed,spatial_enabled,generation FROM store_manifest"); policy.row();
    if (policy.i64(1)) throw std::runtime_error("history policy: sealed stores require a new export");
    if ((policy.i64(0)!=0)==enabled) return;
    if (policy.i64(3)==INT64_MAX) throw std::runtime_error("history policy: generation exhausted");
    Statement hub(source.get(),"SELECT 1 FROM sqlite_schema WHERE name IN ('hub_leases','hub_upstream')");
    if (hub.row()) throw std::runtime_error("history policy: hosted/replicated stores require retained history");
    // Removing history cannot leave dangling pins to an old geometry state.
    Statement pins(source.get(),"SELECT 1 FROM edge_geometry_bindings b JOIN edges e ON e.id=b.edge_id "
        "LEFT JOIN geometry_heads a ON a.object_uuid=(SELECT uuid FROM objects WHERE id=e.a_id) "
        "LEFT JOIN geometry_heads z ON z.object_uuid=(SELECT uuid FROM objects WHERE id=e.b_id) "
        "WHERE b.a_revision IS NOT a.revision_uuid OR b.b_revision IS NOT z.revision_uuid LIMIT 1");
    if (!enabled && pins.row()) throw std::runtime_error("history policy: revalidate historical relationship geometry pins before removing history");
    Sibling staged{path.parent_path()/(path.filename().string()+".policy-"+util::newUuid())};
    auto target=backupTo(source.get(),staged.path); auto* db=target.get();
    auto sourceChunks=attachCollectionChunkResolver(source.get(),path);
    auto targetChunks=attachCollectionChunkResolver(db,path);
    // Both templates come from the real schema constructors. The source one
    // identifies library-owned triggers; a substring match must not erase an
    // extension trigger just because it reads current geometry.
    const auto layout = [&](bool history) {
        Db result(openConnection(":memory:",SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,{},false),sqlite3_close_v2);
        createProvisioning(result.get(),history,policy.i64(2)!=0);
        createSchema(result.get(),history);
        if (history) createHistorySchema(result.get());
        createLifecycleSchema(result.get()); createDependencyIndex(result.get()); createSourceIdentityIndex(result.get());
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
    exec(db,"CREATE TABLE _v2_geometry AS SELECT * FROM geometry_revisions WHERE 0;"
            "CREATE TABLE _v2_chunks(revision_uuid BLOB,array_name TEXT,chunk_index INTEGER,hash TEXT);"
            "CREATE TABLE _v2_revision_ids(old_uuid BLOB PRIMARY KEY,new_uuid BLOB NOT NULL UNIQUE);"
            "CREATE TABLE _v2_bounds AS SELECT 0 object_id,NULL revision_uuid,0.0 min_x,0.0 min_y,0.0 min_z,0.0 max_x,0.0 max_y,0.0 max_z WHERE 0;");
    if (policy.i64(2)) exec(db,"INSERT INTO _v2_bounds SELECT * FROM object_bounds;");
    {
        Statement heads(db,"SELECT r.uuid,r.object_uuid,r.object_generation,r.created_by,r.created_at,r.created_host "
                           "FROM geometry_revisions r JOIN geometry_heads h ON h.revision_uuid=r.uuid");
        while (heads.row()) {
            const auto uuid=formatUuid(heads.blob16(0)); GeometryReader reader(db,uuid,false);
            const auto replacement=parseUuid(util::newUuid());
            execute(db,"INSERT INTO _v2_revision_ids VALUES(?,?)",heads.blob16(0),replacement);
            auto manifest=reader.manifest();
            for (auto& [name,array]:manifest.at("arrays").items()) {
                array.erase("source_revision");
                const auto chunks=array.at("chunks").get<std::vector<std::string>>();
                for (size_t i=0;i<chunks.size();++i) {
                    const auto bytes=reader.chunk(name,int64_t(i));
                    const auto hash=storeSharedPayload(db,name,bytes);
                    if (hash!=chunks[i]) throw std::runtime_error("history policy: materialized chunk hash mismatch");
                    execute(db,"INSERT INTO _v2_chunks VALUES(?,?,?,?)",replacement,name,int64_t(i),hash);
                }
            }
            const auto text=manifest.dump();
            execute(db,"INSERT INTO _v2_geometry VALUES(?,?,NULL,?,?,?,?,?,?)",replacement,heads.blob16(1),
                geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()),text.size())),text,
                writeGeneration(db),heads.text(3),heads.text(4),heads.text(5));
        }
    }
    const std::set<std::string> replaced{
        "journal","object_revisions","object_heads","object_head_records","relationship_revisions",
        "object_relationship_deltas","object_relationship_heads","history_dirty_objects","history_dirty_edges",
        "history_expected","history_checked","history_required","history_validated_edges","geometry_revisions",
        "geometry_heads","geometry_revision_chunks","geometry_deltas","geometry_materializations","geometry_current_chunks",
        "current_geometry","current_geometry_chunks","current_shared_payloads","current_edge_metadata",
        "current_changed_objects","current_unused_chunks","object_bounds"};
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
        for (const auto& [kind,name]:dropped) if (kind==type) exec(db,("DROP "+kind+" IF EXISTS "+quoted(name)).c_str());
    {
        Statement schema(desired.get(),"SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'view' THEN 1 WHEN 'index' THEN 2 ELSE 3 END,rowid");
        while (schema.row()) {
            Statement present(db,"SELECT 1 FROM sqlite_schema WHERE name=?",schema.text(1));
            if (!present.row()) exec(db,schema.text(2).c_str());
        }
    }
    execute(db,"UPDATE store_manifest SET history_enabled=?",int64_t(enabled));
    if (const auto pool=activeStatementPools.find(db); pool!=activeStatementPools.end())
        pool->second->historyEnabled=enabled;
    if (enabled) {
        exec(db,"INSERT INTO geometry_revisions SELECT * FROM _v2_geometry;"
                "INSERT INTO geometry_heads SELECT object_uuid,uuid FROM _v2_geometry;"
                "INSERT INTO geometry_revision_chunks SELECT * FROM _v2_chunks;"
                "INSERT INTO history_dirty_objects SELECT uuid FROM objects WHERE 1 ON CONFLICT DO NOTHING;"
                "INSERT INTO history_dirty_edges SELECT uuid FROM edges WHERE 1 ON CONFLICT DO NOTHING;");
    } else {
        exec(db,"INSERT INTO current_geometry SELECT uuid,object_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host FROM _v2_geometry;"
                "INSERT INTO current_geometry_chunks SELECT * FROM _v2_chunks;");
        Statement rows(db,"SELECT object_uuid,manifest FROM current_geometry");
        while(rows.row()) indexCurrentMetadata(db,formatUuid(rows.blob16(0)),nlohmann::json::parse(rows.text(1)));
        exec(db,"INSERT INTO current_unused_chunks SELECT hash FROM geometry_chunks WHERE 1 ON CONFLICT DO NOTHING;");
    }
    exec(db,"UPDATE edge_geometry_bindings SET "
            "a_revision=coalesce((SELECT new_uuid FROM _v2_revision_ids WHERE old_uuid=a_revision),a_revision),"
            "b_revision=coalesce((SELECT new_uuid FROM _v2_revision_ids WHERE old_uuid=b_revision),b_revision);");
    if (policy.i64(2)) exec(db,"UPDATE _v2_bounds SET revision_uuid=(SELECT new_uuid FROM _v2_revision_ids WHERE old_uuid=revision_uuid) "
                            "WHERE revision_uuid IN (SELECT old_uuid FROM _v2_revision_ids);"
                            "DELETE FROM spatial_regions WHERE array_name=''; INSERT INTO object_bounds SELECT * FROM _v2_bounds;");
    execute(db,"UPDATE objects SET data_checksum=(SELECT content_hash FROM geometry_revisions r JOIN geometry_heads h ON h.revision_uuid=r.uuid WHERE h.object_uuid=objects.uuid),"
               "generation=?,data_generation=?,approved_generation=CASE WHEN approved_generation IS NOT NULL THEN ? ELSE NULL END "
               "WHERE uuid IN (SELECT object_uuid FROM geometry_heads)",writeGeneration(db),writeGeneration(db),writeGeneration(db));
    execute(db,"UPDATE edges SET generation=?,approved_generation=CASE WHEN approved_generation IS NOT NULL THEN ? ELSE NULL END "
               "WHERE id IN (SELECT edge_id FROM edge_geometry_bindings WHERE a_revision IN (SELECT new_uuid FROM _v2_revision_ids) "
               "OR b_revision IN (SELECT new_uuid FROM _v2_revision_ids))",writeGeneration(db),writeGeneration(db));
    {
        Statement edges(db,std::string("SELECT ")+kEdgeCols+" FROM edges");
        while (edges.row()) recordRelationshipValidation(db,readEdgeRow(db,edges));
    }
    if (enabled) publishHistory(db); else validateCurrentRelationships(db);
    exec(db,"DROP TABLE _v2_geometry; DROP TABLE _v2_chunks; DROP TABLE _v2_bounds; DROP TABLE _v2_revision_ids;");
    // Recreate compatible extension triggers and prepare their actions without
    // executing them. References to removed history fail on the staged copy,
    // leaving the original intact instead of publishing a broken extension.
    restorePrograms(db,extensions,false);
    registerStorageTables(db);
    exec(db,"UPDATE store_manifest SET generation=generation+1; UPDATE store_write_state SET dirty=0; COMMIT; PRAGMA foreign_keys=ON;");
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
