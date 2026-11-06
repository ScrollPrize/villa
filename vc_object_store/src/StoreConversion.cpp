#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/Database.hpp"
#include "internal/Schema.hpp"
#include "internal/History.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/RecordJson.hpp"
#include "internal/LegacyCurrentSchema.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/Digest.hpp"
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

const std::set<std::string> legacyCurrentTables{
    "geometry_heads","edge_geometry_bindings","edge_lifecycle_bindings","geometry_edge_dependencies","object_lifecycle"};

const std::map<std::string,std::string> legacyDigestColumns{
    {"geometry_chunks","hash"},{"geometry_revisions","content_hash"},
    {"geometry_revision_chunks","hash"},{"geometry_deltas","manifest_hash"},
    {"geometry_current_chunks","hash"}};

void convertLegacyDigests(sqlite3* db)
{
    // Reuse the native schema instead of maintaining a second set of geometry
    // table definitions. All rows retain their rowid and non-digest values.
    Db layout(openConnection(":memory:",SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,{},false),sqlite3_close_v2);
    createProvisioning(layout.get(),true,false);createSchema(layout.get(),true);createHistorySchema(layout.get());
    auto rebuilt=legacyDigestColumns;rebuilt.emplace("edges","");rebuilt.emplace("relationship_revisions","");
    for(const auto& [table,digest]:rebuilt) {
        std::vector<std::string> names,expected;
        Statement columns(db,"SELECT name FROM pragma_table_xinfo(?) ORDER BY cid",table);
        while(columns.row())names.push_back(columns.text(0));
        Statement native(layout.get(),"SELECT name FROM pragma_table_xinfo(?) ORDER BY cid",table);
        while(native.row())expected.push_back(native.text(0));
        if(table=="edges"){
            std::erase(expected,"deleted");
            for(auto& name:expected){if(name=="a_uuid")name="a_id";if(name=="b_uuid")name="b_id";}
        }
        if(names!=expected)throw std::runtime_error("conversion: custom geometry columns require an explicit migration: "+table);
        // An extension key containing textual hashes also needs conversion;
        // do not silently leave it dangling or invent its data semantics.
        Statement tables(db,"SELECT name FROM sqlite_schema WHERE type='table'");
        while(tables.row()) {
            if(legacyDigestColumns.contains(tables.text(0)))continue;
            Statement keys(db,"SELECT \"table\",\"to\" FROM pragma_foreign_key_list(?)",tables.text(0));
            while(keys.row())if(!digest.empty() && keys.text(0)==table && keys.text(1)==digest)
                throw std::runtime_error("conversion: extension digest foreign key requires an explicit migration: "+tables.text(0));
        }
        std::vector<std::string> indexes;
        Statement index(db,"SELECT name,sql FROM sqlite_schema WHERE type='index' AND tbl_name=? AND sql IS NOT NULL ORDER BY rowid",table);
        while(index.row()){
            auto sql=index.text(1);
            if(table=="edges"&&(sql.find("a_id")!=std::string::npos||sql.find("b_id")!=std::string::npos)){
                if(index.text(0)!="edges_a"&&index.text(0)!="edges_b")
                    throw std::runtime_error("conversion: extension index on legacy endpoint IDs requires explicit migration: "+index.text(0));
                auto actual=sql;
                while(!actual.empty()&&(std::isspace(static_cast<unsigned char>(actual.back()))||actual.back()==';'))actual.pop_back();
                const auto expected="CREATE INDEX "+index.text(0)+" ON edges("+(index.text(0)=="edges_a"?std::string("a_id"):std::string("b_id"))+", kind)";
                if(actual!=expected)throw std::runtime_error("conversion: altered native endpoint index requires explicit migration: "+index.text(0));
                Statement replacement(layout.get(),"SELECT sql FROM sqlite_schema WHERE name=?",index.text(0));
                if(!replacement.row())throw std::runtime_error("conversion: missing native endpoint index");
                sql=replacement.text(0);
            }
            indexes.push_back(std::move(sql));
        }
        Statement schema(layout.get(),"SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",table);schema.row();
        const auto definition=schema.text(0);const auto opening=definition.find('(');
        if(opening==std::string::npos)throw std::runtime_error("conversion: invalid native table schema");
        const auto temporary="_v2_digest_"+table;
        exec(db,("CREATE TABLE "+quoteSqlIdentifier(temporary)+definition.substr(opening)).c_str());
        std::string projection="rowid",targetColumns="rowid",parameters="?";int digestIndex=-1;
        for(size_t i=0;i<names.size();++i) {
            projection+=","+quoteSqlIdentifier(names[i]);parameters+=",?";
            const auto mapped=table=="edges"&&names[i]=="a_id"?"a_uuid":table=="edges"&&names[i]=="b_id"?"b_uuid":names[i];
            targetColumns+=","+quoteSqlIdentifier(mapped);
            if(names[i]==digest)digestIndex=int(i)+1;
        }
        Statement source(db,"SELECT "+projection+" FROM "+quoteSqlIdentifier(table)+" ORDER BY rowid");
        Statement target(db,"INSERT INTO "+quoteSqlIdentifier(temporary)+"("+targetColumns+") VALUES("+parameters+")");
        while(source.row()) {
            for(int c=0;c<sqlite3_column_count(source.value);++c) {
                if(c==digestIndex) {
                    if(sqlite3_column_type(source.value,c)!=SQLITE_TEXT)
                        throw std::runtime_error("conversion: legacy digest is not TEXT: "+table);
                    bindDigest(target,c+1,source.text(c));
                } else if(table=="edges"&&c>0&&(names.at(c-1)=="a_id"||names.at(c-1)=="b_id")){
                    Statement endpoint(db,"SELECT uuid FROM objects WHERE id=?",source.i64(c));
                    if(!endpoint.row())throw std::runtime_error("conversion: dangling legacy endpoint");
                    target.bind(c+1,endpoint.blob16(0));
                } else check(sqlite3_bind_value(target.value,c+1,sqlite3_column_value(source.value,c)),db,"copy legacy geometry value");
            }
            target.stepDone();target.reset();
        }
        source.reset();target.reset();
        exec(db,("DROP TABLE "+quoteSqlIdentifier(table)+"; ALTER TABLE "+quoteSqlIdentifier(temporary)+" RENAME TO "+quoteSqlIdentifier(table)).c_str());
        for(const auto& sql:indexes)exec(db,sql.c_str());
    }
}

void validateLegacyCurrentRelationships(sqlite3* db)
{
    const auto currentMetadata=[&](const std::string& uuid,const Anchor& anchor){
        Statement row(db,"SELECT o.meta,coalesce(l.retired,0),coalesce(l.epoch,'') FROM objects o "
            "LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE o.uuid=?",parseUuid(uuid));
        if(!row.row())throw std::runtime_error("conversion: legacy endpoint missing");
        const auto meta=nlohmann::json::parse(row.text(0));
        return dependencyFields(&meta,row.i64(1)!=0,row.text(2),anchor);
    };
    // A historical pin cannot be silently reinterpreted as a current relation.
    // Keep the source intact and require explicit repair/revalidation in the
    // legacy application when any current observation is no longer current.
    Statement pins(db,"SELECT e.id FROM edges e "
        "LEFT JOIN edge_geometry_bindings g ON g.edge_id=e.id "
        "LEFT JOIN edge_lifecycle_bindings l ON l.edge_id=e.id "
        "LEFT JOIN objects a ON a.id=e.a_id LEFT JOIN objects b ON b.id=e.b_id "
        "LEFT JOIN geometry_heads ga ON ga.object_uuid=a.uuid LEFT JOIN geometry_heads gb ON gb.object_uuid=b.uuid "
        "LEFT JOIN object_lifecycle la ON la.object_id=a.id LEFT JOIN object_lifecycle lb ON lb.object_id=b.id "
        "WHERE g.edge_id IS NULL OR g.a_revision IS NOT ga.revision_uuid OR g.b_revision IS NOT gb.revision_uuid "
        "OR coalesce(la.retired,0)<>0 OR coalesce(lb.retired,0)<>0 "
        "OR coalesce(l.a_epoch,'')<>coalesce(la.epoch,'') OR coalesce(l.b_epoch,'')<>coalesce(lb.epoch,'') LIMIT 1");
    if(pins.row())throw std::runtime_error("conversion: legacy current relationship has stale geometry/lifecycle pins; revalidate or remove it before conversion");
    Statement edges(db,"SELECT uuid FROM edges WHERE coalesce(json_array_length(a_dependencies,'$.metadata_keys'),0)>0 "
        "OR coalesce(json_array_length(b_dependencies,'$.metadata_keys'),0)>0");
    while(edges.row()) {
        Statement relations(db,"SELECT DISTINCT r.record,r.a_revision,r.b_revision FROM relationship_revisions r "
            "JOIN object_relationship_heads h ON h.relationship_uuid=r.uuid WHERE h.edge_uuid=?",formatUuid(edges.blob16(0)));
        bool found=false;
        while(relations.row()) {
            found=true;const auto edge=edgeFromJson(nlohmann::json::parse(relations.text(0)));
            if(historicalDependencyMetadata(db,relations.text(1),edge.aAnchor)!=currentMetadata(edge.aUuid,edge.aAnchor) ||
               historicalDependencyMetadata(db,relations.text(2),edge.bAnchor)!=currentMetadata(edge.bUuid,edge.bAnchor))
                throw std::runtime_error("conversion: legacy current relationship has stale endpoint metadata; revalidate or remove it before conversion");
        }
        if(!found)throw std::runtime_error("conversion: legacy metadata-dependent relationship has no published dependency snapshot");
    }
}

std::vector<SchemaProgram> retainedLegacyPrograms(sqlite3* db)
{
    const auto nativeTriggers=legacyCurrentTriggers();
    const auto endpointTriggers=legacyEndpointHistoryTriggers();
    Db nativeLayout(openConnection(":memory:",SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,{},false),sqlite3_close_v2);
    createProvisioning(nativeLayout.get(),true,false);createSchema(nativeLayout.get(),true);createHistorySchema(nativeLayout.get());
    const std::map<std::string,std::vector<std::string>> expected{
        {"geometry_heads",{"object_uuid","revision_uuid"}},
        {"edge_geometry_bindings",{"edge_id","a_revision","b_revision"}},
        {"edge_lifecycle_bindings",{"edge_id","a_epoch","b_epoch"}},
        {"object_lifecycle",{"object_id","retired","epoch"}},
        {"geometry_edge_dependencies",{"edge_id","object_id","array_name","point_id"}}};
    for(const auto& [table,columns]:expected) {
        Statement schema(db,"SELECT name FROM pragma_table_xinfo(?) ORDER BY cid",table);
        std::vector<std::string> actual;while(schema.row())actual.push_back(schema.text(0));
        if(actual.empty()&&table=="geometry_edge_dependencies")continue;
        if(actual!=columns)throw std::runtime_error("conversion: custom legacy bookkeeping columns require an explicit migration: "+table);
        Statement indexes(db,"SELECT name FROM sqlite_schema WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",table);
        while(indexes.row())if(!((table=="geometry_edge_dependencies"&&indexes.text(0)=="geometry_dependencies_edge")||
                                (table=="object_lifecycle"&&indexes.text(0)=="retired_objects")))
            throw std::runtime_error("conversion: custom index on removed bookkeeping table requires an explicit migration: "+indexes.text(0));
    }
    std::vector<SchemaProgram> result;
    for(const auto& program:schemaPrograms(db)) {
        if(const auto native=endpointTriggers.find(program.name);native!=endpointTriggers.end()) {
            auto sql=program.sql;
            while(!sql.empty()&&(std::isspace(static_cast<unsigned char>(sql.back()))||sql.back()==';'))sql.pop_back();
            if(program.type!="trigger"||sql!=native->second)
                throw std::runtime_error("conversion: altered native endpoint history trigger requires an explicit migration: "+program.name);
            Statement replacement(nativeLayout.get(),"SELECT sql FROM sqlite_schema WHERE type='trigger' AND name=?",program.name);
            if(!replacement.row())throw std::runtime_error("conversion: missing native endpoint history trigger: "+program.name);
            result.push_back({program.type,program.name,program.owner,replacement.text(0)});
            continue;
        }
        if(const auto native=nativeTriggers.find(program.name);native!=nativeTriggers.end()) {
            // SQLite omits the final statement terminator in sqlite_schema.
            // Do not normalize the body: a same-name replacement is an
            // extension, not permission to discard arbitrary behavior.
            auto sql=program.sql;
            while(!sql.empty()&&(std::isspace(static_cast<unsigned char>(sql.back()))||sql.back()==';'))sql.pop_back();
            if(program.type!="trigger"||sql!=native->second)
                throw std::runtime_error("conversion: altered native bookkeeping trigger requires an explicit migration: "+program.name);
            continue;
        }
        if(legacyCurrentTables.contains(program.owner))
            throw std::runtime_error("conversion: extension trigger on removed bookkeeping table requires an explicit migration: "+program.name);
        auto lower=program.sql;
        std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return char(std::tolower(c));});
        for(const auto& table:legacyCurrentTables)if(lower.find(table)!=std::string::npos)
            throw std::runtime_error("conversion: extension references removed current-state bookkeeping: "+program.name);
        // An extension using endpoint row IDs needs a deliberate semantic
        // migration. Reject conservatively rather than guessing SQL aliases.
        if(lower.find("a_id")!=std::string::npos||lower.find("b_id")!=std::string::npos)
            throw std::runtime_error("conversion: extension references legacy endpoint IDs: "+program.name);
        result.push_back(program);
    }
    Statement tables(db,"SELECT name FROM sqlite_schema WHERE type='table'");
    while(tables.row()) {
        if(legacyCurrentTables.contains(tables.text(0)))continue;
        Statement keys(db,"SELECT \"table\" FROM pragma_foreign_key_list(?)",tables.text(0));
        while(keys.row())if(legacyCurrentTables.contains(keys.text(0)))
            throw std::runtime_error("conversion: extension foreign key references removed current-state bookkeeping: "+tables.text(0));
    }
    return result;
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
    validateLegacyCurrentRelationships(source.get());
    Sibling staged{path.parent_path() / (path.filename().string() + ".v2-" + util::newUuid())};
    auto target=backupTo(source.get(),staged.path);
    exec(target.get(), "PRAGMA synchronous=FULL; PRAGMA foreign_keys=OFF; BEGIN IMMEDIATE;");
    createProvisioning(target.get(), true, true);
    exec(target.get(),"INSERT INTO store_required_features VALUES('direct-geometry-v1'),('binary-sha256-v1'),('direct-endpoint-keys-v1'),('inline-lifecycle-v1');");
    for (const auto* name : {"objects","edges"}) {
        Statement ids(target.get(),"SELECT coalesce(max(id),0) FROM "+quoteSqlIdentifier(name)); ids.row();
        Statement owner(target.get(),"SELECT uuid FROM store_identity"); owner.row();
        execute(target.get(),"INSERT INTO key_allocators VALUES(?,?,1,?,?)",std::string(name),owner.blob16(0),int64_t(INT64_MAX),
                ids.i64(0)==INT64_MAX ? std::optional<int64_t>{} : std::optional<int64_t>{ids.i64(0)+1});
    }
    // Legacy publications have no store-global generation. Zero explicitly
    // means pre-v2, not a fabricated ordering between independent old counters.
    exec(target.get(), "ALTER TABLE object_revisions ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;"
                       "ALTER TABLE relationship_revisions ADD COLUMN generation INTEGER NOT NULL DEFAULT 0;"
                       "ALTER TABLE objects ADD COLUMN geometry_manifest TEXT;"
                       "ALTER TABLE objects ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0,1));"
                       "ALTER TABLE objects ADD COLUMN retired INTEGER NOT NULL DEFAULT 0 CHECK(retired IN (0,1));"
                       "ALTER TABLE objects ADD COLUMN lifecycle_epoch TEXT NOT NULL DEFAULT '';");
    exec(target.get(),"DROP INDEX IF EXISTS edges_unique;");
    {
        const auto programs=schemaPrograms(target.get());
        const auto retained=retainedLegacyPrograms(target.get());
        // SQLite validates all views and triggers on ALTER TABLE. Suspend their
        // definitions while the old table name is absent, including indirect
        // references and triggers owned by other tables. No SQL-text guessing.
        suspendPrograms(target.get(),programs,true);
        // This is representation migration, not an object edit. Populate the
        // inline lifecycle only while history/extension triggers are suspended.
        exec(target.get(),"UPDATE objects SET retired=coalesce((SELECT retired FROM object_lifecycle WHERE object_id=objects.id),0),"
            "lifecycle_epoch=coalesce((SELECT epoch FROM object_lifecycle WHERE object_id=objects.id),'');");
        convertLegacyDigests(target.get());
        // Recognized native endpoint history programs are replaced above;
        // unaffected extensions retain their exact SQL. Keep INSTEAD OF triggers
        // on views, whose unsupported *other* write operations need not compile.
        Statement heads(target.get(),"SELECT object_uuid,revision_uuid FROM geometry_heads");
        while(heads.row()) {
            const auto token=formatUuid(heads.blob16(1));
            GeometryReader reader(target.get(),token,false);
            execute(target.get(),"UPDATE objects SET storage_ref=?,geometry_manifest=? WHERE uuid=?",
                token,reader.manifest().dump(),heads.blob16(0));
        }
        heads.reset();
        for(const auto& table:legacyCurrentTables)exec(target.get(),("DROP TABLE IF EXISTS "+quoteSqlIdentifier(table)).c_str());
        restorePrograms(target.get(),retained,true,false);
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
        if (name == "sqlite_temp_schema" || (legacyCurrentTables.contains(name)&&name!="object_lifecycle")) continue;
        Statement columns(source.get(), "SELECT name FROM pragma_table_xinfo(?) ORDER BY cid", name);
        std::string projection = tables.i64(1) ? "" : "rowid";
        std::vector<std::string> columnNames;
        if(!tables.i64(1))columnNames.push_back("rowid");
        while (columns.row()) {
            if (!projection.empty()) projection += ",";
            projection += quoteSqlIdentifier(columns.text(0));
            columnNames.push_back(columns.text(0));
        }
        std::string order = "rowid";
        if (tables.i64(1)) {
            order.clear();
            Statement keys(source.get(), "SELECT name FROM pragma_table_info(?) WHERE pk>0 ORDER BY pk", name);
            while (keys.row()) { if (!order.empty()) order += ","; order += quoteSqlIdentifier(keys.text(0)); }
        }
        // Both ordered reads use the table's primary B-tree. Do not EXCEPT
        // entire BLOB tables into a temporary sorter during large conversions.
        Statement original(source.get(), "SELECT " + projection + " FROM main." + quoteSqlIdentifier(name) + " ORDER BY " + order);
        std::string convertedProjection=projection;
        if(name=="edges"){
            convertedProjection.clear();
            for(const auto& column:columnNames){
                if(!convertedProjection.empty())convertedProjection+=",";
                if(column=="a_id"||column=="b_id")convertedProjection+="(SELECT id FROM main.objects WHERE uuid="+
                    std::string(column=="a_id"?"a_uuid":"b_uuid")+")";
                else convertedProjection+=quoteSqlIdentifier(column);
            }
        }
        const auto convertedQuery=name=="object_lifecycle"
            ? "SELECT id,id,retired,lifecycle_epoch FROM converted.objects WHERE id IN (SELECT object_id FROM main.object_lifecycle) ORDER BY id"
            : "SELECT " + convertedProjection + " FROM converted." + quoteSqlIdentifier(name) + " ORDER BY " + order;
        Statement converted(source.get(), convertedQuery);
        while (true) {
            const bool a = original.row(), b = converted.row();
            if (a != b) throw std::runtime_error("conversion: row count differs: " + name);
            if (!a) break;
            for (int c = 0; c < sqlite3_column_count(original.value); ++c) {
                if(name=="objects"&&columnNames.at(c)=="storage_ref") {
                    // Managed geometry now points directly from the current
                    // record. Check the derived value, not the old empty slot.
                    const auto uuidColumn=std::find(columnNames.begin(),columnNames.end(),"uuid");
                    if(uuidColumn==columnNames.end())throw std::runtime_error("conversion: object UUID column missing");
                    Statement head(source.get(),"SELECT revision_uuid FROM geometry_heads WHERE object_uuid=?",
                        original.blob16(int(uuidColumn-columnNames.begin())));
                    if(head.row()) {
                        if(converted.text(c)!=formatUuid(head.blob16(0)))throw std::runtime_error("conversion: current geometry pointer differs");
                        continue;
                    }
                }
                if(const auto digest=legacyDigestColumns.find(name);digest!=legacyDigestColumns.end() && columnNames.at(c)==digest->second) {
                    if(sqlite3_column_type(original.value,c)!=SQLITE_TEXT ||
                       formatDigest(parseDigest(original.text(c)))!=readDigest(converted,c))
                        throw std::runtime_error("conversion: digest differs: "+name);
                    continue;
                }
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
    for(const auto* name:{"direct-geometry-v1","binary-sha256-v1","direct-endpoint-keys-v1","inline-lifecycle-v1"}) {
        Statement feature(source.get(),"SELECT 1 FROM store_required_features WHERE name=?",std::string(name));
        if(!feature.row())throw std::runtime_error("history policy: legacy current-state layout requires separate conversion");
    }
    const bool enabled=mode==HistoryMode::On;
    Statement policy(source.get(),"SELECT history_enabled,sealed,spatial_enabled,generation FROM store_manifest"); policy.row();
    if (policy.i64(1)) throw std::runtime_error("history policy: sealed stores require a new export");
    if ((policy.i64(0)!=0)==enabled) return;
    if (policy.i64(3)==INT64_MAX) throw std::runtime_error("history policy: generation exhausted");
    Statement hub(source.get(),"SELECT 1 FROM sqlite_schema WHERE name IN ('hub_leases','hub_upstream')");
    if (hub.row()) throw std::runtime_error("history policy: hosted/replicated stores require retained history");
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
    exec(db,"CREATE TABLE _v2_geometry(uuid,object_uuid,parent_uuid,content_hash,manifest,object_generation,created_by,created_at,created_host);"
            "CREATE TABLE _v2_chunks(object_uuid BLOB,revision_uuid BLOB,array_name TEXT,chunk_index INTEGER,hash BLOB);"
            "CREATE TABLE _v2_revision_ids(old_uuid BLOB PRIMARY KEY,new_uuid BLOB NOT NULL UNIQUE);"
            "CREATE TABLE _v2_bounds AS SELECT 0 object_id,NULL revision_uuid,0.0 min_x,0.0 min_y,0.0 min_z,0.0 max_x,0.0 max_y,0.0 max_z WHERE 0;");
    if (policy.i64(2)) exec(db,"INSERT INTO _v2_bounds SELECT * FROM object_bounds;");
    {
        Statement heads(db,"SELECT storage_ref,uuid,data_generation,changed_by,changed_at,changed_host "
                           "FROM objects WHERE geometry_manifest IS NOT NULL");
        while (heads.row()) {
            const auto uuid=readGeometryTokenColumn(heads,0); GeometryReader reader(db,uuid,false);
            const auto object=formatUuid(heads.blob16(1));
            const auto replacement=enabled?util::newUuid():currentGeometryToken(object,writeGeneration(db));
            {Statement mapping(db,"INSERT INTO _v2_revision_ids VALUES(?,?)");
                bindGeometryToken(mapping,1,uuid);bindGeometryToken(mapping,2,replacement);mapping.stepDone();}
            auto manifest=reader.manifest();
            for (auto& [name,array]:manifest.at("arrays").items()) {
                array.erase("source_revision");
                const auto chunks=array.at("chunks").get<std::vector<std::string>>();
                for (size_t i=0;i<chunks.size();++i) {
                    const auto bytes=reader.chunk(name,int64_t(i));
                    const auto hash=storeSharedPayload(db,name,bytes);
                    if (hash!=chunks[i]) throw std::runtime_error("history policy: materialized chunk hash mismatch");
                    Statement chunk(db,"INSERT INTO _v2_chunks VALUES(?,?,?,?,?)");
                    chunk.bind(1,heads.blob16(1));bindGeometryToken(chunk,2,replacement);
                    chunk.bind(3,name);chunk.bind(4,int64_t(i));bindDigest(chunk,5,hash);chunk.stepDone();
                }
            }
            const auto text=manifest.dump();
            Statement geometry(db,"INSERT INTO _v2_geometry VALUES(?,?,NULL,?,?,?,?,?,?)");
            bindGeometryToken(geometry,1,replacement);geometry.bind(2,heads.blob16(1));
            bindDigest(geometry,3,geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()),text.size())));
            geometry.bind(4,text);geometry.bind(5,writeGeneration(db));geometry.bind(6,heads.text(3));
            geometry.bind(7,heads.text(4));geometry.bind(8,heads.text(5));geometry.stepDone();
        }
    }
    const std::set<std::string> replaced{
        "journal","object_revisions","object_heads","object_head_records","relationship_revisions",
        "object_relationship_deltas","object_relationship_heads","history_dirty_objects","history_dirty_edges",
        "history_expected","history_checked","history_required","history_validated_edges","geometry_revisions",
        "geometry_heads","geometry_revision_chunks","geometry_deltas","geometry_materializations","geometry_current_chunks",
"object_bounds"};
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
    execute(db,"UPDATE store_manifest SET history_enabled=?",int64_t(enabled));
    if (const auto pool=activeStatementPools.find(db); pool!=activeStatementPools.end())
        pool->second->historyEnabled=enabled;
    if(!enabled)beginCurrentWrite(db);
    if (enabled) {
        exec(db,"INSERT INTO geometry_revisions SELECT * FROM _v2_geometry;"
                "INSERT INTO geometry_revision_chunks SELECT revision_uuid,array_name,chunk_index,hash FROM _v2_chunks;"
                "INSERT INTO history_dirty_objects SELECT uuid FROM objects WHERE 1 ON CONFLICT DO NOTHING;"
                "INSERT INTO history_dirty_edges SELECT uuid FROM edges WHERE 1 ON CONFLICT DO NOTHING;");
    }
    {
        Statement rows(db,"SELECT object_uuid,uuid,content_hash,manifest,object_generation FROM _v2_geometry");
        while(rows.row())execute(db,"UPDATE objects SET storage_ref=?,data_checksum=?,geometry_manifest=?,data_generation=? WHERE uuid=?",
            readGeometryTokenColumn(rows,1),readDigest(rows,2),rows.text(3),rows.i64(4),rows.blob16(0));
    }
    {Statement sourceIndex(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='source_collections'");
        if(sourceIndex.row())exec(db,"UPDATE source_collections SET revision_uuid=coalesce("
            "(SELECT new_uuid FROM _v2_revision_ids WHERE old_uuid=revision_uuid),revision_uuid);");}
    if (policy.i64(2)) exec(db,"UPDATE _v2_bounds SET revision_uuid=(SELECT new_uuid FROM _v2_revision_ids WHERE old_uuid=revision_uuid) "
                            "WHERE revision_uuid IN (SELECT old_uuid FROM _v2_revision_ids);"
                            "DELETE FROM spatial_regions WHERE array_name=''; INSERT INTO object_bounds SELECT * FROM _v2_bounds;");
    {
        Statement geometries(db,"SELECT object_uuid,content_hash FROM _v2_geometry");
        while(geometries.row())execute(db,"UPDATE objects SET data_checksum=?,generation=?,data_generation=?,"
            "approved_generation=CASE WHEN approved_generation IS NOT NULL THEN ? ELSE NULL END WHERE uuid=?",
            readDigest(geometries,1),writeGeneration(db),writeGeneration(db),writeGeneration(db),geometries.blob16(0));
    }
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
