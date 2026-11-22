#include "migration/ConvertSchema21.hpp"
#include "internal/Schema.hpp"
#include "internal/History.hpp"
#include "internal/Versions.hpp"
#include "internal/SchemaPrograms.hpp"
#include "internal/CollectionSchema.hpp"
#include "internal/Publication.hpp"
#include "internal/JsonFields.hpp"
#include "internal/Artifacts.hpp"
#include "internal/RecordOverrides.hpp"
#include "geometry/DependencyIndex.hpp"
#include "geometry/GeometryDelta.hpp"
#include "geometry/GeometryToken.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include <algorithm>
#include <functional>
#include <fstream>
#include <map>
#include <memory>

namespace vc::objects::migration {
namespace {
using namespace detail;
using Json=nlohmann::json;
using Db=std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)>;
Db raw(const std::filesystem::path& path,bool write){
    OpenOptions options;options.readOnly=!write;
    return {openConnection(path,write?SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE:SQLITE_OPEN_READONLY,options,!write),sqlite3_close_v2};
}
bool exists(sqlite3* db,const std::string& name){Statement q(db,"SELECT 1 FROM sqlite_schema WHERE name=?",name);return q.row();}
void validateExtensionProgram(sqlite3* db,const SchemaProgram& program){
    struct Check {const std::string& name;bool blocked=false;} checkState{program.name};
    sqlite3_set_authorizer(db,[](void* raw,int action,const char* table,const char* column,const char*,const char* origin){
        auto& state=*static_cast<Check*>(raw);
        if(!origin||state.name!=origin)return SQLITE_OK;
        const std::string_view t=table?table:"",c=column?column:"";
        if(action==SQLITE_READ&&((t=="edges"&&(c=="id"||c=="kind"||c=="approval_state"))||(t=="edge_tags"&&c=="edge_id")))state.blocked=true;
        if((action==SQLITE_INSERT||action==SQLITE_UPDATE||action==SQLITE_DELETE)&&(t=="edges"||t=="edge_tags"))state.blocked=true;
        return SQLITE_OK;
    },&checkState);
    struct Reset {sqlite3* db;~Reset(){sqlite3_set_authorizer(db,nullptr,nullptr);}} reset{db};
    if(program.type=="view"){Statement read(db,"SELECT * FROM "+quoteSqlIdentifier(program.name)+" LIMIT 0");}
    else{
        std::string assignments;Statement fields(db,"SELECT name FROM pragma_table_xinfo(?) WHERE hidden=0",program.owner);
        while(fields.row()){if(!assignments.empty())assignments+=',';assignments+=quoteSqlIdentifier(fields.text(0))+"="+quoteSqlIdentifier(fields.text(0));}
        Statement insert(db,"EXPLAIN INSERT INTO "+quoteSqlIdentifier(program.owner)+" DEFAULT VALUES");
        Statement update(db,"EXPLAIN UPDATE "+quoteSqlIdentifier(program.owner)+" SET "+assignments+" WHERE 0");
        Statement erase(db,"EXPLAIN DELETE FROM "+quoteSqlIdentifier(program.owner)+" WHERE 0");
    }
    if(checkState.blocked)throw std::runtime_error("convert: extension "+program.type+' '+program.name+" depends on changed edge column semantics; explicitly port it before converting");
}
std::vector<std::string> columns(sqlite3* db,const std::string& table){
    std::vector<std::string> out;Statement q(db,"SELECT name FROM pragma_table_xinfo(?) WHERE hidden=0 ORDER BY cid",table);
    while(q.row())out.push_back(q.text(0));return out;
}
std::map<std::string,std::string> tables(sqlite3* db){
    std::map<std::string,std::string> out;
    Statement q(db,"SELECT name,sql FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "AND name NOT IN (SELECT name FROM pragma_table_list WHERE type='shadow') ORDER BY rowid");
    while(q.row())out.emplace(q.text(0),q.text(1));return out;
}
void checkDatabase(sqlite3* db){
    Statement integrity(db,"PRAGMA integrity_check");
    if(!integrity.row()||integrity.text(0)!="ok"||integrity.row())throw std::runtime_error("convert: integrity check failed");
    Statement fk(db,"PRAGMA foreign_key_check");if(fk.row())throw std::runtime_error("convert: foreign key check failed on "+fk.text(0));
}
void copyTable(sqlite3* in,sqlite3* out,const std::string& table,bool legacy,
               const std::map<std::string,std::string>& locations){
    const auto target=columns(out,table),source=columns(in,table);
    std::string selection,fields,marks;
    for(const auto& name:target){
        if(std::find(source.begin(),source.end(),name)==source.end()&&!(table=="edge_tags"&&name=="edge_uuid"))
            throw std::runtime_error("convert: unmapped target column "+table+'.'+name);
        if(!fields.empty()){fields+=',';selection+=',';marks+=',';}
        fields+=quoteSqlIdentifier(name);marks+='?';
        selection+=legacy&&table=="edge_tags"&&name=="edge_uuid"?"(SELECT uuid FROM edges WHERE id=s.edge_id)":"s."+quoteSqlIdentifier(name);
    }
    for(const auto& name:source)if(std::find(target.begin(),target.end(),name)==target.end()&&
        !(table=="edges"&&name=="id")&&!(table=="edge_tags"&&name=="edge_id")&&
        !(legacy&&table=="collection_members"&&(name=="edge_low"||name=="edge_high")))
        throw std::runtime_error("convert: unmapped source column "+table+'.'+name);
    Statement read(in,"SELECT "+selection+" FROM "+quoteSqlIdentifier(table)+" s"+
        (table=="key_allocators"?" WHERE name<>'edges'":table=="collection_routes"?" WHERE key_space<>'edges'":""));
    Statement write(out,"INSERT INTO "+quoteSqlIdentifier(table)+'('+fields+") VALUES("+marks+')');
    while(read.row()){
        write.reset();
        for(size_t c=0;c<target.size();++c){const auto& name=target[c];
            if((table=="store_manifest"||table=="database_identity")&&name=="schema_version")write.bind(int(c+1),int64_t(kSchemaVersion));
            else if(legacy&&table=="edges"&&name=="kind")write.bind(int(c+1),edgeKindCode(read.text(int(c))));
            else if(legacy&&table=="edges"&&name=="approval_state")write.bind(int(c+1),edgeApprovalCode(read.text(int(c))));
            else if(table=="collection_members"&&name=="locator"){
                const auto at=std::find(target.begin(),target.end(),"database_uuid")-target.begin();
                write.bind(int(c+1),locations.at(read.uuid(int(at))));
            }
            else if((table=="objects"||table=="edges")&&(name=="meta"||name=="a_dependencies"||name=="b_dependencies")){
                const auto value=parseObjectJson(read.text(int(c)));if(value.empty())write.bindNull(int(c+1));else write.bind(int(c+1),value.dump());
            }else check(sqlite3_bind_value(write.value,int(c+1),sqlite3_column_value(read.value,int(c))),out,"convert: bind value");
        }
        write.stepDone();
    }
}
Json copyPhysical(sqlite3* source,const std::filesystem::path& destination,
                  const std::filesystem::path& sourceLocation,const std::filesystem::path& finalLocation,
                  const std::map<std::string,std::string>& locations){
    const auto version=pragmaInt(source,"user_version");const bool legacy=version==21;
    if(pragmaInt(source,"application_id")!=0x56434f42||(version!=21&&version!=kSchemaVersion))
        throw std::runtime_error("convert: expected schema 21/current physical database");
    Statement policy(source,"SELECT history_enabled,spatial_enabled FROM store_manifest");
    if(!policy.row())throw std::runtime_error("convert: missing source policy");
    const bool history=policy.i64(0),spatial=policy.i64(1),immutable=[](sqlite3* db){
        Statement q(db,"SELECT 1 FROM store_required_features WHERE name='immutable-initial-v1'");return q.row();
    }(source);
    Statement identity(source,"SELECT uuid FROM store_identity");identity.row();
    Statement lineage(source,"SELECT generation FROM store_manifest");lineage.row();
    const auto sourceTables=tables(source);
    // An unknown extension depending on a removed numeric key needs its own
    // conversion rule, rather than an invented mapping or lost reference.
    for(const auto& [name,sql]:sourceTables)if(name!="edge_tags"){
        Statement refs(source,"SELECT \"to\" FROM pragma_foreign_key_list(?) WHERE \"table\"='edges'",name);
        while(refs.row())if(refs.text(0)=="id")throw std::runtime_error("convert: extension table "+name+" references removed numeric edge IDs");
    }
    auto target=raw(destination,true);auto* db=target.get();
    exec(db,"PRAGMA journal_mode=DELETE;PRAGMA synchronous=OFF;PRAGMA foreign_keys=OFF;BEGIN IMMEDIATE");
    exec(db,"PRAGMA application_id=1447251778"); // 0x56434f42
    createProvisioning(db,history,spatial);
    OpenOptions options;options.edgeAdjacencyIndex=exists(source,"edges_a")&&exists(source,"edges_b");
    if(!legacy)options.edgeColumns=columns(source,"edges");
    createSchema(db,history,identity.uuid(0),immutable,options);
    if(history){createHistorySchema(db);createVersionSchema(db,lineage.uuid(0));}
    if(!immutable)createGeometryHistoryIndex(db);
    if(sourceTables.contains("source_collections"))createSourceIdentityIndex(db);
    createStorageDirectory(db);
    if(!immutable)createChangeTracking(db);
    if(sourceTables.contains("collection_members"))createCollectionTables(db);
    auto corePrograms=schemaPrograms(db);suspendPrograms(db,corePrograms,true);
    // Create every additional table from its original definition. Unknown
    // extension data is copied without reinterpretation, loss or JSON dumps.
    const auto targetTables=tables(db);
    for(const auto& [name,sql]:sourceTables)if(!targetTables.contains(name))exec(db,sql.c_str());
    auto allTarget=tables(db);
    for(const auto& [name,sql]:allTarget)exec(db,("DELETE FROM "+quoteSqlIdentifier(name)).c_str());
    Json counts=Json::object();
    for(const auto& [name,sql]:sourceTables){
        copyTable(source,db,name,legacy,locations);
        Statement q(db,"SELECT count(*) FROM "+quoteSqlIdentifier(name));q.row();counts[name]=q.i64(0);
    }
    if(exists(source,"sqlite_sequence")){
        exec(db,"DELETE FROM sqlite_sequence");copyTable(source,db,"sqlite_sequence",legacy,locations);
    }
    // A table keeps its identity, but its key description follows the actual
    // v22 primary key rather than a removed implementation detail.
    for(const auto* table:{"edges","edge_tags"}){
        Statement key(db,"SELECT uuid FROM table_directory WHERE name=?",std::string(table));
        if(key.row()){
            execute(db,"DELETE FROM table_key_columns WHERE table_uuid=?",key.blob16(0));
            execute(db,"INSERT INTO table_key_columns VALUES(?,1,?)",key.blob16(0),std::string(table)=="edges"?std::string("uuid"):std::string("edge_uuid"));
            if(std::string(table)=="edge_tags")execute(db,"INSERT INTO table_key_columns VALUES(?,2,'tag')",key.blob16(0));
        }
    }
    {Statement current(db,"SELECT storage_ref,geometry_manifest,data_checksum FROM objects WHERE geometry_manifest IS NOT NULL");
        while(current.row())loadGeometryManifest(db,readGeometryTokenColumn(current,0),current.text(1),current.text(2));}
    if(exists(db,"geometry_revisions")){
        Statement archived(db,"SELECT uuid,manifest,content_hash FROM geometry_revisions");
        while(archived.row())loadGeometryManifest(db,archived.uuid(0),archived.text(1),archived.hash(2));
    }
    std::set<std::string> coreNames;for(const auto& p:corePrograms)coreNames.insert(p.name);
    std::vector<SchemaProgram> programs=corePrograms;
    for(const auto& p:schemaPrograms(source))if(!coreNames.contains(p.name)){
        if(legacy)validateExtensionProgram(source,p);programs.push_back(p);
    }
    // Target core indexes are already present; retain extension indexes and
    // reject ones whose old numeric edge dependencies no longer make sense.
    Statement indexes(source,"SELECT name,sql,tbl_name FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL");
    while(indexes.row())if(!(indexes.text(0)=="edges_kind_id"&&indexes.text(2)=="edges")&&!exists(db,indexes.text(0))){
        if(legacy&&(indexes.text(2)=="edges"||indexes.text(2)=="edge_tags"))
            throw std::runtime_error("convert: custom index "+indexes.text(0)+" on a changed core table requires explicit porting");
        exec(db,indexes.text(1).c_str());
    }
    relocateExternalLocators(db,sourceLocation,finalLocation);
    restorePrograms(db,programs,true);
    execute(db,"UPDATE database_identity SET schema_version=?",int64_t(kSchemaVersion));
    exec(db,("PRAGMA user_version="+std::to_string(kSchemaVersion)).c_str());
    exec(db,"COMMIT;PRAGMA foreign_keys=ON");checkDatabase(db);
    exec(db,"PRAGMA synchronous=FULL");
    // A no-op transaction need not issue a sync. Ask the owning SQLite VFS
    // explicitly so plain sibling publication is durable; packed finalize()
    // additionally flushes and synchronizes the complete container.
    sqlite3_file* file=nullptr;
    check(sqlite3_file_control(db,"main",SQLITE_FCNTL_FILE_POINTER,&file),db,"convert: database file handle");
    if(!file||!file->pMethods)throw std::runtime_error("convert: missing database file handle");
    check(file->pMethods->xSync(file,SQLITE_SYNC_FULL),db,"convert: sync database");
    return counts;
}
size_t copyPhysicalArtifacts(const std::filesystem::path& source,const std::filesystem::path& target,
                             const std::filesystem::path& destinationArtifacts){
    auto store=ObjectStore::open(target,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=store.read();
    auto* db=ObjectStoreImplAccess::conn(r);std::map<std::string,ArtifactFile> files;
    const auto add=[&](const ObjectRecord& object){for(const auto& file:artifactManifest(object)){
        auto [found,inserted]=files.emplace(file.sha256,file);
        if(!inserted&&found->second.length!=file.length)throw std::runtime_error("convert: inconsistent artifact length");
    }};
    Statement current(db,"SELECT meta FROM objects WHERE state<>2");
    while(current.row()){ObjectRecord object;object.meta=canonicalObjectJson(current.text(0));add(object);}
    if(store.provisioning(r).history==HistoryMode::On){
        Statement objects(db,"SELECT DISTINCT object_uuid FROM object_versions");
        while(objects.row())for(const auto& revision:store.objectHistory(r,objects.uuid(0)))add(revision.record);
    }
    for(const auto& [hash,file]:files){
        const auto input=artifactBlobPath(source,hash),output=artifactBlobPath(destinationArtifacts,hash);
        ArtifactFileRef ref{file.name,file.sha256,file.length};
        if(artifactFileMatches(ref,output))continue;
        if(std::filesystem::file_size(input)!=file.length)throw std::runtime_error("convert: artifact length mismatch: "+input.string());
        std::ifstream reader(input,std::ios::binary);if(!reader)throw std::runtime_error("convert: cannot read artifact: "+input.string());
        std::filesystem::create_directories(output.parent_path());
        writeArtifactFile(ref,output,1024*1024,[&](uint64_t offset,uint64_t count){
            std::vector<uint8_t> bytes(static_cast<size_t>(count));reader.seekg(std::streamoff(offset));
            reader.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(count));
            bytes.resize(size_t(reader.gcount()));return bytes;
        });
    }
    return files.size();
}
struct SourceNode {
    std::filesystem::path source,target;
    Db owned{nullptr,sqlite3_close_v2};sqlite3* db=nullptr;
    std::string uuid,store,generation;int64_t sequence=0;bool sealed=false;
    std::map<std::string,std::string> locations;
};
std::filesystem::path sourceLocator(const std::filesystem::path& input){
    if(isPackedDatabasePath(input))return input;
    auto path=std::filesystem::absolute(input).lexically_normal();
    if(std::filesystem::is_directory(path))path/="collection.vcobj";
    return isPackedStore(path)?packedDatabasePath(path):path;
}
std::filesystem::path referencedLocator(const std::filesystem::path& context,const std::string& text){
    if(text.starts_with("@packed:"))return resolvePackedDatabasePath(context,text);
    if(isPackedDatabasePath(text))throw std::runtime_error("convert: nonportable packed source reference");
    const auto parent=isPackedDatabasePath(context)?packedContainerPath(context).parent_path():context.parent_path();
    return sourceLocator(std::filesystem::path(text).is_absolute()?std::filesystem::path(text):parent/text);
}
}

bool isSchema21Store(const std::filesystem::path& input){
    auto locator=sourceLocator(input);
    auto db=raw(locator,false);return pragmaInt(db.get(),"application_id")==0x56434f42&&pragmaInt(db.get(),"user_version")==21;
}
ConversionResult convertSchema21(const std::filesystem::path& source,const std::filesystem::path& destination,sqlite3* lockedSource,
                                 const std::filesystem::path& memberDirectory){
    if(std::filesystem::exists(destination)||std::filesystem::is_symlink(destination))throw std::runtime_error("convert: destination exists");
    const auto initial=sourceLocator(source);const bool packed=isPackedDatabasePath(initial);
    std::vector<std::unique_ptr<SourceNode>> nodes;std::map<std::string,size_t> byUuid;
    std::set<std::string> visiting;std::vector<size_t> order;
    std::function<size_t(const std::filesystem::path&,sqlite3*)> discover=[&](const auto& path,sqlite3* borrowed){
        auto node=std::make_unique<SourceNode>();node->source=path;
        if(!borrowed)node->owned=raw(path,false);node->db=borrowed?borrowed:node->owned.get();
        auto* db=node->db;
        const auto schema=pragmaInt(db,"user_version");
        if(pragmaInt(db,"application_id")!=0x56434f42||(schema!=21&&schema!=kSchemaVersion))
            throw std::runtime_error("convert: linked closure contains an unsupported schema");
        exec(db,"BEGIN");
        Statement id(db,"SELECT uuid FROM database_identity");if(!id.row())throw std::runtime_error("convert: missing database identity");node->uuid=id.uuid(0);
        Statement pin(db,"SELECT s.uuid,m.generation,m.commit_sequence,m.sealed FROM store_identity s CROSS JOIN store_manifest m");
        if(!pin.row())throw std::runtime_error("convert: missing source state");
        node->store=pin.uuid(0);node->generation=pin.uuid(1);node->sequence=pin.i64(2);node->sealed=pin.i64(3);
        if(auto found=byUuid.find(node->uuid);found!=byUuid.end()){
            if(visiting.contains(node->uuid))throw std::runtime_error("convert: cyclic store references");
            const auto& earlier=*nodes[found->second];
            if(earlier.store!=node->store||earlier.generation!=node->generation||earlier.sequence!=node->sequence)
                throw std::runtime_error("convert: conflicting physical database identity");
            return found->second;
        }
        const auto n=nodes.size();byUuid.emplace(node->uuid,n);visiting.insert(node->uuid);nodes.push_back(std::move(node));
        if(exists(db,"collection_members")){
            Statement refs(db,"SELECT database_uuid,store_uuid,generation,commit_sequence,locator FROM collection_members ORDER BY ordinal");
            while(refs.row()){
                auto child=discover(referencedLocator(path,refs.text(4)),nullptr);const auto& dependency=*nodes[child];
                if(dependency.uuid!=refs.uuid(0)||dependency.store!=refs.uuid(1)||dependency.generation!=refs.uuid(2)||
                   dependency.sequence!=refs.i64(3)||!dependency.sealed)
                    throw std::runtime_error("convert: referenced source snapshot does not match its pin");
            }
        }
        visiting.erase(nodes[n]->uuid);order.push_back(n);return n;
    };
    const auto root=discover(initial,lockedSource);
    // Preserve also physically packed members not reachable from the root.
    if(packed)for(const auto& member:packedDatabases(packedContainerPath(initial)))
        if(!byUuid.contains(member.uuid))discover(packedDatabasePath(packedContainerPath(initial),member.uuid),nullptr);
    struct OwnedDirectory {
        std::filesystem::path path;bool created=false,published=false;
        ~OwnedDirectory(){if(created&&!published){std::error_code ec;std::filesystem::remove_all(path,ec);}}
    } members{memberDirectory.empty()?std::filesystem::path(destination.string()+".members"):memberDirectory},artifacts{std::filesystem::path(destination.string()+".artifacts")};
    if(std::filesystem::exists(artifacts.path)||std::filesystem::is_symlink(artifacts.path))
        throw std::runtime_error("convert: destination artifact directory exists");
    std::unique_ptr<PackedStoreBuilder> builder;std::unique_ptr<StagedDatabase> staged;
    if(packed){builder=std::make_unique<PackedStoreBuilder>(destination);
        for(auto& node:nodes)node->target=builder->addDatabase(node->uuid);
    }else{
        staged=std::make_unique<StagedDatabase>(destination,false);nodes[root]->target=staged->path;
        if(nodes.size()>1){if(std::filesystem::exists(members.path)||std::filesystem::is_symlink(members.path))
            throw std::runtime_error("convert: destination member directory exists");
            members.created=std::filesystem::create_directory(members.path);
            if(!members.created)throw std::runtime_error("convert: cannot create destination member directory");
        }
        for(size_t n=0;n<nodes.size();++n)if(n!=root)nodes[n]->target=members.path/(nodes[n]->uuid+".vcobj");
    }
    for(auto& node:nodes)for(const auto& target:nodes)
        node->locations.emplace(target->uuid,packed?packedMemberReference(target->uuid):
            std::filesystem::absolute(target->target).lexically_relative(std::filesystem::absolute(node->target).parent_path()).generic_string());
    try{
        Json counts=Json::object();for(auto n:order)
            counts[nodes[n]->uuid]=copyPhysical(nodes[n]->db,nodes[n]->target,nodes[n]->source,
                packed?destination:nodes[n]->target,nodes[n]->locations);
        Json mapping={{"store",nodes[root]->store},{"generation",nodes[root]->generation},{"objects",Json::object()},{"relationships",Json::object()}};
        if(exists(nodes[root]->db,"object_versions")){
            Statement objects(nodes[root]->db,"SELECT revision_uuid,object_uuid,version,generation FROM object_versions");
            while(objects.row())mapping["objects"][objects.uuid(0)]={{"object",objects.uuid(1)},{"version",objects.i64(2)},{"generation",objects.uuid(3)}};
            Statement edges(nodes[root]->db,"SELECT revision_uuid,edge_uuid,version,generation FROM relationship_versions WHERE revision_uuid IS NOT NULL");
            while(edges.row())mapping["relationships"][edges.uuid(0)]={{"relationship",edges.uuid(1)},{"version",edges.i64(2)},{"generation",edges.uuid(3)}};
        }
        size_t artifactCount=0;
        for(const auto& node:nodes){
            // Mark ownership before a failed copy can leave a partial directory.
            if(packed||node.get()==nodes[root].get())artifacts.created=true;
            artifactCount+=copyPhysicalArtifacts(node->source,node->target,
                packed||node.get()==nodes[root].get()?destination:node->target);
        }
        for(auto& node:nodes){exec(node->db,"COMMIT");node->owned.reset();}
        // The ordinary current reader validates the complete rewritten closure
        // before publishing the new root. This is converter validation, not an import path.
        {auto checked=ObjectCollection::open(nodes[root]->target,OpenOptions{.readOnly=true});auto read=checked.read();
            if(checked.uuid(read)!=nodes[root]->store)throw std::runtime_error("convert: reopened store identity changed");}
        if(builder)builder->finalize(nodes[root]->uuid);else staged->publish();members.published=true;artifacts.published=true;
        uintmax_t databaseBytes=std::filesystem::file_size(destination);
        if(!packed)for(size_t n=0;n<nodes.size();++n)if(n!=root)databaseBytes+=std::filesystem::file_size(nodes[n]->target);
        Json report={{"source",source.string()},{"destination",destination.string()},{"source_schema",21},{"target_schema",kSchemaVersion},
            {"copied_rows",counts},{"physical_databases",nodes.size()},{"copied_artifacts",artifactCount},{"validation","ok"},
            {"destination_bytes",std::filesystem::file_size(destination)},{"destination_database_bytes",databaseBytes}};
        return {report.dump(2),mapping.dump()};
    }catch(...){for(auto& node:nodes)if(node->db&&node->owned)sqlite3_exec(node->db,"ROLLBACK",nullptr,nullptr,nullptr);throw;}
}
} // namespace vc::objects::migration
