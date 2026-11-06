#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/InitialEdges.hpp"
#include "internal/InitialObjects.hpp"
#include "internal/ImmutableImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "internal/RecordSql.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/RecordJson.hpp"
#include "internal/History.hpp"
#include "internal/CollectionReads.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometryInternal.hpp"
#include "internal/ExternalEndpoint.hpp"
#include "internal/Publication.hpp"
#include <variant>
#include <limits>

namespace vc::objects {
using namespace detail;
bool KeyRoute::matches(int64_t id) const {
    if(id<lower||id>upper)return false;
    if(kind==RoutingKind::Range)return true;
    if(divisor==0||remainder>=divisor||shift>62)throw std::invalid_argument("collection: invalid routing rule");
    uint64_t value=uint64_t(id);
    if(kind==RoutingKind::Bits)value>>=shift;
    if(kind==RoutingKind::HashV1){value=(value^(value>>30))*UINT64_C(0xbf58476d1ce4e5b9);value=(value^(value>>27))*UINT64_C(0x94d049bb133111eb);value^=value>>31;}
    return value%divisor==remainder;
}
namespace {
bool hasCollection(sqlite3* db) {
    Statement s(db,"SELECT 1 FROM store_required_features WHERE name='collection-v1'"); return s.row();
}
void collectionSchema(sqlite3* db) {
    exec(db,R"SQL(
CREATE TABLE collection_members(
 ordinal INTEGER PRIMARY KEY, database_uuid BLOB NOT NULL UNIQUE CHECK(length(database_uuid)=16),
 store_uuid BLOB NOT NULL CHECK(length(store_uuid)=16), generation INTEGER NOT NULL CHECK(generation>0),
 locator TEXT NOT NULL, partition INTEGER NOT NULL CHECK(partition IN (0,1)),
 object_low INTEGER NOT NULL, object_high INTEGER NOT NULL, edge_low INTEGER NOT NULL, edge_high INTEGER NOT NULL
);
CREATE TABLE collection_routes(
 key_space TEXT NOT NULL CHECK(key_space IN ('objects','edges')),
 kind INTEGER NOT NULL CHECK(kind BETWEEN 0 AND 3),
 lower_bound INTEGER NOT NULL CHECK(lower_bound>0), upper_bound INTEGER NOT NULL CHECK(upper_bound>=lower_bound),
 divisor INTEGER NOT NULL CHECK(divisor>0), remainder INTEGER NOT NULL CHECK(remainder>=0 AND remainder<divisor),
 shift INTEGER NOT NULL CHECK(shift BETWEEN 0 AND 62),
 database_uuid BLOB NOT NULL REFERENCES collection_members(database_uuid),
 PRIMARY KEY(key_space,lower_bound,upper_bound,remainder)
) WITHOUT ROWID;
INSERT INTO store_required_features VALUES('collection-v1');
)SQL");
    if(!immutableInitial(db))for(const auto* table:{"collection_members","collection_routes"})
        for(const auto* op:{"INSERT","UPDATE","DELETE"})
            exec(db,(std::string("CREATE TRIGGER ")+table+"_"+op+" AFTER "+op+" ON "+table+
                " BEGIN UPDATE store_write_state SET dirty=1 WHERE singleton=1; END;").c_str());
}
std::filesystem::path normalized(const std::filesystem::path& path) {
    if(isPackedDatabasePath(path)) return path;
    if(isPackedStore(path)) return packedDatabasePath(path);
    return std::filesystem::absolute(path).lexically_normal();
}
std::filesystem::path resolve(const std::filesystem::path& parent,const std::string& ref) {
    if(ref.starts_with("@packed:")) return resolvePackedDatabasePath(parent,ref);
    if(isPackedDatabasePath(ref)) throw std::runtime_error("collection: nonportable packed member locator");
    const auto base=isPackedDatabasePath(parent)?packedContainerPath(parent):parent;
    return normalized(std::filesystem::path(ref).is_absolute()?std::filesystem::path(ref):base.parent_path()/ref);
}
std::string locator(const std::filesystem::path& parent,const std::filesystem::path& target) {
    if(isBuildingPackedDatabasePath(parent)&&isBuildingPackedDatabasePath(target)) {
        const auto ref=packedMemberReference(packedDatabaseUuid(target));
        if(resolvePackedDatabasePath(parent,ref)!=target) throw std::runtime_error("collection: different unpublished containers");
        return ref;
    }
    if(isPackedDatabasePath(target)) {
        // External packed dependencies need both the file and database identity.
        // The manifest separately stores database_uuid; the resolver selects it.
        const auto container=packedContainerPath(target);
        const auto base=isPackedDatabasePath(parent)?packedContainerPath(parent):parent;
        return container.lexically_relative(base.parent_path()).generic_string();
    }
    const auto base=isPackedDatabasePath(parent)?packedContainerPath(parent):parent;
    auto relative=target.lexically_relative(base.parent_path());
    return (relative.empty()?target:relative).generic_string();
}
std::array<int64_t,2> reservation(sqlite3* db,const char* name) {
    Statement s(db,"SELECT lower_bound,upper_bound FROM key_allocators WHERE name=?",std::string(name));
    if(!s.row()) throw std::runtime_error("collection: missing key reservation");
    return {s.i64(0),s.i64(1)};
}
void insertMember(sqlite3* db,int64_t ordinal,const CollectionMember& m,const std::string& ref) {
    const auto objects=m.partition?m.objectIds:std::array<int64_t,2>{0,0};
    const auto edges=m.partition?m.edgeIds:std::array<int64_t,2>{0,0};
    execute(db,"INSERT INTO collection_members VALUES(?,?,?,?,?,?,?,?,?,?)",ordinal,parseUuid(m.databaseUuid),
        parseUuid(m.storeUuid),m.generation,ref,int64_t(m.partition),objects[0],objects[1],edges[0],edges[1]);
}
std::vector<KeyRoute> rangeRoutes(const std::vector<CollectionMember>& members) {
    std::vector<KeyRoute> routes;routes.reserve(2*members.size());
    for(const auto& m:members)for(const auto* space:{"objects","edges"}) {
        const auto r=std::string_view(space)=="objects"?m.objectIds:m.edgeIds;
        routes.push_back({space,RoutingKind::Range,r[0],r[1],1,0,0,m.databaseUuid});
    }
    return routes;
}
// Shared encoding only. Discovery/validation belongs to the caller, not this
// writer: the batch builder already owns every member and reserved range.
void writeShardManifest(ObjectStore& store,const std::filesystem::path& path,
    const std::vector<CollectionMember>& members,const std::vector<KeyRoute>& routes) {
    auto w=store.write();auto* db=ObjectStoreImplAccess::conn(w);collectionSchema(db);
    execute(db,"UPDATE key_allocators SET next_id=NULL");
    int64_t i=0;for(const auto& m:members)insertMember(db,i++,m,locator(path,m.locator));
    for(const auto& route:routes)execute(db,"INSERT INTO collection_routes VALUES(?,?,?,?,?,?,?,?)",route.keySpace,int64_t(route.kind),route.lower,route.upper,int64_t(route.divisor),int64_t(route.remainder),int64_t(route.shift),parseUuid(route.databaseUuid));
    store.seal(w);w.commit();store.checkpoint();
}
ObjectKey canonical(ObjectKey key) {
    if(!key.storeUuid.empty())key.storeUuid=util::canonicalUuid(key.storeUuid);
    key.objectUuid=util::canonicalUuid(key.objectUuid);return key;
}
nlohmann::json editableObject(ObjectRecord record) {
    // Reuse the record serializer so detail schemas stay shared with native
    // reads/writes. Caller-controlled content excludes provenance and approval.
    static const std::set<std::string> fields{"name","format_version","storage_kind","storage_ref",
        "data_generation","data_checksum","meta","tags","fiber","surface","track"};
    std::sort(record.tags.begin(),record.tags.end());record.tags.erase(std::unique(record.tags.begin(),record.tags.end()),record.tags.end());
    if(record.meta.empty())record.meta="{}";
    return objectToJson(record,&fields);
}
void validateInheritedMetadata(const std::vector<CollectionEdge>& edges,const std::string& localOwner,
                               const ObjectKey& key,const ObjectRecord& before,const ObjectRecord& after) {
    const auto original=objectToJson(before),updated=objectToJson(after);
    // Local edges participate in native transaction validation against the
    // same-key override. Only inherited edges require
    // removal/recreation here: their readonly owner cannot revalidate them.
    for(const auto& edge:edges)if(edge.key.storeUuid!=localOwner)
        for(const auto& [endpoint,anchor]:{std::pair{edge.a,edge.record.aAnchor},std::pair{edge.b,edge.record.bAnchor}})
            if(endpoint==key&&dependencyMetadata(original,anchor)!=dependencyMetadata(updated,anchor))
                throw std::runtime_error("collection: explicitly remove/recreate inherited metadata-dependent edges before changing their endpoint metadata");
}
void validateRoutes(const std::vector<KeyRoute>& routes,const std::vector<CollectionMember>& members) {
    for(const auto* space:{"objects","edges"}){
        std::vector<KeyRoute> group;for(const auto& r:routes)if(r.keySpace==space)group.push_back(r);
        if(group.empty())throw std::invalid_argument("collection: missing key-space routing");
        for(size_t a=0;a<group.size();++a){const auto& r=group[a];
            if(r.lower<=0||r.upper<r.lower||r.divisor==0||r.divisor>uint64_t(INT64_MAX)||r.remainder>=r.divisor||r.shift>62||int(r.kind)<0||int(r.kind)>3)
                throw std::invalid_argument("collection: invalid key route");
            if((r.kind!=RoutingKind::Bits&&r.shift!=0)||(r.kind==RoutingKind::Range&&(r.divisor!=1||r.remainder!=0)))
                throw std::invalid_argument("collection: unused routing parameters must have canonical defaults");
            if(r.kind!=group[0].kind||r.divisor!=group[0].divisor||r.shift!=group[0].shift)
                throw std::invalid_argument("collection: one authoritative partition rule required per key space");
            auto m=std::find_if(members.begin(),members.end(),[&](auto& m){return m.databaseUuid==r.databaseUuid&&m.partition;});
            if(m==members.end())throw std::invalid_argument("collection: route destination is not a partition");
            const auto range=r.keySpace=="objects"?m->objectIds:m->edgeIds;
            if(r.lower<range[0]||r.upper>range[1])throw std::invalid_argument("collection: route exceeds reserved namespace");
            if(r.kind==RoutingKind::Bits&&(r.divisor&(r.divisor-1)))throw std::invalid_argument("collection: bit routing divisor must be power of two");
            for(size_t b=0;b<a;++b)if(r.lower<=group[b].upper&&group[b].lower<=r.upper&&
                (r.kind==RoutingKind::Range||r.remainder==group[b].remainder))throw std::invalid_argument("collection: overlapping key routes");
        }
    }
    for(auto& r:routes)if(r.keySpace!="objects"&&r.keySpace!="edges")throw std::invalid_argument("collection: unknown key space");
}
}

struct ObjectCollection::Impl {
    struct Node { std::unique_ptr<ObjectStore> store; CollectionMember member; bool collection=false; std::vector<KeyRoute> routes; std::vector<std::string> children; };
    std::vector<Node> nodes;
    OpenOptions options;
    std::shared_ptr<const InitialEndpointDirectory> initialEndpoints;
    // DFS insertion order is explicit precedence: local, then bases in manifest order.
    // Originals are resolved only after considering all overrides/tombstones.
    std::map<std::string,size_t> byDatabase;
    std::map<std::string,int64_t> pinned;
    void load(const std::filesystem::path&,bool,const std::optional<CollectionMember>&,std::set<std::string>&,sqlite3* existingRoot=nullptr);
    bool depends(size_t n,size_t target)const{
        std::set<size_t> seen;std::vector<size_t> todo{n};
        while(!todo.empty()){auto at=todo.back();todo.pop_back();if(!seen.insert(at).second)continue;
            for(auto& child:nodes[at].children){auto i=byDatabase.at(child);if(i==target)return true;todo.push_back(i);}}
        return false;
    }
    void installCallbacks(sqlite3* sql,size_t n,const std::function<sqlite3*(size_t)>& get)const{
        if(!nodes[n].collection)return;
        std::vector<size_t> chunkSources;
        for(size_t i=0;i<nodes.size();++i)if(i!=n&&depends(n,i))chunkSources.push_back(i);
        auto locate=[sources=std::move(chunkSources),get](const std::string& hash){
            // A hash is already a payload key. A reference does not register
            // a local placeholder or a per-chunk owner record.
            const auto digest=parseDigest(hash);
            for(const auto i:sources){
                auto* source=get(i);
                Statement payload(source,"SELECT length(data) FROM geometry_chunks WHERE hash=?",digest);
                if(payload.row())return std::pair{source,payload.i64(0)};
            }
            return std::pair<sqlite3*,int64_t>{nullptr,0};
        };
        auto* pool=activeStatementPools.at(sql);
        pool->externalGeometryChunkSize=[locate](const auto& hash){return locate(hash).second;};
        pool->externalGeometryChunk=[locate](const auto& hash){
            const auto source=locate(hash).first;
            if(!source)throw std::runtime_error("collection: chunk absent from linked readonly stores");
            return readGeometryPayload(source,hash);
        };
        auto endpoint=[this,n,get](const std::string& uuid){
            return locateSql({{},uuid},get,0,[&](size_t i){
                return i!=n&&(depends(n,i)||nodes[i].member.storeUuid==nodes[n].member.storeUuid);
            });
        };
        pool->externalEndpointOwner=[this,endpoint](const std::string& uuid)->std::optional<std::string>{
            const auto found=endpoint(uuid);
            return found?std::optional(nodes[found->node].member.storeUuid):std::nullopt;
        };
        pool->externalEndpointRecord=[endpoint](const std::string& uuid){
            const auto found=endpoint(uuid);
            if(!found)throw std::runtime_error("collection: missing external endpoint");
            return found->record;
        };
        pool->externalEndpointVersion=[this,endpoint,get](const std::string& uuid){
            const auto found=endpoint(uuid);
            if(!found)throw std::runtime_error("collection: missing external endpoint");
            Statement generation(get(found->node),"SELECT generation FROM store_manifest WHERE singleton=1");
            if(!generation.row())throw std::runtime_error("collection: missing endpoint generation");
            return externalEndpointToken(nodes[found->node].member.storeUuid,uuid,generation.i64(0));
        };
        pool->externalEndpointValidateAnchor=[endpoint,get](const std::string& uuid,const Anchor& anchor){
            const auto found=endpoint(uuid);
            if(!found)throw std::runtime_error("collection: missing external endpoint");
            auto* source=get(found->node);
            const auto geometry=geometryHead(source,uuid);
            if(!geometry){if(anchor.wholeGeometry&&anchor.points.empty())return;throw std::runtime_error("collection: external endpoint geometry absent");}
            validateDependencies(anchor,dependencyGeometry(source,*geometry));
        };
    }
    void check(const CollectionTxn&) const;
    const Txn& txn(const CollectionTxn&,size_t) const;
    sqlite3* db(const CollectionTxn& t,size_t n) const {return ObjectStoreImplAccess::conn(txn(t,n));}
    WriteTxn& writer(CollectionTxn&);
    ObjectKey decode(const CollectionTxn&,size_t n,const std::string& uuid) const {
        return {nodes[n].member.storeUuid,uuid};
    }
    struct Located {size_t node; std::string local; ObjectRecord record;};
    std::optional<Located> locateSql(const ObjectKey&,const std::function<sqlite3*(size_t)>&,size_t start=0,
                                   const std::function<bool(size_t)>& eligible={}) const;
    std::optional<Located> locate(const CollectionTxn&,const ObjectKey&,size_t start=0) const;
    std::optional<std::pair<size_t,GeometryRevision>> geometry(const CollectionTxn&,const ObjectKey&) const;
    std::string ensureLocal(CollectionTxn&,const ObjectKey&);
    std::string ensureLocalGeometry(CollectionTxn&,const ObjectKey&);
    ObjectKey insertEdge(CollectionTxn&,const ObjectKey&,const ObjectKey&,const EdgeRecord&,std::optional<int64_t>);
    CollectionEdge edge(const CollectionTxn& t,size_t n,EdgeRecord r) const {
        auto a=decode(t,n,r.aUuid),b=decode(t,n,r.bUuid);
        ObjectKey key{nodes[n].member.storeUuid,r.uuid};
        r.aUuid=a.objectUuid;r.bUuid=b.objectUuid;return {key,a,b,std::move(r)};
    }
};
struct CollectionTxn::Impl {
    ObjectCollection::Impl* owner;
    bool producingInitialEdges=false;
    std::variant<ReadTxn,WriteTxn> root;
    mutable std::map<size_t,std::unique_ptr<ReadTxn>> reads;
    mutable std::set<StatementPool*> configured;
    Impl(ObjectCollection::Impl* p,bool write):owner(p),root(write?
        std::variant<ReadTxn,WriteTxn>(std::in_place_type<WriteTxn>,p->nodes[0].store->write()):
        std::variant<ReadTxn,WriteTxn>(std::in_place_type<ReadTxn>,p->nodes[0].store->read())){get(0);}
    // Txn::release clears callbacks while it still exclusively owns its leased
    // connection. Never touch pool pointers after commit returns that connection
    // to another thread. This set only remembers installation during our lease.
    void clear(){configured.clear();}
    const Txn& get(size_t n) const {
        const Txn* txn=nullptr;
        if(n==0)txn=&std::visit([](auto& x)->const Txn&{return x;},root);
        else {auto& r=reads[n];if(!r)r=std::make_unique<ReadTxn>(owner->nodes[n].store->read());txn=r.get();}
        auto* db=ObjectStoreImplAccess::conn(*txn);
        auto* pool=activeStatementPools.at(db);
        if(configured.insert(pool).second){
            try{owner->installCallbacks(db,n,[this](size_t i){return ObjectStoreImplAccess::conn(get(i));});}
            catch(...){pool->externalGeometryChunk={};pool->externalGeometryChunkSize={};pool->externalEndpointOwner={};pool->externalEndpointRecord={};pool->externalEndpointVersion={};pool->externalEndpointValidateAnchor={};configured.erase(pool);throw;}
        }
        return *txn;
    }
};
CollectionTxn::CollectionTxn(std::unique_ptr<Impl> p):impl_(std::move(p)){}
CollectionTxn::CollectionTxn(CollectionTxn&&) noexcept=default;
CollectionTxn::~CollectionTxn()=default;
void CollectionTxn::commit(){if(!impl_)throw std::runtime_error("collection: moved transaction");
    if(impl_->producingInitialEdges)throw std::runtime_error("collection: initial edge producer cannot commit the destination");
    try{std::visit([](auto& t){t.commit();},impl_->root);}catch(...){impl_->clear();impl_->reads.clear();throw;}
    impl_->clear();impl_->reads.clear();}
bool CollectionTxn::writable()const{return impl_&&std::holds_alternative<WriteTxn>(impl_->root);}
void ObjectCollection::Impl::check(const CollectionTxn& t)const{
    if(!t.impl_||t.impl_->owner!=this||!std::visit([](auto& x){return x.active();},t.impl_->root))
        throw std::runtime_error("collection: foreign or finished transaction");
}
const Txn& ObjectCollection::Impl::txn(const CollectionTxn& t,size_t n)const{
    check(t);return t.impl_->get(n);
}
WriteTxn& ObjectCollection::Impl::writer(CollectionTxn& t){check(t);if(!t.writable())throw std::runtime_error("collection: read-only transaction");if(t.impl_->producingInitialEdges)throw std::runtime_error("collection: initial edge producer cannot mutate the destination");return std::get<WriteTxn>(t.impl_->root);}

void ObjectCollection::Impl::load(const std::filesystem::path& input,bool root,const std::optional<CollectionMember>& expected,std::set<std::string>& visiting,sqlite3* existingRoot) {
    auto path=normalized(input); auto opt=options;opt.physicalPartitionOnly=true;if(!root)opt.readOnly=true;
    std::unique_ptr<ObjectStore> store;std::unique_ptr<ReadTxn> read;
    if(!existingRoot){store=std::make_unique<ObjectStore>(ObjectStore::open(path,opt));read=std::make_unique<ReadTxn>(store->read());}
    auto* sql=existingRoot?existingRoot:ObjectStoreImplAccess::conn(*read);
    Statement header(sql,"SELECT s.uuid,d.uuid,m.generation,m.sealed FROM store_identity s,database_identity d,store_manifest m");
    if(!header.row())throw std::runtime_error("collection: absent manifest identity");
    StoreProvisioning p;p.generation=header.i64(2);p.sealed=header.i64(3)!=0;
    DatabaseIdentity id;id.uuid=formatUuid(header.blob16(1));
    CollectionMember member{id.uuid,formatUuid(header.blob16(0)),p.generation,path,expected?expected->partition:false,reservation(sql,"objects"),reservation(sql,"edges")};
    if(!root&&!p.sealed)throw std::runtime_error("collection: dependency is not sealed: "+path.string());
    if(expected&&(member.databaseUuid!=expected->databaseUuid||member.storeUuid!=expected->storeUuid||member.generation!=expected->generation||
                  (expected->partition&&(member.objectIds!=expected->objectIds||member.edgeIds!=expected->edgeIds))))
        throw std::runtime_error("collection: dependency identity/generation/reservation mismatch: "+path.string());
    if(visiting.contains(id.uuid))throw std::runtime_error("collection: cyclic database dependency");
    if(byDatabase.contains(id.uuid))return;
    if(auto it=pinned.find(member.storeUuid);it!=pinned.end()&&it->second!=member.generation)
        throw std::runtime_error("collection: inconsistent repeated store generation");
    pinned[member.storeUuid]=member.generation;
    bool composed=hasCollection(sql);std::vector<CollectionMember> children;std::vector<KeyRoute> routes;
    if(composed){
        Statement s(sql,"SELECT database_uuid,store_uuid,generation,locator,partition,object_low,object_high,edge_low,edge_high FROM collection_members ORDER BY ordinal");
        while(s.row()){
            CollectionMember child{formatUuid(s.blob16(0)),formatUuid(s.blob16(1)),s.i64(2),resolve(path,s.text(3)),s.i64(4)!=0,{s.i64(5),s.i64(6)},{s.i64(7),s.i64(8)}};
            // For an external packed file, select its declared physical member.
            if(!s.text(3).starts_with("@packed:")&&isPackedDatabasePath(child.locator))child.locator=packedDatabasePath(packedContainerPath(child.locator),child.databaseUuid);
            if(child.partition?(child.storeUuid!=member.storeUuid||child.generation!=member.generation):
                               (child.storeUuid==member.storeUuid||child.generation>=member.generation))
                throw std::runtime_error("collection: invalid dependency generation/ownership ordering");
            children.push_back(std::move(child));
        }
        Statement routing(sql,"SELECT key_space,kind,lower_bound,upper_bound,divisor,remainder,shift,database_uuid FROM collection_routes ORDER BY key_space,lower_bound,remainder");
        while(routing.row())routes.push_back({routing.text(0),RoutingKind(routing.i64(1)),routing.i64(2),routing.i64(3),uint64_t(routing.i64(4)),uint64_t(routing.i64(5)),unsigned(routing.i64(6)),formatUuid(routing.blob16(7))});
        if(!routes.empty()||std::any_of(children.begin(),children.end(),[](auto& c){return c.partition;}))validateRoutes(routes,children);
    }
    if(read)read->commit();byDatabase[id.uuid]=nodes.size();std::vector<std::string> childIds;for(auto& c:children)childIds.push_back(c.databaseUuid);
    nodes.push_back({std::move(store),member,composed,routes,childIds});visiting.insert(id.uuid);
    for(const auto& child:children)load(child.locator,false,child,visiting);
    visiting.erase(id.uuid);
}
struct detail::CollectionReadContext {
    ObjectCollection::Impl collection;
    sqlite3* root;
    std::unique_ptr<StatementPool> rootPool;
    std::map<size_t,std::unique_ptr<ReadTxn>> readers;
    std::set<StatementPool*> configured;
    CollectionReadContext(sqlite3* sql,const std::filesystem::path& path):root(sql){
        collection.options.readOnly=true;std::set<std::string> visiting;collection.load(path,true,{},visiting,sql);
        if(activeStatementPools.contains(root))throw std::runtime_error("collection: maintenance connection already leased");
        rootPool=std::make_unique<StatementPool>();rootPool->historyEnabled=historyEnabled(root);rootPool->spatialEnabled=spatialEnabled(root);
        activeStatementPools[root]=rootPool.get();
        try{get(0);}catch(...){activeStatementPools.erase(root);throw;}
    }
    ~CollectionReadContext(){for(auto* p:configured){p->externalGeometryChunk={};p->externalGeometryChunkSize={};p->externalEndpointOwner={};p->externalEndpointRecord={};p->externalEndpointVersion={};p->externalEndpointValidateAnchor={};}activeStatementPools.erase(root);}
    sqlite3* get(size_t n){
        sqlite3* sql=root;
        if(n){auto& r=readers[n];if(!r)r=std::make_unique<ReadTxn>(collection.nodes[n].store->read());sql=ObjectStoreImplAccess::conn(*r);}
        auto* pool=activeStatementPools.at(sql);
        if(configured.insert(pool).second){
            try{collection.installCallbacks(sql,n,[this](size_t i){return get(i);});}
            catch(...){pool->externalGeometryChunk={};pool->externalGeometryChunkSize={};pool->externalEndpointOwner={};pool->externalEndpointRecord={};pool->externalEndpointVersion={};pool->externalEndpointValidateAnchor={};configured.erase(pool);throw;}
        }
        return sql;
    }
};
std::shared_ptr<void> detail::attachCollectionChunkResolver(sqlite3* db,const std::filesystem::path& path){
    return hasCollection(db)?std::make_shared<CollectionReadContext>(db,path):std::shared_ptr<void>{};
}
std::optional<ObjectCollection::Impl::Located> ObjectCollection::Impl::locate(const CollectionTxn& t,const ObjectKey& key,size_t start)const{
    check(t);
    return locateSql(key,[&](size_t n){return db(t,n);},start);
}
std::optional<ObjectCollection::Impl::Located> ObjectCollection::Impl::locateSql(const ObjectKey& key,
    const std::function<sqlite3*(size_t)>& get,size_t start,const std::function<bool(size_t)>& eligible)const{
    // UUID is the complete key. The first present row wins, including deletion.
    for(size_t n=start;n<nodes.size();++n)if(!eligible||eligible(n)){
        auto* sql=get(n);
        Statement state(sql,"SELECT deleted FROM objects WHERE uuid=?",parseUuid(key.objectUuid));
        if(!state.row())continue;
        if(state.i64(0))return {};
        Statement row(sql,std::string("SELECT ")+kObjectCols+" FROM objects WHERE uuid=?",parseUuid(key.objectUuid));
        if(!row.row())throw std::runtime_error("collection: object disappeared inside a read snapshot");
        return Located{n,key.objectUuid,readObjectRow(sql,row)};
    }
    return {};
}
std::optional<std::pair<size_t,GeometryRevision>> ObjectCollection::Impl::geometry(const CollectionTxn& t,const ObjectKey& key)const{
    const auto found=locate(t,key);if(!found)return {};
    if(auto geometry=nodes[found->node].store->currentGeometry(txn(t,found->node),key.objectUuid))
        return std::pair{found->node,*geometry};
    return {};
}
std::string ObjectCollection::Impl::ensureLocal(CollectionTxn& t,const ObjectKey& key){
    auto& w=writer(t);
    const auto source=locate(t,key);
    if(!source)throw std::runtime_error("collection: missing/deleted object");
    if(source->node==0)return key.objectUuid;
    const auto inherited=geometry(t,key);
    auto record=source->record;
    record.storageKind.clear();record.storageRef.clear();record.dataChecksum.reset();record.dataGeneration=0;
    nodes[0].store->upsertObject(w,record);
    if(record.retired||!record.lifecycleEpoch.empty())
        writeLifecycle(db(t,0),*objectIdByUuid(db(t,0),parseUuid(key.objectUuid)),record);
    if(inherited){
        auto* target=db(t,0);auto* sourceDb=db(t,inherited->first);
        GeometryReader reader(sourceDb,inherited->second.uuid,false);auto manifest=reader.manifest();
        for(auto& [name,array]:manifest.at("arrays").items()){
            array.erase("source_revision");int64_t index=0;
            for(const auto& entry:array.at("chunks")){
                const auto hash=entry.get<std::string>();
                Statement exists(sourceDb,"SELECT 1 FROM geometry_chunks WHERE hash=?",parseDigest(hash));
                if(!exists.row()&&!hasExternalGeometryPayload(target,hash)){
                    const auto bytes=reader.chunk(name,index);
                    if(storeSharedPayload(target,name,bytes)!=hash)throw std::runtime_error("collection: reconstructed chunk hash mismatch");
                }
                ++index;
            }
        }
        publishReferencedGeometry(*nodes[0].store,w,key.objectUuid,manifest);
    }
    return key.objectUuid;
}
std::string ObjectCollection::Impl::ensureLocalGeometry(CollectionTxn& t,const ObjectKey& key){
    ensureLocal(t,key);
    const auto geometry=nodes[0].store->currentGeometry(txn(t,0),key.objectUuid);
    if(!geometry)throw std::runtime_error("collection: inherited geometry absent");
    return geometry->uuid;
}

ObjectCollection::ObjectCollection(std::unique_ptr<Impl> p):impl_(std::move(p)){}
ObjectCollection::ObjectCollection(ObjectCollection&&) noexcept=default;
ObjectCollection::~ObjectCollection()=default;
ObjectCollection ObjectCollection::open(const std::filesystem::path& path,const OpenOptions& options){
    auto p=std::make_unique<Impl>();p->options=options;std::set<std::string> visiting;p->load(path,true,{},visiting);return ObjectCollection(std::move(p));
}
ObjectCollection ObjectCollection::create(const std::filesystem::path& path,const std::vector<std::filesystem::path>& bases,const OpenOptions& options){
    std::vector<CollectionMember> members;int64_t generation=1;
    for(const auto& base:bases){auto c=open(base,OpenOptions{.readOnly=true});auto t=c.read();auto p=c.provisioning(t);
        if(!p.sealed)throw std::runtime_error("collection: base must be sealed");
        if(p.generation>=INT64_MAX-1)throw std::runtime_error("collection: dependency generation exhausted");
        generation=std::max(generation,p.generation);members.push_back(c.impl_->nodes[0].member);
    }
    auto opt=options;opt.initialGeneration=std::max(opt.initialGeneration,generation);opt.physicalPartitionOnly=true;
    StagedDatabase staged(path,isBuildingPackedDatabasePath(path));
    {auto store=ObjectStore::create(staged.path,opt);auto w=store.write();auto* db=ObjectStoreImplAccess::conn(w);collectionSchema(db);
     int64_t i=0;for(const auto& m:members){if(m.storeUuid==store.uuid(w))throw std::runtime_error("collection: overlay needs an independent store UUID");
         insertMember(db,i++,m,locator(normalized(path),m.locator));}
     w.commit();store.checkpoint();}
    {auto verified=open(staged.path,OpenOptions{.readOnly=true});}
    staged.publish();
    return open(path,options);
}
ObjectCollection detail::InitialCollectionAccess::create(const std::filesystem::path& path,
    const std::vector<std::filesystem::path>& bases,const OpenOptions& options)
{
    // This path is private to the initial builder, which owns staging and
    // publication. No intermediate writable store is published or reopened.
    std::vector<CollectionMember> members;int64_t generation=options.initialGeneration;
    for(const auto& base:bases){auto c=ObjectCollection::open(base,OpenOptions{.readOnly=true});auto r=c.read();
        const auto p=c.provisioning(r);if(!p.sealed||p.generation>=INT64_MAX-1)
            throw std::runtime_error("initial collection needs sealed, non-exhausted dependencies");
        generation=std::max(generation,p.generation);members.push_back(c.impl_->nodes[0].member);}
    auto opt=options;opt.initialGeneration=generation;opt.physicalPartitionOnly=true;
    auto store=std::make_unique<ObjectStore>(ObjectStoreImplAccess::createInitial(path,opt));
    auto p=std::make_unique<ObjectCollection::Impl>();p->options=opt;
    {auto w=store->write();auto* db=ObjectStoreImplAccess::conn(w);collectionSchema(db);
     int64_t ordinal=0;for(const auto& m:members){
        if(m.storeUuid==store->uuid(w))throw std::runtime_error("initial collection needs independent dependency ownership");
        insertMember(db,ordinal++,m,locator(normalized(path),m.locator));}
     CollectionMember root{store->databaseIdentity(w).uuid,store->uuid(w),writeGeneration(db),path,false,opt.objectIdRange,opt.edgeIdRange};
     std::vector<std::string> children;for(const auto& m:members)children.push_back(m.databaseUuid);
     p->byDatabase.emplace(root.databaseUuid,0);p->pinned.emplace(root.storeUuid,root.generation);
     p->nodes.push_back({nullptr,std::move(root),true,{},std::move(children)});
     w.commit();}
    // No reader pool or collection loading for the database just constructed.
    // Pre-existing external dependencies still require their ordinary readers.
    std::set<std::string> visiting{p->nodes[0].member.databaseUuid};
    for(const auto& m:members)p->load(m.locator,false,m,visiting);
    p->nodes[0].store=std::move(store);return ObjectCollection(std::move(p));
}
void detail::InitialCollectionAccess::setEndpoints(ObjectCollection& collection,std::shared_ptr<const InitialEndpointDirectory> endpoints)
{
    collection.impl_->initialEndpoints=std::move(endpoints);
}
CollectionMember detail::InitialCollectionAccess::sealShard(ObjectCollection& collection,CollectionTxn& transaction) {
    collection.seal(transaction);
    auto member=collection.impl_->nodes[0].member;
    member.generation=writeGeneration(collection.impl_->db(transaction,0));
    member.partition=true;
    transaction.commit();collection.checkpoint();
    return member;
}
void detail::InitialCollectionAccess::writeManifest(const std::filesystem::path& path,
    const std::vector<CollectionMember>& members,const OpenOptions& options) {
    const auto& first=members.at(0);
    auto opt=options;opt.history=HistoryMode::Off;opt.spatialIndex=false;
    opt.creationStoreUuid=first.storeUuid;opt.initialGeneration=first.generation-1;opt.physicalPartitionOnly=true;
    auto store=ObjectStoreImplAccess::createInitial(path,opt);
    writeShardManifest(store,path,members,rangeRoutes(members));
}
ObjectCollection ObjectCollection::publishShards(const std::filesystem::path& path,const std::vector<std::filesystem::path>& shards,const OpenOptions& options,const std::vector<KeyRoute>& requestedRoutes){
    if(shards.empty())throw std::runtime_error("collection: empty shard list");
    std::vector<CollectionMember> members;for(const auto& shard:shards){
        auto c=open(shard,OpenOptions{.readOnly=true});auto t=c.read();if(!c.provisioning(t).sealed)throw std::runtime_error("collection: unpublished shard");
        // A physical partition may itself have immutable dependencies (for
        // example relation-only shards referencing a geometry collection).
        // Its own content still resides in one independent SQLite database.
        auto m=c.impl_->nodes[0].member;m.partition=true;members.push_back(m);
    }
    const auto& first=members.front();if(first.generation<2)throw std::runtime_error("collection: invalid shard generation");
    for(auto& m:members)if(m.storeUuid!=first.storeUuid||m.generation!=first.generation)throw std::runtime_error("collection: shards disagree on ownership/generation");
    auto routes=requestedRoutes.empty()?rangeRoutes(members):requestedRoutes;
    validateRoutes(routes,members);
    // Custom routing is verified once at publication. Readers only trust the
    // sealed manifest and never rescan all IDs on every open.
    if(!requestedRoutes.empty())for(auto& m:members){auto store=ObjectStore::open(m.locator,OpenOptions{.readOnly=true});auto r=store.read();
        for(const auto* space:{"objects","edges"}){Statement ids(ObjectStoreImplAccess::conn(r),std::string("SELECT id FROM ")+space);
            while(ids.row()){size_t matches=0;for(auto& route:routes)if(route.keySpace==space&&route.matches(ids.i64(0))){if(route.databaseUuid!=m.databaseUuid)throw std::runtime_error("collection: misplaced shard key");++matches;}
                if(matches!=1)throw std::runtime_error("collection: shard key not uniquely routed");}}}
    auto opt=options;opt.history=HistoryMode::Off;opt.spatialIndex=false;opt.creationStoreUuid=first.storeUuid;opt.initialGeneration=first.generation-1;opt.physicalPartitionOnly=true;
    const bool packed=isBuildingPackedDatabasePath(path);
    StagedDatabase staged(path,packed);const auto& stage=staged.path;
    {auto store=ObjectStore::create(stage,opt);writeShardManifest(store,normalized(path),members,routes);}
    // Validate all targets and the routing map before the publication name exists.
    {auto verified=open(stage,OpenOptions{.readOnly=true});}
    staged.publish();
    auto readonly=options;readonly.readOnly=true;return open(path,readonly);
}
CollectionTxn ObjectCollection::read(){return CollectionTxn(std::make_unique<CollectionTxn::Impl>(impl_.get(),false));}
void ObjectCollection::checkpoint(){impl_->nodes[0].store->checkpoint();}
CollectionTxn ObjectCollection::write(){if(!impl_->nodes[0].collection)throw std::runtime_error("collection: use ObjectStore to edit a plain store");return CollectionTxn(std::make_unique<CollectionTxn::Impl>(impl_.get(),true));}
std::string ObjectCollection::uuid(const CollectionTxn& t)const{impl_->check(t);return impl_->nodes[0].member.storeUuid;}
StoreProvisioning ObjectCollection::provisioning(const CollectionTxn& t)const{return impl_->nodes[0].store->provisioning(impl_->txn(t,0));}
std::vector<CollectionMember> ObjectCollection::members()const{std::vector<CollectionMember> r;for(const auto& n:impl_->nodes)r.push_back(n.member);return r;}
std::optional<CollectionObject> ObjectCollection::object(const CollectionTxn& t,const ObjectKey& input)const{
    const auto key=canonical(input);auto r=impl_->locate(t,key);if(!r)return {};return CollectionObject{key,std::move(r->record)};
}
std::optional<CollectionObject> ObjectCollection::objectById(const CollectionTxn& t,const std::string& owner,int64_t id)const{
    const auto uuid=util::canonicalUuid(owner);std::optional<ObjectKey> found;
    std::optional<std::string> routed;
    for(const auto& node:impl_->nodes)if(node.member.storeUuid==uuid&&!node.routes.empty()){
        for(auto& route:node.routes)if(route.keySpace=="objects"&&route.matches(id)){
            if(routed&&*routed!=route.databaseUuid)throw std::runtime_error("collection: conflicting routing");routed=route.databaseUuid;}
        if(!routed)return {};
    }
    for(size_t n=0;n<impl_->nodes.size();++n){const auto& node=impl_->nodes[n];if(node.member.storeUuid!=uuid||id<node.member.objectIds[0]||id>node.member.objectIds[1])continue;
        if(routed&&node.member.databaseUuid!=*routed)continue;
        Statement s(impl_->db(t,n),"SELECT uuid FROM objects WHERE id=?",id);if(s.row()){
            if(found)throw std::runtime_error("collection: ambiguous numeric key");found=impl_->decode(t,n,formatUuid(s.blob16(0)));}}
    return found?object(t,*found):std::nullopt;
}
void ObjectCollection::scanObjects(const CollectionTxn& t,const std::function<bool(const CollectionObject&)>& visit,const std::optional<std::string>& kind)const{
    std::set<ObjectKey> seen;
    for(size_t n=0;n<impl_->nodes.size();++n){ObjectSelection selection;selection.limit=4096;
        while(true){const auto rows=impl_->nodes[n].store->selectObjects(impl_->txn(t,n),selection);if(rows.empty())break;
            for(const auto& r:rows){auto key=impl_->decode(t,n,r.uuid);if(!seen.insert(key).second)continue;
                auto resolved=object(t,key);if(resolved&&(!kind||resolved->record.kind==*kind)&&!visit(*resolved))return;}
            selection.afterUuid=rows.back().uuid;}}
}
std::optional<GeometryRevision> ObjectCollection::currentGeometry(const CollectionTxn& t,const ObjectKey& key)const{
    const auto normalizedKey=canonical(key);auto g=impl_->geometry(t,normalizedKey);if(!g)return {};g->second.objectUuid=normalizedKey.objectUuid;return g->second;
}
std::vector<uint8_t> ObjectCollection::readGeometryArray(const CollectionTxn& t,const ObjectKey& key,const std::string& array)const{
    auto g=impl_->geometry(t,canonical(key));if(!g)throw std::runtime_error("collection: geometry absent");
    return impl_->nodes[g->first].store->readGeometryArray(impl_->txn(t,g->first),g->second.uuid,array);
}
std::string ObjectCollection::putGeometry(CollectionTxn& t,const ObjectKey& input,const GeometryData& data){
    const auto key=canonical(input);auto& w=impl_->writer(t);Savepoint save(impl_->db(t,0));
    if(currentGeometry(t,key))throw std::invalid_argument("collection: use sparse/structural edit for existing geometry");
    const auto local=impl_->ensureLocal(t,key);const auto result=impl_->nodes[0].store->putGeometry(w,local,data);
    save.release();return result;
}
std::string ObjectCollection::editGeometry(CollectionTxn& t,const ObjectKey& input,const std::vector<GeometryArrayEdit>& edits,const std::string& expected,const std::vector<GeometryEdgeEdit>& decisions){
    const auto key=canonical(input);auto& w=impl_->writer(t);auto current=currentGeometry(t,key);
    if(!current||current->uuid!=expected)throw std::runtime_error("collection: geometry changed; reload before editing");
    for(const auto& edge:edgesOf(t,key))if(edge.key.storeUuid!=uuid(t))
        throw std::runtime_error("collection: explicitly remove/recreate inherited incident edges before editing geometry");
    std::map<std::string,CollectionEdge> observed;
    for(const auto& decision:decisions){
        const auto before=edge(t,{uuid(t),decision.edgeUuid});
        if(!before||before->record.generation!=decision.expectedGeneration)throw std::runtime_error("collection: geometry decision edge changed");
        const auto ga=currentGeometry(t,before->a),gb=currentGeometry(t,before->b);
        if(decision.aRevision!=(ga?std::optional(ga->uuid):std::nullopt)||decision.bRevision!=(gb?std::optional(gb->uuid):std::nullopt))
            throw std::runtime_error("collection: geometry decision endpoint changed");
        observed.emplace(decision.edgeUuid,*before);
    }
    Savepoint save(impl_->db(t,0));const auto localExpected=impl_->ensureLocalGeometry(t,key);const auto local=impl_->ensureLocal(t,key);
    auto adapted=decisions;for(auto& d:adapted){
        const auto after=*impl_->nodes[0].store->edgeByUuid(w,d.edgeUuid);
        if(impl_->decode(t,0,after.aUuid)!=observed.at(d.edgeUuid).a){std::swap(d.aAnchor,d.bAnchor);std::swap(d.aRevision,d.bRevision);}
        d.expectedGeneration=after.generation;
        d.aRevision=geometryHead(impl_->db(t,0),after.aUuid);d.bRevision=geometryHead(impl_->db(t,0),after.bUuid);
    }
    const auto result=impl_->nodes[0].store->editGeometry(w,local,edits,localExpected,adapted);save.release();return result;
}
std::vector<CollectionEdge> ObjectCollection::edgesOf(const CollectionTxn& t,const ObjectKey& input)const{
    const auto key=canonical(input);const auto endpoint=object(t,key);
    if(!endpoint)throw std::runtime_error("collection: object absent");
    if(endpoint->record.retired)return {};
    std::vector<CollectionEdge> out;std::set<std::string> seen;
    for(size_t n=0;n<impl_->nodes.size();++n){
        Statement selected(impl_->db(t,n),"SELECT uuid FROM edges WHERE a_uuid=?1 OR b_uuid=?1",parseUuid(key.objectUuid));
        while(selected.row()){
            const auto uuid=formatUuid(selected.blob16(0));if(!seen.insert(uuid).second)continue;
            auto current=edge(t,{impl_->nodes[n].member.storeUuid,uuid});
            if(current&&(current->a==key||current->b==key))out.push_back(std::move(*current));
        }
    }
    return out;
}
std::vector<CollectionEdge> ObjectCollection::edgesByKind(const CollectionTxn& t,const std::string& kind)const{
    std::vector<CollectionEdge> out;std::set<std::string> seen;
    for(size_t n=0;n<impl_->nodes.size();++n){
        Statement selected(impl_->db(t,n),"SELECT uuid FROM edges WHERE kind=?",kind);
        while(selected.row()){
            const auto uuid=formatUuid(selected.blob16(0));if(!seen.insert(uuid).second)continue;
            auto current=edge(t,{impl_->nodes[n].member.storeUuid,uuid});
            if(current&&current->record.kind==kind)out.push_back(std::move(*current));
        }
    }
    return out;
}
std::optional<CollectionEdge> ObjectCollection::edge(const CollectionTxn& t,const ObjectKey& input)const{
    const auto key=canonical(input);
    for(size_t n=0;n<impl_->nodes.size();++n){
        Statement state(impl_->db(t,n),"SELECT deleted FROM edges WHERE uuid=?",parseUuid(key.objectUuid));
        if(!state.row())continue;
        if(state.i64(0))return {};
        auto r=impl_->nodes[n].store->edgeByUuid(impl_->txn(t,n),key.objectUuid);
        if(!r)throw std::runtime_error("collection: edge disappeared inside read snapshot");
        auto result=impl_->edge(t,n,std::move(*r));
        const auto a=object(t,result.a),b=object(t,result.b);
        if(a&&b&&!a->record.retired&&!b->record.retired)return result;
        return {};
    }
    return {};
}
std::optional<CollectionEdge> ObjectCollection::edgeById(const CollectionTxn& t,const std::string& owner,int64_t id)const{
    const auto uuid=util::canonicalUuid(owner);std::optional<std::string> routed;std::optional<ObjectKey> found;
    for(const auto& node:impl_->nodes)if(node.member.storeUuid==uuid&&!node.routes.empty()){
        for(const auto& rule:node.routes)if(rule.keySpace=="edges"&&rule.matches(id)){
            if(routed&&*routed!=rule.databaseUuid)throw std::runtime_error("collection: conflicting edge routing");routed=rule.databaseUuid;}
        if(!routed)return {};
    }
    for(size_t n=0;n<impl_->nodes.size();++n){auto& m=impl_->nodes[n].member;
        if(m.storeUuid!=uuid||id<m.edgeIds[0]||id>m.edgeIds[1]||(routed&&m.databaseUuid!=*routed))continue;
        Statement row(impl_->db(t,n),"SELECT uuid FROM edges WHERE id=?",id);if(row.row()){
            if(found)throw std::runtime_error("collection: ambiguous numeric edge key");found=ObjectKey{uuid,formatUuid(row.blob16(0))};}}
    return found?edge(t,*found):std::nullopt;
}
ObjectKey ObjectCollection::upsertObject(CollectionTxn& t,const ObjectRecord& r){
    auto& w=impl_->writer(t);
    if(!r.uuid.empty()){
        const ObjectKey key{uuid(t),util::canonicalUuid(r.uuid)};
        if(object(t,key)){replaceObject(t,key,r);return key;}
    }
    return {uuid(t),impl_->nodes[0].store->upsertObject(w,r)};
}
void ObjectCollection::replaceObject(CollectionTxn& t,const ObjectKey& input,const ObjectRecord& r){
    auto key=canonical(input);auto& w=impl_->writer(t);Savepoint save(impl_->db(t,0));
    const auto previous=object(t,key);if(!previous)throw std::runtime_error("collection: missing/deleted object");
    auto proposed=r;proposed.uuid=key.objectUuid;
    if(editableObject(proposed)==editableObject(previous->record)){save.release();return;}
    const auto local=impl_->ensureLocal(t,key);
    impl_->nodes[0].store->upsertObject(w,proposed);
    validateInheritedMetadata(edgesOf(t,key),uuid(t),key,previous->record,*impl_->nodes[0].store->objectByUuid(w,local));
    save.release();
}
void ObjectCollection::setMeta(CollectionTxn& t,const ObjectKey& input,const std::string& field,const std::string& json){
    auto key=canonical(input);auto& w=impl_->writer(t);Savepoint save(impl_->db(t,0));
    const auto previous=object(t,key);if(!previous)throw std::runtime_error("collection: missing/deleted object");
    const auto local=impl_->ensureLocal(t,key),before=impl_->nodes[0].store->objectByUuid(w,local)->meta;
    impl_->nodes[0].store->setMeta(w,local,field,json);
    const auto after=impl_->nodes[0].store->objectByUuid(w,local);
    // No-op edits roll back a tentative same-key local override too.
    if(before==after->meta)return;
    validateInheritedMetadata(edgesOf(t,key),uuid(t),key,previous->record,*after);
    save.release();
}
void ObjectCollection::deleteObject(CollectionTxn& t,const ObjectKey& input){
    const auto key=canonical(input);auto& w=impl_->writer(t);
    const auto previous=object(t,key);if(!previous)throw std::runtime_error("collection: missing/deleted object");
    if(!edgesOf(t,key).empty())throw std::runtime_error("collection: remove incident relationships before deletion");
    Savepoint save(impl_->db(t,0));
    if(!impl_->nodes[0].store->objectByUuid(w,key.objectUuid)){
        // Deletion needs only a same-key ordinary record, never copied payloads.
        ObjectRecord marker;marker.uuid=key.objectUuid;marker.kind=previous->record.kind;
        impl_->nodes[0].store->upsertObject(w,marker);
    }
    impl_->nodes[0].store->deleteObject(w,key.objectUuid);
    save.release();
}
ObjectKey ObjectCollection::Impl::insertEdge(CollectionTxn& t,const ObjectKey& a,const ObjectKey& b,const EdgeRecord& input,std::optional<int64_t> id){
    GeometryPhase phase("collection.edge");
    auto& w=writer(t);Savepoint save(db(t,0));
    {GeometryPhase geometryPhase("collection.edge.geometry");
    for(const auto& [key,anchor]:{std::pair{canonical(a),input.aAnchor},std::pair{canonical(b),input.bAnchor}}){
        auto endpoint=locate(t,key);
        if(!endpoint||endpoint->record.retired)throw std::runtime_error("collection: missing/deleted/retired endpoint");
        if(anchor.wholeGeometry&&anchor.points.empty())continue;
        if(auto g=geometry(t,key)){
            validateDependencies(anchor,dependencyGeometry(*nodes[g->first].store,txn(t,g->first),g->second));
        }
    }
    }
    auto r=input;r.aUuid=canonical(a).objectUuid;r.bUuid=canonical(b).objectUuid;
    const auto result=id?nodes[0].store->insertEdgeWithId(w,r,*id):nodes[0].store->addEdge(w,r);
    save.release();return {nodes[0].member.storeUuid,result};
}
ObjectKey ObjectCollection::addEdge(CollectionTxn& t,const ObjectKey& a,const ObjectKey& b,const EdgeRecord& input){
    return impl_->insertEdge(t,a,b,input,{});
}
ObjectKey ObjectCollection::insertEdgeWithId(CollectionTxn& t,const ObjectKey& a,const ObjectKey& b,const EdgeRecord& input,int64_t id){
    return impl_->insertEdge(t,a,b,input,id);
}
int64_t ObjectCollection::importInitialEdges(CollectionTxn& t,
    const std::function<bool(std::vector<InitialCollectionEdge>&)>& producer,
    const InitialEdgeImportOptions& options)
{
    if(!producer || !options.maxCachedEndpoints)
        throw std::invalid_argument("collection: initial edge import requires a producer and positive endpoint cache limit");
    auto& w=impl_->writer(t);auto* db=impl_->db(t,0);
    Savepoint save(db);
    detail::InitialEdgeInserter writer(*impl_->nodes[0].store,w,options.deferIndexes,options.maxCachedEndpoints,options.sqlBatchRows);
    struct Endpoint {std::string uuid;const PreparedDependencies* prepared=nullptr;
        std::optional<DependencyGeometry> geometry;bool geometryLoaded=false;};
    std::map<ObjectKey,Endpoint> endpoints;
    std::vector<InitialCollectionEdge> batch;
    std::vector<EdgeRecord> native;
    int64_t count=0;
    for(;;){
        batch.clear();
        const auto before=sqlite3_total_changes64(db);
        struct ProducerGuard {
            bool& active;
            explicit ProducerGuard(bool& value):active(value){active=true;}
            ~ProducerGuard(){active=false;}
        };
        const bool more=[&]{ProducerGuard guard(t.impl_->producingInitialEdges);return producer(batch);}();
        if(sqlite3_total_changes64(db)!=before)
            throw std::runtime_error("collection: initial edge producer must not mutate the destination");
        if(!more){
            if(!batch.empty())throw std::invalid_argument("collection: initial edge producer returned EOF with records");
            break;
        }
        if(batch.empty())throw std::invalid_argument("collection: initial edge producer returned an empty nonterminal batch");
        if(batch.size()>static_cast<uint64_t>(INT64_MAX-count))throw std::overflow_error("collection: initial edge count overflow");
        native.clear();native.reserve(batch.size());
        if(endpoints.size()>=options.maxCachedEndpoints)endpoints.clear();
        auto resolveEndpoint=[&](const ObjectKey& original,const Anchor& anchor)->Endpoint&{
            // Accepted canonical keys dominate batch input. A cache hit has
            // already been validated; noncanonical spellings still normalize
            // and resolve through exactly the same path on a miss.
            auto it=endpoints.find(original);
            const auto key=it==endpoints.end()?canonical(original):original;
            if(it==endpoints.end())it=endpoints.find(key);
            if(it==endpoints.end()){
                Endpoint endpoint;
                const InitialEndpoint* prepared=nullptr;
                if(impl_->initialEndpoints){auto p=impl_->initialEndpoints->find(key);if(p!=impl_->initialEndpoints->end())prepared=&p->second;}
                if(prepared){
                    if(prepared->retired)throw std::runtime_error("collection: retired initial endpoint");
                    endpoint.uuid=key.objectUuid;
                    endpoint.prepared=prepared->geometry?&*prepared->geometry:nullptr;endpoint.geometryLoaded=true;
                }else{
                    auto current=impl_->locate(t,key);
                    if(!current||current->record.retired)throw std::runtime_error("collection: missing/deleted/retired endpoint");
                    endpoint.uuid=key.objectUuid;
                }
                it=endpoints.emplace(key,std::move(endpoint)).first;
            }
            auto& endpoint=it->second;
            if(!anchor.wholeGeometry||!anchor.points.empty()){
                if(endpoint.prepared)validateDependencies(anchor,*endpoint.prepared);
                else{
                    if(!endpoint.geometryLoaded){
                        if(auto g=impl_->geometry(t,key))endpoint.geometry=dependencyGeometry(*impl_->nodes[g->first].store,impl_->txn(t,g->first),g->second);
                        endpoint.geometryLoaded=true;
                    }
                    if(endpoint.geometry)validateDependencies(anchor,*endpoint.geometry);
                }
            }
            return endpoint;
        };
        for(auto& input:batch){
            auto& a=resolveEndpoint(input.a,input.record.aAnchor);
            auto& b=resolveEndpoint(input.b,input.record.bAnchor);
            input.record.aUuid=a.uuid;input.record.bUuid=b.uuid;
            native.push_back(std::move(input.record));
        }
        writer.appendResolved(native);
        count+=static_cast<int64_t>(native.size());
    }
    writer.finish();
    save.release();
    return count;
}
std::vector<InitialObjectIdentity> ObjectCollection::importInitialObjects(
    CollectionTxn& t,std::span<const InitialObjectRecord> records)
{
    auto& w=impl_->writer(t);auto* db=impl_->db(t,0);
    if(historyEnabled(db))throw std::runtime_error("initial objects require history-off storage");
    Statement edges(db,"SELECT 1 FROM edges LIMIT 1");
    if(edges.row())throw std::runtime_error("initial owned objects must precede local edges");
    Savepoint save(db);
    const auto inserted=detail::insertInitialObjects(*impl_->nodes[0].store,w,records);
    std::vector<InitialObjectIdentity> result;result.reserve(inserted.size());
    for(const auto& [uuid,id]:inserted)result.push_back({{impl_->nodes[0].member.storeUuid,uuid},id});
    save.release();return result;
}
void ObjectCollection::removeEdge(CollectionTxn& t,const ObjectKey& input){
    const auto key=canonical(input);auto& w=impl_->writer(t);Savepoint save(impl_->db(t,0));
    const auto current=edge(t,key);if(!current)throw std::runtime_error("collection: edge absent");
    if(!impl_->nodes[0].store->edgeByUuid(w,key.objectUuid)){
        auto copy=current->record;copy.aUuid=current->a.objectUuid;copy.bUuid=current->b.objectUuid;
        impl_->nodes[0].store->addEdge(w,copy);
    }
    impl_->nodes[0].store->removeEdge(w,key.objectUuid);
    save.release();
}
void ObjectCollection::seal(CollectionTxn& t){impl_->nodes[0].store->seal(impl_->writer(t));}
}
