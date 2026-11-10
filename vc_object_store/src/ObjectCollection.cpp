#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/InitialEdges.hpp"
#include "internal/InitialObjects.hpp"
#include "internal/ImmutableImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "internal/RecordSql.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/RecordJson.hpp"
#include "internal/History.hpp"
#include "internal/Versions.hpp"
#include "internal/CollectionReads.hpp"
#include "geometry/SharedGeometry.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometryInternal.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/Artifacts.hpp"
#include "internal/Publication.hpp"
#include "internal/Extensions.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/DatabaseCopy.hpp"
#include <variant>
#include <limits>

namespace vc::objects {
using namespace detail;
std::size_t uuidShardIndex(const std::string& uuid,std::size_t shardCount) {
    if(!shardCount)throw std::invalid_argument("collection: UUID shard count must be positive");
    uint64_t hash=UINT64_C(14695981039346656037);
    for(const auto byte:parseUuid(uuid)){hash^=byte;hash*=UINT64_C(1099511628211);}
    return hash%shardCount;
}
bool KeyRoute::matchesUuid(const std::string& uuid) const {
    if(kind!=RoutingKind::UuidHashV1)return false;
    return uuidShardIndex(uuid,divisor)==remainder;
}
bool KeyRoute::matches(int64_t id) const {
    if(kind==RoutingKind::UuidHashV1)return false;
    if(id<lower||id>upper)return false;
    if(kind==RoutingKind::Range)return true;
    if(divisor==0||remainder>=divisor||shift>62)throw std::invalid_argument("collection: invalid routing rule");
    uint64_t value=uint64_t(id);
    if(kind==RoutingKind::Bits)value>>=shift;
    if(kind==RoutingKind::HashV1){value=(value^(value>>30))*UINT64_C(0xbf58476d1ce4e5b9);value=(value^(value>>27))*UINT64_C(0x94d049bb133111eb);value^=value>>31;}
    return value%divisor==remainder;
}
namespace {
OpenOptions readonlyOptions(OpenOptions options) {
    options.readOnly=true;
    options.serviceOwner=false; // Ownership belongs to the writable attachment, never its sealed dependencies.
    return options;
}
bool hasCollection(sqlite3* db) {
    Statement s(db,"SELECT 1 FROM store_required_features WHERE name='collection-v1'"); return s.row();
}
void collectionSchema(sqlite3* db) {
    exec(db,R"SQL(
CREATE TABLE collection_members(
 ordinal INTEGER PRIMARY KEY, database_uuid BLOB NOT NULL UNIQUE CHECK(length(database_uuid)=16),
 store_uuid BLOB NOT NULL CHECK(length(store_uuid)=16), generation BLOB NOT NULL CHECK(length(generation)=16),
 commit_sequence INTEGER NOT NULL CHECK(commit_sequence>0),
 locator TEXT NOT NULL, partition INTEGER NOT NULL CHECK(partition IN (0,1)),
 object_low INTEGER NOT NULL, object_high INTEGER NOT NULL, edge_low INTEGER NOT NULL, edge_high INTEGER NOT NULL
);
CREATE TABLE collection_routes(
 key_space TEXT NOT NULL CHECK(key_space IN ('objects','edges','objects.uuid','edges.uuid')),
 kind INTEGER NOT NULL CHECK(kind BETWEEN 0 AND 4),
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
    // Resolve external paths lexically. Opening/identifying a packed file here
    // would throw before the configured relocation candidates can be tried.
    return std::filesystem::absolute(std::filesystem::path(ref).is_absolute()?std::filesystem::path(ref):base.parent_path()/ref).lexically_normal();
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
        const auto relative=container.lexically_relative(base.parent_path());
        return (relative.empty()?container:relative).generic_string();
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
    execute(db,"INSERT INTO collection_members VALUES(?,?,?,?,?,?,?,?,?,?,?)",ordinal,parseUuid(m.databaseUuid),
        parseUuid(m.storeUuid),parseUuid(m.generation),m.commitSequence,ref,int64_t(m.partition),objects[0],objects[1],edges[0],edges[1]);
}
std::vector<KeyRoute> rangeRoutes(const std::vector<CollectionMember>& members) {
    std::vector<KeyRoute> routes;routes.reserve(2*members.size());
    for(const auto& m:members)for(const auto* space:{"objects","edges"}) {
        const auto r=std::string_view(space)=="objects"?m.objectIds:m.edgeIds;
        routes.push_back({space,RoutingKind::Range,r[0],r[1],1,0,0,m.databaseUuid});
    }
    return routes;
}
std::vector<KeyRoute> initialRoutes(const std::vector<CollectionMember>& members,bool uuidHashRouting) {
    auto routes=rangeRoutes(members);
    if(uuidHashRouting)for(size_t i=0;i<members.size();++i)for(const auto* space:{"objects.uuid","edges.uuid"})
        routes.push_back({space,RoutingKind::UuidHashV1,1,INT64_MAX,members.size(),i,0,members[i].databaseUuid});
    return routes;
}
// Shared encoding only. Discovery/validation belongs to the caller, not this
// writer: the batch builder already owns every member and reserved range.
void writeShardManifest(ObjectStore& store,const std::filesystem::path& path,
    const std::vector<CollectionMember>& members,const std::vector<KeyRoute>& routes) {
    auto w=store.write();auto* db=ObjectStoreImplAccess::conn(w);collectionSchema(db);
    if(std::any_of(routes.begin(),routes.end(),[](const auto& r){return r.kind==RoutingKind::UuidHashV1;}))
        exec(db,"INSERT INTO store_required_features VALUES('uuid-hash-routing-v1');");
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
        "data_checksum","meta","tags","fiber","surface","track"};
    std::sort(record.tags.begin(),record.tags.end());record.tags.erase(std::unique(record.tags.begin(),record.tags.end()),record.tags.end());
    if(record.meta.empty())record.meta="{}";
    return objectToJson(record,&fields);
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
    for(const auto* space:{"objects.uuid","edges.uuid"}){
        std::vector<const KeyRoute*> group;for(const auto& r:routes)if(r.keySpace==space)group.push_back(&r);
        if(group.empty())continue;
        const auto count=std::count_if(members.begin(),members.end(),[](const auto& m){return m.partition;});
        if(group.size()!=size_t(count))throw std::invalid_argument("collection: incomplete UUID shard routing");
        std::set<uint64_t> remainders;std::set<std::string> destinations;
        for(const auto* rule:group){const auto& r=*rule;
            if(r.kind!=RoutingKind::UuidHashV1||r.lower!=1||r.upper!=INT64_MAX||r.shift!=0||r.divisor!=group.size()||r.remainder>=r.divisor||
               !remainders.insert(r.remainder).second||!destinations.insert(r.databaseUuid).second)
                throw std::invalid_argument("collection: invalid UUID shard routing");
            if(std::none_of(members.begin(),members.end(),[&](const auto& m){return m.partition&&m.databaseUuid==r.databaseUuid;}))
                throw std::invalid_argument("collection: UUID destination is not a partition");
        }
    }
    for(auto& r:routes)if(r.keySpace!="objects"&&r.keySpace!="edges"&&r.keySpace!="objects.uuid"&&r.keySpace!="edges.uuid")throw std::invalid_argument("collection: unknown key space");
}
}

struct ObjectCollection::Impl {
    struct Node { std::unique_ptr<ObjectStore> store; CollectionMember member; bool collection=false; std::vector<KeyRoute> routes; std::vector<std::string> children; };
    std::vector<Node> nodes;
    OpenOptions options;
    // DFS insertion order is explicit precedence: local, then bases in manifest order.
    // Originals are resolved only after considering all overrides/tombstones.
    std::map<std::string,size_t> byDatabase;
    std::map<std::string,std::pair<std::string,int64_t>> pinned;
    std::vector<size_t> uuidReadOrder(const std::string& uuid,const char* space)const{
        std::vector<size_t> order;std::set<size_t> included;std::set<std::pair<size_t,bool>> visited;
        const std::string keySpace=std::string(space)+".uuid";
        std::function<void(size_t,bool)> visit=[&](size_t n,bool local){
            if(!visited.emplace(n,local).second)return;
            if(local&&included.insert(n).second)order.push_back(n);
            std::optional<std::string> routed;
            std::optional<size_t> remainder;
            for(const auto& r:nodes[n].routes)if(r.keySpace==keySpace){
                if(!remainder)remainder=uuidShardIndex(uuid,r.divisor);
                if(*remainder==r.remainder)routed=r.databaseUuid;
            }
            // Sibling partitions are one overlay layer, not ordered bases.
            // Query its selected local row before ANY partition's dependencies;
            // otherwise an earlier unselected shard can expose a base record
            // that incorrectly shadows the selected shard's replacement.
            for(const auto& child:nodes[n].children){const auto i=byDatabase.at(child);
                if(nodes[i].member.partition&&(!routed||*routed==child)&&included.insert(i).second)order.push_back(i);
            }
            for(const auto& child:nodes[n].children){const auto i=byDatabase.at(child);
                // Even an unselected shard can hold store_references to readonly dependencies.
                // Skip its rows, not the logical bases it references.
                visit(i,!nodes[i].member.partition);
            }
        };
        visit(0,true);return order;
    }
    void load(const std::filesystem::path&,bool,const std::optional<CollectionMember>&,std::set<std::string>&,sqlite3* existingRoot=nullptr);
    bool depends(size_t n,size_t target)const{
        std::set<size_t> seen;std::vector<size_t> todo{n};
        while(!todo.empty()){auto at=todo.back();todo.pop_back();if(!seen.insert(at).second)continue;
            for(auto& child:nodes[at].children){auto i=byDatabase.at(child);if(i==target)return true;todo.push_back(i);}}
        return false;
    }
    void installCallbacks(sqlite3* sql,size_t n,const std::function<sqlite3*(size_t)>& get)const{
        // Partitions form one logical store. Its exact-reference scope includes
        // dependencies of every partition, not just those of this connection.
        std::set<size_t> scope;
        std::vector<size_t> pending;
        for(size_t i=0;i<nodes.size();++i)
            if(nodes[i].member.storeUuid==nodes[n].member.storeUuid)pending.push_back(i);
        while(!pending.empty()) {
            const auto i=pending.back();pending.pop_back();
            if(!scope.insert(i).second)continue;
            for(const auto& child:nodes[i].children)pending.push_back(byDatabase.at(child));
        }
        std::vector<size_t> sources;
        for(const auto i:scope)if(i!=n)sources.push_back(i);
        if(sources.empty())return;
        auto* pool=activeStatementPools.at(sql);
        pool->referenceSource=[sources=std::move(sources),get](const std::function<bool(sqlite3*)>& matches)->sqlite3*{
            for(const auto i:sources){
                auto* source=get(i);
                if(matches(source))return source;
            }
            return nullptr;
        };
        pool->objectSource=[this,n,get,scope](const std::string& uuid,bool inherited,bool includeDeleted)->sqlite3*{
            const auto found=locateSourceSql({{},uuid},get,0,[&](size_t i){
                return scope.contains(i)&&(!inherited||i!=n);
            },includeDeleted);
            return found?get(*found):nullptr;
        };
        // Edges and objects have the same UUID/tombstone routing semantics.
        const auto visibleScope=scope;
        pool->edgeSource=[this,n,get,visibleScope](const std::string& uuid,bool inherited,bool includeDeleted)->sqlite3*{
            for(const auto i:uuidReadOrder(uuid,"edges"))if(visibleScope.contains(i)&&(!inherited||i!=n)){
                auto* source=get(i);Statement row(source,"SELECT state FROM edges WHERE uuid=?",parseUuid(uuid));
                if(row.row())return includeDeleted||row.i64(0)!=2?source:nullptr;
            }
            return nullptr;
        };
        pool->querySources=[get,visibleScope]{std::vector<sqlite3*> result;
            for(auto i:visibleScope)result.push_back(get(i));return result;};
    }
    void check(const CollectionTxn&) const;
    const Txn& txn(const CollectionTxn&,size_t) const;
    sqlite3* db(const CollectionTxn& t,size_t n) const {return ObjectStoreImplAccess::conn(txn(t,n));}
    WriteTxn& writer(CollectionTxn&);
    ObjectKey decode(const CollectionTxn&,size_t n,const std::string& uuid) const {
        return {nodes[n].member.storeUuid,uuid};
    }
    std::optional<size_t> locateSourceSql(const ObjectKey&,const std::function<sqlite3*(size_t)>&,size_t start=0,
                                         const std::function<bool(size_t)>& eligible={},bool includeDeleted=false) const;
    ObjectKey insertEdge(CollectionTxn&,const ObjectKey&,const ObjectKey&,const EdgeRecord&,std::optional<int64_t>);
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
            catch(...){pool->referenceSource={};pool->objectSource={};pool->edgeSource={};pool->querySources={};configured.erase(pool);throw;}
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
    auto path=input;auto opt=root?options:readonlyOptions(options);opt.physicalPartitionOnly=true;
    std::unique_ptr<ObjectStore> store;std::unique_ptr<ReadTxn> read;
    if(expected&&!existingRoot){
        // Reuse the successful candidate's connection/snapshot. This is file
        // discovery by the pinned database identity, never per-object routing.
        std::set<std::filesystem::path> attempted;std::string directError;
        const auto attempt=[&](const std::filesystem::path& candidate){
            std::filesystem::path selected;
            try{
                if(isPackedDatabasePath(candidate))selected=candidate;
                else if(isPackedStore(candidate))selected=packedDatabasePath(candidate,expected->databaseUuid);
                else selected=std::filesystem::absolute(candidate).lexically_normal();
                if(!attempted.insert(selected).second)return false;
                auto opened=std::make_unique<ObjectStore>(ObjectStore::open(selected,opt));
                auto snapshot=std::make_unique<ReadTxn>(opened->read());auto* candidateSql=ObjectStoreImplAccess::conn(*snapshot);
                Statement pin(candidateSql,"SELECT s.uuid,d.uuid,m.generation,m.commit_sequence,m.sealed FROM store_identity s,database_identity d,store_manifest m");
                if(!pin.row()||formatUuid(pin.blob16(0))!=expected->storeUuid||formatUuid(pin.blob16(1))!=expected->databaseUuid||
                   formatUuid(pin.blob16(2))!=expected->generation||pin.i64(3)!=expected->commitSequence||pin.i64(4)==0||
                   (expected->partition&&(reservation(candidateSql,"objects")!=expected->objectIds||reservation(candidateSql,"edges")!=expected->edgeIds)))
                    throw std::runtime_error("dependency identity/lineage/commit sequence/sealed policy/reservation mismatch");
                path=std::move(selected);store=std::move(opened);read=std::move(snapshot);return true;
            }catch(const std::bad_alloc&){throw;}
            catch(const std::exception& error){if(directError.empty())directError=error.what();return false;}
        };
        bool found=attempt(input);
        for(const auto& candidate:options.storeReferenceFiles)if(!found)found=attempt(candidate);
        for(const auto& directory:options.storeReferenceDirectories){
            if(found)break;
            std::vector<std::filesystem::path> candidates;std::error_code error;
            for(std::filesystem::directory_iterator it(directory,error),end;!error&&it!=end;it.increment(error)){
                std::error_code statusError;if(it->is_regular_file(statusError)&&!statusError)candidates.push_back(it->path());
            }
            std::sort(candidates.begin(),candidates.end());
            for(const auto& candidate:candidates)if(attempt(candidate)){found=true;break;}
        }
        if(!found)throw std::runtime_error("collection: cannot locate pinned dependency "+expected->databaseUuid+
            " at "+input.string()+" or configured search locations: "+directError);
    }else{
        path=normalized(input);
        if(!existingRoot){store=std::make_unique<ObjectStore>(ObjectStore::open(path,opt));read=std::make_unique<ReadTxn>(store->read());}
    }
    auto* sql=existingRoot?existingRoot:ObjectStoreImplAccess::conn(*read);
    Statement header(sql,"SELECT s.uuid,d.uuid,m.generation,m.commit_sequence,m.sealed FROM store_identity s,database_identity d,store_manifest m");
    if(!header.row())throw std::runtime_error("collection: absent manifest identity");
    DatabaseIdentity id;id.uuid=formatUuid(header.blob16(1));
    CollectionMember member{id.uuid,formatUuid(header.blob16(0)),formatUuid(header.blob16(2)),header.i64(3),path,expected?expected->partition:false,reservation(sql,"objects"),reservation(sql,"edges")};
    if(!root&&header.i64(4)==0)throw std::runtime_error("collection: dependency is not sealed: "+path.string());
    if(expected&&(member.databaseUuid!=expected->databaseUuid||member.storeUuid!=expected->storeUuid||member.generation!=expected->generation||member.commitSequence!=expected->commitSequence||
                  (expected->partition&&(member.objectIds!=expected->objectIds||member.edgeIds!=expected->edgeIds))))
        throw std::runtime_error("collection: dependency identity/lineage/commit sequence/reservation mismatch: "+path.string());
    if(visiting.contains(id.uuid))throw std::runtime_error("collection: cyclic database dependency");
    if(byDatabase.contains(id.uuid))return;
    const auto statePin=std::pair{member.generation,member.commitSequence};
    if(auto it=pinned.find(member.storeUuid);it!=pinned.end()&&it->second!=statePin)
        throw std::runtime_error("collection: inconsistent repeated store lineage or commit sequence");
    pinned[member.storeUuid]=statePin;
    bool composed=hasCollection(sql);std::vector<CollectionMember> children;std::vector<KeyRoute> routes;
    if(composed){
        Statement s(sql,"SELECT database_uuid,store_uuid,generation,commit_sequence,locator,partition,object_low,object_high,edge_low,edge_high FROM collection_members ORDER BY ordinal");
        while(s.row()){
            CollectionMember child{formatUuid(s.blob16(0)),formatUuid(s.blob16(1)),formatUuid(s.blob16(2)),s.i64(3),resolve(path,s.text(4)),s.i64(5)!=0,{s.i64(6),s.i64(7)},{s.i64(8),s.i64(9)}};
            // load() identifies external packed files and selects the pinned
            // physical member while trying the recorded/search locators.
            if(child.partition?(child.storeUuid!=member.storeUuid||child.generation!=member.generation||child.commitSequence!=member.commitSequence):
                               (child.storeUuid==member.storeUuid||child.commitSequence>=member.commitSequence))
                throw std::runtime_error("collection: invalid dependency lineage/commit sequence/ownership ordering");
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
    StatementPool* borrowedPool=nullptr;
    std::function<sqlite3*(const std::function<bool(sqlite3*)>&)> previousReferences;
    std::function<sqlite3*(const std::string&,bool,bool)> previousObjects;
    std::function<sqlite3*(const std::string&,bool,bool)> previousEdges;
    std::function<std::vector<sqlite3*>()> previousQueries;
    std::map<size_t,std::unique_ptr<ReadTxn>> readers;
    std::set<StatementPool*> configured;
    CollectionReadContext(sqlite3* sql,const std::filesystem::path& path,const OpenOptions& options):root(sql){
        collection.options=readonlyOptions(options);std::set<std::string> visiting;collection.load(path,true,{},visiting,sql);
        if(const auto found=activeStatementPools.find(root);found!=activeStatementPools.end()) {
            borrowedPool=found->second;previousReferences=borrowedPool->referenceSource;previousObjects=borrowedPool->objectSource;
            previousEdges=borrowedPool->edgeSource;previousQueries=borrowedPool->querySources;
        } else {
            rootPool=std::make_unique<StatementPool>();rootPool->historyEnabled=historyEnabled(root);rootPool->spatialEnabled=spatialEnabled(root);
            activeStatementPools[root]=rootPool.get();
        }
        try{get(0);}catch(...){detach();throw;}
    }
    void detach(){
        const auto active=activeStatementPools.find(root);
        const bool borrowedActive=active!=activeStatementPools.end()&&active->second==borrowedPool;
        for(auto* p:configured){
            if(p==borrowedPool&&!borrowedActive)continue; // Lease already returned by commit/rollback.
            p->geometryCache.reset();p->referenceSource={};p->objectSource={};p->edgeSource={};p->querySources={};
        }
        if(rootPool)activeStatementPools.erase(root);
        else if(const auto found=activeStatementPools.find(root);found!=activeStatementPools.end()&&found->second==borrowedPool){
            borrowedPool->referenceSource=std::move(previousReferences);borrowedPool->objectSource=std::move(previousObjects);
            borrowedPool->edgeSource=std::move(previousEdges);borrowedPool->querySources=std::move(previousQueries);
        }
    }
    ~CollectionReadContext(){detach();}
    sqlite3* get(size_t n){
        sqlite3* sql=root;
        if(n){auto& r=readers[n];if(!r)r=std::make_unique<ReadTxn>(collection.nodes[n].store->read());sql=ObjectStoreImplAccess::conn(*r);}
        auto* pool=activeStatementPools.at(sql);
        if(configured.insert(pool).second){
            try{collection.installCallbacks(sql,n,[this](size_t i){return get(i);});}
            catch(...){pool->referenceSource={};pool->objectSource={};pool->edgeSource={};pool->querySources={};configured.erase(pool);throw;}
        }
        return sql;
    }
};
std::shared_ptr<void> detail::attachCollectionReferenceResolver(sqlite3* db,const std::filesystem::path& path){
    return attachCollectionReferenceResolver(db,path,OpenOptions{});
}
std::shared_ptr<void> detail::attachCollectionReferenceResolver(sqlite3* db,const std::filesystem::path& path,const OpenOptions& options){
    return hasCollection(db)?std::make_shared<CollectionReadContext>(db,path,options):std::shared_ptr<void>{};
}
std::optional<size_t> ObjectCollection::Impl::locateSourceSql(const ObjectKey& key,
    const std::function<sqlite3*(size_t)>& get,size_t start,const std::function<bool(size_t)>& eligible,bool includeDeleted)const{
    // UUID is the complete key. The first present row wins, including deletion.
    for(const auto n:uuidReadOrder(key.objectUuid,"objects"))if(n>=start&&(!eligible||eligible(n))){
        auto* sql=get(n);
        Statement state(sql,"SELECT state FROM objects WHERE uuid=?",parseUuid(key.objectUuid));
        if(!state.row())continue;
        if(state.i64(0)==2&&!includeDeleted)return {};
        return n;
    }
    return {};
}

ObjectCollection::ObjectCollection(std::unique_ptr<Impl> p):impl_(std::move(p)){}
ObjectCollection::ObjectCollection(ObjectCollection&&) noexcept=default;
ObjectCollection::~ObjectCollection()=default;
ObjectCollection ObjectCollection::open(const std::filesystem::path& path,const OpenOptions& options){
    auto p=std::make_unique<Impl>();p->options=options;std::set<std::string> visiting;p->load(path,true,{},visiting);return ObjectCollection(std::move(p));
}
ObjectCollection ObjectCollection::create(const std::filesystem::path& path,const std::vector<std::filesystem::path>& bases,const OpenOptions& options){
    std::vector<CollectionMember> members;int64_t commitSequence=options.initialCommitSequence;
    const auto readOptions=readonlyOptions(options);
    for(const auto& base:bases){auto c=open(base,readOptions);auto t=c.read();auto p=c.provisioning(t);
        if(!p.sealed)throw std::runtime_error("collection: base must be sealed");
        if(p.commitSequence>=INT64_MAX-1)throw std::runtime_error("collection: dependency commit sequence exhausted");
        commitSequence=std::max(commitSequence,p.commitSequence);members.push_back(c.impl_->nodes[0].member);
    }
    auto opt=options;opt.initialCommitSequence=std::max(opt.initialCommitSequence,commitSequence);opt.physicalPartitionOnly=true;
    StagedDatabase staged(path,isBuildingPackedDatabasePath(path));
    {auto store=ObjectStore::create(staged.path,opt);auto w=store.write();auto* db=ObjectStoreImplAccess::conn(w);collectionSchema(db);
     int64_t i=0;for(const auto& m:members){if(m.storeUuid==store.uuid(w))throw std::runtime_error("collection: overlay needs an independent store UUID");
         insertMember(db,i++,m,locator(normalized(path),m.locator));}
     w.commit();store.checkpoint();}
    {auto verified=open(staged.path,readOptions);}
    staged.publish();
    return open(path,options);
}
ObjectCollection detail::InitialCollectionAccess::create(const std::filesystem::path& path,
    const std::vector<std::filesystem::path>& bases,const OpenOptions& options)
{
    // This path is private to the initial builder, which owns staging and
    // publication. No intermediate writable store is published or reopened.
    std::vector<CollectionMember> members;int64_t commitSequence=options.initialCommitSequence;
    const auto readOptions=readonlyOptions(options);
    for(const auto& base:bases){auto c=ObjectCollection::open(base,readOptions);auto r=c.read();
        const auto p=c.provisioning(r);if(!p.sealed||p.commitSequence>=INT64_MAX-1)
            throw std::runtime_error("initial collection needs sealed, non-exhausted dependencies");
        commitSequence=std::max(commitSequence,p.commitSequence);members.push_back(c.impl_->nodes[0].member);}
    auto opt=options;opt.initialCommitSequence=commitSequence;opt.physicalPartitionOnly=true;
    auto store=std::make_unique<ObjectStore>(ObjectStoreImplAccess::createInitial(path,opt));
    auto p=std::make_unique<ObjectCollection::Impl>();p->options=opt;
    {auto w=store->write();auto* db=ObjectStoreImplAccess::conn(w);collectionSchema(db);
     int64_t ordinal=0;for(const auto& m:members){
        if(m.storeUuid==store->uuid(w))throw std::runtime_error("initial collection needs independent dependency ownership");
        insertMember(db,ordinal++,m,locator(normalized(path),m.locator));}
     CollectionMember root{store->databaseIdentity(w).uuid,store->uuid(w),activeGeneration(db),writeCommitSequence(db),path,false,opt.objectIdRange,opt.edgeIdRange};
     std::vector<std::string> children;for(const auto& m:members)children.push_back(m.databaseUuid);
     p->byDatabase.emplace(root.databaseUuid,0);p->pinned.emplace(root.storeUuid,std::pair{root.generation,root.commitSequence});
     p->nodes.push_back({nullptr,std::move(root),true,{},std::move(children)});
     w.commit();}
    // No reader pool or collection loading for the database just constructed.
    // Pre-existing external dependencies still require their ordinary readers.
    std::set<std::string> visiting{p->nodes[0].member.databaseUuid};
    for(const auto& m:members)p->load(m.locator,false,m,visiting);
    p->nodes[0].store=std::move(store);return ObjectCollection(std::move(p));
}
CollectionMember detail::InitialCollectionAccess::sealShard(ObjectCollection& collection,CollectionTxn& transaction) {
    collection.seal(transaction);
    auto member=collection.impl_->nodes[0].member;
    member.commitSequence=writeCommitSequence(collection.impl_->db(transaction,0));
    member.partition=true;
    transaction.commit();collection.checkpoint();
    return member;
}
void detail::InitialCollectionAccess::writeManifest(const std::filesystem::path& path,
    const std::vector<CollectionMember>& members,const OpenOptions& options,bool uuidHashRouting) {
    const auto& first=members.at(0);
    auto opt=options;opt.history=HistoryMode::Off;opt.spatialIndex=false;
    opt.creationStoreUuid=first.storeUuid;opt.creationGeneration=first.generation;
    opt.initialCommitSequence=first.commitSequence-1;opt.physicalPartitionOnly=true;
    auto store=ObjectStoreImplAccess::createInitial(path,opt);
    writeShardManifest(store,path,members,initialRoutes(members,uuidHashRouting));
}
ObjectCollection ObjectCollection::publishShards(const std::filesystem::path& path,const std::vector<std::filesystem::path>& shards,const OpenOptions& options,const std::vector<KeyRoute>& requestedRoutes){
    if(shards.empty())throw std::runtime_error("collection: empty shard list");
    const auto readOptions=readonlyOptions(options);
    std::vector<CollectionMember> members;for(const auto& shard:shards){
        auto c=open(shard,readOptions);auto t=c.read();if(!c.provisioning(t).sealed)throw std::runtime_error("collection: unpublished shard");
        // A physical partition may itself have immutable dependencies (for
        // example relation-only shards referencing a geometry collection).
        // Its own content still resides in one independent SQLite database.
        auto m=c.impl_->nodes[0].member;m.partition=true;members.push_back(m);
    }
    const auto& first=members.front();if(first.commitSequence<1)throw std::runtime_error("collection: invalid shard commit sequence");
    for(auto& m:members)if(m.storeUuid!=first.storeUuid||m.generation!=first.generation||m.commitSequence!=first.commitSequence)
        throw std::runtime_error("collection: shards disagree on ownership/lineage/commit sequence");
    auto routes=requestedRoutes.empty()?rangeRoutes(members):requestedRoutes;
    validateRoutes(routes,members);
    // Custom routing is verified once at publication. Readers only trust the
    // sealed manifest and never rescan all IDs on every open.
    if(!requestedRoutes.empty())for(auto& m:members){auto store=ObjectStore::open(m.locator,OpenOptions{.readOnly=true});auto r=store.read();
        for(const auto* space:{"objects","edges"}){Statement ids(ObjectStoreImplAccess::conn(r),std::string("SELECT id FROM ")+space);
            while(ids.row()){size_t matches=0;for(auto& route:routes)if(route.keySpace==space&&route.matches(ids.i64(0))){if(route.databaseUuid!=m.databaseUuid)throw std::runtime_error("collection: misplaced shard key");++matches;}
                if(matches!=1)throw std::runtime_error("collection: shard key not uniquely routed");}
            const auto uuidSpace=std::string(space)+".uuid";
            if(std::any_of(routes.begin(),routes.end(),[&](const auto& route){return route.keySpace==uuidSpace;})){
                Statement keys(ObjectStoreImplAccess::conn(r),std::string("SELECT uuid FROM ")+space);
                while(keys.row()){size_t matches=0;for(const auto& route:routes)if(route.keySpace==uuidSpace&&route.matchesUuid(formatUuid(keys.blob16(0)))){
                    if(route.databaseUuid!=m.databaseUuid)throw std::runtime_error("collection: misplaced shard UUID");++matches;}
                    if(matches!=1)throw std::runtime_error("collection: shard UUID not uniquely routed");}
            }
        }}
    auto opt=options;opt.history=HistoryMode::Off;opt.spatialIndex=false;opt.creationStoreUuid=first.storeUuid;
    opt.creationGeneration=first.generation;opt.initialCommitSequence=first.commitSequence-1;opt.physicalPartitionOnly=true;
    const bool packed=isBuildingPackedDatabasePath(path);
    StagedDatabase staged(path,packed);const auto& stage=staged.path;
    {auto store=ObjectStore::create(stage,opt);writeShardManifest(store,normalized(path),members,routes);}
    // Validate all targets and the routing map before the publication name exists.
    {auto verified=open(stage,readOptions);}
    staged.publish();
    return open(path,readonlyOptions(options));
}
CollectionTxn ObjectCollection::read(){return CollectionTxn(std::make_unique<CollectionTxn::Impl>(impl_.get(),false));}
void ObjectCollection::checkpoint(){impl_->nodes[0].store->checkpoint();}
CollectionTxn ObjectCollection::write(){if(!impl_->nodes[0].collection)throw std::runtime_error("collection: use ObjectStore to edit a plain store");return CollectionTxn(std::make_unique<CollectionTxn::Impl>(impl_.get(),true));}
std::string ObjectCollection::uuid(const CollectionTxn& t)const{impl_->check(t);return impl_->nodes[0].member.storeUuid;}
StoreProvisioning ObjectCollection::provisioning(const CollectionTxn& t)const{return impl_->nodes[0].store->provisioning(impl_->txn(t,0));}
void ObjectCollection::createExtensionTable(CollectionTxn& t,const ExtensionTable& table){
    auto& w=impl_->writer(t);impl_->nodes[0].store->createExtensionTable(w,table);
}
void ObjectCollection::replaceExtensionRows(CollectionTxn& t,const std::string& table,const ObjectKey& input,
                                             std::span<const ExtensionRow> rows){
    impl_->nodes[0].store->replaceExtensionRows(impl_->writer(t),table,canonical(input).objectUuid,rows);
}
std::vector<ExtensionTableInfo> ObjectCollection::extensionTables(const CollectionTxn& t)const{
    return impl_->nodes[0].store->extensionTables(impl_->txn(t,0));
}
void ObjectCollection::scanExtensionRows(const CollectionTxn& t,const std::string& name,
    const std::function<bool(const ExtensionRow&)>& visitor,std::span<const ExtensionValue> prefix)const{
    impl_->nodes[0].store->scanExtensionRows(impl_->txn(t,0),name,visitor,prefix);
}
std::vector<CollectionMember> ObjectCollection::members()const{std::vector<CollectionMember> r;for(const auto& n:impl_->nodes)r.push_back(n.member);return r;}

void copyObjectCollection(const std::filesystem::path& source, const std::filesystem::path& destination,
                          const OpenOptions& options)
{
    namespace fs=std::filesystem;
    const auto target=fs::absolute(destination).lexically_normal(), sidecar=fs::path(target.string()+".stores");
    if(fs::exists(target)||fs::exists(sidecar)||fs::exists(target.string()+".artifacts"))
        throw std::invalid_argument("collection copy destination or sidecar already exists");
    auto collection=ObjectCollection::open(source,readonlyOptions(options));
    auto snapshot=collection.read();
    const auto members=collection.members();
    const auto provisioning=collection.provisioning(snapshot);
    const auto before=provisioning.commitSequence;
    const bool partitioned=std::any_of(members.begin(),members.end(),[&](const auto& m){
        return m.partition && m.storeUuid==members.front().storeUuid;
    });
    if(partitioned&&!provisioning.sealed)throw std::runtime_error("partitioned collection root must be sealed");
    if(before==std::numeric_limits<int64_t>::max())throw std::overflow_error("copied store commit sequence exhausted");
    StagedDatabase staged(target,false);
    bool ownsSidecar=false;
    try {
        if(members.size()>1||partitioned){
            if(!fs::create_directory(sidecar))throw std::runtime_error("cannot create collection copy sidecar");
            ownsSidecar=true;
        }
        std::map<std::string,fs::path> paths;
        std::vector<fs::path> artifactSources;
        std::vector<fs::path> artifactSnapshots;
        for(size_t i=0;i<members.size();++i){
            const auto& member=members[i];
            paths[member.databaseUuid]=i==0&&!partitioned?staged.path:sidecar/(member.databaseUuid+".vcobj");
            copySqliteSnapshot(ObjectStoreImplAccess::conn(collection.impl_->txn(snapshot,i)),paths.at(member.databaseUuid));
            artifactSources.push_back(member.locator);
            artifactSnapshots.push_back(paths.at(member.databaseUuid));
        }
        ArtifactExport artifacts(artifactSources,target,artifactSnapshots);
        for(size_t i=0;i<members.size();++i){
            const auto& member=members[i];const auto& path=paths.at(member.databaseUuid);
            std::unique_ptr<sqlite3,decltype(&sqlite3_close_v2)> db(
                openConnection(path,SQLITE_OPEN_READWRITE,options,false),sqlite3_close_v2);
            exec(db.get(),"BEGIN IMMEDIATE; PRAGMA defer_foreign_keys=ON;");
            try {
                relocateCollectionMembers(db.get(),[&](const std::string& id){
                    return fs::relative(paths.at(id),path.parent_path()).generic_string();
                });
                if(i==0&&!partitioned){
                    forkStorageIdentity(db.get());
                    const auto newStore=util::newUuid();
                    exec(db.get(),"DROP TRIGGER store_identity_immutable_update; DROP TRIGGER store_identity_immutable_delete;");
                    execute(db.get(),"UPDATE store_identity SET uuid=?",parseUuid(newStore));
                    execute(db.get(),"UPDATE key_allocators SET namespace_uuid=?",parseUuid(newStore));
                    clearTransactionReceipts(db.get());
                    exec(db.get(),"CREATE TRIGGER store_identity_immutable_update BEFORE UPDATE ON store_identity "
                        "BEGIN SELECT RAISE(ABORT,'store identity is immutable'); END;"
                        "CREATE TRIGGER store_identity_immutable_delete BEFORE DELETE ON store_identity "
                        "BEGIN SELECT RAISE(ABORT,'store identity is retained'); END;");
                    execute(db.get(),"UPDATE store_manifest SET commit_sequence=?,sealed=0 WHERE singleton=1",before+1);
                }
                exec(db.get(),"UPDATE store_write_state SET dirty=0 WHERE singleton=1; COMMIT;");
            } catch(...) {sqlite3_exec(db.get(),"ROLLBACK",nullptr,nullptr,nullptr);throw;}
        }
        if(partitioned){
            auto writable=options;writable.history=provisioning.history;writable.readOnly=false;writable.serviceOwner=false;
            auto overlay=ObjectCollection::create(staged.path,{paths.at(members.front().databaseUuid)},writable);
        }
        {auto verified=ObjectCollection::open(staged.path,readonlyOptions(options));}
        staged.publish();artifacts.commit();
    } catch(...) {
        if(ownsSidecar){std::error_code ignored;fs::remove_all(sidecar,ignored);}
        throw;
    }
}
std::optional<ObjectRevision> ObjectCollection::objectAtVersion(const CollectionTxn& t,const ObjectVersion& version)const{
    return impl_->nodes[0].store->objectAtVersion(impl_->txn(t,0),version);
}
std::optional<uint64_t> ObjectCollection::artifactLength(const CollectionTxn& t,const std::string& hash)const{
    return detail::resolvedArtifactLength(impl_->db(t,0),hash);
}
std::vector<uint8_t> ObjectCollection::readArtifact(const CollectionTxn& t,const std::string& hash,uint64_t offset,
                                                  const std::optional<uint64_t>& count)const{
    return detail::readResolvedArtifact(impl_->db(t,0),hash,offset,count);
}
std::optional<RelationshipRevision> ObjectCollection::relationshipAtVersion(const CollectionTxn& t,const RelationshipVersion& version)const{
    return impl_->nodes[0].store->relationshipAtVersion(impl_->txn(t,0),version);
}
void ObjectCollection::scanExtensionRowsAtVersion(const CollectionTxn& t,const std::string& table,const ObjectVersion& version,
    const std::function<bool(const ExtensionRow&)>& visitor)const{
    impl_->nodes[0].store->scanExtensionRowsAtVersion(impl_->txn(t,0),table,version,visitor);
}
void ObjectCollection::scanExtensionRowsAtVersion(const CollectionTxn& t,const std::string& table,const RelationshipVersion& version,
    const std::function<bool(const ExtensionRow&)>& visitor)const{
    impl_->nodes[0].store->scanExtensionRowsAtVersion(impl_->txn(t,0),table,version,visitor);
}
std::vector<uint8_t> ObjectCollection::readGeometryArrayAtVersion(const CollectionTxn& t,const ObjectVersion& version,const std::string& array)const{
    const auto record=objectAtVersion(t,version);
    if(!record||record->deleted||!record->geometryRevision)throw std::invalid_argument("collection: geometry version unavailable");
    return impl_->nodes[0].store->readGeometryArray(impl_->txn(t,0),*record->geometryRevision,array);
}
std::optional<CollectionObject> ObjectCollection::object(const CollectionTxn& t,const ObjectKey& input)const{
    const auto key=canonical(input);
    auto record=impl_->nodes[0].store->objectByUuid(impl_->txn(t,0),key.objectUuid);
    return record?std::optional(CollectionObject{key,std::move(*record)}):std::nullopt;
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
    ObjectSelection selection;selection.kind=kind;selection.limit=4096;
    while(true){const auto rows=impl_->nodes[0].store->selectObjects(impl_->txn(t,0),selection);if(rows.empty())break;
        for(const auto& row:rows){const auto resolved=object(t,{uuid(t),row.uuid});if(resolved&&!visit(*resolved))return;}
        selection.afterUuid=rows.back().uuid;}
}
std::optional<GeometryRevision> ObjectCollection::currentGeometry(const CollectionTxn& t,const ObjectKey& key)const{
    return impl_->nodes[0].store->currentGeometry(impl_->txn(t,0),canonical(key).objectUuid);
}
std::vector<uint8_t> ObjectCollection::readGeometryArray(const CollectionTxn& t,const ObjectKey& key,const std::string& array)const{
    const auto geometry=currentGeometry(t,key);if(!geometry)throw std::runtime_error("collection: geometry absent");
    return impl_->nodes[0].store->readGeometryArray(impl_->txn(t,0),geometry->uuid,array);
}
std::string ObjectCollection::putGeometry(CollectionTxn& t,const ObjectKey& input,const GeometryData& data){
    return impl_->nodes[0].store->putGeometry(impl_->writer(t),canonical(input).objectUuid,data);
}
std::string ObjectCollection::editGeometry(CollectionTxn& t,const ObjectKey& input,const std::vector<GeometryArrayEdit>& edits,const std::string& expected,const std::vector<GeometryEdgeEdit>& decisions){
    return impl_->nodes[0].store->editGeometry(impl_->writer(t),canonical(input).objectUuid,edits,expected,decisions);
}
std::vector<CollectionEdge> ObjectCollection::edgesOf(const CollectionTxn& t,const ObjectKey& input)const{
    std::vector<CollectionEdge> out;
    for(const auto& row:impl_->nodes[0].store->edgesOf(impl_->txn(t,0),canonical(input).objectUuid))
        if(auto current=edge(t,{uuid(t),row.uuid}))out.push_back(std::move(*current));
    return out;
}
std::vector<CollectionEdge> ObjectCollection::edgesByKind(const CollectionTxn& t,const std::string& kind)const{
    std::vector<CollectionEdge> out;
    for(const auto& row:impl_->nodes[0].store->edgesByKind(impl_->txn(t,0),kind))
        if(auto current=edge(t,{uuid(t),row.uuid}))out.push_back(std::move(*current));
    return out;
}
std::optional<CollectionEdge> ObjectCollection::edge(const CollectionTxn& t,const ObjectKey& input)const{
    auto resolved=resolveEdge(impl_->db(t,0),canonical(input).objectUuid);if(!resolved)return {};
    const auto owner=storeUuidText(resolved->source);
    CollectionEdge result{{owner,resolved->record.uuid},{owner,resolved->record.aUuid},
        {owner,resolved->record.bUuid},std::move(resolved->record)};
    const auto a=object(t,result.a),b=object(t,result.b);
    return a&&b&&!a->record.retired&&!b->record.retired?std::optional(std::move(result)):std::nullopt;
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
    impl_->nodes[0].store->upsertObject(w,proposed);
    save.release();
}
void ObjectCollection::setMeta(CollectionTxn& t,const ObjectKey& input,const std::string& field,const std::string& json){
    impl_->nodes[0].store->setMeta(impl_->writer(t),canonical(input).objectUuid,field,json);
}
void ObjectCollection::deleteObject(CollectionTxn& t,const ObjectKey& input){
    impl_->nodes[0].store->deleteObject(impl_->writer(t),canonical(input).objectUuid);
}
ObjectKey ObjectCollection::Impl::insertEdge(CollectionTxn& t,const ObjectKey& a,const ObjectKey& b,const EdgeRecord& input,std::optional<int64_t> id){
    GeometryPhase phase("collection.edge");
    auto& w=writer(t);Savepoint save(db(t,0));
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
    if(!producer)
        throw std::invalid_argument("collection: initial edge import requires a producer");
    auto& w=impl_->writer(t);auto* db=impl_->db(t,0);
    Savepoint save(db);
    detail::InitialEdgeInserter writer(*impl_->nodes[0].store,w,options.deferIndexes,options.sqlBatchRows);
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
        for(auto& input:batch){
            input.record.aUuid=input.a.objectUuid;input.record.bUuid=input.b.objectUuid;
            native.push_back(std::move(input.record));
        }
        writer.append(native);
        std::vector<const InitialExtensionRow*> extensions;
        for(const auto& input:batch)for(const auto& row:input.extensions)extensions.push_back(&row);
        detail::appendInitialExtensions(db,extensions);
        count+=static_cast<int64_t>(native.size());
    }
    writer.finish();
    save.release();
    return count;
}
std::vector<InitialObjectIdentity> ObjectCollection::importInitialObjects(
    CollectionTxn& t,std::span<const InitialObjectRecord> records)
{
    std::vector<const InitialObjectRecord*> refs;refs.reserve(records.size());
    for(const auto& record:records)refs.push_back(&record);
    return importInitialObjectRefs(t,refs);
}
std::vector<InitialObjectIdentity> ObjectCollection::importInitialObjectRefs(
    CollectionTxn& t,std::span<const InitialObjectRecord* const> records)
{
    auto& w=impl_->writer(t);auto* db=impl_->db(t,0);
    if(historyEnabled(db))throw std::runtime_error("initial objects require history-off storage");
    Savepoint save(db);
    const auto inserted=detail::insertInitialObjectRefs(*impl_->nodes[0].store,w,records);
    std::vector<const InitialExtensionRow*> extensions;
    for(const auto* record:records)for(const auto& row:record->extensions)extensions.push_back(&row);
    detail::appendInitialExtensions(db,extensions);
    std::vector<InitialObjectIdentity> result;result.reserve(inserted.size());
    for(const auto& [uuid,id]:inserted)result.push_back({{impl_->nodes[0].member.storeUuid,uuid},id});
    save.release();return result;
}
void ObjectCollection::removeEdge(CollectionTxn& t,const ObjectKey& input){
    impl_->nodes[0].store->removeEdge(impl_->writer(t),canonical(input).objectUuid);
}
void ObjectCollection::seal(CollectionTxn& t){impl_->nodes[0].store->seal(impl_->writer(t));}
}
