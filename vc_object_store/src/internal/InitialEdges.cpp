#include "internal/InitialEdges.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/History.hpp"
#include "geometry/Dependencies.hpp"
#include "geometry/GeometryInternal.hpp"
#include "internal/ImmutableImport.hpp"
#include <limits>

namespace vc::objects::detail {
struct InitialEdgeInserter::Impl {
    ObjectStore& store;
    WriteTxn& txn;
    sqlite3* db;
    size_t limit;
    size_t sqlBatchRows;
    bool failed=false,finished=false;
    bool immutable=false,hasRows=false;
    std::string author,host;
    std::vector<std::string> indexes;
    struct Endpoint {
        int64_t id=0;
        bool retired=false;
        std::optional<std::string> geometry;
        std::optional<DependencyGeometry> dependencies;
    };
    std::unordered_map<std::string,std::shared_ptr<Endpoint>> endpoints;

    Impl(ObjectStore& s,WriteTxn& t,bool defer,size_t maximum,size_t sqlRows)
        :store(s),txn(t),db(ObjectStoreImplAccess::conn(t)),limit(maximum),sqlBatchRows(sqlRows),author(s.author()),host(s.hostname())
    {
        immutable=immutableInitial(db);
        if(historyEnabled(db)) throw std::runtime_error("initial edges require history-off storage");
        if(!limit) throw std::invalid_argument("initial edges require a positive endpoint cache limit");
        if(!sqlBatchRows) throw std::invalid_argument("initial edges require a positive SQL batch size");
        { Statement edges(db,"SELECT 1 FROM edges LIMIT 1");
          if(edges.row()) throw std::runtime_error("initial edges require an empty edge table"); }
        if(defer) {
            // Preserve exact existing definitions and leave extension indexes,
            // uniqueness, primary keys and all foreign keys online.
            std::vector<std::string> deferred;
            for(const auto* name:{"edges_a","edges_b","edge_tags_tag","edges_kind_approval_uuid","edges_kind_uuid","edges_kind_id"}) {
                Statement index(db,"SELECT sql FROM sqlite_schema s WHERE type='index' AND name=? "
                    "AND NOT EXISTS(SELECT 1 FROM pragma_index_list(s.tbl_name) p WHERE p.name=s.name AND p.[unique]!=0)",std::string(name));
                if(index.row()) {indexes.push_back(index.text(0));deferred.emplace_back(name);}
            }
            for(const auto& name:deferred)
                exec(db,(std::string("DROP INDEX IF EXISTS ")+name).c_str());
        }
    }
    void usable() const {
        if(failed) throw std::runtime_error("initial edge stream failed; rollback required");
        if(finished) throw std::runtime_error("initial edge stream already finished");
    }
    std::shared_ptr<Endpoint> endpoint(const std::string& uuid) {
        if(const auto found=endpoints.find(uuid);found!=endpoints.end()) return found->second;
        Statement row(db,"SELECT o.id,coalesce(l.retired,0) "
            "FROM objects o LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE o.uuid=?",parseUuid(uuid));
        if(!row.row()) throw std::runtime_error("object store: edge endpoint not found: "+uuid);
        auto value=std::make_shared<Endpoint>();value->id=row.i64(0);value->retired=row.i64(1)!=0;
        if(value->retired) throw std::runtime_error("initial edge endpoint is retired: "+uuid);
        value->geometry=geometryHead(db,uuid);
        if(endpoints.size()>=limit) endpoints.clear();
        endpoints.emplace(uuid,value);return value;
    }
    void validate(Endpoint& endpoint,const Anchor& anchor) {
        if(!endpoint.geometry||(anchor.wholeGeometry&&anchor.points.empty())) return;
        if(!endpoint.dependencies) {
            const auto revision=store.geometryRevision(txn,*endpoint.geometry);
            if(!revision) throw std::runtime_error("initial edge geometry head is missing");
            endpoint.dependencies=dependencyGeometry(store,txn,*revision);
        }
        validateDependencies(anchor,*endpoint.dependencies);
    }
    int64_t reserve(size_t count) {
        Statement row(db,"SELECT next_id,upper_bound FROM key_allocators WHERE name='edges'");
        if(!row.row()||row.isNull(0)) throw std::runtime_error("object store: reserved edges ID range exhausted");
        const auto first=row.i64(0),last=row.i64(1);
        if(count>uint64_t(last-first)+1) throw std::runtime_error("object store: reserved edges ID range exhausted");
        const auto end=first+int64_t(count-1);
        execute(db,"UPDATE key_allocators SET next_id=? WHERE name='edges'",
            end==last?std::optional<int64_t>{}:std::optional<int64_t>{end+1});
        return first;
    }
};

InitialEdgeInserter::InitialEdgeInserter(ObjectStore& store,WriteTxn& txn,bool defer,size_t maximum,size_t sqlRows)
    :impl_(std::make_unique<Impl>(store,txn,defer,maximum,sqlRows)) {}
InitialEdgeInserter::~InitialEdgeInserter()=default;
void InitialEdgeInserter::invalidateEndpoint(const std::string& uuid) {
    impl_->usable();impl_->endpoints.erase(formatUuid(parseUuid(uuid)));
}
void InitialEdgeInserter::append(std::span<const EdgeRecord> edges) {
    auto& p=*impl_;p.usable();
    try {
        if(edges.empty()) return;
        GeometryPhase phase("native.initial_edges.append");
        const auto first=p.reserve(edges.size()),generation=writeGeneration(p.db);
        const auto now=nowIsoUtc();
        std::vector<EdgeRecord> prepared;prepared.reserve(edges.size());
        std::vector<EdgeInsertRow> rows;rows.reserve(edges.size());
        { GeometryPhase preparation("initial_edges.prepare_and_resolve");
        for(size_t i=0;i<edges.size();++i) {
            auto edge=edges[i];const auto binaryUuid=normalizeNewEdge(edge);
            auto a=p.endpoint(edge.aUuid),b=p.endpoint(edge.bUuid);
            if(a->id>b->id) {std::swap(a,b);swapEndpoints(edge);}
            p.validate(*a,edge.aAnchor);p.validate(*b,edge.bAnchor);
            const auto id=first+int64_t(i);
            prepared.push_back(std::move(edge));
            rows.push_back({&prepared.back(),id,a->id,b->id,binaryUuid});
        }
        }
        if(!p.immutable) { GeometryPhase phase("initial_edges.uuid_collision_check");
          requireUnusedObjectUuids(p.db,rows,256); }
        { GeometryPhase phase("initial_edges.edge_rows");
          insertEdgeRows(p.db,rows,generation,p.author,p.host,now,false,p.sqlBatchRows); }
        p.hasRows=true;
        if(!p.immutable)for(const auto& edge:prepared)recordRelationshipValidation(p.db,edge);
    } catch(...) {p.failed=true;throw;}
}
void InitialEdgeInserter::finish() {
    auto& p=*impl_;p.usable();
    try {
        if(p.immutable&&p.hasRows) {
            GeometryPhase phase("initial_edges.finish_uuid_check");
            // Only private, unpublished construction may defer this check.
            // Walk endpoint namespaces and probe the always-live edge UUID
            // index, rather than probing both namespaces for every input row.
            Statement collision(p.db,
                "SELECT 1 FROM objects o CROSS JOIN edges e ON e.uuid=o.uuid UNION ALL "
                "SELECT 1 FROM external_object_keys k CROSS JOIN edges e ON e.uuid=k.uuid LIMIT 1");
            if(collision.row())throw std::runtime_error("object store: UUID already belongs to an object or endpoint key");
        }
        {GeometryPhase phase("initial_edges.finish_indexes");
            for(const auto& sql:p.indexes) exec(p.db,sql.c_str());}
        p.finished=true;p.endpoints.clear();}
    catch(...) {p.failed=true;throw;}
}
void InitialEdgeInserter::appendResolved(std::span<EdgeRecord> input,std::span<const std::pair<int64_t,int64_t>> endpoints) {
    auto& p=*impl_;p.usable();
    try {
        if(input.size()!=endpoints.size())throw std::logic_error("resolved edge/endpoint count mismatch");
        if(input.empty())return;
        GeometryPhase phase("native.initial_edges.append_resolved");
        const auto first=p.reserve(input.size()),generation=writeGeneration(p.db);
        std::vector<EdgeInsertRow> rows;rows.reserve(input.size());
        {GeometryPhase preparation("initial_edges.prepare_resolved");
        for(size_t i=0;i<input.size();++i){
            auto& rec=input[i];bool swapped=false;
            const auto binaryUuid=normalizeNewEdge(rec,&swapped);
            auto [a,b]=endpoints[i];
            if(swapped)std::swap(a,b);
            if(a>b){std::swap(a,b);swapEndpoints(rec);}
            rows.push_back({&rec,first+int64_t(i),a,b,binaryUuid});
        }}
        if(!p.immutable){GeometryPhase collision("initial_edges.uuid_collision_check");
            requireUnusedObjectUuids(p.db,rows,256);}
        {GeometryPhase insertion("initial_edges.edge_rows");
            insertEdgeRows(p.db,rows,generation,p.author,p.host,nowIsoUtc(),false,p.sqlBatchRows);}
        p.hasRows=true;
        if(!p.immutable)for(const auto& edge:input)recordRelationshipValidation(p.db,edge);
    }catch(...){p.failed=true;throw;}
}
}
