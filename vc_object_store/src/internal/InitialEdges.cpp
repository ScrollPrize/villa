#include "internal/InitialEdges.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include <limits>

namespace vc::objects::detail {
struct InitialEdgeInserter::Impl {
    sqlite3* db;
    size_t sqlBatchRows;
    bool failed=false,finished=false;
    std::string author,host;
    std::vector<std::string> indexes;
    Impl(ObjectStore& s,WriteTxn& t,bool defer,size_t sqlRows)
        :db(ObjectStoreImplAccess::conn(t)),sqlBatchRows(sqlRows),author(s.author()),host(s.hostname())
    {
        if(historyEnabled(db)) throw std::runtime_error("initial edges require history-off storage");
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

InitialEdgeInserter::InitialEdgeInserter(ObjectStore& store,WriteTxn& txn,bool defer,size_t sqlRows)
    :impl_(std::make_unique<Impl>(store,txn,defer,sqlRows)) {}
InitialEdgeInserter::~InitialEdgeInserter()=default;
void InitialEdgeInserter::finish() {
    auto& p=*impl_;p.usable();
    try {
        {GeometryPhase phase("initial_edges.finish_indexes");
            for(const auto& sql:p.indexes) exec(p.db,sql.c_str());}
        p.finished=true;}
    catch(...) {p.failed=true;throw;}
}
void InitialEdgeInserter::append(std::span<EdgeRecord> input) {
    auto& p=*impl_;p.usable();
    try {
        if(input.empty())return;
        GeometryPhase phase("native.initial_edges.append");
        const auto first=p.reserve(input.size()),generation=writeGeneration(p.db);
        std::vector<EdgeInsertRow> rows;rows.reserve(input.size());
        {GeometryPhase preparation("initial_edges.prepare");
        for(size_t i=0;i<input.size();++i){
            auto& rec=input[i];
            const auto binaryUuid=normalizeNewEdge(rec);
            rows.push_back({&rec,first+int64_t(i),parseUuid(rec.aUuid),parseUuid(rec.bUuid),binaryUuid});
        }}
        {GeometryPhase insertion("initial_edges.edge_rows");
            insertEdgeRows(p.db,rows,generation,p.author,p.host,nowIsoUtc(),false,p.sqlBatchRows);}
    }catch(...){p.failed=true;throw;}
}
}
