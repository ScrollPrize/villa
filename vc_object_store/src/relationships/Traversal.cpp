#include "internal/EdgeSql.hpp"
#include "internal/History.hpp"
#include "geometry/GeometryToken.hpp"
#include "internal/RecordResolver.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

namespace vc::objects {
using namespace detail;
std::vector<std::string> ObjectStore::neighbours(const Txn& txn, const std::string& objectUuid,
                                             const std::optional<std::string>& edgeKind,
                                             const std::optional<std::string>& neighbourKind, bool includeRetired,
                                             std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto key=parseUuid(objectUuid);
    const auto own=resolveObject(db,objectUuid);
    if(!own||(!includeRetired&&own->record.retired))return {};
    if (applicableOnly) requirePublishedGraphRead(txn);
    if(querySources(db).size()>1){
        EdgeSelection selection;selection.kind=edgeKind;selection.endpointUuids=std::vector{objectUuid};
        selection.adjacent=adjacentFilter;selection.includeRetired=includeRetired;
        selection.applicableOnly=applicableOnly;selection.attachments=bool(edgeKind);
        std::vector<std::string> result;std::set<std::string> seen;
        for(const auto& edge:selectEdges(txn,selection)){
            const auto next=edge.otherEndpoint(objectUuid);
            const auto endpoint=resolveObject(db,next);
            if(endpoint&&(!neighbourKind||endpoint->record.kind==*neighbourKind)&&seen.insert(next).second)result.push_back(next);
        }
        return result;
    }
    const std::string filter = " AND state=0" + fiberAdjacencySql(adjacentFilter) +
        (edgeKind ? std::string(" AND kind=?2") : " AND kind<>'" + std::string(kAttachmentEdgeKind) + "'") +
        (includeRetired ? "" : " AND " + activeEdgeSql());
    // Filter before deduplication: a rejected parallel edge must not hide a
    // later applicable edge to the same neighbour. Preserve first-edge order.
    std::string sql = "SELECT e.neighbour,e.uuid,owned.id,owned.kind FROM (SELECT id,uuid,b_uuid AS neighbour FROM edges INDEXED BY edges_a WHERE a_uuid=?1" + filter +
        " UNION SELECT id,uuid,a_uuid AS neighbour FROM edges INDEXED BY edges_b WHERE b_uuid=?1" + filter +
        ") e LEFT JOIN objects owned ON owned.uuid=e.neighbour WHERE coalesce(owned.state,0)<>2";
    sql += " ORDER BY e.id";
    Statement rows(db, sql, key);
    if (edgeKind) rows.bind(2, *edgeKind);
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    while (rows.row()) {
        const auto next = formatUuid(rows.blob16(0));
        if (seen.contains(next)) continue;
        const auto endpoint=resolveObject(db,next);
        if(!endpoint||(!includeRetired&&endpoint->record.retired)||
           (neighbourKind&&endpoint->record.kind!=*neighbourKind))continue;
        if (applicableOnly && !currentRelationshipApplicable(*this, txn, formatUuid(rows.blob16(1)))) continue;
        seen.insert(next); out.push_back(next);
    }
    return out;
}

namespace {

void checkTraversalCancelled(const std::atomic_bool* cancel)
{
    if (cancel && cancel->load()) throw std::runtime_error("object store: traversal cancelled");
}

template<class Node, class Adjacent>
std::vector<Node> breadthFirst(Node start, int maxDepth, const std::atomic_bool* cancel, Adjacent adjacent)
{
    std::vector<Node> result;
    std::unordered_set<Node> visited{start};
    std::queue<std::pair<Node, int>> queue; queue.emplace(start, 0);
    while (!queue.empty()) {
        checkTraversalCancelled(cancel);
        const auto [node, depth] = queue.front(); queue.pop();
        result.push_back(node);
        if (depth >= maxDepth) continue;
        adjacent(node, [&](Node next, auto accepted) {
            checkTraversalCancelled(cancel);
            if (!visited.contains(next) && accepted()) {
                visited.insert(next); queue.emplace(next, depth+1);
            }
        });
    }
    return result;
}

struct ProgressGuard {
    sqlite3* db;
    ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
};

std::vector<std::string> traverse(sqlite3* db, const std::string& startUuid, int maxDepth,
                                  const std::vector<std::string>& edgeKindFilter, const std::atomic_bool* cancel, bool includeRetired,
                                  std::optional<bool> adjacentFilter = std::nullopt,
                                  const ObjectStore* store = nullptr, const Txn* txn = nullptr)
{
    auto checkCancelled = [&] { checkTraversalCancelled(cancel); };
    checkCancelled();
    ProgressGuard guard{db};
    if (cancel)
        sqlite3_progress_handler(
            db, 1000, [](void* p) { return static_cast<const std::atomic_bool*>(p)->load() ? 1 : 0; },
            const_cast<std::atomic_bool*>(cancel));
    const auto startBytes = parseUuid(startUuid);
    auto active=[&](const std::string& uuid) {
        const auto object=resolveObject(db,uuid);
        return object&&(includeRetired||!object->record.retired);
    };
    if(!resolveObject(db,startUuid))throw std::runtime_error("object store: object not found: "+startUuid);
    if(!active(startUuid))return {};
    // Probe the endpoint indexes only for nodes reached by this traversal.
    // Visit neighbours in stable internal-ID order.
    std::string filter=" AND state=0";
    if (!edgeKindFilter.empty()) {
        filter += " AND kind IN (";
        for (size_t i = 0; i < edgeKindFilter.size(); ++i)
            filter += (i ? ",?" : "?") + std::to_string(i + 2);
        filter += ")";
    } else {
        filter += " AND kind<>'" + std::string(kAttachmentEdgeKind) + "'";
    }
    filter += fiberAdjacencySql(adjacentFilter);
    if (!includeRetired) filter += " AND " + activeEdgeSql();
    // Each UNION arm uses an endpoint-leading index even without ANALYZE.
    std::string sql = "SELECT id,b_uuid,uuid FROM edges INDEXED BY edges_a WHERE a_uuid=?1" + filter +
        " UNION SELECT id,a_uuid,uuid FROM edges INDEXED BY edges_b WHERE b_uuid=?1" + filter + " ORDER BY id";
    Statement adjacent(db, sql);
    std::unordered_map<std::string,bool> activeNodes;
    activeNodes.emplace(startUuid,true);
    try {
        return breadthFirst(formatUuid(startBytes), maxDepth, cancel, [&](const std::string& uuid, auto visit) {
            adjacent.reset(); adjacent.bind(1, parseUuid(uuid));
            for(size_t i=0;i<edgeKindFilter.size();++i)adjacent.bind(int(i)+2,edgeKindFilter[i]);
            while(adjacent.row()) {
                const auto endpoint=formatUuid(adjacent.blob16(1));
                visit(endpoint,[&] {
                    auto found=activeNodes.find(endpoint);
                    if(found==activeNodes.end())found=activeNodes.emplace(endpoint,active(endpoint)).first;
                    return found->second&&(!store||currentRelationshipApplicable(*store,*txn,formatUuid(adjacent.blob16(2))));
                });
            }
        });
    } catch (const std::runtime_error&) {
        checkCancelled(); // Translate SQLite interrupts into the cancellation contract.
        throw;
    }
}

}  // namespace

namespace {
// The component of `start`, cached on read connections (nullptr elsewhere,
// with the walk in `walked`).
std::shared_ptr<StatementPool::Component> cachedComponent(
    sqlite3* db, const std::string& start, const std::vector<std::string>& kinds, bool includeRetired,
    std::optional<bool> adjacentFilter, const std::atomic_bool* cancel, std::vector<int64_t>& walked)
{
    StatementPool* pool = nullptr;
    if (auto it = activeStatementPools.find(db); it != activeStatementPools.end() && it->second->readSnapshot)
        pool = it->second;
    const auto prefix = nlohmann::json({kinds, includeRetired, adjacentFilter ? nlohmann::json(*adjacentFilter) : nlohmann::json(nullptr)}).dump() + "/";
    const auto key = prefix + formatUuid(parseUuid(start));
    if (pool) if (auto it = pool->components.find(key); it != pool->components.end()) return it->second;
    const auto keys=traverse(db,start,INT_MAX,kinds,cancel,includeRetired,adjacentFilter);
    std::vector<int64_t> ids;ids.reserve(keys.size());
    for(const auto& uuid:keys)if(const auto id=objectIdByUuid(db,parseUuid(uuid)))ids.push_back(*id);
    // Bound the transaction-local membership cache; never reuse across snapshots.
    if (pool && ids.size() <= 262144) {
        if (pool->componentNodes + ids.size() > 262144) { pool->components.clear(); pool->componentNodes = 0; }
        pool->componentNodes += ids.size();
        const auto shared = std::make_shared<StatementPool::Component>(StatementPool::Component{std::move(ids), std::nullopt});
        for (const auto& uuid : keys) pool->components.emplace(prefix + uuid, shared);
        return shared;
    }
    walked = std::move(ids);
    return nullptr;
}
}  // namespace

std::vector<int64_t> detail::componentIds(sqlite3* db, const std::string& start,
                                        const std::vector<std::string>& kinds, bool includeRetired, std::optional<bool> adjacentFilter,
                                        const std::atomic_bool* cancel)
{
    std::vector<int64_t> walked;
    const auto component = cachedComponent(db, start, kinds, includeRetired, adjacentFilter, cancel, walked);
    return component ? component->ids : walked;
}

std::vector<std::string> ObjectStore::componentOf(const Txn& txn, const std::string& objectUuid,
                                              const std::vector<std::string>& edgeKindFilter,
                                              const std::atomic_bool* cancel, bool includeRetired,
                                              std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (applicableOnly) requirePublishedGraphRead(txn);
    if(querySources(db).size()>1)return reachableWithin(txn,objectUuid,INT_MAX,edgeKindFilter,cancel,includeRetired,adjacentFilter,applicableOnly);
    return traverse(db,objectUuid,INT_MAX,edgeKindFilter,cancel,includeRetired,adjacentFilter,applicableOnly?this:nullptr,&txn);
}

std::vector<std::string> ObjectStore::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel, bool includeRetired,
                                              std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    if (maxDepth < 0) throw std::runtime_error("object store: maxDepth must be >= 0");
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (applicableOnly) requirePublishedGraphRead(txn);
    if(querySources(db).size()>1){
        const auto root=resolveObject(db,objectUuid);
        if(!root)throw std::runtime_error("object store: object not found: "+objectUuid);
        if(root->record.retired&&!includeRetired)return {};
        return breadthFirst(formatUuid(parseUuid(objectUuid)),maxDepth,cancel,[&](const auto& uuid,auto visit){
            EdgeSelection selection;selection.endpointUuids=std::vector{uuid};selection.includeRetired=includeRetired;
            selection.adjacent=adjacentFilter;selection.applicableOnly=applicableOnly;selection.attachments=!edgeKindFilter.empty();
            for(const auto& edge:selectEdges(txn,selection)){
                if(!edgeKindFilter.empty()&&std::find(edgeKindFilter.begin(),edgeKindFilter.end(),edge.kind)==edgeKindFilter.end())continue;
                visit(edge.otherEndpoint(uuid),[]{return true;});
            }
        });
    }
    return traverse(db, objectUuid, maxDepth, edgeKindFilter, cancel, includeRetired,
                                adjacentFilter, applicableOnly ? this : nullptr, &txn);
}

namespace {
void finishGraph(GraphReadView& result, sqlite3* db)
{
    result.offsets.resize(result.objectUuids.size() + 1);
    for (const auto& [a, b] : result.endpoints) { ++result.offsets[a+1]; ++result.offsets[b+1]; }
    for (size_t i = 1; i < result.offsets.size(); ++i) result.offsets[i] += result.offsets[i-1];
    result.neighbours.resize(result.offsets.back()); result.edgeIndices.resize(result.offsets.back());
    auto next = result.offsets;
    for (size_t edge = 0; edge < result.endpoints.size(); ++edge) {
        const auto [a, b] = result.endpoints[edge];
        for (const auto& [from, to] : {std::pair{a, b}, std::pair{b, a}}) {
            const auto slot = next[from]++;
            result.neighbours[slot] = to; result.edgeIndices[slot] = edge;
        }
    }
    if(db && historyEnabled(db)) {
        Statement journal(db, "SELECT coalesce(max(seq),0) FROM journal");
        (void)journal.row(); result.journalSequence = journal.i64(0);
    }
}
}

GraphReadView ObjectStore::graphReadView(const Txn& txn, const ObjectSelection& objects,
                                       const EdgeSelection& edges) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    GraphReadView result;
    if (edges.endpointUuids)
        throw std::invalid_argument("graph endpoint scope is specified by its object selection");
    std::unordered_map<std::string, uint64_t> nodes;
    for(const auto& object:selectObjects(txn,objects,std::vector<std::string>{})) {
        const auto node = result.objectUuids.size();
        nodes.emplace(object.uuid,node);
        result.objectUuids.push_back(object.uuid);
        result.versions.push_back(object.version.value_or(0));
        result.geometryRevisions.push_back(geometryHead(db,object.uuid));
    }
    auto selection = edges; selection.bothEndpoints = true;
    selection.includeRetired = objects.includeRetired;
    // Attachments are not graph links unless asked for by kind.
    if (!selection.kind) selection.attachments = false;
    selection.endpointUuids = result.objectUuids;
    for (const auto& edge : selectEdges(txn, selection, std::vector<std::string>{})) {
        result.edgeUuids.push_back(edge.uuid);
        result.endpoints.push_back({nodes.at(edge.aUuid),nodes.at(edge.bUuid)});
    }
    finishGraph(result, db);
    return result;
}

GraphReadView ObjectStore::graphReadViewAt(const Txn& txn, const std::vector<ObjectVersion>& versions,
                                         const EdgeSelection& edges) const
{
    requirePublishedGraphRead(txn);
    auto result = LocalStoreReader(*this, txn).graphReadViewAt(versions, edges);
    if (historyEnabled(ObjectStoreImplAccess::conn(txn))) result.journalSequence = journalPosition(txn);
    return result;
}

GraphReadView StoreReader::graphReadViewAt(const std::vector<ObjectVersion>& versions, const EdgeSelection& edges) const
{
    validateEdgeSelection(edges);
    if (edges.endpointUuids)
        throw std::invalid_argument("graph endpoint scope is specified by its version selection");
    GraphReadView result;
    result.selectedObjects.emplace(); result.selectedRelationships.emplace();
    std::map<std::string,ObjectRevision> resolved;
    for(auto& value:objectsAtVersions(versions)) {
        if(!value||value->deleted)throw std::invalid_argument("object history: selected version is unavailable or deleted");
        if (!resolved.emplace(value->objectUuid,*value).second)
            throw std::invalid_argument("duplicate object in revision selection");
    }
    std::map<std::string, uint64_t> nodes;
    for (auto& [object, value] : resolved) {
        nodes.emplace(object, result.objectUuids.size());
        result.objectUuids.push_back(object); result.versions.push_back(value.record.version.value_or(0));
        result.geometryRevisions.push_back(value.geometryRevision);
        result.selectedObjects->push_back(std::move(value));
    }
    auto graphEdges = edges;
    if (!graphEdges.kind) graphEdges.attachments = false;
    for (auto& selected : relationshipsAt(versions))
        if (selected.applicable && matchesEdgeSelection(selected.relationship.record, graphEdges))
            result.selectedRelationships->push_back(std::move(selected.relationship));
    auto& relationships = *result.selectedRelationships;
    std::sort(relationships.begin(), relationships.end(), [](const auto& a, const auto& b) { return a.record.uuid < b.record.uuid; });
    if (edges.limit && relationships.size() > static_cast<size_t>(*edges.limit)) relationships.resize(*edges.limit);
    for (const auto& relationship : relationships) {
        const auto& edge = relationship.record;
        result.edgeUuids.push_back(edge.uuid);
        result.endpoints.push_back({nodes.at(edge.aUuid), nodes.at(edge.bUuid)});
    }
    finishGraph(result, nullptr);
    return result;
}

GraphReadView StoreReader::graphReadViewAround(const std::vector<ObjectVersion>& versions,
                                              const EdgeSelection& edges, const std::atomic_bool* cancel) const
{
    validateEdgeSelection(edges);
    if (edges.endpointUuids)
        throw std::invalid_argument("graph endpoint scope is specified by its version selection");
    std::map<std::string,ObjectVersion> selected, pending;
    std::set<std::string> pinned;
    for (const auto& object : objectsAtVersions(versions)) {
        if (!object || object->deleted) throw std::invalid_argument("graph seed version is unavailable or deleted");
        const auto reference = object->reference();
        if (!selected.emplace(reference.objectUuid, reference).second)
            throw std::invalid_argument("duplicate object in graph seed selection");
        pinned.insert(reference.objectUuid);
        pending.emplace(reference.objectUuid, reference);
    }
    auto filter = edges;
    if (!filter.kind) filter.attachments = false;
    while (!pending.empty()) {
        if (cancel && cancel->load()) throw std::runtime_error("graph selection cancelled");
        std::vector<ObjectVersion> frontier;
        for (const auto& [uuid, reference] : pending) frontier.push_back(reference);
        pending.clear();
        for (const auto& relationship : relationshipsAt(frontier)) {
            const auto& edge = relationship.relationship;
            if (!matchesEdgeSelection(edge.record, filter)) continue;
            for (const auto& reference : {edge.aReference, edge.bReference}) {
                if (reference.objectUuid.empty() || reference.storeUuid.empty() || reference.generation.empty() || reference.version <= 0)
                    throw std::invalid_argument("recorded graph endpoint reference is unavailable");
                if (pinned.contains(reference.objectUuid)) continue;
                const auto found = selected.find(reference.objectUuid);
                if (found != selected.end()) {
                    if (found->second.storeUuid != reference.storeUuid || found->second.generation != reference.generation)
                        throw std::invalid_argument("ambiguous recorded owners/lineages for graph endpoint " + reference.objectUuid);
                    if (found->second.version >= reference.version) continue;
                }
                selected.insert_or_assign(reference.objectUuid, reference);
                pending.insert_or_assign(reference.objectUuid, reference);
            }
        }
    }
    std::vector<ObjectVersion> closure;
    for (const auto& [uuid, reference] : selected) closure.push_back(reference);
    return graphReadViewAt(closure, edges);
}

std::vector<std::string> GraphReadView::reachableWithin(const std::string& objectUuid, int maxDepth,
                                                       const std::atomic_bool* cancel) const
{
    if (maxDepth < 0) throw std::invalid_argument("object store: maxDepth must be >= 0");
    checkTraversalCancelled(cancel);
    const auto found = std::find(objectUuids.begin(), objectUuids.end(), formatUuid(parseUuid(objectUuid)));
    if (found == objectUuids.end()) throw std::invalid_argument("object not in graph: " + objectUuid);
    const auto start = static_cast<size_t>(found - objectUuids.begin());
    const auto nodes = breadthFirst(start, maxDepth, cancel, [&](size_t node, auto visit) {
        for (auto i = offsets[node]; i < offsets[node+1]; ++i) visit(neighbours[i], [] { return true; });
    });
    std::vector<std::string> result; result.reserve(nodes.size());
    for (auto node : nodes) result.push_back(objectUuids[node]);
    return result;
}

std::vector<std::string> GraphReadView::componentOf(const std::string& objectUuid, const std::atomic_bool* cancel) const
{
    return reachableWithin(objectUuid, INT_MAX, cancel);
}

} // namespace vc::objects
