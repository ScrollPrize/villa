#include "internal/EdgeSql.hpp"
#include "internal/History.hpp"
#include "geometry/GeometryToken.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/ObjectScan.hpp"
#include "geometry/GeometryDelta.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "vc_object_store/objects/StoreReader.hpp"

namespace vc::objects {
using namespace detail;
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
        adjacent(node, [&](Node next,auto accepted) {
            checkTraversalCancelled(cancel);
            if (!visited.contains(next) && accepted()) {
                visited.insert(next);queue.emplace(next, depth+1);
            }
        });
    }
    return result;
}

struct ProgressGuard {
    sqlite3* db;
    ~ProgressGuard() { sqlite3_progress_handler(db, 0, nullptr, nullptr); }
};

// Reuse endpoint-index probes throughout a walk, with the same visibility and
// applicability policy for neighbours, bounded walks and whole components.
class Adjacency {
    sqlite3* db_;
    bool includeRetired_, many_;
    const ObjectStore* store_;
    const Txn* txn_;
    const std::atomic_bool* cancel_;
    std::vector<std::pair<sqlite3*,std::unique_ptr<Statement>>> queries_;
    std::unordered_map<std::string,bool> active_;
public:
    Adjacency(sqlite3* db,const std::vector<std::string>& kinds,bool includeRetired,std::optional<bool> adjacent,
              const ObjectStore* store,const Txn* txn,const std::atomic_bool* cancel=nullptr)
        : db_(db),includeRetired_(includeRetired),store_(store),txn_(txn),cancel_(cancel)
    {
        const auto sources=querySources(db);many_=sources.size()>1;
        std::string filter=" AND state=0"+fiberAdjacencySql(adjacent);
        if(kinds.empty())filter+=" AND kind<>"+std::to_string(edgeKindCode(kAttachmentEdgeKind));
        else {
            filter+=" AND kind IN (";
            for(size_t i=0;i<kinds.size();++i)filter+=(i?",?":"?")+std::to_string(i+2);
            filter+=')';
        }
        if(!many_&&!includeRetired)filter+=" AND "+activeEdgeSql();
        for(auto* source:sources) {
            requireAdjacencyIndex(source);
            const auto table=edgeReadSource(source)+" AS edges";
            auto rows=std::make_unique<Statement>(source,
                "SELECT uuid,b_uuid FROM "+table+" WHERE a_uuid=?1"+filter+
                " UNION SELECT uuid,a_uuid FROM "+table+" WHERE b_uuid=?1"+filter+" ORDER BY 1");
            for(size_t i=0;i<kinds.size();++i)rows->bind(int(i)+2,edgeKindCode(kinds[i]));
            queries_.emplace_back(source,std::move(rows));
        }
    }
    template<class Visit> void visit(const std::string& uuid,Visit visit)
    {
        const auto accept=[&](const std::string& edge,const std::string& next) {
            visit(next,[&] {
                auto found=active_.find(next);
                if(found==active_.end()) {
                    const auto object=resolveObject(db_,next);
                    found=active_.emplace(next,object&&(includeRetired_||!object->record.retired)).first;
                }
                return found->second&&(!store_||currentRelationshipApplicable(*store_,*txn_,edge));
            });
        };
        std::vector<std::pair<std::string,std::string>> merged;
        for(auto& [source,rows]:queries_) {
            rows->resetKeepingBindings();rows->bind(1,parseUuid(uuid));
            while(rows->row()) {
                checkTraversalCancelled(cancel_);
                const auto edge=rows->uuid(0),next=rows->uuid(1);
                if(!many_)accept(edge,next);
                else if(edgeSource(db_,edge)==source)merged.emplace_back(edge,next);
            }
        }
        // UUID order is stable across physical partitions and UUID-only edge tables.
        std::sort(merged.begin(),merged.end());
        for(const auto& [edge,next]:merged)accept(edge,next);
    }
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
    Adjacency adjacent(db,edgeKindFilter,includeRetired,adjacentFilter,store,txn,cancel);
    try {
        return breadthFirst(formatUuid(startBytes), maxDepth, cancel, [&](const std::string& uuid, auto visit) {
            adjacent.visit(uuid,visit);
        });
    } catch (const std::runtime_error&) {
        checkCancelled(); // Translate SQLite interrupts into the cancellation contract.
        throw;
    }
}

}  // namespace

std::vector<std::string> ObjectStore::neighbours(const Txn& txn,const std::string& uuid,
    const std::optional<std::string>& edgeKind,const std::optional<std::string>& neighbourKind,bool includeRetired,
    std::optional<bool> adjacentFilter,bool applicableOnly) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    const auto own=resolveObject(db,uuid);
    if(!own||(!includeRetired&&own->record.retired))return {};
    if(applicableOnly)requirePublishedGraphRead(txn);
    Adjacency adjacent(db,edgeKind?std::vector{*edgeKind}:std::vector<std::string>{},includeRetired,adjacentFilter,
                       applicableOnly?this:nullptr,&txn);
    std::vector<std::string> result;std::unordered_set<std::string> seen;
    adjacent.visit(uuid,[&](const auto& next,auto accepted) {
        if(!seen.contains(next)&&accepted()&&(!neighbourKind||resolveObject(db,next)->record.kind==*neighbourKind)) {
            seen.insert(next);result.push_back(next);
        }
    });
    return result;
}

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
    return reachableWithin(txn,objectUuid,INT_MAX,edgeKindFilter,cancel,includeRetired,adjacentFilter,applicableOnly);
}

std::vector<std::string> ObjectStore::reachableWithin(const Txn& txn, const std::string& objectUuid, int maxDepth,
                                                  const std::vector<std::string>& edgeKindFilter,
                                                  const std::atomic_bool* cancel, bool includeRetired,
                                              std::optional<bool> adjacentFilter, bool applicableOnly) const
{
    if (maxDepth < 0) throw std::runtime_error("object store: maxDepth must be >= 0");
    auto* db = static_cast<sqlite3*>(connection(txn));
    if (applicableOnly) requirePublishedGraphRead(txn);
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

GraphReadView ObjectStore::geometryReadView(const Txn& txn,const ObjectSelection& selection,
                                         const std::optional<StoreStatus>& expectedPosition) const
{
    return geometryReadViewImpl(txn,selection,expectedPosition,false);
}

GraphReadView ObjectStore::geometryReadViewImpl(const Txn& txn,const ObjectSelection& selection,
    const std::optional<StoreStatus>& expectedPosition,bool trustedCurrent) const
{
    requirePublishedGraphRead(txn);
    auto* db=static_cast<sqlite3*>(connection(txn));
    if(selection.minLength||selection.bbox||selection.connectedTo||!selection.edgeKinds.empty()||
       selection.orderBy!="uuid"||selection.afterLength||selection.includeUnbound)
        throw std::invalid_argument("geometry read supports UUID-ordered metadata selection, not measured/connected filters");
    if(selection.offset<0||(selection.limit&&*selection.limit<0)||(selection.afterUuid&&selection.offset))
        throw std::invalid_argument("geometry read has invalid pagination");
    if(selection.volume&&selection.coordinates)
        throw std::invalid_argument("geometry read volume and coordinates filters are exclusive");
    if(!std::isfinite(selection.targetScale)||selection.targetScale<=0)
        throw std::invalid_argument("geometry read target scale must be finite and positive");
    if(expectedPosition){
        const auto current=status(txn);
        if(expectedPosition->commitSequence<0||current.storeUuid!=util::canonicalUuid(expectedPosition->storeUuid)||
           current.generation!=util::canonicalUuid(expectedPosition->generation)||current.commitSequence!=expectedPosition->commitSequence)
            throw StoreBusy("geometry read: source position changed");
    }
    auto physical=selection;physical.volume.reset();physical.limit.reset();physical.offset=0;
    std::optional<VolumeTransforms> transforms=selection.transforms?std::optional(*selection.transforms):std::nullopt;
    GraphReadView result;result.selectedObjects.emplace();result.selectedRelationships.emplace();
    int64_t skipped=0;
    if(!selection.limit||*selection.limit>0)
        scanVisibleObjectRows(db,physical,trustedCurrent?ObjectFields(std::set<std::string>{"meta","storage_kind","storage_ref","data_checksum"}):std::nullopt,
            [&](ObjectRecord&& row,std::optional<SelectedGeometryHead> head){
            if(selection.volume){
                const auto metadata=nlohmann::json::parse(row.meta);
                if(!geometryScaleToVolume(metadata,*selection.volume)){
                    if(!transforms)transforms=volumeTransforms(txn);
                    if(!transforms->toVolume(metadata,*selection.volume,selection.targetScale))return true;
                }
            }
            if(skipped<selection.offset){++skipped;return true;}
            if(head&&!trustedCurrent)(void)loadGeometryManifest(head->source,head->reference,head->manifest,row.dataChecksum.value_or(""));
            ObjectRevision selected;
            selected.metadataComplete=!trustedCurrent; // trusted batch metadata is a projection, not a complete edit reference
            selected.storeUuid=row.versionStoreUuid;selected.generation=row.versionGeneration;
            selected.objectUuid=row.uuid;selected.version=row.version.value_or(0);
            selected.commitSequence=row.commitSequence;
            if(head)selected.geometryRevision=head->reference;
            selected.record=std::move(row);
            result.objectUuids.push_back(selected.objectUuid);result.versions.push_back(selected.version);
            result.geometryRevisions.push_back(selected.geometryRevision);result.selectedObjects->push_back(std::move(selected));
            return !selection.limit||result.objectUuids.size()<size_t(*selection.limit);
        });
    // The empty graph's indexing is useful to existing bounded geometry views;
    // no relationship tables or numbered history are consulted.
    result.offsets.resize(result.objectUuids.size()+1,0);
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
    // A zero relationship limit is an explicit geometry-only selection. Do
    // not enumerate incident evidence merely to discard every row afterward.
    if(!edges.limit||*edges.limit!=0)
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
