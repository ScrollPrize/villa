#include "internal/Versions.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/JsonFields.hpp"
#include "internal/SpatialCache.hpp"
#include "internal/CoordinateFilter.hpp"
#include "internal/VolumeFrames.hpp"
#include "internal/TransformedLength.hpp"
#include "geometry/GeometryToken.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/VisibleRows.hpp"
#include "internal/ObjectScan.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include <charconv>

namespace vc::objects {
using namespace detail;
namespace {
void appendReferenceDigest(PayloadHasher& hash,const std::string& store,const std::string& generation,
                           const std::string& object,int64_t version,std::string& line)
{
    if(store.empty()||generation.empty()||object.empty()||version<=0)
        throw std::invalid_argument("reference digest requires a complete published identity");
    std::array<char,32> number;
    const auto end=std::to_chars(number.data(),number.data()+number.size(),version);
    if(end.ec!=std::errc{})throw std::runtime_error("reference digest version encoding failed");
    line.clear();line.reserve(store.size()+generation.size()+object.size()+36);
    line.append(store);line+=' ';line.append(generation);line+=' ';
    line.append(object);line+=' ';line.append(number.data(),end.ptr);line+='\n';
    hash.update(line.data(),line.size());
}
void beginReferenceDigest(PayloadHasher& hash,const std::string& domain)
{
    hash.update(domain.data(),domain.size());hash.update("\n",1);
}
}

std::string objectReferencesDigest(std::span<const ObjectVersion> references,const std::string& domain)
{
    std::vector<const ObjectVersion*> ordered;ordered.reserve(references.size());
    for(const auto& reference:references)ordered.push_back(&reference);
    // Stable sorting matches the explicit-reference codec when object keys are
    // repeated by a caller; current selected rows are already unique/ordered.
    std::stable_sort(ordered.begin(),ordered.end(),[](const auto* a,const auto* b){return a->objectUuid<b->objectUuid;});
    PayloadHasher hash;beginReferenceDigest(hash,domain);std::string line;
    for(const auto* reference:ordered)appendReferenceDigest(hash,reference->storeUuid,reference->generation,
        reference->objectUuid,reference->version,line);
    return hash.finish();
}

std::string ObjectStore::selectedReferenceDigest(const Txn& txn,const ObjectSelection& selection,const std::string& domain)const
{
    if(selection.orderBy!="uuid"||selection.connectedTo||selection.volume||selection.includeUnbound||selection.minLength)
        throw std::invalid_argument("reference digest requires an ordinary UUID-ordered object selection");
    PayloadHasher hash;beginReferenceDigest(hash,domain);size_t count=0;std::string line;
    scanVisibleObjectReferences(static_cast<sqlite3*>(connection(txn)),selection,
        [&](ObjectVersion&& reference){
            appendReferenceDigest(hash,reference.storeUuid,reference.generation,reference.objectUuid,
                                  reference.version,line);++count;return true;
        });
    if(!selection.uuids.empty()){
        std::set<std::string> requested;
        for(const auto& id:selection.uuids)requested.insert(formatUuid(parseUuid(id)));
        if(requested.size()!=count)throw std::invalid_argument("Unknown requested object in reference digest");
    }
    return hash.finish();
}

void validateObjectFields(const std::optional<std::vector<std::string>>& fields)
{
    (void)projectedObjectColumns(fields ? ObjectFields(std::set<std::string>(fields->begin(), fields->end())) : std::nullopt);
}
void validateEdgeFields(const std::optional<std::vector<std::string>>& fields)
{
    validateEdgeProjection(fields ? ObjectFields(std::set<std::string>(fields->begin(), fields->end())) : std::nullopt);
}

void detail::beginSpatialRead(StatementPool& pool, size_t budget, int version)
{
    if (!budget) { pool.spatialCache.reset(); return; }
    if (!pool.spatialCache) pool.spatialCache = std::make_shared<SpatialReadCache>(budget);
    pool.spatialCache->begin(version);
}

namespace {
// Objects of a kind not bound to a volume, NOT (volume IS v AND
// COALESCE(scale>0,0)), as ranges of objects_coordinates.
std::string unboundObjectIdsSql(const std::string& kind, const std::string& volume)
{
    const auto arm = " UNION ALL SELECT id FROM objects INDEXED BY objects_coordinates WHERE kind=" + kind + " AND ";
    return "SELECT id FROM objects INDEXED BY objects_coordinates WHERE kind=" + kind + " AND coordinate_volume IS NULL" +
        arm + "coordinate_volume<" + volume + arm + "coordinate_volume>" + volume +
        arm + "coordinate_volume=" + volume + " AND coordinate_scale IS NULL" +
        arm + "coordinate_volume=" + volume + " AND coordinate_scale<=0";
}
struct SelectionFilter {
    std::string sql;
    std::vector<std::string> parameters;
};
SelectionFilter objectFilter(sqlite3* db, const ObjectSelection& selection)
{
    if (selection.volume || selection.includeUnbound)
        throw std::invalid_argument("volume targets apply to object selections, spatial regions and geometry regions");
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (selection.orderBy != "uuid" && selection.orderBy != "length_desc")
        throw std::invalid_argument("order_by must be uuid or length_desc");
    if (selection.family || selection.minLength || selection.orderBy == "length_desc")
        if (!selection.kind || (*selection.kind != "fiber" && *selection.kind != "track"))
            throw std::invalid_argument("typed catalog filters require fiber or track kind");
    if (selection.minLength && (!std::isfinite(*selection.minLength) || *selection.minLength < 0))
        throw std::invalid_argument("min_length must be finite and nonnegative");
    if (selection.afterLength && !std::isfinite(*selection.afterLength))
        throw std::invalid_argument("after_length must be finite");
    if (selection.orderBy == "length_desc" ? bool(selection.afterUuid) != bool(selection.afterLength) : bool(selection.afterLength))
        throw std::invalid_argument("length cursor requires after_uuid and after_length with length_desc");
    if (selection.afterUuid && selection.offset)
        throw std::invalid_argument("cursor and offset cannot be combined");
    std::set<int64_t> selected, connected;
    for (const auto& uuid : selection.uuids)
        if (const auto id = objectIdByUuid(db, parseUuid(uuid))) selected.insert(*id);
    if (selection.connectedTo) {
        const auto ids = componentIds(db, *selection.connectedTo, selection.edgeKinds, selection.includeRetired);
        connected.insert(ids.begin(), ids.end());
    }
    SelectionFilter filter;
    // The selected local row already supplies lifecycle state; a correlated
    // self-lookup by its own ID would repeat the same test for every object.
    filter.sql = selection.includeRetired?" AND o.state<>2":" AND o.state=0";
    auto bind = [&](const std::string& clause, const std::string& value) {
        filter.sql += clause; filter.parameters.push_back(value);
    };
    if (!selection.uuids.empty() || selection.connectedTo) {
        nlohmann::json ids = nlohmann::json::array();
        const auto& candidates = selection.uuids.empty() ? connected : selected;
        for (const auto id : candidates) {
            if (selection.connectedTo && !connected.contains(id)) continue;
            ids.push_back(id);
        }
        bind(" AND o.id IN (SELECT value FROM json_each(?))", ids.dump());
    }
    if (selection.kind) bind(" AND o.kind=?", *selection.kind);
    if (selection.coordinates) {
        const auto parameter = "?" + std::to_string(filter.parameters.size() + 1);
        bind(coordinateFilterSql("o", parameter), coordinateFilterValue(*selection.coordinates));
    }
    auto groups = selection.tagGroups;
    if (selection.tag) groups.push_back({*selection.tag});
    for (const auto& group : groups)
        bind(" AND o.id IN (SELECT object_id FROM object_tags INDEXED BY object_tags_tag WHERE tag IN "
             "(SELECT value FROM json_each(?)))", nlohmann::json(group).dump());
    if (!selection.withoutTags.empty())
        bind(" AND o.id NOT IN (SELECT object_id FROM object_tags INDEXED BY object_tags_tag WHERE tag IN "
             "(SELECT value FROM json_each(?)))", nlohmann::json(selection.withoutTags).dump());
    if (selection.family) bind(" AND d.family=?", *selection.family);
    if (selection.minLength) bind(" AND d.length>=?", nlohmann::json(*selection.minLength).dump());
    if (selection.afterUuid) {
        auto hex = formatUuid(parseUuid(*selection.afterUuid));
        hex.erase(std::remove(hex.begin(), hex.end(), '-'), hex.end());
        const auto uuid = "X'" + hex + "'";
        if (selection.orderBy == "length_desc") {
            bind(" AND d.length<=?", nlohmann::json(*selection.afterLength).dump());
            bind(" AND (d.length<?", nlohmann::json(*selection.afterLength).dump());
            filter.sql += " OR o.uuid>" + uuid + ")";
        } else filter.sql += " AND o.uuid>" + uuid;
    }
    return filter;
}
void bindSelection(Statement& rows, const SelectionFilter& filter, const ObjectSelection& selection)
{
    int next = 1;
    for (const auto& value : filter.parameters) rows.bind(next++, value);
    if (selection.bbox) for (double coordinate : *selection.bbox) rows.bind(next++, coordinate);
    rows.bind(next++, selection.limit.value_or(-1));
    rows.bind(next, selection.offset);
}
}

namespace {
void requireVolumeSelection(const ObjectSelection& selection)
{
    if (selection.includeUnbound && !selection.volume)
        throw std::invalid_argument("include_unbound requires a volume");
    if (selection.volume && selection.coordinates)
        throw std::invalid_argument("volume and coordinates filters are exclusive");
    if (!std::isfinite(selection.targetScale) || selection.targetScale <= 0)
        throw std::invalid_argument("selection target scale must be finite and positive");
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (selection.afterUuid && selection.offset)
        throw std::invalid_argument("cursor and offset cannot be combined");
    if (selection.orderBy != "uuid" && selection.orderBy != "length_desc")
        throw std::invalid_argument("order_by must be uuid or length_desc");
    if ((selection.minLength || selection.family || selection.orderBy=="length_desc") &&
        (!selection.kind || (*selection.kind!="fiber" && *selection.kind!="track")))
        throw std::invalid_argument("typed catalog filters require fiber or track kind");
    if (selection.minLength && (!std::isfinite(*selection.minLength) || *selection.minLength<0))
        throw std::invalid_argument("min_length must be finite and nonnegative");
    if (selection.afterLength && (!std::isfinite(*selection.afterLength) || selection.orderBy!="length_desc"))
        throw std::invalid_argument("after_length requires finite length_desc cursor");
    const bool tail = selection.includeUnbound && selection.orderBy == "length_desc" && selection.afterUuid;
    if (selection.orderBy == "length_desc" && bool(selection.afterUuid) != bool(selection.afterLength) && !tail)
        throw std::invalid_argument("length cursor requires after_uuid and after_length with length_desc");
}
// The frames of a volume selection, mapping to target voxels at its scale.
std::vector<VolumeFrame> selectionFrames(sqlite3* db, const ObjectSelection& selection)
{
    auto frames = volumeFrames(db, *selection.volume, selection.kind, selection.includeRetired, selection.transforms.get());
    for (auto& frame : frames) {
        frame.toVolume *= selection.targetScale;
        frame.mapping = VolumeAffine::scale(selection.targetScale) * frame.mapping;
    }
    return frames;
}
// One frame of a volume selection: the caller's filters in that frame's stored units.
ObjectSelection frameSelection(const ObjectSelection& selection, const VolumeFrame& frame)
{
    auto sub = selection;
    sub.volume.reset(); sub.includeUnbound = false;
    sub.coordinates = frame.coordinates;
    sub.afterUuid.reset(); sub.afterLength.reset();
    sub.offset = 0;
    if (selection.limit) sub.limit = *selection.limit + selection.offset;
    if (selection.bbox) {
        validateBounds(*selection.bbox);
        sub.bbox = frame.registered ? transformBounds(*selection.bbox,frame.mapping.inverse()) : storedBox(*selection.bbox, frame.toVolume);
    }
    sub.minLength.reset(); // Length is measured after mapping selected geometry.
    return sub;
}
struct VolumeRow {
    std::string uuid;
    double volume = 0;
    bool bound = true;
};
std::vector<VolumeRow> measuredSelection(sqlite3* db,const ObjectSelection& selection) {
    requireVolumeSelection(selection);
    auto afterUuid = selection.afterUuid;
    if (afterUuid) *afterUuid = formatUuid(parseUuid(*afterUuid));
    const bool byLength=selection.orderBy=="length_desc";
    std::vector<VolumeRow> rows;
    const bool inTail=byLength&&selection.afterUuid&&!selection.afterLength;
    const auto frames=selection.volume?selectionFrames(db,selection):std::vector<VolumeFrame>{{}};
    if(!inTail) for(const auto& frame:frames) {
        auto sub=selection.volume?frameSelection(selection,frame):selection;
        sub.afterUuid.reset();sub.afterLength.reset();sub.offset=0;
        sub.limit.reset();sub.minLength.reset();sub.orderBy="uuid";
        auto query=objectSelectionQuery(db,sub,"o.uuid");
        while(query->row()) {
            VolumeRow row{formatUuid(query->blob16(0))};
            if(byLength||selection.minLength) {
                Statement head(db,"SELECT storage_ref FROM objects WHERE uuid=? AND geometry_manifest IS NOT NULL",parseUuid(row.uuid));
                if(!head.row())continue;
                row.volume=transformedLength(db,readGeometryTokenColumn(head,0),frame.mapping,*selection.kind);
                if(selection.minLength&&row.volume<*selection.minLength)continue;
            }
            if(afterUuid) {
                if(byLength&&selection.afterLength) {
                    if(row.volume>*selection.afterLength||(row.volume==*selection.afterLength&&row.uuid<=*afterUuid))continue;
                } else if(byLength||row.uuid<=*afterUuid)continue;
            }
            rows.push_back(std::move(row));
        }
    }
    if(selection.includeUnbound&&!selection.bbox&&!(selection.minLength&&*selection.minLength>0)) {
        auto sub=selection;sub.volume.reset();sub.includeUnbound=false;sub.limit.reset();sub.offset=0;
        sub.minLength.reset();sub.orderBy="uuid";sub.afterLength.reset();
        if(byLength&&selection.afterLength)sub.afterUuid.reset();
        auto query=objectSelectionQuery(db,sub,"o.uuid,o.meta");
        while(query->row()) {
            const auto id=formatUuid(query->blob16(0));
            if(!selection.transforms->toVolume(parseObjectJson(query->text(1)),*selection.volume))rows.push_back({id,0,false});
        }
    }
    std::sort(rows.begin(),rows.end(),[&](const auto& a,const auto& b){
        if(a.bound!=b.bound)return a.bound;
        if(byLength&&a.volume!=b.volume)return a.volume>b.volume;
        return a.uuid<b.uuid;
    });
    const size_t first=std::min(rows.size(),size_t(selection.offset));
    const size_t last=selection.limit?std::min(rows.size(),first+size_t(*selection.limit)):rows.size();
    return {std::make_move_iterator(rows.begin()+first),std::make_move_iterator(rows.begin()+last)};
}
}

std::unique_ptr<Statement> detail::objectSelectionQuery(sqlite3* db, const ObjectSelection& selection,
                                                       const std::string& columns)
{
    if (selection.bbox) requireSpatial(db);
    auto filter = objectFilter(db, selection);
    const bool candidates = selection.bbox || !selection.uuids.empty() || selection.connectedTo || selection.tag ||
                            !selection.tagGroups.empty();
    // Force primary-key candidate probes instead of scanning objects_kind_uuid
    // to avoid a sort when the R-tree returned only a few matches.
    const bool typed = selection.family || selection.minLength || selection.orderBy == "length_desc";
    const auto table = selection.kind == "fiber" ? "fibers" : "tracks";
    std::string from = "objects o" + std::string(candidates ? " NOT INDEXED" : "");
    if (typed) {
        if (!candidates) from = std::string(table) + " d CROSS JOIN objects o ON o.id=d.object_id";
        else from += std::string(" JOIN ") + table + " d ON d.object_id=o.id";
    }
    std::string sql = "SELECT " + columns + " FROM " + from + " WHERE 1" + filter.sql;
    if (selection.bbox) {
        detail::validateBounds(*selection.bbox);
        sql += " AND o.id IN (SELECT b.object_id FROM spatial_regions_rtree r CROSS JOIN spatial_regions b ON b.id=r.id WHERE b.array_name='' AND ";
        sql += detail::spatialOverlapSql(int(filter.parameters.size()) + 1) + ")";
    }
    sql += std::string(" ORDER BY ") + (selection.orderBy == "length_desc" ? "d.length DESC,o.uuid" : "o.uuid") + " LIMIT ? OFFSET ?";
    auto rows = std::make_unique<Statement>(db, sql);
    bindSelection(*rows, filter, selection);
    return rows;
}

namespace {
std::vector<ObjectRecord> selectPhysicalObjects(sqlite3* db, const ObjectSelection& selection,
    const ObjectFields& fields)
{
    if(selection.bbox)requireSpatial(db);
    if (!selection.volume && !selection.includeUnbound && !selection.minLength && selection.orderBy!="length_desc") {
        auto rows = objectSelectionQuery(db, selection, projectedObjectColumns(fields));
        auto records = readObjectRows(db, *rows, fields);
        return records;
    }
    const auto order = measuredSelection(db, selection);
    if (order.empty()) return {};
    ObjectSelection chosen;
    chosen.includeRetired = selection.includeRetired;
    for (const auto& row : order) chosen.uuids.push_back(row.uuid);
    auto rows = objectSelectionQuery(db, chosen, projectedObjectColumns(fields));
    auto records = readObjectRows(db, *rows, fields);
    std::map<std::string, size_t> position;
    for (size_t i = 0; i < order.size(); ++i) position.emplace(order[i].uuid, i);
    std::sort(records.begin(), records.end(), [&](const ObjectRecord& a, const ObjectRecord& b) {
        return position.at(a.uuid) < position.at(b.uuid);
    });
    if (selection.orderBy == "length_desc")
        for (auto& record : records)
            if (const auto& row = order[position.at(record.uuid)]; row.bound) record.volumeLength=row.volume;
    return records;
}

bool restrictConnected(const ObjectStore& store,const Txn& txn,ObjectSelection& selection)
{
    if(!selection.connectedTo)return true;
    auto component=store.componentOf(txn,*selection.connectedTo,selection.edgeKinds,nullptr,selection.includeRetired);
    if(!selection.uuids.empty())std::erase_if(component,[&](const auto& id){return std::none_of(selection.uuids.begin(),selection.uuids.end(),[&](const auto& u){return formatUuid(parseUuid(u))==id;});});
    selection.uuids=std::move(component);selection.connectedTo.reset();
    return !selection.uuids.empty();
}
}

void detail::scanVisibleObjectRows(sqlite3* db,const ObjectSelection& selection,const ObjectFields& fields,
    const std::function<bool(ObjectRecord&&,std::optional<SelectedGeometryHead>)>& visit,bool includeGeometryHeads)
{
    if(selection.limit&&*selection.limit==0) {
        (void)objectSelectionQuery(db,selection,projectedObjectColumns(fields));
        return;
    }
    auto physical=selection;physical.limit.reset();physical.offset=0;
    const auto columns=projectedObjectColumns(fields)+(includeGeometryHeads?
        ",CASE WHEN geometry_manifest IS NOT NULL THEN storage_ref ELSE NULL END,geometry_manifest":",NULL,NULL");
    VisibleObjectRows rows(db,[&](auto* source){return objectSelectionQuery(source,physical,columns);},1);
    struct Batch {
        std::vector<ObjectRecord> records;
        std::vector<int64_t> ids;
        std::vector<size_t> positions;
        std::string owner,generation;
    };
    std::map<sqlite3*,Batch> batches;
    for(auto* source:querySources(db)) {
        auto& batch=batches[source];
        Statement identity(source,"SELECT s.uuid,m.initial_generation FROM store_identity s CROSS JOIN store_manifest m WHERE s.singleton=1 AND m.singleton=1");
        if(!identity.row())throw std::runtime_error("object scan: source identity is missing");
        batch.owner=identity.uuid(0);batch.generation=identity.uuid(1);
    }
    int64_t skipped=0,emitted=0;
    const size_t capacity=selection.limit?size_t(std::min<int64_t>(*selection.limit,256)):256;
    std::vector<ObjectRecord> ordered;ordered.reserve(capacity);
    std::vector<std::optional<SelectedGeometryHead>> heads;heads.reserve(capacity);
    auto flush=[&]{
        for(auto& [source,batch]:batches) {
            if(batch.records.empty())continue;
            hydrateObjectRows(source,batch.records,batch.ids,fields);
            for(size_t i=0;i<batch.records.size();++i)ordered[batch.positions[i]]=std::move(batch.records[i]);
            batch.records.clear();batch.ids.clear();batch.positions.clear();
        }
        for(size_t i=0;i<ordered.size();++i)if(!visit(std::move(ordered[i]),std::move(heads[i])))return false;
        ordered.clear();heads.clear();return true;
    };
    while(rows.row()) {
        if(skipped<selection.offset){++skipped;continue;}
        auto& batch=batches.at(rows.source());
        batch.ids.push_back(rows.values().i64(0));batch.positions.push_back(ordered.size());
        batch.records.push_back(readObjectRow(rows.source(),rows.values(),false,&batch.generation,false,&batch.owner));
        if(rows.values().isNull(24))heads.emplace_back();
        else heads.push_back(SelectedGeometryHead{rows.source(),readGeometryTokenColumn(rows.values(),23),rows.values().text(24)});
        ordered.emplace_back();++emitted;
        if(ordered.size()==capacity&&!flush())return;
        if(selection.limit&&emitted==*selection.limit)break;
    }
    if(!ordered.empty())flush();
}

void detail::scanVisibleObjects(sqlite3* db,const ObjectSelection& selection,const ObjectFields& fields,
                                const std::function<bool(ObjectRecord&&)>& visit)
{
    scanVisibleObjectRows(db,selection,fields,[&](ObjectRecord&& row,std::optional<SelectedGeometryHead>){
        return visit(std::move(row));
    },false);
}

void detail::scanVisibleObjectReferences(sqlite3* db,const ObjectSelection& selection,
                                       const std::function<bool(ObjectVersion&&)>& visit)
{
    constexpr auto columns="o.uuid,o.version,o.version_generation";
    if(selection.limit&&*selection.limit==0){
        (void)objectSelectionQuery(db,selection,columns);return;
    }
    auto physical=selection;physical.limit.reset();physical.offset=0;
    VisibleObjectRows rows(db,[&](auto* source){return objectSelectionQuery(source,physical,columns);},0);
    struct SourceIdentity {std::string owner,initialGeneration;};
    std::map<sqlite3*,SourceIdentity> sources;
    for(auto* source:querySources(db)){
        Statement identity(source,"SELECT s.uuid,m.initial_generation FROM store_identity s CROSS JOIN store_manifest m WHERE s.singleton=1 AND m.singleton=1");
        if(!identity.row())throw std::runtime_error("reference scan: source identity is missing");
        sources.emplace(source,SourceIdentity{identity.uuid(0),identity.uuid(1)});
    }
    int64_t skipped=0,emitted=0;
    while(rows.row()){
        if(skipped<selection.offset){++skipped;continue;}
        const auto& source=sources.at(rows.source());
        auto reference=readObjectReferenceColumns(rows.source(),rows.values(),0,1,2,
            &source.initialGeneration,&source.owner,&rows.key());
        ++emitted;
        if(!visit(std::move(reference)))return;
        if(selection.limit&&emitted==*selection.limit)return;
    }
}

std::vector<ObjectRecord> ObjectStore::selectObjects(const Txn& txn, const ObjectSelection& requested,
    const std::optional<std::vector<std::string>>& projection) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    auto selection=requested;
    if(selection.offset<0||(selection.limit&&*selection.limit<0))throw std::invalid_argument("selection limit and offset must be nonnegative");
    if(selection.volume&&!selection.transforms)selection.transforms=std::make_shared<VolumeTransforms>(volumeTransforms(txn));
    const ObjectFields fields=projection?ObjectFields(std::in_place,projection->begin(),projection->end()):std::nullopt;
    const bool composed=querySources(db).size()>1;
    if(composed&&!restrictConnected(*this,txn,selection))return {};
    if(!selection.volume&&!selection.includeUnbound&&!selection.minLength&&selection.orderBy=="uuid") {
        std::vector<ObjectRecord> result;
        scanVisibleObjects(db,selection,fields,[&](ObjectRecord&& row){result.push_back(std::move(row));return true;});
        return result;
    }
    if(!composed)return selectPhysicalObjects(db,selection,fields);
    auto physical=selection;physical.limit.reset();physical.offset=0;
    return selectVisibleRecords(db,[&](auto* source){return selectPhysicalObjects(source,physical,fields);},
        [&](const auto& record,auto* source){return objectSource(db,record.uuid)==source;},[&](const auto& a,const auto& b){
        if(selection.orderBy=="length_desc"){
            const auto x=a.volumeLength,y=b.volumeLength;
            if(x!=y)return x>y;
        }
        return a.uuid<b.uuid;
    },selection.offset,selection.limit);
}

int64_t ObjectStore::unboundObjectCount(const Txn& txn, const std::string& kind, const std::string& volume) const
{
    const auto context=volumeTransforms(txn);
    if(!context.graph.registrations().empty()||querySources(static_cast<sqlite3*>(connection(txn))).size()>1) {
        int64_t result=0;
        for(const auto& object:objectsByKind(txn,kind))
            if(!context.toVolume(parseObjectJson(object.meta),volume))++result;
        return result;
    }
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement count(db, "SELECT count(*) FROM objects o NOT INDEXED WHERE o.id IN (" + unboundObjectIdsSql("?1", "?2") +
                        ") AND " + activeObjectSql("o.id"), kind, volume);
    (void)count.row();
    return count.i64(0);
}

namespace {
std::vector<SpatialRegion> selectPhysicalRegions(sqlite3* db, const ObjectSelection& selection)
{
    requireSpatial(db);
    if (selection.orderBy != "uuid") throw std::invalid_argument("spatial regions use UUID order");
    if (!selection.bbox) throw std::invalid_argument("spatial regions require a bbox");
    detail::validateBounds(*selection.bbox);
    if (selection.includeUnbound) throw std::invalid_argument("spatial regions place only objects bound to a volume");
    if(selection.minLength) {
        auto objects=selection;objects.offset=0;objects.limit.reset();objects.afterUuid.reset();objects.afterLength.reset();
        const auto matching=measuredSelection(db,objects);
        if(matching.empty())return {};
        auto filtered=selection;filtered.minLength.reset();filtered.uuids.clear();
        for(const auto& row:matching)filtered.uuids.push_back(row.uuid);
        return selectPhysicalRegions(db,filtered);
    }
    if (selection.volume) {
        requireVolumeSelection(selection);
        std::vector<SpatialRegion> regions;
        for (const auto& frame : selectionFrames(db, selection)) {
            auto sub = frameSelection(selection, frame);
            sub.afterUuid = selection.afterUuid;
            for (auto region : selectPhysicalRegions(db, sub)) {
                // Mapped stored bounds stay conservative: the map is monotonic.
                region.bbox = transformBounds(region.bbox,frame.mapping);
                regions.push_back(std::move(region));
            }
        }
        std::sort(regions.begin(), regions.end(), [](const SpatialRegion& a, const SpatialRegion& b) {
            return std::tie(a.objectUuid, a.arrayName, a.start) < std::tie(b.objectUuid, b.arrayName, b.start);
        });
        const size_t begin = std::min(regions.size(), size_t(selection.offset));
        const size_t end = selection.limit ? std::min(regions.size(), begin + size_t(*selection.limit)) : regions.size();
        return {std::make_move_iterator(regions.begin() + begin), std::make_move_iterator(regions.begin() + end)};
    }
    if (selection.offset < 0 || (selection.limit && *selection.limit < 0))
        throw std::invalid_argument("selection limit and offset must be nonnegative");
    if (const auto lease = activeStatementPools.find(db); lease != activeStatementPools.end())
        if (const auto& cache = lease->second->spatialCache)
            if (auto result = cache->query(db, selection)) return *result;
    const auto filter = objectFilter(db, selection);
    // CROSS JOIN preserves the R-tree-first plan even with a kind filter.
    std::string sql = std::string("SELECT ") + detail::regionColumns + " FROM spatial_regions_rtree r "
        "CROSS JOIN spatial_regions b ON b.id=r.id CROSS JOIN objects o ON o.id=b.object_id "
        "CROSS JOIN object_bounds h ON h.object_id=b.object_id " +
        ((selection.family || selection.minLength) ?
            std::string(" JOIN ") + (selection.kind == "fiber" ? "fibers" : "tracks") + " d ON d.object_id=o.id " : "") +
        "WHERE (b.array_name!='' OR h.revision_uuid IS NULL)" + filter.sql + " AND " +
        detail::spatialOverlapSql(int(filter.parameters.size()) + 1) +
        " ORDER BY o.uuid,b.array_name,b.row_start,b.col_start LIMIT ? OFFSET ?";
    Statement rows(db, sql);
    bindSelection(rows, filter, selection);
    std::vector<SpatialRegion> result;
    while (rows.row()) result.push_back(detail::readSpatialRegion(rows));
    return result;
}
}

std::vector<SpatialRegion> ObjectStore::spatialRegions(const Txn& txn,const ObjectSelection& requested) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    auto selection=requested;
    if(selection.offset<0||(selection.limit&&*selection.limit<0))throw std::invalid_argument("selection limit and offset must be nonnegative");
    if(selection.volume&&!selection.transforms)selection.transforms=std::make_shared<VolumeTransforms>(volumeTransforms(txn));
    if(querySources(db).size()==1)return selectPhysicalRegions(db,selection);
    if(!restrictConnected(*this,txn,selection))return {};
    auto physical=selection;physical.offset=0;physical.limit.reset();
    return selectVisibleRecords(db,[&](auto* source){return selectPhysicalRegions(source,physical);},
        [&](const auto& region,auto* source){return objectSource(db,region.objectUuid)==source;},[](const auto& a,const auto& b){
        return std::tie(a.objectUuid,a.arrayName,a.start)<std::tie(b.objectUuid,b.arrayName,b.start);
    },selection.offset,selection.limit);
}

} // namespace vc::objects
