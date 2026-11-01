#include "geometry/SharedGeometry.hpp"
#include "geometry/SequenceMetadata.hpp"
#include "geometry/SequenceIdentity.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "internal/History.hpp"

namespace vc::objects {
using namespace detail;
namespace {
using Json=nlohmann::json;
void checked(sqlite3* db) {
    if (!activeStatementPools.at(db)->checkedAction) throw std::runtime_error("structural editing requires write(expected revisions)");
}
void requireSource(sqlite3* db,const std::string& uuid) {
    execute(db,"INSERT OR IGNORE INTO history_required VALUES(?)", uuid);
}
std::vector<int64_t> idsOf(GeometryReader& reader,const std::string& ids,int64_t count) {
    std::vector<int64_t> values(count);
    copyScalars<int64_t>(reader.read(ids,0),values);
    std::set<int64_t> seen;
    for (auto id: values) if (id<0 || !seen.insert(id).second) throw std::invalid_argument("point IDs must be nonnegative and unique");
    if (geometryWork) geometryWork->identityRows+=values.size();
    return values;
}
Json annotations(const Txn& txn,const GeometryRevision& revision,GeometryReader& reader,const std::string& name) {
    const auto count=reader.manifest().at("arrays").at(name).at("shape").at(0).get<int64_t>();
    auto result=Json(std::vector<Json>(count,Json::object()));
    const auto& state = reader.manifest().at("sequences").at(name);
    const auto ids = sequenceIdArray(revision.kind, name);
    auto* db = ObjectStoreImplAccess::conn(txn);
    MetadataSnapshot snapshot(db, state.at(ids.empty() ? "annotations" : "identities"));
    const auto values = ids.empty() ? std::vector<int64_t>() : idsOf(reader, ids, count);
    std::vector<bool> seen(count);
    snapshot.entries([&](int64_t id, const auto& value) {
        const int64_t row = ids.empty() ? id : value.at("row").template get<int64_t>();
        if (row < 0 || row >= count || seen[row] || (!ids.empty() && values[row] != id))
            throw std::runtime_error("invalid annotation row");
        seen[row] = true;
        result[row] = unpackAnnotation(db, ids.empty() ? value : value.at("annotation"));
    });
    if (!ids.empty() && std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::runtime_error("missing point identity");
    return result;
}
bool directional(const Json& value) {return annotationDirectional(value);}
GeometryData snapshotData(GeometryReader& reader) {
    const auto& m=reader.manifest();
    GeometryData data; data.kind=m.at("kind").get<std::string>(); data.formatVersion=m.at("format_version");
    data.coordinateSystem=m.at("coordinate_system").get<std::string>(); data.attributes=m.at("attributes").dump();
    for(const auto& [name,a]:m.at("arrays").items())
        data.arrays.push_back({name,a.at("dtype"),a.at("shape"),reader.read(name,0)});
    return data;
}
GeometryArray& arrayNamed(GeometryData& data,const std::string& name) {
    return *std::find_if(data.arrays.begin(),data.arrays.end(),[&](const auto& a){return a.name==name;});
}
}

std::string ObjectStore::geometryPointAnnotations(const Txn& txn,const std::string& revision,const std::string& name) const {
    MetadataCacheScope cache;
    const auto r=geometryRevision(txn,revision);
    if (!r) throw std::invalid_argument("geometry revision not found");
    GeometryReader reader(txn,revision);
    return annotations(txn,*r,reader,name).dump();
}
std::string ObjectStore::editGeometrySequence(WriteTxn& txn,const std::string& objectId,const GeometrySequenceEdit& edit,
    const std::string& expected,const std::vector<GeometryEdgeEdit>& edges)
{
    GeometryPhase phase("ObjectStore::editGeometrySequence");
    auto* db=static_cast<sqlite3*>(connection(txn)); checked(db); Savepoint savepoint(db); MetadataCacheScope cache;
    try {
        const auto uuid=formatUuid(parseUuid(objectId)); const auto current=currentGeometry(txn,uuid);
        if (!current || current->uuid!=formatUuid(parseUuid(expected))) historyConflict(db,"geometry: current revision changed; reload before committing");
        if (current->kind=="surface" || (current->kind=="fiber" ? edit.array!="control_points" && edit.array!="line_points" : edit.array!="points"))
            throw std::invalid_argument("invalid structural sequence array");
        GeometryReader reader(txn,current->uuid); auto manifest=reader.manifest();
        auto data=snapshotData(reader);
        const auto& a=manifest.at("arrays").at(edit.array); const int64_t oldCount=a.at("shape").at(0);
        const auto stride=geometryByteLength(a.at("dtype"),{3}); const std::string dtype=a.at("dtype");
        if (edit.start<0 || edit.eraseCount<0 || edit.start>oldCount || edit.eraseCount>oldCount-edit.start) throw std::out_of_range("sequence span out of range");
        const auto idsName=sequenceIdArray(current->kind,edit.array);
        const auto oldAttrs=annotations(txn,*current,reader,edit.array);
        auto state=manifest.at("sequences").at(edit.array);
        MetadataSnapshot allocated(db,state.value("allocated",Json()));
        std::vector<int64_t> originalIds;
        if(!idsName.empty()) {
            originalIds=idsOf(reader,idsName,oldCount);
            // Restoration must not allow a previously allocated ID to be reused.
            Statement latest(db,"SELECT manifest FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation DESC,rowid DESC LIMIT 1",
                parseUuid(uuid));
            if(latest.row()) {
                const auto tip=Json::parse(latest.text(0));
                if(tip.at("sequences").at(edit.array).at("allocated")!=state.at("allocated"))
                    allocated.mergeAllocated(MetadataSnapshot(db,tip.at("sequences").at(edit.array).at("allocated")));
            }
        }
        const auto tail=edit.start+edit.eraseCount;
        std::set<int64_t> liveIds;
        std::vector<int64_t> changedIds;
        if(!idsName.empty()) {
            changedIds.assign(originalIds.begin()+edit.start,originalIds.begin()+tail);
            for(int64_t i=0;i<oldCount;++i)if(i<edit.start||i>=tail)liveIds.insert(originalIds[i]);
        }
        const auto& original=arrayNamed(data,edit.array).bytes;
        std::vector<uint8_t> coords(original.begin(),original.begin()+edit.start*stride);
        std::vector<int64_t> ids(originalIds.begin(),originalIds.begin()+(idsName.empty()?0:edit.start));
        Json attrs=Json::array();for(int64_t i=0;i<edit.start;++i)attrs.push_back(oldAttrs.at(i));
        bool needsTransform=directional(oldAttrs);
        int64_t count=edit.start;
        for(const auto& part:edit.parts) {
            Json pointAttrs;std::vector<int64_t> pointIds;std::vector<uint8_t> partCoords;
            if(part.revision) {
                if(!part.bytes.empty()||!part.ids.empty()||part.annotations!="[]")throw std::invalid_argument("reference parts cannot supply replacement point data");
                const auto sourceRevision=geometryRevision(txn,*part.revision);
                if(!sourceRevision||sourceRevision->kind!=current->kind)throw std::invalid_argument("sequence source kind mismatch");
                requireSource(db,sourceRevision->objectUuid);GeometryReader source(txn,*part.revision);
                const auto& descriptor=source.manifest().at("arrays").at(edit.array);const int64_t n=descriptor.at("shape").at(0);
                if(part.start<0||part.count<0||part.start>n||part.count>n-part.start)throw std::out_of_range("sequence source span out of range");
                if(descriptor.at("dtype")!=dtype)throw std::invalid_argument("sequence source layout mismatch");
                partCoords=source.read(edit.array,part.start*stride,part.count*stride);
                if(!idsName.empty()) {
                    pointIds.resize(part.count);copyScalars<int64_t>(source.read(idsName,part.start*8,part.count*8),pointIds);
                }
                const auto all=annotations(txn,*sourceRevision,source,edit.array);
                pointAttrs=Json::array();for(auto i=part.start;i<part.start+part.count;++i)pointAttrs.push_back(all.at(i));
            } else {
                if(part.start||part.count<0||uint64_t(part.count)>uint64_t(INT64_MAX/stride)||part.bytes.size()!=uint64_t(part.count*stride))throw std::invalid_argument("new sequence part byte length mismatch");
                if(!idsName.empty()&&part.ids.size()!=size_t(part.count))throw std::invalid_argument("new points require stable IDs");
                if(idsName.empty()&&!part.ids.empty())throw std::invalid_argument("sample sequences do not accept stable IDs");
                pointIds=part.ids;
                for(auto id:pointIds)if(id<0||!allocated.get(id).is_null())throw std::invalid_argument("point ID already used; reference its revision to restore it");
                partCoords=part.bytes;pointAttrs=Json::parse(part.annotations);
                if(pointAttrs.is_array()&&pointAttrs.empty())pointAttrs=Json(std::vector<Json>(part.count,Json::object()));
            }
            if(!pointAttrs.is_array()||pointAttrs.size()!=size_t(part.count))throw std::invalid_argument("new point annotations must match rows");
            for(const auto& value:pointAttrs)if(!value.is_object()&&!value.is_null())throw std::invalid_argument("invalid point annotation");
            needsTransform|=directional(pointAttrs);
            if(count>INT64_MAX/stride-part.count)throw std::overflow_error("sequence length overflow");
            if(part.reverse) {
                for(int64_t i=0;i<part.count/2;++i)for(int64_t j=0;j<stride;++j)std::swap(partCoords[i*stride+j],partCoords[(part.count-i-1)*stride+j]);
                std::reverse(pointIds.begin(),pointIds.end());std::reverse(pointAttrs.begin(),pointAttrs.end());
            }
            coords.insert(coords.end(),partCoords.begin(),partCoords.end());
            for(const auto& value:pointAttrs)attrs.push_back(value);
            for(auto id:pointIds) {
                if(id<0||!liveIds.insert(id).second)throw std::invalid_argument("duplicate resulting point ID");
                allocated.put(id,true);changedIds.push_back(id);
            }
            ids.insert(ids.end(),pointIds.begin(),pointIds.end());count+=part.count;
        }
        const auto remaining=oldCount-tail;
        if(count>INT64_MAX/stride-remaining)throw std::overflow_error("sequence length overflow");
        count+=remaining;coords.insert(coords.end(),original.begin()+tail*stride,original.end());
        for(auto i=tail;i<oldCount;++i)attrs.push_back(oldAttrs.at(i));
        if(!idsName.empty())ids.insert(ids.end(),originalIds.begin()+tail,originalIds.end());
        if(needsTransform&&!edit.transformedAnnotations)throw std::invalid_argument("structural edit requires transformed span/direction annotations");
        if(edit.transformedAnnotations) {
            attrs=Json::parse(*edit.transformedAnnotations);
            if(!attrs.is_array()||attrs.size()!=size_t(count))throw std::invalid_argument("transformed annotations must match resulting rows");
            for(const auto& value:attrs)if(!value.is_object()&&!value.is_null())throw std::invalid_argument("invalid transformed annotation");
        }
        auto& coordinates=arrayNamed(data,edit.array);coordinates.shape[0]=count;coordinates.bytes=std::move(coords);
        MetadataSnapshot metadata(db);
        for(int64_t i=0;i<count;++i) {
            const auto annotation=packAnnotation(db,attrs.at(i));
            if (!idsName.empty() || !attrs.at(i).is_object() || !attrs.at(i).empty())
                metadata.put(idsName.empty()?i:ids[i],idsName.empty()?annotation:Json{{"row",i},{"annotation",annotation}});
        }
        if(idsName.empty())state={{"annotations",metadata.save()}};
        else {
            auto array=scalarArray<int64_t>(idsName,"<i8",{count},ids);
            if(manifest.at("arrays").contains(idsName))arrayNamed(data,idsName)=std::move(array);else data.arrays.push_back(std::move(array));
            state={{"identities",metadata.save()},{"allocated",allocated.save()}};
        }
        manifest["sequences"][edit.array]=state;
        const auto result=publishStructuralSnapshot(*this,txn,uuid,data,std::move(manifest),expected,edges,idsName.empty()?std::nullopt:std::optional(std::pair{idsName,changedIds}));
        savepoint.release();return result;
    } catch(const std::exception& error) {historyConflict(db,error.what());}
}

std::string ObjectStore::resizeSurface(WriteTxn& txn,const std::string& objectId,const GeometrySurfaceResize& resize,
    const std::string& expected,const std::vector<GeometryEdgeEdit>& edges)
{
    GeometryPhase phase("ObjectStore::resizeSurface");
    auto* db=static_cast<sqlite3*>(connection(txn)); checked(db); Savepoint savepoint(db); MetadataCacheScope cache;
    try {
        const auto uuid=formatUuid(parseUuid(objectId)); const auto current=currentGeometry(txn,uuid);
        if (!current || current->uuid!=formatUuid(parseUuid(expected))) historyConflict(db,"geometry: current revision changed; reload before committing");
        if (current->kind!="surface") throw std::invalid_argument("surface resize requires a surface");
        GeometryReader reader(txn,current->uuid); auto manifest=reader.manifest();
        if (resize.rows<0 || resize.columns<0) throw std::invalid_argument("negative surface shape");
        const auto origin=manifest.value("grid_origin",std::array<int64_t,2>{0,0});
        auto add=[](int64_t a,int64_t b) { if ((b>0 && a>INT64_MAX-b) || (b<0 && a<INT64_MIN-b)) throw std::overflow_error("surface origin overflow"); return a+b; };
        (void)add(resize.row,resize.rows); (void)add(resize.column,resize.columns);
        const auto oy=add(origin[0],resize.row), ox=add(origin[1],resize.column);
        // Leave one tile of headroom so grid arithmetic cannot overflow.
        if (oy<INT64_MIN+128 || ox<INT64_MIN+128 || add(oy,resize.rows)>INT64_MAX-128 || add(ox,resize.columns)>INT64_MAX-128)
            throw std::overflow_error("surface grid coordinate overflow");
        std::map<std::string,GeometryArray> fills;
        for (const auto& fill:resize.fill) if (!fills.emplace(fill.name,fill).second) throw std::invalid_argument("duplicate fill channel");
        if (fills.size()!=manifest.at("arrays").size()) throw std::invalid_argument("explicit fill required for every surface array");
        auto data=snapshotData(reader);
        const auto grid=manifest.at("arrays").at("points").at("shape").get<std::vector<int64_t>>();
        for (auto& a:data.arrays) {
            const auto shape=a.shape;
            // Channels finer than the grid resize by the same grid rectangle.
            const auto [ry,rx]=surfaceChannelResolution(a.name,shape,grid);
            if ((ry>1 && resize.rows==0) || (rx>1 && resize.columns==0))
                throw std::invalid_argument("surface resize cannot empty a grid axis carrying finer channels");
            auto scaled=[](int64_t value,int64_t factor) {
                if (value>INT64_MAX/factor || value<INT64_MIN/factor) throw std::overflow_error("surface channel shape overflow");
                return value*factor;
            };
            const auto row=scaled(resize.row,ry), column=scaled(resize.column,rx), rows=scaled(resize.rows,ry), columns=scaled(resize.columns,rx);
            auto newShape=shape; newShape[0]=rows; newShape[1]=columns;
            const auto length=geometryByteLength(a.dtype,newShape), pixel=surfacePixelBytes(a.dtype,shape);
            const auto found=fills.find(a.name);
            if(found==fills.end())throw std::invalid_argument("missing surface fill");
            auto fillShape=shape; fillShape[0]=fillShape[1]=1; const auto& fill=found->second;
            if(fill.dtype!=a.dtype||fill.shape!=fillShape||int64_t(fill.bytes.size())!=pixel)throw std::invalid_argument("surface fill must be one pixel of matching dtype/components");
            std::vector<uint8_t> bytes(length);
            for(int64_t i=0;i<length;i+=pixel)std::memcpy(bytes.data()+i,fill.bytes.data(),pixel);
            const auto top=std::max<int64_t>(0,row),left=std::max<int64_t>(0,column);
            const auto bottom=std::min(shape[0],row+rows),right=std::min(shape[1],column+columns);
            if(top<bottom&&left<right)for(auto r=top;r<bottom;++r)
                std::memcpy(bytes.data()+((r-row)*columns+left-column)*pixel,
                    a.bytes.data()+(r*shape[1]+left)*pixel,(right-left)*pixel);
            a.shape=std::move(newShape); a.bytes=std::move(bytes);
        }
        manifest["grid_origin"]={oy,ox};
        const auto result=publishStructuralSnapshot(*this,txn,uuid,data,std::move(manifest),expected,edges);
        savepoint.release(); return result;
    } catch (const std::exception& error) { historyConflict(db,error.what()); }
}
}
