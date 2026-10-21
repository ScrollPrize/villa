#include "geometry/SharedGeometry.hpp"
#include "geometry/PersistentTree.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "internal/History.hpp"
#include <numeric>
#include <unordered_map>

namespace vc::objects {
using namespace detail;
namespace {
using Json=nlohmann::json;
void checked(sqlite3* db) {
    if (!activeStatementPools.at(db)->checkedAction) throw std::runtime_error("structural editing requires write(expected revisions)");
}
void requireSource(sqlite3* db,const std::string& uuid) {
    Statement required(db,"INSERT OR IGNORE INTO history_required VALUES(?)"); required.bind(1,uuid); required.stepDone();
}
std::string idArray(const std::string& kind,const std::string& name) {
    if (kind=="point_collection" && name=="points") return "point_ids";
    if (kind=="fiber" && name=="control_points") return "control_point_ids";
    return {};
}
std::vector<int64_t> idsOf(GeometryReader& reader,const std::string& ids,int64_t count) {
    std::vector<int64_t> values(count);
    if (reader.manifest().at("arrays").contains(ids)) copyScalars<int64_t>(reader.read(ids,0),values);
    else std::iota(values.begin(),values.end(),0);
    std::set<int64_t> seen;
    for (auto id: values) if (id<0 || !seen.insert(id).second) throw std::invalid_argument("point IDs must be nonnegative and unique");
    if (geometryWork) geometryWork->identityRows+=values.size();
    return values;
}
Json annotations(ObjectStore const& store,const Txn& txn,const GeometryRevision& revision,GeometryReader& reader,const std::string& name) {
    const auto count=reader.manifest().at("arrays").at(name).at("shape").at(0).get<int64_t>();
    auto result=Json(std::vector<Json>(count,Json::object()));
    if (reader.manifest().contains("sequences") && reader.manifest().at("sequences").contains(name)) {
        const auto& state=reader.manifest().at("sequences").at(name);
        const auto ids=idArray(revision.kind,name);
        auto* db=ObjectStoreImplAccess::conn(txn);
        if(state.contains("annotation_tree")) {
            result=Json::array();treeWalk(db,state.at("annotation_tree"),[&](const auto& v){for(const auto& a:v)result.push_back(unpackAnnotation(db,a));});return result;
        }
        const auto values=idsOf(reader,ids,count);
        std::unordered_map<std::string,Json> buckets;
        treeEntries(db,state.at("identities"),[&](const auto& ref,const auto& value){buckets.emplace(ref.at("first").template get<std::string>(),value);});
        for(size_t i=0;i<values.size();++i)result[i]=unpackAnnotation(db,buckets.at(identityKey(values[i])).at(std::to_string(values[i]%128)).at("annotation"));
        return result;
    }
    const auto key=revision.kind=="point_collection" ? "point_attributes" : name+"_attributes";
    // Recover annotations belonging to this geometry, including historical
    // source slices. Unpublished geometry uses the transaction's current record.
    auto object=*store.objectByUuid(txn,revision.objectUuid);
    const auto current=store.currentGeometry(txn,revision.objectUuid);
    if (!current || current->uuid!=revision.uuid) {
        bool found=false;
        for (const auto& history:store.objectHistory(txn,revision.objectUuid))
            if (history.geometryRevision==revision.uuid) { object=history.record; found=true; break; }
        if (!found) throw std::runtime_error("historical sequence lacks revisioned annotations");
    }
    const auto source=Json::parse(object.meta).value("source",Json::object());
    if (source.contains(key)) result=source.at(key);
    if (!result.is_array() || result.size()!=size_t(count)) throw std::invalid_argument("point annotations must match the sequence");
    for (const auto& a:result) if (!a.is_object() && !a.is_null()) throw std::invalid_argument("invalid point annotation");
    return result;
}
bool directional(const Json& value) {return annotationDirectional(value);}
int64_t floorTile(int64_t value,int64_t tile) { auto q=value/tile; if (value%tile<0) --q; return q*tile; }
}

std::string ObjectStore::geometryPointAnnotations(const Txn& txn,const std::string& revision,const std::string& name) const {
    TreeCacheScope cache;
    const auto r=geometryRevision(txn,revision);
    if (!r) throw std::invalid_argument("geometry revision not found");
    GeometryReader reader(txn,revision);
    return annotations(*this,txn,*r,reader,name).dump();
}
std::string ObjectStore::editGeometrySequence(WriteTxn& txn,const std::string& objectId,const GeometrySequenceEdit& edit,
    const std::string& expected,const std::vector<GeometryEdgeEdit>& edges)
{
    GeometryPhase phase("ObjectStore::editGeometrySequence");
    auto* db=static_cast<sqlite3*>(connection(txn)); checked(db); Savepoint savepoint(db); TreeCacheScope cache;
    try {
        const auto uuid=formatUuid(parseUuid(objectId)); const auto current=currentGeometry(txn,uuid);
        if (!current || current->uuid!=formatUuid(parseUuid(expected))) historyConflict(db,"geometry: current revision changed; reload before committing");
        if (current->kind=="patch" || (current->kind=="fiber" ? edit.array!="control_points" && edit.array!="line_points" : edit.array!="points"))
            throw std::invalid_argument("invalid structural sequence array");
        GeometryReader reader(txn,current->uuid); auto manifest=reader.manifest();
        for (auto& [name,a]:manifest.at("arrays").items())
            a=indexedDescriptor(db,reader,name,name=="points"||name=="control_points"||name=="line_points");
        auto& a=manifest.at("arrays").at(edit.array); const int64_t oldCount=a.at("shape").at(0);
        const auto stride=geometryByteLength(a.at("dtype"),{3}); const std::string dtype=a.at("dtype");
        if (edit.start<0 || edit.eraseCount<0 || edit.start>oldCount || edit.eraseCount>oldCount-edit.start) throw std::out_of_range("sequence span out of range");
        const auto idsName=idArray(current->kind,edit.array);
        auto state=manifest.value("sequences",Json::object()).value(edit.array,Json::object());
        auto attributeTree=[&](const Json& attrs) {
            std::vector<Tree> leaves;
            for(size_t i=0;i<attrs.size();i+=128) {
                Json values=Json::array(); for(size_t j=i;j<std::min(i+128,attrs.size());++j) values.push_back(packAnnotation(db,attrs[j]));
                bool spans=false;for(const auto& v:values)spans|=v.at("span_fields").get<bool>();
                leaves.push_back(treeLeaf(db,values,values.size(),nullptr,spans));
            }
            return treeBuild(db,leaves);
        };
        auto attributeSlice=[&](const Tree& root,int64_t start,int64_t count) {
            return treeSlice(db,root,start,count,[&](const Json& v,int64_t begin,int64_t n) {
                Json attrs=Json::array(); for(int64_t i=begin;i<begin+n;++i) attrs.push_back(v.at(i));
                bool spans=false;for(const auto& v:attrs)spans|=v.at("span_fields").get<bool>();
                return treeLeaf(db,attrs,n,nullptr,spans);
            });
        };
        if (!state.contains("identities") && !state.contains("annotation_tree")) {
            const auto attrs=annotations(*this,txn,*current,reader,edit.array);
            if (idsName.empty()) state={{"annotation_tree",attributeTree(attrs)}};
            else {
                const auto ids=idsOf(reader,idsName,oldCount);
                if (!manifest.at("arrays").contains(idsName)) manifest["arrays"][idsName]={{"dtype","<i8"},{"shape",{oldCount}},
                    {"byte_length",oldCount*8},{"tree",byteTree(db,idsName,"<i8",scalarBytes<int64_t>(ids),false)}};
                const auto bytes=reader.read(edit.array,0);
                std::map<std::string,Json> buckets,allocated;
                for(int64_t i=0;i<oldCount;++i) {
                    const auto key=identityKey(ids[i]),slot=std::to_string(ids[i]%128);
                    buckets[key][slot]={{"bytes",std::vector<uint8_t>(bytes.begin()+i*stride,bytes.begin()+(i+1)*stride)},{"annotation",packAnnotation(db,attrs.at(i))}};
                    allocated[key][slot]=true;
                }
                std::vector<Tree> leaves,ledger;
                for(const auto& [key,bucket]:buckets) {
                    bool spans=false;for(const auto& v:bucket)spans|=v.at("annotation").at("span_fields").get<bool>();
                    leaves.push_back(treeLeaf(db,bucket,1,nullptr,spans,key));
                }
                for(const auto& [key,bucket]:allocated) ledger.push_back(treeLeaf(db,bucket,1,nullptr,false,key));
                state={{"identities",treeBuild(db,leaves)},{"allocated",treeBuild(db,ledger)}};
            }
        }
        Tree identities=idsName.empty()?Tree():state.at("identities"), allocated=idsName.empty()?Tree():state.at("allocated");
        if (!idsName.empty()) {
            Statement latest(db,"SELECT manifest FROM geometry_revisions WHERE object_uuid=? ORDER BY object_generation DESC,rowid DESC LIMIT 1"); latest.bindBlob(1,parseUuid(uuid));
            if(latest.row()) {
                const auto text=latest.text(0);if(geometryWork)geometryWork->manifestBytesRead+=text.size();
                const auto tip=Json::parse(text);
                if(tip.contains("sequences") && tip.at("sequences").contains(edit.array) && tip.at("sequences").at(edit.array).contains("allocated"))
                    allocated=unionIdentityLedgers(db,allocated,tip.at("sequences").at(edit.array).at("allocated"));
            }
        }
        bool needsTransform=idsName.empty() ? (!state.at("annotation_tree").is_null() && state.at("annotation_tree").at("directional").get<bool>()) :
            (!identities.is_null() && identities.at("directional").get<bool>());
        auto getIds=[&](const Json& descriptor,int64_t start,int64_t count) {
            std::vector<int64_t> ids(count); copyScalars<int64_t>(readSharedArray(db,descriptor,idsName,start*8,count*8),ids); return ids;
        };
        std::vector<int64_t> changedIds;
        if (!idsName.empty()) {
            changedIds=getIds(manifest.at("arrays").at(idsName),edit.start,edit.eraseCount);
            for(auto id:changedIds)identities=identityPut(db,identities,id,nullptr);
        }
        Tree coords=byteSlice(db,a.at("tree"),edit.array,dtype,0,edit.start*stride,true);
        Tree ids=idsName.empty()?Tree():byteSlice(db,manifest.at("arrays").at(idsName).at("tree"),idsName,"<i8",0,edit.start*8,false);
        Tree attrs=idsName.empty()?attributeSlice(state.at("annotation_tree"),0,edit.start):Tree();
        int64_t count=edit.start;
        for(const auto& part:edit.parts) {
            Json values,pointAttrs; std::vector<int64_t> pointIds; Tree partCoords,partIds;
            if(part.revision) {
                if(!part.bytes.empty()||!part.ids.empty()||part.annotations!="[]") throw std::invalid_argument("reference parts cannot supply replacement point data");
                const auto sourceRevision=geometryRevision(txn,*part.revision);
                if(!sourceRevision||sourceRevision->kind!=current->kind) throw std::invalid_argument("sequence source kind mismatch");
                requireSource(db,sourceRevision->objectUuid); GeometryReader source(txn,*part.revision);
                const auto& descriptor=source.manifest().at("arrays").at(edit.array); const int64_t n=descriptor.at("shape").at(0);
                if(part.start<0||part.count<0||part.start>n||part.count>n-part.start) throw std::out_of_range("sequence source span out of range");
                if(descriptor.at("dtype")!=dtype) throw std::invalid_argument("sequence source layout mismatch");
                auto indexed=indexedDescriptor(db,source,edit.array,true);
                partCoords=byteSlice(db,indexed.at("tree"),edit.array,dtype,part.start*stride,part.count*stride,true);
                if(!idsName.empty()) {
                    if(source.manifest().at("arrays").contains(idsName)) {
                        const auto idBytes=source.read(idsName,part.start*8,part.count*8); pointIds.resize(part.count); copyScalars<int64_t>(idBytes,pointIds);
                    } else { pointIds.resize(part.count); std::iota(pointIds.begin(),pointIds.end(),part.start); }
                }
                pointAttrs=Json::array();
                if(source.manifest().contains("sequences") && source.manifest().at("sequences").contains(edit.array) &&
                   (source.manifest().at("sequences").at(edit.array).contains("identities") || source.manifest().at("sequences").at(edit.array).contains("annotation_tree"))) {
                    const auto& ss=source.manifest().at("sequences").at(edit.array);
                    if(idsName.empty()) treeVisit(db,ss.at("annotation_tree"),part.start,part.count,[&](const auto& v,int64_t start,int64_t size){for(auto i=start;i<start+size;++i)pointAttrs.push_back(unpackAnnotation(db,v.at(i)));});
                    else for(auto id:pointIds) pointAttrs.push_back(unpackAnnotation(db,identityGet(db,ss.at("identities"),id).at("annotation")));
                } else {
                    const auto all=annotations(*this,txn,*sourceRevision,source,edit.array);
                    for(auto i=part.start;i<part.start+part.count;++i)pointAttrs.push_back(all.at(i));
                }
            } else {
                if(part.start||part.count<0||uint64_t(part.count)>uint64_t(INT64_MAX/stride)||part.bytes.size()!=uint64_t(part.count*stride)) throw std::invalid_argument("new sequence part byte length mismatch");
                if(!idsName.empty() && part.ids.size()!=size_t(part.count)) throw std::invalid_argument("new points require stable IDs");
                if(idsName.empty() && !part.ids.empty()) throw std::invalid_argument("sample sequences do not accept stable IDs");
                pointIds=part.ids;
                for(auto id:pointIds) if(id<0||!identityGet(db,allocated,id).is_null()) throw std::invalid_argument("point ID already used; reference its revision to restore it");
                partCoords=byteTree(db,edit.array,dtype,part.bytes,true);
                pointAttrs=Json::parse(part.annotations);
                if(pointAttrs.is_array() && pointAttrs.empty())pointAttrs=Json(std::vector<Json>(part.count,Json::object()));
            }
            if(!pointAttrs.is_array()||pointAttrs.size()!=size_t(part.count))throw std::invalid_argument("new point annotations must match rows");
            for(const auto& value:pointAttrs)if(!value.is_object()&&!value.is_null())throw std::invalid_argument("invalid point annotation");
            needsTransform|=directional(pointAttrs);
            if(count>INT64_MAX/stride-part.count)throw std::overflow_error("sequence length overflow");
            if(part.reverse) {
                auto bytes=treeBytes(db,partCoords,edit.array,0,part.count*stride);
                for(int64_t i=0;i<part.count/2;++i)for(int64_t j=0;j<stride;++j)std::swap(bytes[i*stride+j],bytes[(part.count-i-1)*stride+j]);
                partCoords=byteTree(db,edit.array,dtype,bytes,true);
                std::reverse(pointIds.begin(),pointIds.end()); std::reverse(pointAttrs.begin(),pointAttrs.end());
            }
            coords=byteJoin(db,coords,partCoords,edit.array,dtype,true);
            if(idsName.empty()) attrs=treeJoin(db,attrs,attributeTree(pointAttrs));
            else {
                const auto bytes=treeBytes(db,partCoords,edit.array,0,part.count*stride);
                for(int64_t i=0;i<part.count;++i) {
                    const auto id=pointIds[i]; changedIds.push_back(id);
                    if(id<0||!identityGet(db,identities,id).is_null())throw std::invalid_argument("duplicate resulting point ID");
                    identities=identityPut(db,identities,id,{{"bytes",std::vector<uint8_t>(bytes.begin()+i*stride,bytes.begin()+(i+1)*stride)},{"annotation",packAnnotation(db,pointAttrs.at(i))}});
                    allocated=identityPut(db,allocated,id,true);
                }
                ids=byteJoin(db,ids,byteTree(db,idsName,"<i8",scalarBytes<int64_t>(pointIds),false),idsName,"<i8",false);
            }
            count+=part.count;
        }
        const auto tail=edit.start+edit.eraseCount,remaining=oldCount-tail;
        coords=byteJoin(db,coords,byteSlice(db,a.at("tree"),edit.array,dtype,tail*stride,remaining*stride,true),edit.array,dtype,true);
        if(idsName.empty()) attrs=treeJoin(db,attrs,attributeSlice(state.at("annotation_tree"),tail,remaining));
        else ids=byteJoin(db,ids,byteSlice(db,manifest.at("arrays").at(idsName).at("tree"),idsName,"<i8",tail*8,remaining*8,false),idsName,"<i8",false);
        if(count>INT64_MAX/stride-remaining)throw std::overflow_error("sequence length overflow");
        count+=remaining;
        if(needsTransform&&!edit.transformedAnnotations)throw std::invalid_argument("structural edit requires transformed span/direction annotations");
        if(edit.transformedAnnotations) {
            const auto transformed=Json::parse(*edit.transformedAnnotations);
            if(!transformed.is_array()||transformed.size()!=size_t(count))throw std::invalid_argument("transformed annotations must match resulting rows");
            for(const auto& value:transformed)if(!value.is_object()&&!value.is_null())throw std::invalid_argument("invalid transformed annotation");
            if(idsName.empty())attrs=attributeTree(transformed);
            else {
                std::vector<int64_t> values(count); copyScalars<int64_t>(treeBytes(db,ids,idsName,0,count*8),values);
                for(int64_t i=0;i<count;++i) { auto v=identityGet(db,identities,values[i]);v["annotation"]=packAnnotation(db,transformed.at(i));identities=identityPut(db,identities,values[i],v); }
            }
        }
        a["tree"]=coords;a["shape"][0]=count;a["byte_length"]=count*stride;
        if(idsName.empty())state={{"annotation_tree",attrs}};
        else {
            manifest["arrays"][idsName]={{"dtype","<i8"},{"shape",{count}},{"byte_length",count*8},{"tree",ids}};
            state={{"identities",identities},{"allocated",allocated}};
        }
        manifest["sequences"][edit.array]=state;
        auto source=Json::parse(objectByUuid(txn,uuid)->meta).value("source",Json::object());
        const auto key=current->kind=="point_collection"?"point_attributes":edit.array+"_attributes";
        if(source.erase(key))setMeta(txn,uuid,"source",source.dump());
        const auto result=publishSharedGeometry(*this,txn,uuid,std::move(manifest),expected,edges,idsName.empty()?std::nullopt:std::optional(std::pair{idsName,changedIds})); savepoint.release();return result;
    } catch(const std::exception& e) { historyConflict(db,e.what()); }
}

std::string ObjectStore::resizeSurface(WriteTxn& txn,const std::string& objectId,const GeometrySurfaceResize& resize,
    const std::string& expected,const std::vector<GeometryEdgeEdit>& edges)
{
    GeometryPhase phase("ObjectStore::resizeSurface");
    auto* db=static_cast<sqlite3*>(connection(txn)); checked(db); Savepoint savepoint(db); TreeCacheScope cache;
    try {
        const auto uuid=formatUuid(parseUuid(objectId)); const auto current=currentGeometry(txn,uuid);
        if (!current || current->uuid!=formatUuid(parseUuid(expected))) historyConflict(db,"geometry: current revision changed; reload before committing");
        if (current->kind!="patch") throw std::invalid_argument("surface resize requires a patch");
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
        for (auto& [name,a]:manifest.at("arrays").items()) {
            auto old=indexedDescriptor(db,reader,name,name=="points",true,origin);
            const auto shape=a.at("shape").get<std::vector<int64_t>>();
            auto newShape=shape; newShape[0]=resize.rows; newShape[1]=resize.columns;
            const auto length=geometryByteLength(a.at("dtype"),newShape), pixel=surfacePixelBytes(a.at("dtype"),shape);
            const auto found=fills.find(name);
            if(found==fills.end())throw std::invalid_argument("missing surface fill");
            auto fillShape=shape; fillShape[0]=fillShape[1]=1; const auto& fill=found->second;
            if(fill.dtype!=a.at("dtype").get<std::string>()||fill.shape!=fillShape||int64_t(fill.bytes.size())!=pixel)throw std::invalid_argument("surface fill must be one pixel of matching dtype/components");
            const int64_t tile=old.at("tile_size");
            // Visit only changed tile cells and the four old/new boundaries.
            // Interior keys and subtree bounds remain independent of grid origin.
            std::set<std::pair<int64_t,int64_t>> dirty;
            auto cells=[&](int64_t y,int64_t x,int64_t h,int64_t w) {
                if(!h||!w)return std::array<int64_t,4>{0,0,0,0};
                return std::array<int64_t,4>{floorTile(y,tile),floorTile(x,tile),floorTile(y+h-1,tile)+tile,floorTile(x+w-1,tile)+tile};
            };
            const auto before=cells(origin[0],origin[1],shape[0],shape[1]), after=cells(oy,ox,resize.rows,resize.columns);
            auto rectangle=[&](int64_t top,int64_t left,int64_t bottom,int64_t right) {
                for(auto y=top;y<bottom;y+=tile)for(auto x=left;x<right;x+=tile)dirty.emplace(y,x);
            };
            auto difference=[&](const auto& box,const auto& other) {
                const auto top=std::max(box[0],other[0]),left=std::max(box[1],other[1]),bottom=std::min(box[2],other[2]),right=std::min(box[3],other[3]);
                if(top>=bottom||left>=right)rectangle(box[0],box[1],box[2],box[3]);
                else {
                    rectangle(box[0],box[1],top,box[3]);rectangle(bottom,box[1],box[2],box[3]);
                    rectangle(top,box[1],bottom,left);rectangle(top,right,bottom,box[3]);
                }
            };
            difference(before,after);difference(after,before);
            auto boundary=[&](const auto& box,int side) {
                if(box[0]>=box[2]||box[1]>=box[3])return;
                if(side==0)rectangle(box[0],box[1],box[0]+tile,box[3]);
                if(side==1)rectangle(box[0],box[1],box[2],box[1]+tile);
                if(side==2)rectangle(box[2]-tile,box[1],box[2],box[3]);
                if(side==3)rectangle(box[0],box[3]-tile,box[2],box[3]);
            };
            const std::array<int64_t,4> oldEdges={origin[0],origin[1],origin[0]+shape[0],origin[1]+shape[1]},newEdges={oy,ox,oy+resize.rows,ox+resize.columns};
            for(int side=0;side<4;++side)if(oldEdges[side]!=newEdges[side]){boundary(before,side);boundary(after,side);}
            auto root=old.at("tree"); std::set<std::pair<int64_t,int64_t>> changedTiles;
            for(const auto& [y,x]:dirty) {
                const auto key=tileKey(y,x);
                const auto r=std::max(y,oy),c=std::max(x,ox),h=std::min(y+tile,oy+resize.rows)-r,w=std::min(x+tile,ox+resize.columns)-c;
                if(h<=0||w<=0) {if(!treeGet(db,root,key).is_null())changedTiles.emplace(y,x);root=treePut(db,root,key,nullptr);continue;}
                auto existing=treeGet(db,root,key);
                if(!existing.is_null()&&existing.at("row")==r&&existing.at("column")==c&&existing.at("rows")==h&&existing.at("columns")==w)continue;
                std::vector<uint8_t> bytes(h*w*pixel);
                for(int64_t i=0;i<h*w;++i)std::memcpy(bytes.data()+i*pixel,fill.bytes.data(),pixel);
                const auto top=std::max(r,origin[0]),left=std::max(c,origin[1]),bottom=std::min(r+h,origin[0]+shape[0]),right=std::min(c+w,origin[1]+shape[1]);
                if(top<bottom&&left<right) {
                    const auto previous=readSharedRectangle(db,old,name,top-origin[0],left-origin[1],bottom-top,right-left);
                    for(auto yy=top;yy<bottom;++yy)std::memcpy(bytes.data()+((yy-r)*w+left-c)*pixel,previous.data()+(yy-top)*(right-left)*pixel,(right-left)*pixel);
                }
                changedTiles.emplace(y,x);
                const Json value={{"bounds",name=="points"?coordinateBounds(bytes,a.at("dtype"),true):Json()},{"hash",storeSharedPayload(db,name,bytes)},{"row",r},{"column",c},{"rows",h},{"columns",w}};
                root=treePut(db,root,key,treeLeaf(db,value,1,name=="points"?coordinateBounds(bytes,a.at("dtype"),true):Json(),false,key));
            }
            if(name=="points")root=tileHalos(db,root,tile,changedTiles);
            a=old;a["shape"]=newShape;a["byte_length"]=length;a["tree"]=root;a["origin"]=std::array<int64_t,2>{oy,ox};
        }
        manifest["grid_origin"]={oy,ox};
        const auto result=publishSharedGeometry(*this,txn,uuid,std::move(manifest),expected,edges);
        savepoint.release(); return result;
    } catch (const std::exception& error) { historyConflict(db,error.what()); }
}
}
