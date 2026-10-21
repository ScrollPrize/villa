#include "geometry/PersistentTree.hpp"
#include "geometry/ArrayBytes.hpp"
#include <bit>
#include <iomanip>
#include <sstream>

namespace vc::objects::detail {
namespace {
struct CachedNode { Tree node; size_t bytes; };
using NodeCache=std::map<std::pair<sqlite3*,std::string>,CachedNode>;
thread_local std::unique_ptr<NodeCache> nodeCache;
thread_local size_t nodeCacheBytes=0;
void cacheNode(sqlite3* db,const std::string& hash,const Tree& node) {
    if(!nodeCache)return;
    const auto bytes=node.dump().size();
    const auto charge=bytes*16+256;
    if(charge>8*1024*1024)return;
    if(nodeCacheBytes+charge>8*1024*1024) {nodeCache->clear();nodeCacheBytes=0;}
    nodeCacheBytes+=charge;
    nodeCache->emplace(std::pair{db,hash},CachedNode{node,bytes});
}
int height(const Tree& t) { return t.is_null() ? 0 : t.at("height").get<int>(); }
Tree save(sqlite3* db,Tree node,Tree ref) {
    node["summary"]=ref;
    const auto text=node.dump();
    ref["hash"]=storeSharedPayload(db,"@tree",std::span(reinterpret_cast<const uint8_t*>(text.data()),text.size()));
    cacheNode(db,ref.at("hash"),node);
    if (geometryWork) ++geometryWork->manifestEntries;
    return ref;
}
Tree branch(sqlite3* db,const Tree& a,const Tree& b) {
    if (a.is_null()) return b;
    if (b.is_null()) return a;
    if (treeSize(a)>INT64_MAX-treeSize(b)) throw std::overflow_error("tree size overflow");
    Tree ref={{"size",treeSize(a)+treeSize(b)},{"height",1+std::max(height(a),height(b))},
              {"bounds",mergeTreeBounds(a.at("bounds"),b.at("bounds"))},
              {"directional",a.at("directional").get<bool>() || b.at("directional").get<bool>()}};
    if(a.contains("first_bounds")) {ref["first_bounds"]=a.at("first_bounds");ref["last_bounds"]=b.at("last_bounds");}
    if (a.contains("first")) { ref["first"]=a.at("first"); ref["last"]=b.at("last"); }
    return save(db,{{"left",a},{"right",b}},ref);
}
Tree balance(sqlite3* db,const Tree& a,const Tree& b) {
    if (height(a)>height(b)+1) {
        const auto n=treeNode(db,a); const auto& l=n.at("left"); const auto& r=n.at("right");
        if (height(l)>=height(r)) return branch(db,l,branch(db,r,b));
        const auto m=treeNode(db,r); return branch(db,branch(db,l,m.at("left")),branch(db,m.at("right"),b));
    }
    if (height(b)>height(a)+1) {
        const auto n=treeNode(db,b); const auto& l=n.at("left"); const auto& r=n.at("right");
        if (height(r)>=height(l)) return branch(db,branch(db,a,l),r);
        const auto m=treeNode(db,l); return branch(db,branch(db,a,m.at("left")),branch(db,m.at("right"),r));
    }
    return branch(db,a,b);
}
}
TreeCacheScope::TreeCacheScope():owner_(!nodeCache) {if(owner_){nodeCache=std::make_unique<NodeCache>();nodeCacheBytes=0;}}
TreeCacheScope::~TreeCacheScope(){if(owner_)nodeCache.reset();}
int64_t treeSize(const Tree& t) { return t.is_null()?0:t.at("size").get<int64_t>(); }
Tree mergeTreeBounds(const Tree& a,const Tree& b) {
    if (a.is_null()) return b;
    if (b.is_null()) return a;
    auto out=a;
    for (int i=0;i<3;++i) { out[i]=std::min(a[i].get<double>(),b[i].get<double>()); out[i+3]=std::max(a[i+3].get<double>(),b[i+3].get<double>()); }
    return out;
}
Tree treeNode(sqlite3* db,const Tree& ref) {
    requestWork(&FiberRegionStats::treeVisits);
    GeometryPhase phase("treeNode");
    Tree node;
    if(nodeCache) {const auto found=nodeCache->find({db,ref.at("hash").get<std::string>()});if(found!=nodeCache->end()) { requestBytes(found->second.bytes, &FiberRegionStats::cacheBytes); node=found->second.node; }}
    if(node.is_null()) {const auto bytes=readSharedPayload(db,"@tree",ref.at("hash"));node=Tree::parse(bytes);cacheNode(db,ref.at("hash"),node);}
    auto expected=ref; expected.erase("hash");
    if (node.at("summary")!=expected || height(ref)<1 || height(ref)>96 || treeSize(ref)<=0)
        throw std::runtime_error("geometry: invalid tree summary");
    return node;
}
Tree treeLeaf(sqlite3* db,Tree value,int64_t size,const Tree& bounds,bool directional,const std::string& key) {
    if (!size) return nullptr;
    Tree ref={{"size",size},{"height",1},{"bounds",bounds},{"directional",directional}};
    if(value.is_object()&&value.contains("first_bounds")) {ref["first_bounds"]=value.at("first_bounds");ref["last_bounds"]=value.at("last_bounds");}
    if (!key.empty()) ref["first"]=ref["last"]=key;
    return save(db,{{"value",std::move(value)}},ref);
}
Tree treeJoin(sqlite3* db,const Tree& a,const Tree& b) {
    if (a.is_null()) return b;
    if (b.is_null()) return a;
    if (height(a)>height(b)+1) { const auto n=treeNode(db,a); return balance(db,n.at("left"),treeJoin(db,n.at("right"),b)); }
    if (height(b)>height(a)+1) { const auto n=treeNode(db,b); return balance(db,treeJoin(db,a,n.at("left")),n.at("right")); }
    return branch(db,a,b);
}
Tree treeBuild(sqlite3* db,const std::vector<Tree>& leaves) {
    std::function<Tree(size_t,size_t)> build=[&](size_t lo,size_t hi)->Tree {
        if (lo==hi) return nullptr;
        if (hi-lo==1) return leaves[lo];
        const auto mid=lo+(hi-lo)/2; return branch(db,build(lo,mid),build(mid,hi));
    };
    return build(0,leaves.size());
}
Tree treeSlice(sqlite3* db,const Tree& t,int64_t start,int64_t count,const LeafSlice& slice) {
    if (start<0 || count<0 || start>treeSize(t) || count>treeSize(t)-start) throw std::out_of_range("tree slice out of range");
    if (!count) return nullptr;
    if (!start && count==treeSize(t)) return t;
    const auto n=treeNode(db,t);
    if (n.contains("value")) return slice(n.at("value"),start,count);
    const auto size=treeSize(n.at("left"));
    if (start>=size) return treeSlice(db,n.at("right"),start-size,count,slice);
    if (count<=size-start) return treeSlice(db,n.at("left"),start,count,slice);
    return treeJoin(db,treeSlice(db,n.at("left"),start,size-start,slice),treeSlice(db,n.at("right"),0,count-size+start,slice));
}
void treeVisit(sqlite3* db,const Tree& t,int64_t start,int64_t count,const std::function<void(const Tree&,int64_t,int64_t)>& visit) {
    if (start<0 || count<0 || start>treeSize(t) || count>treeSize(t)-start) throw std::out_of_range("tree visit out of range");
    if (!count) return;
    const auto n=treeNode(db,t);
    if (n.contains("value")) { visit(n.at("value"),start,count); return; }
    const auto size=treeSize(n.at("left"));
    if (start<size) treeVisit(db,n.at("left"),start,std::min(count,size-start),visit);
    if (start+count>size) treeVisit(db,n.at("right"),std::max<int64_t>(0,start-size),start+count-std::max(start,size),visit);
}
void treeEntries(sqlite3* db,const Tree& t,const std::function<void(const Tree&,const Tree&)>& f) {
    if(t.is_null())return;
    const auto node=treeNode(db,t);
    if(node.contains("value"))f(t,node.at("value"));
    else {treeEntries(db,node.at("left"),f);treeEntries(db,node.at("right"),f);}
}
void treeWalk(sqlite3* db,const Tree& t,const std::function<void(const Tree&)>& f) {treeEntries(db,t,[&](const Tree&,const Tree& v){f(v);});}
Tree treeLookup(sqlite3* db,const Tree& t,const std::string& key) {
    if(t.is_null()||key<t.at("first").get<std::string>()||key>t.at("last").get<std::string>())return nullptr;
    const auto n=treeNode(db,t);if(n.contains("value"))return t;
    return treeLookup(db,key<=n.at("left").at("last").get<std::string>()?n.at("left"):n.at("right"),key);
}
Tree treeGet(sqlite3* db,const Tree& t,const std::string& key) {
    const auto ref=treeLookup(db,t,key);return ref.is_null()?Tree():treeNode(db,ref).at("value");
}
Tree treePut(sqlite3* db,const Tree& t,const std::string& key,const Tree& leaf) {
    if (t.is_null()) return leaf;
    if (key<t.at("first").get<std::string>()) return treeJoin(db,leaf,t);
    if (key>t.at("last").get<std::string>()) return treeJoin(db,t,leaf);
    const auto n=treeNode(db,t);
    if (n.contains("value")) return leaf;
    if (key<=n.at("left").at("last").get<std::string>()) return treeJoin(db,treePut(db,n.at("left"),key,leaf),n.at("right"));
    return treeJoin(db,n.at("left"),treePut(db,n.at("right"),key,leaf));
}
Tree coordinateBounds(std::span<const uint8_t> bytes,const std::string& dtype,bool patch) {
    GeometryPhase phase("coordinateBounds");
    std::optional<std::array<double,6>> result;
    auto scan=[&]<class T>() {
        if (bytes.size()%(3*sizeof(T))) throw std::runtime_error("unaligned coordinate tree leaf");
        for (size_t offset=0;offset<bytes.size();offset+=3*sizeof(T)) {
            std::array<T,3> point; copyScalars<T>(bytes.subspan(offset,3*sizeof(T)),point);
            const double x=point[0],y=point[1],z=point[2];
            if (!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||(patch&&x==-1&&z<=0)) continue;
            if(!result)result=std::array<double,6>{x,y,z,x,y,z};
            else for(size_t axis=0;axis<3;++axis) {(*result)[axis]=std::min((*result)[axis],double(point[axis]));(*result)[axis+3]=std::max((*result)[axis+3],double(point[axis]));}
        }
    };
    if (dtype=="<f8") scan.template operator()<double>();
    else if (dtype=="<f4") scan.template operator()<float>();
    else if (dtype=="<i4") scan.template operator()<int32_t>();
    else throw std::runtime_error("unsupported coordinate tree dtype");
    return result?Tree(*result):Tree();
}
Tree byteTree(sqlite3* db,const std::string& name,const std::string& dtype,std::span<const uint8_t> bytes,bool coordinates) {
    const auto stride=geometryByteLength(dtype,{coordinates?3:1}); const size_t block=(65536/stride)*stride;
    std::vector<Tree> leaves;
    for (size_t i=0;i<bytes.size();i+=block) {
        const auto part=bytes.subspan(i,std::min(block,bytes.size()-i));
        Tree value={{"hash",storeSharedPayload(db,name,part)},{"offset",0},{"length",part.size()}};
        if(coordinates) {value["first_bounds"]=coordinateBounds(part.first(stride),dtype,false);value["last_bounds"]=coordinateBounds(part.last(stride),dtype,false);}
        leaves.push_back(treeLeaf(db,value,part.size(),coordinates?coordinateBounds(part,dtype,false):Tree()));
    }
    return treeBuild(db,leaves);
}
std::vector<uint8_t> treeBytes(sqlite3* db,const Tree& root,const std::string& name,int64_t start,int64_t count) {
    std::vector<uint8_t> out; out.reserve(count);
    treeVisit(db,root,start,count,[&](const Tree& v,int64_t offset,int64_t length) {
        const auto bytes=readSharedPayload(db,name,v.at("hash")); const int64_t begin=v.at("offset").get<int64_t>()+offset;
        if (begin<0 || begin>int64_t(bytes.size()) || length>int64_t(bytes.size())-begin) throw std::runtime_error("invalid tree block");
        out.insert(out.end(),bytes.begin()+begin,bytes.begin()+begin+length);
    }); return out;
}
Tree byteSlice(sqlite3* db,const Tree& root,const std::string& name,const std::string& dtype,int64_t start,int64_t count,bool coordinates) {
    GeometryPhase phase("byteSlice");
    if(start<0||count<0||start>treeSize(root)||count>treeSize(root)-start)throw std::out_of_range("byte slice out of range");
    if(!count)return nullptr;
    if(!start&&count==treeSize(root))return root;
    const auto node=treeNode(db,root);
    if(node.contains("value")) {
        auto value=node.at("value");value["offset"]=value.at("offset").get<int64_t>()+start;value["length"]=count;
        Tree bounds;
        if(coordinates) {
            const auto bytes=readSharedPayload(db,name,value.at("hash"));const auto part=std::span(bytes).subspan(value.at("offset").get<int64_t>(),count);
            const auto stride=geometryByteLength(dtype,{3});bounds=coordinateBounds(part,dtype,false);
            value["first_bounds"]=coordinateBounds(part.first(stride),dtype,false);value["last_bounds"]=coordinateBounds(part.last(stride),dtype,false);
        }
        return treeLeaf(db,value,count,bounds);
    }
    const auto size=treeSize(node.at("left"));
    if(start>=size)return byteSlice(db,node.at("right"),name,dtype,start-size,count,coordinates);
    if(count<=size-start)return byteSlice(db,node.at("left"),name,dtype,start,count,coordinates);
    return byteJoin(db,byteSlice(db,node.at("left"),name,dtype,start,size-start,coordinates),
        byteSlice(db,node.at("right"),name,dtype,0,count-size+start,coordinates),name,dtype,coordinates);
}
Tree byteJoin(sqlite3* db,const Tree& a,const Tree& b,const std::string& name,const std::string& dtype,bool coordinates) {
    GeometryPhase phase("byteJoin");
    if (a.is_null() || b.is_null()) return treeJoin(db,a,b);
    auto edge=[&](Tree t,bool right) { while (height(t)>1) { const auto n=treeNode(db,t); t=n.at(right?"right":"left"); } return t; };
    const auto l=edge(a,true),r=edge(b,false); const auto ls=treeSize(l),rs=treeSize(r);
    // Local compaction: every adjacent pair has at least 4 KiB, except at
    // structural boundaries. Never compact an entire object or history chain.
    if (ls+rs<=4096) {
        auto bytes=treeBytes(db,l,name,0,ls); const auto tail=treeBytes(db,r,name,0,rs); bytes.insert(bytes.end(),tail.begin(),tail.end());
        if (geometryWork) ++geometryWork->compactions;
        return byteJoin(db,byteJoin(db,byteSlice(db,a,name,dtype,0,treeSize(a)-ls,coordinates),byteTree(db,name,dtype,bytes,coordinates),name,dtype,coordinates),
            byteSlice(db,b,name,dtype,rs,treeSize(b)-rs,coordinates),name,dtype,coordinates);
    }
    return treeJoin(db,a,b);
}
std::string tileKey(int64_t row,int64_t column) {
    std::ostringstream s; s<<std::hex<<std::setfill('0')<<std::setw(16)<<(uint64_t(row)^(uint64_t(1)<<63))<<std::setw(16)<<(uint64_t(column)^(uint64_t(1)<<63)); return s.str();
}
std::string identityKey(int64_t id) { return tileKey(0,id/128); }
bool annotationDirectional(const Tree& v) {
    if(v.is_object())for(const auto& [k,x]:v.items())if(k=="segment_to_next"||k.find("direction")!=std::string::npos||k.find("tangent")!=std::string::npos||annotationDirectional(x))return true;
    if(v.is_array())for(const auto& x:v)if(annotationDirectional(x))return true;
    return false;
}
Tree packAnnotation(sqlite3* db,const Tree& value) {
    const auto text=value.dump();
    Tree result={{"span_fields",annotationDirectional(value)}};
    if(text.size()<=1024)result["inline"]=value;
    else result["annotation_blob"]=byteTree(db,"@annotations","|u1",std::span(reinterpret_cast<const uint8_t*>(text.data()),text.size()),false);
    return result;
}
Tree unpackAnnotation(sqlite3* db,const Tree& value) {
    if(value.contains("inline"))return value.at("inline");
    const auto& root=value.at("annotation_blob");return Tree::parse(treeBytes(db,root,"@annotations",0,treeSize(root)));
}
Tree identityGet(sqlite3* db,const Tree& root,int64_t id) {
    GeometryPhase phase("identityGet");
    if (geometryWork) ++geometryWork->identityRows;
    const auto bucket=treeGet(db,root,identityKey(id));
    return bucket.is_null()?Tree():bucket.value(std::to_string(id%128),Tree());
}
Tree unionIdentityLedgers(sqlite3* db,const Tree& a,const Tree& b) {
    if(a.is_null())return b;
    if(b.is_null()||a==b)return a;
    if(a.at("last").get<std::string>()<b.at("first").get<std::string>())return treeJoin(db,a,b);
    if(b.at("last").get<std::string>()<a.at("first").get<std::string>())return treeJoin(db,b,a);
    const auto node=treeNode(db,a);
    if(node.contains("value")) {
        const std::string key=a.at("first");auto value=treeGet(db,b,key);
        if(value.is_null())return treePut(db,b,key,a);
        value.update(node.at("value"));return treePut(db,b,key,treeLeaf(db,value,1,nullptr,false,key));
    }
    std::function<std::pair<Tree,Tree>(const Tree&,const std::string&)> split=[&](const Tree& t,const std::string& key)->std::pair<Tree,Tree> {
        if(t.is_null())return {nullptr,nullptr};
        if(t.at("last").get<std::string>()<=key)return {t,nullptr};
        if(t.at("first").get<std::string>()>key)return {nullptr,t};
        const auto n=treeNode(db,t);const auto l=split(n.at("left"),key),r=split(n.at("right"),key);
        return {treeJoin(db,l.first,r.first),treeJoin(db,l.second,r.second)};
    };
    const auto parts=split(b,node.at("left").at("last"));
    return treeJoin(db,unionIdentityLedgers(db,node.at("left"),parts.first),unionIdentityLedgers(db,node.at("right"),parts.second));
}
Tree identityPut(sqlite3* db,const Tree& root,int64_t id,const Tree& value) {
    GeometryPhase phase("identityPut");
    const auto key=identityKey(id); auto bucket=treeGet(db,root,key); if (bucket.is_null()) bucket=Tree::object();
    if (value.is_null()) bucket.erase(std::to_string(id%128)); else bucket[std::to_string(id%128)]=value;
    bool directional=false;
    for(const auto& entry:bucket)if(entry.is_object()&&entry.contains("annotation"))directional |= entry.at("annotation").at("span_fields").get<bool>();
    return treePut(db,root,key,bucket.empty()?Tree():treeLeaf(db,bucket,1,nullptr,directional,key));
}
namespace {
template<class Read, class Rectangle>
Tree makeIndexedDescriptor(sqlite3* db, const Tree& a, const std::string& name, bool coordinates,
    bool patch, const std::array<int64_t,2>& origin, Read read, Rectangle rectangle) {
    if (a.contains("tree")) return a;
    auto out=a; out.erase("chunks"); out.erase("storage"); out.erase("source_revision");
    if (patch) {
        const int64_t tile=a.contains("storage") ? a.at("storage").at("tile_size").get<int64_t>() : a.value("tile_size",int64_t(128)); out["tile_size"]=tile; out["origin"]=origin;
        const auto shape=a.at("shape").get<std::vector<int64_t>>(); std::vector<Tree> leaves; std::map<std::pair<int64_t,int64_t>,Tree> tiles;
        // Initial conversion is intentionally proportional to source size.
        auto floor=[&](int64_t x){auto q=x/tile;if(x%tile<0)--q;return q*tile;};
        for (auto y=floor(origin[0]);shape[0]&&shape[1]&&y<origin[0]+shape[0];y+=tile)
            for(auto x=floor(origin[1]);x<origin[1]+shape[1];x+=tile) {
            const auto r=std::max(y,origin[0])-origin[0],c=std::max(x,origin[1])-origin[1];
            const auto h=std::min(y+tile,origin[0]+shape[0])-origin[0]-r,w=std::min(x+tile,origin[1]+shape[1])-origin[1]-c;
            const auto bytes=rectangle(r,c,h,w);
            const auto key=tileKey(y,x);
            tiles[{y,x}]={{"hash",storeSharedPayload(db,name,bytes)},{"row",origin[0]+r},{"column",origin[1]+c},{"rows",h},{"columns",w},{"bounds",coordinates?coordinateBounds(bytes,a.at("dtype"),true):Tree()}};
        }
        for(const auto& [cell,v]:tiles) {
            Tree bounds;
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                const auto found=tiles.find({cell.first+dy*tile,cell.second+dx*tile});
                if(found!=tiles.end())bounds=mergeTreeBounds(bounds,found->second.at("bounds"));
            }
            leaves.push_back(treeLeaf(db,v,1,bounds,false,tileKey(cell.first,cell.second)));
        }
        out["tree"]=treeBuild(db,leaves);
    } else {
        // Rechunk once into point-aligned bounded leaves; subsequent edits
        // reuse whole subtrees and only inspect split boundary leaves.
        const auto bytes=read();
        out["tree"]=byteTree(db,name,a.at("dtype"),bytes,coordinates);
    }
    return out;
}
} // namespace
Tree indexedDescriptor(sqlite3* db, GeometryReader& reader, const std::string& name, bool coordinates,
    bool patch, const std::array<int64_t,2>& origin) {
    const auto& a=reader.manifest().at("arrays").at(name);
    return makeIndexedDescriptor(db, a, name, coordinates, patch, origin,
        [&] { return reader.read(name,0); },
        [&](auto r,auto c,auto h,auto w) {
            const auto layout=geometryLayout(a);
            // Reuse stored tiles directly; delta-only chunks need reconstruction.
            if (layout.tile && r%layout.tile==0 && c%layout.tile==0) {
                const auto index=r/layout.tile*layout.tileColumns()+c/layout.tile;
                if (layout.rectangle(index)==std::array<int64_t,4>{r,c,h,w}) {
                    const std::string hash=a.at("chunks").at(index);
                    Statement exists(db,"SELECT 1 FROM geometry_chunks WHERE hash=?"); exists.bind(1,hash);
                    if (exists.row()) {
                        auto bytes=readSharedPayload(db,name,hash);
                        if (int64_t(bytes.size())!=layout.chunkBytes(index)) throw std::runtime_error("geometry: corrupt tile size");
                        return bytes;
                    }
                }
            }
            return reader.rectangle(name,r,c,h,w);
        });
}
Tree indexedSurfaceArray(sqlite3* db, const GeometryArray& a, int64_t tile, const std::array<int64_t,2>& origin) {
    const Tree descriptor={{"dtype",a.dtype},{"shape",a.shape},{"byte_length",a.bytes.size()},{"tile_size",tile}};
    const auto pixel=surfacePixelBytes(a.dtype,a.shape);
    return makeIndexedDescriptor(db,descriptor,a.name,false,true,origin,
        [&] { return std::span<const uint8_t>(a.bytes); },
        [&](int64_t r,int64_t c,int64_t h,int64_t w) {
            std::vector<uint8_t> bytes(h*w*pixel);
            for (int64_t y=0;y<h;++y)
                std::memcpy(bytes.data()+y*w*pixel,a.bytes.data()+((r+y)*a.shape[1]+c)*pixel,w*pixel);
            return bytes;
        });
}
Tree tileHalos(sqlite3* db,const Tree& original,int64_t tile,const std::set<std::pair<int64_t,int64_t>>& changed) {
    auto root=original;std::set<std::pair<int64_t,int64_t>> cells;
    for(const auto& [r,c]:changed)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)cells.emplace(r+dy*tile,c+dx*tile);
    for(const auto& [r,c]:cells) {
        const auto key=tileKey(r,c); const auto ref=treeLookup(db,original,key);if(ref.is_null())continue;const auto v=treeNode(db,ref).at("value");
        Tree bounds;
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
            const auto neighbour=treeGet(db,original,tileKey(r+dy*tile,c+dx*tile));
            if(!neighbour.is_null())bounds=mergeTreeBounds(bounds,neighbour.at("bounds"));
        }
        if(bounds!=ref.at("bounds"))root=treePut(db,root,key,treeLeaf(db,v,1,bounds,false,key));
    }
    return root;
}
void transferTree(sqlite3* source,sqlite3* destination,const Tree& root) {
    if (root.is_null()) return;
    Statement old(destination,"SELECT data FROM geometry_chunks WHERE hash=?");old.bind(1,root.at("hash").get<std::string>());
    if(old.row()) {
        const auto bytes=readSharedPayload(source,"@tree",root.at("hash"));
        if(sqlite3_column_bytes(old.value,0)!=int64_t(bytes.size())||std::memcmp(sqlite3_column_blob(old.value,0),bytes.data(),bytes.size()))
            throw std::runtime_error("sync: corrupt or colliding tree node");
        return;
    }
    const auto node=treeNode(source,root);
    if (node.contains("left")) { transferTree(source,destination,node.at("left")); transferTree(source,destination,node.at("right")); }
    else if (node.at("value").contains("hash")) { const auto bytes=readSharedPayload(source,"@transfer",node.at("value").at("hash")); storeSharedPayload(destination,"@transfer",bytes); }
    if(node.contains("value")) {
        std::function<void(const Tree&)> annotations=[&](const Tree& v) {
            if(v.is_object()&&v.contains("span_fields")) {if(v.contains("annotation_blob"))transferTree(source,destination,v.at("annotation_blob"));return;}
            if(v.is_structured())for(const auto& x:v)annotations(x);
        };
        annotations(node.at("value"));
    }
    const auto bytes=readSharedPayload(source,"@tree",root.at("hash")); storeSharedPayload(destination,"@tree",bytes);
}
}
