#pragma once
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include <functional>
#include "geometry/GeometryReader.hpp"
#include "geometry/ArrayLayout.hpp"
#include "geometry/SequenceMetadata.hpp"
#include "imports/SourceIdentityIndex.hpp"

namespace vc::objects::detail {
// A bounded reader for either a published revision or a proposed payload.
struct DependencyGeometry {
    std::string kind, attributes;
    std::array<int64_t,2> gridOrigin{0,0};
    std::map<std::string, std::string> fingerprints;
    std::map<std::string, GeometryArrayInfo> arrays;
    std::function<std::vector<uint8_t>(const std::string&, int64_t, int64_t)> read;
    std::function<std::optional<std::vector<uint8_t>>(const std::string&, int64_t)> identity;
    std::set<std::string> identityArrays;
    struct PointIndex {
        bool loaded = false;
        std::vector<std::pair<int64_t, int64_t>> offsets;
    };
    std::shared_ptr<PointIndex> pointIndex = std::make_shared<PointIndex>();
};

// Identity metadata is fixed between structural checkpoints. Coordinates are
// always read from the selected/proposed revision, including sparse deltas.
inline void bindSequenceIdentities(DependencyGeometry& geometry,sqlite3* db,const nlohmann::json& manifest) {
    if(geometry.kind == "surface") return;
    const auto states=manifest.at("sequences");
    for(const auto& [name,state]:states.items())if(state.contains("identities"))
        geometry.identityArrays.insert(name=="points"?"point_ids":"control_point_ids");
    geometry.identity=[db,states,read=geometry.read,arrays=geometry.arrays](const std::string& name,int64_t id)->std::optional<std::vector<uint8_t>> {
        const auto array=name=="point_ids"?"points":"control_points";
        if(!states.contains(array)||!states.at(array).contains("identities"))return std::nullopt;
        const auto point=identityGet(db,states.at(array).at("identities"),id);
        if(point.is_null())return std::nullopt;
        const int64_t row=point.at("row");
        if(row<0||row>=arrays.at(array).shape[0])throw std::runtime_error("invalid identity row");
        const auto stride=geometryByteLength(arrays.at(array).dtype,{3});
        return read(array,row*stride,stride);
    };
}

// Nested helpers in a publication/repair share revision readers. The cache is
// confined to this operation, not the connection or a future transaction.
using DependencyCache = std::map<std::pair<const Txn*, std::string>, DependencyGeometry>;
inline thread_local DependencyCache* activeDependencies = nullptr;
class DependencyScope {
    MetadataCacheScope metadata_;
    std::unique_ptr<DependencyCache> owned_;
public:
    DependencyScope() {
        if (!activeDependencies) {
            owned_ = std::make_unique<DependencyCache>(); activeDependencies = owned_.get();
        }
    }
    ~DependencyScope() { if (owned_) activeDependencies = nullptr; }
};

inline DependencyGeometry dependencyGeometry(sqlite3* db,const std::string& revision)
{
    auto reader=std::make_shared<GeometryReader>(db,revision);
    const auto& manifest=reader->manifest();DependencyGeometry result;
    result.kind=manifest.at("kind");result.attributes=manifest.at("attributes").dump();
    result.gridOrigin=manifest.value("grid_origin",std::array<int64_t,2>{0,0});
    for(const auto& [name,a]:manifest.at("arrays").items()){
        result.arrays.emplace(name,GeometryArrayInfo{name,a.at("dtype"),a.at("shape"),a.at("byte_length")});
        auto descriptor=a;descriptor.erase("source_revision");result.fingerprints[name]=descriptor.dump();
    }
    result.read=[reader](const auto& name,auto offset,auto count){return reader->read(name,offset,count);};
    bindSequenceIdentities(result,db,manifest);return result;
}
inline DependencyGeometry dependencyGeometry(const ObjectStore& db, const Txn& txn, const GeometryRevision& revision)
{
    const auto key = std::make_pair(&txn, revision.uuid);
    if (activeDependencies) {
        const auto cached = activeDependencies->find(key);
        if (cached != activeDependencies->end()) return cached->second;
    }
    auto result=dependencyGeometry(ObjectStoreImplAccess::conn(txn),revision.uuid);
    if (activeDependencies) {
        if (activeDependencies->size() == 8) activeDependencies->erase(activeDependencies->begin());
        activeDependencies->emplace(key, result);
    }
    return result;
}

// Over a reader (any backend): arrays read through it; point identities are
// resolved from the ids arrays.
inline DependencyGeometry dependencyGeometry(const StoreReader& reader, const GeometryRevision& revision)
{
    DependencyGeometry result; result.kind = revision.kind; result.attributes = revision.attributes; result.gridOrigin = revision.gridOrigin;
    for (const auto& array : revision.arrays) result.arrays.emplace(array.name, array);
    result.read = [&reader, uuid = revision.uuid](const std::string& name, int64_t offset, int64_t count) {
        return reader.readGeometryArray(uuid, name, offset, count);
    };
    return result;
}

inline DependencyGeometry dependencyGeometry(const GeometryData& data)
{
    DependencyGeometry result; result.kind = data.kind; result.attributes = data.attributes;
    for (const auto& array : data.arrays)
        result.arrays.emplace(array.name, GeometryArrayInfo{array.name, array.dtype, array.shape, static_cast<int64_t>(array.bytes.size())});
    result.read = [&data](const auto& name, int64_t offset, int64_t count) {
        const auto& array = *std::find_if(data.arrays.begin(), data.arrays.end(), [&](const auto& a) { return a.name == name; });
        if (offset < 0 || count < 0 || offset > static_cast<int64_t>(array.bytes.size()) - count)
            throw std::runtime_error("geometry dependency range out of bounds");
        return std::vector<uint8_t>(array.bytes.begin() + offset, array.bytes.begin() + offset + count);
    };
    return result;
}

// Missing/deleted points return nullopt. Malformed references never resolve.
inline std::optional<std::string> dependencyCoordinateArray(const std::string& kind,const std::string& name)
{
    if(name=="point_ids") return kind=="point_collection"?std::optional<std::string>("points"):std::nullopt;
    if(name=="control_point_ids") return kind=="fiber"?std::optional<std::string>("control_points"):std::nullopt;
    if(kind=="point_collection") return std::nullopt;
    if(kind=="fiber" ? (name!="control_points"&&name!="line_points") : name!="points") return std::nullopt;
    return name;
}

inline std::optional<int64_t> dependencyPointOffset(const GeometryArrayInfo& array,const std::vector<int64_t>& index)
{
    if(array.shape.size()!=index.size()+1||array.shape.back()!=3)return std::nullopt;
    int64_t offset=0;
    for(size_t i=0;i<index.size();++i){
        if(index[i]<0||index[i]>=array.shape[i])return std::nullopt;
        offset=offset*array.shape[i]+index[i];
    }
    return offset;
}

inline std::optional<int64_t> dependencyIdentityOffset(const std::vector<std::pair<int64_t,int64_t>>& offsets,int64_t key)
{
    const auto id=std::lower_bound(offsets.begin(),offsets.end(),std::pair{key,int64_t(0)});
    if(id==offsets.end()||id->first!=key)return std::nullopt;
    if(id+1!=offsets.end()&&(id+1)->first==key)throw std::runtime_error("geometry dependency has ambiguous point ID");
    return id->second;
}

inline std::optional<std::vector<uint8_t>> pointValue(const DependencyGeometry& geometry, const PointReference& point)
{
    auto index = point.index;
    auto name = point.array;
    const auto coordinate=dependencyCoordinateArray(geometry.kind,name);
    if(!coordinate)return std::nullopt;
    if (name == "point_ids" || name == "control_point_ids") {
        if (index.size() != 1 || index[0] < 0) return std::nullopt;
        if (geometry.identity && geometry.identityArrays.contains(name)) {
            auto bytes=geometry.identity(name,index[0]);
            if(bytes) {
                GeometrySnapshot snapshot;snapshot.arrays["point"]=*bytes;
                const auto coordinate=name=="point_ids"?"points":"control_points";
                if(geometry.arrays.at(coordinate).dtype=="<f8") (void)snapshot.values<double>("point");
                else (void)snapshot.values<float>("point");
            }
            return bytes;
        }
        const auto found = geometry.arrays.find(name);
        if (found == geometry.arrays.end()) return std::nullopt;
        auto& cache = *geometry.pointIndex;
        if (!cache.loaded) {
            GeometrySnapshot snapshot;
            snapshot.arrays[name] = geometry.read(name, 0, found->second.byteLength);
            const auto ids = snapshot.values<int64_t>(name);
            cache.offsets.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) cache.offsets.emplace_back(ids[i], int64_t(i));
            std::sort(cache.offsets.begin(), cache.offsets.end());
            cache.loaded = true;
        }
        const auto row=dependencyIdentityOffset(cache.offsets,index[0]);
        if(!row)return std::nullopt;
        index={*row};name=*coordinate;
    }
    const auto found = geometry.arrays.find(name);
    if (found == geometry.arrays.end()) return std::nullopt;
    const auto& array = found->second;
    const auto offset=dependencyPointOffset(array,index);
    if(!offset)return std::nullopt;
    const int64_t stride = array.dtype == "<f8" ? 24 : 12;
    auto bytes = geometry.read(name, *offset * stride, stride);
    GeometrySnapshot snapshot; snapshot.arrays["point"] = bytes;
    if (array.dtype == "<f8") (void)snapshot.values<double>("point");
    else if (array.dtype == "<f4") {
        const auto xyz = snapshot.values<float>("point");
        if (geometry.kind == "surface") {
            if (!vc::geometry::validSurfacePoint({xyz[0], xyz[1], xyz[2]})) return std::nullopt;
            if (geometry.arrays.contains(surfaceValidChannel) &&
                geometry.read(surfaceValidChannel, *offset, 1).at(0) == 0) return std::nullopt;
        }
    }
    return bytes;
}

template<class HasPoint>
inline void validateDependencyReferences(const Anchor& anchor,const HasPoint& hasPoint)
{
    if (anchor.wholeGeometry && anchor.points.empty()) return;
    if (anchor.wholeGeometry || anchor.points.empty())
        throw std::runtime_error("geometry edit requires explicit point dependencies or whole_geometry");
    for (const auto& point : anchor.points)
        if (!hasPoint(point))
            throw std::runtime_error("geometry edit refers to a missing or invalid point; update or remove the relationship");
}

inline void validateDependencies(const Anchor& anchor,const DependencyGeometry& geometry)
{
    validateDependencyReferences(anchor,[&](const PointReference& point){return pointValue(geometry,point).has_value();});
}

inline bool dependenciesChanged(const Anchor& anchor, const DependencyGeometry& before, const DependencyGeometry& after)
{
    if (before.kind != after.kind || before.attributes != after.attributes || before.gridOrigin != after.gridOrigin) return true;
    auto changed = [&](const std::string& name) {
        const auto a = before.arrays.find(name), b = after.arrays.find(name);
        if (a == before.arrays.end() || b == after.arrays.end()) return a != before.arrays.end() || b != after.arrays.end();
        if (a->second.dtype != b->second.dtype || a->second.shape != b->second.shape) return true;
        if (before.fingerprints.contains(name) && after.fingerprints.contains(name))
            return before.fingerprints.at(name) != after.fingerprints.at(name);
        return before.read(name, 0, a->second.byteLength) != after.read(name, 0, b->second.byteLength);
    };
    for (const auto& channel : anchor.channels) if (changed(channel)) return true;
    if (anchor.wholeGeometry || anchor.points.empty()) {
        // Whole geometry includes every array; unspecified dependencies
        // remain conservative. Explicit point anchors depend only on their
        // declared points and optional channels.
        std::set<std::string> names;
        for (const auto& [name, a] : before.arrays) names.insert(name);
        for (const auto& [name, a] : after.arrays) names.insert(name);
        for (const auto& name : names) if (changed(name)) return true;
        return false;
    }
    auto identical = [&](const std::string& name) {
        return before.fingerprints.contains(name) && after.fingerprints.contains(name) &&
               before.fingerprints.at(name) == after.fingerprints.at(name);
    };
    for (const auto& point : anchor.points) {
        if (before.kind=="fiber" && point.array=="control_points" && point.index.size()==1) {
            auto identity=[&](const DependencyGeometry& geometry) -> std::optional<int64_t> {
                const auto index=point.index[0];
                if (index<0 || index>=geometry.arrays.at("control_points").shape[0]) return std::nullopt;
                const auto bytes=geometry.read("control_point_ids",index*8,8);
                std::array<int64_t,1> value; copyScalars<int64_t>(bytes,value); return value[0];
            };
            if (identity(before)!=identity(after)) return true;
        }
        const bool sameValidity = before.kind != "surface" || identical(surfaceValidChannel) ||
            (!before.arrays.contains(surfaceValidChannel) && !after.arrays.contains(surfaceValidChannel));
        if (sameValidity && identical(point.array) && (point.array == "point_ids" ? identical("points") : point.array == "control_point_ids" ? identical("control_points") : true)) continue;
        const auto old = pointValue(before, point), next = pointValue(after, point);
        if (!old || !next || old != next) return true;
        // Positional identities cannot be assumed to survive insertion/removal.
        if (point.array != "point_ids" && point.array != "control_point_ids" && before.arrays.at(point.array).shape != after.arrays.at(point.array).shape) return true;
    }
    return false;
}

inline bool endpointChanged(const ObjectStore& db, const Txn& txn, const Anchor& anchor,
                            const std::optional<std::string>& bound, const std::optional<std::string>& head)
{
    if (bound == head) return false;
    if (!bound || !head) return true;
    const auto before = db.geometryRevision(txn, *bound), after = db.geometryRevision(txn, *head);
    if (!before || !after) return true;
    return dependenciesChanged(anchor, dependencyGeometry(db, txn, *before), dependencyGeometry(db, txn, *after));
}
} // namespace vc::objects::detail
