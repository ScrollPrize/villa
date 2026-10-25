#include "vc_object_store/store/GeometryViews.hpp"
#include "vc_object_store/geometry/ArrayBytes.hpp"
#include "geometry/ArrayLayout.hpp"
#include <map>
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>

namespace vc::objects {
namespace detail {
struct GeometryReadState {
    std::shared_ptr<ObjectStore> lease;
    const ObjectStore* db;
    // The snapshot's own transaction, or none when it borrows the caller's.
    std::unique_ptr<ReadTxn> owned;
    const Txn* txn;
    explicit GeometryReadState(ObjectStore& store)
        : db(&store), owned(std::make_unique<ReadTxn>(store.read())), txn(owned.get()) {}
    GeometryReadState(const ObjectStore& store, const Txn& borrowed) : db(&store), txn(&borrowed) {}
    void require() const {
        if (!txn || !txn->active()) throw std::runtime_error("geometry snapshot is closed");
    }
};
struct GeometryViewState {
    std::shared_ptr<GeometryReadState> read;
    GeometryRevision revision;
    std::string objectRevision;
    std::optional<std::map<int64_t, int64_t>> pointIndex;
    const GeometryArrayInfo& array(const std::string& name) const {
        read->require();
        const auto it = std::find_if(revision.arrays.begin(), revision.arrays.end(),
            [&](const auto& a) { return a.name == name; });
        if (it == revision.arrays.end()) throw std::invalid_argument("missing geometry array: " + name);
        return *it;
    }
    void require(const char* name, const char* dtype, size_t rank) const {
        requireArrayLayout(array(name), dtype, rank);
    }
    std::vector<uint8_t> range(const std::string& name, int64_t start, int64_t count) const {
        const auto& a = array(name);
        const auto size = a.shape[0];
        if (start < 0 || count < 0 || start > size || count > size - start)
            throw std::out_of_range("geometry point range out of bounds");
        const int64_t stride = (a.dtype == "<f8" || a.dtype == "<i8" ? 8 : 4) * (a.shape.size() > 1 ? 3 : 1);
        return read->db->readGeometryArray(*read->txn, revision.uuid, name, start * stride, count * stride);
    }
    std::optional<int64_t> indexOf(int64_t id) {
        read->require();
        if (id < 0) return std::nullopt;
        if (!pointIndex) {
            const auto bytes = range("point_ids", 0, array("point_ids").shape[0]);
            std::vector<int64_t> ids(bytes.size() / 8);
            copyScalars<int64_t>(bytes, ids);
            std::map<int64_t, int64_t> index;
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] < 0 || !index.emplace(ids[i], int64_t(i)).second)
                    throw std::invalid_argument("point IDs must be nonnegative and unique");
            pointIndex = std::move(index);
        }
        const auto it = pointIndex->find(id);
        return it == pointIndex->end() ? std::nullopt : std::optional<int64_t>(it->second);
    }
};
}
namespace {
using State = detail::GeometryViewState;
template<class T> std::vector<std::array<T, 3>> points(const State& state, const char* name, int64_t start, int64_t count)
{
    auto result = detail::xyzValues<T>(state.range(name, start, count));
    if constexpr (std::is_floating_point_v<T>)
        for (const auto& p : result) for (auto v : p)
            if (!std::isfinite(v)) throw std::invalid_argument("geometry requires finite coordinates");
    return result;
}
}

GeometrySnapshot::GeometrySnapshot(ObjectStore& db) : state_(std::make_shared<detail::GeometryReadState>(db)) {}
GeometrySnapshot::GeometrySnapshot(std::shared_ptr<ObjectStore> db)
{
    if (!db) throw std::invalid_argument("geometry snapshot requires a store");
    state_ = std::make_shared<detail::GeometryReadState>(*db);
    state_->lease = std::move(db);
}
GeometrySnapshot::GeometrySnapshot(const ObjectStore& db, const Txn& txn)
    : state_(std::make_shared<detail::GeometryReadState>(db, txn)) {}
TransformedGeometryView GeometrySnapshot::viewTransformedAt(const std::string& uuid,const std::string& target,
    double targetScale,const VolumeTransforms* context) const
{
    state_->require();
    const auto record=state_->db->objectByUuid(*state_->txn,uuid);
    if(!record)throw std::invalid_argument("object not found");
    const auto local=context?VolumeTransforms{}:state_->db->volumeTransforms(*state_->txn);
    const auto mapping=(context?*context:local).toVolume(nlohmann::json::parse(record->meta),target,targetScale);
    if(!mapping)throw std::runtime_error("object has no unambiguous transform to volume "+target);
    return TransformedGeometryView(geometry(uuid),*mapping,target,targetScale);
}

TransformedGeometryView GeometryView::viewTransformedAt(const std::string& target,double scale) const {
    state_->read->require();
    auto result=GeometrySnapshot(*state_->read->db,*state_->read->txn).viewTransformedAt(state_->revision.objectUuid,target,scale);
    if(result.revision().uuid!=state_->revision.uuid)throw std::invalid_argument("transform a current object view; historical geometry needs its coordinate metadata");
    return result;
}

std::vector<vcr::catalog::TransformResolution> GeometryView::availableTransforms() const {
    state_->read->require();
    return state_->read->db->availableTransforms(*state_->read->txn,state_->revision.objectUuid);
}

void GeometrySnapshot::close() { state_->txn = nullptr; state_->owned.reset(); state_->lease.reset(); }
bool GeometrySnapshot::active() const { return state_->txn && state_->txn->active(); }
std::shared_ptr<State> GeometrySnapshot::view(const std::string& id, const std::string& kind,
                                              const std::optional<std::string>& geometryRevision) const
{
    state_->require();
    const auto object = state_->db->currentObjectRevision(*state_->txn, id);
    if (!object || object->deleted) throw std::invalid_argument("object not found");
    const auto revision = geometryRevision ? state_->db->geometryRevision(*state_->txn, *geometryRevision)
                                           : state_->db->currentGeometry(*state_->txn, id);
    if (!revision) throw std::invalid_argument("object has no stored geometry");
    if (revision->objectUuid != object->objectUuid)
        throw std::invalid_argument("geometry revision belongs to another object");
    if (!kind.empty() && revision->kind != kind) throw std::invalid_argument("geometry kind mismatch: expected " + kind);
    if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz")
        throw std::invalid_argument("unsupported geometry version or coordinate system");
    auto result = std::make_shared<State>();
    result->read = state_; result->revision = *revision;
    if (!geometryRevision) result->objectRevision = object->uuid;
    if (revision->kind == "point_collection") {
        result->require("points", "<f8", 2);
        result->require("point_ids", "<i8", 1);
        if (result->array("points").shape[0] != result->array("point_ids").shape[0])
            throw std::invalid_argument("point IDs and coordinates differ in length");
    } else if (revision->kind == "fiber") {
        result->require("control_points", "<f8", 2);
        result->require("line_points", "<f8", 2);
    } else if (revision->kind == "track") result->require("points", "<i4", 2);
    else if (revision->kind == "umbilicus") result->require("points", "<f8", 2);
    else if (revision->kind == "surface") result->require("points", "<f4", 3);
    else throw std::invalid_argument("unsupported geometry kind");
    return result;
}
PointCollectionView GeometrySnapshot::pointCollection(const std::string& id, const std::optional<std::string>& revision) const { return PointCollectionView(view(id, "point_collection", revision)); }
FiberView GeometrySnapshot::fiber(const std::string& id, const std::optional<std::string>& revision) const { return FiberView(view(id, "fiber", revision)); }
TrackView GeometrySnapshot::track(const std::string& id, const std::optional<std::string>& revision) const { return TrackView(view(id, "track", revision)); }
UmbilicusView GeometrySnapshot::umbilicus(const std::string& id, const std::optional<std::string>& revision) const { return UmbilicusView(view(id, "umbilicus", revision)); }
SurfaceView GeometrySnapshot::surface(const std::string& id, const std::optional<std::string>& revision) const { return SurfaceView(view(id, "surface", revision)); }
TypedGeometryView GeometrySnapshot::geometry(const std::string& id, const std::optional<std::string>& revision) const
{
    auto s = view(id, "", revision);
    if (s->revision.kind == "point_collection") return PointCollectionView(s);
    if (s->revision.kind == "fiber") return FiberView(s);
    if (s->revision.kind == "track") return TrackView(s);
    if (s->revision.kind == "umbilicus") return UmbilicusView(s);
    return SurfaceView(s);
}
GeometrySnapshot ObjectStore::geometrySnapshot() { return GeometrySnapshot(*this); }

GeometryView::GeometryView(std::shared_ptr<State> state) : state_(std::move(state)) {}
const GeometryRevision& GeometryView::revision() const { state_->read->require(); return state_->revision; }
const std::string& GeometryView::objectRevision() const { state_->read->require(); return state_->objectRevision; }
geometry::Geometry GeometryView::loadGeometry() const
{
    state_->read->require();
    GeometryData data;
    data.kind = state_->revision.kind;
    data.formatVersion = state_->revision.formatVersion;
    data.coordinateSystem = state_->revision.coordinateSystem;
    data.attributes = state_->revision.attributes;
    for (const auto& a : state_->revision.arrays)
        data.arrays.push_back({a.name, a.dtype, a.shape,
            state_->read->db->readGeometryArray(*state_->read->txn, state_->revision.uuid, a.name)});
    return geometry::decode(data);
}
int64_t PointCollectionView::size() const { return state_->array("points").shape[0]; }
std::vector<geometry::PointId> PointCollectionView::ids(int64_t start, int64_t count) const
{
    const auto bytes = state_->range("point_ids", start, count);
    std::vector<int64_t> result(bytes.size() / 8);
    detail::copyScalars<int64_t>(bytes, result);
    return result;
}
std::vector<geometry::Point> PointCollectionView::points(int64_t start, int64_t count) const { return objects::points<double>(*state_, "points", start, count); }
std::optional<geometry::Point> PointCollectionView::point(int64_t id) const
{
    const auto index = state_->indexOf(id);
    return index ? std::optional<geometry::Point>(points(*index, 1).front()) : std::nullopt;
}
geometry::PointCollection PointCollectionView::load() const { return std::get<geometry::PointCollection>(loadGeometry()); }
int64_t FiberView::controlPointCount() const { return state_->array("control_points").shape[0]; }
int64_t FiberView::linePointCount() const { return state_->array("line_points").shape[0]; }
std::vector<geometry::Point> FiberView::controlPoints(int64_t start, int64_t count) const { return objects::points<double>(*state_, "control_points", start, count); }
std::vector<geometry::PointId> FiberView::controlPointIds(int64_t start, int64_t count) const
{
    return scalarsFromBytes<int64_t>(state_->range("control_point_ids", start, count));
}
std::vector<geometry::Point> FiberView::linePoints(int64_t start, int64_t count) const { return objects::points<double>(*state_, "line_points", start, count); }
geometry::Fiber FiberView::load() const { return std::get<geometry::Fiber>(loadGeometry()); }
int64_t TrackView::size() const { return state_->array("points").shape[0]; }
std::vector<geometry::TrackPoint> TrackView::points(int64_t start, int64_t count) const { return objects::points<int32_t>(*state_, "points", start, count); }
geometry::Track TrackView::load() const { return std::get<geometry::Track>(loadGeometry()); }
int64_t UmbilicusView::size() const { return state_->array("points").shape[0]; }
std::vector<geometry::Point> UmbilicusView::points(int64_t start, int64_t count) const { return objects::points<double>(*state_, "points", start, count); }
geometry::Umbilicus UmbilicusView::load() const { return std::get<geometry::Umbilicus>(loadGeometry()); }
int64_t SurfaceView::rows() const { return state_->array("points").shape[0]; }
int64_t SurfaceView::columns() const { return state_->array("points").shape[1]; }
std::array<double, 2> SurfaceView::scale() const
{
    state_->read->require();
    auto value = nlohmann::json::parse(state_->revision.attributes).at("scale").get<std::array<double, 2>>();
    for (auto v : value) if (!(v > 0) || !std::isfinite(v)) throw std::invalid_argument("invalid surface scale");
    return value;
}
geometry::SurfacePoint SurfaceView::vertex(int64_t row, int64_t column) const { return readTile(row, column, 1, 1).points.front(); }
bool SurfaceView::valid(int64_t row, int64_t column) const { return readTile(row, column, 1, 1).valid(0, 0); }
geometry::Surface SurfaceView::readTile(int64_t row, int64_t column, int64_t rows, int64_t columns) const
{
    state_->read->require();
    const auto bytes = state_->read->db->readGeometryRectangle(*state_->read->txn,
        state_->revision.uuid, "points", row, column, rows, columns);
    geometry::Surface result{rows, columns, scale(), detail::xyzValues<float>(bytes)};
    // Vertices without coordinates read as the sentinel; exact stored bits stay
    // available through readGeometryRectangle.
    for (auto& p : result.points)
        if (!geometry::validSurfacePoint(p)) p = {-1, -1, -1};
    const auto& arrays = state_->revision.arrays;
    if (std::any_of(arrays.begin(), arrays.end(), [](const auto& a) { return a.name == detail::surfaceValidChannel; }))
        result.validity = state_->read->db->readGeometryRectangle(*state_->read->txn,
            state_->revision.uuid, detail::surfaceValidChannel, row, column, rows, columns);
    result.validate();
    return result;
}
geometry::Surface SurfaceView::load() const { return std::get<geometry::Surface>(loadGeometry()); }


const GeometryRevision& TransformedGeometryView::revision() const {
    return std::visit([](const auto& v)->const GeometryRevision&{return v.revision();},source_);
}
const std::string& TransformedGeometryView::objectRevision() const {
    return std::visit([](const auto& v)->const std::string&{return v.objectRevision();},source_);
}
std::vector<geometry::Point> TransformedGeometryView::points(const std::string& array,int64_t start,int64_t count) const {
    auto result=std::visit([&](const auto& v)->std::vector<geometry::Point>{
        using T=std::decay_t<decltype(v)>;
        if constexpr(std::is_same_v<T,FiberView>) {
            if(array=="control_points")return v.controlPoints(start,count);
            if(array=="line_points")return v.linePoints(start,count);
        } else if constexpr(std::is_same_v<T,SurfaceView>) {
            throw std::invalid_argument("use readTile for a surface");
        } else if constexpr(std::is_same_v<T,TrackView>) {
            if(array=="points") {
                auto stored=v.points(start,count);std::vector<geometry::Point> out;out.reserve(stored.size());
                for(const auto& p:stored)out.push_back({double(p[0]),double(p[1]),double(p[2])});return out;
            }
        } else {if(array=="points")return v.points(start,count);}
        throw std::invalid_argument("not a coordinate array: "+array);
    },source_);
    for(auto& p:result)p=mapping_.apply(p);
    return result;
}
TransformedSurfaceTile TransformedGeometryView::readTile(int64_t row,int64_t column,int64_t rows,int64_t columns) const {
    const auto* surface=std::get_if<SurfaceView>(&source_);
    if(!surface)throw std::invalid_argument("not a surface");
    const auto source=surface->readTile(row,column,rows,columns);
    TransformedSurfaceTile result{rows,columns};
    result.scale=source.scale;
    // A general affine has no single voxel step. Isotropic mappings do, and
    // preserve the existing grid-step convention of a uniformly scaled surface.
    const auto x=mapping_.apply({1,0,0},true),y=mapping_.apply({0,1,0},true),z=mapping_.apply({0,0,1},true);
    const auto dot=[](const auto& a,const auto& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
    const double squared=dot(x,x),tolerance=1e-12*squared;
    if(squared>0&&std::abs(dot(y,y)-squared)<=tolerance&&std::abs(dot(z,z)-squared)<=tolerance&&
       std::abs(dot(x,y))<=tolerance&&std::abs(dot(x,z))<=tolerance&&std::abs(dot(y,z))<=tolerance)
        for(auto& step:result.scale)step/=std::sqrt(squared);
    result.points.reserve(source.points.size());result.validity.reserve(source.points.size());
    for(size_t i=0;i<source.points.size();++i) {
        const bool valid=source.valid(i/columns,i%columns);
        const auto& p=source.points[i];
        result.validity.push_back(valid?255:0);
        result.points.push_back(valid?mapping_.apply({p[0],p[1],p[2]}):geometry::Point{-1,-1,-1});
    }
    return result;
}
std::vector<GeometryArrayEdit> TransformedGeometryView::coordinateEdits(const std::string& array,int64_t start,
    const std::vector<geometry::Point>& desired) const {
    const auto shown=points(array,start,desired.size()); // checks array/range and snapshot lifetime
    const auto inverse=mapping_.inverse();
    std::vector<GeometryArrayEdit> edits;
    for(size_t i=0;i<desired.size();++i) {
        if(desired[i]==shown[i])continue;
        auto canonical=inverse.apply(desired[i]);
        for(double v:canonical)if(!std::isfinite(v))throw std::invalid_argument("coordinate edit is not finite");
        GeometryArrayEdit edit{array,3*(start+int64_t(i)),{}};
        if(std::holds_alternative<TrackView>(source_)) {
            std::array<int32_t,3> integer;
            for(int k=0;k<3;++k) {
                const double v=std::round(canonical[k]);
                if(v<std::numeric_limits<int32_t>::min()||v>std::numeric_limits<int32_t>::max())throw std::overflow_error("track coordinate exceeds int32");
                integer[k]=int32_t(v);
            }
            edit.bytes=detail::scalarBytes<int32_t>(integer);
        } else edit.bytes=detail::scalarBytes<double>(canonical);
        edits.push_back(std::move(edit));
    }
    return edits;
}
void TransformedGeometryView::validateForWrite(const ObjectStore& db,const Txn& txn) const {
    const auto& pinned=revision();
    const auto head=db.currentGeometry(txn,pinned.objectUuid);
    const auto object=db.currentObjectRevision(txn,pinned.objectUuid);
    if(!head||head->uuid!=pinned.uuid||!object||object->uuid!=objectRevision())
        throw std::runtime_error("object changed; reopen its transformed view");
    const auto mapping=db.volumeTransforms(txn).toVolume(nlohmann::json::parse(object->record.meta),target_,targetScale_);
    if(!mapping||*mapping!=mapping_)throw std::runtime_error("volume registration changed; reopen its transformed view");
}
std::string TransformedGeometryView::writeCoordinates(ObjectStore& db,WriteTxn& txn,const std::string& array,int64_t start,
    const std::vector<geometry::Point>& desired,const std::vector<GeometryEdgeEdit>& edges) const {
    validateForWrite(db,txn);
    return db.editGeometry(txn,revision().objectUuid,coordinateEdits(array,start,desired),revision().uuid,edges);
}
std::string TransformedGeometryView::writeSurface(ObjectStore& db,WriteTxn& txn,int64_t row,int64_t column,int64_t rows,int64_t columns,
    const std::vector<geometry::SurfacePoint>& desired,const std::vector<GeometryEdgeEdit>& edges) const {
    validateForWrite(db,txn);
    return db.editGeometry(txn,revision().objectUuid,surfaceEdits(row,column,rows,columns,desired),revision().uuid,edges);
}
namespace {
geometry::SurfacePoint inverseSurfacePoint(const geometry::SurfacePoint& desired,
    const std::optional<geometry::SurfacePoint>& original,bool valid,const VolumeAffine& forward,const VolumeAffine& inverse) {
    const geometry::SurfacePoint missing{-1,-1,-1};
    auto shown=missing;
    if(original&&valid) {
        const auto p=forward.apply({(*original)[0],(*original)[1],(*original)[2]});
        shown={float(p[0]),float(p[1]),float(p[2])};
    }
    if(original&&desired==shown)return *original;
    if(desired==missing)return missing;
    const auto p=inverse.apply({desired[0],desired[1],desired[2]});
    geometry::SurfacePoint result{float(p[0]),float(p[1]),float(p[2])};
    if(!geometry::validSurfacePoint(result))throw std::invalid_argument("surface coordinate cannot be represented in the canonical frame");
    return result;
}
}
geometry::Fiber TransformedGeometryView::canonicalFiber(const geometry::Fiber& desired) const {
    const auto* fiber=std::get_if<FiberView>(&source_);
    if(!fiber)throw std::invalid_argument("not a fiber");
    if(!desired.controlPointIds.empty()&&desired.controlPointIds.size()!=desired.controlPoints.size())
        throw std::invalid_argument("fiber control point IDs do not match its control points");
    const auto stored=fiber->load();
    const auto inverse=mapping_.inverse();
    auto canonical=[&](const geometry::Point& wanted,const geometry::Point* original) {
        if(original&&mapping_.apply(*original)==wanted)return *original;
        const auto p=inverse.apply(wanted);
        for(double v:p)if(!std::isfinite(v))throw std::invalid_argument("fiber coordinate is not finite in the canonical frame");
        return p;
    };
    std::map<geometry::PointId,size_t> byId;
    for(size_t i=0;i<stored.controlPointIds.size();++i)byId.emplace(stored.controlPointIds[i],i);
    auto result=desired;
    for(size_t i=0;i<desired.controlPoints.size();++i) {
        const geometry::Point* original=nullptr;
        if(desired.controlPointIds.empty()) { if(i<stored.controlPoints.size())original=&stored.controlPoints[i]; }
        else if(const auto found=byId.find(desired.controlPointIds[i]);found!=byId.end())original=&stored.controlPoints[found->second];
        result.controlPoints[i]=canonical(desired.controlPoints[i],original);
    }
    for(size_t i=0;i<desired.linePoints.size();++i)
        result.linePoints[i]=canonical(desired.linePoints[i],i<stored.linePoints.size()?&stored.linePoints[i]:nullptr);
    return result;
}
geometry::PointCollection TransformedGeometryView::canonicalPointCollection(const geometry::PointCollection& desired) const {
    const auto* collection=std::get_if<PointCollectionView>(&source_);
    if(!collection)throw std::invalid_argument("not a point collection");
    if(desired.ids.size()!=desired.points.size())
        throw std::invalid_argument("point collection IDs do not match its points");
    const auto stored=collection->load();
    const auto inverse=mapping_.inverse();
    std::map<geometry::PointId,size_t> byId;
    for(size_t i=0;i<stored.ids.size();++i)byId.emplace(stored.ids[i],i);
    auto result=desired;
    for(size_t i=0;i<desired.points.size();++i) {
        const auto found=byId.find(desired.ids[i]);
        if(found!=byId.end()&&mapping_.apply(stored.points[found->second])==desired.points[i]) {
            result.points[i]=stored.points[found->second];
            continue;
        }
        result.points[i]=inverse.apply(desired.points[i]);
        for(double v:result.points[i])if(!std::isfinite(v))throw std::invalid_argument("point coordinate is not finite in the canonical frame");
    }
    return result;
}
geometry::Surface TransformedGeometryView::canonicalSurface(const geometry::Surface& desired) const {
    const auto* surface=std::get_if<SurfaceView>(&source_);
    if(!surface)throw std::invalid_argument("not a surface");
    if(desired.rows<0||desired.columns<0||size_t(desired.rows)*size_t(desired.columns)!=desired.points.size())
        throw std::invalid_argument("surface shape mismatch");
    const auto stored=surface->readTile(0,0,surface->rows(),surface->columns());
    auto result=desired;result.scale=stored.scale;
    if(result.validity.empty()&&!stored.validity.empty()) {
        result.validity.assign(desired.points.size(),255);
        for(int64_t r=0;r<std::min(stored.rows,desired.rows);++r)
            for(int64_t c=0;c<std::min(stored.columns,desired.columns);++c)
                result.validity[r*desired.columns+c]=stored.validity[r*stored.columns+c];
    }
    const auto inverse=mapping_.inverse();
    for(int64_t r=0;r<desired.rows;++r)for(int64_t c=0;c<desired.columns;++c) {
        const auto i=r*desired.columns+c;
        const bool exists=r<stored.rows&&c<stored.columns;
        result.points[i]=inverseSurfacePoint(desired.points[i],exists?std::optional(stored.points[r*stored.columns+c]):std::nullopt,
            exists&&stored.valid(r,c),mapping_,inverse);
    }
    result.validate();return result;
}
std::vector<GeometryArrayEdit> TransformedGeometryView::surfaceEdits(int64_t row,int64_t column,int64_t rows,int64_t columns,
    const std::vector<geometry::SurfacePoint>& desired) const {
    const auto* surface=std::get_if<SurfaceView>(&source_);
    if(!surface)throw std::invalid_argument("not a surface");
    const auto stored=surface->readTile(row,column,rows,columns);
    if(desired.size()!=stored.points.size())throw std::invalid_argument("surface edit shape mismatch");
    const auto inverse=mapping_.inverse();
    std::vector<GeometryArrayEdit> edits;
    for(int64_t r=0;r<rows;++r)for(int64_t c=0;c<columns;++c) {
        const auto i=r*columns+c;
        const auto point=inverseSurfacePoint(desired[i],stored.points[i],stored.valid(r,c),mapping_,inverse);
        if(point==stored.points[i])continue;
        const auto offset=(row+r)*surface->columns()+column+c;
        edits.push_back({"points",offset*3,detail::scalarBytes<float>(point)});
        if(!stored.validity.empty()&&!stored.validity[i]&&geometry::validSurfacePoint(point))
            edits.push_back({"valid",offset,{255}});
    }
    return edits;
}
} // namespace vc::objects
