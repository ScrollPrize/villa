#include "vc_object_store/store/GeometryViews.hpp"
#include "geometry/ArrayBytes.hpp"
#include "geometry/ArrayLayout.hpp"
#include <nlohmann/json.hpp>
#include <cmath>

namespace vc::objects {
namespace detail {
struct GeometryReadState {
    std::shared_ptr<ObjectStore> lease;
    ObjectStore* db;
    std::unique_ptr<ReadTxn> txn;
    explicit GeometryReadState(ObjectStore& store) : db(&store), txn(std::make_unique<ReadTxn>(store.read())) {}
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
template<class T> GeometryArrayEdit pointEdit(const State& state, const char* name, int64_t index, const std::array<T, 3>& value)
{
    const auto count = state.array(name).shape[0];
    if (index < 0 || index >= count) throw std::out_of_range("geometry point out of bounds");
    if constexpr (std::is_floating_point_v<T>)
        for (auto v : value)
            if (!std::isfinite(v)) throw std::invalid_argument("geometry requires finite coordinates");
    return {name, index * 3, detail::scalarBytes<T>(value)};
}
}

GeometrySnapshot::GeometrySnapshot(ObjectStore& db) : state_(std::make_shared<detail::GeometryReadState>(db)) {}
GeometrySnapshot::GeometrySnapshot(std::shared_ptr<ObjectStore> db)
{
    if (!db) throw std::invalid_argument("geometry snapshot requires a store");
    state_ = std::make_shared<detail::GeometryReadState>(*db);
    state_->lease = std::move(db);
}
void GeometrySnapshot::close() { state_->txn.reset(); state_->lease.reset(); }
bool GeometrySnapshot::active() const { return state_->txn && state_->txn->active(); }
std::shared_ptr<State> GeometrySnapshot::view(const std::string& id, const std::string& kind) const
{
    state_->require();
    const auto object = state_->db->currentObjectRevision(*state_->txn, id);
    if (!object || object->deleted) throw std::invalid_argument("object not found");
    const auto revision = state_->db->currentGeometry(*state_->txn, id);
    if (!revision) throw std::invalid_argument("object has no stored geometry");
    if (!kind.empty() && revision->kind != kind) throw std::invalid_argument("geometry kind mismatch: expected " + kind);
    if (revision->formatVersion != 1 || revision->coordinateSystem != "source_xyz")
        throw std::invalid_argument("unsupported geometry version or coordinate system");
    auto result = std::make_shared<State>();
    result->read = state_; result->revision = *revision; result->objectRevision = object->uuid;
    if (revision->kind == "point_collection") {
        result->require("points", "<f8", 2);
        result->require("point_ids", "<i8", 1);
        if (result->array("points").shape[0] != result->array("point_ids").shape[0])
            throw std::invalid_argument("point IDs and coordinates differ in length");
    } else if (revision->kind == "fiber") {
        result->require("control_points", "<f8", 2);
        result->require("line_points", "<f8", 2);
    } else if (revision->kind == "track") result->require("points", "<i4", 2);
    else if (revision->kind == "patch") result->require("points", "<f4", 3);
    else throw std::invalid_argument("unsupported geometry kind");
    return result;
}
PointCollectionView GeometrySnapshot::pointCollection(const std::string& id) const { return PointCollectionView(view(id, "point_collection")); }
FiberView GeometrySnapshot::fiber(const std::string& id) const { return FiberView(view(id, "fiber")); }
TrackView GeometrySnapshot::track(const std::string& id) const { return TrackView(view(id, "track")); }
PatchView GeometrySnapshot::patch(const std::string& id) const { return PatchView(view(id, "patch")); }
TypedGeometryView GeometrySnapshot::geometry(const std::string& id) const
{
    auto s = view(id, "");
    if (s->revision.kind == "point_collection") return PointCollectionView(s);
    if (s->revision.kind == "fiber") return FiberView(s);
    if (s->revision.kind == "track") return TrackView(s);
    return PatchView(s);
}
GeometrySnapshot ObjectStore::geometrySnapshot() { return GeometrySnapshot(*this); }

GeometryView::GeometryView(std::shared_ptr<State> state) : state_(std::move(state)) {}
const GeometryRevision& GeometryView::revision() const { state_->read->require(); return state_->revision; }
const std::string& GeometryView::objectRevision() const { state_->read->require(); return state_->objectRevision; }
GeometryEdit GeometryView::edit() const { state_->read->require(); return GeometryEdit(state_); }
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
std::vector<geometry::Point> FiberView::linePoints(int64_t start, int64_t count) const { return objects::points<double>(*state_, "line_points", start, count); }
geometry::Fiber FiberView::load() const { return std::get<geometry::Fiber>(loadGeometry()); }
int64_t TrackView::size() const { return state_->array("points").shape[0]; }
std::vector<geometry::TrackPoint> TrackView::points(int64_t start, int64_t count) const { return objects::points<int32_t>(*state_, "points", start, count); }
geometry::Track TrackView::load() const { return std::get<geometry::Track>(loadGeometry()); }
int64_t PatchView::rows() const { return state_->array("points").shape[0]; }
int64_t PatchView::columns() const { return state_->array("points").shape[1]; }
std::array<double, 2> PatchView::scale() const
{
    state_->read->require();
    auto value = nlohmann::json::parse(state_->revision.attributes).at("scale").get<std::array<double, 2>>();
    for (auto v : value) if (!(v > 0) || !std::isfinite(v)) throw std::invalid_argument("invalid patch scale");
    return value;
}
geometry::PatchPoint PatchView::vertex(int64_t row, int64_t column) const { return readTile(row, column, 1, 1).points.front(); }
bool PatchView::valid(int64_t row, int64_t column) const { return geometry::validPatchPoint(vertex(row, column)); }
geometry::Patch PatchView::readTile(int64_t row, int64_t column, int64_t rows, int64_t columns) const
{
    state_->read->require();
    const auto bytes = state_->read->db->readGeometryRectangle(*state_->read->txn,
        state_->revision.uuid, "points", row, column, rows, columns);
    geometry::Patch result{rows, columns, scale(), detail::xyzValues<float>(bytes)};
    result.validate();
    return result;
}
geometry::Patch PatchView::load() const { return std::get<geometry::Patch>(loadGeometry()); }

GeometryEdit::GeometryEdit(std::shared_ptr<State> state) : state_(std::move(state)) {}
void GeometryEdit::requireKind(const char* kind) const
{
    state_->read->require();
    if (committed_) throw std::runtime_error("geometry edit is already committed");
    if (state_->revision.kind != kind) throw std::invalid_argument("geometry edit kind mismatch");
}
void GeometryEdit::movePoint(int64_t id, const geometry::Point& p)
{
    requireKind("point_collection");
    const auto index = state_->indexOf(id);
    if (!index) throw std::out_of_range("point ID not found");
    edits_.push_back(pointEdit(*state_, "points", *index, p));
}
void GeometryEdit::moveControlPoint(int64_t index, const geometry::Point& p) { requireKind("fiber"); edits_.push_back(pointEdit(*state_, "control_points", index, p)); }
void GeometryEdit::moveLinePoint(int64_t index, const geometry::Point& p) { requireKind("fiber"); edits_.push_back(pointEdit(*state_, "line_points", index, p)); }
void GeometryEdit::moveTrackPoint(int64_t index, const geometry::TrackPoint& p) { requireKind("track"); edits_.push_back(pointEdit(*state_, "points", index, p)); }
void GeometryEdit::moveVertex(int64_t row, int64_t column, const geometry::PatchPoint& p)
{
    requireKind("patch");
    const auto& shape = state_->array("points").shape;
    if (row < 0 || column < 0 || row >= shape[0] || column >= shape[1])
        throw std::out_of_range("patch vertex out of bounds");
    if (!geometry::validPatchPoint(p) && p != geometry::PatchPoint{-1, -1, -1})
        throw std::invalid_argument("invalid patch vertices must use (-1,-1,-1)");
    edits_.push_back({"points", (row * shape[1] + column) * 3, detail::scalarBytes<float>(p)});
}
std::vector<EdgeRecord> GeometryEdit::affectedRelationships() const
{
    requireKind(state_->revision.kind.c_str());
    return state_->read->db->affectedGeometryEditEdges(*state_->read->txn, state_->revision.objectUuid, edits_, state_->revision.uuid);
}
std::string GeometryEdit::commit(const std::vector<GeometryEdgeEdit>& decisions)
{
    requireKind(state_->revision.kind.c_str());
    auto& db = *state_->read->db;
    RevisionSelection expected{{state_->revision.objectUuid, state_->objectRevision}};
    // A relationship action also revises its other endpoint. Pin those heads
    // to the same snapshot, leaving the existing edge checks authoritative.
    for (const auto& decision : decisions) {
        const auto edge = db.edgeByUuid(*state_->read->txn, decision.edgeUuid);
        if (!edge) throw std::invalid_argument("relationship not found in snapshot");
        if (edge->aUuid != state_->revision.objectUuid && edge->bUuid != state_->revision.objectUuid)
            throw std::invalid_argument("relationship does not belong to edited object");
        for (const auto& id : {edge->aUuid, edge->bUuid}) {
            const auto revision = db.currentObjectRevision(*state_->read->txn, id);
            if (!revision) throw std::invalid_argument("relationship endpoint not found");
            expected[id] = revision->uuid;
        }
    }
    auto write = db.write(expected);
    const auto result = db.editGeometry(write, state_->revision.objectUuid, edits_, state_->revision.uuid, decisions);
    write.commit();
    committed_ = true;
    return result;
}
} // namespace vc::objects
