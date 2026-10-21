#include "vc_object_store/geometry/Geometry.hpp"
#include "geometry/ArrayBytes.hpp"
#include "geometry/ArrayLayout.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <set>

namespace vc::geometry {
namespace {
template<class T> void finite(const std::vector<std::array<T, 3>>& points)
{
    for (const auto& p : points)
        for (auto v : p)
            if (!std::isfinite(v)) throw std::invalid_argument("geometry requires finite coordinates");
}
template<class T> double length(std::span<const std::array<T, 3>> points)
{
    double result = 0;
    for (size_t i = 1; i < points.size(); ++i) {
        double squared = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const double delta = double(points[i][axis]) - double(points[i-1][axis]);
            squared += delta * delta;
        }
        result += std::sqrt(squared);
    }
    if (!std::isfinite(result)) throw std::invalid_argument("non-finite polyline length");
    return result;
}
const objects::GeometryArray& require(const objects::GeometryData& data,
    const char* name, const char* dtype, size_t rank)
{
    const auto it = std::find_if(data.arrays.begin(), data.arrays.end(),
        [&](const auto& a) { return a.name == name; });
    if (it == data.arrays.end() || it->dtype != dtype || it->shape.size() != rank)
        throw std::invalid_argument(std::string("invalid geometry array: ") + name);
    objects::detail::requireArrayLayout({it->name, it->dtype, it->shape, int64_t(it->bytes.size())}, dtype, rank);
    return *it;
}
}

double polylineLength(std::span<const Point> points) { return length(points); }
double polylineLength(std::span<const TrackPoint> points) { return length(points); }
bool validPatchPoint(const PatchPoint& p)
{
    return p[2] > 0 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
void PointCollection::validate() const
{
    if (ids.size() != points.size()) throw std::invalid_argument("point IDs and coordinates differ in length");
    std::set<PointId> seen;
    for (auto id : ids)
        if (id < 0 || !seen.insert(id).second) throw std::invalid_argument("point IDs must be nonnegative and unique");
    finite(points);
}
std::optional<size_t> PointCollection::indexOf(PointId id) const
{
    std::optional<size_t> result;
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == id) {
        if (result) throw std::invalid_argument("ambiguous point ID");
        result = i;
    }
    return result;
}
std::optional<Point> PointCollection::point(PointId id) const
{
    const auto i = indexOf(id);
    return i ? std::optional<Point>(points.at(*i)) : std::nullopt;
}
void Fiber::validate() const {
    finite(controlPoints); finite(linePoints);
    if (!controlPointIds.empty()) PointCollection{controlPointIds,controlPoints}.validate();
}
double Fiber::length() const { validate(); return polylineLength(linePoints.empty() ? controlPoints : linePoints); }
void Track::validate() const {}
double Track::length() const { return polylineLength(points); }
void Patch::validate() const
{
    if (rows < 0 || columns < 0 || (columns && rows > INT64_MAX / columns) ||
        uint64_t(rows) * uint64_t(columns) != points.size())
        throw std::invalid_argument("patch dimensions do not match vertices");
    for (auto v : scale)
        if (!(v > 0) || !std::isfinite(v)) throw std::invalid_argument("patch scale must be positive and finite");
    finite(points);
    for (const auto& p : points)
        if (!validPatchPoint(p) && p != PatchPoint{-1, -1, -1})
            throw std::invalid_argument("invalid patch vertices must use (-1,-1,-1)");
}
const PatchPoint& Patch::vertex(int64_t row, int64_t column) const
{
    if (row < 0 || column < 0 || row >= rows || column >= columns ||
        (columns && row > (INT64_MAX - column) / columns))
        throw std::out_of_range("patch vertex out of bounds");
    return points.at(static_cast<size_t>(row * columns + column));
}
bool Patch::valid(int64_t row, int64_t column) const { return validPatchPoint(vertex(row, column)); }

objects::GeometryData encode(const Geometry& geometry)
{
    using namespace objects::detail;
    return std::visit([](const auto& g) {
        using T = std::decay_t<decltype(g)>;
        g.validate();
        objects::GeometryData data;
        if constexpr (std::is_same_v<T, PointCollection>) {
            data.kind = "point_collection";
            data.arrays = {xyzArray<double>("points", "<f8", {int64_t(g.points.size()), 3}, g.points),
                scalarArray<int64_t>("point_ids", "<i8", {int64_t(g.ids.size())}, g.ids)};
        } else if constexpr (std::is_same_v<T, Fiber>) {
            data.kind = "fiber";
            data.arrays = {xyzArray<double>("control_points", "<f8", {int64_t(g.controlPoints.size()), 3}, g.controlPoints),
                xyzArray<double>("line_points", "<f8", {int64_t(g.linePoints.size()), 3}, g.linePoints)};
            if (!g.controlPointIds.empty()) {
                PointCollection{g.controlPointIds,g.controlPoints}.validate();
                data.arrays.push_back(scalarArray<int64_t>("control_point_ids","<i8",{int64_t(g.controlPointIds.size())},g.controlPointIds));
            }
        } else if constexpr (std::is_same_v<T, Track>) {
            data.kind = "track";
            data.arrays = {xyzArray<int32_t>("points", "<i4", {int64_t(g.points.size()), 3}, g.points)};
        } else {
            data.kind = "patch";
            data.attributes = nlohmann::json{{"scale", g.scale}}.dump();
            data.arrays = {xyzArray<float>("points", "<f4", {g.rows, g.columns, 3}, g.points)};
        }
        return data;
    }, geometry);
}

Geometry decode(const objects::GeometryData& data)
{
    using namespace objects::detail;
    if (data.formatVersion != 1 || data.coordinateSystem != "source_xyz")
        throw std::invalid_argument("unsupported geometry version or coordinate system");
    // Do not silently discard auxiliary geometry when converting to owned types.
    std::set<std::string> names;
    for (const auto& a : data.arrays)
        if (!names.insert(a.name).second) throw std::invalid_argument("duplicate geometry array");
    auto expected = data.kind == "fiber" ? std::set<std::string>{"control_points", "line_points"} :
        data.kind == "point_collection" ? std::set<std::string>{"points", "point_ids"} : std::set<std::string>{"points"};
    if (data.kind=="fiber" && names.contains("control_point_ids")) expected.insert("control_point_ids");
    if (names != expected) throw std::invalid_argument("typed geometry requires exactly its native arrays");
    const auto attributes = nlohmann::json::parse(data.attributes);
    if (!attributes.is_object() || attributes.size() != (data.kind == "patch" ? 1 : 0))
        throw std::invalid_argument("typed geometry cannot discard extra geometry attributes");
    Geometry result;
    if (data.kind == "point_collection") {
        PointCollection g;
        g.points = xyzValues<double>(require(data, "points", "<f8", 2).bytes);
        const auto& ids = require(data, "point_ids", "<i8", 1);
        g.ids.resize(ids.bytes.size() / 8);
        copyScalars<int64_t>(ids.bytes, g.ids);
        result = std::move(g);
    } else if (data.kind == "fiber") {
        result = Fiber{xyzValues<double>(require(data, "control_points", "<f8", 2).bytes),
            xyzValues<double>(require(data, "line_points", "<f8", 2).bytes)};
        if (names.contains("control_point_ids")) {
            auto& fiber=std::get<Fiber>(result);
            const auto& ids=require(data,"control_point_ids","<i8",1);
            fiber.controlPointIds.resize(ids.bytes.size()/8); copyScalars<int64_t>(ids.bytes,fiber.controlPointIds);
            PointCollection{fiber.controlPointIds,fiber.controlPoints}.validate();
        }
    } else if (data.kind == "track") {
        result = Track{xyzValues<int32_t>(require(data, "points", "<i4", 2).bytes)};
    } else if (data.kind == "patch") {
        const auto& a = require(data, "points", "<f4", 3);
        result = Patch{a.shape[0], a.shape[1], attributes.at("scale").get<std::array<double, 2>>(), xyzValues<float>(a.bytes)};
    } else throw std::invalid_argument("unsupported geometry kind");
    std::visit([](const auto& g) { g.validate(); }, result);
    return result;
}
} // namespace vc::geometry
