#pragma once
#include "StorageBindings.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <nanobind/stl/variant.h>
#include <nanobind/ndarray.h>

namespace {
vo::GeometryData geometryInputFromPy(const nb::object& input)
{
    if (nb::isinstance<nb::dict>(input)) return geometryFromPy(nb::cast<nb::dict>(input));
    return vc::geometry::encode(nb::cast<vc::geometry::Geometry>(input));
}
// An owned NumPy array of `shape` holding `values` (the capsule frees them).
template<class T> nb::ndarray<nb::numpy, T> ownedArray(std::vector<T> values, std::vector<size_t> shape)
{
    auto* data = new std::vector<T>(std::move(values));
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(data->data(), shape.size(), shape.data(), owner);
}
template<class T> nb::ndarray<nb::numpy, T> xyzArray(const std::vector<std::array<T, 3>>& points, std::vector<size_t> shape)
{
    std::vector<T> flat(points.size() * 3);
    for (size_t i = 0; i < points.size(); ++i) std::copy(points[i].begin(), points[i].end(), flat.begin() + 3 * i);
    return ownedArray(std::move(flat), std::move(shape));
}
// The store's per-vertex validity of a surface (coordinates and `valid`).
nb::ndarray<nb::numpy, bool> validMask(const vc::geometry::Surface& s)
{
    auto* data = new std::vector<uint8_t>(size_t(s.rows * s.columns));
    for (int64_t r = 0; r < s.rows; ++r)
        for (int64_t c = 0; c < s.columns; ++c) (*data)[size_t(r * s.columns + c)] = s.valid(r, c);
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<uint8_t>*>(p); });
    const size_t shape[2] = {size_t(s.rows), size_t(s.columns)};
    return nb::ndarray<nb::numpy, bool>(reinterpret_cast<bool*>(data->data()), 2, shape, owner);
}
}
namespace vc::python {
inline nb::object volumeTransformsToPy(const std::vector<vcr::catalog::TransformResolution>& resolutions) {
    nlohmann::json result=nlohmann::json::array();
    for(const auto& mapping:resolutions) {
        nlohmann::json candidates=nlohmann::json::array();
        for(const auto& c:mapping.candidates)candidates.push_back({{"matrix",c.forward.json()},{"path",c.path}});
        result.push_back({{"volume",mapping.target},{"available",mapping.status==vcr::catalog::TransformStatus::Resolved},
            {"ambiguous",mapping.status==vcr::catalog::TransformStatus::Ambiguous},{"candidates",candidates}});
    }
    return jsonLoads(result.dump());
}

inline void bindTypedGeometry(nb::module_& m)
{
    namespace vg = vc::geometry;
    nb::class_<vg::PointCollection>(m, "PointCollection")
        .def(nb::init<std::vector<int64_t>, std::vector<vg::Point>>(), nb::arg("ids"), nb::arg("points"))
        .def_rw("ids", &vg::PointCollection::ids)
        .def_rw("points", &vg::PointCollection::points)
        .def("validate", &vg::PointCollection::validate)
        .def("point", &vg::PointCollection::point, nb::arg("point_id"))
        .def("index_of", &vg::PointCollection::indexOf, nb::arg("point_id"));
    nb::class_<vg::Fiber>(m, "Fiber")
        .def(nb::init<std::vector<vg::Point>, std::vector<vg::Point>>(),
             nb::arg("control_points"), nb::arg("line_points") = std::vector<vg::Point>{})
        .def_rw("control_points", &vg::Fiber::controlPoints)
        .def_rw("line_points", &vg::Fiber::linePoints)
        .def_rw("control_point_ids", &vg::Fiber::controlPointIds)
        .def("validate", &vg::Fiber::validate)
        .def("length", &vg::Fiber::length)
        .def_static("from_arrays", [](nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> controls,
                                      std::optional<nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>> line) {
            vg::Fiber f;
            f.controlPoints.resize(controls.shape(0));
            for (size_t i = 0; i < f.controlPoints.size(); ++i) f.controlPoints[i] = {controls.data()[3*i], controls.data()[3*i+1], controls.data()[3*i+2]};
            if (line) {
                f.linePoints.resize(line->shape(0));
                for (size_t i = 0; i < f.linePoints.size(); ++i) f.linePoints[i] = {line->data()[3*i], line->data()[3*i+1], line->data()[3*i+2]};
            }
            f.validate();
            return f;
        }, nb::arg("control_points"), nb::arg("line_points") = nb::none(),
           "A fiber from (n, 3) float64 control points and optional sampled line points; control IDs are 0..n-1.");
    nb::class_<vg::Track>(m, "Track")
        .def(nb::init<std::vector<vg::TrackPoint>>(), nb::arg("points"))
        .def_rw("points", &vg::Track::points)
        .def("validate", &vg::Track::validate)
        .def("length", &vg::Track::length);
    nb::class_<vg::Umbilicus>(m, "Umbilicus")
        .def(nb::init<std::vector<vg::Point>>(), nb::arg("points"))
        .def_rw("points", &vg::Umbilicus::points)
        .def("validate", &vg::Umbilicus::validate)
        .def_static("from_arrays", [](nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> points) {
            vg::Umbilicus u;
            u.points.resize(points.shape(0));
            for (size_t i = 0; i < u.points.size(); ++i) u.points[i] = {points.data()[3*i], points.data()[3*i+1], points.data()[3*i+2]};
            u.validate();
            return u;
        }, nb::arg("points"), "An umbilicus from (n, 3) float64 control points.");
    nb::class_<vg::Surface>(m, "Surface")
        .def(nb::init<int64_t, int64_t, std::array<double, 2>, std::vector<vg::SurfacePoint>>(),
             nb::arg("rows"), nb::arg("columns"), nb::arg("scale"), nb::arg("points"))
        .def_rw("rows", &vg::Surface::rows)
        .def_rw("columns", &vg::Surface::columns)
        .def_rw("scale", &vg::Surface::scale)
        .def_rw("points", &vg::Surface::points)
        .def_prop_rw("validity",
            [](const vg::Surface& self) { return nb::bytes(reinterpret_cast<const char*>(self.validity.data()), self.validity.size()); },
            [](vg::Surface& self, const nb::bytes& value) {
                const auto* p = reinterpret_cast<const uint8_t*>(value.c_str());
                self.validity.assign(p, p + value.size());
            })
        .def_prop_rw("channels",
            [](const vg::Surface& self) {
                nb::list result;
                for (const auto& channel : self.channels) result.append(arrayToPy(channel));
                return result;
            },
            [](vg::Surface& self, const nb::list& value) {
                self.channels.clear();
                for (const auto item : value) self.channels.push_back(arrayFromPy(item));
            })
        .def("validate", &vg::Surface::validate)
        .def("vertex", &vg::Surface::vertex, nb::arg("row"), nb::arg("column"), nb::rv_policy::copy)
        .def("valid", &vg::Surface::valid, nb::arg("row"), nb::arg("column"))
        .def_static("from_arrays", [](nb::ndarray<const float, nb::shape<-1, -1, 3>, nb::c_contig, nb::device::cpu> points,
                                      std::array<double, 2> scale,
                                      std::optional<nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>> validity) {
            vg::Surface s{int64_t(points.shape(0)), int64_t(points.shape(1)), scale, {}};
            s.points.resize(size_t(s.rows * s.columns));
            for (size_t i = 0; i < s.points.size(); ++i) s.points[i] = {points.data()[3*i], points.data()[3*i+1], points.data()[3*i+2]};
            if (validity) {
                if (validity->shape(0) != points.shape(0) || validity->shape(1) != points.shape(1))
                    throw std::invalid_argument("validity must match the points grid");
                s.validity.assign(validity->data(), validity->data() + validity->size());
            }
            s.validate();
            return s;
        }, nb::arg("points"), nb::arg("scale"), nb::arg("validity") = nb::none(),
           "A surface from a (rows, columns, 3) float32 XYZ array (missing vertices (-1,-1,-1)) and optional (rows, columns) uint8 validity.")
        .def("points_array", [](const vg::Surface& s) {
            return xyzArray(s.points, {size_t(s.rows), size_t(s.columns), 3});
        }, "Points as a (rows, columns, 3) float32 array.")
        .def("valid_mask", &validMask, "Per-vertex validity (coordinates and the valid channel) as a (rows, columns) bool array.");

    nb::class_<vo::GeometryView>(m, "GeometryView")
        .def_prop_ro("available_transforms",[](const vo::GeometryView& v){return volumeTransformsToPy(v.availableTransforms());})
        .def("view_transformed_at",&vo::GeometryView::viewTransformedAt,nb::arg("volume"),nb::arg("target_scale")=1.)
        .def_prop_ro("revision", [](const vo::GeometryView& self) { return geometryToPy(self.revision()); })
        .def_prop_ro("object_revision", &vo::GeometryView::objectRevision, nb::rv_policy::copy);
    nb::class_<vo::PointCollectionView, vo::GeometryView>(m, "PointCollectionView")
        .def("__len__", &vo::PointCollectionView::size)
        .def("ids", &vo::PointCollectionView::ids, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("points", &vo::PointCollectionView::points, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("point", &vo::PointCollectionView::point, nb::arg("point_id"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::PointCollectionView::load, nb::call_guard<nb::gil_scoped_release>());
    nb::class_<vo::FiberView, vo::GeometryView>(m, "FiberView")
        .def_prop_ro("control_point_count", &vo::FiberView::controlPointCount)
        .def_prop_ro("line_point_count", &vo::FiberView::linePointCount)
        .def("control_points", &vo::FiberView::controlPoints, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("control_point_ids", &vo::FiberView::controlPointIds, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("line_points", &vo::FiberView::linePoints, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::FiberView::load, nb::call_guard<nb::gil_scoped_release>())
        .def("control_points_array", [](const vo::FiberView& v, int64_t start, std::optional<int64_t> count) {
            const auto points = v.controlPoints(start, count.value_or(v.controlPointCount() - start));
            return xyzArray(points, {points.size(), 3});
        }, nb::arg("start") = 0, nb::arg("count") = nb::none(), "Control points as an (n, 3) float64 array.")
        .def("line_points_array", [](const vo::FiberView& v, int64_t start, std::optional<int64_t> count) {
            const auto points = v.linePoints(start, count.value_or(v.linePointCount() - start));
            return xyzArray(points, {points.size(), 3});
        }, nb::arg("start") = 0, nb::arg("count") = nb::none(), "Sampled line points as an (n, 3) float64 array.");
    nb::class_<vo::TrackView, vo::GeometryView>(m, "TrackView")
        .def("__len__", &vo::TrackView::size)
        .def("points", &vo::TrackView::points, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::TrackView::load, nb::call_guard<nb::gil_scoped_release>());
    nb::class_<vo::UmbilicusView, vo::GeometryView>(m, "UmbilicusView")
        .def("__len__", &vo::UmbilicusView::size)
        .def("points", &vo::UmbilicusView::points, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::UmbilicusView::load, nb::call_guard<nb::gil_scoped_release>())
        .def("points_array", [](const vo::UmbilicusView& v) {
            const auto points = v.points(0, v.size());
            return xyzArray(points, {points.size(), 3});
        }, "Control points as an (n, 3) float64 array.");
    nb::class_<vo::SurfaceView, vo::GeometryView>(m, "SurfaceView")
        .def_prop_ro("rows", &vo::SurfaceView::rows)
        .def_prop_ro("columns", &vo::SurfaceView::columns)
        .def_prop_ro("scale", &vo::SurfaceView::scale)
        .def("vertex", &vo::SurfaceView::vertex, nb::arg("row"), nb::arg("column"), nb::call_guard<nb::gil_scoped_release>())
        .def("valid", &vo::SurfaceView::valid, nb::arg("row"), nb::arg("column"), nb::call_guard<nb::gil_scoped_release>())
        .def("read_tile", &vo::SurfaceView::readTile, nb::arg("row"), nb::arg("column"),
             nb::arg("rows"), nb::arg("columns"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::SurfaceView::load, nb::call_guard<nb::gil_scoped_release>())
        .def("read_arrays", [](const vo::SurfaceView& v, int64_t row, int64_t column, std::optional<int64_t> rows, std::optional<int64_t> columns) {
            const auto tile = v.readTile(row, column, rows.value_or(v.rows() - row), columns.value_or(v.columns() - column));
            return nb::make_tuple(xyzArray(tile.points, {size_t(tile.rows), size_t(tile.columns), 3}), validMask(tile));
        }, nb::arg("row") = 0, nb::arg("column") = 0, nb::arg("rows") = nb::none(), nb::arg("columns") = nb::none(),
           "(points, valid): a rectangle (default: all) as a (rows, columns, 3) float32 array and the store's per-vertex validity as a bool array.");
    nb::class_<vo::TransformedGeometryView>(m,"TransformedGeometryView")
        .def_prop_ro("object_revision",&vo::TransformedGeometryView::objectRevision)
        .def_prop_ro("canonical",&vo::TransformedGeometryView::canonical)
        .def_prop_ro("target_volume",&vo::TransformedGeometryView::targetVolume)
        .def_prop_ro("matrix",[](const vo::TransformedGeometryView& v){return jsonLoads(v.mapping().json().dump());})
        .def("points",&vo::TransformedGeometryView::points,nb::arg("array"),nb::arg("start"),nb::arg("count"))
        .def("read_tile",[](const vo::TransformedGeometryView& v,int64_t row,int64_t col,int64_t rows,int64_t cols){
            auto tile=v.readTile(row,col,rows,cols);nb::dict result;
            result["points"]=xyzArray(tile.points,{size_t(rows),size_t(cols),3});
            result["validity"]=ownedArray(std::move(tile.validity),{size_t(rows),size_t(cols)});result["scale"]=tile.scale;return result;
        },nb::arg("row"),nb::arg("column"),nb::arg("rows"),nb::arg("columns"))
        .def("coordinate_edits",[](const vo::TransformedGeometryView& v,const std::string& array,int64_t start,const std::vector<vg::Point>& desired){
            nb::list result;for(const auto& e:v.coordinateEdits(array,start,desired)){
                nb::dict item;item["array"]=e.array;item["element_offset"]=e.elementOffset;
                item["data"]=nb::bytes(reinterpret_cast<const char*>(e.bytes.data()),e.bytes.size());result.append(item);
            }return result;
        },nb::arg("array"),nb::arg("start"),nb::arg("points"));
    nb::class_<vo::GeometrySnapshot>(m, "GeometrySnapshot")
        .def("__enter__", [](nb::object self) {
            if (!nb::cast<vo::GeometrySnapshot&>(self).active()) throw std::runtime_error("geometry snapshot is closed");
            return self;
        })
        .def("__exit__", [](vo::GeometrySnapshot& self, nb::args) { self.close(); })
        .def("view_transformed_at",[](const vo::GeometrySnapshot& snapshot,const std::string& uuid,const std::string& volume,double scale){
            return snapshot.viewTransformedAt(uuid,volume,scale);
        },nb::arg("object_uuid"),nb::arg("volume"),nb::arg("target_scale")=1.)
        .def("close", &vo::GeometrySnapshot::close)
        .def_prop_ro("active", &vo::GeometrySnapshot::active)
        .def("point_collection", &vo::GeometrySnapshot::pointCollection, nb::arg("object_uuid"), nb::arg("geometry_revision") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("fiber", &vo::GeometrySnapshot::fiber, nb::arg("object_uuid"), nb::arg("geometry_revision") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("track", &vo::GeometrySnapshot::track, nb::arg("object_uuid"), nb::arg("geometry_revision") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("surface", &vo::GeometrySnapshot::surface, nb::arg("object_uuid"), nb::arg("geometry_revision") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("umbilicus", &vo::GeometrySnapshot::umbilicus, nb::arg("object_uuid"), nb::arg("geometry_revision") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("geometry", &vo::GeometrySnapshot::geometry, nb::arg("object_uuid"), nb::arg("geometry_revision") = nb::none(), nb::call_guard<nb::gil_scoped_release>());
}
} // namespace vc::python
