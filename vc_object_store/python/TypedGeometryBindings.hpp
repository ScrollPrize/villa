#pragma once
#include "StorageBindings.hpp"
#include "vc_object_store/store/GeometryViews.hpp"
#include <nanobind/stl/variant.h>

namespace {
vo::GeometryData geometryInputFromPy(const nb::object& input)
{
    if (nb::isinstance<nb::dict>(input)) return geometryFromPy(nb::cast<nb::dict>(input));
    return vc::geometry::encode(nb::cast<vc::geometry::Geometry>(input));
}
}
namespace vc::python {
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
        .def("length", &vg::Fiber::length);
    nb::class_<vg::Track>(m, "Track")
        .def(nb::init<std::vector<vg::TrackPoint>>(), nb::arg("points"))
        .def_rw("points", &vg::Track::points)
        .def("validate", &vg::Track::validate)
        .def("length", &vg::Track::length);
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
        .def("valid", &vg::Surface::valid, nb::arg("row"), nb::arg("column"));

    nb::class_<vo::GeometryView>(m, "GeometryView")
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
        .def("line_points", &vo::FiberView::linePoints, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::FiberView::load, nb::call_guard<nb::gil_scoped_release>());
    nb::class_<vo::TrackView, vo::GeometryView>(m, "TrackView")
        .def("__len__", &vo::TrackView::size)
        .def("points", &vo::TrackView::points, nb::arg("start"), nb::arg("count"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::TrackView::load, nb::call_guard<nb::gil_scoped_release>());
    nb::class_<vo::SurfaceView, vo::GeometryView>(m, "SurfaceView")
        .def_prop_ro("rows", &vo::SurfaceView::rows)
        .def_prop_ro("columns", &vo::SurfaceView::columns)
        .def_prop_ro("scale", &vo::SurfaceView::scale)
        .def("vertex", &vo::SurfaceView::vertex, nb::arg("row"), nb::arg("column"), nb::call_guard<nb::gil_scoped_release>())
        .def("valid", &vo::SurfaceView::valid, nb::arg("row"), nb::arg("column"), nb::call_guard<nb::gil_scoped_release>())
        .def("read_tile", &vo::SurfaceView::readTile, nb::arg("row"), nb::arg("column"),
             nb::arg("rows"), nb::arg("columns"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::SurfaceView::load, nb::call_guard<nb::gil_scoped_release>());
    nb::class_<vo::GeometrySnapshot>(m, "GeometrySnapshot")
        .def("__enter__", [](nb::object self) {
            if (!nb::cast<vo::GeometrySnapshot&>(self).active()) throw std::runtime_error("geometry snapshot is closed");
            return self;
        })
        .def("__exit__", [](vo::GeometrySnapshot& self, nb::args) { self.close(); })
        .def("close", &vo::GeometrySnapshot::close)
        .def_prop_ro("active", &vo::GeometrySnapshot::active)
        .def("point_collection", &vo::GeometrySnapshot::pointCollection, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("fiber", &vo::GeometrySnapshot::fiber, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("track", &vo::GeometrySnapshot::track, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("surface", &vo::GeometrySnapshot::surface, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("geometry", &vo::GeometrySnapshot::geometry, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>());
}
} // namespace vc::python
