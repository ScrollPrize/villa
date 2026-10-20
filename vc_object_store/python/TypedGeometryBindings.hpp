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
        .def("validate", &vg::Fiber::validate)
        .def("length", &vg::Fiber::length);
    nb::class_<vg::Track>(m, "Track")
        .def(nb::init<std::vector<vg::TrackPoint>>(), nb::arg("points"))
        .def_rw("points", &vg::Track::points)
        .def("validate", &vg::Track::validate)
        .def("length", &vg::Track::length);
    nb::class_<vg::Patch>(m, "Patch")
        .def(nb::init<int64_t, int64_t, std::array<double, 2>, std::vector<vg::PatchPoint>>(),
             nb::arg("rows"), nb::arg("columns"), nb::arg("scale"), nb::arg("points"))
        .def_rw("rows", &vg::Patch::rows)
        .def_rw("columns", &vg::Patch::columns)
        .def_rw("scale", &vg::Patch::scale)
        .def_rw("points", &vg::Patch::points)
        .def("validate", &vg::Patch::validate)
        .def("vertex", &vg::Patch::vertex, nb::arg("row"), nb::arg("column"), nb::rv_policy::copy)
        .def("valid", &vg::Patch::valid, nb::arg("row"), nb::arg("column"));

    nb::class_<vo::GeometryEdit>(m, "GeometryEdit")
        .def("move_point", &vo::GeometryEdit::movePoint, nb::arg("point_id"), nb::arg("xyz"))
        .def("move_control_point", &vo::GeometryEdit::moveControlPoint, nb::arg("index"), nb::arg("xyz"))
        .def("move_line_point", &vo::GeometryEdit::moveLinePoint, nb::arg("index"), nb::arg("xyz"))
        .def("move_track_point", &vo::GeometryEdit::moveTrackPoint, nb::arg("index"), nb::arg("xyz"))
        .def("move_vertex", &vo::GeometryEdit::moveVertex, nb::arg("row"), nb::arg("column"), nb::arg("xyz"))
        .def("affected_relationships", [](const vo::GeometryEdit& self) {
            std::vector<vo::EdgeRecord> edges;
            { nb::gil_scoped_release release; edges = self.affectedRelationships(); }
            nb::list result;
            for (const auto& edge : edges) result.append(edgeToPy(edge));
            return result;
        })
        .def("commit", [](vo::GeometryEdit& self, const nb::list& input) {
            auto decisions = edgeEditsFromPy(input);
            nb::gil_scoped_release release;
            return self.commit(decisions);
        }, nb::arg("edge_edits") = nb::list());
    nb::class_<vo::GeometryView>(m, "GeometryView")
        .def_prop_ro("revision", [](const vo::GeometryView& self) { return geometryToPy(self.revision()); })
        .def_prop_ro("object_revision", &vo::GeometryView::objectRevision, nb::rv_policy::copy)
        .def("edit", &vo::GeometryView::edit);
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
    nb::class_<vo::PatchView, vo::GeometryView>(m, "PatchView")
        .def_prop_ro("rows", &vo::PatchView::rows)
        .def_prop_ro("columns", &vo::PatchView::columns)
        .def_prop_ro("scale", &vo::PatchView::scale)
        .def("vertex", &vo::PatchView::vertex, nb::arg("row"), nb::arg("column"), nb::call_guard<nb::gil_scoped_release>())
        .def("valid", &vo::PatchView::valid, nb::arg("row"), nb::arg("column"), nb::call_guard<nb::gil_scoped_release>())
        .def("read_tile", &vo::PatchView::readTile, nb::arg("row"), nb::arg("column"),
             nb::arg("rows"), nb::arg("columns"), nb::call_guard<nb::gil_scoped_release>())
        .def("load", &vo::PatchView::load, nb::call_guard<nb::gil_scoped_release>());
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
        .def("patch", &vo::GeometrySnapshot::patch, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>())
        .def("geometry", &vo::GeometrySnapshot::geometry, nb::arg("object_uuid"), nb::call_guard<nb::gil_scoped_release>());
}
} // namespace vc::python
