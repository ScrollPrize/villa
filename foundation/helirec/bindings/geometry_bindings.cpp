// Python bindings of helirec/geometry.hpp: the detector's map, the poses of every frame from a geometry
// calibration and the reconstruction's per-projection terms (python/helirec/geometry.py and mosaic.py call these;
// a calibration crosses as the JSON text it is saved as).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "helirec/geometry.hpp"
#include "ndarray_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using helirec::py::new_array_t;
namespace hg = helirec::geometry;

namespace {

using DoubleIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::vector<double> vec(const DoubleIn& a) { return std::vector<double>(a.data(), a.data() + a.shape(0)); }

template <class T>
nb::ndarray<nb::numpy, T> arr(const std::vector<T>& v) {
    auto a = new_array_t<T>({v.size()});
    std::copy(v.begin(), v.end(), static_cast<T*>(a.data()));
    return a;
}

nb::dict poses_dict(const hg::Poses& p) {
    nb::dict d;
    d["row"] = arr(p.row), d["row_per_col"] = arr(p.row_per_col), d["axis"] = arr(p.axis), d["col_per_row"] = arr(p.col_per_row);
    d["col_scale"] = arr(p.col_scale), d["row_scale"] = arr(p.row_scale), d["angle_deg"] = arr(p.angle_deg), d["angle_deg_per_row"] = arr(p.angle_deg_per_row);
    d["scale_col"] = p.scale_col;
    return d;
}

nb::dict terms_dict(const hg::Terms& t) {
    nb::dict d;
    d["angles_deg"] = arr(t.angles_deg);
    d["z_translation_scale"] = t.z_translation_scale;
    d["row_offset"] = arr(t.row_offset), d["row_slope"] = arr(t.row_slope), d["slope_col"] = t.slope_col;
    d["col_offset"] = arr(t.col_offset), d["col_slope"] = arr(t.col_slope), d["slope_row"] = t.slope_row;
    d["col_scale"] = arr(t.col_scale), d["scale_col"] = t.scale_col, d["row_scale"] = arr(t.row_scale), d["scale_row"] = t.scale_row;
    return d;
}

nb::dict map_dict(const hg::Map& m) {
    nb::dict d;
    d["c0"] = m.c0, d["cs"] = m.cs, d["v0"] = m.v0, d["vs"] = m.vs;
    d["i"] = m.i, d["j"] = m.j, d["eu"] = m.eu, d["ev"] = m.ev;
    return d;
}

hg::Map map_of(const std::vector<int>& i, const std::vector<int>& j, const std::vector<double>& eu, const std::vector<double>& ev, double c0, double cs,
               double v0, double vs) {
    hg::Map m;
    m.i = i, m.j = j, m.eu = eu, m.ev = ev, m.c0 = c0, m.cs = cs, m.v0 = v0, m.vs = vs;
    if (j.size() != i.size() || eu.size() != i.size() || ev.size() != i.size()) throw helirec::Error("the map's lists differ in length");
    return m;
}

}  // namespace

void bind_geometry(nb::module_& m) {
    m.def("geometry_map_terms", &hg::map_terms, "order"_a);
    m.def("geometry_empty_map", [](int rows, int cols, int order) { return map_dict(hg::empty_map(rows, cols, order)); }, "rows"_a, "cols"_a, "order"_a = 3);
    m.def("geometry_map_displacement",
          [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<double>& eu, const std::vector<double>& ev, double c0, double cs, double v0, double vs,
             DoubleIn c, DoubleIn v) {
              if (c.shape(0) != v.shape(0)) throw helirec::Error("map_displacement: the columns and the rows differ in length");
              const hg::Map map = map_of(i, j, eu, ev, c0, cs, v0, vs);
              auto u = new_array_t<double>({c.shape(0)}), w = new_array_t<double>({c.shape(0)});
              hg::map_displacement(map, c.data(), v.data(), c.shape(0), static_cast<double*>(u.data()), static_cast<double*>(w.data()));
              return nb::make_tuple(u, w);
          },
          "i"_a, "j"_a, "eu"_a, "ev"_a, "c0"_a, "cs"_a, "v0"_a, "vs"_a, "c"_a, "v"_a);
    m.def("geometry_overlap_centre", &hg::overlap_centre, "cols"_a, "move"_a);
    m.def("geometry_reconstruction_terms",
          [](DoubleIn row, DoubleIn row_per_col, DoubleIn axis, DoubleIn col_per_row, DoubleIn col_scale, DoubleIn row_scale, DoubleIn angle_deg, double scale_col,
             DoubleIn heights_rows, double row_sign, int rows, double rot_center, double column_offset, double z_readjustment, double row_origin) {
              hg::Poses p;
              p.row = vec(row), p.row_per_col = vec(row_per_col), p.axis = vec(axis), p.col_per_row = vec(col_per_row);
              p.col_scale = vec(col_scale), p.row_scale = vec(row_scale), p.angle_deg = vec(angle_deg), p.scale_col = scale_col;
              return terms_dict(hg::reconstruction_terms(p, vec(heights_rows), row_sign, rows, rot_center, column_offset, z_readjustment, row_origin));
          },
          "row"_a, "row_per_col"_a, "axis"_a, "col_per_row"_a, "col_scale"_a, "row_scale"_a, "angle_deg"_a, "scale_col"_a, "heights_rows"_a, "row_sign"_a, "rows"_a,
          "rot_center"_a, "column_offset"_a, "z_readjustment"_a = 0.0, "row_origin"_a = 0.0);

    nb::class_<hg::Calibration>(m, "GeometryCalibration", "A two-scan geometry calibration (helirec/geometry.hpp), from its JSON text.")
        .def("__init__", [](hg::Calibration* self, const std::string& json_text) { new (self) hg::Calibration(hg::Calibration::parse(json_text)); }, "json_text"_a)
        .def_prop_ro("rows", &hg::Calibration::rows)
        .def_prop_ro("cols", &hg::Calibration::cols)
        .def_prop_ro("row_sign", &hg::Calibration::row_sign)
        .def_prop_ro("rolling_shutter", &hg::Calibration::rolling_shutter)
        .def_prop_ro("overlap_centre", &hg::Calibration::overlap_centre)
        .def("map", [](const hg::Calibration& c) { return map_dict(c.map()); })
        .def("frame_poses",
             [](const hg::Calibration& c, DoubleIn a0, DoubleIn a1, DoubleIn h0, DoubleIn h1) {
                 std::string not_applied;
                 std::array<hg::Poses, 2> p;
                 {
                     const std::array<std::vector<double>, 2> ang{vec(a0), vec(a1)}, H{vec(h0), vec(h1)};
                     nb::gil_scoped_release release;
                     p = c.frame_poses(ang, H, &not_applied);
                 }
                 nb::list poses;
                 poses.append(poses_dict(p[0]));
                 poses.append(poses_dict(p[1]));
                 return nb::make_tuple(poses, not_applied);
             },
             "angles_deg_0"_a, "angles_deg_1"_a, "heights_rows_0"_a, "heights_rows_1"_a,
             "(the poses of both scans, the terms measured and not applied as JSON text)")
        .def("merged_layout",
             [](const hg::Calibration& c, DoubleIn axis0, DoubleIn row0, DoubleIn axis1, DoubleIn row1, DoubleIn h0, DoubleIn h1, double quantum) {
                 std::array<hg::Poses, 2> p;
                 p[0].axis = vec(axis0), p[0].row = vec(row0), p[1].axis = vec(axis1), p[1].row = vec(row1);
                 const hg::MergedLayout l = c.merged_layout(p, {vec(h0), vec(h1)}, quantum);
                 nb::dict measured;
                 measured["axis_first_scan"] = l.axis_first_scan, measured["axis_second_scan"] = l.axis_second_scan;
                 measured["second_scan_rows"] = l.second_scan_rows, measured["quantum"] = l.quantum;
                 nb::dict d;
                 d["cor_main_detector"] = l.cor_main_detector;
                 d["vector"] = nb::make_tuple(l.vector[0], l.vector[1]);
                 d["measured"] = measured;
                 return d;
             },
             "axis_0"_a, "row_0"_a, "axis_1"_a, "row_1"_a, "heights_rows_0"_a, "heights_rows_1"_a, "quantum"_a = 0.25);

    nb::class_<hg::TileGeometry>(m, "TileGeometry", "The geometry of a mosaic's tiles (helirec/geometry.hpp), from its JSON text.")
        .def("__init__", [](hg::TileGeometry* self, const std::string& json_text) { new (self) hg::TileGeometry(hg::TileGeometry::parse(json_text)); }, "json_text"_a)
        .def("frame_pose",
             [](const hg::TileGeometry& g, const std::string& tile, DoubleIn angle_deg, DoubleIn frame, double row) {
                 const auto p = g.frame_pose(tile, vec(angle_deg), vec(frame), row);
                 return nb::make_tuple(arr(p.angle_deg), arr(p.column), arr(p.height_rows));
             },
             "tile"_a, "angle_deg"_a, "frame"_a, "row"_a = -1.0)
        .def("reconstruction_terms",
             [](const hg::TileGeometry& g, const std::string& tile, DoubleIn angle_deg, DoubleIn height_rows, DoubleIn frame, double rot_center, double row_origin) {
                 return terms_dict(g.reconstruction_terms(tile, vec(angle_deg), vec(height_rows), vec(frame), rot_center, row_origin));
             },
             "tile"_a, "angle_deg"_a, "height_rows"_a, "frame"_a, "rot_center"_a, "row_origin"_a = 0.0);
}
