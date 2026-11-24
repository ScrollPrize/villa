// Python bindings of vc_tomography/geometry.hpp: the detector's map, the poses of every frame from a geometry
// calibration and the reconstruction's per-projection terms (python/vc_tomography/geometry.py and mosaic.py call these;
// a calibration crosses as the JSON text it is saved as).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vc_tomography/geometry.hpp"
#include "conversions.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using vc_tomography::py::new_array_t;
namespace hg = vc_tomography::geometry;

namespace {

using DoubleIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::vector<double> vec(const DoubleIn& a) { return std::vector<double>(a.data(), a.data() + a.shape(0)); }

nb::dict poses_dict(const hg::Poses& p) {
    nb::dict d;
    d["row"] = vc_tomography::py::to_numpy(p.row), d["row_per_col"] = vc_tomography::py::to_numpy(p.row_per_col), d["axis"] = vc_tomography::py::to_numpy(p.axis), d["col_per_row"] = vc_tomography::py::to_numpy(p.col_per_row);
    d["col_scale"] = vc_tomography::py::to_numpy(p.col_scale), d["row_scale"] = vc_tomography::py::to_numpy(p.row_scale), d["angle_deg"] = vc_tomography::py::to_numpy(p.angle_deg), d["angle_deg_per_row"] = vc_tomography::py::to_numpy(p.angle_deg_per_row);
    d["scale_col"] = p.scale_col;
    d["rows_from_middle"] = p.rows_from_middle;
    return d;
}

using vc_tomography::py::terms_dict;

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
    if (j.size() != i.size() || eu.size() != i.size() || ev.size() != i.size()) throw vc_tomography::Error("the map's lists differ in length");
    return m;
}

}  // namespace

void bind_geometry(nb::module_& m) {
    m.def("geometry_map_terms", &hg::map_terms, "order"_a);
    m.def("geometry_empty_map", [](int rows, int cols, int order) { return map_dict(hg::empty_map(rows, cols, order)); }, "rows"_a, "cols"_a, "order"_a = 3);
    m.def("geometry_map_displacement",
          [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<double>& eu, const std::vector<double>& ev, double c0, double cs, double v0, double vs,
             DoubleIn c, DoubleIn v) {
              if (c.shape(0) != v.shape(0)) throw vc_tomography::Error("map_displacement: the columns and the rows differ in length");
              const hg::Map map = map_of(i, j, eu, ev, c0, cs, v0, vs);
              auto u = new_array_t<double>({c.shape(0)}), w = new_array_t<double>({c.shape(0)});
              hg::map_displacement(map, c.data(), v.data(), c.shape(0), static_cast<double*>(u.data()), static_cast<double*>(w.data()));
              return nb::make_tuple(u, w);
          },
          "i"_a, "j"_a, "eu"_a, "ev"_a, "c0"_a, "cs"_a, "v0"_a, "vs"_a, "c"_a, "v"_a);
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

    nb::class_<hg::Calibration>(m, "GeometryCalibration",
                                "A geometry calibration of scans that share their place on the rotation stage (vc_tomography/geometry.hpp), from its JSON text.")
        .def("__init__", [](hg::Calibration* self, const std::string& json_text) { new (self) hg::Calibration(hg::Calibration::parse(json_text)); }, "json_text"_a)
        .def_prop_ro("rows", &hg::Calibration::rows)
        .def_prop_ro("cols", &hg::Calibration::cols)
        .def_prop_ro("row_sign", &hg::Calibration::row_sign)
        .def_prop_ro("rolling_shutter", &hg::Calibration::rolling_shutter)
        .def_prop_ro("rows_from_middle", &hg::Calibration::rows_from_middle)
        .def_prop_ro("overlap_centre", &hg::Calibration::overlap_centre)
        .def("map", [](const hg::Calibration& c) { return map_dict(c.map()); })
        .def_prop_ro("scan_count", &hg::Calibration::scan_count)
        .def("scan_poses",
             [](const hg::Calibration& c, const std::vector<std::vector<double>>& angles_deg, const std::vector<std::vector<double>>& heights_rows) {
                 std::string not_applied;
                 std::vector<hg::Poses> p;
                 {
                     nb::gil_scoped_release release;
                     p = c.scan_poses(angles_deg, heights_rows, &not_applied);
                 }
                 nb::list poses;
                 for (const hg::Poses& q : p) poses.append(poses_dict(q));
                 return nb::make_tuple(poses, not_applied);
             },
             "angles_deg"_a, "heights_rows"_a, "a joint calibration of any number of scans: (the poses of every scan, the terms measured and not applied as JSON text)")
        .def("frames_placed",
             [](const hg::Calibration& c, std::size_t scan) {
                 nb::list runs;
                 for (const auto& r : c.frames_placed(scan)) runs.append(nb::make_tuple(r[0], r[1]));
                 return runs;
             },
             "scan"_a, "[first, end) of each run of a scan's projection frames the calibration was made on")
        .def("merged_layout",
             [](const hg::Calibration& c, const std::vector<std::vector<double>>& axis, const std::vector<std::vector<double>>& row, const std::vector<std::vector<double>>& heights_rows,
                const std::vector<std::size_t>& first_frame, double quantum) {
                 if (row.size() != axis.size()) throw vc_tomography::Error("merged_layout: the axes and the rows are not of the same scans");
                 std::vector<hg::Poses> p(axis.size());
                 for (std::size_t s = 0; s < p.size(); ++s) p[s].axis = axis[s], p[s].row = row[s];
                 const hg::MergedLayout l = c.merged_layout(p, heights_rows, first_frame, quantum);
                 nb::list vectors, measured;
                 for (const auto& v : l.vectors) vectors.append(nb::make_tuple(v[0], v[1]));
                 for (const auto& v : l.measured) measured.append(nb::make_tuple(v[0], v[1]));
                 nb::dict d;
                 d["cor_main_detector"] = l.cor_main_detector;
                 d["vectors"] = vectors;
                 d["measured"] = measured;
                 d["axis_first_scan"] = l.axis_first_scan;
                 d["quantum"] = l.quantum;
                 return d;
             },
             "axis"_a, "row"_a, "heights_rows"_a, "first_frame"_a = std::vector<std::size_t>(), "quantum"_a = 0.25,
             "the merged frame of any number of scans: the axis in the first scan's columns, every other scan's (rows, columns) from it, and per other scan its "
             "(axis column, rows) unrounded");
}
