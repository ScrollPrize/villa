// Python bindings of helirec/calibration/geometry_calibration.hpp: the geometry calibration of a two-scan
// acquisition and its stages. Layouts, comparisons and results cross as JSON text (python/helirec/geometry.py
// turns them into the dicts its callers know).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>

#include "../src/io/json.hpp"
#include "helirec/calibration/geometry_calibration.hpp"
#include "ndarray_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
namespace gc = helirec::calibration::geometry_calibration;

namespace {

using DoubleIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

const Json& member(const Json& j, const char* key) {
    const Json* v = j.find(key);
    if (!v) throw Error(std::string("geometry calibration: no \"") + key + "\" given");
    return *v;
}
double number(const Json& j, const char* key) { return member(j, key).number; }
double number(const Json& j, const char* key, double fallback) {
    const Json* v = j.find(key);
    return v && v->is(Json::Number) ? v->number : fallback;
}

// {rows, cols, per_turn, pixel_um, stretches: [{rest1, helix, rest2}, ..], lengths: [..], row_sign, axis, move,
//  width, heights: {"<scan> <stretch>": [rows per frame]}}
gc::Layout layout_of(const std::string& text) {
    const Json j = Json::parse(text);
    gc::Layout lay;
    lay.rows = static_cast<int>(number(j, "rows")), lay.cols = static_cast<int>(number(j, "cols")), lay.per_turn = static_cast<int>(number(j, "per_turn"));
    lay.pixel_um = number(j, "pixel_um", 0.0);
    lay.row_sign = number(j, "row_sign", -1.0), lay.axis = number(j, "axis", 0.0);
    lay.move = static_cast<int>(number(j, "move", 0.0)), lay.width = static_cast<int>(number(j, "width", 1024.0));
    const Json& st = member(j, "stretches");
    const Json& len = member(j, "lengths");
    for (std::size_t k = 0; k < 2; ++k)
        for (int a = 0; a < 3; ++a) {
            lay.stretch[k][a] = static_cast<int>(number(st.array.at(k), gc::kNames[a]));
            lay.length[k][a] = static_cast<std::int64_t>(number(len.array.at(k), gc::kNames[a]));
        }
    if (const Json* h = j.find("heights"))
        for (const auto& [key, values] : h->object) {
            const int scan = key.at(0) - '0';
            std::vector<double>& out = lay.heights[scan][gc::name_index(key.substr(2))];
            for (const Json& v : values.array) out.push_back(v.number);
        }
    return lay;
}

geometry::Map map_of(const std::string& text) {
    const Json j = Json::parse(text);
    geometry::Map m;
    for (const Json& v : member(j, "i").array) m.i.push_back(static_cast<int>(v.number));
    for (const Json& v : member(j, "j").array) m.j.push_back(static_cast<int>(v.number));
    for (const Json& v : member(j, "eu").array) m.eu.push_back(v.number);
    for (const Json& v : member(j, "ev").array) m.ev.push_back(v.number);
    m.c0 = number(j, "c0"), m.cs = number(j, "cs"), m.v0 = number(j, "v0"), m.vs = number(j, "vs");
    return m;
}

Json whole(long long v) { return Json::whole_number(v); }

std::string plan_json(const std::string& layout) {
    Json out = Json::list();
    for (const gc::Planned& p : gc::plan(layout_of(layout))) {
        const calibration::ViewRequest& q = p.request;
        Json d = Json::dict();
        d.set("label", Json::of(q.label)).set("scan_a", whole(q.scan_a)).set("scan_b", whole(q.scan_b)).set("stretch_a", whole(q.stretch_a)).set("stretch_b", whole(q.stretch_b));
        d.set("frame_a", whole(q.frame_a)).set("frame_hint", whole(q.frame_hint)).set("angle_offset_deg", Json::of(q.angle_offset_deg)).set("mirror", Json::of_bool(q.mirror));
        d.set("frames", whole(q.frames)).set("rows", whole(q.rows)).set("cols", whole(q.cols)).set("row_a", whole(q.row_a)).set("col_a", whole(q.col_a)).set("col_b", whole(q.col_b));
        Json meta = Json::dict();
        meta.set("kind", Json::of(p.kind)).set("scan_a", whole(p.scan_a)).set("scan_b", whole(p.scan_b)).set("stretch_a", Json::of(std::string(gc::kNames[p.a])));
        meta.set("stretch_b", Json::of(std::string(gc::kNames[p.b])));
        out.add(Json::list().add(d).add(meta));
    }
    return out.text();
}

// rows: [{kind, scan_a, stretch_a (name), frame_a, row_a, rows, row_offset, col_a, col_b, cols, stated_dq, dq,
//         stated_D, D, du, slope_a, slope_b}, ..]
std::string fit_json(const std::string& rows_text, const std::string& layout, const std::string& map, double tau, double period, double huber, int iterations) {
    const Json rows_j = Json::parse(rows_text);
    std::vector<gc::Row> rows;
    for (const Json& r : rows_j.array) {
        gc::Row row;
        row.kind = member(r, "kind").string;
        row.scan_a = static_cast<int>(number(r, "scan_a", 0.0));
        row.a = gc::name_index(member(r, "stretch_a").string);
        calibration::ViewMeasurement& m = row.m;
        m.frame_a = static_cast<std::int64_t>(number(r, "frame_a"));
        m.row_a = static_cast<int>(number(r, "row_a")), m.rows = static_cast<int>(number(r, "rows")), m.row_offset = static_cast<int>(number(r, "row_offset"));
        m.col_a = static_cast<int>(number(r, "col_a")), m.col_b = static_cast<int>(number(r, "col_b")), m.cols = static_cast<int>(number(r, "cols"));
        m.stated_dq = number(r, "stated_dq"), m.dq = number(r, "dq"), m.stated_D = number(r, "stated_D"), m.D = number(r, "D"), m.du = number(r, "du");
        m.slope_a = number(r, "slope_a"), m.slope_b = number(r, "slope_b");
        rows.push_back(row);
    }
    gc::Fit f;
    {
        const gc::Layout lay = layout_of(layout);
        const geometry::Map m = map_of(map);
        nb::gil_scoped_release nogil;
        f = gc::fit(rows, lay, m, tau, period, huber, iterations);
    }
    const auto list = [](const std::vector<double>& v) {
        Json j = Json::list();
        for (double x : v) j.add(Json::of(x));
        return j;
    };
    Json rho = Json::dict(), lines = Json::dict(), left = Json::dict(), groups = Json::list();
    for (const int a : {gc::Helix, gc::Rest1, gc::Rest2}) {
        if (const auto it = f.rho.find(a); it != f.rho.end()) rho.set(gc::kNames[a], Json::of(it->second));
        if (const auto it = f.lines.find(a); it != f.lines.end()) lines.set(gc::kNames[a], list({it->second[0], it->second[1]}));
    }
    for (const char* kind : gc::kKinds)
        if (const auto it = f.left.find(kind); it != f.left.end())
            left.set(kind, Json::dict().set("n", whole(static_cast<long long>(it->second.n))).set("px", Json::of(it->second.px)).set("rows", Json::of(it->second.rows)));
    for (const auto& [key, uv] : f.groups)
        groups.add(Json::list().add(Json::of(key.kind)).add(whole(key.scan)).add(Json::of(std::string(gc::kNames[key.a]))).add(whole(key.frame)).add(Json::of(uv[0])).add(Json::of(uv[1])));
    Json out = Json::dict();
    out.set("eu", list(f.eu)).set("ev", list(f.ev)).set("psi", Json::of(f.psi)).set("rho", rho).set("lines", lines).set("n", whole(static_cast<long long>(f.n)));
    out.set("left", left).set("groups", groups);
    return out.text();
}

}  // namespace

void bind_geometry_calibration(nb::module_& m) {
    m.def(
        "calibrate_geometry",
        [](const std::vector<std::string>& scans, std::optional<double> row_time, const std::string& device, int order, int passes, double tolerance, int threads, bool motion,
           int patch, std::optional<std::function<void(const std::string&)>> log) {
            gc::Options opt;
            opt.row_time = row_time ? *row_time : std::nan("");
            if (device != "cpu" && device != "cuda" && device != "auto") throw Error("device is cpu, cuda or auto");
            opt.device = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
            opt.order = order, opt.passes = passes, opt.tolerance = tolerance, opt.threads = threads, opt.motion = motion, opt.patch = patch;
            if (log)
                opt.log = [&](const std::string& line) {
                    nb::gil_scoped_acquire gil;
                    (*log)(line);
                };
            nb::gil_scoped_release nogil;
            return gc::calibrate_geometry(scans, opt);
        },
        "scans"_a, "row_time"_a = nb::none(), "device"_a = "auto", "order"_a = 3, "passes"_a = 6, "tolerance"_a = 0.02, "threads"_a = 8, "motion"_a = true, "patch"_a = 0,
        "log"_a = nb::none(),
        "The geometry calibration of a two-scan helical acquisition, as JSON text\n"
        "(helirec::calibration::geometry_calibration::calibrate_geometry; helirec-calibration geometry).");
    m.def("geometry_calibration_save", &gc::save_text, "text"_a, "path"_a, "The text written whole or not at all.");
    m.def("geometry_calibration_plan", &plan_json, "layout"_a, "The comparisons of one pass: [[request, what it compares], ..] as JSON text.");
    m.def("geometry_calibration_fit", &fit_json, "rows"_a, "layout"_a, "map"_a, "tau"_a, "period"_a, "huber"_a = 0.03, "iterations"_a = 8,
          "What the comparisons leave, as a correction of the map and the poses beside it (JSON text).");
    m.def("geometry_calibration_update_range",
          [](const std::string& map, const std::vector<double>& eu, const std::vector<double>& ev) {
              const std::array<double, 2> r = gc::update_range(map_of(map), eu, ev);
              return nb::make_tuple(r[0], r[1]);
          },
          "map"_a, "eu"_a, "ev"_a);
    m.def("geometry_calibration_nodes",
          [](DoubleIn t1, DoubleIn t2, DoubleIn th, int n_extra, double h, double total) {
              const std::size_t n = th.shape(0);
              if (t1.shape(0) != n || t2.shape(0) != n) throw Error("geometry_calibration_nodes: the three series differ in length");
              int K = 0;
              const std::vector<double> X = gc::nodes_design(std::vector<double>(t1.data(), t1.data() + n), std::vector<double>(t2.data(), t2.data() + n),
                                                             std::vector<double>(th.data(), th.data() + n), n_extra, h, total, K);
              auto out = helirec::py::new_array_t<double>({n, static_cast<std::size_t>(2 * K + n_extra)});
              std::copy(X.begin(), X.end(), static_cast<double*>(out.data()));
              return nb::make_tuple(out, K);
          },
          "t1"_a, "t2"_a, "th"_a, "n_extra"_a, "h"_a, "total"_a);
    m.def("geometry_calibration_block_size", &gc::block_size, "limit"_a);
    m.def("geometry_calibration_second_scan_fit",
          [](const std::vector<std::string>& stretch, DoubleIn t, DoubleIn th, DoubleIn v, DoubleIn du, DoubleIn y, DoubleIn q, int rows, double total) {
              std::vector<gc::PairRow> table(stretch.size());
              for (std::size_t i = 0; i < table.size(); ++i)
                  table[i] = gc::PairRow{gc::name_index(stretch[i]), t.data()[i], th.data()[i], v.data()[i], du.data()[i], y.data()[i], q.data()[i]};
              return gc::second_scan_fit(table, rows, total);
          },
          "stretch"_a, "t"_a, "th"_a, "v"_a, "du"_a, "y"_a, "q"_a, "rows"_a, "total"_a);
    m.def("geometry_calibration_fit_refill_shots",
          [](nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu> x, double T, double E, int pre) -> nb::object {
              std::optional<gc::RefillFit> f;
              {
                  const std::vector<double> levels(x.data(), x.data() + x.shape(0) * x.shape(1));
                  const std::size_t frames = x.shape(0);
                  const int n_rows = static_cast<int>(x.shape(1));
                  nb::gil_scoped_release nogil;
                  f = gc::fit_refill_shots(levels, frames, n_rows, T, E, pre);
              }
              if (!f) return nb::none();
              nb::dict d;
              d["p"] = f->p, d["se"] = f->se, d["noise"] = f->noise, d["cost"] = f->cost, d["shots"] = f->shots;
              return d;
          },
          "x"_a, "T"_a, "E"_a, "pre"_a,
          "The time of a row from the beam dips of a refill: x (frames, rows) the logarithm of each row's level; None if fewer than 12 shots are found.");
}
