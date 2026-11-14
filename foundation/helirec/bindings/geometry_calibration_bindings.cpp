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

helirec::calibration::Session& bound_session(nb::handle h);  // bindings/calibration_bindings.cpp

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

geometry::Map map_of(const Json& j) {
    geometry::Map m;
    for (const Json& v : member(j, "i").array) m.i.push_back(static_cast<int>(v.number));
    for (const Json& v : member(j, "j").array) m.j.push_back(static_cast<int>(v.number));
    for (const Json& v : member(j, "eu").array) m.eu.push_back(v.number);
    for (const Json& v : member(j, "ev").array) m.ev.push_back(v.number);
    m.c0 = number(j, "c0"), m.cs = number(j, "cs"), m.v0 = number(j, "v0"), m.vs = number(j, "vs");
    return m;
}
geometry::Map map_of(const std::string& text) { return map_of(Json::parse(text)); }

Json whole(long long v) { return Json::whole_number(v); }

using Table = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// {rows, cols, per_turn, pixel_um, row_sign, row_time, fan, width, anchor, period_s, scans: [{name, mid_height, axis,
//  moved_on_stage, on_stage: [px, px], stretches: [{name, index, helix, first, end, angle_deg: [..], rows: [..]}]}]}; what is left out
//  keeps its default (a stretch's frames on disk: all of them)
gc::Acquisition acquisition_of(const std::string& text) {
    const Json j = Json::parse(text);
    gc::Acquisition acq;
    acq.rows = static_cast<int>(number(j, "rows")), acq.cols = static_cast<int>(number(j, "cols"));
    acq.per_turn = static_cast<int>(number(j, "per_turn", 0.0)), acq.pixel_um = number(j, "pixel_um", 0.0);
    acq.row_sign = number(j, "row_sign", -1.0), acq.row_time = number(j, "row_time", 0.0), acq.fan = number(j, "fan", 0.0), acq.per_depth = number(j, "per_depth", 0.0);
    acq.width = static_cast<int>(number(j, "width", 512.0));
    acq.anchor = static_cast<int>(number(j, "anchor", 0.0)), acq.period_s = number(j, "period_s", 0.0);
    for (const Json& sj : member(j, "scans").array) {
        gc::ScanLayout sc;
        if (const Json* n = sj.find("name")) sc.name = n->string;
        sc.mid_height = number(sj, "mid_height", 0.0);
        sc.axis = number(sj, "axis", std::nan(""));
        if (const Json* moved = sj.find("moved_on_stage")) sc.moved_on_stage = moved->boolean;
        if (const Json* at = sj.find("on_stage"); at && at->array.size() == 2) sc.on_stage[0] = at->array[0].number, sc.on_stage[1] = at->array[1].number;
        for (const Json& tj : member(sj, "stretches").array) {
            gc::StretchLayout st;
            st.name = member(tj, "name").string;
            st.index = static_cast<int>(number(tj, "index", 0.0));
            if (const Json* h = tj.find("helix")) st.helix = h->boolean;
            for (const Json& v : member(tj, "angle_deg").array) st.course.angle_deg.push_back(v.number);
            for (const Json& v : member(tj, "rows").array) st.course.rows.push_back(v.number);
            if (st.course.angle_deg.size() != st.course.rows.size()) throw Error("geometry: a stretch's angles and heights differ in length");
            st.first = static_cast<std::int64_t>(number(tj, "first", 0.0));
            st.end = static_cast<std::int64_t>(number(tj, "end", static_cast<double>(st.course.rows.size())));
            st.offset = sc.stretches.empty() ? 0 : sc.stretches.back().offset + static_cast<std::int64_t>(sc.stretches.back().course.rows.size());
            if (st.first < 0 || st.end > static_cast<std::int64_t>(st.course.rows.size()) || st.end < st.first) throw Error("geometry: a stretch's frames on disk are not among its frames");
            sc.stretches.push_back(std::move(st));
        }
        acq.scans.push_back(std::move(sc));
    }
    return acq;
}

// as solution_json writes them: {roll, stage: {height_once: [cos, sin], axis_twice: [cos, sin]}, scans: [{axis,
//  axis_slope, axial_scale, d: [d0, d1], height, height_once: [cos, sin], rotation: [x, y, z], stretch_height: {name: rows}}],
//  about_height, map: {i, j, eu, ev, c0, cs, v0, vs}}
gc::Parameters parameters_of(const gc::Acquisition& acq, const std::string& text) {
    gc::Parameters p = gc::empty_parameters(acq);
    if (text.empty()) return p;
    const Json j = Json::parse(text);
    p.roll = number(j, "roll", 0.0);
    p.pitch = number(j, "pitch", 0.0);
    p.about_height = number(j, "about_height", 0.0);
    if (const Json* map = j.find("map"); map && map->is(Json::Object)) p.map = map_of(*map);
    if (const Json* st = j.find("stage")) {
        const auto pair = [&](const char* key, double* out) {
            if (const Json* v = st->find(key))
                for (std::size_t i = 0; i < 2 && i < v->array.size(); ++i) out[i] = v->array[i].number;
        };
        pair("height_once", p.stage.height_once), pair("axis_twice", p.stage.axis_twice);
    }
    const Json& scans = member(j, "scans");
    if (scans.array.size() != acq.scans.size()) throw Error("geometry: the parameters and the acquisition differ in their scans");
    for (std::size_t k = 0; k < scans.array.size(); ++k) {
        const Json& sj = scans.array[k];
        gc::ScanParameters& q = p.scans[k];
        q.axis = number(sj, "axis", 0.0), q.axis_slope = number(sj, "axis_slope", 0.0), q.axial_scale = number(sj, "axial_scale", 0.0);
        q.height = number(sj, "height", 0.0);
        if (const Json* d = sj.find("d"))
            for (std::size_t i = 0; i < 2 && i < d->array.size(); ++i) q.d[i] = d->array[i].number;
        if (const Json* d = sj.find("height_once"))
            for (std::size_t i = 0; i < 2 && i < d->array.size(); ++i) q.height_once[i] = d->array[i].number;
        if (const Json* d = sj.find("rotation"))
            for (std::size_t i = 0; i < 3 && i < d->array.size(); ++i) q.rotation[i] = d->array[i].number;
        if (const Json* h = sj.find("stretch_height"))
            for (std::size_t a = 0; a < acq.scans[k].stretches.size(); ++a) q.stretch_height[a] = number(*h, acq.scans[k].stretches[a].name.c_str(), 0.0);
    }
    return p;
}

// a row of a table of sightings: scan, stretch, frame, row, column
gc::Sighting sighting_of(const double* r) { return {static_cast<int>(r[0]), static_cast<int>(r[1]), r[2], r[3], r[4]}; }

// rows of a table of lens patches: scan a, scan b, the place (x, y, z), what was measured (x, y, z), its sigma (x, y, z)
std::vector<gc::LensPatch> lens_of(const Table& t) {
    if (t.shape(0) && t.shape(1) != 11) throw Error("geometry: a lens patch is scan a, scan b, its place (x, y, z), the displacement (x, y, z) and its sigma (x, y, z)");
    std::vector<gc::LensPatch> out(t.shape(0));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double* r = t.data() + 11 * i;
        out[i].scan_a = static_cast<int>(r[0]), out[i].scan_b = static_cast<int>(r[1]);
        for (int j = 0; j < 3; ++j) out[i].p[j] = r[2 + j], out[i].moved[j] = r[5 + j], out[i].sigma[j] = r[8 + j];
    }
    return out;
}

// a plan as JSON text: per block its kind, scans and stretches, and the request's fields
std::string plan_text(const std::vector<gc::Planned>& planned) {
    Json out = Json::list();
    for (const gc::Planned& p : planned) {
        const calibration::ViewRequest& q = p.request;
        Json d = Json::dict();
        d.set("kind", Json::of(p.kind)).set("scan_a", whole(p.scan_a)).set("scan_b", whole(p.scan_b)).set("a", whole(p.a)).set("b", whole(p.b)).set("from_block", Json::of_bool(p.from_block));
        d.set("request_scan_a", whole(q.scan_a)).set("request_scan_b", whole(q.scan_b));
        d.set("stretch_a", whole(q.stretch_a)).set("stretch_b", whole(q.stretch_b)).set("frame_a", whole(q.frame_a)).set("frame_hint", whole(q.frame_hint));
        d.set("angle_offset_deg", Json::of(q.angle_offset_deg)).set("mirror", Json::of_bool(q.mirror)).set("frames", whole(q.frames)).set("rows", whole(q.rows)).set("cols", whole(q.cols));
        d.set("row_a", whole(q.row_a)).set("col_a", whole(q.col_a)).set("col_b", whole(q.col_b)).set("frame_shear", Json::of(q.frame_shear));
        d.set("sway_px", Json::of(q.sway_px)).set("sway_phase_deg", Json::of(q.sway_phase_deg)).set("sway_row_deg", Json::of(q.sway_row_deg));
        out.add(d);
    }
    return out.text();
}

// the accepted comparisons of a measured plan, as Python takes them
nb::dict measured(const std::vector<gc::Comparison>& got, const std::vector<int>& kept, std::size_t blocks) {
    const std::size_t n = got.size();
    auto a = helirec::py::new_array_t<double>({n, std::size_t(5)});
    auto b = helirec::py::new_array_t<double>({n, std::size_t(5)});
    auto correlation = helirec::py::new_array_t<double>({n});
    auto curvature = helirec::py::new_array_t<double>({n, std::size_t(6)});
    nb::list kinds, opposite, block;
    for (std::size_t i = 0; i < n; ++i) {
        const gc::Comparison& c = got[i];
        const double ra[5] = {static_cast<double>(c.a.scan), static_cast<double>(c.a.stretch), c.a.frame, c.a.row, c.a.col};
        const double rb[5] = {static_cast<double>(c.b.scan), static_cast<double>(c.b.stretch), c.b.frame, c.b.row, c.b.col};
        std::copy(ra, ra + 5, static_cast<double*>(a.data()) + 5 * i);
        std::copy(rb, rb + 5, static_cast<double*>(b.data()) + 5 * i);
        static_cast<double*>(correlation.data())[i] = c.correlation;
        std::copy(c.curvature, c.curvature + 6, static_cast<double*>(curvature.data()) + 6 * i);
        kinds.append(c.kind), opposite.append(c.opposite), block.append(kept[i]);
    }
    nb::dict out;
    out["kinds"] = kinds, out["a"] = a, out["b"] = b, out["opposite"] = opposite, out["correlation"] = correlation, out["curvature"] = curvature;
    out["block"] = block, out["blocks"] = blocks;
    return out;
}

// pairs of scans, as (n, 2)
std::vector<std::array<int, 2>> pairs_of(const std::vector<std::vector<int>>& given) {
    std::vector<std::array<int, 2>> out;
    for (const std::vector<int>& p : given) {
        if (p.size() != 2) throw Error("geometry: a pair of scans is two indices");
        out.push_back({p[0], p[1]});
    }
    return out;
}

gc::PlanCounts tile_counts(int positions, int angles, int bands, int windows) {
    gc::PlanCounts n;
    n.tile_positions = positions, n.tile_angles = angles, n.tile_bands = bands, n.tile_windows = windows;
    return n;
}

}  // namespace

void bind_geometry_calibration(nb::module_& m) {
    nb::class_<gc::Acquisition>(m, "GeometryAcquisition",
                                "The layout of the scans of one acquisition as the joint geometry model takes it: what the records state per frame, the\n"
                                "frames on disk, the sense of the rows, a coarse axis column per scan (helirec::calibration::geometry_calibration::Acquisition).\n"
                                "Made from JSON text (see python/helirec/geometry.py) or read from a session (geometry_read_acquisition).")
        .def("__init__", [](gc::Acquisition* self, const std::string& text) { new (self) gc::Acquisition(acquisition_of(text)); }, "text"_a)
        .def_rw("row_time", &gc::Acquisition::row_time, "The rows' timing in frames per stored row (signed).")
        .def_rw("per_depth", &gc::Acquisition::per_depth, "What the beam's divergence magnifies a sample by per pixel it sits nearer the source (pixel / source distance); 0: left out.")
        .def_prop_ro("summary", [](const gc::Acquisition& a) { return gc::acquisition_json(a); }, "The layout without the per-frame series, as JSON text.")
        .def("course",
             [](const gc::Acquisition& a, int scan, int stretch) {
                 const calibration::Trajectory& t = a.scans.at(static_cast<std::size_t>(scan)).stretches.at(static_cast<std::size_t>(stretch)).course;
                 auto angle = helirec::py::new_array_t<double>({t.angle_deg.size()});
                 auto rows = helirec::py::new_array_t<double>({t.rows.size()});
                 std::copy(t.angle_deg.begin(), t.angle_deg.end(), static_cast<double*>(angle.data()));
                 std::copy(t.rows.begin(), t.rows.end(), static_cast<double*>(rows.data()));
                 return nb::make_tuple(angle, rows);
             },
             "scan"_a, "stretch"_a, "The record's angle [deg, cumulative] and height [rows] of every frame of a stretch.");
    m.def(
        "geometry_read_acquisition",
        [](nb::handle session, const std::string& device, std::optional<std::function<void(const std::string&)>> log) {
            calibration::Session& s = bound_session(session);
            if (device != "cpu" && device != "cuda" && device != "auto") throw Error("device is cpu, cuda or auto");
            const Device dev = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
            gc::Log say;
            if (log)
                say = [&](const std::string& line) {
                    nb::gil_scoped_acquire gil;
                    (*log)(line);
                };
            nb::gil_scoped_release nogil;
            return gc::read_acquisition(s, dev, say);
        },
        "session"_a, "device"_a = "auto", "log"_a = nb::none(),
        "The layout of a CalibrationSession's scans, each with whatever projection stretches it has on disk\n"
        "(helirec::calibration::geometry_calibration::read_acquisition).");
    m.def(
        "geometry_row_time_from_refill",
        [](nb::handle session, const gc::Acquisition& acq, std::optional<std::function<void(const std::string&)>> log) {
            calibration::Session& s = bound_session(session);
            gc::Log say;
            if (log)
                say = [&](const std::string& line) {
                    nb::gil_scoped_acquire gil;
                    (*log)(line);
                };
            nb::gil_scoped_release nogil;
            return gc::row_time_from_refill(s, acq, say);
        },
        "session"_a, "acquisition"_a, "log"_a = nb::none(),
        "Seconds per stored row from a refill in one of the acquisition's stretches, as JSON text (\"null\": none holds one)\n"
        "(helirec::calibration::geometry_calibration::row_time_from_refill).");
    m.def(
        "geometry_plan",
        [](const gc::Acquisition& acq, int angles_per_turn, int max_positions, int n_rest, int n_turn) {
            gc::PlanCounts n;
            n.angles_per_turn = angles_per_turn, n.max_positions = max_positions, n.n_rest = n_rest, n.n_turn = n_turn;
            return plan_text(gc::plan(acq, n));
        },
        "acquisition"_a, "angles_per_turn"_a = 12, "max_positions"_a = 48, "n_rest"_a = 16, "n_turn"_a = 12,
        "The comparisons of each scan with itself, placed from the layout alone, as JSON text\n"
        "(helirec::calibration::geometry_calibration::plan).");
    m.def(
        "geometry_measure",
        [](nb::handle session, const gc::Acquisition& acq, const std::string& device, int patch, int angles_per_turn, int max_positions, int n_rest, int n_turn) {
            calibration::Session& s = bound_session(session);
            if (device != "cpu" && device != "cuda" && device != "auto") throw Error("device is cpu, cuda or auto");
            const Device dev = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
            std::vector<gc::Comparison> got;
            std::vector<int> kept;
            std::size_t blocks = 0;
            {
                nb::gil_scoped_release nogil;
                gc::PlanCounts counts;
                counts.angles_per_turn = angles_per_turn, counts.max_positions = max_positions, counts.n_rest = n_rest, counts.n_turn = n_turn;
                const std::vector<gc::Planned> planned = gc::plan(acq, counts);
                blocks = planned.size();
                got = gc::measure(s, acq, planned, dev, patch, &kept);
            }
            return measured(got, kept, blocks);
        },
        "session"_a, "acquisition"_a, "device"_a = "auto", "patch"_a = 128, "angles_per_turn"_a = 12, "max_positions"_a = 48, "n_rest"_a = 16, "n_turn"_a = 12,
        "The plan of each scan's comparisons with itself measured once, in patches: the accepted ones as kinds, sightings a and b\n"
        "(n, 5: scan, stretch, frame, row, column), opposite, correlation, curvature (n, 6) and the block each came from\n"
        "(helirec::calibration::geometry_calibration::plan, measure).");
    m.def(
        "geometry_plan_tiles",
        [](const gc::Acquisition& acq, const std::string& places, const std::vector<std::vector<int>>& pairs, int positions, int angles, int bands, int windows) {
            return plan_text(gc::plan_tiles(acq, parameters_of(acq, places), pairs_of(pairs), tile_counts(positions, angles, bands, windows)));
        },
        "acquisition"_a, "places"_a, "pairs"_a, "positions"_a = 48, "angles"_a = 24, "bands"_a = 3, "windows"_a = 4,
        "The comparisons of scans at different places on the rotation stage (a mosaic's tiles), placed from the parameters'\n"
        "axis columns, places and heights, as JSON text (helirec::calibration::geometry_calibration::plan_tiles).");
    m.def(
        "geometry_measure_tiles",
        [](nb::handle session, const gc::Acquisition& acq, const std::string& places, const std::vector<std::vector<int>>& pairs, const std::string& device, int patch, int positions, int angles,
           int bands, int windows) {
            calibration::Session& s = bound_session(session);
            if (device != "cpu" && device != "cuda" && device != "auto") throw Error("device is cpu, cuda or auto");
            const Device dev = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
            std::vector<gc::Comparison> got;
            std::vector<int> kept;
            std::size_t blocks = 0;
            {
                const gc::Parameters p = parameters_of(acq, places);
                const std::vector<std::array<int, 2>> which = pairs_of(pairs);
                nb::gil_scoped_release nogil;
                const std::vector<gc::Planned> planned = gc::plan_tiles(acq, p, which, tile_counts(positions, angles, bands, windows));
                blocks = planned.size();
                got = gc::measure(s, acq, planned, dev, patch, &kept);
            }
            return measured(got, kept, blocks);
        },
        "session"_a, "acquisition"_a, "places"_a, "pairs"_a, "device"_a = "auto", "patch"_a = 128, "positions"_a = 48, "angles"_a = 24, "bands"_a = 3, "windows"_a = 4,
        "The plan of the tiles' comparisons with each other (geometry_plan_tiles) measured once, in patches: as geometry_measure\n"
        "returns them (helirec::calibration::geometry_calibration::plan_tiles, measure).");
    m.def(
        "geometry_solve",
        [](const gc::Acquisition& acq, const std::vector<std::string>& kinds, Table a, Table b, const std::vector<bool>& opposite, Table lens_table, const std::string& start, double huber,
           int max_iterations, const std::vector<std::string>& held, const std::vector<std::string>& freed, double lens_huber) {
            const std::size_t n = kinds.size();
            if (a.shape(0) != n || b.shape(0) != n || opposite.size() != n || a.shape(1) != 5 || b.shape(1) != 5)
                throw Error("geometry_solve: one kind, two sightings (scan, stretch, frame, row, column) and one flag per comparison");
            std::vector<gc::Comparison> comparisons(n);
            for (std::size_t i = 0; i < n; ++i) {
                comparisons[i].kind = kinds[i];
                comparisons[i].a = sighting_of(a.data() + 5 * i), comparisons[i].b = sighting_of(b.data() + 5 * i);
                comparisons[i].opposite = opposite[i];
            }
            const std::vector<gc::LensPatch> lens = lens_of(lens_table);
            gc::GeometryOptions opt;
            opt.huber = huber, opt.max_iterations = max_iterations, opt.held = held, opt.freed = freed, opt.lens_huber = lens_huber;
            gc::GeometrySolution s;
            {
                const gc::Parameters p0 = parameters_of(acq, start);
                nb::gil_scoped_release nogil;
                s = gc::solve_geometry(acq, comparisons, lens, p0, opt);
            }
            auto residual = helirec::py::new_array_t<double>({n, std::size_t(3)});
            for (std::size_t i = 0; i < n; ++i) std::copy(s.residual[i].begin(), s.residual[i].end(), static_cast<double*>(residual.data()) + 3 * i);
            auto lens_residual = helirec::py::new_array_t<double>({lens.size(), std::size_t(3)});
            for (std::size_t i = 0; i < lens.size(); ++i) std::copy(s.lens_residual[i].begin(), s.lens_residual[i].end(), static_cast<double*>(lens_residual.data()) + 3 * i);
            return nb::make_tuple(gc::solution_json(acq, s), residual, lens_residual);
        },
        "acquisition"_a, "kinds"_a, "a"_a, "b"_a, "opposite"_a, "lens"_a, "start"_a = "", "huber"_a = 0.03, "max_iterations"_a = 100, "held"_a = std::vector<std::string>(),
        "freed"_a = std::vector<std::string>(), "lens_huber"_a = 1.5,
        "The places of the scans that best explain comparisons of their frames and lens patches of their volumes (lens: (n, 11)\n"
        "scan a, scan b, the place, the displacement, its sigma): (the solution as JSON text, the comparisons' residuals (n, 3):\n"
        "frames, rows, columns, the lens patches' (n, 3): x, y, z). (helirec::calibration::geometry_calibration::solve_geometry)");
    m.def(
        "geometry_lens_residuals",
        [](const gc::Acquisition& acq, const std::string& parameters, Table lens_table) {
            const std::vector<gc::LensPatch> lens = lens_of(lens_table);
            const std::vector<std::array<double, 3>> r = gc::lens_residuals(acq, parameters_of(acq, parameters), lens);
            auto out = helirec::py::new_array_t<double>({lens.size(), std::size_t(3)});
            for (std::size_t i = 0; i < lens.size(); ++i) std::copy(r[i].begin(), r[i].end(), static_cast<double*>(out.data()) + 3 * i);
            return out;
        },
        "acquisition"_a, "parameters"_a, "lens"_a,
        "What the parameters leave of each lens patch: (n, 3) x, y, z [px]. (helirec::calibration::geometry_calibration::lens_residuals)");
    m.def(
        "geometry_residuals",
        [](const gc::Acquisition& acq, const std::string& parameters, Table a, Table b, const std::vector<bool>& opposite) {
            const std::size_t n = opposite.size();
            if (a.shape(0) != n || b.shape(0) != n || a.shape(1) != 5 || b.shape(1) != 5) throw Error("geometry_residuals: two sightings (scan, stretch, frame, row, column) and one flag per comparison");
            std::vector<gc::Comparison> comparisons(n);
            for (std::size_t i = 0; i < n; ++i) comparisons[i].a = sighting_of(a.data() + 5 * i), comparisons[i].b = sighting_of(b.data() + 5 * i), comparisons[i].opposite = opposite[i];
            const std::vector<std::array<double, 3>> r = gc::residuals(acq, parameters_of(acq, parameters), comparisons);
            auto out = helirec::py::new_array_t<double>({n, std::size_t(3)});
            for (std::size_t i = 0; i < n; ++i) std::copy(r[i].begin(), r[i].end(), static_cast<double*>(out.data()) + 3 * i);
            return out;
        },
        "acquisition"_a, "parameters"_a, "a"_a, "b"_a, "opposite"_a,
        "What the parameters leave of each comparison: (n, 3) frames, rows, columns.\n"
        "(helirec::calibration::geometry_calibration::residuals)");
    m.def(
        "geometry_report",
        [](const gc::Acquisition& acq, const std::string& parameters, Table a, Table b, const std::vector<bool>& opposite, const std::vector<std::string>& kinds) {
            const std::size_t n = opposite.size();
            if (a.shape(0) != n || b.shape(0) != n || a.shape(1) != 5 || b.shape(1) != 5 || (!kinds.empty() && kinds.size() != n))
                throw Error("geometry_report: two sightings (scan, stretch, frame, row, column), one flag and one kind per comparison");
            std::vector<gc::Comparison> comparisons(n);
            for (std::size_t i = 0; i < n; ++i) {
                comparisons[i].a = sighting_of(a.data() + 5 * i), comparisons[i].b = sighting_of(b.data() + 5 * i), comparisons[i].opposite = opposite[i];
                if (!kinds.empty()) comparisons[i].kind = kinds[i];
            }
            gc::GeometrySolution s;
            s.parameters = parameters_of(acq, parameters);
            nb::gil_scoped_release nogil;
            s.residual = gc::residuals(acq, s.parameters, comparisons);
            return gc::report_json(acq, gc::report(acq, s, comparisons));
        },
        "acquisition"_a, "parameters"_a, "a"_a, "b"_a, "opposite"_a, "kinds"_a = std::vector<std::string>(),
        "What the parameters leave of each scan's opposite views and what that says (the rows' timing, whose the recorded\n"
        "height's departure is, the pitch a tile's own views bound), the spread over the scans, and which scans overlap,\n"
        "as JSON text. (helirec::calibration::geometry_calibration::report)");
    m.def(
        "geometry_neighbours", [](const gc::Acquisition& acq) {
            const gc::Neighbours n = gc::neighbours(acq);
            nb::list edges, triangles;
            for (const auto& e : n.edges) edges.append(nb::make_tuple(e[0], e[1]));
            for (const auto& t : n.triangles) triangles.append(nb::make_tuple(t[0], t[1], t[2]));
            return nb::make_tuple(edges, triangles);
        },
        "acquisition"_a, "The scans whose reconstructions overlap, and the triangles of them: (edges, triangles) as tuples of scan indices.\n"
                         "(helirec::calibration::geometry_calibration::neighbours)");
    m.def(
        "geometry_rays",
        [](const gc::Acquisition& acq, const std::string& parameters, Table sightings) {
            if (sightings.shape(1) != 5) throw Error("geometry_rays: a sighting is scan, stretch, frame, row, column");
            const gc::Parameters p = parameters_of(acq, parameters);
            const std::size_t n = sightings.shape(0);
            auto out = helirec::py::new_array_t<double>({n, std::size_t(3)});
            double* o = static_cast<double*>(out.data());
            for (std::size_t i = 0; i < n; ++i) {
                const gc::Ray r = gc::ray_of(acq, p, sighting_of(sightings.data() + 5 * i));
                o[3 * i] = r.angle_deg, o[3 * i + 1] = r.s, o[3 * i + 2] = r.z;
            }
            return out;
        },
        "acquisition"_a, "parameters"_a, "sightings"_a,
        "The ray each sighting records under the parameters: (n, 3) angle [deg], s [px], z [rows].\n"
        "(helirec::calibration::geometry_calibration::ray_of)");
    m.def(
        "calibrate_geometry",
        [](const std::vector<std::string>& scans, std::optional<double> row_time, const std::string& device, int order, int threads, int patch,
           std::optional<std::function<void(const std::string&)>> log, int reduce, const std::string& projections) {
            gc::Options opt;
            opt.row_time = row_time ? *row_time : std::nan("");
            if (device != "cpu" && device != "cuda" && device != "auto") throw Error("device is cpu, cuda or auto");
            opt.device = device == "cuda" || (device == "auto" && cuda_available()) ? Device::CUDA : Device::CPU;
            opt.order = order, opt.threads = threads, opt.patch = patch, opt.reduce = reduce, opt.projections = projections;
            if (log)
                opt.log = [&](const std::string& line) {
                    nb::gil_scoped_acquire gil;
                    (*log)(line);
                };
            nb::gil_scoped_release nogil;
            return gc::calibrate_geometry(scans, opt);
        },
        "scans"_a, "row_time"_a = nb::none(), "device"_a = "auto", "order"_a = 3, "threads"_a = 8, "patch"_a = 128, "log"_a = nb::none(), "reduce"_a = 0, "projections"_a = "",
        "The geometry calibration of the scans of one sample (a scan with the rotation axis inside its field first), as JSON text\n"
        "(helirec::calibration::geometry_calibration::calibrate_geometry; helirec-calibration geometry).");
    m.def(
        "geometry_to_stored_pixels",
        [](const gc::Acquisition& acq, const std::string& parameters, int rows, int cols, int f) {
            gc::Acquisition stored = acq;
            gc::GeometrySolution s;
            s.parameters = parameters_of(acq, parameters);
            gc::GeometryReport told;
            told.scans.resize(acq.scans.size());
            gc::to_stored_pixels(stored, s, told, rows, cols, f);
            return nb::make_tuple(gc::parameters_json(stored, s.parameters), gc::acquisition_json(stored));
        },
        "acquisition"_a, "parameters"_a, "rows"_a, "cols"_a, "f"_a,
        "Parameters found on frames read at 1 / f, in the pixels of the frames as stored (rows x cols): (the parameters, the\n"
        "layout without its per-frame series), each as JSON text. (helirec::calibration::geometry_calibration::to_stored_pixels)");
    m.def("geometry_calibration_save", &gc::save_text, "text"_a, "path"_a, "The text written whole or not at all.");
}
