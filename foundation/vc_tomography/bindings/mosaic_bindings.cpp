// nanobind bindings of the whole-canvas reconstruction of a mosaic (vc_tomography/mosaic.hpp)
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../src/common/staged.hpp"
#include "conversions.hpp"
#include "vc_tomography/mosaic.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace vc_tomography;
using namespace vc_tomography::py;

namespace {

using Rows = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Angles = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

std::optional<SliceRoi> roi_of(const std::optional<std::array<int, 4>>& roi) {
    if (!roi) return std::nullopt;
    return SliceRoi{(*roi)[0], (*roi)[1], (*roi)[2], (*roi)[3]};   // (x0, x1, y0, y1)
}

nb::dict debug_dict(const CanvasDebug& d, std::size_t n_angles, std::size_t width) {
    nb::dict out;
    out["sino"] = to_numpy(d.sino, {n_angles, width});
    out["weights"] = to_numpy(d.weights, {n_angles, width});
    out["filtered"] = to_numpy(d.filtered, {d.angles.size(), width});
    if (d.angles.size() == n_angles) out["opposite_only_fraction"] = d.opposite_only_fraction;
    out["angles"] = to_numpy(d.angles);
    return out;
}

}  // namespace

void bind_mosaic(nb::module_& m) {
    nb::class_<MosaicCanvas>(m, "MosaicCanvas",
                             "A mosaic's tiles in full-width filtered backprojection (vc_tomography::MosaicCanvas, vc_tomography/mosaic.hpp):\n"
                             "tiles: (rot_center, ax, ay, cols) of each, the axis' column in its rows and where the axis passes through the canvas.")
        .def("__init__",
             [](MosaicCanvas* self, int n_angles, const std::vector<std::array<double, 4>>& tiles, double pixel_size_m, const std::string& device, int half, int taper, bool levels,
                int level_step, const std::string& bp_interp, int bp_oversample, const std::string& ramp_filter_name, const std::vector<float>& ramp_filter_par, int anchor, bool rebin_angles, const std::vector<std::vector<float>>& fans, const std::string& rebin_kernel, const std::vector<double>& exposure_gates, double exposure_fade) {
                 CanvasConfig c;
                 c.device = parse_device(device);
                 c.n_angles = n_angles, c.half = half, c.pixel_size_m = pixel_size_m, c.taper = taper, c.levels = levels, c.level_step = level_step;
                 c.bp_interp = bp_interp, c.bp_oversample = bp_oversample, c.ramp_filter_name = ramp_filter_name, c.ramp_filter_par = ramp_filter_par, c.anchor = anchor;
                 std::vector<CanvasTile> t;
                 c.rebin_angles = rebin_angles;
                 c.rebin_kernel = rebin_kernel;
                 c.exposure_fade = exposure_fade;
                 require(exposure_gates.empty() || exposure_gates.size() == tiles.size(), "expected one exposure gate in radians per tile");
                 require(fans.empty() || fans.size() == tiles.size(), "expected one fan table per tile");
                 for (std::size_t i = 0; i < tiles.size(); ++i) {
                     const auto& v = tiles[i];
                     t.push_back(CanvasTile{v[0], v[1], v[2], static_cast<int>(v[3]), fans.empty() ? std::vector<float>{} : fans[i],
                                            exposure_gates.empty() ? 0.0 : exposure_gates[i]});
                 }
                 new (self) MosaicCanvas(c, std::move(t));
             },
             "n_angles"_a, "tiles"_a, "pixel_size_m"_a, "device"_a = "cpu", "half"_a = 0, "taper"_a = 0, "levels"_a = true, "level_step"_a = 25, "bp_interp"_a = "bspline3",
             "bp_oversample"_a = 2, "ramp_filter_name"_a = "ramlak", "ramp_filter_par"_a = std::vector<float>{}, "anchor"_a = 0, "rebin_angles"_a = true, "fans"_a = std::vector<std::vector<float>>{}, "rebin_kernel"_a = "exposure", "exposure_gates"_a = std::vector<double>{}, "exposure_fade"_a = 600.)
        .def_prop_ro("size", &MosaicCanvas::size, "pixels a side")
        .def_prop_ro("exposure_weighted", &MosaicCanvas::exposure_weighted)
        .def_prop_ro("half", &MosaicCanvas::half, "the middle pixel: the anchor's rotation axis")
        .def("begin", &MosaicCanvas::begin, "n_slices"_a, "a chunk of n_slices slices: empty sums")
        .def("add_view_angles", [](MosaicCanvas& c, int y, Angles a) {
                 require(a.shape(0) == static_cast<std::size_t>(c.config().n_angles), "expected one angle per view");
                 c.add_view_angles(y, a.data());
             }, "y"_a, "angles"_a)
        .def("add",
             [](MosaicCanvas& c, int tile, int y, Rows rows, Rows weights, Angles angles) {
                 const std::size_t n = rows.size();
                 require(weights.shape(0) == rows.shape(0) && weights.shape(1) == rows.shape(1), "add: rows and weights of one shape");
                 require(static_cast<int>(rows.shape(0)) == c.config().n_angles && angles.shape(0) == rows.shape(0), "add: a row and an angle per angle of the turn");
                 nb::gil_scoped_release nogil;
                 staged(c.config().device, [&](const float* r, const float* w) { c.add(tile, y, r, w, angles.data()); }, to_device(rows.data(), n), to_device(weights.data(), n));
             },
             "tile"_a, "y"_a, "rows"_a, "weights"_a, "angles"_a,
             "tile's rows of slice y: (n_angles, cols) -log values, their measured weights, and the angle of every row [rad] (u = x cos a + y sin a + rot_center)")
        .def("slice",
             [](MosaicCanvas& c, int y, std::optional<std::array<int, 4>> roi, bool debug) -> nb::object {
                 const std::optional<SliceRoi> r = roi_of(roi);
                 const std::size_t ny = r ? r->y1 - r->y0 : c.size(), nx = r ? r->x1 - r->x0 : c.size();
                 auto out = new_array({ny, nx});
                 CanvasDebug d;
                 {
                     nb::gil_scoped_release nogil;
                     c.slice(y, data(out), r ? &*r : nullptr, debug ? &d : nullptr);
                 }
                 if (!debug) return nb::cast(out);
                 nb::dict got = debug_dict(d, static_cast<std::size_t>(c.config().n_angles), static_cast<std::size_t>(c.size()));
                 got["slice"] = out;
                 return got;
             },
             "y"_a, "roi"_a = nb::none(), "debug"_a = false,
             "slice y of the chunk -> (y1 - y0, x1 - x0) float32 [1 / cm], roi = (x0, x1, y0, y1); debug: a dict with the slice, the sinogram as filtered, its weights, "
             "the ordinary filtered rows and their angles")
        .def("levels", [](const MosaicCanvas& c, int y) { return c.levels(y); }, "y"_a, "every tile's level in slice y (once made)")
        .def("angle_spread", &MosaicCanvas::angle_spread, "y"_a, "the largest difference of a tile's angle from the angle its bin is backprojected at [bins]");

    nb::class_<MosaicReconstruction>(m, "MosaicReconstruction",
                                     "A mosaic reconstructed as one slice by full-width filtered backprojection, from its frames and a joint geometry calibration that holds\n"
                                     "its tiles' places (vc_tomography::MosaicReconstruction, vc_tomography/mosaic.hpp).")
        .def("__init__",
             [](MosaicReconstruction* self, const std::string& projections, const std::string& geometry, const std::string& device, const std::string& params, int reduce,
                const std::string& projection_store, const std::string& cache_dir, int taper, bool levels, bool keep_tiles, std::optional<std::function<void(const std::string&)>> log, bool rebin_angles, const std::string& rebin_kernel) {
                 MosaicOptions o;
                 o.device = parse_device(device);
                 o.params = params, o.reduce = reduce, o.projection_store = projection_store, o.cache_dir = cache_dir, o.taper = taper, o.levels = levels, o.keep_tiles = keep_tiles;
                 o.log = python_log(log);
                 o.rebin_angles = rebin_angles;
                 o.rebin_kernel = rebin_kernel;
                 nb::gil_scoped_release nogil;
                 new (self) MosaicReconstruction(projections, geometry, std::move(o));
             },
             "projections"_a, "geometry"_a, "device"_a = "cuda", "params"_a = "{}", "reduce"_a = 0, "projection_store"_a = "", "cache_dir"_a = "", "taper"_a = 200, "levels"_a = true,
             "keep_tiles"_a = false, "log"_a = nb::none(), "rebin_angles"_a = true, "rebin_kernel"_a = "exposure")
        .def_prop_ro("doable", &MosaicReconstruction::doable, "slices [lo, hi) every tile can give")
        .def_prop_ro("max_chunk", &MosaicReconstruction::max_chunk)
        .def_prop_ro("size", &MosaicReconstruction::size, "canvas pixels a side")
        .def_prop_ro("pixel_size_m", &MosaicReconstruction::pixel_size_m)
        .def("info", &MosaicReconstruction::info_json, "doable, size, half, n_angles, pixel_size_m, reduce, row_origin, tiles and what every tile's pipeline is given, as JSON text")
        .def_prop_ro("levels", [](const MosaicReconstruction& r) { return r.levels(); }, "of the last chunk, per slice: every tile's level")
        .def_prop_ro("angle_spread", [](const MosaicReconstruction& r) { return r.angle_spread(); }, "of the last chunk, per slice [bins]")
        .def("reconstruct",
             [](MosaicReconstruction& r, int h_lo, int h_hi, std::optional<std::array<int, 4>> roi, bool debug) -> nb::object {
                 require(h_hi > h_lo, "reconstruct: h_hi > h_lo");
                 const std::optional<SliceRoi> box = roi_of(roi);
                 const std::size_t n = static_cast<std::size_t>(h_hi - h_lo), ny = box ? box->y1 - box->y0 : r.size(), nx = box ? box->x1 - box->x0 : r.size();
                 auto out = new_array({n, ny, nx});
                 std::vector<CanvasDebug> d;
                 {
                     nb::gil_scoped_release nogil;
                     r.reconstruct(h_lo, h_hi, data(out), box ? &*box : nullptr, debug ? &d : nullptr);
                 }
                 if (!debug) return nb::cast(out);
                 const std::string info = r.info_json();
                 const std::size_t n_angles = nb::cast<std::size_t>(json_loads(info)["n_angles"]);
                 nb::list per_slice;
                 for (const CanvasDebug& s : d) per_slice.append(debug_dict(s, n_angles, static_cast<std::size_t>(r.size())));
                 nb::dict got;
                 got["slices"] = out;
                 got["debug"] = per_slice;
                 return got;
             },
             "h_lo"_a, "h_hi"_a, "roi"_a = nb::none(), "debug"_a = false,
             "slices h_lo .. h_hi - 1 -> (n, y1 - y0, x1 - x0) float32 [1 / cm]; debug: dict(slices, debug: per slice the sinogram as filtered, its weights, the filtered rows, the angles)");
}
