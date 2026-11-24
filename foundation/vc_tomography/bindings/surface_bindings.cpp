// Python bindings of vc_tomography/surface_composite.hpp: BandComposite (python/vc_tomography/band.py), parameter sweeps.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vc_tomography/surface_composite.hpp"
#include "ndarray_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using vc_tomography::py::new_array_t;
namespace hs = vc_tomography::surface;

using vc_tomography::py::json_loads;
using vc_tomography::py::to_numpy_object;

void bind_surface(nb::module_& m) {
    m.def("sweep_sets", &hs::sweep_sets, "items"_a,
          "--sweep KEY=V1,V2,... items as parameter sets: a JSON list of objects' text (vc_tomography::surface::sweep_sets)");
    m.def("set_name", &hs::set_name, "changes"_a, "a parameter set's folder name, e.g. db500_us_coeff2 (vc_tomography::surface::set_name)");
    nb::class_<hs::BandComposite>(m, "BandComposite",
                                  "Surface layers reconstructed where their points lie (vc_tomography::surface::BandComposite, surface_composite.hpp)")
        .def(
            "__init__",
            [](hs::BandComposite* self, vc_tomography::Reconstruction& recon, nb::ndarray<const float, nb::ndim<4>, nb::c_contig, nb::device::cpu> pts,
               nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> valid, std::optional<std::array<int, 2>> h_range, int n_sets) {
                vc_tomography::py::require(pts.shape(3) == 3 && valid.shape(0) == pts.shape(1) && valid.shape(1) == pts.shape(2), "pts (L, H, W, 3) and valid (H, W)");
                nb::gil_scoped_release nogil;
                new (self) hs::BandComposite(recon, pts.data(), static_cast<int>(pts.shape(0)), static_cast<int>(pts.shape(1)), static_cast<int>(pts.shape(2)),
                                             valid.data(), h_range, n_sets);
            },
            "recon"_a, "pts"_a, "valid"_a, "h_range"_a = nb::none(), "n_sets"_a = 1, nb::keep_alive<1, 2>())
        .def_prop_ro("band", &hs::BandComposite::band)
        .def_prop_ro("n_points", &hs::BandComposite::n_points)
        .def_prop_ro("h_range", &hs::BandComposite::h_range)
        .def_prop_ro("box", &hs::BandComposite::box, "(x0, x1, y0, y1) of the pixels the points need")
        .def_prop_ro("valid", [](const hs::BandComposite& b) {
            auto a = new_array_t<bool>({static_cast<std::size_t>(b.H()), static_cast<std::size_t>(b.W())});
            std::copy(b.valid().begin(), b.valid().end(), static_cast<bool*>(a.data()));
            return nb::cast(a, nb::rv_policy::move);
        })
        .def(
            "run",
            [](hs::BandComposite& b, int chunk, bool timing, std::vector<std::string> sweep, bool prefetch, std::optional<std::function<void(int, int, double)>> progress) {
                std::function<void(int, int, double)> p = vc_tomography::py::py_callback(progress);
                std::string r;
                {
                    nb::gil_scoped_release nogil;
                    r = b.run(chunk, timing, sweep, prefetch, p);
                }
                return json_loads(r);
            },
            "chunk"_a = 64, "timing"_a = false, "sweep"_a = std::vector<std::string>{}, "prefetch"_a = true, "progress"_a = nb::none(),
            "Reconstruct h_range in chunks; sweep: one filter chain per value set (JSON text). Returns seconds per stage and 'wall'")
        .def("layers", [](const hs::BandComposite& b, int set) {
            return to_numpy_object(b.layers(set), {static_cast<std::size_t>(b.L()), static_cast<std::size_t>(b.H()), static_cast<std::size_t>(b.W())});
        }, "set"_a = 0)
        .def("composites", [](const hs::BandComposite& b, int set) {
            const auto c = b.composites(set);
            const std::vector<std::size_t> s{static_cast<std::size_t>(b.H()), static_cast<std::size_t>(b.W())};
            return nb::make_tuple(to_numpy_object(c[0], s), to_numpy_object(c[1], s));
        }, "set"_a = 0, "(max, mean) over the layers (vc_reader's composite::reduce)");
}
