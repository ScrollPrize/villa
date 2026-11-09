// nanobind bindings of the calibration kernels (helirec._core.calib_*)
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "helirec/calib.hpp"
#include "ndarray_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::py;

namespace {

template <class T>
using Arr = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> gauss(Arr<T> a, int axis, double sigma, const std::string& mode, double truncate) {
    require(a.ndim() >= 1 && a.ndim() <= 3, "gaussian_filter1d: 1 to 3 dimensions");
    std::size_t d[3] = {1, 1, 1};
    const int nd = static_cast<int>(a.ndim());
    for (int i = 0; i < nd; ++i) d[3 - nd + i] = a.shape(i);
    if (axis < 0) axis += nd;
    require(axis >= 0 && axis < nd, "gaussian_filter1d: bad axis");
    std::vector<std::size_t> shape(nd);
    for (int i = 0; i < nd; ++i) shape[i] = a.shape(i);
    auto out = new_array_t<T>(shape);
    nb::gil_scoped_release nogil;
    calib::gaussian_filter1d<T>(a.data(), static_cast<T*>(out.data()), d[0], d[1], d[2], axis + (3 - nd), sigma, mode,
                                truncate);
    return out;
}

using D3 = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

}  // namespace

void bind_calib(nb::module_& m) {
    m.def("gaussian_filter1d", &gauss<float>, "a"_a, "axis"_a, "sigma"_a, "mode"_a = "reflect", "truncate"_a = 4.0,
          "scipy.ndimage.gaussian_filter1d semantics (reflect | nearest), double accumulation");
    m.def("gaussian_filter1d", &gauss<double>, "a"_a, "axis"_a, "sigma"_a, "mode"_a = "reflect", "truncate"_a = 4.0);
    m.def("apodisation_mask",
          [](int Lx, float w_left, float w_right, float min_val, int sh_left, int sh_right) {
              auto v = calib::apodisation_mask(Lx, w_left, w_right, min_val, sh_left, sh_right);
              HostOut a = new_array({v.size()});
              std::copy(v.begin(), v.end(), data(a));
              return a;
          },
          "Lx"_a, "w_left"_a, "w_right"_a, "min_val"_a = 1e-8f, "sh_left"_a = 0, "sh_right"_a = 0);
    m.def("weight_border_v",
          [](int L, double tw, int crop_top, int crop_bottom) {
              auto v = calib::weight_border_v(L, tw, crop_top, crop_bottom);
              HostOut a = new_array({v.size()});
              std::copy(v.begin(), v.end(), data(a));
              return a;
          },
          "L"_a, "transition_width"_a, "crop_top"_a = 0, "crop_bottom"_a = 0);
    m.def(
        "build_all_merits",
        [](nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> sig,
           nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> w, int ovl_min, int ovl_max, int ovs,
           float sigma_wide, float sigma_narrow) {
            require(sig.shape(0) == 2 && w.shape(0) == 2 && sig.shape(1) == w.shape(1) && sig.shape(2) == w.shape(2),
                    "sig, w must be (2, n_lines, W)");
            const std::size_t L = sig.shape(1), W = sig.shape(2);
            const std::size_t steps = static_cast<std::size_t>((ovl_max - ovl_min) * ovs + 1);
            HostOut merits = new_array({L, steps}), energies = new_array({L, steps});
            {
                nb::gil_scoped_release nogil;
                calib::build_all_merits(sig.data(), w.data(), L, W, ovl_min, ovl_max, ovs, sigma_wide, sigma_narrow,
                                        data(merits), data(energies));
            }
            return nb::make_tuple(merits, energies);
        },
        "sig"_a, "w"_a, "ovl_min"_a, "ovl_max"_a, "ovs"_a = 4, "sigma_wide"_a = 10.f, "sigma_narrow"_a = 0.4f,
        "rotation-axis merits (n_lines, n_steps) and energies");
    m.def(
        "find_offset",
        [](D3 img0, D3 w0, D3 img1, D3 w1, int init_dy, int init_dx, int ry, int rx, double sub_range, double sub_step) {
            const std::size_t n = img0.shape(0), Y = img0.shape(1), X = img0.shape(2);
            calib::OffsetResult r;
            {
                nb::gil_scoped_release nogil;
                r = calib::find_offset(img0.data(), w0.data(), img1.data(), w1.data(), n, Y, X, init_dy, init_dx, ry, rx,
                                       sub_range, sub_step);
            }
            auto lm = new_array_t<double>({static_cast<std::size_t>(2 * ry + 1), static_cast<std::size_t>(2 * rx + 1)});
            std::copy(r.loss_map.begin(), r.loss_map.end(), static_cast<double*>(lm.data()));
            nb::dict d;
            d["best_int"] = nb::make_tuple(r.best_dy, r.best_dx);
            d["best_int_loss"] = r.best_loss;
            d["loss_int"] = lm;
            d["best_sub"] = nb::make_tuple(r.sub_dy, r.sub_dx);
            d["best_sub_loss"] = r.sub_loss;
            return d;
        },
        "img0"_a, "w0"_a, "img1"_a, "w1"_a, "init_dy"_a, "init_dx"_a, "ry"_a, "rx"_a, "sub_range"_a = 1.5,
        "sub_step"_a = 0.25, "L1 overlap search: integer (dy, dx) around the guess, then sub-pixel refinement");
}
