// nanobind bindings of the intensity calibration's kernels (calib.hpp)
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

}  // namespace

void bind_calib(nb::module_& m) {
    m.def("gaussian_filter1d", &gauss<float>, "a"_a, "axis"_a, "sigma"_a, "mode"_a = "reflect", "truncate"_a = 4.0,
          "scipy.ndimage.gaussian_filter1d semantics (reflect | nearest), double accumulation");
    m.def("gaussian_filter1d", &gauss<double>, "a"_a, "axis"_a, "sigma"_a, "mode"_a = "reflect", "truncate"_a = 4.0);
    m.def(
        "tile_power_spectra",
        [](nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> img, int tile, std::vector<int> ys,
           std::vector<int> xs) {
            auto out = new_array_t<double>({ys.size(), xs.size(), static_cast<std::size_t>(tile),
                                                         static_cast<std::size_t>(tile / 2 + 1)});
            double* d = static_cast<double*>(out.data());
            std::fill(d, d + ys.size() * xs.size() * tile * (tile / 2 + 1), 0.0);
            {
                nb::gil_scoped_release nogil;
                calib::tile_power_spectra(img.data(), img.shape(0), img.shape(1), img.shape(2), tile, ys.data(),
                                          static_cast<int>(ys.size()), xs.data(), static_cast<int>(xs.size()), d);
            }
            return out;
        },
        "img"_a, "tile"_a, "ys"_a, "xs"_a,
        "Hann-windowed power spectra of the tiles at (ys, xs) of images (n, rows, cols), summed over the images -> "
        "(len(ys), len(xs), tile, tile / 2 + 1) float64, rfft2 layout");
    m.def(
        "detector_blur_field_fit",
        [](nb::ndarray<const double, nb::ndim<4>, nb::c_contig, nb::device::cpu> power, std::vector<int> ys,
           std::vector<int> xs, std::size_t rows, std::size_t cols, double f_max) {
            require(power.shape(0) == ys.size() && power.shape(1) == xs.size(), "power must be (len(ys), len(xs), tile, tile / 2 + 1)");
            require(power.shape(3) == power.shape(2) / 2 + 1, "power must be (len(ys), len(xs), tile, tile / 2 + 1)");
            calib::DetectorBlurField r;
            {
                nb::gil_scoped_release nogil;
                r = calib::detector_blur_field_fit(power.data(), static_cast<int>(ys.size()), static_cast<int>(xs.size()),
                                           static_cast<int>(power.shape(2)), ys.data(), xs.data(), rows, cols, f_max);
            }
            return nb::make_tuple(std::vector<double>(r.field, r.field + 6), r.rms_none, r.rms_fit);
        },
        "power"_a, "ys"_a, "xs"_a, "rows"_a, "cols"_a, "f_max"_a = 0.30,
        "The detector's blur field from the noise power spectra of flats on tiles (tile_power_spectra) -> "
        "([a11, a12, a21, a22, c1, c2], rms_none, rms_fit): a uniform blur of A r + c pixels at r = (column, row) "
        "from the detector's centre in units of 1000 pixels");
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
}
