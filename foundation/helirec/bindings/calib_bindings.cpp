// nanobind bindings of the intensity calibration's kernels (calib.hpp)
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "helirec/calib.hpp"
#include "helirec/raw_reader.hpp"
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
    m.def(
        "mean_frames",
        [](RawReader& reader, IntIn file_index, IntIn frame_index, std::optional<Arr<float>> dark, std::optional<Arr<double>> control_A, double nominal_A,
           std::optional<std::function<void(std::size_t)>> progress) {
            const std::size_t n = file_index.shape(0);
            require(frame_index.shape(0) == n, "mean_frames: one file and one frame index per frame");
            const std::size_t rows = static_cast<std::size_t>(reader.rows()), cols = static_cast<std::size_t>(reader.cols());
            require(dark.has_value() == control_A.has_value(), "mean_frames: a dark and the ring current of every frame go together");
            if (dark) {
                require(dark->ndim() == 2 && dark->shape(0) == rows && dark->shape(1) == cols, "mean_frames: the dark is not a frame");
                require(control_A->ndim() == 1 && control_A->shape(0) == n, "mean_frames: one ring current per frame");
            }
            std::function<void(std::size_t)> report;
            if (progress)
                report = [&](std::size_t k) {
                    nb::gil_scoped_acquire gil;
                    (*progress)(k);
                };
            auto out = new_array_t<float>({rows, cols});
            std::vector<float> mean;
            {
                nb::gil_scoped_release nogil;
                mean = calib::mean_frames(reader, file_index.data(), frame_index.data(), n, dark ? dark->data() : nullptr, control_A ? control_A->data() : nullptr, nominal_A,
                                          report);
            }
            std::copy(mean.begin(), mean.end(), static_cast<float*>(out.data()));
            return out;
        },
        "reader"_a, "file_index"_a, "frame_index"_a, "dark"_a = nb::none(), "control_A"_a = nb::none(), "nominal_A"_a = 0.0, "progress"_a = nb::none(),
        "The mean of frames as nx2etf reduces them (float32, added in order); with dark and control_A each frame F enters as\n"
        "dark + (F - dark) nominal_A / control_A. (helirec::calib::mean_frames)");
    m.def(
        "weight_map",
        [](const std::vector<Arr<float>>& stacks, double transition_width_vertical, double sigma, int crop_top, int crop_bottom) {
            require(!stacks.empty(), "weight_map: no flats");
            std::vector<const float*> flats;
            std::size_t rows = 0, cols = 0;
            for (const auto& st : stacks) {
                require(st.ndim() == 3, "weight_map: stacks of flats (n, rows, cols)");
                if (flats.empty()) rows = st.shape(1), cols = st.shape(2);
                require(st.shape(1) == rows && st.shape(2) == cols, "weight_map: the flats differ in shape");
                for (std::size_t k = 0; k < st.shape(0); ++k) flats.push_back(st.data() + k * rows * cols);
            }
            auto out = new_array_t<float>({rows, cols});
            const std::vector<float> w = calib::weight_map(flats, static_cast<int>(rows), static_cast<int>(cols), transition_width_vertical, sigma, crop_top, crop_bottom);
            std::copy(w.begin(), w.end(), static_cast<float*>(out.data()));
            return out;
        },
        "flat_stacks"_a, "transition_width_vertical"_a = 15.0, "sigma"_a = 10.0, "crop_top"_a = 0, "crop_bottom"_a = 0,
        "prepare_weights_double: gaussian_filter(mean(all flats) / max, sigma) * vertical erf border. (helirec::calib::weight_map)");
    m.def(
        "apodisation_widths",
        [](int n_cols, double est_cor_main_offset, double companion_dx, double transition_width_horizontal, double transition_width_borders) {
            const calib::ApodisationWidths w = calib::apodisation_widths(n_cols, est_cor_main_offset, companion_dx, transition_width_horizontal, transition_width_borders);
            return nb::make_tuple(nb::make_tuple(w.main_left, w.main_right), nb::make_tuple(w.companion_left, w.companion_right));
        },
        "n_cols"_a, "est_cor_main_offset"_a, "companion_dx"_a, "transition_width_horizontal"_a = 1000.0, "transition_width_borders"_a = 50.0);
    m.def(
        "at_resolution",
        [](Arr<double> images, int reduce, int chunk_rows) {
            require(images.ndim() == 3, "at_resolution: images (n, rows, cols)");
            int rows = 0, cols = 0;
            std::vector<float> r;
            {
                nb::gil_scoped_release nogil;
                r = calib::at_resolution(images.data(), images.shape(0), static_cast<int>(images.shape(1)), static_cast<int>(images.shape(2)), reduce, chunk_rows, &rows, &cols);
            }
            auto out = new_array_t<float>({images.shape(0), static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)});
            std::copy(r.begin(), r.end(), static_cast<float*>(out.data()));
            return out;
        },
        "images"_a, "reduce"_a, "chunk_rows"_a,
        "Detector images on the grid of RawReader(reduce=) for frames stored as JPEG 2000. (helirec::calib::at_resolution)");
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
