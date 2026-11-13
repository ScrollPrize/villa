// nanobind bindings of the intensity calibration's kernels (helirec/calibration/intensity.hpp)
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>

#include "fit_summary.hpp"
#include "helirec/calibration/intensity.hpp"
#include "helirec/calibration/transfer.hpp"
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
    calibration::gaussian_filter1d<T>(a.data(), static_cast<T*>(out.data()), d[0], d[1], d[2], axis + (3 - nd), sigma, mode,
                                truncate);
    return out;
}

template <class T>
nb::object array_of(const std::vector<T>& v, std::vector<std::size_t> shape) {
    auto a = new_array_t<T>(shape);
    std::copy(v.begin(), v.end(), static_cast<T*>(a.data()));
    return nb::cast(a, nb::rv_policy::move);
}

nb::object darks_flats_tuple(const calibration::DarksFlats& d) {
    const std::size_t r = static_cast<std::size_t>(d.rows), c = static_cast<std::size_t>(d.cols);
    return nb::make_tuple(array_of(d.dark, {r, c}), array_of(d.flats, {d.series(), r, c}));
}

// a log line handed to a Python callable (None: silent), with the GIL taken for it
std::function<void(const std::string&)> python_log(const std::optional<std::function<void(const std::string&)>>& log) {
    if (!log) return nullptr;
    return [log](const std::string& line) {
        nb::gil_scoped_acquire gil;
        (*log)(line);
    };
}

nb::object json_loads(const std::string& text) { return nb::module_::import_("json").attr("loads")(text); }

calibration::Scan scan_of(nb::handle h) { return nb::cast<calibration::Scan>(h); }

}  // namespace

void bind_intensity(nb::module_& m) {
    nb::class_<calibration::ScanPlacement>(m, "ScanPlacement",
                                           "Where a scan lies in the merged frame: its column 0 at merged column shift_x - x_readjustment, its rows "
                                           "z_readjustment rows from the first scan's (helirec::calibration::ScanPlacement)")
        .def_ro("shift_x", &calibration::ScanPlacement::shift_x)
        .def_ro("x_readjustment", &calibration::ScanPlacement::x_readjustment)
        .def_ro("z_readjustment", &calibration::ScanPlacement::z_readjustment)
        .def("__repr__", [](const calibration::ScanPlacement& p) {
            return "ScanPlacement(shift_x=" + std::to_string(p.shift_x) + ", x_readjustment=" + std::to_string(p.x_readjustment) +
                   ", z_readjustment=" + std::to_string(p.z_readjustment) + ")";
        });
    using IC = calibration::IntensityCalibration;
    nb::class_<IC>(m, "IntensityCalibration",
                   "The intensity calibration of a scan pair in its merged frame (helirec::calibration::calibrate_intensity)")
        .def_ro("main", &IC::main)
        .def_ro("companion", &IC::companion)
        .def_ro("cor_main_detector", &IC::cor_main_detector)
        .def_prop_ro("vector", [](const IC& c) { return nb::make_tuple(c.vector[0], c.vector[1]); })
        .def_ro("rot_center", &IC::rot_center)
        .def_ro("merged_width", &IC::merged_width)
        .def_prop_ro("placement", [](const IC& c) {
            nb::dict d;
            d[nb::str(c.main.c_str())] = nb::cast(c.placement[0]);
            d[nb::str(c.companion.c_str())] = nb::cast(c.placement[1]);
            return d;
        })
        .def_prop_ro("apodisation", [](const IC& c) {
            nb::dict d;
            d[nb::str(c.main.c_str())] = nb::make_tuple(c.apodisation[0][0], c.apodisation[0][1]);
            d[nb::str(c.companion.c_str())] = nb::make_tuple(c.apodisation[1][0], c.apodisation[1][1]);
            return d;
        })
        .def_prop_ro("estimated_cor_from_motor_pix", [](const IC& c) {
            nb::dict d;
            d[nb::str(c.main.c_str())] = c.estimated_cor_from_motor_pix[0];
            d[nb::str(c.companion.c_str())] = c.estimated_cor_from_motor_pix[1];
            return d;
        })
        .def_prop_ro("detector_blur_field", [](const IC& c) { return c.detector_blur.field; })
        .def_prop_ro("details", [](const IC& c) -> nb::object {
            nb::object d = json_loads(c.json());
            return d["details"];
        })
        .def_ro("fingerprint", &IC::fingerprint)
        .def_ro("version", &IC::version)
        .def("to_json", &IC::json, "calibration.json: everything but the arrays")
        .def(
            "arrays",
            [](const IC& c) {
                const std::size_t r = static_cast<std::size_t>(c.rows), k = static_cast<std::size_t>(c.cols);
                nb::dict d;
                d["main_dark"] = array_of(c.main_reduced.dark, {r, k});
                d["main_flats"] = array_of(c.main_reduced.flats, {c.main_reduced.series(), r, k});
                d["companion_dark"] = array_of(c.companion_reduced.dark, {r, k});
                d["companion_flats"] = array_of(c.companion_reduced.flats, {c.companion_reduced.series(), r, k});
                d["weight_map"] = array_of(c.weight_map, {r, k});
                d["main_weight"] = array_of(c.main_weight, {r, k});
                d["companion_weight"] = array_of(c.companion_weight, {r, k});
                return d;
            },
            "the darks and flats reduced, the weight map and each scan's weights (the weight map times its apodisation)");

    m.def(
        "calibrate_intensity",
        [](nb::handle main, nb::handle companion, double cor_main_detector, std::array<double, 2> vector, nb::dict measured, const std::string& cache_dir,
           std::optional<std::function<void(const std::string&)>> log, bool progress) {
            const calibration::Scan a = scan_of(main), b = scan_of(companion);
            geometry::MergedLayout layout;
            layout.cor_main_detector = cor_main_detector;
            layout.vector = vector;
            const auto get = [&](const char* k) { return measured.contains(k) ? nb::cast<double>(measured[k]) : 0.0; };
            layout.axis_first_scan = get("axis_first_scan"), layout.axis_second_scan = get("axis_second_scan");
            layout.second_scan_rows = get("second_scan_rows"), layout.quantum = get("quantum");
            const auto say = python_log(log);
            nb::gil_scoped_release nogil;
            return calibration::calibrate_intensity(a, b, layout, cache_dir, say, progress);
        },
        "main"_a, "companion"_a, "cor_main_detector"_a, "vector"_a, "measured"_a = nb::dict(), "cache_dir"_a = "", "log"_a = nb::none(), "progress"_a = false,
        "The intensity calibration of a scan pair in the merged frame of one axis (in the main scan's columns) and one vector (rows, columns) to the "
        "companion; cache_dir: calibration_arrays.bin and calibration.json kept there. (helirec::calibration::calibrate_intensity)");
    m.def(
        "reduce_darks_flats",
        [](nb::handle scan, std::optional<std::function<void(std::size_t)>> progress) {
            const calibration::Scan s = scan_of(scan);
            std::function<void(std::size_t)> p;
            if (progress)
                p = [progress](std::size_t k) {
                    nb::gil_scoped_acquire gil;
                    (*progress)(k);
                };
            calibration::DarksFlats d;
            {
                nb::gil_scoped_release nogil;
                d = calibration::reduce_darks_flats(s, p);
            }
            return darks_flats_tuple(d);
        },
        "scan"_a, "progress"_a = nb::none(),
        "(dark (rows, cols), flats (series, rows, cols)) of a scan exactly as nx2etf stores them (helirec::calibration::reduce_darks_flats)");
    m.def(
        "reduced_darks_flats",
        [](nb::handle scan, const std::string& cache_dir, std::optional<std::function<void(const std::string&)>> log) {
            const calibration::Scan s = scan_of(scan);
            const auto say = python_log(log);
            calibration::DarksFlats d;
            {
                nb::gil_scoped_release nogil;
                d = calibration::reduced_darks_flats(s, cache_dir, say);
            }
            return darks_flats_tuple(d);
        },
        "scan"_a, "cache_dir"_a = "", "log"_a = nb::none(),
        "reduce_darks_flats(scan), kept in cache_dir as darks_flats_<scan>.bin (helirec::calibration::reduced_darks_flats)");
    m.def(
        "placement_from_vector",
        [](std::array<double, 2> vector, int cols) {
            const calibration::MergedPlacement p = calibration::placement_from_vector(vector, cols);
            nb::list scans;
            for (const auto& s : p.scans) scans.append(nb::cast(s));
            return nb::make_tuple(scans, p.width, p.anchor);
        },
        "vector"_a, "cols"_a,
        "nabuxx check_companion: ([main, companion] placements, merged width, anchor) (helirec::calibration::placement_from_vector)");
    m.def(
        "acquisition_fingerprint", [](nb::handle main, nb::handle companion) { return calibration::acquisition_fingerprint(scan_of(main), scan_of(companion)); },
        "main"_a, "companion"_a, "sha256 over what the calibration of a pair depends on (helirec::calibration::acquisition_fingerprint)");
    m.def(
        "flat_noise_spectra",
        [](nb::handle scan, Arr<float> dark, int n_flats, int tile) {
            const calibration::Scan s = scan_of(scan);
            require(dark.ndim() == 2 && dark.shape(0) == static_cast<std::size_t>(s.rows) && dark.shape(1) == static_cast<std::size_t>(s.cols), "the dark is not a frame");
            calibration::FlatNoise f;
            {
                nb::gil_scoped_release nogil;
                f = calibration::flat_noise_spectra(s, dark.data(), n_flats, tile);
            }
            return nb::make_tuple(array_of(f.power, {f.ys.size(), f.xs.size(), static_cast<std::size_t>(tile), static_cast<std::size_t>(tile / 2 + 1)}), f.ys, f.xs,
                                  f.n_flats);
        },
        "scan"_a, "dark"_a, "n_flats"_a = 200, "tile"_a = calibration::kBlurTile,
        "Noise power spectra of the scan's first flat series on tiles -> (power, ys, xs, flats used) (helirec::calibration::flat_noise_spectra)");
    m.def(
        "measure_detector_blur",
        [](nb::handle scan, Arr<float> dark, int n_flats, std::optional<std::function<void(const std::string&)>> log) -> nb::object {
            const calibration::Scan s = scan_of(scan);
            require(dark.ndim() == 2 && dark.shape(0) == static_cast<std::size_t>(s.rows) && dark.shape(1) == static_cast<std::size_t>(s.cols), "the dark is not a frame");
            const auto say = python_log(log);
            calibration::IntensityCalibration c;
            {
                nb::gil_scoped_release nogil;
                c.detector_blur = calibration::measure_detector_blur(s, dark.data(), n_flats, say);
            }
            nb::object d = json_loads(c.json());
            nb::object details = d["details"];
            return nb::object(details["detector_blur"]);
        },
        "scan"_a, "dark"_a, "n_flats"_a = 200, "log"_a = nb::none(),
        "The detector's blur field of a scan from the noise of its flats, as a dict (helirec::calibration::measure_detector_blur)");
    m.attr("INTENSITY_CALIBRATION_VERSION") = calibration::kIntensityCalibrationVersion;
    m.attr("NOMINAL_CURRENT_A") = calibration::kNominalCurrentA;

    const auto curve = [](const calibration::TransferFit& fit) {
        auto d = fit_summary_dict(fit);
        d["x"] = nb::cast(fit.x); d["residual"] = nb::cast(fit.residual); d["rms"] = fit.rms;
        return d;
    };
    m.def("calibration_fit_transfer", [curve](const std::vector<double>& frequency, const std::vector<double>& gain) {
        calibration::TransferFit fit;
        { nb::gil_scoped_release release; fit = calibration::fit_transfer(frequency,gain); }
        return curve(fit);
    }, "frequency"_a, "gain"_a);
    m.def("calibration_fit_smear", [curve](const std::vector<double>& frequency, const std::vector<double>& ratio) {
        calibration::TransferFit fit;
        { nb::gil_scoped_release release; fit = calibration::fit_smear(frequency,ratio); }
        return curve(fit);
    }, "frequency"_a, "ratio"_a);
    m.def("calibration_transfer_model", &calibration::transfer_model, "parameters"_a, "frequency"_a);
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
                mean = calibration::mean_frames(reader, file_index.data(), frame_index.data(), n, dark ? dark->data() : nullptr, control_A ? control_A->data() : nullptr, nominal_A,
                                          report);
            }
            std::copy(mean.begin(), mean.end(), static_cast<float*>(out.data()));
            return out;
        },
        "reader"_a, "file_index"_a, "frame_index"_a, "dark"_a = nb::none(), "control_A"_a = nb::none(), "nominal_A"_a = 0.0, "progress"_a = nb::none(),
        "The mean of frames as nx2etf reduces them (float32, added in order); with dark and control_A each frame F enters as\n"
        "dark + (F - dark) nominal_A / control_A. (helirec::calibration::mean_frames)");
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
            const std::vector<float> w = calibration::weight_map(flats, static_cast<int>(rows), static_cast<int>(cols), transition_width_vertical, sigma, crop_top, crop_bottom);
            std::copy(w.begin(), w.end(), static_cast<float*>(out.data()));
            return out;
        },
        "flat_stacks"_a, "transition_width_vertical"_a = 15.0, "sigma"_a = 10.0, "crop_top"_a = 0, "crop_bottom"_a = 0,
        "prepare_weights_double: gaussian_filter(mean(all flats) / max, sigma) * vertical erf border. (helirec::calibration::weight_map)");
    m.def(
        "apodisation_widths",
        [](int n_cols, double est_cor_main_offset, double companion_dx, double transition_width_horizontal, double transition_width_borders) {
            const calibration::ApodisationWidths w = calibration::apodisation_widths(n_cols, est_cor_main_offset, companion_dx, transition_width_horizontal, transition_width_borders);
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
                r = calibration::at_resolution(images.data(), images.shape(0), static_cast<int>(images.shape(1)), static_cast<int>(images.shape(2)), reduce, chunk_rows, &rows, &cols);
            }
            auto out = new_array_t<float>({images.shape(0), static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)});
            std::copy(r.begin(), r.end(), static_cast<float*>(out.data()));
            return out;
        },
        "images"_a, "reduce"_a, "chunk_rows"_a,
        "Detector images on the grid of RawReader(reduce=) for frames stored as JPEG 2000. (helirec::calibration::at_resolution)");
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
                calibration::tile_power_spectra(img.data(), img.shape(0), img.shape(1), img.shape(2), tile, ys.data(),
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
            calibration::DetectorBlurField r;
            {
                nb::gil_scoped_release nogil;
                r = calibration::detector_blur_field_fit(power.data(), static_cast<int>(ys.size()), static_cast<int>(xs.size()),
                                           static_cast<int>(power.shape(2)), ys.data(), xs.data(), rows, cols, f_max);
            }
            return nb::make_tuple(std::vector<double>(r.field, r.field + 6), r.rms_none, r.rms_fit, fit_summary_dict(r.solver));
        },
        "power"_a, "ys"_a, "xs"_a, "rows"_a, "cols"_a, "f_max"_a = 0.30,
        "The detector's blur field from the noise power spectra of flats on tiles (tile_power_spectra) -> "
        "([a11, a12, a21, a22, c1, c2], rms_none, rms_fit, solver): a uniform blur of A r + c pixels at r = (column, row) "
        "from the detector's centre in units of 1000 pixels");
    m.def("apodisation_mask",
          [](int Lx, float w_left, float w_right, float min_val, int sh_left, int sh_right) {
              auto v = calibration::apodisation_mask(Lx, w_left, w_right, min_val, sh_left, sh_right);
              HostOut a = new_array({v.size()});
              std::copy(v.begin(), v.end(), data(a));
              return a;
          },
          "Lx"_a, "w_left"_a, "w_right"_a, "min_val"_a = 1e-8f, "sh_left"_a = 0, "sh_right"_a = 0);
    m.def("weight_border_v",
          [](int L, double tw, int crop_top, int crop_bottom) {
              auto v = calibration::weight_border_v(L, tw, crop_top, crop_bottom);
              HostOut a = new_array({v.size()});
              std::copy(v.begin(), v.end(), data(a));
              return a;
          },
          "L"_a, "transition_width"_a, "crop_top"_a = 0, "crop_bottom"_a = 0);
}
