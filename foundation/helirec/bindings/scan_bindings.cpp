// Python bindings of helirec/calibration/scan.hpp: a BLISS tomo scan as the reconstruction reads it, the scans
// below a directory and the families of an extended field of view (python/helirec/bliss.py names them).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "helirec/calibration/scan.hpp"
#include "ndarray_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using helirec::py::new_array_t;
namespace hc = helirec::calibration;

namespace {

// a NumPy copy (as an object: the properties' reference_internal policy does not apply to an array that owns its memory)
template <class T>
nb::object arr(const std::vector<T>& v) {
    auto a = new_array_t<T>({v.size()});
    std::copy(v.begin(), v.end(), static_cast<T*>(a.data()));
    return nb::cast(a, nb::rv_policy::move);
}

std::vector<hc::Scan> scans_of(const nb::list& scans) {
    std::vector<hc::Scan> out;
    for (nb::handle s : scans) out.push_back(nb::cast<const hc::Scan&>(s));
    return out;
}

}  // namespace

void bind_scan(nb::module_& m) {
    nb::class_<hc::FrameList>(m, "FrameList",
                              "Where a sequence of frames is stored: frame i is frame frame_index[i] of `dataset` in files[file_index[i]] "
                              "(helirec::calibration::FrameList)")
        .def_ro("files", &hc::FrameList::files)
        .def_ro("dataset", &hc::FrameList::dataset)
        .def_prop_ro("file_index", [](const hc::FrameList& f) { return arr(f.file_index); })
        .def_prop_ro("frame_index", [](const hc::FrameList& f) { return arr(f.frame_index); })
        .def("__len__", &hc::FrameList::size);

    nb::class_<hc::Scan>(m, "Scan",
                         "A BLISS tomo scan as the reconstruction reads it, derived as nxtomomill h52nx and nx2etf derive it "
                         "(helirec::calibration::Scan)")
        .def_ro("name", &hc::Scan::name)
        .def_ro("master", &hc::Scan::master)
        .def_ro("directory", &hc::Scan::directory)
        .def_prop_ro("frame_shape", [](const hc::Scan& s) { return nb::make_tuple(s.rows, s.cols); })
        .def_ro("dtype", &hc::Scan::dtype)
        .def_ro("projections", &hc::Scan::projections)
        .def_ro("darks", &hc::Scan::darks)
        .def_ro("flats", &hc::Scan::flats)
        .def_prop_ro("n_proj", [](const hc::Scan& s) { return s.projections.size(); })
        .def_prop_ro("angles_deg", [](const hc::Scan& s) { return arr(s.angles_deg); })
        .def_prop_ro("unwrapped_angles_deg", [](const hc::Scan& s) { return arr(s.unwrapped_angles_deg); })
        .def_prop_ro("z_translation_m", [](const hc::Scan& s) { return arr(s.z_translation_m); })
        .def_prop_ro("control_A", [](const hc::Scan& s) { return arr(s.control_A); })
        .def_prop_ro("flats_control_A", [](const hc::Scan& s) {
            nb::list out;
            for (const auto& c : s.flats_control_A) out.append(arr(c));
            return out;
        })
        .def_prop_ro("flats_unwrapped_angles_deg", [](const hc::Scan& s) { return arr(s.flats_unwrapped_angles_deg); })
        .def_prop_ro("darks_control_A", [](const hc::Scan& s) { return arr(s.darks_control_A); })
        .def_ro("y_translation_m", &hc::Scan::y_translation_m)
        .def_ro("estimated_cor_from_motor_pix", &hc::Scan::estimated_cor_from_motor_pix)
        .def_ro("energy_kev", &hc::Scan::energy_kev)
        .def_ro("distance_m", &hc::Scan::distance_m)
        .def_ro("pixel_size_m", &hc::Scan::pixel_size_m)
        .def_ro("detector_pixel_size_m", &hc::Scan::detector_pixel_size_m)
        .def_ro("count_time_s", &hc::Scan::count_time_s)
        .def_ro("exposure_arc", &hc::Scan::exposure_arc)
        .def_ro("detector", &hc::Scan::detector)
        .def_ro("sample_name", &hc::Scan::sample_name)
        .def_ro("start_time", &hc::Scan::start_time)
        .def_ro("end_time", &hc::Scan::end_time)
        .def_prop_ro("entries", [](const hc::Scan& s) {
            nb::dict out;
            for (const auto& e : s.entries) {
                nb::dict d;
                d["role"] = e.role, d["title"] = e.title, d["n_frames"] = e.n_frames;
                out[nb::str(e.name.c_str())] = d;
            }
            return out;
        })
        .def("referenced_files", &hc::Scan::referenced_files)
        .def("check_files", &hc::Scan::check_files)
        .def("__repr__", [](const hc::Scan& s) {
            return "<Scan " + s.name + ": " + std::to_string(s.projections.size()) + " projections, " + std::to_string(s.darks.size()) + " darks, " +
                   std::to_string(s.flats.size()) + " flat series, frames " + std::to_string(s.rows) + " x " + std::to_string(s.cols) + ">";
        });

    m.def("open_scan", &hc::open_scan, "path"_a, nb::call_guard<nb::gil_scoped_release>(),
          "A BLISS tomo scan: `path` is the master file, its scan directory <name>_0001, or a directory holding one such scan");
    m.def("discover_scans", &hc::discover_scans, "root"_a, "max_depth"_a = 3, "require_files"_a = false, nb::call_guard<nb::gil_scoped_release>(),
          "The tomo scans below `root`, sorted by name (masters that cannot be read as a scan are left out)");
    m.def(
        "parse_tetra_name",
        [](const std::string& name) -> nb::object {
            const auto t = hc::parse_tetra_name(name);
            if (!t) return nb::none();
            nb::dict d;
            d["base"] = t->base, d["kind"] = t->kind, d["rank"] = t->rank, d["suffix"] = t->suffix, d["crash"] = t->crash;
            return d;
        },
        "name"_a, "The parts of an extended field of view's scan name <base>_<HA|TA|TA2..>[_fc_|_cs_]_NNNN[_NNNN], or None");
    m.def(
        "group_companions",
        [](const nb::list& scans) -> nb::list {
            nb::list out;
            for (const auto& g : hc::group_companions(scans_of(scans))) {
                nb::list family;
                for (int i : g) family.append(scans[static_cast<std::size_t>(i)]);
                out.append(family);
            }
            return out;
        },
        "scans"_a, "The families of an extended field of view among `scans`, each the main scan first");
    m.def(
        "pair_companions",
        [](const nb::list& scans) -> nb::dict {
            const auto g = hc::pair_companions(scans_of(scans));
            nb::list companions;
            for (std::size_t k = 1; k < g.size(); ++k) companions.append(scans[static_cast<std::size_t>(g[k])]);
            nb::dict d;
            d["main"] = scans[static_cast<std::size_t>(g[0])], d["companion"] = companions[0], d["companions"] = companions;
            return d;
        },
        "scans"_a, "The one family of an extended field of view among `scans`: {main, companion, companions}");
}
