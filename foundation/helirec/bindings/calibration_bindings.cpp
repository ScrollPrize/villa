// nanobind bindings of helirec::calibration (helirec._core.calibration_*): the bottom-up verification of an
// acquisition record against itself and against the raw frames. The work is in C++ (helirec/calibration/*.hpp).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <limits>
#include "fit_summary.hpp"
#include <memory>

#include "helirec/calibration/registration.hpp"
#include "helirec/calibration/solve.hpp"
#include "helirec/calibration/record_checks.hpp"
#include "helirec/calibration/session.hpp"
#include "helirec/calibration/refill.hpp"
#include "helirec/raw_reader.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace helirec;
using namespace helirec::calibration;

namespace {

using Block = nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

// "cpu", "cuda", or "auto": CUDA when the build has it and a GPU is usable
Device device_from(const std::string& name) {
    if (name == "cpu") return Device::CPU;
    if (name == "cuda") return Device::CUDA;
    if (name == "auto") return cuda_available() ? Device::CUDA : Device::CPU;
    throw Error("device is 'cpu', 'cuda' or 'auto'");
}

std::string check_record_json(const std::vector<std::string>& scans, const std::string& checks, const std::string& out_dir, const std::string& inventory_dir,
                        int frames, int rows, int blocks, int threads, const std::string& device, double position_target, int columns, double column_fraction) {
    RecordCheckOptions opt;
    opt.view.device = device_from(device);
    opt.checks = checks;
    opt.out_dir = out_dir;
    opt.inventory_dir = inventory_dir;
    opt.threads = threads;
    opt.view.frames = frames;
    opt.view.rows = rows;
    opt.view.blocks = blocks;
    opt.view.eps = position_target;
    opt.view.columns = columns;
    opt.view.column_fraction = column_fraction;
    nb::gil_scoped_release nogil;
    return run_record_checks(scans, opt).json();
}

nb::dict registration_dict(const Registration& r) {
    nb::dict out;
    out["solver"] = fit_summary_dict(r.solver);
    out["function_evaluations"] = r.function_evaluations;
    nb::list curvature;
    for (int i = 0; i < 3; ++i) curvature.append(nb::make_tuple(r.curvature[i][0], r.curvature[i][1], r.curvature[i][2]));
    out["curvature"] = curvature;
    out["shift"] = nb::make_tuple(r.shift[0], r.shift[1], r.shift[2]);
    out["whole"] = nb::make_tuple(r.whole[0], r.whole[1], r.whole[2]);
    out["correlation"] = r.correlation;
    out["at_edge"] = r.at_edge;
    out["converged"] = r.converged;
    out["evaluations"] = r.evaluations;
    return out;
}

// A registration geometry kept for many pairs of blocks (Registrar): plans, band weights and work memory made once.
struct PyRegistrar {
    int n[3], m[3];
    std::unique_ptr<Registrar> reg;
    PyRegistrar(const std::vector<int>& shape, const std::vector<int>& margin, double f_lo, double f_hi, int stripe_notch, const std::vector<int>& search,
                const std::string& device) {
        if (shape.size() != 3 || margin.size() != 3 || (!search.empty() && search.size() != 3)) throw Error("CalibrationRegistrar: shape, margin and search have three values");
        RegistrationOptions opt;
        opt.f_lo = f_lo;
        opt.f_hi = f_hi;
        opt.stripe_notch = stripe_notch;
        for (int j = 0; j < 3; ++j) {
            n[j] = shape[j];
            m[j] = margin[j];
            if (!search.empty()) opt.search[j] = search[j];
        }
        reg = std::make_unique<Registrar>(n, m, opt, device_from(device));
    }
    nb::dict run(Block a, Block b, const std::vector<double>& row_shift) {
        for (int j = 0; j < 3; ++j)
            if (static_cast<int>(a.shape(j)) != n[j] || static_cast<int>(b.shape(j)) != n[j] + 2 * m[j])
                throw Error("CalibrationRegistrar.run: a has the registrar's shape, b that shape plus the margin on both sides");
        if (!row_shift.empty() && row_shift.size() != b.shape(0)) throw Error("CalibrationRegistrar.run: row_shift has one value per frame of b");
        Registration r;
        {
            nb::gil_scoped_release nogil;
            r = reg->run(a.data(), b.data(), row_shift.empty() ? nullptr : row_shift.data());
        }
        return registration_dict(r);
    }
};

template <class T, class V>
nb::ndarray<nb::numpy, T> array_from(const V& v, std::initializer_list<std::size_t> shape) {
    std::size_t total = 1;
    for (std::size_t s : shape) total *= s;
    if (total != v.size()) throw Error("calibration: array size does not match its shape");
    T* data = new T[total ? total : 1];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.begin(), owner);
}
template <class V>
nb::ndarray<nb::numpy, double> vector_array(const V& v) {
    return array_from<double>(v, {v.size()});
}

using FrameList = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using RowList = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// The scans of one acquisition kept open (Session): what the record states per frame, the raw frames, the dark
// and mean flat, and comparisons of blocks placed by the caller.
struct PySession {
    std::unique_ptr<Session> s;
    PySession(const std::vector<std::string>& scans, int threads) {
        nb::gil_scoped_release nogil;
        s = std::make_unique<Session>(scans, threads);
    }
    const Stretch& stretch(int scan, int index) const { return s->record(scan).stretches.at(static_cast<std::size_t>(index)); }

    nb::list stretches(int scan) const {
        nb::list out;
        const Record& rec = s->record(scan);
        for (std::size_t i = 0; i < rec.stretches.size(); ++i) {
            const Stretch& st = rec.stretches[i];
            nb::dict d;
            d["index"] = static_cast<int>(i);
            d["entry"] = st.entry;
            d["role"] = std::string(role_name(st.role));
            d["n_frames"] = st.n_frames;
            d["latched"] = st.latched();
            d["helical"] = !st.z_center.empty();
            d["title"] = st.title;
            // the frames whose files are on disk, as [first, end) ranges: no other frame can be read
            nb::list held;
            for (const auto& [a, b] : st.frames_on_disk()) held.append(nb::make_tuple(a, b));
            d["on_disk"] = held;
            d["n_on_disk"] = st.n_on_disk();
            out.append(d);
        }
        return out;
    }
    // per-frame (or per-reading) series of a stretch, as the record states them
    nb::ndarray<nb::numpy, double> series(int scan, int index, const std::string& name) const {
        const Stretch& st = stretch(scan, index);
        if (name == "angle_deg") return vector_array(stated_angle(st));
        if (name == "height_mm") return vector_array(stated_height(s->record(scan), st));
        if (name == "epoch_trig") return vector_array(st.epoch_trig);
        if (name == "timer_trig") return vector_array(st.timer_trig);
        if (name == "timer_delta") return vector_array(st.timer_delta);
        if (name == "timer_period") return vector_array(st.timer_period);
        if (name == "rot_trig") return vector_array(st.rot_trig);
        if (name == "rot_delta") return vector_array(st.rot_delta);
        if (name == "z_trig") return vector_array(st.z_trig);
        if (name == "z_center") return vector_array(st.z_center);
        if (name == "z_delta") return vector_array(st.z_delta);
        if (name == "epoch") return vector_array(st.epoch);
        if (name == "time_of_frame") return vector_array(st.files.time_of_frame);
        if (name == "stat_avg") return vector_array(st.stat_avg);
        if (name == "monitor_epoch") return vector_array(st.monitor.epoch);
        if (name == "monitor_current") return vector_array(st.monitor.current);
        throw Error("calibration session: no series named " + name);
    }
    double positioner_of(int scan, int index, const std::string& name, bool end) const { return positioner(stretch(scan, index), name, end); }
    nb::dict info(int scan) const {
        const Record& rec = s->record(scan);
        nb::dict d;
        d["label"] = rec.label;
        d["path"] = rec.path;
        d["rows"] = s->frames(scan).rows();
        d["cols"] = s->frames(scan).cols();
        d["pixel_um"] = pixel_um(rec);
        d["frames_per_turn"] = rec.technique.number("scan/tomo_n");
        d["rotation"] = rec.rotation[1];
        d["translation_z"] = rec.translation_z[1];
        d["translation_y"] = rec.translation_y[1];
        // the sample's stages on the rotation, where the tomo configuration names them ("" otherwise)
        const auto alias = [&](const char* key) {
            const auto& v = rec.technique.texts(std::string("tomoconfig/") + key);
            return v.empty() ? std::string() : v.back();
        };
        d["sample_u"] = alias("sample_u");
        d["sample_v"] = alias("sample_v");
        return d;
    }
    double number(int scan, const std::string& key) const { return s->record(scan).technique.number(key); }
    void set_cache(double gigabytes) {
        for (std::size_t k = 0; k < s->size(); ++k) s->frames(static_cast<int>(k)).set_cache_bytes(static_cast<std::size_t>(std::max(0.0, gigabytes) * (1u << 30)));
    }
    nb::dict cache_counts() {
        std::uint64_t hits = 0, decoded = 0;
        for (std::size_t k = 0; k < s->size(); ++k) {
            std::uint64_t h = 0, d = 0;
            s->frames(static_cast<int>(k)).cache_counts(h, d);
            hits += h, decoded += d;
        }
        nb::dict d;
        d["hits"] = hits;
        d["decoded"] = decoded;
        return d;
    }

    nb::ndarray<nb::numpy, std::uint16_t> read_rows(int scan, int index, FrameList frames, int row0, int nrows) {
        Frames& f = s->frames(scan);
        const std::size_t n = frames.shape(0), cols = static_cast<std::size_t>(f.cols());
        std::vector<std::int32_t> r0(n, row0);
        auto* data = new std::uint16_t[std::max<std::size_t>(1, n * static_cast<std::size_t>(nrows) * cols)];
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<std::uint16_t*>(p); });
        {
            nb::gil_scoped_release nogil;
            f.read_rows(index, frames.data(), r0.data(), n, nrows, data);
        }
        const std::size_t shape[3] = {n, static_cast<std::size_t>(nrows), cols};
        return nb::ndarray<nb::numpy, std::uint16_t>(data, 3, shape, owner);
    }
    // -ln((raw - dark) / mean flat) of rows [row0[i], row0[i] + nrows) x columns [col0, col0 + ncols) of the frames
    nb::ndarray<nb::numpy, float> line_integrals_of(int scan, int index, FrameList frames, RowList row0, int nrows, int col0, int ncols, bool flip) {
        Frames& f = s->frames(scan);
        const std::size_t n = frames.shape(0);
        if (row0.shape(0) != n) throw Error("calibration session: one first row per frame");
        auto* data = new float[std::max<std::size_t>(1, n * static_cast<std::size_t>(nrows) * static_cast<std::size_t>(ncols))];
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        {
            nb::gil_scoped_release nogil;
            const FlatField& ff = s->flat_field(scan);
            std::vector<std::uint16_t> raw(n * static_cast<std::size_t>(nrows) * static_cast<std::size_t>(f.cols()));
            f.read_rows(index, frames.data(), row0.data(), n, nrows, raw.data());
            line_integrals(ff, raw.data(), row0.data(), n, nrows, col0, ncols, flip, data);
        }
        const std::size_t shape[3] = {n, static_cast<std::size_t>(nrows), static_cast<std::size_t>(ncols)};
        return nb::ndarray<nb::numpy, float>(data, 3, shape, owner);
    }
    // the same on the ideal grid of the session's detector map (one first row for all frames)
    nb::ndarray<nb::numpy, float> mapped_integrals(int scan, int index, FrameList frames, int row0, int nrows, int col0, int ncols, bool flip, double frame_shear) {
        const std::size_t n = frames.shape(0);
        auto* data = new float[std::max<std::size_t>(1, n * static_cast<std::size_t>(nrows) * static_cast<std::size_t>(ncols))];
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        {
            nb::gil_scoped_release nogil;
            line_integrals_mapped(s->frames(scan), s->flat_field(scan), s->detector_map(scan), index, frames.data(), row0, n, nrows, col0, ncols, flip, data);
            shear_frames(data, n, nrows, ncols, frame_shear);
        }
        const std::size_t shape[3] = {n, static_cast<std::size_t>(nrows), static_cast<std::size_t>(ncols)};
        return nb::ndarray<nb::numpy, float>(data, 3, shape, owner);
    }
    void set_map(const std::vector<int>& i, const std::vector<int>& j, const std::vector<double>& eu, const std::vector<double>& ev, double c0, double cs, double v0, double vs, int scan) {
        DetectorMap m;
        m.i = i, m.j = j, m.eu = eu, m.ev = ev;
        m.c0 = c0, m.cs = cs, m.v0 = v0, m.vs = vs;
        s->set_detector_map(m, scan);
    }
    nb::ndarray<nb::numpy, float> flat(int scan, bool dark) {
        const FlatField* ff;
        {
            nb::gil_scoped_release nogil;
            ff = &s->flat_field(scan);
        }
        return array_from<float>(dark ? ff->dark : ff->gain, {static_cast<std::size_t>(ff->rows), static_cast<std::size_t>(ff->cols)});
    }

    static std::vector<ViewRequest> requests_from(const nb::list& requests) {
        std::vector<ViewRequest> req;
        for (nb::handle h : requests) {
            const nb::dict d = nb::cast<nb::dict>(h);
            ViewRequest q;
            const auto get_int = [&](const char* key, std::int64_t fallback) { return d.contains(key) ? nb::cast<std::int64_t>(d[key]) : fallback; };
            q.label = d.contains("label") ? nb::cast<std::string>(d["label"]) : std::string();
            q.scan_a = static_cast<int>(get_int("scan_a", 0));
            q.scan_b = static_cast<int>(get_int("scan_b", q.scan_a));
            q.stretch_a = static_cast<int>(nb::cast<std::int64_t>(d["stretch_a"]));
            q.stretch_b = static_cast<int>(get_int("stretch_b", q.stretch_a));
            q.frame_a = nb::cast<std::int64_t>(d["frame_a"]);
            q.frame_hint = get_int("frame_hint", 0);
            q.angle_offset_deg = d.contains("angle_offset_deg") ? nb::cast<double>(d["angle_offset_deg"]) : 0.0;
            q.mirror = d.contains("mirror") && nb::cast<bool>(d["mirror"]);
            q.frames = static_cast<int>(get_int("frames", 32));
            q.rows = static_cast<int>(get_int("rows", 256));
            q.cols = static_cast<int>(get_int("cols", 512));
            q.row_a = static_cast<int>(get_int("row_a", -1));
            q.col_a = static_cast<int>(get_int("col_a", 0));
            q.col_b = static_cast<int>(get_int("col_b", q.col_a));
            q.frame_shear = d.contains("frame_shear") ? nb::cast<double>(d["frame_shear"]) : 0.0;
            q.sway_px = d.contains("sway_px") ? nb::cast<double>(d["sway_px"]) : 0.0;
            q.sway_phase_deg = d.contains("sway_phase_deg") ? nb::cast<double>(d["sway_phase_deg"]) : 0.0;
            q.sway_row_deg = d.contains("sway_row_deg") ? nb::cast<double>(d["sway_row_deg"]) : 0.0;
            req.push_back(q);
        }
        return req;
    }
    static nb::dict measurement_dict(const ViewMeasurement& r) {
        nb::dict d;
        d["label"] = r.label;
        d["planned"] = r.planned;
        d["ok"] = r.ok;
        d["frame_a"] = r.frame_a;
        d["frame_b"] = r.frame_b;
        d["frames"] = r.frames;
        d["row_a"] = r.row_a;
        d["rows"] = r.rows;
        d["row_offset"] = r.row_offset;
        d["col_a"] = r.col_a;
        d["col_b"] = r.col_b;
        d["cols"] = r.cols;
        d["stated_dq"] = r.stated_dq;
        d["dq"] = r.dq;
        d["stated_dv"] = r.stated_dv;
        d["dv"] = r.dv;
        d["stated_D"] = r.stated_D;
        d["D"] = r.D;
        d["du"] = r.du;
        d["correlation"] = r.correlation;
        d["evaluations"] = r.evaluations;
        d["function_evaluations"] = r.function_evaluations;
        d["solver"] = fit_summary_dict(r.solver);
        d["slope_a"] = r.slope_a;
        d["slope_b"] = r.slope_b;
        d["step_deg"] = r.step_deg;
        d["angle_a_deg"] = r.angle_a_deg;
        return d;
    }

    nb::list measure(const nb::list& requests, double row_sign, const std::string& device) {
        const std::vector<ViewRequest> req = requests_from(requests);
        std::vector<ViewMeasurement> res;
        {
            const Device dev = device_from(device);
            nb::gil_scoped_release nogil;
            res = s->measure(req, row_sign, dev);
        }
        nb::list out;
        for (const ViewMeasurement& r : res) out.append(measurement_dict(r));
        return out;
    }

    nb::list measure_patches(const nb::list& requests, double row_sign, int patch_rows, int patch_cols, const std::string& device, bool from_block) {
        const std::vector<ViewRequest> req = requests_from(requests);
        std::vector<PatchMeasurement> res;
        {
            const Device dev = device_from(device);
            nb::gil_scoped_release nogil;
            res = s->measure_patches(req, patch_rows, patch_cols, row_sign, dev, from_block);
        }
        nb::list out;
        for (const PatchMeasurement& r : res) {
            nb::dict d = measurement_dict(r.m);
            d["request"] = r.request;
            d["patch_i"] = r.i;
            d["patch_j"] = r.j;
            d["curvature"] = nb::make_tuple(r.curvature[0], r.curvature[1], r.curvature[2], r.curvature[3], r.curvature[4], r.curvature[5]);
            out.append(d);
        }
        return out;
    }
};

nb::dict register_py(Block a, Block b, const std::vector<int>& margin, const std::vector<double>& row_shift, double f_lo, double f_hi,
                     const std::string& device) {
    const Device dev = device_from(device);
    if (margin.size() != 3) throw Error("calibration_register_blocks: margin has three values");
    const int n[3] = {static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)), static_cast<int>(a.shape(2))};
    const int m[3] = {margin[0], margin[1], margin[2]};
    for (int j = 0; j < 3; ++j)
        if (static_cast<int>(b.shape(j)) != n[j] + 2 * m[j]) throw Error("calibration_register_blocks: b must be a's shape plus the margin on both sides");
    if (!row_shift.empty() && row_shift.size() != b.shape(0)) throw Error("calibration_register_blocks: row_shift has one value per frame of b");
    RegistrationOptions opt;
    opt.f_lo = f_lo;
    opt.f_hi = f_hi;
    Registration r;
    {
        nb::gil_scoped_release nogil;
        r = register_blocks(a.data(), b.data(), n, m, row_shift.empty() ? nullptr : row_shift.data(), opt, dev);
    }
    return registration_dict(r);
}

}  // namespace

void bind_calibration(nb::module_& m) {
    m.def("calibration_check_record", &check_record_json, "scans"_a, "checks"_a = "record,inventory,series,beam,views", "out_dir"_a = "", "inventory_dir"_a = "",
          "frames"_a = 32, "rows"_a = 256, "blocks"_a = 8, "threads"_a = 0, "device"_a = "auto", "position_target"_a = default_position_target_px, "columns"_a = 0, "column_fraction"_a = 0.5,
          "Verify what the acquisition record of one scan, or of the two scans of one acquisition (axis inside the field first), states: against itself "
          "(checks 'record') and against the raw frames ('inventory', 'series', 'beam', 'views'). Returns the report as JSON: a list of findings, each with "
          "the requirement it bears on, what the record states, what was found and a verdict (verified, fails, unverified, note). Any sampled residual above "
          "the position target fails; smaller residuals still require follow-up. `device` ('cpu', 'cuda', 'auto') "
          "is where the blocks of the view checks are registered.");
    m.def("calibration_assess_residuals", [](const std::vector<double>& residuals, double target, const std::string& unit) {
        Report report;
        report.position_target_px = unit == "px" ? target : std::numeric_limits<double>::quiet_NaN();
        auto& finding = report.add("", "", "residuals against a stated model");
        assess_residuals(finding, residuals, target, unit);
        return report.json();
    }, "residuals"_a, "target"_a = default_position_target_px, "unit"_a = "px",
       "Assess every supplied residual. Returns report JSON; block SEM is descriptive, never an uncertainty discount or a proof of agreement.");
    m.def("calibration_register_blocks", &register_py, "a"_a, "b"_a, "margin"_a, "row_shift"_a = std::vector<double>(), "f_lo"_a = 0.01, "f_hi"_a = 0.15, "device"_a = "cpu",
          "Displacement between block a (frames, rows, columns) and block b, which is a's shape plus `margin` samples on both sides of each axis: "
          "a(x) ~ b(x + margin + shift). A non-zero `whole` means b must be cut again that many samples further and the call repeated.");
    m.def("calibration_selftest", [](bool verbose, const std::string& device) { return registration_selftest(verbose, device_from(device)); },
          "verbose"_a = false, "device"_a = "cpu",
          "Synthetic checks of the registration and of the conventions of the view comparisons; True when all pass.");
    nb::class_<PyRegistrar>(m, "CalibrationRegistrar",
                            "One registration geometry kept for many pairs of blocks: a of `shape` (frames, rows, columns), b of that shape plus `margin` on both sides. "
                            "The estimator of calibration_register_blocks; an axis of one sample is not registered.")
        .def(nb::init<const std::vector<int>&, const std::vector<int>&, double, double, int, const std::vector<int>&, const std::string&>(), "shape"_a, "margin"_a, "f_lo"_a = 0.01,
             "f_hi"_a = 0.15, "stripe_notch"_a = 2, "search"_a = std::vector<int>(), "device"_a = "cpu")
        .def("run", &PyRegistrar::run, "a"_a, "b"_a, "row_shift"_a = std::vector<double>(), "a(x) ~ b(x + margin + shift); see calibration_register_blocks.");
    nb::class_<PySession>(m, "CalibrationSession",
                          "The scans of one acquisition kept open (axis inside the field first): what the record states per frame, the raw frames as stored, "
                          "each scan's dark and mean flat, and registrations of blocks placed by the caller. Nothing of the reconstruction's calibration is used.")
        .def(nb::init<const std::vector<std::string>&, int>(), "scans"_a, "threads"_a = 0)
        .def("info", &PySession::info, "scan"_a)
        .def("stretches", &PySession::stretches, "scan"_a)
        .def("series", &PySession::series, "scan"_a, "stretch"_a, "name"_a,
             "angle_deg (middle of exposure, cumulative), height_mm, epoch_trig, timer_trig, timer_delta, timer_period, rot_trig, rot_delta, z_trig, z_center, z_delta, "
             "epoch, time_of_frame, stat_avg, monitor_epoch, monitor_current")
        .def("positioner", &PySession::positioner_of, "scan"_a, "stretch"_a, "name"_a, "end"_a = false)
        .def("number", &PySession::number, "scan"_a, "key"_a, "A number of the sequence's technique group, e.g. scan/source_sample_distance.")
        .def("set_cache", &PySession::set_cache, "gigabytes"_a, "Decoded chunks of compressed frames kept between reads, per scan (default 3; 0: none).")
        .def("set_read_ahead", [](PySession& p, int threads) { p.s->set_read_ahead(threads); }, "threads"_a,
             "Workers per scan that decode the frames of coming comparisons ahead (compressed frames; 0: none; default a third of the machine's threads).")
        .def("cache_counts", &PySession::cache_counts, "Chunks of compressed frames taken from the cache and decoded so far: {'hits', 'decoded'}.")
        .def("read_rows", &PySession::read_rows, "scan"_a, "stretch"_a, "frames"_a, "row0"_a, "nrows"_a, "Raw values (n, nrows, cols) of rows [row0, row0 + nrows) of the frames.")
        .def("line_integrals", &PySession::line_integrals_of, "scan"_a, "stretch"_a, "frames"_a, "row0"_a, "nrows"_a, "col0"_a, "ncols"_a, "flip"_a = false,
             "-ln((raw - dark) / mean flat), (n, nrows, ncols): rows [row0[i], row0[i] + nrows) of frame i (0 outside the detector), columns reversed if flip.")
        .def("set_detector_map", &PySession::set_map, "i"_a, "j"_a, "eu"_a, "ev"_a, "c0"_a = 0.0, "cs"_a = 1.0, "v0"_a = 0.0, "vs"_a = 1.0, "scan"_a = -1,
             "A detector map to be tested: pixel (c, v) looks along the ray at (c + Eu, v + Ev), Eu = sum eu[k] x^i[k] y^j[k], x = (c - c0) / cs, "
             "y = (v - v0) / vs. From then on measure() and mapped_line_integrals() cut blocks on the map's ideal grid; empty lists take the map away. "
             "scan < 0 (the default): for every scan; otherwise for that scan alone.")
        .def("mapped_line_integrals", &PySession::mapped_integrals, "scan"_a, "stretch"_a, "frames"_a, "row0"_a, "nrows"_a, "col0"_a, "ncols"_a, "flip"_a = false, "frame_shear"_a = 0.0,
             "line_integrals on the ideal grid of the session's detector map (cubic convolution), rows [row0, row0 + nrows) of every frame; then, if "
             "frame_shear is not 0, frame q of column c read at q + frame_shear (c - (ncols - 1) / 2).")
        .def("dark", [](PySession& p, int scan) { return p.flat(scan, true); }, "scan"_a, "Mean of the scan's dark frames (rows, cols).")
        .def("gain", [](PySession& p, int scan) { return p.flat(scan, false); }, "scan"_a, "Mean of all the scan's flat frames minus the dark (rows, cols).")
        .def("measure", &PySession::measure, "requests"_a, "row_sign"_a, "device"_a = "auto",
             "Register pairs of blocks placed by the caller, exactly as the view checks cut and register theirs. Each request is a dict: stretch_a, frame_a, and "
             "optionally label, scan_a, scan_b, stretch_b, frame_hint, angle_offset_deg, mirror, frames, rows, cols, row_a (-1: centred), col_a, col_b, frame_shear, "
             "sway_px, sway_phase_deg (B's frames read sway_px cos(angle_b - sway_phase_deg) columns further on: a sample displaced on the rotation stage, tile against tile), "
             "sway_row_deg (each row of B at its own angle, that many degrees per row from the frame's: the rows' timing). "
             "Returns one dict per request with the columns of views.tsv, `planned` and `ok`. row_sign: +1 if a larger axis-stage position puts the sample on higher rows.")
        .def("measure_patches", &PySession::measure_patches, "requests"_a, "row_sign"_a, "patch_rows"_a = 128, "patch_cols"_a = 128, "device"_a = "auto", "from_block"_a = false,
             "The block of each request (as for measure) cut into patches of patch_rows x patch_cols, every patch measured as measure() measures a request of that "
             "size at that place, from one reading of the block's frames; the patches are registered side by side (one per thread; a few at a time on a CUDA device). "
             "Returns one dict per patch: measure's columns, `request` (index of its request), `patch_i`, `patch_j` (column and row index in the block; a mirrored "
             "B's patch i lies at col_b + (cols - patch_cols) - i patch_cols; -1 for a request that cannot be planned) and `curvature`: the second derivatives of "
             "the correlation at the registration, (frames^2, frames rows, frames columns, rows^2, rows columns, columns^2).");
    m.def("calibration_solve", [](nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> row,
                                  nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> col,
                                  nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> value,
                                  nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> y, std::size_t unknowns, double huber, int max_iterations, double rcond,
                                  const std::vector<std::int64_t>& fixed, const std::vector<double>& fixed_value,
                                  double function_tolerance, double gradient_tolerance, double parameter_tolerance) {
        LinearProblem p;
        p.unknowns = unknowns;
        p.row.assign(row.data(), row.data() + row.shape(0));
        p.col.assign(col.data(), col.data() + col.shape(0));
        p.value.assign(value.data(), value.data() + value.shape(0));
        p.y.assign(y.data(), y.data() + y.shape(0));
        SolveOptions o;
        o.huber = huber; o.max_iterations = max_iterations; o.rcond = rcond;
        o.function_tolerance = function_tolerance; o.gradient_tolerance = gradient_tolerance; o.parameter_tolerance = parameter_tolerance;
        p.fixed = fixed; p.fixed_value = fixed_value;
        LinearSolution s;
        {
            nb::gil_scoped_release nogil;
            s = solve_linear(p, o);
        }
        nb::dict d = fit_summary_dict(s);
        d["active"] = s.active;
        d["unused"] = s.unused;
        d["x"] = vector_array(s.x);
        d["residual"] = vector_array(s.residual);
        d["weight"] = vector_array(s.weight);
        d["spread"] = s.spread;
        d["rank"] = s.rank;
        return d;
    }, "row"_a, "col"_a, "value"_a, "y"_a, "unknowns"_a, "huber"_a = 0.0, "max_iterations"_a = 100, "rcond"_a = 1e-12,
       "fixed"_a = std::vector<std::int64_t>(), "fixed_value"_a = std::vector<double>(),
       "function_tolerance"_a = 1e-7, "gradient_tolerance"_a = 1e-12, "parameter_tolerance"_a = 1e-12,
       "Ceres scalar Huber fit of y = A x; duplicate entries add. Fixed indices have explicit values; unused columns are frozen. "
       "Returns parameters, y - prediction, final sqrt robust weights, rank and Ceres diagnostics; check usable before using a fit.");
    m.def("calibration_fit_refill_shots", [](nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu> x,
                                             double T, double E, int pre) {
        RefillSolution s;
        { nb::gil_scoped_release nogil; s = fit_refill_shots(x.data(), x.shape(0), static_cast<int>(x.shape(1)), T, E, pre); }
        auto d = fit_summary_dict(s);
        d["x"] = vector_array(s.x); d["residual"] = vector_array(s.residual); d["shots"] = nb::cast(s.shots);
        d["noise"] = s.noise; d["row_time_se"] = s.row_time_se; d["uncertainty_available"] = s.uncertainty_available;
        return d;
    }, "x"_a, "period"_a, "exposure"_a, "pre"_a, "Ceres exposure-integrated refill fit of (frames, rows) logarithmic levels, with covariance diagnostics.");
    m.def("calibration_fit_line", [](const std::vector<double>& x, const std::vector<double>& y) {
        LineFit s;
        { nb::gil_scoped_release nogil; s = fit_line(x, y); }
        auto d = fit_summary_dict(s);
        d["slope"] = s.slope; d["intercept"] = s.intercept; d["rms"] = s.rms;
        return d;
    }, "x"_a, "y"_a, "Centred and scaled Ceres straight-line fit; check usable before using the coefficients.");
    m.def("calibration_measure_timing", [](bool reset) {
        const MeasureTiming t = measure_timing(reset);
        nb::dict d;
        d["read"] = t.read;
        d["integrals"] = t.integrals;
        d["registration"] = t.registration;
        d["blocks"] = t.blocks;
        d["registrations"] = t.registrations;
        return d;
    }, "reset"_a = false, "Seconds the sessions' measure() and measure_patches() spent reading frames, converting them and registering since the last reset, and the blocks read and registrations run.");
    m.def("jpeg2000_frames_supported", &jpeg2000_frames_supported, "Whether this build reads frames stored as JPEG 2000 chunks (built with OpenJPEG).");
    m.def("jpeg2000_chunk", [](nb::ndarray<const std::uint16_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> values) {
        const std::vector<std::uint8_t> bytes = jpeg2000_chunk(values.data(), static_cast<int>(values.shape(0)), static_cast<int>(values.shape(1)));
        return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }, "values"_a, "The bytes of one HDF5 chunk for (rows, cols) uint16 values as the blosc2 filter (32026) stores JPEG 2000 frames, with a reversible codestream: for tests and synthetic records.");
}
