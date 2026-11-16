// vc_tomography/calibration/intensity.hpp: the intensity calibration of a scan pair (darks and flats, the weight maps, the
// detector's blur field, the merged frame's placement) and its caches.
#include <sys/stat.h>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>

#include "vc_tomography/calibration/intensity.hpp"
#include "vc_tomography/progress.hpp"
#include "vc_tomography/projection_store.hpp"
#include "vc_tomography/raw_reader.hpp"
#include "io/array_file.hpp"
#include "io/json.hpp"
#include "io/sha256.hpp"
#include "common/numpy_sum.hpp"
#include "io/text_file.hpp"

namespace vc_tomography::calibration {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using Log = std::function<void(const std::string&)>;
void say(const Log& log, const std::string& line) {
    if (log) log(line);
}

std::string fmt(const char* format, double v) {
    char b[64];
    std::snprintf(b, sizeof b, format, v);
    return b;
}

using io::exists;

std::string join(const std::string& dir, const std::string& name) { return dir.empty() || dir.back() == '/' ? dir + name : dir + "/" + name; }

void make_dirs(const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) throw Error("cannot make the directory " + dir + ": " + ec.message());
}

// "<absolute path>|<size>|<time of change in ns>" of a file
std::string file_state(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) throw Error("no such file: " + path);
#ifdef __APPLE__
    const long long ns = static_cast<long long>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    const long long ns = static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
    return std::filesystem::absolute(path).lexically_normal().string() + "|" + std::to_string(static_cast<long long>(st.st_size)) + "|" + std::to_string(ns);
}

template <class T>
void hash_values(Sha256& h, const std::vector<T>& v) {
    h.update(v.data(), v.size() * sizeof(T));
}

Json num(double v) { return Json::of(v); }
Json count(long long v) { return Json::whole_number(v); }
Json numbers(const std::vector<double>& v) {
    Json j = Json::list();
    for (double x : v) j.add(num(x));
    return j;
}
double number_of(const Json* j, double fallback = kNaN) { return j && j->is(Json::Number) ? j->number : fallback; }
std::vector<double> numbers_of(const Json* j) {
    std::vector<double> out;
    if (j && j->is(Json::Array))
        for (const Json& v : j->array) out.push_back(v.is(Json::Number) ? v.number : kNaN);
    return out;
}
std::string text_of(const Json* j) { return j && j->is(Json::String) ? j->string : std::string(); }

Json solver_json(const FitSummary& s) {
    return Json::dict()
        .set("cost", num(s.cost))
        .set("iterations", count(s.iterations))
        .set("evaluations", count(s.evaluations))
        .set("usable", Json::of_bool(s.usable))
        .set("termination", Json::of(s.termination))
        .set("message", Json::of(s.message))
        .set("solver_version", Json::of(s.solver_version));
}

Json blur_json(const DetectorBlur& b) {
    Json d = Json::dict();
    d.set("field", numbers(b.field));
    if (!b.was_measured) return d.set("measured", Json()).set("reason", Json::of(b.reason)).set("scan", Json::of(b.scan));
    Json tiles = Json::list().add(count(b.tiles[0])).add(count(b.tiles[1]));
    return d.set("rms_none", num(b.rms_none))
        .set("rms_fit", num(b.rms_fit))
        .set("zero", numbers({b.zero[0], b.zero[1]}))
        .set("tiles", tiles)
        .set("solver", solver_json(b.solver))
        .set("n_flats", count(b.n_flats))
        .set("max_blur_px", num(b.max_blur_px))
        .set("scan", Json::of(b.scan))
        .set("measured", numbers(b.measured));
}

DetectorBlur blur_of(const Json& cal) {
    DetectorBlur b;
    b.field = numbers_of(cal.find("detector_blur_field"));
    const Json* details = cal.find("details");
    const Json* d = details ? details->find("detector_blur") : nullptr;
    if (!d || !d->is(Json::Object)) return b;
    const Json* measured = d->find("measured");
    b.was_measured = measured && measured->is(Json::Array);
    b.measured = numbers_of(measured);
    b.rms_none = number_of(d->find("rms_none"));
    b.rms_fit = number_of(d->find("rms_fit"));
    b.max_blur_px = number_of(d->find("max_blur_px"));
    const auto zero = numbers_of(d->find("zero"));
    if (zero.size() == 2) b.zero = {zero[0], zero[1]};
    const auto tiles = numbers_of(d->find("tiles"));
    if (tiles.size() == 2) b.tiles = {static_cast<int>(tiles[0]), static_cast<int>(tiles[1])};
    b.n_flats = static_cast<int>(number_of(d->find("n_flats"), 0));
    b.scan = text_of(d->find("scan"));
    b.reason = text_of(d->find("reason"));
    if (const Json* s = d->find("solver"); s && s->is(Json::Object)) {
        b.solver.cost = number_of(s->find("cost"));
        b.solver.iterations = static_cast<int>(number_of(s->find("iterations"), 0));
        b.solver.evaluations = static_cast<int>(number_of(s->find("evaluations"), 0));
        const Json* u = s->find("usable");
        b.solver.usable = u && u->is(Json::Bool) && u->boolean;
        b.solver.termination = text_of(s->find("termination"));
        b.solver.message = text_of(s->find("message"));
        b.solver.solver_version = text_of(s->find("solver_version"));
    }
    return b;
}

std::string darks_flats_fingerprint(const Scan& scan) {
    Sha256 h;
    h.update("vc_tomography_darks_flats_v" + std::to_string(kIntensityCalibrationVersion) + "|" + std::to_string(scan.rows) + "," + std::to_string(scan.cols));
    std::vector<const FrameList*> lists = {&scan.darks};
    for (const FrameList& f : scan.flats) lists.push_back(&f);
    for (const FrameList* r : lists) {
        for (std::int32_t k : std::set<std::int32_t>(r->file_index.begin(), r->file_index.end())) h.update(file_state(r->files.at(static_cast<std::size_t>(k))));
        hash_values(h, r->file_index);
        hash_values(h, r->frame_index);
    }
    for (const auto& c : scan.flats_control_A) hash_values(h, c);
    return h.hex();
}

// 2 x 2 solve as LAPACK's dgesv does it (partial pivoting; throws when singular)
std::array<double, 2> solve_2x2(double a11, double a12, double a21, double a22, double b1, double b2) {
    if (std::fabs(a21) > std::fabs(a11)) std::swap(a11, a21), std::swap(a12, a22), std::swap(b1, b2);
    if (a11 == 0) throw Error("singular");
    const double l = a21 * (1.0 / a11);
    const double u22 = a22 - l * a12;
    if (u22 == 0) throw Error("singular");
    const double y2 = b2 - l * b1;
    const double x2 = y2 / u22;
    const double x1 = (b1 - x2 * a12) / a11;
    return {x1, x2};
}

}  // namespace

// ---------------------------------------------------------------- darks and flats

DarksFlats reduce_darks_flats(const Scan& scan, const std::function<void(std::size_t)>& progress) {
    DarksFlats out;
    out.rows = scan.rows;
    out.cols = scan.cols;
    if (!scan.darks.size()) throw Error(scan.name + ": no dark frames");
    {
        RawReader reader(scan.darks.files, scan.darks.dataset, scan.rows, scan.cols);
        out.dark = mean_frames(reader, scan.darks.file_index.data(), scan.darks.frame_index.data(), scan.darks.size(), nullptr, nullptr, 0, progress);
    }
    for (std::size_t k = 0; k < scan.flats.size(); ++k) {
        const FrameList& r = scan.flats[k];
        RawReader reader(r.files, r.dataset, scan.rows, scan.cols);
        const auto f = mean_frames(reader, r.file_index.data(), r.frame_index.data(), r.size(), out.dark.data(), scan.flats_control_A[k].data(), kNominalCurrentA, progress);
        out.flats.insert(out.flats.end(), f.begin(), f.end());
    }
    return out;
}

DarksFlats reduced_darks_flats(const Scan& scan, const std::string& cache_dir, const Log& log) {
    if (cache_dir.empty()) return reduce_darks_flats(scan);
    const std::string fp = darks_flats_fingerprint(scan);
    const std::string path = join(cache_dir, "darks_flats_" + scan.name + ".bin");
    if (exists(path)) {
        try {
            const ArrayFile a = read_array_file(path);
            const NamedArray* dark = a.find("dark");
            const NamedArray* flats = a.find("flats");
            if (text_of(a.header.find("fingerprint")) == fp && dark && flats) {
                DarksFlats out;
                out.rows = scan.rows, out.cols = scan.cols;
                out.dark = dark->values, out.flats = flats->values;
                return out;
            }
        } catch (const Error&) {
        }
        say(log, "darks and flats in " + path + ": of other files, made again");
    }
    DarksFlats out = reduce_darks_flats(scan);
    make_dirs(cache_dir);
    const std::size_t r = static_cast<std::size_t>(out.rows), c = static_cast<std::size_t>(out.cols);
    write_array_file(path, Json::dict().set("fingerprint", Json::of(fp)), {{"dark", {r, c}, out.dark}, {"flats", {out.series(), r, c}, out.flats}});
    return out;
}

// ---------------------------------------------------------------- the merged frame

MergedPlacement placement_from_vectors(const std::vector<std::array<double, 2>>& vectors, int cols) {
    double lowest = 0, highest = 0;  // of the scans' columns against the first scan's
    for (const auto& v : vectors) lowest = std::min(lowest, v[1]), highest = std::max(highest, v[1]);
    MergedPlacement out;
    out.anchor = -lowest;
    out.scans.resize(vectors.size() + 1);
    for (std::size_t j = 0; j < out.scans.size(); ++j) {
        const double d = out.anchor + (j > 0 ? vectors[j - 1][1] : 0.0);
        const int sx = static_cast<int>(std::floor(d));
        out.scans[j].shift_x = sx;
        out.scans[j].x_readjustment = static_cast<double>(static_cast<float>(sx - d));
        out.scans[j].z_readjustment = j > 0 ? vectors[j - 1][0] : 0.0;
    }
    out.width = cols + static_cast<int>(std::ceil(highest - lowest));
    return out;
}

MergedPlacement placement_from_vector(const std::array<double, 2>& vector, int cols) { return placement_from_vectors({vector}, cols); }

// ---------------------------------------------------------------- the detector's blur

FlatNoise flat_noise_spectra(const Scan& scan, const float* dark, int n_flats, int tile, std::array<int, 2> lit) {
    if (scan.flats.empty()) throw Error("the detector's blur needs flats");
    const FrameList& refs = scan.flats[0];
    const int n = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n_flats), refs.size())) / 2 * 2;
    if (n < 2) throw Error("the detector's blur needs at least two flats");
    const int rows = scan.rows, cols = scan.cols;
    FlatNoise out;
    out.n_flats = n;
    if (lit[1] <= lit[0]) lit = {0, rows};
    if (lit[0] < 0 || lit[1] > rows) throw Error("the detector's blur: the lit rows lie outside the detector");
    for (int y = lit[0] + ((lit[1] - lit[0]) % tile) / 2; y <= lit[1] - tile; y += tile) out.ys.push_back(y);
    for (int x = (cols % tile) / 2; x <= cols - tile; x += tile) out.xs.push_back(x);
    const std::size_t per = static_cast<std::size_t>(tile) * static_cast<std::size_t>(tile / 2 + 1);
    const std::size_t cells = out.ys.size() * out.xs.size() * per;
    out.power.assign(cells, 0.0);
    std::vector<double> batch_power(cells);
    const std::size_t px = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    RawReader reader(refs.files, refs.dataset, rows, cols);
    const int batch = 40;
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(batch) * px);
    std::vector<float> fl(static_cast<std::size_t>(batch) * px), diff(static_cast<std::size_t>(batch / 2) * px);
    for (int k0 = 0; k0 < n; k0 += batch) {
        const int m = std::min(n, k0 + batch) - k0;
        reader.read_frames(refs.file_index.data() + k0, refs.frame_index.data() + k0, static_cast<std::size_t>(m), raw.data());
#pragma omp parallel for schedule(static)
        for (int k = 0; k < m; ++k) {
            float* f = fl.data() + static_cast<std::size_t>(k) * px;
            const std::uint16_t* r = raw.data() + static_cast<std::size_t>(k) * px;
            for (std::size_t i = 0; i < px; ++i) f[i] = static_cast<float>(r[i]) - dark[i];
            const float mean = pairwise_sum(f, px) / static_cast<float>(px);
            for (std::size_t i = 0; i < px; ++i) f[i] /= mean;
        }
        for (int k = 0; k < m / 2; ++k) {
            const float* a = fl.data() + static_cast<std::size_t>(2 * k) * px;
            const float* b = a + px;
            float* d = diff.data() + static_cast<std::size_t>(k) * px;
            for (std::size_t i = 0; i < px; ++i) d[i] = a[i] - b[i];
        }
        std::fill(batch_power.begin(), batch_power.end(), 0.0);
        tile_power_spectra(diff.data(), static_cast<std::size_t>(m / 2), static_cast<std::size_t>(rows), static_cast<std::size_t>(cols), tile, out.ys.data(),
                           static_cast<int>(out.ys.size()), out.xs.data(), static_cast<int>(out.xs.size()), batch_power.data());
        for (std::size_t i = 0; i < cells; ++i) out.power[i] += batch_power[i];
    }
    return out;
}

DetectorBlur measure_detector_blur(const Scan& scan, const float* dark, int n_flats, const Log& log, std::array<int, 2> lit) {
    const auto t0 = std::chrono::steady_clock::now();
    const int rows = scan.rows, cols = scan.cols;
    if (lit[1] <= lit[0]) lit = {0, rows};
    DetectorBlur res;
    res.scan = scan.name;
    const int n_have = scan.flats.empty() ? 0 : static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n_flats), scan.flats[0].size()));
    if (n_have < kBlurMinFlats || ((lit[1] - lit[0]) / kBlurTile) * (cols / kBlurTile) < 6) {
        res.reason = n_have < kBlurMinFlats ? std::to_string(n_have) + " flats in the first series"
                                            : "a detector of " + std::to_string(lit[1] - lit[0]) + " x " + std::to_string(cols) + " holds too few tiles of " + std::to_string(kBlurTile);
        say(log, "detector blur: not measured (" + res.reason + ")");
        return res;
    }
    const FlatNoise noise = flat_noise_spectra(scan, dark, n_flats, kBlurTile, lit);
    const DetectorBlurField fit = detector_blur_field_fit(noise.power.data(), static_cast<int>(noise.ys.size()), static_cast<int>(noise.xs.size()), kBlurTile,
                                                          noise.ys.data(), noise.xs.data(), static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    const double* th = fit.field;
    res.was_measured = true;
    res.measured.assign(th, th + 6);
    res.rms_none = fit.rms_none;
    res.rms_fit = fit.rms_fit;
    res.solver = fit.solver;
    res.tiles = {static_cast<int>(noise.ys.size()), static_cast<int>(noise.xs.size())};
    res.n_flats = noise.n_flats;
    try {
        const auto x = solve_2x2(th[0], th[1], th[2], th[3], th[4], th[5]);
        res.zero = {-x[0] * 1000 + 0.5 * (cols - 1), -x[1] * 1000 + 0.5 * (rows - 1)};
    } catch (const Error&) {
        res.zero = {kNaN, kNaN};
    }
    double reach = 0;
    for (double x : {-0.5 * cols / 1000, 0.5 * cols / 1000})
        for (double y : {-0.5 * rows / 1000, 0.5 * rows / 1000}) reach = std::max(reach, std::fabs(th[0] * x + th[1] * y + th[4]));
    res.max_blur_px = reach;
    const bool ok = res.rms_fit < kBlurMinGain * res.rms_none && reach >= kBlurMinPx;
    if (ok) res.field = res.measured;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    say(log, "detector blur from the noise of " + std::to_string(res.n_flats) + " flats of " + scan.name + ": " +
                 (ok ? fmt("%.1f", reach) + " px along the columns at the worst corner, none at column " + fmt("%.0f", res.zero[0]) +
                           " (residual of the noise spectra " + fmt("%.3f", res.rms_none) + " -> " + fmt("%.3f", res.rms_fit) + ")"
                     : "nothing significant (residual " + fmt("%.3f", res.rms_none) + " -> " + fmt("%.3f", res.rms_fit) + ", reach " + fmt("%.2f", reach) + " px)") +
                 "  [" + fmt("%.0f", seconds) + "s]");
    return res;
}

// ---------------------------------------------------------------- the pair

std::string acquisition_fingerprint(const Scan& main, const Scan& companion) { return acquisition_fingerprint(std::vector<const Scan*>{&main, &companion}); }

std::string acquisition_fingerprint(const std::vector<const Scan*>& scans) {
    Sha256 h;
    h.update("vc_tomography_calibration_v" + std::to_string(kIntensityCalibrationVersion));
    for (const Scan* sc : scans) {
        Json flats = Json::list();
        for (const FrameList& f : sc->flats) flats.add(count(static_cast<long long>(f.size())));
        const Json meta = Json::dict()
                              .set("name", Json::of(sc->name))
                              .set("master", Json::of(sc->master))
                              .set("shape", Json::list().add(count(sc->rows)).add(count(sc->cols)))
                              .set("dtype", Json::of(sc->dtype))
                              .set("n_proj", count(static_cast<long long>(sc->projections.size())))
                              .set("n_darks", count(static_cast<long long>(sc->darks.size())))
                              .set("n_flats", flats)
                              .set("energy_kev", num(sc->energy_kev))
                              .set("distance_m", num(sc->distance_m))
                              .set("pixel_size_m", num(sc->pixel_size_m))
                              .set("est_cor", num(sc->estimated_cor_from_motor_pix));
        h.update(meta.text());
        for (const auto* v : {&sc->unwrapped_angles_deg, &sc->z_translation_m, &sc->control_A, &sc->flats_unwrapped_angles_deg}) hash_values(h, *v);
        h.update(file_state(sc->master));
        for (const std::string& f : sc->referenced_files()) h.update(file_state(f));
    }
    return h.hex();
}

DarksFlats store_darks_flats(const Scan& scan, const std::string& tile_group, int reduce) {
    int rows = 0, cols = 0;
    // the mean of a list's frames, the frames of one scan group at a time
    const auto mean_of = [&](const FrameList& list, const float* dark, const std::vector<double>* control) {
        if (!list.size()) throw Error(scan.name + ": no dark or flat frames");
        const FrameList in_store = frames_in_projection_store(list, tile_group);
        std::vector<double> sum;
        std::size_t i0 = 0;
        while (i0 < in_store.size()) {
            std::size_t i1 = i0;
            std::vector<std::int64_t> frames;
            std::vector<double> weight;
            while (i1 < in_store.size() && in_store.file_index[i1] == in_store.file_index[i0]) {
                frames.push_back(in_store.frame_index[i1]);
                if (control) weight.push_back(kNominalCurrentA / control->at(i1));
                ++i1;
            }
            const std::vector<double> part = projection_store_sum(in_store.files[static_cast<std::size_t>(in_store.file_index[i0])], reduce, frames.data(), frames.size(), dark,
                                                                  control ? weight.data() : nullptr, &rows, &cols);
            if (sum.empty()) sum = part;
            else
                for (std::size_t i = 0; i < sum.size(); ++i) sum[i] += part[i];
            i0 = i1;
        }
        std::vector<float> mean(sum.size());
        for (std::size_t i = 0; i < sum.size(); ++i) mean[i] = static_cast<float>(sum[i] / static_cast<double>(in_store.size()));
        return mean;
    };
    DarksFlats out;
    out.dark = mean_of(scan.darks, nullptr, nullptr);
    for (std::size_t k = 0; k < scan.flats.size(); ++k) {
        const std::vector<float> flat = mean_of(scan.flats[k], out.dark.data(), &scan.flats_control_A[k]);
        out.flats.insert(out.flats.end(), flat.begin(), flat.end());
    }
    out.rows = rows, out.cols = cols;
    return out;
}

std::string IntensityCalibration::json() const {
    const auto placement_json = [](const ScanPlacement& p) {
        return Json::dict().set("shift_x", count(p.shift_x)).set("x_readjustment", num(p.x_readjustment)).set("z_readjustment", num(p.z_readjustment));
    };
    const Json measured = Json::dict()
                              .set("axis_first_scan", num(layout.axis_first_scan))
                              .set("axis_second_scan", num(layout.axis_second_scan))
                              .set("second_scan_rows", num(layout.second_scan_rows))
                              .set("quantum", num(layout.quantum));
    Json d = Json::dict();
    d.set("main", Json::of(main)).set("companion", Json::of(companion));
    if (scans.size() != 2) {  // (a pair's file names its two scans as it always did)
        Json names = Json::list(), places = Json::list();
        for (const std::string& n : scans) names.add(Json::of(n));
        for (const auto& v : vectors) places.add(numbers({v[0], v[1]}));
        d.set("scans", names).set("vectors", places);
    }
    if (!projection_store.empty()) d.set("projection_store", Json::of(projection_store)).set("reduce", count(reduce));
    d.set("cor_main_detector", num(cor_main_detector)).set("vector", numbers({vector[0], vector[1]}));
    d.set("rot_center", num(rot_center)).set("merged_width", count(merged_width));
    Json placed = Json::dict(), widths = Json::dict(), motors = Json::dict();
    for (std::size_t k = 0; k < scans.size(); ++k) {
        placed.set(scans[k], placement_json(placement[k]));
        widths.set(scans[k], numbers({apodisation[k][0], apodisation[k][1]}));
        motors.set(scans[k], num(estimated_cor_from_motor_pix[k]));
    }
    d.set("placement", placed).set("apodisation", widths).set("estimated_cor_from_motor_pix", motors);
    d.set("lit_rows", Json::list().add(count(lit[0])).add(count(lit[1])));
    d.set("details", Json::dict().set("layout", measured).set("detector_blur", blur_json(detector_blur)));
    d.set("detector_blur_field", numbers(detector_blur.field));
    d.set("fingerprint", Json::of(fingerprint)).set("version", count(version));
    return d.text();
}

IntensityCalibration calibrate_intensity(const Scan& main, const Scan& companion, const geometry::MergedLayout& layout, const std::string& cache_dir, const Log& log,
                                         bool progress) {
    return calibrate_intensity(std::vector<const Scan*>{&main, &companion}, layout, cache_dir, log, progress);
}

IntensityCalibration calibrate_intensity(const std::vector<const Scan*>& scans, const geometry::MergedLayout& layout, const std::string& cache_dir_given, const Log& log,
                                         bool progress, const std::string& projection_store, int reduce) {
    if (scans.empty()) throw Error("the intensity calibration takes the scans of one acquisition");
    const Scan& main = *scans[0];
    const std::size_t S = scans.size();
    for (const Scan* sc : scans)
        if (sc->rows != main.rows || sc->cols != main.cols) throw Error("the scans' frames differ in shape");
    if (layout.vectors.size() + 1 != S && !(S == 2 && layout.vectors.empty())) throw Error("the merged frame's layout is of another number of scans");
    const bool store = !projection_store.empty();
    if (reduce && !store) throw Error("the intensity calibration of frames read reduced takes its darks and flats from a projection store");
    if (reduce < 0 || reduce > 3) throw Error("the intensity calibration: reduce is 0, 1, 2 or 3");
    const int f = 1 << reduce;
    const std::string cache_dir = store ? std::string() : cache_dir_given;  // (a store's are read in a moment)
    IntensityCalibration cal;
    for (const Scan* sc : scans) cal.scans.push_back(sc->name);
    cal.main = main.name;
    cal.companion = S > 1 ? scans[1]->name : std::string();
    cal.projection_store = projection_store, cal.reduce = reduce;
    // the cache's names of the scans' arrays: the pair's, and companion2, companion3, .. beyond
    const auto tag = [](std::size_t k) { return k == 0 ? std::string("main") : k == 1 ? std::string("companion") : "companion" + std::to_string(k); };
    cal.reduced.resize(S);
    if (store) {
        Sha256 h;
        h.update("vc_tomography_calibration_v" + std::to_string(kIntensityCalibrationVersion) + "|store|" + projection_store + "|" + std::to_string(reduce));
        for (const Scan* sc : scans) h.update(sc->name + "|" + file_state(sc->master));
        cal.fingerprint = h.hex();
        for (std::size_t k = 0; k < S; ++k) {
            const std::string group = projection_store_tile_group(projection_store, scans[k]->sample_name.empty() ? scans[k]->name : scans[k]->sample_name);
            say(log, "[" + tag(k) + "] darks and flats of " + scans[k]->name + " from " + group + " at 1/" + std::to_string(f));
            cal.reduced[k] = store_darks_flats(*scans[k], group, reduce);
            if (cal.reduced[k].rows != main.rows / f || cal.reduced[k].cols != main.cols / f) throw Error(group + ": the store's darks and flats are not of the level's size");
        }
        cal.rows = main.rows / f, cal.cols = main.cols / f;
    } else {
        cal.fingerprint = acquisition_fingerprint(scans);
        cal.rows = main.rows, cal.cols = main.cols;
    }
    const std::size_t R = static_cast<std::size_t>(cal.rows), C = static_cast<std::size_t>(cal.cols);

    // the caches, each only if it is of this acquisition and version
    const std::string arrays_path = cache_dir.empty() ? std::string() : join(cache_dir, "calibration_arrays.bin");
    const std::string json_path = cache_dir.empty() ? std::string() : join(cache_dir, "calibration.json");
    bool have_arrays = store;
    if (!arrays_path.empty() && exists(arrays_path)) {
        try {
            const ArrayFile a = read_array_file(arrays_path);
            bool all = text_of(a.header.find("fingerprint")) == cal.fingerprint && a.find("weight_map");
            for (std::size_t k = 0; k < S; ++k) all = all && a.find(tag(k) + "_dark") && a.find(tag(k) + "_flats");
            if (all) {
                for (std::size_t k = 0; k < S; ++k) cal.reduced[k] = {main.rows, main.cols, a.find(tag(k) + "_dark")->values, a.find(tag(k) + "_flats")->values};
                cal.weight_map = a.find("weight_map")->values;
                have_arrays = true;
            }
        } catch (const Error&) {
        }
        if (!have_arrays) say(log, "calibration cache " + arrays_path + ": other acquisition / version, ignored");
    }
    std::optional<Json> cached;
    if (have_arrays && !json_path.empty() && exists(json_path)) {
        try {
            const std::string text = io::read_text(json_path);
            Json j = Json::parse(text);
            if (text_of(j.find("fingerprint")) == cal.fingerprint && number_of(j.find("version"), -1) == kIntensityCalibrationVersion) cached = std::move(j);
        } catch (const Error&) {
        }
        if (!cached) say(log, "calibration cache " + json_path + ": other acquisition / version, ignored");
    }

    if (!have_arrays) {
        for (std::size_t k = 0; k < S; ++k) {
            const Scan& sc = *scans[k];
            say(log, "[" + tag(k) + "] reducing darks/flats of " + sc.name + " ...");
            std::size_t frames = sc.darks.size();
            for (const FrameList& fl : sc.flats) frames += fl.size();
            std::unique_ptr<ProgressBar> bar;
            if (progress) bar = std::make_unique<ProgressBar>(frames, "darks/flats (" + tag(k) + ")", "frame");
            cal.reduced[k] = reduce_darks_flats(sc, bar ? std::function<void(std::size_t)>([&](std::size_t n) { bar->update(n); }) : nullptr);
        }
    }
    // the rows the beam lights, in every scan: of each scan's flats (their mean) less its dark
    cal.lit = {0, cal.rows};
    for (std::size_t k = 0; k < S; ++k) {
        const DarksFlats& d = cal.reduced[k];
        const std::size_t n = d.series();
        if (!n) throw Error("the scan " + scans[k]->name + " has no flats");
        std::vector<float> flat(R * C, 0.f);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t p = 0; p < R * C; ++p) flat[p] += d.flats[i * R * C + p];
        for (std::size_t p = 0; p < R * C; ++p) flat[p] = flat[p] / static_cast<float>(n) - d.dark[p];
        const std::array<int, 2> lit = lit_rows(flat.data(), cal.rows, cal.cols);
        cal.lit = {std::max(cal.lit[0], lit[0]), std::min(cal.lit[1], lit[1])};
    }
    if (cal.lit[1] <= cal.lit[0]) throw Error("the scans' flats light no row in common");
    if (cal.lit[1] - cal.lit[0] < cal.rows)
        say(log, "the beam lights rows " + std::to_string(cal.lit[0]) + ".." + std::to_string(cal.lit[1] - 1) + " of " + std::to_string(cal.rows) + ": the weights are cropped to them");
    if (cal.weight_map.empty()) {
        // the weight map from all the flats, the last scan's first (the reference's --scan_files order: the companion's, then the main scan's)
        std::vector<const float*> flats;
        for (std::size_t k = S; k-- > 0;)
            for (std::size_t i = 0; i < cal.reduced[k].series(); ++i) flats.push_back(cal.reduced[k].flats.data() + i * R * C);
        // (cropped to the rows the beam lights: where it lights them all, nothing is cropped)
        const int top = cal.lit[0], bottom = cal.rows - cal.lit[1];
        cal.weight_map = store ? weight_map(flats, cal.rows, cal.cols, 15.0 / f, 10.0 / f, top, bottom) : weight_map(flats, cal.rows, cal.cols, 15.0, 10.0, top, bottom);
        if (!cache_dir.empty()) {
            make_dirs(cache_dir);
            std::vector<NamedArray> arrays;
            for (std::size_t k = 0; k < S; ++k) {
                arrays.push_back({tag(k) + "_dark", {R, C}, cal.reduced[k].dark});
                arrays.push_back({tag(k) + "_flats", {cal.reduced[k].series(), R, C}, cal.reduced[k].flats});
            }
            arrays.push_back({"weight_map", {R, C}, cal.weight_map});
            write_array_file(arrays_path, Json::dict().set("fingerprint", Json::of(cal.fingerprint)), arrays);
        }
    }

    // the merged frame
    cal.layout = layout;
    cal.cor_main_detector = layout.cor_main_detector;
    cal.vectors = layout.vectors;
    if (S == 2 && cal.vectors.empty()) cal.vectors = {layout.vector};  // (a pair's layout given by its one vector)
    cal.vector = cal.vectors.empty() ? std::array<double, 2>{} : cal.vectors[0];
    const MergedPlacement pl = placement_from_vectors(cal.vectors, cal.cols);
    cal.placement = pl.scans;
    cal.merged_width = pl.width;
    cal.rot_center = cal.cor_main_detector + pl.anchor;
    // The apodisation: the main scan's slope beyond the rotation axis and its taper towards the next scan; of every
    // other scan a border taper on the side facing the main scan, and one on its far side if a scan lies beyond it.
    for (const Scan* sc : scans) cal.estimated_cor_from_motor_pix.push_back(sc->estimated_cor_from_motor_pix);
    cal.apodisation.resize(S);
    // A scan at the main scan's own columns and another height (above or below it: a grid's first column) is
    // apodised as the main scan: within a quarter of the detector's width of it and more than a quarter of its
    // height from it. Every other scan is beside the main scan.
    {
        const double apart = cal.cols / 4.0, borders = 50.0 / f;
        const auto columns = [&](std::size_t k) { return k == 0 ? 0.0 : cal.vectors[k - 1][1]; };
        const auto over_main = [&](std::size_t k) { return k > 0 && std::fabs(cal.vectors[k - 1][1]) < apart && std::fabs(cal.vectors[k - 1][0]) > cal.rows / 4.0; };
        double side = 0;  // where the scans beside the main scan lie: below nothing on its left, above on its right
        for (std::size_t k = 1; k < S; ++k) {
            if (over_main(k)) continue;
            if (side != 0 && (columns(k) < 0) != (side < 0)) throw Error("scans on both sides of the main scan are not supported");
            side = columns(k);
        }
        if (side == 0) {  // a scan alone, or scans above each other alone: no apodisation across the beam
            for (std::size_t k = 0; k < S; ++k) cal.apodisation[k] = {0.0, 0.0};
        } else {
            const ApodisationWidths aw = apodisation_widths(cal.cols, main.estimated_cor_from_motor_pix / f, side, 1000.0 / f, borders);
            for (std::size_t k = 0; k < S; ++k) {
                if (k == 0 || over_main(k)) {
                    cal.apodisation[k] = {aw.main_left, aw.main_right};
                    continue;
                }
                bool beyond = false;  // a scan lies further out
                for (std::size_t j = 1; j < S; ++j) beyond = beyond || std::fabs(columns(j)) > std::fabs(columns(k)) + apart;
                cal.apodisation[k] = side < 0 ? std::array<double, 2>{beyond ? borders : aw.companion_left, aw.companion_right}
                                              : std::array<double, 2>{aw.companion_left, beyond ? borders : aw.companion_right};
            }
        }
    }

    // the detector's blur: the cached measurement (the detector does not change with the merged frame), or measured
    const Json* cached_field = cached ? cached->find("detector_blur_field") : nullptr;
    if (store) {
        cal.detector_blur.scan = main.name;
        cal.detector_blur.reason = "the frames are a projection store's at 1/" + std::to_string(f) + ": the noise of its flats does not hold the detector's blur";
    } else if (cached_field && cached_field->is(Json::Array)) {
        cal.detector_blur = blur_of(*cached);
        say(log, "calibration loaded from " + cache_dir);
    } else {
        cal.detector_blur = measure_detector_blur(main, cal.reduced[0].dark.data(), 200, log, cal.lit);
        if (!cache_dir.empty()) {
            make_dirs(cache_dir);
            io::write_text_atomic(json_path, cal.json());
        }
    }
    if (S == 2)
        say(log, "merged frame: rotation axis " + fmt("%.3f", cal.cor_main_detector) + " in the main scan's columns, companion at (" + python_repr(cal.vector[0]) + ", " +
                     python_repr(cal.vector[1]) + ") (rows, columns): axis " + fmt("%.3f", cal.rot_center) + " of " + std::to_string(cal.merged_width) + " merged columns");
    else {
        std::string places;
        for (const auto& v : cal.vectors) places += (places.empty() ? "" : ", ") + ("(" + python_repr(v[0]) + ", " + python_repr(v[1]) + ")");
        say(log, "merged frame: rotation axis " + fmt("%.3f", cal.cor_main_detector) + " in the main scan's columns, the other scans at " + places + " (rows, columns): axis " +
                     fmt("%.3f", cal.rot_center) + " of " + std::to_string(cal.merged_width) + " merged columns");
    }

    // per scan: the weight map times its horizontal apodisation
    cal.weight.resize(S);
    for (std::size_t k = 0; k < S; ++k) {
        const auto mask = apodisation_mask(cal.cols, static_cast<float>(cal.apodisation[k][0]), static_cast<float>(cal.apodisation[k][1]), 1e-8f);
        std::vector<float>& w = cal.weight[k];
        w.resize(R * C);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) w[r * C + c] = cal.weight_map[r * C + c] * mask[c];
    }
    return cal;
}

}  // namespace vc_tomography::calibration
