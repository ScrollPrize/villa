// helirec/calibration/intensity.hpp: the intensity calibration of a scan pair (darks and flats, the weight maps, the
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

#include "helirec/calibration/intensity.hpp"
#include "helirec/progress.hpp"
#include "helirec/raw_reader.hpp"
#include "io/array_file.hpp"
#include "io/json.hpp"
#include "io/sha256.hpp"

namespace helirec::calibration {

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

// a double as Python's repr writes it (the shortest text that reads back as the same number)
std::string py_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char b[64];
    int p = 1;
    for (; p < 17; ++p) {
        std::snprintf(b, sizeof b, "%.*e", p - 1, v);
        if (std::strtod(b, nullptr) == v) break;
    }
    std::snprintf(b, sizeof b, "%.*e", p - 1, v);
    const int exponent = std::atoi(std::strchr(b, 'e') + 1);
    if (exponent < -4 || exponent >= 16) return b;
    std::snprintf(b, sizeof b, "%.*f", std::max(p - 1 - exponent, 0), v);
    std::string s = b;
    if (s.find('.') == std::string::npos) s += ".0";
    return s;
}

bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

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

// numpy's pairwise sum of float32 values (the sum numpy.mean takes over a contiguous frame)
float pairwise_sum(const float* a, std::size_t n) {
    if (n < 8) {
        float r = 0.f;
        for (std::size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

std::string darks_flats_fingerprint(const Scan& scan) {
    Sha256 h;
    h.update("helirec-darks-flats-v" + std::to_string(kIntensityCalibrationVersion) + "|" + std::to_string(scan.rows) + "," + std::to_string(scan.cols));
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

const char* const kReductionKeys[] = {"main_dark", "main_flats", "companion_dark", "companion_flats", "weight_map"};

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

MergedPlacement placement_from_vector(const std::array<double, 2>& vector, int cols) {
    const double dy = vector[0], dx = vector[1];
    MergedPlacement out;
    out.anchor = dx < 0 ? -dx : 0.0;
    const double cum[2][2] = {{0.0, 0.0}, {dy, dx}};
    for (int j = 0; j < 2; ++j) {
        const double d = out.anchor + cum[j][1];
        const int sx = static_cast<int>(std::floor(d));
        out.scans[j].shift_x = sx;
        out.scans[j].x_readjustment = static_cast<double>(static_cast<float>(sx - d));
        out.scans[j].z_readjustment = j > 0 ? cum[j][0] : 0.0;
    }
    out.width = cols + static_cast<int>(std::ceil(std::fabs(dx)));
    return out;
}

// ---------------------------------------------------------------- the detector's blur

FlatNoise flat_noise_spectra(const Scan& scan, const float* dark, int n_flats, int tile) {
    if (scan.flats.empty()) throw Error("the detector's blur needs flats");
    const FrameList& refs = scan.flats[0];
    const int n = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n_flats), refs.size())) / 2 * 2;
    if (n < 2) throw Error("the detector's blur needs at least two flats");
    const int rows = scan.rows, cols = scan.cols;
    FlatNoise out;
    out.n_flats = n;
    for (int y = (rows % tile) / 2; y <= rows - tile; y += tile) out.ys.push_back(y);
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

DetectorBlur measure_detector_blur(const Scan& scan, const float* dark, int n_flats, const Log& log) {
    const auto t0 = std::chrono::steady_clock::now();
    const int rows = scan.rows, cols = scan.cols;
    DetectorBlur res;
    res.scan = scan.name;
    const int n_have = scan.flats.empty() ? 0 : static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n_flats), scan.flats[0].size()));
    if (n_have < kBlurMinFlats || (rows / kBlurTile) * (cols / kBlurTile) < 6) {
        res.reason = n_have < kBlurMinFlats ? std::to_string(n_have) + " flats in the first series"
                                            : "a detector of " + std::to_string(rows) + " x " + std::to_string(cols) + " holds too few tiles of " + std::to_string(kBlurTile);
        say(log, "detector blur: not measured (" + res.reason + ")");
        return res;
    }
    const FlatNoise noise = flat_noise_spectra(scan, dark, n_flats);
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

std::string acquisition_fingerprint(const Scan& main, const Scan& companion) {
    Sha256 h;
    h.update("helirec-calibration-v" + std::to_string(kIntensityCalibrationVersion));
    for (const Scan* sc : {&main, &companion}) {
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
    d.set("cor_main_detector", num(cor_main_detector)).set("vector", numbers({vector[0], vector[1]}));
    d.set("rot_center", num(rot_center)).set("merged_width", count(merged_width));
    d.set("placement", Json::dict().set(main, placement_json(placement[0])).set(companion, placement_json(placement[1])));
    d.set("apodisation", Json::dict().set(main, numbers({apodisation[0][0], apodisation[0][1]})).set(companion, numbers({apodisation[1][0], apodisation[1][1]})));
    d.set("estimated_cor_from_motor_pix", Json::dict().set(main, num(estimated_cor_from_motor_pix[0])).set(companion, num(estimated_cor_from_motor_pix[1])));
    d.set("details", Json::dict().set("layout", measured).set("detector_blur", blur_json(detector_blur)));
    d.set("detector_blur_field", numbers(detector_blur.field));
    d.set("fingerprint", Json::of(fingerprint)).set("version", count(version));
    return d.text();
}

IntensityCalibration calibrate_intensity(const Scan& main, const Scan& companion, const geometry::MergedLayout& layout, const std::string& cache_dir,
                                         const Log& log, bool progress) {
    if (main.rows != companion.rows || main.cols != companion.cols) throw Error("the two scans' frames differ in shape");
    IntensityCalibration cal;
    cal.main = main.name;
    cal.companion = companion.name;
    cal.rows = main.rows;
    cal.cols = main.cols;
    cal.fingerprint = acquisition_fingerprint(main, companion);
    const std::size_t R = static_cast<std::size_t>(main.rows), C = static_cast<std::size_t>(main.cols);

    // the caches, each only if it is of this acquisition and version
    const std::string arrays_path = cache_dir.empty() ? std::string() : join(cache_dir, "calibration_arrays.bin");
    const std::string json_path = cache_dir.empty() ? std::string() : join(cache_dir, "calibration.json");
    bool have_arrays = false;
    if (!arrays_path.empty() && exists(arrays_path)) {
        try {
            const ArrayFile a = read_array_file(arrays_path);
            bool all = text_of(a.header.find("fingerprint")) == cal.fingerprint;
            for (const char* k : kReductionKeys) all = all && a.find(k);
            if (all) {
                cal.main_reduced = {main.rows, main.cols, a.find("main_dark")->values, a.find("main_flats")->values};
                cal.companion_reduced = {main.rows, main.cols, a.find("companion_dark")->values, a.find("companion_flats")->values};
                cal.weight_map = a.find("weight_map")->values;
                have_arrays = true;
            }
        } catch (const Error&) {
        }
        if (!have_arrays) say(log, "calibration cache " + arrays_path + ": other acquisition / version, ignored");
    }
    std::optional<Json> cached;
    if (have_arrays && exists(json_path)) {
        try {
            std::ifstream in(json_path);
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            Json j = Json::parse(text);
            if (text_of(j.find("fingerprint")) == cal.fingerprint && number_of(j.find("version"), -1) == kIntensityCalibrationVersion) cached = std::move(j);
        } catch (const Error&) {
        }
        if (!cached) say(log, "calibration cache " + json_path + ": other acquisition / version, ignored");
    }

    if (!have_arrays) {
        for (int s = 0; s < 2; ++s) {
            const Scan& sc = s ? companion : main;
            const std::string tag = s ? "companion" : "main";
            say(log, "[" + tag + "] reducing darks/flats of " + sc.name + " ...");
            std::size_t frames = sc.darks.size();
            for (const FrameList& f : sc.flats) frames += f.size();
            std::unique_ptr<ProgressBar> bar;
            if (progress) bar = std::make_unique<ProgressBar>(frames, "darks/flats (" + tag + ")", "frame");
            DarksFlats r = reduce_darks_flats(sc, bar ? std::function<void(std::size_t)>([&](std::size_t k) { bar->update(k); }) : nullptr);
            (s ? cal.companion_reduced : cal.main_reduced) = std::move(r);
        }
        // the weight map from all the flats, the companion's first (the reference's --scan_files order)
        std::vector<const float*> flats;
        for (const DarksFlats* d : {&cal.companion_reduced, &cal.main_reduced})
            for (std::size_t k = 0; k < d->series(); ++k) flats.push_back(d->flats.data() + k * R * C);
        cal.weight_map = weight_map(flats, main.rows, main.cols);
        if (!cache_dir.empty()) {
            make_dirs(cache_dir);
            write_array_file(arrays_path, Json::dict().set("fingerprint", Json::of(cal.fingerprint)),
                             {{"main_dark", {R, C}, cal.main_reduced.dark},
                              {"main_flats", {cal.main_reduced.series(), R, C}, cal.main_reduced.flats},
                              {"companion_dark", {R, C}, cal.companion_reduced.dark},
                              {"companion_flats", {cal.companion_reduced.series(), R, C}, cal.companion_reduced.flats},
                              {"weight_map", {R, C}, cal.weight_map}});
        }
    }

    // the merged frame
    cal.layout = layout;
    cal.cor_main_detector = layout.cor_main_detector;
    cal.vector = layout.vector;
    const MergedPlacement pl = placement_from_vector(cal.vector, main.cols);
    cal.placement = pl.scans;
    cal.merged_width = pl.width;
    cal.rot_center = cal.cor_main_detector + pl.anchor;
    const ApodisationWidths aw = apodisation_widths(main.cols, main.estimated_cor_from_motor_pix, cal.vector[1]);
    cal.apodisation = {{{aw.main_left, aw.main_right}, {aw.companion_left, aw.companion_right}}};
    cal.estimated_cor_from_motor_pix = {main.estimated_cor_from_motor_pix, companion.estimated_cor_from_motor_pix};

    // the detector's blur: the cached measurement (the detector does not change with the merged frame), or measured
    const Json* cached_field = cached ? cached->find("detector_blur_field") : nullptr;
    if (cached_field && cached_field->is(Json::Array)) {
        cal.detector_blur = blur_of(*cached);
        say(log, "calibration loaded from " + cache_dir);
    } else {
        cal.detector_blur = measure_detector_blur(main, cal.main_reduced.dark.data(), 200, log);
        if (!cache_dir.empty()) {
            make_dirs(cache_dir);
            const std::string tmp = json_path + ".tmp" + std::to_string(static_cast<long>(::getpid()));
            {
                std::ofstream o(tmp);
                o << cal.json() << "\n";
                if (!o) throw Error("cannot write " + tmp);
            }
            if (std::rename(tmp.c_str(), json_path.c_str()) != 0) throw Error("cannot write " + json_path);
        }
    }
    say(log, "merged frame: rotation axis " + fmt("%.3f", cal.cor_main_detector) + " in the main scan's columns, companion at (" + py_repr(cal.vector[0]) + ", " +
                 py_repr(cal.vector[1]) + ") (rows, columns): axis " + fmt("%.3f", cal.rot_center) + " of " + std::to_string(cal.merged_width) + " merged columns");

    // per scan: the weight map times its horizontal apodisation
    for (int s = 0; s < 2; ++s) {
        const auto mask = apodisation_mask(main.cols, static_cast<float>(cal.apodisation[s][0]), static_cast<float>(cal.apodisation[s][1]), 1e-8f);
        std::vector<float>& w = s ? cal.companion_weight : cal.main_weight;
        w.resize(R * C);
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c) w[r * C + c] = cal.weight_map[r * C + c] * mask[c];
    }
    return cal;
}

}  // namespace helirec::calibration
