// helirec/reconstruction.hpp: the reconstruction of a scan pair from its raw folders.
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>
#include <tuple>

#include "helirec/calibration/geometry_calibration.hpp"
#include "helirec/calibration/record.hpp"
#include "helirec/raw_reader.hpp"
#include "helirec/reconstruction.hpp"
#include "io/json.hpp"
#include "io/sha256.hpp"

#ifndef HELIREC_SOURCE_DIR
#define HELIREC_SOURCE_DIR ""
#endif

namespace helirec {

namespace {

using Log = std::function<void(const std::string&)>;
void say(const Log& log, const std::string& line) {
    if (log) log(line);
}

std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw Error("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string base_name(const std::string& p) {
    const std::size_t i = p.find_last_of('/');
    return i == std::string::npos ? p : p.substr(i + 1);
}

std::string absolute(const std::string& p) { return std::filesystem::absolute(p).lexically_normal().string(); }

double mtime(const std::string& p) {
    struct stat st;
    if (::stat(p.c_str(), &st) != 0) return 0;
#ifdef __APPLE__
    return static_cast<double>(st.st_mtimespec.tv_sec) + 1e-9 * st.st_mtimespec.tv_nsec;
#else
    return static_cast<double>(st.st_mtim.tv_sec) + 1e-9 * st.st_mtim.tv_nsec;
#endif
}

std::string strip(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return std::string();
    return s.substr(a, s.find_last_not_of(" \t\n\r") - a + 1);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// (the key by value: taken by reference, GCC's -Wdangling-reference supposes the result may refer to it)
const Json& member(const Json& j, std::string_view key) {
    const Json* v = j.find(std::string(key));
    if (!v) throw Error("no '" + std::string(key) + "' in the parameters or the geometry calibration");
    return *v;
}

// a parameter as text (as Python's str() of it)
std::string text_of(const Json& v) {
    switch (v.type) {
        case Json::String: return v.string;
        case Json::Bool: return v.boolean ? "1" : "0";
        case Json::Number: return v.whole ? v.text() : python_repr(v.number);
        case Json::Null: return std::string();
        default: return v.text();
    }
}

double number_of(const Json& v, const std::string& key) {
    if (v.is(Json::Number)) return v.number;
    if (v.is(Json::Bool)) return v.boolean ? 1 : 0;
    if (v.is(Json::String)) {
        char* end = nullptr;
        const std::string s = strip(v.string);
        const double x = std::strtod(s.c_str(), &end);
        if (!s.empty() && end == s.c_str() + s.size()) return x;
    }
    throw Error("parameter " + key + ": a number, not " + v.text());
}

bool truth(const Json& v) {
    if (v.is(Json::Bool)) return v.boolean;
    if (v.is(Json::Number)) return v.number != 0;
    if (v.is(Json::String)) return !v.string.empty();
    return false;
}

FilterChain filter_chain_of(const Json& chain_in) {
    const Json chain = chain_in.is(Json::String) ? Json::parse(parse_filter_chain(chain_in.string)) : chain_in;
    FilterChain out;
    for (const Json& f : chain.array) {
        FilterSpec s;
        for (const auto& [k, v] : f.object) {
            if (k == "type") s.type = v.string;
            else s.params.values[k] = text_of(v);
        }
        if (s.type.empty()) throw Error("filter without 'type'");
        out.push_back(std::move(s));
    }
    return out;
}

Json numbers(const std::vector<double>& v) {
    Json j = Json::list();
    for (double x : v) j.add(Json::of(x));
    return j;
}

}  // namespace

std::string build_revision() {
    const std::string dir = HELIREC_SOURCE_DIR;
    if (dir.empty() || !exists(dir)) return std::string();
    const std::string command = "git -C '" + dir + "' describe --always --dirty 2>/dev/null";
    FILE* p = ::popen(command.c_str(), "r");
    if (!p) return std::string();
    char buf[256];
    std::string out;
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    ::pclose(p);
    return strip(out);
}

// the reconstructor's settings the parameters name (all but the device, the physics, the axis, the exposure arc,
// the sharpness weighting and the rolling shutter, which come from the scans and the calibrations)
void apply_parameters(const Json& params, ReconConfig& cfg, const ExternalBackprojectorFactory& external, std::string* resolved_sld) {
    cfg.filters = filter_chain_of(member(params, "filters"));
    cfg.margin = static_cast<int>(number_of(member(params, "margin"), "margin"));
    cfg.n_grid = static_cast<int>(number_of(member(params, "n_grid"), "n_grid"));
    cfg.angular_tolerance_steps = number_of(member(params, "angular_tolerance_steps"), "angular_tolerance_steps");
    cfg.luminosity_repartition = static_cast<float>(number_of(member(params, "luminosity_repartition"), "luminosity_repartition"));
    cfg.mend_threshold = static_cast<float>(number_of(member(params, "mend_threshold"), "mend_threshold"));
    cfg.log_clip_min = static_cast<float>(number_of(member(params, "log_clip_min"), "log_clip_min"));
    cfg.log_clip_max = static_cast<float>(number_of(member(params, "log_clip_max"), "log_clip_max"));
    cfg.block_angles = static_cast<int>(number_of(member(params, "block_angles"), "block_angles"));
    apply_backprojector(text_of(member(params, "backprojector")), cfg, external);
    cfg.grid_size = static_cast<int>(number_of(member(params, "grid_size"), "grid_size"));  // (0: the caller's rule)
    cfg.ghbp_legs = static_cast<int>(number_of(member(params, "ghbp_legs"), "ghbp_legs"));
    cfg.ghbp_reduction_steps = static_cast<int>(number_of(member(params, "ghbp_reduction_steps"), "ghbp_reduction_steps"));
    cfg.clip_outer_circle = truth(member(params, "clip_outer_circle"));
    cfg.axis_source_meters = number_of(member(params, "axis_source_meters"), "axis_source_meters");
    const Json& sld = member(params, "sld");
    cfg.sld = sld_params(sld.is(Json::Object) ? sld.text() : text_of(sld), resolved_sld);
    cfg.x_interp = text_of(member(params, "x_interp"));
    cfg.z_interp = text_of(member(params, "z_interp"));
    cfg.angle_interp = text_of(member(params, "angle_interp"));
    cfg.sharp_f = number_of(member(params, "sharp_f"), "sharp_f");
    cfg.sharp_band = text_of(member(params, "sharp_band"));
    std::tie(cfg.angle_subset, cfg.angle_subsets) = parse_angle_subset(text_of(member(params, "angle_subset")));
}

std::pair<std::string, std::string> geometry_calibration_file(const std::string& given, const std::string& cache_dir, const std::vector<std::string>& masters,
                                                              const std::string& row_time_us, Device device, const Log& log, const GeometryMaker& make) {
    if (!given.empty()) return {absolute(given), read_text(given)};
    if (cache_dir.empty())
        throw Error("no geometry calibration: give geometry=PATH (made by helirec-calibration geometry), or a cache directory to make it in and keep it");
    const std::string path = cache_dir + "/geometry.json";
    if (exists(path)) {
        const std::string raw = read_text(path);
        bool same = false;
        try {
            const Json doc = Json::parse(raw);
            const Json* scans = doc.find("scans");
            if (scans && scans->is(Json::Array) && scans->array.size() == masters.size()) {
                same = true;
                for (std::size_t k = 0; k < masters.size(); ++k) same = same && base_name(text_of(scans->array[k])) == base_name(masters[k]);
            }
        } catch (const Error&) {
        }
        if (same) {
            say(log, "geometry calibration loaded from " + path);
            return {absolute(path), raw};
        }
        say(log, "geometry calibration " + path + ": other scans or unreadable, made again");
    }
    say(log, "geometry calibration of the scans (helirec-calibration geometry: layout, rows' timing, detector map, poses, motion) ...");
    calibration::geometry_calibration::Options opt;
    if (!row_time_us.empty()) opt.row_time = number_of(Json::of(row_time_us), "row_time_us") * 1e-6;
    opt.device = device;
    opt.log = log;
    const std::string text = make ? make(masters, opt) : calibration::geometry_calibration::calibrate_geometry(masters, opt);
    std::filesystem::create_directories(cache_dir);
    calibration::geometry_calibration::save_text(text, path);
    return {absolute(path), read_text(path)};
}

struct Reconstruction::Impl {
    std::string projections;
    Options options;
    Json params;
    calibration::Scan main, companion;

    // the geometry calibration in use
    std::string geometry_path, geometry_text, geometry_sha256;
    std::unique_ptr<geometry::Calibration> geometry;
    Json geometry_doc;
    std::array<geometry::Poses, 2> poses;
    std::string not_applied;
    geometry::MergedLayout layout;
    std::array<geometry::Terms, 2> terms;
    double rolling_shutter = 0;
    std::tuple<std::string, double, std::string> geometry_key;

    calibration::IntensityCalibration intensity;
    std::shared_ptr<SliceReconstructor> rec;
    std::string resolved_sld = "{}";
    int lo = 0, hi = 0;

    bool legacy() const { return member(params, "mode").string == "legacy_reference"; }

    std::tuple<std::string, double, std::string> geometry_request() const {
        const std::string path = strip(text_of(member(params, "geometry")));
        return {path, path.empty() ? 0.0 : mtime(path), strip(text_of(member(params, "row_time_us")))};
    }

    void geometry_file() {
        std::tie(geometry_path, geometry_text) =
            geometry_calibration_file(strip(text_of(member(params, "geometry"))), options.cache_dir, {main.master, companion.master},
                                      strip(text_of(member(params, "row_time_us"))), options.device, options.log);
    }

    // the geometry calibration, the poses it gives the frames, the merged frame laid out from them, the intensity
    // calibration in that frame and both scans' terms
    void calibrate() {
        geometry_file();
        geometry_sha256 = Sha256().update(geometry_text).hex();
        geometry_doc = Json::parse(geometry_text);
        geometry = std::make_unique<geometry::Calibration>(geometry::Calibration::parse(geometry_text));
        const std::vector<std::string> made = geometry->scans();
        const calibration::Scan* scans[2] = {&main, &companion};
        for (int k = 0; k < 2; ++k)
            if (made.size() != 2 || base_name(made[static_cast<std::size_t>(k)]) != base_name(scans[k]->master))
                throw Error("geometry " + geometry_path + " was made from " + (made.size() > static_cast<std::size_t>(k) ? made[static_cast<std::size_t>(k)] : "other scans") +
                            ", not from " + scans[k]->master + " (first scan: the one with the rotation axis in its field)");
        const std::array<std::vector<double>, 2> H = {geometry->heights_rows(main.z_translation_m, main.pixel_size_m),
                                                      geometry->heights_rows(companion.z_translation_m, companion.pixel_size_m)};
        poses = geometry->frame_poses({main.unwrapped_angles_deg, companion.unwrapped_angles_deg}, H, &not_applied);
        layout = geometry->merged_layout(poses, H);
        intensity = calibration::calibrate_intensity(main, companion, layout, options.cache_dir, options.log, options.progress);
        const Json& lay = member(geometry_doc, "layout");
        const int first_helix = static_cast<int>(member(member(lay, "frames").array.at(0), "rest1").number);
        const double origin = H[0].at(static_cast<std::size_t>(first_helix)) - H[0][0];  // the helix's first frame lies where the record puts it
        for (int k = 0; k < 2; ++k) {
            const calibration::ScanPlacement& pl = intensity.placement[static_cast<std::size_t>(k)];
            terms[static_cast<std::size_t>(k)] =
                geometry::reconstruction_terms(poses[static_cast<std::size_t>(k)], H[static_cast<std::size_t>(k)], member(lay, "row_sign").number,
                                               static_cast<int>(member(lay, "rows").number), intensity.rot_center, pl.shift_x - pl.x_readjustment,
                                               pl.z_readjustment, origin);
        }
        rolling_shutter = geometry->rolling_shutter();
        geometry_key = geometry_request();
    }

    ScanData scan_data(int k) const {
        const calibration::Scan& scan = k ? companion : main;
        const calibration::DarksFlats& reduced = k ? intensity.companion_reduced : intensity.main_reduced;
        const calibration::ScanPlacement& pl = intensity.placement[static_cast<std::size_t>(k)];
        ScanData d;
        ScanTrajectory& t = d.traj;
        t.z_translation_m = scan.z_translation_m;
        t.pixel_size_m = scan.pixel_size_m;
        t.det_rows = scan.rows;
        t.det_cols = scan.cols;
        t.shift_x = pl.shift_x;
        t.x_readjustment = static_cast<float>(pl.x_readjustment);
        t.z_readjustment = static_cast<float>(pl.z_readjustment);
        d.control_A = scan.control_A;
        d.flat_info.angles_deg.assign(scan.flats_unwrapped_angles_deg.begin(), scan.flats_unwrapped_angles_deg.end());
        d.dark = reduced.dark;
        const std::size_t px = reduced.dark.size();
        d.flats.resize(reduced.flats.size());
        for (std::size_t i = 0; i < d.flats.size(); ++i) d.flats[i] = reduced.flats[i] - reduced.dark[i % px];
        d.weight = k ? intensity.companion_weight : intensity.main_weight;
        d.reader = std::make_shared<RawReader>(scan.projections.files, scan.projections.dataset, scan.rows, scan.cols);
        d.file_index = scan.projections.file_index;
        d.frame_index = scan.projections.frame_index;
        if (legacy()) {  // the reference: the recorded angles and heights, one axis and one vector (the merged frame)
            t.angles_deg = scan.unwrapped_angles_deg;
            t.z_translation_scale = 1.0;
            return d;
        }
        const geometry::Terms& g = terms[static_cast<std::size_t>(k)];
        t.angles_deg = g.angles_deg;
        t.z_translation_scale = g.z_translation_scale;
        t.row_offset = g.row_offset;
        t.row_slope = g.row_slope;
        t.slope_col = static_cast<float>(g.slope_col);
        t.col_offset = g.col_offset;
        t.col_slope = g.col_slope;
        t.slope_row = static_cast<float>(g.slope_row);
        t.col_scale = g.col_scale;
        t.scale_col = static_cast<float>(g.scale_col);
        t.row_scale = g.row_scale;
        t.scale_row = static_cast<float>(g.scale_row);
        const geometry::Map m = geometry->map();
        t.map_i.assign(m.i.begin(), m.i.end());
        t.map_j.assign(m.j.begin(), m.j.end());
        t.map_eu = m.eu;
        t.map_ev = m.ev;
        t.map_c0 = m.c0;
        t.map_cs = m.cs;
        t.map_v0 = m.v0;
        t.map_vs = m.vs;
        return d;
    }

    double exposure_arc() const {
        const Json& v = member(params, "exposure_arc");
        if (v.is(Json::String) && lower(strip(v.string)) == "auto") return main.exposure_arc.value_or(0.0);
        return number_of(v, "exposure_arc");
    }

    std::string sharpness_weighting() const {
        const Json& v = member(params, "sharpness_weighting");
        const auto join = [](const std::vector<double>& x) {
            std::string out;
            char b[32];
            for (double e : x) {
                std::snprintf(b, sizeof b, "%.6g", e);
                out += (out.empty() ? "" : ":") + std::string(b);
            }
            return out;
        };
        if (v.is(Json::Null)) return std::string();
        if (v.is(Json::Array)) {
            std::vector<double> x;
            for (const Json& e : v.array) x.push_back(number_of(e, "sharpness_weighting"));
            if (x.size() != 6) throw Error("sharpness_weighting must be 'auto', 'off' or a11:a12:a21:a22:c1:c2");
            return join(x);
        }
        const std::string t = lower(strip(text_of(v)));
        if (t.empty() || t == "off" || t == "none") return std::string();
        if (t == "auto") return join(intensity.detector_blur.field);
        if (std::count(t.begin(), t.end(), ':') != 5) throw Error("sharpness_weighting must be 'auto', 'off' or a11:a12:a21:a22:c1:c2");
        return text_of(v);
    }

    void build() {
        const bool legacy_mode = legacy();
        ReconConfig cfg;
        cfg.device = options.device;
        cfg.energy_kev = main.energy_kev;
        cfg.distance_m = main.distance_m;
        cfg.pixel_size_m = main.pixel_size_m;
        cfg.rot_center = static_cast<float>(intensity.rot_center);
        cfg.legacy_reference = legacy_mode;
        apply_parameters(params, cfg, options.external_backprojector, &resolved_sld);
        if (!cfg.grid_size) cfg.grid_size = static_cast<int>(std::nearbyint(2.0 * intensity.rot_center)) + 1;  // 2 x axis rounded to even, + 1
        cfg.exposure_arc = exposure_arc();
        cfg.sharpness_weighting = legacy_mode ? std::string() : sharpness_weighting();
        cfg.rolling_shutter = legacy_mode ? 0.0 : rolling_shutter;
        std::vector<ScanData> scans = {scan_data(0), scan_data(1)};
        rec = std::make_shared<SliceReconstructor>(std::move(scans), cfg);
        std::tie(lo, hi) = rec->doable();
        if (rec->angle_interp() == "views" && exposure_arc() != 0)
            say(options.log, "note: exposure_arc is disabled with angle_interp=views (not deconvolved; angle_interp=nearest applies it)");
    }

    Json info_json() const {
        const auto [mx0, mx1] = rec->mend_domain();
        Json counts = Json::list();
        for (int n : rec->projection_counts()) counts.add(Json::whole_number(n));
        return Json::dict()
            .set("doable", Json::list().add(Json::whole_number(lo)).add(Json::whole_number(hi)))
            .set("margin", Json::whole_number(rec->margin()))
            .set("n_grid", Json::whole_number(rec->n_grid()))
            .set("grid_size", Json::whole_number(rec->grid_size()))
            .set("merged_width", Json::whole_number(rec->merged_width()))
            .set("mend_domain", Json::list().add(Json::whole_number(mx0)).add(Json::whole_number(mx1)))
            .set("legacy_reference", Json::of_bool(rec->config().legacy_reference))
            .set("exposure_arc_bins", Json::of(rec->exposure_arc_bins()))
            .set("rolling_shutter", Json::of(rec->rolling_shutter()))
            .set("angle_interp", Json::of(rec->angle_interp()))
            .set("z_interp", Json::of(rec->z_interp()))
            .set("angle_subsets", Json::whole_number(rec->config().angle_subsets))
            .set("angle_subset", Json::whole_number(rec->config().angle_subset))
            .set("projections", counts);
    }
};

Reconstruction::Reconstruction(const std::string& projections, const std::string& params, Options options) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.projections = projections;
    I.options = std::move(options);
    std::vector<calibration::Scan> scans = calibration::discover_scans(projections);
    const std::vector<int> pair = calibration::pair_companions(scans);
    I.main = std::move(scans[static_cast<std::size_t>(pair[0])]);
    I.companion = std::move(scans[static_cast<std::size_t>(pair[1])]);
    say(I.options.log, "scans: " + I.main.name + " (" + std::to_string(I.main.unwrapped_angles_deg.size()) + " projections) + " + I.companion.name + " (" +
                           std::to_string(I.companion.unwrapped_angles_deg.size()) + ")");
    I.main.check_files();
    I.companion.check_files();
    I.params = Json::parse(update_parameters(default_parameters(), params.empty() ? "{}" : params));  // (the mode is checked before the calibration)
    I.calibrate();
    I.build();
}

Reconstruction::~Reconstruction() = default;

void Reconstruction::set_parameters(const std::string& changes) {
    impl_->params = Json::parse(update_parameters(impl_->params.text(), changes));
    if (impl_->geometry_request() != impl_->geometry_key) impl_->calibrate();  // another geometry calibration: the frame and the terms again
    impl_->build();
}

std::string Reconstruction::parameters() const { return impl_->params.text(); }
const calibration::Scan& Reconstruction::main() const { return impl_->main; }
const calibration::Scan& Reconstruction::companion() const { return impl_->companion; }
const calibration::IntensityCalibration& Reconstruction::intensity() const { return impl_->intensity; }
std::shared_ptr<SliceReconstructor> Reconstruction::reconstructor() const { return impl_->rec; }
std::pair<int, int> Reconstruction::doable() const { return {impl_->lo, impl_->hi}; }
std::string Reconstruction::mode() const { return member(impl_->params, "mode").string; }
double Reconstruction::rot_center() const { return impl_->intensity.rot_center; }
double Reconstruction::rolling_shutter() const { return impl_->legacy() ? 0.0 : impl_->rolling_shutter; }
double Reconstruction::exposure_arc() const { return impl_->exposure_arc(); }
std::string Reconstruction::sharpness_weighting() const { return impl_->sharpness_weighting(); }
std::string Reconstruction::resolved_sld() const { return impl_->resolved_sld; }
std::string Reconstruction::info() const { return impl_->info_json().text(); }

SliceInfo Reconstruction::slice_info(double fraction) const {
    if (!(0.0 < fraction && fraction < 1.0)) throw Error("fraction must be in (0, 1)");
    const Impl& I = *impl_;
    const int k = static_cast<int>(std::nearbyint(fraction * (I.hi - I.lo)));  // half to even (the reference's)
    const int h = I.hi - 1 - k;
    const double z0 = I.main.z_translation_m.at(0);
    const double px_mm = I.main.pixel_size_m * 1e3;
    SliceInfo s;
    s.fraction = fraction;
    s.view_height = h;
    s.nabu_index = 999999 - (I.hi - h);
    s.height_mm = px_mm * ((I.main.rows - 1) / 2.0 - h) - z0 * 1e3;
    return s;
}

std::string Reconstruction::provenance() const {
    const Impl& I = *impl_;
    const std::string m = mode();
    const bool legacy = I.legacy();
    Json pinned = Json::list();
    if (legacy) {
        std::vector<std::string> keys;
        for (const auto& kv : Json::parse(legacy_reference_parameters()).object) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (const auto& k : keys) pinned.add(Json::of(k));
    }
    const Json conventions = Json::parse(mode_conventions(m));
    const calibration::IntensityCalibration& cal = I.intensity;
    Json map = Json::dict();
    const Json& doc_map = member(I.geometry_doc, "map");
    for (const char* k : {"c0", "cs", "v0", "vs", "i", "j", "eu", "ev"}) map.set(k, member(doc_map, k));
    const Json layout = Json::dict()
                            .set("cor_main_detector", Json::of(I.layout.cor_main_detector))
                            .set("vector", numbers({I.layout.vector[0], I.layout.vector[1]}))
                            .set("measured", Json::dict()
                                                 .set("axis_first_scan", Json::of(I.layout.axis_first_scan))
                                                 .set("axis_second_scan", Json::of(I.layout.axis_second_scan))
                                                 .set("second_scan_rows", Json::of(I.layout.second_scan_rows))
                                                 .set("quantum", Json::of(I.layout.quantum)));
    const Json geometry = Json::dict()
                              .set("path", Json::of(I.geometry_path))
                              .set("sha256", Json::of(I.geometry_sha256))
                              .set("scans", member(I.geometry_doc, "scans"))
                              .set("map", map)
                              .set("row_time_s_per_row", member(member(I.geometry_doc, "row_time"), "seconds_per_row"))
                              .set("merged_frame", layout)
                              .set("applied", Json::of(legacy ? "the merged frame alone (mode legacy_reference)" : "the detector map and every projection's pose and row timing"))
                              .set("measured_not_applied", I.not_applied.empty() ? Json() : Json::parse(I.not_applied));
    return Json::dict()
        .set("mode", Json::of(m))
        .set("conventions", conventions)
        .set("params", I.params)
        .set("resolved_sld", Json::parse(I.resolved_sld))
        .set("pinned", pinned)
        .set("output_units", member(conventions, "output_units"))
        .set("calibration", Json::dict()
                                .set("fingerprint", Json::of(cal.fingerprint))
                                .set("version", Json::whole_number(cal.version))
                                .set("rot_center", Json::of(cal.rot_center))
                                .set("vector", numbers({cal.vector[0], cal.vector[1]}))
                                .set("cor_main_detector", Json::of(cal.cor_main_detector)))
        .set("resolved_sharpness_weighting", Json::of(sharpness_weighting()))
        .set("resolved_rot_center", Json::of(rot_center()))
        .set("resolved_rolling_shutter", Json::of(rolling_shutter()))
        .set("geometry", geometry)
        .set("reconstructor", I.info_json())
        .set("pixel_size_m", Json::of(I.main.pixel_size_m))
        .set("energy_kev", Json::of(I.main.energy_kev))
        .set("distance_m", Json::of(I.main.distance_m))
        .set("helirec", Json::dict().set("git", Json::of(build_revision())))
        .text();
}

// ---------------------------------------------------------------- one tile of a mosaic

TileReconstruction tile_reconstructor(const std::string& master, const std::string& description, const std::string& label, const TileOptions& o) {
    const Json cal = Json::parse(description);
    const Json& L = member(cal, "layout");
    const auto& tiles = member(L, "tiles").array;
    std::size_t k = 0;
    while (k < tiles.size() && tiles[k].string != label) ++k;
    if (k == tiles.size()) throw Error("the description has no tile '" + label + "'");
    const auto& range = member(L, "frames_on_disk").array.at(k).array;
    TileReconstruction out;
    out.first_frame = static_cast<std::int64_t>(range.at(0).number);
    out.end_frame = static_cast<std::int64_t>(range.at(1).number);
    out.reduce = o.reduce;
    out.scan = calibration::open_scan(master);
    const calibration::Scan& scan = out.scan;
    const int rows_full = scan.rows, cols_full = scan.cols;
    // the helix among the scan's projections (they are kept one stretch after the other)
    std::int64_t before = 0;
    {
        const calibration::Record rec = calibration::read_record(scan.master, false);
        bool found = false;
        for (const calibration::Stretch& st : rec.stretches) {
            if (st.role != calibration::Role::Projection) continue;
            if (st.latched() && !st.z_center.empty()) {
                found = true;
                break;
            }
            before += st.n_frames;
        }
        if (!found) throw Error(scan.name + ": no helix among the projections");
    }
    std::vector<std::int64_t> idx;
    std::vector<double> frame, angle, z, height_rows, control;
    std::vector<std::int32_t> file_index, frame_index;
    const double row_sign = member(L, "row_sign").number;
    for (std::int64_t f = out.first_frame; f < out.end_frame; ++f) {
        const std::size_t i = static_cast<std::size_t>(before + f);
        frame.push_back(static_cast<double>(f));
        angle.push_back(scan.unwrapped_angles_deg.at(i));
        z.push_back(scan.z_translation_m.at(i));
        height_rows.push_back(row_sign * scan.z_translation_m[i] / scan.pixel_size_m);
        control.push_back(scan.control_A.at(i));
        file_index.push_back(scan.projections.file_index.at(i));
        frame_index.push_back(scan.projections.frame_index.at(i));
    }
    if (frame.empty()) throw Error("tile " + label + ": no frames on disk");
    const Json& t = member(member(cal, "tiles"), label);
    const double axis_column = member(t, "axis_column").number;
    const int f = 1 << o.reduce;
    // The slice's grid has its centre on a pixel or between two, and the reconstruction puts the axis there: the axis'
    // column is rounded to half a pixel of the reconstruction (the rest goes into every frame's column offset), so
    // that rot_center is where the axis lies in the slice.
    double rot_center = f * std::nearbyint(2 * axis_column / f) / 2.0;
    if (!o.reduced) say(o.log, "[" + label + "] darks and flats ...");
    out.reduced = o.reduced ? *o.reduced : calibration::reduced_darks_flats(scan, o.cache_dir, o.log);
    const calibration::DarksFlats& red = out.reduced;
    const std::size_t px_full = static_cast<std::size_t>(rows_full) * static_cast<std::size_t>(cols_full);
    std::vector<const float*> stacks;
    for (std::size_t s = 0; s < red.series(); ++s) stacks.push_back(red.flats.data() + s * px_full);
    const std::vector<float> wm = calibration::weight_map(stacks, rows_full, cols_full);
    const bool right = axis_column > (cols_full - 1) / 2.0;
    const double far = right ? 50.0 : std::min(2000.0, axis_column), near = right ? std::min(2000.0, cols_full - axis_column) : 50.0;
    const std::vector<float> mask = calibration::apodisation_mask(cols_full, static_cast<float>(far), static_cast<float>(near), 1e-8f);
    std::vector<float> w(px_full);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows_full); ++r)
        for (std::size_t c = 0; c < static_cast<std::size_t>(cols_full); ++c) w[r * cols_full + c] = wm[r * cols_full + c] * mask[c];
    if (o.weight) o.weight(w, rows_full, cols_full);
    out.row_origin = o.row_origin ? *o.row_origin : -(1.0 + member(cal, "axial_scale").number) * height_rows[height_rows.size() / 2];
    out.terms = geometry::TileGeometry::parse(description).reconstruction_terms(label, angle, height_rows, frame, rot_center, out.row_origin);
    geometry::Terms& T = out.terms;
    const Json& m = member(cal, "map");
    std::vector<double> eu, ev;
    std::vector<std::int32_t> mi, mj;
    for (const Json& v : member(m, "eu").array) eu.push_back(v.number);
    for (const Json& v : member(m, "ev").array) ev.push_back(v.number);
    for (const Json& v : member(m, "i").array) mi.push_back(static_cast<std::int32_t>(v.number));
    for (const Json& v : member(m, "j").array) mj.push_back(static_cast<std::int32_t>(v.number));
    double c0 = member(m, "c0").number, cs = member(m, "cs").number, v0 = member(m, "v0").number, vs = member(m, "vs").number;
    double rad_per_row = member(member(cal, "row_time"), "rad_per_row").number;
    double pixel_size_m = scan.pixel_size_m * f;
    std::vector<float> dark = red.dark, flats = red.flats;
    int rows = rows_full, cols = cols_full;
    if (o.reduce) {
        // every position of the stored detector and of the full reconstruction over f; slopes and scales are ratios
        RawReader probe(scan.projections.files, scan.projections.dataset, rows_full, cols_full);
        const int chunk_rows = probe.storage(file_index[0]).chunk_rows;
        const std::vector<double> d64(dark.begin(), dark.end()), f64(flats.begin(), flats.end());
        int ry = 0, rx = 0;
        dark = calibration::at_resolution(d64.data(), 1, rows_full, cols_full, o.reduce, chunk_rows, &ry, &rx);
        flats = calibration::at_resolution(f64.data(), red.series(), rows_full, cols_full, o.reduce, chunk_rows, &ry, &rx);
        const int rows_r = rows_full / f, cols_r = cols_full / f;
        std::vector<float> wr(static_cast<std::size_t>((rows_full + f - 1) / f) * ((cols_full + f - 1) / f));
        const int wy = (rows_full + f - 1) / f, wx = (cols_full + f - 1) / f;
        for (int y = 0; y < wy; ++y)
            for (int x = 0; x < wx; ++x) wr[static_cast<std::size_t>(y) * wx + x] = w[static_cast<std::size_t>(y * f) * cols_full + x * f];  // the weight of the stored pixel a reduced value sits at
        w = std::move(wr);
        // the reconstruction counts the rows' timing from its middle row, which is the stored row f (rows_r - 1) / 2 and
        // not the stored middle (rows - 1) / 2 the description's angles are for
        const double shift = (rad_per_row * (f * (rows_r - 1) / 2.0 - (rows_full - 1) / 2.0)) * (180.0 / M_PI);
        for (double& a : T.angles_deg) a += shift;
        const float ff = static_cast<float>(f);
        for (float& v : T.row_offset) v /= ff;
        for (float& v : T.col_offset) v /= ff;
        T.slope_col /= f, T.slope_row /= f, T.scale_col /= f, T.scale_row /= f;
        for (double& v : eu) v /= f;
        for (double& v : ev) v /= f;
        c0 /= f, cs /= f, v0 /= f, vs /= f;
        rot_center /= f, rad_per_row *= f;
        rows = rows_r, cols = cols_r;
        (void)wx;
    }
    out.rot_center = rot_center;

    ScanData d;
    ScanTrajectory& tr = d.traj;
    tr.angles_deg = T.angles_deg;
    tr.z_translation_m = z;
    tr.pixel_size_m = pixel_size_m;
    tr.det_rows = rows;
    tr.det_cols = cols;
    tr.z_translation_scale = 1.0;
    tr.row_offset = T.row_offset;
    tr.row_slope = T.row_slope;
    tr.slope_col = static_cast<float>(T.slope_col);
    tr.col_offset = T.col_offset;
    tr.col_slope = T.col_slope;
    tr.slope_row = static_cast<float>(T.slope_row);
    tr.col_scale = T.col_scale;
    tr.scale_col = static_cast<float>(T.scale_col);
    tr.row_scale = T.row_scale;
    tr.scale_row = static_cast<float>(T.scale_row);
    tr.map_i = mi, tr.map_j = mj, tr.map_eu = eu, tr.map_ev = ev;
    tr.map_c0 = c0, tr.map_cs = cs, tr.map_v0 = v0, tr.map_vs = vs;
    d.control_A = control;
    d.flat_info.angles_deg.assign(scan.flats_unwrapped_angles_deg.begin(), scan.flats_unwrapped_angles_deg.end());
    d.dark = dark;
    const std::size_t px = dark.size();
    d.flats.resize(flats.size());
    for (std::size_t i = 0; i < flats.size(); ++i) d.flats[i] = flats[i] - dark[i % px];
    d.weight = w;
    d.reader = std::make_shared<RawReader>(scan.projections.files, scan.projections.dataset, rows_full, cols_full, o.reduce);
    if (o.cache_gb > 0) d.reader->set_cache_bytes(static_cast<std::size_t>(o.cache_gb * (1ull << 30)));
    d.file_index = file_index;
    d.frame_index = frame_index;

    // the parameters: the defaults with fbp, the ramp filter alone, no sld, no sharpness weighting, then o.params
    Json p = Json::parse(default_parameters());
    const auto put = [&](const std::string& key, Json v) {
        for (auto& kv : p.object)
            if (kv.first == key) return void(kv.second = std::move(v));
        p.object.emplace_back(key, std::move(v));
    };
    put("backprojector", Json::of(std::string("fbp")));
    put("filters", Json::list());
    put("sld", Json::of(std::string("")));
    put("sharpness_weighting", Json::of(std::string("off")));
    put("angle_subset", Json::of(std::string("")));
    Json filter_changes = Json::dict();
    for (const auto& [key, v] : Json::parse(o.params.empty() ? "{}" : o.params).object) {
        if (key == "mode") {
            if (v.string != "corrected") throw Error("a tile is reconstructed in mode corrected");
            continue;
        }
        if (is_filter_key(key) && key != "filters") filter_changes.set(key, v);
        else if (!p.find(key)) throw Error("unknown parameter '" + key + "'");
        else put(key, key == "filters" && v.is(Json::String) ? Json::parse(parse_filter_chain(v.string)) : v);
    }
    if (!filter_changes.object.empty()) put("filters", Json::parse(override_filter_chain(p.find("filters")->text(), filter_changes.text())));
    out.parameters = p.text();
    ReconConfig cfg;
    cfg.device = o.device;
    cfg.energy_kev = scan.energy_kev;
    cfg.distance_m = scan.distance_m;
    cfg.pixel_size_m = pixel_size_m;
    cfg.rot_center = static_cast<float>(rot_center);
    cfg.legacy_reference = false;
    apply_parameters(p, cfg, o.external_backprojector, nullptr);
    if (!cfg.grid_size) cfg.grid_size = static_cast<int>(std::nearbyint(2.0 * rot_center)) + 1;
    cfg.exposure_arc = 0.0;
    cfg.sharpness_weighting.clear();
    cfg.rolling_shutter = rad_per_row / (std::fabs(member(L, "step_deg").number) * (M_PI / 180.0)) * (rows - 1);
    std::vector<ScanData> scans;
    scans.push_back(std::move(d));
    out.reconstructor = std::make_shared<SliceReconstructor>(std::move(scans), cfg);
    return out;
}

}  // namespace helirec
