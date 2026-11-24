// vc_tomography/reconstruction.hpp: the reconstruction of a scan pair from its raw folders.
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

#include "vc_tomography/calibration/geometry_calibration.hpp"
#include "vc_tomography/calibration/record.hpp"
#include "vc_tomography/raw_reader.hpp"
#include "vc_tomography/reconstruction.hpp"
#include "vc_tomography/projection_store.hpp"
#include "io/json.hpp"
#include "vc_reader/native/sigv4.hpp"
#include "io/paths.hpp"
#include "io/text_file.hpp"
#include "io/json_values.hpp"

#ifndef VC_TOMOGRAPHY_SOURCE_DIR
#define VC_TOMOGRAPHY_SOURCE_DIR ""
#endif

namespace vc_tomography {

namespace {

using Log = std::function<void(const std::string&)>;
void say(const Log& log, const std::string& line) {
    if (log) log(line);
}

void check_reduce(int reduce) {
    if (reduce < 0 || reduce > 3) throw Error("reduce is 0, 1, 2 or 3");
}

using io::exists;
using io::read_text;

std::string base_name(const std::string& p) { return io::basename(p); }

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

using io::lower;
using io::strip;
using io::truth;

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
    double x = 0;
    if (!io::number_of(v, x)) throw Error("parameter " + key + ": a number, not " + v.text());
    return x;
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

// A scan's terms, made for the frames as stored (rows_full of them), as those of the frames read at 1 / f: every
// position of the stored detector and of the full reconstruction over f; slopes and scales are ratios. The
// reconstruction counts the rows' timing from its middle row, which is the stored row f (rows / f - 1) / 2 and not
// the stored middle (rows - 1) / 2 the terms' angles are for (rad_per_row: the angle between the exposure of one
// stored row and the next). from_middle: the timing counts the rows to both sides of the middle row, and moving the
// middle is no shift of the angle: it is left, the rows within (f - 1) / 2 of the middle off by that many rows' time.
// a scan's per-projection terms and detector map (geometry.hpp) into its trajectory
void set_terms(ScanTrajectory& t, const geometry::Terms& g, const geometry::Map& map) {
    t.angles_deg = g.angles_deg;
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
    t.map = map;
}

void terms_of_reduced_frames(geometry::Terms& T, int f, int rows_full, double rad_per_row, bool from_middle = false) {
    const int rows_r = rows_full / f;
    const double shift = from_middle ? 0.0 : (rad_per_row * (f * (rows_r - 1) / 2.0 - (rows_full - 1) / 2.0)) * (180.0 / M_PI);
    for (double& a : T.angles_deg) a += shift;
    const float ff = static_cast<float>(f);
    for (float& v : T.row_offset) v /= ff;
    for (float& v : T.col_offset) v /= ff;
    T.slope_col /= f, T.slope_row /= f, T.scale_col /= f, T.scale_row /= f;
}

template <class T>
std::vector<T> part_of(const std::vector<T>& v, const std::array<std::size_t, 2>& run) {
    return std::vector<T>(v.begin() + static_cast<std::ptrdiff_t>(run[0]), v.begin() + static_cast<std::ptrdiff_t>(run[1]));
}

}  // namespace

std::string build_revision() {
    const std::string dir = VC_TOMOGRAPHY_SOURCE_DIR;
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
    const NoiseSplit split = parse_noise_split(text_of(member(params, "noise_split")));
    cfg.noise_split_scale = static_cast<float>(split.scale);
    cfg.noise_split_gain = static_cast<float>(split.gain);
    cfg.noise_split_floor = static_cast<float>(split.floor);
    cfg.noise_split_seed = split.seed;
    cfg.noise_split_flat_frames = static_cast<float>(split.flat_frames);
    for (int i = 0; i < 2; ++i) {
        cfg.noise_split_smooth_share[i] = static_cast<float>(split.smooth_share[i]);
        cfg.noise_split_smooth_spacing[i] = static_cast<float>(split.smooth_spacing[i]);
    }
}

std::pair<std::string, std::string> geometry_calibration_file(const std::string& given, const std::string& cache_dir, const std::vector<std::string>& masters,
                                                              const std::string& row_time_us, Device device, const Log& log, const GeometryMaker& make,
                                                              const std::string& projection_store, int reduce) {
    if (!given.empty()) return {absolute(given), read_text(given)};
    if (cache_dir.empty())
        throw Error("no geometry calibration: give geometry=PATH (made by vc_tomography_calibration geometry), or a cache directory to make it in and keep it");
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
    say(log, "geometry calibration of the scans (vc_tomography_calibration geometry: layout, rows' timing, comparisons, solve) ...");
    calibration::geometry_calibration::Options opt;
    if (!row_time_us.empty()) opt.row_time = number_of(Json::of(row_time_us), "row_time_us") * 1e-6;
    opt.device = device;
    opt.log = log;
    opt.projections = projection_store, opt.reduce = reduce;
    const std::string text = make ? make(masters, opt) : calibration::geometry_calibration::calibrate_geometry(masters, opt);
    std::filesystem::create_directories(cache_dir);
    calibration::geometry_calibration::save_text(text, path);
    return {absolute(path), read_text(path)};
}

struct Reconstruction::Impl {
    std::string projections;
    Options options;
    Json params;
    std::vector<calibration::Scan> scans;   // the first (the main scan) has the rotation axis in its field
    std::vector<std::string> store_groups;  // per scan: its group of the projection store ("" each: the detector's files)

    // the geometry calibration in use
    std::string geometry_path, geometry_text, geometry_sha256;
    std::unique_ptr<geometry::Calibration> geometry;
    Json geometry_doc;
    std::vector<geometry::Poses> poses;                  // of every frame of each scan's record
    std::vector<std::array<std::size_t, 2>> used;        // per scan: the frames [first, end) the reconstruction is given
    std::string not_applied;
    geometry::MergedLayout layout;                       // in stored pixels
    std::vector<geometry::Terms> terms;                  // of the frames given, in the pixels of the frames read
    geometry::Map map;                                   // in the pixels of the frames read
    double rolling_shutter = 0;
    bool rolling_shutter_from_middle = false;
    double fan = 0;                                      // the calibration's, per column of the frames read
    std::tuple<std::string, double, std::string> geometry_key;

    calibration::IntensityCalibration intensity;
    std::shared_ptr<SliceReconstructor> rec;
    std::string resolved_sld = "{}";
    int lo = 0, hi = 0;

    const calibration::Scan& main() const { return scans[0]; }
    int factor() const { return 1 << options.reduce; }  // stored pixels per pixel of the frames read
    bool legacy() const { return member(params, "mode").string == "legacy_reference"; }

    std::tuple<std::string, double, std::string> geometry_request() const {
        const std::string path = strip(text_of(member(params, "geometry")));
        return {path, path.empty() ? 0.0 : mtime(path), strip(text_of(member(params, "row_time_us")))};
    }

    void geometry_file() {
        std::vector<std::string> masters;
        for (const calibration::Scan& sc : scans) masters.push_back(sc.master);
        std::tie(geometry_path, geometry_text) =
            geometry_calibration_file(strip(text_of(member(params, "geometry"))), options.cache_dir, masters, strip(text_of(member(params, "row_time_us"))), options.device,
                                      options.log, nullptr, options.projection_store, options.reduce);
    }

    // the geometry calibration, the poses it gives the frames, the merged frame laid out from them, the intensity
    // calibration in that frame and every scan's terms
    void calibrate() {
        if (legacy() && (options.reduce || !options.projection_store.empty()))
            throw Error("mode legacy_reference reconstructs the detector's frames as stored, not a projection store's or reduced ones");
        geometry_file();
        geometry_sha256 = vcr::to_hex(vcr::Sha256::of(geometry_text));
        geometry_doc = Json::parse(geometry_text);
        geometry = std::make_unique<geometry::Calibration>(geometry::Calibration::parse(geometry_text));
        const std::vector<std::string> made = geometry->scans();
        const std::size_t S = scans.size();
        for (std::size_t k = 0; k < S; ++k)
            if (made.size() != S || base_name(made[k]) != base_name(scans[k].master))
                throw Error("geometry " + geometry_path + " was made from " + (made.size() > k ? made[k] : "other scans") + ", not from " + scans[k].master +
                            " (first scan: the one with the rotation axis in its field)");
        std::vector<std::vector<double>> angles(S), H(S);
        for (std::size_t k = 0; k < S; ++k) angles[k] = scans[k].unwrapped_angles_deg, H[k] = geometry->heights_rows(scans[k].z_translation_m, scans[k].pixel_size_m);
        used.assign(S, {0, 0});
        poses = geometry->scan_poses(angles, H, &not_applied);
        for (std::size_t k = 0; k < S; ++k) {
            // the frames the calibration was made on: those are the frames it places, and the frames on disk
            const auto runs = geometry->frames_placed(k);
            if (runs.size() != 1 || runs[0][1] > angles[k].size())
                throw Error("the geometry calibration holds " + std::to_string(runs.size()) + " separate runs of frames of " + scans[k].name +
                            " (stretches that do not follow one another were on disk): the reconstruction takes one run of frames per scan");
            used[k] = runs[0];
        }
        if (legacy())
            for (std::size_t k = 0; k < S; ++k)
                if (used[k][0] != 0 || used[k][1] != angles[k].size()) throw Error("mode legacy_reference takes every frame of the scans, and the geometry calibration was made on a part of them");
        std::vector<std::size_t> first;
        for (std::size_t k = 0; k < S; ++k) first.push_back(used[k][0]);
        layout = geometry->merged_layout(poses, H, first);
        // the merged frame in the pixels of the frames read
        const int f = factor();
        geometry::MergedLayout in_frames = layout;
        if (f > 1) {
            in_frames.cor_main_detector /= f;
            in_frames.vector = {layout.vector[0] / f, layout.vector[1] / f};
            for (auto& v : in_frames.vectors) v = {v[0] / f, v[1] / f};
        }
        std::vector<const calibration::Scan*> all;
        for (const calibration::Scan& sc : scans) all.push_back(&sc);
        intensity = calibration::calibrate_intensity(all, in_frames, options.cache_dir, options.log, options.progress, options.projection_store, options.reduce);
        const Json& lay = member(geometry_doc, "layout");
        const int rows_stored = static_cast<int>(member(lay, "rows").number);
        // the helix's first frame lies where the record puts it, counted from the first frame given
        const double origin = H[0].at(geometry->helix_frames()[0]) - H[0][used[0][0]];
        const double frames_per_row = geometry->rolling_shutter() / (rows_stored - 1);  // the rows' timing, per stored row
        terms.assign(S, {});
        for (std::size_t k = 0; k < S; ++k) {
            const calibration::ScanPlacement& pl = intensity.placement[k];
            const bool whole = used[k][0] == 0 && used[k][1] == angles[k].size();
            geometry::Poses given = poses[k];
            if (!whole)
                for (auto* v : {&given.row, &given.row_per_col, &given.axis, &given.col_per_row, &given.col_scale, &given.row_scale, &given.angle_deg, &given.angle_deg_per_row})
                    *v = part_of(*v, used[k]);
            terms[k] = geometry::reconstruction_terms(given, whole ? H[k] : part_of(H[k], used[k]), member(lay, "row_sign").number, rows_stored, intensity.rot_center * f,
                                                      (pl.shift_x - pl.x_readjustment) * f, pl.z_readjustment * f, origin);
            if (f > 1) terms_of_reduced_frames(terms[k], f, rows_stored, frames_per_row * recorded_angle_step(part_of(angles[k], used[k])) * (M_PI / 180.0), geometry->rows_from_middle());
        }
        map = f > 1 ? geometry->map().reduced(f) : geometry->map();
        // the time from the first row to the last of the frames read, in frame periods
        rolling_shutter = f > 1 ? frames_per_row * f * (rows_stored / f - 1) : geometry->rolling_shutter();
        rolling_shutter_from_middle = geometry->rows_from_middle();
        fan = geometry->fan() * f;
        geometry_key = geometry_request();
    }

    ScanData scan_data(std::size_t k) const {
        const calibration::Scan& scan = scans[k];
        const calibration::DarksFlats& reduced = intensity.reduced[k];
        const calibration::ScanPlacement& pl = intensity.placement[k];
        const std::array<std::size_t, 2>& run = used[k];
        const bool whole = run[0] == 0 && run[1] == scan.unwrapped_angles_deg.size();
        const int f = factor();
        ScanData d;
        ScanTrajectory& t = d.traj;
        t.z_translation_m = whole ? scan.z_translation_m : part_of(scan.z_translation_m, run);
        t.pixel_size_m = scan.pixel_size_m * f;
        t.det_rows = scan.rows / f;
        t.det_cols = scan.cols / f;
        t.shift_x = pl.shift_x;
        t.x_readjustment = static_cast<float>(pl.x_readjustment);
        t.z_readjustment = static_cast<float>(pl.z_readjustment);
        // where slits light a part of the detector, a height is seen through the lit rows alone (the weights are
        // cropped to them: calibrate_intensity)
        if (intensity.lit[1] > intensity.lit[0] && intensity.lit[1] - intensity.lit[0] < t.det_rows) t.row_first = intensity.lit[0], t.row_end = intensity.lit[1];
        d.control_A = whole ? scan.control_A : part_of(scan.control_A, run);
        d.flat_info.angles_deg.assign(scan.flats_unwrapped_angles_deg.begin(), scan.flats_unwrapped_angles_deg.end());
        d.dark = reduced.dark;
        const std::size_t px = reduced.dark.size();
        d.flats.resize(reduced.flats.size());
        for (std::size_t i = 0; i < d.flats.size(); ++i) d.flats[i] = reduced.flats[i] - reduced.dark[i % px];
        d.weight = intensity.weight[k];
        // where the frames are read: the detector's files, or the same frames of the projection store
        calibration::FrameList source = scan.projections;
        if (!whole) {
            std::vector<std::int64_t> index(run[1] - run[0]);
            for (std::size_t i = 0; i < index.size(); ++i) index[i] = static_cast<std::int64_t>(run[0] + i);
            source = scan.projections.subset(index);
        }
        if (!store_groups[k].empty()) source = frames_in_projection_store(source, store_groups[k]);
        d.reader = std::make_shared<RawReader>(source.files, source.dataset, scan.rows, scan.cols, options.reduce);
        d.file_index = source.file_index;
        d.frame_index = source.frame_index;
        if (legacy()) {  // the reference: the recorded angles and heights, one axis and one vector (the merged frame)
            t.angles_deg = scan.unwrapped_angles_deg;
            t.z_translation_scale = 1.0;
            return d;
        }
        set_terms(t, terms[k], map);
        t.z_translation_scale = terms[k].z_translation_scale;
        return d;
    }

    // which scan `only_scan` names: "main", "companion" (the second), "companion2", .., or the kind a scan's name
    // ends in ("HA", "TA", "TA2", ..)
    std::size_t scan_named(const std::string& only) const {
        if (only == "main") return 0;
        for (std::size_t k = 1; k < scans.size(); ++k)
            if (only == (k == 1 ? std::string("companion") : "companion" + std::to_string(k))) return k;
        for (std::size_t k = 0; k < scans.size(); ++k)
            if (const auto t = calibration::parse_tetra_name(scans[k].name); t && lower(t->kind) == only) return k;
        std::string names = "'main'";
        for (std::size_t k = 1; k < scans.size(); ++k) names += k == 1 ? ", 'companion'" : ", 'companion" + std::to_string(k) + "'";
        throw Error("only_scan must be '', " + names + " or the kind a scan's name ends in (HA, TA, TA2, ..)");
    }

    double beam_divergence() const { return legacy() || !truth(member(params, "beam_divergence")) ? 0.0 : fan; }
    double exposure_arc() const {
        const Json& v = member(params, "exposure_arc");
        if (v.is(Json::String) && lower(strip(v.string)) == "auto") return main().exposure_arc.value_or(0.0);
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
        cfg.energy_kev = main().energy_kev;
        cfg.distance_m = main().distance_m;
        cfg.pixel_size_m = main().pixel_size_m * factor();
        cfg.rot_center = static_cast<float>(intensity.rot_center);
        cfg.legacy_reference = legacy_mode;
        apply_parameters(params, cfg, options.external_backprojector, &resolved_sld);
        std::vector<ScanData> data;
        double axis = intensity.rot_center;
        double width = intensity.merged_width;  // the columns the slices are made of
        const std::string only = lower(strip(text_of(member(params, "only_scan"))));
        if (only.empty()) {
            for (std::size_t k = 0; k < scans.size(); ++k) data.push_back(scan_data(k));
        } else {
            // The scan alone in its own columns, the axis where the merged frame has it against them. Every
            // projection's terms are counted from the axis, so they stay as they are.
            ScanData d = scan_data(scan_named(only));
            axis -= static_cast<double>(d.traj.shift_x) - d.traj.x_readjustment;
            d.traj.shift_x = 0, d.traj.x_readjustment = 0;
            cfg.rot_center = static_cast<float>(axis);
            width = d.traj.det_cols;
            data.push_back(std::move(d));
        }
        // The slice reaches as far from the axis as the columns do: 2 x axis rounded to even, + 1 where the columns
        // lie on the axis' left (the reference's layout); where they lie on its right (the axis in the first
        // scan's left half), twice their reach on that side.
        const double reach = std::max(axis, width - 1.0 - axis);
        if (!cfg.grid_size) cfg.grid_size = static_cast<int>(std::nearbyint(2.0 * reach)) + 1;
        cfg.exposure_arc = exposure_arc();
        cfg.sharpness_weighting = legacy_mode ? std::string() : sharpness_weighting();
        cfg.rolling_shutter = legacy_mode ? 0.0 : rolling_shutter;
        cfg.rolling_shutter_from_middle = rolling_shutter_from_middle;
        cfg.fan_per_column = beam_divergence();
        rec = std::make_shared<SliceReconstructor>(std::move(data), cfg);
        std::tie(lo, hi) = rec->doable();
        if (rec->angle_interp() == "views" && exposure_arc() != 0)
            say(options.log, "note: exposure_arc is disabled with angle_interp=views (not deconvolved; angle_interp=nearest applies it)");
    }

    Json info_json() const { return Json::parse(rec->info_json()); }  // (doable: [lo, hi), as build() found them)
};

Reconstruction::Reconstruction(const std::string& projections, const std::string& params, Options options) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.projections = projections;
    I.options = std::move(options);
    check_reduce(I.options.reduce);
    if (I.options.reduce && I.options.projection_store.empty()) throw Error("frames read reduced are a projection store's: give the store (vc_tomography_projections_zarr, or the downloader's --zarr)");
    std::vector<calibration::Scan> found = calibration::discover_scans(projections);
    std::string said;
    for (const int k : calibration::pair_companions(found)) {  // the family: the scan with the axis in its field first
        I.scans.push_back(std::move(found[static_cast<std::size_t>(k)]));
        said += (said.empty() ? "" : " + ") + I.scans.back().name + " (" + std::to_string(I.scans.back().unwrapped_angles_deg.size()) + (said.empty() ? " projections)" : ")");
    }
    say(I.options.log, "scans: " + said);
    I.store_groups.assign(I.scans.size(), std::string());
    for (std::size_t k = 0; k < I.scans.size(); ++k) {
        if (I.options.projection_store.empty()) I.scans[k].check_files();
        else I.store_groups[k] = projection_store_tile_group(I.options.projection_store, I.scans[k].sample_name.empty() ? I.scans[k].name : I.scans[k].sample_name);
    }
    if (!I.options.projection_store.empty())
        say(I.options.log, "frames from " + I.options.projection_store + " at 1/" + std::to_string(I.factor()) + ", those the geometry calibration was made on");
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
const calibration::Scan& Reconstruction::main() const { return impl_->scans[0]; }
const calibration::Scan& Reconstruction::companion() const {
    if (impl_->scans.size() < 2) throw Error("the acquisition is of one scan: it has no companion");
    return impl_->scans[1];
}
const std::vector<calibration::Scan>& Reconstruction::scans() const { return impl_->scans; }
std::pair<std::size_t, std::size_t> Reconstruction::frames(std::size_t scan) const { return {impl_->used.at(scan)[0], impl_->used.at(scan)[1]}; }
int Reconstruction::reduce() const { return impl_->options.reduce; }
const calibration::IntensityCalibration& Reconstruction::intensity() const { return impl_->intensity; }
std::shared_ptr<SliceReconstructor> Reconstruction::reconstructor() const { return impl_->rec; }
std::pair<int, int> Reconstruction::doable() const { return {impl_->lo, impl_->hi}; }
Device Reconstruction::device() const { return impl_->options.device; }
FilterChain filter_chain(const std::string& chain) {
    const std::string t = chain;
    std::size_t i = t.find_first_not_of(" \t\n\r");
    return filter_chain_of(i != std::string::npos && t[i] == '[' ? Json::parse(t) : Json::of(t));
}
std::string Reconstruction::mode() const { return member(impl_->params, "mode").string; }
double Reconstruction::rot_center() const { return impl_->intensity.rot_center; }
double Reconstruction::rolling_shutter() const { return impl_->legacy() ? 0.0 : impl_->rolling_shutter; }
double Reconstruction::beam_divergence() const { return impl_->beam_divergence(); }
double Reconstruction::exposure_arc() const { return impl_->exposure_arc(); }
std::string Reconstruction::sharpness_weighting() const { return impl_->sharpness_weighting(); }
std::string Reconstruction::resolved_sld() const { return impl_->resolved_sld; }
std::string Reconstruction::info() const { return impl_->info_json().text(); }

SliceInfo Reconstruction::slice_info(double fraction) const {
    if (!(0.0 < fraction && fraction < 1.0)) throw Error("fraction must be in (0, 1)");
    const Impl& I = *impl_;
    const int k = static_cast<int>(std::nearbyint(fraction * (I.hi - I.lo)));  // half to even (the reference's)
    const int h = I.hi - 1 - k;
    // (of the frames read, counted from the first frame given)
    const double z0 = I.main().z_translation_m.at(I.used[0][0]);
    const double px_mm = I.main().pixel_size_m * I.factor() * 1e3;
    SliceInfo s;
    s.fraction = fraction;
    s.view_height = h;
    s.nabu_index = 999999 - (I.hi - h);
    s.height_mm = px_mm * ((I.main().rows / I.factor() - 1) / 2.0 - h) - z0 * 1e3;
    return s;
}

std::string Reconstruction::provenance() const {
    const Impl& I = *impl_;
    const std::string m = mode();
    const bool legacy = I.legacy();
    Json pinned = Json::list();
    if (legacy) {
        for (const auto& k : io::sorted_keys(Json::parse(legacy_reference_parameters()))) pinned.add(Json::of(k));
    }
    const Json conventions = Json::parse(mode_conventions(m));
    const calibration::IntensityCalibration& cal = I.intensity;
    Json map = Json::dict();
    const Json& doc_map = member(I.geometry_doc, "map");
    for (const char* k : {"c0", "cs", "v0", "vs", "i", "j", "eu", "ev"}) map.set(k, member(doc_map, k));
    Json layout = Json::dict()
                      .set("cor_main_detector", Json::of(I.layout.cor_main_detector))
                      .set("vector", numbers({I.layout.vector[0], I.layout.vector[1]}))
                      .set("measured", Json::dict()
                                           .set("axis_first_scan", Json::of(I.layout.axis_first_scan))
                                           .set("axis_second_scan", Json::of(I.layout.axis_second_scan))
                                           .set("second_scan_rows", Json::of(I.layout.second_scan_rows))
                                           .set("quantum", Json::of(I.layout.quantum)));
    if (I.scans.size() != 2) {  // (a pair's is its one vector, as it always was)
        Json vectors = Json::list(), measured = Json::list();
        for (const auto& v : I.layout.vectors) vectors.add(numbers({v[0], v[1]}));
        for (const auto& v : I.layout.measured) measured.add(numbers({v[0], v[1]}));
        layout.set("vectors", vectors).set("measured_axis_and_rows", measured);
    }
    // the frames read, where they are not the detector's own, whole: the store and its level, each scan's frames
    Json frames = Json();
    if (!I.options.projection_store.empty()) {
        Json per_scan = Json::list();
        for (std::size_t k = 0; k < I.scans.size(); ++k)
            per_scan.add(Json::dict()
                             .set("scan", Json::of(I.scans[k].name))
                             .set("group", Json::of(I.store_groups[k]))
                             .set("first", Json::whole_number(static_cast<long long>(I.used[k][0])))
                             .set("end", Json::whole_number(static_cast<long long>(I.used[k][1]))));
        frames = Json::dict()
                     .set("projection_store", Json::of(I.options.projection_store))
                     .set("reduce", Json::whole_number(I.options.reduce))
                     .set("pixels", Json::of("positions below the geometry's entry are of the frames read (1/" + std::to_string(I.factor()) + " of the stored ones); the geometry's own are stored pixels"))
                     .set("scans", per_scan);
    }
    const Json geometry = Json::dict()
                              .set("path", Json::of(I.geometry_path))
                              .set("sha256", Json::of(I.geometry_sha256))
                              .set("scans", member(I.geometry_doc, "scans"))
                              .set("map", map)
                              .set("row_time_s_per_row", member(member(I.geometry_doc, "row_time"), "seconds_per_row"))
                              .set("rows_from_middle", Json::of_bool(I.rolling_shutter_from_middle))
                              .set("merged_frame", layout)
                              .set("applied", Json::of(legacy ? "the merged frame alone (mode legacy_reference)" : "the detector map and every projection's pose and row timing; the beam's divergence where resolved_beam_divergence is not 0 and the backprojector is fbp"))
                              .set("measured_not_applied", I.not_applied.empty() ? Json() : Json::parse(I.not_applied));
    Json out = Json::dict()
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
        .set("resolved_beam_divergence", Json::of(beam_divergence()))
        .set("geometry", geometry)
        .set("reconstructor", I.info_json())
        .set("pixel_size_m", Json::of(I.main().pixel_size_m * I.factor()))
        .set("energy_kev", Json::of(I.main().energy_kev))
        .set("distance_m", Json::of(I.main().distance_m))
        .set("vc_tomography", Json::dict().set("git", Json::of(build_revision())));
    if (!frames.is(Json::Null)) out.set("frames", frames);
    return out.text();
}

// ---------------------------------------------------------------- one tile of a mosaic

TileReconstruction tile_reconstructor(const std::string& master, const std::string& description, const std::string& label, const TileOptions& o) {
    check_reduce(o.reduce);
    const Json cal = Json::parse(description);
    const Json& L = member(cal, "layout");
    TileReconstruction out;
    out.reduce = o.reduce;
    out.scan = calibration::open_scan(master);
    const calibration::Scan& scan = out.scan;
    const int rows_full = scan.rows, cols_full = scan.cols;
    if (o.reduced) {
        const auto& red = *o.reduced;
        const std::size_t pixels = static_cast<std::size_t>(rows_full) * cols_full;
        if (red.rows != rows_full || red.cols != cols_full || red.dark.size() != pixels ||
            red.flats.size() != pixels * scan.flats.size())
            throw Error("tile " + label + ": supplied darks and flats must match the detector dimensions and flat series count");
    }
    // a joint geometry calibration (vc_tomography_calibration geometry), of which `label` names a scan
    const geometry::Calibration joined = geometry::Calibration::parse(description);
    const std::size_t joint_scan = joined.scan_named(label);
    if (joined.rows() != rows_full || joined.cols() != cols_full) throw Error("the geometry calibration is of frames of another size than " + scan.name + "'s");
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
    {  // the helix' frames the calibration was made on
        const std::array<std::size_t, 2> on_disk = joined.frames_on_disk(joint_scan);
        if (static_cast<std::int64_t>(on_disk[0]) < before) throw Error("the geometry calibration's helix does not start where " + scan.name + "'s does: it was made from another scan");
        out.first_frame = static_cast<std::int64_t>(on_disk[0]) - before, out.end_frame = static_cast<std::int64_t>(on_disk[1]) - before;
    }
    std::vector<std::int64_t> idx;
    std::vector<double> frame, angle, z, height_rows, control;
    std::vector<std::int32_t> file_index, frame_index;
    const double row_sign = member(L, "row_sign").number;
    if (before + out.end_frame > static_cast<std::int64_t>(scan.unwrapped_angles_deg.size()) || out.first_frame < 0) throw Error("tile " + label + ": the frames named are not among the scan's");
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
    // where the helix frames are read: the detector's files, or the same frames of a projection store
    calibration::FrameList source;
    source.files = scan.projections.files, source.dataset = scan.projections.dataset;
    source.file_index = file_index, source.frame_index = frame_index;
    if (!o.projections.empty()) {
        source = frames_in_projection_store(source, o.projections);
        file_index = source.file_index, frame_index = source.frame_index;
        say(o.log, "[" + label + "] projections from " + o.projections);
    }
    const double axis_column = joined.axis_column(joint_scan);
    const int f = 1 << o.reduce;
    // The slice's grid has its centre on a pixel or between two, and the reconstruction puts the axis there: the axis'
    // column is rounded to half a pixel of the reconstruction (the rest goes into every frame's column offset), so
    // that rot_center is where the axis lies in the slice.
    double rot_center = f * std::nearbyint(2 * axis_column / f) / 2.0;
    // The darks and flats: the detector's files reduced at full resolution; or, where a projection store is given
    // and the detector's dark files are not on disk (a tile kept in the store alone), the store's float images at
    // the level `reduce`, which were taken through the frames' low-pass when the store was made.
    const bool store_flats = !o.projections.empty() && !o.reduced && (scan.darks.files.empty() || !std::filesystem::exists(scan.darks.files[0]));
    const std::size_t px_full = static_cast<std::size_t>(rows_full) * static_cast<std::size_t>(cols_full);
    const bool right = axis_column > (cols_full - 1) / 2.0;
    const double far = right ? 50.0 : std::min(2000.0, axis_column), near = right ? std::min(2000.0, cols_full - axis_column) : 50.0;
    std::vector<float> w, store_dark, store_flat_series;
    if (store_flats) {
        if (o.weight) throw Error("tile " + label + ": a weight function takes the stored frames' weight map, and the darks and flats are a projection store's");
        say(o.log, "[" + label + "] darks and flats from " + o.projections);
        const calibration::DarksFlats held = calibration::store_darks_flats(scan, o.projections, o.reduce);
        store_dark = held.dark, store_flat_series = held.flats;
        const int rr = held.rows, rc = held.cols;
        if (rr != rows_full / f || rc != cols_full / f) throw Error("tile " + label + ": the store's darks and flats are not of the level's size");
        // the weight map formed on the level's grid, its widths divided with the pixels
        const std::size_t px = static_cast<std::size_t>(rr) * static_cast<std::size_t>(rc);
        std::vector<const float*> stacks;
        for (std::size_t s = 0; s < scan.flats.size(); ++s) stacks.push_back(store_flat_series.data() + s * px);
        const std::vector<float> wm = calibration::weight_map(stacks, rr, rc, 15.0 / f, 10.0 / f);
        const std::vector<float> mask = calibration::apodisation_mask(rc, static_cast<float>(far / f), static_cast<float>(near / f), 1e-8f);
        w.resize(px);
        for (std::size_t r = 0; r < static_cast<std::size_t>(rr); ++r)
            for (std::size_t c = 0; c < static_cast<std::size_t>(rc); ++c) w[r * rc + c] = wm[r * rc + c] * mask[c];
    } else {
        if (!o.reduced) say(o.log, "[" + label + "] darks and flats ...");
        out.reduced = o.reduced ? *o.reduced : calibration::reduced_darks_flats(scan, o.cache_dir, o.log);
        std::vector<const float*> stacks;
        for (std::size_t s = 0; s < out.reduced.series(); ++s) stacks.push_back(out.reduced.flats.data() + s * px_full);
        const std::vector<float> wm = calibration::weight_map(stacks, rows_full, cols_full);
        const std::vector<float> mask = calibration::apodisation_mask(cols_full, static_cast<float>(far), static_cast<float>(near), 1e-8f);
        w.resize(px_full);
        for (std::size_t r = 0; r < static_cast<std::size_t>(rows_full); ++r)
            for (std::size_t c = 0; c < static_cast<std::size_t>(cols_full); ++c) w[r * cols_full + c] = wm[r * cols_full + c] * mask[c];
        if (o.weight) o.weight(w, rows_full, cols_full);
    }
    const calibration::DarksFlats& red = out.reduced;
    double step_deg = 0, rad_per_row = 0;  // the angle per frame; the angle between the exposure of one stored row and the next
    bool rows_from_middle = false;         // ... the rows counted from the middle row to both ends
    {
        // the scan's own frame: its poses under the calibration with its place and height left out (geometry.hpp)
        const std::size_t n_all = scan.unwrapped_angles_deg.size();
        std::vector<double> all_angle(scan.unwrapped_angles_deg.begin(), scan.unwrapped_angles_deg.end()), all_height(n_all);
        for (std::size_t i = 0; i < n_all; ++i) all_height[i] = row_sign * scan.z_translation_m[i] / scan.pixel_size_m;
        const geometry::Poses all = joined.own_poses(joint_scan, all_angle, all_height);
        geometry::Poses own;
        const auto part = [&](const std::vector<double>& v) {
            return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(before + out.first_frame), v.begin() + static_cast<std::ptrdiff_t>(before + out.end_frame));
        };
        own.row = part(all.row), own.row_per_col = part(all.row_per_col), own.axis = part(all.axis), own.col_per_row = part(all.col_per_row);
        own.col_scale = part(all.col_scale), own.row_scale = part(all.row_scale), own.angle_deg = part(all.angle_deg), own.angle_deg_per_row = part(all.angle_deg_per_row);
        own.scale_col = all.scale_col;
        // the slices are numbered by the model's height of the sample as this scan holds it, less row_origin; none
        // given: the slice h is about what the row h saw at the middle frame on disk
        out.row_origin = o.row_origin ? *o.row_origin : -own.row[own.row.size() / 2];
        // (row_origin: reconstruction_terms adds it to the poses' rows, and the slice h is read at the row h + row + origin)
        out.terms = geometry::reconstruction_terms(own, height_rows, row_sign, rows_full, rot_center, 0.0, 0.0, out.row_origin);
        if (angle.size() < 2) throw Error("tile " + label + ": one frame on disk");
        step_deg = recorded_angle_step(angle);
        rad_per_row = member(member(cal, "row_time"), "frames_per_row").number * step_deg * (M_PI / 180.0);
        rows_from_middle = own.rows_from_middle = all.rows_from_middle;
    }
    geometry::Terms& T = out.terms;
    geometry::Map map = geometry::parse_map(member(cal, "map").text());
    double pixel_size_m = scan.pixel_size_m * f;
    std::vector<float> dark = store_flats ? store_dark : red.dark, flats = store_flats ? store_flat_series : red.flats;
    int rows = rows_full, cols = cols_full;
    if (o.reduce) {
        // every position of the stored detector and of the full reconstruction over f; slopes and scales are ratios
        const int rows_r = rows_full / f, cols_r = cols_full / f;
        const int wx = (cols_full + f - 1) / f;
        if (!store_flats) {  // (a store's are of the level already)
            RawReader probe(source.files, source.dataset, rows_full, cols_full);
            const int chunk_rows = probe.storage(file_index[0]).chunk_rows;
            const std::vector<double> d64(dark.begin(), dark.end()), f64(flats.begin(), flats.end());
            int ry = 0, rx = 0;
            dark = calibration::at_resolution(d64.data(), 1, rows_full, cols_full, o.reduce, chunk_rows, &ry, &rx);
            flats = calibration::at_resolution(f64.data(), red.series(), rows_full, cols_full, o.reduce, chunk_rows, &ry, &rx);
            std::vector<float> wr(static_cast<std::size_t>((rows_full + f - 1) / f) * ((cols_full + f - 1) / f));
            const int wy = (rows_full + f - 1) / f;
            for (int y = 0; y < wy; ++y)
                for (int x = 0; x < wx; ++x) wr[static_cast<std::size_t>(y) * wx + x] = w[static_cast<std::size_t>(y * f) * cols_full + x * f];  // the weight of the stored pixel a reduced value sits at
            w = std::move(wr);
        }
        terms_of_reduced_frames(T, f, rows_full, rad_per_row, rows_from_middle);
        map = map.reduced(f);
        rot_center /= f, rad_per_row *= f;
        rows = rows_r, cols = cols_r;
        (void)wx;
    }
    out.rot_center = rot_center;

    ScanData d;
    ScanTrajectory& tr = d.traj;
    set_terms(tr, T, map);
    tr.z_translation_m = z;
    tr.pixel_size_m = pixel_size_m;
    tr.det_rows = rows;
    tr.det_cols = cols;
    tr.z_translation_scale = 1.0;
    d.control_A = control;
    d.flat_info.angles_deg.assign(scan.flats_unwrapped_angles_deg.begin(), scan.flats_unwrapped_angles_deg.end());
    d.dark = dark;
    const std::size_t px = dark.size();
    d.flats.resize(flats.size());
    for (std::size_t i = 0; i < flats.size(); ++i) d.flats[i] = flats[i] - dark[i % px];
    d.weight = w;
    d.reader = std::make_shared<RawReader>(source.files, source.dataset, rows_full, cols_full, o.reduce);
    if (o.cache_gb > 0) d.reader->set_cache_bytes(static_cast<std::size_t>(o.cache_gb * (1ull << 30)));
    d.file_index = file_index;
    d.frame_index = frame_index;

    // the parameters: the defaults with fbp, the ramp filter alone, no sld, no sharpness weighting, then o.params
    Json p = Json::parse(default_parameters());
    p.put("backprojector", Json::of(std::string("fbp")));
    p.put("filters", Json::list());
    p.put("sld", Json::of(std::string("")));
    p.put("sharpness_weighting", Json::of(std::string("off")));
    p.put("angle_subset", Json::of(std::string("")));
    p.put("noise_split", Json::of(std::string("")));
    p.put("only_scan", Json::of(std::string("")));
    // o.params as any reconstruction's are read (update_parameters: older names mapped, removed ones refused)
    p = Json::parse(update_parameters(p.text(), o.params.empty() ? "{}" : o.params));
    if (member(p, "mode").string != "corrected") throw Error("a tile is reconstructed in mode corrected");
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
    cfg.rolling_shutter = rad_per_row / (step_deg * (M_PI / 180.0)) * (rows - 1);
    cfg.rolling_shutter_from_middle = rows_from_middle;
    cfg.fan_per_column = truth(member(p, "beam_divergence")) ? joined.fan() * f : 0.0;
    std::vector<ScanData> scans;
    scans.push_back(std::move(d));
    out.reconstructor = std::make_shared<SliceReconstructor>(std::move(scans), cfg);
    return out;
}

}  // namespace vc_tomography
