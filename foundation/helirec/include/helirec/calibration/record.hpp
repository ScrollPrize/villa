#pragma once
// The acquisition record of one scan, read verbatim.
//
// helirec::calibration is the calibration built from the bottom up: it starts from what the acquisition recorded
// and from the raw frames, and takes nothing from helirec::calib, helirec/pipeline.hpp or the Python calibration
// (no axis, pitch, registration, distortion or flat-field model of theirs). Nothing in this directory includes
// or calls them; the only thing shared is the raw frame reader (helirec/raw_reader.hpp), which returns the bytes
// on disk.
//
// A record is the master file a BLISS tomo sequence writes (ESRF BM18): entry 1.1 describes the sequence, each
// further entry N.1 is one stretch of frames (darks, flats, a turn of projections, a helix), and N.2 holds the
// beam monitors sampled during N.1. Everything is stored as the file states it, in the file's units. Nothing is
// derived, corrected or defaulted here: a value the file does not hold is absent.
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "helirec/common.hpp"

namespace helirec::calibration {

enum class Role { Dark, Flat, Projection, Unknown };
const char* role_name(Role r);

// A scalar or small array of the record: numbers (bools as 0/1) or strings.
struct Value {
    std::vector<double> num;
    std::vector<std::string> str;
};

// The scalars and small arrays below one group, keyed by their path relative to it ("scan/tomo_n").
struct Values {
    std::map<std::string, Value> items;
    bool has(const std::string& key) const { return items.count(key) > 0; }
    bool has_number(const std::string& key) const;
    double number(const std::string& key) const;                // throws if absent or not a number
    double number(const std::string& key, double fallback) const;
    std::string text(const std::string& key) const;             // "" if absent
    const std::vector<double>& numbers(const std::string& key) const;
    const std::vector<std::string>& texts(const std::string& key) const;
};

// One piece of a stretch's (virtual) frame dataset: `count` frames starting at frame `first` of the stretch are
// frames [source_first, source_first + count) of `dataset` in `file`.
struct FramePiece {
    std::string file;  // absolute path
    std::string dataset;
    std::int64_t first = 0, count = 0, source_first = 0;
    // false: the file is not on disk (a partial copy of the acquisition). Such frames are never read: the
    // master's virtual dataset would return zeros for them.
    bool present = true;
};

// Sampling counters (ring current and others), each with the software time of its reading.
struct Monitor {
    std::string entry;  // the entry they were read from ("" when the record holds none)
    std::vector<double> epoch, elapsed, current, lifetime, refill, sbcurr;
};

// What the detector software (Lima) wrote into the frame files of a stretch.
struct FrameFiles {
    Values header;                      // of the first file: binning, region, flips, sensor size, ...
    std::vector<double> time_of_frame;  // software time of each frame's arrival [s], all files in order
    std::vector<std::int64_t> frames_in_file;
};

struct Stretch {
    std::string entry;  // "4.1"
    std::string title, start_time, end_time, end_reason;
    bool has_image_key = false;
    int image_key = -1;  // 2 dark, 1 flat, 0 projection
    Role role = Role::Unknown;
    std::int64_t n_frames = 0;
    int rows = 0, cols = 0;
    std::vector<FramePiece> pieces;

    // Latched by the acquisition hardware, per frame (projection stretches). *_raw hold the counts at both edges
    // of every exposure gate: 2 n + 1 values, even indices at the start of an exposure, odd at its end.
    std::vector<std::int32_t> timer_raw, rot_raw, z_raw;
    std::vector<double> epoch_trig;                               // Unix time at the start of the exposure [s]
    std::vector<double> timer_trig, timer_delta, timer_period;    // [s]
    std::vector<double> rot_trig, rot_center, rot_delta, rot_period;  // [deg]; center may be wrapped to one turn
    std::string rot_center_name;                                  // the dataset rot_center came from
    std::vector<double> z_trig, z_center, z_delta, z_period;      // [mm]; empty when the axis stage is at rest

    // Software times (dark and flat stretches)
    std::vector<double> epoch, elapsed;

    // Per-frame statistics the detector software computed over the rectangle (stat_x, stat_y, stat_w, stat_h)
    std::vector<double> stat_avg, stat_min, stat_max, stat_std, stat_sum;
    int stat_x = 0, stat_y = 0, stat_w = 0, stat_h = 0;

    std::map<std::string, double> positioners_start, positioners_end;  // every numeric motor position
    Values scan_parameters;  // instrument/fscan_parameters
    Values technique;        // the entry's technique group
    Values machine;          // instrument/machine at the start of the stretch
    Monitor monitor;
    FrameFiles files;

    bool latched() const { return !rot_trig.empty(); }
    // the frames whose files are on disk: [first, end) ranges in order, and their number
    std::vector<std::pair<std::int64_t, std::int64_t>> frames_on_disk() const;
    std::int64_t n_on_disk() const;
    bool on_disk(std::int64_t frame) const;
};

struct Record {
    std::string path, directory;
    std::string label;      // short name of the scan, taken from the dataset's name (its last "_" field)
    std::string sample_name;
    std::string detector;   // "det3"
    // the motors of the tomo configuration: {controller name, alias}, e.g. {"mrsrot", "srot"}
    std::string rotation[2], translation_z[2], translation_y[2];
    Values technique;   // entry 1.1 technique: scan/..., detector/<det>/..., optic/..., scan_flags/..., subscans/...
    Values instrument;  // entry 1.1 instrument: det3_optic/..., mrtomo_config/..., machine/..., slits
    std::string start_time, end_time, end_reason;
    std::vector<Stretch> stretches;  // in the order of acquisition

    std::vector<int> stretches_with(Role role) const;
    const Stretch& stretch(const std::string& entry) const;
};

// `frame_files`: also open every frame file that is on disk and read what Lima wrote there (a few hundred small
// reads). Frame files that are absent are marked in the pieces (FramePiece::present), not an error.
Record read_record(const std::string& h5_path, bool frame_files = true);

// ---- what the record states per frame

// rotation angle at the middle of each exposure, cumulative [deg]: the latched angle at the start of the exposure
// plus half the arc latched during it. (The record's own "center" may be wrapped to one turn.)
inline std::vector<double> stated_angle(const Stretch& s) {
    std::vector<double> a(s.rot_trig.size());
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = s.rot_trig[i] + 0.5 * s.rot_delta[i];
    return a;
}

inline double positioner(const Stretch& s, const std::string& name, bool end = false) {
    const auto& m = end ? s.positioners_end : s.positioners_start;
    const auto it = m.find(name);
    if (it == m.end()) throw Error("stretch " + s.entry + " records no position of " + name);
    return it->second;
}

// position along the axis at the middle of each exposure [mm]: the latched encoder where the stretch has one,
// otherwise the stage's position at the start of the stretch (a stretch at rest).
inline std::vector<double> stated_height(const Record& rec, const Stretch& s) {
    if (!s.z_center.empty()) return s.z_center;
    return std::vector<double>(static_cast<std::size_t>(s.n_frames), positioner(s, rec.translation_z[1]));
}

inline double pixel_um(const Record& rec) { return rec.technique.number("scan/sample_pixel_size"); }

// the alias of a motor of the tomo configuration ("sample_u" -> "sx"); "" if the record does not name it
inline std::string tomo_motor(const Record& rec, const std::string& key) {
    const auto& v = rec.technique.texts("tomoconfig/" + key);
    return v.empty() ? std::string() : v.back();
}

// What the record states per frame of a stretch: the angle [deg, cumulative] and the position along the axis in
// detector rows, counted so that a larger value puts the sample's content on a higher row (row_sign: +1 if a larger
// axis-stage position does that, -1 if it puts it on a lower one).
struct Trajectory {
    std::vector<double> angle_deg, rows;
};
inline Trajectory trajectory(const Record& rec, const Stretch& s, double row_sign) {
    Trajectory t;
    t.angle_deg = stated_angle(s);
    const std::vector<double> z = stated_height(rec, s);
    const double px_mm = pixel_um(rec) * 1e-3;
    t.rows.resize(z.size());
    for (std::size_t i = 0; i < z.size(); ++i) t.rows[i] = row_sign * z[i] / px_mm;
    return t;
}

}  // namespace helirec::calibration
