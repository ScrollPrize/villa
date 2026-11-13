#pragma once
// Verification of what an acquisition record states.
//
// Two kinds of check, kept apart:
//  * check_record*: the record against itself (counts, clocks, raw encoder counts against the derived angles
//    and heights, commanded against recorded motion). No frame is read.
//  * check_inventory / check_series / check_views: the record against the raw frames. The frames are used as
//    stored; intensity conversion uses the scan's own dark and mean flat (frames.hpp). Registration also
//    pre-aligns the recorded per-frame row displacement before estimating residuals.
//
// A finding names the requirement it bears on by its number in
// development_notes/calibration/00_reconstruction_requirements.md.
#include <cstdint>
#include <memory>
#include <string>
#include <optional>
#include <utility>
#include <vector>

#include "helirec/calibration/frames.hpp"
#include "helirec/calibration/record.hpp"

namespace helirec::calibration {

inline constexpr double default_position_target_px = 0.01;

// Descriptive residuals, not confidence bounds. Correlated blocks and systematic errors do not average away.
struct ResidualCheck {
    std::string unit;
    double target = 0;
    std::size_t samples = 0, invalid = 0, nonzero = 0, above_target = 0;
    double mean = 0, min = 0, max = 0, rms = 0, max_abs = 0, block_sem = 0;
};

struct Finding {
    std::string requirement;  // "5.4.5"
    std::string scan;         // "HA", "TA", "HA+TA"
    std::string quantity;
    std::string stated;       // what the record states
    std::string found;        // what the check found
    // "verified": the specific record/identity check succeeded, not the whole physical requirement;
    // "fails": an observed discrepancy exceeds its target (uncertainty cannot turn it into a pass);
    // "unverified": evidence/coverage is insufficient, including sampled residuals below target;
    // "note": a value read or measured without a completed correctness check.
    std::string verdict;
    std::string note;
    std::optional<ResidualCheck> residuals;
    std::vector<std::pair<std::string, double>> numbers;  // the values behind `stated` and `found`
};

// Always retains sub-target residuals. Any finite sample outside target fails, even when the mean is zero or
// its standard error is large. Below-target samples cannot establish a noise model or unsampled coverage.
void assess_residuals(Finding& finding, const std::vector<double>& residuals, double target, const std::string& unit);

struct Report {
    double position_target_px = default_position_target_px;
    std::vector<Finding> findings;
    Finding& add(std::string requirement, std::string scan, std::string quantity);
    std::string json() const;
    std::string text() const;
};

// Diagnostic targets of section 3, derived from eps (pixels); sub-target residuals remain unaccounted for.
struct Tolerances {
    double eps = default_position_target_px;
    double R = 0;            // largest radius to reconstruct [px]
    double H = 0;            // detector rows
    double step_deg = 0;     // angle between frames
    double angle_deg() const { return eps / R * 57.29577951308232; }
    double angle_frames() const { return angle_deg() / step_deg; }
    double axis_px() const { return eps / 2; }
    double rows() const { return eps; }
    double travel_scale() const { return eps / H; }
};
// radius_px > 0: the largest distance from the axis at which a ray of the scan is used, where that is not the
// detector's width (a mosaic: a tile's rays cross the whole merged volume).
Tolerances tolerances(const Record& rec, double eps = default_position_target_px, double radius_px = 0);

// ---- the record against itself
void check_record(const Record& rec, Report& report, double eps = default_position_target_px, double radius_px = 0);
// Two scans that differ in the lateral stage of the rotation axis (axis inside the field, then moved sideways)
void check_record_pair(const Record& first, const Record& second, Report& report, double eps = default_position_target_px);
bool lateral_stage_differs(const Record& first, const Record& second);
// Two or more scans that differ in where the sample sits on the rotation stage (the tiles of a mosaic): what
// differs, the place of each (6.1.1), which lines each records (6.3.1), and angles, heights and times at equal
// projection index (6.2). radius_px 0: the lever arm is taken from the layout.
void check_record_tiles(const std::vector<Record>& scans, Report& report, double eps = default_position_target_px, double radius_px = 0);

// ---- the record against the frames

// Every frame on disk of every stretch is read once: the detector software's per-frame statistics are recomputed
// and compared (4.1), the raw value range and outliers are collected (8.1.4, 8.1.7), and the mean level of column
// strips is kept per frame for the beam checks (8.2).
//
// step: 1 reads every frame. N > 1 reads one frame in N of a stretch. 0 (the default) reads every frame, but for
// a stretch that is stored compressed and has no recorded statistics: there nothing recorded can be compared
// with a frame, decoding is what takes the time, and one frame in 16 is read. Every chunk of every frame on
// disk is located in its file whatever the step. The report says what was read.
struct Inventory {
    int strip_cols = 32;
    struct PerStretch {
        std::vector<double> sum, min, max;   // over the statistics rectangle, per frame
        std::vector<float> strips;           // (n_frames, n_strips): mean raw value of each column strip in the rectangle
        std::vector<std::uint16_t> frame_max;  // over the whole frame
        std::vector<std::int64_t> histogram;   // of every raw value of the stretch (65536 bins)
        // a frame that was not read (not on disk, or skipped by the step) holds zeros everywhere
        bool was_read(std::size_t frame) const { return frame_max[frame] != 0 || sum[frame] != 0; }
    };
    std::vector<PerStretch> stretches;
};
inline constexpr int default_inventory_sparse_step = 16;
Inventory check_inventory(const Record& rec, Frames& frames, Report& report, int step = 0);
void save_inventory(const Inventory& inv, const Record& rec, const std::string& path);  // <path>.bin + <path>.json
Inventory load_inventory(const Record& rec, int cols, const std::string& path);         // what save_inventory wrote

// Darks and flats: their roles, levels and noise (4.2, 8.1.1, 8.1.2 with a second scan, 8.2.5)
void check_series(const Record& rec, Frames& frames, const FlatField& ff, Report& report,
                  const Record* other = nullptr, Frames* other_frames = nullptr, const FlatField* other_ff = nullptr);
// The dark of a second scan against the first's, per pixel (8.1.2): what check_series does when given `other`.
void check_dark_pair(const Record& rec, const FlatField& ff, const Record& other, const FlatField& other_ff, Report& report);

// The beam against the record: air level per frame from the inventory's strips (4.7, 8.2.1 - 8.2.5)
void check_beam(const Record& rec, const Inventory& inv, const FlatField& ff, Report& report);

struct ViewOptions {
    int frames = 32;        // frames per block
    int rows = 256;         // rows per block (fewer where the overlap is smaller)
    int blocks = 8;         // blocks along each stretch
    int columns = 0;        // diagnostic subwindow width; 0 keeps the full overlap
    double column_fraction = 0.5;  // subwindow position in the overlap: 0 left, 1 right in A
    double eps = default_position_target_px;  // diagnostic target, never permission to discard smaller residuals
    std::string table;      // if set: every block's registration is written there, one tab-separated line each
    Device device = Device::CPU;  // where the blocks are registered (registration.hpp)
};
// Views that record the same lines, registered against each other (registration.hpp): opposite views of a turn
// at rest, a turn at rest against the helix, the helix a turn apart, and one scan against the other.
void check_views(const Record& first, Frames& first_frames, const FlatField& first_ff, const Record* second,
                 Frames* second_frames, const FlatField* second_ff, const ViewOptions& opt, Report& report);

// ---- comparisons placed by the caller

// One pair of blocks that should hold the same lines: n frames of stretch A from frame_a on, against the frames
// of stretch B at A's angle plus angle_offset_deg (the record's angles decide which; frame_hint says in which
// turn of B, as B's frame index minus A's to within a few thousand frames). The blocks are cut, converted and
// registered exactly as the view checks do it; nothing but the placement differs.
struct ViewRequest {
    std::string label;
    int scan_a = 0, scan_b = 0;        // index into the session's scans
    int stretch_a = 0, stretch_b = 0;  // index into the scan's stretches
    std::int64_t frame_a = 0, frame_hint = 0;
    double angle_offset_deg = 0;       // 180 with mirror for opposite views
    bool mirror = false;               // B's columns reversed
    int frames = 32, rows = 256, cols = 512;  // wanted block size; reduced to the next quick transform size
    int row_a = -1;                    // first row of A's block; -1: where A's and B's blocks lie centred
    int col_a = 0, col_b = 0;          // first column of A's block, and of B's block before reversal
    // B's frames read at an angle that changes with the block's column, in frames per column (frames.hpp,
    // shear_frames): what the beam's divergence does to opposite views. 0: as stored.
    double frame_shear = 0;
    // B's content displaced along the columns by a course of B's angle: frame q of B is read
    // sway_px cos(angle_b(q) - sway_phase_deg) columns further on (in B's own columns, before any reversal).
    // What the sample's place on the rotation stage does between two tiles of a mosaic: tiles k and j record the
    // same lines at equal angle on columns n(angle).(d_j - d_k) apart. The measured du is what is left of it.
    double sway_px = 0, sway_phase_deg = 0;
    // The angle at which a row is exposed, in degrees per row from the frame's (the rows' timing times the
    // angular speed): row v of B is then read at the sway of angle_b + sway_row_deg (v - middle row).
    double sway_row_deg = 0;
};
// What view.tsv holds per registration (same meaning), and the plan behind it. planned: the record holds such a
// pair and the blocks fit the detector; ok: the registration converged inside its search window.
struct ViewMeasurement {
    std::string label;
    bool planned = false, ok = false;
    std::int64_t frame_a = 0, frame_b = 0;
    int frames = 0, row_a = 0, rows = 0, row_offset = 0, col_a = 0, col_b = 0, cols = 0;
    double stated_dq = 0, dq = 0, stated_dv = 0, dv = 0, stated_D = 0, D = 0, du = 0;
    double correlation = 0;
    int evaluations = 0;
    double slope_a = 0, slope_b = 0, step_deg = 0, angle_a_deg = 0;
};

// One patch of a request's block (Session::measure_patches).
struct PatchMeasurement {
    int request = 0;     // index into the requests
    int i = -1, j = -1;  // the patch's place in the block: column and row index (-1: the request could not be planned)
    ViewMeasurement m;   // what measure() returns for a request of the patch's size at the patch's place
    // Second derivatives of the correlation at the registration with respect to the displacement: frames^2, frames
    // rows, frames columns, rows^2, rows columns, columns^2 (registration.hpp). How sharply each of the three
    // shifts is determined and how they are tied; minus its inverse, scaled by the noise, is their covariance.
    double curvature[6] = {0, 0, 0, 0, 0, 0};
};

// Where the time of measure() and measure_patches() went since the last reset [s]: reading the frames, converting
// (and resampling) them, registering; and how many blocks were read and registrations run.
struct MeasureTiming {
    double read = 0, integrals = 0, registration = 0;
    int blocks = 0, registrations = 0;
};
MeasureTiming measure_timing(bool reset = false);

// The scans of one acquisition kept open: records, raw frames, and each scan's dark and mean flat (made when
// first needed: every dark and flat frame is read once).
class Session {
  public:
    explicit Session(const std::vector<std::string>& scans, int threads = 0);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    std::size_t size() const;
    const Record& record(int scan) const;
    Frames& frames(int scan);
    const FlatField& flat_field(int scan);
    // A detector map to be tested (frames.hpp): from then on measure() cuts the blocks on the map's ideal grid.
    // An empty map (the default) cuts them from the frames as stored. The view checks never use one.
    // scan < 0: the same map for every scan (a property of the detector); otherwise for that scan alone (what
    // differs between two scans, tested on one of them).
    void set_detector_map(const DetectorMap& map, int scan = -1);
    const DetectorMap& detector_map(int scan = 0) const;
    // row_sign: +1 if a larger axis-stage position puts the sample on higher rows, -1 if on lower (the view
    // checks determine it: finding "sense of the detector's rows against the axis stage").
    // The frames of the comparisons that follow the one in hand are decoded ahead by workers of their own where
    // the frames are stored compressed (set_read_ahead: their number per scan, 0 for none; the default is a third
    // of the machine's threads).
    std::vector<ViewMeasurement> measure(const std::vector<ViewRequest>& requests, double row_sign, Device device = Device::CPU);
    // Each request's block cut into patches of patch_rows x patch_cols (rows / patch_rows by cols / patch_cols of
    // them, at least one each way; with row_a -1 they are placed together), every patch measured as measure()
    // measures a request of that size at that place: the same cut, registration and re-cuts, and the same numbers
    // to the rounding of the transforms. The frames of a block are read once for all its patches, and the patches
    // are registered side by side, one per thread and each on one thread (a few at a time on a CUDA device), so
    // that the result does not depend on the number of threads. A mirrored B's patch k lies at
    // col_b + (cols - patch_cols) - k patch_cols. A request that cannot be planned gives one entry with i = j = -1.
    // from_block: each request's block is registered whole first, from the same readings, and its patches start
    // where it ends rather than at the plan (what a patch needs where the plan is some samples off: the mosaic).
    std::vector<PatchMeasurement> measure_patches(const std::vector<ViewRequest>& requests, int patch_rows, int patch_cols, double row_sign, Device device = Device::CPU,
                                                  bool from_block = false);
    void set_read_ahead(int threads);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- everything, in order

struct VerifyOptions {
    // comma-separated subset of record, inventory, series, beam, views. beam needs the inventory: made by
    // "inventory" (every frame is read) or taken from inventory_dir.
    std::string checks = "record,inventory,series,beam,views";
    std::string out_dir;        // if set: report.json, report.txt, views.tsv and inventory_<scan>.bin are written there
    std::string inventory_dir;  // if set: the inventory is read from there (as an earlier run wrote it) instead of made
    int threads = 0;            // reading threads (0: automatic)
    ViewOptions view;
    bool print = false;         // print the findings to stdout as each group of checks finishes
    double radius_px = 0;       // lever arm of the angle targets; 0: the detector's width, or the layout of the tiles
    int inventory_step = 0;     // see Inventory: 0 automatic, 1 every frame, N one frame in N
};
// scans: the master file of one scan; of two scans of one acquisition that differ in the lateral stage of the
// rotation axis, the scan with the axis inside the detector's field first; or of any number of scans that differ
// in where the sample sits on the rotation stage (the tiles of a mosaic), the reference tile first. The view
// checks cover the first two cases; the comparisons of a mosaic are placed by the caller (Session::measure).
Report verify(const std::vector<std::string>& scans, const VerifyOptions& opt);

}  // namespace helirec::calibration
