#pragma once
// The scans of one acquisition kept open, and comparisons of blocks of their frames.
//
// A comparison cuts two blocks of frames that should hold the same lines of the sample (opposite views, views a
// turn apart, one scan against another), converts them to line integrals with each scan's own dark and mean flat
// (frames.hpp), and registers them (registration.hpp): what comes back is how far the two are from where the
// record puts them. The record checks (record_checks.hpp) and the geometry calibration
// (geometry_calibration.hpp) are both built on these.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "helirec/calibration/frames.hpp"
#include "helirec/calibration/record.hpp"

namespace helirec::calibration {

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

}  // namespace helirec::calibration
