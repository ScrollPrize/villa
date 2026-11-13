#pragma once
// The geometry calibration step of a two-scan helical acquisition (helirec-calibration geometry): the detector's
// map and the poses of both scans, measured from the acquisition itself with nothing typed in.
//
// calibrate_geometry runs the step; its stages are declared below it and can be called on their own: the layout,
// the plan of comparisons, a pass of them, the fit, the rows' timing from a refill, and the motion within a turn.
// What the step writes is read back for the reconstruction by helirec/geometry.hpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "helirec/calibration/session.hpp"
#include "helirec/calibration/solve.hpp"
#include "helirec/geometry.hpp"

namespace helirec::calibration::geometry_calibration {

using Log = std::function<void(const std::string&)>;

struct Options {
    double row_time = std::nan("");  // seconds per stored row (signed) if known for this camera configuration; NaN: measured from a refill
    Device device = Device::CPU;
    int order = 3;            // of the map's polynomial
    int passes = 6;
    double tolerance = 0.02;  // a pass that corrects the map and the second scan's lines by less ends the passes [px]
    int threads = 8;          // reading threads
    bool motion = true;       // the motion within a turn (the poses of every frame need it)
    int patch = 0;            // 0: the plan's blocks registered whole; N: cut into patches of N x N
    Log log;                  // progress, a line at a time (none: silent)
};
// scans: the two master files, the scan with the rotation axis inside the detector's field first. Returns the
// calibration as JSON text (what helirec/geometry.hpp reads); it is an error if the rows' timing is not given and
// no stretch holds a refill to measure it from.
std::string calibrate_geometry(const std::vector<std::string>& scans, const Options& options = {});
// The text written whole or not at all (a reconstruction may be reading the file).
void save_text(const std::string& text, const std::string& path);

// ---------------------------------------------------------------- the stages

// the stretches of a scan, in the order of acquisition: a turn at rest, the helix, a turn at rest
inline constexpr const char* kNames[3] = {"rest1", "helix", "rest2"};
enum { Rest1 = 0, Helix = 1, Rest2 = 2 };
inline int name_index(const std::string& name) {
    for (int a = 0; a < 3; ++a)
        if (name == kNames[a]) return a;
    throw Error("geometry calibration: no stretch named " + name);
}
// the kinds of comparison of a pass, in the order they are reported
inline constexpr const char* kKinds[4] = {"opposite", "resthelix", "turn", "pair"};

// rows that hold something across the whole width (a holder, the end of the sample): (frames sampled, mask)
struct Blocked {
    std::vector<std::int64_t> frames;
    std::vector<std::uint8_t> mask;  // (frames, rows)
};

struct Layout {
    int rows = 0, cols = 0, per_turn = 0;
    double pixel_um = 0;
    int stretch[2][3] = {{0, 0, 0}, {0, 0, 0}};                // per scan: the record's stretch index of rest1, helix, rest2
    std::int64_t length[2][3] = {{0, 0, 0}, {0, 0, 0}};        // frames
    double row_sign = -1;
    double axis = 0;   // coarse mirror column of the first scan
    int move = 0;      // coarse: the second scan's content lies this many columns higher
    int width = 1024;  // columns of a window
    std::map<std::pair<int, int>, Blocked> blocked;  // (scan, stretch) -> rows kept out
    std::vector<double> heights[2][3];               // the record's height of every frame in rows: row_sign x height / pixel

    bool is_blocked(int scan, int name, double frame, int row_lo, int row_hi) const;
};
// the middle column, in the first scan, of what both scans record (as the passes always took it)
inline double overlap_centre(const Layout& lay) { return (lay.cols - lay.move) / 2.0; }

// one comparison of the plan, and what it compares
struct Planned {
    ViewRequest request;
    std::string kind;
    int scan_a = 0, scan_b = 0, a = 0, b = 0;  // a, b: the stretches (Rest1, Helix, Rest2)
};
// an accepted comparison
struct Row {
    std::string kind;
    int scan_a = 0, scan_b = 0, a = 0, b = 0;
    ViewMeasurement m;
    // the motion stage's quantities (measure_motion)
    double q = 0, y = 0, t1 = 0, t2 = 0, th = 0, vA = 0, X = 0, Pa = 0, Pb = 0;
};

Layout read_layout(Session& session, Device device, const Log& log);
std::vector<Planned> plan(const Layout& lay, int n_opp = 8, int n_rest = 16, int n_turn = 12, int n_pair = 10);
// One pass: every comparison of the plan registered on the grid of map `m` (the second scan with its line of the
// stretch added). patch: 0 registers each block as one; N cuts it into patches of N x N.
std::vector<Row> measure(Session& session, const Layout& lay, const geometry::Map& m, const double lines[3][2], Device device, int patch);

struct GroupKey {
    std::string kind;
    int scan = 0, a = 0;
    std::int64_t frame = 0;
    bool operator<(const GroupKey& o) const {  // as Python orders (kind, scan, stretch name, frame)
        if (kind != o.kind) return kind < o.kind;
        if (scan != o.scan) return scan < o.scan;
        const std::string na = kNames[a], nb = kNames[o.a];
        if (na != nb) return na < nb;
        return frame < o.frame;
    }
};
struct Fit {
    std::vector<double> eu, ev;        // the map's correction, term by term
    double psi = 0;                    // the roll of the first scan's axis
    std::map<int, double> rho;         // per stretch with pairs: the second scan's rotation about the beam
    std::map<int, std::array<double, 2>> lines;  // per stretch with pairs: the second scan's line in the column
    std::size_t n = 0;
    struct Left {
        std::size_t n = 0;
        double px = 0, rows = 0;
    };
    std::map<std::string, Left> left;  // per kind: what the fit leaves [rms]
    std::map<GroupKey, std::array<double, 2>> groups;  // per frame position: (u0, v0)
    FitSummary solver;                 // how the solve ended (solve.hpp)
    int rank = 0, active = 0, unused = 0;
};
// What the comparisons leave, as a correction of the map `m` (same terms), the poses beside it, and the residuals
// (python/helirec/geometry.py states the model).
// It is an error if the solve is not usable (a model whose active unknowns depend on each other, no support).
Fit fit(const std::vector<Row>& rows, const Layout& lay, const geometry::Map& m, double tau, double period, double huber = 0.03, int max_iterations = 100);
// the largest displacement the correction (eu, ev) makes anywhere on the detector [px, rows]
std::array<double, 2> update_range(const geometry::Map& m, const std::vector<double>& eu, const std::vector<double>& ev);

// b of X b = y by least squares, through the steps' solver. X: (n, k) row-major. columns: the unknowns fitted (the
// others stay zero; empty: all); rows: the equations used (empty: all). Columns no equation uses are held at
// zero; it is an error if the fitted columns depend on each other or there is no equation.
std::vector<double> least_squares(const std::vector<double>& X, std::size_t n, std::size_t k, const std::vector<double>& y, const std::vector<int>& columns = {},
                                  const std::vector<std::uint8_t>& rows = {});
// Design of [d(t2) - d(t1)] . (cos, sin)(th), d at the nodes h, 2h, .. (d(0) = 0), a line between; then n_extra
// empty columns. Returns (n, 2 K + n_extra) row-major and K.
std::vector<double> nodes_design(const std::vector<double>& t1, const std::vector<double>& t2, const std::vector<double>& th, int n_extra, double h, double total,
                                 int& K);
// The largest block size up to `limit` the registration takes as it is.
int block_size(int limit);

// The second scan against the first at equal frames: per row of the table the stretch, turns from the start, the
// angle [rad], the middle row, columns, rows, the angle in frames.
struct PairRow {
    int a = 0;
    double t = 0, th = 0, v = 0, du = 0, y = 0, q = 0;
};
// Returns what it found as JSON text (the step's "second_scan").
std::string second_scan_fit(const std::vector<PairRow>& table, int rows, double total);
// What the sample and the rotating body do within a turn, on the grid of the map.
// Returns the step's "motion" as JSON text.
std::string measure_motion(Session& session, const Layout& lay, const geometry::Map& m, const double lines[3][2], double tau, double period, Device device, const Log& log);

// Seconds per stored row from a refill in one of the stretches, as the step saves it (JSON text; "null" if no
// stretch holds one): the frames around the refill are found and reduced to the rows' levels here, the dips are
// fitted by calibration::fit_refill_shots (helirec/calibration/refill.hpp).
std::string row_time_from_refill(Session& session, const Layout& lay, const Log& log);

// ---------------------------------------------------------------- the joint model
//
// One model for any number of scans of one sample (a scan alone, a pair moved sideways, the tiles of a mosaic),
// solved as one Ceres problem. A ray through the sample is named by the angle th it is seen at, its distance s
// from the sample's origin across the beam [px] and its height z [rows]. Scan k records it at the ideal detector
// position
//     X = axis_k + axis_slope_k (H - H_mid_k) + u + roll (Y - middle row),        u = s + d_k . (cos th, sin th)
//     Y = z + height_k + stretch height + (1 + axial_scale_k) H - roll u
// with th and H the record's angle and height [rows] at the time the row was exposed: row v of frame f at the time
// of frame f + row_time (v - middle row). The unknowns are the places, nothing else: no comparison has an offset
// of its own, so what the model does not explain stays in the residuals.
//
// A comparison is two sightings of the same lines. Each is taken back to the ray it records, and the residual is
// the difference of the two rays in the sample: s_b - s_a (s_b + s_a where b sees the rays from the other side)
// and z_b - z_a. The frames' displacement is reported as the difference of the two angles and is not fitted. What
// an error of it does to the other two differs:
//   rows     nothing: z stays the same along the track a piece of content follows through the frames (the helix'
//            travel is in H), so a frame's error costs axial_scale x the travel per frame.
//   columns  content at the depth t along the beam (from the rotation axis) crosses t x the angle per frame columns
//            per frame, so an error of e frames in the registration is a column residual of e x the angle per frame
//            x the mean depth of the block's detail. Nothing for detail centred on the axis; for a scan whose axis
//            lies c from the middle of the sample's detail the mean depth is about -c . (-sin th, cos th): a
//            sinusoid of the angle. A column residual of that form is first of all a frames' error.
// (The two-scan step above is being replaced by this; it stays until the new one has reproduced it.)

// The projection stretches of one scan and what its record states of them.
struct StretchLayout {
    std::string name;       // "helix", "rest1", "rest2", ..: rest turns are numbered in the order of acquisition
    int index = 0;          // the record's stretch
    bool helix = false;
    Trajectory course;      // per frame: the angle [deg, cumulative] and the height [rows] (record.hpp)
};
struct ScanLayout {
    std::string name;
    std::vector<StretchLayout> stretches;  // in the order of acquisition
    double mid_height = 0;  // H_mid: the height [rows] the axis column is counted from (the middle of the helix' frames on disk)
};
struct Acquisition {
    int rows = 0, cols = 0;
    double row_time = 0;    // frames per stored row, signed
    std::vector<ScanLayout> scans;
};

// The model's unknowns. The names the solver knows them by: "roll", "scan<k>.axis", "scan<k>.axis_slope",
// "scan<k>.axial_scale", "scan<k>.d0", "scan<k>.d1", "scan<k>.height", "scan<k>.<stretch name>.height".
struct ScanParameters {
    double axis = 0;         // the axis column at the middle row, at the height H_mid
    double axis_slope = 0;   // columns the axis moves per row of the axis stage's travel
    double axial_scale = 0;  // the sample travels (1 + axial_scale) rows per row of the record
    double d[2] = {0, 0};    // where the sample's origin sits on the rotation stage [px]
    double height = 0;       // rows
    std::vector<double> stretch_height;  // per stretch: rows beyond the scan's height
};
struct Parameters {
    double roll = 0;         // of the rotation axis against the detector's columns [rad]
    std::vector<ScanParameters> scans;
};
Parameters empty_parameters(const Acquisition& acq);

// A piece of the sample as one scan holds it: the stored pixel (row, col) of a frame, fractions allowed.
struct Sighting {
    int scan = 0, stretch = 0;  // stretch: index into ScanLayout::stretches
    double frame = 0, row = 0, col = 0;
};
struct Comparison {
    std::string kind;       // "opposite", "turn", "resthelix", "pair": what the report groups by
    Sighting a, b;
    bool opposite = false;  // b sees a's rays from the other side, half a turn on
    double correlation = 0;
    double curvature[6] = {0, 0, 0, 0, 0, 0};  // of the correlation's peak (PatchMeasurement)
};
// The middle of a registered block or patch as two sightings: where A holds it, and where the registration found
// it in B. stretch_a, stretch_b: indices into the scans' layouts. request: the one the measurement answered (its
// sway is in where B's sighting lies).
Comparison comparison_of(const Acquisition& acq, const std::string& kind, int scan_a, int stretch_a, int scan_b, int stretch_b, const ViewRequest& request,
                         const ViewMeasurement& m);

// The ray a sighting records under the parameters: the angle [deg], s [px], z [rows].
struct Ray {
    double angle_deg = 0, s = 0, z = 0;
};
Ray ray_of(const Acquisition& acq, const Parameters& p, const Sighting& g);
// What the parameters leave of each comparison: frames (the two rays' angles, less the half turn of opposite
// views, in B's frames), rows (z_b - z_a) and columns (s_b - s_a, or s_b + s_a for opposite views).
std::vector<std::array<double, 3>> residuals(const Acquisition& acq, const Parameters& p, const std::vector<Comparison>& comparisons);

struct GeometryOptions {
    double huber = 0.03;       // on the length of a comparison's residual (columns, rows) [px]
    int max_iterations = 100;
    // Free unless held: every scan's axis, axis_slope and axial_scale, the roll, and the height of every stretch
    // that is not a helix. Held unless freed: every scan's d and height, the helix' height. A stage of the
    // calibration is these two lists.
    std::vector<std::string> held, freed;
};
struct GeometrySolution {
    Parameters parameters;
    FitSummary solver;
    std::vector<std::string> free;    // the unknowns that were solved for
    std::vector<std::string> unused;  // asked free, but no comparison depends on them: held where they started
    int rank = 0;
    std::vector<std::array<double, 3>> residual;  // per comparison: frames, rows, columns
    // what the solution leaves, per kind and pair of scans (scan_a = scan_b = -1: the kind over all scans)
    struct Left {
        std::string kind;
        int scan_a = -1, scan_b = -1;
        std::size_t n = 0;
        double rms[3] = {0, 0, 0}, median[3] = {0, 0, 0};  // frames, rows, columns
    };
    std::vector<Left> left;
};
// The places that best explain the comparisons, from `start`. It is an error if the free unknowns depend on each
// other (the message names a dependent set) or the solve is not usable.
GeometrySolution solve_geometry(const Acquisition& acq, const std::vector<Comparison>& comparisons, const Parameters& start, const GeometryOptions& options = {});
// As JSON text: the parameters by name, how the solve ended, what is left.
std::string solution_json(const Acquisition& acq, const GeometrySolution& s);

// ---- small numerics the stages share (as numpy does them)

// num evenly spaced values from a to b inclusive
inline std::vector<double> linspace(double a, double b, int num) {
    std::vector<double> y(static_cast<std::size_t>(std::max(num, 0)));
    if (num == 1) y[0] = a;
    if (num < 2) return y;
    const double step = (b - a) / (num - 1);
    for (int i = 0; i < num; ++i) y[static_cast<std::size_t>(i)] = i * step + a;
    y.back() = b;
    return y;
}
inline double median_of(std::vector<double> v) {
    if (v.empty()) throw Error("geometry calibration: the median of nothing");
    const std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m), v.end());
    if (v.size() % 2) return v[m];
    return (*std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(m)) + v[m]) / 2.0;
}
inline double mean_of(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}
inline double rms_of(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x * x;
    return v.empty() ? 0.0 : std::sqrt(s / static_cast<double>(v.size()));
}
// population standard deviation
inline double sd_of(const std::vector<double>& v) {
    const double m = mean_of(v);
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return v.empty() ? 0.0 : std::sqrt(s / static_cast<double>(v.size()));
}
}  // namespace helirec::calibration::geometry_calibration
