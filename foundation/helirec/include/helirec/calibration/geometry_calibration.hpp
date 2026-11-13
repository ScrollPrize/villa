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
};
// What the comparisons leave, as a correction of the map `m` (same terms), the poses beside it, and the residuals
// (python/helirec/geometry.py states the model).
Fit fit(const std::vector<Row>& rows, const Layout& lay, const geometry::Map& m, double tau, double period, double huber = 0.03, int iterations = 8);
// the largest displacement the correction (eu, ev) makes anywhere on the detector [px, rows]
std::array<double, 2> update_range(const geometry::Map& m, const std::vector<double>& eu, const std::vector<double>& ev);

// b of X b = y by least squares, through the steps' solver. X: (n, k) row-major. columns: the unknowns fitted (the
// others stay zero; empty: all); rows: the equations used (empty: all). An unknown the equations do not determine
// comes out zero.
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

// The time of a row from the beam dips of a refill. x: (frames, n_rows), the logarithm of each row's level in
// frames around the refill; T, E: the frame period and the gate [s]; pre: the frames at the start that lie before
// the refill. Nothing if fewer than 12 shots are found.
struct RefillFit {
    std::vector<double> p;  // the row time [s per row, signed], the dips' decay, the first shot's time, the raster, ..
    double se = 0, noise = 0, cost = 0;
    std::vector<int> shots;
};
std::optional<RefillFit> fit_refill_shots(const std::vector<double>& x, std::size_t frames, int n_rows, double T, double E, int pre);
// Seconds per stored row from a refill in one of the stretches, as the step saves it (JSON text; "null" if no
// stretch holds one).
std::string row_time_from_refill(Session& session, const Layout& lay, const Log& log);

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
