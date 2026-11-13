#pragma once
// The geometry calibration step (helirec-calibration geometry): the detector's map and the places of any number of
// scans of one sample (a scan alone, a pair moved sideways, the tiles of a mosaic), measured from the acquisition
// itself with nothing typed in, on one model solved as one Ceres problem.
//
// calibrate_geometry runs the step; its stages are declared below it and can be called on their own: the layout,
// the rows' timing from a refill, the plan of comparisons, their measurement, and the solve.
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
    // seconds per stored row (signed) if known for this camera configuration; 0: not known, and done without;
    // NaN: measured from a refill
    double row_time = std::nan("");
    Device device = Device::CPU;
    int order = 3;    // of the map's polynomial; 0: no map
    int threads = 8;  // reading threads
    int patch = 128;  // the plan's blocks are cut into patches of patch x patch, each registered once
    Log log;          // progress, a line at a time (none: silent)
};
// scans: the master files of the scans of one sample, a scan with the rotation axis inside its field first: the
// layout, the rows' timing, every comparison of the plan measured once, one solve. Returns the calibration as
// JSON text ("model": "joint"; helirec/geometry.hpp reads a pair's); it is an error if the rows' timing is not
// given and no stretch on disk holds a refill to measure it from.
std::string calibrate_geometry(const std::vector<std::string>& scans, const Options& options = {});
// The text written whole or not at all (a reconstruction may be reading the file).
void save_text(const std::string& text, const std::string& path);

// rows that hold something across the whole width (a holder, the end of the sample): (frames sampled, mask)
struct Blocked {
    std::vector<std::int64_t> frames;
    std::vector<std::uint8_t> mask;  // (frames, rows)
    // whether rows [row_lo, row_hi) hold such a thing in the sampled frame nearest `frame`
    bool holds(double frame, int row_lo, int row_hi) const;
};

// one comparison of the plan, and what it compares
struct Planned {
    ViewRequest request;
    std::string kind;
    int scan_a = 0, scan_b = 0, a = 0, b = 0;  // a, b: the stretches, as indices into the scans' stretches
    bool from_block = false;  // its patches start from the block registered whole (Session::measure_patches)
};

// ---------------------------------------------------------------- the model
//
// One model for any number of scans of one sample (a scan alone, a pair moved sideways, the tiles of a mosaic),
// solved as one Ceres problem. A ray through the sample is named by the angle th it is seen at, its distance s
// from the sample's origin across the beam [px] and its height z [rows]. The stored pixel (c, v) looks along the
// ideal position (X, Y) = (c + Eu(c, v), v + Ev(c, v)), Eu and Ev the detector's map (helirec/geometry.hpp; the
// same for every scan on the detector, none unless Parameters::map has terms). Scan k records the ray at the ideal
// position
//     X = axis_k + axis_slope_k (H - H_mid_k) + A(2 th) + u + roll (Y - middle row),        u = s + d_k . (cos th, sin th)
//     Y = z + height_k + stretch height + (1 + axial_scale_k) H + B(th) - roll u
// with th and H the record's angle and height [rows] at the time the row was exposed: row v of frame f at the time
// of frame f + row_time (v - middle row). A and B are what the rotation stage does within a turn, the same for
// every scan on it: B(th) = b_c cos th + b_s sin th lifts the sample once per turn, A(2 th) = a_c cos 2 th +
// a_s sin 2 th moves the axis across the detector twice per turn. Those are the parts opposite views see (the
// height at odd multiples of the turn, the axis column at even ones: PHerc0343P 0.03 rows and 0.02 px, Paris4
// 0.25 rows and 0.25 px); the axis column once per turn is the sample's place d. The unknowns are the places, nothing else: no comparison has an offset
// of its own, so what the model does not explain stays in the residuals.
// One term is not of a ray. A rotation axis pitched towards the beam makes every ray climb by the pitch per pixel
// of depth, so views half a turn apart are not the same rays: their rows differ by twice the pitch x the depth of
// what they register, and for a tile whose axis lies c from the middle of the sample's detail that depth is about
// -c . (-sin th, cos th): a sinusoid of the angle that follows the tile (Paris4: 0.5 to 0.6 rows between the views
// of the six tiles around the middle one). The model has no pitch (the reconstruction takes no ray's
// inclination), and which depth a registration weights is not known, so each scan has that sinusoid as an
// unknown of its own views, ScanParameters::height_once: B_k(th), added to the stage's B(th) in the scan's
// comparisons with itself and in nothing else (not in Y above, not between two scans). Left out, it goes into
// the scan's axial scale wherever the views do not cover whole turns. The anchor's is the stage's B.
// The beam's divergence is in the reported angle alone (Ray::angle_deg, the frames' residual): the ray recorded at
// column c is a parallel beam's at th + fan (c - middle column) frames (Acquisition::fan; the sign as PHerc0343P's
// opposite views show it: their frames lie 2 fan (c - axis) further on than half a turn). No row or column of
// the model takes it: s and z above are formed with the stage's angle.
//
// A comparison is two sightings of the same lines, in one scan or in two (one scan against another at equal angle:
// "pair"). Each is taken back to the ray it records, and the residual is the difference of the two rays in the
// sample: s_b - s_a (s_b + s_a where b sees the rays from the other side) and z_b - z_a. A scan whose rotation
// axis lies outside its field (the second scan of a pair moved sideways) is known through its comparisons with
// another alone: its axis column, the slope of it and its height against the anchor's. The frames' displacement is reported as the difference of the two angles and is not fitted. What
// an error of it does to the other two differs:
//   rows     nothing: z stays the same along the track a piece of content follows through the frames (the helix'
//            travel is in H), so a frame's error costs axial_scale x the travel per frame.
//   columns  content at the depth t along the beam (from the rotation axis) crosses t x the angle per frame columns
//            per frame, so an error of e frames in the registration is a column residual of e x the angle per frame
//            x the mean depth of the block's detail. Nothing for detail centred on the axis; for a scan whose axis
//            lies c from the middle of the sample's detail the mean depth is about -c . (-sin th, cos th): a
//            sinusoid of the angle. A column residual of that form is first of all a frames' error.

// The projection stretches of one scan and what its record states of them.
struct StretchLayout {
    std::string name;       // "helix", "rest1", "rest2", ..: rest turns are numbered in the order of acquisition
    int index = 0;          // the record's stretch
    bool helix = false;
    Trajectory course;      // per frame of the stretch: the angle [deg, cumulative] and the height [rows] (record.hpp)
    std::int64_t first = 0, end = 0;  // the frames on disk: [first, end); nothing is planned outside them
    std::int64_t offset = 0;  // the stretch's first frame among the scan's projection frames, all its projection stretches in the record's order
    Blocked blocked;        // rows that hold something across the whole width, kept out of every comparison
};
struct ScanLayout {
    std::string name;
    std::vector<StretchLayout> stretches;  // the projection stretches that are on disk, in the order of acquisition
    double mid_height = 0;  // H_mid: the height [rows] the axis column is counted from (the middle of the helix' frames on disk)
    // The axis column, coarsely: from profiles of frames half a turn apart where the axis lies in the field;
    // otherwise from where the scan's content lies against that of a scan with the same place on the rotation
    // stage (profiles at equal frames), and then it may lie outside the detector. NaN: not known.
    double axis = std::nan("");
    // What the record's motors say [px], at the start of the scan's first projection stretch: where the sample sits
    // on the rotation stage (the two stages on top of the rotation) and the rotation stage's place across the
    // beam. NaN: the record does not name them.
    double on_stage[2] = {std::nan(""), std::nan("")};
    double across_beam = std::nan("");
    // The sample sits elsewhere on the rotation stage than in the anchor (by more than a pixel): the tiles of a
    // mosaic. Scans that were not moved are compared at equal angle across the columns both hold ("pair"). A
    // scan's place d is an unknown of the solve either way: PHerc0343P's second scan, with the stages on the
    // rotation where they were, holds the sample 0.13 px from where the first does.
    bool moved_on_stage = false;
};
struct Acquisition {
    int rows = 0, cols = 0, per_turn = 0;
    double pixel_um = 0;
    double row_sign = -1;   // +1 if a larger axis-stage position puts the sample on higher rows
    double row_time = 0;    // frames per stored row, signed
    double period_s = 0;    // of a frame (the median of the first scan's helix, or of its first stretch); 0: not known
    double fan = 0;         // the beam's divergence: frames of angle per column (pixel / source distance / angle per frame)
    int width = 512;        // columns of a window, the same for every kind of comparison
    // The scan whose place (d, height) is the origin: zero, and held. Of scans that share their place on the
    // rotation stage the first with the axis inside its field; of a mosaic's tiles the one nearest their middle.
    int anchor = 0;
    std::vector<ScanLayout> scans;
};
// whether a scan's opposite views share a strip wide enough for a window (its rotation axis lies inside the field)
bool axis_in_field(const ScanLayout& scan, int cols);
// The layout of any number of scans, each with whatever projection stretches it has on disk (a helix alone, or
// rest turns beside it): frames per turn, the record's angle and height of every frame, the sense of the rows
// against the axis stage (helix views a turn apart where they overlap by a patch, opposite views otherwise; the
// same in every scan or it is an error), the rows that hold the mount, a coarse axis column per scan, the motors'
// places and the anchor. The rows' timing is left at zero: the caller sets it.
Acquisition read_acquisition(Session& session, Device device, const Log& log = {});
// Seconds per stored row from a refill in one of the acquisition's stretches, as the step saves it (JSON text;
// "null" if no stretch on disk holds one): the frames around the refill are found and reduced to the rows' levels
// here, the dips are fitted by calibration::fit_refill_shots (helirec/calibration/refill.hpp).
std::string row_time_from_refill(Session& session, const Acquisition& acq, const Log& log);
// The layout without the per-frame series, as JSON text (for logs and files).
std::string acquisition_json(const Acquisition& acq);

// The comparisons of the scans, placed from the layout alone. Each scan with itself:
//   opposite    a stretch against itself half a turn on, mirrored about the coarse axis column; in a helix the rows
//               offset by the travel of half a turn, at angles_per_turn angles evenly around the turn and at most
//               max_positions positions along the helix; the patches started from the block registered whole
//   turn        the helix against itself a turn on, where a turn's travel leaves a patch of rows in common
//   resthelix   a rest turn against the helix at equal angle, in the turn of the helix nearest its height
// and one scan against another with the same place on the rotation stage (a pair moved sideways):
//   pair        the stretches of the same name at equal angle, the second scan's window where the coarse axis
//               columns put the first's content, across the columns both hold; pair_positions positions along a
//               helix (as the opposite views': evenly around the turn), n_pair_rest in a rest turn. Not planned
//               where neither the columns nor the rows of the two blocks lie 48 px apart (a pattern fixed to the
//               detector would register).
struct PlanCounts {
    int angles_per_turn = 12, max_positions = 48, n_rest = 16, n_turn = 12, pair_positions = 24, n_pair_rest = 12;
};
std::vector<Planned> plan(const Acquisition& acq, const PlanCounts& counts = {});

// The model's unknowns. The names the solver knows them by: "roll", "stage.height_once_cos", "stage.height_once_sin",
// "stage.axis_twice_cos", "stage.axis_twice_sin", "scan<k>.axis", "scan<k>.axis_slope", "scan<k>.axial_scale",
// "scan<k>.d0", "scan<k>.d1", "scan<k>.height", "scan<k>.height_once_cos", "scan<k>.height_once_sin",
// "scan<k>.<stretch name>.height", and per term of the map "map.eu.i<i>j<j>", "map.ev.i<i>j<j>". A name's
// beginning up to a dot names all that follow it: "stage", "map", "map.ev", "scan1".
struct ScanParameters {
    double axis = 0;         // the axis column at the middle row, at the height H_mid
    double axis_slope = 0;   // columns the axis moves per row of the axis stage's travel
    double axial_scale = 0;  // the sample travels (1 + axial_scale) rows per row of the record
    double d[2] = {0, 0};    // where the sample's origin sits on the rotation stage [px]
    double height = 0;       // rows
    // (cos, sin) [rows]: the height once per turn that the scan's own views show beyond the stage's (the model's
    // note on the pitch, above); |height_once| / the scan's distance from the anchor bounds the pitch from below
    double height_once[2] = {0, 0};
    std::vector<double> stretch_height;  // per stretch: rows beyond the scan's height
};
struct StageParameters {     // the rotation stage within a turn, as (cos, sin) coefficients
    double height_once[2] = {0, 0};  // rows the sample is lifted, of the angle
    double axis_twice[2] = {0, 0};   // columns the axis moves across the detector, of twice the angle
};
struct Parameters {
    double roll = 0;         // of the rotation axis against the detector's columns [rad]
    StageParameters stage;
    // The detector's map (geometry::empty_map gives its terms). No terms: none. Eu's line in the column is a
    // scale of the columns, which nothing tells from the pixel size: it stays what it is given as.
    geometry::Map map;
    std::vector<ScanParameters> scans;
};
Parameters empty_parameters(const Acquisition& acq);

// A piece of the sample as one scan holds it: the stored pixel (row, col) of a frame, fractions allowed.
struct Sighting {
    int scan = 0, stretch = 0;  // stretch: index into ScanLayout::stretches
    double frame = 0, row = 0, col = 0;
};
// the frame at whose time the sighting's row was exposed: the rows' timing counts from the middle row
inline double exposure_frame(const Acquisition& acq, const Sighting& g) { return g.frame + acq.row_time * (g.row - (acq.rows - 1) / 2.0); }
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
// The plan's blocks cut into patches of patch x patch (0: each block as one) and registered once, on the frames as
// stored (the map is the model's). Returns the accepted ones as comparisons: registered inside the search window with a correlation above
// 0.3, and no row of either block among those that hold the mount. kept (optional): per comparison, the index of
// its block in `planned`.
std::vector<Comparison> measure(Session& session, const Acquisition& acq, const std::vector<Planned>& planned, Device device, int patch = 128,
                                std::vector<int>* kept = nullptr);

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
    // Free unless held: every scan's axis, axis_slope and axial_scale, the roll, the stage's four, the height of
    // every stretch that is not a helix, of every scan but the anchor its d and height, and of every scan that
    // sits elsewhere on the rotation stage than the anchor (a mosaic's tile) its height_once. Held unless freed:
    // the anchor's d, height and height_once, the height_once of a scan at the anchor's place, the helix' height,
    // the map. A stage of the calibration is these two lists (names or their beginnings, as above; `held` wins).
    std::vector<std::string> held, freed;
};
struct GeometrySolution {
    Parameters parameters;
    FitSummary solver;
    std::vector<std::string> free;    // the unknowns that were solved for
    // Asked free, but no comparison depends on them: held where they started. A scan's d and height count as
    // depended on only by comparisons of that scan with another (its own views cancel them but for what an error
    // of their frames leaves).
    std::vector<std::string> unused;
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
// As JSON text: the parameters by name (roll, stage, scans, map).
std::string parameters_json(const Acquisition& acq, const Parameters& p);
// As JSON text: the parameters by name, how the solve ended, what is left.
std::string solution_json(const Acquisition& acq, const GeometrySolution& s);

// ---------------------------------------------------------------- what a solution says beyond its parameters

// The scans whose reconstructions share a part of the sample. A scan alone reconstructs a disc about its rotation
// axis out to the detector's far edge; two discs overlap where the axes lie closer than the two radii together: a
// mosaic's neighbours share a lens, the two scans of a pair moved sideways the first one's whole disc. The axes'
// places are the record's motors' (ScanLayout::on_stage; scans the record gives no place lie at one), and a scan
// whose axis column is not known has no disc. triangles: three scans of which each two are an edge (what the
// closure of the edges' displacements is taken around).
struct Neighbours {
    std::vector<std::array<int, 2>> edges;      // (j, k), j < k
    std::vector<std::array<int, 3>> triangles;  // (i, j, k), i < j < k
};
Neighbours neighbours(const Acquisition& acq);

// What a solution leaves of one scan's opposite views (those of its helix; without one, of the stretch that has
// the most), and what they say that is not a parameter. NaN: not to be had from these comparisons.
struct ScanReport {
    int stretch = -1;       // the stretch they are of (index into ScanLayout::stretches); -1: the scan has no opposite views
    std::size_t n = 0;      // how many
    // What is left as a course of the angle A is seen at, per component (frames, rows, columns): a mean, once per
    // turn and twice per turn as (cos, sin), and the rms that leaves. Fitted to the patches, robustly. The rows'
    // once per turn and the columns' twice per turn are the model's (the stage's, and the scan's height_once):
    // nothing, unless they were held.
    double mean[3], once[3][2], twice[3][2], left[3];
    // The rows' timing these views would give [frames per stored row]: their two blocks lie rows_apart rows from
    // each other (half a turn's travel), so the same lines were exposed that many rows' time further apart than
    // half a turn, and the frames the registration finds them at say it: the acquisition's row_time less the median
    // of (frames left / rows between the two views). It stands only where a fraction of a frame is measured on
    // content as deep as the sample (registration.hpp samples the frames; content far from the axis crosses
    // several columns per frame).
    double row_time, rows_apart;
    // The record's height against a straight line over the stretch's frames on disk [rows, rms], how much that
    // departure differs between a comparison's two views [rows, rms], and whose it is: the rows left, regressed on
    // that difference beside a sinusoid of the angle. height_slope is 1 + the regression's slope: 1 if the sample
    // moved as the record says (the record is used frame by frame, and is right), 0 if it moved along the line
    // and the departure is the encoder's. +- height_slope_error.
    double departure, departure_between_views, height_slope, height_slope_error;
    // A scan elsewhere on the rotation stage than the anchor: its distance from the anchor by the motors [px], the
    // pitch its height_once then says [rad] (|height_once| / that distance: the rotation axis leans at least that
    // much towards the beam, more if the depth of what the views register follows the scan's axis by less than
    // the whole), and the direction of height_once as (cos, sin) less the direction of the scan from the anchor
    // [deg]: the same for every scan if it is the pitch.
    double from_anchor, pitch, pitch_phase_deg;
    ScanReport();
};
struct GeometryReport {
    std::vector<ScanReport> scans;
    // over the scans that have opposite views of their own: the largest less the smallest, the standard deviation,
    // the mean. One rotation stage and one axis stage say they should agree.
    struct Over {
        int n = 0;
        double range = 0, sd = 0, mean = 0;
    };
    Over axis, axis_slope, axial_scale;
    Neighbours neighbours;
};
GeometryReport report(const Acquisition& acq, const GeometrySolution& solution, const std::vector<Comparison>& comparisons);
std::string report_json(const Acquisition& acq, const GeometryReport& r);

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
