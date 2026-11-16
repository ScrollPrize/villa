#pragma once
// The geometry calibration step (vc_tomography_calibration geometry): the detector's map and the places of any number of
// scans of one sample (a scan alone, a pair moved sideways, the tiles of a mosaic), measured from the acquisition
// itself with nothing typed in, on one model solved as one Ceres problem.
//
// calibrate_geometry runs the step; its stages are declared below it and can be called on their own: the layout,
// the rows' timing from a refill, the plan of comparisons, their measurement, and the solve.
// What the step writes is read back for the reconstruction by vc_tomography/geometry.hpp.
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

#include "vc_tomography/calibration/session.hpp"
#include "vc_tomography/calibration/solve.hpp"
#include "vc_tomography/geometry.hpp"

namespace vc_tomography::calibration::geometry_calibration {

using Log = std::function<void(const std::string&)>;

struct Options {
    // seconds per stored row (signed) if known for this camera configuration; 0: not known, and done without;
    // NaN: measured from a refill
    double row_time = std::nan("");
    // The rows' timing counts the rows from the middle row to both ends (Acquisition::rows_from_middle): a camera
    // whose two halves are read at once. Not in any record: said here. With it, a row time that is not given (NaN)
    // and that no refill gives is measured from the frames between scans at different heights
    // (row_time_between_scans), and it is an error if the scans are all at one height.
    bool rows_from_middle = false;
    Device device = Device::CPU;
    int order = 3;    // of the map's polynomial; 0: no map
    // the map's four terms of a lens' radial distortion solved as one unknown (GeometryOptions::radial_map), the
    // map then of the third order whatever `order` is: for scans compared along the frames' edges alone
    bool radial_map = false;
    int threads = 8;  // reading threads
    int patch = 128;  // the plan's blocks are cut into patches of patch x patch, each registered once
    // 1, 2, 3: measured on the frames read at a half, a quarter, an eighth (calibration::Session: from the frames'
    // own JPEG 2000 resolutions, or from `projections`). The patches are then of patch / 2^reduce reduced pixels
    // (at least 32: the same piece of the detector), and a helix' opposite views are taken at 12 x 2^reduce angles a
    // turn (at most 48), which costs little there. The calibration is written in stored pixels all the same
    // (to_stored_pixels). It is not the stored frames' to the last tenth of a pixel: the registration's band is in
    // cycles per sample, so it holds coarser detail, and the frames' low band is formed per chunk of rows. On
    // Paris4's tiles, against the stored frames: every axis column 0.09 px lower at a half and 0.22 at a quarter,
    // every axial scale 11 and 207 ppm lower, each the same on all tiles to 0.003 and 0.016 px, 7 and 20 ppm.
    int reduce = 0;
    // a projection store (STORE.zarr of vc_tomography_projections_zarr) to read the frames, darks and flats from, at
    // the level `reduce`, which it must hold; "": the detector's files beside the masters
    std::string projections;
    Log log;          // progress, a line at a time (none: silent)
};
// scans: the master files of the scans of one sample, a scan with the rotation axis inside its field first: the
// layout, the rows' timing, every comparison of the plan measured once, one solve. Returns the calibration as
// JSON text ("model": "joint"; vc_tomography/geometry.hpp reads a pair's); it is an error if the rows' timing is not
// given and nothing measures it (a refill in a stretch on disk; with Options::rows_from_middle, scans at two heights).
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
// ideal position (X, Y) = (c + Eu(c, v), v + Ev(c, v)), Eu and Ev the detector's map (vc_tomography/geometry.hpp; the
// same for every scan on the detector, none unless Parameters::map has terms). Scan k records the ray at the ideal
// position
//     X = axis_k + axis_slope_k (H - H_mid_k) + A(2 th) + u + roll (Y - middle row),        u (1 + column_scale_k) = s + d_k . (cos th, sin th)
//     Y = z + height_k + stretch height + (1 + axial_scale_k) H + B(th) - roll u
// with th and H the record's angle and height [rows] at the time the row was exposed: row v of frame f at the time
// of frame f + row_time (v - middle row), or f + row_time |v - middle row| for a camera whose two halves are read
// at once (Acquisition::rows_from_middle). A and B are what the rotation stage does within a turn, the same for
// every scan on it: B(th) = b_c cos th + b_s sin th lifts the sample once per turn, A(2 th) = a_c cos 2 th +
// a_s sin 2 th moves the axis across the detector twice per turn. Those are the parts opposite views see (the
// height at odd multiples of the turn, the axis column at even ones: PHerc0343P 0.03 rows and 0.02 px, Paris4
// 0.25 rows and 0.25 px); the axis column once per turn is the sample's place d. The unknowns are the places, nothing else: no comparison has an offset
// of its own, so what the model does not explain stays in the residuals.
// One term is not of a ray. A rotation axis pitched towards the beam makes every ray climb by the pitch per pixel
// of depth, so views half a turn apart are not the same rays: their rows differ by twice the pitch x the depth of
// what they register, and for a tile whose axis lies c from the middle of the sample's detail that depth is about
// -c . (-sin th, cos th): a sinusoid of the angle that follows the tile (Paris4: 0.5 to 0.6 rows between the views
// of the six tiles around the middle one). Which depth a registration weights is not known, so each scan has that
// sinusoid as an unknown of its own views, ScanParameters::height_once: B_k(th), added to the stage's B(th) in the
// scan's comparisons with itself and in nothing else (not in Y above, not between two scans). Left out, it goes
// into the scan's axial scale wherever the views do not cover whole turns. The anchor's is the stage's B.
// The pitch itself (Parameters::pitch) is exact between two scans at one angle, whatever the depth: scan k, with
// the sample at d_k on the rotation stage, records every ray pitch x d_k . (-sin th, cos th) rows higher, so the
// rows of two tiles at equal angle differ by pitch x (d_k - d_j) . (-sin th, cos th): a sinusoid of the angle of
// pitch x the tiles' distance, largest where the tiles line up along the beam, and the roll's partner (the roll
// moves the rows in step with the tiles' column offset, the pitch a quarter turn from it). That term is in the
// comparisons of two views at one angle and in no other. Opposite views of two scans have neither it nor an
// unknown of their own: their rows are reported and not fitted (their columns are exact and are).
// The beam's divergence is in the reported angle alone (Ray::angle_deg, the frames' residual): the ray recorded at
// column c is a parallel beam's at th + fan (c - middle column) frames (Acquisition::fan; the sign as PHerc0343P's
// opposite views show it: their frames lie 2 fan (c - axis) further on than half a turn). No row or column of
// the model takes it: s and z above are formed with the stage's angle. The reconstruction takes it in the
// backprojection (geometry::Calibration::fan, ReconConfig::fan_per_column).
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
//
// The other kind of measurement is taken in the volume (LensPatch, below): two scans that can each be reconstructed
// alone hold a part of the sample both (a mosaic's neighbours the lens of their two discs), and where the one's
// reconstruction has a piece of it against the other's is the difference of the two scans' rigid motions there.
// Scan k holds the sample's point p (x, y from the sample's origin along (cos, sin) of the angle 0, z its height)
// moved by
//     D_k(p) = (d_k, height_k) + rotation_k x (p - (0, 0, Parameters::about_height))
// against where the anchor holds it: the place the projections know, and a small rotation of the sample (a tilt
// about x and y, a spin about the rotation axis). A lens patch depends on the two scans' places and rotations and
// on nothing else, and a scan's own comparisons on neither: a mosaic's first solve is two that share no unknown.
// A comparison of two scans in projections holds of a rotation what is the same at every depth along the ray:
// the tilt's part about the beam's direction, tilt . (-sin th, cos th), moves the columns by it x the height from
// about_height and the rows by it x the distance s across the beam (model::turned_back). The tilt's part about
// the direction across the beam moves the rows by the depth, and the spin is the frames': neither is in a
// projection's residual, so tile pairs give the two tilts and not the spin. What ties them is outside any solve: a scan's region is reconstructed
// with the scan's own parameters (axis column and its slope, axial scale, roll, stage, map, rows' timing).

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
    // How the rows' timing counts the rows. false: from the first stored row to the last, row v exposed
    // row_time (v - middle row) frames after the frame's own time. true: from the middle row to both ends, row v
    // exposed row_time |v - middle row| frames after it: a camera whose upper and lower half are read at once (an
    // sCMOS with a rolling shutter, as the PCO.edge). row_time above nothing: the middle row first; below: the
    // first and last row first, the middle row last (Diamond I12 at PHerc. 172: -25.9 us per row, measured from
    // the frames between scans at two heights). The record does not say it: the caller sets it.
    bool rows_from_middle = false;
    double period_s = 0;    // of a frame (the median of the first scan's helix, or of its first stretch); 0: not known
    // The beam's divergence: frames of angle per column (pixel / source distance / angle per frame), signed. Above
    // nothing the ray recorded at a higher column is a parallel beam's further on in angle (ESRF BM18: PHerc0343P's
    // opposite views); below nothing further back: the sample turns the other way against the columns for the same
    // recorded angle (Diamond I12 at PHerc. 172: the frames between scans moved across the beam). The sign is the
    // record's scan/fan_sense (an NXtomo scan's supplement file states it; without it +).
    double fan = 0;
    // What the divergence does between scans at different places on the rotation stage: a sample that sits t px
    // nearer the source is recorded magnified by t x per_depth about the detector's middle (per_depth: the pixel
    // over the source's distance, 6.3e-9 for Paris4; + if the source lies along (-sin th, cos th) of the model,
    // which is the sense PHerc0343P's opposite views give the fan). Scan k holds the sample d_k . (-sin th, cos th)
    // nearer, the same at every depth of the sample: up to 0.1 px at the detector's edge between two of Paris4's
    // neighbours. 0: left out (read_acquisition leaves it so; the caller sets it).
    double per_depth = 0;
    int width = 512;        // columns of a window, the same for every kind of comparison
    // The rows the beam lights, [first, end): around the brightest row of the scans' mean flats (each row's mean
    // over the columns), the rows that reach 0.12 of it, in every scan. A beam cut by slits along the axis lights
    // a part of the detector (Diamond I12 at PHerc. 172: rows 254 to 1903 of 2160, the rows beyond reading 5 to 8 %
    // of the brightest), and the plan keeps its blocks inside it. Nothing (end <= first): every row.
    int lit_rows[2] = {0, 0};
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
// Where no scan has a helix (circular scans), the sense of the rows is that in which two scans at one place across
// the beam and different heights register at equal frames; scans all at one height have no sense to find, and it
// is left at -1. A record that states no source distance gives no fan (Acquisition::fan 0); one that states
// scan/fan_sense -1 gives the fan the other sign.
Acquisition read_acquisition(Session& session, Device device, const Log& log = {});
// Seconds per stored row from a refill in one of the acquisition's stretches, as the step saves it (JSON text;
// "null" if no stretch on disk holds one): the frames around the refill are found and reduced to the rows' levels
// here, the dips are fitted by calibration::fit_refill_shots (vc_tomography/calibration/refill.hpp).
std::string row_time_from_refill(Session& session, const Acquisition& acq, const Log& log);
// The layout without the per-frame series, as JSON text (for logs and files).
std::string acquisition_json(const Acquisition& acq);

// The comparisons of the scans, placed from the layout alone. Each scan with itself:
//   opposite    a stretch against itself half a turn on, mirrored about the coarse axis column; in a helix the rows
//               offset by the travel of half a turn, at angles_per_turn angles evenly around the turn and at most
//               max_positions positions along the helix; the patches started from the block registered whole
//   turn        the helix against itself a turn on, where a turn's travel leaves a patch of rows in common (128;
//               turn_least rows in frames read reduced, where a patch is smaller)
//   resthelix   a rest turn against the helix at equal angle, in the turn of the helix nearest its height
// and one scan against another with the same place on the rotation stage (a pair moved sideways):
//   pair        the stretches of the same name at equal angle, the second scan's window where the coarse axis
//               columns put the first's content, across the columns both hold; pair_positions positions along a
//               helix (as the opposite views': evenly around the turn), n_pair_rest in a rest turn. Not planned
//               where neither the columns nor the rows of the two blocks lie 48 px apart (a pattern fixed to the
//               detector would register).
struct PlanCounts {
    int angles_per_turn = 12, max_positions = 48, n_rest = 16, n_turn = 12, pair_positions = 24, n_pair_rest = 12;
    int turn_least = 128;  // the rows a turn's travel must leave in common for the helix to be compared a turn on
    // plan_tiles: positions along the helix per pair of scans (at tile_angles angles a turn), and at each the
    // most bands of rows and windows of columns
    int tile_positions = 48, tile_angles = 24, tile_bands = 3, tile_windows = 4;
};
std::vector<Planned> plan(const Acquisition& acq, const PlanCounts& counts = {});

// The model's unknowns. The names the solver knows them by: "roll", "pitch", "row_time", "stage.height_once_cos", "stage.height_once_sin",
// "stage.axis_twice_cos", "stage.axis_twice_sin", "scan<k>.axis", "scan<k>.axis_slope", "scan<k>.axial_scale", "scan<k>.column_scale",
// "scan<k>.d0", "scan<k>.d1", "scan<k>.height", "scan<k>.tilt_x", "scan<k>.tilt_y", "scan<k>.spin",
// "scan<k>.height_once_cos", "scan<k>.height_once_sin",
// "scan<k>.<stretch name>.height", and per term of the map "map.eu.i<i>j<j>", "map.ev.i<i>j<j>". A name's
// beginning up to a dot names all that follow it: "stage", "map", "map.ev", "scan1".
struct ScanParameters {
    double axis = 0;         // the axis column at the middle row, at the height H_mid
    double axis_slope = 0;   // columns the axis moves per row of the axis stage's travel
    double axial_scale = 0;  // the sample travels (1 + axial_scale) rows per row of the record
    // The scale of this scan's image across the beam against the anchor's: what the scan records u columns from its
    // rotation axis lies u (1 + column_scale) from the axis, so its image is smaller by column_scale. A scan's own
    // opposite views do not see it (their two sightings lie at u and -u); its comparisons with another scan do, and
    // so do lens patches of a scan reconstructed without it. Held unless freed by name; one scale for every scan is
    // a dilation of the places, so the anchor's stays at nothing. The reconstruction takes it in the scan's poses
    // (Poses::col_scale about the axis column). Paris4's tiles: it grows with the order of acquisition, to 3e-4.
    double column_scale = 0;
    double d[2] = {0, 0};    // where the sample's origin sits on the rotation stage [px]
    double height = 0;       // rows
    // (cos, sin) [rows]: the height once per turn that the scan's own views show beyond the stage's (the model's
    // note on the pitch, above); |height_once| / the scan's distance from the anchor bounds the pitch from below
    double height_once[2] = {0, 0};
    // (x, y, z) [rad]: the sample as this scan holds it is turned by this against the anchor's, about the sample's
    // origin at the height Parameters::about_height: the tilt (x, y) and the spin about the rotation axis (z).
    // Lens patches give it; comparisons of two scans in projections depend on the tilt (not on the spin), and it
    // is free there only if freed by name.
    double rotation[3] = {0, 0, 0};
    std::vector<double> stretch_height;  // per stretch: rows beyond the scan's height
};
struct StageParameters {     // the rotation stage within a turn, as (cos, sin) coefficients
    double height_once[2] = {0, 0};  // rows the sample is lifted, of the angle
    double axis_twice[2] = {0, 0};   // columns the axis moves across the detector, of twice the angle
};
struct Parameters {
    double roll = 0;         // of the rotation axis against the detector's columns [rad]
    // of the rotation axis towards the beam [rad]: a ray climbs by it per pixel of depth. Scans at different places
    // on the rotation stage, compared at equal angle, give it; nothing else does (the model's note, above).
    double pitch = 0;
    // The rows' timing [frames per stored row, signed]: row v of a frame is exposed row_time (v - middle row) frames
    // after the frame's own time (row_time |v - middle row| with Acquisition::rows_from_middle). NaN: the acquisition's (Acquisition::row_time: typed, or from a refill), and a
    // solution holds the value it was solved under. It is held unless freed by name ("row_time"). What gives it:
    // scans at different places on the rotation stage. The column offset of two such scans at one angle,
    // (d_k - d_j) . (cos th, sin th), is of the angle at which each row was exposed, so it is sheared along the rows
    // by the timing; and a scan at d whose opposite views lie rows apart sees them that much off half a turn, a
    // sinusoid of |d| x that angle in their columns. In one scan's own views about its own axis it is all but one
    // with the axial scale (GeometrySolution::correlated lists the two: -0.99998 on a synthetic helix).
    double row_time = std::nan("");
    StageParameters stage;
    // The detector's map (geometry::empty_map gives its terms). No terms: none. Eu's line in the column is a
    // scale of the columns, which nothing tells from the pixel size: it stays what it is given as.
    geometry::Map map;
    // the height of the sample [rows, the model's z] the scans' rotations are about (the middle of the slab the
    // lens patches were taken in): not an unknown, it says what d and height mean beside a rotation
    double about_height = 0;
    std::vector<ScanParameters> scans;
};
Parameters empty_parameters(const Acquisition& acq);

// A piece of the sample as one scan holds it: the stored pixel (row, col) of a frame, fractions allowed.
struct Sighting {
    int scan = 0, stretch = 0;  // stretch: index into ScanLayout::stretches
    double frame = 0, row = 0, col = 0;
};
// the rows the rows' timing counts for the stored row v: from the middle row, to one side below nothing, or to both
// sides above it (Acquisition::rows_from_middle)
inline double timing_rows(const Acquisition& acq, double row) {
    const double from_middle = row - (acq.rows - 1) / 2.0;
    return acq.rows_from_middle ? std::fabs(from_middle) : from_middle;
}
// the frame at whose time the sighting's row was exposed under a rows' timing [frames per row], which counts from
// the middle row; and the rows' timing the parameters hold, or the acquisition's where they hold none
inline double exposure_frame(const Acquisition& acq, double row_time, const Sighting& g) { return g.frame + row_time * timing_rows(acq, g.row); }
inline double row_time_of(const Acquisition& acq, const Parameters& p) { return std::isfinite(p.row_time) ? p.row_time : acq.row_time; }
struct Comparison {
    std::string kind;       // "opposite", "turn", "resthelix", "pair", "tiles", "tiles_opposite": what the report groups by
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

// Two scans' reconstructions of one place of the sample, registered in 3D: where scan b's volume has the piece
// against where scan a's has it (the model's note on the volume, above).
struct LensPatch {
    int scan_a = 0, scan_b = 0;
    double p[3] = {0, 0, 0};      // the place of the sample: x, y [px] from its origin, z [rows] (the model's z)
    double moved[3] = {0, 0, 0};  // D_b(p) - D_a(p) as measured: x, y [px], z [rows]
    // what one such measurement is good to, per component [px]: the floor measured on one scan reconstructed twice
    // from disjoint frames. A patch's residual enters the solve over it.
    double sigma[3] = {1, 1, 1};
    double correlation = 0;
};
// D_b(p) - D_a(p) under the parameters, less what was measured: x, y, z [px]
std::vector<std::array<double, 3>> lens_residuals(const Acquisition& acq, const Parameters& p, const std::vector<LensPatch>& lens);

// The comparisons of scans that sit at different places on the rotation stage (a mosaic's tiles), placed from
// where the parameters put them (each scan's axis column, d and height: a solve of the lens patches, or the
// motors'). pairs: which scans to compare, (j, k) (neighbours(acq).edges: the tiles whose discs overlap). Two such
// scans record the same rays at equal angle on columns (axis_k - axis_j) + (d_k - d_j) . (cos th, sin th) apart,
// which swings by the tiles' distance over a turn, so the helices are compared at tile_positions positions evenly
// around the turn, at each in one of two ways:
//   tiles           at equal angle, where that offset leaves a window on both detectors (about the two angles at
//                   which the tiles line up along the beam; not where neither the columns nor the rows lie 48 px
//                   apart: a pattern fixed to the detector would register)
//   tiles_opposite  the second scan half a turn on (or, where its frames end before that, half a turn back),
//                   mirrored: its columns are then axis_j + axis_k + (d_j - d_k) . (cos th, sin th) less the first's.
//                   With the axis near the detector's edge (a half acquisition) that is on both detectors over the
//                   half of the turn in which the second tile lies on the first's far side
// (at the remaining angles, about a third of the turn for Paris4's neighbours, the two tiles share no ray)
// with up to tile_windows windows across the columns both hold and tile_bands bands of rows. The offset's course
// over a block's frames is given to the request as its sway; the patches start from the block registered whole.
std::vector<Planned> plan_tiles(const Acquisition& acq, const Parameters& places, const std::vector<std::array<int, 2>>& pairs, const PlanCounts& counts = {});

// The ray a sighting records under the parameters: the angle [deg], s [px], z [rows] (the pitch's lift not taken
// out: it belongs to a comparison of two views at one angle).
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
    // of a scan without a helix (it does not travel: a circular scan) its axis_slope, its axial_scale and its
    // first stretch's height,
    // the map, the rows' timing, every scan's column_scale. A scan's rotation is free if lens patches name the scan and it is not the anchor. The pitch is free
    // if two scans at different places on the rotation stage are compared at equal angle. Where no comparison is
    // given (lens patches alone), only places and rotations are free.
    // A stage of the calibration is these two lists (names or their beginnings, as above; `held` wins).
    std::vector<std::string> held, freed;
    // on the length of a lens patch's residual, each component over its sigma (so: in sigmas)
    double lens_huber = 1.5;
    // Above 0: a free term of the map whose estimate is correlated beyond this with another unknown's
    // (GeometrySolution::correlated) is held where it started and the rest solved again, up to four times;
    // GeometrySolution::undetermined lists what was held so. The comparisons cannot tell such a term from the other
    // unknown: where the scans are compared along the frames' edges alone (scans moved sideways by nearly the
    // detector's width, a helix that travels nearly its height per turn), the map's course along the columns alone is
    // one with the scans' axis columns and heights, and its course along the rows alone with the axial scales. What
    // is held is not measured: it is in neither the map nor the residuals. The step solves with 0.95 (PHerc0343P's
    // largest such correlation is 0.91, the roll's with two terms: nothing of it is held).
    double undetermined_map = 0;
    // The map's four terms of a lens' radial distortion as one unknown, "map.radial", in place of the four
    // coefficients: the pixel x columns and y rows from the detector's middle is displaced by k (x, y) (x^2 + y^2),
    //     Eu = k (x^3 + x y^2),        Ev = k (x^2 y + y^3),
    // "map.radial" = k R^3 [px: the displacement R pixels from the middle along the columns, R the map's column
    // scale, half the detector's width]. The map's other terms stay unknowns of their own; all are held unless
    // freed ("map" names them all). The four coefficients (map.eu.i3j0, .eu.i1j2, .ev.i2j1, .ev.i0j3) are then not
    // unknowns, and freeing one by name is an error. The start's map must hold the terms of the third order
    // (geometry::empty_map(rows, cols, 3)); its radial unknown is read from its Eu's term in the column cubed. The
    // solution's map holds the four coefficients written out, so everything that reads a map reads it;
    // GeometrySolution::radial_map says the unknown and k.
    // For scans compared along the frames' edges alone (scans moved sideways or along the axis by nearly a frame: a
    // grid of circular scans). Two such scans compare the detector's one edge with its other, so they give the
    // terms that are odd across the seam and change along it (x y^2 at a seam between columns, x^2 y between rows)
    // and not the course of the columns along the columns or of the rows along the rows (x^3, y^3), which is all
    // that the inside of a frame is displaced by against its edges. A lens' distortion ties the two: the four terms
    // are one. The unknown is not among what GeometryOptions::undetermined_map holds.
    bool radial_map = false;
    // The sample's tilt between scans, solved after the rest: two scans compared at one angle hold the part of it
    // about the beam's direction (rows by the tilt x the distance across the beam, once per turn), so every scan
    // compared so with another, but the anchor, gets its tilt_x and tilt_y in a second solve that starts from the
    // first and keeps every scan's axial scale and the map's terms held above where the first has them. A scan's
    // axial scale is what its own views give (a turn or half a turn apart), which are few beside the comparisons
    // of two scans: solved together with the tilts those pull it away (PHercParis4's four scans at 2.4 um: one
    // scan's by 4e-4, a row per turn against its own views, and its seam with its neighbour 0.8 quarter-pixels off
    // along the axis at the end of the slab on disk; kept, the seam is 0.13). The tilts are about the sample's
    // origin at the start's about_height. The step solves so.
    bool tilts_after = false;
};
struct GeometrySolution {
    Parameters parameters;
    FitSummary solver;
    std::vector<std::string> free;    // the unknowns that were solved for
    // Asked free, but no comparison depends on them: held where they started. A scan's d and height count as
    // depended on only by comparisons of that scan with another and by lens patches (its own views cancel them
    // but for what an error of their frames leaves).
    std::vector<std::string> unused;
    // terms of the map held because the comparisons do not tell them from another unknown (GeometryOptions::undetermined_map)
    std::vector<std::string> undetermined;
    // The map's radial unknown as solved (GeometryOptions::radial_map): "map.radial" [px], and k [1 / px^2]
    struct RadialMap {
        bool solved = false;
        double cubic = 0, k = 0;
    };
    RadialMap radial_map;
    int rank = 0;
    std::vector<std::array<double, 3>> residual;       // per comparison: frames, rows, columns
    std::vector<std::array<double, 3>> lens_residual;  // per lens patch: x, y, z [px] (lens_residuals)
    // what the solution leaves, per kind and pair of scans (scan_a = scan_b = -1: the kind over all scans)
    struct Left {
        std::string kind;  // a comparison's kind, or "lens"
        int scan_a = -1, scan_b = -1;
        std::size_t n = 0;
        double rms[3] = {0, 0, 0}, median[3] = {0, 0, 0};  // frames, rows, columns; of lens patches x, y, z
    };
    std::vector<Left> left;
    // Free unknowns that the measurements hardly tell apart: pairs whose estimates are correlated by more than
    // 0.9 in size (from the Jacobian at the solution; the rank check passes them). Their sum or difference is
    // known, each alone much less.
    struct Correlated {
        std::string a, b;
        double correlation = 0;
    };
    std::vector<Correlated> correlated;
};
// The places that best explain the comparisons and the lens patches, from `start`. It is an error if the free
// unknowns depend on each other (the message names a dependent set) or the solve is not usable.
GeometrySolution solve_geometry(const Acquisition& acq, const std::vector<Comparison>& comparisons, const std::vector<LensPatch>& lens, const Parameters& start,
                                const GeometryOptions& options = {});
inline GeometrySolution solve_geometry(const Acquisition& acq, const std::vector<Comparison>& comparisons, const Parameters& start, const GeometryOptions& options = {}) {
    return solve_geometry(acq, comparisons, {}, start, options);
}
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
    // half a turn, and the frames the registration finds them at say it: the solution's rows' timing less the median
    // of (frames left / rows between the two views). It stands only where a fraction of a frame is measured on
    // content as deep as the sample (registration.hpp samples the frames; content far from the axis crosses
    // several columns per frame). Where the timing counts the rows from the middle to both ends, the rows between
    // the views are those it counts (timing_rows).
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
// What a solution leaves of the comparisons of one scan with another, per kind: as a course of the angle A is seen
// at, per component (frames, rows, columns), a mean and once per turn as (cos, sin), and the rms that leaves.
// NaN where the angles do not determine it. For scans at different places on the rotation stage the rows' once per
// turn at equal angle is what the pitch left (nothing, if it was solved); in opposite views it is what the pitch
// does over the depth of what was registered, which the solve does not fit.
struct PairReport {
    std::string kind;
    int scan_a = 0, scan_b = 0;
    std::size_t n = 0;
    double mean[3], once[3][2], left[3];
    // the rms of the residuals as they are (frames, rows, columns), and their median
    double rms[3], median[3];
    // The rows' timing the pair's frames give [frames per stored row], +- row_time_error: the two views of a patch
    // lie at different rows, so the same lines were exposed (the rows the timing counts between them) x the row
    // time apart, and the frames left follow those rows: the solution's rows' timing less the slope of the frames
    // left against them (beside a constant: the two scans' angles are not known against each other to a fraction
    // of a frame). NaN where the rows between the views are all but the same in every patch (their spread under
    // 16): scans at two heights under a timing that counts from the first row to the last lie the same rows apart
    // everywhere. Under one that counts from the middle to both ends they do not, and this is what measures it.
    double row_time, row_time_error, timing_rows_spread;
    PairReport();
};
// The rows' timing the frames between scans give, over every pair of scans compared at equal angle whose views lie
// at rows the timing counts differently from patch to patch (PairReport::row_time states why that measures it, and
// when it does: scans at two heights under a timing that counts the rows from the middle to both ends): one slope
// of the frames left against those rows, a constant per pair of scans. row_time [frames per stored row]: the
// parameters' rows' timing less that slope, +- error; n: the patches it is of (0: no pair gives it, the rest NaN).
// The frames are those of the record's angles alone, so no parameter but the rows' timing enters.
// It stands as far as a fraction of a frame is measured on content as deep as the sample (ScanReport::row_time).
struct RowTimeBetweenScans {
    std::size_t n = 0, pairs = 0;
    double row_time = std::nan(""), error = std::nan("");
};
RowTimeBetweenScans row_time_between_scans(const Acquisition& acq, const Parameters& p, const std::vector<Comparison>& comparisons);

struct GeometryReport {
    std::vector<ScanReport> scans;
    std::vector<PairReport> pairs;
    RowTimeBetweenScans row_time_between_scans;
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

// A calibration made on frames read at 1 / f in the pixels of the frames as stored (rows x cols): every position
// times f (a reduced value sits at the stored pixel of f times its index, nothing added), every ratio of two
// lengths as it is (the roll, the axial scale, the axis column's slope, a pitch). Two things do not scale: the
// reduced frames' middle row is the stored row f (reduced rows - 1) / 2, (f - 1) / 2 rows above the stored
// frames' middle, so each scan's axis column is moved along the roll to the stored middle row; and the rows'
// timing, counted from the reduced middle row, is left counting from there (a common time of (f - 1) / 2 rows,
// nothing where the row time is zero; where it counts to both ends, the rows within (f - 1) / 2 of the middle are
// off by up to that many rows' time). The solution's residuals and the report are converted with it.
void to_stored_pixels(Acquisition& acq, GeometrySolution& solution, GeometryReport& told, int rows, int cols, int f);

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
}  // namespace vc_tomography::calibration::geometry_calibration
