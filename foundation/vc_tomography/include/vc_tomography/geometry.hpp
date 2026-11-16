#pragma once
// A geometry calibration as the reconstruction reads it: the detector's map, where every frame holds the sample
// (poses), and the per-projection terms the reconstruction is given (helical.hpp, ScanTrajectory).
//
// Two kinds of description are read, both JSON as the calibration steps save them:
//
//   Calibration   a helical acquisition of scans that share their place on the rotation stage (one scan, or
//                 several moved sideways): vc_tomography_calibration geometry's file ("model": "joint";
//                 geometry_calibration.hpp states the model), any number of scans. Files of the earlier
//                 two-scan step (with a motion stage) are refused.
//   TileGeometry  the tiles of a mosaic (python/vc_tomography/mosaic.py states its form; nothing makes one at present)
//
// Nothing here reads frames: the record's angle and height of every frame are given by the caller.
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "vc_tomography/common.hpp"
#include "vc_tomography/detector_map.hpp"

namespace vc_tomography {
struct Json;
}

namespace vc_tomography::geometry {

// The detector's map: Map (detector_map.hpp). parse_map: from the JSON object a calibration stores it as
// (keys i, j, eu, ev, c0, cs, v0, vs).
Map parse_map(const std::string& json_object_text);

// Powers (i, j) of the map: all of total order 2 to `order`, and (1, 0) (the rows' slope across the columns). The
// other linear ones are a scale, a shear and a rotation of the grid, which cannot be told from the poses.
std::vector<std::array<int, 2>> map_terms(int order);
Map empty_map(int rows, int cols, int order = 3);
// (Eu, Ev) at n detector positions (c, v)
void map_displacement(const Map& m, const double* c, const double* v, std::size_t n, double* eu, double* ev);

// Where every frame of one scan holds the sample, one value per frame. The ray at u columns from the rotation
// axis (towards higher columns) in the slice at height h (rows) is recorded by frame k at the position (X, Y) of
// the scan's grid,
//     Y = h + row[k] + row_per_col[k] u,        X = axis[k] + u + col_per_row[k] (Y - middle row),
// which is the detector pixel (c, v) with
//     c + Eu(c, v) + col_scale[k] (c - scale_col) = X,        v + Ev(c, v) + row_scale[k] v = Y,
// recorded at the angle angle_deg[k] + angle_deg_per_row[k] (v - middle row); with rows_from_middle (a camera whose
// two halves are read at once) at angle_deg[k] + angle_deg_per_row[k] |v - middle row|.
struct Poses {
    std::vector<double> row, row_per_col, axis, col_per_row, col_scale, row_scale, angle_deg, angle_deg_per_row;
    double scale_col = 0;
    bool rows_from_middle = false;
};

// One scan's poses as the reconstruction's per-projection terms (ScanTrajectory). The reconstruction puts the
// merged column m at m - rot_center columns from the axis and at the scan's column m - column_offset, and the
// slice h at the row h - (z_pix[k] z_translation_scale + z_readjustment + row_offset[k]), z_pix the record's
// height in pixels from the scan's first projection.
struct Terms {
    std::vector<double> angles_deg;
    double z_translation_scale = 1;
    std::vector<float> row_offset, row_slope, col_offset, col_slope, col_scale, row_scale;
    double slope_col = 0, slope_row = 0, scale_col = 0, scale_row = 0;
};

// The frame the scans are merged in: one rotation axis in the first scan's columns and every other scan's place
// against the first (rows, columns), rounded to `quantum` pixels; `measured`: the unrounded medians.
struct MergedLayout {
    double cor_main_detector = 0;
    std::array<double, 2> vector{};  // the second scan's place (vectors[0]; nothing for a scan alone)
    double axis_first_scan = 0, axis_second_scan = 0, second_scan_rows = 0, quantum = 0;
    // every scan after the first, in the calibration's order: its place against the first (rows, columns), rounded;
    // and, unrounded, the axis' column in its own columns and its rows against the first's
    std::vector<std::array<double, 2>> vectors, measured;
};

// The middle of the columns two scans both hold, in the first scan's columns: the second scan lies `move` columns
// from the first, each `cols` wide.
double overlap_centre(int cols, double move);

// row_origin: rows added to every pose (the slice numbering's origin). Everything the poses hold beyond the record
// goes into the terms, so z_translation_scale is 1.
Terms reconstruction_terms(const Poses& pose, const std::vector<double>& heights_rows, double row_sign, int rows,
                           double rot_center, double column_offset, double z_readjustment = 0, double row_origin = 0);

// The geometry calibration of a helical acquisition of scans that share their place on the rotation stage.
class Calibration {
  public:
    static Calibration parse(const std::string& json_text);
    static Calibration load(const std::string& path);

    int rows() const;
    int cols() const;
    double row_sign() const;
    // the scans it was made from, as named there (the first: the one with the rotation axis in its field)
    std::vector<std::string> scans() const;
    // The record's height of every frame in rows, as the calibration counts them: row_sign x height / pixel size.
    std::vector<double> heights_rows(const std::vector<double>& z_translation_m, double pixel_size_m) const;
    Map map() const;
    // The rows' timing as the reconstruction takes it: the time from the first stored row to the last, in frame periods
    // (the row time x (rows - 1)); and whether it counts the rows from the middle row to both ends (a camera whose two
    // halves are read at once: geometry_calibration.hpp, Acquisition::rows_from_middle), the row v then exposed
    // rolling_shutter x |v - middle row| / (rows - 1) frames after the frame's own time.
    double rolling_shutter() const;
    bool rows_from_middle() const;
    // The beam's divergence [rad per stored column, signed]: the ray recorded at the column c of a frame is a
    // parallel beam's at the frame's angle + fan x (c - the middle column) (geometry_calibration.hpp,
    // Acquisition::fan, there in frames per column). 0: the record stated no source distance.
    double fan() const;
    // The middle of the columns both scans hold, in the first scan's columns.
    double overlap_centre() const;
    // The first scan's helix among its projection frames (all its projection stretches one after the other): [first, end).
    // Of a joint calibration of scans without a helix (circular scans): the first scan's first stretch.
    std::array<std::size_t, 2> helix_frames() const;

    // ---- the scans that share their place on the rotation stage, merged
    std::size_t scan_count() const;
    // The poses of every scan, one entry per scan in the calibration's order (the first: the one with the rotation
    // axis in its field). angles_deg[s], heights_rows[s]: the record's angle (unwrapped) and height (rows: row_sign x
    // height / pixel) of every frame of scan s, the stretches one after the other. not_applied: receives, as JSON,
    // what the model leaves and the poses do not hold. The frames of stretches the calibration does not hold (they
    // were not on disk) stay at zero: frames_placed says which it holds.
    //
    // The poses are the joint model's, frame by frame (th and H the record's angle and height):
    //     row         height + the stretch's height + (1 + axial_scale) H + the stage's lift once per turn - roll d . n(th)
    //                 + pitch d . (-sin th, cos th) (the pitch of the axis towards the beam, where the calibration has
    //                 one: the row of the rays through the sample's origin),
    //                 counted from the first scan's row at its helix' first frame (the slices are numbered from there;
    //                 of scans without a helix, from the first scan's row at its first frame)
    //     axis        axis + axis_slope (H - H_mid) + the stage's axis column twice per turn + d . n(th)
    //     the slopes  col_per_row = roll, row_per_col = -roll
    //     a scan's tilt against the first (its rotation's x and y, about the sample's origin at about_height): with
    //                 along = tilt . (-sin th, cos th), the part about the beam's direction at the frame,
    //                 col_per_row + along, row_per_col - along, axis + along x (the middle row's height - about_height).
    //                 The part about the direction across the beam (rows by the depth along the ray) and the spin
    //                 are not in the poses.
    //     the rows' timing: angle_deg_per_row the angle's step x the frames per row, and row_scale minus the
    //                 travel per frame x the frames per row about the middle row (the sample travels while the
    //                 rows are read). A timing that counts the rows from the middle to both ends is held for
    //                 scans that do not travel while a frame is read (circular scans); with a travel it is an
    //                 error (the rows would be scaled about the middle row in two senses, which poses do not hold)
    std::vector<Poses> scan_poses(const std::vector<std::vector<double>>& angles_deg, const std::vector<std::vector<double>>& heights_rows,
                                  std::string* not_applied = nullptr) const;
    // The frames of a scan the calibration was made on, among the scan's projection frames: [first, end) of each
    // of its stretches, in order, those that follow one another joined.
    std::vector<std::array<std::size_t, 2>> frames_placed(std::size_t scan) const;

    // ---- one scan of a joint calibration of any number of scans (a mosaic's tile), reconstructed alone
    // the scan the calibration's layout names so
    std::size_t scan_named(const std::string& name) const;
    // the frames of its helix (without one: of its first stretch) the calibration was made on, among the scan's
    // projection frames: [first, end)
    std::array<std::size_t, 2> frames_on_disk(std::size_t scan) const;
    double axis_column(std::size_t scan) const;  // at the middle row, at the middle of the helix' frames on disk
    // The scan's poses in its own frame: as scan_poses' with the scan's place d and its
    // height left out, so that u counts from the scan's own rotation axis and the slice h is the model's height z
    // of the sample as this scan holds it (z of the sample + the scan's height). What a scan alone reconstructs
    // depends on its own parameters only (axis column and its slope, axial scale, the roll, the stage's motion,
    // the map, the rows' timing); its place, height and rotation are a rigid motion of that volume. One term of
    // the place stays: the pitch's row, pitch d . (-sin th, cos th), which is no rigid motion of the volume (it
    // puts the rows right for the sample's origin and not for the scan's own axis). The frames of stretches the
    // calibration does not hold stay at zero.
    Poses own_poses(std::size_t scan, const std::vector<double>& angles_deg, const std::vector<double>& heights_rows) const;

    // The frame the scans are merged in (scan_poses): the medians over the frames of the first scan's helix the
    // calibration was made on of the axis' column in each scan and of every other scan's rows against the first's at
    // equal frames. The frame is a nominal one: every projection's own axis and row are in its terms, which are
    // counted from it, so the rounding moves nothing; it fixes the slices' grid. first_frame[s]: the first frame of scan s the reconstruction is given, from which
    // its record's heights count there (none: every scan's first). Scans without a helix (circular scans, which may
    // differ by a frame) are each placed by the medians over their own frames: a scan at another height has its
    // rows in its vector (a grid's second row lies a row's step from the first).
    MergedLayout merged_layout(const std::vector<Poses>& poses, const std::vector<std::vector<double>>& heights_rows,
                               const std::vector<std::size_t>& first_frame = {}, double quantum = 0.25) const;

  private:
    std::shared_ptr<const Json> doc_;
};

// The tiles of a mosaic: for the frame f of a tile at the record's angle (courses are lines between their knots)
//     th = th_record(f) + T(f) + yaw + rad_per_row (v - v0)      the angle at which the row v was exposed
//     u  = a + c1 cos th + s1 sin th + C(f)                       where the sample's origin falls on the detector
//     H  = h + R(f)                                               how far the sample lies from the record along the axis
class TileGeometry {
  public:
    static TileGeometry parse(const std::string& json_text);
    static TileGeometry load(const std::string& path);

    struct Pose {
        std::vector<double> angle_deg, column, height_rows;  // th [deg], u, H per frame
    };
    // angle_deg, frame: the record's angle and the frame number of each frame. row < 0: the middle row.
    Pose frame_pose(const std::string& tile, const std::vector<double>& angle_deg, const std::vector<double>& frame,
                    double row = -1) const;
    // One tile's frames as the reconstruction's terms, in the tile's own frame: the slice's centre is the tile's
    // rotation axis, and the slice h is the same height of the sample in every tile (row_origin, rows, moves
    // that numbering). height_rows: the record's height in rows (row_sign x height / pixel) of each frame.
    Terms reconstruction_terms(const std::string& tile, const std::vector<double>& angle_deg, const std::vector<double>& height_rows,
                               const std::vector<double>& frame, double rot_center, double row_origin = 0) const;

  private:
    std::shared_ptr<const Json> doc_;
};

}  // namespace vc_tomography::geometry
