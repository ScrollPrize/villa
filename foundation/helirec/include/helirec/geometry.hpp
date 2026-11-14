#pragma once
// A geometry calibration as the reconstruction reads it: the detector's map, where every frame holds the sample
// (poses), and the per-projection terms the reconstruction is given (helical.hpp, ScanTrajectory).
//
// Two kinds of description are read, both JSON as the calibration steps save them:
//
//   Calibration   a two-scan helical acquisition: helirec-calibration geometry's file ("model": "joint";
//                 geometry_calibration.hpp states the model), of which the two scans of a pair are read here; and
//                 still a file of the earlier two-scan step (with its motion stage), which nothing writes any more
//   TileGeometry  the tiles of a mosaic (python/helirec/mosaic.py states its form; nothing makes one at present)
//
// Nothing here reads frames: the record's angle and height of every frame are given by the caller.
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "helirec/common.hpp"

namespace helirec {
struct Json;
}

namespace helirec::geometry {

// The detector's map: a polynomial in the column and the row,
//     E(c, v) = sum_k e_k ((c - c0) / cs)^i_k ((v - v0) / vs)^j_k        for the columns (eu) and the rows (ev).
struct Map {
    std::vector<int> i, j;
    std::vector<double> eu, ev;
    double c0 = 0, cs = 1, v0 = 0, vs = 1;
};

// Powers (i, j) of the map: all of total order 2 to `order`, and (1, 0) (the rows' slope across the columns). The
// other linear ones are a scale, a shear and a rotation of the grid, which cannot be told from the poses.
std::vector<std::array<int, 2>> map_terms(int order);
Map empty_map(int rows, int cols, int order = 3);
// the value of each of the map's terms at the detector position (c, v): phi[k] = ((c - c0) / cs)^i_k ((v - v0) / vs)^j_k
void map_basis(const Map& m, double c, double v, double* phi);
// (Eu, Ev) at n detector positions (c, v)
void map_displacement(const Map& m, const double* c, const double* v, std::size_t n, double* eu, double* ev);

// Where every frame of one scan holds the sample, one value per frame. The ray at u columns from the rotation
// axis (towards higher columns) in the slice at height h (rows) is recorded by frame k at the position (X, Y) of
// the scan's grid,
//     Y = h + row[k] + row_per_col[k] u,        X = axis[k] + u + col_per_row[k] (Y - middle row),
// which is the detector pixel (c, v) with
//     c + Eu(c, v) + col_scale[k] (c - scale_col) = X,        v + Ev(c, v) + row_scale[k] v = Y,
// recorded at the angle angle_deg[k] + angle_deg_per_row[k] (v - middle row).
struct Poses {
    std::vector<double> row, row_per_col, axis, col_per_row, col_scale, row_scale, angle_deg, angle_deg_per_row;
    double scale_col = 0;
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

// The frame two scans are merged in: one rotation axis in the first scan's columns and the second scan's place
// against the first (rows, columns), rounded to `quantum` pixels; `measured`: the unrounded medians.
struct MergedLayout {
    double cor_main_detector = 0;
    std::array<double, 2> vector{};
    double axis_first_scan = 0, axis_second_scan = 0, second_scan_rows = 0, quantum = 0;
};

// The middle of the columns two scans both hold, in the first scan's columns: the second scan lies `move` columns
// from the first, each `cols` wide.
double overlap_centre(int cols, double move);

// row_origin: rows added to every pose (the slice numbering's origin). Everything the poses hold beyond the record
// goes into the terms, so z_translation_scale is 1.
Terms reconstruction_terms(const Poses& pose, const std::vector<double>& heights_rows, double row_sign, int rows,
                           double rot_center, double column_offset, double z_readjustment = 0, double row_origin = 0);

// The geometry calibration of a two-scan helical acquisition (rest turn, helix, rest turn in each scan).
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
    // The rows' timing as the reconstruction takes it: the time from the first stored row to the last, in frame periods.
    double rolling_shutter() const;
    // The middle of the columns both scans hold, in the first scan's columns.
    double overlap_centre() const;
    // The first scan's helix among its projection frames (all its projection stretches one after the other): [first, end).
    std::array<std::size_t, 2> helix_frames() const;

    // angles_deg[s], heights_rows[s]: the record's angle (unwrapped) and height (rows: row_sign x height / pixel)
    // of every frame of scan s, the stretches one after the other (rest1, helix, rest2).
    //
    // The first scan is placed by what its own redundant views measured: the helix against the encoder (a line in
    // the turn, from views a turn apart), the rest turns against the helix at equal angle, the axis' column of
    // each stretch with the rotating body's place against it, the height once per turn. The second scan is the
    // first one's model at the same frame plus what the comparison of the two scans at equal frames measured.
    // not_applied: receives, as JSON, the terms the step measured and this does not use.
    //
    // A joint calibration's poses are its model's, frame by frame (th and H the record's angle and height):
    //     row         height + the stretch's height + (1 + axial_scale) H + the stage's lift once per turn - roll d . n(th)
    //                 + pitch d . (-sin th, cos th) (the pitch of the axis towards the beam, where the calibration has
    //                 one: the row of the rays through the sample's origin),
    //                 counted from the first scan's row at its helix' first frame (the slices are numbered from there)
    //     axis        axis + axis_slope (H - H_mid) + the stage's axis column twice per turn + d . n(th)
    //     the slopes  col_per_row = roll, row_per_col = -roll
    //     the rows' timing: angle_deg_per_row the angle's step x the frames per row, and row_scale minus the
    //                 travel per frame x the frames per row about the middle row (the sample travels while the
    //                 rows are read)
    std::array<Poses, 2> frame_poses(const std::array<std::vector<double>, 2>& angles_deg,
                                     const std::array<std::vector<double>, 2>& heights_rows,
                                     std::string* not_applied = nullptr) const;

    // ---- one scan of a joint calibration of any number of scans (a mosaic's tile), reconstructed alone
    // the scan the calibration's layout names so
    std::size_t scan_named(const std::string& name) const;
    // the frames of its helix the calibration was made on, among the scan's projection frames: [first, end)
    std::array<std::size_t, 2> frames_on_disk(std::size_t scan) const;
    double axis_column(std::size_t scan) const;  // at the middle row, at the middle of the helix' frames on disk
    // The scan's poses in its own frame: as frame_poses' of a joint calibration with the scan's place d and its
    // height left out, so that u counts from the scan's own rotation axis and the slice h is the model's height z
    // of the sample as this scan holds it (z of the sample + the scan's height). What a scan alone reconstructs
    // depends on its own parameters only (axis column and its slope, axial scale, the roll, the stage's motion,
    // the map, the rows' timing); its place, height and rotation are a rigid motion of that volume. One term of
    // the place stays: the pitch's row, pitch d . (-sin th, cos th), which is no rigid motion of the volume (it
    // puts the rows right for the sample's origin and not for the scan's own axis). The frames of stretches the
    // calibration does not hold stay at zero.
    Poses own_poses(std::size_t scan, const std::vector<double>& angles_deg, const std::vector<double>& heights_rows) const;

    // The medians over the helix of the axis' column in each scan and of the second scan's rows against the
    // first's at equal frames. The frame is a nominal one: every projection's own axis and row are in its terms,
    // which are counted from it, so the rounding moves nothing; it fixes the slices' grid.
    MergedLayout merged_layout(const std::array<Poses, 2>& poses, const std::array<std::vector<double>, 2>& heights_rows,
                               double quantum = 0.25) const;

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

}  // namespace helirec::geometry
