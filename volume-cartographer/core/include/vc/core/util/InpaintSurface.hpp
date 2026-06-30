#pragma once

#include <opencv2/core/mat.hpp>

namespace vc::core::util {

// Ceres-powered inpainting of invalid cells in a tifxyz-style point grid.
//
// A cell is invalid if any channel is non-finite or x == -1 (sentinel). For
// each connected component of invalid cells that does NOT touch the grid
// border (i.e. genuine interior holes), the function solves a small
// least-squares problem on a dilated ROI with two losses:
//   - DistLoss: preserve 4-neighbor grid spacing (target = unit voxels apart)
//   - StraightLoss: penalize bending along grid rows/columns
// Boundary-ring cells around the ROI are fixed. The unknowns are the invalid
// interior cells, seeded by a quick 4-neighbor diffusion pass.
//
// Cells touching the grid border are left untouched (legitimate outer
// padding). Returns the number of invalid cells that were filled in.
int inpaintSurfaceHoles(cv::Mat_<cv::Vec3f>& points,
                        double unit = 1.0,
                        int max_iters = 2000);

// Heal degenerate grid cells: grid-adjacent points that have collapsed onto
// (nearly) the same 3D location. These are tracer defects -- a healthy sheet
// keeps adjacent cells ~one step (1/scale voxels) apart -- and they produce
// zero/near-zero-area triangles that make SLIM / symmetric-Dirichlet energy
// blow up to NaN. Real folds / scroll wraps are never grid-adjacent, so this
// never touches them.
//
// A cell is dropped (set invalid) if any 4-neighbor valid edge is shorter than
// min_edge_frac * expected_step, where expected_step = 1/scale voxels (from the
// segment meta; pass scale <= 0 to derive it from the median valid step).
// Dropped cells are then refilled via inpaintSurfaceHoles, so the output stays
// a complete grid interpolated from the healthy surroundings.
//
// Returns the number of cells dropped. dropped_mask, if non-null, is sized to
// points and set to 1 where a cell was marked invalid, so callers can clear the
// same cells in other channels (e.g. approval).
int healDegenerateCells(cv::Mat_<cv::Vec3f>& points,
                        double scale,
                        double min_edge_frac = 0.10,
                        cv::Mat_<uchar>* dropped_mask = nullptr);

} // namespace vc::core::util
