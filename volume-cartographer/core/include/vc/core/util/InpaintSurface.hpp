#pragma once

#include <opencv2/core.hpp>

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

} // namespace vc::core::util
