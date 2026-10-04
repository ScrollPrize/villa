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
// interior cells, seeded by a 4-neighbor diffusion pass relaxed to the
// harmonic fill of the hole from its rim.
//
// unit <= 0 (the default) takes the target spacing from the median
// 4-neighbor distance between the known cells of each ROI, i.e. the grid step
// of the surface (1 / scale voxels for a tifxyz); a positive unit is used as
// given.
//
// Cells touching the grid border are left untouched (legitimate outer
// padding). Returns the number of invalid cells that were filled in.
int inpaintSurfaceHoles(cv::Mat_<cv::Vec3f>& points,
                        double unit = 0.0,
                        int max_iters = 2000);

} // namespace vc::core::util
