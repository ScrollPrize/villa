#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace vc {
namespace core {
namespace util {

class GridStore;

/**
 * @brief Aligns the normals of a set of QuadSurface segments and extracts an umbilicus point.
 *
 * This function implements a RANSAC-based approach to find a common umbilicus point
 * for a collection of surface segments. It then aligns the normals of these segments
 * to point outwards from this umbilicus.
 *
 * @param grid_store The GridStore containing the normal grid segments.
 * @return A cv::Vec2f representing the estimated umbilicus point in the 2D grid space.
 */
cv::Vec2f align_and_extract_umbilicus(const GridStore& grid_store);

} // namespace util
} // namespace core
} // namespace vc