#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <optional>
#include <cstdint>

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
 * @param seed Seed for the generator that draws the sampled segments and the candidate
 *             points. With no seed the generator is seeded once from the hardware, as before,
 *             so two calls give different answers; with a seed a run can be repeated exactly.
 * @return A cv::Vec2f representing the estimated umbilicus point in the 2D grid space.
 */
cv::Vec2f align_and_extract_umbilicus(const GridStore& grid_store,
                                      std::optional<std::uint32_t> seed = std::nullopt);

// Adds each trace to the grid, resampled so that consecutive kept points are at
// least `spiral_step` apart. Traces that end up with fewer than two points are
// skipped.
void populate_normal_grid(const std::vector<std::vector<cv::Point>>& traces, GridStore& normal_grid, double spiral_step);

} // namespace util
} // namespace core
} // namespace vc
