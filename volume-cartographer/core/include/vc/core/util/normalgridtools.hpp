#pragma once

#include <vector>
#include <vc_reader/thinning.hpp>
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
 * @param uncrush Strength beta >= 0 of an un-crush step for crushed scrolls: before the search the
 *             section is stretched along the axis its segment normals crowd towards, by the
 *             orientation tensor of those normals to the power beta / 2, and the result is mapped
 *             back. Each part of the section counts once in that tensor however densely it was
 *             sampled, and a section spanning less than half a turn is left as it is. 0, the
 *             default, is the estimator unchanged; 1.5 was chosen on 3 scrolls with manual
 *             umbilici and tested on 13 others.
 * @return A cv::Vec2f representing the estimated umbilicus point in the 2D grid space.
 */
cv::Vec2f align_and_extract_umbilicus(const GridStore& grid_store,
                                      std::optional<std::uint32_t> seed = std::nullopt,
                                      float uncrush = 0.0f);

/**
 * @brief Fits one smooth axis through per-slice umbilicus estimates (robust local-linear, LOWESS).
 *
 * align_and_extract_umbilicus sees one slice at a time, and its estimate is randomised, so the
 * points of neighbouring slices scatter around the axis and now and then one lands far off. The
 * axis itself is a continuous curve in z. For each query z0 this fits x(z) and y(z) by weighted
 * least squares on a line, with tricube weights of half-width `half_width` in z, and repeats the
 * fit with bisquare weights on each estimate's 2D residual over 6 times the median residual, so an
 * estimate far from its neighbours stops counting (Cleveland 1979).
 *
 * @param estimates (z, x, y) of every estimate; several may share a z (repeats).
 * @param query_z The z at which the fitted axis is returned.
 * @param half_width Window half-width, in the units of z. Must be > 0.
 * @param robustness_iterations Bisquare reweightings; 0 gives the plain local-linear fit.
 * @return (x, y) at each query z. Where fewer than 3 estimates fall inside the window, the
 *         coordinatewise median of the estimates at exactly that z, or NaN if there are none.
 */
std::vector<cv::Vec2d> fit_umbilicus_along_z(const std::vector<cv::Vec3d>& estimates,
                                             const std::vector<double>& query_z,
                                             double half_width,
                                             int robustness_iterations = 3);

// Adds each trace to the grid, resampled so that consecutive kept points are at
// least `spiral_step` apart. Traces that end up with fewer than two points are
// skipped.
void populate_normal_grid(const vcr::thinning::Traces& traces, GridStore& normal_grid, double spiral_step);

} // namespace util
} // namespace core
} // namespace vc
