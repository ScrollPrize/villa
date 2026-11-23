#include "vc/core/util/SheetWalk.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::sheet {
namespace {
using Points = std::vector<cv::Vec3f>;

cv::Vec3f unit(const cv::Vec3f& v)
{
    const double length = cv::norm(v);
    if (!std::isfinite(length) || length < 1e-6)
        throw std::runtime_error("Sheet growth edge collapsed; use a smaller mesh spacing.");
    return v / static_cast<float>(length);
}

std::vector<double> arcLengths(const Points& points)
{
    std::vector<double> arc(points.size());
    for (std::size_t i = 1; i < points.size(); ++i)
        arc[i] = arc[i - 1] + cv::norm(points[i] - points[i - 1]);
    return arc;
}

cv::Vec3f across(const Points& points, std::size_t i)
{
    const auto lo = i == 0 ? 0 : i - 1;
    const auto hi = std::min(i + 1, points.size() - 1);
    return unit(points[hi] - points[lo]);
}

// Keep the source edge's arc-length spacing and its middle vertex as the
// parameterization anchor. If the front contracts, continue its ends along
// the sheet instead of piling several vertices onto an endpoint.
Points redistribute(const Points& predicted, const std::vector<double>& spacing,
                    const SheetNormalFn& normal, const SheetWalkOptions& options)
{
    const auto arc = arcLengths(predicted);
    const auto center = (predicted.size() - 1) / 2;
    Points result(predicted.size());
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        const double target = arc[center] + spacing[i] - spacing[center];
        if (target < 0 || target > arc.back()) {
            const bool first = target < 0;
            const auto endpoint = first ? predicted.front() : predicted.back();
            const auto tangent = first ? unit(predicted.front() - predicted[1])
                                       : unit(predicted.back() - predicted[predicted.size() - 2]);
            const double distance = first ? -target : target - arc.back();
            SheetWalk extension{0, 1, {endpoint}, {tangent * static_cast<float>(distance)}, options};
            Points point(1);
            walkSheetLine(extension, 0, normal, point);
            result[i] = point[0];
        } else {
            auto upper = std::upper_bound(arc.begin(), arc.end(), target);
            if (upper == arc.end()) {
                result[i] = predicted.back();
            } else {
                const auto j = static_cast<std::size_t>(upper - arc.begin());
                const double t = (target - arc[j - 1]) / (arc[j] - arc[j - 1]);
                result[i] = predicted[j - 1] + (predicted[j] - predicted[j - 1]) * static_cast<float>(t);
            }
        }
    }
    return result;
}
}

void walkSheetStrip(const SheetWalk& walk, const SheetNormalFn& normal,
                    std::vector<cv::Vec3f>& guess, unsigned workers,
                    const SheetWalkProgress& vertexCompleted)
{
    const auto count = walk.origins.size();
    if (count < 2) {
        walkSheetLines(walk, normal, guess, workers, vertexCompleted);
        return;
    }
    if (walk.cells < 1 || walk.steps.size() != count ||
        guess.size() != count * static_cast<std::size_t>(walk.cells))
        throw std::invalid_argument("Sheet growth edge and output dimensions disagree.");
    const auto spacing = arcLengths(walk.origins);
    for (std::size_t i = 1; i < count; ++i)
        if (!std::isfinite(spacing[i]) || spacing[i] - spacing[i - 1] < 1e-6)
            throw std::runtime_error("Sheet growth needs a nondegenerate starting edge.");

    Points front = walk.origins;
    Points headings = walk.steps;
    std::vector<float> lengths(count);
    for (std::size_t i = 0; i < count; ++i) {
        lengths[i] = static_cast<float>(cv::norm(walk.steps[i]));
        if (!std::isfinite(lengths[i]) || lengths[i] < 1e-6f)
            throw std::invalid_argument("Sheet growth requires a nonzero step along each edge vertex.");
    }
    Points predicted(count);
    for (int cell = 0; cell < walk.cells; ++cell) {
        SheetWalk step{0, 1, front, headings, walk.options};
        // Choose directions from the current edge, not from independent rays
        // launched at the beginning of the entire extension.
        for (std::size_t i = 0; i < count; ++i) {
            // A chord across a curved step can be shorter than its arc length.
            // Missing normals must not progressively shrink output spacing.
            step.steps[i] = unit(headings[i]) * lengths[i];
            if (normal) {
                const auto n = normal(front[i]);
                if (n && std::isfinite(cv::norm(*n)) && cv::norm(*n) > 0.5) {
                    auto forward = unit(across(front, i).cross(*n));
                    if (forward.dot(headings[i]) < 0) forward = -forward;
                    step.steps[i] = forward * lengths[i];
                }
            }
        }
        // Only this single step is independent. Join before redistributing
        // the new edge so subsequent steps see the same coherent front.
        walkSheetLines(step, normal, predicted, workers);
        const auto next = redistribute(predicted, spacing, normal, walk.options);
        for (std::size_t i = 0; i < count; ++i) {
            headings[i] = predicted[i] - front[i];
            guess[i * walk.cells + cell] = next[i];
            if (vertexCompleted) vertexCompleted();
        }
        front = next;
    }
}
} // namespace vc::sheet
