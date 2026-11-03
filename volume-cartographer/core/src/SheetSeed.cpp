#include "vc/core/util/SheetSeed.hpp"

#include <cmath>
#include <mutex>
#include <stdexcept>

namespace vc::sheet {
namespace {
cv::Vec3f unit(const cv::Vec3f& v)
{
    const double length = cv::norm(v);
    if (!std::isfinite(length) || length < 1e-6)
        throw std::runtime_error("Sheet seed has a degenerate tangent or normal.");
    return v / static_cast<float>(length);
}
cv::Vec3f requireNormal(const SheetNormalFn& normal, const cv::Vec3f& point)
{
    const auto n = normal ? normal(point) : std::nullopt;
    if (!n) throw std::runtime_error("No usable CT sheet normal at the seed or central column.");
    return unit(*n);
}
}

cv::Mat_<cv::Vec3f> growSheetSeed(const cv::Vec3f& seed,
                                const SheetNormalFn& normal,
                                const SeedOptions& options)
{
    return growSheetSeed(seed, normal, options, {});
}

cv::Mat_<cv::Vec3f> growSheetSeed(const cv::Vec3f& seed,
                                const SheetNormalFn& normal,
                                const SeedOptions& options,
                                const SeedProgress& progress)
{
    if (options.generations < 1 || options.generations > 1024 ||
        !std::isfinite(options.spacing) || options.spacing < 0.1 || options.spacing > 1000 ||
        !std::isfinite(options.walk.maxStep) || options.walk.maxStep < 0 || options.walk.maxStep > 1000 ||
        options.workers < 1)
        throw std::invalid_argument("Sheet seeds require 1–1024 generations, 0.1–1000 voxel spacing, positive workers, and a 0–1000 voxel integration step.");
    if (!std::isfinite(cv::norm(seed)))
        throw std::invalid_argument("Sheet seed coordinates must be finite.");

    // A single counter per stage keeps parallel completion reports ordered.
    std::mutex progressMutex;
    std::size_t completed = 0;
    const auto beginStage = [&](SeedStage stage, std::size_t total) {
        completed = 0;
        if (progress) progress(stage, 0, total);
    };
    const auto advance = [&](SeedStage stage, std::size_t total) {
        if (!progress) return;
        std::lock_guard lock(progressMutex);
        progress(stage, ++completed, total);
    };
    beginStage(SeedStage::SeedNormal, 1);
    auto n = requireNormal(normal, seed);
    advance(SeedStage::SeedNormal, 1);
    // Fix the otherwise unsigned eigenvector so grid orientation is repeatable.
    int dominant = 0;
    for (int axis = 1; axis < 3; ++axis)
        if (std::abs(n[axis]) > std::abs(n[dominant])) dominant = axis;
    if (n[dominant] < 0) n = -n;
    const cv::Vec3f reference = std::abs(n[2]) < 0.9f
        ? cv::Vec3f(0, 0, 1) : cv::Vec3f(0, 1, 0);
    const auto v = unit(reference - n * reference.dot(n));
    const auto u = unit(v.cross(n));
    const float spacing = static_cast<float>(options.spacing);
    const int radius = options.generations;
    const int size = 2 * radius + 1;
    cv::Mat_<cv::Vec3f> points(size, size);
    points(radius, radius) = seed;

    SheetWalk spine{0, radius, {seed, seed}, {-v * spacing, v * spacing}, options.walk};
    std::vector<cv::Vec3f> column(2 * radius);
    beginStage(SeedStage::CentralColumn, column.size());
    SheetWalkProgress vertexCompleted;
    if (progress) vertexCompleted = [&] { advance(SeedStage::CentralColumn, column.size()); };
    walkSheetLines(spine, normal, column, options.workers, vertexCompleted);
    for (int k = 0; k < radius; ++k) {
        points(radius - k - 1, radius) = column[k];
        points(radius + k + 1, radius) = column[radius + k];
    }

    std::vector<cv::Vec3f> rowDirections(size);
    rowDirections[radius] = u;
    beginStage(SeedStage::RowDirections, 2 * radius);
    for (int sign : {-1, 1}) {
        auto previousNormal = n;
        for (int k = 1; k <= radius; ++k) {
            const int row = radius + sign * k;
            auto localNormal = requireNormal(normal, points(row, radius));
            if (localNormal.dot(previousNormal) < 0) localNormal = -localNormal;
            const auto alongColumn = sign * (points(row, radius) - points(row - sign, radius));
            rowDirections[row] = unit(alongColumn.cross(localNormal));
            previousNormal = localNormal;
            advance(SeedStage::RowDirections, 2 * radius);
        }
    }
    SheetWalk rows;
    rows.cells = radius;
    rows.options = options.walk;
    for (int row = 0; row < size; ++row) {
        for (int sign : {-1, 1}) {
            rows.origins.push_back(points(row, radius));
            rows.steps.push_back(rowDirections[row] * (sign * spacing));
        }
    }
    std::vector<cv::Vec3f> grown(rows.origins.size() * radius);
    beginStage(SeedStage::Rows, grown.size());
    if (progress) vertexCompleted = [&] { advance(SeedStage::Rows, grown.size()); };
    walkSheetLines(rows, normal, grown, options.workers, vertexCompleted);
    for (int row = 0; row < size; ++row) {
        for (int k = 0; k < radius; ++k) {
            points(row, radius - k - 1) = grown[(2 * row) * radius + k];
            points(row, radius + k + 1) = grown[(2 * row + 1) * radius + k];
        }
    }
    return points;
}
} // namespace vc::sheet
