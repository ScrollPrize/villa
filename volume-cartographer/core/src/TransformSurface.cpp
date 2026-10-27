#include "vc/core/util/TransformSurface.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vc::core::util {

namespace {

struct GridAxisSpacing {
    double x = std::numeric_limits<double>::quiet_NaN();
    double y = std::numeric_limits<double>::quiet_NaN();
    int x_count = 0;
    int y_count = 0;
};

bool isRepresentableFloat(double value)
{
    return std::isfinite(value)
        && value >= -static_cast<double>(std::numeric_limits<float>::max())
        && value <= static_cast<double>(std::numeric_limits<float>::max());
}

bool isRepresentableVec3f(const cv::Vec3d& point)
{
    return isRepresentableFloat(point[0])
        && isRepresentableFloat(point[1])
        && isRepresentableFloat(point[2]);
}

cv::Vec3f invalidSurfacePoint()
{
    return cv::Vec3f(-1.0f, -1.0f, -1.0f);
}

double medianValue(std::vector<double>& values)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    double median = *mid;
    if ((values.size() % 2) == 0) {
        const auto leftMid = std::max_element(values.begin(), mid);
        median = 0.5 * (median + *leftMid);
    }
    return median;
}

GridAxisSpacing measureGridAxisSpacing(const cv::Mat_<cv::Vec3f>& points)
{
    GridAxisSpacing spacing;
    if (points.rows < 2 || points.cols < 2) {
        return spacing;
    }

    int rowStart = static_cast<int>(points.rows * 0.1);
    int rowEnd = static_cast<int>(points.rows * 0.9);
    int colStart = static_cast<int>(points.cols * 0.1);
    int colEnd = static_cast<int>(points.cols * 0.9);
    int step = 4;

    if (points.rows < 20 || points.cols < 20) {
        rowStart = 1;
        rowEnd = points.rows;
        colStart = 1;
        colEnd = points.cols;
        step = 1;
    } else {
        rowStart = std::max(1, rowStart);
        colStart = std::max(1, colStart);
        rowEnd = std::max(rowStart + 1, rowEnd);
        colEnd = std::max(colStart + 1, colEnd);
    }

    std::vector<double> horizontal;
    std::vector<double> vertical;
    horizontal.reserve(static_cast<size_t>((rowEnd - rowStart) * (colEnd - colStart)) / 4 + 1);
    vertical.reserve(horizontal.capacity());

    for (int row = rowStart; row < rowEnd; row += step) {
        for (int col = colStart; col < colEnd; col += step) {
            const cv::Vec3f& point = points(row, col);
            if (!vc::geometry::validSurfacePoint(point)) {
                continue;
            }

            const cv::Vec3f& left = points(row, col - 1);
            if (vc::geometry::validSurfacePoint(left)) {
                const cv::Vec3d delta = cv::Vec3d(point) - cv::Vec3d(left);
                const double distance = cv::norm(delta);
                if (std::isfinite(distance) && distance > 0.0) {
                    horizontal.push_back(distance);
                }
            }

            const cv::Vec3f& up = points(row - 1, col);
            if (vc::geometry::validSurfacePoint(up)) {
                const cv::Vec3d delta = cv::Vec3d(point) - cv::Vec3d(up);
                const double distance = cv::norm(delta);
                if (std::isfinite(distance) && distance > 0.0) {
                    vertical.push_back(distance);
                }
            }
        }
    }

    spacing.x_count = static_cast<int>(horizontal.size());
    spacing.y_count = static_cast<int>(vertical.size());
    spacing.x = medianValue(horizontal);
    spacing.y = medianValue(vertical);
    return spacing;
}

} // namespace

void transformSurfacePoints(QuadSurface* surface,
                            double scaleBeforeAffine,
                            const std::optional<vcr::catalog::Affine>& matrix,
                            double scaleAfterAffine)
{
    if (!surface) {
        return;
    }

    if (auto* points = surface->rawPointsPtr()) {
        const cv::Vec2f originalScale = surface->scale();
        const GridAxisSpacing baselineSpacing = measureGridAxisSpacing(*points);

        for (int row = 0; row < points->rows; ++row) {
            for (int col = 0; col < points->cols; ++col) {
                auto& point = (*points)(row, col);
                if (!vc::geometry::validSurfacePoint(point)) {
                    continue;
                }

                point *= static_cast<float>(scaleBeforeAffine);
                if (!vc::geometry::validSurfacePoint(point)) {
                    point = invalidSurfacePoint();
                    continue;
                }

                if (matrix) {
                    const auto mapped = matrix->apply({point[0], point[1], point[2]});
                    point = isRepresentableVec3f({mapped[0], mapped[1], mapped[2]})
                        ? cv::Vec3f(static_cast<float>(mapped[0]), static_cast<float>(mapped[1]), static_cast<float>(mapped[2]))
                        : invalidSurfacePoint();
                }
                if (!vc::geometry::validSurfacePoint(point)) {
                    point = invalidSurfacePoint();
                    continue;
                }

                point *= static_cast<float>(scaleAfterAffine);
                if (!vc::geometry::validSurfacePoint(point)) {
                    point = invalidSurfacePoint();
                }
            }
        }

        const GridAxisSpacing transformedSpacing = measureGridAxisSpacing(*points);
        double spacingScaleX = std::numeric_limits<double>::quiet_NaN();
        double spacingScaleY = std::numeric_limits<double>::quiet_NaN();

        if (baselineSpacing.x_count > 0 && transformedSpacing.x_count > 0
            && std::isfinite(baselineSpacing.x) && baselineSpacing.x > 0.0
            && std::isfinite(transformedSpacing.x) && transformedSpacing.x > 0.0) {
            spacingScaleX = transformedSpacing.x / baselineSpacing.x;
        }
        if (baselineSpacing.y_count > 0 && transformedSpacing.y_count > 0
            && std::isfinite(baselineSpacing.y) && baselineSpacing.y > 0.0
            && std::isfinite(transformedSpacing.y) && transformedSpacing.y > 0.0) {
            spacingScaleY = transformedSpacing.y / baselineSpacing.y;
        }

        const double scaleFallback = scaleBeforeAffine * scaleAfterAffine
            * (matrix ? matrix->similarityScale(2e-4).value_or(1.0) : 1.0);
        if (!std::isfinite(spacingScaleX) || spacingScaleX <= 0.0) {
            spacingScaleX = scaleFallback;
        }
        if (!std::isfinite(spacingScaleY) || spacingScaleY <= 0.0) {
            spacingScaleY = scaleFallback;
        }

        if (std::isfinite(spacingScaleX) && spacingScaleX > 0.0
            && std::isfinite(spacingScaleY) && spacingScaleY > 0.0
            && (std::abs(spacingScaleX - 1.0) > 1e-4 || std::abs(spacingScaleY - 1.0) > 1e-4)) {
            surface->resample(static_cast<float>(spacingScaleX),
                              static_cast<float>(spacingScaleY),
                              cv::INTER_LINEAR);
            surface->_scale = originalScale;
            surface->invalidateCache();
        }
    }
}

void refreshTransformedSurfaceState(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    surface->invalidateCache();

    if (surface->meta.is_null() || !surface->meta.is_object()) {
        surface->meta = utils::Json::object();
    }

    const auto bbox = surface->bbox();
    {
        auto lo = utils::Json::array();
        lo.push_back(bbox.low[0]);
        lo.push_back(bbox.low[1]);
        lo.push_back(bbox.low[2]);
        auto hi = utils::Json::array();
        hi.push_back(bbox.high[0]);
        hi.push_back(bbox.high[1]);
        hi.push_back(bbox.high[2]);
        auto bb = utils::Json::array();
        bb.push_back(std::move(lo));
        bb.push_back(std::move(hi));
        surface->meta["bbox"] = std::move(bb);
    }
    {
        auto sc = utils::Json::array();
        sc.push_back(surface->scale()[0]);
        sc.push_back(surface->scale()[1]);
        surface->meta["scale"] = std::move(sc);
    }
}

} // namespace vc::core::util
