#include "vc/core/util/AffineTransform.hpp"

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

bool isValidSurfacePoint(const cv::Vec3f& point)
{
    return point[0] != -1.0f
        && std::isfinite(point[0])
        && std::isfinite(point[1])
        && std::isfinite(point[2]);
}

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
            if (!isValidSurfacePoint(point)) {
                continue;
            }

            const cv::Vec3f& left = points(row, col - 1);
            if (isValidSurfacePoint(left)) {
                const cv::Vec3d delta = cv::Vec3d(point) - cv::Vec3d(left);
                const double distance = cv::norm(delta);
                if (std::isfinite(distance) && distance > 0.0) {
                    horizontal.push_back(distance);
                }
            }

            const cv::Vec3f& up = points(row - 1, col);
            if (isValidSurfacePoint(up)) {
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

cv::Matx44d parseAffineTransformMatrix(const utils::Json& json)
{
    if (!json.contains("transformation_matrix")) {
        throw std::runtime_error("transform.json is missing transformation_matrix");
    }

    const auto& matrixJson = json.at("transformation_matrix");
    if (!matrixJson.is_array() || (matrixJson.size() != 3 && matrixJson.size() != 4)) {
        throw std::runtime_error("transformation_matrix must be 3x4 or 4x4");
    }

    cv::Matx44d matrix = cv::Matx44d::eye();
    for (int row = 0; row < static_cast<int>(matrixJson.size()); ++row) {
        const auto& rowJson = matrixJson.at(row);
        if (!rowJson.is_array() || rowJson.size() != 4) {
            throw std::runtime_error("each transformation_matrix row must have 4 values");
        }
        for (int col = 0; col < 4; ++col) {
            matrix(row, col) = rowJson.at(col).get_double();
        }
    }

    if (matrixJson.size() == 4) {
        if (std::abs(matrix(3, 0)) > 1e-12 ||
            std::abs(matrix(3, 1)) > 1e-12 ||
            std::abs(matrix(3, 2)) > 1e-12 ||
            std::abs(matrix(3, 3) - 1.0) > 1e-12) {
            throw std::runtime_error("transform.json bottom row must be [0, 0, 0, 1]");
        }
    }

    return matrix;
}

cv::Matx44d loadAffineTransformMatrix(const std::filesystem::path& path)
{
    if (path.empty()) {
        throw std::runtime_error("transform path is empty");
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("transform.json not found");
    }

    return parseAffineTransformMatrix(utils::Json::parse_file(path));
}

cv::Matx44d loadAffineTransformMatrixFromString(const std::string& text)
{
    if (text.empty()) {
        throw std::runtime_error("transform.json not found");
    }
    return parseAffineTransformMatrix(utils::Json::parse(text));
}

cv::Matx44d composeAffineTransform(const cv::Matx44d& first, const cv::Matx44d& second)
{
    return second * first;
}

std::optional<cv::Matx44d> tryInvertAffineTransformMatrix(const cv::Matx44d& matrix)
{
    const cv::Matx33d linear(matrix(0, 0), matrix(0, 1), matrix(0, 2),
                             matrix(1, 0), matrix(1, 1), matrix(1, 2),
                             matrix(2, 0), matrix(2, 1), matrix(2, 2));
    const double determinant = cv::determinant(linear);
    if (!std::isfinite(determinant) || std::abs(determinant) < std::numeric_limits<double>::epsilon()) {
        return std::nullopt;
    }

    cv::Mat linearMat(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            linearMat.at<double>(row, col) = matrix(row, col);
        }
    }

    cv::Mat linearInvMat;
    if (cv::invert(linearMat, linearInvMat, cv::DECOMP_SVD) <= 0.0) {
        return std::nullopt;
    }

    cv::Matx33d linearInv;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            linearInv(row, col) = linearInvMat.at<double>(row, col);
        }
    }

    const cv::Vec3d translation(matrix(0, 3), matrix(1, 3), matrix(2, 3));
    const cv::Vec3d inverseTranslation = -(linearInv * translation);

    cv::Matx44d inverted = cv::Matx44d::eye();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            inverted(row, col) = linearInv(row, col);
        }
    }
    inverted(0, 3) = inverseTranslation[0];
    inverted(1, 3) = inverseTranslation[1];
    inverted(2, 3) = inverseTranslation[2];
    return inverted;
}

cv::Matx44d invertAffineTransformMatrix(const cv::Matx44d& matrix)
{
    if (auto inverted = tryInvertAffineTransformMatrix(matrix)) {
        return *inverted;
    }
    throw std::runtime_error("transform is not invertible");
}

bool applyAffineTransform(const cv::Vec3d& point,
                          const cv::Matx44d& matrix,
                          cv::Vec3d& transformed)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        return false;
    }

    const cv::Vec4d homogeneous(point[0], point[1], point[2], 1.0);
    const cv::Vec4d result = matrix * homogeneous;
    if (!std::isfinite(result[0]) || !std::isfinite(result[1]) || !std::isfinite(result[2])) {
        return false;
    }

    transformed = cv::Vec3d(result[0], result[1], result[2]);
    return true;
}

cv::Vec3f applyAffineTransform(const cv::Vec3f& point,
                               const cv::Matx44d& matrix)
{
    if (point[0] == -1.0f) {
        return point;
    }

    cv::Vec3d transformed;
    if (!applyAffineTransform(cv::Vec3d(point), matrix, transformed) || !isRepresentableVec3f(transformed)) {
        return invalidSurfacePoint();
    }

    return cv::Vec3f(static_cast<float>(transformed[0]),
                     static_cast<float>(transformed[1]),
                     static_cast<float>(transformed[2]));
}

cv::Vec3f transformNormal(const cv::Vec3f& normal,
                          const cv::Matx44d& matrix)
{
    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
        return normal;
    }

    const cv::Matx33d linear(matrix(0, 0), matrix(0, 1), matrix(0, 2),
                             matrix(1, 0), matrix(1, 1), matrix(1, 2),
                             matrix(2, 0), matrix(2, 1), matrix(2, 2));
    const double determinant = cv::determinant(linear);
    if (!std::isfinite(determinant) || std::abs(determinant) < std::numeric_limits<double>::epsilon()) {
        return normal;
    }

    const cv::Matx33d inverseTranspose = linear.inv().t();
    const cv::Vec3d transformed(
        inverseTranspose(0, 0) * normal[0] + inverseTranspose(0, 1) * normal[1] + inverseTranspose(0, 2) * normal[2],
        inverseTranspose(1, 0) * normal[0] + inverseTranspose(1, 1) * normal[1] + inverseTranspose(1, 2) * normal[2],
        inverseTranspose(2, 0) * normal[0] + inverseTranspose(2, 1) * normal[1] + inverseTranspose(2, 2) * normal[2]);

    if (!std::isfinite(transformed[0]) || !std::isfinite(transformed[1]) || !std::isfinite(transformed[2])) {
        return normal;
    }

    const double lengthSquared = transformed.dot(transformed);
    if (!std::isfinite(lengthSquared) || lengthSquared <= 0.0) {
        return normal;
    }

    const double invLength = 1.0 / std::sqrt(lengthSquared);
    return cv::Vec3f(static_cast<float>(transformed[0] * invLength),
                     static_cast<float>(transformed[1] * invLength),
                     static_cast<float>(transformed[2] * invLength));
}

std::optional<double> affineUniformScaleFactor(const cv::Matx44d& matrix)
{
    cv::Mat linear(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            linear.at<double>(row, col) = matrix(row, col);
        }
    }

    cv::SVD svd(linear, cv::SVD::NO_UV);
    if (svd.w.rows < 3) {
        return std::nullopt;
    }

    const double s0 = svd.w.at<double>(0, 0);
    const double s1 = svd.w.at<double>(1, 0);
    const double s2 = svd.w.at<double>(2, 0);
    if (!(std::isfinite(s0) && std::isfinite(s1) && std::isfinite(s2))) {
        return std::nullopt;
    }
    if (s0 <= 0.0 || s1 <= 0.0 || s2 <= 0.0) {
        return std::nullopt;
    }

    const double mean = (s0 + s1 + s2) / 3.0;
    const double maxDeviation = std::max({std::abs(s0 - mean), std::abs(s1 - mean), std::abs(s2 - mean)});
    const double relativeDeviation = maxDeviation / mean;
    if (!std::isfinite(relativeDeviation) || relativeDeviation > 1e-4) {
        return std::nullopt;
    }

    return mean;
}

cv::Vec3f applyPreAffineScale(const cv::Vec3f& point, int scale)
{
    if (point[0] == -1.0f || scale == 1) {
        return point;
    }

    return point * static_cast<float>(scale);
}

void transformSurfacePoints(QuadSurface* surface,
                            int scale,
                            const std::optional<cv::Matx44d>& matrix)
{
    transformSurfacePoints(surface, static_cast<double>(scale), matrix, 1.0);
}

void transformSurfacePoints(QuadSurface* surface,
                            double scaleBeforeAffine,
                            const std::optional<cv::Matx44d>& matrix,
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
                if (!isValidSurfacePoint(point)) {
                    continue;
                }

                point *= static_cast<float>(scaleBeforeAffine);
                if (!isValidSurfacePoint(point)) {
                    point = invalidSurfacePoint();
                    continue;
                }

                if (matrix) {
                    point = applyAffineTransform(point, *matrix);
                }
                if (!isValidSurfacePoint(point)) {
                    point = invalidSurfacePoint();
                    continue;
                }

                point *= static_cast<float>(scaleAfterAffine);
                if (!isValidSurfacePoint(point)) {
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
            * (matrix ? affineUniformScaleFactor(*matrix).value_or(1.0) : 1.0);
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

std::shared_ptr<QuadSurface> cloneSurfaceForTransform(const std::shared_ptr<QuadSurface>& source)
{
    if (!source) {
        return nullptr;
    }

    auto clone = std::make_shared<QuadSurface>(source->rawPoints(), source->scale());
    clone->meta = source->meta.is_null() ? utils::Json::object() : source->meta;
    clone->id = source->id;
    clone->path = source->path;
    clone->setOverlappingIds(source->overlappingIds());

    for (const auto& channelName : source->channelNames()) {
        clone->setChannel(channelName, source->channel(channelName, SURF_CHANNEL_NORESIZE).clone());
    }

    return clone;
}

} // namespace vc::core::util
