#include "SegmentationGrowth.hpp"

#include <filesystem>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/tracer/Tracer.hpp"

namespace
{
cv::Mat_<uint16_t> extendGenerations(const cv::Mat& generations,
                                     SegmentationGrowthDirection direction,
                                     int steps)
{
    if (generations.empty() || steps <= 0) {
        return generations.clone();
    }

    const int rows = generations.rows;
    const int cols = generations.cols;

    cv::Mat_<uint16_t> extended;
    switch (direction) {
    case SegmentationGrowthDirection::Left:
        extended = cv::Mat_<uint16_t>::zeros(rows, cols + steps);
        generations.copyTo(extended(cv::Rect(steps, 0, cols, rows)));
        break;
    case SegmentationGrowthDirection::Right:
        extended = cv::Mat_<uint16_t>::zeros(rows, cols + steps);
        generations.copyTo(extended(cv::Rect(0, 0, cols, rows)));
        break;
    case SegmentationGrowthDirection::Up:
        extended = cv::Mat_<uint16_t>::zeros(rows + steps, cols);
        generations.copyTo(extended(cv::Rect(0, steps, cols, rows)));
        break;
    case SegmentationGrowthDirection::Down:
        extended = cv::Mat_<uint16_t>::zeros(rows + steps, cols);
        generations.copyTo(extended(cv::Rect(0, 0, cols, rows)));
        break;
    case SegmentationGrowthDirection::All:
    default:
        extended = generations.clone();
        break;
    }

    return extended;
}

void writeInterpolatedColumns(const cv::Mat_<cv::Vec3f>& source,
                              cv::Mat_<cv::Vec3f>& target,
                              int steps,
                              bool insertLeft)
{
    CV_Assert(source.cols >= 2);
    const int rows = source.rows;
    const int cols = source.cols;

    const int offset = insertLeft ? steps : 0;
    const cv::Rect dstRect(offset, 0, cols, rows);
    source.copyTo(target(dstRect));

    for (int s = 0; s < steps; ++s) {
        const int dstCol = insertLeft ? steps - 1 - s : cols + s;
        const float scalar = static_cast<float>(s + 1);
        for (int r = 0; r < rows; ++r) {
            const cv::Vec3f base = source(r, insertLeft ? 0 : cols - 1);
            const cv::Vec3f neighbor = source(r, insertLeft ? std::min(1, cols - 1) : std::max(cols - 2, 0));
            const cv::Vec3f delta = neighbor - base;
            target(r, dstCol) = base + delta * scalar;
        }
    }
}

void writeInterpolatedRows(const cv::Mat_<cv::Vec3f>& source,
                           cv::Mat_<cv::Vec3f>& target,
                           int steps,
                           bool insertTop)
{
    CV_Assert(source.rows >= 2);
    const int rows = source.rows;
    const int cols = source.cols;

    const int offset = insertTop ? steps : 0;
    const cv::Rect dstRect(0, offset, cols, rows);
    source.copyTo(target(dstRect));

    for (int s = 0; s < steps; ++s) {
        const int dstRow = insertTop ? steps - 1 - s : rows + s;
        const float scalar = static_cast<float>(s + 1);
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f base = source(insertTop ? 0 : rows - 1, c);
            const cv::Vec3f neighbor = source(insertTop ? std::min(1, rows - 1) : std::max(rows - 2, 0), c);
            const cv::Vec3f delta = neighbor - base;
            target(dstRow, c) = base + delta * scalar;
        }
    }
}

bool ensureGenerationsChannel(QuadSurface* surface)
{
    if (!surface) {
        return false;
    }
    cv::Mat generations = surface->channel("generations");
    if (!generations.empty()) {
        return true;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return false;
    }

    cv::Mat_<uint16_t> seeded(points->rows, points->cols, static_cast<uint16_t>(1));
    surface->setChannel("generations", seeded);
    return true;
}

QString directionToString(SegmentationGrowthDirection direction)
{
    switch (direction) {
    case SegmentationGrowthDirection::Up:
        return QStringLiteral("up");
    case SegmentationGrowthDirection::Down:
        return QStringLiteral("down");
    case SegmentationGrowthDirection::Left:
        return QStringLiteral("left");
    case SegmentationGrowthDirection::Right:
        return QStringLiteral("right");
    case SegmentationGrowthDirection::All:
    default:
        return QStringLiteral("all");
    }
}

void ensureNormalsInward(QuadSurface* surface, const Volume* volume)
{
    if (!surface || !volume) {
        return;
    }
    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int centerRow = std::clamp(points->rows / 2, 0, points->rows - 1);
    const int centerCol = std::clamp(points->cols / 2, 0, points->cols - 1);
    const int nextCol = std::clamp(centerCol + 1, 0, points->cols - 1);
    const int nextRow = std::clamp(centerRow + 1, 0, points->rows - 1);

    const cv::Vec3f p = (*points)(centerRow, centerCol);
    const cv::Vec3f px = (*points)(centerRow, nextCol);
    const cv::Vec3f py = (*points)(nextRow, centerCol);

    cv::Vec3f normal = (px - p).cross(py - p);
    if (cv::norm(normal) < 1e-5f) {
        return;
    }
    cv::normalize(normal, normal);

    cv::Vec3f volumeCenter(static_cast<float>(volume->sliceWidth()) * 0.5f,
                           static_cast<float>(volume->sliceHeight()) * 0.5f,
                           static_cast<float>(volume->numSlices()) * 0.5f);
    cv::Vec3f toCenter = volumeCenter - p;
    toCenter[2] = 0.0f;

    if (normal.dot(toCenter) >= 0.0f) {
        return; // already inward
    }

    cv::Mat normals = surface->channel("normals");
    if (!normals.empty()) {
        cv::Mat_<cv::Vec3f> adjusted = normals;
        adjusted *= -1.0f;
        surface->setChannel("normals", adjusted);
    }
}

nlohmann::json buildTracerParams(const SegmentationGrowthRequest& request)
{
    nlohmann::json params;
    params["rewind_gen"] = -1;
    params["grow_mode"] = directionToString(request.direction).toStdString();
    params["grow_steps"] = std::max(0, request.steps);

    if (request.direction == SegmentationGrowthDirection::Left || request.direction == SegmentationGrowthDirection::Right) {
        params["grow_extra_cols"] = std::max(0, request.steps);
        params["grow_extra_rows"] = 0;
    } else if (request.direction == SegmentationGrowthDirection::Up || request.direction == SegmentationGrowthDirection::Down) {
        params["grow_extra_rows"] = std::max(0, request.steps);
        params["grow_extra_cols"] = 0;
    } else {
        params["grow_extra_rows"] = std::max(0, request.steps);
        params["grow_extra_cols"] = std::max(0, request.steps);
    }

    return params;
}
} // namespace

bool growSurfaceByInterpolation(QuadSurface* surface,
                                SegmentationGrowthDirection direction,
                                int steps,
                                QString* error)
{
    if (!surface) {
        if (error) {
            *error = QStringLiteral("No segmentation surface available");
        }
        return false;
    }
    if (steps <= 0) {
        if (error) {
            *error = QStringLiteral("Steps must be greater than zero for interpolation growth");
        }
        return false;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        if (error) {
            *error = QStringLiteral("Segmentation surface has no points to grow");
        }
        return false;
    }

    if ((direction == SegmentationGrowthDirection::Left || direction == SegmentationGrowthDirection::Right) && points->cols < 2) {
        if (error) {
            *error = QStringLiteral("Need at least two columns to interpolate additional columns");
        }
        return false;
    }

    if ((direction == SegmentationGrowthDirection::Up || direction == SegmentationGrowthDirection::Down) && points->rows < 2) {
        if (error) {
            *error = QStringLiteral("Need at least two rows to interpolate additional rows");
        }
        return false;
    }

    cv::Mat_<cv::Vec3f> grown;

    switch (direction) {
    case SegmentationGrowthDirection::Left:
    case SegmentationGrowthDirection::Right:
        grown = cv::Mat_<cv::Vec3f>(points->rows, points->cols + steps, cv::Vec3f(-1.0f, -1.0f, -1.0f));
        writeInterpolatedColumns(*points, grown, steps, direction == SegmentationGrowthDirection::Left);
        break;
    case SegmentationGrowthDirection::Up:
    case SegmentationGrowthDirection::Down:
        grown = cv::Mat_<cv::Vec3f>(points->rows + steps, points->cols, cv::Vec3f(-1.0f, -1.0f, -1.0f));
        writeInterpolatedRows(*points, grown, steps, direction == SegmentationGrowthDirection::Up);
        break;
    case SegmentationGrowthDirection::All:
        if (error) {
            *error = QStringLiteral("Interpolation growth does not support the 'All' direction");
        }
        return false;
    }

    *points = grown;
    surface->invalidateCache();

    cv::Mat existingGenerations = surface->channel("generations");
    if (!existingGenerations.empty()) {
        cv::Mat_<uint16_t> extended = extendGenerations(existingGenerations, direction, steps);
        surface->setChannel("generations", extended);
    }

    return true;
}

TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context)
{
    TracerGrowthResult result;

    if (!context.resumeSurface || !context.volume || !context.cache) {
        result.error = QStringLiteral("Missing context for tracer growth");
        return result;
    }

    if (!ensureGenerationsChannel(context.resumeSurface)) {
        result.error = QStringLiteral("Segmentation surface lacks a generations channel");
        return result;
    }

    ensureNormalsInward(context.resumeSurface, context.volume);

    z5::Dataset* dataset = context.volume->zarrDataset(0);
    if (!dataset) {
        result.error = QStringLiteral("Unable to access primary volume dataset");
        return result;
    }

    if (!context.cacheRoot.isEmpty()) {
        std::error_code ec;
        std::filesystem::create_directories(context.cacheRoot.toStdString(), ec);
        if (ec) {
            result.error = QStringLiteral("Failed to create cache directory: %1").arg(QString::fromStdString(ec.message()));
            return result;
        }
    }

    nlohmann::json params = buildTracerParams(request);
    params["cache_root"] = context.cacheRoot.toStdString();

    const cv::Vec3f origin(0.0f, 0.0f, 0.0f);

    try {
        QuadSurface* surface = tracer(dataset,
                                      1.0f,
                                      context.cache,
                                      origin,
                                      params,
                                      context.cacheRoot.toStdString(),
                                      static_cast<float>(context.voxelSize),
                                      {},
                                      context.resumeSurface,
                                      std::filesystem::path(),
                                      nlohmann::json{},
                                      VCCollection());
        result.surface = surface;
        result.statusMessage = QStringLiteral("Tracer growth completed");
    } catch (const std::exception& ex) {
        result.error = QStringLiteral("Tracer growth failed: %1").arg(ex.what());
    }

    return result;
}
