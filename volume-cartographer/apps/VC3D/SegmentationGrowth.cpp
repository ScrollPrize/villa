#include "SegmentationGrowth.hpp"

#include <filesystem>
#include <cmath>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/tracer/Tracer.hpp"
#include "vc/ui/VCCollection.hpp"

namespace
{
bool isInvalidPoint(const cv::Vec3f& p)
{
    return !std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
           (p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f);
}

double triangleAreaUm2(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c)
{
    if (isInvalidPoint(a) || isInvalidPoint(b) || isInvalidPoint(c)) {
        return 0.0;
    }
    const cv::Vec3f ab = b - a;
    const cv::Vec3f ac = c - a;
    const cv::Vec3f crossVec = ab.cross(ac);
    return 0.5 * static_cast<double>(cv::norm(crossVec));
}

double computeSurfaceAreaUm2(const cv::Mat_<cv::Vec3f>& points)
{
    if (points.empty() || points.rows < 2 || points.cols < 2) {
        return 0.0;
    }

    double totalArea = 0.0;
    for (int y = 0; y < points.rows - 1; ++y) {
        for (int x = 0; x < points.cols - 1; ++x) {
            const cv::Vec3f& p00 = points(y, x);
            const cv::Vec3f& p10 = points(y, x + 1);
            const cv::Vec3f& p01 = points(y + 1, x);
            const cv::Vec3f& p11 = points(y + 1, x + 1);

            totalArea += triangleAreaUm2(p00, p10, p01);
            totalArea += triangleAreaUm2(p11, p01, p10);
        }
    }

    return totalArea;
}

void ensureMetaObject(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    if (surface->meta && surface->meta->is_object()) {
        return;
    }
    if (surface->meta) {
        delete surface->meta;
    }
    surface->meta = new nlohmann::json(nlohmann::json::object());
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

void populateCorrectionsCollection(const SegmentationCorrectionsPayload& payload, VCCollection& collection)
{
    for (const auto& entry : payload.collections) {
        uint64_t id = collection.addCollection(entry.name);
        collection.setCollectionMetadata(id, entry.metadata);
        collection.setCollectionColor(id, entry.color);

        for (const auto& point : entry.points) {
            ColPoint added = collection.addPoint(entry.name, point.p);
            if (!std::isnan(point.winding_annotation)) {
                added.winding_annotation = point.winding_annotation;
                collection.updatePoint(added);
            }
        }
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

    int startGen = 0;
    if (context.resumeSurface) {
        cv::Mat resumeGenerations = context.resumeSurface->channel("generations");
        if (!resumeGenerations.empty()) {
            double minVal = 0.0;
            double maxVal = 0.0;
            cv::minMaxLoc(resumeGenerations, &minVal, &maxVal);
            startGen = static_cast<int>(std::round(maxVal));
        }

        if (context.resumeSurface->meta && context.resumeSurface->meta->is_object()) {
            const auto& meta = *context.resumeSurface->meta;
            auto it = meta.find("max_gen");
            if (it != meta.end() && it->is_number()) {
                const int metaGen = static_cast<int>(std::round(it->get<double>()));
                startGen = std::max(startGen, metaGen);
            }
        }
    }

    const int requestedSteps = std::max(request.steps, 0);
    int targetGenerations = startGen;

    if (requestedSteps > 0) {
        targetGenerations = startGen + requestedSteps;
    } else if (!context.resumeSurface) {
        targetGenerations = std::max(startGen + 1, 1);
    }

    if (targetGenerations < startGen) {
        targetGenerations = startGen;
    }
    if (targetGenerations <= 0) {
        targetGenerations = startGen;
    }

    params["generations"] = targetGenerations;
    params["rewind_gen"] = startGen -1;
    params["cache_root"] = context.cacheRoot.toStdString();
    if (!context.normalGridPath.isEmpty()) {
        params["normal_grid_path"] = context.normalGridPath.toStdString();
    }

    const cv::Vec3f origin(0.0f, 0.0f, 0.0f);

    VCCollection correctionCollection;
    if (!request.corrections.empty()) {
        populateCorrectionsCollection(request.corrections, correctionCollection);
    }

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
                                      correctionCollection);
        result.surface = surface;
        result.statusMessage = QStringLiteral("Tracer growth completed");
    } catch (const std::exception& ex) {
        result.error = QStringLiteral("Tracer growth failed: %1").arg(ex.what());
    }

    return result;
}

void updateSegmentationSurfaceMetadata(QuadSurface* surface,
                                       double voxelSize)
{
    if (!surface) {
        return;
    }

    ensureMetaObject(surface);

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (points && !points->empty()) {
        const double areaUm2 = computeSurfaceAreaUm2(*points);
        (*surface->meta)["area_cm2"] = areaUm2 * 1e-8;

        if (voxelSize > 0.0) {
            const double voxelAreaUm2 = voxelSize * voxelSize;
            if (voxelAreaUm2 > 0.0) {
                (*surface->meta)["area_vx2"] = areaUm2 / voxelAreaUm2;
            }
        }
    }

    cv::Mat generations = surface->channel("generations");
    if (!generations.empty()) {
        double minGen = 0.0;
        double maxGen = 0.0;
        cv::minMaxLoc(generations, &minGen, &maxGen);
        (*surface->meta)["max_gen"] = static_cast<int>(std::round(maxGen));
    }

    (*surface->meta)["date_last_modified"] = get_surface_time_str();
}
