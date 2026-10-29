#include "SegmentationGrowth.hpp"

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <system_error>

#include "utils/Json.hpp"
#include <opencv2/core.hpp>
#include <omp.h>
#include <QLoggingCategory>
#include <QString>


#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/tracer/Tracer.hpp"
#include "vc_reader/zarr.hpp"
#include "vc/ui/VCCollection.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth)

namespace
{
class ScopedOmpThreadLimit
{
public:
    explicit ScopedOmpThreadLimit(int threadCount)
        : _previousThreadCount(std::max(1, omp_get_max_threads()))
    {
        omp_set_num_threads(std::max(1, threadCount));
    }

    ~ScopedOmpThreadLimit()
    {
        omp_set_num_threads(_previousThreadCount);
    }

    ScopedOmpThreadLimit(const ScopedOmpThreadLimit&) = delete;
    ScopedOmpThreadLimit& operator=(const ScopedOmpThreadLimit&) = delete;

private:
    int _previousThreadCount{1};
};

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
    case SegmentationGrowthDirection::Fill:
        return QStringLiteral("fill");
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
        if (entry.anchor2d.has_value()) {
            collection.setCollectionAnchor2d(id, entry.anchor2d);
        }

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

    auto [w, h, d] = volume->shapeXyz();
    cv::Vec3f volumeCenter(w * 0.5f, h * 0.5f, d * 0.5f);
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

utils::Json buildTracerParams(const SegmentationGrowthRequest& request)
{
    utils::Json params = utils::Json::object();
    params["rewind_gen"] = -1;
    params["grow_mode"] = directionToString(request.direction).toStdString();
    params["grow_steps"] = std::max(0, request.steps);
    params["grow_extra_rows"] = 0;
    params["grow_extra_cols"] = 0;

    if (request.direction == SegmentationGrowthDirection::Fill) {
        params["grow_max_extra_rows"] = 0;
        params["grow_max_extra_cols"] = 0;
        params["disable_grid_expansion"] = true;
    } else if (request.direction == SegmentationGrowthDirection::Left || request.direction == SegmentationGrowthDirection::Right) {
        params["grow_max_extra_cols"] = std::max(0, request.steps);
        params["grow_max_extra_rows"] = 0;
    } else if (request.direction == SegmentationGrowthDirection::Up || request.direction == SegmentationGrowthDirection::Down) {
        params["grow_max_extra_rows"] = std::max(0, request.steps);
        params["grow_max_extra_cols"] = 0;
    } else {
        params["grow_max_extra_rows"] = std::max(0, request.steps);
        params["grow_max_extra_cols"] = std::max(0, request.steps);
    }

    params["inpaint"] = request.inpaintOnly;

    bool allowUp = false;
    bool allowDown = false;
    bool allowLeft = false;
    bool allowRight = false;
    for (auto dir : request.allowedDirections) {
        switch (dir) {
        case SegmentationGrowthDirection::Up:
            allowUp = true;
            break;
        case SegmentationGrowthDirection::Down:
            allowDown = true;
            break;
        case SegmentationGrowthDirection::Left:
            allowLeft = true;
            break;
        case SegmentationGrowthDirection::Right:
            allowRight = true;
            break;
        case SegmentationGrowthDirection::All:
        case SegmentationGrowthDirection::Fill:
        default:
            allowUp = allowDown = allowLeft = allowRight = true;
            break;
        }
        if (allowUp && allowDown && allowLeft && allowRight) {
            break;
        }
    }

    const int allowedCount = static_cast<int>(allowUp) + static_cast<int>(allowDown) +
                             static_cast<int>(allowLeft) + static_cast<int>(allowRight);
    if (allowedCount > 0 && allowedCount < 4) {
        utils::Json allowedArr = utils::Json::array();
        if (allowDown) allowedArr.push_back(utils::Json("down"));
        if (allowRight) allowedArr.push_back(utils::Json("right"));
        if (allowUp) allowedArr.push_back(utils::Json("up"));
        if (allowLeft) allowedArr.push_back(utils::Json("left"));
        params["growth_directions"] = std::move(allowedArr);
    }
    if (!request.customParams.is_null()) {
        for (auto it = request.customParams.begin(); it != request.customParams.end(); ++it) {
            params[it.key()] = *it;
        }
    }
    return params;
}
} // namespace

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


TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context)
{
    TracerGrowthResult result;

    if (!context.resumeSurface || !context.volume) {
        result.error = QStringLiteral("Missing context for tracer growth");
        return result;
    }

    if (!ensureGenerationsChannel(context.resumeSurface)) {
        result.error = QStringLiteral("Segmentation surface lacks a generations channel");
        return result;
    }
    ensureNormalsInward(context.resumeSurface, context.volume);

    if (!context.cacheRoot.isEmpty()) {
        std::error_code ec;
        std::filesystem::create_directories(context.cacheRoot.toStdString(), ec);
        if (ec) {
            result.error = QStringLiteral("Failed to create cache directory: %1").arg(QString::fromStdString(ec.message()));
            return result;
        }
    }

    utils::Json params = buildTracerParams(request);

    int startGen = 0;
    {
        cv::Mat resumeGenerations = context.resumeSurface->channel("generations");
        if (!resumeGenerations.empty()) {
            double minVal = 0.0;
            double maxVal = 0.0;
            cv::minMaxLoc(resumeGenerations, &minVal, &maxVal);
            startGen = static_cast<int>(std::round(maxVal));
        }

        if (!context.resumeSurface->meta.is_null() && context.resumeSurface->meta.is_object()) {
            const auto& meta = context.resumeSurface->meta;
            if (meta.contains("max_gen")) {
                const auto& v = meta["max_gen"];
                if (v.is_number()) {
                    const int metaGen = static_cast<int>(std::round(v.get_double()));
                    startGen = std::max(startGen, metaGen);
                }
            }
        }

        if (startGen <= 0) {
            bool hasValidPoints = false;
            const auto* resumePoints = context.resumeSurface->rawPointsPtr();
            if (resumePoints && !resumePoints->empty()) {
                for (int row = 0; row < resumePoints->rows && !hasValidPoints; ++row) {
                    for (int col = 0; col < resumePoints->cols; ++col) {
                        const cv::Vec3f& point = resumePoints->operator()(row, col);
                        if (point[0] != -1.0f) {
                            hasValidPoints = true;
                            break;
                        }
                    }
                }
            }
            if (hasValidPoints) {
                startGen = 1;
                qCWarning(lcSegGrowth) << "Resume surface missing generation metadata; defaulting start generation to 1.";
            }
        }
    }

    const int requestedSteps = std::max(request.steps, 0);
    int targetGenerations = startGen;

    if (requestedSteps > 0) {
        targetGenerations = startGen + requestedSteps;
    }

    if (targetGenerations < startGen) {
        targetGenerations = startGen;
    }
    if (targetGenerations <= 0) {
        targetGenerations = startGen;
    }

    params["generations"] = targetGenerations;
    if (!params.contains("rewind_gen")) {
        params["rewind_gen"] = -1;
    }
    params["cache_root"] = context.cacheRoot.toStdString();
    if (!context.normalGridPath.isEmpty()) {
        params["normal_grid_path"] = context.normalGridPath.toStdString();
    }

    if (request.correctionsZRange) {
        int zMin = std::max(0, request.correctionsZRange->first);
        int zMax = std::max(zMin, request.correctionsZRange->second);
        params["z_min"] = zMin;
        params["z_max"] = zMax;
    }

    const cv::Vec3f origin(0.0f, 0.0f, 0.0f);

    VCCollection correctionCollection;
    if (!request.corrections.empty()) {
        populateCorrectionsCollection(request.corrections, correctionCollection);
    }

    try {
        qCInfo(lcSegGrowth) << "Calling tracer()";
        qCInfo(lcSegGrowth) << "  cacheRoot:" << context.cacheRoot;
        qCInfo(lcSegGrowth) << "  voxelSize:" << context.voxelSize;
        qCInfo(lcSegGrowth) << "  resumeSurface:" << (context.resumeSurface ? context.resumeSurface->id.c_str() : "<null>");
        const auto collectionCount = correctionCollection.getAllCollections().size();
        qCInfo(lcSegGrowth) << "  corrections collections:" << collectionCount;
        if (request.correctionsZRange) {
            qCInfo(lcSegGrowth) << "  corrections z-range:" << request.correctionsZRange->first << request.correctionsZRange->second;
        }
        qCInfo(lcSegGrowth) << "  params:" << QString::fromStdString(params.dump());

        std::optional<ScopedOmpThreadLimit> regularTracerOmpLimit;
        if (request.method == SegmentationGrowthMethod::Tracer) {
            regularTracerOmpLimit.emplace(1);
            qCInfo(lcSegGrowth) << "  regular tracer OpenMP threads forced to 1";
        }

        // The tracer samples the working level point by point from its memory cache
        // (the tracer's cache_size parameter, as for vc_grow_seg_from_seed).
        auto options = context.volume->sourceOpenOptions();
        options.memoryCache = std::make_shared<vcr::MemoryCache>(size_t(params.value("cache_size", 1e9)));
        auto level = vcr::zarr::openPyramidLevel(
            context.volume->sourceLocation(), context.level + context.volume->baseScaleLevel(), options);
        QuadSurface* surface = tracer(*level.array,
                                      1.0f,
                                      origin,
                                      params,
                                      context.cacheRoot.toStdString(),
                                      static_cast<float>(context.voxelSize),
                                      context.resumeSurface,
                                      std::filesystem::path(),
                                      utils::Json{},
                                      correctionCollection,
                                      context.allowedGrowthMask.empty() ? nullptr : &context.allowedGrowthMask);

        // Note: the mask channel is preserved inside the tracer

        result.surface = surface;
        result.statusMessage = QStringLiteral("Tracer growth completed");
    } catch (const std::exception& ex) {
        result.error = QStringLiteral("Tracer growth failed: %1").arg(ex.what());
    }

    return result;
}
