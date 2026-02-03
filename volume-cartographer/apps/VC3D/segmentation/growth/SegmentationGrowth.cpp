#include "SegmentationGrowth.hpp"

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <system_error>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <QLoggingCategory>
#include <QString>

#include "z5/factory.hxx"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/tracer/Tracer.hpp"
#include "vc/ui/VCCollection.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth)

namespace
{void createRotatingBackup(QuadSurface* surface, const std::filesystem::path& surfacePath, int maxBackups = 10)
{
    if (!surface) {
        return;
    }

    qCInfo(lcSegGrowth) << "Creating backup for:" << QString::fromStdString(surfacePath.string());

    try {
        // Create a rotating backup snapshot
        // This handles path normalization, rotation, and file copying automatically
        surface->saveSnapshot(maxBackups);
        qCInfo(lcSegGrowth) << "Backup creation complete";
    } catch (const std::exception& e) {
        qCWarning(lcSegGrowth) << "Failed to create backup:" << e.what();
    }
}

void ensureMetaObject(QuadSurface* surface)
{
    if (!surface) {
        return;
    }
    if (surface->meta && surface->meta->is_object()) {
        return;
    }
    surface->meta = std::make_unique<nlohmann::json>(nlohmann::json::object());
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

void preserveApprovalMask(QuadSurface* oldSurface, QuadSurface* newSurface)
{
    if (!oldSurface || !newSurface) {
        return;
    }

    // Load old approval mask without auto-resize
    cv::Mat old_approval = oldSurface->channel("approval", SURF_CHANNEL_NORESIZE);
    if (old_approval.empty()) {
        return;  // No approval mask to preserve
    }

    // Get new surface dimensions
    const cv::Mat_<cv::Vec3f>* new_points = newSurface->rawPointsPtr();
    if (!new_points || new_points->empty()) {
        return;
    }

    cv::Size new_size = new_points->size();

    // Create new approval mask with same type as old mask
    // Approval masks can be 1-channel (legacy) or 3-channel (RGB)
    cv::Mat new_approval;
    if (old_approval.channels() == 3) {
        new_approval = cv::Mat(new_size, CV_8UC3, cv::Scalar(0, 0, 0));
    } else {
        new_approval = cv::Mat(new_size, CV_8UC1, cv::Scalar(0));
    }

    // Copy old approval values to same grid positions
    // Grid expansion preserves old point indices, so old[r,c] == new[r,c]
    int copy_rows = std::min(old_approval.rows, new_approval.rows);
    int copy_cols = std::min(old_approval.cols, new_approval.cols);

    if (copy_rows > 0 && copy_cols > 0) {
        cv::Rect src_roi(0, 0, copy_cols, copy_rows);
        cv::Rect dst_roi(0, 0, copy_cols, copy_rows);
        old_approval(src_roi).copyTo(new_approval(dst_roi));

        qCInfo(lcSegGrowth) << "Preserved approval mask from"
                            << old_approval.cols << "x" << old_approval.rows
                            << "to" << new_approval.cols << "x" << new_approval.rows
                            << "(channels:" << old_approval.channels() << ")";
    }

    // Set preserved approval mask on new surface
    newSurface->setChannel("approval", new_approval);
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

bool appendDirectionField(const SegmentationDirectionFieldConfig& config,
                          ChunkCache<uint8_t>* cache,
                          const QString& cacheRoot,
                          std::vector<DirectionField>& out,
                          QString& error)
{
    if (!cache) {
        error = QStringLiteral("Direction field loading failed: chunk cache unavailable");
        return false;
    }

    if (!config.isValid()) {
        return true;
    }

    const QString path = config.path.trimmed();
    if (path.isEmpty()) {
        return true;
    }

    const std::string zarrPath = path.toStdString();
    std::error_code fsError;
    if (!std::filesystem::exists(zarrPath, fsError)) {
        const QString reason = fsError ? QString::fromStdString(fsError.message()) : QString();
        error = reason.isEmpty()
            ? QStringLiteral("Direction field directory does not exist: %1").arg(path)
            : QStringLiteral("Direction field directory error (%1): %2").arg(path, reason);
        return false;
    }

    try {
        z5::filesystem::handle::Group group(zarrPath, z5::FileMode::FileMode::r);
        const int scaleLevel = std::clamp(config.scale, 0, 5);

        std::vector<std::unique_ptr<z5::Dataset>> datasets;
        datasets.reserve(3);
        for (char axis : std::string("xyz")) {
            z5::filesystem::handle::Group axisGroup(group, std::string(1, axis));
            z5::filesystem::handle::Dataset datasetHandle(axisGroup, std::to_string(scaleLevel), ".");
            datasets.push_back(z5::filesystem::openDataset(datasetHandle));
        }

        const float scaleFactor = std::pow(2.0f, -static_cast<float>(scaleLevel));
        const std::string uniqueId = std::to_string(std::hash<std::string>{}(zarrPath + std::to_string(scaleLevel)));
        const std::string cacheRootStr = cacheRoot.toStdString();

        const float weight = static_cast<float>(std::clamp(config.weight, 0.0, 10.0));

        out.emplace_back(segmentationDirectionFieldOrientationKey(config.orientation).toStdString(),
                         std::make_unique<Chunked3dVec3fFromUint8>(std::move(datasets),
                                                                   scaleFactor,
                                                                   cache,
                                                                   cacheRootStr,
                                                                   uniqueId),
                         std::unique_ptr<Chunked3dFloatFromUint8>(),
                         weight);
    } catch (const std::exception& ex) {
        error = QStringLiteral("Failed to load direction field at %1: %2").arg(path, QString::fromStdString(ex.what()));
        return false;
    } catch (...) {
        error = QStringLiteral("Failed to load direction field at %1: unknown error").arg(path);
        return false;
    }

    return true;
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

    auto [w, h, d] = volume->shape();
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
        std::vector<std::string> allowedStrings;
        if (allowDown) allowedStrings.emplace_back("down");
        if (allowRight) allowedStrings.emplace_back("right");
        if (allowUp) allowedStrings.emplace_back("up");
        if (allowLeft) allowedStrings.emplace_back("left");
        params["growth_directions"] = allowedStrings;
    }
    if (request.customParams) {
        for (auto it = request.customParams->begin(); it != request.customParams->end(); ++it) {
            params[it.key()] = it.value();
        }
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
    int rewindGen = -1;
    if (startGen > 1) {
        rewindGen = startGen - 1;
    }
    params["rewind_gen"] = rewindGen;
    params["cache_root"] = context.cacheRoot.toStdString();
    if (!context.normalGridPath.isEmpty()) {
        params["normal_grid_path"] = context.normalGridPath.toStdString();
    }
    if (!context.normal3dZarrPath.isEmpty()) {
        params["normal3d_zarr_path"] = context.normal3dZarrPath.toStdString();
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

    std::vector<DirectionField> directionFields;
    for (const auto& config : request.directionFields) {
        if (!config.isValid()) {
            continue;
        }

        QString loadError;
        if (!appendDirectionField(config, context.cache, context.cacheRoot, directionFields, loadError)) {
            result.error = loadError;
            return result;
        }
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
        if (!directionFields.empty()) {
            int idx = 0;
            for (const auto& config : request.directionFields) {
                if (!config.isValid()) {
                    continue;
                }
                qCInfo(lcSegGrowth)
                    << "  direction field[#" << idx << "] path:" << config.path
                    << "orientation:" << segmentationDirectionFieldOrientationKey(config.orientation)
                    << "scale:" << config.scale
                    << "weight:" << config.weight;
                ++idx;
            }
        }
        qCInfo(lcSegGrowth) << "  params:" << QString::fromStdString(params.dump());
        std::filesystem::path surface_path = context.resumeSurface->path;
        createRotatingBackup(context.resumeSurface, surface_path);
        QuadSurface* surface = tracer(dataset,
                                      1.0f,
                                      context.cache,
                                      origin,
                                      params,
                                      context.cacheRoot.toStdString(),
                                      static_cast<float>(context.voxelSize),
                                      directionFields,
                                      context.resumeSurface,
                                      std::filesystem::path(),
                                      nlohmann::json{},
                                      correctionCollection);

        // Note: approval and mask channels are preserved inside the tracer

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

    const double previousAreaVx2 = vc::json::number_or(surface->meta.get(), "area_vx2", -1.0);
    const double previousAreaCm2 = vc::json::number_or(surface->meta.get(), "area_cm2", -1.0);

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (points && !points->empty()) {
        const double areaVx2 = vc::surface::computeSurfaceAreaVox2(*points);
        (*surface->meta)["area_vx2"] = areaVx2;

        double areaCm2 = std::numeric_limits<double>::quiet_NaN();
        if (voxelSize > 0.0) {
            const double areaUm2 = areaVx2 * voxelSize * voxelSize;
            areaCm2 = areaUm2 * 1e-8;
        } else if (previousAreaVx2 > std::numeric_limits<double>::epsilon() && previousAreaCm2 >= 0.0) {
            const double cm2PerVx2 = previousAreaCm2 / previousAreaVx2;
            areaCm2 = areaVx2 * cm2PerVx2;
        }

        if (std::isfinite(areaCm2)) {
            (*surface->meta)["area_cm2"] = areaCm2;
        } else {
            // Fall back to assuming the geometry is in microns and convert directly.
            const double assumedAreaCm2 = areaVx2 * 1e-8;
            (*surface->meta)["area_cm2"] = assumedAreaCm2;
            qCWarning(lcSegGrowth) << "Fallback surface area conversion applied due to missing voxel size metadata";
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
