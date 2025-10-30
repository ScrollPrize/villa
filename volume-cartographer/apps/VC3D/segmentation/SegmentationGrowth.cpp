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
#include "vc/core/util/JsonSafe.hpp"
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

    // Normalize the surface path to get the canonical path under paths/
    std::filesystem::path canonicalPath = surfacePath;
    std::filesystem::path volpkgRoot;
    std::string segmentName;

    // Check if this is already a backup path (contains /backups/)
    std::string pathStr = canonicalPath.string();
    size_t backupsPos = pathStr.find("/backups/");

    if (backupsPos != std::string::npos) {
        // This is a backup path. We need to reconstruct the canonical path.
        // Backup structure: /path/to/scroll.volpkg/backups/segment_name/N/

        qCInfo(lcSegGrowth) << "Detected backup path, normalizing:"
                           << QString::fromStdString(pathStr);

        // Walk up from the path to find the numeric backup directory
        std::filesystem::path tempPath = canonicalPath;

        while (tempPath.has_parent_path()) {
            std::string filename = tempPath.filename().string();

            // Check if this is a numeric backup directory (0, 1, 2, etc.)
            bool isNumeric = !filename.empty() &&
                           std::all_of(filename.begin(), filename.end(), ::isdigit);

            if (isNumeric) {
                // Check if parent is under backups/
                std::filesystem::path parent = tempPath.parent_path();
                std::filesystem::path grandparent = parent.parent_path();

                if (grandparent.filename() == "backups") {
                    // Found it! The parent is the segment name directory
                    segmentName = parent.filename().string();

                    // Reconstruct volpkg root and canonical path
                    volpkgRoot = grandparent.parent_path();
                    canonicalPath = volpkgRoot / "paths" / segmentName;

                    qCInfo(lcSegGrowth) << "Normalized to canonical path:"
                                       << QString::fromStdString(canonicalPath.string());
                    break;
                }
            }
            tempPath = tempPath.parent_path();
        }

        // If we didn't find the pattern, fall back to treating it as a regular path
        if (segmentName.empty()) {
            qCWarning(lcSegGrowth) << "Could not parse backup path structure, treating as regular path";
            volpkgRoot = canonicalPath.parent_path().parent_path();
            segmentName = canonicalPath.filename().string();
        }
    } else {
        // Regular path: /path/to/scroll.volpkg/paths/segment_name
        volpkgRoot = canonicalPath.parent_path().parent_path();
        segmentName = canonicalPath.filename().string();
    }

    // Create centralized backups directory structure
    std::filesystem::path backupsDir = volpkgRoot / "backups" / segmentName;

    // Create backups directory if it doesn't exist
    std::error_code ec;
    std::filesystem::create_directories(backupsDir, ec);
    if (ec) {
        qCWarning(lcSegGrowth) << "Failed to create backups directory:"
                               << QString::fromStdString(ec.message());
        return;
    }

    // Find existing backup directories and determine next backup number
    std::vector<int> existingBackups;
    if (std::filesystem::exists(backupsDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(backupsDir)) {
            if (entry.is_directory()) {
                try {
                    int backupNum = std::stoi(entry.path().filename().string());
                    if (backupNum >= 0 && backupNum < maxBackups) {
                        existingBackups.push_back(backupNum);
                    }
                } catch (...) {
                    // Skip non-numeric directories
                }
            }
        }
    }

    std::sort(existingBackups.begin(), existingBackups.end());

    std::filesystem::path snapshot_dest;

    if (existingBackups.size() < static_cast<size_t>(maxBackups)) {
        // We have room for more backups, find the first available slot
        int nextBackup = 0;
        for (int i = 0; i < maxBackups; ++i) {
            if (std::find(existingBackups.begin(), existingBackups.end(), i) == existingBackups.end()) {
                nextBackup = i;
                break;
            }
        }
        snapshot_dest = backupsDir / std::to_string(nextBackup);
        qCInfo(lcSegGrowth) << "Creating backup" << nextBackup << "of" << maxBackups;
    } else {
        // We're at the limit, rotate backups
        qCInfo(lcSegGrowth) << "At backup limit, rotating backups";

        // Delete the oldest (0)
        std::filesystem::remove_all(backupsDir / "0", ec);
        if (ec) {
            qCWarning(lcSegGrowth) << "Failed to remove oldest backup:"
                                   << QString::fromStdString(ec.message());
        }

        // Rename all backups down by 1 (1->0, 2->1, etc.)
        for (int i = 1; i < maxBackups; ++i) {
            std::filesystem::path oldPath = backupsDir / std::to_string(i);
            std::filesystem::path newPath = backupsDir / std::to_string(i - 1);
            if (std::filesystem::exists(oldPath)) {
                std::filesystem::rename(oldPath, newPath, ec);
                if (ec) {
                    qCWarning(lcSegGrowth) << "Failed to rotate backup" << i
                                           << "to" << (i-1) << ":"
                                           << QString::fromStdString(ec.message());
                }
            }
        }

        // New backup goes in the last slot
        snapshot_dest = backupsDir / std::to_string(maxBackups - 1);
    }

    qCInfo(lcSegGrowth) << "Saving backup to:" << QString::fromStdString(snapshot_dest.string());

    // Save the backup
    surface->save(snapshot_dest, true);

    // Copy mask.tif and generations.tif if they exist from the canonical path
    std::filesystem::path maskFile = canonicalPath / "mask.tif";
    std::filesystem::path generationsFile = canonicalPath / "generations.tif";

    if (std::filesystem::exists(maskFile)) {
        std::filesystem::path destMask = snapshot_dest / "mask.tif";
        std::filesystem::copy_file(maskFile, destMask,
            std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            qCWarning(lcSegGrowth) << "Failed to copy mask.tif to backup:"
                                   << QString::fromStdString(ec.message());
        } else {
            qCInfo(lcSegGrowth) << "Copied mask.tif to backup";

            // Delete the original mask.tif after successful copy
            std::filesystem::remove(maskFile, ec);
            if (ec) {
                qCWarning(lcSegGrowth) << "Failed to delete original mask.tif:"
                                       << QString::fromStdString(ec.message());
            } else {
                qCInfo(lcSegGrowth) << "Deleted original mask.tif";
            }
        }
    }

    if (std::filesystem::exists(generationsFile)) {
        std::filesystem::path destGenerations = snapshot_dest / "generations.tif";
        std::filesystem::copy_file(generationsFile, destGenerations,
            std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            qCWarning(lcSegGrowth) << "Failed to copy generations.tif to backup:"
                                   << QString::fromStdString(ec.message());
        } else {
            qCInfo(lcSegGrowth) << "Copied generations.tif to backup";
        }
    }

    qCInfo(lcSegGrowth) << "Backup creation complete";
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

bool appendDirectionField(const SegmentationDirectionFieldConfig& config,
                          ChunkCache* cache,
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

    const double previousAreaVx2 = vc::json_safe::number_or(surface->meta, "area_vx2", -1.0);
    const double previousAreaCm2 = vc::json_safe::number_or(surface->meta, "area_cm2", -1.0);

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
