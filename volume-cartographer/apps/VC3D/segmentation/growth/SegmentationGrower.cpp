#include "SegmentationGrower.hpp"
#include <vc_object_store/store/GeometryViews.hpp>


#include "../../NeuralTraceServiceManager.hpp"
#include "../SegmentationModule.hpp"
#include "../SurfaceCorrections.hpp"
#include "../SegmentationWidget.hpp"

#include "../../CState.hpp"
#include "../../ViewerManager.hpp"
#include "../../SurfacePanelController.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <QtConcurrent/QtConcurrent>

#include <QDir>
#include <QFileInfo>
#include <QLoggingCategory>
#include <QUuid>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <utility>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <set>
#include <unordered_set>

#include <opencv2/core.hpp>

#include "utils/Json.hpp"

#include "../../SocketRequest.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth);

const QString kCopyLatestSentinel = QStringLiteral("copy_displacement_latest");

namespace
{
QString cacheRootForVolumePkg(const std::shared_ptr<VolumePkg>& pkg)
{
    if (!pkg) {
        return QString();
    }

    const QString base = QString::fromStdString(pkg->getVolpkgDirectory());
    return QDir(base).filePath(QStringLiteral("cache"));
}

std::optional<std::pair<int, int>> worldToGridIndexApprox(QuadSurface* surface,
                                                          const cv::Vec3f& worldPos,
                                                          cv::Vec3f& pointerSeed,
                                                          bool& pointerSeedValid,
                                                          SurfacePatchIndex* patchIndex = nullptr)
{
    if (!surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    if (!pointerSeedValid) {
        pointerSeed = cv::Vec3f(0, 0, 0);
        pointerSeedValid = true;
    }

    surface->pointTo(pointerSeed, worldPos, std::numeric_limits<float>::max(), 400, patchIndex);
    cv::Vec3f raw = surface->loc_raw(pointerSeed);

    const int rows = points->rows;
    const int cols = points->cols;
    if (rows <= 0 || cols <= 0) {
        return std::nullopt;
    }

    int approxCol = static_cast<int>(std::lround(raw[0]));
    int approxRow = static_cast<int>(std::lround(raw[1]));
    approxRow = std::clamp(approxRow, 0, rows - 1);
    approxCol = std::clamp(approxCol, 0, cols - 1);

    if (!vc::geometry::validSurfacePoint((*points)(approxRow, approxCol))) {
        constexpr int kMaxRadius = 12;
        float bestDistSq = std::numeric_limits<float>::max();
        int bestRow = -1;
        int bestCol = -1;

        auto accumulateCandidate = [&](int r, int c) {
            const cv::Vec3f& candidate = (*points)(r, c);
            if (!vc::geometry::validSurfacePoint(candidate)) {
                return;
            }
            const cv::Vec3f diff = candidate - worldPos;
            const float distSq = diff.dot(diff);
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestRow = r;
                bestCol = c;
            }
        };

        for (int radius = 1; radius <= kMaxRadius; ++radius) {
            const int rowStart = std::max(0, approxRow - radius);
            const int rowEnd = std::min(rows - 1, approxRow + radius);
            const int colStart = std::max(0, approxCol - radius);
            const int colEnd = std::min(cols - 1, approxCol + radius);
            for (int r = rowStart; r <= rowEnd; ++r) {
                for (int c = colStart; c <= colEnd; ++c) {
                    accumulateCandidate(r, c);
                }
            }
            if (bestRow != -1) {
                approxRow = bestRow;
                approxCol = bestCol;
                break;
            }
        }

        if (bestRow == -1) {
            return std::nullopt;
        }
    }

    return std::make_pair(approxRow, approxCol);
}

std::optional<std::pair<int, int>> locateGridIndexWithPatchIndex(QuadSurface* surface,
                                                                 SurfacePatchIndex* patchIndex,
                                                                 const cv::Vec3f& worldPos,
                                                                 cv::Vec3f& pointerSeed,
                                                                 bool& pointerSeedValid)
{
    if (!surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    if (patchIndex) {
        cv::Vec2f gridScale = surface->scale();
        float scale = std::max(gridScale[0], gridScale[1]);
        if (!std::isfinite(scale) || scale <= 0.0f) {
            scale = 1.0f;
        }
        const float tolerance = std::max(8.0f, scale * 8.0f);
        // Query without surface filter, then verify result matches our surface
        SurfacePatchIndex::PointQuery query;
        query.worldPoint = worldPos;
        query.tolerance = tolerance;
        if (auto hit = patchIndex->locate(query)) {
            if (hit->surface.get() == surface) {
                const int col = std::clamp(static_cast<int>(std::round(hit->ptr[0])),
                                           0,
                                           points->cols - 1);
                const int row = std::clamp(static_cast<int>(std::round(hit->ptr[1])),
                                           0,
                                           points->rows - 1);
                return std::make_pair(row, col);
            }
        }
    }

    return worldToGridIndexApprox(surface, worldPos, pointerSeed, pointerSeedValid, patchIndex);
}

std::optional<cv::Rect> computeCorrectionsAffectedBounds(QuadSurface* surface,
                                                      const SegmentationCorrectionsPayload& corrections,
                                                      ViewerManager* viewerManager)
{
    if (!surface || corrections.empty()) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    SurfacePatchIndex* patchIndex = viewerManager ? viewerManager->surfacePatchIndex() : nullptr;

    cv::Vec3f pointerSeed{0.0f, 0.0f, 0.0f};
    bool pointerSeedValid = false;

    const int rows = points->rows;
    const int cols = points->cols;

    constexpr int kPaddingCells = 12;

    bool anyMapped = false;
    int unionRowStart = rows;
    int unionRowEnd = 0;
    int unionColStart = cols;
    int unionColEnd = 0;

    for (const auto& collection : corrections.collections) {
        if (collection.points.empty()) {
            continue;
        }

        int collectionMinRow = rows;
        int collectionMaxRow = -1;
        int collectionMinCol = cols;
        int collectionMaxCol = -1;

        for (const auto& colPoint : collection.points) {
            auto gridIndex = locateGridIndexWithPatchIndex(surface,
                                                           patchIndex,
                                                           colPoint.p,
                                                           pointerSeed,
                                                           pointerSeedValid);
            if (!gridIndex) {
                continue;
            }
            const auto [row, col] = *gridIndex;
            collectionMinRow = std::min(collectionMinRow, row);
            collectionMaxRow = std::max(collectionMaxRow, row);
            collectionMinCol = std::min(collectionMinCol, col);
            collectionMaxCol = std::max(collectionMaxCol, col);
        }

        if (collectionMinRow > collectionMaxRow || collectionMinCol > collectionMaxCol) {
            continue;
        }

        anyMapped = true;
        const int rowStart = std::max(0, collectionMinRow - kPaddingCells);
        const int rowEndExclusive = std::min(rows, collectionMaxRow + kPaddingCells + 1);
        const int colStart = std::max(0, collectionMinCol - kPaddingCells);
        const int colEndExclusive = std::min(cols, collectionMaxCol + kPaddingCells + 1);

        unionRowStart = std::min(unionRowStart, rowStart);
        unionRowEnd = std::max(unionRowEnd, rowEndExclusive);
        unionColStart = std::min(unionColStart, colStart);
        unionColEnd = std::max(unionColEnd, colEndExclusive);
    }

    if (!anyMapped) {
        return cv::Rect(0, 0, cols, rows);
    }

    const int width = std::max(0, unionColEnd - unionColStart);
    const int height = std::max(0, unionRowEnd - unionRowStart);
    if (width == 0 || height == 0) {
        return cv::Rect(0, 0, cols, rows);
    }

    return cv::Rect(unionColStart, unionRowStart, width, height);
}

// Compute 3D world-space bounding box from correction points (min 512^3, centered)
// Returns the world-space box and the corresponding 2D grid region
std::optional<CorrectionsBounds> computeCorrectionsBounds(
    const SegmentationCorrectionsPayload& corrections,
    QuadSurface* surface,
    float minWorldSize = 512.0f)
{
    if (!surface || corrections.empty()) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    // Find min/max of all correction point positions (3D world coords)
    cv::Vec3f worldMin(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
    cv::Vec3f worldMax(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());

    bool hasPoints = false;
    for (const auto& collection : corrections.collections) {
        for (const auto& colPoint : collection.points) {
            hasPoints = true;
            worldMin[0] = std::min(worldMin[0], colPoint.p[0]);
            worldMin[1] = std::min(worldMin[1], colPoint.p[1]);
            worldMin[2] = std::min(worldMin[2], colPoint.p[2]);
            worldMax[0] = std::max(worldMax[0], colPoint.p[0]);
            worldMax[1] = std::max(worldMax[1], colPoint.p[1]);
            worldMax[2] = std::max(worldMax[2], colPoint.p[2]);
        }
    }

    if (!hasPoints) {
        return std::nullopt;
    }

    // Compute center and expand to at least minWorldSize in each dimension
    cv::Vec3f center = (worldMin + worldMax) * 0.5f;
    cv::Vec3f halfSize;
    for (int i = 0; i < 3; ++i) {
        float extent = worldMax[i] - worldMin[i];
        halfSize[i] = std::max(extent * 0.5f, minWorldSize * 0.5f);
    }

    worldMin = center - halfSize;
    worldMax = center + halfSize;

    // Find all grid cells whose 3D positions fall within this world-space box
    const int rows = points->rows;
    const int cols = points->cols;

    int gridRowMin = rows;
    int gridRowMax = -1;
    int gridColMin = cols;
    int gridColMax = -1;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& pt = (*points)(r, c);
            if (!vc::geometry::validSurfacePoint(pt)) {
                continue;
            }

            // Check if point is within the world-space bounding box
            if (pt[0] >= worldMin[0] && pt[0] <= worldMax[0] &&
                pt[1] >= worldMin[1] && pt[1] <= worldMax[1] &&
                pt[2] >= worldMin[2] && pt[2] <= worldMax[2]) {
                gridRowMin = std::min(gridRowMin, r);
                gridRowMax = std::max(gridRowMax, r);
                gridColMin = std::min(gridColMin, c);
                gridColMax = std::max(gridColMax, c);
            }
        }
    }

    if (gridRowMin > gridRowMax || gridColMin > gridColMax) {
        // No grid cells found within bounds, return full grid
        return CorrectionsBounds{worldMin, worldMax, cv::Rect(0, 0, cols, rows)};
    }

    // Clip to surface grid bounds (already implicitly done by loop bounds)
    int width = gridColMax - gridColMin + 1;
    int height = gridRowMax - gridRowMin + 1;

    CorrectionsBounds bounds;
    bounds.worldMin = worldMin;
    bounds.worldMax = worldMax;
    bounds.gridRegion = cv::Rect(gridColMin, gridRowMin, width, height);

    return bounds;
}

void queueIndexUpdateForBounds(SurfacePatchIndex* index,
                               const SurfacePatchIndex::SurfacePtr& surface,
                               const cv::Rect& vertexRect)
{
    if (!index || !surface || vertexRect.width <= 0 || vertexRect.height <= 0) {
        return;
    }

    const int rowStart = vertexRect.y;
    const int rowEnd = vertexRect.y + vertexRect.height;
    const int colStart = vertexRect.x;
    const int colEnd = vertexRect.x + vertexRect.width;

    index->queueCellRangeUpdate(surface, rowStart, rowEnd, colStart, colEnd);
}

void refreshSegmentationViewers(ViewerManager* manager)
{
    if (!manager) {
        return;
    }

    manager->forEachBaseViewer([](VolumeViewerBase* viewer) {
        if (!viewer) {
            return;
        }

        if (viewer->surfName() == "segmentation") {
            viewer->invalidateVis();
            viewer->renderVisible(true);
        }
    });
}
struct CopyDisplacementJob
{
    QString socketPath;
    // The input surface, read at geometryRevision from the store file the
    // service shares with this session; the outputs are new surfaces there.
    QString store;
    std::string surfaceUuid, geometryRevision, volumeUuid;
    QString volumeZarrPath;
    int volumeScale{0};
    QString checkpointPath;
    TtaMode ttaMode{TtaMode::Mirror};
    QString ttaMergeMethod{QStringLiteral("vector_geomedian")};
    double ttaOutlierDropThresh{1.25};
    utils::Json customParams;
};

utils::Json sendSocketJsonRequest(const QString& socketPath, const utils::Json& request)
{
    return utils::Json::parse(vc3d::socketRequestLine(
        socketPath.toStdString(), request.dump(), 300, "displacement copy"));  // 5 minute timeout
}

TracerGrowthResult runCopyDisplacementGrowth(const CopyDisplacementJob& job)
{
    TracerGrowthResult result;

    utils::Json request = utils::Json::object();
    request["request_type"] = "displacement_copy_grow";
    request["store"] = job.store.toStdString();
    request["surface_uuid"] = job.surfaceUuid;
    request["geometry_revision"] = job.geometryRevision;
    request["volume_uuid"] = job.volumeUuid;
    request["volume_path"] = job.volumeZarrPath.toStdString();
    request["volume_scale"] = std::max(0, job.volumeScale);
    request["checkpoint_path"] = job.checkpointPath.toStdString();
    request["tta_merge_method"] = job.ttaMergeMethod.toStdString();
    request["tta_outlier_drop_thresh"] = job.ttaOutlierDropThresh;

    switch (job.ttaMode) {
    case TtaMode::Rotate3:
        request["tta"] = true;
        request["tta_transform"] = "rotate3";
        break;
    case TtaMode::None:
        request["tta"] = false;
        break;
    case TtaMode::Mirror:
    default:
        request["tta"] = true;
        request["tta_transform"] = "mirror";
        break;
    }

    if (!job.customParams.is_null()) {
        if (job.customParams.contains("copy_args") && job.customParams["copy_args"].is_object()) {
            request["copy_args"] = job.customParams["copy_args"];
        }
        if (job.customParams.contains("copy_overrides") && job.customParams["copy_overrides"].is_object()) {
            request["overrides"] = job.customParams["copy_overrides"];
        }
    }

    utils::Json response;
    try {
        response = sendSocketJsonRequest(job.socketPath, request);
    } catch (const std::exception& ex) {
        result.error = QStringLiteral("Displacement copy request failed: %1").arg(ex.what());
        return result;
    }

    if (response.contains("error")) {
        QString serviceError;
        if (response["error"].is_string()) {
            serviceError = QString::fromStdString(response["error"].get_string());
        } else {
            serviceError = QString::fromStdString(response["error"].dump());
        }
        result.error = QStringLiteral("Displacement copy service error: %1").arg(serviceError);
        return result;
    }

    if (!response.contains("output_surfaces") || !response["output_surfaces"].is_object()) {
        result.error = QStringLiteral("Displacement copy response missing output_surfaces.");
        return result;
    }

    const auto& outputs = response["output_surfaces"];
    if (!outputs.contains("front") || !outputs["front"].is_string()) {
        result.error = QStringLiteral("Displacement copy response missing front output.");
        return result;
    }
    if (!outputs.contains("back") || !outputs["back"].is_string()) {
        result.error = QStringLiteral("Displacement copy response missing back output.");
        return result;
    }

    result.createdSurfaces = {outputs["front"].get_string(), outputs["back"].get_string()};
    result.statusMessage = QStringLiteral("Displacement copy completed");
    return result;
}
} // namespace

SegmentationGrower::SegmentationGrower(Context context,
                                       UiCallbacks callbacks,
                                       QObject* parent)
    : QObject(parent)
    , _context(std::move(context))
    , _callbacks(std::move(callbacks))
{
}

void SegmentationGrower::updateContext(Context context)
{
    _context = std::move(context);
}

void SegmentationGrower::setSurfacePanel(SurfacePanelController* panel)
{
    _surfacePanel = panel;
}

bool SegmentationGrower::start(const VolumeContext& volumeContext,
                               SegmentationGrowthMethod method,
                               SegmentationGrowthDirection direction,
                               int steps,
                               bool inpaintOnly)
{
    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };

    if (_running) {
        qCInfo(lcSegGrowth) << "Rejecting growth because another operation is running";
        showStatus(tr("A surface growth operation is already running."), kStatusMedium);
        return false;
    }

    if (!_context.module || !_context.widget || !_context.state) {
        showStatus(tr("Segmentation growth is unavailable."), kStatusLong);
        return false;
    }

    auto segmentationSurface = std::dynamic_pointer_cast<QuadSurface>(_context.state->surface("segmentation"));
    if (!segmentationSurface) {
        qCInfo(lcSegGrowth) << "Rejecting growth because segmentation surface is missing";
        showStatus(tr("Segmentation surface is not available."), kStatusMedium);
        return false;
    }

    ensureGenerationsChannel(segmentationSurface.get());

    std::shared_ptr<Volume> growthVolume;
    std::string growthVolumeId = volumeContext.requestedVolumeId;

    if (volumeContext.package && !volumeContext.requestedVolumeId.empty()) {
        try {
            growthVolume = volumeContext.package->volume(volumeContext.requestedVolumeId);
        } catch (const std::out_of_range&) {
            growthVolume.reset();
        }
    }

    if (!growthVolume) {
        growthVolume = volumeContext.activeVolume;
        growthVolumeId = volumeContext.activeVolumeId;
    }

    if (!growthVolume) {
        qCInfo(lcSegGrowth) << "Rejecting growth because no usable volume is available";
        showStatus(tr("No volume available for growth."), kStatusMedium);
        return false;
    }

    if (!volumeContext.requestedVolumeId.empty() &&
        volumeContext.requestedVolumeId != growthVolumeId) {
        showStatus(tr("Selected growth volume unavailable; using the active volume instead."), kStatusMedium);
    }

    SegmentationCorrectionsPayload corrections = _context.module->buildCorrectionsPayload();
    const bool hasCorrections = !corrections.empty();
    const bool usingCorrections = method == SegmentationGrowthMethod::Corrections && hasCorrections;

    if (method == SegmentationGrowthMethod::Corrections && !hasCorrections) {
        qCInfo(lcSegGrowth) << "Corrections growth requested without correction points; continuing with tracer behavior.";
    }

    if (usingCorrections) {
        qCInfo(lcSegGrowth) << "Including" << corrections.collections.size() << "correction set(s)";
    }

    if (_context.module->growthInProgress()) {
        showStatus(tr("Surface growth already in progress"), kStatusMedium);
        return false;
    }

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    int sanitizedSteps = allowZeroSteps ? std::max(0, steps) : std::max(1, steps);
    if (usingCorrections) {
        // Correction-guided tracer should not advance additional steps.
        sanitizedSteps = 0;
    }

    const SegmentationGrowthDirection effectiveDirection = inpaintOnly
        ? SegmentationGrowthDirection::All
        : direction;

    SegmentationGrowthRequest request;
    request.method = method;
    request.direction = effectiveDirection;
    request.steps = sanitizedSteps;
    request.inpaintOnly = inpaintOnly;
    if (method == SegmentationGrowthMethod::Tracer && _context.module) {
        request.allowedGrowthMask = _context.module->takePendingManualAddTracerMask();
        if (!request.allowedGrowthMask.empty()) {
            qCInfo(lcSegGrowth) << "Manual Add tracer growth mask cells"
                                << cv::countNonZero(request.allowedGrowthMask);
            request.manualAddPreview = _context.module->manualAddMode();
        }
    }

    if (inpaintOnly) {
        // Consume any pending overrides; inpainting ignores directional constraints.
        const auto pendingOverride = _context.module->takeShortcutDirectionOverride();
        if (pendingOverride) {
            qCInfo(lcSegGrowth) << "Ignoring direction override for inpaint request.";
        }
        request.allowedDirections = {SegmentationGrowthDirection::All};
    } else {
        if (auto overrideDirs = _context.module->takeShortcutDirectionOverride()) {
            request.allowedDirections = std::move(*overrideDirs);
        }
        if (request.allowedDirections.empty()) {
            request.allowedDirections = _context.widget->allowedGrowthDirections();
            if (request.allowedDirections.empty()) {
                request.allowedDirections = {
                    SegmentationGrowthDirection::Up,
                    SegmentationGrowthDirection::Down,
                    SegmentationGrowthDirection::Left,
                    SegmentationGrowthDirection::Right
                };
            }
        }
    }

    request.directionFields = _context.widget->directionFieldConfigs();

    if (!_context.widget->customParamsValid()) {
        const QString errorText = _context.widget->customParamsError();
        const QString message = errorText.isEmpty()
            ? tr("Custom params JSON is invalid. Fix the contents and try again.")
            : tr("Custom params JSON is invalid: %1").arg(errorText);
        showStatus(message, kStatusLong);
        return false;
    }
    {
        auto customParams = _context.widget->customParamsJson();
        if (!customParams.is_null()) {
            request.customParams = std::move(customParams);
        }
    }
    if (method == SegmentationGrowthMethod::Tracer || method == SegmentationGrowthMethod::Corrections) {
        if (request.customParams.is_null()) {
            request.customParams = utils::Json::object();
        }
        const int growthScale = std::clamp(_context.widget->growthScale(), 0, 5);
        if (!request.customParams.contains("growth_scale")) {
            request.customParams["growth_scale"] = growthScale;
        }
        if (!request.customParams.contains("normal_grid_level") &&
            !request.customParams.contains("normal_grid_scale")) {
            request.customParams["normal_grid_level"] = growthScale;
        }
    }
    const bool manualAddTracerMask = !request.allowedGrowthMask.empty();
    const bool neuralTracerEnabled = _context.widget->neuralTracerEnabled();
    // Heatmap neural tracer integration - pass neural_socket when enabled, GrowPatch will use it as needed
    if (neuralTracerEnabled && !inpaintOnly) {
        const QString checkpointPath = _context.widget->neuralCheckpointPath();
        const QString pythonPath = _context.widget->neuralPythonPath();
        const QString volumeZarr = _context.widget->volumeZarrPath();
        const int volumeScale = _context.widget->neuralVolumeScale();
        const int batchSize = _context.widget->neuralBatchSize();

        if (checkpointPath.isEmpty()) {
            showStatus(tr("Neural tracer enabled but no checkpoint path specified."), kStatusLong);
            return false;
        }
        if (volumeZarr.isEmpty()) {
            showStatus(tr("Neural tracer enabled but no volume zarr path available."), kStatusLong);
            return false;
        }

        auto& serviceManager = NeuralTraceServiceManager::instance();
        showStatus(tr("Starting neural trace service..."), kStatusLong);

        if (!serviceManager.ensureServiceRunning(checkpointPath, volumeZarr, volumeScale, pythonPath)) {
            const QString error = serviceManager.lastError();
            showStatus(tr("Failed to start neural trace service: %1").arg(error), kStatusLong);
            return false;
        }

        const QString socketPath = serviceManager.socketPath();
        if (socketPath.isEmpty()) {
            showStatus(tr("Neural trace service is running but socket path is unavailable."), kStatusLong);
            return false;
        }

        // Add neural socket parameters to custom params
        if (request.customParams.is_null()) {
            request.customParams = utils::Json::object();
        }
        request.customParams["neural_socket"] = socketPath.toStdString();
        request.customParams["neural_batch_size"] = batchSize;

        qCInfo(lcSegGrowth) << "Neural tracer enabled:"
                            << "socket" << socketPath
                            << "batch_size" << batchSize;
    }

    request.corrections = corrections;
    if (method == SegmentationGrowthMethod::Corrections) {
        if (auto zRange = _context.module->correctionsZRange()) {
            request.correctionsZRange = zRange;
        }
    }

    std::optional<cv::Rect> correctionAffectedBounds;
    if (usingCorrections) {
        correctionAffectedBounds = computeCorrectionsAffectedBounds(segmentationSurface.get(),
                                                                    corrections,
                                                                    _context.viewerManager);
        if (correctionAffectedBounds) {
            const int rowEnd = correctionAffectedBounds->y + correctionAffectedBounds->height;
            const int colEnd = correctionAffectedBounds->x + correctionAffectedBounds->width;
            qCInfo(lcSegGrowth) << "Computed correction affected bounds:"
                                << "rows" << correctionAffectedBounds->y << "to" << rowEnd
                                << "cols" << correctionAffectedBounds->x << "to" << colEnd;
        } else {
            qCInfo(lcSegGrowth) << "Unable to compute correction affected bounds; falling back to full surface rebuild.";
        }
    }

    TracerGrowthContext ctx;
    ctx.resumeSurface = segmentationSurface.get();
    ctx.volume = growthVolume.get();
    ctx.level = 0;
    ctx.cacheRoot = cacheRootForVolumePkg(volumeContext.package);
    ctx.voxelSize = growthVolume->voxelSize();
    ctx.normalGridPath = volumeContext.normalGridPath;
    ctx.allowedGrowthMask = request.allowedGrowthMask;

    ctx.growthVolumeId = growthVolumeId;

    if (ctx.cacheRoot.isEmpty()) {
        const auto volumePath = growthVolume->path();
        ctx.cacheRoot = QDir(QString::fromStdString(volumePath.parent_path().string()))
                            .filePath(QStringLiteral("cache"));
    }

    if (ctx.cacheRoot.isEmpty()) {
        qCInfo(lcSegGrowth) << "Tracer growth aborted because cache root is empty";
        showStatus(tr("Cache root unavailable for tracer growth."), kStatusLong);
        return false;
    }

    if (method == SegmentationGrowthMethod::Corrections) {
        if (usingCorrections) {
            showStatus(tr("Applying correction-guided tracer growth..."), kStatusMedium);
        } else {
            showStatus(tr("No correction points provided; running tracer growth..."), kStatusMedium);
        }
    } else {
        const QString status = inpaintOnly
            ? tr("Running tracer inpainting...")
            : tr("Running tracer-based surface growth...");
        showStatus(status, kStatusMedium);
    }

    qCInfo(lcSegGrowth) << "Segmentation growth requested"
                        << segmentationGrowthMethodToString(method)
                        << segmentationGrowthDirectionToString(effectiveDirection)
                        << "steps" << sanitizedSteps
                        << "inpaintOnly" << inpaintOnly;
    qCInfo(lcSegGrowth) << "Growth volume ID" << QString::fromStdString(growthVolumeId);
    qCInfo(lcSegGrowth) << "Starting tracer growth";

    _running = true;
    _context.module->setGrowthInProgress(true);

    ActiveRequest pending;
    pending.volumeContext = volumeContext;
    pending.growthVolume = growthVolume;
    pending.growthVolumeId = growthVolumeId;
    pending.segmentationSurface = segmentationSurface;
    pending.growthVoxelSize = growthVolume->voxelSize();
    pending.usingCorrections = usingCorrections;
    pending.inpaintOnly = inpaintOnly;
    pending.correctionsAffectedBounds = correctionAffectedBounds;
    pending.manualAddPreview = request.manualAddPreview;

    // Compute corrections bounds and snapshot "before" surface for annotation saving
    if (usingCorrections) {
        pending.corrections = corrections;
        auto bounds = computeCorrectionsBounds(corrections, segmentationSurface.get());
        if (bounds) {
            pending.correctionsBounds = bounds;
            pending.beforePin = _context.module ? _context.module->storePin(segmentationSurface.get()) : std::nullopt;
        }
    }

    _activeRequest = std::move(pending);

    auto future = QtConcurrent::run([request, ctx]() {
        try {
            return runTracerGrowth(request, ctx);
        } catch (const std::exception& e) {
            qCWarning(lcSegGrowth) << "Tracer growth worker failed:" << e.what();
            TracerGrowthResult result;
            result.error = QStringLiteral("Tracer growth failed: %1").arg(e.what());
            return result;
        } catch (...) {
            qCWarning(lcSegGrowth) << "Tracer growth worker failed with an unknown exception";
            TracerGrowthResult result;
            result.error = QStringLiteral("Tracer growth failed with an unknown exception");
            return result;
        }
    });
    _watcher = std::make_unique<QFutureWatcher<TracerGrowthResult>>(this);
    connect(_watcher.get(), &QFutureWatcher<TracerGrowthResult>::finished,
            this, &SegmentationGrower::onFutureFinished);
    _watcher->setFuture(future);

    return true;
}

void SegmentationGrower::finalize(bool ok)
{
    if (_context.module) {
        _context.module->setGrowthInProgress(false);
    }
    _running = false;
    _activeRequest.reset();
}

bool SegmentationGrower::startCopyWithNt(const VolumeContext& volumeContext)
{
    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };

    if (_running) {
        showStatus(tr("A surface growth operation is already running."), kStatusMedium);
        return false;
    }
    if (!_context.module || !_context.widget || !_context.state) {
        showStatus(tr("Segmentation growth is unavailable."), kStatusLong);
        return false;
    }
    if (!_context.widget->neuralTracerEnabled()) {
        showStatus(tr("Enable neural tracer before using Copy with NT."), kStatusLong);
        return false;
    }
    if (_context.widget->neuralModelType() != NeuralTracerModelType::DisplacementCopy) {
        showStatus(tr("Select 'Displacement Copy' model type to use Copy with NT."), kStatusLong);
        return false;
    }
    if (!_context.widget->customParamsValid()) {
        const QString errorText = _context.widget->customParamsError();
        const QString message = errorText.isEmpty()
            ? tr("Custom params JSON is invalid. Fix the contents and try again.")
            : tr("Custom params JSON is invalid: %1").arg(errorText);
        showStatus(message, kStatusLong);
        return false;
    }

    auto segmentationSurface = std::dynamic_pointer_cast<QuadSurface>(_context.state->surface("segmentation"));
    if (!segmentationSurface) {
        showStatus(tr("Segmentation surface is not available."), kStatusMedium);
        return false;
    }

    std::shared_ptr<Volume> growthVolume;
    std::string growthVolumeId = volumeContext.requestedVolumeId;
    if (volumeContext.package && !volumeContext.requestedVolumeId.empty()) {
        try {
            growthVolume = volumeContext.package->volume(volumeContext.requestedVolumeId);
        } catch (const std::out_of_range&) {
            growthVolume.reset();
        }
    }
    if (!growthVolume) {
        growthVolume = volumeContext.activeVolume;
        growthVolumeId = volumeContext.activeVolumeId;
    }
    if (!growthVolume) {
        showStatus(tr("No volume available for growth."), kStatusMedium);
        return false;
    }

    const QString copyCheckpointPath = _context.widget->copyCheckpointPath().trimmed();
    const QString pythonPath = _context.widget->neuralPythonPath();
    const QString volumeZarr = _context.widget->volumeZarrPath().trimmed();
    const int volumeScale = _context.widget->neuralVolumeScale();
    const TtaMode ttaMode = _context.widget->ttaMode();
    const QString ttaMergeMethod = _context.widget->ttaMergeMethod().trimmed();
    const double ttaOutlierDropThresh = _context.widget->ttaOutlierDropThresh();
    const bool usingCopyLatestPreset = copyCheckpointPath == kCopyLatestSentinel;

    if (copyCheckpointPath.isEmpty()) {
        showStatus(tr("Displacement copy requires a checkpoint path."), kStatusLong);
        return false;
    }
    if (!usingCopyLatestPreset &&
        (!QFileInfo::exists(copyCheckpointPath) || !QFileInfo(copyCheckpointPath).isFile())) {
        showStatus(tr("Copy checkpoint does not exist: %1").arg(copyCheckpointPath), kStatusLong);
        return false;
    }
    if (volumeZarr.isEmpty()) {
        showStatus(tr("Displacement copy requires a volume zarr path."), kStatusLong);
        return false;
    }

    auto& serviceManager = NeuralTraceServiceManager::instance();
    showStatus(tr("Starting neural trace service for displacement copy..."), kStatusLong);
    if (!serviceManager.ensureServiceRunning(copyCheckpointPath, volumeZarr, volumeScale, pythonPath)) {
        const QString error = serviceManager.lastError();
        showStatus(tr("Failed to start neural trace service: %1").arg(error), kStatusLong);
        return false;
    }

    const QString socketPath = serviceManager.socketPath();
    if (socketPath.isEmpty()) {
        showStatus(tr("Neural trace service is running but socket path is unavailable."), kStatusLong);
        return false;
    }

    // The service reads the stored surface: publish unsaved edits first, then
    // pin the revision on screen. Its outputs are new surfaces in the same store.
    auto* catalog = _surfacePanel ? _surfacePanel->catalog() : nullptr;
    if (!catalog || !catalog->attachment() || !catalog->volume()) {
        showStatus(tr("Displacement copy needs an attached object store and an open volume."), kStatusLong);
        return false;
    }
    const auto autosave = _context.module->autosaveStatus();
    if ((autosave.pending || autosave.saveInProgress) && !_context.module->saveSurface(segmentationSurface.get())) {
        showStatus(tr("Save the surface's edits before using Copy with NT."), kStatusLong);
        return false;
    }
    const auto pin = _context.module->storePin(segmentationSurface.get());
    const auto* row = catalog->row(segmentationSurface->id);
    if (!pin || !pin->geometryRevision || !row || row->catalog || !row->bound()) {
        showStatus(tr("Copy with NT needs a surface of the writable object store that opens in the current volume."),
                   kStatusLong);
        return false;
    }
    SurfaceCatalog::ToolStore toolStore;
    try {
        toolStore = catalog->toolStore();
    } catch (const std::exception& error) {
        showStatus(QString::fromUtf8(error.what()), kStatusLong);
        return false;
    }

    utils::Json customParams = _context.widget->customParamsJson();

    CopyDisplacementJob copyJob;
    copyJob.socketPath = socketPath;
    copyJob.store = QString::fromStdString(toolStore.store.string());
    copyJob.surfaceUuid = pin->object.objectUuid;
    copyJob.geometryRevision = *pin->geometryRevision;
    copyJob.volumeUuid = toolStore.volume;
    copyJob.volumeZarrPath = volumeZarr;
    copyJob.volumeScale = volumeScale;
    copyJob.checkpointPath = copyCheckpointPath;
    copyJob.ttaMode = ttaMode;
    copyJob.ttaMergeMethod = ttaMergeMethod.isEmpty()
        ? QStringLiteral("vector_geomedian")
        : ttaMergeMethod;
    copyJob.ttaOutlierDropThresh = std::max(0.01, ttaOutlierDropThresh);
    copyJob.customParams = customParams;

    _running = true;
    _context.module->setGrowthInProgress(true);

    ActiveRequest pending;
    pending.volumeContext = volumeContext;
    pending.growthVolume = growthVolume;
    pending.growthVolumeId = growthVolumeId;
    pending.segmentationSurface = segmentationSurface;
    pending.growthVoxelSize = growthVolume->voxelSize();
    pending.usingCorrections = false;
    pending.inpaintOnly = false;
    pending.copyDisplacement = true;
    _activeRequest = std::move(pending);

    showStatus(tr("Running displacement copy (creating front/back segments)..."), kStatusMedium);

    auto future = QtConcurrent::run([copyJob]() {
        try {
            return runCopyDisplacementGrowth(copyJob);
        } catch (const std::exception& e) {
            qCWarning(lcSegGrowth) << "Copy displacement worker failed:" << e.what();
            TracerGrowthResult result;
            result.error = QStringLiteral("Copy displacement failed: %1").arg(e.what());
            return result;
        } catch (...) {
            qCWarning(lcSegGrowth) << "Copy displacement worker failed with an unknown exception";
            TracerGrowthResult result;
            result.error = QStringLiteral("Copy displacement failed with an unknown exception");
            return result;
        }
    });
    _watcher = std::make_unique<QFutureWatcher<TracerGrowthResult>>(this);
    connect(_watcher.get(), &QFutureWatcher<TracerGrowthResult>::finished,
            this, &SegmentationGrower::onFutureFinished);
    _watcher->setFuture(future);

    return true;
}

// Compute bounding box of cells that changed between two point matrices
// Returns bounds in the NEW surface's coordinate system
static cv::Rect computeChangedBounds(const cv::Mat_<cv::Vec3f>& oldPts,
                                     const cv::Mat_<cv::Vec3f>& newPts)
{
    // Handle size differences - the new surface may have padding around it
    // The old content is centered in the new surface
    const int padX = (newPts.cols - oldPts.cols) / 2;
    const int padY = (newPts.rows - oldPts.rows) / 2;

    int minRow = newPts.rows, maxRow = -1;
    int minCol = newPts.cols, maxCol = -1;

    // Compare the overlapping region (old surface embedded in new)
    for (int oldRow = 0; oldRow < oldPts.rows; ++oldRow) {
        const int newRow = oldRow + padY;
        if (newRow < 0 || newRow >= newPts.rows) continue;

        for (int oldCol = 0; oldCol < oldPts.cols; ++oldCol) {
            const int newCol = oldCol + padX;
            if (newCol < 0 || newCol >= newPts.cols) continue;

            const auto& o = oldPts(oldRow, oldCol);
            const auto& n = newPts(newRow, newCol);
            if (o[0] != n[0] || o[1] != n[1] || o[2] != n[2]) {
                minRow = std::min(minRow, newRow);
                maxRow = std::max(maxRow, newRow);
                minCol = std::min(minCol, newCol);
                maxCol = std::max(maxCol, newCol);
            }
        }
    }

    // Check padding cells for valid (non-empty) content that was added
    // Invalid points have x == -1
    auto isValid = [](const cv::Vec3f& p) { return p[0] != -1.0f; };

    // Check top/bottom padding rows
    for (int row = 0; row < padY && row < newPts.rows; ++row) {
        for (int col = 0; col < newPts.cols; ++col) {
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
    }
    for (int row = newPts.rows - padY; row < newPts.rows; ++row) {
        if (row < 0) continue;
        for (int col = 0; col < newPts.cols; ++col) {
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
    }
    // Check left/right padding cols
    for (int row = 0; row < newPts.rows; ++row) {
        for (int col = 0; col < padX && col < newPts.cols; ++col) {
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
        for (int col = newPts.cols - padX; col < newPts.cols; ++col) {
            if (col < 0) continue;
            if (isValid(newPts(row, col))) {
                minRow = std::min(minRow, row);
                maxRow = std::max(maxRow, row);
                minCol = std::min(minCol, col);
                maxCol = std::max(maxCol, col);
            }
        }
    }

    if (maxRow < 0) {
        return cv::Rect();  // No changes
    }

    return cv::Rect(minCol, minRow, maxCol - minCol + 1, maxRow - minRow + 1);
}

void SegmentationGrower::onFutureFinished()
{
    if (!_watcher) {
        finalize(false);
        return;
    }

    const TracerGrowthResult result = _watcher->result();
    _watcher.reset();

    if (!_activeRequest) {
        finalize(false);
        return;
    }

    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };

    ActiveRequest request = std::move(*_activeRequest);
    _activeRequest.reset();

    if (!result.error.isEmpty()) {
        qCInfo(lcSegGrowth) << "Tracer growth error" << result.error;
        showStatus(result.error, kStatusLong);
        finalize(false);
        return;
    }

    if (request.copyDisplacement) {
        // The service created the surfaces; they reach the panel through commit events.
        if (result.createdSurfaces.size() != 2) {
            showStatus(tr("Displacement copy did not produce front/back outputs."), kStatusLong);
            finalize(false);
            return;
        }
        showStatus(tr("Displacement copy created surfaces %1 (front) and %2 (back).")
                       .arg(QString::fromStdString(result.createdSurfaces[0]),
                            QString::fromStdString(result.createdSurfaces[1])),
                   kStatusLong);
        finalize(true);
        return;
    }

    if (!result.surface) {
        qCInfo(lcSegGrowth) << "Tracer growth returned null surface";
        showStatus(tr("Tracer growth did not return a surface."), kStatusMedium);
        finalize(false);
        return;
    }

    if (request.manualAddPreview) {
        const bool ok = _context.module && _context.module->applyManualAddTracerPreview(result.surface);
        delete result.surface;
        showStatus(ok
                       ? tr("Manual Add tracer preview updated.")
                       : tr("Manual Add tracer preview update failed."),
                   ok ? kStatusShort : kStatusLong);
        finalize(ok);
        return;
    }

    cv::Mat generations = result.surface->channel("generations");

    std::vector<SurfacePatchIndex::SurfacePtr> surfacesToUpdate;
    surfacesToUpdate.reserve(3);
    auto appendUniqueSurface = [&](const SurfacePatchIndex::SurfacePtr& surface) {
        if (!surface) {
            return;
        }
        if (std::find(surfacesToUpdate.begin(), surfacesToUpdate.end(), surface) == surfacesToUpdate.end()) {
            surfacesToUpdate.push_back(surface);
        }
    };

    appendUniqueSurface(request.segmentationSurface);
    if (_context.module && _context.module->hasActiveSession()) {
        appendUniqueSurface(_context.module->activeBaseSurfaceShared());
    }

    QuadSurface* primarySurface = surfacesToUpdate.empty() ? nullptr : surfacesToUpdate.front().get();
    cv::Mat_<cv::Vec3f>* primaryPoints = primarySurface ? primarySurface->rawPointsPtr() : nullptr;
    cv::Mat_<cv::Vec3f>* resultPoints = result.surface->rawPointsPtr();

    // Compute the changed region before swapping points (for efficient R-tree update)
    // Only use region update when sizes match; otherwise grid coordinates shift
    cv::Rect changedBounds;
    bool sizeChanged = false;
    if (primarySurface && primaryPoints && resultPoints) {
        sizeChanged = (primaryPoints->size() != resultPoints->size());
        if (!sizeChanged) {
            changedBounds = computeChangedBounds(*primaryPoints, *resultPoints);
            qCInfo(lcSegGrowth) << "Changed bounds:" << changedBounds.x << changedBounds.y
                                << changedBounds.width << "x" << changedBounds.height
                                << "(surface:" << resultPoints->cols << "x" << resultPoints->rows << ")";
        } else {
            qCInfo(lcSegGrowth) << "Surface size changed:" << primaryPoints->cols << "x" << primaryPoints->rows
                                << "->" << resultPoints->cols << "x" << resultPoints->rows
                                << "- using full update";
        }
    }

    if (primarySurface && primaryPoints && resultPoints) {
        std::swap(*primaryPoints, *resultPoints);
        primarySurface->invalidateCache();
    } else if (primarySurface && primaryPoints) {
        result.surface->rawPoints().copyTo(*primaryPoints);
        primarySurface->invalidateCache();
    }

    for (const auto& targetSurfacePtr : surfacesToUpdate) {
        QuadSurface* targetSurface = targetSurfacePtr.get();
        if (!targetSurface) {
            continue;
        }

        if (targetSurface != primarySurface) {
            if (auto* destPoints = targetSurface->rawPointsPtr()) {
                if (primaryPoints) {
                    primaryPoints->copyTo(*destPoints);
                } else if (resultPoints) {
                    resultPoints->copyTo(*destPoints);
                } else {
                    result.surface->rawPoints().copyTo(*destPoints);
                }
            }
            targetSurface->invalidateCache();
        }

        if (!generations.empty()) {
            targetSurface->setChannel("generations", generations);
        }

        // Refresh intersection index for this surface so renderIntersections() has up-to-date data
        if (_context.viewerManager) {
            if (!changedBounds.empty()) {
                _context.viewerManager->refreshSurfacePatchIndex(targetSurfacePtr, changedBounds);
            } else {
                _context.viewerManager->refreshSurfacePatchIndex(targetSurfacePtr);
            }
        }
    }

    QuadSurface* surfaceToPersist = nullptr;
    const bool sessionActive = _context.module && _context.module->hasActiveSession();
    if (sessionActive) {
        surfaceToPersist = _context.module->activeBaseSurface();
    }
    if (!surfaceToPersist) {
        surfaceToPersist = request.segmentationSurface.get();
    }

    // Mask is no longer valid after growth/inpainting
    if (surfaceToPersist) {
        surfaceToPersist->invalidateMask();
    }

    if (!sessionActive) {
        if (surfaceToPersist && _context.module) {
            if (!_context.module->saveSurface(surfaceToPersist)) {
                showStatus(tr("Failed to save the grown segmentation to the object store."), kStatusLong);
            }
        }
    } else if (_context.module) {
        _context.module->requestAutosaveFromGrowth();
    }

    std::vector<std::pair<VolumeViewerBase*, bool>> resetDefaults;
    if (_context.viewerManager) {
        ViewerManager* manager = _context.viewerManager;
        manager->forEachBaseViewer([manager, &resetDefaults](VolumeViewerBase* viewer) {
            if (!viewer || viewer->surfName() != "segmentation") {
                return;
            }
            const bool defaultReset = manager->resetDefaultFor(viewer);
            resetDefaults.emplace_back(viewer, defaultReset);
            viewer->setResetViewOnSurfaceChange(false);
        });
    }

    if (_context.state) {
        _context.state->setSurface("segmentation", request.segmentationSurface, false, true);
        // Note: SurfacePatchIndex is automatically updated via handleSurfaceChanged signal
    }

    if (!resetDefaults.empty()) {
        const bool editingActive = _context.module && _context.module->editingEnabled();
        for (auto& entry : resetDefaults) {
            auto* viewer = entry.first;
            if (!viewer) {
                continue;
            }
            if (editingActive) {
                viewer->setResetViewOnSurfaceChange(false);
            } else {
                viewer->setResetViewOnSurfaceChange(entry.second);
            }
        }
    }

    if (sessionActive && _context.module) {
        _context.module->markNextHandlesFromGrowth();
        bool appliedIncremental = false;
        if (request.correctionsAffectedBounds) {
            appliedIncremental = _context.module->applySurfaceUpdateFromGrowth(*request.correctionsAffectedBounds);
        }
        if (!appliedIncremental) {
            qCInfo(lcSegGrowth) << "Refreshing active segmentation session after tracer growth";
            _context.module->refreshSessionFromSurface(surfaceToPersist);
        }
    }

    QuadSurface* currentSegSurface = nullptr;
    std::shared_ptr<Surface> currentSegSurfaceHolder;  // Keep surface alive during this scope
    if (_context.state) {
        currentSegSurfaceHolder = _context.state->surface("segmentation");
        currentSegSurface = dynamic_cast<QuadSurface*>(currentSegSurfaceHolder.get());
    }
    if (!currentSegSurface) {
        currentSegSurface = request.segmentationSurface.get();
    }


    if (_callbacks.applySliceOrientation) {
        _callbacks.applySliceOrientation(currentSegSurface);
    }

    refreshSegmentationViewers(_context.viewerManager);

    // Keep the growth's corrections with the revisions before and after it.
    auto* annotated = primarySurface ? primarySurface : request.segmentationSurface.get();
    if (request.usingCorrections && request.beforePin && request.correctionsBounds && _context.module && annotated &&
        _context.state && _context.state->currentVolume() && _context.module->saveSurface(annotated)) {
        if (const auto afterPin = _context.module->storePin(annotated)) {
            vc3d::corrections::GrowthAnnotation annotation;
            annotation.surfaceUuid = afterPin->object.objectUuid;
            annotation.beforeGeometry = request.beforePin->geometryRevision.value_or("");
            annotation.afterGeometry = afterPin->geometryRevision.value_or("");
            annotation.volumeUsed = request.growthVolumeId;
            const auto& bounds = *request.correctionsBounds;
            annotation.gridRegion = {bounds.gridRegion.x, bounds.gridRegion.y, bounds.gridRegion.width, bounds.gridRegion.height};
            annotation.boundsMin = {bounds.worldMin[0], bounds.worldMin[1], bounds.worldMin[2]};
            annotation.boundsMax = {bounds.worldMax[0], bounds.worldMax[1], bounds.worldMax[2]};
            for (const auto& collection : request.corrections.collections)
                for (const auto& point : collection.points)
                    annotation.points.push_back({collection.id, point.id, collection.name,
                                                 {point.p[0], point.p[1], point.p[2]}, int64_t(point.creation_time)});
            try {
                vc3d::corrections::writeGrowthAnnotation(*_context.state->objectStoreSession(), afterPin->attachment, annotation,
                                                         *_context.state->currentVolume());
            } catch (const std::exception& ex) {
                qCWarning(lcSegGrowth) << "Failed to save the corrections annotation:" << ex.what();
            }
        }
    }

    // Apply grid offset to correction anchors (for surface growth remapping)
    // and save immediately to persist updated positions
    if (result.surface && !result.surface->meta.is_null() && _context.module) {
        if (result.surface->meta.contains("grid_offset") &&
            result.surface->meta["grid_offset"].is_array() &&
            result.surface->meta["grid_offset"].size() == 2) {
            float offsetX = result.surface->meta["grid_offset"][static_cast<size_t>(0)].get_float();
            float offsetY = result.surface->meta["grid_offset"][static_cast<size_t>(1)].get_float();
            if (offsetX != 0.0f || offsetY != 0.0f) {
                _context.module->applyCorrectionAnchorOffset(offsetX, offsetY);
                // The anchors moved with the grown grid.
                _context.module->saveCorrections();
            }
        }
    }

    if (request.usingCorrections && _context.module) {
        _context.module->clearPendingCorrections();
    }

    qCInfo(lcSegGrowth) << "Tracer growth completed successfully";
    delete result.surface;

    QString message;
    if (!result.statusMessage.isEmpty()) {
        message = result.statusMessage;
    } else if (request.usingCorrections) {
        message = tr("Corrections applied; tracer growth complete.");
    } else if (request.inpaintOnly) {
        message = tr("Tracer inpainting complete.");
    } else {
        message = tr("Tracer growth complete.");
    }
    showStatus(message, kStatusLong);

    finalize(true);
}
