#include "SegmentationGrower.hpp"

#include "SegmentationModule.hpp"
#include "SegmentationWidget.hpp"

#include "CVolumeViewer.hpp"
#include "CSurfaceCollection.hpp"
#include "ViewerManager.hpp"
#include "SurfacePanelController.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <QtConcurrent/QtConcurrent>

#include <QDir>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <utility>
#include <cstdint>

#include <opencv2/core.hpp>

#include <nlohmann/json.hpp>

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth);

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

void ensureGenerationsChannel(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    cv::Mat generations = surface->channel("generations");
    if (!generations.empty()) {
        return;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    cv::Mat_<uint16_t> seeded(points->rows, points->cols, static_cast<uint16_t>(1));
    surface->setChannel("generations", seeded);
}

void ensureSurfaceMetaObject(QuadSurface* surface)
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


bool isInvalidPoint(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
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
        pointerSeed = surface->pointer();
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

    if (isInvalidPoint((*points)(approxRow, approxCol))) {
        constexpr int kMaxRadius = 12;
        float bestDistSq = std::numeric_limits<float>::max();
        int bestRow = -1;
        int bestCol = -1;

        auto accumulateCandidate = [&](int r, int c) {
            const cv::Vec3f& candidate = (*points)(r, c);
            if (isInvalidPoint(candidate)) {
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
        if (auto hit = patchIndex->locate(worldPos, tolerance, surface)) {
            const int col = std::clamp(static_cast<int>(std::round(hit->ptr[0])),
                                       0,
                                       points->cols - 1);
            const int row = std::clamp(static_cast<int>(std::round(hit->ptr[1])),
                                       0,
                                       points->rows - 1);
            return std::make_pair(row, col);
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

void queueIndexUpdateForBounds(SurfacePatchIndex* index,
                               QuadSurface* surface,
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

void synchronizeSurfaceMeta(const std::shared_ptr<VolumePkg>& pkg,
                            QuadSurface* surface,
                            SurfacePanelController* panel)
{
    if (!pkg || !surface) {
        return;
    }

    const auto loadedIds = pkg->getLoadedSurfaceIDs();
    for (const auto& id : loadedIds) {
        auto surfMeta = pkg->getSurface(id);
        if (!surfMeta) {
            continue;
        }

        if (surfMeta->path == surface->path) {
            if (!surfMeta->meta) {
                surfMeta->meta = new nlohmann::json(nlohmann::json::object());
            }

            if (surface->meta) {
                *surfMeta->meta = *surface->meta;
            } else {
                surfMeta->meta->clear();
            }

            surfMeta->bbox = surface->bbox();
            surfMeta->setSurface(surface);

            if (panel) {
                panel->refreshSurfaceMetrics(id);
            }
        }
    }
}

void refreshSegmentationViewers(ViewerManager* manager)
{
    if (!manager) {
        return;
    }

    manager->forEachViewer([](CVolumeViewer* viewer) {
        if (!viewer) {
            return;
        }

        if (viewer->surfName() == "segmentation") {
            viewer->invalidateVis();
            viewer->renderVisible(true);
        }
    });
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

void SegmentationGrower::updateUiCallbacks(UiCallbacks callbacks)
{
    _callbacks = std::move(callbacks);
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

    if (!_context.module || !_context.widget || !_context.surfaces) {
        showStatus(tr("Segmentation growth is unavailable."), kStatusLong);
        return false;
    }

    auto* segmentationSurface = dynamic_cast<QuadSurface*>(_context.surfaces->surface("segmentation"));
    if (!segmentationSurface) {
        qCInfo(lcSegGrowth) << "Rejecting growth because segmentation surface is missing";
        showStatus(tr("Segmentation surface is not available."), kStatusMedium);
        return false;
    }

    ensureGenerationsChannel(segmentationSurface);

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
    if (auto customParams = _context.widget->customParamsJson()) {
        request.customParams = std::move(*customParams);
    }

    request.corrections = corrections;
    if (method == SegmentationGrowthMethod::Corrections) {
        if (auto zRange = _context.module->correctionsZRange()) {
            request.correctionsZRange = zRange;
        }
    }

    std::optional<cv::Rect> correctionAffectedBounds;
    if (usingCorrections) {
        correctionAffectedBounds = computeCorrectionsAffectedBounds(segmentationSurface,
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
    ctx.resumeSurface = segmentationSurface;
    ctx.volume = growthVolume.get();
    ctx.cache = _context.chunkCache;
    ctx.cacheRoot = cacheRootForVolumePkg(volumeContext.package);
    ctx.voxelSize = growthVolume->voxelSize();
    ctx.normalGridPath = volumeContext.normalGridPath;

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
    _activeRequest = std::move(pending);

    auto future = QtConcurrent::run(runTracerGrowth, request, ctx);
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

void SegmentationGrower::handleFailure(const QString& message)
{
    auto showStatus = [&](const QString& text, int timeout) {
        if (_callbacks.showStatus) {
            _callbacks.showStatus(text, timeout);
        }
    };
    if (!message.isEmpty()) {
        showStatus(message, kStatusLong);
    }
    finalize(false);
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

    if (!result.surface) {
        qCInfo(lcSegGrowth) << "Tracer growth returned null surface";
        showStatus(tr("Tracer growth did not return a surface."), kStatusMedium);
        finalize(false);
        return;
    }

    const double voxelSize = request.growthVoxelSize;
    cv::Mat generations = result.surface->channel("generations");

    std::vector<QuadSurface*> surfacesToUpdate;
    surfacesToUpdate.reserve(3);
    auto appendUniqueSurface = [&](QuadSurface* surface) {
        if (!surface) {
            return;
        }
        if (std::find(surfacesToUpdate.begin(), surfacesToUpdate.end(), surface) == surfacesToUpdate.end()) {
            surfacesToUpdate.push_back(surface);
        }
    };

    appendUniqueSurface(request.segmentationSurface);
    if (_context.module && _context.module->hasActiveSession()) {
        appendUniqueSurface(_context.module->activeBaseSurface());
    }

    QuadSurface* primarySurface = surfacesToUpdate.empty() ? nullptr : surfacesToUpdate.front();
    cv::Mat_<cv::Vec3f>* primaryPoints = primarySurface ? primarySurface->rawPointsPtr() : nullptr;
    cv::Mat_<cv::Vec3f>* resultPoints = result.surface->rawPointsPtr();

    if (primarySurface && primaryPoints && resultPoints) {
        std::swap(*primaryPoints, *resultPoints);
        primarySurface->invalidateCache();
    } else if (primarySurface && primaryPoints) {
        result.surface->rawPoints().copyTo(*primaryPoints);
        primarySurface->invalidateCache();
    }

    for (QuadSurface* targetSurface : surfacesToUpdate) {
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

        nlohmann::json preservedTags = nlohmann::json::object();
        bool hadPreservedTags = false;
        if (targetSurface->meta && targetSurface->meta->is_object()) {
            auto tagsIt = targetSurface->meta->find("tags");
            if (tagsIt != targetSurface->meta->end() && tagsIt->is_object()) {
                preservedTags = *tagsIt;
                hadPreservedTags = true;
            }
        }

        if (!generations.empty()) {
            targetSurface->setChannel("generations", generations);
        }

        // Copy preserved approval mask from result surface
        cv::Mat approval = result.surface->channel("approval", SURF_CHANNEL_NORESIZE);
        if (!approval.empty()) {
            targetSurface->setChannel("approval", approval);
        }

        if (result.surface->meta) {
            if (targetSurface->meta) {
                delete targetSurface->meta;
                targetSurface->meta = nullptr;
            }
            targetSurface->meta = new nlohmann::json(*result.surface->meta);
        } else {
            ensureSurfaceMetaObject(targetSurface);
        }

        if (hadPreservedTags && targetSurface->meta && targetSurface->meta->is_object()) {
            nlohmann::json mergedTags = preservedTags;
            auto tagsIt = targetSurface->meta->find("tags");
            if (tagsIt != targetSurface->meta->end() && tagsIt->is_object()) {
                mergedTags.update(*tagsIt);
            }
            (*targetSurface->meta)["tags"] = mergedTags;
        }

        updateSegmentationSurfaceMetadata(targetSurface, voxelSize);

        // Refresh intersection index for this surface so renderIntersections() has up-to-date data
        if (_context.viewerManager) {
            _context.viewerManager->refreshSurfacePatchIndex(targetSurface);
        }
    }

    QuadSurface* surfaceToPersist = nullptr;
    const bool sessionActive = _context.module && _context.module->hasActiveSession();
    if (sessionActive) {
        surfaceToPersist = _context.module->activeBaseSurface();
    }
    if (!surfaceToPersist) {
        surfaceToPersist = request.segmentationSurface;
    }

    // Mask is no longer valid after growth/inpainting
    if (surfaceToPersist) {
        surfaceToPersist->invalidateMask();
    }

    if (!sessionActive) {
        try {
            if (surfaceToPersist) {
                ensureSurfaceMetaObject(surfaceToPersist);
                surfaceToPersist->saveOverwrite();
            }
        } catch (const std::exception& ex) {
            qCInfo(lcSegGrowth) << "Failed to save tracer result" << ex.what();
            showStatus(tr("Failed to save segmentation: %1").arg(ex.what()), kStatusLong);
        }
    } else if (_context.module) {
        _context.module->requestAutosaveFromGrowth();
    }

    std::vector<std::pair<CVolumeViewer*, bool>> resetDefaults;
    if (_context.viewerManager) {
        ViewerManager* manager = _context.viewerManager;
        manager->forEachViewer([manager, &resetDefaults](CVolumeViewer* viewer) {
            if (!viewer || viewer->surfName() != "segmentation") {
                return;
            }
            const bool defaultReset = manager->resetDefaultFor(viewer);
            resetDefaults.emplace_back(viewer, defaultReset);
            viewer->setResetViewOnSurfaceChange(false);
        });
    }

    if (_context.surfaces) {
        _context.surfaces->setSurface("segmentation", request.segmentationSurface, false, false, true);
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
    if (_context.surfaces) {
        currentSegSurface = dynamic_cast<QuadSurface*>(_context.surfaces->surface("segmentation"));
    }
    if (!currentSegSurface) {
        currentSegSurface = request.segmentationSurface;
    }

    // Update approval tool after surface replacement (handles case with no active editing session)
    if (_context.module) {
        _context.module->updateApprovalToolAfterGrowth(currentSegSurface);
    }

    QuadSurface* metaSurface = surfaceToPersist ? surfaceToPersist : request.segmentationSurface;
    synchronizeSurfaceMeta(request.volumeContext.package, metaSurface, _surfacePanel);

    if (_surfacePanel) {
        std::vector<std::string> idsToRefresh;
        idsToRefresh.reserve(surfacesToUpdate.size() + 1);

        auto maybeAddId = [&idsToRefresh](QuadSurface* surface) {
            if (!surface) {
                return;
            }
            const std::string& surfaceId = surface->id;
            if (surfaceId.empty()) {
                return;
            }
            if (std::find(idsToRefresh.begin(), idsToRefresh.end(), surfaceId) == idsToRefresh.end()) {
                idsToRefresh.push_back(surfaceId);
            }
        };

        for (QuadSurface* surface : surfacesToUpdate) {
            maybeAddId(surface);
        }
        maybeAddId(currentSegSurface);
        if (_context.module && _context.module->hasActiveSession()) {
            maybeAddId(_context.module->activeBaseSurface());
        }

        for (const auto& id : idsToRefresh) {
            _surfacePanel->refreshSurfaceMetrics(id);
        }
    }

    if (_callbacks.applySliceOrientation) {
        _callbacks.applySliceOrientation(currentSegSurface);
    }

    refreshSegmentationViewers(_context.viewerManager);

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
