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
#include "vc/core/util/Slicing.hpp"

#include <QtConcurrent/QtConcurrent>

#include <QDir>
#include <QLoggingCategory>

#include <algorithm>
#include <filesystem>
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
    if (_context.module && _context.module->hasActiveSession()) {
        if (auto* baseSurface = _context.module->activeBaseSurface()) {
            surfacesToUpdate.push_back(baseSurface);
        }
    }
    if (std::find(surfacesToUpdate.begin(), surfacesToUpdate.end(), request.segmentationSurface) == surfacesToUpdate.end()) {
        surfacesToUpdate.push_back(request.segmentationSurface);
    }

    for (QuadSurface* targetSurface : surfacesToUpdate) {
        if (!targetSurface) {
            continue;
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

        if (auto* destPoints = targetSurface->rawPointsPtr()) {
            result.surface->rawPoints().copyTo(*destPoints);
        }

        if (!generations.empty()) {
            targetSurface->setChannel("generations", generations);
        }

        targetSurface->invalidateCache();

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
    }

    QuadSurface* surfaceToPersist = nullptr;
    if (_context.module && _context.module->hasActiveSession()) {
        surfaceToPersist = _context.module->activeBaseSurface();
    }
    if (!surfaceToPersist) {
        surfaceToPersist = request.segmentationSurface;
    }

    try {
        if (surfaceToPersist) {
            ensureSurfaceMetaObject(surfaceToPersist);
            surfaceToPersist->saveOverwrite();
        }
    } catch (const std::exception& ex) {
        qCInfo(lcSegGrowth) << "Failed to save tracer result" << ex.what();
        showStatus(tr("Failed to save segmentation: %1").arg(ex.what()), kStatusLong);
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
        _context.surfaces->setSurface("segmentation", request.segmentationSurface, false, false);
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

    if (_context.module && _context.module->hasActiveSession()) {
        _context.module->markNextHandlesFromGrowth();
        qCInfo(lcSegGrowth) << "Refreshing active segmentation session after tracer growth";
        _context.module->refreshSessionFromSurface(surfaceToPersist);
    }

    QuadSurface* currentSegSurface = nullptr;
    if (_context.surfaces) {
        currentSegSurface = dynamic_cast<QuadSurface*>(_context.surfaces->surface("segmentation"));
    }
    if (!currentSegSurface) {
        currentSegSurface = request.segmentationSurface;
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
