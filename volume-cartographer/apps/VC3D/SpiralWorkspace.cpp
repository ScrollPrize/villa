#include "SpiralWorkspace.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ConsoleOutputWidget.hpp"
#include "Keybinds.hpp"
#include "LasagnaServiceManager.hpp"
#include "OpenDataLasagna.hpp"
#include "SpiralPanel.hpp"
#include "SpiralBrushController.hpp"
#include "SpiralServiceManager.hpp"
#include "SurfaceOverlayColors.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "elements/ViewerSplitGrid.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/SpiralOverlayController.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QDialog>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDockWidget>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QGuiApplication>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QMessageBox>
#include <QProgressDialog>
#include <QRegularExpression>
#include <QSettings>
#include <QSaveFile>
#include <QShortcut>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QWindow>
#include <QtEndian>
#include <QtConcurrent/QtConcurrent>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_map>

namespace {

QRgb runDiffMagnitudeColor(float magnitudeFraction)
{
    struct ColorStop {
        float position;
        int red;
        int green;
        int blue;
    };

    // Keep the low end bright enough to remain legible over grayscale slices.
    // Linear interpolation between stops avoids visually implying discrete bands.
    constexpr std::array<ColorStop, 5> stops{{
        {0.00f,   0,  96, 255}, // blue
        {0.25f,   0, 200,  80}, // green
        {0.50f, 255, 224,   0}, // yellow
        {0.75f, 255, 128,   0}, // orange
        {1.00f, 255,  24,  16}, // red
    }};

    const float fraction = std::clamp(magnitudeFraction, 0.0f, 1.0f);
    for (std::size_t index = 1; index < stops.size(); ++index) {
        if (fraction > stops[index].position) continue;
        const ColorStop& lower = stops[index - 1];
        const ColorStop& upper = stops[index];
        const float blend = (fraction - lower.position)
                          / (upper.position - lower.position);
        const auto interpolate = [blend](int from, int to) {
            return static_cast<int>(std::lround(from + blend * (to - from)));
        };
        return qRgba(interpolate(lower.red, upper.red),
                     interpolate(lower.green, upper.green),
                     interpolate(lower.blue, upper.blue), 220);
    }
    return qRgba(stops.back().red, stops.back().green, stops.back().blue, 220);
}

} // namespace

SpiralWorkspace::SpiralWorkspace(CState* mainState, QWidget* parent)
    : QMainWindow(parent), _mainState(mainState)
{
    setObjectName(QStringLiteral("spiralWorkspaceWindow"));
    setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks | QMainWindow::AllowTabbedDocks);
    // Share Main's process-wide decoded-chunk budget. Both workspaces use the
    // single cache owned by their shared Volume.
    _state = new CState(
        _mainState ? _mainState->cacheSizeBytes() : 0,
        this,
        _mainState ? _mainState->decodedCacheBudget() : nullptr);
    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
    // Spiral can trade some intersection detail for substantially cheaper
    // input-patch indexing without changing the main workspace preference.
    _viewerManager->setSurfacePatchSamplingStride(4, false);
    _slices = std::make_unique<AxisAlignedSliceController>(_state, this);
    _slices->setViewerManager(_viewerManager.get());
    // Spiral always uses axis-aligned slices; don't write the user's global
    // preference for the main workspace.
    _slices->setEnabled(true, nullptr, nullptr, false);
    _slices->applyOrientation();
    connect(_viewerManager.get(), &ViewerManager::focusCenteredByUser, this,
            [this](const cv::Vec3f& position) {
                _focusIsAutoDefault = false;
                mirrorFocusToMainWorkspace(position);
            });
    _overlay = std::make_unique<SpiralOverlayController>(this);
    _overlay->bindToViewerManager(_viewerManager.get());
    _brush = std::make_unique<SpiralBrushController>(this);
    _brush->bindToViewerManager(_viewerManager.get());
    _surfaceOverlapOverlay = std::make_unique<SegmentationOverlayController>(_state, this);
    _surfaceOverlapOverlay->setViewerManager(_viewerManager.get());
    _viewerManager->setSegmentationOverlay(_surfaceOverlapOverlay.get());
    _surfaceCategoryVisible = {{QStringLiteral("verified"), false},
                               {QStringLiteral("unverified"), false},
                               {QStringLiteral("shell"), false}};

    _grid = new ViewerSplitGrid(this);
    _grid->setObjectName(QStringLiteral("spiralViewerSplitGrid"));
    setCentralWidget(_grid);
    struct ViewerSpec { const char* surface; std::set<std::string> intersects; };
    const ViewerSpec specs[] = {
        {"segmentation", {"seg xz", "seg yz"}}, {"xy plane", {"segmentation"}},
        {"seg xz", {"segmentation"}}, {"seg yz", {"segmentation"}},
    };
    // Ctrl+click-to-focus is wired by ViewerManager for every viewer.
    for (int pane = 0; pane < 4; ++pane) {
        auto* viewer = _viewerManager->createViewerInWidget(specs[pane].surface, _grid);
        if (pane == 0) {
            _flattenedViewer = viewer;
            _brush->bindFlattenedViewer(viewer);
        }
        viewer->setIntersects(specs[pane].intersects);
        _grid->setViewer(pane, qobject_cast<QWidget*>(viewer->asQObject()));
    }
    _grid->setPaneHidden(2, true);
    QSettings settings;
    _grid->setSplits(settings.value(QStringLiteral("spiral/split_x"), 0.5).toDouble(),
                     settings.value(QStringLiteral("spiral/split_y"), 0.5).toDouble());
    _grid->onSplitChanged = [this]() {
        QSettings settings;
        settings.setValue(QStringLiteral("spiral/split_x"), _grid->splitX());
        settings.setValue(QStringLiteral("spiral/split_y"), _grid->splitY());
    };

    auto* cacheStatsLabel = new QLabel(this);
    cacheStatsLabel->setContentsMargins(8, 0, 8, 0);
    cacheStatsLabel->setText(tr("RAM --  disk --  network --"));
    statusBar()->addPermanentWidget(cacheStatsLabel);
    connect(_viewerManager.get(), &ViewerManager::sharedCacheStatsChanged, cacheStatsLabel,
            [cacheStatsLabel](const QStringList& items) {
                if (!items.isEmpty()) cacheStatsLabel->setText(items.join(QStringLiteral("  ")));
            });

    _service = new SpiralServiceManager(this);
    _transientLasagnaManager = LasagnaServiceManager::createTransient(this);

    _pythonOutputDialog = new QDialog(this, Qt::Window);
    _pythonOutputDialog->setObjectName(QStringLiteral("spiralPythonOutputDialog"));
    _pythonOutputDialog->setWindowTitle(tr("Spiral Python Output"));
    _pythonOutputDialog->setModal(false);
    _pythonOutputDialog->resize(900, 500);
    auto* pythonOutputLayout = new QVBoxLayout(_pythonOutputDialog);
    _pythonOutput = new ConsoleOutputWidget(_pythonOutputDialog);
    _pythonOutput->setTitle(tr("Spiral Python stdout / stderr"));
    _pythonOutput->setMaximumBlockCount(10000);
    pythonOutputLayout->addWidget(_pythonOutput);
    connect(_service, &SpiralServiceManager::logMessage, _pythonOutput,
            [this](const QString& message) {
                const QString line = message.trimmed();
                const bool routineStatusPoll =
                    line.startsWith(QStringLiteral("SPIRAL_HTTP \"GET /session/status HTTP/"))
                    && line.endsWith(QStringLiteral("\" 200 -"));
                const bool routineLogPoll =
                    line.startsWith(QStringLiteral("SPIRAL_HTTP \"GET /logs?after="))
                    && line.contains(QStringLiteral(" HTTP/"))
                    && line.endsWith(QStringLiteral("\" 200 -"));
                if (!routineStatusPoll && !routineLogPoll)
                    _pythonOutput->appendOutput(message);
            });
    connect(_service, &SpiralServiceManager::errorOccurred, this, [this](const QString& error) {
        statusBar()->showMessage(error, 15000);
        _pythonOutput->appendOutput(tr("Error: %1").arg(error));
        _pythonOutputDialog->show();
        _pythonOutputDialog->raise();
    });
    connect(_service, &SpiralServiceManager::inputUploadFinished, this,
            [this](const QString& inputId, const QString& error) {
                statusBar()->showMessage(
                    error.isEmpty()
                        ? tr("Added %1 to the current spiral fit; it is used on the next run").arg(inputId)
                        : tr("Adding %1 to the spiral fit failed: %2").arg(inputId, error),
                    15000);
                auto pending = _pendingBrushPatches.find(inputId);
                if (pending != _pendingBrushPatches.end()) {
                    const PendingBrushPatch patch = pending.value();
                    _pendingBrushPatches.erase(pending);
                    if (error.isEmpty()) {
                        _brushProvisionalPaths[inputId] = patch.path;
                        _unverifiedBrushIds.insert(inputId);
                        registerPendingPatchSurface(inputId, patch.surface, patch.color);
                        _brush->finalizationSucceeded(inputId);
                    } else {
                        QDir(patch.path).removeRecursively();
                        _brush->finalizationFailed(inputId);
                        if (_pendingExitAction) {
                            _commitAfterBrushUploads = false;
                            _pendingExitAction = {};
                            QMessageBox::warning(this, tr("Brush upload failed"), error);
                        }
                    }
                    maybeCommitForPendingExit();
                    return;
                }
                auto pointCollections = _pendingPointCollectionPaths.find(inputId);
                if (pointCollections == _pendingPointCollectionPaths.end()) return;
                const QString path = pointCollections.value();
                _pendingPointCollectionPaths.erase(pointCollections);
                if (error.isEmpty()) {
                    _pointCollectionProvisionalPaths[inputId] = path;
                    _uncommittedPointCollectionIds.insert(inputId);
                    _visibleUncommittedPointCollectionIds.insert(inputId);
                    _brush->setVisiblePointCollectionIds(
                        _visibleUncommittedPointCollectionIds);
                    _brush->finalizationSucceeded(inputId);
                } else {
                    QFile::remove(path);
                    _brush->finalizationFailed(inputId);
                    if (_pendingExitAction) {
                        _commitAfterBrushUploads = false;
                        _pendingExitAction = {};
                        QMessageBox::warning(this, tr("Control-point upload failed"), error);
                    }
                }
                maybeCommitForPendingExit();
            });
    connect(_service, &SpiralServiceManager::commitInputsFinished, this,
            [this](const QStringList& committed, const QString& error) {
                if (!error.isEmpty()) {
                    _commitAfterBrushUploads = false;
                    _pendingExitAction = {};
                    QMessageBox::warning(this, tr("Commit failed"), error);
                    return;
                }
                for (const QString& id : committed) {
                    const QString path = _brushProvisionalPaths.take(id);
                    if (!path.isEmpty()) QDir(path).removeRecursively();
                    _unverifiedBrushIds.remove(id);
                    const QString pclPath = _pointCollectionProvisionalPaths.take(id);
                    if (!pclPath.isEmpty()) QFile::remove(pclPath);
                    _uncommittedPointCollectionIds.remove(id);
                    _visibleUncommittedPointCollectionIds.remove(id);
                }
                _brush->setVisiblePointCollectionIds(
                    _visibleUncommittedPointCollectionIds);
                if (_pendingExitAction && !hasPendingBrushWork()) {
                    auto continuation = std::move(_pendingExitAction);
                    _pendingExitAction = {};
                    continuation();
                }
            });
    auto* finalizeBrushShortcut = new QShortcut(QKeySequence(Qt::SHIFT | Qt::Key_E), this);
    finalizeBrushShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(finalizeBrushShortcut, &QShortcut::activated,
            this, &SpiralWorkspace::finalizeBrushPaint);
    connect(_brush.get(), &SpiralBrushController::brushDiameterChanged,
            this, [this](int diameter) {
                statusBar()->showMessage(tr("Spiral brush diameter: %1 px").arg(diameter), 2500);
            });
    connect(_brush.get(), &SpiralBrushController::pointPlacementRejected,
            this, [this](const QString& message) {
                statusBar()->showMessage(message, 5000);
            });

    _panel = new SpiralPanel(_service, this);
    _panel->setSessionExitGuard([this](std::function<void()> continuation) {
        requestSessionExit(std::move(continuation));
    });
    auto* dock = new QDockWidget(tr("Spiral"), this);
    dock->setObjectName(QStringLiteral("spiralControlDock"));
    dock->setFeatures(QDockWidget::DockWidgetMovable
                      | QDockWidget::DockWidgetFloatable);
    dock->setWidget(_panel);
    addDockWidget(Qt::LeftDockWidgetArea, dock);
    resizeDocks({dock}, {390}, Qt::Horizontal);

    // Match the workaround used by Main's other movable docks. On Wayland,
    // Qt can retain a failed mouse grab after a dock drag and stop delivering
    // mouse events until that grab is explicitly released.
    if (QGuiApplication::platformName() == QLatin1String("wayland")) {
        auto releaseStaleMouseGrab = []() {
            QTimer::singleShot(100, []() {
                if (auto* grabber = QWidget::mouseGrabber())
                    grabber->releaseMouse();
                for (auto* window : QGuiApplication::topLevelWindows())
                    window->setMouseGrabEnabled(false);
            });
        };
        connect(dock, &QDockWidget::topLevelChanged, this, releaseStaleMouseGrab);
        connect(dock, &QDockWidget::dockLocationChanged, this, releaseStaleMouseGrab);
    }

    connect(_panel, &SpiralPanel::volumeSelected, this, &SpiralWorkspace::selectVolume);
    connect(_panel, &SpiralPanel::pythonOutputRequested, this, [this]() {
        _pythonOutputDialog->show();
        _pythonOutputDialog->raise();
        _pythonOutputDialog->activateWindow();
    });
    connect(_panel, &SpiralPanel::visibilityChanged, this, [this](const QString& category, bool shown) {
        if (category == QStringLiteral("output")) {
            _outputVisible = shown;
            _state->setSurface("segmentation", shown ? _currentPreview : nullptr);
            updateSurfaceIntersections();
        } else if (category == QStringLiteral("fibers") || category == QStringLiteral("tracks") || category == QStringLiteral("pcls")) {
            _overlay->setCategoryVisible(category, shown);
        } else if (category == QStringLiteral("pending_only")) {
            _pendingPatchesOnly = shown;
            updateSurfaceIntersections();
        } else if (_surfaceCategoryVisible.contains(category)) {
            setSurfaceCategoryVisible(category, shown);
        }
    });
    connect(_panel, &SpiralPanel::windingRangeChanged, this,
            [this](int minimum, int maximum) {
                _minimumDisplayedWinding = minimum;
                _maximumDisplayedWinding = maximum;
                applyPreviewWindingRange(true);
            });
    connect(_panel, &SpiralPanel::surfaceIntersectionsChanged, this, [this](bool shown) {
        _showSurfaceIntersections = shown;
        updateSurfaceIntersections();
    });
    connect(_panel, &SpiralPanel::surfaceIntersectionStrideChanged,
            this, [this](int stride) {
                _viewerManager->setSurfacePatchSamplingStride(stride, false);
            });
    connect(_panel, &SpiralPanel::surfaceOverlapChanged, this, [this](bool shown) {
        _showSurfaceOverlap = shown;
        updateSurfaceIntersections();
    });
    connect(_panel, &SpiralPanel::runDiffChanged, this, [this](bool shown) {
        _runDiffVisible = shown;
        _overlay->setRunDiffVisible(shown);
        if (shown) {
            loadRunDiff();
        } else {
            ++_runDiffRequestRevision;
            _previewRunDiffImage = {};
            _overlay->publishRunDiff({}, {});
        }
    });
    connect(_panel, &SpiralPanel::lossMapChanged, this,
            [this](const QString& name, qreal opacity) {
                _selectedLossMap = name;
                _lossMapOpacity = opacity;
                updateLossMapOverlay();
            });
    connect(_panel, &SpiralPanel::flattenWithLasagnaRequested,
            this, &SpiralWorkspace::startLasagnaFlatten);

    auto connectLasagnaManager = [this](LasagnaServiceManager* manager) {
        connect(manager, &LasagnaServiceManager::resultsPlaced,
                this, [this, manager](const QString& outputDir,
                                     const QStringList& segmentNames) {
                    if (_activeLasagnaManager != manager) return;
                    handleLasagnaResults(outputDir, segmentNames);
                });
        connect(manager, &LasagnaServiceManager::jobsUpdated,
                this, [this, manager](const QJsonArray& jobs) {
                    if (_activeLasagnaManager != manager || !_lasagnaFlattenRunning) return;
                    for (const QJsonValue& value : jobs) {
                        const QJsonObject job = value.toObject();
                        if (job.value(QStringLiteral("output_name")).toString()
                            != _pendingLasagnaOutputName) continue;
                        _pendingLasagnaJobId =
                            job.value(QStringLiteral("job_id")).toString();
                        const QString state = job.value(QStringLiteral("state")).toString();
                        if (state == QStringLiteral("error")
                            || state == QStringLiteral("cancelled")) {
                            failLasagnaFlatten(
                                job.value(QStringLiteral("error")).toString(
                                    state == QStringLiteral("cancelled")
                                        ? tr("Lasagna job was cancelled")
                                        : tr("Lasagna job failed")),
                                state == QStringLiteral("cancelled"));
                            return;
                        }
                        updateLasagnaFlattenProgress(job);
                        return;
                    }
                });
    };
    connectLasagnaManager(_transientLasagnaManager);
    connectLasagnaManager(&LasagnaServiceManager::instance());
    connect(_transientLasagnaManager, &LasagnaServiceManager::optimizationError,
            this, [this](const QString& error) {
                if (_activeLasagnaManager == _transientLasagnaManager)
                    failLasagnaFlatten(error);
            }, Qt::QueuedConnection);
    connect(_transientLasagnaManager, &LasagnaServiceManager::serviceError,
            this, [this](const QString& error) {
                if (_activeLasagnaManager == _transientLasagnaManager)
                    failLasagnaFlatten(error);
            }, Qt::QueuedConnection);
    connect(_transientLasagnaManager, &LasagnaServiceManager::serviceStopped,
            this, [this]() {
                if (_activeLasagnaManager == _transientLasagnaManager
                    && _lasagnaFlattenRunning) {
                    failLasagnaFlatten(
                        _lasagnaFlattenCancelRequested
                            ? tr("Lasagna job was cancelled")
                            : tr("The Lasagna service stopped unexpectedly"),
                        _lasagnaFlattenCancelRequested);
                }
            }, Qt::QueuedConnection);
    connect(_service, &SpiralServiceManager::previewAvailable, this, &SpiralWorkspace::loadPreview);
    connect(_service, &SpiralServiceManager::connectionStateChanged, this,
            [this](SpiralServiceManager::ConnectionState state, const QString&) {
                using CS = SpiralServiceManager::ConnectionState;
                if (state == CS::Starting || state == CS::Connecting) {
                    _requestedPreviewGeneration = -1;
                    _inputSurfaceGeneration = 0;
                    _geometryManifestPath.clear();
                    _pendingPatchIds.clear();
                    _haveRunDiffBaseline = false;
                    _runDiffPreviousSource.reset();
                    _runDiffPreviousComponents.clear();
                    ++_runDiffRequestRevision;
                    _previewRunDiffImage = {};
                    _previewLossMaps.clear();
                    _loadedLossMap.clear();
                    _loadedLossMapImage = {};
                    _selectedLossMap.clear();
                    _panel->setLossMapOptions({});
                    _panel->setLossMapLegend({});
                    _overlay->reset();
                    updateSurfaceIntersections();
                }
                if (state == CS::Ready && !_service->ownsProcess()) {
                    _pythonOutput->appendOutput(
                        tr("Connected to an independently started service; Python "
                           "stdout / stderr will be relayed every 10 seconds."));
                }
            });
    connect(_service, &SpiralServiceManager::sessionActiveChanged, this,
            &SpiralWorkspace::spiralSessionActiveChanged);
    connect(_service, &SpiralServiceManager::sessionStatusChanged, this,
            &SpiralWorkspace::updatePendingPatchIds);
    connect(_service, &SpiralServiceManager::sessionAccepted, this,
            [this](const QJsonObject& paths, qint64 generation) {
                _sessionPaths = paths;
                _flattenedPreviewActive = false;
                _previewSource.reset();
                _previewComponents.clear();
                _previewConnected = false;
                _runDiffPreviousSource.reset();
                _runDiffPreviousComponents.clear();
                _lasagnaFlattenRunning = false;
                _lasagnaFlattenCancelRequested = false;
                _pendingLasagnaSource.reset();
                _pendingLasagnaJobId.clear();
                closeLasagnaFlattenProgress();
                _panel->setLasagnaFlattenRunning(false);
                _brush->resetSession();
                _pendingBrushPatches.clear();
                _brushProvisionalPaths.clear();
                _unverifiedBrushIds.clear();
                for (const QString& path : std::as_const(_pendingPointCollectionPaths))
                    QFile::remove(path);
                for (const QString& path : std::as_const(_pointCollectionProvisionalPaths))
                    QFile::remove(path);
                _pendingPointCollectionPaths.clear();
                _pointCollectionProvisionalPaths.clear();
                _uncommittedPointCollectionIds.clear();
                _visibleUncommittedPointCollectionIds.clear();
                _brush->setVisiblePointCollectionIds({});
                // A newly loaded fit starts a new comparison sequence even when
                // it reuses the existing service connection.
                _haveRunDiffBaseline = false;
                ++_runDiffRequestRevision;
                _previewRunDiffImage = {};
                _previewLossMaps.clear();
                _loadedLossMap.clear();
                _loadedLossMapImage = {};
                _selectedLossMap.clear();
                _panel->setLossMapOptions({});
                _panel->setLossMapLegend({});
                _overlay->publishRunDiff({}, {});
                _overlay->publishLossMap({}, {}, _lossMapOpacity);
                loadInputSurfaces(paths, static_cast<quint64>(generation));
                updateLasagnaFlattenAvailability();
            });
    // Geometry snapshots arrive as artifacts already transferred to the local
    // cache; the manager hands over a local manifest path.
    connect(_service, &SpiralServiceManager::geometryAvailable, this,
            [this](const QString& manifestPath, quint64 generation) {
                if (manifestPath.isEmpty() || manifestPath == _geometryManifestPath) return;
                _geometryManifestPath = manifestPath;
                loadGeometrySnapshot(manifestPath, generation);
            });
    if (_mainState) {
        connect(_mainState, &CState::vpkgChanged, this, [this](const std::shared_ptr<VolumePkg>&) { refreshVolumes(); });
        connect(_mainState, &CState::volumeChanged, this, [this](const std::shared_ptr<Volume>& volume, const std::string&) {
            if (!_state->currentVolume()) _state->setCurrentVolume(volume);
            refreshVolumes();
        });
    }
    refreshVolumes();
}

QString SpiralWorkspace::mapServicePath(const QString& servicePath) const
{
    const SpiralServiceProfile& profile = _service->profile();
    if (profile.isLocalhost()) return servicePath;
    if (profile.serviceRootPrefix.isEmpty() || profile.localRootPrefix.isEmpty()
        || !servicePath.startsWith(profile.serviceRootPrefix))
        return {};
    // Translate separators as well as prefixes: a Windows viewer may map a
    // POSIX service root.
    QString rest = servicePath.mid(profile.serviceRootPrefix.size());
    rest.replace(QLatin1Char('\\'), QLatin1Char('/'));
    QString local = profile.localRootPrefix;
    while (local.endsWith(QLatin1Char('/')) || local.endsWith(QLatin1Char('\\'))) local.chop(1);
    if (!rest.startsWith(QLatin1Char('/'))) rest.prepend(QLatin1Char('/'));
    return QDir::toNativeSeparators(local + rest);
}

void SpiralWorkspace::loadInputSurfaces(const QJsonObject& servicePaths, quint64 generation)
{
    if (_shuttingDown || generation < _inputSurfaceGeneration) return;
    _inputSurfaceGeneration = generation;
    // Input surfaces are loaded from viewer-local paths. On a remote profile
    // they are translated through the optional path mapping; categories with
    // no local mapping are marked unavailable without blocking the generated
    // preview or geometry display.
    QJsonObject paths;
    QStringList unavailable;
    for (const char* key : {"verified_patches", "unverified_patches", "outer_shell"}) {
        const QString servicePath = servicePaths.value(QString::fromLatin1(key)).toString();
        if (servicePath.isEmpty()) continue;
        const QString local = mapServicePath(servicePath);
        if (local.isEmpty()) unavailable.push_back(QString::fromLatin1(key));
        else paths[QString::fromLatin1(key)] = local;
    }
    if (!unavailable.isEmpty())
        statusBar()->showMessage(
            tr("Input surface overlays unavailable without a local path mapping: %1")
                .arg(unavailable.join(QStringLiteral(", "))), 15000);
    auto* watcher = new QFutureWatcher<InputSurfaceLoadResult>(this);
    connect(watcher, &QFutureWatcher<InputSurfaceLoadResult>::finished, this,
            [this, watcher, generation]() {
                const auto result = watcher->result();
                watcher->deleteLater();
                if (!_shuttingDown && generation == _inputSurfaceGeneration)
                    installInputSurfaces(result, generation);
            });
    watcher->setFuture(QtConcurrent::run([paths, generation]() {
        InputSurfaceLoadResult result;
        const std::pair<const char*, const char*> inputs[] = {
            {"verified", "verified_patches"},
            {"unverified", "unverified_patches"},
            {"shell", "outer_shell"},
        };
        for (const auto& [categoryText, pathKey] : inputs) {
            const QString category = QString::fromLatin1(categoryText);
            const QString rootPath = paths.value(QString::fromLatin1(pathKey)).toString();
            if (rootPath.isEmpty()) continue;
            QStringList candidates;
            const QFileInfo root(rootPath);
            if (root.isDir() && QFileInfo(QDir(rootPath).filePath(QStringLiteral("meta.json"))).isFile()) {
                candidates.push_back(root.absoluteFilePath());
            } else if (root.isDir()) {
                const QFileInfoList children = QDir(rootPath).entryInfoList(
                    QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
                for (const QFileInfo& child : children)
                    if (QFileInfo(QDir(child.absoluteFilePath()).filePath(QStringLiteral("meta.json"))).isFile())
                        candidates.push_back(child.absoluteFilePath());
            }
            if (candidates.isEmpty()) {
                result.warnings.push_back(QObject::tr("No TIFXYZ surfaces found for %1 at %2")
                                              .arg(category, rootPath));
                continue;
            }
            for (int index = 0; index < candidates.size(); ++index) {
                try {
                    auto surface = std::make_shared<QuadSurface>(
                        std::filesystem::path(candidates[index].toStdString()));
                    const QString id = QStringLiteral("spiral/%1/g%2/%3-%4")
                        .arg(category).arg(generation)
                        .arg(QFileInfo(candidates[index]).fileName()).arg(index);
                    surface->id = id.toStdString();
                    result.surfaces.push_back({category, id,
                                               QFileInfo(candidates[index]).fileName(),
                                               std::move(surface)});
                } catch (const std::exception& error) {
                    result.warnings.push_back(QObject::tr("Failed to load %1: %2")
                                                  .arg(candidates[index], QString::fromUtf8(error.what())));
                }
            }
        }
        return result;
    }));
}

void SpiralWorkspace::installInputSurfaces(const InputSurfaceLoadResult& result, quint64 generation)
{
    if (_shuttingDown || generation != _inputSurfaceGeneration) return;
    QHash<QString, QStringList> replacement;
    for (const QString& category : {QStringLiteral("verified"), QStringLiteral("unverified"),
                                    QStringLiteral("shell")})
        replacement[category] = {};
    QHash<QString, QString> replacementSourceIds;
    std::map<std::string, cv::Vec3b> replacementColors;
    std::vector<std::pair<std::string, std::shared_ptr<Surface>>> surfaceUpdates;
    surfaceUpdates.reserve(result.surfaces.size());
    QSet<QString> replacementIds;
    for (const auto& entry : result.surfaces) {
        surfaceUpdates.emplace_back(entry.id.toStdString(), entry.surface);
        replacementIds.insert(entry.id);
        replacement[entry.category].push_back(entry.id);
        replacementSourceIds[entry.id] = entry.sourceId;
        const std::string colorKey = QStringLiteral("%1/%2")
                                         .arg(entry.category == QStringLiteral("shell")
                                                  ? QStringLiteral("shell")
                                                  : QStringLiteral("patch"),
                                              entry.sourceId)
                                         .toStdString();
        auto assignment = _surfaceOverlayColorAssignments.find(colorKey);
        if (assignment == _surfaceOverlayColorAssignments.end()) {
            assignment = _surfaceOverlayColorAssignments
                             .emplace(colorKey, _nextSurfaceOverlayColorIndex++)
                             .first;
        }
        replacementColors.emplace(entry.id.toStdString(),
                                   vc3d::surfaceOverlayColorBgr(assignment->second));
    }
    const auto retired = _surfaceCategoryIds;
    for (auto it = retired.begin(); it != retired.end(); ++it) {
        for (const QString& id : it.value()) {
            if (!replacementIds.contains(id)) {
                surfaceUpdates.emplace_back(id.toStdString(), nullptr);
            }
        }
    }
    _surfaceCategoryIds = replacement;
    _surfaceSourceIds = std::move(replacementSourceIds);
    _surfaceOverlayColors = std::move(replacementColors);
    _state->setSurfacesBatch(surfaceUpdates);
    updateSurfaceIntersections();
    if (!result.warnings.isEmpty())
        statusBar()->showMessage(result.warnings.join(QStringLiteral("; ")), 15000);
}

void SpiralWorkspace::registerPendingPatchSurface(
    const QString& inputId, const std::shared_ptr<QuadSurface>& surface,
    const std::optional<QColor>& explicitColor)
{
    if (!surface || inputId.isEmpty()) return;
    const QString category = explicitColor ? QStringLiteral("brush") : QStringLiteral("ephemeral");
    for (const QString& id : _surfaceCategoryIds.value(category)) {
        if (_surfaceSourceIds.value(id) == inputId) return;
    }
    const QString id = QStringLiteral("spiral/%1/g%2/%3")
                           .arg(category).arg(_inputSurfaceGeneration).arg(inputId);
    _state->setSurface(id.toStdString(), surface);
    _surfaceCategoryIds[category].push_back(id);
    _surfaceSourceIds[id] = inputId;
    const std::string colorKey = QStringLiteral("patch/%1").arg(inputId).toStdString();
    auto assignment = _surfaceOverlayColorAssignments.find(colorKey);
    if (assignment == _surfaceOverlayColorAssignments.end()) {
        assignment = _surfaceOverlayColorAssignments
                         .emplace(colorKey, _nextSurfaceOverlayColorIndex++)
                         .first;
    }
    if (explicitColor) {
        _surfaceOverlayColors[id.toStdString()] = {
            static_cast<uchar>(explicitColor->blue()),
            static_cast<uchar>(explicitColor->green()),
            static_cast<uchar>(explicitColor->red())};
    } else {
        _surfaceOverlayColors[id.toStdString()] =
            vc3d::surfaceOverlayColorBgr(assignment->second);
    }
    if (_pendingPatchesOnly || explicitColor) updateSurfaceIntersections();
}

void SpiralWorkspace::setSurfaceCategoryVisible(const QString& category, bool visible)
{
    if (!_surfaceCategoryVisible.contains(category)) return;
    _surfaceCategoryVisible[category] = visible;
    updateSurfaceIntersections();
}

void SpiralWorkspace::updatePendingPatchIds(const QJsonObject& status)
{
    QSet<QString> pendingPatches;
    QSet<QString> uncommittedDrawnPointCollections;
    for (const QJsonValue& value : status.value(QStringLiteral("ephemeral_inputs")).toArray()) {
        const QJsonObject input = value.toObject();
        if (input.value(QStringLiteral("kind")).toString() == QStringLiteral("patch")
            && !input.value(QStringLiteral("committed")).toBool()) {
            pendingPatches.insert(input.value(QStringLiteral("id")).toString());
        }
        if (input.value(QStringLiteral("kind")).toString() == QStringLiteral("pcl")
            && (input.value(QStringLiteral("role")).toString()
                    == QStringLiteral("drawn_control_points")
                || input.value(QStringLiteral("role")).toString()
                    == QStringLiteral("same_winding"))
            && !input.value(QStringLiteral("committed")).toBool()) {
            uncommittedDrawnPointCollections.insert(
                input.value(QStringLiteral("id")).toString());
        }
    }
    if (uncommittedDrawnPointCollections != _visibleUncommittedPointCollectionIds) {
        _visibleUncommittedPointCollectionIds =
            std::move(uncommittedDrawnPointCollections);
        _brush->setVisiblePointCollectionIds(
            _visibleUncommittedPointCollectionIds);
    }
    if (pendingPatches != _pendingPatchIds) {
        _pendingPatchIds = std::move(pendingPatches);
        if (_pendingPatchesOnly) updateSurfaceIntersections();
    }
}

void SpiralWorkspace::updateSurfaceIntersections()
{
    std::set<std::string> intersections;
    std::map<std::string, cv::Vec3b> surfaceOverlays;
    QSet<QString> shownPendingPatchIds;
    if (_outputVisible && _currentPreview) intersections.insert("segmentation");
    auto addCategory = [this, &intersections, &surfaceOverlays, &shownPendingPatchIds](
                           const QString& category, bool pendingOnly) {
        for (const QString& id : _surfaceCategoryIds.value(category)) {
            const QString sourceId = _surfaceSourceIds.value(id);
            if (pendingOnly
                && (!_pendingPatchIds.contains(sourceId)
                    || shownPendingPatchIds.contains(sourceId))) continue;
            if (pendingOnly) shownPendingPatchIds.insert(sourceId);
            intersections.insert(id.toStdString());
            if (_showSurfaceOverlap) {
                const auto color = _surfaceOverlayColors.find(id.toStdString());
                if (color != _surfaceOverlayColors.end())
                    surfaceOverlays.emplace(color->first, color->second);
            }
        }
    };
    if (_pendingPatchesOnly) {
        addCategory(QStringLiteral("verified"), true);
        addCategory(QStringLiteral("unverified"), true);
        addCategory(QStringLiteral("ephemeral"), true);
    } else {
        for (auto visible = _surfaceCategoryVisible.begin();
             visible != _surfaceCategoryVisible.end(); ++visible) {
            if (visible.value()) addCategory(visible.key(), false);
        }
    }
    // Brush-created patches are session annotations and remain visible even
    // when the dataset input categories are hidden.
    addCategory(QStringLiteral("brush"), false);
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (!viewer) continue;
        if (viewer == _flattenedViewer) {
            viewer->setSurfaceOverlays(surfaceOverlays);
            viewer->setSurfaceOverlayEnabled(_showSurfaceOverlap
                                             && !surfaceOverlays.empty());
            viewer->requestRender("Spiral surface overlays changed");
            continue;
        }
        // Patch-overlap rendering belongs exclusively to the one flattened
        // output viewer, independent of the surface type currently installed.
        viewer->setSurfaceOverlays({});
        viewer->setSurfaceOverlayEnabled(false);
        viewer->setIntersects(_showSurfaceIntersections ? intersections
                                                        : std::set<std::string>{});
        viewer->requestRender("Spiral surface visibility changed");
    }
}

bool SpiralWorkspace::hasActiveSpiralSession() const
{
    return _service && _service->hasActiveSession();
}

void SpiralWorkspace::addPatchToCurrentFit(
    const QString& tifxyzDirectory, const std::shared_ptr<QuadSurface>& surface)
{
    if (!_service) return;
    const QString inputId = QFileInfo(tifxyzDirectory).fileName();
    registerPendingPatchSurface(inputId, surface);
    statusBar()->showMessage(tr("Uploading patch %1 to the Spiral session…").arg(inputId));
    _service->uploadPatch(tifxyzDirectory, inputId);
}

void SpiralWorkspace::addFiberToCurrentFit(const QString& fiberJsonPath)
{
    if (!_service) return;
    const QString inputId = QFileInfo(fiberJsonPath).completeBaseName();
    statusBar()->showMessage(tr("Uploading fiber %1 to the Spiral session…").arg(inputId));
    _service->uploadJsonInput(QStringLiteral("fiber"), fiberJsonPath, inputId);
}

QString SpiralWorkspace::provisionalBrushRoot() const
{
    const QString serviceRoot = _sessionPaths.value(QStringLiteral("dataset_root")).toString();
    const QString localRoot = serviceRoot.isEmpty() ? QString() : mapServicePath(serviceRoot);
    if (!localRoot.isEmpty())
        return QDir(localRoot).filePath(QStringLiteral("provisional_meshes"));
    return QFileInfo(vc3d::settingsFilePath()).dir().filePath(QStringLiteral("provisional_meshes"));
}

void SpiralWorkspace::finalizeBrushPaint()
{
    if (!_service || !_service->hasActiveSession()) {
        statusBar()->showMessage(tr("Load a Spiral fit before finalizing drawn inputs"), 10000);
        return;
    }
    QStringList warnings;
    auto patches = _brush->preparePatches(warnings);
    auto pointCollections = _brush->preparePointCollections(warnings);
    if (!warnings.isEmpty()) statusBar()->showMessage(warnings.join(QStringLiteral("; ")), 10000);
    if (patches.empty() && pointCollections.empty()) {
        maybeCommitForPendingExit();
        return;
    }
    const QString root = provisionalBrushRoot();
    if (!QDir().mkpath(root)) {
        for (const auto& patch : patches) _brush->finalizationFailed(patch.id);
        for (const auto& document : pointCollections)
            _brush->finalizationFailed(document.id);
        QMessageBox::warning(this, tr("Cannot save drawn inputs"),
                             tr("Could not create %1").arg(root));
        _pendingExitAction = {};
        _commitAfterBrushUploads = false;
        return;
    }
    for (const auto& patch : patches) {
        const QString path = QDir(root).filePath(patch.id);
        _pendingBrushPatches.insert(patch.id, {path, patch.color, patch.surface});
        auto* watcher = new QFutureWatcher<QString>(this);
        connect(watcher, &QFutureWatcher<QString>::finished, this,
                [this, watcher, id = patch.id, path]() {
                    const QString error = watcher->result();
                    watcher->deleteLater();
                    auto pending = _pendingBrushPatches.find(id);
                    if (pending == _pendingBrushPatches.end()) {
                        QDir(path).removeRecursively();
                        return;
                    }
                    if (!error.isEmpty()) {
                        _pendingBrushPatches.erase(pending);
                        QDir(path).removeRecursively();
                        _brush->finalizationFailed(id);
                        _commitAfterBrushUploads = false;
                        _pendingExitAction = {};
                        QMessageBox::warning(this, tr("Cannot save brush patch"), error);
                        return;
                    }
                    _service->uploadPatch(path, id);
                });
        const auto surface = patch.surface;
        watcher->setFuture(QtConcurrent::run([surface, path]() -> QString {
            try {
                surface->save(path.toStdString(), false);
                return {};
            } catch (const std::exception& error) {
                return QString::fromUtf8(error.what());
            }
        }));
    }
    for (const auto& document : pointCollections) {
        const QString path = QDir(root).filePath(document.id + QStringLiteral(".json"));
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly)
            || file.write(document.document.toJson(QJsonDocument::Indented)) < 0
            || !file.commit()) {
            _brush->finalizationFailed(document.id);
            _commitAfterBrushUploads = false;
            _pendingExitAction = {};
            QMessageBox::warning(this, tr("Cannot save point collections"),
                                 tr("Could not write %1").arg(path));
        } else {
            _pendingPointCollectionPaths[document.id] = path;
            _service->uploadJsonInput(QStringLiteral("pcl"), path, document.id,
                                      document.role);
        }
    }
}

bool SpiralWorkspace::hasPendingBrushWork() const
{
    return (_brush && (_brush->hasUnfinalizedPaint() || _brush->hasUnfinalizedPolylines()))
        || !_pendingBrushPatches.isEmpty() || !_unverifiedBrushIds.isEmpty()
        || !_pendingPointCollectionPaths.isEmpty()
        || !_uncommittedPointCollectionIds.isEmpty();
}

void SpiralWorkspace::discardBrushWork()
{
    if (_brush) _brush->discardUnfinalized();
    for (const auto& pending : std::as_const(_pendingBrushPatches))
        if (!pending.path.isEmpty()) QDir(pending.path).removeRecursively();
    for (const QString& path : std::as_const(_brushProvisionalPaths))
        if (!path.isEmpty()) QDir(path).removeRecursively();
    for (const QString& path : std::as_const(_pendingPointCollectionPaths))
        if (!path.isEmpty()) QFile::remove(path);
    for (const QString& path : std::as_const(_pointCollectionProvisionalPaths))
        if (!path.isEmpty()) QFile::remove(path);
    _pendingBrushPatches.clear();
    _brushProvisionalPaths.clear();
    _unverifiedBrushIds.clear();
    _pendingPointCollectionPaths.clear();
    _pointCollectionProvisionalPaths.clear();
    _uncommittedPointCollectionIds.clear();
    _visibleUncommittedPointCollectionIds.clear();
    _brush->setVisiblePointCollectionIds({});
    const QStringList brushSurfaceIds = _surfaceCategoryIds.take(QStringLiteral("brush"));
    for (const QString& id : brushSurfaceIds) {
        _surfaceSourceIds.remove(id);
        _surfaceOverlayColors.erase(id.toStdString());
        _state->setSurface(id.toStdString(), nullptr);
    }
    updateSurfaceIntersections();
}

void SpiralWorkspace::requestSessionExit(std::function<void()> continuation)
{
    if (!hasPendingBrushWork()) {
        continuation();
        return;
    }
    QMessageBox box(QMessageBox::Warning, tr("Uncommitted Spiral drawn inputs"),
                    tr("This Spiral session contains brush paint, control-point lines, or "
                       "same-winding point collections that "
                       "have not been committed to the dataset."), QMessageBox::NoButton, this);
    auto* commit = box.addButton(tr("Commit"), QMessageBox::AcceptRole);
    auto* exit = box.addButton(tr("Exit Without Commit"), QMessageBox::DestructiveRole);
    box.addButton(QMessageBox::Cancel);
    box.exec();
    if (box.clickedButton() == commit) {
        _pendingExitAction = std::move(continuation);
        _commitAfterBrushUploads = true;
        if (_brush->hasUnfinalizedPaint() || _brush->hasUnfinalizedPolylines())
            finalizeBrushPaint();
        else maybeCommitForPendingExit();
    } else if (box.clickedButton() == exit) {
        discardBrushWork();
        continuation();
    }
}

void SpiralWorkspace::maybeCommitForPendingExit()
{
    if (!_commitAfterBrushUploads || !_pendingExitAction || !_pendingBrushPatches.isEmpty()
        || !_pendingPointCollectionPaths.isEmpty()) return;
    if (_brush->hasUnfinalizedPaint() || _brush->hasUnfinalizedPolylines()) {
        // A too-small gesture was intentionally left editable. Do not silently
        // discard it during an exit commit.
        _commitAfterBrushUploads = false;
        _pendingExitAction = {};
        QMessageBox::warning(this, tr("Drawn input not committed"),
                             tr("At least one drawn input could not be finalized."));
        return;
    }
    _commitAfterBrushUploads = false;
    if (_unverifiedBrushIds.isEmpty() && _uncommittedPointCollectionIds.isEmpty()) {
        auto continuation = std::move(_pendingExitAction);
        _pendingExitAction = {};
        continuation();
        return;
    }
    _service->commitInputs();
}

SpiralWorkspace::~SpiralWorkspace()
{
    _shuttingDown = true;
    _activeLasagnaManager = nullptr;
    if (_transientLasagnaManager) _transientLasagnaManager->stopService();
    if (_viewerManager) _viewerManager->beginShutdown();
    // Disconnecting never terminates a service VC3D did not launch; only an
    // owned local process is stopped.
    if (_service) _service->disconnectFromService();
    if (_state) {
        _state->setVpkg(nullptr); // the package is borrowed from Main; drop it so closeAll() cannot unload Main's surfaces
        _state->closeAll();
    }
}

void SpiralWorkspace::setFiberViewDistance(double distance)
{
    _overlay->setFiberViewDistance(distance);
}

void SpiralWorkspace::keyPressEvent(QKeyEvent* event)
{
    using namespace vc3d::keybinds;
    if (event && event->key() == keypress::CenterFocusOnCursor.key &&
        event->modifiers() == keypress::CenterFocusOnCursor.modifiers) {
        _viewerManager->centerFocusOnCursor();
        event->accept();
        return;
    }
    if (event && event->key() == keypress::RecenterFocus.key &&
        event->modifiers() == keypress::RecenterFocus.modifiers) {
        _viewerManager->recenterViewersOnCurrentFocus();
        event->accept();
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void SpiralWorkspace::mirrorFocusToMainWorkspace(const cv::Vec3f& position)
{
    // The spiral workspace borrows Main's volume package, so world coordinates
    // are shared: a user-initiated focus move here (R / Ctrl+click) also moves
    // Main's focus. Spiral-local surface ids are not forwarded.
    if (!_mainState) return;
    POI* focus = _mainState->poi("focus");
    if (!focus) focus = new POI;
    focus->p = position;
    focus->surfacePtr.reset();
    focus->suppressViewerRecenter = false;
    focus->suppressTransientPlaneIntersections = true;
    _mainState->setPOI("focus", focus);
}

void SpiralWorkspace::ensureInitialFocus()
{
    if (!_state || _state->poi("focus")) return;
    if (_currentPreview) {
        initializePreviewFocus();
        return;
    }
    // No preview yet: default to the volume center (same policy as the main
    // workspace) so the plane viewers show data immediately.
    if (_viewerManager->resetFocusForVolumeChange(true)) _focusIsAutoDefault = true;
}

void SpiralWorkspace::initializePreviewFocus()
{
    if (!_state || !_currentPreview) return;
    if (_state->poi("focus") && !_focusIsAutoDefault) return;
    auto focus = _state->createSurfaceFocusPoi(*_currentPreview);
    if (!focus) return;
    _state->setPOI("focus", focus.release());
    _focusIsAutoDefault = false;
    // Plane viewers recenter via ViewerManager::handleFocusPoiChanged; also
    // bring the segmentation viewer to the new preview's focus point.
    _viewerManager->recenterViewersOnCurrentFocus();
}

void SpiralWorkspace::refreshVolumes()
{
    QVector<VolumeSelector::VolumeOption> options;
    auto package = _mainState ? _mainState->vpkg() : nullptr;
    // Borrow Main's package so volume-ID resolution, coordinate identity and
    // the remote chunk-cache root all match Main's viewers. Teardown clears it
    // again before closeAll() so the shared package is never unloaded from here.
    if (_state->vpkg() != package) _state->setVpkg(package);
    if (package) {
        for (const auto& id : package->volumeIDs()) {
            auto volume = package->volume(id);
            if (!volume) continue;
            options.push_back({QString::fromStdString(id), QString::fromStdString(volume->name()),
                               QString::fromStdString(volume->path().string())});
        }
    }
    QString selected = QString::fromStdString(_state->currentVolumeId());
    if (selected.isEmpty() && _mainState) selected = QString::fromStdString(_mainState->currentVolumeId());
    _panel->setVolumes(options, selected);
    if (!_state->currentVolume() && _mainState) {
        _state->setCurrentVolume(_mainState->currentVolume());
    }
    ensureInitialFocus();
}

void SpiralWorkspace::selectVolume(const QString& id)
{
    auto package = _mainState ? _mainState->vpkg() : nullptr;
    if (!package || id.isEmpty()) return;
    auto volume = package->volume(id.toStdString());
    if (!volume) return;
    if (volume == _state->currentVolume()) {
        ensureInitialFocus();
        return;
    }
    const bool hadFocus = _state->poi("focus") != nullptr;
    _viewerManager->switchVolume(volume);
    if (!hadFocus) {
        // switchVolume created a volume-center default; prefer the preview
        // focus when one is already loaded.
        _focusIsAutoDefault = true;
        initializePreviewFocus();
    }
    for (auto* viewer : _viewerManager->baseViewers()) if (viewer) viewer->requestRender("Spiral display volume changed");
}

void SpiralWorkspace::updateLasagnaFlattenAvailability()
{
    if (!_panel) return;
    QString reason;
    bool available = !_lasagnaFlattenRunning && _previewSource && _previewConnected;
    if (!_previewSource) {
        reason = tr("Run Spiral once to create a preview before flattening");
    } else if (!_previewConnected) {
        reason = tr("Run Spiral again to create a connected schema-v2 preview");
    } else {
        const QString output = _sessionPaths.value(QStringLiteral("output_directory")).toString();
        if (output.isEmpty() || mapServicePath(output).isEmpty()) {
            available = false;
            reason = tr("The Spiral output directory is not locally accessible; configure a path map");
        } else if (LasagnaServiceManager::findConfigFile(
                       QStringLiteral("flatten_fast_nofilter.json")).isEmpty()) {
            available = false;
            reason = tr("Cannot find Lasagna configs/flatten_fast_nofilter.json");
        }
    }
    _panel->setLasagnaFlattenAvailable(available, reason);
}

QString SpiralWorkspace::lasagnaDataDirectorySettingsKey() const
{
    const CState* state = _state ? _state : _mainState;
    QString projectPath = state ? state->vpkgPath() : QString{};
    if (projectPath.isEmpty()) {
        projectPath =
            _sessionPaths.value(QStringLiteral("dataset_root")).toString();
    }
    if (!projectPath.isEmpty()) {
        projectPath = QFileInfo(projectPath).absoluteFilePath();
    }
    const QString volumeId =
        state ? QString::fromStdString(state->currentVolumeId()) : QString{};
    const QByteArray identity =
        (projectPath + QLatin1Char('\n') + volumeId).toUtf8();
    const QString digest = QString::fromLatin1(
        QCryptographicHash::hash(identity, QCryptographicHash::Sha256).toHex());
    return QStringLiteral("spiral/lasagna_data_dirs/%1").arg(digest);
}

QString SpiralWorkspace::resolveLasagnaDataDirectory()
{
    auto isDataDirectory = [](const QString& path) {
        if (path.trimmed().isEmpty()) return false;
        const QDir dir(path);
        return dir.exists() &&
            !dir.entryList(QStringList{QStringLiteral("*.lasagna.json")},
                           QDir::Files | QDir::Readable | QDir::NoDotAndDotDot)
                 .isEmpty();
    };

    QSettings settings;
    const QString settingsKey = lasagnaDataDirectorySettingsKey();
    const QString persisted = settings.value(settingsKey).toString();

    QStringList candidates;
    auto addCandidate = [&candidates](const QString& path) {
        if (path.trimmed().isEmpty()) return;
        const QString absolute = QFileInfo(path).absoluteFilePath();
        if (!candidates.contains(absolute)) candidates.push_back(absolute);
    };
    addCandidate(persisted);
    addCandidate(_sessionPaths.value(QStringLiteral("dataset_root")).toString());

    const QString scrollZarr =
        _sessionPaths.value(QStringLiteral("scroll_zarr")).toString();
    if (!scrollZarr.isEmpty()) addCandidate(QFileInfo(scrollZarr).absolutePath());

    const CState* state = _state ? _state : _mainState;
    if (state && state->currentVolume()) {
        addCandidate(QFileInfo(QString::fromStdString(
                         state->currentVolume()->path().string())).absolutePath());
    }
    if (state && state->vpkg()) {
        try {
            const auto resolved = vc3d::opendata::resolveLasagnaForVolume(
                *state->vpkg(), state->currentVolumeId());
            if (resolved && !resolved->manifestPath.empty()) {
                addCandidate(QString::fromStdString(
                    resolved->manifestPath.parent_path().string()));
            }
        } catch (...) {
            // A missing or malformed catalog entry should simply fall through
            // to the local candidates and directory picker.
        }
    }
    if (state) {
        const QString projectPath = state->vpkgPath();
        addCandidate(projectPath);
        if (!projectPath.isEmpty())
            addCandidate(QFileInfo(projectPath).absolutePath());
    }

    for (const QString& candidate : candidates) {
        if (!isDataDirectory(candidate)) continue;
        settings.setValue(settingsKey, candidate);
        return candidate;
    }

    QString initialDirectory = persisted;
    if (initialDirectory.isEmpty() || !QFileInfo(initialDirectory).isDir()) {
        initialDirectory =
            _sessionPaths.value(QStringLiteral("dataset_root")).toString();
    }
    if (initialDirectory.isEmpty() || !QFileInfo(initialDirectory).isDir()) {
        initialDirectory = QDir::homePath();
    }

    while (true) {
        const QString selected = QFileDialog::getExistingDirectory(
            this, tr("Select Lasagna data directory"), initialDirectory,
            QFileDialog::ShowDirsOnly);
        if (selected.isEmpty()) return {};
        if (isDataDirectory(selected)) {
            const QString absolute = QFileInfo(selected).absoluteFilePath();
            settings.setValue(settingsKey, absolute);
            return absolute;
        }
        QMessageBox::warning(
            this, tr("Lasagna data directory"),
            tr("%1 does not contain any .lasagna.json dataset files.")
                .arg(selected));
        initialDirectory = selected;
    }
}

void SpiralWorkspace::updateLasagnaFlattenProgress(const QJsonObject& job)
{
    if (!_lasagnaFlattenProgress) return;

    const QString state = job.value(QStringLiteral("state")).toString();
    if (state == QStringLiteral("upload")) {
        const double progress = std::clamp(
            job.value(QStringLiteral("upload_progress")).toDouble(), 0.0, 1.0);
        const int current =
            job.value(QStringLiteral("upload_current")).toInt();
        const int total = job.value(QStringLiteral("upload_total")).toInt();
        QString label =
            job.value(QStringLiteral("upload_label")).toString();
        if (label.isEmpty()) label = tr("Uploading Spiral surface");
        if (total > 0) {
            label += tr(" (%1/%2)").arg(current).arg(total);
        }
        _lasagnaFlattenProgress->setRange(0, 1000);
        _lasagnaFlattenProgress->setValue(
            static_cast<int>(progress * 1000.0));
        _lasagnaFlattenProgress->setLabelText(label);
        return;
    }

    if (state == QStringLiteral("waiting")) {
        const int position =
            job.value(QStringLiteral("queue_position")).toInt();
        _lasagnaFlattenProgress->setRange(0, 0);
        _lasagnaFlattenProgress->setLabelText(
            position > 0
                ? tr("Waiting for Lasagna (queue position %1)…").arg(position)
                : tr("Waiting for Lasagna…"));
        return;
    }

    if (state == QStringLiteral("running")) {
        double progress =
            job.value(QStringLiteral("overall_progress")).toDouble();
        const int step = job.value(QStringLiteral("step")).toInt();
        const int total = job.value(QStringLiteral("total_steps")).toInt();
        if (progress <= 0.0 && step > 0 && total > 0) {
            progress = static_cast<double>(step) / static_cast<double>(total);
        }
        progress = std::clamp(progress, 0.0, 1.0);
        QString stage =
            job.value(QStringLiteral("stage_name")).toString();
        if (stage.isEmpty()) {
            stage = job.value(QStringLiteral("stage")).toString();
        }
        if (stage.isEmpty()) stage = tr("Flattening");

        QString label = total > 0
            ? tr("%1 — step %2/%3").arg(stage).arg(step).arg(total)
            : stage;
        if (job.contains(QStringLiteral("loss"))) {
            label += tr(" — loss %1")
                         .arg(job.value(QStringLiteral("loss")).toDouble(),
                              0, 'g', 5);
        }
        _lasagnaFlattenProgress->setRange(0, 1000);
        _lasagnaFlattenProgress->setValue(
            static_cast<int>(progress * 1000.0));
        _lasagnaFlattenProgress->setLabelText(label);
        return;
    }

    if (state == QStringLiteral("finished")) {
        _lasagnaFlattenProgress->setRange(0, 0);
        _lasagnaFlattenProgress->setLabelText(
            tr("Downloading Lasagna result…"));
    }
}

void SpiralWorkspace::closeLasagnaFlattenProgress()
{
    if (!_lasagnaFlattenProgress) return;
    _lasagnaFlattenProgress->close();
    _lasagnaFlattenProgress->deleteLater();
    _lasagnaFlattenProgress = nullptr;
}

void SpiralWorkspace::failLasagnaFlatten(const QString& error, bool cancelled)
{
    if (!_lasagnaFlattenRunning) return;
    _lasagnaFlattenRunning = false;
    _lasagnaFlattenCancelRequested = false;
    _pendingLasagnaSource.reset();
    _pendingLasagnaJobId.clear();
    releaseLasagnaFlattenService();
    _panel->setLasagnaFlattenRunning(false);
    closeLasagnaFlattenProgress();
    updateLasagnaFlattenAvailability();
    statusBar()->showMessage(
        tr("Lasagna flatten failed: %1").arg(error), 15000);
    if (!_shuttingDown && !cancelled) {
        QMessageBox::warning(
            this, tr("Flatten with Lasagna"),
            tr("Lasagna flatten failed: %1").arg(error));
    }
}

void SpiralWorkspace::releaseLasagnaFlattenService()
{
    LasagnaServiceManager* manager = _activeLasagnaManager;
    _activeLasagnaManager = nullptr;
    if (manager && manager == _transientLasagnaManager)
        manager->stopService();
}

void SpiralWorkspace::startLasagnaFlatten()
{
    updateLasagnaFlattenAvailability();
    if (_lasagnaFlattenRunning || !_previewSource || !_previewConnected) return;

    const QString configPath = LasagnaServiceManager::findConfigFile(
        QStringLiteral("flatten_fast_nofilter.json"));
    QFile configFile(configPath);
    if (!configFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Lasagna configuration"),
                             tr("Cannot read %1").arg(configPath));
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument configDocument =
        QJsonDocument::fromJson(configFile.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !configDocument.isObject()) {
        QMessageBox::warning(
            this, tr("Lasagna configuration"),
            tr("Invalid flatten_fast_nofilter.json: %1").arg(parseError.errorString()));
        return;
    }

    const QString sourcePath = QString::fromStdString(_previewSource->path.string());
    const QJsonObject upload =
        LasagnaServiceManager::makeTifxyzArtifactUpload(sourcePath);
    const QJsonObject sourceRef = upload.value(QStringLiteral("object")).toObject();
    if (sourcePath.isEmpty() || upload.isEmpty() || sourceRef.isEmpty()) {
        QMessageBox::warning(this, tr("Lasagna input"),
                             tr("Cannot package the current Spiral preview"));
        return;
    }

    const QString serviceOutput =
        _sessionPaths.value(QStringLiteral("output_directory")).toString();
    const QString localOutput = mapServicePath(serviceOutput);
    if (localOutput.isEmpty() || !QDir().mkpath(localOutput)) {
        QMessageBox::warning(this, tr("Lasagna output"),
                             tr("Cannot access the Spiral output directory: %1")
                                 .arg(serviceOutput));
        return;
    }

    auto& sharedLasagna = LasagnaServiceManager::instance();
    LasagnaServiceManager* lasagna = nullptr;
    QString dataDirectory;
    if (sharedLasagna.isExternal()) {
        if (!sharedLasagna.isRunning()) {
            QMessageBox::warning(
                this, tr("Lasagna service"),
                tr("The configured external Lasagna service is not connected."));
            return;
        }
        lasagna = &sharedLasagna;
    } else {
        dataDirectory = resolveLasagnaDataDirectory();
        if (dataDirectory.isEmpty()) return;
        lasagna = _transientLasagnaManager;
    }

    QJsonObject config = configDocument.object();
    config[QStringLiteral("external_surfaces")] = QJsonArray{sourceRef};
    if (_state && _state->currentVolume()) {
        try {
            const double voxelSize = _state->currentVolume()->voxelSize();
            if (std::isfinite(voxelSize) && voxelSize > 0.0)
                config[QStringLiteral("voxel_size_um")] = voxelSize;
        } catch (...) {
        }
    }

    QString stem = _previewSourceId;
    stem.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9_.-]+")),
                 QStringLiteral("-"));
    const QString outputName = QStringLiteral("%1-lasagna-flat-%2.tifxyz")
        .arg(stem)
        .arg(QDateTime::currentMSecsSinceEpoch());

    QJsonObject jobSpec;
    jobSpec[QStringLiteral("config")] = config;
    jobSpec[QStringLiteral("linked_surfaces")] = QJsonArray{sourceRef};

    QJsonObject request;
    request[QStringLiteral("config")] = config;
    request[QStringLiteral("job_spec")] = jobSpec;
    request[QStringLiteral("single_segment")] = true;
    request[QStringLiteral("config_name")] =
        QStringLiteral("flatten_fast_nofilter.json");
    request[QStringLiteral("output_name")] = outputName;
    request[QStringLiteral("_objects")] = QJsonArray{upload};
    request[QStringLiteral("source")] = QStringLiteral("VC3D Spiral workspace");

    _lasagnaFlattenRunning = true;
    _lasagnaFlattenCancelRequested = false;
    _pendingLasagnaSource = _previewSource;
    _pendingLasagnaOutputDir = QFileInfo(localOutput).absoluteFilePath();
    _pendingLasagnaOutputName = outputName;
    _pendingLasagnaJobId.clear();
    _activeLasagnaManager = lasagna;
    _panel->setLasagnaFlattenRunning(true);

    closeLasagnaFlattenProgress();
    _lasagnaFlattenProgress = new QProgressDialog(
        tr("Preparing Spiral surface for Lasagna…"), tr("Cancel"),
        0, 0, this);
    _lasagnaFlattenProgress->setWindowTitle(tr("Flatten with Lasagna"));
    _lasagnaFlattenProgress->setWindowModality(Qt::WindowModal);
    _lasagnaFlattenProgress->setMinimumDuration(0);
    _lasagnaFlattenProgress->setAutoClose(false);
    _lasagnaFlattenProgress->setAutoReset(false);
    connect(_lasagnaFlattenProgress, &QProgressDialog::canceled,
            this, [this]() {
                if (!_lasagnaFlattenRunning) return;
                _lasagnaFlattenCancelRequested = true;
                LasagnaServiceManager* manager = _activeLasagnaManager;
                if (!manager) return;
                if (!_pendingLasagnaJobId.isEmpty()) {
                    manager->cancelJob(_pendingLasagnaJobId);
                } else if (!manager->isExternal()) {
                    manager->stopService();
                } else {
                    manager->stopOptimization();
                }
                statusBar()->showMessage(
                    tr("Cancelling Lasagna flatten…"), 5000);
            });
    _lasagnaFlattenProgress->show();

    if (!lasagna->isExternal()) {
        _lasagnaFlattenProgress->setLabelText(
            tr("Starting Lasagna service…"));
        if (!lasagna->ensureServiceRunning({}, dataDirectory)) {
            const bool cancelled = _lasagnaFlattenCancelRequested;
            failLasagnaFlatten(
                cancelled ? tr("Lasagna job was cancelled")
                          : tr("Cannot start Lasagna: %1")
                                .arg(lasagna->lastError()),
                cancelled);
            return;
        }
    }
    if (_lasagnaFlattenCancelRequested) {
        failLasagnaFlatten(tr("Lasagna job was cancelled"), true);
        return;
    }
    if (_lasagnaFlattenProgress) {
        _lasagnaFlattenProgress->setLabelText(
            tr("Preparing Spiral surface for Lasagna…"));
    }
    statusBar()->showMessage(tr("Queued Lasagna flatten: %1").arg(outputName));
    lasagna->startOptimization(request, localOutput);
}

void SpiralWorkspace::handleLasagnaResults(
    const QString& outputDir, const QStringList& segmentNames)
{
    if (!_lasagnaFlattenRunning
        || QFileInfo(outputDir).absoluteFilePath() != _pendingLasagnaOutputDir
        || !segmentNames.contains(_pendingLasagnaOutputName)) {
        return;
    }

    const QString outputName = _pendingLasagnaOutputName;
    const QString resultPath = QDir(outputDir).filePath(outputName);
    const auto source = _pendingLasagnaSource;
    _pendingLasagnaJobId.clear();
    if (_lasagnaFlattenProgress) {
        _lasagnaFlattenProgress->setRange(0, 0);
        _lasagnaFlattenProgress->setLabelText(
            tr("Loading flattened surface…"));
        _lasagnaFlattenProgress->setCancelButton(nullptr);
    }
    // resultsPlaced means the archive is safely local; Python/Torch is no
    // longer needed while the surface is parsed on the C++ worker.
    releaseLasagnaFlattenService();

    auto* watcher = new QFutureWatcher<std::shared_ptr<QuadSurface>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<QuadSurface>>::finished,
            this, [this, watcher, source, outputName]() {
                const auto flattened = watcher->result();
                watcher->deleteLater();
                _lasagnaFlattenRunning = false;
                _lasagnaFlattenCancelRequested = false;
                _pendingLasagnaSource.reset();
                _pendingLasagnaJobId.clear();
                _panel->setLasagnaFlattenRunning(false);
                closeLasagnaFlattenProgress();
                updateLasagnaFlattenAvailability();
                if (!flattened) {
                    statusBar()->showMessage(
                        tr("Lasagna finished, but its output could not be loaded"), 15000);
                    if (!_shuttingDown) {
                        QMessageBox::warning(
                            this, tr("Flatten with Lasagna"),
                            tr("Lasagna finished, but its output could not be loaded."));
                    }
                    return;
                }
                if (_shuttingDown || source != _previewSource) {
                    statusBar()->showMessage(
                        tr("Lasagna output was saved but not displayed because a newer Spiral run exists"),
                        10000);
                    return;
                }

                const quint64 revision = ++_previewDisplayRevision;
                const QString registrationId =
                    QStringLiteral("%1-active").arg(outputName);
                flattened->id = registrationId.toStdString();
                _flattenedPreviewActive = true;
                _overlay->publishRunDiff({}, {});
                _overlay->publishLossMap({}, {}, _lossMapOpacity);
                _state->setSurface(registrationId.toStdString(), flattened);
                installPreviewAliasWhenIndexed(
                    flattened, registrationId, _requestedPreviewGeneration,
                    revision, true, 0);
                statusBar()->showMessage(
                    tr("Lasagna flatten ready: %1").arg(outputName), 10000);
            });
    watcher->setFuture(QtConcurrent::run([resultPath]() {
        try {
            auto loaded = load_quad_from_tifxyz(resultPath.toStdString());
            return std::shared_ptr<QuadSurface>(std::move(loaded));
        } catch (...) {
            return std::shared_ptr<QuadSurface>{};
        }
    }));
}

void SpiralWorkspace::loadPreview(const QString& manifestPath, qint64 generation)
{
    if (_shuttingDown || generation < _requestedPreviewGeneration) return;
    _requestedPreviewGeneration = generation;
    auto* watcher = new QFutureWatcher<PreviewLoadResult>(this);
    connect(watcher, &QFutureWatcher<PreviewLoadResult>::finished, this, [this, watcher, generation]() {
        const auto result = watcher->result();
        watcher->deleteLater();
        if (!_shuttingDown && generation == _requestedPreviewGeneration) installPreview(result, generation);
    });
    watcher->setFuture(QtConcurrent::run([manifestPath]() -> PreviewLoadResult {
        QFile file(manifestPath);
        if (!file.open(QIODevice::ReadOnly)) return {{}, {}, {}, QObject::tr("Cannot read Spiral preview manifest")};
        const QJsonObject manifest = QJsonDocument::fromJson(file.readAll()).object();
        const int schemaVersion = manifest.value(QStringLiteral("schema_version")).toInt();
        if ((schemaVersion != 1 && schemaVersion != 2)
            || manifest.value(QStringLiteral("kind")).toString() != QStringLiteral("spiral_combined_preview"))
            return {{}, {}, {}, QObject::tr("Unsupported Spiral preview manifest")};
        QString surfacePath = manifest.value(QStringLiteral("surface_path")).toString();
        const QString surfaceId = manifest.value(QStringLiteral("surface_id")).toString();
        if (surfacePath.isEmpty() || surfaceId.isEmpty()) return {{}, {}, {}, QObject::tr("Malformed Spiral preview manifest")};
        // The manifest's surface_path is a service-host path; a cache-resident
        // artifact keeps the surface directory (named by its id) beside the
        // manifest, so prefer that local layout when it exists.
        const QString localSurfacePath = QDir(QFileInfo(manifestPath).absolutePath()).filePath(surfaceId);
        if (QFileInfo(QDir(localSurfacePath).filePath(QStringLiteral("meta.json"))).isFile())
            surfacePath = localSurfacePath;
        const bool connected = schemaVersion >= 2;
        const QString rangesKey = connected
            ? QStringLiteral("winding_column_ranges")
            : QStringLiteral("components");
        const QJsonArray components = manifest.value(rangesKey).toArray();
        const QJsonArray windingIds = manifest.value(QStringLiteral("winding_ids")).toArray();
        if (components.isEmpty() || components.size() != windingIds.size()
            || windingIds.at(0).toInt() != 10)
            return {{}, {}, {}, QObject::tr("Invalid Spiral preview components/winding mapping")};
        std::vector<PreviewComponent> previewComponents;
        previewComponents.reserve(components.size());
        int previousEnd = -1;
        for (int index = 0; index < components.size(); ++index) {
            const QJsonArray range = components[index].toArray();
            const int rangeStart = range.size() == 2 ? range[0].toInt() : -1;
            const bool invalidStart = connected
                ? (index == 0 ? rangeStart != 0 : rangeStart != previousEnd)
                : rangeStart <= previousEnd;
            if (range.size() != 2 || invalidStart
                || range[1].toInt() <= range[0].toInt()
                || windingIds[index].toInt() != 10 + index)
                return {{}, {}, {}, QObject::tr("Invalid Spiral winding column range")};
            previewComponents.push_back(
                {range[0].toInt(), range[1].toInt(), windingIds[index].toInt()});
            previousEnd = range[1].toInt();
        }
        QFile metadata(QDir(surfacePath).filePath(QStringLiteral("meta.json")));
        if (!metadata.open(QIODevice::ReadOnly))
            return {{}, {}, {}, QObject::tr("Spiral preview surface metadata is missing")};
        const QJsonObject meta = QJsonDocument::fromJson(metadata.readAll()).object();
        if (meta.value(rangesKey) != manifest.value(rangesKey)
            || meta.value(QStringLiteral("component_winding_ids")) != manifest.value(QStringLiteral("winding_ids"))
            || meta.value(QStringLiteral("uuid")).toString() != surfaceId)
            return {{}, {}, {}, QObject::tr("Spiral preview metadata does not match its generation manifest")};
        try {
            // Keep the artifact as a lazy descriptor. Only the selected winding
            // columns are decoded when applyPreviewWindingRange() builds the
            // compact display surface.
            auto surface = std::make_shared<QuadSurface>(surfacePath.toStdString());
            surface->id = surfaceId.toStdString();
            const cv::Size gridSize = surface->gridSize();
            std::vector<PreviewLoadResult::LossMap> lossMaps;
            const QString artifactRoot = QFileInfo(manifestPath).absolutePath();
            for (const QJsonValue& value : manifest.value(QStringLiteral("loss_maps")).toArray()) {
                const QJsonObject entry = value.toObject();
                const QString name = entry.value(QStringLiteral("name")).toString();
                const QString relativePath = QDir::cleanPath(
                    entry.value(QStringLiteral("path")).toString());
                if (name.isEmpty() || relativePath.isEmpty()
                    || QDir::isAbsolutePath(relativePath)
                    || relativePath == QStringLiteral("..")
                    || relativePath.startsWith(QStringLiteral("../")))
                    continue;
                const QString imagePath = QDir(artifactRoot).filePath(relativePath);
                QImageReader reader(imagePath);
                if (!reader.canRead()
                    || reader.size().width() != gridSize.width
                    || reader.size().height() != gridSize.height)
                    continue;
                PreviewLoadResult::LossMap map;
                map.name = name;
                map.imagePath = imagePath;
                map.weight = entry.value(QStringLiteral("weight")).toDouble();
                map.p50 = entry.value(QStringLiteral("p50")).toDouble();
                map.p95 = entry.value(QStringLiteral("p95")).toDouble();
                map.maximum = entry.value(QStringLiteral("maximum")).toDouble();
                map.displayMaximum = entry.value(QStringLiteral("display_maximum")).toDouble();
                map.sampleCount = entry.value(QStringLiteral("sample_count")).toInteger();
                map.eligibleSampleCount = entry.contains(QStringLiteral("eligible_sample_count"))
                    ? entry.value(QStringLiteral("eligible_sample_count")).toInteger()
                    : map.sampleCount;
                map.projectedSampleCount = entry.contains(QStringLiteral("projected_sample_count"))
                    ? entry.value(QStringLiteral("projected_sample_count")).toInteger()
                    : map.sampleCount;
                map.offSurfaceSampleCount =
                    entry.value(QStringLiteral("off_surface_sample_count")).toInteger();
                map.omittedSampleCount =
                    entry.value(QStringLiteral("omitted_sample_count")).toInteger();
                map.supportedPixels = entry.value(QStringLiteral("supported_pixels")).toInteger();
                lossMaps.push_back(std::move(map));
            }
            return {surface, surfaceId, std::move(previewComponents), {},
                    std::move(lossMaps), connected};
        } catch (const std::exception& error) {
            return {{}, {}, {}, QString::fromUtf8(error.what())};
        }
    }));
}

void SpiralWorkspace::installPreview(const PreviewLoadResult& result, qint64 generation)
{
    if (!result.surface) { statusBar()->showMessage(result.error, 15000); return; }
    _flattenedPreviewActive = false;
    if (_haveRunDiffBaseline && _previewSource) {
        _runDiffPreviousSource = _previewSource;
        _runDiffPreviousComponents = _previewComponents;
    } else {
        _runDiffPreviousSource.reset();
        _runDiffPreviousComponents.clear();
    }
    _previewSource = result.surface;
    _previewSourceId = result.surfaceId;
    _previewComponents = result.components;
    _previewConnected = result.connected;
    ++_runDiffRequestRevision;
    _previewRunDiffImage = {};
    _previewLossMaps.clear();
    _loadedLossMap.clear();
    _loadedLossMapImage = {};
    QStringList lossMapNames;
    for (const auto& map : result.lossMaps) {
        _previewLossMaps.insert(map.name, map);
        lossMapNames.push_back(map.name);
    }
    _panel->setLossMapOptions(lossMapNames);
    _overlay->publishRunDiff({}, {});
    _overlay->publishLossMap({}, {}, _lossMapOpacity);
    applyPreviewWindingRange(false);
    _haveRunDiffBaseline = true;
    updateLasagnaFlattenAvailability();
}

void SpiralWorkspace::loadRunDiff()
{
    ++_runDiffRequestRevision;
    _previewRunDiffImage = {};
    if (!_runDiffVisible || _flattenedPreviewActive || !_runDiffPreviousSource
        || !_currentPreview || _currentPreviewComponents.empty()) {
        updateRunDiffOverlay();
        return;
    }

    std::vector<PreviewComponent> previousSelected;
    for (const PreviewComponent& component : _runDiffPreviousComponents) {
        if (component.winding < _minimumDisplayedWinding) continue;
        if (_maximumDisplayedWinding >= 0
            && component.winding > _maximumDisplayedWinding)
            continue;
        previousSelected.push_back(component);
    }
    if (previousSelected.empty()) {
        updateRunDiffOverlay();
        return;
    }

    const int firstColumn = previousSelected.front().firstColumn;
    const int endColumn = previousSelected.back().endColumn;
    const cv::Size previousGridSize = _runDiffPreviousSource->gridSize();
    if (firstColumn < 0 || endColumn > previousGridSize.width
        || firstColumn >= endColumn || previousGridSize.height <= 0) {
        updateRunDiffOverlay();
        return;
    }
    for (PreviewComponent& component : previousSelected) {
        component.firstColumn -= firstColumn;
        component.endColumn -= firstColumn;
    }

    const auto previousSource = _runDiffPreviousSource;
    const auto current = _currentPreview;
    const auto currentComponents = _currentPreviewComponents;
    const qint64 generation = _requestedPreviewGeneration;
    const quint64 displayRevision = _previewDisplayRevision;
    const quint64 requestRevision = _runDiffRequestRevision;
    const cv::Rect region(firstColumn, 0, endColumn - firstColumn,
                          previousGridSize.height);
    auto* watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this,
            [this, watcher, previousSource, current, generation,
             displayRevision, requestRevision]() {
                const QImage image = watcher->result();
                watcher->deleteLater();
                if (_shuttingDown || !_runDiffVisible
                    || generation != _requestedPreviewGeneration
                    || displayRevision != _previewDisplayRevision
                    || requestRevision != _runDiffRequestRevision
                    || previousSource != _runDiffPreviousSource
                    || current != _currentPreview)
                    return;
                _previewRunDiffImage = image;
                updateRunDiffOverlay();
            });
    watcher->setFuture(QtConcurrent::run(
        [previousSource, previousSelected, current, currentComponents, region]() {
            try {
                auto loaded = load_quad_from_tifxyz_region(
                    previousSource->path, region);
                auto previous = std::shared_ptr<QuadSurface>(std::move(loaded));
                return buildRunDiffImage(previous, previousSelected,
                                         current, currentComponents);
            } catch (const std::exception&) {
                return QImage{};
            }
        }));
}

QImage SpiralWorkspace::buildRunDiffImage(
    const std::shared_ptr<QuadSurface>& previous,
    const std::vector<PreviewComponent>& previousComponents,
    const std::shared_ptr<QuadSurface>& current,
    const std::vector<PreviewComponent>& currentComponents)
{
    if (!previous || !current) return {};
    const cv::Mat_<cv::Vec3f>* previousPoints = previous->rawPointsPtr();
    const cv::Mat_<cv::Vec3f>* currentPoints = current->rawPointsPtr();
    if (!previousPoints || !currentPoints || previousPoints->empty() || currentPoints->empty())
        return {};

    std::unordered_map<int, PreviewComponent> previousByWinding;
    previousByWinding.reserve(previousComponents.size());
    for (const PreviewComponent& component : previousComponents)
        previousByWinding.emplace(component.winding, component);

    struct ComponentPair {
        PreviewComponent previous;
        PreviewComponent current;
        int width = 0;
    };
    std::vector<ComponentPair> pairs;
    pairs.reserve(currentComponents.size());
    for (const PreviewComponent& currentComponent : currentComponents) {
        const auto found = previousByWinding.find(currentComponent.winding);
        if (found == previousByWinding.end()) continue;
        const int width = std::min(
            currentComponent.endColumn - currentComponent.firstColumn,
            found->second.endColumn - found->second.firstColumn);
        if (width > 0)
            pairs.push_back({found->second, currentComponent, width});
    }

    const int commonRows = std::min(previousPoints->rows, currentPoints->rows);
    auto forEachMagnitude = [&](const auto& visitor) {
        for (const ComponentPair& pair : pairs) {
            for (int row = 0; row < commonRows; ++row) {
                for (int offset = 0; offset < pair.width; ++offset) {
                    const int currentColumn = pair.current.firstColumn + offset;
                    const int previousColumn = pair.previous.firstColumn + offset;
                    if (currentColumn < 0 || currentColumn >= currentPoints->cols
                        || previousColumn < 0
                        || previousColumn >= previousPoints->cols)
                        continue;
                    const cv::Vec3f& before =
                        (*previousPoints)(row, previousColumn);
                    const cv::Vec3f& after =
                        (*currentPoints)(row, currentColumn);
                    if (before[0] == -1.0f || after[0] == -1.0f
                        || !std::isfinite(before[0])
                        || !std::isfinite(before[1])
                        || !std::isfinite(before[2])
                        || !std::isfinite(after[0])
                        || !std::isfinite(after[1])
                        || !std::isfinite(after[2]))
                        continue;
                    const cv::Vec3f delta = after - before;
                    const float magnitude = std::sqrt(delta.dot(delta));
                    if (!(magnitude > 1e-6f)
                        || !std::isfinite(magnitude))
                        continue;
                    visitor(row, currentColumn, magnitude);
                }
            }
        }
    };

    float minimumPositive = std::numeric_limits<float>::infinity();
    float maximum = 0.0f;
    std::size_t changedCount = 0;
    forEachMagnitude([&](int, int, float magnitude) {
        minimumPositive = std::min(minimumPositive, magnitude);
        maximum = std::max(maximum, magnitude);
        ++changedCount;
    });

    QImage image(currentPoints->cols, currentPoints->rows, QImage::Format_ARGB32);
    if (image.isNull()) return {};
    image.fill(Qt::transparent);
    if (changedCount == 0) return image;

    // Revisit the compact ranges to build a logarithmic histogram. The extra
    // arithmetic pass is much cheaper than retaining a surface-sized float
    // magnitude matrix.
    constexpr std::size_t histogramSize = 2048;
    std::array<std::size_t, histogramSize> histogram{};
    const double logMinimum = std::log(static_cast<double>(minimumPositive));
    const double logMaximum = std::log(static_cast<double>(maximum));
    const double logSpan = logMaximum - logMinimum;
    forEachMagnitude([&](int, int, float magnitude) {
        std::size_t bin = 0;
        if (logSpan > 1e-12) {
            const double fraction =
                (std::log(static_cast<double>(magnitude)) - logMinimum)
                / logSpan;
            bin = std::min(histogramSize - 1,
                           static_cast<std::size_t>(
                               fraction * histogramSize));
        }
        ++histogram[bin];
    });
    const std::size_t percentileTarget = (changedCount * 95 + 99) / 100;
    std::size_t accumulated = 0;
    std::size_t percentileBin = histogramSize - 1;
    for (std::size_t bin = 0; bin < histogramSize; ++bin) {
        accumulated += histogram[bin];
        if (accumulated >= percentileTarget) {
            percentileBin = bin;
            break;
        }
    }
    const float displayMaximum = logSpan > 1e-12
        ? static_cast<float>(std::exp(logMinimum
              + logSpan * static_cast<double>(percentileBin + 1) / histogramSize))
        : maximum;

    const double displayLogMaximum = std::log(static_cast<double>(displayMaximum));
    const double displayLogSpan = displayLogMaximum - logMinimum;
    forEachMagnitude([&](int row, int col, float magnitude) {
        QRgb* pixels = reinterpret_cast<QRgb*>(image.scanLine(row));
        const float magnitudeFraction = displayLogSpan > 1e-12
            ? static_cast<float>(
                  (std::log(static_cast<double>(magnitude)) - logMinimum)
                  / displayLogSpan)
            : 1.0f;
        pixels[col] = runDiffMagnitudeColor(magnitudeFraction);
    });
    return image;
}

void SpiralWorkspace::updateRunDiffOverlay()
{
    if (_flattenedPreviewActive) {
        _overlay->publishRunDiff({}, {});
        return;
    }
    if (!_runDiffVisible || !_currentPreview || _previewRunDiffImage.isNull()) {
        _overlay->publishRunDiff({}, {});
        return;
    }
    _overlay->publishRunDiff(_currentPreview, _previewRunDiffImage);
}

void SpiralWorkspace::updateLossMapOverlay()
{
    if (_flattenedPreviewActive) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        return;
    }
    const auto found = _previewLossMaps.constFind(_selectedLossMap);
    if (_selectedLossMap.isEmpty() || found == _previewLossMaps.constEnd()
        || !_currentPreview || _currentPreviewComponents.empty()
        || found->imagePath.isEmpty()) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        _panel->setLossMapLegend({});
        return;
    }

    const auto& map = found.value();
    if (_loadedLossMap != map.name) {
        QImage image(map.imagePath);
        if (image.isNull()) {
            _loadedLossMap.clear();
            _loadedLossMapImage = {};
            _overlay->publishLossMap({}, {}, _lossMapOpacity);
            _panel->setLossMapLegend(tr("Could not load loss overlay %1").arg(map.name));
            return;
        }
        _loadedLossMap = map.name;
        _loadedLossMapImage = image.convertToFormat(QImage::Format_ARGB32);
    }
    _panel->setLossMapLegend(
        tr("%1 — weighted residual (weight %2)\n"
           "p50 %3   p95 %4   max %5\n"
           "%6 displayed samples / %7 pixels   %8 projected   "
           "%9 off-surface   %10 omitted   %11 eligible")
            .arg(map.name)
            .arg(map.weight, 0, 'g', 5)
            .arg(map.p50, 0, 'g', 5)
            .arg(map.p95, 0, 'g', 5)
            .arg(map.maximum, 0, 'g', 5)
            .arg(map.sampleCount)
            .arg(map.supportedPixels)
            .arg(map.projectedSampleCount)
            .arg(map.offSurfaceSampleCount)
            .arg(map.omittedSampleCount)
            .arg(map.eligibleSampleCount));
    if (_currentPreview == _previewSource) {
        _overlay->publishLossMap(_currentPreview, _loadedLossMapImage, _lossMapOpacity);
        return;
    }

    std::vector<PreviewComponent> selected;
    for (const PreviewComponent& component : _previewComponents) {
        if (component.winding < _minimumDisplayedWinding) continue;
        if (_maximumDisplayedWinding >= 0 && component.winding > _maximumDisplayedWinding)
            continue;
        selected.push_back(component);
    }
    if (selected.empty()) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        return;
    }
    const int firstColumn = selected.front().firstColumn;
    const int endColumn = selected.back().endColumn;
    if (firstColumn < 0 || endColumn > _loadedLossMapImage.width()
        || firstColumn >= endColumn) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        return;
    }
    _overlay->publishLossMap(
        _currentPreview,
        _loadedLossMapImage.copy(firstColumn, 0, endColumn - firstColumn,
                                 _loadedLossMapImage.height()),
        _lossMapOpacity);
}

std::optional<SpiralWorkspace::PreviewDisplaySelection>
SpiralWorkspace::displayedPreviewSelection() const
{
    if (!_previewSource || _previewComponents.empty()) return std::nullopt;

    std::vector<PreviewComponent> selected;
    for (const PreviewComponent& component : _previewComponents) {
        if (component.winding < _minimumDisplayedWinding) continue;
        if (_maximumDisplayedWinding >= 0 && component.winding > _maximumDisplayedWinding)
            continue;
        selected.push_back(component);
    }
    if (selected.empty()) return std::nullopt;
    const int firstColumn = selected.front().firstColumn;
    const int endColumn = selected.back().endColumn;
    PreviewDisplaySelection selection;
    selection.firstColumn = firstColumn;
    selection.endColumn = endColumn;
    selection.diffComponents = selected;
    for (PreviewComponent& component : selection.diffComponents) {
        component.firstColumn -= firstColumn;
        component.endColumn -= firstColumn;
    }
    if (!_previewConnected) {
        selection.surfaceComponents.reserve(selected.size());
        for (const PreviewComponent& component : selection.diffComponents) {
            selection.surfaceComponents.emplace_back(component.firstColumn,
                                                     component.endColumn);
        }
    }
    selection.registrationId = QStringLiteral("%1-display-%2")
                                   .arg(_previewSourceId)
                                   .arg(_previewDisplayRevision);
    return selection;
}

void SpiralWorkspace::applyPreviewWindingRange(bool preserveFocus)
{
    if (_shuttingDown || !_previewSource || _flattenedPreviewActive) return;
    const quint64 revision = ++_previewDisplayRevision;
    ++_runDiffRequestRevision;
    _previewRunDiffImage = {};
    _currentPreviewComponents.clear();
    _overlay->publishRunDiff({}, {});
    _overlay->publishLossMap({}, {}, _lossMapOpacity);

    const auto selection = displayedPreviewSelection();
    if (!selection) {
        if (_outputVisible) _state->setSurface("segmentation", nullptr);
        const QString previousRegistration = _currentPreviewRegistrationId;
        _currentPreview.reset();
        _currentPreviewComponents.clear();
        _brush->setPaintSurface({});
        _currentPreviewRegistrationId.clear();
        updateSurfaceIntersections();
        if (!previousRegistration.isEmpty())
            _state->setSurface(previousRegistration.toStdString(), nullptr);
        return;
    }

    const auto source = _previewSource;
    const cv::Size sourceSize = source->gridSize();
    if (selection->firstColumn < 0
        || selection->endColumn > sourceSize.width
        || selection->firstColumn >= selection->endColumn
        || sourceSize.height <= 0) {
        statusBar()->showMessage(tr("Spiral preview winding range is out of bounds"),
                                 15000);
        return;
    }
    const qint64 generation = _requestedPreviewGeneration;
    const cv::Rect region(selection->firstColumn, 0,
                          selection->endColumn - selection->firstColumn,
                          sourceSize.height);
    auto* watcher =
        new QFutureWatcher<std::shared_ptr<QuadSurface>>(this);
    connect(watcher,
            &QFutureWatcher<std::shared_ptr<QuadSurface>>::finished,
            this,
            [this, watcher, source, selection = *selection, generation,
             revision, preserveFocus]() {
                const auto preview = watcher->result();
                watcher->deleteLater();
                if (_shuttingDown || source != _previewSource
                    || generation != _requestedPreviewGeneration
                    || revision != _previewDisplayRevision)
                    return;
                if (!preview) {
                    statusBar()->showMessage(
                        tr("Could not load the selected Spiral winding range"),
                        15000);
                    return;
                }
                _state->setSurface(selection.registrationId.toStdString(),
                                   preview);
                installPreviewAliasWhenIndexed(
                    preview, selection.registrationId, generation, revision,
                    preserveFocus, 0, selection.diffComponents);
            });
    watcher->setFuture(QtConcurrent::run(
        [source, selection = *selection, region]() {
            try {
                auto loaded =
                    load_quad_from_tifxyz_region(source->path, region);
                loaded->id = selection.registrationId.toStdString();
                loaded->setComponents(selection.surfaceComponents);
                return std::shared_ptr<QuadSurface>(std::move(loaded));
            } catch (const std::exception&) {
                return std::shared_ptr<QuadSurface>{};
            }
        }));
}

void SpiralWorkspace::installPreviewAliasWhenIndexed(
    const std::shared_ptr<QuadSurface>& preview, const QString& registrationId,
    qint64 generation, quint64 revision, bool preserveFocus, int attempt,
    std::vector<PreviewComponent> diffComponents)
{
    const bool stale = _shuttingDown || generation != _requestedPreviewGeneration
                       || revision != _previewDisplayRevision;
    if (stale) {
        if (_state->surface(registrationId.toStdString()) == preview
            && registrationId != _currentPreviewRegistrationId)
            _state->setSurface(registrationId.toStdString(), nullptr);
        return;
    }
    auto* index = _viewerManager->surfacePatchIndexIfReady();
    if (!index || !index->containsSurface(preview)) {
        if (attempt >= 600) {
            if (_state->surface(registrationId.toStdString()) == preview)
                _state->setSurface(registrationId.toStdString(), nullptr);
            statusBar()->showMessage(tr("Timed out indexing the new Spiral preview; keeping the previous preview"), 15000);
            return;
        }
        QTimer::singleShot(50, this, [this, preview, registrationId, generation,
                                     revision, preserveFocus, attempt,
                                     diffComponents]() {
            installPreviewAliasWhenIndexed(preview, registrationId, generation, revision,
                                           preserveFocus, attempt + 1,
                                           diffComponents);
        });
        return;
    }
    const QString previousRegistration = _currentPreviewRegistrationId;
    _currentPreview = preview;
    _currentPreviewComponents = std::move(diffComponents);
    _brush->setPaintSurface(preview);
    _currentPreviewRegistrationId = registrationId;
    if (_outputVisible) _state->setSurface("segmentation", preview, false, preserveFocus);
    if (_runDiffVisible)
        loadRunDiff();
    else
        updateRunDiffOverlay();
    updateLossMapOverlay();
    // No-op unless the focus is still missing or the automatic default.
    initializePreviewFocus();
    updateSurfaceIntersections();
    for (auto* viewer : _viewerManager->baseViewers()) if (viewer) {
        viewer->invalidateIntersect("segmentation");
        viewer->renderIntersections("Spiral preview installed");
        viewer->requestRender("Spiral preview installed");
    }
    if (!previousRegistration.isEmpty() && previousRegistration != registrationId)
        _state->setSurface(previousRegistration.toStdString(), nullptr);
}

void SpiralWorkspace::loadGeometrySnapshot(const QString& manifestPath, quint64 generation)
{
    auto* watcher = new QFutureWatcher<GeometryLoadResult>(this);
    connect(watcher, &QFutureWatcher<GeometryLoadResult>::finished, this, [this, watcher, generation]() {
        const auto result = watcher->result();
        watcher->deleteLater();
        if (_shuttingDown) return;
        if (!result.index) { statusBar()->showMessage(result.error, 15000); return; }
        _overlay->publishIndex(result.index, generation);
    });
    watcher->setFuture(QtConcurrent::run([manifestPath]() -> GeometryLoadResult {
        try {
            QFile manifestFile(manifestPath);
            if (!manifestFile.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot read geometry manifest");
            const QJsonObject manifest = QJsonDocument::fromJson(manifestFile.readAll()).object();
            if (manifest.value("schema_version").toInt() != 1 || manifest.value("coordinate_order").toString() != "XYZ")
                throw std::runtime_error("Unsupported Spiral geometry snapshot schema");
            const std::filesystem::path root = QFileInfo(manifestPath).absolutePath().toStdString();
            std::vector<PolylineIndex::Polyline> polylines;
            uint64_t objectId = 0;
            const QJsonObject categories = manifest.value("categories").toObject();
            for (auto category = categories.begin(); category != categories.end(); ++category) {
                const QJsonObject entry = category.value().toObject();
                const uint64_t pointCount = static_cast<uint64_t>(entry.value("point_count").toInteger());
                const uint64_t polylineCount = static_cast<uint64_t>(entry.value("polyline_count").toInteger());
                if (pointCount > 1'000'000'000ULL || polylineCount > pointCount)
                    throw std::runtime_error("Geometry snapshot counts exceed limits");
                const auto offsetsPath = root / entry.value("offsets_file").toString().toStdString();
                const auto pointsPath = root / entry.value("points_file").toString().toStdString();
                if (std::filesystem::file_size(offsetsPath) != (polylineCount + 1) * sizeof(uint64_t)
                    || std::filesystem::file_size(pointsPath) != pointCount * 3 * sizeof(float))
                    throw std::runtime_error("Geometry snapshot file size mismatch");
                std::ifstream offsetsStream(offsetsPath, std::ios::binary);
                std::vector<uint64_t> offsets(polylineCount + 1);
                offsetsStream.read(reinterpret_cast<char*>(offsets.data()), static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
                for (auto& offset : offsets) offset = qFromLittleEndian(offset);
                if (offsets.empty() || offsets.front() != 0 || offsets.back() != pointCount)
                    throw std::runtime_error("Geometry snapshot offsets are out of range");
                std::ifstream pointsStream(pointsPath, std::ios::binary);
                for (uint64_t line = 0; line < polylineCount; ++line) {
                    if (offsets[line + 1] <= offsets[line]) throw std::runtime_error("Zero-length geometry polyline");
                    PolylineIndex::Polyline polyline;
                    polyline.objectId = objectId++;
                    polyline.category = category.key().toStdString();
                    polyline.points.resize(static_cast<std::size_t>(offsets[line + 1] - offsets[line]));
                    pointsStream.read(reinterpret_cast<char*>(polyline.points.data()),
                                      static_cast<std::streamsize>(polyline.points.size() * sizeof(cv::Vec3f)));
                    if (!pointsStream) throw std::runtime_error("Truncated geometry point file");
                    if constexpr (std::endian::native == std::endian::big) {
                        for (auto& point : polyline.points) for (int axis = 0; axis < 3; ++axis) {
                            uint32_t bits; std::memcpy(&bits, &point[axis], sizeof(bits));
                            bits = qFromLittleEndian(bits); std::memcpy(&point[axis], &bits, sizeof(bits));
                        }
                    }
                    for (const auto& point : polyline.points)
                        for (int axis = 0; axis < 3; ++axis)
                            if (!std::isfinite(point[axis]))
                                throw std::runtime_error("Geometry snapshot contains a non-finite coordinate");
                    polylines.push_back(std::move(polyline));
                }
            }
            auto index = std::make_shared<PolylineIndex>();
            index->build(std::move(polylines), 2.0f);
            return {index, {}};
        } catch (const std::exception& error) {
            return {{}, QString::fromUtf8(error.what())};
        }
    }));
}
