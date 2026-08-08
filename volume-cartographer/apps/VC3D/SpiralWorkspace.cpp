#include "SpiralWorkspace.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ConsoleOutputWidget.hpp"
#include "Keybinds.hpp"
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

#include <opencv2/imgcodecs.hpp>

#include <QDialog>
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
#include <QRegularExpression>
#include <QScopedValueRollback>
#include <QSettings>
#include <QSaveFile>
#include <QShortcut>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QWindow>
#include <QtConcurrent/QtConcurrent>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

namespace {

QImage maskOverlayToSurface(
    QImage image, const std::shared_ptr<QuadSurface>& surface)
{
    if (image.isNull() || !surface) return {};
    image = image.convertToFormat(QImage::Format_ARGB32);
    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->cols != image.width() || points->rows != image.height())
        return {};
    for (int row = 0; row < points->rows; ++row) {
        QRgb* pixels = reinterpret_cast<QRgb*>(image.scanLine(row));
        for (int column = 0; column < points->cols; ++column) {
            if ((*points)(row, column)[0] == -1.0f)
                pixels[column] = qRgba(0, 0, 0, 0);
        }
    }
    return image;
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
    // Spiral's plane panes get their own hard-capped decoded-chunk pool instead
    // of sharing the Volume's, so browsing slices here can never displace the
    // main workspace's working set, and the flattened pane renders from tiles of
    // resampled surface space. Applied before the viewers exist so they pick up
    // the policy on construction.
    _viewerManager->applySpiralCacheSettings();
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

    _pythonOutputDialog = new QDialog(this, Qt::Window);
    _pythonOutputDialog->setObjectName(QStringLiteral("spiralPythonOutputDialog"));
    _pythonOutputDialog->setWindowTitle(tr("Spiral Python Output"));
    _pythonOutputDialog->setModal(false);
    _pythonOutputDialog->resize(900, 500);
    auto* pythonOutputLayout = new QVBoxLayout(_pythonOutputDialog);
    _pythonOutput = new ConsoleOutputWidget(_pythonOutputDialog);
    _pythonOutput->setTitle(tr("Spiral Python stdout / stderr"));
    // A fit can run long enough to produce many tqdm redraws.  Retain enough
    // blocks that the loading-stage bars and early fitter diagnostics remain
    // available alongside the live fit-stage bar.
    _pythonOutput->setMaximumBlockCount(100000);
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
    connect(_service, &SpiralServiceManager::previewAvailable, this, &SpiralWorkspace::loadPreview);
    connect(_service, &SpiralServiceManager::connectionStateChanged, this,
            [this](SpiralServiceManager::ConnectionState state, const QString&) {
                using CS = SpiralServiceManager::ConnectionState;
                if (state == CS::Starting || state == CS::Connecting) {
                    _requestedPreviewGeneration = -1;
                    _inputSurfaceGeneration = 0;
                    _pendingPatchIds.clear();
                    _previewRunDiffImagePath.clear();
                    ++_runDiffRequestRevision;
                    _previewRunDiffImage = {};
                    _previewLossMaps.clear();
                    _fetchingLossMaps.clear();
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
    connect(_service, &SpiralServiceManager::sessionSynchronized, this,
            [this](const QJsonObject& request, const QJsonObject& status) {
                const QJsonObject paths =
                    request.value(QStringLiteral("paths")).toObject();
                const qint64 generation =
                    status.value(QStringLiteral("session_generation")).toInteger();
                _sessionPaths = paths;
                _previewSource.reset();
                _previewComponents.clear();
                _previewWindingIds.release();
                _previewRunDiffImagePath.clear();
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
                ++_runDiffRequestRevision;
                _previewRunDiffImage = {};
                _previewLossMaps.clear();
                _fetchingLossMaps.clear();
                _loadedLossMap.clear();
                _loadedLossMapImage = {};
                _selectedLossMap.clear();
                _panel->setLossMapOptions({});
                _panel->setLossMapLegend({});
                _overlay->publishRunDiff({}, {});
                _overlay->publishLossMap({}, {}, _lossMapOpacity);
                loadInputSurfaces(paths, static_cast<quint64>(generation));
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

bool SpiralWorkspace::supportsUnverifiedPatchUploads() const
{
    return _service && _service->supportsPatchClassification();
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

void SpiralWorkspace::addUnverifiedPatchToCurrentFit(
    const QString& tifxyzDirectory, const std::shared_ptr<QuadSurface>& surface)
{
    if (!_service) return;
    const QString inputId = QFileInfo(tifxyzDirectory).fileName();
    if (!_service->supportsPatchClassification()) {
        statusBar()->showMessage(
            tr("The connected Spiral service cannot preserve unverified patch classification; update it before adding %1")
                .arg(inputId),
            15000);
        return;
    }
    registerPendingPatchSurface(inputId, surface);
    statusBar()->showMessage(
        tr("Uploading %1 as an unverified Spiral hint\u2026").arg(inputId));
    _service->uploadPatch(
        tifxyzDirectory, inputId,
        SpiralServiceManager::PatchClassification::Unverified);
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
    if (_viewerManager) _viewerManager->beginShutdown();
    // Disconnecting never terminates a service VC3D did not launch; only an
    // owned local process is stopped.
    if (_service) _service->disconnectFromService();
    if (_state) {
        _state->setVpkg(nullptr); // the package is borrowed from Main; drop it so closeAll() cannot unload Main's surfaces
        _state->closeAll();
    }
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
        auto failure = [](const QString& message) {
            PreviewLoadResult result;
            result.error = message;
            return result;
        };
        QFile file(manifestPath);
        if (!file.open(QIODevice::ReadOnly))
            return failure(QObject::tr("Cannot read Spiral preview manifest"));
        const QJsonObject manifest = QJsonDocument::fromJson(file.readAll()).object();
        const int schemaVersion = manifest.value(QStringLiteral("schema_version")).toInt();
        if (schemaVersion != 3
            || manifest.value(QStringLiteral("kind")).toString() != QStringLiteral("spiral_combined_preview"))
            return failure(QObject::tr("Unsupported Spiral preview manifest"));
        QString surfacePath = manifest.value(QStringLiteral("surface_path")).toString();
        const QString surfaceId = manifest.value(QStringLiteral("surface_id")).toString();
        if (surfacePath.isEmpty() || surfaceId.isEmpty())
            return failure(QObject::tr("Malformed Spiral preview manifest"));
        // The manifest's surface_path is a service-host path; a cache-resident
        // artifact keeps the surface directory (named by its id) beside the
        // manifest, so prefer that local layout when it exists.
        const QString localSurfacePath = QDir(QFileInfo(manifestPath).absolutePath()).filePath(surfaceId);
        if (QFileInfo(QDir(localSurfacePath).filePath(QStringLiteral("meta.json"))).isFile())
            surfacePath = localSurfacePath;

        const QJsonArray bounds = manifest.value(QStringLiteral("winding_bounds")).toArray();
        const QJsonArray windingIds = manifest.value(QStringLiteral("winding_ids")).toArray();
        if (bounds.isEmpty() || bounds.size() != windingIds.size()
            || std::abs(manifest.value(QStringLiteral("output_step_vx")).toDouble()
                        - 20.0) > 1.0e-9)
            return failure(QObject::tr("Invalid Spiral preview winding mapping"));
        std::vector<PreviewComponent> previewComponents;
        std::set<int> expectedWindingIds;
        previewComponents.reserve(bounds.size());
        for (int index = 0; index < bounds.size(); ++index) {
            const QJsonObject entry = bounds[index].toObject();
            const PreviewComponent component{
                entry.value(QStringLiteral("row_begin")).toInt(-1),
                entry.value(QStringLiteral("row_end")).toInt(-1),
                entry.value(QStringLiteral("column_begin")).toInt(-1),
                entry.value(QStringLiteral("column_end")).toInt(-1),
                entry.value(QStringLiteral("winding")).toInt(-1),
            };
            if (component.rowBegin < 0 || component.rowEnd <= component.rowBegin
                || component.columnBegin < 0
                || component.columnEnd <= component.columnBegin
                || component.winding != windingIds[index].toInt()
                || component.winding < 0
                || !expectedWindingIds.insert(component.winding).second)
                return failure(QObject::tr("Invalid Spiral winding bounds"));
            previewComponents.push_back(component);
        }

        const QString artifactRoot = QFileInfo(manifestPath).absolutePath();
        const QString windingMapName =
            QDir::cleanPath(manifest.value(QStringLiteral("winding_id_map")).toString());
        if (windingMapName.isEmpty() || QDir::isAbsolutePath(windingMapName)
            || windingMapName == QStringLiteral("..")
            || windingMapName.startsWith(QStringLiteral("../")))
            return failure(QObject::tr("Invalid Spiral winding-ID map path"));
        const QString windingMapPath = QDir(artifactRoot).filePath(windingMapName);
        cv::Mat windingImage = cv::imread(
            windingMapPath.toStdString(), cv::IMREAD_UNCHANGED);
        if (windingImage.empty() || windingImage.channels() != 1)
            return failure(QObject::tr("Cannot read Spiral winding-ID map"));
        cv::Mat windingValues;
        windingImage.convertTo(windingValues, CV_64F);
        cv::Mat_<int32_t> mappedWindings;
        mappedWindings.create(windingImage.rows, windingImage.cols);
        std::map<int, PreviewComponent> actualBounds;
        for (int row = 0; row < windingValues.rows; ++row) {
            const double* values = windingValues.ptr<double>(row);
            for (int column = 0; column < windingValues.cols; ++column) {
                const double value = values[column];
                const double rounded = std::nearbyint(value);
                if (!std::isfinite(value) || std::abs(value - rounded) > 1.0e-6
                    || rounded < -1.0
                    || rounded > std::numeric_limits<int32_t>::max())
                    return failure(QObject::tr(
                        "Spiral winding-ID map contains a non-integer value"));
                const int winding = static_cast<int>(rounded);
                mappedWindings(row, column) = winding;
                if (winding == -1) continue;
                if (expectedWindingIds.find(winding)
                    == expectedWindingIds.end())
                    return failure(QObject::tr(
                        "Spiral winding-ID map contains an undeclared winding"));
                auto [found, inserted] = actualBounds.try_emplace(
                    winding,
                    PreviewComponent{
                        row, row + 1, column, column + 1, winding});
                if (!inserted) {
                    found->second.rowBegin =
                        std::min(found->second.rowBegin, row);
                    found->second.rowEnd =
                        std::max(found->second.rowEnd, row + 1);
                    found->second.columnBegin =
                        std::min(found->second.columnBegin, column);
                    found->second.columnEnd =
                        std::max(found->second.columnEnd, column + 1);
                }
            }
        }
        if (actualBounds.size() != previewComponents.size())
            return failure(QObject::tr(
                "Spiral winding-ID map omits a declared winding"));
        for (const PreviewComponent& expected : previewComponents) {
            const auto found = actualBounds.find(expected.winding);
            if (found == actualBounds.end()
                || found->second.rowBegin != expected.rowBegin
                || found->second.rowEnd != expected.rowEnd
                || found->second.columnBegin != expected.columnBegin
                || found->second.columnEnd != expected.columnEnd)
                return failure(QObject::tr(
                    "Spiral winding bounds do not match the winding-ID map"));
        }

        QFile metadata(QDir(surfacePath).filePath(QStringLiteral("meta.json")));
        if (!metadata.open(QIODevice::ReadOnly))
            return failure(QObject::tr("Spiral preview surface metadata is missing"));
        const QJsonObject meta = QJsonDocument::fromJson(metadata.readAll()).object();
        if (meta.value(QStringLiteral("winding_id_map"))
                != manifest.value(QStringLiteral("winding_id_map"))
            || meta.value(QStringLiteral("winding_bounds"))
                != manifest.value(QStringLiteral("winding_bounds"))
            || meta.value(QStringLiteral("component_winding_ids"))
                != manifest.value(QStringLiteral("winding_ids"))
            || meta.value(QStringLiteral("grid_shape"))
                != manifest.value(QStringLiteral("grid_shape"))
            || meta.value(QStringLiteral("output_step_vx"))
                != manifest.value(QStringLiteral("output_step_vx"))
            || meta.value(QStringLiteral("uuid")).toString() != surfaceId)
            return failure(QObject::tr(
                "Spiral preview metadata does not match its generation manifest"));
        try {
            // Keep the artifact as a lazy descriptor. Only the selected winding
            // columns are decoded when applyPreviewWindingRange() builds the
            // compact display surface.
            auto surface = std::make_shared<QuadSurface>(surfacePath.toStdString());
            surface->id = surfaceId.toStdString();
            surface->setStrictQuadRenderValidity(true);
            const cv::Size gridSize = surface->gridSize();
            const QJsonArray gridShape =
                manifest.value(QStringLiteral("grid_shape")).toArray();
            if (mappedWindings.cols != gridSize.width
                || mappedWindings.rows != gridSize.height
                || gridShape.size() != 2
                || gridShape[0].toInt(-1) != gridSize.height
                || gridShape[1].toInt(-1) != gridSize.width)
                return failure(QObject::tr(
                    "Spiral winding-ID map dimensions do not match the surface"));
            for (const PreviewComponent& component : previewComponents) {
                if (component.rowEnd > gridSize.height
                    || component.columnEnd > gridSize.width)
                    return failure(QObject::tr(
                        "Spiral winding bounds exceed the surface grid"));
            }
            std::vector<PreviewLoadResult::LossMap> lossMaps;
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
                PreviewLoadResult::LossMap map;
                map.name = name;
                map.relativePath = relativePath;
                const QString imagePath = QDir(artifactRoot).filePath(relativePath);
                if (QFileInfo::exists(imagePath)) map.imagePath = imagePath;
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
            QString runDiffPath;
            const QString runDiffRelative = QDir::cleanPath(
                manifest.value(QStringLiteral("run_diff")).toObject()
                    .value(QStringLiteral("path")).toString());
            if (!runDiffRelative.isEmpty()
                && !QDir::isAbsolutePath(runDiffRelative)
                && runDiffRelative != QStringLiteral("..")
                && !runDiffRelative.startsWith(QStringLiteral("../"))) {
                const QString candidate = QDir(artifactRoot).filePath(runDiffRelative);
                QImageReader reader(candidate);
                if (reader.canRead() && reader.size().width() == gridSize.width
                    && reader.size().height() == gridSize.height)
                    runDiffPath = candidate;
            }
            PreviewLoadResult result;
            result.surface = std::move(surface);
            result.surfaceId = surfaceId;
            result.components = std::move(previewComponents);
            result.windingIds = std::move(mappedWindings);
            result.lossMaps = std::move(lossMaps);
            result.runDiffImagePath = runDiffPath;
            return result;
        } catch (const std::exception& error) {
            return failure(QString::fromUtf8(error.what()));
        }
    }));
}

void SpiralWorkspace::installPreview(const PreviewLoadResult& result, qint64 generation)
{
    if (!result.surface) { statusBar()->showMessage(result.error, 15000); return; }
    _previewSource = result.surface;
    _previewSourceId = result.surfaceId;
    _previewComponents = result.components;
    _previewWindingIds = result.windingIds;
    _previewRunDiffImagePath = result.runDiffImagePath;
    ++_runDiffRequestRevision;
    _previewRunDiffImage = {};
    _previewLossMaps.clear();
    _fetchingLossMaps.clear();
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
    if (_runDiffVisible) loadRunDiff();
}

void SpiralWorkspace::loadRunDiff()
{
    const quint64 requestRevision = ++_runDiffRequestRevision;
    _previewRunDiffImage = {};
    if (!_runDiffVisible || !_currentPreview
        || _previewRunDiffImagePath.isEmpty()) {
        updateRunDiffOverlay();
        return;
    }
    const auto current = _currentPreview;
    const QString imagePath = _previewRunDiffImagePath;
    const auto selection = displayedPreviewSelection();
    if (!selection) {
        updateRunDiffOverlay();
        return;
    }
    auto* watcher = new QFutureWatcher<QImage>(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, this,
            [this, watcher, current, requestRevision]() {
                const QImage image = watcher->result();
                watcher->deleteLater();
                if (_shuttingDown || !_runDiffVisible
                    || requestRevision != _runDiffRequestRevision
                    || current != _currentPreview)
                    return;
                _previewRunDiffImage = image;
                updateRunDiffOverlay();
            });
    watcher->setFuture(QtConcurrent::run([imagePath, region = selection->region]() {
        const QImage image(imagePath);
        if (image.isNull()) return QImage{};
        return image.copy(region.x, region.y, region.width, region.height)
            .convertToFormat(QImage::Format_ARGB32);
    }));
}

void SpiralWorkspace::updateRunDiffOverlay()
{
    if (!_runDiffVisible || !_currentPreview || _previewRunDiffImage.isNull()) {
        _overlay->publishRunDiff({}, {});
        return;
    }
    _overlay->publishRunDiff(
        _currentPreview,
        maskOverlayToSurface(_previewRunDiffImage, _currentPreview));
}

void SpiralWorkspace::updateLossMapOverlay()
{
    auto found = _previewLossMaps.find(_selectedLossMap);
    if (_selectedLossMap.isEmpty() || found == _previewLossMaps.end()
        || !_currentPreview) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        _panel->setLossMapLegend({});
        return;
    }

    auto& map = found.value();
    if (map.imagePath.isEmpty()) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        if (_fetchingLossMaps.contains(map.name)) {
            _panel->setLossMapLegend(tr("Downloading loss overlay %1…").arg(map.name));
            return;
        }
        const QString requestedName = map.name;
        const QString relativePath = map.relativePath;
        const qint64 previewGeneration = _requestedPreviewGeneration;
        _fetchingLossMaps.insert(requestedName);
        _panel->setLossMapLegend(tr("Downloading loss overlay %1…").arg(requestedName));
        _service->fetchPreviewFile(
            relativePath,
            [this, requestedName, relativePath, previewGeneration](
                const QString& localPath, const QString& error) {
                if (_shuttingDown
                    || previewGeneration != _requestedPreviewGeneration)
                    return;
                _fetchingLossMaps.remove(requestedName);
                if (localPath.isEmpty()) {
                    if (requestedName == _selectedLossMap)
                        _panel->setLossMapLegend(
                            tr("Could not download loss overlay %1: %2")
                                .arg(requestedName, error));
                    return;
                }
                QImageReader reader(localPath);
                const cv::Size expected =
                    _previewSource ? _previewSource->gridSize() : cv::Size{};
                if (!reader.canRead() || reader.size().width() != expected.width
                    || reader.size().height() != expected.height) {
                    if (requestedName == _selectedLossMap)
                        _panel->setLossMapLegend(
                            tr("Downloaded loss overlay %1 has invalid dimensions")
                                .arg(requestedName));
                    return;
                }
                auto current = _previewLossMaps.find(requestedName);
                if (current == _previewLossMaps.end()
                    || current->relativePath != relativePath)
                    return;
                current->imagePath = localPath;
                if (requestedName == _selectedLossMap) updateLossMapOverlay();
            });
        return;
    }
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
    const auto selection = displayedPreviewSelection();
    if (!selection) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        return;
    }
    const cv::Rect& region = selection->region;
    if (region.x < 0 || region.y < 0
        || region.x + region.width > _loadedLossMapImage.width()
        || region.y + region.height > _loadedLossMapImage.height()) {
        _overlay->publishLossMap({}, {}, _lossMapOpacity);
        return;
    }
    _overlay->publishLossMap(
        _currentPreview,
        maskOverlayToSurface(
            _loadedLossMapImage.copy(
                region.x, region.y, region.width, region.height),
            _currentPreview),
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
    int rowBegin = std::numeric_limits<int>::max();
    int rowEnd = 0;
    int columnBegin = std::numeric_limits<int>::max();
    int columnEnd = 0;
    for (const PreviewComponent& component : selected) {
        rowBegin = std::min(rowBegin, component.rowBegin);
        rowEnd = std::max(rowEnd, component.rowEnd);
        columnBegin = std::min(columnBegin, component.columnBegin);
        columnEnd = std::max(columnEnd, component.columnEnd);
    }
    PreviewDisplaySelection selection;
    selection.region = cv::Rect(
        columnBegin, rowBegin, columnEnd - columnBegin, rowEnd - rowBegin);
    selection.minimumWinding = _minimumDisplayedWinding;
    selection.maximumWinding = _maximumDisplayedWinding;
    selection.registrationId = QStringLiteral("%1-display-%2")
                                   .arg(_previewSourceId)
                                   .arg(_previewDisplayRevision);
    return selection;
}

void SpiralWorkspace::applyPreviewWindingRange(bool preserveFocus)
{
    if (_shuttingDown || !_previewSource) return;
    const quint64 revision = ++_previewDisplayRevision;
    ++_runDiffRequestRevision;
    _previewRunDiffImage = {};
    _overlay->publishRunDiff({}, {});
    _overlay->publishLossMap({}, {}, _lossMapOpacity);

    const auto selection = displayedPreviewSelection();
    if (!selection) {
        if (_outputVisible) _state->setSurface("segmentation", nullptr);
        const QString previousRegistration = _currentPreviewRegistrationId;
        _currentPreview.reset();
        _brush->setPaintSurface({});
        _currentPreviewRegistrationId.clear();
        updateSurfaceIntersections();
        if (!previousRegistration.isEmpty())
            _state->setSurface(previousRegistration.toStdString(), nullptr);
        return;
    }

    const auto source = _previewSource;
    const cv::Size sourceSize = source->gridSize();
    if (selection->region.x < 0 || selection->region.y < 0
        || selection->region.width <= 0 || selection->region.height <= 0
        || selection->region.x + selection->region.width > sourceSize.width
        || selection->region.y + selection->region.height > sourceSize.height) {
        statusBar()->showMessage(tr("Spiral preview winding range is out of bounds"),
                                 15000);
        return;
    }
    const qint64 generation = _requestedPreviewGeneration;
    const cv::Rect region = selection->region;
    const cv::Mat_<int32_t> windingRegion =
        _previewWindingIds(region).clone();
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
                    preserveFocus, 0);
            });
    watcher->setFuture(QtConcurrent::run(
        [source, selection = *selection, region, windingRegion]() {
            try {
                auto loaded =
                    load_quad_from_tifxyz_region(source->path, region);
                loaded->id = selection.registrationId.toStdString();
                cv::Mat_<cv::Vec3f>* points = loaded->rawPointsPtr();
                for (int row = 0; row < points->rows; ++row) {
                    for (int column = 0; column < points->cols; ++column) {
                        const int winding = windingRegion(row, column);
                        if (winding < selection.minimumWinding
                            || (selection.maximumWinding >= 0
                                && winding > selection.maximumWinding)) {
                            (*points)(row, column) =
                                cv::Vec3f(-1.0f, -1.0f, -1.0f);
                        }
                    }
                }
                loaded->setStrictQuadRenderValidity(true);
                return std::shared_ptr<QuadSurface>(std::move(loaded));
            } catch (const std::exception&) {
                return std::shared_ptr<QuadSurface>{};
            }
        }));
}

void SpiralWorkspace::installPreviewAliasWhenIndexed(
    const std::shared_ptr<QuadSurface>& preview, const QString& registrationId,
    qint64 generation, quint64 revision, bool preserveFocus, int attempt)
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
                                     revision, preserveFocus, attempt]() {
            installPreviewAliasWhenIndexed(preview, registrationId, generation, revision,
                                           preserveFocus, attempt + 1);
        });
        return;
    }
    const QString previousRegistration = _currentPreviewRegistrationId;
    _currentPreview = preview;
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
