#include "SpiralWorkspace.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ConsoleOutputWidget.hpp"
#include "Keybinds.hpp"
#include "SpiralPanel.hpp"
#include "SpiralServiceManager.hpp"
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
#include <QDockWidget>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QSettings>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>
#include <QtEndian>
#include <QtConcurrent/QtConcurrent>

#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>

namespace {
cv::Vec3b surfaceCategoryColor(const QString& category)
{
    if (category == QStringLiteral("verified")) return {80, 220, 80};
    if (category == QStringLiteral("unverified")) return {60, 180, 255};
    return {255, 180, 80};
}
}

SpiralWorkspace::SpiralWorkspace(CState* mainState, QWidget* parent)
    : QMainWindow(parent), _mainState(mainState)
{
    setObjectName(QStringLiteral("spiralWorkspaceWindow"));
    setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks | QMainWindow::AllowTabbedDocks);
    // Mirror Main's cache budget so sharedChunkCacheForVolume() resolves to the
    // same ChunkCache instance for the same volume (the budget is part of the
    // cache key). Volume::setCacheBudget is a no-op for an unchanged budget, so
    // re-applying it here never resets the warm cache.
    _state = new CState(_mainState ? _mainState->cacheSizeBytes() : 0, this);
    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
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
    _pythonOutput->setMaximumBlockCount(10000);
    pythonOutputLayout->addWidget(_pythonOutput);
    connect(_service, &SpiralServiceManager::logMessage, _pythonOutput,
            [this](const QString& message) {
                const QString line = message.trimmed();
                const bool routineStatusPoll =
                    line.startsWith(QStringLiteral("SPIRAL_HTTP \"GET /session/status HTTP/"))
                    && line.endsWith(QStringLiteral("\" 200 -"));
                if (!routineStatusPoll) _pythonOutput->appendOutput(message);
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
            });

    _panel = new SpiralPanel(_service, this);
    auto* dock = new QDockWidget(tr("Spiral"), this);
    dock->setObjectName(QStringLiteral("spiralControlDock"));
    dock->setFeatures(QDockWidget::NoDockWidgetFeatures);
    dock->setWidget(_panel);
    addDockWidget(Qt::LeftDockWidgetArea, dock);
    resizeDocks({dock}, {390}, Qt::Horizontal);

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
        } else if (_surfaceCategoryVisible.contains(category)) {
            setSurfaceCategoryVisible(category, shown);
        }
    });
    connect(_service, &SpiralServiceManager::previewAvailable, this, &SpiralWorkspace::loadPreview);
    connect(_service, &SpiralServiceManager::connectionStateChanged, this,
            [this](SpiralServiceManager::ConnectionState state, const QString&) {
                using CS = SpiralServiceManager::ConnectionState;
                if (state == CS::Starting || state == CS::Connecting) {
                    _requestedPreviewGeneration = -1;
                    _inputSurfaceGeneration = 0;
                    _geometryManifestPath.clear();
                    _overlay->reset();
                }
                if (state == CS::Ready && !_service->ownsProcess())
                    _pythonOutput->appendOutput(
                        tr("Connected to a service VC3D does not own; the service's "
                           "Python logs are available on the service host."));
            });
    connect(_service, &SpiralServiceManager::sessionActiveChanged, this,
            &SpiralWorkspace::spiralSessionActiveChanged);
    connect(_service, &SpiralServiceManager::sessionAccepted, this,
            [this](const QJsonObject& paths, qint64 generation) {
                loadInputSurfaces(paths, static_cast<quint64>(generation));
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
                    auto loaded = load_quad_from_tifxyz(candidates[index].toStdString());
                    auto surface = std::shared_ptr<QuadSurface>(std::move(loaded));
                    const QString id = QStringLiteral("spiral/%1/g%2/%3-%4")
                        .arg(category).arg(generation)
                        .arg(QFileInfo(candidates[index]).fileName()).arg(index);
                    surface->id = id.toStdString();
                    result.surfaces.push_back({category, id, std::move(surface)});
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
    for (const auto& entry : result.surfaces) {
        _state->setSurface(entry.id.toStdString(), entry.surface);
        replacement[entry.category].push_back(entry.id);
    }
    const auto retired = _surfaceCategoryIds;
    _surfaceCategoryIds = replacement;
    updateSurfaceIntersections();
    QTimer::singleShot(0, this, [this, retired]() {
        if (_shuttingDown) return;
        for (auto it = retired.begin(); it != retired.end(); ++it)
            for (const QString& id : it.value())
                _state->setSurface(id.toStdString(), nullptr);
    });
    if (!result.warnings.isEmpty())
        statusBar()->showMessage(result.warnings.join(QStringLiteral("; ")), 15000);
}

void SpiralWorkspace::setSurfaceCategoryVisible(const QString& category, bool visible)
{
    if (!_surfaceCategoryVisible.contains(category)) return;
    _surfaceCategoryVisible[category] = visible;
    updateSurfaceIntersections();
}

void SpiralWorkspace::updateSurfaceIntersections()
{
    std::set<std::string> intersections;
    std::map<std::string, cv::Vec3b> surfaceOverlays;
    if (_outputVisible && _currentPreview) intersections.insert("segmentation");
    for (auto visible = _surfaceCategoryVisible.begin(); visible != _surfaceCategoryVisible.end(); ++visible) {
        if (!visible.value()) continue;
        const cv::Vec3b color = surfaceCategoryColor(visible.key());
        for (const QString& id : _surfaceCategoryIds.value(visible.key())) {
            intersections.insert(id.toStdString());
            surfaceOverlays.emplace(id.toStdString(), color);
        }
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (!viewer) continue;
        if (viewer->surfName() == "segmentation") {
            viewer->setSurfaceOverlays(surfaceOverlays);
            viewer->setSurfaceOverlayEnabled(!surfaceOverlays.empty());
            viewer->requestRender("Spiral surface overlays changed");
            continue;
        }
        viewer->setIntersects(intersections);
        viewer->requestRender("Spiral surface visibility changed");
    }
}

bool SpiralWorkspace::hasActiveSpiralSession() const
{
    return _service && _service->hasActiveSession();
}

void SpiralWorkspace::addPatchToCurrentFit(const QString& tifxyzDirectory)
{
    if (!_service) return;
    const QString inputId = QFileInfo(tifxyzDirectory).fileName();
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
        QFile file(manifestPath);
        if (!file.open(QIODevice::ReadOnly)) return {{}, {}, QObject::tr("Cannot read Spiral preview manifest")};
        const QJsonObject manifest = QJsonDocument::fromJson(file.readAll()).object();
        if (manifest.value(QStringLiteral("schema_version")).toInt() != 1
            || manifest.value(QStringLiteral("kind")).toString() != QStringLiteral("spiral_combined_preview"))
            return {{}, {}, QObject::tr("Unsupported Spiral preview manifest")};
        QString surfacePath = manifest.value(QStringLiteral("surface_path")).toString();
        const QString surfaceId = manifest.value(QStringLiteral("surface_id")).toString();
        if (surfacePath.isEmpty() || surfaceId.isEmpty()) return {{}, {}, QObject::tr("Malformed Spiral preview manifest")};
        // The manifest's surface_path is a service-host path; a cache-resident
        // artifact keeps the surface directory (named by its id) beside the
        // manifest, so prefer that local layout when it exists.
        const QString localSurfacePath = QDir(QFileInfo(manifestPath).absolutePath()).filePath(surfaceId);
        if (QFileInfo(QDir(localSurfacePath).filePath(QStringLiteral("meta.json"))).isFile())
            surfacePath = localSurfacePath;
        const QJsonArray components = manifest.value(QStringLiteral("components")).toArray();
        const QJsonArray windingIds = manifest.value(QStringLiteral("winding_ids")).toArray();
        if (components.isEmpty() || components.size() != windingIds.size()
            || windingIds.at(0).toInt() != 10)
            return {{}, {}, QObject::tr("Invalid Spiral preview components/winding mapping")};
        int previousEnd = -1;
        for (int index = 0; index < components.size(); ++index) {
            const QJsonArray range = components[index].toArray();
            if (range.size() != 2 || range[0].toInt() <= previousEnd
                || range[1].toInt() <= range[0].toInt()
                || windingIds[index].toInt() != 10 + index)
                return {{}, {}, QObject::tr("Invalid disconnected Spiral component range")};
            previousEnd = range[1].toInt();
        }
        QFile metadata(QDir(surfacePath).filePath(QStringLiteral("meta.json")));
        if (!metadata.open(QIODevice::ReadOnly))
            return {{}, {}, QObject::tr("Spiral preview surface metadata is missing")};
        const QJsonObject meta = QJsonDocument::fromJson(metadata.readAll()).object();
        if (meta.value(QStringLiteral("components")) != manifest.value(QStringLiteral("components"))
            || meta.value(QStringLiteral("component_winding_ids")) != manifest.value(QStringLiteral("winding_ids"))
            || meta.value(QStringLiteral("uuid")).toString() != surfaceId)
            return {{}, {}, QObject::tr("Spiral preview metadata does not match its generation manifest")};
        try {
            auto loaded = load_quad_from_tifxyz(surfacePath.toStdString());
            auto surface = std::shared_ptr<QuadSurface>(std::move(loaded));
            surface->id = surfaceId.toStdString();
            return {surface, surfaceId, {}};
        } catch (const std::exception& error) {
            return {{}, {}, QString::fromUtf8(error.what())};
        }
    }));
}

void SpiralWorkspace::installPreview(const PreviewLoadResult& result, qint64 generation)
{
    if (!result.surface) { statusBar()->showMessage(result.error, 15000); return; }
    _state->setSurface(result.surfaceId.toStdString(), result.surface);
    installPreviewAliasWhenIndexed(result, generation, 0);
}

void SpiralWorkspace::installPreviewAliasWhenIndexed(const PreviewLoadResult& result,
                                                      qint64 generation, int attempt)
{
    if (_shuttingDown || generation != _requestedPreviewGeneration) return;
    auto* index = _viewerManager->surfacePatchIndexIfReady();
    if (!index || !index->containsSurface(result.surface)) {
        if (attempt >= 600) {
            _state->setSurface(result.surfaceId.toStdString(), nullptr);
            statusBar()->showMessage(tr("Timed out indexing the new Spiral preview; keeping the previous preview"), 15000);
            return;
        }
        QTimer::singleShot(50, this, [this, result, generation, attempt]() {
            installPreviewAliasWhenIndexed(result, generation, attempt + 1);
        });
        return;
    }
    auto previous = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (previous) _retiredPreviews.emplace_back(QString::fromStdString(previous->id), previous);
    _currentPreview = result.surface;
    if (_outputVisible) _state->setSurface("segmentation", result.surface);
    // No-op unless the focus is still missing or the automatic default.
    initializePreviewFocus();
    updateSurfaceIntersections();
    for (auto* viewer : _viewerManager->baseViewers()) if (viewer) {
        viewer->invalidateIntersect("segmentation");
        viewer->requestRender("Spiral preview installed");
    }
    while (_retiredPreviews.size() > 2) {
        const auto retired = _retiredPreviews.front();
        _retiredPreviews.erase(_retiredPreviews.begin());
        _state->setSurface(retired.first.toStdString(), nullptr);
    }
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
