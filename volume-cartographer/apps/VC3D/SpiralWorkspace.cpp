#include "SpiralWorkspace.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ConsoleOutputWidget.hpp"
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
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QCursor>
#include <QDialog>
#include <QDockWidget>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
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
    _state = new CState(0, this); // zero budget: never resets the shared Volume cache
    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
    _slices = std::make_unique<AxisAlignedSliceController>(_state, this);
    _slices->setViewerManager(_viewerManager.get());
    _slices->setEnabled(true);
    _slices->applyOrientation();
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
    for (int pane = 0; pane < 4; ++pane) {
        auto* viewer = _viewerManager->createViewerInWidget(specs[pane].surface, _grid);
        viewer->setIntersects(specs[pane].intersects);
        if (auto* chunkedViewer = dynamic_cast<CChunkedVolumeViewer*>(viewer)) {
            connect(chunkedViewer, &CChunkedVolumeViewer::sendVolumeClicked, this,
                    [this](cv::Vec3f position, cv::Vec3f normal, Surface* surface,
                           Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                if (button != Qt::LeftButton ||
                    modifiers.testFlag(Qt::ShiftModifier) ||
                    !modifiers.testFlag(Qt::ControlModifier)) {
                    return;
                }
                const std::string surfaceId =
                    _state && surface ? _state->findSurfaceId(surface) : std::string{};
                setFocusAt(position, normal, surfaceId);
            });
        }
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
    connect(_service, &SpiralServiceManager::logMessage,
            _pythonOutput, &ConsoleOutputWidget::appendOutput);
    connect(_service, &SpiralServiceManager::errorOccurred, this, [this](const QString& error) {
        statusBar()->showMessage(error, 15000);
        _pythonOutput->appendOutput(tr("Error: %1").arg(error));
        _pythonOutputDialog->show();
        _pythonOutputDialog->raise();
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
    connect(_service, &SpiralServiceManager::serviceStateChanged, this, [this](const QString& state) {
        if (state == tr("Starting")) {
            _requestedPreviewGeneration = -1;
            _inputSurfaceGeneration = 0;
            _geometryManifestPath.clear();
            _overlay->reset();
        }
    });
    connect(_service, &SpiralServiceManager::sessionAccepted, this,
            [this](const QJsonObject& paths, qint64 generation) {
                loadInputSurfaces(paths, static_cast<quint64>(generation));
            });
    connect(_service, &SpiralServiceManager::sessionStatusChanged, this, [this](const QJsonObject& status) {
        const QString path = status.value(QStringLiteral("geometry_snapshot_manifest_path")).toString();
        if (!path.isEmpty() && path != _geometryManifestPath) {
            _geometryManifestPath = path;
            loadGeometrySnapshot(path, static_cast<quint64>(status.value(QStringLiteral("session_generation")).toInteger()));
        }
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

void SpiralWorkspace::loadInputSurfaces(const QJsonObject& paths, quint64 generation)
{
    if (_shuttingDown || generation < _inputSurfaceGeneration) return;
    _inputSurfaceGeneration = generation;
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

SpiralWorkspace::~SpiralWorkspace()
{
    _shuttingDown = true;
    if (_viewerManager) _viewerManager->beginShutdown();
    if (_service) _service->stopService();
    if (_state) _state->closeAll(); // no package was ever assigned, so Main cannot be unloaded
}

void SpiralWorkspace::keyPressEvent(QKeyEvent* event)
{
    if (event && event->key() == Qt::Key_R && event->modifiers() == Qt::NoModifier) {
        centerFocusOnCursor();
        event->accept();
        return;
    }
    QMainWindow::keyPressEvent(event);
}

bool SpiralWorkspace::centerFocusOnCursor()
{
    if (!_state || !_viewerManager) return false;
    const QPoint globalPosition = QCursor::pos();
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (!viewer) continue;
        auto* graphicsView = viewer->graphicsView();
        auto* viewport = graphicsView ? graphicsView->viewport() : nullptr;
        if (!viewport || !viewport->isVisible()) continue;
        const QPoint viewportPosition = viewport->mapFromGlobal(globalPosition);
        if (!viewport->rect().contains(viewportPosition)) continue;

        const QPointF scenePosition = graphicsView->mapToScene(viewportPosition);
        const cv::Vec3f position = viewer->sceneToVolume(scenePosition);
        if (!std::isfinite(position[0]) || !std::isfinite(position[1]) ||
            !std::isfinite(position[2])) {
            return false;
        }
        cv::Vec3f normal(0, 0, 0);
        if (auto* plane = dynamic_cast<PlaneSurface*>(viewer->currentSurface()))
            normal = plane->normal({}, {});
        setFocusAt(position, normal, viewer->surfName());
        return true;
    }
    return false;
}

void SpiralWorkspace::setFocusAt(const cv::Vec3f& position, const cv::Vec3f& normal,
                                 const std::string& surfaceId)
{
    if (!_state) return;
    POI* focus = _state->poi("focus");
    if (!focus) focus = new POI;
    focus->p = position;
    if (cv::norm(normal) > 0.0f) focus->n = normal;
    focus->surfaceId = surfaceId.empty() ? "segmentation" : surfaceId;
    focus->surfacePtr.reset();
    focus->suppressTransientPlaneIntersections = true;
    _state->setPOI("focus", focus);
    finishFocusChange();
}

void SpiralWorkspace::finishFocusChange()
{
    if (!_state || !_viewerManager || !_slices) return;
    POI* focus = _state->poi("focus");
    if (!focus) return;

    _slices->applyOrientation();
    const cv::Vec3f position = focus->p;
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (!viewer) continue;
        viewer->centerOnVolumePoint(position);
        viewer->invalidateIntersect();
        viewer->renderIntersections("Spiral focus changed");
        viewer->requestRender("Spiral focus changed");
    }
}

void SpiralWorkspace::initializePreviewFocus()
{
    if (!_state || !_currentPreview) return;
    if (!_state->poi("focus")) {
        auto focus = _state->createSurfaceFocusPoi(*_currentPreview);
        if (!focus) return;
        _state->setPOI("focus", focus.release());
    }
    finishFocusChange();
}

void SpiralWorkspace::refreshVolumes()
{
    QVector<VolumeSelector::VolumeOption> options;
    auto package = _mainState ? _mainState->vpkg() : nullptr;
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
        if (!_state->poi("focus")) initializePreviewFocus();
    }
}

void SpiralWorkspace::selectVolume(const QString& id)
{
    auto package = _mainState ? _mainState->vpkg() : nullptr;
    if (!package || id.isEmpty()) return;
    auto volume = package->volume(id.toStdString());
    if (!volume) return;
    if (volume == _state->currentVolume()) {
        if (!_state->poi("focus")) initializePreviewFocus();
        return;
    }
    _state->setCurrentVolume(volume);
    if (!_state->poi("focus")) initializePreviewFocus();
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
        const QString surfacePath = manifest.value(QStringLiteral("surface_path")).toString();
        const QString surfaceId = manifest.value(QStringLiteral("surface_id")).toString();
        if (surfacePath.isEmpty() || surfaceId.isEmpty()) return {{}, {}, QObject::tr("Malformed Spiral preview manifest")};
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
    const bool hadFocus = _state->poi("focus") != nullptr;
    auto previous = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (previous) _retiredPreviews.emplace_back(QString::fromStdString(previous->id), previous);
    _currentPreview = result.surface;
    if (_outputVisible) _state->setSurface("segmentation", result.surface);
    if (!hadFocus) initializePreviewFocus();
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
