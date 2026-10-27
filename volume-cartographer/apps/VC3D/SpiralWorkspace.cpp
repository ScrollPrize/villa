#include "SpiralWorkspace.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ConsoleOutputWidget.hpp"
#include "Keybinds.hpp"
#include "SpiralPanel.hpp"
#include "SpiralBrushController.hpp"
#include "SpiralServiceManager.hpp"
#include "SpiralStoreInputs.hpp"
#include "SpiralMinimap.hpp"
#include "SurfaceOverlayColors.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "elements/ViewerSplitGrid.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/SpiralOverlayController.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <opencv2/imgcodecs.hpp>

#include <QDialog>
#include <QDockWidget>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QKeyEvent>
#include <QLabel>
#include <QMessageBox>
#include <QProgressBar>
#include <QRegularExpression>
#include <QScopedValueRollback>
#include <QSettings>
#include <QSaveFile>
#include <QShortcut>
#include <QStatusBar>
#include <QUuid>
#include <QTimer>
#include <QTemporaryFile>
#include <QVBoxLayout>
#include <QWindow>
#include <QtConcurrent/QtConcurrent>

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

class SpiralLineDraftOverlay final : public ViewerOverlayControllerBase
{
public:
    explicit SpiralLineDraftOverlay(QObject* parent = nullptr)
        : ViewerOverlayControllerBase("spiral_line_annotation_draft", parent) {}

    void setDraft(VolumeViewerBase* viewer, std::vector<cv::Vec2f> points)
    {
        _viewer = viewer;
        _points = std::move(points);
        refreshAll();
    }

protected:
    bool isOverlayEnabledFor(VolumeViewerBase* viewer) const override
    {
        return viewer && viewer == _viewer && !_points.empty();
    }

    void collectPrimitives(VolumeViewerBase* viewer, OverlayBuilder& builder) override
    {
        if (!isOverlayEnabledFor(viewer)) return;
        OverlayStyle line;
        line.penColor = QColor(255, 205, 40, 245);
        line.penWidth = 3.0;
        line.z = 110.0;
        if (_points.size() >= 2) builder.addSurfaceLineStrip(_points, false, line);
        OverlayStyle point = line;
        point.penColor = QColor(20, 20, 20, 255);
        point.brushColor = QColor(255, 205, 40, 255);
        point.penWidth = 1.5;
        point.z = 111.0;
        for (const auto& position : _points)
            builder.addSurfacePoint(position, 5.0, point);
    }

private:
    VolumeViewerBase* _viewer = nullptr;
    std::vector<cv::Vec2f> _points;
};

namespace {

std::optional<std::array<std::size_t, 3>> parseBaseShapeZYX(
    const QJsonValue& value, QString* errorMessage)
{
    if (value.isUndefined() || value.isNull()) return std::nullopt;
    const QJsonArray shape = value.toArray();
    if (shape.size() != 3) {
        if (errorMessage) *errorMessage = QObject::tr(
            "Spiral preview base_shape_zyx must contain three positive integers");
        return std::nullopt;
    }
    std::array<std::size_t, 3> parsed{};
    for (int axis = 0; axis < shape.size(); ++axis) {
        const double extent = shape.at(axis).toDouble(
            std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(extent) || extent < 1.0 || std::floor(extent) != extent ||
            extent > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            if (errorMessage) *errorMessage = QObject::tr(
                "Spiral preview base_shape_zyx must contain three positive integers");
            return std::nullopt;
        }
        parsed[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(extent);
    }
    return parsed;
}

std::optional<int> omeScaledownForGroup(
    const QString& rootPath, const QString& group, QString* errorMessage)
{
    QFile attributesFile(QDir(rootPath).filePath(QStringLiteral(".zattrs")));
    if (!attributesFile.open(QIODevice::ReadOnly)) {
        if (errorMessage) *errorMessage = QObject::tr("Cannot read OME metadata at %1")
            .arg(attributesFile.fileName());
        return std::nullopt;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        attributesFile.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (errorMessage) *errorMessage = QObject::tr("Malformed OME metadata at %1: %2")
            .arg(attributesFile.fileName(), parseError.errorString());
        return std::nullopt;
    }
    const QJsonArray multiscales = document.object()
        .value(QStringLiteral("multiscales")).toArray();
    if (multiscales.isEmpty()) {
        if (errorMessage) *errorMessage = QObject::tr("OME multiscales metadata is missing");
        return std::nullopt;
    }
    const QJsonArray datasets = multiscales.at(0).toObject()
        .value(QStringLiteral("datasets")).toArray();
    auto datasetScale = [&](const QString& wanted) -> std::optional<double> {
        for (const QJsonValue& value : datasets) {
            const QJsonObject dataset = value.toObject();
            if (dataset.value(QStringLiteral("path")).toString() != wanted) continue;
            std::optional<double> scale;
            for (const QJsonValue& transformValue : dataset.value(
                     QStringLiteral("coordinateTransformations")).toArray()) {
                const QJsonObject transform = transformValue.toObject();
                const QString type = transform.value(QStringLiteral("type")).toString();
                const QJsonArray values = transform.value(type).toArray();
                if (values.size() != 3) return std::nullopt;
                if (type == QStringLiteral("translation")) {
                    for (const QJsonValue& component : values)
                        if (!component.isDouble() || !std::isfinite(component.toDouble()) ||
                            std::abs(component.toDouble()) > 1.0e-9) return std::nullopt;
                } else if (type == QStringLiteral("scale")) {
                    const double first = values.at(0).toDouble(
                        std::numeric_limits<double>::quiet_NaN());
                    if (!std::isfinite(first) || first <= 0.0) return std::nullopt;
                    for (const QJsonValue& component : values) {
                        const double current = component.toDouble(
                            std::numeric_limits<double>::quiet_NaN());
                        if (!std::isfinite(current) || current <= 0.0 ||
                            std::abs(current - first) >
                                1.0e-9 * std::max({1.0, current, first})) return std::nullopt;
                    }
                    scale = first;
                } else return std::nullopt;
            }
            return scale;
        }
        return std::nullopt;
    };
    const auto baseScale = datasetScale(QStringLiteral("0"));
    const auto groupScale = datasetScale(group);
    if (!baseScale || !groupScale) {
        if (errorMessage) *errorMessage = QObject::tr(
            "OME scale metadata for groups 0 and %1 is required").arg(group);
        return std::nullopt;
    }
    const double levelValue = std::log2(*groupScale / *baseScale);
    const int level = static_cast<int>(std::llround(levelValue));
    if (level < 0 || level > 30 || !std::isfinite(levelValue) ||
        std::abs(levelValue - static_cast<double>(level)) > 1.0e-9) {
        if (errorMessage) *errorMessage = QObject::tr(
            "OME scale for group %1 is not a dyadic L0 scale").arg(group);
        return std::nullopt;
    }
    return level;
}

// Shape of an OME store's level-zero array, the grid its group scaledowns
// are relative to. Reads the array metadata directly (Zarr v2 ``.zarray`` or
// v3 ``zarr.json``) so it needs no chunk access.
std::optional<std::array<std::size_t, 3>> omeLevelZeroShapeZYX(
    const QString& rootPath, QString* errorMessage)
{
    const QDir level(QDir(rootPath).filePath(QStringLiteral("0")));
    for (const QString& name : {QStringLiteral(".zarray"), QStringLiteral("zarr.json")}) {
        QFile metadataFile(level.filePath(name));
        if (!metadataFile.open(QIODevice::ReadOnly)) continue;
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(
            metadataFile.readAll(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            if (errorMessage) *errorMessage = QObject::tr("Malformed Zarr metadata at %1: %2")
                .arg(metadataFile.fileName(), parseError.errorString());
            return std::nullopt;
        }
        const QJsonArray shape = document.object().value(QStringLiteral("shape")).toArray();
        std::array<std::size_t, 3> result{};
        if (shape.size() == 3) {
            bool valid = true;
            for (int axis = 0; axis < 3; ++axis) {
                const double value = shape.at(axis).toDouble(-1.0);
                if (!(value > 0.0) || value != std::floor(value)) { valid = false; break; }
                result[axis] = static_cast<std::size_t>(value);
            }
            if (valid) return result;
        }
        if (errorMessage) *errorMessage = QObject::tr(
            "Zarr metadata at %1 must declare a three-axis positive shape")
            .arg(metadataFile.fileName());
        return std::nullopt;
    }
    if (errorMessage) *errorMessage = QObject::tr(
        "Cannot read level-zero Zarr array metadata below %1").arg(rootPath);
    return std::nullopt;
}

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
    // Regular volume chunks use the process service. Derived surface tiles
    // remain local to each ViewerManager.
    _state = new CState(
        this, _mainState && _mainState->debugDownloadQueueEnabled(), _mainState);
    connect(_state,&CState::objectStoreChanged,this,[this](const vc::objects::StoreSessionEvent& event) {
        if(event.kind==vc::objects::StoreSessionEvent::Kind::Attached || event.kind == vc::objects::StoreSessionEvent::Kind::Detached) {
            cancelLineAnnotationDraft();
            if (_service)
                _service->setObjectStore(_state->objectStoreSession(), _state->objectStoreAttachment());
        } else {
            noteStoreChanges(event);
        }
    });
    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
    // Spiral can trade some intersection detail for substantially cheaper
    // input-patch indexing without changing the main workspace preference.
    _viewerManager->setSurfacePatchSamplingStride(4);
    // The flattened preview is this workspace's primary pane: stop publishing
    // raw-path chunk demand for frames the SurfaceCache fully serves, so its
    // tile fills are not starved behind interactive fetches the frame never
    // reads.
    _viewerManager->setPreferSurfaceTileFills(true);
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
    _brush->setPatchIndexProvider([this]() { return _viewerManager->surfacePatchIndexIfReady(); });
    _brush->bindToViewerManager(_viewerManager.get());
    _lineDraftOverlay = std::make_unique<SpiralLineDraftOverlay>(this);
    _lineDraftOverlay->bindToViewerManager(_viewerManager.get());
    for (const auto role : vc3d::spiral::kEditablePclRoles) {
        auto& state = pclOverlay(role);
        state.collection = std::make_unique<VCCollection>(this);
        state.overlay = std::make_unique<PointsOverlayController>(
            state.collection.get(), this, true);
        state.overlay->bindToViewerManager(_viewerManager.get());
        state.overlay->setVisible(false);
        // Relative-winding points are read by their winding labels.
        state.overlay->setShowWindingLabels(
            vc3d::spiral::pclRoleHasWindingAnnotations(role));
        _brush->setPclHitOverlay(role, state.overlay.get());
    }
    _surfaceOverlapOverlay = std::make_unique<SegmentationOverlayController>(_state, this);
    _surfaceOverlapOverlay->setViewerManager(_viewerManager.get());
    _viewerManager->setSegmentationOverlay(_surfaceOverlapOverlay.get());
    _surfaceCategoryVisible = {{QStringLiteral("verified"), false},
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
            if (auto* view = viewer->graphicsView()) {
                connect(view, &CVolumeViewerView::sendMouseRelease, this,
                        [this](QPointF scenePoint, Qt::MouseButton button,
                               Qt::KeyboardModifiers modifiers) {
                            if (button == Qt::LeftButton)
                                appendLineAnnotationDraftPoint(scenePoint, modifiers);
                        });
            }
        } else {
            // Point collections can also be placed on the plane views.
            _brush->bindPlaneViewer(viewer);
        }
        viewer->setIntersects(specs[pane].intersects);
        _grid->setViewer(pane, qobject_cast<QWidget*>(viewer->asQObject()));
    }
    // The winding minimap docks below the flattened viewer's graphics view,
    // inside the same pane; the viewer widget's layout only holds the view.
    if (auto* flattenedWidget =
            qobject_cast<QWidget*>(_flattenedViewer->asQObject());
        flattenedWidget && flattenedWidget->layout()) {
        _windingMinimap = new SpiralMinimap(flattenedWidget);
        flattenedWidget->layout()->addWidget(_windingMinimap);
        connect(_windingMinimap, &SpiralMinimap::columnClicked, this,
                [this](float column) { panFlattenedViewerToColumn(column); });
        _flattenedViewer->connectOverlaysUpdated(
            this, [this]() { updateMinimapViewIndicator(); });
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
    cacheStatsLabel->setText(tr("RAM --  disk --  net --"));
    statusBar()->addPermanentWidget(cacheStatsLabel);
    connect(_viewerManager.get(), &ViewerManager::sharedCacheStatsChanged, cacheStatsLabel,
            [cacheStatsLabel](const QStringList& items) {
                if (!items.isEmpty()) cacheStatsLabel->setText(items.join(QStringLiteral("  ")));
            });

    _service = new SpiralServiceManager(this);
    // A local service is launched bound to the attached store, and any
    // service's inputs are displayed only from the same store.
    _service->setObjectStore(_state->objectStoreSession(), _state->objectStoreAttachment());

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
    // The service suppresses successful polling access lines at the source
    // (SpiralHandler.log_request), so every relayed message is worth showing.
    connect(_service, &SpiralServiceManager::logMessage, _pythonOutput,
            [this](const QString& message) {
                _pythonOutput->appendOutput(message);
            });
    connect(_service, &SpiralServiceManager::errorOccurred, this, [this](const QString& error) {
        statusBar()->showMessage(error, 15000);
        // The dialog stays where the user left it; the status bar carries the
        // error and the panel's "Logs" button opens the detail on demand.
        _pythonOutput->appendOutput(tr("Error: %1").arg(error));
    });
    connect(_service, &SpiralServiceManager::sessionLeaseReleased, this,
            [this]() { cancelLineAnnotationDraft(); });
    auto* finalizeBrushShortcut = new QShortcut(QKeySequence(Qt::SHIFT | Qt::Key_E), this);
    finalizeBrushShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(finalizeBrushShortcut, &QShortcut::activated, this, [this]() {
        if (_lineAnnotationDraft) {
            finalizeLineAnnotationDraft();
            return;
        }
        if (!_service || !_service->hasActiveSession()) {
            statusBar()->showMessage(tr("Load a Spiral fit before readying drawn inputs"), 10000);
            return;
        }
        _brush->markDraftsReady();
        statusBar()->showMessage(
            tr("Drawing draft is ready; use Add to current fit to submit it"), 5000);
    });
    connect(_brush.get(), &SpiralBrushController::brushDiameterChanged,
            this, [this](int diameter) {
                statusBar()->showMessage(tr("Spiral brush diameter: %1 px").arg(diameter), 2500);
            });
    connect(_brush.get(), &SpiralBrushController::pointPlacementRejected,
            this, [this](const QString& message) {
                statusBar()->showMessage(message, 5000);
            });

    _panel = new SpiralPanel(_service, this);
    connect(_panel, &SpiralPanel::removeLocalPatchRequested, this,
            [this](const QString& id) { _brush->removePatchDraft(id); });
    connect(_panel, &SpiralPanel::addDraftsRequested,
            this, &SpiralWorkspace::submitReadyDrafts);
    connect(_brush.get(), &SpiralBrushController::paintStateChanged,
            this, [this]() {
                _panel->setLocalDraftsReady(_brush->hasReadyDrafts() || _brush->hasUnfinalizedPaint()
                    || _brush->hasUnfinalizedPolylines());
                _panel->setLocalPatchDrafts(_brush->patchDrafts());
                pruneBrushPreviewSurfaces();
            });
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
                _viewerManager->setSurfacePatchSamplingStride(stride);
            });
    connect(_panel, &SpiralPanel::surfaceOverlapChanged, this, [this](bool shown) {
        _showSurfaceOverlap = shown;
        updateSurfaceIntersections();
    });
    connect(_panel, &SpiralPanel::pclOverlayChanged, this,
            [this](vc3d::spiral::PclRole role, bool shown) {
                auto& state = pclOverlay(role);
                state.visible = shown;
                if (_brush) _brush->setPclSourceVisible(role, shown);
                if (state.overlay) state.overlay->setVisible(shown);
            });
    const auto applyPointViewTolerance = [this](double tolerance) {
        for (auto& state : _pclOverlays) {
            if (state.overlay) state.overlay->setViewTolerance(tolerance);
        }
        if (_brush) _brush->setPointViewTolerance(tolerance);
    };
    applyPointViewTolerance(_panel->pointViewTolerance());
    connect(_panel, &SpiralPanel::pointViewToleranceChanged,
            this, applyPointViewTolerance);
    connect(_brush.get(),
            &SpiralBrushController::suppressedPclCollectionIdsChanged,
            this, [this](vc3d::spiral::PclRole role, const QSet<QString>& ids) {
                auto& state = pclOverlay(role);
                if (!state.overlay) return;
                QSet<qulonglong> numericIds;
                for (const QString& id : ids) {
                    bool ok = false;
                    const qulonglong numeric = id.toULongLong(&ok);
                    if (ok) numericIds.insert(numeric);
                }
                state.overlay->setHiddenCollectionIds(numericIds);
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
    connect(_panel, &SpiralPanel::windingTransitionsChanged, this, [this](bool shown) {
        _windingTransitionsVisible = shown;
        updateWindingTransitionOverlay();
    });
    connect(_panel, &SpiralPanel::lossMapChanged, this,
            [this](const QString& name, qreal opacity) {
                _selectedLossMap = name;
                _lossMapOpacity = opacity;
                updateLossMapOverlay();
            });
    connect(_panel, &SpiralPanel::previewDiagnosticsChanged, this,
            [this](bool enabled) {
                if (_service) _service->setPreviewDiagnostics(enabled);
            });
    connect(_service, &SpiralServiceManager::previewAvailable, this, &SpiralWorkspace::loadPreview);
    connect(_service, &SpiralServiceManager::previewDiagnosticsAvailable, this,
            &SpiralWorkspace::installPreviewDiagnostics);
    connect(_service, &SpiralServiceManager::inputManifestAvailable, this,
            [this](const QString& manifestPath, const QString& digest) {
                _inputManifestPath = manifestPath;
                _inputDigest = digest;
                loadPinnedInputs();
            });
    // The inputs can be displayed only from the store the service read.
    connect(_service, &SpiralServiceManager::storeBindingChanged, this,
            [this]() { if (!_inputManifestPath.isEmpty()) loadPinnedInputs(); });
    connect(_service, &SpiralServiceManager::connectionStateChanged, this,
            [this](SpiralServiceManager::ConnectionState state, const QString&) {
                using CS = SpiralServiceManager::ConnectionState;
                // Disconnect invalidates callbacks from the old connection.
                if (state == CS::Disconnected) {

                }
                if (state == CS::Starting || state == CS::Connecting) {
                    _requestedPreviewGeneration = -1;
                    _inputSurfaceGeneration = 0;
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
            [this](bool active) {
                emit spiralSessionActiveChanged(active);
                if (active) return;
                _inputManifestPath.clear();
                _inputDigest.clear();
                clearPinnedInputs({});
                if (!_service->sessionLeaseId().isEmpty()) return;
                cancelLineAnnotationDraft();
            });
    connect(_service, &SpiralServiceManager::sessionSynchronized, this,
            [this](const QJsonObject& request, const QJsonObject& status) {
                const QJsonObject paths =
                    request.value(QStringLiteral("paths")).toObject();
                const qint64 generation =
                    status.value(QStringLiteral("session_generation")).toInteger();
                _sessionPaths = paths;
                _sessionRunConfig = request.value(QStringLiteral("run")).toObject()
                    .value(QStringLiteral("config")).toObject();
                _previewSource.reset();
                _previewBaseShapeZYX.reset();
                _fiberBaseShapeZYX.reset();
                _previewToFiberBaseScale.reset();
                _previewCoordinateError.clear();
                _previewComponents.clear();
                _previewWindingIds.release();
                _previewRunDiffImagePath.clear();
                // Drafts and captured submissions belong to the editing
                // workspace, so reconnect and resident rebuild retain them.
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
    synchronizeVolume(_mainState ? _mainState->currentVolume() : nullptr);
}

QString SpiralWorkspace::mapServicePath(const QString& servicePath) const
{
    const SpiralServiceProfile& profile = _service->profile();
    if (profile.isLocalhost()) return servicePath;
    // The service side of the mapping is the dataset root the service
    // advertises; the profile only names where this computer mounts it.
    QString serviceRoot = _service->advertisedDataset()
                              .value(QStringLiteral("root")).toString();
    while (serviceRoot.endsWith(QLatin1Char('/'))) serviceRoot.chop(1);
    if (serviceRoot.isEmpty() || profile.localRootPrefix.isEmpty()
        || !servicePath.startsWith(serviceRoot))
        return {};
    // Translate separators as well as prefixes: a Windows viewer may map a
    // POSIX service root.
    QString rest = servicePath.mid(serviceRoot.size());
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
    for (const char* key : {"outer_shell"}) {
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
        // A new session drops the previous one's drafted surfaces too.
        result.categories = {QStringLiteral("shell"), QStringLiteral("brush"),
                             QStringLiteral("ephemeral")};
        const std::pair<const char*, const char*> inputs[] = {
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
    QHash<QString, QStringList> replacement = _surfaceCategoryIds;
    for (const QString& category : result.categories) replacement[category] = {};
    QHash<QString, QString> replacementSourceIds = _surfaceSourceIds;
    std::map<std::string, cv::Vec3b> replacementColors = _surfaceOverlayColors;
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
    for (const QString& category : result.categories) {
        for (const QString& id : _surfaceCategoryIds.value(category)) {
            if (!replacementIds.contains(id)) {
                surfaceUpdates.emplace_back(id.toStdString(), nullptr);
                replacementSourceIds.remove(id);
                replacementColors.erase(id.toStdString());
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
    auto addCategory = [this, &intersections, &surfaceOverlays](const QString& category) {
        for (const QString& id : _surfaceCategoryIds.value(category)) {
            intersections.insert(id.toStdString());
            if (_showSurfaceOverlap) {
                const auto color = _surfaceOverlayColors.find(id.toStdString());
                if (color != _surfaceOverlayColors.end())
                    surfaceOverlays.emplace(color->first, color->second);
            }
        }
    };
    for (auto visible = _surfaceCategoryVisible.begin();
         visible != _surfaceCategoryVisible.end(); ++visible) {
        if (visible.value()) addCategory(visible.key());
    }
    // Brush-created patches are session annotations and remain visible even
    // when the dataset input categories are hidden.
    addCategory(QStringLiteral("brush"));
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (!viewer) continue;
        // Identify the flattened output viewer by its surface name rather
        // than a cached pointer: the pointer can go stale across
        // ViewerManager recreations and caused the surface-overlay toggle
        // (spiralShowSurfaceOverlap) to silently clear the flattened viewer's
        // overlays while treating it as a plane viewer.
        const bool isFlattened = viewer->surfName() == "segmentation";
        if (isFlattened) {
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

void SpiralWorkspace::setLineAnnotationController(LineAnnotationController* controller)
{
    if (_lineAnnotationController) disconnect(_lineAnnotationController, nullptr, this, nullptr);
    _lineAnnotationController = controller;
}

bool SpiralWorkspace::isFlattenedViewer(const VolumeViewerBase* viewer) const
{
    return viewer && viewer == _flattenedViewer;
}

QString SpiralWorkspace::lineAnnotationDraftUnavailableReason() const
{
    if (!_lineAnnotationController) return tr("Line Annotation is not initialized");
    if (!_service || !_service->hasActiveSession()) return tr("Load a Spiral session first");
    if (!_currentPreview || !_flattenedViewer ||
        _flattenedViewer->currentSurface() != _currentPreview.get())
        return tr("Wait for a flattened Spiral preview");
    if (!_previewToFiberBaseScale)
        return _previewCoordinateError.isEmpty()
            ? tr("Spiral preview and fiber coordinates have not been paired")
            : _previewCoordinateError;
    return {};
}

void SpiralWorkspace::startLineAnnotationDraft()
{
    QString reason = lineAnnotationDraftUnavailableReason();
    if (reason.isEmpty() && !_state->objectStoreAttachment())
        reason = tr("Attach an object store before creating fibers");
    if (!reason.isEmpty()) {
        statusBar()->showMessage(reason, 10000);
        return;
    }
    LineAnnotationDraft draft;
    draft.surface = _currentPreview;
    draft.saveAllowed = std::make_shared<std::atomic_bool>(true);
    _lineAnnotationDraft = std::move(draft);
    _lineDraftOverlay->setDraft(_flattenedViewer, {});
    statusBar()->showMessage(tr(
        "2D line annotation: left-click controls, Backspace/Ctrl+Z undo, "
        "Escape cancel, Shift+E optimize"));
}

void SpiralWorkspace::cancelLineAnnotationDraft()
{
    if (!_lineAnnotationDraft) return;
    if (_lineAnnotationDraft->saveAllowed)
        _lineAnnotationDraft->saveAllowed->store(false);
    _lineAnnotationDraft.reset();
    if (_lineDraftOverlay) _lineDraftOverlay->setDraft(_flattenedViewer, {});
}

void SpiralWorkspace::undoLineAnnotationDraftPoint()
{
    if (!_lineAnnotationDraft || _lineAnnotationDraft->optimizing ||
        _lineAnnotationDraft->surfacePoints.empty()) return;
    _lineAnnotationDraft->surfacePoints.pop_back();
    std::vector<cv::Vec2f> points;
    points.reserve(_lineAnnotationDraft->surfacePoints.size());
    for (const auto& point : _lineAnnotationDraft->surfacePoints)
        points.emplace_back(point.x(), point.y());
    _lineDraftOverlay->setDraft(_flattenedViewer, std::move(points));
}

void SpiralWorkspace::appendLineAnnotationDraftPoint(
    const QPointF& scenePoint, Qt::KeyboardModifiers modifiers)
{
    if (!_lineAnnotationDraft || _lineAnnotationDraft->optimizing ||
        modifiers != Qt::NoModifier || !_flattenedViewer ||
        _lineAnnotationDraft->surface.get() != _flattenedViewer->currentSurface()) return;
    const cv::Vec2f surface = _flattenedViewer->sceneToSurfaceCoords(scenePoint);
    if (!std::isfinite(surface[0]) || !std::isfinite(surface[1]) ||
        !_lineAnnotationDraft->surface->sampleAtSurface({surface[0], surface[1]})) {
        statusBar()->showMessage(tr("Point must lie on valid Spiral surface data"), 5000);
        return;
    }
    _lineAnnotationDraft->surfacePoints.emplace_back(surface[0], surface[1]);
    std::vector<cv::Vec2f> points;
    points.reserve(_lineAnnotationDraft->surfacePoints.size());
    for (const auto& point : _lineAnnotationDraft->surfacePoints)
        points.emplace_back(point.x(), point.y());
    _lineDraftOverlay->setDraft(_flattenedViewer, std::move(points));
}

QStringList SpiralWorkspace::fallbackFiberManifests() const
{
    const QString serviceRoot = _sessionPaths.value(QStringLiteral("dataset_root")).toString();
    const QString localRoot = serviceRoot.isEmpty() ? QString() : mapServicePath(serviceRoot);
    QStringList manifests;
    if (!localRoot.isEmpty()) {
        QDirIterator it(QDir(localRoot).filePath(QStringLiteral("fiber_zarrs")),
                        {QStringLiteral("*.lasagna.json")}, QDir::Files,
                        QDirIterator::Subdirectories);
        while (it.hasNext()) manifests.push_back(it.next());
    }
    manifests.sort();
    return manifests;
}

std::optional<std::array<std::size_t, 3>>
SpiralWorkspace::resolveFiberBaseShape(QString* errorMessage) const
{
    if (!_state || !_state->vpkg()) {
        if (errorMessage) *errorMessage = tr("No volume package is loaded");
        return std::nullopt;
    }
    const auto vpkg = _state->vpkg();
    QStringList candidates;
    const QString selected = QString::fromStdString(vpkg->selectedFiberInferenceDataset());
    if (!selected.isEmpty()) candidates.push_back(selected);
    const QStringList fallbacks = fallbackFiberManifests();
    if (fallbacks.size() == 1 && !candidates.contains(fallbacks.front()))
        candidates.push_back(fallbacks.front());

    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = vc3d::remoteCachePathFs();
    QStringList failures;
    for (const QString& candidate : candidates) {
        try {
            const std::string location = candidate.toStdString();
            const std::string resolved = vc::project::isLocationRemote(location)
                ? location
                : vc::project::resolveLocalPath(
                      location, vpkg->path().parent_path()).string();
            const auto dataset = vc::lasagna::LasagnaDataset::openLocation(resolved, options);
            if (!dataset.manifest().baseShapeZYX) {
                failures.push_back(tr("%1 has no base_shape_zyx").arg(candidate));
                continue;
            }
            return dataset.manifest().baseShapeZYX;
        } catch (const std::exception& ex) {
            failures.push_back(tr("%1: %2").arg(candidate, QString::fromUtf8(ex.what())));
        }
    }
    if (errorMessage) {
        if (candidates.isEmpty()) {
            *errorMessage = fallbacks.isEmpty()
                ? tr("No fiber-inference manifest was found below fiber_zarrs")
                : tr("Multiple fiber-inference manifests were found below fiber_zarrs; select one in the project");
        } else {
            *errorMessage = tr("No fiber-inference base shape is available: %1")
                                .arg(failures.join(QStringLiteral("; ")));
        }
    }
    return std::nullopt;
}

void SpiralWorkspace::updatePreviewCoordinateScale()
{
    _fiberBaseShapeZYX.reset();
    _previewToFiberBaseScale.reset();
    _previewCoordinateError.clear();
    if (!_previewBaseShapeZYX) {
        _previewCoordinateError = tr(
            "Spiral preview metadata has no base_shape_zyx; publish a new preview");
        return;
    }
    QString error;
    const auto fiberShape = resolveFiberBaseShape(&error);
    if (!fiberShape) {
        _previewCoordinateError = error;
        return;
    }
    try {
        const double previewToFiber = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
            *_previewBaseShapeZYX, *fiberShape, 5);
        _fiberBaseShapeZYX = fiberShape;
        _previewToFiberBaseScale = previewToFiber;
    } catch (const std::exception& ex) {
        _previewCoordinateError = tr(
            "Spiral preview and fiber base shapes are incompatible: %1")
                                      .arg(QString::fromUtf8(ex.what()));
    }
}

std::optional<LineAnnotationController::ResolvedFiberOptimizationInputs>
SpiralWorkspace::resolveLineAnnotationInputs(QString* errorMessage) const
{
    if (!_lineAnnotationController) return std::nullopt;
    const QJsonObject scroll = _service->advertisedDataset()
                                   .value(QStringLiteral("scroll_spec")).toObject();
    const QString group = scroll.value(QStringLiteral("normal_zarr_group"))
                              .toString(QStringLiteral("4"));
    std::optional<int> omeScale;
    // The grid the normal stores' scaledowns are relative to. It is the
    // stores' own level zero, which need not be the fiber-inference base
    // grid: the manifest declares it so the optimizer derives the true
    // fiber-to-normal conversion rather than assuming the grids coincide.
    std::optional<std::array<std::size_t, 3>> normalBaseShape;
    QString omeError;
    QString fallbackNormal;
    QTemporaryFile manifest(QDir(QDir::tempPath()).filePath(
        QStringLiteral("vc3d-spiral-normal-XXXXXX.lasagna.json")));
    manifest.setAutoRemove(false);
    const QStringList normalKeys{QStringLiteral("normal_x"), QStringLiteral("normal_y"),
                                 QStringLiteral("gradient_magnitude")};
    QStringList mapped;
    for (const QString& key : normalKeys) {
        const QString servicePath = _sessionPaths.value(key).toString();
        mapped.push_back(servicePath.isEmpty() ? QString() : mapServicePath(servicePath));
    }
    if (std::all_of(mapped.begin(), mapped.end(),
                    [](const QString& path) { return !path.isEmpty(); })) {
        for (const QString& path : mapped) {
            QString candidateError;
            const auto candidate = omeScaledownForGroup(path, group, &candidateError);
            if (!candidate) { omeError = candidateError; omeScale.reset(); break; }
            if (omeScale && *omeScale != *candidate) {
                omeError = tr("Spiral normal inputs declare different OME scales");
                omeScale.reset();
                break;
            }
            const auto candidateShape = omeLevelZeroShapeZYX(path, &candidateError);
            if (!candidateShape) { omeError = candidateError; omeScale.reset(); break; }
            if (normalBaseShape && *normalBaseShape != *candidateShape) {
                omeError = tr("Spiral normal inputs declare different level-zero shapes");
                omeScale.reset();
                break;
            }
            omeScale = candidate;
            normalBaseShape = candidateShape;
        }
    }
    if (omeScale && std::all_of(mapped.begin(), mapped.end(),
                                [](const QString& path) { return !path.isEmpty(); }) &&
        manifest.open()) {
        const double encodeScale = _sessionRunConfig
            .value(QStringLiteral("dense_grad_mag_encode_scale")).toDouble(1000.0);
        const double gradFactor = _sessionRunConfig
            .value(QStringLiteral("dense_grad_mag_factor")).toDouble(0.25);
        QJsonObject groups;
        const QStringList channels{QStringLiteral("nx"), QStringLiteral("ny"),
                                   QStringLiteral("grad_mag")};
        for (int i = 0; i < mapped.size(); ++i) {
            groups[channels[i]] = QJsonObject{
                {QStringLiteral("zarr"), QDir(mapped[i]).filePath(group)},
                {QStringLiteral("scaledown"), *omeScale},
                {QStringLiteral("channels"), QJsonArray{channels[i]}},
            };
        }
        QJsonObject root{
            {QStringLiteral("version"), 2},
            {QStringLiteral("source_to_base"), 1.0},
            {QStringLiteral("grad_mag_encode_scale"), encodeScale},
            {QStringLiteral("grad_mag_factor"), gradFactor},
            {QStringLiteral("groups"), groups},
        };
        if (normalBaseShape) {
            QJsonArray baseShape;
            for (const std::size_t extent : *normalBaseShape)
                baseShape.append(static_cast<qint64>(extent));
            root[QStringLiteral("base_shape_zyx")] = baseShape;
        }
        manifest.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        manifest.close();
        fallbackNormal = manifest.fileName();
    }

    QString fallbackFiber;
    const QString serviceRoot = _sessionPaths.value(QStringLiteral("dataset_root")).toString();
    const QStringList manifests = fallbackFiberManifests();
    if (manifests.size() == 1) fallbackFiber = manifests.front();
    auto result = _lineAnnotationController->resolveFiberOptimizationInputs(
        fallbackNormal.toStdString(), fallbackFiber.toStdString(), errorMessage);
    if (result && result->normalManifestLocation == fallbackNormal.toStdString())
        result->normalManifestLocation = QStringLiteral("spiral-session:%1#normals")
            .arg(serviceRoot).toStdString();
    if (!fallbackNormal.isEmpty()) QFile::remove(fallbackNormal);
    if (!result && errorMessage) {
        if (fallbackNormal.isEmpty())
            *errorMessage += tr(" Spiral fallback normals require mapped normal_x, normal_y, "
                                "gradient_magnitude, and valid OME scale metadata: %1").arg(omeError);
        if (manifests.size() != 1)
            *errorMessage += manifests.isEmpty()
                ? tr(" No fiber-inference manifest was found below fiber_zarrs.")
                : tr(" Multiple fiber-inference manifests were found below fiber_zarrs; select one in the project.");
    }
    return result;
}

void SpiralWorkspace::finalizeLineAnnotationDraft()
{
    if (!_lineAnnotationDraft || _lineAnnotationDraft->optimizing) return;
    if (_lineAnnotationDraft->surfacePoints.size() < 2) {
        statusBar()->showMessage(tr(
            "A 2D line annotation needs at least two distinct points"), 10000);
        return;
    }
    QString error;
    auto inputs = resolveLineAnnotationInputs(&error);
    if (!inputs) { statusBar()->showMessage(error, 15000); return; }
    const auto& fiberShape = inputs->fiberDataset->manifest().baseShapeZYX;
    if (!_previewBaseShapeZYX || !fiberShape) {
        statusBar()->showMessage(tr(
            "Spiral preview and fiber-inference metadata must both declare base_shape_zyx"), 15000);
        return;
    }
    double previewToFiberBase;
    try {
        previewToFiberBase = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
            *_previewBaseShapeZYX, *fiberShape, 5);
    } catch (const std::exception& ex) {
        statusBar()->showMessage(tr("Spiral preview and fiber coordinates are incompatible: %1")
            .arg(QString::fromUtf8(ex.what())), 15000);
        return;
    }
    std::vector<cv::Vec3d> controls;
    controls.reserve(_lineAnnotationDraft->surfacePoints.size());
    for (const QPointF& point : _lineAnnotationDraft->surfacePoints) {
        const auto sample = _lineAnnotationDraft->surface->sampleAtSurface({point.x(), point.y()});
        if (!sample) {
            statusBar()->showMessage(tr(
                "A saved control no longer samples valid preview geometry"), 10000);
            return;
        }
        controls.emplace_back(sample.volume[0] * previewToFiberBase,
                              sample.volume[1] * previewToFiberBase,
                              sample.volume[2] * previewToFiberBase);
    }
    for (size_t i = 0; i < controls.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (cv::norm(controls[i] - controls[j]) <= 1.0e-5) {
                statusBar()->showMessage(tr(
                    "A 2D line annotation cannot contain duplicate controls"), 10000);
                return;
            }
        }
    }
    _lineAnnotationDraft->optimizing = true;
    statusBar()->showMessage(tr("Optimizing 2D line annotation…"));
    LineAnnotationController::HeadlessFiberOptimizationRequest request;
    request.controlPointsL0 = std::move(controls);
    request.inputs = std::move(*inputs);
    const auto saveAllowed = _lineAnnotationDraft->saveAllowed;
    request.shouldSave = [saveAllowed]() { return saveAllowed && saveAllowed->load(); };
    QPointer<SpiralWorkspace> self(this);
    _lineAnnotationController->optimizeAndSaveFiberHeadless(
        std::move(request), [self, saveAllowed](bool ok, const QString& message, uint64_t fiberId) {
            if (!self || !self->_lineAnnotationDraft ||
                self->_lineAnnotationDraft->saveAllowed != saveAllowed) return;
            if (!ok) {
                self->_lineAnnotationDraft->optimizing = false;
                self->statusBar()->showMessage(
                    self->tr("2D line annotation failed: %1").arg(message), 15000);
                return;
            }
            self->cancelLineAnnotationDraft();
            if (self->_lineAnnotationController)
                self->_lineAnnotationController->openFiber(fiberId);
            self->statusBar()->showMessage(
                self->tr("Saved 2D line annotation as a local draft"), 10000);
        });
}

void SpiralWorkspace::addPatchToCurrentFit(
    const QString& tifxyzDirectory, const std::shared_ptr<QuadSurface>& surface)
{
    Q_UNUSED(tifxyzDirectory);
    Q_UNUSED(surface);
    QMessageBox::information(this, tr("Add patch to the Spiral fit"),
                             SpiralServiceManager::inputsFrozenMessage());
}

void SpiralWorkspace::addFibersToCurrentFit(const std::vector<uint64_t>& fiberIds)
{
    Q_UNUSED(fiberIds);
    QMessageBox::information(this, tr("Add fibers to the Spiral fit"),
                             SpiralServiceManager::inputsFrozenMessage());
}

void SpiralWorkspace::finalizeBrushPaint()
{
    // Painted patches and drawn collections stay in memory: they cannot join
    // a fit whose inputs are frozen, and are kept for store-backed editing.
    statusBar()->showMessage(SpiralServiceManager::inputsFrozenMessage(), 15000);
}

void SpiralWorkspace::submitReadyDrafts(bool commitAfterAdd)
{
    // Drafts stay local, ready for store-backed editing; a frozen fit takes
    // none of them.
    Q_UNUSED(commitAfterAdd);
    statusBar()->showMessage(SpiralServiceManager::inputsFrozenMessage(), 15000);
}

bool SpiralWorkspace::hasPendingBrushWork() const
{
    return _brush && (_brush->hasUnfinalizedPaint() || _brush->hasUnfinalizedPolylines()
                      || _brush->hasReadyDrafts());
}

void SpiralWorkspace::discardBrushWork()
{
    if (_brush) _brush->discardUnfinalized();
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
    _state->objectStoreSession()->finalize();
    auto exit = [this, continuation]() { _service->releaseSessionLease(continuation); };
    if (!hasPendingBrushWork()) {
        exit();
        return;
    }
    QMessageBox box(QMessageBox::Warning, tr("Unfinished Spiral drawings"),
                    tr("This Spiral workspace holds drawn patches or point collections "
                       "that are not part of the fit: its inputs are frozen and come "
                       "from the object store. Leaving discards them."),
                    QMessageBox::NoButton, this);
    auto* discard = box.addButton(tr("Discard and Exit"), QMessageBox::DestructiveRole);
    box.addButton(QMessageBox::Cancel);
    box.exec();
    if (box.clickedButton() == discard) {
        discardBrushWork();
        exit();
    }
}

SpiralWorkspace::~SpiralWorkspace()
{
    _shuttingDown = true;
    cancelLineAnnotationDraft();
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
    if (event && _lineAnnotationDraft) {
        if (event->key() == Qt::Key_Escape) {
            cancelLineAnnotationDraft();
            event->accept();
            return;
        }
        if (event->key() == Qt::Key_Backspace ||
            (event->key() == Qt::Key_Z &&
             event->modifiers() == Qt::ControlModifier)) {
            undoLineAnnotationDraftPoint();
            event->accept();
            return;
        }
    }
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

QComboBox* SpiralWorkspace::volumeSelectionControl() const
{
    return _panel ? _panel->volumeSelectionControl() : nullptr;
}

void SpiralWorkspace::synchronizeVolume(
    const std::shared_ptr<Volume>& volume,
    const std::optional<vcr::catalog::Affine>& navigationTransform)
{
    const auto package = _mainState ? _mainState->vpkg() : nullptr;
    if (_state->vpkg() != package) {
        // Spiral borrows Main's package; it never owns or independently closes it.
        _state->setVpkg(package);
    }
    if (volume == _state->currentVolume()) {
        ensureInitialFocus();
        return;
    }
    const bool hadFocus = _state->poi("focus") != nullptr;
    _viewerManager->switchVolume(volume, navigationTransform);
    if (volume && !hadFocus) {
        // switchVolume created a volume-center default; prefer the preview
        // focus when one is already loaded.
        _focusIsAutoDefault = true;
        initializePreviewFocus();
    }
    for (auto* viewer : _viewerManager->baseViewers())
        if (viewer) viewer->requestRender("Shared display volume changed");
}

std::vector<SpiralWorkspace::PreviewLoadResult::LossMap>
SpiralWorkspace::parseLossMaps(const QJsonObject& manifest,
                               const QString& artifactRoot)
{
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
    return lossMaps;
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
        QString shapeError;
        const bool hasBaseShape = manifest.contains(QStringLiteral("base_shape_zyx"));
        const auto baseShapeZYX = parseBaseShapeZYX(
            manifest.value(QStringLiteral("base_shape_zyx")), &shapeError);
        if (hasBaseShape && !baseShapeZYX) return failure(shapeError);
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
            || (hasBaseShape && meta.value(QStringLiteral("base_shape_zyx"))
                != manifest.value(QStringLiteral("base_shape_zyx"))))
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
            std::vector<PreviewLoadResult::LossMap> lossMaps =
                parseLossMaps(manifest, artifactRoot);
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
            result.baseShapeZYX = baseShapeZYX;
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
    cancelLineAnnotationDraft();
    _previewSource = result.surface;
    _previewSourceId = result.surfaceId;
    _previewBaseShapeZYX = result.baseShapeZYX;
    updatePreviewCoordinateScale();
    refreshPclOverlays();
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

void SpiralWorkspace::installPreviewDiagnostics(const QString& manifestPath,
                                                qint64 generation)
{
    // The overlays are published after the surface they were mapped through,
    // so by the time they arrive the workspace may have moved on to a newer
    // preview. Both the generation and the surface id are checked: drawing an
    // overlay sampled through a different flatten would be silently wrong.
    if (_shuttingDown || generation != _requestedPreviewGeneration
        || !_previewSource)
        return;
    QFile file(manifestPath);
    if (!file.open(QIODevice::ReadOnly)) {
        statusBar()->showMessage(
            tr("Could not open the Spiral preview diagnostics manifest"), 15000);
        return;
    }
    const QJsonObject manifest = QJsonDocument::fromJson(file.readAll()).object();
    if (manifest.value(QStringLiteral("surface_id")).toString() != _previewSourceId)
        return;
    const std::vector<PreviewLoadResult::LossMap> lossMaps =
        parseLossMaps(manifest, QFileInfo(manifestPath).absolutePath());

    _previewLossMaps.clear();
    _fetchingLossMaps.clear();
    _loadedLossMap.clear();
    _loadedLossMapImage = {};
    QStringList lossMapNames;
    for (const auto& map : lossMaps) {
        _previewLossMaps.insert(map.name, map);
        lossMapNames.push_back(map.name);
    }
    _panel->setLossMapOptions(lossMapNames);
    // A selection that survived the new preview is now backed by these
    // overlays; anything else leaves the overlay cleared.
    updateLossMapOverlay();
}

void SpiralWorkspace::refreshPclOverlays()
{
    for (const auto role : vc3d::spiral::kEditablePclRoles) refreshPclOverlay(role);
}

void SpiralWorkspace::refreshPclOverlay(vc3d::spiral::PclRole role)
{
    auto& state = pclOverlay(role);
    if (!_panel || !state.collection || !state.overlay) return;
    const QString roleName = vc3d::spiral::pclRoleDisplayName(role);
    state.overlay->setVisible(false);
    state.collection->clearAll();
    if (state.documentPath.isEmpty()) {
        _brush->setPclSource(role, {}, 1.0, {}, false);
        _panel->setPclOverlayAvailable(
            role, false, _pinnedInputs ? tr("The fit has no %1 point collections").arg(roleName)
                                       : QString());
        return;
    }
    // The pinned collections are in reference-volume voxels: the scroll's
    // base shape, mapped onto the preview's coordinate domain.
    QString shapeError;
    const auto scrollShape = parseBaseShapeZYX(
        _service->advertisedDataset().value(QStringLiteral("scroll_spec")).toObject()
            .value(QStringLiteral("base_shape_zyx")), &shapeError);
    if (!_previewBaseShapeZYX || !scrollShape) {
        _panel->setPclOverlayAvailable(
            role, false,
            tr("The preview or the scroll specification has no coordinate-domain metadata"));
        return;
    }
    double scale = 1.0;
    try {
        scale = vc::lasagna::dyadicCoordinateScaleBetweenShapes(*scrollShape, *_previewBaseShapeZYX);
    } catch (const std::exception& error) {
        _panel->setPclOverlayAvailable(
            role, false, tr("The %1 PCL coordinate domain is incompatible: %2")
                             .arg(roleName, QString::fromUtf8(error.what())));
        return;
    }
    if (!state.collection->loadFromJSON(state.documentPath.toStdString())) {
        _panel->setPclOverlayAvailable(
            role, false, tr("The pinned %1 point collections are invalid").arg(roleName));
        return;
    }
    // Fitted inputs are frozen: shown, never edited through the overlay.
    _brush->setPclSource(role, QJsonDocument(state.document), scale, _inputDigest, false);
    _brush->setPclSourceVisible(role, state.visible);
    state.overlay->setCoordinateScale(scale);
    _panel->setPclOverlayAvailable(role, true);
    state.overlay->setVisible(state.visible);
}

void SpiralWorkspace::noteStoreChanges(const vc::objects::StoreSessionEvent& event)
{
    if (!_pinnedInputs || event.kind != vc::objects::StoreSessionEvent::Kind::Committed
        || !event.touches({"surface", "point_collection", "fiber", "point-surface",
                           "same-surface", "fiber_branch"}))
        return;
    const auto before = _changedPinnedRecords.size() + _changedOtherInputs.size();
    for (const auto& change : event.changes) {
        const QString uuid = QString::fromStdString(change.recordUuid);
        if (_pinnedRecords.contains(uuid)) _changedPinnedRecords.insert(uuid);
        else if (change.kind == "surface" || change.kind == "point_collection" || change.kind == "fiber")
            _changedOtherInputs.insert(uuid);
    }
    if (_changedPinnedRecords.size() + _changedOtherInputs.size() != before) updateInputFreshness();
}

void SpiralWorkspace::updateInputFreshness()
{
    if (!_panel) return;
    if (!_pinnedInputs) {
        _panel->setInputFreshness({});
        return;
    }
    using vc3d::spiral::PinState;
    const auto& inputs = *_pinnedInputs;
    QStringList notes;
    if (const auto missing = inputs.count(PinState::Missing))
        notes << tr("%n fitted input(s) are missing from the attached store and are not shown", "", int(missing));
    const auto changed = inputs.count(PinState::Edited) + inputs.count(PinState::MetadataChanged);
    if (changed)
        notes << tr("%n fitted input(s) changed in the store before this view loaded", "", int(changed));
    if (!_changedPinnedRecords.isEmpty())
        notes << tr("%n fitted object(s) or link(s) changed since", "", int(_changedPinnedRecords.size()));
    if (!_changedOtherInputs.isEmpty())
        notes << tr("%n other surface(s), collection(s) or fiber(s) changed, which a "
                    "rebuild's selection may include", "", int(_changedOtherInputs.size()));
    if (notes.isEmpty()) {
        _panel->setInputFreshness(tr("The attached store holds every fitted input as fitted"));
        return;
    }
    // Shown, never applied: the overlays keep the fitted revisions.
    _panel->setInputFreshness(tr("Stale: %1. The overlays show the fitted revisions; "
                                 "Rebuild Fit reads the current inputs.")
                                  .arg(notes.join(QStringLiteral("; "))));
}

void SpiralWorkspace::clearPinnedInputs(const QString& reason)
{
    ++_pinnedInputGeneration;
    _pinnedInputs.reset();
    _pinnedRecords.clear();
    _changedPinnedRecords.clear();
    _changedOtherInputs.clear();
    updateInputFreshness();
    for (auto& state : _pclOverlays) {
        state.document = {};
        state.documentPath.clear();
    }
    InputSurfaceLoadResult cleared;
    cleared.categories = {QStringLiteral("verified")};
    installInputSurfaces(cleared, _inputSurfaceGeneration);
    refreshPclOverlays();
    if (!reason.isEmpty()) statusBar()->showMessage(reason, 15000);
}

void SpiralWorkspace::loadPinnedInputs()
{
    if (_shuttingDown) return;
    if (_inputManifestPath.isEmpty()) {
        clearPinnedInputs({});
        return;
    }
    const auto& binding = _service->storeBinding();
    const auto session = _state->objectStoreSession();
    const auto attachment = _state->objectStoreAttachment();
    if (!binding.matches() || !session || !attachment) {
        // Never substitute another store's objects, or current heads, for
        // the fitted inputs.
        clearPinnedInputs(binding.localUuid.isEmpty()
            ? tr("Fit inputs are not displayed: no object store is attached")
            : tr("Fit inputs are not displayed: the service read store %1, not the attached %2")
                  .arg(binding.serviceUuid, binding.localUuid));
        return;
    }
    QFile file(_inputManifestPath);
    nlohmann::json document;
    try {
        if (!file.open(QIODevice::ReadOnly)) throw std::runtime_error("cannot open the manifest");
        document = nlohmann::json::parse(file.readAll().toStdString());
    } catch (const std::exception& error) {
        clearPinnedInputs(tr("The fit's input manifest cannot be read: %1").arg(QString::fromUtf8(error.what())));
        return;
    }
    const quint64 generation = ++_pinnedInputGeneration;
    auto* watcher = new QFutureWatcher<std::shared_ptr<vc3d::spiral::PinnedStoreInputs>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<vc3d::spiral::PinnedStoreInputs>>::finished, this,
            [this, watcher, generation]() {
                const auto inputs = watcher->result();
                watcher->deleteLater();
                if (!_shuttingDown && generation == _pinnedInputGeneration)
                    installPinnedInputs(inputs, generation);
            });
    watcher->setFuture(QtConcurrent::run([session, attachment = *attachment, document]() {
        auto inputs = std::make_shared<vc3d::spiral::PinnedStoreInputs>();
        try {
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                *inputs = vc3d::spiral::readPinnedStoreInputs(db, txn, document);
            });
        } catch (const std::exception& error) {
            inputs->problems.push_back(error.what());
        }
        return inputs;
    }));
}

void SpiralWorkspace::installPinnedInputs(
    const std::shared_ptr<vc3d::spiral::PinnedStoreInputs>& inputs, quint64 generation)
{
    if (generation != _pinnedInputGeneration) return;
    _pinnedInputs = inputs;
    _pinnedRecords.clear();
    _changedPinnedRecords.clear();
    _changedOtherInputs.clear();
    for (const auto& entry : inputs->surfaces) _pinnedRecords.insert(QString::fromStdString(entry.pin.uuid));
    for (const auto& entry : inputs->pointCollections) _pinnedRecords.insert(QString::fromStdString(entry.pin.uuid));
    for (const auto& entry : inputs->fibers) _pinnedRecords.insert(QString::fromStdString(entry.pin.uuid));
    for (const auto& uuid : inputs->relationshipUuids) _pinnedRecords.insert(QString::fromStdString(uuid));
    updateInputFreshness();
    using vc3d::spiral::PinState;
    InputSurfaceLoadResult surfaces;
    surfaces.categories = {QStringLiteral("verified")};
    const QString digest = QString::fromStdString(inputs->digest).left(12);
    for (const auto& entry : inputs->surfaces) {
        if (!entry.surface) continue;
        const QString uuid = QString::fromStdString(entry.pin.uuid);
        // Keyed and coloured by object UUID, which survives renames.
        const QString id = QStringLiteral("spiral/verified/%1/%2").arg(digest, uuid);
        entry.surface->id = id.toStdString();
        surfaces.surfaces.push_back({QStringLiteral("verified"), id, uuid, entry.surface});
    }
    installInputSurfaces(surfaces, _inputSurfaceGeneration);

    // Absolute and relative collections carry windings and share the
    // relative overlay; unannotated ones are same-winding collections.
    std::map<vc3d::spiral::PclRole, QJsonObject> collections;
    int collectionId = 0;
    for (const auto& entry : inputs->pointCollections) {
        if (entry.state == PinState::Missing) continue;
        const bool annotated = entry.absolute
            || std::any_of(entry.windings.begin(), entry.windings.end(),
                           [](const auto& wind) { return wind.has_value(); });
        QJsonObject points;
        for (std::size_t i = 0; i < entry.ids.size(); ++i) {
            QJsonObject point{
                {QStringLiteral("p"), QJsonArray{entry.points[i][0], entry.points[i][1], entry.points[i][2]}},
                {QStringLiteral("creation_time"), 0}};
            point[QStringLiteral("wind_a")] = entry.windings[i] ? QJsonValue(*entry.windings[i]) : QJsonValue();
            points[QString::number(entry.ids[i])] = point;
        }
        const auto role = annotated ? vc3d::spiral::PclRole::Relative : vc3d::spiral::PclRole::SameWinding;
        collections[role][QString::number(++collectionId)] = QJsonObject{
            {QStringLiteral("name"), QString::fromStdString(entry.pin.name)},
            {QStringLiteral("points"), points},
            {QStringLiteral("metadata"), QJsonObject{
                {QStringLiteral("object_uuid"), QString::fromStdString(entry.pin.uuid)},
                {QStringLiteral("winding_is_absolute"), entry.absolute}}}};
    }
    for (const auto role : vc3d::spiral::kEditablePclRoles) {
        auto& state = pclOverlay(role);
        state.document = {};
        state.documentPath.clear();
        if (!collections.contains(role) || !_pinnedPclDocuments.isValid()) continue;
        state.document = QJsonObject{{QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")},
                                     {QStringLiteral("collections"), collections[role]}};
        const QString path = _pinnedPclDocuments.filePath(
            QStringLiteral("%1-%2.json").arg(digest, vc3d::spiral::pclRoleDisplayName(role)));
        QSaveFile out(path);
        const QByteArray bytes = QJsonDocument(state.document).toJson(QJsonDocument::Compact);
        if (out.open(QIODevice::WriteOnly) && out.write(bytes) == bytes.size() && out.commit())
            state.documentPath = path;
    }
    refreshPclOverlays();

    QStringList notes;
    for (const auto& problem : inputs->problems) notes << QString::fromStdString(problem);
    if (!notes.isEmpty()) statusBar()->showMessage(notes.join(QStringLiteral("; ")), 15000);
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

namespace
{

// Trace the boundaries between winding regions of the displayed preview crop
// as row-ordered polylines, one curve per winding pair. Each row is walked
// left to right over the winding-id map; wherever the winding changes between
// two valid pixels no more than kMaxGapColumns apart, the midpoint between
// them is a boundary point. Per pair, points chain into segments that break
// when the boundary vanishes for more than kMaxRowGap rows or jumps by more
// than kMaxGapColumns columns, so curves never bridge holes in the surface.
// Windings outside the displayed range count as invalid: their vertices were
// masked off the surface, so a boundary against them would float in a void.
std::vector<SpiralOverlayController::WindingTransitionCurve>
traceWindingTransitions(const cv::Mat_<int32_t>& windings,
                        int minimumWinding, int maximumWinding)
{
    constexpr int kMaxGapColumns = 8;
    constexpr float kMaxRowGap = 4.0f;
    constexpr float kMaxRowStep = 16.0f;
    constexpr float kColumnTolerance = 0.5f;
    std::map<std::pair<int, int>, std::vector<cv::Vec2f>> pointsByPair;
    for (int row = 0; row < windings.rows; ++row) {
        int previousColumn = -1;
        int previousWinding = -1;
        for (int column = 0; column < windings.cols; ++column) {
            const int winding = windings(row, column);
            if (winding < minimumWinding
                || (maximumWinding >= 0 && winding > maximumWinding))
                continue;
            if (previousColumn >= 0 && winding != previousWinding
                && column - previousColumn <= kMaxGapColumns) {
                pointsByPair[{previousWinding, winding}].emplace_back(
                    static_cast<float>(previousColumn + column) / 2.0f,
                    static_cast<float>(row));
            }
            previousColumn = column;
            previousWinding = winding;
        }
    }
    std::vector<SpiralOverlayController::WindingTransitionCurve> curves;
    for (auto& [pair, points] : pointsByPair) {
        SpiralOverlayController::WindingTransitionCurve curve;
        curve.fromWinding = pair.first;
        curve.toWinding = pair.second;
        std::vector<std::vector<cv::Vec2f>> segments(1);
        for (const cv::Vec2f& point : points) {
            if (!segments.back().empty()
                && (point[1] - segments.back().back()[1] > kMaxRowGap
                    || std::abs(point[0] - segments.back().back()[0])
                        > static_cast<float>(kMaxGapColumns)))
                segments.emplace_back();
            segments.back().push_back(point);
        }
        for (std::vector<cv::Vec2f>& segment : segments) {
            if (segment.size() < 2) continue;
            // Decimate near-vertical runs: keep both endpoints plus any point
            // that moved in column or is kMaxRowStep rows past the last kept.
            std::vector<cv::Vec2f> kept{segment.front()};
            for (std::size_t index = 1; index + 1 < segment.size(); ++index) {
                if (std::abs(segment[index][0] - kept.back()[0])
                        >= kColumnTolerance
                    || segment[index][1] - kept.back()[1] >= kMaxRowStep)
                    kept.push_back(segment[index]);
            }
            kept.push_back(segment.back());
            curve.segments.push_back(std::move(kept));
        }
        if (!curve.segments.empty()) curves.push_back(std::move(curve));
    }
    return curves;
}

} // namespace

void SpiralWorkspace::updateWindingMinimap()
{
    if (!_windingMinimap) return;
    const auto selection = displayedPreviewSelection();
    if (!selection || !_currentPreview) {
        _windingMinimap->clearBands();
        return;
    }
    const cv::Rect region = selection->region;
    std::vector<SpiralMinimap::Band> bands;
    for (const PreviewComponent& component : _previewComponents) {
        if (component.winding < selection->minimumWinding
            || (selection->maximumWinding >= 0
                && component.winding > selection->maximumWinding))
            continue;
        bands.push_back({component.winding,
                         static_cast<float>(component.columnBegin - region.x),
                         static_cast<float>(component.columnEnd - region.x)});
    }
    _windingMinimap->setBands(std::move(bands), 0.0f,
                              static_cast<float>(region.width));
    updateMinimapViewIndicator();
}

void SpiralWorkspace::updateMinimapViewIndicator()
{
    if (!_windingMinimap || !_windingMinimap->isVisible()) return;
    if (!_currentPreview || !_flattenedViewer
        || _flattenedViewer->currentSurface() != _currentPreview.get()) {
        _windingMinimap->setViewIndicator(0.0f, -1.0f);
        return;
    }
    auto* view = _flattenedViewer->graphicsView();
    if (!view || !view->viewport()) return;
    const QRect viewport = view->viewport()->rect();
    const cv::Vec2f scale = _currentPreview->scale();
    const cv::Vec3f center = _currentPreview->center();
    const auto columnAt = [&](const QPoint& viewportPoint) {
        const cv::Vec2f surface = _flattenedViewer->sceneToSurfaceCoords(
            view->mapToScene(viewportPoint));
        return (surface[0] + center[0]) * scale[0];
    };
    _windingMinimap->setViewIndicator(
        columnAt(QPoint(viewport.left(), viewport.center().y())),
        columnAt(QPoint(viewport.right(), viewport.center().y())));
}

void SpiralWorkspace::panFlattenedViewerToColumn(float column)
{
    if (!_currentPreview || !_flattenedViewer
        || _flattenedViewer->currentSurface() != _currentPreview.get())
        return;
    const cv::Vec2f scale = _currentPreview->scale();
    const cv::Vec3f center = _currentPreview->center();
    if (std::abs(scale[0]) < 1e-6f) return;
    // Pan horizontally only: keep the viewer's current vertical position,
    // read back through the inverse of the same mapping the minimap uses.
    auto* view = _flattenedViewer->graphicsView();
    if (!view || !view->viewport()) return;
    const cv::Vec2f viewCenter = _flattenedViewer->sceneToSurfaceCoords(
        view->mapToScene(view->viewport()->rect().center()));
    const float surfaceX = column / scale[0] - center[0];
    _flattenedViewer->centerOnSurfacePoint(cv::Vec2f(surfaceX, viewCenter[1]),
                                           true);
}

void SpiralWorkspace::updateWindingTransitionOverlay()
{
    const auto selection = displayedPreviewSelection();
    if (!_windingTransitionsVisible || !selection || !_currentPreview
        || _previewWindingIds.empty()) {
        _overlay->publishWindingTransitions({}, {});
        return;
    }
    // Tracing on the winding-range crop keeps the curves in the displayed
    // surface's own grid coordinates.
    _overlay->publishWindingTransitions(
        _currentPreview,
        traceWindingTransitions(_previewWindingIds(selection->region),
                                selection->minimumWinding,
                                selection->maximumWinding));
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
    _overlay->publishWindingTransitions({}, {});

    const auto selection = displayedPreviewSelection();
    if (!selection) {
        if (_outputVisible) _state->setSurface("segmentation", nullptr);
        const QString previousRegistration = _currentPreviewRegistrationId;
        _currentPreview.reset();
        if (_windingMinimap) _windingMinimap->clearBands();
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
                cv::Mat windingDepth;
                windingRegion.convertTo(windingDepth, CV_32F);
                loaded->setChannel("d", windingDepth);
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
    if (_brush->dragging()) {
        QTimer::singleShot(50, this, [this, preview, registrationId, generation,
                                     revision, preserveFocus, attempt]() {
            installPreviewAliasWhenIndexed(preview, registrationId, generation, revision,
                                           preserveFocus, attempt);
        });
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
    updateWindingTransitionOverlay();
    updateWindingMinimap();
    // No-op unless the focus is still missing or the automatic default.
    initializePreviewFocus();
    updateSurfaceIntersections();
    for (auto* viewer : _viewerManager->baseViewers()) if (viewer) {
        viewer->invalidateIntersect("segmentation");
        viewer->renderIntersections("Spiral preview installed");
        viewer->requestRender("Spiral preview installed");
    }
    if (!previousRegistration.isEmpty() && previousRegistration != registrationId) {
        if (_brush->usesPaintSurface(std::dynamic_pointer_cast<QuadSurface>(
                _state->surface(previousRegistration.toStdString()))))
            _retainedBrushPreviewIds.insert(previousRegistration);
        else
            _state->setSurface(previousRegistration.toStdString(), nullptr);
    }
}

void SpiralWorkspace::pruneBrushPreviewSurfaces()
{
    const auto retained = _retainedBrushPreviewIds;
    for (const QString& id : retained) {
        if (_brush->usesPaintSurface(std::dynamic_pointer_cast<QuadSurface>(
                _state->surface(id.toStdString())))) continue;
        _state->setSurface(id.toStdString(), nullptr);
        _retainedBrushPreviewIds.remove(id);
    }
}
