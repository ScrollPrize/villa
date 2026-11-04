#include "SpiralWorkspace.hpp"
#include "SpiralPreviewWindings.hpp"

#include "AxisAlignedSliceController.hpp"
#include "CState.hpp"
#include "ConsoleOutputWidget.hpp"
#include "Keybinds.hpp"
#include "SpiralPanel.hpp"
#include "SpiralBrushController.hpp"
#include "SpiralProjectInputs.hpp"
#include "SpiralServiceManager.hpp"
#include "SpiralStoreInputs.hpp"
#include "SpiralStoredFits.hpp"
#include "SpiralStoreEdits.hpp"
#include "SpiralMinimap.hpp"
#include "SurfaceOverlayColors.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "elements/GrowOnlyLabel.hpp"
#include "elements/ViewerSplitGrid.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/SpiralFiberOverlayController.hpp"
#include "overlays/SpiralOverlayController.hpp"
#include "overlays/StoreSurfaceOverlayController.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "volume_viewers/CChunkedVolumeViewer.hpp"
#include "volume_viewers/CVolumeViewerView.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/Uuid.hpp>
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <opencv2/imgcodecs.hpp>

#include <QDialog>
#include <QDockWidget>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QInputDialog>
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

// Why saved drawings could not reach the service's fit, or empty when the
// attached store is the one it fits (or no service has named its store).
QString fitStoreMismatch(const SpiralServiceManager::StoreBinding& binding)
{
    if (!binding.advertised() || binding.matches()) return {};
    if (binding.serviceUuid == binding.localUuid)
        return QObject::tr("The attached store is not a replica of the Spiral service's hub (%1): "
                           "clone that hub to save drawings the fit can use")
            .arg(binding.serviceHubIdentity);
    return QObject::tr("The Spiral service fits store %1, not the attached store %2: "
                       "drawings saved here would never reach the fit")
        .arg(binding.serviceUuid, binding.localUuid);
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
            bindInputStore();
        } else if (event.attachment == inputStore()) {
            if (event.kind == vc::objects::StoreSessionEvent::Kind::Hub && _service)
                _service->refreshLocalHub();
            noteStoreChanges(event);
        }
    });
    connect(_state,&CState::storeRoutesChanged,this,[this]() {
        bindInputStore();
        cancelLineAnnotationDraft();
    });
    // Drafts are synced before anything else hears of a surface change.
    connect(_state, &CState::surfaceChanged, this,
            [this](std::string, std::shared_ptr<Surface> surface, bool edit) {
                syncSurfaceDrafts(edit ? std::dynamic_pointer_cast<QuadSurface>(surface) : nullptr);
            });
    connect(_state, &CState::volumeChanged, this, [this](std::shared_ptr<Volume>, const std::string&) {
        clearSurfaceDrafts();
        syncSurfaceDrafts();
    });
    _viewerManager = std::make_unique<ViewerManager>(_state, _state->pointCollection(), this);
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
    _storeSurfaceOverlay = std::make_unique<StoreSurfaceOverlayController>(_state, _viewerManager.get(), this);
    _brush = std::make_unique<SpiralBrushController>(this);
    _brush->setPatchLocator([this](const QuadSurface& surface, const cv::Vec3f& point, float tolerance) {
        const auto hit = _viewerManager->locate(surface, point, tolerance);
        return hit ? std::optional<cv::Vec3f>(hit->ptr) : std::nullopt;
    });
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
    _fiberOverlay = std::make_unique<SpiralFiberOverlayController>(this);
    _fiberOverlay->bindToViewerManager(_viewerManager.get());
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

    auto* cacheStatsLabel = new GrowOnlyLabel(this);
    cacheStatsLabel->setContentsMargins(8, 0, 8, 0);
    cacheStatsLabel->setText(tr("RAM --  disk --  net --"));
    statusBar()->addPermanentWidget(cacheStatsLabel);
    connect(_viewerManager.get(), &ViewerManager::sharedCacheStatsChanged, cacheStatsLabel,
            [cacheStatsLabel](const QStringList& items) {
                if (!items.isEmpty()) cacheStatsLabel->setText(items.join(QStringLiteral("  ")));
            });

    _service = new SpiralServiceManager(this);
    bindInputStore();
    // A fit reads the open project's facts and umbilicus; the service
    // receives them with every build, as it cannot read the project file.
    _service->setProjectSource(
        [this]() {
            const auto pkg = _state->vpkg();
            if (!pkg) throw std::runtime_error("no project is open");
            return vc3d::spiral::projectLasagna(*pkg, vc3d::remoteCachePathFs());
        },
        [this](const std::optional<vc3d::spiral::ProjectLasagna>& lasagna) {
            const auto pkg = _state->vpkg();
            if (!pkg) throw std::runtime_error("no project is open");
            return vc3d::spiral::projectBlock(pkg->name(), vc3d::spiral::projectVolumeFacts(*pkg), lasagna,
                                              vc3d::spiral::projectFiberInference(*pkg, vc3d::remoteCachePathFs()),
                                              _state->objectStoreSession().get(),
                                              _state->storeFor("umbilicus"));
        });

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
            tr("Drawing draft is ready; use Save Drafts to Store to save it"), 5000);
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
    connect(_panel, &SpiralPanel::cloneHubRequested, this, &SpiralWorkspace::hubReplicaRequested);
    connect(_panel, &SpiralPanel::openStoredFitRequested, this, &SpiralWorkspace::openStoredFit);
    connect(_service, &SpiralServiceManager::inputUpdateProgress, this, [this](const QString& phase, const QString&) {
        if (phase != QStringLiteral("pushing")) return;
        _updatingDrafts.clear();
        _updatingPatches.clear();
        for (const auto& id : _savedDraftIds) _updatingDrafts[id] = _savedCollections.value(id).revision;
        for (const auto& id : _savedPatchIds) _updatingPatches[id] = _brushPatches.value(id).revision;
    });
    connect(_service, &SpiralServiceManager::inputUpdateFinished, this,
            [this](const SpiralServiceManager::InputUpdate& update) {
                const auto drafts = std::exchange(_updatingDrafts, {});
                const auto patches = std::exchange(_updatingPatches, {});
                if (!update.error.isEmpty() || !_brush) return;
                // The fit holds what was saved before the update began: those
                // drafts are shown as its inputs from now on, unless one was
                // edited or saved again meanwhile (then it stays a drawing
                // whose next save replaces its object).
                auto included = [this](const QHash<QString, std::string>& saved,
                                       const QHash<QString, SavedObject>& objects) {
                    QStringList ids;
                    for (auto it = saved.cbegin(); it != saved.cend(); ++it)
                        if (objects.value(it.key()).revision == it.value() && _brush->savedUnchanged(it.key()))
                            ids << it.key();
                    return ids;
                };
                const QStringList fittedDrafts = included(drafts, _savedCollections);
                const QStringList fittedPatches = included(patches, _brushPatches);
                _brush->commitSucceeded(fittedDrafts);
                for (const auto& id : fittedDrafts) {
                    _savedDraftIds.removeAll(id);
                    _savedCollections.remove(id);
                }
                for (const auto& id : fittedPatches) {
                    _brush->removePatchDraft(id);
                    _brushPatches.remove(id);
                    _savedPatchIds.removeAll(id);
                }
            });
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
    connect(_panel, &SpiralPanel::fiberOverlayChanged, this, [this](bool shown) {
        _fiberOverlayVisible = shown;
        if (_fiberOverlay) _fiberOverlay->setVisible(shown);
    });
    const auto applyPointViewTolerance = [this](double tolerance) {
        for (auto& state : _pclOverlays) {
            if (state.overlay) state.overlay->setViewTolerance(tolerance);
        }
        if (_fiberOverlay) _fiberOverlay->setSlabHalfDepth(tolerance);
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
                loadFitInputs();
            });
    // Store commits touching the fitted inputs re-read what they changed, coalesced.
    _fitInputReload = new QTimer(this);
    _fitInputReload->setSingleShot(true);
    _fitInputReload->setInterval(500);
    connect(_fitInputReload, &QTimer::timeout, this, &SpiralWorkspace::loadFitInputChanges);
    // The inputs can be displayed only from the store the service read. A
    // sync's pulls arrive as commits (noteStoreChanges); only a different
    // binding needs a full read.
    connect(_service, &SpiralServiceManager::storeBindingChanged, this, [this]() {
        if (!_inputManifestPath.isEmpty() && _service->storeBinding().identity() != _fitInputBinding)
            loadFitInputs();
    });
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
                clearFitInputs({});
                if (!_service->sessionLeaseId().isEmpty()) return;
                cancelLineAnnotationDraft();
            });
    connect(_service, &SpiralServiceManager::sessionSynchronized, this,
            [this](const QJsonObject& request, const QJsonObject& status) {
                const qint64 generation =
                    status.value(QStringLiteral("session_generation")).toInteger();
                _sessionRunConfig = request.value(QStringLiteral("run")).toObject()
                    .value(QStringLiteral("config")).toObject();
                _sessionScroll = request.value(QStringLiteral("scroll")).toObject();
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
                resetSessionSurfaces(static_cast<quint64>(generation));
            });
    synchronizeVolume(_mainState ? _mainState->currentVolume() : nullptr);
}

void SpiralWorkspace::resetSessionSurfaces(quint64 generation)
{
    if (_shuttingDown || generation < _inputSurfaceGeneration) return;
    _inputSurfaceGeneration = generation;
    // A new session drops the previous one's shell and drafted surfaces; its
    // outer shell arrives with its fit inputs (installFitInputs).
    InputSurfaceLoadResult cleared;
    cleared.categories = {QStringLiteral("shell"), QStringLiteral("brush"),
                          QStringLiteral("ephemeral")};
    installInputSurfaces(cleared, generation);
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
        if (!replacement[entry.category].contains(entry.id)) replacement[entry.category].push_back(entry.id);
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
    for (const QString& id : result.removedIds) {
        surfaceUpdates.emplace_back(id.toStdString(), nullptr);
        for (auto& ids : replacement) ids.removeAll(id);
        replacementSourceIds.remove(id);
        replacementColors.erase(id.toStdString());
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
    // when the fit input categories are hidden.
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

const std::vector<std::string>& SpiralWorkspace::inputKinds()
{
    static const std::vector<std::string> kinds{"surface", "fiber", "point_collection"};
    return kinds;
}

std::optional<vc::objects::AttachmentIdentity> SpiralWorkspace::inputStore() const
{
    const auto store = _state->storeFor(inputKinds().front());
    for (const auto& kind : inputKinds())
        if (_state->storeFor(kind) != store) return std::nullopt;
    return store;
}

void SpiralWorkspace::bindInputStore()
{
    // A local service is launched bound to this store, and any service's
    // inputs are displayed only from the same store.
    const auto store = inputStore();
    QString unavailable;
    if (!store) {
        const auto project = _mainState ? _mainState->vpkg() : _state->vpkg();
        if (!project)
            unavailable = tr("Open a project to start the local Spiral service.");
        else {
            const auto& routes = project->document().data().at("selections");
            const auto selected = routes.value(inputKinds().front(), nlohmann::json());
            const bool sharedRoute = !selected.is_null() && std::all_of(
                inputKinds().begin(), inputKinds().end(), [&](const auto& kind) {
                    return routes.value(kind, nlohmann::json()) == selected;
                });
            unavailable = sharedRoute
                ? tr("The project's selected object store is unavailable. Check its location in File > Stores.")
                : tr("Select the same object store for surfaces, point collections and fibers in File > Stores.");
        }
    }
    if (_service) _service->setObjectStore(_state->objectStoreSession(), store, unavailable);
}

void SpiralWorkspace::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    bindInputStore();
    if (_panel) _panel->activateWorkspace();
}

void SpiralWorkspace::startLineAnnotationDraft()
{
    QString reason = lineAnnotationDraftUnavailableReason();
    if (reason.isEmpty() && !inputStore())
        reason = tr("Route surfaces, fibers and point collections to one store before creating fibers");
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

std::optional<std::array<std::size_t, 3>>
SpiralWorkspace::resolveFiberBaseShape(QString* errorMessage) const
{
    if (!_state || !_state->vpkg()) {
        if (errorMessage) *errorMessage = tr("No volume package is loaded");
        return std::nullopt;
    }
    const auto vpkg = _state->vpkg();
    // The project's selected fiber-inference dataset; nothing is discovered
    // from a service directory layout.
    QStringList candidates;
    const QString selected = QString::fromStdString(vpkg->selectedFiberInferenceDataset());
    if (!selected.isEmpty()) candidates.push_back(selected);

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
            *errorMessage = tr("The project selects no fiber-inference dataset; select one in the project");
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
    // The normals and the fiber inference are the project's selected Lasagna
    // and fiber-inference datasets, which Spiral fits read too.
    return _lineAnnotationController->resolveFiberOptimizationInputs(
        {}, {}, errorMessage);
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
                             tr("Surfaces in the object store are fit inputs when the fit's selection "
                                "includes them (by default, every active surface). %1")
                                 .arg(SpiralServiceManager::inputChangesMessage()));
}

void SpiralWorkspace::addFibersToCurrentFit(const std::vector<uint64_t>& fiberIds)
{
    Q_UNUSED(fiberIds);
    QMessageBox::information(this, tr("Add fibers to the Spiral fit"),
                             tr("Fibers in the object store are fit inputs when the fit's selection "
                                "includes them (by default, every active fiber). %1")
                                 .arg(SpiralServiceManager::inputChangesMessage()));
}

void SpiralWorkspace::finalizeBrushPaint()
{
    submitReadyDrafts(false);
}

void SpiralWorkspace::submitReadyDrafts(bool updateAfter)
{
    // A saved edit is a checked revision of the attached store, pushed to its
    // hub by the session's worker. The fit sees it at its next input update.
    const auto session = _state->objectStoreSession();
    const auto attachment = inputStore();
    QString problem;
    std::optional<double> scale;
    if (!attachment || !session->writable(*attachment))
        problem = tr("Route surfaces, fibers and point collections to one writable store to save Spiral drawings");
    else if (const QString mismatch = fitStoreMismatch(_service->storeBinding()); !mismatch.isEmpty())
        problem = mismatch;
    else if (!_fitInputs || _fitInputs->volumeUuid.empty())
        problem = tr("Spiral drawings are saved in the fit's volume: wait until the fit's inputs are shown");
    else
        scale = fitInputScale(&problem);
    if (!scale) {
        statusBar()->showMessage(problem, 15000);
        return;
    }
    _brush->markDraftsReady();
    QStringList warnings;
    auto patches = _brush->preparePatches(warnings);
    auto collections = _brush->preparePointCollections(warnings);
    if (!warnings.isEmpty()) {
        for (const auto& patch : patches) _brush->finalizationFailed(patch.id);
        for (const auto& collection : collections) _brush->finalizationFailed(collection.id);
        statusBar()->showMessage(warnings.join(QStringLiteral("; ")), 15000);
        return;
    }
    // Prepared here; written on a worker thread, since taking leases on a
    // hosted replica contacts the hub; applied back here.
    struct PatchJob {
        QString id;
        vc3d::spiral::BrushPatchSave save;
        std::optional<vc3d::spiral::SavedBrushPatch> written;
        QString error;
        std::string theirs;  // refused: the revision another editor saved
    };
    struct Result {
        std::vector<PatchJob> patches;
        std::vector<vc3d::spiral::EditablePclDraft> drafts;
        QStringList ids, fitted, savedIds;  // savedIds: the collection of each draft
        std::vector<vc3d::spiral::SavedPointCollection> collections;
        QString collectionError;
        std::string changedUuid, changedTo;  // refused: the object and its revision now
    };
    auto result = std::make_shared<Result>();
    for (const auto& patch : patches) {
        const auto known = _brushPatches.constFind(patch.id);
        const bool remove = patch.operation == QStringLiteral("delete");
        if (remove && known == _brushPatches.cend()) {
            // Never saved: there is nothing in the store to retire.
            _brush->finalizationSucceeded(patch.id);
            continue;
        }
        const SavedObject previous = known == _brushPatches.cend() ? SavedObject{} : *known;
        result->patches.push_back({patch.id, {previous.uuid, previous.revision, patch.id.toStdString(),
                                              patch.surface, remove}, std::nullopt, {}});
    }
    for (const auto& collection : collections) {
        result->ids << collection.id;
        if (collection.draft) {
            result->drafts.push_back(*collection.draft);
        } else {
            // A drawn control-point line: one same-winding collection.
            auto imported = vc3d::spiral::importEditablePcls(collection.document, 1.0, {}, true,
                                                             vc3d::spiral::PclRole::SameWinding);
            if (imported.empty()) continue;
            result->drafts.push_back(std::move(imported.front()));
        }
        if (vc3d::spiral::pointCollectionObject(result->drafts.back())) {
            result->fitted << collection.id;
        } else if (const auto known = _savedCollections.constFind(collection.id);
                   known != _savedCollections.cend()) {
            // Saved before but not in the fit yet: replace that object.
            vc3d::spiral::setPointCollectionObject(result->drafts.back(), known->uuid, known->revision);
        }
        result->savedIds << collection.id;
    }
    if (result->patches.empty() && result->ids.isEmpty()) {
        statusBar()->showMessage(tr("No Spiral drawings to save"), 10000);
        if (updateAfter) _panel->requestInputUpdate();
        return;
    }
    const std::string volume = _fitInputs->volumeUuid;
    const double previewScale = *scale;
    ++_savesInFlight;
    statusBar()->showMessage(tr("Saving Spiral drawings to the object store…"));
    auto* watcher = new QFutureWatcher<void>(this);
    connect(watcher, &QFutureWatcher<void>::finished, this, [this, watcher, result, updateAfter]() {
        watcher->deleteLater();
        --_savesInFlight;
        QStringList errors, saved;
        for (const auto& job : result->patches) {
            if (!job.written) {
                // Saving again replaces their version knowingly: it starts
                // from the revision the refusal named.
                if (!job.theirs.empty() && _brushPatches.contains(job.id))
                    _brushPatches[job.id].revision = job.theirs;
                _brush->finalizationFailed(job.id, job.error);
                errors << tr("%1: %2").arg(job.id, job.error);
                continue;
            }
            if (job.save.remove) _brushPatches.remove(job.id);
            else _brushPatches[job.id] = {job.written->uuid, job.written->revision};
            _brush->finalizationSucceeded(job.id);
            if (!_savedPatchIds.contains(job.id)) _savedPatchIds << job.id;
            saved << job.id;
        }
        if (!result->ids.isEmpty()) {
            if (!result->collectionError.isEmpty()) {
                // A new collection saved before: saving again starts from the
                // revision the refusal named.
                for (auto& saved : _savedCollections)
                    if (!result->changedUuid.empty() && saved.uuid == result->changedUuid)
                        saved.revision = result->changedTo;
                for (const auto& id : result->ids) _brush->finalizationFailed(id, result->collectionError);
                errors << tr("point collections: %1").arg(result->collectionError);
            } else {
                for (std::size_t i = 0; i < result->collections.size(); ++i) {
                    const QString& id = result->savedIds[int(i)];
                    if (result->fitted.contains(id)) continue;
                    if (result->collections[i].revision.empty()) _savedCollections.remove(id);
                    else _savedCollections[id] = {result->collections[i].uuid, result->collections[i].revision};
                }
                for (const auto& id : result->ids) _brush->finalizationSucceeded(id);
                // Edits of fitted collections are shown from the store now;
                // new ones stay drawn until an update brings them into the fit.
                _brush->commitSucceeded(result->fitted);
                for (const auto& id : result->ids)
                    if (!result->fitted.contains(id) && !_savedDraftIds.contains(id)) _savedDraftIds << id;
                saved << result->ids;
            }
        }
        if (!errors.isEmpty()) {
            statusBar()->clearMessage();
            QMessageBox::warning(this, tr("Cannot save Spiral drawings"), errors.join(QStringLiteral("\n")));
            return;
        }
        statusBar()->showMessage(tr("Saved %n drawing(s) to the object store", nullptr, int(saved.size())), 10000);
        if (updateAfter) _panel->requestInputUpdate();
    });
    watcher->setFuture(QtConcurrent::run([session, attachment = *attachment, volume, previewScale, result]() {
        for (auto& job : result->patches) {
            try {
                job.written = vc3d::spiral::saveBrushPatches(*session, attachment, volume, previewScale,
                                                             {job.save}).front();
            } catch (const vc3d::spiral::StoreObjectChanged& changed) {
                job.error = QString::fromUtf8(changed.what());
                job.theirs = changed.current;
            } catch (const std::exception& error) {
                job.error = QString::fromUtf8(error.what());
            }
        }
        if (result->drafts.empty()) return;
        try {
            result->collections = vc3d::spiral::savePointCollections(*session, attachment, volume, result->drafts);
        } catch (const vc3d::spiral::StoreObjectChanged& changed) {
            result->collectionError = QString::fromUtf8(changed.what());
            result->changedUuid = changed.uuid;
            result->changedTo = changed.current;
        } catch (const std::exception& error) {
            result->collectionError = QString::fromUtf8(error.what());
        }
    }));
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
    if (_savesInFlight > 0) {
        // The store writes finish on their own; their results are applied here.
        QMessageBox::information(this, tr("Saving Spiral drawings"),
                                 tr("Spiral drawings are still being saved to the object store. "
                                    "Leave the workspace once the save has finished."));
        return;
    }
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
    clearSurfaceDrafts();
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
    const quint64 serial = ++_previewLoadSerial;
    auto* watcher = new QFutureWatcher<PreviewLoadResult>(this);
    connect(watcher, &QFutureWatcher<PreviewLoadResult>::finished, this, [this, watcher, generation, serial]() {
        const auto result = watcher->result();
        watcher->deleteLater();
        if (!_shuttingDown && generation == _requestedPreviewGeneration && serial == _previewLoadSerial)
            installPreview(result, generation);
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

        if (std::abs(manifest.value(QStringLiteral("output_step_vx")).toDouble() - 20.0) > 1.0e-9)
            return failure(QObject::tr("Invalid Spiral preview winding mapping"));
        QString windingError;
        auto components = vc3d::spiral::parseWindingBounds(
            manifest.value(QStringLiteral("winding_bounds")).toArray(),
            manifest.value(QStringLiteral("winding_ids")).toArray(), &windingError);
        if (!components) return failure(windingError);
        std::vector<PreviewComponent> previewComponents = std::move(*components);

        const QString artifactRoot = QFileInfo(manifestPath).absolutePath();
        const QString windingMapName =
            QDir::cleanPath(manifest.value(QStringLiteral("winding_id_map")).toString());
        if (windingMapName.isEmpty() || QDir::isAbsolutePath(windingMapName)
            || windingMapName == QStringLiteral("..")
            || windingMapName.startsWith(QStringLiteral("../")))
            return failure(QObject::tr("Invalid Spiral winding-ID map path"));
        const QString windingMapPath = QDir(artifactRoot).filePath(windingMapName);
        auto windings = vc3d::spiral::windingIdsFromImage(
            cv::imread(windingMapPath.toStdString(), cv::IMREAD_UNCHANGED), &windingError);
        if (!windings) return failure(windingError);
        cv::Mat_<int32_t> mappedWindings = std::move(*windings);
        if (!vc3d::spiral::windingMapMatchesBounds(mappedWindings, previewComponents, &windingError))
            return failure(windingError);

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

void SpiralWorkspace::openStoredFit(const QString& requested)
{
    if (_shuttingDown) return;
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    const auto attachment = inputStore();
    if (!session || !attachment) {
        statusBar()->showMessage(
            tr("Route Spiral's inputs to the object store the fit was committed to first"), 15000);
        return;
    }
    QString uuid = requested;
    if (uuid.isEmpty()) {
        std::vector<vc3d::spiral::StoredFitEntry> fits;
        try {
            fits = vc3d::spiral::listStoredFits(*session, *attachment);
        } catch (const std::exception& error) {
            statusBar()->showMessage(tr("Could not list committed fits: %1").arg(QString::fromUtf8(error.what())),
                                     15000);
            return;
        }
        if (fits.empty()) {
            statusBar()->showMessage(tr("The object store holds no committed Spiral fits"), 15000);
            return;
        }
        QStringList labels;
        for (const auto& fit : fits) {
            QString label = QString::fromStdString(fit.name);
            if (fit.sourceIteration) label += tr(" — iteration %1").arg(*fit.sourceIteration);
            label += tr(" — %1 by %2").arg(QString::fromStdString(fit.createdAt),
                                          QString::fromStdString(fit.createdBy));
            labels.push_back(label);
        }
        bool chosen = false;
        const QString label = QInputDialog::getItem(this, tr("Open Stored Fit"), tr("Committed Spiral fit:"),
                                                    labels, 0, false, &chosen);
        if (!chosen) return;
        uuid = QString::fromStdString(fits[std::size_t(labels.indexOf(label))].uuid);
    }

    // The fit volume's full-resolution shape, which a committed fit does not
    // repeat: the project's facts of the volume it is bound to.
    std::map<std::string, vc3d::spiral::ProjectVolumeFacts> facts;
    if (const auto pkg = _state->vpkg()) facts = vc3d::spiral::projectVolumeFacts(*pkg);
    // It replaces whatever is displayed: a preview load still in flight is
    // dropped, and the next published preview replaces it in turn.
    const qint64 generation = ++_requestedPreviewGeneration;
    const quint64 serial = ++_previewLoadSerial;
    statusBar()->showMessage(tr("Opening the committed fit %1…").arg(uuid));
    auto* watcher = new QFutureWatcher<PreviewLoadResult>(this);
    connect(watcher, &QFutureWatcher<PreviewLoadResult>::finished, this, [this, watcher, generation, serial]() {
        const auto result = watcher->result();
        watcher->deleteLater();
        if (!_shuttingDown && generation == _requestedPreviewGeneration && serial == _previewLoadSerial)
            installPreview(result, generation);
    });
    watcher->setFuture(QtConcurrent::run(
        [session, attachment = *attachment, uuid = uuid.toStdString(), facts]() -> PreviewLoadResult {
            PreviewLoadResult result;
            try {
                auto fit = vc3d::spiral::readStoredFit(session, attachment, uuid);
                fit.surface->setStrictQuadRenderValidity(true);
                result.surface = fit.surface;
                result.surfaceId = QString::fromStdString(fit.uuid);
                result.components = std::move(fit.components);
                result.windingIds = std::move(fit.windingIds);
                if (const auto found = facts.find(fit.volume); found != facts.end())
                    result.baseShapeZYX = found->second.baseShapeZYX;
                result.storedFitUuid = QString::fromStdString(fit.uuid);
                result.storedFitName = QString::fromStdString(fit.name);
            } catch (const std::exception& error) {
                result.error = QObject::tr("Could not open the committed fit: %1")
                                   .arg(QString::fromUtf8(error.what()));
            }
            return result;
        }));
}

void SpiralWorkspace::installPreview(const PreviewLoadResult& result, qint64 generation)
{
    if (!result.surface) { statusBar()->showMessage(result.error, 15000); return; }
    cancelLineAnnotationDraft();
    _previewSource = result.surface;
    _previewSourceId = result.surfaceId;
    _previewStoredFit = result.storedFitUuid;
    _panel->setDisplayedStoredFit(result.storedFitUuid, result.storedFitName);
    if (!result.storedFitUuid.isEmpty())
        statusBar()->showMessage(tr("Showing the committed fit %1").arg(result.storedFitName), 15000);
    _previewBaseShapeZYX = result.baseShapeZYX;
    updatePreviewCoordinateScale();
    refreshFitOverlays();
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

void SpiralWorkspace::refreshFitOverlays()
{
    for (const auto role : vc3d::spiral::kEditablePclRoles) refreshPclOverlay(role);
    refreshFiberOverlay();
}

std::optional<double> SpiralWorkspace::fitInputScale(QString* error) const
{
    // The fit's inputs are read in reference-volume voxels: the session's
    // scroll's base shape, mapped onto the preview's coordinate domain.
    QString shapeError;
    const auto scrollShape = parseBaseShapeZYX(
        _sessionScroll.value(QStringLiteral("base_shape_zyx")), &shapeError);
    if (!_previewBaseShapeZYX || !scrollShape) {
        *error = tr("The preview or the session's scroll has no coordinate-domain metadata");
        return std::nullopt;
    }
    try {
        return vc::lasagna::dyadicCoordinateScaleBetweenShapes(*scrollShape, *_previewBaseShapeZYX);
    } catch (const std::exception& exception) {
        *error = tr("The coordinate domain is incompatible: %1").arg(QString::fromUtf8(exception.what()));
        return std::nullopt;
    }
}

void SpiralWorkspace::refreshFiberOverlay()
{
    if (!_panel || !_fiberOverlay) return;
    _fiberOverlay->setFibers({});
    if (_fitFibers.empty()) {
        _panel->setFiberOverlayAvailable(false, _fitInputs ? tr("The fit has no fibers") : QString());
        return;
    }
    QString error;
    const auto scale = fitInputScale(&error);
    if (!scale) {
        _panel->setFiberOverlayAvailable(false, error);
        return;
    }
    auto fibers = _fitFibers;
    for (auto& fiber : fibers)
        for (auto& point : fiber.line) point *= *scale;
    _fiberOverlay->setFibers(std::move(fibers));
    _panel->setFiberOverlayAvailable(true);
    _fiberOverlay->setVisible(_fiberOverlayVisible);
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
            role, false, _fitInputs ? tr("The fit has no %1 point collections").arg(roleName)
                                       : QString());
        return;
    }
    QString scaleError;
    const auto inputScale = fitInputScale(&scaleError);
    if (!inputScale) {
        _panel->setPclOverlayAvailable(role, false, scaleError);
        return;
    }
    const double scale = *inputScale;
    if (!state.collection->loadFromJSON(state.documentPath.toStdString())) {
        _panel->setPclOverlayAvailable(
            role, false, tr("The fit's %1 point collections are invalid").arg(roleName));
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
    if (!_fitDocument || event.kind != vc::objects::StoreSessionEvent::Kind::Committed
        || !event.touches({"surface", "point_collection", "fiber", "track", "point-surface",
                           "same-surface", "fiber_branch", "crossing"}))
        return;
    const auto before = _changedOtherInputs.size();
    std::vector<std::string> otherSurfaces;
    for (const auto& change : event.changes) {
        const QString uuid = QString::fromStdString(change.recordUuid);
        const bool object = change.kind == "surface" || change.kind == "point_collection" || change.kind == "fiber"
            || change.kind == "track";
        // A fitted input, or a link that may involve one, is read again by
        // itself. Other inputs only matter to a rebuild.
        if (object && _fitRecords.contains(uuid))
            _pendingObjectChanges.insert(change.recordUuid);
        else if (change.kind == "point-surface" || change.kind == "same-surface" || change.kind == "fiber_branch"
                 || change.kind == "crossing")
            _pendingLinkChanges.insert(change.recordUuid);
        else if (change.kind == "surface")
            otherSurfaces.push_back(change.recordUuid);
        else if (object)
            _changedOtherInputs.insert(uuid);
    }
    if (!otherSurfaces.empty()) {
        // A committed Spiral fit (a commit, or one pulled from the hub) is
        // never in a rebuild's selection: leave it out. Should the read
        // fail, every surface counts.
        std::set<std::string> fits;
        if (const auto session = _state ? _state->objectStoreSession() : nullptr) {
            try {
                fits = vc3d::spiral::spiralFitsAmong(*session, event.attachment, otherSurfaces);
            } catch (const std::exception&) {
                fits.clear();
            }
        }
        for (const auto& uuid : otherSurfaces)
            if (!fits.count(uuid)) _changedOtherInputs.insert(QString::fromStdString(uuid));
    }
    if (!_pendingObjectChanges.empty() || !_pendingLinkChanges.empty()) _fitInputReload->start();
    if (_changedOtherInputs.size() != before) updateInputFreshness();
}

void SpiralWorkspace::updateInputFreshness()
{
    if (!_panel) return;
    if (!_fitInputs) {
        _panel->setInputFreshness({});
        return;
    }
    using vc3d::spiral::InputState;
    using vc3d::spiral::LinkState;
    const auto& inputs = *_fitInputs;
    QStringList notes;
    if (const auto missing = inputs.count(InputState::Missing))
        notes << tr("%n fitted input(s) are no longer in the attached store", "", int(missing));
    if (const auto edited = inputs.count(InputState::Edited))
        notes << tr("%n fitted input(s) were edited since the fit read them", "", int(edited));
    if (const auto metadata = inputs.count(InputState::MetadataChanged))
        notes << tr("%n fitted input(s) have other metadata than the fit read", "", int(metadata));
    if (const auto links = inputs.count(LinkState::Changed))
        notes << tr("%n fitted link(s) differ in the attached store", "", int(links));
    if (const auto links = inputs.count(LinkState::Dropped))
        notes << tr("%n fitted link(s) would no longer be consumed", "", int(links));
    if (!inputs.addedRelationships.empty())
        notes << tr("%n link(s) among the fitted inputs were not fitted", "", int(inputs.addedRelationships.size()));
    if (!_changedOtherInputs.isEmpty())
        notes << tr("%n other surface(s), collection(s) or fiber(s) changed, which a "
                    "rebuild's selection may include", "", int(_changedOtherInputs.size()));
    if (notes.isEmpty()) {
        _panel->setInputFreshness(tr("The attached store holds the fit's inputs as fitted"));
        return;
    }
    // The overlays follow the store; the resident fit does not.
    _panel->setInputFreshness(tr("Stale: %1. The overlays show the store as it is now; "
                                 "Rebuild Fit reads it.")
                                  .arg(notes.join(QStringLiteral("; "))));
}

void SpiralWorkspace::clearFitInputs(const QString& reason)
{
    ++_fitInputGeneration;
    _fitInputs.reset();
    _fitDocument.reset();
    _fitRecords.clear();
    _changedOtherInputs.clear();
    _pendingObjectChanges.clear();
    _pendingLinkChanges.clear();
    if (_fitInputReload) _fitInputReload->stop();
    updateInputFreshness();
    for (auto& state : _pclOverlays) {
        state.document = {};
        state.documentPath.clear();
    }
    _fitFibers.clear();
    InputSurfaceLoadResult cleared;
    cleared.categories = {QStringLiteral("verified"), QStringLiteral("shell")};
    installInputSurfaces(cleared, _inputSurfaceGeneration);
    refreshFitOverlays();
    if (!reason.isEmpty()) statusBar()->showMessage(reason, 15000);
}

void SpiralWorkspace::loadFitInputs()
{
    if (_shuttingDown) return;
    if (_inputManifestPath.isEmpty()) {
        clearFitInputs({});
        return;
    }
    const auto& binding = _service->storeBinding();
    _fitInputBinding = binding.identity();
    const auto session = _state->objectStoreSession();
    const auto attachment = inputStore();
    if (!binding.matches() || !session || !attachment) {
        // Never substitute another store's objects, or current heads, for
        // the fitted inputs.
        clearFitInputs(binding.localUuid.isEmpty()
            ? tr("Fit inputs are not displayed: surfaces, fibers and point collections are not "
                 "routed to one store")
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
        clearFitInputs(tr("The fit's input manifest cannot be read: %1").arg(QString::fromUtf8(error.what())));
        return;
    }
    const quint64 generation = ++_fitInputGeneration;
    // Commits from here on are re-read after this read installs; earlier
    // ones are in it.
    _fitDocument = std::make_shared<const nlohmann::json>(document);
    _pendingObjectChanges.clear();
    _pendingLinkChanges.clear();
    _fitRecords.clear();
    for (const auto& pin : vc3d::spiral::manifestPins(document)) _fitRecords.insert(QString::fromStdString(pin.uuid));
    auto* watcher = new QFutureWatcher<std::shared_ptr<vc3d::spiral::FitInputs>>(this);
    connect(watcher, &QFutureWatcher<std::shared_ptr<vc3d::spiral::FitInputs>>::finished, this,
            [this, watcher, generation]() {
                const auto inputs = watcher->result();
                watcher->deleteLater();
                if (!_shuttingDown && generation == _fitInputGeneration)
                    installFitInputs(inputs, generation);
            });
    watcher->setFuture(QtConcurrent::run([session, attachment = *attachment, document]() {
        auto inputs = std::make_shared<vc3d::spiral::FitInputs>();
        try {
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                *inputs = vc3d::spiral::readFitInputs(db, txn, document);
            });
        } catch (const std::exception& error) {
            inputs->problems.push_back(error.what());
        }
        return inputs;
    }));
}

void SpiralWorkspace::installFitInputs(
    const std::shared_ptr<vc3d::spiral::FitInputs>& inputs, quint64 generation)
{
    if (generation != _fitInputGeneration) return;
    // Reloads of one fit keep counting the other inputs a rebuild may take.
    if (!_fitInputs || _fitInputs->digest != inputs->digest) _changedOtherInputs.clear();
    _fitInputs = inputs;
    _installedFitInputGeneration = generation;
    updateInputFreshness();
    InputSurfaceLoadResult surfaces;
    surfaces.categories = {QStringLiteral("verified"), QStringLiteral("shell")};
    for (const auto& entry : inputs->surfaces)
        if (entry.surface) surfaces.surfaces.push_back(fitSurfaceEntry(entry));
    if (inputs->outerShell && inputs->outerShell->surface)
        surfaces.surfaces.push_back(fitSurfaceEntry(*inputs->outerShell, QStringLiteral("shell")));
    installInputSurfaces(surfaces, _inputSurfaceGeneration);
    rebuildFitPclDocuments();
    rebuildFitFibers();
    refreshFitOverlays();
    showFitInputProblems(*inputs);
    if (!_pendingObjectChanges.empty() || !_pendingLinkChanges.empty()) _fitInputReload->start();
}

SpiralWorkspace::InputSurfaceEntry SpiralWorkspace::fitSurfaceEntry(const vc3d::spiral::FitSurface& entry,
                                                                    const QString& category) const
{
    const QString uuid = QString::fromStdString(entry.pin.uuid);
    // Keyed and coloured by object UUID, which survives renames.
    const QString id = QStringLiteral("spiral/%1/%2/%3")
                           .arg(category, QString::fromStdString(_fitInputs->digest).left(12), uuid);
    // The CState key is display-specific, but store queries need the actual
    // object UUID. Do not turn a committed input into a session draft.
    entry.surface->id = entry.pin.uuid;
    return {category, id, uuid, entry.surface};
}

void SpiralWorkspace::rebuildFitPclDocuments()
{
    // Absolute and relative collections carry windings and share the
    // relative overlay; unannotated ones are same-winding collections.
    std::map<vc3d::spiral::PclRole, QJsonObject> collections;
    int collectionId = 0;
    for (const auto& entry : _fitInputs->pointCollections) {
        if (entry.ids.empty()) continue;
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
                {QLatin1String(vc3d::spiral::kLocalObjectRevisionKey), QString::fromStdString(entry.revision)},
                {QStringLiteral("winding_is_absolute"), entry.absolute}}}};
    }
    const QString digest = QString::fromStdString(_fitInputs->digest).left(12);
    for (const auto role : vc3d::spiral::kEditablePclRoles) {
        auto& state = pclOverlay(role);
        state.document = {};
        state.documentPath.clear();
        if (!collections.contains(role) || !_fitPclDocuments.isValid()) continue;
        state.document = QJsonObject{{QStringLiteral("vc_pointcollections_json_version"), QStringLiteral("1")},
                                     {QStringLiteral("collections"), collections[role]}};
        const QString path = _fitPclDocuments.filePath(
            QStringLiteral("%1-%2.json").arg(digest, vc3d::spiral::pclRoleDisplayName(role)));
        QSaveFile out(path);
        const QByteArray bytes = QJsonDocument(state.document).toJson(QJsonDocument::Compact);
        if (out.open(QIODevice::WriteOnly) && out.write(bytes) == bytes.size() && out.commit())
            state.documentPath = path;
    }
}

void SpiralWorkspace::rebuildFitFibers()
{
    _fitFibers.clear();
    for (const auto& entry : _fitInputs->fibers)
        if (entry.line.size() > 1)
            _fitFibers.push_back({entry.line, QString::fromStdString(entry.family)});
}

void SpiralWorkspace::showFitInputProblems(const vc3d::spiral::FitInputs& inputs)
{
    QStringList notes;
    for (const auto& problem : inputs.problems) notes << QString::fromStdString(problem);
    const auto add = [&notes](const auto& entries) {
        for (const auto& entry : entries)
            if (!entry.problem.empty()) notes << QString::fromStdString(entry.problem);
    };
    add(inputs.surfaces);
    add(inputs.pointCollections);
    add(inputs.fibers);
    if (!notes.isEmpty()) statusBar()->showMessage(notes.join(QStringLiteral("; ")), 15000);
}

void SpiralWorkspace::loadFitInputChanges()
{
    // One read at a time, on top of the installed full read of the current
    // manifest; what arrives meanwhile waits for it.
    if (_shuttingDown || !_fitInputs || !_fitDocument || _fitChangesReading
        || _installedFitInputGeneration != _fitInputGeneration)
        return;
    if (_pendingObjectChanges.empty() && _pendingLinkChanges.empty()) return;
    const auto session = _state->objectStoreSession();
    const auto attachment = inputStore();
    if (!session || !attachment) return;
    _fitChangesReading = true;
    const quint64 generation = _fitInputGeneration;
    auto objects = std::exchange(_pendingObjectChanges, {});
    auto links = std::exchange(_pendingLinkChanges, {});
    using Changes = std::shared_ptr<vc3d::spiral::FitInputChanges>;
    auto* watcher = new QFutureWatcher<Changes>(this);
    connect(watcher, &QFutureWatcher<Changes>::finished, this, [this, watcher, generation]() {
        const auto changes = watcher->result();
        watcher->deleteLater();
        _fitChangesReading = false;
        if (_shuttingDown) return;
        if (generation == _fitInputGeneration && generation == _installedFitInputGeneration && _fitInputs)
            installFitInputChanges(*changes);
        if (!_pendingObjectChanges.empty() || !_pendingLinkChanges.empty()) _fitInputReload->start();
    });
    watcher->setFuture(QtConcurrent::run([session, attachment = *attachment, document = _fitDocument,
                                          objects = std::move(objects), links = std::move(links)]() {
        auto changes = std::make_shared<vc3d::spiral::FitInputChanges>();
        try {
            session->read(attachment, [&](const auto& db, const auto& txn, const auto&) {
                *changes = vc3d::spiral::readFitInputChanges(db, txn, *document, objects, links);
            });
        } catch (const std::exception& error) {
            changes->inputs.problems.push_back(error.what());
        }
        return changes;
    }));
}

void SpiralWorkspace::installFitInputChanges(vc3d::spiral::FitInputChanges changes)
{
    // Only what changed is replaced: one surface, the PCL overlays when a
    // collection changed, the fiber overlay when a fiber did.
    InputSurfaceLoadResult surfaces;
    for (const auto& entry : changes.inputs.surfaces) {
        if (entry.surface) {
            surfaces.surfaces.push_back(fitSurfaceEntry(entry));
        } else {
            surfaces.removedIds << QStringLiteral("spiral/verified/%1/%2")
                                       .arg(QString::fromStdString(_fitInputs->digest).left(12),
                                            QString::fromStdString(entry.pin.uuid));
        }
    }
    const bool collectionsChanged = !changes.inputs.pointCollections.empty();
    const bool fibersChanged = !changes.inputs.fibers.empty();
    showFitInputProblems(changes.inputs);
    vc3d::spiral::applyFitInputChanges(*_fitInputs, std::move(changes));
    if (!surfaces.surfaces.empty() || !surfaces.removedIds.isEmpty())
        installInputSurfaces(surfaces, _inputSurfaceGeneration);
    if (collectionsChanged) {
        rebuildFitPclDocuments();
        for (const auto role : vc3d::spiral::kEditablePclRoles) refreshPclOverlay(role);
    }
    if (fibersChanged) {
        rebuildFitFibers();
        refreshFiberOverlay();
    }
    updateInputFreshness();
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
    // A committed fit is already held whole; an artifact is decoded lazily.
    const bool stored = !_previewStoredFit.isEmpty();
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
                installPreviewAlias(preview, selection.registrationId, generation, revision, preserveFocus);
            });
    watcher->setFuture(QtConcurrent::run(
        [source, selection = *selection, region, windingRegion, stored]() {
            try {
                auto loaded = stored
                    ? std::make_unique<QuadSurface>(
                          new cv::Mat_<cv::Vec3f>((*source->rawPointsPtr())(region).clone()), source->scale())
                    : load_quad_from_tifxyz_region(source->path, region);
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

void SpiralWorkspace::installPreviewAlias(
    const std::shared_ptr<QuadSurface>& preview, const QString& registrationId,
    qint64 generation, quint64 revision, bool preserveFocus)
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
        QTimer::singleShot(50, this, [this, preview, registrationId, generation, revision, preserveFocus]() {
            installPreviewAlias(preview, registrationId, generation, revision, preserveFocus);
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

void SpiralWorkspace::syncSurfaceDrafts(const std::shared_ptr<QuadSurface>& edited)
{
    if (_shuttingDown) return;
    const auto session = _state ? _state->objectStoreSession() : nullptr;
    if (!session) {
        return;
    }
    const auto& volume = _state->annotationVolume();
    std::set<std::string> shown;
    if (volume) {
        for (const auto& surface : _state->surfaces()) {
            const auto quad = std::dynamic_pointer_cast<QuadSurface>(surface);
            if (!quad) {
                continue;
            }
            if (_fitRecords.contains(QString::fromStdString(quad->id))) {
                continue;
            }
            if (!vc::objects::util::isUuid(quad->id)) {
                quad->id = vc::objects::util::newUuid();
            }
            if (!shown.insert(quad->id).second) {
                continue;
            }
            auto& drafted = _surfaceDrafts[quad->id];
            if (drafted.lock() == quad && quad != edited) {
                continue;
            }
            session->setDraft(quad->id, quad_surface_grid(*quad), vc::objects::volumeCoordinates(*volume));
            drafted = quad;
        }
    }
    for (auto it = _surfaceDrafts.begin(); it != _surfaceDrafts.end();) {
        if (shown.contains(it->first)) {
            ++it;
            continue;
        }
        session->clearDraft(it->first);
        it = _surfaceDrafts.erase(it);
    }
}

void SpiralWorkspace::clearSurfaceDrafts()
{
    if (const auto session = _state ? _state->objectStoreSession() : nullptr)
        for (const auto& [uuid, surface] : _surfaceDrafts) session->clearDraft(uuid);
    _surfaceDrafts.clear();
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
