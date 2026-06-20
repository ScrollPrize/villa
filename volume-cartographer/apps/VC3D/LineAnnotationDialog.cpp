#include "LineAnnotationDialog.hpp"

#include "FiberSliceGeometry.hpp"
#include "Keybinds.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "ViewerManager.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QBrush>
#include <QCloseEvent>
#include <QComboBox>
#include <QEvent>
#include <QKeyEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPushButton>
#include <QRect>
#include <QResizeEvent>
#include <QShortcut>
#include <QSplitter>
#include <QTimer>
#include <QVariantAnimation>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {

// Half-width (in line-point indices) of the window used to least-squares-fit the side-view
// plane orientation around the cursor. The fit is interpolated between adjacent window centers
// so the orientation tracks the cursor continuously (see updateSidePlaneSurface).
constexpr int kSideFitHalfWindow = 20;
constexpr float kCurrentCutRotationStepRadians = 3.14159265358979323846f / 36.0f;

std::string staticStripOverlayKey(const std::string& surfaceName)
{
    return surfaceName + "_static";
}

std::string dynamicStripOverlayKey(const std::string& surfaceName)
{
    return surfaceName + "_dynamic";
}

CChunkedVolumeViewer::CameraState generatedPaneCamera(CChunkedVolumeViewer* viewer,
                                                      const CChunkedVolumeViewer::CameraState& fallback)
{
    CChunkedVolumeViewer::CameraState camera = fallback;
    camera.surfacePtrX = 0.0f;
    camera.surfacePtrY = 0.0f;
    camera.zOffset = 0.0f;
    camera.zOffsetWorldDir = {0, 0, 0};

    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad) {
        return camera;
    }

    const cv::Size size = quad->size();
    if (size.width <= 0 || size.height <= 0) {
        return camera;
    }

    constexpr float kNominalGeneratedRowWidth = 900.0f;
    constexpr float kNominalGeneratedRowHeight = 260.0f;
    constexpr float kPadding = 0.85f;
    const float scaleX = kNominalGeneratedRowWidth / static_cast<float>(std::max(1, size.width));
    const float scaleY = kNominalGeneratedRowHeight / static_cast<float>(std::max(1, size.height));
    camera.scale = std::clamp(std::min(scaleX, scaleY) * kPadding, 0.5f, 16.0f);
    return camera;
}

std::optional<cv::Vec2f> generatedStripSurfaceCenter(CChunkedVolumeViewer* viewer,
                                                     double linePosition)
{
    auto* quad = viewer ? dynamic_cast<QuadSurface*>(viewer->currentSurface()) : nullptr;
    if (!quad || !std::isfinite(linePosition)) {
        return std::nullopt;
    }
    const auto* points = quad->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return std::nullopt;
    }
    const float surfaceX = (static_cast<float>(linePosition) -
                            static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float centerRow = static_cast<float>(points->rows / 2);
    const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return cv::Vec2f{surfaceX, surfaceY};
}

bool finitePoint(const cv::Vec3f& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

cv::Vec3f normalizedOrNan(const cv::Vec3f& vector)
{
    const float n = cv::norm(vector);
    if (!finitePoint(vector) || n <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return vector * (1.0f / n);
}

} // namespace

LineAnnotationDialog::LineAnnotationDialog(ViewerManager* viewerManager,
                                           VolumeSelectorFactory volumeSelectorFactory,
                                           QWidget* parent)
    : QMainWindow(parent)
    , _viewerManager(viewerManager)
{
    setWindowTitle(tr("Line Annotation"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(900, 700);

    auto* content = new QWidget(this);
    setCentralWidget(content);

    _layout = new QVBoxLayout(content);
    _layout->setContentsMargins(0, 0, 0, 0);
    _layout->setSpacing(0);

    auto* buttonRow = new QWidget(content);
    buttonRow->installEventFilter(this);
    auto* buttonLayout = new QHBoxLayout(buttonRow);
    buttonLayout->setContentsMargins(6, 6, 6, 6);
    buttonLayout->setSpacing(6);
    _initialDirectionCombo = new QComboBox(buttonRow);
    _initialDirectionCombo->addItem(tr("sideways"), static_cast<int>(InitialDirectionMode::Sideways));
    _initialDirectionCombo->addItem(tr("z (in/out)"), static_cast<int>(InitialDirectionMode::ZInOut));
    _initialDirectionCombo->setCurrentIndex(1);
    _initialDirectionCombo->installEventFilter(this);
    buttonLayout->addWidget(_initialDirectionCombo);
    _reoptimizationCombo = new QComboBox(buttonRow);
    _reoptimizationCombo->addItem(tr("auto-reopt"),
                                  static_cast<int>(ReoptimizationMode::AutoReoptimize));
    _reoptimizationCombo->addItem(tr("no optimization"),
                                  static_cast<int>(ReoptimizationMode::NoOptimization));
    _reoptimizationCombo->setCurrentIndex(0);
    _reoptimizationCombo->installEventFilter(this);
    buttonLayout->addWidget(_reoptimizationCombo);
    connect(_reoptimizationCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            [this](int) {
                emit reoptimizationModeChanged(reoptimizationMode());
            });
    _shiftScrollCombo = new QComboBox(buttonRow);
    _shiftScrollCombo->addItem(tr("along-line"),
                               static_cast<int>(ShiftScrollMode::AlongLine));
    _shiftScrollCombo->addItem(tr("straight"),
                               static_cast<int>(ShiftScrollMode::StraightNormal));
    _shiftScrollCombo->setCurrentIndex(0);
    _shiftScrollCombo->installEventFilter(this);
    buttonLayout->addWidget(_shiftScrollCombo);
    if (volumeSelectorFactory) {
        if (auto* volumeSelector = volumeSelectorFactory(buttonRow)) {
            volumeSelector->installEventFilter(this);
            buttonLayout->addWidget(volumeSelector);
        }
    }
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setText(tr("Step: %1").arg(_viewerManager ? _viewerManager->sliceStepSize() : 1));
    _sliceStepLabel->setToolTip(tr("Shift+Scroll step size. Use Shift+G / Shift+H to adjust."));
    _sliceStepLabel->installEventFilter(this);
    buttonLayout->addWidget(_sliceStepLabel);
    if (_viewerManager) {
        connect(_viewerManager, &ViewerManager::sliceStepSizeChanged, this, [this](int size) {
            if (_sliceStepLabel) {
                _sliceStepLabel->setText(tr("Step: %1").arg(size));
            }
        });
    }
    _showAsMeshButton = new QPushButton(tr("show as mesh"), buttonRow);
    _showAsMeshButton->setEnabled(false);
    _showAsMeshButton->installEventFilter(this);
    buttonLayout->addWidget(_showAsMeshButton);
    _fullOptimizationButton = new QPushButton(tr("reinit reopt"), buttonRow);
    _fullOptimizationButton->setEnabled(false);
    _fullOptimizationButton->installEventFilter(this);
    buttonLayout->addWidget(_fullOptimizationButton);
    _resetViewsButton = new QPushButton(tr("Reset views"), buttonRow);
    _resetViewsButton->setEnabled(false);
    _resetViewsButton->installEventFilter(this);
    buttonLayout->addWidget(_resetViewsButton);
    buttonLayout->addStretch(1);
    _layout->addWidget(buttonRow, 0);
    connect(_showAsMeshButton, &QPushButton::clicked, this, [this]() {
        emit showAsMeshRequested();
    });
    connect(_fullOptimizationButton, &QPushButton::clicked, this, [this]() {
        emit fullOptimizationRequested();
    });
    connect(_resetViewsButton, &QPushButton::clicked, this, [this]() {
        resetGeneratedViews();
    });
    installGeneratedViewShortcuts();

    _mdiArea = new QMdiArea(content);
    _mdiArea->installEventFilter(this);
    _layout->addWidget(_mdiArea);
}

LineAnnotationDialog::InitialDirectionMode LineAnnotationDialog::initialDirectionMode() const
{
    if (!_initialDirectionCombo) {
        return InitialDirectionMode::Sideways;
    }
    return static_cast<InitialDirectionMode>(_initialDirectionCombo->currentData().toInt());
}

LineAnnotationDialog::ReoptimizationMode LineAnnotationDialog::reoptimizationMode() const
{
    if (!_reoptimizationCombo) {
        return ReoptimizationMode::AutoReoptimize;
    }
    return static_cast<ReoptimizationMode>(_reoptimizationCombo->currentData().toInt());
}

LineAnnotationDialog::ShiftScrollMode LineAnnotationDialog::shiftScrollMode() const
{
    if (!_shiftScrollCombo) {
        return ShiftScrollMode::AlongLine;
    }
    return static_cast<ShiftScrollMode>(_shiftScrollCombo->currentData().toInt());
}

void LineAnnotationDialog::setGeneratedControlPoints(
    std::vector<GeneratedOverlay::ControlPointMarker> controlPoints)
{
    if (!_hasGeneratedViews) {
        return;
    }
    _generatedViews.controlPoints = std::move(controlPoints);
    _generatedControlIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
            _generatedViews.controlPoints);
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::setGeneratedPredSnapPoints(
    std::vector<GeneratedOverlay::PredSnapMarker> predSnapPoints)
{
    if (!_hasGeneratedViews) {
        return;
    }
    _generatedViews.predSnapPoints = std::move(predSnapPoints);
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::setOptimizationBusy(bool busy)
{
    auto* content = centralWidget();
    if (!content) {
        return;
    }
    if (!_optimizationOverlay) {
        auto* overlay = new QWidget(content);
        overlay->setObjectName(QStringLiteral("lineAnnotationOptimizationOverlay"));
        overlay->setAttribute(Qt::WA_StyledBackground, true);
        overlay->setStyleSheet(QStringLiteral(
            "#lineAnnotationOptimizationOverlay { background-color: rgba(32, 32, 32, 120); }"
            "#lineAnnotationOptimizationOverlay QLabel { color: white; font-weight: 600; }"));
        auto* layout = new QVBoxLayout(overlay);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* label = new QLabel(tr("Optimizing..."), overlay);
        label->setAlignment(Qt::AlignCenter);
        layout->addStretch(1);
        layout->addWidget(label);
        layout->addStretch(1);
        overlay->hide();
        _optimizationOverlay = overlay;
    }
    updateOptimizationOverlayGeometry();
    _optimizationOverlay->setVisible(busy);
    if (busy) {
        _optimizationOverlay->raise();
    }
}

void LineAnnotationDialog::setCloseAfterFinalizationAllowed(bool allowed)
{
    _closeAfterFinalizationAllowed = allowed;
}

void LineAnnotationDialog::closeEvent(QCloseEvent* event)
{
    if (_closeAfterFinalizationAllowed) {
        QMainWindow::closeEvent(event);
        return;
    }
    event->ignore();
    emit closeFinalizationRequested(event);
}

CChunkedVolumeViewer* LineAnnotationDialog::addPane(
    const std::string& surfaceName,
    const QString& title,
    const CChunkedVolumeViewer::CameraState& camera)
{
    if (!_viewerManager || !_mdiArea) {
        return nullptr;
    }

    auto* base = _viewerManager->createViewer(surfaceName,
                                             title,
                                             _mdiArea,
                                             ViewerManager::ViewerRole::Annotation);
    if (!base) {
        return nullptr;
    }

    auto* viewer = qobject_cast<CChunkedVolumeViewer*>(base->asQObject());
    if (!viewer) {
        return nullptr;
    }

    auto* subWindow = qobject_cast<QMdiSubWindow*>(viewer->parentWidget());
    if (subWindow) {
        subWindow->showMaximized();
        connect(subWindow, &QObject::destroyed, this, [this, surfaceName]() {
            if (!_suppressPaneClosed) {
                emit paneClosed(surfaceName);
            }
        });
    }

    viewer->applyCameraState(camera, false);
    bindPaneInteractions(surfaceName, viewer, true);
    _panes.push_back(Pane{surfaceName, viewer, subWindow});
    return viewer;
}

bool LineAnnotationDialog::setGeneratedRows(
    const std::vector<std::vector<std::pair<std::string, QString>>>& rows,
    const CChunkedVolumeViewer::CameraState& camera,
    const std::map<std::string, GeneratedOverlay>& overlays)
{
    if (!_viewerManager || !_layout) {
        return false;
    }

    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(false);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(false);
    }
    if (_resetViewsButton) {
        _resetViewsButton->setEnabled(false);
    }

    clearGeneratedOverlayRefreshConnections();
    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
    _panes.clear();
    _stripViewers.clear();
    _currentCutViewer = nullptr;
    _sideCutViewer = nullptr;
    _hasGeneratedViews = false;
    _currentCutManualRotation = cv::Matx33f::eye();
    _currentCutManualRotationActive = false;
    _currentCutStraightOffsetActive = false;
    _generatedControlIndex = {};
    _haveInitialCurrentCutCamera = false;
    _haveInitialSideCutCamera = false;
    _initialStripCameras.clear();

    for (const auto& row : rows) {
        if (row.empty()) {
            continue;
        }

        auto* rowWidget = new QWidget(this);
        rowWidget->installEventFilter(this);
        auto* rowLayout = new QHBoxLayout(rowWidget);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(0);
        _layout->addWidget(rowWidget, 1);

        for (const auto& [surfaceName, title] : row) {
            auto* base = _viewerManager->createViewerInWidget(
                surfaceName,
                rowWidget,
                ViewerManager::ViewerRole::Annotation);
            if (!base) {
                return false;
            }
            auto* viewer = qobject_cast<CChunkedVolumeViewer*>(base->asQObject());
            if (!viewer) {
                return false;
            }
            viewer->setObjectName(title);
            viewer->applyCameraState(generatedPaneCamera(viewer, camera), false);
            bindPaneInteractions(surfaceName, viewer, false);
            rowLayout->addWidget(viewer, 1);
            _panes.push_back(Pane{surfaceName, viewer, {}});
            if (auto overlay = overlays.find(surfaceName); overlay != overlays.end()) {
                setGeneratedOverlay(surfaceName, viewer, overlay->second);
            }
        }
    }
    const bool ok = !_panes.empty();
    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(ok);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(ok);
    }
    return ok;
}

void LineAnnotationDialog::bindPaneInteractions(const std::string& surfaceName,
                                                CChunkedVolumeViewer* viewer,
                                                bool seedPlacementEnabled)
{
    if (!viewer) {
        return;
    }

    viewer->setLineAnnotationPlacementPreviewEnabled(seedPlacementEnabled);
    viewer->installEventFilter(this);
    if (auto* view = viewer->graphicsView()) {
        view->installEventFilter(this);
        if (auto* viewport = view->viewport()) {
            viewport->installEventFilter(this);
        }
    }
    if (!seedPlacementEnabled) {
        return;
    }
    connect(viewer,
            &CChunkedVolumeViewer::sendLineAnnotationSeedRequested,
            this,
            [this, surfaceName](cv::Vec3f volumePoint, QPointF scenePoint) {
                emit lineSeedRequested(surfaceName, volumePoint, scenePoint);
            });
}

void LineAnnotationDialog::connectGeneratedOverlayRefresh(CChunkedVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }
    _generatedOverlayRefreshConnections.push_back(
        viewer->connectOverlaysUpdated(this, [this]() {
            rebuildGeneratedOverlays();
        }));
}

void LineAnnotationDialog::clearGeneratedOverlayRefreshConnections()
{
    for (const auto& connection : _generatedOverlayRefreshConnections) {
        QObject::disconnect(connection);
    }
    _generatedOverlayRefreshConnections.clear();
}

void LineAnnotationDialog::setGeneratedOverlay(const std::string& surfaceName,
                                               CChunkedVolumeViewer* viewer,
                                               const GeneratedOverlay& overlay)
{
    if (!viewer) {
        return;
    }

    QPointer<CChunkedVolumeViewer> viewerPtr(viewer);
    const auto apply = [this, surfaceName, viewerPtr, overlay]() {
        if (!viewerPtr) {
            return;
        }
        applyGeneratedOverlay(surfaceName, viewerPtr, overlay);
    };
    viewer->renderVisible(true, "line annotation overlay");
    apply();
    viewer->connectOverlaysUpdated(this, apply);
}

bool LineAnnotationDialog::setGeneratedLineViews(
    const GeneratedViews& views,
    const CChunkedVolumeViewer::CameraState& camera)
{
    if (!_viewerManager || !_layout || views.linePoints.empty() ||
        views.lineUpVectors.size() != views.linePoints.size() ||
        !views.currentCutSurface || !views.sideCutSurface) {
        return false;
    }

    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(false);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(false);
    }
    if (_resetViewsButton) {
        _resetViewsButton->setEnabled(false);
    }

    const bool replacingGeneratedViews = _hasGeneratedViews;
    const double previousCurrentLinePosition = _currentLinePosition;

    bool haveCurrentCutCamera = false;
    CChunkedVolumeViewer::CameraState currentCutCamera;
    if (_currentCutViewer) {
        currentCutCamera = _currentCutViewer->cameraState();
        haveCurrentCutCamera = true;
    }

    bool haveSideCutCamera = false;
    CChunkedVolumeViewer::CameraState sideCutCamera;
    if (_sideCutViewer) {
        sideCutCamera = _sideCutViewer->cameraState();
        haveSideCutCamera = true;
    }

    std::vector<CChunkedVolumeViewer::CameraState> stripCameras;
    stripCameras.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            stripCameras.push_back(viewer->cameraState());
        }
    }

    if (_generatedOuterSplitter) {
        _savedOuterSplitterSizes = _generatedOuterSplitter->sizes();
    }
    if (_generatedTopSplitter) {
        _savedTopSplitterSizes = _generatedTopSplitter->sizes();
    }
    if (_generatedStripSplitter) {
        _savedStripSplitterSizes = _generatedStripSplitter->sizes();
    }

    clearGeneratedOverlayRefreshConnections();
    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
    for (auto& container : _generatedContainers) {
        if (container) {
            _layout->removeWidget(container);
            delete container;
        }
    }
    _generatedContainers.clear();
    _generatedTopWidget = nullptr;
    _panes.clear();
    _stripViewers.clear();
    _currentCutViewer = nullptr;
    _sideCutViewer = nullptr;

    _generatedViews = views;
    _linePointsd.clear();
    _linePointsd.reserve(_generatedViews.linePoints.size());
    for (const auto& p : _generatedViews.linePoints) {
        _linePointsd.emplace_back(p[0], p[1], p[2]);
    }
    _sideFitBracket[0] = SideFit{};
    _sideFitBracket[1] = SideFit{};
    _generatedControlIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(
            _generatedViews.controlPoints);
    _hasGeneratedViews = true;
    _currentCutFollowsStripMouse = views.initialCurrentCutFollowsStripMouse;
    _currentCutStraightOffsetActive = false;
    if (!replacingGeneratedViews) {
        _currentCutManualRotation = cv::Matx33f::eye();
        _currentCutManualRotationActive = false;
    }
    const double maxLinePosition = static_cast<double>(views.linePoints.size() - 1);
    _currentLinePosition = replacingGeneratedViews
        ? std::clamp(previousCurrentLinePosition, 0.0, maxLinePosition)
        : std::clamp(static_cast<double>(views.initialCenterIndex), 0.0, maxLinePosition);
    if (!updatePlaneSurface(views.currentCutSurface.get(), _currentLinePosition)) {
        return false;
    }
    if (!updateSidePlaneSurface(views.sideCutSurface.get(), _currentLinePosition)) {
        return false;
    }

    auto* outerSplitter = new QSplitter(Qt::Vertical, this);
    outerSplitter->setObjectName(QStringLiteral("LineAnnotationOuterSplitter"));
    outerSplitter->setChildrenCollapsible(false);
    outerSplitter->installEventFilter(this);
    _generatedContainers.push_back(outerSplitter);
    _generatedOuterSplitter = outerSplitter;
    _layout->addWidget(outerSplitter, 1);

    auto* topSplitter = new QSplitter(Qt::Horizontal, outerSplitter);
    topSplitter->setObjectName(QStringLiteral("LineAnnotationTopSplitter"));
    topSplitter->setChildrenCollapsible(false);
    topSplitter->installEventFilter(this);
    _generatedTopWidget = topSplitter;
    _generatedTopSplitter = topSplitter;
    outerSplitter->addWidget(topSplitter);

    auto* currentBase = _viewerManager->createViewerInWidget(
        views.currentCutName,
        topSplitter,
        ViewerManager::ViewerRole::Annotation);
    auto* currentViewer = currentBase
        ? qobject_cast<CChunkedVolumeViewer*>(currentBase->asQObject())
        : nullptr;
    if (!currentViewer) {
        return false;
    }
    currentViewer->setObjectName(tr("Current Line Cut"));
    currentViewer->applyCameraState(haveCurrentCutCamera
                                        ? currentCutCamera
                                        : generatedPaneCamera(currentViewer, camera),
                                    false);
    if (!haveCurrentCutCamera && finitePoint(_generatedViews.focusPoint)) {
        currentViewer->centerOnVolumePoint(_generatedViews.focusPoint, false);
    }
    currentViewer->setShiftScrollOverride(
        [this](int steps, QPointF, Qt::KeyboardModifiers) {
            if (shiftScrollMode() == ShiftScrollMode::StraightNormal) {
                return shiftCurrentCutPlaneStraightByScrollSteps(steps);
            }
            return shiftCurrentLinePositionByScrollSteps(steps);
        });
    bindPaneInteractions(views.currentCutName, currentViewer, false);
    connect(currentViewer,
            &CChunkedVolumeViewer::sendMousePressVolume,
            this,
            [this](cv::Vec3f volumePoint,
                   cv::Vec3f,
                   Qt::MouseButton button,
                   Qt::KeyboardModifiers modifiers,
                   QPointF) {
                if (button == Qt::LeftButton && modifiers == Qt::ShiftModifier) {
                    emit generatedPredSnapPointRequested(_generatedViews.currentCutName,
                                                         volumePoint);
                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                    setCurrentCutFollowsStripMouse(true);
                    emit generatedControlPointRequested(_generatedViews.currentCutName,
                                                        volumePoint,
                                                        _currentLinePosition);
                }
            });
    topSplitter->addWidget(currentViewer);
    _currentCutViewer = currentViewer;
    _panes.push_back(Pane{views.currentCutName, currentViewer, {}});
    connectGeneratedOverlayRefresh(currentViewer);

    auto* sideBase = _viewerManager->createViewerInWidget(
        views.sideCutName,
        topSplitter,
        ViewerManager::ViewerRole::Annotation);
    auto* sideViewer = sideBase
        ? qobject_cast<CChunkedVolumeViewer*>(sideBase->asQObject())
        : nullptr;
    if (!sideViewer) {
        return false;
    }
    sideViewer->setObjectName(tr("Line Side Cut"));
    sideViewer->applyCameraState(haveSideCutCamera
                                     ? sideCutCamera
                                     : generatedPaneCamera(sideViewer, camera),
                                 false);
    if (!haveSideCutCamera && finitePoint(_generatedViews.focusPoint)) {
        sideViewer->centerOnVolumePoint(_generatedViews.focusPoint, false);
    }
    sideViewer->setShiftScrollOverride(
        [this](int steps, QPointF, Qt::KeyboardModifiers) {
            return shiftCurrentLinePositionByScrollSteps(steps);
        });
    bindPaneInteractions(views.sideCutName, sideViewer, false);
    connect(sideViewer,
            &CChunkedVolumeViewer::sendMousePressVolume,
            this,
            [this](cv::Vec3f volumePoint,
                   cv::Vec3f,
                   Qt::MouseButton button,
                   Qt::KeyboardModifiers modifiers,
                   QPointF) {
                if (button == Qt::LeftButton && modifiers == Qt::ShiftModifier) {
                    emit generatedPredSnapPointRequested(_generatedViews.sideCutName,
                                                         volumePoint);
                } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                    setCurrentCutFollowsStripMouse(true);
                    emit generatedControlPointRequested(_generatedViews.sideCutName,
                                                        volumePoint,
                                                        _currentLinePosition);
                }
            });
    topSplitter->addWidget(sideViewer);
    _sideCutViewer = sideViewer;
    _panes.push_back(Pane{views.sideCutName, sideViewer, {}});
    connectGeneratedOverlayRefresh(sideViewer);
    topSplitter->setStretchFactor(0, 1);
    topSplitter->setStretchFactor(1, 1);

    auto* stripSplitter = new QSplitter(Qt::Vertical, outerSplitter);
    stripSplitter->setObjectName(QStringLiteral("LineAnnotationStripSplitter"));
    stripSplitter->setChildrenCollapsible(false);
    stripSplitter->installEventFilter(this);
    _generatedStripSplitter = stripSplitter;
    outerSplitter->addWidget(stripSplitter);

    const std::pair<std::string, QString> stripSpecs[] = {
        {views.lineSurfaceName, views.lineSurfaceTitle},
        {views.lineSideSliceName, views.lineSideSliceTitle},
    };
    int stripIndex = 0;
    for (const auto& [surfaceName, title] : stripSpecs) {
        auto* base = _viewerManager->createViewerInWidget(
            surfaceName,
            stripSplitter,
            ViewerManager::ViewerRole::Annotation);
        auto* viewer = base ? qobject_cast<CChunkedVolumeViewer*>(base->asQObject()) : nullptr;
        if (!viewer) {
            return false;
        }
        viewer->setObjectName(title);
        const bool haveStripCamera = static_cast<size_t>(stripIndex) < stripCameras.size();
        auto stripCamera = haveStripCamera
            ? stripCameras[static_cast<size_t>(stripIndex)]
            : generatedPaneCamera(viewer, camera);
        if (!haveStripCamera) {
            if (const auto center =
                    generatedStripSurfaceCenter(viewer, _currentLinePosition)) {
                stripCamera.surfacePtrX = (*center)[0];
                stripCamera.surfacePtrY = (*center)[1];
            }
        }
        viewer->applyCameraState(stripCamera, false);
        bindPaneInteractions(surfaceName, viewer, false);
        connect(viewer,
                &CChunkedVolumeViewer::sendMouseMoveVolume,
                this,
                [this, viewer](cv::Vec3f, Qt::MouseButtons, Qt::KeyboardModifiers, QPointF scenePoint) {
                    if (!_currentCutFollowsStripMouse) {
                        return;
                    }
                    const double position = linePositionFromStripScene(viewer, scenePoint);
                    if (std::isfinite(position)) {
                        requestCurrentLinePosition(position);
                    }
                });
        connect(viewer,
                &CChunkedVolumeViewer::sendMousePressVolume,
                this,
                [this, viewer, surfaceName](cv::Vec3f volumePoint,
                                            cv::Vec3f,
                                            Qt::MouseButton button,
                                            Qt::KeyboardModifiers modifiers,
                                            QPointF scenePoint) {
                    if (button != Qt::LeftButton ||
                        (modifiers != Qt::NoModifier && modifiers != Qt::ShiftModifier)) {
                        return;
                    }
                    const double position = linePositionFromStripScene(viewer, scenePoint);
                    if (std::isfinite(position)) {
                        setCurrentCutFollowsStripMouse(true);
                        setCurrentLinePosition(position);
                        if (modifiers == Qt::ShiftModifier) {
                            emit generatedPredSnapPointRequested(surfaceName, volumePoint);
                        } else {
                            emit generatedControlPointRequested(surfaceName, volumePoint, position);
                        }
                    }
                });
        stripSplitter->addWidget(viewer);
        _stripViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
        connectGeneratedOverlayRefresh(viewer);
        ++stripIndex;
    }

    stripSplitter->setStretchFactor(0, 1);
    stripSplitter->setStretchFactor(1, 1);
    outerSplitter->setStretchFactor(0, 2);
    outerSplitter->setStretchFactor(1, 1);

    // Restore user-adjusted splitter sizes across rebuilds (point placement re-runs this
    // function); fall back to the default 2:1 top/strip ratio on first build.
    if (_savedTopSplitterSizes.size() == topSplitter->count()) {
        topSplitter->setSizes(_savedTopSplitterSizes);
    }
    if (_savedStripSplitterSizes.size() == stripSplitter->count()) {
        stripSplitter->setSizes(_savedStripSplitterSizes);
    }
    if (_savedOuterSplitterSizes.size() == outerSplitter->count()) {
        outerSplitter->setSizes(_savedOuterSplitterSizes);
    } else {
        outerSplitter->setSizes({2, 1});
    }

    rebuildGeneratedOverlays();
    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(true);
    }
    if (_fullOptimizationButton) {
        _fullOptimizationButton->setEnabled(true);
    }
    captureInitialGeneratedViewState();
    if (_resetViewsButton) {
        _resetViewsButton->setEnabled(true);
    }
    return true;
}

LineAnnotationDialog::GeneratedControlPointContextResult
LineAnnotationDialog::showGeneratedControlPointContextMenu(const std::string& surfaceName,
                                                           CChunkedVolumeViewer* viewer,
                                                           const QPointF& scenePoint,
                                                           const QPoint& globalPos)
{
    if (!viewer || !_hasGeneratedViews || _generatedViews.controlPoints.empty() ||
        _generatedViews.linePoints.empty()) {
        return GeneratedControlPointContextResult::None;
    }

    double linePosition = std::numeric_limits<double>::quiet_NaN();
    if (viewer == _currentCutViewer || viewer == _sideCutViewer) {
        linePosition = _currentLinePosition;
    } else {
        for (size_t i = 0; i < _stripViewers.size(); ++i) {
            if (viewer == _stripViewers[i]) {
                linePosition = linePositionFromStripScene(viewer, scenePoint);
                break;
            }
        }
    }

    if (!vc3d::line_annotation::validGeneratedLinePosition(linePosition,
                                                           _generatedViews.linePoints.size())) {
        return GeneratedControlPointContextResult::None;
    }

    const bool stripViewer =
        std::any_of(_stripViewers.begin(),
                    _stripViewers.end(),
                    [viewer](const QPointer<CChunkedVolumeViewer>& candidate) {
                        return candidate == viewer;
                    });

    vc3d::line_annotation::GeneratedControlPointContextMenuOptions options;
    options.parent = this;
    options.surfaceName = surfaceName;
    options.viewer = viewer;
    options.scenePoint = scenePoint;
    options.globalPos = globalPos;
    options.controlPoints = _generatedViews.controlPoints;
    options.linePointCount = _generatedViews.linePoints.size();
    options.linePosition = linePosition;
    options.stripViewer = stripViewer;
    options.deleteControlPoint = [this, surfaceName](double selectedLinePosition,
                                                     cv::Vec3f selectedPoint) {
        emit generatedControlPointDeleteRequested(surfaceName,
                                                  selectedLinePosition,
                                                  selectedPoint);
    };
    return vc3d::line_annotation::showGeneratedControlPointContextMenu(options);
}

void LineAnnotationDialog::applyGeneratedOverlay(const std::string& surfaceName,
                                                 CChunkedVolumeViewer* viewer,
                                                 const GeneratedOverlay& overlay)
{
    vc3d::line_annotation::applyGeneratedOverlay(viewer, surfaceName, overlay);
}

void LineAnnotationDialog::applyOverlayForViewer(const std::string& surfaceName,
                                                 CChunkedVolumeViewer* viewer,
                                                 const GeneratedOverlay& overlay)
{
    vc3d::line_annotation::applyGeneratedOverlay(viewer, surfaceName, overlay);
}

void LineAnnotationDialog::clearControlPointContextPreview(const std::string& surfaceName,
                                                           CChunkedVolumeViewer* viewer)
{
    vc3d::line_annotation::clearGeneratedControlPointContextPreview(viewer, surfaceName);
}

double LineAnnotationDialog::linePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                                        const QPointF& scenePoint) const
{
    if (!viewer || !_hasGeneratedViews) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return vc3d::line_annotation::generatedLinePositionFromStripScene(viewer, scenePoint);
}

void LineAnnotationDialog::requestCurrentLinePosition(double position)
{
    // Hot path: called once per mouse-move event while following a strip viewer. Defer the
    // actual (potentially O(N)) plane + overlay rebuild to a single render-tick-cadence flush so
    // a fast cursor doesn't back up the event loop with one full update per move.
    _pendingLinePosition = position;
    _lineUpdatePending = true;
    if (!_lineUpdateTimer) {
        _lineUpdateTimer = new QTimer(this);
        _lineUpdateTimer->setSingleShot(true);
        _lineUpdateTimer->setInterval(16);  // ~one global render tick
        connect(_lineUpdateTimer, &QTimer::timeout, this, [this]() {
            if (_lineUpdatePending) {
                setCurrentLinePosition(_pendingLinePosition);
            }
        });
    }
    if (!_lineUpdateTimer->isActive()) {
        _lineUpdateTimer->start();
    }
}

void LineAnnotationDialog::setCurrentLinePosition(double position)
{
    // An immediate apply supersedes any coalesced mouse-follow update still pending in the timer,
    // so a discrete jump/click/scroll isn't clobbered by a stale flush a few ms later.
    _lineUpdatePending = false;
    if (_lineUpdateTimer) {
        _lineUpdateTimer->stop();
    }
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }
    position = std::clamp(position, 0.0, static_cast<double>(_generatedViews.linePoints.size() - 1));
    const bool currentChanged = std::abs(position - _currentLinePosition) >= 1.0e-3;
    if (!currentChanged) {
        return;
    }
    if (_generatedViews.currentCutSurface && !_currentCutStraightOffsetActive) {
        _currentLinePosition = position;
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    } else {
        _currentLinePosition = position;
    }
    if (_currentCutViewer) {
        // Non-force: let the global render tick coalesce a burst of cursor moves into one
        // render per tick instead of force-submitting synchronously on every mouse event.
        _currentCutViewer->renderVisible(false, "line annotation current cut");
    }
    if (_generatedViews.sideCutSurface) {
        (void)updateSidePlaneSurface(_generatedViews.sideCutSurface.get(), _currentLinePosition);
    }
    if (_sideCutViewer) {
        // Moving the cursor along the strips moves along the line, so keep the current position
        // centered/visible in the side view. centerOnVolumePoint(false) pans and schedules the
        // (coalesced) render, so no separate renderVisible call is needed here.
        const cv::Vec3f sidePoint = interpolatedLinePoint(_currentLinePosition);
        if (finitePoint(sidePoint)) {
            _sideCutViewer->centerOnVolumePoint(sidePoint, false);
        } else {
            _sideCutViewer->renderVisible(false, "line annotation side cut");
        }
    }
    rebuildGeneratedDynamicOverlays();
}

void LineAnnotationDialog::cancelControlPointPreviewAnimation()
{
    if (!_controlPointPreviewAnimation) {
        return;
    }
    auto* animation = _controlPointPreviewAnimation.data();
    _controlPointPreviewAnimation = nullptr;
    animation->stop();
    animation->deleteLater();
}

void LineAnnotationDialog::jumpToPreviousControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    const auto previous = vc3d::line_annotation::previousGeneratedControlPointLinePosition(
        _currentLinePosition,
        positions);
    if (previous) {
        setCurrentLinePosition(*previous);
    }
}

void LineAnnotationDialog::jumpToNextControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    const auto next = vc3d::line_annotation::nextGeneratedControlPointLinePosition(
        _currentLinePosition,
        positions);
    if (next) {
        setCurrentLinePosition(*next);
    }
}

void LineAnnotationDialog::previewClosestControlPoint()
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return;
    }
    cancelControlPointPreviewAnimation();
    const double originalPosition = _currentLinePosition;
    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(
        _generatedViews.controlPoints);
    const auto closest = vc3d::line_annotation::closestGeneratedControlPointLinePosition(
        originalPosition,
        positions);
    if (!closest) {
        return;
    }
    setCurrentLinePosition(*closest);

    auto* animation = new QVariantAnimation(this);
    _controlPointPreviewAnimation = animation;
    constexpr int kClosestControlPointHoldMs = 500;
    constexpr int kClosestControlPointReturnMs = 2000;
    constexpr int kClosestControlPointTotalMs =
        kClosestControlPointHoldMs + kClosestControlPointReturnMs;
    animation->setDuration(kClosestControlPointTotalMs);
    animation->setStartValue(*closest);
    animation->setKeyValueAt(static_cast<double>(kClosestControlPointHoldMs) /
                                 static_cast<double>(kClosestControlPointTotalMs),
                             *closest);
    animation->setEndValue(originalPosition);
    connect(animation, &QVariantAnimation::valueChanged, this, [this, animation](const QVariant& value) {
        if (_controlPointPreviewAnimation == animation) {
            setCurrentLinePosition(value.toDouble());
        }
    });
    connect(animation, &QVariantAnimation::finished, this, [this, animation]() {
        if (_controlPointPreviewAnimation == animation) {
            _controlPointPreviewAnimation = nullptr;
            animation->deleteLater();
        }
    });
    QTimer::singleShot(30, animation, [animation]() {
        animation->start();
    });
}

bool LineAnnotationDialog::shiftCurrentLinePositionByScrollSteps(int steps)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return true;
    }
    const int sliceStepSize = _viewerManager ? _viewerManager->sliceStepSize() : 1;
    const double position = vc3d::line_annotation::shiftedLinePosition(
        _currentLinePosition,
        steps,
        sliceStepSize,
        static_cast<int>(_generatedViews.linePoints.size()));
    _currentCutStraightOffsetActive = false;
    setCurrentLinePosition(position);
    return true;
}

bool LineAnnotationDialog::shiftCurrentCutPlaneStraightByScrollSteps(int steps)
{
    if (!_hasGeneratedViews || !_generatedViews.currentCutSurface) {
        return true;
    }
    auto* plane = _generatedViews.currentCutSurface.get();
    const int sliceStepSize = _viewerManager ? _viewerManager->sliceStepSize() : 1;
    const cv::Vec3f origin = plane->origin();
    const cv::Vec3f normal = plane->normal({0.0f, 0.0f, 0.0f});
    const cv::Vec3f shiftedOrigin =
        vc3d::line_annotation::shiftedPlaneOriginAlongNormal(origin,
                                                             normal,
                                                             steps,
                                                             sliceStepSize);
    if (!finitePoint(shiftedOrigin)) {
        return true;
    }
    plane->setOrigin(shiftedOrigin);
    _currentCutStraightOffsetActive = true;
    if (_currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation current cut straight shift");
    }
    rebuildGeneratedDynamicOverlays();
    return true;
}

void LineAnnotationDialog::setCurrentCutFollowsStripMouse(bool follows)
{
    _currentCutFollowsStripMouse = follows;
}

void LineAnnotationDialog::installGeneratedViewShortcuts()
{
    const auto bindNavigationShortcut = [this](Qt::Key key, void (LineAnnotationDialog::*slot)()) {
        auto* shortcut = new QShortcut(QKeySequence(key), this);
        shortcut->setContext(Qt::WindowShortcut);
        connect(shortcut, &QShortcut::activated, this, slot);
    };

    const auto bindRotationShortcut =
        [this](Qt::Key key, vc3d::line_annotation::GeneratedCutRotationAxis axis, float radians) {
            auto* shortcut = new QShortcut(QKeySequence(key), this);
            shortcut->setContext(Qt::WindowShortcut);
            connect(shortcut, &QShortcut::activated, this, [this, axis, radians]() {
                (void)rotateCurrentCut(axis, radians);
            });
        };

    using vc3d::line_annotation::GeneratedCutRotationAxis;
    bindRotationShortcut(Qt::Key_W, GeneratedCutRotationAxis::Horizontal, kCurrentCutRotationStepRadians);
    bindRotationShortcut(Qt::Key_S, GeneratedCutRotationAxis::Horizontal, -kCurrentCutRotationStepRadians);
    bindRotationShortcut(Qt::Key_A, GeneratedCutRotationAxis::Vertical, -kCurrentCutRotationStepRadians);
    bindRotationShortcut(Qt::Key_D, GeneratedCutRotationAxis::Vertical, kCurrentCutRotationStepRadians);
    bindNavigationShortcut(Qt::Key_E, &LineAnnotationDialog::jumpToPreviousControlPoint);
    bindNavigationShortcut(Qt::Key_R, &LineAnnotationDialog::previewClosestControlPoint);
    bindNavigationShortcut(Qt::Key_T, &LineAnnotationDialog::jumpToNextControlPoint);
}

cv::Vec3f LineAnnotationDialog::currentCutViewerCenterVolumePoint() const
{
    auto* viewer = _currentCutViewer.data();
    auto* view = viewer ? viewer->graphicsView() : nullptr;
    auto* viewport = view ? view->viewport() : nullptr;
    if (viewer && view && viewport && viewport->width() > 0 && viewport->height() > 0) {
        const QPointF sceneCenter = view->mapToScene(viewport->rect().center());
        const cv::Vec3f center = viewer->sceneToVolume(sceneCenter);
        if (finitePoint(center)) {
            return center;
        }
    }
    return interpolatedLinePoint(_currentLinePosition);
}

bool LineAnnotationDialog::rotateCurrentCut(vc3d::line_annotation::GeneratedCutRotationAxis axis,
                                            float radians)
{
    if (!_hasGeneratedViews || !_generatedViews.currentCutSurface || !_currentCutViewer) {
        return false;
    }
    const cv::Vec3f centerVolumePoint = currentCutViewerCenterVolumePoint();
    _currentCutManualRotation =
        vc3d::line_annotation::accumulatedGeneratedCutRotation(_currentCutManualRotation,
                                                               axis,
                                                               radians);
    _currentCutManualRotationActive = true;
    if (!updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition)) {
        return false;
    }
    if (finitePoint(centerVolumePoint)) {
        _currentCutViewer->centerOnVolumePoint(centerVolumePoint, false);
    }
    _currentCutViewer->renderVisible(true, "line annotation current cut rotation");
    rebuildGeneratedDynamicOverlays();
    return true;
}

void LineAnnotationDialog::captureInitialGeneratedViewState()
{
    _initialCurrentLinePosition = _currentLinePosition;

    _haveInitialCurrentCutCamera = false;
    if (_currentCutViewer) {
        _initialCurrentCutCamera = _currentCutViewer->cameraState();
        _haveInitialCurrentCutCamera = true;
    }
    _haveInitialSideCutCamera = false;
    if (_sideCutViewer) {
        _initialSideCutCamera = _sideCutViewer->cameraState();
        _haveInitialSideCutCamera = true;
    }
    _initialStripCameras.clear();
    _initialStripCameras.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            _initialStripCameras.push_back(viewer->cameraState());
        }
    }
}

void LineAnnotationDialog::restoreInitialGeneratedViewerCameras()
{
    if (_currentCutViewer && _haveInitialCurrentCutCamera) {
        _currentCutViewer->applyCameraState(_initialCurrentCutCamera, false);
    }
    if (_sideCutViewer && _haveInitialSideCutCamera) {
        _sideCutViewer->applyCameraState(_initialSideCutCamera, false);
    }
    for (size_t i = 0; i < _stripViewers.size() && i < _initialStripCameras.size(); ++i) {
        if (_stripViewers[i]) {
            _stripViewers[i]->applyCameraState(_initialStripCameras[i], false);
        }
    }
}

void LineAnnotationDialog::resetGeneratedViews()
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }

    const auto state = vc3d::line_annotation::resetGeneratedLineViewNavigationState(
        _initialCurrentLinePosition,
        _initialCurrentLinePosition,
        vc3d::line_annotation::kDefaultBottomCrossSliceLineStep);
    _currentLinePosition = std::clamp(state.currentLinePosition,
                                      0.0,
                                      static_cast<double>(_generatedViews.linePoints.size() - 1));
    _currentCutManualRotation = state.currentCutManualRotation;
    _currentCutManualRotationActive = state.currentCutManualRotationActive;
    _currentCutStraightOffsetActive = false;
    _currentCutFollowsStripMouse = true;

    if (_generatedViews.currentCutSurface) {
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    }
    if (_generatedViews.sideCutSurface) {
        (void)updateSidePlaneSurface(_generatedViews.sideCutSurface.get(), _currentLinePosition);
    }
    restoreInitialGeneratedViewerCameras();
    if (_currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation reset current cut");
    }
    if (_sideCutViewer) {
        _sideCutViewer->renderVisible(true, "line annotation reset side cut");
    }
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            viewer->renderVisible(true, "line annotation reset strip view");
        }
    }
    rebuildGeneratedOverlays();
}

double LineAnnotationDialog::snappedControlPointPosition(double position) const
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return position;
    }
    std::vector<double> controlLinePositions;
    controlLinePositions.reserve(_generatedViews.controlPoints.size());
    for (const auto& control : _generatedViews.controlPoints) {
        controlLinePositions.push_back(control.linePosition);
    }
    return vc3d::line_annotation::snappedControlPointLinePosition(position, controlLinePositions);
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::stripOverlay() const
{
    return vc3d::line_annotation::makeGeneratedStripOverlay(_generatedViews,
                                                            _currentLinePosition,
                                                            {});
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::staticStripOverlay() const
{
    return vc3d::line_annotation::makeGeneratedStaticStripOverlay(_generatedViews);
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::dynamicStripOverlay() const
{
    return vc3d::line_annotation::makeGeneratedDynamicStripOverlay(_generatedViews,
                                                                   _currentLinePosition,
                                                                   {});
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::zSliceOverlay(double linePosition,
                                                                           bool emphasized,
                                                                           CChunkedVolumeViewer* viewer,
                                                                           PlaneSurface* plane) const
{
    return vc3d::line_annotation::makeGeneratedCrossSliceOverlayForPlane(_generatedViews,
                                                                         linePosition,
                                                                         emphasized,
                                                                         viewer,
                                                                         plane,
                                                                         &_generatedControlIndex);
}

void LineAnnotationDialog::rebuildGeneratedStaticStripOverlays()
{
    if (!_hasGeneratedViews) {
        return;
    }

    const GeneratedOverlay strip = staticStripOverlay();
    for (size_t i = 0; i < _stripViewers.size(); ++i) {
        auto* viewer = _stripViewers[i].data();
        if (!viewer) {
            continue;
        }
        const std::string key = i == 0 ? _generatedViews.lineSurfaceName
                                       : _generatedViews.lineSideSliceName;
        applyOverlayForViewer(staticStripOverlayKey(key), viewer, strip);
    }
}

void LineAnnotationDialog::rebuildGeneratedDynamicOverlays()
{
    if (!_hasGeneratedViews) {
        return;
    }

    const GeneratedOverlay strip = dynamicStripOverlay();
    for (size_t i = 0; i < _stripViewers.size(); ++i) {
        auto* viewer = _stripViewers[i].data();
        if (!viewer) {
            continue;
        }
        const std::string key = i == 0 ? _generatedViews.lineSurfaceName
                                       : _generatedViews.lineSideSliceName;
        applyOverlayForViewer(dynamicStripOverlayKey(key), viewer, strip);
    }

    if (_currentCutViewer) {
        applyOverlayForViewer("line-z-slice-current",
                              _currentCutViewer,
                              zSliceOverlay(_currentLinePosition,
                                            true,
                                            _currentCutViewer,
                                            _generatedViews.currentCutSurface.get()));
    }

    if (_sideCutViewer && _generatedViews.sideCutSurface) {
        // Draw the full line on the side view by projecting each line point onto the plane and
        // connecting consecutive points (linear interpolation), in addition to the current-position
        // marker and nearby control points from the cross-slice overlay.
        GeneratedOverlay sideOverlay = zSliceOverlay(_currentLinePosition,
                                                     true,
                                                     _sideCutViewer,
                                                     _generatedViews.sideCutSurface.get());
        sideOverlay.linePoints = _generatedViews.linePoints;
        // Highlight the live cursor position on the line. The cross-slice overlay's emphasized
        // marker otherwise sits at the static focus/seed point; override it to the current
        // position so the highlight tracks the cursor as it moves along the line.
        const cv::Vec3f currentPoint = interpolatedLinePoint(_currentLinePosition);
        if (finitePoint(currentPoint)) {
            sideOverlay.pointMarker = currentPoint;
            sideOverlay.emphasizedPointMarker = true;
        }
        applyOverlayForViewer("line-z-slice-side", _sideCutViewer, sideOverlay);
    }
}

void LineAnnotationDialog::rebuildGeneratedOverlays()
{
    rebuildGeneratedStaticStripOverlays();
    rebuildGeneratedDynamicOverlays();
}

cv::Vec3f LineAnnotationDialog::interpolatedLinePoint(double linePosition) const
{
    return vc3d::line_annotation::interpolatedGeneratedLinePoint(_generatedViews.linePoints,
                                                                 linePosition);
}

cv::Vec3f LineAnnotationDialog::interpolatedLineTangent(double linePosition) const
{
    if (_generatedViews.linePoints.size() < 2) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    const double maxPosition = static_cast<double>(_generatedViews.linePoints.size() - 1);
    linePosition = std::clamp(linePosition, 0.0, maxPosition);
    // Smooth, continuous tangent: a central difference of the (piecewise-linear) line over a
    // window around the cursor, rather than the raw difference of the two bracketing integer
    // points. The raw adjacent-point difference is piecewise-constant in linePosition and snaps
    // the cut-plane orientation at every integer crossing (the main source of the cross-section
    // "jumpiness"); averaging over +/-kTangentHalfWindow line indices removes that stepping while
    // staying locally faithful to the line direction. interpolatedLinePoint() is continuous, so
    // the result varies continuously with the cursor.
    constexpr double kTangentHalfWindow = 4.0;
    const double lo = std::max(0.0, linePosition - kTangentHalfWindow);
    const double hi = std::min(maxPosition, linePosition + kTangentHalfWindow);
    cv::Vec3f tangent = interpolatedLinePoint(hi) - interpolatedLinePoint(lo);
    if (cv::norm(tangent) <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return normalizedOrNan(tangent);
}

cv::Vec3f LineAnnotationDialog::interpolatedLineUp(double linePosition, const cv::Vec3f& tangent) const
{
    if (_generatedViews.lineUpVectors.empty()) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }

    linePosition = std::clamp(linePosition,
                              0.0,
                              static_cast<double>(_generatedViews.lineUpVectors.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(_generatedViews.lineUpVectors.size()) - 1);
    cv::Vec3f lowerUp = _generatedViews.lineUpVectors[static_cast<size_t>(lower)];
    cv::Vec3f upperUp = _generatedViews.lineUpVectors[static_cast<size_t>(upper)];
    if (!finitePoint(lowerUp) || !finitePoint(upperUp) ||
        cv::norm(lowerUp) <= 1.0e-6f ||
        cv::norm(upperUp) <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    if (lowerUp.dot(upperUp) < 0.0f) {
        upperUp *= -1.0f;
    }

    const float t = static_cast<float>(linePosition - static_cast<double>(lower));
    cv::Vec3f up = lowerUp * (1.0f - t) + upperUp * t;
    up -= tangent * up.dot(tangent);
    if (cv::norm(up) <= 1.0e-6f) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    return normalizedOrNan(up);
}

bool LineAnnotationDialog::updatePlaneSurface(PlaneSurface* plane, double linePosition) const
{
    if (!plane) {
        return false;
    }
    const cv::Vec3f origin = interpolatedLinePoint(linePosition);
    const cv::Vec3f tangent = interpolatedLineTangent(linePosition);
    const cv::Vec3f upHint = interpolatedLineUp(linePosition, tangent);
    if (!finitePoint(origin) || !finitePoint(tangent) || !finitePoint(upHint) ||
        cv::norm(tangent) <= 1.0e-6f ||
        cv::norm(upHint) <= 1.0e-6f) {
        return false;
    }
    if (_currentCutManualRotationActive &&
        _generatedViews.currentCutSurface &&
        plane == _generatedViews.currentCutSurface.get()) {
        const auto frame =
            vc3d::line_annotation::generatedCutFrameWithManualRotation(tangent,
                                                                       upHint,
                                                                       _currentCutManualRotation);
        if (!vc3d::line_annotation::generatedCutFrameIsOrthonormal(frame)) {
            return false;
        }
        plane->setFromNormalAndUp(origin, frame.normal, frame.vertical);
    } else {
        plane->setFromNormalAndUp(origin, tangent, upHint);
    }
    return true;
}

bool LineAnnotationDialog::computeSideFit(int center, cv::Vec3f& normal, cv::Vec3f& upHint) const
{
    const auto& linePoints = _generatedViews.linePoints;
    const int count = static_cast<int>(linePoints.size());
    if (count < 3) {
        return false;
    }
    center = std::clamp(center, 0, count - 1);
    const int first = std::max(0, center - kSideFitHalfWindow);
    const int last = std::min(count - 1, center + kSideFitHalfWindow);

    vc3d::fiber_slice::ControlSpanSelection span;
    span.firstLineIndex = static_cast<size_t>(first);
    span.lastLineIndex = static_cast<size_t>(last);
    span.samples.reserve(static_cast<size_t>(last - first + 1));
    cv::Vec3d centroid{0.0, 0.0, 0.0};
    for (int i = first; i <= last; ++i) {
        const cv::Vec3f& p = linePoints[static_cast<size_t>(i)];
        if (!finitePoint(p)) {
            continue;
        }
        const cv::Vec3d pd(p[0], p[1], p[2]);
        span.samples.push_back(pd);
        centroid += pd;
    }
    if (span.samples.size() < 3) {
        return false;
    }
    span.centroid = centroid * (1.0 / static_cast<double>(span.samples.size()));
    span.valid = true;

    // fitLeastSquaresPlane reads linePoints[firstLineIndex/lastLineIndex] to derive the in-plane
    // direction hint, so it needs the full polyline as cv::Vec3d. _linePointsd is the cached
    // double-precision copy built once when the views were generated.
    const auto fit = vc3d::fiber_slice::fitLeastSquaresPlane(span, _linePointsd);
    if (!fit.valid) {
        return false;
    }
    normal = cv::Vec3f(static_cast<float>(fit.normal[0]),
                       static_cast<float>(fit.normal[1]),
                       static_cast<float>(fit.normal[2]));
    upHint = cv::Vec3f(static_cast<float>(fit.upHint[0]),
                       static_cast<float>(fit.upHint[1]),
                       static_cast<float>(fit.upHint[2]));
    if (!finitePoint(normal) || !finitePoint(upHint) ||
        cv::norm(normal) <= 1.0e-6f || cv::norm(upHint) <= 1.0e-6f) {
        return false;
    }
    return true;
}

bool LineAnnotationDialog::updateSidePlaneSurface(PlaneSurface* plane, double linePosition)
{
    if (!plane) {
        return false;
    }
    const int count = static_cast<int>(_generatedViews.linePoints.size());
    if (count < 3) {
        return false;
    }

    // Continuous side-view orientation: fit the best-fit plane at the two integer window centers
    // straddling the cursor and interpolate between them by the fractional position. Each fit
    // depends only on the static line geometry, so we cache the two bracketing fits and recompute
    // a center only when the straddle shifts (typically once per integer crossing). This keeps
    // the orientation smooth -- no snapping at discrete window centers -- while staying cheap.
    const double pos = std::clamp(linePosition, 0.0, static_cast<double>(count - 1));
    const int lowerCenter = static_cast<int>(std::floor(pos));
    const int upperCenter = std::min(lowerCenter + 1, count - 1);
    const int wantCenters[2] = {lowerCenter, upperCenter};

    SideFit next[2];
    for (int slot = 0; slot < 2; ++slot) {
        next[slot].center = wantCenters[slot];
        bool reused = false;
        for (const SideFit& cached : _sideFitBracket) {
            if (cached.valid && cached.center == wantCenters[slot]) {
                next[slot] = cached;
                reused = true;
                break;
            }
        }
        if (!reused) {
            next[slot].valid =
                computeSideFit(wantCenters[slot], next[slot].normal, next[slot].upHint);
        }
    }
    _sideFitBracket[0] = next[0];
    _sideFitBracket[1] = next[1];

    cv::Vec3f normal;
    cv::Vec3f upHint;
    if (next[0].valid && next[1].valid) {
        cv::Vec3f normal1 = next[1].normal;
        cv::Vec3f upHint1 = next[1].upHint;
        // PCA normals/up hints are sign-arbitrary; align the upper fit to the lower one before
        // blending so interpolation doesn't cancel them out near a sign flip.
        if (next[0].normal.dot(normal1) < 0.0f) {
            normal1 = -normal1;
        }
        if (next[0].upHint.dot(upHint1) < 0.0f) {
            upHint1 = -upHint1;
        }
        const float t = static_cast<float>(pos - static_cast<double>(lowerCenter));
        normal = next[0].normal * (1.0f - t) + normal1 * t;
        upHint = next[0].upHint * (1.0f - t) + upHint1 * t;
    } else if (next[0].valid || next[1].valid) {
        const SideFit& fit = next[0].valid ? next[0] : next[1];
        normal = fit.normal;
        upHint = fit.upHint;
    } else {
        return false;
    }

    normal = normalizedOrNan(normal);
    if (!finitePoint(normal)) {
        return false;
    }
    upHint -= normal * upHint.dot(normal);
    upHint = normalizedOrNan(upHint);
    if (!finitePoint(upHint)) {
        return false;
    }

    // Keep the plane centered on the cursor position rather than a window centroid; the origin
    // slides smoothly along the line while the interpolated orientation tracks it continuously.
    const cv::Vec3f origin = interpolatedLinePoint(pos);
    if (!finitePoint(origin)) {
        return false;
    }
    plane->setFromNormalAndUp(origin, normal, upHint);
    return true;
}

QPointF LineAnnotationDialog::stripLinePositionToScene(CChunkedVolumeViewer* viewer,
                                                       QuadSurface* surface,
                                                       double linePosition) const
{
    if (!viewer || !surface) {
        return {};
    }
    const auto* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return {};
    }
    const cv::Vec2f scale = surface->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f) {
        return {};
    }
    const float surfaceX = (static_cast<float>(linePosition) -
                            static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float centerRow = static_cast<float>(points->rows / 2);
    const float surfaceY = (centerRow - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
}

void LineAnnotationDialog::keyPressEvent(QKeyEvent* event)
{
    if (handleKeyPress(event)) {
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void LineAnnotationDialog::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    updateOptimizationOverlayGeometry();
}

bool LineAnnotationDialog::handleKeyPress(QKeyEvent* event)
{
    if (!event) {
        return false;
    }
    if (event->key() == Qt::Key_Space && event->modifiers() == Qt::NoModifier) {
        if (shiftScrollMode() == ShiftScrollMode::StraightNormal &&
            _currentCutStraightOffsetActive) {
            _currentCutStraightOffsetActive = false;
            _currentCutManualRotation = cv::Matx33f::eye();
            _currentCutManualRotationActive = false;
            (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
            if (_currentCutViewer) {
                _currentCutViewer->renderVisible(true, "line annotation current cut snap to line");
            }
            rebuildGeneratedDynamicOverlays();
        }
        if (_currentCutFollowsStripMouse) {
            setCurrentLinePosition(snappedControlPointPosition(_currentLinePosition));
            setCurrentCutFollowsStripMouse(false);
        } else {
            setCurrentCutFollowsStripMouse(true);
        }
        event->accept();
        return true;
    }
    if (_viewerManager &&
        event->modifiers() == vc3d::keybinds::keypress::SliceStepDecrease.modifiers) {
        if (event->key() == vc3d::keybinds::keypress::SliceStepDecrease.key) {
            const int newStep = std::max(1, _viewerManager->sliceStepSize() - 1);
            _viewerManager->setSliceStepSize(newStep);
            event->accept();
            return true;
        }
        if (event->key() == vc3d::keybinds::keypress::SliceStepIncrease.key) {
            const int newStep = std::min(100, _viewerManager->sliceStepSize() + 1);
            _viewerManager->setSliceStepSize(newStep);
            event->accept();
            return true;
        }
    }
    if (event->key() == Qt::Key_Escape) {
        close();
        event->accept();
        return true;
    }
    return false;
}

bool LineAnnotationDialog::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (handleKeyPress(keyEvent)) {
            return true;
        }
    }
    return QMainWindow::eventFilter(watched, event);
}

void LineAnnotationDialog::updateOptimizationOverlayGeometry()
{
    if (!_optimizationOverlay || !centralWidget()) {
        return;
    }
    _optimizationOverlay->setGeometry(centralWidget()->rect());
    if (_optimizationOverlay->isVisible()) {
        _optimizationOverlay->raise();
    }
}
