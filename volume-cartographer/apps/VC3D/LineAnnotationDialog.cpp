#include "LineAnnotationDialog.hpp"

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
#include <QTimer>
#include <QVariantAnimation>
#include <QVBoxLayout>
#include <QWidget>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {

constexpr int kBottomCrossSliceCount = 7;
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
    _bottomSliceLineStep = vc3d::line_annotation::kDefaultBottomCrossSliceLineStep;

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
    _bottomSliceStepLabel = new QLabel(this);
    _bottomSliceStepLabel->setToolTip(
        tr("Small cross-slice spacing along the line. Use Ctrl+Shift+Scroll anywhere in this window to adjust."));
    _bottomSliceStepLabel->installEventFilter(this);
    updateBottomSliceStepLabel();
    buttonLayout->addWidget(_bottomSliceStepLabel);
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
    _bottomSliceViewers.clear();
    _currentCutViewer = nullptr;
    _hasGeneratedViews = false;
    _currentCutManualRotation = cv::Matx33f::eye();
    _currentCutManualRotationActive = false;
    _currentCutStraightOffsetActive = false;
    _generatedControlIndex = {};
    _haveInitialCurrentCutCamera = false;
    _initialStripCameras.clear();
    _initialBottomSliceCameras.clear();

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
        !views.currentCutSurface || views.bottomCutSurfaces.empty()) {
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
    const double previousBottomCenterPosition = _bottomCenterPosition;

    bool haveCurrentCutCamera = false;
    CChunkedVolumeViewer::CameraState currentCutCamera;
    if (_currentCutViewer) {
        currentCutCamera = _currentCutViewer->cameraState();
        haveCurrentCutCamera = true;
    }

    std::vector<CChunkedVolumeViewer::CameraState> stripCameras;
    stripCameras.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            stripCameras.push_back(viewer->cameraState());
        }
    }

    std::vector<CChunkedVolumeViewer::CameraState> bottomSliceCameras;
    bottomSliceCameras.reserve(_bottomSliceViewers.size());
    for (const auto& viewer : _bottomSliceViewers) {
        if (viewer) {
            bottomSliceCameras.push_back(viewer->cameraState());
        }
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
    _bottomSliceViewers.clear();
    _currentCutViewer = nullptr;

    _generatedViews = views;
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
    _bottomCenterPosition = replacingGeneratedViews
        ? std::clamp(previousBottomCenterPosition, 0.0, maxLinePosition)
        : _currentLinePosition;
    if (!updatePlaneSurface(views.currentCutSurface.get(), _currentLinePosition)) {
        return false;
    }

    auto* topWidget = new QWidget(this);
    auto* topLayout = new QHBoxLayout(topWidget);
    topLayout->setContentsMargins(0, 0, 0, 0);
    topLayout->setSpacing(0);
    topWidget->installEventFilter(this);
    _generatedTopWidget = topWidget;
    _generatedContainers.push_back(topWidget);
    _layout->addWidget(topWidget, 2);

    auto* currentBase = _viewerManager->createViewerInWidget(
        views.currentCutName,
        topWidget,
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
        [this](int steps, QPointF, Qt::KeyboardModifiers modifiers) {
            if (modifiers.testFlag(Qt::ControlModifier)) {
                return scaleBottomSliceLineStepByScrollSteps(steps);
            }
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
    topLayout->addWidget(currentViewer, 0);
    _currentCutViewer = currentViewer;
    _panes.push_back(Pane{views.currentCutName, currentViewer, {}});
    connectGeneratedOverlayRefresh(currentViewer);

    auto* stripStack = new QWidget(topWidget);
    stripStack->installEventFilter(this);
    auto* stripLayout = new QVBoxLayout(stripStack);
    stripLayout->setContentsMargins(0, 0, 0, 0);
    stripLayout->setSpacing(0);
    topLayout->addWidget(stripStack, 1);

    const std::pair<std::string, QString> stripSpecs[] = {
        {views.lineSurfaceName, views.lineSurfaceTitle},
        {views.lineSideSliceName, views.lineSideSliceTitle},
    };
    int stripIndex = 0;
    for (const auto& [surfaceName, title] : stripSpecs) {
        auto* base = _viewerManager->createViewerInWidget(
            surfaceName,
            stripStack,
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
                        setCurrentLinePosition(position);
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
        stripLayout->addWidget(viewer, 1);
        _stripViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
        connectGeneratedOverlayRefresh(viewer);
        ++stripIndex;
    }

    auto* bottomWidget = new QWidget(this);
    bottomWidget->installEventFilter(this);
    auto* bottomLayout = new QHBoxLayout(bottomWidget);
    bottomLayout->setContentsMargins(0, 0, 0, 0);
    bottomLayout->setSpacing(0);
    _generatedContainers.push_back(bottomWidget);
    _layout->addWidget(bottomWidget, 1);

    const int bottomCount = std::min<int>(kBottomCrossSliceCount,
                                          static_cast<int>(views.bottomCutSurfaces.size()));
    for (int slot = 0; slot < bottomCount; ++slot) {
        const double position = bottomSliceLinePosition(slot, bottomCount);
        const auto& [surfaceName, plane] = views.bottomCutSurfaces[static_cast<size_t>(slot)];
        if (!updatePlaneSurface(plane.get(), position)) {
            return false;
        }
        auto* base = _viewerManager->createViewerInWidget(
            surfaceName,
            bottomWidget,
            ViewerManager::ViewerRole::Annotation);
        auto* viewer = base ? qobject_cast<CChunkedVolumeViewer*>(base->asQObject()) : nullptr;
        if (!viewer) {
            return false;
        }
        viewer->setObjectName(tr("Line Z Slice %1").arg(slot));
        viewer->applyCameraState(static_cast<size_t>(slot) < bottomSliceCameras.size()
                                     ? bottomSliceCameras[static_cast<size_t>(slot)]
                                     : generatedPaneCamera(viewer, camera),
                                 false);
        viewer->setShiftScrollOverride(
            [this](int steps, QPointF, Qt::KeyboardModifiers modifiers) {
                if (modifiers.testFlag(Qt::ControlModifier)) {
                    return scaleBottomSliceLineStepByScrollSteps(steps);
                }
                return shiftBottomSlicesByScrollSteps(steps);
            });
        bindPaneInteractions(surfaceName, viewer, false);
        connect(viewer,
                &CChunkedVolumeViewer::sendMousePressVolume,
                this,
                [this, surfaceName, slot](cv::Vec3f volumePoint,
                                          cv::Vec3f,
                                          Qt::MouseButton button,
                                          Qt::KeyboardModifiers modifiers,
                                          QPointF) {
                    if (button == Qt::LeftButton && modifiers == Qt::ShiftModifier) {
                        emit generatedPredSnapPointRequested(surfaceName, volumePoint);
                    } else if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
                        const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
                        const double linePosition = bottomSliceLinePosition(slot, bottomCount);
                        setCurrentCutFollowsStripMouse(true);
                        setCurrentLinePosition(linePosition);
                        emit generatedControlPointRequested(surfaceName, volumePoint, linePosition);
                    }
                });
        bottomLayout->addWidget(viewer, 1);
        _bottomSliceViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
        connectGeneratedOverlayRefresh(viewer);
    }

    if (_generatedTopWidget && _currentCutViewer) {
        _currentCutViewer->setFixedWidth(std::max(1, _generatedTopWidget->height()));
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
    if (viewer == _currentCutViewer) {
        linePosition = _currentLinePosition;
    } else {
        for (size_t i = 0; i < _stripViewers.size(); ++i) {
            if (viewer == _stripViewers[i]) {
                linePosition = linePositionFromStripScene(viewer, scenePoint);
                break;
            }
        }
        if (!std::isfinite(linePosition)) {
            const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
            for (int slot = 0; slot < bottomCount; ++slot) {
                if (viewer == _bottomSliceViewers[static_cast<size_t>(slot)]) {
                    linePosition = bottomSliceLinePosition(slot, bottomCount);
                    break;
                }
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

void LineAnnotationDialog::setCurrentLinePosition(double position)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }
    position = std::clamp(position, 0.0, static_cast<double>(_generatedViews.linePoints.size() - 1));
    const bool currentChanged = std::abs(position - _currentLinePosition) >= 1.0e-3;
    const bool bottomChanged = std::abs(position - _bottomCenterPosition) >= 1.0e-3;
    if (!currentChanged && !bottomChanged) {
        return;
    }
    if (currentChanged && _generatedViews.currentCutSurface && !_currentCutStraightOffsetActive) {
        _currentLinePosition = position;
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    } else {
        _currentLinePosition = position;
    }
    if (currentChanged && _currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation current cut");
    }
    if (bottomChanged) {
        _bottomCenterPosition = position;
        renderBottomSlicePlanes("line annotation bottom cuts follow current");
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

bool LineAnnotationDialog::shiftBottomSlicesByScrollSteps(int steps)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return true;
    }
    const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
    if (bottomCount <= 0) {
        return true;
    }
    const int sliceStepSize = _viewerManager ? _viewerManager->sliceStepSize() : 1;
    const double position = vc3d::line_annotation::shiftedLinePosition(
        _bottomCenterPosition,
        steps,
        sliceStepSize,
        static_cast<int>(_generatedViews.linePoints.size()));
    setCurrentLinePosition(position);
    return true;
}

bool LineAnnotationDialog::scaleBottomSliceLineStepByScrollSteps(int steps)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return true;
    }
    const double lineStep = vc3d::line_annotation::adjustedBottomCrossSliceLineStep(
        _bottomSliceLineStep,
        steps,
        static_cast<int>(_generatedViews.linePoints.size()));
    if (std::abs(lineStep - _bottomSliceLineStep) < 1.0e-6) {
        return true;
    }
    _bottomSliceLineStep = lineStep;
    updateBottomSliceStepLabel();
    renderBottomSlicePlanes("line annotation bottom cut spacing");
    rebuildGeneratedDynamicOverlays();
    return true;
}

bool LineAnnotationDialog::handleBottomSliceStepWheel(QWheelEvent* event)
{
    if (!event) {
        return false;
    }
    const Qt::KeyboardModifiers modifiers = event->modifiers();
    if (!modifiers.testFlag(Qt::ControlModifier) ||
        !modifiers.testFlag(Qt::ShiftModifier) ||
        !_hasGeneratedViews ||
        _generatedViews.linePoints.empty()) {
        return false;
    }

    _bottomSliceStepWheelAccum += event->angleDelta().y();
    constexpr int kStepThreshold = 120;
    const int steps = _bottomSliceStepWheelAccum / kStepThreshold;
    if (steps != 0) {
        _bottomSliceStepWheelAccum -= steps * kStepThreshold;
        scaleBottomSliceLineStepByScrollSteps(steps);
    }
    event->accept();
    return true;
}

void LineAnnotationDialog::setCurrentCutFollowsStripMouse(bool follows)
{
    _currentCutFollowsStripMouse = follows;
}

void LineAnnotationDialog::recenterBottomSlicesOnCurrentPosition()
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }
    setCurrentLinePosition(snappedControlPointPosition(_currentLinePosition));
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
    _initialBottomCenterPosition = _bottomCenterPosition;
    _initialBottomSliceLineStep = _bottomSliceLineStep;

    _haveInitialCurrentCutCamera = false;
    if (_currentCutViewer) {
        _initialCurrentCutCamera = _currentCutViewer->cameraState();
        _haveInitialCurrentCutCamera = true;
    }
    _initialStripCameras.clear();
    _initialStripCameras.reserve(_stripViewers.size());
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            _initialStripCameras.push_back(viewer->cameraState());
        }
    }
    _initialBottomSliceCameras.clear();
    _initialBottomSliceCameras.reserve(_bottomSliceViewers.size());
    for (const auto& viewer : _bottomSliceViewers) {
        if (viewer) {
            _initialBottomSliceCameras.push_back(viewer->cameraState());
        }
    }
}

void LineAnnotationDialog::restoreInitialGeneratedViewerCameras()
{
    if (_currentCutViewer && _haveInitialCurrentCutCamera) {
        _currentCutViewer->applyCameraState(_initialCurrentCutCamera, false);
    }
    for (size_t i = 0; i < _stripViewers.size() && i < _initialStripCameras.size(); ++i) {
        if (_stripViewers[i]) {
            _stripViewers[i]->applyCameraState(_initialStripCameras[i], false);
        }
    }
    for (size_t i = 0; i < _bottomSliceViewers.size() && i < _initialBottomSliceCameras.size(); ++i) {
        if (_bottomSliceViewers[i]) {
            _bottomSliceViewers[i]->applyCameraState(_initialBottomSliceCameras[i], false);
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
        _initialBottomCenterPosition,
        _initialBottomSliceLineStep);
    _currentLinePosition = std::clamp(state.currentLinePosition,
                                      0.0,
                                      static_cast<double>(_generatedViews.linePoints.size() - 1));
    _bottomCenterPosition = std::clamp(state.bottomCenterPosition,
                                       0.0,
                                       static_cast<double>(_generatedViews.linePoints.size() - 1));
    _bottomSliceLineStep = state.bottomSliceLineStep;
    _currentCutManualRotation = state.currentCutManualRotation;
    _currentCutManualRotationActive = state.currentCutManualRotationActive;
    _currentCutStraightOffsetActive = false;
    _bottomSliceStepWheelAccum = 0;
    _currentCutFollowsStripMouse = true;
    updateBottomSliceStepLabel();

    if (_generatedViews.currentCutSurface) {
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    }
    renderBottomSlicePlanes("line annotation reset bottom cuts");
    restoreInitialGeneratedViewerCameras();
    if (_currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation reset current cut");
    }
    for (const auto& viewer : _stripViewers) {
        if (viewer) {
            viewer->renderVisible(true, "line annotation reset strip view");
        }
    }
    for (const auto& viewer : _bottomSliceViewers) {
        if (viewer) {
            viewer->renderVisible(true, "line annotation reset bottom view");
        }
    }
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::renderBottomSlicePlanes(const char* reason)
{
    const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
    if (bottomCount <= 0) {
        return;
    }
    for (int slot = 0; slot < bottomCount; ++slot) {
        auto* viewer = _bottomSliceViewers[static_cast<size_t>(slot)].data();
        if (!viewer) {
            continue;
        }
        const double position = bottomSliceLinePosition(slot, bottomCount);
        const auto& plane = _generatedViews.bottomCutSurfaces[static_cast<size_t>(slot)].second;
        (void)updatePlaneSurface(plane.get(), position);
        viewer->renderVisible(true, reason);
    }
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
    const int count = static_cast<int>(_generatedViews.linePoints.size());
    const int visibleCount = std::min(kBottomCrossSliceCount, count);
    std::vector<double> markerLinePositions;
    markerLinePositions.reserve(static_cast<size_t>(visibleCount + 1));
    for (int i = 0; i < visibleCount; ++i) {
        markerLinePositions.push_back(bottomSliceLinePosition(i, visibleCount));
    }
    return vc3d::line_annotation::makeGeneratedStripOverlay(_generatedViews,
                                                            _currentLinePosition,
                                                            markerLinePositions);
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::staticStripOverlay() const
{
    return vc3d::line_annotation::makeGeneratedStaticStripOverlay(_generatedViews);
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::dynamicStripOverlay() const
{
    const int count = static_cast<int>(_generatedViews.linePoints.size());
    const int visibleCount = std::min(kBottomCrossSliceCount, count);
    std::vector<double> markerLinePositions;
    markerLinePositions.reserve(static_cast<size_t>(visibleCount + 1));
    for (int i = 0; i < visibleCount; ++i) {
        markerLinePositions.push_back(bottomSliceLinePosition(i, visibleCount));
    }
    return vc3d::line_annotation::makeGeneratedDynamicStripOverlay(_generatedViews,
                                                                   _currentLinePosition,
                                                                   markerLinePositions);
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

    const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
    if (bottomCount <= 0 || _generatedViews.linePoints.empty()) {
        return;
    }
    for (int slot = 0; slot < bottomCount; ++slot) {
        auto* viewer = _bottomSliceViewers[static_cast<size_t>(slot)].data();
        if (!viewer) {
            continue;
        }
        const double position = bottomSliceLinePosition(slot, bottomCount);
        const auto& plane = _generatedViews.bottomCutSurfaces[static_cast<size_t>(slot)].second;
        applyOverlayForViewer(_generatedViews.bottomCutSurfaces[static_cast<size_t>(slot)].first,
                              viewer,
                              zSliceOverlay(position, false, viewer, plane.get()));
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
    linePosition = std::clamp(linePosition,
                              0.0,
                              static_cast<double>(_generatedViews.linePoints.size() - 1));
    int lower = static_cast<int>(std::floor(linePosition));
    int upper = std::min<int>(lower + 1, static_cast<int>(_generatedViews.linePoints.size()) - 1);
    if (lower == upper && lower > 0) {
        --lower;
    }
    cv::Vec3f tangent = _generatedViews.linePoints[static_cast<size_t>(upper)] -
                        _generatedViews.linePoints[static_cast<size_t>(lower)];
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

double LineAnnotationDialog::bottomSliceLinePosition(int slot, int bottomCount) const
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty() || bottomCount <= 0) {
        return 0.0;
    }
    return vc3d::line_annotation::bottomCrossSliceLinePosition(
        _bottomCenterPosition,
        slot,
        bottomCount,
        static_cast<int>(_generatedViews.linePoints.size()),
        _bottomSliceLineStep);
}

void LineAnnotationDialog::updateBottomSliceStepLabel()
{
    if (!_bottomSliceStepLabel) {
        return;
    }
    _bottomSliceStepLabel->setText(tr("Small step: %1")
                                       .arg(QString::number(_bottomSliceLineStep, 'g', 3)));
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
    if (event->type() == QEvent::Wheel) {
        auto* wheelEvent = static_cast<QWheelEvent*>(event);
        if (handleBottomSliceStepWheel(wheelEvent)) {
            return true;
        }
    }
    if (event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (handleKeyPress(keyEvent)) {
            return true;
        }
    }
    if (watched == _generatedTopWidget && event->type() == QEvent::Resize && _currentCutViewer) {
        _currentCutViewer->setFixedWidth(std::max(1, _generatedTopWidget->height()));
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
