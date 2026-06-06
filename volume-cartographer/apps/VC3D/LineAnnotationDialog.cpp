#include "LineAnnotationDialog.hpp"

#include "Keybinds.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "ViewerManager.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QBrush>
#include <QComboBox>
#include <QEvent>
#include <QKeyEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPushButton>
#include <QRect>
#include <QVBoxLayout>
#include <QWidget>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {

constexpr int kBottomCrossSliceCount = 7;

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

LineAnnotationDialog::LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent)
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
    _fullOptimizationButton = new QPushButton(tr("full optimization"), buttonRow);
    _fullOptimizationButton->setEnabled(false);
    _fullOptimizationButton->installEventFilter(this);
    buttonLayout->addWidget(_fullOptimizationButton);
    buttonLayout->addStretch(1);
    _layout->addWidget(buttonRow, 0);
    connect(_showAsMeshButton, &QPushButton::clicked, this, [this]() {
        emit showAsMeshRequested();
    });
    connect(_fullOptimizationButton, &QPushButton::clicked, this, [this]() {
        emit fullOptimizationRequested();
    });

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

    clearGeneratedOverlayRefreshConnections();
    _suppressPaneClosed = true;
    if (_mdiArea) {
        _layout->removeWidget(_mdiArea);
        delete _mdiArea;
        _mdiArea = nullptr;
    }
    _suppressPaneClosed = false;
    _panes.clear();

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
    _hasGeneratedViews = true;
    _currentCutFollowsStripMouse = true;
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
    currentViewer->setShiftScrollOverride(
        [this](int steps, QPointF, Qt::KeyboardModifiers modifiers) {
            if (modifiers.testFlag(Qt::ControlModifier)) {
                return scaleBottomSliceLineStepByScrollSteps(steps);
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
                if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
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
        viewer->applyCameraState(static_cast<size_t>(stripIndex) < stripCameras.size()
                                     ? stripCameras[static_cast<size_t>(stripIndex)]
                                     : generatedPaneCamera(viewer, camera),
                                 false);
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
                    if (button != Qt::LeftButton || modifiers != Qt::NoModifier) {
                        return;
                    }
                    const double position = linePositionFromStripScene(viewer, scenePoint);
                    if (std::isfinite(position)) {
                        setCurrentCutFollowsStripMouse(true);
                        setCurrentLinePosition(position);
                        emit generatedControlPointRequested(surfaceName, volumePoint, position);
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
                    if (button == Qt::LeftButton && modifiers == Qt::NoModifier) {
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
    _currentLinePosition = position;
    if (currentChanged && _generatedViews.currentCutSurface) {
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    }
    if (currentChanged && _currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation current cut");
    }
    _bottomCenterPosition = position;
    renderBottomSlicePlanes("line annotation bottom cuts follow current");
    rebuildGeneratedOverlays();
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
    setCurrentLinePosition(position);
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
    rebuildGeneratedOverlays();
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

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::zSliceOverlay(double linePosition,
                                                                           bool emphasized,
                                                                           CChunkedVolumeViewer* viewer,
                                                                           PlaneSurface* plane) const
{
    return vc3d::line_annotation::makeGeneratedCrossSliceOverlayForPlane(_generatedViews,
                                                                         linePosition,
                                                                         emphasized,
                                                                         viewer,
                                                                         plane);
}

void LineAnnotationDialog::rebuildGeneratedOverlays()
{
    if (!_hasGeneratedViews) {
        return;
    }

    const GeneratedOverlay strip = stripOverlay();
    for (size_t i = 0; i < _stripViewers.size(); ++i) {
        auto* viewer = _stripViewers[i].data();
        if (!viewer) {
            continue;
        }
        const std::string key = i == 0 ? _generatedViews.lineSurfaceName
                                       : _generatedViews.lineSideSliceName;
        applyOverlayForViewer(key, viewer, strip);
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
    plane->setFromNormalAndUp(origin, tangent, upHint);
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

bool LineAnnotationDialog::handleKeyPress(QKeyEvent* event)
{
    if (!event) {
        return false;
    }
    if (event->key() == Qt::Key_Space && event->modifiers() == Qt::NoModifier) {
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
    if (event->key() == Qt::Key_Escape ||
        (event->key() == Qt::Key_X && event->modifiers() == Qt::NoModifier)) {
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
