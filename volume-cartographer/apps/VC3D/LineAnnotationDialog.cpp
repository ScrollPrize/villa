#include "LineAnnotationDialog.hpp"

#include "ViewerManager.hpp"
#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QBrush>
#include <QComboBox>
#include <QEvent>
#include <QKeyEvent>
#include <QHBoxLayout>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

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

bool finiteScenePoint(const QPointF& point)
{
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

QPointF quadGridToScene(CChunkedVolumeViewer* viewer, QuadSurface* surface, int row, int col)
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
    const float surfaceX = (static_cast<float>(col) - static_cast<float>(points->cols) / 2.0f) / scale[0];
    const float surfaceY = (static_cast<float>(row) - static_cast<float>(points->rows) / 2.0f) / scale[1];
    return viewer->surfaceCoordsToScene(surfaceX, surfaceY);
}

} // namespace

LineAnnotationDialog::LineAnnotationDialog(ViewerManager* viewerManager, QWidget* parent)
    : QDialog(parent)
    , _viewerManager(viewerManager)
{
    setWindowTitle(tr("Line Annotation"));
    setAttribute(Qt::WA_DeleteOnClose);
    resize(900, 700);

    _layout = new QVBoxLayout(this);
    _layout->setContentsMargins(0, 0, 0, 0);
    _layout->setSpacing(0);

    auto* buttonRow = new QWidget(this);
    auto* buttonLayout = new QHBoxLayout(buttonRow);
    buttonLayout->setContentsMargins(6, 6, 6, 6);
    buttonLayout->setSpacing(6);
    _initialDirectionCombo = new QComboBox(buttonRow);
    _initialDirectionCombo->addItem(tr("sideways"), static_cast<int>(InitialDirectionMode::Sideways));
    _initialDirectionCombo->addItem(tr("z (in/out)"), static_cast<int>(InitialDirectionMode::ZInOut));
    _initialDirectionCombo->setCurrentIndex(1);
    _initialDirectionCombo->installEventFilter(this);
    buttonLayout->addWidget(_initialDirectionCombo);
    _showAsMeshButton = new QPushButton(tr("show as mesh"), buttonRow);
    _showAsMeshButton->setEnabled(false);
    _showAsMeshButton->installEventFilter(this);
    buttonLayout->addWidget(_showAsMeshButton);
    buttonLayout->addStretch(1);
    _layout->addWidget(buttonRow, 0);
    connect(_showAsMeshButton, &QPushButton::clicked, this, [this]() {
        emit showAsMeshRequested();
    });

    _mdiArea = new QMdiArea(this);
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
    const double maxLinePosition = static_cast<double>(views.linePoints.size() - 1);
    _currentLinePosition = std::clamp(static_cast<double>(views.initialCenterIndex),
                                      0.0,
                                      maxLinePosition);
    _bottomCenterPosition = _currentLinePosition;
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
    currentViewer->applyCameraState(generatedPaneCamera(currentViewer, camera), false);
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
                    emit generatedControlPointRequested(_generatedViews.currentCutName,
                                                        volumePoint,
                                                        _currentLinePosition);
                }
            });
    topLayout->addWidget(currentViewer, 0);
    _currentCutViewer = currentViewer;
    _panes.push_back(Pane{views.currentCutName, currentViewer, {}});

    auto* stripStack = new QWidget(topWidget);
    auto* stripLayout = new QVBoxLayout(stripStack);
    stripLayout->setContentsMargins(0, 0, 0, 0);
    stripLayout->setSpacing(0);
    topLayout->addWidget(stripStack, 1);

    const std::pair<std::string, QString> stripSpecs[] = {
        {views.lineSurfaceName, views.lineSurfaceTitle},
        {views.lineSideSliceName, views.lineSideSliceTitle},
    };
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
        viewer->applyCameraState(generatedPaneCamera(viewer, camera), false);
        bindPaneInteractions(surfaceName, viewer, false);
        connect(viewer,
                &CChunkedVolumeViewer::sendMouseMoveVolume,
                this,
                [this, viewer](cv::Vec3f, Qt::MouseButtons, Qt::KeyboardModifiers, QPointF scenePoint) {
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
                        setCurrentLinePosition(position);
                        emit generatedControlPointRequested(surfaceName, volumePoint, position);
                    }
                });
        stripLayout->addWidget(viewer, 1);
        _stripViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
    }

    auto* bottomWidget = new QWidget(this);
    auto* bottomLayout = new QHBoxLayout(bottomWidget);
    bottomLayout->setContentsMargins(0, 0, 0, 0);
    bottomLayout->setSpacing(0);
    _generatedContainers.push_back(bottomWidget);
    _layout->addWidget(bottomWidget, 1);

    const int bottomCount = std::min<int>(7, static_cast<int>(views.bottomCutSurfaces.size()));
    const double first = std::clamp(_bottomCenterPosition - static_cast<double>(bottomCount / 2),
                                    0.0,
                                    std::max(0.0, maxLinePosition - static_cast<double>(bottomCount - 1)));
    for (int slot = 0; slot < bottomCount; ++slot) {
        const double position = std::clamp(first + static_cast<double>(slot),
                                           0.0,
                                           maxLinePosition);
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
        viewer->applyCameraState(generatedPaneCamera(viewer, camera), false);
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
                        const double maxLinePosition = _generatedViews.linePoints.empty()
                            ? 0.0
                            : static_cast<double>(_generatedViews.linePoints.size() - 1);
                        const double first = std::clamp(
                            _bottomCenterPosition - static_cast<double>(bottomCount / 2),
                            0.0,
                            std::max(0.0, maxLinePosition - static_cast<double>(bottomCount - 1)));
                        const double linePosition = std::clamp(first + static_cast<double>(slot),
                                                               0.0,
                                                               maxLinePosition);
                        setCurrentLinePosition(linePosition);
                        emit generatedControlPointRequested(surfaceName, volumePoint, linePosition);
                    }
                });
        bottomLayout->addWidget(viewer, 1);
        _bottomSliceViewers.push_back(viewer);
        _panes.push_back(Pane{surfaceName, viewer, {}});
    }

    if (_generatedTopWidget && _currentCutViewer) {
        _currentCutViewer->setFixedWidth(std::max(1, _generatedTopWidget->height()));
    }

    rebuildGeneratedOverlays();
    if (_showAsMeshButton) {
        _showAsMeshButton->setEnabled(true);
    }
    return true;
}

void LineAnnotationDialog::applyGeneratedOverlay(const std::string& surfaceName,
                                                 CChunkedVolumeViewer* viewer,
                                                 const GeneratedOverlay& overlay)
{
    applyOverlayForViewer(surfaceName, viewer, overlay);
}

void LineAnnotationDialog::applyOverlayForViewer(const std::string& surfaceName,
                                                 CChunkedVolumeViewer* viewer,
                                                 const GeneratedOverlay& overlay)
{
    if (!viewer) {
        return;
    }

    const auto key = "line_annotation_overlay_" + surfaceName;
    std::vector<ViewerOverlayControllerBase::OverlayPrimitive> primitives;
    primitives.reserve(3);

    ViewerOverlayControllerBase::OverlayStyle lineStyle;
    lineStyle.penColor = QColor(0, 220, 255, 190);
    lineStyle.penWidth = 1.0;
    lineStyle.z = 150.0;

    ViewerOverlayControllerBase::OverlayStyle seedStyle;
    seedStyle.penColor = QColor(255, 230, 0, 220);
    seedStyle.brushColor = QColor(255, 230, 0, 170);
    seedStyle.penWidth = 1.5;
    seedStyle.z = 152.0;

    ViewerOverlayControllerBase::OverlayStyle markerStyle;
    markerStyle.penColor = QColor(0, 220, 255, 210);
    markerStyle.brushColor = QColor(0, 220, 255, 150);
    markerStyle.penWidth = 1.0;
    markerStyle.z = 151.0;

    ViewerOverlayControllerBase::OverlayStyle currentMarkerStyle = markerStyle;
    currentMarkerStyle.penColor = QColor(0, 245, 255, 245);
    currentMarkerStyle.brushColor = QColor(0, 245, 255, 210);
    currentMarkerStyle.penWidth = 1.5;
    currentMarkerStyle.z = 153.0;

    std::vector<QPointF> sceneLine;
    QPointF seedScene;
    bool hasSeedScene = false;

    if (overlay.useSurfaceCenterLine) {
        auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
        const auto* points = quad ? quad->rawPointsPtr() : nullptr;
        if (points && !points->empty()) {
            const int row = points->rows / 2;
            sceneLine.reserve(static_cast<size_t>(points->cols));
            for (int col = 0; col < points->cols; ++col) {
                const QPointF scenePoint = quadGridToScene(viewer, quad, row, col);
                if (finiteScenePoint(scenePoint)) {
                    sceneLine.push_back(scenePoint);
                }
            }
            if (overlay.controlPoints.empty() &&
                overlay.seedLineIndex >= 0 &&
                overlay.seedLineIndex < points->cols) {
                seedScene = stripLinePositionToScene(viewer, quad, overlay.seedLineIndex);
                hasSeedScene = finiteScenePoint(seedScene);
            }
            if (std::isfinite(overlay.currentLinePosition)) {
                const QPointF markerScene =
                    stripLinePositionToScene(viewer, quad, overlay.currentLinePosition);
                if (finiteScenePoint(markerScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                        markerScene,
                        4.0,
                        true,
                        currentMarkerStyle});
                }
            }
            for (const double position : overlay.markerLinePositions) {
                if (!std::isfinite(position) ||
                    position < 0.0 ||
                    position > static_cast<double>(points->cols - 1) ||
                    std::abs(position - overlay.currentLinePosition) < 1.0e-6) {
                    continue;
                }
                const QPointF markerScene = stripLinePositionToScene(viewer, quad, position);
                if (finiteScenePoint(markerScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                        markerScene,
                        2.5,
                        true,
                        markerStyle});
                }
            }
            for (const auto& control : overlay.controlPoints) {
                if (!std::isfinite(control.linePosition) ||
                    control.linePosition < 0.0 ||
                    control.linePosition > static_cast<double>(points->cols - 1)) {
                    continue;
                }
                const QPointF controlScene = stripLinePositionToScene(viewer, quad, control.linePosition);
                if (finiteScenePoint(controlScene)) {
                    primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                        controlScene,
                        control.isSeed ? 5.5 : 5.0,
                        true,
                        seedStyle});
                }
            }
        }
    } else if (!overlay.linePoints.empty()) {
        sceneLine.reserve(overlay.linePoints.size());
        for (const auto& point : overlay.linePoints) {
            if (!finitePoint(point)) {
                continue;
            }
            const QPointF scenePoint = viewer->volumeToScene(point);
            if (finiteScenePoint(scenePoint)) {
                sceneLine.push_back(scenePoint);
            }
        }
    }

    if (!overlay.useSurfaceCenterLine) {
        for (const auto& control : overlay.controlPoints) {
            if (!finitePoint(control.point)) {
                continue;
            }
            const QPointF controlScene = viewer->volumeToScene(control.point);
            if (finiteScenePoint(controlScene)) {
                primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                    controlScene,
                    control.isSeed ? 5.5 : 5.0,
                    true,
                    seedStyle});
            }
        }
    }

    if (sceneLine.size() >= 2) {
        primitives.push_back(ViewerOverlayControllerBase::LineStripPrimitive{
            sceneLine,
            false,
            lineStyle});
    }

    if (finitePoint(overlay.pointMarker)) {
        const QPointF markerScene = viewer->volumeToScene(overlay.pointMarker);
        if (finiteScenePoint(markerScene)) {
            primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
                markerScene,
                overlay.emphasizedPointMarker ? 6.0 : 4.0,
                true,
                overlay.emphasizedPointMarker ? currentMarkerStyle : markerStyle});
        }
    }

    if (!hasSeedScene && finitePoint(overlay.seedPoint)) {
        seedScene = viewer->volumeToScene(overlay.seedPoint);
        hasSeedScene = finiteScenePoint(seedScene);
    }

    if (hasSeedScene) {
        const bool emphasizedSeed = overlay.emphasizedPointMarker &&
                                    !finitePoint(overlay.pointMarker);
        const qreal radius = emphasizedSeed ? 6.0 : 4.0;
        if (emphasizedSeed) {
            seedStyle.penColor = QColor(255, 245, 0, 255);
            seedStyle.brushColor = QColor(255, 245, 0, 220);
            seedStyle.penWidth = 2.0;
        }
        primitives.push_back(ViewerOverlayControllerBase::CirclePrimitive{
            seedScene,
            radius,
            true,
            seedStyle});
    }

    ViewerOverlayControllerBase::applyPrimitives(viewer, key, std::move(primitives));
}

double LineAnnotationDialog::linePositionFromStripScene(CChunkedVolumeViewer* viewer,
                                                        const QPointF& scenePoint) const
{
    if (!viewer || !_hasGeneratedViews) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto* quad = dynamic_cast<QuadSurface*>(viewer->currentSurface());
    const auto* points = quad ? quad->rawPointsPtr() : nullptr;
    if (!points || points->cols <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Vec2f surfacePoint = viewer->sceneToSurfaceCoords(scenePoint);
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || !std::isfinite(surfacePoint[0])) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double position = static_cast<double>(surfacePoint[0] * scale[0]) +
                            static_cast<double>(points->cols) / 2.0;
    return std::clamp(position, 0.0, static_cast<double>(points->cols - 1));
}

void LineAnnotationDialog::setCurrentLinePosition(double position)
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }
    position = std::clamp(position, 0.0, static_cast<double>(_generatedViews.linePoints.size() - 1));
    if (std::abs(position - _currentLinePosition) < 1.0e-3 && _currentCutViewer) {
        return;
    }
    _currentLinePosition = position;
    if (_generatedViews.currentCutSurface) {
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    }
    if (_currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation current cut");
    }
    rebuildGeneratedOverlays();
}

void LineAnnotationDialog::recenterBottomSlicesOnCurrentPosition()
{
    if (!_hasGeneratedViews || _generatedViews.linePoints.empty()) {
        return;
    }
    _currentLinePosition = snappedControlPointPosition(_currentLinePosition);
    if (_generatedViews.currentCutSurface) {
        (void)updatePlaneSurface(_generatedViews.currentCutSurface.get(), _currentLinePosition);
    }
    if (_currentCutViewer) {
        _currentCutViewer->renderVisible(true, "line annotation current cut");
    }
    _bottomCenterPosition = _currentLinePosition;
    const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
    if (bottomCount <= 0) {
        return;
    }
    const double maxLinePosition = static_cast<double>(_generatedViews.linePoints.size() - 1);
    const double first = std::clamp(_bottomCenterPosition - static_cast<double>(bottomCount / 2),
                                    0.0,
                                    std::max(0.0, maxLinePosition - static_cast<double>(bottomCount - 1)));
    for (int slot = 0; slot < bottomCount; ++slot) {
        auto* viewer = _bottomSliceViewers[static_cast<size_t>(slot)].data();
        if (!viewer) {
            continue;
        }
        const double position = std::clamp(first + static_cast<double>(slot),
                                           0.0,
                                           maxLinePosition);
        const auto& plane = _generatedViews.bottomCutSurfaces[static_cast<size_t>(slot)].second;
        (void)updatePlaneSurface(plane.get(), position);
        viewer->renderVisible(true, "line annotation bottom cuts");
    }
    rebuildGeneratedOverlays();
}

double LineAnnotationDialog::snappedControlPointPosition(double position) const
{
    if (!_hasGeneratedViews || _generatedViews.controlPoints.empty()) {
        return position;
    }
    double bestPosition = position;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const auto& control : _generatedViews.controlPoints) {
        if (!std::isfinite(control.linePosition)) {
            continue;
        }
        const double distance = std::abs(control.linePosition - position);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = control.linePosition;
        }
    }
    return bestDistance <= 0.5 ? bestPosition : position;
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::stripOverlay() const
{
    GeneratedOverlay overlay;
    overlay.linePoints = _generatedViews.linePoints;
    overlay.seedPoint = _generatedViews.seedPoint;
    overlay.seedLineIndex = _generatedViews.controlPoints.empty()
        ? _generatedViews.seedLineIndex
        : -1;
    overlay.useSurfaceCenterLine = true;
    overlay.currentLinePosition = _currentLinePosition;
    overlay.controlPoints = _generatedViews.controlPoints;

    const int count = static_cast<int>(_generatedViews.linePoints.size());
    const int visibleCount = std::min(7, count);
    const double maxFirst = std::max(0.0, static_cast<double>(count - visibleCount));
    const double first = visibleCount > 0
        ? std::clamp(_bottomCenterPosition - static_cast<double>(visibleCount / 2),
                     0.0,
                     maxFirst)
        : 0.0;
    overlay.markerLinePositions.reserve(static_cast<size_t>(visibleCount + 1));
    for (int i = 0; i < visibleCount; ++i) {
        overlay.markerLinePositions.push_back(std::clamp(first + static_cast<double>(i),
                                                         0.0,
                                                         static_cast<double>(count - 1)));
    }
    return overlay;
}

LineAnnotationDialog::GeneratedOverlay LineAnnotationDialog::zSliceOverlay(double linePosition,
                                                                           bool emphasized) const
{
    GeneratedOverlay overlay;
    overlay.pointMarker = interpolatedLinePoint(linePosition);
    overlay.emphasizedPointMarker = emphasized;
    for (const auto& control : _generatedViews.controlPoints) {
        if (std::isfinite(control.linePosition) &&
            std::abs(control.linePosition - linePosition) <= 0.5) {
            overlay.controlPoints.push_back(control);
        }
    }
    return overlay;
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
                              zSliceOverlay(_currentLinePosition, true));
    }

    const int bottomCount = static_cast<int>(_bottomSliceViewers.size());
    if (bottomCount <= 0 || _generatedViews.linePoints.empty()) {
        return;
    }
    const double maxLinePosition = static_cast<double>(_generatedViews.linePoints.size() - 1);
    const double first = std::clamp(_bottomCenterPosition - static_cast<double>(bottomCount / 2),
                                    0.0,
                                    std::max(0.0, maxLinePosition - static_cast<double>(bottomCount - 1)));
    for (int slot = 0; slot < bottomCount; ++slot) {
        auto* viewer = _bottomSliceViewers[static_cast<size_t>(slot)].data();
        if (!viewer) {
            continue;
        }
        const double position = std::clamp(first + static_cast<double>(slot),
                                           0.0,
                                           maxLinePosition);
        applyOverlayForViewer(_generatedViews.bottomCutSurfaces[static_cast<size_t>(slot)].first,
                              viewer,
                              zSliceOverlay(position, false));
    }
}

cv::Vec3f LineAnnotationDialog::interpolatedLinePoint(double linePosition) const
{
    if (_generatedViews.linePoints.empty()) {
        return {std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()};
    }
    linePosition = std::clamp(linePosition,
                              0.0,
                              static_cast<double>(_generatedViews.linePoints.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(_generatedViews.linePoints.size()) - 1);
    const float t = static_cast<float>(linePosition - static_cast<double>(lower));
    return _generatedViews.linePoints[static_cast<size_t>(lower)] * (1.0f - t) +
           _generatedViews.linePoints[static_cast<size_t>(upper)] * t;
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
    cv::normalize(tangent, tangent);
    return tangent;
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
    cv::normalize(up, up);
    return up;
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
    if (event->key() == Qt::Key_Space && event->modifiers() == Qt::NoModifier) {
        recenterBottomSlicesOnCurrentPosition();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Escape ||
        (event->key() == Qt::Key_X && event->modifiers() == Qt::NoModifier)) {
        close();
        event->accept();
        return;
    }
    QDialog::keyPressEvent(event);
}

bool LineAnnotationDialog::eventFilter(QObject* watched, QEvent* event)
{
    if ((watched == _initialDirectionCombo || watched == _showAsMeshButton) &&
        event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Space && keyEvent->modifiers() == Qt::NoModifier) {
            recenterBottomSlicesOnCurrentPosition();
            keyEvent->accept();
            return true;
        }
    }
    if (watched == _generatedTopWidget && event->type() == QEvent::Resize && _currentCutViewer) {
        _currentCutViewer->setFixedWidth(std::max(1, _generatedTopWidget->height()));
    }
    return QDialog::eventFilter(watched, event);
}
