#pragma once

#include <QGraphicsView>
#include <QHash>
#include <QMainWindow>
#include <QPointer>
#include <QPoint>
#include <QPointF>
#include <QRectF>
#include <QString>

#include <cstdint>
#include <vector>

#include "FiberNetworkLayout.hpp"

class LineAnnotationController;
class QDockWidget;
class QEvent;
class QGraphicsItem;
class QGraphicsPathItem;
class QGraphicsScene;
class QLabel;
class QMouseEvent;
class QSpinBox;
class QTreeWidget;
class QWheelEvent;

// Pan/zoom view of the fiber map, with the same gestures as the volume viewers:
// right-drag pans, the wheel zooms. Left clicks are reported as selection
// requests; ctrl+right-click without a drag asks for the control-point menu.
class FiberMapView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit FiberMapView(QWidget* parent = nullptr);

signals:
    void clicked(QPointF scenePos);
    void controlPointMenuRequested(QPointF scenePos, QPoint globalPos);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    // Right-button press position, against which the pan is called a drag (and
    // the ctrl+right menu suppressed), plus the running pan reference.
    QPoint _pressPosition;
    QPoint _panPosition;
    bool _pressed = false;
    bool _panning = false;
    bool _panDragged = false;
    bool _menuPending = false;
};

// Interactive 2D map of the linked fiber networks, unrolled about the scroll
// umbilicus. The layout is only ever rebuilt on explicit request; fiber
// changes just mark the current one stale.
class FiberMapWorkspace : public QMainWindow
{
    Q_OBJECT

public:
    explicit FiberMapWorkspace(LineAnnotationController* controller,
                               QWidget* parent = nullptr);

signals:
    void openFiberAtControlPointRequested(uint64_t fiberId, int controlPointIndex);

protected:
    // The map's colours follow the application theme, and a switch is only
    // announced by a palette change.
    void changeEvent(QEvent* event) override;

private:
    // Scene-space copy of the placed fiber (y negated once, so scroll z reads
    // upward) alongside the two path items carrying its geometry.
    struct FiberEntry {
        vc3d::fiber_map::PlacedFiber fiber;
        QGraphicsPathItem* tracedItem = nullptr;
        QGraphicsPathItem* interpolatedItem = nullptr;
    };

    void rebuildLayout();
    void rebuildScene(const QString& emptyMessage);
    void rebuildTree();
    void markStale();
    void setHighlightedFiber(uint64_t fiberId);
    void clearControlPointDots();
    void handleSceneClick(const QPointF& scenePos);
    void handleControlPointMenu(const QPointF& scenePos, const QPoint& globalPos);
    void selectFiberRow(uint64_t fiberId);
    [[nodiscard]] uint64_t fiberAt(const QPointF& scenePos) const;
    [[nodiscard]] double sceneTolerance(double viewPixels) const;

    // QPointer: the controller is owned elsewhere and dies before this widget
    // during CWindow teardown; guards keep late signals harmless.
    QPointer<LineAnnotationController> _controller;
    FiberMapView* _view = nullptr;
    QGraphicsScene* _scene = nullptr;
    QTreeWidget* _tree = nullptr;
    QDockWidget* _fiberDock = nullptr;
    QSpinBox* _topNetworkSpin = nullptr;
    QSpinBox* _minFiberSpin = nullptr;
    QLabel* _statusLabel = nullptr;
    vc3d::fiber_map::Result _layout;
    QHash<uint64_t, FiberEntry> _entries;
    std::vector<QGraphicsItem*> _controlPointDots;
    // Scroll top in scene z, cm; 0 when the volume's extent is unknown.
    double _scrollZMaxCm = 0.0;
    // The scene rect keeps slack on either side so a zoomed-in view can pan
    // past the outer panels; this is the tight rect around the content, which
    // is what the first-build fit frames.
    QRectF _contentRect;
    // What the empty scene last said, so a theme change can rebuild the scene as
    // it stands rather than take a fresh snapshot to work out the message again.
    QString _emptyMessage;
    uint64_t _highlightedFiber = 0;
    bool _syncingSelection = false;
    bool _viewFitted = false;
    bool _fiberDockSized = false;
    bool _retheming = false;
};
