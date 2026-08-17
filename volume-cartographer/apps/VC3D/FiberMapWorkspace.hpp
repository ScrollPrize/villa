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
#include <optional>
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
class QShowEvent;
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
    // Catches a layout built from fiber data that has since changed.
    void showEvent(QShowEvent* event) override;

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
    // Compares the controller's fiber generation against the one this layout
    // was built from, marking stale on a mismatch; returns whether the map is
    // stale. Cheap (one integer compare), so it guards interaction too.
    bool refreshStaleState();
    // Appends the package's umbilicus state to a status line, resolving it at
    // most once per fiber generation: it is filesystem work, and the whole point
    // of the generation check is to keep that off the annotation paths.
    [[nodiscard]] QString withCachedUmbilicusStatus(const QString& status);
    // Scene units (voxels) per centimetre, from the package's voxel size when it
    // has one and from the documented assumption otherwise. This is the only
    // route from the map's cm-valued styling constants into the voxel-space
    // scene; it is never allowed to produce displayed text, because when the
    // voxel size is unknown it is a guess.
    [[nodiscard]] double sceneVxPerCm() const;
    // A layout length (voxels) as display text: centimetres when the voxel size
    // is known, otherwise the voxel count itself, which is the one figure still
    // true when the package cannot say how big a voxel is.
    [[nodiscard]] QString formatMapLength(double valueVx) const;
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
    // Annotation voxel size of the snapshot the current layout came from, in µm;
    // unset when the package could not say, in which case nothing physical is
    // displayed.
    std::optional<double> _voxelSizeUm;
    // Scroll top in scene z, i.e. voxels; 0 when the volume's extent is unknown.
    double _scrollZMaxVx = 0.0;
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
    // Fiber generation the current layout came from, and whether a change has
    // been seen since; a fresh workspace is stale until its first rebuild.
    uint64_t _layoutGeneration = 0;
    bool _stale = false;
    QString _umbilicusStatusText;
    uint64_t _umbilicusStatusGeneration = 0;
    bool _umbilicusStatusValid = false;
};
