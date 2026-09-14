#pragma once

#include <QColor>
#include <QWidget>

#include <optional>
#include <vector>

#include "FiberNetworkLayout.hpp"

class QGraphicsView;
class QPainter;

// What the rulers read off the current layout. An empty model (hasLayout
// false) paints a blank band.
struct FiberMapRulerModel {
    bool hasLayout = false;
    // Winding gridlines: scene x per integer winding.
    std::vector<vc3d::fiber_map::WindingMark> windings;
    // Sheet distance as a function of scene x (see FiberNetworkLayout.hpp).
    vc3d::fiber_map::SheetModel sheet;
    // Unset when the package could not say, in which case the distance rulers
    // count voxels rather than guess a physical length.
    std::optional<double> voxelSizeUm;
};

struct FiberMapRulerStyle {
    QColor background;
    QColor ink;
    QColor tick;
};

// One ruler band along an edge of the Fiber Map view. It lives in the view's
// viewport margin, so it never covers the map, and it reads the view
// transform on every paint: whatever is scrolled or zoomed into view, the
// labels for it are on the edge. Three modes:
//   Windings      - the winding number at every gridline (top edge)
//   SheetDistance - distance along the sheet from winding 0 (bottom edge)
//   Height        - scroll height above the volume floor (left edge)
// The distance modes label in physical units when the voxel size is known and
// in voxels otherwise; the tick step comes from a 1-2-5 ladder so that ticks
// stay a readable distance apart at any zoom.
class FiberMapRuler : public QWidget
{
    Q_OBJECT

public:
    enum class Edge { Top, Left, Bottom };
    enum class Mode { Windings, SheetDistance, Height };

    // Band thickness across the edge, in device-independent pixels.
    static int thicknessFor(Edge edge);

    FiberMapRuler(QGraphicsView* view, Edge edge, Mode mode, QWidget* parent = nullptr);

    void setModel(FiberMapRulerModel model);
    void setRulerStyle(const FiberMapRulerStyle& style);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void paintWindings(QPainter& painter);
    void paintSheetDistance(QPainter& painter);
    void paintHeight(QPainter& painter);
    // The unit caption at the far end of the band; returns the rect it took so
    // labels can stay clear of it.
    QRect paintCaption(QPainter& painter, const QString& caption);
    void updateToolTip();

    QGraphicsView* _view = nullptr;
    Edge _edge;
    Mode _mode;
    FiberMapRulerModel _model;
    FiberMapRulerStyle _style;
};
