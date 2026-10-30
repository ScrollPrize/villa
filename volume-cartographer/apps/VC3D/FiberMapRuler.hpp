#pragma once

#include <QColor>
#include <QFont>
#include <QRect>
#include <QString>

#include <optional>
#include <vector>

#include "FiberNetworkLayout.hpp"

class QGraphicsView;
class QPainter;

// What the rulers read off the current layout. An empty model (hasLayout
// false) paints nothing.
struct FiberMapRulerModel {
    bool hasLayout = false;
    // Winding gridlines: scene x per integer winding, ascending in x.
    std::vector<vc3d::fiber_map::WindingMark> windings;
    // The fitted sheet model the scene's x was scaled by (see
    // sheetDistanceMonotoneVx in FiberNetworkLayout.hpp). Scene x is already
    // the sheet distance from winding 0, so the rulers convert nothing; they
    // read the model for the tooltip and for a fallback winding width.
    vc3d::fiber_map::SheetModel sheet;
    // Unset when the package could not say, in which case the distance rulers
    // count voxels rather than guess a physical length.
    std::optional<double> voxelSizeUm;
    // The scroll extent the axes attach to and run along, in scene
    // coordinates: the ceiling (scene y of the top), the floor (scene y of the
    // bottom) and the map's left and right edges. A band never extends past
    // the extent: the horizontal ones run from the left edge to the right,
    // the vertical one from the ceiling to the floor.
    double extentTopSceneY = 0.0;
    double extentBottomSceneY = 0.0;
    double extentLeftSceneX = 0.0;
    double extentRightSceneX = 0.0;
    // The kollesis seams (FiberMapKollesis.hpp), ascending in scene x: a
    // tick each on the kollesis band, and between neighbours the sheet
    // length. Empty hides the band.
    struct KollesisMark {
        // Scene x of the seam.
        double xVx = 0.0;
        // Sheet length to the next mark (voxels); NaN on the last.
        double toNextVx = 0.0;
        // Predicted from the sheet-length statistics rather than tagged.
        bool predicted = false;
    };
    std::vector<KollesisMark> kollesis;
};

struct FiberMapRulerStyle {
    QColor background;
    QColor ink;
    QColor tick;
    // The kollesis band's ticks (the map's kollesis yellow), and the ticks
    // and lengths of predicted seams (the map's prediction amber).
    QColor accent;
    QColor accentPredicted;
};

// One axis of the Fiber Map, painted as an overlay in the view's foreground
// pass. It floats at the edge of the scroll extent while that edge is on
// screen - the labels sit just outside the map, in the empty ground beside it
// - and clamps to the viewport edge once the extent scrolls off, so whatever
// is in view is always labelled. It reads the view transform on every paint.
// Three modes:
//   Windings      - the winding number at every gridline (above the ceiling)
//   SheetDistance - distance along the sheet from winding 0 (below the
//                   floor); this is the scene x itself
//   Height        - scroll height above the volume floor (left of the map)
//   Kollesis      - a tick per kollesis seam and the sheet length between
//                   neighbours (above the ceiling, stacked outside the
//                   winding band; absent while the model has no seams)
// The distance modes label in physical units when the voxel size is known and
// in voxels otherwise; the tick step comes from a 1-2-5 ladder so that ticks
// stay a readable distance apart at any zoom.
class FiberMapRuler
{
public:
    enum class Edge { Top, Left, Bottom };
    enum class Mode { Windings, SheetDistance, Height, Kollesis };

    // Band thickness across the edge, in device-independent pixels.
    static int thicknessFor(Edge edge);

    // stackLevel: how many bands sit between this one and the extent edge
    // (0: against the edge). Bands on one edge stack outward from the map.
    FiberMapRuler(QGraphicsView* view, Edge edge, Mode mode, int stackLevel = 0);

    void setModel(FiberMapRulerModel model);
    void setStyle(const FiberMapRulerStyle& style);
    void setFont(const QFont& font);
    // How many bands with content stack outside this one on its edge, so
    // that once the extent edge scrolls off and the stack clamps to the
    // viewport the outer bands keep their place beyond this one.
    void setStackDepth(int depth);
    // Whether the band has anything to show for the current model (a
    // kollesis band with no seams has not, and takes no room).
    [[nodiscard]] bool hasContent() const;
    [[nodiscard]] Edge edge() const { return _edge; }
    [[nodiscard]] Mode mode() const { return _mode; }

    // Where the band lies for the current transform, in viewport
    // coordinates: against the extent edge when that is inside the viewport,
    // against the matching viewport edge otherwise, and along the edge only
    // as far as the extent reaches. Empty without a layout or when the
    // extent is entirely off screen along the band.
    [[nodiscard]] QRect bandRect(const QRect& viewport) const;

    // Paints the band into `painter`, which must be in viewport (device)
    // coordinates - the caller disables the world transform first.
    void paint(QPainter& painter, const QRect& viewport);

    // What the band means, for a tooltip over it.
    [[nodiscard]] QString toolTipText() const;

private:
    void paintWindings(QPainter& painter, const QRect& band);
    void paintSheetDistance(QPainter& painter, const QRect& band);
    void paintHeight(QPainter& painter, const QRect& band);
    void paintKollesis(QPainter& painter, const QRect& band);
    // The unit caption at the far end of the band; returns the rect it took so
    // labels can stay clear of it.
    QRect paintCaption(QPainter& painter, const QRect& band, const QString& caption);

    QGraphicsView* _view = nullptr;
    Edge _edge;
    Mode _mode;
    int _stackLevel = 0;
    int _stackDepth = 0;
    FiberMapRulerModel _model;
    FiberMapRulerStyle _style;
    QFont _font;
};
