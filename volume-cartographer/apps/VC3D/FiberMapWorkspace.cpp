#include "FiberMapWorkspace.hpp"

#include "FiberMapRuler.hpp"
#include "LineAnnotationController.hpp"
#include "LineAnnotationGeneratedViews.hpp"

#include "vc/core/util/Logging.hpp"

#include <QAbstractButton>
#include <QAction>
#include <QColor>
#include <QDockWidget>
#include <QElapsedTimer>
#include <QEvent>
#include <QFont>
#include <QFontMetricsF>
#include <QGraphicsItem>
#include <QGraphicsLineItem>
#include <QGraphicsPathItem>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QGuiApplication>
#include <QHeaderView>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QHelpEvent>
#include <QPainterPath>
#include <QPalette>
#include <QPen>
#include <QPushButton>
#include <QScopeGuard>
#include <QScrollBar>
#include <QtConcurrent/QtConcurrent>
#include <QStyleOptionGraphicsItem>
#include <QTimer>
#include <QToolBar>
#include <QToolTip>
#include <QTransform>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVariant>
#include <QWheelEvent>
#include <QWindow>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <utility>

namespace
{

// Everything about the map that depends on the application theme. The dark row
// is the review script's own dark theme (fiber_network_unroll.py THEME["dark"]);
// the light row takes the script's light surface/ink/winding and pairs them with
// H/V hues of the same families darkened enough to read on white.
struct FiberMapPalette {
    QColor surface;
    QColor ink;
    QColor inkSoft;
    QColor horizontal;
    QColor vertical;
    QColor winding;
    QColor chipHorizontal;
    QColor chipVertical;
    QColor chipInk;
};

const FiberMapPalette kDarkPalette{
    .surface = QColor(QStringLiteral("#1a1a19")),
    .ink = QColor(QStringLiteral("#ffffff")),
    .inkSoft = QColor(QStringLiteral("#c3c2b7")),
    .horizontal = QColor(QStringLiteral("#3bc3d7")),
    .vertical = QColor(QStringLiteral("#48c964")),
    .winding = QColor(QStringLiteral("#9a978c")),
    .chipHorizontal = QColor(QStringLiteral("#aee7f0")),
    .chipVertical = QColor(QStringLiteral("#b8ecc4")),
    .chipInk = QColor(QStringLiteral("#0b0b0b")),
};

const FiberMapPalette kLightPalette{
    .surface = QColor(QStringLiteral("#fcfcfb")),
    .ink = QColor(QStringLiteral("#0b0b0b")),
    .inkSoft = QColor(QStringLiteral("#52514e")),
    .horizontal = QColor(QStringLiteral("#0f96ab")),
    .vertical = QColor(QStringLiteral("#2d9e4d")),
    .winding = QColor(QStringLiteral("#8e8b80")),
    // Deeper pastels than the dark theme's, so the chips still separate from a
    // white ground while carrying the same near-black text.
    .chipHorizontal = QColor(QStringLiteral("#bfe9f1")),
    .chipVertical = QColor(QStringLiteral("#c8edd2")),
    .chipInk = QColor(QStringLiteral("#0b0b0b")),
};

// The theme in force right now. Every build reads this afresh rather than
// caching it, so a theme switch only has to rebuild the scene and the tree.
// Which way the application palette leans is the same test CWindow uses to
// decide whether to install its dark palette, and that installed palette is
// what the widgets here inherit.
const FiberMapPalette& activePalette()
{
    const QColor window = QGuiApplication::palette().color(QPalette::Window);
    return window.lightness() < 128 ? kDarkPalette : kLightPalette;
}

// Red for winding-suspect links, and the link palette below, are the same in
// either theme: they read against both grounds and, in the link case, mirror
// colours fixed by the line annotation.
const QColor kSuspect(QStringLiteral("#ff6b6b"));

// Link markers use the line annotation's branch-link palette verbatim, so a
// crossing reads the same here as it does in the slice and generated views:
// H/V links are violet, same-type (H-H, V-V) links orange, and both go pale
// blue / pale orange while they await review. These four rows mirror
// apps/VC3D/overlays/FiberOverlayController.cpp and
// apps/VC3D/LineAnnotationGeneratedViews.cpp and must stay in sync with them.
struct LinkPalette {
    QColor pen;
    QColor brush;
};
const LinkPalette kLinkCross{QColor(210, 95, 255, 245), QColor(210, 95, 255, 175)};
const LinkPalette kLinkCrossPending{QColor(80, 150, 255, 245), QColor(80, 150, 255, 175)};
const LinkPalette kLinkSameType{QColor(255, 140, 0, 245), QColor(255, 140, 0, 175)};
const LinkPalette kLinkSameTypePending{QColor(255, 190, 120, 245),
                                       QColor(255, 190, 120, 175)};

// The map draws a link as a dot on each of its two control points, which
// overlap when zoomed out; the fill is thinned so the two stacked compound
// to roughly the palette's 175 rather than to near-opaque.
constexpr int kLinkEndpointFillAlpha = 120;

// A control point tagged kollesis_termination is marked on every fiber,
// selected or not, as the line annotation's hollow yellow ring: an unlinked
// one as an unfilled ring, a linked one as its link endpoint dot with the
// yellow ring around the link fill.
QColor kollesisColor(int alpha)
{
    return vc3d::line_annotation::generatedKollesisTerminationColor(alpha);
}
constexpr qreal kKollesisRimWidthPx = 2.0;

// A link is same-type only when both fibers carry the same known H/V tag; an
// unknown tag on either end falls back to the cross-type colours.
const LinkPalette& linkPalette(char hvTagA, char hvTagB, bool pending)
{
    const bool sameType = hvTagA == hvTagB && hvTagA != '?';
    if (sameType) {
        return pending ? kLinkSameTypePending : kLinkSameType;
    }
    return pending ? kLinkCrossPending : kLinkCross;
}

constexpr qreal kTracedWidth = 2.2;
constexpr qreal kInterpolatedWidth = 1.4;
constexpr qreal kTracedHighlightWidth = 3.6;
constexpr qreal kInterpolatedHighlightWidth = 2.4;
// The selected fiber's linked network: a gentle semi-transparent glow behind
// each member's unchanged lines - visible next to the crowd, clearly
// subordinate to the selection itself.
constexpr qreal kNetworkGlowWidthPx = 20.0;
constexpr int kNetworkGlowAlpha = 70;
constexpr qreal kPanelZ = -3.0;
constexpr qreal kNetworkGlowZ = 1.5;
constexpr qreal kFiberZ = 2.0;
constexpr qreal kHighlightZ = 7.0;
// Error rings (dropped crossings, suspect-link endpoints) draw above
// everything, the selected fiber and its control dots included: a ring in a
// dense tangle is the one thing the map must not bury.
constexpr qreal kSuspectRingZ = kHighlightZ + 2.0;
// Dots (control points, link crossings, suspect-link rings) are drawn in scene
// units, so they grow with the zoom, but their on-screen radius is clamped
// from both sides: never smaller than kMin*Px, so they stay visible when a
// whole network is in view, and never larger than kMax*Px, so zooming in to
// inspect a fiber does not bury the 2 px line under a marker sized for print.
// The *BoundsCm value is the ceiling for the scene-space radius the pixel
// floor can demand when zoomed far out, and with it the painting bounds: the
// dots stop growing in scene units rather than outrun their bounding rect.
//
// The cm sizes are what the markers are meant to measure on a printed map;
// the scene is in voxels, so each is multiplied by sceneVxPerCm() at build
// time. Nothing below may reach the scene without that conversion.
constexpr qreal kControlDotRadiusCm = 0.06;
constexpr qreal kMinControlDotPx = 3.5;
constexpr qreal kMaxControlDotPx = 6.0;
constexpr qreal kControlDotBoundsCm = 0.5;
constexpr qreal kCrossingDotRadiusCm = 0.10;
constexpr qreal kMinCrossingDotPx = 5.2;
constexpr qreal kMaxCrossingDotPx = 8.0;
constexpr qreal kCrossingDotBoundsCm = 0.4;
constexpr qreal kSuspectRingRadiusCm = 0.08;
constexpr qreal kMinSuspectRingPx = 4.0;
constexpr qreal kMaxSuspectRingPx = 7.0;
constexpr qreal kSuspectRingBoundsCm = 0.6;
// Slack kept on either side of the content so a zoomed-in view can pan the outer
// panels away from the edge, in cm; it is the floor under the quarter-of-the-width
// margin, so it only decides maps narrower than 12 cm.
constexpr qreal kMinSceneMarginCm = 3.0;
// Label chips hide once a whole winding maps to fewer screen pixels than
// this: chips are ~40 px wide and ignore the view transform, so as windings
// compress the labels collide across them and bury the geometry instead of
// annotating it.
constexpr double kMinChipPixelsPerWinding = 180.0;
constexpr double kFiberHitTolerancePx = 14.0;
constexpr double kControlDotTolerancePx = 10.0;
constexpr int kClickSlopPx = 4;

// Stand-in voxel size for a package that cannot say how big its voxels are. It
// is the resolution of the open-data scrolls, so the common case is unaffected,
// and it decides two things only: the scale the map's cm-valued styling
// constants are converted at, and the physical intents handed to the layout as
// voxel lengths. No physical figure is ever displayed from it — see
// formatMapLength(), which reports voxels instead.
constexpr double kAssumedVoxelSizeUm = 2.4;
constexpr double kUmPerCm = 10000.0;

QColor tint(const QColor& color, const QColor& toward, double amount)
{
    const auto blend = [amount](int from, int to) {
        return static_cast<int>(std::lround(from + (to - from) * amount));
    };
    return QColor(blend(color.red(), toward.red()),
                  blend(color.green(), toward.green()),
                  blend(color.blue(), toward.blue()));
}

QColor fiberColor(char hvTag, const FiberMapPalette& theme)
{
    if (hvTag == 'H') {
        return theme.horizontal;
    }
    if (hvTag == 'V') {
        return theme.vertical;
    }
    return theme.inkSoft;
}

QPen cosmeticPen(const QColor& color, qreal width)
{
    QPen pen(color);
    pen.setWidthF(width);
    pen.setCosmetic(true);
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    return pen;
}

QPen interpolatedPen(const QColor& color, qreal width)
{
    QPen pen(color);
    pen.setWidthF(width);
    pen.setCosmetic(true);
    pen.setCapStyle(Qt::FlatCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setStyle(Qt::CustomDashLine);
    pen.setDashPattern({5.0, 2.2});
    return pen;
}

QPainterPath pathForRuns(const vc3d::fiber_map::PlacedFiber& fiber, bool traced)
{
    QPainterPath path;
    for (const vc3d::fiber_map::Run& run : fiber.runs) {
        if (run.traced != traced || run.points.size() < 2) {
            continue;
        }
        path.moveTo(run.points.front());
        for (std::size_t i = 1; i < run.points.size(); ++i) {
            path.lineTo(run.points[i]);
        }
    }
    return path;
}

double distanceToSegment(const QPointF& point, const QPointF& a, const QPointF& b)
{
    const double dx = b.x() - a.x();
    const double dy = b.y() - a.y();
    const double lengthSquared = dx * dx + dy * dy;
    double t = 0.0;
    if (lengthSquared > 0.0) {
        t = ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / lengthSquared;
        t = std::clamp(t, 0.0, 1.0);
    }
    const double ex = a.x() + t * dx - point.x();
    const double ey = a.y() + t * dy - point.y();
    return std::sqrt(ex * ex + ey * ey);
}

QRectF fiberBounds(const vc3d::fiber_map::PlacedFiber& fiber)
{
    // Accumulated by hand rather than through united(): a zero-size QRectF is null,
    // so seeding with one and testing isNull() never accumulated anything, and the
    // result was a degenerate rect at the last point -- which then failed its own
    // callers' isNull() check, so label chips were never placed and selecting a
    // fiber in the tree never centred the view.
    bool havePoint = false;
    double left = 0.0;
    double top = 0.0;
    double right = 0.0;
    double bottom = 0.0;
    for (const vc3d::fiber_map::Run& run : fiber.runs) {
        for (const QPointF& point : run.points) {
            if (!havePoint) {
                left = right = point.x();
                top = bottom = point.y();
                havePoint = true;
                continue;
            }
            left = std::min(left, point.x());
            right = std::max(right, point.x());
            top = std::min(top, point.y());
            bottom = std::max(bottom, point.y());
        }
    }
    if (!havePoint) {
        return {};
    }
    return QRectF(QPointF(left, top), QPointF(right, bottom));
}

// Text pinned to a scene position but drawn at a fixed pixel size, offset by
// whole device pixels.
void pinText(QGraphicsSimpleTextItem* item, const QPointF& scenePosition,
             qreal offsetX, qreal offsetY, bool centered)
{
    item->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    item->setPos(scenePosition);
    const qreal dx = centered ? offsetX - 0.5 * item->boundingRect().width() : offsetX;
    item->setTransform(QTransform::fromTranslate(dx, offsetY));
}

// The chips are the only scene items a click resolves by hit test, so they are
// recognised among the items under the cursor by an item type of their own.
constexpr int kChipItemType = QGraphicsItem::UserType + 1;

// Rounded label chip drawn at a fixed pixel size; the fiber id travels on
// data(0) so a click on the chip resolves to its fiber.
class FiberLabelChip : public QGraphicsItem
{
public:
    FiberLabelChip(const QString& text, const QColor& fill, const QColor& ink,
                   const QFont& font)
        : _text(text)
        , _fill(fill)
        , _ink(ink)
        , _font(font)
    {
        const QFontMetricsF metrics(_font);
        const qreal width = metrics.horizontalAdvance(_text) + 8.0;
        const qreal height = metrics.height() + 4.0;
        _rect = QRectF(0.0, -0.5 * height, width, height);
        setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
    }

    int type() const override { return kChipItemType; }

    QRectF boundingRect() const override { return _rect.adjusted(-1.0, -1.0, 1.0, 1.0); }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override
    {
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setPen(Qt::NoPen);
        painter->setBrush(_fill);
        painter->drawRoundedRect(_rect, 3.0, 3.0);
        painter->setFont(_font);
        painter->setPen(_ink);
        painter->drawText(_rect, Qt::AlignCenter, _text);
    }

    qreal width() const { return _rect.width(); }

private:
    QString _text;
    QColor _fill;
    QColor _ink;
    QFont _font;
    QRectF _rect;
};

// The radius a ScaledDot is drawn with at a given view scale (scene units per
// screen pixel is 1/scale): the scene radius, held between the pixel floor
// and the pixel ceiling, and never past the scene bound the bounding rect was
// sized for. Shared with the workspace's control-dot hit test so the grab
// area and the visible dot agree at every zoom.
qreal scaledDotRadius(qreal radius, qreal minPixels, qreal maxPixels, qreal maxRadius,
                      qreal scale)
{
    if (!(scale > 0.0)) {
        return std::min(radius, maxRadius);
    }
    const qreal floorRadius = minPixels / scale;
    const qreal ceilingRadius = std::max(floorRadius, maxPixels / scale);
    return std::min(std::clamp(radius, floorRadius, ceilingRadius), maxRadius);
}

// Round marker of the map: the highlighted fiber's control points, the link
// crossings and the suspect-link rings. Unlike the pinned chips it lives in
// scene space, so zooming in makes it a bigger target, but only up to a pixel
// ceiling: past that it stays a small marker on the line rather than covering
// it, and when zoomed out a pixel floor keeps it visible.
class ScaledDot : public QGraphicsItem
{
public:
    // radius and maxRadius are scene units (voxels); minPixels and maxPixels
    // are on screen, and the level-of-detail factor converts between the two,
    // so this needs to know nothing about what a scene unit measures.
    ScaledDot(const QBrush& fill, const QPen& outline, qreal radius,
              qreal minPixels, qreal maxPixels, qreal maxRadius)
        : _fill(fill)
        , _outline(outline)
        , _radius(radius)
        , _minPixels(minPixels)
        , _maxPixels(maxPixels)
        , _maxRadius(maxRadius)
    {
    }

    QRectF boundingRect() const override
    {
        return QRectF(-_maxRadius, -_maxRadius, 2.0 * _maxRadius, 2.0 * _maxRadius);
    }

    // Hit testing stays tight to the scene-space radius; the ctrl+right-click
    // search in the workspace covers the pixel-clamped part.
    QPainterPath shape() const override
    {
        QPainterPath path;
        path.addEllipse(QPointF(0.0, 0.0), _radius, _radius);
        return path;
    }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override
    {
        const qreal lod =
            QStyleOptionGraphicsItem::levelOfDetailFromTransform(painter->worldTransform());
        qreal radius = scaledDotRadius(_radius, _minPixels, _maxPixels, _maxRadius, lod);
        // The outline is cosmetic, so at the scene-radius ceiling half its
        // width would fall outside boundingRect(); the fill gives way to it.
        if (lod > 0.0) {
            const qreal strokeHalf = 0.5 * std::max<qreal>(_outline.widthF(), 1.0) / lod;
            radius = std::max<qreal>(0.0, std::min(radius, _maxRadius - strokeHalf));
        }
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->setPen(_outline);
        painter->setBrush(_fill);
        painter->drawEllipse(QPointF(0.0, 0.0), radius, radius);
    }

private:
    QBrush _fill;
    QPen _outline;
    qreal _radius = 0.0;
    qreal _minPixels = 0.0;
    qreal _maxPixels = 0.0;
    qreal _maxRadius = 0.0;
};

// Appends the package's umbilicus state to a pre-rebuild status line. Unrolling
// is impossible without one, so the workspace says which file it would use (or
// that there is none, and how to attach one) before the user rebuilds to find
// out. Nothing is appended when no package is loaded.
QString withUmbilicusStatus(const QString& status, LineAnnotationController* controller)
{
    if (!controller) {
        return status;
    }
    const LineAnnotationController::UmbilicusStatus umbilicus =
        controller->umbilicusStatus();
    if (umbilicus.available) {
        return status + QObject::tr(" · umbilicus: %1").arg(umbilicus.text);
    }
    if (umbilicus.text.isEmpty()) {
        return status;
    }
    return status + QObject::tr(" · %1 — File > Attach Umbilicus…").arg(umbilicus.text);
}

} // namespace

FiberMapView::FiberMapView(QWidget* parent)
    : QGraphicsView(parent)
{
    // Panning is done by hand (right-drag, as in the volume viewers), so no drag
    // mode and no hand cursors: the pointer stays an arrow throughout.
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    setFrameShape(QFrame::NoFrame);
    // The axes are painted in viewport coordinates over the scene. The
    // default minimal update mode scrolls the viewport pixels on a pan and
    // repaints only the exposed strips, which drags stale copies of the axes
    // along with the map; a full repaint per pan keeps them in place.
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setCursor(Qt::ArrowCursor);
    // The right button drives the pan, so the platform must not turn it into a
    // context-menu event that would reach the surrounding QMainWindow.
    setContextMenuPolicy(Qt::PreventContextMenu);

    _rulers.push_back(std::make_unique<FiberMapRuler>(
        this, FiberMapRuler::Edge::Top, FiberMapRuler::Mode::Windings));
    _rulers.push_back(std::make_unique<FiberMapRuler>(
        this, FiberMapRuler::Edge::Left, FiberMapRuler::Mode::Height));
    _rulers.push_back(std::make_unique<FiberMapRuler>(
        this, FiberMapRuler::Edge::Bottom, FiberMapRuler::Mode::SheetDistance));
    for (const auto& ruler : _rulers) {
        ruler->setFont(font());
    }
}

FiberMapView::~FiberMapView() = default;

void FiberMapView::setRulerModel(const FiberMapRulerModel& model)
{
    for (const auto& ruler : _rulers) {
        ruler->setModel(model);
    }
    viewport()->update();
}

void FiberMapView::setRulerStyle(const FiberMapRulerStyle& style)
{
    for (const auto& ruler : _rulers) {
        ruler->setStyle(style);
    }
    viewport()->update();
}

void FiberMapView::drawForeground(QPainter* painter, const QRectF& rect)
{
    QGraphicsView::drawForeground(painter, rect);
    if (!painter) {
        return;
    }
    // The axes are laid out in device pixels against the viewport, so the
    // scene transform comes off for the duration.
    painter->save();
    painter->setWorldMatrixEnabled(false);
    const QRect area = viewport()->rect();
    for (const auto& ruler : _rulers) {
        ruler->paint(*painter, area);
    }
    painter->restore();
}

bool FiberMapView::viewportEvent(QEvent* event)
{
    if (event && event->type() == QEvent::ToolTip) {
        auto* help = static_cast<QHelpEvent*>(event);
        const QRect area = viewport()->rect();
        for (const auto& ruler : _rulers) {
            const QRect band = ruler->bandRect(area);
            if (band.contains(help->pos())) {
                QToolTip::showText(help->globalPos(), ruler->toolTipText(), viewport(), band);
                event->accept();
                return true;
            }
        }
        QToolTip::hideText();
    }
    return QGraphicsView::viewportEvent(event);
}

void FiberMapView::wheelEvent(QWheelEvent* event)
{
    const double steps = event->angleDelta().y() / 120.0;
    if (steps == 0.0) {
        QGraphicsView::wheelEvent(event);
        return;
    }
    const double factor = std::pow(1.15, steps);
    scale(factor, factor);
    emit zoomed();
    event->accept();
}

void FiberMapView::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::RightButton) {
        // The control-point menu can only be told apart from a pan once the
        // button comes back up, so it waits for the release.
        _pressPosition = event->pos();
        _panPosition = event->pos();
        _panning = true;
        _panDragged = false;
        _menuPending = (event->modifiers() & Qt::ControlModifier) != 0;
        event->accept();
        return;
    }
    if (event->button() == Qt::LeftButton) {
        _pressed = true;
    }
    QGraphicsView::mousePressEvent(event);
}

void FiberMapView::mouseMoveEvent(QMouseEvent* event)
{
    if (_panning && (event->buttons() & Qt::RightButton) != 0) {
        const QPoint position = event->pos();
        const QPoint scroll = _panPosition - position;
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() + scroll.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() + scroll.y());
        _panPosition = position;
        if ((position - _pressPosition).manhattanLength() >= kClickSlopPx) {
            _panDragged = true;
        }
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void FiberMapView::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::RightButton) {
        const bool wantsMenu = _menuPending && !_panDragged;
        _panning = false;
        _panDragged = false;
        _menuPending = false;
        if (wantsMenu) {
            emit controlPointMenuRequested(mapToScene(event->pos()),
                                           event->globalPosition().toPoint());
        }
        event->accept();
        return;
    }
    const bool wasPressed = _pressed && event->button() == Qt::LeftButton;
    _pressed = false;
    QGraphicsView::mouseReleaseEvent(event);
    if (wasPressed) {
        emit clicked(mapToScene(event->pos()));
    }
}

FiberMapWorkspace::FiberMapWorkspace(LineAnnotationController* controller,
                                     QWidget* parent)
    : QMainWindow(parent)
    , _controller(controller)
{
    setObjectName(QStringLiteral("fiberMapWorkspace"));
    setWindowTitle(tr("Fiber Map"));

    // The background is set by rebuildScene, which is called at the end of this
    // constructor and again whenever the theme changes.
    _scene = new QGraphicsScene(this);
    _view = new FiberMapView(this);
    _view->setScene(_scene);
    setCentralWidget(_view);

    auto* toolBar = addToolBar(tr("Fiber Map"));
    toolBar->setObjectName(QStringLiteral("fiberMapToolBar"));
    toolBar->setMovable(false);
    _updateButton = new QPushButton(tr("Update"), toolBar);
    _updateButton->setToolTip(
        tr("Rebuild the map, reusing cached work for unchanged fibers.\n"
           "Identical result to Full rebuild, much faster."));
    toolBar->addWidget(_updateButton);
    _fullRebuildButton = new QPushButton(tr("Full rebuild"), toolBar);
    _fullRebuildButton->setToolTip(
        tr("Recompute everything from scratch. When an Update preceded it\n"
           "on unchanged inputs, also verify the memoized result.\n"
           "Use if the map ever looks wrong."));
    toolBar->addWidget(_fullRebuildButton);
    toolBar->addSeparator();
    _statusLabel =
        new QLabel(tr("press Update"), toolBar);
    toolBar->addWidget(_statusLabel);

    _tree = new QTreeWidget(this);
    _tree->setColumnCount(5);
    _tree->setHeaderLabels(
        {tr("Fiber"), tr("H/V"), tr("Winding"), tr("Anchor"), tr("Annotation")});
    _tree->setUniformRowHeights(true);
    _tree->setSelectionMode(QAbstractItemView::SingleSelection);
    // Everything but the annotation name takes only what it needs; the
    // annotation name gets the rest of the dock.
    for (int column = 0; column < 4; ++column) {
        _tree->header()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
    }
    _tree->header()->setStretchLastSection(true);
    _fiberDock = new QDockWidget(tr("Fibers"), this);
    _fiberDock->setObjectName(QStringLiteral("fiberMapFiberDock"));
    _fiberDock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    _fiberDock->setWidget(_tree);
    addDockWidget(Qt::LeftDockWidgetArea, _fiberDock);
    resizeDocks({_fiberDock}, {360}, Qt::Horizontal);

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
        connect(_fiberDock, &QDockWidget::topLevelChanged, this, releaseStaleMouseGrab);
        connect(_fiberDock, &QDockWidget::dockLocationChanged, this, releaseStaleMouseGrab);
    }

    connect(_updateButton, &QPushButton::clicked, this,
            [this]() { requestRebuild(false); });
    connect(_fullRebuildButton, &QPushButton::clicked, this,
            [this]() { requestRebuild(true); });
    connect(_view, &FiberMapView::clicked, this, &FiberMapWorkspace::handleSceneClick);
    connect(_view, &FiberMapView::zoomed, this,
            &FiberMapWorkspace::updateLabelChipVisibility);
    connect(_view, &FiberMapView::controlPointMenuRequested,
            this, &FiberMapWorkspace::handleControlPointMenu);
    _tree->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(_tree, &QTreeWidget::customContextMenuRequested,
            this, &FiberMapWorkspace::handleTreeContextMenu);
    connect(_tree, &QTreeWidget::currentItemChanged, this,
            [this](QTreeWidgetItem* current, QTreeWidgetItem*) {
                if (_syncingSelection || !current) {
                    return;
                }
                // Read off the item before anything can invalidate it: the
                // verdict below may call for clearLayout(), which clears the tree
                // and so deletes `current` while this emission is still being
                // delivered.
                const uint64_t fiberId = current->data(0, Qt::UserRole).toULongLong();
                if (fiberId == 0) {
                    return;
                }
                // The same gate the scene click and the control-point menu use;
                // acting on a map whose dependencies moved is showing a wrong
                // picture, not a late one. Evaluated here and applied on the next
                // turn of the event loop, because the destructive half cannot run
                // from inside the tree's own signal.
                const auto verdict = evaluateDependencies();
                if (verdict.action != StaleVerdict::Action::Fresh) {
                    // Re-evaluated inside the callback rather than carried into it:
                    // anything could change between the two, and applying a stale
                    // verdict could announce a new package's layout current.
                    QMetaObject::invokeMethod(
                        this, [this]() { refreshStaleState(); },
                        Qt::QueuedConnection);
                    return;
                }
                setHighlightedFiber(fiberId);
                const auto entry = _entries.constFind(fiberId);
                if (entry != _entries.constEnd()) {
                    const QRectF bounds = fiberBounds(entry->fiber);
                    if (!bounds.isNull()) {
                        _view->centerOn(bounds.center());
                    }
                }
            });

    // Nothing is connected to the controller or to CState on purpose. A workspace
    // that may never be opened must cost annotation work nothing — in particular
    // no filesystem work from anyone else's change handlers — and a slot here
    // would have to either do the work or defer it anyway.
    //
    // Instead the controller keeps counters that are cheap to bump, and this
    // compares them at the moments it matters: on show, on rebuild, before
    // acting on a click or a fiber-list selection — and, while the tab is
    // visible, on a light poll, so the automatic update notices changes even
    // when the user is not touching the map. A hidden tab costs nothing: the
    // poll stops with hideEvent and showEvent's refresh covers the gap.
    _stalePollTimer = new QTimer(this);
    _stalePollTimer->setInterval(1000);
    connect(_stalePollTimer, &QTimer::timeout, this, [this]() {
        if (_rebuildQueue.state() !=
                vc3d::fiber_map::FiberMapRebuildQueue::State::Idle ||
            !_layoutBuilt) {
            return;
        }
        refreshStaleState();
    });

    // The rebuild worker: a private one-thread pool (no starvation from the
    // global pool's other users, bounded teardown) and a watcher that hands
    // the finished job back on the GUI thread. The watcher is parented, so
    // its connection dies with the workspace.
    _rebuildPool.setMaxThreadCount(1);
    _rebuildWatcher =
        new QFutureWatcher<std::shared_ptr<RebuildJobResult>>(this);
    connect(_rebuildWatcher,
            &QFutureWatcher<std::shared_ptr<RebuildJobResult>>::finished, this,
            [this]() {
                // result() rethrows an exception the future stored if one
                // escaped the callable itself (runRebuildJob() catches
                // everything, so this is the transport layer only). Letting
                // it escape a slot is unsupported and would strand the queue
                // in Running with the buttons disabled.
                try {
                    applyRebuild(_rebuildWatcher->result());
                } catch (...) {
                    Logger()->error(
                        "Fiber map: rebuild result transport failed");
                    markStale(tr("rebuild failed — press Update"));
                    _rebuildQueue.beginApply();
                    finishRebuild();
                }
            });
    _progressMarquee = new QTimer(this);
    _progressMarquee->setInterval(30);
    connect(_progressMarquee, &QTimer::timeout, this,
            [this]() { tickRebuildProgress(); });

    rebuildScene(tr("press Update"));
}

FiberMapWorkspace::~FiberMapWorkspace()
{
    // No new starts, pending dropped, and any in-flight publication refused
    // by the epoch; the wait is only for the private pool's clean teardown -
    // the worker owns its own data. The wait is bounded by one solve (a few
    // seconds at worst): buildGlobalLayout() has no cancellation point, and
    // the alternative - detaching the pool - would trade a bounded pause on
    // close for an unowned thread outliving the application's teardown.
    _rebuildQueue.shutdown();
    if (_stalePollTimer) {
        _stalePollTimer->stop();
    }
    if (_progressMarquee) {
        _progressMarquee->stop();
    }
    if (_rebuildWatcher) {
        _rebuildWatcher->disconnect(this);
    }
    _rebuildPool.waitForDone();
}

double FiberMapWorkspace::sceneVxPerCm() const
{
    return kUmPerCm / _voxelSizeUm.value_or(kAssumedVoxelSizeUm);
}

QString FiberMapWorkspace::formatMapLength(double valueVx) const
{
    if (_voxelSizeUm) {
        return tr("%1 cm").arg(valueVx * *_voxelSizeUm / kUmPerCm, 0, 'f', 2);
    }
    // The layout works in voxels, so the voxel count is exactly what it computed;
    // only the trip to centimetres needs a voxel size, and there is none.
    return tr("%1 vx").arg(std::llround(valueVx));
}

QString FiberMapWorkspace::withCachedUmbilicusStatus(const QString& status)
{
    const QString fingerprint =
        _controller ? _controller->umbilicusFingerprint() : QString();
    if (!_umbilicusStatusValid || fingerprint != _umbilicusStatusFingerprint) {
        _umbilicusStatusText = withUmbilicusStatus(QString(), _controller);
        _umbilicusStatusFingerprint = fingerprint;
        _umbilicusStatusValid = true;
    }
    return status + _umbilicusStatusText;
}

void FiberMapWorkspace::showStale(const QString& reason)
{
    _staleReason = reason;
    if (_statusLabel) {
        _statusLabel->setStyleSheet({});
        _statusLabel->setText(withCachedUmbilicusStatus(reason));
    }
}

void FiberMapWorkspace::markStale(const QString& reason)
{
    // The latching form, for staleness asserted rather than derived — the
    // invariant-violation defenses. The latched wording is kept in its own
    // field: a higher-priority derived reason (a voxel-size change, say) may
    // be displayed over it and later revert, and the latch must resurface
    // with its original wording rather than whatever the label last said.
    // Nothing in the dependency sets can prove it wrong, so it survives until
    // the layout it describes is rebuilt or cleared. Derived staleness goes
    // through applyStaleVerdict() instead and clears itself when its cause
    // reverts.
    _latchedReason = reason;
    showStale(reason);
}

void FiberMapWorkspace::clearLayout(const QString& reason)
{
    // Emptying the layout first is what makes rebuildScene() draw the reason
    // instead of geometry; it also owns tearing down the items, the entries and
    // the highlight, so none of that is repeated here.
    _layout = {};
    _layoutGeneration = 0;
    _layoutFrame = {};
    _layoutUmbilicusFingerprint.clear();
    _layoutPackageGeneration = 0;
    _layoutUmbilicusGeneration = 0;
    _layoutBuilt = false;
    _layoutCache.clear();
    _haveLastDigests = false;
    // An in-flight build started in a world this clear just removed; the
    // epoch bump refuses its publication.
    _rebuildQueue.invalidate();
    _voxelSizeUm.reset();
    _scrollZMaxVx = 0.0;
    // A fresh fit belongs to the next layout, which is not this one's frame.
    _viewFitted = false;
    // Whatever was cached about the umbilicus described the old package.
    _umbilicusStatusValid = false;
    // Nothing is built any more, so nothing is stale: every check early-outs
    // until the next rebuild. The reason becomes the resting status instead,
    // so it also survives a later fresh verdict re-applying that status. The
    // latch goes with the layout it described.
    _staleReason.clear();
    _latchedReason.clear();
    if (_tree) {
        _tree->clear();
    }
    rebuildScene(reason);
    _restingReason = reason;
    _freshStatus = withCachedUmbilicusStatus(reason);
    _freshStatusStyle.clear();
    if (_statusLabel) {
        _statusLabel->setStyleSheet(_freshStatusStyle);
        _statusLabel->setText(_freshStatus);
    }
}

vc3d::fiber_map::FiberMapDependencies
FiberMapWorkspace::currentDependencies() const
{
    vc3d::fiber_map::FiberMapDependencies deps;
    if (!_controller) {
        return deps;
    }
    deps.fiberGeneration = _controller->fiberDataGeneration();
    deps.packageGeneration = _controller->packageGeneration();
    deps.umbilicusGeneration = _controller->umbilicusGeneration();
    deps.umbilicusFingerprint = _controller->umbilicusFingerprint();
    deps.frame = _controller->annotationFrame();
    return deps;
}

vc3d::fiber_map::FiberMapDependencies
FiberMapWorkspace::layoutDependencies() const
{
    vc3d::fiber_map::FiberMapDependencies deps;
    deps.fiberGeneration = _layoutGeneration;
    deps.packageGeneration = _layoutPackageGeneration;
    deps.umbilicusGeneration = _layoutUmbilicusGeneration;
    deps.umbilicusFingerprint = _layoutUmbilicusFingerprint;
    deps.frame = _layoutFrame;
    return deps;
}

vc3d::fiber_map::StaleVerdict FiberMapWorkspace::evaluateDependencies() const
{
    if (!_controller) {
        return {};
    }
    return vc3d::fiber_map::staleVerdictFor(
        layoutDependencies(),
        currentDependencies(),
        _layoutBuilt,
        _latchedReason);
}

bool FiberMapWorkspace::applyStaleVerdict(const StaleVerdict& verdict)
{
    switch (verdict.action) {
    case StaleVerdict::Action::ClearLayout:
        clearLayout(verdict.reason);
        return true;
    case StaleVerdict::Action::MarkStale:
        // Unconditionally, not only when the reason text changed: the line also
        // carries the cached umbilicus suffix, and a fingerprint change under an
        // unchanged higher-priority reason must still refresh what that suffix
        // names. showStale() is idempotent when nothing moved.
        if ((verdict.cause == StaleVerdict::Cause::Fibers ||
             verdict.cause == StaleVerdict::Cause::Umbilicus) &&
            isVisible()) {
            scheduleAutoUpdate();
            // The banner reflects that no user action is needed: the update
            // is already on its way.
            QString reason = verdict.reason;
            reason.replace(tr("press Update"), tr("updating…"));
            showStale(reason);
        } else {
            showStale(verdict.reason);
        }
        return true;
    case StaleVerdict::Action::Fresh:
        // Derived staleness whose cause reverted — a setting moved back, one
        // scan's volume switched away and back. A latched reason cannot land
        // here: evaluateDependencies() feeds _latchedReason in and the verdict
        // stays MarkStale until a rebuild or clear.
        if (!_staleReason.isEmpty() && _layoutBuilt) {
            _staleReason.clear();
            if (_statusLabel) {
                _statusLabel->setStyleSheet(_freshStatusStyle);
                _statusLabel->setText(_freshStatus);
            }
        }
        break;
    }
    return false;
}

bool FiberMapWorkspace::refreshStaleState()
{
    return applyStaleVerdict(evaluateDependencies());
}

namespace
{
constexpr const char* kRebuildProgressOverlayName = "fiberMapRebuildProgress";
} // namespace

void FiberMapWorkspace::startRebuildProgress(QPushButton* button)
{
    _progressButton = button;
    _progressPhase = 0;
    if (_progressMarquee) {
        _progressMarquee->start();
    }
    tickRebuildProgress();
}

void FiberMapWorkspace::tickRebuildProgress()
{
    QPushButton* button = _progressButton;
    if (button == nullptr) {
        return;
    }
    auto* overlay =
        button->findChild<QWidget*>(QLatin1String(kRebuildProgressOverlayName),
                                    Qt::FindDirectChildrenOnly);
    if (overlay == nullptr) {
        overlay = new QWidget(button);
        overlay->setObjectName(QLatin1String(kRebuildProgressOverlayName));
        // Purely decorative: never intercept the click it reports on.
        overlay->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        overlay->setStyleSheet(QStringLiteral(
            "background-color: rgba(80, 150, 255, 110); border-radius: 2px;"));
    }
    // A marquee: a bar one third of the button wide, sweeping left to right
    // and wrapping. The event loop is live while the worker runs, so this is
    // a real animation - no forced synchronous paints needed any more.
    constexpr qint64 kSweepMs = 1100;
    const double phase =
        static_cast<double>(_progressPhase % kSweepMs) / kSweepMs;
    _progressPhase += _progressMarquee ? _progressMarquee->interval() : 30;
    const int barWidth = std::max(8, button->width() / 3);
    const int travel = button->width() + barWidth;
    const int x = static_cast<int>(std::lround(phase * travel)) - barWidth;
    overlay->setGeometry(x, 0, barWidth, button->height());
    overlay->show();
    overlay->raise();
}

void FiberMapWorkspace::clearRebuildProgress()
{
    if (_progressMarquee) {
        _progressMarquee->stop();
    }
    _progressButton = nullptr;
    for (QPushButton* button : {_updateButton, _fullRebuildButton}) {
        if (button == nullptr) {
            continue;
        }
        if (auto* overlay = button->findChild<QWidget*>(
                QLatin1String(kRebuildProgressOverlayName),
                Qt::FindDirectChildrenOnly)) {
            overlay->hide();
        }
        button->update();
    }
}

void FiberMapWorkspace::scheduleAutoUpdate()
{
    if (!isVisible()) {
        return;
    }
    // A build in flight: fold the request into the pending slot directly -
    // the running build's epilogue dispatches it, and a timer here could
    // only race that dispatch.
    if (_rebuildQueue.state() !=
        vc3d::fiber_map::FiberMapRebuildQueue::State::Idle) {
        (void)_rebuildQueue.request(false);
        return;
    }
    if (_autoUpdateScheduled) {
        return;
    }
    _autoUpdateScheduled = true;
    // Queued and debounced: the gates run inside click and selection
    // handlers whose scene items a publication would replace, and the
    // verdict is re-evaluated when the shot fires - anything can change in
    // between, including the staleness resolving itself.
    QTimer::singleShot(150, this, [this]() {
        _autoUpdateScheduled = false;
        if (!isVisible() || !_layoutBuilt) {
            return;
        }
        const StaleVerdict verdict = evaluateDependencies();
        if (verdict.action == StaleVerdict::Action::MarkStale &&
            (verdict.cause == StaleVerdict::Cause::Fibers ||
             verdict.cause == StaleVerdict::Cause::Umbilicus)) {
            // requestRebuild coalesces if a build started in the meantime.
            requestRebuild(false);
        }
    });
}

// Everything a rebuild consumes and produces, owned by the job so the two
// threads share no mutable state: the worker gets the snapshot, params, and
// the memoization cache (moved out of the workspace for the flight); the
// apply step takes the products back only after validating that the world
// the job started in still exists.
struct FiberMapWorkspace::RebuildJobResult {
    bool fullRebuild = false;
    uint64_t epoch = 0;
    // The world as of the start, for apply-time validation.
    QString preReadUmbilicusFingerprint;
    uint64_t builtPackageGeneration = 0;
    uint64_t builtUmbilicusGeneration = 0;
    // Inputs (snapshot fibers are consumed by the worker's conversion).
    LineAnnotationController::FiberMapSnapshot snapshot;
    vc3d::fiber_map::GlobalLayoutParams params;
    bool hadFibers = false;
    bool hadUmbilicus = false;
    // The workspace's memoization cache, exclusive to the job in flight.
    vc3d::fiber_map::GlobalLayoutCache cache;
    // Products.
    vc3d::fiber_map::GlobalResult layout;
    vc3d::fiber_map::ContentDigest inputsDigest;
    vc3d::fiber_map::ContentDigest outputDigest;
    vc3d::fiber_map::GlobalLayoutCache::Stats stats;
    QString error;
    qint64 snapshotMs = 0;
    qint64 convertMs = 0;
    qint64 layoutMs = 0;
};

namespace
{

// The worker: conversion, input digest, layout, output digest - everything
// that does not need the GUI thread. Exceptions become the job's error;
// nothing escapes into Qt.
void runRebuildJob(const std::shared_ptr<FiberMapWorkspace::RebuildJobResult>& job)
{
    try {
        const auto convertBegin = std::chrono::steady_clock::now();
        std::vector<vc3d::fiber_map::InputFiber> inputs;
        inputs.reserve(job->snapshot.fibers.size());
        for (auto& fiber : job->snapshot.fibers) {
            vc3d::fiber_map::InputFiber input;
            input.id = fiber.id;
            input.fileName = fiber.fileName;
            input.label = fiber.label;
            input.hvTag = fiber.hvTag;
            input.controlPoints = std::move(fiber.controlPoints);
            input.linePoints = std::move(fiber.linePoints);
            input.tracedSegments = std::move(fiber.tracedSegments);
            input.kollesisTerminations = std::move(fiber.kollesisTerminations);
            input.links.reserve(fiber.links.size());
            for (const auto& link : fiber.links) {
                input.links.push_back(
                    vc3d::fiber_map::InputLink{link.controlPointIndex,
                                               link.branchFiberId,
                                               link.branchControlPointIndex,
                                               link.pending});
            }
            inputs.push_back(std::move(input));
        }
        job->inputsDigest = vc3d::fiber_map::digestGlobalInputs(
            inputs, job->snapshot.umbilicusCenters, job->params);
        const auto layoutBegin = std::chrono::steady_clock::now();
        job->convertMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             layoutBegin - convertBegin)
                             .count();
        job->layout = vc3d::fiber_map::buildGlobalLayout(
            inputs, job->snapshot.umbilicusCenters, job->params, &job->cache);
        job->layoutMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - layoutBegin)
                            .count();
        job->outputDigest = vc3d::fiber_map::digestGlobalResult(job->layout);
        job->stats = job->cache.lastStats();
    } catch (const std::exception& ex) {
        job->error = QString::fromUtf8(ex.what());
    } catch (...) {
        job->error = QStringLiteral("unknown rebuild error");
    }
}

} // namespace

void FiberMapWorkspace::hideEvent(QHideEvent* event)
{
    QMainWindow::hideEvent(event);
    if (_stalePollTimer) {
        _stalePollTimer->stop();
    }
    // The marquee is a paint effect on a hidden button: 33 wakeups a second
    // for nobody. The build itself keeps running; showEvent() resumes the
    // animation if it is still in flight.
    if (_progressMarquee) {
        _progressMarquee->stop();
    }
}

void FiberMapWorkspace::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    if (_stalePollTimer) {
        _stalePollTimer->start();
    }
    // A build still in flight: resume the marquee hideEvent() paused
    // (_progressButton is non-null exactly while progress UI is active).
    if (_progressMarquee && _progressButton &&
        _rebuildQueue.state() !=
            vc3d::fiber_map::FiberMapRebuildQueue::State::Idle) {
        _progressMarquee->start();
    }
    // Becoming visible is the first of the three moments a stale layout has to
    // be caught; the others are a rebuild and any attempt to act on the map.
    if (refreshStaleState()) {
        return;
    }
    // Nothing built yet: keyed on nothing-built, not on an empty network list,
    // because a built-but-empty map has a real summary as its resting status,
    // which a re-show must not replace with a generic prompt. A cleared map's
    // resting status is its clear reason, stored in _freshStatus; only before
    // the first build is there nothing better to say than the prompt — and
    // that is where the umbilicus state gets looked up, so a package that will
    // not unroll says so before the user presses Rebuild.
    if (!_layoutBuilt && _statusLabel) {
        // Recomposed rather than replayed: _freshStatus froze its umbilicus
        // suffix when the layout was cleared, and the package may have
        // changed since (nothing is built, so no dependency comparison will
        // ever say so). The cache re-resolves when the fingerprint moved.
        _statusLabel->setStyleSheet({});
        _statusLabel->setText(withCachedUmbilicusStatus(
            _restingReason.isEmpty() ? tr("press Update")
                                     : _restingReason));
    }
}

void FiberMapWorkspace::requestRebuild(bool fullRebuild)
{
    if (!_controller) {
        return;
    }
    switch (_rebuildQueue.request(fullRebuild)) {
    case vc3d::fiber_map::FiberMapRebuildQueue::Request::Refused:
        return;
    case vc3d::fiber_map::FiberMapRebuildQueue::Request::Coalesced:
        // Folded into the pending slot; the running build's epilogue
        // dispatches it.
        return;
    case vc3d::fiber_map::FiberMapRebuildQueue::Request::Start:
        startRebuild(fullRebuild);
        return;
    }
}

void FiberMapWorkspace::startRebuild(bool fullRebuild)
{
    // The queue granted a Start: every exit either launches the worker or
    // runs the epilogue so the queue returns to Idle.
    // A package switch makes every cached artifact and the displayed scene
    // meaningless, and asynchrony exposes the interval - so it gets the FULL
    // clear (scene, tree, watermarks, cache, epoch), not the cache-only
    // reset that sufficed while rebuilds were synchronous. Grid changes stay
    // reversible MarkStale and never clear.
    if (_layoutBuilt) {
        const StaleVerdict verdict = vc3d::fiber_map::staleVerdictFor(
            layoutDependencies(), currentDependencies(),
            /*layoutBuilt=*/true, QString());
        if (verdict.action == StaleVerdict::Action::ClearLayout) {
            clearLayout(verdict.reason);
        }
    }
    if (fullRebuild || _memoizationDisabled) {
        _layoutCache.clear();
    }

    // One rollback for the whole launch: the queue is already Running, so
    // any throw between here and a successfully created future must run the
    // epilogue itself or every later request would merely coalesce forever.
    // The future creation stays inside the guarded region; once it exists,
    // the watcher's finished path owns cleanup.
    std::shared_ptr<RebuildJobResult> job;
    bool cacheMoved = false;
    QFuture<std::shared_ptr<RebuildJobResult>> future;
    try {
        job = std::make_shared<RebuildJobResult>();
        job->fullRebuild = fullRebuild;
        // Captured after the pre-check: a clear above advanced the epoch, and
        // this job publishes into the world as it stands now.
        job->epoch = _rebuildQueue.epoch();
        // Read before the snapshot: fiberMapSnapshot() parses the umbilicus,
        // so a rewrite during that parse that moves the file's size or mtime
        // — the token's contract; a same-size rewrite inside one timestamp
        // tick is beyond it — leaves the recorded token disagreeing with the
        // disk, and the publish-time refresh raises the banner.
        job->preReadUmbilicusFingerprint = _controller->umbilicusFingerprint();
        QElapsedTimer snapshotTimer;
        snapshotTimer.start();
        job->snapshot = _controller->fiberMapSnapshot();
        job->snapshotMs = snapshotTimer.elapsed();
        job->builtPackageGeneration = _controller->packageGeneration();
        job->builtUmbilicusGeneration = _controller->umbilicusGeneration();
        job->hadFibers = !job->snapshot.fibers.empty();
        job->hadUmbilicus = !job->snapshot.umbilicusCenters.empty();

        // No smoothing of the drawn fibers: with the markers pixel-capped,
        // a de-bumped curve read as a distortion of where the fibers really
        // run, and the winding-suspect rings (placed from the raw unrolled
        // geometry, then projected onto the drawn curve) sat off it by the
        // de-bumping. The resampling stays; it only interpolates the raw
        // polyline.
        job->params.smoothVx = 0.0;
        // The layout and solver are unit-free, so the physical intents behind
        // their tuning lengths are converted here — once the voxel size is
        // known, exactly as documented on GlobalLayoutParams and SolverParams.
        // Left alone when it is not, so the defaults (the same intents at
        // 2.4 µm) stand in and the map still lays out sensibly.
        if (job->snapshot.voxelSizeUm) {
            const double vxPerCm = kUmPerCm / *job->snapshot.voxelSizeUm;
            job->params.resampleStepVx = 0.025 * vxPerCm;  // 0.025 cm resample step
            job->params.minPadXVx = 2.2 * vxPerCm;         // 2.2 cm label pad across
            job->params.minPadYVx = 1.6 * vxPerCm;         // 1.6 cm label pad up
            job->params.solver.tieBandVx = 0.03 * vxPerCm;           // sheet-thickness scale
            job->params.solver.minUmbilicusRadiusVx = 0.1 * vxPerCm; // angular conditioning
            job->params.solver.zMergeVx = 0.2 * vxPerCm;             // crossing dedup span
            job->params.solver.neighborhoodZVx = 0.5 * vxPerCm;      // ordinal window
            job->params.solver.neighborhoodArcVx = 0.5 * vxPerCm;
        }

        // The cache travels WITH the job: the worker is its only toucher
        // while the build is in flight, by construction rather than by
        // discipline. Any GUI path that "clears the cache" mid-flight clears
        // this empty stand-in, and the epoch such paths also bump refuses
        // the job's publication.
        job->cache = std::move(_layoutCache);
        _layoutCache = vc3d::fiber_map::GlobalLayoutCache{};
        cacheMoved = true;

        _updateButton->setEnabled(false);
        _fullRebuildButton->setEnabled(false);
        startRebuildProgress(fullRebuild ? _fullRebuildButton : _updateButton);

        future = QtConcurrent::run(&_rebuildPool, [job]() {
            runRebuildJob(job);
            return job;
        });
    } catch (const std::exception& ex) {
        Logger()->error("Fiber map: rebuild start failed: {}", ex.what());
        if (job && cacheMoved) {
            // The worker never launched (the future is created last), so the
            // cache is safe to take back.
            _layoutCache = std::move(job->cache);
        }
        markStale(tr("rebuild failed — press Update"));
        _rebuildQueue.beginApply();
        finishRebuild();
        return;
    } catch (...) {
        Logger()->error("Fiber map: rebuild start failed (unknown)");
        if (job && cacheMoved) {
            _layoutCache = std::move(job->cache);
        }
        markStale(tr("rebuild failed — press Update"));
        _rebuildQueue.beginApply();
        finishRebuild();
        return;
    }
    // Effectively non-throwing: stores the future and wires signals. Kept
    // outside the rollback because once run() has returned the worker may
    // already be touching job->cache — taking it back would race.
    _rebuildWatcher->setFuture(future);
}

void FiberMapWorkspace::applyRebuild(const std::shared_ptr<RebuildJobResult>& job)
{
    // Publication is allowed only into the epoch the job started in, while
    // the queue still expects this build; checked before beginApply flips
    // the state.
    const bool expected = job != nullptr && _rebuildQueue.mayPublish(job->epoch);
    _rebuildQueue.beginApply();
    const auto epilogue = qScopeGuard([this]() { finishRebuild(); });
    // Declared after the epilogue so it runs first (guards unwind in reverse):
    // the watcher's future keeps the job's shared_ptr alive until the next
    // setFuture(), so "drop" on a discard or error means clearing the job's
    // heavy members here - otherwise a rejected 452-fiber layout, its
    // snapshot, and its cache would stay resident in an idle workspace.
    const auto release = qScopeGuard([&job]() {
        if (job) {
            job->snapshot = {};
            job->layout = {};
            job->cache = {};
        }
    });

    if (!expected || !_controller) {
        // A package switch, explicit clear, memoization-policy change, or
        // shutdown happened mid-flight: the job's world is gone. Its cache
        // is dropped with it - the events that bump the epoch are exactly
        // the ones that invalidate content - and the gates already show the
        // right state.
        return;
    }
    if (!job->error.isEmpty()) {
        Logger()->error("Fiber map: rebuild failed: {}",
                        job->error.toStdString());
        markStale(tr("rebuild failed — press Update"));
        return;
    }
    // A package switch no gate happened to observe mid-flight: the epoch
    // could not catch it, so it is validated directly. The result and its
    // cache, built from the old package's content, are discarded. With a
    // layout built, the verdict machinery produces the proper ClearLayout;
    // before a first build it has nothing to compare (verdict Fresh), yet
    // the status still names the old package's umbilicus - so the clear
    // that recomposes it runs directly.
    if (_controller->packageGeneration() != job->builtPackageGeneration) {
        if (!refreshStaleState()) {
            clearLayout(tr("project changed — press Update"));
        }
        return;
    }
    // The full frame, not just the grid: parameters were converted through
    // the snapshot's voxel size, so a same-grid volume with a different
    // recorded voxel size (the reversible VoxelSize stale cause) would make
    // this result stale on arrival - publishing it would destroy a layout
    // that is still valid for the volume the user switched back to. The
    // cache IS kept: its slots are content-keyed (params included), so
    // switching back to the built frame re-warms.
    if (!vc3d::annotation::sameAnnotationFrame(_controller->annotationFrame(),
                                               job->snapshot.frame)) {
        _layoutCache = std::move(job->cache);
        if (!refreshStaleState()) {
            showStale(tr(
                "viewing another volume's grid — switch back, or press Update"));
        }
        return;
    }
    // Fibers or the umbilicus changed mid-flight: publishing a result
    // already known stale would put a wrong picture on screen, so the
    // reviewer's rule is followed literally - discard and re-run. The job's
    // cache IS kept: its slots are content-keyed digests, exact across
    // edits, so the immediate re-run stays warm and cheap.
    if (_controller->fiberDataGeneration() != job->snapshot.generation ||
        _controller->umbilicusGeneration() != job->builtUmbilicusGeneration ||
        _controller->umbilicusFingerprint() != job->preReadUmbilicusFingerprint) {
        _layoutCache = std::move(job->cache);
        if (isVisible()) {
            (void)_rebuildQueue.request(job->fullRebuild);
            showStale(tr("changed during update — updating…"));
        } else {
            // The visible-only contract: edits made while the workspace is
            // hidden must not chain background rebuilds. The gates re-arm
            // the automatic update on the next show.
            showStale(tr("changed during update — press Update"));
        }
        return;
    }

    try {
        publishRebuild(*job);
    } catch (const std::exception& ex) {
        // A throw from scene or tree construction can leave the scene and
        // watermarks disagreeing; the latch refuses interaction until a
        // rebuild succeeds.
        Logger()->error("Fiber map: rebuild publication failed: {}", ex.what());
        markStale(tr("rebuild failed — press Update"));
    } catch (...) {
        Logger()->error("Fiber map: rebuild publication failed (unknown)");
        markStale(tr("rebuild failed — press Update"));
    }
}

void FiberMapWorkspace::publishRebuild(RebuildJobResult& job)
{
    QElapsedTimer publishTimer;
    publishTimer.start();
    // The job's cache and layout become the workspace's, and every
    // dependency watermark commits together.
    _layoutCache = std::move(job.cache);
    _layout = std::move(job.layout);
    _layoutUmbilicusFingerprint = job.preReadUmbilicusFingerprint;
    _layoutGeneration = job.snapshot.generation;
    _layoutFrame = job.snapshot.frame;
    _layoutPackageGeneration = job.builtPackageGeneration;
    _layoutUmbilicusGeneration = job.builtUmbilicusGeneration;
    _layoutBuilt = true;
    _staleReason.clear();
    _latchedReason.clear();
    _restingReason.clear();
    _voxelSizeUm = job.snapshot.voxelSizeUm;

    // Full rebuild doubles as the memoization check: when nothing the layout
    // consumes changed since the last memoized Update, the from-scratch
    // output must digest identically. Digest bookkeeping happens ONLY here,
    // inside a successful publication - discarded and failed builds leave it
    // untouched.
    QString verificationNote;
    if (job.fullRebuild) {
        if (_haveLastDigests && job.inputsDigest == _lastInputsDigest) {
            if (job.outputDigest == _lastOutputDigest) {
                verificationNote = tr(" · cache verified");
            } else {
                verificationNote = tr(" · CACHE MISMATCH — memoization disabled");
                _memoizationDisabled = true;
                _layoutCache.clear();
                _rebuildQueue.invalidate();
                Logger()->error(
                    "Fiber map: full rebuild output differs from the memoized "
                    "build on identical inputs; memoization disabled");
            }
        }
        _haveLastDigests = false;
    } else {
        const bool reusedSomething =
            job.stats.used &&
            (job.stats.fibersReused > 0 || job.stats.pairsReused > 0);
        if (reusedSomething && !_memoizationDisabled) {
            _lastInputsDigest = job.inputsDigest;
            _lastOutputDigest = job.outputDigest;
            _haveLastDigests = true;
        } else {
            _haveLastDigests = false;
        }
    }

    // Scene space is voxels and the slice count already is one, so the
    // scroll extent needs no voxel size at all.
    _scrollZMaxVx = job.snapshot.annotationZSlices > 0
        ? static_cast<double>(job.snapshot.annotationZSlices)
        : 0.0;

    QString emptyMessage;
    if (!job.hadFibers) {
        emptyMessage = tr("no fibers");
    } else if (!job.hadUmbilicus) {
        // The resolver's own words when it has any; they name the file it
        // rejected or the candidates it could not choose between.
        emptyMessage = job.snapshot.umbilicusMessage.isEmpty()
            ? tr("no umbilicus found — cannot unroll")
            : job.snapshot.umbilicusMessage;
        // Whatever the resolver's complaint was, the way out is the same.
        emptyMessage += QLatin1Char('\n');
        emptyMessage += tr("Attach one via File > Attach Umbilicus…");
    } else if (_layout.fibers.empty()) {
        emptyMessage = tr("no placeable fibers");
    }
    rebuildScene(emptyMessage);
    rebuildTree();
    const qint64 publishMs = publishTimer.elapsed();
    Logger()->info(
        "fiber map rebuild: GUI stalls snapshot {} ms + publish {} ms · "
        "worker convert {} ms · layout {} ms (prep {:.0f}, detect {:.0f}, "
        "solve {:.0f}, geometry {:.0f})",
        job.snapshotMs, publishMs, job.convertMs, job.layoutMs,
        _layout.prepMs, _layout.detectMs, _layout.solveMs, _layout.geometryMs);

    // Default the dock to a width that shows every column of the first real
    // tree; afterwards the width is the user's to manage.
    if (!_fiberDockSized && _fiberDock && _tree->topLevelItemCount() > 0) {
        int width = 2 * _tree->frameWidth() + _tree->indentation() +
                    _tree->verticalScrollBar()->sizeHint().width() + 12;
        for (int column = 0; column < _tree->columnCount(); ++column) {
            // The stretch on the last section re-expands it after this pass;
            // resizing first makes columnWidth() report the content width.
            _tree->resizeColumnToContents(column);
            width += _tree->columnWidth(column);
        }
        resizeDocks({_fiberDock}, {width}, Qt::Horizontal);
        _fiberDockSized = true;
    }

    // The errors lead: what the red marks add up to - every ring (dropped
    // crossings, each group conflict's rings) and every suspect link - so
    // one glance says whether the map is clean, before any tally of how it
    // was built.
    const int errorCount = static_cast<int>(_layout.suspectCrossings.size()) +
                           _layout.suspectLinkCount;
    QString status = errorCount > 0 ? tr("%1 errors").arg(errorCount) : tr("no errors");
    status += tr(" · %1 fibers · %2 windings · %3 islands · %4 suspect links")
                  .arg(_layout.fibers.size())
                  .arg(_layout.windings.size())
                  .arg(_layout.islandCount)
                  .arg(_layout.suspectLinkCount);
    if (!_layout.unplaced.empty()) {
        status += tr(" · %1 unplaceable").arg(_layout.unplaced.size());
    }
    if (_layout.unresolvedCount > 0) {
        status += tr(" · %1 unresolved").arg(_layout.unresolvedCount);
    }
    if (_layout.droppedCrossingCount > 0) {
        status += tr(" · %1 dropped crossings").arg(_layout.droppedCrossingCount);
    }
    if (_layout.declaredGroupCount > 0) {
        // A traversal group read together and still contradicted by the map:
        // one conflict, ringed at each place the pair met.
        status += tr(" · %1 group conflicts").arg(_layout.declaredGroupCount);
    }
    if (_layout.traversalGroupCount > 0) {
        status += tr(" · %1 grouped").arg(_layout.traversalGroupCount);
    }
    if (_layout.kollesisCrossingCount > 0) {
        status += tr(" · %1 kollesis").arg(_layout.kollesisCrossingCount);
        if (_layout.kollesisInferredCount > 0) {
            status += tr(" (%1 inferred)").arg(_layout.kollesisInferredCount);
        }
    }
    const qint64 totalMs =
        job.snapshotMs + job.convertMs + job.layoutMs + publishMs;
    if (job.stats.used && !job.fullRebuild) {
        status += tr(" · %1 ms, %2/%3 pairs reused")
                      .arg(totalMs)
                      .arg(job.stats.pairsReused)
                      .arg(job.stats.pairsReused + job.stats.pairsRecomputed);
    } else {
        status += tr(" · %1 ms").arg(totalMs);
    }
    status += verificationNote;
    if (_layout.gatedSegmentCount > 0 || _layout.tangentialCount > 0) {
        // Gate-hit tallies, not a geometry proportion (one segment can be
        // counted once per branch and translate it was tried against): a
        // nonzero value says the map may be underconstrained for reasons the
        // drawn fibers cannot show.
        status += tr(" · %1 solver gate hits")
                      .arg(_layout.gatedSegmentCount + _layout.tangentialCount);
    }
    if (!_voxelSizeUm) {
        // No physical figure on the map means anything, so say why once
        // rather than leave the voxel counts looking like an odd unit.
        status += tr(" · voxel size unknown — lengths in vx");
    }
    if (job.hadUmbilicus && !job.snapshot.umbilicusLabel.isEmpty()) {
        // The controller composes this: which grid the umbilicus indexes,
        // whether that came from the file's own metadata or from the z-span
        // guess, and any frame inconsistency it noticed on the way.
        status += QStringLiteral(" · ") + job.snapshot.umbilicusLabel;
    }
    _freshStatus = status;
    // Red and bold while anything is ringed; plain once the map is clean.
    _freshStatusStyle = errorCount > 0
        ? QStringLiteral("QLabel { color: %1; font-weight: bold; }").arg(kSuspect.name())
        : QString();
    _statusLabel->setStyleSheet(_freshStatusStyle);
    _statusLabel->setText(status);

    if (!_viewFitted && !_layout.fibers.empty()) {
        _view->fitInView(_contentRect, Qt::KeepAspectRatio);
        _viewFitted = true;
    }
    // fitInView changes the scale without a wheel event.
    updateLabelChipVisibility();

    // The one moment every dependency is re-examined against what this build
    // just recorded. An umbilicus file rewritten while the snapshot was
    // being read differs from the pre-read token recorded above, so the
    // banner goes up here — after the summary assignment, which must never
    // be what a stale map is left saying. If it schedules an automatic
    // update, that coalesces into the pending slot the epilogue dispatches.
    refreshStaleState();
}

void FiberMapWorkspace::finishRebuild()
{
    clearRebuildProgress();
    if (_updateButton) {
        _updateButton->setEnabled(true);
    }
    if (_fullRebuildButton) {
        _fullRebuildButton->setEnabled(true);
    }
    const auto pending = _rebuildQueue.finishApply();
    if (pending == vc3d::fiber_map::FiberMapRebuildQueue::Pending::None) {
        return;
    }
    const bool full =
        pending == vc3d::fiber_map::FiberMapRebuildQueue::Pending::Full;
    // A pending Update was armed by a gate that compared against the OLD
    // build's watermarks mid-flight; when the build that just published
    // already covers the change, the verdict here is Fresh and the update
    // would recompute a digest-identical map. A latched failure or genuine
    // staleness still dispatches, and a pending Full always does - it is
    // the user's explicit escape hatch.
    if (!full && _layoutBuilt &&
        evaluateDependencies().action ==
            vc3d::fiber_map::StaleVerdict::Action::Fresh) {
        return;
    }
    requestRebuild(full);
}

void FiberMapWorkspace::rebuildScene(const QString& emptyMessage)
{
    _entries.clear();
    clearControlPointDots();
    _highlightedFiber = 0;
    _networkEmphasized.clear();
    _labelChips.clear();
    _chipHideScale = 0.0;
    _scene->clear();

    // Kept so a theme change can rebuild the scene as it stands, without asking
    // the controller for a fresh snapshot.
    _emptyMessage = emptyMessage;

    const FiberMapPalette& theme = activePalette();
    _scene->setBackgroundBrush(theme.surface);

    // The axes read the layout through the view; an empty layout blanks them.
    // Their colours are the map's, so a theme switch re-styles them with the
    // same rebuild. The model is completed below once the extent is known.
    _view->setRulerStyle(FiberMapRulerStyle{theme.surface, theme.inkSoft, theme.winding});

    if (_layout.fibers.empty()) {
        _view->setRulerModel(FiberMapRulerModel{});
        auto* message = _scene->addSimpleText(emptyMessage);
        message->setBrush(theme.ink);
        _contentRect = message->boundingRect().adjusted(-40.0, -40.0, 40.0, 40.0);
        _scene->setSceneRect(_contentRect);
        return;
    }

    // Scene coordinates are (x, -y) in voxels: negating z once here keeps the
    // scroll axis reading upward without ever mirroring text.
    const double topY = -_layout.yMaxVx;
    const double bottomY = -_layout.yMinVx;
    const double sceneWidth = std::max(_layout.x1Vx - _layout.x0Vx, 1e-6);
    // The one conversion of this rebuild. Every scene-space size below that was
    // chosen as a physical length goes through it, and nothing else does.
    const double vxPerCm = sceneVxPerCm();
    const qreal crossingDotRadius = kCrossingDotRadiusCm * vxPerCm;
    const qreal crossingDotBounds = kCrossingDotBoundsCm * vxPerCm;
    const qreal suspectRingRadius = kSuspectRingRadiusCm * vxPerCm;
    const qreal suspectRingBounds = kSuspectRingBoundsCm * vxPerCm;
    QFont labelFont = font();
    labelFont.setPointSizeF(8.0);

    // The scroll floor and ceiling, so the map reads against the volume's own
    // z extent instead of floating on its own; the winding gridlines and the
    // ground span the same range. Without that extent the layout's own y
    // range has to do.
    const bool scrollExtentKnown = _scrollZMaxVx > 0.0;
    const double extentBottomY = scrollExtentKnown ? 0.0 : bottomY;
    const double extentTopY = scrollExtentKnown ? -_scrollZMaxVx : topY;
    const double sceneTopY = std::min(extentTopY, topY);
    const double sceneBottomY = std::max(extentBottomY, bottomY);

    // Each link's endpoints were registered as fibers before the links are
    // drawn, so the entries always cover both ends.
    const auto hvTagOf = [this](uint64_t fiberId) {
        const auto entry = _entries.constFind(fiberId);
        return entry == _entries.constEnd() ? '?' : entry->fiber.hvTag;
    };
    const auto isKollesisTermination = [this](uint64_t fiberId, int controlIndex) {
        const auto entry = _entries.constFind(fiberId);
        if (entry == _entries.constEnd() || controlIndex < 0) {
            return false;
        }
        const auto& flags = entry->fiber.kollesisTerminations;
        const auto index = static_cast<std::size_t>(controlIndex);
        return index < flags.size() && flags[index];
    };

    {
        FiberMapRulerModel rulerModel;
        rulerModel.hasLayout = true;
        rulerModel.windings = _layout.windings;
        rulerModel.sheet = vc3d::fiber_map::sheetModelOf(_layout);
        rulerModel.voxelSizeUm = _voxelSizeUm;
        rulerModel.extentTopSceneY = extentTopY;
        rulerModel.extentBottomSceneY = extentBottomY;
        rulerModel.extentLeftSceneX = _layout.x0Vx;
        rulerModel.extentRightSceneX = _layout.x1Vx;
        _view->setRulerModel(rulerModel);
    }

    // One ground for the whole map, spanning the scroll's own z extent.
    auto* ground = _scene->addRect(
        QRectF(QPointF(_layout.x0Vx, extentTopY), QPointF(_layout.x1Vx, extentBottomY)),
        QPen(Qt::NoPen), QBrush(tint(theme.surface, theme.ink, 0.045)));
    ground->setZValue(kPanelZ);

    // The winding grid, one line per integer winding. The numbers are the
    // top ruler's, which labels whatever is in view; the scene carries only
    // the gridlines.
    for (const vc3d::fiber_map::WindingMark& mark : _layout.windings) {
        auto* line = _scene->addLine(mark.xVx, extentTopY, mark.xVx, extentBottomY);
        QPen pen(theme.winding);
        pen.setWidthF(0.8);
        pen.setCosmetic(true);
        pen.setStyle(Qt::DotLine);
        line->setPen(pen);
        line->setZValue(0.0);
    }

    for (const vc3d::fiber_map::GlobalPlacedFiber& placed : _layout.fibers) {
        FiberEntry entry;
        entry.fiber = placed.fiber;
        entry.networkId = placed.meta.networkId;
        for (vc3d::fiber_map::Run& run : entry.fiber.runs) {
            for (QPointF& point : run.points) {
                point.setY(-point.y());
            }
        }
        for (QPointF& point : entry.fiber.controlPoints) {
            point.setY(-point.y());
        }

        // The path items only carry geometry: clicks resolve through
        // fiberAt()'s proximity search, never through the items themselves.
        const QColor color = fiberColor(entry.fiber.hvTag, theme);
        const QPainterPath tracedPath = pathForRuns(entry.fiber, true);
        if (!tracedPath.isEmpty()) {
            entry.tracedItem = _scene->addPath(tracedPath, cosmeticPen(color, kTracedWidth));
            entry.tracedItem->setZValue(kFiberZ);
        }
        const QPainterPath interpolatedPath = pathForRuns(entry.fiber, false);
        if (!interpolatedPath.isEmpty()) {
            entry.interpolatedItem = _scene->addPath(
                interpolatedPath,
                interpolatedPen(tint(color, theme.surface, 0.45), kInterpolatedWidth));
            entry.interpolatedItem->setZValue(kFiberZ);
        }

        // Label chip at whichever fiber end sits nearest a map edge
        // (H fibers: left vs right, V fibers: bottom vs top).
        const QRectF bounds = fiberBounds(entry.fiber);
        if (!bounds.isNull()) {
            QPointF anchor;
            qreal offsetX = 0.0;
            qreal offsetY = 0.0;
            bool anchorRight = false;
            const auto endpoint = [&entry](bool minimizeX, bool useX) {
                QPointF best;
                double bestValue = minimizeX ? std::numeric_limits<double>::infinity()
                                             : -std::numeric_limits<double>::infinity();
                for (const vc3d::fiber_map::Run& run : entry.fiber.runs) {
                    for (const QPointF& point : run.points) {
                        const double value = useX ? point.x() : point.y();
                        if (minimizeX ? value < bestValue : value > bestValue) {
                            bestValue = value;
                            best = point;
                        }
                    }
                }
                return best;
            };
            if (entry.fiber.hvTag == 'V') {
                // Scene y is inverted, so the smaller y is the top end.
                const QPointF top = endpoint(true, false);
                const QPointF low = endpoint(false, false);
                const bool atTop = (top.y() - topY) < (bottomY - low.y());
                anchor = atTop ? top : low;
                offsetX = 8.0;
                offsetY = atTop ? -10.0 : 10.0;
            } else {
                const QPointF left = endpoint(true, true);
                const QPointF right = endpoint(false, true);
                const bool atRight =
                    (_layout.x1Vx - right.x()) < (left.x() - _layout.x0Vx);
                anchor = atRight ? right : left;
                offsetX = atRight ? 10.0 : -10.0;
                anchorRight = !atRight;
            }
            auto* chip = new FiberLabelChip(
                entry.fiber.label,
                entry.fiber.hvTag == 'V' ? theme.chipVertical : theme.chipHorizontal,
                theme.chipInk, labelFont);
            chip->setData(0, QVariant::fromValue<qulonglong>(entry.fiber.id));
            chip->setZValue(6.0);
            chip->setPos(anchor);
            chip->setTransform(QTransform::fromTranslate(
                anchorRight ? offsetX - chip->width() : offsetX, offsetY));
            _scene->addItem(chip);
            _labelChips.push_back(chip);
        }

        const uint64_t fiberId = entry.fiber.id;
        _entries.insert(fiberId, std::move(entry));
    }

    for (const vc3d::fiber_map::PlacedLink& link : _layout.links) {
        const QPointF a(link.a.x(), -link.a.y());
        const QPointF b(link.b.x(), -link.b.y());
        const QPointF middle = 0.5 * (a + b);
        if (!link.suspect) {
            // A winding-suspect link keeps its own red treatment below;
            // everything else takes the annotation's branch-link colours.
            // A link joins two control points on two fibers, so it is drawn
            // as a dot on each, joined by a dotted line: zoomed out the two
            // dots overlap into one and the line is sub-pixel, zoomed in
            // they part and each dot stays on its own fiber - a single dot
            // at the midpoint sat on neither.
            const LinkPalette& palette =
                linkPalette(hvTagOf(link.fiberA), hvTagOf(link.fiberB), link.pending);
            QPen connectorPen = cosmeticPen(palette.pen, 1.0);
            connectorPen.setStyle(Qt::DotLine);
            auto* connector = _scene->addLine(QLineF(a, b));
            connector->setPen(connectorPen);
            connector->setZValue(3.9);
            // Two stacked fills read darker than one; this alpha compounds,
            // where the dots overlap, to about the palette's own.
            QColor fill = palette.brush;
            fill.setAlpha(kLinkEndpointFillAlpha);
            const std::array<bool, 2> tagged{
                isKollesisTermination(link.fiberA, link.cpA),
                isKollesisTermination(link.fiberB, link.cpB)};
            std::size_t endpointIndex = 0;
            for (const QPointF& endpoint : {a, b}) {
                const bool kollesis = tagged[endpointIndex++];
                auto* dot = new ScaledDot(QBrush(fill),
                                          kollesis
                                              ? cosmeticPen(kollesisColor(245), kKollesisRimWidthPx)
                                              : cosmeticPen(palette.pen, 1.0),
                                          crossingDotRadius,
                                          kMinCrossingDotPx, kMaxCrossingDotPx,
                                          crossingDotBounds);
                _scene->addItem(dot);
                dot->setPos(endpoint);
                // A tagged endpoint sits above its untagged twin where the two
                // overlap zoomed out, so the rim stays visible.
                dot->setZValue(kollesis ? 4.1 : 4.0);
            }
            continue;
        }
        QPen suspectPen(kSuspect);
        suspectPen.setWidthF(1.0);
        suspectPen.setCosmetic(true);
        suspectPen.setStyle(Qt::DashLine);
        auto* line = _scene->addLine(QLineF(a, b));
        line->setPen(suspectPen);
        line->setZValue(4.0);
        for (const QPointF& endpoint : {a, b}) {
            auto* ring = new ScaledDot(QBrush(Qt::NoBrush), cosmeticPen(kSuspect, 1.4),
                                       suspectRingRadius, kMinSuspectRingPx,
                                       kMaxSuspectRingPx, suspectRingBounds);
            _scene->addItem(ring);
            ring->setPos(endpoint);
            ring->setZValue(kSuspectRingZ);
        }
        auto* label = _scene->addSimpleText(
            tr("+%1 turn").arg(link.turnErr, 0, 'f', 1), labelFont);
        label->setBrush(kSuspect);
        pinText(label, middle, 0.0, -14.0, true);
        label->setZValue(5.0);
    }

    // Kollesis terminations on unlinked control points: a linked one was
    // already drawn above as its link endpoint dot with the pale yellow rim.
    {
        // Suspect links draw rings, not endpoint dots, so their tagged
        // endpoints still need the marker.
        std::set<std::pair<uint64_t, int>> linkedEndpoints;
        for (const vc3d::fiber_map::PlacedLink& link : _layout.links) {
            if (link.suspect) {
                continue;
            }
            linkedEndpoints.emplace(link.fiberA, link.cpA);
            linkedEndpoints.emplace(link.fiberB, link.cpB);
        }
        const QPen rim = cosmeticPen(kollesisColor(245), kKollesisRimWidthPx);
        const QBrush fill(Qt::NoBrush);
        for (auto entry = _entries.constBegin(); entry != _entries.constEnd(); ++entry) {
            const vc3d::fiber_map::PlacedFiber& fiber = entry->fiber;
            for (std::size_t i = 0;
                 i < fiber.kollesisTerminations.size() && i < fiber.controlPoints.size();
                 ++i) {
                if (!fiber.kollesisTerminations[i] ||
                    linkedEndpoints.count({fiber.id, static_cast<int>(i)}) != 0) {
                    continue;
                }
                auto* dot = new ScaledDot(fill, rim, crossingDotRadius,
                                          kMinCrossingDotPx, kMaxCrossingDotPx,
                                          crossingDotBounds);
                _scene->addItem(dot);
                dot->setPos(fiber.controlPoints[i]);
                dot->setZValue(4.1);
            }
        }
    }

    // Crossings the winding repair had to drop: contradicted evidence, marked
    // where the H fiber made the pass.
    for (const vc3d::fiber_map::CrossingMark& mark : _layout.suspectCrossings) {
        auto* ring = new ScaledDot(QBrush(Qt::NoBrush), cosmeticPen(kSuspect, 1.4),
                                   suspectRingRadius, kMinSuspectRingPx,
                                   kMaxSuspectRingPx, suspectRingBounds);
        _scene->addItem(ring);
        ring->setPos(QPointF(mark.posVx.x(), -mark.posVx.y()));
        ring->setZValue(kSuspectRingZ);
    }

    // The scroll extent, when known, is part of what the first-build fit
    // shows. The axes float just outside the extent, so the fit keeps a
    // slice of room above the ceiling and below the floor for their bands.
    const double height = std::max(sceneBottomY - sceneTopY, 1e-6);
    _contentRect =
        QRectF(_layout.x0Vx, sceneTopY - 0.06 * height, sceneWidth, 1.12 * height);

    // Panning stops at the scene rect, so the rect runs wider than the content:
    // zoomed in, the map's edges can be dragged away from the viewport edge
    // instead of being pinned to it.
    const double xMargin = std::max(0.25 * sceneWidth, kMinSceneMarginCm * vxPerCm);
    _scene->setSceneRect(_contentRect.adjusted(-xMargin, 0.0, xMargin, 0.0));

    if (_layout.rRefVx > 0.0) {
        _chipHideScale =
            kMinChipPixelsPerWinding / (2.0 * M_PI * _layout.rRefVx);
    }
    updateLabelChipVisibility();
}

void FiberMapWorkspace::updateLabelChipVisibility()
{
    if (!_view || _labelChips.empty()) {
        return;
    }
    const bool visible =
        std::abs(_view->transform().m11()) >= _chipHideScale;
    for (QGraphicsItem* chip : _labelChips) {
        chip->setVisible(visible);
    }
}

void FiberMapWorkspace::rebuildTree()
{
    const bool guard = _syncingSelection;
    _syncingSelection = true;
    _tree->clear();
    // The rows carry the map's own colours, so they follow the theme with it;
    // everything else about the tree is the widget palette's business.
    const FiberMapPalette& theme = activePalette();

    // Grouped by linked network, largest first (the layout numbers network
    // ids by size), then every unlinked fiber flat; inner -> outer within
    // each group.
    std::map<int, std::vector<const vc3d::fiber_map::GlobalPlacedFiber*>> networks;
    std::vector<const vc3d::fiber_map::GlobalPlacedFiber*> individual;
    for (const vc3d::fiber_map::GlobalPlacedFiber& fiber : _layout.fibers) {
        if (fiber.meta.networkId >= 0) {
            networks[fiber.meta.networkId].push_back(&fiber);
        } else {
            individual.push_back(&fiber);
        }
    }
    const auto innerToOuter = [](const vc3d::fiber_map::GlobalPlacedFiber* a,
                                 const vc3d::fiber_map::GlobalPlacedFiber* b) {
        if (a->meta.windingLo != b->meta.windingLo) {
            return a->meta.windingLo < b->meta.windingLo;
        }
        if (a->fiber.label != b->fiber.label) {
            return a->fiber.label < b->fiber.label;
        }
        return a->fiber.id < b->fiber.id;
    };
    for (auto& [id, members] : networks) {
        std::sort(members.begin(), members.end(), innerToOuter);
    }
    std::sort(individual.begin(), individual.end(), innerToOuter);

    // A multi-turn H fiber has no single winding, so the column shows the
    // range it spans.
    const auto windingText = [](const vc3d::fiber_map::GlobalFiberMeta& meta) {
        const auto lo = static_cast<long long>(std::floor(meta.windingLo));
        const auto hi = static_cast<long long>(std::floor(meta.windingHi));
        if (lo == hi) {
            return QString::number(lo);
        }
        return QStringLiteral("%1–%2").arg(lo).arg(hi);
    };
    // How the fiber's component got its absolute winding — the UI must not
    // imply winding knowledge the solve does not have.
    const auto anchorText = [this](const vc3d::fiber_map::GlobalFiberMeta& meta) {
        QString text;
        switch (meta.anchor) {
        case vc3d::fiber_map::GlobalAnchor::Primary:
            text = tr("crossings");
            break;
        case vc3d::fiber_map::GlobalAnchor::Radius:
            text = tr("radius");
            break;
        case vc3d::fiber_map::GlobalAnchor::AmbiguousRadius:
            text = tr("radius?");
            break;
        case vc3d::fiber_map::GlobalAnchor::Unresolved:
            text = tr("unresolved");
            break;
        }
        if (meta.sheetDriftSuspect) {
            text += tr(" · drift?");
        }
        if (meta.onKollesis) {
            text += tr(" · kollesis");
        }
        return text;
    };
    const auto addFiberRow = [&](QTreeWidgetItem* parent,
                                 const vc3d::fiber_map::GlobalPlacedFiber* row) {
        const QString annotationName =
            _controller ? _controller->fiberDisplayName(row->fiber.id) : QString();
        auto* item = parent != nullptr
            ? new QTreeWidgetItem(
                  parent, {row->fiber.label, QString(QLatin1Char(row->fiber.hvTag)),
                           windingText(row->meta), anchorText(row->meta),
                           annotationName})
            : new QTreeWidgetItem(
                  _tree, {row->fiber.label, QString(QLatin1Char(row->fiber.hvTag)),
                          windingText(row->meta), anchorText(row->meta),
                          annotationName});
        item->setData(0, Qt::UserRole, QVariant::fromValue<qulonglong>(row->fiber.id));
        const QColor color = fiberColor(row->fiber.hvTag, theme);
        for (int column = 0; column < _tree->columnCount(); ++column) {
            item->setForeground(column, color);
        }
        item->setForeground(3, theme.inkSoft);
    };

    for (const auto& [id, members] : networks) {
        auto* networkItem = new QTreeWidgetItem(
            _tree, {tr("Network %1 — %2 fibers")
                        .arg(id + 1)
                        .arg(members.size())});
        networkItem->setForeground(0, theme.inkSoft);
        // A header across the whole row, so the columns stay narrow.
        networkItem->setFirstColumnSpanned(true);
        for (const vc3d::fiber_map::GlobalPlacedFiber* row : members) {
            addFiberRow(networkItem, row);
        }
        networkItem->setExpanded(true);
    }
    for (const vc3d::fiber_map::GlobalPlacedFiber* row : individual) {
        addFiberRow(nullptr, row);
    }
    for (const vc3d::fiber_map::UnplacedFiber& unplaced : _layout.unplaced) {
        const QString annotationName =
            _controller ? _controller->fiberDisplayName(unplaced.id) : QString();
        auto* item = new QTreeWidgetItem(
            _tree, {unplaced.label, QString(QLatin1Char(unplaced.hvTag)),
                    QStringLiteral("—"), tr("unplaceable"), annotationName});
        item->setData(0, Qt::UserRole, QVariant::fromValue<qulonglong>(unplaced.id));
        for (int column = 0; column < _tree->columnCount(); ++column) {
            item->setForeground(column, theme.inkSoft);
        }
    }
    _syncingSelection = guard;
}

// A theme switch changes every colour of the map, and both the scene and the
// tree hold theirs as fixed brushes and pens. Rebuilding from the layout in hand
// recolours them without recomputing anything, so the switch needs no Rebuild.
void FiberMapWorkspace::changeEvent(QEvent* event)
{
    QMainWindow::changeEvent(event);
    if (!event || (event->type() != QEvent::PaletteChange &&
                   event->type() != QEvent::ApplicationPaletteChange)) {
        return;
    }
    // Both event types can arrive for one switch, and rebuilding sets widget
    // properties that may deliver more; the first pass does the work. A palette
    // change can also reach a half-built window, which has nothing to recolour
    // yet: the constructor's own rebuild covers it.
    if (_retheming || !_scene || !_tree) {
        return;
    }
    _retheming = true;
    // rebuildScene clears the highlight, so it is restored afterwards: a theme
    // switch should not cost the user their selection.
    const uint64_t highlighted = _highlightedFiber;
    const QString emptyMessage = _emptyMessage;
    rebuildScene(emptyMessage);
    rebuildTree();
    if (highlighted != 0 && _entries.contains(highlighted)) {
        setHighlightedFiber(highlighted);
        selectFiberRow(highlighted);
    }
    _retheming = false;
}

double FiberMapWorkspace::sceneTolerance(double viewPixels) const
{
    const double scale = std::abs(_view->transform().m11());
    if (scale <= 0.0) {
        return viewPixels;
    }
    return viewPixels / scale;
}

uint64_t FiberMapWorkspace::fiberAt(const QPointF& scenePos) const
{
    // Only the label chips answer by hit test. A fiber path's shape() is its
    // painter path stroked with the pen width read as scene units, and the fiber
    // pens are cosmetic (2.2 device pixels, hence a 2.2-voxel ribbon in the
    // scene), so consulting the items would hand every click to whichever of the
    // overlapping ribbons happens to stack highest instead of to the nearest
    // fiber.
    const QList<QGraphicsItem*> under = _scene->items(
        scenePos, Qt::IntersectsItemShape, Qt::DescendingOrder, _view->transform());
    for (const QGraphicsItem* item : under) {
        if (item->type() != kChipItemType || !item->isVisible()) {
            continue;
        }
        const uint64_t fiberId = item->data(0).toULongLong();
        if (fiberId != 0 && _entries.contains(fiberId)) {
            return fiberId;
        }
    }

    // Everything else is decided by proximity to the placed runs: nearest fiber
    // within the tolerance wins.
    const double tolerance = sceneTolerance(kFiberHitTolerancePx);
    uint64_t best = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (auto entry = _entries.constBegin(); entry != _entries.constEnd(); ++entry) {
        double distance = std::numeric_limits<double>::infinity();
        for (const vc3d::fiber_map::Run& run : entry->fiber.runs) {
            for (std::size_t i = 1; i < run.points.size(); ++i) {
                distance = std::min(
                    distance, distanceToSegment(scenePos, run.points[i - 1], run.points[i]));
            }
        }
        if (distance > tolerance) {
            continue;
        }
        // _entries iterates in hash order, so an exact tie is settled by the
        // fiber id rather than by whichever fiber came up first.
        if (distance < bestDistance || (distance == bestDistance && entry.key() < best)) {
            bestDistance = distance;
            best = entry.key();
        }
    }
    return best;
}

void FiberMapWorkspace::handleSceneClick(const QPointF& scenePos)
{
    // A stale map's runtime ids may name different fibers than they did when it
    // was built, so it stops responding until rebuilt.
    if (refreshStaleState()) {
        return;
    }
    const uint64_t fiberId = fiberAt(scenePos);
    setHighlightedFiber(fiberId);
    if (fiberId != 0) {
        selectFiberRow(fiberId);
    }
}

void FiberMapWorkspace::selectFiberRow(uint64_t fiberId)
{
    const bool guard = _syncingSelection;
    _syncingSelection = true;
    const auto matches = [fiberId](QTreeWidgetItem* item) {
        return item->data(0, Qt::UserRole).toULongLong() == fiberId;
    };
    for (int row = 0; row < _tree->topLevelItemCount(); ++row) {
        QTreeWidgetItem* item = _tree->topLevelItem(row);
        QTreeWidgetItem* hit = matches(item) ? item : nullptr;
        for (int child = 0; hit == nullptr && child < item->childCount(); ++child) {
            if (matches(item->child(child))) {
                hit = item->child(child);
            }
        }
        if (hit != nullptr) {
            _tree->setCurrentItem(hit);
            _tree->scrollToItem(hit);
            break;
        }
    }
    _syncingSelection = guard;
}

void FiberMapWorkspace::clearControlPointDots()
{
    for (QGraphicsItem* dot : _controlPointDots) {
        _scene->removeItem(dot);
        delete dot;
    }
    _controlPointDots.clear();
}

void FiberMapWorkspace::paintFiberEmphasis(FiberEntry& entry,
                                           FiberEmphasis emphasis)
{
    const FiberMapPalette& theme = activePalette();
    const QColor color = fiberColor(entry.fiber.hvTag, theme);
    const bool selected = emphasis == FiberEmphasis::Selected;
    if (entry.tracedItem) {
        entry.tracedItem->setPen(cosmeticPen(
            color, selected ? kTracedHighlightWidth : kTracedWidth));
        entry.tracedItem->setZValue(selected ? kHighlightZ : kFiberZ);
    }
    if (entry.interpolatedItem) {
        entry.interpolatedItem->setPen(interpolatedPen(
            tint(color, theme.surface, 0.45),
            selected ? kInterpolatedHighlightWidth : kInterpolatedWidth));
        entry.interpolatedItem->setZValue(selected ? kHighlightZ : kFiberZ);
    }
    // The network role adds a halo behind the unchanged lines; every other
    // role removes it. The halo strokes the fiber's whole geometry (traced
    // and interpolated runs alike) in one soft ribbon.
    if (emphasis == FiberEmphasis::Network) {
        if (entry.glowItem == nullptr) {
            QPainterPath path;
            for (const vc3d::fiber_map::Run& run : entry.fiber.runs) {
                if (run.points.size() < 2) {
                    continue;
                }
                path.moveTo(run.points.front());
                for (std::size_t i = 1; i < run.points.size(); ++i) {
                    path.lineTo(run.points[i]);
                }
            }
            QColor glow = color;
            glow.setAlpha(kNetworkGlowAlpha);
            entry.glowItem =
                _scene->addPath(path, cosmeticPen(glow, kNetworkGlowWidthPx));
            entry.glowItem->setZValue(kNetworkGlowZ);
        }
    } else if (entry.glowItem != nullptr) {
        _scene->removeItem(entry.glowItem);
        delete entry.glowItem;
        entry.glowItem = nullptr;
    }
}

void FiberMapWorkspace::setHighlightedFiber(uint64_t fiberId)
{
    if (_highlightedFiber == fiberId) {
        return;
    }
    // Restore the previous selection and its network's glow.
    if (const auto previous = _entries.find(_highlightedFiber);
        previous != _entries.end()) {
        paintFiberEmphasis(*previous, FiberEmphasis::Plain);
    }
    for (const uint64_t member : _networkEmphasized) {
        if (const auto entry = _entries.find(member); entry != _entries.end()) {
            paintFiberEmphasis(*entry, FiberEmphasis::Plain);
        }
    }
    _networkEmphasized.clear();
    clearControlPointDots();
    _highlightedFiber = fiberId;

    const auto entry = _entries.find(fiberId);
    if (entry == _entries.end()) {
        return;
    }
    // The whole linked network glows; the selected fiber itself gets the
    // full treatment instead.
    if (entry->networkId >= 0) {
        for (auto other = _entries.begin(); other != _entries.end(); ++other) {
            if (other->networkId == entry->networkId &&
                other.key() != fiberId) {
                paintFiberEmphasis(*other, FiberEmphasis::Network);
                _networkEmphasized.push_back(other.key());
            }
        }
    }
    paintFiberEmphasis(*entry, FiberEmphasis::Selected);
    const FiberMapPalette& theme = activePalette();
    const QColor color = fiberColor(entry->fiber.hvTag, theme);
    const double vxPerCm = sceneVxPerCm();
    // The selected fiber's dots cover the scene's kollesis markers, so a
    // tagged point keeps its look here: a hollow yellow ring when unlinked,
    // the link fill inside the yellow ring when linked.
    const char selfTag = entry->fiber.hvTag;
    const auto linkFillFor = [this, selfTag](uint64_t fiberId,
                                             int controlIndex) -> std::optional<QColor> {
        for (const vc3d::fiber_map::PlacedLink& link : _layout.links) {
            const bool atA = link.fiberA == fiberId && link.cpA == controlIndex;
            const bool atB = link.fiberB == fiberId && link.cpB == controlIndex;
            if ((!atA && !atB) || link.suspect) {
                continue;
            }
            const auto other = _entries.constFind(atA ? link.fiberB : link.fiberA);
            const char otherTag = other == _entries.constEnd() ? '?' : other->fiber.hvTag;
            return linkPalette(selfTag, otherTag, link.pending).brush;
        }
        return std::nullopt;
    };
    for (std::size_t i = 0; i < entry->fiber.controlPoints.size(); ++i) {
        QBrush fill(color);
        QPen rim = cosmeticPen(theme.chipInk, 1.0);
        if (i < entry->fiber.kollesisTerminations.size() && entry->fiber.kollesisTerminations[i]) {
            if (const auto linkFill = linkFillFor(fiberId, static_cast<int>(i))) {
                fill = QBrush(*linkFill);
            } else {
                fill = QBrush(Qt::NoBrush);
            }
            rim = cosmeticPen(kollesisColor(255), kKollesisRimWidthPx);
        }
        auto* dot = new ScaledDot(fill, rim,
                                  kControlDotRadiusCm * vxPerCm, kMinControlDotPx,
                                  kMaxControlDotPx, kControlDotBoundsCm * vxPerCm);
        _scene->addItem(dot);
        dot->setPos(entry->fiber.controlPoints[i]);
        dot->setZValue(kHighlightZ + 1.0);
        dot->setData(0, QVariant::fromValue<qulonglong>(fiberId));
        dot->setData(1, static_cast<int>(i));
        _controlPointDots.push_back(dot);
    }
}

void FiberMapWorkspace::handleControlPointMenu(const QPointF& scenePos, const QPoint& globalPos)
{
    // The menu acts on the selected fiber only, and only when the click lands
    // on it - its line or one of its control dots. Selecting by left click
    // first is what says, unambiguously, which fiber "Delete" means; a
    // ctrl+right-click elsewhere does nothing rather than guess.
    if (_highlightedFiber == 0 || !_controller) {
        return;
    }
    if (refreshStaleState()) {
        return;
    }
    const uint64_t fiberId = _highlightedFiber;
    const auto entry = _entries.constFind(fiberId);
    if (entry == _entries.constEnd()) {
        return;
    }

    // Grabbing a dot must work wherever it is drawn: kControlDotTolerancePx is
    // the floor, the dot's drawn radius at this zoom takes over once it is
    // the bigger of the two.
    const double vxPerCm = sceneVxPerCm();
    const double drawnRadius = scaledDotRadius(
        kControlDotRadiusCm * vxPerCm, kMinControlDotPx, kMaxControlDotPx,
        kControlDotBoundsCm * vxPerCm, std::abs(_view->transform().m11()));
    const double tolerance = std::max(sceneTolerance(kControlDotTolerancePx), drawnRadius);
    int bestIndex = -1;
    double bestDistance = tolerance;
    for (QGraphicsItem* dot : _controlPointDots) {
        const QPointF delta = scenePos - dot->pos();
        const double distance = std::sqrt(QPointF::dotProduct(delta, delta));
        if (distance <= bestDistance) {
            bestDistance = distance;
            bestIndex = dot->data(1).toInt();
        }
    }
    if (bestIndex < 0 && fiberAt(scenePos) != fiberId) {
        return;
    }

    const std::string fileName = entry->fiber.fileName;
    const QString displayName = _controller->fiberDisplayName(fiberId);
    // menu.exec() runs a nested event loop, so the fiber set can change while
    // the menu is open. Two protections: the dependency set is captured now and
    // re-compared when an action fires, because bestIndex indexes the control
    // points as they were when the menu was built — an edit in between could
    // have made it mean a different point, or none; and the fiber must still
    // be loaded under the captured id and file name at that same moment.
    const vc3d::fiber_map::FiberMapDependencies menuDependencies =
        currentDependencies();
    // Parentless: exec() runs a nested event loop, and a parented stack menu would
    // be deleted by its parent if the workspace went away inside it and then
    // destroyed again by stack unwinding.
    QMenu menu;
    if (bestIndex >= 0) {
        QAction* action = menu.addAction(tr("Go to control point %1 in %2")
                                            .arg(bestIndex)
                                            .arg(displayName));
    connect(action, &QAction::triggered, this,
            [this, fiberId, fileName, bestIndex, menuDependencies]() {
                if (!_controller) {
                    return;
                }
                // The shared decision, against the menu's own capture rather
                // than the layout's: the question here is whether anything
                // moved while the menu was open. Applied non-destructively —
                // this runs inside menu.exec()'s nested event loop, and
                // refreshStaleState() can reach clearLayout(), which tears
                // down scene items while the press that opened the menu is
                // still unwinding. The banner goes up now; a clear, if one is
                // due, happens at the next natural moment.
                const StaleVerdict verdict = vc3d::fiber_map::staleVerdictFor(
                    menuDependencies, currentDependencies(),
                    /*layoutBuilt=*/true, QString());
                if (verdict.action != StaleVerdict::Action::Fresh) {
                    showStale(verdict.reason);
                    // The destructive half (a clear, or scheduling the
                    // automatic update) runs on the next event-loop turn,
                    // after menu.exec()'s nested loop has unwound - exactly
                    // like the tree handler's deferral, and for the same
                    // reason: applying a verdict here can tear down scene
                    // items mid-delivery.
                    QMetaObject::invokeMethod(
                        this, [this]() { refreshStaleState(); },
                        Qt::QueuedConnection);
                    Logger()->warn(
                        "Fiber map: dependencies changed while the menu was open; "
                        "not navigating to control point {} in {}",
                        bestIndex,
                        fileName);
                    return;
                }
                // The defense the generation cannot give: a fiber no longer
                // loaded under its id and name under an unchanged generation
                // means a bump was missed somewhere, and this map cannot be
                // trusted until it is rebuilt — the one staleness that latches.
                if (!_controller->hasLoadedFiber(fiberId, fileName)) {
                    markStale(tr("Fibers changed — press Update"));
                    Logger()->warn("Fiber map: {} is no longer loaded; not navigating",
                                   fileName);
                    return;
                }
                emit openFiberAtControlPointRequested(fiberId, bestIndex);
            });
        menu.addSeparator();
    }
    QAction* deleteAction = menu.addAction(tr("Delete %1…").arg(displayName));
    deleteAction->setEnabled(!_deleteInFlight);
    connect(deleteAction, &QAction::triggered, this,
            [this, fiberId, fileName, displayName, menuDependencies]() {
                // Deferred past menu.exec()'s nested loop: the confirmation
                // is modal, and the delete itself ends in a scene rebuild
                // that must not tear items down while the press that opened
                // the menu is still unwinding.
                QMetaObject::invokeMethod(
                    this,
                    [this, fiberId, fileName, displayName, menuDependencies]() {
                        confirmAndDeleteFiber(fiberId, fileName, displayName, menuDependencies);
                    },
                    Qt::QueuedConnection);
            });
    menu.exec(globalPos);
}

void FiberMapWorkspace::handleTreeContextMenu(const QPoint& pos)
{
    if (!_tree || !_controller) {
        return;
    }
    QTreeWidgetItem* item = _tree->itemAt(pos);
    if (!item) {
        return;
    }
    // Network headers carry no fiber.
    const uint64_t fiberId = item->data(0, Qt::UserRole).toULongLong();
    if (fiberId == 0) {
        return;
    }
    if (refreshStaleState()) {
        return;
    }
    // The row is selected first, so the fiber the menu names is the fiber
    // highlighted on the map - the same rule as the map's own menu.
    if (_tree->currentItem() != item) {
        _tree->setCurrentItem(item);
    }
    std::string fileName;
    if (const auto entry = _entries.constFind(fiberId); entry != _entries.constEnd()) {
        fileName = entry->fiber.fileName;
    } else {
        for (const vc3d::fiber_map::UnplacedFiber& unplaced : _layout.unplaced) {
            if (unplaced.id == fiberId) {
                fileName = unplaced.fileName;
                break;
            }
        }
    }
    if (fileName.empty()) {
        return;
    }
    const QString displayName = _controller->fiberDisplayName(fiberId);
    const vc3d::fiber_map::FiberMapDependencies menuDependencies =
        currentDependencies();
    QMenu menu;
    QAction* deleteAction = menu.addAction(tr("Delete %1…").arg(displayName));
    deleteAction->setEnabled(!_deleteInFlight);
    connect(deleteAction, &QAction::triggered, this,
            [this, fiberId, fileName, displayName, menuDependencies]() {
                QMetaObject::invokeMethod(
                    this,
                    [this, fiberId, fileName, displayName, menuDependencies]() {
                        confirmAndDeleteFiber(fiberId, fileName, displayName, menuDependencies);
                    },
                    Qt::QueuedConnection);
            });
    menu.exec(_tree->viewport()->mapToGlobal(pos));
}

void FiberMapWorkspace::confirmAndDeleteFiber(
    uint64_t fiberId,
    const std::string& fileName,
    const QString& displayName,
    const vc3d::fiber_map::FiberMapDependencies& menuDependencies)
{
    if (!_controller) {
        return;
    }
    // Anything moved since the menu was built? Then this map is not the thing
    // to be deleting from; it refreshes instead (this runs outside any nested
    // loop, so the destructive half may run inline).
    const auto dependenciesMoved = [this, &menuDependencies, &fileName](const char* when) {
        const StaleVerdict verdict = vc3d::fiber_map::staleVerdictFor(
            menuDependencies, currentDependencies(), /*layoutBuilt=*/true, QString());
        if (verdict.action == StaleVerdict::Action::Fresh) {
            return false;
        }
        refreshStaleState();
        Logger()->warn("Fiber map: dependencies changed while the {} was open; "
                       "not deleting {}",
                       when, fileName);
        return true;
    };
    if (dependenciesMoved("menu")) {
        return;
    }
    // One delete at a time. The controller's deleteFibers drains queued
    // saves in a nested loop that processes input, so without this a second
    // confirmation of the same fiber could start a second delete that, once
    // the first had removed the file, fell back to an unrelated one.
    if (_deleteInFlight) {
        Logger()->warn("Fiber map: a delete is already pending; ignoring {}", fileName);
        return;
    }
    _deleteInFlight = true;
    // The confirmation is modeless (open(), not exec()): a nested loop with a
    // parented dialog would be undefined if the workspace were torn down
    // meanwhile, whereas this dialog simply dies with its parent and the
    // handler, being connected in the workspace's context, is dropped.
    auto* dialog = new QMessageBox(
        QMessageBox::Question,
        tr("Delete fiber"),
        tr("Delete fiber %1?\n\nThis removes its file from the package and cannot be undone.")
            .arg(displayName),
        QMessageBox::Yes | QMessageBox::Cancel,
        this);
    dialog->setDefaultButton(QMessageBox::Cancel);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    connect(dialog, &QMessageBox::buttonClicked, this,
            [this, dialog, fiberId, fileName, menuDependencies](QAbstractButton* button) {
                // Only the answer is read here. The delete itself is queued
                // out of the dialog's own signal delivery: it drains queued
                // saves in a nested loop, and a workspace torn down during
                // that would take the parented dialog with it while Qt is
                // still finishing this very emission on it.
                if (dialog->standardButton(button) != QMessageBox::Yes) {
                    return;
                }
                QMetaObject::invokeMethod(
                    this,
                    [this, fiberId, fileName, menuDependencies]() {
                        deleteConfirmedFiber(fiberId, fileName, menuDependencies);
                    },
                    Qt::QueuedConnection);
            });
    // Any way of closing the dialog other than Yes (Cancel, Escape, the
    // window close) releases the guard; Yes hands it to deleteConfirmedFiber.
    // finished() carries the standard button for a button click, and
    // QDialog::Rejected for a close, neither of which is Yes.
    connect(dialog, &QMessageBox::finished, this, [this](int result) {
        if (result != QMessageBox::Yes) {
            _deleteInFlight = false;
        }
    });
    dialog->open();
}

void FiberMapWorkspace::deleteConfirmedFiber(
    uint64_t fiberId,
    const std::string& fileName,
    const vc3d::fiber_map::FiberMapDependencies& menuDependencies)
{
    // The guard taken at confirmation is released on every way out of here,
    // including after a deleteFibers that outlived the workspace (then there
    // is nothing left to release).
    const QPointer<FiberMapWorkspace> self(this);
    const auto releaseGuard = qScopeGuard([self]() {
        if (self) {
            self->_deleteInFlight = false;
        }
    });
    if (!_controller) {
        return;
    }
    // Anything could have happened while the dialog stood open (a reload, a
    // package switch): the dependency set is checked again, and then the
    // fiber must still be loaded under the id AND the name the menu named
    // (runtime ids are stable and unique across sources; a bare name is
    // not, so the id is what is deleted). A fiber gone under unchanged
    // dependencies means the map is not to be trusted until rebuilt - the
    // one staleness that latches.
    const StaleVerdict verdict = vc3d::fiber_map::staleVerdictFor(
        menuDependencies, currentDependencies(), /*layoutBuilt=*/true, QString());
    if (verdict.action != StaleVerdict::Action::Fresh) {
        refreshStaleState();
        Logger()->warn("Fiber map: dependencies changed while the confirmation was open; "
                       "not deleting {}",
                       fileName);
        return;
    }
    if (!_controller->hasLoadedFiber(fiberId, fileName)) {
        markStale(tr("Fibers changed — press Update"));
        Logger()->warn("Fiber map: {} is no longer loaded; not deleting", fileName);
        return;
    }
    Logger()->info("Fiber map: deleting fiber {}", fileName);
    // deleteFibers drains queued saves in a nested loop, during which this
    // workspace could be destroyed; `self` keeps the epilogue off a dead
    // object.
    _controller->deleteFibers({fiberId});
    if (!self) {
        return;
    }
    // The fiber generation moved; rather than wait for the visible-only
    // poll's next tick, notice it now so the automatic update starts at once.
    refreshStaleState();
}
