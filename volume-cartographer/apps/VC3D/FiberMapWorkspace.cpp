#include "FiberMapWorkspace.hpp"

#include "LineAnnotationController.hpp"

#include "vc/core/util/Logging.hpp"

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
#include <QMouseEvent>
#include <QPainter>
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
#include <QTransform>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVariant>
#include <QWheelEvent>
#include <QWindow>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <limits>
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
// Dots (control points, link crossings, suspect-link rings) are drawn in scene
// units, so they grow with the zoom, but never smaller than their kMin*Px on
// screen: a few pixels when a whole network is in view, an easy target once
// zoomed in. The third number of each triple is the ceiling for the
// pixel-clamped radius, and with it the painting bounds: once zoomed far enough
// out the dots stop growing in scene units rather than outrun their bounding
// rect.
//
// These are the sizes the markers are meant to have on a printed map, in cm;
// the scene is in voxels, so each is multiplied by sceneVxPerCm() at build time.
// Nothing below may reach the scene without that conversion.
constexpr qreal kControlDotRadiusCm = 0.06;
constexpr qreal kMinControlDotPx = 3.5;
constexpr qreal kControlDotBoundsCm = 0.5;
constexpr qreal kCrossingDotRadiusCm = 0.10;
constexpr qreal kMinCrossingDotPx = 5.2;
constexpr qreal kCrossingDotBoundsCm = 0.4;
constexpr qreal kSuspectRingRadiusCm = 0.08;
constexpr qreal kMinSuspectRingPx = 4.0;
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

// Round marker of the map: the highlighted fiber's control points, the link
// crossings and the suspect-link rings. Unlike the pinned chips it lives in
// scene space, so zooming in makes it a bigger target; the on-screen radius is
// only clamped from below so the markers stay visible when zoomed out.
class ScaledDot : public QGraphicsItem
{
public:
    // radius and maxRadius are scene units (voxels); minPixels is on screen, and
    // the level-of-detail factor converts between the two, so this needs to know
    // nothing about what a scene unit measures.
    ScaledDot(const QBrush& fill, const QPen& outline, qreal radius,
              qreal minPixels, qreal maxRadius)
        : _fill(fill)
        , _outline(outline)
        , _radius(radius)
        , _minPixels(minPixels)
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
        const qreal radius = lod > 0.0
            ? std::clamp<qreal>(_minPixels / lod, _radius, _maxRadius)
            : _radius;
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
    setCursor(Qt::ArrowCursor);
    // The right button drives the pan, so the platform must not turn it into a
    // context-menu event that would reach the surrounding QMainWindow.
    setContextMenuPolicy(Qt::PreventContextMenu);
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
    if (_statusLabel) {
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

        // The layout and solver are unit-free, so the physical intents behind
        // their tuning lengths are converted here — once the voxel size is
        // known, exactly as documented on GlobalLayoutParams and SolverParams.
        // Left alone when it is not, so the defaults (the same intents at
        // 2.4 µm) stand in and the map still lays out sensibly.
        if (job->snapshot.voxelSizeUm) {
            const double vxPerCm = kUmPerCm / *job->snapshot.voxelSizeUm;
            job->params.smoothVx = 0.12 * vxPerCm;         // 1.2 mm arclength sigma
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

    QString status = tr("%1 fibers · %2 windings · %3 islands · %4 suspect links")
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

    if (_layout.fibers.empty()) {
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

    // One ground for the whole map, spanning the scroll's own z extent.
    auto* ground = _scene->addRect(
        QRectF(QPointF(_layout.x0Vx, extentTopY), QPointF(_layout.x1Vx, extentBottomY)),
        QPen(Qt::NoPen), QBrush(tint(theme.surface, theme.ink, 0.045)));
    ground->setZValue(kPanelZ);

    // The winding grid, one numbered line per integer winding: the number IS
    // the winding coordinate, innermost anchored winding zero.
    for (const vc3d::fiber_map::WindingMark& mark : _layout.windings) {
        auto* line = _scene->addLine(mark.xVx, extentTopY, mark.xVx, extentBottomY);
        QPen pen(theme.winding);
        pen.setWidthF(0.8);
        pen.setCosmetic(true);
        pen.setStyle(Qt::DotLine);
        line->setPen(pen);
        line->setZValue(0.0);
        auto* number = _scene->addSimpleText(QString::number(mark.number), labelFont);
        number->setBrush(theme.winding);
        pinText(number, QPointF(mark.xVx, extentTopY), 0.0, -16.0, true);
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
            const LinkPalette& palette =
                linkPalette(hvTagOf(link.fiberA), hvTagOf(link.fiberB), link.pending);
            auto* dot = new ScaledDot(QBrush(palette.brush),
                                      cosmeticPen(palette.pen, 1.0),
                                      crossingDotRadius,
                                      kMinCrossingDotPx, crossingDotBounds);
            _scene->addItem(dot);
            dot->setPos(middle);
            dot->setZValue(4.0);
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
                                       suspectRingBounds);
            _scene->addItem(ring);
            ring->setPos(endpoint);
            ring->setZValue(5.0);
        }
        auto* label = _scene->addSimpleText(
            tr("+%1 turn").arg(link.turnErr, 0, 'f', 1), labelFont);
        label->setBrush(kSuspect);
        pinText(label, middle, 0.0, -14.0, true);
        label->setZValue(5.0);
    }

    // Crossings the winding repair had to drop: contradicted evidence, marked
    // where the H fiber made the pass.
    for (const vc3d::fiber_map::CrossingMark& mark : _layout.suspectCrossings) {
        auto* ring = new ScaledDot(QBrush(Qt::NoBrush), cosmeticPen(kSuspect, 1.4),
                                   suspectRingRadius, kMinSuspectRingPx,
                                   suspectRingBounds);
        _scene->addItem(ring);
        ring->setPos(QPointF(mark.posVx.x(), -mark.posVx.y()));
        ring->setZValue(5.0);
    }

    // The winding numbers hang above the top edge in device pixels, so the
    // scene keeps a slice of room for them above the layout. The scroll
    // extent, when known, is part of what the first-build fit shows.
    const double height = std::max(sceneBottomY - sceneTopY, 1e-6);
    _contentRect =
        QRectF(_layout.x0Vx, sceneTopY - 0.10 * height, sceneWidth, 1.12 * height);

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
    for (std::size_t i = 0; i < entry->fiber.controlPoints.size(); ++i) {
        auto* dot = new ScaledDot(QBrush(color), cosmeticPen(theme.chipInk, 1.0),
                                  kControlDotRadiusCm * vxPerCm, kMinControlDotPx,
                                  kControlDotBoundsCm * vxPerCm);
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
    if (_highlightedFiber == 0 || _controlPointDots.empty() || !_controller) {
        return;
    }
    if (refreshStaleState()) {
        return;
    }
    // Grabbing a dot must work wherever it is drawn: kControlDotTolerancePx is
    // the floor, the scene-space radius takes over once zoomed in.
    const double tolerance = std::max(sceneTolerance(kControlDotTolerancePx),
                                      kControlDotRadiusCm * sceneVxPerCm());
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
    if (bestIndex < 0) {
        return;
    }

    const uint64_t fiberId = _highlightedFiber;
    const auto entry = _entries.constFind(fiberId);
    if (entry == _entries.constEnd()) {
        return;
    }
    const std::string fileName = entry->fiber.fileName;
    // Parentless: exec() runs a nested event loop, and a parented stack menu would
    // be deleted by its parent if the workspace went away inside it and then
    // destroyed again by stack unwinding.
    QMenu menu;
    QAction* action = menu.addAction(tr("Go to control point %1 in %2")
                                        .arg(bestIndex)
                                        .arg(_controller->fiberDisplayName(fiberId)));
    // menu.exec() runs a nested event loop, so the fiber set can change while
    // the menu is open. Two protections: the dependency set is captured now and
    // re-compared when the action fires, because bestIndex indexes the control
    // points as they were when the menu was built — an edit in between could
    // have made it mean a different point, or none; and the runtime id is
    // resolved from the file name at that same moment, because a reload
    // reassigns ids.
    const vc3d::fiber_map::FiberMapDependencies menuDependencies =
        currentDependencies();
    connect(action, &QAction::triggered, this,
            [this, fileName, bestIndex, menuDependencies]() {
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
                // The defense the generation cannot give: a file name that no
                // longer resolves under an unchanged generation means a bump
                // was missed somewhere, and this map cannot be trusted until
                // it is rebuilt — the one staleness that latches.
                const uint64_t target = _controller->fiberIdForFileName(fileName);
                if (target == 0) {
                    markStale(tr("Fibers changed — press Update"));
                    Logger()->warn("Fiber map: {} is no longer loaded; not navigating",
                                   fileName);
                    return;
                }
                emit openFiberAtControlPointRequested(target, bestIndex);
            });
    menu.exec(globalPos);
}
