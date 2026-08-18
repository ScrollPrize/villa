#include "FiberMapWorkspace.hpp"

#include "LineAnnotationController.hpp"

#include "vc/core/util/Logging.hpp"

#include <QAction>
#include <QColor>
#include <QDockWidget>
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
#include <QScrollBar>
#include <QSpinBox>
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
#include <cmath>
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
constexpr qreal kPanelZ = -3.0;
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
    QRectF bounds;
    for (const vc3d::fiber_map::Run& run : fiber.runs) {
        for (const QPointF& point : run.points) {
            bounds = bounds.isNull() ? QRectF(point, QSizeF(0.0, 0.0))
                                     : bounds.united(QRectF(point, QSizeF(0.0, 0.0)));
        }
    }
    return bounds;
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
    toolBar->addWidget(new QLabel(tr("Top networks"), toolBar));
    _topNetworkSpin = new QSpinBox(toolBar);
    _topNetworkSpin->setRange(1, 20);
    _topNetworkSpin->setValue(3);
    toolBar->addWidget(_topNetworkSpin);
    toolBar->addSeparator();
    toolBar->addWidget(new QLabel(tr("Min fibers"), toolBar));
    _minFiberSpin = new QSpinBox(toolBar);
    _minFiberSpin->setRange(2, 99);
    _minFiberSpin->setValue(3);
    toolBar->addWidget(_minFiberSpin);
    toolBar->addSeparator();
    auto* rebuildButton = new QPushButton(tr("Rebuild layout"), toolBar);
    toolBar->addWidget(rebuildButton);
    toolBar->addSeparator();
    _statusLabel =
        new QLabel(tr("press Rebuild layout"), toolBar);
    toolBar->addWidget(_statusLabel);

    _tree = new QTreeWidget(this);
    _tree->setColumnCount(3);
    _tree->setHeaderLabels({tr("Fiber"), tr("H/V"), tr("Annotation")});
    _tree->setUniformRowHeights(true);
    _tree->setSelectionMode(QAbstractItemView::SingleSelection);
    // The short label and H/V take only what they need; the annotation name gets
    // the rest of the dock.
    _tree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _tree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
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

    connect(rebuildButton, &QPushButton::clicked, this, &FiberMapWorkspace::rebuildLayout);
    const auto markSettingsStale = [this]() {
        markStale(tr("Settings changed — press Rebuild layout"));
    };
    connect(_topNetworkSpin, &QSpinBox::valueChanged, this, markSettingsStale);
    connect(_minFiberSpin, &QSpinBox::valueChanged, this, markSettingsStale);
    connect(_view, &FiberMapView::clicked, this, &FiberMapWorkspace::handleSceneClick);
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
                // Relabelling rebuilds the tree too, so it is deferred on the
                // same grounds; the click after it lands on refreshed labels.
                const auto verdict = evaluateDependencies();
                if (verdict.action != StaleVerdict::Action::Fresh ||
                    verdict.relabelOnly) {
                    QMetaObject::invokeMethod(
                        this,
                        [this, verdict]() { applyStaleVerdict(verdict); },
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
    // that may never be opened must cost annotation work nothing, and a slot here
    // would have to either do the work or defer it anyway — an earlier attempt
    // reached the umbilicus resolver from a package-change handler, searching
    // directories and parsing JSON for a tab nobody had looked at.
    //
    // Instead the controller keeps counters that are cheap to bump, and this
    // compares them at the moments it matters: on show, on rebuild, and before
    // acting on a click or a fiber-list selection. The cost of that is deferred
    // detection — a tab already visible when the volume or umbilicus changes keeps
    // its picture until the user does something — which is the accepted trade.

    rebuildScene(tr("press Rebuild layout"));
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

vc3d::annotation::AnnotationFrame FiberMapWorkspace::currentFrame() const
{
    return _controller ? _controller->annotationFrame()
                       : vc3d::annotation::AnnotationFrame{};
}

void FiberMapWorkspace::markStale(const QString& reason)
{
    _stale = true;
    if (_statusLabel) {
        _statusLabel->setText(withCachedUmbilicusStatus(reason));
    }
}

void FiberMapWorkspace::clearLayout(const QString& reason)
{
    // Emptying the layout first is what makes rebuildScene() draw the reason
    // instead of geometry; it also owns tearing down the items, the entries and
    // the highlight, so none of that is repeated here.
    _layout = {};
    _layoutGeneration = 0;
    _layoutFrame = {};
    _layoutScaleSource.reset();
    _layoutUmbilicusFingerprint.clear();
    _layoutPackageGeneration = 0;
    _layoutUmbilicusGeneration = 0;
    _layoutBuilt = false;
    _voxelSizeUm.reset();
    _scrollZMaxVx = 0.0;
    // A fresh fit belongs to the next layout, which is not this one's frame.
    _viewFitted = false;
    // Whatever was cached about the umbilicus described the old package.
    _umbilicusStatusValid = false;
    _stale = true;
    if (_tree) {
        _tree->clear();
    }
    rebuildScene(reason);
    if (_statusLabel) {
        _statusLabel->setText(withCachedUmbilicusStatus(reason));
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
    deps.scaleSource = _layoutScaleSource;
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
        _stale,
        _statusLabel ? _statusLabel->text() : QString());
}

bool FiberMapWorkspace::applyStaleVerdict(const StaleVerdict& verdict)
{
    switch (verdict.action) {
    case StaleVerdict::Action::ClearLayout:
        clearLayout(verdict.reason);
        return true;
    case StaleVerdict::Action::MarkStale:
        if (!_stale) {
            markStale(verdict.reason);
        }
        return true;
    case StaleVerdict::Action::Fresh:
        break;
    }
    if (verdict.relabelOnly) {
        // The geometry is right and the map stays usable; only the physical
        // figures beside it moved.
        _layoutFrame = currentFrame();
        refreshUnitLabels();
    }
    return false;
}

bool FiberMapWorkspace::refreshStaleState()
{
    return applyStaleVerdict(evaluateDependencies());
}

void FiberMapWorkspace::refreshUnitLabels()
{
    _voxelSizeUm = _layoutFrame.voxelSizeUm;
    // The network headers and the status line are the only places a physical
    // figure appears, and both are redrawn from the layout as it stands.
    rebuildScene(_emptyMessage);
    rebuildTree();
    if (_statusLabel) {
        _statusLabel->setText(withCachedUmbilicusStatus(tr("layout current")));
    }
}

void FiberMapWorkspace::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    // Becoming visible is the first of the three moments a stale layout has to
    // be caught; the others are a rebuild and any attempt to act on the map.
    if (refreshStaleState()) {
        return;
    }
    // Nothing built yet: this is where the umbilicus state gets looked up, so a
    // package that will not unroll says so before the user presses Rebuild.
    if (_layout.networks.empty() && _statusLabel) {
        _statusLabel->setText(withCachedUmbilicusStatus(tr("press Rebuild layout")));
    }
}

void FiberMapWorkspace::rebuildLayout()
{
    if (!_controller) {
        return;
    }
    LineAnnotationController::FiberMapSnapshot snapshot = _controller->fiberMapSnapshot();
    _layoutGeneration = snapshot.generation;
    // Everything the snapshot was derived from, so a later check compares against
    // what this build actually used rather than against whatever is current. The
    // frame comes from the snapshot rather than a second derivation for that
    // reason; only the umbilicus token has to be read separately.
    _layoutFrame = snapshot.frame;
    // Which reading produced the scale, so a later voxel-size change can be told
    // apart from one that actually moved the geometry.
    _layoutScaleSource = snapshot.umbilicusScaleSource;
    _layoutUmbilicusFingerprint = _controller->umbilicusFingerprint();
    _layoutPackageGeneration = _controller->packageGeneration();
    _layoutUmbilicusGeneration = _controller->umbilicusGeneration();
    // Set before the build so an empty result still counts as built: it was
    // derived from these dependencies and goes out of date with them.
    _layoutBuilt = true;
    _stale = false;

    // The snapshot's geometry is handed straight to the layout; every fiber of
    // the package is in it, so a second copy is worth avoiding.
    std::vector<vc3d::fiber_map::InputFiber> inputs;
    inputs.reserve(snapshot.fibers.size());
    for (auto& fiber : snapshot.fibers) {
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
            input.links.push_back(vc3d::fiber_map::InputLink{link.controlPointIndex,
                                                             link.branchFiberId,
                                                             link.branchControlPointIndex,
                                                             link.pending});
        }
        inputs.push_back(std::move(input));
    }

    _voxelSizeUm = snapshot.voxelSizeUm;

    vc3d::fiber_map::LayoutParams params;
    params.minFibers = _minFiberSpin->value();
    params.maxNetworks = _topNetworkSpin->value();
    // The layout is unit-free, so the physical intents behind its tuning lengths
    // are converted here — once the voxel size is known, exactly as documented on
    // LayoutParams. Left alone when it is not, so the defaults (the same intents
    // at 2.4 µm) stand in and the map still lays out sensibly.
    if (_voxelSizeUm) {
        const double vxPerCm = sceneVxPerCm();
        params.smoothVx = 0.12 * vxPerCm;         // 1.2 mm arclength sigma
        params.resampleStepVx = 0.025 * vxPerCm;  // 0.025 cm resample step
        params.minPadXVx = 2.2 * vxPerCm;         // 2.2 cm label pad across
        params.minPadYVx = 1.6 * vxPerCm;         // 1.6 cm label pad up
        params.panelTickVx = 5.0 * vxPerCm;       // 5 cm winding-label grid
        params.minGapVx = 1.0 * vxPerCm;          // 1 cm minimum panel gap
    }
    _layout = vc3d::fiber_map::buildLayout(inputs, snapshot.umbilicusCenters, params);
    // Scene space is voxels and the slice count already is one, so the scroll
    // extent needs no voxel size at all.
    _scrollZMaxVx = snapshot.annotationZSlices > 0
        ? static_cast<double>(snapshot.annotationZSlices)
        : 0.0;

    QString emptyMessage;
    if (snapshot.fibers.empty()) {
        emptyMessage = tr("no fibers");
    } else if (snapshot.umbilicusCenters.empty()) {
        // The resolver's own words when it has any; they name the file it
        // rejected or the candidates it could not choose between.
        emptyMessage = snapshot.umbilicusMessage.isEmpty()
            ? tr("no umbilicus found — cannot unroll")
            : snapshot.umbilicusMessage;
        // Whatever the resolver's complaint was, the way out is the same.
        emptyMessage += QLatin1Char('\n');
        emptyMessage += tr("Attach one via File > Attach Umbilicus…");
    } else if (_layout.networks.empty()) {
        emptyMessage = tr("no networks with ≥ %1 linked fibers").arg(params.minFibers);
    }
    rebuildScene(emptyMessage);
    rebuildTree();

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

    int placedFibers = 0;
    for (const auto& network : _layout.networks) {
        placedFibers += static_cast<int>(network.fibers.size());
    }
    QString status = tr("%1 fibers · %2 networks · %3 suspect links")
                         .arg(placedFibers)
                         .arg(_layout.networks.size())
                         .arg(_layout.suspectLinkCount);
    if (!_voxelSizeUm) {
        // No physical figure on the map means anything, so say why once rather
        // than leave the voxel counts looking like an odd choice of unit.
        status += tr(" · voxel size unknown — lengths in vx");
    }
    if (!snapshot.umbilicusCenters.empty() && !snapshot.umbilicusLabel.isEmpty()) {
        // The controller composes this: which grid the umbilicus indexes,
        // whether that came from the file's own metadata or from the z-span
        // guess, and any frame inconsistency it noticed on the way.
        status += QStringLiteral(" · ") + snapshot.umbilicusLabel;
    }
    _statusLabel->setText(status);

    if (!_viewFitted && !_layout.networks.empty()) {
        _view->fitInView(_contentRect, Qt::KeepAspectRatio);
        _viewFitted = true;
    }
}

void FiberMapWorkspace::rebuildScene(const QString& emptyMessage)
{
    _entries.clear();
    clearControlPointDots();
    _highlightedFiber = 0;
    _scene->clear();

    // Kept so a theme change can rebuild the scene as it stands, without asking
    // the controller for a fresh snapshot.
    _emptyMessage = emptyMessage;

    const FiberMapPalette& theme = activePalette();
    _scene->setBackgroundBrush(theme.surface);

    if (_layout.networks.empty()) {
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
    const double sceneWidth = std::max(_layout.widthVx, 1e-6);
    // The one conversion of this rebuild. Every scene-space size below that was
    // chosen as a physical length goes through it, and nothing else does.
    const double vxPerCm = sceneVxPerCm();
    const qreal crossingDotRadius = kCrossingDotRadiusCm * vxPerCm;
    const qreal crossingDotBounds = kCrossingDotBoundsCm * vxPerCm;
    const qreal suspectRingRadius = kSuspectRingRadiusCm * vxPerCm;
    const qreal suspectRingBounds = kSuspectRingBoundsCm * vxPerCm;
    QFont labelFont = font();
    labelFont.setPointSizeF(8.0);
    QFont headerFont = font();
    headerFont.setPointSizeF(10.5);

    // The scroll floor and ceiling, so the networks read against the volume's
    // own z extent instead of floating on their own; the winding gridlines and
    // the panel grounds span the same range. Without that extent the layout's
    // own y range has to do.
    const bool scrollExtentKnown = _scrollZMaxVx > 0.0;
    const double extentBottomY = scrollExtentKnown ? 0.0 : bottomY;
    const double extentTopY = scrollExtentKnown ? -_scrollZMaxVx : topY;
    const double sceneTopY = std::min(extentTopY, topY);
    const double sceneBottomY = std::max(extentBottomY, bottomY);

    // Link endpoints always live in the same network as the link, and each
    // network registers its fibers before its links are drawn, so the entries
    // built so far always cover both ends.
    const auto hvTagOf = [this](uint64_t fiberId) {
        const auto entry = _entries.constFind(fiberId);
        return entry == _entries.constEnd() ? '?' : entry->fiber.hvTag;
    };

    for (const vc3d::fiber_map::PlacedNetwork& network : _layout.networks) {
        // A hair-lighter ground per panel, spanning exactly the scroll extent:
        // with no break line between panels, the gap between the grounds is
        // what says where one network ends and the next begins, and their
        // top/bottom edges are the scroll ceiling and floor.
        auto* panel = _scene->addRect(
            QRectF(QPointF(network.x0Vx, extentTopY), QPointF(network.x1Vx, extentBottomY)),
            QPen(Qt::NoPen), QBrush(tint(theme.surface, theme.ink, 0.045)));
        panel->setZValue(kPanelZ);

        for (const vc3d::fiber_map::WindingMark& mark : network.windings) {
            auto* line = _scene->addLine(mark.xVx, extentTopY, mark.xVx, extentBottomY);
            QPen pen(theme.winding);
            pen.setWidthF(0.8);
            pen.setCosmetic(true);
            pen.setStyle(Qt::DotLine);
            line->setPen(pen);
            line->setZValue(0.0);
        }

        // The reference radius belongs to the fiber tree's network rows; the map
        // only needs to say which panel this is.
        auto* header = _scene->addSimpleText(
            tr("network %1").arg(network.networkIndex + 1), headerFont);
        header->setBrush(theme.inkSoft);
        pinText(header, QPointF(0.5 * (network.x0Vx + network.x1Vx), topY), 0.0, -36.0, true);

        for (const vc3d::fiber_map::PlacedFiber& placed : network.fibers) {
            FiberEntry entry;
            entry.fiber = placed;
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

            // Label chip at whichever fiber end sits nearest its panel edge
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
                    const bool atRight = (network.x1Vx - right.x()) < (left.x() - network.x0Vx);
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
            }

            const uint64_t fiberId = entry.fiber.id;
            _entries.insert(fiberId, std::move(entry));
        }

        for (const vc3d::fiber_map::PlacedLink& link : network.links) {
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
    }

    // The panel headers hang above the top edge in device pixels, so the scene
    // keeps a slice of room for them above the layout. The scroll extent, when
    // known, is part of what the first-build fit shows.
    const double height = std::max(sceneBottomY - sceneTopY, 1e-6);
    _contentRect = QRectF(0.0, sceneTopY - 0.10 * height, sceneWidth, 1.12 * height);

    // Panning stops at the scene rect, so the rect runs wider than the content:
    // zoomed in, the outermost panels can be dragged away from the edge instead
    // of being pinned to it.
    const double xMargin = std::max(0.25 * sceneWidth, kMinSceneMarginCm * vxPerCm);
    _scene->setSceneRect(_contentRect.adjusted(-xMargin, 0.0, xMargin, 0.0));
}

void FiberMapWorkspace::rebuildTree()
{
    const bool guard = _syncingSelection;
    _syncingSelection = true;
    _tree->clear();
    // The rows carry the map's own colours, so they follow the theme with it;
    // everything else about the tree is the widget palette's business.
    const FiberMapPalette& theme = activePalette();
    for (const vc3d::fiber_map::PlacedNetwork& network : _layout.networks) {
        const int suspectCount = static_cast<int>(
            std::count_if(network.links.begin(), network.links.end(),
                          [](const vc3d::fiber_map::PlacedLink& link) { return link.suspect; }));
        QString title = tr("Network %1 — %2 fibers · r ≈ %3")
                            .arg(network.networkIndex + 1)
                            .arg(network.fibers.size())
                            .arg(formatMapLength(network.rRefVx));
        if (suspectCount > 0) {
            title += tr(" · %1 winding-suspect").arg(suspectCount);
        }
        auto* networkItem = new QTreeWidgetItem(_tree, {title});
        networkItem->setForeground(0, theme.inkSoft);
        // The network line is a header, not a cell of the label column: it runs
        // across the whole row so the columns can stay narrow.
        networkItem->setFirstColumnSpanned(true);
        for (const vc3d::fiber_map::PlacedFiber& fiber : network.fibers) {
            const QString annotationName =
                _controller ? _controller->fiberDisplayName(fiber.id) : QString();
            auto* fiberItem = new QTreeWidgetItem(
                networkItem,
                {fiber.label, QString(QLatin1Char(fiber.hvTag)), annotationName});
            fiberItem->setData(0, Qt::UserRole, QVariant::fromValue<qulonglong>(fiber.id));
            const QColor color = fiberColor(fiber.hvTag, theme);
            for (int column = 0; column < 3; ++column) {
                fiberItem->setForeground(column, color);
            }
        }
        networkItem->setExpanded(true);
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
        if (item->type() != kChipItemType) {
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
    for (int networkRow = 0; networkRow < _tree->topLevelItemCount(); ++networkRow) {
        QTreeWidgetItem* networkItem = _tree->topLevelItem(networkRow);
        for (int fiberRow = 0; fiberRow < networkItem->childCount(); ++fiberRow) {
            QTreeWidgetItem* fiberItem = networkItem->child(fiberRow);
            if (fiberItem->data(0, Qt::UserRole).toULongLong() == fiberId) {
                _tree->setCurrentItem(fiberItem);
                _tree->scrollToItem(fiberItem);
                _syncingSelection = guard;
                return;
            }
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

void FiberMapWorkspace::setHighlightedFiber(uint64_t fiberId)
{
    if (_highlightedFiber == fiberId) {
        return;
    }
    const FiberMapPalette& theme = activePalette();
    if (const auto previous = _entries.constFind(_highlightedFiber);
        previous != _entries.constEnd()) {
        const QColor color = fiberColor(previous->fiber.hvTag, theme);
        if (previous->tracedItem) {
            previous->tracedItem->setPen(cosmeticPen(color, kTracedWidth));
            previous->tracedItem->setZValue(kFiberZ);
        }
        if (previous->interpolatedItem) {
            previous->interpolatedItem->setPen(
                interpolatedPen(tint(color, theme.surface, 0.45), kInterpolatedWidth));
            previous->interpolatedItem->setZValue(kFiberZ);
        }
    }
    clearControlPointDots();
    _highlightedFiber = fiberId;

    const auto entry = _entries.constFind(fiberId);
    if (entry == _entries.constEnd()) {
        return;
    }
    const QColor color = fiberColor(entry->fiber.hvTag, theme);
    if (entry->tracedItem) {
        entry->tracedItem->setPen(cosmeticPen(color, kTracedHighlightWidth));
        entry->tracedItem->setZValue(kHighlightZ);
    }
    if (entry->interpolatedItem) {
        entry->interpolatedItem->setPen(
            interpolatedPen(tint(color, theme.surface, 0.45), kInterpolatedHighlightWidth));
        entry->interpolatedItem->setZValue(kHighlightZ);
    }
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
    QMenu menu(this);
    QAction* action = menu.addAction(tr("Go to control point %1 in %2")
                                        .arg(bestIndex)
                                        .arg(_controller->fiberDisplayName(fiberId)));
    // menu.exec() runs a nested event loop, so the fiber set can change while the
    // menu is open. Both captures matter: the runtime id is resolved from the file
    // name because a reload reassigns ids, and the generation is compared because
    // bestIndex indexes the control points as they were when the menu was built —
    // an edit in between could have made it mean a different point, or none.
    const uint64_t menuGeneration = _controller->fiberDataGeneration();
    const vc3d::annotation::AnnotationFrame menuFrame = currentFrame();
    const QString menuUmbilicus = _controller->umbilicusFingerprint();
    const uint64_t menuPackage = _controller->packageGeneration();
    const uint64_t menuUmbilicusGeneration = _controller->umbilicusGeneration();
    connect(action, &QAction::triggered, this,
            [this, fileName, bestIndex, menuGeneration, menuFrame, menuUmbilicus,
             menuPackage, menuUmbilicusGeneration]() {
                if (!_controller) {
                    return;
                }
                // Every dependency, not just the fibers: a frame change leaves the
                // generation untouched while making the map's geometry — and so
                // the point this index was chosen from — mean something else.
                //
                // Compared rather than routed through refreshStaleState(), which
                // can clear the layout: this runs inside menu.exec()'s nested
                // event loop, and tearing the scene down there would destroy items
                // while the press that opened the menu is still unwinding. The
                // clear happens at the next natural moment instead.
                // The package and umbilicus counters are here too: two packages
                // can share a frame, a fingerprint and a generation value, so
                // without them this could resolve an old file name inside a new
                // package and navigate using an index chosen from the old one.
                const bool changed =
                    _controller->fiberDataGeneration() != menuGeneration ||
                    _controller->packageGeneration() != menuPackage ||
                    _controller->umbilicusGeneration() != menuUmbilicusGeneration ||
                    !vc3d::annotation::sameAnnotationGrid(currentFrame(), menuFrame) ||
                    _controller->umbilicusFingerprint() != menuUmbilicus;
                if (changed) {
                    markStale(tr("Map changed — press Rebuild layout"));
                    Logger()->warn(
                        "Fiber map: dependencies changed while the menu was open; "
                        "not navigating to control point {} in {}",
                        bestIndex,
                        fileName);
                    return;
                }
                const uint64_t target = _controller->fiberIdForFileName(fileName);
                if (target == 0) {
                    markStale(tr("Fibers changed — press Rebuild layout"));
                    Logger()->warn("Fiber map: {} is no longer loaded; not navigating",
                                   fileName);
                    return;
                }
                emit openFiberAtControlPointRequested(target, bestIndex);
            });
    menu.exec(globalPos);
}
