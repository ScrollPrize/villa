#include "CVolumeViewerView.hpp"

#include <QGraphicsView>
#include <QGraphicsProxyWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPainterPath>
#include <QCoreApplication>
#include <QFontMetricsF>
#include <QPolygonF>
#include <cmath>
#include <limits>
#include <algorithm>
#include "vc/core/util/PlaneSurface.hpp"


CVolumeViewerView::CVolumeViewerView(QWidget* parent) : QGraphicsView(parent)
{
    setMouseTracking(true);
    if (viewport()) {
        viewport()->setMouseTracking(true);
    }
};

double CVolumeViewerView::chooseNiceLength(double nominal) const
{
    double expn = std::floor(std::log10(nominal));
    double base = std::pow(10.0, expn);
    double d    = nominal / base;
    if (d < 2.0)      return 1.0   * base;
    else if (d < 5.0) return 2.0   * base;
    else              return 5.0   * base;
}

CVolumeViewerView::ScaleBarLabel CVolumeViewerView::formatScaleBarLength(double barUm)
{
    ScaleBarLabel label;
    label.displayLength = barUm;
    label.unit = QStringLiteral(" µm");
    if (barUm < 1.0) {
        label.displayLength = barUm * 1000.0;
        label.unit = QStringLiteral(" nm");
    } else if (barUm >= 1.0e6) {
        label.displayLength = barUm / 1.0e6;
        label.unit = QStringLiteral(" m");
    } else if (barUm >= 1.0e4) {
        label.displayLength = barUm / 1.0e4;
        label.unit = QStringLiteral(" cm");
    } else if (barUm >= 1.0e3) {
        label.displayLength = barUm / 1.0e3;
        label.unit = QStringLiteral(" mm");
    }
    label.text = QString::number(label.displayLength, 'g', 4) + label.unit;
    return label;
}

void CVolumeViewerView::setCoordinateFrame(PlaneSurface* plane)
{
    if (!plane) {
        return;
    }
    _showFrame = true;
    cv::Matx33d M = plane->frame();
    _frame[0] = cv::Vec3f(M(0,0), M(1,0), M(2,0));
    _frame[1] = cv::Vec3f(M(0,1), M(1,1), M(2,1));
    _frame[2] = cv::Vec3f(M(0,2), M(1,2), M(2,2));
    update();
}

void CVolumeViewerView::showCoordinateFrame(bool visible)
{
    _showFrame = visible;
    update();
}

void CVolumeViewerView::drawForeground(QPainter* p, const QRectF& sceneRect)
{
    QGraphicsView::drawForeground(p, sceneRect);
    drawCoordinateFrame(p);
    drawTiltHandle(p);
    drawGrid(p);
    drawScaleBar(p);
    drawLineProfile(p);
    drawHistogram(p);
    drawLoupe(p);
}

void CVolumeViewerView::drawCoordinateFrame(QPainter* p) const
{
    if (!_showFrame) {
        return;
    }

    const double radius = 40.0;
    const double head = 10.0;
    const double margin = 10.0;
    const QColor colors[3] = {QColor(230, 30, 30), QColor(30, 230, 30), QColor(30, 40, 220)};
    const QString labels[3] = {QStringLiteral("X"), QStringLiteral("Y"), QStringLiteral("Z")};
    
    const QPointF center(viewport()->width() - margin - radius, margin + radius);

    // Sort by depth, furthest first. Higher z coord = further into the screen.
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [this](int a, int b) {
        return _frame[a][2] > _frame[b][2];
    });

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);

    for (int i : order) {
        const QPointF dir = QPointF(_frame[i][0], _frame[i][1]);
        const QPointF tip(center.x() + dir.x() * radius, center.y() + dir.y() * radius);
        const double length = std::hypot(tip.x() - center.x(), tip.y() - center.y());
        const QColor& color = colors[i];

        if (length < 2.0) {
            if (_frame[i][2] > 0.5) {
                // into the screen
                p->setPen(QPen(color, 1.0));
                p->setBrush(Qt::NoBrush);
            } else {
                // out of the screen
                p->setPen(Qt::NoPen);
                p->setBrush(color);
            }
            p->drawEllipse(center, head/2, head/2);
            p->setPen(color);
            p->drawText(QPointF(center.x() + 5.0, center.y() - 5.0), labels[i]);
            continue;
        }

        const double ux = (tip.x() - center.x()) / length;
        const double uy = (tip.y() - center.y()) / length;

        p->setPen(QPen(color, 1.6, Qt::SolidLine, Qt::RoundCap));
        p->setBrush(Qt::NoBrush);
        p->drawLine(center, tip);

        QPolygonF arrow;
        arrow << tip
              << QPointF(tip.x() - ux * head - uy * head * 0.45,
                         tip.y() - uy * head + ux * head * 0.45)
              << QPointF(tip.x() - ux * head + uy * head * 0.45,
                         tip.y() - uy * head - ux * head * 0.45);
        if (_frame[i][2] > 0.01) {
            // into the screen
            p->setPen(QPen(color, 1.0));
            p->setBrush(Qt::NoBrush);
        } else {
            // out of the screen
            p->setPen(Qt::NoPen);
            p->setBrush(color);
        }
        p->drawPolygon(arrow);
    }

    p->restore();
}

void CVolumeViewerView::setTiltHandle(TiltHandleMode mode, QPointF tilt)
{
    _tiltHandleMode = mode;
    _tiltHandleValue = tilt;
    if (_tiltHandleMode == TiltHandleMode::SemiCircleX) {
        _tiltHandleValue.setY(0.0);
    } else if (_tiltHandleMode == TiltHandleMode::SemiCircleY) {
        _tiltHandleValue.setX(0.0);
    }
    update();
}

QRectF CVolumeViewerView::tiltHandleRect() const
{
    constexpr double kSize = 46.0;
    constexpr double kMargin = 14.0;
    return QRectF(viewport()->width() - kSize - kMargin,
                  viewport()->height() - kSize - kMargin,
                  kSize,
                  kSize);
}

bool CVolumeViewerView::pointInTiltHandle(const QPointF& viewportPos) const
{
    if (_tiltHandleMode == TiltHandleMode::Hidden) {
        return false;
    }
    return tiltHandleRect().adjusted(-5.0, -5.0, 5.0, 5.0).contains(viewportPos);
}

bool CVolumeViewerView::pointInSceneWidget(const QPointF& viewportPos) const
{
    QGraphicsItem* item = itemAt(viewportPos.toPoint());
    while (item) {
        if (qgraphicsitem_cast<QGraphicsProxyWidget*>(item)) {
            return true;
        }
        item = item->parentItem();
    }
    return false;
}

QPointF CVolumeViewerView::tiltFromHandlePos(const QPointF& viewportPos) const
{
    const QRectF r = tiltHandleRect();
    const QPointF c = r.center();
    const double radius = r.width() * 0.38;
    const double nx = std::clamp((viewportPos.x() - c.x()) / radius, -1.0, 1.0);
    const double ny = std::clamp((viewportPos.y() - c.y()) / radius, -1.0, 1.0);

    if (_tiltHandleMode == TiltHandleMode::Square) {
        const double len = std::hypot(nx, ny);
        if (len > 1.0) {
            return QPointF(nx / len, ny / len);
        }
        return QPointF(nx, ny);
    }

    if (_tiltHandleMode == TiltHandleMode::SemiCircleX) {
        return QPointF(nx, 0.0);
    }
    if (_tiltHandleMode == TiltHandleMode::SemiCircleY) {
        return QPointF(0.0, nx);
    }
    return QPointF(0.0, 0.0);
}

void CVolumeViewerView::drawTiltHandle(QPainter* p) const
{
    if (_tiltHandleMode == TiltHandleMode::Hidden) {
        return;
    }

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);

    const QRectF r = tiltHandleRect();
    const QPointF c = r.center();
    const double radius = r.width() * 0.38;

    QPen cyan(QColor(0, 220, 255), 1.6, Qt::DashLine);
    cyan.setCosmetic(true);
    p->setPen(cyan);
    p->setBrush(Qt::NoBrush);

    QPointF dot = c;
    if (_tiltHandleMode == TiltHandleMode::Square) {
        const QRectF box(c.x() - radius, c.y() - radius, radius * 2.0, radius * 2.0);
        p->drawRect(box);
        dot = QPointF(c.x() + _tiltHandleValue.x() * radius,
                      c.y() + _tiltHandleValue.y() * radius);
    } else {
        const QRectF arc(c.x() - radius, c.y() - radius, radius * 2.0, radius * 2.0);
        p->drawArc(arc, 0, 180 * 16);
        const double v = _tiltHandleMode == TiltHandleMode::SemiCircleX
            ? _tiltHandleValue.x()
            : _tiltHandleValue.y();
        const double clamped = std::clamp(v, -1.0, 1.0);
        constexpr double kPi = 3.14159265358979323846;
        const double theta = (1.0 - clamped) * kPi * 0.5;
        dot = QPointF(c.x() + std::cos(theta) * radius,
                      c.y() - std::sin(theta) * radius);
    }

    p->setPen(QPen(QColor(40, 28, 0), 1.0));
    p->setBrush(QColor(255, 214, 42));
    p->drawEllipse(dot, 4.0, 4.0);
    p->restore();
}

void CVolumeViewerView::drawScaleBar(QPainter* p) const
{
    if (property("vc_hide_scalebar").toBool()) {
        return;
    }
    const double m11 = transform().m11();
    const int vpW = viewport()->width();
    const int vpH = viewport()->height();

    // Recompute cached scalebar only when inputs change
    if (_cachedM11 != m11 || _cachedVpW != vpW ||
        _cachedVpH != vpH || _cachedVx != m_vx ||
        _cachedPhysicalUnits != m_physicalUnits) {
        _cachedM11 = m11;
        _cachedVpW = vpW;
        _cachedVpH = vpH;
        _cachedVx = m_vx;
        _cachedPhysicalUnits = m_physicalUnits;

        _cachedFont = p->font();
        _cachedFont.setPointSizeF(12);

        // Logical pixels, to match the painter after resetTransform().
        double pxPerScene = m11;
        double pxPerUm = pxPerScene / m_vx;
        double wPx = vpW;
        double wUm = wPx / pxPerUm;
        double barUm = chooseNiceLength(wUm / 4.0);
        _cachedBarPx = barUm * pxPerUm;

        _cachedBarLabel = m_physicalUnits
            ? formatScaleBarLength(barUm).text
            : QString::number(barUm) + QStringLiteral(" vx");
    }

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    p->setPen(QPen(Qt::red, 2));
    p->setFont(_cachedFont);

    constexpr int M = 10;
    int bottom = vpH - M;
    p->drawLine(M, bottom, static_cast<int>(M + _cachedBarPx), bottom);
    p->drawText(M, bottom - 5, _cachedBarLabel);
    p->restore();
}

void CVolumeViewerView::drawBackground(QPainter* painter, const QRectF& /*rect*/)
{
    painter->resetTransform();
    painter->fillRect(viewport()->rect(), Qt::black);
    if (_directFb && !_directFb->isNull()) {
        const QRectF target(
            _directFbOffset,
            QSizeF(_directFb->width() * _directFbScale,
                   _directFb->height() * _directFbScale));
        painter->drawImage(target, *_directFb);
    }
}

void CVolumeViewerView::paintEvent(QPaintEvent* event)
{
    QGraphicsView::paintEvent(event);
}

void CVolumeViewerView::wheelEvent(QWheelEvent *event)
{
    if (pointInSceneWidget(event->position())) {
        QGraphicsView::wheelEvent(event);
        return;
    }

    // macOS turns Shift+scroll into a horizontal scroll gesture at the system
    // level, so the delta arrives on the x axis with y left at zero.
    const QPoint angleDelta = event->angleDelta();
    int delta = angleDelta.y();
    if (delta == 0 && event->modifiers().testFlag(Qt::ShiftModifier)) {
        delta = angleDelta.x();
    }

    // A partial notch gathered under one modifier set must not complete a
    // step under another: the line annotation current cut binds Shift and
    // Ctrl+Shift to different motions, so a remainder carried across the change
    // would fire the wrong one.
    if (event->modifiers() != _wheelAccumModifiers) {
        _wheelAccum = 0;
        _wheelAccumModifiers = event->modifiers();
    }
    _wheelAccum += delta;
    constexpr int kStepThreshold = 120;  // one notch = one step
    int steps = _wheelAccum / kStepThreshold;
    if (steps == 0) {
        event->accept();
        return;
    }
    _wheelAccum -= steps * kStepThreshold;

    QPointF vp_loc = viewport()->mapFromGlobal(event->globalPosition());
    sendZoom(steps, vp_loc, event->modifiers());
    event->accept();
}

void CVolumeViewerView::mouseReleaseEvent(QMouseEvent *event)
{
    if ((_profileDragging || _profileDragCancelled) && event->button() == Qt::LeftButton) {
        // The release ends the profile's drag, finished or cancelled; the tools never
        // see it.
        if (_profileDragging && _profileLine) {
            _profileLine->setP2(event->position());
            if (_profileLine->length() < 3.0) {
                _profileLine.reset();  // an armed click clears the line
            }
        }
        _profileDragging = false;
        _profileDragCancelled = false;
        viewport()->update();
        event->accept();
        return;
    }

    if (_sceneWidgetMouseCapture) {
        QGraphicsView::mouseReleaseEvent(event);
        if (event->buttons() == Qt::NoButton) {
            _sceneWidgetMouseCapture = false;
        }
        event->accept();
        return;
    }

    if (_tiltHandleDragging) {
        if (event->button() == Qt::LeftButton) {
            emit sendTiltHandleChanged(tiltFromHandlePos(event->position()));
            _tiltHandleDragging = false;
            unsetCursor();
            event->accept();
            return;
        }
    }

    if (event->button() == Qt::MiddleButton)
    {
        if (_middleButtonPanEnabled)
        {
            setCursor(Qt::ArrowCursor);
            event->accept();
            if (_regular_pan) {
                _regular_pan = false;
                sendPanRelease(event->button(), event->modifiers());
            }
        }
        else
        {
            QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
            QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
            sendMouseRelease(scene_loc, event->button(), event->modifiers());
            event->accept();
        }
        return;
    }
    else if (event->button() == Qt::RightButton)
    {
        if (_right_button_mouse_forwarded) {
            QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
            QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
            sendMouseRelease(scene_loc, event->button(), event->modifiers());
            _right_button_mouse_forwarded = false;
            event->accept();
            return;
        }

        setCursor(Qt::ArrowCursor);
        event->accept();
        if (_regular_pan) {
            _regular_pan = false;
            sendPanRelease(event->button(), event->modifiers());
        }
        return;
    }
    else if (event->button() == Qt::LeftButton)
    {
        QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
        QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
        sendVolumeClicked(scene_loc, event->button(), event->modifiers());
        sendMouseRelease(scene_loc, event->button(), event->modifiers());
        
        event->accept();
        return;
    }
    event->ignore();
}

void CVolumeViewerView::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Escape) {
        // Escape that cancels the profile belongs to it alone, auto-repeats and release
        // included, so Measure and the other tools keep their state.
        if (_profileOwnsEscape && event->isAutoRepeat()) {
            event->accept();
            return;
        }
        if (_profileArmed || _profileDragging) {
            _profileOwnsEscape = true;
            cancelProfileGesture();
            emit profileLineCancelled();
            event->accept();
            return;
        }
        _profileOwnsEscape = false;
    }
    switch (event->key()) {
    case Qt::Key_Escape:
    case Qt::Key_Left:
    case Qt::Key_Right:
    case Qt::Key_Up:
    case Qt::Key_Down:
        emit sendKeyPress(event->key(), event->modifiers());
        event->accept();
        return;
    default:
        break;
    }

    QGraphicsView::keyPressEvent(event);
    event->ignore();  // let unhandled keys propagate to CWindow
}

void CVolumeViewerView::keyReleaseEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Escape && _profileOwnsEscape) {
        if (!event->isAutoRepeat()) {
            _profileOwnsEscape = false;
        }
        event->accept();
        return;
    }
    emit sendKeyRelease(event->key(), event->modifiers());
    QGraphicsView::keyReleaseEvent(event);
    event->ignore();  // let unhandled releases propagate to CWindow
}

void CVolumeViewerView::mousePressEvent(QMouseEvent *event)
{
    if (pointInSceneWidget(event->position())) {
        _sceneWidgetMouseCapture = true;
        QGraphicsView::mousePressEvent(event);
        event->accept();
        return;
    }

    if (_profileDragCancelled) {
        if (event->button() == Qt::LeftButton || !(event->buttons() & Qt::LeftButton)) {
            _profileDragCancelled = false;  // the cancelled drag's release never arrived
        } else {
            event->accept();  // other buttons pressed during a cancelled drag stay with it
            return;
        }
    }

    if (_profileArmed && event->button() == Qt::LeftButton &&
        !pointInTiltHandle(event->position())) {
        // An armed view takes this one left drag for the profile line; every other
        // press goes to the tools as usual.
        armProfileLine(false);
        _profileDragging = true;
        _profileLine = QLineF(event->position(), event->position());
        emit profileLineDrawStarted();
        viewport()->update();
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton && pointInTiltHandle(event->position())) {
        _tiltHandleDragging = true;
        setCursor(Qt::CrossCursor);
        emit sendTiltHandleChanged(tiltFromHandlePos(event->position()));
        event->accept();
        return;
    }

    if (event->button() == Qt::MiddleButton)
    {
        if (_middleButtonPanEnabled)
        {
            _regular_pan = true;
            _last_pan_position = QPoint(event->position().x(), event->position().y());
            sendPanStart(event->button(), event->modifiers());
            setCursor(Qt::ClosedHandCursor);
            event->accept();
        }
        else
        {
            QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
            QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
            sendMousePress(scene_loc, event->button(), event->modifiers());
            event->accept();
        }
        return;
    }
    else if (event->button() == Qt::RightButton)
    {
        if (event->modifiers().testFlag(Qt::ShiftModifier)) {
            QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
            QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
            _right_button_mouse_forwarded = true;
            sendMousePress(scene_loc, event->button(), event->modifiers());
            event->accept();
            return;
        }

        if (event->modifiers().testFlag(Qt::ControlModifier)) {
            QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
            QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
            emit sendAnnotationContextMenuRequested(scene_loc,
                                                    event->globalPosition().toPoint(),
                                                    event->modifiers());
            event->accept();
            return;
        }

        _regular_pan = true;
        _last_pan_position = QPoint(event->position().x(), event->position().y());
        sendPanStart(event->button(), event->modifiers());
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    else if (event->button() == Qt::LeftButton)
    {
        QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
        QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
        
        sendMousePress(scene_loc, event->button(), event->modifiers());
        event->accept();
        return;
    }
    event->ignore();
}

void CVolumeViewerView::mouseDoubleClickEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && pointInTiltHandle(event->position())) {
        _tiltHandleDragging = false;
        emit sendTiltHandleReset();
        event->accept();
        return;
    }
    if (event->button() == Qt::LeftButton) {
        QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
        QPointF scene_loc = mapToScene({int(global_loc.x()), int(global_loc.y())});
        emit sendMouseDoubleClick(scene_loc, event->button(), event->modifiers());
        event->accept();
        return;
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

void CVolumeViewerView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    emit sendResized();
}

void CVolumeViewerView::mouseMoveEvent(QMouseEvent *event)
{
    _lastCursorVp = event->position();
    if (_hud.loupe) {
        viewport()->update();
    }
    if (_profileDragging && _profileLine) {
        _profileLine->setP2(event->position());
        viewport()->update();
        event->accept();
        return;
    }
    if (_profileDragCancelled) {
        if (event->buttons() & Qt::LeftButton) {
            event->accept();
            return;
        }
        _profileDragCancelled = false;
    }

    if (_sceneWidgetMouseCapture) {
        QGraphicsView::mouseMoveEvent(event);
        event->accept();
        return;
    }

    if (_tiltHandleDragging) {
        emit sendTiltHandleChanged(tiltFromHandlePos(event->position()));
        event->accept();
        return;
    }

    if (_regular_pan)
    {
        _last_pan_position = QPoint(event->position().x(), event->position().y());
        event->accept();
    }

    QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
    QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});

    emit sendCursorMove(scene_loc);

    // mouse move events must be forwarded even without a pressed button so tools that
    // rely on hover state (e.g. segmentation editing) receive continuous
    // volume coordinates.
    sendMouseMove(scene_loc, event->buttons(), event->modifiers());
}

void CVolumeViewerView::leaveEvent(QEvent *event)
{
    _lastCursorVp = QPointF(-1.0, -1.0);
    if (_hud.loupe) {
        viewport()->update();
    }
    emit sendMouseLeftView();
    QGraphicsView::leaveEvent(event);
}

bool CVolumeViewerView::viewportEvent(QEvent* event)
{
    if (event && event->type() == QEvent::Leave) {
        emit sendMouseLeftView();
    }
    return QGraphicsView::viewportEvent(event);
}


// ---- Measurement overlays (ViewerHud.hpp) ----------------------------------------------

void CVolumeViewerView::setHudOptions(const vc3d::hud::Options& options)
{
    if (_hud == options) {
        return;
    }
    _hud = options;
    if (!_hud.lineProfile) {
        // A drag in progress keeps its release, so the tools never get half a gesture.
        _profileDragCancelled = _profileDragCancelled || _profileDragging;
        _profileDragging = false;
        armProfileLine(false);
    }
    viewport()->update();
}

void CVolumeViewerView::setProfileLine(std::optional<QLineF> line)
{
    _profileLine = line;
    _profileDragCancelled = _profileDragCancelled || _profileDragging;
    _profileDragging = false;
    viewport()->update();
}

// Application-wide, so it sees the profile gesture's events before the filters other
// tools install on the view or its viewport (Spiral point placement, the line
// annotation dialog). It removes itself at the first key or mouse event that finds no
// gesture active; other events (a cursor change while a gesture ends) never decide it.
class ProfileGestureFilter : public QObject {
public:
    explicit ProfileGestureFilter(CVolumeViewerView* view) : QObject(view), _view(view) {}
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        switch (event->type()) {
        case QEvent::ShortcutOverride:
        case QEvent::KeyPress:
        case QEvent::KeyRelease:
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonDblClick:
        case QEvent::MouseMove:
        case QEvent::MouseButtonRelease:
            break;
        default:
            return false;
        }
        if (!_view->profileGestureActive()) {
            QCoreApplication::instance()->removeEventFilter(this);
            return false;
        }
        return _view->takeProfileGestureEvent(watched, event);
    }

private:
    CVolumeViewerView* _view;
};

bool CVolumeViewerView::takeProfileGestureEvent(QObject* watched, QEvent* event)
{
    if (!event || !watched || !watched->isWidgetType())
        return false;
    auto* widget = static_cast<QWidget*>(watched);
    switch (event->type()) {
    case QEvent::ShortcutOverride:
    case QEvent::KeyPress:
    case QEvent::KeyRelease: {
        auto* key = static_cast<QKeyEvent*>(event);
        if (key->key() != Qt::Key_Escape || (widget != this && !isAncestorOf(widget)))
            return false;
        const bool repeatOwned = _profileOwnsEscape && key->isAutoRepeat();
        if (event->type() == QEvent::KeyRelease) {
            if (!_profileOwnsEscape)
                return false;
            keyReleaseEvent(key);
            return true;
        }
        if (!(repeatOwned || _profileArmed || _profileDragging))
            return false;
        if (event->type() == QEvent::ShortcutOverride) {
            event->accept();  // no Escape shortcut fires; the press comes here instead
            return true;
        }
        keyPressEvent(key);
        return true;
    }
    case QEvent::MouseButtonPress:
    case QEvent::MouseMove:
    case QEvent::MouseButtonRelease: {
        if (widget != viewport())
            return false;
        auto* mouse = static_cast<QMouseEvent*>(event);
        bool take = false;
        if (event->type() == QEvent::MouseMove) {
            take = _profileDragging || (_profileDragCancelled && (mouse->buttons() & Qt::LeftButton));
        } else if (event->type() == QEvent::MouseButtonRelease) {
            take = (_profileDragging || _profileDragCancelled) && mouse->button() == Qt::LeftButton;
        } else {
            take = _profileDragCancelled ||
                   (_profileArmed && mouse->button() == Qt::LeftButton &&
                    !pointInSceneWidget(mouse->position()) && !pointInTiltHandle(mouse->position()));
        }
        if (!take)
            return false;
        viewportEvent(event);
        return true;
    }
    default:
        return false;
    }
}

void CVolumeViewerView::armProfileLine(bool armed)
{
    armed = armed && _hud.lineProfile;
    if (_profileArmed == armed) {
        return;
    }
    _profileArmed = armed;
    if (armed) {
        if (!_gestureFilter)
            _gestureFilter = new ProfileGestureFilter(this);
        // Installed again on each arming, so it runs first among application filters.
        QCoreApplication::instance()->installEventFilter(_gestureFilter);
        setCursor(Qt::CrossCursor);
    } else if (!_tiltHandleDragging && !_regular_pan) {
        unsetCursor();
    }
}

void CVolumeViewerView::cancelProfileGesture()
{
    armProfileLine(false);
    if (_profileDragging) {
        // The line being drawn is dropped; the rest of the drag stays with the profile.
        _profileDragging = false;
        _profileDragCancelled = true;
        _profileLine.reset();
    }
    viewport()->update();
}

void CVolumeViewerView::setMeasurementFrameSource(std::function<FrameResult()> source)
{
    _frameSource = std::move(source);
    _histogramSerial = 0;
    viewport()->update();
}

CVolumeViewerView::FrameResult CVolumeViewerView::measurementFrame(bool needSamples) const
{
    if (!_frameSource) {
        return {std::nullopt, tr("no frame to measure")};
    }
    // A retained frame slid or scaled to follow a pan or zoom is not the live view.
    if (_directFbScale != 1.0 || _directFbOffset != QPointF(0.0, 0.0)) {
        return {std::nullopt, tr("updating")};
    }
    FrameResult result = _frameSource();
    if (result.frame && (result.frame->values.empty() ||
                         result.frame->values.cols != viewport()->width() ||
                         result.frame->values.rows != viewport()->height() ||
                         (needSamples && !result.frame->samples))) {
        return {std::nullopt, tr("updating")};
    }
    return result;
}

namespace {
vc3d::hud::Raster rasterOf(const CVolumeViewerView::MeasurementFrame& f)
{
    vc3d::hud::Raster r;
    r.values = f.values.ptr<std::uint8_t>(0);
    r.width = f.values.cols;
    r.height = f.values.rows;
    r.valuesStride = f.values.step[0];
    if (!f.coverage.empty() && f.coverage.size() == f.values.size()) {
        r.coverage = f.coverage.ptr<std::uint8_t>(0);
        r.coverageStride = f.coverage.step[0];
    }
    return r;
}

std::array<double, 3> toArray(const cv::Vec3f& v)
{
    return {double(v[0]), double(v[1]), double(v[2])};
}

// The volume position of a frame pixel from the frame's own geometry, NaN when the
// pixel has none (off the surface, a hole).
std::array<double, 3> framePosition(const CVolumeViewerView::MeasurementFrame& f, QPointF p)
{
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 3> v = f.planeMap ? f.planeMap->at(p)
                              : f.toVolume ? toArray(f.toVolume(p))
                                           : std::array<double, 3>{nan, nan, nan};
    if (!vc3d::hud::validPosition(v)) {
        v = {nan, nan, nan};
    }
    return v;
}

QString compositeUnavailable(const QString& composite)
{
    return QCoreApplication::translate("CVolumeViewerView", "the view shows a %1, not one slice").arg(composite);
}
}  // namespace

CVolumeViewerView::ProfileReadout CVolumeViewerView::profileReadout() const
{
    ProfileReadout r;
    if (!_profileLine) {
        r.unavailable = tr("no profile line");
        return r;
    }
    const FrameResult fr = measurementFrame();
    if (!fr.frame) {
        r.unavailable = fr.unavailable;
        return r;
    }
    const MeasurementFrame& frame = *fr.frame;
    if (!frame.composite.isEmpty()) {
        // A slab composite's value can come from any of its slices, so it has no
        // single position to plot against.
        r.unavailable = compositeUnavailable(frame.composite);
        return r;
    }
    // The viewport shows this frame unscaled, so viewport pixels are frame pixels.
    const QPointF a = _profileLine->p1();
    const QPointF b = _profileLine->p2();
    r.values = vc3d::hud::sampleLine(rasterOf(frame), a, b);
    const int n = static_cast<int>(r.values.size());

    std::vector<std::array<double, 3>> positions(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[std::size_t(i)] = framePosition(frame, vc3d::hud::samplePoint(a, b, i, n));
    }
    r.physical = frame.voxelSizeUm > 0.0;
    r.alongSurface = !frame.planeMap;
    r.complete = frame.complete;
    r.arc = vc3d::hud::cumulativeLength(positions);
    if (r.physical) {
        for (double& d : r.arc) {
            d *= frame.voxelSizeUm;
        }
    }
    if (!r.arc.empty() && std::isfinite(r.arc.back())) {
        r.length = r.arc.back();
    }
    if (!positions.empty() && vc3d::hud::validPosition(positions.front())) {
        r.start = positions.front();
    }
    if (!positions.empty() && vc3d::hud::validPosition(positions.back())) {
        r.end = positions.back();
    }

    double sum = 0, sum2 = 0;
    int finite = 0;
    for (float v : r.values) {
        if (!std::isnan(v)) {
            sum += v;
            sum2 += double(v) * v;
            ++finite;
        }
    }
    r.missing = n - finite;
    const double sd = finite > 1 ? std::sqrt(std::max(0.0, sum2 / finite - (sum / finite) * (sum / finite))) : 0.0;
    r.peaks = vc3d::hud::findPeaks(r.values, static_cast<float>(std::max(8.0, 0.5 * sd)));
    r.available = true;
    return r;
}

QRectF CVolumeViewerView::scaleBarRect() const
{
    if (property("vc_hide_scalebar").toBool() || _cachedVpW != viewport()->width() ||
        _cachedVpH != viewport()->height()) {
        return {};
    }
    // Matches drawScaleBar: a bar along the bottom-left with its label just above it.
    constexpr double M = 10.0;
    const QFontMetricsF fm(_cachedFont);
    const double bottom = viewport()->height() - M;
    const double width = std::max(_cachedBarPx, fm.horizontalAdvance(_cachedBarLabel));
    return QRectF(QPointF(M - 2.0, bottom - 5.0 - fm.ascent() - 2.0), QPointF(M + width + 2.0, bottom + 2.0));
}

vc3d::hud::PanelLayout CVolumeViewerView::panelLayout() const
{
    const QSizeF pane(viewport()->width(), viewport()->height());
    // The profile keeps its place while a line is being drawn, so the histogram does not
    // jump when the drag ends.
    const auto profile = _hud.lineProfile && _profileLine ? std::optional<QSizeF>(QSizeF(380, 170)) : std::nullopt;
    const auto histogram = _hud.histogram ? std::optional<QSizeF>(QSizeF(320, 150)) : std::nullopt;
    std::vector<QRectF> reserved;
    if (const QRectF bar = scaleBarRect(); !bar.isEmpty()) {
        reserved.push_back(bar);
    }
    return vc3d::hud::layoutPanels(pane, profile, histogram, reserved);
}

namespace {
QRectF hudPanel(QPainter* p, const QRectF& r)
{
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(0, 0, 0, 170));
    p->drawRoundedRect(r, 6, 6);
    return r.adjusted(10, 8, -10, -8);
}

QString lengthText(double value, bool physical)
{
    if (!physical) {
        return QString::number(value, 'f', value < 10.0 ? 1 : 0) + QStringLiteral(" vx");
    }
    return CVolumeViewerView::formatScaleBarLength(value).text;
}

QString coordText(const std::array<double, 3>& v)
{
    return QStringLiteral("(%1, %2, %3)").arg(v[0], 0, 'f', 0).arg(v[1], 0, 'f', 0).arg(v[2], 0, 'f', 0);
}

// A window limit as set: whole numbers plain, fractional ones to two decimals.
QString windowText(double v)
{
    return std::abs(v - std::round(v)) < 1e-9 ? QString::number(v, 'f', 0) : QString::number(v, 'f', 2);
}

void setHudFont(QPainter* p, qreal pointSize)
{
    QFont f = p->font();
    f.setPointSizeF(pointSize);
    p->setFont(f);
}

// One line of panel text, cut to the panel's width.
void drawPanelText(QPainter* p, const QRectF& in, double baseline, const QString& text)
{
    const QFontMetricsF fm(p->font());
    p->drawText(QPointF(in.left(), baseline), fm.elidedText(text, Qt::ElideRight, in.width()));
}
}  // namespace

void CVolumeViewerView::drawHistogram(QPainter* p) const
{
    if (!_hud.histogram) {
        return;
    }
    const auto layout = panelLayout();
    if (!layout.histogram) {
        return;
    }
    const FrameResult fr = measurementFrame();
    if (!fr.frame) {
        _histogramSerial = 0;
    } else if (_histogramSerial != fr.frame->serial || fr.frame->serial == 0) {
        _histogramSerial = fr.frame->serial;
        _histogram = vc3d::hud::computeHistogram(rasterOf(*fr.frame),
                                                 QRect(0, 0, viewport()->width(), viewport()->height()), 2,
                                                 fr.frame->displayed);
    }

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    const QRectF in = hudPanel(p, *layout.histogram);
    setHudFont(p, 9);
    p->setPen(Qt::white);
    const QString what = fr.frame && !fr.frame->composite.isEmpty()
        ? tr("Visible %1, 8-bit before window (log)").arg(fr.frame->composite)
        : tr("Visible samples, 8-bit before window (log)");
    drawPanelText(p, in, in.top() + 11, what);
    if (!fr.frame || _histogram.count == 0) {
        drawPanelText(p, in, in.top() + 40,
                      !fr.frame ? tr("Histogram unavailable: %1").arg(fr.unavailable)
                                : tr("No measured pixels in view"));
        p->restore();
        return;
    }
    const auto& h = _histogram;
    const QRectF plot(in.left(), in.top() + 18, in.width(), std::max(1.0, in.height() - 50));

    double peak = 1;
    for (std::uint32_t c : h.bins) {
        peak = std::max(peak, std::log1p(double(c)));
    }
    QPainterPath area;
    area.moveTo(plot.bottomLeft());
    for (int v = 0; v < 256; ++v) {
        const double x = plot.left() + plot.width() * v / 255.0;
        const double y = plot.bottom() - plot.height() * std::log1p(double(h.bins[std::size_t(v)])) / peak;
        area.lineTo(x, y);
    }
    area.lineTo(plot.bottomRight());
    area.closeSubpath();
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(120, 190, 255, 200));
    p->drawPath(area);

    auto marker = [&](double v, const QColor& c, Qt::PenStyle style) {
        const double x = plot.left() + plot.width() * std::clamp(v, 0.0, 255.0) / 255.0;
        p->setPen(QPen(c, 1.5, style));
        p->drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
    };
    marker(fr.frame->windowLow, QColor(120, 255, 220), Qt::SolidLine);
    marker(fr.frame->windowHigh, QColor(120, 255, 220), Qt::SolidLine);
    marker(h.low, QColor(255, 210, 90), Qt::DashLine);
    marker(h.high, QColor(255, 210, 90), Qt::DashLine);
    marker(h.mean, QColor(255, 255, 255), Qt::DashLine);

    p->setPen(Qt::white);
    drawPanelText(p, in, in.bottom() - 14,
                  tr("1-99%: %1-%2   mean %3%4%5")
                      .arg(h.low).arg(h.high).arg(h.mean, 0, 'f', 0)
                      .arg(fr.frame->overlayBlended ? tr("   overlay not counted") : QString())
                      .arg(fr.frame->complete ? QString() : tr("   loading")));
    drawPanelText(p, in, in.bottom(),
                  tr("window %1-%2 shows %3% as black, %4% as white")
                      .arg(windowText(fr.frame->windowLow), windowText(fr.frame->windowHigh))
                      .arg(100.0 * h.belowWindow, 0, 'f', 1).arg(100.0 * h.aboveWindow, 0, 'f', 1));
    p->restore();
}

void CVolumeViewerView::drawLineProfile(QPainter* p) const
{
    if (!_hud.lineProfile || !_profileLine) {
        return;
    }
    const ProfileReadout r = profileReadout();
    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);

    // The line itself, with dots where peaks fall.
    const QLineF line = *_profileLine;
    p->setPen(QPen(QColor(0, 0, 0, 160), 4));
    p->drawLine(line);
    p->setPen(QPen(QColor(255, 210, 90), 2));
    p->drawLine(line);
    const int n = static_cast<int>(r.values.size());
    p->setBrush(QColor(255, 90, 90));
    p->setPen(Qt::NoPen);
    for (int idx : r.peaks) {
        p->drawEllipse(vc3d::hud::samplePoint(line.p1(), line.p2(), idx, n), 3.5, 3.5);
    }

    const auto layout = panelLayout();
    if (_profileDragging || !layout.profile) {
        p->restore();
        return;
    }
    const QRectF in = hudPanel(p, *layout.profile);
    setHudFont(p, 9);
    p->setPen(Qt::white);
    if (!r.available || n < 2) {
        drawPanelText(p, in, in.top() + 11, tr("Line profile unavailable: %1").arg(r.unavailable));
        p->restore();
        return;
    }
    drawPanelText(p, in, in.top() + 11,
                  tr("Line profile, 8-bit samples before window: %1 peaks").arg(r.peaks.size()));
    const QRectF plot(in.left(), in.top() + 18, in.width(), std::max(1.0, in.height() - 52));
    QPainterPath curve;
    bool started = false;
    for (int i = 0; i < n; ++i) {
        const float v = r.values[std::size_t(i)];
        if (std::isnan(v)) {
            started = false;
            continue;
        }
        const QPointF pt(plot.left() + plot.width() * i / (n - 1), plot.bottom() - plot.height() * v / 255.0);
        if (!started) {
            curve.moveTo(pt);
            started = true;
        } else {
            curve.lineTo(pt);
        }
    }
    p->setBrush(Qt::NoBrush);
    p->setPen(QPen(QColor(255, 210, 90), 1.6));
    p->drawPath(curve);
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(255, 90, 90));
    for (int idx : r.peaks) {
        const float v = r.values[std::size_t(idx)];
        p->drawEllipse(QPointF(plot.left() + plot.width() * idx / (n - 1),
                               plot.bottom() - plot.height() * v / 255.0), 3, 3);
    }

    p->setPen(Qt::white);
    const QString along = r.alongSurface ? tr("along the surface") : tr("along the plane");
    QString foot = r.length ? tr("length %1 %2").arg(lengthText(*r.length, r.physical), along)
                            : tr("length unknown: the line leaves the surface");
    if (r.peaks.size() >= 2) {
        const double first = r.arc[std::size_t(r.peaks.front())];
        const double last = r.arc[std::size_t(r.peaks.back())];
        if (std::isfinite(first) && std::isfinite(last)) {
            foot += tr("   peak spacing %1")
                        .arg(lengthText((last - first) / double(r.peaks.size() - 1), r.physical));
        }
    }
    if (r.missing > 0) {
        foot += tr("   %1 of %2 samples missing%3")
                    .arg(r.missing).arg(n).arg(r.complete ? QString() : tr(", loading"));
    }
    drawPanelText(p, in, in.bottom() - 14, foot);
    if (r.start && r.end) {
        drawPanelText(p, in, in.bottom(), tr("%1 to %2").arg(coordText(*r.start), coordText(*r.end)));
    }
    p->restore();
}

void CVolumeViewerView::drawGrid(QPainter* p) const
{
    if (!_hud.grid) {
        return;
    }
    const FrameResult fr = measurementFrame(false);
    if (!fr.frame || !fr.frame->planeMap) {
        return;
    }
    const auto& map = *fr.frame->planeMap;
    const QRectF rect(0, 0, viewport()->width(), viewport()->height());
    double step = _hud.gridStep;
    if (!(step > 0)) {
        // About one line per 120 pixels at the current zoom.
        double perPx = 0;
        for (int k = 0; k < 3; ++k) {
            perPx = std::max(perPx, std::hypot(map.dx[std::size_t(k)], map.dy[std::size_t(k)]));
        }
        step = vc3d::fiber_map::ruler::niceStepAtLeast(perPx * 120.0);
    }
    const auto lines = vc3d::hud::gridLines(map, rect, step);
    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    setHudFont(p, 8);
    const QColor colors[3] = {QColor(255, 120, 120, 210), QColor(120, 255, 150, 210), QColor(130, 170, 255, 210)};
    for (const auto& gl : lines) {
        p->setPen(QPen(colors[gl.axis], 1, Qt::DashLine));
        p->drawLine(gl.line);
        // Label at the end nearest the top or left edge, kept inside small panes.
        const QPointF at = gl.line.p1().y() < gl.line.p2().y() ? gl.line.p1() : gl.line.p2();
        const QString label = vc3d::hud::gridLabel(gl.axis, gl.value, step);
        const QPointF textAt(vc3d::hud::clampInto(at.x() + 3, 2.0, rect.width() - 60),
                             vc3d::hud::clampInto(at.y() + 12, 12.0, rect.height() - 4));
        p->setPen(QColor(0, 0, 0, 200));
        p->drawText(textAt + QPointF(1, 1), label);
        p->setPen(colors[gl.axis].lighter(130));
        p->drawText(textAt, label);
    }
    p->restore();
}

void CVolumeViewerView::drawLoupe(QPainter* p) const
{
    if (!_hud.loupe || !_directFb || _directFb->isNull() || _lastCursorVp.x() < 0) {
        return;
    }
    const FrameResult fr = measurementFrame();
    if (!fr.frame) {
        return;  // the image on screen is not the frame the readout would describe
    }
    const QSizeF pane(viewport()->width(), viewport()->height());
    const auto target = vc3d::hud::loupeRect(pane, _lastCursorVp, 168.0);
    if (!target) {
        return;
    }
    constexpr int kSrc = 24;  // frame pixels shown across the loupe
    const int fx = static_cast<int>(std::floor(_lastCursorVp.x()));
    const int fy = static_cast<int>(std::floor(_lastCursorVp.y()));
    const QRect src(fx - kSrc / 2, fy - kSrc / 2, kSrc, kSrc);
    const QImage crop = _directFb->copy(src).scaled(int(target->width()), int(target->height()),
                                                    Qt::IgnoreAspectRatio, Qt::FastTransformation);
    const QPointF center = target->center();

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    QPainterPath clip;
    clip.addEllipse(*target);
    p->setClipPath(clip);
    p->drawImage(*target, crop);
    p->setClipping(false);
    p->setPen(QPen(QColor(255, 210, 90), 2));
    p->setBrush(Qt::NoBrush);
    p->drawEllipse(*target);
    p->setPen(QPen(QColor(255, 255, 255, 180), 1));
    p->drawLine(QPointF(center.x() - 8, center.y()), QPointF(center.x() + 8, center.y()));
    p->drawLine(QPointF(center.x(), center.y() - 8), QPointF(center.x(), center.y() + 8));

    QString text;
    if (!fr.frame->composite.isEmpty()) {
        // A composite value comes from one of several slices: no single sample or position.
        text = fr.frame->composite;
    } else {
        const auto raster = rasterOf(*fr.frame);
        text = raster.covered(fx, fy) ? tr("sample %1").arg(raster.at(fx, fy)) : tr("not measured");
        const std::array<double, 3> v = framePosition(*fr.frame, QPointF(fx, fy));
        if (vc3d::hud::validPosition(v)) {
            text += QStringLiteral("  ") + coordText(v);
        }
    }
    setHudFont(p, 9);
    const double labelW = std::min(target->width() + 60.0, pane.width() - 4.0);
    const QRectF label(vc3d::hud::clampInto(target->left() - 30.0, 2.0, pane.width() - labelW - 2.0),
                       target->bottom() + 2, labelW, 18);
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(0, 0, 0, 170));
    p->drawRoundedRect(label, 4, 4);
    p->setPen(Qt::white);
    p->drawText(label, Qt::AlignCenter, QFontMetricsF(p->font()).elidedText(text, Qt::ElideRight, label.width() - 4));
    p->restore();
}
