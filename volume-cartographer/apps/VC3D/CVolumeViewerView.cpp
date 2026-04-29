#include "CVolumeViewerView.hpp"

#include <QGraphicsView>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QScrollBar>
#include <cmath>
#include <algorithm>



double CVolumeViewerView::chooseNiceLength(double nominal) const
{
    double expn = std::floor(std::log10(nominal));
    double base = std::pow(10.0, expn);
    double d    = nominal / base;
    if (d < 2.0)      return 1.0   * base;
    else if (d < 5.0) return 2.0   * base;
    else              return 5.0   * base;
}

void CVolumeViewerView::drawForeground(QPainter* p, const QRectF& sceneRect)
{
    // 1) Let QGraphicsView draw any foreground items
    QGraphicsView::drawForeground(p, sceneRect);

    // 2) Scalebar overlay, in **viewport** coords so it never moves
    const double dpr = devicePixelRatioF();
    const double m11 = transform().m11();
    const int vpW = viewport()->width();
    const int vpH = viewport()->height();

    // Recompute cached scalebar only when inputs change
    if (_cachedM11 != m11 || _cachedDpr != dpr || _cachedVpW != vpW ||
        _cachedVpH != vpH || _cachedVx != m_vx || _scalebarCacheDirty) {
        _cachedM11 = m11;
        _cachedDpr = dpr;
        _cachedVpW = vpW;
        _cachedVpH = vpH;
        _cachedVx = m_vx;
        _scalebarCacheDirty = false;

        _cachedFont = p->font();
        _cachedFont.setPointSizeF(12 * dpr);

        double pxPerScene = m11 * dpr;
        double pxPerUm = pxPerScene / m_vx;
        double wPx = vpW * dpr;
        double wUm = wPx / pxPerUm;
        double barUm = chooseNiceLength(wUm / 4.0);
        _cachedBarPx = barUm * pxPerUm;

        double displayLength = barUm;
        QString unit = QStringLiteral(" µm");
        if (barUm >= 1000.0) {
            displayLength = barUm / 1000.0;
            unit = QStringLiteral(" mm");
        }
        _cachedBarLabel = QString::number(displayLength) + unit;
    }

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    p->setPen(QPen(Qt::red, 2));
    p->setFont(_cachedFont);

    constexpr int M = 10;
    int bottom = static_cast<int>(vpH * dpr) - M;
    p->drawLine(M, bottom, static_cast<int>(M + _cachedBarPx), bottom);
    p->drawText(M, bottom - 5, _cachedBarLabel);
    p->restore();

    drawTiltHandle(p);
}

CVolumeViewerView::CVolumeViewerView(QWidget* parent) : QGraphicsView(parent)
{
    setMouseTracking(true);
    if (viewport()) {
        viewport()->setMouseTracking(true);
    }
};

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

void CVolumeViewerView::drawBackground(QPainter* painter, const QRectF& /*rect*/)
{
    if (_directFb && !_directFb->isNull()) {
        painter->resetTransform();
        painter->drawImage(0, 0, *_directFb);
    }
}

void CVolumeViewerView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx,dy);
    sendScrolled();  // Emit after scroll so renderVisible sees the new viewport position
}

void CVolumeViewerView::wheelEvent(QWheelEvent *event)
{
    // Accumulate fractional degrees so high-resolution trackpads and mice
    // with fine scroll steps don't lose precision to integer truncation.
    // Standard mice send 120 units (= 15 degrees) per notch.
    _wheelAccum += event->angleDelta().y();
    constexpr int kStepThreshold = 120;  // one notch = one step
    int steps = _wheelAccum / kStepThreshold;
    if (steps == 0) {
        event->accept();
        return;
    }
    _wheelAccum -= steps * kStepThreshold;

    // Pass viewport-relative mouse position directly (not scene coords).
    // Scene coords depend on Qt scroll position which shifts when the
    // tile grid is windowed, causing zoom-at-point to jump.
    QPointF vp_loc = viewport()->mapFromGlobal(event->globalPosition());
    sendZoom(steps, vp_loc, event->modifiers());
    event->accept();
}

void CVolumeViewerView::mouseReleaseEvent(QMouseEvent *event)
{
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
        
        // Emit both signals - the clicked signal for compatibility and the release signal
        // to allow for drawing
        sendVolumeClicked(scene_loc, event->button(), event->modifiers());
        sendMouseRelease(scene_loc, event->button(), event->modifiers());
        
        _left_button_pressed = false;
        event->accept();
        return;
    }
    event->ignore();
}

void CVolumeViewerView::keyPressEvent(QKeyEvent *event)
{
    // When scroll-pan is disabled (tiled renderer), block arrow keys from
    // reaching QGraphicsView's built-in scroll handler.  They'll be handled
    // via sendKeyRelease → onKeyRelease in CTiledVolumeViewer instead.
    if (_scrollPanDisabled) {
        switch (event->key()) {
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
    }

    QGraphicsView::keyPressEvent(event);
    event->ignore();  // let unhandled keys propagate to CWindow
}

void CVolumeViewerView::keyReleaseEvent(QKeyEvent *event)
{
    emit sendKeyRelease(event->key(), event->modifiers());
    QGraphicsView::keyReleaseEvent(event);
}

void CVolumeViewerView::mousePressEvent(QMouseEvent *event)
{
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
        _left_button_pressed = true;
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
    // Base class first so viewport()->size() reflects the new dimensions.
    // Otherwise the tiled viewer's onResized reads the stale viewport size,
    // leaves the framebuffer/sceneRect at the old dims, and subsequent mouse
    // events map through an offset proportional to the resize delta.
    QGraphicsView::resizeEvent(event);
    emit sendResized();
}

void CVolumeViewerView::mouseMoveEvent(QMouseEvent *event)
{
    if (_tiltHandleDragging) {
        emit sendTiltHandleChanged(tiltFromHandlePos(event->position()));
        event->accept();
        return;
    }

    if (_regular_pan)
    {
        if (!_scrollPanDisabled) {
            QPoint scroll = _last_pan_position - QPoint(event->position().x(), event->position().y());

            int x = horizontalScrollBar()->value() + scroll.x();
            horizontalScrollBar()->setValue(x);
            int y = verticalScrollBar()->value() + scroll.y();
            verticalScrollBar()->setValue(y);
        }

        _last_pan_position = QPoint(event->position().x(), event->position().y());
        event->accept();

        // Tiled viewers disable scrollbars and perform panning in their
        // mouse-move handler, so they still need pan-motion updates here.
        if (!_scrollPanDisabled) {
            return;
        }
    }

    QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
    QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});

    // Forward mouse move events even without a pressed button so tools that
    // rely on hover state (e.g. segmentation editing) receive continuous
    // volume coordinates. Consumers that only care about drags can still
    // ignore events where no buttons are pressed.
    sendMouseMove(scene_loc, event->buttons(), event->modifiers());
}
