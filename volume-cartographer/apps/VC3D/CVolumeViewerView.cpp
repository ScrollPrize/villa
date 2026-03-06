#include "CVolumeViewerView.hpp"

#include <QGraphicsView>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QScrollBar>
#include <cmath>



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
}

CVolumeViewerView::CVolumeViewerView(QWidget* parent) : QGraphicsView(parent)
{ 
    setMouseTracking(true);
};

void CVolumeViewerView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx,dy);
    sendScrolled();  // Emit after scroll so renderVisible sees the new viewport position
}

void CVolumeViewerView::wheelEvent(QWheelEvent *event)
{
    // Get raw delta value and use smaller divisor for higher sensitivity
    int num_degrees = event->angleDelta().y() / 8;
    
    QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
    QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});

    // Send the zoom event with a more sensitive delta value
    // Changed from /15 to /5 to make it more responsive to small wheel movements
    sendZoom(num_degrees/5, scene_loc, event->modifiers());
    
    event->accept();
}

void CVolumeViewerView::mouseReleaseEvent(QMouseEvent *event)
{
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
            event->accept();
            return;
        default:
            break;
        }
    }

    QGraphicsView::keyPressEvent(event);
}

void CVolumeViewerView::keyReleaseEvent(QKeyEvent *event)
{
    emit sendKeyRelease(event->key(), event->modifiers());
    QGraphicsView::keyReleaseEvent(event);
}

void CVolumeViewerView::mousePressEvent(QMouseEvent *event)
{
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

void CVolumeViewerView::resizeEvent(QResizeEvent *event)
{
    emit sendResized();
    QGraphicsView::resizeEvent(event);
}

void CVolumeViewerView::mouseMoveEvent(QMouseEvent *event)
{
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
        // sendCursorMove handler, so they still need pan-motion updates here.
        if (!_scrollPanDisabled) {
            return;
        }
    }

    QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
    QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});

    sendCursorMove(scene_loc);

    if (_regular_pan) {
        return;
    }

    // Forward mouse move events even without a pressed button so tools that
    // rely on hover state (e.g. segmentation editing) receive continuous
    // volume coordinates. Consumers that only care about drags can still
    // ignore events where no buttons are pressed.
    sendMouseMove(scene_loc, event->buttons(), event->modifiers());
}
