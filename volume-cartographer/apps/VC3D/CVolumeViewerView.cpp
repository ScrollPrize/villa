#include "CVolumeViewerView.hpp"
#include "CVolumeViewer.hpp"

#include <QGraphicsView>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPainter>
#include <cmath>
#include <QPainter>



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
    p->save();
    // reset any scene→view transform so we draw in raw pixels
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);

    // red, 2px pen
    QPen pen(Qt::red, 2);
    p->setPen(pen);

    // font (scaled for HiDPI)
    QFont f = p->font();
    f.setPointSizeF(12 * devicePixelRatioF());
    p->setFont(f);

    constexpr int M = 10;  // margin in px
    // transform: scene units → view pixels
    QTransform t = transform();
    const double dpr = devicePixelRatioF();

    // 1) how many device-px per scene‐unit
    double pxPerScene = transform().m11() * dpr;

    // 2) how many device-px in the viewport
    double wPx = viewport()->width() * dpr;

    // 3) device-px per µm
    double pxPerUm = pxPerScene / m_vx;

    // now compute the physical width in µm …
    double wUm   = wPx / pxPerUm;
    double ideal = wUm / 4.0;
    double barUm = chooseNiceLength(ideal);
    double barPx = barUm * pxPerUm;

    // decide on unit and display value
    double displayLength = barUm;
    QString unit = QStringLiteral(" µm");
    if (barUm >= 1000.0) {
        displayLength = barUm / 1000.0;      // convert to mm
        unit = QStringLiteral(" mm");
    }

    // draw the line (in pixels)
    p->drawLine(int(M), int(viewport()->height()*dpr) - M, 
                int(M + barPx), int(viewport()->height()*dpr) - M);

    // draw the label
    QString label = QString::number(displayLength) + unit;
    p->drawText(int(M), int(viewport()->height()*dpr) - M - 5, label);
    p->restore();
}

CVolumeViewerView::CVolumeViewerView(QWidget* parent) : QGraphicsView(parent)
{ 
    setMouseTracking(true);
};

void CVolumeViewerView::scrollContentsBy(int dx, int dy)
{
    sendScrolled();
    QGraphicsView::scrollContentsBy(dx,dy);
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
    // Key handling moved to global QShortcut objects in CWindow
    // Pass the event to the base class
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
        QPoint scroll = _last_pan_position - QPoint(event->position().x(), event->position().y());
        
        int x = horizontalScrollBar()->value() + scroll.x();
        horizontalScrollBar()->setValue(x);
        int y = verticalScrollBar()->value() + scroll.y();
        verticalScrollBar()->setValue(y);
        
        _last_pan_position = QPoint(event->position().x(), event->position().y());
        event->accept();
        return;
    }
    else {
        QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
        QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
        
        sendCursorMove(scene_loc);

        // Forward mouse move events even without a pressed button so tools that
        // rely on hover state (e.g. segmentation editing) receive continuous
        // volume coordinates. Consumers that only care about drags can still
        // ignore events where no buttons are pressed.
        sendMouseMove(scene_loc, event->buttons(), event->modifiers());
    }
    event->ignore();
}
