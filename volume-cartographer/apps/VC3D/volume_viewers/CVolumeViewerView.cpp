#include "CVolumeViewerView.hpp"

#include <QGraphicsView>
#include <QGraphicsProxyWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPainterPath>
#include <QPolygonF>
#include <QScrollBar>
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
        _cachedPhysicalUnits != m_physicalUnits || _scalebarCacheDirty) {
        _cachedM11 = m11;
        _cachedVpW = vpW;
        _cachedVpH = vpH;
        _cachedVx = m_vx;
        _cachedPhysicalUnits = m_physicalUnits;
        _scalebarCacheDirty = false;

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
    emit paintCompleted();
}

void CVolumeViewerView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx,dy);
    sendScrolled();  // Emit after scroll so renderVisible sees the new viewport position
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
    if (_profileDragging && event->button() == Qt::LeftButton) {
        _profileDragging = false;
        if (_profileLine) {
            _profileLine->setP2(event->position());
            if (_profileLine->length() < 3.0) {
                _profileLine.reset();  // a click clears the line
            }
        }
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
        
        _left_button_pressed = false;
        event->accept();
        return;
    }
    event->ignore();
}

void CVolumeViewerView::keyPressEvent(QKeyEvent *event)
{
    if (_scrollPanDisabled) {
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
    }

    QGraphicsView::keyPressEvent(event);
    event->ignore();  // let unhandled keys propagate to CWindow
}

void CVolumeViewerView::keyReleaseEvent(QKeyEvent *event)
{
    emit sendKeyRelease(event->key(), event->modifiers());
    QGraphicsView::keyReleaseEvent(event);
    event->ignore();  // let unhandled releases propagate to CWindow
}

void CVolumeViewerView::mousePressEvent(QMouseEvent *event)
{
    if (_hud.lineProfile && event->button() == Qt::LeftButton &&
        event->modifiers() == Qt::NoModifier && !pointInTiltHandle(event->position())) {
        // In line-profile mode a plain left drag draws the profile line.
        _profileDragging = true;
        _profileLine = QLineF(event->position(), event->position());
        viewport()->update();
        event->accept();
        return;
    }

    if (pointInSceneWidget(event->position())) {
        _sceneWidgetMouseCapture = true;
        QGraphicsView::mousePressEvent(event);
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
        if (!_scrollPanDisabled) {
            QPoint scroll = _last_pan_position - QPoint(event->position().x(), event->position().y());

            int x = horizontalScrollBar()->value() + scroll.x();
            horizontalScrollBar()->setValue(x);
            int y = verticalScrollBar()->value() + scroll.y();
            verticalScrollBar()->setValue(y);
        }

        _last_pan_position = QPoint(event->position().x(), event->position().y());
        event->accept();

        if (!_scrollPanDisabled) {
            return;
        }
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
        _profileDragging = false;
    }
    viewport()->update();
}

void CVolumeViewerView::setProfileLine(std::optional<QLineF> line)
{
    _profileLine = line;
    _profileDragging = false;
    viewport()->update();
}

void CVolumeViewerView::setVolumeMapping(VolumeMapping mapping)
{
    _mapping = std::move(mapping);
}

bool CVolumeViewerView::showsVolume() const
{
    if (!_mapping.sceneToVolume) {
        return true;
    }
    const cv::Vec3f c = _mapping.sceneToVolume(mapToScene(viewport()->rect().center()));
    return std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2]);
}

std::function<bool(QPointF)> CVolumeViewerView::insideVolume() const
{
    const auto map = viewportToVolumeMap();
    const auto extent = _mapping.extent ? _mapping.extent() : std::nullopt;
    if (!map || !extent) {
        return {};
    }
    return [m = *map, e = *extent](QPointF vp) {
        const auto v = m.at(vp);
        for (int k = 0; k < 3; ++k) {
            if (!(v[std::size_t(k)] >= -0.5 && v[std::size_t(k)] < e[k] - 0.5)) {
                return false;
            }
        }
        return true;
    };
}

QPointF CVolumeViewerView::viewportToFramebuffer(const QPointF& vp) const
{
    const qreal s = _directFbScale > 0 ? _directFbScale : 1.0;
    return (vp - _directFbOffset) / s;
}

std::optional<vc3d::hud::AffineMap> CVolumeViewerView::viewportToVolumeMap() const
{
    if (!_mapping.sceneToVolume || !_mapping.planar || !_mapping.planar()) {
        return std::nullopt;
    }
    const int w = viewport()->width();
    const int h = viewport()->height();
    if (w < 2 || h < 2) {
        return std::nullopt;
    }
    const cv::Vec3f o = _mapping.sceneToVolume(mapToScene(QPoint(0, 0)));
    const cv::Vec3f ex = _mapping.sceneToVolume(mapToScene(QPoint(w - 1, 0)));
    const cv::Vec3f ey = _mapping.sceneToVolume(mapToScene(QPoint(0, h - 1)));
    vc3d::hud::AffineMap m;
    for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(o[k]) || !std::isfinite(ex[k]) || !std::isfinite(ey[k])) {
            return std::nullopt;
        }
        m.origin[k] = o[k];
        m.dx[k] = (ex[k] - o[k]) / (w - 1);
        m.dy[k] = (ey[k] - o[k]) / (h - 1);
    }
    return m;
}

CVolumeViewerView::ProfileReadout CVolumeViewerView::profileReadout() const
{
    ProfileReadout r;
    if (!_profileLine || !_directFb || _directFb->isNull()) {
        return r;
    }
    r.values = vc3d::hud::sampleLine(*_directFb, viewportToFramebuffer(_profileLine->p1()),
                                     viewportToFramebuffer(_profileLine->p2()));
    if (const auto inside = insideVolume()) {
        const int n = static_cast<int>(r.values.size());
        for (int i = 0; i < n; ++i) {
            if (!inside(_profileLine->pointAt(n > 1 ? double(i) / (n - 1) : 0.0))) {
                r.values[std::size_t(i)] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    double sum = 0, sum2 = 0;
    int n = 0;
    for (float v : r.values) {
        if (!std::isnan(v)) {
            sum += v;
            sum2 += double(v) * v;
            ++n;
        }
    }
    const double sd = n > 1 ? std::sqrt(std::max(0.0, sum2 / n - (sum / n) * (sum / n))) : 0.0;
    r.peaks = vc3d::hud::findPeaks(r.values, static_cast<float>(std::max(8.0, 0.5 * sd)));
    const double sceneLength = QLineF(mapToScene(_profileLine->p1().toPoint()),
                                      mapToScene(_profileLine->p2().toPoint())).length();
    r.lengthUnits = sceneLength * m_vx;
    return r;
}

namespace {
QRectF hudPanel(QPainter* p, const QRectF& r)
{
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(0, 0, 0, 170));
    p->drawRoundedRect(r, 6, 6);
    return r.adjusted(10, 8, -10, -8);
}

QString unitText(double units, bool physical)
{
    if (!physical) {
        return QString::number(units, 'f', 0) + QStringLiteral(" vx");
    }
    return CVolumeViewerView::formatScaleBarLength(units).text;
}
}  // namespace

void CVolumeViewerView::drawHistogram(QPainter* p) const
{
    if (!_hud.histogram || !_directFb || _directFb->isNull() || !showsVolume()) {
        return;
    }
    if (_histogramKey != _directFb->cacheKey()) {
        _histogramKey = _directFb->cacheKey();
        // Leave out the background around the volume, which would count as clipped black.
        std::function<bool(int, int)> include;
        if (const auto inside = insideVolume()) {
            const qreal s = _directFbScale > 0 ? _directFbScale : 1.0;
            const QPointF offset = _directFbOffset;
            include = [inside, s, offset](int x, int y) { return inside(QPointF(x, y) * s + offset); };
        }
        _histogram = vc3d::hud::computeHistogram(*_directFb, 2, include);
    }
    const auto& h = _histogram;
    if (h.count == 0) {
        return;
    }
    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    const QRectF outer(viewport()->width() - 290, viewport()->height() - 150, 280, 140);
    const QRectF in = hudPanel(p, outer);
    const QRectF plot(in.left(), in.top() + 18, in.width(), in.height() - 36);

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

    auto marker = [&](int v, const QColor& c) {
        const double x = plot.left() + plot.width() * v / 255.0;
        p->setPen(QPen(c, 1.5, Qt::DashLine));
        p->drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
    };
    marker(h.low, QColor(255, 210, 90));
    marker(h.high, QColor(255, 210, 90));
    marker(static_cast<int>(std::lround(h.mean)), QColor(255, 255, 255));

    QFont f = p->font();
    f.setPointSizeF(9);
    p->setFont(f);
    p->setPen(Qt::white);
    p->drawText(QPointF(in.left(), in.top() + 11), QStringLiteral("On-screen histogram (log)"));
    p->drawText(QPointF(in.left(), in.bottom()),
                QStringLiteral("1-99%: %1-%2   mean %3   clipped %4% / %5%")
                    .arg(h.low).arg(h.high).arg(h.mean, 0, 'f', 0)
                    .arg(100.0 * h.black, 0, 'f', 1).arg(100.0 * h.white, 0, 'f', 1));
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

    // The line itself, with ticks where peaks fall.
    const QLineF line = *_profileLine;
    p->setPen(QPen(QColor(0, 0, 0, 160), 4));
    p->drawLine(line);
    p->setPen(QPen(QColor(255, 210, 90), 2));
    p->drawLine(line);
    const int n = static_cast<int>(r.values.size());
    for (int idx : r.peaks) {
        const QPointF at = line.pointAt(n > 1 ? double(idx) / (n - 1) : 0.0);
        p->setBrush(QColor(255, 90, 90));
        p->setPen(Qt::NoPen);
        p->drawEllipse(at, 3.5, 3.5);
    }
    if (n < 2) {
        p->restore();
        return;
    }

    const QRectF outer(10, viewport()->height() - 200, 360, 160);
    const QRectF in = hudPanel(p, outer);
    const QRectF plot(in.left(), in.top() + 18, in.width(), in.height() - 52);
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

    QFont f = p->font();
    f.setPointSizeF(9);
    p->setFont(f);
    p->setPen(Qt::white);
    p->drawText(QPointF(in.left(), in.top() + 11), QStringLiteral("Line profile, %1 bright crossings")
                                                      .arg(r.peaks.size()));
    QString foot = QStringLiteral("length %1").arg(unitText(r.lengthUnits, m_physicalUnits));
    if (r.peaks.size() >= 2) {
        const double per = r.lengthUnits * (r.peaks.back() - r.peaks.front()) / (n - 1) / (r.peaks.size() - 1);
        foot += QStringLiteral("   mean spacing %1").arg(unitText(per, m_physicalUnits));
    }
    p->drawText(QPointF(in.left(), in.bottom() - 14), foot);
    if (auto map = viewportToVolumeMap()) {
        const auto a = map->at(line.p1());
        const auto b = map->at(line.p2());
        p->drawText(QPointF(in.left(), in.bottom()),
                    QStringLiteral("(%1, %2, %3) to (%4, %5, %6)")
                        .arg(a[0], 0, 'f', 0).arg(a[1], 0, 'f', 0).arg(a[2], 0, 'f', 0)
                        .arg(b[0], 0, 'f', 0).arg(b[1], 0, 'f', 0).arg(b[2], 0, 'f', 0));
    }
    p->restore();
}

void CVolumeViewerView::drawGrid(QPainter* p) const
{
    if (!_hud.grid) {
        return;
    }
    const auto map = viewportToVolumeMap();
    if (!map) {
        return;
    }
    const QRectF rect(0, 0, viewport()->width(), viewport()->height());
    double step = _hud.gridStep;
    if (!(step > 0)) {
        // About one line per 120 pixels at the current zoom.
        double perPx = 0;
        for (int k = 0; k < 3; ++k) {
            perPx = std::max(perPx, std::hypot(map->dx[std::size_t(k)], map->dy[std::size_t(k)]));
        }
        step = vc3d::fiber_map::ruler::niceStepAtLeast(perPx * 120.0);
    }
    const auto lines = vc3d::hud::gridLines(*map, rect, step);
    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    QFont f = p->font();
    f.setPointSizeF(8);
    p->setFont(f);
    const QColor colors[3] = {QColor(255, 120, 120, 210), QColor(120, 255, 150, 210), QColor(130, 170, 255, 210)};
    for (const auto& gl : lines) {
        p->setPen(QPen(colors[gl.axis], 1, Qt::DashLine));
        p->drawLine(gl.line);
        // Label at the end nearest the top or left edge.
        const QPointF at = gl.line.p1().y() < gl.line.p2().y() ? gl.line.p1() : gl.line.p2();
        const QString label = vc3d::hud::axisName(gl.axis) + QStringLiteral("=") + QString::number(gl.value, 'f', 0);
        const QPointF textAt(std::clamp(at.x() + 3, 2.0, rect.width() - 60), std::clamp(at.y() + 12, 12.0, rect.height() - 4));
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
    constexpr int kSrc = 24;     // framebuffer pixels shown
    constexpr int kSize = 168;   // loupe diameter on screen
    const QPointF fb = viewportToFramebuffer(_lastCursorVp);
    const QRect src(static_cast<int>(fb.x()) - kSrc / 2, static_cast<int>(fb.y()) - kSrc / 2, kSrc, kSrc);
    const QImage crop = _directFb->copy(src).scaled(kSize, kSize, Qt::IgnoreAspectRatio, Qt::FastTransformation);
    QPointF center = _lastCursorVp + QPointF(kSize * 0.75, -kSize * 0.75);
    center.setX(std::clamp(center.x(), kSize / 2.0 + 4, viewport()->width() - kSize / 2.0 - 4));
    center.setY(std::clamp(center.y(), kSize / 2.0 + 4, viewport()->height() - kSize / 2.0 - 24));
    const QRectF target(center - QPointF(kSize / 2.0, kSize / 2.0), QSizeF(kSize, kSize));

    p->save();
    p->resetTransform();
    p->setRenderHint(QPainter::Antialiasing);
    QPainterPath clip;
    clip.addEllipse(target);
    p->setClipPath(clip);
    p->drawImage(target, crop);
    p->setClipping(false);
    p->setPen(QPen(QColor(255, 210, 90), 2));
    p->setBrush(Qt::NoBrush);
    p->drawEllipse(target);
    p->setPen(QPen(QColor(255, 255, 255, 180), 1));
    p->drawLine(QPointF(center.x() - 8, center.y()), QPointF(center.x() + 8, center.y()));
    p->drawLine(QPointF(center.x(), center.y() - 8), QPointF(center.x(), center.y() + 8));
    const int fx = static_cast<int>(fb.x());
    const int fy = static_cast<int>(fb.y());
    QString text;
    if (fx >= 0 && fy >= 0 && fx < _directFb->width() && fy < _directFb->height()) {
        text = QStringLiteral("value %1").arg(vc3d::hud::grayAt(*_directFb, fx, fy));
    }
    if (_mapping.sceneToVolume) {
        const cv::Vec3f v = _mapping.sceneToVolume(mapToScene(_lastCursorVp.toPoint()));
        if (std::isfinite(v[0])) {
            text += QStringLiteral("  (%1, %2, %3)").arg(v[0], 0, 'f', 0).arg(v[1], 0, 'f', 0).arg(v[2], 0, 'f', 0);
        }
    }
    QFont f = p->font();
    f.setPointSizeF(9);
    p->setFont(f);
    const QRectF label(target.left() - 30, target.bottom() + 2, target.width() + 60, 18);
    p->setPen(Qt::NoPen);
    p->setBrush(QColor(0, 0, 0, 170));
    p->drawRoundedRect(label, 4, 4);
    p->setPen(Qt::white);
    p->drawText(label, Qt::AlignCenter, text);
    p->restore();
}
